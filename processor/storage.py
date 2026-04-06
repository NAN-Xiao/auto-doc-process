#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量数据持久化工具 — PostgreSQL + pgvector

功能：
1. 从 processed 目录读取已生成的 embeddings 和 metadata
2. 存储到 PostgreSQL（pgvector 扩展）
3. 支持增量更新（按文档删旧、插新）和完全重建

⚠️ 重要说明：
  - 本脚本 **不做任何 embedding 计算**
  - 只读取 embedding_generator 预先生成的向量文件
  - 纯粹的数据读取和数据库写入操作

工作流程：
  document_splitter  → 拆分文档 (chunks + images)
  ↓
  embedding_generator → 生成向量 (embeddings + metadata) ← embedding 在这里生成
  ↓
  vector_storage      → 持久化存储 (PostgreSQL/pgvector) ← 只读取已生成的 embedding
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from ..core.config import load_processor_config as load_config
from ..core.logger import Logger


class PgVectorStorage:
    """PostgreSQL + pgvector 持久化管理器

    事务安全说明：
      - 所有写操作在单个事务中完成（DELETE 旧版 + INSERT 新版）
      - 写事务开头使用 pg_advisory_xact_lock 获取排他写锁，
        保证同一时刻只有一个进程在修改 doc_chunks 表
      - PostgreSQL MVCC 机制天然保证：读不阻塞写、写不阻塞读
        → 另一个 RAG 系统在读取时不会被阻塞，也不会读到半写数据
      - advisory lock 随事务自动释放，无需手动清理
    """

    # doc_chunks 表的写锁 ID（pg_advisory_xact_lock 使用的 bigint 常量）
    # 选一个不易与其他业务冲突的数字即可
    ADVISORY_LOCK_DOC_CHUNKS = 728301

    DDL_INIT = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS doc_chunks (
        id                SERIAL PRIMARY KEY,
        doc_name          TEXT    NOT NULL DEFAULT '',
        doc_format        TEXT    NOT NULL DEFAULT '',
        doc_timestamp     TEXT    NOT NULL DEFAULT '',
        chunk_id          TEXT    NOT NULL DEFAULT '',
        chunk_index       INTEGER NOT NULL DEFAULT 0,
        chunk_text        TEXT    NOT NULL,
        char_count        INTEGER NOT NULL DEFAULT 0,
        chunk_token_count INTEGER NOT NULL DEFAULT 0,
        chunk_hash        TEXT    NOT NULL DEFAULT '',
        page_number       INTEGER,
        has_images        BOOLEAN NOT NULL DEFAULT FALSE,
        image_count       INTEGER NOT NULL DEFAULT 0,
        images_json       JSONB   NOT NULL DEFAULT '[]'::jsonb,
        image_titles_json JSONB   NOT NULL DEFAULT '[]'::jsonb,
        content_quality_score DOUBLE PRECISION DEFAULT 0.0,
        is_structured_chunk BOOLEAN NOT NULL DEFAULT FALSE,
        quality_flags_json JSONB NOT NULL DEFAULT '[]'::jsonb,
        content_type      TEXT    NOT NULL DEFAULT 'section',
        title             TEXT    NOT NULL DEFAULT '',
        section_path      TEXT    NOT NULL DEFAULT '',
        parent_chunk_id   TEXT    NOT NULL DEFAULT '',
        table_index       INTEGER NOT NULL DEFAULT 0,
        table_headers_json JSONB  NOT NULL DEFAULT '[]'::jsonb,
        row_data_json     JSONB   NOT NULL DEFAULT '{{}}'::jsonb,
        keywords_json     JSONB   NOT NULL DEFAULT '[]'::jsonb,
        source_file       TEXT    NOT NULL DEFAULT '',
        processed_batch_id TEXT DEFAULT '',
        doc_version_hash  TEXT DEFAULT '',
        source_doc_id     TEXT DEFAULT '',
        source_updated_at TEXT DEFAULT '',
        embedding         vector({dim}),
        processed_at      TEXT    DEFAULT '',
        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_name
        ON doc_chunks (doc_name);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_chunk_id
        ON doc_chunks (chunk_id);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_source_type
        ON doc_chunks (source_doc_id, content_type);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_order
        ON doc_chunks (doc_name, chunk_index);

    CREATE UNIQUE INDEX IF NOT EXISTS uq_doc_chunks_source_chunk
        ON doc_chunks (source_doc_id, chunk_id)
        WHERE source_doc_id <> '';

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding
        ON doc_chunks USING hnsw (embedding vector_cosine_ops);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_keywords_json
        ON doc_chunks USING gin (keywords_json);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_row_data_json
        ON doc_chunks USING gin (row_data_json);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_images_json
        ON doc_chunks USING gin (images_json);
    """

    # 增量 DDL：为已有表添加新列（幂等）
    DDL_ADD_SOURCE_COLUMNS = """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'space_id'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN space_id TEXT NOT NULL DEFAULT '';
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_space_id ON doc_chunks (space_id);
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'source_url'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN source_url TEXT NOT NULL DEFAULT '';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'chunk_token_count'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN chunk_token_count INTEGER NOT NULL DEFAULT 0;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'image_titles_json'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN image_titles_json JSONB NOT NULL DEFAULT '[]'::jsonb;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'chunk_hash'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN chunk_hash TEXT NOT NULL DEFAULT '';
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_chunk_hash ON doc_chunks (chunk_hash);
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'content_quality_score'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN content_quality_score DOUBLE PRECISION DEFAULT 0.0;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'is_structured_chunk'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN is_structured_chunk BOOLEAN NOT NULL DEFAULT FALSE;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'quality_flags_json'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN quality_flags_json JSONB NOT NULL DEFAULT '[]'::jsonb;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'processed_batch_id'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN processed_batch_id TEXT DEFAULT '';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'doc_version_hash'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN doc_version_hash TEXT DEFAULT '';
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_version_hash ON doc_chunks (doc_version_hash);
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'source_doc_id'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN source_doc_id TEXT DEFAULT '';
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_source_doc_id ON doc_chunks (source_doc_id);
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'source_updated_at'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN source_updated_at TEXT DEFAULT '';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'content_type'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN content_type TEXT NOT NULL DEFAULT 'section';
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_content_type ON doc_chunks (content_type);
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'title'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN title TEXT NOT NULL DEFAULT '';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'section_path'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN section_path TEXT NOT NULL DEFAULT '';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'parent_chunk_id'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN parent_chunk_id TEXT NOT NULL DEFAULT '';
            CREATE INDEX IF NOT EXISTS idx_doc_chunks_parent_chunk_id ON doc_chunks (parent_chunk_id);
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'table_index'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN table_index INTEGER NOT NULL DEFAULT 0;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'table_headers_json'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN table_headers_json JSONB NOT NULL DEFAULT '[]'::jsonb;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'row_data_json'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN row_data_json JSONB NOT NULL DEFAULT '{}'::jsonb;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'keywords_json'
        ) THEN
            ALTER TABLE doc_chunks ADD COLUMN keywords_json JSONB NOT NULL DEFAULT '[]'::jsonb;
        END IF;

        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'images_json' AND data_type <> 'jsonb'
        ) THEN
            ALTER TABLE doc_chunks
                ALTER COLUMN images_json TYPE JSONB
                USING CASE
                    WHEN images_json IS NULL OR images_json = '' THEN '[]'::jsonb
                    ELSE images_json::jsonb
                END;
            ALTER TABLE doc_chunks ALTER COLUMN images_json SET DEFAULT '[]'::jsonb;
            ALTER TABLE doc_chunks ALTER COLUMN images_json SET NOT NULL;
        END IF;

        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'image_titles_json' AND data_type <> 'jsonb'
        ) THEN
            ALTER TABLE doc_chunks
                ALTER COLUMN image_titles_json TYPE JSONB
                USING CASE
                    WHEN image_titles_json IS NULL OR image_titles_json = '' THEN '[]'::jsonb
                    ELSE image_titles_json::jsonb
                END;
            ALTER TABLE doc_chunks ALTER COLUMN image_titles_json SET DEFAULT '[]'::jsonb;
            ALTER TABLE doc_chunks ALTER COLUMN image_titles_json SET NOT NULL;
        END IF;

        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'quality_flags_json' AND data_type <> 'jsonb'
        ) THEN
            ALTER TABLE doc_chunks
                ALTER COLUMN quality_flags_json TYPE JSONB
                USING CASE
                    WHEN quality_flags_json IS NULL OR quality_flags_json = '' THEN '[]'::jsonb
                    ELSE quality_flags_json::jsonb
                END;
            ALTER TABLE doc_chunks ALTER COLUMN quality_flags_json SET DEFAULT '[]'::jsonb;
            ALTER TABLE doc_chunks ALTER COLUMN quality_flags_json SET NOT NULL;
        END IF;

        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'table_headers_json' AND data_type <> 'jsonb'
        ) THEN
            ALTER TABLE doc_chunks
                ALTER COLUMN table_headers_json TYPE JSONB
                USING CASE
                    WHEN table_headers_json IS NULL OR table_headers_json = '' THEN '[]'::jsonb
                    ELSE table_headers_json::jsonb
                END;
            ALTER TABLE doc_chunks ALTER COLUMN table_headers_json SET DEFAULT '[]'::jsonb;
            ALTER TABLE doc_chunks ALTER COLUMN table_headers_json SET NOT NULL;
        END IF;

        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'row_data_json' AND data_type <> 'jsonb'
        ) THEN
            ALTER TABLE doc_chunks
                ALTER COLUMN row_data_json TYPE JSONB
                USING CASE
                    WHEN row_data_json IS NULL OR row_data_json = '' THEN '{}'::jsonb
                    ELSE row_data_json::jsonb
                END;
            ALTER TABLE doc_chunks ALTER COLUMN row_data_json SET DEFAULT '{}'::jsonb;
            ALTER TABLE doc_chunks ALTER COLUMN row_data_json SET NOT NULL;
        END IF;

        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'doc_chunks' AND column_name = 'keywords_json' AND data_type <> 'jsonb'
        ) THEN
            ALTER TABLE doc_chunks
                ALTER COLUMN keywords_json TYPE JSONB
                USING CASE
                    WHEN keywords_json IS NULL OR keywords_json = '' THEN '[]'::jsonb
                    ELSE keywords_json::jsonb
                END;
            ALTER TABLE doc_chunks ALTER COLUMN keywords_json SET DEFAULT '[]'::jsonb;
            ALTER TABLE doc_chunks ALTER COLUMN keywords_json SET NOT NULL;
        END IF;
    END
    $$;
    """

    # 全文检索索引（BM25/关键词匹配）
    DDL_FULLTEXT_INDEX = """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE indexname = 'idx_doc_chunks_fulltext'
        ) THEN
            -- chunk_text 列添加 GIN 全文检索索引（简体中文 + 默认分词）
            CREATE INDEX idx_doc_chunks_fulltext
                ON doc_chunks USING gin (to_tsvector('simple', chunk_text));
        END IF;
    END
    $$;
    """

    DDL_EXTRA_INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_doc_chunks_source_type
        ON doc_chunks (source_doc_id, content_type);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_order
        ON doc_chunks (doc_name, chunk_index);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_space_type
        ON doc_chunks (space_id, content_type);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_processed_batch
        ON doc_chunks (processed_batch_id);

    CREATE UNIQUE INDEX IF NOT EXISTS uq_doc_chunks_source_chunk
        ON doc_chunks (source_doc_id, chunk_id)
        WHERE source_doc_id <> '';

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_keywords_json
        ON doc_chunks USING gin (keywords_json);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_row_data_json
        ON doc_chunks USING gin (row_data_json);

    CREATE INDEX IF NOT EXISTS idx_doc_chunks_images_json
        ON doc_chunks USING gin (images_json);
    """

    def __init__(self, db_config: dict = None, vector_dim: int = 512):
        """
        初始化

        Args:
            db_config: 数据库配置 {"host", "port", "database", "user", "password"}
                       为 None 时从配置文件读取
            vector_dim: 向量维度（bge-small-zh-v1.5 = 512）
        """
        if db_config is None:
            config = load_config()
            db_config = config.get("database", {})

        self.db_config = db_config
        self.vector_dim = vector_dim

        Logger.info("初始化 PostgreSQL 向量存储")
        Logger.info(
            f"{db_config.get('host', 'localhost')}:"
            f"{db_config.get('port', 5432)}/"
            f"{db_config.get('database', '')}", indent=1
        )

    # -------------------- 连接 --------------------

    def _get_conn(self, autocommit: bool = False):
        conn = psycopg.connect(
            host=self.db_config.get("host", "localhost"),
            port=self.db_config.get("port", 5432),
            dbname=self.db_config.get("database", ""),
            user=self.db_config.get("user", ""),
            password=self.db_config.get("password", ""),
            connect_timeout=int(self.db_config.get("connect_timeout", 10)),
            application_name=self.db_config.get("application_name", "auto-doc-process"),
            autocommit=autocommit,
        )
        register_vector(conn)
        return conn

    # -------------------- 建表 --------------------

    def init_table(self):
        """创建表和索引（幂等），并执行增量迁移"""
        conn = self._get_conn(autocommit=True)
        try:
            with conn.cursor() as cur:
                cur.execute(self.DDL_INIT.format(dim=self.vector_dim))
                # 增量迁移：为已有表添加 space_id / source_url 列
                cur.execute(self.DDL_ADD_SOURCE_COLUMNS)
                # 全文检索索引（支持 BM25 关键词匹配）
                cur.execute(self.DDL_FULLTEXT_INDEX)
                # 额外组合索引和 JSONB GIN 索引
                cur.execute(self.DDL_EXTRA_INDEXES)
            Logger.info("数据库表 doc_chunks 就绪")
        finally:
            conn.close()

    def reset_table(self):
        """删除重建表"""
        Logger.warning("重建表 doc_chunks")
        conn = self._get_conn(autocommit=True)
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS doc_chunks CASCADE")
            self.init_table()
        finally:
            conn.close()

    # -------------------- 加载单个 chunk --------------------

    def _load_chunk_data(self, chunk_index: int,
                         dir_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """读取单个 chunk 的文本 + embedding + metadata"""
        try:
            # embedding
            emb_file = dir_info["embeddings_dir"] / f"chunk_{chunk_index:04d}.json"
            if not emb_file.exists():
                return None
            with open(emb_file, "r", encoding="utf-8") as f:
                emb_data = json.load(f)

            # metadata
            meta_file = dir_info["metadata_dir"] / f"chunk_{chunk_index:04d}.json"
            if not meta_file.exists():
                return None
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

            # chunk text
            txt_file = dir_info["chunks_dir"] / f"chunk_{chunk_index:04d}.txt"
            chunk_text = ""
            if txt_file.exists():
                with open(txt_file, "r", encoding="utf-8") as f:
                    chunk_text = f.read()
            if not chunk_text.strip():
                return None

            raw_meta = meta_data.get("metadata", {})
            images_list = raw_meta.get("images", [])
            image_titles = raw_meta.get("image_titles", [])
            if not image_titles and images_list:
                image_titles = [img.get("title", "") for img in images_list if img.get("title")]

            return {
                "id": meta_data.get("id", ""),
                "doc_name": dir_info["doc_name"],
                "doc_format": raw_meta.get("doc_format", ""),
                "doc_timestamp": dir_info["timestamp"],
                "chunk_id": raw_meta.get("chunk_id", ""),
                "chunk_index": raw_meta.get("chunk_index", chunk_index),
                "chunk_text": chunk_text,
                "char_count": raw_meta.get("char_count", len(chunk_text)),
                "chunk_token_count": raw_meta.get("chunk_token_count", 0),
                "chunk_hash": raw_meta.get("chunk_hash", ""),
                "page_number": raw_meta.get("page_number"),
                "has_images": raw_meta.get("has_images", False),
                "image_count": raw_meta.get("image_count", 0),
                "images_json": images_list or [],
                "image_titles_json": image_titles or [],
                "content_quality_score": raw_meta.get("content_quality_score", 0.0),
                "is_structured_chunk": raw_meta.get("is_structured_chunk", False),
                "quality_flags_json": raw_meta.get("quality_flags", []),
                "content_type": raw_meta.get("content_type", "section"),
                "title": raw_meta.get("title", ""),
                "section_path": raw_meta.get("section_path", ""),
                "parent_chunk_id": raw_meta.get("parent_chunk_id", ""),
                "table_index": raw_meta.get("table_index", 0),
                "table_headers_json": raw_meta.get("table_headers", []),
                "row_data_json": raw_meta.get("row_data", {}),
                "keywords_json": raw_meta.get("keywords", []),
                "source_file": raw_meta.get("source", ""),
                "processed_batch_id": raw_meta.get("processed_batch_id", dir_info["timestamp"]),
                "doc_version_hash": raw_meta.get("doc_version_hash", dir_info.get("doc_version_hash", "")),
                "source_doc_id": raw_meta.get("source_doc_id", dir_info.get("source_doc_id", "")),
                "source_updated_at": raw_meta.get("source_updated_at", dir_info.get("source_updated_at", "")),
                "space_id": dir_info.get("space_id", ""),
                "source_url": dir_info.get("source_url", ""),
                "embedding": emb_data["embedding"],
                "processed_at": raw_meta.get("processed_at", ""),
            }
        except Exception as e:
            Logger.warning(f"加载失败 chunk_{chunk_index:04d}: {e}", indent=1)
            return None

    # -------------------- 查询 --------------------

    def get_stored_doc_names(self) -> set:
        """查询数据库中已有哪些 doc_name（去重），用于判断哪些文档需要处理"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT doc_name FROM doc_chunks")
                return {row[0] for row in cur.fetchall()}
        except Exception:
            return set()
        finally:
            conn.close()

    def _get_existing_doc_snapshot(self, cur, source_doc_id: str = "", doc_name: str = "") -> Optional[Dict[str, Any]]:
        """读取库中已有文档的版本快照，用于跳过未变化重写。"""
        if source_doc_id:
            cur.execute(
                """
                SELECT source_doc_id, doc_name, doc_version_hash, COUNT(*)::int AS chunk_count
                FROM doc_chunks
                WHERE source_doc_id = %s
                GROUP BY source_doc_id, doc_name, doc_version_hash
                ORDER BY chunk_count DESC
                LIMIT 1
                """,
                (source_doc_id,),
            )
        else:
            cur.execute(
                """
                SELECT source_doc_id, doc_name, doc_version_hash, COUNT(*)::int AS chunk_count
                FROM doc_chunks
                WHERE doc_name = %s
                GROUP BY source_doc_id, doc_name, doc_version_hash
                ORDER BY chunk_count DESC
                LIMIT 1
                """,
                (doc_name,),
            )

        row = cur.fetchone()
        if not row:
            return None
        return {
            "source_doc_id": row[0] or "",
            "doc_name": row[1] or "",
            "doc_version_hash": row[2] or "",
            "chunk_count": int(row[3] or 0),
        }

    # -------------------- 写入 --------------------

    def _delete_doc(self, cur, doc_name: str, timestamp: str = None, source_doc_id: str = ""):
        """
        删除指定文档的旧数据

        行为：优先按 source_doc_id 删除该文档的 **所有** 旧版本 chunks，
        确保重命名后仍能清理同一来源文档；没有 source_doc_id 时回退到 doc_name。
        """
        if source_doc_id:
            cur.execute(
                "DELETE FROM doc_chunks WHERE source_doc_id = %s",
                (source_doc_id,),
            )
            return
        cur.execute("DELETE FROM doc_chunks WHERE doc_name = %s", (doc_name,))

    def store_document(self, dir_info: Dict[str, Any]) -> int:
        """
        将一个文档的所有 chunks 写入 PostgreSQL（单文档事务）

        Returns:
            写入的 chunk 数量
        """
        Logger.info(f"处理文档: {dir_info['doc_name']} ({dir_info['timestamp']})")

        # 收集 embedding 文件
        emb_files = sorted(dir_info["embeddings_dir"].glob("chunk_*.json"))
        Logger.info(f"找到 {len(emb_files)} 个 embedding 文件", indent=1)

        # 加载所有 chunks
        chunks = []
        for ef in emb_files:
            idx = int(ef.stem.split("_")[1])
            data = self._load_chunk_data(idx, dir_info)
            if data:
                chunks.append(data)

        if not chunks:
            Logger.warning("没有有效的 chunks", indent=1)
            return 0

        Logger.info(f"加载了 {len(chunks)} 个有效 chunks", indent=1)

        now = datetime.now()
        conn = self._get_conn()  # autocommit=False → 事务模式
        try:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (self.ADVISORY_LOCK_DOC_CHUNKS,),
                    )
                    existing = self._get_existing_doc_snapshot(
                        cur,
                        dir_info.get("source_doc_id", ""),
                        dir_info["doc_name"],
                    )
                    if existing and existing.get("doc_version_hash") == dir_info.get("doc_version_hash", ""):
                        Logger.info(
                            f"检测到文档版本未变化，跳过重写: {dir_info['doc_name']} "
                            f"(source_doc_id={existing.get('source_doc_id') or dir_info.get('source_doc_id', '')})",
                            indent=1,
                        )
                        return existing.get("chunk_count", len(chunks))
                    self._delete_doc(
                        cur,
                        dir_info["doc_name"],
                        dir_info["timestamp"],
                        dir_info.get("source_doc_id", ""),
                    )
                    self._insert_chunks(cur, chunks, now)

            Logger.success(f"成功存储 {len(chunks)} chunks", indent=1)

            with_images = sum(1 for c in chunks if c["has_images"])
            if with_images:
                Logger.info(f"包含图片的 chunks: {with_images}", indent=1)

            return len(chunks)
        finally:
            conn.close()

    def batch_store_documents(self, dir_infos: List[Dict[str, Any]]) -> int:
        """
        批量统一入库：所有文档在同一个事务中写入 PostgreSQL

        流程：
          1. 获取 advisory lock（排他写锁）
          2. 逐个文档: DELETE 旧版 → INSERT 新版
          3. 事务提交（全部成功才写入，任何失败全部回滚）

        这保证了：
          - RAG 系统要么看到旧的完整数据，要么看到新的完整数据
          - 不会出现部分文档更新、部分还是旧版的中间状态

        Args:
            dir_infos: store_document 格式的 dir_info 列表

        Returns:
            成功入库的文档数
        """
        if not dir_infos:
            return 0

        Logger.info(f"批量入库: {len(dir_infos)} 个文档")

        # 预加载所有文档的 chunks（在事务外完成，减少锁持有时间）
        all_doc_chunks: List[tuple] = []  # [(dir_info, chunks), ...]
        for di in dir_infos:
            emb_files = sorted(di["embeddings_dir"].glob("chunk_*.json"))
            chunks = []
            for ef in emb_files:
                idx = int(ef.stem.split("_")[1])
                data = self._load_chunk_data(idx, di)
                if data:
                    chunks.append(data)
            if chunks:
                all_doc_chunks.append((di, chunks))
                Logger.info(f"  {di['doc_name']}: {len(chunks)} chunks", indent=1)
            else:
                Logger.warning(f"  {di['doc_name']}: 无有效 chunks，跳过", indent=1)

        if not all_doc_chunks:
            Logger.warning("没有任何有效 chunks 需要入库")
            return 0

        total_chunks = sum(len(chunks) for _, chunks in all_doc_chunks)
        Logger.info(f"共计 {total_chunks} 个 chunks，开始写入...")

        now = datetime.now()
        conn = self._get_conn()
        try:
            stored_docs = 0
            with conn.transaction():
                with conn.cursor() as cur:
                    # 获取 advisory lock（事务级排他锁）
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (self.ADVISORY_LOCK_DOC_CHUNKS,),
                    )

                    for di, chunks in all_doc_chunks:
                        existing = self._get_existing_doc_snapshot(
                            cur,
                            di.get("source_doc_id", ""),
                            di["doc_name"],
                        )
                        if existing and existing.get("doc_version_hash") == di.get("doc_version_hash", ""):
                            Logger.info(
                                f"  ↷ {di['doc_name']}: 版本未变化，跳过重写",
                                indent=1,
                            )
                            continue
                        self._delete_doc(
                            cur,
                            di["doc_name"],
                            di["timestamp"],
                            di.get("source_doc_id", ""),
                        )
                        self._insert_chunks(cur, chunks, now)
                        Logger.info(f"  ✓ {di['doc_name']}: {len(chunks)} chunks", indent=1)
                        stored_docs += 1

            Logger.success(f"批量入库成功: {stored_docs} 个文档, {total_chunks} 个 chunks")
            return stored_docs

        except Exception as e:
            Logger.error(f"批量入库失败（已全部回滚）: {e}")
            raise
        finally:
            conn.close()

    # -------------------- 内部：批量插入 --------------------

    def _insert_chunks(self, cur, chunks: List[Dict[str, Any]], now: datetime):
        """将一组 chunks 批量插入数据库（在已有事务和游标中执行）"""
        if not chunks:
            return
        sql = """
            INSERT INTO doc_chunks
                (doc_name, doc_format, doc_timestamp, chunk_id, chunk_index,
                 chunk_text, char_count, chunk_token_count, chunk_hash, page_number, has_images, image_count,
                 images_json, image_titles_json, content_quality_score, is_structured_chunk, quality_flags_json,
                 content_type, title, section_path, parent_chunk_id, table_index, table_headers_json, row_data_json, keywords_json,
                 source_file, processed_batch_id, doc_version_hash, source_doc_id, source_updated_at, space_id, source_url,
                 embedding, processed_at,
                 created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s, %s,%s)
        """
        params = [
            (
                c["doc_name"], c["doc_format"], c["doc_timestamp"],
                c["chunk_id"], c["chunk_index"],
                c["chunk_text"], c["char_count"], c["chunk_token_count"], c["chunk_hash"], c["page_number"],
                c["has_images"], c["image_count"],
                Jsonb(c["images_json"]), Jsonb(c["image_titles_json"]), c["content_quality_score"],
                c["is_structured_chunk"], Jsonb(c["quality_flags_json"]),
                c["content_type"], c["title"], c["section_path"], c["parent_chunk_id"],
                c["table_index"], Jsonb(c["table_headers_json"]), Jsonb(c["row_data_json"]), Jsonb(c["keywords_json"]),
                c["source_file"],
                c["processed_batch_id"], c["doc_version_hash"], c["source_doc_id"], c["source_updated_at"],
                c["space_id"], c["source_url"],
                c["embedding"], c["processed_at"],
                now, now,
            )
            for c in chunks
        ]
        cur.executemany(sql, params)
