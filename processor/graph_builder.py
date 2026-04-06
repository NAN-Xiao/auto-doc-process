#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightRAG 知识图谱构建器

功能：
1. 读取预处理的文档 chunks
2. 使用 LightRAG 构建知识图谱（实体抽取 + 关系建模）
3. 将图谱数据（实体、关系、chunks）导出到 PostgreSQL

工作流程：
  processed/ 目录（已拆分的 chunks）
    ↓
  LightRAG 处理（LLM 实体关系抽取 → 图谱构建 → 写入本地 JSON/graphml）
    ↓
  另一个 RAG 工程监控 lightrag_workspace/ 文件变化，写入完成后读取最新内容
    ↓
  PostgreSQL 导出（lightrag_entities + lightrag_relations + lightrag_chunks 二次备份）

依赖：
  - lightrag-hku >= 1.4.0
  - DeepSeek API（用于实体关系抽取）
  - BAAI/bge-small-zh-v1.5（本地 embedding 模型）
"""

import os
import json
import asyncio
import hashlib
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop

from ..core.config import load_processor_config, load_db_config, CONFIGS_DIR, resolve_app_path
from ..core.logger import Logger
from ..core.utils import atomic_write_json, workspace_begin_write, workspace_end_write
from .quality import summarize_batch_quality

_tenacity_logger = logging.getLogger("feishu_sync")


# ==================== 配置加载 ====================

def _load_lightrag_yaml() -> dict:
    """加载 lightrag.yaml 配置"""
    import yaml
    path = CONFIGS_DIR / "lightrag.yaml"
    if not path.exists():
        Logger.error(f"LightRAG 配置文件不存在: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("lightrag", {})


def load_lightrag_config() -> dict:
    """
    加载 LightRAG 完整配置（合并 lightrag.yaml + doc_splitter.yaml + db_info.yml）

    Returns:
        合并后的配置字典
    """
    lightrag_cfg = _load_lightrag_yaml()
    proc_cfg = load_processor_config()
    db_cfg = load_db_config()

    # 从 doc_splitter.yaml 继承 LLM 公共配置（lightrag.yaml 中的值优先）
    llm_proc = proc_cfg.get("llm", {})
    lightrag_llm = lightrag_cfg.setdefault("llm", {})
    for k in ("api_key", "api_base", "model"):
        if not lightrag_llm.get(k):
            lightrag_llm[k] = llm_proc.get(k, "")

    # 从 doc_splitter.yaml 继承 embedding 公共配置
    emb_proc = proc_cfg.get("embedding", {})
    hf_cfg = emb_proc.get("huggingface", {})
    lightrag_emb = lightrag_cfg.setdefault("embedding", {})
    if not lightrag_emb.get("model"):
        lightrag_emb["model"] = emb_proc.get("model", "BAAI/bge-small-zh-v1.5")
    lightrag_emb["cache_folder"] = hf_cfg.get("cache_folder", "./models")
    lightrag_emb["device"] = hf_cfg.get("device", "cpu")

    # 数据库配置
    lightrag_cfg["database"] = db_cfg

    # 路径配置
    paths = proc_cfg.get("paths", {})
    lightrag_cfg["documents_dir"] = paths.get("documents_dir", "../documents")
    lightrag_cfg["processed_dir"] = paths.get("processed_dir", "../processed")

    return lightrag_cfg


# ==================== Embedding 包装 ====================

def _create_local_embedding_func(config: dict) -> EmbeddingFunc:
    """
    创建基于本地模型的 embedding 函数（自动选择 ONNX 或 torch 后端）

    Args:
        config: lightrag 配置

    Returns:
        EmbeddingFunc 实例
    """
    from .onnx_embedder import create_embeddings
    from ..core.config import load_processor_config

    emb_cfg = config.get("embedding", {})
    dim = emb_cfg.get("dim", 512)
    max_token_size = emb_cfg.get("max_token_size", 512)

    # 使用统一的 embedding 工厂（自动选择 ONNX 或 torch）
    proc_config = load_processor_config()
    embeddings_engine = create_embeddings(proc_config)

    async def _embed(texts: list[str]) -> np.ndarray:
        """异步 embedding 包装（实际同步执行）"""
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None, embeddings_engine.embed_documents, texts
        )
        return np.array(vectors, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=max_token_size,
        func=_embed,
    )


# ==================== LLM 包装 ====================

def _create_llm_func(config: dict):
    """
    创建 LLM 调用函数（OpenAI 兼容格式）

    Args:
        config: lightrag 配置

    Returns:
        异步 LLM 调用函数
    """
    llm_cfg = config.get("llm", {})
    api_base = llm_cfg.get("api_base", "https://aikey.elex-tech.com/v1")
    api_key = llm_cfg.get("api_key", "")
    model_name = llm_cfg.get("model", "qwen3.5-plus")

    if not api_key:
        Logger.error("未配置 LLM API Key！请检查 doc_splitter.yaml 或 lightrag.yaml")
        raise ValueError("Missing LLM API Key")

    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("OPENAI_API_BASE", api_base)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
        reraise=True,
    )
    async def _llm_complete(
        prompt: str,
        system_prompt: str = None,
        history_messages: list = None,
        **kwargs,
    ) -> str:
        """
        LLM 调用（OpenAI 兼容格式，带自动重试）

        注意：LightRAG 内部调用签名为 func(prompt, system_prompt=..., **kwargs)，
        model 已在闭包中绑定。
        """
        return await openai_complete_if_cache(
            model=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=api_base,
            api_key=api_key,
            **kwargs,
        )

    return _llm_complete


# ==================== PostgreSQL 图谱导出（二次备份） ====================

class PgGraphExporter:
    """
    将 LightRAG 图谱数据导出到 PostgreSQL（二次备份）

    LightRAG 的核心数据存储在本地 JSON/graphml 文件中，
    另一个 RAG 工程通过监控文件变化来读取。
    此导出器将实体/关系额外保存到 PG 作为备份和分析用途。
    """

    DDL_ENTITIES = """
    CREATE TABLE IF NOT EXISTS {table} (
        id              SERIAL PRIMARY KEY,
        entity_name     TEXT NOT NULL,
        entity_type     TEXT DEFAULT '',
        description     TEXT DEFAULT '',
        source_doc      TEXT DEFAULT '',
        source_chunk_id TEXT DEFAULT '',
        embedding       vector({dim}),
        batch_timestamp TEXT DEFAULT '',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(entity_name, batch_timestamp)
    );
    CREATE INDEX IF NOT EXISTS idx_{table}_name ON {table} (entity_name);
    CREATE INDEX IF NOT EXISTS idx_{table}_type ON {table} (entity_type);
    CREATE INDEX IF NOT EXISTS idx_{table}_embedding
        ON {table} USING hnsw (embedding vector_cosine_ops);
    """

    DDL_RELATIONS = """
    CREATE TABLE IF NOT EXISTS {table} (
        id              SERIAL PRIMARY KEY,
        source_entity   TEXT NOT NULL,
        target_entity   TEXT NOT NULL,
        relation_type   TEXT DEFAULT '',
        description     TEXT DEFAULT '',
        weight          FLOAT DEFAULT 1.0,
        source_doc      TEXT DEFAULT '',
        batch_timestamp TEXT DEFAULT '',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_{table}_source ON {table} (source_entity);
    CREATE INDEX IF NOT EXISTS idx_{table}_target ON {table} (target_entity);
    CREATE INDEX IF NOT EXISTS idx_{table}_relation ON {table} (relation_type);
    """

    DDL_CHUNKS = """
    CREATE TABLE IF NOT EXISTS {table} (
        id              SERIAL PRIMARY KEY,
        chunk_id        TEXT NOT NULL,
        doc_name        TEXT NOT NULL,
        content         TEXT NOT NULL,
        token_count     INTEGER DEFAULT 0,
        chunk_order     INTEGER DEFAULT 0,
        full_doc_id     TEXT DEFAULT '',
        batch_timestamp TEXT DEFAULT '',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_{table}_doc ON {table} (doc_name);
    CREATE INDEX IF NOT EXISTS idx_{table}_chunk ON {table} (chunk_id);
    """

    # lightrag 图谱表的写锁 ID（pg_advisory_xact_lock）
    ADVISORY_LOCK_GRAPH = 728302

    def __init__(self, db_config: dict, export_config: dict):
        self.db_config = db_config
        self.entity_table = export_config.get("entity_table", "lightrag_entities")
        self.relation_table = export_config.get("relation_table", "lightrag_relations")
        self.chunk_table = export_config.get("chunk_table", "lightrag_chunks")
        self.vector_dim = export_config.get("vector_dim", 512)
        self.export_embeddings = export_config.get("export_entity_embeddings", True)

    def _get_conn(self, autocommit: bool = False):
        conn = psycopg.connect(
            host=self.db_config.get("host", "localhost"),
            port=self.db_config.get("port", 5432),
            dbname=self.db_config.get("database", ""),
            user=self.db_config.get("user", ""),
            password=self.db_config.get("password", ""),
            autocommit=autocommit,
        )
        register_vector(conn)
        return conn

    def init_tables(self):
        """创建图谱导出表（幂等）"""
        conn = self._get_conn(autocommit=True)
        try:
            with conn.cursor() as cur:
                # 确保 pgvector 扩展
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                # 创建表
                cur.execute(self.DDL_ENTITIES.format(
                    table=self.entity_table, dim=self.vector_dim
                ))
                cur.execute(self.DDL_RELATIONS.format(table=self.relation_table))
                cur.execute(self.DDL_CHUNKS.format(table=self.chunk_table))
            Logger.info("PostgreSQL 图谱表已就绪")
        finally:
            conn.close()

    def reset_tables(self):
        """删除重建图谱表（实体 + 关系 + chunks）"""
        Logger.warning("重建图谱表: "
                       f"{self.entity_table}, {self.relation_table}, {self.chunk_table}")
        conn = self._get_conn(autocommit=True)
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.entity_table} CASCADE")
                cur.execute(f"DROP TABLE IF EXISTS {self.relation_table} CASCADE")
                cur.execute(f"DROP TABLE IF EXISTS {self.chunk_table} CASCADE")
            self.init_tables()
        finally:
            conn.close()

    def export_all(
        self,
        rag: LightRAG,
        chunks_data: List[Dict[str, Any]],
        batch_timestamp: str,
        embedding_func=None,
        entity_embed_batch: int = 32,
    ) -> Dict[str, int]:
        """
        将图谱数据（实体 + 关系 + chunks）在 **单个事务** 中导出到 PostgreSQL

        这样 RAG 召回端看到的永远是一致快照：
          - COMMIT 之前：看到旧数据
          - COMMIT 之后：同时看到新实体、新关系、新 chunks
          - 任何阶段出错 → 全部回滚，无脏数据

        Args:
            rag: LightRAG 实例
            chunks_data: chunk 信息列表
            batch_timestamp: 批次时间戳
            embedding_func: 可选的 embedding 函数（用于生成实体向量）

        Returns:
            {"entities": N, "relations": M, "chunks": C}
        """
        Logger.info("开始导出图谱数据到 PostgreSQL（单事务原子写入）...")

        # ── 读取图谱数据（内存操作，不涉及 DB 事务） ──
        graph_storage = rag.chunk_entity_relation_graph
        loop = always_get_an_event_loop()
        all_nodes = loop.run_until_complete(graph_storage.get_all_nodes())
        all_edges = loop.run_until_complete(graph_storage.get_all_edges())

        Logger.info(f"图谱节点数: {len(all_nodes)}")
        Logger.info(f"图谱边数: {len(all_edges)}")

        # ── 生成实体 embedding（内存操作） ──
        entity_embeddings: Dict[str, list] = {}
        if self.export_embeddings and embedding_func and all_nodes:
            Logger.info("为实体生成 embedding...")
            embed_texts = []
            embed_keys = []
            for node in all_nodes:
                name = node.get("id", "")
                desc = node.get("description", "")
                text = f"{name}: {desc}" if desc else name
                embed_texts.append(text)
                embed_keys.append(name)

            try:
                for i in range(0, len(embed_texts), entity_embed_batch):
                    batch_texts = embed_texts[i:i + entity_embed_batch]
                    batch_keys = embed_keys[i:i + entity_embed_batch]
                    vectors = loop.run_until_complete(embedding_func.func(batch_texts))
                    for key, vec in zip(batch_keys, vectors):
                        entity_embeddings[key] = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
                Logger.info(f"已生成 {len(entity_embeddings)} 个实体 embedding")
            except Exception as e:
                Logger.warning(f"实体 embedding 生成失败，将跳过: {e}")

        # ── 单事务写入：实体 + 关系 + chunks ──
        conn = self._get_conn()  # autocommit=False → 事务模式
        entity_count = 0
        relation_count = 0
        chunk_count = 0

        try:
            with conn.transaction():
                with conn.cursor() as cur:
                    # 获取 advisory lock（事务级排他锁）
                    # 保证同一时刻只有一个进程在写图谱表
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (self.ADVISORY_LOCK_GRAPH,),
                    )

                    # ---------- 1. 导出实体 ----------
                    for node in all_nodes:
                        entity_name = node.get("id", "")
                        entity_type = node.get("entity_type", "")
                        description = node.get("description", "")
                        source_id = node.get("source_id", "")
                        source_doc = source_id.split("<SEP>")[0] if source_id else ""

                        emb = entity_embeddings.get(entity_name)
                        cur.execute(
                            f"""
                            INSERT INTO {self.entity_table}
                                (entity_name, entity_type, description,
                                 source_doc, source_chunk_id, embedding, batch_timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (entity_name, batch_timestamp) DO UPDATE SET
                                entity_type = EXCLUDED.entity_type,
                                description = EXCLUDED.description,
                                source_doc = EXCLUDED.source_doc,
                                source_chunk_id = EXCLUDED.source_chunk_id,
                                embedding = EXCLUDED.embedding
                            """,
                            (entity_name, entity_type, description,
                             source_doc, source_id, emb, batch_timestamp),
                        )
                        entity_count += 1

                    # ---------- 2. 导出关系 ----------
                    for edge in all_edges:
                        src = edge.get("source", "")
                        tgt = edge.get("target", "")
                        rel_type = edge.get("description", "")
                        weight = edge.get("weight", 1.0)
                        source_id = edge.get("source_id", "")
                        keywords = edge.get("keywords", "")

                        cur.execute(
                            f"""
                            INSERT INTO {self.relation_table}
                                (source_entity, target_entity, relation_type,
                                 description, weight, source_doc, batch_timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (src, tgt, rel_type, keywords,
                             float(weight) if weight else 1.0,
                             source_id, batch_timestamp),
                        )
                        relation_count += 1

                    # ---------- 3. 导出 chunks ----------
                    for chunk in (chunks_data or []):
                        cur.execute(
                            f"""
                            INSERT INTO {self.chunk_table}
                                (chunk_id, doc_name, content, token_count,
                                 chunk_order, full_doc_id, batch_timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                chunk.get("chunk_id", ""),
                                chunk.get("doc_name", ""),
                                chunk.get("content", ""),
                                chunk.get("char_count", 0),
                                chunk.get("index", 0),
                                chunk.get("full_doc_id", ""),
                                batch_timestamp,
                            ),
                        )
                        chunk_count += 1

            # 事务提交成功后才打印
            Logger.success(
                f"图谱导出完成（原子事务）: "
                f"{entity_count} 实体, {relation_count} 关系, {chunk_count} chunks"
            )

        finally:
            conn.close()

        return {
            "entities": entity_count,
            "relations": relation_count,
            "chunks": chunk_count,
        }


# ==================== 主构建器 ====================

class LightRAGGraphBuilder:
    """
    LightRAG 知识图谱构建器

    使用本地存储（JSON + NetworkX），另一个 RAG 工程通过监控
    lightrag_workspace/ 目录下的文件变化来感知数据更新。
    构建完成后额外导出到 PostgreSQL 作为二次备份。
    """

    def __init__(self, config: dict = None):
        """
        初始化

        Args:
            config: LightRAG 配置（None 则自动加载）
        """
        self.config = config or load_lightrag_config()
        self.rag: Optional[LightRAG] = None
        self._embedding_func = None

    def _resolve_working_dir(self) -> str:
        """解析工作目录（与 documents_dir/processed_dir 使用同一相对路径规则）。"""
        raw = self.config.get("working_dir", "../lightrag_workspace")
        p = resolve_app_path(raw)
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def initialize(self) -> LightRAG:
        """初始化 LightRAG 实例"""
        Logger.separator()
        Logger.info("初始化 LightRAG 知识图谱构建器")

        working_dir = self._resolve_working_dir()
        Logger.info(f"工作目录: {working_dir}", indent=1)

        # 创建 embedding 函数
        self._embedding_func = _create_local_embedding_func(self.config)

        # 创建 LLM 函数
        llm_func = _create_llm_func(self.config)

        # 图谱参数
        graph_cfg = self.config.get("graph", {})
        llm_cfg = self.config.get("llm", {})
        storage_cfg = self.config.get("storage", {})

        Logger.info(f"LLM: {llm_cfg.get('provider', 'deepseek')} / {llm_cfg.get('model', 'deepseek-chat')}", indent=1)
        Logger.info(f"Embedding: {self.config.get('embedding', {}).get('model', 'bge-small-zh')}", indent=1)
        Logger.info(f"存储后端: KV={storage_cfg.get('kv')}, Vector={storage_cfg.get('vector')}, Graph={storage_cfg.get('graph')}", indent=1)

        self.rag = LightRAG(
            working_dir=working_dir,
            # LLM
            llm_model_func=llm_func,
            llm_model_name=llm_cfg.get("model", "deepseek-chat"),
            llm_model_max_async=llm_cfg.get("max_async", 4),
            default_llm_timeout=llm_cfg.get("timeout", 180),
            # Embedding
            embedding_func=self._embedding_func,
            embedding_batch_num=self.config.get("embedding", {}).get("batch_num", 10),
            # 图谱参数
            chunk_token_size=graph_cfg.get("chunk_token_size", 1200),
            chunk_overlap_token_size=graph_cfg.get("chunk_overlap_token_size", 100),
            entity_extract_max_gleaning=graph_cfg.get("entity_extract_max_gleaning", 1),
            max_graph_nodes=graph_cfg.get("max_graph_nodes", 1000),
            max_parallel_insert=graph_cfg.get("max_parallel_insert", 2),
            enable_llm_cache=graph_cfg.get("enable_llm_cache", True),
            # 本地存储后端（另一个 RAG 工程通过监控文件变化来读取）
            kv_storage=storage_cfg.get("kv", "JsonKVStorage"),
            vector_storage=storage_cfg.get("vector", "NanoVectorDBStorage"),
            graph_storage=storage_cfg.get("graph", "NetworkXStorage"),
            doc_status_storage=storage_cfg.get("doc_status", "JsonDocStatusStorage"),
        )

        # 初始化所有存储后端（LightRAG 1.4+ 要求显式调用）
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.rag.initialize_storages())

        Logger.success("LightRAG 初始化完成")
        Logger.separator()
        return self.rag

    # ── 图谱增量清单 ──

    GRAPH_MANIFEST_NAME = "_graph_manifest.json"
    WORKSPACE_MANIFEST_NAME = "_workspace_manifest.json"

    @staticmethod
    def _content_hash(chunks: List[Dict[str, Any]]) -> str:
        """计算文档所有 chunks 文本的 SHA-256 哈希，用于增量判断"""
        h = hashlib.sha256()
        for c in chunks:
            h.update(c.get("content", "").encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _load_graph_manifest(processed_dir: Path) -> dict:
        """
        加载图谱增量清单

        格式：{ "doc_name": {"hash": "...", "built_at": "..."}, ... }
        """
        manifest_path = processed_dir / LightRAGGraphBuilder.GRAPH_MANIFEST_NAME
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _save_graph_manifest(processed_dir: Path, manifest: dict):
        """保存图谱增量清单（原子写入 + 自动重试）"""
        manifest_path = processed_dir / LightRAGGraphBuilder.GRAPH_MANIFEST_NAME
        atomic_write_json(manifest_path, manifest)

    @staticmethod
    def _save_workspace_manifest(working_dir: Path, manifest: dict):
        """保存工作区清单，供下游 RAG 直接读取。"""
        atomic_write_json(working_dir / LightRAGGraphBuilder.WORKSPACE_MANIFEST_NAME, manifest)

    @staticmethod
    def clear_graph_data(processed_dir: Path, working_dir: str = None):
        """
        清除图谱增量清单和 LightRAG 工作目录缓存

        用于 --reset-db 时强制全量重建。
        """
        # 清除增量清单
        manifest_path = processed_dir / LightRAGGraphBuilder.GRAPH_MANIFEST_NAME
        if manifest_path.exists():
            manifest_path.unlink()
            Logger.info(f"已删除图谱清单: {manifest_path}")

        # 清除 LightRAG 工作目录（缓存的图谱数据）
        if working_dir:
            wd = Path(working_dir)
            if wd.exists():
                shutil.rmtree(wd, ignore_errors=True)
                wd.mkdir(parents=True, exist_ok=True)
                Logger.info(f"已清空 LightRAG 工作目录: {wd}")

    def _insert_with_retry(
        self,
        loop,
        doc_text: str,
        doc_name: str,
        doc_dir: str,
        *,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """
        带自动重试的 LightRAG 文档插入。

        LightRAG 在 ainsert 时会写入本地 JSON/graphml 文件，
        如果遇到 IO 异常（磁盘、权限、并发冲突）则指数退避重试。
        """
        import time as _time

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                track_id = loop.run_until_complete(
                    self.rag.ainsert(
                        doc_text,
                        ids=[doc_name],
                        file_paths=[doc_dir],
                    )
                )
                return track_id
            except (IOError, OSError) as e:
                last_err = e
                if attempt < max_retries:
                    delay = retry_delay * (2 ** (attempt - 1))
                    Logger.warning(
                        f"    ainsert IO 异常 ({doc_name}), "
                        f"重试 {attempt}/{max_retries}, {delay:.1f}s 后: {e}"
                    )
                    _time.sleep(delay)
                else:
                    raise
            except Exception:
                raise

        raise last_err  # unreachable, for type checker

    def _delete_doc_if_exists(self, loop, doc_name: str) -> bool:
        """
        如果 LightRAG workspace 中已存在同名文档，则先删除旧贡献。

        这是图谱侧“同文档覆盖式增量更新”的关键：
        LightRAG 的 ainsert(ids=[doc_name]) 不会覆盖已存在 doc_id，
        而是会把它当成 duplicate 跳过，因此内容变化时必须先删后插。
        """
        existing_doc = loop.run_until_complete(self.rag.doc_status.get_by_id(doc_name))
        if not existing_doc:
            return False

        delete_result = loop.run_until_complete(self.rag.adelete_by_doc_id(doc_name))
        delete_status = getattr(delete_result, "status", "")
        if delete_status not in {"success", "not_found"}:
            message = getattr(delete_result, "message", "未知错误")
            raise RuntimeError(f"删除旧图谱失败: {doc_name} ({delete_status}) {message}")
        return delete_status == "success"

    def build_from_processed_dir(
        self,
        processed_dir: Path,
        batch_timestamp: str = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        从 processed 目录读取预处理文档并 **增量** 构建图谱

        增量逻辑：
          1. 加载 _graph_manifest.json（记录每个文档的内容哈希）
          2. 对比当前 chunks 内容哈希与清单
          3. 跳过未变化的文档，只处理新增/修改的文档
          4. 成功后更新清单

        Args:
            processed_dir: processed 根目录
            batch_timestamp: 批次时间戳（仅用于 DB 标记）
            force_rebuild: True = 全量重建（忽略增量清单），由 --reset-db 触发

        Returns:
            构建报告
        """
        if self.rag is None:
            self.initialize()

        if batch_timestamp is None:
            batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        Logger.separator()
        Logger.info(f"从预处理目录构建图谱: {processed_dir}")
        Logger.info(f"批次时间戳: {batch_timestamp}")
        if force_rebuild:
            Logger.info("⚠ 全量重建模式（忽略增量清单）")

        # ── 加载增量清单 ──
        manifest = {} if force_rebuild else self._load_graph_manifest(processed_dir)

        # 扫描子目录（每个子目录是一个文档）
        doc_dirs = [
            d for d in processed_dir.iterdir()
            if d.is_dir() and (d / "chunks_index.json").exists()
        ]

        if not doc_dirs:
            Logger.error("未找到任何已处理的文档目录")
            return {"success": False, "error": "无文档目录"}

        Logger.info(f"找到 {len(doc_dirs)} 个文档")

        # ── 通知 RAG 读端：即将写入 workspace，请暂停读取 ──
        working_dir = self._resolve_working_dir()
        workspace_begin_write(Path(working_dir))
        Logger.info("已放置 .writing 标记，RAG 读端应暂停读取")

        try:
            report = self._do_build(
                processed_dir, manifest, doc_dirs, batch_timestamp, force_rebuild,
            )
        except Exception:
            # 即使构建异常，也必须清理 .writing 标记，否则 RAG 读端永远阻塞
            workspace_end_write(Path(working_dir), summary={"error": True})
            Logger.warning("构建异常，已清理 .writing 标记")
            raise

        # ── 通知 RAG 读端：workspace 写入完成，可安全读取 ──
        workspace_end_write(Path(working_dir), summary={
            "built": report["built_documents"],
            "skipped": report["skipped_documents"],
            "batch_timestamp": batch_timestamp,
            "doc_versions": report.get("doc_versions", {}),
        })
        Logger.info("已写入 .ready 信号，RAG 读端可恢复读取")

        return report

    def _do_build(
        self,
        processed_dir: Path,
        manifest: dict,
        doc_dirs: list,
        batch_timestamp: str,
        force_rebuild: bool,
    ) -> Dict[str, Any]:
        """核心构建循环（由 build_from_processed_dir 在 .writing 保护下调用）"""
        all_chunks_data = []
        total_chunks = 0
        doc_results = []
        skipped_count = 0
        doc_quality_reports = []

        for doc_dir in sorted(doc_dirs):
            doc_name = doc_dir.name
            chunks = self._load_doc_chunks(doc_dir)
            if not chunks:
                Logger.warning(f"  跳过（无有效 chunks）: {doc_name}")
                doc_results.append({"doc_name": doc_name, "chunks": 0, "success": False})
                continue

            current_hash = self._content_hash(chunks)

            prev = manifest.get(doc_name, {})
            if not force_rebuild and prev.get("hash") == current_hash:
                skipped_count += 1
                Logger.info(f"  跳过（未变化）: {doc_name}")
                quality_report = self._load_doc_quality_report(doc_dir)
                if quality_report:
                    doc_quality_reports.append(quality_report)
                doc_results.append({
                    "doc_name": doc_name, "chunks": len(chunks),
                    "success": True, "skipped": True,
                    "doc_version_hash": prev.get("doc_version_hash", quality_report.get("doc_version_hash", "") if quality_report else ""),
                })
                for chunk in chunks:
                    chunk["doc_name"] = doc_name
                    chunk["full_doc_id"] = doc_name
                all_chunks_data.extend(chunks)
                total_chunks += len(chunks)
                continue

            doc_text = "\n\n".join(c["content"] for c in chunks)
            Logger.info(f"  构建文档: {doc_name} ({len(chunks)} chunks, {len(doc_text)} 字符)")

            try:
                loop = always_get_an_event_loop()
                insert_max_retries = self.config.get("performance", {}).get("insert_max_retries", 3)
                insert_retry_delay = self.config.get("performance", {}).get("insert_retry_delay", 2.0)
                had_prev_manifest = bool(prev)

                if not force_rebuild:
                    deleted_old_doc = self._delete_doc_if_exists(loop, doc_name)
                    if deleted_old_doc:
                        Logger.info(f"    已删除旧图谱贡献: {doc_name}")
                    elif had_prev_manifest:
                        Logger.info(f"    未发现旧 doc_status，按新文档重建: {doc_name}")

                track_id = self._insert_with_retry(
                    loop, doc_text, doc_name, str(doc_dir),
                    max_retries=insert_max_retries,
                    retry_delay=insert_retry_delay,
                )
                Logger.success(f"    图谱构建成功 (track_id: {track_id})")

                quality_report = self._load_doc_quality_report(doc_dir)
                if quality_report:
                    doc_quality_reports.append(quality_report)

                manifest[doc_name] = {
                    "hash": current_hash,
                    "built_at": datetime.now().isoformat(),
                    "doc_version_hash": quality_report.get("doc_version_hash", "") if quality_report else "",
                    "processed_batch_id": quality_report.get("processed_batch_id", batch_timestamp) if quality_report else batch_timestamp,
                }

                for chunk in chunks:
                    chunk["doc_name"] = doc_name
                    chunk["full_doc_id"] = doc_name
                all_chunks_data.extend(chunks)
                total_chunks += len(chunks)

                doc_results.append({
                    "doc_name": doc_name,
                    "chunks": len(chunks),
                    "chars": len(doc_text),
                    "track_id": track_id,
                    "success": True,
                    "doc_version_hash": manifest[doc_name].get("doc_version_hash", ""),
                })
            except Exception as e:
                Logger.error(f"    构建失败: {e}")
                import traceback
                Logger.error(traceback.format_exc())
                doc_results.append({
                    "doc_name": doc_name,
                    "chunks": len(chunks),
                    "success": False,
                    "error": str(e),
                })

        built_count = sum(
            1 for d in doc_results if d.get("success") and not d.get("skipped")
        )
        Logger.info(f"本次构建: {built_count} 个文档, 跳过: {skipped_count} 个（未变化）")

        # ── 保存增量清单（原子写入） ──
        self._save_graph_manifest(processed_dir, manifest)
        quality_summary = summarize_batch_quality(doc_quality_reports)
        doc_versions = {
            item.get("doc_name", ""): item.get("doc_version_hash", "")
            for item in doc_quality_reports
            if item.get("doc_name")
        }

        # ---------- 导出到 PostgreSQL（二次备份） ----------
        pg_export_cfg = self.config.get("pg_export", {})
        pg_result = {}

        if pg_export_cfg.get("enabled", True) and built_count > 0:
            db_cfg = self.config.get("database", {})
            if db_cfg:
                Logger.separator()
                Logger.info("导出图谱到 PostgreSQL（二次备份）...")

                exporter = PgGraphExporter(db_cfg, pg_export_cfg)
                exporter.init_tables()

                perf_cfg = self.config.get("performance", {})
                pg_result = exporter.export_all(
                    self.rag, all_chunks_data, batch_timestamp,
                    embedding_func=self._embedding_func,
                    entity_embed_batch=perf_cfg.get("entity_embed_batch", 32),
                )
            else:
                Logger.warning("未配置数据库，跳过 PostgreSQL 导出")
        elif built_count == 0:
            Logger.info("无新文档需要构建，跳过 PostgreSQL 导出")

        report = {
            "success": True,
            "batch_timestamp": batch_timestamp,
            "processed_dir": str(processed_dir),
            "total_documents": len(doc_dirs),
            "built_documents": built_count,
            "skipped_documents": skipped_count,
            "success_documents": sum(1 for d in doc_results if d.get("success")),
            "total_chunks": total_chunks,
            "doc_versions": doc_versions,
            "quality_summary": quality_summary,
            "pg_export": pg_result,
            "documents": doc_results,
            "completed_at": datetime.now().isoformat(),
        }

        self._save_workspace_manifest(Path(self._resolve_working_dir()), {
            "batch_timestamp": batch_timestamp,
            "completed_at": report["completed_at"],
            "built_documents": built_count,
            "skipped_documents": skipped_count,
            "doc_versions": doc_versions,
            "quality_summary": quality_summary,
            "graph_manifest": str(processed_dir / self.GRAPH_MANIFEST_NAME),
        })

        Logger.separator()
        Logger.success("知识图谱构建完成")
        Logger.info(f"文档: {report['success_documents']}/{report['total_documents']}"
                     f" (新构建 {built_count}, 跳过 {skipped_count})", indent=1)
        Logger.info(f"Chunks: {report['total_chunks']}", indent=1)
        if pg_result:
            Logger.info(
                f"PostgreSQL: {pg_result.get('entities', 0)} 实体, "
                f"{pg_result.get('relations', 0)} 关系, "
                f"{pg_result.get('chunks', 0)} chunks",
                indent=1,
            )
        Logger.separator()

        return report

    def _load_doc_chunks(self, doc_dir: Path) -> List[Dict[str, Any]]:
        """从文档目录加载 chunks"""
        try:
            index_file = doc_dir / "chunks_index.json"
            if not index_file.exists():
                return []

            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            chunks = []
            chunks_dir = doc_dir / "chunks"
            metadata_dir = doc_dir / "metadata"

            for chunk_info in index_data.get("chunks", []):
                chunk_index = chunk_info["index"]
                chunk_file = chunks_dir / f"chunk_{chunk_index:04d}.txt"
                metadata_file = metadata_dir / f"chunk_{chunk_index:04d}.json"

                if not chunk_file.exists():
                    continue

                with open(chunk_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if not content.strip():
                    continue

                metadata = chunk_info.get("metadata", {})
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as mf:
                            metadata = (json.load(mf) or {}).get("metadata", metadata)
                    except Exception:
                        pass

                chunks.append({
                    "chunk_id": chunk_info.get("chunk_id", ""),
                    "index": chunk_info["index"],
                    "content": content,
                    "char_count": chunk_info.get("char_count", len(content)),
                    "metadata": metadata,
                })

            return chunks

        except Exception as e:
            Logger.error(f"加载 chunks 失败: {e}")
            return []

    def _load_doc_quality_report(self, doc_dir: Path) -> Dict[str, Any]:
        """加载单文档质量报告。"""
        report_file = doc_dir / "quality_report.json"
        if not report_file.exists():
            return {}
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

