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

from ..core.config import load_processor_config, load_db_config, CONFIGS_DIR
from ..core.logger import Logger

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

    # 从 doc_splitter.yaml 读取 DeepSeek API Key
    deepseek_key = proc_cfg.get("deepseek", {}).get("api_key", "")
    if not lightrag_cfg.get("llm", {}).get("api_key"):
        lightrag_cfg.setdefault("llm", {})["api_key"] = deepseek_key

    # 从 doc_splitter.yaml 读取 embedding 相关配置
    emb_proc = proc_cfg.get("embedding", {})
    hf_cfg = emb_proc.get("huggingface", {})
    lightrag_cfg.setdefault("embedding", {})
    lightrag_cfg["embedding"]["cache_folder"] = hf_cfg.get("cache_folder", "./models")
    lightrag_cfg["embedding"]["device"] = hf_cfg.get("device", "cpu")

    # 数据库配置
    lightrag_cfg["database"] = db_cfg

    # 路径配置
    paths = proc_cfg.get("paths", {})
    lightrag_cfg["documents_dir"] = paths.get("documents_dir", "./")
    lightrag_cfg["processed_subdir"] = paths.get("processed_subdir", "processed")

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
    创建 DeepSeek LLM 调用函数（兼容 OpenAI 格式）

    Args:
        config: lightrag 配置

    Returns:
        异步 LLM 调用函数
    """
    llm_cfg = config.get("llm", {})
    api_base = llm_cfg.get("api_base", "https://api.deepseek.com")
    api_key = llm_cfg.get("api_key", "")
    model_name = llm_cfg.get("model", "deepseek-chat")

    if not api_key:
        Logger.error("未配置 DeepSeek API Key！请检查 doc_splitter.yaml 或 lightrag.yaml")
        raise ValueError("Missing DeepSeek API Key")

    # 设置环境变量（部分 LightRAG 内部逻辑会读取）
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
        DeepSeek LLM 调用（OpenAI 兼容格式，带自动重试）

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
                vectors = loop.run_until_complete(embedding_func.func(embed_texts))
                for key, vec in zip(embed_keys, vectors):
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
        """解析工作目录（相对路径基于 auto-doc-process/，与其他配置一致）"""
        from ..core.config import MODULE_DIR
        raw = self.config.get("working_dir", "../lightrag_workspace")
        p = Path(raw)
        if not p.is_absolute():
            p = (MODULE_DIR / p).resolve()
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
        """保存图谱增量清单"""
        manifest_path = processed_dir / LightRAGGraphBuilder.GRAPH_MANIFEST_NAME
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

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

        # ── 增量过滤：计算哈希，跳过未变化文档 ──
        all_chunks_data = []
        total_chunks = 0
        doc_results = []
        skipped_count = 0

        for doc_dir in sorted(doc_dirs):
            doc_name = doc_dir.name
            chunks = self._load_doc_chunks(doc_dir)
            if not chunks:
                Logger.warning(f"  跳过（无有效 chunks）: {doc_name}")
                doc_results.append({"doc_name": doc_name, "chunks": 0, "success": False})
                continue

            # 计算当前内容哈希
            current_hash = self._content_hash(chunks)

            # 增量判断：哈希未变 → 跳过
            prev = manifest.get(doc_name, {})
            if not force_rebuild and prev.get("hash") == current_hash:
                skipped_count += 1
                Logger.info(f"  跳过（未变化）: {doc_name}")
                doc_results.append({
                    "doc_name": doc_name, "chunks": len(chunks),
                    "success": True, "skipped": True,
                })
                # 仍然收集 chunk 数据用于 PG 导出
                for chunk in chunks:
                    chunk["doc_name"] = doc_name
                    chunk["full_doc_id"] = doc_name
                all_chunks_data.extend(chunks)
                total_chunks += len(chunks)
                continue

            # ── 需要构建 ──
            doc_text = "\n\n".join(c["content"] for c in chunks)
            Logger.info(f"  构建文档: {doc_name} ({len(chunks)} chunks, {len(doc_text)} 字符)")

            try:
                # 使用稳定文档 ID（不含时间戳），让 LightRAG 能识别同一文档
                loop = always_get_an_event_loop()
                track_id = loop.run_until_complete(
                    self.rag.ainsert(
                    doc_text,
                        ids=[doc_name],
                    file_paths=[str(doc_dir)],
                    )
                )
                Logger.success(f"    图谱构建成功 (track_id: {track_id})")

                # 更新增量清单
                manifest[doc_name] = {
                    "hash": current_hash,
                    "built_at": datetime.now().isoformat(),
                }

                # 记录 chunk 信息（用于后续导出到 PG）
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

        # ── 保存增量清单 ──
        self._save_graph_manifest(processed_dir, manifest)

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

                # 单事务原子导出：实体 + 关系 + chunks
                pg_result = exporter.export_all(
                    self.rag, all_chunks_data, batch_timestamp,
                    embedding_func=self._embedding_func,
                )
            else:
                Logger.warning("未配置数据库，跳过 PostgreSQL 导出")
        elif built_count == 0:
            Logger.info("无新文档需要构建，跳过 PostgreSQL 导出")

        # 构建报告
        report = {
            "success": True,
            "batch_timestamp": batch_timestamp,
            "processed_dir": str(processed_dir),
            "total_documents": len(doc_dirs),
            "built_documents": built_count,
            "skipped_documents": skipped_count,
            "success_documents": sum(1 for d in doc_results if d.get("success")),
            "total_chunks": total_chunks,
            "pg_export": pg_result,
            "documents": doc_results,
            "completed_at": datetime.now().isoformat(),
        }

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

            for chunk_info in index_data.get("chunks", []):
                chunk_index = chunk_info["index"]
                chunk_file = chunks_dir / f"chunk_{chunk_index:04d}.txt"

                if not chunk_file.exists():
                    continue

                with open(chunk_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if not content.strip():
                    continue

                chunks.append({
                    "chunk_id": chunk_info.get("chunk_id", ""),
                    "index": chunk_info["index"],
                    "content": content,
                    "char_count": chunk_info.get("char_count", len(content)),
                    "metadata": chunk_info.get("metadata", {}),
                })

            return chunks

        except Exception as e:
            Logger.error(f"加载 chunks 失败: {e}")
            return []

