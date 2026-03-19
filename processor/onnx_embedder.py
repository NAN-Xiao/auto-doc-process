#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX Runtime Embedding 引擎

在无 PyTorch 的服务器环境中使用 ONNX Runtime + tokenizers 生成 embeddings，
替代 langchain_huggingface.HuggingFaceEmbeddings（需要 torch ~450MB）。

依赖：
  - onnxruntime (~20MB)
  - tokenizers   (已随 huggingface_hub 安装)

用法：
  embeddings = create_embeddings(config)
  vectors = embeddings.embed_documents(["文本1", "文本2"])

当检测到 models/onnx/model.onnx 存在时自动使用 ONNX 引擎，
否则回退到 HuggingFaceEmbeddings（需要 torch）。
"""

import json
import numpy as np
from pathlib import Path
from typing import List

from ..core.config import MODULE_DIR
from ..core.logger import Logger


class OnnxEmbeddings:
    """
    基于 ONNX Runtime 的轻量 Embedding 引擎

    实现与 langchain HuggingFaceEmbeddings 相同的接口：
      - embed_documents(texts: List[str]) -> List[List[float]]
      - embed_query(text: str) -> List[float]
    """

    def __init__(self, onnx_dir: str, max_length: int = 512,
                 normalize: bool = True, pooling: str = "cls",
                 intra_op_num_threads: int = 2, inter_op_num_threads: int = 2):
        """
        Args:
            onnx_dir: ONNX 模型目录（包含 model.onnx + tokenizer 文件）
            max_length: 最大 token 长度
            normalize: 是否对输出做 L2 归一化
            pooling: 池化方式 ("cls" 或 "mean")
            intra_op_num_threads: 算子内线程数（低配可 2，多核可调大）
            inter_op_num_threads: 算子间线程数
        """
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_dir = Path(onnx_dir)
        model_path = onnx_dir / "model.onnx"
        tokenizer_path = onnx_dir / "tokenizer.json"

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"分词器不存在: {tokenizer_path}")

        # 加载 ONNX 模型（线程数由配置或参数指定）
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.inter_op_num_threads = inter_op_num_threads

        self.session = ort.InferenceSession(
            str(model_path), sess_options,
            providers=["CPUExecutionProvider"],
        )

        # 加载分词器
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(length=None)  # 动态 padding

        self.max_length = max_length
        self.normalize = normalize
        self.pooling = pooling

        # 检查模型输入名称
        self.input_names = [inp.name for inp in self.session.get_inputs()]

        Logger.info(f"ONNX Embedding 引擎已加载: {model_path.name}")
        Logger.info(
            f"  池化: {pooling}, 归一化: {normalize}, max_length: {max_length}, "
            f"线程: intra={intra_op_num_threads} inter={inter_op_num_threads}",
            indent=1,
        )

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """将一批文本编码为向量"""
        # 分词
        encodings = self.tokenizer.encode_batch(texts)

        # 构建输入
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # 部分模型需要 token_type_ids
        if "token_type_ids" in self.input_names:
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
            feeds["token_type_ids"] = token_type_ids

        # 推理
        outputs = self.session.run(None, feeds)
        hidden_states = outputs[0]  # (batch, seq_len, hidden_dim)

        # 池化
        if self.pooling == "cls":
            embeddings = hidden_states[:, 0, :]  # CLS token
        else:
            # mean pooling
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_embeddings = np.sum(hidden_states * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            embeddings = sum_embeddings / sum_mask

        # L2 归一化
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            embeddings = embeddings / norms

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成文档向量（与 LangChain HuggingFaceEmbeddings 接口兼容）

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量为 List[float]
        """
        if not texts:
            return []
        embeddings = self._encode_batch(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """生成查询向量"""
        return self.embed_documents([text])[0]


def is_onnx_available(config: dict = None) -> bool:
    """检查 ONNX 模型和 onnxruntime 是否可用"""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False

    onnx_dir = _resolve_onnx_dir(config)
    return (onnx_dir / "model.onnx").exists()


def _resolve_onnx_dir(config: dict = None) -> Path:
    """解析 ONNX 模型目录路径"""
    if config:
        emb_cfg = config.get("embedding", {})
        hf_cfg = emb_cfg.get("huggingface", {})
        cache = hf_cfg.get("cache_folder", "")
        if cache:
            p = Path(cache)
            if not p.is_absolute():
                p = (MODULE_DIR / p).resolve()
            return p / "onnx"
    return MODULE_DIR / "models" / "onnx"


def create_embeddings(config: dict = None):
    """
    创建 Embedding 引擎（自动选择 ONNX 或 torch 后端）

    优先使用 ONNX（轻量），不可用时回退到 HuggingFaceEmbeddings（需要 torch）。

    Args:
        config: 处理器配置字典

    Returns:
        实现 embed_documents / embed_query 接口的对象
    """
    if is_onnx_available(config):
        onnx_dir = _resolve_onnx_dir(config)
        # 读取元信息
        meta_file = onnx_dir / "model_meta.json"
        pooling = "cls"
        normalize = True
        max_length = 512
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            pooling = meta.get("pooling_mode", "cls")
            normalize = meta.get("normalize", True)
            max_length = meta.get("max_length", 512)

        # 从配置读取 ONNX 线程数（多核可调大）
        onnx_cfg = (config or {}).get("embedding", {}).get("onnx", {})
        intra = onnx_cfg.get("intra_op_num_threads", 2)
        inter = onnx_cfg.get("inter_op_num_threads", 2)

        Logger.info("使用 ONNX Runtime 引擎（轻量模式）")
        return OnnxEmbeddings(
            onnx_dir=str(onnx_dir),
            max_length=max_length,
            normalize=normalize,
            pooling=pooling,
            intra_op_num_threads=intra,
            inter_op_num_threads=inter,
        )
    else:
        Logger.info("ONNX 模型不可用，使用 HuggingFace/torch 引擎")
        from langchain_huggingface import HuggingFaceEmbeddings

        if config is None:
            from ..core.config import load_processor_config
            config = load_processor_config()

        emb_cfg = config.get("embedding", {})
        model_name = emb_cfg.get("model", "BAAI/bge-small-zh-v1.5")
        hf_cfg = emb_cfg.get("huggingface", {})
        cache_folder = hf_cfg.get("cache_folder", "./models")
        device = hf_cfg.get("device", "cpu")
        normalize_emb = hf_cfg.get("normalize_embeddings", True)

        Logger.info(f"加载 Embedding 模型: {model_name} (device: {device})")
        Logger.info(f"模型缓存目录: {cache_folder}", indent=1)

        return HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize_emb},
        )

