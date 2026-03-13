#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 HuggingFace Transformer 模型导出为 ONNX 格式

用法（在 auto-doc-process 目录下执行，需要 torch 环境）：
  venv\\Scripts\\python.exe tools\\export_onnx.py

导出后，服务器端仅需 onnxruntime + tokenizers，不再需要 torch（节省 ~600MB）。

输出：
  models/onnx/model.onnx          ONNX 模型文件（~25MB）
  models/onnx/tokenizer.json      分词器文件
  models/onnx/pooling_config.json 池化配置
"""

import os
import sys
import json
import shutil
from pathlib import Path

# 项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"


def find_model_snapshot(model_name: str) -> Path:
    """查找 HuggingFace 模型缓存快照目录"""
    safe_name = model_name.replace("/", "--")
    model_dir = MODELS_DIR / f"models--{safe_name}"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"模型缓存不存在: {model_dir}\n"
            f"请先运行一次完整流程以下载模型，或手动下载到 {MODELS_DIR}"
        )
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"模型快照目录不存在: {snapshots}")
    # 取最新快照
    dirs = sorted(snapshots.iterdir())
    if not dirs:
        raise FileNotFoundError(f"模型快照目录为空: {snapshots}")
    return dirs[-1]


def export_to_onnx(model_name: str = "BAAI/bge-small-zh-v1.5"):
    """将模型导出为 ONNX 格式"""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("[错误] 导出需要 torch 和 transformers，请在开发环境中运行此脚本")
        sys.exit(1)

    snapshot_dir = find_model_snapshot(model_name)
    output_dir = MODELS_DIR / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ONNX 模型导出")
    print("=" * 60)
    print(f"  源模型:   {snapshot_dir}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)
    print()

    # 1. 加载模型和分词器
    print("[1/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
    model = AutoModel.from_pretrained(str(snapshot_dir))
    model.eval()

    # 2. 创建虚拟输入（batch_size > 1，确保导出时批量维度不被固定）
    print("[2/4] 准备导出...")
    dummy_texts = ["这是一个测试文本", "另一段文本用于确保动态维度"]
    inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # 3. 导出 ONNX
    print("[3/4] 导出 ONNX 模型...")
    onnx_path = output_dir / "model.onnx"

    # 使用 dynamo 导出 + dynamic_shapes 确保批处理维度可变
    batch = torch.export.Dim("batch", min=1, max=256)
    seq = torch.export.Dim("seq", min=1, max=512)

    # 删除旧文件（dynamo 导出可能创建 .data 外部文件）
    for old_f in onnx_path.parent.glob("model.onnx*"):
        old_f.unlink()

    # 使用 kwargs 传递输入，确保 dynamic_shapes 映射正确
    kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs.get("token_type_ids"),
    }

    torch.onnx.export(
        model,
        (),  # 无位置参数
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        kwargs=kwargs,
        dynamic_shapes={
            "input_ids": {0: batch, 1: seq},
            "attention_mask": {0: batch, 1: seq},
            "token_type_ids": {0: batch, 1: seq},
        },
        opset_version=18,
    )
    print(f"  ONNX 模型已保存: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 4. 复制分词器文件和配置
    print("[4/4] 复制配置文件...")
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        src = snapshot_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"  已复制: {fname}")

    # 复制池化配置
    pooling_src = snapshot_dir / "1_Pooling" / "config.json"
    if pooling_src.exists():
        shutil.copy2(pooling_src, output_dir / "pooling_config.json")
        print("  已复制: pooling_config.json")

    # 写入模型元信息
    meta = {
        "model_name": model_name,
        "embedding_dim": 512,
        "pooling_mode": "cls",
        "normalize": True,
        "max_length": 512,
        "exported_from": str(snapshot_dir),
    }
    with open(output_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("  已写入: model_meta.json")

    # 统计
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print()
    print("=" * 60)
    print("  导出完成！")
    print("=" * 60)
    print(f"  ONNX 目录: {output_dir}")
    print(f"  总大小:    {total_size / 1024 / 1024:.1f} MB")
    print()
    print("  服务器部署时，只需复制 models/onnx/ 目录，")
    print("  安装 onnxruntime 替代 torch，即可运行。")
    print("=" * 60)


if __name__ == "__main__":
    export_to_onnx()

