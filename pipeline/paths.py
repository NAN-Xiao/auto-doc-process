#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预处理产物路径解析。"""

from pathlib import Path

from ..core.config import MODULE_DIR, load_processor_config


def resolve_processed_dir() -> Path:
    """获取 processed 目录路径。"""
    proc_config = load_processor_config()
    paths = proc_config.get("paths", {})

    processed_dir = paths.get("processed_dir", "")
    if processed_dir:
        path = Path(processed_dir)
        if not path.is_absolute():
            path = (MODULE_DIR / path).resolve()
        return path

    documents_dir = paths.get("documents_dir", "../")
    docs_path = Path(documents_dir)
    if not docs_path.is_absolute():
        docs_path = (MODULE_DIR / docs_path).resolve()
    return docs_path / "processed"
