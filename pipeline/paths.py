#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预处理产物路径解析。"""

from pathlib import Path

from ..core.config import resolve_app_path, load_processor_config


def resolve_processed_dir() -> Path:
    """获取 processed 目录路径。"""
    proc_config = load_processor_config()
    paths = proc_config.get("paths", {})

    processed_dir = paths.get("processed_dir", "")
    if processed_dir:
        return resolve_app_path(processed_dir)

    documents_dir = paths.get("documents_dir", "../")
    docs_path = resolve_app_path(documents_dir)
    return docs_path / "processed"
