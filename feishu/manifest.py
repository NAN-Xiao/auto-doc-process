#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增量下载清单

职责：
  - 记录已下载文档的 token、编辑时间、文件路径
  - 判断文档是否有修改（需要重新下载）
  - 支持断点续传
  - 清单文件路径可通过配置指定
"""

import os
import json
from pathlib import Path
from datetime import datetime

from ..core.config import log, DEFAULT_MANIFEST_PATH


def _resolve_path(manifest_path: Path = None) -> Path:
    """获取清单文件路径"""
    return manifest_path if manifest_path else DEFAULT_MANIFEST_PATH


def load(manifest_path: Path = None) -> dict:
    """
    加载下载清单

    Args:
        manifest_path: 清单文件路径（None 使用默认路径）

    Returns:
        {token: {"downloaded_at": ..., "file_path": ..., "obj_edit_time": ..., "name": ...}, ...}
    """
    path = _resolve_path(manifest_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save(manifest: dict, manifest_path: Path = None):
    """
    保存下载清单

    Args:
        manifest: 清单数据
        manifest_path: 清单文件路径（None 使用默认路径）
    """
    path = _resolve_path(manifest_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except IOError as e:
        log.warning(f"保存清单失败: {e}")


def is_changed(manifest: dict, token: str, edit_time: str = None) -> bool:
    """
    判断文档是否有修改（需要重新下载）

    Args:
        manifest: 下载清单
        token: 文档 token
        edit_time: 文档最后编辑时间

    Returns:
        True = 需要下载，False = 可跳过
    """
    record = manifest.get(token)
    if not record:
        return True  # 从未下载

    # 文件已被删除，需要重新下载
    old_path = record.get("file_path", "")
    if old_path and not os.path.exists(old_path):
        return True

    # 比较编辑时间
    if edit_time and record.get("obj_edit_time"):
        return str(edit_time) != str(record["obj_edit_time"])

    return False  # 无编辑时间信息，已下载过，跳过


def record_download(manifest: dict, token: str, file_path: str,
                    edit_time: str = "", name: str = "",
                    manifest_path: Path = None):
    """
    记录一次成功下载

    Args:
        manifest: 下载清单（会被原地修改）
        token: 文档 token
        file_path: 下载文件路径
        edit_time: 文档编辑时间
        name: 文档名称
        manifest_path: 清单文件路径（None 使用默认路径）
    """
    manifest[token] = {
        "downloaded_at": datetime.now().isoformat(),
        "file_path": file_path,
        "obj_edit_time": edit_time,
        "name": name,
    }
    save(manifest, manifest_path)
