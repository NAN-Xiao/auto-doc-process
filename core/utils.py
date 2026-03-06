#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数

职责：
  - 飞书文档 URL 解析
  - 安全文件名生成
  - 进程锁（防止重复运行）
  - 文档类型常量（默认值，可被配置覆盖）
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

from .config import log, DEFAULT_LOCK_PATH  # noqa: same-package import


# ==================== 文档类型常量（默认值） ====================

# 可导出的文档类型 → 默认导出格式（运行时可被配置覆盖）
EXPORTABLE_TYPES = {
    "doc": "docx",
    "docx": "docx",
    "sheet": "xlsx",
    "bitable": "xlsx",
}

# 不可导出的类型（跳过，不报错）
SKIP_TYPES = {"mindnote", "file", "slides", "catalog"}


# ==================== URL 解析 ====================

_URL_PATTERNS = [
    (r"feishu\.cn/wiki/([A-Za-z0-9_-]+)", "wiki", "docx"),
    (r"feishu\.cn/docx/([A-Za-z0-9_-]+)", "docx", "docx"),
    (r"feishu\.cn/docs?/([A-Za-z0-9_-]+)", "doc", "docx"),
    (r"feishu\.cn/sheets?/([A-Za-z0-9_-]+)", "sheet", "xlsx"),
    (r"feishu\.cn/(?:base|bitable)/([A-Za-z0-9_-]+)", "bitable", "xlsx"),
]


def parse_feishu_url(url: str) -> dict:
    """
    从飞书文档 URL 中解析 token、文档类型和默认导出格式

    Returns:
        {"token": ..., "doc_type": ..., "default_ext": ..., "url": ...}
        解析失败返回 None
    """
    for pattern, doc_type, default_ext in _URL_PATTERNS:
        m = re.search(pattern, url)
        if m:
            return {
                "token": m.group(1),
                "doc_type": doc_type,
                "default_ext": default_ext,
                "url": url,
            }
    return None


# ==================== 文件名 ====================

def safe_filename(name: str, fallback: str) -> str:
    """生成合法的文件名（去除 Windows 非法字符）"""
    s = name or fallback
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = s.strip(". ")
    return s or fallback


# ==================== 进程锁 ====================

def acquire_lock(lock_path: Path = None) -> bool:
    """
    获取进程锁，防止重复运行

    Args:
        lock_path: 锁文件路径（None 使用默认路径）

    Returns:
        True = 获取成功，False = 已有实例在运行
    """
    lock_file = lock_path or DEFAULT_LOCK_PATH

    if lock_file.exists():
        try:
            with open(lock_file, "r") as f:
                info = json.load(f)
            pid = info.get("pid", 0)
            start_time = info.get("start_time", "")

            try:
                os.kill(pid, 0)  # 仅检查进程是否存在
                log.warning(f"另一个实例正在运行 (PID={pid}, 启动于 {start_time})")
                return False
            except (OSError, ProcessLookupError):
                log.info(f"清理残留锁文件 (旧 PID={pid})")
        except (json.JSONDecodeError, IOError):
            pass

    try:
        with open(lock_file, "w") as f:
            json.dump({"pid": os.getpid(), "start_time": datetime.now().isoformat()}, f)
        return True
    except IOError as e:
        log.error(f"无法创建锁文件: {e}")
        return False


def release_lock(lock_path: Path = None):
    """释放进程锁"""
    lock_file = lock_path or DEFAULT_LOCK_PATH
    try:
        if lock_file.exists():
            lock_file.unlink()
    except IOError:
        pass
