#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数

职责：
  - 飞书文档 URL 解析
  - 安全文件名生成
  - 进程锁（防止重复运行，使用 OS 级文件锁消除竞态条件）
  - 文档类型常量（默认值，可被配置覆盖）
"""

import os
import sys
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


def extract_docx_title(file_path: str) -> str:
    """
    从 docx 文件中提取真实文档标题

    优先级：
      1. 第一个 Heading 样式段落的文本
      2. 第一个非空段落的文本（兜底）

    Args:
        file_path: docx 文件路径

    Returns:
        提取到的标题（去除首尾空白），提取失败返回空字符串
    """
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)

        first_text = ""
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # 记住第一个非空段落作为兜底
            if not first_text:
                first_text = text
            # 优先返回 Heading 样式
            style_name = (para.style.name or "").lower()
            if style_name.startswith("heading") or style_name.startswith("title"):
                return text

        return first_text
    except Exception as e:
        log.debug(f"提取 docx 标题失败 ({file_path}): {e}")
        return ""


# ==================== 进程锁（OS 级文件锁） ====================

# 全局持有锁文件句柄，防止被 GC 关闭
_lock_file_handle = None


def acquire_lock(lock_path: Path = None) -> bool:
    """
    获取进程锁，防止重复运行。

    使用 OS 级别文件锁（Windows: msvcrt, POSIX: fcntl）消除 TOCTOU 竞态条件。

    Args:
        lock_path: 锁文件路径（None 使用默认路径）

    Returns:
        True = 获取成功，False = 已有实例在运行
    """
    global _lock_file_handle
    lock_file = lock_path or DEFAULT_LOCK_PATH

    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        # 以读写模式打开（不存在则创建）
        fh = open(lock_file, "w")

        if sys.platform == "win32":
            import msvcrt
            try:
                # 非阻塞独占锁（锁定第一个字节即可）
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except (OSError, IOError):
                fh.close()
                log.warning("另一个实例正在运行（无法获取文件锁）")
                return False
        else:
            import fcntl
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, IOError):
                fh.close()
                log.warning("另一个实例正在运行（无法获取文件锁）")
                return False

        # 锁定成功，写入进程信息
        fh.truncate(0)
        fh.seek(0)
        json.dump({"pid": os.getpid(), "start_time": datetime.now().isoformat()}, fh)
        fh.flush()

        # 保持句柄打开（锁依赖于文件句柄的生命周期）
        _lock_file_handle = fh
        return True

    except Exception as e:
        log.error(f"无法获取进程锁: {e}")
        return False


def release_lock(lock_path: Path = None):
    """释放进程锁"""
    global _lock_file_handle
    lock_file = lock_path or DEFAULT_LOCK_PATH

    if _lock_file_handle is not None:
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    _lock_file_handle.seek(0)
                    msvcrt.locking(_lock_file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            else:
                import fcntl
                fcntl.flock(_lock_file_handle.fileno(), fcntl.LOCK_UN)
            _lock_file_handle.close()
        except Exception:
            pass
        _lock_file_handle = None

    # 尝试删除锁文件
    try:
        if lock_file.exists():
            lock_file.unlink()
    except OSError:
        pass
