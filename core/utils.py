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
import time
import tempfile
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
    """生成合法且 URL 安全的文件名

    处理规则：
      1. 去除 Windows 非法字符  \\ / : * ? " < > |
      2. 空格 → 下划线（防止 Markdown URL 断裂）
      3. 半角 URL 特殊字符  # % & + ( ) [ ] { } → 下划线
      4. 全角括号/特殊符号  （）【】｛｝＃％＆＋ → 下划线
      5. 合并连续下划线、去除首尾空白和点号
    """
    s = name or fallback
    # Windows 非法字符 + URL 特殊字符 + 空格 → 下划线（半角）
    s = re.sub(r'[\\/:*?"<>|\s#%&+\(\)\[\]{}]', "_", s)
    # 全角括号 / 方括号 / 花括号 / 特殊符号 → 下划线
    s = re.sub(r'[（）【】｛｝＃％＆＋　]', "_", s)
    # 合并连续下划线
    s = re.sub(r'_+', '_', s)
    s = s.strip("._ ")
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


# ==================== 原子文件写入 ====================

def atomic_write_json(
    filepath: Path,
    data,
    *,
    max_retries: int = 3,
    retry_delay: float = 0.5,
    indent: int = 2,
) -> None:
    """
    原子写入 JSON 文件：写临时文件 → os.replace() 原子替换。

    保证：
      - 写入过程中崩溃不会损坏原文件
      - 读端永远看到完整的旧文件或完整的新文件，不会读到半写内容
      - 失败自动重试（指数退避）

    Args:
        filepath: 目标文件路径
        data: 要序列化的数据
        max_retries: 最大重试次数
        retry_delay: 首次重试延迟（秒），后续指数增长
        indent: JSON 缩进
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(data, ensure_ascii=False, indent=indent)

    last_err = None
    for attempt in range(1, max_retries + 1):
        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp",
                prefix=f".{filepath.stem}_",
                dir=str(filepath.parent),
            )
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = None

            os.replace(tmp_path, str(filepath))
            return
        except Exception as e:
            last_err = e
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                log.warning(
                    f"atomic_write_json 失败 ({filepath.name}), "
                    f"重试 {attempt}/{max_retries}, {delay:.1f}s 后: {e}"
                )
                time.sleep(delay)

    log.error(f"atomic_write_json 最终失败 ({filepath.name}): {last_err}")
    raise last_err


# ==================== Workspace 读写协调信号 ====================

_WRITING_MARKER = ".writing"
_READY_MARKER = ".ready"


def workspace_begin_write(workspace_dir: Path) -> None:
    """
    在 workspace 目录放置 .writing 标记，通知 RAG 读端暂停读取。

    RAG 系统应在读取前检查此文件：存在则等待或跳过本次读取。
    """
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    marker = workspace_dir / _WRITING_MARKER
    marker.write_text(
        json.dumps({
            "pid": os.getpid(),
            "started_at": datetime.now().isoformat(),
        }),
        encoding="utf-8",
    )

    ready = workspace_dir / _READY_MARKER
    if ready.exists():
        try:
            ready.unlink()
        except OSError:
            pass


def workspace_end_write(workspace_dir: Path, summary: dict = None) -> None:
    """
    移除 .writing 标记，写入 .ready 信号文件（原子替换）。

    .ready 内容包含构建时间戳和摘要，RAG 读端可据此判断是否需要重新加载。
    """
    workspace_dir = Path(workspace_dir)

    writing = workspace_dir / _WRITING_MARKER
    if writing.exists():
        try:
            writing.unlink()
        except OSError:
            pass

    ready_data = {
        "completed_at": datetime.now().isoformat(),
        "pid": os.getpid(),
    }
    if summary:
        ready_data["summary"] = summary

    atomic_write_json(workspace_dir / _READY_MARKER, ready_data)
