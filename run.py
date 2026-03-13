#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
飞书文档自动处理管线 — 统一启动入口

用法（在上级目录执行）：
  .\auto-doc-process\venv\Scripts\python.exe auto-doc-process\run.py
  .\auto-doc-process\venv\Scripts\python.exe auto-doc-process\run.py --dry-run
  .\auto-doc-process\venv\Scripts\python.exe auto-doc-process\run.py --full
  .\auto-doc-process\venv\Scripts\python.exe auto-doc-process\run.py --no-graph
  .\auto-doc-process\venv\Scripts\python.exe auto-doc-process\run.py --schedule
"""
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types

# ─── 环境准备 ───────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ─── 包名 & 路径 ─────────────────────────────────────────────
# 目录名含连字符（auto-doc-process），Python import 不支持连字符。
# 注册一个下划线名（auto_doc_process）并安装自定义 finder，
# 让所有 auto_doc_process.* 的导入都能正确找到文件。
PKG = "auto_doc_process"
PKG_DIR = script_dir
CACHE_TAG = sys.implementation.cache_tag  # e.g. "cpython-311"


# ─── 自定义 Finder ───────────────────────────────────────────
class _PkgFinder(importlib.abc.MetaPathFinder):
    """处理 auto_doc_process 包及其所有子模块的导入。

    兼容两种部署方式：
      - 源码部署（.py 文件存在）
      - 编译部署（仅 __pycache__/*.pyc）
    """

    @staticmethod
    def _locate(parts):
        """根据模块路径片段定位文件系统上的 .py 或 .pyc。

        返回 (filepath, is_package)，找不到则返回 (None, False)。
        """
        rel = os.path.join(PKG_DIR, *parts)

        # 1) 目录 → 包
        if os.path.isdir(rel):
            # 优先 __init__.py
            init_py = os.path.join(rel, "__init__.py")
            if os.path.isfile(init_py):
                return init_py, True
            # 其次 __pycache__/__init__.cpython-XXX.pyc
            init_pyc = os.path.join(rel, "__pycache__",
                                    f"__init__.{CACHE_TAG}.pyc")
            if os.path.isfile(init_pyc):
                return init_pyc, True
            # namespace 包（无 __init__）
            return rel, True

        # 2) .py 文件
        py = rel + ".py"
        if os.path.isfile(py):
            return py, False

        # 3) __pycache__/*.pyc
        parent = os.path.dirname(rel)
        name = os.path.basename(rel)
        pyc = os.path.join(parent, "__pycache__", f"{name}.{CACHE_TAG}.pyc")
        if os.path.isfile(pyc):
            return pyc, False

        return None, False

    def find_spec(self, fullname, path, target=None):
        # 仅处理本包
        if fullname != PKG and not fullname.startswith(PKG + "."):
            return None

        # "auto_doc_process"        → parts = []
        # "auto_doc_process.core"   → parts = ["core"]
        # "auto_doc_process.core.config" → parts = ["core", "config"]
        suffix = fullname[len(PKG):]
        parts = suffix.lstrip(".").split(".") if suffix else []

        # 顶级包
        if not parts:
            init_py = os.path.join(PKG_DIR, "__init__.py")
            init_pyc = os.path.join(PKG_DIR, "__pycache__",
                                    f"__init__.{CACHE_TAG}.pyc")
            fp = init_py if os.path.isfile(init_py) else (
                init_pyc if os.path.isfile(init_pyc) else None)
            spec = importlib.machinery.ModuleSpec(
                PKG, None, is_package=True)
            spec.submodule_search_locations = [PKG_DIR]
            if fp:
                spec.origin = fp
            return spec

        filepath, is_pkg = self._locate(parts)
        if filepath is None:
            return None

        if is_pkg and os.path.isdir(filepath):
            # namespace 包
            spec = importlib.machinery.ModuleSpec(
                fullname, None, is_package=True)
            spec.submodule_search_locations = [filepath]
            return spec

        if is_pkg:
            return importlib.util.spec_from_file_location(
                fullname, filepath,
                submodule_search_locations=[
                    os.path.join(PKG_DIR, *parts)])
        else:
            return importlib.util.spec_from_file_location(
                fullname, filepath)


# 安装 finder（放在最前面，优先于默认 finder）
sys.meta_path.insert(0, _PkgFinder())

# 注册顶级包
pkg = types.ModuleType(PKG)
pkg.__path__ = [PKG_DIR]
pkg.__package__ = PKG
pkg.__file__ = os.path.join(PKG_DIR, "__init__.py")
sys.modules[PKG] = pkg

# ─── 启动 ────────────────────────────────────────────────────
if "--schedule" in sys.argv:
    sys.argv.remove("--schedule")
    scheduler_mod = importlib.import_module(f"{PKG}.scheduler")
    scheduler_mod.main()
elif "--_worker" in sys.argv:
    # 内部子进程模式（崩溃隔离）：处理文档列表，逐条写入 JSONL
    idx = sys.argv.index("--_worker")
    _input_file = sys.argv[idx + 1]
    _output_file = sys.argv[idx + 2]
    main_mod = importlib.import_module(f"{PKG}.__main__")
    main_mod._worker_process_documents(_input_file, _output_file)
else:
    main_mod = importlib.import_module(f"{PKG}.__main__")
    main_mod._entry()
