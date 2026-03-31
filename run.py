#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
飞书文档自动处理管线 — 统一启动入口

用法（在上级目录执行）：
  .\auto-doc-process\venv\Scripts\python.exe auto-doc-process\run.py
  .\auto-doc-process\venv\Scripts\python.exe auto-doc-process\run.py --dry-run
  .\auto-doc-process\venv\Scripts\python.exe auto-doc-process\run.py --full
  .\auto-doc-process\venv\Scripts\python.exe auto-doc-process\run.py --no-graph
  .\auto-doc-process\venv\Scripts\python.exe auto-doc-process\run.py --schedule
"""
import glob
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

# ─── Python 版本校验（编译部署时 .pyc 版本必须匹配） ────────
_version_file = os.path.join(script_dir, "version.txt")
if os.path.isfile(_version_file):
    with open(_version_file, encoding="utf-8") as _vf:
        for _line in _vf:
            if _line.startswith("python="):
                _build_py = _line.strip().split("=", 1)[1]  # e.g. "3.11"
                _runtime_py = f"{sys.version_info.major}.{sys.version_info.minor}"
                if _build_py != _runtime_py:
                    print(
                        f"\n[ERROR] Python version mismatch!\n"
                        f"  Build:   Python {_build_py}  (.pyc bytecode)\n"
                        f"  Runtime: Python {_runtime_py}\n\n"
                        f"  .pyc bytecode is NOT portable across Python minor versions.\n"
                        f"  Solutions:\n"
                        f"    1. Install Python {_build_py}.x on this machine\n"
                        f"    2. Or rebuild the package using Python {_runtime_py}\n",
                        file=sys.stderr, flush=True,
                    )
                    sys.exit(1)
                break

# ─── 包名 & 路径 ─────────────────────────────────────────────
# 目录名含连字符（auto-doc-process），Python import 不支持连字符。
# 注册一个下划线名（auto_doc_process）并安装自定义 finder，
# 让所有 auto_doc_process.* 的导入都能正确找到文件。
PKG = "auto_doc_process"
PKG_DIR = script_dir
CACHE_TAG = sys.implementation.cache_tag  # e.g. "cpython-311"


def _find_pyc(cache_dir, module_name):
    """在 __pycache__ 中查找 .pyc，优先精确匹配当前 cache tag，
    否则回退到任意 cpython-*.pyc（跨小版本兼容）。
    找到非精确匹配时检测大版本差异并给出警告。
    """
    exact = os.path.join(cache_dir, f"{module_name}.{CACHE_TAG}.pyc")
    if os.path.isfile(exact):
        return exact

    pattern = os.path.join(cache_dir, f"{module_name}.cpython-*.pyc")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    pyc = candidates[0]
    # 从文件名提取构建时的 Python 版本，如 cpython-311 → 3.11
    base = os.path.basename(pyc)
    tag_part = base.rsplit(".", 2)[1] if base.count(".") >= 2 else ""
    build_ver = tag_part.replace("cpython-", "") if tag_part.startswith("cpython-") else ""
    runtime_ver = f"{sys.version_info.major}{sys.version_info.minor}"

    if build_ver and build_ver != runtime_ver:
        bv = f"{build_ver[0]}.{build_ver[1:]}"
        rv = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(
            f"[ERROR] Python version mismatch!\n"
            f"  Build:   Python {bv}\n"
            f"  Runtime: Python {rv}\n"
            f"  .pyc bytecode is NOT compatible across minor versions.\n"
            f"  Please install Python {bv}.x on this machine,\n"
            f"  or rebuild the package with Python {rv}.",
            file=sys.stderr,
        )
        sys.exit(1)

    return pyc


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
            init_py = os.path.join(rel, "__init__.py")
            if os.path.isfile(init_py):
                return init_py, True
            cache_dir = os.path.join(rel, "__pycache__")
            init_pyc = _find_pyc(cache_dir, "__init__")
            if init_pyc:
                return init_pyc, True
            return rel, True

        # 2) .py 文件
        py = rel + ".py"
        if os.path.isfile(py):
            return py, False

        # 3) __pycache__/*.pyc
        parent = os.path.dirname(rel)
        name = os.path.basename(rel)
        cache_dir = os.path.join(parent, "__pycache__")
        pyc = _find_pyc(cache_dir, name)
        if pyc:
            return pyc, False

        return None, False

    def find_spec(self, fullname, path, target=None):
        if fullname != PKG and not fullname.startswith(PKG + "."):
            return None

        suffix = fullname[len(PKG):]
        parts = suffix.lstrip(".").split(".") if suffix else []

        # 顶级包
        if not parts:
            init_py = os.path.join(PKG_DIR, "__init__.py")
            cache_dir = os.path.join(PKG_DIR, "__pycache__")
            init_pyc = _find_pyc(cache_dir, "__init__")
            fp = init_py if os.path.isfile(init_py) else (
                init_pyc if init_pyc else None)
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
