#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打包部署脚本 — 编译 .py → .pyc 字节码，不含源码

用法（在 auto-doc-process 目录下执行）：
  venv\\Scripts\\python.exe build.py                          # 基础打包
  venv\\Scripts\\python.exe build.py --slim                   # 轻量部署（ONNX 模型 + 精简依赖）
  venv\\Scripts\\python.exe build.py --include-models         # 包含 Embedding 模型（约 100MB）
  venv\\Scripts\\python.exe build.py --include-venv           # 包含虚拟环境（约 1-2GB，目标机无需再装）

输出：
  ../dist/auto-doc-process/     编译后的部署包（可直接复制到目标电脑）

注意：
  - 目标电脑的 Python 大版本须与编译版本一致（如都是 3.11.x）
  - 如果使用 --include-venv 则无需在目标机安装 Python
  - --slim 模式下自动包含 ONNX 模型，使用 requirements-server.txt 安装依赖
"""

import os
import sys
import shutil
import py_compile
import compileall
import argparse
from pathlib import Path


def build(include_models: bool = False, include_venv: bool = False,
          slim: bool = False):
    src_dir = Path(__file__).parent.resolve()
    dist_root = src_dir.parent / "dist"
    dist_dir = dist_root / "auto-doc-process"

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    # --slim 隐含包含 ONNX 模型
    if slim:
        include_models = True

    print("=" * 60)
    print("  飞书文档自动处理管线 — 打包部署")
    print("=" * 60)
    print(f"  源目录:     {src_dir}")
    print(f"  输出目录:   {dist_dir}")
    print(f"  Python:     {py_ver} ({sys.executable})")
    print(f"  模式:       {'轻量 (ONNX + 精简依赖)' if slim else '完整 (torch)'}")
    print(f"  包含模型:   {'是' if include_models else '否'}")
    print(f"  包含 venv:  {'是' if include_venv else '否'}")
    print("=" * 60)
    print()

    # ─── 1. 清理旧构建 ───────────────────────────────────────
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print("[清理] 已删除旧构建目录")

    # ─── 2. 复制项目到 dist ──────────────────────────────────
    ignore_list = [
        "venv", "__pycache__", "*.pyc", "*.pyo",
        "build.py", "dist", ".git", ".gitignore",
        "*.log", ".feishu_export*", "_runtime", "tools",
    ]
    if not include_models:
        ignore_list.append("models")

    shutil.copytree(
        src_dir, dist_dir,
        ignore=shutil.ignore_patterns(*ignore_list),
    )
    print("[复制] 项目文件已复制")

    # --slim：仅保留 ONNX 模型，删除 HuggingFace 原始模型
    if slim and include_models:
        models_dist = dist_dir / "models"
        onnx_dir = models_dist / "onnx"
        if onnx_dir.exists():
            # 删除非 onnx 的目录（HuggingFace 缓存）
            for item in models_dist.iterdir():
                if item.name != "onnx":
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            print("[瘦身] 已移除 HuggingFace 原始模型，仅保留 ONNX")
        else:
            print("[警告] ONNX 模型不存在 (models/onnx/)，请先运行 tools/export_onnx.py")

    # ─── 3. 删除敏感配置（只保留 .example 模板） ─────────────
    configs_dir = dist_dir / "configs"
    for sensitive in ("feishu.yaml", "db_info.yml", "doc_splitter.yaml"):
        f = configs_dir / sensitive
        if f.exists():
            f.unlink()
            print(f"[安全] 删除敏感配置: {sensitive}")

    # ─── 4. 编译 .py → __pycache__/*.cpython-3XX.pyc ────────
    print("[编译] 编译 Python 字节码（unchecked-hash 模式，无需源码校验）...")
    ok = compileall.compile_dir(
        str(dist_dir),
        force=True,
        quiet=1,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    if not ok:
        print("[错误] 编译失败！")
        sys.exit(1)
    print("[编译] 全部编译完成")

    # ─── 5. 删除 .py 源码（仅保留 run.py 启动入口） ─────────
    # run.py 只有 15 行引导代码，无任何业务逻辑，保留为 .py 方便 bat 调用
    keep_source = {"run.py"}
    deleted = 0
    for py_file in dist_dir.rglob("*.py"):
        rel = py_file.relative_to(dist_dir)
        # 只保留根目录下的 run.py
        if py_file.name in keep_source and py_file.parent == dist_dir:
            continue
        py_file.unlink()
        deleted += 1
    print(f"[清理] 删除 {deleted} 个 .py 源码文件")

    # ─── 6. 可选：包含虚拟环境 ──────────────────────────────
    if include_venv:
        venv_src = src_dir / "venv"
        venv_dst = dist_dir / "venv"
        if venv_src.exists():
            print("[复制] 复制虚拟环境（可能需要几分钟）...")
            shutil.copytree(venv_src, venv_dst)
            print("[复制] 虚拟环境复制完成")
        else:
            print("[警告] 未找到 venv/ 目录，跳过")

    # ─── 7. 创建 setup.bat（目标机首次部署用） ───────────────
    req_file = "requirements-server.txt" if slim else "requirements.txt"
    setup_bat = dist_dir / "setup.bat"
    setup_bat.write_text(
        '@echo off\n'
        'chcp 65001 >nul\n'
        'echo.\n'
        'echo ====================================================\n'
        'echo  飞书文档自动处理 - 首次部署\n'
        'echo ====================================================\n'
        + ('echo  模式: 轻量部署 (ONNX Runtime)\n' if slim else '')
        + 'echo.\n'
        '\n'
        'cd /d "%~dp0"\n'
        '\n'
        'if not exist "venv\\Scripts\\python.exe" (\n'
        '    echo [1/3] 创建虚拟环境...\n'
        '    python -m venv venv\n'
        '    if errorlevel 1 (\n'
        '        echo [错误] 创建虚拟环境失败，请确认已安装 Python ' + py_ver + '\n'
        '        pause\n'
        '        exit /b 1\n'
        '    )\n'
        ') else (\n'
        '    echo [1/3] 虚拟环境已存在，跳过\n'
        ')\n'
        '\n'
        'echo [2/3] 安装依赖...\n'
        'venv\\Scripts\\pip install -r ' + req_file + '\n'
        'if errorlevel 1 (\n'
        '    echo [错误] 依赖安装失败\n'
        '    pause\n'
        '    exit /b 1\n'
        ')\n'
        '\n'
        'echo.\n'
        'echo [3/3] 请配置以下文件：\n'
        'echo   configs\\feishu.yaml.example     -^>  feishu.yaml\n'
        'echo   configs\\db_info.yml.example      -^>  db_info.yml\n'
        'echo   configs\\doc_splitter.yaml.example -^>  doc_splitter.yaml\n'
        'echo.\n'
        'echo 最低配置: 2 核 CPU / 4GB RAM / 10GB 磁盘\n'
        'echo 推荐配置: 4 核 CPU / 8GB RAM / 20GB 磁盘\n'
        'echo 系统要求: PostgreSQL 14+ (pgvector), Python ' + py_ver + '\n'
        'echo.\n'
        'echo 部署完成后运行:\n'
        'echo   start.bat                双击手动执行\n'
        'echo   deploy.bat install       注册 Windows 定时任务\n'
        'echo.\n'
        'pause\n',
        encoding="utf-8",
    )
    print("[生成] setup.bat（目标机首次部署脚本）")

    # ─── 8. 统计 ─────────────────────────────────────────────
    total_size = sum(f.stat().st_size for f in dist_dir.rglob("*") if f.is_file())
    pyc_count = len(list(dist_dir.rglob("*.pyc")))
    py_count = len(list(dist_dir.rglob("*.py")))

    print()
    print("=" * 60)
    print("  构建成功！")
    print("=" * 60)
    print(f"  输出目录:  {dist_dir}")
    print(f"  .pyc 文件: {pyc_count} 个（已编译字节码）")
    print(f"  .py 文件:  {py_count} 个（仅 run.py 启动入口）")
    print(f"  总大小:    {total_size / 1024 / 1024:.1f} MB")
    print(f"  Python:    {py_ver}（目标机须一致）")
    print()
    print("  部署到目标电脑：")
    print(f"    1. 复制 {dist_dir.name}/ 到目标目录")
    if not include_venv:
        print(f"    2. 双击 setup.bat 创建虚拟环境并安装依赖")
        print(f"    3. 配置 configs/ 下的三个文件")
        print(f"    4. deploy.bat install 注册定时任务")
    else:
        print(f"    2. 配置 configs/ 下的三个文件")
        print(f"    3. deploy.bat install 注册定时任务")
    if slim:
        print()
        print("  轻量部署说明：")
        print("    - 使用 ONNX Runtime 替代 PyTorch（磁盘节省 ~600MB）")
        print("    - 依赖清单: requirements-server.txt")
        print("    - 最低配置: 2 核 CPU / 4GB RAM / 10GB 磁盘")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="飞书文档自动处理 — 打包部署脚本（编译为 .pyc，不含源码）",
    )
    parser.add_argument(
        "--slim", action="store_true",
        help="轻量部署（ONNX 模型 + requirements-server.txt，磁盘 ~0.5GB vs ~1.3GB）",
    )
    parser.add_argument(
        "--include-models", action="store_true",
        help="包含 Embedding 模型文件（约 100MB，否则目标机首次运行时自动下载）",
    )
    parser.add_argument(
        "--include-venv", action="store_true",
        help="包含虚拟环境（约 1-2GB，目标机无需安装 Python 和依赖）",
    )
    args = parser.parse_args()

    build(
        include_models=args.include_models,
        include_venv=args.include_venv,
        slim=args.slim,
    )

