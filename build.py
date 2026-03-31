#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打包部署脚本 — 编译 .py → .pyc 字节码，不含源码

用法（在 auto-doc-process 目录下执行）：
  venv\\Scripts\\python.exe build.py                          # 基础打包
  venv\\Scripts\\python.exe build.py --slim                   # 轻量部署（ONNX 模型 + 精简依赖）
  venv\\Scripts\\python.exe build.py --no-models              # 不包含 Embedding 模型（目标机首次运行自动下载）
  venv\\Scripts\\python.exe build.py --include-venv           # 包含虚拟环境（约 1-2GB，目标机无需再装）

输出：
  ../dist/auto-doc-process/     编译后的部署包（可直接复制到目标电脑）

注意：
  - 目标电脑的 Python 大版本须与编译版本一致（如都是 3.11.x）
  - 如果使用 --include-venv 则无需在目标机安装 Python
  - --slim 模式下自动包含 ONNX 模型，仅打包轻量依赖
"""

import os
import sys
import shutil
import py_compile
import compileall
import argparse
from pathlib import Path


def _force_rmtree(path: Path):
    """强制删除目录树（处理 Windows 长路径 & 只读文件）"""
    import stat
    import subprocess as _sp

    def _on_error(func, fpath, exc_info):
        """权限不足时先去掉只读再重试"""
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)

    try:
        shutil.rmtree(path, onerror=_on_error)
    except OSError:
        # shutil 仍然失败（长路径），用 cmd 的 rmdir 兜底
        _sp.run(
            ["cmd", "/c", "rmdir", "/s", "/q", str(path)],
            capture_output=True,
        )
        if path.exists():
            # 最终手段：UNC 长路径前缀
            import subprocess
            subprocess.run(
                ["powershell", "-Command",
                 f'Remove-Item -LiteralPath "\\\\?\\{path}" -Recurse -Force'],
                capture_output=True,
            )


def build(include_models: bool = True, include_venv: bool = False,
          slim: bool = False, keep_configs: bool = False):
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
    print(f"  保留配置:   {'是' if keep_configs else '否（构建后需手动填写 configs/）'}")
    print("=" * 60)
    print()

    # ─── 1. 清理旧构建 ───────────────────────────────────────
    import time as _build_time
    for attempt in range(3):
        if dist_dir.exists():
            _force_rmtree(dist_dir)
            _build_time.sleep(0.5)  # Windows 文件句柄释放延迟
        if not dist_dir.exists():
            break
        if attempt < 2:
            print("[清理] 删除失败，可能被占用，2 秒后重试...")
            _build_time.sleep(2)
        else:
            raise SystemExit(
                f"[错误] 无法删除 {dist_dir}\n"
                "请关闭占用该目录的程序（资源管理器、IDE、杀毒软件等）后重试。"
            )
    if dist_dir.exists():
        raise SystemExit(f"[错误] 无法删除旧构建目录: {dist_dir}")
    print("[清理] 已删除旧构建目录")

    # ─── 2. 复制项目到 dist ──────────────────────────────────
    ignore_list = [
        "venv", "__pycache__", "*.pyc", "*.pyo",
        "build.py", "dist", ".git", ".gitignore",
        "*.log", ".feishu_export*", "_runtime",
    ]
    if not include_models:
        ignore_list.append("models")

    shutil.copytree(
        src_dir, dist_dir,
        ignore=shutil.ignore_patterns(*ignore_list),
    )
    print("[复制] 项目文件已复制")

    # 删除 tools/ 中的开发专用脚本，保留运行时所需脚本
    tools_dist = dist_dir / "tools"
    if tools_dist.exists():
        for name in ["export_onnx.py"]:
            f = tools_dist / name
            if f.exists():
                f.unlink()
        print("[清理] 已移除开发专用脚本")

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

    # ─── 3. 处理配置（生成 .example，可选删除敏感配置） ────────
    # --keep-configs 时跳过删除，保留源码中的配置（便于与 slg-config 等共用）
    configs_dir = dist_dir / "configs"
    sensitive_configs = ("feishu.yaml", "db_info.yml", "doc_splitter.yaml", "lightrag.yaml")
    for sensitive in sensitive_configs:
        src_cfg = configs_dir / sensitive
        example_cfg = configs_dir / f"{sensitive}.example"
        if src_cfg.exists() and not example_cfg.exists():
            shutil.copy2(src_cfg, example_cfg)
            print(f"[配置] 生成模板: {example_cfg.name}")

    if not keep_configs:
        for sensitive in sensitive_configs:
            f = configs_dir / sensitive
            if f.exists():
                f.unlink()
                print(f"[安全] 删除敏感配置: {sensitive}")
    else:
        print("[配置] 保留 configs/ 中的敏感配置（--keep-configs）")

    # ─── 3.5 移除内部不需要发布的脚本 ──────────────────────
    for name in ("invoke.py",):
        f = dist_dir / name
        if f.exists():
            f.unlink()

    # ─── 3.6 写入版本号 ──────────────────────────────────────
    try:
        # 从 __init__.py 中读取 __version__
        init_file = src_dir / "__init__.py"
        version = "unknown"
        if init_file.exists():
            import re as _re
            text = init_file.read_text(encoding="utf-8")
            m = _re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
            if m:
                version = m.group(1)
        from datetime import datetime as _dt
        build_time = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        version_file = dist_dir / "version.txt"
        version_file.write_text(
            f"version={version}\nbuild_time={build_time}\npython={py_ver}\n",
            encoding="utf-8",
        )
        print(f"[版本] 写入 version.txt: v{version} ({build_time})")
    except Exception as e:
        print(f"[警告] 写入 version.txt 失败: {e}")

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

    # ─── 5. 删除 .py 源码（保留入口和运行时脚本） ──────────
    # run.py: 启动入口; tools/preflight_check.py: deploy.bat 预检脚本
    keep_source_paths = {
        dist_dir / "run.py",
        dist_dir / "tools" / "preflight_check.py",
    }
    deleted = 0
    for py_file in dist_dir.rglob("*.py"):
        if py_file.resolve() in {p.resolve() for p in keep_source_paths}:
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
    req_file = "requirements.txt"
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
        ':: ── Python 版本校验（解析 python --version 输出） ──\n'
        'set PY_VER=\n'
        'for /f "tokens=2 delims= " %%v in (\'python --version 2^>^&1\') do set PY_FULL=%%v\n'
        'for /f "tokens=1,2 delims=." %%a in ("%PY_FULL%") do set PY_VER=%%a.%%b\n'
        'if not "%PY_VER%"=="' + py_ver + '" (\n'
        '    echo [错误] Python 版本不匹配！\n'
        '    echo   需要: Python ' + py_ver + '  ^(.pyc 编译版本^)\n'
        '    echo   当前: Python %PY_VER% ^(%PY_FULL%^)\n'
        '    echo.\n'
        '    echo   请安装 Python ' + py_ver + '.x 后重试\n'
        '    echo   下载: https://www.python.org/downloads/\n'
        '    pause\n'
        '    exit /b 1\n'
        ')\n'
        'echo [Python] 版本 %PY_VER% OK\n'
        '\n'
        'set REUSE_DEPS=0\n'
        'if exist "venv\\Scripts\\python.exe" (\n'
        '    echo [1/4] 检测到已存在的 venv，将重建以修复路径（复用已有依赖）...\n'
        '    if exist "venv_old" rmdir /S /Q venv_old\n'
        '    ren venv venv_old\n'
        '    set REUSE_DEPS=1\n'
        ') else (\n'
        '    echo [1/4] 创建虚拟环境...\n'
        ')\n'
        '\n'
        'if not exist "venv\\Scripts\\python.exe" (\n'
        '    python -m venv venv\n'
        '    if errorlevel 1 (\n'
        '        echo [错误] 创建虚拟环境失败，请确认已安装 Python ' + py_ver + '\n'
        '        pause\n'
        '        exit /b 1\n'
        '    )\n'
        ')\n'
        '\n'
        'if "%REUSE_DEPS%"=="1" (\n'
        '    echo [2/4] 复用已安装的依赖（复制 site-packages）...\n'
        '    if exist "venv_old\\Lib\\site-packages" (\n'
        '        xcopy /E /I /Y /Q "venv_old\\Lib\\site-packages\\*" "venv\\Lib\\site-packages\\" >nul\n'
        '    )\n'
        '    echo [3/4] 清理临时目录...\n'
        '    rmdir /S /Q venv_old\n'
        '    echo [4/4] 校验依赖...\n'
        ') else (\n'
        '    echo [2/4] 安装依赖...\n'
        ')\n'
        '\n'
        'venv\\Scripts\\python.exe -m pip install -r ' + req_file + '\n'
        'if errorlevel 1 (\n'
        '    echo [错误] 依赖安装失败\n'
        '    pause\n'
        '    exit /b 1\n'
        ')\n'
        '\n'
        'echo.\n'
        'echo [完成] 请配置以下文件：\n'
        'echo   configs\\feishu.yaml.example        -^>  feishu.yaml\n'
        'echo   configs\\db_info.yml.example       -^>  db_info.yml\n'
        'echo   configs\\doc_splitter.yaml.example -^>  doc_splitter.yaml\n'
        'echo   configs\\lightrag.yaml.example     -^>  lightrag.yaml\n'
        'echo.\n'
        'echo 最低配置: 2 核 CPU / 4GB RAM / 10GB 磁盘\n'
        'echo 推荐配置: 4 核 CPU / 8GB RAM / 20GB 磁盘\n'
        'echo 系统要求: PostgreSQL 14+ (pgvector), Python ' + py_ver + '\n'
        'echo.\n'
        'echo 部署完成后运行:\n'
        'echo   start.bat                双击手动执行\n'
        'echo   install.bat              注册 Windows 定时任务\n'
        'echo.\n'
        'pause\n',
        encoding="utf-8",
    )
    print("[生成] setup.bat（目标机首次部署脚本）")

    # ─── 8. 生成外层精简 bat（避免中文注释在 cmd 中解析出错）───
    parent_dist = dist_dir.parent  # dist/
    proj_name = dist_dir.name      # auto-doc-process

    # 删除内部 bat（外层替代）
    for bat in ("deploy.bat", "start.bat"):
        p = dist_dir / bat
        if p.exists():
            p.unlink()

    # ── 外层 start.bat：精简转发 ──
    start_lines = [
        '@echo off',
        'chcp 65001 >nul',
        'title FeishuDocSync',
        'cd /d "%~dp0"',
        '',
        ':: 运维命令（不传给 run.py）',
        'if /i "%~1"=="stop" goto :do_stop',
        'if /i "%~1"=="status" goto :do_status',
        'if /i "%~1"=="reset" goto :do_reset',
        'if /i "%~1"=="refresh" goto :do_refresh',
        '',
        ':: 参数校验：非空且不以 -- 开头 → 无效命令',
        'if not "%~1"=="" (',
        '    echo %~1 | findstr /b /c:"--" >nul',
        '    if errorlevel 1 goto :usage',
        ')',
        '',
        f'if not exist "{proj_name}\\venv\\Scripts\\python.exe" (',
        f'    echo [ERROR] venv not found. Run {proj_name}\\setup.bat first.',
        '    pause & exit /b 1',
        ')',
        '',
        f'"{proj_name}\\venv\\Scripts\\python.exe" "{proj_name}\\run.py" %*',
        'set EXIT_CODE=%ERRORLEVEL%',
        'if %EXIT_CODE% EQU 0 (echo [OK] Done) else (echo [FAIL] Exit code: %EXIT_CODE%)',
        'if "%~1"=="" pause',
        'exit /b %EXIT_CODE%',
        '',
        ':do_stop',
        'taskkill /f /im python.exe >nul 2>&1',
        'set LOCK_FILE=%~dp0_runtime\\.lock',
        'if exist "%LOCK_FILE%" del /f "%LOCK_FILE%"',
        'echo [OK] Stopped',
        'exit /b 0',
        '',
        ':do_status',
        'tasklist /fi "imagename eq python.exe" /fo table 2>nul | findstr /i "python"',
        'set LOCK_FILE=%~dp0_runtime\\.lock',
        'if exist "%LOCK_FILE%" (echo Lock: ACTIVE) else (echo Lock: IDLE)',
        'exit /b 0',
        '',
        ':do_reset',
        'taskkill /f /im python.exe >nul 2>&1',
        'set LOCK_FILE=%~dp0_runtime\\.lock',
        'if exist "%LOCK_FILE%" del /f "%LOCK_FILE%"',
        f'"{proj_name}\\venv\\Scripts\\python.exe" "{proj_name}\\run.py" --reset',
        'echo [OK] Reset complete',
        'exit /b 0',
        '',
        ':do_refresh',
        'echo [REFRESH] Refresh DB from processed/ data...',
        'taskkill /f /im python.exe >nul 2>&1',
        'set LOCK_FILE=%~dp0_runtime\\.lock',
        'if exist "%LOCK_FILE%" del /f "%LOCK_FILE%"',
        f'"{proj_name}\\venv\\Scripts\\python.exe" "{proj_name}\\run.py" --reset-db --step store',
        'set EXIT_CODE=%ERRORLEVEL%',
        'if %EXIT_CODE% EQU 0 (echo [OK] DB refreshed) else (echo [FAIL] Exit code: %EXIT_CODE%)',
        'exit /b %EXIT_CODE%',
        '',
        ':usage',
        'echo.',
        'echo  start.bat                        Run full pipeline',
        'echo  start.bat --step download         Download only',
        'echo  start.bat --step process          Process only',
        'echo  start.bat --step store            Store only',
        'echo  start.bat --step graph            Build graph only',
        'echo  start.bat --full                  Full sync (ignore incremental)',
        'echo  start.bat --reset-db              Reset DB and rebuild',
        'echo  start.bat --dry-run               Preview mode',
        'echo  start.bat --no-graph              Skip graph building',
        'echo.',
        'echo  start.bat stop                    Stop processes',
        'echo  start.bat status                  Show status',
        'echo  start.bat reset                   Clean all artifacts',
        'echo  start.bat refresh                 Refresh DB from processed/ data',
        'echo.',
        'exit /b 1',
    ]
    (parent_dist / "start.bat").write_bytes(
        "\r\n".join(start_lines).encode("utf-8")
    )

    # ── 外层 install.bat：双击注册定时任务（自动请求管理员权限） ──
    install_lines = [
        '@echo off',
        'chcp 65001 >nul',
        'setlocal enabledelayedexpansion',
        'title Install FeishuDocSync',
        '',
        ':: ── 自动提权（注册定时任务需要管理员） ──',
        'net session >nul 2>&1',
        'if %ERRORLEVEL% NEQ 0 (',
        '    echo [UAC] Requesting administrator privileges...',
        '    powershell -Command "Start-Process cmd -ArgumentList \'/c \"\"%~f0\"\"\' -Verb RunAs"',
        '    exit /b',
        ')',
        '',
        'cd /d "%~dp0"',
        f'cd /d "%~dp0{proj_name}"',
        '',
        'set TASK_NAME=FeishuDocSync',
        'set RUN_TIME=02:00',
        'set ABS_PYTHON=%CD%\\venv\\Scripts\\python.exe',
        'set ABS_RUN_PY=%CD%\\run.py',
        '',
        'if exist "configs\\feishu.yaml" (',
        '    for /f "usebackq tokens=2" %%a in (`findstr /c:"task_name:" "configs\\feishu.yaml"`) do set "TASK_NAME=%%~a"',
        '    for /f "usebackq tokens=2" %%a in (`findstr /c:"run_time:" "configs\\feishu.yaml"`) do set "RUN_TIME=%%~a"',
        ')',
        '',
        'if not exist "venv\\Scripts\\python.exe" (',
        '    echo [ERROR] venv not found. Run setup.bat first.',
        '    pause & exit /b 1',
        ')',
        '',
        ':: ── Preflight ──',
        'echo.',
        'echo [Preflight] Checking environment...',
        'set CHECK_FAIL=0',
        'for %%f in (feishu.yaml db_info.yml doc_splitter.yaml) do (',
        '    if not exist "configs\\%%f" (',
        '        echo   [FAIL] Missing: configs\\%%f',
        '        set CHECK_FAIL=1',
        '    )',
        ')',
        'if "!CHECK_FAIL!"=="0" (',
        '    "venv\\Scripts\\python.exe" "tools\\preflight_check.py" "configs"',
        '    if errorlevel 1 set CHECK_FAIL=1',
        ')',
        'if "!CHECK_FAIL!"=="1" (',
        '    echo [FAIL] Preflight failed. Fix errors above.',
        '    pause & exit /b 1',
        ')',
        'echo [Preflight] OK',
        '',
        ':: ── 首次全量同步（若无历史数据） ──',
        'set NEED_INIT=0',
        'if not exist "..\\..\\processed" set NEED_INIT=1',
        'if "!NEED_INIT!"=="0" (',
        '    dir /b "..\\..\\processed\\" 2>nul | findstr /r "." >nul 2>&1',
        '    if errorlevel 1 set NEED_INIT=1',
        ')',
        'if "!NEED_INIT!"=="1" (',
        '    echo.',
        '    echo [Init] First run - executing full sync...',
        '    "venv\\Scripts\\python.exe" "run.py" --full',
        '    if errorlevel 1 (',
        '        echo [FAIL] Initial sync failed.',
        '        pause & exit /b 1',
        '    )',
        ')',
        '',
        ':: ── 注册定时任务 ──',
        'schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1',
        'schtasks /Create /TN "%TASK_NAME%" /TR "\\"%ABS_PYTHON%\\" \\"%ABS_RUN_PY%\\"" /SC DAILY /ST %RUN_TIME% /RL HIGHEST /F',
        'if %ERRORLEVEL% EQU 0 (',
        '    echo.',
        '    echo ====================================================',
        '    echo   [OK] Scheduled task installed successfully!',
        '    echo   Task:  %TASK_NAME%',
        '    echo   Time:  Daily at %RUN_TIME%',
        '    echo ====================================================',
        ') else (',
        '    echo [FAIL] Failed to register scheduled task.',
        ')',
        'echo.',
        'pause',
        'endlocal',
    ]
    (parent_dist / "install.bat").write_bytes(
        "\r\n".join(install_lines).encode("utf-8")
    )

    # ── 外层 uninstall.bat：双击移除定时任务（自动请求管理员权限） ──
    uninstall_lines = [
        '@echo off',
        'chcp 65001 >nul',
        'setlocal enabledelayedexpansion',
        'title Uninstall FeishuDocSync',
        '',
        ':: ── 自动提权 ──',
        'net session >nul 2>&1',
        'if %ERRORLEVEL% NEQ 0 (',
        '    echo [UAC] Requesting administrator privileges...',
        '    powershell -Command "Start-Process cmd -ArgumentList \'/c \"\"%~f0\"\"\' -Verb RunAs"',
        '    exit /b',
        ')',
        '',
        f'cd /d "%~dp0{proj_name}"',
        '',
        'set TASK_NAME=FeishuDocSync',
        'if exist "configs\\feishu.yaml" (',
        '    for /f "usebackq tokens=2" %%a in (`findstr /c:"task_name:" "configs\\feishu.yaml"`) do set "TASK_NAME=%%~a"',
        ')',
        '',
        ':: ── 停止正在运行的任务 ──',
        'schtasks /End /TN "%TASK_NAME%" >nul 2>&1',
        '',
        ':: ── 终止可能残留的 Python 进程 ──',
        f'for /f "tokens=2" %%i in (\'wmic process where "commandline like \'\'%%run.py%%\'\' and commandline like \'\'%%{proj_name}%%\'\'" get processid 2^>nul ^| findstr /r "[0-9]"\') do (',
        '    echo [Stop] Terminating PID: %%i',
        '    taskkill /PID %%i /F >nul 2>&1',
        ')',
        '',
        ':: ── 删除定时任务 ──',
        'schtasks /Delete /TN "%TASK_NAME%" /F',
        'if %ERRORLEVEL% EQU 0 (',
        '    echo.',
        '    echo ====================================================',
        '    echo   [OK] Scheduled task "%TASK_NAME%" removed.',
        '    echo ====================================================',
        ') else (',
        '    echo [INFO] Task "%TASK_NAME%" not found ^(already removed^).',
        ')',
        '',
        ':: ── 清理锁文件 ──',
        'set LOCK_FILE=%~dp0_runtime\\.lock',
        'if exist "%LOCK_FILE%" del /f "%LOCK_FILE%"',
        '',
        'echo.',
        'pause',
        'endlocal',
    ]
    (parent_dist / "uninstall.bat").write_bytes(
        "\r\n".join(uninstall_lines).encode("utf-8")
    )

    print(f"[生成] 外层 install.bat / uninstall.bat / start.bat（{parent_dist}）")

    # ─── 9. 统计 ─────────────────────────────────────────────
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
    print(f"    1. 在目标位置创建一个根目录，将 {dist_root.name}/ 放入其中")
    if not include_venv:
        print(f"    2. 进入 {dist_root.name}/{dist_dir.name}/ 双击 setup.bat 创建虚拟环境")
        print(f"    3. 配置 {dist_root.name}/{dist_dir.name}/configs/ 下的配置文件")
        print(f"    4. 进入 {dist_root.name}/ 双击 install.bat 注册定时任务")
    else:
        print(f"    2. 配置 {dist_root.name}/{dist_dir.name}/configs/ 下的配置文件")
        print(f"    3. 进入 {dist_root.name}/ 双击 install.bat 注册定时任务")
    print()
    print("  目录结构：")
    print(f"    根目录/")
    print(f"    ├── {dist_root.name}/                   ← 程序打包目录")
    print(f"    │   ├── install.bat             ← 双击注册定时任务")
    print(f"    │   ├── uninstall.bat           ← 双击移除定时任务")
    print(f"    │   ├── start.bat               ← 手动执行 / 运维")
    print(f"    │   └── {dist_dir.name}/")
    print(f"    │       ├── configs/            ← 配置文件")
    print(f"    │       ├── setup.bat           ← 首次部署")
    print(f"    │       └── run.py              ← 程序入口")
    print(f"    ├── documents/              ← 飞书文档（运行后生成）")
    print(f"    └── processed/              ← 处理产物（运行后生成）")
    if slim:
        print()
        print("  轻量部署说明：")
        print("    - 使用 ONNX Runtime 替代 PyTorch（磁盘节省 ~600MB）")
        print("    - 依赖清单: requirements.txt（默认轻量模式）")
        print("    - 最低配置: 2 核 CPU / 4GB RAM / 10GB 磁盘")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="飞书文档自动处理 — 打包部署脚本（编译为 .pyc，不含源码）",
    )
    parser.add_argument(
        "--slim", action="store_true",
        help="轻量部署（仅包含 ONNX 模型，不含 HuggingFace 缓存）",
    )
    parser.set_defaults(include_models=True)
    parser.add_argument(
        "--include-models", dest="include_models", action="store_true",
        help="包含 Embedding 模型文件（默认开启）",
    )
    parser.add_argument(
        "--no-models", dest="include_models", action="store_false",
        help="不包含 Embedding 模型文件（目标机首次运行时自动下载）",
    )
    parser.add_argument(
        "--include-venv", action="store_true",
        help="包含虚拟环境（约 1-2GB，目标机无需安装 Python 和依赖）",
    )
    parser.add_argument(
        "--keep-configs", action="store_true",
        help="保留 configs/ 中的敏感配置（与 slg-config 等共用时无需重复填写）",
    )
    args = parser.parse_args()

    build(
        include_models=args.include_models,
        include_venv=args.include_venv,
        slim=args.slim,
        keep_configs=args.keep_configs,
    )

