@echo off
chcp 65001 >nul
title 飞书文档自动处理管线

:: ============================================================
:: 飞书文档自动处理 - 一键启动
::
:: 双击即执行全流程：
::   拉取飞书文档 → 拆分/向量化/存库 → 重建图谱 → 导出到 PG
::
:: 调试参数（可选，通过命令行传入）：
::   start.bat --dry-run       预览模式，只列文档不下载
::   start.bat --full          全量同步，忽略增量清单
::   start.bat --no-graph      跳过图谱构建
:: ============================================================

:: 定位到脚本所在目录的上级（doctment/）
cd /d "%~dp0.."

:: 检查虚拟环境是否存在
if not exist "auto-doc-process\venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境: auto-doc-process\venv\
    echo        请先运行: cd auto-doc-process ^&^& python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: 执行管线
echo ──────────────────────────────────────────────────
echo  飞书文档自动处理管线
echo  时间: %date% %time%
echo ──────────────────────────────────────────────────
echo.

"auto-doc-process\venv\Scripts\python.exe" -m auto-doc-process %*

:: 记录退出码
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
    echo [完成] 任务执行成功
) else (
    echo [异常] 退出码: %EXIT_CODE%
)

:: 双击运行时暂停，命令行调用时不暂停
if "%~1"=="" pause

exit /b %EXIT_CODE%
