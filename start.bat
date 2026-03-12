@echo off
chcp 65001 >nul
title FeishuDocSync

REM Feishu document auto-processing pipeline - one-click start
REM Usage:
REM   start.bat              full pipeline (incremental)
REM   start.bat --dry-run    preview only
REM   start.bat --full       full sync
REM   start.bat --no-graph   skip graph building
REM   start.bat --schedule   start daemon scheduler

cd /d "%~dp0.."

if not exist "auto-doc-process\venv\Scripts\python.exe" (
    echo [ERROR] venv not found: auto-doc-process\venv\
    echo         Run: cd auto-doc-process ^&^& python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo ==================================================
echo  FeishuDocSync  %date% %time%
echo ==================================================
echo.

"auto-doc-process\venv\Scripts\python.exe" "auto-doc-process\run.py" %*

set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
    echo [OK] Done
) else (
    echo [FAIL] Exit code: %EXIT_CODE%
)

if "%~1"=="" pause

exit /b %EXIT_CODE%
