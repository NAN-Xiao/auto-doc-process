@echo off
chcp 65001 >nul
title FeishuDocSync

REM Feishu document auto-processing pipeline
REM
REM Full pipeline (default):
REM   start.bat                              download -> process -> store -> graph
REM
REM Single stage:
REM   start.bat --step download              download only (build manifest)
REM   start.bat --step process               split + embed only
REM   start.bat --step store                 batch store to pgvector only
REM   start.bat --step graph                 build LightRAG graph only
REM
REM Combined stages:
REM   start.bat --step download,process      download + process (no store, no graph)
REM   start.bat --step download,process,store  download + process + store (no graph)
REM   start.bat --step process,store         process + store
REM
REM Other options:
REM   start.bat --dry-run                    preview only (list documents)
REM   start.bat --full                       full sync (ignore manifest)
REM   start.bat --reset-db                   clear DB then full rebuild (implies --full)
REM   start.bat --no-graph                   skip graph building
REM   start.bat --schedule                   start daemon scheduler

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
