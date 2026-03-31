@echo off
chcp 65001 >nul
title FeishuDocSync

REM ══════════════════════════════════════════════
REM  飞书文档自动处理管线
REM ══════════════════════════════════════════════
REM
REM 全流程（默认）:
REM   start.bat                              download -> process -> store -> graph
REM
REM 单阶段:
REM   start.bat --step download              只下载
REM   start.bat --step process               只处理（拆分+向量化）
REM   start.bat --step store                 只入库
REM   start.bat --step graph                 只构建图谱
REM
REM 组合:
REM   start.bat --step download,process      下载+处理
REM   start.bat --step download,process,store  下载+处理+入库（不建图）
REM
REM 其他:
REM   start.bat --dry-run                    预览（只列文档不执行）
REM   start.bat --full                       全量同步（忽略增量清单）
REM   start.bat --reset-db                   清空数据库后全量重建
REM   start.bat --no-graph                   跳过图谱构建
REM
REM 运维:
REM   start.bat stop                         停止运行中的进程并清理锁文件
REM   start.bat status                       查看进程状态
REM   start.bat reset                        清理所有构建产物（数据库+缓存+清单），不重建
REM   start.bat refresh                      刷新数据库（清空后用 processed/ 数据重建）

cd /d "%~dp0.."

REM ── 处理 stop / status / reset 命令 ──
if /i "%~1"=="stop" goto :do_stop
if /i "%~1"=="status" goto :do_status
if /i "%~1"=="reset" goto :do_reset
if /i "%~1"=="refresh" goto :do_refresh

REM ── 正常启动 ──
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


REM ══════════════════════════════════════════════
REM  stop — 停止进程 + 清理锁文件
REM ══════════════════════════════════════════════
:do_stop
echo [STOP] 正在停止 FeishuDocSync 进程...

REM 查找并终止 run.py 相关进程
set FOUND=0
for /f "tokens=2" %%p in ('tasklist /fi "imagename eq python.exe" /fo csv /nh 2^>nul ^| findstr /i "python"') do (
    set FOUND=1
)

taskkill /f /im python.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Python 进程已终止
) else (
    echo   [--] 没有运行中的 Python 进程
)

REM 清理锁文件
set LOCK_FILE=%~dp0..\_runtime\.lock
if exist "%LOCK_FILE%" (
    del /f "%LOCK_FILE%"
    echo   [OK] 锁文件已清除: %LOCK_FILE%
) else (
    echo   [--] 无锁文件
)

echo [STOP] 完成
exit /b 0


REM ══════════════════════════════════════════════
REM  status — 查看进程状态
REM ══════════════════════════════════════════════
:do_status
echo [STATUS] FeishuDocSync 进程状态:
echo.

tasklist /fi "imagename eq python.exe" /fo table 2>nul | findstr /i "python" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Python 进程:
    tasklist /fi "imagename eq python.exe" /fo table 2>nul | findstr /i "python"
) else (
    echo   没有运行中的 Python 进程
)

echo.
set LOCK_FILE=%~dp0..\_runtime\.lock
if exist "%LOCK_FILE%" (
    echo   锁文件: 存在 ^(有任务占用^)
) else (
    echo   锁文件: 无 ^(空闲^)
)

echo.
exit /b 0


REM ══════════════════════════════════════════════
REM  reset — 清理所有构建产物（只清理不重建）
REM ══════════════════════════════════════════════
:do_reset
echo [RESET] 清理所有构建产物...
echo.

REM 先停止进程
taskkill /f /im python.exe >nul 2>&1

REM 清理锁文件
set LOCK_FILE=%~dp0..\_runtime\.lock
if exist "%LOCK_FILE%" del /f "%LOCK_FILE%"

REM 调用 Python 的 --reset 逻辑（清数据库+缓存+清单）
"auto-doc-process\venv\Scripts\python.exe" "auto-doc-process\run.py" --reset

echo.
echo [RESET] 完成。下次运行将从零开始。
exit /b 0


REM ══════════════════════════════════════════════
REM  refresh — 刷新数据库（清空后用 processed/ 重建）
REM ══════════════════════════════════════════════
:do_refresh
echo [REFRESH] 刷新数据库（清空表 → 用 processed/ 数据重建）...
echo.

REM 先停止进程
taskkill /f /im python.exe >nul 2>&1

REM 清理锁文件
set LOCK_FILE=%~dp0..\_runtime\.lock
if exist "%LOCK_FILE%" del /f "%LOCK_FILE%"

REM 清空数据库 + 从 processed/ 重新入库（不重建图谱，图谱需要调 LLM API）
"auto-doc-process\venv\Scripts\python.exe" "auto-doc-process\run.py" --reset-db --step store

set EXIT_CODE=%ERRORLEVEL%
echo.
if %EXIT_CODE% EQU 0 (
    echo [REFRESH] 数据库刷新完成
) else (
    echo [REFRESH] 刷新失败，退出码: %EXIT_CODE%
)
exit /b %EXIT_CODE%
