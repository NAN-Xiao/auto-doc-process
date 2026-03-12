@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
:: 飞书文档自动处理 — Windows 任务计划程序 管理工具
::
:: 用法（以管理员身份运行）：
::   deploy.bat install     注册定时任务（每天凌晨 2:00 执行）
::   deploy.bat uninstall   删除定时任务
::   deploy.bat stop        暂停定时 + 终止正在运行的任务（用于更新版本）
::   deploy.bat start       恢复定时
::   deploy.bat run         立即执行一次
::   deploy.bat status      查看任务状态
:: ============================================================

:: ─── 可修改参数 ────────────────────────────────────────────
set TASK_NAME=FeishuDocSync
set RUN_TIME=02:00
:: ─── 结束可修改参数 ────────────────────────────────────────

:: 定位路径
set SCRIPT_DIR=%~dp0
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

set PARENT_DIR=%SCRIPT_DIR%\..
pushd "%PARENT_DIR%"
set WORK_DIR=%CD%
popd

set PYTHON_EXE=%SCRIPT_DIR%\venv\Scripts\python.exe
set RUN_PY=%SCRIPT_DIR%\run.py

:: 检查虚拟环境（install/run 时需要）
if /i "%~1"=="install" goto :check_venv
if /i "%~1"=="run" goto :check_venv
goto :skip_venv_check

:check_venv
if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到虚拟环境: %PYTHON_EXE%
    echo        请先创建: cd "%SCRIPT_DIR%" ^&^& python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
    goto :end
)
:skip_venv_check

:: 检查参数
if "%~1"=="" goto :usage
if /i "%~1"=="install" goto :install
if /i "%~1"=="uninstall" goto :uninstall
if /i "%~1"=="stop" goto :stop
if /i "%~1"=="start" goto :start_task
if /i "%~1"=="run" goto :run_now
if /i "%~1"=="status" goto :status
goto :usage

:: ─── 安装 ──────────────────────────────────────────────────
:install
echo.
echo ====================================================
echo  注册 Windows 任务计划程序
echo ====================================================
echo  任务名称: %TASK_NAME%
echo  执行时间: 每天 %RUN_TIME%
echo  工作目录: %WORK_DIR%
echo  Python:   %PYTHON_EXE%
echo  脚本:     %RUN_PY%
echo ====================================================
echo.

schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

schtasks /Create ^
    /TN "%TASK_NAME%" ^
    /TR "\"%PYTHON_EXE%\" \"%RUN_PY%\"" ^
    /SC DAILY ^
    /ST %RUN_TIME% ^
    /RL HIGHEST ^
    /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [成功] 定时任务 "%TASK_NAME%" 已注册
    echo        每天 %RUN_TIME% 执行
) else (
    echo [失败] 注册失败，请以管理员身份运行此脚本
)
goto :end

:: ─── 停止（暂停定时 + 终止当前运行） ────────────────────────
:stop
echo.
echo [停止] 暂停定时任务...
schtasks /Change /TN "%TASK_NAME%" /DISABLE >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [停止] 定时已暂停
) else (
    echo [提示] 任务不存在或已暂停
)

echo [停止] 终止正在运行的任务...
schtasks /End /TN "%TASK_NAME%" >nul 2>&1

:: 也终止可能残留的 Python 进程（仅终止本项目的）
for /f "tokens=2" %%i in ('wmic process where "commandline like '%%run.py%%' and commandline like '%%auto-doc-process%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo [停止] 终止进程 PID: %%i
    taskkill /PID %%i /F >nul 2>&1
)

echo.
echo [完成] 任务已停止，可以安全更新版本
echo        更新后运行: deploy.bat start
goto :end

:: ─── 恢复 ──────────────────────────────────────────────────
:start_task
echo.
schtasks /Change /TN "%TASK_NAME%" /ENABLE >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [成功] 定时任务已恢复运行
) else (
    echo [失败] 恢复失败（任务可能不存在，请先 deploy.bat install）
)
goto :end

:: ─── 立即执行一次 ──────────────────────────────────────────
:run_now
echo.
echo [执行] 立即运行一次...
schtasks /Run /TN "%TASK_NAME%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [成功] 任务已触发
    echo        查看日志: logs\feishu_export_*.log
) else (
    echo [提示] 通过任务计划触发失败，直接执行...
    "%PYTHON_EXE%" "%RUN_PY%"
)
goto :end

:: ─── 卸载 ──────────────────────────────────────────────────
:uninstall
echo.
schtasks /End /TN "%TASK_NAME%" >nul 2>&1
schtasks /Delete /TN "%TASK_NAME%" /F
if %ERRORLEVEL% EQU 0 (
    echo [成功] 定时任务 "%TASK_NAME%" 已删除
) else (
    echo [失败] 删除失败（任务可能不存在）
)
goto :end

:: ─── 状态 ──────────────────────────────────────────────────
:status
echo.
schtasks /Query /TN "%TASK_NAME%" /V /FO LIST 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [提示] 任务 "%TASK_NAME%" 不存在，请先运行 deploy.bat install
)
goto :end

:: ─── 帮助 ──────────────────────────────────────────────────
:usage
echo.
echo ====================================================
echo  飞书文档自动处理 — 定时任务管理工具
echo ====================================================
echo.
echo  deploy.bat install     注册定时任务（每天 %RUN_TIME%）
echo  deploy.bat stop        停止任务（更新版本前执行）
echo  deploy.bat start       恢复任务（更新版本后执行）
echo  deploy.bat run         立即执行一次
echo  deploy.bat status      查看任务状态
echo  deploy.bat uninstall   彻底删除定时任务
echo.
echo  版本更新流程：
echo    1. deploy.bat stop        停止
echo    2. 覆盖新版本文件
echo    3. deploy.bat start       恢复
echo    4. deploy.bat run         验证一次
echo.

:: ─── 结束 ──────────────────────────────────────────────────
:end
echo.
pause
endlocal
