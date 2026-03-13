@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
:: 飞书文档自动处理 — Windows 任务计划程序 管理工具
::
:: 用法（以管理员身份运行）：
::   deploy.bat install     注册定时任务
::   deploy.bat uninstall   删除定时任务
::   deploy.bat stop        暂停定时 + 终止正在运行的任务
::   deploy.bat start       恢复定时
::   deploy.bat run         立即执行一次
::   deploy.bat status      查看任务状态
:: ============================================================

:: ─── 默认参数 ────────────────────────────────────────────
set TASK_NAME=FeishuDocSync
set RUN_TIME=02:00
:: ─── 结束默认参数 ────────────────────────────────────────

:: 切换到脚本所在目录，后续全部使用相对路径
cd /d "%~dp0"

:: 绝对路径（仅 schtasks /TR 需要，定时任务必须用绝对路径）
set ABS_PYTHON=%CD%\venv\Scripts\python.exe
set ABS_RUN_PY=%CD%\run.py

:: 从 feishu.yaml 读取 run_time（覆盖默认值）
if exist "configs\feishu.yaml" (
    for /f "usebackq tokens=2" %%a in (`findstr /c:"run_time:" "configs\feishu.yaml"`) do set "RUN_TIME=%%~a"
)

:: 检查虚拟环境（install/run 时需要）
if /i "%~1"=="install" goto :check_venv
if /i "%~1"=="run" goto :check_venv
goto :skip_venv_check

:check_venv
if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境: venv\Scripts\python.exe
    echo        请先运行: python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
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

:: ─── 环境预检 ─────────────────────────────────────────────
:preflight
echo.
echo [预检] 检查配置文件和环境...
set CHECK_FAIL=0

:: 1) 配置文件
for %%f in (feishu.yaml db_info.yml doc_splitter.yaml) do (
    if not exist "configs\%%f" (
        echo   [FAIL] 配置文件缺失: configs\%%f
        echo          请从 configs\%%f.example 复制并填写
        set CHECK_FAIL=1
    ) else (
        echo   [ OK ] configs\%%f
    )
)

:: 2) lightrag.yaml (optional)
if not exist "configs\lightrag.yaml" (
    echo   [WARN] configs\lightrag.yaml 不存在（图谱构建将使用默认参数）
) else (
    echo   [ OK ] configs\lightrag.yaml
)

:: 3) ONNX model
if exist "models\onnx\model.onnx" (
    echo   [ OK ] models\onnx\model.onnx
) else (
    echo   [WARN] ONNX 模型不存在（models\onnx\model.onnx），将尝试 HuggingFace 回退
)

:: 4) DB connection + auto-create database/pgvector + Feishu API check
if "!CHECK_FAIL!"=="0" (
    echo.
    echo [预检] 测试数据库连接和 API 配置...
    "venv\Scripts\python.exe" "tools\preflight_check.py" "configs"
    if errorlevel 3 (
        echo.
        echo          feishu.yaml 中 app_id / app_secret 未配置或仍是模板值
        set CHECK_FAIL=1
    ) else if errorlevel 2 (
        echo.
        echo          pgvector 扩展安装失败，请手动安装后重试
        set CHECK_FAIL=1
    ) else if errorlevel 1 (
        echo.
        echo          请检查:
        echo            - PostgreSQL 服务是否启动
        echo            - db_info.yml 中的连接信息是否正确
        set CHECK_FAIL=1
    )
)

echo.
if "!CHECK_FAIL!"=="1" (
    echo ====================================================
    echo  [预检失败] 请修复以上错误后重新运行 deploy.bat
    echo ====================================================
    goto :end
)
echo [预检] 全部通过
echo.
goto :eof

:: ─── 安装 ──────────────────────────────────────────────────
:install
echo.
echo ====================================================
echo  注册 Windows 任务计划程序
echo ====================================================
echo  任务名称: %TASK_NAME%
echo  执行时间: 每天 %RUN_TIME%
echo  工作目录: %CD%
echo  Python:   venv\Scripts\python.exe
echo  脚本:     run.py
echo ====================================================
echo.

:: 执行环境预检
call :preflight
if "!CHECK_FAIL!"=="1" goto :end

:: 检查是否已有构建产物
set NEED_INIT=0
if not exist "..\processed" set NEED_INIT=1
if "!NEED_INIT!"=="0" (
    dir /b "..\processed\" 2>nul | findstr /r "." >nul 2>&1
    if errorlevel 1 set NEED_INIT=1
)

if "!NEED_INIT!"=="1" (
    echo [初始化] 未检测到已构建数据，先执行 dry-run 验证配置联通性...
    echo.
    "venv\Scripts\python.exe" "run.py" --dry-run
    if errorlevel 1 (
        echo.
        echo ====================================================
        echo  [初始化失败] dry-run 验证未通过，定时任务不会注册
        echo ====================================================
        echo  请检查:
        echo    1. configs\feishu.yaml  — app_id / app_secret 是否正确
        echo    2. configs\db_info.yml  — 数据库连接是否正常
        echo    3. _runtime\logs\       — 查看详细日志
        echo.
        echo  修复后重新运行: deploy.bat install
        goto :end
    )
    echo [初始化] 配置验证通过，执行首次全量同步...
    echo.
    "venv\Scripts\python.exe" "run.py" --full
    if errorlevel 1 (
        echo.
        echo [警告] 首次全量同步失败（退出码: !ERRORLEVEL!），定时任务仍将注册
        echo        请检查日志后手动运行: deploy.bat run
        echo.
    ) else (
        echo.
        echo [初始化] 首次全量同步完成
        echo.
    )
)

schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

schtasks /Create ^
    /TN "%TASK_NAME%" ^
    /TR "\"%ABS_PYTHON%\" \"%ABS_RUN_PY%\"" ^
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
:: 执行前也预检
call :preflight
if "!CHECK_FAIL!"=="1" goto :end
echo.
echo [执行] 立即运行一次...
schtasks /Run /TN "%TASK_NAME%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [成功] 任务已触发
    echo        查看日志: ..\\_runtime\logs\
) else (
    echo [提示] 通过任务计划触发失败，直接执行...
    "venv\Scripts\python.exe" "run.py"
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
echo ─── 版本信息 ───
if exist "version.txt" (
    type "version.txt"
) else (
    echo   版本信息不可用（开发模式或未打包）
)
echo.
echo ─── 定时任务状态 ───
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
