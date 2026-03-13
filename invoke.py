#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
外部进程调用入口 — 带重试 & 异常捕获

用法：
  python invoke.py                          # 全流程，失败自动重试
  python invoke.py --step download,process  # 指定阶段
  python invoke.py --max-retries 5          # 最多重试5次（默认3次）
  python invoke.py --retry-delay 120        # 重试间隔120秒（默认60秒）

外部进程调用：
  subprocess.run([python_exe, "invoke.py"])
  返回码: 0=成功, 1=重试耗尽仍失败, 2=参数错误

日志输出到 _runtime/logs/invoke.log
"""
import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# ── 路径 ──
SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = SCRIPT_DIR.parent / "_runtime"
LOCK_FILE = RUNTIME_DIR / ".lock"
LOG_DIR = RUNTIME_DIR / "logs"
PYTHON_EXE = SCRIPT_DIR / "venv" / "Scripts" / "python.exe"
RUN_PY = SCRIPT_DIR / "run.py"

# ── 默认参数 ──
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 60       # 秒
DEFAULT_TIMEOUT = 7200         # 2小时


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "invoke.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("invoke")


def clear_lock():
    """清理残留锁文件（进程崩溃后可能遗留）"""
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
            return True
        except OSError:
            return False
    return False


def run_pipeline(args: list, timeout: int) -> tuple[int, str]:
    """
    执行一次管线，返回 (exit_code, error_message)
    exit_code: 0=成功, >0=失败, -1=超时, -2=崩溃
    """
    cmd = [str(PYTHON_EXE), str(RUN_PY)] + args

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode == 0:
            return 0, ""

        # 提取最后几行 stderr 作为错误摘要
        stderr_lines = (result.stderr or "").strip().splitlines()
        error_summary = "\n".join(stderr_lines[-10:]) if stderr_lines else "无 stderr 输出"
        return result.returncode, error_summary

    except subprocess.TimeoutExpired:
        return -1, f"执行超时（>{timeout}s）"

    except Exception as e:
        return -2, f"启动失败: {e}"


def main():
    log = setup_logging()

    # ── 解析自身参数 ──
    max_retries = DEFAULT_MAX_RETRIES
    retry_delay = DEFAULT_RETRY_DELAY
    timeout = DEFAULT_TIMEOUT
    pipeline_args = []

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--max-retries" and i + 1 < len(sys.argv):
            max_retries = int(sys.argv[i + 1])
            i += 2
        elif arg == "--retry-delay" and i + 1 < len(sys.argv):
            retry_delay = int(sys.argv[i + 1])
            i += 2
        elif arg == "--timeout" and i + 1 < len(sys.argv):
            timeout = int(sys.argv[i + 1])
            i += 2
        else:
            # 其余参数传给 run.py
            pipeline_args.append(arg)
            i += 1

    log.info("=" * 50)
    log.info("调度启动  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("  管线参数: %s", pipeline_args or "(全流程)")
    log.info("  最大重试: %d 次", max_retries)
    log.info("  重试间隔: %d 秒", retry_delay)
    log.info("  超时限制: %d 秒", timeout)
    log.info("=" * 50)

    # ── 检查环境 ──
    if not PYTHON_EXE.exists():
        log.error("Python 不存在: %s", PYTHON_EXE)
        sys.exit(2)
    if not RUN_PY.exists():
        log.error("run.py 不存在: %s", RUN_PY)
        sys.exit(2)

    # ── 重试循环 ──
    for attempt in range(1, max_retries + 1):
        log.info("第 %d/%d 次执行...", attempt, max_retries)

        # 重试前清理可能残留的锁文件
        if attempt > 1 and clear_lock():
            log.info("  已清理残留锁文件")

        start_time = time.time()
        exit_code, error_msg = run_pipeline(pipeline_args, timeout)
        elapsed = time.time() - start_time

        if exit_code == 0:
            log.info("✓ 执行成功（耗时 %.0f 秒）", elapsed)
            sys.exit(0)

        # ── 失败处理 ──
        log.warning(
            "✗ 第 %d 次失败（耗时 %.0f 秒，退出码 %d）",
            attempt, elapsed, exit_code,
        )
        if error_msg:
            for line in error_msg.splitlines():
                log.warning("  | %s", line)

        # 超时后强制杀进程
        if exit_code == -1:
            log.warning("  超时，尝试终止残留进程...")
            subprocess.run(
                ["taskkill", "/f", "/im", "python.exe"],
                capture_output=True,
            )

        # 还有重试机会
        if attempt < max_retries:
            log.info("  等待 %d 秒后重试...", retry_delay)
            time.sleep(retry_delay)
        else:
            log.error("✗ 重试 %d 次均失败，放弃", max_retries)
            sys.exit(1)


if __name__ == "__main__":
    main()

