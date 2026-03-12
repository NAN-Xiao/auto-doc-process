#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
内置定时调度守护进程

以固定间隔循环执行飞书文档同步管线，适用于：
  - 无法配置 Windows 任务计划程序的环境
  - 开发调试阶段需要持续运行的场景

生产部署推荐使用 deploy.bat 注册 Windows 任务计划程序。

用法（在 doctment 目录或 auto-doc-process 目录下执行）：
  venv\\Scripts\\python.exe scheduler.py              # 使用默认配置（feishu.yaml 中的 schedule 节）
  venv\\Scripts\\python.exe scheduler.py --interval 60  # 覆盖为 60 分钟间隔
  venv\\Scripts\\python.exe scheduler.py --no-graph     # 跳过图谱构建
"""

import importlib
import os
import sys
import time
import signal
import logging
from datetime import datetime, timedelta

# ─── 环境准备 ───────────────────────────────────────────────
project_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_parent)
if project_parent not in sys.path:
    sys.path.insert(0, project_parent)

PKG = "auto-doc-process"

# 导入包
importlib.import_module(PKG)

# 导入核心模块
config_mod = importlib.import_module(f"{PKG}.core.config")
main_mod = importlib.import_module(f"{PKG}.__main__")

log = config_mod.log

# ─── 优雅退出 ───────────────────────────────────────────────
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    log.info("收到退出信号，等待当前任务完成后退出...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── 调度主循环 ─────────────────────────────────────────────

def run_scheduler(
    interval_minutes: int = 120,
    run_on_start: bool = True,
    build_graph: bool = True,
    full: bool = False,
):
    """
    定时调度主循环

    Args:
        interval_minutes: 执行间隔（分钟）
        run_on_start: 是否启动后立即执行一次
        build_graph: 是否构建图谱
        full: 是否全量同步
    """
    log.info("=" * 60)
    log.info("飞书文档同步 - 定时调度守护进程")
    log.info(f"  执行间隔: {interval_minutes} 分钟")
    log.info(f"  启动立即执行: {'是' if run_on_start else '否'}")
    log.info(f"  图谱构建: {'启用' if build_graph else '跳过'}")
    log.info(f"  同步模式: {'全量' if full else '增量'}")
    log.info(f"  PID: {os.getpid()}")
    log.info("=" * 60)
    log.info("按 Ctrl+C 优雅退出")

    run_count = 0

    if not run_on_start:
        next_run = datetime.now() + timedelta(minutes=interval_minutes)
        log.info(f"首次执行时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        _wait_until(next_run)

    while not _shutdown:
        run_count += 1
        log.info("")
        log.info(f"{'─' * 40} 第 {run_count} 次执行 {'─' * 40}")

        start_time = datetime.now()
        try:
            config = config_mod.load_full_config()
            log_level = config.get("log_level", "INFO")
            config_mod.setup_logging(log_level, config.get("log_dir", ""))

            main_mod.run_sync(
                config,
                full=full,
                dry_run=False,
                build_graph=build_graph,
            )
            log.info(f"本次执行完成，耗时 {(datetime.now() - start_time).total_seconds():.1f} 秒")
        except SystemExit:
            # run_sync 内部 sys.exit() 不退出调度器
            log.warning("管线执行异常退出，调度器继续运行")
        except Exception as e:
            log.error(f"执行异常: {e}")
            import traceback
            log.error(traceback.format_exc())

        if _shutdown:
            break

        next_run = datetime.now() + timedelta(minutes=interval_minutes)
        log.info(f"下次执行时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        _wait_until(next_run)

    log.info("调度器已退出")


def _wait_until(target: datetime):
    """等待到目标时间，每 5 秒检查一次退出信号"""
    while not _shutdown and datetime.now() < target:
        remaining = (target - datetime.now()).total_seconds()
        time.sleep(min(5, max(0, remaining)))


# ─── 入口 ──────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="飞书文档同步 - 内置定时调度守护进程",
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="执行间隔（分钟），覆盖配置文件中的值",
    )
    parser.add_argument(
        "--no-graph", action="store_true",
        help="跳过图谱构建步骤",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="全量同步（忽略增量清单）",
    )
    parser.add_argument(
        "--no-run-on-start", action="store_true",
        help="启动后不立即执行，等到下个周期再执行",
    )

    args = parser.parse_args()

    # 读配置
    feishu_cfg = config_mod.load_feishu_config()
    schedule_cfg = feishu_cfg.get("schedule", {})

    interval = args.interval or schedule_cfg.get("interval_minutes", 120)
    run_on_start = not args.no_run_on_start and schedule_cfg.get("run_on_start", True)

    # 初始化日志
    log_cfg = feishu_cfg.get("log", {})
    config_mod.setup_logging(
        log_cfg.get("level", "INFO"),
        log_cfg.get("dir", ""),
    )

    run_scheduler(
        interval_minutes=interval,
        run_on_start=run_on_start,
        build_graph=not args.no_graph,
        full=args.full,
    )


if __name__ == "__main__":
    main()

