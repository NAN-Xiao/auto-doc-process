#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logger 适配层

提供 Logger 静态类，供 processor/ 模块使用。
底层复用 feishu_sync logger，保持日志统一输出。

用法:
    from ..core.logger import Logger

    Logger.info("hello")
    Logger.info("indented", indent=1)
    Logger.success("done")
    Logger.separator()
"""

import logging


def _get_logger() -> logging.Logger:
    """获取 feishu_sync logger（如果尚未配置则回退到基础 logger）"""
    logger = logging.getLogger("feishu_sync")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(ch)
    return logger


class Logger:
    """静态 Logger — 代替 print 的统一日志接口"""

    @staticmethod
    def _indent(msg: str, indent: int = 0) -> str:
        return ("  " * indent) + msg if indent else msg

    @staticmethod
    def info(msg: str, indent: int = 0):
        _get_logger().info(Logger._indent(msg, indent))

    @staticmethod
    def error(msg: str, indent: int = 0):
        _get_logger().error(Logger._indent(msg, indent))

    @staticmethod
    def warning(msg: str, indent: int = 0):
        _get_logger().warning(Logger._indent(msg, indent))

    @staticmethod
    def success(msg: str, indent: int = 0):
        _get_logger().info(Logger._indent(f"[OK] {msg}", indent))

    @staticmethod
    def separator(char: str = "=", length: int = 50):
        _get_logger().info(char * length)

