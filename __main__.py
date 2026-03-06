#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
飞书文档同步 - 入口

用法（由 Windows 任务计划程序定时调用）：
  python -m auto-doc                          → 自动发现并同步所有文档
  python -m auto-doc --space-id 761373590378  → 只同步指定知识空间
  python -m auto-doc --full                   → 全量同步（忽略增量清单）
  python -m auto-doc --dry-run                → 预览模式，只列出不下载
  python -m auto-doc --config path/to/cfg.yaml
"""

import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path

from .core.config import load_full_config, log, setup_logging
from .feishu.exporter import discover_documents, batch_export
from .feishu.api import create_lark_client
from .core.utils import acquire_lock, release_lock


def run_sync(config: dict, space_id: str = None,
             full: bool = False, dry_run: bool = False):
    """
    执行一次飞书文档同步

    Args:
        config: 完整配置
        space_id: 指定知识空间（None = 全部）
        full: 全量模式（忽略增量清单）
        dry_run: 预览模式（只列出不下载）
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("=" * 50)
    log.info(f"飞书文档同步  {now}")
    log.info("=" * 50)

    lock_path = config.get("lock_path")

    if not dry_run:
        if not acquire_lock(lock_path):
            log.warning("另一个同步任务正在运行，跳过")
            return

    try:
        feishu_config = {
            "app_id": config["app_id"],
            "app_secret": config["app_secret"],
            "base_url": config["base_url"],
            "type_format_map": config.get("type_format_map"),
        }

        # 发现文档
        space_ids = [space_id] if space_id else config.get("space_ids", [])
        if space_ids:
            all_entries = []
            for sid in space_ids:
                entries = discover_documents(feishu_config, space_id=sid)
                all_entries.extend(entries)
        else:
            all_entries = discover_documents(feishu_config)

        if not all_entries:
            log.info("没有发现可导出的文档")
            return

        log.info(f"发现 {len(all_entries)} 个文档")
        for i, e in enumerate(all_entries, 1):
            name_part = f" [{e['name']}]" if e.get("name") else ""
            log.info(f"  {i}. {e['doc_type']:6s} → .{e['ext']:4s}  "
                     f"{e['token'][:16]}...{name_part}")

        if dry_run:
            log.info("预览模式，不执行下载")
            return

        # 创建 Client
        client = create_lark_client(feishu_config)
        if not client:
            log.error("创建 SDK Client 失败")
            sys.exit(1)

        # 输出目录
        out_path = config.get("output_dir", Path("feishu_exports"))

        # 批量导出
        success, fail, skip = batch_export(
            client, config, all_entries, out_path,
            incremental=not full,
        )

        log.info(f"同步完成: 成功={len(success)} 跳过={len(skip)} 失败={len(fail)}")

        # ---- 文档处理：拆分 → 向量化 → 存入 PostgreSQL ----
        if success and config.get("vec_enabled") and config.get("db"):
            log.info("=" * 50)
            log.info("开始文档处理管线：拆分 → 向量化 → pgvector 入库")
            log.info("=" * 50)

            from .processor.workflow import BatchWorkflow
            from .core.config import load_processor_config

            proc_config = load_processor_config()
            use_llm = proc_config.get('doc_splitter', {}).get('image_naming', {}).get('use_llm', False)

            workflow = BatchWorkflow(
                use_llm_naming=use_llm,
                db_config=config["db"],
            )

            # 同一批次共用时间戳
            batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            proc_success = 0
            proc_fail = 0
            for item in success:
                file_path = item.get("path", "")
                if not file_path or not Path(file_path).exists():
                    continue

                doc_path = Path(file_path)
                # 只处理支持的格式
                if doc_path.suffix.lower() not in (".docx", ".pdf", ".xlsx"):
                    log.info(f"  跳过不支持的格式: {doc_path.name}")
                    continue

                result = workflow.process_single_document(doc_path, batch_timestamp=batch_timestamp)
                if result and result.get("success"):
                    proc_success += 1
                else:
                    proc_fail += 1

            log.info(f"文档处理完成: 成功={proc_success} 失败={proc_fail}")

        elif success and not config.get("db"):
            log.info("未配置数据库，跳过文档处理管线")

        if fail:
            sys.exit(1)

    except Exception as e:
        log.error(f"同步异常: {e}")
        traceback.print_exc()
        sys.exit(2)
    finally:
        if not dry_run:
            release_lock(lock_path)


def main():
    parser = argparse.ArgumentParser(
        description="飞书文档同步（由系统定时任务调度）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    parser.add_argument("--space-id", type=str, default=None,
                        help="只同步指定知识空间 ID")
    parser.add_argument("--full", action="store_true",
                        help="全量同步（忽略增量清单）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，只列出不下载")
    parser.add_argument("--log-level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="覆盖配置文件中的日志级别")
    args = parser.parse_args()

    # 加载配置
    config = load_full_config(args.config)

    # 日志级别（命令行优先）
    log_level = args.log_level or config["log_level"]
    setup_logging(log_level, config["log_dir"])

    # 校验凭证
    if not config.get("app_id") or not config.get("app_secret"):
        log.error("缺少飞书凭证 (app_id / app_secret)，请检查配置文件")
        sys.exit(1)

    run_sync(config, space_id=args.space_id,
             full=args.full, dry_run=args.dry_run)


main()
