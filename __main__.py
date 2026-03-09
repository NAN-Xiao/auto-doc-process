#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
飞书文档自动处理管线

定时批处理：拉取飞书文档 → 拆分/向量化/存库 → 重建图谱 → 导出到 PostgreSQL

用法：
  python -m auto-doc-process                  # 全流程
  python -m auto-doc-process --dry-run        # 预览模式（只列文档不下载）
  python -m auto-doc-process --full           # 全量同步（忽略增量清单）
  python -m auto-doc-process --no-graph       # 跳过图谱构建
"""

import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

from .core.config import load_full_config, load_processor_config, log, setup_logging, ConfigError
from .core.utils import acquire_lock, release_lock


def run_sync(config: dict, full: bool = False, dry_run: bool = False,
             build_graph: bool = True):
    """
    飞书文档同步与处理（完整管线）

    流程：
      1. 拉取飞书文档列表 → 增量下载 docx/pdf/xlsx
      2. 拆分 chunks → 向量化 → 生成元数据 → 存入 pgvector
      3. LightRAG 重建图谱文件（lightrag_workspace/）
      4. 导出实体/关系到 PostgreSQL
    """
    from .feishu.exporter import discover_documents, batch_export
    from .feishu.api import create_lark_client

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("=" * 50)
    log.info(f"飞书文档同步  {now}")
    log.info("=" * 50)

    lock_path = config.get("lock_path")

    if not dry_run:
        if not acquire_lock(lock_path):
            log.warning("有任务在运行，跳过")
            return

    try:
        feishu_config = {
            "app_id": config["app_id"],
            "app_secret": config["app_secret"],
            "base_url": config["base_url"],
            "type_format_map": config.get("type_format_map"),
        }

        # 获取文档列表
        space_ids = config.get("space_ids", [])
        if space_ids:
            all_entries = []
            for sid in space_ids:
                entries = discover_documents(feishu_config, space_id=sid)
                all_entries.extend(entries)
        else:
            all_entries = discover_documents(feishu_config)

        if not all_entries:
            log.info("未发现可导出文档")
            return

        log.info(f"发现 {len(all_entries)} 个文档")
        for i, e in enumerate(all_entries, 1):
            name_part = f" [{e['name']}]" if e.get("name") else ""
            log.info(f"  {i}. {e['doc_type']:6s} → .{e['ext']:4s}  "
                     f"{e['token'][:16]}...{name_part}")

        if dry_run:
            log.info("预览模式，未下载")
            return

        # 创建 Client
        client = create_lark_client(feishu_config)
        if not client:
            log.error("创建 SDK Client 失败")
            sys.exit(1)

        out_path = config.get("output_dir", Path("feishu_exports"))

        # 批量导出
        success, fail, skip = batch_export(
            client, config, all_entries, out_path,
            incremental=not full,
        )

        log.info(f"同步完成: 成功={len(success)} 跳过={len(skip)} 失败={len(fail)}")

        batch_timestamp = None

        if success and config.get("vec_enabled") and config.get("db"):
            log.info("=" * 50)
            log.info("文档处理管线")
            log.info("=" * 50)

            from .processor.workflow import BatchWorkflow

            proc_config = load_processor_config()
            use_llm = proc_config.get('doc_splitter', {}).get('image_naming', {}).get('use_llm', False)

            workflow = BatchWorkflow(
                use_llm_naming=use_llm,
                db_config=config["db"],
            )

            batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            proc_success = 0
            proc_fail = 0
            for item in success:
                file_path = item.get("path", "")
                if not file_path or not Path(file_path).exists():
                    continue

                doc_path = Path(file_path)
                # 仅处理 docx / pdf（项目不处理 Excel）
                if doc_path.suffix.lower() not in (".docx", ".pdf"):
                    log.info(f"  跳过: {doc_path.name}")
                    continue

                # 构建来源元数据，传递到 pgvector 存储
                entry = item.get("entry", {})
                doc_meta = {
                    "space_id": entry.get("space_id", ""),
                    "source_url": entry.get("url", ""),
                }

                result = workflow.process_single_document(
                    doc_path, batch_timestamp=batch_timestamp,
                    doc_meta=doc_meta,
                )

                if result and result.get("success"):
                    proc_success += 1
                else:
                    proc_fail += 1

            log.info(f"文档处理完成: 成功={proc_success} 失败={proc_fail}")

        elif success and not config.get("db"):
            log.info("未配置数据库，跳过文档处理")

        # 图谱构建
        if build_graph and batch_timestamp:
            _run_graph_build(batch_timestamp)

        if fail:
            sys.exit(1)

    except Exception as e:
        log.error(f"同步异常: {e}")
        traceback.print_exc()
        sys.exit(2)
    finally:
        if not dry_run:
            release_lock(lock_path)


def _find_latest_batch(processed_dir: Path) -> Path:
    """获取最新批次目录"""
    batch_dirs = [
        d for d in processed_dir.iterdir()
        if d.is_dir() and d.name[0].isdigit()
    ]
    if not batch_dirs:
        return None
    return max(batch_dirs, key=lambda d: d.name)


def _resolve_processed_dir() -> Path:
    """获取 processed 目录路径"""
    from .core.config import MODULE_DIR
    proc_config = load_processor_config()
    paths = proc_config.get("paths", {})
    documents_dir = paths.get("documents_dir", "../")
    processed_subdir = paths.get("processed_subdir", "processed")

    docs_path = Path(documents_dir)
    if not docs_path.is_absolute():
        docs_path = (MODULE_DIR / docs_path).resolve()

    return docs_path / processed_subdir


def _run_graph_build(batch_name: str = None):
    """
    构建 LightRAG 图谱（内部调用，由 run_sync 触发）

    Args:
        batch_name: 批次名（None为最新）
    """
    from .processor.graph_builder import LightRAGGraphBuilder, load_lightrag_config

    lightrag_cfg = load_lightrag_config()

    if not lightrag_cfg.get("pg_export", {}).get("enabled", True):
        log.info("图谱未启用，跳过")
        return None

    processed_dir = _resolve_processed_dir()

    if not processed_dir.exists():
        log.error(f"预处理目录不存在: {processed_dir}")
        return None

    # 批次目录
    if batch_name:
        batch_dir = processed_dir / batch_name
        if not batch_dir.exists():
            log.error(f"批次目录不存在: {batch_dir}")
            return None
    else:
        batch_dir = _find_latest_batch(processed_dir)
        if batch_dir is None:
            log.error(f"未找到批次目录: {processed_dir}")
            return None

    log.info("=" * 50)
    log.info(f"LightRAG 图谱构建: {batch_dir.name}")
    log.info("=" * 50)

    builder = LightRAGGraphBuilder()
    builder.initialize()

    report = builder.build_from_processed_dir(batch_dir)

    report_file = batch_dir / "lightrag_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info(f"图谱报告已保存: {report_file}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="飞书文档自动处理管线（定时批处理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
流程：拉取飞书文档 → 拆分/向量化/存库 → 重建图谱 → 导出到 PostgreSQL

示例:
  python -m auto-doc-process                  # 全流程
  python -m auto-doc-process --dry-run        # 预览（只列文档不下载）
  python -m auto-doc-process --full           # 全量同步
  python -m auto-doc-process --no-graph       # 跳过图谱构建
        """,
    )

    parser.add_argument("--full", action="store_true",
                        help="全量同步（忽略增量清单，重新下载所有文档）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式（只列出文档列表，不下载不处理）")
    parser.add_argument("--no-graph", action="store_true",
                        help="跳过图谱构建步骤")
    parser.add_argument("--log-level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="指定日志级别（默认从配置文件读取）")

    args = parser.parse_args()

    config = load_full_config()
    log_level = args.log_level or config["log_level"]
    setup_logging(log_level, config["log_dir"])

    if not config.get("app_id") or not config.get("app_secret"):
        log.error("缺少飞书凭证 (app_id / app_secret)")
        sys.exit(1)

    run_sync(
        config,
        full=args.full,
        dry_run=args.dry_run,
        build_graph=not args.no_graph,
    )


try:
    main()
except ConfigError as e:
    log.error(str(e))
    sys.exit(1)
