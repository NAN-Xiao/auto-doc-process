#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
飞书文档自动处理管线 - 统一入口

完整流程（由 Windows 任务计划程序定时调用）：
  飞书下载 → 预处理（拆分+向量化+pgvector）→ LightRAG 图谱构建 → 导出到 PostgreSQL

用法：
  python -m auto-doc-process                     → 完整流程：同步 + 预处理 + 图谱构建
  python -m auto-doc-process sync                → 仅同步飞书文档 + 预处理 + 图谱
  python -m auto-doc-process sync --dry-run      → 预览模式，只列出不下载
  python -m auto-doc-process sync --no-graph     → 同步 + 预处理，跳过图谱构建
  python -m auto-doc-process graph               → 仅构建图谱（处理最新批次）
  python -m auto-doc-process graph --batch xxx   → 构建指定批次的图谱
  python -m auto-doc-process query "问题"         → 查询知识图谱
  python -m auto-doc-process export out.csv      → 导出图谱数据
"""

import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

from .core.config import load_full_config, load_processor_config, log, setup_logging
from .core.utils import acquire_lock, release_lock


# ==================== 飞书同步 ====================

def run_sync(config: dict, space_id: str = None,
             full: bool = False, dry_run: bool = False,
             build_graph: bool = True):
    """
    执行一次飞书文档同步 + 预处理 + 图谱构建

    Args:
        config: 完整配置
        space_id: 指定知识空间（None = 全部）
        full: 全量模式（忽略增量清单）
        dry_run: 预览模式（只列出不下载）
        build_graph: 是否在预处理后构建知识图谱
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

        # ---- 步骤2：文档处理：拆分 → 向量化 → 存入 PostgreSQL ----
        batch_timestamp = None

        if success and config.get("vec_enabled") and config.get("db"):
            log.info("=" * 50)
            log.info("开始文档处理管线：拆分 → 向量化 → pgvector 入库")
            log.info("=" * 50)

            from .processor.workflow import BatchWorkflow

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

        # ---- 步骤3：LightRAG 知识图谱构建 ----
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


# ==================== 图谱构建 ====================

def _find_latest_batch(processed_dir: Path) -> Path:
    """找到最新的批次目录"""
    batch_dirs = [
        d for d in processed_dir.iterdir()
        if d.is_dir() and d.name[0].isdigit()
    ]
    if not batch_dirs:
        return None
    return max(batch_dirs, key=lambda d: d.name)


def _resolve_processed_dir() -> Path:
    """解析 processed 目录的绝对路径"""
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
    执行 LightRAG 图谱构建

    Args:
        batch_name: 批次目录名（None = 最新批次）
    """
    from .processor.graph_builder import LightRAGGraphBuilder, load_lightrag_config

    lightrag_cfg = load_lightrag_config()

    # 检查是否启用了图谱功能
    if not lightrag_cfg.get("pg_export", {}).get("enabled", True):
        log.info("LightRAG 图谱构建未启用（pg_export.enabled=false），跳过")
        return None

    processed_dir = _resolve_processed_dir()

    if not processed_dir.exists():
        log.error(f"预处理目录不存在: {processed_dir}")
        return None

    # 确定批次目录
    if batch_name:
        batch_dir = processed_dir / batch_name
        if not batch_dir.exists():
            log.error(f"指定的批次目录不存在: {batch_dir}")
            return None
    else:
        batch_dir = _find_latest_batch(processed_dir)
        if batch_dir is None:
            log.error(f"未找到任何批次目录: {processed_dir}")
            return None

    log.info("=" * 50)
    log.info(f"LightRAG 图谱构建: {batch_dir.name}")
    log.info("=" * 50)

    # 初始化构建器
    builder = LightRAGGraphBuilder()
    builder.initialize()

    # 构建图谱
    report = builder.build_from_processed_dir(batch_dir)

    # 保存报告
    report_file = batch_dir / "lightrag_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info(f"图谱报告已保存: {report_file}")

    return report


def _run_graph_query(question: str, mode: str = "hybrid"):
    """查询知识图谱"""
    from .processor.graph_builder import LightRAGGraphBuilder

    builder = LightRAGGraphBuilder()
    builder.initialize()

    log.info(f"查询模式: {mode}")
    log.info(f"问题: {question}")

    answer = builder.query(question, mode=mode)
    print(f"\n💡 回答:\n{answer}")


def _run_graph_export(output_path: str):
    """导出图谱数据"""
    from .processor.graph_builder import LightRAGGraphBuilder

    builder = LightRAGGraphBuilder()
    builder.initialize()

    ext = Path(output_path).suffix.lower()
    fmt_map = {".csv": "csv", ".xlsx": "excel", ".md": "md", ".txt": "txt"}
    fmt = fmt_map.get(ext, "csv")

    builder.export_graph_visualization(output_path, format=fmt)
    log.info(f"图谱数据已导出: {output_path}")


# ==================== CLI 入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="飞书文档自动处理管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m auto-doc-process                           # 完整流程
  python -m auto-doc-process sync                      # 同步 + 预处理 + 图谱
  python -m auto-doc-process sync --no-graph           # 同步 + 预处理，跳过图谱
  python -m auto-doc-process sync --dry-run            # 预览模式
  python -m auto-doc-process graph                     # 构建最新批次图谱
  python -m auto-doc-process graph --batch 20260306    # 构建指定批次图谱
  python -m auto-doc-process query "GVG怎么玩？"        # 查询图谱
  python -m auto-doc-process export graph.csv          # 导出图谱
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- sync 子命令 ----
    sync_parser = subparsers.add_parser("sync", help="同步飞书文档 + 预处理 + 图谱构建")
    sync_parser.add_argument("--config", type=str, default=None,
                             help="配置文件路径")
    sync_parser.add_argument("--space-id", type=str, default=None,
                             help="只同步指定知识空间 ID")
    sync_parser.add_argument("--full", action="store_true",
                             help="全量同步（忽略增量清单）")
    sync_parser.add_argument("--dry-run", action="store_true",
                             help="预览模式，只列出不下载")
    sync_parser.add_argument("--no-graph", action="store_true",
                             help="跳过 LightRAG 图谱构建")
    sync_parser.add_argument("--log-level", type=str, default=None,
                             choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                             help="覆盖日志级别")

    # ---- graph 子命令 ----
    graph_parser = subparsers.add_parser("graph", help="构建知识图谱（从已有预处理数据）")
    graph_parser.add_argument("--batch", type=str, default=None,
                              help="指定批次目录名（默认处理最新批次）")

    # ---- query 子命令 ----
    query_parser = subparsers.add_parser("query", help="查询知识图谱")
    query_parser.add_argument("question", type=str, help="查询问题")
    query_parser.add_argument("--mode", type=str, default="hybrid",
                              choices=["naive", "local", "global", "hybrid"],
                              help="查询模式（默认 hybrid）")

    # ---- export 子命令 ----
    export_parser = subparsers.add_parser("export", help="导出图谱数据")
    export_parser.add_argument("output", type=str,
                               help="输出文件路径（支持 .csv/.xlsx/.md/.txt）")

    # ---- 全局参数（兼容旧用法） ----
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
                        help="覆盖日志级别")

    args = parser.parse_args()

    # 根据子命令分发
    if args.command == "graph":
        _run_graph_build(args.batch)

    elif args.command == "query":
        _run_graph_query(args.question, mode=args.mode)

    elif args.command == "export":
        _run_graph_export(args.output)

    elif args.command == "sync" or args.command is None:
        # sync 或无子命令（兼容旧用法）→ 完整流程
        config_path = getattr(args, "config", None)
        config = load_full_config(config_path)

        log_level = getattr(args, "log_level", None) or config["log_level"]
        setup_logging(log_level, config["log_dir"])

        if not config.get("app_id") or not config.get("app_secret"):
            log.error("缺少飞书凭证 (app_id / app_secret)，请检查配置文件")
            sys.exit(1)

        build_graph = not getattr(args, "no_graph", False)

        run_sync(
            config,
            space_id=getattr(args, "space_id", None),
            full=getattr(args, "full", False),
            dry_run=getattr(args, "dry_run", False),
            build_graph=build_graph,
        )


main()
