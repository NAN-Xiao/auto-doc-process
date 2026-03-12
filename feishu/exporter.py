#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量导出编排

职责：
  - 自动发现文档（遍历知识空间）
  - 批量导出流程编排（发现 → 下载）
  - 增量模式控制
  - 汇总报告

这是批量导出的「主逻辑」，串联 api / manifest / utils 各模块。
"""

import os
import sys
import time
import argparse
from pathlib import Path

from ..core.config import log, load_feishu_config, load_full_config, DEFAULT_EXPORT_DIR, setup_logging
from . import api
from . import manifest
from ..core.utils import (
    EXPORTABLE_TYPES, SKIP_TYPES,
    safe_filename, extract_docx_title, parse_feishu_url,
    acquire_lock, release_lock,
)


# ==================== 文档发现 ====================

def discover_documents(config: dict, space_id: str = None) -> list:
    """
    自动发现机器人有权限的所有可导出文档

    完全使用 TAT（应用身份），无需用户交互。
    前提：机器人已被添加为知识空间成员（可阅读）。

    Args:
        config: 飞书凭证配置
        space_id: 指定知识空间 ID（None = 遍历全部）

    Returns:
        [{"token", "doc_type", "ext", "name", "url", "obj_edit_time", ...}, ...]
    """
    base_url = config.get("base_url", "https://open.feishu.cn")
    access_token = api.get_tenant_access_token(config)
    if not access_token:
        return []

    # 使用配置中的类型映射，回退到默认值
    exportable_types = config.get("type_format_map", EXPORTABLE_TYPES)

    # 获取知识空间列表
    if space_id:
        spaces = [{"space_id": space_id, "name": f"指定空间 {space_id}"}]
    else:
        spaces = api.list_wiki_spaces(access_token, base_url)

    if not spaces:
        log.warning("没有发现任何知识空间（请确认机器人已被添加为知识空间成员）")
        return []

    # 遍历每个知识空间
    all_nodes = []
    for space in spaces:
        sid = space["space_id"]
        name = space["name"]
        log.info(f"遍历知识空间: {name} ({sid})")
        nodes = api.list_wiki_nodes(access_token, sid, base_url=base_url)
        all_nodes.extend(nodes)
        log.info(f"  共 {len(nodes)} 个节点")

    # 过滤可导出文档
    entries = []
    skipped = {}
    for node in all_nodes:
        obj_type = node["obj_type"]
        if obj_type in exportable_types:
            title = node["title"]

            # 飞书 wiki 节点偶尔返回空标题，用 get_node API 补查
            if not title:
                try:
                    _, _, resolved_title = api.resolve_wiki_token(
                        config, node["node_token"])
                    if resolved_title:
                        title = resolved_title
                        log.info(f"  补查标题: {node['obj_token'][:16]}... → [{title}]")
                except Exception:
                    pass

            entries.append({
                "token": node["obj_token"],
                "doc_type": obj_type,
                "ext": exportable_types[obj_type],
                "name": title,
                "url": f"wiki_node:{node['node_token']}",
                "space_id": node["space_id"],
                "obj_edit_time": node.get("obj_edit_time", ""),
            })
        else:
            skipped[obj_type] = skipped.get(obj_type, 0) + 1

    log.info(f"发现 {len(entries)} 个可导出文档")
    if skipped:
        log.info(f"已跳过: {', '.join(f'{t}({c})' for t, c in skipped.items())}")

    return entries


# ==================== 配置文件加载 ====================

def load_doc_list(yaml_path: str) -> list:
    """从 YAML 配置文件加载文档列表"""
    import yaml
    path = Path(yaml_path)
    if not path.exists():
        log.error(f"文档列表文件不存在: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    docs = data.get("docs", [])
    log.info(f"加载文档列表: {len(docs)} 条（来自 {path}）")
    return docs


def build_doc_entries(raw_docs: list) -> list:
    """将 YAML/URL 列表转为标准 entries 结构"""
    entries = []
    for item in raw_docs:
        url = item.get("url", "").strip()
        if not url:
            continue
        parsed = parse_feishu_url(url)
        if not parsed:
            log.warning(f"无法解析 URL: {url}")
            continue
        entries.append({
            "url": url,
            "token": parsed["token"],
            "doc_type": parsed["doc_type"],
            "ext": item.get("ext") or parsed["default_ext"],
            "name": item.get("name", ""),
        })
    return entries


def entries_from_urls(url_str: str) -> list:
    """从逗号分隔的 URL 字符串构建 entries"""
    urls = [u.strip() for u in url_str.split(",") if u.strip()]
    return build_doc_entries([{"url": u} for u in urls])


# ==================== 批量导出 ====================

def batch_export(client, config: dict, entries: list, output_dir: Path,
                 incremental: bool = True) -> tuple:
    """
    批量导出文档

    Args:
        client: lark_oapi Client
        config: 完整配置（飞书凭证 + 导出参数）
        entries: 文档条目列表
        output_dir: 输出目录
        incremental: 增量模式

    Returns:
        (success_list, fail_list, skip_list)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从配置读取导出参数
    poll_max_wait = config.get("poll_max_wait", 60)
    poll_interval = config.get("poll_interval", 2)
    poll_initial_delay = config.get("poll_initial_delay", 3)
    rate_limit_delay = config.get("rate_limit_delay", 1)
    manifest_path = config.get("manifest_path")

    man = manifest.load(manifest_path) if incremental else {}
    total = len(entries)
    success_list = []
    fail_list = []
    skip_list = []

    log.info(f"批量导出开始: {total} 个文档 → {output_dir}")
    if incremental:
        log.info(f"增量模式: ON (清单中已有 {len(man)} 条记录)")

    for idx, entry in enumerate(entries, 1):
        token = entry["token"]
        doc_type = entry["doc_type"]
        ext = entry["ext"]
        name = entry.get("name", "")
        edit_time = entry.get("obj_edit_time", "")

        prefix = f"[{idx}/{total}]"

        # 增量检查
        if incremental and not manifest.is_changed(man, token, edit_time):
            log.info(f"{prefix} 跳过(未修改): {name or token[:16]}")
            # 把 manifest 中记录的文件路径带上，以便后续流程判断是否需要入库
            man_record = man.get(token, {})
            skip_list.append({
                "entry": entry,
                "path": man_record.get("file_path", ""),
            })
            continue

        log.info(f"{prefix} 导出: {name or token[:16]} ({doc_type} → .{ext})")

        try:
            actual_token, actual_type = token, doc_type
            if doc_type == "wiki":
                actual_token, actual_type, resolved_title = api.resolve_wiki_token(config, token)
                if not actual_token:
                    raise RuntimeError("Wiki token 解析失败")
                # 用解析出的标题补充空名称
                if resolved_title and not name:
                    name = resolved_title

            # 输出路径（在 wiki 解析之后确定，确保用到解析后的名称）
            file_prefix = safe_filename(name, token)
            out_path = str(output_dir / f"{file_prefix}.{ext}")

            ticket = api.create_export_task(client, actual_token, actual_type, ext)
            if not ticket:
                raise RuntimeError("创建导出任务失败")

            result = api.poll_export_task(
                client, ticket, actual_token,
                max_wait=poll_max_wait,
                interval=poll_interval,
                initial_delay=poll_initial_delay,
            )
            if not result or not result.get("file_token"):
                raise RuntimeError("导出任务未完成")

            ok = api.download_export_file(client, result["file_token"], out_path)
            if not ok:
                raise RuntimeError("下载文件失败")

            # ---- 尝试从 docx 内容读取真实标题并重命名 ----
            actual_name = name
            if ext == "docx":
                real_title = extract_docx_title(out_path)
                if real_title and real_title != name:
                    new_prefix = safe_filename(real_title, token)
                    new_path = str(output_dir / f"{new_prefix}.{ext}")
                    # 避免覆盖已有的其他文件
                    if new_path != out_path:
                        if os.path.exists(new_path):
                            os.remove(new_path)
                        os.rename(out_path, new_path)
                        log.info(f"  文件名修正: {Path(out_path).name} → {Path(new_path).name}")
                        out_path = new_path
                        actual_name = real_title

            success_list.append({"entry": entry, "path": out_path})
            manifest.record_download(man, token, out_path, edit_time, actual_name,
                                     manifest_path=manifest_path)

        except Exception as e:
            log.error(f"{prefix} 失败: {e}")
            fail_list.append({"entry": entry, "error": str(e)})

        if idx < total:
            time.sleep(rate_limit_delay)

    # 汇总
    log.info("=" * 50)
    log.info(f"批量导出完成")
    log.info(f"  总计: {total}  成功: {len(success_list)}  "
             f"跳过: {len(skip_list)}  失败: {len(fail_list)}")

    if success_list:
        log.info("成功:")
        for item in success_list:
            log.info(f"  ✓ {item['path']}")

    if fail_list:
        log.warning("失败:")
        for item in fail_list:
            log.warning(f"  ✗ {item['entry'].get('name', item['entry']['token'][:16])}: {item['error']}")

    return success_list, fail_list, skip_list


# ==================== CLI 入口 ====================

def main():
    """命令行入口（也可独立运行）"""
    parser = argparse.ArgumentParser(
        description="飞书云文档批量导出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m feishu_sync.exporter --discover
  python -m feishu_sync.exporter --discover --dry-run
  python -m feishu_sync.exporter --discover --space-id 7613735903789370589
  python -m feishu_sync.exporter --discover --full
  python -m feishu_sync.exporter --list docs.yaml
  python -m feishu_sync.exporter --urls "https://xxx.feishu.cn/wiki/ABC,https://xxx.feishu.cn/docx/DEF"
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--discover", action="store_true",
                       help="自动发现机器人有权限的知识空间文档")
    group.add_argument("--list", type=str, metavar="YAML",
                       help="文档列表 YAML 配置文件路径")
    group.add_argument("--urls", type=str, metavar="URL1,URL2,...",
                       help="逗号分隔的飞书文档 URL")

    parser.add_argument("--space-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="只列出，不下载")
    parser.add_argument("--full", action="store_true", help="全量下载（忽略增量清单）")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--log-level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # 加载完整配置
    full_config = load_full_config(args.config)

    # 命令行参数覆盖配置文件
    log_level = args.log_level or full_config["log_level"]
    setup_logging(log_level, full_config["log_dir"])

    from datetime import datetime
    log.info("=" * 50)
    log.info("飞书云文档批量导出")
    log.info(f"启动时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("=" * 50)

    lock_path = full_config["lock_path"]
    if not args.dry_run and not acquire_lock(lock_path):
        log.error("无法获取进程锁，退出")
        sys.exit(1)

    try:
        if args.discover:
            entries = discover_documents(full_config, space_id=args.space_id)
        elif args.list:
            entries = build_doc_entries(load_doc_list(args.list))
        else:
            entries = entries_from_urls(args.urls)

        if not entries:
            log.warning("没有发现任何可导出的文档")
            sys.exit(0)

        log.info(f"待处理文档: {len(entries)} 个")
        for i, e in enumerate(entries, 1):
            name_part = f" [{e['name']}]" if e.get("name") else ""
            log.info(f"  {i}. {e['doc_type']:6s} → .{e['ext']:4s}  {e['token'][:16]}...{name_part}")

        if args.dry_run:
            log.info("预览模式，不执行下载")
            return

        client = api.create_lark_client(full_config)
        if not client:
            sys.exit(1)

        # 命令行 --output-dir 优先，否则用配置文件中的
        output_dir = Path(args.output_dir) if args.output_dir else full_config["output_dir"]

        success, fail, skip = batch_export(
            client, full_config, entries, output_dir,
            incremental=not args.full,
        )

        if fail:
            sys.exit(1)

    except KeyboardInterrupt:
        log.info("用户中断")
        sys.exit(130)
    except Exception as e:
        log.error(f"未预期的错误: {e}", exc_info=True)
        sys.exit(2)
    finally:
        if not args.dry_run:
            release_lock(lock_path)


if __name__ == "__main__":
    main()
