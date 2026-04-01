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

            # 补查后仍无标题，跳过该文档
            if not title or not title.strip():
                log.info(f"  跳过(无标题): {node['obj_token'][:16]}...")
                skipped["untitled"] = skipped.get("untitled", 0) + 1
                continue

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
    批量导出文档（三阶段并行模式）

    阶段1: 批量创建导出任务（飞书服务端并行处理）
    阶段2: 统一轮询所有任务完成状态
    阶段3: 批量下载已完成的文件

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
    poll_max_wait = config.get("poll_max_wait", 300)
    poll_interval = config.get("poll_interval", 3)
    poll_initial_delay = config.get("poll_initial_delay", 5)
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

    # ========== 预处理：增量过滤 + Wiki Token 解析 ==========
    # 存放需要导出的任务信息
    tasks = []  # [{"idx", "entry", "name", "actual_token", "actual_type", "ext", "edit_time", "out_path"}, ...]

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
            man_record = man.get(token, {})
            skip_list.append({
                "entry": entry,
                "path": man_record.get("file_path", ""),
            })
            continue

        log.info(f"{prefix} 准备导出: {name or token[:16]} ({doc_type} → .{ext})")

        try:
            actual_token, actual_type = token, doc_type
            if doc_type == "wiki":
                actual_token, actual_type, resolved_title = api.resolve_wiki_token(config, token)
                if not actual_token:
                    raise RuntimeError("Wiki token 解析失败")
                if resolved_title and not name:
                    name = resolved_title

            # 解析后仍无标题，跳过
            if not name or not name.strip():
                log.info(f"{prefix} 跳过(无标题): {token[:16]}")
                skip_list.append({"entry": entry, "path": ""})
                continue

            file_prefix = safe_filename(name, token)
            out_path = str(output_dir / f"{file_prefix}.{ext}")

            tasks.append({
                "idx": idx,
                "entry": entry,
                "name": name,
                "actual_token": actual_token,
                "actual_type": actual_type,
                "ext": ext,
                "edit_time": edit_time,
                "out_path": out_path,
            })
        except Exception as e:
            log.error(f"{prefix} 预处理失败: {e}")
            fail_list.append({"entry": entry, "error": str(e)})

    if not tasks:
        log.info("没有需要导出的文档")
        _print_summary(total, success_list, fail_list, skip_list)
        return success_list, fail_list, skip_list

    # ========== 阶段1: 批量创建导出任务 ==========
    log.info("=" * 50)
    log.info(f"阶段1: 批量创建导出任务 ({len(tasks)} 个)")

    pending = []  # [{"task": ..., "ticket": ..., "attempt": 1}, ...]
    for i, task in enumerate(tasks):
        prefix = f"[{task['idx']}/{total}]"
        try:
            ticket = api.create_export_task(
                client, task["actual_token"], task["actual_type"], task["ext"]
            )
            if not ticket:
                raise RuntimeError("创建导出任务失败")
            log.info(f"  {prefix} 任务已创建: {task['name'] or task['actual_token'][:16]}")
            pending.append({"task": task, "ticket": ticket, "attempt": 1})
        except Exception as e:
            log.error(f"  {prefix} 创建失败: {e}")
            log.error(f"    token={task['actual_token']}, type={task['actual_type']}, "
                      f"ext={task['ext']}, name={task['name']}")
            fail_list.append({"entry": task["entry"], "error": str(e)})

        # 限速：每创建一个任务等一下，避免触发频率限制
        if i < len(tasks) - 1:
            time.sleep(rate_limit_delay)

    if not pending:
        log.warning("所有任务创建失败")
        _print_summary(total, success_list, fail_list, skip_list)
        return success_list, fail_list, skip_list

    # ========== 阶段2: 统一轮询所有任务 ==========
    log.info("=" * 50)
    log.info(f"阶段2: 等待 {len(pending)} 个导出任务完成...")

    time.sleep(poll_initial_delay)  # 给服务端一点启动时间

    completed = []   # [{"task": ..., "result": ...}, ...]
    start_time = time.time()

    while pending and (time.time() - start_time) < poll_max_wait:
        still_pending = []

        for item in pending:
            task = item["task"]
            ticket = item["ticket"]
            prefix = f"[{task['idx']}/{total}]"

            result = _check_export_status(client, ticket, task["actual_token"])

            if result == "done":
                # 任务完成，获取完整结果
                full_result = _get_export_result(client, ticket, task["actual_token"])
                if full_result:
                    log.info(f"  {prefix} ✓ 导出完成: {task['name'] or task['actual_token'][:16]}")
                    completed.append({"task": task, "result": full_result})
                else:
                    log.error(f"  {prefix} ✗ 获取结果失败")
                    fail_list.append({"entry": task["entry"], "error": "获取导出结果失败"})
            elif result == "failed":
                # 尝试重试（最多 3 次）
                if item["attempt"] < 3:
                    log.warning(f"  {prefix} 导出失败，重试中...")
                    try:
                        new_ticket = api.create_export_task(
                            client, task["actual_token"], task["actual_type"], task["ext"]
                        )
                        if new_ticket:
                            still_pending.append({
                                "task": task, "ticket": new_ticket,
                                "attempt": item["attempt"] + 1,
                            })
                            continue
                    except Exception:
                        pass
                log.error(f"  {prefix} ✗ 导出失败: {task['name'] or task['actual_token'][:16]}")
                fail_list.append({"entry": task["entry"], "error": "导出任务失败"})
            else:
                # 仍在处理中
                still_pending.append(item)

        pending = still_pending

        if pending:
            remaining = [p["task"]["name"] or p["task"]["actual_token"][:12] for p in pending]
            elapsed = int(time.time() - start_time)
            log.info(f"  等待中 ({elapsed}s/{poll_max_wait}s)... "
                     f"剩余 {len(pending)} 个: {', '.join(remaining[:3])}"
                     f"{'...' if len(remaining) > 3 else ''}")
            time.sleep(poll_interval)

    # 超时未完成的任务
    for item in pending:
        task = item["task"]
        prefix = f"[{task['idx']}/{total}]"
        log.error(f"  {prefix} ✗ 超时: {task['name'] or task['actual_token'][:16]}")
        fail_list.append({"entry": task["entry"], "error": f"导出超时 ({poll_max_wait}s)"})

    if not completed:
        log.warning("没有任务完成")
        _print_summary(total, success_list, fail_list, skip_list)
        return success_list, fail_list, skip_list

    # ========== 阶段3: 批量下载 ==========
    log.info("=" * 50)
    log.info(f"阶段3: 下载 {len(completed)} 个文件")

    for item in completed:
        task = item["task"]
        result = item["result"]
        prefix = f"[{task['idx']}/{total}]"
        out_path = task["out_path"]
        ext = task["ext"]
        name = task["name"]
        entry = task["entry"]
        edit_time = task["edit_time"]
        token = entry["token"]

        try:
            ok = api.download_export_file(client, result["file_token"], out_path)
            if not ok:
                raise RuntimeError("下载文件失败")

            file_size = os.path.getsize(out_path)
            log.info(f"  {prefix} 下载完成: {Path(out_path).name} ({file_size:,} bytes)")

            # ---- 尝试从 docx 内容读取真实标题并重命名 ----
            actual_name = name
            if ext == "docx":
                real_title = extract_docx_title(out_path)
                if real_title and real_title != name:
                    new_prefix = safe_filename(real_title, token)
                    new_path = str(output_dir / f"{new_prefix}.{ext}")
                    if new_path != out_path:
                        if os.path.exists(new_path):
                            os.remove(new_path)
                        os.rename(out_path, new_path)
                        log.info(f"    文件名修正: {Path(out_path).name} → {Path(new_path).name}")
                        out_path = new_path
                        actual_name = real_title

            success_list.append({"entry": entry, "path": out_path})
            manifest.record_download(man, token, out_path, edit_time, actual_name,
                                     manifest_path=manifest_path)

        except Exception as e:
            log.error(f"  {prefix} 下载失败: {e}")
            fail_list.append({"entry": entry, "error": str(e)})

    # 汇总
    _print_summary(total, success_list, fail_list, skip_list)
    return success_list, fail_list, skip_list


def _check_export_status(client, ticket: str, doc_token: str) -> str:
    """
    检查单个导出任务状态（不阻塞）

    Returns:
        "done" / "failed" / "pending"
    """
    from lark_oapi.api.drive.v1 import GetExportTaskRequest

    try:
        request = GetExportTaskRequest.builder() \
            .ticket(ticket) \
            .token(doc_token) \
            .build()

        response = client.drive.v1.export_task.get(request)

        if not response.success():
            # API 返回非成功：可能是限流/临时错误，视为"仍在处理"而非"失败"
            log.debug(f"查询导出状态: code={response.code}, msg={response.msg}")
            return "pending"

        result = response.data.result if response.data else None
        if result:
            file_token = getattr(result, "file_token", None)
            job_status = getattr(result, "job_status", None)

            if file_token:
                return "done"
            elif job_status in (2, 3):  # 2=失败, 3=过期
                err = getattr(result, "job_error_msg", "")
                log.warning(f"导出任务状态异常: status={job_status}, msg={err}")
                return "failed"

        return "pending"
    except Exception as e:
        log.debug(f"查询导出状态异常（视为 pending）: {e}")
        return "pending"


def _get_export_result(client, ticket: str, doc_token: str) -> dict:
    """获取已完成的导出任务的结果"""
    from lark_oapi.api.drive.v1 import GetExportTaskRequest

    try:
        request = GetExportTaskRequest.builder() \
            .ticket(ticket) \
            .token(doc_token) \
            .build()

        response = client.drive.v1.export_task.get(request)

        if not response.success():
            return None

        result = response.data.result if response.data else None
        if result:
            file_token = getattr(result, "file_token", None)
            if file_token:
                return {
                    "file_token": file_token,
                    "file_name": getattr(result, "file_name", None),
                    "file_size": getattr(result, "file_size", None),
                }
        return None
    except Exception:
        return None


def _print_summary(total, success_list, fail_list, skip_list):
    """打印导出汇总"""
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
