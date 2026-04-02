#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
飞书文档自动处理管线

四阶段管线，可整体执行或单独调试：
  1. download  — 拉取飞书文档（增量下载 + 构建 manifest）
  2. process   — 拆分 chunks + 向量化（生成 metadata/embeddings）
  3. store     — 将已处理的文档批量写入 pgvector
  4. graph     — LightRAG 知识图谱构建

目录结构（扁平，无时间戳层）：
  documents/              ← 下载的原始 docx
  documents/processed/    ← 处理后的数据
    ├─ 文档A/
    │   ├─ chunks/
    │   ├─ embeddings/
    │   └─ metadata/
    └─ 文档B/
        ├─ chunks/
        ├─ embeddings/
        └─ metadata/

用法（通过 run.py / start.bat 启动）：
  run.py                          # 全流程（1→2→3→4）
  run.py --step download          # 只下载
  run.py --step process           # 只处理（需先下载）
  run.py --step store             # 只入库（需先处理）
  run.py --step graph             # 只构建图谱（需先处理）
  run.py --step download,process  # 下载 + 处理
  run.py --dry-run                # 预览（列出文档不下载）
  run.py --full                   # 全量同步（忽略增量清单）
  run.py --reset-db               # 清空数据库 → 全量重建
  run.py --schedule               # 守护进程模式
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from ..core.config import load_full_config, load_processor_config, log, setup_logging, ConfigError, MODULE_DIR
from ..core.utils import acquire_lock, release_lock, atomic_write_json
from .paths import resolve_processed_dir


# ==================== 阶段常量 ====================
STEPS_ALL = ["download", "process", "store", "graph"]


# ==================== 阶段 1: 下载 ====================

def step_download(config: dict, full: bool = False, dry_run: bool = False) -> dict:
    """
    阶段 1: 从飞书拉取文档（增量下载 + 构建 manifest）

    Returns:
        {"success": [...], "fail": [...], "skip": [...], "entries": [...]}
    """
    from ..feishu.exporter import discover_documents, batch_export
    from ..feishu.api import create_lark_client

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
        return {"success": [], "fail": [], "skip": [], "entries": []}

    log.info(f"发现 {len(all_entries)} 个文档")
    for i, e in enumerate(all_entries, 1):
        name_part = f" [{e['name']}]" if e.get("name") else ""
        log.info(f"  {i}. {e['doc_type']:6s} → .{e['ext']:4s}  "
                 f"{e['token'][:16]}...{name_part}")

    if dry_run:
        log.info("预览模式，未下载")
        return {"success": [], "fail": [], "skip": [], "entries": all_entries}

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

    log.info(f"下载完成: 成功={len(success)} 跳过={len(skip)} 失败={len(fail)}")
    return {"success": success, "fail": fail, "skip": skip, "entries": all_entries}


# ==================== 阶段 2: 处理 ====================

def step_process(config: dict, download_result: dict = None,
                 reset_db: bool = False) -> dict:
    """
    阶段 2: 拆分 + 向量化（不入库）

    通过子进程隔离执行文档处理。如果某个文档导致子进程崩溃（C层段错误等），
    主进程会捕获崩溃、报告异常，并自动重启子进程继续处理剩余文档。

    Args:
        config: 完整配置
        download_result: 阶段1的输出（None 时自动从 manifest 读取已下载文档）
        reset_db: 是否全量重建模式（跳过"已入库"判断，处理所有文档）

    Returns:
        {"results": [...], "success": N, "skip": N, "fail": N, "workflow": workflow}
    """
    from ..processor.workflow import BatchWorkflow

    proc_config = load_processor_config()
    use_llm = proc_config.get('doc_splitter', {}).get('image_naming', {}).get('use_llm', False)

    workflow = BatchWorkflow(
        use_llm_naming=use_llm,
        db_config=config.get("db"),
    )

    # ── 确定要处理的文档列表 ──
    if workflow.vector_storage:
        workflow.vector_storage.init_table()
        if reset_db:
            stored_names = set()
            log.info("全量重建模式：将处理所有文档")
        else:
            stored_names = workflow.vector_storage.get_stored_doc_names()
            log.info(f"数据库中已有 {len(stored_names)} 个文档")
    else:
        stored_names = set()
        if not config.get("vec_enabled"):
            log.info("向量化已关闭：仅执行无向量处理流程")
        elif not config.get("db"):
            log.info("数据库未配置：执行本地处理流程，不做入库去重")

    items_to_process = _collect_items_to_process(config, download_result, stored_names)

    if not items_to_process:
        log.info("没有待处理文件")
        return {"results": [], "success": 0, "skip": 0, "fail": 0, "workflow": workflow}

    # 时间戳仅用于 DB 元数据字段（doc_timestamp），不影响目录结构
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("=" * 50)
    log.info(f"预处理阶段启动 — 共 {len(items_to_process)} 个文件")
    log.info("=" * 50)

    # ── 子进程隔离处理（崩溃保护） ──
    # 从 lightrag.yaml 读取性能参数
    from ..processor.graph_builder import load_lightrag_config as _load_lr_cfg
    _perf = _load_lr_cfg().get("performance", {})
    proc_results = _run_process_with_isolation(
        items_to_process, batch_timestamp,
        max_crash_restarts=_perf.get("max_crash_restarts", 5),
    )

    proc_success = sum(1 for r in proc_results if r and r.get("success") and not r.get("skipped"))
    proc_skip = sum(1 for r in proc_results if r and r.get("skipped"))
    proc_fail = sum(1 for r in proc_results if not r or not r.get("success"))

    log.info(f"处理完成: 成功={proc_success} 跳过(空文档)={proc_skip} 失败={proc_fail}")

    try:
        quality_report = workflow.build_batch_quality_report(proc_results)
        quality_report.update({
            "batch_timestamp": batch_timestamp,
            "success_documents": proc_success,
            "skipped_documents": proc_skip,
            "failed_documents": proc_fail,
        })
        processed_dir = resolve_processed_dir()
        report_file = processed_dir / "preprocessing_quality_report.json"
        atomic_write_json(report_file, quality_report)
        log.info(f"预处理质量报告已保存（原子写入）: {report_file}")
    except Exception as e:
        log.warning(f"生成预处理质量报告失败: {e}")

    return {
        "results": proc_results,
        "success": proc_success,
        "skip": proc_skip,
        "fail": proc_fail,
        "workflow": workflow,
    }


def _collect_items_to_process(config: dict, download_result: dict,
                              stored_names: set) -> list:
    """收集待处理文档列表"""
    items = []

    if download_result:
        # 有阶段1的输出：新下载 + 已下载但未入库
        items = list(download_result.get("success", []))

        for sk in download_result.get("skip", []):
            sk_path = sk.get("path", "")
            if not sk_path or not Path(sk_path).exists():
                continue
            doc_stem = Path(sk_path).stem
            if doc_stem not in stored_names:
                log.info(f"  补充处理(已下载但未入库): {Path(sk_path).name}")
                items.append(sk)
    else:
        # 无阶段1输出：从 manifest 扫描所有已下载文档
        from ..feishu import manifest as mf
        manifest_path = config.get("manifest_path")
        man = mf.load(manifest_path)
        log.info(f"从 manifest 加载 {len(man)} 条记录")

        for token, record in man.items():
            file_path = record.get("file_path", "")
            if not file_path or not Path(file_path).exists():
                continue
            doc_stem = Path(file_path).stem
            if doc_stem not in stored_names:
                log.info(f"  待处理(未入库): {Path(file_path).name}")
                items.append({
                    "path": file_path,
                    "entry": {"space_id": "", "url": "", "obj_edit_time": record.get("obj_edit_time", "")},
                })

        # manifest 为空或没找到待处理文档时，兜底扫描文档目录
        if not items:
            docs_dir = config.get("output_dir")
            if docs_dir and Path(docs_dir).exists():
                log.info(f"manifest 无可用记录，扫描文档目录: {docs_dir}")
                for f in sorted(Path(docs_dir).iterdir()):
                    if f.is_file() and f.suffix.lower() in (".docx", ".pdf"):
                        if f.stem not in stored_names:
                            log.info(f"  待处理(目录扫描): {f.name}")
                            items.append({
                                "path": str(f),
                                "entry": {"space_id": "", "url": "", "obj_edit_time": ""},
                            })

    # 过滤无效项
    proc_config = load_processor_config()
    supported_formats = {
        str(fmt).lower()
        for fmt in proc_config.get("doc_splitter", {}).get("supported_formats", [".pdf", ".docx", ".xlsx", ".xls"])
    }
    valid = []
    for item in items:
        fp = item.get("path", "")
        if fp and Path(fp).exists() and Path(fp).suffix.lower() in supported_formats:
            valid.append(item)

    valid.extend(_collect_excel_items(proc_config))

    deduped = []
    seen_paths = set()
    for item in valid:
        path_key = str(Path(item["path"]).resolve()).lower()
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        deduped.append(item)
    return deduped


def _collect_excel_items(proc_config: dict) -> list:
    """从 excel_dir 收集 Excel 元数据处理任务。"""
    excel_dir = Path(proc_config.get("paths", {}).get("excel_dir", ""))
    if not excel_dir.exists():
        return []

    items = []
    for pattern in ("*.xlsx", "*.xls"):
        for file_path in sorted(excel_dir.glob(pattern)):
            if not file_path.is_file():
                continue
            items.append({
                "path": str(file_path),
                "entry": {"space_id": "", "url": "", "obj_edit_time": ""},
            })
    return items


def _run_process_with_isolation(items: list, batch_timestamp: str,
                                max_crash_restarts: int = 5) -> list:
    """
    在子进程中处理文档（崩溃隔离）

    策略：
      1. 将待处理文档列表写入临时 JSON 文件
      2. 启动子进程（通过 run.py --_worker）处理所有文档
      3. 子进程逐个处理，每完成一个就追加到 JSONL 输出文件
      4. 如果子进程崩溃：
         - 读取已完成的结果
         - 识别崩溃的文档，记录错误
         - 对剩余文档启动新子进程继续处理
      5. 最多重启 MAX_CRASH_RESTARTS 次

    Returns:
        所有文档的处理结果列表
    """
    run_py = str(MODULE_DIR / "run.py")
    all_results = []
    remaining = list(items)
    crash_count = 0

    while remaining:
        # ── 准备临时文件 ──
        fd_in, input_file = tempfile.mkstemp(suffix="_input.json", prefix="docproc_")
        fd_out, output_file = tempfile.mkstemp(suffix="_output.jsonl", prefix="docproc_")
        os.close(fd_in)
        os.close(fd_out)

        try:
            # 构建子进程任务
            task = {
                "items": [
                    {
                        "path": item["path"],
                        "doc_meta": {
                            "space_id": item.get("entry", {}).get("space_id", ""),
                            "source_url": item.get("entry", {}).get("url", ""),
                            "source_updated_at": item.get("entry", {}).get("obj_edit_time", ""),
                        },
                    }
                    for item in remaining
                ],
                "batch_timestamp": batch_timestamp,
            }
            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(task, f, ensure_ascii=False)

            # ── 启动子进程 ──
            log.info(f"启动子进程处理 {len(remaining)} 个文档...")
            ret = subprocess.run(
                [sys.executable, run_py, "--_worker", input_file, output_file],
                cwd=str(MODULE_DIR.parent),
            )

            # ── 读取已完成的结果 ──
            completed = _read_jsonl_results(output_file)
            all_results.extend(completed)
            completed_paths = {r.get("doc_path", "") for r in completed}

            if ret.returncode == 0:
                # 正常结束
                remaining = []
            else:
                # ── 子进程崩溃 ──
                crash_count += 1
                # 找到崩溃的文档（第一个未完成的）
                crashed_doc = None
                new_remaining = []
                found_crash = False

                for item in remaining:
                    item_path = item.get("path", "")
                    if item_path in completed_paths:
                        continue
                    if not found_crash:
                        crashed_doc = item
                        found_crash = True
                    else:
                        new_remaining.append(item)

                if crashed_doc:
                    crash_name = Path(crashed_doc["path"]).name
                    log.error("=" * 50)
                    log.error(f"⚠ 子进程崩溃 (exit code: {ret.returncode})")
                    log.error(f"  崩溃文档: {crash_name}")
                    log.error(f"  已完成: {len(completed)} 个, 剩余: {len(new_remaining)} 个")
                    log.error("=" * 50)

                    all_results.append({
                        "success": False,
                        "document": crash_name,
                        "doc_path": crashed_doc["path"],
                        "error": f"子进程崩溃 (exit code: {ret.returncode})",
                    })

                remaining = new_remaining

                if crash_count >= max_crash_restarts:
                    log.error(f"子进程连续崩溃 {crash_count} 次，放弃剩余 {len(remaining)} 个文档")
                    for item in remaining:
                        all_results.append({
                            "success": False,
                            "document": Path(item["path"]).name,
                            "doc_path": item["path"],
                            "error": "因多次崩溃被跳过",
                        })
                    remaining = []
                elif remaining:
                    log.info(f"自动重启子进程（第 {crash_count} 次），继续处理剩余 {len(remaining)} 个文档...")
        finally:
            # ── 清理临时文件 ──
            for tmp in (input_file, output_file):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    return all_results


def _read_jsonl_results(filepath: str) -> list:
    """读取 JSONL 格式的处理结果（每行一个 JSON 对象）"""
    results = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        # 将路径字符串转回 Path 对象（供 batch_store 使用）
                        if "_dir_info" in obj:
                            di = obj["_dir_info"]
                            for k in ("embeddings_dir", "metadata_dir", "chunks_dir"):
                                if k in di and isinstance(di[k], str):
                                    di[k] = Path(di[k])
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return results


# ==================== 子进程 Worker ====================

def _worker_process_documents(input_file: str, output_file: str):
    """
    子进程入口：处理文档列表，逐条写入 JSONL 结果文件。

    由 run.py --_worker <input> <output> 调用。
    每完成一个文档立即写入结果，确保崩溃时已完成的结果不丢失。
    """
    from ..processor.workflow import BatchWorkflow

    config = load_full_config()
    setup_logging(config.get("log_level", "INFO"), config.get("log_dir"))

    with open(input_file, "r", encoding="utf-8") as f:
        task = json.load(f)

    items = task["items"]
    batch_timestamp = task["batch_timestamp"]

    proc_config = load_processor_config()
    use_llm = proc_config.get("doc_splitter", {}).get("image_naming", {}).get("use_llm", False)

    workflow = BatchWorkflow(
        use_llm_naming=use_llm,
        db_config=config.get("db"),
    )

    for idx, item in enumerate(items, 1):
        file_path = item.get("path", "")
        doc_meta = item.get("doc_meta", {})
        doc_path = Path(file_path)

        log.info(f"[{idx}/{len(items)}] 处理: {doc_path.name}")

        result = workflow.process_single_document(
            doc_path,
            batch_timestamp=batch_timestamp,
            doc_meta=doc_meta,
            store_to_db=False,
        )

        if result is None:
            result = {
                "success": False,
                "document": doc_path.name,
                "doc_path": str(doc_path),
                "error": "process_single_document 返回 None",
            }

        # 确保 doc_path 字段存在（供主进程识别完成状态）
        result["doc_path"] = str(doc_path)

        # 序列化 _dir_info 中的 Path 对象
        if "_dir_info" in result:
            di = result["_dir_info"]
            for k in ("embeddings_dir", "metadata_dir", "chunks_dir"):
                if k in di and hasattr(di[k], "__fspath__"):
                    di[k] = str(di[k])

        # 立即追加到输出文件（崩溃安全）
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            f.flush()

    log.info(f"子进程处理完成: {len(items)} 个文档")


# ==================== 阶段 3: 入库 ====================

def step_store(config: dict, process_result: dict = None,
               reset_db: bool = False) -> int:
    """
    阶段 3: 将处理好的文档批量写入 pgvector（单事务）

    Args:
        config: 完整配置
        process_result: 阶段2的输出（None 时自动从 processed/ 目录扫描）
        reset_db: 全量重建模式（跳过"已入库"判断）

    Returns:
        入库的文档数
    """
    if not config.get("db"):
        log.warning("未配置数据库，跳过入库")
        return 0

    workflow = None
    proc_results = []

    if process_result:
        workflow = process_result.get("workflow")
        proc_results = process_result.get("results", [])

    # 如果没有 workflow，自己创建一个
    if not workflow:
        from ..processor.workflow import BatchWorkflow
        proc_config = load_processor_config()
        use_llm = proc_config.get('doc_splitter', {}).get('image_naming', {}).get('use_llm', False)
        workflow = BatchWorkflow(use_llm_naming=use_llm, db_config=config["db"])
        workflow.vector_storage.init_table()

    # 如果没有处理结果，从 processed/ 目录扫描
    if not proc_results:
        proc_results = _load_results_from_processed_dir(workflow, reset_db=reset_db)

    valid_results = [r for r in proc_results if r and r.get("success") and not r.get("skipped")]

    if not valid_results:
        log.info("没有需要入库的文档")
        return 0

    log.info("=" * 50)
    log.info(f"统一入库（单事务批量写入）— {len(valid_results)} 个文档")
    log.info("=" * 50)

    try:
        stored = workflow.batch_store(valid_results)
        log.info(f"入库完成: {stored} 个文档")
        return stored
    except Exception as e:
        log.error(f"统一入库失败（已回滚）: {e}")
        return 0


def _load_results_from_processed_dir(workflow, reset_db: bool = False) -> list:
    """
    从 processed/ 目录扫描所有已处理但未入库的文档，
    构建 result 对象列表供 batch_store 使用。

    目录结构：processed/{doc_name}/metadata/chunk_*.json
    """
    processed_dir = resolve_processed_dir()
    if not processed_dir.exists():
        log.warning(f"processed 目录不存在: {processed_dir}")
        return []

    if reset_db:
        stored_names = set()  # 全量重建：不跳过任何文档
    else:
        stored_names = workflow.vector_storage.get_stored_doc_names()
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for doc_dir in sorted(processed_dir.iterdir()):
        if not doc_dir.is_dir():
            continue

        doc_name = doc_dir.name
        if doc_name in stored_names:
            log.info(f"  已入库，跳过: {doc_name}")
            continue

        metadata_dir = doc_dir / "metadata"
        embeddings_dir = doc_dir / "embeddings"
        chunks_dir = doc_dir / "chunks"

        if not metadata_dir.exists() or not embeddings_dir.exists():
            continue

        chunk_files = sorted(metadata_dir.glob("chunk_*.json"))
        if not chunk_files:
            continue

        log.info(f"  加载: {doc_name} ({len(chunk_files)} chunks)")

        # 独立执行 store 时，尽量从已处理 metadata 回读来源信息
        # 这样 download->process 与后续单独 store 拆开执行时，space_id/source_url 不会丢失
        space_id = ""
        source_url = ""
        try:
            with open(chunk_files[0], "r", encoding="utf-8") as f:
                first_meta = json.load(f) or {}
            raw_meta = first_meta.get("metadata", {}) if isinstance(first_meta, dict) else {}
            if isinstance(raw_meta, dict):
                space_id = raw_meta.get("space_id", "") or ""
                source_url = raw_meta.get("source_url", "") or ""
        except Exception:
            # metadata 读取失败时回退空值，保持兼容
            pass

        results.append({
            "success": True,
            "doc_name": doc_name,
            "_dir_info": {
                "doc_name": doc_name,
                "timestamp": timestamp,
                "embeddings_dir": embeddings_dir,
                "metadata_dir": metadata_dir,
                "chunks_dir": chunks_dir,
                "space_id": space_id,
                "source_url": source_url,
            },
        })

    return results


# ==================== 阶段 4: 图谱 ====================

def step_graph(reset_db: bool = False):
    """
    阶段 4: LightRAG 知识图谱构建（支持增量）

    增量逻辑：
      - 首次运行：构建所有文档
      - 后续运行：只构建新增/修改的文档（基于内容哈希）
      - --reset-db：清除图谱缓存后全量重建

    Args:
        reset_db: 是否强制全量重建（清除图谱清单 + LightRAG 工作目录）
    """
    from ..processor.graph_builder import LightRAGGraphBuilder, load_lightrag_config

    lightrag_cfg = load_lightrag_config()

    if not lightrag_cfg.get("enabled", True):
        log.info("图谱未启用（lightrag.enabled=false），跳过")
        return None
    processed_dir = resolve_processed_dir()

    if not processed_dir.exists():
        log.error(f"预处理目录不存在: {processed_dir}")
        return None

    # 检查是否有已处理的文档目录
    doc_dirs = [
        d for d in processed_dir.iterdir()
        if d.is_dir() and (d / "chunks_index.json").exists()
    ]
    if not doc_dirs:
        log.error(f"processed/ 目录中无已处理的文档: {processed_dir}")
        return None

    log.info("=" * 50)
    log.info(f"LightRAG 图谱构建: {processed_dir}")
    log.info(f"文档数: {len(doc_dirs)}")
    if reset_db:
        log.info("⚠ 全量重建模式（清除图谱缓存）")
    log.info("=" * 50)

    builder = LightRAGGraphBuilder()

    # --reset-db: 清除图谱增量清单和 LightRAG 工作目录缓存
    if reset_db:
        working_dir = builder._resolve_working_dir()
        LightRAGGraphBuilder.clear_graph_data(processed_dir, working_dir)

    builder.initialize()

    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = builder.build_from_processed_dir(
        processed_dir,
        batch_timestamp=batch_timestamp,
        force_rebuild=reset_db,
    )

    report_file = processed_dir / "lightrag_report.json"
    atomic_write_json(report_file, report)
    log.info(f"图谱报告已保存（原子写入）: {report_file}")

    return report


# ==================== 环境预检 ====================

def _preflight_check(config: dict, steps: list):
    """
    启动前环境预检（磁盘空间、数据库连接、模型文件）

    遇到致命问题抛出异常，非致命问题仅警告。
    """
    issues_warn = []
    issues_fatal = []

    # ── 1. 磁盘空间检查 ──
    try:
        import shutil as _shutil
        out_dir = config.get("output_dir", ".")
        out_path = Path(out_dir)
        if not out_path.is_absolute():
            from ..core.config import MODULE_DIR
            out_path = (MODULE_DIR / out_path).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        usage = _shutil.disk_usage(str(out_path))
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 1.0:
            issues_fatal.append(f"磁盘空间不足: {free_gb:.1f} GB (最低需要 1GB)")
        elif free_gb < 5.0:
            issues_warn.append(f"磁盘空间较少: {free_gb:.1f} GB (建议 ≥ 5GB)")
        else:
            log.info(f"  ✓ 磁盘空间: {free_gb:.1f} GB")
    except Exception as e:
        issues_warn.append(f"无法检测磁盘空间: {e}")

    # ── 2. 数据库连接检查 ──
    if any(s in steps for s in ("process", "store", "graph")):
        db_cfg = config.get("db")
        if db_cfg:
            try:
                import psycopg
                db_host = db_cfg.get('host', '127.0.0.1')
                db_port = db_cfg.get('port', 5432)
                db_name = db_cfg.get('database', '')
                db_user = db_cfg.get('user', 'postgres')
                db_pass = db_cfg.get('password', '')
                conn_str = (
                    f"host={db_host} "
                    f"port={db_port} "
                    f"dbname={db_name} "
                    f"user={db_user} "
                    f"password={db_pass}"
                )
                with psycopg.connect(conn_str, connect_timeout=5) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                log.info(f"  ✓ 数据库连接: {db_host}:{db_port}/{db_name}")
            except Exception as e:
                issues_fatal.append(f"数据库连接失败: {e}")
        else:
            if any(s in steps for s in ("store", "graph")):
                issues_fatal.append("未配置数据库 (db_info.yml)，无法执行 store/graph 阶段")

    # ── 3. Embedding 模型检查 ──
    if any(s in steps for s in ("process",)):
        from ..processor.onnx_embedder import is_onnx_available
        proc_config = load_processor_config()

        if is_onnx_available(proc_config):
            log.info("  ✓ Embedding 引擎: ONNX Runtime (轻量)")
        else:
            # 检查 HuggingFace 模型
            emb_cfg = proc_config.get("embedding", {})
            hf_cfg = emb_cfg.get("huggingface", {})
            cache = hf_cfg.get("cache_folder", "./models")
            model_name = emb_cfg.get("model", "BAAI/bge-small-zh-v1.5")
            safe_name = model_name.replace("/", "--")
            from ..core.config import MODULE_DIR
            cache_path = Path(cache)
            if not cache_path.is_absolute():
                cache_path = (MODULE_DIR / cache_path).resolve()
            model_dir = cache_path / f"models--{safe_name}"

            try:
                import torch  # noqa: F401
                if model_dir.exists():
                    log.info(f"  ✓ Embedding 引擎: HuggingFace/torch ({model_name})")
                else:
                    issues_warn.append(
                        f"Embedding 模型缓存不存在: {model_dir}\n"
                        f"      首次运行将自动下载（需要网络访问 huggingface.co）"
                    )
            except ImportError:
                issues_fatal.append(
                    "未找到可用的 Embedding 引擎:\n"
                    "  - ONNX 模型不存在 (models/onnx/model.onnx)\n"
                    "  - PyTorch 未安装 (pip install torch)\n"
                    "  请运行 tools/export_onnx.py 导出 ONNX 模型，或安装 torch"
                )

    # ── 输出结果 ──
    for w in issues_warn:
        log.warning(f"  ⚠ {w}")

    if issues_fatal:
        for f in issues_fatal:
            log.error(f"  ✗ {f}")
        log.error("")
        log.error("环境预检失败，请修复上述问题后重试")
        sys.exit(1)

    if not issues_warn and not issues_fatal:
        log.info("  环境预检通过")


# ==================== 全流程编排 ====================

def run_sync(config: dict, full: bool = False, dry_run: bool = False,
             steps: list = None, reset_db: bool = False):
    """
    飞书文档同步与处理

    Args:
        config: 完整配置
        full: 全量同步
        dry_run: 预览模式
        steps: 要执行的阶段列表（None = 全部）
        reset_db: 清空数据库后全量重建
    """
    if steps is None:
        steps = list(STEPS_ALL)

    # --reset-db 隐含 --full（需重新处理所有文档）
    if reset_db:
        full = True

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("=" * 50)
    log.info(f"飞书文档同步  {now}")
    log.info(f"执行阶段: {' → '.join(steps)}")
    if reset_db:
        log.info("⚠ 模式: 清空数据库 → 全量重建")
    log.info("=" * 50)

    # ── 环境预检 ──
    if not dry_run:
        log.info("")
        log.info("环境预检...")
        _preflight_check(config, steps)
        log.info("")

    lock_path = config.get("lock_path")

    if not dry_run:
        if not acquire_lock(lock_path):
            log.warning("有任务在运行，跳过")
            return

    try:
        download_result = None
        process_result = None

        # ── 清空数据库（--reset-db） ──
        if reset_db and not dry_run:
            _reset_all_tables(config)

        # ── 阶段 1: 下载 ──
        if "download" in steps:
            log.info("")
            log.info("▶ 阶段1: 下载文档")
            log.info("=" * 50)
            download_result = step_download(config, full=full, dry_run=dry_run)

            if dry_run:
                return

            if download_result.get("fail"):
                log.warning(f"有 {len(download_result['fail'])} 个文档下载失败")

        # ── 阶段 2: 处理 ──
        if "process" in steps:
            log.info("")
            log.info("▶ 阶段2: 文档处理（拆分 + 向量化）")
            log.info("=" * 50)
            process_result = step_process(config, download_result=download_result,
                                          reset_db=reset_db)

        # ── 阶段 3: 入库 ──
        if "store" in steps:
            log.info("")
            log.info("▶ 阶段3: 批量入库")
            log.info("=" * 50)
            step_store(config, process_result=process_result, reset_db=reset_db)

        # ── 阶段 4: 图谱 ──
        if "graph" in steps:
            log.info("")
            log.info("▶ 阶段4: 知识图谱构建")
            log.info("=" * 50)
            step_graph(reset_db=reset_db)

    except Exception as e:
        log.error(f"同步异常: {e}")
        traceback.print_exc()
        sys.exit(2)
    finally:
        if not dry_run:
            release_lock(lock_path)


# ==================== 工具函数 ====================

def _reset_all_tables(config: dict):
    """清空所有数据库表（doc_chunks + 图谱表），然后重建"""
    log.info("")
    log.info("⚠ 清空数据库...")
    log.info("=" * 50)

    db_cfg = config.get("db")
    if not db_cfg:
        log.warning("未配置数据库，跳过清空")
        return

    # 1) 清空 doc_chunks 表
    from ..processor.storage import PgVectorStorage
    vec_storage = PgVectorStorage(db_config=db_cfg)
    vec_storage.reset_table()
    log.info("  ✓ doc_chunks 表已重建")

    # 2) 清空图谱表
    try:
        from ..processor.graph_builder import PgGraphExporter, load_lightrag_config
        lightrag_cfg = load_lightrag_config()
        pg_export_cfg = lightrag_cfg.get("pg_export", {})
        if pg_export_cfg.get("enabled", True):
            exporter = PgGraphExporter(db_cfg, pg_export_cfg)
            exporter.reset_tables()
            log.info("  ✓ 图谱表已重建")
    except Exception as e:
        log.warning(f"  图谱表重建失败（可忽略）: {e}")

    log.info("数据库已清空，开始全量重建...")
    log.info("")


def _do_reset(config: dict):
    """
    清理所有构建产物（只清理，不重建）

    清理内容：
      1. 数据库表（doc_chunks + lightrag_* 表）
      2. processed/ 目录（拆分/向量化的缓存）
      3. LightRAG 工作目录（图谱缓存）
      4. 图谱增量清单（_graph_manifest.json）
      5. 下载增量清单（manifest.json）
      6. 锁文件
    """
    import shutil
    from ..core.config import RUNTIME_DIR

    log.info("")
    log.info("=" * 50)
    log.info("⚠ 全量清理（Reset）")
    log.info("=" * 50)

    # ── 1. 清空数据库表 ──
    db_cfg = config.get("db")
    if db_cfg:
        try:
            from ..processor.storage import PgVectorStorage
            vec_storage = PgVectorStorage(db_config=db_cfg)
            vec_storage.reset_table()
            log.info("  ✓ doc_chunks 表已清空")
        except Exception as e:
            log.warning(f"  doc_chunks 清空失败: {e}")

        try:
            from ..processor.graph_builder import PgGraphExporter, load_lightrag_config
            lightrag_cfg = load_lightrag_config()
            pg_export_cfg = lightrag_cfg.get("pg_export", {})
            exporter = PgGraphExporter(db_cfg, pg_export_cfg)
            exporter.reset_tables()
            log.info("  ✓ 图谱表已清空")
        except Exception as e:
            log.warning(f"  图谱表清空失败: {e}")
    else:
        log.info("  -- 未配置数据库，跳过")

    # ── 2. 清理 processed/ 目录 ──
    processed_dir = resolve_processed_dir()
    if processed_dir.exists():
        shutil.rmtree(processed_dir, ignore_errors=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"  ✓ processed/ 已清空: {processed_dir}")
    else:
        log.info("  -- processed/ 不存在，跳过")

    # ── 3. 清理 LightRAG 工作目录 ──
    try:
        from ..processor.graph_builder import LightRAGGraphBuilder
        builder = LightRAGGraphBuilder()
        working_dir = builder._resolve_working_dir()
        wd = Path(working_dir)
        if wd.exists():
            shutil.rmtree(wd, ignore_errors=True)
            wd.mkdir(parents=True, exist_ok=True)
            log.info(f"  ✓ LightRAG 工作目录已清空: {wd}")
    except Exception as e:
        log.warning(f"  LightRAG 目录清空失败: {e}")

    # ── 4. 清理下载增量清单 ──
    manifest_path = Path(config.get("manifest_path", RUNTIME_DIR / "manifest.json"))
    if manifest_path.exists():
        manifest_path.unlink()
        log.info(f"  ✓ 下载清单已删除: {manifest_path}")
    else:
        log.info("  -- 下载清单不存在，跳过")

    # ── 5. 清理锁文件 ──
    lock_path = Path(config.get("lock_path", RUNTIME_DIR / ".lock"))
    if lock_path.exists():
        lock_path.unlink()
        log.info(f"  ✓ 锁文件已清除")

    log.info("")
    log.info("✓ 全量清理完成。下次运行将从零开始。")
    log.info("")


_resolve_processed_dir = resolve_processed_dir


# ==================== CLI 入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="飞书文档自动处理管线（四阶段可拆分执行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
阶段说明：
  download  拉取飞书文档（增量下载，构建 manifest）
  process   拆分 chunks + 向量化（生成 metadata / embeddings）
  store     将已处理文档批量写入 pgvector（单事务）
  graph     LightRAG 知识图谱构建

示例:
  run.py                              # 全流程（download→process→store→graph）
  run.py --step download              # 只下载
  run.py --step process               # 只处理（基于已下载文档）
  run.py --step store                 # 只入库（基于 processed/ 目录）
  run.py --step graph                 # 只构建图谱
  run.py --step download,process      # 下载 + 处理（不入库不建图）
  run.py --step download,process,store  # 下载 + 处理 + 入库（不建图）
  run.py --dry-run                    # 预览（列出文档不下载）
  run.py --full                       # 全量同步（忽略增量清单）
  run.py --reset-db                   # 清空数据库 → 全量重建（隐含 --full）
  run.py --reset                      # 只清理（数据库+缓存+清单），不重建
        """,
    )

    parser.add_argument("--step", type=str, default=None,
                        help="指定执行阶段（逗号分隔）: download,process,store,graph")
    parser.add_argument("--full", action="store_true",
                        help="全量同步（忽略增量清单，重新下载所有文档）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式（只列出文档列表，不下载不处理）")
    parser.add_argument("--no-graph", action="store_true",
                        help="跳过图谱构建步骤（等同于去掉 graph 阶段）")
    parser.add_argument("--reset-db", action="store_true",
                        help="清空数据库后全量重建（删除 doc_chunks + 图谱表，然后重新处理并入库）")
    parser.add_argument("--reset", action="store_true",
                        help="只清理不重建（清空数据库 + processed/ + 图谱缓存 + 增量清单）")
    parser.add_argument("--log-level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="指定日志级别（默认从配置文件读取）")

    args = parser.parse_args()

    config = load_full_config()
    log_level = args.log_level or config["log_level"]
    setup_logging(log_level, config["log_dir"])

    # --reset: 只清理，不运行管线
    if args.reset:
        _do_reset(config)
        return

    if not config.get("app_id") or not config.get("app_secret"):
        log.error("缺少飞书凭证 (app_id / app_secret)")
        sys.exit(1)

    # 解析阶段参数
    steps = None
    if args.step:
        steps = [s.strip().lower() for s in args.step.split(",")]
        invalid = [s for s in steps if s not in STEPS_ALL]
        if invalid:
            log.error(f"无效的阶段: {', '.join(invalid)}")
            log.error(f"可选阶段: {', '.join(STEPS_ALL)}")
            sys.exit(1)
    elif args.no_graph:
        steps = ["download", "process", "store"]

    run_sync(
        config,
        full=args.full,
        dry_run=args.dry_run,
        steps=steps,
        reset_db=args.reset_db,
    )


def _entry():
    """包入口（由 run.py 显式调用，不在 import 时自动执行）"""
    try:
        main()
    except ConfigError as e:
        log.error(str(e))
        sys.exit(1)

