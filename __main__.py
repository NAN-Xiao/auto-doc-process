#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""兼容入口，实际编排已迁移到 pipeline/。"""

from .pipeline.orchestrator import (
    STEPS_ALL,
    _collect_items_to_process,
    _do_reset,
    _entry,
    _load_results_from_processed_dir,
    _preflight_check,
    _read_jsonl_results,
    _reset_all_tables,
    _resolve_processed_dir,
    _run_process_with_isolation,
    _worker_process_documents,
    main,
    run_sync,
    step_download,
    step_graph,
    step_process,
    step_store,
)

__all__ = [
    "STEPS_ALL",
    "main",
    "_entry",
    "run_sync",
    "step_download",
    "step_process",
    "step_store",
    "step_graph",
    "_collect_items_to_process",
    "_run_process_with_isolation",
    "_read_jsonl_results",
    "_worker_process_documents",
    "_load_results_from_processed_dir",
    "_preflight_check",
    "_reset_all_tables",
    "_do_reset",
    "_resolve_processed_dir",
]
