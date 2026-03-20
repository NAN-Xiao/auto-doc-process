#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预处理质量分析工具"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List


_WS_RE = re.compile(r"\s+")
_IMAGE_LINE_RE = re.compile(r"^图片：(./images/|\./images/|images/).+$", re.MULTILINE)
_PATH_ONLY_RE = re.compile(r"^(\./)?images/[^\s]+$", re.MULTILINE)


def normalize_text_for_hash(text: str) -> str:
    """标准化文本，保证 hash 在重跑时稳定。"""
    compact = _WS_RE.sub(" ", (text or "")).strip()
    return compact


def compute_chunk_hash(text: str) -> str:
    """基于标准化文本生成稳定 hash。"""
    normalized = normalize_text_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_document_version_hash(chunks: List[Dict[str, Any]]) -> str:
    """基于 chunk hash 生成文档版本 hash。"""
    h = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.get("index", 0)):
        chunk_hash = chunk.get("chunk_hash") or compute_chunk_hash(chunk.get("content", ""))
        h.update(chunk_hash.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _repeat_ratio(text: str) -> float:
    normalized = normalize_text_for_hash(text)
    if not normalized:
        return 0.0
    units = re.findall(r'[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]', normalized)
    if len(units) < 8:
        return 0.0
    unique = len(set(units))
    return max(0.0, 1.0 - unique / max(1, len(units)))


def _whitespace_ratio(raw_text: str) -> float:
    if not raw_text:
        return 0.0
    ws = sum(1 for ch in raw_text if ch.isspace())
    return ws / len(raw_text)


def detect_structured_chunk(text: str) -> bool:
    """粗略判断是否为结构化块（表格/图片增强/列表密集块）。"""
    raw = text or ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return False
    image_lines = sum(1 for line in lines if line.startswith("图片：") or line.startswith("图片标题："))
    tableish_lines = sum(1 for line in lines if "|" in line or "\t" in line or "： " in line)
    return image_lines >= 2 or tableish_lines >= max(2, len(lines) // 3)


def analyze_chunk_quality(text: str) -> Dict[str, Any]:
    """规则化 chunk 质量评分。"""
    raw = text or ""
    normalized = normalize_text_for_hash(raw)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    blank_ratio = _whitespace_ratio(raw)
    repeat_ratio = _repeat_ratio(raw)
    image_line_count = len(_IMAGE_LINE_RE.findall(raw))
    path_only_count = len(_PATH_ONLY_RE.findall(raw))
    title_line_count = sum(
        1 for line in lines
        if re.match(r'^([一二三四五六七八九十]+[，、.]|[0-9]+(\.[0-9]+)*[、.]|#+\s)', line)
    )
    content_line_count = max(0, len(lines) - title_line_count)
    template_hits = sum(
        1 for token in ("点击图片可查看完整电子表格", "图片前文：", "图片后文：", "图片标题：")
        if token in raw
    )

    flags: List[str] = []
    score = 1.0

    if not normalized:
        flags.append("empty")
        score = 0.0
    if blank_ratio > 0.35:
        flags.append("high_whitespace_ratio")
        score -= 0.15
    if repeat_ratio > 0.55:
        flags.append("high_repeat_ratio")
        score -= 0.35
    if image_line_count >= max(2, len(lines)) and content_line_count <= 1:
        flags.append("image_placeholder_dominant")
        score -= 0.35
    if path_only_count > 0:
        flags.append("path_only_lines")
        score -= 0.2
    if title_line_count > 0 and content_line_count == 0:
        flags.append("title_only_chunk")
        score -= 0.2
    if template_hits >= 2 and len(normalized) < 180:
        flags.append("template_text_dominant")
        score -= 0.15

    is_structured = detect_structured_chunk(raw)
    if is_structured:
        flags.append("structured_chunk")

    score = max(0.0, min(1.0, round(score, 4)))
    return {
        "score": score,
        "flags": sorted(set(flags)),
        "blank_ratio": round(blank_ratio, 4),
        "repeat_ratio": round(repeat_ratio, 4),
        "title_line_count": title_line_count,
        "content_line_count": content_line_count,
        "image_line_count": image_line_count,
        "is_structured_chunk": is_structured,
    }


def summarize_document_quality(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总单文档质量。"""
    total = len(chunks)
    token_counts = [int(c.get("chunk_token_count", 0)) for c in chunks if c.get("chunk_token_count") is not None]
    scores = [float(c.get("content_quality_score", 0.0)) for c in chunks if c.get("content_quality_score") is not None]
    low_quality = [c for c in chunks if float(c.get("content_quality_score", 1.0)) < 0.55]
    structured = [c for c in chunks if c.get("is_structured_chunk")]
    with_images = [
        c for c in chunks
        if (c.get("metadata") or {}).get("has_images") or c.get("has_images")
    ]
    return {
        "total_chunks": total,
        "avg_chunk_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        "avg_quality_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "low_quality_chunks": len(low_quality),
        "structured_chunks": len(structured),
        "chunks_with_images": len(with_images),
        "overlong_chunks": sum(1 for t in token_counts if t > 480),
        "short_chunks": sum(1 for t in token_counts if 0 < t < 40),
        "quality_flags": _collect_flag_counts(chunks),
    }


def summarize_batch_quality(doc_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总批次质量报告。"""
    if not doc_summaries:
        return {
            "document_count": 0,
            "chunk_count": 0,
            "avg_chunk_tokens": 0,
            "avg_quality_score": 0.0,
            "low_quality_chunks": 0,
            "structured_chunks": 0,
            "chunks_with_images": 0,
            "overlong_chunks": 0,
            "short_chunks": 0,
            "documents_with_warnings": [],
        }

    chunk_count = sum(item.get("total_chunks", 0) for item in doc_summaries)
    avg_tokens_base = sum(item.get("avg_chunk_tokens", 0) * item.get("total_chunks", 0) for item in doc_summaries)
    avg_quality_base = sum(item.get("avg_quality_score", 0.0) * item.get("total_chunks", 0) for item in doc_summaries)
    warning_docs = [
        item.get("doc_name", "")
        for item in doc_summaries
        if item.get("low_quality_chunks", 0) > 0 or item.get("overlong_chunks", 0) > 0
    ]
    return {
        "document_count": len(doc_summaries),
        "chunk_count": chunk_count,
        "avg_chunk_tokens": round(avg_tokens_base / chunk_count, 2) if chunk_count else 0,
        "avg_quality_score": round(avg_quality_base / chunk_count, 4) if chunk_count else 0.0,
        "low_quality_chunks": sum(item.get("low_quality_chunks", 0) for item in doc_summaries),
        "structured_chunks": sum(item.get("structured_chunks", 0) for item in doc_summaries),
        "chunks_with_images": sum(item.get("chunks_with_images", 0) for item in doc_summaries),
        "overlong_chunks": sum(item.get("overlong_chunks", 0) for item in doc_summaries),
        "short_chunks": sum(item.get("short_chunks", 0) for item in doc_summaries),
        "documents_with_warnings": [doc for doc in warning_docs if doc],
    }


def _collect_flag_counts(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for chunk in chunks:
        flags = chunk.get("quality_flags", []) or (chunk.get("quality", {}) or {}).get("flags", [])
        for flag in flags:
            counts[flag] = counts.get(flag, 0) + 1
    return counts


def to_pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
