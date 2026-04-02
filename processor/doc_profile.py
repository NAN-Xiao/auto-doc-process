#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""docx 文档类型画像与轻量归一逻辑。"""

import re
from typing import List

from .contracts import DocumentProfile, StructuredElement


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def classify_word_document(elements: List[StructuredElement]) -> DocumentProfile:
    """基于 Word 元素结构识别文档类型。"""
    prompt_patterns = (
        "点击图片可查看完整电子表格",
        "点击图片查看",
        "完整电子表格查看",
    )
    ui_patterns = ("UI", "key", "Key", "browse/", "http://", "https://")

    paragraphs = [e for e in elements if e.element_type == "paragraph"]
    non_empty_paragraphs = [e for e in paragraphs if (e.text or "").strip()]
    tables = [e for e in elements if e.element_type == "table"]
    total_text_length = sum(len((e.text or "").strip()) for e in non_empty_paragraphs)
    avg_paragraph_length = round(total_text_length / len(non_empty_paragraphs), 2) if non_empty_paragraphs else 0.0
    prompt_repeat_count = 0
    link_count = 0
    ui_keyword_count = 0
    table_text_length = sum(len((e.text or "").strip()) for e in tables)

    for element in non_empty_paragraphs:
        text = element.text or ""
        prompt_repeat_count += sum(text.count(pattern) for pattern in prompt_patterns)
        link_count += len(re.findall(r"https?://|browse/", text, flags=re.IGNORECASE))
        ui_keyword_count += sum(text.count(pattern) for pattern in ui_patterns)

    image_count = sum(1 for e in elements if e.has_image) + sum(
        1 for e in elements if (e.text or "").startswith("[图片:")
    )
    total_elements = len(elements)
    table_text_ratio = round(table_text_length / max(1, (table_text_length + total_text_length)), 4)
    image_element_ratio = round(image_count / max(1, total_elements), 4)

    doc_type = "article"
    if image_count >= 5 and (total_text_length < 1500 or prompt_repeat_count >= 4):
        doc_type = "image_heavy"
    elif len(tables) >= 3 and table_text_ratio >= 0.4:
        doc_type = "table_heavy"
    elif (
        len([e for e in non_empty_paragraphs if e.is_heading]) >= 3
        or len([e for e in non_empty_paragraphs if e.is_list_item]) >= 5
        or link_count >= 2
        or ui_keyword_count >= 3
    ):
        doc_type = "spec_mixed"

    return DocumentProfile(
        doc_type=doc_type,
        total_elements=total_elements,
        paragraph_count=len(paragraphs),
        non_empty_paragraph_count=len(non_empty_paragraphs),
        heading_count=sum(1 for e in non_empty_paragraphs if e.is_heading),
        list_item_count=sum(1 for e in non_empty_paragraphs if e.is_list_item),
        table_count=len(tables),
        image_count=image_count,
        total_text_length=total_text_length + table_text_length,
        avg_paragraph_length=avg_paragraph_length,
        prompt_repeat_count=prompt_repeat_count,
        link_count=link_count,
        ui_keyword_count=ui_keyword_count,
        table_text_ratio=table_text_ratio,
        image_element_ratio=image_element_ratio,
    )


def normalize_elements_by_doc_type(elements: List[StructuredElement],
                                   doc_profile: DocumentProfile) -> List[StructuredElement]:
    """按文档类型对结构化元素做轻量归一。"""
    if doc_profile.doc_type != "image_heavy":
        return elements

    normalized: List[StructuredElement] = []
    seen_prompts = set()
    for element in elements:
        text = _compact_whitespace(element.text)
        if not text:
            normalized.append(element)
            continue

        if text.startswith("图片："):
            normalized.append(element)
            continue

        if text == "点击图片可查看完整电子表格":
            if text in seen_prompts:
                continue
            seen_prompts.add(text)

        normalized.append(
            StructuredElement(
                element_type=element.element_type,
                text=text,
                has_image=element.has_image,
                is_heading=element.is_heading,
                heading_level=element.heading_level,
                is_list_item=element.is_list_item,
                list_level=element.list_level,
                table_headers=element.table_headers,
                table_rows=element.table_rows,
                table_index=element.table_index,
                caption=element.caption,
            )
        )
    return normalized
