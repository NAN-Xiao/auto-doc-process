#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预处理阶段共享的数据结构。"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """文本块数据结构"""
    chunk_id: str
    index: int
    content: str
    char_count: int
    page_number: Optional[int] = None
    metadata: Optional[Dict] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class StructuredElement:
    """结构化文档元素"""
    element_type: str
    text: str = ""
    has_image: bool = False
    is_heading: bool = False
    heading_level: int = 0
    is_list_item: bool = False
    list_level: int = 0
    table_headers: Optional[List[str]] = None
    table_rows: Optional[List[List[str]]] = None
    table_index: int = 0
    caption: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class ContextSegment:
    """上下文片段（带格式信息）"""
    text: str
    distance: int
    is_heading: bool = False
    heading_level: int = 0
    font_size: float = 0

    def to_dict(self):
        return {
            "text": self.text,
            "distance": self.distance,
            "is_heading": self.is_heading,
            "heading_level": self.heading_level,
            "font_size": self.font_size,
        }


@dataclass
class ImageInfo:
    """图片信息"""
    page_number: int
    image_index: int
    position: int
    context_before: str
    context_after: str
    original_filename: str
    smart_filename: str
    context_segments_before: Optional[List[ContextSegment]] = None
    context_segments_after: Optional[List[ContextSegment]] = None
    description: str = ""


@dataclass
class DocumentInfo:
    """文档信息"""
    filename: str
    format: str
    total_pages: int
    total_chunks: int
    total_images: int
    created_at: str
    output_dir: str
    doc_type: str = ""
    profile: Optional[Dict[str, Any]] = None


@dataclass
class DocumentProfile:
    """文档类型画像，用于差异化预处理。"""
    doc_type: str
    total_elements: int
    paragraph_count: int
    non_empty_paragraph_count: int
    heading_count: int
    list_item_count: int
    table_count: int
    image_count: int
    total_text_length: int
    avg_paragraph_length: float
    prompt_repeat_count: int
    link_count: int
    ui_keyword_count: int
    table_text_ratio: float
    image_element_ratio: float

    def to_dict(self):
        return asdict(self)
