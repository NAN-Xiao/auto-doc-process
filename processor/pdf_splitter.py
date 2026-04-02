#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PDF 文档拆分实现。"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from ..core.logger import Logger
from .contracts import ContextSegment, DocumentInfo, ImageInfo
from .splitter_base import (
    DocumentSplitter,
    generate_smart_image_name_simple,
    generate_smart_image_name_with_llm,
)


class PDFSplitter(DocumentSplitter):
    """PDF文档拆分器 - 使用语义分割"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 use_llm_naming: bool = False, config: dict = None):
        """
        初始化PDF拆分器
        
        Args:
            chunk_size: 每个chunk的字符数（默认1000，适合中文文档）
            chunk_overlap: chunk重叠字符数（默认200，保证上下文连贯）
            use_llm_naming: 是否使用 LLM 智能生成图片名称（默认False）
            config: 处理器配置字典（None 则自动加载）
        """
        super().__init__(chunk_size, chunk_overlap, use_llm_naming, config=config)
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
        except ImportError as e:
            raise ImportError(f"缺少依赖 PyMuPDF: {e}。请安装: pip install PyMuPDF") from e
    
    def process(self, pdf_path: Path, output_dir: Path) -> DocumentInfo:
        """
        处理PDF文档
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录（直接在此目录下创建 chunks 和 images）
            
        Returns:
            文档信息
        """
        Logger.info(f"处理PDF文档: {pdf_path.name}")
        
        # 创建输出目录结构
        doc_dir = output_dir
        chunks_dir = doc_dir / "chunks"
        images_dir = doc_dir / "images"
        
        chunks_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取文本和图片（带位置信息）
        full_text, image_infos, total_pages = self._extract_text_and_images(pdf_path)
        
        # 提取并保存图片（使用智能命名）
        images_count = self._save_images_with_smart_names(pdf_path, images_dir, image_infos, doc_dir)
        
        # 更新文本中的占位符，使用智能文件名和相对路径格式
        full_text = self._update_image_placeholders(full_text, image_infos)
        
        # 拆分为chunks
        text_chunks = self.split_text(full_text)
        
        # 保存chunks
        chunks = self._build_plain_text_chunks(text_chunks, pdf_path.name)
        for chunk in chunks:
            chunk_file = chunks_dir / f"chunk_{chunk.index:04d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk.content)
        
        # 保存chunks索引
        self._save_chunks_index(chunks, doc_dir)
        
        # 保存文档信息
        doc_info = DocumentInfo(
            filename=pdf_path.name,
            format="PDF",
            total_pages=total_pages,
            total_chunks=len(chunks),
            total_images=images_count,
            created_at=datetime.now().isoformat(),
            output_dir=str(doc_dir)
        )
        
        self._save_doc_info(doc_info, doc_dir)
        
        Logger.success(f"完成: {len(chunks)} 个chunks, {images_count} 张图片")
        return doc_info
    
    def _extract_text_and_images(self, pdf_path: Path) -> Tuple[str, List[ImageInfo], int]:
        """
        提取PDF文本和图片位置信息，在文本中插入图片占位符
        增强：识别可能的标题（基于字体大小）
        
        Returns:
            (完整文本, 图片信息列表, 总页数)
        """
        try:
            doc = self.fitz.open(pdf_path)
        except Exception as e:
            error_msg = f"无法打开PDF文档（可能已损坏或不是有效的 PDF 文件）: {pdf_path.name}"
            Logger.error(error_msg)
            Logger.error(f"错误详情: {e}", indent=1)
            raise ValueError(error_msg) from e
        
        # 第一步：收集所有页面的文本、图片信息和文本块（带字体信息）
        all_pages_data = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            images = page.get_images()
            
            # 获取文本块及字体信息（用于识别标题）
            text_blocks = []
            try:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            max_font_size = 0
                            for span in line["spans"]:
                                line_text += span["text"]
                                max_font_size = max(max_font_size, span.get("size", 0))
                            if line_text.strip():
                                text_blocks.append({
                                    'text': line_text.strip(),
                                    'font_size': max_font_size
                                })
            except Exception:
                # 如果无法获取详细信息，使用简单文本
                text_blocks = [{'text': page_text, 'font_size': 0}]
            
            all_pages_data.append({
                'page_num': page_num,
                'text': page_text,
                'images': images,
                'text_blocks': text_blocks
            })
        
        # 从配置读取上下文提取参数
        context_config = self.config.get('doc_splitter', {}).get('context_extraction', {})
        
        prev_page_len = context_config.get('pdf_prev_page_length', 400)
        next_page_len = context_config.get('pdf_next_page_length', 400)
        max_length = context_config.get('max_length', 600)
        
        # 计算平均字体大小（用于判断标题）
        all_font_sizes = []
        for page_data in all_pages_data:
            for block in page_data.get('text_blocks', []):
                if block['font_size'] > 0:
                    all_font_sizes.append(block['font_size'])
        avg_font_size = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 12
        
        # 第二步：处理图片，提取更丰富的上下文（带格式信息）
        full_text_parts = []
        image_infos = []
        current_position = 0
        
        for page_idx, page_data in enumerate(all_pages_data):
            page_num = page_data['page_num']
            page_text = page_data['text']
            images = page_data['images']
            text_blocks = page_data.get('text_blocks', [])
            
            if not images:
                # 没有图片，直接添加文本
                full_text_parts.append(page_text)
                current_position += len(page_text) + 2
            else:
                # 有图片，需要提取上下文并插入占位符
                page_parts = page_text.split('\n\n')
                
                for img_index, img in enumerate(images):
                    # 获取更丰富的上下文（新增：带格式信息）
                    context_segments_before = []
                    context_segments_after = []
                    
                    # 从前一页和当前页提取上下文
                    distance = 0
                    
                    # 前一页的文本块
                    if page_idx > 0:
                        prev_blocks = all_pages_data[page_idx - 1].get('text_blocks', [])
                        for block in reversed(prev_blocks[-5:]):  # 最多取前一页的最后5个块
                            if block['text']:
                                is_heading = block['font_size'] > avg_font_size * 1.2  # 字体大小超过平均值20%可能是标题
                                heading_level = 1 if block['font_size'] > avg_font_size * 1.5 else 2
                                
                                segment = ContextSegment(
                                    text=block['text'],
                                    distance=distance + 3,  # 前一页的距离较远
                                    is_heading=is_heading,
                                    heading_level=heading_level if is_heading else 0,
                                    font_size=block['font_size']
                                )
                                context_segments_before.insert(0, segment)  # 插入到开头保持顺序
                    
                    # 当前页的前半部分文本块
                    current_blocks = text_blocks[:len(text_blocks)//2]
                    distance = len(current_blocks) - 1
                    for block in current_blocks:
                        if block['text']:
                            is_heading = block['font_size'] > avg_font_size * 1.2
                            heading_level = 1 if block['font_size'] > avg_font_size * 1.5 else 2
                            
                            segment = ContextSegment(
                                text=block['text'],
                                distance=max(0, distance),
                                is_heading=is_heading,
                                heading_level=heading_level if is_heading else 0,
                                font_size=block['font_size']
                            )
                            context_segments_before.append(segment)
                            distance -= 1
                    
                    # 当前页的后半部分文本块
                    distance = 0
                    after_blocks = text_blocks[len(text_blocks)//2:]
                    for block in after_blocks:
                        if block['text']:
                            is_heading = block['font_size'] > avg_font_size * 1.2
                            heading_level = 1 if block['font_size'] > avg_font_size * 1.5 else 2
                            
                            segment = ContextSegment(
                                text=block['text'],
                                distance=distance,
                                is_heading=is_heading,
                                heading_level=heading_level if is_heading else 0,
                                font_size=block['font_size']
                            )
                            context_segments_after.append(segment)
                            distance += 1
                    
                    # 后一页的文本块
                    if page_idx < len(all_pages_data) - 1:
                        next_blocks = all_pages_data[page_idx + 1].get('text_blocks', [])
                        for block in next_blocks[:5]:  # 最多取后一页的前5个块
                            if block['text']:
                                is_heading = block['font_size'] > avg_font_size * 1.2
                                heading_level = 1 if block['font_size'] > avg_font_size * 1.5 else 2
                                
                                segment = ContextSegment(
                                    text=block['text'],
                                    distance=distance + 3,  # 后一页的距离较远
                                    is_heading=is_heading,
                                    heading_level=heading_level if is_heading else 0,
                                    font_size=block['font_size']
                                )
                                context_segments_after.append(segment)
                                distance += 1
                    
                    # 构建简单的上下文字符串（兼容旧版）
                    context_before_parts = []
                    if page_idx > 0:
                        prev_page_text = all_pages_data[page_idx - 1]['text']
                        context_before_parts.append(
                            prev_page_text[-prev_page_len:] if len(prev_page_text) > prev_page_len else prev_page_text
                        )
                    context_before_parts.extend(page_parts[:len(page_parts)//2])
                    context_before = ' '.join(context_before_parts)
                    
                    context_after_parts = page_parts[len(page_parts)//2:]
                    if page_idx < len(all_pages_data) - 1:
                        next_page_text = all_pages_data[page_idx + 1]['text']
                        context_after_parts.append(
                            next_page_text[:next_page_len] if len(next_page_text) > next_page_len else next_page_text
                        )
                    context_after = ' '.join(context_after_parts)
                    
                    # 限制上下文长度（可配置）
                    context_before = context_before[-max_length:] if len(context_before) > max_length else context_before
                    context_after = context_after[:max_length] if len(context_after) > max_length else context_after
                    
                    # 生成图片信息
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_ext = base_image["ext"]
                    original_filename = f"page_{page_num + 1:04d}_img_{img_index + 1:03d}.{image_ext}"
                    
                    image_info = ImageInfo(
                        page_number=page_num + 1,
                        image_index=img_index + 1,
                        position=current_position,
                        context_before=context_before,
                        context_after=context_after,
                        original_filename=original_filename,
                        smart_filename="",  # 稍后生成
                        context_segments_before=context_segments_before,  # 新增
                        context_segments_after=context_segments_after     # 新增
                    )
                    image_infos.append(image_info)
                
                # 在页面文本中插入图片占位符
                # 简单策略：每个图片插入在段落之间
                if len(images) > 0:
                    insert_point = len(page_parts) // 2
                    for img_idx, img_info in enumerate([info for info in image_infos if info.page_number == page_num + 1]):
                        placeholder = f"\n[图片: images/{img_info.original_filename}]\n"
                        if insert_point < len(page_parts):
                            page_parts.insert(insert_point + img_idx, placeholder)
                
                page_text_with_images = '\n\n'.join(page_parts)
                full_text_parts.append(page_text_with_images)
                current_position += len(page_text_with_images) + 2
        
        total_pages = len(doc)
        doc.close()
        
        full_text = "\n\n".join(full_text_parts)
        return full_text, image_infos, total_pages
    
    def _save_images_with_smart_names(self, pdf_path: Path, images_dir: Path, 
                                       image_infos: List[ImageInfo], doc_dir: Path) -> int:
        """
        提取并保存PDF图片，使用智能命名
        
        Returns:
            图片数量
        """
        doc = self.fitz.open(pdf_path)
        image_count = 0
        
        for image_info in image_infos:
            page_num = image_info.page_number - 1
            img_index = image_info.image_index - 1
            
            page = doc[page_num]
            images = page.get_images()
            
            if img_index < len(images):
                img = images[img_index]
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # 生成智能图片名称（使用 LLM 或简单算法）
                if self.use_llm_naming and self.llm:
                    smart_name = generate_smart_image_name_with_llm(
                        image_info.context_before, 
                        image_info.context_after,
                        max_length=self.image_max_length,
                        llm=self.llm,
                        context_segments_before=image_info.context_segments_before,
                        context_segments_after=image_info.context_segments_after
                    )
                else:
                    smart_name = generate_smart_image_name_simple(
                        image_info.context_before, 
                        image_info.context_after,
                        max_length=self.image_max_length
                    )
                
                # 更新图片信息
                image_info.smart_filename = f"{smart_name}_p{image_info.page_number:04d}_{image_info.image_index:03d}.{image_ext}"
                
                # 保存图片（使用智能命名）
                image_path = images_dir / image_info.smart_filename
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                Logger.info(f"保存图片: {image_info.smart_filename}", indent=1)
                image_count += 1
        
        # 保存图片信息索引
        self._save_images_index(image_infos, doc_dir)
        
        doc.close()
        return image_count
    
    def _save_images_index(self, image_infos: List[ImageInfo], doc_dir: Path):
        """保存图片信息索引"""
        index_file = doc_dir / "images_index.json"
        index_data = {
            "total_images": len(image_infos),
            "images": [
                {
                    "page_number": info.page_number,
                    "image_index": info.image_index,
                    "smart_filename": info.smart_filename,
                    "original_filename": info.original_filename,
                    "context_before": info.context_before[:100],  # 只保存前100字符
                    "context_after": info.context_after[:100],
                    # 新增：保存格式化的上下文片段
                    "context_segments_before": [seg.to_dict() for seg in info.context_segments_before] if info.context_segments_before else [],
                    "context_segments_after": [seg.to_dict() for seg in info.context_segments_after] if info.context_segments_after else [],
                }
                for info in image_infos
            ]
        }
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)




