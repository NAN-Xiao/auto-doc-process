#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档拆分工具
支持PDF、Word等格式的文档拆分，使用LangChain的智能语义拆分，并提取图片
每个文档生成独立的目录，包含chunks和images子目录
支持使用 LLM 智能生成图片名称
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime
import re

# 导入 LangChain 文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ..core.config import load_processor_config as load_config
from ..core.logger import Logger


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
class ContextSegment:
    """上下文片段（带格式信息）"""
    text: str
    distance: int  # 距离图片的段落数（0=紧邻，1=相隔1段，以此类推）
    is_heading: bool = False  # 是否是标题
    heading_level: int = 0  # 标题级别（1-6，0表示非标题）
    font_size: float = 0  # 字体大小（PDF用）
    
    def to_dict(self):
        return {
            'text': self.text,
            'distance': self.distance,
            'is_heading': self.is_heading,
            'heading_level': self.heading_level,
            'font_size': self.font_size
        }


@dataclass
class ImageInfo:
    """图片信息"""
    page_number: int
    image_index: int
    position: int  # 在文本中的位置
    context_before: str  # 图片前的文字（兼容旧版）
    context_after: str  # 图片后的文字（兼容旧版）
    original_filename: str
    smart_filename: str
    context_segments_before: Optional[List[ContextSegment]] = None  # 图片前的上下文片段列表
    context_segments_after: Optional[List[ContextSegment]] = None  # 图片后的上下文片段列表


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


def generate_smart_image_name_with_llm(
    context_before: str, 
    context_after: str, 
    max_length: int = 10,
    llm=None,
    context_segments_before: Optional[List[ContextSegment]] = None,
    context_segments_after: Optional[List[ContextSegment]] = None
) -> str:
    """
    使用 LLM 根据图片周围的上下文生成智能图片名称
    支持考虑文本格式（标题）和距离权重
    
    Args:
        context_before: 图片前的文字（兼容旧版）
        context_after: 图片后的文字（兼容旧版）
        max_length: 最大长度（中文字符数）
        llm: LLM 实例（如果为 None，则使用简单算法）
        context_segments_before: 图片前的上下文片段列表（带格式信息）
        context_segments_after: 图片后的上下文片段列表（带格式信息）
        
    Returns:
        智能图片名称（不超过max_length个中文字）
    """
    if llm is None:
        # 如果没有 LLM，回退到简单算法
        return generate_smart_image_name_simple(context_before, context_after, max_length)
    
    try:
        # 如果有结构化的上下文片段，使用增强的提示词
        if context_segments_before or context_segments_after:
            # 构建带权重的上下文描述
            context_parts = []
            
            if context_segments_before:
                context_parts.append("【图片前的内容】")
                for seg in reversed(context_segments_before):  # 从最近的开始
                    prefix = ""
                    if seg.is_heading:
                        prefix = f"【标题{seg.heading_level}级】"
                    elif seg.distance == 0:
                        prefix = "【紧邻】"
                    elif seg.distance <= 2:
                        prefix = f"【距离{seg.distance}段】"
                    else:
                        prefix = f"【较远{seg.distance}段】"
                    
                    context_parts.append(f"{prefix} {seg.text}")
            
            if context_segments_after:
                context_parts.append("\n【图片后的内容】")
                for seg in context_segments_after:
                    prefix = ""
                    if seg.is_heading:
                        prefix = f"【标题{seg.heading_level}级】"
                    elif seg.distance == 0:
                        prefix = "【紧邻】"
                    elif seg.distance <= 2:
                        prefix = f"【距离{seg.distance}段】"
                    else:
                        prefix = f"【较远{seg.distance}段】"
                    
                    context_parts.append(f"{prefix} {seg.text}")
            
            context = "\n".join(context_parts)
            
            prompt = f"""请根据以下图片前后的文字内容，为这张图片生成一个简洁、完整、语义清晰的描述性名称。

上下文说明：
- 【紧邻】：表示这段文字紧挨着图片，与图片内容最直接相关，**权重最高**
- 【距离X段】：表示距离图片的段落数，数字越小越接近图片，权重越高
- 【较远X段】：表示距离较远，仅供参考
- 【标题X级】：表示这段文字是标题，提供所属章节/分类信息，仅作为补充参考

命名要求：
1. **只输出图片名称，不要有任何其他文字、解释、标点符号或引号**
2. 名称长度：{max_length}个中文字以内（尽量用足，保证语义完整）
3. **【紧邻】和【距离1-2段】的文字权重最高**，图片名称应主要根据这些紧邻文字来描述图片的具体内容
4. **标题仅作为分类参考**，不要让标题主导命名；如果紧邻文字已经能清晰描述图片，可以不使用标题
5. **必须保证语义完整**：不要出现"副本开启界"这种截断的名称，而应该是"副本开启界面"或"副本开启界面流程图"
6. 优先使用领域关键词：系统架构、流程图、数据模型、界面设计、功能示意图、配置表、关系图等
7. 名称结构建议：[具体内容]+[类型]，例如："装备强化数值配置表"、"角色升级经验曲线图"、"背包系统交互流程图"

{context}

请直接输出完整的图片名称（{max_length}个中文字以内，不要标点符号和引号）："""
        else:
            # 使用简单的上下文（兼容旧版）
            context = f"图片前的内容：{context_before}\n\n图片后的内容：{context_after}"
            
            prompt = f"""请根据以下图片前后的文字内容，为这张图片生成一个简洁、完整、语义清晰的描述性名称。

要求：
1. **只输出图片名称，不要有任何其他文字、解释、标点符号或引号**
2. 名称长度：{max_length}个中文字以内（尽量用足，保证语义完整）
3. **必须保证语义完整**：不要出现截断的名称，如"副本开启界"应该是"副本开启界面"
4. 优先使用文中提到的关键词（如"系统架构"、"流程图"、"数据模型"、"界面设计"等）
5. 名称结构建议：[主题]+[类型]，例如："装备系统流程图"、"角色属性配置表"
6. 如果文中没有明确提示，请根据上下文推测图片主题

{context}

请直接输出完整的图片名称（{max_length}个中文字以内，不要标点符号和引号）："""

        # 调用 LLM（原生 openai SDK）
        _client = llm["client"]
        _resp = _client.chat.completions.create(
            model=llm["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=llm.get("temperature", 0.7),
            max_tokens=llm.get("max_tokens", 100),
            frequency_penalty=llm.get("frequency_penalty", 0.0),
            presence_penalty=llm.get("presence_penalty", 0.0),
        )
        image_name = _resp.choices[0].message.content.strip()
        
        # 清理结果：移除引号、特殊字符、空格等（保证 URL 安全）
        image_name = re.sub(r'["\'\n\r\s，。！？、；：""''（）【】《》<>…·`~!@#$%^&*()\-_=+\[\]{}|\\;:,.<>?/]', '', image_name)
        image_name = image_name.strip()
        
        # 限制长度
        if len(image_name) > max_length:
            image_name = image_name[:max_length]
        
        # 如果结果为空或太短，使用简单算法
        if not image_name or len(image_name) < 2:
            return generate_smart_image_name_simple(context_before, context_after, max_length)
        
        Logger.info(f"LLM生成: {image_name}", indent=2)
        return image_name
        
    except Exception as e:
        Logger.warning(f"LLM生成失败: {e}，使用简单算法", indent=2)
        return generate_smart_image_name_simple(context_before, context_after, max_length)


def generate_smart_image_name_simple(context_before: str, context_after: str,
                                     max_length: int = 10, config: dict = None) -> str:
    """
    使用简单算法根据图片周围的上下文生成智能图片名称（不使用 LLM）
    
    Args:
        context_before: 图片前的文字
        context_after: 图片后的文字
        max_length: 最大长度（中文字符数）
        config: 处理器配置字典（None 则自动加载）
        
    Returns:
        智能图片名称（不超过max_length个中文字）
    """
    # 合并上下文
    context = context_before + " " + context_after
    
    # 清理文本：移除多余空白和特殊字符
    context = re.sub(r'\s+', '', context)
    context = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', context)
    
    if not context:
        return "图片"
    
    # 从配置读取关键词列表
    if config is None:
        config = load_config()
    image_naming_config = config.get('doc_splitter', {}).get('image_naming', {})
    keywords = image_naming_config.get('keywords', [
        '图', '表', '示意', '流程', '结构', '架构', '模型', '界面', '截图'
    ])
    
    # 提取关键词的简单策略：
    # 1. 优先使用"图"、"表"、"示意图"等关键词周围的文字
    for keyword in keywords:
        idx = context.find(keyword)
        if idx != -1:
            # 提取关键词前后的文字
            start = max(0, idx - 3)
            end = min(len(context), idx + max_length)
            name = context[start:end]
            if len(name) <= max_length:
                return name
    
    # 2. 如果没有关键词，取前面的文字
    if len(context_before) > 0:
        context_clean = re.sub(r'\s+', '', context_before)
        if len(context_clean) > 0:
            return context_clean[-max_length:] if len(context_clean) > max_length else context_clean
    
    # 3. 取后面的文字
    if len(context_after) > 0:
        context_clean = re.sub(r'\s+', '', context_after)
        if len(context_clean) > 0:
            return context_clean[:max_length] if len(context_clean) > max_length else context_clean
    
    # 4. 默认名称
    return "图片"


class DocumentSplitter:
    """文档拆分器基类 - 使用LangChain的语义分割"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 use_llm_naming: bool = False, config: dict = None):
        """
        初始化文档拆分器
        
        Args:
            chunk_size: 每个chunk的目标字符数（默认1000，推荐800-1500）
            chunk_overlap: chunk之间的重叠字符数（默认200，推荐15-20%）
            use_llm_naming: 是否使用 LLM 智能生成图片名称（默认False）
            config: 处理器配置字典（None 则自动加载）
        """
        self.use_llm_naming = use_llm_naming
        self.llm = None
        
        # 配置注入：优先使用传入的 config，否则从文件加载
        self.config = config if config is not None else load_config()
        
        # chunk 参数优先从配置读取，构造函数参数作为 fallback
        splitter_cfg = self.config.get('doc_splitter', {}).get('text_splitter', {})
        self.chunk_size = splitter_cfg.get('chunk_size', chunk_size)
        self.chunk_overlap = splitter_cfg.get('chunk_overlap', chunk_overlap)
        
        # 从配置读取图片命名参数
        image_naming_config = self.config.get('doc_splitter', {}).get('image_naming', {})
        self.image_max_length = image_naming_config.get('max_length', 18)
        
        # 如果启用 LLM 命名，初始化 LLM
        if use_llm_naming:
            try:
                deepseek_config = self.config.get('deepseek', {})
                api_key = deepseek_config.get('api_key', '')
                model = deepseek_config.get('default_model', 'deepseek-chat')
                
                llm_config = image_naming_config.get('llm', {})
                
                if api_key:
                    import openai as _openai
                    self.llm = {
                        "client": _openai.OpenAI(
                            api_key=api_key,
                            base_url=llm_config.get('api_base', 'https://api.deepseek.com'),
                        ),
                        "model": model,
                        "temperature": llm_config.get('temperature', 0.7),
                        "max_tokens": llm_config.get('max_tokens', 100),
                        "frequency_penalty": llm_config.get('frequency_penalty', 0.0),
                        "presence_penalty": llm_config.get('presence_penalty', 0.0),
                        }
                    Logger.info(f"LLM智能命名已启用: {model}")
                else:
                    Logger.warning("未配置 DEEPSEEK_API_KEY，使用简单算法命名")
                    self.use_llm_naming = False
            except Exception as e:
                Logger.warning(f"LLM初始化失败: {e}，使用简单算法命名")
                self.use_llm_naming = False
        
        # 直接初始化 LangChain 的递归字符分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=splitter_cfg.get('separators', [
                "\n\n", "\n", "。", "！", "？", "；", "，", " ", ""
            ]),
            length_function=len,
        )
        
        Logger.info(f"语义分割器已初始化: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        使用LangChain的递归语义拆分文本
        
        工作原理：
        1. 优先按段落 (\n\n) 分割，保持段落完整性
        2. 如果段落太长，按句子 (。！？) 分割
        3. 如果句子还太长，按更小的单位递归分割
        4. 智能处理chunk之间的重叠，保证上下文连贯
        
        Args:
            text: 要拆分的文本
            metadata: 可选的元数据
            
        Returns:
            拆分后的文本块列表
        """
        if not text:
            return []
        
        # 创建 Document 对象
        metadata = metadata or {}
        doc = Document(page_content=text, metadata=metadata)
        
        # 使用 LangChain 的递归分割器
        doc_chunks = self.text_splitter.split_documents([doc])
        
        # 提取文本内容
        text_chunks = [chunk.page_content for chunk in doc_chunks]
        
        Logger.info(f"语义拆分: {len(text)} 字符 -> {len(text_chunks)} 个chunks", indent=1)
        
        return text_chunks
    
    def generate_chunk_id(self, content: str, index: int) -> str:
        """生成chunk的唯一ID"""
        hash_obj = hashlib.md5(f"{content[:100]}{index}".encode())
        return hash_obj.hexdigest()[:12]
    
    def _save_chunks_index(self, chunks: List[Chunk], doc_dir: Path):
        """保存chunks索引（仅元数据，不含content）"""
        index_file = doc_dir / "chunks_index.json"
        index_data = {
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "index": chunk.index,
                    "char_count": chunk.char_count,
                    "page_number": chunk.page_number,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ]
        }
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    def _save_doc_info(self, doc_info: DocumentInfo, doc_dir: Path):
        """保存文档信息"""
        info_file = doc_dir / "doc_info.json"
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(doc_info), f, ensure_ascii=False, indent=2)

    def _update_image_placeholders(self, text: str, image_infos: List[ImageInfo]) -> str:
        """
        更新文本中的图片占位符，使用智能文件名和相对路径格式
        
        Args:
            text: 原始文本（包含临时占位符）
            image_infos: 图片信息列表（已包含智能文件名）
            
        Returns:
            更新后的文本
        """
        updated_text = text
        
        for image_info in image_infos:
            # 旧的占位符格式：[图片: images/img_001_image1.png]
            old_placeholder = f"[图片: images/{image_info.original_filename}]"
            
            # 文件名已在生成时做过 URL 安全清洗（无空格、无特殊字符）
            # 新的占位符格式：图片：./images/副本开启界面_001.png
            new_placeholder = f"图片：./images/{image_info.smart_filename}"
            
            # 替换占位符
            updated_text = updated_text.replace(old_placeholder, new_placeholder)
        
        return updated_text


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
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                chunk_id=self.generate_chunk_id(chunk_text, i),
                index=i,
                content=chunk_text,
                char_count=len(chunk_text),
                metadata={"source": pdf_path.name}
            )
            chunks.append(chunk)
            
            # 保存chunk为文本文件
            chunk_file = chunks_dir / f"chunk_{i:04d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk_text)
        
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


class WordSplitter(DocumentSplitter):
    """Word文档拆分器 - 使用语义分割"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 use_llm_naming: bool = False, config: dict = None):
        """
        初始化Word拆分器
        
        Args:
            chunk_size: 每个chunk的字符数（默认1000，适合中文文档）
            chunk_overlap: chunk重叠字符数（默认200，保证上下文连贯）
            use_llm_naming: 是否使用 LLM 智能生成图片名称（默认False）
            config: 处理器配置字典（None 则自动加载）
        """
        super().__init__(chunk_size, chunk_overlap, use_llm_naming, config=config)
        try:
            from docx import Document
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import _Cell, Table
            from docx.text.paragraph import Paragraph
            self.Document = Document
            self.CT_Tbl = CT_Tbl
            self.CT_P = CT_P
        except ImportError as e:
            raise ImportError(f"缺少依赖 python-docx: {e}。请安装: pip install python-docx") from e
    
    def process(self, word_path: Path, output_dir: Path) -> DocumentInfo:
        """
        处理Word文档
        
        Args:
            word_path: Word文件路径
            output_dir: 输出目录（直接在此目录下创建 chunks 和 images）
            
        Returns:
            文档信息
        """
        Logger.info(f"处理Word文档: {word_path.name}")
        
        # 创建输出目录结构
        doc_dir = output_dir
        chunks_dir = doc_dir / "chunks"
        images_dir = doc_dir / "images"
        
        chunks_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载文档（验证文件有效性）
        try:
            doc = self.Document(word_path)
        except Exception as e:
            error_msg = f"无法打开Word文档（可能已损坏或不是有效的 .docx 文件）: {word_path.name}"
            Logger.error(error_msg)
            Logger.error(f"错误详情: {e}", indent=1)
            raise ValueError(error_msg) from e
        
        # 提取文本和图片信息（带占位符）
        full_text, image_infos = self._extract_text_with_image_placeholders(doc, word_path)
        
        # 提取并保存图片（使用智能命名）
        images_count = self._save_word_images_with_smart_names(word_path, images_dir, image_infos, doc_dir)
        
        # 更新文本中的占位符，使用智能文件名和相对路径格式
        full_text = self._update_image_placeholders(full_text, image_infos)
        
        # 拆分为chunks
        text_chunks = self.split_text(full_text)
        
        # 保存chunks
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                chunk_id=self.generate_chunk_id(chunk_text, i),
                index=i,
                content=chunk_text,
                char_count=len(chunk_text),
                metadata={"source": word_path.name}
            )
            chunks.append(chunk)
            
            # 保存chunk为文本文件
            chunk_file = chunks_dir / f"chunk_{i:04d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk_text)
        
        # 保存chunks索引
        self._save_chunks_index(chunks, doc_dir)
        
        # 保存文档信息
        doc_info = DocumentInfo(
            filename=word_path.name,
            format="Word",
            total_pages=len(doc.sections),
            total_chunks=len(chunks),
            total_images=images_count,
            created_at=datetime.now().isoformat(),
            output_dir=str(doc_dir)
        )
        
        self._save_doc_info(doc_info, doc_dir)
        
        Logger.success(f"完成: {len(chunks)} 个chunks, {images_count} 张图片")
        return doc_info
    
    def _extract_text_with_image_placeholders(self, doc, word_path: Path) -> Tuple[str, List[ImageInfo]]:
        """
        提取Word文档文本，并在图片位置插入占位符
        使用两次遍历策略获取更完整的上下文
        增强：识别标题样式
        
        Returns:
            (完整文本, 图片信息列表)
        """
        import zipfile
        
        # 获取所有图片文件名
        image_files = []
        with zipfile.ZipFile(word_path) as docx_zip:
            for file_info in docx_zip.filelist:
                if file_info.filename.startswith('word/media/'):
                    image_files.append(Path(file_info.filename).name)
        
        # 第一步：收集所有段落文本、图片位置和格式信息
        all_elements = []  # (type, content, has_image, is_heading, heading_level)
        
        for element in doc.element.body:
            if isinstance(element, self.CT_P):
                # 检查是否有图片
                element_xml = str(element.xml)
                has_image = ('a:blip' in element_xml or 'blip' in element_xml.lower() or 
                            'v:imagedata' in element_xml or 'imagedata' in element_xml.lower())
                
                para_text = ''.join(element.itertext()).strip()
                
                # 去除段落内的明显重复（如"GVG报名GVG报名GVG报名"）
                para_text = self._remove_internal_duplicates(para_text)
                
                # 识别标题样式
                is_heading = False
                heading_level = 0
                try:
                    # 尝试从段落样式中获取标题信息
                    style_element = element.find('.//w:pStyle', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'})
                    if style_element is not None:
                        style_name = style_element.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val', '')
                        # 检查是否是标题样式
                        if 'Heading' in style_name or '标题' in style_name:
                            is_heading = True
                            # 提取标题级别
                            import re
                            level_match = re.search(r'\d+', style_name)
                            if level_match:
                                heading_level = int(level_match.group())
                            else:
                                heading_level = 1  # 默认为1级标题
                except Exception:
                    pass  # 如果无法获取样式，就当作普通段落
                
                all_elements.append(('paragraph', para_text, has_image, is_heading, heading_level))
                
            elif isinstance(element, self.CT_Tbl):
                # 处理表格
                table_text = self._extract_table_text(element)
                if table_text:
                    all_elements.append(('table', table_text, False, False, 0))
        
        # 从配置读取上下文提取参数
        context_config = self.config.get('doc_splitter', {}).get('context_extraction', {})
        
        lookback = context_config.get('word_lookback_paragraphs', 8)
        lookahead = context_config.get('word_lookahead_paragraphs', 5)
        max_length = context_config.get('max_length', 600)
        
        # 第二步：处理图片，提取更丰富的上下文
        image_infos = []
        image_index = 0
        para_list = []
        current_position = 0
        
        for idx, (elem_type, content, has_image, is_heading, heading_level) in enumerate(all_elements):
            if has_image and image_index < len(image_files):
                # 提取上下文：可配置的段落数
                context_before_parts = []
                context_after_parts = []
                context_segments_before = []  # 新增：带格式的上下文片段
                context_segments_after = []   # 新增：带格式的上下文片段
                
                # 向前看最多 lookback 个有内容的元素
                distance = 0
                for i in range(idx - 1, max(0, idx - lookback) - 1, -1):
                    if all_elements[i][1]:  # 如果有内容
                        elem_text = all_elements[i][1]
                        elem_is_heading = all_elements[i][3]
                        elem_heading_level = all_elements[i][4]
                        
                        context_before_parts.append(elem_text)
                        
                        # 创建上下文片段
                        segment = ContextSegment(
                            text=elem_text,
                            distance=distance,
                            is_heading=elem_is_heading,
                            heading_level=elem_heading_level
                        )
                        context_segments_before.append(segment)
                        distance += 1
                
                # 向后看最多 lookahead 个有内容的元素
                distance = 0
                # 如果当前段落有文本，也加入 context_after
                if content:
                    context_after_parts.insert(0, content)
                    segment = ContextSegment(
                        text=content,
                        distance=0,
                        is_heading=is_heading,
                        heading_level=heading_level
                    )
                    context_segments_after.append(segment)
                    distance = 1
                
                for i in range(idx + 1, min(len(all_elements), idx + lookahead + 1)):
                    if all_elements[i][1]:
                        elem_text = all_elements[i][1]
                        elem_is_heading = all_elements[i][3]
                        elem_heading_level = all_elements[i][4]
                        
                        context_after_parts.append(elem_text)
                        
                        # 创建上下文片段
                        segment = ContextSegment(
                            text=elem_text,
                            distance=distance,
                            is_heading=elem_is_heading,
                            heading_level=elem_heading_level
                        )
                        context_segments_after.append(segment)
                        distance += 1
                
                # 合并上下文（兼容旧版）
                context_before = ' '.join(context_before_parts)
                context_after = ' '.join(context_after_parts)
                
                # 限制上下文长度（可配置）
                context_before = context_before[-max_length:] if len(context_before) > max_length else context_before
                context_after = context_after[:max_length] if len(context_after) > max_length else context_after
                
                # 创建图片信息
                original_filename = f"img_{image_index + 1:03d}_{image_files[image_index]}"
                image_info = ImageInfo(
                    page_number=0,
                    image_index=image_index + 1,
                    position=current_position,
                    context_before=context_before,
                    context_after=context_after,
                    original_filename=original_filename,
                    smart_filename="",
                    context_segments_before=context_segments_before,  # 新增
                    context_segments_after=context_segments_after     # 新增
                )
                image_infos.append(image_info)
                
                # 构建文本
                if content:
                    para_list.append(content)
                    current_position += len(content) + 2
                
                placeholder = f"[图片: images/{original_filename}]"
                para_list.append(placeholder)
                current_position += len(placeholder) + 2
                
                image_index += 1
                
            elif content:
                # 普通段落或表格
                para_list.append(content)
                current_position += len(content) + 2
        
        full_text = "\n\n".join(para_list)
        return full_text, image_infos
    
    def _remove_internal_duplicates(self, text: str) -> str:
        """
        去除文本内部的明显重复
        例如：'GVG报名GVG报名GVG报名' -> 'GVG报名'
        
        Args:
            text: 原始文本
            
        Returns:
            去重后的文本
        """
        # 从配置读取去重参数
        dedup_config = self.config.get('doc_splitter', {}).get('text_deduplication', {})
        
        # 检查是否启用去重
        if not dedup_config.get('enabled', True):
            return text
        
        min_length = dedup_config.get('min_text_length', 10)
        min_pattern = dedup_config.get('min_pattern_length', 3)
        threshold = dedup_config.get('repeat_ratio_threshold', 0.6)
        
        if not text or len(text) < min_length:
            return text
        
        # 策略：检测连续重复的模式
        # 尝试不同的重复长度（从min_pattern到文本长度的1/2）
        for pattern_len in range(min_pattern, len(text) // 2 + 1):
            pattern = text[:pattern_len]
            
            # 检查这个模式是否连续重复
            if len(pattern) >= min_pattern:
                # 计算这个模式连续重复了多少次
                repeat_count = 1
                pos = pattern_len
                
                while pos + pattern_len <= len(text):
                    if text[pos:pos + pattern_len] == pattern:
                        repeat_count += 1
                        pos += pattern_len
                    else:
                        break
                
                # 如果重复了2次或以上，且重复部分占据了超过阈值的文本
                if repeat_count >= 2 and (repeat_count * pattern_len) >= len(text) * threshold:
                    # 保留一次，后面加上剩余的非重复部分
                    remaining = text[repeat_count * pattern_len:]
                    return pattern + remaining
        
        # 没有发现明显重复，返回原文
        return text
    
    def _extract_table_text(self, table_element) -> str:
        """提取表格文本"""
        rows = []
        for row in table_element.findall('.//w:tr', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
            cells = []
            for cell in row.findall('.//w:tc', namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                cell_text = ''.join(cell.itertext())
                cells.append(cell_text.strip())
            if cells:
                rows.append(' | '.join(cells))
        
        return '\n'.join(rows) if rows else ""
    
    def _save_word_images_with_smart_names(self, word_path: Path, images_dir: Path,
                                            image_infos: List[ImageInfo], doc_dir: Path) -> int:
        """
        提取并保存Word文档图片，使用智能命名
        
        Returns:
            图片数量
        """
        import zipfile
        
        image_count = 0
        
        with zipfile.ZipFile(word_path) as docx_zip:
            media_files = [f for f in docx_zip.filelist if f.filename.startswith('word/media/')]
            
            for idx, file_info in enumerate(media_files):
                if idx < len(image_infos):
                    image_info = image_infos[idx]
                    
                    # 提取图片数据
                    image_data = docx_zip.read(file_info.filename)
                    image_filename = Path(file_info.filename).name
                    image_ext = Path(image_filename).suffix
                    
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
                    image_info.smart_filename = f"{smart_name}_{idx + 1:03d}{image_ext}"
                    
                    # 保存图片（使用智能命名）
                    image_path = images_dir / image_info.smart_filename
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    
                    Logger.info(f"保存图片: {image_info.smart_filename}", indent=1)
                    image_count += 1
        
        # 保存图片信息索引
        self._save_images_index(image_infos, doc_dir)
        
        return image_count
    
    def _save_images_index(self, image_infos: List[ImageInfo], doc_dir: Path):
        """保存图片信息索引"""
        index_file = doc_dir / "images_index.json"
        index_data = {
            "total_images": len(image_infos),
            "images": [
                {
                    "page_number": info.page_number if info.page_number > 0 else "N/A",
                    "image_index": info.image_index,
                    "smart_filename": info.smart_filename,
                    "original_filename": info.original_filename,
                    "context_before": info.context_before[:100],
                    "context_after": info.context_after[:100],
                }
                for info in image_infos
            ]
        }
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)


def generate_output_path(input_path: Path, root_dir: str = "processed", 
                        **_kwargs) -> Path:
    """
    生成输出目录路径（增量模式，扁平结构）

    输出结构：processed/{doc_name}/
      - 同名文档再次处理时直接覆盖旧目录
      - 不再按批次时间戳分层
      - doc_name 会做 URL 安全清洗（空格→下划线等）
    
    Args:
        input_path: 输入文件路径
        root_dir: 输出根目录（相对于输入文件目录或绝对路径）
        **_kwargs: 保留旧参数兼容性（batch_timestamp, add_timestamp 等已废弃）
        
    Returns:
        输出目录路径，如 .../processed/文档名/
    """
    from ..core.utils import safe_filename
    doc_name = safe_filename(input_path.stem, input_path.stem)

    if Path(root_dir).is_absolute():
        base_dir = Path(root_dir)
    else:
        base_dir = input_path.parent / root_dir

    return base_dir / doc_name


def process_document(input_path: Path, output_dir: Path = None, chunk_size: int = 1000,
                     chunk_overlap: int = 200, use_llm_naming: bool = False,
                     batch_timestamp: str = None, config: dict = None) -> Optional[DocumentInfo]:
    """
    处理单个文档（使用LangChain语义拆分）
    
    Args:
        input_path: 输入文档路径
        output_dir: 输出目录（可选，不指定则从配置读取）
        chunk_size: chunk大小（默认1000字符，推荐800-1500）
        chunk_overlap: chunk重叠大小（默认200字符，推荐15-20%）
        use_llm_naming: 是否使用 LLM 智能生成图片名称（默认False）
        batch_timestamp: 批次时间戳（同一批次共用；None 则自动生成）
        config: 处理器配置字典（None 则自动加载）
        
    Returns:
        文档信息，失败返回None
    """
    if config is None:
        config = load_config()

    suffix = input_path.suffix.lower()
    
    # 如果没有指定输出目录，从配置中生成
    if output_dir is None:
        output_config = config.get('doc_splitter', {}).get('output', {})
        root_dir = output_config.get('root_dir', 'processed')
        structure = output_config.get('structure', 'nested')
        add_timestamp = output_config.get('add_timestamp', True)
        
        output_dir = generate_output_path(input_path, root_dir, structure, add_timestamp, batch_timestamp=batch_timestamp)
    
    try:
        if suffix == '.pdf':
            splitter = PDFSplitter(chunk_size, chunk_overlap, use_llm_naming, config=config)
            return splitter.process(input_path, output_dir)
        elif suffix in ['.docx', '.doc']:
            if suffix == '.doc':
                Logger.warning(f".doc格式需要转换为.docx，跳过: {input_path.name}")
                return None
            splitter = WordSplitter(chunk_size, chunk_overlap, use_llm_naming, config=config)
            return splitter.process(input_path, output_dir)
        else:
            Logger.warning(f"不支持的文件格式: {suffix}")
            return None
    except ValueError as e:
        # 已经在内部处理过的错误（如损坏的文档），只打印错误信息
        Logger.warning(f"跳过: {input_path.name} - {e}")
        return None
    except Exception as e:
        Logger.error(f"处理失败 {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None



