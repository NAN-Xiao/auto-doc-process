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
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import asdict
import hashlib
from datetime import datetime
import re

# 导入 LangChain 文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ..core.config import load_processor_config as load_config
from ..core.logger import Logger
from .contracts import (
    Chunk,
    ContextSegment,
    DocumentInfo,
    DocumentProfile,
    ImageInfo,
    StructuredElement,
)
from .doc_profile import classify_word_document, normalize_elements_by_doc_type


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
        self._token_length: Callable[[str], int] = len
        
        # 配置注入：优先使用传入的 config，否则从文件加载
        self.config = config if config is not None else load_config()
        
        # chunk 参数优先从配置读取，构造函数参数作为 fallback
        splitter_cfg = self.config.get('doc_splitter', {}).get('text_splitter', {})
        configured_chunk_size = splitter_cfg.get('chunk_size', chunk_size)
        configured_chunk_overlap = splitter_cfg.get('chunk_overlap', chunk_overlap)
        
        # 从配置读取图片命名参数
        image_naming_config = self.config.get('doc_splitter', {}).get('image_naming', {})
        self.image_max_length = image_naming_config.get('max_length', 18)

        # 优先按 embedding tokenizer 的 token 长度切分，避免向量化阶段被截断
        token_length, safe_chunk_size, safe_chunk_overlap = self._build_length_function(
            configured_chunk_size, configured_chunk_overlap
        )
        self._token_length = token_length
        self.chunk_size = safe_chunk_size
        self.chunk_overlap = safe_chunk_overlap
        
        # 如果启用 LLM 命名，初始化 LLM
        if use_llm_naming:
            try:
                # 优先从新的 llm 配置块读取，回退到旧的 deepseek 配置（向后兼容）
                llm_config = self.config.get('llm', {})
                deepseek_config = self.config.get('deepseek', {})
                api_key = llm_config.get('api_key') or deepseek_config.get('api_key', '')
                model = llm_config.get('model') or llm_config.get('default_model') or deepseek_config.get('default_model', 'deepseek-chat')

                image_llm_config = image_naming_config.get('llm', {})
                if api_key:
                    import openai as _openai
                    self.llm = {
                        "client": _openai.OpenAI(
                            api_key=api_key,
                            base_url=llm_config.get('api_base', image_llm_config.get('api_base', 'https://api.deepseek.com')),
                        ),
                        "model": model,
                        "temperature": image_naming_config.get('temperature') or llm_config.get('temperature') or image_llm_config.get('temperature', 0.7),
                        "max_tokens": image_naming_config.get('max_tokens') or llm_config.get('max_tokens') or image_llm_config.get('max_tokens', 100),
                        "frequency_penalty": image_naming_config.get('frequency_penalty') or llm_config.get('frequency_penalty') or image_llm_config.get('frequency_penalty', 0.0),
                        "presence_penalty": image_naming_config.get('presence_penalty') or llm_config.get('presence_penalty') or image_llm_config.get('presence_penalty', 0.0),
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
            length_function=self._token_length,
        )
        
        Logger.info(f"语义分割器已初始化: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def _build_length_function(self, configured_chunk_size: int,
                               configured_chunk_overlap: int) -> Tuple[Callable[[str], int], int, int]:
        """
        构建长度函数和安全的 chunk 参数。

        优先使用 embedding tokenizer 统计 token 数，并把 chunk 大小限制在
        embedding 模型最大 token 长度以内，避免向量化阶段截断。
        """
        max_tokens = 512
        reserve_tokens = 32

        try:
            from .onnx_embedder import _resolve_onnx_dir
            onnx_dir = _resolve_onnx_dir(self.config)
            meta_file = onnx_dir / "model_meta.json"
            tokenizer_path = onnx_dir / "tokenizer.json"

            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    model_meta = json.load(f)
                max_tokens = int(model_meta.get("max_length", max_tokens))

            if tokenizer_path.exists():
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(tokenizer_path))

                def token_length(text: str) -> int:
                    if not text:
                        return 0
                    return len(tokenizer.encode(text).ids)

                safe_chunk_size = min(int(configured_chunk_size), max_tokens - reserve_tokens)
                if safe_chunk_size <= 0:
                    safe_chunk_size = max(128, max_tokens // 2)
                safe_chunk_overlap = min(int(configured_chunk_overlap), max(0, safe_chunk_size // 4))

                if configured_chunk_size != safe_chunk_size:
                    Logger.warning(
                        f"chunk_size={configured_chunk_size} 超过 embedding 安全上限，已自动调整为 {safe_chunk_size} tokens",
                        indent=1,
                    )
                if configured_chunk_overlap != safe_chunk_overlap:
                    Logger.warning(
                        f"chunk_overlap={configured_chunk_overlap} 过大，已自动调整为 {safe_chunk_overlap} tokens",
                        indent=1,
                    )

                return token_length, safe_chunk_size, safe_chunk_overlap
        except Exception as e:
            Logger.warning(f"token 长度函数初始化失败，回退到字符长度切分: {e}", indent=1)

        safe_chunk_overlap = min(int(configured_chunk_overlap), max(0, int(configured_chunk_size) // 4))
        return len, int(configured_chunk_size), safe_chunk_overlap
    
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
    
    def _compact_whitespace(self, text: str) -> str:
        """压缩空白，保证稳定 hash。"""
        return re.sub(r"\s+", " ", (text or "")).strip()

    def generate_source_doc_id(self, source_name: str, source_hint: str = "") -> str:
        """生成稳定的文档来源 ID。"""
        stable_source = source_hint or source_name
        digest = hashlib.sha1(self._compact_whitespace(stable_source).lower().encode("utf-8")).hexdigest()
        return digest[:20]

    def generate_chunk_id(self, content: str, index: int, metadata: Optional[Dict] = None) -> str:
        """生成稳定的 chunk ID，优先使用结构锚点而不是顺序。"""
        metadata = metadata or {}
        anchor_parts = [
            metadata.get("source_doc_id", ""),
            metadata.get("content_type", ""),
            metadata.get("section_path", ""),
            metadata.get("title", ""),
            str(metadata.get("table_index", "")),
            json.dumps(metadata.get("row_data", {}), ensure_ascii=False, sort_keys=True),
            self._compact_whitespace(content),
        ]
        anchor = "\n".join(part for part in anchor_parts if part)
        if not anchor:
            anchor = f"{self._compact_whitespace(content)}\n{index}"
        return hashlib.sha1(anchor.encode("utf-8")).hexdigest()[:16]
    
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

    def _sanitize_metadata_keywords(self, *parts: Any) -> List[str]:
        """从多个文本片段提取可用于检索过滤的关键词。"""
        seen = set()
        keywords: List[str] = []
        for part in parts:
            if part is None:
                continue
            if isinstance(part, dict):
                values = part.values()
            elif isinstance(part, (list, tuple, set)):
                values = part
            else:
                values = [part]

            for value in values:
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                for token in re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9_][A-Za-z0-9_\-/]{1,}', text):
                    normalized = token.strip().lower()
                    if len(normalized) < 2 or normalized in seen:
                        continue
                    seen.add(normalized)
                    keywords.append(token[:64])
        return keywords[:64]

    def _looks_like_heading_text(self, text: str) -> Tuple[bool, int]:
        """基于文本形态识别可能的标题。"""
        candidate = self._compact_whitespace(text)
        if not candidate:
            return False, 0
        if len(candidate) > 60:
            return False, 0

        heading_patterns = [
            (r'^(第[一二三四五六七八九十百零0-9]+[章节篇部卷])', 1),
            (r'^[一二三四五六七八九十]+[、.．]\s*', 2),
            (r'^[0-9]+(\.[0-9]+){0,3}\s+', 2),
            (r'^[（(]?[一二三四五六七八九十0-9]+[)）]\s*', 3),
            (r'^(附录|说明|备注|规则|流程|配置|概述|目标)[：: ]?$', 2),
        ]
        for pattern, level in heading_patterns:
            if re.match(pattern, candidate):
                return True, level

        if len(candidate) <= 18 and not re.search(r'[。！？；,.!?;]', candidate):
            return True, 3
        return False, 0

    def _extract_word_paragraph_structure(self, paragraph, para_text: str) -> Tuple[bool, int, bool, int]:
        """识别 Word 段落的标题和列表层级。"""
        is_heading = False
        heading_level = 0
        is_list_item = False
        list_level = 0
        style_name = ""

        try:
            style_name = getattr(paragraph.style, 'name', '') if paragraph.style else ''
            if style_name:
                if 'Heading' in style_name or '标题' in style_name:
                    is_heading = True
                    level_match = re.search(r'\d+', style_name)
                    heading_level = int(level_match.group()) if level_match else 1
                if any(token in style_name for token in ('List', '列表', 'Bullet', '项目符号', '编号')):
                    is_list_item = True
        except Exception:
            pass

        try:
            ppr = getattr(getattr(paragraph, '_p', None), 'pPr', None)
            if ppr is not None and getattr(ppr, 'numPr', None) is not None:
                is_list_item = True
                ilvl = getattr(ppr.numPr, 'ilvl', None)
                if ilvl is not None and getattr(ilvl, 'val', None) is not None:
                    list_level = int(ilvl.val) + 1
        except Exception:
            pass

        heading_guess, guessed_level = self._looks_like_heading_text(para_text)
        if heading_guess and not is_list_item and not is_heading:
            is_heading = True
            heading_level = guessed_level or 3

        if not is_list_item and re.match(r'^([-*•·]|[0-9]+[.)、]|[A-Za-z][.)])\s+', para_text or ''):
            is_list_item = True
            list_level = max(list_level, 1)

        return is_heading, heading_level, is_list_item, list_level

    def _normalize_table_rows(self, rows: List[List[str]]) -> List[List[str]]:
        """归一化表格行长度，并尽量补齐合并单元格带来的空洞。"""
        if not rows:
            return rows
        max_cols = max(len(row) for row in rows)
        normalized: List[List[str]] = []
        vertical_memory = [""] * max_cols

        for row in rows:
            padded = list(row) + [""] * (max_cols - len(row))
            for idx, cell in enumerate(padded):
                text = self._compact_whitespace(cell)
                if text:
                    vertical_memory[idx] = text
                    padded[idx] = text
                elif vertical_memory[idx]:
                    padded[idx] = vertical_memory[idx]
            normalized.append(padded)
        return normalized

    def _infer_table_header_depth(self, rows: List[List[str]]) -> int:
        """推断多级表头深度。"""
        if not rows:
            return 0
        header_depth = 0
        for row in rows[:3]:
            non_empty = [cell for cell in row if cell]
            if not non_empty:
                continue
            numeric_like = sum(1 for cell in non_empty if re.search(r'\d', cell))
            if header_depth == 0 or numeric_like <= max(1, len(non_empty) // 3):
                header_depth += 1
            else:
                break
        return min(max(header_depth, 1), min(3, len(rows)))

    def _merge_table_headers(self, header_rows: List[List[str]]) -> List[str]:
        """将多级表头合并成单层表头。"""
        if not header_rows:
            return []
        col_count = max(len(row) for row in header_rows)
        merged_headers: List[str] = []
        for col_idx in range(col_count):
            parts: List[str] = []
            for row in header_rows:
                if col_idx >= len(row):
                    continue
                cell = self._compact_whitespace(row[col_idx])
                if cell and cell not in parts:
                    parts.append(cell)
            merged_headers.append(" / ".join(parts) if parts else f"col_{col_idx + 1}")
        return merged_headers

    def _build_chunk(self, content: str, index: int, metadata: Optional[Dict] = None,
                     page_number: Optional[int] = None) -> Chunk:
        """统一构建 chunk，保证 metadata 结构稳定。"""
        metadata = dict(metadata or {})
        metadata.setdefault("content_type", "section")
        metadata.setdefault("section_path", "")
        metadata.setdefault("title", "")
        metadata.setdefault("parent_chunk_id", "")
        metadata.setdefault("keywords", [])
        metadata.setdefault("table_headers", [])
        metadata.setdefault("row_data", {})
        metadata.setdefault("source_doc_id", self.generate_source_doc_id(
            metadata.get("source", ""), metadata.get("source_hint", "")
        ))

        return Chunk(
            chunk_id=self.generate_chunk_id(content, index, metadata),
            index=index,
            content=content,
            char_count=len(content),
            page_number=page_number,
            metadata=metadata,
        )

    def _build_plain_text_chunks(self, text_chunks: List[str], source_name: str) -> List[Chunk]:
        """为非结构化来源构建默认 section chunks。"""
        chunks: List[Chunk] = []
        for i, chunk_text in enumerate(text_chunks):
            keywords = self._sanitize_metadata_keywords(chunk_text[:200])
            chunk = self._build_chunk(
                content=chunk_text,
                index=i,
                metadata={
                    "source": source_name,
                    "source_hint": source_name,
                    "content_type": "section",
                    "title": "",
                    "section_path": "",
                    "keywords": keywords,
                }
            )
            chunks.append(chunk)
        return chunks

    def _flush_section_buffer(self, source_name: str, section_lines: List[str], section_stack: List[str],
                              current_title: str, chunks: List[Chunk], chunk_index: int,
                              doc_type: str = "article") -> int:
        """将正文缓存刷新为 section chunk。"""
        section_text = "\n\n".join(line for line in section_lines if line and line.strip()).strip()
        if not section_text:
            return chunk_index

        section_path = " > ".join(section_stack)
        text_chunks = self.split_text(section_text, metadata={"section_path": section_path})
        for chunk_text in text_chunks:
            metadata = {
                "source": source_name,
                "source_hint": source_name,
                "content_type": "section",
                "title": current_title,
                "section_path": section_path,
                "doc_type": doc_type,
                "keywords": self._sanitize_metadata_keywords(current_title, section_path, chunk_text[:240]),
            }
            chunks.append(self._build_chunk(chunk_text, chunk_index, metadata=metadata))
            chunk_index += 1
        return chunk_index

    def _serialize_row_payload(self, headers: List[str], row: List[str]) -> Dict[str, str]:
        """将表格行转换为稳定的 key-value 结构。"""
        payload: Dict[str, str] = {}
        max_len = max(len(headers), len(row))
        for idx in range(max_len):
            header = headers[idx] if idx < len(headers) and headers[idx] else f"col_{idx + 1}"
            value = row[idx] if idx < len(row) else ""
            payload[header.strip() or f"col_{idx + 1}"] = value.strip()
        return payload

    def _build_table_summary_text(self, table_index: int, caption: str, section_path: str,
                                  headers: List[str], rows: List[List[str]]) -> str:
        """构建表格摘要文本，供语义召回使用。"""
        summary_lines = [
            f"表格编号：{table_index}",
        ]
        if caption:
            summary_lines.append(f"表格主题：{caption}")
        if section_path:
            summary_lines.append(f"所属章节：{section_path}")
        if headers:
            summary_lines.append(f"表头：{' | '.join(headers)}")
        summary_lines.append(f"数据行数：{len(rows)}")
        preview_rows = []
        for row in rows[:3]:
            preview_rows.append(" | ".join(cell for cell in row if cell))
        if preview_rows:
            summary_lines.append("示例行：")
            summary_lines.extend(preview_rows)
        return "\n".join(summary_lines)

    def _build_structured_word_chunks(self, source_name: str, elements: List[StructuredElement],
                                      doc_profile: Optional[DocumentProfile] = None) -> List[Chunk]:
        """基于标题层级和表格结构为 Word 文档生成召回友好的 chunks。"""
        chunks: List[Chunk] = []
        section_stack: List[str] = []
        section_lines: List[str] = []
        current_title = ""
        chunk_index = 0
        last_text_for_caption = ""
        doc_profile = doc_profile or DocumentProfile(
            doc_type="article",
            total_elements=len(elements),
            paragraph_count=0,
            non_empty_paragraph_count=0,
            heading_count=0,
            list_item_count=0,
            table_count=0,
            image_count=0,
            total_text_length=0,
            avg_paragraph_length=0.0,
            prompt_repeat_count=0,
            link_count=0,
            ui_keyword_count=0,
            table_text_ratio=0.0,
            image_element_ratio=0.0,
        )

        retrieval_cfg = self.config.get("doc_splitter", {}).get("retrieval", {})
        table_row_window = max(1, int(retrieval_cfg.get("table_row_window", 1)))

        for element in elements:
            text = (element.text or "").strip()

            if element.is_heading and text:
                chunk_index = self._flush_section_buffer(
                    source_name, section_lines, section_stack, current_title, chunks, chunk_index,
                    doc_profile.doc_type
                )
                section_lines = []
                level = max(1, int(element.heading_level or 1))
                while len(section_stack) >= level:
                    section_stack.pop()
                section_stack.append(text)
                current_title = text
                section_lines.append(text)
                last_text_for_caption = text
                continue

            if element.is_list_item and text:
                list_prefix = "  " * max(0, element.list_level - 1)
                section_lines.append(f"{list_prefix}- {text}")
                last_text_for_caption = text
                continue

            if element.element_type == "table":
                chunk_index = self._flush_section_buffer(
                    source_name, section_lines, section_stack, current_title, chunks, chunk_index,
                    doc_profile.doc_type
                )
                section_lines = []

                headers = [header for header in (element.table_headers or []) if header]
                rows = element.table_rows or []
                section_path = " > ".join(section_stack)
                caption = element.caption or last_text_for_caption or current_title or f"表格{element.table_index}"
                summary_text = self._build_table_summary_text(
                    element.table_index, caption, section_path, headers, rows
                )
                summary_metadata = {
                    "source": source_name,
                "source_hint": source_name,
                "content_type": "table_summary",
                "title": caption,
                "section_path": section_path,
                "doc_type": doc_profile.doc_type,
                "table_index": element.table_index,
                "table_headers": headers,
                "row_data": {},
                    "keywords": self._sanitize_metadata_keywords(caption, section_path, headers),
                }
                summary_chunk = self._build_chunk(summary_text, chunk_index, metadata=summary_metadata)
                chunks.append(summary_chunk)
                chunk_index += 1

                for row_start in range(0, len(rows), table_row_window):
                    row_group = rows[row_start:row_start + table_row_window]
                    row_payloads = [self._serialize_row_payload(headers, row) for row in row_group]
                    row_lines = [
                        f"表格主题：{caption}",
                        f"所属章节：{section_path}" if section_path else "",
                        f"表头：{' | '.join(headers)}" if headers else "",
                        f"行范围：{row_start + 1}-{row_start + len(row_group)}",
                    ]
                    for payload in row_payloads:
                        row_lines.append(
                            " | ".join(f"{key}={value}" for key, value in payload.items())
                        )
                    row_text = "\n".join(line for line in row_lines if line)
                    row_metadata = {
                        "source": source_name,
                        "source_hint": source_name,
                        "content_type": "table_row",
                        "title": caption,
                        "section_path": section_path,
                        "doc_type": doc_profile.doc_type,
                        "parent_chunk_id": summary_chunk.chunk_id,
                        "table_index": element.table_index,
                        "table_headers": headers,
                        "row_data": row_payloads[0] if len(row_payloads) == 1 else {"rows": row_payloads},
                        "keywords": self._sanitize_metadata_keywords(caption, section_path, headers, row_payloads),
                    }
                    chunks.append(self._build_chunk(row_text, chunk_index, metadata=row_metadata))
                    chunk_index += 1

                continue

            if text:
                section_lines.append(text)
                last_text_for_caption = text

        self._flush_section_buffer(
            source_name, section_lines, section_stack, current_title, chunks, chunk_index,
            doc_profile.doc_type
        )
        return chunks

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

            image_title = Path(image_info.smart_filename).stem
            image_title = re.sub(r'_(p\d{4}_\d{3}|\d{3})$', '', image_title)

            context_parts = []
            if image_info.context_before:
                context_parts.append(f"图片前文：{self._compact_text(image_info.context_before, 80)}")
            if image_info.context_after:
                context_parts.append(f"图片后文：{self._compact_text(image_info.context_after, 80)}")

            placeholder_lines = [
                f"图片：./images/{image_info.smart_filename}",
                f"图片标题：{image_title}",
            ]
            placeholder_lines.extend(context_parts)
            new_placeholder = "\n".join(placeholder_lines)
            
            # 替换占位符
            updated_text = updated_text.replace(old_placeholder, new_placeholder)
        
        return updated_text

    def _replace_element_image_placeholders(self, elements: List[StructuredElement],
                                            image_infos: List[ImageInfo]) -> List[StructuredElement]:
        """将结构化元素中的图片占位符更新为最终图片路径。"""
        updated_elements: List[StructuredElement] = []
        for element in elements:
            updated_text = self._update_image_placeholders(element.text, image_infos) if element.text else element.text
            updated_elements.append(
                StructuredElement(
                    element_type=element.element_type,
                    text=updated_text,
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
        return updated_elements

    def _compact_text(self, text: str, max_length: int = 120) -> str:
        """压缩文本中的空白和连续重复片段，适合作为图片上下文摘要。"""
        compacted = re.sub(r'\s+', ' ', text).strip()
        compacted = self._deduplicate_text(compacted)
        if len(compacted) > max_length:
            return compacted[:max_length].rstrip()
        return compacted

    def _deduplicate_text(self, text: str) -> str:
        """
        去除文本内部的明显重复
        例如：'GVG报名GVG报名GVG报名' -> 'GVG报名'
        """
        dedup_config = self.config.get('doc_splitter', {}).get('text_deduplication', {})
        if not dedup_config.get('enabled', True):
            return text

        min_length = dedup_config.get('min_text_length', 10)
        min_pattern = dedup_config.get('min_pattern_length', 3)
        threshold = dedup_config.get('repeat_ratio_threshold', 0.6)

        if not text or len(text) < min_length:
            return text

        cleaned = re.sub(r'\s+', ' ', text).strip()

        for pattern_len in range(min_pattern, min(len(cleaned) // 2 + 1, 40)):
            pattern = cleaned[:pattern_len]
            if len(pattern) < min_pattern:
                continue

            repeat_count = 1
            pos = pattern_len
            while pos + pattern_len <= len(cleaned):
                if cleaned[pos:pos + pattern_len] == pattern:
                    repeat_count += 1
                    pos += pattern_len
                else:
                    break

            if repeat_count >= 2 and (repeat_count * pattern_len) >= len(cleaned) * threshold:
                remaining = cleaned[repeat_count * pattern_len:]
                cleaned = pattern + remaining
                break

        changed = True
        while changed:
            changed = False
            for pattern_len in range(min_pattern, min(len(cleaned) // 2 + 1, 24)):
                pattern = re.compile(rf'(.{{{pattern_len}}})(\1{{1,}})')
                updated = pattern.sub(r'\1', cleaned)
                if updated != cleaned:
                    cleaned = updated
                    changed = True

        return cleaned



