#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从已处理的文档生成 Embeddings

功能：
1. 扫描 processed 目录
2. 读取 chunks 和图片信息
3. 生成 embeddings
4. 创建包含图片占位符的 metadata
5. 保存 embeddings 到文件（JSON格式）

注意：此脚本只生成 embeddings，不直接存储到向量数据库
使用 vector_storage.py 将 embeddings 存储到 PostgreSQL (pgvector)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from ..core.config import load_processor_config as load_config
from ..core.logger import Logger
from .onnx_embedder import create_embeddings
from .quality import analyze_chunk_quality, compute_chunk_hash


class EmbeddingGenerator:
    """Embedding 生成器"""
    
    def __init__(self, config: dict = None):
        """
        初始化

        Args:
            config: 处理器配置字典（None 则自动加载）
        """
        # 配置注入：优先使用传入的 config
        self.config = config if config is not None else load_config()
        
        # 初始化 embedding 引擎（自动选择 ONNX 或 torch 后端）
        self.embeddings = create_embeddings(self.config)
        self._token_counter = self._build_token_counter()
        
        Logger.info("Embeddings 将保存到各文档的 processed 目录")

    def _build_token_counter(self):
        """构建 token 计数器，优先复用 embedding tokenizer。"""
        tokenizer = getattr(self.embeddings, "tokenizer", None)
        if tokenizer is not None:
            def count_tokens(text: str) -> int:
                if not text:
                    return 0
                try:
                    return len(tokenizer.encode(text).ids)
                except Exception:
                    return len(text)
            return count_tokens

        def fallback_count(text: str) -> int:
            if not text:
                return 0
            # 退化策略：中英文混合文本按词段近似计数，至少返回字符数的保守下界
            units = re.findall(r'[\u4e00-\u9fff]|[A-Za-z0-9_]+', text)
            return len(units) if units else len(text)

        return fallback_count

    def _extract_image_title(self, img_filename: str) -> str:
        """从图片文件名提取稳定的标题文本。"""
        stem = Path(img_filename).stem
        stem = re.sub(r'_(p\d{4}_\d{3}|\d{3})$', '', stem)
        return stem.strip()
        
    def extract_image_references(self, chunk_text: str) -> List[str]:
        """
        从 chunk 文本中提取图片引用
        
        Args:
            chunk_text: chunk 文本内容
            
        Returns:
            图片路径列表
        """
        import re
        
        # 匹配格式：图片：./images/文件名.扩展名
        pattern = r'图片：\./images/([^\s\n]+)'
        matches = re.findall(pattern, chunk_text)
        
        return matches
    
    def build_metadata(self, chunk_info: Dict[str, Any], doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建 chunk 的 metadata
        
        Args:
            chunk_info: chunk 信息
            doc_data: 文档数据
            
        Returns:
            metadata 字典
        """
        metadata = {
            # 基本信息
            'source': doc_data['doc_name'],
            'chunk_id': chunk_info.get('chunk_id', ''),
            'chunk_index': chunk_info.get('index', 0),
            'char_count': chunk_info.get('char_count', 0),
            'chunk_token_count': self._token_counter(chunk_info.get('content', '')),
            'chunk_hash': compute_chunk_hash(chunk_info.get('content', '')),
            
            # 文档信息
            'doc_format': doc_data['doc_info'].get('format', 'Unknown'),
            'doc_timestamp': doc_data['timestamp'],
            'processed_at': doc_data['doc_info'].get('created_at', ''),
            'processed_batch_id': doc_data['timestamp'],
            'doc_version_hash': doc_data['doc_info'].get('doc_version_hash', ''),
            'space_id': doc_data['doc_info'].get('space_id', ''),
            'source_url': doc_data['doc_info'].get('source_url', ''),
            'source_updated_at': doc_data['doc_info'].get('source_updated_at', ''),
            
            # 图片信息
            'has_images': False,
            'image_count': 0,
            'images': [],
            'image_titles': [],
        }

        quality = analyze_chunk_quality(chunk_info.get('content', ''))
        metadata['content_quality_score'] = quality['score']
        metadata['quality_flags'] = quality['flags']
        metadata['blank_ratio'] = quality['blank_ratio']
        metadata['repeat_ratio'] = quality['repeat_ratio']
        metadata['is_structured_chunk'] = quality['is_structured_chunk']
        
        # 提取图片引用
        chunk_text = chunk_info.get('content', '')
        image_refs = self.extract_image_references(chunk_text)
        
        if image_refs:
            metadata['has_images'] = True
            metadata['image_count'] = len(image_refs)
            
            # 添加图片详细信息
            for img_filename in image_refs:
                if img_filename in doc_data['images']:
                    img_info = doc_data['images'][img_filename]
                    image_title = self._extract_image_title(img_filename)
                    metadata['images'].append({
                        'filename': img_filename,
                        'original_filename': img_info.get('original_filename', ''),
                        'path': f"./images/{img_filename}",
                        'title': image_title,
                        'context_before': img_info.get('context_before', '')[:100],  # 限制长度
                        'context_after': img_info.get('context_after', '')[:100]
                    })
                    metadata['image_titles'].append(image_title)
                else:
                    # 图片信息未找到，仍然记录引用
                    image_title = self._extract_image_title(img_filename)
                    metadata['images'].append({
                        'filename': img_filename,
                        'path': f"./images/{img_filename}",
                        'title': image_title,
                    })
                    metadata['image_titles'].append(image_title)

        return metadata
    
    def build_embeddings_for_document(self, doc_data: Dict[str, Any]) -> int:
        """
        为单个文档构建 embeddings，每个 chunk 单独保存
        
        Args:
            doc_data: 文档数据
            
        Returns:
            添加的 chunk 数量
        """
        Logger.info(f"处理文档: {doc_data['doc_name']} ({doc_data['timestamp']})")
        
        # 在 processed 目录下创建 embeddings 和 metadata 子目录
        # 路径：processed/时间戳/文档名/embeddings/ 和 metadata/
        embeddings_dir = doc_data['path'] / 'embeddings'
        metadata_dir = doc_data['path'] / 'metadata'
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        texts = []
        metadatas = []
        ids = []
        chunk_indices = []
        
        for chunk_info in doc_data['chunks']:
            chunk_text = chunk_info.get('content', '')
            if not chunk_text.strip():
                continue
            
            # 构建 metadata
            metadata = self.build_metadata(chunk_info, doc_data)
            
            texts.append(chunk_text)
            metadatas.append(metadata)
            chunk_indices.append(chunk_info['index'])
            ids.append(f"{doc_data['doc_name']}_{doc_data['timestamp']}_{chunk_info['chunk_id']}")
        
        if not texts:
            Logger.warning("没有有效的 chunks", indent=1)
            return 0
        
        # 分批生成 embeddings（避免大文档一次性加载导致 OOM/崩溃）
        batch_size = self.config.get('embedding', {}).get('batch_size', 16)
        Logger.info(f"生成 {len(texts)} 个 embeddings（batch_size={batch_size}）...", indent=1)

        embeddings = []
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
            if len(texts) > batch_size:
                Logger.info(f"  批次 {batch_start // batch_size + 1}/"
                            f"{(len(texts) + batch_size - 1) // batch_size}: "
                            f"{len(batch_embeddings)} 个", indent=1)
        
        # 每个 chunk 分别保存 embedding 和 metadata
        Logger.info("保存 embeddings 和 metadata（每个 chunk 单独文件）...", indent=1)
        
        for i in range(len(texts)):
            chunk_idx = chunk_indices[i]
            
            # 1. 保存 embedding（只包含向量）
            embedding_file = embeddings_dir / f"chunk_{chunk_idx:04d}.json"
            embedding_data = {
                'chunk_index': chunk_idx,
                'embedding': embeddings[i],
                'model': self.config.get('embedding', {}).get('model', 'BAAI/bge-small-zh-v1.5'),
                'created_at': datetime.now().isoformat()
            }
            
            with open(embedding_file, 'w', encoding='utf-8') as f:
                json.dump(embedding_data, f, ensure_ascii=False, indent=2)
            
            # 2. 保存 metadata（元数据）
            metadata_file = metadata_dir / f"chunk_{chunk_idx:04d}.json"
            metadata_full = {
                'id': ids[i],
                'doc_name': doc_data['doc_name'],
                'timestamp': doc_data['timestamp'],
                'chunk_index': chunk_idx,
                'metadata': metadatas[i],
                'created_at': datetime.now().isoformat()
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_full, f, ensure_ascii=False, indent=2)
        
        Logger.success(f"成功保存 {len(texts)} 个 embeddings", indent=1)
        Logger.info(f"Embeddings: {embeddings_dir}", indent=1)
        Logger.info(f"Metadata: {metadata_dir}", indent=1)
        
        # 统计图片信息
        chunks_with_images = sum(1 for m in metadatas if m['has_images'])
        total_images = sum(m['image_count'] for m in metadatas)
        
        if chunks_with_images > 0:
            Logger.info(f"包含图片的 chunks: {chunks_with_images}", indent=1)
            Logger.info(f"图片引用总数: {total_images}", indent=1)
        
        return len(texts)
