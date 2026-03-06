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
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from ..core.config import load_processor_config as load_config
from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingGenerator:
    """Embedding 生成器"""
    
    def __init__(self):
        """
        初始化
        """
        # 加载配置
        self.config = load_config()
        
        # 初始化 embedding 模型
        embedding_config = self.config.get('embedding', {})
        model_name = embedding_config.get('model', 'BAAI/bge-small-zh-v1.5')
        
        # 从配置读取 HuggingFace 参数
        hf_config = embedding_config.get('huggingface', {})
        cache_folder = hf_config.get('cache_folder', './models')
        device = hf_config.get('device', 'cpu')
        normalize_embeddings = hf_config.get('normalize_embeddings', True)
        
        print(f"📦 加载 Embedding 模型: {model_name} (device: {device})")
        print(f"📁 模型缓存目录: {cache_folder}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize_embeddings}
        )
        
        print(f"💾 Embeddings 将保存到各文档的 processed 目录\n")
        
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
            
            # 文档信息
            'doc_format': doc_data['doc_info'].get('format', 'Unknown'),
            'doc_timestamp': doc_data['timestamp'],
            'processed_at': doc_data['doc_info'].get('created_at', ''),
            
            # 图片信息
            'has_images': False,
            'image_count': 0,
            'images': []
        }
        
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
                    metadata['images'].append({
                        'filename': img_filename,
                        'original_filename': img_info.get('original_filename', ''),
                        'path': f"./images/{img_filename}",
                        'context_before': img_info.get('context_before', '')[:100],  # 限制长度
                        'context_after': img_info.get('context_after', '')[:100]
                    })
                else:
                    # 图片信息未找到，仍然记录引用
                    metadata['images'].append({
                        'filename': img_filename,
                        'path': f"./images/{img_filename}"
                    })
        
        return metadata
    
    def build_embeddings_for_document(self, doc_data: Dict[str, Any]) -> int:
        """
        为单个文档构建 embeddings，每个 chunk 单独保存
        
        Args:
            doc_data: 文档数据
            
        Returns:
            添加的 chunk 数量
        """
        print(f"\n📄 处理文档: {doc_data['doc_name']} ({doc_data['timestamp']})")
        
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
            print("  ⚠️  没有有效的 chunks")
            return 0
        
        # 生成 embeddings
        print(f"  📊 生成 {len(texts)} 个 embeddings...")
        embeddings = self.embeddings.embed_documents(texts)
        
        # 每个 chunk 分别保存 embedding 和 metadata
        print(f"  💾 保存 embeddings 和 metadata（每个 chunk 单独文件）...")
        
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
        
        print(f"  ✅ 成功保存 {len(texts)} 个 embeddings")
        print(f"  📁 Embeddings: {embeddings_dir}")
        print(f"  📁 Metadata: {metadata_dir}")
        
        # 统计图片信息
        chunks_with_images = sum(1 for m in metadatas if m['has_images'])
        total_images = sum(m['image_count'] for m in metadatas)
        
        if chunks_with_images > 0:
            print(f"  🖼️  包含图片的 chunks: {chunks_with_images}")
            print(f"  🖼️  图片引用总数: {total_images}")
        
        return len(texts)

