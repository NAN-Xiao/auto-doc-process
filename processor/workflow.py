#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量处理工具 - 文档拆分 + Embedding 生成 + pgvector 入库"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from ..core.config import load_processor_config as load_config
from ..core.logger import Logger
from ..core.utils import safe_filename, atomic_write_json
from .splitter import process_document, generate_output_path
from .embedder import EmbeddingGenerator
from .storage import PgVectorStorage
from .quality import compute_chunk_hash, compute_document_version_hash, summarize_document_quality, summarize_batch_quality


class BatchWorkflow:
    """批量工作流"""
    
    def __init__(self, use_llm_naming: bool = False, db_config: dict = None,
                 config: dict = None):
        """
        初始化批量工作流

        Args:
            use_llm_naming: 是否使用 LLM 图片命名
            db_config: 数据库配置（None 则从 config 中取）
            config: 处理器配置字典（None 则自动加载）
        """
        self.config = config if config is not None else load_config()
        self.use_llm_naming = use_llm_naming
        self.embedding_generator = EmbeddingGenerator(config=self.config)
        
        # 数据库存储
        _db_cfg = db_config or self.config.get('database', {})
        if _db_cfg:
            self.vector_storage = PgVectorStorage(db_config=_db_cfg)
        else:
            self.vector_storage = None
        
        paths_config = self.config.get('paths', {})
        self.documents_dir = Path(paths_config.get('documents_dir', './documents'))

        # 优先使用独立的 processed_dir，向后兼容 processed_subdir
        processed_dir_raw = paths_config.get('processed_dir', '')
        if processed_dir_raw:
            from ..core.config import MODULE_DIR
            p = Path(processed_dir_raw)
            self.processed_dir = p if p.is_absolute() else (MODULE_DIR / p).resolve()
        else:
            self.processed_dir = self.documents_dir / paths_config.get('processed_subdir', 'processed')

        doc_config = self.config.get('doc_splitter', {})
        self.supported_formats = doc_config.get('supported_formats', ['.pdf', '.docx'])
        
        Logger.separator()
        Logger.info("批量处理工具已初始化")
        Logger.info(f"文档目录: {self.documents_dir}", indent=1)
        Logger.info(f"处理目录: {self.processed_dir}", indent=1)
        Logger.info(f"支持格式: {', '.join(self.supported_formats)}", indent=1)
        Logger.info(f"LLM 命名: {'启用' if self.use_llm_naming else '禁用'}", indent=1)
        Logger.separator()
    
    def scan_documents(self) -> List[Path]:
        """扫描文档目录"""
        if not self.documents_dir.exists():
            Logger.error(f"文档目录不存在: {self.documents_dir}")
            return []
        
        documents = []
        for fmt in self.supported_formats:
            documents.extend(self.documents_dir.glob(f"*{fmt}"))
        
        documents = sorted(documents)
        
        return documents
    
    def process_single_document(self, doc_path: Path, batch_timestamp: str = None,
                                doc_meta: Dict[str, Any] = None,
                                store_to_db: bool = True) -> Optional[Dict[str, Any]]:
        """
        处理单个文档（拆分 + 向量化，可选入库）

        Args:
            doc_path: 文档路径
            batch_timestamp: 处理时间戳（仅用于 DB 元数据，不影响目录结构）
            doc_meta: 文档来源元数据（如 space_id, source_url），用于写入 pgvector
            store_to_db: 是否立即入库（False = 只处理不入库，后续调用 batch_store）
        """
        Logger.separator()
        Logger.info(f"处理文档: {doc_path.name}")
        Logger.separator()

        start_time = datetime.now()
        if batch_timestamp is None:
            batch_timestamp = start_time.strftime("%Y%m%d_%H%M%S")

        result = {
            'document': doc_path.name,
            'doc_path': str(doc_path),
            'started_at': start_time.isoformat(),
            'batch_timestamp': batch_timestamp,
            'step1_split': None,
            'step2_embeddings': None,
            'success': False,
            'error': None
        }

        try:
            Logger.info("步骤1：拆分文档...")

            # 输出到独立的 processed_dir/{doc_name}/
            output_path = self.processed_dir / safe_filename(doc_path.stem, doc_path.stem)

            doc_info = process_document(
                input_path=doc_path,
                output_dir=output_path,
                use_llm_naming=self.use_llm_naming,
                config=self.config,
            )

            if doc_info:
                result['step1_split'] = {
                    'success': True,
                    'output_path': str(output_path),
                    'chunks_count': doc_info.total_chunks,
                    'images_count': doc_info.total_images,
                    'processed_at': doc_info.created_at
                }
                Logger.success("文档拆分成功")
                Logger.info(f"输出目录: {output_path}", indent=1)
                Logger.info(f"文本块数: {doc_info.total_chunks}", indent=1)
                Logger.info(f"图片数: {doc_info.total_images}", indent=1)
            else:
                result['step1_split'] = {'success': False, 'error': '拆分失败'}
                result['error'] = '步骤1：文档拆分失败'
                Logger.error("文档拆分失败")
                return result

            # 空文档（0 chunks）→ 跳过后续步骤，不视为错误
            if doc_info.total_chunks == 0:
                Logger.warning(f"文档内容为空（0 chunks），跳过: {doc_path.name}")
                result['step2_embeddings'] = {'success': True, 'chunk_count': 0}
                result['success'] = True
                result['skipped'] = True
                result['completed_at'] = datetime.now().isoformat()
                return result

            Logger.info("步骤2：生成 Embeddings...")

            chunks = self._load_chunks(output_path)
            if not chunks:
                result['step2_embeddings'] = {'success': False, 'error': '无法加载 chunks'}
                result['error'] = '步骤2：无法加载 chunks'
                Logger.error("无法加载 chunks 数据")
                return result

            images = self._load_images(output_path)
            for chunk in chunks:
                chunk["chunk_hash"] = compute_chunk_hash(chunk.get("content", ""))
            doc_version_hash = compute_document_version_hash(chunks)

            doc_data = {
                'path': output_path,
                'doc_name': safe_filename(doc_path.stem, doc_path.stem),
                'timestamp': batch_timestamp,
                'doc_info': {
                    'format': doc_info.format,
                    'chunks_count': doc_info.total_chunks,
                    'images_count': doc_info.total_images,
                    'created_at': doc_info.created_at,
                    'doc_version_hash': doc_version_hash,
                    'space_id': (doc_meta or {}).get('space_id', ''),
                    'source_url': (doc_meta or {}).get('source_url', ''),
                    'source_updated_at': (doc_meta or {}).get('source_updated_at', ''),
                },
                'chunks': chunks,
                'images': images
            }

            chunk_count = self.embedding_generator.build_embeddings_for_document(doc_data)

            if chunk_count > 0:
                result['step2_embeddings'] = {
                    'success': True,
                    'chunk_count': chunk_count,
                    'embeddings_dir': str(output_path / 'embeddings'),
                    'metadata_dir': str(output_path / 'metadata')
                }
                Logger.success("Embeddings 生成成功")
                Logger.info(f"Embeddings 数: {chunk_count}", indent=1)
            else:
                result['step2_embeddings'] = {'success': False, 'error': 'Embeddings 生成失败'}
                result['error'] = '步骤2：Embeddings 生成失败'
                Logger.error("Embeddings 生成失败")
                return result

            # 保存 dir_info 供后续 batch_store 使用
            _meta = doc_meta or {}
            result['_dir_info'] = {
                'doc_name': doc_data['doc_name'],
                'timestamp': doc_data['timestamp'],
                'embeddings_dir': output_path / 'embeddings',
                'metadata_dir': output_path / 'metadata',
                'chunks_dir': output_path / 'chunks',
                'space_id': _meta.get('space_id', ''),
                'source_url': _meta.get('source_url', ''),
                'doc_version_hash': doc_version_hash,
                'source_updated_at': _meta.get('source_updated_at', ''),
            }
            result['doc_version_hash'] = doc_version_hash

            quality_report = self._build_document_quality_report(
                output_path=output_path,
                doc_name=doc_data['doc_name'],
                batch_timestamp=batch_timestamp,
                doc_version_hash=doc_version_hash,
            )
            result['quality_report'] = quality_report

            # ---- 步骤3：存入 PostgreSQL (pgvector) ----
            if store_to_db and self.vector_storage:
                Logger.info("步骤3：存入 PostgreSQL...")
                try:
                    self.vector_storage.init_table()
                    stored = self.vector_storage.store_document(result['_dir_info'])
                    result['step3_store'] = {
                        'success': stored > 0,
                        'stored_count': stored,
                    }
                    if stored > 0:
                        Logger.success(f"数据库入库成功: {stored} chunks", indent=1)
                    else:
                        Logger.warning("数据库入库: 无有效 chunks")
                except Exception as e:
                    result['step3_store'] = {'success': False, 'error': str(e)}
                    Logger.error(f"数据库入库失败: {e}")
            elif not store_to_db:
                Logger.info("步骤3：延迟入库（等待批量统一入库）")
            else:
                Logger.info("步骤3：跳过（未配置数据库）")

            result['success'] = True
            end_time = datetime.now()
            result['completed_at'] = end_time.isoformat()
            result['duration_seconds'] = (end_time - start_time).total_seconds()

            Logger.separator()
            Logger.success(f"文档处理完成: {doc_path.name}")
            Logger.info(f"总耗时: {result['duration_seconds']:.2f} 秒", indent=1)
            Logger.separator()

            return result

        except Exception as e:
            result['error'] = str(e)
            Logger.error(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            return result

    def batch_store(self, results: List[Dict[str, Any]]) -> int:
        """
        批量统一入库：将所有已处理文档在一个事务中写入 PostgreSQL

        Args:
            results: process_single_document 返回的结果列表（需包含 _dir_info）

        Returns:
            成功入库的文档数
        """
        if not self.vector_storage:
            Logger.warning("未配置数据库，跳过批量入库")
            return 0

        # 收集所有需要入库的 dir_info（排除空文档）
        dir_infos = []
        for r in results:
            if r and r.get('success') and r.get('_dir_info') and not r.get('skipped'):
                dir_infos.append(r['_dir_info'])

        if not dir_infos:
            Logger.warning("没有需要入库的文档")
            return 0

        Logger.separator()
        Logger.info(f"批量统一入库: {len(dir_infos)} 个文档")
        Logger.separator()

        self.vector_storage.init_table()
        stored_count = self.vector_storage.batch_store_documents(dir_infos)

        Logger.separator()
        Logger.success(f"批量入库完成: {stored_count} 个文档")
        Logger.separator()

        return stored_count
    
    def _load_images(self, doc_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        从文档目录加载 images 数据
        
        Args:
            doc_dir: 文档处理输出目录
            
        Returns:
            images 字典，key 是 smart_filename，value 是图片信息
        """
        try:
            # 读取 images 索引
            index_file = doc_dir / "images_index.json"
            if not index_file.exists():
                Logger.warning(f"images 索引文件不存在: {index_file}，可能文档没有图片")
                return {}
            
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # 转换为字典格式，key 为 smart_filename
            images_dict = {}
            for img_info in index_data.get('images', []):
                smart_filename = img_info.get('smart_filename', '')
                if smart_filename:
                    images_dict[smart_filename] = {
                        'original_filename': img_info.get('original_filename', ''),
                        'page_number': img_info.get('page_number'),
                        'context_before': img_info.get('context_before', ''),
                        'context_after': img_info.get('context_after', ''),
                        'smart_filename': smart_filename
                    }
            
            return images_dict
            
        except Exception as e:
            Logger.warning(f"加载 images 失败: {e}")
            return {}

    def _build_document_quality_report(self, output_path: Path, doc_name: str,
                                       batch_timestamp: str, doc_version_hash: str) -> Dict[str, Any]:
        """汇总单文档质量报告并落盘。"""
        metadata_dir = output_path / "metadata"
        chunk_reports: List[Dict[str, Any]] = []
        for meta_file in sorted(metadata_dir.glob("chunk_*.json")):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                m = meta.get("metadata", {})
                chunk_reports.append({
                    "chunk_index": m.get("chunk_index", meta.get("chunk_index", 0)),
                    "chunk_id": m.get("chunk_id", ""),
                    "chunk_token_count": m.get("chunk_token_count", 0),
                    "content_quality_score": m.get("content_quality_score", 0.0),
                    "quality_flags": m.get("quality_flags", []),
                    "is_structured_chunk": m.get("is_structured_chunk", False),
                    "has_images": m.get("has_images", False),
                    "metadata": m,
                })
            except Exception as e:
                Logger.warning(f"读取质量元数据失败 {meta_file.name}: {e}", indent=1)

        summary = summarize_document_quality(chunk_reports)
        summary.update({
            "doc_name": doc_name,
            "processed_batch_id": batch_timestamp,
            "doc_version_hash": doc_version_hash,
            "generated_at": datetime.now().isoformat(),
        })
        atomic_write_json(output_path / "quality_report.json", summary)
        return summary

    def build_batch_quality_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总批次质量报告。"""
        doc_reports = []
        for item in results:
            if item and item.get("quality_report"):
                doc_reports.append(item["quality_report"])
        report = summarize_batch_quality(doc_reports)
        report["generated_at"] = datetime.now().isoformat()
        report["documents"] = doc_reports
        return report
    
    def _load_chunks(self, doc_dir: Path) -> List[Dict[str, Any]]:
        """
        从文档目录加载 chunks 数据
        
        Args:
            doc_dir: 文档处理输出目录
            
        Returns:
            chunks 列表，每个元素包含 chunk_id, index, content 等
        """
        try:
            # 读取 chunks 索引
            index_file = doc_dir / "chunks_index.json"
            if not index_file.exists():
                Logger.error(f"chunks 索引文件不存在: {index_file}")
                return []
            
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            chunks = []
            chunks_dir = doc_dir / "chunks"
            
            for chunk_info in index_data.get('chunks', []):
                chunk_index = chunk_info['index']
                chunk_file = chunks_dir / f"chunk_{chunk_index:04d}.txt"
                
                if not chunk_file.exists():
                    Logger.warning(f"chunk 文件不存在: {chunk_file}")
                    continue
                
                # 读取 chunk 内容
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 组合完整的 chunk 数据
                chunk_data = {
                    'chunk_id': chunk_info['chunk_id'],
                    'index': chunk_info['index'],
                    'content': content,
                    'char_count': chunk_info['char_count'],
                    'page_number': chunk_info.get('page_number'),
                    'metadata': chunk_info.get('metadata', {})
                }
                chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            Logger.error(f"加载 chunks 失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_all_documents(self) -> Dict[str, Any]:
        """批量处理所有文档"""
        documents = self.scan_documents()
        
        if not documents:
            Logger.error("未找到任何文档")
            return {
                'total_documents': 0,
                'success_count': 0,
                'failed_count': 0,
                'documents': []
            }
        
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        Logger.info(f"找到 {len(documents)} 个文档待处理")
        
        results = []
        success_count = 0
        
        for i, doc_path in enumerate(documents, 1):
            Logger.info(f"[{i}/{len(documents)}] 正在处理: {doc_path.name}")
            
            result = self.process_single_document(doc_path, batch_timestamp=batch_timestamp)
            results.append(result)
            
            if result and result.get('success'):
                success_count += 1
        
        report = {
            'total_documents': len(documents),
            'success_count': success_count,
            'failed_count': len(documents) - success_count,
            'batch_timestamp': batch_timestamp,
            'processed_at': datetime.now().isoformat(),
            'use_llm_naming': self.use_llm_naming,
            'documents_dir': str(self.documents_dir),
            'processed_dir': str(self.processed_dir),
            'documents': results
        }

        # 报告保存到 processed/ 根目录
        report_dir = self.processed_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / "batch_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        Logger.separator()
        Logger.success("批量处理完成")
        Logger.info(f"成功: {success_count}/{len(documents)}", indent=1)
        Logger.info(f"失败: {len(documents) - success_count}/{len(documents)}", indent=1)
        Logger.info(f"报告: {report_file}", indent=1)
        Logger.separator()
        
        return report
