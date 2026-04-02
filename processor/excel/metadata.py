#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Excel 元数据提取器。"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from ...core.config import load_processor_config as load_config
from ...core.logger import Logger
from ...core.utils import atomic_write_json


class ExcelMetadataExtractor:
    """Excel 元数据提取器。"""
    
    def __init__(self, 
                 desc_row: int = 0,
                 name_row: int = 1,
                 type_row: int = 2):
        """
        初始化提取器
        
        Args:
            desc_row: 描述行索引（默认第0行）
            name_row: 字段名行索引（默认第1行）
            type_row: 类型行索引（默认第2行）
        """
        self.desc_row = desc_row
        self.name_row = name_row
        self.type_row = type_row
        
        config = load_config()
        paths_config = config.get('paths', {})
        
        self.excel_source_dir = Path(paths_config.get('excel_dir', './excel'))
        
        processed_dir_raw = paths_config.get('processed_dir', '')
        if processed_dir_raw:
            from ...core.config import MODULE_DIR
            p = Path(processed_dir_raw)
            processed_base = p if p.is_absolute() else (MODULE_DIR / p).resolve()
        else:
            documents_dir = Path(paths_config.get('documents_dir', './documents'))
            processed_base = documents_dir / 'processed'
        self.output_dir = processed_base / 'excel'

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_output_file(self, excel_path: Path) -> Path:
        """返回单个 Excel 元数据输出路径。"""
        return self.output_dir / f"{excel_path.stem}_metadata.json"

    def _write_metadata_file(self, output_file: Path, payload: Dict[str, Any]) -> None:
        """统一写 JSON，避免半写状态。"""
        atomic_write_json(output_file, payload)

    def process_single_excel(self, excel_path: Path) -> Dict[str, Any]:
        """处理单个 Excel 文件并返回流程结果。"""
        start_time = datetime.now()
        result = {
            'document': excel_path.name,
            'doc_path': str(excel_path),
            'started_at': start_time.isoformat(),
            'processor': 'excel_metadata',
            'step1_split': None,
            'step2_embeddings': None,
            'success': False,
            'error': None,
        }

        if not excel_path.exists():
            result['error'] = 'Excel 文件不存在'
            Logger.error(f"Excel 文件不存在: {excel_path}")
            return result

        metadata = self.extract_excel_metadata(excel_path)
        if not metadata:
            result['error'] = 'Excel 元数据提取失败'
            return result

        output_file = self._build_output_file(excel_path)
        self._write_metadata_file(output_file, metadata)

        field_count = sum(sheet.get('field_count', 0) for sheet in metadata.get('sheets', []))
        result['step1_split'] = {
            'success': True,
            'output_path': str(output_file),
            'sheet_count': metadata.get('sheet_count', 0),
            'extracted_sheets': metadata.get('extracted_sheets', 0),
            'field_count': field_count,
            'processed_at': metadata.get('extracted_at', ''),
        }
        result['success'] = True
        end_time = datetime.now()
        result['completed_at'] = end_time.isoformat()
        result['duration_seconds'] = (end_time - start_time).total_seconds()
        return result
    
    def extract_sheet_metadata(self, 
                               excel_path: Path, 
                               sheet_name: str) -> Optional[Dict[str, Any]]:
        """提取单个工作表的元数据"""
        try:
            # 读取前三行
            df = pd.read_excel(
                excel_path, 
                sheet_name=sheet_name, 
                header=None,
                nrows=3
            )
            
            if df.empty or len(df) < 3:
                Logger.warning(f"工作表 {sheet_name} 行数不足3行，跳过", indent=1)
                return None
            
            # 提取三行数据
            desc_row_data = df.iloc[self.desc_row].tolist()
            name_row_data = df.iloc[self.name_row].tolist()
            type_row_data = df.iloc[self.type_row].tolist()
            
            # 构建字段列表
            fields = []
            for i, (desc, name, dtype) in enumerate(zip(desc_row_data, name_row_data, type_row_data)):
                # 跳过空列
                if pd.isna(name) or str(name).strip() == '':
                    continue
                
                field = {
                    'index': i,
                    'name': str(name).strip(),
                    'description': str(desc).strip() if not pd.isna(desc) else '',
                    'data_type': str(dtype).strip() if not pd.isna(dtype) else 'string'
                }
                fields.append(field)
            
            metadata = {
                'sheet_name': sheet_name,
                'field_count': len(fields),
                'fields': fields
            }
            
            Logger.info(f"工作表 {sheet_name}: {len(fields)} 个字段", indent=1)
            return metadata
            
        except Exception as e:
            Logger.error(f"提取工作表 {sheet_name} 元数据失败: {e}", indent=1)
            return None
    
    def extract_excel_metadata(self, excel_path: Path) -> Optional[Dict[str, Any]]:
        """提取单个 Excel 文件的元数据"""
        Logger.info(f"处理 Excel: {excel_path.name}")
        
        try:
            # 获取所有工作表名称
            xl = pd.ExcelFile(excel_path)
            sheet_names = xl.sheet_names
            
            Logger.info(f"发现 {len(sheet_names)} 个工作表", indent=1)
            
            # 提取每个工作表的元数据
            sheets_metadata = []
            for sheet_name in sheet_names:
                sheet_meta = self.extract_sheet_metadata(excel_path, sheet_name)
                if sheet_meta:
                    sheets_metadata.append(sheet_meta)
            
            if not sheets_metadata:
                Logger.warning(f"未提取到任何工作表元数据", indent=1)
                return None
            
            # 构建文件元数据
            file_metadata = {
                'file_name': excel_path.name,
                'file_stem': excel_path.stem,
                'sheet_count': len(sheet_names),
                'extracted_sheets': len(sheets_metadata),
                'extracted_at': datetime.now().isoformat(),
                'sheets': sheets_metadata
            }
            
            Logger.success(f"提取完成: {len(sheets_metadata)}/{len(sheet_names)} 个工作表")
            return file_metadata
            
        except Exception as e:
            Logger.error(f"处理 Excel 文件失败: {e}")
            return None
    
    def extract_all_excel_metadata(self) -> Dict[str, Any]:
        """批量提取所有 Excel 文件的元数据"""
        Logger.separator()
        Logger.info("Excel 元数据提取器启动")
        Logger.info(f"源目录: {self.excel_source_dir}", indent=1)
        Logger.info(f"输出目录: {self.output_dir}", indent=1)
        Logger.separator()
        
        if not self.excel_source_dir.exists():
            Logger.error(f"Excel 源目录不存在: {self.excel_source_dir}")
            return {
                'success': False,
                'error': 'Excel 源目录不存在',
                'files': []
            }
        
        # 扫描所有 Excel 文件
        excel_files = list(self.excel_source_dir.glob('*.xlsx')) + \
                     list(self.excel_source_dir.glob('*.xls'))
        
        if not excel_files:
            Logger.warning("未找到任何 Excel 文件")
            return {
                'success': True,
                'total_files': 0,
                'success_count': 0,
                'files': []
            }
        
        Logger.info(f"找到 {len(excel_files)} 个 Excel 文件")
        
        # 逐个处理
        all_metadata = []
        success_count = 0
        
        for i, excel_path in enumerate(excel_files, 1):
            Logger.info(f"\n[{i}/{len(excel_files)}] {excel_path.name}")
            
            metadata = self.extract_excel_metadata(excel_path)
            if metadata:
                all_metadata.append(metadata)
                success_count += 1
                
                # 保存单个文件的元数据
                output_file = self._build_output_file(excel_path)
                self._write_metadata_file(output_file, metadata)
                Logger.info(f"已保存: {output_file.name}", indent=1)
        
        # 生成汇总报告
        summary = {
            'success': True,
            'total_files': len(excel_files),
            'success_count': success_count,
            'failed_count': len(excel_files) - success_count,
            'extracted_at': datetime.now().isoformat(),
            'source_dir': str(self.excel_source_dir),
            'output_dir': str(self.output_dir),
            'files': all_metadata
        }
        
        # 保存汇总文件
        summary_file = self.output_dir / 'excel_metadata_summary.json'
        self._write_metadata_file(summary_file, summary)
        
        Logger.separator()
        Logger.success("元数据提取完成")
        Logger.info(f"成功: {success_count}/{len(excel_files)}", indent=1)
        Logger.info(f"汇总文件: {summary_file}", indent=1)
        Logger.separator()
        
        return summary


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Excel 元数据提取器')
    parser.add_argument('--desc-row', type=int, default=0, help='描述行索引（默认0）')
    parser.add_argument('--name-row', type=int, default=1, help='字段名行索引（默认1）')
    parser.add_argument('--type-row', type=int, default=2, help='类型行索引（默认2）')
    
    args = parser.parse_args()
    
    extractor = ExcelMetadataExtractor(
        desc_row=args.desc_row,
        name_row=args.name_row,
        type_row=args.type_row
    )
    
    summary = extractor.extract_all_excel_metadata()
    
    if summary['success'] and summary['success_count'] > 0:
        Logger.success("全部成功")
        sys.exit(0)
    elif summary.get('success_count', 0) > 0:
        Logger.warning("部分成功")
        sys.exit(0)
    else:
        Logger.error("处理失败")
        sys.exit(1)


if __name__ == '__main__':
    main()

