#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 Excel 元数据 JSON 文件的 token 数量"""

import os
import sys
import json
import tiktoken
from pathlib import Path

from ...core.config import load_processor_config as load_config

def count_tokens(directory: str):
    """统计目录下所有 JSON 文件的 token 数量"""
    enc = tiktoken.encoding_for_model('gpt-4')
    
    total_tokens = 0
    total_chars = 0
    file_count = 0
    file_details = []
    
    excel_dir = Path(directory)
    if not excel_dir.exists():
        print(f"❌ 目录不存在: {excel_dir}")
        return
    
    json_files = sorted([f for f in excel_dir.iterdir() if f.suffix == '.json'])
    
    print(f"📊 开始统计 {len(json_files)} 个 JSON 文件的 token 数量...\n")
    print(f"{'文件名':<50} {'字符数':>12} {'Token数':>12}")
    print("=" * 76)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为字符串
            text = json.dumps(data, ensure_ascii=False)
            char_count = len(text)
            
            # 计算 token 数
            tokens = enc.encode(text)
            token_count = len(tokens)
            
            file_details.append({
                'name': json_file.name,
                'chars': char_count,
                'tokens': token_count
            })
            
            total_chars += char_count
            total_tokens += token_count
            file_count += 1
            
            print(f"{json_file.name:<50} {char_count:>12,} {token_count:>12,}")
            
        except Exception as e:
            print(f"❌ 处理文件 {json_file.name} 失败: {e}")
    
    print("=" * 76)
    print(f"{'总计:':<50} {total_chars:>12,} {total_tokens:>12,}")
    print(f"\n📈 统计摘要:")
    print(f"  文件数量: {file_count}")
    print(f"  总字符数: {total_chars:,}")
    print(f"  总Token数: {total_tokens:,}")
    print(f"  平均每个文件: {total_tokens // file_count if file_count > 0 else 0:,} tokens")
    
    # 计算大约需要的上下文窗口
    print(f"\n💡 上下文窗口估算:")
    print(f"  GPT-4 (8K): 可容纳 {8192 // (total_tokens // file_count if file_count > 0 else 1)} 个平均文件")
    print(f"  GPT-4 (32K): 可容纳 {32768 // (total_tokens // file_count if file_count > 0 else 1)} 个平均文件")
    print(f"  GPT-4 (128K): 可容纳 {131072 // (total_tokens // file_count if file_count > 0 else 1)} 个平均文件")
    
    # 找出最大的文件
    if file_details:
        largest = max(file_details, key=lambda x: x['tokens'])
        smallest = min(file_details, key=lambda x: x['tokens'])
        print(f"\n📦 文件大小范围:")
        print(f"  最大: {largest['name']} ({largest['tokens']:,} tokens)")
        print(f"  最小: {smallest['name']} ({smallest['tokens']:,} tokens)")

if __name__ == "__main__":
    # 从配置文件读取路径
    config = load_config()
    paths_config = config.get('paths', {})
    documents_dir = Path(paths_config.get('documents_dir', './documents'))
    processed_subdir = paths_config.get('processed_subdir', 'processed')
    
    excel_metadata_dir = documents_dir / processed_subdir / "excel"
    
    print(f"📁 Excel 元数据目录: {excel_metadata_dir}\n")
    count_tokens(str(excel_metadata_dir))

