#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档处理管线：拆分 → 向量化 → pgvector 入库

子模块:
  - splitter     文档拆分（PDF / Word）
  - embedder     向量生成（HuggingFace / bge-small-zh）
  - storage      pgvector 持久化
  - workflow     批量工作流（串联上述三步）
  - excel        Excel 元数据提取
"""

