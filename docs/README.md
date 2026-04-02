# Docs Index

工程文档建议统一从这里进入，根目录保留兼容入口。

## 运行与接入

- [项目总览](../README.md)
- [RAG 接入说明](../RAG_INTEGRATION.md)
- [集成说明](../INTEGRATION.md)

## 质量与优化

- [预处理质量说明](../PREPROCESSING_QUALITY.md)
- [优化记录](../OPTIMIZATION.md)

## 工程结构建议

当前推荐把代码按职责理解为：

- `core/`
  通用基础设施：配置、日志、工具函数
- `feishu/`
  上游文档同步与导出
- `pipeline/`
  预处理编排层：阶段调度、预检、重置、批处理入口
- `processor/`
  预处理能力层：拆分、画像、embedding、存储、图谱
- `tools/`
  运维/模型辅助脚本
- `configs/`
  运行配置

当前已开始做的代码工程化整理：

- 文档数据结构抽到 `processor/contracts.py`
- 文档类型画像抽到 `processor/doc_profile.py`
- 管线编排从根目录 `__main__.py` 抽到 `pipeline/orchestrator.py`
- 路径解析抽到 `pipeline/paths.py`
- 文档拆分入口收敛为 `processor/splitter.py`，具体实现拆到 `processor/splitter_base.py`、`processor/pdf_splitter.py`、`processor/word_splitter.py`

后续若继续工程化，建议优先按这个方向推进：

1. 把 `pipeline/orchestrator.py` 再拆成 `stages.py`、`maintenance.py`、`cli.py`
2. 继续收紧 `processor/splitter_base.py`，把图片命名/OCR/结构化构块进一步拆模块
3. 把根目录脚本逐步收敛到 `scripts/`
