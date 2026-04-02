# 项目优化建议

本文档基于当前代码与管线（download → process → store → graph）的梳理，给出可落地的优化方向与优先级。

**说明**：本工程仅做预处理（不做召回），若优先关注**预处理质量**（拆分、图片命名、元数据、图谱），请参见 [预处理质量说明](PREPROCESSING_QUALITY.md)。

---

## 一、性能优化

### 1. 高优先级

| 项 | 现状 | 建议 | 预期 |
|---|------|------|------|
| **ONNX 线程数** | `onnx_embedder.py` 中写死 `intra_op_num_threads=2, inter_op_num_threads=2` | 从 `doc_splitter.yaml` 的 `embedding.onnx` 读取，默认 2，多核机可调大 | 多核环境下 embedding 阶段加速 |
| **数据库连接** | 每次 `_get_conn()` 新建连接，用后关闭 | 对 `PgVectorStorage` 使用连接池（如 `psycopg_pool`）或进程内单例复用 | 减少 store/graph 阶段建连开销 |
| **配置重复加载** | `load_full_config()` 与 `load_processor_config()` 分别加载 `db_info.yml` | 在 `load_full_config()` 中复用已加载的 db 配置，或统一由一处产出「完整配置」 | 减少重复读盘与解析 |

### 2. 中优先级

| 项 | 现状 | 建议 | 预期 |
|---|------|------|------|
| **入库方式** | `batch_store_documents` 使用 `executemany` 逐行 INSERT | 当单次 chunks 数量很大（如 >5000）时，可增加 `COPY FROM` 或分批 INSERT + 中间 commit 的选项 | 大批量入库时减少锁持有时间与内存峰值 |
| **Process 阶段并行** | 文档在子进程内串行处理（一个 worker 进程） | 保持当前子进程崩溃隔离，可在**多子进程**间分配文档（每进程独立 worker），需注意 ONNX/模型是否支持多进程 | 多文档时缩短 process 总时长 |
| **chunk 文件读取** | `_load_chunks` 对每个 chunk 单独 `open` 读 `.txt` | 若单文档 chunk 数很多，可考虑顺序读时加大 buffer 或一次性读入目录再批量处理 | 略减 I/O 与系统调用次数 |

### 3. 低优先级

| 项 | 现状 | 建议 | 预期 |
|---|------|------|------|
| **日志量** | 每个文档、每个批次都打 INFO | 将「每个 chunk/每批」的进度改为 DEBUG，仅保留「每文档开始/完成」为 INFO | 降低 I/O、便于在日志中抓重点 |
| **Embedding 批大小** | 默认 16，可配置 | 在显存/内存允许下，适当提高（如 32/64）可提高 GPU/CPU 利用率 | 略减 embedding 阶段时间 |

---

## 二、结构与可维护性

| 项 | 建议 |
|---|------|
| **配置入口** | 将「飞书 + 处理 + DB + LightRAG」的扁平配置收敛到 1～2 个加载入口，避免多处各自 `load_processor_config` / `load_db_config`，减少不一致与重复读 YAML。 |
| **大文件拆分** | 当前已拆为 `processor/splitter_base.py`、`processor/pdf_splitter.py`、`processor/word_splitter.py`，后续可继续把图片命名、结构化构块、OCR 再拆成更细模块；流程编排已迁到 `pipeline/orchestrator.py`。 |
| **类型注解** | 对 `__main__.py`、`workflow.py`、`storage.py` 等入口与公共接口补全类型注解，便于静态检查与重构。 |

---

## 三、已有优点（可保留）

- **子进程隔离**：process 阶段用独立子进程 + JSONL 结果文件，单文档崩溃不影响其余，且可恢复。
- **单事务批量入库**：store 阶段先预加载所有 chunk 再在事务内写库，锁持有时间可控；advisory lock 避免多进程写冲突。
- **ONNX 优先**：无 PyTorch 时用 ONNX Runtime，依赖更轻。
- **配置缓存**：YAML 与 processor 配置已有缓存，scheduler 可调用 `clear_config_cache()` 做热更新。
- **预检**：启动前磁盘、数据库、embedding 模型检查，失败快速退出。

---

## 四、建议实施顺序

1. **立刻可做**：ONNX 线程数配置化、减少重复配置加载、日志级别微调（见上「高优先级」与「低优先级」）。
2. **短期**：为 `PgVectorStorage` 增加可选连接池或单例连接复用。
3. **中期**：根据实际文档量与机器资源，评估「多 worker 进程分配文档」与「大批量 COPY/分批 INSERT」。
4. **长期**：继续细化 `pipeline/` 与 `processor/` 的边界、统一配置入口、补全类型与单测。

---

## 五、配置示例（建议新增）

在 `configs/doc_splitter.yaml` 的 `embedding` 下可增加 ONNX 线程配置，例如：

```yaml
embedding:
  model: "BAAI/bge-small-zh-v1.5"
  batch_size: 16
  onnx:
    intra_op_num_threads: 2   # 算子内线程数
    inter_op_num_threads: 2    # 算子间线程数（多核可调大）
  huggingface:
    cache_folder: "./models"
    device: "cpu"
```

代码在 `processor/onnx_embedder.py` 的 `OnnxEmbeddings.__init__` 中从 `config` 读取上述字段即可。
