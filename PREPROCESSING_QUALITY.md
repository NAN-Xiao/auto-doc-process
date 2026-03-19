# 预处理质量说明

本工程只做**预处理**（下载、拆分、向量化、入库、图谱），不做召回。下文说明影响预处理质量的配置与可落地的质量评估方式。

---

## 一、影响质量的主要环节

| 环节 | 产出 | 质量相关配置 | 关注点 |
|------|------|--------------|--------|
| **拆分** | chunks 文本 + 图片 | `chunk_size` / `chunk_overlap` / `separators` | 边界是否在合理断句处、是否截断表格/列表、长度分布 |
| **图片** | 图片文件 + 智能命名 | `image_naming.use_llm` / `max_length` | 命名是否语义清晰、是否便于下游展示与检索 |
| **向量化** | embeddings | `embedding.model` / `batch_size` | 与下游检索使用同一模型即可，质量由模型决定 |
| **元数据** | metadata（source、chunk_id、图片引用等） | 代码内逻辑 | 是否带齐 source、页码、图片引用，便于召回后溯源 |
| **图谱** | 实体 / 关系 / chunk 关联 | LightRAG `chunk_token_size`、LLM 模型 | 实体是否准确、关系是否合理、是否便于图检索 |

---

## 二、拆分质量（chunk）

- **chunk_size / chunk_overlap**  
  - `chunk_size` 过大：单 chunk 信息过载，检索粒度粗，易带无关内容。  
  - 过小：上下文不足，易割裂语义。  
  - 建议：中文长文 800～1200 字常见；若多为短条（如 FAQ），可适当减小。  
  - `chunk_overlap` 保证跨块连贯，一般 100～300 字即可。

- **separators**  
  - 当前使用：`["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]`。  
  - 优先按段落、再按句、再按词切，可减少“半句”或“断表”的情况。

- **如何自查**  
  - 在 `processed/{文档名}/chunks/` 下抽样查看 `chunk_*.txt`：  
    - 是否在句号/段落边界结束；  
    - 表格、列表是否被拆散；  
    - 长度分布是否集中、有无异常超长/超短。

---

## 三、图片命名质量

- **use_llm: true**  
  - 用 LLM 根据上下文生成图片名，语义更好，便于后续检索或展示。  
  - 受 `image_naming.max_length`、`temperature` 等影响，过长易冗余。

- **use_llm: false**  
  - 关键词简单命名，速度快、成本低，但可读性一般。

- **如何自查**  
  - 看 `processed/{文档名}/images/` 下文件名与 `images_index.json` 中的 `smart_filename`、`context_before/after` 是否一致、是否易懂。

---

## 四、元数据完整性（供下游溯源）

入库后每条 chunk 带有：

- `doc_name`、`doc_timestamp`、`chunk_id`、`chunk_index`  
- `source_file`、`space_id`、`source_url`（飞书来源）  
- `page_number`、`has_images`、`images_json`（图片引用）

**质量检查**：抽样查库或看 `metadata/chunk_*.json`，确认 `source_url`、`page_number`、`images_json` 是否齐全，便于召回后跳转原文或定位图片。

---

## 五、图谱质量（LightRAG）

- **chunk_token_size**  
  - 图谱构建时按多大块喂给 LLM 做实体/关系抽取，影响实体粒度和关系密度。

- **LLM 模型与 prompt**  
  - 实体、关系抽取效果依赖模型能力，一般与“图片命名”共用 `doc_splitter.yaml` 里 `llm` 配置。

- **如何自查**  
  - 看 `lightrag_workspace/` 下产出（或 PG 中 lightrag 相关表）：实体名是否合理、关系是否与文档一致、是否大量无关实体。

---

## 六、建议的质量评估流程

1. **抽样文档**  
   选 3～5 篇有代表性的文档（含长文、表格、多图）。

2. **看拆分**  
   打开 `processed/{文档名}/chunks/` 与 `chunks_index.json`，看边界、长度、表格是否被拆乱。

3. **看图片**  
   看 `images/` 与 `images_index.json`，检查命名是否语义化、上下文是否匹配。

4. **看元数据**  
   查库或看 `metadata/chunk_*.json`，确认 source、页码、图片引用完整。

5. **看图谱（若开启）**  
   看实体/关系抽样是否与文档内容一致、是否有明显噪音。

6. **调参复跑**  
   根据问题调整 `chunk_size`/`overlap`/`separators` 或 `image_naming`、LightRAG 参数，重新跑 process/store/graph，再抽样对比。

---

## 七、与召回的关系（仅说明边界）

- 本工程**不负责**：检索策略、相似度阈值、重排序、多路召回融合等。  
- 本工程**负责**：产出**高质量、结构清晰、元数据完整**的 chunk、embedding、图谱，供下游 RAG/检索服务使用。  
- 预处理质量好，下游召回与展示的上限会更高；质量评估以“抽样 + 上述检查项”为主即可。
