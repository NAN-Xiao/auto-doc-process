# RAG 接入说明

本文面向下游 RAG / 检索系统，说明本工程处理后的产物结构、推荐召回方式，以及如何快速接入 PostgreSQL `doc_chunks`。

## 接入目标

这版预处理不再只输出“普通文本块”，而是把文档拆成更适合召回的知识单元：

- `section`
  章节正文块，适合规则说明、流程说明、概念解释类问题
- `table_summary`
  表格摘要块，适合先把某张表召回出来
- `table_row`
  表格行块，适合字段查询、条件查询、数值查询

同时，`docx` 预处理会先进行轻量文档类型识别，当前会标记为：

- `article`
  正文/说明型文档
- `spec_mixed`
  设计说明、规则、链接、列表、表格混合型文档
- `table_heavy`
  表格主导型文档
- `image_heavy`
  图片或截图提示语主导型文档

这些类型会同时写入：

- 本地处理目录 `processed/<doc_name>/chunks_index.json`
- 本地处理目录 `processed/<doc_name>/metadata/chunk_*.json`
- PostgreSQL 表 `doc_chunks`

## 推荐召回策略

建议使用 hybrid retrieval，而不是只做向量检索。

推荐顺序：

1. 先按查询意图做轻量分类
   - 说明类问题：偏向 `section`
   - 配置/字段类问题：偏向 `table_summary` + `table_row`
   - 精确值/ID/枚举类问题：偏向 `table_row` + 关键词检索
2. 同时做两路召回
   - 向量召回：查语义相近内容
   - 关键词/BM25：查字段名、ID、术语、枚举值
3. 用 metadata 做过滤和加权
   - `content_type`
   - `section_path`
   - `title`
   - `keywords_json`
4. 命中 `table_row` 后，回查它的父块
   - 通过 `parent_chunk_id` 找对应的 `table_summary`
   - 通过 `section_path` 补齐章节上下文

## doc_chunks 关键字段

### 基础字段

- `doc_name`
  文档名，适合做溯源和按文档过滤
- `source_doc_id`
  稳定的文档来源 ID，优先基于上游 `source_url` 生成，适合增量同步和跨批次对齐
- `chunk_id`
  当前 chunk 唯一 ID
- `chunk_index`
  当前 chunk 在文档内的顺序
- `chunk_text`
  主检索文本
- `chunk_token_count`
  token 数，可用于超长过滤
- `content_quality_score`
  内容质量分，可用于低质量过滤

### 召回增强字段

- `content_type`
  值为 `section` / `table_summary` / `table_row`
- `doc_type`
  文档类型画像，例如 `article` / `spec_mixed` / `table_heavy` / `image_heavy`
- `title`
  当前 chunk 的标题或主题
- `section_path`
  章节路径，例如 `玩法系统 > GVG > 报名规则`
- `parent_chunk_id`
  子块的父块 ID，当前主要用于 `table_row -> table_summary`
- `table_index`
  文档内表格编号
- `table_headers_json`
  `JSONB` 表头数组
- `row_data_json`
  `JSONB` 表格行的结构化键值对
- `keywords_json`
  `JSONB` 关键词数组，可用于过滤和 rerank

### 图片与图文增强字段

- `has_images`
- `image_count`
- `images_json`
  `JSONB` 图片详情数组
- `image_titles_json`
  `JSONB` 图片标题数组

### 版本与增量字段

- `processed_batch_id`
- `doc_version_hash`
- `source_updated_at`

入库侧生产行为：

- 优先按 `source_doc_id` 替换旧版本，而不是只按 `doc_name`
- 如果库里已有相同 `source_doc_id` 且 `doc_version_hash` 未变化，入库阶段会跳过重复写入

## 三种 chunk 的内容设计

### 1. `section`

示例：

```text
GVG报名规则

报名时间为每周六20:00前。
会长或副会长可发起报名。
报名人数需达到20人。
```

用途：

- 回答“规则是什么”
- 回答“某个机制怎么触发”
- 提供回答时的自然语言上下文

### 2. `table_summary`

示例：

```text
表格编号：3
表格主题：GVG报名配置
所属章节：玩法系统 > GVG > 报名规则
表头：段位 | 最小人数 | 开启时间 | 奖励系数
数据行数：5
示例行：
王者 | 20 | 周六20:00 | 1.5
钻石 | 15 | 周六20:00 | 1.2
```

用途：

- 先把相关表召回出来
- 帮 rerank 判断问题是否与这张表相关

### 3. `table_row`

示例：

```text
表格主题：GVG报名配置
所属章节：玩法系统 > GVG > 报名规则
表头：段位 | 最小人数 | 开启时间 | 奖励系数
行范围：1-1
段位=王者 | 最小人数=20 | 开启时间=周六20:00 | 奖励系数=1.5
```

用途：

- 回答“王者段位最小人数是多少”
- 回答“奖励系数是多少”
- 回答“配置表里某字段的取值”

## 推荐 SQL 用法

### 按类型过滤

```sql
SELECT chunk_id, content_type, title, section_path, chunk_text
FROM doc_chunks
WHERE content_type IN ('table_summary', 'table_row')
  AND source_doc_id = $1;
```

### 命中行块后回查父块

```sql
SELECT parent.chunk_id, parent.title, parent.section_path, parent.chunk_text
FROM doc_chunks child
JOIN doc_chunks parent
  ON parent.chunk_id = child.parent_chunk_id
WHERE child.chunk_id = $1;
```

### 关键词召回

```sql
SELECT chunk_id, content_type, title, section_path, chunk_text
FROM doc_chunks
WHERE to_tsvector('simple', chunk_text) @@ plainto_tsquery('simple', $1)
ORDER BY chunk_index
LIMIT 20;
```

### JSONB 过滤示例

```sql
SELECT chunk_id, title, row_data_json
FROM doc_chunks
WHERE content_type = 'table_row'
  AND source_doc_id = $1
  AND row_data_json @> '{"段位": "王者"}'::jsonb;
```

### 关键词数组过滤示例

```sql
SELECT chunk_id, title, keywords_json
FROM doc_chunks
WHERE keywords_json ? 'GVG'
  AND content_type IN ('section', 'table_summary');
```

## 推荐融合逻辑

一个简单可用的融合策略：

1. 向量召回 topK=20
2. BM25/全文检索 topK=20
3. 合并去重
4. 规则加权
   - `content_type='table_row'` 且 query 包含字段词时加分
   - `section_path` 命中 query 主题词时加分
   - `content_quality_score < 0.55` 时降权
5. 若命中 `table_row`，附带其 `table_summary`

## 本地产物读取方式

如果不直接接 PostgreSQL，也可以读 `processed/<doc_name>/metadata/chunk_*.json`。

每个 metadata 文件里有：

- `id`
- `chunk_index`
- `metadata.content_type`
- `metadata.title`
- `metadata.section_path`
- `metadata.parent_chunk_id`
- `metadata.table_headers`
- `metadata.row_data`
- `metadata.keywords`

这套字段和 PostgreSQL 里的列含义保持一致。

## 接入建议

最小可用接法建议这样做：

1. 检索主文本使用 `chunk_text`
2. 过滤和 rerank 使用 `content_type`、`section_path`、`title`
3. 如果需要精确过滤，优先使用 `source_doc_id` 而不是 `doc_name`
4. 精确问答优先关注 `table_row`
5. 命中 `table_row` 后带出父 `table_summary`
6. 回答展示时附带 `doc_name + section_path + title`

这样接入以后，系统对“规则说明”和“配置查询”两类问题都会稳很多。
