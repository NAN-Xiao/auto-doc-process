# RAG 系统对接文档

本文档面向需要从 **auto-doc-process** 产出的数据中进行召回的外部 RAG 系统。

---

## 数据总览

auto-doc-process 输出两类可消费的数据源：

| 数据源 | 位置 | 适合场景 |
|--------|------|----------|
| **本地文件**（chunks + embeddings + 图谱） | `processed/` 目录 | 文件监控、轻量部署、无 DB 依赖 |
| **PostgreSQL 表**（doc_chunks + 图谱表） | 配置的数据库 | SQL 查询、pgvector 向量检索、全文检索 |

---

## 一、本地文件数据

### 1.1 目录结构

```
processed/
├── 文档A/                          ← 每个文档一个目录（目录名 = 文档标题）
│   ├── doc_info.json               ← 文档元数据
│   ├── chunks_index.json           ← chunk 索引（不含正文）
│   ├── images_index.json           ← 图片索引
│   ├── chunks/                     ← 文本块（纯文本）
│   │   ├── chunk_0000.txt
│   │   ├── chunk_0001.txt
│   │   └── ...
│   ├── images/                     ← 提取的图片
│   │   ├── 流程图_系统架构.png
│   │   └── ...
│   ├── embeddings/                 ← 向量文件（每个 chunk 一个 JSON）
│   │   ├── chunk_0000.json
│   │   └── ...
│   └── metadata/                   ← chunk 元数据（每个 chunk 一个 JSON）
│       ├── chunk_0000.json
│       └── ...
│
├── 文档B/
│   └── ...
│
├── lightrag_workspace/             ← LightRAG 知识图谱数据
│   ├── graph_chunk_entity_relation.graphml   ← NetworkX 图谱（XML）
│   ├── kv_store_*.json             ← KV 存储（实体/关系描述）
│   ├── vdb_*.json                  ← NanoVectorDB 向量索引
│   ├── doc_status.json             ← 文档处理状态
│   ├── .writing                    ← 写入中标记（存在时勿读）
│   └── .ready                      ← 写入完成信号
│
├── lightrag_report.json            ← 最近一次图谱构建报告
└── _graph_manifest.json            ← 增量构建清单（内部用）
```

### 1.2 文件格式

#### doc_info.json — 文档元数据

```json
{
  "filename": "游戏系统设计.docx",
  "format": "Word",
  "total_pages": 5,
  "total_chunks": 23,
  "total_images": 4,
  "created_at": "2026-03-15T02:00:12.345678",
  "output_dir": "D:/work/processed/游戏系统设计"
}
```

#### chunks_index.json — chunk 索引

```json
{
  "total_chunks": 23,
  "chunks": [
    {
      "chunk_id": "a1b2c3d4e5f6",
      "index": 0,
      "char_count": 856,
      "page_number": 1,
      "metadata": { "source": "游戏系统设计.docx" }
    },
    {
      "chunk_id": "f6e5d4c3b2a1",
      "index": 1,
      "char_count": 1023,
      "page_number": 1,
      "metadata": { "source": "游戏系统设计.docx" }
    }
  ]
}
```

> `chunk_id` 为 MD5 前 12 位，基于 chunk 内容 + 序号生成，全局唯一。

#### chunks/chunk_NNNN.txt — chunk 正文

纯 UTF-8 文本，直接 `open().read()` 即可。文本中的图片以如下占位符表示：

```
图片：./images/流程图_系统架构.png
```

RAG 系统可选择保留或过滤这些占位符。

#### embeddings/chunk_NNNN.json — 向量

```json
{
  "chunk_index": 0,
  "embedding": [0.0123, -0.0456, 0.0789, "...共 512 维"],
  "model": "BAAI/bge-small-zh-v1.5",
  "created_at": "2026-03-15T02:01:05.123456"
}
```

- **模型**: `BAAI/bge-small-zh-v1.5`
- **维度**: 512
- **归一化**: 是（可直接用余弦相似度或内积）

#### metadata/chunk_NNNN.json — chunk 元数据

```json
{
  "id": "游戏系统设计_20260315_020012_a1b2c3d4e5f6",
  "doc_name": "游戏系统设计",
  "timestamp": "20260315_020012",
  "chunk_index": 0,
  "metadata": {
    "source": "游戏系统设计.docx",
    "chunk_id": "a1b2c3d4e5f6",
    "chunk_index": 0,
    "char_count": 856,
    "doc_format": "Word",
    "doc_timestamp": "20260315_020012",
    "processed_at": "2026-03-15T02:00:12.345678",
    "has_images": true,
    "image_count": 1,
    "images": [
      {
        "filename": "流程图_系统架构.png",
        "original_filename": "image_001.png",
        "path": "./images/流程图_系统架构.png",
        "context_before": "如下图所示，系统整体架构...",
        "context_after": "该架构分为三个核心模块..."
      }
    ]
  },
  "created_at": "2026-03-15T02:01:05.123456"
}
```

### 1.3 读取文件数据（Python 示例）

```python
import json
from pathlib import Path
import numpy as np

PROCESSED_DIR = Path("processed")

def load_all_chunks():
    """加载全部文档的 chunks + embeddings，返回可用于检索的列表"""
    results = []
    for doc_dir in PROCESSED_DIR.iterdir():
        if not doc_dir.is_dir() or doc_dir.name.startswith("."):
            continue

        # 读取文档信息
        doc_info_file = doc_dir / "doc_info.json"
        if not doc_info_file.exists():
            continue
        with open(doc_info_file, "r", encoding="utf-8") as f:
            doc_info = json.load(f)

        # 逐个 chunk 加载
        chunks_dir = doc_dir / "chunks"
        embeddings_dir = doc_dir / "embeddings"
        metadata_dir = doc_dir / "metadata"

        for chunk_file in sorted(chunks_dir.glob("chunk_*.txt")):
            idx = int(chunk_file.stem.split("_")[1])

            text = chunk_file.read_text(encoding="utf-8")
            if not text.strip():
                continue

            # 向量
            emb_file = embeddings_dir / f"chunk_{idx:04d}.json"
            embedding = None
            if emb_file.exists():
                with open(emb_file, "r", encoding="utf-8") as f:
                    embedding = json.load(f)["embedding"]

            # 元数据
            meta_file = metadata_dir / f"chunk_{idx:04d}.json"
            metadata = {}
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f).get("metadata", {})

            results.append({
                "doc_name": doc_dir.name,
                "chunk_index": idx,
                "text": text,
                "embedding": embedding,
                "metadata": metadata,
            })

    return results


def search(query_embedding, chunks, top_k=5):
    """余弦相似度检索"""
    q = np.array(query_embedding, dtype=np.float32)
    q = q / np.linalg.norm(q)

    scores = []
    for i, chunk in enumerate(chunks):
        if chunk["embedding"] is None:
            continue
        v = np.array(chunk["embedding"], dtype=np.float32)
        v = v / np.linalg.norm(v)
        scores.append((i, float(np.dot(q, v))))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in scores[:top_k]]
```

---

## 二、PostgreSQL 数据

### 2.1 表结构

#### doc_chunks — 文档向量表（主表）

```sql
CREATE TABLE doc_chunks (
    id                SERIAL PRIMARY KEY,
    doc_name          TEXT    NOT NULL DEFAULT '',     -- 文档名（目录名）
    doc_format        TEXT    NOT NULL DEFAULT '',     -- Word / PDF
    doc_timestamp     TEXT    NOT NULL DEFAULT '',     -- 处理批次时间戳
    chunk_id          TEXT    NOT NULL DEFAULT '',     -- chunk 唯一 ID（MD5 前 12 位）
    chunk_index       INTEGER NOT NULL DEFAULT 0,     -- chunk 在文档中的序号
    chunk_text        TEXT    NOT NULL,                -- chunk 正文
    char_count        INTEGER NOT NULL DEFAULT 0,     -- 字符数
    page_number       INTEGER,                        -- 所在页码
    has_images        BOOLEAN NOT NULL DEFAULT FALSE,  -- 是否包含图片引用
    image_count       INTEGER NOT NULL DEFAULT 0,     -- 图片引用数
    images_json       TEXT    DEFAULT '[]',            -- 图片信息 JSON 数组
    source_file       TEXT    NOT NULL DEFAULT '',     -- 源文件名
    space_id          TEXT    NOT NULL DEFAULT '',     -- 飞书空间 ID
    source_url        TEXT    NOT NULL DEFAULT '',     -- 飞书文档 URL
    embedding         vector(512),                    -- bge-small-zh-v1.5 向量
    processed_at      TEXT    DEFAULT '',              -- 处理时间
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_doc_chunks_doc_name   ON doc_chunks (doc_name);
CREATE INDEX idx_doc_chunks_chunk_id   ON doc_chunks (chunk_id);
CREATE INDEX idx_doc_chunks_space_id   ON doc_chunks (space_id);
CREATE INDEX idx_doc_chunks_embedding  ON doc_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_doc_chunks_fulltext   ON doc_chunks USING gin (to_tsvector('simple', chunk_text));
```

#### lightrag_entities — 知识图谱实体表（可选，默认关闭）

```sql
CREATE TABLE lightrag_entities (
    id              SERIAL PRIMARY KEY,
    entity_name     TEXT NOT NULL,                    -- 实体名
    entity_type     TEXT DEFAULT '',                  -- 实体类型
    description     TEXT DEFAULT '',                  -- 描述
    source_doc      TEXT DEFAULT '',                  -- 来源文档
    source_chunk_id TEXT DEFAULT '',                  -- 来源 chunk ID
    embedding       vector(512),                      -- 实体向量
    batch_timestamp TEXT DEFAULT '',                  -- 构建批次
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_name, batch_timestamp)
);
```

#### lightrag_relations — 知识图谱关系表（可选，默认关闭）

```sql
CREATE TABLE lightrag_relations (
    id              SERIAL PRIMARY KEY,
    source_entity   TEXT NOT NULL,                    -- 源实体
    target_entity   TEXT NOT NULL,                    -- 目标实体
    relation_type   TEXT DEFAULT '',                  -- 关系类型/描述
    description     TEXT DEFAULT '',                  -- 关键词
    weight          FLOAT DEFAULT 1.0,               -- 关系权重
    source_doc      TEXT DEFAULT '',                  -- 来源
    batch_timestamp TEXT DEFAULT '',                  -- 构建批次
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### lightrag_chunks — 图谱关联的文本块（可选，默认关闭）

```sql
CREATE TABLE lightrag_chunks (
    id              SERIAL PRIMARY KEY,
    chunk_id        TEXT NOT NULL,
    doc_name        TEXT NOT NULL,
    content         TEXT NOT NULL,
    token_count     INTEGER DEFAULT 0,
    chunk_order     INTEGER DEFAULT 0,
    full_doc_id     TEXT DEFAULT '',
    batch_timestamp TEXT DEFAULT '',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

> `lightrag_entities`/`lightrag_relations`/`lightrag_chunks` 默认关闭（`lightrag.yaml` 中 `pg_export.enabled: false`）。
> 如果你的 RAG 系统需要图谱数据入库，联系管理员开启。

### 2.2 向量相似度检索（SQL）

```sql
-- 余弦相似度检索 Top 10
SELECT
    doc_name,
    chunk_index,
    chunk_text,
    source_url,
    1 - (embedding <=> $1::vector) AS similarity
FROM doc_chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

> `$1` 为 512 维 query 向量。`<=>` 是 pgvector 的余弦距离运算符，`1 - distance = similarity`。

### 2.3 全文检索（BM25 关键词匹配）

```sql
-- 关键词全文检索
SELECT
    doc_name,
    chunk_index,
    chunk_text,
    ts_rank(to_tsvector('simple', chunk_text), query) AS rank
FROM doc_chunks, plainto_tsquery('simple', '战斗系统 技能') AS query
WHERE to_tsvector('simple', chunk_text) @@ query
ORDER BY rank DESC
LIMIT 10;
```

### 2.4 混合检索（向量 + 关键词）

```sql
-- 向量召回 + 关键词过滤
SELECT
    doc_name,
    chunk_index,
    chunk_text,
    1 - (embedding <=> $1::vector) AS vec_sim
FROM doc_chunks
WHERE embedding IS NOT NULL
  AND chunk_text LIKE '%技能%'
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

### 2.5 按文档过滤

```sql
-- 只搜索特定文档
SELECT chunk_text, 1 - (embedding <=> $1::vector) AS similarity
FROM doc_chunks
WHERE doc_name = '游戏系统设计'
  AND embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 5;

-- 按飞书空间过滤
SELECT chunk_text, 1 - (embedding <=> $1::vector) AS similarity
FROM doc_chunks
WHERE space_id = '7613735903789370589'
  AND embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 5;
```

### 2.6 获取图片上下文

```sql
-- 查询包含图片的 chunks
SELECT
    doc_name,
    chunk_text,
    images_json
FROM doc_chunks
WHERE has_images = true
  AND doc_name = '游戏系统设计';
```

`images_json` 为 JSON 数组，每个元素：

```json
{
  "filename": "流程图_系统架构.png",
  "original_filename": "image_001.png",
  "path": "./images/流程图_系统架构.png",
  "context_before": "...",
  "context_after": "..."
}
```

实际图片文件位于 `processed/{doc_name}/images/{filename}`。

### 2.7 Python 接入示例

```python
import psycopg
from pgvector.psycopg import register_vector

def search_chunks(query_embedding, top_k=5):
    conn = psycopg.connect(
        host="localhost", port=5432,
        dbname="slgRAG", user="postgres", password="your_password",
    )
    register_vector(conn)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT doc_name, chunk_index, chunk_text, source_url,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM doc_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))

        results = []
        for row in cur.fetchall():
            results.append({
                "doc_name": row[0],
                "chunk_index": row[1],
                "text": row[2],
                "source_url": row[3],
                "similarity": float(row[4]),
            })

    conn.close()
    return results
```

---

## 三、LightRAG 图谱数据

### 3.1 文件说明

图谱数据位于 `processed/lightrag_workspace/`：

| 文件 | 格式 | 说明 |
|------|------|------|
| `graph_chunk_entity_relation.graphml` | GraphML (XML) | 实体-关系图谱，可用 NetworkX 加载 |
| `kv_store_full_docs.json` | JSON | 完整文档内容缓存 |
| `kv_store_text_chunks.json` | JSON | 文本块缓存 |
| `kv_store_llm_response_cache.json` | JSON | LLM 响应缓存（加速增量构建） |
| `vdb_entities.json` | JSON | 实体向量索引（NanoVectorDB） |
| `vdb_relationships.json` | JSON | 关系向量索引 |
| `vdb_chunks.json` | JSON | chunk 向量索引 |
| `doc_status.json` | JSON | 已处理文档的状态记录 |

### 3.2 加载图谱（Python 示例）

```python
import networkx as nx

graph = nx.read_graphml("processed/lightrag_workspace/graph_chunk_entity_relation.graphml")

# 查看所有实体
for node_id, data in graph.nodes(data=True):
    print(f"实体: {node_id}, 类型: {data.get('entity_type', '')}")
    print(f"  描述: {data.get('description', '')[:100]}")

# 查看所有关系
for src, tgt, data in graph.edges(data=True):
    print(f"{src} --[{data.get('description', '')}]--> {tgt}")

# 查询某个实体的关联
entity = "战斗系统"
if entity in graph:
    neighbors = list(graph.neighbors(entity))
    print(f"'{entity}' 关联实体: {neighbors}")
```

---

## 四、读写协调协议

auto-doc-process 定时运行（默认每天 02:00），运行期间会写入上述文件和数据库。外部 RAG 系统需注意以下协调机制。

### 4.1 LightRAG 工作目录（文件级）

| 信号文件 | 含义 | RAG 读端动作 |
|----------|------|-------------|
| `.writing` 存在 | 正在写入，数据不完整 | **等待或跳过**，不读取目录内容 |
| `.ready` 存在 | 写入完成，数据完整 | 可安全读取，检查时间戳判断是否需要重新加载 |
| 两者都不存在 | 尚未构建 | 无数据可用 |

#### .ready 文件内容

```json
{
  "completed_at": "2026-03-15T02:15:30.123456",
  "pid": 12345,
  "summary": {
    "built": 5,
    "skipped": 18,
    "batch_timestamp": "20260315_020012"
  }
}
```

#### 推荐轮询逻辑

```python
import json, time
from pathlib import Path

WORKSPACE = Path("processed/lightrag_workspace")
last_completed_at = None

def check_and_reload():
    global last_completed_at

    writing = WORKSPACE / ".writing"
    ready = WORKSPACE / ".ready"

    # 正在写入，跳过
    if writing.exists():
        return False

    # 检查 .ready
    if not ready.exists():
        return False

    with open(ready, "r", encoding="utf-8") as f:
        info = json.load(f)

    completed_at = info.get("completed_at")
    if completed_at == last_completed_at:
        return False  # 未更新

    # 有新数据，重新加载
    last_completed_at = completed_at
    reload_graph()
    return True

def reload_graph():
    """重新加载图谱数据"""
    # ... 你的加载逻辑 ...
    pass

# 定时检查（如每 5 分钟）
while True:
    if check_and_reload():
        print("图谱已更新，已重新加载")
    time.sleep(300)
```

### 4.2 PostgreSQL 表（数据库级）

数据库写入使用 **事务 + advisory lock**，天然保证一致性：

- **写入时**：使用 `pg_advisory_xact_lock` 排他锁，防止并发写
- **读取时**：PostgreSQL MVCC 机制保证读不阻塞写、写不阻塞读
- **可见性**：读端看到的要么是旧的完整数据，要么是新的完整数据，不会看到半写

> RAG 系统直接 SELECT 即可，无需额外协调。

### 4.3 文件级写入安全

所有 JSON 文件写入使用原子操作：
1. 写入临时文件（`.tmp` 后缀）
2. `os.replace()` 原子替换目标文件

读端在任何时刻读到的都是完整文件，不会读到半写内容。

---

## 五、Embedding 模型信息

| 项目 | 值 |
|------|------|
| 模型 | `BAAI/bge-small-zh-v1.5` |
| 维度 | 512 |
| 归一化 | 是 |
| 相似度 | 余弦相似度（或内积，已归一化时等价） |
| 语言 | 中文优化 |

**重要**：你的 RAG 系统生成 query embedding 时，必须使用相同模型（`BAAI/bge-small-zh-v1.5`），否则向量空间不对齐，检索无效。

```python
# 使用 sentence-transformers 生成 query embedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
query_embedding = model.encode("战斗系统如何设计").tolist()
```

---

## 六、数据更新频率

| 事件 | 频率 | 影响范围 |
|------|------|----------|
| 飞书文档同步 | 每天 02:00（可配置） | `documents/` 目录 |
| 文档处理 | 同步后自动触发 | `processed/{doc}/` 各子目录 |
| pgvector 入库 | 处理后自动触发 | `doc_chunks` 表 |
| 图谱构建 | 入库后自动触发 | `lightrag_workspace/` + 图谱表 |

增量模式下，只有新增/修改的文档会被重新处理和入库。

---

## 七、快速接入检查清单

- [ ] 确认 Embedding 模型一致（`BAAI/bge-small-zh-v1.5`，512 维）
- [ ] 选择数据源：文件读取 or PostgreSQL 查询
- [ ] 如果使用 PostgreSQL：确认可连接数据库，安装 `pgvector` 扩展
- [ ] 如果使用文件：实现 `.writing` / `.ready` 检查逻辑
- [ ] 实现 query embedding 生成（使用相同模型）
- [ ] 实现 Top-K 向量检索（余弦相似度）
- [ ] （可选）实现混合检索：向量 + 关键词全文检索
- [ ] （可选）加载 LightRAG 图谱做实体/关系增强召回
- [ ] 测试：确认召回结果中 `chunk_text` 内容正确、相似度合理
