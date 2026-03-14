# 飞书文档自动同步工具 — 部署使用手册

飞书知识空间 → 下载 → 拆分 → 向量化 → pgvector 入库 + LightRAG 知识图谱

---

## 目录结构

```
部署根目录/
├── deploy.bat                  ← 定时任务管理（需管理员）
├── start.bat                   ← 手动执行入口
├── _runtime/                   ← 运行时数据（日志/清单/锁）
├── processed/                  ← 处理结果（chunks/embeddings）
├── lightrag_workspace/         ← 图谱数据（实体/关系/graphml）
├── *.docx / *.pdf              ← 下载的原始文档
└── auto-doc-process/           ← 程序目录（一般不需要进入）
    ├── configs/                ← 配置文件
    ├── models/onnx/            ← Embedding 模型
    ├── tools/                  ← 工具脚本
    ├── venv/                   ← Python 虚拟环境
    ├── run.py                  ← 程序入口
    └── setup.bat               ← 首次部署（创建 venv + 安装依赖）
```

---

## 首次部署

### 1. 创建虚拟环境

进入 `auto-doc-process/` 目录，双击 `setup.bat`，自动创建 venv 并安装依赖。

### 2. 填写配置

`auto-doc-process/configs/` 下三个文件，从 `.example` 复制后修改：

| 文件 | 必填项 |
|------|--------|
| `feishu.yaml` | `app_id`、`app_secret`（飞书应用凭证） |
| `db_info.yml` | `host`、`port`、`database`、`user`、`password` |
| `doc_splitter.yaml` | `llm.api_key`（DeepSeek API Key） |

示例：

```yaml
# feishu.yaml
feishu:
  app_id: "cli_xxxxxxxx"
  app_secret: "xxxxxxxxxxxxxxxx"

# db_info.yml
database:
  host: localhost
  port: 5432
  database: twd
  user: postgres
  password: "你的密码"

# doc_splitter.yaml
llm:
  api_key: "sk-xxxxxxxxxxxxxxxx"
```

> 飞书应用需开通权限：`wiki:wiki:readonly`、`docx:document:readonly`、`drive:drive:readonly`

### 3. 注册定时任务

回到**部署根目录**，以**管理员身份**打开 CMD：

```cmd
deploy.bat install
```

首次会自动：检查环境 → 全量同步一次 → 注册每天 02:00 的定时任务。

---

## 日常命令

所有命令在**部署根目录**执行。

### start.bat — 手动执行

```cmd
start.bat                                全流程（下载→处理→入库→图谱）
start.bat --step download                只下载
start.bat --step process                 只处理（拆分+向量化）
start.bat --step store                   只入库
start.bat --step graph                   只构建图谱
start.bat --step download,process,store  组合执行（跳过图谱）
start.bat --full                         全量同步（忽略增量记录）
start.bat --dry-run                      预览模式（只列文档不执行）
start.bat --reset-db                     清空数据库后全量重建
start.bat --no-graph                     跳过图谱构建
```

### start.bat — 运维

```cmd
start.bat stop                           停止进程 + 清理锁文件
start.bat status                         查看进程状态和锁文件
start.bat reset                          清理所有构建产物（不重建）
```

### deploy.bat — 定时任务管理（需管理员）

```cmd
deploy.bat install       注册定时任务（每天 02:00）
deploy.bat status        查看任务状态和下次执行时间
deploy.bat run           立即触发执行一次
deploy.bat stop          暂停定时 + 终止当前进程
deploy.bat start         恢复定时
deploy.bat uninstall     删除定时任务
```

---

## 处理管线

```
download  →  process  →  store  →  graph
 下载文档     拆分+向量化   pgvector入库   LightRAG图谱
```

- **增量模式**（默认）：只处理新增或修改的文档
- **全量模式**（`--full`）：忽略增量记录，重新处理所有文档
- **重建模式**（`--reset-db`）：清空数据库 + 全量重建

---

## 性能调优

`auto-doc-process/configs/` 中可调的性能参数：

### lightrag.yaml

| 参数 | 默认 | 说明 |
|------|------|------|
| `llm.max_async` | 12 | LLM 并发请求数（仅网络，不占本地资源） |
| `graph.max_parallel_insert` | 4 | 文档并行插入数（占内存，低配别调大） |
| `graph.chunk_token_size` | 1500 | 图谱分块大小（影响召回精度，勿随意修改） |
| `performance.entity_embed_batch` | 32 | 实体 embedding 批次（OOM 调小） |
| `performance.max_crash_restarts` | 5 | 子进程崩溃最大重启次数 |

### doc_splitter.yaml

| 参数 | 默认 | 说明 |
|------|------|------|
| `embedding.batch_size` | 16 | 文档 embedding 批次（OOM 调小，如 8） |
| `doc_splitter.text_splitter.chunk_size` | 1000 | 文本块大小（字符） |
| `doc_splitter.text_splitter.chunk_overlap` | 200 | 文本块重叠（字符） |

### feishu.yaml

| 参数 | 默认 | 说明 |
|------|------|------|
| `export.poll_max_wait` | 300 | 导出任务最大等待（秒） |
| `schedule.run_time` | 02:00 | 定时任务执行时间 |

---

## 稳定性机制

- **子进程隔离** — 每个文档独立子进程处理，C 层崩溃不影响主进程
- **崩溃自动恢复** — 识别问题文档并跳过，继续处理剩余文档
- **Embedding 分批** — 防止大文档 OOM
- **事务原子性** — pgvector 入库使用事务 + advisory lock，无脏数据
- **文件锁** — 防止多实例并发执行
- **增量同步** — 通过 manifest 跟踪文档变化，只处理新增/修改
- **环境预检** — 启动时自动检查磁盘/数据库/模型，失败则报告原因
- **LLM 缓存** — 相同文本不重复调用 LLM（图谱构建阶段）

---

## 常见问题

| 问题 | 解决 |
|------|------|
| `[ERROR] venv not found` | 进入 `auto-doc-process/` 运行 `setup.bat` |
| `Preflight failed` | 按提示检查配置文件和数据库连接 |
| `[FAIL] Run as Administrator` | `deploy.bat` 需要管理员权限，右键 CMD 以管理员运行 |
| 飞书权限不足 | 确认应用已开通 wiki/docx/drive 三个只读权限 |
| 数据库连接失败 | 检查 PostgreSQL 服务是否启动 + `db_info.yml` 配置 |
| vector 扩展缺失 | 在 PG 中执行 `CREATE EXTENSION vector;` |
| 首次图谱构建很慢 | 正常（需对每个文档做 LLM 抽取），后续增量更新很快 |
| 某个文档处理崩溃 | 日志中搜 `子进程崩溃` 查看问题文档，其余文档不受影响 |
| 清理所有数据重来 | `start.bat reset` 清理后 `start.bat --full` 重建 |
| 机器重启后任务丢失 | 不会。`schtasks` 注册的任务持久化在系统中 |
| 错过执行时间（如关机） | 任务计划程序 → FeishuDocSync → 属性 → 设置 → 勾选"过了计划时间立即启动" |

