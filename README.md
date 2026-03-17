# 飞书文档自动同步工具

飞书知识空间 → 下载 → 拆分 → 向量化 → pgvector 入库 + LightRAG 知识图谱

---

## 目录结构

```
部署根目录/
├── install.bat                 ← 双击安装定时任务
├── uninstall.bat               ← 双击卸载定时任务
├── start.bat                   ← 手动执行
├── deploy.bat                  ← 高级管理
│
├── _runtime/                   ← 运行时（日志/锁）
├── processed/                  ← 处理结果（chunks/embeddings）
├── lightrag_workspace/         ← 图谱数据（实体/关系）
│   ├── .writing                ← 写入中标记（存在时勿读取）
│   └── .ready                  ← 写入完成信号
├── *.docx                      ← 下载的原始文档
│
└── auto-doc-process/           ← 程序目录
    ├── configs/                ← 配置文件 ★
    ├── models/                 ← Embedding 模型
    ├── venv/                   ← Python 环境
    ├── run.py                  ← 程序入口
    └── setup.bat               ← 首次部署
```

---

## 快速部署（3 步）

### 1. 环境准备

进入 `auto-doc-process/`，双击 **`setup.bat`**（自动创建 venv + 安装依赖）。

> 如果是 `--include-venv` 打包的版本，跳过此步。

### 2. 填写配置

进入 `auto-doc-process/configs/`，把 4 个 `.example` 文件各复制一份去掉后缀：

| 复制为 | 必填项 | 获取方式 |
|--------|--------|----------|
| `feishu.yaml` | `app_id`、`app_secret` | [飞书开放平台](https://open.feishu.cn) → 自建应用 → 凭证 |
| `db_info.yml` | `database`、`password` | PostgreSQL 14+，需 pgvector 扩展 |
| `doc_splitter.yaml` | `llm.api_key`、`llm.api_base` | 默认 DeepSeek，可换任何 OpenAI 兼容 API |
| `lightrag.yaml` | 一般无需改 | 默认即可，可关闭图谱 `enabled: false` |

最小配置示例：

```yaml
# feishu.yaml
feishu:
  app_id: "cli_xxxxxxxx"
  app_secret: "xxxxxxxxxxxxxxxx"

# db_info.yml
database:
  host: localhost
  port: 5432
  database: myRAG
  user: postgres
  password: "你的密码"

# doc_splitter.yaml — LLM 配置（图片命名、图谱构建共用）
llm:
  api_key: "sk-xxxxxxxxxxxxxxxx"
  api_base: "https://api.deepseek.com"    # 默认 DeepSeek，换其他 LLM 改这里
  model: "deepseek-chat"                  # 模型名
```

> 飞书应用需开通权限：`wiki:wiki:readonly`、`docx:document:readonly`、`drive:drive:readonly`

### 3. 安装定时任务

回到**部署根目录**，双击 **`install.bat`**（自动请求管理员权限）。

首次会自动：预检环境 → 全量同步一次 → 注册每天定时任务。

卸载：双击 **`uninstall.bat`**。

---

## 命令速查

### start.bat — 手动执行

```
start.bat                            全流程（下载→处理→入库→图谱）
start.bat --step download            只下载
start.bat --step process             只处理
start.bat --step store               只入库
start.bat --step graph               只构建图谱
start.bat --step download,process    组合执行
start.bat --full                     全量同步（忽略增量记录）
start.bat --reset-db                 清空数据库 + 全量重建
start.bat --dry-run                  预览模式（不执行）
start.bat --no-graph                 跳过图谱构建
```

### start.bat — 运维

```
start.bat stop                       停止进程
start.bat status                     查看状态
start.bat reset                      清理所有产物
```

### deploy.bat — 高级管理（需管理员）

```
deploy.bat install       注册定时任务
deploy.bat uninstall     删除定时任务
deploy.bat run           立即触发一次
deploy.bat stop          暂停定时 + 终止进程
deploy.bat start         恢复定时
deploy.bat status        查看任务状态
```

---

## 处理管线

```
download  →  process  →  store  →  graph
 飞书下载     拆分+向量化   pgvector入库   LightRAG图谱
```

- **默认增量**：只处理新增/修改的文档
- **`--full`**：忽略增量记录，重新处理全部
- **`--reset-db`**：清空数据库后全量重建

---

## 图谱读写协调

构建图谱时会在 `lightrag_workspace/` 目录产生信号文件，供外部 RAG 系统协调读取：

| 文件 | 含义 |
|------|------|
| `.writing` 存在 | **正在写入，请勿读取**（数据不完整） |
| `.ready` 存在 | 写入完成，可安全读取（内含时间戳） |
| 两者都不存在 | 无数据或尚未构建 |

RAG 读端建议逻辑：检查 `.writing` → 存在则等待/跳过 → 不存在则读取 `.ready` 判断是否需要刷新。

---

## 关键调优参数

| 配置文件 | 参数 | 默认 | 说明 |
|----------|------|------|------|
| `feishu.yaml` | `schedule.task_name` | FeishuDocSync | 定时任务名（多实例须不同） |
| `feishu.yaml` | `schedule.run_time` | 02:00 | 每日执行时间 |
| `doc_splitter.yaml` | `embedding.batch_size` | 16 | Embedding 批次（OOM 调小） |
| `lightrag.yaml` | `llm.max_async` | 12 | LLM 并发数 |
| `lightrag.yaml` | `graph.max_parallel_insert` | 4 | 文档并行插入（低配调小） |
| `lightrag.yaml` | `lightrag.enabled` | true | 图谱总开关 |

---

## 多实例部署

复制整个部署目录到不同位置，每份改：

1. `feishu.yaml` → `schedule.task_name` 改为不同名称
2. `feishu.yaml` → `app_id`/`app_secret`/`space_ids` 配各自应用
3. `db_info.yml` → 建议使用不同数据库
4. 各自双击 `install.bat`

---

## 常见问题

| 问题 | 解决 |
|------|------|
| `venv not found` | 进 `auto-doc-process/` 运行 `setup.bat` |
| `Preflight failed` | 按提示检查配置和数据库连接 |
| `Run as Administrator` | 右键以管理员运行，或用 `install.bat`（自动提权） |
| 飞书权限不足 | 确认应用已开通 wiki/docx/drive 只读权限 |
| 数据库连接失败 | 检查 PostgreSQL 服务 + `db_info.yml` |
| `vector` 扩展缺失 | PG 中执行 `CREATE EXTENSION vector;` |
| 首次图谱构建很慢 | 正常，需 LLM 逐文档抽取，后续增量很快 |
| 文档处理崩溃 | 日志搜 `子进程崩溃`，问题文档被跳过，不影响其余 |
| 清理重来 | `start.bat reset` 后 `start.bat --full` |
| 机器重启任务丢失？ | 不会，`schtasks` 任务持久化在系统 |
| 错过执行时间 | 任务计划程序 → 属性 → 勾选"过了计划时间立即启动" |
