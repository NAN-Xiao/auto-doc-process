# 飞书文档自动同步工具

**项目定位**：本工程仅做**预处理管线**（下载 → 拆分 → 向量化 → 入库 → 图谱），**不做召回/检索**。下游 RAG 或检索服务另行对接 pgvector / 图谱文件。重点可放在**预处理质量**的评估与调优（见 [预处理质量说明](PREPROCESSING_QUALITY.md)）。

---

## 快速部署（3 步）

### 1. 环境准备

进入 `dist/auto-doc-process/`，双击 **`setup.bat`**（自动创建 venv + 安装依赖）。

> `--include-venv` 打包的版本跳过此步。

### 2. 填写配置

进入 `dist/auto-doc-process/configs/`，把 4 个 `.example` 复制一份去掉后缀，按下方说明填写。

---

#### feishu.yaml — 飞书连接

> 获取方式：[飞书开放平台](https://open.feishu.cn) → 创建企业自建应用 → 凭证与基础信息
>
> 需开通权限：`wiki:wiki:readonly`、`docx:document:readonly`、`drive:drive:readonly`

```yaml
feishu:
  app_id: ""                    # ★ 必填 - 飞书 App ID
  app_secret: ""                # ★ 必填 - 飞书 App Secret
  base_url: "https://open.feishu.cn"

  output_dir: "../../documents" # 文档下载目录（相对于 auto-doc-process/）

  space_ids: []                 # 同步哪些空间，空=所有有权限的空间
  # space_ids:
  #   - "7613735903789370589"   # 指定空间 ID

  log:
    level: "INFO"               # 日志级别：DEBUG / INFO / WARNING / ERROR

  export:
    poll_max_wait: 300          # 导出任务最大等待（秒）
    poll_interval: 3            # 轮询间隔（秒）
    type_format_map:            # 文档类型 → 导出格式
      doc: "docx"
      docx: "docx"
      # sheet: "xlsx"           # 取消注释可导出表格
      # bitable: "xlsx"
    skip_types: ["mindnote", "file", "slides", "catalog"]

  schedule:
    task_name: "FeishuDocSync"  # 定时任务名（多实例部署须不同）
    run_time: "02:00"           # 每天执行时间
    run_on_start: true          # 安装后立即执行一次
```

---

#### db_info.yml — 数据库连接

> 要求：PostgreSQL 14+，已安装 pgvector 扩展

```yaml
database:
  enabled: true                 # false = 禁用数据库（仅本地文件处理）
  host: localhost               # 数据库地址
  port: 5432                    # 端口
  database: ""                  # ★ 必填 - 数据库名
  user: postgres                # 用户名
  password: ""                  # ★ 必填 - 密码
```

---

#### doc_splitter.yaml — 文档处理 + LLM

```yaml
# ── LLM 配置（图片命名、图谱构建共用） ──
llm:
  api_key: ""                             # ★ 必填 - API Key
  api_base: "https://api.deepseek.com"    # ★ 必填 - API 地址（默认 DeepSeek，可换 OpenAI 兼容接口）
  model: "deepseek-chat"                  # 模型名

# ── 文本分割 ──
doc_splitter:
  text_splitter:
    chunk_size: 1000            # 文本块大小（字符），影响 RAG 检索粒度
    chunk_overlap: 200          # 块间重叠（字符），保证上下文连贯

  image_naming:
    use_llm: true               # true=LLM 智能命名, false=关键词简单命名

  processing:
    skip_existing: false        # true=跳过已处理的文档
    continue_on_error: true     # true=单文档出错不中断整体

# ── 路径配置（相对于 auto-doc-process/，../../ = 根目录） ──
paths:
  documents_dir: "../../documents"        # 源文档目录（与 feishu.yaml output_dir 一致）
  processed_dir: "../../processed"        # 处理产物目录
  excel_dir: "../../documents/excel"      # Excel 目录

# ── Embedding 模型 ──
embedding:
  model: "BAAI/bge-small-zh-v1.5"        # 模型名（本地优先，无则自动下载）
  batch_size: 16                          # 批次大小（OOM 调小）
  huggingface:
    cache_folder: "./models"              # 模型缓存目录
    device: "cpu"                         # cpu / cuda
```

---

#### lightrag.yaml — 知识图谱

> 一般无需修改，默认值即可。不需要图谱可设 `enabled: false`。

```yaml
lightrag:
  enabled: true                           # 图谱总开关（false=跳过阶段4）
  working_dir: "../../processed/lightrag_workspace"  # 图谱工作目录

  llm:
    model: "deepseek-chat"                # 图谱抽取用的模型
    max_async: 12                         # LLM 并发数（仅网络）
    timeout: 180                          # 单次请求超时（秒）

  graph:
    chunk_token_size: 1500                # 图谱分块大小（token）
    max_parallel_insert: 4                # 文档并行插入数（低配调小）
    enable_llm_cache: true                # LLM 结果缓存

  performance:
    entity_embed_batch: 32                # 实体 embedding 批次（OOM 调小）
    max_crash_restarts: 5                 # 子进程崩溃最大重启次数

  pg_export:
    enabled: false                        # PG 导出（默认关闭，本地文件已足够）
```

### 3. 安装定时任务

进入 **`dist/`** 目录，双击 **`install.bat`**（自动弹 UAC 提权）。

首次会自动：预检环境 → 全量同步 → 注册每天定时任务。

卸载：双击 **`uninstall.bat`**。

---

## 命令速查

> 在 `dist/` 目录下执行。PowerShell 中需加 `.\` 前缀，如 `.\start.bat status`

### 手动执行

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

### 运维

```
start.bat stop                       停止进程
start.bat status                     查看状态
start.bat reset                      清理所有产物
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

## 路径配置

所有路径相对于 `dist/auto-doc-process/`，`../../` 即根目录：

| 配置文件 | 参数 | 默认值 | 指向 |
|----------|------|--------|------|
| `feishu.yaml` | `output_dir` | `../../documents` | 文档下载目录 |
| `doc_splitter.yaml` | `paths.documents_dir` | `../../documents` | 源文档目录（同上） |
| `doc_splitter.yaml` | `paths.processed_dir` | `../../processed` | 处理产物目录 |
| `lightrag.yaml` | `working_dir` | `../../processed/lightrag_workspace` | 图谱工作目录 |

> `output_dir` 和 `documents_dir` 须指向同一目录。

---

## 图谱读写协调

构建图谱时在 `lightrag_workspace/` 产生信号文件，供外部 RAG 系统协调：

| 文件 | 含义 |
|------|------|
| `.writing` 存在 | **正在写入，请勿读取**（数据不完整） |
| `.ready` 存在 | 写入完成，可安全读取（内含时间戳） |
| 两者都不存在 | 无数据或尚未构建 |

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

复制整个根目录到不同位置，每份改：

1. `feishu.yaml` → `schedule.task_name` 改为不同名称
2. `feishu.yaml` → `app_id`/`app_secret`/`space_ids` 配各自应用
3. `db_info.yml` → 建议使用不同数据库
4. 各自双击 `install.bat`

---

## 常见问题

| 问题 | 解决 |
|------|------|
| `venv not found` | 进 `dist/auto-doc-process/` 运行 `setup.bat` |
| `Preflight failed` | 按提示检查配置和数据库连接 |
| `Run as Administrator` | `install.bat` / `uninstall.bat` 会自动弹 UAC 提权 |
| 飞书权限不足 | 确认应用已开通 wiki/docx/drive 只读权限 |
| 数据库连接失败 | 检查 PostgreSQL 服务 + `db_info.yml` |
| `vector` 扩展缺失 | PG 中执行 `CREATE EXTENSION vector;` |
| 首次图谱构建很慢 | 正常，需 LLM 逐文档抽取，后续增量很快 |
| 文档处理崩溃 | 日志搜 `子进程崩溃`，问题文档被跳过，不影响其余 |
| 清理重来 | `start.bat reset` 后 `start.bat --full` |
| 机器重启任务丢失？ | 不会，`schtasks` 任务持久化在系统 |
| 错过执行时间 | 任务计划程序 → 属性 → 勾选"过了计划时间立即启动" |
