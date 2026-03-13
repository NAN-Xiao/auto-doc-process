# 飞书文档自动同步工具

飞书知识空间 → 下载 → 拆分 → 向量化 → pgvector + LightRAG 知识图谱

---

## 配置

`configs\` 下三个文件，从 `.example` 复制后修改：

```yaml
# feishu.yaml — 飞书凭证
feishu:
  app_id: "cli_xxxxxxxx"
  app_secret: "xxxxxxxxxxxxxxxx"
  # space_ids: ["7613735903789370589"]   # 可选，留空=遍历全部

# db_info.yml — 数据库连接
database:
  host: localhost
  port: 5432
  database: slg_config
  user: postgres
  password: "你的密码"

# doc_splitter.yaml — DeepSeek（图片智能命名）
deepseek:
  api_key: "sk-xxxxxxxxxxxxxxxx"
```

> 飞书权限：`wiki:wiki:readonly`、`docx:document:readonly`、`drive:drive:readonly`。其余参数保持默认。

---

## 使用

### 管线阶段

```
download → process → store → graph
  下载       处理     入库    图谱
```

### 命令

```cmd
start.bat                                :: 全流程
start.bat --step download                :: 只下载
start.bat --step process                 :: 只处理
start.bat --step store                   :: 只入库
start.bat --step graph                   :: 只建图谱
start.bat --step download,process,store  :: 组合执行
start.bat --dry-run                      :: 预览（不下载）
start.bat --full                         :: 全量同步（忽略增量记录）
start.bat --reset-db                     :: 清空数据库 → 全量重建
start.bat --no-graph                     :: 跳过图谱
start.bat --log-level DEBUG              :: 调试日志
```

### 定时任务（管理员权限）

```cmd
deploy.bat install      :: 注册（每天 02:00）
deploy.bat stop         :: 停止
deploy.bat start        :: 恢复
deploy.bat run          :: 立即执行一次
deploy.bat status       :: 查看状态
deploy.bat uninstall    :: 删除任务
```

---

## 目录结构

```
doctment/
├── auto-doc-process/           ← 项目代码
│   ├── configs/                ← 配置（3 个 yaml）
│   ├── processor/              ← 拆分/向量/存储/图谱
│   ├── feishu/                 ← 飞书 API
│   ├── models/onnx/            ← ONNX 模型（轻量部署）
│   ├── run.py                  ← 启动入口
│   ├── start.bat / deploy.bat  ← 执行/定时任务
│   └── build.py                ← 打包脚本
├── _runtime/                   ← 日志/增量清单/进程锁
├── processed/                  ← 处理结果（chunks/embeddings/metadata）
└── *.docx / *.pdf              ← 下载的原始文档
```

---

## 稳定性机制

- **子进程隔离** — 每个文档独立子进程处理，C 层崩溃不影响主进程
- **崩溃自动重启** — 识别问题文档并跳过，继续处理剩余文档（最多 5 次）
- **Embedding 分批** — batch_size=16 防止大文档 OOM
- **事务原子性** — pgvector 入库 + 图谱导出使用事务 + advisory lock
- **环境预检** — 启动时检查磁盘/数据库/模型，失败则给出修复建议

---

## FAQ

| 问题 | 解决 |
|---|---|
| 首次运行很慢 | 完整模式首次下载模型 ~100MB；轻量模式无需下载 |
| 飞书权限不足 | 确认应用已开通上述三个权限 |
| 数据库连接失败 | 检查 PostgreSQL 服务 + `db_info.yml` 配置 + `CREATE EXTENSION vector` |
| 全量重新下载 | `_runtime\manifest.json` 被删会触发全量，属正常行为 |
| 文档处理崩溃 | 子进程隔离已处理，日志中搜 `子进程崩溃` 查看问题文档 |
| 清空数据库重建 | `start.bat --reset-db` |
