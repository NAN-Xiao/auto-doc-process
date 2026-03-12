# 飞书文档自动同步工具 — 部署与使用说明

自动从飞书知识空间下载文档，处理后存入数据库，供 RAG 系统使用。

---

## 一、部署前准备

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11 / Server |
| Python | 3.11（必须与开发环境一致） |
| PostgreSQL | 14 以上，需安装 pgvector 扩展 |
| 网络 | 能访问 `open.feishu.cn` 和 `api.deepseek.com` |

> 如果拿到的是含 `venv` 的全量包，则不需要单独安装 Python。

### 数据库准备

在 PostgreSQL 中执行一次：

```sql
\c slg_config
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 二、安装

1. 将 `auto-doc-process` 整个文件夹复制到目标目录，例如：

```
D:\feishu-docs\
└── auto-doc-process\      ← 放这里
```

2. **如果没有 venv 目录**，双击 `setup.bat`（自动创建虚拟环境并安装依赖）。

3. 进入 `configs\` 目录，将三个 `.example` 模板复制为正式配置文件：

```
feishu.yaml.example       →  feishu.yaml
db_info.yml.example       →  db_info.yml
doc_splitter.yaml.example →  doc_splitter.yaml
```

4. 按照下方「配置说明」填写实际的密钥和连接信息。

---

## 三、配置说明

需要修改 3 个文件，都在 `configs\` 目录下。

### feishu.yaml — 飞书凭证

打开后只需改两行：

```yaml
feishu:
  app_id: "cli_xxxxxxxx"           # ← 换成你的 App ID
  app_secret: "xxxxxxxxxxxxxxxx"    # ← 换成你的 App Secret
```

**App ID / App Secret 获取方式：**
1. 登录 [飞书开放平台](https://open.feishu.cn)
2. 找到或创建企业自建应用
3. 确认已开通权限：知识空间只读、文档导出、云文档读取
4. 在应用详情页复制 App ID 和 App Secret

其他参数一般不用改。如需指定特定的知识空间，填写 `space_ids`：

```yaml
  space_ids:
    - "7613735903789370589"
```

### db_info.yml — 数据库连接

```yaml
database:
  enabled: true
  host: localhost          # 数据库地址
  port: 5432
  database: slg_config     # 数据库名
  user: postgres
  password: "你的密码"      # ← 换成实际密码
```

### doc_splitter.yaml — DeepSeek 密钥

打开后只需改一行：

```yaml
deepseek:
  api_key: "sk-xxxxxxxxxxxxxxxx"   # ← 换成你的 DeepSeek API Key
```

其他参数（分块大小、图片命名方式等）保持默认即可。

> ⚠️ 以上三个配置文件包含密钥，请勿提交到 Git 或分享给无关人员。

---

## 四、使用方法

### 手动执行一次

双击 `start.bat` 即可。或者在命令行中：

```cmd
start.bat              :: 正常执行（增量下载 + 处理 + 入库）
start.bat --dry-run    :: 预览模式（只列出文档，不下载不处理）
start.bat --full       :: 全量模式（忽略增量记录，重新下载所有文档）
start.bat --no-graph   :: 跳过知识图谱构建
```

### 注册为定时任务（推荐）

以**管理员身份**打开命令提示符，进入 `auto-doc-process` 目录执行：

```cmd
deploy.bat install     :: 注册（默认每天凌晨 2:00 执行）
```

注册成功后会自动在后台定时运行，无需保持窗口打开。

---

## 五、定时任务管理

所有命令都需要**管理员权限**，在 `auto-doc-process` 目录下运行：

| 命令 | 说明 |
|---|---|
| `deploy.bat install` | 注册定时任务（每天凌晨 2:00） |
| `deploy.bat stop` | 停止任务（暂停定时 + 终止运行中的进程） |
| `deploy.bat start` | 恢复定时 |
| `deploy.bat run` | 立即手动触发一次 |
| `deploy.bat status` | 查看当前任务状态 |
| `deploy.bat uninstall` | 彻底删除定时任务 |

---

## 六、版本更新

```
1. deploy.bat stop               ← 停止任务
2. 用新版文件覆盖旧文件           ← ⚠️ 不要覆盖 configs/ 下的三个配置文件
3. deploy.bat start              ← 恢复任务
4. deploy.bat run                ← 手动跑一次验证
```

如果新版更新了依赖（requirements.txt 有变化），在第 2 步后多执行一步：

```cmd
venv\Scripts\pip install -r requirements.txt
```

---

## 七、查看日志

运行日志保存在项目内的 `_runtime\logs\` 目录中：

```
auto-doc-process\
└── _runtime\
    ├── logs\
    │   └── feishu_export_20260311.log    ← 每天一个文件
    ├── manifest.json                     ← 增量下载清单
    └── .lock                             ← 进程锁
```

所有运行时产生的文件（日志、增量清单、进程锁）都集中在 `_runtime\` 目录，不会与文档数据混淆。

双击 `start.bat` 执行时，控制台窗口也会实时显示日志。

---

## 八、常见问题

**Q：首次运行很慢？**
首次需要下载 Embedding 模型（约 100MB），之后会缓存在 `models/` 目录中。如果目标机无法联网，请联系开发人员提供含模型的完整包。

**Q：飞书报权限不足？**
确认应用已开通并通过审核：知识空间只读（`wiki:wiki:readonly`）、文档导出（`docx:document:readonly`）、云文档读取（`drive:drive:readonly`）。

**Q：数据库连接失败？**
1. 确认 PostgreSQL 服务已启动
2. 确认 `db_info.yml` 中的地址、端口、密码正确
3. 确认已执行过 `CREATE EXTENSION IF NOT EXISTS vector;`

**Q：每次都重新下载全部文档？**
增量状态记录在 `_runtime\manifest.json` 文件中。如果这个文件被删除，下次会变成全量下载。这是正常行为，不影响数据正确性。

**Q：定时任务没有执行？**
运行 `deploy.bat status` 查看状态。如果显示"已禁用"，执行 `deploy.bat start` 恢复。
