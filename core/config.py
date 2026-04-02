#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一配置加载 & 日志初始化

提供:
  - 路径常量 (MODULE_DIR, PROJECT_ROOT, CONFIGS_DIR)
  - 日志系统 (setup_logging / log)
  - load_feishu_config()     → 飞书凭证
  - load_db_config()         → 数据库配置
  - load_full_config()       → 飞书下载管线完整配置
  - load_processor_config()  → 文档处理管线配置（doc_splitter.yaml + db_info.yml）
"""

import sys
import logging
from pathlib import Path
from datetime import datetime


class ConfigError(Exception):
    """配置加载失败时抛出的异常"""
    pass


# ==================== 路径常量 ====================

# core/ 的父目录 → auto-doc-process/
# 兼容两种情况：
#   源码: core/config.py          → parent = core/ → parent.parent = auto-doc-process/
#   编译: core/__pycache__/x.pyc  → parent = __pycache__/ → 需要多退一层
_this_dir = Path(__file__).parent
if _this_dir.name == "__pycache__":
    _this_dir = _this_dir.parent          # __pycache__ → core/
MODULE_DIR = _this_dir.parent.resolve()   # core/ → auto-doc-process/
PROJECT_ROOT = MODULE_DIR.parent
CONFIGS_DIR = MODULE_DIR / "configs"

# 运行时数据目录（放在项目父目录，与文档同级，避免混入项目源码）
RUNTIME_DIR = PROJECT_ROOT / "_runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# 默认路径（可被配置覆盖）
DEFAULT_LOG_DIR = RUNTIME_DIR / "logs"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "feishu_exports"
DEFAULT_MANIFEST_PATH = RUNTIME_DIR / "manifest.json"
DEFAULT_LOCK_PATH = RUNTIME_DIR / ".lock"


# ==================== 日志 ====================

def setup_logging(log_level: str = "INFO", log_dir: str = "") -> logging.Logger:
    """
    配置日志系统：同时输出到控制台和日志文件

    Args:
        log_level: 日志级别 (DEBUG / INFO / WARNING / ERROR)
        log_dir: 日志目录（空字符串使用默认值）

    Returns:
        logger 实例
    """
    actual_log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    actual_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = actual_log_dir / f"feishu_export_{datetime.now():%Y%m%d}.log"

    logger = logging.getLogger("feishu_sync")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False  # 不传播到 root logger，避免重复输出

    # 避免重复添加 handler
    if logger.handlers:
        # 更新已有 handler 的级别
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# 默认 logger（会在 load_full_config 后根据配置重新初始化）
log = setup_logging()


# ==================== YAML 加载（带缓存） ====================

_yaml_cache: dict = {}


def _load_yaml(name: str) -> dict:
    """加载 configs/ 下的 YAML 文件（带缓存）"""
    if name not in _yaml_cache:
        path = CONFIGS_DIR / name
        if not path.exists():
            log.warning(f"配置文件不存在: {path}")
            _yaml_cache[name] = {}
        else:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                _yaml_cache[name] = yaml.safe_load(f) or {}
    return _yaml_cache[name]


def clear_config_cache():
    """
    清除所有配置缓存，下次 load_*() 调用时重新从磁盘读取。

    在 scheduler 守护模式下每次执行循环开始时调用，
    确保运行期间修改的配置文件能够热更新生效。
    """
    global _processor_config
    _yaml_cache.clear()
    _processor_config = None


# ==================== 飞书凭证 ====================

def _find_config_file(config_path: str = None) -> Path:
    """查找飞书配置文件路径"""
    if config_path:
        path = Path(config_path)
        if path.exists():
            return path
        raise ConfigError(f"配置文件不存在: {path}")

    candidates = [
        CONFIGS_DIR / "feishu.yaml",
        MODULE_DIR / "feishu.yaml",
        Path("configs/feishu.yaml"),
        Path("feishu.yaml"),
    ]
    path = next((c for c in candidates if c.exists()), None)
    if not path:
        raise ConfigError(f"找不到配置文件，已尝试: {[str(c) for c in candidates]}")
    return path


def load_feishu_config(config_path: str = None) -> dict:
    """
    加载飞书应用凭证配置

    Args:
        config_path: 自定义配置文件路径（默认自动查找）

    Returns:
        feishu 配置节的完整字典
    """
    if config_path:
        # 自定义路径：直接读取，不走缓存
        import yaml
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        # 默认路径：走 _load_yaml 缓存
        path = _find_config_file(None)
        config = _load_yaml(path.name) if path.parent.resolve() == CONFIGS_DIR.resolve() else None
        if config is None:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

    feishu = config.get("feishu", {})
    log.info(f"加载配置: {path}")
    log.info(f"  app_id: {feishu.get('app_id', '未设置')}")
    return feishu


# ==================== 数据库配置 ====================

def load_db_config() -> dict:
    """
    从 db_info.yml 加载数据库配置

    Returns:
        {"host", "port", "database", "user", "password"} 或空字典
    """
    raw = _load_yaml("db_info.yml")
    db = raw.get("database", {})

    if not db.get("enabled", True):
        log.info("数据库已禁用 (database.enabled = false)")
        return {}

    log.info(f"加载数据库配置: {db.get('host', 'localhost')}:{db.get('port', 5432)}/{db.get('database', '')}")
    return {
        "host": db.get("host", "localhost"),
        "port": db.get("port", 5432),
        "database": db.get("database", ""),
        "user": db.get("user", ""),
        "password": db.get("password", ""),
    }


# ==================== 飞书下载完整配置 ====================

def load_full_config(config_path: str = None) -> dict:
    """
    加载完整配置（凭证 + 所有参数），返回扁平化结构供各模块使用

    Returns:
        {
            # 飞书凭证
            "app_id", "app_secret", "base_url",
            # 输出
            "output_dir",
            # 日志
            "log_level", "log_dir",
            # 同步范围
            "space_ids",
            # 导出参数
            "poll_max_wait", "poll_interval", "poll_initial_delay",
            "rate_limit_delay", "type_format_map", "skip_types",
            # 清单 & 锁
            "manifest_path", "lock_path",
            # 向量化（仅开关，模型/分块参数见 doc_splitter.yaml）
            "vec_enabled",
            # 数据库
            "db",
        }
    """
    feishu = load_feishu_config(config_path)

    export = feishu.get("export", {})
    log_cfg = feishu.get("log", {})
    manifest_cfg = feishu.get("manifest", {})
    lock_cfg = feishu.get("lock", {})

    # 输出目录（相对路径基于 MODULE_DIR / auto-doc）
    output_dir_raw = feishu.get("output_dir", "")
    if output_dir_raw:
        output_dir = Path(output_dir_raw)
        if not output_dir.is_absolute():
            output_dir = (MODULE_DIR / output_dir).resolve()
    else:
        output_dir = DEFAULT_EXPORT_DIR

    # 日志目录
    log_dir_raw = log_cfg.get("dir", "")

    # 清单路径
    manifest_raw = manifest_cfg.get("path", "")
    if manifest_raw:
        manifest_path = Path(manifest_raw)
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
    else:
        manifest_path = DEFAULT_MANIFEST_PATH

    # 锁文件路径
    lock_raw = lock_cfg.get("path", "")
    if lock_raw:
        lock_path = Path(lock_raw)
        if not lock_path.is_absolute():
            lock_path = PROJECT_ROOT / lock_path
    else:
        lock_path = DEFAULT_LOCK_PATH

    # 导出类型映射
    default_type_map = {"doc": "docx", "docx": "docx", "sheet": "xlsx", "bitable": "xlsx"}
    type_format_map = export.get("type_format_map", default_type_map)

    # 跳过的类型
    default_skip = ["mindnote", "file", "slides", "catalog"]
    skip_types = set(export.get("skip_types", default_skip))

    # 知识空间 ID 列表
    space_ids = feishu.get("space_ids", [])

    # 向量化开关（模型 / 分块参数统一在 doc_splitter.yaml）
    vec_cfg = feishu.get("vectorize", {})
    vec_enabled = vec_cfg.get("enabled", True)

    # 加载数据库配置
    db_config = load_db_config()

    return {
        # 飞书凭证
        "app_id": feishu.get("app_id", ""),
        "app_secret": feishu.get("app_secret", ""),
        "base_url": feishu.get("base_url", "https://open.feishu.cn"),
        # 输出
        "output_dir": output_dir,
        # 日志
        "log_level": log_cfg.get("level", "INFO"),
        "log_dir": log_dir_raw,
        # 同步范围
        "space_ids": space_ids,
        # 导出参数
        "poll_max_wait": export.get("poll_max_wait", 60),
        "poll_interval": export.get("poll_interval", 2),
        "poll_initial_delay": export.get("poll_initial_delay", 3),
        "rate_limit_delay": export.get("rate_limit_delay", 1),
        "type_format_map": type_format_map,
        "skip_types": skip_types,
        # 清单 & 锁
        "manifest_path": manifest_path,
        "lock_path": lock_path,
        # 向量化（仅开关，模型/分块参数见 doc_splitter.yaml）
        "vec_enabled": vec_enabled,
        # 数据库
        "db": db_config,
    }


# ==================== 文档处理管线配置 ====================

_processor_config: dict | None = None


def load_processor_config() -> dict:
    """
    加载文档处理管线配置（doc_splitter.yaml + db_info.yml）

    供 processor/ 模块使用，返回合并后的配置字典。
    """
    global _processor_config
    if _processor_config is not None:
        return _processor_config

    config = dict(_load_yaml("doc_splitter.yaml"))
    config["database"] = load_db_config()

    # --- 路径解析：相对路径 → 绝对路径（基于 MODULE_DIR / auto-doc-process） ---
    paths = config.get("paths", {})
    for key in ("documents_dir", "processed_dir", "excel_dir"):
        raw = paths.get(key, "")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                paths[key] = str((MODULE_DIR / p).resolve())
    config["paths"] = paths

    # embedding 模型缓存目录
    emb = config.get("embedding", {})
    hf = emb.get("huggingface", {})
    cache = hf.get("cache_folder", "")
    if cache:
        p = Path(cache)
        if not p.is_absolute():
            hf["cache_folder"] = str((MODULE_DIR / p).resolve())
        emb["huggingface"] = hf
        config["embedding"] = emb

    _processor_config = config
    return config

