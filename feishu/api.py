#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
飞书 Open API 封装

职责：
  - 获取 tenant_access_token (TAT)
  - 创建 lark_oapi SDK Client
  - Wiki 文档 token 解析
  - 知识空间 & 节点发现
  - 文档导出三步骤（创建任务 → 轮询 → 下载）

所有函数仅依赖 requests / lark_oapi，不依赖项目其他模块。
"""

import os
import json
import time
import requests
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)
import logging

from ..core.config import log

# tenacity 日志：重试前打印到 feishu_sync logger
_tenacity_logger = logging.getLogger("feishu_sync")


# ==================== Token ====================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
    reraise=True,
)
def get_tenant_access_token(config: dict) -> str:
    """
    获取 tenant_access_token（应用身份）

    Args:
        config: {"app_id": ..., "app_secret": ..., "base_url": ...}
    """
    base_url = config.get("base_url", "https://open.feishu.cn")
    resp = requests.post(
        f"{base_url}/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": config["app_id"], "app_secret": config["app_secret"]},
        timeout=10,
    )
    data = resp.json()
    token = data.get("tenant_access_token", "")
    if not token:
        log.error(f"获取 tenant_access_token 失败: {data}")
    return token


# ==================== SDK Client ====================

def create_lark_client(config: dict):
    """
    创建 lark_oapi SDK Client

    Returns:
        lark.Client 实例，失败返回 None
    """
    app_id = config.get("app_id", "")
    app_secret = config.get("app_secret", "")

    if not app_id or not app_secret:
        log.error("app_id 或 app_secret 未配置")
        return None

    try:
        import lark_oapi as lark
        client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .log_level(lark.LogLevel.WARNING) \
            .build()
        log.info(f"SDK Client 创建成功 (app_id: {app_id})")
        return client
    except ImportError:
        log.error("lark_oapi 未安装，请运行: pip install lark-oapi")
        return None
    except Exception as e:
        log.error(f"SDK Client 创建失败: {e}")
        return None


# ==================== Wiki 解析 ====================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
    reraise=True,
)
def resolve_wiki_token(config: dict, wiki_token: str) -> tuple:
    """
    解析 Wiki 文档的真实 token、类型和标题

    Args:
        config: 飞书凭证配置
        wiki_token: Wiki 页面 token（/wiki/xxx 中的 xxx，也接受 node_token 或 obj_token）

    Returns:
        (obj_token, obj_type, title) 或 (None, None, "")
    """
    access_token = get_tenant_access_token(config)
    if not access_token:
        return None, None, ""

    base_url = config.get("base_url", "https://open.feishu.cn")
    resp = requests.get(
        f"{base_url}/open-apis/wiki/v2/spaces/get_node?token={wiki_token}",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    data = resp.json()

    if data.get("code", -1) != 0:
        log.error(f"Wiki 解析失败: {data.get('msg', '')}")
        return None, None, ""

    node = data.get("data", {}).get("node", {})
    obj_token = node.get("obj_token", "")
    obj_type = node.get("obj_type", "")
    title = node.get("title", "")

    log.info(f"Wiki 解析: {wiki_token} → {obj_token} ({obj_type}) [{title}]")
    return obj_token, obj_type, title


# ==================== 知识空间发现 ====================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
    reraise=True,
)
def list_wiki_spaces(access_token: str, base_url: str = "https://open.feishu.cn") -> list:
    """
    列出机器人有权限的所有知识空间

    API: GET /open-apis/wiki/v2/spaces

    Returns:
        [{"space_id": ..., "name": ..., "description": ...}, ...]
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    spaces = []
    page_token = ""

    while True:
        params = {"page_size": 50}
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(
            f"{base_url}/open-apis/wiki/v2/spaces",
            headers=headers, params=params, timeout=15,
        )
        data = resp.json()

        if data.get("code", -1) != 0:
            log.error(f"获取知识空间失败: {data.get('msg', '')}")
            break

        for item in data.get("data", {}).get("items", []):
            spaces.append({
                "space_id": item.get("space_id", ""),
                "name": item.get("name", ""),
                "description": item.get("description", ""),
            })

        if not data.get("data", {}).get("has_more", False):
            break
        page_token = data.get("data", {}).get("page_token", "")

    log.info(f"发现 {len(spaces)} 个知识空间")
    for s in spaces:
        log.info(f"  [{s['space_id']}] {s['name']}")

    return spaces


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
    reraise=True,
)
def list_wiki_nodes(access_token: str, space_id: str,
                    parent_node_token: str = None,
                    depth: int = 0, max_depth: int = 10,
                    base_url: str = "https://open.feishu.cn") -> list:
    """
    递归列出知识空间中的所有节点

    API: GET /open-apis/wiki/v2/spaces/:space_id/nodes

    Returns:
        [{"node_token", "obj_token", "obj_type", "title",
          "space_id", "obj_edit_time", "depth"}, ...]
    """
    if depth > max_depth:
        return []

    headers = {"Authorization": f"Bearer {access_token}"}
    nodes = []
    page_token = ""

    while True:
        params = {"page_size": 50}
        if page_token:
            params["page_token"] = page_token
        if parent_node_token:
            params["parent_node_token"] = parent_node_token

        resp = requests.get(
            f"{base_url}/open-apis/wiki/v2/spaces/{space_id}/nodes",
            headers=headers, params=params, timeout=15,
        )
        data = resp.json()

        if data.get("code", -1) != 0:
            log.error(f"获取节点失败: {data.get('msg', '')}")
            break

        for item in data.get("data", {}).get("items", []):
            nodes.append({
                "node_token": item.get("node_token", ""),
                "obj_token": item.get("obj_token", ""),
                "obj_type": item.get("obj_type", ""),
                "title": item.get("title", ""),
                "space_id": space_id,
                "obj_edit_time": item.get("obj_edit_time", ""),
                "depth": depth,
            })

            if item.get("has_child", False):
                children = list_wiki_nodes(
                    access_token, space_id,
                    parent_node_token=item["node_token"],
                    depth=depth + 1, max_depth=max_depth,
                    base_url=base_url,
                )
                nodes.extend(children)

        if not data.get("data", {}).get("has_more", False):
            break
        page_token = data.get("data", {}).get("page_token", "")

    return nodes


# ==================== 导出三步骤 ====================

def create_export_task(client, doc_token: str, doc_type: str, file_ext: str) -> str:
    """
    步骤1：创建导出任务

    Returns:
        ticket 字符串，失败返回 None
    """
    from lark_oapi.api.drive.v1 import CreateExportTaskRequest, ExportTask

    task = ExportTask.builder() \
        .file_extension(file_ext) \
        .token(doc_token) \
        .type(doc_type) \
        .build()

    request = CreateExportTaskRequest.builder() \
        .request_body(task) \
        .build()

    response = client.drive.v1.export_task.create(request)

    if not response.success():
        log.error(f"创建导出任务失败: code={response.code}, msg={response.msg}")
        return None

    ticket = response.data.ticket if response.data else None
    log.debug(f"导出任务创建: ticket={ticket}")
    return ticket


def download_export_file(client, file_token: str, output_path: str) -> bool:
    """
    步骤3：下载导出文件

    Returns:
        是否成功
    """
    from lark_oapi.api.drive.v1 import DownloadExportTaskRequest

    request = DownloadExportTaskRequest.builder() \
        .file_token(file_token) \
        .build()

    response = client.drive.v1.export_task.download(request)

    if not response.success():
        log.error(f"下载失败: code={response.code}, msg={response.msg}")
        return False

    try:
        file_content = response.file
        if file_content:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(file_content.read() if hasattr(file_content, "read") else file_content)
            file_size = os.path.getsize(output_path)
            log.info(f"  下载完成: {output_path} ({file_size:,} bytes)")
            return True
        else:
            log.error("响应中没有文件内容")
            return False
    except Exception as e:
        log.error(f"写入文件失败: {e}")
        return False

