"""腾讯云 CLS 的只读查询封装。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from tencentcloud.cls.v20201016 import cls_client, models
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

from app.config import config

DEFAULT_SYSTEM_TOPIC_ALIASES = {
    "current",
    "current-system",
    "system",
    "server",
    "host",
    "monitor",
    "alert",
    "当前系统",
    "服务器",
    "系统",
    "监控",
    "告警",
}


class TencentCLSConfigError(ValueError):
    """腾讯云 CLS 本地配置不完整。"""


def matches_default_topic(service_name: str, fuzzy: bool = True) -> bool:
    """判断工具输入是否指向当前配置的默认系统主题。"""
    query = service_name.strip().lower().replace("_", "-")
    if query in DEFAULT_SYSTEM_TOPIC_ALIASES:
        return True

    configured_service = config.tencent_cls_service_name.strip().lower()
    if not configured_service:
        return False
    if fuzzy:
        return query in configured_service or configured_service in query
    return query == configured_service


def _require_settings() -> None:
    missing = []
    if not config.tencentcloud_secret_id:
        missing.append("TENCENTCLOUD_SECRET_ID")
    if not config.tencentcloud_secret_key:
        missing.append("TENCENTCLOUD_SECRET_KEY")
    if not config.tencent_cls_region:
        missing.append("TENCENT_CLS_REGION")
    if missing:
        raise TencentCLSConfigError("腾讯云 CLS 配置不完整，缺少环境变量：" + ", ".join(missing))


def build_cls_client(region_code: str | None = None) -> cls_client.ClsClient:
    """根据环境变量创建腾讯云 CLS 客户端。"""
    _require_settings()
    cred = credential.Credential(
        config.tencentcloud_secret_id,
        config.tencentcloud_secret_key,
    )
    http_profile = HttpProfile()
    http_profile.endpoint = config.tencent_cls_endpoint
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    return cls_client.ClsClient(
        cred,
        region_code or config.tencent_cls_region,
        client_profile,
    )


def _topic_to_dict(topic: models.TopicInfo, region_code: str) -> dict[str, Any]:
    return {
        "topic_id": topic.TopicId,
        "topic_name": topic.TopicName,
        "service_name": config.tencent_cls_service_name or topic.TopicName,
        "region_code": region_code,
        "create_time": topic.CreateTime,
        "index_enabled": topic.Index,
        "collection_enabled": topic.Status,
        "description": topic.Describes or "",
        "logset_id": topic.LogsetId,
    }


def describe_topics(
    *,
    topic_name: str | None = None,
    topic_id: str | None = None,
    region_code: str | None = None,
    fuzzy: bool = True,
) -> dict[str, Any]:
    """从腾讯云查询日志主题。"""
    region = region_code or config.tencent_cls_region
    request = models.DescribeTopicsRequest()
    params: dict[str, Any] = {"Limit": 100, "BizType": 0}
    if topic_id:
        params["Filters"] = [{"Key": "topicId", "Values": [topic_id]}]
    elif topic_name:
        params["Filters"] = [{"Key": "topicName", "Values": [topic_name]}]
        params["PreciseSearch"] = 0 if fuzzy else 1
    request.from_json_string(json.dumps(params))

    response = build_cls_client(region).DescribeTopics(request)
    topics = [_topic_to_dict(topic, region) for topic in response.Topics or []]
    return {
        "total": response.TotalCount or len(topics),
        "topics": topics,
        "request_id": response.RequestId,
        "data_source": "tencent_cls",
    }


def _decode_log(log_info: models.LogInfo) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if log_info.LogJson:
        try:
            parsed = json.loads(log_info.LogJson)
            if isinstance(parsed, dict):
                fields = parsed
        except (TypeError, json.JSONDecodeError):
            fields = {"raw": log_info.LogJson}

    message = fields.get("__CONTENT__") or fields.get("content")
    if not message:
        message = log_info.RawLog or log_info.LogJson or ""

    timestamp = None
    if log_info.Time is not None:
        timestamp = datetime.fromtimestamp(log_info.Time / 1000).astimezone().isoformat()

    result = {
        "timestamp": timestamp,
        "message": message,
        "source": log_info.Source,
        "host_name": log_info.HostName,
        "file_name": log_info.FileName,
        "topic_id": log_info.TopicId,
        "topic_name": log_info.TopicName,
        "fields": fields,
    }
    if "level" in fields:
        result["level"] = fields["level"]
    return result


def search_logs(
    *,
    topic_id: str,
    start_time: int,
    end_time: int,
    query: str | None,
    limit: int,
) -> dict[str, Any]:
    """调用 SearchLog 查询一个主题的原始日志。"""
    if start_time >= end_time:
        raise ValueError("start_time 必须小于 end_time")
    if not 1 <= limit <= 1000:
        raise ValueError("limit 必须在 1 到 1000 之间")

    resolved_topic_id = topic_id or config.tencent_cls_topic_id
    if not resolved_topic_id:
        raise TencentCLSConfigError("没有提供 topic_id，且未配置 TENCENT_CLS_TOPIC_ID")

    request = models.SearchLogRequest()
    request.from_json_string(
        json.dumps(
            {
                "From": start_time,
                "To": end_time,
                "QueryString": query or "*",
                "QuerySyntax": 1,
                "TopicId": resolved_topic_id,
                "Sort": "desc",
                "Limit": limit,
                "UseNewAnalysis": True,
            }
        )
    )

    response = build_cls_client().SearchLog(request)
    logs = [_decode_log(item) for item in response.Results or []]
    return {
        "topic_id": resolved_topic_id,
        "start_time": start_time,
        "end_time": end_time,
        "query": query or "*",
        "limit": limit,
        "total": len(logs),
        "logs": logs,
        "list_over": response.ListOver,
        "context": response.Context,
        "request_id": response.RequestId,
        "data_source": "tencent_cls",
        "message": f"从腾讯云 CLS 查询到 {len(logs)} 条真实日志",
    }


def safe_cls_error(exc: Exception) -> dict[str, Any]:
    """把 SDK 异常转换成 MCP 可返回的结构，不泄露密钥。"""
    if isinstance(exc, TencentCloudSDKException):
        return {
            "error": True,
            "error_type": "tencent_cloud_sdk_error",
            "message": exc.message,
            "request_id": exc.requestId,
            "data_source": "tencent_cls",
        }
    return {
        "error": True,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "data_source": "tencent_cls",
    }
