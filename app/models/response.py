"""响应数据模型。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

SUMMARY_METADATA_KEYS: tuple[str, ...] = (
    "intent",
    "doc_id",
    "retrieval_stage",
    "intent_strategy",
    "recall_channels",
    "reranker_provider",
    "reranker_fallback",
    "timeout",
    "degraded",
    "top_hits",
    "warnings",
)

TRACE_METADATA_KEYS: set[str] = {
    "diagnostics",
    "evidence",
    "query",
    "query_understanding",
    "raw_candidates",
    "raw_query",
    "recall",
    "recall_candidates",
    "request_id",
    "reranker",
    "session_id",
    "trace",
}


def sanitize_summary_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return only Summary-level metadata fields that are safe for API responses."""
    if not isinstance(metadata, Mapping):
        return None

    sanitized = {
        key: _json_safe_summary_value(metadata[key])
        for key in SUMMARY_METADATA_KEYS
        if key in metadata
    }
    return sanitized or None


def _json_safe_summary_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_summary_value(item)
            for key, item in value.items()
            if str(key) not in TRACE_METADATA_KEYS
        }
    if isinstance(value, list | tuple):
        return [_json_safe_summary_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_json_safe_summary_value(item) for item in sorted(value, key=str)]
    return str(value)


class ChatResponse(BaseModel):
    """旧版对话响应。"""

    answer: str = Field(..., description="AI 回答")
    session_id: str = Field(..., description="会话 ID")


class CompetitionChatData(BaseModel):
    """比赛标准对话数据。"""

    answer: str = Field(..., description="智能体回答")
    session_id: str = Field(..., description="会话 ID")
    timestamp: int = Field(..., description="响应时间戳（秒）")
    metadata: dict[str, Any] | None = Field(None, description="可选检索诊断摘要")


class CompetitionChatResponse(BaseModel):
    """比赛标准响应结构。"""

    code: int = Field(..., description="业务状态码")
    msg: str = Field(..., description="消息")
    data: CompetitionChatData = Field(..., description="响应数据")


class SessionInfoResponse(BaseModel):
    """会话信息响应。"""

    session_id: str = Field(..., description="会话 ID")
    message_count: int = Field(..., description="消息数量")
    history: list[dict[str, str]] = Field(..., description="历史消息列表")


class ApiResponse(BaseModel):
    """通用 API 响应。"""

    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    data: Any | None = Field(None, description="数据")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(..., description="状态")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")
