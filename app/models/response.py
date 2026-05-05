"""响应数据模型。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """旧版对话响应。"""

    answer: str = Field(..., description="AI 回答")
    session_id: str = Field(..., description="会话 ID")


class CompetitionChatData(BaseModel):
    """比赛标准对话数据。"""

    answer: str = Field(..., description="智能体回答")
    session_id: str = Field(..., description="会话 ID")
    timestamp: int = Field(..., description="响应时间戳（秒）")


class CompetitionChatResponse(BaseModel):
    """比赛标准响应结构。"""

    code: int = Field(..., description="业务状态码")
    msg: str = Field(..., description="消息")
    data: CompetitionChatData = Field(..., description="响应数据")


class SessionInfoResponse(BaseModel):
    """会话信息响应。"""

    session_id: str = Field(..., description="会话 ID")
    message_count: int = Field(..., description="消息数量")
    history: List[Dict[str, str]] = Field(..., description="历史消息列表")


class ApiResponse(BaseModel):
    """通用 API 响应。"""

    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    data: Optional[Any] = Field(None, description="数据")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(..., description="状态")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")
