"""请求数据模型。"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


ALLOWED_IMAGE_PREFIXES = (
    "data:image/png;base64,",
    "data:image/jpg;base64,",
    "data:image/jpeg;base64,",
    "data:image/webp;base64,",
)
MAX_IMAGE_COUNT = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024


class ChatRequest(BaseModel):
    """统一对话请求。

    同时兼容：
    - 比赛标准字段：question / images / session_id / stream
    - 旧字段：Id / Question
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        str_strip_whitespace=True,
    )

    question: str = Field(
        ...,
        min_length=1,
        description="用户问题",
        validation_alias=AliasChoices("question", "Question"),
    )
    images: list[str] = Field(
        default_factory=list,
        description="Base64 图片列表，最多 3 张",
    )
    session_id: str | None = Field(
        default=None,
        description="会话 ID",
        validation_alias=AliasChoices("session_id", "Id"),
    )
    stream: bool = Field(
        default=False,
        description="是否流式返回",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question 不能为空")
        return value.strip()

    @field_validator("images")
    @classmethod
    def validate_images(cls, value: list[str]) -> list[str]:
        if len(value) > MAX_IMAGE_COUNT:
            raise ValueError(f"images 最多支持 {MAX_IMAGE_COUNT} 张")

        normalized_images: list[str] = []
        for image in value:
            if not isinstance(image, str):
                raise ValueError("images 中每一项都必须是字符串")
            normalized = image.strip()
            if not normalized:
                raise ValueError("images 中不允许空字符串")
            if not normalized.startswith(ALLOWED_IMAGE_PREFIXES):
                raise ValueError(
                    "图片必须使用 data:image/{png|jpg|jpeg|webp};base64,... 格式"
                )

            _, encoded = normalized.split(",", 1)
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("images 中包含非法 Base64 内容") from exc

            if len(decoded) > MAX_IMAGE_BYTES:
                raise ValueError("images 中每张图片不能超过 5MB")

            normalized_images.append(normalized)

        return normalized_images

    @classmethod
    def from_legacy_payload(cls, payload: dict[str, Any]) -> "ChatRequest":
        """供测试或外部脚本在需要时显式兼容旧字段。"""
        return cls.model_validate(payload)


class ClearRequest(BaseModel):
    """清空会话请求。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: str = Field(..., description="会话 ID", alias="sessionId")
