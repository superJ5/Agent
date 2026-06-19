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
IMAGE_FORMAT_PREFIXES = {
    "png": "data:image/png;base64,",
    "jpeg": "data:image/jpeg;base64,",
    "webp": "data:image/webp;base64,",
}


def _detect_image_prefix(image_bytes: bytes) -> str | None:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return IMAGE_FORMAT_PREFIXES["png"]
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return IMAGE_FORMAT_PREFIXES["jpeg"]
    if (
        len(image_bytes) >= 12
        and image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        return IMAGE_FORMAT_PREFIXES["webp"]
    return None


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
    model: str | None = Field(
        default=None,
        max_length=100,
        description="本次请求使用的模型名；不传则使用 .env 中的默认模型",
    )
    dashscope_api_key: str | None = Field(
        default=None,
        max_length=300,
        description="本次请求使用的 DashScope API Key；不传则使用服务端 .env 配置",
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
            prefix = next(
                (
                    allowed_prefix
                    for allowed_prefix in ALLOWED_IMAGE_PREFIXES
                    if normalized.startswith(allowed_prefix)
                ),
                None,
            )
            if prefix:
                encoded = normalized.split(",", 1)[1]
            else:
                encoded = normalized

            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("images 中包含非法 Base64 内容") from exc

            if len(decoded) > MAX_IMAGE_BYTES:
                raise ValueError("images 中每张图片不能超过 5MB")

            if not prefix:
                prefix = _detect_image_prefix(decoded)
                if not prefix:
                    raise ValueError("无法识别裸 Base64 图片类型，仅支持 png、jpg/jpeg、webp")
                normalized = f"{prefix}{encoded}"

            normalized_images.append(normalized)

        return normalized_images

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if any(char.isspace() for char in normalized):
            raise ValueError("model 不能包含空白字符")
        return normalized

    @field_validator("dashscope_api_key")
    @classmethod
    def validate_dashscope_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if any(char.isspace() for char in normalized):
            raise ValueError("dashscope_api_key 不能包含空白字符")
        return normalized

    @classmethod
    def from_legacy_payload(cls, payload: dict[str, Any]) -> "ChatRequest":
        """供测试或外部脚本在需要时显式兼容旧字段。"""
        return cls.model_validate(payload)


class ClearRequest(BaseModel):
    """清空会话请求。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: str = Field(..., description="会话 ID", alias="sessionId")
