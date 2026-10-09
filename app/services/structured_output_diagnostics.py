"""Opt-in capture of failed structured model responses during evaluations."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel

from app.core.request_context import current_structured_trace_path

ModelT = TypeVar("ModelT", bound=BaseModel)


def _append_failure(path: Path, event: dict[str, Any]) -> None:
    """Append one restricted-permission JSONL record without storing the prompt."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as output:
            fcntl.flock(output, fcntl.LOCK_EX)
            output.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            fcntl.flock(output, fcntl.LOCK_UN)
    except Exception as exc:
        logger.warning("写入结构化响应诊断失败: {}", exc)


def _raw_payload(raw: Any) -> Any:
    if isinstance(raw, BaseModel):
        return raw.model_dump(mode="json")
    return raw


async def invoke_structured_output(
    llm: Any,
    schema: type[ModelT],
    messages: list[Any],
    *,
    session_id: str,
) -> ModelT:
    """Keep the normal path unchanged; capture raw AIMessage on eval parse failures."""
    trace_path = current_structured_trace_path()
    if trace_path is None:
        result = await llm.with_structured_output(schema).ainvoke(messages)
        return schema.model_validate(result)

    base_event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "schema": schema.__name__,
    }
    try:
        envelope = await llm.with_structured_output(schema, include_raw=True).ainvoke(messages)
    except Exception as exc:
        _append_failure(trace_path, {
            **base_event,
            "stage": "model_invocation",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "raw_response": None,
        })
        raise

    if not isinstance(envelope, dict):
        _append_failure(trace_path, {
            **base_event,
            "stage": "unexpected_envelope",
            "raw_response": _raw_payload(envelope),
        })
        return schema.model_validate(envelope)

    parsed = envelope.get("parsed")
    parsing_error = envelope.get("parsing_error")
    if parsed is None or parsing_error is not None:
        _append_failure(trace_path, {
            **base_event,
            "stage": "structured_parsing",
            "raw_response": _raw_payload(envelope.get("raw")),
            "parsing_error": repr(parsing_error) if parsing_error is not None else None,
            "parsed_is_none": parsed is None,
        })
    if parsing_error is not None:
        raise ValueError(f"{schema.__name__} 结构化响应解析失败") from parsing_error
    return schema.model_validate(parsed)
