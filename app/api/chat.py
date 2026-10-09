"""对话接口。"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Query, status
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.config import config
from app.core.request_context import current_memory_user_id, memory_enabled_override
from app.models.request import ChatRequest, ClearRequest
from app.models.response import (
    ApiResponse,
    LongTermMemoryListResponse,
    SessionInfoResponse,
    SessionStateResponse,
    ShortTermMemoryResponse,
)
from app.services.competition_answer_formatter import format_answer_images
from app.services.rag_agent_service import rag_agent_service

_memory_service: Any
try:
    from app.services.memory_service import memory_service as _memory_service
except Exception:  # pragma: no cover - keeps API compatibility tests importable with stubs.
    class _NoopMemoryService:
        def append_message(self, *args: Any, **kwargs: Any) -> None:
            return None

    _memory_service = _NoopMemoryService()

memory_service: Any = _memory_service

_short_term_memory_service: Any
try:
    from app.services.short_term_memory_service import (
        short_term_memory_service as _short_term_memory_service,
    )
except Exception:  # pragma: no cover - keeps API compatibility tests importable with stubs.
    class _NoopShortTermMemoryService:
        def load_memory(self, *args: Any, **kwargs: Any) -> str:
            return ""

    _short_term_memory_service = _NoopShortTermMemoryService()

short_term_memory_service: Any = _short_term_memory_service

_session_state_service: Any
try:
    from app.services.session_state_service import (
        session_state_service as _session_state_service,
    )
except Exception:  # pragma: no cover - keeps API compatibility tests importable with stubs.
    class _NoopSessionStateService:
        def load_state(self, *args: Any, **kwargs: Any) -> None:
            return None

    _session_state_service = _NoopSessionStateService()

session_state_service: Any = _session_state_service

_long_term_memory_service: Any
try:
    from app.services.long_term_memory_service import (
        long_term_memory_service as _long_term_memory_service,
    )
except Exception:  # pragma: no cover - keeps API compatibility tests importable with stubs.
    class _NoopLongTermMemoryService:
        def list_memories(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            return []

    _long_term_memory_service = _NoopLongTermMemoryService()

long_term_memory_service: Any = _long_term_memory_service

router = APIRouter()
competition_router = APIRouter()

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


COMPETITION_AGENT_TIMEOUT_SECONDS = _float_env("COMPETITION_AGENT_TIMEOUT_SECONDS", 27.0)
COMPETITION_FALLBACK_TIMEOUT_SECONDS = _float_env("COMPETITION_FALLBACK_TIMEOUT_SECONDS", 2.5)
FALLBACK_MAX_HITS = 3
FALLBACK_MAX_CHARS_PER_HIT = 180


def _resolve_session_id(session_id: str | None) -> str:
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    return f"kf_session_{uuid.uuid4().hex}"


def _require_bearer_token(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )

    provided_token = authorization.removeprefix("Bearer ").strip()
    if not provided_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token",
        )

    configured_token = (config.api_bearer_token or "").strip()
    if configured_token and provided_token != configured_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )


def _competition_success_payload(
    answer: str,
    session_id: str,
) -> dict:
    data: dict[str, Any] = {
        "answer": answer,
        "session_id": session_id,
        "timestamp": int(time.time()),
    }

    return {
        "code": 0,
        "msg": "success",
        "data": data,
    }


def _competition_error_payload(message: str, session_id: str) -> dict:
    return {
        "code": 500,
        "msg": message,
        "data": {
            "answer": "",
            "session_id": session_id,
            "timestamp": int(time.time()),
        },
    }


def _build_stream_response(
    question: str,
    session_id: str,
    images: list[str] | None = None,
    model: str | None = None,
    dashscope_api_key: str | None = None,
    memory_enabled: bool | None = None,
    source: str = "chat_stream",
) -> EventSourceResponse:
    async def event_generator_impl():
        answer_parts: list[str] = []
        assistant_logged = False
        memory_service.append_message(
            session_id,
            "user",
            question,
            metadata={
                "source": source,
                "stream": True,
                "images_count": len(images or []),
                "model": model,
                "dashscope_api_key_provided": bool(dashscope_api_key),
            },
        )
        try:
            query_stream_kwargs: dict[str, Any] = {"session_id": session_id}
            if _call_accepts_keyword(rag_agent_service.query_stream, "images"):
                query_stream_kwargs["images"] = images
            if _call_accepts_keyword(rag_agent_service.query_stream, "model"):
                query_stream_kwargs["model"] = model
            if _call_accepts_keyword(rag_agent_service.query_stream, "dashscope_api_key"):
                query_stream_kwargs["dashscope_api_key"] = dashscope_api_key
            async for chunk in rag_agent_service.query_stream(question, **query_stream_kwargs):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                if chunk_type == "debug":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "debug",
                                "node": chunk.get("node", "unknown"),
                                "message_type": chunk.get("message_type", "unknown"),
                            },
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type in {"tool_call", "search_results", "content"}:
                    if chunk_type == "content" and isinstance(chunk_data, str):
                        answer_parts.append(chunk_data)
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": chunk_type,
                                "data": chunk_data,
                            },
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "complete":
                    memory_service.append_message(
                        session_id,
                        "assistant",
                        "".join(answer_parts),
                        metadata={
                            "source": source,
                            "stream": True,
                            "status": "success",
                        },
                    )
                    assistant_logged = True
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "done",
                                "data": chunk_data,
                            },
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "error":
                    memory_service.append_message(
                        session_id,
                        "assistant",
                        "".join(answer_parts),
                        metadata={
                            "source": source,
                            "stream": True,
                            "status": "error",
                            "error": str(chunk_data),
                        },
                    )
                    assistant_logged = True
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "error",
                                "data": str(chunk_data),
                            },
                            ensure_ascii=False,
                        ),
                    }
        except Exception as exc:
            logger.error(f"流式对话接口错误: {exc}")
            if not assistant_logged:
                memory_service.append_message(
                    session_id,
                    "assistant",
                    "".join(answer_parts),
                    metadata={
                        "source": source,
                        "stream": True,
                        "status": "error",
                        "error": str(exc),
                    },
                )
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "error",
                        "data": str(exc),
                    },
                    ensure_ascii=False,
                ),
            }

    async def event_generator():
        with memory_enabled_override(memory_enabled):
            async for event in event_generator_impl():
                yield event

    return EventSourceResponse(event_generator())


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == keyword
        for name, parameter in signature.parameters.items()
    )


def _default_chat_model_name() -> str:
    return str(getattr(config, "rag_model", None) or getattr(config, "dashscope_model", "") or "")


async def _query_rag_agent(
    question: str,
    *,
    session_id: str,
    images: list[str] | None,
    model: str | None,
    dashscope_api_key: str | None,
) -> str:
    kwargs: dict[str, Any] = {"session_id": session_id}
    if _call_accepts_keyword(rag_agent_service.query, "images"):
        kwargs["images"] = images
    if _call_accepts_keyword(rag_agent_service.query, "model"):
        kwargs["model"] = model
    if _call_accepts_keyword(rag_agent_service.query, "dashscope_api_key"):
        kwargs["dashscope_api_key"] = dashscope_api_key
    return str(await rag_agent_service.query(question, **kwargs))


async def _query_rag_agent_with_competition_deadline(
    question: str,
    *,
    session_id: str,
    images: list[str] | None,
    model: str | None,
    dashscope_api_key: str | None,
) -> tuple[str, bool]:
    try:
        answer = await asyncio.wait_for(
            _query_rag_agent(
                question,
                session_id=session_id,
                images=images,
                model=model,
                dashscope_api_key=dashscope_api_key,
            ),
            timeout=COMPETITION_AGENT_TIMEOUT_SECONDS,
        )
        return answer, False
    except TimeoutError:
        logger.warning(
            "[会话 {}] 比赛标准对话接近 30s 限制，改用检索证据兜底返回",
            session_id,
        )
        answer = await _build_timeout_fallback_answer(
            question,
            session_id=session_id,
        )
        return answer, True


async def _build_timeout_fallback_answer(
    question: str,
    *,
    session_id: str,
) -> str:
    try:
        cached_answer = await _wait_for_cached_retrieval_fallback_answer(
            session_id,
            timeout=COMPETITION_FALLBACK_TIMEOUT_SECONDS,
        )
        if cached_answer:
            return cached_answer
    finally:
        _clear_cached_retrieval_fallback_answer(session_id)

    return "根据当前已完成的信息，暂时没有拿到足够可靠的资料来给出完整结论。"


async def _wait_for_cached_retrieval_fallback_answer(
    session_id: str,
    *,
    timeout: float,
) -> str | None:
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        answer = _get_cached_retrieval_fallback_answer(session_id)
        if answer:
            return answer
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(0.1)


def _get_cached_retrieval_fallback_answer(session_id: str) -> str | None:
    try:
        from app.tools.knowledge_tool import get_last_retrieval_fallback_answer

        return get_last_retrieval_fallback_answer(session_id=session_id)
    except Exception as exc:
        logger.debug(f"读取检索兜底答案失败，忽略: {exc}")
        return None


def _clear_cached_retrieval_fallback_answer(session_id: str) -> None:
    try:
        from app.tools.knowledge_tool import set_last_retrieval_fallback_answer

        set_last_retrieval_fallback_answer(None, session_id=session_id)
    except Exception as exc:
        logger.debug(f"清理检索兜底答案失败，忽略: {exc}")


def _format_fallback_answer_from_hits(hits: list[Any]) -> str:
    evidence_lines: list[str] = []
    for hit in hits[:FALLBACK_MAX_HITS]:
        text = _fallback_hit_text(hit)
        if not text:
            continue
        title = _fallback_hit_title(hit)
        if title:
            evidence_lines.append(f"{title}: {text}")
        else:
            evidence_lines.append(text)

    if not evidence_lines:
        return "根据当前已完成的信息，暂时没有检索到足够可靠的资料来给出完整结论。"

    return "根据已检索到的资料，简要结论如下：" + "；".join(evidence_lines)


def _fallback_hit_text(hit: Any) -> str:
    metadata = _fallback_hit_metadata(hit)
    raw_text = (
        getattr(hit, "content", None)
        or metadata.get("text")
        or metadata.get("content")
        or metadata.get("summary")
        or metadata.get("index_text")
        or ""
    )
    text = " ".join(str(raw_text).split())
    if len(text) > FALLBACK_MAX_CHARS_PER_HIT:
        text = text[:FALLBACK_MAX_CHARS_PER_HIT].rstrip() + "..."
    return text


def _fallback_hit_title(hit: Any) -> str:
    metadata = _fallback_hit_metadata(hit)
    title = metadata.get("title") or metadata.get("section_title")
    if title:
        return str(title).strip()
    section_path = metadata.get("section_path") or []
    if isinstance(section_path, list):
        return " > ".join(str(part).strip() for part in section_path if str(part).strip())
    return ""


def _fallback_hit_metadata(hit: Any) -> dict[str, Any]:
    metadata = getattr(hit, "metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


@competition_router.post("/chat")
async def competition_chat(
    request: ChatRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
):
    """比赛标准接口。"""
    _require_bearer_token(authorization)

    session_id = _resolve_session_id(request.session_id)
    logger.info(
        "[会话 {}] 收到比赛标准对话请求: question='{}', images={}, stream={}, model={}, dashscope_api_key_provided={}",
        session_id,
        request.question,
        len(request.images),
        request.stream,
        request.model or _default_chat_model_name(),
        bool(request.dashscope_api_key),
    )

    if request.stream:
        return _build_stream_response(
            request.question,
            session_id,
            request.images,
            request.model,
            request.dashscope_api_key,
            request.memory_enabled,
            source="competition_chat",
        )

    memory_context = memory_enabled_override(request.memory_enabled)
    try:
        memory_context.__enter__()
        memory_service.append_message(
            session_id,
            "user",
            request.question,
            metadata={
                "source": "competition_chat",
                "stream": False,
                "images_count": len(request.images),
                "model": request.model,
                "dashscope_api_key_provided": bool(request.dashscope_api_key),
                "memory_enabled_override": request.memory_enabled,
            },
        )
        (
            answer,
            used_timeout_fallback,
        ) = await _query_rag_agent_with_competition_deadline(
            request.question,
            session_id=session_id,
            images=request.images,
            model=request.model,
            dashscope_api_key=request.dashscope_api_key,
        )
        memory_service.append_message(
            session_id,
            "assistant",
            answer,
            metadata={
                "source": "competition_chat",
                "stream": False,
                "status": "timeout_fallback" if used_timeout_fallback else "success",
            },
        )
        logger.info(f"[会话 {session_id}] 比赛标准对话完成")
        formatted_answer = format_answer_images(answer)
        return _competition_success_payload(formatted_answer, session_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"比赛标准对话接口错误: {exc}")
        memory_service.append_message(
            session_id,
            "assistant",
            "",
            metadata={
                "source": "competition_chat",
                "stream": False,
                "status": "error",
                "error": str(exc),
            },
        )
        return _competition_error_payload(str(exc), session_id)
    finally:
        memory_context.__exit__(None, None, None)


@router.post("/chat")
async def chat(request: ChatRequest):
    """旧版快速对话接口。"""
    session_id = _resolve_session_id(request.session_id)

    try:
        with memory_enabled_override(request.memory_enabled):
            logger.info(f"[会话 {session_id}] 收到快速对话请求: {request.question}")
            memory_service.append_message(
                session_id,
                "user",
                request.question,
                metadata={
                    "source": "legacy_chat",
                    "stream": False,
                    "images_count": len(request.images),
                    "model": request.model,
                    "dashscope_api_key_provided": bool(request.dashscope_api_key),
                    "memory_enabled_override": request.memory_enabled,
                },
            )
            answer = await _query_rag_agent(
                request.question,
                session_id=session_id,
                images=request.images,
                model=request.model,
                dashscope_api_key=request.dashscope_api_key,
            )
            memory_service.append_message(
                session_id,
                "assistant",
                answer,
                metadata={
                    "source": "legacy_chat",
                    "stream": False,
                    "status": "success",
                },
            )

        logger.info(f"[会话 {session_id}] 快速对话完成")

        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": answer,
                "errorMessage": None,
            },
        }

    except Exception as exc:
        logger.error(f"对话接口错误: {exc}")
        memory_service.append_message(
            session_id,
            "assistant",
            "",
            metadata={
                "source": "legacy_chat",
                "stream": False,
                "status": "error",
                "error": str(exc),
            },
        )
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(exc),
            },
        }


@router.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """旧版流式接口。"""
    session_id = _resolve_session_id(request.session_id)
    logger.info(f"[会话 {session_id}] 收到流式对话请求: {request.question}")
    return _build_stream_response(
        request.question,
        session_id,
        request.images,
        request.model,
        request.dashscope_api_key,
        request.memory_enabled,
        source="legacy_chat_stream",
    )


@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(request: ClearRequest):
    """清空会话历史。"""
    try:
        success = rag_agent_service.clear_session(request.session_id)
        logger.info(f"清空会话: {request.session_id}, 结果: {success}")

        return ApiResponse(
            status="success" if success else "error",
            message="会话已清空" if success else "清空会话失败",
            data=None,
        )

    except Exception as exc:
        logger.error(f"清空会话错误: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(
    session_id: str,
    memory_enabled: bool | None = Query(default=None),
) -> SessionInfoResponse:
    """查询会话历史。"""
    try:
        with memory_enabled_override(memory_enabled):
            history = rag_agent_service.get_session_history(session_id)

        return SessionInfoResponse(
            session_id=session_id,
            message_count=len(history),
            history=history,
        )

    except Exception as exc:
        logger.error(f"获取会话信息错误: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/chat/session/{session_id}/short-term-memory",
    response_model=ShortTermMemoryResponse,
)
async def get_short_term_memory(
    session_id: str,
    memory_enabled: bool | None = Query(default=None),
) -> ShortTermMemoryResponse:
    """查询当前 session 的短期语义记忆。"""
    try:
        with memory_enabled_override(memory_enabled):
            content = str(short_term_memory_service.load_memory(session_id) or "")
        return ShortTermMemoryResponse(
            session_id=session_id,
            exists=bool(content.strip()),
            content=content,
        )
    except Exception as exc:
        logger.error(f"获取短期语义记忆错误: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/chat/session/{session_id}/session-state",
    response_model=SessionStateResponse,
)
async def get_session_state(
    session_id: str,
    memory_enabled: bool | None = Query(default=None),
) -> SessionStateResponse:
    """查询当前 session 的结构化状态。"""
    try:
        with memory_enabled_override(memory_enabled):
            state = session_state_service.load_state(session_id)
        state_payload = state.model_dump() if hasattr(state, "model_dump") else state
        return SessionStateResponse(
            session_id=session_id,
            exists=state is not None,
            state=state_payload if isinstance(state_payload, dict) else None,
        )
    except Exception as exc:
        logger.error(f"获取 Session State 错误: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/chat/memory/long-term", response_model=LongTermMemoryListResponse)
async def list_long_term_memory(
    include_inactive: bool = False,
    limit: int = 100,
    memory_enabled: bool | None = Query(default=None),
) -> LongTermMemoryListResponse:
    """查询长期记忆列表。"""
    try:
        with memory_enabled_override(memory_enabled):
            user_id = current_memory_user_id()
            memories = long_term_memory_service.list_memories(
                user_id=user_id,
                include_inactive=include_inactive,
                limit=limit,
            )
        return LongTermMemoryListResponse(
            user_id=user_id,
            count=len(memories),
            memories=memories,
        )
    except Exception as exc:
        logger.error(f"获取长期记忆错误: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
