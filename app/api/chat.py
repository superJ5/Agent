"""对话接口。"""

from __future__ import annotations

import json
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, status
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.config import config
from app.models.request import ChatRequest, ClearRequest
from app.models.response import (
    ApiResponse,
    SessionInfoResponse,
)
from app.services.memory_service import memory_service
from app.services.rag_agent_service import rag_agent_service


router = APIRouter()
competition_router = APIRouter()


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


def _competition_success_payload(answer: str, session_id: str) -> dict:
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "answer": answer,
            "session_id": session_id,
            "timestamp": int(time.time()),
        },
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
    source: str = "chat_stream",
) -> EventSourceResponse:
    async def event_generator():
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
            },
        )
        try:
            async for chunk in rag_agent_service.query_stream(
                question,
                session_id=session_id,
                images=images,
            ):
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

    return EventSourceResponse(event_generator())


@competition_router.post("/chat")
async def competition_chat(
    request: ChatRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
):
    """比赛标准接口。"""
    _require_bearer_token(authorization)

    session_id = _resolve_session_id(request.session_id)
    logger.info(
        "[会话 {}] 收到比赛标准对话请求: question='{}', images={}, stream={}",
        session_id,
        request.question,
        len(request.images),
        request.stream,
    )

    if request.stream:
        return _build_stream_response(
            request.question,
            session_id,
            request.images,
            source="competition_chat",
        )

    try:
        memory_service.append_message(
            session_id,
            "user",
            request.question,
            metadata={
                "source": "competition_chat",
                "stream": False,
                "images_count": len(request.images),
            },
        )
        answer = await rag_agent_service.query(
            request.question,
            session_id=session_id,
            images=request.images,
        )
        memory_service.append_message(
            session_id,
            "assistant",
            answer,
            metadata={
                "source": "competition_chat",
                "stream": False,
                "status": "success",
            },
        )
        logger.info(f"[会话 {session_id}] 比赛标准对话完成")
        return _competition_success_payload(answer, session_id)
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


@router.post("/chat")
async def chat(request: ChatRequest):
    """旧版快速对话接口。"""
    session_id = _resolve_session_id(request.session_id)

    try:
        logger.info(f"[会话 {session_id}] 收到快速对话请求: {request.question}")
        memory_service.append_message(
            session_id,
            "user",
            request.question,
            metadata={
                "source": "legacy_chat",
                "stream": False,
                "images_count": len(request.images),
            },
        )
        answer = await rag_agent_service.query(
            request.question,
            session_id=session_id,
            images=request.images,
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
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str) -> SessionInfoResponse:
    """查询会话历史。"""
    try:
        history = rag_agent_service.get_session_history(session_id)

        return SessionInfoResponse(
            session_id=session_id,
            message_count=len(history),
            history=history,
        )

    except Exception as exc:
        logger.error(f"获取会话信息错误: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
