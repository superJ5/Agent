from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeRagAgentService:
    def __init__(self) -> None:
        self.answer = "compat answer"
        self.metadata = None
        self.delay_seconds = 0.0
        self.calls: list[tuple[str, str]] = []
        self.metadata_session_id: str | None = None

    async def query(self, question: str, session_id: str) -> str:
        self.calls.append((question, session_id))
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return self.answer

    def get_last_retrieval_metadata(self, session_id: str):
        self.metadata_session_id = session_id
        return self.metadata

    async def query_stream(self, question: str, session_id: str):
        yield {"type": "complete", "data": None}


@pytest.fixture()
def competition_client(monkeypatch):
    service = FakeRagAgentService()
    config_module = SimpleNamespace(
        config=SimpleNamespace(api_bearer_token="secret-token", debug=False)
    )
    service_module = SimpleNamespace(rag_agent_service=service)
    short_term_module = SimpleNamespace(
        short_term_memory_service=SimpleNamespace(
            load_memory=lambda session_id: (
                "目标：测试短期记忆" if session_id == "session-memory" else ""
            )
        )
    )
    session_state_module = SimpleNamespace(
        session_state_service=SimpleNamespace(
            load_state=lambda session_id: (
                SimpleNamespace(
                    model_dump=lambda: {
                        "session_id": "session-state",
                        "goal": "定位支付接口变慢原因",
                        "confirmed_facts": ["22:10 后 P95 升高"],
                        "current_hypothesis": ["第三方回调超时可能导致变慢"],
                        "rejected_hypotheses": ["数据库慢查询导致变慢"],
                        "next_actions": ["检查回调重试逻辑"],
                        "user_constraints": [],
                        "updated_at": "2026-06-12T00:00:00+00:00",
                    }
                )
                if session_id == "session-state"
                else None
            )
        )
    )
    long_term_module = SimpleNamespace(
        long_term_memory_service=SimpleNamespace(
            list_memories=lambda user_id="default", include_inactive=False, limit=100: [
                {
                    "memory_id": "ltm-test",
                    "user_id": user_id,
                    "type": "preference",
                    "content": "用户喜欢大白话解释技术问题",
                    "evidence": "用户明确表达了讲解偏好",
                    "confidence": 0.95,
                    "status": "active",
                    "source_session_id": "session-memory",
                    "created_at": "2026-06-12T00:00:00+00:00",
                    "updated_at": "2026-06-12T00:00:00+00:00",
                    "expires_at": None,
                }
            ]
        )
    )

    monkeypatch.setitem(sys.modules, "app.config", config_module)
    monkeypatch.setitem(sys.modules, "app.services.rag_agent_service", service_module)
    monkeypatch.setitem(sys.modules, "app.services.short_term_memory_service", short_term_module)
    monkeypatch.setitem(sys.modules, "app.services.session_state_service", session_state_module)
    monkeypatch.setitem(sys.modules, "app.services.long_term_memory_service", long_term_module)
    sys.modules.pop("app.api.chat", None)

    chat_module = importlib.import_module("app.api.chat")
    monkeypatch.setattr(chat_module.time, "time", lambda: 1710000000)

    app = FastAPI()
    app.include_router(chat_module.competition_router)
    app.include_router(chat_module.router, prefix="/api")
    with TestClient(app) as client:
        yield client, service

    sys.modules.pop("app.api.chat", None)


def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer secret-token"}


def test_competition_chat_omits_metadata_when_empty(competition_client):
    client, service = competition_client
    service.metadata = {}

    response = client.post(
        "/chat",
        json={"question": "hello", "session_id": "session-a", "stream": False},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "code": 0,
        "msg": "success",
        "data": {
            "answer": "compat answer",
            "session_id": "session-a",
            "timestamp": 1710000000,
        },
    }
    assert "metadata" not in body["data"]
    assert service.calls == [("hello", "session-a")]
    assert service.metadata_session_id == "session-a"


def test_competition_chat_formats_markdown_images_for_api_response(competition_client):
    client, service = competition_client
    service.answer = (
        "按下鼠标底部的配对按钮。\n"
        "![Manual27_10](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_10.jpg)\n"
        "按下USB蓝牙接收器底部的按钮。\n"
        "![Manual27_11](data/manuals/raw/蓝牙激光鼠标手册/images/Manual27_11.jpg)"
    )

    response = client.post(
        "/chat",
        json={"question": "如何快速配对？", "session_id": "session-image"},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert response.json()["data"]["answer"] == (
        '"按下鼠标底部的配对按钮。\\n<PIC>\\n'
        '按下USB蓝牙接收器底部的按钮。\\n<PIC>",'
        '["Manual27_10", "Manual27_11"]'
    )
    assert service.answer.startswith("按下鼠标底部的配对按钮")


def test_competition_chat_includes_only_summary_metadata(competition_client):
    client, service = competition_client
    service.metadata = {
        "intent": "procedure",
        "doc_id": "manual-1",
        "retrieval_stage": "hybrid_search",
        "intent_strategy": "hybrid",
        "recall_channels": ["vector", "bm25"],
        "reranker_provider": "lexical",
        "reranker_fallback": True,
        "timeout": False,
        "degraded": True,
        "top_hits": [
            {
                "chunk_id": "chunk-1",
                "score": 0.91,
                "channels": ["vector"],
                "trace": {"raw_score": 123},
            }
        ],
        "warnings": ["fallback used"],
        "trace": {"raw_query": "secret"},
        "request_id": "retrieval-secret",
        "query": "secret",
        "diagnostics": {"raw_candidates": []},
    }

    response = client.post(
        "/chat",
        json={"question": "how to install", "session_id": "session-b"},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["answer"] == "compat answer"
    assert body["data"]["session_id"] == "session-b"
    assert body["data"]["timestamp"] == 1710000000
    assert body["data"]["metadata"] == {
        "intent": "procedure",
        "doc_id": "manual-1",
        "retrieval_stage": "hybrid_search",
        "intent_strategy": "hybrid",
        "recall_channels": ["vector", "bm25"],
        "reranker_provider": "lexical",
        "reranker_fallback": True,
        "timeout": False,
        "degraded": True,
        "top_hits": [
            {"chunk_id": "chunk-1", "score": 0.91, "channels": ["vector"]}
        ],
        "warnings": ["fallback used"],
    }
    assert "trace" not in body["data"]["metadata"]
    assert "request_id" not in body["data"]["metadata"]
    assert "query" not in body["data"]["metadata"]


def test_competition_chat_accepts_legacy_request_aliases_and_images(competition_client):
    client, service = competition_client

    response = client.post(
        "/chat",
        json={
            "Question": "  legacy question  ",
            "Id": "legacy-session",
            "images": ["data:image/png;base64,AAAA"],
            "stream": False,
            "ignored": "still ignored",
        },
        headers=auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["answer"] == "compat answer"
    assert body["data"]["session_id"] == "legacy-session"
    assert body["data"]["timestamp"] == 1710000000
    assert service.calls == [("legacy question", "legacy-session")]


def test_competition_chat_keeps_auth_and_question_validation(competition_client):
    client, _ = competition_client

    missing_auth = client.post("/chat", json={"question": "hello"})
    invalid_auth = client.post(
        "/chat",
        json={"question": "hello"},
        headers={"Authorization": "Bearer wrong"},
    )
    blank_question = client.post(
        "/chat",
        json={"question": "   "},
        headers=auth_headers(),
    )

    assert missing_auth.status_code == 401
    assert invalid_auth.status_code == 401
    assert blank_question.status_code == 422


def test_competition_chat_returns_fallback_on_agent_timeout(competition_client, monkeypatch):
    client, service = competition_client
    chat_module = sys.modules["app.api.chat"]
    service.delay_seconds = 0.05

    monkeypatch.setattr(chat_module, "COMPETITION_AGENT_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(chat_module, "COMPETITION_FALLBACK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        chat_module,
        "_get_cached_retrieval_fallback_answer",
        lambda: "根据已检索到的资料，简要结论如下：清洁说明: 清洁前请拔掉电源并等待设备冷却。",
    )

    response = client.post(
        "/chat",
        json={"question": "空气炸锅怎么清洁？", "session_id": "session-timeout"},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"]["answer"].startswith("根据已检索到的资料")
    assert "清洁前请拔掉电源" in body["data"]["answer"]
    assert body["data"]["metadata"]["timeout"] is True
    assert body["data"]["metadata"]["degraded"] is True


def test_short_term_memory_debug_endpoint(competition_client):
    client, _ = competition_client

    response = client.get("/api/chat/session/session-memory/short-term-memory")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "session-memory",
        "exists": True,
        "content": "目标：测试短期记忆",
    }


def test_session_state_debug_endpoint(competition_client):
    client, _ = competition_client

    response = client.get("/api/chat/session/session-state/session-state")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "session-state",
        "exists": True,
        "state": {
            "session_id": "session-state",
            "goal": "定位支付接口变慢原因",
            "confirmed_facts": ["22:10 后 P95 升高"],
            "current_hypothesis": ["第三方回调超时可能导致变慢"],
            "rejected_hypotheses": ["数据库慢查询导致变慢"],
            "next_actions": ["检查回调重试逻辑"],
            "user_constraints": [],
            "updated_at": "2026-06-12T00:00:00+00:00",
        },
    }

    missing = client.get("/api/chat/session/missing/short-term-memory")

    assert missing.status_code == 200
    assert missing.json() == {
        "session_id": "missing",
        "exists": False,
        "content": "",
    }


def test_long_term_memory_debug_endpoint(competition_client):
    client, _ = competition_client

    response = client.get("/api/chat/memory/long-term")

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "default",
        "count": 1,
        "memories": [
            {
                "memory_id": "ltm-test",
                "user_id": "default",
                "type": "preference",
                "content": "用户喜欢大白话解释技术问题",
                "evidence": "用户明确表达了讲解偏好",
                "confidence": 0.95,
                "status": "active",
                "source_session_id": "session-memory",
                "created_at": "2026-06-12T00:00:00+00:00",
                "updated_at": "2026-06-12T00:00:00+00:00",
                "expires_at": None,
            }
        ],
    }
