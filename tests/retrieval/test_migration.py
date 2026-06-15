from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

SUMMARY_METADATA_KEYS = (
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
TRACE_METADATA_KEYS = {
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


@dataclass
class Result:
    id: str
    content: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Bundle:
    intent: str = "general"
    retrieval_stage: str = "none"
    hits: list[Result] = field(default_factory=list)
    support_hits: list[Result] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_hits(self) -> list[Result]:
        return [*self.hits, *self.support_hits]


class DocumentStub:
    def __init__(self, page_content: str, metadata: dict[str, Any]) -> None:
        self.page_content = page_content
        self.metadata = metadata


class NoopLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


def install_module(monkeypatch, name: str, **attrs):
    module = ModuleType(name)
    if name in {
        "app",
        "app.models",
        "app.retrieval",
        "app.services",
        "langchain_core",
    }:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def load_module(monkeypatch, module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def sanitize_summary_metadata(metadata):
    if not isinstance(metadata, dict):
        return None
    sanitized = {
        key: json_safe_summary_value(metadata[key])
        for key in SUMMARY_METADATA_KEYS
        if key in metadata
    }
    return sanitized or None


def json_safe_summary_value(value):
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {
            str(key): json_safe_summary_value(item)
            for key, item in value.items()
            if str(key) not in TRACE_METADATA_KEYS
        }
    if isinstance(value, list | tuple):
        return [json_safe_summary_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return [json_safe_summary_value(item) for item in sorted(value, key=str)]
    return str(value)


def fake_tool(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda func: func


def make_summary_metadata(**overrides):
    metadata = {
        "intent": "procedure",
        "doc_id": "manual-1",
        "retrieval_stage": "hybrid_search",
        "intent_strategy": "hybrid",
        "recall_channels": ["vector"],
        "reranker_provider": "lexical",
        "reranker_fallback": False,
        "timeout": False,
        "degraded": False,
        "top_hits": [
            {
                "chunk_id": "chunk-1",
                "score": 0.9,
                "channels": ["vector"],
                "trace": {"raw_score": 99},
            }
        ],
        "warnings": [],
        "trace": {"raw_query": "secret"},
        "request_id": "retrieval-secret",
        "query": "secret",
    }
    metadata.update(overrides)
    return metadata


def install_knowledge_dependencies(monkeypatch, bundle: Bundle | None = None):
    install_module(monkeypatch, "app")
    install_module(
        monkeypatch,
        "app.config",
        config=SimpleNamespace(rag_top_k=3),
    )
    install_module(monkeypatch, "app.models")
    install_module(
        monkeypatch,
        "app.models.response",
        sanitize_summary_metadata=sanitize_summary_metadata,
    )
    install_module(monkeypatch, "app.services")
    install_module(
        monkeypatch,
        "app.services.vector_search_service",
        SearchResult=Result,
        vector_search_service=SimpleNamespace(),
    )
    install_module(monkeypatch, "langchain_core")
    install_module(
        monkeypatch,
        "langchain_core.documents",
        Document=DocumentStub,
    )
    install_module(monkeypatch, "langchain_core.tools", tool=fake_tool)
    install_module(monkeypatch, "loguru", logger=NoopLogger())

    install_module(monkeypatch, "app.retrieval")
    orchestrator = install_module(monkeypatch, "app.retrieval.orchestrator")
    orchestrator.calls = []
    orchestrator.error = None
    orchestrator.bundle = bundle

    def retrieve(query):
        orchestrator.calls.append(query)
        if orchestrator.error is not None:
            raise orchestrator.error
        return orchestrator.bundle

    orchestrator.retrieve = retrieve

    install_module(
        monkeypatch,
        "app.retrieval.evidence",
        search_result_to_document=lambda result: DocumentStub(
            page_content=result.content,
            metadata={**(result.metadata or {}), "score": result.score},
        ),
        format_bundle=lambda used_bundle, query=None: (
            f"context::{query}::{used_bundle.intent}::{len(used_bundle.hits)}"
        ),
        bundle_to_evidence_payload=lambda used_bundle, query=None: {
            "query": query,
            "hits": list(used_bundle.hits),
        },
        search_result_to_evidence_hit=lambda result, rank, role: {
            "rank": rank,
            "role": role,
            "chunk_id": result.metadata.get("chunk_id") or result.id,
        },
        unique_flatten=lambda items: [
            item
            for group in items
            for item in (group if isinstance(group, list | tuple | set) else [group])
        ],
        format_search_results=lambda results: "\n".join(result.content for result in results),
    )
    return orchestrator


def load_knowledge_tool(monkeypatch, bundle: Bundle | None = None):
    orchestrator = install_knowledge_dependencies(monkeypatch, bundle)
    module = load_module(
        monkeypatch,
        "knowledge_tool_migration_under_test",
        "app/tools/knowledge_tool.py",
    )
    assert module.retrieval_orchestrator is orchestrator
    return module, orchestrator


def test_routed_retrieve_prioritizes_new_orchestrator_and_sets_summary_metadata(monkeypatch):
    hit = Result("hit-1", "primary text", 0.9, {"chunk_id": "chunk-1"})
    bundle = Bundle(
        intent="procedure",
        retrieval_stage="hybrid_search",
        hits=[hit],
        metadata=make_summary_metadata(),
    )
    module, orchestrator = load_knowledge_tool(monkeypatch, bundle)

    returned = module.routed_retrieve("install battery")

    assert returned is bundle
    assert orchestrator.calls == ["install battery"]
    metadata = module.get_last_retrieval_metadata()
    assert metadata == {
        "intent": "procedure",
        "doc_id": "manual-1",
        "retrieval_stage": "hybrid_search",
        "intent_strategy": "hybrid",
        "recall_channels": ["vector"],
        "reranker_provider": "lexical",
        "reranker_fallback": False,
        "timeout": False,
        "degraded": False,
        "top_hits": [
            {"chunk_id": "chunk-1", "score": 0.9, "channels": ["vector"]}
        ],
        "warnings": [],
    }
    assert "trace" not in metadata
    assert "request_id" not in metadata
    assert "query" not in metadata


def test_routed_retrieve_falls_back_to_legacy_when_orchestrator_fails(monkeypatch):
    module, orchestrator = load_knowledge_tool(monkeypatch, Bundle())
    orchestrator.error = RuntimeError("orchestrator down")
    legacy_calls = []
    legacy_bundle = module.RetrievalBundle(
        intent="general",
        retrieval_stage="legacy_stage",
        hits=[],
        support_hits=[],
        warnings=[],
        metadata=make_summary_metadata(
            intent="general",
            doc_id=None,
            retrieval_stage="legacy_stage",
            intent_strategy="legacy_routed",
            recall_channels=["scan"],
            top_hits=[],
        ),
    )

    def legacy(query):
        legacy_calls.append(query)
        return legacy_bundle

    monkeypatch.setattr(module, "_legacy_routed_retrieve", legacy)

    returned = module.routed_retrieve("fallback question")

    assert returned is legacy_bundle
    assert orchestrator.calls == ["fallback question"]
    assert legacy_calls == ["fallback question"]
    assert legacy_bundle.warnings == ["orchestrator fallback: orchestrator down"]
    metadata = module.get_last_retrieval_metadata()
    assert metadata["intent"] == "general"
    assert metadata["retrieval_stage"] == "legacy_stage"
    assert metadata["intent_strategy"] == "legacy_routed"
    assert metadata["degraded"] is True
    assert metadata["warnings"] == ["orchestrator fallback: orchestrator down"]
    assert "trace" not in metadata


def test_retrieve_knowledge_still_returns_context_and_documents(monkeypatch):
    hit = Result("hit-1", "primary text", 0.9, {"chunk_id": "chunk-1"})
    bundle = Bundle(
        intent="procedure",
        retrieval_stage="hybrid_search",
        hits=[hit],
        metadata=make_summary_metadata(),
    )
    module, _ = load_knowledge_tool(monkeypatch, bundle)

    context, docs = module.retrieve_knowledge("install battery")

    assert context == "context::install battery::procedure::1"
    assert len(docs) == 1
    assert docs[0].page_content == "primary text"
    assert docs[0].metadata["chunk_id"] == "chunk-1"
    assert module.get_last_retrieval_metadata()["intent"] == "procedure"


def test_retrieve_knowledge_passes_original_question_as_rerank_query(monkeypatch):
    hit = Result("hit-1", "primary text", 0.9, {"chunk_id": "chunk-1"})
    bundle = Bundle(
        intent="procedure",
        retrieval_stage="hybrid_search",
        hits=[hit],
        metadata=make_summary_metadata(),
    )
    module, orchestrator = load_knowledge_tool(monkeypatch, bundle)
    calls = []

    def retrieve(query, *, rerank_query=None):
        calls.append((query, rerank_query))
        return bundle

    orchestrator.retrieve = retrieve
    monkeypatch.setattr(
        module,
        "get_trace_chat_context",
        lambda: {"question": "How to start my jetski in different situations?"},
    )

    context, docs = module.retrieve_knowledge("jetski start different situations")

    assert calls == [
        (
            "jetski start different situations",
            "How to start my jetski in different situations?",
        )
    ]
    assert context == "context::jetski start different situations::procedure::1"
    assert len(docs) == 1


def test_retrieval_fallback_answers_are_isolated_by_session(monkeypatch):
    module, _ = load_knowledge_tool(monkeypatch, Bundle())

    module.set_last_retrieval_fallback_answer("answer-a", session_id="session-a")
    module.set_last_retrieval_fallback_answer("answer-b", session_id="session-b")

    assert module.get_last_retrieval_fallback_answer(session_id="session-a") == "answer-a"
    assert module.get_last_retrieval_fallback_answer(session_id="session-b") == "answer-b"

    module.clear_last_retrieval_metadata(session_id="session-a")

    assert module.get_last_retrieval_fallback_answer(session_id="session-a") is None
    assert module.get_last_retrieval_fallback_answer(session_id="session-b") == "answer-b"
