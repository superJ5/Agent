from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SearchResultStub:
    id: str
    score: float
    metadata: dict
    content: str = ""


@dataclass
class RecallCandidateStub:
    result: SearchResultStub
    chunk_id: str = ""
    recall_channels: set[str] = field(default_factory=set)
    channel_scores: dict[str, float] = field(default_factory=dict)
    merged_score: float | None = 0.0
    diagnostics: dict = field(default_factory=dict)


class FakeLogger:
    def __init__(self):
        self.debug_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))


def install_module(monkeypatch, name: str, **attrs):
    module = SimpleNamespace(**attrs)
    if name == "app":
        module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def diagnostics_module(monkeypatch):
    install_module(monkeypatch, "app")
    install_module(monkeypatch, "app.config", config=SimpleNamespace(debug=False))
    install_module(monkeypatch, "loguru", logger=FakeLogger())

    spec = importlib.util.spec_from_file_location(
        "diagnostics_under_test",
        ROOT / "app" / "retrieval" / "diagnostics.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "diagnostics_under_test", module)
    spec.loader.exec_module(module)
    return module


def make_candidate(
    chunk_id: str,
    channels: set[str],
    merged_score: float = 0.0,
    diagnostics: dict | None = None,
) -> RecallCandidateStub:
    return RecallCandidateStub(
        result=SearchResultStub(
            id=f"{chunk_id}_result",
            score=0.42,
            metadata={"chunk_id": chunk_id},
        ),
        chunk_id=chunk_id,
        recall_channels=channels,
        channel_scores=dict.fromkeys(channels, 1.0),
        merged_score=merged_score,
        diagnostics=diagnostics or {},
    )


def test_summarize_top_hits_outputs_compact_ranked_fields(diagnostics_module):
    candidates = [
        make_candidate(
            "chunk_001",
            {"bm25", "vector"},
            merged_score=0.71,
            diagnostics={"reranker_score": 0.91},
        ),
        make_candidate(
            "chunk_002",
            {"scan"},
            merged_score=0.21,
            diagnostics={"lexical_score": 12.5},
        ),
        make_candidate("chunk_003", {"bm25"}, merged_score=0.33),
    ]

    assert diagnostics_module.summarize_top_hits(candidates, limit=2) == [
        {
            "chunk_id": "chunk_001",
            "score": 0.91,
            "channels": ["vector", "bm25"],
            "reranker_score": 0.91,
        },
        {"chunk_id": "chunk_002", "score": 12.5, "channels": ["scan"]},
    ]


def test_summarize_top_hits_falls_back_to_result_chunk_id_and_score(diagnostics_module):
    candidate = RecallCandidateStub(
        result=SearchResultStub(
            id="result_id",
            score=0.77,
            metadata={"chunk_id": "metadata_chunk"},
        ),
        recall_channels={"custom"},
        merged_score=None,
    )

    assert diagnostics_module.summarize_top_hits([candidate]) == [
        {"chunk_id": "metadata_chunk", "score": 0.77, "channels": ["custom"]}
    ]


def test_collect_recall_channels_returns_unique_preferred_order(diagnostics_module):
    candidates = [
        make_candidate("chunk_001", {"scan", "bm25"}),
        make_candidate("chunk_002", {"vector", "bm25"}),
        make_candidate("chunk_003", {"custom"}),
    ]

    assert diagnostics_module.collect_recall_channels(candidates) == [
        "vector",
        "bm25",
        "scan",
        "custom",
    ]


def test_build_summary_metadata_includes_contract_fields_and_warnings(diagnostics_module):
    candidates = [
        make_candidate(
            "chunk_001",
            {"vector", "bm25"},
            diagnostics={"reranker_score": 0.91},
        )
    ]
    diagnostics = SimpleNamespace(
        request_id="req-1",
        summary={"timeout": True},
        trace={},
        warnings=["bm25 recall failed"],
    )
    analysis = SimpleNamespace(
        primary_intent="procedure",
        primary_doc_id="manual_60e374ac",
        strategy="hybrid",
        warnings=[],
    )
    bundle = SimpleNamespace(
        intent="procedure",
        retrieval_stage="hybrid_search",
        warnings=["bm25 recall failed"],
    )
    rerank_result = SimpleNamespace(
        candidates=candidates,
        provider="lexical",
        fallback_used=True,
        warnings=["reranker fallback used"],
    )

    metadata = diagnostics_module.build_summary_metadata(
        diagnostics, bundle, analysis, rerank_result
    )

    assert list(metadata) == [
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
    ]
    assert "request_id" not in metadata
    assert "trace" not in metadata
    assert metadata == {
        "intent": "procedure",
        "doc_id": "manual_60e374ac",
        "retrieval_stage": "hybrid_search",
        "intent_strategy": "hybrid",
        "recall_channels": ["vector", "bm25"],
        "reranker_provider": "lexical",
        "reranker_fallback": True,
        "timeout": True,
        "degraded": True,
        "top_hits": [
            {
                "chunk_id": "chunk_001",
                "score": 0.91,
                "channels": ["vector", "bm25"],
                "reranker_score": 0.91,
            }
        ],
        "warnings": ["bm25 recall failed", "reranker fallback used"],
    }


def test_build_summary_metadata_handles_one_shot_candidate_iterables(diagnostics_module):
    candidates = [
        make_candidate("chunk_001", {"vector"}, diagnostics={"reranker_score": 0.91}),
        make_candidate("chunk_002", {"bm25"}, merged_score=0.33),
    ]
    diagnostics = SimpleNamespace(summary={}, trace={"raw_query": "secret"}, warnings=[])
    analysis = SimpleNamespace(
        primary_intent="faq",
        primary_doc_id=None,
        strategy="rules",
        warnings=[],
    )
    bundle = SimpleNamespace(intent="general", retrieval_stage="hybrid_search", warnings=[])
    rerank_result = SimpleNamespace(
        candidates=(candidate for candidate in candidates),
        provider="none",
        fallback_used=False,
        warnings=[],
    )

    metadata = diagnostics_module.build_summary_metadata(
        diagnostics, bundle, analysis, rerank_result
    )

    assert metadata["recall_channels"] == ["vector", "bm25"]
    assert metadata["top_hits"] == [
        {
            "chunk_id": "chunk_001",
            "score": 0.91,
            "channels": ["vector"],
            "reranker_score": 0.91,
        },
        {"chunk_id": "chunk_002", "score": 0.33, "channels": ["bm25"]},
    ]
    assert "raw_query" not in metadata


def test_record_trace_event_appends_json_safe_events_and_request_id(diagnostics_module):
    diagnostics = SimpleNamespace(request_id="req-trace", trace={}, warnings=[])

    diagnostics_module.record_trace_event(
        diagnostics, "recall", {"channels": {"bm25", "vector"}}
    )
    diagnostics_module.record_trace_event(
        diagnostics, "recall", SimpleNamespace(stage="rerank", score=0.8)
    )

    assert diagnostics.trace == {
        "request_id": "req-trace",
        "recall": [
            {"channels": ["bm25", "vector"]},
            {"stage": "rerank", "score": 0.8},
        ],
    }


def test_write_trace_if_enabled_logs_when_debug_disabled(
    diagnostics_module, monkeypatch, tmp_path
):
    fake_logger = FakeLogger()
    trace_path = tmp_path / "retrieval_trace.jsonl"
    diagnostics = SimpleNamespace(
        request_id="req-log",
        trace={"recall": [{"chunk_id": "chunk_001"}]},
        warnings=[],
    )

    monkeypatch.setattr(diagnostics_module, "config", SimpleNamespace(debug=False))
    monkeypatch.setattr(diagnostics_module, "logger", fake_logger)
    monkeypatch.setattr(diagnostics_module, "TRACE_LOG_PATH", trace_path)

    diagnostics_module.write_trace_if_enabled(diagnostics)

    assert not trace_path.exists()
    assert fake_logger.debug_calls[0][0][0] == "retrieval trace: {}"
    assert fake_logger.debug_calls[0][0][1] == {
        "recall": [{"chunk_id": "chunk_001"}],
        "request_id": "req-log",
    }


def test_write_trace_if_enabled_appends_jsonl_when_debug_enabled(
    diagnostics_module, monkeypatch, tmp_path
):
    trace_path = tmp_path / "retrieval_trace.jsonl"
    diagnostics = SimpleNamespace(
        request_id="req-file",
        trace={"reranker": [{"channels": {"scan", "vector"}}]},
        warnings=["lexical fallback used"],
    )

    monkeypatch.setattr(diagnostics_module, "config", SimpleNamespace(debug=True))
    monkeypatch.setattr(diagnostics_module, "TRACE_LOG_PATH", trace_path)

    diagnostics_module.write_trace_if_enabled(diagnostics)

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "request_id": "req-file",
        "reranker": [{"channels": ["scan", "vector"]}],
        "warnings": ["lexical fallback used"],
    }


def test_write_trace_if_enabled_includes_current_chat_context(
    diagnostics_module, monkeypatch, tmp_path
):
    trace_path = tmp_path / "retrieval_trace.jsonl"
    diagnostics = SimpleNamespace(
        request_id="req-chat",
        trace={"query": "吹风机 安全要点"},
        warnings=[],
    )

    monkeypatch.setattr(diagnostics_module, "config", SimpleNamespace(debug=True))
    monkeypatch.setattr(diagnostics_module, "TRACE_LOG_PATH", trace_path)

    token = diagnostics_module.set_trace_chat_context(
        question="操作吹风机时，人员需要注意哪些安全要点？",
        session_id="kf_session_test",
    )
    try:
        diagnostics_module.write_trace_if_enabled(diagnostics)
    finally:
        diagnostics_module.reset_trace_chat_context(token)

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "request_id": "req-chat",
        "query": "吹风机 安全要点",
        "question": "操作吹风机时，人员需要注意哪些安全要点？",
        "session_id": "kf_session_test",
    }


def test_module_exposes_retrieval_diagnostics_name(diagnostics_module):
    assert diagnostics_module.RetrievalDiagnostics is not None
