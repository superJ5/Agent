from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Result:
    id: str
    content: str = ""
    score: float = 0.0
    metadata: dict[str, object] | None = None


class FakeLogger:
    def __init__(self) -> None:
        self.debug_calls: list[object] = []

    def debug(self, *args, **kwargs) -> None:
        self.debug_calls.append((args, kwargs))


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def load_orchestrator_module():
    module_names = (
        "app",
        "app.config",
        "app.retrieval",
        "app.retrieval.schemas",
        "app.retrieval.diagnostics",
        "app.retrieval.evidence",
        "app.retrieval.query_understanding",
        "app.retrieval.recall",
        "app.retrieval.reranker",
        "loguru",
    )
    sentinel = object()
    previous = {name: sys.modules.get(name, sentinel) for name in module_names}

    app_module = make_module("app")
    app_module.__path__ = [str(ROOT / "app")]
    retrieval_module = make_module("app.retrieval")
    retrieval_module.__path__ = [str(ROOT / "app" / "retrieval")]

    sys.modules["app"] = app_module
    sys.modules["app.config"] = make_module(
        "app.config",
        config=SimpleNamespace(debug=False),
    )
    sys.modules["app.retrieval"] = retrieval_module
    sys.modules["loguru"] = make_module("loguru", logger=FakeLogger())

    schemas = load_module(
        "app.retrieval.schemas",
        ROOT / "app" / "retrieval" / "schemas.py",
    )
    diagnostics = load_module(
        "app.retrieval.diagnostics",
        ROOT / "app" / "retrieval" / "diagnostics.py",
    )

    sys.modules["app.retrieval.query_understanding"] = make_module(
        "app.retrieval.query_understanding",
        analyze_query=lambda *args, **kwargs: None,
    )
    sys.modules["app.retrieval.recall"] = make_module(
        "app.retrieval.recall",
        recall_candidates=lambda *args, **kwargs: [],
    )
    sys.modules["app.retrieval.reranker"] = make_module(
        "app.retrieval.reranker",
        rerank_candidates=lambda *args, **kwargs: schemas.RerankResult(),
        lexical_fallback=lambda query, candidates, diagnostics, warnings=None: schemas.RerankResult(
            candidates=list(candidates),
            provider="lexical",
            fallback_used=True,
            warnings=list(warnings or []),
            score_field="lexical_score",
        ),
    )
    sys.modules["app.retrieval.evidence"] = make_module(
        "app.retrieval.evidence",
        build_retrieval_bundle=lambda *args, **kwargs: schemas.RetrievalBundle(),
    )

    orchestrator = load_module(
        "orchestrator_under_test",
        ROOT / "app" / "retrieval" / "orchestrator.py",
    )

    for name, module in previous.items():
        if module is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

    return orchestrator, schemas, diagnostics


def test_retrieve_runs_pipeline_from_config_and_writes_summary_metadata(monkeypatch):
    orchestrator, schemas, _ = load_orchestrator_module()
    result = Result(
        "result-1",
        content="install battery",
        score=0.2,
        metadata={"chunk_id": "chunk-1"},
    )
    candidate = schemas.RecallCandidate(
        result=result,
        chunk_id="chunk-1",
        recall_channels={"vector"},
        channel_scores={"vector": 1.0},
        merged_score=0.7,
        diagnostics={"reranker_score": 0.88},
    )
    analysis = schemas.QueryAnalysis(
        query="install battery",
        strategy="rules",
        intent_candidates=[schemas.IntentCandidate("procedure", 0.9, "rules")],
        doc_candidates=[schemas.DocCandidate("manual-1", 0.9, "rules")],
    )
    trace_calls = []
    seen = {}

    def analyze_query(query, options, diagnostics):
        seen["options"] = options
        assert diagnostics.request_id == "req-orch"
        return analysis

    def recall_candidates(query, used_analysis, options, diagnostics):
        assert used_analysis is analysis
        return [candidate]

    def rerank_candidates(query, candidates, options, diagnostics):
        assert candidates == [candidate]
        return schemas.RerankResult(
            candidates=list(candidates),
            provider="model",
            fallback_used=False,
            score_field="reranker_score",
        )

    def build_retrieval_bundle(query, used_analysis, rerank_result, options, diagnostics):
        assert rerank_result.provider == "model"
        return schemas.RetrievalBundle(
            intent="procedure",
            retrieval_stage="hybrid_search",
            hits=[result],
            metadata={"evidence_schema_version": "test"},
        )

    monkeypatch.setattr(orchestrator, "config", SimpleNamespace(
        rag_intent_strategy="rules",
        rag_enable_vector_recall=False,
        rag_enable_bm25_recall=True,
        rag_top_k=2,
    ))
    monkeypatch.setattr(orchestrator, "new_request_id", lambda: "req-orch")
    monkeypatch.setattr(orchestrator, "analyze_query", analyze_query)
    monkeypatch.setattr(orchestrator, "recall_candidates", recall_candidates)
    monkeypatch.setattr(orchestrator, "rerank_candidates", rerank_candidates)
    monkeypatch.setattr(orchestrator, "build_retrieval_bundle", build_retrieval_bundle)
    monkeypatch.setattr(
        orchestrator,
        "write_trace_if_enabled",
        lambda diagnostics: trace_calls.append(diagnostics),
    )

    bundle = orchestrator.retrieve("install battery")

    assert isinstance(bundle, schemas.RetrievalBundle)
    assert seen["options"].intent_strategy == "rules"
    assert seen["options"].enable_vector_recall is False
    assert seen["options"].enable_bm25_recall is True
    assert seen["options"].top_k == 2
    assert bundle.metadata["evidence_schema_version"] == "test"
    assert bundle.metadata["intent"] == "procedure"
    assert bundle.metadata["doc_id"] == "manual-1"
    assert bundle.metadata["recall_channels"] == ["vector"]
    assert bundle.metadata["reranker_provider"] == "model"
    assert bundle.metadata["top_hits"] == [
        {
            "chunk_id": "chunk-1",
            "score": 0.88,
            "channels": ["vector"],
            "reranker_score": 0.88,
        }
    ]
    assert trace_calls[0].summary == {
        "intent": "procedure",
        "doc_id": "manual-1",
        "retrieval_stage": "hybrid_search",
        "intent_strategy": "rules",
        "recall_channels": ["vector"],
        "reranker_provider": "model",
        "reranker_fallback": False,
        "timeout": False,
        "degraded": False,
        "top_hits": [
            {
                "chunk_id": "chunk-1",
                "score": 0.88,
                "channels": ["vector"],
                "reranker_score": 0.88,
            }
        ],
        "warnings": [],
    }


def test_query_understanding_exception_degrades_to_none(monkeypatch):
    orchestrator, schemas, _ = load_orchestrator_module()
    captured = {}
    trace_calls = []

    def recall_candidates(query, analysis, options, diagnostics):
        captured["analysis"] = analysis
        return []

    monkeypatch.setattr(
        orchestrator,
        "analyze_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken rules")),
    )
    monkeypatch.setattr(orchestrator, "recall_candidates", recall_candidates)
    monkeypatch.setattr(
        orchestrator,
        "rerank_candidates",
        lambda *args, **kwargs: schemas.RerankResult(candidates=[], provider="none"),
    )
    monkeypatch.setattr(
        orchestrator,
        "build_retrieval_bundle",
        lambda *args, **kwargs: schemas.RetrievalBundle(intent="general", retrieval_stage="none"),
    )
    monkeypatch.setattr(
        orchestrator,
        "write_trace_if_enabled",
        lambda diagnostics: trace_calls.append(diagnostics),
    )

    bundle = orchestrator.retrieve("install battery", schemas.RetrievalOptions())

    assert captured["analysis"].strategy == "none"
    assert captured["analysis"].query_terms == []
    assert bundle.metadata["intent_strategy"] == "none"
    assert bundle.metadata["degraded"] is True
    assert bundle.metadata["warnings"] == [
        "query understanding failed: broken rules",
        "query understanding degraded to none",
    ]
    assert trace_calls[0].trace["query_understanding"]["error"] == "broken rules"


def test_recall_exception_degrades_to_empty_candidates(monkeypatch):
    orchestrator, schemas, _ = load_orchestrator_module()
    analysis = schemas.QueryAnalysis(query="query", strategy="none")
    seen = {}

    monkeypatch.setattr(orchestrator, "analyze_query", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        orchestrator,
        "recall_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("store down")),
    )

    def rerank_candidates(query, candidates, options, diagnostics):
        seen["candidates"] = candidates
        return schemas.RerankResult(candidates=[], provider="none", fallback_used=False)

    monkeypatch.setattr(orchestrator, "rerank_candidates", rerank_candidates)
    monkeypatch.setattr(
        orchestrator,
        "build_retrieval_bundle",
        lambda *args, **kwargs: schemas.RetrievalBundle(intent="general", retrieval_stage="none"),
    )
    monkeypatch.setattr(orchestrator, "write_trace_if_enabled", lambda diagnostics: None)

    bundle = orchestrator.retrieve("query", schemas.RetrievalOptions())

    assert seen["candidates"] == []
    assert bundle.metadata["warnings"] == ["recall failed: store down"]
    assert bundle.metadata["degraded"] is True


def test_reranker_exception_uses_lexical_fallback(monkeypatch):
    orchestrator, schemas, _ = load_orchestrator_module()
    low = schemas.RecallCandidate(
        result=Result("low", metadata={"chunk_id": "low"}),
        chunk_id="low",
        recall_channels={"vector"},
        channel_scores={"vector": 0.4},
        merged_score=0.4,
    )
    high = schemas.RecallCandidate(
        result=Result("high", metadata={"chunk_id": "high"}),
        chunk_id="high",
        recall_channels={"bm25"},
        channel_scores={"bm25": 0.8},
        merged_score=0.8,
    )
    fallback_calls = []

    monkeypatch.setattr(
        orchestrator,
        "analyze_query",
        lambda query, options, diagnostics: schemas.QueryAnalysis(query=query, strategy="none"),
    )
    monkeypatch.setattr(
        orchestrator,
        "recall_candidates",
        lambda *args, **kwargs: [low, high],
    )
    monkeypatch.setattr(
        orchestrator,
        "rerank_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad model")),
    )

    def lexical_fallback(query, candidates, diagnostics, warnings=None):
        fallback_calls.append({"candidates": list(candidates), "warnings": list(warnings or [])})
        high.diagnostics["lexical_score"] = 9.0
        low.diagnostics["lexical_score"] = 1.0
        return schemas.RerankResult(
            candidates=[high, low],
            provider="lexical",
            fallback_used=True,
            warnings=list(warnings or []),
            score_field="lexical_score",
        )

    monkeypatch.setattr(orchestrator, "lexical_fallback", lexical_fallback)
    monkeypatch.setattr(
        orchestrator,
        "build_retrieval_bundle",
        lambda query, analysis, rerank_result, options, diagnostics: schemas.RetrievalBundle(
            intent="general",
            retrieval_stage="hybrid_search",
            hits=[candidate.result for candidate in rerank_result.candidates[:1]],
        ),
    )
    monkeypatch.setattr(orchestrator, "write_trace_if_enabled", lambda diagnostics: None)

    bundle = orchestrator.retrieve("query", schemas.RetrievalOptions(top_k=1))

    assert fallback_calls == [
        {
            "candidates": [low, high],
            "warnings": ["reranker failed: bad model"],
        }
    ]
    assert bundle.hits == [high.result]
    assert bundle.metadata["reranker_provider"] == "lexical"
    assert bundle.metadata["reranker_fallback"] is True
    assert bundle.metadata["warnings"] == ["reranker failed: bad model"]


def test_evidence_exception_returns_primary_candidate_fallback_bundle(monkeypatch):
    orchestrator, schemas, _ = load_orchestrator_module()
    first = schemas.RecallCandidate(
        result=Result("first", metadata={"chunk_id": "first"}),
        chunk_id="first",
        recall_channels={"vector"},
        channel_scores={"vector": 1.0},
        merged_score=0.7,
    )
    second = schemas.RecallCandidate(
        result=Result("second", metadata={"chunk_id": "second"}),
        chunk_id="second",
        recall_channels={"bm25"},
        channel_scores={"bm25": 1.0},
        merged_score=0.4,
    )
    analysis = schemas.QueryAnalysis(
        query="install",
        strategy="rules",
        intent_candidates=[schemas.IntentCandidate("procedure", 0.9, "rules")],
    )

    monkeypatch.setattr(orchestrator, "analyze_query", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(orchestrator, "recall_candidates", lambda *args, **kwargs: [first, second])
    monkeypatch.setattr(
        orchestrator,
        "rerank_candidates",
        lambda *args, **kwargs: schemas.RerankResult(
            candidates=[first, second],
            provider="lexical",
            fallback_used=True,
            score_field="lexical_score",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "build_retrieval_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("formatter down")),
    )
    monkeypatch.setattr(orchestrator, "write_trace_if_enabled", lambda diagnostics: None)

    bundle = orchestrator.retrieve("install", schemas.RetrievalOptions(top_k=1))

    assert bundle.intent == "procedure"
    assert bundle.retrieval_stage == "hybrid_search"
    assert bundle.hits == [first.result]
    assert bundle.support_hits == []
    assert bundle.metadata["query"] == "install"
    assert bundle.metadata["warnings"] == ["evidence organization failed: formatter down"]
    assert bundle.metadata["top_hits"] == [
        {"chunk_id": "first", "score": 0.7, "channels": ["vector"]},
        {"chunk_id": "second", "score": 0.4, "channels": ["bm25"]},
    ]
