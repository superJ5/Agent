from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_reranker_module():
    module_names = ("app", "app.retrieval", "app.retrieval.schemas")
    sentinel = object()
    previous = {name: sys.modules.get(name, sentinel) for name in module_names}

    app_module = ModuleType("app")
    app_module.__path__ = [str(ROOT / "app")]
    retrieval_module = ModuleType("app.retrieval")
    retrieval_module.__path__ = [str(ROOT / "app" / "retrieval")]
    sys.modules["app"] = app_module
    sys.modules["app.retrieval"] = retrieval_module

    schemas_module = load_module(
        "app.retrieval.schemas",
        ROOT / "app" / "retrieval" / "schemas.py",
    )
    reranker_module = load_module(
        "reranker_under_test",
        ROOT / "app" / "retrieval" / "reranker.py",
    )

    for name, module in previous.items():
        if module is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

    return reranker_module, schemas_module


reranker, schemas = load_reranker_module()

RecallCandidate = schemas.RecallCandidate
RerankResult = schemas.RerankResult
RetrievalDiagnostics = schemas.RetrievalDiagnostics


class Result:
    def __init__(
        self,
        id: str,
        score: float = 0.0,
        content: str = "",
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.id = id
        self.score = score
        self.content = content
        self.metadata = metadata or {"chunk_id": id}


class FakeReranker:
    provider = "custom"

    def __init__(self, ranked: list[RecallCandidate] | None = None) -> None:
        self.ranked = ranked
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RecallCandidate],
        top_n: int,
    ) -> RerankResult:
        self.calls.append({"query": query, "candidates": candidates, "top_n": top_n})
        ranked = self.ranked if self.ranked is not None else list(reversed(candidates))
        for score, candidate in enumerate(ranked, start=1):
            candidate.diagnostics["reranker_score"] = float(score)
        return RerankResult(
            candidates=ranked,
            provider=self.provider,
            fallback_used=False,
            score_field="reranker_score",
        )


def make_options(**overrides):
    defaults = {
        "reranker_provider": "none",
        "reranker_model": "qwen3-rerank",
        "reranker_endpoint": "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
        "reranker_timeout_ms": 3000,
        "reranker_top_n": 2,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_candidate(
    chunk_id: str,
    *,
    text: str = "",
    index_text: str | None = None,
    score: float = 0.0,
    merged_score: float = 0.0,
) -> RecallCandidate:
    metadata: dict[str, object] = {"chunk_id": chunk_id}
    if index_text is not None:
        metadata["index_text"] = index_text
    return RecallCandidate(
        result=Result(chunk_id, score=score, content=text, metadata=metadata),
        chunk_id=chunk_id,
        recall_channels={"vector"},
        channel_scores={"vector": 1.0},
        merged_score=merged_score,
    )


@pytest.fixture(autouse=True)
def isolate_legacy_helpers(monkeypatch):
    monkeypatch.setattr(reranker, "_extract_query_terms", lambda query: query.split())
    monkeypatch.setattr(reranker, "_detect_intent", lambda query: "general")


def test_none_provider_uses_lexical_fallback_and_records_rank_changes(monkeypatch):
    candidates = [
        make_candidate("low", text="alpha"),
        make_candidate("high", text="beta alpha"),
        make_candidate("middle", text="alpha"),
    ]
    scores = {"low": 1.0, "high": 9.0, "middle": 4.0}
    monkeypatch.setattr(
        reranker,
        "_score_candidate",
        lambda result, **kwargs: scores[result.metadata["chunk_id"]],
    )
    diagnostics = RetrievalDiagnostics(request_id="req-rerank")

    result = reranker.rerank_candidates(
        "alpha",
        candidates,
        make_options(reranker_provider="none"),
        diagnostics,
    )

    assert [candidate.chunk_id for candidate in result.candidates] == [
        "high",
        "middle",
        "low",
    ]
    assert result.provider == "lexical"
    assert result.fallback_used is True
    assert result.score_field == "lexical_score"
    assert [candidate.diagnostics["lexical_score"] for candidate in candidates] == [
        1.0,
        9.0,
        4.0,
    ]
    assert diagnostics.summary["reranker_provider"] == "lexical"
    assert diagnostics.summary["reranker_fallback"] is True
    assert diagnostics.trace["reranker"]["pre_rank"][0]["chunk_id"] == "low"
    assert diagnostics.trace["reranker"]["post_rank"][0]["chunk_id"] == "high"
    assert diagnostics.trace["reranker"]["rank_changes"] == [
        {"chunk_id": "high", "before_rank": 2, "after_rank": 1, "delta": 1},
        {"chunk_id": "middle", "before_rank": 3, "after_rank": 2, "delta": 1},
        {"chunk_id": "low", "before_rank": 1, "after_rank": 3, "delta": -2},
    ]


def test_model_provider_success_records_reranker_result(monkeypatch):
    candidates = [
        make_candidate("first"),
        make_candidate("second"),
        make_candidate("third"),
    ]
    fake = FakeReranker(ranked=[candidates[2], candidates[0]])
    monkeypatch.setattr(reranker, "create_reranker", lambda provider: fake)
    diagnostics = RetrievalDiagnostics()

    result = reranker.rerank_candidates(
        "query",
        candidates,
        make_options(reranker_provider="custom", reranker_top_n=2),
        diagnostics,
    )

    assert fake.calls[0]["top_n"] == 2
    assert [candidate.chunk_id for candidate in result.candidates] == [
        "third",
        "first",
        "second",
    ]
    assert result.fallback_used is False
    assert diagnostics.summary["reranker_provider"] == "custom"
    assert diagnostics.summary["reranker_fallback"] is False
    assert diagnostics.trace["reranker"]["rank_changes"][0] == {
        "chunk_id": "third",
        "before_rank": 3,
        "after_rank": 1,
        "delta": 2,
    }


def test_model_exception_falls_back_to_lexical_score(monkeypatch):
    candidates = [make_candidate("first"), make_candidate("second")]

    def raise_provider(provider: str):
        raise RuntimeError("no endpoint")

    monkeypatch.setattr(reranker, "create_reranker", raise_provider)
    monkeypatch.setattr(
        reranker,
        "_score_candidate",
        lambda result, **kwargs: 10.0 if result.metadata["chunk_id"] == "second" else 1.0,
    )
    diagnostics = RetrievalDiagnostics()

    result = reranker.rerank_candidates(
        "query",
        candidates,
        make_options(reranker_provider="dashscope"),
        diagnostics,
    )

    assert [candidate.chunk_id for candidate in result.candidates] == ["second", "first"]
    assert result.provider == "lexical"
    assert result.fallback_used is True
    assert diagnostics.warnings == ["reranker failed: no endpoint"]
    assert result.warnings == ["reranker failed: no endpoint"]


def test_timeout_falls_back_and_sets_timeout_summary(monkeypatch):
    candidates = [make_candidate("first"), make_candidate("second")]

    class SlowReranker:
        provider = "custom"

        def rerank(self, query, candidates, top_n):
            time.sleep(0.05)
            return RerankResult(candidates=list(reversed(candidates)), provider="custom")

    monkeypatch.setattr(reranker, "create_reranker", lambda provider: SlowReranker())
    monkeypatch.setattr(
        reranker,
        "_score_candidate",
        lambda result, **kwargs: 5.0 if result.metadata["chunk_id"] == "first" else 1.0,
    )
    diagnostics = RetrievalDiagnostics()

    result = reranker.rerank_candidates(
        "query",
        candidates,
        make_options(reranker_provider="custom", reranker_timeout_ms=1),
        diagnostics,
    )

    assert [candidate.chunk_id for candidate in result.candidates] == ["first", "second"]
    assert result.fallback_used is True
    assert diagnostics.summary["timeout"] is True
    assert diagnostics.trace["reranker"]["timeout"] is True
    assert diagnostics.warnings == ["reranker timed out after 1 ms"]


def test_empty_model_result_falls_back(monkeypatch):
    candidates = [make_candidate("first"), make_candidate("second")]

    class EmptyReranker:
        provider = "custom"

        def rerank(self, query, candidates, top_n):
            return RerankResult(candidates=[], provider="custom")

    monkeypatch.setattr(reranker, "create_reranker", lambda provider: EmptyReranker())
    monkeypatch.setattr(
        reranker,
        "_score_candidate",
        lambda result, **kwargs: 2.0 if result.metadata["chunk_id"] == "first" else 8.0,
    )
    diagnostics = RetrievalDiagnostics()

    result = reranker.rerank_candidates(
        "query",
        candidates,
        make_options(reranker_provider="custom"),
        diagnostics,
    )

    assert [candidate.chunk_id for candidate in result.candidates] == ["second", "first"]
    assert result.provider == "lexical"
    assert diagnostics.warnings == ["reranker returned empty result"]
    assert result.warnings == ["reranker returned empty result"]


def test_no_candidates_returns_empty_without_fallback():
    diagnostics = RetrievalDiagnostics()

    result = reranker.rerank_candidates(
        "query",
        [],
        make_options(reranker_provider="custom"),
        diagnostics,
    )

    assert result.candidates == []
    assert result.provider == "custom"
    assert result.fallback_used is False
    assert diagnostics.summary["reranker_fallback"] is False
    assert diagnostics.trace["reranker"]["rank_changes"] == []


def test_create_reranker_exposes_reserved_entries_and_rejects_unknown():
    created = reranker.create_reranker("dashscope", api_key="")

    assert created.provider == "dashscope"
    with pytest.raises(reranker.RerankerUnavailableError):
        created.rerank("query", [make_candidate("first")], 1)
    with pytest.raises(reranker.RerankerUnavailableError):
        reranker.create_reranker("unknown")


def test_dashscope_provider_posts_text_and_records_scores(monkeypatch):
    candidates = [
        make_candidate("first", text="raw first", index_text="index first"),
        make_candidate("second", text="raw second", index_text="index second"),
        make_candidate("third", text="raw third", index_text="index third"),
    ]
    calls = []

    def fake_post_dashscope_rerank(**kwargs):
        calls.append(kwargs)
        return {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                ]
            }
        }

    monkeypatch.setattr(reranker, "_post_dashscope_rerank", fake_post_dashscope_rerank)
    monkeypatch.setattr(
        reranker,
        "_config_value",
        lambda name, default: "test-key" if name == "dashscope_api_key" else default,
    )
    diagnostics = RetrievalDiagnostics()

    result = reranker.rerank_candidates(
        "query",
        candidates,
        make_options(
            reranker_provider="dashscope",
            reranker_endpoint="https://example.test/reranks",
            reranker_top_n=2,
        ),
        diagnostics,
    )

    assert calls[0]["endpoint"] == "https://example.test/reranks"
    assert calls[0]["payload"] == {
        "model": "qwen3-rerank",
        "query": "query",
        "documents": ["raw first", "raw second"],
        "top_n": 2,
    }
    assert [candidate.chunk_id for candidate in result.candidates] == [
        "second",
        "first",
        "third",
    ]
    assert result.provider == "dashscope"
    assert result.fallback_used is False
    assert candidates[1].diagnostics["reranker_score"] == 0.91
    assert candidates[0].diagnostics["reranker_score"] == 0.42
    assert diagnostics.summary["reranker_provider"] == "dashscope"
    assert diagnostics.summary["reranker_fallback"] is False
    assert diagnostics.trace["reranker"]["model"] == "qwen3-rerank"
    assert diagnostics.trace["reranker"]["endpoint"] == "https://example.test/reranks"


def test_dashscope_missing_key_falls_back_to_lexical(monkeypatch):
    candidates = [make_candidate("first"), make_candidate("second")]
    monkeypatch.setattr(reranker, "_config_value", lambda name, default: "")
    monkeypatch.setattr(
        reranker,
        "_score_candidate",
        lambda result, **kwargs: 3.0 if result.metadata["chunk_id"] == "first" else 1.0,
    )
    diagnostics = RetrievalDiagnostics()

    result = reranker.rerank_candidates(
        "query",
        candidates,
        make_options(reranker_provider="dashscope"),
        diagnostics,
    )

    assert [candidate.chunk_id for candidate in result.candidates] == ["first", "second"]
    assert result.provider == "lexical"
    assert result.fallback_used is True
    assert diagnostics.summary["reranker_fallback"] is True
    assert diagnostics.warnings == ["reranker failed: dashscope api key is missing"]


def test_run_with_timeout_returns_value_and_propagates_errors():
    assert reranker.run_with_timeout(lambda: "ok", timeout_ms=100) == "ok"

    with pytest.raises(ValueError):
        reranker.run_with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), 100)
