from __future__ import annotations

from types import SimpleNamespace

from app.retrieval import recall


class Diagnostics:
    def __init__(self) -> None:
        self.summary: dict[str, object] = {}
        self.trace: dict[str, object] = {}
        self.warnings: list[str] = []

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


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
        self.metadata = metadata or {}


class FakeVectorSearchService:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []
        self.scan_calls: list[dict[str, object]] = []

    def search_similar_documents(self, **kwargs):
        self.search_calls.append(kwargs)
        doc_id = kwargs.get("doc_id")
        if doc_id:
            return [
                Result(
                    "scoped",
                    0.1,
                    "battery install",
                    {"chunk_id": "chunk-scoped", "doc_id": doc_id},
                )
            ]
        return [
            Result(
                "safe",
                0.2,
                "battery safe path",
                {"chunk_id": "chunk-safe", "doc_id": "other-doc"},
            )
        ]

    def query_all_documents(self, **kwargs):
        self.scan_calls.append(kwargs)
        return [
            Result("scan-1", 0.0, "battery first", {"chunk_id": "scan-1"}),
            Result("scan-2", 0.0, "battery second", {"chunk_id": "scan-2"}),
            Result("scan-3", 0.0, "battery third", {"chunk_id": "scan-3"}),
        ]


class FakeBM25Provider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return [
            Result("bm25-1", 12.0, "keyword hit", {"chunk_id": "shared"}),
            Result("bm25-2", 8.0, "keyword second", {"chunk_id": "bm25-2"}),
        ]


def make_options(**overrides):
    defaults = {
        "enable_vector_recall": True,
        "enable_bm25_recall": False,
        "enable_scan_fallback": False,
        "vector_weight": 0.6,
        "bm25_weight": 0.4,
        "reranker_top_n": 8,
        "scan_candidate_limit": 4096,
        "top_k": 3,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_analysis(
    *,
    query: str = "battery",
    intent: str = "general",
    doc_score: float | None = None,
    doc_id: str = "doc-1",
    terms: list[str] | None = None,
    language: str | None = None,
):
    doc_candidates = []
    if doc_score is not None:
        doc_candidates.append(SimpleNamespace(doc_id=doc_id, score=doc_score))
    values = {
        "query": query,
        "primary_intent": intent,
        "primary_doc_id": doc_id if doc_candidates else None,
        "doc_candidates": doc_candidates,
        "query_terms": [SimpleNamespace(term=term) for term in (terms or [query])],
    }
    if language is not None:
        values["language"] = language
    return SimpleNamespace(
        **values,
    )


def patch_lightweight_helpers(monkeypatch):
    monkeypatch.setattr(
        recall,
        "_filter_results_to_active_docs",
        lambda results, scoped_doc_id: list(results),
    )
    monkeypatch.setattr(
        recall,
        "_rerank_scan_results",
        lambda results, **kwargs: list(results),
    )
    monkeypatch.setattr(recall, "_query_variants", lambda query, intent, query_terms: [query])
    monkeypatch.setattr(
        recall,
        "_resolve_chunk_types",
        lambda chunk_families, doc_id=None: list(chunk_families) if chunk_families else None,
    )
    monkeypatch.setattr(
        recall,
        "_route_for_intent",
        lambda intent: {"tiers": ("primary",), "chunk_families": ("component",)},
    )


def test_normalize_channel_scores_handles_vector_cosine_and_scan_rank():
    vector_scores = recall.normalize_channel_scores(
        [Result("high_similarity", 0.9), Result("low_similarity", 0.1)],
        "vector",
    )
    scan_scores = recall.normalize_channel_scores(
        [Result("first"), Result("second"), Result("third")],
        "scan",
    )

    assert vector_scores[0] > vector_scores[1]
    assert vector_scores[0] == 1.0
    assert scan_scores == [1.0, 2 / 3, 1 / 3]


def test_merge_recall_results_deduplicates_by_chunk_and_merges_channels():
    diagnostics = Diagnostics()
    options = make_options()
    vector_hit = Result("vector-id", 0.2, metadata={"chunk_id": "shared"})
    bm25_hit = Result("bm25-id", 7.0, metadata={"chunk_id": "shared"})
    bm25_only = Result("bm25-only", 5.0, metadata={"chunk_id": "bm25-only"})

    merged = recall.merge_recall_results(
        [("vector", [vector_hit]), ("bm25", [bm25_hit, bm25_only])],
        options,
        diagnostics,
    )

    shared = next(candidate for candidate in merged if candidate.chunk_id == "shared")
    assert shared.recall_channels == {"vector", "bm25"}
    assert set(shared.channel_scores) == {"vector", "bm25"}
    assert shared.merged_score > 0
    assert diagnostics.trace["recall"]["pre_merge"]["vector"][0]["chunk_id"] == "shared"
    assert diagnostics.trace["recall"]["post_merge"]


def test_recall_candidates_supports_bm25_only_channel(monkeypatch):
    patch_lightweight_helpers(monkeypatch)
    provider = FakeBM25Provider()
    recall.set_bm25_provider(provider)
    diagnostics = Diagnostics()

    try:
        candidates = recall.recall_candidates(
            "battery",
            make_analysis(terms=["battery"]),
            make_options(
                enable_vector_recall=False,
                enable_bm25_recall=True,
                enable_scan_fallback=False,
            ),
            diagnostics,
        )
    finally:
        recall.set_bm25_provider(None)

    assert [candidate.chunk_id for candidate in candidates] == ["shared", "bm25-2"]
    assert all(candidate.recall_channels == {"bm25"} for candidate in candidates)
    assert diagnostics.trace["recall_candidates"][0]["chunk_id"] == "shared"


def test_recall_candidates_merges_vector_and_bm25_channels(monkeypatch):
    patch_lightweight_helpers(monkeypatch)
    fake_service = FakeVectorSearchService()
    monkeypatch.setattr(recall, "vector_search_service", fake_service)
    monkeypatch.setattr(
        fake_service,
        "search_similar_documents",
        lambda **kwargs: [
            Result("vector-shared", 0.1, "keyword hit", {"chunk_id": "shared"}),
            Result("vector-only", 0.3, "vector second", {"chunk_id": "vector-only"}),
        ],
    )
    provider = FakeBM25Provider()
    recall.set_bm25_provider(provider)

    try:
        candidates = recall.recall_candidates(
            "battery",
            make_analysis(terms=["battery"]),
            make_options(
                enable_vector_recall=True,
                enable_bm25_recall=True,
                enable_scan_fallback=False,
            ),
            Diagnostics(),
        )
    finally:
        recall.set_bm25_provider(None)

    shared = next(candidate for candidate in candidates if candidate.chunk_id == "shared")
    assert shared.recall_channels == {"vector", "bm25"}
    assert set(shared.channel_scores) == {"vector", "bm25"}
    assert {candidate.chunk_id for candidate in candidates} == {
        "shared",
        "vector-only",
        "bm25-2",
    }


def test_vector_recall_uses_high_confidence_doc_filter_and_unscoped_safety_path(monkeypatch):
    patch_lightweight_helpers(monkeypatch)
    fake_service = FakeVectorSearchService()
    monkeypatch.setattr(recall, "vector_search_service", fake_service)

    results = recall.vector_recall(
        "battery",
        make_analysis(doc_score=0.95, terms=["battery"]),
        make_options(reranker_top_n=8),
    )

    assert {result.metadata["chunk_id"] for result in results} == {
        "chunk-scoped",
        "chunk-safe",
    }
    assert [call["doc_id"] for call in fake_service.search_calls] == ["doc-1", None]
    assert [call["language"] for call in fake_service.search_calls] == ["en", "en"]


def test_vector_recall_passes_chinese_language_filter(monkeypatch):
    patch_lightweight_helpers(monkeypatch)
    fake_service = FakeVectorSearchService()
    monkeypatch.setattr(recall, "vector_search_service", fake_service)

    recall.vector_recall(
        "如何安装电池",
        make_analysis(query="如何安装电池", doc_score=0.95, terms=["安装", "电池"]),
        make_options(reranker_top_n=8),
    )

    assert [call["language"] for call in fake_service.search_calls] == ["zh", "zh"]


def test_vector_recall_does_not_use_low_confidence_doc_as_only_filter(monkeypatch):
    patch_lightweight_helpers(monkeypatch)
    fake_service = FakeVectorSearchService()
    monkeypatch.setattr(recall, "vector_search_service", fake_service)

    recall.vector_recall(
        "battery",
        make_analysis(doc_score=0.2, terms=["battery"]),
        make_options(reranker_top_n=8),
    )

    assert [call["doc_id"] for call in fake_service.search_calls] == [None]


def test_bm25_unavailable_degrades_to_empty_with_warning():
    recall.set_bm25_provider(None)
    diagnostics = Diagnostics()

    results = recall.bm25_recall(
        "battery",
        make_analysis(),
        make_options(enable_bm25_recall=True),
        diagnostics,
    )

    assert results == []
    assert diagnostics.warnings == ["BM25 recall unavailable: no provider configured"]


def test_bm25_provider_results_are_returned(monkeypatch):
    patch_lightweight_helpers(monkeypatch)
    provider = FakeBM25Provider()
    recall.set_bm25_provider(provider)

    try:
        results = recall.bm25_recall(
            "battery",
            make_analysis(doc_score=0.9),
            make_options(enable_bm25_recall=True),
            Diagnostics(),
        )
    finally:
        recall.set_bm25_provider(None)

    assert [result.metadata["chunk_id"] for result in results] == ["shared", "bm25-2"]
    assert [call["doc_id"] for call in provider.calls] == ["doc-1", None]
    assert [call["language"] for call in provider.calls] == ["en", "en"]


def test_bm25_recall_detects_language_when_analysis_has_no_language(monkeypatch):
    patch_lightweight_helpers(monkeypatch)
    provider = FakeBM25Provider()
    recall.set_bm25_provider(provider)
    query = "\u7535\u6c60"

    try:
        recall.bm25_recall(
            query,
            make_analysis(query=query, terms=[query]),
            make_options(enable_bm25_recall=True),
            Diagnostics(),
        )
    finally:
        recall.set_bm25_provider(None)

    assert [call["language"] for call in provider.calls] == ["zh"]


def test_scan_recall_preserves_unscoped_safety_path_for_high_confidence_doc(monkeypatch):
    patch_lightweight_helpers(monkeypatch)
    fake_service = FakeVectorSearchService()
    monkeypatch.setattr(recall, "vector_search_service", fake_service)

    def query_all_documents(**kwargs):
        fake_service.scan_calls.append(kwargs)
        if kwargs.get("doc_id"):
            return [
                Result(
                    "scoped-scan",
                    0.0,
                    "battery scoped",
                    {"chunk_id": "scoped-scan", "doc_id": kwargs["doc_id"]},
                )
            ]
        return [
            Result(
                "safe-scan",
                0.0,
                "battery safe",
                {"chunk_id": "safe-scan", "doc_id": "other-doc"},
            )
        ]

    monkeypatch.setattr(fake_service, "query_all_documents", query_all_documents)

    results = recall.scan_recall(
        "battery",
        make_analysis(doc_score=0.95, terms=["battery"]),
        make_options(scan_candidate_limit=4, reranker_top_n=8),
    )

    assert [result.metadata["chunk_id"] for result in results] == [
        "scoped-scan",
        "safe-scan",
    ]
    assert [call["doc_id"] for call in fake_service.scan_calls] == ["doc-1", None]
    assert [call["language"] for call in fake_service.scan_calls] == ["en", "en"]


def test_scan_recall_passes_chinese_language_filter(monkeypatch):
    patch_lightweight_helpers(monkeypatch)
    fake_service = FakeVectorSearchService()
    monkeypatch.setattr(recall, "vector_search_service", fake_service)

    recall.scan_recall(
        "如何安装电池",
        make_analysis(query="如何安装电池", doc_score=0.95, terms=["安装", "电池"]),
        make_options(scan_candidate_limit=4, reranker_top_n=8),
    )

    assert [call["language"] for call in fake_service.scan_calls] == ["zh", "zh"]


def test_empty_recall_triggers_scan_and_respects_scan_limit(monkeypatch):
    patch_lightweight_helpers(monkeypatch)
    fake_service = FakeVectorSearchService()
    monkeypatch.setattr(recall, "vector_search_service", fake_service)
    monkeypatch.setattr(fake_service, "search_similar_documents", lambda **kwargs: [])
    diagnostics = Diagnostics()

    candidates = recall.recall_candidates(
        "battery",
        make_analysis(terms=["battery"]),
        make_options(enable_scan_fallback=True, scan_candidate_limit=2, reranker_top_n=8),
        diagnostics,
    )

    assert [candidate.chunk_id for candidate in candidates] == ["scan-1", "scan-2"]
    assert all(candidate.recall_channels == {"scan"} for candidate in candidates)
    assert fake_service.scan_calls[0]["batch_size"] == 2
    assert diagnostics.trace["recall"]["scan_triggered"] is True
    assert diagnostics.trace["recall"]["language_filter"] == "en"
    assert diagnostics.trace["recall"]["channel_filters"]["vector"] == {"language": "en"}
    assert diagnostics.trace["recall"]["channel_filters"]["scan"] == {"language": "en"}
