from __future__ import annotations

import importlib.util
import sys
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_PATH = ROOT / "app" / "retrieval" / "schemas.py"

spec = importlib.util.spec_from_file_location("retrieval_schemas_under_test", SCHEMAS_PATH)
assert spec and spec.loader
schemas = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = schemas
spec.loader.exec_module(schemas)

DocCandidate = schemas.DocCandidate
IntentCandidate = schemas.IntentCandidate
QueryAnalysis = schemas.QueryAnalysis
QueryTerm = schemas.QueryTerm
RecallCandidate = schemas.RecallCandidate
RerankResult = schemas.RerankResult
RetrievalBundle = schemas.RetrievalBundle
RetrievalDiagnostics = schemas.RetrievalDiagnostics
RetrievalOptions = schemas.RetrievalOptions
chunk_id_from_search_result = schemas.chunk_id_from_search_result
derive_chunk_id = schemas.derive_chunk_id
get_chunk_id_from_search_result = schemas.get_chunk_id_from_search_result
get_result_chunk_id = schemas.get_result_chunk_id
search_result_chunk_id = schemas.search_result_chunk_id


@dataclass
class FakeSearchResult:
    id: str
    content: str = ""
    score: float = 0.0
    metadata: dict | None = None


def test_candidate_dataclasses_keep_expected_fields_and_are_frozen():
    intent = IntentCandidate(
        intent="procedure",
        score=0.9,
        source="rules",
        reason="matched workflow verb",
    )
    doc = DocCandidate(doc_id="manual_60e374ac", score=0.7, source="summary")
    term = QueryTerm(term="fuel", source="query", weight=1.5)

    assert intent.intent == "procedure"
    assert intent.reason == "matched workflow verb"
    assert doc.reason == ""
    assert term.weight == 1.5

    with pytest.raises(FrozenInstanceError):
        intent.score = 0.1


def test_query_analysis_primary_intent_returns_highest_score_candidate():
    analysis = QueryAnalysis(
        query="how do I install the battery",
        strategy="hybrid",
        intent_candidates=[
            IntentCandidate("general", 0.2, "profile"),
            IntentCandidate("procedure", 0.95, "rules"),
            IntentCandidate("safety", 0.7, "llm"),
        ],
    )

    assert analysis.primary_intent == "procedure"


def test_query_analysis_primary_intent_is_none_without_candidates():
    analysis = QueryAnalysis(query="hello", strategy="none")

    assert analysis.primary_intent is None


def test_query_analysis_primary_doc_id_returns_highest_score_candidate():
    analysis = QueryAnalysis(
        query="which manual covers the fuel cap",
        strategy="hybrid",
        doc_candidates=[
            DocCandidate("manual_a", 0.3, "profile"),
            DocCandidate("manual_b", 0.8, "summary"),
        ],
    )

    assert analysis.primary_doc_id == "manual_b"


def test_query_analysis_primary_doc_id_is_none_without_candidates():
    analysis = QueryAnalysis(query="hello", strategy="none")

    assert analysis.primary_doc_id is None


def test_chunk_id_helpers_prefer_metadata_chunk_id():
    result = FakeSearchResult(
        id="milvus_id",
        content="content",
        score=0.42,
        metadata={"chunk_id": "metadata_chunk", "doc_id": "manual_1"},
    )

    assert chunk_id_from_search_result(result) == "metadata_chunk"
    assert get_result_chunk_id(result) == "metadata_chunk"
    assert get_chunk_id_from_search_result(result) == "metadata_chunk"
    assert search_result_chunk_id(result) == "metadata_chunk"
    assert derive_chunk_id(result) == "metadata_chunk"


def test_chunk_id_helper_falls_back_to_result_id_when_metadata_chunk_id_missing():
    result = FakeSearchResult(
        id="milvus_id",
        content="content",
        score=0.42,
        metadata={"doc_id": "manual_1"},
    )

    assert chunk_id_from_search_result(result) == "milvus_id"


def test_chunk_id_helper_accepts_mapping_like_results():
    result = {"id": "dict_id", "metadata": {"chunk_id": "dict_chunk"}}

    assert chunk_id_from_search_result(result) == "dict_chunk"


def test_recall_candidate_derives_chunk_id_in_post_init():
    result = FakeSearchResult(
        id="milvus_id",
        content="content",
        score=0.42,
        metadata={"chunk_id": "metadata_chunk", "doc_id": "manual_1"},
    )

    candidate = RecallCandidate(result=result)

    assert candidate.chunk_id == "metadata_chunk"


def test_recall_candidate_from_search_result_sets_channel_details():
    result = FakeSearchResult(
        id="chunk_1",
        content="content",
        score=0.73,
        metadata={"chunk_id": "chunk_1"},
    )

    candidate = RecallCandidate.from_search_result(result, recall_channel="vector")

    assert candidate.chunk_id == "chunk_1"
    assert candidate.recall_channels == {"vector"}
    assert candidate.channel_scores == {"vector": 0.73}
    assert candidate.merged_score == 0.73
    assert candidate.diagnostics == {}


def test_recall_candidate_from_search_result_accepts_positional_channel_and_score():
    result = FakeSearchResult(id="chunk_1", score=0.73, metadata={"chunk_id": "chunk_1"})

    candidate = RecallCandidate.from_search_result(result, "bm25", 0.5)

    assert candidate.recall_channels == {"bm25"}
    assert candidate.channel_scores == {"bm25": 0.5}
    assert candidate.merged_score == 0.5


def test_rerank_result_defaults_are_isolated():
    first = RerankResult()
    second = RerankResult()

    first.warnings.append("fallback")

    assert first.candidates == []
    assert first.provider == "none"
    assert first.fallback_used is False
    assert first.score_field == "reranker_score"
    assert second.warnings == []


def test_retrieval_diagnostics_add_warning_and_summary_metadata_excludes_trace():
    diagnostics = RetrievalDiagnostics(
        request_id="req-1",
        summary={"intent": "procedure", "top_hits": [{"chunk_id": "chunk_1"}]},
        trace={"raw_candidates": ["chunk_1", "chunk_2"]},
    )

    diagnostics.add_warning("reranker fallback")
    metadata = diagnostics.to_summary_metadata()

    assert diagnostics.warnings == ["reranker fallback"]
    assert metadata == {
        "request_id": "req-1",
        "intent": "procedure",
        "top_hits": [{"chunk_id": "chunk_1"}],
        "warnings": ["reranker fallback"],
        "degraded": True,
    }
    assert "trace" not in metadata
    assert "raw_candidates" not in metadata


def test_retrieval_bundle_keeps_legacy_shape_and_deduplicates_all_hits():
    primary = FakeSearchResult(id="primary-id", metadata={"chunk_id": "shared"})
    duplicate_support = FakeSearchResult(id="support-id", metadata={"chunk_id": "shared"})
    support = FakeSearchResult(id="support-2", metadata={"doc_id": "manual_1"})

    bundle = RetrievalBundle(
        intent="procedure",
        retrieval_stage="hybrid_search",
        hits=[primary],
        support_hits=[duplicate_support, support],
        warnings=["fallback"],
    )

    assert bundle.intent == "procedure"
    assert bundle.retrieval_stage == "hybrid_search"
    assert bundle.warnings == ["fallback"]
    assert bundle.all_hits == [primary, support]


def test_retrieval_options_from_config_reads_rag_fields():
    config = SimpleNamespace(
        rag_intent_strategy="rules",
        rag_enable_vector_recall=False,
        rag_enable_bm25_recall=True,
        rag_vector_weight=0.25,
        rag_bm25_weight=0.75,
        rag_reranker_provider="dashscope",
        rag_reranker_model="qwen3-rerank",
        rag_reranker_endpoint="https://example.test/reranks",
        rag_reranker_timeout_ms=1200,
        rag_reranker_top_n=12,
        rag_enable_scan_fallback=False,
        rag_scan_candidate_limit=128,
        rag_top_k=5,
    )

    options = RetrievalOptions.from_config(config)

    assert options == RetrievalOptions(
        intent_strategy="rules",
        enable_vector_recall=False,
        enable_bm25_recall=True,
        vector_weight=0.25,
        bm25_weight=0.75,
        reranker_provider="dashscope",
        reranker_model="qwen3-rerank",
        reranker_endpoint="https://example.test/reranks",
        reranker_timeout_ms=1200,
        reranker_top_n=12,
        enable_scan_fallback=False,
        scan_candidate_limit=128,
        top_k=5,
    )


def test_retrieval_options_from_config_uses_defaults_for_missing_fields():
    options = RetrievalOptions.from_config(SimpleNamespace(rag_top_k=7))

    assert options.intent_strategy == "hybrid"
    assert options.enable_vector_recall is True
    assert options.enable_bm25_recall is True
    assert options.vector_weight == 0.6
    assert options.bm25_weight == 0.4
    assert options.reranker_provider == "none"
    assert options.reranker_model == "qwen3-rerank"
    assert options.reranker_endpoint == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    assert options.reranker_timeout_ms == 3000
    assert options.reranker_top_n == 32
    assert options.enable_scan_fallback is True
    assert options.scan_candidate_limit == 4096
    assert options.top_k == 7


def test_retrieval_options_from_config_accepts_mapping_and_string_values():
    options = RetrievalOptions.from_config(
        {
            "rag_enable_vector_recall": "false",
            "rag_enable_bm25_recall": "true",
            "rag_vector_weight": "0.2",
            "rag_bm25_weight": "0.8",
            "rag_reranker_timeout_ms": "900",
            "rag_reranker_top_n": "9",
            "rag_enable_scan_fallback": "off",
            "rag_scan_candidate_limit": "64",
            "rag_top_k": "4",
        }
    )

    assert options.enable_vector_recall is False
    assert options.enable_bm25_recall is True
    assert options.vector_weight == 0.2
    assert options.bm25_weight == 0.8
    assert options.reranker_timeout_ms == 900
    assert options.reranker_top_n == 9
    assert options.enable_scan_fallback is False
    assert options.scan_candidate_limit == 64
    assert options.top_k == 4


def test_retrieval_options_from_config_accepts_unprefixed_fallback_keys():
    options = RetrievalOptions.from_config({"top_k": "6", "intent_strategy": "rules"})

    assert options.top_k == 6
    assert options.intent_strategy == "rules"
