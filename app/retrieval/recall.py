"""Recall and merge stage for the retrieval pipeline."""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from app.retrieval.tier_policy import RECALL_INCLUDED_TIERS, is_recallable_tier

if TYPE_CHECKING:
    from app.retrieval.schemas import (
        QueryAnalysis,
        RecallCandidate,
        RetrievalDiagnostics,
        RetrievalOptions,
    )
else:
    try:  # The schema module is owned by a sibling task; keep this import contract stable.
        from app.retrieval.schemas import (
            QueryAnalysis,
            RecallCandidate,
            RetrievalDiagnostics,
            RetrievalOptions,
        )
    except ImportError:  # pragma: no cover - used only while sibling modules are absent.
        QueryAnalysis = Any
        RetrievalDiagnostics = Any

        @dataclass
        class RecallCandidate:  # type: ignore[no-redef]
            result: Any
            chunk_id: str
            recall_channels: set[str]
            channel_scores: dict[str, float]
            merged_score: float = 0.0
            diagnostics: dict[str, Any] = field(default_factory=dict)

        @dataclass(frozen=True)
        class RetrievalOptions:  # type: ignore[no-redef]
            intent_strategy: str = "hybrid"
            enable_vector_recall: bool = True
            enable_bm25_recall: bool = True
            vector_weight: float = 0.6
            bm25_weight: float = 0.4
            reranker_provider: str = "none"
            reranker_timeout_ms: int = 3000
            reranker_top_n: int = 32
            enable_scan_fallback: bool = True
            scan_candidate_limit: int = 4096
            top_k: int = 3

if TYPE_CHECKING:
    from app.services.vector_search_service import SearchResult
else:
    SearchResult = Any

vector_search_service: Any
try:
    from app.services.vector_search_service import vector_search_service as _vector_search_service
except Exception:  # pragma: no cover - lets isolated tests import without Milvus deps.
    vector_search_service = None
else:
    vector_search_service = _vector_search_service


DOC_FILTER_CONFIDENCE_THRESHOLD = 0.7
SCAN_CHANNEL_WEIGHT = 0.2
MIN_RECALL_FETCH = 8


class BM25Provider(Protocol):
    """Minimal BM25 provider contract used by the recall stage."""

    def search(
        self,
        query: str,
        *,
        top_k: int,
        doc_id: str | None = None,
        retrieval_tiers: Sequence[str] | None = None,
        chunk_types: Sequence[str] | None = None,
        language: str | None = None,
    ) -> list[SearchResult]:
        """Return lexical search hits."""


_bm25_provider: BM25Provider | None = None


def effective_recall_tiers(route_tiers: Sequence[str] | None) -> list[str]:
    """Return the tiers allowed at a normal recall service boundary."""
    if not route_tiers:
        return list(RECALL_INCLUDED_TIERS)
    tiers = [route_tiers] if isinstance(route_tiers, str) else list(route_tiers)
    return list(
        dict.fromkeys(
            tier
            for raw_tier in tiers
            if (tier := str(raw_tier or "").strip()) and is_recallable_tier(tier)
        )
    )


def filter_recallable_results(results: Sequence[SearchResult]) -> list[SearchResult]:
    """Keep only results whose retrieval tier may participate in normal recall."""
    return [
        result
        for result in results
        if is_recallable_tier(_result_retrieval_tier(result))
    ]


def set_bm25_provider(provider: BM25Provider | None) -> None:
    """Install or clear the process-local BM25 provider."""
    global _bm25_provider
    _bm25_provider = provider


def recall_candidates(
    query: str,
    analysis: QueryAnalysis,
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
) -> list[RecallCandidate]:
    """Run enabled recall channels, merge them, and optionally scan as fallback."""
    channel_results: list[tuple[str, list[SearchResult]]] = []
    trace = _recall_trace(diagnostics)
    trace["query"] = query
    language = _analysis_language(analysis, query)
    trace["language"] = language
    trace["language_filter"] = language
    trace["effective_recall_routes"] = _serialize_effective_recall_routes(
        _primary_intent(analysis),
        analysis,
    )
    channel_filters = trace.setdefault("channel_filters", {})

    if getattr(options, "enable_vector_recall", True):
        try:
            vector_results = vector_recall(query, analysis, options)
            channel_results.append(("vector", vector_results))
            if isinstance(channel_filters, dict):
                channel_filters["vector"] = {"language": language}
            trace.setdefault("channels", {})["vector"] = _serialize_results(vector_results)
        except Exception as exc:
            _add_warning(diagnostics, f"vector recall failed: {exc}")

    if getattr(options, "enable_bm25_recall", True):
        try:
            bm25_results = bm25_recall(query, analysis, options, diagnostics)
            channel_results.append(("bm25", bm25_results))
            if isinstance(channel_filters, dict):
                channel_filters["bm25"] = {"language": language}
            trace.setdefault("channels", {})["bm25"] = _serialize_results(bm25_results)
        except Exception as exc:
            _add_warning(diagnostics, f"bm25 recall failed: {exc}")

    merged = merge_recall_results(channel_results, options, diagnostics)

    if should_trigger_scan(merged, analysis, options):
        try:
            scan_results = scan_recall(query, analysis, options)
            channel_results.append(("scan", scan_results))
            if isinstance(channel_filters, dict):
                channel_filters["scan"] = {"language": language}
            trace.setdefault("channels", {})["scan"] = _serialize_results(scan_results)
            merged = merge_recall_results(channel_results, options, diagnostics)
            trace["scan_triggered"] = True
        except Exception as exc:
            _add_warning(diagnostics, f"scan fallback failed: {exc}")
    else:
        trace["scan_triggered"] = False

    serialized_candidates = serialize_candidates(merged)
    trace["recall_candidates"] = serialized_candidates
    root_trace = getattr(diagnostics, "trace", None)
    if isinstance(root_trace, dict):
        root_trace["recall_candidates"] = serialized_candidates
    _summary(diagnostics)["recall_channels"] = sorted(
        {channel for candidate in merged for channel in candidate.recall_channels}
    )
    return merged


def vector_recall(
    query: str,
    analysis: QueryAnalysis,
    options: RetrievalOptions,
) -> list[SearchResult]:
    """Recall candidates through vector search with a safe unscoped path."""
    service = _require_vector_search_service()
    fetch_k = _recall_fetch_limit(options)
    intent = _primary_intent(analysis)
    query_terms = _query_terms(analysis)
    language = _analysis_language(analysis, query)
    results: list[SearchResult] = []

    for route in _recall_routes(intent, analysis):
        retrieval_tiers = effective_recall_tiers(route["tiers"])
        if not retrieval_tiers:
            continue
        chunk_types = _resolve_chunk_types(route["chunk_families"], doc_id=route["doc_id"])
        for variant in _query_variants(query, intent, query_terms)[:3]:
            hits = service.search_similar_documents(
                query=variant,
                top_k=fetch_k,
                doc_id=route["doc_id"],
                language=language,
                retrieval_tiers=retrieval_tiers,
                chunk_types=list(chunk_types) if chunk_types else None,
            )
            active_hits = _filter_results_to_active_docs(hits, scoped_doc_id=route["doc_id"])
            results.extend(filter_recallable_results(active_hits))

    reranked = _rerank_scan_results(
        deduplicate_results(results),
        query=query,
        query_terms=query_terms,
        intent=intent,
        prefer_support=False,
    )
    return reranked[:fetch_k]


def bm25_recall(
    query: str,
    analysis: QueryAnalysis,
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics | None = None,
) -> list[SearchResult]:
    """Recall candidates through a pluggable BM25 provider.

    The project does not currently ship a concrete BM25 index. Until one is
    installed, this channel degrades to an empty result list and records a warning.
    """
    provider = _bm25_provider
    if provider is None:
        _add_warning(diagnostics, "BM25 recall unavailable: no provider configured")
        return []

    fetch_k = _recall_fetch_limit(options)
    intent = _primary_intent(analysis)
    language = _analysis_language(analysis, query)
    results: list[SearchResult] = []

    for route in _recall_routes(intent, analysis):
        retrieval_tiers = effective_recall_tiers(route["tiers"])
        if not retrieval_tiers:
            continue
        chunk_types = _resolve_chunk_types(route["chunk_families"], doc_id=route["doc_id"])
        hits = _search_bm25_provider(
            provider,
            query=query,
            top_k=fetch_k,
            doc_id=route["doc_id"],
            language=language,
            retrieval_tiers=retrieval_tiers,
            chunk_types=list(chunk_types) if chunk_types else None,
        )
        active_hits = _filter_results_to_active_docs(hits, scoped_doc_id=route["doc_id"])
        results.extend(filter_recallable_results(active_hits))

    return deduplicate_results(results)[:fetch_k]


def _search_bm25_provider(
    provider: BM25Provider,
    *,
    query: str,
    top_k: int,
    doc_id: str | None,
    language: str,
    retrieval_tiers: Sequence[str] | None,
    chunk_types: Sequence[str] | None,
) -> list[SearchResult]:
    kwargs: dict[str, Any] = {
        "top_k": top_k,
        "doc_id": doc_id,
        "retrieval_tiers": retrieval_tiers,
        "chunk_types": chunk_types,
    }
    if _call_accepts_keyword(provider.search, "language"):
        kwargs["language"] = language
    return provider.search(query, **kwargs)


def scan_recall(
    query: str,
    analysis: QueryAnalysis,
    options: RetrievalOptions,
) -> list[SearchResult]:
    """Recall candidates through a metadata full scan and local reranking."""
    service = _require_vector_search_service()
    intent = _primary_intent(analysis)
    query_terms = _query_terms(analysis)
    language = _analysis_language(analysis, query)
    limit = max(int(getattr(options, "scan_candidate_limit", 4096) or 0), 1)
    fetch_k = min(_recall_fetch_limit(options), limit)
    results: list[SearchResult] = []

    routes = _scan_routes(intent, analysis)
    per_route_limit = max(limit // max(len(routes), 1), 1)

    for index, route in enumerate(routes):
        retrieval_tiers = effective_recall_tiers(route["tiers"])
        if not retrieval_tiers:
            continue
        remaining_limit = limit - len(results)
        if remaining_limit <= 0:
            break
        route_limit = (
            remaining_limit
            if index == len(routes) - 1
            else min(per_route_limit, remaining_limit)
        )
        chunk_types = _resolve_chunk_types(route["chunk_families"], doc_id=route["doc_id"])
        hits = service.query_all_documents(
            doc_id=route["doc_id"],
            language=language,
            retrieval_tiers=retrieval_tiers,
            chunk_types=list(chunk_types) if chunk_types else None,
            batch_size=route_limit,
        )
        filtered = _filter_results_to_active_docs(hits, scoped_doc_id=route["doc_id"])
        results.extend(filter_recallable_results(filtered)[:route_limit])

    reranked = _rerank_scan_results(
        deduplicate_results(results[:limit]),
        query=query,
        query_terms=query_terms,
        intent=intent,
        prefer_support=True,
    )
    return reranked[:fetch_k]


def should_trigger_scan(
    candidates: Sequence[RecallCandidate],
    analysis: QueryAnalysis,
    options: RetrievalOptions,
) -> bool:
    """Decide whether scan fallback should run."""
    if not getattr(options, "enable_scan_fallback", True):
        return False
    if not candidates:
        return True

    intent = _primary_intent(analysis)
    if intent in {"image_trace", "ocr_audit"}:
        return True
    if _has_critical_term_groups(_analysis_query(analysis)):
        return True

    top_candidate = max(candidates, key=lambda item: item.merged_score)
    query_terms = _query_terms(analysis)
    if query_terms:
        lexical_score = _lexical_score(
            top_candidate.result,
            query=_analysis_query(analysis),
            query_terms=query_terms,
            intent=intent,
        )
        threshold = 24.0 if intent in {"safety", "troubleshooting", "general"} else 20.0
        return lexical_score < threshold

    return bool(top_candidate.merged_score < 0.18)


def merge_recall_results(
    channel_results: Sequence[tuple[str, Sequence[SearchResult]]],
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
) -> list[RecallCandidate]:
    """Merge channel results by chunk id and compute weighted scores."""
    by_chunk_id: dict[str, RecallCandidate] = {}
    weights = {
        "vector": float(getattr(options, "vector_weight", 0.6)),
        "bm25": float(getattr(options, "bm25_weight", 0.4)),
        "scan": SCAN_CHANNEL_WEIGHT,
    }
    trace = _recall_trace(diagnostics)
    trace["pre_merge"] = {
        channel: _serialize_results(results) for channel, results in channel_results
    }
    trace["filtered_pre_merge"] = {}
    trace["recall_filter"] = {}
    trace["normalized_scores"] = {}

    for channel, raw_results in channel_results:
        results = filter_recallable_results(raw_results)
        trace["filtered_pre_merge"][channel] = _serialize_results(results)
        trace["recall_filter"][channel] = {
            "before": len(raw_results),
            "after": len(results),
            "dropped": max(len(raw_results) - len(results), 0),
        }
        normalized_scores = normalize_channel_scores(results, channel)
        trace["normalized_scores"][channel] = list(normalized_scores)
        for result, score in zip(results, normalized_scores, strict=True):
            chunk_id = get_chunk_id(result)
            if not chunk_id:
                continue

            candidate = by_chunk_id.get(chunk_id)
            if candidate is None:
                candidate = RecallCandidate(
                    result=result,
                    chunk_id=chunk_id,
                    recall_channels=set(),
                    channel_scores={},
                )
                by_chunk_id[chunk_id] = candidate

            candidate.recall_channels.add(channel)
            candidate.channel_scores[channel] = max(
                score,
                candidate.channel_scores.get(channel, 0.0),
            )

    for candidate in by_chunk_id.values():
        candidate.merged_score = sum(
            candidate.channel_scores[channel] * weights.get(channel, 0.0)
            for channel in candidate.recall_channels
        )
        candidate.diagnostics.setdefault("channel_weights", dict(weights))

    merged = sorted(
        by_chunk_id.values(),
        key=lambda item: (item.merged_score, sorted(item.recall_channels), item.chunk_id),
        reverse=True,
    )
    trace["post_merge"] = serialize_candidates(merged)
    return merged


def normalize_channel_scores(
    results: Sequence[SearchResult],
    channel: str,
) -> list[float]:
    """Normalize one recall channel to scores in [0, 1]."""
    if not results:
        return []
    if channel == "scan":
        return _rank_scores(len(results))

    raw_scores = [_safe_float(getattr(result, "score", 0.0)) for result in results]
    if channel == "vector":
        channel_scores = raw_scores
    else:
        channel_scores = raw_scores

    minimum = min(channel_scores)
    maximum = max(channel_scores)
    if maximum == minimum:
        return _rank_scores(len(results))
    return [(score - minimum) / (maximum - minimum) for score in channel_scores]


def serialize_candidates(candidates: Sequence[RecallCandidate]) -> list[dict[str, Any]]:
    """Serialize recall candidates for diagnostics trace."""
    return [
        {
            "chunk_id": candidate.chunk_id,
            "merged_score": candidate.merged_score,
            "recall_channels": sorted(candidate.recall_channels),
            "channel_scores": dict(candidate.channel_scores),
            "result": _serialize_result(candidate.result),
        }
        for candidate in candidates
    ]


def get_chunk_id(result: SearchResult) -> str:
    """Return the stable chunk identifier for a search result."""
    metadata = getattr(result, "metadata", None) or {}
    chunk_id = metadata.get("chunk_id") or getattr(result, "id", "")
    return str(chunk_id or "")


def _result_retrieval_tier(result: SearchResult) -> str | None:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        tier = metadata.get("retrieval_tier")
        if tier is not None:
            return str(tier)
    tier = getattr(result, "retrieval_tier", None)
    return str(tier) if tier is not None else None


def deduplicate_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Deduplicate search results by chunk id while preserving first-seen order."""
    deduped: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        chunk_id = get_chunk_id(result)
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(result)
    return deduped


def _recall_routes(intent: str, analysis: QueryAnalysis) -> list[dict[str, Any]]:
    """Build recall routes for all intents × all doc_ids, plus a full-collection safety path."""
    intents = _all_intents(analysis)
    doc_ids = _all_doc_ids(analysis)
    routes: list[dict[str, Any]] = []

    for route_intent in intents:
        base = _route_for_intent(route_intent)
        for doc_id in doc_ids:
            routes.append({**base, "doc_id": doc_id})
        routes.append({**base, "doc_id": None})  # full-collection safety path

    if not routes:
        base = _route_for_intent("general")
        routes.append({**base, "doc_id": None})

    return _dedupe_routes(routes)


def _scan_routes(intent: str, analysis: QueryAnalysis) -> list[dict[str, Any]]:
    return _recall_routes(intent, analysis)


def _route_for_intent(intent: str) -> dict[str, Any]:
    try:
        from app.tools.knowledge_tool import INTENT_STAGE_CONFIG

        stage = INTENT_STAGE_CONFIG.get(intent, INTENT_STAGE_CONFIG["general"])[0]
        return {
            "tiers": tuple(stage.get("tiers") or ()),
            "chunk_families": tuple(stage.get("families") or ()),
        }
    except Exception:
        route_map: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
            "component": (("primary",), ("component",)),
            "procedure": (("primary",), ("procedure",)),
            "legal": (("primary",), ("legal",)),
            "safety": (("primary",), ("safety",)),
            "troubleshooting": (("primary",), ("troubleshooting",)),
            "image_trace": (("auxiliary",), ("auxiliary",)),
            "ocr_audit": (("primary", "support"), ("ocr", "overview")),
            "overview": (("support",), ("overview",)),
            "general": (
                ("primary",),
                ("component", "procedure", "safety", "legal", "troubleshooting"),
            ),
        }
        tiers, families = route_map.get(intent, route_map["general"])
        return {"tiers": tiers, "chunk_families": families}


def _dedupe_routes(routes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, tuple[str, ...], tuple[str, ...]]] = set()
    for route in routes:
        key = (
            route.get("doc_id"),
            tuple(route.get("tiers") or ()),
            tuple(route.get("chunk_families") or ()),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(route)
    return deduped


def _serialize_effective_recall_routes(
    intent: str,
    analysis: QueryAnalysis,
) -> list[dict[str, Any]]:
    routes = _recall_routes(intent, analysis)
    return [
        {
            "doc_id": route.get("doc_id"),
            "route_tiers": _serialize_route_tiers(route.get("tiers")),
            "effective_recall_tiers": effective_recall_tiers(route.get("tiers")),
            "chunk_families": list(route.get("chunk_families") or ()),
        }
        for route in routes
    ]


def _serialize_route_tiers(route_tiers: object) -> list[str] | None:
    if route_tiers is None:
        return None
    if isinstance(route_tiers, str):
        return [route_tiers]
    if isinstance(route_tiers, Sequence):
        return [str(tier) for tier in route_tiers]
    return [str(route_tiers)]


def _all_intents(analysis: QueryAnalysis) -> list[str]:
    """Return all intent labels from analysis, defaulting to ['general']."""
    intents = getattr(analysis, "all_intents", None)
    if intents:
        return list(intents)
    primary = getattr(analysis, "primary_intent", None)
    if primary:
        return [str(primary)]
    return ["general"]


def _all_doc_ids(analysis: QueryAnalysis) -> list[str]:
    """Return all doc_ids from analysis."""
    doc_candidates = getattr(analysis, "doc_candidates", None) or []
    high_confidence_doc_ids: list[str] = []
    for candidate in doc_candidates:
        doc_id = getattr(candidate, "doc_id", None)
        if not doc_id:
            continue
        if _safe_float(getattr(candidate, "score", 0.0)) < DOC_FILTER_CONFIDENCE_THRESHOLD:
            continue
        high_confidence_doc_ids.append(str(doc_id))
    if high_confidence_doc_ids:
        return list(dict.fromkeys(high_confidence_doc_ids))

    if not doc_candidates:
        doc_ids = getattr(analysis, "all_doc_ids", None)
        if doc_ids:
            return list(doc_ids)
        primary = getattr(analysis, "primary_doc_id", None)
        if primary:
            return [str(primary)]
    return []


def _primary_intent(analysis: QueryAnalysis) -> str:
    return str(getattr(analysis, "primary_intent", None) or "general")


def _analysis_query(analysis: QueryAnalysis) -> str:
    return str(getattr(analysis, "query", "") or "")


def _analysis_language(analysis: QueryAnalysis, query: str) -> str:
    for attr_name in ("language", "detected_language", "query_language"):
        language = _normalise_language(getattr(analysis, attr_name, None))
        if language is not None:
            return language

    metadata = getattr(analysis, "metadata", None)
    if isinstance(metadata, dict):
        language = _normalise_language(metadata.get("language"))
        if language is not None:
            return language

    return _detect_language(query or _analysis_query(analysis))


def _detect_language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", str(text or "")) else "en"


def _normalise_language(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-")
    if not normalized:
        return None
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("zh") or normalized in {"cn", "chinese"}:
        return "zh"
    return None


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == keyword
        for name, parameter in signature.parameters.items()
    )


def _query_terms(analysis: QueryAnalysis) -> list[str]:
    terms: list[str] = []
    for item in getattr(analysis, "query_terms", None) or []:
        term = getattr(item, "term", item)
        if term:
            terms.append(str(term))
    return list(dict.fromkeys(terms))


def _query_variants(query: str, intent: str, query_terms: Sequence[str]) -> list[str]:
    try:
        from app.tools.knowledge_tool import expand_queries

        return [
            str(variant)
            for variant in expand_queries(query, intent, query_terms)
            if str(variant or "").strip()
        ]
    except Exception:
        variants = [query.strip()]
        if query_terms:
            variants.append(" ".join(dict.fromkeys(query_terms)))
        return [variant for variant in dict.fromkeys(variants) if variant]


def _resolve_chunk_types(
    chunk_families: Sequence[str] | None,
    doc_id: str | None = None,
) -> list[str] | None:
    try:
        from app.tools.knowledge_tool import resolve_chunk_types

        resolved = resolve_chunk_types(chunk_families, doc_id=doc_id)
        if not resolved:
            return None
        return list(dict.fromkeys(str(item) for item in resolved if item))
    except Exception:
        if not chunk_families:
            return None
        return list(dict.fromkeys(str(item) for item in chunk_families if item))


def _filter_results_to_active_docs(
    results: Sequence[SearchResult],
    scoped_doc_id: str | None,
) -> list[SearchResult]:
    try:
        from app.tools.knowledge_tool import filter_results_to_active_docs

        return list(filter_results_to_active_docs(results, scoped_doc_id=scoped_doc_id))
    except Exception:
        return list(results)


def _rerank_scan_results(
    results: Sequence[SearchResult],
    *,
    query: str,
    query_terms: Sequence[str],
    intent: str,
    prefer_support: bool,
) -> list[SearchResult]:
    try:
        from app.tools.knowledge_tool import post_filter_results, rerank_results

        reranked = rerank_results(
            results,
            original_query=query,
            query_terms=query_terms,
            intent=intent,
            prefer_support=prefer_support,
        )
        return list(post_filter_results(reranked, intent=intent, query_terms=query_terms))
    except Exception:
        scored = [
            (_simple_lexical_score(result, query=query, query_terms=query_terms), result)
            for result in deduplicate_results(results)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in scored]


def _lexical_score(
    result: SearchResult,
    *,
    query: str,
    query_terms: Sequence[str],
    intent: str,
) -> float:
    try:
        from app.tools.knowledge_tool import lexical_score

        return float(
            lexical_score(
                result,
                original_query=query,
                query_terms=query_terms,
                intent=intent,
                prefer_support=False,
            )
        )
    except Exception:
        return _simple_lexical_score(result, query=query, query_terms=query_terms)


def _simple_lexical_score(
    result: SearchResult,
    *,
    query: str,
    query_terms: Sequence[str],
) -> float:
    metadata = getattr(result, "metadata", None) or {}
    haystack = " ".join(
        [
            str(getattr(result, "content", "") or ""),
            str(metadata.get("text") or ""),
            str(metadata.get("title") or metadata.get("section_title") or ""),
            str(metadata.get("index_text") or ""),
            " ".join(str(item) for item in metadata.get("section_path") or []),
        ]
    ).lower()
    terms = list(query_terms) or [query]
    score = 1.0 / (1.0 + max(_safe_float(getattr(result, "score", 0.0)), 0.0))
    for term in terms:
        normalized = str(term or "").strip().lower()
        if normalized and normalized in haystack:
            score += max(4.0, min(len(normalized), 20.0))
    return score


def _has_critical_term_groups(query: str) -> bool:
    try:
        from app.tools.knowledge_tool import normalize_text, query_critical_term_groups

        return bool(query_critical_term_groups(normalize_text(query)))
    except Exception:
        normalized = "".join(query.lower().split())
        markers = ("冷机", "热机", "加油", "燃油混合", "防护装备", "ocr", "缺失")
        return any(marker in normalized for marker in markers)


def _recall_fetch_limit(options: RetrievalOptions) -> int:
    return max(
        int(getattr(options, "reranker_top_n", 32) or 0),
        int(getattr(options, "top_k", 3) or 0) * 4,
        MIN_RECALL_FETCH,
    )


def _rank_scores(count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    return [(count - index) / count for index in range(count)]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _require_vector_search_service() -> Any:
    if vector_search_service is None:
        raise RuntimeError("vector search service unavailable")
    return vector_search_service


def _recall_trace(diagnostics: RetrievalDiagnostics | None) -> dict[str, Any]:
    trace = getattr(diagnostics, "trace", None)
    if not isinstance(trace, dict):
        return {}
    recall_trace = trace.setdefault("recall", {})
    return cast(dict[str, Any], recall_trace) if isinstance(recall_trace, dict) else {}


def _summary(diagnostics: RetrievalDiagnostics | None) -> dict[str, Any]:
    summary = getattr(diagnostics, "summary", None)
    return summary if isinstance(summary, dict) else {}


def _add_warning(diagnostics: RetrievalDiagnostics | None, message: str) -> None:
    if diagnostics is None:
        return
    add_warning = getattr(diagnostics, "add_warning", None)
    if callable(add_warning):
        add_warning(message)
        return
    warnings = getattr(diagnostics, "warnings", None)
    if isinstance(warnings, list):
        warnings.append(message)


def _serialize_results(results: Sequence[SearchResult]) -> list[dict[str, Any]]:
    return [_serialize_result(result) for result in results]


def _serialize_result(result: SearchResult) -> dict[str, Any]:
    raw_metadata = getattr(result, "metadata", None) or {}
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    return {
        "id": getattr(result, "id", None),
        "chunk_id": metadata.get("chunk_id") or getattr(result, "id", None),
        "score": getattr(result, "score", None),
        "doc_id": metadata.get("doc_id"),
        "language": metadata.get("language"),
        "retrieval_tier": metadata.get("retrieval_tier"),
        "chunk_type": metadata.get("chunk_type"),
    }
