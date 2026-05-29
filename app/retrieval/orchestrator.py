"""Orchestrate the modular retrieval pipeline."""

from __future__ import annotations

import re
from dataclasses import is_dataclass
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from app.retrieval.diagnostics import build_summary_metadata, write_trace_if_enabled
from app.retrieval.evidence import build_retrieval_bundle
from app.retrieval.query_understanding import analyze_query
from app.retrieval.recall import recall_candidates
from app.retrieval.reranker import lexical_fallback, rerank_candidates
from app.retrieval.schemas import (
    QueryAnalysis,
    RecallCandidate,
    RerankResult,
    RetrievalBundle,
    RetrievalDiagnostics,
    RetrievalOptions,
)

config: Any
try:  # pragma: no cover - keeps isolated tests importable without app settings.
    from app.config import config
except Exception:  # pragma: no cover
    config = SimpleNamespace()


def retrieve(query: str, options: RetrievalOptions | None = None) -> RetrievalBundle:
    """Run query understanding, recall, reranking, and evidence organization."""
    resolved_options = options or load_retrieval_options_from_config()
    diagnostics = RetrievalDiagnostics(request_id=new_request_id())
    _initialize_trace(diagnostics, query, resolved_options)

    analysis = _run_query_understanding(query, resolved_options, diagnostics)
    candidates = _run_recall(query, analysis, resolved_options, diagnostics)
    rerank_result = _run_reranker(query, candidates, resolved_options, diagnostics)
    bundle = _run_evidence(query, analysis, rerank_result, resolved_options, diagnostics)

    _finalize_metadata(bundle, diagnostics, analysis, rerank_result)
    _write_trace(diagnostics)
    return bundle


def load_retrieval_options_from_config() -> RetrievalOptions:
    """Create retrieval options from the process config object."""
    return RetrievalOptions.from_config(config)


def new_request_id() -> str:
    """Return a trace-friendly request id for one retrieval call."""
    return f"retrieval-{uuid4().hex}"


def _run_query_understanding(
    query: str,
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
) -> QueryAnalysis:
    try:
        analysis = analyze_query(query, options, diagnostics)
        _record_language(diagnostics, analysis, query)
        return analysis
    except Exception as exc:
        message = f"query understanding failed: {exc}"
        diagnostics.add_warning(message)
        diagnostics.trace.setdefault("query_understanding", {})["error"] = str(exc)
        language = _detect_language(query)
        diagnostics.trace.setdefault("query_understanding", {})["language"] = language
        return QueryAnalysis(
            query=query,
            strategy="none",
            intent_candidates=[],
            doc_candidates=[],
            query_terms=[],
            warnings=["query understanding degraded to none"],
            language=language,
        )


def _run_recall(
    query: str,
    analysis: QueryAnalysis,
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
) -> list[RecallCandidate]:
    try:
        return list(recall_candidates(query, analysis, options, diagnostics) or [])
    except Exception as exc:
        message = f"recall failed: {exc}"
        diagnostics.add_warning(message)
        diagnostics.trace.setdefault("recall", {})["error"] = str(exc)
        diagnostics.trace["recall_candidates"] = []
        return []


def _run_reranker(
    query: str,
    candidates: list[RecallCandidate],
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
) -> RerankResult:
    try:
        return rerank_candidates(query, candidates, options, diagnostics)
    except Exception as exc:
        message = f"reranker failed: {exc}"
        diagnostics.add_warning(message)
        diagnostics.trace.setdefault("reranker", {})["error"] = str(exc)

    try:
        return lexical_fallback(query, candidates, diagnostics, warnings=[message])
    except Exception as fallback_exc:
        fallback_message = f"reranker lexical fallback failed: {fallback_exc}"
        diagnostics.add_warning(fallback_message)
        diagnostics.trace.setdefault("reranker", {})["fallback_error"] = str(fallback_exc)
        diagnostics.summary["reranker_provider"] = "lexical"
        diagnostics.summary["reranker_fallback"] = True
        return RerankResult(
            candidates=list(candidates),
            provider="lexical",
            fallback_used=True,
            warnings=[message, fallback_message],
            score_field="lexical_score",
        )


def _run_evidence(
    query: str,
    analysis: QueryAnalysis,
    rerank_result: RerankResult,
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
) -> RetrievalBundle:
    try:
        return build_retrieval_bundle(query, analysis, rerank_result, options, diagnostics)
    except Exception as exc:
        message = f"evidence organization failed: {exc}"
        diagnostics.add_warning(message)
        diagnostics.trace.setdefault("evidence", {})["error"] = str(exc)
        return _fallback_bundle(query, analysis, rerank_result, options, diagnostics)


def _fallback_bundle(
    query: str,
    analysis: QueryAnalysis,
    rerank_result: RerankResult,
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
) -> RetrievalBundle:
    candidates = list(getattr(rerank_result, "candidates", []) or [])
    top_k = _safe_top_k(options)
    primary_hits = [
        candidate.result
        for candidate in candidates[:top_k]
        if getattr(candidate, "result", None) is not None
    ]
    diagnostics.summary["top_hits"] = [
        {
            "chunk_id": candidate.chunk_id,
            "score": _candidate_score(candidate),
            "channels": sorted(candidate.recall_channels),
        }
        for candidate in candidates[:top_k]
    ]
    return RetrievalBundle(
        intent=str(getattr(analysis, "primary_intent", None) or "general"),
        retrieval_stage="hybrid_search" if primary_hits else "none",
        hits=primary_hits,
        support_hits=[],
        warnings=list(diagnostics.warnings),
        metadata={"query": query},
    )


def _finalize_metadata(
    bundle: RetrievalBundle,
    diagnostics: RetrievalDiagnostics,
    analysis: QueryAnalysis,
    rerank_result: RerankResult,
) -> None:
    bundle.warnings = _dedupe_warnings([*bundle.warnings, *diagnostics.warnings])
    summary_metadata = build_summary_metadata(diagnostics, bundle, analysis, rerank_result)
    diagnostics.summary.update(summary_metadata)
    bundle.metadata.update(summary_metadata)
    bundle.warnings = list(summary_metadata.get("warnings", bundle.warnings))


def _write_trace(diagnostics: RetrievalDiagnostics) -> None:
    try:
        write_trace_if_enabled(diagnostics)
    except Exception as exc:  # pragma: no cover - depends on filesystem/logger runtime.
        diagnostics.add_warning(f"trace write failed: {exc}")


def _initialize_trace(
    diagnostics: RetrievalDiagnostics,
    query: str,
    options: RetrievalOptions,
) -> None:
    diagnostics.trace["request_id"] = diagnostics.request_id
    diagnostics.trace["query"] = query
    diagnostics.trace["language"] = _detect_language(query)
    diagnostics.trace["options"] = _json_safe(options)


def _record_language(
    diagnostics: RetrievalDiagnostics,
    analysis: QueryAnalysis,
    query: str,
) -> None:
    language = _normalize_language(getattr(analysis, "language", None)) or _detect_language(query)
    try:
        analysis.language = language
    except Exception:
        pass
    diagnostics.trace["language"] = language
    diagnostics.trace.setdefault("query_understanding", {})["language"] = language


def _detect_language(query: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", str(query or "")) else "en"


def _normalize_language(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"en", "zh"} else None


def _safe_top_k(options: RetrievalOptions) -> int:
    try:
        return max(int(getattr(options, "top_k", 3) or 0), 0)
    except (TypeError, ValueError):
        return 3


def _candidate_score(candidate: RecallCandidate) -> Any:
    diagnostics = getattr(candidate, "diagnostics", None) or {}
    if isinstance(diagnostics, dict):
        for key in ("reranker_score", "lexical_score"):
            if key in diagnostics:
                return diagnostics[key]
    return getattr(candidate, "merged_score", None)


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        warning_text = str(warning)
        if not warning_text or warning_text in seen:
            continue
        seen.add(warning_text)
        deduped.append(warning_text)
    return deduped


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return dict(vars(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return sorted(value, key=str)
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)


__all__ = [
    "load_retrieval_options_from_config",
    "new_request_id",
    "retrieve",
]
