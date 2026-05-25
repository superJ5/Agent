"""Diagnostics helpers for the retrieval pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol


class _DebugLogger(Protocol):
    def debug(self, *args: Any, **kwargs: Any) -> None: ...


class _NoopLogger:
    def debug(self, *args: Any, **kwargs: Any) -> None:
        return None


logger: _DebugLogger
try:  # pragma: no cover - exercised only when optional dependency is absent.
    from loguru import logger as _loguru_logger
except Exception:  # pragma: no cover
    logger = _NoopLogger()
else:
    logger = _loguru_logger

config: Any
try:  # pragma: no cover - fallback is for isolated module tests.
    from app.config import config
except Exception:  # pragma: no cover
    config = SimpleNamespace(debug=False)

try:  # pragma: no cover - isolated unit tests may load this file without the package.
    from app.retrieval.schemas import (
        QueryAnalysis,
        RecallCandidate,
        RerankResult,
        RetrievalBundle,
        RetrievalDiagnostics,
    )
except Exception:  # pragma: no cover
    if TYPE_CHECKING:
        from app.retrieval.schemas import (
            QueryAnalysis,
            RecallCandidate,
            RerankResult,
            RetrievalBundle,
            RetrievalDiagnostics,
        )
    else:
        QueryAnalysis = RecallCandidate = RerankResult = Any
        RetrievalBundle = RetrievalDiagnostics = Any


TRACE_LOG_PATH = Path("logs/retrieval_trace.jsonl")

_CHANNEL_ORDER = ("vector", "bm25", "scan")
_SCORE_FIELDS = ("reranker_score", "lexical_score", "score")
_SUMMARY_METADATA_KEYS = (
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


def build_summary_metadata(
    diagnostics: RetrievalDiagnostics,
    bundle: RetrievalBundle,
    analysis: QueryAnalysis,
    rerank_result: RerankResult,
) -> dict[str, Any]:
    """Build the Summary-level metadata intended for API responses."""
    warnings = _collect_warnings(diagnostics, bundle, analysis, rerank_result)
    diagnostics_summary = _read_field(diagnostics, "summary", {}) or {}
    candidates = list(_read_field(rerank_result, "candidates", []) or [])

    metadata = {
        "intent": _read_field(analysis, "primary_intent") or _read_field(bundle, "intent"),
        "doc_id": _read_field(analysis, "primary_doc_id"),
        "retrieval_stage": _read_field(bundle, "retrieval_stage"),
        "intent_strategy": _read_field(analysis, "strategy"),
        "recall_channels": collect_recall_channels(candidates),
        "reranker_provider": _read_field(rerank_result, "provider"),
        "reranker_fallback": bool(_read_field(rerank_result, "fallback_used", False)),
        "timeout": bool(diagnostics_summary.get("timeout", False)),
        "degraded": bool(diagnostics_summary.get("degraded", False) or warnings),
        "top_hits": summarize_top_hits(candidates),
        "warnings": warnings,
    }
    return {key: metadata[key] for key in _SUMMARY_METADATA_KEYS}


def summarize_top_hits(
    candidates: Iterable[RecallCandidate],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return a compact, API-safe summary of ranked retrieval hits."""
    summary: list[dict[str, Any]] = []

    for candidate in candidates:
        if limit is not None and len(summary) >= limit:
            break

        item = {
            "chunk_id": _candidate_chunk_id(candidate),
            "score": _candidate_score(candidate),
            "channels": _candidate_channels(candidate),
        }
        reranker_score = _candidate_diagnostics(candidate).get("reranker_score")
        if reranker_score is not None:
            item["reranker_score"] = _coerce_score(reranker_score)
        summary.append(item)

    return summary


def collect_recall_channels(candidates: Iterable[RecallCandidate]) -> list[str]:
    """Collect unique recall channels from candidates in a stable order."""
    seen: list[str] = []

    for candidate in candidates:
        for channel in _candidate_channels(candidate):
            if channel not in seen:
                seen.append(channel)

    return _ordered_channels(seen)


def record_trace_event(
    diagnostics: RetrievalDiagnostics,
    section: str,
    payload: Any,
) -> None:
    """Append a JSON-safe trace event under a named section."""
    trace = _ensure_trace_dict(diagnostics)
    request_id = _read_field(diagnostics, "request_id")
    if request_id:
        trace.setdefault("request_id", request_id)

    events = trace.setdefault(section, [])
    if not isinstance(events, list):
        events = [events]
        trace[section] = events
    events.append(_json_safe(payload))


def write_trace_if_enabled(diagnostics: RetrievalDiagnostics) -> None:
    """Log trace data, or append it to a JSONL file when debug mode is enabled."""
    payload = _trace_payload(diagnostics)

    if not bool(getattr(config, "debug", False)):
        logger.debug("retrieval trace: {}", payload)
        return

    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_LOG_PATH.open("a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _collect_warnings(*objects: Any) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()

    for item in objects:
        for warning in _read_field(item, "warnings", []) or []:
            warning_text = str(warning)
            if warning_text in seen:
                continue
            seen.add(warning_text)
            warnings.append(warning_text)

    return warnings


def _candidate_chunk_id(candidate: Any) -> str | None:
    chunk_id = _read_field(candidate, "chunk_id")
    if chunk_id:
        return str(chunk_id)

    result = _read_field(candidate, "result")
    metadata = _result_metadata(result)
    metadata_chunk_id = metadata.get("chunk_id")
    if metadata_chunk_id:
        return str(metadata_chunk_id)

    result_id = _read_field(result, "id")
    return str(result_id) if result_id else None


def _candidate_score(candidate: Any) -> float | int | str | None:
    diagnostics = _candidate_diagnostics(candidate)
    if diagnostics:
        for field_name in _SCORE_FIELDS:
            if diagnostics.get(field_name) is not None:
                return _coerce_score(diagnostics[field_name])

    merged_score = _read_field(candidate, "merged_score")
    if merged_score is not None:
        return _coerce_score(merged_score)

    result = _read_field(candidate, "result")
    result_score = _read_field(result, "score")
    return _coerce_score(result_score)


def _candidate_diagnostics(candidate: Any) -> dict[str, Any]:
    diagnostics = _read_field(candidate, "diagnostics", {}) or {}
    return diagnostics if isinstance(diagnostics, dict) else {}


def _candidate_channels(candidate: Any) -> list[str]:
    channels = _read_field(candidate, "recall_channels", []) or []
    if isinstance(channels, str):
        return [channels]
    return _ordered_channels(str(channel) for channel in channels if channel)


def _ordered_channels(channels: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for channel in channels:
        if channel not in seen:
            seen.append(channel)

    ordered = [channel for channel in _CHANNEL_ORDER if channel in seen]
    ordered.extend(channel for channel in seen if channel not in _CHANNEL_ORDER)
    return ordered


def _result_metadata(result: Any) -> dict[str, Any]:
    metadata = _read_field(result, "metadata", {}) or {}
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _coerce_score(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return str(value)


def _ensure_trace_dict(diagnostics: Any) -> dict[str, Any]:
    trace = _read_field(diagnostics, "trace", None)
    if isinstance(trace, dict):
        return trace

    new_trace: dict[str, Any] = {}
    if isinstance(diagnostics, dict):
        diagnostics["trace"] = new_trace
    else:
        diagnostics.trace = new_trace
    return new_trace


def _trace_payload(diagnostics: Any) -> dict[str, Any]:
    safe_trace = _json_safe(_ensure_trace_dict(diagnostics))
    trace: dict[str, Any] = safe_trace if isinstance(safe_trace, dict) else {"events": safe_trace}

    request_id = _read_field(diagnostics, "request_id")
    if request_id:
        trace.setdefault("request_id", request_id)

    warnings = _collect_warnings(diagnostics)
    if warnings:
        trace.setdefault("warnings", warnings)

    return trace


def _read_field(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if is_dataclass(value):
        return _json_safe(vars(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    if hasattr(value, "__dict__"):
        return {
            key: _json_safe(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


__all__ = [
    "TRACE_LOG_PATH",
    "RetrievalDiagnostics",
    "build_summary_metadata",
    "collect_recall_channels",
    "record_trace_event",
    "summarize_top_hits",
    "write_trace_if_enabled",
]
