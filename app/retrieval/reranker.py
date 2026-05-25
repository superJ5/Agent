"""Reranking stage for the modular retrieval pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Protocol, TypeVar, cast

import httpx

from app.retrieval.schemas import (
    RecallCandidate,
    RerankResult,
    RetrievalDiagnostics,
    RetrievalOptions,
)

DEFAULT_PROVIDER = "none"
LEXICAL_PROVIDER = "lexical"
DASHSCOPE_PROVIDER = "dashscope"
DEFAULT_DASHSCOPE_MODEL = "qwen3-rerank"
DEFAULT_DASHSCOPE_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
RESERVED_PROVIDERS = {"local", "custom"}

T = TypeVar("T")


class RerankerUnavailableError(RuntimeError):
    """Raised when a configured reranker provider has no usable implementation."""


class RerankerTimeoutError(TimeoutError):
    """Raised when a reranker call exceeds the configured timeout."""


class BaseReranker(Protocol):
    """Minimal contract for pluggable reranker implementations."""

    provider: str

    def rerank(
        self,
        query: str,
        candidates: list[RecallCandidate],
        top_n: int,
    ) -> RerankResult:
        """Return candidates ordered by reranker relevance."""


@dataclass
class ReservedReranker:
    """Placeholder for provider-specific implementations wired in later."""

    provider: str

    def rerank(
        self,
        query: str,
        candidates: list[RecallCandidate],
        top_n: int,
    ) -> RerankResult:
        raise RerankerUnavailableError(
            f"{self.provider} reranker is not configured in this runtime"
        )


@dataclass
class DashScopeReranker:
    """DashScope OpenAI-compatible reranker implementation."""

    model: str = DEFAULT_DASHSCOPE_MODEL
    endpoint: str = DEFAULT_DASHSCOPE_ENDPOINT
    api_key: str = ""
    timeout_ms: int = 3000
    provider: str = DASHSCOPE_PROVIDER

    def rerank(
        self,
        query: str,
        candidates: list[RecallCandidate],
        top_n: int,
    ) -> RerankResult:
        """Call DashScope rerank and return candidates in model-ranked order."""
        if not self.api_key:
            raise RerankerUnavailableError("dashscope api key is missing")

        limited_candidates = list(candidates or [])[: _positive_int(top_n, 32)]
        if not limited_candidates:
            return RerankResult(
                candidates=[],
                provider=self.provider,
                fallback_used=False,
            )

        payload = {
            "model": self.model,
            "query": query,
            "documents": [_candidate_rerank_text(candidate) for candidate in limited_candidates],
            "top_n": min(_positive_int(top_n, len(limited_candidates)), len(limited_candidates)),
        }
        response_payload = _post_dashscope_rerank(
            endpoint=self.endpoint,
            api_key=self.api_key,
            payload=payload,
            timeout_ms=self.timeout_ms,
        )
        return RerankResult(
            candidates=_rank_from_dashscope_response(response_payload, limited_candidates),
            provider=self.provider,
            fallback_used=False,
            score_field="reranker_score",
        )


def create_reranker(
    provider: str | None,
    *,
    model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout_ms: int | None = None,
) -> BaseReranker:
    """Create a reranker instance for a named provider."""
    provider_name = normalize_provider(provider)
    if provider_name == DASHSCOPE_PROVIDER:
        return DashScopeReranker(
            model=model or str(_config_value("rag_reranker_model", DEFAULT_DASHSCOPE_MODEL)),
            endpoint=endpoint
            or str(_config_value("rag_reranker_endpoint", DEFAULT_DASHSCOPE_ENDPOINT)),
            api_key=str(
                api_key if api_key is not None else _config_value("dashscope_api_key", "")
            ),
            timeout_ms=_non_negative_int(
                timeout_ms
                if timeout_ms is not None
                else _config_value("rag_reranker_timeout_ms", 3000),
                3000,
            ),
        )
    if provider_name in RESERVED_PROVIDERS:
        return ReservedReranker(provider_name)
    raise RerankerUnavailableError(f"unsupported reranker provider: {provider_name}")


def rerank_candidates(
    query: str,
    candidates: list[RecallCandidate],
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
) -> RerankResult:
    """Run the configured reranker and degrade to lexical fallback when needed."""
    provider = normalize_provider(getattr(options, "reranker_provider", DEFAULT_PROVIDER))
    candidate_list = list(candidates or [])
    trace = _reranker_trace(diagnostics)
    trace["query"] = query
    trace["provider"] = provider
    trace["pre_rank"] = _rank_snapshot(candidate_list)
    trace["model"] = getattr(options, "reranker_model", DEFAULT_DASHSCOPE_MODEL)
    trace["endpoint"] = getattr(options, "reranker_endpoint", DEFAULT_DASHSCOPE_ENDPOINT)
    trace["top_n"] = _positive_int(getattr(options, "reranker_top_n", 32), 32)
    trace["timeout_ms"] = _non_negative_int(getattr(options, "reranker_timeout_ms", 3000), 3000)
    trace["input_count"] = min(len(candidate_list), int(trace["top_n"]))

    if not candidate_list:
        result = RerankResult(candidates=[], provider=provider, fallback_used=False)
        trace["post_rank"] = []
        trace["rank_changes"] = []
        _update_summary(diagnostics, provider=provider, fallback_used=False, timeout=False)
        return result

    if provider in {DEFAULT_PROVIDER, LEXICAL_PROVIDER, ""}:
        return lexical_fallback(query, candidate_list, diagnostics)

    try:
        reranker = _create_reranker_for_options(provider, options)
        raw_result = run_with_timeout(
            lambda: reranker.rerank(
                query,
                candidate_list,
                _positive_int(getattr(options, "reranker_top_n", 32), 32),
            ),
            timeout_ms=_non_negative_int(
                getattr(options, "reranker_timeout_ms", 3000),
                3000,
            ),
        )
        result = _coerce_rerank_result(raw_result, provider)
        if not result.candidates:
            message = "reranker returned empty result"
            _add_warning(diagnostics, message)
            return lexical_fallback(
                query,
                candidate_list,
                diagnostics,
                warnings=[message],
            )

        result.candidates = _append_missing_candidates(result.candidates, candidate_list)
        result.provider = result.provider or provider
        result.fallback_used = bool(result.fallback_used)
        result.score_field = result.score_field or "reranker_score"

        trace["fallback_used"] = result.fallback_used
        trace["provider"] = result.provider
        trace["post_rank"] = _rank_snapshot(result.candidates)
        trace["rank_changes"] = _rank_changes(candidate_list, result.candidates)
        _update_summary(
            diagnostics,
            provider=result.provider,
            fallback_used=result.fallback_used,
            timeout=False,
        )
        return result
    except RerankerTimeoutError as exc:
        message = str(exc)
        _add_warning(diagnostics, message)
        _summary(diagnostics)["timeout"] = True
        trace["timeout"] = True
        trace["error"] = message
        return lexical_fallback(
            query,
            candidate_list,
            diagnostics,
            warnings=[message],
        )
    except Exception as exc:
        message = f"reranker failed: {exc}"
        _add_warning(diagnostics, message)
        trace["error"] = str(exc)
        return lexical_fallback(
            query,
            candidate_list,
            diagnostics,
            warnings=[message],
        )


def lexical_fallback(
    query: str,
    candidates: list[RecallCandidate],
    diagnostics: RetrievalDiagnostics,
    warnings: Sequence[str] | None = None,
) -> RerankResult:
    """Rank candidates with the legacy lexical scoring heuristics."""
    candidate_list = list(candidates or [])
    trace = _reranker_trace(diagnostics)
    trace.setdefault("pre_rank", _rank_snapshot(candidate_list))
    trace["fallback_used"] = True

    query_terms = _extract_query_terms(query)
    intent = _detect_intent(query)
    scored: list[tuple[float, int, RecallCandidate]] = []

    for index, candidate in enumerate(candidate_list):
        score = _score_candidate(
            candidate.result,
            query=query,
            query_terms=query_terms,
            intent=intent,
        )
        candidate.diagnostics["lexical_score"] = score
        scored.append((score, index, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [candidate for _, _, candidate in scored]

    trace["provider"] = LEXICAL_PROVIDER
    trace["fallback_reason"] = list(warnings or [])
    trace["query_terms"] = list(query_terms)
    trace["intent"] = intent
    trace["post_rank"] = _rank_snapshot(ranked)
    trace["rank_changes"] = _rank_changes(candidate_list, ranked)

    _update_summary(
        diagnostics,
        provider=LEXICAL_PROVIDER,
        fallback_used=True,
        timeout=bool(_summary(diagnostics).get("timeout", False)),
    )
    return RerankResult(
        candidates=ranked,
        provider=LEXICAL_PROVIDER,
        fallback_used=True,
        warnings=list(warnings or []),
        score_field="lexical_score",
    )


def run_with_timeout(operation: Callable[[], T], timeout_ms: int) -> T:
    """Run a synchronous operation with a best-effort daemon-thread timeout."""
    timeout_seconds = max(timeout_ms, 0) / 1000.0
    if timeout_seconds <= 0:
        return operation()

    results: Queue[tuple[str, Any]] = Queue(maxsize=1)

    def target() -> None:
        try:
            results.put(("result", operation()))
        except BaseException as exc:  # noqa: BLE001 - propagate provider failures.
            results.put(("error", exc))

    worker = Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)

    if worker.is_alive():
        raise RerankerTimeoutError(f"reranker timed out after {timeout_ms} ms")

    kind, payload = results.get()
    if kind == "error":
        raise payload
    return cast(T, payload)


def normalize_provider(provider: str | None) -> str:
    """Normalize provider names from config values."""
    provider_name = str(provider or DEFAULT_PROVIDER).strip().lower()
    return provider_name or DEFAULT_PROVIDER


def _create_reranker_for_options(provider: str, options: RetrievalOptions) -> BaseReranker:
    try:
        return create_reranker(
            provider,
            model=str(getattr(options, "reranker_model", DEFAULT_DASHSCOPE_MODEL)),
            endpoint=str(getattr(options, "reranker_endpoint", DEFAULT_DASHSCOPE_ENDPOINT)),
            api_key=str(_config_value("dashscope_api_key", "")),
            timeout_ms=_non_negative_int(getattr(options, "reranker_timeout_ms", 3000), 3000),
        )
    except TypeError:
        return create_reranker(provider)


def _candidate_rerank_text(candidate: RecallCandidate) -> str:
    result = getattr(candidate, "result", None)
    metadata = getattr(result, "metadata", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    for value in (
        metadata.get("index_text"),
        getattr(result, "content", None),
        metadata.get("text"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _post_dashscope_rerank(
    *,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_ms: int,
) -> dict[str, Any]:
    timeout = None if timeout_ms <= 0 else max(timeout_ms, 1) / 1000.0
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(endpoint, headers=headers, json=payload)

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise RerankerUnavailableError("dashscope rerank returned non-json response") from exc

    if not isinstance(response_payload, dict):
        raise RerankerUnavailableError("dashscope rerank returned invalid response")

    if response.status_code >= 400:
        error = response_payload.get("message") or response_payload.get("code") or response.text
        raise RerankerUnavailableError(f"dashscope rerank failed: {error}")

    if response_payload.get("code"):
        error = response_payload.get("message") or response_payload.get("code")
        raise RerankerUnavailableError(f"dashscope rerank failed: {error}")

    return response_payload


def _rank_from_dashscope_response(
    response_payload: dict[str, Any],
    candidates: Sequence[RecallCandidate],
) -> list[RecallCandidate]:
    results = _dashscope_results(response_payload)
    if not results:
        raise RerankerUnavailableError("dashscope rerank returned empty results")

    ranked_entries: list[tuple[int, float | None, RecallCandidate]] = []
    seen_indexes: set[int] = set()
    for response_rank, item in enumerate(results, start=1):
        index = _dashscope_result_index(item)
        if index is None or index < 0 or index >= len(candidates):
            raise RerankerUnavailableError("dashscope rerank returned invalid index")
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        candidate = candidates[index]
        raw_score = _dashscope_result_score(item)
        score = None
        if raw_score is None:
            candidate.diagnostics["reranker_score_missing"] = True
        else:
            score = _safe_float(raw_score)
            candidate.diagnostics["reranker_score"] = score
        candidate.diagnostics["reranker_rank"] = response_rank
        ranked_entries.append((response_rank, score, candidate))

    if not ranked_entries:
        raise RerankerUnavailableError("dashscope rerank returned no usable results")

    if any(score is not None for _, score, _ in ranked_entries):
        ranked_entries.sort(
            key=lambda entry: (
                entry[1] is None,
                -entry[1] if entry[1] is not None else 0.0,
                entry[0],
            )
        )
    return [candidate for _, _, candidate in ranked_entries]


def _dashscope_results(response_payload: dict[str, Any]) -> list[Any]:
    output = response_payload.get("output")
    if isinstance(output, dict) and isinstance(output.get("results"), list):
        return list(output["results"])
    if isinstance(response_payload.get("results"), list):
        return list(response_payload["results"])
    data = response_payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return list(data["results"])
    return []


def _dashscope_result_index(item: Any) -> int | None:
    value = _item_value(item, "index")
    if value is None:
        value = _item_value(item, "document_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dashscope_result_score(item: Any) -> Any:
    for field_name in ("relevance_score", "score", "reranker_score"):
        value = _item_value(item, field_name)
        if value is not None:
            return value
    return None


def _item_value(item: Any, field_name: str) -> Any:
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)


def _config_value(name: str, default: Any) -> Any:
    try:
        from app.config import config

        return getattr(config, name, default)
    except Exception:
        return default


def _coerce_rerank_result(raw_result: Any, provider: str) -> RerankResult:
    if isinstance(raw_result, RerankResult):
        return raw_result
    if isinstance(raw_result, list):
        return RerankResult(
            candidates=list(raw_result),
            provider=provider,
            fallback_used=False,
        )
    if raw_result is None:
        return RerankResult(candidates=[], provider=provider, fallback_used=False)
    candidates = getattr(raw_result, "candidates", None)
    if candidates is not None:
        return RerankResult(
            candidates=list(candidates),
            provider=str(getattr(raw_result, "provider", provider) or provider),
            fallback_used=bool(getattr(raw_result, "fallback_used", False)),
            warnings=list(getattr(raw_result, "warnings", []) or []),
            score_field=str(getattr(raw_result, "score_field", "reranker_score")),
        )
    raise TypeError(f"invalid reranker result: {type(raw_result).__name__}")


def _append_missing_candidates(
    ranked: Sequence[RecallCandidate],
    original: Sequence[RecallCandidate],
) -> list[RecallCandidate]:
    completed: list[RecallCandidate] = list(ranked)
    seen = {_candidate_key(candidate) for candidate in completed}
    for candidate in original:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        completed.append(candidate)
    return completed


def _score_candidate(
    result: Any,
    *,
    query: str,
    query_terms: Sequence[str],
    intent: str,
) -> float:
    try:
        from app.tools.knowledge_tool import lexical_score

        return _safe_float(
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


def _extract_query_terms(query: str) -> list[str]:
    try:
        from app.tools.knowledge_tool import extract_query_terms

        return list(extract_query_terms(query))
    except Exception:
        return _simple_query_terms(query)


def _detect_intent(query: str) -> str:
    try:
        from app.tools.knowledge_tool import detect_intent

        return str(detect_intent(query) or "general")
    except Exception:
        return "general"


def _simple_lexical_score(
    result: Any,
    *,
    query: str,
    query_terms: Sequence[str],
) -> float:
    metadata = getattr(result, "metadata", None) or {}
    text_parts = [
        str(getattr(result, "content", "") or ""),
        str(metadata.get("text") or ""),
        str(metadata.get("title") or metadata.get("section_title") or ""),
        str(metadata.get("index_text") or ""),
    ]
    section_path = metadata.get("section_path") or []
    if isinstance(section_path, list | tuple):
        text_parts.extend(str(part) for part in section_path)
    else:
        text_parts.append(str(section_path))

    haystack = " ".join(text_parts).lower()
    terms = list(query_terms) or _simple_query_terms(query)
    score = (1.0 / (1.0 + max(_safe_float(getattr(result, "score", 0.0)), 0.0))) * 10.0
    for term in terms:
        normalized = str(term or "").strip().lower()
        if not normalized:
            continue
        if normalized in haystack:
            score += max(4.0, min(float(len(normalized)), 24.0))
    return score


def _simple_query_terms(query: str) -> list[str]:
    return [term for term in str(query or "").replace("_", " ").split() if term]


def _rank_snapshot(candidates: Sequence[RecallCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "chunk_id": getattr(candidate, "chunk_id", "") or _result_chunk_id(candidate.result),
            "merged_score": getattr(candidate, "merged_score", None),
            "reranker_score": (getattr(candidate, "diagnostics", {}) or {}).get(
                "reranker_score"
            ),
            "lexical_score": (getattr(candidate, "diagnostics", {}) or {}).get(
                "lexical_score"
            ),
        }
        for index, candidate in enumerate(candidates, start=1)
    ]


def _rank_changes(
    before: Sequence[RecallCandidate],
    after: Sequence[RecallCandidate],
) -> list[dict[str, Any]]:
    before_ranks = {
        _candidate_key(candidate): index for index, candidate in enumerate(before, start=1)
    }
    changes: list[dict[str, Any]] = []
    for after_rank, candidate in enumerate(after, start=1):
        key = _candidate_key(candidate)
        before_rank = before_ranks.get(key)
        changes.append(
            {
                "chunk_id": getattr(candidate, "chunk_id", "")
                or _result_chunk_id(candidate.result),
                "before_rank": before_rank,
                "after_rank": after_rank,
                "delta": None if before_rank is None else before_rank - after_rank,
            }
        )
    return changes


def _candidate_key(candidate: RecallCandidate) -> str:
    chunk_id = getattr(candidate, "chunk_id", "") or _result_chunk_id(candidate.result)
    return str(chunk_id or id(candidate))


def _result_chunk_id(result: Any) -> str:
    metadata = getattr(result, "metadata", None) or {}
    return str(metadata.get("chunk_id") or getattr(result, "id", "") or "")


def _reranker_trace(diagnostics: RetrievalDiagnostics | None) -> dict[str, Any]:
    trace = getattr(diagnostics, "trace", None)
    if isinstance(trace, dict):
        reranker_trace = trace.setdefault("reranker", {})
        return cast(dict[str, Any], reranker_trace) if isinstance(reranker_trace, dict) else {}
    return {}


def _summary(diagnostics: RetrievalDiagnostics | None) -> dict[str, Any]:
    summary = getattr(diagnostics, "summary", None)
    if isinstance(summary, dict):
        return summary
    return {}


def _update_summary(
    diagnostics: RetrievalDiagnostics | None,
    *,
    provider: str,
    fallback_used: bool,
    timeout: bool,
) -> None:
    summary = _summary(diagnostics)
    summary["reranker_provider"] = provider
    summary["reranker_fallback"] = fallback_used
    if timeout:
        summary["timeout"] = True


def _add_warning(diagnostics: RetrievalDiagnostics | None, message: str) -> None:
    if diagnostics is None or not message:
        return
    add_warning = getattr(diagnostics, "add_warning", None)
    if callable(add_warning):
        add_warning(message)
        return
    warnings = getattr(diagnostics, "warnings", None)
    if isinstance(warnings, list):
        warnings.append(message)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "BaseReranker",
    "DashScopeReranker",
    "ReservedReranker",
    "RerankerTimeoutError",
    "RerankerUnavailableError",
    "create_reranker",
    "lexical_fallback",
    "normalize_provider",
    "rerank_candidates",
    "run_with_timeout",
]
