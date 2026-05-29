"""Shared data contracts for the retrieval pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any


def _mapping_get(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _result_metadata(result: Any) -> dict[str, Any]:
    metadata = _mapping_get(result, "metadata", {})
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def chunk_id_from_search_result(result: Any) -> str:
    """Return the stable chunk key for a SearchResult-like object."""
    metadata = _result_metadata(result)
    chunk_id = _clean_string(metadata.get("chunk_id"))
    if chunk_id:
        return chunk_id
    return _clean_string(_mapping_get(result, "id", ""))


def get_result_chunk_id(result: Any) -> str:
    """Backward-compatible alias for chunk_id_from_search_result."""
    return chunk_id_from_search_result(result)


def get_chunk_id_from_search_result(result: Any) -> str:
    """Backward-compatible alias for chunk_id_from_search_result."""
    return chunk_id_from_search_result(result)


def search_result_chunk_id(result: Any) -> str:
    """Backward-compatible alias for chunk_id_from_search_result."""
    return chunk_id_from_search_result(result)


def derive_chunk_id(result: Any) -> str:
    """Backward-compatible alias for chunk_id_from_search_result."""
    return chunk_id_from_search_result(result)


@dataclass(frozen=True)
class IntentCandidate:
    intent: str
    score: float = 0.0
    source: str = ""
    reason: str = ""


@dataclass(frozen=True)
class DocCandidate:
    doc_id: str
    score: float = 0.0
    source: str = ""
    reason: str = ""


@dataclass(frozen=True)
class QueryTerm:
    term: str
    source: str
    weight: float = 1.0


@dataclass
class QueryAnalysis:
    query: str = ""
    strategy: str = "none"
    intent_candidates: list[IntentCandidate] = field(default_factory=list)
    doc_candidates: list[DocCandidate] = field(default_factory=list)
    query_terms: list[QueryTerm] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    language: str = ""

    @property
    def primary_intent(self) -> str | None:
        if not self.intent_candidates:
            return None
        return max(self.intent_candidates, key=lambda item: item.score).intent or None

    @property
    def primary_doc_id(self) -> str | None:
        if not self.doc_candidates:
            return None
        return max(self.doc_candidates, key=lambda item: item.score).doc_id or None

    @property
    def all_intents(self) -> list[str]:
        """Return deduplicated intent labels from all candidates."""
        seen: set[str] = set()
        result: list[str] = []
        for c in self.intent_candidates:
            if c.intent and c.intent not in seen:
                seen.add(c.intent)
                result.append(c.intent)
        return result

    @property
    def all_doc_ids(self) -> list[str]:
        """Return deduplicated doc_ids from all candidates."""
        seen: set[str] = set()
        result: list[str] = []
        for c in self.doc_candidates:
            if c.doc_id and c.doc_id not in seen:
                seen.add(c.doc_id)
                result.append(c.doc_id)
        return result


@dataclass
class RecallCandidate:
    result: Any
    chunk_id: str = ""
    recall_channels: set[str] = field(default_factory=set)
    channel_scores: dict[str, float] = field(default_factory=dict)
    merged_score: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = chunk_id_from_search_result(self.result)

    @classmethod
    def from_search_result(
        cls,
        result: Any,
        channel: str | None = None,
        score: float | None = None,
        *,
        recall_channel: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> RecallCandidate:
        """Create a recall candidate from a SearchResult-like object."""
        channel_name = recall_channel or channel or ""
        default_score = _read_float(_mapping_get(result, "score", 0.0))
        result_score = _read_float(score, default_score) if score is not None else default_score
        recall_channels = {channel_name} if channel_name else set()
        channel_scores = {channel_name: result_score} if channel_name else {}
        return cls(
            result=result,
            chunk_id=chunk_id_from_search_result(result),
            recall_channels=recall_channels,
            channel_scores=channel_scores,
            merged_score=result_score,
            diagnostics=dict(diagnostics or {}),
        )


@dataclass
class RerankResult:
    candidates: list[RecallCandidate] = field(default_factory=list)
    provider: str = "none"
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)
    score_field: str = "reranker_score"


@dataclass
class RetrievalDiagnostics:
    request_id: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        if not message:
            return
        self.warnings.append(str(message))
        self.summary["warnings"] = list(self.warnings)

    def to_summary_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"request_id": self.request_id}
        metadata.update(self.summary)
        metadata["warnings"] = list(self.warnings)
        metadata.setdefault("degraded", bool(self.warnings))
        return metadata


@dataclass
class RetrievalBundle:
    """Container compatible with the existing routed_retrieve return shape."""

    intent: str = "general"
    retrieval_stage: str = "none"
    hits: list[Any] = field(default_factory=list)
    support_hits: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_hits(self) -> list[Any]:
        combined: list[Any] = []
        seen: set[str] = set()
        for result in [*self.hits, *self.support_hits]:
            chunk_id = chunk_id_from_search_result(result)
            dedupe_key = chunk_id or str(id(result))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            combined.append(result)
        return combined


@dataclass(frozen=True)
class RetrievalOptions:
    intent_strategy: str = "hybrid"
    enable_vector_recall: bool = True
    enable_bm25_recall: bool = True
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    reranker_provider: str = "none"
    reranker_model: str = "qwen3-rerank"
    reranker_endpoint: str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    reranker_timeout_ms: int = 3000
    reranker_top_n: int = 32
    enable_scan_fallback: bool = True
    scan_candidate_limit: int = 4096
    top_k: int = 3

    @classmethod
    def from_config(cls, config: Any) -> RetrievalOptions:
        values: dict[str, Any] = {}
        for option_field in fields(cls):
            name = option_field.name
            default = option_field.default
            raw_value = _config_value(config, f"rag_{name}", name, default)
            values[name] = _coerce_config_value(raw_value, default)
        return cls(**values)


def _config_value(config: Any, preferred_key: str, fallback_key: str, default: Any) -> Any:
    if config is None:
        return default
    sentinel = object()
    value = _mapping_get(config, preferred_key, sentinel)
    if value is sentinel:
        value = _mapping_get(config, fallback_key, sentinel)
    return default if value is sentinel or value is None else value


def _coerce_config_value(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        return _read_bool(value, default)
    if isinstance(default, int) and not isinstance(default, bool):
        return _read_int(value, default)
    if isinstance(default, float):
        return _read_float(value, default)
    if isinstance(default, str):
        return str(value)
    return value


def _read_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _read_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
