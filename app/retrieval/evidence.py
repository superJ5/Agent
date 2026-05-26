"""Evidence organization for the modular retrieval pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from app.retrieval.schemas import RetrievalBundle, chunk_id_from_search_result

if TYPE_CHECKING:
    from app.services.vector_search_service import SearchResult
else:
    SearchResult = Any

vector_search_service: Any
try:  # pragma: no cover - tests can provide a lightweight service stub.
    from app.services.vector_search_service import vector_search_service as _vector_search_service
except Exception:  # pragma: no cover
    vector_search_service = None
else:
    vector_search_service = _vector_search_service

if TYPE_CHECKING:
    from app.retrieval.schemas import (
        QueryAnalysis,
        RecallCandidate,
        RerankResult,
        RetrievalDiagnostics,
        RetrievalOptions,
    )


EVIDENCE_SCHEMA_VERSION = "retrieval_evidence_v1"
DEFAULT_TOP_K = 3
CHANNEL_ORDER = ("vector", "bm25", "scan")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_retrieval_bundle(
    query: str,
    analysis: QueryAnalysis,
    rerank_result: RerankResult,
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
) -> RetrievalBundle:
    """Build the final retrieval bundle from reranked candidates."""
    top_k = safe_top_k(options)
    top_candidates = list((_read_field(rerank_result, "candidates", []) or [])[:top_k])
    primary_hits = [_read_field(candidate, "result") for candidate in top_candidates]

    for warning in _read_field(rerank_result, "warnings", []) or []:
        add_diagnostic_warning(diagnostics, str(warning))

    support_hits = fetch_parent_support_hits(primary_hits, diagnostics)
    intent = _read_field(analysis, "primary_intent") or "general"

    summary = _ensure_summary(diagnostics)
    summary["top_hits"] = summarize_top_hits(
        top_candidates,
        score_field=_read_field(rerank_result, "score_field", "reranker_score"),
    )

    trace = _ensure_trace(diagnostics)
    trace["evidence_primary_hits"] = [
        evidence_hit_summary(result) for result in primary_hits if result is not None
    ]
    trace["evidence_support_hits"] = [
        evidence_hit_summary(result) for result in support_hits if result is not None
    ]
    trace["evidence_images"] = image_trace(primary_hits, support_hits)

    return RetrievalBundle(
        intent=str(intent),
        retrieval_stage="hybrid_search",
        hits=[result for result in primary_hits if result is not None],
        support_hits=support_hits,
        warnings=list(_read_field(diagnostics, "warnings", []) or []),
        metadata={
            "query": query,
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        },
    )


def fetch_parent_support_hits(
    hits: list[SearchResult],
    diagnostics: RetrievalDiagnostics,
) -> list[SearchResult]:
    """Fetch one-hop support parent chunks for primary hits."""
    parent_ids = collect_parent_chunk_ids(hits)
    trace = _ensure_trace(diagnostics)
    trace["support_parent_ids"] = list(parent_ids)
    trace["support_parent_request"] = {
        "retrieval_tiers": ["support"],
        "chunk_ids": list(parent_ids),
        "limit": len(parent_ids),
    }

    if not parent_ids:
        trace["support_parent_hits"] = []
        return []

    if vector_search_service is None:
        message = "support expansion failed: vector search service unavailable"
        add_diagnostic_warning(diagnostics, message)
        trace["support_parent_error"] = message
        return []

    try:
        parents = vector_search_service.query_documents(
            retrieval_tiers=["support"],
            chunk_ids=parent_ids,
            limit=len(parent_ids),
        )
    except Exception as exc:
        add_diagnostic_warning(diagnostics, f"support expansion failed: {exc}")
        trace["support_parent_error"] = str(exc)
        return []

    support_hits = filter_and_order_parent_hits(parents, parent_ids)
    trace["support_parent_hits"] = [evidence_hit_summary(result) for result in support_hits]

    found_parent_ids = {chunk_id_for_result(result) for result in support_hits}
    missing_parent_ids = [
        parent_id for parent_id in parent_ids if parent_id not in found_parent_ids
    ]
    if missing_parent_ids:
        trace["support_parent_missing_ids"] = missing_parent_ids

    return support_hits


def collect_parent_chunk_ids(hits: Sequence[SearchResult]) -> list[str]:
    """Collect unique primary parent chunk ids in first-seen order."""
    parent_ids: list[str] = []
    seen: set[str] = set()

    for hit in hits:
        metadata = result_metadata(hit)
        if metadata.get("retrieval_tier") != "primary":
            continue

        parent_id = _clean_string(metadata.get("parent_chunk_id"))
        if not parent_id or parent_id in seen:
            continue

        seen.add(parent_id)
        parent_ids.append(parent_id)

    return parent_ids


def filter_and_order_parent_hits(
    parents: Sequence[SearchResult],
    parent_ids: Sequence[str],
) -> list[SearchResult]:
    """Keep only requested support parent ids and return them in request order."""
    requested = set(parent_ids)
    parent_by_id: dict[str, SearchResult] = {}

    for parent in parents:
        metadata = result_metadata(parent)
        retrieval_tier = metadata.get("retrieval_tier")
        if retrieval_tier and retrieval_tier != "support":
            continue

        parent_chunk_id = chunk_id_for_result(parent)
        if parent_chunk_id not in requested or parent_chunk_id in parent_by_id:
            continue

        parent_by_id[parent_chunk_id] = parent

    return [parent_by_id[parent_id] for parent_id in parent_ids if parent_id in parent_by_id]


def search_result_to_document(result: SearchResult) -> Document:
    """Convert a SearchResult back into a LangChain Document."""
    metadata = result_metadata(result)
    metadata["score"] = _read_field(result, "score", 0.0)
    metadata["evidence_schema_version"] = EVIDENCE_SCHEMA_VERSION
    return Document(
        page_content=str(_read_field(result, "content", "") or ""),
        metadata=metadata,
    )


def bundle_to_evidence_payload(
    bundle: RetrievalBundle,
    query: str | None = None,
) -> dict[str, Any]:
    """Convert retrieval output into a stable evidence payload for upper layers."""
    primary_hits = [
        search_result_to_evidence_hit(result, rank=index, role="primary")
        for index, result in enumerate(_read_field(bundle, "hits", []) or [], start=1)
    ]
    support_hits = [
        search_result_to_evidence_hit(result, rank=index, role="support")
        for index, result in enumerate(
            _read_field(bundle, "support_hits", []) or [],
            start=1,
        )
    ]

    all_hits = [*primary_hits, *support_hits]
    all_pic_ids = unique_flatten(hit.get("pic_ids") or [] for hit in all_hits)
    all_image_paths = unique_flatten(hit.get("image_paths") or [] for hit in all_hits)
    source_issue_flags = unique_flatten(
        hit.get("source_issue_flags") or [] for hit in all_hits
    )

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "query": query,
        "intent": _read_field(bundle, "intent"),
        "retrieval_stage": _read_field(bundle, "retrieval_stage"),
        "warnings": list(_read_field(bundle, "warnings", []) or []),
        "summary": {
            "primary_count": len(primary_hits),
            "support_count": len(support_hits),
            "pic_ids": all_pic_ids,
            "image_paths": all_image_paths,
            "source_issue_flags": source_issue_flags,
        },
        "hits": primary_hits,
        "support_hits": support_hits,
    }


def search_result_to_evidence_hit(
    result: SearchResult,
    rank: int,
    role: str,
) -> dict[str, Any]:
    """Convert one search result into a compact evidence hit."""
    metadata = result_metadata(result)
    return {
        "rank": rank,
        "role": role,
        "chunk_id": metadata.get("chunk_id") or _read_field(result, "id"),
        "doc_id": metadata.get("doc_id"),
        "doc_name": metadata.get("doc_name"),
        "title": metadata.get("title") or metadata.get("section_title"),
        "section_path": list_value(metadata.get("section_path")),
        "retrieval_tier": metadata.get("retrieval_tier"),
        "chunk_type": metadata.get("chunk_type"),
        "parent_chunk_id": metadata.get("parent_chunk_id"),
        "score": _read_field(result, "score", 0.0),
        "pic_ids": list_value(metadata.get("pic_ids")),
        "image_paths": display_path_list(metadata.get("image_paths")),
        "source_file": metadata.get("source_file") or metadata.get("_source"),
        "source_lines": list_value(metadata.get("source_lines")),
        "source_quality": metadata.get("source_quality"),
        "source_issue_flags": list_value(metadata.get("source_issue_flags")),
        "source_issue_note": metadata.get("source_issue_note") or "",
        "text": _read_field(result, "content", ""),
    }


def format_bundle(bundle: RetrievalBundle, query: str | None = None) -> str:
    """Format retrieval results into a model-readable context block."""
    evidence = bundle_to_evidence_payload(bundle, query=query)
    summary = evidence["summary"]
    sections: list[str] = [
        f"[Evidence schema] {evidence['schema_version']}",
        f"[Retrieval intent] {_read_field(bundle, 'intent')}",
        f"[Retrieval stage] {_read_field(bundle, 'retrieval_stage')}",
        "[Evidence summary] "
        f"primary hits {summary['primary_count']}; "
        f"support context {summary['support_count']}; "
        f"images {len(summary['pic_ids'])}.",
    ]

    warnings = list(_read_field(bundle, "warnings", []) or [])
    if warnings:
        warning_lines = "\n".join(f"- {warning}" for warning in warnings)
        sections.append("[Retrieval warnings]\n" + warning_lines)

    hits = list(_read_field(bundle, "hits", []) or [])
    if hits:
        sections.append("[Primary hits]\n" + format_search_results(hits))

    support_hits = list(_read_field(bundle, "support_hits", []) or [])
    if support_hits:
        sections.append("[Support context]\n" + format_search_results(support_hits))

    return "\n\n".join(sections)


def format_search_results(results: Sequence[SearchResult]) -> str:
    """Format search results into readable context."""
    parts: list[str] = []

    for index, result in enumerate(results, 1):
        metadata = result_metadata(result)
        doc_name = metadata.get("doc_name") or ""
        title = metadata.get("title") or metadata.get("section_title") or ""
        section_path = list_value(metadata.get("section_path"))
        section_path_str = " > ".join(str(part) for part in section_path if part)
        source_lines = list_value(metadata.get("source_lines"))
        pic_ids = list_value(metadata.get("pic_ids"))
        image_paths = display_path_list(metadata.get("image_paths"))
        chunk_id = metadata.get("chunk_id") or _read_field(result, "id")
        retrieval_tier = metadata.get("retrieval_tier") or ""
        chunk_type = metadata.get("chunk_type") or ""
        parent_chunk_id = metadata.get("parent_chunk_id") or ""

        block = [f"[Reference {index}]"]
        if doc_name:
            block.append(f"Manual: {doc_name}")
        if title:
            block.append(f"Title: {title}")
        if section_path_str:
            block.append(f"Section path: {section_path_str}")
        if retrieval_tier or chunk_type:
            block.append(f"Tier: {retrieval_tier} / {chunk_type}")
        if chunk_id:
            block.append(f"Chunk ID: {chunk_id}")
        if parent_chunk_id:
            block.append(f"Parent chunk ID: {parent_chunk_id}")
        if source_lines:
            block.append(f"Source lines: {source_lines}")
        if pic_ids:
            block.append(f"Picture IDs: {pic_ids}")
        image_refs = image_reference_lines(pic_ids, image_paths)
        if image_refs:
            block.append("Related images, cite with this exact markdown:")
            block.extend(image_refs)
        block.append(f"Content:\n{_read_field(result, 'content', '')}")
        parts.append("\n".join(block))

    return "\n\n".join(parts)


def summarize_top_hits(
    candidates: Sequence[RecallCandidate],
    score_field: str = "reranker_score",
) -> list[dict[str, Any]]:
    """Summarize reranked candidates for diagnostics summary metadata."""
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        result = _read_field(candidate, "result")
        summaries.append(
            {
                "chunk_id": _read_field(candidate, "chunk_id")
                or chunk_id_for_result(result),
                "score": candidate_score(candidate, score_field),
                "channels": ordered_channels(
                    _read_field(candidate, "recall_channels", []) or []
                ),
            }
        )
    return summaries


def evidence_hit_summary(result: SearchResult) -> dict[str, Any]:
    """Build a trace-friendly compact result summary."""
    metadata = result_metadata(result)
    return {
        "chunk_id": metadata.get("chunk_id") or _read_field(result, "id"),
        "doc_id": metadata.get("doc_id"),
        "retrieval_tier": metadata.get("retrieval_tier"),
        "chunk_type": metadata.get("chunk_type"),
        "parent_chunk_id": metadata.get("parent_chunk_id"),
        "score": _read_field(result, "score", 0.0),
        "pic_ids": list_value(metadata.get("pic_ids")),
        "image_paths": display_path_list(metadata.get("image_paths")),
    }


def image_trace(
    primary_hits: Sequence[SearchResult],
    support_hits: Sequence[SearchResult],
) -> dict[str, Any]:
    """Return trace details for image metadata carried with evidence hits."""
    hit_summaries: list[dict[str, Any]] = []

    for role, hits in (("primary", primary_hits), ("support", support_hits)):
        for result in hits:
            metadata = result_metadata(result)
            pic_ids = list_value(metadata.get("pic_ids"))
            image_paths = display_path_list(metadata.get("image_paths"))
            if not pic_ids and not image_paths:
                continue
            hit_summaries.append(
                {
                    "role": role,
                    "chunk_id": metadata.get("chunk_id") or _read_field(result, "id"),
                    "pic_ids": pic_ids,
                    "image_paths": image_paths,
                }
            )

    return {
        "pic_ids": unique_flatten(hit["pic_ids"] for hit in hit_summaries),
        "image_paths": unique_flatten(hit["image_paths"] for hit in hit_summaries),
        "hits": hit_summaries,
    }


def candidate_score(candidate: RecallCandidate, score_field: str) -> Any:
    """Pick the best available score for a top-hit summary."""
    diagnostics = _read_field(candidate, "diagnostics", {}) or {}
    if isinstance(diagnostics, Mapping) and score_field in diagnostics:
        return diagnostics[score_field]

    result = _read_field(candidate, "result")
    metadata = result_metadata(result)
    if score_field in metadata:
        return metadata[score_field]

    merged_score = _read_field(candidate, "merged_score")
    if merged_score is not None:
        return merged_score
    return _read_field(result, "score", 0.0)


def safe_top_k(options: RetrievalOptions) -> int:
    """Read top_k defensively while keeping the documented default."""
    try:
        top_k = int(_read_field(options, "top_k", DEFAULT_TOP_K))
    except (TypeError, ValueError):
        return DEFAULT_TOP_K
    return max(top_k, 0)


def add_diagnostic_warning(diagnostics: RetrievalDiagnostics, message: str) -> None:
    """Add a warning through the diagnostics object without duplicating it."""
    if not message:
        return

    warnings = _read_field(diagnostics, "warnings", None)
    if isinstance(warnings, list) and message in warnings:
        return

    add_warning = _read_field(diagnostics, "add_warning")
    if callable(add_warning):
        add_warning(message)
        return

    if isinstance(warnings, list):
        warnings.append(message)


def chunk_id_for_result(result: SearchResult) -> str:
    """Return the stable chunk id for a search result."""
    chunk_id = chunk_id_from_search_result(result)
    return chunk_id or str(_read_field(result, "id", ""))


def result_metadata(result: SearchResult) -> dict[str, Any]:
    """Return a defensive copy of SearchResult metadata."""
    metadata = _read_field(result, "metadata", {}) or {}
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def list_value(value: Any) -> list[Any]:
    """Normalize metadata list fields while preserving scalar values."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def display_path(value: Any) -> str:
    """Render project-local absolute paths as portable relative paths."""
    text = str(value or "").strip()
    if not text:
        return ""

    normalized = text.replace("\\", "/")
    try:
        path = Path(text)
        if not path.is_absolute():
            return normalized
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, RuntimeError, ValueError):
        return normalized


def display_path_list(value: Any) -> list[str]:
    """Normalize path metadata for evidence payloads and model context."""
    paths: list[str] = []
    for item in list_value(value):
        rendered = display_path(item)
        if rendered:
            paths.append(rendered)
    return paths


def image_reference_lines(pic_ids: Sequence[Any], image_paths: Sequence[str]) -> list[str]:
    """Pair picture ids with paths so answers keep non-empty image placeholders."""
    lines: list[str] = []
    pic_id_texts = [str(pic_id).strip() for pic_id in pic_ids if str(pic_id).strip()]

    for index, image_path in enumerate(image_paths):
        pic_id = pic_id_for_image_path(image_path, pic_id_texts, index)
        if pic_id:
            lines.append(f"- ![{pic_id}]({image_path})")
        else:
            lines.append(f"- {image_path}")
    return lines


def pic_id_for_image_path(
    image_path: str,
    pic_ids: Sequence[str],
    index: int,
) -> str:
    """Pick the best picture id for one image path."""
    image_stem = Path(str(image_path)).stem
    for pic_id in pic_ids:
        if pic_id == image_stem or pic_id in image_stem:
            return pic_id

    if index < len(pic_ids):
        return pic_ids[index]
    return image_stem


def unique_flatten(items: Iterable[Iterable[Any] | Any]) -> list[Any]:
    """Flatten nested iterables while preserving first-seen order."""
    flattened: list[Any] = []
    seen: set[str] = set()

    for group in items:
        if group is None:
            continue
        values = [group] if isinstance(group, str | bytes) else group
        try:
            iterator = iter(values)
        except TypeError:
            iterator = iter([values])

        for item in iterator:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            flattened.append(item)

    return flattened


def ordered_channels(channels: Iterable[Any]) -> list[str]:
    """Return channels with known recall channels first."""
    seen: list[str] = []
    for channel in channels:
        channel_text = str(channel)
        if channel_text and channel_text not in seen:
            seen.append(channel_text)

    ordered = [channel for channel in CHANNEL_ORDER if channel in seen]
    ordered.extend(channel for channel in seen if channel not in CHANNEL_ORDER)
    return ordered


def _read_field(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _ensure_summary(diagnostics: Any) -> dict[str, Any]:
    summary = _read_field(diagnostics, "summary", None)
    if isinstance(summary, dict):
        return summary

    new_summary: dict[str, Any] = {}
    if isinstance(diagnostics, Mapping):
        diagnostics["summary"] = new_summary  # type: ignore[index]
    else:
        diagnostics.summary = new_summary
    return new_summary


def _ensure_trace(diagnostics: Any) -> dict[str, Any]:
    trace = _read_field(diagnostics, "trace", None)
    if isinstance(trace, dict):
        return trace

    new_trace: dict[str, Any] = {}
    if isinstance(diagnostics, Mapping):
        diagnostics["trace"] = new_trace  # type: ignore[index]
    else:
        diagnostics.trace = new_trace
    return new_trace


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "RetrievalBundle",
    "build_retrieval_bundle",
    "bundle_to_evidence_payload",
    "fetch_parent_support_hits",
    "format_bundle",
    "format_search_results",
    "search_result_to_document",
    "search_result_to_evidence_hit",
    "unique_flatten",
]
