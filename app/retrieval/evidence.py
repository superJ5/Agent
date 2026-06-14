"""Evidence organization for the modular retrieval pipeline."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from app.retrieval.reranker import rerank_candidates
from app.retrieval.schemas import (
    RecallCandidate,
    RetrievalBundle,
    RetrievalDiagnostics,
    chunk_id_from_search_result,
)
from app.retrieval.tier_policy import (
    BIG_SUPPORT_TIER,
    DESCENDANT_QUERY_TIERS,
    EVIDENCE_PARENT_TIERS,
    PRIMARY_TIER,
    SUPPORT_TIER,
)

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
        RerankResult,
        RetrievalOptions,
    )


EVIDENCE_SCHEMA_VERSION = "retrieval_evidence_v1"
DEFAULT_TOP_K = 3
CHANNEL_ORDER = ("vector", "bm25", "scan")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIG_SUPPORT_TOP_N = 3
BIG_SUPPORT_MAX_DEPTH = 4
BIG_SUPPORT_MAX_CANDIDATES = 100


def build_retrieval_bundle(
    query: str,
    analysis: QueryAnalysis,
    rerank_result: RerankResult,
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
    *,
    rerank_query: str | None = None,
) -> RetrievalBundle:
    """Build the final retrieval bundle from reranked candidates."""
    effective_rerank_query = str(rerank_query or "").strip() or query
    top_k = safe_top_k(options)
    top_candidates = list((_read_field(rerank_result, "candidates", []) or [])[:top_k])
    primary_hits = [_read_field(candidate, "result") for candidate in top_candidates]

    for warning in _read_field(rerank_result, "warnings", []) or []:
        add_diagnostic_warning(diagnostics, str(warning))

    language = language_filter(
        analysis=analysis,
        diagnostics=diagnostics,
        hits=primary_hits,
        query=query,
    )
    support_hits = fetch_parent_support_hits(
        primary_hits,
        diagnostics,
        language=language,
        query=query,
        rerank_query=effective_rerank_query,
        options=options,
    )
    intent = _read_field(analysis, "primary_intent") or "general"

    summary = _ensure_summary(diagnostics)
    summary["top_hits"] = summarize_top_hits(
        top_candidates,
        score_field=_read_field(rerank_result, "score_field", "reranker_score"),
    )

    trace = _ensure_trace(diagnostics)
    trace["language"] = language
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
    language: str | None = None,
    query: str | None = None,
    rerank_query: str | None = None,
    options: RetrievalOptions | None = None,
    expand_big_support: bool = True,
) -> list[SearchResult]:
    """Fetch supplemental support context for primary hits."""
    language = language_filter(diagnostics=diagnostics, hits=hits, language=language)
    parent_ids = collect_parent_chunk_ids(hits)
    primary_hit_ids = {chunk_id_for_result(hit) for hit in hits if hit is not None}
    trace = _ensure_trace(diagnostics)
    trace["support_parent_ids"] = list(parent_ids)
    trace["support_parent_request"] = {
        "retrieval_tiers": list(EVIDENCE_PARENT_TIERS),
        "chunk_ids": list(parent_ids),
        "language": language,
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
        query_kwargs: dict[str, Any] = {
            "retrieval_tiers": list(EVIDENCE_PARENT_TIERS),
            "chunk_ids": parent_ids,
            "limit": len(parent_ids),
        }
        if language:
            query_kwargs["language"] = language
        parents = vector_search_service.query_documents(**query_kwargs)
    except Exception as exc:
        add_diagnostic_warning(diagnostics, f"support expansion failed: {exc}")
        trace["support_parent_error"] = str(exc)
        return []

    parent_hits = filter_and_order_parent_hits(
        parents,
        parent_ids,
        allowed_tiers=EVIDENCE_PARENT_TIERS,
    )
    trace["support_parent_hits"] = [
        evidence_hit_summary(result) for result in parent_hits
    ]

    support_hits: list[SearchResult] = []
    support_seen: set[str] = set()
    for parent in parent_hits:
        parent_tier = result_metadata(parent).get("retrieval_tier")
        append_support_context(
            support_hits,
            parent,
            primary_hit_ids=primary_hit_ids,
            support_seen=support_seen,
        )
        if parent_tier != BIG_SUPPORT_TIER:
            continue
        if not expand_big_support:
            record_big_support_skip(
                trace,
                chunk_id_for_result(parent),
                reason="expand_big_support_false",
            )
            continue
        if not query or options is None:
            record_big_support_skip(
                trace,
                chunk_id_for_result(parent),
                reason="missing_query_or_options",
            )
            continue
        expand_big_support_parent(
            parent,
            query=query,
            rerank_query=rerank_query,
            options=options,
            diagnostics=diagnostics,
            language=language,
            primary_hit_ids=primary_hit_ids,
            support_hits=support_hits,
            support_seen=support_seen,
        )

    found_parent_ids = {chunk_id_for_result(result) for result in parent_hits}
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
    allowed_tiers: Sequence[str] = (SUPPORT_TIER,),
) -> list[SearchResult]:
    """Keep requested parent ids in request order and restrict retrieval tiers."""
    requested = set(parent_ids)
    allowed = set(allowed_tiers)
    parent_by_id: dict[str, SearchResult] = {}

    for parent in parents:
        metadata = result_metadata(parent)
        retrieval_tier = metadata.get("retrieval_tier")
        if retrieval_tier and retrieval_tier not in allowed:
            continue

        parent_chunk_id = chunk_id_for_result(parent)
        if parent_chunk_id not in requested or parent_chunk_id in parent_by_id:
            continue

        parent_by_id[parent_chunk_id] = parent

    return [parent_by_id[parent_id] for parent_id in parent_ids if parent_id in parent_by_id]


def append_support_context(
    support_hits: list[SearchResult],
    result: SearchResult,
    *,
    primary_hit_ids: set[str],
    support_seen: set[str],
) -> bool:
    """Append supplemental context once, without duplicating primary hits."""
    chunk_id = chunk_id_for_result(result)
    if not chunk_id or chunk_id in primary_hit_ids or chunk_id in support_seen:
        return False

    support_seen.add(chunk_id)
    support_hits.append(result)
    return True


def expand_big_support_parent(
    parent: SearchResult,
    *,
    query: str,
    rerank_query: str | None = None,
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
    language: str | None,
    primary_hit_ids: set[str],
    support_hits: list[SearchResult],
    support_seen: set[str],
) -> None:
    """Expand one big_support parent into its most relevant primary descendants."""
    trace = _ensure_trace(diagnostics)
    parent_id = chunk_id_for_result(parent)
    parent_ids = trace.setdefault("big_support_parent_ids", [])
    if isinstance(parent_ids, list) and parent_id not in parent_ids:
        parent_ids.append(parent_id)

    descendants, truncated = collect_big_support_descendants(
        parent,
        diagnostics=diagnostics,
        language=language,
    )

    descendant_count = trace.setdefault("big_support_descendant_count", {})
    if isinstance(descendant_count, dict):
        descendant_count[parent_id] = len(descendants)

    selected_descendants = select_big_support_descendants(
        query=str(rerank_query or "").strip() or query,
        descendants=descendants,
        options=options,
        diagnostics=diagnostics,
        parent_id=parent_id,
    )

    for descendant in selected_descendants:
        append_support_context(
            support_hits,
            descendant,
            primary_hit_ids=primary_hit_ids,
            support_seen=support_seen,
        )

    direct_support_parents = fetch_direct_support_parents(
        selected_descendants,
        diagnostics=diagnostics,
        language=language,
    )
    for support_parent in direct_support_parents:
        append_support_context(
            support_hits,
            support_parent,
            primary_hit_ids=primary_hit_ids,
            support_seen=support_seen,
        )

    expanded_hits = trace.setdefault("big_support_expanded_hits", [])
    if isinstance(expanded_hits, list):
        expanded_hits.append(
            {
                "parent_id": parent_id,
                "descendant_count": len(descendants),
                "max_candidates": BIG_SUPPORT_MAX_CANDIDATES,
                "truncated": truncated,
                "selected": [
                    evidence_hit_summary(result) for result in selected_descendants
                ],
                "direct_support_parents": [
                    evidence_hit_summary(result) for result in direct_support_parents
                ],
            }
        )


def collect_big_support_descendants(
    parent: SearchResult,
    *,
    diagnostics: RetrievalDiagnostics,
    language: str | None,
) -> tuple[list[SearchResult], bool]:
    """Collect primary descendants under a big_support parent by walking children."""
    parent_id = chunk_id_for_result(parent)
    doc_id = _clean_string(result_metadata(parent).get("doc_id")) or None
    current_parent_ids = [parent_id] if parent_id else []
    visited_parent_ids = set(current_parent_ids)
    descendants: list[SearchResult] = []
    descendant_seen: set[str] = set()
    truncated = False

    for _depth in range(1, BIG_SUPPORT_MAX_DEPTH + 1):
        if not current_parent_ids or len(descendants) >= BIG_SUPPORT_MAX_CANDIDATES:
            break

        children = query_direct_children(
            current_parent_ids,
            diagnostics=diagnostics,
            language=language,
            doc_id=doc_id,
        )
        next_parent_ids: list[str] = []

        for child in sort_results_for_hierarchy(children):
            child_id = chunk_id_for_result(child)
            if not child_id:
                continue
            tier = result_metadata(child).get("retrieval_tier")
            if tier == PRIMARY_TIER:
                if child_id in descendant_seen:
                    continue
                descendant_seen.add(child_id)
                descendants.append(child)
                if len(descendants) >= BIG_SUPPORT_MAX_CANDIDATES:
                    truncated = True
                    break
            elif tier in {SUPPORT_TIER, BIG_SUPPORT_TIER}:
                if child_id in visited_parent_ids:
                    continue
                visited_parent_ids.add(child_id)
                next_parent_ids.append(child_id)
            else:
                continue

        current_parent_ids = next_parent_ids

    return descendants, truncated


def query_direct_children(
    parent_ids: Sequence[str],
    *,
    diagnostics: RetrievalDiagnostics,
    language: str | None,
    doc_id: str | None,
) -> list[SearchResult]:
    """Query direct children for one hierarchy layer."""
    if vector_search_service is None:
        message = "big_support expansion failed: vector search service unavailable"
        add_diagnostic_warning(diagnostics, message)
        _ensure_trace(diagnostics)["big_support_descendant_error"] = message
        return []

    query_kwargs: dict[str, Any] = {
        "doc_id": doc_id,
        "retrieval_tiers": list(DESCENDANT_QUERY_TIERS),
        "parent_chunk_ids": list(parent_ids),
    }
    if language:
        query_kwargs["language"] = language

    query_all_documents = getattr(vector_search_service, "query_all_documents", None)
    try:
        if callable(query_all_documents):
            return list(query_all_documents(**query_kwargs))
        query_kwargs["limit"] = max(BIG_SUPPORT_MAX_CANDIDATES * 4, 256)
        return list(vector_search_service.query_documents(**query_kwargs))
    except Exception as exc:
        add_diagnostic_warning(diagnostics, f"big_support expansion failed: {exc}")
        _ensure_trace(diagnostics)["big_support_descendant_error"] = str(exc)
        return []


def select_big_support_descendants(
    *,
    query: str,
    descendants: Sequence[SearchResult],
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
    parent_id: str,
) -> list[SearchResult]:
    """Rerank big_support primary descendants and return the top few."""
    if not descendants:
        return []

    candidates = [
        RecallCandidate.from_search_result(
            result,
            recall_channel="big_support_descendant",
        )
        for result in descendants
    ]
    child_diagnostics = RetrievalDiagnostics(
        request_id=str(_read_field(diagnostics, "request_id", "") or "")
    )
    try:
        reranked = rerank_candidates(query, candidates, options, child_diagnostics)
    except Exception as exc:
        add_diagnostic_warning(
            diagnostics,
            f"big_support rerank failed for {parent_id}: {exc}",
        )
        return [candidate.result for candidate in candidates[:BIG_SUPPORT_TOP_N]]

    for warning in child_diagnostics.warnings:
        add_diagnostic_warning(diagnostics, str(warning))

    trace = _ensure_trace(diagnostics)
    rerank_trace = trace.setdefault("big_support_rerank", {})
    if isinstance(rerank_trace, dict):
        rerank_trace[parent_id] = child_diagnostics.trace.get("reranker", {})

    return [
        candidate.result
        for candidate in list(_read_field(reranked, "candidates", []) or [])[
            :BIG_SUPPORT_TOP_N
        ]
    ]


def fetch_direct_support_parents(
    descendants: Sequence[SearchResult],
    *,
    diagnostics: RetrievalDiagnostics,
    language: str | None,
) -> list[SearchResult]:
    """Fetch direct ordinary support parents for selected descendant primary chunks."""
    parent_ids = collect_parent_chunk_ids(descendants)
    if not parent_ids:
        return []

    if vector_search_service is None:
        message = "selected descendant support expansion failed: vector search service unavailable"
        add_diagnostic_warning(diagnostics, message)
        _ensure_trace(diagnostics)["big_support_selected_parent_error"] = message
        return []

    query_kwargs: dict[str, Any] = {
        "retrieval_tiers": [SUPPORT_TIER],
        "chunk_ids": parent_ids,
        "limit": len(parent_ids),
    }
    if language:
        query_kwargs["language"] = language

    try:
        parents = vector_search_service.query_documents(**query_kwargs)
    except Exception as exc:
        add_diagnostic_warning(
            diagnostics,
            f"selected descendant support expansion failed: {exc}",
        )
        _ensure_trace(diagnostics)["big_support_selected_parent_error"] = str(exc)
        return []

    direct_support_parents = filter_and_order_parent_hits(
        parents,
        parent_ids,
        allowed_tiers=(SUPPORT_TIER,),
    )
    trace = _ensure_trace(diagnostics)
    trace["big_support_selected_parent_ids"] = list(parent_ids)
    trace["big_support_selected_parent_hits"] = [
        evidence_hit_summary(parent) for parent in direct_support_parents
    ]
    return direct_support_parents


def sort_results_for_hierarchy(results: Sequence[SearchResult]) -> list[SearchResult]:
    """Sort children by stable metadata order when it is available."""
    indexed_results = list(enumerate(results))
    indexed_results.sort(key=lambda item: hierarchy_order_key(item[1], item[0]))
    return [result for _, result in indexed_results]


def hierarchy_order_key(result: SearchResult, fallback_index: int) -> tuple[int, int, int, int]:
    metadata = result_metadata(result)
    source_lines = list_value(metadata.get("source_lines"))
    source_start = safe_int(source_lines[0]) if source_lines else None
    chunk_index = safe_int(metadata.get("chunk_index"), default=1_000_000)
    return (
        chunk_index if chunk_index is not None else 1_000_000,
        source_start if source_start is not None else 1_000_000,
        chunk_id_numeric_suffix(chunk_id_for_result(result)),
        fallback_index,
    )


def chunk_id_numeric_suffix(chunk_id: str) -> int:
    match = re.search(r"_(\d+)$", str(chunk_id or ""))
    if not match:
        return 1_000_000
    return int(match.group(1))


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def record_big_support_skip(trace: dict[str, Any], parent_id: str, *, reason: str) -> None:
    skipped = trace.setdefault("big_support_expansion_skipped", [])
    if isinstance(skipped, list):
        skipped.append({"parent_id": parent_id, "reason": reason})


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
        "language": metadata.get("language"),
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


def language_filter(
    *,
    analysis: Any = None,
    diagnostics: Any = None,
    hits: Sequence[SearchResult] | None = None,
    query: str | None = None,
    language: str | None = None,
) -> str | None:
    """Resolve the language filter used by evidence support queries."""
    for candidate in (
        language,
        _read_field(analysis, "language"),
        _trace_language(diagnostics),
        _hits_language(hits or []),
    ):
        normalized = normalize_language(candidate)
        if normalized:
            return normalized
    if query:
        return detect_language(query)
    return None


def _trace_language(diagnostics: Any) -> str | None:
    trace = _read_field(diagnostics, "trace", {}) or {}
    if not isinstance(trace, Mapping):
        return None
    query_trace = trace.get("query_understanding")
    if isinstance(query_trace, Mapping):
        language = normalize_language(query_trace.get("language"))
        if language:
            return language
    return normalize_language(trace.get("language"))


def _hits_language(hits: Sequence[SearchResult]) -> str | None:
    for hit in hits:
        language = normalize_language(result_metadata(hit).get("language"))
        if language:
            return language
    return None


def detect_language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", str(text or "")) else "en"


def normalize_language(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if not normalized:
        return None
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("zh") or normalized in {"cn", "chinese"}:
        return "zh"
    return None


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
