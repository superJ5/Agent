"""Trace the retrieval pipeline stage by stage for debugging recall failures.

This script is intentionally diagnostic-only: it does not change retrieval
behavior. It mirrors the current routed retrieval flow and records raw vector
hits, metadata filters, lexical rerank scores, scan fallback results, and final
support expansion so we can see where a query goes wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = Path("./reports/retrieval_traces")
DEFAULT_TOP_K = 5
_PIPELINE_DEPENDENCIES_LOADED = False
_KNOWLEDGE_TOOL_SYMBOLS = (
    "INTENT_STAGE_CONFIG",
    "deduplicate_results",
    "detect_intent",
    "expand_queries",
    "extract_query_terms",
    "fetch_support_hits",
    "filter_results_to_active_docs",
    "infer_doc_id",
    "is_strong_hit",
    "lexical_score",
    "merge_ranked_results",
    "normalize_text",
    "post_filter_results",
    "query_critical_term_groups",
    "rerank_results",
    "reset_profile_caches",
    "resolve_chunk_types",
    "should_scan_stage",
    "trim_results_for_intent",
)

SearchResult = Any
vector_search_service: Any = None
INTENT_STAGE_CONFIG: Any = None
deduplicate_results: Any = None
detect_intent: Any = None
expand_queries: Any = None
extract_query_terms: Any = None
fetch_support_hits: Any = None
filter_results_to_active_docs: Any = None
infer_doc_id: Any = None
is_strong_hit: Any = None
lexical_score: Any = None
merge_ranked_results: Any = None
normalize_text: Any = None
post_filter_results: Any = None
query_critical_term_groups: Any = None
rerank_results: Any = None
reset_profile_caches: Any = None
resolve_chunk_types: Any = None
should_scan_stage: Any = None
trim_results_for_intent: Any = None


def _load_pipeline_dependencies() -> None:
    global SearchResult, _PIPELINE_DEPENDENCIES_LOADED, vector_search_service

    if _PIPELINE_DEPENDENCIES_LOADED:
        return

    from app.config import config as app_config
    from app.services.vector_search_service import (
        SearchResult as search_result_cls,
        vector_search_service as loaded_vector_search_service,
    )
    from app.tools import knowledge_tool

    _ = app_config
    SearchResult = search_result_cls
    vector_search_service = loaded_vector_search_service
    for symbol in _KNOWLEDGE_TOOL_SYMBOLS:
        globals()[symbol] = getattr(knowledge_tool, symbol)
    _PIPELINE_DEPENDENCIES_LOADED = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a stage-by-stage retrieval trace report.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Question to trace. Can be passed multiple times.",
    )
    parser.add_argument(
        "--queries-file",
        help="TXT one query per line, or JSON/JSONL rows with query/case_id/doc_id.",
    )
    parser.add_argument("--doc-id", help="Force a doc_id for all ad-hoc queries.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--case-prefix",
        default="trace",
        help="Prefix used for ad-hoc case ids.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = load_cases(args)
    if not cases:
        raise SystemExit("No queries provided. Use --query or --queries-file.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _load_pipeline_dependencies()
    except Exception as exc:
        write_trace_report(
            output_dir=output_dir,
            rows=build_blocked_rows(
                cases=cases,
                forced_doc_id=args.doc_id,
                error=exc,
            ),
        )
        return

    reset_profile_caches()
    rows = [
        trace_case(
            case=case,
            forced_doc_id=args.doc_id,
            top_k=args.top_k,
        )
        for case in cases
    ]
    write_trace_report(output_dir=output_dir, rows=rows)


def write_trace_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = build_summary(rows)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "cases": rows,
    }

    stem = f"retrieval_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "markdown": str(md_path),
                "json": str(json_path),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_blocked_rows(
    cases: Sequence[dict[str, Any]],
    forced_doc_id: str | None,
    error: Exception,
) -> list[dict[str, Any]]:
    error_text = f"{type(error).__name__}: {error}"
    return [
        {
            "case_id": case.get("case_id"),
            "query": str(case.get("query") or ""),
            "intent": None,
            "inferred_doc_id": None,
            "effective_doc_id": forced_doc_id or case.get("doc_id"),
            "query_terms": [],
            "query_variants": [],
            "selected_stage": "blocked",
            "status": "blocked",
            "error": error_text,
            "evaluation": {
                "expected_chunk_ids": [
                    str(item) for item in (case.get("expected_chunk_ids") or [])
                ],
                "expected_terms": [str(item) for item in (case.get("expected_terms") or [])],
                "expected_pic_ids": [
                    str(item) for item in (case.get("expected_pic_ids") or [])
                ],
                "expected_in_final_top3": False,
                "expected_in_support": False,
                "expected_terms_in_top3": {},
                "pic_hit": False,
                "failure_hypothesis": "blocked_pipeline_dependencies_unavailable",
            },
            "stage_traces": [],
            "fallback_trace": None,
            "final_hits": [],
            "support_hits": [],
        }
        for case in cases
    ]


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, query in enumerate(args.query, start=1):
        cases.append(
            {
                "case_id": f"{args.case_prefix}_{index:03d}",
                "query": query,
                "doc_id": args.doc_id,
            }
        )

    if args.queries_file:
        cases.extend(load_cases_from_file(Path(args.queries_file)))
    return cases


def load_cases_from_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("cases") or data.get("queries") or [data]
        return [normalize_case(row, index) for index, row in enumerate(data, start=1)]

    rows: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if path.suffix.lower() == ".jsonl":
            rows.append(normalize_case(json.loads(stripped), index))
        else:
            rows.append({"case_id": f"case_{index:03d}", "query": stripped})
    return rows


def normalize_case(row: Any, index: int) -> dict[str, Any]:
    if isinstance(row, str):
        return {"case_id": f"case_{index:03d}", "query": row}
    if not isinstance(row, dict) or not row.get("query"):
        raise ValueError(f"Invalid case at row {index}: {row!r}")
    return {
        "case_id": row.get("case_id") or f"case_{index:03d}",
        "query": row["query"],
        "doc_id": row.get("doc_id"),
        "expected_chunk_ids": row.get("expected_chunk_ids") or [],
        "expected_terms": row.get("expected_terms") or [],
        "expected_pic_ids": row.get("expected_pic_ids") or [],
    }


def trace_case(case: dict[str, Any], forced_doc_id: str | None, top_k: int) -> dict[str, Any]:
    query = str(case["query"])
    intent = detect_intent(query)
    inferred_doc_id = infer_doc_id(query)
    effective_doc_id = forced_doc_id or case.get("doc_id") or inferred_doc_id
    query_terms = extract_query_terms(query)
    query_variants = expand_queries(query, intent, query_terms)
    stage_config = INTENT_STAGE_CONFIG.get(intent, INTENT_STAGE_CONFIG["general"])

    selected_stage = "none"
    final_hits: list[SearchResult] = []
    stage_traces: list[dict[str, Any]] = []

    for stage in stage_config:
        stage_trace, stage_hits = trace_stage(
            stage=stage,
            query=query,
            query_terms=query_terms,
            query_variants=query_variants,
            intent=intent,
            doc_id=effective_doc_id,
            top_k=top_k,
        )
        stage_traces.append(stage_trace)
        if stage_hits:
            selected_stage = str(stage["name"])
            final_hits = trim_results_for_intent(stage_hits, intent)
            if is_strong_hit(stage_hits, query_terms, intent):
                stage_trace["decision"] = "stop_strong_hit"
                break
            stage_trace["decision"] = "continue_weak_hit"
        else:
            stage_trace["decision"] = "continue_no_hit"

    fallback_trace: dict[str, Any] | None = None
    if not final_hits:
        fallback_trace, final_hits = trace_fallback_full_collection(
            query=query,
            query_terms=query_terms,
            intent=intent,
            doc_id=effective_doc_id,
            top_k=top_k,
        )
        selected_stage = "fallback_full_collection" if final_hits else "none"

    support_hits = fetch_support_hits(final_hits, effective_doc_id, query_terms, intent)
    final_hit_payloads = [hit_to_payload(hit, query, query_terms, intent) for hit in final_hits]
    support_hit_payloads = [hit_to_payload(hit, query, query_terms, intent) for hit in support_hits]
    evaluation = evaluate_trace(case, final_hit_payloads, support_hit_payloads, stage_traces)

    return {
        "case_id": case.get("case_id"),
        "query": query,
        "intent": intent,
        "inferred_doc_id": inferred_doc_id,
        "effective_doc_id": effective_doc_id,
        "query_terms": query_terms,
        "query_variants": query_variants,
        "selected_stage": selected_stage,
        "evaluation": evaluation,
        "stage_traces": stage_traces,
        "fallback_trace": fallback_trace,
        "final_hits": final_hit_payloads,
        "support_hits": support_hit_payloads,
    }


def trace_stage(
    stage: dict[str, Any],
    query: str,
    query_terms: Sequence[str],
    query_variants: Sequence[str],
    intent: str,
    doc_id: str | None,
    top_k: int,
) -> tuple[dict[str, Any], list[SearchResult]]:
    stage_fetch_k = max(top_k * 3, 8)
    tiers = stage.get("tiers")
    families = stage.get("families")
    chunk_types = resolve_chunk_types(families, doc_id=doc_id)
    prefer_support = "support" in (tiers or ())
    vector_attempts: list[dict[str, Any]] = []
    aggregated: list[SearchResult] = []

    for variant in query_variants[:3]:
        try:
            raw_hits = filter_results_to_active_docs(
                vector_search_service.search_similar_documents(
                    query=variant,
                    top_k=stage_fetch_k,
                    doc_id=doc_id,
                    retrieval_tiers=list(tiers) if tiers else None,
                    chunk_types=list(chunk_types) if chunk_types else None,
                ),
                scoped_doc_id=doc_id,
            )
            aggregated.extend(raw_hits)
            vector_attempts.append(
                {
                    "query_variant": variant,
                    "hit_count": len(raw_hits),
                    "top_hits": [
                        hit_to_payload(hit, query, query_terms, intent, prefer_support=prefer_support)
                        for hit in raw_hits[:top_k]
                    ],
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            vector_attempts.append(
                {
                    "query_variant": variant,
                    "error": str(exc),
                    "hit_count": 0,
                    "top_hits": [],
                }
            )

    deduped = deduplicate_results(aggregated)
    reranked = rerank_results(
        deduped,
        original_query=query,
        query_terms=query_terms,
        intent=intent,
        prefer_support=prefer_support,
    )
    post_filtered = post_filter_results(
        reranked,
        intent=intent,
        query_terms=query_terms,
    )

    critical_groups = query_critical_term_groups(normalize_text(query))
    should_scan = should_scan_stage(post_filtered, query_terms, intent)
    scan_triggered = bool(should_scan or critical_groups)
    scan_hits: list[SearchResult] = []
    scan_candidate_count = 0
    if scan_triggered:
        scan_hits, scan_candidate_count = run_scan_for_trace(
            query=query,
            query_terms=query_terms,
            intent=intent,
            doc_id=doc_id,
            tiers=tiers,
            chunk_types=chunk_types,
            top_k=stage_fetch_k,
            prefer_support=prefer_support,
        )
        stage_hits = rerank_results(
            merge_ranked_results(post_filtered, scan_hits),
            original_query=query,
            query_terms=query_terms,
            intent=intent,
            prefer_support=prefer_support,
        )[:top_k]
    else:
        stage_hits = post_filtered[:top_k]

    trace = {
        "stage_name": stage.get("name"),
        "tiers": tiers,
        "families": families,
        "resolved_chunk_types": chunk_types,
        "stage_fetch_k": stage_fetch_k,
        "vector_attempts": vector_attempts,
        "vector_deduped_count": len(deduped),
        "reranked_top_hits": [
            hit_to_payload(hit, query, query_terms, intent, prefer_support=prefer_support)
            for hit in reranked[:top_k]
        ],
        "post_filter_count": len(post_filtered),
        "post_filter_top_hits": [
            hit_to_payload(hit, query, query_terms, intent, prefer_support=prefer_support)
            for hit in post_filtered[:top_k]
        ],
        "critical_groups": [list(group) for group in critical_groups],
        "scan_triggered": scan_triggered,
        "scan_reason": {
            "weak_vector_hit": should_scan,
            "critical_term_groups": bool(critical_groups),
        },
        "scan_candidate_count": scan_candidate_count,
        "scan_top_hits": [
            hit_to_payload(hit, query, query_terms, intent, prefer_support=prefer_support)
            for hit in scan_hits[:top_k]
        ],
        "selected_top_hits": [
            hit_to_payload(hit, query, query_terms, intent, prefer_support=prefer_support)
            for hit in stage_hits[:top_k]
        ],
        "strong_hit": is_strong_hit(stage_hits, query_terms, intent) if stage_hits else False,
    }
    return trace, stage_hits


def trace_fallback_full_collection(
    query: str,
    query_terms: Sequence[str],
    intent: str,
    doc_id: str | None,
    top_k: int,
) -> tuple[dict[str, Any], list[SearchResult]]:
    hits, scan_candidate_count = run_scan_for_trace(
        query=query,
        query_terms=query_terms,
        intent=intent,
        doc_id=doc_id,
        tiers=["primary", "support", "auxiliary"],
        chunk_types=None,
        top_k=max(top_k, 6),
        prefer_support=False,
    )
    return {
        "stage_name": "fallback_full_collection",
        "tiers": ["primary", "support", "auxiliary"],
        "families": None,
        "resolved_chunk_types": None,
        "scan_candidate_count": scan_candidate_count,
        "scan_top_hits": [hit_to_payload(hit, query, query_terms, intent) for hit in hits[:top_k]],
    }, hits[:top_k]


def run_scan_for_trace(
    query: str,
    query_terms: Sequence[str],
    intent: str,
    doc_id: str | None,
    tiers: Sequence[str] | None,
    chunk_types: Sequence[str] | None,
    top_k: int,
    prefer_support: bool,
) -> tuple[list[SearchResult], int]:
    raw_scanned = vector_search_service.query_all_documents(
        doc_id=doc_id,
        retrieval_tiers=list(tiers) if tiers else None,
        chunk_types=list(chunk_types) if chunk_types else None,
    )
    scanned = filter_results_to_active_docs(
        raw_scanned,
        scoped_doc_id=doc_id,
    )
    reranked = rerank_results(
        scanned,
        original_query=query,
        query_terms=query_terms,
        intent=intent,
        prefer_support=prefer_support,
    )
    return post_filter_results(reranked, intent=intent, query_terms=query_terms)[:top_k], len(raw_scanned)


def hit_to_payload(
    hit: SearchResult,
    query: str,
    query_terms: Sequence[str],
    intent: str,
    prefer_support: bool = False,
) -> dict[str, Any]:
    metadata = hit.metadata or {}
    title = str(metadata.get("title") or metadata.get("section_title") or "")
    section_path = metadata.get("section_path") or []
    section_text = " > ".join(str(part) for part in section_path) if isinstance(section_path, list) else str(section_path)
    text = str(metadata.get("text") or hit.content or "")
    index_text = str(metadata.get("index_text") or "")
    haystack = normalize_text(" ".join([title, section_text, index_text, text]))
    matched_terms = [
        term
        for term in query_terms
        if normalize_text(str(term)) and normalize_text(str(term)) in haystack
    ]
    return {
        "chunk_id": metadata.get("chunk_id") or hit.id,
        "doc_id": metadata.get("doc_id"),
        "title": title,
        "section_path": section_path,
        "retrieval_tier": metadata.get("retrieval_tier"),
        "chunk_type": metadata.get("chunk_type"),
        "parent_chunk_id": metadata.get("parent_chunk_id"),
        "vector_l2_distance": round(float(hit.score), 6),
        "embedding_similarity_proxy": round(1.0 / (1.0 + max(float(hit.score), 0.0)), 6),
        "lexical_rerank_score": round(
            lexical_score(hit, query, query_terms, intent, prefer_support=prefer_support),
            6,
        ),
        "matched_query_terms": matched_terms,
        "pic_ids": metadata.get("pic_ids") or [],
        "source_issue_flags": metadata.get("source_issue_flags") or [],
        "source_issue_note": metadata.get("source_issue_note") or "",
        "text_preview": compact_preview(text),
    }


def compact_preview(text: str, limit: int = 220) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def evaluate_trace(
    case: dict[str, Any],
    final_hits: list[dict[str, Any]],
    support_hits: list[dict[str, Any]],
    stage_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_chunk_ids = [str(item) for item in (case.get("expected_chunk_ids") or [])]
    expected_terms = [str(item) for item in (case.get("expected_terms") or [])]
    expected_pic_ids = [str(item) for item in (case.get("expected_pic_ids") or [])]
    final_chunk_ids = [str(hit.get("chunk_id")) for hit in final_hits]
    all_hits = [*final_hits, *support_hits]

    expected_in_final_top3 = contains_expected(final_chunk_ids[:3], expected_chunk_ids)
    expected_in_support = contains_expected(
        [str(hit.get("chunk_id")) for hit in support_hits],
        expected_chunk_ids,
    )
    term_hit_top3 = terms_in_hits(expected_terms, final_hits[:3])
    pic_hit = not expected_pic_ids or bool(
        set(expected_pic_ids)
        & {
            str(pic_id)
            for hit in all_hits
            for pic_id in (hit.get("pic_ids") or [])
        }
    )

    return {
        "expected_chunk_ids": expected_chunk_ids,
        "expected_terms": expected_terms,
        "expected_pic_ids": expected_pic_ids,
        "expected_in_final_top3": expected_in_final_top3,
        "expected_in_support": expected_in_support,
        "expected_terms_in_top3": term_hit_top3,
        "pic_hit": pic_hit,
        "failure_hypothesis": diagnose_failure(expected_chunk_ids, final_hits, stage_traces),
    }


def contains_expected(values: Sequence[str], expected: Sequence[str]) -> bool:
    if not expected:
        return False
    return any(exp and any(exp == value or exp in value for value in values) for exp in expected)


def terms_in_hits(expected_terms: Sequence[str], hits: Sequence[dict[str, Any]]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for term in expected_terms:
        normalized_term = normalize_text(term)
        results[term] = any(
            normalized_term
            and normalized_term
            in normalize_text(" ".join([str(hit.get("title") or ""), str(hit.get("text_preview") or "")]))
            for hit in hits
        )
    return results


def diagnose_failure(
    expected_chunk_ids: Sequence[str],
    final_hits: Sequence[dict[str, Any]],
    stage_traces: Sequence[dict[str, Any]],
) -> str:
    if not expected_chunk_ids:
        return "no_expected_chunk_ids_provided"
    final_ids = [str(hit.get("chunk_id")) for hit in final_hits]
    if contains_expected(final_ids[:3], expected_chunk_ids):
        return "ok_expected_in_final_top3"

    vector_ids = [
        str(hit.get("chunk_id"))
        for stage in stage_traces
        for attempt in stage.get("vector_attempts") or []
        for hit in attempt.get("top_hits") or []
    ]
    scan_ids = [
        str(hit.get("chunk_id"))
        for stage in stage_traces
        for hit in stage.get("scan_top_hits") or []
    ]
    rerank_ids = [
        str(hit.get("chunk_id"))
        for stage in stage_traces
        for hit in stage.get("selected_top_hits") or []
    ]

    if contains_expected(vector_ids, expected_chunk_ids) and not contains_expected(rerank_ids, expected_chunk_ids):
        return "expected_seen_in_vector_but_lost_after_rerank_or_filter"
    if contains_expected(scan_ids, expected_chunk_ids) and not contains_expected(final_ids[:3], expected_chunk_ids):
        return "expected_seen_in_scan_but_lost_after_merge_or_late_stage"
    if any(stage.get("post_filter_count") == 0 for stage in stage_traces):
        return "candidate_removed_by_post_filter_or_metadata_filter"
    if not vector_ids and not scan_ids:
        return "no_candidates_after_metadata_filters"
    return "expected_not_seen_in_traced_top_candidates"


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocked_count = sum(1 for row in rows if row.get("status") == "blocked")
    return {
        "case_count": len(rows),
        "blocked_count": blocked_count,
        "expected_top3_hits": sum(
            1 for row in rows if row.get("evaluation", {}).get("expected_in_final_top3")
        ),
        "term_top3_full_hits": sum(
            1
            for row in rows
            if row.get("evaluation", {}).get("expected_terms_in_top3")
            and all(row["evaluation"]["expected_terms_in_top3"].values())
        ),
        "stage_counts": count_values(row.get("selected_stage") for row in rows),
        "intent_counts": count_values(row.get("intent") for row in rows),
    }


def count_values(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def render_markdown(payload: dict[str, Any]) -> str:
    rows = payload["cases"]
    lines = [
        "# Retrieval Pipeline Trace",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "Embedding note: `vector_l2_distance` is Milvus L2 distance, lower is better. "
        "`embedding_similarity_proxy = 1 / (1 + distance)` is only a readable proxy.",
        "",
        "## Summary",
        "",
        f"- Cases: `{payload['summary']['case_count']}`",
        f"- Blocked: `{payload['summary'].get('blocked_count', 0)}`",
        f"- Expected chunk H@3: `{payload['summary']['expected_top3_hits']}`",
        f"- Stage counts: `{payload['summary']['stage_counts']}`",
        f"- Intent counts: `{payload['summary']['intent_counts']}`",
        "",
        "| Case | Query | Intent | Doc | Stage | Top1 | Failure hypothesis |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        top1 = (row.get("final_hits") or [{}])[0]
        top1_text = "not run" if row.get("status") == "blocked" else f"{top1.get('chunk_id')} / {top1.get('title')}"
        lines.append(
            "| {case_id} | {query} | {intent} | {doc} | {stage} | {top1} | {why} |".format(
                case_id=escape(row.get("case_id")),
                query=escape(row.get("query")),
                intent=escape(row.get("intent")),
                doc=escape(row.get("effective_doc_id")),
                stage=escape(row.get("selected_stage")),
                top1=escape(top1_text),
                why=escape(row["evaluation"]["failure_hypothesis"]),
            )
        )

    lines.extend(["", "## Details"])
    for row in rows:
        lines.extend(render_case_detail(row))
    lines.append("")
    return "\n".join(lines)


def render_case_detail(row: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"### {escape(row.get('case_id'))}",
        "",
        f"- Query: `{escape(row.get('query'))}`",
        f"- Intent: `{escape(row.get('intent'))}`",
        f"- Inferred doc_id: `{escape(row.get('inferred_doc_id'))}`",
        f"- Effective doc_id: `{escape(row.get('effective_doc_id'))}`",
        f"- Query terms: `{escape(row.get('query_terms'))}`",
        f"- Query variants: `{escape(row.get('query_variants'))}`",
        f"- Final stage: `{escape(row.get('selected_stage'))}`",
        f"- Failure hypothesis: `{escape(row['evaluation']['failure_hypothesis'])}`",
        "",
    ]

    if row.get("status") == "blocked":
        lines.extend(
            [
                f"- Status: `{escape(row.get('status'))}`",
                f"- Error: `{escape(row.get('error'))}`",
                "- Trace execution: blocked before retrieval; no vector, scan, or rerank stages were run.",
                "",
            ]
        )

    for stage in row.get("stage_traces") or []:
        lines.extend(
            [
                f"#### Stage: {escape(stage.get('stage_name'))}",
                "",
                f"- Tiers: `{escape(stage.get('tiers'))}`",
                f"- Families: `{escape(stage.get('families'))}`",
                f"- Resolved chunk_types: `{escape(stage.get('resolved_chunk_types'))}`",
                f"- Vector deduped count: `{stage.get('vector_deduped_count')}`",
                f"- Post-filter count: `{stage.get('post_filter_count')}`",
                f"- Scan triggered: `{stage.get('scan_triggered')}` / `{escape(stage.get('scan_reason'))}`",
                f"- Scan candidate count: `{stage.get('scan_candidate_count')}`",
                f"- Strong hit: `{stage.get('strong_hit')}`",
                f"- Decision: `{escape(stage.get('decision'))}`",
                "",
                "Vector attempts:",
            ]
        )
        for attempt in stage.get("vector_attempts") or []:
            lines.append("")
            lines.append(f"- Variant: `{escape(attempt.get('query_variant'))}`; hits=`{attempt.get('hit_count')}`")
            lines.extend(render_hits_table(attempt.get("top_hits") or []))

        lines.extend(["", "Scan top hits:"])
        lines.extend(render_hits_table(stage.get("scan_top_hits") or []))
        lines.extend(["", "Selected stage hits:"])
        lines.extend(render_hits_table(stage.get("selected_top_hits") or []))

    if row.get("fallback_trace"):
        lines.extend(["", "#### Fallback Full Collection", ""])
        lines.append(f"- Scan candidate count: `{row['fallback_trace'].get('scan_candidate_count')}`")
        lines.append("")
        lines.extend(render_hits_table(row["fallback_trace"].get("scan_top_hits") or []))

    lines.extend(["", "Final hits:"])
    lines.extend(render_hits_table(row.get("final_hits") or []))
    if row.get("support_hits"):
        lines.extend(["", "Support hits:"])
        lines.extend(render_hits_table(row.get("support_hits") or []))
    return lines


def render_hits_table(hits: Sequence[dict[str, Any]]) -> list[str]:
    if not hits:
        return ["- No hits."]
    lines = [
        "| Chunk | Title | Tier/Type | L2 | SimProxy | LexScore | Matched terms |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for hit in hits:
        lines.append(
            "| {chunk} | {title} | {tier}/{typ} | {dist} | {sim} | {lex} | {terms} |".format(
                chunk=escape(hit.get("chunk_id")),
                title=escape(hit.get("title")),
                tier=escape(hit.get("retrieval_tier")),
                typ=escape(hit.get("chunk_type")),
                dist=hit.get("vector_l2_distance"),
                sim=hit.get("embedding_similarity_proxy"),
                lex=hit.get("lexical_rerank_score"),
                terms=escape(hit.get("matched_query_terms")),
            )
        )
    return lines


def escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
