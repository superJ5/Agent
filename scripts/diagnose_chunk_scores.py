"""Diagnose vector and BM25 scores for one chunk against one query."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

bm25_module: Any = None
JiebaBM25Provider: Any = None
vector_search_service: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Query text to score.")
    parser.add_argument("--chunk-id", required=True, help="Target chunk_id.")
    parser.add_argument(
        "--language",
        default=None,
        help="Language filter, usually zh or en. Omit to disable language filtering.",
    )
    parser.add_argument(
        "--retrieval-tiers",
        default="primary",
        help="Comma-separated retrieval_tier filter. Default: primary. Use empty string to disable.",
    )
    parser.add_argument(
        "--chunk-types",
        default="",
        help="Comma-separated chunk_type filter. Default: disabled.",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Optional doc_id filter for BM25 rank simulation.",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=32,
        help="Recall fetch limit used to judge whether the chunk would be returned. Default: 32.",
    )
    parser.add_argument(
        "--vector-rank-k",
        type=int,
        default=256,
        help="How many vector hits to fetch when estimating vector rank. Default: 256.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dependencies()

    from app.core.milvus_client import milvus_manager

    milvus_manager.connect()

    target = fetch_target_chunk(args.chunk_id)
    if target is None:
        raise SystemExit(f"Chunk not found: {args.chunk_id}")

    filters = {
        "language": clean_optional(args.language),
        "retrieval_tiers": split_csv(args.retrieval_tiers),
        "chunk_types": split_csv(args.chunk_types),
        "doc_id": clean_optional(args.doc_id),
    }

    print_report(args, target, filters)


def load_dependencies() -> None:
    global JiebaBM25Provider, bm25_module, vector_search_service

    from app.retrieval import bm25_provider as loaded_bm25_module
    from app.retrieval.bm25_provider import JiebaBM25Provider as LoadedBM25Provider
    from app.services.vector_search_service import vector_search_service as loaded_vector_service

    bm25_module = loaded_bm25_module
    JiebaBM25Provider = LoadedBM25Provider
    vector_search_service = loaded_vector_service


def fetch_target_chunk(chunk_id: str) -> SearchResult | None:
    rows = vector_search_service.query_documents(chunk_ids=[chunk_id], limit=1)
    return rows[0] if rows else None


def print_report(args: argparse.Namespace, target: SearchResult, filters: dict[str, Any]) -> None:
    metadata = target.metadata or {}
    print("# Chunk Score Diagnosis")
    print()
    print(f"- query: `{args.query}`")
    print(f"- chunk_id: `{args.chunk_id}`")
    print(f"- doc_id: `{metadata.get('doc_id') or ''}`")
    print(f"- language: `{metadata.get('language') or ''}`")
    print(f"- retrieval_tier: `{metadata.get('retrieval_tier') or ''}`")
    print(f"- chunk_type: `{metadata.get('chunk_type') or ''}`")
    print(f"- title: `{metadata.get('title') or metadata.get('section_title') or ''}`")
    print(f"- filters: `{filters}`")
    print()

    print("## Vector")
    print()
    print_vector_report(args, filters)
    print()

    print("## BM25")
    print()
    print_bm25_report(args, target, filters)


def print_vector_report(args: argparse.Namespace, filters: dict[str, Any]) -> None:
    unfiltered_direct = vector_search_service.search_similar_documents(
        query=args.query,
        top_k=1,
        chunk_ids=[args.chunk_id],
    )
    filtered_direct = vector_search_service.search_similar_documents(
        query=args.query,
        top_k=1,
        chunk_ids=[args.chunk_id],
        language=filters["language"],
        retrieval_tiers=filters["retrieval_tiers"] or None,
        chunk_types=filters["chunk_types"] or None,
    )
    filtered_rank_hits = vector_search_service.search_similar_documents(
        query=args.query,
        top_k=max(args.vector_rank_k, 1),
        doc_id=filters["doc_id"],
        language=filters["language"],
        retrieval_tiers=filters["retrieval_tiers"] or None,
        chunk_types=filters["chunk_types"] or None,
    )
    rank = find_rank(filtered_rank_hits, args.chunk_id)

    print_result_line("direct_score_no_filters", score_of_first(unfiltered_direct))
    print_result_line("direct_score_with_filters", score_of_first(filtered_direct))
    print_result_line("rank_with_filters", rank)
    print_result_line("within_fetch_k", bool(rank and rank <= args.fetch_k))
    if rank is None:
        print(f"- note: not found in vector top {max(args.vector_rank_k, 1)} with filters")


def print_bm25_report(args: argparse.Namespace, target: SearchResult, filters: dict[str, Any]) -> None:
    all_rows = vector_search_service.query_all_documents()
    provider = JiebaBM25Provider()
    provider.build_index(all_rows)

    query_tokens = provider._tokenize(args.query, language=filters["language"])  # noqa: SLF001
    bm25 = provider._bm25  # noqa: SLF001
    raw_scores = list(bm25.get_scores(query_tokens)) if bm25 is not None else []

    target_index = find_provider_index(provider, args.chunk_id)
    if target_index is None:
        print("- bm25_raw_score: target not found in provider index")
        return

    target_score = float(raw_scores[target_index]) if target_index < len(raw_scores) else 0.0
    filtered_scores = filtered_bm25_scores(
        provider,
        raw_scores,
        filters=filters,
    )
    rank = find_bm25_rank(filtered_scores, args.chunk_id)
    target_matches_filters = bm25_matches_filters(target, filters)

    print_result_line("query_tokens", query_tokens)
    print_result_line("bm25_raw_score", target_score)
    print_result_line("matches_filters", target_matches_filters)
    print_result_line("rank_with_filters", rank)
    print_result_line("within_fetch_k", bool(rank and rank <= args.fetch_k))
    if target_score <= 0:
        print("- note: BM25 score is 0, so this chunk cannot enter BM25 recall for this query")
    elif not target_matches_filters:
        print("- note: BM25 score is positive, but filters exclude this chunk")
    elif rank is not None and rank > args.fetch_k:
        print(f"- note: BM25 score is positive, but rank {rank} is outside fetch_k={args.fetch_k}")


def filtered_bm25_scores(
    provider: JiebaBM25Provider,
    raw_scores: list[Any],
    *,
    filters: dict[str, Any],
) -> list[tuple[float, int, SearchResult]]:
    scored: list[tuple[float, int, SearchResult]] = []
    for index, result in enumerate(provider._results):  # noqa: SLF001
        if index >= len(raw_scores):
            break
        score = safe_float(raw_scores[index])
        if score <= 0:
            continue
        if not bm25_matches_filters(result, filters):
            continue
        scored.append((score, index, result))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def bm25_matches_filters(result: SearchResult, filters: dict[str, Any]) -> bool:
    return bool(
        bm25_module._matches_filters(  # noqa: SLF001
            result,
            doc_id=filters["doc_id"],
            retrieval_tiers=set(filters["retrieval_tiers"]),
            chunk_types=set(filters["chunk_types"]),
            language=bm25_module._normalise_language(filters["language"]),  # noqa: SLF001
        )
    )


def find_provider_index(provider: JiebaBM25Provider, chunk_id: str) -> int | None:
    for index, result in enumerate(provider._results):  # noqa: SLF001
        if result_chunk_id(result) == chunk_id:
            return index
    return None


def find_bm25_rank(scored: list[tuple[float, int, SearchResult]], chunk_id: str) -> int | None:
    for rank, (_score, _index, result) in enumerate(scored, start=1):
        if result_chunk_id(result) == chunk_id:
            return rank
    return None


def find_rank(results: list[SearchResult], chunk_id: str) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result_chunk_id(result) == chunk_id:
            return rank
    return None


def result_chunk_id(result: SearchResult) -> str:
    metadata = getattr(result, "metadata", None) or {}
    return str(metadata.get("chunk_id") or getattr(result, "id", "") or "")


def score_of_first(results: list[SearchResult]) -> float | None:
    if not results:
        return None
    return safe_float(getattr(results[0], "score", None))


def print_result_line(key: str, value: Any) -> None:
    print(f"- {key}: `{value}`")


def split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
