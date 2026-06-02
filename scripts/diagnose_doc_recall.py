"""Diagnose why one document does or does not appear in retrieval traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

from app.retrieval.bm25_provider import JiebaBM25Provider
from app.services.vector_search_service import vector_search_service


DEFAULT_QUERIES = (
    "battery conversion feature before sailing",
    "ship steering how the ship steers",
    "over temperature warning boat engine",
    "storage compartments wet items on board",
    "open battery compartment boat manipulation",
    "anchor light installation boat moving",
    "jet wash function clean boat cleaning instructions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", default="manual_f769aafb")
    parser.add_argument("--trace", default="logs/retrieval_trace.jsonl")
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--output",
        default="doc/manual_f769aafb_recall_diagnosis.md",
    )
    parser.add_argument("queries", nargs="*")
    return parser.parse_args()


def read_trace_records(path: Path, tail: int) -> list[dict[str, Any]]:
    records: deque[dict[str, Any]] = deque(maxlen=max(tail, 1))
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as trace_file:
        for line in trace_file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(records)


def metadata(result: Any) -> dict[str, Any]:
    value = getattr(result, "metadata", None)
    return value if isinstance(value, dict) else {}


def result_score(result: Any) -> Any:
    return getattr(result, "score", None)


def trace_channel_hits(record: dict[str, Any], channel: str, doc_id: str) -> list[tuple[int, dict[str, Any]]]:
    channels = ((record.get("recall") or {}).get("channels") or {})
    items = channels.get(channel) or []
    return [
        (index, item)
        for index, item in enumerate(items, start=1)
        if isinstance(item, dict) and item.get("doc_id") == doc_id
    ]


def trace_merged_hits(record: dict[str, Any], doc_id: str) -> list[tuple[int, dict[str, Any]]]:
    items = ((record.get("recall") or {}).get("recall_candidates") or [])
    hits: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        result = item.get("result") or {}
        if isinstance(result, dict) and result.get("doc_id") == doc_id:
            hits.append((index, item))
    return hits


def trace_rerank_hits(record: dict[str, Any], doc_id: str) -> list[tuple[int, dict[str, Any]]]:
    # reranker.post_rank currently does not always carry doc_id, so join by chunk ids from recall.
    chunk_to_doc: dict[str, str] = {}
    for item in ((record.get("recall") or {}).get("recall_candidates") or []):
        if not isinstance(item, dict):
            continue
        result = item.get("result") or {}
        if not isinstance(result, dict):
            continue
        chunk_id = str(result.get("chunk_id") or item.get("chunk_id") or "")
        item_doc_id = str(result.get("doc_id") or "")
        if chunk_id and item_doc_id:
            chunk_to_doc[chunk_id] = item_doc_id

    hits: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(((record.get("reranker") or {}).get("post_rank") or []), start=1):
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        if chunk_to_doc.get(chunk_id) == doc_id:
            hits.append((index, item))
    return hits


def trace_evidence_hits(record: dict[str, Any], key: str, doc_id: str) -> list[tuple[int, dict[str, Any]]]:
    items = record.get(key) or []
    return [
        (index, item)
        for index, item in enumerate(items, start=1)
        if isinstance(item, dict) and item.get("doc_id") == doc_id
    ]


def format_trace_hit(index: int, item: dict[str, Any]) -> str:
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    chunk_id = item.get("chunk_id") or result.get("chunk_id") or result.get("id")
    doc_id = item.get("doc_id") or result.get("doc_id")
    score = (
        item.get("reranker_score")
        if item.get("reranker_score") is not None
        else item.get("merged_score")
        if item.get("merged_score") is not None
        else item.get("score")
    )
    title = item.get("title") or result.get("title")
    channels = item.get("recall_channels")
    channel_text = f" channels={','.join(channels)}" if isinstance(channels, list) else ""
    return f"  - rank={index} chunk={chunk_id} doc={doc_id} score={score}{channel_text} title={title}"


def format_result(index: int, result: Any) -> str:
    item_metadata = metadata(result)
    return (
        f"  - rank={index} chunk={item_metadata.get('chunk_id')} "
        f"score={result_score(result)} title={item_metadata.get('title')}"
    )


def build_report(args: argparse.Namespace) -> str:
    doc_rows = vector_search_service.query_all_documents(doc_id=args.doc_id)
    tier_counts = Counter(metadata(row).get("retrieval_tier") for row in doc_rows)
    type_counts = Counter(metadata(row).get("chunk_type") for row in doc_rows)

    all_rows = vector_search_service.query_all_documents()
    provider = JiebaBM25Provider()
    provider.build_index(all_rows)

    trace_records = read_trace_records(Path(args.trace), args.tail)
    queries = list(args.queries) if args.queries else [str(record.get("query") or "") for record in trace_records]
    queries = [query for query in queries if query] or list(DEFAULT_QUERIES)

    lines: list[str] = [
        "# Document Recall Diagnosis",
        "",
        f"- Target doc_id: `{args.doc_id}`",
        f"- Trace file: `{args.trace}`",
        f"- Trace records inspected: `{len(trace_records)}`",
        f"- Current DB chunks for target: `{len(doc_rows)}`",
        f"- Current DB tier counts: `{dict(tier_counts)}`",
        f"- Current DB top chunk types: `{type_counts.most_common(10)}`",
        "",
        "## Interpretation",
        "",
        "- If the trace section has no target doc but direct BM25 ranks it highly, the running service BM25 index is likely stale or the report is from an old trace.",
        "- If direct BM25 also ranks it poorly, inspect tokenization, language, tier, and index_text.",
        "- If recall contains the target doc but rerank/evidence does not, inspect reranker and top_k truncation.",
        "",
    ]

    trace_by_query = {str(record.get("query") or ""): record for record in trace_records}

    for query in queries:
        lines.extend(["---", "", f"## Query: `{query}`", ""])
        bm25_hits = provider.search(
            query,
            top_k=args.top_k,
            language=args.language,
            retrieval_tiers=["primary"],
        )
        target_bm25 = [
            (index, result)
            for index, result in enumerate(bm25_hits, start=1)
            if metadata(result).get("doc_id") == args.doc_id
        ]
        lines.append("### Direct BM25 From Current DB")
        if target_bm25:
            for index, result in target_bm25[:20]:
                lines.append(format_result(index, result))
        else:
            lines.append("  - target doc not found in direct BM25 top_k")

        record = trace_by_query.get(query)
        if record is None:
            lines.extend(["", "### Trace", "  - no exact query match in inspected trace tail"])
            continue

        lines.append("")
        lines.append("### Trace Channel Presence")
        for channel in ("vector", "bm25", "scan"):
            hits = trace_channel_hits(record, channel, args.doc_id)
            lines.append(f"- {channel}: {len(hits)} hit(s)")
            for index, item in hits[:10]:
                lines.append(format_trace_hit(index, item))

        merged_hits = trace_merged_hits(record, args.doc_id)
        rerank_hits = trace_rerank_hits(record, args.doc_id)
        primary_hits = trace_evidence_hits(record, "evidence_primary_hits", args.doc_id)
        support_hits = trace_evidence_hits(record, "evidence_support_hits", args.doc_id)

        lines.append(f"- merged: {len(merged_hits)} hit(s)")
        for index, item in merged_hits[:10]:
            lines.append(format_trace_hit(index, item))
        lines.append(f"- rerank_post: {len(rerank_hits)} hit(s)")
        for index, item in rerank_hits[:10]:
            lines.append(format_trace_hit(index, item))
        lines.append(f"- evidence_primary: {len(primary_hits)} hit(s)")
        for index, item in primary_hits[:10]:
            lines.append(format_trace_hit(index, item))
        lines.append(f"- evidence_support: {len(support_hits)} hit(s)")
        for index, item in support_hits[:10]:
            lines.append(format_trace_hit(index, item))

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
