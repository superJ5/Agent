"""Build human-readable review pages from retrieval trace JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "single"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "single_review"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one-line retrieval trace JSONL files into Markdown review pages.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="retrieval_trace_last400_*.jsonl")
    parser.add_argument("--max-channel-hits", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-rerank", type=int, default=20)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    item_dir = output_dir / "items"
    output_dir.mkdir(parents=True, exist_ok=True)
    item_dir.mkdir(parents=True, exist_ok=True)

    trace_files = sorted(input_dir.glob(args.pattern))
    if not trace_files:
        raise SystemExit(f"No trace files found in {input_dir} with pattern {args.pattern!r}")

    rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []

    for index, path in enumerate(trace_files, 1):
        trace = read_trace(path)
        review_name = f"trace_{index:03d}.md"
        review_path = item_dir / review_name
        summary = build_summary_row(index, path, review_name, trace)
        rows.append(summary)
        if summary["warnings"]:
            warning_rows.append(summary)
        review_path.write_text(
            render_trace_review(
                index=index,
                source_path=path,
                trace=trace,
                max_channel_hits=args.max_channel_hits,
                max_candidates=args.max_candidates,
                max_rerank=args.max_rerank,
            ),
            encoding="utf-8",
        )

    (output_dir / "index.md").write_text(render_index(rows, input_dir, output_dir), encoding="utf-8")
    (output_dir / "warnings.md").write_text(render_warnings(warning_rows), encoding="utf-8")
    write_csv(output_dir / "summary.csv", rows)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Trace files: {len(rows)}")
    print(f"Review pages: {item_dir}")
    print(f"Index: {output_dir / 'index.md'}")
    print(f"CSV: {output_dir / 'summary.csv'}")


def read_trace(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return {}
    line = next((item for item in text.splitlines() if item.strip()), "{}")
    trace = json.loads(line)
    return normalize_trace(trace)


def normalize_trace(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, Mapping):
        return {}
    for key in ("trace", "retrieval_trace"):
        nested = trace.get(key)
        if isinstance(nested, Mapping):
            merged = dict(nested)
            for passthrough_key in ("question", "session_id", "request_id"):
                if passthrough_key not in merged and passthrough_key in trace:
                    merged[passthrough_key] = trace[passthrough_key]
            return merged
    return dict(trace)


def build_summary_row(index: int, source_path: Path, review_name: str, trace: Mapping[str, Any]) -> dict[str, Any]:
    recall = as_mapping(trace.get("recall"))
    channels = as_mapping(recall.get("channels"))
    reranker = as_mapping(trace.get("reranker"))
    primary_hits = as_sequence(trace.get("evidence_primary_hits"))
    support_hits = as_sequence(trace.get("evidence_support_hits"))
    recall_candidates = trace_recall_candidates(trace, recall)
    warnings = as_sequence(trace.get("warnings"))

    return {
        "index": index,
        "source_file": source_path.name,
        "review_file": f"items/{review_name}",
        "request_id": str(trace.get("request_id") or ""),
        "session_id": str(trace.get("session_id") or ""),
        "question": str(trace.get("question") or ""),
        "query": str(trace.get("query") or ""),
        "language": str(trace.get("language") or ""),
        "vector_count": len(as_sequence(channels.get("vector"))),
        "bm25_count": len(as_sequence(channels.get("bm25"))),
        "scan_count": len(as_sequence(channels.get("scan"))),
        "candidate_count": len(recall_candidates),
        "reranker_provider": str(reranker.get("provider") or ""),
        "reranker_model": str(reranker.get("model") or ""),
        "reranker_fallback": str(reranker.get("fallback_used") or False),
        "reranker_input": value_or_blank(reranker.get("input_count")),
        "primary_count": len(primary_hits),
        "support_count": len(support_hits),
        "big_support_count": len(as_sequence(trace.get("big_support_expanded_hits"))),
        "image_count": image_count(trace),
        "warnings": "; ".join(str(item) for item in warnings),
    }


def render_index(rows: Sequence[Mapping[str, Any]], input_dir: Path, output_dir: Path) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Retrieval Trace 审查索引",
        "",
        f"- 生成时间：{generated_at}",
        f"- 输入目录：`{input_dir}`",
        f"- 输出目录：`{output_dir}`",
        f"- Trace 数量：{len(rows)}",
        "",
        "## 汇总表",
        "",
        "| # | Review | Question | Query | Lang | Vector | BM25 | Scan | Candidates | Reranker | Fallback | Primary | Support | BigSupport | Warnings |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {index} | [{source_file}]({review_file}) | {question} | {query} | {language} | {vector_count} | "
            "{bm25_count} | {scan_count} | {candidate_count} | {reranker_provider} | "
            "{reranker_fallback} | {primary_count} | {support_count} | {big_support_count} | {warnings} |".format(
                index=row["index"],
                source_file=escape_md(str(row["source_file"])),
                review_file=str(row["review_file"]).replace("\\", "/"),
                question=escape_md(shorten(str(row["question"]), 80)),
                query=escape_md(shorten(str(row["query"]), 80)),
                language=escape_md(str(row["language"])),
                vector_count=row["vector_count"],
                bm25_count=row["bm25_count"],
                scan_count=row["scan_count"],
                candidate_count=row["candidate_count"],
                reranker_provider=escape_md(str(row["reranker_provider"])),
                reranker_fallback=escape_md(str(row["reranker_fallback"])),
                primary_count=row["primary_count"],
                support_count=row["support_count"],
                big_support_count=row["big_support_count"],
                warnings=escape_md(shorten(str(row["warnings"]), 80)),
            )
        )
    lines.extend(
        [
            "",
            "## 文件说明",
            "",
            f"- `items/trace_001.md` 到 `items/trace_{len(rows):03d}.md`：每条 trace 的详细审查页。",
            "- `summary.csv`：同样的汇总数据，方便导入 Excel 或飞书表格。",
            "- `warnings.md`：只列出带 warning 的 trace。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_warnings(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Retrieval Trace Warnings",
        "",
    ]
    if not rows:
        lines.append("没有带 warning 的 trace。")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| # | Review | Question | Query | Warnings |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['index']} | [{escape_md(str(row['source_file']))}]({row['review_file']}) | "
            f"{escape_md(shorten(str(row['question']), 80))} | "
            f"{escape_md(shorten(str(row['query']), 80))} | {escape_md(str(row['warnings']))} |"
        )
    return "\n".join(lines) + "\n"


def render_trace_review(
    *,
    index: int,
    source_path: Path,
    trace: Mapping[str, Any],
    max_channel_hits: int,
    max_candidates: int,
    max_rerank: int,
) -> str:
    recall = as_mapping(trace.get("recall"))
    reranker = as_mapping(trace.get("reranker"))
    query_understanding = as_mapping(trace.get("query_understanding"))

    lines = [
        f"# Trace {index:03d}",
        "",
        "## 1. 基本信息",
        "",
        markdown_kv_table(
            [
                ("source_file", source_path.name),
                ("request_id", trace.get("request_id")),
                ("session_id", trace.get("session_id")),
                ("question", trace.get("question")),
                ("query", trace.get("query")),
                ("language", trace.get("language")),
                ("warnings", "; ".join(str(item) for item in as_sequence(trace.get("warnings")))),
            ]
        ),
        "",
        "## 2. Query Understanding",
        "",
        render_query_understanding(query_understanding),
        "",
        "## 3. Recall 召回",
        "",
        render_recall_summary(recall),
        "",
        render_channel_hits(recall, max_channel_hits),
        "",
        render_recall_candidates(trace, recall, max_candidates),
        "",
        "## 4. Reranker 重排",
        "",
        render_reranker(reranker, max_rerank),
        "",
        "## 5. Evidence 给最终模型看的证据选择",
        "",
        render_evidence(trace),
        "",
        "## 6. Options",
        "",
        fenced_json(trace.get("options") or {}),
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_query_understanding(query_understanding: Mapping[str, Any]) -> str:
    analysis = as_mapping(query_understanding.get("analysis"))
    terms = as_sequence(analysis.get("query_terms"))
    doc_candidates = as_sequence(analysis.get("doc_candidates"))
    intent_candidates = as_sequence(analysis.get("intent_candidates"))

    lines = [
        markdown_kv_table(
            [
                ("strategy", query_understanding.get("strategy") or analysis.get("strategy")),
                ("language", query_understanding.get("language") or analysis.get("language")),
                ("analysis_query", analysis.get("query")),
                ("enabled_strategies", ", ".join(str(item) for item in as_sequence(query_understanding.get("enabled_strategies")))),
            ]
        )
    ]

    lines.append("")
    lines.append("### Query Terms")
    lines.append("")
    lines.append(markdown_table(["term", "source", "weight"], [
        [
            item.get("term"),
            item.get("source"),
            fmt_score(item.get("weight")),
        ]
        for item in map(as_mapping, terms)
    ]))

    if doc_candidates:
        lines.append("")
        lines.append("### Doc Candidates")
        lines.append("")
        lines.append(markdown_table(["doc_id", "score", "source", "reason"], [
            [item.get("doc_id"), fmt_score(item.get("score")), item.get("source"), item.get("reason")]
            for item in map(as_mapping, doc_candidates)
        ]))

    if intent_candidates:
        lines.append("")
        lines.append("### Intent Candidates")
        lines.append("")
        lines.append(markdown_table(["intent", "score", "source", "reason"], [
            [item.get("intent"), fmt_score(item.get("score")), item.get("source"), item.get("reason")]
            for item in map(as_mapping, intent_candidates)
        ]))

    return "\n".join(lines)


def render_recall_summary(recall: Mapping[str, Any]) -> str:
    channels = as_mapping(recall.get("channels"))
    recall_filter = as_mapping(recall.get("recall_filter"))
    rows = []
    for channel in ("vector", "bm25", "scan"):
        filter_info = as_mapping(recall_filter.get(channel))
        rows.append(
            [
                channel,
                len(as_sequence(channels.get(channel))),
                value_or_blank(filter_info.get("before")),
                value_or_blank(filter_info.get("after")),
                value_or_blank(filter_info.get("dropped")),
            ]
        )
    lines = [
        markdown_kv_table(
            [
                ("query", recall.get("query")),
                ("language", recall.get("language")),
                ("language_filter", recall.get("language_filter")),
                ("scan_triggered", recall.get("scan_triggered")),
            ]
        ),
        "",
        markdown_table(["channel", "raw_count", "filter_before", "filter_after", "dropped"], rows),
    ]
    routes = recall.get("effective_recall_routes")
    if routes:
        lines.extend(["", "### Effective Recall Routes", "", fenced_json(routes)])
    return "\n".join(lines)


def render_channel_hits(recall: Mapping[str, Any], max_channel_hits: int) -> str:
    channels = as_mapping(recall.get("channels"))
    lines = ["### 各通道 Top Hits", ""]
    for channel in ("vector", "bm25", "scan"):
        hits = as_sequence(channels.get(channel))
        lines.extend([f"#### {channel}", ""])
        lines.append(render_result_table(hits[:max_channel_hits], include_rank=True))
        lines.append("")
    return "\n".join(lines).rstrip()


def render_recall_candidates(trace: Mapping[str, Any], recall: Mapping[str, Any], max_candidates: int) -> str:
    candidates = trace_recall_candidates(trace, recall)[:max_candidates]
    rows = []
    for rank, candidate in enumerate(map(as_mapping, candidates), 1):
        result = as_mapping(candidate.get("result"))
        rows.append(
            [
                rank,
                candidate.get("chunk_id"),
                ", ".join(str(item) for item in as_sequence(candidate.get("recall_channels"))),
                compact_json(candidate.get("channel_scores")),
                fmt_score(candidate.get("merged_score")),
                result.get("doc_id"),
                result.get("language"),
                result.get("retrieval_tier"),
                result.get("chunk_type"),
                fmt_score(result.get("score")),
            ]
        )
    return "\n".join(
        [
            "### 合并后的 RecallCandidate",
            "",
            markdown_table(
                [
                    "rank",
                    "chunk_id",
                    "channels",
                    "channel_scores",
                    "merged_score",
                    "doc_id",
                    "lang",
                    "tier",
                    "chunk_type",
                    "raw_score",
                ],
                rows,
            ),
        ]
    )


def render_reranker(reranker: Mapping[str, Any], max_rerank: int) -> str:
    lines = [
        markdown_kv_table(
            [
                ("provider", reranker.get("provider")),
                ("model", reranker.get("model")),
                ("fallback_used", reranker.get("fallback_used")),
                ("input_count", reranker.get("input_count")),
                ("top_n", reranker.get("top_n")),
                ("timeout_ms", reranker.get("timeout_ms")),
            ]
        ),
        "",
        "### Post Rank",
        "",
        render_rank_table(as_sequence(reranker.get("post_rank"))[:max_rerank]),
    ]
    rank_changes = as_sequence(reranker.get("rank_changes"))
    if rank_changes:
        lines.extend(["", "### Rank Changes", "", render_rank_changes(rank_changes[:max_rerank])])
    return "\n".join(lines)


def render_evidence(trace: Mapping[str, Any]) -> str:
    primary_hits = as_sequence(trace.get("evidence_primary_hits"))
    support_hits = as_sequence(trace.get("evidence_support_hits"))
    evidence_images = as_mapping(trace.get("evidence_images"))
    support_parent_request = trace.get("support_parent_request")
    support_parent_ids = trace.get("support_parent_ids")
    support_parent_hits = trace.get("support_parent_hits")
    big_support_parent_ids = trace.get("big_support_parent_ids")
    big_support_descendant_count = trace.get("big_support_descendant_count")
    big_support_expanded_hits = as_sequence(trace.get("big_support_expanded_hits"))
    big_support_rerank = as_mapping(trace.get("big_support_rerank"))
    big_support_selected_parent_ids = trace.get("big_support_selected_parent_ids")
    big_support_selected_parent_hits = trace.get("big_support_selected_parent_hits")

    lines = [
        "### Primary Hits",
        "",
        render_result_table(primary_hits, include_rank=True),
        "",
        "### Support Hits",
        "",
        render_result_table(support_hits, include_rank=True),
        "",
        "### Images",
        "",
        markdown_kv_table(
            [
                ("pic_ids", ", ".join(str(item) for item in as_sequence(evidence_images.get("pic_ids")))),
                ("image_paths", "\n".join(str(item) for item in as_sequence(evidence_images.get("image_paths")))),
            ]
        ),
    ]
    image_hits = as_sequence(evidence_images.get("hits"))
    if image_hits:
        lines.extend(
            [
                "",
                "#### Image Hit Trace",
                "",
                markdown_table(
                    ["role", "chunk_id", "pic_ids", "image_paths"],
                    [
                        [
                            as_mapping(item).get("role"),
                            as_mapping(item).get("chunk_id"),
                            ", ".join(str(pic) for pic in as_sequence(as_mapping(item).get("pic_ids"))),
                            "\n".join(str(path) for path in as_sequence(as_mapping(item).get("image_paths"))),
                        ]
                        for item in image_hits
                    ],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "### Support Parent Trace",
            "",
            markdown_kv_table(
                [
                    ("support_parent_ids", compact_json(support_parent_ids)),
                    ("support_parent_request", compact_json(support_parent_request)),
                    ("support_parent_hits", compact_json(support_parent_hits)),
                ]
            ),
        ]
    )
    if any(
        value not in (None, "", [], {})
        for value in (
            big_support_parent_ids,
            big_support_descendant_count,
            big_support_expanded_hits,
            big_support_rerank,
            big_support_selected_parent_ids,
            big_support_selected_parent_hits,
        )
    ):
        lines.extend(
            [
                "",
                "### Big Support Trace",
                "",
                markdown_kv_table(
                    [
                        ("big_support_parent_ids", compact_json(big_support_parent_ids)),
                        ("big_support_descendant_count", compact_json(big_support_descendant_count)),
                        ("big_support_selected_parent_ids", compact_json(big_support_selected_parent_ids)),
                    ]
                ),
                "",
                "#### Big Support Expanded Hits",
                "",
                render_big_support_expanded_hits(big_support_expanded_hits),
            ]
        )
        if big_support_selected_parent_hits:
            lines.extend(
                [
                    "",
                    "#### Selected Support Parents",
                    "",
                    render_result_table(as_sequence(big_support_selected_parent_hits), include_rank=True),
                ]
            )
        if big_support_rerank:
            lines.extend(
                [
                    "",
                    "#### Big Support Rerank",
                    "",
                    render_big_support_rerank(big_support_rerank),
                ]
            )
    return "\n".join(lines)


def render_big_support_expanded_hits(expanded_hits: Sequence[Any]) -> str:
    rows = []
    for item in map(as_mapping, expanded_hits):
        rows.append(
            [
                item.get("parent_id"),
                value_or_blank(item.get("descendant_count")),
                value_or_blank(item.get("max_candidates")),
                value_or_blank(item.get("truncated")),
                ", ".join(
                    str(as_mapping(result).get("chunk_id") or as_mapping(result).get("id"))
                    for result in as_sequence(item.get("selected"))
                ),
                ", ".join(
                    str(as_mapping(result).get("chunk_id") or as_mapping(result).get("id"))
                    for result in as_sequence(item.get("direct_support_parents"))
                ),
            ]
        )
    return markdown_table(
        [
            "parent_id",
            "descendant_count",
            "max_candidates",
            "truncated",
            "selected_primary_chunks",
            "direct_support_parents",
        ],
        rows,
    )


def render_big_support_rerank(big_support_rerank: Mapping[str, Any]) -> str:
    lines = []
    for parent_id, rerank_info in big_support_rerank.items():
        info = as_mapping(rerank_info)
        lines.extend(
            [
                f"##### Parent `{parent_id}`",
                "",
                markdown_kv_table(
                    [
                        ("provider", info.get("provider")),
                        ("model", info.get("model")),
                        ("fallback_used", info.get("fallback_used")),
                        ("input_count", info.get("input_count")),
                        ("top_n", info.get("top_n")),
                        ("timeout_ms", info.get("timeout_ms")),
                    ]
                ),
                "",
                render_rank_table(as_sequence(info.get("post_rank"))),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def render_result_table(results: Sequence[Any], *, include_rank: bool = False) -> str:
    headers = [
        "rank",
        "chunk_id",
        "doc_id",
        "lang",
        "tier",
        "chunk_type",
        "parent",
        "score",
        "pic_ids",
        "image_paths",
    ] if include_rank else [
        "chunk_id",
        "doc_id",
        "lang",
        "tier",
        "chunk_type",
        "parent",
        "score",
        "pic_ids",
        "image_paths",
    ]
    rows = []
    for rank, raw_item in enumerate(results, 1):
        item = as_mapping(raw_item)
        row = [
            item.get("chunk_id") or item.get("id"),
            item.get("doc_id"),
            item.get("language"),
            item.get("retrieval_tier"),
            item.get("chunk_type"),
            item.get("parent_chunk_id"),
            fmt_score(item.get("score")),
            ", ".join(str(pic) for pic in as_sequence(item.get("pic_ids"))),
            "\n".join(str(path) for path in as_sequence(item.get("image_paths"))),
        ]
        if include_rank:
            row.insert(0, rank)
        rows.append(row)
    return markdown_table(headers, rows)


def render_rank_table(items: Sequence[Any]) -> str:
    return markdown_table(
        ["rank", "chunk_id", "reranker_score", "merged_score", "lexical_score"],
        [
            [
                as_mapping(item).get("rank"),
                as_mapping(item).get("chunk_id"),
                fmt_score(as_mapping(item).get("reranker_score")),
                fmt_score(as_mapping(item).get("merged_score")),
                fmt_score(as_mapping(item).get("lexical_score")),
            ]
            for item in items
        ],
    )


def render_rank_changes(items: Sequence[Any]) -> str:
    return markdown_table(
        ["chunk_id", "before_rank", "after_rank", "delta"],
        [
            [
                as_mapping(item).get("chunk_id"),
                value_or_blank(as_mapping(item).get("before_rank")),
                value_or_blank(as_mapping(item).get("after_rank")),
                value_or_blank(as_mapping(item).get("delta")),
            ]
            for item in items
        ],
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_kv_table(items: Sequence[tuple[str, Any]]) -> str:
    return markdown_table(["字段", "值"], [[key, value_or_blank(value)] for key, value in items])


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_无_"
    lines = [
        "| " + " | ".join(escape_md(str(header)) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = list(row) + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(escape_md(format_cell(value)) for value in padded[: len(headers)]) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    text = value_or_blank(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def value_or_blank(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return fmt_score(value)
    return str(value)


def fmt_score(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def compact_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def fenced_json(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def shorten(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 1, 0)] + "…"


def escape_md(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|")


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def trace_recall_candidates(trace: Mapping[str, Any], recall: Mapping[str, Any]) -> list[Any]:
    candidates = as_sequence(recall.get("recall_candidates"))
    if candidates:
        return candidates
    return as_sequence(trace.get("recall_candidates"))


def image_count(trace: Mapping[str, Any]) -> int:
    evidence_images = as_mapping(trace.get("evidence_images"))
    pic_ids = as_sequence(evidence_images.get("pic_ids"))
    if pic_ids:
        return len(pic_ids)
    return len(as_sequence(evidence_images.get("image_paths")))


if __name__ == "__main__":
    main()
