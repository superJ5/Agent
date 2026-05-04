"""Compare structured chunks against the simple baseline chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.services.vector_search_service import SearchResult, vector_search_service


DEFAULT_OUTPUT = Path("./data/manuals/baseline_chunks/baseline_comparison_report.md")

TEST_CASES = [
    {
        "query": "腕托怎么安装",
        "structured_doc_id": "manual_7f829388",
        "baseline_doc_id": "manual_7f829388_baseline",
        "expected_terms": ["腕托", "安装"],
        "structured_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["procedure_step", "feature_group", "subsection"],
        },
        "baseline_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["baseline_simple_chunk"],
        },
    },
    {
        "query": "VR头显后侧追踪灯状态怎么看",
        "structured_doc_id": "vr_bfe50169",
        "baseline_doc_id": "vr_bfe50169_baseline",
        "expected_terms": ["后侧追踪灯", "蓝色"],
        "structured_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["subsection", "feature_group"],
        },
        "baseline_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["baseline_simple_chunk"],
        },
    },
    {
        "query": "功能键盘图片 Manual21_8 对应什么",
        "structured_doc_id": "manual_7f829388",
        "baseline_doc_id": "manual_7f829388_baseline",
        "expected_terms": ["Manual21_8"],
        "structured_filter": {
            "retrieval_tiers": ["auxiliary"],
            "chunk_types": ["metadata_image_path"],
        },
        "baseline_filter": {
            "retrieval_tiers": ["auxiliary"],
            "chunk_types": ["metadata_image_path"],
        },
    },
    {
        "query": "致英国用户这段OCR有什么问题",
        "structured_doc_id": "vr_bfe50169",
        "baseline_doc_id": "vr_bfe50169_baseline",
        "expected_terms": ["致英国用户", "OCR"],
        "structured_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["subsection"],
        },
        "baseline_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["baseline_simple_chunk"],
        },
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small retrieval comparison between structured and baseline chunks.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case in TEST_CASES:
        structured = run_case(
            query=case["query"],
            doc_id=case["structured_doc_id"],
            top_k=args.top_k,
            expected_terms=case["expected_terms"],
            filters=case["structured_filter"],
        )
        baseline = run_case(
            query=case["query"],
            doc_id=case["baseline_doc_id"],
            top_k=args.top_k,
            expected_terms=case["expected_terms"],
            filters=case["baseline_filter"],
        )
        rows.append(
            {
                "query": case["query"],
                "expected_terms": case["expected_terms"],
                "structured": structured,
                "baseline": baseline,
            }
        )

    output_path.write_text(render_markdown(rows), encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output_path), "json": str(json_path)}, ensure_ascii=False, indent=2))


def run_case(
    query: str,
    doc_id: str,
    top_k: int,
    expected_terms: list[str],
    filters: dict[str, Any],
) -> dict[str, Any]:
    results = vector_search_service.search_similar_documents(
        query=query,
        top_k=top_k,
        doc_id=doc_id,
        retrieval_tiers=filters.get("retrieval_tiers"),
        chunk_types=filters.get("chunk_types"),
    )
    rendered = [summarize_result(result) for result in results]
    return {
        "doc_id": doc_id,
        "hit_at_1": contains_expected(results[:1], expected_terms),
        "hit_at_3": contains_expected(results[:3], expected_terms),
        "top_results": rendered,
    }


def summarize_result(result: SearchResult) -> dict[str, Any]:
    metadata = result.metadata or {}
    return {
        "chunk_id": metadata.get("chunk_id") or result.id,
        "score": result.score,
        "title": metadata.get("title"),
        "tier": metadata.get("retrieval_tier"),
        "type": metadata.get("chunk_type"),
        "pic_ids": metadata.get("pic_ids") or [],
        "image_paths": metadata.get("image_paths") or [],
    }


def contains_expected(results: list[SearchResult], expected_terms: list[str]) -> bool:
    if not expected_terms:
        return False
    for result in results:
        metadata = result.metadata or {}
        haystack = " ".join(
            [
                str(metadata.get("title") or ""),
                str(metadata.get("index_text") or ""),
                str(metadata.get("text") or result.content or ""),
                " ".join(str(pic_id) for pic_id in metadata.get("pic_ids") or []),
            ]
        ).lower()
        if all(term.lower() in haystack for term in expected_terms):
            return True
    return False


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Baseline Retrieval Comparison",
        "",
        "This report compares the reviewed structured chunks with the simple character-based baseline chunks.",
        "",
        "| Query | Expected | Structured H@1 / H@3 | Baseline H@1 / H@3 | Structured Top1 | Baseline Top1 |",
        "|---|---|---|---|---|---|",
    ]

    for row in rows:
        structured = row["structured"]
        baseline = row["baseline"]
        structured_top1 = top_title(structured)
        baseline_top1 = top_title(baseline)
        lines.append(
            "| {query} | {expected} | {s1} / {s3} | {b1} / {b3} | {stop} | {btop} |".format(
                query=escape(row["query"]),
                expected=escape(", ".join(row["expected_terms"])),
                s1=structured["hit_at_1"],
                s3=structured["hit_at_3"],
                b1=baseline["hit_at_1"],
                b3=baseline["hit_at_3"],
                stop=escape(structured_top1),
                btop=escape(baseline_top1),
            )
        )

    lines.append("")
    lines.append("## Details")
    for row in rows:
        lines.append("")
        lines.append(f"### {row['query']}")
        for label in ("structured", "baseline"):
            lines.append("")
            lines.append(f"#### {label}")
            for item in row[label]["top_results"]:
                lines.append(
                    "- `{chunk_id}` | {title} | {tier}/{typ} | pic_ids={pic_ids}".format(
                        chunk_id=item["chunk_id"],
                        title=item["title"],
                        tier=item["tier"],
                        typ=item["type"],
                        pic_ids=item["pic_ids"],
                    )
                )
    lines.append("")
    return "\n".join(lines)


def top_title(result_group: dict[str, Any]) -> str:
    top_results = result_group["top_results"]
    if not top_results:
        return ""
    top = top_results[0]
    return f"{top['chunk_id']} / {top['title']}"


def escape(text: str) -> str:
    return text.replace("|", "\\|")


if __name__ == "__main__":
    main()
