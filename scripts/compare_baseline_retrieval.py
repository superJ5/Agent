"""Compare structured chunks against simple character-based baseline chunks.

This benchmark focuses on chunk quality at retrieval time. It intentionally uses
the same vector search backend for both corpora and constrains the structured
side with task-relevant tier/type filters, so the comparison stays centered on
"how well does this chunking strategy expose the right evidence?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.services.vector_search_service import SearchResult, vector_search_service


DEFAULT_OUTPUT = Path("./data/manuals/baseline_chunks/baseline_comparison_report.md")

TEST_CASES: list[dict[str, Any]] = [
    {
        "case_id": "kbd_wrist_rest_install",
        "query": "\u8155\u6258\u600e\u4e48\u5b89\u88c5",
        "structured_doc_id": "manual_7f829388",
        "baseline_doc_id": "manual_7f829388_baseline",
        "expected_terms": ["\u8155\u6258", "\u5b89\u88c5"],
        "expected_pic_ids": [],
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
        "case_id": "vr_back_tracking_light_status",
        "query": "VR\u5934\u663e\u540e\u4fa7\u8ffd\u8e2a\u706f\u72b6\u6001\u600e\u4e48\u770b",
        "structured_doc_id": "vr_bfe50169",
        "baseline_doc_id": "vr_bfe50169_baseline",
        "expected_terms": ["\u540e\u4fa7\u8ffd\u8e2a\u706f", "\u84dd\u8272"],
        "expected_pic_ids": [],
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
        "case_id": "kbd_pic_manual21_8",
        "query": "\u529f\u80fd\u952e\u76d8\u56fe\u7247 Manual21_8 \u5bf9\u5e94\u4ec0\u4e48",
        "structured_doc_id": "manual_7f829388",
        "baseline_doc_id": "manual_7f829388_baseline",
        "expected_terms": ["Manual21_8"],
        "expected_pic_ids": ["Manual21_8"],
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
        "case_id": "vr_uk_ocr_issue",
        "query": "\u81f4\u82f1\u56fd\u7528\u6237\u8fd9\u6bb5OCR\u6709\u4ec0\u4e48\u95ee\u9898",
        "structured_doc_id": "vr_bfe50169",
        "baseline_doc_id": "vr_bfe50169_baseline",
        "expected_terms": ["\u81f4\u82f1\u56fd\u7528\u6237", "ocr"],
        "expected_pic_ids": [],
        "structured_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["subsection"],
        },
        "baseline_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["baseline_simple_chunk"],
        },
    },
    {
        "case_id": "fridge_lamp_replace",
        "query": "\u51b0\u7bb1\u5185\u90e8\u706f\u600e\u4e48\u66f4\u6362",
        "structured_doc_id": "manual_9199cd53",
        "baseline_doc_id": "manual_9199cd53_baseline",
        "expected_terms": ["\u5185\u90e8\u706f", "\u66f4\u6362"],
        "expected_pic_ids": [],
        "structured_filter": {
            "retrieval_tiers": ["primary", "support"],
            "chunk_types": ["procedure_step", "procedure_overview", "subsection"],
        },
        "baseline_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["baseline_simple_chunk"],
        },
    },
    {
        "case_id": "fridge_power_safety",
        "query": "\u51b0\u7bb1\u8fde\u63a5\u7535\u6e90\u65f6\u8981\u6ce8\u610f\u4ec0\u4e48",
        "structured_doc_id": "manual_9199cd53",
        "baseline_doc_id": "manual_9199cd53_baseline",
        "expected_terms": ["\u8fde\u63a5\u7535\u6e90\u65f6"],
        "expected_pic_ids": [],
        "structured_filter": {
            "retrieval_tiers": ["primary", "support"],
            "chunk_types": ["safety_clause", "caution_clause", "section_summary"],
        },
        "baseline_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["baseline_simple_chunk"],
        },
    },
    {
        "case_id": "fridge_not_cooling_reason",
        "query": "\u51b0\u7bb1\u4e0d\u5236\u51b7\u53ef\u80fd\u662f\u4ec0\u4e48\u539f\u56e0",
        "structured_doc_id": "manual_9199cd53",
        "baseline_doc_id": "manual_9199cd53_baseline",
        "expected_terms": ["\u4e0d\u5236\u51b7", "\u539f\u56e0"],
        "expected_pic_ids": [],
        "structured_filter": {
            "retrieval_tiers": ["primary", "support"],
            "chunk_types": ["troubleshooting_case", "troubleshooting_condition", "section_summary"],
        },
        "baseline_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["baseline_simple_chunk"],
        },
    },
    {
        "case_id": "fridge_disposal",
        "query": "\u65e7\u7535\u5668\u5e9f\u5f03\u5904\u7406\u662f\u4ec0\u4e48",
        "structured_doc_id": "manual_9199cd53",
        "baseline_doc_id": "manual_9199cd53_baseline",
        "expected_terms": ["\u65e7\u7535\u5668\u5e9f\u5f03\u5904\u7406"],
        "expected_pic_ids": [],
        "structured_filter": {
            "retrieval_tiers": ["primary", "support"],
            "chunk_types": ["legal_clause", "section_summary", "subsection"],
        },
        "baseline_filter": {
            "retrieval_tiers": ["primary"],
            "chunk_types": ["baseline_simple_chunk"],
        },
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a retrieval comparison between structured and baseline chunks.",
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
            expected_pic_ids=case["expected_pic_ids"],
            filters=case["structured_filter"],
        )
        baseline = run_case(
            query=case["query"],
            doc_id=case["baseline_doc_id"],
            top_k=args.top_k,
            expected_terms=case["expected_terms"],
            expected_pic_ids=case["expected_pic_ids"],
            filters=case["baseline_filter"],
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "expected_terms": case["expected_terms"],
                "expected_pic_ids": case["expected_pic_ids"],
                "structured": structured,
                "baseline": baseline,
            }
        )

    summary = build_summary(rows)
    output_path.write_text(render_markdown(rows, summary), encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"report": str(output_path), "json": str(json_path), "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


def run_case(
    query: str,
    doc_id: str,
    top_k: int,
    expected_terms: list[str],
    expected_pic_ids: list[str],
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
        "hit_at_1": contains_expected(results[:1], expected_terms, expected_pic_ids),
        "hit_at_3": contains_expected(results[:3], expected_terms, expected_pic_ids),
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


def contains_expected(
    results: list[SearchResult],
    expected_terms: list[str],
    expected_pic_ids: list[str],
) -> bool:
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
        pic_ids = {str(pic_id) for pic_id in metadata.get("pic_ids") or []}
        term_ok = not expected_terms or all(term.lower() in haystack for term in expected_terms)
        pic_ok = not expected_pic_ids or bool(pic_ids & set(expected_pic_ids))
        if term_ok and pic_ok:
            return True
    return False


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "case_count": len(rows),
        "structured_hit_at_1": 0,
        "structured_hit_at_3": 0,
        "baseline_hit_at_1": 0,
        "baseline_hit_at_3": 0,
    }
    for row in rows:
        summary["structured_hit_at_1"] += int(row["structured"]["hit_at_1"])
        summary["structured_hit_at_3"] += int(row["structured"]["hit_at_3"])
        summary["baseline_hit_at_1"] += int(row["baseline"]["hit_at_1"])
        summary["baseline_hit_at_3"] += int(row["baseline"]["hit_at_3"])
    return summary


def render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Baseline Retrieval Comparison",
        "",
        "This report compares reviewed structured chunks against simple character-based baseline chunks.",
        "",
        "## Summary",
        "",
        f"- Case count: `{summary['case_count']}`",
        f"- Structured H@1: `{summary['structured_hit_at_1']}` / `{summary['case_count']}`",
        f"- Structured H@3: `{summary['structured_hit_at_3']}` / `{summary['case_count']}`",
        f"- Baseline H@1: `{summary['baseline_hit_at_1']}` / `{summary['case_count']}`",
        f"- Baseline H@3: `{summary['baseline_hit_at_3']}` / `{summary['case_count']}`",
        "",
        "| Case | Query | Expected | Structured H@1 / H@3 | Baseline H@1 / H@3 | Structured Top1 | Baseline Top1 |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        structured = row["structured"]
        baseline = row["baseline"]
        structured_top1 = top_title(structured)
        baseline_top1 = top_title(baseline)
        expected = ", ".join(row["expected_terms"] + row["expected_pic_ids"])
        lines.append(
            "| {case_id} | {query} | {expected} | {s1} / {s3} | {b1} / {b3} | {stop} | {btop} |".format(
                case_id=escape(row["case_id"]),
                query=escape(row["query"]),
                expected=escape(expected),
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
        lines.append(f"### {row['case_id']}: {row['query']}")
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
