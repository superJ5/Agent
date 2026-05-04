"""Build a Step 7 retrieval evidence baseline from the current Milvus index."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.tools.knowledge_tool import bundle_to_evidence_payload, routed_retrieve


DEFAULT_OUTPUT_DIR = Path("./data/manuals/retrieval_baselines/step7_evidence_v1")

BASELINE_CASES: list[dict[str, Any]] = [
    {
        "case_id": "kbd_wrist_rest_install",
        "query": "腕托怎么安装",
        "expected_chunk_ids": ["kbd_sec5_step_b"],
        "expected_pic_ids": [],
        "intent": "procedure",
    },
    {
        "case_id": "kbd_pic_manual21_8",
        "query": "功能键盘图片 Manual21_8 对应什么",
        "expected_chunk_ids": ["kbd_meta_Manual21_8"],
        "expected_pic_ids": ["Manual21_8"],
        "intent": "image_trace",
    },
    {
        "case_id": "vr_back_tracking_light_status",
        "query": "VR头显后侧追踪灯状态怎么看",
        "expected_chunk_ids": ["vr_headset_back_status"],
        "expected_pic_ids": [],
        "intent": "procedure",
    },
    {
        "case_id": "vr_uk_ocr_issue",
        "query": "致英国用户这段OCR有什么问题",
        "expected_chunk_ids": ["vr_warning_uk"],
        "expected_pic_ids": [],
        "intent": "ocr_audit",
    },
    {
        "case_id": "kbd_profile_switch",
        "query": "功能键盘怎么切换配置文件",
        "expected_chunk_ids": ["kbd_sec7_c", "kbd_sec4_group_5_8"],
        "expected_pic_ids": [],
        "intent": "procedure",
    },
    {
        "case_id": "kbd_warranty_scope",
        "query": "功能键盘保修范围是什么",
        "expected_chunk_ids": ["kbd_sec9_2_iii", "kbd_sec9_2_overview"],
        "expected_pic_ids": [],
        "intent": "legal",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persist the current retrieval evidence output as a Step 7 baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where baseline artifacts will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case in BASELINE_CASES:
        bundle = routed_retrieve(case["query"])
        evidence = bundle_to_evidence_payload(bundle, query=case["query"])
        evaluation = evaluate_case(case, evidence)
        rows.append(
            {
                "case": case,
                "evaluation": evaluation,
                "evidence": evidence,
            }
        )

    write_jsonl(output_dir / "cases.jsonl", BASELINE_CASES)
    write_jsonl(output_dir / "evidence_results.jsonl", rows)
    (output_dir / "evidence_report.md").write_text(
        render_markdown(rows),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        render_readme(output_dir),
        encoding="utf-8",
    )

    summary = {
        "output_dir": str(output_dir),
        "case_count": len(rows),
        "hit_at_1": sum(1 for row in rows if row["evaluation"]["hit_at_1"]),
        "hit_at_3": sum(1 for row in rows if row["evaluation"]["hit_at_3"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_case(case: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    expected_chunk_ids = set(case.get("expected_chunk_ids") or [])
    expected_pic_ids = set(case.get("expected_pic_ids") or [])
    hits = evidence.get("hits") or []
    all_hits = [*hits, *(evidence.get("support_hits") or [])]
    ranked_chunk_ids = [hit.get("chunk_id") for hit in hits]
    all_chunk_ids = [hit.get("chunk_id") for hit in all_hits]
    all_pic_ids = set(evidence.get("summary", {}).get("pic_ids") or [])

    return {
        "intent_match": evidence.get("intent") == case.get("intent"),
        "hit_at_1": contains_any(ranked_chunk_ids[:1], expected_chunk_ids),
        "hit_at_3": contains_any(ranked_chunk_ids[:3], expected_chunk_ids),
        "hit_anywhere": contains_any(all_chunk_ids, expected_chunk_ids),
        "pic_hit": not expected_pic_ids or bool(expected_pic_ids & all_pic_ids),
        "expected_chunk_ids": sorted(expected_chunk_ids),
        "actual_top_chunk_ids": ranked_chunk_ids[:5],
        "expected_pic_ids": sorted(expected_pic_ids),
        "actual_pic_ids": sorted(all_pic_ids),
    }


def contains_any(values: list[Any], expected: set[str]) -> bool:
    if not expected:
        return False
    return any(value in expected for value in values)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Step 7 Retrieval Evidence Baseline",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "| Case | Query | Intent | H@1 | H@3 | Pic | Top Evidence |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        case = row["case"]
        evidence = row["evidence"]
        evaluation = row["evaluation"]
        top_hit = (evidence.get("hits") or [{}])[0]
        lines.append(
            "| {case_id} | {query} | {intent} | {h1} | {h3} | {pic} | {top} |".format(
                case_id=escape(case["case_id"]),
                query=escape(case["query"]),
                intent=escape(str(evidence.get("intent"))),
                h1=evaluation["hit_at_1"],
                h3=evaluation["hit_at_3"],
                pic=evaluation["pic_hit"],
                top=escape(f"{top_hit.get('chunk_id')} / {top_hit.get('title')}"),
            )
        )

    lines.extend(["", "## Details"])
    for row in rows:
        case = row["case"]
        evidence = row["evidence"]
        evaluation = row["evaluation"]
        lines.extend(
            [
                "",
                f"### {case['case_id']}",
                "",
                f"- Query: `{case['query']}`",
                f"- Intent: `{evidence.get('intent')}`",
                f"- Stage: `{evidence.get('retrieval_stage')}`",
                f"- Expected chunks: `{evaluation['expected_chunk_ids']}`",
                f"- Actual top chunks: `{evaluation['actual_top_chunk_ids']}`",
                f"- Warnings: `{evidence.get('warnings') or []}`",
                "",
                "Primary evidence:",
            ]
        )
        for hit in evidence.get("hits") or []:
            lines.append(
                "- `{chunk_id}` | {title} | {tier}/{typ} | pics={pics}".format(
                    chunk_id=hit.get("chunk_id"),
                    title=hit.get("title"),
                    tier=hit.get("retrieval_tier"),
                    typ=hit.get("chunk_type"),
                    pics=hit.get("pic_ids") or [],
                )
            )
        if evidence.get("support_hits"):
            lines.append("")
            lines.append("Support evidence:")
            for hit in evidence.get("support_hits") or []:
                lines.append(
                    "- `{chunk_id}` | {title} | {tier}/{typ}".format(
                        chunk_id=hit.get("chunk_id"),
                        title=hit.get("title"),
                        tier=hit.get("retrieval_tier"),
                        typ=hit.get("chunk_type"),
                    )
                )

    lines.append("")
    return "\n".join(lines)


def render_readme(output_dir: Path) -> str:
    return "\n".join(
        [
            "# Step 7 Evidence Baseline",
            "",
            "This directory stores the current retrieval evidence output as a baseline for the two reviewed manuals.",
            "",
            "Files:",
            "",
            "- `cases.jsonl`: baseline test cases and expected chunk/image ids.",
            "- `evidence_results.jsonl`: full structured evidence payload per query.",
            "- `evidence_report.md`: human-readable summary and hit checks.",
            "",
            "Schema:",
            "",
            "- `schema_version`: evidence payload version.",
            "- `intent`: routed query intent.",
            "- `retrieval_stage`: final retrieval stage used.",
            "- `hits`: primary evidence chunks sent to the upper agent.",
            "- `support_hits`: parent/support chunks attached for context.",
            "- `warnings`: source-quality and fallback notes.",
            "- `summary.pic_ids`: image ids bound to the returned evidence.",
            "",
            f"Path: `{output_dir}`",
            "",
        ]
    )


def escape(text: str) -> str:
    return text.replace("|", "\\|")


if __name__ == "__main__":
    main()
