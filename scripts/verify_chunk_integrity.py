"""Verify that chunked manual files preserve source content and image refs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PARSED_ROOT = Path("data/manuals/parsed")
CHUNK_ROOT = Path("data/manuals/chunks")


@dataclass
class IntegrityReportItem:
    manual_name: str
    parsed_path: str
    chunk_path: str
    raw_text_match: bool
    raw_text_match_loose: bool
    clean_text_match: bool
    clean_text_match_loose: bool
    pic_count_match: bool
    pic_sequence_match: bool
    content_preserved: bool
    source_raw_pic_count: int
    chunk_raw_pic_count: int
    source_pic_ref_count: int
    chunk_pic_ref_count: int
    source_clean_chars: int
    chunk_clean_chars: int
    first_raw_diff: dict[str, Any] | None
    first_clean_diff: dict[str, Any] | None
    first_pic_diff: dict[str, Any] | None


def load_jsonl_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def load_chunk_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_compare_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_compare_text_loose(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def first_diff(left: str, right: str, window: int = 40) -> dict[str, Any] | None:
    if left == right:
        return None

    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return {
                "index": idx,
                "left_excerpt": left[max(0, idx - window) : idx + window],
                "right_excerpt": right[max(0, idx - window) : idx + window],
            }

    return {
        "index": limit,
        "left_excerpt": left[max(0, limit - window) : limit + window],
        "right_excerpt": right[max(0, limit - window) : limit + window],
    }


def first_pic_diff(source_ids: list[str], chunk_ids: list[str]) -> dict[str, Any] | None:
    if source_ids == chunk_ids:
        return None

    limit = min(len(source_ids), len(chunk_ids))
    for idx in range(limit):
        if source_ids[idx] != chunk_ids[idx]:
            return {
                "index": idx,
                "source_image_id": source_ids[idx],
                "chunk_image_id": chunk_ids[idx],
            }

    return {
        "index": limit,
        "source_image_id": source_ids[limit] if limit < len(source_ids) else None,
        "chunk_image_id": chunk_ids[limit] if limit < len(chunk_ids) else None,
    }


def main() -> None:
    reports: list[IntegrityReportItem] = []
    skipped: list[dict[str, str]] = []

    parsed_files = sorted(PARSED_ROOT.glob("*.jsonl"))
    chunk_files = {path.name: path for path in CHUNK_ROOT.glob("*.jsonl")}

    for parsed_path in parsed_files:
        chunk_path = chunk_files.get(parsed_path.name)
        if chunk_path is None:
            skipped.append(
                {
                    "manual_name": parsed_path.stem,
                    "reason": "chunk file not found",
                }
            )
            continue

        parsed_record = load_jsonl_record(parsed_path)
        chunk_records = load_chunk_records(chunk_path)

        source_raw = normalize_compare_text(parsed_record["raw_text"])
        chunk_raw = normalize_compare_text("\n".join(item["raw_text"] for item in chunk_records))
        source_raw_loose = normalize_compare_text_loose(parsed_record["raw_text"])
        chunk_raw_loose = normalize_compare_text_loose(
            "\n".join(item["raw_text"] for item in chunk_records)
        )

        source_clean = normalize_compare_text(parsed_record["clean_text"])
        chunk_clean = normalize_compare_text("\n".join(item["text"] for item in chunk_records))
        source_clean_loose = normalize_compare_text_loose(parsed_record["clean_text"])
        chunk_clean_loose = normalize_compare_text_loose("\n".join(item["text"] for item in chunk_records))

        source_pic_ids = [item["image_id"] for item in parsed_record["pic_refs"]]
        chunk_pic_ids = [
            pic_ref["image_id"] for item in chunk_records for pic_ref in item.get("pic_refs", [])
        ]

        raw_text_match = source_raw == chunk_raw
        raw_text_match_loose = source_raw_loose == chunk_raw_loose
        clean_text_match = source_clean == chunk_clean
        clean_text_match_loose = source_clean_loose == chunk_clean_loose
        pic_count_match = parsed_record["pic_placeholder_count"] == sum(
            item["raw_text"].count("<PIC>") for item in chunk_records
        )
        pic_sequence_match = source_pic_ids == chunk_pic_ids
        content_preserved = (
            raw_text_match_loose and clean_text_match_loose and pic_count_match and pic_sequence_match
        )

        reports.append(
            IntegrityReportItem(
                manual_name=parsed_path.stem,
                parsed_path=str(parsed_path),
                chunk_path=str(chunk_path),
                raw_text_match=raw_text_match,
                raw_text_match_loose=raw_text_match_loose,
                clean_text_match=clean_text_match,
                clean_text_match_loose=clean_text_match_loose,
                pic_count_match=pic_count_match,
                pic_sequence_match=pic_sequence_match,
                content_preserved=content_preserved,
                source_raw_pic_count=parsed_record["raw_text"].count("<PIC>"),
                chunk_raw_pic_count=sum(item["raw_text"].count("<PIC>") for item in chunk_records),
                source_pic_ref_count=len(source_pic_ids),
                chunk_pic_ref_count=len(chunk_pic_ids),
                source_clean_chars=len(parsed_record["clean_text"]),
                chunk_clean_chars=sum(len(item["text"]) for item in chunk_records),
                first_raw_diff=first_diff(source_raw, chunk_raw),
                first_clean_diff=first_diff(source_clean, chunk_clean),
                first_pic_diff=first_pic_diff(source_pic_ids, chunk_pic_ids),
            )
        )

    extra_chunk_files = [
        {"chunk_file": str(path), "reason": "parsed source not found"}
        for name, path in chunk_files.items()
        if not (PARSED_ROOT / name).exists()
    ]

    summary = {
        "checked_count": len(reports),
        "skipped_count": len(skipped),
        "strict_passed_count": sum(
            1
            for item in reports
            if item.raw_text_match
            and item.clean_text_match
            and item.pic_count_match
            and item.pic_sequence_match
        ),
        "strict_failed_count": sum(
            1
            for item in reports
            if not (
                item.raw_text_match
                and item.clean_text_match
                and item.pic_count_match
                and item.pic_sequence_match
            )
        ),
        "content_preserved_count": sum(1 for item in reports if item.content_preserved),
        "content_suspect_count": sum(1 for item in reports if not item.content_preserved),
        "reports": [asdict(item) for item in reports],
        "skipped": skipped,
        "extra_chunk_files": extra_chunk_files,
    }

    report_path = CHUNK_ROOT / "chunk_integrity_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"checked manuals: {summary['checked_count']}")
    print(f"strict passed: {summary['strict_passed_count']}")
    print(f"strict failed: {summary['strict_failed_count']}")
    print(f"content preserved: {summary['content_preserved_count']}")
    print(f"content suspect: {summary['content_suspect_count']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
