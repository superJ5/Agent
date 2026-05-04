"""Batch convert aligned manual datasets into structured JSONL files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from convert_manual_dataset import build_output_record, slugify


RAW_ROOT = Path("data/manuals/raw")
PARSED_ROOT = Path("data/manuals/parsed")
ORGANIZATION_REPORT = RAW_ROOT / "organization_report.json"


@dataclass
class ConversionReportItem:
    manual_name: str
    txt_path: str
    output_path: str
    doc_id: str
    pic_placeholder_count: int
    image_id_count: int
    resolved_image_count: int
    unresolved_image_ids: list[str]
    alignment_ok: bool


def main() -> None:
    if not ORGANIZATION_REPORT.exists():
        raise FileNotFoundError(f"organization report not found: {ORGANIZATION_REPORT}")

    payload = json.loads(ORGANIZATION_REPORT.read_text(encoding="utf-8"))
    PARSED_ROOT.mkdir(parents=True, exist_ok=True)

    converted: list[ConversionReportItem] = []
    skipped: list[dict[str, str]] = []

    for item in payload.get("manuals", []):
        manual_dir = Path(item["target_dir"])
        manual_name = manual_dir.name

        if not item.get("alignment_ok", False):
            skipped.append(
                {
                    "manual_name": manual_name,
                    "reason": "alignment_ok=false",
                }
            )
            continue

        txt_candidates = list(manual_dir.glob("*.txt"))
        image_dir = manual_dir / "images"
        if not txt_candidates:
            skipped.append(
                {
                    "manual_name": manual_name,
                    "reason": "txt file not found",
                }
            )
            continue
        if not image_dir.exists():
            skipped.append(
                {
                    "manual_name": manual_name,
                    "reason": "images directory not found",
                }
            )
            continue

        txt_path = txt_candidates[0]
        doc_id = slugify(manual_name)
        output_path = PARSED_ROOT / f"{manual_name}.jsonl"
        record = build_output_record(txt_path.resolve(), image_dir.resolve(), doc_id)
        output_path.write_text(
            json.dumps(record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        unresolved_image_ids = [
            ref["image_id"] for ref in record["pic_refs"] if ref["image_path"] is None
        ]
        resolved_image_count = sum(1 for ref in record["pic_refs"] if ref["image_path"] is not None)

        converted.append(
            ConversionReportItem(
                manual_name=manual_name,
                txt_path=str(txt_path),
                output_path=str(output_path),
                doc_id=doc_id,
                pic_placeholder_count=record["pic_placeholder_count"],
                image_id_count=record["image_id_count"],
                resolved_image_count=resolved_image_count,
                unresolved_image_ids=unresolved_image_ids,
                alignment_ok=record["alignment_ok"],
            )
        )

    report = {
        "converted_count": len(converted),
        "skipped_count": len(skipped),
        "converted": [asdict(item) for item in converted],
        "skipped": skipped,
    }
    report_path = PARSED_ROOT / "conversion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"converted manuals: {len(converted)}")
    print(f"skipped manuals: {len(skipped)}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
