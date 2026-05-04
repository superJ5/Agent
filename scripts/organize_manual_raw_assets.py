"""Organize manual txt files and images into per-manual folders.

The source import folder is kept intact. This script creates a structured
layout under ``data/manuals/raw`` using hard links when possible, falling
back to file copies only when linking is unavailable.
"""

from __future__ import annotations

import ast
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RAW_ROOT = Path("data/manuals/raw")
IMPORT_DIR_NAME = "手册"
IMAGE_DIR_NAME = "插图"
DEMO_DIR_NAMES = {"demo_images"}
SKIP_TXT_NAMES = {"汇总英文手册.txt"}


@dataclass
class ManualReport:
    manual_name: str
    txt_name: str
    pic_placeholder_count: int
    image_ref_count: int
    linked_image_count: int
    missing_image_ids: list[str]
    alignment_ok: bool
    target_dir: str


def parse_payload(raw_text: str) -> Any:
    raw_text = raw_text.lstrip("\ufeff").strip()

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(raw_text)
        except Exception:
            continue

    raise ValueError("unsupported txt record format")


def normalize_payload(payload: Any) -> tuple[str, list[str]]:
    if isinstance(payload, list) and len(payload) == 2:
        text, image_ids = payload
    elif isinstance(payload, dict):
        text = payload.get("text") or payload.get("content")
        image_ids = (
            payload.get("images")
            or payload.get("image_ids")
            or payload.get("pic_refs")
        )
    else:
        raise ValueError("unsupported payload shape")

    if not isinstance(text, str):
        raise ValueError("record text must be a string")
    if not isinstance(image_ids, list) or not all(isinstance(item, str) for item in image_ids):
        raise ValueError("record images must be a list[str]")

    return text, image_ids


def discover_import_root() -> Path:
    for path in RAW_ROOT.iterdir():
        if path.is_dir() and path.name not in DEMO_DIR_NAMES and path.name == IMPORT_DIR_NAME:
            return path
    raise FileNotFoundError(f"cannot find import directory under {RAW_ROOT}")


def build_image_index(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in image_root.iterdir():
        if image_path.is_file():
            index[image_path.stem] = image_path
    return index


def ensure_hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return

    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def organize_manual(
    txt_path: Path,
    image_index: dict[str, Path],
    destination_root: Path,
) -> ManualReport:
    manual_name = txt_path.stem
    manual_dir = destination_root / manual_name
    images_dir = manual_dir / "images"

    payload = parse_payload(txt_path.read_text(encoding="utf-8"))
    text, image_ids = normalize_payload(payload)
    unique_image_ids = list(dict.fromkeys(image_ids))

    ensure_hardlink_or_copy(txt_path, manual_dir / txt_path.name)

    linked_count = 0
    missing_image_ids: list[str] = []

    for image_id in unique_image_ids:
        source_image = image_index.get(image_id)
        if source_image is None:
            missing_image_ids.append(image_id)
            continue
        ensure_hardlink_or_copy(source_image, images_dir / source_image.name)
        linked_count += 1

    return ManualReport(
        manual_name=manual_name,
        txt_name=txt_path.name,
        pic_placeholder_count=text.count("<PIC>"),
        image_ref_count=len(image_ids),
        linked_image_count=linked_count,
        missing_image_ids=missing_image_ids,
        alignment_ok=text.count("<PIC>") == len(image_ids),
        target_dir=str(manual_dir),
    )


def write_report(report_path: Path, reports: list[ManualReport], skipped: list[dict[str, str]]) -> None:
    payload = {
        "organized_manual_count": len(reports),
        "skipped_count": len(skipped),
        "manuals": [asdict(item) for item in reports],
        "skipped": skipped,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    import_root = discover_import_root()
    image_root = import_root / IMAGE_DIR_NAME
    if not image_root.exists():
        raise FileNotFoundError(f"cannot find image directory under {import_root}")

    image_index = build_image_index(image_root)
    reports: list[ManualReport] = []
    skipped: list[dict[str, str]] = []

    for txt_path in sorted(import_root.glob("*.txt")):
        if txt_path.name in SKIP_TXT_NAMES:
            skipped.append({"txt_name": txt_path.name, "reason": "explicitly skipped"})
            continue

        try:
            reports.append(organize_manual(txt_path, image_index, RAW_ROOT))
        except Exception as exc:
            skipped.append({"txt_name": txt_path.name, "reason": str(exc)})

    report_path = RAW_ROOT / "organization_report.json"
    write_report(report_path, reports, skipped)

    print(f"organized manuals: {len(reports)}")
    print(f"skipped files: {len(skipped)}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
