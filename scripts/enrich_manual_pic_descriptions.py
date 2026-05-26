"""Enrich manual chunk JSONL files with image descriptions.

The script is intentionally deterministic: humans/agents provide image
descriptions, and this script applies them to chunks with one shared format.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


PIC_RE = re.compile(r"<PIC:([^>]+)>")
PIC_WITH_DESC_RE = re.compile(
    r"(<PIC:([^>]+)>)(?:\s*\{\{此图片描述为:\s*.*?\}\})?",
    re.DOTALL,
)
DESC_WRAPPER_RE = re.compile(r"\{\{此图片描述为:\s*(.*?)\}\}", re.DOTALL)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MARKDOWN_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
MARKDOWN_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
MARKDOWN_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
MARKDOWN_ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
MARKDOWN_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANKS_RE = re.compile(r"\n{3,}")

FIELD_ORDER = [
    "chunk_id",
    "doc_id",
    "source_file",
    "source_lines",
    "chunk_type",
    "retrieval_tier",
    "section_path",
    "title",
    "parent_chunk_id",
    "pic_ids",
    "pic_descriptions",
    "text",
    "index_text",
    "source_quality",
    "source_issue_flags",
    "source_issue_note",
    "item_no",
    "item_kind",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def dump_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(ordered_record(record), ensure_ascii=False))
            handle.write("\n")


def ordered_record(record: dict[str, Any]) -> OrderedDict[str, Any]:
    ordered: OrderedDict[str, Any] = OrderedDict()
    for key in FIELD_ORDER:
        if key in record:
            ordered[key] = record[key]
    for key, value in record.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def find_image_index(raw_dir: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for image_dir in raw_dir.rglob("images"):
        if not image_dir.is_dir():
            continue
        for image_path in image_dir.iterdir():
            if not image_path.is_file():
                continue
            image_id = image_path.stem
            index.setdefault(image_id, []).append(str(image_path))
    return index


def unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_pic_ids(text: str | None) -> list[str]:
    if not text:
        return []
    return [match.group(1) for match in PIC_RE.finditer(text)]


def load_descriptions(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "descriptions" in data:
        data = data["descriptions"]

    descriptions: dict[str, str] = {}
    if isinstance(data, dict):
        iterable = data.items()
        for pic_id, description in iterable:
            descriptions[str(pic_id)] = normalize_description(str(description))
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            pic_id = item.get("pic_id")
            description = item.get("description")
            if pic_id and description:
                descriptions[str(pic_id)] = normalize_description(str(description))
    else:
        raise ValueError(f"Unsupported description JSON shape in {path}")
    return descriptions


def normalize_description(description: str) -> str:
    description = DESC_WRAPPER_RE.sub(r"\1", description)
    description = re.sub(r"\s+", " ", description).strip()
    return description.strip("；;，, ")


def existing_descriptions(record: dict[str, Any]) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    raw = record.get("pic_descriptions") or []
    if isinstance(raw, dict):
        for pic_id, description in raw.items():
            if description:
                descriptions[str(pic_id)] = normalize_description(str(description))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            pic_id = item.get("pic_id")
            description = item.get("description")
            if pic_id and description:
                descriptions[str(pic_id)] = normalize_description(str(description))
    return descriptions


def description_items(pic_ids: list[str], descriptions: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"pic_id": pic_id, "description": descriptions[pic_id]}
        for pic_id in unique_in_order(pic_ids)
        if pic_id in descriptions and descriptions[pic_id]
    ]


def enrich_text(text: str, descriptions: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        pic_id = match.group(2)
        description = descriptions.get(pic_id)
        if not description:
            return match.group(1)
        return f"{match.group(1)}\n{{{{此图片描述为: {description}}}}}"

    return PIC_WITH_DESC_RE.sub(replace, text)


def text_to_index_text(record: dict[str, Any], descriptions: dict[str, str]) -> str:
    text = str(record.get("text") or "")

    def replace_pic(match: re.Match[str]) -> str:
        return " " + descriptions.get(match.group(2), "") + " "

    text = PIC_WITH_DESC_RE.sub(replace_pic, text)
    text = DESC_WRAPPER_RE.sub(r" \1 ", text)
    text = markdown_to_plain_text(text)

    section_path = record.get("section_path")
    prefix = ""
    if isinstance(section_path, list) and section_path:
        prefix = " > ".join(str(part) for part in section_path if part)
    elif record.get("title"):
        prefix = str(record["title"])
    if prefix:
        prefix = PIC_WITH_DESC_RE.sub(replace_pic, prefix)
        prefix = markdown_to_plain_text(prefix)

    if prefix and text:
        return f"{prefix}\n{text}"
    return prefix or text


def markdown_to_plain_text(text: str) -> str:
    text = MARKDOWN_FENCE_RE.sub(" ", text)
    text = MARKDOWN_IMAGE_RE.sub(" ", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = MARKDOWN_INLINE_CODE_RE.sub(r"\1", text)
    text = MARKDOWN_TABLE_SEP_RE.sub(" ", text)
    text = MARKDOWN_HEADING_RE.sub("", text)
    text = MARKDOWN_BULLET_RE.sub("", text)
    text = MARKDOWN_ORDERED_RE.sub("", text)
    text = text.replace("|", " ")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = BLANKS_RE.sub("\n\n", text)
    return text.strip()


def enrich_records(
    records: list[dict[str, Any]],
    provided_descriptions: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    used_ids: set[str] = set()
    missing_ids: set[str] = set()
    output: list[dict[str, Any]] = []

    for record in records:
        record = dict(record)
        text = str(record.get("text") or "")
        pic_ids = unique_in_order(extract_pic_ids(text) + [str(pic_id) for pic_id in record.get("pic_ids", [])])
        descriptions = existing_descriptions(record)
        for pic_id in pic_ids:
            if pic_id not in descriptions and pic_id in provided_descriptions:
                descriptions[pic_id] = provided_descriptions[pic_id]
            if pic_id in descriptions and descriptions[pic_id]:
                used_ids.add(pic_id)
            else:
                missing_ids.add(pic_id)

        if pic_ids:
            record["pic_descriptions"] = description_items(pic_ids, descriptions)
        elif "pic_descriptions" in record:
            record["pic_descriptions"] = description_items([], descriptions)

        record["text"] = enrich_text(text, descriptions)
        record["index_text"] = text_to_index_text(record, descriptions)
        output.append(record)

    report = {
        "used_description_ids": sorted(used_ids),
        "missing_description_ids": sorted(missing_ids),
        "unused_provided_ids": sorted(set(provided_descriptions) - used_ids),
    }
    return output, report


def gather(args: argparse.Namespace) -> None:
    chunk_dir = Path(args.chunk_dir)
    raw_dir = Path(args.raw_dir)
    image_index = find_image_index(raw_dir)
    rows: list[dict[str, Any]] = []

    for chunk_path in sorted(chunk_dir.glob("*.jsonl")):
        records = load_jsonl(chunk_path)
        for record in records:
            text = str(record.get("text") or "")
            for pic_id in unique_in_order(extract_pic_ids(text)):
                context = compact_context(text, pic_id)
                rows.append(
                    {
                        "chunk_file": chunk_path.name,
                        "doc_id": record.get("doc_id"),
                        "chunk_id": record.get("chunk_id"),
                        "title": record.get("title"),
                        "section_path": record.get("section_path"),
                        "pic_id": pic_id,
                        "image_paths": image_index.get(pic_id, []),
                        "context": context,
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def compact_context(text: str, pic_id: str, radius: int = 260) -> str:
    marker = f"<PIC:{pic_id}>"
    pos = text.find(marker)
    if pos < 0:
        return markdown_to_plain_text(text)[: radius * 2]
    start = max(0, pos - radius)
    end = min(len(text), pos + len(marker) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def apply(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    descriptions = load_descriptions(Path(args.descriptions))
    records = load_jsonl(input_path)
    enriched, report = enrich_records(records, descriptions)
    dump_jsonl(output_path, enriched)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def validate(args: argparse.Namespace) -> None:
    original_dir = Path(args.original_dir)
    new_dir = Path(args.new_dir)
    results: list[dict[str, Any]] = []

    for original_path in sorted(original_dir.glob("*.jsonl")):
        new_path = new_dir / original_path.name
        result: dict[str, Any] = {"file": original_path.name, "exists": new_path.exists()}
        if not new_path.exists():
            results.append(result)
            continue

        original_records = load_jsonl(original_path)
        new_records = load_jsonl(new_path)
        original_placeholders = sum(len(extract_pic_ids(str(r.get("text") or ""))) for r in original_records)
        new_placeholders = sum(len(extract_pic_ids(str(r.get("text") or ""))) for r in new_records)
        new_desc_wrappers = sum(str(r.get("text") or "").count("{{此图片描述为:") for r in new_records)
        new_index_placeholders = sum(len(extract_pic_ids(str(r.get("index_text") or ""))) for r in new_records)
        records_with_pic_ids = sum(1 for r in new_records if r.get("pic_ids"))
        records_with_pic_descriptions = sum(1 for r in new_records if r.get("pic_descriptions"))
        missing_desc_ids: set[str] = set()
        for record in new_records:
            pic_ids = unique_in_order(extract_pic_ids(str(record.get("text") or "")) + [str(x) for x in record.get("pic_ids", [])])
            desc_ids = {str(item.get("pic_id")) for item in record.get("pic_descriptions", []) if isinstance(item, dict)}
            for pic_id in pic_ids:
                if pic_id not in desc_ids:
                    missing_desc_ids.add(pic_id)

        result.update(
            {
                "original_records": len(original_records),
                "new_records": len(new_records),
                "original_text_pic_occurrences": original_placeholders,
                "new_text_pic_occurrences": new_placeholders,
                "new_text_description_wrappers": new_desc_wrappers,
                "new_index_pic_occurrences": new_index_placeholders,
                "records_with_pic_ids": records_with_pic_ids,
                "records_with_pic_descriptions": records_with_pic_descriptions,
                "missing_description_ids": sorted(missing_desc_ids),
            }
        )
        results.append(result)

    output = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)


def contact_sheets(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw, ImageFont

    tasks_path = Path(args.tasks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    by_file: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        chunk_file = row["chunk_file"]
        pic_id = row["pic_id"]
        by_file.setdefault(chunk_file, {})
        by_file[chunk_file].setdefault(pic_id, row)

    manifest: list[dict[str, Any]] = []
    for chunk_file, task_map in sorted(by_file.items()):
        tasks = list(task_map.values())
        for sheet_no, start in enumerate(range(0, len(tasks), args.per_sheet), 1):
            sheet_tasks = tasks[start : start + args.per_sheet]
            sheet_path = output_dir / f"{Path(chunk_file).stem}__sheet_{sheet_no:02d}.jpg"
            make_contact_sheet(
                sheet_tasks,
                sheet_path,
                thumb_width=args.thumb_width,
                columns=args.columns,
                font_size=args.font_size,
                Image=Image,
                ImageDraw=ImageDraw,
                ImageFont=ImageFont,
            )
            manifest.append(
                {
                    "chunk_file": chunk_file,
                    "sheet": str(sheet_path),
                    "pic_ids": [task["pic_id"] for task in sheet_tasks],
                }
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sheet_count": len(manifest), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


def make_contact_sheet(
    tasks: list[dict[str, Any]],
    output_path: Path,
    *,
    thumb_width: int,
    columns: int,
    font_size: int,
    Image: Any,
    ImageDraw: Any,
    ImageFont: Any,
) -> None:
    label_height = font_size * 2 + 10
    padding = 18
    gap = 16
    cells: list[tuple[str, Any]] = []
    font = ImageFont.load_default()

    for task in tasks:
        image_paths = task.get("image_paths") or []
        pic_id = task["pic_id"]
        if not image_paths:
            thumb = Image.new("RGB", (thumb_width, thumb_width), "white")
        else:
            with Image.open(image_paths[0]) as image:
                image = image.convert("RGB")
                width, height = image.size
                scale = thumb_width / max(width, 1)
                thumb_height = max(1, int(height * scale))
                thumb = image.resize((thumb_width, thumb_height))
        cells.append((pic_id, thumb))

    cell_height = max((thumb.height for _, thumb in cells), default=thumb_width) + label_height
    rows = (len(cells) + columns - 1) // columns
    sheet_width = padding * 2 + columns * thumb_width + (columns - 1) * gap
    sheet_height = padding * 2 + rows * cell_height + max(0, rows - 1) * gap
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, (pic_id, thumb) in enumerate(cells):
        row = idx // columns
        col = idx % columns
        x = padding + col * (thumb_width + gap)
        y = padding + row * (cell_height + gap)
        draw.rectangle([x, y, x + thumb_width - 1, y + label_height - 1], fill=(245, 245, 245), outline=(180, 180, 180))
        draw.text((x + 6, y + 6), pic_id, fill=(0, 0, 0), font=font)
        sheet.paste(thumb, (x, y + label_height))
        draw.rectangle(
            [x, y + label_height, x + thumb_width - 1, y + label_height + thumb.height - 1],
            outline=(180, 180, 180),
        )

    sheet.save(output_path, quality=92)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gather_parser = subparsers.add_parser("gather", help="Write picture-description task rows.")
    gather_parser.add_argument("--chunk-dir", default="data/manuals/chunks")
    gather_parser.add_argument("--raw-dir", default="data/manuals/raw")
    gather_parser.add_argument("--output", default="data/manuals/new_chunk/pic_description_tasks.jsonl")
    gather_parser.set_defaults(func=gather)

    apply_parser = subparsers.add_parser("apply", help="Apply descriptions to one chunk JSONL file.")
    apply_parser.add_argument("--input", required=True)
    apply_parser.add_argument("--descriptions", required=True)
    apply_parser.add_argument("--output", required=True)
    apply_parser.add_argument("--report")
    apply_parser.set_defaults(func=apply)

    validate_parser = subparsers.add_parser("validate", help="Validate enriched chunk files.")
    validate_parser.add_argument("--original-dir", default="data/manuals/chunks")
    validate_parser.add_argument("--new-dir", default="data/manuals/new_chunk")
    validate_parser.add_argument("--output")
    validate_parser.set_defaults(func=validate)

    contact_parser = subparsers.add_parser("contact-sheets", help="Build labeled contact sheets for visual review.")
    contact_parser.add_argument("--tasks", default="data/manuals/new_chunk/pic_description_tasks.jsonl")
    contact_parser.add_argument("--output-dir", default="data/manuals/new_chunk/contact_sheets")
    contact_parser.add_argument("--thumb-width", type=int, default=640)
    contact_parser.add_argument("--columns", type=int, default=2)
    contact_parser.add_argument("--per-sheet", type=int, default=8)
    contact_parser.add_argument("--font-size", type=int, default=18)
    contact_parser.set_defaults(func=contact_sheets)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
