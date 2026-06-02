"""Mark super-large English support chunks as big_support.

This pass only adjusts the merged English outputs in
data/manuals/new_chunk/chunk_integrate.  For support chunks whose index_text is
larger than BIG_SUPPORT_THRESHOLD, it keeps source_lines unchanged for review,
but trims text/index_text/pic metadata to the support node's own text before the
first direct child starts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

BIG_SUPPORT_THRESHOLD = 8000

ENGLISH_FILES = [
    "manual_1bf5d984__Television.jsonl",
    "manual_34457b32__Earphones.jsonl",
    "manual_35519890__Motherboard.jsonl",
    "manual_3660cac1__Microwave.jsonl",
    "manual_3adbbed1__WaveRunner.jsonl",
    "manual_475de6ea__Grill.jsonl",
    "manual_4fee7ce2__Air_Fryer.jsonl",
    "manual_59aaa0db__Washing_Machine.jsonl",
    "manual_6cd4b7d0__Vacuum.jsonl",
    "manual_8cc6fc45__Landline.jsonl",
    "manual_967d35e4__Camera.jsonl",
    "manual_969b7bb4__Pressure_Cooker_Air_Fryer.jsonl",
    "manual_9810aa2b__Fax.jsonl",
    "manual_9f330644__Lawn_Mower.jsonl",
    "manual_b10b6174__Toothbrush.jsonl",
    "manual_bb5e4ceb__EReader.jsonl",
    "manual_d14b4448__Security_Camera.jsonl",
    "manual_e724f855__Snowmobile.jsonl",
    "manual_f2c7ec3d__Espresso_Machine.jsonl",
    "manual_f769aafb__Boat.jsonl",
]

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
    "language",
]

PIC_RE = re.compile(r"<PIC:([^>]+)>")
PIC_WITH_DESC_RE = re.compile(
    r"<PIC:([^>]+)>(?:\s*\{\{(?:image description|图片描述|此图片描述为)[:：]\s*(.*?)\}\})?",
    re.IGNORECASE | re.DOTALL,
)
DESC_WRAPPER_RE = re.compile(
    r"\{\{(?:image description|图片描述|此图片描述为)[:：]\s*(.*?)\}\}",
    re.IGNORECASE | re.DOTALL,
)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MARKDOWN_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
MARKDOWN_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
MARKDOWN_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
MARKDOWN_ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
MARKDOWN_TABLE_SEP_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$",
    re.MULTILINE,
)
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANKS_RE = re.compile(r"\n{3,}")


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


def ordered_record(record: dict[str, Any]) -> OrderedDict[str, Any]:
    ordered: OrderedDict[str, Any] = OrderedDict()
    for key in FIELD_ORDER:
        if key in record:
            ordered[key] = record[key]
    for key, value in record.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def dump_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(ordered_record(record), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def unique_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def markdown_to_plain_text(text: str) -> str:
    text = PIC_WITH_DESC_RE.sub(lambda match: f" {match.group(2) or ''} ", text)
    text = DESC_WRAPPER_RE.sub(r" \1 ", text)
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


def source_file_lines(source_file: str, cache: dict[str, list[str]]) -> list[str]:
    path = Path(source_file)
    if not path.exists():
        raise FileNotFoundError(f"source_file does not exist: {source_file}")
    if source_file not in cache:
        cache[source_file] = path.read_text(encoding="utf-8").splitlines()
    return cache[source_file]


def read_source_slice(
    source_file: str,
    start_line: int,
    end_line: int,
    cache: dict[str, list[str]],
) -> str:
    lines = source_file_lines(source_file, cache)
    start = max(1, start_line)
    end = min(len(lines), max(start, end_line))
    return "\n".join(lines[start - 1 : end]).rstrip()


def pic_description_map(record: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in record.get("pic_descriptions") or []:
        if not isinstance(item, dict):
            continue
        pic_id = str(item.get("pic_id") or "").strip()
        description = str(item.get("description") or "").strip()
        if pic_id and description and pic_id not in result:
            result[pic_id] = description
    return result


def inject_pic_descriptions(text: str, descriptions: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        pic_id = match.group(1)
        description = descriptions.get(pic_id)
        if not description:
            return match.group(0)
        return f"<PIC:{pic_id}>\n{{{{image description: {description}}}}}"

    return PIC_RE.sub(repl, text)


def valid_source_lines(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    )


def direct_child_start_line(parent: dict[str, Any], children: list[dict[str, Any]]) -> int | None:
    parent_lines = parent.get("source_lines")
    if not valid_source_lines(parent_lines):
        return None
    parent_start = parent_lines[0]
    starts = [
        child["source_lines"][0]
        for child in children
        if valid_source_lines(child.get("source_lines"))
        and child["source_lines"][0] >= parent_start
    ]
    return min(starts) if starts else None


def trim_big_support_record(
    record: dict[str, Any],
    children: list[dict[str, Any]],
    source_cache: dict[str, list[str]],
) -> dict[str, Any]:
    source_lines = record.get("source_lines")
    if not valid_source_lines(source_lines):
        raise ValueError(f"{record.get('chunk_id')}: invalid source_lines: {source_lines!r}")

    source_file = str(record.get("source_file") or "")
    if not source_file:
        raise ValueError(f"{record.get('chunk_id')}: missing source_file")

    original_source_lines = list(source_lines)
    own_start = original_source_lines[0]
    child_start = direct_child_start_line(record, children)
    own_end = child_start - 1 if child_start is not None else original_source_lines[1]
    if own_end < own_start:
        own_end = own_start

    text = read_source_slice(source_file, own_start, own_end, source_cache)
    descriptions = pic_description_map(record)
    text = inject_pic_descriptions(text, descriptions).strip()

    pic_ids = unique_values(PIC_RE.findall(text))
    pic_descriptions = [
        {"pic_id": pic_id, "description": descriptions[pic_id]}
        for pic_id in pic_ids
        if descriptions.get(pic_id)
    ]
    title = str(record.get("title") or "")

    updated = dict(record)
    updated["retrieval_tier"] = "big_support"
    updated["source_lines"] = original_source_lines
    updated["pic_ids"] = pic_ids
    updated["pic_descriptions"] = pic_descriptions
    updated["text"] = text
    updated["index_text"] = markdown_to_plain_text(f"{title}\n{text}")
    updated["_big_support_own_lines"] = [own_start, own_end]
    updated["_big_support_direct_child_start"] = child_start
    return updated


def process_file(path: Path, threshold: int, dry_run: bool) -> dict[str, Any]:
    records = load_jsonl(path)
    if any((record.get("language") or "en") != "en" for record in records):
        raise ValueError(f"{path.name}: expected only English records")

    children_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("retrieval_tier") == "auxiliary":
            continue
        parent_id = record.get("parent_chunk_id")
        if parent_id:
            children_by_parent[str(parent_id)].append(record)

    source_cache: dict[str, list[str]] = {}
    changed = 0
    changed_rows: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    for record in records:
        if record.get("retrieval_tier") == "support" and len(record.get("index_text") or "") > threshold:
            old_index_len = len(record.get("index_text") or "")
            old_text_len = len(record.get("text") or "")
            updated = trim_big_support_record(record, children_by_parent.get(str(record.get("chunk_id")), []), source_cache)
            changed += 1
            changed_rows.append(
                {
                    "chunk_id": record.get("chunk_id"),
                    "title": record.get("title"),
                    "source_lines": record.get("source_lines"),
                    "own_lines": updated.pop("_big_support_own_lines"),
                    "direct_child_start": updated.pop("_big_support_direct_child_start"),
                    "old_index_len": old_index_len,
                    "new_index_len": len(updated.get("index_text") or ""),
                    "old_text_len": old_text_len,
                    "new_text_len": len(updated.get("text") or ""),
                    "old_pic_count": len(record.get("pic_ids") or []),
                    "new_pic_count": len(updated.get("pic_ids") or []),
                    "direct_child_count": len(children_by_parent.get(str(record.get("chunk_id")), [])),
                }
            )
            output.append(updated)
        else:
            output.append(record)

    if changed and not dry_run:
        dump_jsonl(path, output)

    return {
        "file": path.name,
        "records": len(records),
        "changed": changed,
        "rows": changed_rows,
    }


def validate_file(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    support_over_threshold = [
        record.get("chunk_id")
        for record in records
        if record.get("retrieval_tier") == "support"
        and len(record.get("index_text") or "") > BIG_SUPPORT_THRESHOLD
    ]
    big_support = [record for record in records if record.get("retrieval_tier") == "big_support"]
    pic_mismatches: list[str] = []
    for record in big_support:
        text_pic_ids = unique_values(PIC_RE.findall(record.get("text") or ""))
        if text_pic_ids != (record.get("pic_ids") or []):
            pic_mismatches.append(str(record.get("chunk_id")))
    return {
        "file": path.name,
        "support_over_threshold": support_over_threshold,
        "big_support_count": len(big_support),
        "big_support_over_threshold": [
            record.get("chunk_id")
            for record in big_support
            if len(record.get("index_text") or "") > BIG_SUPPORT_THRESHOLD
        ],
        "pic_mismatches": pic_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/manuals/new_chunk/chunk_integrate"),
    )
    parser.add_argument("--threshold", type=int, default=BIG_SUPPORT_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for filename in ENGLISH_FILES:
        path = args.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        summaries.append(process_file(path, args.threshold, args.dry_run))

    validations = [validate_file(args.base_dir / filename) for filename in ENGLISH_FILES]

    print(json.dumps(
        {
            "dry_run": args.dry_run,
            "base_dir": str(args.base_dir),
            "threshold": args.threshold,
            "files": len(summaries),
            "changed": sum(item["changed"] for item in summaries),
            "changed_by_file": {
                item["file"]: item["changed"]
                for item in summaries
                if item["changed"]
            },
            "changed_rows": [
                row
                for item in summaries
                for row in item["rows"]
            ],
            "validation": {
                "support_over_threshold": {
                    item["file"]: item["support_over_threshold"]
                    for item in validations
                    if item["support_over_threshold"]
                },
                "big_support_count": sum(item["big_support_count"] for item in validations),
                "big_support_over_threshold": {
                    item["file"]: item["big_support_over_threshold"]
                    for item in validations
                    if item["big_support_over_threshold"]
                },
                "pic_mismatches": {
                    item["file"]: item["pic_mismatches"]
                    for item in validations
                    if item["pic_mismatches"]
                },
            },
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
