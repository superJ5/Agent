"""Restore English manual image descriptions from another git branch.

The current branch keeps the latest chunk hierarchy. The source branch keeps
VLM image descriptions. This script copies descriptions by pic_id only; it does
not copy chunk structure from the source branch.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any


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

PIC_RE = re.compile(
    r"<PIC:([^>]+)>(?:\s*\{\{(?:image description|图片描述|此图片描述为)[:：]\s*(.*?)\}\})?",
    re.IGNORECASE | re.DOTALL,
)
PIC_ID_RE = re.compile(r"<PIC:([^>]+)>")
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


def load_jsonl_text(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_jsonl_path(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_text(path.read_text(encoding="utf-8"))


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


def markdown_to_plain_text(text: str) -> str:
    text = PIC_RE.sub(lambda match: f" {match.group(2) or ''} ", text)
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


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def branch_file_records(repo: Path, branch: str, filename: str) -> list[dict[str, Any]]:
    rel = f"data/manuals/chunks/{filename}"
    data = subprocess.check_output(["git", "show", f"{branch}:{rel}"], cwd=repo)
    return load_jsonl_text(data.decode("utf-8"))


def descriptions_from_source(records: list[dict[str, Any]]) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for record in records:
        for item in record.get("pic_descriptions") or []:
            if not isinstance(item, dict):
                continue
            pic_id = item.get("pic_id")
            description = item.get("description")
            if not pic_id or not description:
                continue
            pic_id = str(pic_id)
            description = str(description)
            if pic_id in descriptions and descriptions[pic_id] != description:
                raise ValueError(f"conflicting descriptions for {pic_id}")
            descriptions[pic_id] = description
    return descriptions


def inject_descriptions(text: str, descriptions: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        pic_id = match.group(1)
        description = descriptions.get(pic_id)
        if not description:
            return match.group(0)
        return f"<PIC:{pic_id}>\n{{{{image description: {description}}}}}"

    return PIC_RE.sub(repl, text)


def pic_ids_for_record(record: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    ids.extend(str(pic_id) for pic_id in (record.get("pic_ids") or []) if pic_id)
    ids.extend(PIC_ID_RE.findall(record.get("text") or ""))
    return unique_in_order(ids)


def restore_file(records: list[dict[str, Any]], descriptions: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "records_with_pic_descriptions": 0,
        "description_items": 0,
        "text_wrappers": 0,
        "index_text_updated": 0,
    }
    output: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        pic_ids = pic_ids_for_record(record)
        record_descriptions = [
            {"pic_id": pic_id, "description": descriptions[pic_id]}
            for pic_id in pic_ids
            if pic_id in descriptions
        ]
        if record_descriptions:
            record["pic_descriptions"] = record_descriptions
            stats["records_with_pic_descriptions"] += 1
            stats["description_items"] += len(record_descriptions)
        else:
            record.pop("pic_descriptions", None)

        text = record.get("text")
        if isinstance(text, str) and pic_ids:
            new_text = inject_descriptions(text, descriptions)
            record["text"] = new_text
            stats["text_wrappers"] += len(DESC_WRAPPER_RE.findall(new_text))
            record["index_text"] = markdown_to_plain_text(f"{record.get('title') or ''}\n{new_text}")
            stats["index_text_updated"] += 1
        output.append(record)
    return output, stats


def validate(records: list[dict[str, Any]]) -> tuple[int, int]:
    missing_wrappers = 0
    missing_index_descriptions = 0
    for record in records:
        descriptions = record.get("pic_descriptions") or []
        if descriptions and not DESC_WRAPPER_RE.search(record.get("text") or ""):
            missing_wrappers += 1
        index_text = record.get("index_text") or ""
        for item in descriptions:
            description = item.get("description") if isinstance(item, dict) else None
            if description and description[:50] not in index_text:
                missing_index_descriptions += 1
                break
    return missing_wrappers, missing_index_descriptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-branch", default="chunk_middle_result")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/manuals/chunks"))
    args = parser.parse_args()

    totals = {
        "files": 0,
        "records_with_pic_descriptions": 0,
        "description_items": 0,
        "text_wrappers": 0,
        "index_text_updated": 0,
    }

    for filename in ENGLISH_FILES:
        path = args.repo / args.chunks_dir / filename
        current = load_jsonl_path(path)
        source = branch_file_records(args.repo, args.source_branch, filename)
        descriptions = descriptions_from_source(source)

        current_pic_ids = sorted({pic_id for record in current for pic_id in pic_ids_for_record(record)})
        missing = [pic_id for pic_id in current_pic_ids if pic_id not in descriptions]
        if missing:
            raise ValueError(f"{filename}: missing descriptions for pic_ids: {missing[:20]}")

        restored, stats = restore_file(current, descriptions)
        missing_wrappers, missing_index = validate(restored)
        if missing_wrappers or missing_index:
            raise ValueError(
                f"{filename}: validation failed, "
                f"missing_wrappers={missing_wrappers}, missing_index={missing_index}"
            )
        dump_jsonl(path, restored)

        totals["files"] += 1
        for key, value in stats.items():
            totals[key] += value
        print(f"{filename}: {stats}")

    print(f"TOTAL: {totals}")


if __name__ == "__main__":
    main()
