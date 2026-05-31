"""Complete missing support nodes for the 20 English manual chunk files.

This pass does not merge chunks. It builds a section_path prefix tree, creates
missing support nodes for absent exact prefixes, rebuilds parent_chunk_id, and
renumbers all chunks in each output file.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict, defaultdict
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

AUX_TYPES = {"metadata_image_path", "aux_navigation"}
PIC_RE = re.compile(r"<PIC:([^>]+)>")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
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

QUALITY_ORDER = {
    "normal": 0,
    "warning": 1,
    "needs_review": 2,
    "issue": 3,
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(ordered_record(record), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def is_aux_chunk(record: dict[str, Any]) -> bool:
    return record.get("chunk_type") in AUX_TYPES or record.get("retrieval_tier") == "auxiliary"


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


def unique_pic_descriptions(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for record in records:
        for item in record.get("pic_descriptions") or []:
            if not isinstance(item, dict):
                continue
            pic_id = item.get("pic_id")
            description = item.get("description")
            if not pic_id or pic_id in seen:
                continue
            seen.add(pic_id)
            result.append({"pic_id": str(pic_id), "description": str(description or "")})
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


def inject_pic_descriptions(text: str, descriptions: list[dict[str, str]]) -> str:
    desc_by_pic = {
        item["pic_id"]: item.get("description", "")
        for item in descriptions
        if item.get("pic_id") and item.get("description")
    }

    def repl(match: re.Match[str]) -> str:
        pic_id = match.group(1)
        description = desc_by_pic.get(pic_id)
        if not description:
            return match.group(0)
        return f"<PIC:{pic_id}>\n{{{{image description: {description}}}}}"

    return PIC_RE.sub(repl, text)


def source_lines_for(records: list[dict[str, Any]]) -> list[int]:
    starts = [record["source_lines"][0] for record in records]
    ends = [record["source_lines"][1] for record in records]
    return [min(starts), max(ends)]


def worst_source_quality(records: list[dict[str, Any]]) -> str:
    return max(
        (str(record.get("source_quality") or "normal") for record in records),
        key=lambda quality: QUALITY_ORDER.get(quality, -1),
        default="normal",
    )


def merged_issue_flags(records: list[dict[str, Any]]) -> list[Any]:
    flags: list[Any] = []
    for record in records:
        value = record.get("source_issue_flags")
        if isinstance(value, list):
            flags.extend(value)
    return unique_values(flags)


def merged_issue_note(records: list[dict[str, Any]]) -> str:
    notes = [str(record.get("source_issue_note") or "").strip() for record in records]
    return " | ".join(unique_values([note for note in notes if note]))


SOURCE_LINE_CACHE: dict[str, list[str]] = {}


def source_file_lines(source_file: str) -> list[str]:
    path = Path(source_file)
    if not path.exists():
        raise FileNotFoundError(f"source_file does not exist: {source_file}")
    if source_file not in SOURCE_LINE_CACHE:
        SOURCE_LINE_CACHE[source_file] = path.read_text(encoding="utf-8").splitlines()
    return SOURCE_LINE_CACHE[source_file]


def read_source_slice(source_file: str, source_lines: list[int]) -> str:
    lines = source_file_lines(source_file)
    start, end = source_lines
    return "\n".join(lines[start - 1 : end]).rstrip()


def normalize_heading(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def heading_start_line(source_file: str, title: str, fallback_start: int) -> int:
    normalized_title = normalize_heading(title)
    lines = source_file_lines(source_file)
    start = min(max(fallback_start, 1), len(lines))
    for line_no in range(start, 0, -1):
        match = HEADING_RE.match(lines[line_no - 1])
        if match and normalize_heading(match.group(1)) == normalized_title:
            return line_no
    return fallback_start


def starts_with_path(record: dict[str, Any], prefix: tuple[str, ...]) -> bool:
    path = tuple(record.get("section_path") or [])
    return len(path) >= len(prefix) and path[: len(prefix)] == prefix


def nearest_parent_path(path: tuple[str, ...], path_nodes: dict[tuple[str, ...], dict[str, Any]]) -> tuple[str, ...] | None:
    parent_path = path[:-1]
    while parent_path:
        if parent_path in path_nodes:
            return parent_path
        parent_path = parent_path[:-1]
    return None


def make_synthetic_support(
    prefix: tuple[str, ...],
    descendants: list[dict[str, Any]],
    synthetic_index: int,
) -> dict[str, Any]:
    source_lines = source_lines_for(descendants)
    source_file = descendants[0]["source_file"]
    source_lines[0] = heading_start_line(source_file, prefix[-1], source_lines[0])
    pic_ids = unique_values([pic_id for record in descendants for pic_id in (record.get("pic_ids") or [])])
    pic_descriptions = unique_pic_descriptions(descendants)
    text = inject_pic_descriptions(read_source_slice(source_file, source_lines), pic_descriptions)
    title = prefix[-1]
    index_text = markdown_to_plain_text(f"{title}\n{text}")
    doc_id = descendants[0]["doc_id"]
    return {
        "chunk_id": f"__synthetic_support_{synthetic_index:04d}",
        "doc_id": doc_id,
        "source_file": source_file,
        "source_lines": source_lines,
        "chunk_type": "section_summary",
        "retrieval_tier": "support",
        "section_path": list(prefix),
        "title": title,
        "parent_chunk_id": None,
        "pic_ids": pic_ids,
        "pic_descriptions": pic_descriptions,
        "text": text,
        "index_text": index_text,
        "source_quality": worst_source_quality(descendants),
        "source_issue_flags": merged_issue_flags(descendants),
        "source_issue_note": merged_issue_note(descendants),
        "item_no": None,
        "item_kind": None,
        "language": descendants[0].get("language") or "en",
        "_synthetic": True,
    }


def materialize_support_records(records: list[dict[str, Any]]) -> None:
    children_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if is_aux_chunk(record):
            continue
        parent_id = record.get("parent_chunk_id")
        if parent_id:
            children_by_parent[parent_id].append(record)

    def collect_subtree(record: dict[str, Any]) -> list[dict[str, Any]]:
        collected = [record]
        for child in children_by_parent.get(record["_old_chunk_id"], []):
            collected.extend(collect_subtree(child))
        return collected

    for record in records:
        if is_aux_chunk(record) or record.get("retrieval_tier") != "support":
            continue
        subtree = sorted(collect_subtree(record), key=lambda item: item["_order"])
        source_lines = source_lines_for(subtree)
        source_file = record["source_file"]
        section_path = record.get("section_path") or []
        section_title = section_path[-1] if section_path else (record.get("title") or "")
        record["title"] = section_title
        source_lines[0] = heading_start_line(source_file, section_title, source_lines[0])
        pic_ids = unique_values([pic_id for item in subtree for pic_id in (item.get("pic_ids") or [])])
        pic_descriptions = unique_pic_descriptions(subtree)
        text = inject_pic_descriptions(read_source_slice(source_file, source_lines), pic_descriptions)
        record["source_lines"] = source_lines
        record["pic_ids"] = pic_ids
        record["pic_descriptions"] = pic_descriptions
        record["text"] = text
        record["index_text"] = markdown_to_plain_text(f"{section_title}\n{text}")
        record["source_quality"] = worst_source_quality(subtree)
        record["source_issue_flags"] = merged_issue_flags(subtree)
        record["source_issue_note"] = merged_issue_note(subtree)


def complete_one_file(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    working: list[dict[str, Any]] = []
    for order, record in enumerate(records):
        item = dict(record)
        item["_old_chunk_id"] = record.get("chunk_id")
        item["_order"] = order
        item["_synthetic"] = False
        working.append(item)

    non_aux = [record for record in working if not is_aux_chunk(record)]
    exact_heads: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in non_aux:
        path = tuple(record.get("section_path") or [])
        if path and path not in exact_heads:
            exact_heads[path] = record

    required_prefixes: set[tuple[str, ...]] = set()
    for record in non_aux:
        path = tuple(record.get("section_path") or [])
        for depth in range(1, len(path) + 1):
            required_prefixes.add(path[:depth])

    missing_prefixes = sorted(
        (prefix for prefix in required_prefixes if prefix not in exact_heads),
        key=lambda prefix: (
            min(record["_order"] for record in non_aux if starts_with_path(record, prefix)),
            len(prefix),
            prefix,
        ),
    )

    synthetic_records: list[dict[str, Any]] = []
    for index, prefix in enumerate(missing_prefixes, 1):
        descendants = [record for record in non_aux if starts_with_path(record, prefix)]
        synthetic = make_synthetic_support(prefix, descendants, index)
        synthetic["_old_chunk_id"] = synthetic["chunk_id"]
        synthetic["_order"] = min(record["_order"] for record in descendants)
        synthetic_records.append(synthetic)

    path_nodes = dict(exact_heads)
    for synthetic in synthetic_records:
        path_nodes[tuple(synthetic["section_path"])] = synthetic

    for record in [*working, *synthetic_records]:
        if is_aux_chunk(record):
            continue
        path = tuple(record.get("section_path") or [])
        head = path_nodes.get(path)
        if head is not None and record is not head:
            record["parent_chunk_id"] = head["_old_chunk_id"]
            continue
        parent_path = nearest_parent_path(path, path_nodes)
        record["parent_chunk_id"] = path_nodes[parent_path]["_old_chunk_id"] if parent_path else None

    non_aux_with_synthetic = [record for record in [*working, *synthetic_records] if not is_aux_chunk(record)]
    children_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in non_aux_with_synthetic:
        parent_id = record.get("parent_chunk_id")
        if parent_id:
            children_by_parent[parent_id].append(record)

    for record in non_aux_with_synthetic:
        record["retrieval_tier"] = "support" if record["_old_chunk_id"] in children_by_parent else "primary"

    materialize_support_records(non_aux_with_synthetic)

    output_records = sorted(
        [*working, *synthetic_records],
        key=lambda record: (
            record["_order"],
            0 if record.get("_synthetic") else 1,
            len(record.get("section_path") or []),
            tuple(record.get("section_path") or []),
        ),
    )

    provisional_to_new: dict[str, str] = {}
    for index, record in enumerate(output_records, 1):
        provisional_id = record["_old_chunk_id"]
        new_id = f"{record['doc_id']}_{index:04d}"
        provisional_to_new[provisional_id] = new_id
        record["chunk_id"] = new_id

    for record in output_records:
        parent_id = record.get("parent_chunk_id")
        if parent_id is None:
            continue
        if parent_id not in provisional_to_new:
            raise ValueError(f"dangling parent reference before renumber: {parent_id}")
        record["parent_chunk_id"] = provisional_to_new[parent_id]

    for record in output_records:
        record.pop("_old_chunk_id", None)
        record.pop("_order", None)
        record.pop("_synthetic", None)

    report = {
        "input_records": len(records),
        "output_records": len(output_records),
        "created_support_nodes": len(synthetic_records),
        "created_by_depth": dict(Counter(len(record["section_path"]) for record in synthetic_records)),
    }
    return output_records, report


def validate_output(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {record["chunk_id"] for record in records}
    dangling = [
        (record["chunk_id"], record.get("parent_chunk_id"))
        for record in records
        if record.get("parent_chunk_id") is not None and record.get("parent_chunk_id") not in ids
    ]
    non_aux = [record for record in records if not is_aux_chunk(record)]
    exact_paths = {tuple(record.get("section_path") or []) for record in non_aux}
    missing_prefixes: list[tuple[str, ...]] = []
    for record in non_aux:
        path = tuple(record.get("section_path") or [])
        for depth in range(1, len(path) + 1):
            prefix = path[:depth]
            if prefix not in exact_paths:
                missing_prefixes.append(prefix)
    return {
        "dangling_parent_count": len(dangling),
        "dangling_parents": dangling[:10],
        "missing_prefix_count": len(set(missing_prefixes)),
        "missing_prefixes": [list(prefix) for prefix in sorted(set(missing_prefixes))[:10]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/manuals/chunks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manuals/new_chunk/chunk_integrate5"),
    )
    args = parser.parse_args()

    totals = Counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ENGLISH_FILES:
        input_path = args.input_dir / filename
        output_path = args.output_dir / filename
        records = load_jsonl(input_path)
        output_records, report = complete_one_file(records)
        validation = validate_output(output_records)
        if validation["dangling_parent_count"] or validation["missing_prefix_count"]:
            raise ValueError(f"{filename}: validation failed: {validation}")
        dump_jsonl(output_path, output_records)
        totals["files"] += 1
        totals["input_records"] += report["input_records"]
        totals["output_records"] += report["output_records"]
        totals["created_support_nodes"] += report["created_support_nodes"]
        print(
            f"{filename}: {report['input_records']} -> {report['output_records']}, "
            f"created_support_nodes={report['created_support_nodes']}, "
            f"created_by_depth={report['created_by_depth']}"
        )
    print(
        "TOTAL: "
        f"files={totals['files']}, input_records={totals['input_records']}, "
        f"output_records={totals['output_records']}, "
        f"created_support_nodes={totals['created_support_nodes']}"
    )


if __name__ == "__main__":
    main()
