"""Integrate the 20 English manual chunk files after parent repair.

This script is intentionally separate from integrate_manual_chunks.py because
the English pass uses different size thresholds and relies on the repaired
section_path-derived parent tree.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EN_MIN_CHUNK_SIZE = 800
EN_TARGET_CHUNK_SIZE = 1600
EN_MAX_CHUNK_SIZE = 3200

AUX_TYPES = {"metadata_image_path", "aux_navigation"}

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

NEW_CHUNK_TYPES = {
    "product_overview",
    "component_description",
    "assembly_steps",
    "operation_guide",
    "maintenance_guide",
    "troubleshooting_qa",
    "safety_warning",
    "legal_statement",
    "parts_list",
    "specification_table",
    "general_info",
}
ALLOWED_CHUNK_TYPES = NEW_CHUNK_TYPES | AUX_TYPES

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
MARKDOWN_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANKS_RE = re.compile(r"\n{3,}")
EN_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?](?=\s|$)|[^.!?\n]+$")

QUALITY_ORDER = {
    "normal": 0,
    "warning": 1,
    "needs_review": 2,
    "issue": 3,
}


@dataclass
class Group:
    old_ids: list[str]
    parent_old_id: str | None
    content_records: list[dict[str, Any]]
    first_order: int
    retrieval_tier: str
    contains_aux: bool = False
    collapsed_parent_child: bool = False
    merged_siblings: bool = False
    forced_over_max: bool = False
    padded: bool = False


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


def unique_key(item: Any) -> str:
    if isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False, sort_keys=True)
    return str(item)


def unique_in_order(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for item in items:
        key = unique_key(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def is_aux_chunk(record: dict[str, Any]) -> bool:
    return record.get("chunk_type") in AUX_TYPES or record.get("retrieval_tier") == "auxiliary"


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


def dedupe_lines(text: str) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if result and result[-1]:
                result.append("")
            continue
        key = WHITESPACE_RE.sub(" ", line)
        if key in seen:
            continue
        seen.add(key)
        result.append(line)
    while result and not result[-1]:
        result.pop()
    return "\n".join(result)


def body_starts_with_title(body: str, title: str) -> bool:
    if not body or not title:
        return False
    first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
    return first_line == title or body.strip().startswith(title)


def build_index_text(record: dict[str, Any], pad_before: str = "", pad_after: str = "") -> str:
    section_path = [str(part) for part in record.get("section_path") or [] if part]
    path = markdown_to_plain_text(" > ".join(section_path)) if section_path else ""
    title = markdown_to_plain_text(str(record.get("title") or ""))
    body = markdown_to_plain_text(str(record.get("text") or ""))
    pieces: list[str] = []
    if path:
        pieces.append(path)
    elif title:
        pieces.append(title)
    if title and title != path and not body_starts_with_title(body, title):
        pieces.append(title)
    if pad_before.strip():
        pieces.append(pad_before.strip())
    if body:
        pieces.append(body)
    if pad_after.strip():
        pieces.append(pad_after.strip())
    return dedupe_lines("\n".join(piece for piece in pieces if piece))


def group_text(records: list[dict[str, Any]]) -> str:
    return "\n\n".join(str(record.get("text") or "") for record in records if str(record.get("text") or ""))


def group_probe(group: Group) -> dict[str, Any]:
    first = group.content_records[0]
    return {
        "title": first.get("title"),
        "section_path": first.get("section_path") or [],
        "text": group_text(group.content_records),
    }


def group_index_text(group: Group) -> str:
    return build_index_text(group_probe(group))


def group_index_len(group: Group) -> int:
    return len(group_index_text(group))


def group_is_mergeable_primary(group: Group) -> bool:
    return not group.contains_aux and group.retrieval_tier == "primary" and group_index_len(group) <= EN_MAX_CHUNK_SIZE


def source_lines(records: list[dict[str, Any]]) -> list[int] | None:
    starts: list[int] = []
    ends: list[int] = []
    for record in records:
        raw = record.get("source_lines")
        if isinstance(raw, list) and len(raw) >= 2 and raw[0] is not None and raw[1] is not None:
            starts.append(int(raw[0]))
            ends.append(int(raw[1]))
    if not starts or not ends:
        return None
    return [min(starts), max(ends)]


def pic_description_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        raw_items = record.get("pic_descriptions") or []
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict) or not item.get("pic_id"):
                continue
            key = str(item["pic_id"])
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
    return result


def worst_quality(records: list[dict[str, Any]]) -> str:
    return max((str(record.get("source_quality") or "normal") for record in records), key=lambda q: QUALITY_ORDER.get(q, 0))


def merged_issue_flags(records: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for record in records:
        flags.extend(str(flag) for flag in record.get("source_issue_flags") or [])
    return unique_in_order(flags)


def merged_issue_note(records: list[dict[str, Any]]) -> str:
    notes = [
        str(record.get("source_issue_note") or "").strip()
        for record in records
        if str(record.get("source_issue_note") or "").strip()
    ]
    return "; ".join(unique_in_order(notes))


def classify_chunk_type(records: list[dict[str, Any]]) -> str:
    old_types = [str(record.get("chunk_type") or "") for record in records if record.get("chunk_type")]
    if old_types and all(chunk_type in AUX_TYPES for chunk_type in old_types) and len(set(old_types)) == 1:
        return old_types[0]

    old_type_set = set(old_types)
    signal = "\n".join(
        " ".join(
            [
                " > ".join(str(part) for part in record.get("section_path") or []),
                str(record.get("title") or ""),
                str(record.get("index_text") or ""),
                str(record.get("text") or ""),
            ]
        )
        for record in records
    ).lower()

    def has(*keywords: str) -> bool:
        return any(keyword in signal for keyword in keywords)

    if "legal_clause" in old_type_set or has("warranty", "copyright", "fcc", "compliance", "liability", "trademark"):
        return "legal_statement"
    if has("warning", "caution", "danger", "safety", "electric shock", "injury", "fire hazard"):
        return "safety_warning"
    if has("troubleshooting", "problem", "cause", "solution", "error code", "cannot", "does not work"):
        return "troubleshooting_qa"
    if has("package contents", "parts list", "supplied accessories", "included", "contents of package"):
        return "parts_list"
    if "table" in old_type_set or has("specification", "specifications", "rating", "dimension", "voltage", "model", "capacity"):
        return "specification_table"
    if has("install", "installation", "assemble", "assembly", "mount", "connect", "insert", "remove", "attach", "replace"):
        return "assembly_steps"
    if has("clean", "cleaning", "maintenance", "store", "storage", "inspect", "lubricat", "filter", "service"):
        return "maintenance_guide"
    if "procedure_step" in old_type_set or has("press", "select", "set", "start", "stop", "use ", "operate", "adjust", "configure"):
        return "operation_guide"
    if old_type_set & {"text_image_atomic", "feature_group"} or has("location", "overview", "component", "button", "connector", "control", "feature"):
        return "component_description"
    if "section_summary" in old_type_set or has("introduction", "about this", "description", "general information"):
        return "product_overview"
    return "general_info"


def merge_group_list(groups: list[Group], parent_old_id: str | None) -> list[Group]:
    if not groups:
        return []
    merged: list[Group] = []
    buffer: list[Group] = []
    forced_over_max = False

    def flush() -> None:
        nonlocal buffer, forced_over_max
        if not buffer:
            return
        if len(buffer) == 1:
            merged.append(buffer[0])
        else:
            content_records = [record for group in buffer for record in group.content_records]
            old_ids = [old_id for group in buffer for old_id in group.old_ids]
            merged.append(
                Group(
                    old_ids=old_ids,
                    parent_old_id=parent_old_id,
                    content_records=content_records,
                    first_order=min(group.first_order for group in buffer),
                    retrieval_tier="primary",
                    merged_siblings=True,
                    forced_over_max=forced_over_max,
                )
            )
        buffer = []
        forced_over_max = False

    for group in groups:
        if not group_is_mergeable_primary(group):
            flush()
            merged.append(group)
            continue
        if not buffer:
            buffer = [group]
            continue

        buffer_group = Group(
            old_ids=[old_id for item in buffer for old_id in item.old_ids],
            parent_old_id=parent_old_id,
            content_records=[record for item in buffer for record in item.content_records],
            first_order=min(item.first_order for item in buffer),
            retrieval_tier="primary",
        )
        buffer_len = group_index_len(buffer_group)
        combined_group = Group(
            old_ids=buffer_group.old_ids + group.old_ids,
            parent_old_id=parent_old_id,
            content_records=buffer_group.content_records + group.content_records,
            first_order=buffer_group.first_order,
            retrieval_tier="primary",
        )
        combined_len = group_index_len(combined_group)

        if buffer_len < EN_MIN_CHUNK_SIZE:
            buffer.append(group)
            forced_over_max = forced_over_max or combined_len > EN_MAX_CHUNK_SIZE
            continue
        if buffer_len < EN_TARGET_CHUNK_SIZE and combined_len <= EN_MAX_CHUNK_SIZE:
            buffer.append(group)
            continue
        flush()
        buffer = [group]

    flush()

    changed = True
    while changed:
        changed = False
        for idx, group in enumerate(list(merged)):
            if not group_is_mergeable_primary(group) or group_index_len(group) >= EN_MIN_CHUNK_SIZE:
                continue
            neighbor_idx = None
            if idx + 1 < len(merged):
                nxt = merged[idx + 1]
                combined = Group(
                    old_ids=group.old_ids + nxt.old_ids,
                    parent_old_id=parent_old_id,
                    content_records=group.content_records + nxt.content_records,
                    first_order=min(group.first_order, nxt.first_order),
                    retrieval_tier="primary",
                )
                if group_is_mergeable_primary(nxt) and group_index_len(combined) <= EN_MAX_CHUNK_SIZE:
                    neighbor_idx = idx + 1
            if neighbor_idx is None and idx > 0:
                prev = merged[idx - 1]
                combined = Group(
                    old_ids=prev.old_ids + group.old_ids,
                    parent_old_id=parent_old_id,
                    content_records=prev.content_records + group.content_records,
                    first_order=min(prev.first_order, group.first_order),
                    retrieval_tier="primary",
                )
                if group_is_mergeable_primary(prev) and group_index_len(combined) <= EN_MAX_CHUNK_SIZE:
                    neighbor_idx = idx - 1
            if neighbor_idx is None:
                continue

            first, second = sorted([idx, neighbor_idx])
            left = merged[first]
            right = merged[second]
            merged[first:second + 1] = [
                Group(
                    old_ids=left.old_ids + right.old_ids,
                    parent_old_id=parent_old_id,
                    content_records=left.content_records + right.content_records,
                    first_order=min(left.first_order, right.first_order),
                    retrieval_tier="primary",
                    merged_siblings=True,
                    forced_over_max=left.forced_over_max or right.forced_over_max,
                )
            ]
            changed = True
            break
    return merged


def integrate_non_aux_groups(records: list[dict[str, Any]]) -> list[Group]:
    by_id = {str(record["chunk_id"]): record for record in records}
    non_aux_ids = {chunk_id for chunk_id, record in by_id.items() if not is_aux_chunk(record)}
    order = {str(record["chunk_id"]): idx for idx, record in enumerate(records)}
    children_by_parent: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if is_aux_chunk(record):
            continue
        parent_id = record.get("parent_chunk_id")
        children_by_parent[str(parent_id) if parent_id else None].append(record)
    for children in children_by_parent.values():
        children.sort(key=lambda record: order[str(record["chunk_id"])])

    def process_node(record: dict[str, Any]) -> list[Group]:
        record_id = str(record["chunk_id"])
        direct_children = children_by_parent.get(record_id, [])
        if not direct_children:
            return [
                Group(
                    old_ids=[record_id],
                    parent_old_id=str(record.get("parent_chunk_id")) if record.get("parent_chunk_id") else None,
                    content_records=[record],
                    first_order=order[record_id],
                    retrieval_tier="primary",
                )
            ]

        child_outputs: list[Group] = []
        primary_buffer: list[Group] = []
        for child in direct_children:
            child_groups = process_node(child)
            direct_child_is_single_primary = (
                len(child_groups) == 1
                and child_groups[0].retrieval_tier == "primary"
                and child_groups[0].parent_old_id == record_id
            )
            if direct_child_is_single_primary:
                primary_buffer.append(child_groups[0])
                continue
            child_outputs.extend(merge_group_list(primary_buffer, record_id))
            primary_buffer = []
            child_outputs.extend(child_groups)
        child_outputs.extend(merge_group_list(primary_buffer, record_id))

        if len(child_outputs) == 1:
            child_group = child_outputs[0]
            return [
                Group(
                    old_ids=[record_id] + child_group.old_ids,
                    parent_old_id=str(record.get("parent_chunk_id")) if record.get("parent_chunk_id") else None,
                    content_records=[record],
                    first_order=order[record_id],
                    retrieval_tier="primary",
                    collapsed_parent_child=True,
                    forced_over_max=child_group.forced_over_max,
                )
            ]

        return [
            Group(
                old_ids=[record_id],
                parent_old_id=str(record.get("parent_chunk_id")) if record.get("parent_chunk_id") else None,
                content_records=[record],
                first_order=order[record_id],
                retrieval_tier="support",
            )
        ] + child_outputs

    roots = [
        record
        for record in records
        if not is_aux_chunk(record)
        and (
            not record.get("parent_chunk_id")
            or str(record.get("parent_chunk_id")) not in non_aux_ids
        )
    ]
    roots.sort(key=lambda record: order[str(record["chunk_id"])])

    output: list[Group] = []
    root_primary_buffer: list[Group] = []
    for root in roots:
        root_groups = process_node(root)
        direct_root_is_single_primary = (
            len(root_groups) == 1
            and root_groups[0].retrieval_tier == "primary"
            and root_groups[0].parent_old_id is None
        )
        if direct_root_is_single_primary:
            root_primary_buffer.append(root_groups[0])
            continue
        output.extend(merge_group_list(root_primary_buffer, None))
        root_primary_buffer = []
        output.extend(root_groups)
    output.extend(merge_group_list(root_primary_buffer, None))
    return output


def aux_groups(records: list[dict[str, Any]]) -> list[Group]:
    groups: list[Group] = []
    for idx, record in enumerate(records):
        if not is_aux_chunk(record):
            continue
        record_id = str(record["chunk_id"])
        groups.append(
            Group(
                old_ids=[record_id],
                parent_old_id=str(record.get("parent_chunk_id")) if record.get("parent_chunk_id") else None,
                content_records=[record],
                first_order=idx,
                retrieval_tier="auxiliary",
                contains_aux=True,
            )
        )
    return groups


def resolve_parent_group(group: Group, old_to_group: dict[str, Group]) -> Group | None:
    if not group.parent_old_id:
        return None
    parent_group = old_to_group.get(str(group.parent_old_id))
    if parent_group is group:
        return None
    return parent_group


def merge_final_primary_siblings(groups: list[Group]) -> list[Group]:
    changed = True
    while changed:
        changed = False
        old_to_group: dict[str, Group] = {}
        for group in groups:
            for old_id in group.old_ids:
                old_to_group[old_id] = group

        idx = 0
        while idx + 1 < len(groups):
            left = groups[idx]
            right = groups[idx + 1]
            left_parent = resolve_parent_group(left, old_to_group)
            right_parent = resolve_parent_group(right, old_to_group)
            if left_parent is not right_parent:
                idx += 1
                continue
            if not group_is_mergeable_primary(left) or not group_is_mergeable_primary(right):
                idx += 1
                continue

            candidate_parent_old_id = left.parent_old_id
            if left_parent is not None:
                candidate_parent_old_id = left_parent.old_ids[0]
            combined = Group(
                old_ids=left.old_ids + right.old_ids,
                parent_old_id=candidate_parent_old_id,
                content_records=left.content_records + right.content_records,
                first_order=min(left.first_order, right.first_order),
                retrieval_tier="primary",
                merged_siblings=True,
                forced_over_max=left.forced_over_max or right.forced_over_max,
            )
            left_len = group_index_len(left)
            right_len = group_index_len(right)
            combined_len = group_index_len(combined)
            should_merge = (
                left_len < EN_MIN_CHUNK_SIZE
                or right_len < EN_MIN_CHUNK_SIZE
                or (left_len < EN_TARGET_CHUNK_SIZE and combined_len <= EN_MAX_CHUNK_SIZE)
            )
            if not should_merge:
                idx += 1
                continue
            if combined_len > EN_MAX_CHUNK_SIZE and left_len >= EN_MIN_CHUNK_SIZE and right_len >= EN_MIN_CHUNK_SIZE:
                idx += 1
                continue
            combined.forced_over_max = combined.forced_over_max or combined_len > EN_MAX_CHUNK_SIZE
            groups[idx:idx + 2] = [combined]
            changed = True
            break
    return groups


def collapse_single_primary_leaf_children(groups: list[Group]) -> list[Group]:
    changed = True
    while changed:
        changed = False
        old_to_group: dict[str, Group] = {}
        for group in groups:
            for old_id in group.old_ids:
                old_to_group[old_id] = group

        for parent_group in list(groups):
            if parent_group.contains_aux or parent_group.retrieval_tier != "support":
                continue
            child_groups = [
                group
                for group in groups
                if group is not parent_group
                and not group.contains_aux
                and resolve_parent_group(group, old_to_group) is parent_group
            ]
            if len(child_groups) != 1:
                continue
            child_group = child_groups[0]
            if child_group.retrieval_tier != "primary":
                continue
            parent_group.old_ids.extend(child_group.old_ids)
            parent_group.retrieval_tier = "primary"
            parent_group.collapsed_parent_child = True
            parent_group.forced_over_max = parent_group.forced_over_max or child_group.forced_over_max
            groups.remove(child_group)
            changed = True
            break
    return groups


def output_record_from_group(group: Group, new_id: str, new_parent_id: str | None) -> dict[str, Any]:
    first = group.content_records[0]
    if group.contains_aux:
        record = dict(first)
        record["chunk_id"] = new_id
        record["parent_chunk_id"] = new_parent_id
        return record

    content_records = group.content_records
    merged: dict[str, Any] = {
        "chunk_id": new_id,
        "doc_id": first.get("doc_id"),
        "source_file": first.get("source_file"),
        "source_lines": source_lines(content_records),
        "chunk_type": classify_chunk_type(content_records),
        "retrieval_tier": group.retrieval_tier,
        "section_path": first.get("section_path") or [],
        "title": first.get("title"),
        "parent_chunk_id": new_parent_id,
        "pic_ids": unique_in_order([pic_id for record in content_records for pic_id in (record.get("pic_ids") or [])]),
        "text": group_text(content_records),
        "source_quality": worst_quality(content_records),
        "source_issue_flags": merged_issue_flags(content_records),
        "source_issue_note": merged_issue_note(content_records),
        "item_no": first.get("item_no"),
        "item_kind": first.get("item_kind"),
        "language": first.get("language"),
    }
    descriptions = pic_description_items(content_records)
    if descriptions:
        merged["pic_descriptions"] = descriptions
    elif "pic_descriptions" in first:
        merged["pic_descriptions"] = None
    merged["index_text"] = build_index_text(merged)
    return merged


def clean_context_text(source: str, current: dict[str, Any]) -> str:
    forbidden = {markdown_to_plain_text(str(current.get("title") or ""))}
    path_parts = [str(part) for part in current.get("section_path") or [] if part]
    forbidden.update(markdown_to_plain_text(part) for part in path_parts)
    forbidden.add(markdown_to_plain_text(" > ".join(path_parts)))
    clean_lines: list[str] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line in {"...", "..."}:
            continue
        plain = markdown_to_plain_text(line)
        if not plain or plain in forbidden or " > " in line:
            continue
        clean_lines.append(plain)
    return dedupe_lines("\n".join(clean_lines))


def sentence_parts(text: str) -> list[str]:
    normalized = WHITESPACE_RE.sub(" ", text.replace("\n", " ")).strip()
    return [match.group(0).strip() for match in EN_SENTENCE_RE.finditer(normalized) if match.group(0).strip()]


def context_from_previous(source: str, current: dict[str, Any], target: int) -> str:
    cleaned = clean_context_text(source, current)
    sentences = sentence_parts(cleaned)
    if not sentences:
        return ""
    chosen: list[str] = []
    for sentence in reversed(sentences):
        chosen.insert(0, sentence)
        if len(" ".join(chosen)) >= target:
            break
    return " ".join(chosen)


def context_from_next(source: str, current: dict[str, Any], target: int) -> str:
    cleaned = clean_context_text(source, current)
    sentences = sentence_parts(cleaned)
    if not sentences:
        return ""
    chosen: list[str] = []
    for sentence in sentences:
        chosen.append(sentence)
        if len(" ".join(chosen)) >= target:
            break
    return " ".join(chosen)


def pad_short_primary_records(records: list[dict[str, Any]]) -> list[str]:
    by_parent: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("retrieval_tier") == "primary":
            parent = str(record.get("parent_chunk_id")) if record.get("parent_chunk_id") else None
            by_parent[parent].append(record)

    padded_ids: list[str] = []
    for siblings in by_parent.values():
        for idx, record in enumerate(siblings):
            current = str(record.get("index_text") or "")
            if len(current) >= EN_MIN_CHUNK_SIZE:
                continue

            previous_record = siblings[idx - 1] if idx > 0 else None
            next_record = siblings[idx + 1] if idx + 1 < len(siblings) else None
            needed = EN_MIN_CHUNK_SIZE - len(current)
            before = ""
            after = ""
            if previous_record is not None:
                previous_target = max(needed // (2 if next_record is not None else 1), 160)
                before = context_from_previous(str(previous_record.get("index_text") or ""), record, previous_target)
            if next_record is not None:
                next_target = max(needed // (2 if previous_record is not None else 1), 160)
                after = context_from_next(str(next_record.get("index_text") or ""), record, next_target)
            if not before and not after:
                continue
            updated = build_index_text(record, pad_before=before, pad_after=after)
            if len(updated) < EN_MIN_CHUNK_SIZE:
                if previous_record is not None:
                    before = context_from_previous(str(previous_record.get("index_text") or ""), record, needed)
                if next_record is not None:
                    after = context_from_next(str(next_record.get("index_text") or ""), record, needed)
                updated = build_index_text(record, pad_before=before, pad_after=after)
            if updated != current:
                record["index_text"] = updated
                padded_ids.append(str(record["chunk_id"]))
    return padded_ids


def integrate_records(input_path: Path, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    non_aux = integrate_non_aux_groups(records)
    groups = non_aux + aux_groups(records)
    groups.sort(key=lambda group: group.first_order)
    groups = merge_final_primary_siblings(groups)
    groups.sort(key=lambda group: group.first_order)
    groups = collapse_single_primary_leaf_children(groups)
    groups.sort(key=lambda group: group.first_order)

    doc_id = str(records[0].get("doc_id") or input_path.stem)
    group_id_by_obj = {id(group): f"{doc_id}_{idx:04d}" for idx, group in enumerate(groups, 1)}

    old_to_new: dict[str, str] = {}
    duplicate_old_ids: list[str] = []
    for group in groups:
        new_id = group_id_by_obj[id(group)]
        for old_id in group.old_ids:
            if old_id in old_to_new:
                duplicate_old_ids.append(old_id)
            old_to_new[old_id] = new_id

    output: list[dict[str, Any]] = []
    for group in groups:
        new_id = group_id_by_obj[id(group)]
        new_parent_id = old_to_new.get(str(group.parent_old_id)) if group.parent_old_id else None
        if new_parent_id == new_id:
            new_parent_id = None
        output.append(output_record_from_group(group, new_id, new_parent_id))

    padded_ids = pad_short_primary_records(output)
    report = build_report(input_path, records, output, groups, old_to_new, duplicate_old_ids, padded_ids)
    return output, report


def build_report(
    input_path: Path,
    original: list[dict[str, Any]],
    output: list[dict[str, Any]],
    groups: list[Group],
    old_to_new: dict[str, str],
    duplicate_old_ids: list[str],
    padded_ids: list[str],
) -> dict[str, Any]:
    original_ids = {str(record["chunk_id"]) for record in original}
    new_ids = {str(record["chunk_id"]) for record in output}
    output_by_id = {str(record["chunk_id"]): record for record in output}
    broken_new_parent_refs = [
        record["chunk_id"]
        for record in output
        if record.get("parent_chunk_id") and str(record["parent_chunk_id"]) not in new_ids
    ]
    self_parent_refs = [
        record["chunk_id"]
        for record in output
        if record.get("parent_chunk_id") and str(record["parent_chunk_id"]) == str(record["chunk_id"])
    ]
    parent_child_counts = Counter(str(record.get("parent_chunk_id")) for record in output if record.get("parent_chunk_id"))
    non_aux_children = defaultdict(list)
    for record in output:
        if record.get("parent_chunk_id") and not is_aux_chunk(record):
            non_aux_children[str(record["parent_chunk_id"])].append(record)

    tier_mismatches: list[str] = []
    for record in output:
        if is_aux_chunk(record):
            if record.get("retrieval_tier") != "auxiliary":
                tier_mismatches.append(str(record["chunk_id"]))
            continue
        has_non_aux_children = bool(non_aux_children.get(str(record["chunk_id"])))
        expected = "support" if has_non_aux_children else "primary"
        if record.get("retrieval_tier") != expected:
            tier_mismatches.append(str(record["chunk_id"]))

    original_pic_ids = set(pic_id for record in original for pic_id in (record.get("pic_ids") or []))
    output_pic_ids = set(pic_id for record in output for pic_id in (record.get("pic_ids") or []))
    original_pic_desc_ids = {
        str(item["pic_id"])
        for record in original
        for item in (record.get("pic_descriptions") or [])
        if isinstance(item, dict) and item.get("pic_id")
    }
    output_pic_desc_ids = {
        str(item["pic_id"])
        for record in output
        for item in (record.get("pic_descriptions") or [])
        if isinstance(item, dict) and item.get("pic_id")
    }

    lengths = [len(str(record.get("index_text") or "")) for record in output]
    primary_lengths = [len(str(record.get("index_text") or "")) for record in output if record.get("retrieval_tier") == "primary"]
    support_lengths = [len(str(record.get("index_text") or "")) for record in output if record.get("retrieval_tier") == "support"]
    auxiliary_short_ids = [
        record["chunk_id"]
        for record in output
        if is_aux_chunk(record) and len(str(record.get("index_text") or "")) < EN_MIN_CHUNK_SIZE
    ]
    short_primary_ids = [
        record["chunk_id"]
        for record in output
        if record.get("retrieval_tier") == "primary" and len(str(record.get("index_text") or "")) < EN_MIN_CHUNK_SIZE
    ]
    support_short_ids = [
        record["chunk_id"]
        for record in output
        if record.get("retrieval_tier") == "support" and len(str(record.get("index_text") or "")) < EN_MIN_CHUNK_SIZE
    ]
    id_format_errors = [
        record["chunk_id"]
        for idx, record in enumerate(output, 1)
        if str(record["chunk_id"]) != f"{record.get('doc_id')}_{idx:04d}"
    ]

    aux_value_changes = []
    original_by_id = {str(record["chunk_id"]): record for record in original}
    for group in groups:
        if not group.contains_aux:
            continue
        old_id = group.old_ids[0]
        new_id = old_to_new[old_id]
        old_record = original_by_id[old_id]
        new_record = output_by_id[new_id]
        for key, old_value in old_record.items():
            if key in {"chunk_id", "parent_chunk_id"}:
                continue
            if new_record.get(key) != old_value:
                aux_value_changes.append({"old_id": old_id, "new_id": new_id, "field": key})

    return {
        "input_file": input_path.name,
        "original_records": len(original),
        "output_records": len(output),
        "reduction": len(original) - len(output),
        "original_chunk_ids": len(original_ids),
        "new_chunk_ids": len(new_ids),
        "all_original_ids_mapped": len(old_to_new) == len(original_ids),
        "duplicate_old_ids": duplicate_old_ids,
        "old_to_new": old_to_new,
        "broken_new_parent_refs": broken_new_parent_refs,
        "self_parent_refs": self_parent_refs,
        "id_format_errors": id_format_errors,
        "tier_mismatches": tier_mismatches,
        "missing_pic_ids": sorted(original_pic_ids - output_pic_ids),
        "extra_pic_ids": sorted(output_pic_ids - original_pic_ids),
        "missing_pic_description_ids": sorted(original_pic_desc_ids - output_pic_desc_ids),
        "extra_pic_description_ids": sorted(output_pic_desc_ids - original_pic_desc_ids),
        "type_counts": dict(Counter(str(record.get("chunk_type") or "") for record in output)),
        "tier_counts": dict(Counter(str(record.get("retrieval_tier") or "") for record in output)),
        "index_length": {
            "lt_800": sum(1 for length in lengths if length < EN_MIN_CHUNK_SIZE),
            "between_800_3200": sum(1 for length in lengths if EN_MIN_CHUNK_SIZE <= length <= EN_MAX_CHUNK_SIZE),
            "gt_3200": sum(1 for length in lengths if length > EN_MAX_CHUNK_SIZE),
            "primary_lt_800": len(short_primary_ids),
            "support_lt_800": len(support_short_ids),
            "auxiliary_lt_800": len(auxiliary_short_ids),
            "primary_gt_3200": sum(1 for length in primary_lengths if length > EN_MAX_CHUNK_SIZE),
            "support_gt_3200": sum(1 for length in support_lengths if length > EN_MAX_CHUNK_SIZE),
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "short_primary_ids": short_primary_ids,
        "support_short_ids": support_short_ids,
        "auxiliary_short_ids": auxiliary_short_ids,
        "padded_chunk_ids": padded_ids,
        "auxiliary_padded_ids": [
            chunk_id for chunk_id in padded_ids if is_aux_chunk(output_by_id.get(str(chunk_id), {}))
        ],
        "original_auxiliary_records": sum(1 for record in original if is_aux_chunk(record)),
        "output_auxiliary_records": sum(1 for record in output if is_aux_chunk(record)),
        "aux_value_changes": aux_value_changes,
        "collapsed_parent_child_count": sum(1 for group in groups if group.collapsed_parent_child),
        "merged_sibling_group_count": sum(1 for group in groups if group.merged_siblings),
        "forced_over_max_count": sum(1 for group in groups if group.forced_over_max),
        "single_child_parent_ids": sorted(parent_id for parent_id, count in parent_child_counts.items() if count == 1),
        "disallowed_chunk_types": sorted({str(record.get("chunk_type") or "") for record in output} - ALLOWED_CHUNK_TYPES),
        "index_text_pic_occurrences": sum(str(record.get("index_text") or "").count("<PIC:") for record in output),
    }


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    output_records = sum(int(report.get("output_records") or 0) for report in reports)
    return {
        "files": len(reports),
        "original_records": sum(int(report.get("original_records") or 0) for report in reports),
        "output_records": output_records,
        "reduction": sum(int(report.get("reduction") or 0) for report in reports),
        "index_length": {
            "lt_800": sum(int((report.get("index_length") or {}).get("lt_800") or 0) for report in reports),
            "between_800_3200": sum(int((report.get("index_length") or {}).get("between_800_3200") or 0) for report in reports),
            "gt_3200": sum(int((report.get("index_length") or {}).get("gt_3200") or 0) for report in reports),
            "primary_lt_800": sum(int((report.get("index_length") or {}).get("primary_lt_800") or 0) for report in reports),
            "support_lt_800": sum(int((report.get("index_length") or {}).get("support_lt_800") or 0) for report in reports),
            "auxiliary_lt_800": sum(int((report.get("index_length") or {}).get("auxiliary_lt_800") or 0) for report in reports),
            "primary_gt_3200": sum(int((report.get("index_length") or {}).get("primary_gt_3200") or 0) for report in reports),
            "support_gt_3200": sum(int((report.get("index_length") or {}).get("support_gt_3200") or 0) for report in reports),
        },
        "padded_chunks": sum(len(report.get("padded_chunk_ids") or []) for report in reports),
        "collapsed_parent_child": sum(int(report.get("collapsed_parent_child_count") or 0) for report in reports),
        "merged_sibling_groups": sum(int(report.get("merged_sibling_group_count") or 0) for report in reports),
        "forced_over_max": sum(int(report.get("forced_over_max_count") or 0) for report in reports),
        "original_auxiliary_records": sum(int(report.get("original_auxiliary_records") or 0) for report in reports),
        "output_auxiliary_records": sum(int(report.get("output_auxiliary_records") or 0) for report in reports),
        "files_with_broken_parent_refs": [report["input_file"] for report in reports if report.get("broken_new_parent_refs")],
        "files_with_self_parent_refs": [report["input_file"] for report in reports if report.get("self_parent_refs")],
        "files_with_id_format_errors": [report["input_file"] for report in reports if report.get("id_format_errors")],
        "files_with_tier_mismatches": [report["input_file"] for report in reports if report.get("tier_mismatches")],
        "files_with_missing_pic_ids": [report["input_file"] for report in reports if report.get("missing_pic_ids")],
        "files_with_missing_pic_descriptions": [report["input_file"] for report in reports if report.get("missing_pic_description_ids")],
        "files_with_aux_value_changes": [report["input_file"] for report in reports if report.get("aux_value_changes")],
        "files_with_disallowed_chunk_types": [report["input_file"] for report in reports if report.get("disallowed_chunk_types")],
        "files_with_index_pic_occurrences": [report["input_file"] for report in reports if report.get("index_text_pic_occurrences")],
        "type_counts": dict(sum((Counter(report.get("type_counts") or {}) for report in reports), Counter())),
        "tier_counts": dict(sum((Counter(report.get("tier_counts") or {}) for report in reports), Counter())),
    }


def integrate_english_dir(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    for filename in ENGLISH_FILES:
        input_path = input_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        output_path = output_dir / filename
        report_path = report_dir / f"{input_path.stem}__integrate_report.json"
        records = load_jsonl(input_path)
        output, report = integrate_records(input_path, records)
        dump_jsonl(output_path, output)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append(report)

    summary = summarize_reports(reports)
    summary["input_dir"] = str(input_dir)
    summary["output_dir"] = str(output_dir)
    summary["files_processed"] = ENGLISH_FILES
    (output_dir / "integration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/manuals/chunks")
    parser.add_argument("--output-dir", default="data/manuals/new_chunk/chunk_integrate4")
    args = parser.parse_args()
    summary = integrate_english_dir(Path(args.input_dir), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
