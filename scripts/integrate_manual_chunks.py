"""Merge enriched manual chunks into denser retrieval chunks."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any


MIN_CHUNK_SIZE = 200
TARGET_CHUNK_SIZE = 400
MAX_CHUNK_SIZE = 800

AUX_TYPES = {"metadata_image_path", "aux_navigation"}
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

PIC_WITH_DESC_RE = re.compile(
    r"<PIC:([^>]+)>(?:\s*\{\{(?:此图片描述为|图片描述|image description)[:：]\s*(.*?)\}\})?",
    re.DOTALL | re.IGNORECASE,
)
DESC_WRAPPER_RE = re.compile(
    r"\{\{(?:此图片描述为|图片描述|image description)[:：]\s*(.*?)\}\}",
    re.DOTALL | re.IGNORECASE,
)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MARKDOWN_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
MARKDOWN_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
MARKDOWN_BULLET_RE = re.compile(r"^\s*[-*+·]\s*", re.MULTILINE)
MARKDOWN_ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
MARKDOWN_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANKS_RE = re.compile(r"\n{3,}")
SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")

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


def pic_description_items(records: list[dict[str, Any]]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for record in records:
        raw_items = record.get("pic_descriptions") or []
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if isinstance(item, dict) and item.get("pic_id"):
                key = f"pic_id:{item['pic_id']}"
            else:
                key = unique_key(item)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
    return result


def is_aux_chunk(record: dict[str, Any]) -> bool:
    return record.get("chunk_type") in AUX_TYPES or record.get("retrieval_tier") == "auxiliary"


def original_index_len(record: dict[str, Any]) -> int:
    return len(str(record.get("index_text") or ""))


def is_original_over_max(record: dict[str, Any]) -> bool:
    return original_index_len(record) > MAX_CHUNK_SIZE


def markdown_to_plain_text(text: str) -> str:
    text = PIC_WITH_DESC_RE.sub(lambda m: f" {m.group(2) or ''} ", text)
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


def manual_short_name(input_path: Path, first_record: dict[str, Any]) -> str:
    stem = input_path.stem
    if "__" in stem:
        left, right = stem.split("__", 1)
        candidate = right if left.startswith(("manual_", "vr_")) else left
    else:
        candidate = str(first_record.get("doc_id") or stem)
    candidate = candidate.replace("手册", "").replace("使用说明书", "")
    candidate = candidate.strip("_- ")
    return candidate or str(first_record.get("doc_id") or stem)


def short_path(record: dict[str, Any], doc_short: str) -> str:
    path = [str(part) for part in record.get("section_path") or [] if part]
    cleaned = [
        part
        for part in path
        if part not in {"1. 重建正文", "重建正文"}
        and not part.startswith("文档元数据")
        and not part.startswith("图片锚点顺序")
    ]
    parent_section = ""
    if len(cleaned) > 1:
        parent_section = cleaned[-2]
    elif cleaned:
        parent_section = cleaned[-1]
    parent_section = markdown_to_plain_text(parent_section)
    if parent_section and parent_section != doc_short:
        return f"{doc_short} > {parent_section}"
    return doc_short


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


def build_index_text(record: dict[str, Any], doc_short: str, pad_context: str = "") -> str:
    title = markdown_to_plain_text(str(record.get("title") or ""))
    body = markdown_to_plain_text(str(record.get("text") or ""))
    path = short_path(record, doc_short)
    pieces: list[str] = []
    if title and not body_starts_with_title(body, title):
        pieces.append(title)
    if pad_context.strip():
        pieces.append(pad_context.strip())
    if body:
        pieces.append(body)
    if path:
        pieces.append(path)
    return dedupe_lines("\n".join(piece for piece in pieces if piece))


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


def worst_quality(records: list[dict[str, Any]]) -> str:
    return max((str(r.get("source_quality") or "normal") for r in records), key=lambda q: QUALITY_ORDER.get(q, 0))


def merged_issue_flags(records: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for record in records:
        flags.extend(str(flag) for flag in record.get("source_issue_flags") or [])
    return unique_in_order(flags)


def merged_issue_note(records: list[dict[str, Any]]) -> str:
    notes = [str(r.get("source_issue_note") or "").strip() for r in records if str(r.get("source_issue_note") or "").strip()]
    return "；".join(unique_in_order(notes))


def classify_chunk_type(records: list[dict[str, Any]]) -> str:
    old_types = [str(r.get("chunk_type") or "") for r in records if r.get("chunk_type")]
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
        return any(keyword.lower() in signal for keyword in keywords)

    if old_type_set & {"safety_clause", "caution_clause"} or has("安全", "警告", "危险", "禁止", "防护", "伤害", "佩戴"):
        return "safety_warning"
    if old_type_set & {"legal_clause"} or has("保修", "法律", "声明", "版权", "fcc", "合规", "责任限制", "回收", "废弃"):
        return "legal_statement"
    if old_type_set & {
        "troubleshooting_case",
        "troubleshooting_condition",
        "troubleshooting_cause",
        "troubleshooting_problem",
        "troubleshooting_answer",
    } or has("故障", "故障排除", "问题", "原因", "解决", "无法", "不能启动", "不工作"):
        return "troubleshooting_qa"
    if has("零件清单", "部件清单", "配件清单", "装箱清单", "清单表格", "parts list"):
        return "parts_list"
    if old_type_set & {"table"} or has(
        "技术参数",
        "规格",
        "参数",
        "额定",
        "尺寸",
        "重量",
        "载重",
        "电池",
        "电压",
        "保险丝",
        "型号",
        "适用年龄",
    ):
        return "specification_table"
    if has("组装", "安装", "装配", "固定", "插入", "连接", "拆卸", "接线", "充电步骤"):
        return "assembly_steps"
    if has("维护", "保养", "清洁", "更换", "检查", "滤清器", "滤网", "火花塞", "润滑", "存放", "维修"):
        return "maintenance_guide"
    if old_type_set & {"procedure_step", "procedure_overview"} or has(
        "操作",
        "使用",
        "启动",
        "停止",
        "设置",
        "模式",
        "按键",
        "控制",
        "配对",
        "拍摄",
        "播放",
        "开关",
        "充电",
    ):
        return "operation_guide"
    if has("产品介绍", "概述", "简介", "目录", "适用范围", "说明书"):
        return "product_overview"
    if old_type_set & {"text_image_atomic", "feature_group"} or has(
        "产品描述",
        "部件",
        "组件",
        "结构",
        "功能",
        "按钮",
        "接口",
        "外观",
        "图解",
        "位置",
        "标注",
    ):
        return "component_description"
    return "general_info"


def group_text(records: list[dict[str, Any]]) -> str:
    return "\n\n".join(str(record.get("text") or "") for record in records if str(record.get("text") or ""))


def group_index_text(records: list[dict[str, Any]], doc_short: str, pad_context: str = "") -> str:
    first = records[0]
    merged = {
        "title": first.get("title"),
        "section_path": first.get("section_path") or [],
        "text": group_text(records),
    }
    return build_index_text(merged, doc_short, pad_context=pad_context)


def group_index_len(records: list[dict[str, Any]], doc_short: str, pad_context: str = "") -> int:
    return len(group_index_text(records, doc_short, pad_context=pad_context))


def compatible_for_merge(buffer_records: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
    if is_aux_chunk(candidate) or is_original_over_max(candidate):
        return False
    if any(is_aux_chunk(record) or is_original_over_max(record) for record in buffer_records):
        return False
    buffer_tiers = {str(record.get("retrieval_tier") or "primary") for record in buffer_records}
    candidate_tier = str(candidate.get("retrieval_tier") or "primary")
    return len(buffer_tiers | {candidate_tier}) == 1


def is_padding_path_line(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped in {"...", "……", "重建正文", "1. 重建正文", "文档元数据", "图片锚点顺序"}
        or " > " in stripped
        or set(stripped) <= {".", "。", "…"}
    )


def padding_context_from_text(source_text: str, current_record: dict[str, Any], doc_short: str, needed: int = 0) -> str:
    forbidden = {
        markdown_to_plain_text(str(current_record.get("title") or "")),
        short_path(current_record, doc_short),
    }
    forbidden.update(markdown_to_plain_text(str(part)) for part in current_record.get("section_path") or [])
    clean_lines = []
    for raw_line in source_text.splitlines():
        line = raw_line.strip()
        if is_padding_path_line(line) or line in forbidden:
            continue
        clean_lines.append(line)
    clean_text = dedupe_lines("\n".join(clean_lines)).replace("...", "").replace("……", "")
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(clean_text) if match.group(0).strip()]
    complete = [sentence for sentence in sentences if sentence[-1:] in "。！？!?；;"]
    source_sentences = complete or sentences
    chosen: list[str] = []
    for sentence in reversed(source_sentences):
        chosen.insert(0, sentence)
        if len(chosen) >= 2 and (needed <= 0 or len("\n".join(chosen)) >= needed):
            break
    if chosen:
        return dedupe_lines("\n".join(chosen))

    chosen_lines: list[str] = []
    for line in reversed(clean_lines):
        chosen_lines.insert(0, line)
        if len(chosen_lines) >= 2 and (needed <= 0 or len("\n".join(chosen_lines)) >= needed):
            break
    return dedupe_lines("\n".join(chosen_lines))


def make_group(records: list[dict[str, Any]], **flags: Any) -> dict[str, Any]:
    group = {
        "records": records,
        "parent_old_id": records[0].get("parent_chunk_id") if records else None,
        "forced_over_max": False,
        "padded": False,
        "pad_context": "",
        "root_short_exception": False,
        "parent_child_collapse": False,
        "contains_aux": any(is_aux_chunk(record) for record in records),
        "contains_original_over_max": any(is_original_over_max(record) for record in records),
    }
    group.update(flags)
    return group


def merge_child_candidates(children: list[dict[str, Any]], doc_short: str) -> list[dict[str, Any]]:
    if not children:
        return []

    groups: list[dict[str, Any]] = []
    buffer = [children[0]]
    forced_over_max = False

    for child in children[1:]:
        buf_len = group_index_len(buffer, doc_short)
        combined_len = group_index_len(buffer + [child], doc_short)
        if compatible_for_merge(buffer, child) and buf_len < MIN_CHUNK_SIZE:
            buffer.append(child)
            forced_over_max = forced_over_max or combined_len > MAX_CHUNK_SIZE
            continue
        if compatible_for_merge(buffer, child) and buf_len < TARGET_CHUNK_SIZE and combined_len <= MAX_CHUNK_SIZE:
            buffer.append(child)
            continue
        groups.append(make_group(buffer, forced_over_max=forced_over_max))
        buffer = [child]
        forced_over_max = False

    final_group = make_group(buffer, forced_over_max=forced_over_max)
    if group_index_len(buffer, doc_short) < MIN_CHUNK_SIZE and groups:
        previous_text = group_index_text(groups[-1]["records"], doc_short, groups[-1].get("pad_context") or "")
        needed = MIN_CHUNK_SIZE - group_index_len(buffer, doc_short)
        context = padding_context_from_text(previous_text, buffer[0], doc_short, needed=needed)
        if context:
            final_group["pad_context"] = context
            final_group["padded"] = True
    groups.append(final_group)
    return groups


def collapse_single_child_groups(groups: list[dict[str, Any]]) -> None:
    """Collapse final non-auxiliary parent groups that only have one child group."""
    changed = True
    while changed:
        changed = False
        old_to_group: dict[str, dict[str, Any]] = {}
        for group in groups:
            for record in group["records"]:
                old_to_group[str(record["chunk_id"])] = group

        children_by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for group in groups:
            parent_old_id = group.get("parent_old_id")
            if not parent_old_id:
                continue
            parent_group = old_to_group.get(str(parent_old_id))
            if parent_group is None or parent_group is group:
                continue
            children_by_parent[id(parent_group)].append(group)

        for parent_group in list(groups):
            child_groups = children_by_parent.get(id(parent_group), [])
            if len(child_groups) != 1:
                continue
            child_group = child_groups[0]
            if parent_group.get("contains_aux") or child_group.get("contains_aux"):
                continue
            if parent_group.get("contains_original_over_max") or child_group.get("contains_original_over_max"):
                continue
            parent_group["records"].extend(child_group["records"])
            parent_group["parent_child_collapse"] = True
            parent_group["forced_over_max"] = bool(parent_group.get("forced_over_max") or child_group.get("forced_over_max"))
            parent_group["padded"] = bool(parent_group.get("padded") or child_group.get("padded"))
            if not parent_group.get("pad_context"):
                parent_group["pad_context"] = child_group.get("pad_context") or ""
            groups.remove(child_group)
            changed = True
            break


def integrate_records(input_path: Path, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records:
        return [], {"input_file": input_path.name}

    order = {str(record["chunk_id"]): idx for idx, record in enumerate(records)}
    children_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        parent_id = record.get("parent_chunk_id")
        if parent_id:
            children_map[str(parent_id)].append(record)
    for children in children_map.values():
        children.sort(key=lambda record: order[str(record["chunk_id"])])

    doc_short = manual_short_name(input_path, records[0])
    groups: list[dict[str, Any]] = []
    assigned: dict[str, dict[str, Any]] = {}

    def add_group(group: dict[str, Any]) -> None:
        duplicate_ids = [str(record["chunk_id"]) for record in group["records"] if str(record["chunk_id"]) in assigned]
        if duplicate_ids:
            raise ValueError(f"{input_path.name}: duplicate group assignment for {duplicate_ids}")
        groups.append(group)
        for record in group["records"]:
            assigned[str(record["chunk_id"])] = group

    for record in records:
        record_id = str(record["chunk_id"])
        direct_children = children_map.get(record_id, [])
        has_children = bool(direct_children)

        if has_children:
            eligible_children = [
                child
                for child in direct_children
                if str(child["chunk_id"]) not in assigned
                and not is_aux_chunk(child)
                and not is_original_over_max(child)
                and not children_map.get(str(child["chunk_id"]))
            ]
            other_non_aux_children = [
                child
                for child in direct_children
                if not is_aux_chunk(child) and child not in eligible_children
            ]
            child_groups = merge_child_candidates(eligible_children, doc_short)
            collapse_allowed = (
                record_id not in assigned
                and not is_aux_chunk(record)
                and not is_original_over_max(record)
                and len(child_groups) == 1
                and not other_non_aux_children
            )
            if collapse_allowed:
                add_group(
                    make_group(
                        [record] + child_groups[0]["records"],
                        parent_old_id=record.get("parent_chunk_id"),
                        parent_child_collapse=True,
                    )
                )
            else:
                if record_id not in assigned:
                    add_group(
                        make_group(
                            [record],
                            root_short_exception=not record.get("parent_chunk_id")
                            and not is_aux_chunk(record)
                            and original_index_len(record) < MIN_CHUNK_SIZE,
                        )
                    )
                for group in child_groups:
                    add_group(group)

        if not has_children and not record.get("parent_chunk_id") and record_id not in assigned:
            add_group(
                make_group(
                    [record],
                    root_short_exception=not is_aux_chunk(record) and original_index_len(record) < MIN_CHUNK_SIZE,
                )
            )

    for record in records:
        record_id = str(record["chunk_id"])
        if record_id in assigned:
            continue
        add_group(
            make_group(
                [record],
                root_short_exception=not record.get("parent_chunk_id")
                and not is_aux_chunk(record)
                and original_index_len(record) < MIN_CHUNK_SIZE,
            )
        )

    groups.sort(key=lambda group: min(order[str(record["chunk_id"])] for record in group["records"]))
    collapse_single_child_groups(groups)
    groups.sort(key=lambda group: min(order[str(record["chunk_id"])] for record in group["records"]))

    doc_id = str(records[0].get("doc_id") or input_path.stem)
    group_id_by_obj = {id(group): f"{doc_id}_{idx:04d}" for idx, group in enumerate(groups, 1)}

    old_to_new: dict[str, str] = {}
    for group in groups:
        new_id = group_id_by_obj[id(group)]
        for record in group["records"]:
            old_to_new[str(record["chunk_id"])] = new_id

    output: list[dict[str, Any]] = []
    for group in groups:
        group_records = group["records"]
        first = group_records[0]
        new_id = group_id_by_obj[id(group)]
        original_parent_id = group.get("parent_old_id")
        new_parent_id = old_to_new.get(str(original_parent_id)) if original_parent_id else None
        if new_parent_id == new_id:
            new_parent_id = None

        merged: dict[str, Any] = {
            "chunk_id": new_id,
            "doc_id": first.get("doc_id"),
            "source_file": first.get("source_file"),
            "source_lines": source_lines(group_records),
            "chunk_type": classify_chunk_type(group_records),
            "retrieval_tier": "primary"
            if group.get("parent_child_collapse")
            else str(first.get("retrieval_tier") or "primary"),
            "section_path": first.get("section_path") or [],
            "title": first.get("title"),
            "parent_chunk_id": new_parent_id,
            "pic_ids": unique_in_order([pic_id for record in group_records for pic_id in (record.get("pic_ids") or [])]),
            "text": group_text(group_records),
            "source_quality": worst_quality(group_records),
            "source_issue_flags": merged_issue_flags(group_records),
            "source_issue_note": merged_issue_note(group_records),
            "item_no": first.get("item_no"),
            "item_kind": first.get("item_kind"),
        }
        descriptions = pic_description_items(group_records)
        if descriptions:
            merged["pic_descriptions"] = descriptions
        merged["index_text"] = build_index_text(merged, doc_short, pad_context=str(group.get("pad_context") or ""))
        output.append(merged)

    root_short_exception_ids = {
        output[idx]["chunk_id"]
        for idx, group in enumerate(groups)
        if group.get("root_short_exception")
    }
    initial_padded_ids = [
        output[idx]["chunk_id"]
        for idx, group in enumerate(groups)
        if group.get("padded")
    ]
    post_padded_ids = pad_short_output_index_text(output, root_short_exception_ids, doc_short)
    report = build_report(input_path, records, output, groups, old_to_new, unique_in_order(initial_padded_ids + post_padded_ids))
    return output, report


def pad_short_output_index_text(output: list[dict[str, Any]], root_short_exception_ids: set[str], doc_short: str) -> list[str]:
    by_id = {str(record["chunk_id"]): record for record in output}
    padded_ids: list[str] = []
    for idx, record in enumerate(output):
        current = str(record.get("index_text") or "")
        if len(current) >= MIN_CHUNK_SIZE:
            continue
        if str(record["chunk_id"]) in root_short_exception_ids:
            continue
        if is_aux_chunk(record):
            continue

        context_sources: list[str] = []
        parent = by_id.get(str(record.get("parent_chunk_id")))
        if parent and not is_aux_chunk(parent):
            context_sources.append(str(parent.get("index_text") or ""))
        for previous in reversed(output[:idx]):
            if not is_aux_chunk(previous):
                previous_text = str(previous.get("index_text") or "")
                if previous_text not in context_sources:
                    context_sources.append(previous_text)
        for following in output[idx + 1 :]:
            if not is_aux_chunk(following):
                following_text = str(following.get("index_text") or "")
                if following_text not in context_sources:
                    context_sources.append(following_text)
        if not context_sources:
            continue

        updated = current
        for context_source in context_sources:
            needed = MIN_CHUNK_SIZE - len(updated)
            if needed <= 0:
                break
            context = padding_context_from_text(context_source, record, doc_short, needed=needed)
            if context:
                updated = dedupe_lines(f"{context}\n{updated}")
        if updated != current:
            record["index_text"] = updated
            padded_ids.append(str(record["chunk_id"]))
    return padded_ids

def build_report(
    input_path: Path,
    original: list[dict[str, Any]],
    output: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    old_to_new: dict[str, str],
    padded_ids: list[str],
) -> dict[str, Any]:
    original_ids = {str(record["chunk_id"]) for record in original}
    new_ids = {str(record["chunk_id"]) for record in output}
    broken_new_parents = [
        record["chunk_id"]
        for record in output
        if record.get("parent_chunk_id") and str(record["parent_chunk_id"]) not in new_ids
    ]
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
    short_records = [record for record in output if len(str(record.get("index_text") or "")) < MIN_CHUNK_SIZE]
    root_short_exception_ids = [
        output[idx]["chunk_id"]
        for idx, group in enumerate(groups)
        if group.get("root_short_exception") and len(str(output[idx].get("index_text") or "")) < MIN_CHUNK_SIZE
    ]
    auxiliary_short_ids = [
        record["chunk_id"]
        for record in short_records
        if is_aux_chunk(record)
    ]
    auxiliary_padded_ids = [
        chunk_id
        for chunk_id in padded_ids
        if any(str(record["chunk_id"]) == str(chunk_id) and is_aux_chunk(record) for record in output)
    ]
    original_over_max_merged_ids = [
        str(record["chunk_id"])
        for group in groups
        if len(group["records"]) > 1
        for record in group["records"]
        if is_original_over_max(record)
    ]
    id_format_errors = [
        record["chunk_id"]
        for idx, record in enumerate(output, 1)
        if str(record["chunk_id"]) != f"{record.get('doc_id')}_{idx:04d}"
    ]
    parent_child_counts = Counter(str(record.get("parent_chunk_id")) for record in output if record.get("parent_chunk_id"))
    single_child_parent_ids = sorted(parent_id for parent_id, count in parent_child_counts.items() if count == 1)

    return {
        "input_file": input_path.name,
        "original_records": len(original),
        "output_records": len(output),
        "reduction": len(original) - len(output),
        "original_chunk_ids": len(original_ids),
        "new_chunk_ids": len(new_ids),
        "all_original_ids_mapped": len(old_to_new) == len(original_ids),
        "broken_new_parent_refs": broken_new_parents,
        "id_format_errors": id_format_errors,
        "missing_pic_ids": sorted(original_pic_ids - output_pic_ids),
        "extra_pic_ids": sorted(output_pic_ids - original_pic_ids),
        "missing_pic_description_ids": sorted(original_pic_desc_ids - output_pic_desc_ids),
        "extra_pic_description_ids": sorted(output_pic_desc_ids - original_pic_desc_ids),
        "type_counts": dict(Counter(str(record.get("chunk_type") or "") for record in output)),
        "tier_counts": dict(Counter(str(record.get("retrieval_tier") or "") for record in output)),
        "index_length": {
            "lt_200": sum(1 for length in lengths if length < MIN_CHUNK_SIZE),
            "between_200_800": sum(1 for length in lengths if MIN_CHUNK_SIZE <= length <= MAX_CHUNK_SIZE),
            "gt_800": sum(1 for length in lengths if length > MAX_CHUNK_SIZE),
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "padded_chunk_ids": padded_ids,
        "root_short_exception_ids": root_short_exception_ids,
        "auxiliary_short_ids": auxiliary_short_ids,
        "auxiliary_padded_ids": auxiliary_padded_ids,
        "original_auxiliary_records": sum(1 for record in original if is_aux_chunk(record)),
        "output_auxiliary_records": sum(1 for record in output if is_aux_chunk(record)),
        "parent_child_collapse_count": sum(1 for group in groups if group.get("parent_child_collapse")),
        "forced_over_max_count": sum(1 for group in groups if group.get("forced_over_max")),
        "original_over_max_records": sum(1 for record in original if is_original_over_max(record)),
        "original_over_max_merged_ids": original_over_max_merged_ids,
        "disallowed_chunk_types": sorted({str(record.get("chunk_type") or "") for record in output} - ALLOWED_CHUNK_TYPES),
        "index_text_pic_occurrences": sum(str(record.get("index_text") or "").count("<PIC:") for record in output),
        "single_child_parent_ids": single_child_parent_ids,
    }


def integrate_file(input_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    records = load_jsonl(input_path)
    output, report = integrate_records(input_path, records)
    dump_jsonl(output_path, output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def manual_jsonl_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.jsonl") if path.name != "pic_description_tasks.jsonl")


def next_available_output_dir(requested: Path) -> Path:
    if not requested.exists() or not any(requested.iterdir()):
        return requested
    base_name = requested.name
    parent = requested.parent
    suffix = 2
    while True:
        candidate = parent / f"{base_name}{suffix}"
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        suffix += 1


def integrate_command(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    report = integrate_file(input_path, output_path, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def integrate_dir_command(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = next_available_output_dir(Path(args.output_dir))
    report_dir = output_dir / "reports"
    reports = []
    for input_path in manual_jsonl_files(input_dir):
        reports.append(
            integrate_file(
                input_path,
                output_dir / input_path.name,
                report_dir / f"{input_path.stem}__integrate_report.json",
            )
        )
    summary = summarize_reports(reports)
    summary["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "integration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    output_records = sum(int(report.get("output_records") or 0) for report in reports)
    between = sum(int((report.get("index_length") or {}).get("between_200_800") or 0) for report in reports)
    return {
        "files": len(reports),
        "original_records": sum(int(report.get("original_records") or 0) for report in reports),
        "output_records": output_records,
        "reduction": sum(int(report.get("reduction") or 0) for report in reports),
        "index_length": {
            "lt_200": sum(int((report.get("index_length") or {}).get("lt_200") or 0) for report in reports),
            "between_200_800": between,
            "gt_800": sum(int((report.get("index_length") or {}).get("gt_800") or 0) for report in reports),
            "between_200_800_ratio": round(between / output_records, 4) if output_records else 0,
        },
        "padded_chunks": sum(len(report.get("padded_chunk_ids") or []) for report in reports),
        "root_short_exceptions": sum(len(report.get("root_short_exception_ids") or []) for report in reports),
        "auxiliary_short_chunks": sum(len(report.get("auxiliary_short_ids") or []) for report in reports),
        "auxiliary_padded_chunks": sum(len(report.get("auxiliary_padded_ids") or []) for report in reports),
        "original_auxiliary_records": sum(int(report.get("original_auxiliary_records") or 0) for report in reports),
        "output_auxiliary_records": sum(int(report.get("output_auxiliary_records") or 0) for report in reports),
        "parent_child_collapses": sum(int(report.get("parent_child_collapse_count") or 0) for report in reports),
        "forced_over_max": sum(int(report.get("forced_over_max_count") or 0) for report in reports),
        "original_over_max_records": sum(int(report.get("original_over_max_records") or 0) for report in reports),
        "files_with_broken_parent_refs": [report["input_file"] for report in reports if report.get("broken_new_parent_refs")],
        "files_with_id_format_errors": [report["input_file"] for report in reports if report.get("id_format_errors")],
        "files_with_missing_pic_ids": [report["input_file"] for report in reports if report.get("missing_pic_ids")],
        "files_with_missing_pic_descriptions": [report["input_file"] for report in reports if report.get("missing_pic_description_ids")],
        "files_with_disallowed_chunk_types": [report["input_file"] for report in reports if report.get("disallowed_chunk_types")],
        "files_with_index_pic_occurrences": [report["input_file"] for report in reports if report.get("index_text_pic_occurrences")],
        "files_with_original_over_max_merged": [report["input_file"] for report in reports if report.get("original_over_max_merged_ids")],
        "type_counts": dict(sum((Counter(report.get("type_counts") or {}) for report in reports), Counter())),
        "tier_counts": dict(sum((Counter(report.get("tier_counts") or {}) for report in reports), Counter())),
    }


def validate_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    reports = []
    for report_path in sorted((output_dir / "reports").glob("*__integrate_report.json")):
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    summary = summarize_reports(reports)
    padded_by_file = {
        str(report.get("input_file")): set(str(chunk_id) for chunk_id in report.get("padded_chunk_ids") or [])
        for report in reports
    }

    parsed_files = 0
    field_errors: list[str] = []
    for jsonl_path in sorted(output_dir.glob("*.jsonl")):
        parsed_files += 1
        records = load_jsonl(jsonl_path)
        ids = {str(record.get("chunk_id")) for record in records}
        for idx, record in enumerate(records, 1):
            chunk_id = str(record.get("chunk_id"))
            expected_id = f"{record.get('doc_id')}_{idx:04d}"
            if chunk_id != expected_id:
                field_errors.append(f"{jsonl_path.name}:{chunk_id}: expected id {expected_id}")
            if record.get("parent_chunk_id") and str(record["parent_chunk_id"]) not in ids:
                field_errors.append(f"{jsonl_path.name}:{chunk_id}: broken parent")
            if record.get("chunk_type") not in ALLOWED_CHUNK_TYPES:
                field_errors.append(f"{jsonl_path.name}:{chunk_id}: disallowed chunk_type {record.get('chunk_type')}")
            if "<PIC:" in str(record.get("index_text") or ""):
                field_errors.append(f"{jsonl_path.name}:{chunk_id}: index_text contains PIC placeholder")
            if "pic_descriptions" in record and not isinstance(record["pic_descriptions"], list):
                field_errors.append(f"{jsonl_path.name}:{chunk_id}: pic_descriptions is not a list")
            if is_aux_chunk(record) and chunk_id in padded_by_file.get(jsonl_path.name, set()):
                field_errors.append(f"{jsonl_path.name}:{chunk_id}: auxiliary chunk was padded")
    summary["parsed_output_files"] = parsed_files
    summary["field_errors"] = field_errors
    if args.output:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    one = subparsers.add_parser("integrate", help="Integrate one manual chunk JSONL.")
    one.add_argument("--input", required=True)
    one.add_argument("--output", required=True)
    one.add_argument("--report", required=True)
    one.set_defaults(func=integrate_command)

    many = subparsers.add_parser("integrate-dir", help="Integrate all manual chunk JSONL files in a directory.")
    many.add_argument("--input-dir", default="data/manuals/new_chunk")
    many.add_argument("--output-dir", default="data/manuals/new_chunk/chunk_integrate")
    many.set_defaults(func=integrate_dir_command)

    validate = subparsers.add_parser("validate", help="Validate integrated chunk outputs.")
    validate.add_argument("--output-dir", default="data/manuals/new_chunk/chunk_integrate")
    validate.add_argument("--output")
    validate.set_defaults(func=validate_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
