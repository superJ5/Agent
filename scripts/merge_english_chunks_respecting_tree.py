"""Merge the 20 English manual chunk files while preserving the section tree.

Rules implemented here:
- input: data/manuals/chunks English files
- output: data/manuals/new_chunk/chunk_integrate same filenames
- auxiliary chunks are kept, not merged, not used for padding; IDs are remapped
- adjacent primary chunks may merge only when they share the same direct parent
- support chunks are kept as independent parent chunks unless downgraded
- if a non-aux parent has exactly one primary child after child merging and the
  materialized parent index_text stays within EN_MAX, the parent absorbs that
  child and becomes primary
- length thresholds are measured by index_text character count
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

AUX_TYPES = {"metadata_image_path", "aux_navigation"}

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

QUALITY_ORDER = {
    "normal": 0,
    "warning": 1,
    "needs_review": 2,
    "issue": 3,
}

WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANKS_RE = re.compile(r"\n{3,}")
HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?](?=\s|$)")
PIC_RE = re.compile(r"<PIC:([^>]+)>")


@dataclass
class OutNode:
    tmp_id: str
    record: dict[str, Any]
    old_ids: list[str]
    first_order: int
    children: list["OutNode"] = field(default_factory=list)
    parent: "OutNode | None" = None
    merged_siblings: bool = False
    collapsed_parent_child: bool = False
    padded: bool = False
    kept_over_max: bool = False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    return record.get("retrieval_tier") == "auxiliary" or record.get("chunk_type") in AUX_TYPES


def index_len(record: dict[str, Any]) -> int:
    return len(record.get("index_text") or "")


def line_dedupe_text(text: str) -> str:
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


def unique_key(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def unique_in_order(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = unique_key(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def pic_descriptions_for(nodes: list[OutNode]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for node in nodes:
        for item in node.record.get("pic_descriptions") or []:
            if not isinstance(item, dict):
                continue
            pic_id = item.get("pic_id")
            description = item.get("description")
            if not pic_id or pic_id in seen:
                continue
            seen.add(pic_id)
            result.append({"pic_id": str(pic_id), "description": str(description or "")})
    return result


def worst_quality(nodes: list[OutNode]) -> str:
    return max(
        (str(node.record.get("source_quality") or "normal") for node in nodes),
        key=lambda quality: QUALITY_ORDER.get(quality, -1),
        default="normal",
    )


def merged_issue_flags(nodes: list[OutNode]) -> list[Any]:
    values: list[Any] = []
    for node in nodes:
        flags = node.record.get("source_issue_flags")
        if isinstance(flags, list):
            values.extend(flags)
    return unique_in_order(values)


def merged_issue_note(nodes: list[OutNode]) -> str:
    notes = [str(node.record.get("source_issue_note") or "").strip() for node in nodes]
    return " | ".join(unique_in_order([note for note in notes if note]))


def source_lines_for(nodes: list[OutNode]) -> list[int] | Any:
    ranges = [
        node.record.get("source_lines")
        for node in nodes
        if isinstance(node.record.get("source_lines"), list)
        and len(node.record.get("source_lines")) == 2
        and all(isinstance(x, int) for x in node.record.get("source_lines"))
    ]
    if not ranges:
        return nodes[0].record.get("source_lines")
    return [min(item[0] for item in ranges), max(item[1] for item in ranges)]


def common_section_path(nodes: list[OutNode]) -> list[str]:
    paths = [node.record.get("section_path") or [] for node in nodes]
    if not paths:
        return []
    common: list[str] = []
    for values in zip(*paths):
        if all(value == values[0] for value in values):
            common.append(values[0])
        else:
            break
    return common


def title_for_merged(nodes: list[OutNode], section_path: list[str]) -> str:
    if len(nodes) == 1:
        return str(nodes[0].record.get("title") or "")
    if section_path:
        return section_path[-1]
    titles = [str(node.record.get("title") or "").strip() for node in nodes]
    titles = [title for title in titles if title]
    return " / ".join(unique_in_order(titles[:3]))


def can_merge_nodes(nodes: list[OutNode]) -> bool:
    lengths = [index_len(node.record) for node in nodes]
    if any(length > EN_MAX_CHUNK_SIZE for length in lengths):
        return False
    merged_index = merged_index_text(nodes)
    return len(merged_index) <= EN_MAX_CHUNK_SIZE


def merged_index_text(nodes: list[OutNode]) -> str:
    return line_dedupe_text("\n\n".join(str(node.record.get("index_text") or "") for node in nodes)).strip()


def merge_primary_nodes(nodes: list[OutNode], tmp_id: str) -> OutNode:
    if len(nodes) == 1:
        return nodes[0]
    first = nodes[0].record
    section_path = common_section_path(nodes)
    record = dict(first)
    record["chunk_id"] = tmp_id
    record["source_lines"] = source_lines_for(nodes)
    record["retrieval_tier"] = "primary"
    record["section_path"] = section_path
    record["title"] = title_for_merged(nodes, section_path)
    record["text"] = "\n\n".join(str(node.record.get("text") or "").strip() for node in nodes if node.record.get("text"))
    record["index_text"] = merged_index_text(nodes)
    record["pic_ids"] = unique_in_order([pic_id for node in nodes for pic_id in (node.record.get("pic_ids") or [])])
    record["pic_descriptions"] = pic_descriptions_for(nodes)
    record["source_quality"] = worst_quality(nodes)
    record["source_issue_flags"] = merged_issue_flags(nodes)
    record["source_issue_note"] = merged_issue_note(nodes)
    record["item_no"] = first.get("item_no")
    record["item_kind"] = first.get("item_kind")
    return OutNode(
        tmp_id=tmp_id,
        record=record,
        old_ids=[old_id for node in nodes for old_id in node.old_ids],
        first_order=min(node.first_order for node in nodes),
        merged_siblings=True,
    )


def should_add_to_group(group: list[OutNode], candidate: OutNode) -> bool:
    current_len = len(merged_index_text(group))
    combined_len = len(merged_index_text([*group, candidate]))
    if combined_len > EN_MAX_CHUNK_SIZE:
        return False
    if current_len < EN_MIN_CHUNK_SIZE:
        return True
    return combined_len <= EN_TARGET_CHUNK_SIZE


def merge_primary_run(run: list[OutNode], tmp_counter: list[int]) -> list[OutNode]:
    groups: list[list[OutNode]] = []
    index = 0
    while index < len(run):
        node = run[index]
        if index_len(node.record) > EN_MAX_CHUNK_SIZE:
            node.kept_over_max = True
            groups.append([node])
            index += 1
            continue
        group = [node]
        index += 1
        while index < len(run) and should_add_to_group(group, run[index]):
            group.append(run[index])
            index += 1
        groups.append(group)

    # If the final group is still short, attach it to the previous group when
    # doing so respects EN_MAX. This avoids avoidable tail fragments.
    if len(groups) >= 2 and len(merged_index_text(groups[-1])) < EN_MIN_CHUNK_SIZE:
        combined = [*groups[-2], *groups[-1]]
        if can_merge_nodes(combined):
            groups[-2] = combined
            groups.pop()

    merged: list[OutNode] = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
        else:
            tmp_counter[0] += 1
            merged.append(merge_primary_nodes(group, f"__merged_{tmp_counter[0]:06d}"))
    return merged


def merge_adjacent_primary_children(children: list[OutNode], tmp_counter: list[int]) -> list[OutNode]:
    result: list[OutNode] = []
    run: list[OutNode] = []
    for child in children:
        if child.record.get("retrieval_tier") == "primary":
            run.append(child)
            continue
        if run:
            result.extend(merge_primary_run(run, tmp_counter))
            run = []
        result.append(child)
    if run:
        result.extend(merge_primary_run(run, tmp_counter))
    return result


def collect_nodes(root: OutNode) -> list[OutNode]:
    result = [root]
    for child in root.children:
        result.extend(collect_nodes(child))
    return result


def process_non_aux_tree(records: list[dict[str, Any]], stats: Counter[str]) -> list[OutNode]:
    old_by_id = {record["chunk_id"]: record for record in records}
    children_by_parent: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        parent_id = record.get("parent_chunk_id")
        if parent_id not in old_by_id:
            parent_id = None
        children_by_parent[parent_id].append(record)
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda item: item["_order"])

    tmp_counter = [0]

    def make_leaf(record: dict[str, Any]) -> OutNode:
        copy = {key: value for key, value in record.items() if not key.startswith("_")}
        return OutNode(
            tmp_id=record["_old_id"],
            record=copy,
            old_ids=[record["_old_id"]],
            first_order=record["_order"],
        )

    def process(record: dict[str, Any]) -> OutNode:
        node = make_leaf(record)
        processed_children = [process(child) for child in children_by_parent.get(record["_old_id"], [])]
        processed_children = merge_adjacent_primary_children(processed_children, tmp_counter)
        for child in processed_children:
            child.parent = node

        if not processed_children:
            node.record["retrieval_tier"] = "primary"
            stats["downgraded_leaf_support_to_primary"] += int(record.get("retrieval_tier") == "support")
            return node

        if (
            len(processed_children) == 1
            and processed_children[0].record.get("retrieval_tier") == "primary"
            and index_len(node.record) <= EN_MAX_CHUNK_SIZE
        ):
            # The support text is already materialized with the child content.
            node.record["retrieval_tier"] = "primary"
            node.old_ids.extend(processed_children[0].old_ids)
            node.collapsed_parent_child = True
            stats["collapsed_parent_unique_child"] += 1
            return node

        node.record["retrieval_tier"] = "support"
        node.children = processed_children
        for child in node.children:
            child.parent = node
        return node

    roots = [process(record) for record in children_by_parent.get(None, [])]
    return roots


def clean_padding_source(text: str, current: OutNode) -> str:
    section_titles = {str(part).strip() for part in (current.record.get("section_path") or []) if str(part).strip()}
    section_titles.add(str(current.record.get("title") or "").strip())
    result: list[str] = []
    for raw_line in text.splitlines():
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line or line == "...":
            continue
        if HEADING_LINE_RE.match(line):
            continue
        if line in section_titles:
            continue
        result.append(line)
    return BLANKS_RE.sub("\n\n", "\n".join(result)).strip()


def sentence_tail(text: str, max_sentences: int = 2) -> str:
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(text)]
    if sentences:
        return " ".join(sentences[-max_sentences:]).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_sentences:]).strip()


def sentence_head(text: str, max_sentences: int = 2) -> str:
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(text)]
    if sentences:
        return " ".join(sentences[:max_sentences]).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:max_sentences]).strip()


def apply_padding(nodes: list[OutNode], stats: Counter[str]) -> None:
    children_by_parent: dict[str | None, list[OutNode]] = defaultdict(list)
    for node in nodes:
        if node.record.get("retrieval_tier") == "auxiliary":
            continue
        parent_tmp = node.parent.tmp_id if node.parent else None
        children_by_parent[parent_tmp].append(node)
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda node: node.first_order)
        for index, node in enumerate(siblings):
            if node.record.get("retrieval_tier") != "primary":
                continue
            if index_len(node.record) >= EN_MIN_CHUNK_SIZE:
                continue
            previous_primary = next(
                (
                    sibling
                    for sibling in reversed(siblings[:index])
                    if sibling.record.get("retrieval_tier") == "primary"
                ),
                None,
            )
            next_primary = next(
                (
                    sibling
                    for sibling in siblings[index + 1 :]
                    if sibling.record.get("retrieval_tier") == "primary"
                ),
                None,
            )
            original = node.record.get("index_text") or ""
            padded = original
            if previous_primary is not None:
                context = sentence_tail(clean_padding_source(previous_primary.record.get("index_text") or "", node))
                if context:
                    padded = f"{context}\n\n{padded}"
            elif next_primary is not None:
                context = sentence_head(clean_padding_source(next_primary.record.get("index_text") or "", node))
                if context:
                    padded = f"{padded}\n\n{context}"
            padded = line_dedupe_text(padded)
            if padded != original:
                node.record["index_text"] = padded
                node.padded = True
                stats["padded_primary_chunks"] += 1


def flatten_nodes(roots: list[OutNode]) -> list[OutNode]:
    flattened: list[OutNode] = []
    for root in sorted(roots, key=lambda node: node.first_order):
        flattened.extend(collect_nodes(root))
    return flattened


def create_aux_nodes(records: list[dict[str, Any]]) -> list[OutNode]:
    nodes: list[OutNode] = []
    for record in records:
        clean = {key: value for key, value in record.items() if not key.startswith("_")}
        nodes.append(
            OutNode(
                tmp_id=record["_old_id"],
                record=clean,
                old_ids=[record["_old_id"]],
                first_order=record["_order"],
            )
        )
    return nodes


def finalize_records(nodes: list[OutNode]) -> list[dict[str, Any]]:
    nodes = sorted(nodes, key=lambda node: (node.first_order, 0 if node.record.get("retrieval_tier") == "support" else 1))
    old_to_node: dict[str, OutNode] = {}
    for node in nodes:
        for old_id in node.old_ids:
            old_to_node[old_id] = node

    tmp_to_new: dict[str, str] = {}
    for index, node in enumerate(nodes, 1):
        doc_id = node.record["doc_id"]
        new_id = f"{doc_id}_{index:04d}"
        tmp_to_new[node.tmp_id] = new_id
        node.record["chunk_id"] = new_id

    for node in nodes:
        if node.record.get("retrieval_tier") == "auxiliary" or is_aux_chunk(node.record):
            old_parent = node.record.get("parent_chunk_id")
            if old_parent is None:
                node.record["parent_chunk_id"] = None
            elif old_parent in old_to_node:
                node.record["parent_chunk_id"] = tmp_to_new[old_to_node[old_parent].tmp_id]
            else:
                node.record["parent_chunk_id"] = None
            continue
        if node.parent is not None:
            node.record["parent_chunk_id"] = tmp_to_new[node.parent.tmp_id]
        else:
            node.record["parent_chunk_id"] = None

    return [node.record for node in nodes]


def validate(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {record["chunk_id"] for record in records}
    non_aux = [record for record in records if not is_aux_chunk(record)]
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dangling = []
    for record in records:
        parent_id = record.get("parent_chunk_id")
        if parent_id is None:
            continue
        if parent_id not in ids:
            dangling.append((record["chunk_id"], parent_id))
        elif not is_aux_chunk(record):
            children[parent_id].append(record)

    primary_with_children = [
        record["chunk_id"]
        for record in non_aux
        if record.get("retrieval_tier") == "primary" and children.get(record["chunk_id"])
    ]
    support_without_children = [
        record["chunk_id"]
        for record in non_aux
        if record.get("retrieval_tier") == "support" and not children.get(record["chunk_id"])
    ]
    oversized_merged_primary = [
        record["chunk_id"]
        for record in non_aux
        if record.get("retrieval_tier") == "primary"
        and len(record.get("index_text") or "") > EN_MAX_CHUNK_SIZE
        and bool(record.get("_merged_marker"))
    ]
    return {
        "dangling_parent_count": len(dangling),
        "primary_with_children_count": len(primary_with_children),
        "support_without_children_count": len(support_without_children),
        "oversized_merged_primary_count": len(oversized_merged_primary),
        "dangling_parent_sample": dangling[:5],
        "primary_with_children_sample": primary_with_children[:5],
        "support_without_children_sample": support_without_children[:5],
        "oversized_merged_primary_sample": oversized_merged_primary[:5],
    }


def merge_file(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for order, record in enumerate(records):
        item = dict(record)
        item["_old_id"] = record["chunk_id"]
        item["_order"] = order
        prepared.append(item)

    aux_records = [record for record in prepared if is_aux_chunk(record)]
    non_aux_records = [record for record in prepared if not is_aux_chunk(record)]
    stats: Counter[str] = Counter()
    roots = process_non_aux_tree(non_aux_records, stats)
    non_aux_nodes = flatten_nodes(roots)
    apply_padding(non_aux_nodes, stats)
    aux_nodes = create_aux_nodes(aux_records)
    all_nodes = [*non_aux_nodes, *aux_nodes]

    for node in all_nodes:
        if node.merged_siblings:
            node.record["_merged_marker"] = True
            stats["merged_sibling_chunks"] += 1
        if node.collapsed_parent_child:
            stats["collapsed_output_chunks"] += 1
        if node.kept_over_max:
            stats["kept_existing_over_max_primary"] += 1

    output = finalize_records(all_nodes)
    for record in output:
        record.pop("_merged_marker", None)

    length_counts = Counter()
    for record in output:
        if is_aux_chunk(record):
            continue
        length = len(record.get("index_text") or "")
        if length < EN_MIN_CHUNK_SIZE:
            length_counts["below_min"] += 1
        elif length <= EN_MAX_CHUNK_SIZE:
            length_counts["within_range"] += 1
        else:
            length_counts["over_max"] += 1

    report = {
        "input_records": len(records),
        "output_records": len(output),
        "auxiliary_records": len(aux_records),
        "non_aux_input_records": len(non_aux_records),
        "non_aux_output_records": len([record for record in output if not is_aux_chunk(record)]),
        "stats": dict(stats),
        "length_counts": dict(length_counts),
        "validation": validate(output),
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/manuals/chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manuals/new_chunk/chunk_integrate"))
    args = parser.parse_args()

    reports: dict[str, Any] = {}
    totals = Counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in ENGLISH_FILES:
        records = load_jsonl(args.input_dir / filename)
        output, report = merge_file(records)
        validation = report["validation"]
        if any(
            validation[key]
            for key in [
                "dangling_parent_count",
                "primary_with_children_count",
                "support_without_children_count",
                "oversized_merged_primary_count",
            ]
        ):
            raise ValueError(f"{filename}: validation failed: {validation}")
        dump_jsonl(args.output_dir / filename, output)
        reports[filename] = report
        totals["files"] += 1
        totals["input_records"] += report["input_records"]
        totals["output_records"] += report["output_records"]
        totals["auxiliary_records"] += report["auxiliary_records"]
        totals["non_aux_input_records"] += report["non_aux_input_records"]
        totals["non_aux_output_records"] += report["non_aux_output_records"]
        for key, value in report["stats"].items():
            totals[key] += value
        for key, value in report["length_counts"].items():
            totals[f"length_{key}"] += value
        print(
            f"{filename}: {report['input_records']} -> {report['output_records']}, "
            f"non_aux {report['non_aux_input_records']} -> {report['non_aux_output_records']}, "
            f"stats={report['stats']}, lengths={report['length_counts']}"
        )

    report_path = args.output_dir / "_merge_report.json"
    report_path.write_text(
        json.dumps({"totals": dict(totals), "files": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"TOTAL: {dict(totals)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
