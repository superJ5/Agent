"""Build simple character-based baseline chunks with picture references.

This is intentionally much simpler than the reviewed structured chunk pipeline.
It exists as a retrieval baseline: rough text chunks plus any <PIC:...> ids
that appear inside the same chunk.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REVIEWED_ROOT = Path(
    r"E:\.codex\worktrees\d883\ai_agent_competition\results\manual_batch\05_final_reviewed\manual_7f829388"
)
DEFAULT_CHUNKS_ROOT = DEFAULT_REVIEWED_ROOT / "chunk_outputs"
DEFAULT_OUTPUT_ROOT = Path("./data/manuals/baseline_chunks")
PIC_RE = re.compile(r"<PIC:([^>]+)>")
ABSOLUTE_IMAGE_PATH_RE = re.compile(r"(?:缁濆璺緞|绝对路径)[：:]?\s*`([^`]+)`")
REFERENCE_MD_PATH_RE = re.compile(r"`([^`]+\.md)`")


@dataclass(frozen=True)
class ManualSpec:
    source_file: Path
    structured_chunks_file: Path
    doc_id: str
    doc_name: str
    output_dir_name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build simple baseline chunks from reviewed structured chunks.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where baseline chunk outputs will be written.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Approximate character length per baseline chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Approximate trailing character overlap between chunks.",
    )
    parser.add_argument(
        "--chunks-file",
        action="append",
        default=[],
        help="Path to a reviewed structured chunks.jsonl file. Repeatable.",
    )
    parser.add_argument(
        "--chunks-root",
        default=str(DEFAULT_CHUNKS_ROOT),
        help=(
            "Directory to recursively discover reviewed structured chunks.jsonl files "
            "when --chunks-file is not provided."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manual_specs = discover_manual_specs(
        chunks_files=[Path(path).resolve() for path in args.chunks_file],
        chunks_root=Path(args.chunks_root).resolve() if args.chunks_root else None,
    )

    report: dict[str, Any] = {
        "output_root": str(output_root),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "manuals": [],
    }

    for spec in manual_specs:
        rows = build_manual_rows(spec, args.chunk_size, args.overlap)
        manual_dir = output_root / spec.output_dir_name
        manual_dir.mkdir(parents=True, exist_ok=True)

        chunks_path = manual_dir / "chunks.jsonl"
        inventory_path = manual_dir / "chunk_inventory.md"
        chunks_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        inventory_path.write_text(build_inventory(spec, rows), encoding="utf-8")

        report["manuals"].append(
            {
                "doc_id": spec.doc_id,
                "doc_name": spec.doc_name,
                "source_file": str(spec.source_file),
                "structured_chunks_file": str(spec.structured_chunks_file),
                "chunks_file": str(chunks_path),
                "row_count": len(rows),
                "primary_count": sum(row["retrieval_tier"] == "primary" for row in rows),
                "auxiliary_count": sum(row["retrieval_tier"] == "auxiliary" for row in rows),
            }
        )

    (output_root / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def discover_manual_specs(chunks_files: list[Path], chunks_root: Path | None) -> list[ManualSpec]:
    discovered_files: list[Path]
    if chunks_files:
        discovered_files = sorted(dict.fromkeys(path.resolve() for path in chunks_files))
    elif chunks_root and chunks_root.exists():
        discovered_files = sorted(chunks_root.rglob("chunks.jsonl"))
    else:
        discovered_files = []

    if not discovered_files:
        raise FileNotFoundError(
            "No reviewed structured chunks.jsonl files found. "
            "Pass --chunks-file or point --chunks-root to a reviewed chunk_outputs directory."
        )

    return [build_manual_spec(chunks_file) for chunks_file in discovered_files]


def build_manual_spec(chunks_file: Path) -> ManualSpec:
    rows = load_jsonl(chunks_file)
    if not rows:
        raise ValueError(f"Structured chunks file is empty: {chunks_file}")

    first = rows[0]
    source_value = first.get("source_file")
    if not isinstance(source_value, str) or not source_value.strip():
        raise ValueError(f"Missing source_file in structured chunks: {chunks_file}")

    source_file = Path(source_value)
    if not source_file.exists():
        raise FileNotFoundError(f"Source markdown not found for baseline build: {source_file}")

    doc_id = infer_baseline_doc_id(first, chunks_file)
    base_doc_name = infer_doc_name(rows, chunks_file, source_file)
    doc_name = f"{base_doc_name} baseline"
    output_dir_name = build_output_dir_name(doc_id, base_doc_name)

    return ManualSpec(
        source_file=source_file,
        structured_chunks_file=chunks_file,
        doc_id=doc_id,
        doc_name=doc_name,
        output_dir_name=output_dir_name,
    )


def infer_baseline_doc_id(first_row: dict[str, Any], chunks_file: Path) -> str:
    doc_id_value = first_row.get("doc_id")
    if isinstance(doc_id_value, str) and doc_id_value.strip():
        doc_id = doc_id_value.strip()
    else:
        doc_id = chunks_file.parent.name.split("__", 1)[0]
    return doc_id if doc_id.endswith("_baseline") else f"{doc_id}_baseline"


def infer_doc_name(
    rows: Iterable[dict[str, Any]],
    chunks_file: Path,
    source_file: Path,
) -> str:
    for row in rows:
        value = row.get("doc_name")
        if isinstance(value, str) and value.strip():
            return value.strip().removesuffix(" baseline")

    parent_name = chunks_file.parent.name
    if "__" in parent_name:
        return parent_name.split("__", 1)[1]

    stem = source_file.stem
    stem = stem.removesuffix(".reviewed_reconstructed")
    stem = stem.removesuffix("_heading_system_reconstructed")
    return stem


def build_output_dir_name(doc_id: str, base_doc_name: str) -> str:
    baseline_doc_id = doc_id if doc_id.endswith("_baseline") else f"{doc_id}_baseline"
    return f"{baseline_doc_id}__{base_doc_name}"


def build_manual_rows(spec: ManualSpec, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    source_text = spec.source_file.read_text(encoding="utf-8")
    image_lookup = build_image_lookup(spec.structured_chunks_file)
    chunks = split_text_by_lines(source_text, chunk_size, overlap)

    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        pic_ids = extract_pic_ids(chunk["text"])
        title = guess_title(chunk["text"], spec.doc_name, index)
        rows.append(
            {
                "chunk_id": f"{spec.doc_id}_{index:04d}",
                "doc_id": spec.doc_id,
                "doc_name": spec.doc_name,
                "source_file": str(spec.source_file),
                "source_lines": [chunk["start_line"], chunk["end_line"]],
                "chunk_type": "baseline_simple_chunk",
                "retrieval_tier": "primary",
                "section_path": ["baseline_simple_char_split", title],
                "title": title,
                "parent_chunk_id": None,
                "pic_ids": pic_ids,
                "text": chunk["text"].strip(),
                "index_text": build_index_text(title, chunk["text"]),
                "source_quality": "baseline",
                "source_issue_flags": [],
                "source_issue_note": "baseline: simple character/line chunking with local PIC extraction",
                "item_no": None,
                "item_kind": "baseline_chunk",
            }
        )

    rows.extend(build_image_metadata_rows(spec, image_lookup))
    return rows


def build_image_lookup(chunks_file: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    referenced_markdown_files: list[Path] = []
    for record in load_jsonl(chunks_file):
        if record.get("chunk_type") != "metadata_image_path":
            continue
        pic_ids = record.get("pic_ids")
        image_path = extract_absolute_image_path(record)
        if isinstance(pic_ids, list) and pic_ids and image_path:
            for pic_id in pic_ids:
                if isinstance(pic_id, str) and pic_id.strip():
                    lookup[pic_id.strip()] = image_path
            continue

        referenced_markdown = extract_reference_markdown_path(record)
        if referenced_markdown and referenced_markdown.exists():
            referenced_markdown_files.append(referenced_markdown)

    for referenced_markdown in referenced_markdown_files:
        lookup.update(parse_markdown_image_lookup(referenced_markdown))
    return lookup


def build_image_metadata_rows(spec: ManualSpec, image_lookup: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pic_id, image_path in sorted(image_lookup.items()):
        text = f"- pic_id: {pic_id}\n- absolute_image_path: `{image_path}`"
        rows.append(
            {
                "chunk_id": f"{spec.doc_id}_meta_{pic_id}",
                "doc_id": spec.doc_id,
                "doc_name": spec.doc_name,
                "source_file": str(spec.source_file),
                "source_lines": [],
                "chunk_type": "metadata_image_path",
                "retrieval_tier": "auxiliary",
                "section_path": ["baseline_image_path_metadata", pic_id],
                "title": f"<PIC:{pic_id}>",
                "parent_chunk_id": None,
                "pic_ids": [pic_id],
                "text": text,
                "index_text": f"{pic_id} image path {image_path}",
                "source_quality": "baseline",
                "source_issue_flags": [],
                "source_issue_note": "",
                "item_no": pic_id,
                "item_kind": "image_path",
            }
        )
    return rows


def split_text_by_lines(text: str, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_start = 1

    for line_no, line in enumerate(lines, start=1):
        if not current_lines:
            current_start = line_no
        current_lines.append(line)

        if len("\n".join(current_lines)) >= chunk_size:
            chunks.append(
                {
                    "text": "\n".join(current_lines),
                    "start_line": current_start,
                    "end_line": line_no,
                }
            )
            current_lines, current_start = carry_overlap_lines(current_lines, overlap, line_no)

    if current_lines:
        chunks.append(
            {
                "text": "\n".join(current_lines),
                "start_line": current_start,
                "end_line": len(lines),
            }
        )

    return chunks


def carry_overlap_lines(lines: list[str], overlap: int, end_line_no: int) -> tuple[list[str], int]:
    if overlap <= 0:
        return [], end_line_no + 1

    carried: list[str] = []
    char_count = 0
    for line in reversed(lines):
        carried.insert(0, line)
        char_count += len(line) + 1
        if char_count >= overlap:
            break

    start_line = max(1, end_line_no - len(carried) + 1)
    return carried, start_line


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def extract_absolute_image_path(record: dict[str, Any]) -> str | None:
    for field in ("text", "index_text"):
        value = record.get(field)
        if not isinstance(value, str):
            continue
        match = ABSOLUTE_IMAGE_PATH_RE.search(value)
        if match:
            return match.group(1).strip()
    return None


def extract_reference_markdown_path(record: dict[str, Any]) -> Path | None:
    for field in ("text", "index_text"):
        value = record.get(field)
        if not isinstance(value, str):
            continue
        match = REFERENCE_MD_PATH_RE.search(value)
        if match:
            return Path(match.group(1).strip())
    return None


def parse_markdown_image_lookup(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    lookup: dict[str, str] = {}
    current_pic_id: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("### <PIC:"):
            pic_ids = extract_pic_ids(line)
            current_pic_id = pic_ids[0] if pic_ids else None
            continue

        if not current_pic_id:
            continue

        match = ABSOLUTE_IMAGE_PATH_RE.search(line)
        if match:
            absolute_path = match.group(1).strip()
            if absolute_path:
                lookup[current_pic_id] = absolute_path
            current_pic_id = None
    return lookup


def extract_pic_ids(text: str) -> list[str]:
    return list(dict.fromkeys(match.strip() for match in PIC_RE.findall(text) if match.strip()))


def guess_title(text: str, doc_name: str, index: int) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or f"{doc_name} baseline chunk {index:04d}"
    return f"{doc_name} baseline chunk {index:04d}"


def build_index_text(title: str, text: str) -> str:
    cleaned = PIC_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return f"{title}\n{cleaned}"


def build_inventory(spec: ManualSpec, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {spec.doc_name} baseline chunk inventory",
        "",
        f"- doc_id: `{spec.doc_id}`",
        f"- source_file: `{spec.source_file}`",
        f"- structured_chunks_file: `{spec.structured_chunks_file}`",
        f"- total_rows: `{len(rows)}`",
        "",
        "| chunk_id | tier | type | title | pic_ids | source_lines |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {chunk_id} | {tier} | {typ} | {title} | {pics} | {lines_} |".format(
                chunk_id=row["chunk_id"],
                tier=row["retrieval_tier"],
                typ=row["chunk_type"],
                title=str(row["title"]).replace("|", "\\|"),
                pics=", ".join(row["pic_ids"]),
                lines_=row["source_lines"],
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
