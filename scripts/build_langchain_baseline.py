"""Build LangChain-style baseline chunks for A/B comparison.

Uses the same MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
pipeline from document_splitter_service.py, applied to pre-cut .md source
files.  Outputs a chunks.jsonl in the same schema as build_baseline_chunks.py
so the results can be indexed and compared directly.

Usage:
    # Standalone: just chunk + stats (no Milvus, no main app dependency)
    python scripts/build_langchain_baseline.py

    # With Milvus indexing: chunk + stats + push to vector DB
    python scripts/build_langchain_baseline.py --index
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIC_RE = re.compile(r"<PIC:([^>]+)>")

# 源 .md 文件目录：请将审阅后的 .md 文件放入此目录
DEFAULT_SOURCE_MD_DIR = Path("./data/manuals/reviewed_md")
DEFAULT_STRUCTURED_CHUNKS_DIR = Path("./data/manuals/chunks")
DEFAULT_OUTPUT_DIR = Path("./data/manuals/langchain_baseline")

# Chunk parameters matching document_splitter_service.py defaults
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MERGE_MIN_SIZE = 300


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManualSpec:
    source_file: Path
    structured_chunks_file: Path | None
    doc_id: str
    doc_name: str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build LangChain-style baseline chunks from a reviewed .md file. "
            "Uses MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter."
        ),
    )
    parser.add_argument(
        "--source-md-dir",
        default=str(DEFAULT_SOURCE_MD_DIR),
        help=(
            "Directory containing reviewed .md files. "
            "Each .md file will be processed as a separate manual."
        ),
    )
    parser.add_argument(
        "--source-md",
        default=None,
        help="Process a single .md file instead of scanning the directory.",
    )
    parser.add_argument(
        "--structured-chunks-dir",
        default=str(DEFAULT_STRUCTURED_CHUNKS_DIR),
        help="Directory with structured .jsonl files (for image lookup and comparison).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where output chunks.jsonl will be written.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Target chunk size for RecursiveCharacterTextSplitter.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Overlap between chunks.",
    )
    parser.add_argument(
        "--merge-min",
        type=int,
        default=MERGE_MIN_SIZE,
        help="Minimum chunk size; smaller chunks get merged with neighbors.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        default=False,
        help=(
            "After generating chunks, index them into Milvus via the main app's "
            "vector_index_service. Requires the main project environment (app.services). "
            "Without this flag the script runs fully standalone."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# LangChain splitting pipeline (mirrors document_splitter_service.py)
# ---------------------------------------------------------------------------

def langchain_split_markdown(
    content: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    merge_min: int = MERGE_MIN_SIZE,
) -> list[dict[str, Any]]:
    """Three-stage LangChain split: header → recursive char → merge small."""

    # Stage 1: Split by Markdown headers (# and ##)
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False,
    )
    header_docs = md_splitter.split_text(content)

    # Stage 2: Recursive character split on each header section
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(header_docs)

    # Stage 3: Merge small chunks (< merge_min chars)
    merged_docs = merge_small_chunks(split_docs, merge_min, chunk_size)

    # Convert to plain dicts
    results: list[dict[str, Any]] = []
    for doc in merged_docs:
        section_path = []
        for key in ("h1", "h2", "h3", "h4"):
            val = doc.metadata.get(key)
            if val:
                section_path.append(val)
        results.append({
            "text": doc.page_content,
            "section_path": section_path,
        })
    return results


def merge_small_chunks(documents: list, min_size: int, max_size: int) -> list:
    """Merge adjacent chunks smaller than min_size (same as document_splitter_service)."""
    if not documents:
        return []

    merged = []
    current = None

    for doc in documents:
        if current is None:
            current = doc
        elif len(doc.page_content) < min_size and len(current.page_content) < max_size * 2:
            current.page_content += "\n\n" + doc.page_content
        else:
            merged.append(current)
            current = doc

    if current is not None:
        merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# Image lookup (reuse from structured chunks)
# ---------------------------------------------------------------------------

def build_image_lookup(structured_chunks_file: Path | None) -> dict[str, str]:
    """Extract pic_id → absolute_path mapping from structured chunks."""
    if not structured_chunks_file or not structured_chunks_file.exists():
        return {}

    lookup: dict[str, str] = {}
    for line in structured_chunks_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("chunk_type") != "metadata_image_path":
            continue
        pic_ids = record.get("pic_ids", [])
        text = record.get("text", "")
        # Extract absolute path from text
        match = re.search(r"绝对路径[：:]?\s*`([^`]+)`", text)
        if match and pic_ids:
            for pid in pic_ids:
                lookup[pid] = match.group(1)
    return lookup


def extract_pic_ids(text: str) -> list[str]:
    return list(dict.fromkeys(m.strip() for m in PIC_RE.findall(text) if m.strip()))


def guess_section_title(section_path: list[str], text: str, index: int) -> str:
    """Derive a title from section_path or first heading in text."""
    if section_path:
        return section_path[-1]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return f"langchain_chunk_{index:04d}"


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def build_rows(
    spec: ManualSpec,
    chunks: list[dict[str, Any]],
    image_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert LangChain split results into baseline chunk rows."""
    rows: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        text = chunk["text"].strip()
        section_path = chunk["section_path"]
        pic_ids = extract_pic_ids(text)
        title = guess_section_title(section_path, text, index)

        # Build index_text: section path + cleaned text
        cleaned = PIC_RE.sub("", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        index_text = f"{' > '.join(section_path)}\n{cleaned}" if section_path else cleaned

        rows.append({
            "chunk_id": f"{spec.doc_id}_{index:04d}",
            "doc_id": spec.doc_id,
            "doc_name": spec.doc_name,
            "source_file": str(spec.source_file),
            "source_lines": [],
            "chunk_type": "langchain_baseline_chunk",
            "retrieval_tier": "primary",
            "section_path": ["langchain_md_header_split"] + section_path,
            "title": title,
            "parent_chunk_id": None,
            "pic_ids": pic_ids,
            "text": text,
            "index_text": index_text,
            "source_quality": "langchain_baseline",
            "source_issue_flags": [],
            "source_issue_note": "LangChain MarkdownHeader + RecursiveChar split baseline",
            "item_no": None,
            "item_kind": "langchain_baseline",
        })

    # Add image metadata rows
    for pid, path in sorted(image_lookup.items()):
        rows.append({
            "chunk_id": f"{spec.doc_id}_meta_{pid}",
            "doc_id": spec.doc_id,
            "doc_name": spec.doc_name,
            "source_file": str(spec.source_file),
            "source_lines": [],
            "chunk_type": "metadata_image_path",
            "retrieval_tier": "auxiliary",
            "section_path": ["langchain_image_metadata", pid],
            "title": f"<PIC:{pid}>",
            "parent_chunk_id": None,
            "pic_ids": [pid],
            "text": f"- pic_id: {pid}\n- absolute_image_path: `{path}`",
            "index_text": f"{pid} image path {path}",
            "source_quality": "langchain_baseline",
            "source_issue_flags": [],
            "source_issue_note": "",
            "item_no": pid,
            "item_kind": "image_path",
        })

    return rows


# ---------------------------------------------------------------------------
# Stats & Report
# ---------------------------------------------------------------------------

def print_stats(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Print and return chunk statistics."""
    text_rows = [r for r in rows if r["chunk_type"] != "metadata_image_path"]
    lengths = [len(r["text"]) for r in text_rows]

    if not lengths:
        print(f"  [{label}] No text chunks.")
        return {}

    stats = {
        "label": label,
        "total_chunks": len(text_rows),
        "image_meta_chunks": len(rows) - len(text_rows),
        "total_chars": sum(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "mean_len": round(statistics.mean(lengths), 1),
        "median_len": round(statistics.median(lengths), 1),
        "stdev_len": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
        "chunks_with_pics": sum(1 for r in text_rows if r["pic_ids"]),
    }

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for key, val in stats.items():
        if key == "label":
            continue
        print(f"  {key:>20s}: {val}")
    print(f"{'='*60}")

    return stats


def compare_with_structured(
    langchain_rows: list[dict[str, Any]],
    structured_path: Path | None,
) -> None:
    """Load structured chunks and print side-by-side comparison."""
    if not structured_path or not structured_path.exists():
        print("\n  [SKIP] No structured chunks file for comparison.")
        return

    structured_rows = []
    for line in structured_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            structured_rows.append(json.loads(line))

    s_stats = print_stats(structured_rows, "Structured (chunks.jsonl)")
    l_stats = print_stats(langchain_rows, "LangChain Baseline")

    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    for key in ("total_chunks", "min_len", "max_len", "mean_len", "median_len", "stdev_len"):
        s_val = s_stats.get(key, "N/A")
        l_val = l_stats.get(key, "N/A")
        print(f"  {key:>20s}:  structured={s_val:<10}  langchain={l_val}")

    # Show chunk length distribution buckets
    for label, rows in [("Structured", structured_rows), ("LangChain", langchain_rows)]:
        text_rows = [r for r in rows if r.get("chunk_type") != "metadata_image_path"]
        buckets = Counter()
        for r in text_rows:
            length = len(r["text"])
            if length < 50:
                buckets["<50"] += 1
            elif length < 100:
                buckets["50-99"] += 1
            elif length < 200:
                buckets["100-199"] += 1
            elif length < 500:
                buckets["200-499"] += 1
            elif length < 1000:
                buckets["500-999"] += 1
            else:
                buckets["1000+"] += 1
        print(f"\n  {label} length distribution:")
        for bucket in ["<50", "50-99", "100-199", "200-499", "500-999", "1000+"]:
            count = buckets.get(bucket, 0)
            bar = "█" * count
            print(f"    {bucket:>10s}: {count:>3d} {bar}")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_source_files(args) -> list[Path]:
    """Find .md source files to process."""
    if args.source_md:
        single = Path(args.source_md).resolve()
        if not single.exists():
            raise FileNotFoundError(f"Source file not found: {single}")
        return [single]

    source_dir = Path(args.source_md_dir).resolve()
    if not source_dir.exists():
        source_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created source directory: {source_dir}")
        print("Please place your reviewed .md files there and re-run.")
        return []

    files = sorted(source_dir.glob("*.md"))
    if not files:
        print(f"No .md files found in {source_dir}")
        print("Please place your reviewed .md files there and re-run.")
    return files


def infer_doc_id_from_filename(filename: str) -> str:
    """Derive a doc_id from the .md filename."""
    stem = Path(filename).stem
    # Strip common suffixes
    for suffix in (".reviewed_reconstructed", "_heading_system_reconstructed", ".reviewed", "_reconstructed_text"):
        stem = stem.removesuffix(suffix)
    return f"{stem}_langchain"


def find_matching_structured_chunks(structured_dir: Path, source_name: str) -> Path | None:
    """Try to find a structured .jsonl file matching the source .md name."""
    if not structured_dir.exists():
        return None
    stem = Path(source_name).stem
    for suffix in (".reviewed_reconstructed", "_heading_system_reconstructed", ".reviewed", "_reconstructed_text"):
        stem = stem.removesuffix(suffix)
    # Search for .jsonl files containing the manual name
    # Prefer files without variant suffixes (e.g., __origin, __restrict)
    candidates = [j for j in sorted(structured_dir.glob("*.jsonl")) if stem in j.name]
    if not candidates:
        return None
    # Prefer a file whose name ends cleanly (no __origin, __restrict, etc.)
    for c in candidates:
        base = c.stem.split("__")[-1] if "__" in c.stem else c.stem
        if base == f"{stem}手册" or base == stem:
            return c
    return candidates[0]


def main() -> None:
    args = build_parser().parse_args()

    source_files = discover_source_files(args)
    if not source_files:
        return

    structured_dir = Path(args.structured_chunks_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_reports: list[dict] = []

    for source_md in source_files:
        print(f"\n{'='*60}")
        print(f"  Processing: {source_md.name}")
        print(f"{'='*60}")

        doc_id = infer_doc_id_from_filename(source_md.name)
        doc_name = source_md.stem
        for suffix in (".reviewed_reconstructed", "_heading_system_reconstructed", ".reviewed", "_reconstructed_text"):
            doc_name = doc_name.removesuffix(suffix)
        doc_name = f"{doc_name} langchain baseline"

        spec = ManualSpec(
            source_file=source_md,
            structured_chunks_file=find_matching_structured_chunks(structured_dir, source_md.name),
            doc_id=doc_id,
            doc_name=doc_name,
        )

        content = source_md.read_text(encoding="utf-8")
        print(f"Source size: {len(content)} chars, {content.count(chr(10))+1} lines")

        # Run LangChain splitting
        print(f"LangChain split (chunk_size={args.chunk_size}, overlap={args.chunk_overlap}, merge_min={args.merge_min})...")
        chunks = langchain_split_markdown(content, args.chunk_size, args.chunk_overlap, args.merge_min)
        print(f"LangChain produced {len(chunks)} text chunks.")

        # Build image lookup from matching structured chunks
        image_lookup = build_image_lookup(spec.structured_chunks_file)
        print(f"Image lookup: {len(image_lookup)} pic_ids found.")

        # Build rows
        rows = build_rows(spec, chunks, image_lookup)

        # Write per-manual output
        manual_output_dir = output_dir / doc_id
        manual_output_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = manual_output_dir / "chunks.jsonl"
        chunks_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} rows to {chunks_path}")

        # Stats & comparison
        if spec.structured_chunks_file:
            compare_with_structured(rows, spec.structured_chunks_file)
        else:
            print_stats(rows, f"{doc_name} (no structured chunks for comparison)")

        all_reports.append({
            "source_md": str(source_md),
            "doc_id": doc_id,
            "doc_name": doc_name,
            "matched_structured_chunks": str(spec.structured_chunks_file) if spec.structured_chunks_file else None,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "merge_min": args.merge_min,
            "total_rows": len(rows),
            "text_chunks": len([r for r in rows if r["chunk_type"] != "metadata_image_path"]),
            "image_meta_chunks": len([r for r in rows if r["chunk_type"] == "metadata_image_path"]),
            "output_path": str(chunks_path),
        })

    # Write summary report
    report_path = output_dir / "langchain_baseline_report.json"
    report_path.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary report: {report_path}")
    print(f"Processed {len(all_reports)} manual(s).")

    # Optional: index to Milvus
    if args.index:
        index_to_milvus(output_dir)


def index_to_milvus(output_dir: Path) -> None:
    """Push generated baseline chunks into Milvus for end-to-end retrieval testing.

    This function lazy-imports the main project's vector_index_service so the
    script can remain standalone when --index is not used.
    """
    print(f"\n{'='*60}")
    print("  Indexing to Milvus...")
    print(f"{'='*60}")
    try:
        from app.services.vector_index_service import vector_index_service
    except ImportError as exc:
        print(f"  ERROR: Cannot import vector_index_service: {exc}")
        print("  Make sure you are running from the project root with the main app installed.")
        print("  Skipping indexing.")
        return

    result = vector_index_service.index_manual_chunks(str(output_dir))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print("Milvus indexing complete.")


if __name__ == "__main__":
    main()
