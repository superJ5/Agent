"""Chunk Markdown AIOps runbooks by section and index them into Milvus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.aiops_chunking import chunk_markdown_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="aiops-docs")
    parser.add_argument("--output", default="data/aiops/chunks")
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--chunk-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    files = sorted(source_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"No Markdown files found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    for source_path in files:
        records = chunk_markdown_document(
            source_path.read_text(encoding="utf-8"),
            source_path.as_posix(),
            max_chars=args.max_chars,
        )
        output_path = output_dir / f"{source_path.stem}.jsonl"
        output_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        print(f"chunked {source_path}: {len(records)} chunks -> {output_path}")
        all_records.extend(records)

    print(f"generated {len(all_records)} chunks from {len(files)} documents")
    if args.chunk_only:
        return
    from app.services.aiops_knowledge_service import aiops_knowledge_service

    if args.rebuild:
        aiops_knowledge_service.rebuild()
    inserted = aiops_knowledge_service.add_records(all_records)
    print(f"indexed {inserted} chunks into {aiops_knowledge_service.collection_name}")


if __name__ == "__main__":
    main()
