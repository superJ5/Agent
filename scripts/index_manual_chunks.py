"""CLI helper to index structured manual chunks into Milvus."""

from __future__ import annotations

import argparse
import json

from app.services.vector_index_service import vector_index_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Index structured manual chunk JSONL files into Milvus. "
            "Supports a single JSONL file or a root directory with nested chunks.jsonl files."
        ),
    )
    parser.add_argument(
        "--directory",
        default="./data/manuals/chunks",
        help=(
            "A chunk JSONL file, or a directory containing chunk JSONL files. "
            "When a root directory is provided, nested chunks.jsonl files are discovered recursively."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = vector_index_service.index_manual_chunks(args.directory)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
