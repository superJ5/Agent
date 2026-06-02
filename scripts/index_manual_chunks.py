"""CLI helper to index structured manual chunks into Milvus."""

from __future__ import annotations

import argparse
import json

from app.services.vector_index_service import vector_index_service
from app.services.vector_store_manager import vector_store_manager


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
    parser.add_argument(
        "--embedding-limit-report",
        default="logs/embedding_input_limit_report.jsonl",
        help=(
            "JSONL report path for chunks whose raw embedding input is rejected "
            "by the provider input-token limit and then retried with a fallback."
        ),
    )
    parser.add_argument(
        "--append-embedding-limit-report",
        action="store_true",
        help=(
            "Append to the embedding limit report instead of clearing it at startup. "
            "Use this when indexing multiple files via a shell loop."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.append_embedding_limit_report:
        vector_store_manager.append_embedding_limit_report(args.embedding_limit_report)
    else:
        vector_store_manager.reset_embedding_limit_report(args.embedding_limit_report)
    result = vector_index_service.index_manual_chunks(args.directory)
    payload = result.to_dict()
    payload["embedding_limit_report"] = (
        vector_store_manager.embedding_limit_report_summary()
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
