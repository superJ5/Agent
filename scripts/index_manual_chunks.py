"""CLI helper to index structured manual chunks into Milvus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--skipped-chunk-report",
        default="logs/index_skipped_chunks_report.jsonl",
        help=(
            "JSONL report path for chunks skipped after per-chunk retry failed. "
            "The rest of the JSONL file continues indexing."
        ),
    )
    parser.add_argument(
        "--append-skipped-chunk-report",
        action="store_true",
        help=(
            "Append to the skipped chunk report instead of clearing it at startup. "
            "Use this when indexing multiple files via a shell loop."
        ),
    )
    return parser


def read_jsonl_report(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL report if it exists; malformed lines are surfaced as entries."""
    report_path = Path(path)
    if not report_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        report_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            entries.append(value if isinstance(value, dict) else {"value": value})
        except json.JSONDecodeError as exc:
            entries.append(
                {
                    "report_path": report_path.as_posix(),
                    "line_number": line_number,
                    "parse_error": str(exc),
                    "raw_line": line,
                }
            )
    return entries


def build_final_error_summary(
    *,
    failed_files: dict[str, str],
    skipped_chunk_report: str | Path,
    embedding_limit_report: str | Path,
) -> dict[str, Any]:
    """Build one final, end-of-run summary for all indexing problems."""
    file_errors = [
        {
            "file": file_path,
            "error": error,
        }
        for file_path, error in failed_files.items()
    ]
    single_chunk_errors = read_jsonl_report(skipped_chunk_report)
    embedding_limit_fallbacks = read_jsonl_report(embedding_limit_report)

    return {
        "has_errors": bool(file_errors or single_chunk_errors),
        "file_error_count": len(file_errors),
        "single_chunk_error_count": len(single_chunk_errors),
        "embedding_limit_fallback_count": len(embedding_limit_fallbacks),
        "file_errors": file_errors,
        "single_chunk_errors": single_chunk_errors,
        "embedding_limit_fallbacks": embedding_limit_fallbacks,
        "note": (
            "file_errors are whole-jsonl failures. single_chunk_errors are chunks "
            "skipped after per-chunk retry failed. embedding_limit_fallbacks are "
            "chunks whose raw embedding input hit the provider 8192-token limit and "
            "were retried with a truncated fallback."
        ),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.append_embedding_limit_report:
        vector_store_manager.append_embedding_limit_report(args.embedding_limit_report)
    else:
        vector_store_manager.reset_embedding_limit_report(args.embedding_limit_report)
    if args.append_skipped_chunk_report:
        vector_store_manager.append_skipped_chunk_report(args.skipped_chunk_report)
    else:
        vector_store_manager.reset_skipped_chunk_report(args.skipped_chunk_report)
    result = vector_index_service.index_manual_chunks(args.directory)
    payload = result.to_dict()
    payload["embedding_limit_report"] = (
        vector_store_manager.embedding_limit_report_summary()
    )
    payload["skipped_chunk_report"] = vector_store_manager.skipped_chunk_report_summary()
    payload["final_error_summary"] = build_final_error_summary(
        failed_files=result.failed_files,
        skipped_chunk_report=args.skipped_chunk_report,
        embedding_limit_report=args.embedding_limit_report,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
