"""Split a retrieval_trace JSONL file into one JSONL file per line."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "single"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a JSONL trace file into single-record JSONL files.",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Source retrieval_trace JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=None,
        help="Only split the last N records. Default: split all records.",
    )
    parser.add_argument(
        "--prefix",
        default="retrieval_trace",
        help="Output filename prefix. Default: retrieval_trace",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Source file encoding. Default: utf-8-sig",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing files with the same prefix in output-dir.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()

    if not source.exists():
        raise SystemExit(f"Source file does not exist: {source}")
    if args.tail is not None and args.tail <= 0:
        raise SystemExit("--tail must be a positive integer when provided.")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tail is None:
        records = list(iter_jsonl_records(source, args.encoding))
        suffix = "all"
    else:
        records = list(tail_jsonl_records(source, args.encoding, args.tail))
        suffix = f"last{args.tail}"

    if not args.keep_existing:
        for old_file in output_dir.glob(f"{args.prefix}_{suffix}_*.jsonl"):
            old_file.unlink()

    width = max(3, len(str(len(records))))
    for index, record in enumerate(records, 1):
        target = output_dir / f"{args.prefix}_{suffix}_{index:0{width}d}.jsonl"
        target.write_text(record.rstrip("\n") + "\n", encoding="utf-8")

    print(f"Source: {source}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'all records' if args.tail is None else f'last {args.tail} records'}")
    print(f"Files written: {len(records)}")
    if records:
        print(f"First: {args.prefix}_{suffix}_{1:0{width}d}.jsonl")
        print(f"Last: {args.prefix}_{suffix}_{len(records):0{width}d}.jsonl")


def iter_jsonl_records(source: Path, encoding: str):
    with source.open("r", encoding=encoding) as handle:
        for line in handle:
            if line.strip():
                yield line.rstrip("\n")


def tail_jsonl_records(source: Path, encoding: str, limit: int):
    records: deque[str] = deque(maxlen=limit)
    for line in iter_jsonl_records(source, encoding):
        records.append(line)
    return records


if __name__ == "__main__":
    main()
