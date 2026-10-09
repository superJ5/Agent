"""Convert the selected Cloud-OpsBench cases to a compact JSONL file."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.cloud_ops_bench import prepare_cases, write_jsonl

DEFAULT_ROOT = Path("data/aiops_eval/cloud_ops_bench_30")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.dataset_root / "cases.jsonl"
    cases = prepare_cases(args.dataset_root)
    count = write_jsonl(cases, output)
    print(f"prepared {count} cases -> {output}")


if __name__ == "__main__":
    main()
