"""CLI helper to index persistent memory files into Milvus."""

from __future__ import annotations

import argparse
import json

from app.services.memory_index_service import memory_index_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index data/memory files into the Milvus memory collection.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the memory collection before indexing.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = memory_index_service.index_memory(rebuild=args.rebuild)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
