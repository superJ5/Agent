"""CLI helper to summarize raw JSONL chat memory into daily memory."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date
import json

from app.services.memory_summary_service import memory_summary_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize data/memory/sessions JSONL records into daily memory.",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Local date to summarize, in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Update the Milvus memory index after writing daily memory.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Drop and rebuild the memory collection when --index is used.",
    )
    return parser


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date)
    result = await memory_summary_service.summarize_date(
        target_date,
        update_index=args.index,
        rebuild_index=args.rebuild_index,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
