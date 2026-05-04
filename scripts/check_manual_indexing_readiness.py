"""Preflight checks for manual chunk indexing."""

from __future__ import annotations

import json
import socket
from pathlib import Path

from app.config import config


PLACEHOLDER_API_KEYS = {
    "",
    "your-api-key-here",
}


def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_report() -> dict:
    chunk_dir = Path("./data/manuals/chunks").resolve()
    chunk_files = sorted(chunk_dir.glob("*.jsonl")) if chunk_dir.exists() else []
    dashscope_key = (config.dashscope_api_key or "").strip()

    report = {
        "ready": False,
        "dashscope_api_key_ready": dashscope_key not in PLACEHOLDER_API_KEYS,
        "milvus_reachable": is_port_open(config.milvus_host, config.milvus_port),
        "manual_chunk_dir": str(chunk_dir),
        "manual_chunk_dir_exists": chunk_dir.exists(),
        "manual_chunk_file_count": len(chunk_files),
        "sample_chunk_files": [path.name for path in chunk_files[:5]],
        "milvus_host": config.milvus_host,
        "milvus_port": config.milvus_port,
    }
    report["ready"] = (
        report["dashscope_api_key_ready"]
        and report["milvus_reachable"]
        and report["manual_chunk_dir_exists"]
        and report["manual_chunk_file_count"] > 0
    )
    return report


def main() -> None:
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
