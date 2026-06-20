"""Persistent session memory service.

This module stores raw chat messages as JSONL files under a configurable
project-relative directory. It is intentionally independent from Agent tools:
API routes use it directly for archiving and current-session recovery.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from loguru import logger

from app.config import config
from app.core.request_context import is_memory_enabled


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAFE_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class MemoryService:
    """Append-only JSONL storage for raw chat messages."""

    def __init__(self) -> None:
        self.memory_root = self._resolve_project_path(config.memory_root)
        self.sessions_dir = self.memory_root / "sessions"

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one chat message to ``sessions/<session_id>.jsonl``."""
        if not is_memory_enabled():
            logger.debug("跳过会话记忆写入: MEMORY_ENABLED=false")
            return
        if not config.memory_write_enabled:
            logger.debug("跳过会话记忆写入: MEMORY_WRITE_ENABLED=false")
            return

        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
            }
            session_file = self._session_file(session_id)
            with session_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(
                "写入会话记忆失败: session_id={}, role={}, error={}",
                session_id,
                role,
                exc,
            )

    def load_recent_messages(
        self,
        session_id: str,
        limit: int | None = None,
        exclude_latest_user_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load recent user/assistant messages from a session JSONL file."""
        if not is_memory_enabled():
            return []

        session_file = self._session_file(session_id)
        if not session_file.exists():
            return []

        max_items = limit if limit is not None else config.memory_recent_limit
        records: list[dict[str, Any]] = []
        try:
            for line in session_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("跳过非法会话记忆行: {}", session_file)
                    continue
                role = record.get("role")
                content = record.get("content")
                if role not in {"user", "assistant"}:
                    continue
                if not isinstance(content, str) or not content.strip():
                    continue
                records.append(record)
        except Exception as exc:
            logger.warning("读取会话记忆失败: session_id={}, error={}", session_id, exc)
            return []

        if (
            records
            and exclude_latest_user_content is not None
            and records[-1].get("role") == "user"
            and str(records[-1].get("content", "")).strip() == exclude_latest_user_content.strip()
        ):
            records = records[:-1]

        return records[-max_items:] if max_items > 0 else []

    def _session_file(self, session_id: str) -> Path:
        safe_session_id = self._safe_session_id(session_id)
        return self.sessions_dir / f"{safe_session_id}.jsonl"

    @staticmethod
    def _to_project_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        normalized = SAFE_SESSION_ID_RE.sub("_", str(session_id).strip())
        normalized = normalized.strip("._")
        return normalized or "default"

    @staticmethod
    def _resolve_project_path(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


memory_service = MemoryService()
