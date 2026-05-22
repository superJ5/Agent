"""Persistent session memory service.

This module stores raw chat messages as JSONL files under a configurable
project-relative directory. It is intentionally independent from Agent tools:
API routes use it directly for archiving and current-session recovery.
Agent memory search intentionally uses only curated long-term/daily notes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from loguru import logger

from app.config import config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAFE_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class MemoryService:
    """Append-only JSONL storage for raw chat messages."""

    def __init__(self) -> None:
        self.memory_root = self._resolve_project_path(config.memory_root)
        self.sessions_dir = self.memory_root / "sessions"
        self.daily_dir = self.memory_root / "daily"
        self.long_term_memory_file = self.memory_root / "MEMORY.md"
        self.memory_pending_file = self.memory_root / "MEMORY_PENDING.md"

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one chat message to ``sessions/<session_id>.jsonl``."""
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

    def load_long_term_memory(self) -> str:
        """Load project-level long-term memory from MEMORY.md."""
        try:
            if not self.long_term_memory_file.exists():
                return ""
            return self.long_term_memory_file.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("读取长期记忆失败: {}", exc)
            return ""

    def append_daily_note(self, text: str) -> None:
        """Append one line to today's daily memory note."""
        note = str(text or "").strip()
        if not note:
            return

        try:
            self.daily_dir.mkdir(parents=True, exist_ok=True)
            date_key = datetime.now(timezone.utc).date().isoformat()
            daily_file = self.daily_dir / f"{date_key}.md"
            if not daily_file.exists():
                daily_file.write_text(f"# {date_key}\n\n", encoding="utf-8")
            timestamp = datetime.now(timezone.utc).isoformat()
            with daily_file.open("a", encoding="utf-8") as file:
                file.write(f"- {timestamp} {note}\n")
        except Exception as exc:
            logger.warning("写入每日记忆失败: {}", exc)

    def search_memory(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Simple keyword search over curated daily memory."""
        terms = self._extract_terms(query)
        if not terms:
            return []

        candidates: list[dict[str, Any]] = []
        if self.daily_dir.exists():
            for path in sorted(self.daily_dir.glob("*.md"), reverse=True):
                candidates.extend(self._search_text_file(path, "daily", terms))

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[: max(limit, 0)]

    def _session_file(self, session_id: str) -> Path:
        safe_session_id = self._safe_session_id(session_id)
        return self.sessions_dir / f"{safe_session_id}.jsonl"

    def _search_text_file(
        self,
        path: Path,
        source_type: str,
        terms: set[str],
    ) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        hits: list[dict[str, Any]] = []
        try:
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                score = self._score_text(line, terms)
                if score <= 0:
                    continue
                hits.append(
                    {
                        "source_type": source_type,
                        "path": self._to_project_relative(path),
                        "line": line_no,
                        "content": line.strip(),
                        "score": score,
                    }
                )
        except Exception as exc:
            logger.warning("搜索记忆文件失败: path={}, error={}", path, exc)
        return hits

    @staticmethod
    def _extract_terms(query: str) -> set[str]:
        return {
            term.lower()
            for term in re.findall(r"[A-Za-z0-9_+\-]{2,}|[\u4e00-\u9fff]{2,}", str(query or ""))
        }

    @staticmethod
    def _score_text(text: str, terms: set[str]) -> int:
        lowered = text.lower()
        return sum(1 for term in terms if term in lowered)

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
