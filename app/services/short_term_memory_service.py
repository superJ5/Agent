"""Short-term summary snapshots and raw-message context for one session."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config
from app.core.request_context import is_memory_enabled
from app.services.context_budget import estimate_text_tokens
from app.services.memory_service import memory_service

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHORT_TERM_MAX_TOKENS = 2048
RECENT_DIALOGUE_ROUNDS = 3
SUMMARY_BOUNDARY_RE = re.compile(r"^<!-- summary_through_message_seq: (\d+) -->$", re.MULTILINE)


@dataclass(frozen=True)
class CompactionPlan:
    old_summary: str
    messages_to_summarize: list[dict[str, Any]]
    through_message_seq: int


class ShortTermMemoryService:
    """Read and write per-session summary snapshots and context."""

    def __init__(self) -> None:
        self.short_term_dir = self._resolve_project_path(config.memory_root) / "short_term"

    def load_memory(self, session_id: str) -> str:
        """Load only a summary with an explicit raw-message boundary."""
        if not is_memory_enabled():
            return ""

        if self.load_summary_through_message_seq(session_id) is None:
            return ""
        path = self._memory_file(session_id)
        if not path.exists():
            return ""
        try:
            return "\n".join(
                line for line in path.read_text(encoding="utf-8").splitlines()
                if not line.startswith("<!-- updated_at:")
                and not line.startswith("<!-- summary_through_message_seq:")
            ).strip()
        except Exception as exc:
            logger.warning("读取短期记忆失败: session_id={}, error={}", session_id, exc)
            return ""

    def load_summary_through_message_seq(self, session_id: str) -> int | None:
        """Read the compacted-prefix boundary, if a valid snapshot exists."""
        if not is_memory_enabled():
            return None
        path = self._memory_file(session_id)
        if not path.exists():
            return None
        try:
            match = SUMMARY_BOUNDARY_RE.search(path.read_text(encoding="utf-8"))
            return int(match.group(1)) if match else None
        except Exception as exc:
            logger.warning("读取摘要边界失败: session_id={}, error={}", session_id, exc)
            return None

    def save_summary_with_boundary(
        self, session_id: str, content: str, *, through_message_seq: int
    ) -> None:
        """Persist a summary with an explicit raw-message boundary.

        This is the storage API for the later compaction step.
        """
        events = memory_service.load_message_events(session_id)
        if through_message_seq < 1 or through_message_seq > len(events):
            raise ValueError("summary_through_message_seq 必须指向已保存的原始消息")
        previous = self.load_summary_through_message_seq(session_id)
        if previous is not None and through_message_seq < previous:
            raise ValueError("summary_through_message_seq 不能倒退")
        cleaned = self._clean_memory(content)
        if not cleaned:
            raise ValueError("摘要不能为空")
        if estimate_text_tokens(cleaned) > SHORT_TERM_MAX_TOKENS:
            raise ValueError("摘要超过 token 预算，不允许截断后推进边界")
        self._write_memory(session_id, cleaned, summary_through_message_seq=through_message_seq)

    def prepare_compaction(
        self, session_id: str, *, current_question: str | None = None
    ) -> CompactionPlan | None:
        """Summarize only the unsummarized prefix before the latest three user turns."""
        if not is_memory_enabled():
            return None
        events = memory_service.load_message_events(session_id)
        if (
            events and current_question is not None and events[-1]["role"] == "user"
            and str(events[-1]["content"]).strip() == current_question.strip()
        ):
            events = events[:-1]
        user_indexes = [i for i, event in enumerate(events) if event["role"] == "user"]
        if len(user_indexes) <= RECENT_DIALOGUE_ROUNDS:
            return None
        prefix = events[:user_indexes[-RECENT_DIALOGUE_ROUNDS]]
        boundary = self.load_summary_through_message_seq(session_id) or 0
        pending = [event for event in prefix if event["message_seq"] > boundary]
        if not pending:
            return None
        return CompactionPlan(
            old_summary=self.load_memory(session_id) if boundary else "",
            messages_to_summarize=pending,
            through_message_seq=pending[-1]["message_seq"],
        )

    def load_answer_context(
        self, session_id: str, *, current_question: str | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return summary and every message after its boundary, without overlap."""
        if not is_memory_enabled():
            return "", []
        events = memory_service.load_message_events(session_id)
        if (
            events
            and current_question is not None
            and events[-1]["role"] == "user"
            and str(events[-1]["content"]).strip() == current_question.strip()
        ):
            events = events[:-1]

        boundary = self.load_summary_through_message_seq(session_id)
        if boundary is None:
            return "", events
        if not events or boundary > events[-1]["message_seq"]:
            logger.warning("摘要边界超过原始消息: session_id={}, boundary={}", session_id, boundary)
            return "", events
        return self.load_memory(session_id), [
            event for event in events if event["message_seq"] > boundary
        ]

    def load_recent_dialogue(
        self,
        session_id: str,
        *,
        current_question: str | None = None,
        rounds: int = RECENT_DIALOGUE_ROUNDS,
    ) -> list[dict[str, Any]]:
        """Load recent raw dialogue records used as a small context aid."""
        if not is_memory_enabled():
            return []

        limit = max(rounds, 0) * 2
        if limit <= 0:
            return []
        return memory_service.load_recent_messages(
            session_id,
            limit=limit,
            exclude_latest_user_content=current_question,
            include_failed=True,
        )

    def build_context_block(
        self,
        session_id: str,
        *,
        current_question: str | None = None,
    ) -> str:
        """Build the short-term memory block injected into the answer context."""
        sections: list[str] = []
        memory, dialogue = self.load_answer_context(
            session_id,
            current_question=current_question,
        )
        if memory:
            sections.append("【短期语义记忆】\n" + memory)

        recent_text = self.format_dialogue(dialogue)
        if recent_text:
            sections.append("【未压缩的原始对话】\n" + recent_text)

        return "\n\n".join(sections).strip()

    @staticmethod
    def format_dialogue(records: list[dict[str, Any]]) -> str:
        """Format raw records into a compact readable dialogue snippet."""
        lines: list[str] = []
        for record in records:
            role = str(record.get("role") or "").strip()
            content = " ".join(str(record.get("content") or "").split()).strip()
            metadata = record.get("metadata")
            if role == "assistant" and isinstance(metadata, dict) and metadata.get("status") == "error":
                lines.append("助手: （本次回答失败，没有有效回答）")
                continue
            if not content or role not in {"user", "assistant"}:
                continue
            label = "用户" if role == "user" else "助手"
            if isinstance(metadata, dict) and metadata.get("source_date"):
                label += f"（{metadata['source_date']}）"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    def _write_memory(
        self, session_id: str, content: str, *, summary_through_message_seq: int | None = None
    ) -> None:
        self.short_term_dir.mkdir(parents=True, exist_ok=True)
        path = self._memory_file(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"<!-- updated_at: {datetime.now(UTC).isoformat()} -->\n"
        if summary_through_message_seq is not None:
            header += f"<!-- summary_through_message_seq: {summary_through_message_seq} -->\n"
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.short_term_dir,
                prefix=f".{path.stem}.", suffix=".tmp", delete=False,
            ) as file:
                temporary_path = file.name
                file.write(header + content.strip() + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _memory_file(self, session_id: str) -> Path:
        return memory_service.scoped_session_path(self.short_term_dir, session_id, ".md")

    @staticmethod
    def _clean_memory(content: str) -> str:
        text = str(content or "").strip()
        if not text:
            return ""
        lines = [
            line.rstrip()
            for line in text.splitlines()
            if not line.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _resolve_project_path(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


short_term_memory_service = ShortTermMemoryService()
