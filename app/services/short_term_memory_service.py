"""Short-term semantic memory for one chat session.

This layer keeps a compact, task-focused rolling summary for the current
session. Raw session JSONL remains the source of truth; this file is only the
model-readable working memory used by future turns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config
from app.services.memory_service import memory_service


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHORT_TERM_MAX_CHARS = 2000
RECENT_DIALOGUE_ROUNDS = 3


class ShortTermMemoryService:
    """Read, write, and update per-session short-term semantic memory."""

    def __init__(self) -> None:
        self.short_term_dir = self._resolve_project_path(config.memory_root) / "short_term"

    def load_memory(self, session_id: str) -> str:
        """Load the current short-term memory markdown for one session."""
        path = self._memory_file(session_id)
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("读取短期记忆失败: session_id={}, error={}", session_id, exc)
            return ""

    def load_recent_dialogue(
        self,
        session_id: str,
        *,
        current_question: str | None = None,
        rounds: int = RECENT_DIALOGUE_ROUNDS,
    ) -> list[dict[str, Any]]:
        """Load recent raw dialogue records used as a small context aid."""
        limit = max(rounds, 0) * 2
        if limit <= 0:
            return []
        return memory_service.load_recent_messages(
            session_id,
            limit=limit,
            exclude_latest_user_content=current_question,
        )

    def build_context_block(
        self,
        session_id: str,
        *,
        current_question: str | None = None,
    ) -> str:
        """Build the short-term memory block injected into the answer context."""
        sections: list[str] = []
        memory = self.load_memory(session_id)
        if memory:
            sections.append("【短期语义记忆】\n" + memory)

        recent = self.load_recent_dialogue(
            session_id,
            current_question=current_question,
        )
        recent_text = self.format_dialogue(recent)
        if recent_text:
            sections.append("【最近原始对话】\n" + recent_text)

        return "\n\n".join(sections).strip()

    async def update_after_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_message: str,
        prior_dialogue: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update short-term memory after one completed user/assistant turn."""
        if not config.memory_write_enabled:
            logger.debug("跳过短期记忆写入: MEMORY_WRITE_ENABLED=false")
            return

        user_text = str(user_message or "").strip()
        assistant_text = str(assistant_message or "").strip()
        if not user_text and not assistant_text:
            return

        old_memory = self.load_memory(session_id)
        prior_text = self.format_dialogue(prior_dialogue or [])
        try:
            new_memory = await self._generate_updated_memory(
                old_memory=old_memory,
                prior_dialogue=prior_text,
                user_message=user_text,
                assistant_message=assistant_text,
            )
        except Exception as exc:
            logger.warning("短期记忆更新失败: session_id={}, error={}", session_id, exc)
            return

        new_memory = self._clean_memory(new_memory)
        if not new_memory:
            return
        if new_memory == old_memory.strip():
            return

        self._write_memory(session_id, new_memory)

    @staticmethod
    def format_dialogue(records: list[dict[str, Any]]) -> str:
        """Format raw records into a compact readable dialogue snippet."""
        lines: list[str] = []
        for record in records:
            role = str(record.get("role") or "").strip()
            content = " ".join(str(record.get("content") or "").split()).strip()
            if not content or role not in {"user", "assistant"}:
                continue
            label = "用户" if role == "user" else "助手"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    async def _generate_updated_memory(
        self,
        *,
        old_memory: str,
        prior_dialogue: str,
        user_message: str,
        assistant_message: str,
    ) -> str:
        llm = self._build_llm()
        messages = [
            SystemMessage(content=self._update_system_prompt()),
            HumanMessage(
                content=dedent(f"""
                    旧短期语义记忆：
                    {old_memory or "（空）"}

                    最近 1-3 轮原始对话：
                    {prior_dialogue or "（空）"}

                    本轮用户问题：
                    {user_message}

                    本轮助手回答：
                    {assistant_message}

                    请输出更新后的短期语义记忆。
                """).strip()
            ),
        ]
        result = await llm.ainvoke(messages)
        return self._message_text(result)

    @staticmethod
    def _update_system_prompt() -> str:
        return dedent("""
            你是短期记忆更新器。你的任务是维护当前 session 的语义短期记忆，
            让下一轮回答能接上当前任务进展。

            请严格遵守：
            - 只保留对当前 session 后续回答有帮助的信息。
            - 不要记录寒暄、重复内容、模板话术、无关细节。
            - 不要编造输入中没有的信息。
            - 用户或证据确认过的信息才能写入“关键事实”。
            - 模型猜测但尚未确认的信息只能写入“当前假设”。
            - 被证伪但对后续有用的信息写入“已排除方向”，不要继续放在假设里。
            - 如果新信息纠正旧信息，以新信息为准，并移除或改写旧内容。
            - 输出必须简洁，整体不超过 1200 个中文字符。
            - 只输出 Markdown 正文，不要解释你的更新过程。

            固定使用以下栏目；没有内容的栏目写“无”：
            目标：
            关键事实：
            当前假设：
            已排除方向：
            下一步动作：
            用户约束：
        """).strip()

    def _write_memory(self, session_id: str, content: str) -> None:
        try:
            self.short_term_dir.mkdir(parents=True, exist_ok=True)
            path = self._memory_file(session_id)
            header = (
                f"<!-- updated_at: {datetime.now(timezone.utc).isoformat()} -->\n"
            )
            path.write_text(header + content.strip() + "\n", encoding="utf-8")
        except Exception as exc:
            logger.warning("写入短期记忆失败: session_id={}, error={}", session_id, exc)

    def _memory_file(self, session_id: str) -> Path:
        safe_session_id = memory_service._safe_session_id(session_id)
        return self.short_term_dir / f"{safe_session_id}.md"

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
        if len(text) > SHORT_TERM_MAX_CHARS:
            text = text[:SHORT_TERM_MAX_CHARS].rstrip()
        return text

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text") or block.get("content") or ""))
                else:
                    parts.append(str(block))
            return "\n".join(part for part in parts if part.strip())
        return str(content or "")

    @staticmethod
    def _build_llm() -> ChatQwen:
        model_name = config.short_term_memory_model or config.rag_model
        return ChatQwen(
            model=model_name,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0,
            timeout=60,
        )

    @staticmethod
    def _resolve_project_path(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


short_term_memory_service = ShortTermMemoryService()
