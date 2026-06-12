"""Structured session state for the current task.

Session state is a compact JSON state table maintained per session. Unlike
short-term semantic memory, it is meant to be program-readable and easy to
validate, merge, and inject into the model as a concise status block.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.services.memory_service import memory_service
from app.services.short_term_memory_service import short_term_memory_service


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_LIST_ITEMS = 8
MAX_ITEM_CHARS = 180


class SessionState(BaseModel):
    """Program-readable current-session task state."""

    session_id: str = ""
    goal: str = ""
    confirmed_facts: list[str] = Field(default_factory=list)
    current_hypothesis: list[str] = Field(default_factory=list)
    rejected_hypotheses: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    user_constraints: list[str] = Field(default_factory=list)
    updated_at: str = ""


class SessionStateUpdate(BaseModel):
    """Structured output from the session-state updater model."""

    should_update: bool = False
    reason: str = ""
    state: SessionState = Field(default_factory=SessionState)


class SessionStateService:
    """Read, write, inject, and update per-session structured state."""

    def __init__(self) -> None:
        self.session_state_dir = self._resolve_project_path(config.memory_root) / "session_state"

    def load_state(self, session_id: str) -> SessionState | None:
        """Load current session state if it exists and has meaningful content."""
        path = self._state_file(session_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            state = SessionState.model_validate(raw)
            cleaned = self._clean_state(state, session_id=session_id)
            return cleaned if self._has_meaningful_state(cleaned) else None
        except Exception as exc:
            logger.warning("读取 Session State 失败: session_id={}, error={}", session_id, exc)
            return None

    def build_context_block(self, session_id: str) -> str:
        """Render session state as model-readable context."""
        state = self.load_state(session_id)
        if state is None:
            return ""

        sections: list[str] = ["【当前会话状态】"]
        if state.goal:
            sections.append(f"目标：{state.goal}")
        sections.extend(self._render_list("已确认事实", state.confirmed_facts))
        sections.extend(self._render_list("当前假设", state.current_hypothesis))
        sections.extend(self._render_list("已排除方向", state.rejected_hypotheses))
        sections.extend(self._render_list("下一步动作", state.next_actions))
        sections.extend(self._render_list("用户约束", state.user_constraints))
        sections.append("如当前用户问题与上述状态冲突，以当前用户问题为准。")
        return "\n".join(sections).strip()

    async def update_after_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_message: str,
        prior_dialogue: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update state after one completed turn."""
        if not config.memory_write_enabled:
            logger.debug("跳过 Session State 写入: MEMORY_WRITE_ENABLED=false")
            return

        user_text = str(user_message or "").strip()
        assistant_text = str(assistant_message or "").strip()
        if not user_text and not assistant_text:
            return

        old_state = self.load_state(session_id)
        short_term_memory = short_term_memory_service.load_memory(session_id)
        prior_text = short_term_memory_service.format_dialogue(prior_dialogue or [])

        try:
            update = await self._generate_update(
                session_id=session_id,
                old_state=old_state,
                short_term_memory=short_term_memory,
                prior_dialogue=prior_text,
                user_message=user_text,
                assistant_message=assistant_text,
            )
        except Exception as exc:
            logger.warning("Session State 更新失败: session_id={}, error={}", session_id, exc)
            return

        if not update.should_update:
            logger.debug(
                "Session State 无需更新: session_id={}, reason={}",
                session_id,
                update.reason,
            )
            return

        new_state = self._clean_state(update.state, session_id=session_id)
        if not self._has_meaningful_state(new_state):
            logger.debug(
                "Session State 更新结果为空，跳过写入: session_id={}, reason={}",
                session_id,
                update.reason,
            )
            return

        old_payload = old_state.model_dump() if old_state else None
        if old_payload == new_state.model_dump():
            return

        self._write_state(new_state)

    async def _generate_update(
        self,
        *,
        session_id: str,
        old_state: SessionState | None,
        short_term_memory: str,
        prior_dialogue: str,
        user_message: str,
        assistant_message: str,
    ) -> SessionStateUpdate:
        llm = self._build_llm()
        structured_llm = llm.with_structured_output(SessionStateUpdate)
        messages = [
            SystemMessage(content=self._update_system_prompt()),
            HumanMessage(
                content=dedent(f"""
                    session_id：
                    {session_id}

                    旧 session_state JSON：
                    {self._state_json(old_state) if old_state else "（空）"}

                    旧短期语义记忆：
                    {short_term_memory or "（空）"}

                    最近 1-3 轮原始对话：
                    {prior_dialogue or "（空）"}

                    本轮用户问题：
                    {user_message}

                    本轮助手回答：
                    {assistant_message}

                    请输出 SessionStateUpdate。
                """).strip()
            ),
        ]
        result = await structured_llm.ainvoke(messages)
        if isinstance(result, SessionStateUpdate):
            return result
        return SessionStateUpdate.model_validate(result)

    @staticmethod
    def _update_system_prompt() -> str:
        return dedent("""
            你是 Session State 更新器。你的任务是维护当前 session 的结构化任务状态。

            你会收到旧 session_state、短期语义记忆、最近原始对话、本轮用户问题和本轮助手回答。
            请输出一个完整的新版 SessionStateUpdate，而不是增量 patch。

            输出规则：
            - should_update 表示本轮是否值得写入或覆盖 session_state 文件。
            - 如果只是寒暄、感谢、简单闲聊、无明确任务状态变化，should_update=false。
            - 如果本轮新增了目标、确认事实、排除方向、当前假设、下一步动作或用户约束，should_update=true。
            - state 必须是完整的新状态，不是只包含本轮新增内容。
            - 不要编造输入中没有的信息。
            - confirmed_facts 只放用户明确表达、工具/日志证据、或已被本轮确认的信息。
            - current_hypothesis 只放仍可能成立、需要继续验证的判断。
            - rejected_hypotheses 放已经被用户、证据或排查结果排除的方向。
            - rejected_hypotheses 里的内容不能继续出现在 current_hypothesis。
            - next_actions 只保留当前仍需要执行的动作，移除已完成动作。
            - user_constraints 只保留用户明确提出的限制、偏好或条件。
            - 如果用户改变任务目标，可以更新 goal，并保留仍相关的事实。
            - reason 简短说明为什么更新或为什么不更新。
        """).strip()

    def _write_state(self, state: SessionState) -> None:
        try:
            self.session_state_dir.mkdir(parents=True, exist_ok=True)
            self._state_file(state.session_id).write_text(
                json.dumps(state.model_dump(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("写入 Session State 失败: session_id={}, error={}", state.session_id, exc)

    def _state_file(self, session_id: str) -> Path:
        safe_session_id = memory_service._safe_session_id(session_id)
        return self.session_state_dir / f"{safe_session_id}.json"

    def _clean_state(self, state: SessionState, *, session_id: str) -> SessionState:
        rejected = self._clean_items(state.rejected_hypotheses)
        current = [
            item
            for item in self._clean_items(state.current_hypothesis)
            if not self._conflicts_with_any(item, rejected)
        ]
        return SessionState(
            session_id=session_id,
            goal=self._clean_text(state.goal, max_chars=240),
            confirmed_facts=self._clean_items(state.confirmed_facts),
            current_hypothesis=current,
            rejected_hypotheses=rejected,
            next_actions=self._clean_items(state.next_actions),
            user_constraints=self._clean_items(state.user_constraints),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _clean_items(items: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = SessionStateService._clean_text(item, max_chars=MAX_ITEM_CHARS)
            key = text.lower()
            if not text or key in seen:
                continue
            cleaned.append(text)
            seen.add(key)
            if len(cleaned) >= MAX_LIST_ITEMS:
                break
        return cleaned

    @staticmethod
    def _clean_text(value: str, *, max_chars: int) -> str:
        text = " ".join(str(value or "").split()).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip()
        return text

    @staticmethod
    def _conflicts_with_any(item: str, rejected_items: list[str]) -> bool:
        normalized = item.lower()
        return any(
            normalized == rejected.lower() or normalized in rejected.lower() or rejected.lower() in normalized
            for rejected in rejected_items
        )

    @staticmethod
    def _has_meaningful_state(state: SessionState) -> bool:
        return bool(
            state.goal.strip()
            or state.confirmed_facts
            or state.rejected_hypotheses
            or state.next_actions
            or state.user_constraints
        )

    @staticmethod
    def _render_list(title: str, items: list[str]) -> list[str]:
        if not items:
            return []
        lines = [f"{title}："]
        lines.extend(f"- {item}" for item in items)
        return lines

    @staticmethod
    def _state_json(state: SessionState) -> str:
        return json.dumps(state.model_dump(), ensure_ascii=False, indent=2)

    @staticmethod
    def _build_llm() -> ChatQwen:
        model_name = (
            config.session_state_model
            or config.short_term_memory_model
            or config.rag_model
        )
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


session_state_service = SessionStateService()
