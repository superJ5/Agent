"""Replay supplied LongMemEval conversations through this project's memory layers.

Historical assistant messages are imported verbatim. Only the final benchmark
question is answered by a model; the gold answer is never supplied to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen

from app.config import config
from app.core.request_context import (
    memory_enabled_override,
    memory_user_identity,
    structured_output_trace,
)
from app.services.context_budget import assess_context_budget
from app.services.long_term_memory_service import long_term_memory_service
from app.services.memory_service import memory_service
from app.services.session_state_service import session_state_service
from app.services.short_term_memory_service import short_term_memory_service

DATE_RE = re.compile(r"(\d{4}/\d{2}/\d{2}).*?(\d{2}:\d{2})")


@dataclass(frozen=True)
class ReplayResult:
    question_id: str
    hypothesis: str
    user_id: str
    session_count: int
    completed_turn_count: int
    compaction_count: int


def _session_timestamp(source_date: str) -> str:
    """Preserve the benchmark's local wall-clock date without inventing a zone."""
    match = DATE_RE.search(source_date)
    if not match:
        raise ValueError(f"无法解析 LongMemEval 会话日期: {source_date!r}")
    return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y/%m/%d %H:%M").isoformat()


def _eval_prompt(question_date: str) -> str:
    return (
        "You are answering a question about this user's past conversations. "
        "Use only the supplied session state, short-term context, and retrieved "
        "long-term memories. Do not use product-manual retrieval. If the needed "
        "fact is unavailable, say that you do not know rather than guessing. "
        f"The question is asked on {question_date}."
    )


class _MemoryOnlyContext:
    """Use the production memory services without initializing manual RAG."""

    def _build_short_term_context_messages(self, session_id, question):
        memory, answer_dialogue = short_term_memory_service.load_answer_context(
            session_id, current_question=question,
        )
        recent_dialogue = short_term_memory_service.load_recent_dialogue(
            session_id, current_question=question,
        )
        recent_text = short_term_memory_service.format_dialogue(answer_dialogue)
        sections = []
        if memory:
            sections.append("【短期语义记忆】\n" + memory)
        if recent_text:
            sections.append("【未压缩的原始对话】\n" + recent_text)
        if not sections:
            return [], recent_dialogue
        return [SystemMessage(content=(
            "以下是当前 session 的短期上下文，只作为续接当前任务的参考；"
            "如与当前用户问题冲突，以当前用户问题为准。\n\n"
            + "\n\n".join(sections)
        ))], recent_dialogue

    def _build_session_state_context_messages(self, session_id):
        context = session_state_service.build_context_block(session_id)
        return [SystemMessage(content=context)] if context else []

    def _build_long_term_context_messages(
        self, *, session_id, question, session_state_messages, short_term_messages,
    ):
        context, memories = long_term_memory_service.build_context_block(
            question=question,
            session_id=session_id,
            session_state_context="\n\n".join(str(m.content) for m in session_state_messages),
            short_term_context="\n\n".join(str(m.content) for m in short_term_messages),
        )
        return ([SystemMessage(content=context)] if context else []), memories

    @staticmethod
    def _budget(messages):
        return assess_context_budget(
            messages,
            working_window_tokens=config.memory_context_working_window_tokens,
            output_reserve_tokens=config.memory_context_output_reserve_tokens,
            retrieval_reserve_tokens=config.memory_context_retrieval_reserve_tokens,
            safety_margin_tokens=config.memory_context_safety_margin_tokens,
            compact_ratio=config.memory_context_compact_ratio,
        )

    async def _compact_if_needed(
        self, session_id, question, messages, *, model=None, dashscope_api_key=None,
    ):
        if not self._budget(messages).should_compact:
            return False
        plan = short_term_memory_service.prepare_compaction(
            session_id, current_question=question,
        )
        if plan is None:
            return False
        summarizer = ChatQwen(
            model=config.short_term_memory_model or model or config.rag_model,
            api_key=cast(Any, dashscope_api_key or config.dashscope_api_key),
            base_url=config.dashscope_api_base,
            temperature=0,
            streaming=False,
        )
        prompt = (
            "请将旧摘要与新增对话合并成一份当前会话摘要。只依据给定内容，不编造。"
            "保留目标、设备型号和编号、关键数字和时间、已确认事实、当前及已推翻假设、"
            "未完成任务和用户约束。用户问题若未得到有效回答，可记录为待处理；"
            "标记为回答失败的助手内容不能当成已确认结论。只输出摘要正文，不要代码块。"
            "尽量精炼，控制在约 1800 token 内。"
        )
        dialogue = short_term_memory_service.format_dialogue(plan.messages_to_summarize)
        response = await summarizer.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"旧摘要：\n{plan.old_summary or '（无）'}\n\n新增对话：\n{dialogue}"),
        ])
        short_term_memory_service.save_summary_with_boundary(
            session_id, self._message_text(response).strip(),
            through_message_seq=plan.through_message_seq,
        )
        return True

    def _ensure_context_budget(self, session_id, messages):
        budget = self._budget(messages)
        if budget.over_budget:
            raise RuntimeError(
                f"会话 {session_id} 的上下文超过输入预算："
                f"{budget.estimated_input_tokens}/{budget.available_input_tokens} tokens"
            )

    @staticmethod
    def _model_for_request():
        return ChatQwen(
            model=config.rag_model,
            api_key=cast(Any, config.dashscope_api_key),
            base_url=config.dashscope_api_base,
            temperature=0.2,
            streaming=False,
        )

    @staticmethod
    def _message_text(response):
        content = response.content
        return content if isinstance(content, str) else str(content)


class LongMemEvalReplayer:
    """One benchmark instance maps to one isolated test user and many sessions."""

    def __init__(self) -> None:
        self.agent = _MemoryOnlyContext()
        self.raw = memory_service
        self.state = session_state_service
        self.long = long_term_memory_service

    async def replay_case(self, entry: dict[str, Any], *, run_id: str) -> ReplayResult:
        if not config.memory_write_enabled:
            raise RuntimeError("MEMORY_WRITE_ENABLED=false；无法执行记忆回放")
        question_id = str(entry["question_id"])
        user_id = f"lme_{run_id}_{question_id}"
        if len(user_id) > 100:
            raise ValueError("评测 user_id 超过 Milvus 字段长度")

        session_ids = entry["haystack_session_ids"]
        session_dates = entry["haystack_dates"]
        sessions = entry["haystack_sessions"]
        if not (len(session_ids) == len(session_dates) == len(sessions)):
            raise ValueError(f"题目 {question_id} 的历史会话字段长度不一致")

        completed_turns = 0
        compactions = 0
        trace_path = memory_service.memory_root / "traces" / f"{run_id}.jsonl"
        with (
            memory_enabled_override(True),
            memory_user_identity(user_id),
            structured_output_trace(trace_path),
        ):
            for source_id, source_date, messages in zip(
                session_ids, session_dates, sessions, strict=True,
            ):
                session_id = f"lme_{run_id}_{question_id}_{source_id}"
                timestamp = _session_timestamp(source_date)
                pending: dict[str, Any] | None = None
                for item in messages:
                    role = item.get("role")
                    content = item.get("content")
                    if role not in {"user", "assistant"} or not isinstance(content, str):
                        raise ValueError(f"题目 {question_id} 存在不支持的历史消息")
                    metadata = {
                        "source": "longmemeval_replay",
                        "source_session_id": source_id,
                        "source_date": source_date,
                    }
                    if role == "user":
                        self.raw.append_message(
                            session_id, role, content, metadata=metadata, timestamp=timestamp,
                        )
                        short_messages, prior_dialogue = self.agent._build_short_term_context_messages(
                            session_id, content,
                        )
                        state_messages = self.agent._build_session_state_context_messages(session_id)
                        long_messages, retrieved_memories = self.agent._build_long_term_context_messages(
                            session_id=session_id, question=content,
                            session_state_messages=state_messages,
                            short_term_messages=short_messages,
                        )
                        budget_messages = [
                            SystemMessage(content=_eval_prompt(source_date)),
                            *long_messages, *state_messages, *short_messages,
                            HumanMessage(content=content),
                        ]
                        if await self.agent._compact_if_needed(
                            session_id, content, budget_messages,
                            model=None, dashscope_api_key=None,
                        ):
                            compactions += 1
                            short_messages, prior_dialogue = self.agent._build_short_term_context_messages(
                                session_id, content,
                            )
                            long_messages, retrieved_memories = self.agent._build_long_term_context_messages(
                                session_id=session_id, question=content,
                                session_state_messages=state_messages,
                                short_term_messages=short_messages,
                            )
                            budget_messages = [
                                SystemMessage(content=_eval_prompt(source_date)),
                                *long_messages, *state_messages, *short_messages,
                                HumanMessage(content=content),
                            ]
                        self.agent._ensure_context_budget(session_id, budget_messages)
                        pending = {
                            "user_text": content,
                            "prior_dialogue": prior_dialogue,
                            "state_context": "\n\n".join(str(m.content) for m in state_messages),
                            "short_context": "\n\n".join(str(m.content) for m in short_messages),
                            "retrieved_memories": retrieved_memories,
                        }
                        continue

                    self.raw.append_message(
                        session_id, role, content, metadata=metadata, timestamp=timestamp,
                    )
                    if pending is None or not content.strip():
                        continue
                    dated_user_text = f"[Historical session date: {source_date}] {pending['user_text']}"
                    await self.long.update_after_turn(
                        session_id=session_id,
                        user_id=user_id,
                        user_message=dated_user_text,
                        assistant_message=content,
                        session_state_context=pending["state_context"],
                        short_term_memory=pending["short_context"],
                        prior_dialogue=pending["prior_dialogue"],
                        retrieved_memories=pending["retrieved_memories"],
                    )
                    await self.state.update_after_turn(
                        session_id=session_id,
                        user_message=dated_user_text,
                        assistant_message=content,
                        prior_dialogue=pending["prior_dialogue"],
                    )
                    completed_turns += 1
                    pending = None

            question = str(entry["question"])
            question_date = str(entry["question_date"])
            final_session = f"lme_{run_id}_{question_id}_question"
            self.raw.append_message(
                final_session, "user", question,
                metadata={"source": "longmemeval_question", "source_date": question_date},
                timestamp=_session_timestamp(question_date),
            )
            short_messages, _ = self.agent._build_short_term_context_messages(final_session, question)
            state_messages = self.agent._build_session_state_context_messages(final_session)
            long_messages, _ = self.agent._build_long_term_context_messages(
                session_id=final_session, question=question,
                session_state_messages=state_messages, short_term_messages=short_messages,
            )
            answer_messages = [
                SystemMessage(content=_eval_prompt(question_date)),
                *long_messages, *state_messages, *short_messages,
                HumanMessage(content=question),
            ]
            self.agent._ensure_context_budget(final_session, answer_messages)
            response = await self.agent._model_for_request().ainvoke(answer_messages)
            hypothesis = self.agent._message_text(response).strip()
            self.raw.append_message(
                final_session, "assistant", hypothesis,
                metadata={"source": "longmemeval_answer", "status": "success"},
                timestamp=_session_timestamp(question_date),
            )

        return ReplayResult(
            question_id=question_id,
            hypothesis=hypothesis,
            user_id=user_id,
            session_count=len(sessions),
            completed_turn_count=completed_turns,
            compaction_count=compactions,
        )
