"""Automatic summarization for persistent memory files."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import json
from pathlib import Path
import re
from textwrap import dedent
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.services.memory_service import PROJECT_ROOT, memory_service


LOCAL_TZ = ZoneInfo("Asia/Shanghai")


class DailyMemorySummary(BaseModel):
    """Structured daily memory summary returned by the LLM."""

    summary_items: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    todos: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    long_term_candidates: list[str] = Field(default_factory=list)


class CompactedDailyMemory(BaseModel):
    """Structured compacted daily memory returned by the LLM."""

    content: str = Field(description="Compacted Markdown content for the daily memory file.")


@dataclass
class MemorySummaryResult:
    """Result payload for summary CLI and logs."""

    success: bool = False
    date: str = ""
    records_read: int = 0
    daily_path: str = ""
    pending_path: str = ""
    daily_chars: int = 0
    pending_candidates: int = 0
    compacted: bool = False
    indexed: bool = False
    error_message: str = ""
    indexed_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "date": self.date,
            "records_read": self.records_read,
            "daily_path": self.daily_path,
            "pending_path": self.pending_path,
            "daily_chars": self.daily_chars,
            "pending_candidates": self.pending_candidates,
            "compacted": self.compacted,
            "indexed": self.indexed,
            "error_message": self.error_message,
            "indexed_sources": self.indexed_sources,
        }


class MemorySummaryService:
    """Summarize raw JSONL chat archives into curated daily memory."""

    async def summarize_date(
        self,
        target_date: date,
        update_index: bool = False,
        rebuild_index: bool = False,
    ) -> MemorySummaryResult:
        """Summarize one local date into daily memory and pending long-term candidates."""
        result = MemorySummaryResult(date=target_date.isoformat())
        try:
            records = self._load_records_for_date(target_date)
            result.records_read = len(records)
            result.daily_path = self._to_project_relative(self._daily_file(target_date))
            result.pending_path = self._to_project_relative(memory_service.memory_pending_file)

            if not records:
                result.success = True
                logger.info("No memory records found for {}", target_date)
                return result

            transcript = self._format_records(records)
            if not transcript:
                self._remove_daily(target_date)
                result.success = True
                logger.info("No high-value memory records found for {}", target_date)
                return result

            summary = await self._summarize_transcript(target_date, transcript)
            summary = self._filter_summary(summary)
            if not self._has_meaningful_summary(summary):
                self._remove_daily(target_date)
                result.success = True
                logger.info("Summary for {} has no high-value memory; daily file skipped", target_date)
                return result

            daily_content = self._render_daily(target_date, summary)

            if len(daily_content) > config.memory_daily_max_chars:
                daily_content = await self._compact_daily(target_date, daily_content)
                result.compacted = True

            self._write_daily(target_date, daily_content)
            result.daily_chars = len(daily_content)

            candidates = self._clean_items(summary.long_term_candidates)
            if candidates:
                self._append_pending(target_date, candidates)
            result.pending_candidates = len(candidates)

            if update_index:
                from app.services.memory_index_service import memory_index_service

                index_result = memory_index_service.index_memory(rebuild=rebuild_index)
                result.indexed = index_result.success
                result.indexed_sources = index_result.indexed_sources
                if not index_result.success:
                    result.error_message = index_result.error_message
                    result.success = False
                    return result

            result.success = True
            return result
        except Exception as exc:
            logger.error("Memory summary failed: {}", exc, exc_info=True)
            result.success = False
            result.error_message = str(exc)
            return result

    def _load_records_for_date(self, target_date: date) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not memory_service.sessions_dir.exists():
            return records

        for path in sorted(memory_service.sessions_dir.glob("*.jsonl")):
            records.extend(self._load_records_from_file(path, target_date))

        records.sort(key=lambda item: str(item.get("timestamp") or ""))
        return records

    def _load_records_from_file(self, path: Path, target_date: date) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid memory JSON: {}:{}", path, line_no)
                    continue

                timestamp = str(record.get("timestamp") or "")
                if self._local_date(timestamp) != target_date:
                    continue
                role = str(record.get("role") or "")
                content = str(record.get("content") or "").strip()
                if role not in {"user", "assistant"} or not content:
                    continue
                records.append(record)
        except Exception as exc:
            logger.warning("Failed to read memory file {}: {}", path, exc)
        return records

    @staticmethod
    def _local_date(timestamp: str) -> date | None:
        try:
            parsed = datetime.fromisoformat(timestamp)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=LOCAL_TZ)
            return parsed.astimezone(LOCAL_TZ).date()
        except Exception:
            return None

    def _format_records(self, records: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for record in records:
            timestamp = str(record.get("timestamp") or "")
            session_id = str(record.get("session_id") or "")
            role = str(record.get("role") or "")
            content = str(record.get("content") or "").strip()
            if self._is_low_value_message(content):
                continue
            lines.append(
                f"[{timestamp}] session={session_id} role={role}\n{content}"
            )

        transcript = "\n\n".join(lines).strip()
        max_chars = max(config.memory_summary_input_max_chars, 1000)
        if len(transcript) <= max_chars:
            return transcript
        return transcript[-max_chars:]

    @staticmethod
    def _is_low_value_message(content: str) -> bool:
        text = content.strip()
        if not text:
            return True
        normalized = re.sub(r"\s+", "", text.lower())
        low_value = {
            "你好",
            "您好",
            "hi",
            "hello",
            "谢谢",
            "好的",
            "ok",
            "嗯",
        }
        if normalized in low_value:
            return True
        if normalized.startswith("你好") and len(normalized) <= 12:
            return True
        if "小助手" in normalized and len(normalized) <= 16:
            return True

        generic_assistant_markers = [
            "我是你的智能助手",
            "我是你的ai助手",
            "我是专业的ai助手",
            "很高兴为你服务",
            "很高兴为您服务",
            "有什么可以帮你",
            "有什么可以帮您",
            "我可以帮助你",
            "我可以帮助您",
            "欢迎回来",
        ]
        if any(marker in normalized for marker in generic_assistant_markers):
            return True

        high_value_markers = [
            "记忆",
            "memory",
            "rag",
            "milvus",
            "fastapi",
            "agent",
            "jsonl",
            "daily",
            "pending",
            "memory.md",
            "接口",
            "模型",
            "多模态",
            "向量",
            "索引",
            "方案",
            "决策",
            "配置",
            "路径",
            "代码",
        ]
        has_high_value_marker = any(marker in normalized for marker in high_value_markers)
        if not has_high_value_marker and any(
            marker in normalized
            for marker in [
                "星期",
                "周几",
                "今天是",
                "昨天是",
                "明天是",
                "当前时间",
                "北京时间",
                "具体日期",
            ]
        ):
            return True
        if not has_high_value_marker and any(
            marker in normalized
            for marker in [
                "简单的加法",
                "简单加法",
                "基础的数学",
                "数学运算",
                "结果是",
                "答案是",
            ]
        ):
            return True

        low_value_patterns = [
            r"^(请问)?今天(是)?(星期|周)几[？?。.!！]*$",
            r"^(今天|昨天|明天)(是)?(星期|周)几[？?。.!！]*$",
            r"^今天(的)?日期(是)?多少[？?。.!！]*$",
            r"^现在几点[？?。.!！]*$",
            r"^\d+\s*[+\-*/÷×]\s*\d+\s*=?\s*[？?。.!！]*$",
            r"^\d+\s*[+\-*/÷×]\s*\d+\s*=?\s*几[？?。.!！]*$",
            r"^\d+\s*[+\-*/÷×]\s*\d+\s*=\s*\d+\s*[？?。.!！]*$",
            r"^\d+\s*[+\-*/÷×]\s*\d+\s*=\s*\d+.*$",
        ]
        return any(re.match(pattern, normalized) for pattern in low_value_patterns)

    async def _summarize_transcript(
        self,
        target_date: date,
        transcript: str,
    ) -> DailyMemorySummary:
        llm = self._build_llm()
        structured_llm = llm.with_structured_output(DailyMemorySummary)
        messages = [
            SystemMessage(content=self._summary_system_prompt()),
            HumanMessage(
                content=dedent(f"""
                    日期：{target_date.isoformat()}

                    以下是当天原始聊天记录。请只提取对后续有价值的信息。

                    {transcript}
                """).strip()
            ),
        ]
        result = await structured_llm.ainvoke(messages)
        if isinstance(result, DailyMemorySummary):
            return result
        return DailyMemorySummary.model_validate(result)

    async def _compact_daily(self, target_date: date, daily_content: str) -> str:
        llm = self._build_llm()
        structured_llm = llm.with_structured_output(CompactedDailyMemory)
        messages = [
            SystemMessage(content=self._compact_system_prompt()),
            HumanMessage(
                content=dedent(f"""
                    日期：{target_date.isoformat()}

                    当前 daily 内容超过长度阈值，请压缩：

                    {daily_content}
                """).strip()
            ),
        ]
        result = await structured_llm.ainvoke(messages)
        if not isinstance(result, CompactedDailyMemory):
            result = CompactedDailyMemory.model_validate(result)
        content = result.content.strip()
        return content or daily_content[: config.memory_daily_max_chars]

    @staticmethod
    def _summary_system_prompt() -> str:
        return dedent("""
            你是记忆系统的整理器。你的任务是把原始聊天流水整理成可复用的 daily memory。

            请遵守：
            - 只保留对未来有价值的信息。
            - 保留用户偏好、项目背景、技术决策、接口/路径/配置、未完成任务、风险点。
            - 删除普通问候、简单计算、临时闲聊、重复内容、明显错误的中间尝试。
            - 删除日期查询、星期几查询、当前时间查询、简单数学计算。
            - 如果当天只有低价值内容，请返回所有字段为空列表。
            - 不要编造聊天中没有出现的信息。
            - long_term_candidates 只放适合进入 MEMORY.md 的稳定长期记忆候选。
        """).strip()

    @staticmethod
    def _compact_system_prompt() -> str:
        return dedent("""
            你是记忆压缩器。请把 daily memory 压缩成更短的 Markdown。

            要求：
            - 保留重要结论、用户偏好、项目决策、待办和风险。
            - 删除流水、重复、闲聊和临时调试噪声。
            - 不要引入原文没有的信息。
        """).strip()

    def _render_daily(self, target_date: date, summary: DailyMemorySummary) -> str:
        sections = [
            f"# {target_date.isoformat()}",
            self._render_section("Summary", summary.summary_items),
            self._render_section("Decisions", summary.decisions),
            self._render_section("Todos", summary.todos),
            self._render_section("Risks", summary.risks),
        ]
        return "\n\n".join(section for section in sections if section.strip()).strip() + "\n"

    def _filter_summary(self, summary: DailyMemorySummary) -> DailyMemorySummary:
        return DailyMemorySummary(
            summary_items=self._filter_high_value_items(summary.summary_items),
            decisions=self._filter_high_value_items(summary.decisions),
            todos=self._filter_high_value_items(summary.todos),
            risks=self._filter_high_value_items(summary.risks),
            long_term_candidates=self._filter_high_value_items(summary.long_term_candidates),
        )

    @staticmethod
    def _has_meaningful_summary(summary: DailyMemorySummary) -> bool:
        return any(
            [
                summary.summary_items,
                summary.decisions,
                summary.todos,
                summary.risks,
                summary.long_term_candidates,
            ]
        )

    def _filter_high_value_items(self, items: list[str]) -> list[str]:
        return [
            item
            for item in self._clean_items(items)
            if not self._is_low_value_summary_item(item)
        ]

    def _is_low_value_summary_item(self, item: str) -> bool:
        text = item.strip()
        if not text:
            return True
        low_value_markers = [
            "星期",
            "日期",
            "几点",
            "当前时间",
            "简单数学",
            "简单计算",
            "1+1",
            "1+2",
            "1+3",
            "1+4",
            "算术",
        ]
        normalized = re.sub(r"\s+", "", text.lower())
        if any(marker in text for marker in low_value_markers):
            return True
        if re.search(r"\d+\s*[+\-*/÷×]\s*\d+", normalized):
            return True
        return False

    def _render_section(self, title: str, items: list[str]) -> str:
        cleaned = self._clean_items(items)
        if not cleaned:
            return f"## {title}\n\n- None"
        lines = [f"## {title}"]
        lines.extend(f"- {item}" for item in cleaned)
        return "\n".join(lines)

    @staticmethod
    def _clean_items(items: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text or text.lower() == "none":
                continue
            if text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        return cleaned

    def _write_daily(self, target_date: date, content: str) -> None:
        memory_service.daily_dir.mkdir(parents=True, exist_ok=True)
        self._daily_file(target_date).write_text(content, encoding="utf-8")

    def _remove_daily(self, target_date: date) -> None:
        daily_file = self._daily_file(target_date)
        if daily_file.exists():
            daily_file.unlink()

    def _append_pending(self, target_date: date, candidates: list[str]) -> None:
        memory_service.memory_root.mkdir(parents=True, exist_ok=True)
        pending_file = memory_service.memory_pending_file
        existing = pending_file.read_text(encoding="utf-8") if pending_file.exists() else "# MEMORY_PENDING\n"
        existing_items = set(existing.splitlines())
        lines: list[str] = []
        header = f"## {target_date.isoformat()}"
        if header not in existing:
            lines.extend(["", header])
        for candidate in candidates:
            line = f"- [ ] {candidate}"
            if line not in existing_items:
                lines.append(line)
        if lines:
            with pending_file.open("a", encoding="utf-8") as file:
                file.write("\n".join(lines).rstrip() + "\n")

    @staticmethod
    def _daily_file(target_date: date) -> Path:
        return memory_service.daily_dir / f"{target_date.isoformat()}.md"

    @staticmethod
    def _to_project_relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _build_llm() -> ChatQwen:
        model_name = config.memory_summary_model or config.rag_model
        return ChatQwen(
            model=model_name,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_api_base,
            temperature=0,
            timeout=60,
        )


memory_summary_service = MemorySummaryService()
