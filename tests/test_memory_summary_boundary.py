"""Step 1: raw-message sequences and explicit summary boundaries."""

from __future__ import annotations

import json

import pytest

from app.config import config
from app.services import (
    memory_service as memory_module,
    short_term_memory_service as summary_module,
)


@pytest.fixture
def memory_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_write_enabled", True)
    monkeypatch.setattr(memory_module, "is_memory_enabled", lambda: True)
    monkeypatch.setattr(summary_module, "is_memory_enabled", lambda: True)
    monkeypatch.setattr(memory_module.memory_service, "sessions_dir", tmp_path / "sessions")
    monkeypatch.setattr(summary_module.short_term_memory_service, "short_term_dir", tmp_path / "short_term")
    return memory_module.memory_service, summary_module.short_term_memory_service


def test_message_sequence_includes_failed_and_unanswered_events(memory_dirs):
    raw, summary = memory_dirs
    session_id = "old-session"
    raw.append_message(session_id, "user", "未得到回答的问题")
    raw.append_message(session_id, "user", "第一个问题")
    raw.append_message(session_id, "assistant", "第一个答案")
    raw.append_message(session_id, "user", "失败的问题")
    raw.append_message(session_id, "assistant", "误导性的部分输出", metadata={"status": "error", "error": "连接失败"})
    raw.append_message(session_id, "user", "第二个问题")
    raw.append_message(session_id, "assistant", "第二个答案", metadata={"status": "success"})
    raw.append_message(session_id, "user", "尚未回答")
    raw.append_message(session_id, "assistant", "", metadata={"status": "error", "error": "连接失败"})

    events = raw.load_message_events(session_id)
    assert [event["message_seq"] for event in events] == list(range(1, 10))
    raw_lines = raw._session_file(session_id).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["message_seq"] for line in raw_lines] == list(range(1, 10))
    assert events[4]["content"] == "误导性的部分输出"
    assert events[8]["content"] == ""

    recent = raw.load_recent_messages(session_id, limit=8)
    assert "误导性的部分输出" not in [item["content"] for item in recent]
    context = summary.format_dialogue(raw.load_recent_messages(session_id, limit=8, include_failed=True))
    assert "失败的问题" in context
    assert "本次回答失败" in context
    assert "误导性的部分输出" not in context
    assert "连接失败" not in context


def test_summary_boundary_is_explicit_and_legacy_summary_is_unknown(memory_dirs):
    raw, summary = memory_dirs
    session_id = "summary-session"
    raw.append_message(session_id, "user", "问题一")
    raw.append_message(session_id, "assistant", "回答一")
    raw.append_message(session_id, "user", "问题二")
    raw.append_message(session_id, "assistant", "回答二")
    raw.append_message(session_id, "user", "未得到回答的事实")
    raw.append_message(session_id, "assistant", "", metadata={"status": "error"})

    summary._write_memory(session_id, "旧版滚动摘要")
    assert summary.load_summary_through_message_seq(session_id) is None
    assert summary.load_memory(session_id) == ""
    no_summary, raw_context = summary.load_answer_context(session_id)
    assert no_summary == ""
    assert [event["message_seq"] for event in raw_context] == [1, 2, 3, 4, 5, 6]

    summary.save_summary_with_boundary(session_id, "明确覆盖前两条消息", through_message_seq=2)
    assert summary.load_summary_through_message_seq(session_id) == 2
    assert "明确覆盖前两条消息" in summary.load_memory(session_id)
    assert len(raw.load_message_events(session_id)) == 6
    compacted, tail = summary.load_answer_context(session_id)
    assert "明确覆盖前两条消息" in compacted
    assert [event["message_seq"] for event in tail] == [3, 4, 5, 6]

    with pytest.raises(ValueError):
        summary.save_summary_with_boundary(session_id, "越界", through_message_seq=7)
    assert summary.load_summary_through_message_seq(session_id) == 2

    summary.save_summary_with_boundary(session_id, "包含失败的用户消息", through_message_seq=5)
    with pytest.raises(ValueError):
        summary.save_summary_with_boundary(session_id, "错误倒退", through_message_seq=1)
    assert summary.load_summary_through_message_seq(session_id) == 5
    _, tail = summary.load_answer_context(session_id)
    assert [event["message_seq"] for event in tail] == [6]


def test_current_question_is_not_duplicated_in_answer_context(memory_dirs):
    raw, summary = memory_dirs
    session_id = "current-question"
    raw.append_message(session_id, "user", "之前的问题")
    raw.append_message(session_id, "assistant", "之前的回答")
    raw.append_message(session_id, "user", "现在的问题")

    _, dialogue = summary.load_answer_context(session_id, current_question="现在的问题")
    assert [event["content"] for event in dialogue] == ["之前的问题", "之前的回答"]


def test_compaction_keeps_latest_three_user_turns_and_only_new_prefix(memory_dirs):
    raw, summary = memory_dirs
    session_id = "compact-session"
    for n in range(1, 6):
        raw.append_message(session_id, "user", f"问题{n}")
        raw.append_message(session_id, "assistant", f"回答{n}")
    raw.append_message(session_id, "user", "当前问题")

    plan = summary.prepare_compaction(session_id, current_question="当前问题")
    assert plan is not None
    assert plan.old_summary == ""
    assert [event["message_seq"] for event in plan.messages_to_summarize] == [1, 2, 3, 4]
    assert plan.through_message_seq == 4
    summary.save_summary_with_boundary(session_id, "前两轮的摘要", through_message_seq=4)
    compacted, tail = summary.load_answer_context(session_id, current_question="当前问题")
    assert compacted == "前两轮的摘要"
    assert [event["message_seq"] for event in tail] == [5, 6, 7, 8, 9, 10]
    assert summary.prepare_compaction(session_id, current_question="当前问题") is None

    raw.append_message(session_id, "assistant", "当前回答")
    raw.append_message(session_id, "user", "下一个问题")
    next_plan = summary.prepare_compaction(session_id, current_question="下一个问题")
    assert next_plan is not None
    assert next_plan.old_summary == "前两轮的摘要"
    assert [event["message_seq"] for event in next_plan.messages_to_summarize] == [5, 6]
    assert next_plan.through_message_seq == 6


def test_compaction_preserves_failed_attempt_as_raw_tail(memory_dirs):
    raw, summary = memory_dirs
    session_id = "failed-tail"
    for n in range(1, 3):
        raw.append_message(session_id, "user", f"问题{n}")
        raw.append_message(session_id, "assistant", f"回答{n}")
    raw.append_message(session_id, "user", "失败问题")
    raw.append_message(session_id, "assistant", "错误片段", metadata={"status": "error"})
    raw.append_message(session_id, "user", "问题4")
    raw.append_message(session_id, "assistant", "回答4")
    raw.append_message(session_id, "user", "问题5")
    raw.append_message(session_id, "assistant", "回答5")
    plan = summary.prepare_compaction(session_id)
    assert plan is not None and plan.through_message_seq == 4
    _, tail = summary.load_answer_context(session_id)
    formatted = summary.format_dialogue(tail)
    assert "失败问题" in formatted
    assert "错误片段" not in formatted
    assert "本次回答失败" in formatted


def test_oversized_summary_does_not_advance_boundary(memory_dirs):
    raw, summary = memory_dirs
    raw.append_message("oversized", "user", "问题")
    with pytest.raises(ValueError, match="不允许截断"):
        summary.save_summary_with_boundary("oversized", "内容" * 5000, through_message_seq=1)
    assert summary.load_summary_through_message_seq("oversized") is None
