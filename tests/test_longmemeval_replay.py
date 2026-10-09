"""The benchmark runner must import history, not regenerate historical answers."""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, SystemMessage

from app.config import config
from app.core.request_context import current_memory_user_id, current_structured_trace_path
from app.evaluation.longmemeval_replay import LongMemEvalReplayer, _session_timestamp


class FakeRaw:
    def __init__(self):
        self.events = []

    def append_message(self, session_id, role, content, metadata=None, timestamp=None):
        self.events.append((current_memory_user_id(), session_id, role, content, metadata, timestamp))


class FakeLong:
    def __init__(self):
        self.updates = []

    async def update_after_turn(self, **kwargs):
        self.updates.append(kwargs)


class FakeState:
    def __init__(self):
        self.updates = []
        self.trace_paths = []

    async def update_after_turn(self, **kwargs):
        self.updates.append(kwargs)
        self.trace_paths.append(current_structured_trace_path())


class FakeModel:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="recalled answer")


class FakeAgent:
    def __init__(self):
        self.model = FakeModel()
        self.lookup_users = []
        self.compaction_calls = 0

    def _build_short_term_context_messages(self, session_id, question):
        return [], []

    def _build_session_state_context_messages(self, session_id):
        return []

    def _build_long_term_context_messages(self, *, session_id, question, **kwargs):
        self.lookup_users.append(current_memory_user_id())
        return [SystemMessage(content="stored user fact")], []

    async def _compact_if_needed(self, *args, **kwargs):
        self.compaction_calls += 1
        return False

    def _ensure_context_budget(self, session_id, messages):
        return None

    def _model_for_request(self):
        return self.model

    def _message_text(self, response):
        return response.content


def _case():
    return {
        "question_id": "test-1",
        "question": "What is the user fact?",
        "question_date": "2024/03/03 (Sun) 10:20",
        "answer": "SECRET GOLD ANSWER",
        "haystack_session_ids": ["session-a", "session-b"],
        "haystack_dates": ["2024/01/01 (Mon) 08:00", "2024/02/02 (Fri) 09:10"],
        "haystack_sessions": [
            [{"role": "user", "content": "Remember my fact."},
             {"role": "assistant", "content": "I will remember it."}],
            [{"role": "user", "content": "Here is another fact."},
             {"role": "assistant", "content": "I understand."}],
        ],
    }


def test_replay_imports_verbatim_history_and_isolates_runs(monkeypatch):
    monkeypatch.setattr(config, "memory_write_enabled", True)
    replayer = LongMemEvalReplayer()
    replayer.raw = FakeRaw()
    replayer.long = FakeLong()
    replayer.state = FakeState()
    replayer.agent = FakeAgent()

    first = asyncio.run(replayer.replay_case(_case(), run_id="run-a"))
    second = asyncio.run(replayer.replay_case(_case(), run_id="run-b"))

    assert first.hypothesis == "recalled answer"
    assert first.completed_turn_count == 2
    assert first.session_count == 2
    assert first.user_id != second.user_id
    assert {row[0] for row in replayer.raw.events} == {first.user_id, second.user_id}
    assert len(replayer.long.updates) == len(replayer.state.updates) == 4
    assert replayer.long.updates[0]["assistant_message"] == "I will remember it."
    assert replayer.long.updates[0]["user_id"] == first.user_id
    assert replayer.raw.events[0][5] == "2024-01-01T08:00:00"
    assert replayer.raw.events[0][4]["source_date"] == "2024/01/01 (Mon) 08:00"
    assert all("run-a" in row[1] for row in replayer.raw.events[:6])
    assert all("run-b" in row[1] for row in replayer.raw.events[6:])
    assert set(replayer.agent.lookup_users) == {first.user_id, second.user_id}
    assert all(path is not None for path in replayer.state.trace_paths)
    assert {path.name for path in replayer.state.trace_paths} == {"run-a.jsonl", "run-b.jsonl"}
    assert current_structured_trace_path() is None
    assert replayer.agent.compaction_calls == 4
    final_prompt = "\n".join(str(m.content) for m in replayer.agent.model.calls[0])
    assert "stored user fact" in final_prompt
    assert "2024/03/03" in final_prompt
    assert "SECRET GOLD ANSWER" not in final_prompt


def test_replay_rejects_disabled_writes(monkeypatch):
    monkeypatch.setattr(config, "memory_write_enabled", False)
    replayer = LongMemEvalReplayer()
    try:
        asyncio.run(replayer.replay_case(_case(), run_id="run-a"))
    except RuntimeError as exc:
        assert "MEMORY_WRITE_ENABLED" in str(exc)
    else:
        raise AssertionError("Expected replay to reject disabled writes")


def test_timestamp():
    assert _session_timestamp("2024/03/03 (Sun) 10:20") == "2024-03-03T10:20:00"
