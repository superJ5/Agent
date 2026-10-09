"""Trusted evaluation identities isolate session files and long-term actions."""

from __future__ import annotations

import asyncio

from app.config import config
from app.core.request_context import (
    current_memory_user_id,
    memory_enabled_override,
    memory_user_identity,
)
from app.services import (
    memory_service as raw_module,
    session_state_service as state_module,
    short_term_memory_service as summary_module,
)
from app.services.long_term_memory_service import (
    LongTermMemory,
    LongTermMemoryAction,
    LongTermMemoryService,
)


def test_same_session_id_is_scoped_to_user_for_all_file_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "memory_write_enabled", True)
    raw = raw_module.memory_service
    summary = summary_module.short_term_memory_service
    state = state_module.session_state_service
    monkeypatch.setattr(raw, "sessions_dir", tmp_path / "sessions")
    monkeypatch.setattr(summary, "short_term_dir", tmp_path / "short_term")
    monkeypatch.setattr(state, "session_state_dir", tmp_path / "session_state")

    with memory_enabled_override(True), memory_user_identity("eval-question-a"):
        raw.append_message("shared-session", "user", "A 的事实")
        raw.append_message("shared-session", "assistant", "A 的回答")
        summary.save_summary_with_boundary("shared-session", "A 的摘要", through_message_seq=2)
        state._write_state(state_module.SessionState(session_id="shared-session", goal="A 的目标"))
        a_file = raw._session_file("shared-session")

    with memory_enabled_override(True), memory_user_identity("eval-question-b"):
        assert raw.load_message_events("shared-session") == []
        assert summary.load_memory("shared-session") == ""
        assert state.load_state("shared-session") is None
        raw.append_message("shared-session", "user", "B 的事实")
        b_file = raw._session_file("shared-session")
        assert [event["content"] for event in raw.load_message_events("shared-session")] == ["B 的事实"]

    assert a_file != b_file
    with memory_enabled_override(True), memory_user_identity("eval-question-a"):
        assert [event["content"] for event in raw.load_message_events("shared-session")] == [
            "A 的事实", "A 的回答",
        ]
        assert summary.load_memory("shared-session") == "A 的摘要"
        assert state.load_state("shared-session").goal == "A 的目标"
    assert current_memory_user_id() == "default"


def test_long_term_search_and_mutations_are_filtered_before_access(monkeypatch):
    service = LongTermMemoryService()
    expressions: list[str] = []

    class FakeCollection:
        def search(self, **kwargs):
            expressions.append(kwargs["expr"])
            return [[]]

        def query(self, **kwargs):
            expressions.append(kwargs["expr"])
            return []

        def delete(self, **kwargs):
            expressions.append(kwargs["expr"])

        def flush(self):
            return None

    monkeypatch.setattr(service, "ensure_collection", lambda: FakeCollection())
    monkeypatch.setattr(
        "app.services.long_term_memory_service.vector_embedding_service.embed_query",
        lambda query: [0.0],
    )
    with memory_enabled_override(True), memory_user_identity("eval-question-b"):
        assert service.retrieve_relevant_memories(question="学历") == []
        service._apply_action(
            LongTermMemoryAction(action="delete", target_memory_id="another-user-memory"),
            session_id="session-b", user_id=current_memory_user_id(),
        )
        service._apply_action(
            LongTermMemoryAction(
                action="update", target_memory_id="another-user-memory",
                memory=LongTermMemory(content="不能覆盖别人的记忆", confidence=0.9),
            ),
            session_id="session-b", user_id=current_memory_user_id(),
        )

    assert expressions[0] == 'user_id == "eval-question-b"'
    assert all('user_id == "eval-question-b"' in expr for expr in expressions)
    assert all("eval-question-a" not in expr for expr in expressions)


def test_background_task_keeps_identity_after_parent_scope_exits():
    async def check():
        async def read_identity():
            await asyncio.sleep(0)
            return current_memory_user_id()

        with memory_user_identity("eval-background"):
            task = asyncio.create_task(read_identity())
        assert current_memory_user_id() == "default"
        return await task

    assert asyncio.run(check()) == "eval-background"
