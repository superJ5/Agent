"""Evaluation-only failed structured outputs retain their unparsed response."""

from __future__ import annotations

import asyncio
import json
import stat

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ValidationError

from app.core.request_context import current_structured_trace_path, structured_output_trace
from app.services.structured_output_diagnostics import invoke_structured_output


class ExampleUpdate(BaseModel):
    should_update: bool


class FakeStructuredCall:
    def __init__(self, result):
        self.result = result

    async def ainvoke(self, messages):
        return self.result


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.include_raw = None

    def with_structured_output(self, schema, *, include_raw=False):
        self.include_raw = include_raw
        return FakeStructuredCall(self.result)


def test_failure_saves_raw_response_only_in_eval_scope(tmp_path):
    path = tmp_path / "traces" / "run.jsonl"
    raw = AIMessage(
        content="not valid structured output",
        response_metadata={"finish_reason": "stop"},
    )
    model = FakeModel({"raw": raw, "parsed": None, "parsing_error": None})

    with structured_output_trace(path):
        assert current_structured_trace_path() == path
        with pytest.raises(ValidationError):
            asyncio.run(invoke_structured_output(
                model, ExampleUpdate, ["secret prompt"], session_id="eval-session",
            ))

    assert current_structured_trace_path() is None
    assert model.include_raw is True
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["stage"] == "structured_parsing"
    assert record["session_id"] == "eval-session"
    assert record["raw_response"]["content"] == "not valid structured output"
    assert record["raw_response"]["response_metadata"]["finish_reason"] == "stop"
    assert record["parsed_is_none"] is True
    assert "secret prompt" not in path.read_text(encoding="utf-8")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_success_does_not_write_raw_response(tmp_path):
    path = tmp_path / "run.jsonl"
    model = FakeModel({
        "raw": AIMessage(content="ok"),
        "parsed": ExampleUpdate(should_update=False),
        "parsing_error": None,
    })
    with structured_output_trace(path):
        result = asyncio.run(invoke_structured_output(
            model, ExampleUpdate, [], session_id="eval-session",
        ))
    assert result.should_update is False
    assert not path.exists()


def test_normal_business_path_does_not_request_raw_response():
    model = FakeModel(ExampleUpdate(should_update=True))
    result = asyncio.run(invoke_structured_output(
        model, ExampleUpdate, [], session_id="normal-session",
    ))
    assert result.should_update is True
    assert model.include_raw is False
