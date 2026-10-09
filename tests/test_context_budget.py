"""Pre-call context budget estimates are conservative and configurable."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.context_budget import (
    assess_context_budget,
    estimate_messages_tokens,
    estimate_text_tokens,
    extract_model_usage,
)


def test_estimator_counts_chinese_text_and_message_overhead():
    messages = [SimpleNamespace(content="设备异常"), SimpleNamespace(content="检查端口")]
    assert estimate_text_tokens("设备异常") == 4
    assert estimate_messages_tokens(messages) == 16


def test_budget_reserves_output_retrieval_and_safety_space():
    messages = [SimpleNamespace(content="a" * 120)]
    budget = assess_context_budget(
        messages,
        working_window_tokens=100,
        output_reserve_tokens=10,
        retrieval_reserve_tokens=20,
        safety_margin_tokens=10,
        compact_ratio=0.7,
    )
    assert budget.available_input_tokens == 60
    assert budget.compact_trigger_tokens == 42
    assert budget.estimated_input_tokens == 44
    assert budget.should_compact is True
    assert budget.over_budget is False


def test_non_text_block_uses_reserve_instead_of_base64_length():
    messages = [SimpleNamespace(content=[{"type": "image_url", "image_url": "data:image/png;base64," + "A" * 50000}])]
    assert estimate_messages_tokens(messages) == 1028


def test_invalid_budget_is_rejected():
    with pytest.raises(ValueError, match="不得占满"):
        assess_context_budget(
            [],
            working_window_tokens=100,
            output_reserve_tokens=60,
            retrieval_reserve_tokens=40,
            safety_margin_tokens=0,
            compact_ratio=0.8,
        )


def test_actual_usage_accepts_langchain_and_provider_shapes():
    langchain = SimpleNamespace(usage_metadata={"input_tokens": 12, "output_tokens": 4})
    provider = SimpleNamespace(response_metadata={"token_usage": {"prompt_tokens": 18, "completion_tokens": 5}})
    assert extract_model_usage(langchain).input_tokens == 12
    assert extract_model_usage(provider).output_tokens == 5
    assert extract_model_usage(SimpleNamespace(content="no usage")) is None
