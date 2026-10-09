"""Conservative, pre-call context budget estimates for session compaction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

MESSAGE_OVERHEAD_TOKENS = 4
NON_TEXT_BLOCK_RESERVE_TOKENS = 1024


@dataclass(frozen=True)
class ContextBudgetEstimate:
    estimated_input_tokens: int
    available_input_tokens: int
    compact_trigger_tokens: int
    should_compact: bool
    over_budget: bool


@dataclass(frozen=True)
class ModelTokenUsage:
    input_tokens: int
    output_tokens: int


def estimate_text_tokens(text: str) -> int:
    """Estimate text tokens conservatively from UTF-8 bytes, not model-exact."""
    return ceil(len(text.encode("utf-8")) / 3) if text else 0


def estimate_messages_tokens(messages: Sequence[Any]) -> int:
    """Estimate the currently assembled prompt, including its message overhead."""
    total = 0
    for message in messages:
        content = getattr(message, "content", message)
        total += MESSAGE_OVERHEAD_TOKENS + _estimate_content_tokens(content)
    return total


def assess_context_budget(
    messages: Sequence[Any],
    *,
    working_window_tokens: int,
    output_reserve_tokens: int,
    retrieval_reserve_tokens: int,
    safety_margin_tokens: int,
    compact_ratio: float,
) -> ContextBudgetEstimate:
    """Decide whether to compact before an answer-model call.

    The retrieval reserve covers tool/RAG text that may be added after this
    check. The trigger starts compaction; the available-input limit is a
    separate hard stop before the answer model is called.
    """
    if working_window_tokens <= 0 or not 0 < compact_ratio <= 1:
        raise ValueError("working_window_tokens 和 compact_ratio 配置无效")
    if min(output_reserve_tokens, retrieval_reserve_tokens, safety_margin_tokens) < 0:
        raise ValueError("预算预留不能为负数")
    available = (
        working_window_tokens
        - output_reserve_tokens
        - retrieval_reserve_tokens
        - safety_margin_tokens
    )
    if available <= 0:
        raise ValueError("预留 token 不得占满工作窗口")
    estimated = estimate_messages_tokens(messages)
    trigger = max(1, int(available * compact_ratio))
    return ContextBudgetEstimate(
        estimated_input_tokens=estimated,
        available_input_tokens=available,
        compact_trigger_tokens=trigger,
        should_compact=estimated >= trigger,
        over_budget=estimated >= available,
    )


def extract_model_usage(message: Any) -> ModelTokenUsage | None:
    """Read post-call token usage when the provider exposes it."""
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, Mapping):
        metadata = getattr(message, "response_metadata", None)
        usage = metadata.get("token_usage") if isinstance(metadata, Mapping) else None
    if not isinstance(usage, Mapping):
        return None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    return ModelTokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _estimate_content_tokens(content: Any) -> int:
    if isinstance(content, str):
        return estimate_text_tokens(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    total += estimate_text_tokens(text)
                else:
                    total += NON_TEXT_BLOCK_RESERVE_TOKENS
            elif isinstance(block, str):
                total += estimate_text_tokens(block)
            else:
                total += NON_TEXT_BLOCK_RESERVE_TOKENS
        return total
    return estimate_text_tokens(str(content))
