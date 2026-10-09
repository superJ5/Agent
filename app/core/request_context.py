"""Per-request runtime overrides."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from app.config import config

_memory_enabled_override: ContextVar[bool | None] = ContextVar(
    "memory_enabled_override",
    default=None,
)
_memory_user_id: ContextVar[str] = ContextVar("memory_user_id", default="default")
_structured_trace_path: ContextVar[Path | None] = ContextVar(
    "structured_trace_path", default=None,
)


def current_memory_user_id() -> str:
    """Return the trusted internal identity for the current evaluation/request."""
    return _memory_user_id.get()


def current_structured_trace_path() -> Path | None:
    """Return an opt-in trace destination for evaluation-only model failures."""
    return _structured_trace_path.get()


@contextmanager
def structured_output_trace(path: Path) -> Iterator[None]:
    """Capture unparsed structured outputs only within this evaluation scope."""
    token = _structured_trace_path.set(path)
    try:
        yield
    finally:
        _structured_trace_path.reset(token)


@contextmanager
def memory_user_identity(user_id: str) -> Iterator[None]:
    """Scope all memory operations to one internally assigned user identity."""
    normalized = str(user_id or "").strip()
    if not normalized:
        raise ValueError("memory user_id 不能为空")
    token = _memory_user_id.set(normalized)
    try:
        yield
    finally:
        _memory_user_id.reset(token)


def is_memory_enabled() -> bool:
    """Return the effective memory flag for the current request."""
    override = _memory_enabled_override.get()
    if override is not None:
        return override
    return bool(getattr(config, "memory_enabled", True))


@contextmanager
def memory_enabled_override(value: bool | None) -> Iterator[None]:
    """Temporarily override memory behavior for one request."""
    if value is None:
        yield
        return

    token = _memory_enabled_override.set(value)
    try:
        yield
    finally:
        _memory_enabled_override.reset(token)
