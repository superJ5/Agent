"""Per-request runtime overrides."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.config import config


_memory_enabled_override: ContextVar[bool | None] = ContextVar(
    "memory_enabled_override",
    default=None,
)


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
