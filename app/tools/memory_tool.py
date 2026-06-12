"""Legacy memory search tool.

The old daily/MEMORY.md memory path is intentionally disabled while the new
short-term/session-state/long-term memory system is introduced.
"""

from __future__ import annotations

from langchain_core.tools import tool

@tool
def memory_search(query: str) -> str:
    """Return an explicit disabled response for legacy memory search."""
    return "旧版 daily/MEMORY.md 记忆检索已停用；当前只使用会话上下文。"
