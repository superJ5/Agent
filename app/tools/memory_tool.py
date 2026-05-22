"""Memory search tool for Agent-visible daily memory."""

from __future__ import annotations

from langchain_core.tools import tool
from loguru import logger

from app.services.memory_service import memory_service
from app.services.memory_vector_store import memory_vector_store


@tool
def memory_search(query: str) -> str:
    """Search curated daily memory notes."""
    try:
        try:
            vector_hits = memory_vector_store.search(query)
        except Exception as exc:
            logger.warning("向量记忆检索失败，回退关键词检索: {}", exc)
            vector_hits = []

        if vector_hits:
            return _format_vector_hits(vector_hits)

        hits = memory_service.search_memory(
            query,
            limit=5,
        )
        if not hits:
            return "没有找到相关历史记忆。"

        sections: list[str] = ["【历史记忆检索结果】"]
        for index, hit in enumerate(hits, 1):
            location = hit.get("path", "")
            line = hit.get("line")
            role = hit.get("role")
            timestamp = hit.get("timestamp")
            content = hit.get("content", "")
            header = f"【记忆 {index}】来源: {hit.get('source_type', '')}"
            if location:
                header += f"；位置: {location}"
            if line:
                header += f":{line}"
            if role:
                header += f"；角色: {role}"
            if timestamp:
                header += f"；时间: {timestamp}"
            sections.append(f"{header}\n{content}")
        return "\n\n".join(sections)
    except Exception as exc:
        logger.error(f"历史记忆检索失败: {exc}")
        return f"历史记忆检索失败: {exc}"


def _format_vector_hits(hits: list[dict]) -> str:
    sections: list[str] = ["【历史记忆向量检索结果】"]
    for index, hit in enumerate(hits, 1):
        metadata = hit.get("metadata") or {}
        source_path = metadata.get("source_path", "")
        line = metadata.get("line")
        memory_type = metadata.get("memory_type", "")
        header = f"【记忆 {index}】类型: {memory_type}"
        if source_path:
            header += f"；来源: {source_path}"
        if line:
            header += f":{line}"
        header += f"；距离: {hit.get('score')}"
        sections.append(f"{header}\n{hit.get('content', '')}")
    return "\n\n".join(sections)
