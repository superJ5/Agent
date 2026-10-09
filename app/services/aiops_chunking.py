"""Pure Markdown chunking helpers for AIOps runbooks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _split_oversized_section(text: str, max_chars: int) -> list[str]:
    """Split an oversized Markdown section at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start : start + max_chars])
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def chunk_markdown_document(
    content: str,
    source_file: str,
    *,
    max_chars: int = 1400,
) -> list[dict[str, Any]]:
    """Split Markdown by headings, then split oversized sections by paragraph."""
    lines = content.splitlines()
    document_title = Path(source_file).stem
    heading_stack: list[str] = []
    sections: list[tuple[list[str], list[str]]] = []
    section_lines: list[str] = []
    section_path: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal section_lines
        text = "\n".join(section_lines).strip()
        has_body = any(line.strip() and not HEADING_RE.match(line) for line in section_lines)
        if text and has_body:
            sections.append((list(section_path), list(section_lines)))
        section_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence

        match = None if in_fence else HEADING_RE.match(line)
        if match:
            flush()
            depth = len(match.group(1))
            title = match.group(2).strip()
            if depth == 1:
                document_title = title
            heading_stack = heading_stack[: depth - 1]
            while len(heading_stack) < depth - 1:
                heading_stack.append("")
            heading_stack.append(title)
            section_path = [part for part in heading_stack if part]
        section_lines.append(line)
    flush()

    doc_hash = hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:10]
    records: list[dict[str, Any]] = []
    chunk_number = 0
    for path, raw_lines in sections:
        section_text = "\n".join(raw_lines).strip()
        hierarchy = " > ".join(path) if path else document_title
        prefix = f"文档：{document_title}\n章节：{hierarchy}\n\n"
        available_chars = max(200, max_chars - len(prefix))
        for part in _split_oversized_section(section_text, available_chars):
            chunk_number += 1
            records.append(
                {
                    "chunk_id": f"aiops_{doc_hash}_{chunk_number:04d}",
                    "doc_id": f"aiops_{doc_hash}",
                    "doc_name": document_title,
                    "source_file": source_file,
                    "section_path": path,
                    "title": path[-1] if path else document_title,
                    "chunk_type": "runbook_section",
                    "knowledge_domain": "aiops",
                    "text": f"{prefix}{part}".strip(),
                }
            )
    return records
