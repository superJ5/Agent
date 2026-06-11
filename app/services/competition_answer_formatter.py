"""Format generated answers for the competition response contract."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
MARKDOWN_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")
MARKDOWN_QUOTE_RE = re.compile(r"(?m)^[ \t]{0,3}>[ \t]?")
MARKDOWN_DIVIDER_RE = re.compile(
    r"(?m)^[ \t]{0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})[ \t]*\n?"
)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]+\)")
MARKDOWN_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__")
MARKDOWN_ITALIC_RE = re.compile(r"(?<!\*)\*(?=\S)(.+?)(?<=\S)\*(?!\*)")
EXCESS_BLANK_LINES_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
MARKDOWN_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def format_answer_images(answer: str) -> str:
    """Convert an answer to plain text with ordered ``<PIC>`` placeholders."""
    picture_ids: list[str] = []

    def replace_image(match: re.Match[str]) -> str:
        alt_text = match.group(1).strip()
        image_path = match.group(2).strip()
        picture_id = alt_text or _picture_id_from_path(image_path)
        if not picture_id:
            return match.group(0)

        picture_ids.append(picture_id)
        return "<PIC>"

    plain_answer = markdown_to_plain_text(answer)
    body = MARKDOWN_IMAGE_RE.sub(replace_image, plain_answer).strip()
    if not picture_ids:
        return body

    return (
        f"{json.dumps(body, ensure_ascii=False)},"
        f"{json.dumps(picture_ids, ensure_ascii=False)}"
    )


def markdown_to_plain_text(text: str) -> str:
    """Remove clear Markdown syntax without damaging ordinary punctuation."""
    text = markdown_tables_to_plain_text(text)
    text = MARKDOWN_DIVIDER_RE.sub("", text)
    text = MARKDOWN_HEADING_RE.sub("", text)
    text = MARKDOWN_QUOTE_RE.sub("", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = MARKDOWN_BOLD_RE.sub(lambda match: match.group(1) or match.group(2), text)
    text = MARKDOWN_ITALIC_RE.sub(r"\1", text)
    return EXCESS_BLANK_LINES_RE.sub("\n\n", text).strip()


def markdown_tables_to_plain_text(text: str) -> str:
    """Expand standard Markdown tables into labeled plain-text rows."""
    lines = text.splitlines()
    output: list[str] = []
    index = 0

    while index < len(lines):
        if index + 1 >= len(lines):
            output.append(lines[index])
            break

        headers = _markdown_table_cells(lines[index])
        separators = _markdown_table_cells(lines[index + 1])
        if not headers or not _is_markdown_table_separator(separators, len(headers)):
            output.append(lines[index])
            index += 1
            continue

        index += 2
        converted_rows: list[str] = []
        while index < len(lines):
            values = _markdown_table_cells(lines[index])
            if not values:
                break

            pairs = [
                f"{header}：{values[column]}"
                for column, header in enumerate(headers)
                if header and column < len(values) and values[column]
            ]
            if pairs:
                converted_rows.append("；".join(pairs))
            index += 1

        output.extend(converted_rows or ["；".join(headers)])

    return "\n".join(output)


def _markdown_table_cells(line: str) -> list[str]:
    """Return cells for a pipe-delimited Markdown table line."""
    stripped = line.strip()
    if "|" not in stripped:
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_markdown_table_separator(cells: list[str], header_count: int) -> bool:
    """Return whether cells form a Markdown table separator row."""
    return (
        len(cells) == header_count
        and bool(cells)
        and all(MARKDOWN_TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells)
    )


def _picture_id_from_path(image_path: str) -> str:
    """Extract a picture ID from a Markdown image path."""
    normalized = image_path.replace("\\", "/").split("?", 1)[0].split("#", 1)[0]
    return PurePosixPath(normalized).stem
