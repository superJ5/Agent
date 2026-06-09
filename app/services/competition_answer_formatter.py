"""Format generated answers for the competition response contract."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def format_answer_images(answer: str) -> str:
    """Convert Markdown images to ordered ``<PIC>`` placeholders and picture IDs."""
    picture_ids: list[str] = []

    def replace_image(match: re.Match[str]) -> str:
        alt_text = match.group(1).strip()
        image_path = match.group(2).strip()
        picture_id = alt_text or _picture_id_from_path(image_path)
        if not picture_id:
            return match.group(0)

        picture_ids.append(picture_id)
        return "<PIC>"

    body = MARKDOWN_IMAGE_RE.sub(replace_image, answer).strip()
    if not picture_ids:
        return body

    return (
        f"{json.dumps(body, ensure_ascii=False)},"
        f"{json.dumps(picture_ids, ensure_ascii=False)}"
    )


def _picture_id_from_path(image_path: str) -> str:
    """Extract a picture ID from a Markdown image path."""
    normalized = image_path.replace("\\", "/").split("?", 1)[0].split("#", 1)[0]
    return PurePosixPath(normalized).stem
