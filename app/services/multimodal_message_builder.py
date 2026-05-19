"""Build LangChain messages for text-only and multimodal chat models."""

from __future__ import annotations

from langchain_core.messages import HumanMessage


def build_user_message(question: str, images: list[str] | None = None) -> HumanMessage:
    """Create a user message that works for plain text and image-capable Qwen models."""
    normalized_images = [
        image.strip()
        for image in images or []
        if isinstance(image, str) and image.strip()
    ]

    if not normalized_images:
        return HumanMessage(content=question)

    content: list[dict] = [{"type": "text", "text": question}]
    for image in normalized_images:
        content.append({"type": "image_url", "image_url": {"url": image}})

    return HumanMessage(content=content)
