"""Convert manual txt dataset records into structured JSONL.

Supported source record shapes:
1. ["text with <PIC> placeholders", ["img_001", "img_002"]]
2. {"text": "...", "images": ["img_001", "img_002"]}

The output keeps the original text and also aligns each <PIC> placeholder
with an image id and resolved image path when possible.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PIC_TOKEN = "<PIC>"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def slugify(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}_{digest}" if normalized else f"manual_{digest}"


def parse_source_text(raw: str) -> Any:
    raw = raw.lstrip("\ufeff").strip()
    if not raw:
        raise ValueError("source file is empty")

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(raw)
        except Exception:
            continue

    raise ValueError("source content is not valid JSON or Python literal")


def normalize_record(payload: Any) -> tuple[str, list[str]]:
    if isinstance(payload, list) and len(payload) == 2:
        text, images = payload
    elif isinstance(payload, dict):
        text = payload.get("text") or payload.get("content")
        images = payload.get("images") or payload.get("image_ids") or payload.get("pic_refs")
    else:
        raise ValueError("unsupported record shape")

    if not isinstance(text, str):
        raise ValueError("record text must be a string")
    if not isinstance(images, list) or not all(isinstance(item, str) for item in images):
        raise ValueError("record images must be a list[str]")

    return text, images


def find_image_path(image_dir: Path, image_id: str) -> str | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return str(candidate)

    matches = list(image_dir.glob(f"{image_id}.*"))
    for match in matches:
        if match.suffix.lower() in IMAGE_EXTENSIONS:
            return str(match)

    return None


def build_pic_refs(text: str, image_ids: list[str], image_dir: Path) -> list[dict[str, Any]]:
    pic_count = text.count(PIC_TOKEN)
    refs: list[dict[str, Any]] = []

    for index, image_id in enumerate(image_ids):
        refs.append(
            {
                "pic_index": index,
                "image_id": image_id,
                "image_path": find_image_path(image_dir, image_id),
                "matched_placeholder": index < pic_count,
            }
        )

    return refs


def remove_pic_tokens(text: str) -> str:
    text = text.replace(PIC_TOKEN, "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_output_record(
    source_path: Path,
    image_dir: Path,
    doc_id: str | None,
) -> dict[str, Any]:
    payload = parse_source_text(source_path.read_text(encoding="utf-8"))
    text, image_ids = normalize_record(payload)
    resolved_doc_id = doc_id or slugify(source_path.stem)
    pic_refs = build_pic_refs(text, image_ids, image_dir)

    return {
        "doc_id": resolved_doc_id,
        "doc_name": source_path.stem,
        "source_txt": str(source_path),
        "image_dir": str(image_dir),
        "raw_text": text,
        "clean_text": remove_pic_tokens(text),
        "pic_placeholder_count": text.count(PIC_TOKEN),
        "image_id_count": len(image_ids),
        "pic_refs": pic_refs,
        "alignment_ok": text.count(PIC_TOKEN) == len(image_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert manual txt dataset to JSONL")
    parser.add_argument("--input", required=True, help="Path to txt/json record file")
    parser.add_argument("--image-dir", required=True, help="Directory containing image files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--doc-id", help="Optional explicit doc_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = build_output_record(input_path, image_dir, args.doc_id)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"converted: {input_path}")
    print(f"output: {output_path}")
    print(
        "alignment:"
        f" placeholders={record['pic_placeholder_count']},"
        f" image_ids={record['image_id_count']},"
        f" ok={record['alignment_ok']}"
    )


if __name__ == "__main__":
    main()
