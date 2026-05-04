"""Build structure-aware chunks from parsed manual JSONL files."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PARSED_ROOT = Path("data/manuals/parsed")
CHUNK_ROOT = Path("data/manuals/chunks")
TARGET_CHARS = 900
MAX_CHARS = 1300
MIN_CHARS = 280
FRONT_MATTER_CHARS = 1600
FRONT_MATTER_TITLE = "Front Matter"

PIC_RE = re.compile(r"<PIC(?::[^>]+)?>")
HEADING_RE = re.compile(r"^\s*#")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\u3002\uff01\uff1f\uff1b;.!?])\s+")


@dataclass
class ChunkReportItem:
    manual_name: str
    doc_id: str
    chunk_count: int
    min_chars: int
    max_chars: int
    avg_chars: float
    chunk_file: str


def load_record(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty parsed file: {path}")
    return json.loads(lines[0])


def normalize_raw_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\s+(?=#\s*)", "\n", text)
    text = re.sub(r"\s+(?=[\u2022\u25cf])", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = PIC_RE.sub("\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_pic_placeholders(text: str) -> str:
    text = PIC_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(text: str) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_lines: list[str] = []
    current_title: str | None = None

    for line in text.splitlines():
        if HEADING_RE.match(line):
            if current_lines:
                sections.append(
                    {
                        "title": current_title or FRONT_MATTER_TITLE,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
            current_title = line.strip()
            current_lines = [line.strip()]
        else:
            current_lines.append(line.rstrip())

    if current_lines:
        sections.append(
            {
                "title": current_title or FRONT_MATTER_TITLE,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["text"]]


def clean_heading_line(line: str) -> str:
    line = re.sub(r"^\s*#+\s*", "", line.strip())
    return line or FRONT_MATTER_TITLE


def extract_chunk_title(raw_text: str) -> str:
    for line in raw_text.splitlines():
        if HEADING_RE.match(line):
            return clean_heading_line(line)
    plain = strip_pic_placeholders(raw_text).replace("\n", " ").strip()
    return plain[:60] if plain else FRONT_MATTER_TITLE


def starts_with_heading(text: str) -> bool:
    for line in text.splitlines():
        if not line.strip():
            continue
        return HEADING_RE.match(line) is not None
    return False


def is_front_matter_chunk(title: str) -> bool:
    return title == FRONT_MATTER_TITLE


def split_long_unit(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for paragraph in [part.strip() for part in text.split("\n\n") if part.strip()]:
        if len(strip_pic_placeholders(paragraph)) <= max_chars:
            pieces.append(paragraph)
            continue

        sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(paragraph) if part.strip()]
        if len(sentences) <= 1:
            lines = [part.strip() for part in paragraph.splitlines() if part.strip()]
            if len(lines) <= 1:
                pieces.append(paragraph)
                continue
            current = lines[0]
            for line in lines[1:]:
                candidate = f"{current}\n{line}"
                if len(strip_pic_placeholders(candidate)) > max_chars and current:
                    pieces.append(current)
                    current = line
                else:
                    current = candidate
            if current:
                pieces.append(current)
            continue

        current = sentences[0]
        for sentence in sentences[1:]:
            candidate = f"{current} {sentence}"
            if len(strip_pic_placeholders(candidate)) > max_chars and current:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces.append(current)

    return pieces or [text]


def split_large_section(text: str, max_chars: int) -> list[str]:
    if len(strip_pic_placeholders(text)) <= max_chars:
        return [text]

    units = split_long_unit(text, max_chars)
    chunks: list[str] = []
    current = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        candidate = unit if not current else f"{current}\n\n{unit}"
        limit = FRONT_MATTER_CHARS if (chunks == [] and is_front_matter_chunk(extract_chunk_title(candidate))) else max_chars
        if current and len(strip_pic_placeholders(candidate)) > limit:
            chunks.append(current.strip())
            current = unit
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


def merge_small_raw_chunks(chunks: list[str]) -> list[str]:
    merged: list[str] = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        chunk_len = len(strip_pic_placeholders(chunk))
        if not merged:
            merged.append(chunk)
            continue

        last = merged[-1]
        last_len = len(strip_pic_placeholders(last))
        same_heading_mode = starts_with_heading(chunk) == starts_with_heading(last)
        can_merge = same_heading_mode and (chunk_len < MIN_CHARS or last_len < MIN_CHARS)
        if can_merge and len(strip_pic_placeholders(f"{last}\n\n{chunk}")) <= MAX_CHARS:
            merged[-1] = f"{last}\n\n{chunk}".strip()
        else:
            merged.append(chunk)

    return merged


def build_raw_chunks(record: dict[str, Any]) -> list[str]:
    raw_text = normalize_raw_text(record.get("raw_text", ""))
    sections = extract_sections(raw_text)

    if not sections:
        return []

    chunks: list[str] = []
    for section in sections:
        title = clean_heading_line(section["title"])
        limit = FRONT_MATTER_CHARS if is_front_matter_chunk(title) else MAX_CHARS
        chunks.extend(split_large_section(section["text"], limit))

    return merge_small_raw_chunks(chunks)


def build_chunk_records(record: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    raw_chunks = build_raw_chunks(record)
    doc_id = record.get("doc_id") or source_path.stem
    doc_name = source_path.stem
    pic_refs_all = record.get("pic_refs") or []
    pic_cursor = 0

    chunk_records: list[dict[str, Any]] = []
    for index, raw_chunk in enumerate(raw_chunks, start=1):
        pic_count = len(PIC_RE.findall(raw_chunk))
        chunk_pic_refs = pic_refs_all[pic_cursor : pic_cursor + pic_count]
        pic_cursor += pic_count

        text = strip_pic_placeholders(raw_chunk)
        title = extract_chunk_title(raw_chunk)
        chunk_records.append(
            {
                "chunk_id": f"{doc_id}_chunk_{index:03d}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "chunk_index": index,
                "source_parsed": str(source_path),
                "section_title": title,
                "text": text,
                "raw_text": raw_chunk,
                "char_count": len(text),
                "pic_count": pic_count,
                "pic_refs": [item.get("image_id") for item in chunk_pic_refs if item.get("image_id")],
                "image_paths": [item.get("image_path") for item in chunk_pic_refs if item.get("image_path")],
            }
        )

    return chunk_records


def main() -> None:
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)

    report_items: list[ChunkReportItem] = []
    for parsed_path in sorted(PARSED_ROOT.glob("*.jsonl")):
        record = load_record(parsed_path)
        chunk_records = build_chunk_records(record, parsed_path)

        chunk_path = CHUNK_ROOT / f"{parsed_path.stem}.jsonl"
        with chunk_path.open("w", encoding="utf-8") as fh:
            for chunk in chunk_records:
                fh.write(json.dumps(chunk, ensure_ascii=False))
                fh.write("\n")

        char_counts = [chunk["char_count"] for chunk in chunk_records] or [0]
        report_items.append(
            ChunkReportItem(
                manual_name=parsed_path.stem,
                doc_id=record.get("doc_id") or parsed_path.stem,
                chunk_count=len(chunk_records),
                min_chars=min(char_counts),
                max_chars=max(char_counts),
                avg_chars=round(sum(char_counts) / len(char_counts), 2) if char_counts else 0.0,
                chunk_file=str(chunk_path),
            )
        )

    report = {
        "manual_count": len(report_items),
        "chunk_files": [asdict(item) for item in report_items],
    }
    (CHUNK_ROOT / "chunking_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "generated_chunk_files": len(report_items),
                "report_path": str(CHUNK_ROOT / "chunking_report.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
