"""Build vector index chunks from persistent memory files."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

from loguru import logger

from app.services.memory_service import PROJECT_ROOT, memory_service
from app.services.memory_vector_store import memory_vector_store


@dataclass
class MemoryIndexResult:
    success: bool = False
    memory_root: str = ""
    total_chunks: int = 0
    indexed_chunks: int = 0
    error_message: str = ""
    indexed_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "memory_root": self.memory_root,
            "total_chunks": self.total_chunks,
            "indexed_chunks": self.indexed_chunks,
            "error_message": self.error_message,
            "indexed_sources": self.indexed_sources,
        }


class MemoryIndexService:
    """Create memory chunks from files and index them into Milvus."""

    def index_memory(self, rebuild: bool = False) -> MemoryIndexResult:
        result = MemoryIndexResult(memory_root=self._to_project_relative(memory_service.memory_root))
        try:
            if rebuild:
                memory_vector_store.drop_collection()

            chunks = self.build_memory_chunks()
            result.total_chunks = len(chunks)
            if not chunks:
                result.success = True
                return result

            for source in sorted({chunk["metadata"]["source_path"] for chunk in chunks}):
                memory_vector_store.delete_by_source(source)

            ids = memory_vector_store.add_memory_chunks(chunks)
            result.indexed_chunks = len(ids)
            result.indexed_sources = sorted({chunk["metadata"]["source_path"] for chunk in chunks})
            result.success = True
            return result
        except Exception as exc:
            logger.error("Memory indexing failed: {}", exc)
            result.success = False
            result.error_message = str(exc)
            return result

    def build_memory_chunks(self) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        root = memory_service.memory_root

        if memory_service.daily_dir.exists():
            for path in sorted(memory_service.daily_dir.glob("*.md")):
                chunks.extend(self._build_markdown_chunks(path, "daily"))

        logger.info("Built {} memory chunks from {}", len(chunks), root)
        return chunks

    def _build_markdown_chunks(self, path: Path, memory_type: str) -> list[dict[str, Any]]:
        source_path = self._to_project_relative(path)
        chunks: list[dict[str, Any]] = []
        lines = path.read_text(encoding="utf-8").splitlines()
        buffer: list[str] = []
        start_line = 1

        for line_no, line in enumerate(lines, 1):
            if line.startswith("#") and buffer:
                chunks.append(
                    self._make_chunk(
                        source_path=source_path,
                        memory_type=memory_type,
                        content="\n".join(buffer).strip(),
                        line=start_line,
                    )
                )
                buffer = []
                start_line = line_no
            if line.strip():
                buffer.append(line)

        if buffer:
            chunks.append(
                self._make_chunk(
                    source_path=source_path,
                    memory_type=memory_type,
                    content="\n".join(buffer).strip(),
                    line=start_line,
                )
            )
        return [chunk for chunk in chunks if chunk["content"]]

    def _make_chunk(
        self,
        source_path: str,
        memory_type: str,
        content: str,
        line: int,
    ) -> dict[str, Any]:
        digest_input = f"{source_path}:{line}:{memory_type}:{content}"
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()
        chunk_id = f"mem_{digest[:32]}"
        metadata = {
            "memory_type": memory_type,
            "source_path": source_path,
            "line": line,
            "content_hash": digest,
            "index_text": content,
        }
        return {
            "id": chunk_id,
            "content": content,
            "index_text": content,
            "metadata": metadata,
        }

    @staticmethod
    def _to_project_relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.resolve().as_posix()


memory_index_service = MemoryIndexService()
