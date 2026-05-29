"""Vector indexing service for uploads and structured manual chunks."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager

CHUNK_REPORT_FILES = {"chunking_report.json", "chunk_integrity_report.json"}
ABSOLUTE_IMAGE_PATH_RE = re.compile(r"绝对路径[：:]\s*`([^`]+)`")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class IndexingResult:
    """Summary for a directory indexing run."""

    def __init__(self) -> None:
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.error_message = ""
        self.failed_files: dict[str, str] = {}

    def increment_success_count(self) -> None:
        self.success_count += 1

    def increment_fail_count(self) -> None:
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str) -> None:
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }


class VectorIndexService:
    """Read files, build documents, and store them in Milvus."""

    def __init__(self) -> None:
        self.upload_path = "./uploads"
        self.manual_chunk_path = "./data/manuals/chunks"
        logger.info("VectorIndexService initialized")

    def index_directory(self, directory_path: str | None = None) -> IndexingResult:
        """Index all supported files in a directory."""
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"Invalid directory path: {target_path}")

            result.directory_path = self._to_repo_relative_path(dir_path)
            files = self._collect_plain_text_files(dir_path)
            return self._index_files(files, result, target_path)

        except Exception as exc:
            logger.error(f"Directory indexing failed: {exc}")
            result.success = False
            result.error_message = str(exc)
            result.end_time = datetime.now()
            return result

    def index_manual_chunks(self, directory_path: str | None = None) -> IndexingResult:
        """Index generated manual chunk JSONL files."""
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            target_path = directory_path if directory_path else self.manual_chunk_path
            target = Path(target_path).resolve()

            if not target.exists():
                raise ValueError(f"Invalid manual chunk path: {target_path}")

            result.directory_path = self._to_repo_relative_path(target)
            files = self._collect_manual_chunk_files(target)
            return self._index_files(files, result, target_path)

        except Exception as exc:
            logger.error(f"Manual chunk indexing failed: {exc}")
            result.success = False
            result.error_message = str(exc)
            result.end_time = datetime.now()
            return result

    @staticmethod
    def _collect_plain_text_files(dir_path: Path) -> list[Path]:
        return sorted(
            list(dir_path.glob("*.txt"))
            + list(dir_path.glob("*.md"))
            + [
                path
                for path in dir_path.glob("*.jsonl")
                if path.name not in CHUNK_REPORT_FILES
            ]
        )

    @staticmethod
    def _collect_manual_chunk_files(target: Path) -> list[Path]:
        if target.is_file():
            if target.suffix.lower() != ".jsonl":
                raise ValueError(f"Manual chunk input must be a JSONL file: {target}")
            return [target]

        nested_chunk_files = sorted(target.rglob("chunks.jsonl"))
        if nested_chunk_files:
            return nested_chunk_files

        return sorted(
            path
            for path in target.glob("*.jsonl")
            if path.name not in CHUNK_REPORT_FILES
        )

    def _index_files(
        self,
        files: list[Path],
        result: IndexingResult,
        target_path: str,
    ) -> IndexingResult:
        if not files:
            logger.warning("No supported files found in directory: {}", target_path)
            result.success = True
            result.end_time = datetime.now()
            return result

        result.total_files = len(files)
        logger.info("Indexing directory {} with {} files", target_path, len(files))

        for file_path in files:
            try:
                self.index_single_file(str(file_path))
                result.increment_success_count()
                logger.info("Indexed file successfully: {}", file_path.name)
            except Exception as exc:
                result.increment_fail_count()
                result.add_failed_file(str(file_path), str(exc))
                logger.error("Failed to index file {}: {}", file_path.name, exc)

        result.success = result.fail_count == 0
        result.end_time = datetime.now()
        return result

    def index_single_file(self, file_path: str) -> None:
        """Index a single supported file."""
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"File does not exist: {file_path}")

        logger.info("Start indexing file: {}", path)

        try:
            if path.suffix.lower() == ".jsonl":
                self._index_chunk_jsonl(path)
            else:
                self._index_plain_text_file(path)
        except Exception as exc:
            logger.error("Failed to index file {}: {}", file_path, exc)
            raise RuntimeError(f"Failed to index file: {exc}") from exc

    def _index_plain_text_file(self, path: Path) -> None:
        """Index legacy txt / md uploads."""
        content = path.read_text(encoding="utf-8")
        normalized_path = self._to_repo_relative_path(path)

        vector_store_manager.delete_by_source(normalized_path)
        vector_store_manager.delete_by_source(path.as_posix())
        documents = document_splitter_service.split_document(content, normalized_path)

        if not documents:
            logger.warning("No documents produced from {}", path)
            return

        vector_store_manager.add_documents(documents)
        logger.info("Indexed {} text documents from {}", len(documents), path)

    def _index_chunk_jsonl(self, path: Path) -> None:
        """Index structured chunk JSONL files built from manuals."""
        chunk_records = self._load_chunk_records(path)
        if not chunk_records:
            logger.warning("Chunk JSONL is empty: {}", path)
            return

        doc_id = chunk_records[0].get("doc_id")
        if isinstance(doc_id, str) and doc_id.strip():
            vector_store_manager.delete_by_doc_id(doc_id)
        vector_store_manager.delete_by_source(self._to_repo_relative_path(path))
        vector_store_manager.delete_by_source(path.resolve().as_posix())

        documents = self._build_documents_from_chunk_records(chunk_records, path)
        if not documents:
            logger.warning("No chunk documents produced from {}", path)
            return

        vector_store_manager.add_documents(documents)
        self._refresh_retrieval_metadata_cache()
        logger.info("Indexed {} chunk documents from {}", len(documents), path)

    @staticmethod
    def _load_chunk_records(path: Path) -> list[dict[str, Any]]:
        """Load all chunk rows from a JSONL file."""
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records

    @staticmethod
    def _build_documents_from_chunk_records(
        chunk_records: list[dict[str, Any]],
        source_path: Path,
    ) -> list[Document]:
        """Convert chunk rows to LangChain documents."""
        documents: list[Document] = []
        image_path_lookup = VectorIndexService._build_image_path_lookup(chunk_records)

        for index, record in enumerate(chunk_records, start=1):
            if VectorIndexService._is_structured_chunk_record(record):
                document = VectorIndexService._build_structured_document(
                    record,
                    source_path,
                    image_path_lookup,
                    index,
                )
            else:
                document = VectorIndexService._build_legacy_document(
                    record,
                    source_path,
                    index,
                )

            if document is not None:
                documents.append(document)

        return documents

    @staticmethod
    def _is_structured_chunk_record(record: dict[str, Any]) -> bool:
        return any(
            key in record
            for key in (
                "retrieval_tier",
                "chunk_type",
                "section_path",
                "index_text",
                "parent_chunk_id",
            )
        )

    @staticmethod
    def _build_image_path_lookup(chunk_records: list[dict[str, Any]]) -> dict[str, str]:
        image_paths: dict[str, str] = {}

        for record in chunk_records:
            if record.get("chunk_type") != "metadata_image_path":
                continue

            pic_ids = record.get("pic_ids") or []
            if not isinstance(pic_ids, list) or not pic_ids:
                continue

            image_path = VectorIndexService._extract_absolute_image_path(record)
            if not image_path:
                continue

            for pic_id in pic_ids:
                if isinstance(pic_id, str) and pic_id.strip():
                    image_paths[pic_id] = image_path

        return image_paths

    @staticmethod
    def _extract_absolute_image_path(record: dict[str, Any]) -> str | None:
        for field in ("text", "index_text"):
            value = record.get(field)
            if not isinstance(value, str):
                continue
            match = ABSOLUTE_IMAGE_PATH_RE.search(value)
            if match:
                return VectorIndexService._normalize_portable_path(match.group(1).strip())
        return None

    @staticmethod
    def _to_repo_relative_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return resolved.as_posix()

    @staticmethod
    def _normalize_portable_path(value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None

        normalized = re.sub(r"/+", "/", value.strip().replace("\\", "/"))
        path = Path(normalized)
        if path.exists():
            return VectorIndexService._to_repo_relative_path(path)

        for marker in ("data/", "docs/", "uploads/"):
            marker_index = normalized.find(marker)
            if marker_index >= 0:
                return normalized[marker_index:]

        reviewed_candidate = PROJECT_ROOT / "data" / "manuals" / "reviewed_md" / Path(normalized).name
        if reviewed_candidate.exists():
            return VectorIndexService._to_repo_relative_path(reviewed_candidate)

        return normalized

    @staticmethod
    def _derive_doc_name(record: dict[str, Any], source_path: Path) -> str:
        doc_name = record.get("doc_name")
        if isinstance(doc_name, str) and doc_name.strip():
            return doc_name.strip()

        parent_name = source_path.parent.name
        if "__" in parent_name:
            suffix = parent_name.split("__", 1)[1].strip()
            if suffix:
                return suffix

        doc_id = record.get("doc_id")
        if isinstance(doc_id, str) and doc_id.strip():
            return doc_id.strip()

        return source_path.stem

    @staticmethod
    def _build_structured_document(
        record: dict[str, Any],
        source_path: Path,
        image_path_lookup: dict[str, str],
        fallback_index: int,
    ) -> Document | None:
        content = str(record.get("text", "")).strip()
        if not content:
            return None

        index_text = str(record.get("index_text") or content).strip()
        raw_pic_ids = record.get("pic_ids")
        pic_ids: list[Any] = raw_pic_ids if isinstance(raw_pic_ids, list) else []
        image_paths = [
            image_path_lookup[pic_id]
            for pic_id in pic_ids
            if isinstance(pic_id, str) and pic_id in image_path_lookup
        ]

        title = record.get("title")
        raw_section_path = record.get("section_path")
        section_path: list[Any] = raw_section_path if isinstance(raw_section_path, list) else []
        section_title = title or (section_path[-1] if section_path else None)

        metadata = {
            "_source": VectorIndexService._to_repo_relative_path(source_path),
            "_extension": source_path.suffix,
            "_file_name": source_path.name,
            "doc_id": record.get("doc_id"),
            "doc_name": VectorIndexService._derive_doc_name(record, source_path),
            "language": record.get("language"),
            "chunk_id": record.get("chunk_id"),
            "chunk_index": record.get("chunk_index", fallback_index),
            "section_title": section_title,
            "title": title,
            "retrieval_tier": record.get("retrieval_tier"),
            "chunk_type": record.get("chunk_type"),
            "section_path": section_path,
            "parent_chunk_id": record.get("parent_chunk_id"),
            "pic_ids": pic_ids,
            "pic_refs": pic_ids,
            "image_paths": image_paths,
            "source_file": VectorIndexService._normalize_portable_path(record.get("source_file")),
            "source_lines": record.get("source_lines"),
            "source_quality": record.get("source_quality"),
            "source_issue_flags": record.get("source_issue_flags", []),
            "source_issue_note": record.get("source_issue_note", ""),
            "item_no": record.get("item_no"),
            "item_kind": record.get("item_kind"),
            "index_text": index_text,
            "text": content,
            "char_count": len(content),
            "pic_count": len(pic_ids),
        }
        return Document(page_content=content, metadata=metadata)

    @staticmethod
    def _build_legacy_document(
        record: dict[str, Any],
        source_path: Path,
        fallback_index: int,
    ) -> Document | None:
        content = str(record.get("text", "")).strip()
        if not content:
            return None

        raw_image_paths_value = record.get("image_paths")
        raw_image_paths: list[Any] = (
            raw_image_paths_value if isinstance(raw_image_paths_value, list) else []
        )
        image_paths = [
            VectorIndexService._normalize_portable_path(image_path)
            for image_path in raw_image_paths
        ]
        raw_pic_refs = record.get("pic_refs")
        pic_refs: list[Any] = raw_pic_refs if isinstance(raw_pic_refs, list) else []

        metadata = {
            "_source": VectorIndexService._to_repo_relative_path(source_path),
            "_extension": source_path.suffix,
            "_file_name": source_path.name,
            "doc_id": record.get("doc_id"),
            "doc_name": VectorIndexService._derive_doc_name(record, source_path),
            "language": record.get("language"),
            "chunk_id": record.get("chunk_id"),
            "chunk_index": record.get("chunk_index", fallback_index),
            "section_title": record.get("section_title"),
            "title": record.get("section_title"),
            "image_paths": image_paths,
            "pic_refs": pic_refs,
            "pic_ids": pic_refs,
            "source_parsed": VectorIndexService._normalize_portable_path(record.get("source_parsed")),
            "index_text": content,
            "text": content,
            "char_count": record.get("char_count"),
            "pic_count": record.get("pic_count"),
        }
        return Document(page_content=content, metadata=metadata)

    @staticmethod
    def _refresh_retrieval_metadata_cache() -> None:
        """Refresh cached document routing metadata after structured reindexing."""
        try:
            from app.tools.knowledge_tool import reset_profile_caches

            reset_profile_caches()
        except Exception as exc:
            logger.warning("Failed to refresh retrieval metadata cache: {}", exc)


vector_index_service = VectorIndexService()
