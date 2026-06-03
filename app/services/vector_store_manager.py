"""Vector store manager wrapping Milvus operations."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service

COLLECTION_NAME = "biz"
MAX_PRIMARY_KEY_LENGTH = 100
EMBEDDING_BATCH_SIZE = 10
MAX_EMBEDDING_INPUT_CHARS = 8192
MAX_CONTENT_BYTES = 8000
MAX_METADATA_BYTES = 60000
EMBEDDING_LIMIT_REPORT_PATH = Path("logs/embedding_input_limit_report.jsonl")
SKIPPED_CHUNK_REPORT_PATH = Path("logs/index_skipped_chunks_report.jsonl")
PreparedVectorRow = tuple[str, Document, dict[str, Any]]


class VectorStoreManager:
    """High-level Milvus vector store helper."""

    def __init__(self) -> None:
        self.vector_store: Milvus | None = None
        self.collection_name = COLLECTION_NAME
        self.embedding_limit_report_path = EMBEDDING_LIMIT_REPORT_PATH
        self.embedding_limit_report_count = 0
        self.skipped_chunk_report_path = SKIPPED_CHUNK_REPORT_PATH
        self.skipped_chunk_report_count = 0
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """Initialize LangChain Milvus vector store."""
        try:
            _ = milvus_manager.connect()

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,
                drop_old=False,
                text_field="content",
                vector_field="vector",
                primary_field="id",
                metadata_field="metadata",
            )

            logger.info(
                "VectorStore initialized: {}:{}, collection={}",
                config.milvus_host,
                config.milvus_port,
                self.collection_name,
            )
        except Exception as exc:
            logger.error(f"VectorStore initialization failed: {exc}")
            raise

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to Milvus using index_text embeddings and text as display content."""
        if not documents:
            return []

        try:
            start_time = time.time()
            collection = milvus_manager.get_collection()
            ids = [self._build_document_id(document) for document in documents]
            inserted_ids: list[str] = []

            for batch_start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
                batch_documents = documents[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                batch_ids = ids[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                prepared_rows = self._prepare_rows_for_batch(batch_ids, batch_documents)
                inserted_ids.extend(self._insert_rows_with_retry(collection, prepared_rows))

            if inserted_ids:
                collection.flush()
            elapsed = time.time() - start_time
            logger.info(
                "Added {} documents to VectorStore in {:.2f}s; skipped {} chunk(s)",
                len(inserted_ids),
                elapsed,
                len(documents) - len(inserted_ids),
            )
            return inserted_ids
        except Exception as exc:
            logger.error(f"Failed to add documents: {exc}")
            raise

    def _prepare_rows_for_batch(
        self,
        batch_ids: list[str],
        batch_documents: list[Document],
    ) -> list[PreparedVectorRow]:
        raw_embedding_inputs = [
            self._get_raw_embedding_text(document) for document in batch_documents
        ]
        embedded_documents = self._embed_batch_documents_for_indexing(
            batch_ids,
            batch_documents,
            raw_embedding_inputs,
        )

        prepared_rows: list[PreparedVectorRow] = []
        for doc_id, document, embedding, embedding_text in embedded_documents:
            try:
                stored_content = self._truncate_varchar_bytes(document.page_content)
                prepared_rows.append(
                    (
                        doc_id,
                        document,
                        {
                            "id": doc_id,
                            "vector": embedding,
                            "content": stored_content,
                            "metadata": self._prepare_metadata(
                                document.metadata or {},
                                stored_content=stored_content,
                                embedding_text=embedding_text,
                            ),
                        },
                    )
                )
            except Exception as exc:
                self._record_skipped_chunk(
                    document,
                    stage="metadata_prepare",
                    error=str(exc),
                    retry_attempted=False,
                    vector_id=doc_id,
                )

        return prepared_rows

    def _embed_batch_documents_for_indexing(
        self,
        batch_ids: list[str],
        batch_documents: list[Document],
        raw_embedding_inputs: list[str],
    ) -> list[tuple[str, Document, list[float], str]]:
        batch_error: str | None = None
        try:
            embeddings = vector_embedding_service.embed_documents(raw_embedding_inputs)
            if len(embeddings) != len(batch_documents):
                raise RuntimeError(
                    "Embedding service returned "
                    f"{len(embeddings)} vectors for {len(batch_documents)} chunks"
                )
            return [
                (doc_id, document, embedding, raw_text)
                for doc_id, document, embedding, raw_text in zip(
                    batch_ids,
                    batch_documents,
                    embeddings,
                    raw_embedding_inputs,
                    strict=True,
                )
            ]
        except Exception as batch_exc:
            batch_error = str(batch_exc)
            logger.warning(
                "Batch embedding failed for {} chunk(s); retrying individually: {}",
                len(batch_documents),
                batch_error,
            )

        embedded_documents: list[tuple[str, Document, list[float], str]] = []
        for doc_id, document, raw_text in zip(
            batch_ids,
            batch_documents,
            raw_embedding_inputs,
            strict=True,
        ):
            embedded = self._embed_single_document_with_retry(
                document,
                raw_text,
                batch_error=batch_error,
                vector_id=doc_id,
            )
            if embedded is None:
                continue
            embedding, embedding_text = embedded
            embedded_documents.append((doc_id, document, embedding, embedding_text))

        return embedded_documents

    def _embed_single_document_with_retry(
        self,
        document: Document,
        raw_text: str,
        *,
        batch_error: str | None = None,
        vector_id: str | None = None,
    ) -> tuple[list[float], str] | None:
        first_error = ""
        try:
            return vector_embedding_service.embed_documents([raw_text])[0], raw_text
        except Exception as first_exc:
            first_error = str(first_exc)
            fallback_text = self._truncate_embedding_text(raw_text)
            if self._is_embedding_input_limit_error(first_exc):
                self._record_embedding_limit_hit(
                    document,
                    original_text=raw_text,
                    fallback_text=fallback_text,
                    provider_error=first_error,
                )

        try:
            return vector_embedding_service.embed_documents([fallback_text])[0], fallback_text
        except Exception as retry_exc:
            self._record_skipped_chunk(
                document,
                stage="embedding",
                error=str(retry_exc),
                retry_attempted=True,
                first_error=first_error,
                batch_error=batch_error,
                vector_id=vector_id,
                original_embedding_chars=len(raw_text),
                fallback_embedding_chars=len(fallback_text),
            )
            return None

    def _insert_rows_with_retry(
        self,
        collection: Any,
        prepared_rows: list[PreparedVectorRow],
    ) -> list[str]:
        if not prepared_rows:
            return []

        batch_error = ""
        try:
            _ = collection.insert([row for _, _, row in prepared_rows])
            return [doc_id for doc_id, _, _ in prepared_rows]
        except Exception as batch_exc:
            batch_error = str(batch_exc)
            logger.warning(
                "Batch insert failed for {} chunk(s); retrying individually: {}",
                len(prepared_rows),
                batch_error,
            )

        inserted_ids: list[str] = []
        for doc_id, document, row in prepared_rows:
            try:
                _ = collection.insert([row])
                inserted_ids.append(doc_id)
            except Exception as retry_exc:
                self._record_skipped_chunk(
                    document,
                    stage="milvus_insert",
                    error=str(retry_exc),
                    retry_attempted=True,
                    first_error=batch_error,
                    vector_id=doc_id,
                    metadata_size_bytes=self._metadata_size_bytes(row.get("metadata") or {}),
                )

        return inserted_ids

    @staticmethod
    def _build_document_id(document: Document) -> str:
        """Use chunk ids as primary keys when present."""
        metadata = document.metadata or {}
        chunk_id = metadata.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id.strip():
            cleaned_chunk_id = chunk_id.strip()
            if len(cleaned_chunk_id) <= MAX_PRIMARY_KEY_LENGTH:
                return cleaned_chunk_id
            digest = hashlib.sha1(cleaned_chunk_id.encode("utf-8")).hexdigest()
            return f"chunk_{digest[:MAX_PRIMARY_KEY_LENGTH - 6]}"
        return str(uuid.uuid4())

    @staticmethod
    def _get_embedding_text(document: Document) -> str:
        raw_text = VectorStoreManager._get_raw_embedding_text(document)
        return VectorStoreManager._truncate_embedding_text(raw_text)

    @staticmethod
    def _get_raw_embedding_text(document: Document) -> str:
        metadata = document.metadata or {}
        index_text = metadata.get("index_text")
        if isinstance(index_text, str) and index_text.strip():
            return index_text.strip()
        return str(document.page_content).strip()

    def _embed_documents_with_limit_report(
        self,
        documents: list[Document],
        raw_embedding_inputs: list[str],
    ) -> tuple[list[list[float]], list[str]]:
        """Embed raw inputs; on provider token-limit errors, locate exact chunks."""
        try:
            return (
                vector_embedding_service.embed_documents(raw_embedding_inputs),
                raw_embedding_inputs,
            )
        except RuntimeError as exc:
            if not self._is_embedding_input_limit_error(exc):
                raise

        embeddings: list[list[float]] = []
        used_inputs: list[str] = []
        for document, raw_text in zip(documents, raw_embedding_inputs, strict=True):
            try:
                embedding = vector_embedding_service.embed_documents([raw_text])[0]
                embeddings.append(embedding)
                used_inputs.append(raw_text)
                continue
            except RuntimeError as exc:
                if not self._is_embedding_input_limit_error(exc):
                    raise
                fallback_text = self._truncate_embedding_text(raw_text)
                self._record_embedding_limit_hit(
                    document,
                    original_text=raw_text,
                    fallback_text=fallback_text,
                    provider_error=str(exc),
                )

            embeddings.append(vector_embedding_service.embed_documents([fallback_text])[0])
            used_inputs.append(fallback_text)

        return embeddings, used_inputs

    @staticmethod
    def _is_embedding_input_limit_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "Range of input length should be [1, 8192]" in message
            or ("8192" in message and "input length" in message.lower())
        )

    @staticmethod
    def _truncate_embedding_text(text: str) -> str:
        stripped = str(text or "").strip()
        if len(stripped) <= MAX_EMBEDDING_INPUT_CHARS:
            return stripped
        return stripped[:MAX_EMBEDDING_INPUT_CHARS]

    @staticmethod
    def _truncate_varchar_bytes(value: str, max_bytes: int = MAX_CONTENT_BYTES) -> str:
        """Trim text to fit Milvus varchar byte limits while preserving UTF-8 validity."""
        if not isinstance(value, str):
            value = str(value or "")

        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value

        suffix = "..."
        suffix_bytes = suffix.encode("utf-8")
        budget = max_bytes - len(suffix_bytes)
        if budget <= 0:
            return suffix[:max_bytes]

        truncated = encoded[:budget]
        while truncated:
            try:
                return truncated.decode("utf-8") + suffix
            except UnicodeDecodeError:
                truncated = truncated[:-1]

        return suffix

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            json.loads(json.dumps(metadata, ensure_ascii=False, default=str)),
        )

    @classmethod
    def _prepare_metadata(
        cls,
        metadata: dict[str, Any],
        *,
        stored_content: str,
        embedding_text: str,
    ) -> dict[str, Any]:
        prepared = dict(metadata)
        if isinstance(prepared.get("text"), str):
            prepared["text"] = stored_content
        if isinstance(prepared.get("index_text"), str):
            prepared["index_text"] = embedding_text

        sanitized = cls._sanitize_metadata(prepared)
        if cls._metadata_size_bytes(sanitized) <= MAX_METADATA_BYTES:
            return sanitized

        for field_name in ("index_text", "text"):
            if isinstance(sanitized.get(field_name), str):
                sanitized[field_name] = cls._truncate_varchar_bytes(
                    sanitized[field_name],
                    max_bytes=2000,
                )
            if cls._metadata_size_bytes(sanitized) <= MAX_METADATA_BYTES:
                return sanitized

        for field_name in ("index_text", "text"):
            sanitized.pop(field_name, None)
            if cls._metadata_size_bytes(sanitized) <= MAX_METADATA_BYTES:
                return sanitized

        return sanitized

    @staticmethod
    def _metadata_size_bytes(metadata: dict[str, Any]) -> int:
        return len(json.dumps(metadata, ensure_ascii=False, default=str).encode("utf-8"))

    def reset_embedding_limit_report(
        self,
        path: str | Path | None = None,
    ) -> None:
        """Start a fresh embedding input limit report for one indexing run."""
        self.embedding_limit_report_path = Path(path or EMBEDDING_LIMIT_REPORT_PATH)
        self.embedding_limit_report_count = 0
        try:
            self.embedding_limit_report_path.parent.mkdir(parents=True, exist_ok=True)
            if self.embedding_limit_report_path.exists():
                self.embedding_limit_report_path.unlink()
        except OSError as exc:
            logger.warning("Failed to reset embedding input limit report: {}", exc)

    def append_embedding_limit_report(
        self,
        path: str | Path | None = None,
    ) -> None:
        """Use an existing embedding input limit report without clearing it."""
        self.embedding_limit_report_path = Path(path or EMBEDDING_LIMIT_REPORT_PATH)
        try:
            self.embedding_limit_report_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to prepare embedding input limit report: {}", exc)
        self.embedding_limit_report_count = self._count_report_entries(
            self.embedding_limit_report_path,
        )

    @staticmethod
    def _count_report_entries(path: Path) -> int:
        try:
            if not path.exists():
                return 0
            return sum(
                1
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError as exc:
            logger.warning("Failed to count embedding input limit report: {}", exc)
            return 0

    def embedding_limit_report_summary(self) -> dict[str, Any]:
        """Return the report location and hit count for script output."""
        return {
            "path": self.embedding_limit_report_path.as_posix(),
            "count": self.embedding_limit_report_count,
            "count_note": (
                "Count is the number of JSONL entries currently tracked for this "
                "report path. In append mode it includes entries from earlier "
                "script invocations."
            ),
            "limit_note": (
                "This report records chunks whose raw embedding input was rejected "
                "by the provider with the 8192 input-token limit, then re-embedded "
                "with a local truncated fallback."
            ),
        }

    def reset_skipped_chunk_report(
        self,
        path: str | Path | None = None,
    ) -> None:
        """Start a fresh report for chunks skipped after per-chunk retry failed."""
        self.skipped_chunk_report_path = Path(path or SKIPPED_CHUNK_REPORT_PATH)
        self.skipped_chunk_report_count = 0
        try:
            self.skipped_chunk_report_path.parent.mkdir(parents=True, exist_ok=True)
            if self.skipped_chunk_report_path.exists():
                self.skipped_chunk_report_path.unlink()
        except OSError as exc:
            logger.warning("Failed to reset skipped chunk report: {}", exc)

    def append_skipped_chunk_report(
        self,
        path: str | Path | None = None,
    ) -> None:
        """Use an existing skipped chunk report without clearing it."""
        self.skipped_chunk_report_path = Path(path or SKIPPED_CHUNK_REPORT_PATH)
        try:
            self.skipped_chunk_report_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to prepare skipped chunk report: {}", exc)
        self.skipped_chunk_report_count = self._count_report_entries(
            self.skipped_chunk_report_path,
        )

    def skipped_chunk_report_summary(self) -> dict[str, Any]:
        """Return the skip report location and skipped chunk count."""
        return {
            "path": self.skipped_chunk_report_path.as_posix(),
            "count": self.skipped_chunk_report_count,
            "count_note": (
                "Count is the number of chunks skipped after per-chunk retry failed. "
                "In append mode it includes entries from earlier script invocations."
            ),
        }

    def _record_embedding_limit_hit(
        self,
        document: Document,
        *,
        original_text: str,
        fallback_text: str,
        provider_error: str,
    ) -> None:
        metadata = document.metadata or {}
        entry = {
            "generated_at": datetime.now(UTC).isoformat(),
            "doc_id": metadata.get("doc_id"),
            "doc_name": metadata.get("doc_name"),
            "chunk_id": metadata.get("chunk_id"),
            "retrieval_tier": metadata.get("retrieval_tier"),
            "chunk_type": metadata.get("chunk_type"),
            "title": metadata.get("title") or metadata.get("section_title"),
            "section_path": metadata.get("section_path") or [],
            "section_depth": len(metadata.get("section_path") or []),
            "hierarchy": " > ".join(str(part) for part in metadata.get("section_path") or []),
            "parent_chunk_id": metadata.get("parent_chunk_id"),
            "source_file": metadata.get("source_file") or metadata.get("_source"),
            "source_lines": metadata.get("source_lines"),
            "original_embedding_chars": len(original_text),
            "fallback_embedding_chars": len(fallback_text),
            "max_embedding_input_chars": MAX_EMBEDDING_INPUT_CHARS,
            "limit_kind": "provider_8192_token_limit_rejected_raw_input",
            "provider_error": provider_error,
            "token_limit_note": (
                "The provider limit is 8192 tokens. Character counts are recorded "
                "only to show the size of the raw input and fallback text."
            ),
        }
        try:
            self.embedding_limit_report_path.parent.mkdir(parents=True, exist_ok=True)
            with self.embedding_limit_report_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str))
                handle.write("\n")
            self.embedding_limit_report_count += 1
        except OSError as exc:
            logger.warning("Failed to write embedding input limit report: {}", exc)

    def _record_skipped_chunk(
        self,
        document: Document,
        *,
        stage: str,
        error: str,
        retry_attempted: bool,
        first_error: str | None = None,
        batch_error: str | None = None,
        vector_id: str | None = None,
        **extra: Any,
    ) -> None:
        metadata = document.metadata or {}
        entry = {
            "generated_at": datetime.now(UTC).isoformat(),
            "skip_kind": "single_chunk_indexing_failed_after_retry",
            "stage": stage,
            "retry_attempted": retry_attempted,
            "error": error,
            "first_error": first_error,
            "batch_error": batch_error,
            "vector_id": vector_id,
            "doc_id": metadata.get("doc_id"),
            "doc_name": metadata.get("doc_name"),
            "chunk_id": metadata.get("chunk_id"),
            "retrieval_tier": metadata.get("retrieval_tier"),
            "chunk_type": metadata.get("chunk_type"),
            "title": metadata.get("title") or metadata.get("section_title"),
            "section_path": metadata.get("section_path") or [],
            "section_depth": len(metadata.get("section_path") or []),
            "hierarchy": " > ".join(str(part) for part in metadata.get("section_path") or []),
            "parent_chunk_id": metadata.get("parent_chunk_id"),
            "source_file": metadata.get("source_file") or metadata.get("_source"),
            "source_lines": metadata.get("source_lines"),
            **extra,
        }
        logger.warning(
            "Skipped chunk after retry: chunk_id={}, retrieval_tier={}, stage={}, error={}",
            entry.get("chunk_id"),
            entry.get("retrieval_tier"),
            stage,
            error,
        )
        try:
            self.skipped_chunk_report_path.parent.mkdir(parents=True, exist_ok=True)
            with self.skipped_chunk_report_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str))
                handle.write("\n")
            self.skipped_chunk_report_count += 1
        except OSError as exc:
            logger.warning("Failed to write skipped chunk report: {}", exc)

    @staticmethod
    def _normalize_filter_values(
        values: str | Sequence[str] | None,
    ) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            stripped = values.strip()
            return [stripped] if stripped else []

        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if stripped:
                normalized.append(stripped)
        return normalized

    @staticmethod
    def build_metadata_equals_expr(field_name: str, value: str | None) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        quoted_value = json.dumps(value.strip(), ensure_ascii=False)
        return f'metadata["{field_name}"] == {quoted_value}'

    @classmethod
    def _build_metadata_in_expr(
        cls,
        field_name: str,
        values: str | Sequence[str] | None,
    ) -> str | None:
        normalized = cls._normalize_filter_values(values)
        if not normalized:
            return None
        if len(normalized) == 1:
            return cls.build_metadata_equals_expr(field_name, normalized[0])

        quoted_values = ", ".join(
            json.dumps(value, ensure_ascii=False) for value in normalized
        )
        return f'metadata["{field_name}"] in [{quoted_values}]'

    @classmethod
    def build_metadata_filter_expr(
        cls,
        doc_id: str | None = None,
        language: str | None = None,
        retrieval_tiers: str | Sequence[str] | None = None,
        chunk_types: str | Sequence[str] | None = None,
        chunk_ids: str | Sequence[str] | None = None,
        parent_chunk_ids: str | Sequence[str] | None = None,
    ) -> str | None:
        optional_parts: list[str | None] = [
            cls.build_metadata_equals_expr("doc_id", doc_id),
            cls.build_metadata_equals_expr("language", language),
            cls._build_metadata_in_expr("retrieval_tier", retrieval_tiers),
            cls._build_metadata_in_expr("chunk_type", chunk_types),
            cls._build_metadata_in_expr("chunk_id", chunk_ids),
            cls._build_metadata_in_expr("parent_chunk_id", parent_chunk_ids),
        ]
        parts = [part for part in optional_parts if part]
        if not parts:
            return None
        return " and ".join(parts)

    def delete_by_source(self, file_path: str) -> int:
        """Delete rows by original source file path."""
        try:
            collection = milvus_manager.get_collection()
            expr = self.build_metadata_equals_expr("_source", file_path)
            if expr is None:
                return 0
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            logger.info("Deleted {} rows for source={}", deleted_count, file_path)
            return int(deleted_count)
        except Exception as exc:
            logger.warning(f"Delete by source skipped for {file_path}: {exc}")
            return 0

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete rows for a structured manual document id."""
        try:
            collection = milvus_manager.get_collection()
            expr = self.build_metadata_filter_expr(doc_id=doc_id)
            if expr is None:
                return 0
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            logger.info("Deleted {} rows for doc_id={}", deleted_count, doc_id)
            return int(deleted_count)
        except Exception as exc:
            logger.warning(f"Delete by doc_id skipped for {doc_id}: {exc}")
            return 0

    def get_vector_store(self) -> Milvus:
        """Get the LangChain Milvus vector store instance."""
        if self.vector_store is None:
            raise RuntimeError("VectorStore is not initialized")
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """Run a similarity search directly through LangChain Milvus."""
        if self.vector_store is None:
            raise RuntimeError("VectorStore is not initialized")

        try:
            docs: list[Document] = self.vector_store.similarity_search(query, k=k)
            logger.debug("Similarity search finished for query='{}', count={}", query, len(docs))
            return docs
        except Exception as exc:
            logger.error(f"Similarity search failed: {exc}")
            return []


vector_store_manager = VectorStoreManager()
