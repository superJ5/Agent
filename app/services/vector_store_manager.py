"""Vector store manager wrapping Milvus operations."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, List, Sequence

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


COLLECTION_NAME = "biz"
MAX_PRIMARY_KEY_LENGTH = 100
EMBEDDING_BATCH_SIZE = 10
MAX_CONTENT_BYTES = 8000


class VectorStoreManager:
    """High-level Milvus vector store helper."""

    def __init__(self) -> None:
        self.vector_store: Milvus | None = None
        self.collection_name = COLLECTION_NAME
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

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to Milvus using index_text embeddings and text as display content."""
        if not documents:
            return []

        try:
            start_time = time.time()
            collection = milvus_manager.get_collection()
            ids = [self._build_document_id(document) for document in documents]
            rows = []
            for batch_start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
                batch_documents = documents[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                batch_ids = ids[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                embedding_inputs = [
                    self._get_embedding_text(document) for document in batch_documents
                ]
                embeddings = vector_embedding_service.embed_documents(embedding_inputs)

                for doc_id, document, embedding in zip(
                    batch_ids,
                    batch_documents,
                    embeddings,
                ):
                    stored_content = self._truncate_varchar_bytes(document.page_content)
                    rows.append(
                        {
                            "id": doc_id,
                            "vector": embedding,
                            "content": stored_content,
                            "metadata": self._sanitize_metadata(document.metadata or {}),
                        }
                    )

            _ = collection.insert(rows)
            collection.flush()
            elapsed = time.time() - start_time
            logger.info(
                "Added {} documents to VectorStore in {:.2f}s",
                len(documents),
                elapsed,
            )
            return ids
        except Exception as exc:
            logger.error(f"Failed to add documents: {exc}")
            raise

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
        metadata = document.metadata or {}
        index_text = metadata.get("index_text")
        if isinstance(index_text, str) and index_text.strip():
            return index_text.strip()
        return document.page_content

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
        return json.loads(json.dumps(metadata, ensure_ascii=False, default=str))

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
        retrieval_tiers: str | Sequence[str] | None = None,
        chunk_types: str | Sequence[str] | None = None,
        chunk_ids: str | Sequence[str] | None = None,
    ) -> str | None:
        parts = [
            cls.build_metadata_equals_expr("doc_id", doc_id),
            cls._build_metadata_in_expr("retrieval_tier", retrieval_tiers),
            cls._build_metadata_in_expr("chunk_type", chunk_types),
            cls._build_metadata_in_expr("chunk_id", chunk_ids),
        ]
        parts = [part for part in parts if part]
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
            return deleted_count
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
            return deleted_count
        except Exception as exc:
            logger.warning(f"Delete by doc_id skipped for {doc_id}: {exc}")
            return 0

    def get_vector_store(self) -> Milvus:
        """Get the LangChain Milvus vector store instance."""
        if self.vector_store is None:
            raise RuntimeError("VectorStore is not initialized")
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Run a similarity search directly through LangChain Milvus."""
        if self.vector_store is None:
            raise RuntimeError("VectorStore is not initialized")

        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug("Similarity search finished for query='{}', count={}", query, len(docs))
            return docs
        except Exception as exc:
            logger.error(f"Similarity search failed: {exc}")
            return []


vector_store_manager = VectorStoreManager()
