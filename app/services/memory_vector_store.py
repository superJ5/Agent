"""Milvus vector store for persistent memory."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from loguru import logger
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusException, utility

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


VECTOR_DIM = 1024
ID_MAX_LENGTH = 100
CONTENT_MAX_LENGTH = 8000
EMBEDDING_BATCH_SIZE = 10


class MemoryVectorStore:
    """Small Milvus wrapper for memory chunks."""

    def __init__(self) -> None:
        self.collection_name = config.memory_collection_name
        self._collection: Collection | None = None

    def ensure_collection(self) -> Collection:
        """Connect to Milvus and create/load the memory collection if needed."""
        if self._collection is not None:
            return self._collection

        _ = milvus_manager.connect()
        if not utility.has_collection(self.collection_name):
            logger.info("collection '{}' 不存在，正在创建...", self.collection_name)
            self._create_collection()
        else:
            self._collection = Collection(self.collection_name)
            self._ensure_vector_dim()

        self._load_collection()
        return self._collection

    def add_memory_chunks(self, chunks: list[dict[str, Any]]) -> list[str]:
        """Embed and insert memory chunks into Milvus."""
        if not chunks:
            return []

        collection = self.ensure_collection()
        ids = [self._build_chunk_id(chunk) for chunk in chunks]
        rows: list[dict[str, Any]] = []
        start_time = time.time()

        for batch_start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_chunks = chunks[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
            batch_ids = ids[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
            embedding_inputs = [
                str(chunk.get("index_text") or chunk.get("content") or "")
                for chunk in batch_chunks
            ]
            embeddings = vector_embedding_service.embed_documents(embedding_inputs)

            for chunk_id, chunk, embedding in zip(batch_ids, batch_chunks, embeddings):
                rows.append(
                    {
                        "id": chunk_id,
                        "vector": embedding,
                        "content": self._truncate_varchar_bytes(str(chunk.get("content") or "")),
                        "metadata": self._sanitize_metadata(chunk.get("metadata") or {}),
                    }
                )

        collection.insert(rows)
        collection.flush()
        logger.info(
            "Added {} memory chunks to collection '{}' in {:.2f}s",
            len(rows),
            self.collection_name,
            time.time() - start_time,
        )
        return ids

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Vector-search curated daily memory chunks."""
        if not query or not query.strip():
            return []

        collection = self.ensure_collection()
        query_vector = vector_embedding_service.embed_query(query)
        limit = top_k or config.memory_search_top_k
        expr = self._build_scope_expr()

        try:
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=limit,
                expr=expr,
                output_fields=["id", "content", "metadata"],
            )
        except Exception as exc:
            logger.warning("Memory vector scoped search failed; retrying with post-filter: {}", exc)
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=max(limit * 5, config.memory_search_top_k * 3),
                output_fields=["id", "content", "metadata"],
            )

        hits: list[dict[str, Any]] = []
        for hit_group in results:
            for hit in hit_group:
                metadata = hit.entity.get("metadata", {}) or {}
                if not self._is_in_scope(metadata):
                    continue
                hits.append(
                    {
                        "id": hit.entity.get("id"),
                        "content": hit.entity.get("content") or "",
                        "metadata": metadata,
                        "score": float(hit.distance),
                    }
                )
        return hits[:limit]

    def delete_by_source(self, source_path: str) -> int:
        """Delete memory chunks from one source path."""
        if not source_path:
            return 0

        try:
            collection = self.ensure_collection()
            expr = f'metadata["source_path"] == {json.dumps(source_path, ensure_ascii=False)}'
            result = collection.delete(expr)
            collection.flush()
            return result.delete_count if hasattr(result, "delete_count") else 0
        except Exception as exc:
            logger.warning("Delete memory source skipped: source={}, error={}", source_path, exc)
            return 0

    def drop_collection(self) -> None:
        """Drop memory collection for a full rebuild."""
        _ = milvus_manager.connect()
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        self._collection = None

    def _create_collection(self) -> None:
        fields = [
            FieldSchema("id", DataType.VARCHAR, max_length=ID_MAX_LENGTH, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema("content", DataType.VARCHAR, max_length=CONTENT_MAX_LENGTH),
            FieldSchema("metadata", DataType.JSON),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="Persistent memory collection",
            enable_dynamic_field=False,
        )
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
            num_shards=2,
        )
        self._collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )

    def _ensure_vector_dim(self) -> None:
        collection = self._collection
        if collection is None:
            return
        for field in collection.schema.fields:
            if field.name == "vector" and getattr(field, "params", {}).get("dim") != VECTOR_DIM:
                logger.warning("memory collection vector dim mismatch; rebuilding collection")
                utility.drop_collection(self.collection_name)
                self._create_collection()
                return

    def _load_collection(self) -> None:
        if self._collection is None:
            self._collection = Collection(self.collection_name)
        try:
            self._collection.load()
        except MilvusException as exc:
            if "loaded" not in str(exc).lower():
                raise

    @staticmethod
    def _build_scope_expr() -> str:
        return 'metadata["memory_type"] == "daily"'

    @staticmethod
    def _is_in_scope(metadata: dict[str, Any]) -> bool:
        memory_type = str(metadata.get("memory_type") or "")
        return memory_type == "daily"

    @staticmethod
    def _build_chunk_id(chunk: dict[str, Any]) -> str:
        raw_id = str(chunk.get("id") or "").strip()
        if raw_id and len(raw_id) <= ID_MAX_LENGTH:
            return raw_id

        digest_input = json.dumps(chunk, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()
        return f"mem_{digest[: ID_MAX_LENGTH - 4]}"

    @staticmethod
    def _truncate_varchar_bytes(value: str, max_bytes: int = CONTENT_MAX_LENGTH) -> str:
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value
        suffix = "..."
        budget = max_bytes - len(suffix.encode("utf-8"))
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


memory_vector_store = MemoryVectorStore()
