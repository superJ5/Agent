"""Section-aware chunking and Milvus storage for AIOps runbooks."""

from __future__ import annotations

from typing import Any

from loguru import logger
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.config import config
from app.services.vector_embedding_service import vector_embedding_service

VECTOR_DIM = 1024
MAX_CONTENT_BYTES = 8000


class AIOpsKnowledgeService:
    """Own the dedicated ``aiops_knowledge`` Milvus collection."""

    def __init__(self) -> None:
        self.collection_name = config.aiops_knowledge_collection_name
        self.alias = "aiops_knowledge_connection"
        self._collection: Collection | None = None

    def _connect(self) -> Collection:
        if self._collection is not None:
            return self._collection

        connections.connect(
            alias=self.alias,
            host=config.milvus_host,
            port=str(config.milvus_port),
            timeout=config.milvus_timeout / 1000,
        )
        if not utility.has_collection(self.collection_name, using=self.alias):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=MAX_CONTENT_BYTES),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(
                fields=fields,
                description="AIOps runbook knowledge",
                enable_dynamic_field=False,
            )
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                using=self.alias,
            )
            collection.create_index(
                field_name="vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 256},
                },
            )
        self._collection = Collection(self.collection_name, using=self.alias)
        self._collection.load()
        return self._collection

    def rebuild(self) -> None:
        """Drop only the dedicated AIOps collection and recreate it."""
        connections.connect(
            alias=self.alias,
            host=config.milvus_host,
            port=str(config.milvus_port),
            timeout=config.milvus_timeout / 1000,
        )
        if utility.has_collection(self.collection_name, using=self.alias):
            utility.drop_collection(self.collection_name, using=self.alias)
        self._collection = None
        self._connect()

    def add_records(self, records: list[dict[str, Any]]) -> int:
        """Embed and insert generated chunk records."""
        if not records:
            return 0
        collection = self._connect()
        inserted = 0
        batch_size = 10
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [str(record["text"]) for record in batch]
            vectors = vector_embedding_service.embed_documents(texts)
            metadata = [
                {key: value for key, value in record.items() if key != "text"} for record in batch
            ]
            collection.insert(
                [
                    [str(record["chunk_id"]) for record in batch],
                    vectors,
                    texts,
                    metadata,
                ]
            )
            inserted += len(batch)
        collection.flush()
        logger.info("Indexed {} AIOps knowledge chunks", inserted)
        return inserted

    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        """Search AIOps runbooks by semantic similarity."""
        if not query.strip():
            return []
        collection = self._connect()
        query_vector = vector_embedding_service.embed_query(query)
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k or config.aiops_knowledge_top_k,
            output_fields=["content", "metadata"],
        )
        matches: list[dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.entity
            matches.append(
                {
                    "score": float(hit.distance),
                    "content": entity.get("content"),
                    "metadata": entity.get("metadata") or {},
                }
            )
        return matches


aiops_knowledge_service = AIOpsKnowledgeService()
