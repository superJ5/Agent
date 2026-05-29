"""Vector search service built on top of Milvus."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from loguru import logger
from pymilvus import Collection

from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service
from app.services.vector_store_manager import vector_store_manager


METADATA_QUERY_BATCH_SIZE = 1000


class SearchResult:
    """Search result value object."""

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dict."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class VectorSearchService:
    """Thin Milvus search wrapper with optional metadata filtering."""

    def __init__(self):
        logger.info("VectorSearchService initialized")

    @staticmethod
    def _rows_to_search_results(rows: Iterable[dict[str, Any]]) -> List[SearchResult]:
        """Convert Milvus query rows into search results."""
        results: List[SearchResult] = []
        for row in rows:
            metadata = row.get("metadata", {}) or {}
            content = metadata.get("text") or row.get("content") or ""
            results.append(
                SearchResult(
                    id=row.get("id", ""),
                    content=content,
                    score=0.0,
                    metadata=metadata,
                )
            )
        return results

    @staticmethod
    def _close_query_iterator(iterator: Any) -> None:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    def _build_query_expr(
        self,
        doc_id: str | None = None,
        retrieval_tiers: str | Sequence[str] | None = None,
        chunk_types: str | Sequence[str] | None = None,
        chunk_ids: str | Sequence[str] | None = None,
    ) -> str:
        filter_expr = vector_store_manager.build_metadata_filter_expr(
            doc_id=doc_id,
            retrieval_tiers=retrieval_tiers,
            chunk_types=chunk_types,
            chunk_ids=chunk_ids,
        )
        return filter_expr or 'id != ""'

    def search_similar_documents(
        self,
        query: str,
        top_k: int = 3,
        doc_id: str | None = None,
        retrieval_tiers: str | Sequence[str] | None = None,
        chunk_types: str | Sequence[str] | None = None,
        chunk_ids: str | Sequence[str] | None = None,
    ) -> List[SearchResult]:
        """Search similar documents with optional metadata filters."""
        try:
            logger.info(
                "Start vector search: query='{}', top_k={}, doc_id={}, retrieval_tiers={}, chunk_types={}, chunk_ids={}",
                query,
                top_k,
                doc_id,
                retrieval_tiers,
                chunk_types,
                chunk_ids,
            )

            query_vector = vector_embedding_service.embed_query(query)
            collection: Collection = milvus_manager.get_collection()
            filter_expr = vector_store_manager.build_metadata_filter_expr(
                doc_id=doc_id,
                retrieval_tiers=retrieval_tiers,
                chunk_types=chunk_types,
                chunk_ids=chunk_ids,
            )

            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64},
            }

            search_kwargs: dict[str, Any] = {
                "data": [query_vector],
                "anns_field": "vector",
                "param": search_params,
                "limit": top_k,
                "output_fields": ["id", "content", "metadata"],
            }
            if filter_expr:
                search_kwargs["expr"] = filter_expr

            results = collection.search(**search_kwargs)

            search_results: List[SearchResult] = []
            for hits in results:
                for hit in hits:
                    metadata = hit.entity.get("metadata", {}) or {}
                    content = metadata.get("text") or hit.entity.get("content") or ""
                    search_results.append(
                        SearchResult(
                            id=hit.entity.get("id"),
                            content=content,
                            score=hit.distance,
                            metadata=metadata,
                        )
                    )

            logger.info(
                "Vector search finished: {} results, filter_expr={}",
                len(search_results),
                filter_expr,
            )
            return search_results

        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            raise RuntimeError(f"搜索失败: {exc}") from exc

    def query_documents(
        self,
        doc_id: str | None = None,
        retrieval_tiers: str | Sequence[str] | None = None,
        chunk_types: str | Sequence[str] | None = None,
        chunk_ids: str | Sequence[str] | None = None,
        limit: int = 256,
    ) -> List[SearchResult]:
        """Query documents by metadata filters without vector similarity search."""
        try:
            collection: Collection = milvus_manager.get_collection()
            expr = self._build_query_expr(
                doc_id=doc_id,
                retrieval_tiers=retrieval_tiers,
                chunk_types=chunk_types,
                chunk_ids=chunk_ids,
            )

            rows = collection.query(
                expr=expr,
                limit=limit,
                output_fields=["id", "content", "metadata"],
            )
            results = self._rows_to_search_results(rows)

            logger.info(
                "Metadata query finished: {} results, expr={}",
                len(results),
                expr,
            )
            return results
        except Exception as exc:
            logger.error(f"Metadata query failed: {exc}")
            raise RuntimeError(f"查询失败: {exc}") from exc

    def query_all_documents(
        self,
        doc_id: str | None = None,
        retrieval_tiers: str | Sequence[str] | None = None,
        chunk_types: str | Sequence[str] | None = None,
        chunk_ids: str | Sequence[str] | None = None,
        batch_size: int = METADATA_QUERY_BATCH_SIZE,
    ) -> List[SearchResult]:
        """Query all documents matching metadata filters without vector similarity search."""
        try:
            collection: Collection = milvus_manager.get_collection()
            expr = self._build_query_expr(
                doc_id=doc_id,
                retrieval_tiers=retrieval_tiers,
                chunk_types=chunk_types,
                chunk_ids=chunk_ids,
            )
            safe_batch_size = max(int(batch_size), 1)

            query_iterator = getattr(collection, "query_iterator", None)
            if callable(query_iterator):
                iterator = None
                results: List[SearchResult] = []
                try:
                    iterator = query_iterator(
                        expr=expr,
                        batch_size=safe_batch_size,
                        limit=-1,
                        output_fields=["id", "content", "metadata"],
                    )
                    while True:
                        rows = iterator.next()
                        if not rows:
                            break
                        results.extend(self._rows_to_search_results(rows))
                finally:
                    if iterator is not None:
                        self._close_query_iterator(iterator)

                logger.info(
                    "Metadata full query finished: {} results, expr={}, batch_size={}",
                    len(results),
                    expr,
                    safe_batch_size,
                )
                return results

            logger.warning(
                "Milvus query_iterator is unavailable; falling back to offset pagination."
            )
            results = []
            offset = 0
            while True:
                rows = collection.query(
                    expr=expr,
                    limit=safe_batch_size,
                    offset=offset,
                    output_fields=["id", "content", "metadata"],
                )
                if not rows:
                    break
                results.extend(self._rows_to_search_results(rows))
                if len(rows) < safe_batch_size:
                    break
                offset += len(rows)

            logger.info(
                "Metadata full query finished via pagination: {} results, expr={}, batch_size={}",
                len(results),
                expr,
                safe_batch_size,
            )
            return results
        except Exception as exc:
            logger.error(f"Full metadata query failed: {exc}")
            raise RuntimeError(f"Full metadata query failed: {exc}") from exc


vector_search_service = VectorSearchService()
