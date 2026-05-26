from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NoopLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def install_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    if name in {"app", "app.core", "app.models", "app.services", "langchain_core"}:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def load_module(monkeypatch, module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def install_vector_dependencies(monkeypatch, collection, filter_expr: str = "metadata_filter"):
    install_module(monkeypatch, "app")
    install_module(monkeypatch, "app.core")
    install_module(monkeypatch, "app.services")
    install_module(monkeypatch, "loguru", logger=NoopLogger())
    install_module(monkeypatch, "pymilvus", Collection=object)
    install_module(
        monkeypatch,
        "app.core.milvus_client",
        milvus_manager=types.SimpleNamespace(get_collection=lambda: collection),
    )
    install_module(
        monkeypatch,
        "app.services.vector_embedding_service",
        vector_embedding_service=types.SimpleNamespace(embed_query=lambda query: [0.0]),
    )
    install_module(
        monkeypatch,
        "app.services.vector_store_manager",
        vector_store_manager=types.SimpleNamespace(
            build_metadata_filter_expr=lambda **kwargs: filter_expr
        ),
    )


def make_rows(count: int):
    return [
        {
            "id": f"chunk_{index}",
            "content": f"stored content {index}",
            "metadata": {"text": f"metadata text {index}", "chunk_id": f"chunk_{index}"},
        }
        for index in range(count)
    ]


def test_query_all_documents_reads_all_iterator_batches_and_closes(monkeypatch):
    rows = make_rows(250)

    class FakeIterator:
        def __init__(self):
            self.batches = [rows[:100], rows[100:200], rows[200:], []]
            self.closed = False

        def next(self):
            return self.batches.pop(0)

        def close(self):
            self.closed = True

    class FakeCollection:
        def __init__(self):
            self.iterator = FakeIterator()
            self.query_iterator_kwargs = None

        def query_iterator(self, **kwargs):
            self.query_iterator_kwargs = kwargs
            return self.iterator

    collection = FakeCollection()
    install_vector_dependencies(monkeypatch, collection)
    module = load_module(
        monkeypatch,
        "vector_search_service_under_test",
        "app/services/vector_search_service.py",
    )

    results = module.VectorSearchService().query_all_documents(doc_id="manual_1", batch_size=100)

    assert len(results) == 250
    assert results[0].content == "metadata text 0"
    assert collection.iterator.closed is True
    assert collection.query_iterator_kwargs["batch_size"] == 100
    assert collection.query_iterator_kwargs["limit"] == -1
    assert collection.query_iterator_kwargs["expr"] == "metadata_filter"


def test_query_all_documents_paginates_when_iterator_is_unavailable(monkeypatch):
    rows = make_rows(250)

    class FakeCollection:
        def __init__(self):
            self.calls = []

        def query(self, **kwargs):
            self.calls.append(kwargs)
            offset = kwargs.get("offset", 0)
            limit = kwargs["limit"]
            return rows[offset : offset + limit]

    collection = FakeCollection()
    install_vector_dependencies(monkeypatch, collection)
    module = load_module(
        monkeypatch,
        "vector_search_service_under_test",
        "app/services/vector_search_service.py",
    )

    results = module.VectorSearchService().query_all_documents(batch_size=100)

    assert len(results) == 250
    assert [call["offset"] for call in collection.calls] == [0, 100, 200]
    assert all(call["limit"] == 100 for call in collection.calls)


def test_run_scan_stage_reranks_candidates_beyond_old_scan_limit(monkeypatch):
    class SearchResult:
        def __init__(self, id, content, score=0.0, metadata=None):
            self.id = id
            self.content = content
            self.score = score
            self.metadata = metadata or {"chunk_id": id}

    rows = [
        SearchResult(f"chunk_{index}", "ordinary chunk")
        for index in range(220)
    ]
    rows[199] = SearchResult("chunk_199", "needle chunk from beyond old limit")

    class FakeVectorService:
        def __init__(self):
            self.query_all_calls = []
            self.query_documents_calls = []

        def query_all_documents(self, **kwargs):
            self.query_all_calls.append(kwargs)
            return list(rows)

        def query_documents(self, **kwargs):
            self.query_documents_calls.append(kwargs)
            return []

    vector_service = FakeVectorService()

    install_module(monkeypatch, "app")
    install_module(monkeypatch, "app.models")
    install_module(monkeypatch, "app.services")
    install_module(monkeypatch, "langchain_core")
    install_module(monkeypatch, "app.config", config=types.SimpleNamespace(rag_top_k=3))
    install_module(
        monkeypatch,
        "app.models.response",
        sanitize_summary_metadata=lambda metadata: dict(metadata or {}),
    )
    install_module(monkeypatch, "loguru", logger=NoopLogger())
    install_module(
        monkeypatch,
        "langchain_core.documents",
        Document=type("Document", (), {}),
    )
    install_module(
        monkeypatch,
        "langchain_core.tools",
        tool=lambda *args, **kwargs: (lambda func: func),
    )
    install_module(
        monkeypatch,
        "app.services.vector_search_service",
        SearchResult=SearchResult,
        vector_search_service=vector_service,
    )

    module = load_module(monkeypatch, "knowledge_tool_under_test", "app/tools/knowledge_tool.py")
    module.resolve_chunk_types = lambda chunk_families, doc_id=None: ["subsection"]
    module.filter_results_to_active_docs = lambda results, scoped_doc_id=None: list(results)
    module.post_filter_results = lambda results, intent, query_terms: list(results)

    def rerank(results, original_query, query_terms, intent, prefer_support=False):
        return sorted(results, key=lambda item: "needle" in item.content, reverse=True)

    module.rerank_results = rerank

    hits = module.run_scan_stage(
        original_query="needle",
        query_terms=["needle"],
        intent="general",
        doc_id="manual_1",
        tiers=["primary"],
        chunk_families=["subsection"],
        top_k=3,
    )

    assert hits[0].id == "chunk_199"
    assert len(hits) == 3
    assert vector_service.query_documents_calls == []
    assert vector_service.query_all_calls == [
        {
            "doc_id": "manual_1",
            "retrieval_tiers": ["primary"],
            "chunk_types": ["subsection"],
        }
    ]
