from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


class NoopLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class DocumentStub:
    def __init__(self, page_content: str, metadata: dict[str, object]) -> None:
        self.page_content = page_content
        self.metadata = metadata


def install_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    if name in {
        "app",
        "app.core",
        "app.services",
        "langchain_core",
    }:
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


def install_document_dependencies(monkeypatch) -> None:
    install_module(monkeypatch, "langchain_core")
    install_module(monkeypatch, "langchain_core.documents", Document=DocumentStub)


def install_vector_store_dependencies(monkeypatch) -> None:
    install_document_dependencies(monkeypatch)
    install_module(monkeypatch, "app")
    install_module(monkeypatch, "app.core")
    install_module(monkeypatch, "app.services")
    install_module(
        monkeypatch,
        "app.config",
        config=SimpleNamespace(milvus_host="localhost", milvus_port="19530"),
    )
    install_module(
        monkeypatch,
        "app.core.milvus_client",
        milvus_manager=SimpleNamespace(connect=lambda: object()),
    )
    install_module(
        monkeypatch,
        "app.services.vector_embedding_service",
        vector_embedding_service=object(),
    )
    install_module(monkeypatch, "langchain_milvus", Milvus=lambda **kwargs: object())
    install_module(monkeypatch, "loguru", logger=NoopLogger())


def install_vector_search_dependencies(monkeypatch, collection, store_manager) -> None:
    install_module(monkeypatch, "app")
    install_module(monkeypatch, "app.core")
    install_module(monkeypatch, "app.services")
    install_module(monkeypatch, "loguru", logger=NoopLogger())
    install_module(monkeypatch, "pymilvus", Collection=object)
    install_module(
        monkeypatch,
        "app.core.milvus_client",
        milvus_manager=SimpleNamespace(get_collection=lambda: collection),
    )
    install_module(
        monkeypatch,
        "app.services.vector_embedding_service",
        vector_embedding_service=SimpleNamespace(embed_query=lambda query: [0.1, 0.2]),
    )
    install_module(
        monkeypatch,
        "app.services.vector_store_manager",
        vector_store_manager=store_manager,
    )


def install_vector_index_dependencies(monkeypatch) -> None:
    install_document_dependencies(monkeypatch)
    install_module(monkeypatch, "app")
    install_module(monkeypatch, "app.services")
    install_module(monkeypatch, "loguru", logger=NoopLogger())
    install_module(
        monkeypatch,
        "app.services.document_splitter_service",
        document_splitter_service=object(),
    )
    install_module(
        monkeypatch,
        "app.services.vector_store_manager",
        vector_store_manager=object(),
    )


def test_metadata_filter_expr_supports_language_and_and_combination(monkeypatch):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test",
        "app/services/vector_store_manager.py",
    )

    assert (
        module.VectorStoreManager.build_metadata_filter_expr(language="en")
        == 'metadata["language"] == "en"'
    )

    expr = module.VectorStoreManager.build_metadata_filter_expr(
        doc_id="manual-1",
        language="zh",
        retrieval_tiers=["primary", "support"],
        chunk_types="procedure_step",
        chunk_ids=["chunk-1", "chunk-2"],
    )

    assert expr == (
        'metadata["doc_id"] == "manual-1" and '
        'metadata["language"] == "zh" and '
        'metadata["retrieval_tier"] in ["primary", "support"] and '
        'metadata["chunk_type"] == "procedure_step" and '
        'metadata["chunk_id"] in ["chunk-1", "chunk-2"]'
    )


def test_vector_search_methods_pass_language_to_filter_builder(monkeypatch):
    class FakeStoreManager:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def build_metadata_filter_expr(self, **kwargs):
            self.calls.append(kwargs)
            return f'expr language={kwargs.get("language")}'

    class FakeCollection:
        def __init__(self) -> None:
            self.search_kwargs: dict[str, object] | None = None
            self.query_calls: list[dict[str, object]] = []

        def search(self, **kwargs):
            self.search_kwargs = kwargs
            hit = SimpleNamespace(
                distance=0.42,
                entity={
                    "id": "chunk-1",
                    "content": "stored",
                    "metadata": {"text": "metadata text"},
                },
            )
            return [[hit]]

        def query(self, **kwargs):
            self.query_calls.append(kwargs)
            return []

    collection = FakeCollection()
    store_manager = FakeStoreManager()
    install_vector_search_dependencies(monkeypatch, collection, store_manager)
    module = load_module(
        monkeypatch,
        "vector_search_service_under_test_metadata",
        "app/services/vector_search_service.py",
    )
    service = module.VectorSearchService()

    service.search_similar_documents(
        "install battery",
        doc_id="manual-1",
        language="en",
        retrieval_tiers=["primary"],
    )
    service.query_documents(language="zh", chunk_ids=["parent-1"])
    service.query_all_documents(language="en", batch_size=10)

    assert [call["language"] for call in store_manager.calls] == ["en", "zh", "en"]
    assert collection.search_kwargs is not None
    assert collection.search_kwargs["expr"] == "expr language=en"
    assert [call["expr"] for call in collection.query_calls] == [
        "expr language=zh",
        "expr language=en",
    ]


def test_chunk_documents_include_language_metadata_for_structured_and_legacy_records(
    monkeypatch,
    tmp_path,
):
    install_vector_index_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_index_service_under_test",
        "app/services/vector_index_service.py",
    )

    documents = module.VectorIndexService._build_documents_from_chunk_records(
        [
            {
                "text": "Install the batteries.",
                "doc_id": "manual-en",
                "language": "en",
                "chunk_id": "chunk-en-1",
                "retrieval_tier": "primary",
                "chunk_type": "procedure_step",
                "section_path": ["Install"],
            },
            {
                "text": "Install the batteries in zh manual.",
                "doc_id": "manual-zh",
                "language": "zh",
                "chunk_id": "chunk-zh-1",
            },
        ],
        tmp_path / "chunks.jsonl",
    )

    assert [document.metadata["language"] for document in documents] == ["en", "zh"]
