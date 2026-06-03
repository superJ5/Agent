from __future__ import annotations

import importlib.util
import json
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


def test_metadata_filter_expr_supports_single_parent_chunk_id(monkeypatch):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test_single_parent",
        "app/services/vector_store_manager.py",
    )

    assert (
        module.VectorStoreManager.build_metadata_filter_expr(
            parent_chunk_ids="parent-1"
        )
        == 'metadata["parent_chunk_id"] == "parent-1"'
    )


def test_metadata_filter_expr_supports_multiple_parent_chunk_ids(monkeypatch):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test_multiple_parents",
        "app/services/vector_store_manager.py",
    )

    assert (
        module.VectorStoreManager.build_metadata_filter_expr(
            parent_chunk_ids=["parent-1", "parent-2"]
        )
        == 'metadata["parent_chunk_id"] in ["parent-1", "parent-2"]'
    )


def test_metadata_filter_expr_combines_parent_with_existing_filters(monkeypatch):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test_parent_combo",
        "app/services/vector_store_manager.py",
    )

    expr = module.VectorStoreManager.build_metadata_filter_expr(
        doc_id="manual-1",
        language="en",
        retrieval_tiers=["support", "big_support"],
        parent_chunk_ids=["parent-1", "parent-2"],
    )

    assert expr == (
        'metadata["doc_id"] == "manual-1" and '
        'metadata["language"] == "en" and '
        'metadata["retrieval_tier"] in ["support", "big_support"] and '
        'metadata["parent_chunk_id"] in ["parent-1", "parent-2"]'
    )


def test_vector_store_caps_embedding_text_to_provider_limit(monkeypatch):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test_embedding_limit",
        "app/services/vector_store_manager.py",
    )
    document = DocumentStub(
        "fallback text",
        {"index_text": "x" * (module.MAX_EMBEDDING_INPUT_CHARS + 100)},
    )

    embedding_text = module.VectorStoreManager._get_embedding_text(document)

    assert len(embedding_text) == module.MAX_EMBEDDING_INPUT_CHARS


def test_vector_store_writes_embedding_limit_hierarchy_report(monkeypatch, tmp_path):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test_embedding_report",
        "app/services/vector_store_manager.py",
    )

    class FakeEmbeddingService:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_documents(self, texts: list[str]):
            self.calls.append(list(texts))
            if any(len(text) > module.MAX_EMBEDDING_INPUT_CHARS for text in texts):
                raise RuntimeError(
                    "批量嵌入失败: Error code: 400 - "
                    "{'error': {'message': 'Range of input length should be [1, 8192]'}}"
                )
            return [[float(len(text))] for text in texts]

    fake_embedding_service = FakeEmbeddingService()
    monkeypatch.setattr(module, "vector_embedding_service", fake_embedding_service)
    manager = module.VectorStoreManager()
    report_path = tmp_path / "embedding_limit_report.jsonl"
    manager.reset_embedding_limit_report(report_path)
    raw_index_text = "x" * (module.MAX_EMBEDDING_INPUT_CHARS + 1)
    document = DocumentStub(
        "fallback text",
        {
            "doc_id": "manual-en",
            "doc_name": "Camera",
            "chunk_id": "manual-en-001",
            "retrieval_tier": "auxiliary",
            "chunk_type": "aux_navigation",
            "title": "Image Anchor Order",
            "section_path": ["Camera", "Image Anchor Order"],
            "parent_chunk_id": "manual-en-parent",
            "source_file": "data/manuals/reviewed_md/camera.md",
            "source_lines": [10, 20],
            "index_text": raw_index_text,
        },
    )

    embeddings, used_inputs = manager._embed_documents_with_limit_report(
        [document],
        [raw_index_text],
    )

    assert embeddings == [[float(module.MAX_EMBEDDING_INPUT_CHARS)]]
    assert len(used_inputs[0]) == module.MAX_EMBEDDING_INPUT_CHARS
    assert manager.embedding_limit_report_summary()["count"] == 1
    entry = json.loads(report_path.read_text(encoding="utf-8"))
    assert entry["chunk_id"] == "manual-en-001"
    assert entry["retrieval_tier"] == "auxiliary"
    assert entry["chunk_type"] == "aux_navigation"
    assert entry["title"] == "Image Anchor Order"
    assert entry["hierarchy"] == "Camera > Image Anchor Order"
    assert entry["section_depth"] == 2
    assert entry["parent_chunk_id"] == "manual-en-parent"
    assert entry["original_embedding_chars"] == module.MAX_EMBEDDING_INPUT_CHARS + 1
    assert entry["fallback_embedding_chars"] == module.MAX_EMBEDDING_INPUT_CHARS
    assert entry["limit_kind"] == "provider_8192_token_limit_rejected_raw_input"
    assert "Range of input length should be [1, 8192]" in entry["provider_error"]


def test_vector_store_does_not_report_when_provider_accepts_long_text(monkeypatch, tmp_path):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test_no_false_report",
        "app/services/vector_store_manager.py",
    )

    class AcceptingEmbeddingService:
        def embed_documents(self, texts: list[str]):
            return [[float(len(text))] for text in texts]

    monkeypatch.setattr(module, "vector_embedding_service", AcceptingEmbeddingService())
    manager = module.VectorStoreManager()
    report_path = tmp_path / "embedding_limit_report.jsonl"
    manager.reset_embedding_limit_report(report_path)
    raw_index_text = "x" * (module.MAX_EMBEDDING_INPUT_CHARS + 100)
    document = DocumentStub("fallback text", {"index_text": raw_index_text})

    embeddings, used_inputs = manager._embed_documents_with_limit_report(
        [document],
        [raw_index_text],
    )

    assert embeddings == [[float(len(raw_index_text))]]
    assert used_inputs == [raw_index_text]
    assert manager.embedding_limit_report_summary()["count"] == 0
    assert not report_path.exists()


def test_vector_store_can_append_embedding_limit_report(monkeypatch, tmp_path):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test_append_report",
        "app/services/vector_store_manager.py",
    )

    manager = module.VectorStoreManager()
    report_path = tmp_path / "embedding_limit_report.jsonl"
    report_path.write_text('{"chunk_id": "previous"}\n\n', encoding="utf-8")

    manager.append_embedding_limit_report(report_path)
    assert manager.embedding_limit_report_summary()["count"] == 1

    document = DocumentStub(
        "fallback text",
        {
            "doc_id": "manual-en",
            "chunk_id": "manual-en-002",
            "retrieval_tier": "auxiliary",
            "chunk_type": "aux_navigation",
            "section_path": ["Manual", "Image Anchor Order"],
            "index_text": "x" * (module.MAX_EMBEDDING_INPUT_CHARS + 1),
        },
    )
    manager._record_embedding_limit_hit(
        document,
        original_text="x" * (module.MAX_EMBEDDING_INPUT_CHARS + 1),
        fallback_text="x" * module.MAX_EMBEDDING_INPUT_CHARS,
        provider_error="Range of input length should be [1, 8192]",
    )

    lines = [line for line in report_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    assert json.loads(lines[0])["chunk_id"] == "previous"
    assert json.loads(lines[1])["chunk_id"] == "manual-en-002"
    assert manager.embedding_limit_report_summary()["count"] == 2


def test_vector_store_compacts_large_text_fields_before_metadata_insert(monkeypatch):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test_metadata_compaction",
        "app/services/vector_store_manager.py",
    )

    metadata = module.VectorStoreManager._prepare_metadata(
        {
            "chunk_id": "huge-auxiliary",
            "retrieval_tier": "auxiliary",
            "text": "a" * 70000,
            "index_text": "b" * 20000,
        },
        stored_content="a" * module.MAX_CONTENT_BYTES,
        embedding_text="b" * module.MAX_EMBEDDING_INPUT_CHARS,
    )

    assert metadata["chunk_id"] == "huge-auxiliary"
    assert metadata["retrieval_tier"] == "auxiliary"
    assert len(metadata.get("text", "")) <= module.MAX_CONTENT_BYTES
    assert len(metadata.get("index_text", "")) <= module.MAX_EMBEDDING_INPUT_CHARS
    assert len(json.dumps(metadata, ensure_ascii=False).encode("utf-8")) <= (
        module.MAX_METADATA_BYTES
    )


def test_add_documents_skips_single_chunk_when_embedding_retry_fails(monkeypatch, tmp_path):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test_embedding_skip",
        "app/services/vector_store_manager.py",
    )

    class FakeEmbeddingService:
        def embed_documents(self, texts: list[str]):
            if any("bad embedding input" in text for text in texts):
                raise RuntimeError("embedding failed")
            return [[float(len(text))] for text in texts]

    class FakeCollection:
        def __init__(self) -> None:
            self.inserted: list[dict[str, object]] = []
            self.flushed = False

        def insert(self, rows: list[dict[str, object]]):
            self.inserted.extend(rows)

        def flush(self):
            self.flushed = True

    collection = FakeCollection()
    monkeypatch.setattr(module, "vector_embedding_service", FakeEmbeddingService())
    monkeypatch.setattr(
        module,
        "milvus_manager",
        SimpleNamespace(connect=lambda: object(), get_collection=lambda: collection),
    )
    manager = module.VectorStoreManager()
    report_path = tmp_path / "skipped_chunks.jsonl"
    manager.reset_skipped_chunk_report(report_path)

    inserted_ids = manager.add_documents(
        [
            DocumentStub(
                "good content",
                {
                    "doc_id": "manual-en",
                    "chunk_id": "chunk-good",
                    "retrieval_tier": "primary",
                    "chunk_type": "text",
                    "section_path": ["Manual", "Good"],
                    "index_text": "good embedding input",
                },
            ),
            DocumentStub(
                "bad content",
                {
                    "doc_id": "manual-en",
                    "chunk_id": "chunk-bad",
                    "retrieval_tier": "primary",
                    "chunk_type": "text",
                    "section_path": ["Manual", "Bad"],
                    "index_text": "bad embedding input",
                },
            ),
        ]
    )

    assert inserted_ids == ["chunk-good"]
    assert [row["id"] for row in collection.inserted] == ["chunk-good"]
    assert collection.flushed is True
    assert manager.skipped_chunk_report_summary()["count"] == 1
    entry = json.loads(report_path.read_text(encoding="utf-8"))
    assert entry["chunk_id"] == "chunk-bad"
    assert entry["retrieval_tier"] == "primary"
    assert entry["stage"] == "embedding"
    assert entry["retry_attempted"] is True


def test_add_documents_retries_insert_and_skips_only_bad_row(monkeypatch, tmp_path):
    install_vector_store_dependencies(monkeypatch)
    module = load_module(
        monkeypatch,
        "vector_store_manager_under_test_insert_skip",
        "app/services/vector_store_manager.py",
    )

    class FakeEmbeddingService:
        def embed_documents(self, texts: list[str]):
            return [[float(len(text))] for text in texts]

    class FakeCollection:
        def __init__(self) -> None:
            self.inserted: list[dict[str, object]] = []
            self.flushed = False

        def insert(self, rows: list[dict[str, object]]):
            if len(rows) > 1:
                raise RuntimeError("batch insert failed")
            if rows[0]["id"] == "chunk-bad":
                raise RuntimeError("metadata length exceeds max length")
            self.inserted.extend(rows)

        def flush(self):
            self.flushed = True

    collection = FakeCollection()
    monkeypatch.setattr(module, "vector_embedding_service", FakeEmbeddingService())
    monkeypatch.setattr(
        module,
        "milvus_manager",
        SimpleNamespace(connect=lambda: object(), get_collection=lambda: collection),
    )
    manager = module.VectorStoreManager()
    report_path = tmp_path / "skipped_chunks.jsonl"
    manager.reset_skipped_chunk_report(report_path)

    inserted_ids = manager.add_documents(
        [
            DocumentStub(
                "good content",
                {
                    "doc_id": "manual-en",
                    "chunk_id": "chunk-good",
                    "retrieval_tier": "primary",
                    "chunk_type": "text",
                    "section_path": ["Manual", "Good"],
                    "index_text": "good embedding input",
                },
            ),
            DocumentStub(
                "bad content",
                {
                    "doc_id": "manual-en",
                    "chunk_id": "chunk-bad",
                    "retrieval_tier": "auxiliary",
                    "chunk_type": "aux_navigation",
                    "section_path": ["Manual", "Bad"],
                    "index_text": "bad embedding input",
                },
            ),
        ]
    )

    assert inserted_ids == ["chunk-good"]
    assert [row["id"] for row in collection.inserted] == ["chunk-good"]
    assert collection.flushed is True
    assert manager.skipped_chunk_report_summary()["count"] == 1
    entry = json.loads(report_path.read_text(encoding="utf-8"))
    assert entry["chunk_id"] == "chunk-bad"
    assert entry["retrieval_tier"] == "auxiliary"
    assert entry["stage"] == "milvus_insert"
    assert entry["retry_attempted"] is True
    assert "metadata length exceeds max length" in entry["error"]


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
    service.query_documents(language="zh", parent_chunk_ids=["parent-1"])
    service.query_all_documents(
        language="en",
        parent_chunk_ids="parent-2",
        batch_size=10,
    )

    assert [call["language"] for call in store_manager.calls] == ["en", "zh", "en"]
    assert store_manager.calls[1]["parent_chunk_ids"] == ["parent-1"]
    assert store_manager.calls[2]["parent_chunk_ids"] == "parent-2"
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


def test_index_script_builds_final_error_summary(monkeypatch, tmp_path):
    install_module(monkeypatch, "app")
    install_module(monkeypatch, "app.services")
    install_module(
        monkeypatch,
        "app.services.vector_index_service",
        vector_index_service=object(),
    )
    install_module(
        monkeypatch,
        "app.services.vector_store_manager",
        vector_store_manager=object(),
    )
    module = load_module(
        monkeypatch,
        "index_manual_chunks_under_test_summary",
        "scripts/index_manual_chunks.py",
    )
    skipped_report = tmp_path / "skipped.jsonl"
    skipped_report.write_text(
        json.dumps(
            {
                "doc_id": "manual-en",
                "chunk_id": "chunk-bad",
                "retrieval_tier": "primary",
                "stage": "embedding",
                "error": "retry failed",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    embedding_report = tmp_path / "embedding.jsonl"
    embedding_report.write_text(
        json.dumps(
            {
                "doc_id": "manual-en",
                "chunk_id": "chunk-long",
                "retrieval_tier": "auxiliary",
                "limit_kind": "provider_8192_token_limit_rejected_raw_input",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.build_final_error_summary(
        failed_files={"bad-file.jsonl": "whole file failed"},
        skipped_chunk_report=skipped_report,
        embedding_limit_report=embedding_report,
    )

    assert summary["has_errors"] is True
    assert summary["file_error_count"] == 1
    assert summary["single_chunk_error_count"] == 1
    assert summary["embedding_limit_fallback_count"] == 1
    assert summary["file_errors"][0]["file"] == "bad-file.jsonl"
    assert summary["single_chunk_errors"][0]["chunk_id"] == "chunk-bad"
    assert summary["embedding_limit_fallbacks"][0]["chunk_id"] == "chunk-long"
