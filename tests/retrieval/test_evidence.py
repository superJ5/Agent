from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Result:
    id: str
    content: str = ""
    score: float = 0.0
    metadata: dict[str, object] | None = None


class DocumentStub:
    def __init__(self, page_content: str, metadata: dict[str, object]) -> None:
        self.page_content = page_content
        self.metadata = metadata


class FakeVectorSearchService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.query_all_calls: list[dict[str, object]] = []
        self.parents: list[Result] = []
        self.children: list[Result] = []
        self.error: Exception | None = None

    def query_documents(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return list(self.parents)

    def query_all_documents(self, **kwargs):
        self.query_all_calls.append(kwargs)
        if self.error is not None:
            raise self.error

        parent_ids = _values(kwargs.get("parent_chunk_ids"))
        tiers = _values(kwargs.get("retrieval_tiers"))
        doc_id = kwargs.get("doc_id")
        language = kwargs.get("language")
        results: list[Result] = []
        for child in self.children:
            metadata = child.metadata or {}
            if parent_ids and metadata.get("parent_chunk_id") not in parent_ids:
                continue
            if tiers and metadata.get("retrieval_tier") not in tiers:
                continue
            if doc_id and metadata.get("doc_id") != doc_id:
                continue
            if language and metadata.get("language") != language:
                continue
            results.append(child)
        return results


def _values(value: object) -> set[object]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return set(value)
    return {value}


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_evidence_module():
    module_names = (
        "app",
        "app.retrieval",
        "app.retrieval.schemas",
        "app.services",
        "app.services.vector_search_service",
        "langchain_core",
        "langchain_core.documents",
    )
    sentinel = object()
    previous = {name: sys.modules.get(name, sentinel) for name in module_names}

    app_module = ModuleType("app")
    app_module.__path__ = [str(ROOT / "app")]
    retrieval_module = ModuleType("app.retrieval")
    retrieval_module.__path__ = [str(ROOT / "app" / "retrieval")]
    services_module = ModuleType("app.services")
    services_module.__path__ = []

    fake_service = FakeVectorSearchService()
    vector_module = ModuleType("app.services.vector_search_service")
    vector_module.SearchResult = Result
    vector_module.vector_search_service = fake_service

    langchain_module = ModuleType("langchain_core")
    langchain_module.__path__ = []
    documents_module = ModuleType("langchain_core.documents")
    documents_module.Document = DocumentStub

    sys.modules["app"] = app_module
    sys.modules["app.retrieval"] = retrieval_module
    sys.modules["app.services"] = services_module
    sys.modules["app.services.vector_search_service"] = vector_module
    sys.modules["langchain_core"] = langchain_module
    sys.modules["langchain_core.documents"] = documents_module

    schemas_module = load_module(
        "app.retrieval.schemas",
        ROOT / "app" / "retrieval" / "schemas.py",
    )
    retrieval_module.schemas = schemas_module
    services_module.vector_search_service = vector_module
    langchain_module.documents = documents_module

    evidence_module = load_module(
        "evidence_under_test",
        ROOT / "app" / "retrieval" / "evidence.py",
    )

    for name, module in previous.items():
        if module is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

    return evidence_module, schemas_module, fake_service


def make_candidate(
    schemas,
    result: Result,
    *,
    channels: set[str] | None = None,
    merged_score: float = 0.0,
    diagnostics: dict[str, object] | None = None,
):
    return schemas.RecallCandidate(
        result=result,
        chunk_id=str((result.metadata or {}).get("chunk_id") or result.id),
        recall_channels=channels or {"vector"},
        channel_scores=dict.fromkeys(channels or {"vector"}, 1.0),
        merged_score=merged_score,
        diagnostics=diagnostics or {},
    )


def test_build_retrieval_bundle_fetches_only_primary_parents_and_traces_images():
    evidence, schemas, service = load_evidence_module()
    primary = Result(
        "child-id",
        "child content",
        0.2,
        {
            "chunk_id": "child-1",
            "doc_id": "manual-1",
            "retrieval_tier": "primary",
            "chunk_type": "procedure_step",
            "parent_chunk_id": "parent-1",
            "pic_ids": ["pic-child"],
            "image_paths": ["child.png"],
            "section_path": ["Safety", "Startup"],
            "source_lines": [12, 13],
        },
    )
    already_support = Result(
        "support-child",
        "already support",
        0.3,
        {
            "chunk_id": "support-child",
            "retrieval_tier": "support",
            "parent_chunk_id": "must-not-query",
        },
    )
    parent = Result(
        "parent-id",
        "parent content",
        0.0,
        {
            "chunk_id": "parent-1",
            "doc_id": "manual-1",
            "retrieval_tier": "support",
            "chunk_type": "parent_context",
            "parent_chunk_id": "grandparent-not-queried",
            "pic_ids": ["pic-parent"],
            "image_paths": ["parent.png"],
            "section_path": ["Safety"],
            "source_lines": [1, 20],
        },
    )
    service.parents = [
        Result("wrong-tier", metadata={"chunk_id": "parent-1", "retrieval_tier": "primary"}),
        Result("sibling", metadata={"chunk_id": "sibling", "retrieval_tier": "support"}),
        parent,
    ]

    rerank_result = schemas.RerankResult(
        candidates=[
            make_candidate(schemas, primary, diagnostics={"lexical_score": 9.0}),
            make_candidate(schemas, already_support, channels={"bm25"}),
        ],
        provider="lexical",
        fallback_used=True,
        warnings=["reranker fallback"],
        score_field="lexical_score",
    )
    diagnostics = schemas.RetrievalDiagnostics(request_id="req-1")

    bundle = evidence.build_retrieval_bundle(
        query="startup safety",
        analysis=SimpleNamespace(primary_intent="procedure"),
        rerank_result=rerank_result,
        options=SimpleNamespace(top_k=2),
        diagnostics=diagnostics,
    )

    assert isinstance(bundle, schemas.RetrievalBundle)
    assert bundle.hits == [primary, already_support]
    assert bundle.support_hits == [parent]
    assert service.calls == [
        {
            "retrieval_tiers": ["support", "big_support"],
            "chunk_ids": ["parent-1"],
            "language": "en",
            "limit": 1,
        }
    ]
    assert "doc_id" not in service.calls[0]
    assert diagnostics.warnings == ["reranker fallback"]
    assert diagnostics.summary["top_hits"][0] == {
        "chunk_id": "child-1",
        "score": 9.0,
        "channels": ["vector"],
    }
    assert diagnostics.trace["support_parent_ids"] == ["parent-1"]
    assert diagnostics.trace["support_parent_request"]["language"] == "en"
    assert diagnostics.trace["support_parent_hits"][0]["chunk_id"] == "parent-1"
    assert diagnostics.trace["evidence_images"]["pic_ids"] == ["pic-child", "pic-parent"]
    assert diagnostics.trace["evidence_images"]["image_paths"] == [
        "child.png",
        "parent.png",
    ]


def test_fetch_parent_support_hits_skips_missing_or_non_primary_parent_ids():
    evidence, schemas, service = load_evidence_module()
    hits = [
        Result(
            "primary-no-parent",
            metadata={
                "chunk_id": "primary-no-parent",
                "retrieval_tier": "primary",
            },
        ),
        Result(
            "support-with-parent",
            metadata={
                "chunk_id": "support-with-parent",
                "retrieval_tier": "support",
                "parent_chunk_id": "ignored-parent",
            },
        ),
    ]
    diagnostics = schemas.RetrievalDiagnostics(request_id="req-2")

    assert evidence.fetch_parent_support_hits(hits, diagnostics) == []
    assert service.calls == []
    assert diagnostics.trace["support_parent_ids"] == []
    assert diagnostics.trace["support_parent_hits"] == []


def test_fetch_parent_support_hits_passes_hit_language_to_parent_query():
    evidence, schemas, service = load_evidence_module()
    primary = Result(
        "child-id",
        "child content",
        0.2,
        {
            "chunk_id": "child-1",
            "retrieval_tier": "primary",
            "parent_chunk_id": "parent-1",
            "language": "zh",
        },
    )
    parent = Result(
        "parent-id",
        "parent content",
        0.0,
        {"chunk_id": "parent-1", "retrieval_tier": "support", "language": "zh"},
    )
    service.parents = [parent]
    diagnostics = schemas.RetrievalDiagnostics(request_id="req-language")

    assert evidence.fetch_parent_support_hits([primary], diagnostics) == [parent]
    assert service.calls == [
        {
            "retrieval_tiers": ["support", "big_support"],
            "chunk_ids": ["parent-1"],
            "limit": 1,
            "language": "zh",
        }
    ]
    assert diagnostics.trace["support_parent_request"]["language"] == "zh"


def test_big_support_expands_top_primary_descendants_and_direct_support_parent():
    evidence, schemas, service = load_evidence_module()
    primary = Result(
        "child-id",
        "child content",
        0.2,
        {
            "chunk_id": "child-1",
            "doc_id": "manual-1",
            "retrieval_tier": "primary",
            "parent_chunk_id": "big-1",
            "language": "en",
        },
    )
    big_parent = Result(
        "big-id",
        "big support heading only",
        0.0,
        {
            "chunk_id": "big-1",
            "doc_id": "manual-1",
            "retrieval_tier": "big_support",
            "language": "en",
        },
    )
    support_parent = Result(
        "support-id",
        "ordinary support context",
        0.0,
        {
            "chunk_id": "support-1",
            "doc_id": "manual-1",
            "retrieval_tier": "support",
            "parent_chunk_id": "big-1",
            "language": "en",
            "chunk_index": 2,
        },
    )
    direct_primary = Result(
        "direct-primary-id",
        "direct primary",
        0.0,
        {
            "chunk_id": "direct-primary",
            "doc_id": "manual-1",
            "retrieval_tier": "primary",
            "parent_chunk_id": "big-1",
            "language": "en",
            "chunk_index": 3,
        },
    )
    primary_a = Result(
        "primary-a-id",
        "primary a",
        0.0,
        {
            "chunk_id": "primary-a",
            "doc_id": "manual-1",
            "retrieval_tier": "primary",
            "parent_chunk_id": "support-1",
            "language": "en",
            "chunk_index": 4,
        },
    )
    primary_b = Result(
        "primary-b-id",
        "primary b",
        0.0,
        {
            "chunk_id": "primary-b",
            "doc_id": "manual-1",
            "retrieval_tier": "primary",
            "parent_chunk_id": "support-1",
            "language": "en",
            "chunk_index": 5,
        },
    )
    primary_c = Result(
        "primary-c-id",
        "primary c",
        0.0,
        {
            "chunk_id": "primary-c",
            "doc_id": "manual-1",
            "retrieval_tier": "primary",
            "parent_chunk_id": "support-1",
            "language": "en",
            "chunk_index": 6,
        },
    )
    service.parents = [big_parent, support_parent]
    service.children = [support_parent, direct_primary, primary_a, primary_b, primary_c]

    def fake_rerank(query, candidates, options, diagnostics):
        del query, options
        diagnostics.trace["reranker"] = {"provider": "fake"}
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        return schemas.RerankResult(
            candidates=[
                by_id["primary-b"],
                by_id["primary-c"],
                by_id["direct-primary"],
            ],
            provider="fake",
        )

    evidence.rerank_candidates = fake_rerank
    diagnostics = schemas.RetrievalDiagnostics(request_id="req-big")
    bundle = evidence.build_retrieval_bundle(
        query="find safety",
        analysis=SimpleNamespace(primary_intent="general"),
        rerank_result=schemas.RerankResult(
            candidates=[make_candidate(schemas, primary)],
            provider="lexical",
        ),
        options=SimpleNamespace(top_k=1),
        diagnostics=diagnostics,
    )

    assert bundle.hits == [primary]
    assert [hit.metadata["chunk_id"] for hit in bundle.support_hits] == [
        "big-1",
        "primary-b",
        "primary-c",
        "direct-primary",
        "support-1",
    ]
    assert service.calls == [
        {
            "retrieval_tiers": ["support", "big_support"],
            "chunk_ids": ["big-1"],
            "language": "en",
            "limit": 1,
        },
        {
            "retrieval_tiers": ["support"],
            "chunk_ids": ["support-1", "big-1"],
            "limit": 2,
            "language": "en",
        },
    ]
    assert [call["parent_chunk_ids"] for call in service.query_all_calls] == [
        ["big-1"],
        ["support-1"],
    ]
    assert diagnostics.trace["big_support_parent_ids"] == ["big-1"]
    assert diagnostics.trace["big_support_descendant_count"] == {"big-1": 4}
    assert diagnostics.trace["big_support_expanded_hits"][0]["selected"][0][
        "chunk_id"
    ] == "primary-b"


def test_big_support_descendant_collection_caps_at_one_hundred_primary_chunks():
    evidence, schemas, service = load_evidence_module()
    big_parent = Result(
        "big-id",
        "big support heading only",
        0.0,
        {
            "chunk_id": "big-1",
            "doc_id": "manual-1",
            "retrieval_tier": "big_support",
            "language": "en",
        },
    )
    service.children = [
        Result(
            f"primary-{index}",
            metadata={
                "chunk_id": f"primary-{index}",
                "doc_id": "manual-1",
                "retrieval_tier": "primary",
                "parent_chunk_id": "big-1",
                "language": "en",
                "chunk_index": index,
            },
        )
        for index in range(101)
    ]
    diagnostics = schemas.RetrievalDiagnostics(request_id="req-cap")

    descendants, truncated = evidence.collect_big_support_descendants(
        big_parent,
        diagnostics=diagnostics,
        language="en",
    )

    assert len(descendants) == 100
    assert truncated is True
    assert descendants[0].metadata["chunk_id"] == "primary-0"
    assert descendants[-1].metadata["chunk_id"] == "primary-99"


def test_big_support_descendant_collection_stops_after_four_layers():
    evidence, schemas, service = load_evidence_module()
    big_parent = Result(
        "big-id",
        "big support heading only",
        0.0,
        {
            "chunk_id": "big-1",
            "doc_id": "manual-1",
            "retrieval_tier": "big_support",
            "language": "en",
        },
    )
    service.children = [
        Result(
            "support-1-id",
            metadata={
                "chunk_id": "support-1",
                "doc_id": "manual-1",
                "retrieval_tier": "support",
                "parent_chunk_id": "big-1",
                "language": "en",
            },
        ),
        Result(
            "support-2-id",
            metadata={
                "chunk_id": "support-2",
                "doc_id": "manual-1",
                "retrieval_tier": "support",
                "parent_chunk_id": "support-1",
                "language": "en",
            },
        ),
        Result(
            "support-3-id",
            metadata={
                "chunk_id": "support-3",
                "doc_id": "manual-1",
                "retrieval_tier": "support",
                "parent_chunk_id": "support-2",
                "language": "en",
            },
        ),
        Result(
            "support-4-id",
            metadata={
                "chunk_id": "support-4",
                "doc_id": "manual-1",
                "retrieval_tier": "support",
                "parent_chunk_id": "support-3",
                "language": "en",
            },
        ),
        Result(
            "too-deep-primary-id",
            metadata={
                "chunk_id": "too-deep-primary",
                "doc_id": "manual-1",
                "retrieval_tier": "primary",
                "parent_chunk_id": "support-4",
                "language": "en",
            },
        ),
    ]
    diagnostics = schemas.RetrievalDiagnostics(request_id="req-depth")

    descendants, truncated = evidence.collect_big_support_descendants(
        big_parent,
        diagnostics=diagnostics,
        language="en",
    )

    assert descendants == []
    assert truncated is False
    assert [call["parent_chunk_ids"] for call in service.query_all_calls] == [
        ["big-1"],
        ["support-1"],
        ["support-2"],
        ["support-3"],
    ]


def test_big_support_parent_without_descendants_records_zero_count():
    evidence, schemas, service = load_evidence_module()
    primary = Result(
        "child-id",
        "child content",
        0.2,
        {
            "chunk_id": "child-1",
            "doc_id": "manual-1",
            "retrieval_tier": "primary",
            "parent_chunk_id": "big-1",
            "language": "en",
        },
    )
    big_parent = Result(
        "big-id",
        "big support heading only",
        0.0,
        {
            "chunk_id": "big-1",
            "doc_id": "manual-1",
            "retrieval_tier": "big_support",
            "language": "en",
        },
    )
    service.parents = [big_parent]
    diagnostics = schemas.RetrievalDiagnostics(request_id="req-empty")

    support_hits = evidence.fetch_parent_support_hits(
        [primary],
        diagnostics,
        language="en",
        query="query",
        options=SimpleNamespace(top_k=1),
    )

    assert support_hits == [big_parent]
    assert diagnostics.trace["big_support_descendant_count"] == {"big-1": 0}
    assert diagnostics.trace["big_support_expanded_hits"][0]["selected"] == []


def test_big_support_expand_false_keeps_parent_and_skips_descendant_queries():
    evidence, schemas, service = load_evidence_module()
    primary = Result(
        "child-id",
        "child content",
        0.2,
        {
            "chunk_id": "child-1",
            "doc_id": "manual-1",
            "retrieval_tier": "primary",
            "parent_chunk_id": "big-1",
            "language": "en",
        },
    )
    big_parent = Result(
        "big-id",
        "big support heading only",
        0.0,
        {
            "chunk_id": "big-1",
            "doc_id": "manual-1",
            "retrieval_tier": "big_support",
            "language": "en",
        },
    )
    service.parents = [big_parent]
    service.children = [
        Result(
            "primary-a-id",
            metadata={
                "chunk_id": "primary-a",
                "doc_id": "manual-1",
                "retrieval_tier": "primary",
                "parent_chunk_id": "big-1",
                "language": "en",
            },
        )
    ]
    diagnostics = schemas.RetrievalDiagnostics(request_id="req-skip")

    support_hits = evidence.fetch_parent_support_hits(
        [primary],
        diagnostics,
        language="en",
        query="query",
        options=SimpleNamespace(top_k=1),
        expand_big_support=False,
    )

    assert support_hits == [big_parent]
    assert service.query_all_calls == []
    assert diagnostics.trace["big_support_expansion_skipped"] == [
        {"parent_id": "big-1", "reason": "expand_big_support_false"}
    ]


def test_support_parent_failure_keeps_primary_hits_and_records_warning():
    evidence, schemas, service = load_evidence_module()
    primary = Result(
        "child-id",
        "child content",
        0.2,
        {
            "chunk_id": "child-1",
            "retrieval_tier": "primary",
            "parent_chunk_id": "parent-1",
        },
    )
    service.error = RuntimeError("backend down")
    diagnostics = schemas.RetrievalDiagnostics(request_id="req-3")

    bundle = evidence.build_retrieval_bundle(
        query="query",
        analysis=SimpleNamespace(primary_intent=None),
        rerank_result=schemas.RerankResult(
            candidates=[make_candidate(schemas, primary)],
            provider="lexical",
        ),
        options=SimpleNamespace(top_k=1),
        diagnostics=diagnostics,
    )

    assert bundle.intent == "general"
    assert bundle.hits == [primary]
    assert bundle.support_hits == []
    assert diagnostics.warnings == ["support expansion failed: backend down"]
    assert diagnostics.trace["support_parent_error"] == "backend down"


def test_document_and_evidence_payload_preserve_metadata_fields():
    evidence, schemas, _ = load_evidence_module()
    primary = Result(
        "result-id",
        "evidence text",
        0.42,
        {
            "chunk_id": "chunk-1",
            "doc_id": "manual-1",
            "doc_name": "Manual One",
            "title": "Install",
            "section_path": ("Chapter", "Install"),
            "retrieval_tier": "primary",
            "chunk_type": "procedure_step",
            "parent_chunk_id": "parent-1",
            "pic_ids": ["pic-1", "pic-2"],
            "image_paths": ["image-1.png"],
            "source_file": "manual.pdf",
            "source_lines": (10, 12),
            "source_quality": "clean",
            "source_issue_flags": ["ocr"],
            "source_issue_note": "reviewed",
        },
    )
    support = Result(
        "support-id",
        "support text",
        0.0,
        {
            "chunk_id": "parent-1",
            "retrieval_tier": "support",
            "pic_ids": ["pic-2", "pic-3"],
            "image_paths": ["image-1.png", "image-2.png"],
        },
    )
    bundle = schemas.RetrievalBundle(
        intent="procedure",
        retrieval_stage="hybrid_search",
        hits=[primary],
        support_hits=[support],
        warnings=["fallback"],
    )

    document = evidence.search_result_to_document(primary)
    assert document.page_content == "evidence text"
    assert document.metadata["chunk_id"] == "chunk-1"
    assert document.metadata["score"] == 0.42
    assert document.metadata["evidence_schema_version"] == evidence.EVIDENCE_SCHEMA_VERSION

    payload = evidence.bundle_to_evidence_payload(bundle, query="install")
    hit = payload["hits"][0]
    assert hit["retrieval_tier"] == "primary"
    assert hit["chunk_type"] == "procedure_step"
    assert hit["parent_chunk_id"] == "parent-1"
    assert hit["section_path"] == ["Chapter", "Install"]
    assert hit["source_lines"] == [10, 12]
    assert hit["pic_ids"] == ["pic-1", "pic-2"]
    assert hit["image_paths"] == ["image-1.png"]
    assert payload["summary"]["pic_ids"] == ["pic-1", "pic-2", "pic-3"]
    assert payload["summary"]["image_paths"] == ["image-1.png", "image-2.png"]
    assert payload["warnings"] == ["fallback"]


def test_format_bundle_includes_primary_support_and_source_fields():
    evidence, schemas, _ = load_evidence_module()
    primary = Result(
        "child-id",
        "child content",
        0.2,
        {
            "chunk_id": "child-1",
            "doc_name": "Manual One",
            "title": "Startup",
            "section_path": ["Chapter 1", "Startup"],
            "retrieval_tier": "primary",
            "chunk_type": "procedure_step",
            "parent_chunk_id": "parent-1",
            "pic_ids": ["pic-1"],
            "image_paths": ["image-1.png"],
            "source_lines": [8, 9],
        },
    )
    support = Result(
        "parent-id",
        "parent content",
        0.0,
        {"chunk_id": "parent-1", "retrieval_tier": "support"},
    )
    bundle = schemas.RetrievalBundle(
        intent="procedure",
        retrieval_stage="hybrid_search",
        hits=[primary],
        support_hits=[support],
        warnings=[],
    )

    formatted = evidence.format_bundle(bundle, query="startup")

    assert "[Primary hits]" in formatted
    assert "[Support context]" in formatted
    assert "Manual: Manual One" in formatted
    assert "Section path: Chapter 1 > Startup" in formatted
    assert "Parent chunk ID: parent-1" in formatted
    assert "Source lines: [8, 9]" in formatted
    assert "- ![pic-1](image-1.png)" in formatted
    assert "child content" in formatted
    assert "parent content" in formatted


def test_unique_flatten_preserves_first_seen_order():
    evidence, _, _ = load_evidence_module()

    assert evidence.unique_flatten([["a", "b"], ("a", "c"), "d", None, 3]) == [
        "a",
        "b",
        "c",
        "d",
        3,
    ]
