from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

from app.retrieval import bm25_provider
from app.retrieval.bm25_provider import JiebaBM25Provider


@dataclass
class Result:
    id: str
    content: str = ""
    score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


class FakeBM25Okapi:
    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        query_terms = set(query_tokens)
        return [
            float(sum(1 for token in document if token in query_terms))
            for document in self.corpus
        ]


def whitespace_tokenizer(text: str) -> list[str]:
    return text.split()


def make_provider() -> JiebaBM25Provider:
    return JiebaBM25Provider(tokenizer=whitespace_tokenizer, bm25_factory=FakeBM25Okapi)


def test_english_tokenizer_removes_stop_words_and_keeps_required_terms() -> None:
    provider = make_provider()

    assert provider._tokenize(  # noqa: SLF001 - focused tokenizer regression coverage.
        "How to install the battery pack?",
        language="en",
    ) == ["install", "battery", "pack"]
    assert provider._tokenize(  # noqa: SLF001
        "What should not be washed?",
        language="en",
    ) == ["not", "washed", "wash"]
    assert provider._tokenize(  # noqa: SLF001
        "use set run turn change check open close start stop",
        language="en",
    ) == [
        "use",
        "set",
        "run",
        "turn",
        "change",
        "check",
        "open",
        "close",
        "start",
        "stop",
    ]


def test_english_tokenizer_adds_light_morphology_forms() -> None:
    provider = make_provider()

    assert provider._tokenize(  # noqa: SLF001
        "steps cleaning cleaned cleans batteries boxes connected connecting",
        language="en",
    ) == [
        "steps",
        "step",
        "cleaning",
        "clean",
        "cleaned",
        "clean",
        "cleans",
        "clean",
        "batteries",
        "battery",
        "boxes",
        "box",
        "connected",
        "connect",
        "connecting",
        "connect",
    ]


def test_chinese_tokenizer_keeps_configured_jieba_path() -> None:
    provider = make_provider()

    assert provider._tokenize(  # noqa: SLF001
        "\u7535\u6c60 \u5b89\u88c5",
        language="zh",
    ) == ["\u7535\u6c60", "\u5b89\u88c5"]


def test_build_and_search_returns_positive_bm25_hits() -> None:
    provider = make_provider()
    docs = [
        Result(
            "chunk-1",
            "battery install battery",
            metadata={
                "chunk_id": "chunk-1",
                "doc_id": "manual-a",
                "retrieval_tier": "primary",
                "chunk_type": "procedure",
            },
        ),
        Result(
            "chunk-2",
            metadata={
                "chunk_id": "chunk-2",
                "text": "battery safety",
                "title": "Safety",
                "section_path": ["Operations", "Battery"],
                "index_text": "ppe checklist",
            },
        ),
        Result("empty", metadata={"chunk_id": "empty"}),
    ]

    provider.build_index(docs)
    results = provider.search("battery", top_k=2)

    assert [result.id for result in results] == ["chunk-1", "chunk-2"]
    assert results[0] is docs[0]
    assert results[0].score == 2.0
    assert provider.document_count == 3


def test_english_bm25_matches_morphology_variants() -> None:
    provider = make_provider()
    docs = [
        Result(
            "cleaning",
            "cleaning the snowmobile",
            metadata={"language": "en", "retrieval_tier": "primary"},
        ),
        Result(
            "batteries",
            "replace batteries in the tracker",
            metadata={"language": "en", "retrieval_tier": "primary"},
        ),
        Result(
            "unrelated",
            "install the regulator",
            metadata={"language": "en", "retrieval_tier": "primary"},
        ),
    ]

    provider.build_index(docs)

    clean_results = provider.search("clean snowmobile", top_k=5, language="en")
    battery_results = provider.search("battery tracker", top_k=5, language="en")

    assert [result.id for result in clean_results] == ["cleaning"]
    assert [result.id for result in battery_results] == ["batteries"]


def test_search_filters_by_doc_id() -> None:
    provider = make_provider()
    provider.build_index(
        [
            Result("manual-a", "battery", metadata={"doc_id": "manual-a"}),
            Result("manual-b", "battery", metadata={"doc_id": "manual-b"}),
        ]
    )

    results = provider.search("battery", top_k=5, doc_id="manual-b")

    assert [result.id for result in results] == ["manual-b"]


def test_search_filters_by_tier_and_chunk_type() -> None:
    provider = make_provider()
    provider.build_index(
        [
            Result(
                "procedure-primary",
                "battery",
                metadata={"retrieval_tier": "primary", "chunk_type": "procedure"},
            ),
            Result(
                "overview-support",
                "battery",
                metadata={"retrieval_tier": "support", "chunk_type": "overview"},
            ),
            Result(
                "legal-primary",
                "battery",
                metadata={"retrieval_tier": "primary", "chunk_type": "legal"},
            ),
        ]
    )

    results = provider.search(
        "battery",
        top_k=5,
        retrieval_tiers=["primary"],
        chunk_types=["procedure"],
    )

    assert [result.id for result in results] == ["procedure-primary"]


def test_search_always_excludes_big_support_tier() -> None:
    provider = make_provider()
    provider.build_index(
        [
            Result(
                "primary",
                "battery",
                metadata={"retrieval_tier": "primary"},
            ),
            Result(
                "big-support",
                "battery battery battery",
                metadata={"retrieval_tier": "big_support"},
            ),
            Result(
                "support",
                "battery",
                metadata={"retrieval_tier": "support"},
            ),
        ]
    )

    unfiltered_results = provider.search("battery", top_k=5)
    support_results = provider.search(
        "battery",
        top_k=5,
        retrieval_tiers=["support", "big_support"],
    )

    assert [result.id for result in unfiltered_results] == ["primary", "support"]
    assert [result.id for result in support_results] == ["support"]


def test_search_does_not_treat_long_index_text_as_big_support() -> None:
    provider = make_provider()
    long_index_text = "battery " * 2000
    provider.build_index(
        [
            Result(
                "long-primary",
                metadata={
                    "retrieval_tier": "primary",
                    "index_text": long_index_text,
                },
            ),
            Result(
                "long-support",
                metadata={
                    "retrieval_tier": "support",
                    "index_text": long_index_text,
                },
            ),
        ]
    )

    results = provider.search("battery", top_k=5)

    assert [result.id for result in results] == ["long-primary", "long-support"]


def test_search_does_not_make_auxiliary_recallable_without_matching_filter() -> None:
    provider = make_provider()
    provider.build_index(
        [
            Result(
                "primary",
                "battery",
                metadata={"retrieval_tier": "primary"},
            ),
            Result(
                "auxiliary",
                "battery battery",
                metadata={"retrieval_tier": "auxiliary"},
            ),
        ]
    )

    results = provider.search("battery", top_k=5, retrieval_tiers=["primary"])

    assert [result.id for result in results] == ["primary"]


def test_search_filters_by_language() -> None:
    provider = make_provider()
    provider.build_index(
        [
            Result(
                "english",
                "the battery pack install procedure",
                metadata={"language": "en"},
            ),
            Result(
                "chinese",
                "battery pack install procedure",
                metadata={"language": "zh"},
            ),
        ]
    )

    results = provider.search("How to install the battery pack?", top_k=5, language="en")

    assert [result.id for result in results] == ["english"]


def test_search_without_build_returns_empty() -> None:
    provider = make_provider()

    assert provider.search("battery", top_k=3) == []


def test_init_bm25_provider_registers_provider(monkeypatch: Any) -> None:
    docs = [Result("chunk-1", "battery")]
    registered: list[object] = []

    class FakeProvider:
        def __init__(self) -> None:
            self.docs: list[Result] = []
            self.document_count = 0

        def build_index(self, all_results: list[Result]) -> None:
            self.docs = list(all_results)
            self.document_count = len(self.docs)

    recall_module = ModuleType("app.retrieval.recall")
    recall_module.set_bm25_provider = registered.append  # type: ignore[attr-defined]
    vector_module = ModuleType("app.services.vector_search_service")
    vector_module.vector_search_service = SimpleNamespace(  # type: ignore[attr-defined]
        query_all_documents=lambda: docs
    )
    monkeypatch.setitem(sys.modules, "app.retrieval.recall", recall_module)
    monkeypatch.setitem(sys.modules, "app.services.vector_search_service", vector_module)
    monkeypatch.setattr(bm25_provider, "JiebaBM25Provider", FakeProvider)

    bm25_provider.init_bm25_provider()

    assert isinstance(registered[-1], FakeProvider)
    assert registered[-1].docs == docs


def test_init_bm25_provider_failure_clears_provider_and_does_not_raise(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    registered: list[object | None] = []

    recall_module = ModuleType("app.retrieval.recall")
    recall_module.set_bm25_provider = registered.append  # type: ignore[attr-defined]
    vector_module = ModuleType("app.services.vector_search_service")
    vector_module.vector_search_service = SimpleNamespace(  # type: ignore[attr-defined]
        query_all_documents=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setitem(sys.modules, "app.retrieval.recall", recall_module)
    monkeypatch.setitem(sys.modules, "app.services.vector_search_service", vector_module)

    bm25_provider.init_bm25_provider()

    assert registered == [None]
    assert "BM25 provider initialization failed" in caplog.text
