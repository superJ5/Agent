from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


class NoopLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class Result:
    def __init__(self, id: str = "chunk", content: str = "", score: float = 0.0):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = {"chunk_id": id}


class DocumentStub:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def install_module(monkeypatch, name: str, **attrs):
    module = ModuleType(name)
    if name in {"app", "app.models", "app.retrieval", "app.services", "langchain_core"}:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def load_knowledge_tool(monkeypatch):
    install_module(monkeypatch, "app")
    install_module(monkeypatch, "app.config", config=SimpleNamespace(rag_top_k=3))
    install_module(monkeypatch, "app.models")
    install_module(
        monkeypatch,
        "app.models.response",
        sanitize_summary_metadata=lambda metadata: dict(metadata or {}),
    )
    install_module(monkeypatch, "app.retrieval")
    install_module(monkeypatch, "app.services")
    install_module(
        monkeypatch,
        "app.services.vector_search_service",
        SearchResult=Result,
        vector_search_service=SimpleNamespace(),
    )
    install_module(monkeypatch, "langchain_core")
    install_module(monkeypatch, "langchain_core.documents", Document=DocumentStub)
    install_module(monkeypatch, "langchain_core.tools", tool=lambda **kwargs: lambda func: func)
    install_module(monkeypatch, "loguru", logger=NoopLogger())

    spec = importlib.util.spec_from_file_location(
        "knowledge_tool_query_preprocessing_under_test",
        ROOT / "app" / "tools" / "knowledge_tool.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "knowledge_tool_query_preprocessing_under_test", module)
    spec.loader.exec_module(module)
    return module


def test_detect_lang_uses_any_chinese_character_rule(monkeypatch):
    module = load_knowledge_tool(monkeypatch)

    assert module.detect_lang("\u4f7f\u7528 VR-300 \u5934\u663e\u5982\u4f55\u64cd\u4f5c") == "zh"
    assert module.detect_lang("How to install the battery pack?") == "en"
    assert module.detect_lang("VR-300 DCB101") == "en"


def test_english_query_terms_regex_stopwords_and_protected_terms(monkeypatch):
    module = load_knowledge_tool(monkeypatch)

    terms = module.extract_query_terms(
        "How to not use or no set/run turn-change check open close start stop "
        "the DCB101 remote-control batteries?"
    )

    assert terms == [
        "not",
        "use",
        "no",
        "set",
        "run",
        "turn",
        "change",
        "check",
        "open",
        "close",
        "start",
        "stop",
        "dcb101",
        "remote",
        "control",
        "batteries",
    ]
    assert "how" not in terms
    assert "to" not in terms
    assert "the" not in terms


def test_english_expand_queries_only_original_and_keyword_join(monkeypatch):
    module = load_knowledge_tool(monkeypatch)
    query = "How to install the DCB101 remote-control batteries?"

    terms = module.extract_query_terms(query)
    variants = module.expand_queries(query, "procedure", terms)

    assert terms == ["install", "dcb101", "remote", "control", "batteries"]
    assert "remote control" not in terms
    assert "battery" not in terms
    assert variants == [query, "install dcb101 remote control batteries"]


def test_chinese_query_terms_keep_jieba_branch(monkeypatch):
    module = load_knowledge_tool(monkeypatch)
    calls = []
    jieba_module = ModuleType("jieba")
    analyse_module = ModuleType("jieba.analyse")

    def extract_tags(query, topK, withWeight):
        calls.append((query, topK, withWeight))
        return ["\u51b7\u673a", "\u542f\u52a8"]

    analyse_module.extract_tags = extract_tags
    jieba_module.analyse = analyse_module
    monkeypatch.setitem(sys.modules, "jieba", jieba_module)
    monkeypatch.setitem(sys.modules, "jieba.analyse", analyse_module)
    monkeypatch.setattr(module, "extract_pic_id", lambda query: None)
    monkeypatch.setattr(module, "match_profile_terms_in_query", lambda normalized, limit=4: [])

    terms = module.extract_query_terms("VR-300 \u51b7\u673a\u542f\u52a8")

    assert terms == ["\u51b7\u673a", "\u542f\u52a8"]
    assert calls == [("VR-300 \u51b7\u673a\u542f\u52a8", 8, False)]


def test_english_critical_term_groups_empty_but_chinese_retained(monkeypatch):
    module = load_knowledge_tool(monkeypatch)

    assert module.query_critical_term_groups(module.normalize_text("\u51b7\u673a\u542f\u52a8")) == [
        ("\u51b7\u673a",)
    ]
    assert module.query_critical_term_groups(module.normalize_text("OCR source missing")) == []
