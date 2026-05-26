from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_query_understanding_module():
    module_names = (
        "app",
        "app.retrieval",
        "app.retrieval.schemas",
        "app.retrieval.diagnostics",
    )
    sentinel = object()
    previous = {name: sys.modules.get(name, sentinel) for name in module_names}

    app_module = ModuleType("app")
    app_module.__path__ = [str(ROOT / "app")]
    retrieval_module = ModuleType("app.retrieval")
    retrieval_module.__path__ = [str(ROOT / "app" / "retrieval")]
    sys.modules["app"] = app_module
    sys.modules["app.retrieval"] = retrieval_module

    schemas_module = load_module(
        "app.retrieval.schemas",
        ROOT / "app" / "retrieval" / "schemas.py",
    )
    load_module(
        "app.retrieval.diagnostics",
        ROOT / "app" / "retrieval" / "diagnostics.py",
    )
    query_module = load_module(
        "query_understanding_under_test",
        ROOT / "app" / "retrieval" / "query_understanding.py",
    )

    for name, module in previous.items():
        if module is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

    return query_module, schemas_module


qu, schemas = load_query_understanding_module()

DocCandidate = schemas.DocCandidate
IntentCandidate = schemas.IntentCandidate
QueryTerm = schemas.QueryTerm
RetrievalDiagnostics = schemas.RetrievalDiagnostics
RetrievalOptions = schemas.RetrievalOptions


@pytest.fixture(autouse=True)
def isolate_legacy_dependencies(monkeypatch):
    monkeypatch.setattr(qu, "_knowledge_tool", lambda: None)
    monkeypatch.setattr(qu, "LLM_ANALYZER", None)


def options(strategy: str) -> RetrievalOptions:
    return RetrievalOptions(intent_strategy=strategy)


def profile(
    doc_id: str,
    doc_name: str,
    *,
    terms: tuple[str, ...] = (),
    pic_prefixes: tuple[str, ...] = (),
    chunk_families: tuple[str, ...] = (),
):
    return SimpleNamespace(
        doc_id=doc_id,
        doc_name=doc_name,
        terms=terms,
        pic_prefixes=pic_prefixes,
        chunk_types=(),
        chunk_families=chunk_families,
        source_issue_flags=(),
        chunk_count=1,
    )


def test_resolve_enabled_strategies_expands_known_modes():
    assert qu.resolve_enabled_strategies("none") == ()
    assert qu.resolve_enabled_strategies("rules") == ("rules",)
    assert qu.resolve_enabled_strategies(" hybrid ") == (
        "profile",
        "rules",
        "summary",
        "llm",
    )
    assert qu.resolve_enabled_strategies("unknown") == (
        "profile",
        "rules",
        "summary",
        "llm",
    )


def test_analyze_query_none_records_trace_and_extracts_terms():
    diagnostics = RetrievalDiagnostics(request_id="req-none")

    analysis = qu.analyze_query("battery install", options("none"), diagnostics)

    assert analysis.strategy == "none"
    assert analysis.intent_candidates == []
    assert analysis.doc_candidates == []
    assert [term.term for term in analysis.query_terms] == ["battery", "install"]
    assert diagnostics.trace["query_understanding"]["enabled_strategies"] == []
    assert diagnostics.trace["query_understanding"]["analysis"]["strategy"] == "none"


def test_profile_strategy_uses_existing_profiles_for_docs_intents_and_terms(monkeypatch):
    doc_name = "\u84dd\u7259\u6fc0\u5149\u9f20\u6807\u624b\u518c"
    monkeypatch.setattr(
        qu,
        "_load_doc_profiles",
        lambda: (
            profile(
                "manual_mouse",
                doc_name,
                terms=("\u84dd\u7259", "\u9f20\u6807", "\u914d\u5bf9"),
                chunk_families=("procedure", "component"),
            ),
        ),
    )

    part = qu.analyze_with_profile(f"{doc_name}\u5982\u4f55\u914d\u5bf9")

    assert part.doc_candidates[0].doc_id == "manual_mouse"
    assert part.doc_candidates[0].score >= 0.75
    assert {candidate.intent for candidate in part.intent_candidates} == {
        "procedure",
        "component",
    }
    assert any(term.term == "\u914d\u5bf9" and "profile" in term.source for term in part.query_terms)


def test_rules_strategy_detects_pic_intent_and_explicit_doc_candidate(monkeypatch):
    monkeypatch.setattr(
        qu,
        "_load_doc_profiles",
        lambda: (
            profile(
                "manual_chair",
                "\u4eba\u4f53\u5de5\u5b66\u6905\u624b\u518c",
                pic_prefixes=("manual02_",),
            ),
        ),
    )

    part = qu.analyze_with_rules(
        "Manual02_14 \u8fd9\u5f20\u56fe\u7247\u5bf9\u5e94\u54ea\u4e2a\u90e8\u4ef6"
    )

    assert part.intent_candidates == [
        IntentCandidate(
            intent="image_trace",
            score=0.98,
            source="rules",
            reason="high-confidence keyword or identifier rule",
        )
    ]
    assert part.doc_candidates == [
        DocCandidate(
            doc_id="manual_chair",
            score=0.95,
            source="rules",
            reason="picture id matched profile prefix: manual02_14",
        )
    ]


def test_summary_strategy_reads_offline_summary_index(monkeypatch):
    monkeypatch.setattr(
        qu,
        "load_summary_index",
        lambda: [
            {
                "doc_id": "manual_oven",
                "doc_name": "oven manual",
                "summary": "preheat temperature and cooking setup",
                "terms": ["preheat", "temperature"],
            }
        ],
    )

    part = qu.analyze_with_summary("preheat oven")

    assert part.doc_candidates[0].doc_id == "manual_oven"
    assert part.doc_candidates[0].source == "summary"
    assert "matched summary terms" in part.doc_candidates[0].reason
    assert any(term.term == "preheat" and "summary" in term.source for term in part.query_terms)


def test_summary_strategy_can_use_existing_profile_information(monkeypatch):
    monkeypatch.setattr(qu, "load_summary_index", lambda: [])
    monkeypatch.setattr(
        qu,
        "_load_doc_profiles",
        lambda: (
            profile(
                "manual_keyboard",
                "keyboard manual",
                terms=("macro", "shortcut"),
                chunk_families=("procedure",),
            ),
        ),
    )

    part = qu.analyze_with_summary("macro shortcut")

    assert part.doc_candidates[0].doc_id == "manual_keyboard"
    assert part.doc_candidates[0].source == "summary"


def test_llm_strategy_degrades_with_warning_when_analyzer_unavailable():
    diagnostics = RetrievalDiagnostics(request_id="req-llm")

    analysis = qu.analyze_query("battery install", options("llm"), diagnostics)

    assert analysis.intent_candidates == []
    assert analysis.doc_candidates == []
    assert qu.LLM_UNAVAILABLE_WARNING in analysis.warnings
    assert diagnostics.warnings == [qu.LLM_UNAVAILABLE_WARNING]
    assert diagnostics.trace["query_understanding"]["llm"]["warnings"] == [
        qu.LLM_UNAVAILABLE_WARNING
    ]


def test_hybrid_strategy_merges_candidates_and_records_each_part(monkeypatch):
    monkeypatch.setattr(
        qu,
        "analyze_with_profile",
        lambda query: qu.QueryAnalysisPart(
            strategy="profile",
            intent_candidates=[IntentCandidate("procedure", 0.5, "profile", "family")],
            doc_candidates=[DocCandidate("manual_a", 0.6, "profile", "profile match")],
            query_terms=[QueryTerm("battery", "profile", 0.8)],
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        qu,
        "analyze_with_rules",
        lambda query: qu.QueryAnalysisPart(
            strategy="rules",
            intent_candidates=[IntentCandidate("procedure", 0.7, "rules", "keyword")],
            doc_candidates=[DocCandidate("manual_a", 0.5, "rules", "explicit hint")],
            query_terms=[QueryTerm("install", "rules", 0.9)],
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        qu,
        "analyze_with_summary",
        lambda query: qu.QueryAnalysisPart(
            strategy="summary",
            intent_candidates=[],
            doc_candidates=[DocCandidate("manual_b", 0.2, "summary", "weak overlap")],
            query_terms=[],
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        qu,
        "analyze_with_llm",
        lambda query: qu.QueryAnalysisPart(
            strategy="llm",
            intent_candidates=[IntentCandidate("safety", 0.6, "llm", "model hint")],
            doc_candidates=[],
            query_terms=[],
            warnings=["llm soft warning"],
        ),
    )
    diagnostics = RetrievalDiagnostics(request_id="req-hybrid")

    analysis = qu.analyze_query("battery install", options("hybrid"), diagnostics)

    procedure = next(candidate for candidate in analysis.intent_candidates if candidate.intent == "procedure")
    manual_a = next(candidate for candidate in analysis.doc_candidates if candidate.doc_id == "manual_a")

    assert procedure.score == pytest.approx(0.85)
    assert procedure.source == "profile+rules"
    assert manual_a.score == pytest.approx(0.8)
    assert manual_a.source == "profile+rules"
    assert "filter=strong" in manual_a.reason
    assert analysis.primary_doc_id == "manual_a"
    assert analysis.warnings == ["llm soft warning"]
    assert diagnostics.warnings == ["llm soft warning"]
    assert set(diagnostics.trace["query_understanding"]) >= {
        "profile",
        "rules",
        "summary",
        "llm",
        "analysis",
    }


def test_low_confidence_doc_candidates_are_marked_sort_only():
    analysis = qu.merge_analysis_parts(
        "battery install",
        "rules",
        [
            qu.QueryAnalysisPart(
                strategy="rules",
                intent_candidates=[],
                doc_candidates=[DocCandidate("manual_weak", 0.2, "rules", "weak hint")],
                query_terms=[],
                warnings=[],
            )
        ],
    )

    assert analysis.primary_doc_id == "manual_weak"
    assert analysis.doc_candidates[0].score == 0.2
    assert "filter=sort_only" in analysis.doc_candidates[0].reason


def test_strategy_exception_is_captured_in_diagnostics(monkeypatch):
    def fail_rules(query: str) -> qu.QueryAnalysisPart:
        raise RuntimeError("broken rules")

    monkeypatch.setattr(qu, "analyze_with_rules", fail_rules)
    diagnostics = RetrievalDiagnostics(request_id="req-error")

    analysis = qu.analyze_query("install battery", options("rules"), diagnostics)

    assert analysis.intent_candidates == []
    assert analysis.doc_candidates == []
    assert analysis.query_terms
    assert diagnostics.warnings == ["rules analysis failed: broken rules"]
    assert diagnostics.trace["query_understanding"]["rules"] == {"error": "broken rules"}
