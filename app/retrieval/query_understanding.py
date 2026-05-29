"""Query understanding strategies for the retrieval pipeline."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any, cast

from app.retrieval.diagnostics import RetrievalDiagnostics
from app.retrieval.schemas import (
    DocCandidate,
    IntentCandidate,
    QueryAnalysis,
    QueryTerm,
    RetrievalOptions,
)

KNOWN_STRATEGIES = ("none", "profile", "rules", "summary", "llm", "hybrid")
HYBRID_STRATEGIES = ("profile", "rules", "summary", "llm")
MAX_CANDIDATES = 8
MAX_TERMS = 16
DOC_CANDIDATE_MIN_SCORE = 0.18
INTENT_CANDIDATE_MIN_SCORE = 0.18
DOC_FILTER_CONFIDENCE_THRESHOLD = 0.7

# 硬编码的关键词 → doc_id 映射表。
# 极度保守：只写绝对确定的映射，宁缺勿错。后续根据评测结果手动补充。
RULES_DOC_MAPPING: dict[tuple[str, ...], str] = {
    ("水泵",): "manual_4e007096",
    ("电钻",): "manual_60e374ac",
    ("吹风机",): "manual_84d80d19",
}
LLM_UNAVAILABLE_WARNING = "LLM query analysis unavailable; skipped"
LLM_ANALYZER: Callable[[str], Any] | None = None


@dataclass
class QueryAnalysisPart:
    """Internal normalized result from one query-understanding strategy."""

    strategy: str
    intent_candidates: list[IntentCandidate]
    doc_candidates: list[DocCandidate]
    query_terms: list[QueryTerm]
    warnings: list[str] | None = None


def analyze_query(
    query: str,
    options: RetrievalOptions,
    diagnostics: RetrievalDiagnostics,
) -> QueryAnalysis:
    """Analyze a query with the selected strategy and record trace diagnostics."""
    strategy = _normalize_strategy(getattr(options, "intent_strategy", "hybrid"))
    language = _detect_lang(query)
    trace = _ensure_query_understanding_trace(diagnostics)
    trace["strategy"] = strategy
    trace["language"] = language

    if strategy == "none":
        analysis = QueryAnalysis(
            query=query,
            strategy="none",
            intent_candidates=[],
            doc_candidates=[],
            query_terms=extract_terms(query),
            warnings=[],
            language=language,
        )
        trace["enabled_strategies"] = []
        trace["analysis"] = _serialize_analysis(analysis)
        return analysis

    enabled_strategies = resolve_enabled_strategies(strategy)
    trace["enabled_strategies"] = list(enabled_strategies)
    parts: list[QueryAnalysisPart] = []

    for enabled_strategy in enabled_strategies:
        try:
            part = _run_strategy(enabled_strategy, query)
        except Exception as exc:  # pragma: no cover - exact failures are integration-specific.
            message = f"{enabled_strategy} analysis failed: {exc}"
            _add_diagnostic_warning(diagnostics, message)
            trace[enabled_strategy] = {"error": str(exc)}
            continue

        parts.append(part)
        trace[enabled_strategy] = _serialize_part(part)
        for warning in part.warnings or []:
            _add_diagnostic_warning(diagnostics, warning)

    analysis = merge_analysis_parts(query=query, strategy=strategy, parts=parts)
    trace["analysis"] = _serialize_analysis(analysis)
    return analysis


def resolve_enabled_strategies(intent_strategy: str) -> tuple[str, ...]:
    """Expand a configured intent strategy into concrete strategy names."""
    strategy = _normalize_strategy(intent_strategy)
    if strategy == "none":
        return ()
    if strategy == "hybrid":
        return HYBRID_STRATEGIES
    return (strategy,)


def analyze_with_profile(query: str) -> QueryAnalysisPart:
    """Use indexed document profiles to infer document and term candidates."""
    normalized_query = _normalize_text(query)
    query_terms = extract_terms(query)
    normalized_terms = [_normalize_text(term.term) for term in query_terms if term.term]
    pic_id = _legacy_extract_pic_id(query)
    normalized_pic_id = _normalize_text(pic_id or "")

    doc_candidates: list[DocCandidate] = []
    profile_terms: list[QueryTerm] = []
    family_scores: dict[str, float] = {}
    profiles = _load_doc_profiles()

    for profile in profiles:
        score, reasons, matched_terms = _score_profile(
            profile=profile,
            normalized_query=normalized_query,
            normalized_terms=normalized_terms,
            normalized_pic_id=normalized_pic_id,
        )
        if score < DOC_CANDIDATE_MIN_SCORE:
            continue

        doc_candidates.append(
            DocCandidate(
                doc_id=str(getattr(profile, "doc_id", "")),
                score=score,
                source="profile",
                reason="; ".join(reasons),
            )
        )
        for term in matched_terms[:6]:
            profile_terms.append(QueryTerm(term=term, source="profile", weight=0.8))

        for family in getattr(profile, "chunk_families", ()) or ():
            intent = _family_to_intent(str(family))
            if not intent:
                continue
            family_scores[intent] = max(family_scores.get(intent, 0.0), min(0.7, score * 0.65))

    intent_candidates = [
        IntentCandidate(
            intent=intent,
            score=score,
            source="profile",
            reason="matched document profile families",
        )
        for intent, score in family_scores.items()
        if score >= INTENT_CANDIDATE_MIN_SCORE
    ]

    return QueryAnalysisPart(
        strategy="profile",
        intent_candidates=sorted(intent_candidates, key=lambda item: item.score, reverse=True),
        doc_candidates=sorted(doc_candidates, key=lambda item: item.score, reverse=True),
        query_terms=_dedupe_terms([*query_terms, *profile_terms])[:MAX_TERMS],
        warnings=[],
    )


def analyze_with_rules(query: str) -> QueryAnalysisPart:
    """Use conservative high-certainty rules to infer intent and explicit docs."""
    intent = _detect_intent_with_local_rules(query)
    intent_candidates: list[IntentCandidate] = []
    if intent and intent != "general":
        score = _intent_rule_score(query, intent)
        intent_candidates.append(
            IntentCandidate(
                intent=intent,
                score=score,
                source="rules",
                reason="high-confidence keyword or identifier rule",
            )
        )

    doc_candidates = _rules_doc_candidates(query)
    return QueryAnalysisPart(
        strategy="rules",
        intent_candidates=intent_candidates,
        doc_candidates=doc_candidates,
        query_terms=extract_terms(query),
        warnings=[],
    )


def _rules_doc_candidates(query: str) -> list[DocCandidate]:
    """Match query against the hardcoded RULES_DOC_MAPPING table."""
    normalized = _normalize_text(query)
    candidates: list[DocCandidate] = [
        DocCandidate(
            doc_id=candidate.doc_id,
            score=candidate.score or 0.95,
            source="rules",
            reason=candidate.reason,
        )
        for candidate in _explicit_doc_candidates(query)
    ]
    seen: set[str] = set()
    seen.update(candidate.doc_id for candidate in candidates)
    for keywords, doc_id in RULES_DOC_MAPPING.items():
        if doc_id in seen:
            continue
        if any(kw in normalized for kw in keywords):
            seen.add(doc_id)
            candidates.append(
                DocCandidate(
                    doc_id=doc_id,
                    score=0.95,
                    source="rules",
                    reason=f"hardcoded keyword match: {', '.join(keywords)}",
                )
            )
    return candidates


def analyze_with_summary(query: str) -> QueryAnalysisPart:
    """Use an offline document summary index to produce doc candidates."""
    summaries = load_summary_index() or _profile_summary_index()
    query_terms = extract_terms(query)
    normalized_query = _normalize_text(query)
    normalized_terms = [_normalize_text(term.term) for term in query_terms if term.term]

    doc_candidates: list[DocCandidate] = []
    summary_terms: list[QueryTerm] = []
    for item in summaries:
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            continue
        haystack = _normalize_text(
            " ".join(
                str(value or "")
                for value in (
                    item.get("doc_name"),
                    item.get("title"),
                    item.get("summary"),
                    " ".join(str(term) for term in item.get("terms", []) or []),
                )
            )
        )
        if not haystack:
            continue

        score = 0.0
        reasons: list[str] = []
        matched_terms: list[str] = []
        for term in normalized_terms:
            if not term:
                continue
            if term in haystack:
                score += 0.16 if len(term) >= 4 else 0.1
                matched_terms.append(term)
        if normalized_query and normalized_query in haystack:
            score += 0.4
            reasons.append("query phrase matched summary")
        if matched_terms:
            reasons.append(f"matched summary terms: {', '.join(matched_terms[:6])}")

        score = min(0.92, score)
        if score < DOC_CANDIDATE_MIN_SCORE:
            continue

        doc_candidates.append(
            DocCandidate(
                doc_id=doc_id,
                score=score,
                source="summary",
                reason="; ".join(reasons) or "summary term overlap",
            )
        )
        summary_terms.extend(
            QueryTerm(term=term, source="summary", weight=0.7) for term in matched_terms[:4]
        )

    return QueryAnalysisPart(
        strategy="summary",
        intent_candidates=[],
        doc_candidates=sorted(doc_candidates, key=lambda item: item.score, reverse=True),
        query_terms=_dedupe_terms([*query_terms, *summary_terms])[:MAX_TERMS],
        warnings=[],
    )


def analyze_with_llm(query: str) -> QueryAnalysisPart:
    """Use an optional injected LLM analyzer, without constructing one per request."""
    if LLM_ANALYZER is None:
        return QueryAnalysisPart(
            strategy="llm",
            intent_candidates=[],
            doc_candidates=[],
            query_terms=[],
            warnings=[LLM_UNAVAILABLE_WARNING],
        )

    raw_result = LLM_ANALYZER(query)
    return _coerce_llm_result(raw_result)


def extract_terms(query: str) -> list[QueryTerm]:
    """Extract query terms while preserving the schema-level QueryTerm contract."""
    terms: list[str]
    try:
        terms = [str(term) for term in _legacy_extract_query_terms(query) if str(term).strip()]
    except Exception:
        terms = _extract_terms_with_local_rules(query)

    return _dedupe_terms(QueryTerm(term=term, source="query", weight=1.0) for term in terms)[
        :MAX_TERMS
    ]


def merge_analysis_parts(
    query: str,
    strategy: str,
    parts: Sequence[QueryAnalysisPart],
) -> QueryAnalysis:
    """Merge strategy parts into a single QueryAnalysis."""
    intent_candidates = _merge_intent_candidates(
        candidate for part in parts for candidate in part.intent_candidates
    )
    doc_candidates = _merge_doc_candidates(
        candidate for part in parts for candidate in part.doc_candidates
    )
    query_terms = _dedupe_terms(
        [
            *extract_terms(query),
            *(term for part in parts for term in part.query_terms),
        ]
    )[:MAX_TERMS]
    warnings = _dedupe_strings(
        warning for part in parts for warning in (part.warnings or []) if warning
    )

    return QueryAnalysis(
        query=query,
        strategy=_normalize_strategy(strategy),
        intent_candidates=intent_candidates,
        doc_candidates=doc_candidates,
        query_terms=query_terms,
        warnings=warnings,
        language=_detect_lang(query),
    )


def load_summary_index() -> list[dict[str, Any]]:
    """Load an optional offline summary index from configured or conventional files."""
    path_candidates = _summary_index_path_candidates()
    for path in path_candidates:
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        return _normalize_summary_index(raw)
    return []


def _run_strategy(strategy: str, query: str) -> QueryAnalysisPart:
    runners = {
        "profile": analyze_with_profile,
        "rules": analyze_with_rules,
        "summary": analyze_with_summary,
        "llm": analyze_with_llm,
    }
    return runners[strategy](query)


def _normalize_strategy(intent_strategy: str | None) -> str:
    strategy = str(intent_strategy or "hybrid").strip().lower()
    if strategy not in KNOWN_STRATEGIES:
        return "hybrid"
    return strategy


def _ensure_query_understanding_trace(diagnostics: RetrievalDiagnostics) -> dict[str, Any]:
    trace = getattr(diagnostics, "trace", None)
    if trace is None:
        diagnostics.trace = {}
        trace = diagnostics.trace
    if not isinstance(trace, dict):
        return {}
    query_trace = trace.setdefault("query_understanding", {})
    return cast(dict[str, Any], query_trace) if isinstance(query_trace, dict) else {}


def _add_diagnostic_warning(diagnostics: RetrievalDiagnostics, message: str) -> None:
    warnings = getattr(diagnostics, "warnings", None)
    if isinstance(warnings, list) and message in warnings:
        return

    add_warning = getattr(diagnostics, "add_warning", None)
    if callable(add_warning):
        add_warning(message)
        return
    if warnings is None:
        diagnostics.warnings = []
        warnings = diagnostics.warnings
    warnings.append(message)


def _score_profile(
    profile: Any,
    normalized_query: str,
    normalized_terms: Sequence[str],
    normalized_pic_id: str,
) -> tuple[float, list[str], list[str]]:
    score = 0.0
    reasons: list[str] = []
    matched_terms: list[str] = []

    pic_prefixes = tuple(str(prefix) for prefix in (getattr(profile, "pic_prefixes", ()) or ()))
    if normalized_pic_id and any(normalized_pic_id.startswith(prefix) for prefix in pic_prefixes):
        score += 1.0
        reasons.append(f"picture id matched profile prefix: {normalized_pic_id}")
        matched_terms.append(normalized_pic_id)

    doc_name = _normalize_text(str(getattr(profile, "doc_name", "") or ""))
    if doc_name and doc_name in normalized_query:
        score += 0.85
        reasons.append("doc name appeared in query")
        matched_terms.append(doc_name)

    doc_id = _normalize_text(str(getattr(profile, "doc_id", "") or ""))
    if doc_id and doc_id in normalized_query:
        score += 0.75
        reasons.append("doc id appeared in query")
        matched_terms.append(doc_id)

    profile_terms = [
        _normalize_text(str(term))
        for term in (getattr(profile, "terms", ()) or ())
        if _normalize_text(str(term))
    ]
    query_term_bonus = 0.0
    for term in profile_terms[:256]:
        if term and term in normalized_query:
            query_term_bonus += 0.12 if len(term) >= 4 else 0.08
            matched_terms.append(term)
    if query_term_bonus:
        score += min(0.45, query_term_bonus)
        reasons.append("profile terms appeared in query")

    profile_term_set = set(profile_terms)
    overlap_bonus = 0.0
    for term in normalized_terms:
        if term in profile_term_set:
            overlap_bonus += 0.1
            matched_terms.append(term)
            continue
        if any(
            len(term) >= 2
            and len(profile_term) >= 2
            and (term in profile_term or profile_term in term)
            for profile_term in profile_terms[:32]
        ):
            overlap_bonus += 0.06
            matched_terms.append(term)
    if overlap_bonus:
        score += min(0.3, overlap_bonus)
        reasons.append("query terms overlapped profile terms")

    return min(1.0, score), reasons or ["profile overlap"], _dedupe_strings(matched_terms)


def _family_to_intent(family: str) -> str | None:
    normalized = family.strip().lower()
    if normalized == "image":
        return "image_trace"
    if normalized == "ocr":
        return "ocr_audit"
    if normalized in {
        "component",
        "procedure",
        "legal",
        "safety",
        "troubleshooting",
        "overview",
    }:
        return normalized
    return None


def _explicit_doc_candidates(query: str) -> list[DocCandidate]:
    """Match query against document profiles (doc_id, doc_name, pic_prefix)."""
    normalized_query = _normalize_text(query)
    candidates: list[DocCandidate] = []
    seen: set[str] = set()
    pic_id = _legacy_extract_pic_id(query)
    normalized_pic_id = _normalize_text(pic_id or "")

    for profile in _load_doc_profiles():
        doc_id = str(getattr(profile, "doc_id", "") or "")
        if not doc_id or doc_id in seen:
            continue

        normalized_doc_id = _normalize_text(doc_id)
        doc_name = _normalize_text(str(getattr(profile, "doc_name", "") or ""))
        pic_prefixes = tuple(str(prefix) for prefix in (getattr(profile, "pic_prefixes", ()) or ()))
        if normalized_pic_id and any(normalized_pic_id.startswith(prefix) for prefix in pic_prefixes):
            seen.add(doc_id)
            candidates.append(
                DocCandidate(
                    doc_id=doc_id,
                    score=0.95,
                    source="profile",
                    reason=f"picture id matched profile prefix: {normalized_pic_id}",
                )
            )
            continue
        if normalized_doc_id and normalized_doc_id in normalized_query:
            seen.add(doc_id)
            candidates.append(
                DocCandidate(
                    doc_id=doc_id,
                    score=0.9,
                    source="profile",
                    reason="doc id appeared in query",
                )
            )
            continue
        if doc_name and doc_name in normalized_query:
            seen.add(doc_id)
            candidates.append(
                DocCandidate(
                    doc_id=doc_id,
                    score=0.85,
                    source="profile",
                    reason="doc name appeared in query",
                )
            )

    return candidates


def _intent_rule_score(query: str, intent: str | None) -> float:
    if not intent or intent == "general":
        return 0.0
    if intent == "image_trace" and _legacy_extract_pic_id(query):
        return 0.98
    scores = {
        "image_trace": 0.86,
        "ocr_audit": 0.86,
        "safety": 0.84,
        "troubleshooting": 0.84,
        "legal": 0.82,
        "procedure": 0.8,
        "overview": 0.78,
        "component": 0.76,
    }
    return scores.get(intent, 0.0)


def _detect_intent_with_local_rules(query: str) -> str:
    normalized = _normalize_text(query)
    keyword_groups = (
        ("image_trace", ("pic", "image", "figure", "diagram", "图片", "图示")),
        ("ocr_audit", ("ocr", "source", "rawtext", "原文", "识别")),
        ("safety", ("safety", "warning", "caution", "安全", "警告")),
        (
            "troubleshooting",
            ("fault", "error", "broken", "repair", "故障", "排查"),
        ),
        ("legal", ("warranty", "policy", "legal", "保修", "声明")),
        (
            "procedure",
            ("howto", "how", "install", "replace", "setup", "如何", "步骤"),
        ),
        ("overview", ("overview", "toc", "chapter", "目录", "概览")),
        (
            "component",
            ("where", "button", "indicator", "port", "part", "按钮", "部件"),
        ),
    )
    for intent, keywords in keyword_groups:
        if any(keyword in normalized for keyword in keywords):
            return intent
    return "general"


def _merge_intent_candidates(candidates: Iterable[IntentCandidate]) -> list[IntentCandidate]:
    """Deduplicate intent candidates by intent label, merge sources."""
    grouped: dict[str, list[IntentCandidate]] = {}
    for candidate in candidates:
        if not candidate.intent:
            continue
        key = str(candidate.intent)
        grouped.setdefault(key, []).append(candidate)

    merged: list[IntentCandidate] = []
    for intent, items in grouped.items():
        sources = "+".join(_dedupe_strings(str(item.source) for item in items))
        reasons = "; ".join(_dedupe_strings(str(item.reason) for item in items))
        score = _fuse_scores(float(item.score or 0.0) for item in items)
        merged.append(
            IntentCandidate(
                intent=intent,
                score=score,
                source=sources,
                reason=reasons,
            )
        )
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:MAX_CANDIDATES]


def _merge_doc_candidates(candidates: Iterable[DocCandidate]) -> list[DocCandidate]:
    """Deduplicate doc candidates by doc_id, merge sources."""
    grouped: dict[str, list[DocCandidate]] = {}
    for candidate in candidates:
        if not candidate.doc_id:
            continue
        key = str(candidate.doc_id)
        grouped.setdefault(key, []).append(candidate)

    merged: list[DocCandidate] = []
    for doc_id, items in grouped.items():
        sources = "+".join(_dedupe_strings(str(item.source) for item in items))
        base_reasons = _dedupe_strings(str(item.reason) for item in items)
        score = _fuse_scores(float(item.score or 0.0) for item in items)
        filter_label = (
            "filter=strong"
            if score >= DOC_FILTER_CONFIDENCE_THRESHOLD
            else "filter=sort_only"
        )
        reasons = "; ".join([*base_reasons, filter_label])
        merged.append(
            DocCandidate(
                doc_id=doc_id,
                score=score,
                source=sources,
                reason=reasons,
            )
        )
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:MAX_CANDIDATES]


def _fuse_scores(scores: Iterable[float]) -> float:
    normalized_scores = [max(0.0, min(1.0, float(score))) for score in scores]
    if not normalized_scores:
        return 0.0
    fused = 1.0 - reduce(mul, (1.0 - score for score in normalized_scores), 1.0)
    return round(min(1.0, fused), 4)


def _dedupe_terms(terms: Iterable[QueryTerm]) -> list[QueryTerm]:
    by_key: dict[str, QueryTerm] = {}
    source_by_key: dict[str, list[str]] = {}
    for term in terms:
        if not term.term:
            continue
        key = _normalize_text(str(term.term))
        if not key:
            continue
        current = by_key.get(key)
        source_by_key.setdefault(key, [])
        if getattr(term, "source", None):
            source_by_key[key].append(str(term.source))
        if current is None or float(term.weight) > float(current.weight):
            by_key[key] = term

    deduped: list[QueryTerm] = []
    for key, term in by_key.items():
        sources = "+".join(_dedupe_strings(source_by_key.get(key, []))) or str(term.source)
        deduped.append(QueryTerm(term=term.term, source=sources, weight=term.weight))
    deduped.sort(key=lambda item: (-float(item.weight), str(item.term)))
    return deduped


def _dedupe_strings(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _coerce_llm_result(raw_result: Any) -> QueryAnalysisPart:
    if isinstance(raw_result, QueryAnalysisPart):
        return raw_result
    if isinstance(raw_result, QueryAnalysis):
        return QueryAnalysisPart(
            strategy="llm",
            intent_candidates=list(raw_result.intent_candidates),
            doc_candidates=list(raw_result.doc_candidates),
            query_terms=list(raw_result.query_terms),
            warnings=list(raw_result.warnings),
        )
    if not isinstance(raw_result, Mapping):
        raise TypeError("llm analyzer must return QueryAnalysisPart, QueryAnalysis, or dict")

    return QueryAnalysisPart(
        strategy="llm",
        intent_candidates=[
            IntentCandidate(
                intent=str(item.get("intent")),
                score=float(item.get("score", 0.0)),
                source=str(item.get("source") or "llm"),
                reason=str(item.get("reason") or ""),
            )
            for item in raw_result.get("intent_candidates", []) or []
            if item.get("intent")
        ],
        doc_candidates=[
            DocCandidate(
                doc_id=str(item.get("doc_id")),
                score=float(item.get("score", 0.0)),
                source=str(item.get("source") or "llm"),
                reason=str(item.get("reason") or ""),
            )
            for item in raw_result.get("doc_candidates", []) or []
            if item.get("doc_id")
        ],
        query_terms=[
            QueryTerm(
                term=str(item.get("term")),
                source=str(item.get("source") or "llm"),
                weight=float(item.get("weight", 1.0)),
            )
            for item in raw_result.get("query_terms", []) or []
            if item.get("term")
        ],
        warnings=[str(warning) for warning in raw_result.get("warnings", []) or []],
    )


def _normalize_summary_index(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    if isinstance(raw, Mapping):
        items = raw.get("documents", raw)
        if isinstance(items, list):
            return [dict(item) for item in items if isinstance(item, Mapping)]
        if isinstance(items, Mapping):
            normalized = []
            for doc_id, value in items.items():
                if isinstance(value, Mapping):
                    item = dict(value)
                    item.setdefault("doc_id", doc_id)
                    normalized.append(item)
                else:
                    normalized.append({"doc_id": doc_id, "summary": str(value)})
            return normalized
    return []


def _profile_summary_index() -> list[dict[str, Any]]:
    """Expose existing profiles as a lightweight summary-like index."""
    summaries: list[dict[str, Any]] = []
    for profile in _load_doc_profiles():
        doc_id = str(getattr(profile, "doc_id", "") or "").strip()
        if not doc_id:
            continue
        terms = [str(term) for term in (getattr(profile, "terms", ()) or ()) if term]
        families = [str(family) for family in (getattr(profile, "chunk_families", ()) or ()) if family]
        summaries.append(
            {
                "doc_id": doc_id,
                "doc_name": str(getattr(profile, "doc_name", "") or ""),
                "title": str(getattr(profile, "doc_name", "") or ""),
                "summary": " ".join([*terms[:32], *families]),
                "terms": terms,
            }
        )
    return summaries


def _summary_index_path_candidates() -> list[Path]:
    config: Any = None
    try:
        from app.config import config as app_config
        config = app_config
    except Exception:
        config = None

    configured_paths = []
    if config is not None:
        for attr_name in (
            "rag_summary_index_path",
            "rag_doc_summary_index_path",
            "rag_document_summary_path",
        ):
            value = getattr(config, attr_name, None)
            if value:
                configured_paths.append(Path(str(value)))

    root = Path(__file__).resolve().parents[2]
    conventional_paths = [
        root / "data" / "doc_summaries.json",
        root / "data" / "retrieval_doc_summaries.json",
        root / "doc" / "doc_summaries.json",
    ]
    return [*configured_paths, *conventional_paths]


def _extract_terms_with_local_rules(query: str) -> list[str]:
    pic_id = _legacy_extract_pic_id(query)
    terms: list[str] = [pic_id] if pic_id else []
    terms.extend(re.findall(r"[A-Za-z0-9_+\-]{3,}", query))
    terms.extend(re.findall(r"[\u4e00-\u9fff]{2,16}", query))
    if not terms:
        compact = re.sub(r"\s+", "", query)
        if compact:
            terms.append(compact[:32])
    return _dedupe_strings(terms)[:MAX_TERMS]


def _legacy_extract_query_terms(query: str) -> list[str]:
    knowledge_tool = _knowledge_tool()
    if knowledge_tool is None:
        return _extract_terms_with_local_rules(query)
    return list(knowledge_tool.extract_query_terms(query))


def _legacy_detect_intent(query: str) -> str:
    knowledge_tool = _knowledge_tool()
    if knowledge_tool is None:
        return _detect_intent_with_local_rules(query)
    return str(knowledge_tool.detect_intent(query))


def detect_lang(text: str) -> str:
    return _detect_lang(text)


def _detect_lang(text: str) -> str:
    knowledge_tool = _knowledge_tool()
    if knowledge_tool is not None and hasattr(knowledge_tool, "detect_lang"):
        try:
            return str(knowledge_tool.detect_lang(text))
        except Exception:
            pass
    return "zh" if re.search(r"[\u4e00-\u9fff]", str(text or "")) else "en"


def _legacy_extract_pic_id(query: str) -> str | None:
    knowledge_tool = _knowledge_tool()
    if knowledge_tool is not None and hasattr(knowledge_tool, "extract_pic_id"):
        try:
            return cast(str | None, knowledge_tool.extract_pic_id(query))
        except Exception:
            pass
    match = re.search(r"([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)", query)
    return match.group(1) if match else None


def _load_doc_profiles() -> tuple[Any, ...]:
    knowledge_tool = _knowledge_tool()
    if knowledge_tool is None:
        return ()
    return tuple(knowledge_tool.load_doc_profiles())


def _normalize_text(text: str) -> str:
    knowledge_tool = _knowledge_tool()
    if knowledge_tool is not None and hasattr(knowledge_tool, "normalize_text"):
        try:
            return str(knowledge_tool.normalize_text(text))
        except Exception:
            pass
    return re.sub(r"\s+", "", str(text or "")).lower()


def _knowledge_tool() -> Any | None:
    try:
        from app.tools import knowledge_tool
    except Exception:
        return None
    return knowledge_tool


def _serialize_part(part: QueryAnalysisPart) -> dict[str, Any]:
    return {
        "strategy": part.strategy,
        "intent_candidates": [_serialize_dataclass(item) for item in part.intent_candidates],
        "doc_candidates": [_serialize_dataclass(item) for item in part.doc_candidates],
        "query_terms": [_serialize_dataclass(item) for item in part.query_terms],
        "warnings": list(part.warnings or []),
    }


def _serialize_analysis(analysis: QueryAnalysis) -> dict[str, Any]:
    return {
        "query": analysis.query,
        "language": getattr(analysis, "language", None) or _detect_lang(analysis.query),
        "strategy": analysis.strategy,
        "primary_intent": getattr(analysis, "primary_intent", None),
        "primary_doc_id": getattr(analysis, "primary_doc_id", None),
        "intent_candidates": [_serialize_dataclass(item) for item in analysis.intent_candidates],
        "doc_candidates": [_serialize_dataclass(item) for item in analysis.doc_candidates],
        "query_terms": [_serialize_dataclass(item) for item in analysis.query_terms],
        "warnings": list(analysis.warnings),
    }


def _serialize_dataclass(item: Any) -> dict[str, Any]:
    if is_dataclass(item):
        return dict(vars(item))
    if hasattr(item, "__dict__"):
        return dict(item.__dict__)
    return {"value": item}
