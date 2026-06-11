"""Knowledge retrieval tool with intent routing, reranking, and fallback scan."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.models.response import sanitize_summary_metadata
from app.services.vector_search_service import SearchResult, vector_search_service

retrieval_orchestrator: Any = None
try:
    from app.retrieval import orchestrator as retrieval_orchestrator
except Exception:
    retrieval_orchestrator = None

_evidence_bundle_to_evidence_payload: Any = None
_evidence_format_bundle: Any = None
_evidence_format_search_results: Any = None
_evidence_display_path_list: Any = None
_evidence_image_reference_lines: Any = None
_evidence_search_result_to_document: Any = None
_evidence_search_result_to_evidence_hit: Any = None
_evidence_unique_flatten: Any = None
try:
    from app.retrieval import evidence as _evidence_module

    _evidence_bundle_to_evidence_payload = getattr(
        _evidence_module,
        "bundle_to_evidence_payload",
        None,
    )
    _evidence_display_path_list = getattr(_evidence_module, "display_path_list", None)
    _evidence_format_bundle = getattr(_evidence_module, "format_bundle", None)
    _evidence_format_search_results = getattr(
        _evidence_module,
        "format_search_results",
        None,
    )
    _evidence_image_reference_lines = getattr(
        _evidence_module,
        "image_reference_lines",
        None,
    )
    _evidence_search_result_to_document = getattr(
        _evidence_module,
        "search_result_to_document",
        None,
    )
    _evidence_search_result_to_evidence_hit = getattr(
        _evidence_module,
        "search_result_to_evidence_hit",
        None,
    )
    _evidence_unique_flatten = getattr(_evidence_module, "unique_flatten", None)
except Exception:
    pass


EVIDENCE_SCHEMA_VERSION = "retrieval_evidence_v1"
LEGACY_INTENT_STRATEGY = "legacy_routed"
LEGACY_RERANKER_PROVIDER = "lexical"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LAST_RETRIEVAL_METADATA: ContextVar[dict[str, Any] | None] = ContextVar(
    "last_retrieval_metadata",
    default=None,
)
_last_retrieval_metadata_fallback: dict[str, Any] | None = None
_LAST_RETRIEVAL_FALLBACK_ANSWER: ContextVar[str | None] = ContextVar(
    "last_retrieval_fallback_answer",
    default=None,
)
_last_retrieval_fallback_answer_fallback: str | None = None
PIC_ID_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)", re.IGNORECASE)
PROFILE_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]{1,}|[\u4e00-\u9fff]{2,16}")
PROFILE_SCAN_LIMIT = 6000
PROFILE_STOP_TERMS = {
    "pic",
    "baseline",
    "metadata",
    "image",
    "path",
    "jsonl",
    "chunk",
    "manual",
}
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
EN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
EN_PROTECTED_QUERY_TERMS = frozenset(
    {
        "not",
        "no",
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
    }
)
_EN_STOP_WORD_CANDIDATES = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "how",
    "what",
    "where",
    "when",
    "why",
    "which",
    "who",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "may",
    "might",
    "shall",
    "i",
    "my",
    "me",
    "we",
    "our",
    "you",
    "your",
    "if",
    "but",
    "so",
    "then",
}
EN_STOP_WORDS = frozenset(
    term for term in _EN_STOP_WORD_CANDIDATES if term not in EN_PROTECTED_QUERY_TERMS
)

ACTION_TERMS = [
    "安装",
    "设置",
    "使用",
    "步骤",
    "切换",
    "更换",
    "拆卸",
    "调节",
    "启动",
    "停机",
    "操作",
    "加油",
    "加注",
    "混合",
    "清洁",
    "维护",
    "保养",
    "检查",
    "范围",
    "问题",
]

SAFETY_TERMS = [
    "安全",
    "警告",
    "警示",
    "注意",
    "小心",
    "危险",
    "禁止",
    "防护",
    "防护装备",
    "个人防护装备",
    "佩戴",
    "宽松衣物",
    "围巾",
    "项链",
    "标识",
    "标签",
    "电源",
    "触电",
    "雷雨",
]

TROUBLESHOOTING_TERMS = [
    "故障",
    "异常",
    "原因",
    "排查",
    "解决",
    "维修",
    "损坏",
    "破损",
    "磨损",
    "不制冷",
    "不工作",
    "报警",
    "失灵",
]

MANUAL_QUERY_TERMS = [
    "个人防护装备",
    "防护装备",
    "宽松衣物",
    "警示标识",
    "警示标签",
    "燃油安全",
    "燃油混合",
    "混合方法",
    "加油操作",
    "冷机启动",
    "热机启动",
    "吹风管",
    "吹管",
    "喷口",
    "空气滤清器",
]

QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "吹管": ("吹风管", "喷口"),
    "吹风管": ("吹管", "喷口"),
    "防护装备": ("个人防护装备", "听力防护", "眼部防护", "面罩", "工作靴"),
    "个人防护装备": ("防护装备", "听力防护", "眼部防护", "面罩", "工作靴"),
    "宽松衣物": ("衣物", "宽松", "围巾", "项链", "长发"),
    "警示标识": ("警示标签", "警告标识", "破损", "磨损"),
    "警示标签": ("警示标识", "警告标识", "破损", "磨损"),
    "加油": ("加注", "油箱", "加油操作"),
    "燃油混合": ("混合方法", "汽油", "机油", "1:50"),
    "混合": ("燃油混合", "混合方法"),
    "冷机启动": ("冷机", "启动"),
    "热机启动": ("热机", "启动"),
    "维护": ("清洁", "保养", "检查"),
    "空气滤清器": ("滤清器", "清洁", "维护"),
}

CHUNK_TYPE_FAMILY: dict[str, str] = {
    "product_overview": "overview",
    "component_description": "component",
    "assembly_steps": "procedure",
    "operation_guide": "procedure",
    "maintenance_guide": "procedure",
    "troubleshooting_qa": "troubleshooting",
    "safety_warning": "safety",
    "legal_statement": "legal",
    "parts_list": "component",
    "specification_table": "component",
    "general_info": "general",
    "metadata_image_path": "auxiliary",
    "aux_navigation": "auxiliary",
}

FAMILY_ROUTE_TYPES: dict[str, tuple[str, ...]] = {
    "component": ("component_description", "parts_list", "specification_table", "general_info"),
    "procedure": ("assembly_steps", "operation_guide", "maintenance_guide", "general_info"),
    "legal": ("legal_statement", "general_info"),
    "overview": ("product_overview", "general_info"),
    "safety": ("safety_warning", "general_info"),
    "troubleshooting": ("troubleshooting_qa", "general_info"),
    "general": (
        "product_overview",
        "component_description",
        "assembly_steps",
        "operation_guide",
        "maintenance_guide",
        "troubleshooting_qa",
        "safety_warning",
        "legal_statement",
        "parts_list",
        "specification_table",
        "general_info",
    ),
    "auxiliary": ("metadata_image_path", "aux_navigation"),
    "image": ("component_description", "parts_list", "specification_table", "general_info", "metadata_image_path", "aux_navigation"),
    "ocr": ("general_info", "metadata_image_path", "aux_navigation"),
}

INTENT_STAGE_CONFIG: dict[str, list[dict[str, Any]]] = {
    "component": [
        {"name": "primary_route", "tiers": ["primary"], "families": ["component"]},
        {"name": "support_route", "tiers": ["support"], "families": ["overview"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "families": ["component", "overview", "procedure"]},
    ],
    "procedure": [
        {"name": "primary_route", "tiers": ["primary"], "families": ["procedure"]},
        {"name": "support_route", "tiers": ["support"], "families": ["overview"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "families": ["procedure", "component", "overview"]},
    ],
    "legal": [
        {"name": "primary_route", "tiers": ["primary"], "families": ["legal"]},
        {"name": "support_route", "tiers": ["support"], "families": ["overview"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "families": ["legal", "overview", "general"]},
    ],
    "safety": [
        {"name": "primary_route", "tiers": ["primary"], "families": ["safety"]},
        {"name": "support_route", "tiers": ["support"], "families": ["overview"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "families": ["safety", "overview", "general"]},
    ],
    "troubleshooting": [
        {"name": "primary_route", "tiers": ["primary"], "families": ["troubleshooting"]},
        {"name": "support_route", "tiers": ["support"], "families": ["overview"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "families": ["troubleshooting", "overview", "general"]},
    ],
    "image_trace": [
        {"name": "auxiliary_direct", "tiers": ["auxiliary"], "families": ["auxiliary"]},
        {"name": "primary_route", "tiers": ["primary"], "families": ["image", "component"]},
        {"name": "expanded_route", "tiers": ["primary", "support", "auxiliary"], "families": ["auxiliary", "image", "component", "overview"]},
    ],
    "ocr_audit": [
        {"name": "auxiliary_direct", "tiers": ["auxiliary"], "families": ["auxiliary"]},
        {"name": "primary_route", "tiers": ["primary", "support"], "families": ["ocr", "overview"]},
        {"name": "expanded_route", "tiers": ["primary", "support", "auxiliary"], "families": None},
    ],
    "overview": [
        {"name": "primary_route", "tiers": ["primary", "support"], "families": ["overview"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "families": ["overview", "component", "procedure"]},
    ],
    "general": [
        {"name": "primary_route", "tiers": ["primary"], "families": None},
        {"name": "expanded_route", "tiers": ["primary", "support"], "families": None},
    ],
}

SUPPORT_TYPE_BOOST = {"section_summary", "procedure_overview"}
LEGAL_MARKERS = ("保修", "法规", "声明", "责任", "政策", "支持与服务", "网站", "网址", "附录", "废弃", "回收", "报废", "环保")
OCR_MARKERS = ("ocr", "识别", "截断", "缺失", "原文", "源数据", "英国用户")


@dataclass
class RetrievalBundle:
    """Container for retrieval pipeline outputs."""

    intent: str
    retrieval_stage: str
    hits: list[SearchResult]
    support_hits: list[SearchResult]
    warnings: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_hits(self) -> list[SearchResult]:
        combined: list[SearchResult] = []
        seen: set[str] = set()
        for result in [*self.hits, *self.support_hits]:
            if result.id in seen:
                continue
            seen.add(result.id)
            combined.append(result)
        return combined


def set_last_retrieval_metadata(metadata: dict[str, Any] | None) -> None:
    """Store the latest Summary-level retrieval metadata for the current tool call."""
    global _last_retrieval_metadata_fallback

    sanitized = sanitize_summary_metadata(metadata)
    snapshot = dict(sanitized) if sanitized else None
    _LAST_RETRIEVAL_METADATA.set(snapshot)
    _last_retrieval_metadata_fallback = dict(snapshot) if snapshot else None


def set_last_retrieval_fallback_answer(answer: str | None) -> None:
    """Store a compact evidence-based answer for request timeout fallback."""
    global _last_retrieval_fallback_answer_fallback

    snapshot = answer.strip() if isinstance(answer, str) and answer.strip() else None
    _LAST_RETRIEVAL_FALLBACK_ANSWER.set(snapshot)
    _last_retrieval_fallback_answer_fallback = snapshot


def clear_last_retrieval_metadata() -> None:
    """Clear stale retrieval metadata before a new agent query starts."""
    set_last_retrieval_metadata(None)
    set_last_retrieval_fallback_answer(None)


def get_last_retrieval_metadata() -> dict[str, Any] | None:
    """Return a defensive copy of the latest Summary-level retrieval metadata."""
    metadata = _LAST_RETRIEVAL_METADATA.get() or _last_retrieval_metadata_fallback
    return dict(metadata) if metadata else None


def get_last_retrieval_fallback_answer() -> str | None:
    """Return the latest compact evidence answer for timeout fallback."""
    return _LAST_RETRIEVAL_FALLBACK_ANSWER.get() or _last_retrieval_fallback_answer_fallback


@dataclass(frozen=True)
class DocumentProfile:
    """Derived document routing profile built from indexed chunk metadata."""

    doc_id: str
    doc_name: str
    terms: tuple[str, ...]
    pic_prefixes: tuple[str, ...]
    chunk_types: tuple[str, ...]
    chunk_families: tuple[str, ...]
    source_issue_flags: tuple[str, ...]
    chunk_count: int


def detect_lang(text: str) -> str:
    """Classify a query as Chinese when it contains any CJK character."""
    return "zh" if CHINESE_CHAR_RE.search(str(text or "")) else "en"


def extract_english_query_terms(text: str) -> list[str]:
    """Extract English query terms without stemming, phrases, or synonym expansion."""
    tokens = EN_TOKEN_RE.findall(str(text or "").lower())
    terms = [token for token in tokens if token and token not in EN_STOP_WORDS]
    return list(dict.fromkeys(terms))[:16]


def infer_families_from_type_name(chunk_type: str) -> tuple[str, ...]:
    """Infer retrieval families from flexible/rich chunk type names."""
    type_name = str(chunk_type or "").strip().lower()
    if not type_name:
        return ()

    families: set[str] = set()
    if any(marker in type_name for marker in ("metadata", "navigation", "image_path", "aux")):
        families.add("auxiliary")
    if any(marker in type_name for marker in ("overview", "summary", "group")):
        families.add("overview")
    if any(marker in type_name for marker in ("safety", "warning", "caution", "ppe", "protection")):
        families.add("safety")
    if any(marker in type_name for marker in ("legal", "warranty", "regulation", "emissions")):
        families.add("legal")
    if any(marker in type_name for marker in ("troubleshooting", "fault", "problem", "prevention")):
        families.add("troubleshooting")
    if any(
        marker in type_name
        for marker in (
            "step",
            "task",
            "install",
            "installation",
            "startup",
            "start",
            "stop",
            "fueling",
            "mix",
            "mixing",
            "operation",
            "adjustment",
            "cleaning",
            "maintenance",
            "service",
            "prevention",
            "requirement",
        )
    ):
        families.add("procedure")
    if any(
        marker in type_name
        for marker in (
            "component",
            "item",
            "device",
            "accessory",
            "part",
            "port",
            "button",
            "indicator",
            "label",
            "tube",
            "nozzle",
            "filter",
            "carburetor",
            "muffler",
            "needle",
        )
    ):
        families.add("component")

    return tuple(sorted(families))


def infer_chunk_families(
    chunk_type: str,
    title: str = "",
    section_path: Sequence[str] | None = None,
    text: str = "",
) -> tuple[str, ...]:
    """Derive stable retrieval families from one chunk."""
    families: set[str] = set()
    normalized_type = str(chunk_type or "").strip()
    direct_family = CHUNK_TYPE_FAMILY.get(normalized_type)
    if direct_family:
        families.add(direct_family)
    families.update(infer_families_from_type_name(normalized_type))

    haystack = normalize_text(
        " ".join(
            [
                str(title or ""),
                " ".join(str(part) for part in (section_path or [])),
                str(text or ""),
            ]
        )
    )

    if any(normalize_text(term) in haystack for term in SAFETY_TERMS):
        families.add("safety")
    if any(normalize_text(term) in haystack for term in TROUBLESHOOTING_TERMS):
        families.add("troubleshooting")
    if any(normalize_text(term) in haystack for term in LEGAL_MARKERS):
        families.add("legal")
    if any(normalize_text(term) in haystack for term in ACTION_TERMS):
        families.add("procedure")
    if any(normalize_text(term) in haystack for term in ("部件", "接口", "位置", "按键", "按钮", "指示灯", "追踪灯", "配件", "滤清器", "吹风管", "喷口")):
        families.add("component")
    if "目录" in haystack or "概览" in haystack or "附录" in haystack:
        families.add("overview")

    if not families and normalized_type:
        families.add("subsection")
    return tuple(sorted(families))


def resolve_chunk_types(
    chunk_families: Sequence[str] | None,
    doc_id: str | None = None,
) -> list[str] | None:
    """Expand retrieval families into concrete chunk types for Milvus filtering."""
    if not chunk_families:
        return None

    resolved: list[str] = []
    for family in chunk_families:
        resolved.extend(FAMILY_ROUTE_TYPES.get(family, (family,)))

    if doc_id:
        profile = next((item for item in load_doc_profiles() if item.doc_id == doc_id), None)
        if profile:
            doc_types = set(profile.chunk_types)
            # User requested: bypass intent-based filtering and absorb all chunk types.
            resolved.extend(profile.chunk_types)
            narrowed = [chunk_type for chunk_type in resolved if chunk_type in doc_types]
            if narrowed:
                resolved = narrowed

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk_type in resolved:
        if not chunk_type or chunk_type in seen:
            continue
        seen.add(chunk_type)
        deduped.append(chunk_type)
    return deduped or None


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> tuple[str, list[Document]]:
    """Retrieve relevant manual knowledge for a user query."""
    clear_last_retrieval_metadata()
    try:
        logger.info("Knowledge retrieval called: query='{}'", query)
        bundle = routed_retrieve(query)
        set_last_retrieval_metadata(bundle.metadata)
        docs = [search_result_to_document(result) for result in bundle.all_hits]

        if not docs:
            logger.warning("No relevant documents found after fallback.")
            return "没有找到相关信息。", []

        context = format_bundle(bundle, query=query)
        set_last_retrieval_fallback_answer(
            build_evidence_fallback_answer(bundle.all_hits)
        )
        logger.info(
            "Retrieved {} documents, intent={}, stage={}",
            len(docs),
            bundle.intent,
            bundle.retrieval_stage,
        )
        return context, docs
    except Exception as exc:
        clear_last_retrieval_metadata()
        logger.error(f"Knowledge retrieval failed: {exc}")
        return f"检索知识时发生错误: {str(exc)}", []


def routed_retrieve(query: str) -> RetrievalBundle:
    """Run the modular retrieval pipeline, falling back to the legacy path."""
    if retrieval_orchestrator is not None:
        try:
            bundle = retrieval_orchestrator.retrieve(query)
            set_last_retrieval_metadata(bundle.metadata)
            return cast(RetrievalBundle, bundle)
        except Exception as exc:
            logger.warning(
                "Modular retrieval failed; falling back to legacy retrieval: {}",
                exc,
            )
            fallback_bundle = _legacy_routed_retrieve(query)
            _mark_legacy_fallback_metadata(fallback_bundle, exc)
            set_last_retrieval_metadata(fallback_bundle.metadata)
            return fallback_bundle

    fallback_bundle = _legacy_routed_retrieve(query)
    set_last_retrieval_metadata(fallback_bundle.metadata)
    return fallback_bundle


def legacy_routed_retrieve(query: str) -> RetrievalBundle:
    """Backward-compatible explicit entry point for the legacy retrieval path."""
    return _legacy_routed_retrieve(query)


def _mark_legacy_fallback_metadata(bundle: RetrievalBundle, exc: Exception) -> None:
    """Keep fallback metadata Summary-only while recording degraded execution."""
    warning = f"orchestrator fallback: {exc}"
    if warning not in bundle.warnings:
        bundle.warnings.append(warning)

    metadata = dict(bundle.metadata or {})
    warnings = list(metadata.get("warnings") or [])
    if warning not in warnings:
        warnings.append(warning)
    metadata.update(
        {
            "warnings": warnings,
            "degraded": True,
            "intent_strategy": metadata.get("intent_strategy") or LEGACY_INTENT_STRATEGY,
            "reranker_provider": metadata.get("reranker_provider")
            or LEGACY_RERANKER_PROVIDER,
        }
    )
    bundle.metadata = sanitize_summary_metadata(metadata) or {}


def _legacy_routed_retrieve(query: str) -> RetrievalBundle:
    """Run intent-routed retrieval with layered fallback."""
    intent = detect_intent(query)
    doc_id = infer_doc_id(query)
    query_terms = extract_query_terms(query)
    query_variants = expand_queries(query, intent, query_terms)
    warnings: list[str] = []

    if doc_id:
        warnings.append(f"已根据问题内容收窄到文档: {doc_id}")

    stage_config = INTENT_STAGE_CONFIG.get(intent, INTENT_STAGE_CONFIG["general"])
    retrieval_stage = "none"
    hits: list[SearchResult] = []

    for stage in stage_config:
        stage_hits = run_search_stage(
            query_variants=query_variants,
            original_query=query,
            query_terms=query_terms,
            intent=intent,
            doc_id=doc_id,
            tiers=stage["tiers"],
            chunk_families=stage.get("families"),
            top_k=max(config.rag_top_k, 4),
        )
        if stage_hits:
            retrieval_stage = stage["name"]
            hits = trim_results_for_intent(stage_hits, intent)
            if is_strong_hit(stage_hits, query_terms, intent):
                break
            warnings.append(f"主路检索命中较弱，继续执行 fallback: {stage['name']}")

    if not hits:
        hits = run_scan_stage(
            original_query=query,
            query_terms=query_terms,
            intent=intent,
            doc_id=doc_id,
            tiers=["primary", "support", "auxiliary"],
            chunk_families=None,
            top_k=max(config.rag_top_k, 6),
        )
        if hits:
            retrieval_stage = "fallback_full_collection"
            warnings.append("已触发全库 chunk 遍历保底。")

    support_hits = fetch_support_hits(hits, doc_id, query_terms, intent)
    warnings.extend(collect_source_warnings(hits, support_hits))

    return RetrievalBundle(
        intent=intent,
        retrieval_stage=retrieval_stage,
        hits=hits,
        support_hits=support_hits,
        warnings=warnings,
        metadata=build_legacy_summary_metadata(
            intent=intent,
            doc_id=doc_id,
            retrieval_stage=retrieval_stage,
            hits=hits,
            warnings=warnings,
        ),
    )


def build_legacy_summary_metadata(
    intent: str,
    doc_id: str | None,
    retrieval_stage: str,
    hits: Sequence[SearchResult],
    warnings: Sequence[str],
) -> dict[str, Any]:
    """Build API-safe Summary metadata for the pre-orchestrator retrieval path."""
    recall_channels = infer_recall_channels(retrieval_stage, hits)
    metadata = {
        "intent": intent,
        "doc_id": doc_id,
        "retrieval_stage": retrieval_stage,
        "intent_strategy": LEGACY_INTENT_STRATEGY,
        "recall_channels": recall_channels,
        "reranker_provider": LEGACY_RERANKER_PROVIDER,
        "reranker_fallback": retrieval_stage == "fallback_full_collection",
        "timeout": False,
        "degraded": bool(warnings) or retrieval_stage == "none",
        "top_hits": [
            summarize_search_result_for_metadata(result, recall_channels)
            for result in hits[: max(config.rag_top_k, 3)]
        ],
        "warnings": list(warnings),
    }
    return sanitize_summary_metadata(metadata) or {}


def infer_recall_channels(
    retrieval_stage: str,
    hits: Sequence[SearchResult],
) -> list[str]:
    """Infer compact recall-channel labels from the legacy retrieval stage."""
    if not hits or retrieval_stage == "none":
        return []
    if retrieval_stage == "fallback_full_collection":
        return ["scan"]
    return ["vector", "scan"]


def summarize_search_result_for_metadata(
    result: SearchResult,
    recall_channels: Sequence[str],
) -> dict[str, Any]:
    """Return a compact top-hit summary without trace-level fields."""
    metadata = result.metadata or {}
    return {
        "chunk_id": metadata.get("chunk_id") or result.id,
        "score": result.score,
        "channels": list(recall_channels),
    }


def detect_intent(query: str) -> str:
    """Heuristically classify the query intent."""
    normalized = normalize_text(query)
    if extract_pic_id(query) or any(keyword.lower() in normalized for keyword in ("图片", "图示", "配图", "对应图", "pic")):
        return "image_trace"
    if any(keyword.lower() in normalized for keyword in ("ocr", "原文", "源数据", "识别错误", "乱码", "截断", "缺失")):
        return "ocr_audit"
    if (
        "状态" in normalized or any(keyword in normalized for keyword in ("红色", "白色", "蓝色", "灯灭"))
    ) and any(keyword in normalized for keyword in ("指示灯", "追踪灯", "状态指示灯", "后侧灯光", "内部灯")):
        return "component"
    if any(keyword.lower() in normalized for keyword in SAFETY_TERMS):
        return "safety"
    if any(keyword.lower() in normalized for keyword in TROUBLESHOOTING_TERMS):
        return "troubleshooting"
    if any(keyword.lower() in normalized for keyword in ("保修", "法规", "声明", "责任", "政策", "支持与服务", "网站", "网址", "废弃", "回收", "报废", "环保")):
        return "legal"
    if any(keyword.lower() in normalized for keyword in ("怎么", "如何", "步骤", "安装", "设置", "更换", "切换", "使用", "拆卸", "开启", "关闭", "调节")):
        return "procedure"
    if any(keyword.lower() in normalized for keyword in ("目录", "概览", "章节", "哪些内容", "有哪些", "整章", "附录")):
        return "overview"
    if any(keyword.lower() in normalized for keyword in ("是什么", "在哪", "位置", "接口", "按键", "按钮", "指示灯", "部件")):
        return "component"
    return "general"


@lru_cache(maxsize=1)
def load_doc_profiles() -> tuple[DocumentProfile, ...]:
    """Build lightweight document routing profiles from indexed chunk metadata."""
    try:
        results = vector_search_service.query_documents(limit=PROFILE_SCAN_LIMIT)
    except Exception as exc:
        logger.warning("Failed to load document profiles from Milvus: {}", exc)
        return ()

    buckets: dict[str, dict[str, Any]] = {}
    for result in results:
        metadata = result.metadata or {}
        doc_id = str(metadata.get("doc_id") or "").strip()
        if not doc_id:
            continue
        if doc_id.endswith("_baseline"):
            continue

        bucket = buckets.setdefault(
            doc_id,
            {
                "doc_name": "",
                "terms": [],
                "pic_prefixes": set(),
                "chunk_types": set(),
                "chunk_families": set(),
                "source_issue_flags": set(),
                "chunk_count": 0,
            },
        )
        bucket["chunk_count"] += 1

        doc_name = str(metadata.get("doc_name") or metadata.get("_file_name") or "").strip()
        if doc_name and (not bucket["doc_name"] or len(doc_name) < len(bucket["doc_name"])):
            bucket["doc_name"] = doc_name

        section_path = metadata.get("section_path") or []
        title = metadata.get("title") or metadata.get("section_title") or ""
        note = str(metadata.get("source_issue_note") or "").strip()
        flags = [
            flag
            for flag in (metadata.get("source_issue_flags") or [])
            if isinstance(flag, str) and flag
        ]
        bucket["terms"].extend(extract_profile_terms(doc_name, title, note, *flags))
        if isinstance(section_path, list):
            bucket["terms"].extend(
                term
                for section in section_path
                for term in extract_profile_terms(str(section))
            )

        chunk_type = str(metadata.get("chunk_type") or "").strip()
        chunk_families = infer_chunk_families(
            chunk_type,
            title=title,
            section_path=section_path if isinstance(section_path, list) else [section_path],
            text=str(metadata.get("text") or ""),
        )
        if chunk_type:
            bucket["chunk_types"].add(chunk_type)
        for chunk_family in chunk_families:
            bucket["chunk_families"].add(chunk_family)

        for flag in flags:
            bucket["source_issue_flags"].add(flag)

        for pic_id in metadata.get("pic_ids") or []:
            if not isinstance(pic_id, str) or not pic_id:
                continue
            bucket["terms"].append(normalize_text(pic_id))
            pic_prefix = extract_pic_prefix(pic_id)
            if pic_prefix:
                bucket["pic_prefixes"].add(pic_prefix)

    profiles: list[DocumentProfile] = []
    for doc_id, bucket in buckets.items():
        term_counts = Counter(bucket["terms"])
        ranked_terms = tuple(term for term, _ in term_counts.most_common(256))
        profiles.append(
            DocumentProfile(
                doc_id=doc_id,
                doc_name=bucket["doc_name"] or doc_id,
                terms=ranked_terms,
                pic_prefixes=tuple(sorted(bucket["pic_prefixes"])),
                chunk_types=tuple(sorted(bucket["chunk_types"])),
                chunk_families=tuple(sorted(bucket["chunk_families"])),
                source_issue_flags=tuple(sorted(bucket["source_issue_flags"])),
                chunk_count=int(bucket["chunk_count"]),
            )
        )

    profiles.sort(key=lambda profile: profile.doc_id)
    logger.info("Loaded {} dynamic document profiles", len(profiles))
    return tuple(profiles)


def extract_profile_terms(*text_blobs: str) -> list[str]:
    """Extract normalized routing terms from compact metadata text."""
    terms: list[str] = []
    for blob in text_blobs:
        raw = str(blob or "").strip()
        if not raw:
            continue

        normalized = normalize_text(raw)
        if should_keep_profile_term(normalized) and len(normalized) <= 32:
            terms.append(normalized)

        for token in PROFILE_TERM_RE.findall(raw):
            normalized_token = normalize_text(token)
            if should_keep_profile_term(normalized_token):
                terms.append(normalized_token)

    return terms


def should_keep_profile_term(term: str) -> bool:
    """Filter noisy metadata terms out of document profiles."""
    normalized = normalize_text(term)
    if len(normalized) < 2:
        return False
    if normalized in PROFILE_STOP_TERMS:
        return False
    if re.fullmatch(r"manual\d+", normalized):
        return False
    if re.fullmatch(r"page\d+", normalized):
        return False
    return True


def extract_pic_prefix(pic_id: str) -> str | None:
    """Derive a stable picture-id prefix from one picture identifier."""
    normalized = normalize_text(pic_id)
    if "_" not in normalized:
        return None

    parts = normalized.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]) + "_"
    return parts[0] + "_"


@lru_cache(maxsize=1)
def load_profile_term_index() -> tuple[str, ...]:
    """Flatten all document profile terms into a ranked lookup table."""
    terms = {
        term
        for profile in load_doc_profiles()
        for term in profile.terms
        if should_keep_profile_term(term)
    }
    return tuple(sorted(terms, key=lambda item: (-len(item), item)))


@lru_cache(maxsize=1)
def load_active_doc_ids() -> tuple[str, ...]:
    """Return the set of document ids currently exposed to generic retrieval."""
    return tuple(profile.doc_id for profile in load_doc_profiles())


def reset_profile_caches() -> None:
    """Clear cached document-routing metadata after reindexing."""
    load_doc_profiles.cache_clear()
    load_profile_term_index.cache_clear()
    load_active_doc_ids.cache_clear()


def match_profile_terms_in_query(normalized_query: str, limit: int = 8) -> list[str]:
    """Return profile-derived terms that appear inside the query."""
    matches: list[str] = []
    seen: set[str] = set()
    for term in load_profile_term_index():
        if term in normalized_query:
            matches.append(term)
            seen.add(term)
            if len(matches) >= limit:
                return matches

    chinese_spans = re.findall(r"[\u4e00-\u9fff]{2,20}", normalized_query)
    profile_terms = load_profile_term_index()
    for span in chinese_spans:
        max_width = min(len(span), 8)
        for width in range(max_width, 1, -1):
            for start in range(0, len(span) - width + 1):
                piece = span[start : start + width]
                if piece in seen or not should_keep_profile_term(piece):
                    continue
                if any(piece in term or term in piece for term in profile_terms):
                    matches.append(piece)
                    seen.add(piece)
                    if len(matches) >= limit:
                        return matches
    return matches


def infer_doc_id(query: str) -> str | None:
    """Infer a likely target manual from query hints."""
    normalized = normalize_text(query)
    pic_id = extract_pic_id(query)
    query_terms = [normalize_text(term) for term in extract_query_terms(query) if term]
    profiles = load_doc_profiles()

    best_doc_id: str | None = None
    best_score = 0
    second_best = 0

    for profile in profiles:
        score = 0
        doc_name = normalize_text(profile.doc_name)
        doc_id_term = normalize_text(profile.doc_id)

        if pic_id:
            normalized_pic = normalize_text(pic_id)
            if any(normalized_pic.startswith(prefix) for prefix in profile.pic_prefixes):
                score += 120

        if doc_name and doc_name in normalized:
            score += 80
        if doc_id_term and doc_id_term in normalized:
            score += 60

        for term in profile.terms:
            if term and term in normalized:
                score += 16 if len(term) >= 4 else 8

        for term in query_terms:
            if term in profile.terms:
                score += 10
                continue
            if any(
                len(term) >= 2
                and len(profile_term) >= 2
                and (term in profile_term or profile_term in term)
                for profile_term in profile.terms[:24]
            ):
                score += 6

        if score > best_score:
            second_best = best_score
            best_doc_id = profile.doc_id
            best_score = score
        elif score > second_best:
            second_best = score

    if best_score <= 0:
        return None
    if second_best and best_score - second_best < 6 and best_score < 24:
        return None
    return best_doc_id


def filter_results_to_active_docs(
    results: Sequence[SearchResult],
    scoped_doc_id: str | None,
) -> list[SearchResult]:
    """Keep unscoped retrieval focused on structured, profiled documents only."""
    if scoped_doc_id:
        return list(results)

    active_doc_ids = set(load_active_doc_ids())
    if not active_doc_ids:
        return list(results)

    filtered = [
        result
        for result in results
        if str((result.metadata or {}).get("doc_id") or "").strip() in active_doc_ids
    ]
    return filtered or list(results)


def expand_queries(query: str, intent: str, query_terms: Sequence[str]) -> list[str]:
    """Create retrieval-friendly rewrites for short manual questions."""
    variants = [query.strip()]

    if detect_lang(query) == "en":
        if query_terms:
            variants.append(" ".join(dict.fromkeys(query_terms)))

        english_deduped: list[str] = []
        english_seen: set[str] = set()
        for variant in variants:
            cleaned = variant.strip()
            if not cleaned or cleaned in english_seen:
                continue
            english_seen.add(cleaned)
            english_deduped.append(cleaned)
        return english_deduped

    compact = re.sub(r"\s+", "", query)
    stripped = compact
    for prefix in ("怎么", "如何", "请问", "想问", "帮我", "一下", "这个", "这个是", "请问一下"):
        stripped = stripped.replace(prefix, "")
    for suffix in ("是什么", "在哪", "怎么用", "怎么看", "怎么办", "是什么图", "对应什么"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]

    if stripped and stripped != compact:
        variants.append(stripped)

    if query_terms:
        variants.append(" ".join(dict.fromkeys(query_terms)))

    if intent == "procedure" and query_terms:
        variants.append(f"{' '.join(query_terms)} 步骤 安装 设置 使用")
    if intent == "component" and query_terms:
        variants.append(f"{' '.join(query_terms)} 部件 位置 说明")
    if intent == "legal" and query_terms:
        variants.append(f"{' '.join(query_terms)} 条款 保修 声明")
    if intent == "safety" and query_terms:
        variants.append(f"{' '.join(query_terms)} 安全 注意 警告 电源")
    if intent == "troubleshooting" and query_terms:
        variants.append(f"{' '.join(query_terms)} 故障 原因 排查 解决")
    if intent == "ocr_audit" and query_terms:
        variants.append(f"{' '.join(query_terms)} OCR 原文 识别错误 缺失")
    if intent == "image_trace":
        pic_id = extract_pic_id(query)
        if pic_id:
            variants.append(pic_id)
            variants.append(f"{pic_id} 图片 对应")
        elif query_terms:
            variants.append(f"{' '.join(query_terms)} 图片 配图")
    if intent == "general" and query_terms:
        variants.append(" ".join(dict.fromkeys(term for term in query_terms if len(normalize_text(term)) >= 2)))

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        cleaned = variant.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def run_search_stage(
    query_variants: Sequence[str],
    original_query: str,
    query_terms: Sequence[str],
    intent: str,
    doc_id: str | None,
    tiers: Sequence[str] | None,
    chunk_families: Sequence[str] | None,
    top_k: int,
) -> list[SearchResult]:
    """Execute vector search and optional metadata scan rerank for a stage."""
    aggregated: list[SearchResult] = []
    stage_fetch_k = max(top_k * 3, 8)
    chunk_types = resolve_chunk_types(chunk_families, doc_id=doc_id)

    for variant in query_variants[:3]:
        try:
            aggregated.extend(
                filter_results_to_active_docs(
                    vector_search_service.search_similar_documents(
                    query=variant,
                    top_k=stage_fetch_k,
                    doc_id=doc_id,
                    retrieval_tiers=list(tiers) if tiers else None,
                    chunk_types=list(chunk_types) if chunk_types else None,
                    ),
                    scoped_doc_id=doc_id,
                )
            )
        except Exception as exc:
            logger.warning(
                "Vector stage failed: query='{}', doc_id={}, tiers={}, chunk_families={}, chunk_types={}, error={}",
                variant,
                doc_id,
                tiers,
                chunk_families,
                chunk_types,
                exc,
            )

    reranked = rerank_results(
        deduplicate_results(aggregated),
        original_query=original_query,
        query_terms=query_terms,
        intent=intent,
        prefer_support="support" in (tiers or ()),
    )
    reranked = post_filter_results(
        reranked,
        intent=intent,
        query_terms=query_terms,
    )

    if should_scan_stage(reranked, query_terms, intent) or query_critical_term_groups(normalize_text(original_query)):
        scanned = run_scan_stage(
            original_query=original_query,
            query_terms=query_terms,
            intent=intent,
            doc_id=doc_id,
            tiers=tiers,
            chunk_families=chunk_families,
            top_k=stage_fetch_k,
        )
        merged = rerank_results(
            merge_ranked_results(reranked, scanned),
            original_query=original_query,
            query_terms=query_terms,
            intent=intent,
            prefer_support="support" in (tiers or ()),
        )
        return merged[:top_k]

    return reranked[:top_k]


def run_scan_stage(
    original_query: str,
    query_terms: Sequence[str],
    intent: str,
    doc_id: str | None,
    tiers: Sequence[str] | None,
    chunk_families: Sequence[str] | None,
    top_k: int,
) -> list[SearchResult]:
    """Run metadata-filtered full scan fallback and rerank locally."""
    try:
        chunk_types = resolve_chunk_types(chunk_families, doc_id=doc_id)
        scanned = filter_results_to_active_docs(
            vector_search_service.query_all_documents(
                doc_id=doc_id,
                retrieval_tiers=list(tiers) if tiers else None,
                chunk_types=list(chunk_types) if chunk_types else None,
            ),
            scoped_doc_id=doc_id,
        )
        reranked = rerank_results(
            scanned,
            original_query=original_query,
            query_terms=query_terms,
            intent=intent,
            prefer_support="support" in (tiers or ()),
        )
        reranked = post_filter_results(
            reranked,
            intent=intent,
            query_terms=query_terms,
        )
        return reranked[:top_k]
    except Exception as exc:
        logger.warning(
            "Scan stage failed: doc_id={}, tiers={}, chunk_types={}, error={}",
            doc_id,
            tiers,
            resolve_chunk_types(chunk_families, doc_id=doc_id),
            exc,
        )
        return []


def fetch_support_hits(
    hits: Sequence[SearchResult],
    doc_id: str | None,
    query_terms: Sequence[str],
    intent: str,
) -> list[SearchResult]:
    """Fetch support parent chunks for primary hits."""
    if intent == "ocr_audit":
        return []

    parent_ids: list[str] = []
    for hit in hits:
        metadata = hit.metadata or {}
        parent_chunk_id = metadata.get("parent_chunk_id")
        retrieval_tier = metadata.get("retrieval_tier")
        if retrieval_tier == "primary" and isinstance(parent_chunk_id, str) and parent_chunk_id.strip():
            parent_ids.append(parent_chunk_id.strip())

    unique_parent_ids = list(dict.fromkeys(parent_ids))
    if not unique_parent_ids:
        return []

    try:
        parents = vector_search_service.query_documents(
            doc_id=doc_id,
            retrieval_tiers=["support"],
            chunk_ids=unique_parent_ids,
            limit=len(unique_parent_ids),
        )
        ranked_parents = rerank_results(
            parents,
            original_query=" ".join(query_terms),
            query_terms=query_terms,
            intent="legal" if intent == "legal" else ("overview" if intent == "overview" else "general"),
            prefer_support=True,
        )
        ranked_parents = post_filter_results(
            ranked_parents,
            intent="legal" if intent == "legal" else "general",
            query_terms=query_terms,
        )
        return trim_results_for_intent(ranked_parents, intent if intent == "legal" else "general")
    except Exception as exc:
        logger.warning("Support expansion failed: {}", exc)
        return []


def rerank_results(
    results: Iterable[SearchResult],
    original_query: str,
    query_terms: Sequence[str],
    intent: str,
    prefer_support: bool = False,
) -> list[SearchResult]:
    """Rerank vector or scanned hits with lightweight lexical scoring."""
    scored: list[tuple[float, SearchResult]] = []
    for result in deduplicate_results(results):
        score = lexical_score(result, original_query, query_terms, intent, prefer_support)
        scored.append((score, result))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in scored]


def query_critical_term_groups(normalized_query: str) -> list[tuple[str, ...]]:
    """Return must-match-ish term groups for common manual questions."""
    if detect_lang(normalized_query) == "en":
        return []

    groups: list[tuple[str, ...]] = []
    if "冷机" in normalized_query:
        groups.append(("冷机",))
    if "热机" in normalized_query:
        groups.append(("热机",))
    if "加油" in normalized_query:
        groups.append(("加油", "加注", "油箱"))
    if "燃油" in normalized_query and "混合" in normalized_query:
        groups.append(("燃油混合", "混合方法", "1:50", "汽油", "机油"))
    elif "混合" in normalized_query:
        groups.append(("混合", "1:50"))
    if "防护装备" in normalized_query or "个人防护" in normalized_query:
        groups.append(("个人防护装备", "防护装备", "听力防护", "眼部防护", "面罩", "工作靴"))
    if "宽松" in normalized_query or "衣物" in normalized_query:
        groups.append(("宽松", "衣物", "围巾", "项链", "长发"))
    if "警示" in normalized_query or "标识" in normalized_query or "标签" in normalized_query:
        groups.append(("警示标识", "警示标签", "警告标识", "标识", "标签"))
    if "吹管" in normalized_query or "吹风管" in normalized_query:
        groups.append(("吹风管", "吹管", "喷口"))
    if "空气滤清器" in normalized_query or "滤清器" in normalized_query:
        groups.append(("空气滤清器", "滤清器"))
    if "ocr" in normalized_query or "缺失" in normalized_query or "识别" in normalized_query:
        groups.append(("ocr", "缺失", "识别", "源数据"))
    return groups


def lexical_score(
    result: SearchResult,
    original_query: str,
    query_terms: Sequence[str],
    intent: str,
    prefer_support: bool,
) -> float:
    """Compute a heuristic relevance score for a candidate chunk."""
    metadata = result.metadata or {}
    title = str(metadata.get("title") or metadata.get("section_title") or "")
    section_path = metadata.get("section_path") or []
    section_path_text = " > ".join(str(part) for part in section_path) if isinstance(section_path, list) else str(section_path)
    index_text = str(metadata.get("index_text") or "")
    text = str(metadata.get("text") or result.content or "")
    chunk_type = str(metadata.get("chunk_type") or "")
    retrieval_tier = str(metadata.get("retrieval_tier") or "")
    pic_ids = metadata.get("pic_ids") or []
    source_issue_flags = metadata.get("source_issue_flags") or []
    source_issue_note = str(metadata.get("source_issue_note") or "")
    chunk_families = set(
        infer_chunk_families(
            chunk_type,
            title=title,
            section_path=section_path if isinstance(section_path, list) else [section_path],
            text=text,
        )
    )

    pic_text = " ".join(str(pic_id) for pic_id in pic_ids)
    hay_title = normalize_text(title)
    hay_section = normalize_text(section_path_text)
    hay_index = normalize_text(index_text)
    hay_text = normalize_text(text)
    hay_pics = normalize_text(pic_text)
    hay_note = normalize_text(source_issue_note)
    hay_heading = hay_title + hay_section
    hay_all = hay_heading + hay_index + hay_text + hay_pics + hay_note

    score = (1.0 / (1.0 + max(float(result.score), 0.0))) * 10.0

    pic_id = extract_pic_id(original_query)
    if pic_id:
        normalized_pic = normalize_text(pic_id)
        if normalized_pic in hay_pics:
            score += 120
        if normalized_pic in hay_title or normalized_pic in hay_index or normalized_pic in hay_text:
            score += 80

    normalized_query = normalize_text(original_query)
    if query_terms and not any(normalize_text(term) in hay_all for term in query_terms):
        score -= 28

    for term in query_terms:
        normalized_term = normalize_text(term)
        if not normalized_term:
            continue
        if normalized_term in hay_title:
            score += 28
        if normalized_term in hay_section:
            score += 24
        if normalized_term in hay_index:
            score += 14
        if normalized_term in hay_text:
            score += 8
        if normalized_term in hay_pics:
            score += 14
        if normalized_term in hay_note:
            score += 8

    for group in query_critical_term_groups(normalized_query):
        normalized_group = [normalize_text(term) for term in group if normalize_text(term)]
        if any(term in hay_heading for term in normalized_group):
            score += 55
        elif any(term in hay_index or term in hay_text for term in normalized_group):
            score += 30
        else:
            score -= 60

    intent_family_map = {
        "component": "component",
        "procedure": "procedure",
        "legal": "legal",
        "image_trace": "auxiliary",
        "ocr_audit": "ocr",
        "overview": "overview",
        "safety": "safety",
        "troubleshooting": "troubleshooting",
    }
    target_family = intent_family_map.get(intent)
    if target_family:
        if target_family in chunk_families:
            score += 18
        elif intent in {"component", "procedure", "legal", "safety", "troubleshooting"}:
            score -= 12

    if intent == "procedure":
        type_name = chunk_type.lower()
        if any(marker in type_name for marker in ("step", "task", "install", "start", "fueling", "mix", "maintenance", "cleaning")):
            score += 16
        for marker in ("安装", "设置", "使用", "步骤", "切换", "更换", "拆卸", "调节"):
            normalized_marker = normalize_text(marker)
            if normalized_marker in hay_title or normalized_marker in hay_index:
                score += 8
        if "safety" in chunk_families:
            score -= 30
    elif intent == "component":
        for marker in ("部件", "接口", "指示灯", "位置", "按键", "按钮", "追踪灯", "状态", "红色", "白色", "蓝色"):
            normalized_marker = normalize_text(marker)
            if normalized_marker in hay_title or normalized_marker in hay_index:
                score += 6
        if "safety" in chunk_families or "troubleshooting" in chunk_families:
            score -= 14
    elif intent == "legal":
        if chunk_type == "legal_clause":
            score += 35
        for marker in ("保修", "条款", "声明", "政策", "责任", "法规"):
            normalized_marker = normalize_text(marker)
            if normalized_marker in hay_title or normalized_marker in hay_index:
                score += 8
        if chunk_type != "legal_clause" and "保修" not in hay_title + hay_section + hay_index + hay_text:
            score -= 30
    elif intent == "safety":
        type_name = chunk_type.lower()
        if any(marker in type_name for marker in ("safety", "warning", "caution", "ppe", "label")):
            score += 18
        if "safety" in chunk_families:
            score += 24
        if "troubleshooting" in chunk_families or "component" in chunk_families:
            score -= 12
    elif intent == "troubleshooting":
        if "troubleshooting" in chunk_families:
            score += 28
        if "safety" in chunk_families:
            score -= 18
    elif intent == "image_trace":
        if retrieval_tier == "auxiliary":
            score += 12
        if chunk_type == "metadata_image_path":
            score += 18
        if pic_ids:
            score += 8
    elif intent == "ocr_audit":
        if source_issue_flags:
            score += 60
        if source_issue_note:
            score += 24
        if "(ocr)" in title.lower() or "ocr" in hay_title:
            score += 36
    elif intent == "overview":
        if chunk_type in SUPPORT_TYPE_BOOST or retrieval_tier == "support":
            score += 20
    elif intent == "general":
        if "safety" in chunk_families and not any(normalize_text(term) in normalized_query for term in SAFETY_TERMS):
            score -= 18

    if prefer_support and retrieval_tier == "support":
        score += 10
    if not prefer_support and retrieval_tier == "primary":
        score += 4

    return score


def should_scan_stage(
    reranked: Sequence[SearchResult],
    query_terms: Sequence[str],
    intent: str,
) -> bool:
    """Decide whether a lexical scan fallback should be triggered."""
    if not reranked:
        return True
    if intent in {"image_trace", "ocr_audit"}:
        return True

    top_score = lexical_score(
        reranked[0],
        original_query=" ".join(query_terms),
        query_terms=query_terms,
        intent=intent,
        prefer_support=False,
    )
    threshold = 20
    if intent in {"safety", "troubleshooting", "general"}:
        threshold = 24
    return top_score < threshold


def is_strong_hit(
    results: Sequence[SearchResult],
    query_terms: Sequence[str],
    intent: str,
) -> bool:
    """Decide whether a stage result is strong enough to stop fallback."""
    if not results:
        return False
    top_score = lexical_score(
        results[0],
        original_query=" ".join(query_terms),
        query_terms=query_terms,
        intent=intent,
        prefer_support=False,
    )
    if intent in {"image_trace", "ocr_audit"}:
        return top_score >= 40
    if intent in {"safety", "troubleshooting"}:
        return top_score >= 28
    return top_score >= 22


def collect_source_warnings(
    hits: Sequence[SearchResult],
    support_hits: Sequence[SearchResult],
) -> list[str]:
    """Collect OCR/source-quality warnings from matched chunks."""
    warnings: list[str] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()

    for result in [*hits, *support_hits]:
        metadata = result.metadata or {}
        flags = metadata.get("source_issue_flags") or []
        note = metadata.get("source_issue_note") or ""
        title = metadata.get("title") or metadata.get("chunk_id") or "未命名片段"
        if not flags and not note:
            continue

        flags_tuple = tuple(flag for flag in flags if isinstance(flag, str))
        key = (str(title), flags_tuple, str(note))
        if key in seen:
            continue
        seen.add(key)

        warning = f"片段“{title}”存在源数据风险"
        if flags_tuple:
            warning += f"（{', '.join(flags_tuple)}）"
        if note:
            warning += f"：{note}"
        warnings.append(warning)

    return warnings


def post_filter_results(
    results: Sequence[SearchResult],
    intent: str,
    query_terms: Sequence[str],
) -> list[SearchResult]:
    """Apply stricter post-filters for noisy intents."""
    if intent == "ocr_audit":
        return [result for result in results if is_ocr_candidate(result, query_terms)]
    if intent == "legal":
        return [result for result in results if is_legal_candidate(result, query_terms)]
    return list(results)


def is_ocr_candidate(result: SearchResult, query_terms: Sequence[str]) -> bool:
    """Keep OCR audit results tightly focused on flagged or explicitly OCR-marked chunks."""
    metadata = result.metadata or {}
    if metadata.get("retrieval_tier") in {"auxiliary", "support"}:
        return False

    title = str(metadata.get("title") or metadata.get("section_title") or "")
    section_path = metadata.get("section_path") or []
    section_path_text = " > ".join(str(part) for part in section_path) if isinstance(section_path, list) else str(section_path)
    index_text = str(metadata.get("index_text") or "")
    note = str(metadata.get("source_issue_note") or "")
    flags = metadata.get("source_issue_flags") or []

    haystack = normalize_text(" ".join([title, section_path_text, index_text, note]))
    markers = [normalize_text(marker) for marker in OCR_MARKERS]
    normalized_terms = [normalize_text(term) for term in query_terms]

    ocr_like_flags = {
        "ocr_corruption",
        "possible_missing_content",
        "truncated_list",
        "header_missing",
        "ambiguous_image_binding",
    }

    if any(isinstance(flag, str) and flag in ocr_like_flags for flag in flags):
        return True
    if any(marker in haystack for marker in markers):
        return True
    if any(term and term in haystack for term in normalized_terms):
        return True
    return False


def is_legal_candidate(result: SearchResult, query_terms: Sequence[str]) -> bool:
    """Keep legal queries focused on legal/appendix/support-policy content."""
    metadata = result.metadata or {}
    if metadata.get("retrieval_tier") == "auxiliary":
        return False

    title = str(metadata.get("title") or metadata.get("section_title") or "")
    section_path = metadata.get("section_path") or []
    section_path_text = " > ".join(str(part) for part in section_path) if isinstance(section_path, list) else str(section_path)
    chunk_type = str(metadata.get("chunk_type") or "")

    haystack = normalize_text(" ".join([title, section_path_text]))
    markers = [normalize_text(marker) for marker in LEGAL_MARKERS]
    legal_query_terms = [
        normalize_text(term)
        for term in query_terms
        if any(normalize_text(marker) in normalize_text(term) for marker in LEGAL_MARKERS)
    ]

    if chunk_type == "legal_clause":
        return True
    if any(marker in haystack for marker in markers):
        return True
    return any(term and term in haystack for term in legal_query_terms)


def trim_results_for_intent(results: Sequence[SearchResult], intent: str) -> list[SearchResult]:
    """Trim noisy result lists more aggressively for special intents."""
    if intent == "ocr_audit":
        return list(results[:1])
    if intent in {"legal", "safety", "troubleshooting"}:
        return list(results[:3])
    return list(results)


def extract_query_terms(query: str) -> list[str]:
    """Extract high-signal terms from the query."""
    if detect_lang(query) == "en":
        return extract_english_query_terms(query)

    import jieba.analyse

    # 1. TF-IDF 提取关键词（不依赖任何词表）
    keywords = jieba.analyse.extract_tags(query, topK=8, withWeight=False)

    # 2. 补充图片 ID
    pic_id = extract_pic_id(query)
    if pic_id:
        keywords.insert(0, pic_id)

    # 3. 补充 profile 术语（领域加权）
    normalized = normalize_text(query)
    keywords.extend(match_profile_terms_in_query(normalized, limit=4))

    # 4. 去重
    return list(dict.fromkeys(keywords))[:16]


def expand_query_terms_with_synonyms(query_terms: Sequence[str]) -> list[str]:
    """Add compact manual-domain synonyms while preserving term priority."""
    expanded: list[str] = []
    for term in query_terms:
        normalized_term = normalize_text(term)
        if not normalized_term:
            continue
        expanded.append(term)
        for key, synonyms in QUERY_SYNONYMS.items():
            normalized_key = normalize_text(key)
            if normalized_key in normalized_term or (len(normalized_term) >= 3 and normalized_term in normalized_key):
                expanded.extend(synonyms)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in expanded:
        normalized = normalize_text(term)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(term)
    return deduped


def extract_pic_id(query: str) -> str | None:
    """Extract a picture identifier from the query if present."""
    match = PIC_ID_RE.search(query)
    if match:
        return match.group(1)

    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    if not tokens:
        return None

    known_prefixes = {
        prefix
        for profile in load_doc_profiles()
        for prefix in profile.pic_prefixes
    }
    for token in tokens:
        normalized = normalize_text(token)
        if any(normalized.startswith(prefix) for prefix in known_prefixes):
            return str(token)
    return None


def merge_ranked_results(primary: Sequence[SearchResult], fallback: Sequence[SearchResult]) -> list[SearchResult]:
    """Merge two ranked result lists without duplicate chunk ids."""
    merged: list[SearchResult] = []
    seen: set[str] = set()
    for result in [*primary, *fallback]:
        if result.id in seen:
            continue
        seen.add(result.id)
        merged.append(result)
    return merged


def deduplicate_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Deduplicate search results by chunk id."""
    deduped: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        if result.id in seen:
            continue
        seen.add(result.id)
        deduped.append(result)
    return deduped


def search_result_to_document(result: SearchResult) -> Document:
    """Convert a SearchResult back into a LangChain Document."""
    if _evidence_search_result_to_document is not None:
        return cast(Document, _evidence_search_result_to_document(result))

    metadata = dict(result.metadata or {})
    metadata["score"] = result.score
    metadata["evidence_schema_version"] = EVIDENCE_SCHEMA_VERSION
    return Document(page_content=result.content, metadata=metadata)


def bundle_to_evidence_payload(
    bundle: RetrievalBundle,
    query: str | None = None,
) -> dict[str, Any]:
    """Convert retrieval output into a stable evidence payload for upper layers."""
    if _evidence_bundle_to_evidence_payload is not None:
        return cast(
            dict[str, Any],
            _evidence_bundle_to_evidence_payload(cast(Any, bundle), query=query),
        )

    primary_hits = [
        search_result_to_evidence_hit(result, rank=index, role="primary")
        for index, result in enumerate(bundle.hits, start=1)
    ]
    support_hits = [
        search_result_to_evidence_hit(result, rank=index, role="support")
        for index, result in enumerate(bundle.support_hits, start=1)
    ]

    all_hits = [*primary_hits, *support_hits]
    all_pic_ids = unique_flatten(hit.get("pic_ids") or [] for hit in all_hits)
    all_image_paths = unique_flatten(hit.get("image_paths") or [] for hit in all_hits)
    source_issue_flags = unique_flatten(hit.get("source_issue_flags") or [] for hit in all_hits)

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "query": query,
        "intent": bundle.intent,
        "retrieval_stage": bundle.retrieval_stage,
        "warnings": bundle.warnings,
        "summary": {
            "primary_count": len(primary_hits),
            "support_count": len(support_hits),
            "pic_ids": all_pic_ids,
            "image_paths": all_image_paths,
            "source_issue_flags": source_issue_flags,
        },
        "hits": primary_hits,
        "support_hits": support_hits,
    }


def search_result_to_evidence_hit(
    result: SearchResult,
    rank: int,
    role: str,
) -> dict[str, Any]:
    """Convert one search result into a compact evidence hit."""
    if _evidence_search_result_to_evidence_hit is not None:
        return cast(
            dict[str, Any],
            _evidence_search_result_to_evidence_hit(result, rank=rank, role=role),
        )

    metadata = result.metadata or {}
    return {
        "rank": rank,
        "role": role,
        "chunk_id": metadata.get("chunk_id") or result.id,
        "doc_id": metadata.get("doc_id"),
        "doc_name": metadata.get("doc_name"),
        "title": metadata.get("title") or metadata.get("section_title"),
        "section_path": metadata.get("section_path") or [],
        "retrieval_tier": metadata.get("retrieval_tier"),
        "chunk_type": metadata.get("chunk_type"),
        "parent_chunk_id": metadata.get("parent_chunk_id"),
        "score": result.score,
        "pic_ids": metadata.get("pic_ids") or [],
        "image_paths": display_path_list(metadata.get("image_paths") or []),
        "source_file": metadata.get("source_file") or metadata.get("_source"),
        "source_lines": metadata.get("source_lines") or [],
        "source_quality": metadata.get("source_quality"),
        "source_issue_flags": metadata.get("source_issue_flags") or [],
        "source_issue_note": metadata.get("source_issue_note") or "",
        "text": result.content,
    }


def unique_flatten(items: Iterable[Iterable[Any]]) -> list[Any]:
    """Flatten nested iterables while preserving first-seen order."""
    if _evidence_unique_flatten is not None:
        return cast(list[Any], _evidence_unique_flatten(items))

    flattened: list[Any] = []
    seen: set[str] = set()
    for group in items:
        for item in group:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            flattened.append(item)
    return flattened


def display_path_list(value: Any) -> list[str]:
    """Render project-local absolute paths as portable relative paths."""
    if _evidence_display_path_list is not None:
        return cast(list[str], _evidence_display_path_list(value))

    values = value if isinstance(value, list | tuple | set) else [value]
    rendered_paths: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        normalized = text.replace("\\", "/")
        try:
            path = Path(text)
            if path.is_absolute():
                normalized = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except (OSError, RuntimeError, ValueError):
            pass
        rendered_paths.append(normalized)
    return rendered_paths


def image_reference_lines(pic_ids: Sequence[Any], image_paths: Sequence[str]) -> list[str]:
    """Pair picture ids with paths so answers keep non-empty image placeholders."""
    if _evidence_image_reference_lines is not None:
        return cast(list[str], _evidence_image_reference_lines(pic_ids, image_paths))

    pic_id_texts = [str(pic_id).strip() for pic_id in pic_ids if str(pic_id).strip()]
    lines: list[str] = []
    for index, image_path in enumerate(image_paths):
        image_stem = Path(str(image_path)).stem
        pic_id = next(
            (
                candidate
                for candidate in pic_id_texts
                if candidate == image_stem or candidate in image_stem
            ),
            pic_id_texts[index] if index < len(pic_id_texts) else image_stem,
        )
        lines.append(f"- ![{pic_id}]({image_path})" if pic_id else f"- {image_path}")
    return lines


def format_bundle(bundle: RetrievalBundle, query: str | None = None) -> str:
    """Format retrieval results into a model-readable context block."""
    if _evidence_format_bundle is not None:
        return str(_evidence_format_bundle(cast(Any, bundle), query=query))

    evidence = bundle_to_evidence_payload(bundle, query=query)
    sections: list[str] = [
        f"【证据结构】{evidence['schema_version']}",
        f"【检索意图】{bundle.intent}",
        f"【检索阶段】{bundle.retrieval_stage}",
        "【证据摘要】"
        f"主命中 {evidence['summary']['primary_count']} 条；"
        f"补充上下文 {evidence['summary']['support_count']} 条；"
        f"图片 {len(evidence['summary']['pic_ids'])} 张。",
    ]

    if bundle.warnings:
        sections.append("【检索提示】\n" + "\n".join(f"- {warning}" for warning in bundle.warnings))

    if bundle.hits:
        sections.append("【主命中】\n" + format_search_results(bundle.hits))

    if bundle.support_hits:
        sections.append("【补充上下文】\n" + format_search_results(bundle.support_hits))

    return "\n\n".join(sections)


def format_search_results(results: Sequence[SearchResult]) -> str:
    """Format search results into readable context."""
    if _evidence_format_search_results is not None:
        return str(_evidence_format_search_results(results))

    parts: list[str] = []

    for index, result in enumerate(results, 1):
        metadata = result.metadata or {}
        doc_name = metadata.get("doc_name") or ""
        title = metadata.get("title") or metadata.get("section_title") or ""
        section_path = metadata.get("section_path") or []
        section_path_str = " > ".join(str(part) for part in section_path if part) if isinstance(section_path, list) else ""
        source_lines = metadata.get("source_lines") or []
        image_paths = display_path_list(metadata.get("image_paths") or [])
        chunk_id = metadata.get("chunk_id") or result.id
        retrieval_tier = metadata.get("retrieval_tier") or ""
        chunk_type = metadata.get("chunk_type") or ""

        block = [f"【参考资料 {index}】"]
        if doc_name:
            block.append(f"手册: {doc_name}")
        if title:
            block.append(f"标题: {title}")
        if section_path_str:
            block.append(f"章节路径: {section_path_str}")
        if retrieval_tier or chunk_type:
            block.append(f"层级: {retrieval_tier} / {chunk_type}")
        if chunk_id:
            block.append(f"切片ID: {chunk_id}")
        if source_lines:
            block.append(f"行号: {source_lines}")
        image_refs = image_reference_lines(metadata.get("pic_ids") or [], image_paths)
        if image_refs:
            block.append("相关图片，引用时必须使用以下 Markdown 格式:")
            block.extend(image_refs)
        block.append(f"内容:\n{result.content}")
        parts.append("\n".join(block))

    return "\n\n".join(parts)


def build_evidence_fallback_answer(results: Sequence[SearchResult]) -> str:
    """Build a compact answer from retrieved evidence when the model times out."""
    evidence_lines: list[str] = []
    for result in results[:3]:
        metadata = result.metadata or {}
        title = str(metadata.get("title") or metadata.get("section_title") or "").strip()
        text = " ".join(str(result.content or metadata.get("text") or "").split())
        if not text:
            continue
        if len(text) > 180:
            text = text[:180].rstrip() + "..."
        evidence_lines.append(f"{title}: {text}" if title else text)

    if not evidence_lines:
        return "根据当前已完成的信息，暂时没有检索到足够可靠的资料来给出完整结论。"
    return "根据已检索到的资料，简要结论如下：" + "；".join(evidence_lines)


def normalize_text(text: str) -> str:
    """Normalize text for heuristic matching."""
    return re.sub(r"\s+", "", text).lower()
