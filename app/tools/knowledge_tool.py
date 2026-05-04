"""Knowledge retrieval tool with intent routing, reranking, and fallback scan."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_search_service import SearchResult, vector_search_service


EVIDENCE_SCHEMA_VERSION = "retrieval_evidence_v1"
PIC_ID_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)", re.IGNORECASE)
PROFILE_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]{1,}|[\u4e00-\u9fff]{2,16}")
PROFILE_SCAN_LIMIT = 4096
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

ACTION_TERMS = [
    "安装",
    "设置",
    "使用",
    "步骤",
    "切换",
    "更换",
    "拆卸",
    "调节",
    "范围",
    "问题",
]

INTENT_STAGE_CONFIG: dict[str, list[dict[str, Any]]] = {
    "component": [
        {"name": "primary_route", "tiers": ["primary"], "types": ["text_image_atomic", "feature_group", "subsection"]},
        {"name": "support_route", "tiers": ["support"], "types": ["section_summary", "procedure_overview"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "types": None},
    ],
    "procedure": [
        {"name": "primary_route", "tiers": ["primary"], "types": ["procedure_step", "subsection", "feature_group"]},
        {"name": "support_route", "tiers": ["support"], "types": ["procedure_overview", "section_summary"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "types": None},
    ],
    "legal": [
        {"name": "primary_route", "tiers": ["primary"], "types": ["legal_clause", "subsection"]},
        {"name": "support_route", "tiers": ["support"], "types": ["section_summary"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "types": None},
    ],
    "image_trace": [
        {"name": "auxiliary_direct", "tiers": ["auxiliary"], "types": ["metadata_image_path", "aux_navigation"]},
        {"name": "primary_route", "tiers": ["primary"], "types": ["text_image_atomic", "feature_group", "subsection"]},
        {"name": "expanded_route", "tiers": ["primary", "support", "auxiliary"], "types": None},
    ],
    "ocr_audit": [
        {"name": "auxiliary_direct", "tiers": ["auxiliary"], "types": ["aux_navigation", "metadata_image_path"]},
        {"name": "primary_route", "tiers": ["primary", "support"], "types": None},
        {"name": "expanded_route", "tiers": ["primary", "support", "auxiliary"], "types": None},
    ],
    "overview": [
        {"name": "support_route", "tiers": ["support"], "types": ["section_summary", "procedure_overview"]},
        {"name": "primary_route", "tiers": ["primary"], "types": ["subsection", "feature_group"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "types": None},
    ],
    "general": [
        {"name": "primary_route", "tiers": ["primary"], "types": ["subsection", "feature_group", "text_image_atomic", "procedure_step"]},
        {"name": "support_route", "tiers": ["support"], "types": ["section_summary", "procedure_overview"]},
        {"name": "expanded_route", "tiers": ["primary", "support"], "types": None},
    ],
}

SUPPORT_TYPE_BOOST = {"section_summary", "procedure_overview"}
LEGAL_MARKERS = ("保修", "法规", "声明", "责任", "政策", "支持与服务", "网站", "网址", "附录")
OCR_MARKERS = ("ocr", "识别", "截断", "缺失", "原文", "源数据", "英国用户")


@dataclass
class RetrievalBundle:
    """Container for retrieval pipeline outputs."""

    intent: str
    retrieval_stage: str
    hits: list[SearchResult]
    support_hits: list[SearchResult]
    warnings: list[str]

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


@dataclass(frozen=True)
class DocumentProfile:
    """Derived document routing profile built from indexed chunk metadata."""

    doc_id: str
    doc_name: str
    terms: tuple[str, ...]
    pic_prefixes: tuple[str, ...]
    chunk_types: tuple[str, ...]
    source_issue_flags: tuple[str, ...]
    chunk_count: int


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """Retrieve relevant manual knowledge for a user query."""
    try:
        logger.info("Knowledge retrieval called: query='{}'", query)
        bundle = routed_retrieve(query)
        docs = [search_result_to_document(result) for result in bundle.all_hits]

        if not docs:
            logger.warning("No relevant documents found after fallback.")
            return "没有找到相关信息。", []

        context = format_bundle(bundle, query=query)
        logger.info(
            "Retrieved {} documents, intent={}, stage={}",
            len(docs),
            bundle.intent,
            bundle.retrieval_stage,
        )
        return context, docs
    except Exception as exc:
        logger.error(f"Knowledge retrieval failed: {exc}")
        return f"检索知识时发生错误: {str(exc)}", []


def routed_retrieve(query: str) -> RetrievalBundle:
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
            chunk_types=stage["types"],
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
            chunk_types=None,
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
    )


def detect_intent(query: str) -> str:
    """Heuristically classify the query intent."""
    normalized = normalize_text(query)
    if extract_pic_id(query) or any(keyword.lower() in normalized for keyword in ("图片", "图示", "配图", "对应图", "pic")):
        return "image_trace"
    if any(keyword.lower() in normalized for keyword in ("ocr", "原文", "源数据", "识别错误", "乱码", "截断", "缺失")):
        return "ocr_audit"
    if any(keyword.lower() in normalized for keyword in ("保修", "法规", "声明", "责任", "政策", "支持与服务", "网站", "网址")):
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
        return tuple()

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
        if chunk_type:
            bucket["chunk_types"].add(chunk_type)

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
    if intent == "ocr_audit" and query_terms:
        variants.append(f"{' '.join(query_terms)} OCR 原文 识别错误 缺失")
    if intent == "image_trace":
        pic_id = extract_pic_id(query)
        if pic_id:
            variants.append(pic_id)
            variants.append(f"{pic_id} 图片 对应")
        elif query_terms:
            variants.append(f"{' '.join(query_terms)} 图片 配图")

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
    chunk_types: Sequence[str] | None,
    top_k: int,
) -> list[SearchResult]:
    """Execute vector search and optional metadata scan rerank for a stage."""
    aggregated: list[SearchResult] = []
    stage_fetch_k = max(top_k * 3, 8)

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
                "Vector stage failed: query='{}', doc_id={}, tiers={}, chunk_types={}, error={}",
                variant,
                doc_id,
                tiers,
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

    if should_scan_stage(reranked, query_terms, intent):
        scanned = run_scan_stage(
            original_query=original_query,
            query_terms=query_terms,
            intent=intent,
            doc_id=doc_id,
            tiers=tiers,
            chunk_types=chunk_types,
            top_k=stage_fetch_k,
        )
        merged = merge_ranked_results(reranked, scanned)
        return merged[:top_k]

    return reranked[:top_k]


def run_scan_stage(
    original_query: str,
    query_terms: Sequence[str],
    intent: str,
    doc_id: str | None,
    tiers: Sequence[str] | None,
    chunk_types: Sequence[str] | None,
    top_k: int,
) -> list[SearchResult]:
    """Run metadata-filtered full scan fallback and rerank locally."""
    try:
        scanned = filter_results_to_active_docs(
            vector_search_service.query_documents(
                doc_id=doc_id,
                retrieval_tiers=list(tiers) if tiers else None,
                chunk_types=list(chunk_types) if chunk_types else None,
                limit=max(top_k * 10, 64),
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
            chunk_types,
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

    pic_text = " ".join(str(pic_id) for pic_id in pic_ids)
    hay_title = normalize_text(title)
    hay_section = normalize_text(section_path_text)
    hay_index = normalize_text(index_text)
    hay_text = normalize_text(text)
    hay_pics = normalize_text(pic_text)
    hay_note = normalize_text(source_issue_note)

    score = (1.0 / (1.0 + max(result.score, 0.0))) * 10.0

    pic_id = extract_pic_id(original_query)
    if pic_id:
        normalized_pic = normalize_text(pic_id)
        if normalized_pic in hay_pics:
            score += 120
        if normalized_pic in hay_title or normalized_pic in hay_index or normalized_pic in hay_text:
            score += 80

    normalized_query = normalize_text(original_query)
    for term in query_terms:
        normalized_term = normalize_text(term)
        if not normalized_term:
            continue
        if normalized_term in hay_title:
            score += 18
        if normalized_term in hay_section:
            score += 12
        if normalized_term in hay_index:
            score += 10
        if normalized_term in hay_text:
            score += 6
        if normalized_term in hay_pics:
            score += 14
        if normalized_term in hay_note:
            score += 8

    if intent == "procedure":
        for marker in ("安装", "设置", "使用", "步骤", "切换", "更换", "拆卸", "调节"):
            normalized_marker = normalize_text(marker)
            if normalized_marker in hay_title or normalized_marker in hay_index:
                score += 8
    elif intent == "component":
        for marker in ("部件", "接口", "指示灯", "位置", "按键", "按钮", "追踪灯"):
            normalized_marker = normalize_text(marker)
            if normalized_marker in hay_title or normalized_marker in hay_index:
                score += 6
    elif intent == "legal":
        if chunk_type == "legal_clause":
            score += 35
        for marker in ("保修", "条款", "声明", "政策", "责任", "法规"):
            normalized_marker = normalize_text(marker)
            if normalized_marker in hay_title or normalized_marker in hay_index:
                score += 8
        if chunk_type != "legal_clause" and "保修" not in hay_title + hay_section + hay_index + hay_text:
            score -= 30
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
    return top_score < 20


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
    if intent == "legal":
        return list(results[:3])
    return list(results)


def extract_query_terms(query: str) -> list[str]:
    """Extract high-signal terms from the query."""
    matched_terms: list[str] = []
    normalized = normalize_text(query)

    pic_id = extract_pic_id(query)
    if pic_id:
        matched_terms.append(pic_id)

    matched_terms.extend(match_profile_terms_in_query(normalized))
    matched_terms.extend(term for term in ACTION_TERMS if normalize_text(term) in normalized)
    if matched_terms:
        return list(dict.fromkeys(matched_terms))

    alnum_terms = re.findall(r"[A-Za-z0-9_\-]{3,}", query)
    if alnum_terms:
        return alnum_terms[:4]

    generic_terms = [term for term in PROFILE_TERM_RE.findall(query) if len(normalize_text(term)) >= 2]
    if generic_terms:
        return list(dict.fromkeys(generic_terms))[:4]

    compact = re.sub(r"\s+", "", query)
    if compact:
        return [compact] if len(compact) <= 12 else [compact[:12]]
    return []


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
            return token
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
    metadata = dict(result.metadata or {})
    metadata["score"] = result.score
    metadata["evidence_schema_version"] = EVIDENCE_SCHEMA_VERSION
    return Document(page_content=result.content, metadata=metadata)


def bundle_to_evidence_payload(
    bundle: RetrievalBundle,
    query: str | None = None,
) -> dict[str, Any]:
    """Convert retrieval output into a stable evidence payload for upper layers."""
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
        "image_paths": metadata.get("image_paths") or [],
        "source_file": metadata.get("source_file") or metadata.get("_source"),
        "source_lines": metadata.get("source_lines") or [],
        "source_quality": metadata.get("source_quality"),
        "source_issue_flags": metadata.get("source_issue_flags") or [],
        "source_issue_note": metadata.get("source_issue_note") or "",
        "text": result.content,
    }


def unique_flatten(items: Iterable[Iterable[Any]]) -> list[Any]:
    """Flatten nested iterables while preserving first-seen order."""
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


def format_bundle(bundle: RetrievalBundle, query: str | None = None) -> str:
    """Format retrieval results into a model-readable context block."""
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
    parts: list[str] = []

    for index, result in enumerate(results, 1):
        metadata = result.metadata or {}
        doc_name = metadata.get("doc_name") or ""
        title = metadata.get("title") or metadata.get("section_title") or ""
        section_path = metadata.get("section_path") or []
        section_path_str = " > ".join(str(part) for part in section_path if part) if isinstance(section_path, list) else ""
        source_lines = metadata.get("source_lines") or []
        image_paths = metadata.get("image_paths") or []
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
        if image_paths:
            block.append("相关图片路径:")
            for image_path in image_paths:
                block.append(f"- {image_path}")
        block.append(f"内容:\n{result.content}")
        parts.append("\n".join(block))

    return "\n\n".join(parts)


def normalize_text(text: str) -> str:
    """Normalize text for heuristic matching."""
    return re.sub(r"\s+", "", text).lower()
