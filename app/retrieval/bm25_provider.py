"""Jieba/rank-bm25 backed lexical recall provider."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from typing import Any, Protocol, TypeVar, cast

from app.retrieval.tier_policy import BIG_SUPPORT_TIER

SearchResultT = TypeVar("SearchResultT")
Tokenizer = Callable[[str], Iterable[str]]

logger = logging.getLogger(__name__)

_EMPTY_DOCUMENT_TOKEN = "__bm25_empty_document__"
_ENGLISH_PRESERVED_TERMS = {
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
_ENGLISH_STOP_WORDS = {
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
} - _ENGLISH_PRESERVED_TERMS
_ENGLISH_IRREGULAR_FORMS = {
    "children": "child",
    "feet": "foot",
    "men": "man",
    "teeth": "tooth",
    "women": "woman",
}


class BM25Index(Protocol):
    """Small protocol for rank_bm25-compatible scorers."""

    def get_scores(self, query_tokens: Sequence[str]) -> Iterable[Any]:
        """Return one score per indexed document."""


BM25Factory = Callable[[list[list[str]]], BM25Index]


class JiebaBM25Provider:
    """BM25 provider that indexes SearchResult-like chunks with language-aware tokens."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        bm25_factory: BM25Factory | None = None,
    ) -> None:
        self._tokenizer = tokenizer or _load_jieba_tokenizer()
        self._bm25_factory = bm25_factory or _load_bm25_factory()
        self._results: list[Any] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Index | None = None

    @property
    def document_count(self) -> int:
        """Return the number of indexed SearchResult-like objects."""
        return len(self._results)

    def build_index(self, all_results: Iterable[SearchResultT]) -> None:
        """Build an in-memory BM25 index over SearchResult-like chunks."""
        self._results = []
        self._corpus = []
        self._bm25 = None

        results = list(all_results)
        corpus = [self._tokens_for_result(result) for result in results]
        bm25 = self._bm25_factory(corpus) if corpus else None

        self._results = list(results)
        self._corpus = corpus
        self._bm25 = bm25

    def search(
        self,
        query: str,
        *,
        top_k: int,
        doc_id: str | None = None,
        retrieval_tiers: Sequence[str] | None = None,
        chunk_types: Sequence[str] | None = None,
        language: str | None = None,
    ) -> list[Any]:
        """Search indexed chunks and return matching SearchResult-like objects."""
        if self._bm25 is None or not self._results:
            return []

        language_filter = _normalise_language(language)
        query_tokens = self._tokenize(query, language=language_filter)
        if not query_tokens:
            return []

        safe_top_k = _positive_int(top_k)
        tier_filter = _normalise_filter_values(retrieval_tiers)
        type_filter = _normalise_filter_values(chunk_types)
        raw_scores = list(self._bm25.get_scores(query_tokens))

        scored: list[tuple[float, int, Any]] = []
        for index, result in enumerate(self._results):
            if index >= len(raw_scores):
                break
            score = _safe_float(raw_scores[index])
            if score <= 0:
                continue
            if not _matches_filters(
                result,
                doc_id=doc_id,
                retrieval_tiers=tier_filter,
                chunk_types=type_filter,
                language=language_filter,
            ):
                continue
            scored.append((score, index, result))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [_set_result_score(result, score) for score, _, result in scored[:safe_top_k]]

    def _tokens_for_result(self, result: object) -> list[str]:
        index_text = _result_index_text(result)
        tokens = self._tokenize(
            index_text,
            language=_result_language(result) or _detect_language(index_text),
        )
        return tokens or [_EMPTY_DOCUMENT_TOKEN]

    def _tokenize(self, text: object, *, language: str | None = None) -> list[str]:
        if text is None:
            return []
        text_value = str(text)
        token_language = _normalise_language(language) or _detect_language(text_value)
        if token_language == "en":
            return _english_tokens(text_value)

        tokens: list[str] = []
        for raw_token in self._tokenizer(text_value):
            token = str(raw_token or "").strip().lower()
            if token:
                tokens.append(token)
        return tokens


def init_bm25_provider() -> None:
    """Initialize and register the process-local BM25 provider.

    Initialization is best-effort: failures are logged and clear the provider so
    service startup can continue without lexical recall.
    """
    try:
        from app.retrieval.recall import set_bm25_provider
    except Exception as exc:  # pragma: no cover - import failures are environment-specific.
        logger.warning("BM25 provider initialization failed: %s", exc, exc_info=True)
        return

    try:
        from app.services.vector_search_service import vector_search_service

        provider = JiebaBM25Provider()
        provider.build_index(vector_search_service.query_all_documents())
        set_bm25_provider(provider)
        logger.info("BM25 provider initialized with %s documents", provider.document_count)
    except Exception as exc:
        logger.warning(
            "BM25 provider initialization failed; continuing without BM25: %s",
            exc,
            exc_info=True,
        )
        try:
            set_bm25_provider(None)
        except Exception as clear_exc:  # pragma: no cover - defensive startup guard.
            logger.warning("Failed to clear BM25 provider: %s", clear_exc, exc_info=True)


def _load_jieba_tokenizer() -> Tokenizer:
    jieba = import_module("jieba")
    cut = getattr(jieba, "cut", None)
    if not callable(cut):
        raise RuntimeError("jieba.cut is unavailable")
    return cast(Tokenizer, cut)


def _load_bm25_factory() -> BM25Factory:
    rank_bm25 = import_module("rank_bm25")
    bm25_okapi = getattr(rank_bm25, "BM25Okapi", None)
    if not callable(bm25_okapi):
        raise RuntimeError("rank_bm25.BM25Okapi is unavailable")
    return cast(BM25Factory, bm25_okapi)


def _result_index_text(result: object) -> str:
    metadata = _metadata_for_result(result)
    parts = [
        metadata.get("text"),
        getattr(result, "content", None),
        metadata.get("content"),
        metadata.get("title"),
        metadata.get("section_title"),
        metadata.get("section_path"),
        metadata.get("index_text"),
        metadata.get("summary"),
        metadata.get("keywords"),
    ]
    return " ".join(part for value in parts for part in _string_parts(value))


def _result_language(result: object) -> str | None:
    metadata = _metadata_for_result(result)
    return _normalise_language(_lookup_value(result, metadata, ("language", "lang")))


def _detect_language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def _english_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+", text.lower()):
        if not token or token in _ENGLISH_STOP_WORDS:
            continue
        tokens.extend(_english_token_forms(token))
    return tokens


def _english_token_forms(token: str) -> list[str]:
    """Return BM25 lexical forms for light English morphology matching."""
    forms = [token]
    if token in _ENGLISH_PRESERVED_TERMS:
        return forms

    normalized = _normalise_english_token(token)
    if normalized and normalized != token and normalized not in forms:
        forms.append(normalized)
    return forms


def _normalise_english_token(token: str) -> str:
    if token in _ENGLISH_IRREGULAR_FORMS:
        return _ENGLISH_IRREGULAR_FORMS[token]
    if len(token) <= 3 or token.isdigit():
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return _normalise_english_suffix_stem(token[:-3])
    if token.endswith("ied") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ed") and len(token) > 4:
        return _normalise_english_suffix_stem(token[:-2])
    if token.endswith(("sses", "xes", "zes", "ches", "shes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _normalise_english_suffix_stem(stem: str) -> str:
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
        return stem[:-1]
    return stem


def _string_parts(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Mapping):
        return [
            part
            for key, nested_value in value.items()
            for part in _string_parts(key) + _string_parts(nested_value)
        ]
    if isinstance(value, Iterable):
        return [part for item in value for part in _string_parts(item)]
    text = str(value).strip()
    return [text] if text else []


def _matches_filters(
    result: object,
    *,
    doc_id: str | None,
    retrieval_tiers: set[str],
    chunk_types: set[str],
    language: str | None,
) -> bool:
    metadata = _metadata_for_result(result)
    tier = _lookup_value(result, metadata, ("retrieval_tier", "retrieval_tiers", "tier"))
    if BIG_SUPPORT_TIER in _normalise_candidate_values(tier):
        return False
    if doc_id is not None and not _value_matches(
        _lookup_value(result, metadata, ("doc_id", "document_id")),
        {doc_id},
    ):
        return False
    if retrieval_tiers and not _value_matches(
        _lookup_value(result, metadata, ("retrieval_tier", "retrieval_tiers", "tier")),
        retrieval_tiers,
    ):
        return False
    if chunk_types and not _value_matches(
        _lookup_value(
            result,
            metadata,
            ("chunk_type", "chunk_types", "type", "chunk_family", "family"),
        ),
        chunk_types,
    ):
        return False
    if language is not None and not _language_matches(
        _lookup_value(result, metadata, ("language", "lang")),
        language,
    ):
        return False
    return True


def _metadata_for_result(result: object) -> Mapping[str, Any]:
    metadata = getattr(result, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _lookup_value(
    result: object,
    metadata: Mapping[str, Any],
    keys: Sequence[str],
) -> object:
    for key in keys:
        if key in metadata:
            return metadata[key]
    for key in keys:
        value = getattr(result, key, None)
        if value is not None:
            return value
    return None


def _normalise_filter_values(values: Sequence[str] | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        return {values.strip()} if values.strip() else set()
    return {str(value).strip() for value in values if str(value).strip()}


def _normalise_language(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-")
    if not normalized:
        return None
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("zh") or normalized in {"cn", "chinese"}:
        return "zh"
    return None


def _value_matches(value: object, allowed: set[str]) -> bool:
    if not allowed:
        return True
    candidates = _normalise_candidate_values(value)
    return any(candidate in allowed for candidate in candidates)


def _language_matches(value: object, allowed: str) -> bool:
    candidates = {
        normalized
        for candidate in _normalise_candidate_values(value)
        if (normalized := _normalise_language(candidate)) is not None
    }
    return allowed in candidates


def _normalise_candidate_values(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        stripped = value.strip()
        return {stripped} if stripped else set()
    if isinstance(value, Iterable):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()} if str(value).strip() else set()


def _positive_int(value: object) -> int:
    try:
        if isinstance(value, bool):
            parsed = int(value)
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            parsed = int(value)
        elif isinstance(value, (str, bytes, bytearray)):
            parsed = int(value)
        else:
            return 1
    except (TypeError, ValueError, OverflowError):
        return 1
    return max(parsed, 1)


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _set_result_score(result: SearchResultT, score: float) -> SearchResultT:
    try:
        result_with_score = cast(Any, result)
        result_with_score.score = score
    except Exception:
        logger.debug("Unable to set BM25 score on result %r", result, exc_info=True)
    return result
