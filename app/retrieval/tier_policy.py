"""Central retrieval tier policy helpers."""

from __future__ import annotations

PRIMARY_TIER = "primary"
SUPPORT_TIER = "support"
BIG_SUPPORT_TIER = "big_support"
AUXILIARY_TIER = "auxiliary"

RECALL_INCLUDED_TIERS = (PRIMARY_TIER, SUPPORT_TIER)
EVIDENCE_PARENT_TIERS = (SUPPORT_TIER, BIG_SUPPORT_TIER)
DESCENDANT_QUERY_TIERS = (PRIMARY_TIER, SUPPORT_TIER, BIG_SUPPORT_TIER)


def is_recallable_tier(tier: str | None) -> bool:
    """Return whether a tier may participate in normal recall."""
    return tier in RECALL_INCLUDED_TIERS


def is_big_support_tier(tier: str | None) -> bool:
    """Return whether a tier is the special big support tier."""
    return tier == BIG_SUPPORT_TIER


def is_evidence_parent_tier(tier: str | None) -> bool:
    """Return whether a tier may be fetched as evidence parent context."""
    return tier in EVIDENCE_PARENT_TIERS


__all__ = [
    "AUXILIARY_TIER",
    "BIG_SUPPORT_TIER",
    "DESCENDANT_QUERY_TIERS",
    "EVIDENCE_PARENT_TIERS",
    "PRIMARY_TIER",
    "RECALL_INCLUDED_TIERS",
    "SUPPORT_TIER",
    "is_big_support_tier",
    "is_evidence_parent_tier",
    "is_recallable_tier",
]
