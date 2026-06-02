from __future__ import annotations

from app.retrieval import tier_policy


def test_recallable_tiers_are_primary_and_support() -> None:
    assert tier_policy.is_recallable_tier(tier_policy.PRIMARY_TIER)
    assert tier_policy.is_recallable_tier(tier_policy.SUPPORT_TIER)
    assert not tier_policy.is_recallable_tier(tier_policy.BIG_SUPPORT_TIER)
    assert not tier_policy.is_recallable_tier(tier_policy.AUXILIARY_TIER)
    assert not tier_policy.is_recallable_tier(None)


def test_evidence_parent_tiers_include_support_and_big_support() -> None:
    assert tier_policy.EVIDENCE_PARENT_TIERS == (
        tier_policy.SUPPORT_TIER,
        tier_policy.BIG_SUPPORT_TIER,
    )
    assert tier_policy.is_evidence_parent_tier(tier_policy.SUPPORT_TIER)
    assert tier_policy.is_evidence_parent_tier(tier_policy.BIG_SUPPORT_TIER)
    assert not tier_policy.is_evidence_parent_tier(tier_policy.PRIMARY_TIER)
    assert not tier_policy.is_evidence_parent_tier(tier_policy.AUXILIARY_TIER)


def test_descendant_query_tiers_include_hierarchy_but_not_auxiliary() -> None:
    assert tier_policy.DESCENDANT_QUERY_TIERS == (
        tier_policy.PRIMARY_TIER,
        tier_policy.SUPPORT_TIER,
        tier_policy.BIG_SUPPORT_TIER,
    )
    assert tier_policy.AUXILIARY_TIER not in tier_policy.DESCENDANT_QUERY_TIERS


def test_big_support_helper_matches_only_big_support() -> None:
    assert tier_policy.is_big_support_tier(tier_policy.BIG_SUPPORT_TIER)
    assert not tier_policy.is_big_support_tier(tier_policy.SUPPORT_TIER)
    assert not tier_policy.is_big_support_tier(None)
