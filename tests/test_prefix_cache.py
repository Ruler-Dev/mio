"""Tests for the prefix cache mechanism on MioEngine + runtime."""

from __future__ import annotations

import pytest

from mio.config import MioConfig


def test_longest_common_prefix():
    from mio.engine import MioEngine
    cfg = MioConfig.default()
    eng = MioEngine(tier_config=cfg.tiers["small"])
    assert eng._longest_common_prefix([1, 2, 3, 4], [1, 2, 5, 4]) == 2
    assert eng._longest_common_prefix([1, 2, 3], [1, 2, 3]) == 3
    assert eng._longest_common_prefix([1, 2], []) == 0
    assert eng._longest_common_prefix([], [1, 2]) == 0
    assert eng._longest_common_prefix([1, 2, 3], [4, 5, 6]) == 0


def test_prefix_cache_enabled_gating():
    from mio.engine import MioEngine
    cfg = MioConfig.default()
    tier = cfg.tiers["small"]
    eng = MioEngine(tier_config=tier)
    # Default: PQ4 is on, which disables prefix cache
    assert eng._prefix_cache_enabled() is False
    # Disabling PQ re-enables prefix cache
    tier.pq_bits = 16
    assert eng._prefix_cache_enabled() is True
    # TQ4 disables it
    tier.tq_bits = 4
    assert eng._prefix_cache_enabled() is False
    tier.tq_bits = 16
    # BMP disables it
    tier.bmp_paths = 2
    assert eng._prefix_cache_enabled() is False
    tier.bmp_paths = 1
    assert eng._prefix_cache_enabled() is True


def test_prefix_cache_lookup_finds_longest_match():
    from mio.engine import MioEngine
    cfg = MioConfig.default()
    eng = MioEngine(tier_config=cfg.tiers["small"])
    # Disable min-token threshold for the unit test (it filters out tiny matches
    # in production to avoid negative cache speedups, but we want to exercise
    # the lookup+truncation logic here).
    eng._prefix_cache_min_tokens = 1

    # Populate cache with two entries. Empty lists for cache structures so
    # _truncate_warm_state is a no-op.
    eng._prefix_cache[tuple([1, 2, 3])] = {"target_cache": [], "draft_cache": [], "offset": 3}
    eng._prefix_cache[tuple([1, 2, 3, 4, 5])] = {"target_cache": [], "draft_cache": [], "offset": 5}

    # [1,2,3,4,5,6,7] matches the length-5 entry fully → offset becomes 5
    hit = eng._prefix_cache_lookup([1, 2, 3, 4, 5, 6, 7])
    assert hit is not None
    assert hit["offset"] == 5

    # Re-populate (lookup rents entries out)
    eng._prefix_cache[tuple([1, 2, 3])] = {"target_cache": [], "draft_cache": [], "offset": 3}
    eng._prefix_cache[tuple([1, 2, 3, 4, 5])] = {"target_cache": [], "draft_cache": [], "offset": 5}

    # [1,2,3,9,...] matches the length-5 entry for first 3 tokens AND the
    # length-3 entry fully — both yield match=3. Tie-break picks whichever
    # the lookup hits first with match > best_match.
    hit = eng._prefix_cache_lookup([1, 2, 3, 9, 8, 7])
    assert hit is not None
    assert hit["offset"] == 3

    # Re-populate
    eng._prefix_cache[tuple([1, 2, 3])] = {"target_cache": [], "draft_cache": [], "offset": 3}

    # No match at all → None
    hit = eng._prefix_cache_lookup([99, 100, 101])
    assert hit is None


def test_prefix_cache_invalidate():
    from mio.engine import MioEngine
    cfg = MioConfig.default()
    eng = MioEngine(tier_config=cfg.tiers["small"])
    eng._prefix_cache[tuple([1, 2])] = {}
    eng._last_prompt_tokens = [1, 2, 3]
    eng._prefix_cache_invalidate()
    assert len(eng._prefix_cache) == 0
    assert eng._last_prompt_tokens == []


def test_prefix_cache_store_skips_below_threshold():
    """Common prefix < min_tokens shouldn't populate the cache (would not pay back)."""
    from mio.engine import MioEngine
    cfg = MioConfig.default()
    eng = MioEngine(tier_config=cfg.tiers["small"])
    eng._prefix_cache_min_tokens = 100
    eng._last_prompt_tokens = [1, 2, 3]  # prev prompt
    # Current prompt shares only 3 tokens with prev → below 100 threshold
    fake_state = {"target_cache": [], "draft_cache": [], "offset": 0}
    eng._prefix_cache_store([1, 2, 3, 99, 100], fake_state)
    assert len(eng._prefix_cache) == 0
