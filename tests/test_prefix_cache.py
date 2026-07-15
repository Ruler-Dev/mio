"""Tests for the prefix cache mechanism on MioEngine + runtime."""

from __future__ import annotations

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


def test_hybrid_prefix_cache_never_rewinds_recurrent_state():
    import mlx_lm.models.cache as cache_mod

    from mio.dflash.recurrent_rollback_cache import RecurrentRollbackCache
    from mio.engine import MioEngine

    cfg = MioConfig.default()
    eng = MioEngine(tier_config=cfg.tiers["small"])
    eng._prefix_cache_min_tokens = 1

    recurrent = RecurrentRollbackCache(size=2)
    attention = cache_mod.KVCache()
    attention.offset = 100
    cached = list(range(100))
    state = {
        "target_cache": [recurrent, attention],
        "draft_cache": [],
        "offset": 100,
    }
    eng._prefix_cache[tuple(cached)] = state

    # A divergent tail would require rewinding both cache types.  Recurrent
    # GDN state is not trimmable, so Mio must turn this into a cold miss.
    assert eng._prefix_cache_lookup(cached[:50] + [999, 1000]) is None
    assert attention.offset == 100
    assert tuple(cached) in eng._prefix_cache

    # Reusing the complete cached state for a strict prompt extension needs no
    # rewind and is therefore safe even for a hybrid target.
    hit = eng._prefix_cache_lookup(cached + [100, 101])
    assert hit is not None
    assert hit["offset"] == 100
    assert attention.offset == 100


def test_prefix_cache_invalidate():
    from mio.engine import MioEngine
    cfg = MioConfig.default()
    eng = MioEngine(tier_config=cfg.tiers["small"])
    eng._prefix_cache[tuple([1, 2])] = {}
    eng._last_prompt_tokens = [1, 2, 3]
    eng._prefix_cache_invalidate()
    assert len(eng._prefix_cache) == 0
    assert eng._last_prompt_tokens == []


def test_position_aware_draft_cache_truncation_drops_divergent_tail():
    """Sink/window draft caches must be truncated by RoPE position, not shape."""
    import mlx.core as mx

    from mio.dflash.model import ContextOnlyDraftKVCache
    from mio.engine import MioEngine

    cache = ContextOnlyDraftKVCache(sink_size=2, window_size=4)
    cache.keys = mx.arange(6, dtype=mx.float32).reshape(1, 1, 6, 1)
    cache.values = cache.keys
    cache.positions = mx.array([0, 1, 4996, 4997, 4998, 4999], dtype=mx.int32)
    cache.offset = 5000

    state = {"target_cache": [], "draft_cache": [cache], "offset": 5000}
    MioEngine._truncate_warm_state(state, 4000)

    assert cache.offset == 4000
    assert cache.keys.shape[2] == 2
    assert cache.values.shape[2] == 2
    assert cache.positions.tolist() == [0, 1]


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
