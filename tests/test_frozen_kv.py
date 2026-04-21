"""Tests for mio.frozen_kv.

Scope: deterministic, no model load. Uses mlx_lm's real cache classes
with synthesized small arrays so round-trips exercise the actual .state /
.meta_state contracts the live engine will use.

The tests are organized around the contracts the engine relies on:
  1. Fingerprint is a pure function — same inputs, same hash, always.
  2. Config changes invalidate — different model/pq/tq/ctx → different hash.
  3. Round-trip preserves contents exactly for every cache type we ship.
  4. try_load refuses mismatched / corrupt / wrong-version snapshots
     silently (returns None, never raises).
  5. prune_cache_dir evicts oldest under both caps.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models import cache as mlx_cache

from mio.dflash.recurrent_rollback_cache import RecurrentRollbackCache
from mio.frozen_kv import (
    FROZEN_KV_VERSION,
    common_prefix_fingerprints,
    fingerprint,
    freeze,
    prune_cache_dir,
    snapshot_path,
    try_load,
)


# --- helpers -----------------------------------------------------------------


def _kv_cache_with(n_tokens: int, *, heads: int = 4, dim: int = 8) -> mlx_cache.KVCache:
    c = mlx_cache.KVCache()
    keys = mx.random.normal((1, heads, n_tokens, dim))
    values = mx.random.normal((1, heads, n_tokens, dim))
    c.update_and_fetch(keys, values)
    return c


def _quantized_kv_cache_with(
    n_tokens: int, *, heads: int = 4, dim: int = 64
) -> mlx_cache.QuantizedKVCache:
    c = mlx_cache.QuantizedKVCache(group_size=32, bits=8)
    keys = mx.random.normal((1, heads, n_tokens, dim))
    values = mx.random.normal((1, heads, n_tokens, dim))
    c.update_and_fetch(keys, values)
    return c


def _recurrent_cache_with(size: int = 2) -> RecurrentRollbackCache:
    c = RecurrentRollbackCache(size=size, conv_kernel_size=4)
    c.cache[0] = mx.random.normal((1, 3, 16))
    c.cache[1] = mx.random.normal((1, 2, 4, 8))
    return c


def _arrays_cache_with(size: int = 2) -> mlx_cache.ArraysCache:
    c = mlx_cache.ArraysCache(size=size)
    c.cache[0] = mx.random.normal((1, 5, 3))
    c.cache[1] = mx.random.normal((1, 7, 2))
    return c


def _states_equal(a, b) -> bool:
    """Deep equality for cache state objects (arrays, tuples of arrays, lists)."""
    if isinstance(a, mx.array) and isinstance(b, mx.array):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(mx.all(a == b).item())
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_states_equal(x, y) for x, y in zip(a, b, strict=True))
    if a is None and b is None:
        return True
    return False


def _canonical_config() -> dict:
    return {
        "model_id": "test/qwen-mini",
        "pq_bits": 4,
        "tq_bits": 16,
        "ctx_window": 32768,
    }


# --- fingerprint -------------------------------------------------------------


def test_fingerprint_is_deterministic():
    cfg = _canonical_config()
    tokens = list(range(1024))
    a = fingerprint(tokens, prefix_len=512, **cfg)
    b = fingerprint(tokens, prefix_len=512, **cfg)
    c = fingerprint(list(tokens), prefix_len=512, **cfg)  # fresh list
    assert a == b == c
    assert len(a) == 64  # sha256 hex


def test_fingerprint_differs_on_any_field():
    cfg = _canonical_config()
    tokens = list(range(1024))
    base = fingerprint(tokens, prefix_len=512, **cfg)

    assert base != fingerprint(tokens, prefix_len=256, **cfg)
    assert base != fingerprint(tokens + [], prefix_len=513, **(cfg | {}))
    assert base != fingerprint(tokens, prefix_len=512, **(cfg | {"model_id": "other"}))
    assert base != fingerprint(tokens, prefix_len=512, **(cfg | {"pq_bits": 3}))
    assert base != fingerprint(tokens, prefix_len=512, **(cfg | {"tq_bits": 4}))
    assert base != fingerprint(tokens, prefix_len=512, **(cfg | {"ctx_window": 65536}))

    # First token change invalidates.
    altered = [9999] + tokens[1:]
    assert base != fingerprint(altered, prefix_len=512, **cfg)

    # Change beyond prefix_len does NOT invalidate (crucial: the whole point
    # is a shared prefix hash).
    beyond = tokens[:512] + [777] + tokens[513:]
    assert base == fingerprint(beyond, prefix_len=512, **cfg)


def test_fingerprint_rejects_bad_inputs():
    cfg = _canonical_config()
    with pytest.raises(ValueError):
        fingerprint([1, 2, 3], prefix_len=0, **cfg)
    with pytest.raises(ValueError):
        fingerprint([1, 2, 3], prefix_len=-5, **cfg)
    with pytest.raises(ValueError):
        fingerprint([1, 2, 3], prefix_len=4, **cfg)  # too short
    with pytest.raises(ValueError):
        fingerprint([1, 2, 3], prefix_len=3, **(cfg | {"model_id": ""}))


# --- round trip --------------------------------------------------------------


def test_freeze_then_load_plain_kvcache(tmp_path: Path):
    cfg = _canonical_config()
    tokens = list(range(2048))
    src = [_kv_cache_with(n_tokens=128) for _ in range(3)]

    freeze(
        src,
        prompt_tokens=tokens,
        prefix_len=1024,
        base_dir=tmp_path,
        **cfg,
    )

    fresh = [mlx_cache.KVCache() for _ in range(3)]
    loaded = try_load(
        fresh,
        prompt_tokens=tokens,
        prefix_len=1024,
        base_dir=tmp_path,
        **cfg,
    )
    assert loaded is not None
    restored, meta = loaded
    assert meta["token_hash"] == fingerprint(tokens, prefix_len=1024, **cfg)
    assert meta["mio_frozen_kv_version"] == str(FROZEN_KV_VERSION)
    for live, orig in zip(restored, src, strict=True):
        assert live.offset == orig.offset
        assert _states_equal(live.state, orig.state)


def test_freeze_then_load_quantized_kvcache(tmp_path: Path):
    cfg = _canonical_config()
    tokens = list(range(1024))
    src = [_quantized_kv_cache_with(n_tokens=64) for _ in range(2)]
    freeze(
        src, prompt_tokens=tokens, prefix_len=512, base_dir=tmp_path, **cfg
    )
    fresh = [mlx_cache.QuantizedKVCache(group_size=32, bits=8) for _ in range(2)]
    loaded = try_load(
        fresh, prompt_tokens=tokens, prefix_len=512, base_dir=tmp_path, **cfg
    )
    assert loaded is not None
    restored, _ = loaded
    for live, orig in zip(restored, src, strict=True):
        assert live.offset == orig.offset
        assert live.bits == orig.bits
        assert live.group_size == orig.group_size
        assert _states_equal(live.state, orig.state)


def test_freeze_then_load_recurrent_rollback(tmp_path: Path):
    cfg = _canonical_config()
    tokens = list(range(2048))
    src = [_recurrent_cache_with(size=2) for _ in range(2)]
    freeze(
        src, prompt_tokens=tokens, prefix_len=256, base_dir=tmp_path, **cfg
    )
    fresh = [
        RecurrentRollbackCache(size=2, conv_kernel_size=4) for _ in range(2)
    ]
    loaded = try_load(
        fresh, prompt_tokens=tokens, prefix_len=256, base_dir=tmp_path, **cfg
    )
    assert loaded is not None
    restored, _ = loaded
    for live, orig in zip(restored, src, strict=True):
        assert _states_equal(live.state, orig.state)
        # conv_kernel_size comes from the template, not the snapshot.
        assert live.conv_kernel_size == 4


def test_freeze_then_load_mixed_caches(tmp_path: Path):
    """Hybrid model layout: some layers plain KV, some recurrent."""
    cfg = _canonical_config()
    tokens = list(range(1024))
    src = [
        _kv_cache_with(n_tokens=64),
        _recurrent_cache_with(size=2),
        _kv_cache_with(n_tokens=64),
        _arrays_cache_with(size=2),
    ]
    freeze(
        src, prompt_tokens=tokens, prefix_len=512, base_dir=tmp_path, **cfg
    )
    fresh = [
        mlx_cache.KVCache(),
        RecurrentRollbackCache(size=2, conv_kernel_size=4),
        mlx_cache.KVCache(),
        mlx_cache.ArraysCache(size=2),
    ]
    loaded = try_load(
        fresh, prompt_tokens=tokens, prefix_len=512, base_dir=tmp_path, **cfg
    )
    assert loaded is not None


# --- invalidation ------------------------------------------------------------


def test_try_load_returns_none_when_no_file(tmp_path: Path):
    cfg = _canonical_config()
    tokens = list(range(256))
    fresh = [mlx_cache.KVCache()]
    assert try_load(
        fresh, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **cfg
    ) is None


def test_try_load_returns_none_on_config_mismatch(tmp_path: Path):
    cfg = _canonical_config()
    tokens = list(range(256))
    src = [_kv_cache_with(n_tokens=32)]
    freeze(src, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **cfg)
    fresh = [mlx_cache.KVCache()]
    # Change pq_bits — different fingerprint, no file exists.
    other = cfg | {"pq_bits": 3}
    assert try_load(
        fresh, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **other
    ) is None


def test_try_load_returns_none_on_class_mismatch(tmp_path: Path):
    """Freezing KVCache but loading into QuantizedKVCache → refuse."""
    cfg = _canonical_config()
    tokens = list(range(256))
    src = [_kv_cache_with(n_tokens=32)]
    freeze(src, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **cfg)
    fresh = [mlx_cache.QuantizedKVCache(group_size=32, bits=8)]
    assert try_load(
        fresh, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **cfg
    ) is None


def test_try_load_returns_none_on_layer_count_mismatch(tmp_path: Path):
    cfg = _canonical_config()
    tokens = list(range(256))
    src = [_kv_cache_with(n_tokens=32), _kv_cache_with(n_tokens=32)]
    freeze(src, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **cfg)
    fresh = [mlx_cache.KVCache()]  # only one layer, snapshot has two
    assert try_load(
        fresh, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **cfg
    ) is None


def test_try_load_returns_none_on_corrupt_file(tmp_path: Path):
    cfg = _canonical_config()
    tokens = list(range(256))
    fp = fingerprint(tokens, prefix_len=128, **cfg)
    path = snapshot_path(fp, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not a safetensors file")
    fresh = [mlx_cache.KVCache()]
    assert try_load(
        fresh, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **cfg
    ) is None


def test_try_load_returns_none_on_prompt_too_short(tmp_path: Path):
    cfg = _canonical_config()
    tokens = list(range(64))
    fresh = [mlx_cache.KVCache()]
    # fingerprint() raises ValueError for prefix_len > len(tokens); try_load
    # must swallow that and return None rather than propagating.
    assert try_load(
        fresh, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **cfg
    ) is None


# --- candidate-length iterator ----------------------------------------------


def test_common_prefix_fingerprints_respects_candidates():
    cfg = _canonical_config()
    tokens = list(range(2048))
    results = common_prefix_fingerprints(
        tokens,
        candidate_lens=[4096, 2048, 2048, 1024, 0, -5, 512],
        **cfg,
    )
    lengths = [length for length, _ in results]
    # 4096 exceeds prompt length; 2048 dedup; 0/-5 filtered.
    assert lengths == [2048, 1024, 512]
    # Fingerprints unique across lengths.
    fps = [fp for _, fp in results]
    assert len(set(fps)) == 3


# --- pruning -----------------------------------------------------------------


def test_prune_cache_dir_evicts_by_age_and_bytes(tmp_path: Path):
    # Create three synthetic safetensors files, bump mtimes to order them.
    import os as _os
    files = []
    for i in range(3):
        p = tmp_path / f"file{i}.safetensors"
        p.write_bytes(b"x" * 1024)
        # force distinct mtimes (oldest first)
        _os.utime(p, (1000 + i, 1000 + i))
        files.append(p)

    # max_entries=2, large byte cap → evict one oldest.
    n = prune_cache_dir(tmp_path, max_entries=2, max_bytes=10 * 1024 * 1024)
    assert n == 1
    assert not files[0].exists()
    assert files[1].exists() and files[2].exists()

    # max_bytes=1 byte, large entry cap → evict all remaining.
    n2 = prune_cache_dir(tmp_path, max_entries=10, max_bytes=1)
    assert n2 == 2
    assert not files[1].exists() and not files[2].exists()


def test_prune_cache_dir_noop_on_missing(tmp_path: Path):
    missing = tmp_path / "nope"
    assert prune_cache_dir(missing) == 0


# --- snapshot_path & version tag --------------------------------------------


def test_snapshot_path_uses_base_dir(tmp_path: Path):
    p = snapshot_path("deadbeef", tmp_path)
    assert p.parent == tmp_path
    assert p.name == "deadbeef.safetensors"


def test_engine_frozen_kv_env_gating(monkeypatch):
    """Engine helpers respond to env vars without requiring a model."""
    from mio.config import TierConfig
    from mio.engine import MioEngine

    tc = TierConfig(
        name="t", target_model="p", draft_model="d",
        context_window=32768, max_output_tokens=1024,
    )
    eng = MioEngine(tier_config=tc)

    monkeypatch.delenv("MIO_FROZEN_KV", raising=False)
    assert eng._frozen_kv_enabled() is False
    monkeypatch.setenv("MIO_FROZEN_KV", "1")
    assert eng._frozen_kv_enabled() is True
    monkeypatch.setenv("MIO_FROZEN_KV", "yes")
    assert eng._frozen_kv_enabled() is True
    monkeypatch.setenv("MIO_FROZEN_KV", "0")
    assert eng._frozen_kv_enabled() is False


def test_engine_frozen_kv_prefix_override(monkeypatch):
    from mio.config import TierConfig
    from mio.engine import MioEngine
    tc = TierConfig(
        name="t", target_model="p", draft_model="d",
        context_window=32768, max_output_tokens=1024,
    )
    eng = MioEngine(tier_config=tc)
    monkeypatch.delenv("MIO_FROZEN_KV_PREFIX", raising=False)
    assert eng._frozen_kv_prefix_len() == 4096
    monkeypatch.setenv("MIO_FROZEN_KV_PREFIX", "12000")
    assert eng._frozen_kv_prefix_len() == 12000
    monkeypatch.setenv("MIO_FROZEN_KV_PREFIX", "garbage")
    assert eng._frozen_kv_prefix_len() == 4096  # ValueError fallback


def test_engine_frozen_kv_config_includes_all_fields():
    from mio.config import TierConfig
    from mio.engine import MioEngine
    tc = TierConfig(
        name="t", target_model="/path/to/m", draft_model="d",
        context_window=65536, max_output_tokens=2048, pq_bits=3, tq_bits=16,
    )
    eng = MioEngine(tier_config=tc)
    cfg = eng._frozen_kv_config()
    assert cfg == {
        "model_id": "t|/path/to/m",
        "pq_bits": 3,
        "tq_bits": 16,
        "ctx_window": 65536,
    }


def test_engine_frozen_kv_roundtrip_with_synthetic_cache(tmp_path: Path, monkeypatch):
    """End-to-end: engine config → frozen_kv freeze → frozen_kv load succeeds."""
    from mio.config import TierConfig
    from mio.engine import MioEngine
    tc = TierConfig(
        name="t", target_model="p", draft_model="d",
        context_window=32768, max_output_tokens=1024,
    )
    eng = MioEngine(tier_config=tc)
    monkeypatch.setenv("MIO_FROZEN_KV_DIR", str(tmp_path))
    assert eng._frozen_kv_base_dir() == tmp_path

    tokens = list(range(1024))
    caches = [_kv_cache_with(n_tokens=64) for _ in range(2)]
    cfg = eng._frozen_kv_config()
    freeze(
        caches, prompt_tokens=tokens, prefix_len=512,
        base_dir=eng._frozen_kv_base_dir(), **cfg,
    )

    fresh = [mlx_cache.KVCache() for _ in range(2)]
    loaded = try_load(
        fresh, prompt_tokens=tokens, prefix_len=512,
        base_dir=eng._frozen_kv_base_dir(), **cfg,
    )
    assert loaded is not None
    restored, meta = loaded
    # Cross-check engine-derived model_id is in metadata.
    assert meta["model_id"] == "t|p"
    for live, orig in zip(restored, caches, strict=True):
        assert live.offset == orig.offset


def test_bundle_freeze_then_load_roundtrip(tmp_path: Path):
    """Full warm_state envelope round-trip: target + draft + hidden."""
    from mio.dflash.model import ContextOnlyDraftKVCache
    from mio.frozen_kv import bundle_freeze, bundle_try_load

    cfg = _canonical_config()
    tokens = list(range(2048))

    # Synthesize a warm_state as the runtime would produce it.
    target_cache = [_kv_cache_with(n_tokens=128) for _ in range(3)]
    draft_cache = [
        ContextOnlyDraftKVCache(sink_size=64, window_size=256) for _ in range(2)
    ]
    draft_cache[0].keys = mx.random.normal((1, 2, 32, 8))
    draft_cache[0].values = mx.random.normal((1, 2, 32, 8))
    draft_cache[0].offset = 32
    # Leave draft_cache[1] empty (keys/values = None, offset = 0).
    target_hidden = mx.random.normal((1, 128, 16))

    path = bundle_freeze(
        target_cache=target_cache,
        draft_cache=draft_cache,
        target_hidden=target_hidden,
        offset=128,
        prompt_tokens=tokens,
        prefix_len=1024,
        base_dir=tmp_path,
        **cfg,
    )
    assert path.exists()

    # Fresh templates, simulate what engine.py would build.
    fresh_target = [mlx_cache.KVCache() for _ in range(3)]
    fresh_draft = [
        ContextOnlyDraftKVCache(sink_size=1, window_size=1) for _ in range(2)
    ]

    loaded = bundle_try_load(
        target_template=fresh_target,
        draft_template=fresh_draft,
        prompt_tokens=tokens,
        prefix_len=1024,
        base_dir=tmp_path,
        **cfg,
    )
    assert loaded is not None
    assert loaded["target_cache"] is fresh_target  # mutated in-place
    assert loaded["draft_cache"] is fresh_draft
    assert loaded["offset"] == 128
    assert loaded["target_hidden"] is not None
    assert loaded["target_hidden"].shape == target_hidden.shape

    # Target caches match.
    for live, orig in zip(fresh_target, target_cache, strict=True):
        assert live.offset == orig.offset
        assert _states_equal(live.state, orig.state)

    # Draft state restored correctly per layer.
    assert fresh_draft[0].sink_size == 64
    assert fresh_draft[0].window_size == 256
    assert fresh_draft[0].offset == 32
    assert fresh_draft[0].keys is not None
    assert bool(mx.all(fresh_draft[0].keys == draft_cache[0].keys).item())
    assert fresh_draft[0].values is not None
    assert bool(mx.all(fresh_draft[0].values == draft_cache[0].values).item())
    # Empty draft layer restored as empty (not leaking template defaults).
    assert fresh_draft[1].keys is None
    assert fresh_draft[1].values is None
    assert fresh_draft[1].offset == 0


def test_bundle_try_load_returns_none_on_draft_count_mismatch(tmp_path: Path):
    from mio.dflash.model import ContextOnlyDraftKVCache
    from mio.frozen_kv import bundle_freeze, bundle_try_load

    cfg = _canonical_config()
    tokens = list(range(512))
    bundle_freeze(
        target_cache=[_kv_cache_with(n_tokens=32)],
        draft_cache=[
            ContextOnlyDraftKVCache(sink_size=8, window_size=16) for _ in range(2)
        ],
        target_hidden=None,
        offset=32,
        prompt_tokens=tokens,
        prefix_len=128,
        base_dir=tmp_path,
        **cfg,
    )
    # Try to load with only ONE draft layer → mismatch.
    loaded = bundle_try_load(
        target_template=[mlx_cache.KVCache()],
        draft_template=[ContextOnlyDraftKVCache(sink_size=1, window_size=1)],
        prompt_tokens=tokens,
        prefix_len=128,
        base_dir=tmp_path,
        **cfg,
    )
    assert loaded is None


def test_bundle_try_load_returns_none_on_no_file(tmp_path: Path):
    from mio.dflash.model import ContextOnlyDraftKVCache
    from mio.frozen_kv import bundle_try_load

    cfg = _canonical_config()
    assert bundle_try_load(
        target_template=[mlx_cache.KVCache()],
        draft_template=[ContextOnlyDraftKVCache(sink_size=1, window_size=1)],
        prompt_tokens=list(range(256)),
        prefix_len=128,
        base_dir=tmp_path,
        **cfg,
    ) is None


def test_version_bump_invalidates(tmp_path: Path, monkeypatch):
    """If FROZEN_KV_VERSION is bumped, existing snapshots stop loading."""
    cfg = _canonical_config()
    tokens = list(range(256))
    src = [_kv_cache_with(n_tokens=32)]
    freeze(src, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **cfg)

    # Monkey-patch the version constant the module reads at call time.
    import mio.frozen_kv as fkv
    monkeypatch.setattr(fkv, "FROZEN_KV_VERSION", fkv.FROZEN_KV_VERSION + 1)

    fresh = [mlx_cache.KVCache()]
    # fingerprint now includes new version → snapshot on disk keyed by old
    # fingerprint, and version string won't match anyway.
    assert fkv.try_load(
        fresh, prompt_tokens=tokens, prefix_len=128, base_dir=tmp_path, **cfg
    ) is None
