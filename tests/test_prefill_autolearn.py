"""Tests for mio.theories.prefill_autolearn.

Deterministic unit tests, no model load. Exercises:
  - Prototype store: add, persist, reload, evict.
  - Nearest-neighbor cosine search.
  - Config-filtered retrieval.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mio.theories.prefill_autolearn.prototype_store import (
    PrototypeStore, _prompt_id,
)


def _make_unit_embedding(seed: int, d: int = 64) -> mx.array:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return mx.array(v)


def test_prompt_id_stable():
    tokens = [1, 2, 3, 4, 5]
    assert _prompt_id(tokens) == _prompt_id(tokens)
    assert _prompt_id(tokens) != _prompt_id([1, 2, 3, 4, 6])


def test_add_and_reload(tmp_path: Path):
    store = PrototypeStore(base_dir=tmp_path, max_entries=10)
    assert len(store) == 0

    emb = _make_unit_embedding(seed=1)
    proto = store.add(
        tokens=[1, 2, 3, 4, 5], embedding=emb,
        frozen_kv_path="/dev/null", model_id="test",
        pq_bits=4, tq_bits=16, ctx_window=8192,
    )
    assert len(store) == 1
    assert proto.model_id == "test"

    # Reload from disk
    store2 = PrototypeStore(base_dir=tmp_path)
    assert len(store2) == 1
    assert store2.all()[0].id == proto.id


def test_add_duplicate_bumps_hit_count(tmp_path: Path):
    store = PrototypeStore(base_dir=tmp_path)
    emb = _make_unit_embedding(seed=2)
    p1 = store.add(
        tokens=[10, 20, 30], embedding=emb,
        frozen_kv_path="/x", model_id="m", pq_bits=4, tq_bits=16, ctx_window=4096,
    )
    p2 = store.add(
        tokens=[10, 20, 30], embedding=emb,
        frozen_kv_path="/x", model_id="m", pq_bits=4, tq_bits=16, ctx_window=4096,
    )
    assert p1.id == p2.id
    assert len(store) == 1
    assert p2.hit_count == 1


def test_lru_evict(tmp_path: Path):
    store = PrototypeStore(base_dir=tmp_path, max_entries=3)
    for i in range(5):
        emb = _make_unit_embedding(seed=i)
        store.add(
            tokens=[i, i, i, i, i, i, i, i], embedding=emb,
            frozen_kv_path=f"/x{i}", model_id="m",
            pq_bits=4, tq_bits=16, ctx_window=4096,
        )
    # Evicted down to 3.
    assert len(store) == 3


def test_nearest_returns_most_similar(tmp_path: Path):
    store = PrototypeStore(base_dir=tmp_path)
    # Build 3 prototypes with known embeddings.
    emb_a = _make_unit_embedding(seed=100)
    emb_b = _make_unit_embedding(seed=200)
    emb_c = _make_unit_embedding(seed=300)
    store.add(
        tokens=[1] * 10, embedding=emb_a,
        frozen_kv_path="/a", model_id="m", pq_bits=4, tq_bits=16, ctx_window=4096,
    )
    store.add(
        tokens=[2] * 10, embedding=emb_b,
        frozen_kv_path="/b", model_id="m", pq_bits=4, tq_bits=16, ctx_window=4096,
    )
    store.add(
        tokens=[3] * 10, embedding=emb_c,
        frozen_kv_path="/c", model_id="m", pq_bits=4, tq_bits=16, ctx_window=4096,
    )
    # Query with a near-copy of emb_a — add small noise then renormalize.
    q = emb_a + 0.01 * _make_unit_embedding(seed=999)
    q = q / mx.sqrt(mx.sum(q * q) + 1e-9)
    hits = store.nearest(q, k=3, min_similarity=0.0, model_id="m",
                         pq_bits=4, tq_bits=16)
    assert len(hits) >= 1
    # emb_a should be the top match.
    assert hits[0][0].frozen_kv_path == "/a"
    assert hits[0][1] > 0.9


def test_nearest_filters_by_config(tmp_path: Path):
    store = PrototypeStore(base_dir=tmp_path)
    emb = _make_unit_embedding(seed=5)
    # Same embedding, different configs.
    store.add(
        tokens=[1] * 10, embedding=emb,
        frozen_kv_path="/x", model_id="m1", pq_bits=4, tq_bits=16, ctx_window=4096,
    )
    store.add(
        tokens=[2] * 10, embedding=emb,
        frozen_kv_path="/y", model_id="m2", pq_bits=4, tq_bits=16, ctx_window=4096,
    )
    hits_m1 = store.nearest(emb, model_id="m1", min_similarity=0.0)
    hits_m2 = store.nearest(emb, model_id="m2", min_similarity=0.0)
    assert len(hits_m1) == 1
    assert len(hits_m2) == 1
    assert hits_m1[0][0].frozen_kv_path == "/x"
    assert hits_m2[0][0].frozen_kv_path == "/y"


def test_nearest_respects_min_similarity(tmp_path: Path):
    store = PrototypeStore(base_dir=tmp_path)
    emb_a = _make_unit_embedding(seed=400)
    store.add(
        tokens=[1] * 10, embedding=emb_a,
        frozen_kv_path="/a", model_id="m", pq_bits=4, tq_bits=16, ctx_window=4096,
    )
    # Orthogonal-ish query.
    emb_orth = _make_unit_embedding(seed=401)
    # Filter out low-similarity matches.
    hits = store.nearest(emb_orth, min_similarity=0.9, model_id="m",
                         pq_bits=4, tq_bits=16)
    assert hits == []


def test_empty_store(tmp_path: Path):
    store = PrototypeStore(base_dir=tmp_path)
    emb = _make_unit_embedding(seed=0)
    hits = store.nearest(emb, min_similarity=0.0)
    assert hits == []
