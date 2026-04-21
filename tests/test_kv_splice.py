"""Unit tests for mio.kv_splice store + detector.

Deterministic, no model load. Tests the storage contract, chunk hashing,
chunk detection algorithm, LRU eviction, and boundary cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mio.kv_splice.store import ChunkStore, chunk_hash, KV_SPLICE_VERSION
from mio.kv_splice.detect import detect_chunks, SpliceSite


# ---- helpers ----

def _fake_kv(n_kv=2, L=16, d_head=64):
    return {
        "k_base": mx.random.normal((n_kv, L, d_head)).astype(mx.float16),
        "v": mx.random.normal((n_kv, L, d_head)).astype(mx.float16),
    }


# ---- chunk_hash ----

def test_chunk_hash_stable():
    a = [1, 2, 3, 4, 5]
    assert chunk_hash(a) == chunk_hash(a)
    assert len(chunk_hash(a)) == 16


def test_chunk_hash_distinguishes():
    assert chunk_hash([1, 2, 3]) != chunk_hash([1, 2, 4])
    assert chunk_hash([1, 2, 3]) != chunk_hash([3, 2, 1])


# ---- store: save / load ----

def test_store_save_load_roundtrip(tmp_path: Path):
    store = ChunkStore(base_dir=tmp_path)
    tokens = [100, 101, 102, 103] * 4  # 16 tokens
    kv = {3: _fake_kv(L=len(tokens)), 7: _fake_kv(L=len(tokens))}
    entry = store.save(
        tokens=tokens, model_id="test",
        layers=[3, 7], n_kv_heads=2, d_head=64,
        kv_per_layer=kv,
    )
    assert entry.chunk_id == chunk_hash(tokens)
    assert len(store) == 1

    loaded = store.load_kv(entry.chunk_id)
    assert loaded is not None
    assert 3 in loaded and 7 in loaded
    # K_base from disk should match saved.
    assert bool(mx.all(loaded[3]["k_base"] == kv[3]["k_base"]).item())
    assert bool(mx.all(loaded[3]["v"] == kv[3]["v"]).item())


def test_store_reload_after_restart(tmp_path: Path):
    store = ChunkStore(base_dir=tmp_path)
    tokens = list(range(50, 100))
    kv = {3: _fake_kv(L=len(tokens))}
    store.save(tokens=tokens, model_id="m", layers=[3],
               n_kv_heads=2, d_head=64, kv_per_layer=kv)

    # Fresh store reads the same index.
    store2 = ChunkStore(base_dir=tmp_path)
    assert len(store2) == 1
    entry = store2.get_by_tokens(tokens)
    assert entry is not None


def test_store_load_missing_returns_none(tmp_path: Path):
    store = ChunkStore(base_dir=tmp_path)
    assert store.load_kv("deadbeef") is None


def test_store_eviction(tmp_path: Path):
    store = ChunkStore(base_dir=tmp_path, max_entries=3)
    for i in range(5):
        tokens = list(range(i * 100, i * 100 + 20))
        kv = {3: _fake_kv(L=len(tokens))}
        store.save(
            tokens=tokens, model_id="m", layers=[3],
            n_kv_heads=2, d_head=64, kv_per_layer=kv,
        )
    assert len(store) == 3


# ---- detect_chunks ----

def test_detect_no_chunks(tmp_path: Path):
    store = ChunkStore(base_dir=tmp_path)
    assert detect_chunks([1, 2, 3, 4], store) == []


def test_detect_single_match(tmp_path: Path):
    store = ChunkStore(base_dir=tmp_path)
    # Store a 16-token chunk (>= min_chunk_len default).
    chunk = list(range(100, 116))
    store.save(tokens=chunk, model_id="m", layers=[3],
               n_kv_heads=2, d_head=64, kv_per_layer={3: _fake_kv(L=16)})
    # New prompt embeds the chunk starting at offset 5.
    prompt = [999] * 5 + chunk + [777] * 10
    sites = detect_chunks(prompt, store, min_chunk_len=16)
    assert len(sites) == 1
    assert sites[0].start == 5
    assert sites[0].end == 5 + 16
    assert sites[0].chunk_len == 16


def test_detect_multiple_non_overlapping(tmp_path: Path):
    store = ChunkStore(base_dir=tmp_path)
    c1 = list(range(100, 120))
    c2 = list(range(200, 220))
    store.save(tokens=c1, model_id="m", layers=[3], n_kv_heads=2, d_head=64,
               kv_per_layer={3: _fake_kv(L=20)})
    store.save(tokens=c2, model_id="m", layers=[3], n_kv_heads=2, d_head=64,
               kv_per_layer={3: _fake_kv(L=20)})
    prompt = [999] * 10 + c1 + [777] * 5 + c2 + [888] * 3
    sites = detect_chunks(prompt, store, min_chunk_len=16)
    assert len(sites) == 2
    assert sites[0].start == 10
    assert sites[1].start == 10 + 20 + 5


def test_detect_greedy_picks_longest(tmp_path: Path):
    """When both a short and a long chunk start at the same position,
    longest-match wins."""
    store = ChunkStore(base_dir=tmp_path)
    short_c = list(range(500, 520))          # 20 tokens
    long_c  = list(range(500, 550))          # 50 tokens
    store.save(tokens=short_c, model_id="m", layers=[3], n_kv_heads=2, d_head=64,
               kv_per_layer={3: _fake_kv(L=20)})
    store.save(tokens=long_c, model_id="m", layers=[3], n_kv_heads=2, d_head=64,
               kv_per_layer={3: _fake_kv(L=50)})
    prompt = [0] * 3 + long_c + [99] * 5
    sites = detect_chunks(prompt, store, min_chunk_len=16)
    assert len(sites) == 1
    assert sites[0].chunk_len == 50


def test_detect_filters_by_model_id(tmp_path: Path):
    store = ChunkStore(base_dir=tmp_path)
    chunk = list(range(100, 130))
    store.save(tokens=chunk, model_id="m1", layers=[3], n_kv_heads=2, d_head=64,
               kv_per_layer={3: _fake_kv(L=30)})
    # Looking for model m2's chunks — should find nothing.
    prompt = chunk + [1, 2, 3]
    sites = detect_chunks(prompt, store, model_id="m2", min_chunk_len=16)
    assert sites == []
    # But looking for m1's → found.
    sites = detect_chunks(prompt, store, model_id="m1", min_chunk_len=16)
    assert len(sites) == 1


def test_detect_respects_min_chunk_len(tmp_path: Path):
    store = ChunkStore(base_dir=tmp_path)
    # Short chunk — but store raw (min_chunk_len gate is on detect, not save).
    chunk = list(range(100, 110))  # 10 tokens
    store.save(tokens=chunk, model_id="m", layers=[3], n_kv_heads=2, d_head=64,
               kv_per_layer={3: _fake_kv(L=10)})
    prompt = [0] * 5 + chunk + [0] * 5
    # With min_chunk_len=16, the 10-token chunk is filtered out.
    sites = detect_chunks(prompt, store, min_chunk_len=16)
    assert sites == []
    # With min_chunk_len=5, it passes.
    sites = detect_chunks(prompt, store, min_chunk_len=5)
    assert len(sites) == 1


def test_detect_handles_partial_overlap_at_boundary(tmp_path: Path):
    """Chunk's first token matches but full sequence doesn't — don't splice."""
    store = ChunkStore(base_dir=tmp_path)
    chunk = list(range(100, 120))
    store.save(tokens=chunk, model_id="m", layers=[3], n_kv_heads=2, d_head=64,
               kv_per_layer={3: _fake_kv(L=20)})
    # Prompt starts with chunk's first token but diverges at token 3.
    prompt = [100, 101, 102, 999, 999] + [200] * 5
    sites = detect_chunks(prompt, store, min_chunk_len=16)
    assert sites == []
