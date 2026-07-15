"""Tests for Batched Multi-Path DFlash (BMP) primitives."""

from __future__ import annotations

import math

import mlx.core as mx
from mlx_lm.models import cache as cache_mod

from mio.config import TierConfig
from mio.dflash.bmp import (
    build_bmp_batch,
    expand_cache_batch,
    extract_top_k,
    filter_cache_batch,
    per_row_acceptance,
)
from mio.dflash.recurrent_rollback_cache import RecurrentRollbackCache
from mio.engine import MioEngine


def test_extract_top_k_shapes():
    V = 32
    L = 4
    mx.random.seed(0)
    logits = mx.random.normal((1, L, V))
    tokens, logps = extract_top_k(logits, k=5)
    assert len(tokens) == L and len(tokens[0]) == 5
    assert len(logps) == L and len(logps[0]) == 5
    # Logprobs should be nonincreasing within each row
    for row in logps:
        for a, b in zip(row, row[1:]):
            assert a >= b - 1e-6


def test_nonstream_engine_keeps_bmp_with_text_stop_and_adjusts_usage(monkeypatch):
    tier = TierConfig(
        name="test",
        target_model="unused",
        draft_model="unused",
        context_window=4096,
        max_output_tokens=64,
        bmp_paths=3,
        pq_bits=16,
    )
    engine = MioEngine(tier)
    engine._loaded = True
    engine._target_model = object()
    engine._draft_model = object()
    engine._tokenizer = type(
        "Tokenizer",
        (),
        {
            "decode": staticmethod(
                lambda _ids, skip_special_tokens=True: "keep STOP discard"
            ),
            "encode": staticmethod(
                lambda text, add_special_tokens=False: text.split()
            ),
        },
    )()
    monkeypatch.setattr(engine, "_apply_chat_template", lambda _messages, tools=None: [1, 2])
    monkeypatch.setattr(engine, "_eos_token_ids", lambda: [0])
    monkeypatch.setattr(engine, "_make_sampler", lambda *_args: object())
    calls = []

    def fake_bmp(**kwargs):
        calls.append(kwargs)
        return {
            "generated_token_ids": [10, 11, 12],
            "elapsed_us": 300_000,
            "prefill_us": 100_000,
            "generation_tokens": 3,
            "prompt_token_count": 2,
        }

    monkeypatch.setattr("mio.dflash.bmp_runtime.generate_bmp_dflash_once", fake_bmp)

    text, metrics = engine.generate(
        [{"role": "user", "content": "prompt"}],
        max_tokens=8,
        stop=["STOP"],
    )

    assert len(calls) == 1
    assert calls[0]["num_paths"] == 3
    assert text == "keep "
    assert metrics.prompt_tokens == 2
    assert metrics.completion_tokens == 1
    assert metrics.total_tokens == 3


def test_extract_top_k_logprobs_sum_to_leq_one():
    """Softmax properties: sum of exp(logp) over top-K ≤ 1."""
    V = 16
    L = 3
    mx.random.seed(1)
    logits = mx.random.normal((1, L, V))
    _, logps = extract_top_k(logits, k=V)  # all of V
    for row in logps:
        total = sum(math.exp(x) for x in row)
        assert abs(total - 1.0) < 1e-3  # effectively 1.0


def test_build_bmp_batch_K1_returns_argmax_path():
    top_tokens = [[5, 7], [10, 20], [30, 40]]
    top_logps = [
        [math.log(0.9), math.log(0.1)],
        [math.log(0.8), math.log(0.2)],
        [math.log(0.7), math.log(0.3)],
    ]
    batch, paths = build_bmp_batch(
        bonus_token=99, top_k_tokens=top_tokens, top_k_logprobs=top_logps,
        num_paths=1, block_len=4,
    )
    # K=1, row length = block_len = 4
    assert batch.shape == (1, 4)
    # [bonus, argmax_pos1, argmax_pos2, argmax_pos3]
    assert batch.tolist() == [[99, 5, 10, 30]]
    assert paths == [[5, 10, 30]]


def test_build_bmp_batch_K2_covers_depth1_alternatives():
    top_tokens = [[5, 7], [10, 20], [30, 40]]
    top_logps = [
        [math.log(0.55), math.log(0.45)],  # near-tie at depth 1
        [math.log(0.95), math.log(0.05)],
        [math.log(0.95), math.log(0.05)],
    ]
    batch, paths = build_bmp_batch(
        bonus_token=99, top_k_tokens=top_tokens, top_k_logprobs=top_logps,
        num_paths=2, block_len=4,
    )
    assert batch.shape[0] == 2
    # Both paths should share the bonus and same depth-2/3 argmax tail,
    # differ at depth 1 (path 0 uses rank-1 = 5, path 1 uses rank-2 = 7).
    depth1_tokens = {batch[0, 1].item(), batch[1, 1].item()}
    assert depth1_tokens == {5, 7}


def test_build_bmp_batch_padding_when_block_longer_than_draft():
    """If block_len-1 > L_draft, pad tail with marginal argmax (falling back to last)."""
    top_tokens = [[1, 2], [3, 4]]
    top_logps = [[math.log(0.8), math.log(0.2)], [math.log(0.7), math.log(0.3)]]
    batch, _paths = build_bmp_batch(
        bonus_token=99, top_k_tokens=top_tokens, top_k_logprobs=top_logps,
        num_paths=1, block_len=5,  # 1 bonus + 4 continuation tokens
    )
    assert batch.shape == (1, 5)
    # [99, 1, 3, padded, padded] — last two padded with argmax of last available pos (3)
    row = batch[0].tolist()
    assert row[0] == 99
    assert row[1] == 1
    assert row[2] == 3


def test_build_bmp_batch_K_clipped_if_few_alternatives():
    """With tiny top-K and low budget, K should clip to available paths."""
    # Only one token at each depth (top-K=1) → one possible path
    top_tokens = [[5], [10]]
    top_logps = [[math.log(1.0)], [math.log(1.0)]]
    batch, _paths = build_bmp_batch(
        bonus_token=99, top_k_tokens=top_tokens, top_k_logprobs=top_logps,
        num_paths=4, block_len=3,
    )
    # Can generate at most 1 distinct path
    assert batch.shape[0] == 1


def test_per_row_acceptance_all_matched():
    verify = mx.array([[10, 20, 30], [100, 200, 300]], dtype=mx.uint32)
    posterior = mx.array([[20, 30, 99], [200, 300, 400]], dtype=mx.uint32)
    accept = per_row_acceptance(verify, posterior)
    # Row 0: verify[1]=20 vs posterior[0]=20 ✓; verify[2]=30 vs posterior[1]=30 ✓ → 2
    # Row 1: verify[1]=200 vs posterior[0]=200 ✓; verify[2]=300 vs posterior[1]=300 ✓ → 2
    assert accept == [2, 2]


def test_per_row_acceptance_early_mismatch():
    verify = mx.array([[10, 999, 30]], dtype=mx.uint32)
    posterior = mx.array([[20, 30, 40]], dtype=mx.uint32)
    accept = per_row_acceptance(verify, posterior)
    # verify[1]=999 vs posterior[0]=20 → mismatch at first check → 0
    assert accept == [0]


def test_per_row_acceptance_partial_per_row():
    verify = mx.array([[10, 20, 999, 40], [10, 50, 60, 70]], dtype=mx.uint32)
    posterior = mx.array([[20, 30, 40, 50], [50, 60, 70, 80]], dtype=mx.uint32)
    accept = per_row_acceptance(verify, posterior)
    # Row 0: v[1]=20 ✓ v[0]=20; v[2]=999 ✗ p[1]=30 → 1
    # Row 1: v[1]=50 ✓ p[0]=50; v[2]=60 ✓ p[1]=60; v[3]=70 ✓ p[2]=70 → 3
    assert accept == [1, 3]


def test_kvcache_expand_and_filter_roundtrip():
    kv = cache_mod.KVCache()
    k = mx.random.normal((1, 2, 4, 16))
    v = mx.random.normal((1, 2, 4, 16))
    kv.update_and_fetch(k, v)
    assert kv.keys.shape[0] == 1

    expand_cache_batch([kv], K=3)
    assert kv.keys.shape[0] == 3

    # Row 1 should equal row 0 (broadcast)
    assert mx.array_equal(kv.keys[0], kv.keys[1])
    assert mx.array_equal(kv.values[0], kv.values[2])

    filter_cache_batch([kv], winner_idx=2)
    assert kv.keys.shape[0] == 1


def test_arrays_cache_expand_and_filter():
    ac = cache_mod.ArraysCache(size=2)
    arr = mx.random.normal((1, 3, 8))
    ac[0] = arr
    ac[1] = arr
    expand_cache_batch([ac], K=2)
    assert ac.cache[0].shape[0] == 2
    filter_cache_batch([ac], winner_idx=0)
    assert ac.cache[0].shape[0] == 1


def test_recurrent_rollback_cache_expand_and_filter():
    rc = RecurrentRollbackCache(size=1, conv_kernel_size=4)
    rc.cache[0] = mx.random.normal((1, 8, 16))
    expand_cache_batch([rc], K=4)
    assert rc.cache[0].shape[0] == 4
    filter_cache_batch([rc], winner_idx=2)
    assert rc.cache[0].shape[0] == 1
