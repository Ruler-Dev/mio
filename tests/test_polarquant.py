"""PolarQuant KV-cache compression tests.

Covers:
  1. Hadamard matrix generation and orthogonality.
  2. PolarQuantKVCache inherits from QuantizedKVCache.
  3. 4-bit mode (no rotation — zero overhead).
  4. 3-bit mode (with Hadamard rotation — quality boost).
  5. SDPA correctness at 3-bit with rotation.
  6. Cache ops (trim, expansion).
"""

from __future__ import annotations

import mlx.core as mx
import pytest


HEAD_DIM = 128
N_KV_HEADS = 2
N_Q_HEADS = 8
T_NEW = 32


def _random_kv(T: int, D: int = HEAD_DIM, seed: int = 0):
    mx.random.seed(seed)
    k = mx.random.normal((1, N_KV_HEADS, T, D)).astype(mx.float32)
    v = mx.random.normal((1, N_KV_HEADS, T, D)).astype(mx.float32)
    mx.eval(k, v)
    return k, v


# --- Hadamard ---

def test_hadamard_orthogonality():
    from mio.polarquant.hadamard import hadamard_matrix

    H = hadamard_matrix(HEAD_DIM)
    mx.eval(H)
    err = mx.max(mx.abs(H @ H.T - mx.eye(HEAD_DIM)))
    assert float(err.item()) < 1e-5


def test_hadamard_power_of_2_only():
    from mio.polarquant.hadamard import hadamard_matrix
    with pytest.raises(ValueError):
        hadamard_matrix(100)


def test_hadamard_self_inverse():
    from mio.polarquant.hadamard import hadamard_matrix
    H = hadamard_matrix(HEAD_DIM)
    mx.random.seed(42)
    x = mx.random.normal((4, HEAD_DIM))
    mx.eval(x)
    err = mx.max(mx.abs((x @ H) @ H - x))
    assert float(err.item()) < 1e-5


# --- Cache subclass ---

def test_inherits_from_quantized_kv_cache():
    from mlx_lm.models.cache import QuantizedKVCache
    from mio.polarquant.cache import PolarQuantKVCache

    cache = PolarQuantKVCache(head_dim=HEAD_DIM, bits=4, group_size=64)
    assert isinstance(cache, QuantizedKVCache)
    assert cache.bits == 4
    assert cache.group_size == 64


# --- 4-bit mode (no rotation, zero overhead) ---

def test_4bit_no_rotation():
    from mio.polarquant.cache import PolarQuantKVCache

    cache = PolarQuantKVCache(head_dim=HEAD_DIM, bits=4, group_size=64)
    assert not cache._use_rotation
    assert cache.H is None


def test_4bit_update_and_fetch():
    from mio.polarquant.cache import PolarQuantKVCache

    k, v = _random_kv(T_NEW)
    cache = PolarQuantKVCache(head_dim=HEAD_DIM, bits=4, group_size=64)
    q_keys, q_values = cache.update_and_fetch(k, v)
    assert cache.offset == T_NEW
    assert isinstance(q_keys, tuple) and len(q_keys) == 3
    mx.eval(*q_keys, *q_values)


def test_4bit_reconstruction_quality():
    """4-bit without rotation should still give good quality (~0.998 cosine)."""
    from mio.polarquant.cache import PolarQuantKVCache

    k, v = _random_kv(64, seed=42)
    cache = PolarQuantKVCache(head_dim=HEAD_DIM, bits=4, group_size=64)
    q_keys, _ = cache.update_and_fetch(k, v)

    k_deq = mx.dequantize(*q_keys, group_size=64, bits=4)
    k_flat = k.reshape(-1)
    r_flat = k_deq.reshape(-1)
    cos_sim = mx.sum(k_flat * r_flat) / (mx.sqrt(mx.sum(k_flat ** 2)) * mx.sqrt(mx.sum(r_flat ** 2)))
    assert float(cos_sim.item()) > 0.98


# --- 3-bit mode (with Hadamard rotation) ---

def test_3bit_uses_rotation():
    from mio.polarquant.cache import PolarQuantKVCache

    cache = PolarQuantKVCache(head_dim=HEAD_DIM, bits=3, group_size=64)
    assert cache._use_rotation
    assert cache.H is not None


def test_3bit_update_and_fetch():
    from mio.polarquant.cache import PolarQuantKVCache

    k, v = _random_kv(T_NEW)
    cache = PolarQuantKVCache(head_dim=HEAD_DIM, bits=3, group_size=64)
    q_keys, q_values = cache.update_and_fetch(k, v)
    assert cache.offset == T_NEW
    mx.eval(*q_keys, *q_values)


def test_3bit_rotation_improves_quality():
    """Hadamard rotation should improve quality at 3-bit vs raw quantization."""
    from mio.polarquant.cache import PolarQuantKVCache
    from mlx_lm.models.cache import QuantizedKVCache

    k, v = _random_kv(64, seed=7)

    # With rotation (PolarQuant)
    pq_cache = PolarQuantKVCache(head_dim=HEAD_DIM, bits=3, group_size=64)
    pq_keys, _ = pq_cache.update_and_fetch(k, v)
    pq_deq = mx.dequantize(*pq_keys, group_size=64, bits=3) @ pq_cache.H
    pq_err = float(mx.mean((pq_deq - k) ** 2).item())

    # Without rotation (plain QuantizedKVCache)
    raw_cache = QuantizedKVCache(group_size=64, bits=3)
    raw_keys, _ = raw_cache.update_and_fetch(k, v)
    raw_deq = mx.dequantize(*raw_keys, group_size=64, bits=3)
    raw_err = float(mx.mean((raw_deq - k) ** 2).item())

    # Rotation should reduce error
    assert pq_err < raw_err, f"PQ err {pq_err:.6f} >= raw err {raw_err:.6f}"


# --- SDPA correctness at 3-bit ---

def _exact_sdpa(q, k, v, scale):
    scores = (q * scale) @ k.transpose(0, 1, 3, 2)
    weights = mx.softmax(scores, axis=-1, precise=True)
    return weights @ v


def test_3bit_sdpa_correctness():
    """Rotated Q + quantized SDPA should approximate exact attention at 3-bit."""
    from mio.polarquant.cache import PolarQuantKVCache
    from mlx_lm.models.base import quantized_scaled_dot_product_attention

    mx.random.seed(99)
    D = HEAD_DIM
    scale = D ** -0.5

    k = mx.random.normal((1, N_KV_HEADS, 64, D))
    v = mx.random.normal((1, N_KV_HEADS, 64, D))
    q = mx.random.normal((1, N_KV_HEADS, 1, D))
    mx.eval(k, v, q)

    exact = _exact_sdpa(q, k, v, scale)
    mx.eval(exact)

    cache = PolarQuantKVCache(head_dim=D, bits=3, group_size=64)
    q_keys, q_values = cache.update_and_fetch(k, v)
    mx.eval(*q_keys, *q_values)

    q_rot = q @ cache.H
    out_rot = quantized_scaled_dot_product_attention(
        q_rot, q_keys, q_values,
        scale=scale, mask=None,
        group_size=cache.group_size, bits=cache.bits,
    )
    pq_out = out_rot @ cache.H
    mx.eval(pq_out)

    err = mx.mean((pq_out - exact) ** 2) / mx.mean(exact ** 2)
    assert float(err.item()) < 0.1, f"3-bit SDPA error {err.item():.4f} > 0.1"


# --- Cache ops ---

def test_cache_trim():
    from mio.polarquant.cache import PolarQuantKVCache

    k, v = _random_kv(64, seed=5)
    cache = PolarQuantKVCache(head_dim=HEAD_DIM, bits=4, group_size=64)
    cache.update_and_fetch(k, v)
    assert cache.offset == 64
    assert cache.trim(16) == 16
    assert cache.offset == 48


def test_cache_expansion():
    from mio.polarquant.cache import PolarQuantKVCache

    cache = PolarQuantKVCache(head_dim=HEAD_DIM, bits=4, group_size=64)
    for i in range(10):
        k, v = _random_kv(32, seed=i)
        cache.update_and_fetch(k, v)
    assert cache.offset == 320
