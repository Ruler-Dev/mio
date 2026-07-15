"""TurboQuant TQ4 parity + perf-degradation tests.

Covers:
  1. pack_4bit/unpack_4bit kernel round-trip (Metal + MLX paths).
  2. V1 cache (custom Metal kernels) at bits=2/3/4.
  3. V2 cache (mx.quantize) at bits=2/3/4.
  4. V3 cache (Lloyd-Max codebook) at bits=2/3/4.
  5. Reconstruction error monotonicity: err(2) >= err(3) >= err(4).
  6. Per-token update_and_fetch latency: TQ4 within 3x of TQ2 on V1.
"""

from __future__ import annotations

import time

import mlx.core as mx
import pytest


HEAD_DIM = 128
N_KV_HEADS = 4
T_NEW = 32


def _random_kv(T: int, D: int = HEAD_DIM, seed: int = 0):
    mx.random.seed(seed)
    k = mx.random.normal((1, N_KV_HEADS, T, D)).astype(mx.float32)
    v = mx.random.normal((1, N_KV_HEADS, T, D)).astype(mx.float32)
    mx.eval(k, v)
    return k, v


# --- 1. pack/unpack round-trip ---

def test_pack_unpack_4bit_kernel_roundtrip():
    from mio.turboquant.kernels import pack_4bit_indices, unpack_4bit_indices

    mx.random.seed(1)
    idx = mx.random.randint(0, 16, (8, 128)).astype(mx.uint8)
    packed = pack_4bit_indices(idx)
    assert packed.shape == (8, 16)
    assert packed.dtype == mx.uint32

    unpacked = unpack_4bit_indices(packed, 128)
    assert unpacked.shape == (8, 128)
    assert mx.array_equal(unpacked, idx.astype(mx.uint32))


def test_pack_unpack_4bit_mlx_matches_kernel():
    from mio.turboquant.codebook_ops import pack_4bit, unpack_4bit
    from mio.turboquant.kernels import pack_4bit_indices, unpack_4bit_indices

    mx.random.seed(2)
    idx = mx.random.randint(0, 16, (4, 64)).astype(mx.uint8)

    packed_kern = pack_4bit_indices(idx)
    packed_mlx = pack_4bit(idx)
    assert mx.array_equal(packed_kern, packed_mlx)

    unp_kern = unpack_4bit_indices(packed_kern, 64)
    unp_mlx = unpack_4bit(packed_mlx, 64)
    assert mx.array_equal(unp_kern, unp_mlx)


# --- 2/3/4. caches round-trip at bits in {2, 3, 4} ---

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_v1_cache_bits(bits):
    from mio.turboquant.cache import TurboQuantKVCache

    k, v = _random_kv(T_NEW)
    cache = TurboQuantKVCache(head_dim=HEAD_DIM, mse_bits=bits, use_qjl=False)
    out_k, out_v = cache.update_and_fetch(k, v)
    assert out_k.shape == k.shape
    assert cache.offset == T_NEW

    kidx = cache.get_key_indices()
    vidx = cache.get_value_indices()
    assert kidx.shape == (1, N_KV_HEADS, T_NEW, HEAD_DIM)
    assert vidx.shape == (1, N_KV_HEADS, T_NEW, HEAD_DIM)
    max_idx = (1 << bits) - 1
    assert int(kidx.max().item()) <= max_idx
    assert int(vidx.max().item()) <= max_idx


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_v2_cache_bits(bits):
    from mio.turboquant.cache_v2 import TurboQuantKVCacheV2

    k, v = _random_kv(T_NEW)
    cache = TurboQuantKVCacheV2(
        head_dim=HEAD_DIM, bits=bits, group_size=64, use_qjl=False,
    )
    q_keys, q_values = cache.update_and_fetch(k, v)
    assert cache.offset == T_NEW
    assert len(q_keys) == 3  # (data, scales, biases) tuple
    mx.eval(*q_keys, *q_values)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_v3_cache_bits(bits):
    from mio.turboquant.cache_v3 import TurboQuantKVCacheV3

    k, v = _random_kv(T_NEW)
    cache = TurboQuantKVCacheV3(head_dim=HEAD_DIM, bits=bits, use_qjl=False)
    cache.update_and_fetch(k, v)
    assert cache.offset == T_NEW


# --- 5. reconstruction error decreases monotonically with bits ---

def _recon_error_v3(bits: int) -> float:
    from mio.turboquant.cache_v3 import TurboQuantKVCacheV3

    k, v = _random_kv(64, seed=7)
    cache = TurboQuantKVCacheV3(head_dim=HEAD_DIM, bits=bits, use_qjl=False)
    cache.update_and_fetch(k, v)
    # Cached centroids are normalized+rotated. Reconstruct original-space keys:
    # k_hat = (centroids * norm) @ rotation_matrix  (since rot is orthogonal)
    k_rec_rot_norm = cache.get_key_centroids()[:, :, :cache.offset, :]
    k_norms = cache.key_norms[:, :, :cache.offset, None]
    k_hat = (k_rec_rot_norm * k_norms) @ cache.rotation_matrix
    err = mx.mean((k_hat - k) ** 2) / mx.mean(k ** 2)
    return float(err.item())


def test_v3_error_monotone_in_bits():
    e2 = _recon_error_v3(2)
    e3 = _recon_error_v3(3)
    e4 = _recon_error_v3(4)
    # Tight monotonicity; allow small slack for stochastic rotation.
    assert e4 < e3 < e2, f"expected e4<e3<e2, got e2={e2:.4f} e3={e3:.4f} e4={e4:.4f}"


# --- 6. perf degradation guardrail ---

def _time_v1_cache(bits: int, reps: int = 5) -> float:
    from mio.turboquant.cache import TurboQuantKVCache

    k, v = _random_kv(T_NEW, seed=11)
    # Warmup
    cache = TurboQuantKVCache(head_dim=HEAD_DIM, mse_bits=bits, use_qjl=False)
    cache.update_and_fetch(k, v)
    mx.eval(*cache.state)

    best = float("inf")
    for _ in range(reps):
        cache = TurboQuantKVCache(head_dim=HEAD_DIM, mse_bits=bits, use_qjl=False)
        start = time.perf_counter()
        cache.update_and_fetch(k, v)
        mx.eval(*cache.state)
        best = min(best, time.perf_counter() - start)
    return best


def test_tq4_perf_within_3x_of_tq2():
    """TQ4 per-update should not be more than 3x slower than TQ2.

    Guardrail only — kernels do similar work (one extra MLX bit shift width).
    """
    t2 = _time_v1_cache(2)
    t4 = _time_v1_cache(4)
    assert t4 < 3 * t2 + 5e-3, f"TQ4 ({t4*1e3:.2f}ms) >> 3x TQ2 ({t2*1e3:.2f}ms)"


@pytest.mark.parametrize("cache_version", ["v1", "v2", "v3"])
@pytest.mark.parametrize("use_qjl", [False, True])
def test_cache_state_roundtrip_restores_offset_and_arrays(cache_version, use_qjl):
    """Every advertised cache state must be restorable after trim/snapshot."""
    if cache_version == "v1":
        from mio.turboquant.cache import TurboQuantKVCache

        def make_cache():
            return TurboQuantKVCache(
                head_dim=HEAD_DIM, mse_bits=4, use_qjl=use_qjl
            )
    elif cache_version == "v2":
        from mio.turboquant.cache_v2 import TurboQuantKVCacheV2

        def make_cache():
            return TurboQuantKVCacheV2(
                head_dim=HEAD_DIM,
                bits=4,
                group_size=64,
                use_qjl=use_qjl,
                use_normalization=True,
            )
    else:
        from mio.turboquant.cache_v3 import TurboQuantKVCacheV3

        def make_cache():
            return TurboQuantKVCacheV3(
                head_dim=HEAD_DIM, bits=4, use_qjl=use_qjl
            )

    keys, values = _random_kv(9, seed=37)
    original = make_cache()
    original.update_and_fetch(keys, values)
    mx.eval(*original.state)

    restored = make_cache()
    restored.state = list(original.state)
    restored.meta_state = original.meta_state

    assert restored.offset == 9
    assert len(restored.state) == len(original.state)
    for expected, actual in zip(original.state, restored.state):
        assert mx.array_equal(expected, actual)

    restored.state = []
    assert restored.empty()
    assert restored.offset == 0


def test_v3_mixed_state_roundtrip():
    from mio.turboquant.cache_v3 import TurboQuantKVCacheV3

    keys, values = _random_kv(7, seed=41)
    original = TurboQuantKVCacheV3(
        head_dim=HEAD_DIM,
        bits=2,
        n_outlier=32,
        outlier_bits=3,
        use_qjl=True,
    )
    original.update_and_fetch(keys, values)
    mx.eval(*original.state)

    restored = TurboQuantKVCacheV3(
        head_dim=HEAD_DIM,
        bits=2,
        n_outlier=32,
        outlier_bits=3,
        use_qjl=True,
    )
    restored.state = list(original.state)
    restored.meta_state = original.meta_state

    assert restored.offset == 7
    assert len(restored.state) == 10
    for expected, actual in zip(original.state, restored.state):
        assert mx.array_equal(expected, actual)
