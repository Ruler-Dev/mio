"""Validate per-position RoPE matches mlx.fast.rope for contiguous positions.

If our manual implementation matches mlx.fast.rope on dense input, we trust it
to apply correct RoPE for arbitrary (sparse) positions in SpecPrefill.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experimental.spec_prefill.rope_pos import apply_rope_per_position  # type: ignore


def _stock_mlx_rope(x: mx.array, offset: int, theta: float) -> mx.array:
    """Mirror what mlx_lm Qwen3 does:  self.rope(x, offset=offset)."""
    import mlx.nn as nn
    head_dim = int(x.shape[-1])
    rope = nn.RoPE(dims=head_dim, traditional=False, base=theta)
    return rope(x, offset=offset)


@pytest.mark.parametrize("theta", [10_000.0, 1_000_000.0])
@pytest.mark.parametrize("offset,T", [(0, 8), (5, 8), (123, 32)])
def test_dense_matches_mlx_rope(theta, offset, T):
    """Per-position RoPE with positions=[offset..offset+T-1] should equal mlx RoPE(offset)."""
    mx.random.seed(0)
    B, H, D = 2, 4, 64
    x = mx.random.normal((B, H, T, D)).astype(mx.float32)
    positions = mx.arange(offset, offset + T, dtype=mx.int32)

    out_ours = apply_rope_per_position(x, positions, theta=theta)
    out_stock = _stock_mlx_rope(x, offset=offset, theta=theta)
    diff = mx.abs(out_ours - out_stock).max().item()
    assert diff < 1e-4, f"max diff {diff:.6f} (theta={theta}, offset={offset}, T={T})"


def test_sparse_positions_correctness():
    """For positions [0, 1, 3, 6, 7], the per-position RoPE result at each
    selected index should equal applying RoPE at offset=0 to the dense [0..7]
    sequence and gathering those indices.
    """
    mx.random.seed(1)
    B, H, T, D = 1, 2, 8, 32
    x_dense = mx.random.normal((B, H, T, D)).astype(mx.float32)

    # Dense reference
    dense_rope = _stock_mlx_rope(x_dense, offset=0, theta=1e6)

    # Sparse: pick rows {0, 1, 3, 6, 7}
    keep = mx.array([0, 1, 3, 6, 7], dtype=mx.int32)
    x_sparse = x_dense[:, :, keep, :]
    sparse_rope = apply_rope_per_position(x_sparse, keep, theta=1e6)

    # Compare to gathered dense
    expected = dense_rope[:, :, keep, :]
    diff = mx.abs(sparse_rope - expected).max().item()
    assert diff < 1e-4, f"max diff {diff:.6f}"


def test_dtype_preserved():
    """Output dtype matches input dtype."""
    mx.random.seed(2)
    x = mx.random.normal((1, 2, 4, 32)).astype(mx.bfloat16)
    pos = mx.arange(4, dtype=mx.int32)
    out = apply_rope_per_position(x, pos)
    assert out.dtype == mx.bfloat16
