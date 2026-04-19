"""Per-position RoPE for sparse prefill.

The stock mlx_lm Qwen3 attention applies RoPE based on `cache.offset` (assumes
contiguous positions). For SpecPrefill we feed K << N selected tokens with
their ORIGINAL positions, so RoPE must be applied per-token from explicit
position IDs.

This module provides:

  apply_rope_per_position(x, positions, theta) — rotate (B, H, T, D) tensor
  using the given (T,) absolute positions, producing the same numerical result
  that mlx.fast.rope would for contiguous offset+i positions.

We validate parity with mlx.fast.rope on contiguous positions in tests/.
"""

from __future__ import annotations

import mlx.core as mx


def apply_rope_per_position(
    x: mx.array,
    positions: mx.array,
    theta: float = 1_000_000.0,
) -> mx.array:
    """Apply RoPE to (B, H, T, D) tensor with arbitrary per-token positions.

    Args:
        x:         (B, H, T, D) float — Q or K post-projection-and-norm.
        positions: (T,) int or float — absolute position for each of T tokens.
        theta:     RoPE base. Qwen3 uses 1e6 (Qwen2.5 uses 1e6 too).

    Returns:
        Tensor of same shape with RoPE applied per position.

    Convention: GPT-J-style ("non-interleaved" / `traditional=False`):
        x = [x_first_half, x_second_half], rotations operate on (x[i], x[i+D/2]).
    """
    if x.ndim != 4:
        raise ValueError(f"expected (B, H, T, D), got {x.shape}")
    B, H, T, D = x.shape
    if T != int(positions.shape[0]):
        raise ValueError(
            f"positions length {positions.shape[0]} != T {T}"
        )
    if D % 2 != 0:
        raise ValueError(f"head dim must be even, got {D}")

    half_D = D // 2
    # Match mlx.fast.rope's convention exactly:
    #   inv_freq[k] = 1 / (base ** (2k / D)) for k in [0, half_D)
    # i.e. mx.arange(0, dims, 2)[..]/dims used as exponent with base=theta.
    inv_freq = (1.0 / (float(theta) ** (mx.arange(0, D, 2, dtype=mx.float32) / D)))
    # (T, half_D) — keep cos/sin in float32; cast at the end after the rotation
    # so we only round once.
    freqs = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    cos = mx.cos(freqs)  # (T, D/2) float32
    sin = mx.sin(freqs)

    # Reshape to broadcast with (B, H, T, D)
    # cos, sin → (1, 1, T, D/2)
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]

    x1 = x[..., :half_D].astype(mx.float32)
    x2 = x[..., half_D:].astype(mx.float32)
    out_first = x1 * cos_b - x2 * sin_b
    out_second = x1 * sin_b + x2 * cos_b
    out = mx.concatenate([out_first, out_second], axis=-1)
    return out.astype(x.dtype)
