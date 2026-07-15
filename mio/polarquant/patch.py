"""Monkey-patch for mlx-lm's SDPA dispatch — PolarQuant path.

Only active when Hadamard rotation is used (sub-4-bit). At 4-bit,
PolarQuantKVCache behaves identically to QuantizedKVCache and needs
no patching — mlx_lm's native quantized SDPA handles it.

When rotation is active (bits < 4):
  1. Rotate queries: Q_rot = Q @ H
  2. Delegate to mlx_lm's quantized_scaled_dot_product_attention
  3. Inverse-rotate output: out = out_rot @ H
"""

import sys

import mlx_lm.models.base as _base

from mio.polarquant.cache import PolarQuantKVCache

_original_sdpa = _base.scaled_dot_product_attention
_patched = False


def _patched_sdpa(queries, keys, values, cache, scale, mask, **kwargs):
    if isinstance(cache, PolarQuantKVCache) and cache._use_rotation:
        q_rot = queries @ cache.H
        out_rot = _base.quantized_scaled_dot_product_attention(
            q_rot, keys, values,
            scale=scale, mask=mask,
            group_size=cache.group_size, bits=cache.bits,
        )
        return out_rot @ cache.H
    return _original_sdpa(queries, keys, values, cache, scale, mask, **kwargs)


def _rebind(target):
    _base.scaled_dot_product_attention = target
    for name, mod in list(sys.modules.items()):
        if mod is None or mod is _base:
            continue
        if not (name.startswith("mlx_lm.") or name.startswith("mio.")):
            continue
        try:
            current = mod.__dict__.get("scaled_dot_product_attention", None)
        except Exception:
            continue
        if current is _original_sdpa or current is _patched_sdpa:
            mod.scaled_dot_product_attention = target


def apply():
    """Activate PolarQuant SDPA patch. Idempotent."""
    global _patched
    if _patched:
        return
    _rebind(_patched_sdpa)
    _patched = True


def revert():
    """Remove the patch."""
    global _patched
    _rebind(_original_sdpa)
    _patched = False
