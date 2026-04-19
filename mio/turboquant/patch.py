"""Monkey-patch for mlx-lm's SDPA dispatch.

Supports TurboQuant V1 (fused Metal kernel), V2 (mx.quantized_matmul),
V3 (Lloyd-Max codebook with software dequant).

Note: modules that do `from mlx_lm.models.base import scaled_dot_product_attention`
capture a local reference at import time, so patching `_base.scaled_dot_product_attention`
alone misses them. We also explicitly rebind known call-site modules
(e.g. `mio.dflash.runtime`).
"""

import importlib
import sys

import mlx.core as mx
import mlx_lm.models.base as _base

from mio.turboquant.attention_fused import turboquant_fused_sdpa
from mio.turboquant.attention_v2 import turboquant_v2_sdpa
from mio.turboquant.attention_v3 import turboquant_v3_sdpa
from mio.turboquant.cache import TurboQuantKVCache
from mio.turboquant.cache_v2 import TurboQuantKVCacheV2
from mio.turboquant.cache_v3 import TurboQuantKVCacheV3

_original_sdpa = _base.scaled_dot_product_attention
_patched = False

def _patched_sdpa(queries, keys, values, cache, scale, mask, **kwargs):
    if isinstance(cache, TurboQuantKVCacheV3):
        return turboquant_v3_sdpa(queries, cache, scale, mask)
    if isinstance(cache, TurboQuantKVCacheV2):
        return turboquant_v2_sdpa(queries, keys, values, cache, scale, mask)
    if isinstance(cache, TurboQuantKVCache):
        return turboquant_fused_sdpa(queries, cache, scale, mask)
    return _original_sdpa(queries, keys, values, cache, scale, mask, **kwargs)


def _rebind(target):
    """Rebind `scaled_dot_product_attention` everywhere it's been imported by name.

    `from mlx_lm.models.base import scaled_dot_product_attention` in mlx-lm's
    per-model modules (qwen3, qwen3_moe, ...) and in mio.dflash.runtime
    captures a local reference at import time. Patching `_base.<attr>` alone
    misses them — we rebind every already-loaded module that has the symbol.
    """
    _base.scaled_dot_product_attention = target
    for name, mod in list(sys.modules.items()):
        if mod is None or mod is _base:
            continue
        # Scope to modules that plausibly import the SDPA symbol.
        # Scanning all modules triggers lazy `__getattr__` in packages like
        # transformers, which tries to import torch and crashes.
        if not (name.startswith("mlx_lm.") or name.startswith("mio.")):
            continue
        try:
            current = mod.__dict__.get("scaled_dot_product_attention", None)
        except Exception:
            continue
        if current is _original_sdpa or current is _patched_sdpa:
            mod.scaled_dot_product_attention = target


def apply():
    """Activates the TurboQuant SDPA patch. Idempotent."""
    global _patched
    if _patched:
        return
    _rebind(_patched_sdpa)
    _patched = True


def revert():
    """Removes the patch."""
    global _patched
    _rebind(_original_sdpa)
    _patched = False
