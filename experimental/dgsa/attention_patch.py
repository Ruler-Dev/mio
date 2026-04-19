"""Monkey-patch Qwen3NextAttention to support sparse-key prefill.

When `dgsa_active(keep_indices)` is in effect, the patched `__call__` slices
keys and values down to the kept indices BEFORE storing in cache and BEFORE
attention. The rest of the layer (Q computation, RoPE, gate, output proj) is
unchanged. SSM (gated_delta_net) layers are NOT patched.

After prefill, the KV cache contains exactly K entries per attention layer
(K = len(keep_indices)). Decode appends new entries normally; new queries
attend to all (sparse-prefill + new-decode) entries.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3_next import Qwen3NextAttention

from experimental.dgsa.state import get_state


_PATCHED = False
_ORIGINAL_CALL = None


def _dgsa_attention_call(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
) -> mx.array:
    """Replacement for Qwen3NextAttention.__call__ supporting sparse-key prefill."""
    state = get_state()
    if not state.active:
        # Fallback: original behaviour.
        return _ORIGINAL_CALL(self, x, mask, cache)

    keep = state.keep_indices
    if keep is None or int(keep.shape[0]) == 0:
        return _ORIGINAL_CALL(self, x, mask, cache)

    B, L, D = x.shape

    # Standard projection + reshape (mirrors stock).
    q_proj_output = self.q_proj(x)
    queries, gate = mx.split(
        q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(B, L, -1)

    keys = self.k_proj(x)
    values = self.v_proj(x)

    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(
        keys.reshape(B, L, self.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = values.reshape(
        B, L, self.num_key_value_heads, -1
    ).transpose(0, 2, 1, 3)

    # Apply RoPE to ALL positions first (correct absolute positions).
    if cache is not None:
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)

    # Slice K, V down to important indices BEFORE cache update.
    keys_sparse = keys[:, :, keep, :]
    values_sparse = values[:, :, keep, :]

    if cache is not None:
        # Stage the sliced K, V into the cache. update_and_fetch returns the
        # full accumulated K, V — which now includes only kept positions.
        keys_sparse, values_sparse = cache.update_and_fetch(keys_sparse, values_sparse)

    # Attention: queries are dense (all L positions), keys/values are sparse.
    # Build a mask: query at position i can attend to a sparse key at original
    # position keep[j] iff keep[j] <= i (causal).
    q_pos = mx.arange(L, dtype=mx.int32)        # (L,)
    if cache is None:
        # Fresh prefill — kept positions are exactly keep_indices.
        k_pos = keep                            # (K,)
    else:
        # Cache may already contain some kept positions; assume the freshly-
        # added ones live at positions `keep` and pre-existing ones live at
        # positions captured by their original (cache) indexing.
        # For pure prefill with empty cache initially, this collapses to keep.
        k_pos = keep
    causal_bool = q_pos[:, None] >= k_pos[None, :]   # (L, K) True where allowed
    attn_mask = mx.where(
        causal_bool,
        mx.array(0.0, dtype=keys_sparse.dtype),
        mx.array(-1e9, dtype=keys_sparse.dtype),
    )

    output = scaled_dot_product_attention(
        queries, keys_sparse, values_sparse,
        cache=cache, scale=self.scale, mask=attn_mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output * mx.sigmoid(gate))


def patch():
    """Install the DGSA attention patch. Idempotent."""
    global _PATCHED, _ORIGINAL_CALL
    if _PATCHED:
        return
    _ORIGINAL_CALL = Qwen3NextAttention.__call__
    Qwen3NextAttention.__call__ = _dgsa_attention_call
    _PATCHED = True


def unpatch():
    """Restore stock Qwen3NextAttention.__call__."""
    global _PATCHED, _ORIGINAL_CALL
    if not _PATCHED:
        return
    Qwen3NextAttention.__call__ = _ORIGINAL_CALL
    _ORIGINAL_CALL = None
    _PATCHED = False
