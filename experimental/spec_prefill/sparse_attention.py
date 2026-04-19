"""Position-aware Qwen3 attention forward for sparse prefill.

Reimplements `mlx_lm.models.qwen3.Attention.__call__` and the model forward
with explicit position_ids support, so we can feed K selected tokens with
their ORIGINAL positions during prefill and continue decoding with logical
positions starting at the original prompt length.

Verified to produce identical output to mlx-lm's stock Qwen3 forward when
position_ids = arange(prompt_len) — see tests/test_sparse_attention.py.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.base import (
    create_attention_mask,
    scaled_dot_product_attention,
)

from experimental.spec_prefill.rope_pos import apply_rope_per_position


def sparse_attention_forward(
    attn,
    x: mx.array,
    *,
    positions: mx.array,
    mask: Optional[Any] = None,
    cache: Optional[Any] = None,
    rope_theta: float,
) -> mx.array:
    """Run a single Qwen3 attention block with explicit per-token positions.

    Args:
        attn:       The mlx_lm.models.qwen3.Attention instance.
        x:          (B, L, D) — pre-attention input (post input_layernorm).
        positions:  (L,) int — original positions for each of L tokens.
        mask:       Optional attention mask. "causal" or array, passed to SDPA.
        cache:      Optional KVCache. If given, keys/values are appended.
        rope_theta: RoPE base (Qwen3 uses 1e6).
    """
    B, L, _ = x.shape
    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads

    queries = attn.q_proj(x)
    keys = attn.k_proj(x)
    values = attn.v_proj(x)

    queries = attn.q_norm(queries.reshape(B, L, n_heads, -1)).transpose(0, 2, 1, 3)
    keys = attn.k_norm(keys.reshape(B, L, n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

    queries = apply_rope_per_position(queries, positions, theta=rope_theta)
    keys = apply_rope_per_position(keys, positions, theta=rope_theta)

    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)

    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=attn.scale, mask=mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(output)


def sparse_block_forward(
    block,
    x: mx.array,
    *,
    positions: mx.array,
    mask: Optional[Any] = None,
    cache: Optional[Any] = None,
    rope_theta: float,
) -> mx.array:
    """Run a single Qwen3 TransformerBlock with explicit positions."""
    r = sparse_attention_forward(
        block.self_attn, block.input_layernorm(x),
        positions=positions, mask=mask, cache=cache, rope_theta=rope_theta,
    )
    h = x + r
    r = block.mlp(block.post_attention_layernorm(h))
    return h + r


def sparse_model_forward(
    model,
    input_ids: mx.array,
    *,
    positions: mx.array,
    cache: Optional[list] = None,
    return_hidden: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """Forward an mlx_lm Qwen3 Model with explicit position_ids for RoPE.

    Args:
        model:        The top-level Model from `mlx_lm.models.qwen3`.
        input_ids:    (B, L) uint32.
        positions:    (L,) int — original positions for each of L tokens.
        cache:        Optional list of per-layer caches.
        return_hidden: if True, also return the final pre-LM-head hidden state.
    """
    inner = model.model  # Qwen3Model
    rope_theta = float(inner.args.rope_theta)

    h = inner.embed_tokens(input_ids)
    if cache is None:
        cache = [None] * len(inner.layers)

    # Causal mask. SpecPrefill feeds selected tokens in order of original
    # position, so a standard upper-triangular causal mask is correct in
    # sparse-token-order. Use mlx-lm's helper for parity with the stock path.
    L = int(input_ids.shape[1])
    mask = create_attention_mask(h, cache[0]) if L > 1 else None

    for layer, layer_cache in zip(inner.layers, cache):
        h = sparse_block_forward(
            layer, h,
            positions=positions, mask=mask, cache=layer_cache,
            rope_theta=rope_theta,
        )

    norm_h = inner.norm(h)

    if model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(norm_h)
    else:
        logits = model.lm_head(norm_h)

    if return_hidden:
        return logits, norm_h
    return logits
