"""Token-importance selection strategies for DGSA.

Three strategies offered, in order of complexity:

  1. anchor_strided     — first F + last W + every Nth in between (no scoring)
  2. attention_scored   — score via target's first attention layer attention pattern
  3. draft_scored       — score via DFlash draft model (full speculator path)

The output is always a 1-D int32 tensor of sorted ascending positions to KEEP
during sparse-attention prefill. SSM layers see all positions; attention layers
read keys/values only at these positions.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


def anchor_strided(
    prompt_len: int,
    *,
    keep_first: int = 4,
    keep_last: int = 64,
    stride: int = 8,
) -> mx.array:
    """Always-keep first F + last W + every Nth token in between."""
    keep: set[int] = set()
    for i in range(min(keep_first, prompt_len)):
        keep.add(i)
    for i in range(max(0, prompt_len - keep_last), prompt_len):
        keep.add(i)
    for i in range(0, prompt_len, max(1, stride)):
        keep.add(i)
    return mx.array(sorted(keep), dtype=mx.int32)


def attention_scored(
    target_model,
    input_ids: mx.array,
    *,
    keep_ratio: float = 0.30,
    keep_first: int = 4,
    keep_last: int = 32,
    score_layer: int = 3,
) -> mx.array:
    """Score by aggregating attention received in the FIRST full-attention layer.

    For Qwen3.5: layer 3 is the first full-attention layer (every 4th).
    Runs the model up through that layer + captures attention pattern.
    """
    inner = (
        getattr(target_model, "text_model", None)
        or getattr(target_model, "language_model", None)
        or getattr(target_model, "model", None)
        or target_model
    )
    h = inner.embed_tokens(input_ids)
    L = int(input_ids.shape[1])
    from mlx_lm.models.base import create_attention_mask
    fa_mask = create_attention_mask(h, None)
    # Run layers up to score_layer.
    for layer_idx, layer in enumerate(inner.layers):
        if layer_idx == score_layer and not layer.is_linear:
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            B = 1
            q_proj_out = attn.q_proj(x)
            queries, _gate = mx.split(
                q_proj_out.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1,
            )
            keys = attn.k_proj(x)
            queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
            keys = attn.k_norm(
                keys.reshape(B, L, attn.num_key_value_heads, -1)
            ).transpose(0, 2, 1, 3)
            queries = attn.rope(queries)
            keys = attn.rope(keys)
            # GQA tile keys to match Q heads.
            n_repeats = queries.shape[1] // keys.shape[1]
            if n_repeats > 1:
                keys_tiled = mx.repeat(keys, n_repeats, axis=1)
            else:
                keys_tiled = keys
            # Approximate importance: mean_q · k^T  (skip softmax for speed).
            mean_q = queries.mean(axis=2, keepdims=True)        # (B, H, 1, D)
            scores = (mean_q * attn.scale) @ keys_tiled.transpose(0, 1, 3, 2)
            importance = scores.squeeze(-2).max(axis=1).squeeze(0)  # (L,) max over heads
            mx.eval(importance)
            return _select_top_with_anchors(
                importance, keep_ratio,
                keep_first=keep_first, keep_last=keep_last, prompt_len=L,
            )
        # Forward this layer to advance hidden state for subsequent layers.
        if layer.is_linear:
            mask = create_attention_mask(h, None)  # placeholder
            h = layer(h, mask=mask, cache=None)
        else:
            h = layer(h, mask=fa_mask, cache=None)

    # If we never hit a full-attention layer, fall back to anchor_strided.
    return anchor_strided(L, keep_first=keep_first, keep_last=keep_last, stride=int(1 / keep_ratio))


def _select_top_with_anchors(
    importance: mx.array,
    keep_ratio: float,
    *,
    keep_first: int,
    keep_last: int,
    prompt_len: int,
) -> mx.array:
    target = max(int(prompt_len * keep_ratio), keep_first + keep_last)
    target = min(target, prompt_len)
    order = mx.argsort(-importance).tolist()
    keep = set(order[:target])
    for i in range(min(keep_first, prompt_len)):
        keep.add(i)
    for i in range(max(0, prompt_len - keep_last), prompt_len):
        keep.add(i)
    return mx.array(sorted(keep), dtype=mx.int32)
