"""Token importance scoring for SpecPrefill.

Per the paper (Yang et al. 2025): run a small "speculator" model on the prompt,
optionally run N look-ahead decode steps, then aggregate attention scores via
max over (layers, heads) and mean over (look-ahead positions) to produce a
single importance scalar per prompt token.

For mio's experimental setup we use a smaller Qwen3 model as the speculator
(could be the same family, since acceptance is highest in-family). Initial
implementation: 0-step look-ahead — score from the speculator's own attention
when consuming the prompt. We capture attention patterns by overriding the
attention forward of each speculator layer.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import mlx.core as mx


def _qkv_attention_scores(
    q: mx.array, k: mx.array, scale: float
) -> mx.array:
    """Compute softmax attention scores (B, H, T, T) for one layer.

    Mirrors mlx.fast.scaled_dot_product_attention's score computation but
    materializes the (T, T) score matrix so we can capture it.
    """
    # GQA: tile keys to match query heads
    nq, nk = q.shape[1], k.shape[1]
    if nq != nk:
        n_repeats = nq // nk
        k = mx.repeat(k, n_repeats, axis=1)
    scores = (q * scale) @ k.transpose(0, 1, 3, 2)  # (B, H, T_q, T_k)
    # Apply causal mask
    T_q, T_k = q.shape[2], k.shape[2]
    if T_q == T_k:
        causal = mx.triu(mx.ones((T_q, T_k), dtype=mx.bool_), k=1)
        scores = mx.where(causal, mx.array(-1e9, dtype=scores.dtype), scores)
    return mx.softmax(scores.astype(mx.float32), axis=-1)


def _attention_received_per_token(
    q: mx.array, k: mx.array, scale: float
) -> mx.array:
    """Approximate mean-attention-received per token without materializing (T, T).

    Trick: for token i, the attention received is sum_{j > i} softmax(q_j @ k_i^T).
    A computationally cheap proxy: take the mean of (q_j @ k_i) values over j > i.
    Skips softmax → no exp normalization, but the relative ordering of importance
    scores is largely preserved (consistent with paper's "max-mean" intuition).
    Returns (B, H, T_k).
    """
    # GQA tile
    nq, nk = q.shape[1], k.shape[1]
    if nq != nk:
        n_repeats = nq // nk
        k = mx.repeat(k, n_repeats, axis=1)
    # Sum of (q_j) over future j for each k_i: equivalent to summing q[..., j_future, :] then dotting with k_i.
    # For "all queries" (no causal masking), this is the average q vector dotted with each key.
    # We use that approximation — paper's max-mean over heads/layers absorbs the exact normalization.
    mean_q = q.mean(axis=2, keepdims=True)  # (B, H, 1, D)
    # (B, H, 1, D) @ (B, H, D, T_k) → (B, H, 1, T_k)
    scores = (mean_q * scale) @ k.transpose(0, 1, 3, 2)
    return scores.squeeze(-2)  # (B, H, T_k)


def score_prompt_tokens(
    speculator_model,
    input_ids: mx.array,
    *,
    layers_to_use: int | None = None,
    heads_to_use: int | None = None,
    early_exit_after: int | None = None,
    fast_score: bool = True,
) -> mx.array:
    """Compute per-token importance scores by running the speculator and
    capturing per-layer attention received from later positions.

    Aggregation: per layer, per head, we compute mean attention received over
    all later query positions. Then take max over layers and heads (paper's
    "max-mean": max over layer+head, mean over look-ahead).

    Args:
        speculator_model: A loaded Qwen3 model. We invoke a single forward
            on the prompt and capture each attention layer's softmax matrix.
        input_ids: (1, L) uint32 — the prompt.
        layers_to_use: optionally use only the last N layers. None = all.
        heads_to_use: optionally use only the first N heads. None = all.

    Returns:
        (L,) float32 — per-token importance, normalized to [0, 1].
    """
    inner = speculator_model.model
    n_layers = len(inner.layers)
    if layers_to_use is None:
        layer_indices_to_aggregate = set(range(n_layers))
    else:
        layer_indices_to_aggregate = set(range(max(0, n_layers - layers_to_use), n_layers))

    h = inner.embed_tokens(input_ids)
    L = int(input_ids.shape[1])

    # Build causal mask explicitly so we can re-use it in score capture.
    from mlx_lm.models.base import create_attention_mask
    mask = create_attention_mask(h, None)
    rope_theta = float(inner.args.rope_theta)

    # Per-layer attention pattern aggregator.
    # We compute mean-attention-received per token (mean over T_q axis).
    layer_scores: list[mx.array] = []  # each (H, L)

    last_layer_idx = (
        n_layers - 1 if early_exit_after is None
        else min(early_exit_after - 1, n_layers - 1)
    )
    for layer_idx, layer in enumerate(inner.layers):
        if layer_idx > last_layer_idx:
            break
        attn = layer.self_attn
        x = layer.input_layernorm(h)
        B = 1
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads
        queries = attn.q_proj(x)
        keys = attn.k_proj(x)
        values = attn.v_proj(x)
        queries = attn.q_norm(queries.reshape(B, L, n_heads, -1)).transpose(0, 2, 1, 3)
        keys = attn.k_norm(keys.reshape(B, L, n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
        # Apply RoPE (use stock since contiguous positions).
        queries = attn.rope(queries, offset=0)
        keys = attn.rope(keys, offset=0)

        # Compute scores (only for layers we want to aggregate)
        if layer_idx in layer_indices_to_aggregate:
            if fast_score:
                # O(D · T) approximate path — no (T, T) materialization.
                mean_attn = _attention_received_per_token(
                    queries, keys, attn.scale,
                ).squeeze(0)  # (H, L)
            else:
                scores = _qkv_attention_scores(queries, keys, attn.scale)  # (B, H, L, L)
                mean_attn = scores.mean(axis=2).squeeze(0)
            if heads_to_use is not None:
                mean_attn = mean_attn[:heads_to_use]
            layer_scores.append(mean_attn)

        # Continue forward to get next layer's input
        # Reuse a real SDPA pass for the actual hidden state propagation.
        from mlx_lm.models.base import scaled_dot_product_attention
        attn_out = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=attn.scale, mask=mask,
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        attn_out = attn.o_proj(attn_out)
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    # Stack: (n_layers, H, L)
    stacked = mx.stack(layer_scores, axis=0)
    # max over layers and heads → (L,)
    importance = stacked.max(axis=(0, 1))
    # Normalize to [0, 1]
    imp_min = importance.min()
    imp_max = importance.max()
    importance = (importance - imp_min) / (imp_max - imp_min + 1e-9)
    return importance


def select_top_k_tokens(
    importance: mx.array,
    keep_ratio: float,
    *,
    always_keep_first: int = 4,
    always_keep_last: int = 16,
    chunk_size: int = 1,
) -> mx.array:
    """Select indices of the most important tokens (sorted ascending).

    Args:
        importance:        (L,) float32 — per-token scores.
        keep_ratio:        Fraction of tokens to keep (e.g., 0.4 = 40%).
        always_keep_first: Always keep this many tokens at the start (BOS / sys preamble).
        always_keep_last:  Always keep this many tokens at the end (most recent context).
        chunk_size:        If > 1, average importance within consecutive chunks then
                          select top-K chunks, keeping all tokens within each chunk.

    Returns:
        (K,) int32 — sorted ascending indices of selected tokens.
    """
    L = int(importance.shape[0])
    target_keep = max(int(L * keep_ratio), always_keep_first + always_keep_last)
    target_keep = min(target_keep, L)

    if chunk_size > 1:
        # Chunk-based selection (paper §3.2.3)
        n_chunks = (L + chunk_size - 1) // chunk_size
        # Pad importance to multiple of chunk_size
        pad_n = n_chunks * chunk_size - L
        padded = mx.concatenate([importance, mx.zeros((pad_n,), dtype=importance.dtype)]) if pad_n else importance
        chunked = padded.reshape(n_chunks, chunk_size).mean(axis=-1)
        n_chunks_keep = max(1, target_keep // chunk_size)
        # Top-K chunks
        order = mx.argsort(-chunked).tolist()
        keep_chunks = sorted(order[:n_chunks_keep])
        idx_set = set()
        for c in keep_chunks:
            for j in range(c * chunk_size, min((c + 1) * chunk_size, L)):
                idx_set.add(j)
    else:
        # Token-level selection
        order = mx.argsort(-importance).tolist()
        idx_set = set(order[:target_keep])

    # Add boundary keep
    for i in range(min(always_keep_first, L)):
        idx_set.add(i)
    for i in range(max(0, L - always_keep_last), L):
        idx_set.add(i)

    return mx.array(sorted(idx_set), dtype=mx.int32)
