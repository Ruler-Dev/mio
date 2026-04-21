"""Prompt embedding via target model's own early-layer hidden state.

We run the target through the first `n_layers` decoder blocks only, then
mean-pool across the sequence dimension, giving a single vector in d_model
space that represents the prompt's position in the model's feature space.

Why this works:
  - The model is already running. Taking early hiddens adds near-zero
    latency because we'd be computing them anyway during a normal prefill.
  - Semantic similarity in this space is the "correct" proxy for "will
    these two prompts produce similar KV" — prompts that cluster here
    tend to produce similar downstream activations.
  - No separate embedding model to load (distilbert/MiniLM would be
    ~30 MB extra; we avoid it).

Design decision: we only compute the embedding when we need it (cache
miss). On cache hit we skip this step and go directly to full prefill
on the cached warm state.
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx


def embed_prompt(
    target_model: Any, tokens: list[int], *, n_early: int = 4,
) -> mx.array:
    """Return a d_model-dim unit-norm embedding for `tokens`.

    Runs the first `n_early` decoder layers on the tokens (no cache —
    this is an independent forward pass). Mean-pools the hidden state
    across the sequence dimension, L2-normalizes the result.

    Cost: O(n_early * L * d^2) FLOPs. For n_early=4, L=2000, d=2048:
    ~67 GFLOPs → ~2 ms on M4 Max. Negligible vs the full-prefill
    hundreds of ms we're trying to save.
    """
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)

    # Embed input tokens.
    input_ids = mx.array(tokens, dtype=mx.uint32)[None]
    h = text.embed_tokens(input_ids)  # (1, L, d)

    # Run first n_early layers without a cache. Pass mask=None so causal
    # default applies (only matters for attention layers; fine for GDN).
    for i in range(min(n_early, len(text.layers))):
        h = text.layers[i](h, mask=None, cache=None)

    # Mean-pool across L, L2 normalize.
    pooled = mx.mean(h.astype(mx.float32), axis=1)  # (1, d)
    norm = mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True) + 1e-9)
    return (pooled / norm)[0]  # (d,)


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Cosine between two unit-norm vectors."""
    return float(mx.sum(a * b).item())


def build_embedding_matrix(prototypes: list[dict]) -> mx.array:
    """Stack unit-norm embeddings into a (N, d) matrix for batch cosine search."""
    if not prototypes:
        return mx.zeros((0, 1), dtype=mx.float32)
    vecs = [p["embedding"] for p in prototypes]
    return mx.stack(vecs, axis=0).astype(mx.float32)


def top_k_cosine(
    query: mx.array, matrix: mx.array, k: int = 5,
) -> list[tuple[int, float]]:
    """Return (idx, similarity) for the top-k most similar rows of matrix.

    query: (d,), matrix: (N, d). Both assumed L2-normalized.
    """
    if matrix.shape[0] == 0:
        return []
    scores = mx.matmul(matrix, query)  # (N,)
    mx.eval(scores)
    s = scores.tolist()
    idx = sorted(range(len(s)), key=lambda i: -s[i])[:k]
    return [(i, float(s[i])) for i in idx]
