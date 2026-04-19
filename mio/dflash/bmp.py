"""Batched Multi-Path DFlash (BMP-DFlash).

Hybrid-architecture-compatible adaptation of DDTree (Ringel & Romano 2026).

Problem solved: DDTree's tree-attention verification cannot run on models with
SSM / linear-attention layers (gated delta-net in Qwen3.5), because SSM state
is recurrent and does not permit multiple ancestor-only branches within one
sequence forward pass.

BMP-DFlash idea: branch in the **batch dimension** instead of the sequence
dimension. One DFlash draft forward produces L per-position marginal distributions.
Use DDTree's Algorithm 1 to pick the top-K root-to-leaf paths, pad each to length
L with the marginal argmax continuation, stack as a (K, L) batch, then run ONE
target forward with batch_size=K. SSM, attention, and FFN all process each batch
row independently — no cross-branch state corruption. Pick the row with the
longest accepted prefix; slice all caches back to batch=1 along the winning row.

Cost:   K× verify activations (one target forward, wider batch dim).
Gain:   acceptance_len = max over K paths, strictly ≥ vanilla DFlash.

See docs/08-bmp-dflash.md for full derivation and tradeoffs.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models import cache as cache_mod

from mio.dflash.ddtree import build_ddtree, enumerate_paths
from mio.dflash.recurrent_rollback_cache import RecurrentRollbackCache


def extract_top_k(
    draft_logits: mx.array,
    k: int,
    suppress_token_mask: mx.array | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    """Per-position top-K token IDs and logprobs from draft logits.

    Args:
        draft_logits: (1, L, V) float — draft model logits for L future positions.
        k: how many tokens to keep per position.
        suppress_token_mask: (V,) optional mask (logprob offset) to mask tokens.

    Returns:
        tokens_per_pos:  list[list[int]] of shape (L, k)
        logprobs_per_pos: list[list[float]] of shape (L, k), descending by prob.
    """
    if draft_logits.ndim != 3 or draft_logits.shape[0] != 1:
        raise ValueError(f"draft_logits must be (1, L, V), got {draft_logits.shape}")
    _, L, V = draft_logits.shape
    k = min(k, V)

    if suppress_token_mask is not None:
        draft_logits = draft_logits + suppress_token_mask[None, None, :]

    # Softmax → log-softmax for numerical stability.
    log_probs = draft_logits[0] - mx.logsumexp(draft_logits[0], axis=-1, keepdims=True)
    # Top-K via argpartition is not available on MLX; use argsort (negative for desc).
    top_indices = mx.argsort(-log_probs, axis=-1)[:, :k]  # (L, k)
    # Gather logprobs
    row_idx = mx.arange(L)[:, None]
    top_logps = log_probs[row_idx, top_indices]
    mx.eval(top_indices, top_logps)

    tokens_per_pos = top_indices.tolist()
    logps_per_pos = top_logps.tolist()
    return tokens_per_pos, logps_per_pos


def build_bmp_batch(
    bonus_token: int,
    top_k_tokens: list[list[int]],
    top_k_logprobs: list[list[float]],
    num_paths: int,
    block_len: int,
    tree_budget: int | None = None,
) -> tuple[mx.array, list[list[int]]]:
    """Build a (K, block_len) batch tensor of K candidate paths for BMP verify.

    Each row starts with the bonus token, then block_len-1 candidate continuation
    tokens. Paths are the top-K root-to-leaf paths of a DDTree; shorter leaves
    get padded with marginal-argmax continuations so every row has length block_len.

    Args:
        bonus_token:    The target-produced token prepended to every path.
        top_k_tokens:   Per-position top-K tokens (list of length L).
        top_k_logprobs: Per-position log-probabilities matching top_k_tokens.
        num_paths:      Desired K (≥ 1). Will be clipped to available leaf count.
        block_len:      Total length of each path, including the bonus token.
        tree_budget:    DDTree node budget. Defaults to max(num_paths * (L-1), 16).

    Returns:
        batch_ids: (K_effective, block_len) mx.array uint32. K_effective ≤ num_paths.
        paths:     list of length-K_effective, each a list[int] of path tokens
                   (excluding bonus), useful for downstream match-computation.
    """
    L_draft = len(top_k_tokens)
    L_tail = block_len - 1
    if L_tail <= 0:
        return mx.array([[bonus_token]], dtype=mx.uint32), []

    if num_paths < 1:
        num_paths = 1

    # Deepest-draft prefix length we can exploit: min(L_tail, L_draft)
    L_cover = min(L_tail, L_draft)

    if tree_budget is None:
        # Enough budget to let the heap find K leaves at varying depths.
        tree_budget = max(num_paths * max(L_cover, 1), 16)

    if num_paths == 1:
        # Degenerate fast path: vanilla DFlash argmax trajectory.
        argmax_path = [top_k_tokens[i][0] for i in range(L_cover)]
        # Extend (can't — we're out of draft positions). Pad with argmax of last pos.
        if len(argmax_path) < L_tail:
            pad_tok = top_k_tokens[-1][0] if top_k_tokens else bonus_token
            argmax_path.extend([pad_tok] * (L_tail - len(argmax_path)))
        batch_ids = mx.array([[bonus_token] + argmax_path], dtype=mx.uint32)
        return batch_ids, [argmax_path]

    # Top-K via DDTree on the drafted positions.
    layout = build_ddtree(top_k_tokens[:L_cover], top_k_logprobs[:L_cover], tree_budget)
    raw_paths = enumerate_paths(layout, top_k_tokens[:L_cover], top_k_logprobs[:L_cover], num_paths)

    # Fallback: if DDTree returned fewer than num_paths leaves (e.g. all the
    # budget was spent on one deep trunk), top up with depth-1 alternatives.
    if len(raw_paths) < num_paths:
        argmax_tail = [top_k_tokens[i][0] for i in range(L_cover)]
        for rank in range(1, len(top_k_tokens[0]) if top_k_tokens else 0):
            if len(raw_paths) >= num_paths:
                break
            alt = [top_k_tokens[0][rank]] + argmax_tail[1:]
            if alt not in raw_paths:
                raw_paths.append(alt)

    # Pad every path to length L_tail with marginal argmax at the trailing positions.
    padded: list[list[int]] = []
    for p in raw_paths:
        p = list(p)
        while len(p) < L_tail:
            # Use argmax at the next position if draft covers it; else reuse last.
            next_pos = len(p)
            if next_pos < L_draft:
                p.append(top_k_tokens[next_pos][0])
            else:
                p.append(p[-1] if p else bonus_token)
        padded.append(p[:L_tail])

    K = len(padded)
    rows = [[bonus_token] + p for p in padded]
    batch_ids = mx.array(rows, dtype=mx.uint32)
    return batch_ids, padded


def expand_cache_batch(caches: list[Any], K: int) -> None:
    """In-place: replicate each cache's state along batch dim from 1 to K."""
    if K <= 1:
        return
    for c in caches:
        if isinstance(c, cache_mod.KVCache):
            if c.keys is not None:
                c.keys = mx.concatenate([c.keys] * K, axis=0)
                c.values = mx.concatenate([c.values] * K, axis=0)
        elif isinstance(c, cache_mod.ArraysCache):
            c.cache = [
                mx.concatenate([x] * K, axis=0) if x is not None else None
                for x in c.cache
            ]
        elif isinstance(c, RecurrentRollbackCache):
            c.cache = [
                mx.concatenate([x] * K, axis=0) if x is not None else None
                for x in c.cache
            ]
            # Clear any rollback tape — it becomes meaningless after batch expand.
            c._armed = False
            c._tape = None
            c._tape_k = None
            c._tape_g = None
            c._tape_qkv = None
            c._snapshot = None


def snapshot_cache(caches: list[Any]) -> list[dict]:
    """Capture enough state per cache to restore it after the verify forward.

    Snapshots are shallow (reference-only); MLX arrays are immutable, so as long
    as update_and_fetch rebinds (rather than mutates in place), snapshots stay valid.
    """
    snaps: list[dict] = []
    for c in caches:
        entry: dict = {"type": type(c).__name__}
        if isinstance(c, cache_mod.KVCache):
            entry["keys"] = c.keys
            entry["values"] = c.values
            entry["offset"] = c.offset
        elif isinstance(c, cache_mod.ArraysCache):
            entry["cache"] = list(c.cache)
        elif isinstance(c, RecurrentRollbackCache):
            entry["cache"] = list(c.cache)
            entry["lengths"] = c.lengths
            entry["left_padding"] = c.left_padding
        else:
            # Best effort for unknown cache types — fall back to offset.
            if hasattr(c, "offset"):
                entry["offset"] = c.offset
        snaps.append(entry)
    return snaps


def restore_cache(caches: list[Any], snaps: list[dict]) -> None:
    """In-place: revert each cache to its pre-verify snapshot state."""
    for c, snap in zip(caches, snaps):
        if isinstance(c, cache_mod.KVCache):
            c.keys = snap.get("keys")
            c.values = snap.get("values")
            if "offset" in snap:
                c.offset = snap["offset"]
        elif isinstance(c, cache_mod.ArraysCache):
            c.cache = list(snap.get("cache", c.cache))
        elif isinstance(c, RecurrentRollbackCache):
            c.cache = list(snap.get("cache", c.cache))
            c.lengths = snap.get("lengths")
            c.left_padding = snap.get("left_padding")
        elif "offset" in snap:
            c.offset = snap["offset"]


def filter_cache_batch(caches: list[Any], winner_idx: int) -> None:
    """In-place: slice each cache's batch dim down to just [winner_idx:winner_idx+1].

    For RecurrentRollbackCache this goes through its own `filter` method so that
    rollback tapes and checkpoint snapshots are filtered in lock-step.
    """
    slc = slice(winner_idx, winner_idx + 1)
    for c in caches:
        if isinstance(c, cache_mod.KVCache):
            if c.keys is not None:
                c.keys = c.keys[slc]
                c.values = c.values[slc]
        elif isinstance(c, cache_mod.ArraysCache):
            c.cache = [x[slc] if x is not None else None for x in c.cache]
        elif isinstance(c, RecurrentRollbackCache):
            c.filter(slc)


def per_row_acceptance(
    verify_token_ids: mx.array,
    posterior: mx.array,
) -> list[int]:
    """For each batch row, return acceptance length = # of positions where
    the drafted token at pos i+1 matches the target's argmax at pos i.

    Args:
        verify_token_ids: (K, block_len) uint32 — the input tokens
        posterior:        (K, block_len) uint32 — target's argmax at each position

    Returns:
        list of K ints. row k's acceptance = longest i s.t. for all j≤i,
        verify_token_ids[k, j+1] == posterior[k, j]. The bonus token at pos 0 is
        always "accepted" (it's guaranteed from prior round); counted 1 + i.
    """
    if verify_token_ids.shape[0] != posterior.shape[0]:
        raise ValueError(
            f"shape mismatch: {verify_token_ids.shape} vs {posterior.shape}"
        )
    K, block_len = verify_token_ids.shape
    if block_len < 2:
        return [0] * K

    # match[k, i] = True if verify[k, i+1] == posterior[k, i], for i in 0..block_len-2
    match = verify_token_ids[:, 1:] == posterior[:, :-1]
    mx.eval(match)
    match_list = match.tolist()

    result: list[int] = []
    for row in match_list:
        count = 0
        for m in row:
            if not m:
                break
            count += 1
        result.append(count)
    return result
