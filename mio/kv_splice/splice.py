"""Production splice hooks.

Given a model and a list of splice sites (detected chunks at prompt
positions), install hooks that override k_proj/v_proj output at those
positions on the spliceable layers (L3-L19 by default).

Supports MULTIPLE chunks in a single prompt. Each attention layer call
during prefill receives x of shape (B, L, D). We build a mask of
"is this position covered by a splice" and assemble the final K and V
output by piecewise replacement.

Only triggers on the prefill pass (L >= longest-splice-end). Decode
passes pass through untouched.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from mio.kv_splice import SPLICEABLE_LAYERS_QWEN36_A3B
from mio.kv_splice.detect import SpliceSite
from mio.kv_splice.store import ChunkStore


def install_splice_hooks(
    target_model: Any,
    sites: list[SpliceSite],
    store: ChunkStore,
    *,
    layer_set: tuple[int, ...] = SPLICEABLE_LAYERS_QWEN36_A3B,
) -> Any:
    """Install hooks for all sites. Returns a cleanup callable.

    Behavior:
      - For each attention layer in layer_set, override k_proj/v_proj
        output at the site positions with the stored spliced values.
      - k_proj override is pre-RoPE (K_base); the model's own RoPE call
        rotates it to the target position.
      - v_proj override is straight value replacement.
      - Non-layer_set layers untouched.
      - Decode-time calls (L smaller than max(end)) untouched.
    """
    if not sites:
        return lambda: None

    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)

    # Map layer index → attention module for the subset we'll patch.
    attn_instances = {
        i: text.layers[i].self_attn for i in layer_set
        if i < len(text.layers) and not bool(getattr(text.layers[i], "is_linear", False))
    }
    id_to_layer_idx = {id(attn): i for i, attn in attn_instances.items()}
    if not attn_instances:
        return lambda: None

    # Load spliced KV for each site × layer. Stored arrays are already
    # float16 with shape (n_kv_heads, chunk_len, d_head). We cast to
    # matching dtype at patch time.
    site_kvs: list[dict[int, dict[str, mx.array]]] = []
    for site in sites:
        kv = store.load_kv(site.chunk_id)
        if kv is None:
            site_kvs.append({})
        else:
            site_kvs.append(kv)

    # Expected min sequence length at which to splice (must cover the
    # farthest site).
    min_L = max(s.end for s in sites) if sites else 0

    # Precompute for each layer: list of (site_idx, start, end, k_base, v)
    # so we don't scan all layers for every site in the hot path.
    layer_payload: dict[int, list[tuple[int, int, int, mx.array, mx.array]]] = {}
    for li in attn_instances:
        entries: list[tuple[int, int, int, mx.array, mx.array]] = []
        for si, site in enumerate(sites):
            kv = site_kvs[si]
            if li not in kv:
                continue
            entries.append((si, site.start, site.end, kv[li]["k_base"], kv[li]["v"]))
        layer_payload[li] = entries

    # Patch each distinct attention class (usually just one).
    distinct_attn_cls = {}
    for attn in attn_instances.values():
        cls = type(attn)
        if cls not in distinct_attn_cls:
            distinct_attn_cls[cls] = cls.__call__

    active_idx = {"i": None}
    linear_cls = type(list(attn_instances.values())[0].k_proj)
    orig_linear_call = linear_cls.__call__

    def linear_wrap(self, x):
        y = orig_linear_call(self, x)
        li = active_idx["i"]
        if li is None:
            return y
        attn = attn_instances[li]
        is_k = id(self) == id(attn.k_proj)
        is_v = id(self) == id(attn.v_proj)
        if not (is_k or is_v):
            return y
        B, L, total = y.shape
        if L < min_L:
            return y  # decode-time call, no splice
        entries = layer_payload[li]
        if not entries:
            return y

        # In-place overlay: operate directly on y in (B, L, total) shape.
        # Avoids the two big (B, L, n_kv, d_h) <-> (B, n_kv, L, d_h)
        # transposes of the naive implementation (each copies the full
        # sequence). Plumbing cost now scales with chunk size, not context
        # length, because pre/post are views and only the small spliced
        # block is materialized.
        #
        # Multi-site support: partition y into interleaved segments and
        # concat once. Sites are non-overlapping by construction (detect
        # enforces this via the `taken` list), so a single ordered pass
        # covers every position.
        entries_sorted = sorted(entries, key=lambda e: e[1])
        parts = []
        cursor = 0
        for (_, start, end, k_base, v) in entries_sorted:
            spliced_full = k_base if is_k else v
            site_len = end - start
            stored_len = int(spliced_full.shape[1])
            if stored_len < site_len:
                continue
            # (n_kv, site_len, d_h) -> (site_len, n_kv, d_h) -> (site_len, total)
            spliced = (
                spliced_full[:, :site_len, :]
                .transpose(1, 0, 2)
                .reshape(site_len, total)
            )
            spliced = mx.broadcast_to(
                spliced[None, :, :], (B, site_len, total),
            ).astype(y.dtype)
            if cursor < start:
                parts.append(y[:, cursor:start, :])
            parts.append(spliced)
            cursor = end
        if not parts:
            return y
        if cursor < L:
            parts.append(y[:, cursor:, :])
        if len(parts) == 1:
            return parts[0]
        return mx.concatenate(parts, axis=1)

    def attn_wrap(original_call):
        def wrapper(self, x, mask=None, cache=None):
            li = id_to_layer_idx.get(id(self))
            if li is not None:
                active_idx["i"] = li
                try:
                    return original_call(self, x, mask=mask, cache=cache)
                finally:
                    active_idx["i"] = None
            return original_call(self, x, mask=mask, cache=cache)
        return wrapper

    for cls, orig in distinct_attn_cls.items():
        cls.__call__ = attn_wrap(orig)
    linear_cls.__call__ = linear_wrap

    # Bump hit counts.
    for s in sites:
        store.bump_hit(s.chunk_id)

    def cleanup() -> None:
        for cls, orig in distinct_attn_cls.items():
            cls.__call__ = orig
        linear_cls.__call__ = orig_linear_call

    return cleanup
