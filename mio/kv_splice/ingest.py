"""Chunk ingestion — capture K_base + V for a token sequence, write to store.

Runs a prefill where the target token sequence is the chunk itself
(wrapped with the chat template prefix). Hooks the attention layers in
layer_set to capture their k_proj and v_proj output at the chunk positions.
Writes to the ChunkStore.

In practice you'd ingest a chunk ONCE per mio installation — the stored
KV is reused across thousands of subsequent prompts.
"""

from __future__ import annotations

from typing import Any, Iterable

import mlx.core as mx
import numpy as np

from mio.kv_splice import SPLICEABLE_LAYERS_QWEN36_A3B
from mio.kv_splice.store import ChunkStore


def _capture_chunk_kv(
    target_model: Any,
    chunk_start: int,
    chunk_end: int,
    layer_indices: Iterable[int],
) -> tuple[Any, dict[int, dict[str, mx.array]]]:
    """Hook specified attention layers to capture k_base + v at chunk positions."""
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    attn_instances = {
        i: text.layers[i].self_attn for i in layer_indices
        if i < len(text.layers) and not bool(getattr(text.layers[i], "is_linear", False))
    }
    id_to_layer_idx = {id(attn): i for i, attn in attn_instances.items()}
    storage: dict[int, dict[str, mx.array]] = {
        i: {"k_base": None, "v": None} for i in attn_instances
    }

    distinct_cls: dict[type, Any] = {}
    for attn in attn_instances.values():
        cls = type(attn)
        if cls not in distinct_cls:
            distinct_cls[cls] = cls.__call__

    def _wrap(original_call):
        def wrapper(self, x, mask=None, cache=None):
            li = id_to_layer_idx.get(id(self))
            if li is not None:
                B, L, _ = x.shape
                n_kv = int(self.num_key_value_heads)
                d_h = int(self.head_dim)
                k = self.k_proj(x).reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)
                v = self.v_proj(x).reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)
                storage[li]["k_base"] = mx.array(k[0, :, chunk_start:chunk_end, :])
                storage[li]["v"] = mx.array(v[0, :, chunk_start:chunk_end, :])
                mx.eval(storage[li]["k_base"], storage[li]["v"])
            return original_call(self, x, mask=mask, cache=cache)
        return wrapper

    for cls, orig in distinct_cls.items():
        cls.__call__ = _wrap(orig)

    def cleanup() -> None:
        for cls, orig in distinct_cls.items():
            cls.__call__ = orig

    return cleanup, storage


def ingest_chunk(
    *,
    engine: Any,
    chunk_text: str,
    wrapper_prefix: str = "",
    store: ChunkStore,
    layer_set: tuple[int, ...] = SPLICEABLE_LAYERS_QWEN36_A3B,
    model_id: str | None = None,
) -> Any:
    """Add a chunk to the store. Returns the ChunkEntry.

    Args:
        engine: loaded MioEngine (we use its tokenizer + target model).
        chunk_text: the raw text of the chunk.
        wrapper_prefix: optional content placed before the chunk. The
            chunk's KV is captured after this prefix is processed, so
            the stored K_base reflects "chunk seen after wrapper". A
            longer/more-varied wrapper reduces context specificity.
        store: destination ChunkStore.
        layer_set: which attention layer indices to capture.
        model_id: tag identifying the model (so chunks aren't used
            across incompatible models).
    """
    from mio.dflash.runtime import _target_text_model, generate_dflash_once

    tokenizer = engine._tokenizer
    text = _target_text_model(engine._target_model)

    # Tokenize wrapper_prefix alone and wrapper_prefix + chunk to find
    # the chunk's token boundaries after chat-template processing.
    prefix_msg = [{"role": "user", "content": wrapper_prefix}]
    full_msg = [{"role": "user", "content": wrapper_prefix + chunk_text}]
    prefix_tokens = engine._apply_chat_template(prefix_msg)
    full_tokens = engine._apply_chat_template(full_msg)

    # Chunk start = longest common prefix between prefix_tokens and full_tokens.
    cs = 0
    while (cs < min(len(prefix_tokens), len(full_tokens))
           and prefix_tokens[cs] == full_tokens[cs]):
        cs += 1
    ce = len(full_tokens) - 8  # skip trailing template tokens
    if ce - cs < 16:
        raise ValueError(
            f"chunk too short after tokenization: cs={cs}, ce={ce}. "
            f"Consider longer chunk_text."
        )
    chunk_tokens = full_tokens[cs:ce]
    print(f"[ingest] chunk tokens: {len(chunk_tokens)} (positions {cs}..{ce})")

    # Find attention config.
    attn_layer = text.layers[layer_set[0]].self_attn
    n_kv_heads = int(attn_layer.num_key_value_heads)
    d_head = int(attn_layer.head_dim)

    # Capture.
    cleanup, storage = _capture_chunk_kv(
        engine._target_model, cs, ce, layer_set,
    )
    try:
        generate_dflash_once(
            target_model=engine._target_model,
            tokenizer=tokenizer,
            draft_model=engine._draft_model,
            prompt="", max_new_tokens=0,
            prompt_tokens_override=full_tokens,
            tq_bits=engine._resolved_tq_bits(),
            pq_bits=engine._resolved_pq_bits(),
            return_final_state=False, prefill_only=True,
        )
    finally:
        cleanup()

    if model_id is None:
        tc = engine.tier_config
        model_id = f"{tc.name}|{tc.target_model}"

    entry = store.save(
        tokens=chunk_tokens, model_id=model_id,
        layers=list(layer_set), n_kv_heads=n_kv_heads, d_head=d_head,
        kv_per_layer=storage,
        chunk_text=chunk_text,
    )
    print(f"[ingest] stored chunk_id={entry.chunk_id} at {entry.path}")
    return entry
