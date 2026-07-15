"""DGSA session: full prefill + decode using sparse attention on Qwen3.5 hybrid_gdn.

Uses mio's existing Qwen3.5 model loaded via mlx-lm. Patches
Qwen3NextAttention to slice K/V to the kept indices during sparse prefill.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from experimental.dgsa.attention_patch import patch as dgsa_patch
from experimental.dgsa.attention_patch import unpatch as dgsa_unpatch
from experimental.dgsa.selection import (
    anchor_strided,
    attention_scored,
)
from experimental.dgsa.state import dgsa_active


@dataclass
class DGSAResult:
    text: str
    generated_tokens: list[int]
    prompt_tokens: int
    kept_tokens: int
    keep_ratio: float
    prefill_ms: float
    decode_ms: float

    def summary(self) -> str:
        return (
            f"prompt={self.prompt_tokens} kept={self.kept_tokens} "
            f"({self.keep_ratio:.0%})  "
            f"prefill={self.prefill_ms:.1f}ms  decode={self.decode_ms:.1f}ms"
        )


def _make_target_cache(model) -> list:
    """Build per-layer cache. SSM layers use ArraysCache; attention layers use KVCache."""
    inner = (
        getattr(model, "text_model", None)
        or getattr(model, "language_model", None)
        or getattr(model, "model", None)
        or model
    )
    return [ArraysCache(size=2) if layer.is_linear else KVCache() for layer in inner.layers]


def _select(strategy: str, model, ids_arr: mx.array, prompt_len: int, **kwargs) -> mx.array:
    if strategy == "anchor_strided":
        return anchor_strided(prompt_len, **kwargs)
    if strategy == "attention_scored":
        return attention_scored(model, ids_arr, **kwargs)
    raise ValueError(f"unknown strategy: {strategy}")


class DGSASession:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        strategy: str = "anchor_strided",
        keep_ratio: float = 0.30,
        keep_first: int = 4,
        keep_last: int = 64,
        stride: int = 8,
    ):
        self.model = model
        self.tok = tokenizer
        self.strategy = strategy
        self.keep_ratio = keep_ratio
        self.keep_first = keep_first
        self.keep_last = keep_last
        self.stride = stride
        # Apply the attention patch (idempotent).
        dgsa_patch()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        verbose: bool = False,
    ) -> DGSAResult:
        ids = self.tok.encode(prompt)
        prompt_len = len(ids)
        ids_arr = mx.array(ids, dtype=mx.uint32)[None]

        # ---- Select kept indices ----
        sel_kwargs: dict = {}
        if self.strategy == "anchor_strided":
            sel_kwargs = dict(
                keep_first=self.keep_first, keep_last=self.keep_last, stride=self.stride,
            )
        elif self.strategy == "attention_scored":
            sel_kwargs = dict(
                keep_ratio=self.keep_ratio, keep_first=self.keep_first,
                keep_last=self.keep_last,
            )
        keep = _select(self.strategy, self.model, ids_arr, prompt_len, **sel_kwargs)
        K = int(keep.shape[0])
        if verbose:
            print(f"[dgsa] kept {K}/{prompt_len} ({K/prompt_len:.0%})", flush=True)

        # ---- Sparse-attention prefill ----
        cache = _make_target_cache(self.model)
        t0 = time.perf_counter()
        with dgsa_active(keep):
            logits = self.model(ids_arr, cache=cache)
            mx.eval(logits)
        prefill_ms = (time.perf_counter() - t0) * 1000

        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated = [next_tok]
        eos = getattr(self.tok, "eos_token_id", None)
        stop_set = {int(eos)} if eos is not None else set()

        # ---- Decode (no DGSA — full attention over (sparse-prefill + new-decode)) ----
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if next_tok in stop_set:
                break
            x = mx.array([[next_tok]], dtype=mx.uint32)
            logits = self.model(x, cache=cache)
            mx.eval(logits)
            next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            generated.append(next_tok)
        decode_ms = (time.perf_counter() - t1) * 1000

        return DGSAResult(
            text=self.tok.decode(generated),
            generated_tokens=generated,
            prompt_tokens=prompt_len,
            kept_tokens=K,
            keep_ratio=K / max(prompt_len, 1),
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
        )

    def __del__(self):
        try:
            dgsa_unpatch()
        except Exception:
            pass
