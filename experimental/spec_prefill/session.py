"""SpecPrefill session: full TTFT-optimized inference loop on Qwen3-8B.

Pipeline:
  1. Score prompt tokens via a small speculator's attention pattern.
  2. Select top-K% (chunk-aware, with BOS/recent buffer).
  3. Sparse-prefill the target with selected tokens at their ORIGINAL positions.
  4. Decode autoregressively with a logical-position offset = original prompt len.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
from mlx_lm.models import cache as cache_mod

from experimental.spec_prefill.rope_pos import apply_rope_per_position
from experimental.spec_prefill.scoring import score_prompt_tokens, select_top_k_tokens
from experimental.spec_prefill.sparse_attention import (
    sparse_block_forward,
    sparse_model_forward,
)


@dataclass
class SpecPrefillResult:
    text: str
    generated_tokens: list[int]
    prompt_tokens: int
    selected_tokens: int
    keep_ratio: float
    prefill_ms: float
    decode_ms: float
    total_ms: float
    prefill_tps: float
    decode_tps: float

    def summary(self) -> str:
        return (
            f"prompt={self.prompt_tokens} selected={self.selected_tokens} "
            f"({self.keep_ratio:.0%})  "
            f"prefill={self.prefill_ms:.1f}ms ({self.prefill_tps:.0f} t/s)  "
            f"decode={self.decode_ms:.1f}ms ({self.decode_tps:.0f} t/s)  "
            f"total={self.total_ms:.1f}ms"
        )


class SpecPrefillSession:
    """Owns target + speculator models and runs SpecPrefill inference."""

    def __init__(
        self,
        target_model,
        target_tokenizer,
        speculator_model,
        keep_ratio: float = 0.4,
        chunk_size: int = 8,
        always_keep_first: int = 4,
        always_keep_last: int = 32,
        score_early_exit: int | None = 4,
    ):
        self.target = target_model
        self.tok = target_tokenizer
        self.speculator = speculator_model
        self.keep_ratio = float(keep_ratio)
        self.chunk_size = int(chunk_size)
        self.always_keep_first = int(always_keep_first)
        self.always_keep_last = int(always_keep_last)
        self.score_early_exit = score_early_exit

    def _make_target_cache(self) -> list:
        inner = self.target.model
        return [cache_mod.KVCache() for _ in range(len(inner.layers))]

    def _decode_step(
        self, last_tok: int, cache: list, logical_pos: int
    ) -> tuple[int, mx.array]:
        """One decode step at logical position `logical_pos`."""
        ids = mx.array([[last_tok]], dtype=mx.uint32)
        positions = mx.array([logical_pos], dtype=mx.int32)
        # No mask needed for batch=1 single token decode (causal vs cache always holds).
        inner = self.target.model
        rope_theta = float(inner.args.rope_theta)
        h = inner.embed_tokens(ids)
        for layer, layer_cache in zip(inner.layers, cache):
            h = sparse_block_forward(
                layer, h, positions=positions, mask=None, cache=layer_cache,
                rope_theta=rope_theta,
            )
        h = inner.norm(h)
        if self.target.args.tie_word_embeddings:
            logits = inner.embed_tokens.as_linear(h)
        else:
            logits = self.target.lm_head(h)
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        return next_tok, logits

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        stop_token_ids: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> SpecPrefillResult:
        """Run a single SpecPrefill generation."""
        ids = self.tok.encode(prompt)
        prompt_len = len(ids)
        ids_arr = mx.array(ids, dtype=mx.uint32)[None]
        stop_set = set(stop_token_ids or [])
        eos = getattr(self.tok, "eos_token_id", None)
        if eos is not None:
            stop_set.add(int(eos))

        # ---- Phase 1: score + select ----
        prefill_start = time.perf_counter()
        importance = score_prompt_tokens(
            self.speculator, ids_arr, early_exit_after=self.score_early_exit,
        )
        mx.eval(importance)
        keep = select_top_k_tokens(
            importance, self.keep_ratio,
            always_keep_first=self.always_keep_first,
            always_keep_last=self.always_keep_last,
            chunk_size=self.chunk_size,
        )
        K = int(keep.shape[0])
        if verbose:
            print(f"[score+select] {K}/{prompt_len} ({K/prompt_len:.0%})", flush=True)

        # ---- Phase 2: sparse prefill ----
        sel_ids = ids_arr[:, keep]
        positions = keep  # int32, original positions
        cache = self._make_target_cache()
        logits = sparse_model_forward(self.target, sel_ids, positions=positions, cache=cache)
        mx.eval(logits)
        prefill_end = time.perf_counter()
        prefill_ms = (prefill_end - prefill_start) * 1000

        # First decode token from prefill logits.
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated = [next_tok]

        # ---- Phase 3: decode ----
        decode_start = time.perf_counter()
        # Start logical positions AFTER the original prompt.
        for step in range(1, max_new_tokens):
            if next_tok in stop_set:
                break
            logical_pos = prompt_len + step - 1  # position of the token we're about to encode in cache
            next_tok, _ = self._decode_step(next_tok, cache, logical_pos)
            generated.append(next_tok)
        decode_end = time.perf_counter()
        decode_ms = (decode_end - decode_start) * 1000

        text = self.tok.decode(generated)
        total_ms = prefill_ms + decode_ms
        return SpecPrefillResult(
            text=text,
            generated_tokens=generated,
            prompt_tokens=prompt_len,
            selected_tokens=K,
            keep_ratio=K / max(prompt_len, 1),
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=total_ms,
            prefill_tps=prompt_len / max(prefill_ms / 1000, 1e-9),
            decode_tps=len(generated) / max(decode_ms / 1000, 1e-9),
        )
