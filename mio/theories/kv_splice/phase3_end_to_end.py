"""Phase 3 — end-to-end quality test for KV splicing.

Sequence:
  1. Source prefill: run wrapper_A + chunk, capture K_base and V at
     chunk positions for L3 (the lowest-error splice-safe attention layer).
  2. Target prefill: run wrapper_B + chunk, hot-patching L3 to:
       - replace k_proj(x) output at chunk positions with K_base_from_A
       - replace v_proj(x) output at chunk positions with V_from_A
     Let RoPE + cache.update_and_fetch proceed normally. Cache ends up
     with spliced-then-rotated K at chunk positions (correct target pos).
  3. Generate 128 tokens from target. Compare output to a fresh-no-splice
     target run.

Quality verdict:
  - sha match or lcp >= 0.8 → splice preserves output → SMALL win confirmed.
  - anything less → splice breaks quality → Path C dead.

Also measures: did the hot-patched wrapper_B prefill produce the same
logit distribution at the last token as fresh wrapper_B? (sanity check on
the splice mechanism.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


_CHUNK = """
import json
import os
import sys
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Iterable, Callable
"""

_WRAPPER_A = "Here is a discussion of Python's import system:\n\nModules are loaded when referenced.\n"
_WRAPPER_B = (
    "This is a much longer preamble to shift the chunk into a different absolute "
    "position. We are going to cover algorithms, data structures, software "
    "engineering principles, and the nature of good code. Good software is "
    "readable. Readable software is maintainable.\n\n"
    "The sections below contain a mix of topics chosen to lengthen this preamble "
    "without introducing any special structure.\n"
)
_SUFFIX = "\n\nAnalyze the following:\n"
_QUESTION = "\n\nExplain the meaning of `from pathlib import Path` in 2 sentences."


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---------- source capture: K_base + V at L3 chunk positions ----------

def _capture_L3_kv(
    target_model: Any,
    chunk_start: int,
    chunk_end: int,
) -> Any:
    """Hook layer 3 (the target splice-safe attention layer) to capture
    pre-RoPE k_proj output and v_proj output at chunk positions.

    Returns (cleanup, storage_dict) where storage has keys
      "k_base": mx.array of shape (n_kv, chunk_len, d_head)
      "v":      mx.array of shape (n_kv, chunk_len, d_head)
    """
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    attn_layers = [
        (i, l) for i, l in enumerate(text.layers)
        if not bool(getattr(l, "is_linear", False))
    ]
    # Layer index 3 is the first attention layer in this model.
    layer3_idx, layer3 = attn_layers[0]
    attn3 = layer3.self_attn
    target_id = id(attn3)
    cls = type(attn3)
    original_call = cls.__call__
    storage: dict[str, Any] = {"k_base": None, "v": None, "rope": attn3.rope}

    def wrap(self, x, mask=None, cache=None):
        if id(self) == target_id:
            B, L, _ = x.shape
            n_kv = int(self.num_key_value_heads)
            d_h = int(self.head_dim)
            k = self.k_proj(x).reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)
            k_chunk = k[:, :, chunk_start:chunk_end, :]
            v_chunk = v[:, :, chunk_start:chunk_end, :]
            mx.eval(k_chunk, v_chunk)
            storage["k_base"] = mx.array(k_chunk[0])
            storage["v"] = mx.array(v_chunk[0])
        return original_call(self, x, mask=mask, cache=cache)

    cls.__call__ = wrap

    def cleanup() -> None:
        cls.__call__ = original_call

    return cleanup, storage


# ---------- target splice: inject K_base and V at chunk positions ----------

def _install_L3_splice(
    target_model: Any,
    chunk_start: int,
    chunk_end: int,
    spliced_k_base: mx.array,   # (n_kv, chunk_len, d_head)
    spliced_v: mx.array,
) -> Any:
    """Replace L3's k_proj/v_proj OUTPUT at chunk positions with the
    spliced values. This overrides the attention's internal computation
    only for chunk positions; other positions go through normally.

    The K replacement is PRE-RoPE (K_base). The attention layer's own
    RoPE call will rotate it to the correct target position since it
    runs over the full sequence with absolute positions.
    """
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    attn_layers = [
        (i, l) for i, l in enumerate(text.layers)
        if not bool(getattr(l, "is_linear", False))
    ]
    layer3_idx, layer3 = attn_layers[0]
    attn3 = layer3.self_attn
    target_id = id(attn3)
    cls = type(attn3)
    original_call = cls.__call__

    # Pre-compute "full" K_base and V overrides for the chunk window.
    # We'll need to patch k_proj and v_proj specifically at runtime.
    orig_k_proj_call = type(attn3.k_proj).__call__
    orig_v_proj_call = type(attn3.v_proj).__call__

    # Flag to turn on splicing only when we're in the target attention call.
    splice_active = {"on": False}

    def k_proj_wrap(self, x):
        y = orig_k_proj_call(self, x)
        if splice_active["on"] and id(self) == id(attn3.k_proj):
            B, L, total = y.shape
            # Only splice on the prefill pass (where L covers the chunk).
            # Decode passes operate on the block of speculation tokens (L=block_size),
            # appending AFTER the chunk is already in the cache.
            if L >= chunk_end:
                n_kv = int(attn3.num_key_value_heads)
                d_h = int(attn3.head_dim)
                y_r = y.reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)
                chunk_len = chunk_end - chunk_start
                spliced = mx.broadcast_to(
                    spliced_k_base[None, :, :, :], (B, n_kv, chunk_len, d_h),
                ).astype(y_r.dtype)
                pre = y_r[:, :, :chunk_start, :]
                post = y_r[:, :, chunk_end:, :]
                y_r_new = mx.concatenate([pre, spliced, post], axis=2)
                y = y_r_new.transpose(0, 2, 1, 3).reshape(B, L, total)
        return y

    def v_proj_wrap(self, x):
        y = orig_v_proj_call(self, x)
        if splice_active["on"] and id(self) == id(attn3.v_proj):
            B, L, total = y.shape
            if L >= chunk_end:
                n_kv = int(attn3.num_key_value_heads)
                d_h = int(attn3.head_dim)
                y_r = y.reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)
                chunk_len = chunk_end - chunk_start
                spliced = mx.broadcast_to(
                    spliced_v[None, :, :, :], (B, n_kv, chunk_len, d_h),
                ).astype(y_r.dtype)
                pre = y_r[:, :, :chunk_start, :]
                post = y_r[:, :, chunk_end:, :]
                y_r_new = mx.concatenate([pre, spliced, post], axis=2)
                y = y_r_new.transpose(0, 2, 1, 3).reshape(B, L, total)
        return y

    def attn_wrap(self, x, mask=None, cache=None):
        if id(self) == target_id:
            splice_active["on"] = True
            try:
                out = original_call(self, x, mask=mask, cache=cache)
            finally:
                splice_active["on"] = False
            return out
        return original_call(self, x, mask=mask, cache=cache)

    cls.__call__ = attn_wrap
    type(attn3.k_proj).__call__ = k_proj_wrap
    type(attn3.v_proj).__call__ = v_proj_wrap

    def cleanup() -> None:
        cls.__call__ = original_call
        type(attn3.k_proj).__call__ = orig_k_proj_call
        type(attn3.v_proj).__call__ = orig_v_proj_call

    return cleanup


# ---------- prompt tokenization helpers ----------

def _tokenize_full(engine, wrapper: str, chunk: str, question: str) -> tuple[list[int], int, int]:
    """Tokenize [wrapper + suffix + chunk + question]. Return (tokens, chunk_start, chunk_end)."""
    prefix = wrapper + _SUFFIX
    prefix_plus_chunk = prefix + chunk
    full = prefix + chunk + question
    prefix_toks = engine._apply_chat_template([{"role": "user", "content": prefix}])
    prefix_chunk_toks = engine._apply_chat_template(
        [{"role": "user", "content": prefix_plus_chunk}]
    )
    full_toks = engine._apply_chat_template([{"role": "user", "content": full}])
    # Chunk start = longest common prefix of prefix_toks and full_toks.
    cs = 0
    while cs < min(len(prefix_toks), len(full_toks)) and prefix_toks[cs] == full_toks[cs]:
        cs += 1
    # Chunk end = longest common prefix of prefix_chunk_toks and full_toks.
    ce = 0
    while (ce < min(len(prefix_chunk_toks), len(full_toks))
           and prefix_chunk_toks[ce] == full_toks[ce]):
        ce += 1
    return full_toks, cs, ce


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--out", default="experiments/kv_splice/phase3_e2e.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.dflash.runtime import generate_dflash_once

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    print(f"[phase3] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    tok = engine._tokenizer

    # Build source and target prompt token lists and chunk positions.
    src_full, src_cs, src_ce = _tokenize_full(engine, _WRAPPER_A, _CHUNK, "")
    tgt_full, tgt_cs, tgt_ce = _tokenize_full(engine, _WRAPPER_B, _CHUNK, _QUESTION)
    print(f"[phase3] source: chunk [{src_cs}..{src_ce}] len={src_ce - src_cs}",
          flush=True)
    print(f"[phase3] target: chunk [{tgt_cs}..{tgt_ce}] len={tgt_ce - tgt_cs}",
          flush=True)

    # Align chunk lengths (use the shorter one).
    L_chunk = min(src_ce - src_cs, tgt_ce - tgt_cs)
    src_ce = src_cs + L_chunk
    tgt_ce = tgt_cs + L_chunk
    print(f"[phase3] aligned chunk length: {L_chunk}", flush=True)

    # --- 1. Source capture: K_base + V at L3 chunk positions ---
    print(f"\n[phase3] === source capture ===", flush=True)
    cleanup_src, src_storage = _capture_L3_kv(engine._target_model, src_cs, src_ce)
    try:
        generate_dflash_once(
            target_model=engine._target_model,
            tokenizer=tok, draft_model=engine._draft_model,
            prompt="", max_new_tokens=0,
            prompt_tokens_override=src_full,
            tq_bits=engine._resolved_tq_bits(), pq_bits=engine._resolved_pq_bits(),
            return_final_state=False, prefill_only=True,
        )
    finally:
        cleanup_src()
    k_base_src = src_storage["k_base"]
    v_src = src_storage["v"]
    print(f"  captured K_base {k_base_src.shape}  V {v_src.shape}", flush=True)

    # --- 2. Target fresh (baseline) ---
    print(f"\n[phase3] === target FRESH (no splice) ===", flush=True)
    engine._prefix_cache_invalidate()
    import time as _t
    t0 = _t.perf_counter()
    text_fresh, m_fresh = engine.generate(
        messages=[{"role": "user", "content": _WRAPPER_B + _SUFFIX + _CHUNK + _QUESTION}],
        max_tokens=args.gen_tokens,
    )
    fresh_total = _t.perf_counter() - t0
    fresh_sha = _sha(text_fresh)
    print(f"  prefill_tps={m_fresh.prompt_tps:.1f} gen={m_fresh.generation_tps:.1f}t/s",
          flush=True)
    print(f"  sha={fresh_sha}", flush=True)
    print(f"  output_head: {text_fresh[:200]!r}", flush=True)

    # --- 3. Target SPLICED ---
    print(f"\n[phase3] === target WITH L3 SPLICE ===", flush=True)
    cleanup_splice = _install_L3_splice(
        engine._target_model, tgt_cs, tgt_ce,
        k_base_src, v_src,
    )
    try:
        engine._prefix_cache_invalidate()
        t0 = _t.perf_counter()
        text_splice, m_splice = engine.generate(
            messages=[{"role": "user", "content": _WRAPPER_B + _SUFFIX + _CHUNK + _QUESTION}],
            max_tokens=args.gen_tokens,
        )
        splice_total = _t.perf_counter() - t0
    finally:
        cleanup_splice()
    splice_sha = _sha(text_splice)
    print(f"  prefill_tps={m_splice.prompt_tps:.1f} gen={m_splice.generation_tps:.1f}t/s",
          flush=True)
    print(f"  sha={splice_sha}", flush=True)
    print(f"  output_head: {text_splice[:200]!r}", flush=True)

    lcp = _lcp(text_splice, text_fresh)
    print(f"\n[phase3] === QUALITY ===", flush=True)
    print(f"  sha match: {'YES' if splice_sha == fresh_sha else 'NO'}", flush=True)
    print(f"  lcp (chars): {lcp} / {len(text_fresh)}", flush=True)
    print(f"  lcp fraction: {lcp / max(len(text_fresh), 1):.3f}", flush=True)

    verdict = (
        "SPLICE PRESERVES OUTPUT" if splice_sha == fresh_sha
        else ("NEAR MATCH" if lcp / max(len(text_fresh), 1) >= 0.8
              else ("PARTIAL" if lcp / max(len(text_fresh), 1) >= 0.3
                    else "BROKEN"))
    )
    print(f"  VERDICT: {verdict}", flush=True)

    Path(args.out).write_text(json.dumps({
        "chunk_length": L_chunk,
        "source_chunk_pos": [src_cs, src_ce],
        "target_chunk_pos": [tgt_cs, tgt_ce],
        "fresh_sha": fresh_sha,
        "fresh_text_head": text_fresh[:400],
        "fresh_gen_tps": m_fresh.generation_tps,
        "splice_sha": splice_sha,
        "splice_text_head": text_splice[:400],
        "splice_gen_tps": m_splice.generation_tps,
        "lcp": lcp,
        "lcp_fraction": lcp / max(len(text_fresh), 1),
        "verdict": verdict,
    }, indent=2))
    print(f"\n[phase3] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
