"""Phase 2 — RoPE math verification.

Two tests:
  2a. IMPLEMENTATION SANITY: for a single wrapper, capture both K_base
      (pre-RoPE via k_proj) and K_post (post-RoPE via attention forward)
      at the same chunk tokens. Verify that
          rope_fn(K_base, offset=source_pos) == K_post
      bit-exact (or within fp16 rounding). If not, RoPE function in
      the model is not what we think it is.

  2b. SPLICE VALIDITY: for two wrappers (A, B) where the same chunk
      lands at different source positions p_A, p_B:
      Take K_base from A, apply rope_fn(K_base_A, offset=p_B), compare
      to fresh K_post from B at positions p_B.. p_B+len(chunk).
      Error = norm of difference. Should equal the Phase 1 context-
      robustness error (K_base differs only because of context drift;
      rotation math itself is exact).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


_CHUNK_IMPORTS = """
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
    "readable. Readable software is maintainable. Maintainable software survives.\n\n"
    "The sections below contain a mix of topics chosen to lengthen this preamble "
    "without introducing any special structure. The extra text ensures that the "
    "chunk of interest will appear at a later position in the prompt than it did "
    "in the first wrapper.\n"
)
_SUFFIX = "\n\nAnalyze the following:\n"


def _install_dual_capture(
    target_model: Any,
    chunk_start: int,
    chunk_end: int,
    storage: dict,
) -> Any:
    """Capture both K_base (pre-RoPE) and K_post (post-RoPE) at chunk positions.

    storage layout:
      storage[layer_idx] = {"k_base": [...], "k_post": [...]}
    Each list has one entry per prefill pass (one per wrapper).
    """
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    attn_layers = [
        (i, l) for i, l in enumerate(text.layers)
        if not bool(getattr(l, "is_linear", False))
    ]
    id_to_slot: dict[int, int] = {id(l.self_attn): i for i, l in attn_layers}
    for i, _ in attn_layers:
        storage.setdefault(i, {"k_base": [], "k_post": [], "rope": None})

    distinct: dict[type, Any] = {}
    for _, l in attn_layers:
        cls = type(l.self_attn)
        if cls not in distinct:
            distinct[cls] = cls.__call__

    def _wrap(original_call):
        def wrapper(self, x, mask=None, cache=None):
            slot = id_to_slot.get(id(self), None)
            if slot is None:
                return original_call(self, x, mask=mask, cache=cache)
            # Compute K_base ourselves (k_proj output pre-RoPE).
            B, L, _ = x.shape
            k_out = self.k_proj(x)
            n_kv = int(self.num_key_value_heads)
            d_h = int(self.head_dim)
            k_base_all = k_out.reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)
            # (B, n_kv, L, d_head)
            k_base_chunk = k_base_all[:, :, chunk_start:chunk_end, :]

            # Apply RoPE to the full k_base at the correct positions — we
            # need positions [0..L), NOT chunk-local. The model's rope
            # module handles absolute-position rotation when called with
            # offset=0 on the full sequence (it internally uses per-position
            # angles).
            k_post_all = self.rope(k_base_all, offset=0)
            k_post_chunk = k_post_all[:, :, chunk_start:chunk_end, :]

            # Cast to float32 for analysis.
            kb = k_base_chunk.astype(mx.float32)
            kp = k_post_chunk.astype(mx.float32)
            mx.eval(kb, kp)
            storage[slot]["k_base"].append(np.array(kb[0], copy=True))
            storage[slot]["k_post"].append(np.array(kp[0], copy=True))
            # Keep a reference to the rope module (for later use in Phase 2a/b).
            storage[slot]["rope"] = self.rope

            return original_call(self, x, mask=mask, cache=cache)
        return wrapper

    for cls, orig in distinct.items():
        cls.__call__ = _wrap(orig)

    def cleanup() -> None:
        for cls, orig in distinct.items():
            cls.__call__ = orig

    return cleanup


def _find_chunk_positions(engine, wrapper_prefix: str, chunk_text: str) -> tuple[list[int], int]:
    """Tokenize full = wrapper + suffix + chunk; find where chunk starts."""
    prefix = wrapper_prefix + _SUFFIX
    full = prefix + chunk_text
    full_toks = engine._apply_chat_template([{"role": "user", "content": full}])
    prefix_toks = engine._apply_chat_template([{"role": "user", "content": prefix}])
    common = 0
    while (common < min(len(prefix_toks), len(full_toks))
           and prefix_toks[common] == full_toks[common]):
        common += 1
    chunk_end = len(full_toks) - 8  # skip trailing template tokens
    return full_toks, common, chunk_end


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/kv_splice/phase2_rope.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.dflash.runtime import generate_dflash_once

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    print(f"[phase2] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    print(f"[phase2] loaded.", flush=True)

    # Build both prompts and find chunk boundaries.
    full_A, start_A, end_A = _find_chunk_positions(engine, _WRAPPER_A, _CHUNK_IMPORTS)
    full_B, start_B, end_B = _find_chunk_positions(engine, _WRAPPER_B, _CHUNK_IMPORTS)
    print(f"[phase2] wrapper A: chunk at [{start_A}..{end_A}]  len={end_A - start_A}",
          flush=True)
    print(f"[phase2] wrapper B: chunk at [{start_B}..{end_B}]  len={end_B - start_B}",
          flush=True)

    # Align chunks — use the shorter length to compare apples-to-apples.
    L_chunk = min(end_A - start_A, end_B - start_B)
    end_A = start_A + L_chunk
    end_B = start_B + L_chunk
    print(f"[phase2] aligned chunk length: {L_chunk}", flush=True)

    # Capture for both wrappers.
    storage_A: dict = {}
    storage_B: dict = {}

    print(f"\n[phase2] capture A (source positions {start_A}..{end_A}) ...",
          flush=True)
    cleanup = _install_dual_capture(engine._target_model, start_A, end_A, storage_A)
    try:
        generate_dflash_once(
            target_model=engine._target_model,
            tokenizer=engine._tokenizer,
            draft_model=engine._draft_model,
            prompt="", max_new_tokens=0,
            prompt_tokens_override=full_A,
            tq_bits=engine._resolved_tq_bits(), pq_bits=engine._resolved_pq_bits(),
            return_final_state=False, prefill_only=True,
        )
    finally:
        cleanup()

    print(f"\n[phase2] capture B (target positions {start_B}..{end_B}) ...",
          flush=True)
    cleanup = _install_dual_capture(engine._target_model, start_B, end_B, storage_B)
    try:
        generate_dflash_once(
            target_model=engine._target_model,
            tokenizer=engine._tokenizer,
            draft_model=engine._draft_model,
            prompt="", max_new_tokens=0,
            prompt_tokens_override=full_B,
            tq_bits=engine._resolved_tq_bits(), pq_bits=engine._resolved_pq_bits(),
            return_final_state=False, prefill_only=True,
        )
    finally:
        cleanup()

    # ------ Test 2a: sanity. rope(k_base, offset=start_A) should == k_post at wrapper A chunk positions ------
    print(f"\n[phase2] === 2a: implementation sanity check ===", flush=True)
    report_2a: dict[int, dict] = {}
    for layer_idx in sorted(storage_A.keys()):
        slotA = storage_A[layer_idx]
        if not slotA["k_base"] or not slotA["k_post"]:
            continue
        k_base_A = slotA["k_base"][0]  # (n_kv, L_chunk, d_head) numpy float32
        k_post_A = slotA["k_post"][0]
        # Check exact match between our captured k_base -> rope'd -> k_post.
        diff = k_post_A - _apply_rope_np(
            k_base_A, offset=start_A, rope_module=slotA["rope"],
        )
        rmse = float(np.sqrt((diff ** 2).mean()))
        baseline_norm = float(np.sqrt((k_post_A ** 2).mean()))
        rel = rmse / max(baseline_norm, 1e-9)
        report_2a[layer_idx] = {"rmse": rmse, "rel_rmse": rel}
        print(f"  L{layer_idx:2d}  rmse={rmse:.5f}  rel={rel:.5f}", flush=True)

    # ------ Test 2b: splice validity ------
    # Rotate K_base from wrapper A to wrapper B's position.
    # Compare to fresh K_post from wrapper B.
    print(f"\n[phase2] === 2b: splice validity (A's K_base @ B's position vs B's fresh) ===",
          flush=True)
    report_2b: dict[int, dict] = {}
    for layer_idx in sorted(storage_A.keys()):
        slotA = storage_A[layer_idx]
        slotB = storage_B[layer_idx]
        if not slotA["k_base"] or not slotB["k_post"]:
            continue
        k_base_A = slotA["k_base"][0]
        # rotate to position start_B
        k_spliced = _apply_rope_np(k_base_A, offset=start_B, rope_module=slotA["rope"])
        k_post_B_fresh = slotB["k_post"][0]
        diff = k_post_B_fresh - k_spliced
        rmse = float(np.sqrt((diff ** 2).mean()))
        baseline_norm = float(np.sqrt((k_post_B_fresh ** 2).mean()))
        rel = rmse / max(baseline_norm, 1e-9)
        report_2b[layer_idx] = {"rmse": rmse, "rel_rmse": rel}
        print(f"  L{layer_idx:2d}  rmse={rmse:.5f}  rel={rel:.5f}", flush=True)

    # Summary
    print(f"\n[phase2] === SUMMARY ===", flush=True)
    if report_2a:
        mean_2a = np.mean([v["rel_rmse"] for v in report_2a.values()])
        print(f"  Test 2a mean relative RMSE: {mean_2a:.5f}  (<=0.001 == bit-exact)",
              flush=True)
    if report_2b:
        mean_2b = np.mean([v["rel_rmse"] for v in report_2b.values()])
        print(f"  Test 2b mean relative RMSE: {mean_2b:.5f}  "
              f"(Phase 1 predicted ~0.1-0.2 for context drift)",
              flush=True)

    Path(args.out).write_text(json.dumps({
        "wrapper_A_chunk_positions": [start_A, end_A],
        "wrapper_B_chunk_positions": [start_B, end_B],
        "chunk_length": L_chunk,
        "test_2a_sanity": report_2a,
        "test_2b_splice": report_2b,
    }, indent=2))
    print(f"\n[phase2] wrote {args.out}", flush=True)


def _apply_rope_np(k_base_np: np.ndarray, *, offset: int, rope_module: Any) -> np.ndarray:
    """Apply the model's RoPE module to numpy input at given absolute offset.

    k_base_np shape: (n_kv, L_chunk, d_head). The module takes shape
    (B, n_heads, L, d_head).
    """
    k_mx = mx.array(k_base_np[None, :, :, :], dtype=mx.float16)
    k_post = rope_module(k_mx, offset=offset)
    k_post_f32 = k_post.astype(mx.float32)
    mx.eval(k_post_f32)
    return np.array(k_post_f32[0], copy=True)


if __name__ == "__main__":
    main()
