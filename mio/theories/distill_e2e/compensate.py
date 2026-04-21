"""Compensatory fine-tuning: OLS patch at L1 + train post-attn LN on layers 2..39.

Sequence:
  1. Load model (40 layers, ~30 GDN + 10 attention, d_model=2048).
  2. Run baseline forward on 8 calibration prompts, cache pre-LM-head
     hidden state per prompt. This is the teacher target.
  3. Install OLS W patch at layer 1 (replaces layer1(x) with x + W @ x).
  4. Sanity: measure loss (MSE vs cached target). Should be non-zero.
  5. Build a trainable parameter tree containing only
     `post_attention_layernorm.weight` for layers 2..39.
  6. Fine-tune: forward through patched model, MSE loss vs cached
     target, backprop only through LN weights, Adam update.
  7. After training, run generate() on 4 held-out prompts and compare
     to pre-patch baseline (sha / lcp).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten, tree_map


# ---------------- teacher capture ----------------

def _capture_teacher_hiddens(
    engine: Any,
    prompts: list[str],
    target_tokens: int,
) -> list[dict]:
    """For each prompt, return {"tokens": [...], "final_hidden": mx.array (1, L, d)}.

    Hooks the text_model.norm (final RMS) output — that's the input to
    the LM head. MSE against this captures logit differences at a fixed
    projection scale.
    """
    from mio.dflash.runtime import _target_text_model, generate_dflash_once
    text = _target_text_model(engine._target_model)
    norm_module = text.norm
    cls = type(norm_module)
    original_call = cls.__call__

    slot: dict[str, Any] = {"last": None}

    def wrap(self, x):
        out = original_call(self, x)
        if id(self) == id(norm_module):
            slot["last"] = out
        return out

    cls.__call__ = wrap
    try:
        captured: list[dict] = []
        for i, p in enumerate(prompts):
            slot["last"] = None
            messages = [{"role": "user", "content": p}]
            tokens = engine._apply_chat_template(messages)
            generate_dflash_once(
                target_model=engine._target_model,
                tokenizer=engine._tokenizer,
                draft_model=engine._draft_model,
                prompt="",
                max_new_tokens=0,
                prompt_tokens_override=tokens,
                tq_bits=engine._resolved_tq_bits(),
                pq_bits=engine._resolved_pq_bits(),
                return_final_state=False,
                prefill_only=True,
            )
            if slot["last"] is None:
                raise RuntimeError("norm hook did not fire")
            # mx.array doesn't accept `copy=`; use a simple reassignment —
            # slot["last"] is already an mx.array from the hook return.
            # We need to break reference to the model's compute graph so
            # later patching doesn't mutate the cached value. astype() + eval()
            # materializes a fresh array.
            h = slot["last"].astype(slot["last"].dtype)
            mx.eval(h)
            captured.append({"tokens": tokens, "final_hidden": h})
            print(f"  teacher capture {i+1}/{len(prompts)}: tokens={len(tokens)}, "
                  f"hidden={h.shape}", flush=True)
    finally:
        cls.__call__ = original_call
    return captured


# ---------------- L1 OLS patch ----------------

def _install_layer1_ols_patch(target_model: Any, W: mx.array) -> Any:
    """Replace decoder layer 1's __call__ with x + W @ x."""
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    layer = text.layers[1]  # layer index 1 = L1
    cls = type(layer)
    original_call = cls.__call__
    target_id = id(layer)

    def wrap(self, x, mask=None, cache=None):
        if id(self) == target_id:
            delta = mx.matmul(x.astype(W.dtype), W).astype(x.dtype)
            return x + delta
        return original_call(self, x, mask=mask, cache=cache)

    cls.__call__ = wrap
    return lambda: setattr(cls, "__class__", cls) or setattr(cls, "__call__", original_call)


# ---------------- LN parameter collection ----------------

def _collect_ln_params(target_model: Any, layer_range: range):
    """Return dict {layer_idx: layer_obj} for post_attention_layernorm params in layer_range."""
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    pal: dict[int, Any] = {}
    for i in layer_range:
        if i >= len(text.layers):
            continue
        pal[i] = text.layers[i].post_attention_layernorm
    return pal


# ---------------- main ----------------

_PROMPTS = [
    "Explain the difference between a list and a tuple in Python in 3 bullets.",
    "Write a Python function `factorial(n)` that returns n! as an int.",
    "What's the time complexity of quicksort in the worst case?",
    "Implement a Python `Stack` class with push, pop, peek, and is_empty.",
    "Write a function that reverses a string without using slicing.",
    "Implement binary search on a sorted list.",
    "Explain dictionary comprehension with a short example.",
    "Write a function that checks if a string is a palindrome.",
]

_TEST_PROMPTS: list[tuple[str, str, int]] = [
    ("fib_memo",
     "Write a Python function `fib(n)` that computes the n-th Fibonacci "
     "number using memoization. Include a docstring and 2 example calls.",
     128),
    ("binsearch",
     "Write a Python function `binary_search(arr, target)` that returns "
     "the index of `target` in a sorted list. Include 3 test-case calls.",
     128),
    ("list_dedupe",
     "Write a Python function `dedupe(items)` that removes duplicates "
     "from a list while preserving order.",
     96),
    ("class_bst",
     "Write a minimal BinarySearchTree class in Python with `insert`, "
     "`contains`, and `inorder` methods.",
     192),
]


def _load_layer1_W(regression_path: str, harvest_dir: str) -> mx.array:
    """Fit OLS W for layer 1 from harvested data. Returns float16 mx.array."""
    harvest = Path(harvest_dir)
    X = np.load(harvest / "layer1_X.npy")
    Y = np.load(harvest / "layer1_Y.npy")
    X_f32 = X.astype(np.float32)
    delta = Y.astype(np.float32) - X_f32
    d = X.shape[1]
    reg = 1e-3 * np.eye(d, dtype=np.float32)
    W_np = np.linalg.solve(X_f32.T @ X_f32 + reg, X_f32.T @ delta)
    W = mx.array(W_np, dtype=mx.float16)
    mx.eval(W)
    print(f"[compensate] loaded OLS W for L1: shape {W.shape}", flush=True)
    return W


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--harvest", default="experiments/distill_linear/harvest")
    p.add_argument("--regression", default="experiments/distill_linear/regression.json")
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="experiments/distill_e2e/compensate_report.json")
    args = p.parse_args()

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.dflash.runtime import _target_text_model
    from mio.theories.a_layer_skip.ablation_multiprompt import _pad_to, _sha, _lcp, _measure

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    print(f"[compensate] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    tok = engine._tokenizer

    # Step 1: capture teacher final hidden per prompt (pre-patch).
    print(f"\n[compensate] === step 1: capture teacher hiddens ===", flush=True)
    capture_prompts = [_pad_to(p, args.ctx, tok) for p in _PROMPTS]
    teacher = _capture_teacher_hiddens(engine, capture_prompts, args.ctx)
    print(f"  captured {len(teacher)} teacher hiddens", flush=True)

    # Step 2: load OLS W for L1, install patch.
    print(f"\n[compensate] === step 2: install L1 OLS patch ===", flush=True)
    W = _load_layer1_W(args.regression, args.harvest)
    cleanup = _install_layer1_ols_patch(engine._target_model, W)

    # Step 3: measure initial loss on the 8 prompts.
    print(f"\n[compensate] === step 3: initial loss ===", flush=True)
    from mio.dflash.runtime import generate_dflash_once, _target_text_model

    text = _target_text_model(engine._target_model)

    # Helper: forward prompt, return final hidden.
    norm_slot = {"last": None}
    norm_module = text.norm
    norm_cls = type(norm_module)
    norm_orig = norm_cls.__call__

    def norm_wrap(self, x):
        out = norm_orig(self, x)
        if id(self) == id(norm_module):
            norm_slot["last"] = out
        return out

    norm_cls.__call__ = norm_wrap

    def forward_and_get_hidden(tokens):
        norm_slot["last"] = None
        generate_dflash_once(
            target_model=engine._target_model,
            tokenizer=tok,
            draft_model=engine._draft_model,
            prompt="",
            max_new_tokens=0,
            prompt_tokens_override=tokens,
            tq_bits=engine._resolved_tq_bits(),
            pq_bits=engine._resolved_pq_bits(),
            return_final_state=False,
            prefill_only=True,
        )
        return norm_slot["last"]

    # Compute initial MSE per prompt.
    init_mse = []
    for i, entry in enumerate(teacher):
        h_patched = forward_and_get_hidden(entry["tokens"])
        target = entry["final_hidden"]
        if h_patched.shape != target.shape:
            print(f"  shape mismatch at prompt {i}: patched={h_patched.shape} teacher={target.shape}",
                  flush=True)
            continue
        mse = float(mx.mean((h_patched.astype(mx.float32) - target.astype(mx.float32)) ** 2).item())
        init_mse.append(mse)
        print(f"  prompt {i}: init MSE = {mse:.4f}", flush=True)
    avg_init_mse = sum(init_mse) / max(len(init_mse), 1)
    print(f"  avg init MSE: {avg_init_mse:.4f}", flush=True)

    norm_cls.__call__ = norm_orig
    cleanup()

    # For now, STOP after step 3 — the training loop in MLX on a 35B
    # model with frozen-except-LN params is complex; get the measurement
    # pipeline working first and scale in the next iteration.
    #
    # The key datum: if avg_init_mse is much larger than teacher_hidden
    # per-element variance, we know the L1 patch does material damage.
    # That confirms we need compensation and establishes the starting
    # loss.

    # Measure teacher hidden variance for normalization.
    all_hidden = mx.concatenate([e["final_hidden"] for e in teacher], axis=1)
    teacher_var = float(mx.var(all_hidden.astype(mx.float32)).item())
    relative_mse = avg_init_mse / max(teacher_var, 1e-12)
    print(f"\n[compensate] teacher_hidden variance: {teacher_var:.4f}", flush=True)
    print(f"[compensate] relative MSE (init / teacher_var): {relative_mse:.4f}",
          flush=True)
    print(f"[compensate] interpretation: 1.0 means L1 patch loses all signal; "
          f"0.0 means zero damage", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "layer_patched": 1,
        "n_prompts": len(teacher),
        "init_mse_per_prompt": init_mse,
        "avg_init_mse": avg_init_mse,
        "teacher_hidden_var": teacher_var,
        "relative_mse": relative_mse,
    }, indent=2))
    print(f"\n[compensate] wrote {args.out}")


if __name__ == "__main__":
    main()
