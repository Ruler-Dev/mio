"""Swap ablation: replace top-K layers with their learned W and test quality.

For each (layer, W) from the regression output, hot-patch the decoder
layer's __call__ at inference to return:

    x + W @ x  (the delta form, matching regression fit)

where W was trained to approximate `x_out - x_in`.

Quality test mirrors the A-ablation study: 4 diverse coding prompts, run
generate(), compare sha/lcp to baseline. Measure prefill delta too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import mlx.core as mx

from mio.theories.a_layer_skip.ablation_multiprompt import (
    _PROMPTS, _pad_to, _sha, _lcp, _measure,
)


def _fit_delta_W(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Fit W such that (Y - X) ≈ X @ W. Returns W ∈ (d, d)."""
    X = X.astype(np.float32)
    delta = Y.astype(np.float32) - X
    d = X.shape[1]
    reg = ridge * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(X.T @ X + reg, X.T @ delta)
    return W


def _install_swap(target_model: Any, swap_map: dict[int, mx.array]) -> Any:
    """Replace each layer in swap_map with `x + W @ x` as its __call__.

    swap_map: {layer_idx: W} where W is an mx.array of shape (d, d).
    Because mlx_lm's residual pattern is `y = x + block(LN(x))`, we
    approximate the ENTIRE block output by `W @ x` (not LN(x)) since
    that's what our regression fit against. The residual + will be
    added by the layer class's own `__call__` contract — NO: we return
    the full y, not just the block contribution.

    Implementation: if `layer(x)` in the baseline returns `y = x + block(...)`,
    we return `y = x + (W @ x)` which matches the `delta ≈ X @ W` fit.
    """
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    layers = list(text.layers)
    id_to_W: dict[int, mx.array] = {
        id(layers[idx]): W for idx, W in swap_map.items()
    }
    distinct: dict[type, Any] = {}
    for l in layers:
        cls = type(l)
        if cls not in distinct:
            distinct[cls] = cls.__call__

    def _make_wrapper(original_call):
        def wrapper(self, x, mask=None, cache=None):
            W = id_to_W.get(id(self))
            if W is not None:
                # delta = x @ W, output = x + delta.
                # x shape: (1, L, d). W shape: (d, d).
                # Cast to float16 to match the original dtype.
                delta = mx.matmul(x.astype(W.dtype), W).astype(x.dtype)
                return x + delta
            return original_call(self, x, mask=mask, cache=cache)
        return wrapper

    for cls, orig in distinct.items():
        cls.__call__ = _make_wrapper(orig)

    def cleanup() -> None:
        for cls, orig in distinct.items():
            cls.__call__ = orig

    return cleanup


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--harvest", default="experiments/distill_linear/harvest")
    p.add_argument("--regression",
                   default="experiments/distill_linear/regression.json")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--r2-threshold", type=float, default=0.9)
    p.add_argument("--K", type=int, default=0,
                   help="Override: take top-K layers by delta R² regardless of threshold")
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--out", default="experiments/distill_linear/swap_ablation.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    reg = json.loads(Path(args.regression).read_text())
    layer_results = reg["layer_results"]
    # Select layers.
    if args.K > 0:
        ranked = sorted(layer_results, key=lambda r: -r["delta_r2_test"])
        selected = ranked[:args.K]
    else:
        selected = [r for r in layer_results if r["delta_r2_test"] >= args.r2_threshold]
        selected.sort(key=lambda r: -r["delta_r2_test"])
    layer_ids = [r["layer_idx"] for r in selected]
    print(f"[swap] selecting {len(selected)} layers (r2>={args.r2_threshold} or top-K={args.K}):",
          flush=True)
    for r in selected:
        print(f"  L{r['layer_idx']:2d}  kind={'GDN' if r['is_linear'] else 'attn'}  "
              f"r2={r['delta_r2_test']:.3f}",
              flush=True)

    # Fit W for each selected layer.
    harvest_dir = Path(args.harvest)
    print(f"\n[swap] fitting W for each selected layer ...", flush=True)
    swap_W: dict[int, mx.array] = {}
    t_fit = time.perf_counter()
    for r in selected:
        idx = r["layer_idx"]
        X = np.load(harvest_dir / f"layer{idx}_X.npy")
        Y = np.load(harvest_dir / f"layer{idx}_Y.npy")
        W_np = _fit_delta_W(X, Y, ridge=reg.get("ridge", 1e-3))
        # Cast to fp16 for matmul speed, same as activations.
        W_mx = mx.array(W_np, dtype=mx.float16)
        mx.eval(W_mx)
        swap_W[idx] = W_mx
        del X, Y, W_np
    print(f"  fit complete: {len(swap_W)} layers in {time.perf_counter() - t_fit:.1f}s",
          flush=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    print(f"\n[swap] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()

    # Build prompts.
    prompts_with_gen: list[tuple[str, str, int]] = []
    for (pid, q, gtok) in _PROMPTS:
        padded = _pad_to(q, args.ctx, engine._tokenizer)
        prompts_with_gen.append((pid, padded, gtok))
    # Warmup
    m0 = [{"role": "user", "content": prompts_with_gen[0][1]}]
    _measure(engine, m0, prompts_with_gen[0][2])

    # Baseline
    print(f"\n[swap] === BASELINE ===", flush=True)
    baselines: dict[str, dict] = {}
    for (pid, pprompt, gtok) in prompts_with_gen:
        out = _measure(engine, [{"role": "user", "content": pprompt}], gtok)
        out["sha"] = _sha(out["text"])
        baselines[pid] = out
        print(f"  {pid}: sha={out['sha']} prefill={out['prefill_ms']:.0f}ms "
              f"gen={out['gen_tps']:.1f}t/s",
              flush=True)

    # Install swap, run test
    print(f"\n[swap] === WITH {len(swap_W)} LAYERS SWAPPED (r2>={args.r2_threshold}) ===",
          flush=True)
    cleanup = _install_swap(engine._target_model, swap_W)
    try:
        results = []
        for (pid, pprompt, gtok) in prompts_with_gen:
            out = _measure(engine, [{"role": "user", "content": pprompt}], gtok)
            sha = _sha(out["text"])
            lcp = _lcp(out["text"][:800], baselines[pid]["text"][:800])
            match = sha == baselines[pid]["sha"]
            results.append({
                "prompt_id": pid,
                "prefill_ms": out["prefill_ms"],
                "gen_tps": out["gen_tps"],
                "accept": out["accept"],
                "sha": sha,
                "lcp": lcp,
                "text_head": out["text"][:400],
                "delta_prefill_ms": out["prefill_ms"] - baselines[pid]["prefill_ms"],
            })
            print(f"  {pid:>12s}  prefill={out['prefill_ms']:7.0f}ms "
                  f"delta={out['prefill_ms']-baselines[pid]['prefill_ms']:+5.0f}  "
                  f"gen={out['gen_tps']:5.1f}t/s  "
                  f"lcp={lcp:3d}/800  "
                  f"{'MATCH' if match else 'diff '}",
                  flush=True)
    finally:
        cleanup()

    # Summary
    matches = sum(1 for r in results if r["sha"] == baselines[r["prompt_id"]]["sha"])
    avg_lcp = sum(r["lcp"] for r in results) / max(len(results), 1) / 800.0
    avg_delta = sum(r["delta_prefill_ms"] for r in results) / max(len(results), 1)
    print(f"\n[swap] === SUMMARY ===", flush=True)
    print(f"  matches: {matches}/{len(results)}  avg_lcp={avg_lcp:.2f}  "
          f"avg_prefill_delta={avg_delta:+.0f}ms",
          flush=True)

    Path(args.out).write_text(json.dumps({
        "r2_threshold": args.r2_threshold,
        "selected_layers": [
            {"idx": r["layer_idx"], "kind": "GDN" if r["is_linear"] else "attn",
             "r2": r["delta_r2_test"]}
            for r in selected
        ],
        "baselines": {k: {kk: (vv[:400] if kk == "text" else vv) for kk, vv in v.items()}
                      for k, v in baselines.items()},
        "swap_results": results,
        "matches": matches,
        "avg_lcp": avg_lcp,
        "avg_prefill_delta_ms": avg_delta,
    }, indent=2))
    print(f"[swap] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
