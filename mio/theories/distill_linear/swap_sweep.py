"""Sweep K — how many top-K layers can we swap before quality breaks?

For K = 1, 2, 3, 5, 8, 13, swap the top-K layers (by delta R²) with
their learned W and measure:
  - Output sha / lcp across 4 prompts
  - Prefill delta

Finds the compositional budget where the linear-replacement pattern
stays quality-neutral.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import mlx.core as mx

from mio.theories.a_layer_skip.ablation_multiprompt import (
    _PROMPTS, _pad_to, _sha, _lcp, _measure,
)
from mio.theories.distill_linear.swap_ablation import _fit_delta_W, _install_swap


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--harvest", default="experiments/distill_linear/harvest")
    p.add_argument("--regression",
                   default="experiments/distill_linear/regression.json")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--K-values", nargs="+", type=int,
                   default=[1, 2, 3, 5, 8, 13])
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--out",
                   default="experiments/distill_linear/swap_sweep.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    reg = json.loads(Path(args.regression).read_text())
    ranked = sorted(reg["layer_results"], key=lambda r: -r["delta_r2_test"])

    # Fit W for all layers we'll need (union of top K values).
    max_K = max(args.K_values)
    harvest_dir = Path(args.harvest)
    all_W: dict[int, mx.array] = {}
    print(f"[sweep] fitting W for top-{max_K} layers ...", flush=True)
    for r in ranked[:max_K]:
        idx = r["layer_idx"]
        X = np.load(harvest_dir / f"layer{idx}_X.npy")
        Y = np.load(harvest_dir / f"layer{idx}_Y.npy")
        W_np = _fit_delta_W(X, Y, ridge=reg.get("ridge", 1e-3))
        W_mx = mx.array(W_np, dtype=mx.float16)
        mx.eval(W_mx)
        all_W[idx] = W_mx
        del X, Y, W_np
        print(f"  L{idx:2d} fit", flush=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    print(f"\n[sweep] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()

    prompts = [
        (pid, _pad_to(q, args.ctx, engine._tokenizer), gtok)
        for (pid, q, gtok) in _PROMPTS
    ]
    # warmup
    _measure(engine, [{"role": "user", "content": prompts[0][1]}], prompts[0][2])

    # Baselines
    print(f"\n[sweep] === BASELINE ===", flush=True)
    baselines = {}
    for (pid, pprompt, gtok) in prompts:
        out = _measure(engine, [{"role": "user", "content": pprompt}], gtok)
        out["sha"] = _sha(out["text"])
        baselines[pid] = out
        print(f"  {pid}: sha={out['sha']} prefill={out['prefill_ms']:.0f}ms "
              f"gen={out['gen_tps']:.1f}t/s", flush=True)

    all_results: dict[int, dict] = {}
    for K in sorted(args.K_values):
        top_K_layers = [r["layer_idx"] for r in ranked[:K]]
        swap_map = {idx: all_W[idx] for idx in top_K_layers}
        print(f"\n[sweep] === K={K} layers swapped: {top_K_layers} ===", flush=True)
        cleanup = _install_swap(engine._target_model, swap_map)
        try:
            results = []
            for (pid, pprompt, gtok) in prompts:
                out = _measure(engine, [{"role": "user", "content": pprompt}], gtok)
                sha = _sha(out["text"])
                lcp = _lcp(out["text"][:800], baselines[pid]["text"][:800])
                match = sha == baselines[pid]["sha"]
                delta = out["prefill_ms"] - baselines[pid]["prefill_ms"]
                results.append({
                    "prompt_id": pid, "prefill_ms": out["prefill_ms"],
                    "gen_tps": out["gen_tps"], "accept": out["accept"],
                    "sha": sha, "lcp": lcp, "delta_prefill_ms": delta,
                    "text_head": out["text"][:300],
                })
                print(f"  {pid:>12s}  prefill={out['prefill_ms']:6.0f}ms delta={delta:+5.0f}  "
                      f"gen={out['gen_tps']:5.1f}t/s  lcp={lcp:3d}/800  "
                      f"{'MATCH' if match else 'diff '}",
                      flush=True)
            matches = sum(1 for r in results if r["sha"] == baselines[r["prompt_id"]]["sha"])
            avg_lcp = sum(r["lcp"] for r in results) / max(len(results), 1) / 800.0
            avg_delta = sum(r["delta_prefill_ms"] for r in results) / max(len(results), 1)
            all_results[K] = {
                "swapped_layers": top_K_layers,
                "matches": matches,
                "avg_lcp": avg_lcp,
                "avg_prefill_delta_ms": avg_delta,
                "per_prompt": results,
            }
            print(f"  --> K={K}: matches={matches}/4  lcp={avg_lcp:.2f}  "
                  f"prefill_delta={avg_delta:+.0f}ms",
                  flush=True)
        finally:
            cleanup()

    # Summary
    print(f"\n[sweep] === K-SWEEP SUMMARY ===", flush=True)
    print(f"  {'K':>3}  {'matches':>7}  {'avg_lcp':>7}  {'prefill_delta':>13}")
    for K in sorted(all_results.keys()):
        r = all_results[K]
        print(f"  {K:>3}  {r['matches']}/4        {r['avg_lcp']:>7.2f}  "
              f"{r['avg_prefill_delta_ms']:>+13.0f}ms", flush=True)

    Path(args.out).write_text(json.dumps({
        "baselines": {k: {kk: (vv[:200] if kk == "text" else vv)
                          for kk, vv in v.items()}
                      for k, v in baselines.items()},
        "by_K": {str(K): v for K, v in all_results.items()},
    }, indent=2))
    print(f"[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
