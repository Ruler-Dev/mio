"""GDN (GatedDeltaNet) layer ablation.

Same methodology as attention ablation, but targets `linear_attn`
instead of `self_attn`. Zero its output (residual passthrough).
Measures quality drop on 4 prompts. With 48 GDN layers, we sample
every 6th one (layers 0, 6, 12, ..., 42) to keep runtime bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from mio.theories.a_layer_skip.ablation_multiprompt import (
    _PROMPTS, _SHIM, _pad_to, _sha, _lcp, _measure, PromptResult, LayerScore,
)


def _install_gdn_skip(target_model: Any, skip_idx: int | None) -> Any:
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    layers = list(text.layers)
    gdn_indices = [
        i for i, l in enumerate(layers)
        if bool(getattr(l, "is_linear", False))
    ]
    if skip_idx is None:
        return lambda: None
    target_layer_idx = gdn_indices[skip_idx]
    gdn = layers[target_layer_idx].linear_attn
    cls = type(gdn)
    original_call = cls.__call__
    target_id = id(gdn)

    def skipping_call(self, x, mask=None, cache=None):
        if id(self) == target_id:
            import mlx.core as mx
            return mx.zeros(x.shape, dtype=x.dtype)
        return original_call(self, x, mask=mask, cache=cache)

    cls.__call__ = skipping_call
    return lambda: setattr(cls, "__call__", original_call)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--step", type=int, default=6,
                   help="Sample every N-th GDN layer")
    p.add_argument("--out", default="experiments/a_ablation/gdn_multi.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.dflash.runtime import _target_text_model

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    engine = MioEngine(tier_config=tc)
    engine.load()
    text = _target_text_model(engine._target_model)
    n_gdn = sum(1 for l in text.layers if bool(getattr(l, "is_linear", False)))
    print(f"[gdn-ablation] loaded. {n_gdn} GDN layers. Sampling every {args.step}th.",
          flush=True)

    sampled_gdn = list(range(0, n_gdn, args.step))

    prompts: list[tuple[str, str, int]] = []
    for (pid, q, gtok) in _PROMPTS:
        padded = _pad_to(q, args.ctx, engine._tokenizer)
        prompts.append((pid, padded, gtok))

    messages0 = [{"role": "user", "content": prompts[0][1]}]
    _measure(engine, messages0, prompts[0][2])

    baselines: dict[str, dict] = {}
    for pid, pprompt, gtok in prompts:
        messages = [{"role": "user", "content": pprompt}]
        b = _measure(engine, messages, gtok)
        b["sha"] = _sha(b["text"])
        baselines[pid] = b
        print(f"  baseline {pid}: sha={b['sha']} prefill={b['prefill_ms']:.0f}ms",
              flush=True)

    per_layer: dict[int, list] = {}
    for skip_idx in sampled_gdn:
        cleanup = _install_gdn_skip(engine._target_model, skip_idx=skip_idx)
        try:
            for (pid, pprompt, gtok) in prompts:
                messages = [{"role": "user", "content": pprompt}]
                out = _measure(engine, messages, gtok)
                base = baselines[pid]
                match = _sha(out["text"]) == base["sha"]
                lcp = _lcp(out["text"][:400], base["text"][:400])
                r = {
                    "prompt_id": pid, "skip_idx": skip_idx,
                    "prompt_tokens": out["prompt_tokens"],
                    "prefill_ms": out["prefill_ms"], "gen_tps": out["gen_tps"],
                    "accept": out["accept"], "sha": _sha(out["text"]),
                    "lcp": lcp,
                    "text_head": out["text"][:400],
                    "delta_ms": out["prefill_ms"] - base["prefill_ms"],
                }
                per_layer.setdefault(skip_idx, []).append(r)
                print(
                    f"  gdn={skip_idx:2d}|{pid:>11s}  "
                    f"prefill={r['prefill_ms']:6.0f}ms  "
                    f"delta={r['delta_ms']:+5.0f}  "
                    f"lcp={lcp:3d}/400  "
                    f"{'MATCH' if match else 'diff '}",
                    flush=True,
                )
        finally:
            cleanup()

    # Score.
    print(f"\n[gdn-ablation] === SKIPPABILITY RANK (sampled) ===", flush=True)
    scored = []
    for skip_idx in sampled_gdn:
        rows = per_layer.get(skip_idx, [])
        matches = sum(1 for r in rows if r["sha"] == baselines[r["prompt_id"]]["sha"])
        near = sum(1 for r in rows if r["lcp"] >= int(0.90 * 400))
        avg_lcp = (sum(r["lcp"] for r in rows) / max(len(rows), 1)) / 400.0
        avg_delta = sum(r["delta_ms"] for r in rows) / max(len(rows), 1)
        scored.append({
            "gdn_idx": skip_idx, "matches": matches, "near": near,
            "total": len(rows), "avg_lcp": avg_lcp, "avg_delta_ms": avg_delta,
        })
    scored.sort(key=lambda s: (-s["matches"], -s["near"], s["avg_delta_ms"]))
    for s in scored:
        print(
            f"  gdn={s['gdn_idx']:2d}  "
            f"matches={s['matches']}/{s['total']}  "
            f"near={s['near']}/{s['total']}  "
            f"avg_delta={s['avg_delta_ms']:+6.0f}ms  "
            f"avg_lcp={s['avg_lcp']:.2f}",
            flush=True,
        )

    Path(args.out).write_text(json.dumps({
        "n_gdn_layers": n_gdn,
        "sampled_gdn": sampled_gdn,
        "ctx_target": args.ctx,
        "layer_scores": scored,
        "per_layer_results": {str(k): v for k, v in per_layer.items()},
    }, indent=2))
    print(f"\n[gdn-ablation] wrote {args.out}")


if __name__ == "__main__":
    main()
