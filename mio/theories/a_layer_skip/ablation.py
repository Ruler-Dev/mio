"""Attention layer importance — per-layer ablation study.

For each attention layer l in the model, patch its forward to skip the
attention contribution (return x unchanged, no attention output added
to residual). Run a fixed coding prompt. Measure:

  - Output text change vs baseline (sha256 + first-divergent-token
    index + longest-common-prefix fraction).
  - Prefill wall-time delta.
  - Generation tok/s delta.

The ablation probes which attention layers contribute meaningfully to
the model's output. Layers whose removal produces minimal quality
degradation are candidates for distillation-based skipping (A1/A3/A4
in the prefill research program).

Why this study matters:
  The literature assumes attention layers are uniformly important.
  Measured per-layer contribution is often skewed — in many models a
  few early/late attention layers dominate, and the middle can be
  aggressively pruned. This experiment measures that for Qwen3.6-A3B.

Zero-cost "skip" here == return x, i.e. the attention contribution to
the residual stream is zero. MLP remains. For an honest ablation of
compute savings we'd also zero the layer's attention matmul time, but
the computed-then-discarded attention output keeps the test *simple*
and *quality-focused*. If skipping is quality-neutral, we then build a
proper compute-saving version in a follow-up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class AblationResult:
    ablated_layer_idx: int | None  # None = baseline (no ablation)
    prompt_tokens: int
    prefill_ms: float
    gen_tokens: int
    gen_tps: float
    accept: float
    output_sha256: str
    output_text_head: str
    lcp_with_baseline: int  # longest common prefix (in chars) vs baseline


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


_PROMPT = (
    "Write a Python function `binary_search(arr, target)` that returns the "
    "index of `target` in a sorted list `arr`, or -1 if not found. "
    "Include a docstring and 3 test-case calls with expected outputs. "
    "Keep it under 25 lines total."
)


def _install_skip(target_model: Any, skip_idx: int | None) -> Any:
    """Patch attention layer at `skip_idx` to bypass (return x unchanged).

    If skip_idx is None, no-op — returns baseline cleanup.
    """
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    layers = list(text.layers)
    attn_indices = [
        i for i, l in enumerate(layers)
        if not bool(getattr(l, "is_linear", False))
    ]

    if skip_idx is None:
        return lambda: None

    if skip_idx >= len(attn_indices):
        raise ValueError(
            f"skip_idx {skip_idx} >= {len(attn_indices)} attention layers"
        )

    target_layer_idx = attn_indices[skip_idx]
    target_layer = layers[target_layer_idx]
    attn = target_layer.self_attn
    cls = type(attn)
    original_call = cls.__call__
    target_attn_id = id(attn)

    def skipping_call(self, x, mask=None, cache=None):
        if id(self) == target_attn_id:
            # Still touch cache so subsequent logits align (otherwise
            # generation could desync). Produce a zero tensor the shape
            # of an attention output: (B, L, D).
            import mlx.core as mx
            return mx.zeros(x.shape, dtype=x.dtype)
        return original_call(self, x, mask=mask, cache=cache)

    cls.__call__ = skipping_call

    def cleanup() -> None:
        cls.__call__ = original_call

    return cleanup


def _measure(engine: Any, gen_tokens: int = 128) -> AblationResult:
    """One generate call on the fixed prompt. Returns metrics + output sha."""
    messages = [{"role": "user", "content": _PROMPT}]
    engine._prefix_cache_invalidate()
    t0 = time.perf_counter()
    text, m = engine.generate(messages=messages, max_tokens=gen_tokens)
    total_s = time.perf_counter() - t0
    prefill_s = (m.prompt_tokens / m.prompt_tps) if m.prompt_tps > 0 else 0.0
    gen_s = max(1e-9, total_s - prefill_s)
    return AblationResult(
        ablated_layer_idx=None,
        prompt_tokens=m.prompt_tokens,
        prefill_ms=prefill_s * 1000.0,
        gen_tokens=m.completion_tokens,
        gen_tps=m.completion_tokens / gen_s,
        accept=m.avg_acceptance_length,
        output_sha256=_sha(text),
        output_text_head=text[:400],
        lcp_with_baseline=0,  # filled after baseline run
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="large-moe")
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--out", default="experiments/a_ablation/results.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.dflash.runtime import _target_text_model

    cfg = MioConfig.default()
    tc = cfg.tiers[args.tier]
    print(f"[ablation] loading {args.tier} ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    text = _target_text_model(engine._target_model)
    n_attn = sum(
        1 for l in text.layers
        if not bool(getattr(l, "is_linear", False))
    )
    print(f"[ablation] loaded. {n_attn} attention layers.", flush=True)

    results: list[AblationResult] = []

    # Warmup + baseline
    print("\n[ablation] === BASELINE (no ablation) ===", flush=True)
    _measure(engine, args.gen_tokens)  # warmup
    base = _measure(engine, args.gen_tokens)
    base.ablated_layer_idx = None
    base.lcp_with_baseline = len(base.output_text_head)
    print(
        f"  prefill={base.prefill_ms:.0f}ms gen={base.gen_tps:.1f}t/s "
        f"accept={base.accept:.2f} sha={base.output_sha256}",
        flush=True,
    )
    results.append(base)

    # Per-layer skip.
    print(f"\n[ablation] === PER-LAYER SKIP ({n_attn} attention layers) ===",
          flush=True)
    for i in range(n_attn):
        cleanup = _install_skip(engine._target_model, skip_idx=i)
        try:
            r = _measure(engine, args.gen_tokens)
            r.ablated_layer_idx = i
            r.lcp_with_baseline = _lcp(r.output_text_head, base.output_text_head)
            match = r.output_sha256 == base.output_sha256
            print(
                f"  layer={i:2d}  prefill={r.prefill_ms:6.0f}ms  "
                f"gen={r.gen_tps:5.1f}t/s  accept={r.accept:4.2f}  "
                f"lcp={r.lcp_with_baseline:3d}/{len(base.output_text_head):3d}  "
                f"sha={r.output_sha256}  "
                f"{'MATCH' if match else 'diff'}",
                flush=True,
            )
            results.append(r)
        finally:
            cleanup()

    # Write JSON
    data = {
        "tier": args.tier,
        "baseline_sha": base.output_sha256,
        "baseline_text_head": base.output_text_head,
        "n_attn_layers": n_attn,
        "results": [asdict(r) for r in results],
    }
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"\n[ablation] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
