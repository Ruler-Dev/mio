"""Multi-prompt, real-context layer ablation.

Same idea as ablation.py but:
  - 4 diverse prompts (coding, explanation, long-context, structured).
  - Prompts padded to ~4K tokens so prefill time is measurable.
  - Per layer, count how many prompts still produce baseline-match output.
  - Per layer, report avg prefill delta vs baseline.

Exports a "skippability score" per layer: fraction of prompts where
skipping that layer is quality-neutral (sha match or lcp >= 90%).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


@dataclass
class PromptResult:
    prompt_id: str
    ablated_layer_idx: int | None
    prompt_tokens: int
    prefill_ms: float
    gen_tps: float
    accept: float
    output_sha: str
    output_head: str
    lcp_with_baseline: int


@dataclass
class LayerScore:
    layer_idx: int
    matches: int
    near_matches: int  # lcp >= 90%
    total: int
    avg_prefill_delta_ms: float
    avg_lcp_fraction: float


_PROMPTS: list[tuple[str, str, int]] = [
    ("fib_memo",
     "Write a Python function `fib(n)` that computes the n-th Fibonacci "
     "number using memoization. Include a docstring and 2 example calls.",
     128),
    ("binsearch",
     "Write a Python function `binary_search(arr, target)` that returns "
     "the index of `target` in a sorted list `arr`, or -1 if not found. "
     "Include 3 test-case calls.",
     128),
    ("list_dedupe",
     "Write a Python function `dedupe(items)` that removes duplicates "
     "from a list while preserving order. Explain the approach briefly "
     "(1-2 lines), then the code.",
     96),
    ("class_bst",
     "Write a minimal BinarySearchTree class in Python with `insert`, "
     "`contains`, and `inorder` methods. Include a short example.",
     192),
]


_SHIM = """\
# project_utils.py — internal helpers

from __future__ import annotations
import hashlib, json, os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def touch(self) -> None:
        self.hit_count += 1


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._store: dict[str, CacheEntry] = {}
        self._order: list[str] = []

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        entry.touch()
        self._order.remove(key); self._order.append(key)
        return entry.value

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store[key].value = value
            self._order.remove(key); self._order.append(key); return
        self._store[key] = CacheEntry(key=key, value=value)
        self._order.append(key)
        while len(self._store) > self.capacity:
            oldest = self._order.pop(0); del self._store[oldest]
"""


def _pad_to(prompt: str, target_tokens: int, tokenizer) -> str:
    current = _SHIM
    while len(tokenizer.encode(current + "\n\n---\n\n" + prompt)) < target_tokens:
        current = current + "\n\n" + _SHIM
    while len(tokenizer.encode(current + "\n\n---\n\n" + prompt)) > target_tokens + 200:
        current = current[: int(len(current) * 0.95)]
    return current + "\n\n---\n\n" + prompt


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _install_skip(target_model: Any, skip_idx: int | None) -> Any:
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    layers = list(text.layers)
    attn_indices = [
        i for i, l in enumerate(layers)
        if not bool(getattr(l, "is_linear", False))
    ]
    if skip_idx is None:
        return lambda: None
    target_layer_idx = attn_indices[skip_idx]
    attn = layers[target_layer_idx].self_attn
    cls = type(attn)
    original_call = cls.__call__
    target_attn_id = id(attn)

    def skipping_call(self, x, mask=None, cache=None):
        if id(self) == target_attn_id:
            import mlx.core as mx
            return mx.zeros(x.shape, dtype=x.dtype)
        return original_call(self, x, mask=mask, cache=cache)

    cls.__call__ = skipping_call
    return lambda: setattr(cls, "__call__", original_call)


def _measure(engine: Any, messages: list[dict], gen_tokens: int) -> dict:
    engine._prefix_cache_invalidate()
    t0 = time.perf_counter()
    text, m = engine.generate(messages=messages, max_tokens=gen_tokens)
    total_s = time.perf_counter() - t0
    prefill_s = (m.prompt_tokens / m.prompt_tps) if m.prompt_tps > 0 else 0.0
    gen_s = max(1e-9, total_s - prefill_s)
    return {
        "text": text,
        "prompt_tokens": m.prompt_tokens,
        "prefill_ms": prefill_s * 1000.0,
        "gen_tps": m.completion_tokens / gen_s,
        "accept": m.avg_acceptance_length,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--out", default="experiments/a_ablation/multi_prompt.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.dflash.runtime import _target_text_model

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    print(f"[ablation] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    text = _target_text_model(engine._target_model)
    n_attn = sum(
        1 for l in text.layers
        if not bool(getattr(l, "is_linear", False))
    )
    print(f"[ablation] loaded. {n_attn} attention layers.", flush=True)

    # Build padded prompts, get baseline outputs (no ablation).
    print(f"\n[ablation] === BUILD PROMPTS @ ctx~{args.ctx} ===", flush=True)
    prompts: list[tuple[str, str, int]] = []
    for (pid, q, gtok) in _PROMPTS:
        padded = _pad_to(q, args.ctx, engine._tokenizer)
        actual = len(engine._tokenizer.encode(padded))
        prompts.append((pid, padded, gtok))
        print(f"  {pid}: {actual} tokens, gen={gtok}", flush=True)

    # Warmup on first prompt.
    print(f"\n[ablation] === WARMUP ===", flush=True)
    messages = [{"role": "user", "content": prompts[0][1]}]
    _measure(engine, messages, prompts[0][2])

    # Baselines per prompt.
    print(f"\n[ablation] === BASELINE (no skip) ===", flush=True)
    baselines: dict[str, dict] = {}
    for pid, pprompt, gtok in prompts:
        messages = [{"role": "user", "content": pprompt}]
        b = _measure(engine, messages, gtok)
        b["sha"] = _sha(b["text"])
        baselines[pid] = b
        print(
            f"  {pid}: prefill={b['prefill_ms']:.0f}ms gen={b['gen_tps']:.1f}t/s "
            f"sha={b['sha']}",
            flush=True,
        )

    # Per-layer skip sweep.
    per_layer: dict[int, list[PromptResult]] = {}
    print(f"\n[ablation] === PER-LAYER SKIP SWEEP ===", flush=True)
    for skip_idx in range(n_attn):
        cleanup = _install_skip(engine._target_model, skip_idx=skip_idx)
        try:
            for (pid, pprompt, gtok) in prompts:
                messages = [{"role": "user", "content": pprompt}]
                out = _measure(engine, messages, gtok)
                base = baselines[pid]
                r = PromptResult(
                    prompt_id=pid,
                    ablated_layer_idx=skip_idx,
                    prompt_tokens=out["prompt_tokens"],
                    prefill_ms=out["prefill_ms"],
                    gen_tps=out["gen_tps"],
                    accept=out["accept"],
                    output_sha=_sha(out["text"]),
                    output_head=out["text"][:400],
                    lcp_with_baseline=_lcp(out["text"][:400], base["text"][:400]),
                )
                per_layer.setdefault(skip_idx, []).append(r)
                match = r.output_sha == base["sha"]
                print(
                    f"  L{skip_idx:2d}|{pid:>11s}  "
                    f"prefill={r.prefill_ms:6.0f}ms  "
                    f"lcp={r.lcp_with_baseline:3d}/400  "
                    f"{'MATCH' if match else 'diff '}  "
                    f"delta={r.prefill_ms - base['prefill_ms']:+5.0f}ms",
                    flush=True,
                )
        finally:
            cleanup()

    # Score layers.
    scores: list[LayerScore] = []
    for skip_idx in range(n_attn):
        rows = per_layer.get(skip_idx, [])
        matches = sum(
            1 for r in rows if r.output_sha == baselines[r.prompt_id]["sha"]
        )
        near_matches = sum(
            1 for r in rows if r.lcp_with_baseline >= int(0.90 * 400)
        )
        avg_delta = (
            sum(
                r.prefill_ms - baselines[r.prompt_id]["prefill_ms"]
                for r in rows
            ) / max(len(rows), 1)
        )
        avg_lcp = (
            sum(r.lcp_with_baseline for r in rows) / max(len(rows), 1)
        ) / 400.0
        scores.append(LayerScore(
            layer_idx=skip_idx, matches=matches,
            near_matches=near_matches, total=len(rows),
            avg_prefill_delta_ms=avg_delta,
            avg_lcp_fraction=avg_lcp,
        ))

    scores.sort(key=lambda s: (-s.matches, -s.near_matches, s.avg_prefill_delta_ms))

    print(f"\n[ablation] === SKIPPABILITY RANK ===", flush=True)
    print(f"  layer  matches/total  near/total  avg_delta_ms  avg_lcp")
    for s in scores:
        print(
            f"  L{s.layer_idx:2d}     {s.matches}/{s.total}          "
            f"{s.near_matches}/{s.total}         {s.avg_prefill_delta_ms:+6.0f}    "
            f"{s.avg_lcp_fraction:.2f}",
            flush=True,
        )

    # Serialize
    data = {
        "n_attn_layers": n_attn,
        "ctx_target": args.ctx,
        "baselines": {pid: {k: v for k, v in b.items() if k != "text"} | {"text": b["text"][:400]} for pid, b in baselines.items()},
        "layer_scores": [asdict(s) for s in scores],
        "per_prompt_results": {
            str(k): [asdict(r) for r in v] for k, v in per_layer.items()
        },
    }
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"\n[ablation] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
