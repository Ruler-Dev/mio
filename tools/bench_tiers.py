"""Head-to-head tier comparison: end-to-end TTFT + decode tok/s.

Loads one tier per run (sequentially, not both at once — 22 GB + 2 GB
would nearly OOM on 96 GB even before Metal buffers). For each tier:
  * Prefill-only pass at N in {512, 1024, 2048, 4096, 8192}.
  * Full generate (prefill + 128 decoded tokens) for gen_tps.

Tiers: small (Qwen3.5-4B-4bit + DFlash) vs large-moe (Qwen3.6-35B-A3B
+ DFlash). DFlash is on for both (standard mio path).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TierRow:
    tier: str
    ctx: int
    prefill_ms: float
    gen_tps: float
    accept: float
    gen_tokens: int


def _ctx_prompt(target_tokens: int, tokenizer) -> str:
    shim = (
        "# project_utils.py — internal helpers\n"
        "from __future__ import annotations\n"
        "import hashlib, json, os, time\n"
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        "\n"
        "@dataclass\nclass CacheEntry:\n    key: str\n    value: object\n"
        "    created_at: float = 0.0\n    hit_count: int = 0\n\n"
        "class LRUCache:\n    def __init__(self, cap):\n        self.cap = cap\n"
        "        self._store: dict = {}\n        self._order: list = []\n"
        "    def get(self, key):\n        e = self._store.get(key)\n"
        "        if e is None: return None\n        e.hit_count += 1\n"
        "        self._order.remove(key); self._order.append(key)\n"
        "        return e.value\n"
        "    def put(self, k, v):\n"
        "        if k in self._store:\n            self._store[k].value = v\n"
        "            self._order.remove(k); self._order.append(k); return\n"
        "        self._store[k] = CacheEntry(k, v); self._order.append(k)\n"
        "        while len(self._store) > self.cap:\n"
        "            oldest = self._order.pop(0); del self._store[oldest]\n"
    )
    current = shim
    while len(tokenizer.encode(current)) < target_tokens:
        current = current + "\n\n" + shim
    while len(tokenizer.encode(current)) > target_tokens + 200:
        current = current[: int(len(current) * 0.95)]
    return current


_QUESTION = (
    "Write a Python function `fib(n)` that computes the n-th Fibonacci number "
    "using memoization. Include a short docstring and 2 example calls. "
    "Keep it under 20 lines."
)


def bench_tier(tier_name: str, ctxs: list[int], gen_tokens: int = 128) -> list[TierRow]:
    from mio.config import MioConfig
    from mio.engine import MioEngine

    cfg = MioConfig.default()
    tc = cfg.tiers[tier_name]
    print(f"\n== {tier_name} ==  ({tc.target_model})", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    print("  loaded.", flush=True)
    tok = engine._tokenizer

    rows: list[TierRow] = []
    for ctx in ctxs:
        if ctx > tc.context_window:
            print(f"  skipping ctx={ctx} (> {tc.context_window})", flush=True)
            continue
        prompt = _ctx_prompt(ctx, tok)
        messages = [
            {"role": "user", "content": f"{prompt}\n\n---\n\n{_QUESTION}"}
        ]
        # Warmup (rep 0)
        engine._prefix_cache_invalidate()
        _t, _m = engine.generate(messages=messages, max_tokens=gen_tokens)

        # Measurement (rep 1 — warm)
        engine._prefix_cache_invalidate()
        t0 = time.perf_counter()
        text, m = engine.generate(messages=messages, max_tokens=gen_tokens)
        total_s = time.perf_counter() - t0
        if m.prompt_tps > 0 and m.prompt_tokens > 0:
            prefill_s = m.prompt_tokens / m.prompt_tps
        else:
            prefill_s = 0.0
        gen_s = max(0.0, total_s - prefill_s)
        gen_tps = m.completion_tokens / gen_s if gen_s > 0 else 0.0

        row = TierRow(
            tier=tier_name,
            ctx=ctx,
            prefill_ms=prefill_s * 1000.0,
            gen_tps=gen_tps,
            accept=m.avg_acceptance_length,
            gen_tokens=m.completion_tokens,
        )
        rows.append(row)
        print(
            f"  ctx={ctx:5d}  prefill={row.prefill_ms:7.0f}ms  "
            f"gen={row.gen_tps:6.1f} tok/s  accept={row.accept:5.2f}  "
            f"out_tokens={row.gen_tokens}",
            flush=True,
        )

    engine.unload()
    import gc; gc.collect()
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tiers", nargs="+", default=["small", "large-moe"])
    p.add_argument(
        "--ctx", nargs="+", type=int,
        default=[512, 1024, 2048, 4096, 8192],
    )
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument(
        "--out", default="experiments/phase0_tier_compare/results.json",
    )
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[TierRow] = []
    for t in args.tiers:
        all_rows.extend(bench_tier(t, args.ctx, args.gen_tokens))

    data = {
        "tiers": args.tiers,
        "ctx": args.ctx,
        "gen_tokens": args.gen_tokens,
        "rows": [asdict(r) for r in all_rows],
    }
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"\n[bench] wrote {args.out}\n", flush=True)

    # Side-by-side table
    by = {(r.tier, r.ctx): r for r in all_rows}
    tiers = args.tiers
    ctxs = sorted({r.ctx for r in all_rows})
    print(f"{'ctx':>6}  " + "  ".join(f"{t:>18s}" for t in tiers))
    print("-" * (6 + 2 + sum(20 for _ in tiers)))
    for ctx in ctxs:
        cells = []
        for t in tiers:
            r = by.get((t, ctx))
            if r is None:
                cells.append("           --        ")
            else:
                cells.append(
                    f"{r.prefill_ms:5.0f}ms {r.gen_tps:5.1f}t/s"
                )
        print(f"{ctx:>6}  " + "  ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
