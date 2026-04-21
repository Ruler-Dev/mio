"""Sweep cheap runtime knobs for prefill speed on large-moe.

Knobs:
  - split-full-attention chunk_size: {8 (baseline), 16, 32, 64}
  - PolarQuant: {on (4-bit, baseline), off (16-bit plain KV)}

Four cells per context; measure warm prefill (best of 2 reps after warmup).
All else identical to mio's default path: DFlash draft loaded, prefix
cache invalidated each rep.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SweepRow:
    ctx: int
    pq_bits: int
    chunk_size: int
    prefill_ms: float
    gen_tps: float
    accept: float


def _ctx_prompt(target_tokens: int, tokenizer) -> str:
    shim = (
        "# project_utils.py — internal helpers\n"
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        "\n"
        "@dataclass\nclass CacheEntry:\n    key: str\n    value: object\n"
        "    hit_count: int = 0\n\n"
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


def _measure(engine, messages, gen_tokens: int) -> tuple[float, float, float]:
    """Return (prefill_ms, gen_tps, avg_accept). Single warm run after warmup."""
    # Warmup
    engine._prefix_cache_invalidate()
    engine.generate(messages=messages, max_tokens=gen_tokens)
    # Measurement
    engine._prefix_cache_invalidate()
    t0 = time.perf_counter()
    _, m = engine.generate(messages=messages, max_tokens=gen_tokens)
    total_s = time.perf_counter() - t0
    prefill_s = (m.prompt_tokens / m.prompt_tps) if m.prompt_tps > 0 else 0.0
    gen_s = max(1e-9, total_s - prefill_s)
    return prefill_s * 1000.0, m.completion_tokens / gen_s, m.avg_acceptance_length


def _set_chunk_size(engine, chunk_size: int) -> None:
    from mio.dflash.runtime import configure_full_attention_split
    # Leave the split machinery enabled but re-configure the size.
    configure_full_attention_split(
        engine._target_model, enabled=True, chunk_size=chunk_size,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ctx", nargs="+", type=int,
        default=[2048, 4096, 8192, 16384],
    )
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument(
        "--out", default="experiments/phase0_sweep/results.json",
    )
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine

    rows: list[SweepRow] = []

    for pq_bits in (4, 16):
        print(f"\n=== PQ {pq_bits}-bit ===", flush=True)
        cfg = MioConfig.default()
        tc = cfg.tiers["large-moe"]
        tc.pq_bits = pq_bits
        engine = MioEngine(tier_config=tc)
        engine.load()

        for chunk_size in args.chunk_sizes:
            _set_chunk_size(engine, chunk_size)
            print(f"  [chunk={chunk_size}]", flush=True)
            for ctx in args.ctx:
                prompt = _ctx_prompt(ctx, engine._tokenizer)
                messages = [
                    {"role": "user", "content": f"{prompt}\n\n---\n\n{_QUESTION}"}
                ]
                pms, gtps, accept = _measure(engine, messages, args.gen_tokens)
                row = SweepRow(ctx=ctx, pq_bits=pq_bits, chunk_size=chunk_size,
                               prefill_ms=pms, gen_tps=gtps, accept=accept)
                rows.append(row)
                print(
                    f"    ctx={ctx:5d} prefill={pms:7.0f}ms "
                    f"gen={gtps:5.1f}t/s accept={accept:4.2f}",
                    flush=True,
                )
        engine.unload()
        import gc; gc.collect()

    Path(args.out).write_text(
        json.dumps({"rows": [asdict(r) for r in rows]}, indent=2)
    )
    print(f"\n[sweep] wrote {args.out}", flush=True)

    # Pretty summary: baseline = pq=4, chunk=8; show delta per cell
    baseline: dict[int, float] = {
        r.ctx: r.prefill_ms for r in rows if r.pq_bits == 4 and r.chunk_size == 8
    }
    print("\nPrefill ms (prefill column shows delta vs baseline PQ4/chunk=8):")
    print(f"{'ctx':>6} {'pq':>3} {'chunk':>5}  {'prefill':>9} {'delta':>7}  {'gen/s':>5}")
    for r in rows:
        base = baseline.get(r.ctx, 0.0)
        delta = (r.prefill_ms - base) / base if base > 0 else 0.0
        print(
            f"{r.ctx:>6} {r.pq_bits:>3} {r.chunk_size:>5}  "
            f"{r.prefill_ms:>9.0f} {delta*100:+6.1f}%  {r.gen_tps:>5.1f}"
        )


if __name__ == "__main__":
    main()
