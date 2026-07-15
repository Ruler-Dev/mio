"""Prefix-cache bench: measure TTFT improvement on repeated prompts across all tiers.

Simulates a chat session: 1 cold call + 3 warm calls sharing a long system prompt.
Reports TTFT per call and aggregate speedup.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

import mlx.core as mx

from mio.config import MioConfig
from mio.engine import MioEngine


def bench_tier(tier_name: str, sys_repeat: int = 80, max_tokens: int = 16) -> dict:
    cfg = MioConfig.default()
    eng = MioEngine(tier_config=cfg.tiers[tier_name])
    eng.load()

    sys_content = (
        "You are Mio, a fast local coding assistant. "
        "Follow the user's request precisely. Be concise. "
    ) * sys_repeat
    users = [
        "Write fib(n) in Python.",
        "Write a Rust hello world.",
        "Explain the Raft consensus algorithm.",
        "What is 2+2?",
        "Describe a sunset in one sentence.",
    ]

    # Warmup (cold, throwaway)
    warm_msg = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": "hi"},
    ]
    eng.generate(warm_msg, max_tokens=4)
    eng._prefix_cache_invalidate()

    times = []
    for i, u in enumerate(users, 1):
        msg = [{"role": "system", "content": sys_content}, {"role": "user", "content": u}]
        if hasattr(mx, "reset_peak_memory"):
            try:
                mx.reset_peak_memory()
            except Exception:
                pass
        t0 = time.perf_counter()
        _, m = eng.generate(msg, max_tokens=max_tokens)
        wall = time.perf_counter() - t0
        is_hit = i >= 3  # call 1 is cold; call 2 warms cache; calls 3+ hit
        times.append({
            "call": i,
            "wall_ms": wall * 1000,
            "prompt_tokens": m.prompt_tokens,
            "gen_tps": m.generation_tps,
            "cache_hit": is_hit,
        })

    cold = times[0]["wall_ms"]
    warm_avg = sum(t["wall_ms"] for t in times[2:]) / max(len(times[2:]), 1)
    print(f"\n[{tier_name}]  sys_tokens_approx={times[0]['prompt_tokens']}")
    for t in times:
        tag = "HIT " if t["cache_hit"] else "MISS"
        print(f"  call {t['call']} {tag} wall={t['wall_ms']:>7.1f}ms  gen_tps={t['gen_tps']:>6.1f}")
    speedup = cold / max(warm_avg, 1e-6)
    print(f"  cold={cold:.1f}ms  warm_avg={warm_avg:.1f}ms  speedup={speedup:.2f}×")

    # Cleanup
    eng.unload()
    gc.collect()
    if hasattr(mx, "clear_cache"):
        try:
            mx.clear_cache()
        except Exception:
            pass

    return {
        "tier": tier_name,
        "prompt_tokens": times[0]["prompt_tokens"],
        "cold_ms": cold,
        "warm_avg_ms": warm_avg,
        "speedup": speedup,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", default="small,medium,large,large-moe")
    parser.add_argument("--sys-repeat", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    results = []
    for t in args.tiers.split(","):
        t = t.strip()
        try:
            results.append(bench_tier(t, args.sys_repeat, args.max_tokens))
        except Exception as e:
            print(f"[{t}] FAILED: {e}", flush=True)

    print("\n" + "=" * 70)
    print(f"{'tier':>12s} {'prompt':>7s} {'cold ms':>10s} {'warm ms':>10s} {'speedup':>8s}")
    print("-" * 70)
    for r in results:
        print(f"{r['tier']:>12s} {r['prompt_tokens']:>7d} "
              f"{r['cold_ms']:>10.1f} {r['warm_avg_ms']:>10.1f} {r['speedup']:>7.2f}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
