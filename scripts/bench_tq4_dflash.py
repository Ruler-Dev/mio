"""Bench TQ4 vs baseline *through the full MioEngine* (DFlash on).

Runs four conditions on one tier so we see the interaction:
  - baseline (no TQ, no draft)
  - tq4    (TQ4, no draft)
  - dflash (no TQ, draft on)
  - dflash+tq4  (TQ4, draft on)

Uses a real-ish tiled prompt at N tokens and a fixed max_tokens decode.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

import mlx.core as mx

from mio.config import MioConfig
from mio.engine import MioEngine


SEED_PROMPT = (
    "In distributed systems, consensus algorithms like Raft and Paxos "
    "guarantee safety under asynchronous networks with bounded failures. "
    "Cache coherence, replication factor, and leader election interact in "
    "subtle ways that determine both latency and correctness. "
)


def tile_to_tokens(tokenizer, target_tokens: int) -> str:
    seed_ids = tokenizer.encode(SEED_PROMPT)
    per = len(seed_ids)
    reps = max(1, target_tokens // per + 1)
    return SEED_PROMPT * reps


def bench(tier_name: str, prompt_tokens: int, max_tokens: int, tq4: bool, dflash: bool) -> dict:
    cfg = MioConfig.default()
    tier = cfg.tiers[tier_name]
    if tq4:
        tier.tq_bits = 4
    else:
        tier.tq_bits = 16

    eng = MioEngine(tier_config=tier)
    eng.load()

    if not dflash:
        eng._draft_model = None

    text_prompt = tile_to_tokens(eng._tokenizer, prompt_tokens)
    messages = [{"role": "user", "content": text_prompt}]

    # Warmup small
    eng.generate(messages, max_tokens=8)

    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass

    t0 = time.perf_counter()
    text, m = eng.generate(messages, max_tokens=max_tokens)
    wall = time.perf_counter() - t0

    result = {
        "tier": tier_name,
        "cond": f"{'tq4' if tq4 else 'base'}+{'dflash' if dflash else 'nodraft'}",
        "prompt_tokens": m.prompt_tokens,
        "completion_tokens": m.completion_tokens,
        "prompt_tps": m.prompt_tps,
        "generation_tps": m.generation_tps,
        "end_to_end_tps": m.end_to_end_tps,
        "acceptance_ratio": m.acceptance_ratio,
        "avg_accept_len": m.avg_acceptance_length,
        "peak_gb": m.peak_memory_gb,
        "wall_s": wall,
    }

    eng.unload()
    del eng
    gc.collect()
    if hasattr(mx, "clear_cache"):
        try:
            mx.clear_cache()
        except Exception:
            pass

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="medium")
    parser.add_argument("--prompt-tokens", type=int, default=4000)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    conditions = [
        (False, False),  # baseline nodraft
        (True, False),   # tq4 nodraft
        (False, True),   # dflash only
        (True, True),    # tq4 + dflash
    ]

    rows = []
    for tq4, dflash in conditions:
        label = f"{'tq4' if tq4 else 'base'}+{'dflash' if dflash else 'nodraft'}"
        print(f"\n[bench] condition={label}", flush=True)
        try:
            row = bench(args.tier, args.prompt_tokens, args.max_tokens, tq4, dflash)
            rows.append(row)
            print(
                f"  prompt={row['prompt_tokens']} gen={row['completion_tokens']} "
                f"prompt_tps={row['prompt_tps']:7.1f} "
                f"gen_tps={row['generation_tps']:6.2f} e2e_tps={row['end_to_end_tps']:6.2f} "
                f"accept_len={row['avg_accept_len']:4.2f} peak={row['peak_gb']:4.1f}GB",
                flush=True,
            )
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

    print("\n" + "=" * 90)
    print(f"tier={args.tier}  prompt_tokens~={args.prompt_tokens}  max_tokens={args.max_tokens}")
    print("=" * 90)
    print(f"{'cond':>18s} {'prompt_tps':>11s} {'gen_tps':>9s} {'e2e_tps':>9s} "
          f"{'accept_len':>11s} {'peak_gb':>8s}")
    for r in rows:
        print(f"{r['cond']:>18s} {r['prompt_tps']:>11.1f} {r['generation_tps']:>9.2f} "
              f"{r['end_to_end_tps']:>9.2f} {r['avg_accept_len']:>11.2f} {r['peak_gb']:>8.1f}")

    # Ratios against baseline (no tq, no dflash)
    base = next((r for r in rows if r["cond"] == "base+nodraft"), None)
    if base:
        print("\nSpeedup vs base+nodraft (end_to_end_tps):")
        for r in rows:
            if r["cond"] != "base+nodraft":
                ratio = r["end_to_end_tps"] / max(base["end_to_end_tps"], 1e-9)
                print(f"  {r['cond']:>18s}  {ratio:5.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
