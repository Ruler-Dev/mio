"""Bench PolarQuant vs baseline through the full MioEngine (DFlash on).

Compares three conditions:
  - dflash         (plain DFlash, no cache compression)
  - dflash+pq4     (DFlash + PolarQuant 4-bit)
  - dflash+tq4     (DFlash + TurboQuant 4-bit, for comparison)

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


def bench(tier_name: str, prompt_tokens: int, max_tokens: int, mode: str) -> dict:
    cfg = MioConfig.default()
    tier = cfg.tiers[tier_name]

    # Reset both to off
    tier.tq_bits = 16
    tier.pq_bits = 16

    if mode == "pq4":
        tier.pq_bits = 4
    elif mode == "tq4":
        tier.tq_bits = 4

    eng = MioEngine(tier_config=tier)
    eng.load()

    text_prompt = tile_to_tokens(eng._tokenizer, prompt_tokens)
    messages = [{"role": "user", "content": text_prompt}]

    # Warmup
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
        "mode": mode,
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
    parser = argparse.ArgumentParser(
        description="Benchmark DFlash with/without PolarQuant/TurboQuant"
    )
    parser.add_argument("--tier", default="medium")
    parser.add_argument("--prompt-tokens", type=int, default=4000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--skip-tq", action="store_true",
                        help="Skip TurboQuant comparison")
    args = parser.parse_args()

    modes = ["baseline", "pq4"]
    if not args.skip_tq:
        modes.append("tq4")

    rows = []
    for mode in modes:
        label = f"dflash+{mode}" if mode != "baseline" else "dflash"
        print(f"\n[bench] mode={label}", flush=True)
        try:
            row = bench(args.tier, args.prompt_tokens, args.max_tokens, mode)
            rows.append(row)
            print(
                f"  prompt={row['prompt_tokens']} gen={row['completion_tokens']} "
                f"prompt_tps={row['prompt_tps']:7.1f} "
                f"gen_tps={row['generation_tps']:6.2f} e2e_tps={row['end_to_end_tps']:6.2f} "
                f"accept_ratio={row['acceptance_ratio']:.2f} "
                f"accept_len={row['avg_accept_len']:4.2f} peak={row['peak_gb']:4.1f}GB",
                flush=True,
            )
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

    print("\n" + "=" * 100)
    print(f"tier={args.tier}  prompt_tokens~={args.prompt_tokens}  max_tokens={args.max_tokens}")
    print("=" * 100)
    header = (
        f"{'mode':>16s} {'prompt_tps':>11s} {'gen_tps':>9s} {'e2e_tps':>9s} "
        f"{'accept_ratio':>13s} {'accept_len':>11s} {'peak_gb':>8s}"
    )
    print(header)
    for r in rows:
        mode_label = f"dflash+{r['mode']}" if r["mode"] != "baseline" else "dflash"
        print(
            f"{mode_label:>16s} {r['prompt_tps']:>11.1f} {r['generation_tps']:>9.2f} "
            f"{r['end_to_end_tps']:>9.2f} {r['acceptance_ratio']:>13.2f} "
            f"{r['avg_accept_len']:>11.2f} {r['peak_gb']:>8.1f}"
        )

    # Ratios against baseline dflash
    base = next((r for r in rows if r["mode"] == "baseline"), None)
    if base:
        print("\nSpeedup vs plain dflash (gen_tps):")
        for r in rows:
            if r["mode"] != "baseline":
                ratio = r["generation_tps"] / max(base["generation_tps"], 1e-9)
                mem_ratio = r["peak_gb"] / max(base["peak_gb"], 1e-9) if base["peak_gb"] else 0
                label = f"dflash+{r['mode']}"
                print(f"  {label:>16s}  gen_tps: {ratio:5.3f}x  peak_mem: {mem_ratio:5.3f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
