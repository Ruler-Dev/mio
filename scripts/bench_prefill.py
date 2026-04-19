"""Prefill-only microbench. Separates prefill tok/s from decode tok/s.

Runs per (tier, prompt_tokens) pair with adequate warmup. Reports:
  - prefill tok/s  (higher better)
  - time-to-first-token in ms
  - peak memory
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

import mlx.core as mx

from mio.config import MioConfig
from mio.dflash.runtime import load_target_bundle, load_draft_bundle, generate_dflash_once


SEED_SENTENCE = (
    "The observable universe contains approximately 2 trillion galaxies, each "
    "with hundreds of billions of stars. Dark matter and dark energy together "
    "account for 95% of the total mass-energy content. "
)


def tile_tokens(tokenizer, target: int) -> list[int]:
    ids = tokenizer.encode(SEED_SENTENCE)
    out: list[int] = []
    while len(out) < target:
        out.extend(ids)
    return out[:target]


def bench_tier(
    tier_name: str,
    target_path: str,
    draft_path: str,
    prompt_lens: list[int],
    warmup: int = 2,
    reps: int = 3,
) -> list[dict]:
    """Load a tier once, bench multiple prompt lengths."""
    print(f"\n[loading] {tier_name} target={target_path}", flush=True)
    t0 = time.perf_counter()
    target, tok, _ = load_target_bundle(target_path, lazy=True, split_full_attention_sdpa=True)
    draft, _ = load_draft_bundle(draft_path)
    mx.eval(target.parameters())
    print(f"[loaded] in {time.perf_counter() - t0:.1f}s", flush=True)
    # Each bench call has a unique prompt to defeat any upstream prefix cache.

    results: list[dict] = []
    for plen in prompt_lens:
        prompt_ids = tile_tokens(tok, plen)
        if len(prompt_ids) < plen:
            continue
        # Warmup
        for _ in range(warmup):
            generate_dflash_once(
                target_model=target, tokenizer=tok, draft_model=draft,
                prompt="", max_new_tokens=8,
                prompt_tokens_override=prompt_ids,
            )

        # Measure: multiple reps, take best
        best_prefill = float("inf")
        best_ttft = float("inf")
        best_peak = 0.0
        for _ in range(reps):
            if hasattr(mx, "reset_peak_memory"):
                try: mx.reset_peak_memory()
                except Exception: pass
            t0 = time.perf_counter()
            r = generate_dflash_once(
                target_model=target, tokenizer=tok, draft_model=draft,
                prompt="", max_new_tokens=1,  # minimal decode to isolate prefill
                prompt_tokens_override=prompt_ids,
            )
            ttft_wall = time.perf_counter() - t0
            phase = r.get("phase_timings_us") or {}
            prefill_us = phase.get("prefill", 0)
            if prefill_us > 0:
                prefill_s = prefill_us / 1e6
                best_prefill = min(best_prefill, prefill_s)
                best_ttft = min(best_ttft, ttft_wall)
                peak_gb = r.get("peak_memory_gb") or 0.0
                best_peak = max(best_peak, peak_gb)
        row = {
            "tier": tier_name,
            "prompt_tokens": plen,
            "prefill_s": best_prefill,
            "prefill_tps": plen / max(best_prefill, 1e-9),
            "ttft_ms": best_ttft * 1000,
            "peak_gb": best_peak,
        }
        results.append(row)
        print(
            f"  plen={plen:>6d}  prefill={row['prefill_tps']:>8.1f} t/s  "
            f"ttft={row['ttft_ms']:>8.1f} ms  peak={row['peak_gb']:>5.1f} GB",
            flush=True,
        )

    del target, draft
    gc.collect()
    if hasattr(mx, "clear_cache"):
        try: mx.clear_cache()
        except Exception: pass
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", default="small,medium,large,large-moe")
    parser.add_argument("--prompt-lens", default="512,1024,4096")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    cfg = MioConfig.default()
    prompt_lens = [int(x) for x in args.prompt_lens.split(",")]

    all_rows: list[dict] = []
    for t in args.tiers.split(","):
        t = t.strip()
        if t not in cfg.tiers:
            continue
        tier = cfg.tiers[t]
        all_rows.extend(bench_tier(
            t, tier.target_model, tier.draft_model,
            prompt_lens=prompt_lens, warmup=args.warmup, reps=args.reps,
        ))

    print("\n" + "=" * 70)
    print(f"{'tier':>12s} {'plen':>6s} {'prefill t/s':>14s} {'ttft ms':>10s} {'peak':>7s}")
    print("-" * 70)
    for r in all_rows:
        print(f"{r['tier']:>12s} {r['prompt_tokens']:>6d} "
              f"{r['prefill_tps']:>14.1f} {r['ttft_ms']:>10.1f} {r['peak_gb']:>5.1f}GB")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
