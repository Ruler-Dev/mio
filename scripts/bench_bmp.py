"""Bench BMP-DFlash vs vanilla DFlash on mio tiers.

Sweeps num_paths K ∈ {1, 2, 3, 4} and compares against vanilla DFlash baseline.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

import mlx.core as mx

from mio.config import MioConfig
from mio.dflash.bmp_runtime import generate_bmp_dflash_once
from mio.dflash.runtime import (
    generate_dflash_once,
    load_draft_bundle,
    load_target_bundle,
)


PROMPTS = [
    (
        "code",
        "Write a Python function `topk_softmax(logits, k)` that returns the top-k "
        "softmax probabilities and their indices using numpy. Include type hints, "
        "a docstring, and handle k > len(logits) gracefully. Keep it concise.",
    ),
    (
        "math",
        "The function f satisfies f(x) + f(y) = f(x + y) - xy - 1 for all real "
        "numbers x and y. If f(1) = 1, find all integers n such that f(n) = n. "
        "Show your work step by step.",
    ),
    (
        "prose",
        "Explain speculative decoding in three short paragraphs: the core idea, "
        "the acceptance criterion, and the main performance tradeoff.",
    ),
]


def bench(tier_name: str, max_tokens: int = 128,
          override_target: str | None = None,
          override_draft: str | None = None) -> None:
    cfg = MioConfig.default()
    if override_target:
        target_path = override_target
        draft_path = override_draft or ""
    else:
        tier = cfg.tiers[tier_name]
        target_path = tier.target_model
        draft_path = tier.draft_model

    print(f"\n[loading] tier={tier_name} target={target_path}", flush=True)
    target, tok, _ = load_target_bundle(
        target_path, lazy=True, split_full_attention_sdpa=True,
    )
    draft, _ = load_draft_bundle(draft_path)
    print("[loaded]", flush=True)

    # Warmup (small run) to amortize one-shot JIT on every cache/kernel path.
    warm_ids = tok.encode("hello world " * 20)
    generate_dflash_once(
        target_model=target, tokenizer=tok, draft_model=draft,
        prompt="", max_new_tokens=8, block_tokens=16,
        prompt_tokens_override=warm_ids,
    )
    generate_bmp_dflash_once(
        target_model=target, tokenizer=tok, draft_model=draft,
        prompt="", max_new_tokens=8, num_paths=2, block_tokens=16,
        prompt_tokens_override=warm_ids,
    )

    header = f"{'prompt':<8s} {'mode':>12s} {'gen':>5s} {'tok/s':>8s} {'tpc':>6s} {'accept':>7s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for label, prompt_text in PROMPTS:
        prompt_ids = tok.encode(prompt_text)

        # Vanilla DFlash
        if hasattr(mx, "reset_peak_memory"):
            try:
                mx.reset_peak_memory()
            except Exception:
                pass
        t0 = time.perf_counter()
        r = generate_dflash_once(
            target_model=target, tokenizer=tok, draft_model=draft,
            prompt="", max_new_tokens=max_tokens, block_tokens=16,
            prompt_tokens_override=prompt_ids,
        )
        wall = time.perf_counter() - t0
        print(
            f"{label:<8s} {'dflash':>12s} {r['generation_tokens']:>5d} "
            f"{r['generation_tokens']/wall:>8.1f} {r['tokens_per_cycle']:>6.2f} "
            f"{r['acceptance_ratio']:>7.2f}",
            flush=True,
        )

        # BMP K = 2, 3, 4
        for K in (2, 3, 4):
            if hasattr(mx, "reset_peak_memory"):
                try:
                    mx.reset_peak_memory()
                except Exception:
                    pass
            t0 = time.perf_counter()
            r = generate_bmp_dflash_once(
                target_model=target, tokenizer=tok, draft_model=draft,
                prompt="", max_new_tokens=max_tokens, num_paths=K, block_tokens=16,
                prompt_tokens_override=prompt_ids,
            )
            wall = time.perf_counter() - t0
            print(
                f"{label:<8s} {'bmp_K=' + str(K):>12s} {r['generation_tokens']:>5d} "
                f"{r['generation_tokens']/wall:>8.1f} {r['tokens_per_cycle']:>6.2f} "
                f"{r['acceptance_ratio']:>7.2f}",
                flush=True,
            )

    del target, draft
    gc.collect()
    if hasattr(mx, "clear_cache"):
        try:
            mx.clear_cache()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", default="small,medium", help="comma-separated")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--target", default=None, help="Override target model path")
    parser.add_argument("--draft", default=None, help="Override draft model path")
    parser.add_argument("--label", default="custom", help="Name for override run")
    args = parser.parse_args()

    if args.target:
        bench(
            args.label, max_tokens=args.max_tokens,
            override_target=args.target, override_draft=args.draft,
        )
    else:
        for name in [t.strip() for t in args.tiers.split(",") if t.strip()]:
            bench(name, max_tokens=args.max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
