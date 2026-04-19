"""Quick benchmark: mio MioEngine on a single tier, 2 runs (warmup + measured)."""

from __future__ import annotations

import argparse
import sys
import time

from mio.config import MioConfig
from mio.engine import MioEngine


DEFAULT_PROMPT = (
    "The function $f$ satisfies the functional equation "
    "\\[ f(x) + f(y) = f(x + y) - xy - 1 \\] "
    "for all real numbers $x$ and $y$. If $f(1) = 1$, then find all "
    "integers $n$ such that $f(n) = n$. Enter all such integers, "
    "separated by commas.\nPlease reason step by step, and put your final "
    "answer within \\boxed{}."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="large-moe")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--stream", action="store_true", help="Benchmark generate_stream (chat/agent path)")
    parser.add_argument("--paro", action="store_true", help="Use PARO tiers")
    args = parser.parse_args()

    if args.paro:
        from mio.models.registry import PARO_TIERS
        cfg = MioConfig(tiers=dict(PARO_TIERS))
    else:
        cfg = MioConfig.default()
    tier = cfg.tiers[args.tier]
    print(f"[bench] tier={args.tier} target={tier.target_model}")
    print(f"[bench] draft={tier.draft_model}")

    eng = MioEngine(tier_config=tier)
    load_start = time.perf_counter()
    eng.load()
    print(f"[bench] load took {time.perf_counter() - load_start:.1f}s")

    messages = [{"role": "user", "content": args.prompt}]

    for i in range(args.warmup):
        print(f"[warmup {i + 1}/{args.warmup}]", flush=True)
        _text, m = eng.generate(messages, max_tokens=min(args.max_tokens, 64))
        print(
            f"  prompt={m.prompt_tokens} out={m.completion_tokens} "
            f"gen_tps={m.generation_tps:.2f} accept={m.avg_acceptance_length:.2f}"
        )

    print(f"[measured] mode={'stream' if args.stream else 'batch'}", flush=True)
    t0 = time.perf_counter()
    if args.stream:
        chunks = []
        for chunk, _metrics in eng.generate_stream(messages, max_tokens=args.max_tokens):
            chunks.append(chunk)
        text = "".join(chunks)
        m = eng.last_metrics
    else:
        text, m = eng.generate(messages, max_tokens=args.max_tokens)
    wall = time.perf_counter() - t0

    print("=" * 60)
    print(f"tier:                {args.tier}")
    print(f"prompt tokens:       {m.prompt_tokens}")
    print(f"generated tokens:    {m.completion_tokens}")
    print(f"prompt tps:          {m.prompt_tps:.2f}")
    print(f"generation tps:      {m.generation_tps:.2f}")
    print(f"end-to-end tps:      {m.end_to_end_tps:.2f}")
    print(f"acceptance ratio:    {m.acceptance_ratio:.3f}")
    print(f"avg accept length:   {m.avg_acceptance_length:.2f}")
    print(f"wall (python):       {wall:.2f}s")
    print(f"peak memory:         {m.peak_memory_gb:.2f} GB")
    print(f"fallback AR:         {m.fallback_ar}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
