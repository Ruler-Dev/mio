"""Quickcheck SpecPrefill end-to-end.

Loads Qwen3-8B-4bit. Uses ITSELF as the speculator (simplest baseline; in real
usage we'd want a smaller model). Measures prefill+decode latency at several
keep ratios and compares output to a dense baseline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experimental.spec_prefill.session import SpecPrefillSession


PROMPT_SHORT = (
    "You are a helpful assistant. " * 30
    + "\n\nUser question: What is the time complexity of merge sort?\n\nAnswer:"
)

PROMPT_LONG = (
    "You are an expert software engineer. Be concise but precise. " * 80
    + "\n\nThe user asks: explain the difference between BFS and DFS in one paragraph.\n\nAnswer:"
)


def main():
    target_path = "/Users/ruler/Documents/mio/models/Qwen3-8B-4bit"
    print(f"Loading target {target_path}...", flush=True)
    model, tok = load(target_path)
    mx.eval(model.parameters())
    print("Loaded.", flush=True)

    # Dense baseline (no SpecPrefill)
    print("\n=== Dense baseline ===", flush=True)
    ids = tok.encode(PROMPT_SHORT)
    prompt_len = len(ids)
    print(f"prompt tokens: {prompt_len}")

    # Warmup
    _ = model(mx.array(ids[:32], dtype=mx.uint32)[None]); mx.eval(_)

    t0 = time.perf_counter()
    arr = mx.array(ids, dtype=mx.uint32)[None]
    logits = model(arr); mx.eval(logits)
    next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    dense_prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"dense prefill: {dense_prefill_ms:.1f}ms ({prompt_len/(dense_prefill_ms/1000):.0f} t/s)")
    print(f"first token: {tok.decode([next_tok])!r}")

    # SpecPrefill at multiple keep ratios — short prompt
    for keep in [0.20, 0.30, 0.50]:
        print(f"\n=== SpecPrefill keep={keep:.0%} (SHORT prompt) ===", flush=True)
        session = SpecPrefillSession(
            target_model=model, target_tokenizer=tok, speculator_model=model,
            keep_ratio=keep, chunk_size=8, score_early_exit=4,
        )
        _ = session.generate("hello world", max_new_tokens=4)  # warmup
        result = session.generate(PROMPT_SHORT, max_new_tokens=32, verbose=True)
        print(result.summary())
        print(f"first chars: {result.text[:100]!r}")

    # Long prompt baselines + SpecPrefill
    print(f"\n=== Dense baseline (LONG prompt) ===", flush=True)
    ids_long = tok.encode(PROMPT_LONG)
    print(f"prompt tokens: {len(ids_long)}")
    arr = mx.array(ids_long, dtype=mx.uint32)[None]
    t0 = time.perf_counter()
    logits = model(arr); mx.eval(logits)
    next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    long_dense_ms = (time.perf_counter() - t0) * 1000
    print(f"dense prefill: {long_dense_ms:.1f}ms ({len(ids_long)/(long_dense_ms/1000):.0f} t/s)")
    print(f"first token: {tok.decode([next_tok])!r}")

    for keep in [0.15, 0.25, 0.40]:
        print(f"\n=== SpecPrefill keep={keep:.0%} (LONG prompt) ===", flush=True)
        session = SpecPrefillSession(
            target_model=model, target_tokenizer=tok, speculator_model=model,
            keep_ratio=keep, chunk_size=16, score_early_exit=4,
        )
        result = session.generate(PROMPT_LONG, max_new_tokens=32, verbose=True)
        print(result.summary())
        print(f"first chars: {result.text[:100]!r}")


if __name__ == "__main__":
    main()
