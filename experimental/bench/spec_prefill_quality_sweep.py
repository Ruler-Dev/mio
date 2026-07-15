"""Quality + speed sweep for SpecPrefill across prompts and keep ratios.

For each prompt × keep_ratio:
  - Run dense baseline → record output text + prefill time.
  - Run SpecPrefill → record output text + prefill time.
  - Compute prefix-match length and exact-match flag against dense.
  - Aggregate per keep_ratio: mean speedup, fraction of identical outputs,
    median prefix-match ratio.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experimental.spec_prefill.session import SpecPrefillSession


# Diverse prompts: code, math, prose, factual Q&A, multi-turn-style.
SYS = "You are a helpful coding assistant. Be concise and correct. " * 30

PROMPTS = [
    SYS + "\n\nUser: Write a Python function `is_prime(n)` that returns True iff n is prime.\nAssistant:",
    SYS + "\n\nUser: What is 17 * 23?\nAssistant:",
    SYS + "\n\nUser: Explain time complexity of binary search.\nAssistant:",
    SYS + "\n\nUser: List the first 5 prime numbers.\nAssistant:",
    SYS + "\n\nUser: Translate 'Hello, world' to French.\nAssistant:",
    SYS + "\n\nUser: What is the capital of Australia?\nAssistant:",
    SYS + "\n\nUser: Write the JavaScript reverse-string one-liner.\nAssistant:",
    SYS + "\n\nUser: Briefly describe how garbage collection works.\nAssistant:",
]


def dense_generate(model, tok, prompt: str, max_new: int) -> tuple[str, float, float]:
    """Return (text, prefill_ms, decode_ms)."""
    ids = tok.encode(prompt)
    arr = mx.array(ids, dtype=mx.uint32)[None]
    t0 = time.perf_counter()
    logits = model(arr)
    mx.eval(logits)
    next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    prefill_ms = (time.perf_counter() - t0) * 1000

    cache = None
    from mlx_lm.models import cache as cm

    inner = model.model
    cache = [cm.KVCache() for _ in range(len(inner.layers))]
    # Re-prefill using the engine's standard path so cache is consistent for decode.
    _ = model(arr, cache=cache)
    mx.eval(_)
    next_tok = int(mx.argmax(_[:, -1, :], axis=-1).item())
    generated = [next_tok]

    eos = getattr(tok, "eos_token_id", None)
    stop_set = set([int(eos)]) if eos is not None else set()

    t1 = time.perf_counter()
    for _step in range(1, max_new):
        if next_tok in stop_set:
            break
        x = mx.array([[next_tok]], dtype=mx.uint32)
        logits = model(x, cache=cache)
        mx.eval(logits)
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(next_tok)
    decode_ms = (time.perf_counter() - t1) * 1000

    return tok.decode(generated), prefill_ms, decode_ms


def prefix_match_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    target_path = "/Users/ruler/Documents/mio/models/Qwen3-8B-4bit"
    print(f"Loading {target_path}...", flush=True)
    model, tok = load(target_path)
    mx.eval(model.parameters())
    print("Loaded.\n", flush=True)

    # Warmup
    _ = dense_generate(model, tok, "Hello world", 4)

    keep_ratios = [0.10, 0.15, 0.20, 0.30, 0.50]
    max_new = 32
    early_exit = 4

    # Dense reference per prompt (used for SPEED comparison)
    dense_results = []
    for i, p in enumerate(PROMPTS):
        text, pre_ms, dec_ms = dense_generate(model, tok, p, max_new)
        dense_results.append(
            {
                "text": text,
                "prefill_ms": pre_ms,
                "decode_ms": dec_ms,
                "prompt_len": len(tok.encode(p)),
            }
        )
        print(f"[dense {i}] plen={dense_results[-1]['prompt_len']}  prefill={pre_ms:.0f}ms  text={text[:60]!r}")

    # Sparse@100% reference per prompt (used for QUALITY comparison — same forward
    # path as SpecPrefill, just no token dropping. Isolates bf16 RoPE drift from
    # the actual selection effect.)
    sparse100 = SpecPrefillSession(
        target_model=model,
        target_tokenizer=tok,
        speculator_model=model,
        keep_ratio=0.999,
        chunk_size=8,
        score_early_exit=early_exit,
        always_keep_first=4,
        always_keep_last=4,
    )
    sparse100._extra_keep = True  # marker
    sparse_results = []
    _ = sparse100.generate("hi", 4)
    for i, p in enumerate(PROMPTS):
        r = sparse100.generate(p, max_new_tokens=max_new)
        sparse_results.append({"text": r.text, "prompt_len": dense_results[i]["prompt_len"]})
        match = sum(1 for a, b in zip(r.text, dense_results[i]["text"]) if a == b)
        print(f"[sparse100 {i}] dense_match={match}/{len(dense_results[i]['text'])}")

    # Sweep keep ratios
    rows = []
    for keep in keep_ratios:
        session = SpecPrefillSession(
            target_model=model,
            target_tokenizer=tok,
            speculator_model=model,
            keep_ratio=keep,
            chunk_size=8,
            score_early_exit=early_exit,
        )
        # Warmup at this ratio
        _ = session.generate("Hello world", max_new_tokens=4)

        prefill_speedups = []
        prefix_dense = []
        prefix_s100 = []
        id_dense = 0
        id_s100 = 0
        for i, p in enumerate(PROMPTS):
            r = session.generate(p, max_new_tokens=max_new)
            d = dense_results[i]
            s = sparse_results[i]
            speedup = d["prefill_ms"] / max(r.prefill_ms, 1e-6)
            pm_d = prefix_match_len(r.text, d["text"]) / max(len(d["text"]), 1)
            pm_s = prefix_match_len(r.text, s["text"]) / max(len(s["text"]), 1)
            prefill_speedups.append(speedup)
            prefix_dense.append(pm_d)
            prefix_s100.append(pm_s)
            if r.text == d["text"]:
                id_dense += 1
            if r.text == s["text"]:
                id_s100 += 1
            rows.append(
                {
                    "keep": keep,
                    "prompt": i,
                    "speedup": speedup,
                    "pm_d": pm_d,
                    "pm_s": pm_s,
                    "id_dense": int(r.text == d["text"]),
                    "id_s100": int(r.text == s["text"]),
                    "selected": r.selected_tokens,
                    "plen": d["prompt_len"],
                }
            )
        mean_speedup = sum(prefill_speedups) / len(prefill_speedups)
        mean_pm_d = sum(prefix_dense) / len(prefix_dense)
        mean_pm_s = sum(prefix_s100) / len(prefix_s100)
        print(
            f"\n[keep={keep:.0%}]  mean_speedup={mean_speedup:.2f}×  "
            f"vs_dense_match={mean_pm_d:.0%} ({id_dense}/8 id)  "
            f"vs_sparse100_match={mean_pm_s:.0%} ({id_s100}/8 id)"
        )

    print("\n=== Detail ===")
    print(
        f"{'keep':>6s} {'prompt':>6s} {'plen':>5s} {'sel':>4s} {'speedup':>8s} {'pm_d':>6s} {'pm_s':>6s} {'idd':>3s} {'ids':>3s}"
    )
    for r in rows:
        print(
            f"{r['keep']:>6.0%} {r['prompt']:>6d} {r['plen']:>5d} {r['selected']:>4d} "
            f"{r['speedup']:>8.2f}× {r['pm_d'] * 100:>5.0f}% {r['pm_s'] * 100:>5.0f}% {r['id_dense']:>3d} {r['id_s100']:>3d}"
        )


if __name__ == "__main__":
    main()
