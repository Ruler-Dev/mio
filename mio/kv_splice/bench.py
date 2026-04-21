"""End-to-end benchmark for the production kv_splice pipeline.

Sequence:
  1. Load large-moe.
  2. Ingest an "imports" chunk via ChunkStore.ingest().
  3. For each test prompt (3-5 prompts that contain the chunk at
     various positions and surrounding contexts):
     a. Run fresh prefill+generate as baseline.
     b. Install splice hooks using detect_chunks() output.
     c. Run patched prefill+generate.
     d. Compare: sha match? lcp? wall-clock delta?
  4. Report table: per-prompt fresh vs spliced timings.

Uses the ACTUAL production path — detect_chunks → install_splice_hooks →
engine.generate(). Not a scripted hook like Phase 3 was.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import mlx.core as mx


_CHUNK = """
import json
import os
import sys
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Iterable, Callable
"""


_TEST_PROMPTS = [
    # Each is (description, wrapper_prefix, question).
    # The wrapper places the chunk at a DIFFERENT absolute position than
    # the ingest-time position, which is the whole point of Path C.
    ("short_wrapper",
     "I am going to show you some Python code. Please read it carefully.",
     "\n\nExplain the role of `from pathlib import Path` in one sentence."),
    ("long_wrapper",
     ("Software engineering is a broad field covering many disciplines "
      "including program design, testing, deployment, and observability. "
      "Effective engineering requires both breadth and depth: a working "
      "understanding of many topics combined with deep expertise in a few. "
      "Consider the Python ecosystem for a moment. Python has become a "
      "dominant language for data work, web backends, and scripting because "
      "of its readability and vast standard library."),
     "\n\nWhat does `from dataclasses import dataclass` provide?"),
    ("question_before",
     "What are some useful Python standard library imports? Here are examples:",
     "\n\nList 3 of the above imports and describe each."),
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--out", default="experiments/kv_splice/phase4_bench.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.kv_splice.store import ChunkStore
    from mio.kv_splice.detect import detect_chunks_text
    from mio.kv_splice.splice import install_splice_hooks
    from mio.kv_splice.ingest import ingest_chunk

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    print(f"[bench] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    print(f"[bench] loaded.", flush=True)

    # Set up chunk store in a temp directory (clean slate for bench).
    tmp = tempfile.mkdtemp(prefix="kv-splice-bench-")
    store = ChunkStore(base_dir=Path(tmp))
    print(f"[bench] store dir: {tmp}", flush=True)

    # --- Ingest the chunk ---
    print(f"\n[bench] === ingest chunk ===", flush=True)
    ingest_chunk(
        engine=engine,
        chunk_text=_CHUNK,
        wrapper_prefix="Here is a Python file:\n",
        store=store,
    )
    print(f"[bench] store now has {len(store)} chunk(s)", flush=True)

    model_id = f"{tc.name}|{tc.target_model}"
    results: list[dict] = []

    # --- Run each test prompt fresh + spliced ---
    for (desc, wrapper, question) in _TEST_PROMPTS:
        full_text = wrapper + _CHUNK + question
        messages = [{"role": "user", "content": full_text}]
        tokens = engine._apply_chat_template(messages)
        print(f"\n[bench] === prompt '{desc}' (tokens={len(tokens)}) ===", flush=True)

        # Build the chat-template-rendered text for offset-mapping detection.
        try:
            rendered = engine._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            rendered = None
        sites = []
        if rendered is not None:
            sites = detect_chunks_text(
                rendered, engine._tokenizer, store,
                model_id=model_id, min_chunk_len=32,
            )
        print(f"  detected {len(sites)} splice site(s): "
              f"{[(s.start, s.end) for s in sites]}", flush=True)

        # Fresh baseline.
        engine._prefix_cache_invalidate()
        t0 = time.perf_counter()
        text_fresh, m_fresh = engine.generate(
            messages=messages, max_tokens=args.gen_tokens,
        )
        fresh_wall = time.perf_counter() - t0
        fresh_prefill_ms = m_fresh.prompt_tokens / max(m_fresh.prompt_tps, 1e-9) * 1000
        fresh_sha = _sha(text_fresh)
        print(f"  fresh: prefill_ms={fresh_prefill_ms:.0f} gen={m_fresh.generation_tps:.1f}t/s "
              f"sha={fresh_sha} wall={fresh_wall*1000:.0f}ms",
              flush=True)

        # Spliced run.
        if sites:
            cleanup = install_splice_hooks(engine._target_model, sites, store)
            try:
                engine._prefix_cache_invalidate()
                t0 = time.perf_counter()
                text_splice, m_splice = engine.generate(
                    messages=messages, max_tokens=args.gen_tokens,
                )
                splice_wall = time.perf_counter() - t0
            finally:
                cleanup()
            splice_prefill_ms = m_splice.prompt_tokens / max(m_splice.prompt_tps, 1e-9) * 1000
            splice_sha = _sha(text_splice)
            lcp = _lcp(text_splice, text_fresh)
            lcp_frac = lcp / max(len(text_fresh), 1)
            print(f"  splice: prefill_ms={splice_prefill_ms:.0f} gen={m_splice.generation_tps:.1f}t/s "
                  f"sha={splice_sha} wall={splice_wall*1000:.0f}ms",
                  flush=True)
            print(f"  quality: lcp={lcp}/{len(text_fresh)} ({lcp_frac:.3f}) "
                  f"sha_match={splice_sha == fresh_sha}",
                  flush=True)
            print(f"  speedup: prefill "
                  f"{(fresh_prefill_ms - splice_prefill_ms) / fresh_prefill_ms * 100:+.1f}%  "
                  f"wall {(fresh_wall - splice_wall) / fresh_wall * 100:+.1f}%",
                  flush=True)
        else:
            splice_sha = None
            splice_prefill_ms = None
            splice_wall = None
            lcp = 0
            lcp_frac = 0.0
            print(f"  splice: no sites detected, skipping", flush=True)

        results.append({
            "prompt_id": desc,
            "prompt_tokens": len(tokens),
            "fresh_prefill_ms": fresh_prefill_ms,
            "fresh_gen_tps": m_fresh.generation_tps,
            "fresh_wall_ms": fresh_wall * 1000,
            "fresh_sha": fresh_sha,
            "sites": [{"start": s.start, "end": s.end, "chunk_id": s.chunk_id}
                      for s in sites],
            "splice_prefill_ms": splice_prefill_ms,
            "splice_gen_tps": m_splice.generation_tps if sites else None,
            "splice_wall_ms": splice_wall * 1000 if sites else None,
            "splice_sha": splice_sha,
            "lcp": lcp,
            "lcp_fraction": lcp_frac,
            "sha_match": splice_sha == fresh_sha if sites else None,
        })

    # --- Summary ---
    print(f"\n[bench] === SUMMARY ===", flush=True)
    print(f"  {'prompt':>18}  {'tokens':>6}  {'fresh_pre':>9}  {'spl_pre':>8}  "
          f"{'delta':>7}  {'lcp':>6}  {'sha':>5}")
    for r in results:
        if r["splice_prefill_ms"] is not None:
            delta = (r["fresh_prefill_ms"] - r["splice_prefill_ms"]) / max(r["fresh_prefill_ms"], 1) * 100
            print(f"  {r['prompt_id']:>18}  {r['prompt_tokens']:>6}  "
                  f"{r['fresh_prefill_ms']:>9.0f}  {r['splice_prefill_ms']:>8.0f}  "
                  f"{delta:>+6.1f}%  {r['lcp_fraction']:>6.2f}  "
                  f"{'YES' if r['sha_match'] else 'NO':>5}",
                  flush=True)
        else:
            print(f"  {r['prompt_id']:>18}  {r['prompt_tokens']:>6}  {'no-site':>9}",
                  flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
