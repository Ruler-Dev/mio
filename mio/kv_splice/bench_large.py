"""Larger-context Phase 4 bench.

Pads prompts to ~4K tokens and varies chunk position. Measures prefill
savings (not wall-clock, which is dominated by decode).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path


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


_SHIM = """
Software engineering is a broad field with many sub-disciplines. It
covers design, testing, deployment, and observability. Code quality
matters for maintenance. Readable code is easier to debug and extend.
"""


_TESTS = [
    # (desc, preceding_tokens_target, following_tokens_target)
    ("chunk_at_4K_of_15K",   4000, 11000),
    ("chunk_at_8K_of_15K",   8000,  7000),
    ("chunk_at_12K_of_15K", 12000,  3000),
]


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _pad_text(target_tokens: int, tokenizer) -> str:
    text = _SHIM
    while len(tokenizer.encode(text)) < target_tokens:
        text = text + "\n\n" + _SHIM
    while len(tokenizer.encode(text)) > target_tokens + 100:
        text = text[:int(len(text) * 0.95)]
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--out", default="experiments/kv_splice/phase4_large.json")
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
    print(f"[bench-large] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    print(f"[bench-large] loaded.", flush=True)
    tok = engine._tokenizer

    tmp = tempfile.mkdtemp(prefix="kv-splice-large-")
    store = ChunkStore(base_dir=Path(tmp))
    print(f"[bench-large] ingesting chunk ...", flush=True)
    ingest_chunk(
        engine=engine, chunk_text=_CHUNK,
        wrapper_prefix="Here is a Python file:\n", store=store,
    )
    model_id = f"{tc.name}|{tc.target_model}"

    results = []
    for (desc, pre_tokens, post_tokens) in _TESTS:
        pre = _pad_text(pre_tokens, tok)
        post = _pad_text(post_tokens, tok)
        question = "\n\nSummarize the above imports in one sentence."
        full_text = pre + _CHUNK + post + question
        messages = [{"role": "user", "content": full_text}]
        tokens = engine._apply_chat_template(messages)
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        sites = detect_chunks_text(
            rendered, tok, store, model_id=model_id, min_chunk_len=32,
        )

        print(f"\n[bench-large] === {desc} (tokens={len(tokens)}) ===", flush=True)
        print(f"  detected {len(sites)} site(s): "
              f"{[(s.start, s.end) for s in sites]}", flush=True)

        # Warmup on first prompt only.
        if desc == _TESTS[0][0]:
            engine._prefix_cache_invalidate()
            engine.generate(messages=messages, max_tokens=args.gen_tokens)

        # Fresh.
        engine._prefix_cache_invalidate()
        t0 = time.perf_counter()
        text_fresh, m_fresh = engine.generate(
            messages=messages, max_tokens=args.gen_tokens,
        )
        fresh_wall = time.perf_counter() - t0
        fresh_prefill_ms = m_fresh.prompt_tokens / max(m_fresh.prompt_tps, 1e-9) * 1000
        fresh_sha = _sha(text_fresh)
        print(f"  fresh:  prefill={fresh_prefill_ms:.0f}ms gen={m_fresh.generation_tps:.1f}t/s "
              f"sha={fresh_sha} wall={fresh_wall*1000:.0f}ms", flush=True)

        if not sites:
            continue

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
        prefill_delta = (fresh_prefill_ms - splice_prefill_ms) / fresh_prefill_ms * 100
        print(f"  splice: prefill={splice_prefill_ms:.0f}ms gen={m_splice.generation_tps:.1f}t/s "
              f"sha={splice_sha} wall={splice_wall*1000:.0f}ms", flush=True)
        print(f"  quality: lcp={lcp_frac:.3f}  sha_match={splice_sha == fresh_sha}", flush=True)
        print(f"  prefill speedup: {prefill_delta:+.1f}%", flush=True)

        results.append({
            "desc": desc, "tokens": len(tokens),
            "site_start": sites[0].start, "site_end": sites[0].end,
            "fresh_prefill_ms": fresh_prefill_ms,
            "splice_prefill_ms": splice_prefill_ms,
            "prefill_delta_pct": prefill_delta,
            "lcp_fraction": lcp_frac,
            "sha_match": splice_sha == fresh_sha,
            "fresh_gen_tps": m_fresh.generation_tps,
            "splice_gen_tps": m_splice.generation_tps,
        })

    print(f"\n[bench-large] === SUMMARY ===", flush=True)
    print(f"  {'desc':>15}  {'toks':>5}  {'site':>12}  {'fresh_ms':>8}  {'splice_ms':>9}  "
          f"{'delta':>7}  {'lcp':>6}")
    for r in results:
        print(f"  {r['desc']:>15}  {r['tokens']:>5}  "
              f"{str(r['site_start']) + '-' + str(r['site_end']):>12}  "
              f"{r['fresh_prefill_ms']:>8.0f}  {r['splice_prefill_ms']:>9.0f}  "
              f"{r['prefill_delta_pct']:>+6.1f}%  {r['lcp_fraction']:>6.2f}",
              flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[bench-large] wrote {args.out}")


if __name__ == "__main__":
    main()
