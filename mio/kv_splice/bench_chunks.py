"""Chunk-size sweep bench.

Fixes context at ~8K tokens and varies chunk size from 128 → 1024 tokens.
Measures fresh vs splice prefill to find the break-even point where
saved attention compute overtakes per-layer mx.concatenate overhead.

Chunk content is realistic (simulated tool-definition JSON / doc paragraph)
and built by concatenating primitives to reach the target token count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path


_SHIM = """
Software engineering is a broad field with many sub-disciplines. It
covers design, testing, deployment, and observability. Code quality
matters for maintenance. Readable code is easier to debug and extend.
"""


# Tool-def-flavored chunk primitive — repeats to build larger chunks.
_TOOL_DEF = """
{
  "name": "search_documents",
  "description": "Search the corpus for documents matching a query. Returns a ranked list of document IDs plus a short excerpt. Supports filters on date, author, and tags.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Free-text query."},
      "top_k": {"type": "integer", "description": "Max results to return.", "default": 10},
      "filters": {
        "type": "object",
        "properties": {
          "after": {"type": "string", "format": "date"},
          "before": {"type": "string", "format": "date"},
          "tags": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    "required": ["query"]
  }
}
"""


_CHUNK_SIZES = [128, 256, 512, 1024]
_CTX_TOKENS = 8000  # realistic workload


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _build_chunk(target_tokens: int, tokenizer) -> str:
    """Build a chunk of ~target_tokens by repeating _TOOL_DEF."""
    text = _TOOL_DEF
    while len(tokenizer.encode(text)) < target_tokens:
        text = text + "\n\n" + _TOOL_DEF
    # Trim back to target.
    while len(tokenizer.encode(text)) > target_tokens + 8:
        text = text[: int(len(text) * 0.98)]
    return text


def _pad_text(target_tokens: int, tokenizer) -> str:
    text = _SHIM
    while len(tokenizer.encode(text)) < target_tokens:
        text = text + "\n\n" + _SHIM
    while len(tokenizer.encode(text)) > target_tokens + 100:
        text = text[: int(len(text) * 0.95)]
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-tokens", type=int, default=32)
    p.add_argument("--ctx", type=int, default=_CTX_TOKENS)
    p.add_argument("--out", default="experiments/kv_splice/phase4_chunks.json")
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
    print(f"[bench-chunks] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    print(f"[bench-chunks] loaded.", flush=True)
    tok = engine._tokenizer

    model_id = f"{tc.name}|{tc.target_model}"

    results = []
    for (i_size, chunk_tokens) in enumerate(_CHUNK_SIZES):
        tmp = tempfile.mkdtemp(prefix=f"kv-splice-chunks-{chunk_tokens}-")
        store = ChunkStore(base_dir=Path(tmp))
        chunk_text = _build_chunk(chunk_tokens, tok)
        actual_chunk_len = len(tok.encode(chunk_text))
        print(f"\n[bench-chunks] === chunk_target={chunk_tokens} actual={actual_chunk_len} ===", flush=True)
        print(f"[bench-chunks] ingesting ...", flush=True)
        ingest_chunk(
            engine=engine,
            chunk_text=chunk_text,
            wrapper_prefix="Here is a tool definition:\n",
            store=store,
        )

        # Build prompt with chunk at middle.
        half = (args.ctx - actual_chunk_len) // 2
        pre = _pad_text(half, tok)
        post = _pad_text(half, tok)
        question = "\n\nSummarize the above definition in one sentence."
        full_text = pre + chunk_text + post + question
        messages = [{"role": "user", "content": full_text}]
        tokens = engine._apply_chat_template(messages)
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        sites = detect_chunks_text(
            rendered, tok, store, model_id=model_id, min_chunk_len=32,
        )
        total_tokens = len(tokens)
        print(f"  total_tokens={total_tokens}  sites={[(s.start, s.end) for s in sites]}", flush=True)
        if not sites:
            print(f"  NO SITES DETECTED — skipping", flush=True)
            continue
        site = sites[0]
        site_len = site.end - site.start

        # Warmup on every size — keeps fresh/splice baseline comparable
        # across chunks (first call at a new shape incurs MLX graph
        # compilation that otherwise inflates the fresh timing).
        engine._prefix_cache_invalidate()
        engine.generate(messages=messages, max_tokens=args.gen_tokens)

        # Fresh
        engine._prefix_cache_invalidate()
        t0 = time.perf_counter()
        text_fresh, m_fresh = engine.generate(messages=messages, max_tokens=args.gen_tokens)
        fresh_wall = time.perf_counter() - t0
        fresh_prefill_ms = m_fresh.prompt_tokens / max(m_fresh.prompt_tps, 1e-9) * 1000
        fresh_sha = _sha(text_fresh)
        print(f"  fresh:  prefill={fresh_prefill_ms:.0f}ms gen={m_fresh.generation_tps:.1f}t/s "
              f"sha={fresh_sha} wall={fresh_wall*1000:.0f}ms", flush=True)

        # Splice
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
        prefill_delta_pct = (fresh_prefill_ms - splice_prefill_ms) / fresh_prefill_ms * 100
        print(f"  splice: prefill={splice_prefill_ms:.0f}ms gen={m_splice.generation_tps:.1f}t/s "
              f"sha={splice_sha} wall={splice_wall*1000:.0f}ms", flush=True)
        print(f"  quality: lcp={lcp_frac:.3f}  sha_match={splice_sha == fresh_sha}", flush=True)
        print(f"  prefill delta: {prefill_delta_pct:+.1f}%  "
              f"(chunk_frac={site_len/total_tokens:.3f})", flush=True)

        results.append({
            "chunk_target": chunk_tokens,
            "chunk_actual": actual_chunk_len,
            "total_tokens": total_tokens,
            "site_start": site.start,
            "site_end": site.end,
            "site_len": site_len,
            "chunk_fraction": site_len / total_tokens,
            "fresh_prefill_ms": fresh_prefill_ms,
            "splice_prefill_ms": splice_prefill_ms,
            "prefill_delta_pct": prefill_delta_pct,
            "lcp_fraction": lcp_frac,
            "sha_match": splice_sha == fresh_sha,
            "fresh_gen_tps": m_fresh.generation_tps,
            "splice_gen_tps": m_splice.generation_tps,
        })

    print(f"\n[bench-chunks] === SUMMARY (ctx ~{args.ctx}) ===", flush=True)
    print(f"  {'chunk':>6}  {'frac':>5}  {'fresh_ms':>8}  {'splice_ms':>9}  "
          f"{'delta':>7}  {'lcp':>6}  {'sha':>6}")
    for r in results:
        print(f"  {r['site_len']:>6}  {r['chunk_fraction']:>5.2%}  "
              f"{r['fresh_prefill_ms']:>8.0f}  {r['splice_prefill_ms']:>9.0f}  "
              f"{r['prefill_delta_pct']:>+6.1f}%  {r['lcp_fraction']:>6.2f}  "
              f"{'Y' if r['sha_match'] else 'n':>6}",
              flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[bench-chunks] wrote {args.out}")


if __name__ == "__main__":
    main()
