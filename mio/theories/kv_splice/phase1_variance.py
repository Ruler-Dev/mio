"""Phase 1 — measure K_base context-robustness.

For each chunk (4 candidates) × wrapper (8 different preceding contexts):
  1. Build a prompt = wrapper_prefix + chunk + wrapper_suffix.
  2. Run prefill.
  3. Hook each attention layer's k_proj, capture its output at the
     chunk's token positions.
  4. Store per (chunk_id, wrapper_id, layer, chunk_position, head): K_base.

Then compute variance statistics per (chunk_id, layer, chunk_position, head)
across wrappers. Normalize by total K_base variance. Report per-chunk per-
layer median and worst-case ratios.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


# ---------------- chunks ----------------

_CHUNK_IMPORTS = """
import json
import os
import sys
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Iterable, Callable
from collections import defaultdict, deque, Counter
import heapq
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
"""

_CHUNK_CLASS = """
class LRUCache:
    \"\"\"Least-recently-used cache with a fixed byte budget.

    Evicts entries in insertion order when the budget is exceeded.
    Not thread-safe; callers must guard concurrent access.
    \"\"\"

    def __init__(self, capacity: int, byte_budget: int | None = None):
        self.capacity = int(capacity)
        self.byte_budget = byte_budget
        self._store: dict = {}
        self._order: list = []

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError

    def _evict(self):
        raise NotImplementedError
"""

_CHUNK_MARKDOWN = """
## Environment Configuration

The following options control runtime behavior.

- `MIO_LOG_LEVEL`: one of `debug`, `info`, `warn`, `error` (default: `info`).
- `MIO_CACHE_DIR`: path to the prefix-cache directory (default: `~/.mio/cache`).
- `MIO_MAX_WORKERS`: integer (default: `8`).

Example configuration:

```bash
export MIO_LOG_LEVEL=debug
export MIO_CACHE_DIR=/tmp/mio-cache
export MIO_MAX_WORKERS=16
```

See the reference documentation for advanced tuning flags.
"""

_CHUNK_TOOLDEF = """
{
  "name": "search_web",
  "description": "Search the web for information about a given query.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query to execute."
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of results to return (default: 10)."
      },
      "language": {
        "type": "string",
        "enum": ["en", "es", "fr", "de", "zh"],
        "description": "Preferred language for search results."
      }
    },
    "required": ["query"]
  }
}
"""

_CHUNKS = [
    ("imports", _CHUNK_IMPORTS),
    ("class_skel", _CHUNK_CLASS),
    ("markdown", _CHUNK_MARKDOWN),
    ("tooldef", _CHUNK_TOOLDEF),
]


# ---------------- wrappers (8 distinct preceding contexts) ----------------

_WRAPPERS_PREFIX = [
    "Here is a brief discussion of how Python's import system works:\n\nThe import system is responsible for loading modules when they are referenced.\n",
    "The following questions and answers cover common Python topics:\n\nQ: What is duck typing?\nA: Duck typing is a programming style where object suitability is determined by what methods and attributes it has.\n",
    "I need help with a data analysis task. I have a CSV file with the following columns: timestamp, user_id, action, duration.\n\nI want to compute per-user average session length.\n",
    "Below is the source code for a small web server written in Go. I'd like you to translate it to Python using asyncio.\n\n```go\npackage main\nimport \"net/http\"\nfunc main() { http.ListenAndServe(\":8080\", nil) }\n```\n",
    "Consider the following mathematical problem: given a connected graph G with N vertices, find the minimum spanning tree using Prim's algorithm.\n\nThe time complexity is O((V+E) log V).\n",
    "In software engineering, cohesion refers to the degree to which elements of a module belong together. High cohesion is generally preferable.\n",
    "The Raft consensus algorithm ensures agreement among distributed nodes despite failures. It divides the problem into leader election, log replication, and safety.\n",
    "Poetry can reveal truths inaccessible to prose. Consider the imagery of Eliot's Four Quartets:\n\n    Time present and time past\n    Are both perhaps present in time future\n",
]

_WRAPPERS_SUFFIX = "\n\nGiven the above context, please analyze the following section:\n"


# ---------------- capture hook ----------------

def _install_kproj_capture(
    target_model: Any,
    chunk_start_tokens: int,
    chunk_end_tokens: int,
    storage: dict,
) -> Any:
    """Hook attention layer k_proj to capture output at chunk positions.

    storage layout: storage[layer_idx] = list of (n_chunk_positions, n_kv_heads, d_head) arrays
      appended one per capture call (one per prefill pass).
    """
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    attn_layers = [
        (i, l) for i, l in enumerate(text.layers)
        if not bool(getattr(l, "is_linear", False))
    ]

    id_to_slot: dict[int, int] = {id(l.self_attn): i for i, l in attn_layers}
    for i, l in attn_layers:
        storage.setdefault(i, [])

    distinct: dict[type, Any] = {}
    for _, l in attn_layers:
        cls = type(l.self_attn)
        if cls not in distinct:
            distinct[cls] = cls.__call__

    def _wrap(original_call):
        def wrapper(self, x, mask=None, cache=None):
            slot = id_to_slot.get(id(self), None)
            if slot is not None:
                # Compute k_proj(x) ourselves (extra compute — fine for calibration).
                k_full = self.k_proj(x)  # (1, L, n_kv_heads * d_head)
                _, L, total = k_full.shape
                d_head = int(self.head_dim)
                n_kv = int(self.num_key_value_heads)
                k_reshaped = k_full.reshape(1, L, n_kv, d_head)
                # Extract chunk positions.
                chunk_k = k_reshaped[:, chunk_start_tokens:chunk_end_tokens, :, :]
                chunk_k = chunk_k.astype(mx.float32)
                mx.eval(chunk_k)
                storage[slot].append(np.array(chunk_k[0], copy=True))
            return original_call(self, x, mask=mask, cache=cache)
        return wrapper

    for cls, orig in distinct.items():
        cls.__call__ = _wrap(orig)

    def cleanup() -> None:
        for cls, orig in distinct.items():
            cls.__call__ = orig

    return cleanup


# ---------------- main ----------------

def _build_prompt(wrapper_prefix: str, chunk: str) -> str:
    return wrapper_prefix + _WRAPPERS_SUFFIX + chunk


@dataclass
class ChunkVarianceReport:
    chunk_id: str
    n_wrappers: int
    n_chunk_tokens: int
    per_layer: list[dict]  # [{layer, median_ratio, mean_ratio, worst_ratio}]
    median_across_layers: float
    mean_across_layers: float


def _analyze(
    storage: dict[int, list[np.ndarray]],
    chunk_id: str,
) -> ChunkVarianceReport:
    per_layer = []
    ratios_across_layers = []
    for layer_idx in sorted(storage.keys()):
        captures = storage[layer_idx]
        if len(captures) < 2:
            continue
        # Stack to (n_wrappers, chunk_len, n_kv, d_head)
        stacked = np.stack(captures, axis=0).astype(np.float32)
        # Variance across wrappers per (position, head, dim) — then sum over head/dim.
        var_per_pos_head_dim = np.var(stacked, axis=0)  # (chunk_len, n_kv, d_head)
        # Total variance of K at all chunk positions in a single wrapper (arbitrary ref wrapper).
        ref = stacked[0]  # (chunk_len, n_kv, d_head)
        total_var = float(np.var(ref))
        if total_var <= 0:
            continue
        ratio_per_pos = var_per_pos_head_dim.sum(axis=(1, 2)) / (np.var(ref, axis=(1, 2)).sum() + 1e-12)
        # Simpler scalar ratio: mean variance across chunk / variance of K
        scalar_ratio = float(var_per_pos_head_dim.mean() / (total_var + 1e-12))
        worst_ratio = float(ratio_per_pos.max())
        median_ratio = float(np.median(ratio_per_pos))
        per_layer.append({
            "layer": layer_idx,
            "scalar_ratio": scalar_ratio,
            "median_pos_ratio": median_ratio,
            "worst_pos_ratio": worst_ratio,
        })
        ratios_across_layers.append(scalar_ratio)
    return ChunkVarianceReport(
        chunk_id=chunk_id,
        n_wrappers=len(captures),
        n_chunk_tokens=int(stacked.shape[1]),
        per_layer=per_layer,
        median_across_layers=float(np.median(ratios_across_layers)) if ratios_across_layers else 1.0,
        mean_across_layers=float(np.mean(ratios_across_layers)) if ratios_across_layers else 1.0,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/kv_splice/phase1_variance.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.dflash.runtime import generate_dflash_once

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    print(f"[kv_splice] loading large-moe ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    tok = engine._tokenizer

    reports: dict[str, ChunkVarianceReport] = {}
    for (chunk_id, chunk_text) in _CHUNKS:
        print(f"\n[kv_splice] === chunk '{chunk_id}' ===", flush=True)
        # Encode chunk once, find its token boundaries relative to each wrapped prompt.
        # We tokenize wrapper_prefix + suffix then find where chunk starts.
        storage: dict[int, list[np.ndarray]] = {}

        for wi, wrapper in enumerate(_WRAPPERS_PREFIX):
            full = _build_prompt(wrapper, chunk_text)
            # Tokenize and find chunk boundaries by tokenizing the prefix separately.
            prefix = wrapper + _WRAPPERS_SUFFIX
            prefix_tokens = engine._apply_chat_template(
                [{"role": "user", "content": prefix}]
            )
            full_tokens = engine._apply_chat_template(
                [{"role": "user", "content": full}]
            )
            # Chunk starts where the prefix tokens end. There may be a small
            # template-adjustment at the boundary; we handle by finding the
            # common-prefix length between prefix_tokens and full_tokens.
            common = 0
            while (
                common < min(len(prefix_tokens), len(full_tokens))
                and prefix_tokens[common] == full_tokens[common]
            ):
                common += 1
            # Trim a few extra tokens to skip the "analyze the following" phrasing
            # (which is already in the template tail). The chunk itself starts at
            # `common` minus the chat-template trailing tokens.
            chunk_start = common
            chunk_end = len(full_tokens) - 8  # skip trailing template tokens
            # Guard: need at least 50 chunk tokens.
            if chunk_end - chunk_start < 50:
                print(f"  wrapper {wi}: chunk too short ({chunk_end - chunk_start}); skipping", flush=True)
                continue
            # Install hook and run prefill.
            cleanup = _install_kproj_capture(
                engine._target_model,
                chunk_start_tokens=chunk_start,
                chunk_end_tokens=chunk_end,
                storage=storage,
            )
            try:
                generate_dflash_once(
                    target_model=engine._target_model,
                    tokenizer=tok,
                    draft_model=engine._draft_model,
                    prompt="",
                    max_new_tokens=0,
                    prompt_tokens_override=full_tokens,
                    tq_bits=engine._resolved_tq_bits(),
                    pq_bits=engine._resolved_pq_bits(),
                    return_final_state=False,
                    prefill_only=True,
                )
                print(f"  wrapper {wi}: captured {chunk_end - chunk_start} chunk tokens "
                      f"(prefix {common})", flush=True)
            finally:
                cleanup()

        # Normalize capture lengths across wrappers.
        # Trim each capture to min chunk length across wrappers.
        min_len = min(len(s) for _, sl in storage.items() for s in sl) if storage else 0
        if min_len == 0:
            print(f"  no valid captures for chunk '{chunk_id}'", flush=True)
            continue
        min_tokens = min(s.shape[0] for sl in storage.values() for s in sl)
        for li in storage:
            storage[li] = [s[:min_tokens] for s in storage[li]]

        report = _analyze(storage, chunk_id)
        reports[chunk_id] = report
        print(f"  median ratio across layers: {report.median_across_layers:.4f}", flush=True)
        print(f"  mean   ratio across layers: {report.mean_across_layers:.4f}", flush=True)
        # Per-layer detail
        for pl in report.per_layer:
            print(f"    L{pl['layer']:>2d}: scalar={pl['scalar_ratio']:.4f}  "
                  f"median_pos={pl['median_pos_ratio']:.4f}  "
                  f"worst_pos={pl['worst_pos_ratio']:.4f}",
                  flush=True)

    # Aggregate summary
    print(f"\n[kv_splice] === SUMMARY ===", flush=True)
    print(f"{'chunk':>12}  {'median':>8}  {'mean':>8}  {'verdict':>20}")
    for chunk_id, r in reports.items():
        if r.median_across_layers < 0.3:
            verdict = "splice-safe"
        elif r.median_across_layers < 0.5:
            verdict = "marginal"
        else:
            verdict = "context-dependent"
        print(f"  {chunk_id:>10}  {r.median_across_layers:>8.3f}  "
              f"{r.mean_across_layers:>8.3f}  {verdict:>20}",
              flush=True)

    Path(args.out).write_text(json.dumps(
        {cid: asdict(r) for cid, r in reports.items()}, indent=2,
    ))
    print(f"\n[kv_splice] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
