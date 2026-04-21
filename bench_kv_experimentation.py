"""Benchmark harness for kv-experimentation branch work.

Loads large-moe once, runs:
  - baseline: mio default (PolarQuant 4-bit + DFlash)
  - ddtree:   MIO_DDTREE_BUDGET=4 (8-bit KV + tree-attention verify)
  - frozen_kv: MIO_FROZEN_KV=1, two-pass (cold then warm)

across several coding prompts at two context sizes. Reports timings and
checks output equivalence where expected to be lossless.

B2 (speculative prefill) is NOT benchmarked — the partial-forward hook is
a no-op; the runtime always falls back. Running it would measure baseline
plus Python overhead.

Usage:
    python3 bench_kv_experimentation.py --out bench_results.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---- prompt library ---------------------------------------------------------

CODING_PROMPTS: list[tuple[str, str, int]] = [
    (
        "fib",
        "Write a Python function `fib(n)` that computes the n-th Fibonacci "
        "number using memoization. Include a short docstring and 3 example "
        "calls. Keep it under 20 lines.",
        192,
    ),
    (
        "sort_bug",
        "Fix the bug in this function so that it sorts an array of "
        "integers correctly, including duplicates:\n\n"
        "```python\n"
        "def sort(a):\n"
        "    for i in range(len(a)):\n"
        "        for j in range(i, len(a)):\n"
        "            if a[i] > a[j]:\n"
        "                a[i] = a[j]\n"
        "    return a\n"
        "```\n\n"
        "Return only the corrected function and one short paragraph "
        "explaining what was wrong.",
        256,
    ),
    (
        "bst",
        "Write a minimal BinarySearchTree class in Python with methods "
        "`insert`, `contains`, and `inorder` (returning a list). No "
        "external libraries. Include a quick example at the bottom.",
        384,
    ),
    (
        "nqueens",
        "Implement the classic N-queens backtracking algorithm in Python. "
        "Your `solve(n)` should return a list of solutions, each a list "
        "of column indices per row. Explain the pruning briefly in a "
        "comment block at the top, then the code.",
        512,
    ),
]


# A ~1.5K-token "codebase context" block we can repeat to grow the prefix.
# Using real-looking source so the model's hidden states are in-distribution.
CONTEXT_SHIM = """# project_utils.py — internal helpers

from __future__ import annotations
import hashlib, json, os, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def touch(self) -> None:
        self.hit_count += 1


class LRUCache:
    \"\"\"Simple LRU with eviction callback and byte-budget awareness.\"\"\"

    def __init__(self, capacity: int, byte_budget: int | None = None) -> None:
        self.capacity = int(capacity)
        self.byte_budget = byte_budget
        self._store: dict[str, CacheEntry] = {}
        self._order: list[str] = []

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        entry.touch()
        self._order.remove(key)
        self._order.append(key)
        return entry.value

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store[key].value = value
            self._order.remove(key)
            self._order.append(key)
            return
        self._store[key] = CacheEntry(key=key, value=value)
        self._order.append(key)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self._store) > self.capacity:
            oldest = self._order.pop(0)
            del self._store[oldest]
        if self.byte_budget is not None:
            total = sum(_sizeof(e.value) for e in self._store.values())
            while total > self.byte_budget and self._order:
                oldest = self._order.pop(0)
                total -= _sizeof(self._store[oldest].value)
                del self._store[oldest]


def _sizeof(value: Any) -> int:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return 64


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def read_config(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} must be a JSON object")
    return data


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


# request_router.py — inbound API dispatch

from typing import Callable

HandlerMap = dict[str, Callable[[dict], dict]]

def make_router(handlers: HandlerMap) -> Callable[[dict], dict]:
    def route(request: dict) -> dict:
        op = request.get("op")
        if op not in handlers:
            return {"error": f"unknown op {op!r}"}
        try:
            return {"ok": True, "data": handlers[op](request)}
        except Exception as exc:
            return {"error": str(exc)}
    return route


def chain(*middlewares):
    def outer(handler):
        wrapped = handler
        for mw in reversed(middlewares):
            wrapped = mw(wrapped)
        return wrapped
    return outer


def logging_middleware(handler):
    def wrapper(request):
        started = time.perf_counter()
        result = handler(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"[trace] op={request.get('op')} ms={elapsed_ms:.1f}", flush=True)
        return result
    return wrapper


def retry_middleware(max_attempts: int = 3):
    def decorator(handler):
        def wrapper(request):
            last_exc = None
            for _ in range(max_attempts):
                try:
                    return handler(request)
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.05)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
"""


def _build_prompt_at_size(
    user_q: str, target_tokens: int, tokenizer: Any
) -> str:
    """Pad a user question with CONTEXT_SHIM until its token count ~= target."""
    # Start with just the shim repeated.
    full = CONTEXT_SHIM
    while len(tokenizer.encode(full + "\n\n" + user_q)) < target_tokens:
        full = full + "\n\n" + CONTEXT_SHIM
    # Trim slightly over to just under the target.
    while len(tokenizer.encode(full + "\n\n" + user_q)) > target_tokens + 200:
        full = full[: int(len(full) * 0.9)]
    return full + "\n\n---\n\n" + user_q


# ---- runner -----------------------------------------------------------------


@dataclass
class Run:
    config: str
    prompt_id: str
    ctx_target: int
    prompt_tokens: int
    prefill_ms: float
    gen_ms: float
    gen_tps: float
    gen_tokens: int
    avg_accept: float
    output_text: str
    output_sha256: str
    peak_mem_gb: float = 0.0


@dataclass
class BenchResult:
    runs: list[Run] = field(default_factory=list)

    def dict(self) -> dict:
        return {"runs": [asdict(r) for r in self.runs]}


@contextlib.contextmanager
def _patch_env(overrides: dict[str, Optional[str]]):
    """Set env vars for a block; restore on exit."""
    prev: dict[str, Optional[str]] = {}
    for k, v in overrides.items():
        prev[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _run_once(
    engine: Any,
    messages: list[dict],
    max_tokens: int,
    prompt_tokens: int,
    *,
    config: str,
    prompt_id: str,
    ctx_target: int,
) -> Run:
    t0 = time.perf_counter()
    text, metrics = engine.generate(messages=messages, max_tokens=max_tokens)
    elapsed = time.perf_counter() - t0
    prefill_ms = 0.0
    gen_ms = elapsed * 1000.0
    # MioEngine.last_metrics has prompt_tps (prefill), generation_tps (decode).
    if metrics.prompt_tps > 0 and metrics.prompt_tokens > 0:
        prefill_ms = (metrics.prompt_tokens / metrics.prompt_tps) * 1000.0
        gen_ms = max(0.0, elapsed * 1000.0 - prefill_ms)
    return Run(
        config=config,
        prompt_id=prompt_id,
        ctx_target=ctx_target,
        prompt_tokens=metrics.prompt_tokens or prompt_tokens,
        prefill_ms=prefill_ms,
        gen_ms=gen_ms,
        gen_tps=metrics.generation_tps,
        gen_tokens=metrics.completion_tokens,
        avg_accept=metrics.avg_acceptance_length,
        output_text=text,
        output_sha256=_sha256(text),
        peak_mem_gb=metrics.peak_memory_gb,
    )


def _best_of(runs: list[Run], key=lambda r: r.gen_ms) -> Run:
    """Return the run with the minimum key (fastest)."""
    return min(runs, key=key)


def run_bench(
    *,
    tier_name: str,
    out_path: Path,
    ctx_sizes: list[int],
    repeats: int,
    frozen_prefix_len: int,
) -> BenchResult:
    print(f"[bench] loading tier={tier_name}...", flush=True)
    from mio.config import MioConfig
    from mio.engine import MioEngine

    cfg = MioConfig.default()
    tc = cfg.tiers[tier_name]
    engine = MioEngine(tier_config=tc)
    engine.load()
    tok = engine._tokenizer
    print(f"[bench] loaded. ctx={tc.context_window} draft={tc.draft_model}", flush=True)

    result = BenchResult()

    # --- Round 1: BASELINE (default PQ4 + DFlash) -------------------------
    print("\n[bench] === BASELINE (PolarQuant-4 + DFlash) ===", flush=True)
    for prompt_id, user_q, max_out in CODING_PROMPTS:
        for ctx in ctx_sizes:
            shim_prompt = _build_prompt_at_size(user_q, ctx, tok)
            messages = [{"role": "user", "content": shim_prompt}]
            est_tokens = len(tok.encode(shim_prompt))
            print(
                f"  prompt={prompt_id} ctx~{ctx} actual={est_tokens} "
                f"max_out={max_out}", flush=True,
            )
            runs = []
            for rep in range(repeats):
                r = _run_once(
                    engine, messages, max_out, est_tokens,
                    config="baseline", prompt_id=prompt_id, ctx_target=ctx,
                )
                runs.append(r)
                print(
                    f"    rep{rep}: prefill={r.prefill_ms:.0f}ms "
                    f"gen={r.gen_ms:.0f}ms tps={r.gen_tps:.1f} "
                    f"accept={r.avg_accept:.2f} "
                    f"out={r.output_sha256}",
                    flush=True,
                )
            result.runs.append(_best_of(runs))

    engine._prefix_cache_invalidate()

    # --- Round 2: DDTREE ---------------------------------------------------
    print("\n[bench] === DDTREE budget=4 ===", flush=True)
    with _patch_env({"MIO_DDTREE_BUDGET": "4"}):
        for prompt_id, user_q, max_out in CODING_PROMPTS:
            for ctx in ctx_sizes:
                shim_prompt = _build_prompt_at_size(user_q, ctx, tok)
                messages = [{"role": "user", "content": shim_prompt}]
                est_tokens = len(tok.encode(shim_prompt))
                print(
                    f"  prompt={prompt_id} ctx~{ctx} actual={est_tokens} "
                    f"max_out={max_out}", flush=True,
                )
                runs = []
                for rep in range(repeats):
                    r = _run_once(
                        engine, messages, max_out, est_tokens,
                        config="ddtree", prompt_id=prompt_id, ctx_target=ctx,
                    )
                    runs.append(r)
                    print(
                        f"    rep{rep}: prefill={r.prefill_ms:.0f}ms "
                        f"gen={r.gen_ms:.0f}ms tps={r.gen_tps:.1f} "
                        f"accept={r.avg_accept:.2f} "
                        f"out={r.output_sha256}",
                        flush=True,
                    )
                result.runs.append(_best_of(runs))

    engine._prefix_cache_invalidate()

    # --- Round 3: FROZEN KV -----------------------------------------------
    # For each ctx size, run the same prompt twice (cold + warm) and
    # measure prefill delta. Use a short user message appended after the
    # shim so the first frozen_prefix_len tokens are shared.
    print("\n[bench] === FROZEN KV (cold + warm) ===", flush=True)
    with tempfile.TemporaryDirectory() as td, _patch_env(
        {
            "MIO_FROZEN_KV": "1",
            "MIO_FROZEN_KV_DIR": td,
            "MIO_FROZEN_KV_PREFIX": str(frozen_prefix_len),
        }
    ):
        # Pick one representative coding task for warm-reuse runs.
        warm_task_id, warm_user_q, warm_max_out = CODING_PROMPTS[1]  # sort_bug
        for ctx in ctx_sizes:
            if ctx <= frozen_prefix_len:
                print(
                    f"  skipping ctx={ctx} (<= frozen_prefix={frozen_prefix_len})",
                    flush=True,
                )
                continue
            # Build a shared prefix of exactly ~ctx tokens, then append the
            # (short, unique) coding question after. Frozen-KV will hash the
            # first frozen_prefix_len tokens, which are inside the shared shim.
            shim_prompt = _build_prompt_at_size(warm_user_q, ctx, tok)
            messages = [{"role": "user", "content": shim_prompt}]
            est_tokens = len(tok.encode(shim_prompt))
            print(
                f"  prompt={warm_task_id} ctx~{ctx} actual={est_tokens} "
                f"frozen_prefix={frozen_prefix_len}", flush=True,
            )

            engine._prefix_cache_invalidate()
            cold = _run_once(
                engine, messages, warm_max_out, est_tokens,
                config="frozen_cold", prompt_id=warm_task_id, ctx_target=ctx,
            )
            print(
                f"    cold: prefill={cold.prefill_ms:.0f}ms "
                f"gen={cold.gen_ms:.0f}ms out={cold.output_sha256}",
                flush=True,
            )
            result.runs.append(cold)

            # Explicit warm-and-freeze pass: runs a clean prefill_only through
            # the target + draft, freezes the post-prefill cache. Cost: one
            # extra prefill paid once. Correctness: final_state is at
            # offset=prompt_len with no decode mutations.
            engine._prefix_cache_invalidate()
            import time as _t
            t_wf = _t.perf_counter()
            freeze_path = engine.warm_and_freeze(messages)
            warm_and_freeze_ms = (_t.perf_counter() - t_wf) * 1000
            print(
                f"    warm_and_freeze: {warm_and_freeze_ms:.0f}ms "
                f"path={freeze_path.name if freeze_path else 'FAILED'}",
                flush=True,
            )
            engine._prefix_cache_invalidate()

            # Warm run — should hit frozen KV on disk.
            warm_runs = []
            for rep in range(repeats):
                engine._prefix_cache_invalidate()
                r = _run_once(
                    engine, messages, warm_max_out, est_tokens,
                    config="frozen_warm", prompt_id=warm_task_id, ctx_target=ctx,
                )
                warm_runs.append(r)
                print(
                    f"    warm rep{rep}: prefill={r.prefill_ms:.0f}ms "
                    f"gen={r.gen_ms:.0f}ms out={r.output_sha256}",
                    flush=True,
                )
            best_warm = _best_of(warm_runs, key=lambda r: r.prefill_ms)
            result.runs.append(best_warm)

            # Quality check: frozen-warm output MUST equal cold output (same
            # prompt, same model, same greedy decode).
            if best_warm.output_sha256 != cold.output_sha256:
                print(
                    f"    ! QUALITY WARNING: warm sha={best_warm.output_sha256} "
                    f"!= cold sha={cold.output_sha256}",
                    flush=True,
                )
            else:
                print("    OK frozen-KV warm output identical to cold", flush=True)

    out_path.write_text(json.dumps(result.dict(), indent=2))
    print(f"\n[bench] results saved to {out_path}", flush=True)
    return result


def summarize(result: BenchResult) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" BENCHMARK SUMMARY")
    lines.append("=" * 72)
    lines.append(
        f"{'config':>14}  {'prompt':>9}  {'ctx':>6}  "
        f"{'ptok':>5}  {'prefill_ms':>10}  {'gen_ms':>7}  "
        f"{'tps':>6}  {'accept':>6}  {'sha':>8}"
    )
    lines.append("-" * 90)
    for r in result.runs:
        lines.append(
            f"{r.config:>14}  {r.prompt_id:>9}  {r.ctx_target:>6}  "
            f"{r.prompt_tokens:>5}  {r.prefill_ms:>10.0f}  {r.gen_ms:>7.0f}  "
            f"{r.gen_tps:>6.1f}  {r.avg_accept:>6.2f}  {r.output_sha256:>8}"
        )

    # Aggregate speedups.
    lines.append("\n--- aggregates ---")
    buckets: dict[tuple[str, str, int], Run] = {}
    for r in result.runs:
        buckets[(r.config, r.prompt_id, r.ctx_target)] = r
    configs = sorted({r.config for r in result.runs})
    prompts = sorted({r.prompt_id for r in result.runs})
    ctxs = sorted({r.ctx_target for r in result.runs})

    # Gen tps vs baseline at same (prompt, ctx).
    for prompt_id in prompts:
        for ctx in ctxs:
            base = buckets.get(("baseline", prompt_id, ctx))
            if base is None:
                continue
            for c in configs:
                if c == "baseline":
                    continue
                other = buckets.get((c, prompt_id, ctx))
                if other is None:
                    continue
                if other.gen_tps == 0 or base.gen_tps == 0:
                    continue
                ratio = other.gen_tps / base.gen_tps
                lines.append(
                    f"  gen tps {c}/baseline {prompt_id} ctx{ctx}: "
                    f"{ratio:.3f}x  ({other.gen_tps:.1f} vs {base.gen_tps:.1f})"
                )

    # Frozen KV prefill speedup (warm vs cold).
    for ctx in ctxs:
        cold = buckets.get(("frozen_cold", "sort_bug", ctx))
        warm = buckets.get(("frozen_warm", "sort_bug", ctx))
        if cold and warm and cold.prefill_ms > 0:
            ratio = warm.prefill_ms / cold.prefill_ms
            saved_ms = cold.prefill_ms - warm.prefill_ms
            lines.append(
                f"  frozen_kv prefill ctx{ctx}: warm/cold={ratio:.3f}x  "
                f"saved={saved_ms:.0f}ms"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="large-moe")
    parser.add_argument("--out", default="bench_results.json")
    parser.add_argument(
        "--ctx", nargs="+", type=int, default=[4096, 16384, 32768],
        help="Prompt token counts to benchmark at.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--frozen-prefix", type=int, default=2048,
        help="Token count the frozen-KV hash covers.",
    )
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    result = run_bench(
        tier_name=args.tier,
        out_path=out_path,
        ctx_sizes=args.ctx,
        repeats=args.repeats,
        frozen_prefix_len=args.frozen_prefix,
    )
    print("\n" + summarize(result))


if __name__ == "__main__":
    main()
