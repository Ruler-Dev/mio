"""Harvest per-layer (x_in, x_out) pairs during prefill.

For each decoder layer, we capture:
  - x_in: residual stream at the layer's input (pre layernorm).
  - x_out: residual stream after the layer's contribution is added.

The target for our linear-replacement hypothesis is `x_out - x_in` — the
layer's residual delta. If that delta is linearly related to x_in (after
a layernorm), we can approximate the layer as `x + W @ LN(x)`.

Implementation: wrap the DecoderLayer class __call__ to read `x` (input)
and `out` (return value), saving both per layer. One call per sample —
we run prefill_only so no decode noise enters the harvest.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def _synth_prompts(tokenizer, n: int = 30, target_tokens: int = 2048) -> list[str]:
    """Generate n diverse code-like prompts at roughly target_tokens length."""
    # Varied shims so samples are linguistically diverse.
    shims = [
        "from dataclasses import dataclass\nfrom pathlib import Path\nimport json, time, os, hashlib\n",
        "import numpy as np\nimport math\nfrom typing import Callable\n\n",
        "import asyncio\nfrom concurrent.futures import ThreadPoolExecutor\n\n",
        "from collections import defaultdict, deque, Counter\nimport heapq\n\n",
    ]
    bodies = [
        """
@dataclass
class CacheEntry:
    key: str
    value: object
    hits: int = 0

class LRU:
    def __init__(self, cap):
        self.cap = cap
        self._d = {}
        self._order = []
    def get(self, k):
        e = self._d.get(k)
        if e is None: return None
        e.hits += 1
        self._order.remove(k); self._order.append(k)
        return e.value
    def put(self, k, v):
        if k in self._d:
            self._d[k].value = v; return
        self._d[k] = CacheEntry(k, v)
        self._order.append(k)
        while len(self._d) > self.cap:
            old = self._order.pop(0); del self._d[old]
""",
        """
def dijkstra(graph, source):
    dist = {v: float('inf') for v in graph}
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in graph[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
""",
        """
async def fetch_all(urls):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as exec:
        tasks = [loop.run_in_executor(exec, fetch_one, u) for u in urls]
        return await asyncio.gather(*tasks)
""",
        """
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])
    return dp[n][capacity]
""",
    ]
    questions = [
        "Explain how the LRU eviction works in the above code in 2 sentences.",
        "Add a type hint for the `graph` parameter in the dijkstra function.",
        "Write a Python `is_prime(n)` that returns a bool.",
        "Fix this: `def sort(a): return a.sort()` — explain why it's wrong.",
        "Convert the `fetch_all` function to use pure asyncio without ThreadPoolExecutor.",
        "Write a Python function that reverses a linked list iteratively.",
        "Implement a trie data structure in Python with insert and search methods.",
        "Write a docstring for the knapsack function above.",
        "Give 3 unit tests for `is_prime`.",
        "Refactor the LRU class to use OrderedDict.",
    ]

    prompts = []
    for i in range(n):
        shim = shims[i % len(shims)]
        body = bodies[i % len(bodies)]
        q = questions[i % len(questions)]
        prompt = shim + body + f"\n# Sample {i}\n"
        # Pad to target_tokens by repeating bodies.
        while len(tokenizer.encode(prompt + f"\n---\n{q}")) < target_tokens:
            prompt = prompt + body
        while len(tokenizer.encode(prompt + f"\n---\n{q}")) > target_tokens + 150:
            prompt = prompt[: int(len(prompt) * 0.95)]
        prompts.append(prompt + f"\n---\n{q}")
    return prompts


def _install_harvest(target_model: Any, storage: dict[int, dict[str, list]]) -> Any:
    """Hook each decoder layer's __call__ to capture (x_in, x_out) per-sample.

    storage[layer_idx]["X"] = [np.array per sample] of shape (L, d)
    storage[layer_idx]["Y"] = [np.array per sample] of shape (L, d)
    storage[layer_idx]["is_linear"] = bool
    """
    from mio.dflash.runtime import _target_text_model
    text = _target_text_model(target_model)
    layers = list(text.layers)
    id_to_idx: dict[int, int] = {id(l): i for i, l in enumerate(layers)}
    for i, l in enumerate(layers):
        storage[i] = {"X": [], "Y": [], "is_linear": bool(getattr(l, "is_linear", False))}

    distinct: dict[type, Any] = {}
    for l in layers:
        cls = type(l)
        if cls not in distinct:
            distinct[cls] = cls.__call__

    def _make_wrapper(original_call):
        def wrapper(self, x, mask=None, cache=None):
            idx = id_to_idx.get(id(self), -1)
            out = original_call(self, x, mask=mask, cache=cache)
            if idx >= 0:
                x_cpu = np.array(x[0].astype(mx.float32), copy=True)
                out_cpu = np.array(out[0].astype(mx.float32), copy=True)
                storage[idx]["X"].append(x_cpu)
                storage[idx]["Y"].append(out_cpu)
            return out
        return wrapper

    for cls, orig in distinct.items():
        cls.__call__ = _make_wrapper(orig)

    def cleanup() -> None:
        for cls, orig in distinct.items():
            cls.__call__ = orig

    return cleanup


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="large-moe")
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--out", default="experiments/distill_linear/harvest.npz")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine
    from mio.dflash.runtime import generate_dflash_once

    cfg = MioConfig.default()
    tc = cfg.tiers[args.tier]
    print(f"[harvest] loading {args.tier} ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    print(f"[harvest] loaded.", flush=True)

    prompts = _synth_prompts(engine._tokenizer, n=args.n_samples, target_tokens=args.ctx)
    print(f"[harvest] {len(prompts)} prompts built @ ~{args.ctx} tokens each", flush=True)

    storage: dict[int, dict[str, list]] = {}
    cleanup = _install_harvest(engine._target_model, storage)

    try:
        t0 = time.perf_counter()
        for i, p in enumerate(prompts):
            messages = [{"role": "user", "content": p}]
            prompt_tokens = engine._apply_chat_template(messages)
            generate_dflash_once(
                target_model=engine._target_model,
                tokenizer=engine._tokenizer,
                draft_model=engine._draft_model,
                prompt="",
                max_new_tokens=0,
                prompt_tokens_override=prompt_tokens,
                tq_bits=engine._resolved_tq_bits(),
                pq_bits=engine._resolved_pq_bits(),
                return_final_state=False,
                prefill_only=True,
            )
            if (i + 1) % 5 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  sample {i+1}/{len(prompts)}  elapsed={elapsed:.1f}s",
                      flush=True)
    finally:
        cleanup()

    # Save per-layer to separate .npy files — avoids 2x transient memory
    # from np.savez_compressed which holds everything in RAM then zips.
    print(f"[harvest] saving per-layer ...", flush=True)
    out_dir = Path(args.out).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {"layers": [], "d_model": None}
    for i in sorted(storage.keys()):
        slot = storage[i]
        if not slot["X"]:
            continue
        # Concatenate this layer, write, free memory before next.
        X = np.concatenate(slot["X"], axis=0).astype(np.float16)
        Y = np.concatenate(slot["Y"], axis=0).astype(np.float16)
        np.save(out_dir / f"layer{i}_X.npy", X)
        np.save(out_dir / f"layer{i}_Y.npy", Y)
        meta["layers"].append({
            "idx": i,
            "is_linear": slot["is_linear"],
            "n_samples": int(X.shape[0]),
            "d_model": int(X.shape[1]),
        })
        meta["d_model"] = int(X.shape[1])
        # Drop references from storage to free memory.
        slot["X"] = []
        slot["Y"] = []
        del X, Y
        print(
            f"  layer {i:2d} ({'GDN' if slot['is_linear'] else 'attn'})  saved",
            flush=True,
        )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[harvest] wrote {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
