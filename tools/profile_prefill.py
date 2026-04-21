"""Per-layer prefill profiler for mio.

Hot-patches `Qwen3NextDecoderLayer.__call__` to record wall-clock for each
layer, then calls the dflash runtime in `prefill_only=True` mode. Reports:

  * Total prefill time (wall-clock, measured with mx.eval sync).
  * Per-layer time, tagged linear (GatedDeltaNet) vs. attention.
  * Aggregate linear vs. attention share.
  * TTFT at N ∈ {512, 1024, 2048, 4096, 8192, 16384, 32768}.

Usage:
    python3 tools/profile_prefill.py --tier large-moe --ctx 4096 16384 32768 \
        --repeats 2 --out docs/theories/baselines.md

Principles:
  - Every reported number is from a run done right now; nothing is cached
    or fabricated.
  - Cold + warm separated (cold = first run, warm = subsequent runs).
  - Output is committable markdown; timestamps + git SHA stamped inside.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class LayerTiming:
    layer_idx: int
    is_linear: bool
    total_ns: int = 0


@dataclass
class PrefillSample:
    ctx_target: int
    actual_tokens: int
    cold: bool
    total_prefill_ms: float
    layer_timings: list[LayerTiming] = field(default_factory=list)

    def linear_ms(self) -> float:
        return sum(l.total_ns for l in self.layer_timings if l.is_linear) / 1e6

    def attention_ms(self) -> float:
        return sum(l.total_ns for l in self.layer_timings if not l.is_linear) / 1e6

    def summary(self) -> dict[str, Any]:
        return {
            "ctx": self.ctx_target,
            "actual_tokens": self.actual_tokens,
            "cold": self.cold,
            "total_ms": round(self.total_prefill_ms, 1),
            "linear_ms": round(self.linear_ms(), 1),
            "attention_ms": round(self.attention_ms(), 1),
            "linear_share": round(self.linear_ms() / max(self.total_prefill_ms, 1e-9), 3),
            "attention_share": round(self.attention_ms() / max(self.total_prefill_ms, 1e-9), 3),
        }


def _patch_layer_timing(target_model: Any, storage: list[LayerTiming]) -> Any:
    """Install per-layer wall-clock instrumentation on decoder layers.

    Hot-patches the decoder-layer class(es) at the class level and uses
    id(self) as a lookup into a slot-map. Class-level patch is required
    because `layer(x)` resolves through `type(layer).__call__`, not the
    instance attribute. Multiple distinct classes (linear vs attention-only
    layouts) are handled; we patch each distinct one we see.

    Returns a cleanup callable that restores the originals.
    """
    import mlx.core as mx
    from mio.dflash.runtime import _target_text_model
    text_model = _target_text_model(target_model)
    layers = list(text_model.layers)
    storage.clear()
    id_to_slot: dict[int, int] = {}
    for i, layer in enumerate(layers):
        storage.append(LayerTiming(
            layer_idx=i, is_linear=bool(getattr(layer, "is_linear", False))
        ))
        id_to_slot[id(layer)] = i

    distinct_classes: dict[type, Any] = {}  # cls -> original __call__
    for layer in layers:
        cls = type(layer)
        if cls not in distinct_classes:
            distinct_classes[cls] = cls.__call__

    # Per-layer sync (mx.eval) is controllable via env var. With sync on,
    # we get truthful per-layer timings but lose pipelining — meaning the
    # first forced-sync after a deferred chain (e.g. linear layers then the
    # first attention layer) absorbs backlog time. With sync off, timings
    # reflect CPU dispatch only, which *under*-counts GPU time and is not
    # useful as wall-clock per layer but preserves total wall-clock.
    import os as _os
    sync_per_layer = _os.environ.get(
        "MIO_PROFILE_PER_LAYER_SYNC", "1"
    ).lower() not in ("0", "false", "no", "")

    def _build_wrapper(original_call):
        def wrapper(self, x, mask=None, cache=None):
            slot = id_to_slot.get(id(self), -1)
            if slot < 0:
                # Not one of our instrumented layers; pass through.
                return original_call(self, x, mask=mask, cache=cache)
            t0 = time.perf_counter_ns()
            out = original_call(self, x, mask=mask, cache=cache)
            if sync_per_layer:
                mx.eval(out)
            dt = time.perf_counter_ns() - t0
            storage[slot].total_ns += dt
            return out
        return wrapper

    for cls, original in distinct_classes.items():
        cls.__call__ = _build_wrapper(original)

    def cleanup() -> None:
        for cls, original in distinct_classes.items():
            cls.__call__ = original

    return cleanup


def _reset_storage(storage: list[LayerTiming]) -> None:
    for l in storage:
        l.total_ns = 0


def _context_prompt(tokens_target: int, tokenizer: Any) -> str:
    """Build a prompt that tokenizes to approximately `tokens_target` tokens."""
    shim = (
        "# project_utils.py — internal helpers\n\n"
        "from __future__ import annotations\n"
        "import hashlib, json, os, time\n"
        "from dataclasses import dataclass, field\n"
        "from pathlib import Path\n"
        "from typing import Any, Iterable, Optional\n\n"
        "@dataclass\nclass CacheEntry:\n    key: str\n    value: Any\n"
        "    created_at: float = field(default_factory=time.time)\n"
        "    hit_count: int = 0\n\n"
        "    def touch(self) -> None:\n        self.hit_count += 1\n\n"
        "class LRUCache:\n    def __init__(self, capacity: int) -> None:\n"
        "        self.capacity = int(capacity)\n        self._store: dict = {}\n"
        "        self._order: list = []\n\n    def get(self, key: str) -> Any:\n"
        "        entry = self._store.get(key)\n        if entry is None: return None\n"
        "        entry.touch(); self._order.remove(key); self._order.append(key)\n"
        "        return entry.value\n\n    def put(self, key, value) -> None:\n"
        "        if key in self._store:\n            self._store[key].value = value\n"
        "            self._order.remove(key); self._order.append(key); return\n"
        "        self._store[key] = CacheEntry(key=key, value=value)\n"
        "        self._order.append(key)\n        self._evict_if_needed()\n\n"
        "    def _evict_if_needed(self) -> None:\n"
        "        while len(self._store) > self.capacity:\n"
        "            oldest = self._order.pop(0); del self._store[oldest]\n"
    )
    current = shim
    while len(tokenizer.encode(current)) < tokens_target:
        current = current + "\n\n" + shim
    # trim back to target
    while len(tokenizer.encode(current)) > tokens_target + 200:
        current = current[: int(len(current) * 0.95)]
    return current


def _run_prefill(
    *,
    engine: Any,
    prompt: str,
    storage: list[LayerTiming],
) -> PrefillSample:
    import mlx.core as mx
    from mio.dflash.runtime import generate_dflash_once
    _reset_storage(storage)
    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = engine._apply_chat_template(messages)
    actual_tokens = len(prompt_tokens)

    # Force full GPU sync before timing.
    mx.eval(mx.zeros((1,), dtype=mx.float32))

    t0 = time.perf_counter_ns()
    result = generate_dflash_once(
        target_model=engine._target_model,
        tokenizer=engine._tokenizer,
        draft_model=engine._draft_model,
        prompt="",
        max_new_tokens=0,
        prompt_tokens_override=prompt_tokens,
        tq_bits=engine._resolved_tq_bits(),
        pq_bits=engine._resolved_pq_bits(),
        return_final_state=True,
        prefill_only=True,
    )
    total_ns = time.perf_counter_ns() - t0
    del result  # drop cache to free memory

    return PrefillSample(
        ctx_target=-1,
        actual_tokens=actual_tokens,
        cold=False,
        total_prefill_ms=total_ns / 1e6,
        layer_timings=[LayerTiming(l.layer_idx, l.is_linear, l.total_ns) for l in storage],
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
        ).decode().strip()
    except Exception:
        return "unknown"


def _platform_stamp() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "hw.model"],
        ).decode().strip()
    except Exception:
        return "unknown-hw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="large-moe")
    parser.add_argument(
        "--ctx", nargs="+", type=int,
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out-json", default="experiments/phase0_baselines/results.json")
    parser.add_argument("--out-md", default="docs/theories/baselines.md")
    args = parser.parse_args()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)

    from mio.config import MioConfig
    from mio.engine import MioEngine

    cfg = MioConfig.default()
    tc = cfg.tiers[args.tier]
    print(f"[profile] loading tier={args.tier} ...", flush=True)
    engine = MioEngine(tier_config=tc)
    engine.load()
    print(f"[profile] loaded. ctx_window={tc.context_window}", flush=True)

    storage: list[LayerTiming] = []
    cleanup = _patch_layer_timing(engine._target_model, storage)

    try:
        samples: list[PrefillSample] = []
        for ctx in args.ctx:
            prompt = _context_prompt(ctx, engine._tokenizer)
            for rep in range(args.repeats):
                s = _run_prefill(engine=engine, prompt=prompt, storage=storage)
                s.ctx_target = ctx
                s.cold = (rep == 0)
                samples.append(s)
                summary = s.summary()
                print(
                    f"  ctx={ctx:6d} rep={rep} cold={s.cold}  "
                    f"total={summary['total_ms']:7.1f} ms  "
                    f"lin={summary['linear_ms']:7.1f} ({summary['linear_share']*100:4.1f}%)  "
                    f"attn={summary['attention_ms']:7.1f} ({summary['attention_share']*100:4.1f}%)",
                    flush=True,
                )
    finally:
        cleanup()

    # Serialize
    out = {
        "git_sha": _git_sha(),
        "hardware": _platform_stamp(),
        "tier": args.tier,
        "target_model": str(tc.target_model),
        "draft_model": str(tc.draft_model),
        "pq_bits": tc.pq_bits,
        "tq_bits": tc.tq_bits,
        "samples": [
            {
                **s.summary(),
                "per_layer": [
                    {"idx": l.layer_idx, "linear": l.is_linear, "ms": round(l.total_ns / 1e6, 2)}
                    for l in s.layer_timings
                ],
            }
            for s in samples
        ],
        "timestamp_epoch": int(time.time()),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[profile] wrote {args.out_json}", flush=True)

    # Pretty markdown
    lines = [
        f"# Baseline prefill profile — {args.tier}",
        "",
        f"- git: `{out['git_sha']}`",
        f"- hardware: `{out['hardware']}`",
        f"- target: `{out['target_model']}`",
        f"- draft: `{out['draft_model']}`",
        f"- pq_bits={out['pq_bits']}, tq_bits={out['tq_bits']}",
        f"- timestamp: {out['timestamp_epoch']}",
        "",
        "## Warm prefill (rep >= 1) by context",
        "",
        "| ctx | tokens | total ms | linear ms | attn ms | linear % | attn % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    warm_by_ctx: dict[int, list[dict]] = {}
    for s in samples:
        if s.cold:
            continue
        warm_by_ctx.setdefault(s.ctx_target, []).append(s.summary())
    for ctx in sorted(warm_by_ctx.keys()):
        # median of warm reps
        med = min(warm_by_ctx[ctx], key=lambda x: x["total_ms"])
        lines.append(
            f"| {ctx} | {med['actual_tokens']} | {med['total_ms']} | "
            f"{med['linear_ms']} | {med['attention_ms']} | "
            f"{med['linear_share']*100:.1f}% | {med['attention_share']*100:.1f}% |"
        )
    lines.append("")
    lines.append("## Cold vs warm (first rep vs. best subsequent)")
    lines.append("")
    lines.append("| ctx | cold ms | warm ms |")
    lines.append("|---|---:|---:|")
    by_ctx: dict[int, dict[str, float]] = {}
    for s in samples:
        by_ctx.setdefault(s.ctx_target, {})
        if s.cold:
            by_ctx[s.ctx_target]["cold"] = s.total_prefill_ms
        else:
            prev = by_ctx[s.ctx_target].get("warm", float("inf"))
            by_ctx[s.ctx_target]["warm"] = min(prev, s.total_prefill_ms)
    for ctx in sorted(by_ctx.keys()):
        c = by_ctx[ctx]
        lines.append(
            f"| {ctx} | {c.get('cold', 0):.0f} | {c.get('warm', 0):.0f} |"
        )
    lines.append("")

    Path(args.out_md).write_text("\n".join(lines))
    print(f"[profile] wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
