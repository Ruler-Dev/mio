"""Frozen-KV-only benchmark on large-moe. Cold prefill, warm_and_freeze,
warm runs at ctx=[4K, 16K, 32K]. Reports prefill wall-clock deltas and
output hash match.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from bench_kv_experimentation import (  # re-use helpers
    CODING_PROMPTS,
    CONTEXT_SHIM,
    Run,
    _best_of,
    _build_prompt_at_size,
    _patch_env,
    _run_once,
    _sha256,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", nargs="+", type=int, default=[4096, 16384, 32768])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--out", default="/tmp/bench-runs/frozen_only.json")
    args = p.parse_args()

    from mio.config import MioConfig
    from mio.engine import MioEngine

    cfg = MioConfig.default()
    tc = cfg.tiers["large-moe"]
    eng = MioEngine(tier_config=tc)
    print(f"[frozen] loading {tc.name}...", flush=True)
    eng.load()
    print(f"[frozen] loaded, ctx={tc.context_window}", flush=True)

    results: list[Run] = []
    warm_task_id, warm_user_q, warm_max_out = CODING_PROMPTS[1]  # sort_bug

    with tempfile.TemporaryDirectory() as td, _patch_env(
        {"MIO_FROZEN_KV": "1", "MIO_FROZEN_KV_DIR": td, "MIO_FROZEN_KV_PREFIX": "512"}
    ):
        for ctx in args.ctx:
            shim_prompt = _build_prompt_at_size(warm_user_q, ctx, eng._tokenizer)
            messages = [{"role": "user", "content": shim_prompt}]
            est_tokens = len(eng._tokenizer.encode(shim_prompt))
            print(f"\n[frozen] ctx~{ctx} actual={est_tokens}", flush=True)

            eng._prefix_cache_invalidate()
            cold = _run_once(
                eng, messages, warm_max_out, est_tokens,
                config="frozen_cold", prompt_id=warm_task_id, ctx_target=ctx,
            )
            results.append(cold)
            print(
                f"  cold: prefill={cold.prefill_ms:.0f}ms gen={cold.gen_ms:.0f}ms "
                f"tps={cold.gen_tps:.1f} out={cold.output_sha256}",
                flush=True,
            )

            eng._prefix_cache_invalidate()
            t0 = time.perf_counter()
            path = eng.warm_and_freeze(messages)
            wf_ms = (time.perf_counter() - t0) * 1000
            print(
                f"  warm_and_freeze: {wf_ms:.0f}ms  path={path.name if path else 'FAILED'}",
                flush=True,
            )

            warms: list[Run] = []
            for rep in range(args.repeats):
                eng._prefix_cache_invalidate()
                r = _run_once(
                    eng, messages, warm_max_out, est_tokens,
                    config="frozen_warm", prompt_id=warm_task_id, ctx_target=ctx,
                )
                warms.append(r)
                print(
                    f"  warm rep{rep}: prefill={r.prefill_ms:.0f}ms "
                    f"gen={r.gen_ms:.0f}ms tps={r.gen_tps:.1f} out={r.output_sha256}",
                    flush=True,
                )
            best = _best_of(warms, key=lambda x: x.prefill_ms)
            results.append(best)
            saved = cold.prefill_ms - best.prefill_ms
            print(
                f"  -> prefill saved: {saved:.0f}ms "
                f"({best.prefill_ms/cold.prefill_ms:.3f}x of cold, "
                f"{cold.prefill_ms/max(best.prefill_ms,1):.1f}x speedup)",
                flush=True,
            )
            match = "MATCH" if best.output_sha256 == cold.output_sha256 else "DIFFER"
            print(f"  output-sha vs cold: {match}", flush=True)

    out = {"runs": [r.__dict__ for r in results]}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[frozen] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
