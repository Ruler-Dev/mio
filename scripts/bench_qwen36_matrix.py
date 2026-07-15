"""Reproducible Qwen 3.6 baseline/DFlash/cache benchmark matrix.

The harness loads one target/draft pair, warms every execution mode, then
records per-repetition timings and parity against greedy autoregressive output.
Results are written as JSON so documentation and papers can consume the raw
measurements rather than copying terminal summaries.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import mlx.core as mx

from mio.config import MioConfig
from mio.dflash.runtime import (
    bind_draft_target_model,
    generate_baseline_once,
    generate_dflash_once,
    load_draft_bundle,
    load_target_bundle,
    validate_draft_target_compatibility,
)


SEED_TEXT = (
    "Reliable software engineering combines explicit invariants, small reviewable "
    "changes, deterministic tests, observability, and careful failure handling. "
)


@dataclass(frozen=True)
class Mode:
    name: str
    dflash: bool
    pq_bits: int | None = None
    tq_bits: int | None = None


MODES = {
    "baseline": Mode("baseline", dflash=False),
    "dflash": Mode("dflash", dflash=True),
    "pq4": Mode("pq4", dflash=True, pq_bits=4),
    "tq4": Mode("tq4", dflash=True, tq_bits=4),
}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _portable_model_reference(reference: str) -> str:
    """Keep benchmark artifacts portable when a local checkpoint was used."""

    path = Path(reference)
    return path.name if path.is_absolute() else reference


def _tile_tokens(tokenizer: Any, target_length: int) -> list[int]:
    seed = list(tokenizer.encode(SEED_TEXT, add_special_tokens=False))
    if not seed:
        raise RuntimeError("tokenizer produced an empty benchmark seed")
    repeats = (target_length + len(seed) - 1) // len(seed)
    return (seed * repeats)[:target_length]


def _run_once(
    mode: Mode,
    *,
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    kwargs = {
        "target_model": target_model,
        "tokenizer": tokenizer,
        "prompt": "",
        "max_new_tokens": max_new_tokens,
        "prompt_tokens_override": prompt_tokens,
        "pq_bits": mode.pq_bits,
        "tq_bits": mode.tq_bits,
    }
    if mode.dflash:
        result = generate_dflash_once(draft_model=draft_model, **kwargs)
    else:
        result = generate_baseline_once(**kwargs)

    elapsed_us = float(result.get("elapsed_us", 0.0))
    prefill_us = float(
        result.get("prefill_us")
        or (result.get("phase_timings_us") or {}).get("prefill", 0.0)
    )
    generation_tokens = int(result.get("generation_tokens", 0))
    decode_us = max(0.0, elapsed_us - prefill_us)
    token_ids = [int(token) for token in result.get("generated_token_ids", [])]
    return {
        "elapsed_us": elapsed_us,
        "prefill_us": prefill_us,
        "decode_us": decode_us,
        "prompt_tokens": len(prompt_tokens),
        "generation_tokens": generation_tokens,
        "prefill_tps": len(prompt_tokens) / max(prefill_us / 1e6, 1e-9),
        "decode_tps": generation_tokens / max(decode_us / 1e6, 1e-9),
        "end_to_end_tps": generation_tokens / max(elapsed_us / 1e6, 1e-9),
        "acceptance_ratio": float(result.get("acceptance_ratio", 0.0)),
        "tokens_per_cycle": float(result.get("tokens_per_cycle", 0.0)),
        "peak_memory_gb": float(result.get("peak_memory_gb") or 0.0),
        "fallback_ar": bool(result.get("fallback_ar", False)),
        "token_ids": token_ids,
        "token_hash": hashlib.sha256(
            ",".join(str(token) for token in token_ids).encode("ascii")
        ).hexdigest(),
    }


def _aggregate(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = (
        "elapsed_us",
        "prefill_us",
        "decode_us",
        "prefill_tps",
        "decode_tps",
        "end_to_end_tps",
        "acceptance_ratio",
        "tokens_per_cycle",
        "peak_memory_gb",
    )
    return {
        f"median_{key}": statistics.median(float(row[key]) for row in repetitions)
        for key in numeric
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="large")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--modes", default="baseline,dflash,pq4,tq4")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail when unquantized DFlash differs from greedy baseline",
    )
    args = parser.parse_args()

    requested_modes = [name.strip() for name in args.modes.split(",") if name.strip()]
    unknown = sorted(set(requested_modes) - MODES.keys())
    if unknown:
        parser.error(f"unknown modes: {', '.join(unknown)}")
    if "baseline" not in requested_modes:
        requested_modes.insert(0, "baseline")

    tier = MioConfig.default().tiers[args.tier]
    print(f"[load] target={tier.target_model}", flush=True)
    print(f"[load] draft={tier.draft_model}", flush=True)
    load_start = time.perf_counter()
    target_model, tokenizer, target_meta = load_target_bundle(
        tier.target_model,
        lazy=True,
        split_full_attention_sdpa=True,
    )
    draft_model, draft_meta = load_draft_bundle(tier.draft_model, lazy=True)
    compatibility = validate_draft_target_compatibility(
        target_meta.get("config") or {},
        draft_meta.get("config") or {},
    )
    bind_draft_target_model(draft_model, target_model)
    mx.eval(target_model.parameters(), draft_model.parameters())
    load_seconds = time.perf_counter() - load_start
    print(f"[load] ready in {load_seconds:.2f}s", flush=True)

    prompt_tokens = _tile_tokens(tokenizer, args.prompt_tokens)
    rows: dict[str, Any] = {}
    baseline_tokens: list[int] | None = None
    for mode_name in requested_modes:
        mode = MODES[mode_name]
        print(f"[mode] {mode_name}: warmup={args.warmup} reps={args.reps}", flush=True)
        for _ in range(args.warmup):
            _run_once(
                mode,
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=args.max_tokens,
            )

        repetitions = []
        for repetition in range(args.reps):
            result = _run_once(
                mode,
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=args.max_tokens,
            )
            if baseline_tokens is None and mode_name == "baseline":
                baseline_tokens = result["token_ids"]
            result["matches_baseline"] = result["token_ids"] == baseline_tokens
            repetitions.append(result)
            print(
                f"  rep={repetition + 1} prefill={result['prefill_tps']:.1f} tok/s "
                f"decode={result['decode_tps']:.2f} tok/s "
                f"accept={result['acceptance_ratio']:.3f} "
                f"peak={result['peak_memory_gb']:.2f} GB "
                f"parity={result['matches_baseline']}",
                flush=True,
            )
        rows[mode_name] = {
            "mode": asdict(mode),
            "aggregate": _aggregate(repetitions),
            "all_match_baseline": all(row["matches_baseline"] for row in repetitions),
            "repetitions": repetitions,
        }

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "software": {
            "python": platform.python_version(),
            "mlx": _package_version("mlx"),
            "mlx-lm": _package_version("mlx-lm"),
            "dflash-mlx": _package_version("dflash-mlx"),
        },
        "models": {
            "tier": args.tier,
            "target": _portable_model_reference(tier.target_model),
            "draft": _portable_model_reference(tier.draft_model),
            "compatibility": compatibility,
        },
        "parameters": {
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "warmup": args.warmup,
            "reps": args.reps,
            "modes": requested_modes,
        },
        "load_seconds": load_seconds,
        "results": rows,
    }

    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path("benchmarks/results") / f"qwen36-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[result] {output}", flush=True)

    dflash_parity = rows.get("dflash", {}).get("all_match_baseline", True)
    if args.strict_parity and not dflash_parity:
        print("[error] unquantized DFlash output diverged from greedy baseline", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
