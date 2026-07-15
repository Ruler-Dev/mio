"""Reproducible Qwen 3.6 baseline/DFlash/cache benchmark matrix.

The harness loads one target/draft pair, warms every execution mode, then runs
seeded balanced mode-order blocks and records paired timings and parity against
greedy autoregressive output. Results are written as JSON so documentation and
papers can consume the raw measurements rather than terminal summaries.
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
import random
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
    exact: bool
    pq_bits: int | None = None
    tq_bits: int | None = None


MODES = {
    "baseline": Mode("baseline", dflash=False, exact=True),
    "dflash": Mode("dflash", dflash=True, exact=True),
    "pq4": Mode("pq4", dflash=True, exact=False, pq_bits=4),
    "tq4": Mode("tq4", dflash=True, exact=False, tq_bits=4),
}

SCHEMA_VERSION = 2
PHASE_TIMING_NAMES = (
    "prefill",
    "draft",
    "draft_prefill",
    "draft_incremental",
    "verify",
    "replay",
    "rebuild",
    "commit",
)
SPEEDUP_METRICS = ("prefill_tps", "decode_tps", "end_to_end_tps")


def _balanced_mode_schedule(
    mode_names: list[str],
    repetitions: int,
    *,
    seed: int,
) -> list[list[str]]:
    """Build seeded randomized Latin-rotation blocks.

    Every repetition contains every mode exactly once.  Within each complete
    block of ``len(mode_names)`` repetitions, every mode occupies every
    execution position exactly once.  Shuffling the base permutation and the
    rotation order keeps the schedule deterministic for a given seed without
    making the baseline systematically first.
    """

    if repetitions < 0:
        raise ValueError("repetitions must be non-negative")
    if not mode_names:
        raise ValueError("at least one mode is required")
    if len(set(mode_names)) != len(mode_names):
        raise ValueError("mode names must be unique")

    rng = random.Random(seed)
    schedule: list[list[str]] = []
    mode_count = len(mode_names)
    while len(schedule) < repetitions:
        base = list(mode_names)
        rng.shuffle(base)
        rotations = list(range(mode_count))
        rng.shuffle(rotations)
        for offset in rotations:
            schedule.append(base[offset:] + base[:offset])
            if len(schedule) == repetitions:
                break
    return schedule


def _paired_speedup(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float | None]:
    """Return candidate/baseline paired throughput ratios."""

    ratios: dict[str, float | None] = {}
    for metric in SPEEDUP_METRICS:
        baseline_value = float(baseline.get(metric, 0.0) or 0.0)
        candidate_value = float(candidate.get(metric, 0.0) or 0.0)
        ratios[metric] = (
            candidate_value / baseline_value if baseline_value > 0.0 else None
        )
    return ratios


def _organize_repetitions(
    mode_names: list[str],
    repetition_blocks: list[dict[str, dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    """Annotate paired results and check baseline self-determinism."""

    if not repetition_blocks:
        raise ValueError("at least one measured repetition is required")
    canonical_baseline = list(repetition_blocks[0]["baseline"]["token_ids"])
    baseline_deterministic = all(
        list(block["baseline"]["token_ids"]) == canonical_baseline
        for block in repetition_blocks
    )
    grouped = {mode_name: [] for mode_name in mode_names}
    for block in repetition_blocks:
        baseline = block["baseline"]
        for mode_name in mode_names:
            result = block[mode_name]
            reference_tokens = (
                canonical_baseline if mode_name == "baseline" else baseline["token_ids"]
            )
            result["matches_baseline"] = result["token_ids"] == reference_tokens
            result["paired_speedup_vs_baseline"] = _paired_speedup(result, baseline)
            grouped[mode_name].append(result)
    return grouped, baseline_deterministic


def _strict_parity_failures(
    rows: dict[str, dict[str, Any]],
    mode_names: list[str],
) -> list[str]:
    """Return explicitly exact modes that did not match target AR."""

    return [
        mode_name
        for mode_name in mode_names
        if MODES[mode_name].exact
        and not bool(rows[mode_name].get("all_match_baseline", False))
    ]


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


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


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
    raw_phase_timings = dict(result.get("phase_timings_us") or {})
    prefill_us = float(
        result.get("prefill_us")
        or raw_phase_timings.get("prefill", 0.0)
    )
    phase_timings_us = {
        name: float(raw_phase_timings.get(name, 0.0) or 0.0)
        for name in PHASE_TIMING_NAMES
    }
    # Baseline results expose prefill as a top-level value rather than a phase.
    phase_timings_us["prefill"] = prefill_us
    generation_tokens = int(result.get("generation_tokens", 0))
    decode_us = max(0.0, elapsed_us - prefill_us)
    token_ids = [int(token) for token in result.get("generated_token_ids", [])]
    cache_commit_mode = result.get("cache_commit_mode")
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
        "phase_timings_us": phase_timings_us,
        "cache_commit_mode": (
            str(cache_commit_mode) if cache_commit_mode is not None else None
        ),
        "rebuilt_target_tokens": int(result.get("rebuilt_target_tokens", 0) or 0),
        "exact_acceptance_corrections": int(
            result.get("exact_acceptance_corrections", 0) or 0
        ),
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
    aggregate = {
        f"median_{key}": statistics.median(float(row[key]) for row in repetitions)
        for key in numeric
    }
    aggregate["median_phase_timings_us"] = {
        name: statistics.median(
            float((row.get("phase_timings_us") or {}).get(name, 0.0))
            for row in repetitions
        )
        for name in PHASE_TIMING_NAMES
    }
    aggregate["median_rebuilt_target_tokens"] = statistics.median(
        int(row.get("rebuilt_target_tokens", 0) or 0) for row in repetitions
    )
    aggregate["median_exact_acceptance_corrections"] = statistics.median(
        int(row.get("exact_acceptance_corrections", 0) or 0)
        for row in repetitions
    )
    aggregate["cache_commit_modes"] = sorted(
        {
            str(mode)
            for row in repetitions
            if (mode := row.get("cache_commit_mode")) is not None
        }
    )
    aggregate["median_paired_speedup_vs_baseline"] = {}
    for metric in SPEEDUP_METRICS:
        paired_values = [
            float(value)
            for row in repetitions
            if (
                value := (row.get("paired_speedup_vs_baseline") or {}).get(metric)
            )
            is not None
        ]
        aggregate["median_paired_speedup_vs_baseline"][metric] = (
            statistics.median(paired_values) if paired_values else None
        )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="large")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260715,
        help="seed for the balanced per-repetition execution order",
    )
    parser.add_argument("--modes", default="baseline,dflash,pq4,tq4")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail when any mode marked exact differs from greedy baseline",
    )
    args = parser.parse_args()

    if args.reps < 1:
        parser.error("--reps must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    requested_modes = list(
        dict.fromkeys(name.strip() for name in args.modes.split(",") if name.strip())
    )
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
    warmup_schedule = _balanced_mode_schedule(
        requested_modes,
        args.warmup,
        seed=args.seed ^ 0x5A17,
    )
    execution_schedule = _balanced_mode_schedule(
        requested_modes,
        args.reps,
        seed=args.seed,
    )

    for warmup_index, mode_order in enumerate(warmup_schedule, start=1):
        print(
            f"[warmup] round={warmup_index}/{args.warmup} "
            f"order={','.join(mode_order)}",
            flush=True,
        )
        for mode_name in mode_order:
            _run_once(
                MODES[mode_name],
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=args.max_tokens,
            )

    repetition_blocks: list[dict[str, dict[str, Any]]] = []
    execution_index = 0
    for repetition, mode_order in enumerate(execution_schedule, start=1):
        print(
            f"[repetition] {repetition}/{args.reps} order={','.join(mode_order)}",
            flush=True,
        )
        block: dict[str, dict[str, Any]] = {}
        for position, mode_name in enumerate(mode_order, start=1):
            execution_index += 1
            result = _run_once(
                MODES[mode_name],
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=args.max_tokens,
            )
            result["repetition"] = repetition
            result["position_in_repetition"] = position
            result["execution_index"] = execution_index
            block[mode_name] = result
            print(
                f"  mode={mode_name} prefill={result['prefill_tps']:.1f} tok/s "
                f"decode={result['decode_tps']:.2f} tok/s "
                f"accept={result['acceptance_ratio']:.3f} "
                f"peak={result['peak_memory_gb']:.2f} GB",
                flush=True,
            )
        repetition_blocks.append(block)

    grouped_repetitions, baseline_deterministic = _organize_repetitions(
        requested_modes,
        repetition_blocks,
    )
    rows: dict[str, Any] = {}
    for mode_name in requested_modes:
        repetitions = grouped_repetitions[mode_name]
        rows[mode_name] = {
            "mode": asdict(MODES[mode_name]),
            "aggregate": _aggregate(repetitions),
            "all_match_baseline": all(row["matches_baseline"] for row in repetitions),
            "repetitions": repetitions,
        }
    rows["baseline"]["deterministic_across_repetitions"] = baseline_deterministic
    parity_failures = _strict_parity_failures(rows, requested_modes)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
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
            "seed": args.seed,
            "modes": requested_modes,
        },
        "warmup_order": [
            {"round": index, "modes": mode_order}
            for index, mode_order in enumerate(warmup_schedule, start=1)
        ],
        "execution_order": [
            {"repetition": index, "modes": mode_order}
            for index, mode_order in enumerate(execution_schedule, start=1)
        ],
        "checks": {
            "baseline_deterministic": baseline_deterministic,
            "exact_mode_parity": {
                mode_name: rows[mode_name]["all_match_baseline"]
                for mode_name in requested_modes
                if MODES[mode_name].exact
            },
            "strict_parity_failures": parity_failures,
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

    if args.strict_parity and parity_failures:
        print(
            "[error] exact mode parity failed: " + ", ".join(parity_failures),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
