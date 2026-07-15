"""Matched, reproducible Mio R&D harness for speculative decoding.

The benchmark loads one Qwen3 target and compares three greedy, lossless paths:
plain autoregressive decoding, mlx-dspark, and the native dflash-mlx runtime.
Every measured repetition is paired by prompt and uses a seeded Latin-rotation
execution order.  Candidate output is always truncated to ``max_tokens`` before
parity and throughput calculations, because some upstream speculative loops may
commit a final block that crosses the requested boundary.

This script is deliberately separate from product inference.  It records raw
evidence and applies conservative single-workload candidate gates; it never
changes Mio's runtime defaults or claims a global breakthrough from one model,
corpus, machine, or benchmark run.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
from typing import Any


SCHEMA_NAME = "mio.speculative-rd"
SCHEMA_VERSION = 2
CORPUS_VERSION = 1
MODES = ("baseline", "mlx-dspark", "dflash-mlx")
CANDIDATES = MODES[1:]
P95_PROBABILITY = 0.95

GLOBAL_BREAKTHROUGH_MISSING_EVIDENCE = (
    "held-out_corpora",
    "multiple_model_scales",
    "independent_hardware_replication",
    "independent_implementation_replication",
    "quality_and_task_success_non_regression",
    "sustained_load_and_concurrency",
)

DEFAULT_TARGET = "models/Qwen3-4B-8bit"
DEFAULT_DSPARK_DRAFT = "spd/dspark_qwen3_4b_block7"
DEFAULT_DFLASH_DRAFT = "spd/Qwen3-4B-DFlash-b16"

BUILTIN_CORPUS = (
    {
        "id": "python-refactor",
        "prompt": (
            "Refactor this Python function to be clear, deterministic, and linear-time. "
            "Explain the invariants, then return the complete implementation:\n\n"
            "def unique(items):\n    out = []\n    for item in items:\n"
            "        if item not in out:\n            out.append(item)\n    return out"
        ),
    },
    {
        "id": "debugging",
        "prompt": (
            "A service occasionally writes duplicate ledger entries after a timeout. "
            "Design a debugging plan and a minimal idempotency fix. Include the race "
            "condition, observability signals, tests, and rollback strategy."
        ),
    },
    {
        "id": "structured-json",
        "prompt": (
            "Return JSON only. Build a five-step release checklist. Each item must have "
            "the keys id, owner, precondition, action, verification, and rollback."
        ),
    },
    {
        "id": "copy-heavy",
        "prompt": (
            "Continue the following repeating sequence for sixteen more rows and then "
            "state the rule:\nalpha,beta,gamma,delta\nalpha,beta,gamma,delta\n"
            "alpha,beta,gamma,delta\nalpha,beta,gamma,delta"
        ),
    },
)


@dataclass(frozen=True)
class PromptCase:
    id: str
    prompt: str


@dataclass
class RawGeneration:
    token_ids: list[int]
    fallback: bool = False
    fallback_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


Generator = Callable[[list[int], int, int, Callable[[], None]], RawGeneration]


@dataclass
class BenchmarkRuntime:
    encode_prompt: Callable[[str, bool], list[int]]
    generators: dict[str, Generator]
    reset_peak_memory: Callable[[], None]
    get_peak_memory: Callable[[], int]
    metadata: dict[str, Any] = field(default_factory=dict)


def _balanced_mode_schedule(
    mode_names: Sequence[str],
    blocks: int,
    *,
    seed: int,
) -> list[list[str]]:
    """Return randomized Latin rotations, balanced in every complete block."""

    if blocks < 0:
        raise ValueError("blocks must be non-negative")
    if not mode_names:
        raise ValueError("at least one mode is required")
    if len(set(mode_names)) != len(mode_names):
        raise ValueError("mode names must be unique")

    rng = random.Random(seed)
    result: list[list[str]] = []
    width = len(mode_names)
    while len(result) < blocks:
        base = list(mode_names)
        rng.shuffle(base)
        rotations = list(range(width))
        rng.shuffle(rotations)
        for offset in rotations:
            result.append(base[offset:] + base[:offset])
            if len(result) == blocks:
                break
    return result


def _slug(value: str, index: int) -> str:
    cleaned = "-".join(part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split())
    return cleaned[:64] or f"prompt-{index}"


def _coerce_prompt_cases(values: Any) -> list[PromptCase]:
    if isinstance(values, dict) and "prompts" in values:
        values = values["prompts"]
    if not isinstance(values, list):
        raise ValueError("prompt corpus must be a JSON list or an object with a prompts list")

    cases: list[PromptCase] = []
    seen: set[str] = set()
    for index, value in enumerate(values, start=1):
        if isinstance(value, str):
            prompt = value.strip()
            identifier = _slug(prompt[:48], index)
        elif isinstance(value, dict):
            prompt = str(value.get("prompt", "")).strip()
            identifier = _slug(str(value.get("id", "")), index)
        else:
            raise ValueError(f"prompt {index} must be a string or object")
        if not prompt:
            raise ValueError(f"prompt {index} is empty")
        original = identifier
        suffix = 2
        while identifier in seen:
            identifier = f"{original}-{suffix}"
            suffix += 1
        seen.add(identifier)
        cases.append(PromptCase(identifier, prompt))
    if not cases:
        raise ValueError("prompt corpus is empty")
    return cases


def _load_corpus(path: Path | None) -> list[PromptCase]:
    if path is None:
        return [PromptCase(str(row["id"]), str(row["prompt"])) for row in BUILTIN_CORPUS]

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _coerce_prompt_cases(json.loads(text))
    if suffix == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
        return _coerce_prompt_cases(values)
    blocks = [block.strip() for block in text.split("\n---\n") if block.strip()]
    return _coerce_prompt_cases(blocks)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap_ci(
    pairs: Sequence[dict[str, Any]],
    *,
    metric_key: str,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Cluster-bootstrap paired ratios, keeping prompt repetitions together."""

    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    clusters: dict[str, list[float]] = {}
    expected = sum(bool(pair.get("eligible")) for pair in pairs)
    for pair in pairs:
        if not pair.get("eligible") or pair.get("prompt_id") is None:
            continue
        value = pair.get(metric_key)
        if value is None:
            continue
        value_float = float(value)
        if not math.isfinite(value_float) or value_float <= 0.0:
            continue
        clusters.setdefault(str(pair["prompt_id"]), []).append(value_float)

    clean = [value for cluster in clusters.values() for value in cluster]
    status = "unavailable" if not clean else "measured" if len(clean) == expected else "partial"
    result: dict[str, Any] = {
        "estimand": "median_paired_speedup_cluster_bootstrap_by_prompt",
        "metric_key": metric_key,
        "cluster_key": "prompt_id",
        "expected_pairs": expected,
        "n_pairs": len(clean),
        "n_clusters": len(clusters),
        "measurement_status": status,
        "point_estimate": statistics.median(clean) if clean else None,
        "confidence_level": confidence,
        "lower": None,
        "upper": None,
        "bootstrap_samples": samples,
    }
    if not clean:
        return result

    rng = random.Random(seed)
    cluster_ids = list(clusters)
    cluster_count = len(cluster_ids)
    estimates: list[float] = []
    for _ in range(samples):
        resampled = [
            value for _ in range(cluster_count) for value in clusters[cluster_ids[rng.randrange(cluster_count)]]
        ]
        estimates.append(statistics.median(resampled))
    alpha = (1.0 - confidence) / 2.0
    result["lower"] = _percentile(estimates, alpha)
    result["upper"] = _percentile(estimates, 1.0 - alpha)
    return result


def _paired_percentile_ratio_ci(
    pairs: Sequence[dict[str, Any]],
    *,
    baseline_key: str,
    candidate_key: str,
    probability: float,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Estimate a paired percentile speedup without inventing missing values.

    The estimand is ``baseline percentile / candidate percentile``. Bootstrap
    resampling keeps each baseline/candidate observation paired and all repeated
    observations for one prompt in the same cluster. Zero, missing, non-finite,
    and ineligible measurements are excluded and explicitly counted.
    """

    if not 0.0 < probability < 1.0:
        raise ValueError("percentile probability must be between zero and one")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    clusters: dict[str, list[tuple[float, float]]] = {}
    for pair in pairs:
        if not pair.get("eligible") or pair.get("prompt_id") is None:
            continue
        baseline_value = pair.get(baseline_key)
        candidate_value = pair.get(candidate_key)
        if baseline_value is None or candidate_value is None:
            continue
        baseline_float = float(baseline_value)
        candidate_float = float(candidate_value)
        if (
            not math.isfinite(baseline_float)
            or not math.isfinite(candidate_float)
            or baseline_float <= 0.0
            or candidate_float <= 0.0
        ):
            continue
        clusters.setdefault(str(pair["prompt_id"]), []).append((baseline_float, candidate_float))

    expected = sum(bool(pair.get("eligible")) for pair in pairs)
    measured = [value for cluster in clusters.values() for value in cluster]
    status = "unavailable" if not measured else "measured" if len(measured) == expected else "partial"
    result: dict[str, Any] = {
        "estimand": f"paired_p{probability * 100:g}_baseline_over_candidate",
        "probability": probability,
        "expected_pairs": expected,
        "n_pairs": len(measured),
        "n_clusters": len(clusters),
        "cluster_key": "prompt_id",
        "measurement_status": status,
        "baseline_percentile": None,
        "candidate_percentile": None,
        "point_estimate": None,
        "confidence_level": confidence,
        "lower": None,
        "upper": None,
        "bootstrap_samples": samples,
    }
    if not measured:
        return result

    baseline_values = [item[0] for item in measured]
    candidate_values = [item[1] for item in measured]
    baseline_percentile = _percentile(baseline_values, probability)
    candidate_percentile = _percentile(candidate_values, probability)
    result.update(
        {
            "baseline_percentile": baseline_percentile,
            "candidate_percentile": candidate_percentile,
            "point_estimate": baseline_percentile / candidate_percentile,
        }
    )

    rng = random.Random(seed)
    cluster_ids = list(clusters)
    cluster_count = len(cluster_ids)
    estimates: list[float] = []
    for _ in range(samples):
        resampled = [
            value for _ in range(cluster_count) for value in clusters[cluster_ids[rng.randrange(cluster_count)]]
        ]
        baseline_sample = _percentile([item[0] for item in resampled], probability)
        candidate_sample = _percentile([item[1] for item in resampled], probability)
        estimates.append(baseline_sample / candidate_sample)
    alpha = (1.0 - confidence) / 2.0
    result["lower"] = _percentile(estimates, alpha)
    result["upper"] = _percentile(estimates, 1.0 - alpha)
    return result


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token)) for token in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _run_once(
    runtime: BenchmarkRuntime,
    mode: str,
    *,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute and normalize one run without coupling tests to model packages."""

    first_output_at: float | None = None

    def mark_first_output() -> None:
        nonlocal first_output_at
        if first_output_at is None:
            first_output_at = clock()

    runtime.reset_peak_memory()
    started = clock()
    raw = runtime.generators[mode](prompt_ids, max_tokens, seed, mark_first_output)
    finished = clock()

    raw_tokens = [int(token) for token in raw.token_ids]
    token_ids = raw_tokens[:max_tokens]
    raw_count = len(raw_tokens)
    normalized_count = len(token_ids)
    wall_seconds = max(0.0, finished - started)
    ttft_seconds = max(0.0, min(finished, first_output_at) - started) if first_output_at is not None else None
    decode_seconds = max(0.0, wall_seconds - ttft_seconds) if ttft_seconds is not None else None
    decode_tokens = max(0, normalized_count - 1)
    decode_tps = (
        decode_tokens / decode_seconds
        if decode_seconds is not None and decode_seconds > 0.0 and decode_tokens > 0
        else None
    )
    peak_bytes = int(runtime.get_peak_memory() or 0)

    return {
        "status": "ok",
        "mode": mode,
        "prompt_tokens": len(prompt_ids),
        "requested_tokens": max_tokens,
        "raw_token_count": raw_count,
        "normalized_token_count": normalized_count,
        "overshoot_tokens": max(0, raw_count - max_tokens),
        "shortfall_tokens": max(0, max_tokens - normalized_count),
        "exact_normalized_length": normalized_count == max_tokens,
        "normalization": "prefix_truncate_to_requested_max",
        "timing_includes_upstream_overshoot": raw_count > max_tokens,
        "token_ids": token_ids,
        "token_hash": _token_hash(token_ids),
        "wall_seconds": wall_seconds,
        "ttft_seconds": ttft_seconds,
        "ttft_observed": first_output_at is not None,
        "ttft_definition": "call_start_to_first_output_event",
        "decode_seconds": decode_seconds,
        "decode_tokens": decode_tokens,
        "decode_tps": decode_tps,
        "normalized_end_to_end_tps": (normalized_count / wall_seconds if wall_seconds > 0.0 else None),
        "peak_memory_bytes": peak_bytes,
        "peak_memory_gb": peak_bytes / 1e9,
        "fallback": bool(raw.fallback),
        "fallback_reason": raw.fallback_reason,
        "diagnostics": raw.diagnostics,
    }


def _error_run(
    mode: str,
    *,
    prompt_tokens: int,
    max_tokens: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "status": "error",
        "mode": mode,
        "prompt_tokens": prompt_tokens,
        "requested_tokens": max_tokens,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "fallback": False,
        "exact_normalized_length": False,
        "ttft_observed": False,
    }


def _run_has_valid_timing(run: dict[str, Any] | None) -> bool:
    """Return true only for complete, finite, positive TTFT/decode timing."""

    if not run or not run.get("ttft_observed"):
        return False
    try:
        ttft_seconds = float(run["ttft_seconds"])
        decode_seconds = float(run["decode_seconds"])
        decode_tps = float(run["decode_tps"])
        wall_seconds = float(run["wall_seconds"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        all(math.isfinite(value) and value > 0.0 for value in (ttft_seconds, decode_seconds, decode_tps, wall_seconds))
        and wall_seconds > ttft_seconds
        and math.isclose(wall_seconds, ttft_seconds + decode_seconds, rel_tol=1e-6, abs_tol=1e-9)
    )


def _pair_runs(
    runs: Sequence[dict[str, Any]],
    *,
    prompt_count: int,
    repetitions: int,
) -> dict[str, list[dict[str, Any]]]:
    by_key = {(str(run["prompt_id"]), int(run["repetition"]), str(run["mode"])): run for run in runs}
    prompt_ids = list(dict.fromkeys(str(run["prompt_id"]) for run in runs))
    paired: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in CANDIDATES}
    for prompt_id in prompt_ids[:prompt_count]:
        for repetition in range(1, repetitions + 1):
            baseline = by_key.get((prompt_id, repetition, "baseline"))
            for candidate in CANDIDATES:
                contender = by_key.get((prompt_id, repetition, candidate))
                baseline_ok = bool(
                    baseline and baseline.get("status") == "ok" and baseline.get("exact_normalized_length")
                )
                candidate_ok = bool(
                    contender and contender.get("status") == "ok" and contender.get("exact_normalized_length")
                )
                baseline_timing_valid = _run_has_valid_timing(baseline)
                candidate_timing_valid = _run_has_valid_timing(contender)
                eligible = baseline_ok and candidate_ok and baseline_timing_valid and candidate_timing_valid
                parity = bool(eligible and contender["token_ids"] == baseline["token_ids"])

                def ratio(numerator: Any, denominator: Any) -> float | None:
                    if numerator is None or denominator is None:
                        return None
                    numerator_f = float(numerator)
                    denominator_f = float(denominator)
                    return numerator_f / denominator_f if denominator_f > 0.0 else None

                paired[candidate].append(
                    {
                        "prompt_id": prompt_id,
                        "repetition": repetition,
                        "baseline_run_id": baseline.get("run_id") if baseline else None,
                        "candidate_run_id": contender.get("run_id") if contender else None,
                        "eligible": eligible,
                        "baseline_timing_valid": baseline_timing_valid,
                        "candidate_timing_valid": candidate_timing_valid,
                        "token_parity": parity,
                        "ttft_speedup": (
                            ratio(baseline.get("ttft_seconds"), contender.get("ttft_seconds")) if eligible else None
                        ),
                        "decode_speedup": (
                            ratio(contender.get("decode_tps"), baseline.get("decode_tps")) if eligible else None
                        ),
                        "end_to_end_speedup": (
                            ratio(baseline.get("wall_seconds"), contender.get("wall_seconds")) if eligible else None
                        ),
                        "baseline_ttft_seconds": baseline.get("ttft_seconds") if baseline_ok else None,
                        "candidate_ttft_seconds": contender.get("ttft_seconds") if candidate_ok else None,
                        "baseline_decode_seconds": baseline.get("decode_seconds") if baseline_ok else None,
                        "candidate_decode_seconds": contender.get("decode_seconds") if candidate_ok else None,
                        "baseline_peak_memory_bytes": baseline.get("peak_memory_bytes") if baseline_ok else None,
                        "candidate_peak_memory_bytes": contender.get("peak_memory_bytes") if candidate_ok else None,
                        "candidate_fallback": bool(contender and contender.get("fallback")),
                    }
                )
    return paired


def _stable_seed(seed: int, *parts: str) -> int:
    suffix = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()[:8]
    return seed ^ int.from_bytes(suffix, "big")


def _candidate_analysis(
    candidate: str,
    pairs: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    min_ttft_speedup: float,
    min_decode_speedup: float,
    ci_floor: float,
    required_parity: float,
    min_pairs: int,
    min_distinct_prompts: int = 4,
    max_p95_latency_regression: float = 0.0,
    max_peak_memory_regression: float = 0.05,
    baseline_deterministic: bool = True,
    strict_passed: bool = True,
) -> dict[str, Any]:
    expected = len(pairs)
    parity_count = sum(bool(pair["token_parity"]) for pair in pairs)
    parity_rate = parity_count / expected if expected else 0.0
    fallback_count = sum(bool(pair["candidate_fallback"]) for pair in pairs)
    eligible_count = sum(bool(pair["eligible"]) for pair in pairs)
    ttft = _paired_bootstrap_ci(
        pairs,
        metric_key="ttft_speedup",
        samples=bootstrap_samples,
        confidence=confidence,
        seed=_stable_seed(seed, candidate, "ttft"),
    )
    decode = _paired_bootstrap_ci(
        pairs,
        metric_key="decode_speedup",
        samples=bootstrap_samples,
        confidence=confidence,
        seed=_stable_seed(seed, candidate, "decode"),
    )
    end_to_end = _paired_bootstrap_ci(
        pairs,
        metric_key="end_to_end_speedup",
        samples=bootstrap_samples,
        confidence=confidence,
        seed=_stable_seed(seed, candidate, "end-to-end"),
    )

    ttft_p95 = _paired_percentile_ratio_ci(
        pairs,
        baseline_key="baseline_ttft_seconds",
        candidate_key="candidate_ttft_seconds",
        probability=P95_PROBABILITY,
        samples=bootstrap_samples,
        confidence=confidence,
        seed=_stable_seed(seed, candidate, "ttft-p95"),
    )
    decode_p95 = _paired_percentile_ratio_ci(
        pairs,
        baseline_key="baseline_decode_seconds",
        candidate_key="candidate_decode_seconds",
        probability=P95_PROBABILITY,
        samples=bootstrap_samples,
        confidence=confidence,
        seed=_stable_seed(seed, candidate, "decode-p95"),
    )
    peak_memory_p95 = _paired_percentile_ratio_ci(
        pairs,
        baseline_key="baseline_peak_memory_bytes",
        candidate_key="candidate_peak_memory_bytes",
        probability=P95_PROBABILITY,
        samples=bootstrap_samples,
        confidence=confidence,
        seed=_stable_seed(seed, candidate, "peak-memory-p95"),
    )

    p95_latency_floor = 1.0 / (1.0 + max_p95_latency_regression)
    peak_memory_floor = 1.0 / (1.0 + max_peak_memory_regression)

    required_conditions = {
        "strict_run_checks": strict_passed,
        "baseline_deterministic": baseline_deterministic,
        "minimum_paired_samples": min(ttft["n_pairs"], decode["n_pairs"]) >= min_pairs,
        "minimum_distinct_prompts": min(ttft["n_clusters"], decode["n_clusters"]) >= min_distinct_prompts,
        "parity_requirement": parity_rate >= required_parity,
        "zero_fallbacks": fallback_count == 0,
        "all_runs_eligible": eligible_count == expected,
        "ttft_point_estimate": bool(ttft["point_estimate"] is not None and ttft["point_estimate"] >= min_ttft_speedup),
        "ttft_ci_lower_bound": bool(ttft["lower"] is not None and ttft["lower"] > ci_floor),
        "decode_point_estimate": bool(
            decode["point_estimate"] is not None and decode["point_estimate"] >= min_decode_speedup
        ),
        "decode_ci_lower_bound": bool(decode["lower"] is not None and decode["lower"] > ci_floor),
        "ttft_p95_minimum_samples": ttft_p95["n_pairs"] >= min_pairs,
        "ttft_p95_point_non_regression": bool(
            ttft_p95["point_estimate"] is not None and ttft_p95["point_estimate"] >= p95_latency_floor
        ),
        "ttft_p95_ci_non_regression": bool(ttft_p95["lower"] is not None and ttft_p95["lower"] >= p95_latency_floor),
        "decode_p95_minimum_samples": decode_p95["n_pairs"] >= min_pairs,
        "decode_p95_point_non_regression": bool(
            decode_p95["point_estimate"] is not None and decode_p95["point_estimate"] >= p95_latency_floor
        ),
        "decode_p95_ci_non_regression": bool(
            decode_p95["lower"] is not None and decode_p95["lower"] >= p95_latency_floor
        ),
    }
    memory_gate_applied = (
        peak_memory_p95["measurement_status"] == "measured" and peak_memory_p95["n_pairs"] >= min_pairs
    )
    conditional_conditions: dict[str, bool | None] = {
        "peak_memory_p95_point_non_regression": (
            bool(
                peak_memory_p95["point_estimate"] is not None and peak_memory_p95["point_estimate"] >= peak_memory_floor
            )
            if memory_gate_applied
            else None
        ),
        "peak_memory_p95_ci_non_regression": (
            bool(peak_memory_p95["lower"] is not None and peak_memory_p95["lower"] >= peak_memory_floor)
            if memory_gate_applied
            else None
        ),
    }
    workload_candidate = all(required_conditions.values()) and all(
        condition is not False for condition in conditional_conditions.values()
    )
    return {
        "candidate": candidate,
        "expected_pairs": expected,
        "eligible_pairs": eligible_count,
        "parity_count": parity_count,
        "parity_rate": parity_rate,
        "fallback_count": fallback_count,
        "metrics": {
            "ttft_speedup": ttft,
            "decode_speedup": decode,
            "end_to_end_speedup": end_to_end,
            "ttft_p95_latency_speedup": ttft_p95,
            "decode_p95_latency_speedup": decode_p95,
            "peak_memory_p95_efficiency": peak_memory_p95,
        },
        "criterion": {
            "minimum_ttft_speedup": min_ttft_speedup,
            "minimum_decode_speedup": min_decode_speedup,
            "confidence_interval_lower_bound_strictly_above": ci_floor,
            "required_parity_rate": required_parity,
            "minimum_paired_samples": min_pairs,
            "minimum_distinct_prompts": min_distinct_prompts,
            "bootstrap_resampling_unit": "prompt_id_cluster_with_all_repetitions",
            "maximum_p95_latency_regression": max_p95_latency_regression,
            "minimum_p95_latency_speedup": p95_latency_floor,
            "maximum_peak_memory_regression": max_peak_memory_regression,
            "minimum_peak_memory_efficiency": peak_memory_floor,
            "zero_fallbacks_required": True,
            "deterministic_baseline_required": True,
            "memory_gate_policy": "apply_only_when_all_eligible_pairs_have_positive_measurements",
        },
        "conditions": required_conditions,
        "conditional_conditions": conditional_conditions,
        "evidence_completeness": {
            "tail_latency": (
                ttft_p95["measurement_status"] == "measured" and decode_p95["measurement_status"] == "measured"
            ),
            "peak_memory": peak_memory_p95["measurement_status"] == "measured",
            "peak_memory_gate_applied": memory_gate_applied,
        },
        "claim_scope": "single_model_single_machine_single_corpus_workload",
        "candidate_workload_speedup": workload_candidate,
        "global_breakthrough_evaluable": False,
        "global_breakthrough": False,
        "breakthrough": False,
        "breakthrough_field_status": "deprecated_always_false_use_candidate_workload_speedup",
        "pairs": list(pairs),
    }


def _baseline_determinism(runs: Sequence[dict[str, Any]], prompts: Sequence[PromptCase]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for prompt in prompts:
        relevant = [run for run in runs if run["mode"] == "baseline" and run["prompt_id"] == prompt.id]
        valid = [run for run in relevant if run.get("status") == "ok"]
        checks[prompt.id] = bool(
            valid and len(valid) == len(relevant) and all(run["token_ids"] == valid[0]["token_ids"] for run in valid)
        )
    return checks


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_value(arguments: Sequence[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _command_value(arguments: Sequence[str]) -> str | None:
    try:
        value = subprocess.run(
            list(arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _model_provenance(reference: str) -> dict[str, Any]:
    path = Path(reference).expanduser()
    local = path.exists()
    config_hash = None
    if local:
        config_path = path / "config.json" if path.is_dir() else path
        if config_path.is_file():
            config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return {
        "reference": path.name if path.is_absolute() else reference,
        "local": local,
        "config_sha256": config_hash,
    }


def _provenance(args: argparse.Namespace, runtime: BenchmarkRuntime) -> dict[str, Any]:
    try:
        import mlx.core as mx

        mlx_device = _jsonable(mx.device_info())
    except Exception:  # pragma: no cover - only a degraded non-MLX environment
        mlx_device = None
    git_status = _git_value(["status", "--porcelain"])
    return {
        "git": {
            "revision": _git_value(["rev-parse", "HEAD"]),
            "dirty": bool(git_status) if git_status is not None else None,
        },
        "software": {
            "python": platform.python_version(),
            "mlx": _package_version("mlx"),
            "mlx-lm": _package_version("mlx-lm"),
            "mlx-dspark": _package_version("mlx-dspark"),
            "dflash-mlx": _package_version("dflash-mlx"),
            "huggingface-hub": _package_version("huggingface-hub"),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_brand": _command_value(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_bytes": _command_value(["sysctl", "-n", "hw.memsize"]),
            "mlx_device": mlx_device,
        },
        "models": {
            "target": _model_provenance(args.model),
            "mlx-dspark_draft": _model_provenance(args.dspark_draft),
            "dflash-mlx_draft": _model_provenance(args.dflash_draft),
        },
        "runtime": _jsonable(runtime.metadata),
    }


def _collect_native_events(
    stream: Iterable[Any],
    *,
    token_event_type: type,
    summary_event_type: type,
    on_first_output: Callable[[], None],
    baseline: bool,
) -> RawGeneration:
    summary = None
    observed_tokens: list[int] = []
    fallback = False
    fallback_reason = None
    try:
        for event in stream:
            if isinstance(event, token_event_type):
                if not observed_tokens:
                    on_first_output()
                observed_tokens.append(int(event.token_id))
                if not baseline and bool(getattr(event, "fallback_ar", False)):
                    fallback = True
                    fallback_reason = getattr(event, "fallback_reason", None)
            elif isinstance(event, summary_event_type):
                summary = event
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    if summary is None:
        raise RuntimeError("native MLX stream ended without a summary event")
    token_ids = [int(token) for token in summary.generated_token_ids]
    if observed_tokens and observed_tokens != token_ids[: len(observed_tokens)]:
        raise RuntimeError("token events disagree with the native summary")
    if not baseline and bool(getattr(summary, "fallback_ar", False)):
        fallback = True
        fallback_reason = getattr(summary, "fallback_reason", None)
    return RawGeneration(
        token_ids=token_ids,
        fallback=fallback,
        fallback_reason=fallback_reason,
        diagnostics={
            "engine_elapsed_us": float(summary.elapsed_us),
            "engine_generation_tokens": int(summary.generation_tokens),
            "acceptance_ratio": float(summary.acceptance_ratio),
            "cycles_completed": int(summary.cycles_completed),
            "tokens_per_cycle": float(getattr(summary, "tokens_per_cycle", 0.0)),
            "phase_timings_us": _jsonable(summary.phase_timings_us),
        },
    )


def _load_runtime(args: argparse.Namespace) -> BenchmarkRuntime:
    """Load the one shared target and both draft models (deferred from import)."""

    import mlx.core as mx
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    from dflash_mlx.engine.fallback import stream_baseline_generate
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context
    from mlx_dspark import encode_prompt, speculative_generate
    from mlx_dspark.load import apply_wired_limit, load_drafter
    from mlx_dspark.target import Target

    apply_wired_limit()
    dflash_context = build_offline_runtime_context(
        verify_len_cap=args.dflash_verify_cap,
        verify_mode=args.dflash_verify_mode,
    )
    bundle = load_runtime_bundle(
        model_ref=args.model,
        draft_ref=args.dflash_draft,
        draft_quant=args.dflash_draft_quant,
        verify_config=dflash_context.verify,
        lazy=True,
    )
    dspark_target = Target(bundle.target_model, bundle.tokenizer)
    dspark_target.verify_tap()
    dspark_quantized = args.dspark_draft_bits > 0
    dspark_draft, dspark_config = load_drafter(
        args.dspark_draft,
        quantize=dspark_quantized,
        bits=max(2, args.dspark_draft_bits),
        group_size=args.dspark_draft_group_size,
    )
    mx.eval(bundle.target_model.parameters(), bundle.draft_model.parameters(), dspark_draft.parameters())
    stop_token_ids = get_stop_token_ids(bundle.tokenizer)

    def baseline_generator(
        prompt_ids: list[int],
        max_tokens: int,
        seed: int,
        on_first_output: Callable[[], None],
    ) -> RawGeneration:
        mx.random.seed(seed)
        stream = stream_baseline_generate(
            target_model=bundle.target_model,
            target_ops=bundle.target_ops,
            tokenizer=bundle.tokenizer,
            prompt="",
            max_new_tokens=max_tokens,
            use_chat_template=False,
            stop_token_ids=stop_token_ids,
            prompt_tokens_override=prompt_ids,
        )
        result = _collect_native_events(
            stream,
            token_event_type=TokenEvent,
            summary_event_type=SummaryEvent,
            on_first_output=on_first_output,
            baseline=True,
        )
        result.diagnostics["engine"] = "dflash_mlx.stream_baseline_generate"
        return result

    def dspark_generator(
        prompt_ids: list[int],
        max_tokens: int,
        seed: int,
        on_first_output: Callable[[], None],
    ) -> RawGeneration:
        result = speculative_generate(
            dspark_target,
            bundle.tokenizer,
            dspark_draft,
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            max_draft_tokens=(args.dspark_max_draft_tokens if args.dspark_max_draft_tokens > 0 else None),
            lookup_drafts=args.dspark_lookup,
            temperature=0.0,
            seed=seed,
            apply_chat_template=False,
            on_text=lambda _text: on_first_output(),
        )
        return RawGeneration(
            token_ids=[int(token) for token in result.token_ids],
            diagnostics={
                "engine": "mlx_dspark.speculative_generate",
                "engine_elapsed_seconds": float(result.seconds),
                "engine_generation_tokens": int(result.num_tokens),
                "rounds": int(result.num_rounds),
                "target_forwards": int(result.target_forwards),
                "mean_accept_length": float(result.mean_accept_len),
                "accept_lengths": [int(value) for value in result.accept_lengths],
                "lookup_rounds": int(result.lookup_rounds),
            },
        )

    def dflash_generator(
        prompt_ids: list[int],
        max_tokens: int,
        seed: int,
        on_first_output: Callable[[], None],
    ) -> RawGeneration:
        mx.random.seed(seed)
        stream = stream_dflash_generate(
            target_model=bundle.target_model,
            target_ops=bundle.target_ops,
            tokenizer=bundle.tokenizer,
            draft_model=bundle.draft_model,
            draft_backend=bundle.draft_backend,
            prompt="",
            max_new_tokens=max_tokens,
            use_chat_template=False,
            block_tokens=args.dflash_block_tokens,
            stop_token_ids=stop_token_ids,
            prompt_tokens_override=prompt_ids,
            publish_generation_snapshot=False,
            runtime_context=dflash_context,
        )
        result = _collect_native_events(
            stream,
            token_event_type=TokenEvent,
            summary_event_type=SummaryEvent,
            on_first_output=on_first_output,
            baseline=False,
        )
        result.diagnostics["engine"] = "dflash_mlx.stream_dflash_generate"
        return result

    config_summary = {
        "block_size": int(getattr(dspark_config, "block_size", 0)),
        "target_layer_ids": [int(value) for value in getattr(dspark_config, "target_layer_ids", ())],
    }
    return BenchmarkRuntime(
        encode_prompt=lambda prompt, use_chat: list(encode_prompt(bundle.tokenizer, prompt, use_chat=use_chat)),
        generators={
            "baseline": baseline_generator,
            "mlx-dspark": dspark_generator,
            "dflash-mlx": dflash_generator,
        },
        reset_peak_memory=mx.reset_peak_memory,
        get_peak_memory=mx.get_peak_memory,
        metadata={
            "shared_target_instance": True,
            "target_family": bundle.target_meta.get("target_family"),
            "dflash_verify_mode": args.dflash_verify_mode,
            "dflash_effective_draft_quant": bundle.effective_draft_quant,
            "dspark_draft_quantized": dspark_quantized,
            "dspark_config": config_summary,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_TARGET, help="target model path or HF id")
    parser.add_argument(
        "--dspark-draft",
        default=DEFAULT_DSPARK_DRAFT,
        help="matched mlx-dspark draft path or HF id",
    )
    parser.add_argument(
        "--dflash-draft",
        default=DEFAULT_DFLASH_DRAFT,
        help="matched dflash-mlx draft path or HF id",
    )
    parser.add_argument("--corpus-file", type=Path, help="optional .json, .jsonl, or --- separated text")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1, help="warmup blocks per prompt")
    parser.add_argument("--reps", type=int, default=3, help="measured paired blocks per prompt")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dspark-max-draft-tokens", type=int, default=2, help="0 uses the full block")
    parser.add_argument("--dspark-lookup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dspark-draft-bits", type=int, choices=(0, 2, 4, 8), default=4)
    parser.add_argument("--dspark-draft-group-size", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--dflash-draft-quant", default="w4:gs64")
    parser.add_argument("--dflash-block-tokens", type=int)
    parser.add_argument("--dflash-verify-cap", type=int)
    parser.add_argument(
        "--dflash-verify-mode",
        choices=("dflash", "adaptive", "ddtree", "off"),
        default="dflash",
    )
    parser.add_argument("--min-ttft-speedup", type=float, default=1.05)
    parser.add_argument("--min-decode-speedup", type=float, default=1.05)
    parser.add_argument("--ci-floor", type=float, default=1.0)
    parser.add_argument("--required-parity", type=float, default=1.0)
    parser.add_argument("--min-pairs", type=int, default=9)
    parser.add_argument(
        "--min-distinct-prompts",
        type=int,
        default=4,
        help="minimum independent prompt clusters required by confidence gates",
    )
    parser.add_argument(
        "--max-p95-latency-regression",
        type=float,
        default=0.0,
        help="maximum allowed p95 TTFT/decode latency regression (0 requires non-regression)",
    )
    parser.add_argument(
        "--max-peak-memory-regression",
        type=float,
        default=0.05,
        help="maximum allowed p95 peak-memory regression when complete measurements exist",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="return non-zero on execution, fallback, timing, determinism, or parity failures",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2 to measure steady-state decode")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between zero and one")
    if args.min_ttft_speedup <= 0.0 or args.min_decode_speedup <= 0.0:
        parser.error("speedup thresholds must be positive")
    if args.ci_floor <= 0.0:
        parser.error("--ci-floor must be positive")
    if not 0.0 < args.required_parity <= 1.0:
        parser.error("--required-parity must be in (0, 1]")
    if args.min_pairs < 1:
        parser.error("--min-pairs must be positive")
    if args.min_distinct_prompts < 2:
        parser.error("--min-distinct-prompts must be at least 2")
    if args.max_p95_latency_regression < 0.0:
        parser.error("--max-p95-latency-regression must be non-negative")
    if args.max_peak_memory_regression < 0.0:
        parser.error("--max-peak-memory-regression must be non-negative")
    for name in ("dspark_max_draft_tokens", "dflash_block_tokens", "dflash_verify_cap"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    prompts = _load_corpus(args.corpus_file)

    print(
        f"[load] target={args.model} dspark={args.dspark_draft} dflash={args.dflash_draft}",
        flush=True,
    )
    load_started = time.perf_counter()
    runtime = _load_runtime(args)
    load_seconds = time.perf_counter() - load_started
    encoded = {prompt.id: runtime.encode_prompt(prompt.prompt, args.chat_template) for prompt in prompts}

    warmup_jobs = [(prompt, round_index) for prompt in prompts for round_index in range(1, args.warmup + 1)]
    warmup_order = _balanced_mode_schedule(MODES, len(warmup_jobs), seed=args.seed ^ 0x5A17)
    warmup_failures: list[dict[str, Any]] = []
    for block_index, ((prompt, round_index), order) in enumerate(zip(warmup_jobs, warmup_order), start=1):
        print(
            f"[warmup] {block_index}/{len(warmup_jobs)} prompt={prompt.id} order={','.join(order)}",
            flush=True,
        )
        call_seed = args.seed ^ 0xA11CE ^ (block_index << 8)
        for mode in order:
            try:
                _run_once(
                    runtime,
                    mode,
                    prompt_ids=encoded[prompt.id],
                    max_tokens=args.max_tokens,
                    seed=call_seed,
                )
            except Exception as exc:  # noqa: BLE001 - preserve all experimental failures
                warmup_failures.append(
                    {
                        "prompt_id": prompt.id,
                        "round": round_index,
                        "mode": mode,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    measured_jobs = [(prompt, repetition) for prompt in prompts for repetition in range(1, args.reps + 1)]
    execution_order = _balanced_mode_schedule(MODES, len(measured_jobs), seed=args.seed)
    runs: list[dict[str, Any]] = []
    execution_index = 0
    for block_index, ((prompt, repetition), order) in enumerate(zip(measured_jobs, execution_order), start=1):
        print(
            f"[measure] {block_index}/{len(measured_jobs)} prompt={prompt.id} rep={repetition} order={','.join(order)}",
            flush=True,
        )
        call_seed = args.seed ^ (block_index << 12) ^ repetition
        for position, mode in enumerate(order, start=1):
            execution_index += 1
            try:
                run = _run_once(
                    runtime,
                    mode,
                    prompt_ids=encoded[prompt.id],
                    max_tokens=args.max_tokens,
                    seed=call_seed,
                )
            except Exception as exc:  # noqa: BLE001 - the JSON must retain experimental failures
                run = _error_run(
                    mode,
                    prompt_tokens=len(encoded[prompt.id]),
                    max_tokens=args.max_tokens,
                    exc=exc,
                )
            run.update(
                {
                    "run_id": f"run-{execution_index:04d}",
                    "prompt_id": prompt.id,
                    "repetition": repetition,
                    "position_in_block": position,
                    "execution_index": execution_index,
                    "call_seed": call_seed,
                }
            )
            runs.append(run)

    paired = _pair_runs(runs, prompt_count=len(prompts), repetitions=args.reps)
    deterministic = _baseline_determinism(runs, prompts)
    execution_failures = [run["run_id"] for run in runs if run.get("status") != "ok"]
    fallback_runs = [run["run_id"] for run in runs if run.get("fallback")]
    length_failures = [
        run["run_id"] for run in runs if run.get("status") == "ok" and not run.get("exact_normalized_length")
    ]
    timing_failures = [run["run_id"] for run in runs if run.get("status") == "ok" and not _run_has_valid_timing(run)]
    parity_failures = [
        f"{candidate}:{pair['prompt_id']}:{pair['repetition']}"
        for candidate, candidate_pairs in paired.items()
        for pair in candidate_pairs
        if not pair["token_parity"]
    ]
    strict_passed = not any(
        (
            warmup_failures,
            execution_failures,
            fallback_runs,
            length_failures,
            timing_failures,
            parity_failures,
        )
    ) and all(deterministic.values())
    analyses = {
        candidate: _candidate_analysis(
            candidate,
            paired[candidate],
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
            min_ttft_speedup=args.min_ttft_speedup,
            min_decode_speedup=args.min_decode_speedup,
            ci_floor=args.ci_floor,
            required_parity=args.required_parity,
            min_pairs=args.min_pairs,
            min_distinct_prompts=args.min_distinct_prompts,
            max_p95_latency_regression=args.max_p95_latency_regression,
            max_peak_memory_regression=args.max_peak_memory_regression,
            baseline_deterministic=all(deterministic.values()),
            strict_passed=strict_passed,
        )
        for candidate in CANDIDATES
    }

    payload = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": _provenance(args, runtime),
        "configuration": {
            "max_tokens": args.max_tokens,
            "warmup_blocks_per_prompt": args.warmup,
            "repetitions_per_prompt": args.reps,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "confidence": args.confidence,
            "min_distinct_prompts": args.min_distinct_prompts,
            "chat_template": args.chat_template,
            "dspark_max_draft_tokens": args.dspark_max_draft_tokens,
            "dspark_lookup": args.dspark_lookup,
            "dspark_draft_bits": args.dspark_draft_bits,
            "dspark_draft_group_size": args.dspark_draft_group_size,
            "dflash_draft_quant": args.dflash_draft_quant,
            "dflash_block_tokens": args.dflash_block_tokens,
            "dflash_verify_cap": args.dflash_verify_cap,
            "dflash_verify_mode": args.dflash_verify_mode,
            "max_p95_latency_regression": args.max_p95_latency_regression,
            "max_peak_memory_regression": args.max_peak_memory_regression,
            "load_seconds": load_seconds,
        },
        "corpus": {
            "source": str(args.corpus_file) if args.corpus_file else "builtin",
            "builtin_version": CORPUS_VERSION if args.corpus_file is None else None,
            "prompts": [
                {
                    **asdict(prompt),
                    "prompt_sha256": hashlib.sha256(prompt.prompt.encode("utf-8")).hexdigest(),
                    "prompt_tokens": len(encoded[prompt.id]),
                }
                for prompt in prompts
            ],
        },
        "schedule": {
            "design": "seeded_randomized_latin_rotation",
            "warmup": [
                {
                    "block": index,
                    "prompt_id": prompt.id,
                    "round": round_index,
                    "modes": order,
                }
                for index, ((prompt, round_index), order) in enumerate(zip(warmup_jobs, warmup_order), start=1)
            ],
            "measurements": [
                {
                    "block": index,
                    "prompt_id": prompt.id,
                    "repetition": repetition,
                    "modes": order,
                }
                for index, ((prompt, repetition), order) in enumerate(zip(measured_jobs, execution_order), start=1)
            ],
        },
        "runs": runs,
        "paired_comparisons": analyses,
        "checks": {
            "baseline_deterministic_by_prompt": deterministic,
            "warmup_failures": warmup_failures,
            "execution_failure_run_ids": execution_failures,
            "fallback_run_ids": fallback_runs,
            "length_failure_run_ids": length_failures,
            "timing_failure_run_ids": timing_failures,
            "parity_failures": parity_failures,
            "strict_passed": strict_passed,
        },
        "research_claim": {
            "definition": (
                "A workload candidate requires paired median TTFT/decode gains, confidence "
                "bounds, p95 TTFT/decode non-regression, deterministic exact-token parity, "
                "complete runs, and zero fallbacks. Complete positive memory measurements "
                "also activate a p95 peak-memory non-regression gate."
            ),
            "scope": "single_model_single_machine_single_corpus_workload",
            "workload_candidate_results": {
                candidate: bool(strict_passed and analysis["candidate_workload_speedup"])
                for candidate, analysis in analyses.items()
            },
            "any_workload_candidate": bool(
                strict_passed and any(analysis["candidate_workload_speedup"] for analysis in analyses.values())
            ),
            "global_breakthrough_evaluable": False,
            "global_breakthrough": False,
            "global_breakthrough_missing_evidence": list(GLOBAL_BREAKTHROUGH_MISSING_EVIDENCE),
        },
        "breakthrough": {
            "field_status": "deprecated_compatibility_alias_always_false",
            "definition": (
                "This single-workload harness cannot establish a global breakthrough. "
                "Use research_claim.workload_candidate_results for scoped candidate evidence."
            ),
            "candidate_results": {candidate: False for candidate in analyses},
            "any_candidate": False,
        },
        "method_notes": {
            "matched_target": "All modes share the exact same loaded target model instance.",
            "greedy": "temperature=0 for baseline and both speculative candidates.",
            "ttft": (
                "External call start to first token event for native dflash-mlx paths; "
                "external call start to first text callback for mlx-dspark. This is a TTFT/prefill proxy."
            ),
            "decode": "(normalized output tokens - 1) / (wall time - TTFT proxy).",
            "tail_latency": (
                "Prompt-clustered paired bootstrap p95 speedup is baseline p95 latency divided by candidate "
                "p95 latency. All repetitions of one prompt remain together; both TTFT and normalized "
                "decode-duration p95 must satisfy the configured non-regression floor."
            ),
            "independence_unit": (
                "Confidence intervals resample prompt_id clusters, never individual repetitions. The configured "
                "minimum number of distinct prompts must pass in addition to the paired-run count."
            ),
            "peak_memory": (
                "Per-run MLX peak bytes are reset and sampled around generation. Because all modes share one "
                "process with all models resident, this compares execution peaks on common residency, not "
                "standalone deployment footprint. The gate is not applied to missing, zero, or partial data."
            ),
            "claim_scope": (
                "Passing gates identifies a candidate speedup only for this measured workload; held-out, "
                "multi-scale, independent, quality, and sustained-load evidence is required for a global claim."
            ),
            "overshoot": (
                "Token IDs and throughput numerator are truncated to exactly max_tokens. "
                "Wall time conservatively includes any upstream final-block overshoot."
            ),
        },
    }

    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path("benchmarks/results") / f"speculative-matched-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[result] {output}", flush=True)
    for candidate, analysis in analyses.items():
        ttft = analysis["metrics"]["ttft_speedup"]
        decode = analysis["metrics"]["decode_speedup"]
        print(
            f"[summary] {candidate} parity={analysis['parity_rate']:.3f} "
            f"ttft={ttft['point_estimate']} decode={decode['point_estimate']} "
            f"workload_candidate={analysis['candidate_workload_speedup']} global_breakthrough=False",
            flush=True,
        )
    return 2 if args.strict and not strict_passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
