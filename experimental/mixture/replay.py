"""Offline replay for the isolated mixture-of-drafters router.

The benchmark JSON contains outcomes for both drafters.  During replay, only
the selected outcome is revealed to the online router.  Counterfactual outcomes
are read after routing solely to compute offline static/oracle comparators.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model import DrafterObservation, RouteContext
from .router import OnlineDrafterRouter, RouterConfig


SUPPORTED_ARMS = ("dflash-mlx", "mlx-dspark")


@dataclass(frozen=True)
class ReplayCase:
    key: tuple[str, int]
    context: RouteContext
    runs: dict[str, Mapping[str, Any]]
    baseline: Mapping[str, Any] | None


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _token_parity(
    run: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> bool | None:
    if baseline is None:
        return None
    candidate_tokens = run.get("token_ids")
    baseline_tokens = baseline.get("token_ids")
    if not isinstance(candidate_tokens, list) or not isinstance(baseline_tokens, list):
        return None
    return candidate_tokens == baseline_tokens


def observation_from_run(
    run: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None,
) -> DrafterObservation:
    arm = str(run.get("mode") or "")
    if arm not in SUPPORTED_ARMS:
        raise ValueError(f"unsupported replay arm: {arm!r}")
    wall_seconds = _positive_float(run.get("wall_seconds"))
    if wall_seconds is None:
        raise ValueError(f"{arm} run has no positive wall_seconds")
    ttft = float(run.get("ttft_seconds") or 0.0)
    output_tokens = _positive_int(run.get("normalized_token_count"))
    if output_tokens is None:
        raise ValueError(f"{arm} run has no positive normalized token count")

    diagnostics = run.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    rounds = _positive_int(
        diagnostics.get("rounds")
        if arm == "mlx-dspark"
        else diagnostics.get("cycles_completed")
    )
    if rounds is None:
        rounds = _positive_int(diagnostics.get("target_forwards"))
    accepted_per_round = _positive_float(
        diagnostics.get("mean_accept_length")
        if arm == "mlx-dspark"
        else diagnostics.get("tokens_per_cycle")
    )
    if accepted_per_round is None and rounds is not None:
        accepted_per_round = output_tokens / rounds

    verify_seconds: float | None = None
    phase_timings = diagnostics.get("phase_timings_us")
    if isinstance(phase_timings, Mapping):
        verify_us = _positive_float(phase_timings.get("verify"))
        if verify_us is not None:
            verify_seconds = verify_us / 1_000_000.0

    peak_memory = _positive_int(run.get("peak_memory_bytes"))
    return DrafterObservation(
        arm=arm,
        wall_seconds=wall_seconds,
        ttft_seconds=ttft,
        output_tokens=output_tokens,
        rounds=rounds,
        accepted_per_round=accepted_per_round,
        verify_seconds=verify_seconds,
        target_forwards=_positive_int(diagnostics.get("target_forwards")),
        fallback=bool(run.get("fallback", False)),
        parity=_token_parity(run, baseline),
        peak_memory_bytes=peak_memory,
        diagnostics={
            "acceptance_ratio": diagnostics.get("acceptance_ratio"),
            "rounds": diagnostics.get("rounds"),
            "cycles_completed": diagnostics.get("cycles_completed"),
            "target_forwards": diagnostics.get("target_forwards"),
            "mean_accept_length": diagnostics.get("mean_accept_length"),
            "tokens_per_cycle": diagnostics.get("tokens_per_cycle"),
        },
    )


def load_cases(path: str | Path) -> tuple[dict[str, Any], list[ReplayCase]]:
    benchmark_path = Path(path)
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    schema = data.get("schema")
    if not isinstance(schema, Mapping) or schema.get("name") != "mio.speculative-rd":
        raise ValueError("expected a mio.speculative-rd benchmark JSON")

    runs = data.get("runs")
    if not isinstance(runs, list):
        raise ValueError("benchmark JSON has no runs list")
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for run in runs:
        if not isinstance(run, Mapping) or run.get("status") != "ok":
            continue
        prompt_id = str(run.get("prompt_id") or "")
        repetition = _positive_int(run.get("repetition"))
        mode = str(run.get("mode") or "")
        if not prompt_id or repetition is None:
            continue
        grouped.setdefault((prompt_id, repetition), {})[mode] = run

    scheduled: list[tuple[str, int]] = []
    schedule = data.get("schedule")
    if isinstance(schedule, Mapping):
        measurements = schedule.get("measurements")
        if isinstance(measurements, list):
            for item in measurements:
                if not isinstance(item, Mapping):
                    continue
                prompt_id = str(item.get("prompt_id") or "")
                repetition = _positive_int(item.get("repetition"))
                key = (prompt_id, repetition or 0)
                if prompt_id and repetition is not None and key not in scheduled:
                    scheduled.append(key)
    if not scheduled:
        scheduled = sorted(grouped)

    cases: list[ReplayCase] = []
    for index, key in enumerate(scheduled, start=1):
        by_mode = grouped.get(key, {})
        if not all(arm in by_mode for arm in SUPPORTED_ARMS):
            continue
        sample = by_mode[SUPPORTED_ARMS[0]]
        prompt_tokens = int(sample.get("prompt_tokens") or 0)
        requested_tokens = _positive_int(sample.get("requested_tokens"))
        if requested_tokens is None:
            raise ValueError(f"case {key!r} has no requested token count")
        cases.append(
            ReplayCase(
                key=key,
                context=RouteContext(
                    request_id=f"replay-{index:04d}:{key[0]}:{key[1]}",
                    prompt_tokens=prompt_tokens,
                    requested_tokens=requested_tokens,
                    # Prompt identity is intentionally excluded from routing
                    # features: memorizing benchmark ids would leak replay
                    # labels and could not transfer to held-out prompts.
                    workload="matched-benchmark",
                ),
                runs={arm: by_mode[arm] for arm in SUPPORTED_ARMS},
                baseline=by_mode.get("baseline"),
            )
        )
    if not cases:
        raise ValueError("benchmark has no complete DSpark/DFlash replay cases")
    return data, cases


def _offline_static_summary(cases: list[ReplayCase]) -> dict[str, Any]:
    totals = {
        arm: sum(float(case.runs[arm]["wall_seconds"]) for case in cases)
        for arm in SUPPORTED_ARMS
    }
    best = min(SUPPORTED_ARMS, key=lambda arm: (totals[arm], arm))
    other = next(arm for arm in SUPPORTED_ARMS if arm != best)
    winners = Counter(
        min(
            SUPPORTED_ARMS,
            key=lambda arm: (float(case.runs[arm]["wall_seconds"]), arm),
        )
        for case in cases
    )
    return {
        "total_wall_seconds": totals,
        "best_arm": best,
        "best_speedup_over_other": totals[other] / totals[best],
        "per_request_wins": dict(winners),
        "oracle_total_wall_seconds": sum(
            min(float(case.runs[arm]["wall_seconds"]) for arm in SUPPORTED_ARMS)
            for case in cases
        ),
    }


def replay_benchmark(
    path: str | Path,
    *,
    config: RouterConfig | None = None,
    convergence_window: int = 5,
) -> dict[str, Any]:
    data, cases = load_cases(path)
    router = OnlineDrafterRouter(SUPPORTED_ARMS, config=config)
    selections: list[str] = []
    selected_observations: list[DrafterObservation] = []
    decisions: list[dict[str, Any]] = []

    for case in cases:
        decision = router.route(case.context)
        selected_run = case.runs[decision.arm]
        observation = observation_from_run(selected_run, baseline=case.baseline)
        router.observe(case.context, decision, observation)
        selections.append(decision.arm)
        selected_observations.append(observation)
        decisions.append(
            {
                **decision.to_dict(),
                "case": {"prompt_id": case.key[0], "repetition": case.key[1]},
                "realized_wall_seconds": observation.wall_seconds,
                "accepted_per_round": observation.accepted_per_round,
                "verify_seconds": observation.verify_seconds,
                "target_forwards": observation.target_forwards,
            }
        )

    offline = _offline_static_summary(cases)
    router_total = sum(item.wall_seconds for item in selected_observations)
    best_static = str(offline["best_arm"])
    best_total = float(offline["total_wall_seconds"][best_static])
    oracle_total = float(offline["oracle_total_wall_seconds"])
    window = max(1, min(int(convergence_window), len(selections)))
    tail = selections[-window:]
    converged_arm = tail[0] if len(set(tail)) == 1 else None

    selected_p95 = _percentile(
        (item.wall_seconds for item in selected_observations), 0.95
    )
    static_p95 = _percentile(
        (float(case.runs[best_static]["wall_seconds"]) for case in cases), 0.95
    )
    selected_memory = [
        float(item.peak_memory_bytes)
        for item in selected_observations
        if item.peak_memory_bytes is not None
    ]
    static_memory = [
        float(case.runs[best_static]["peak_memory_bytes"])
        for case in cases
        if _positive_int(case.runs[best_static].get("peak_memory_bytes")) is not None
    ]
    selected_memory_p95 = _percentile(selected_memory, 0.95)
    static_memory_p95 = _percentile(static_memory, 0.95)
    parity_values = [
        item.parity for item in selected_observations if item.parity is not None
    ]
    parity_rate = (
        sum(bool(value) for value in parity_values) / len(parity_values)
        if parity_values
        else None
    )
    latency_p95_speedup = (
        static_p95 / selected_p95
        if static_p95 is not None and selected_p95 is not None and selected_p95 > 0
        else None
    )
    memory_efficiency = (
        static_memory_p95 / selected_memory_p95
        if static_memory_p95 is not None
        and selected_memory_p95 is not None
        and selected_memory_p95 > 0
        else None
    )
    router_speedup = best_total / router_total

    measured_conditions = {
        "minimum_5pct_wall_time_gain_vs_best_static": router_speedup >= 1.05,
        "p95_latency_non_regression": (
            latency_p95_speedup is not None and latency_p95_speedup >= 1.0
        ),
        "peak_memory_within_5pct": (
            memory_efficiency is not None and memory_efficiency >= (1.0 / 1.05)
        ),
        "exact_greedy_token_parity": parity_rate == 1.0,
        "zero_fallbacks": not any(item.fallback for item in selected_observations),
    }
    measured_speed_candidate = all(measured_conditions.values())

    return {
        "schema": {"name": "mio.mixture-of-drafters-replay", "version": 1},
        "source": {
            "path": str(Path(path)),
            "benchmark_schema": data.get("schema"),
            "cases": len(cases),
        },
        "online_protocol": {
            "selection_unit": "one_drafter_per_request",
            "observations_consumed": router.observation_count,
            "online_updates_per_request": 1,
            "counterfactual_arms_used_for_online_updates": 0,
            "counterfactual_data_use": "offline_comparators_only",
            "exploration_decisions": router.exploration_count,
            "exploration_budget": router.config.max_exploration_decisions,
        },
        "routing": {
            "selection_counts": dict(Counter(selections)),
            "selections": selections,
            "convergence_window": window,
            "converged_arm": converged_arm,
            "final_online_best_static": router.best_static_arm(),
            "fallback_latched": router.fallback_latched,
            "decisions": decisions,
            "final_router_state": router.snapshot(),
        },
        "offline_comparators": offline,
        "measurements": {
            "router_total_wall_seconds": router_total,
            "router_speedup_vs_best_static": router_speedup,
            "router_speedup_vs_oracle": oracle_total / router_total,
            "selected_p95_wall_seconds": selected_p95,
            "best_static_p95_wall_seconds": static_p95,
            "p95_latency_speedup_vs_best_static": latency_p95_speedup,
            "selected_p95_peak_memory_bytes": selected_memory_p95,
            "best_static_p95_peak_memory_bytes": static_memory_p95,
            "peak_memory_efficiency_vs_best_static": memory_efficiency,
            "selected_parity_rate": parity_rate,
            "selected_fallbacks": sum(item.fallback for item in selected_observations),
        },
        "claim": {
            "measured_conditions": measured_conditions,
            "same_corpus_speed_candidate": measured_speed_candidate,
            "workload_candidate_evaluable": False,
            "workload_candidate": False,
            "global_breakthrough": False,
            "why_not_evaluable": [
                "router_training_and_evaluation_share_the_same_online_replay",
                "no_prompt_cluster_bootstrap_confidence_interval",
                "no_held_out_corpus",
                "single_model_scale",
                "single_machine",
                "no_sustained_concurrency_test",
            ],
            "real_gain_criterion": {
                "minimum_wall_time_speedup_vs_best_static": 1.05,
                "cluster_bootstrap_ci_lower_bound_strictly_above": 1.0,
                "p95_latency_speedup_minimum": 1.0,
                "maximum_peak_memory_regression": 0.05,
                "required_exact_greedy_parity": 1.0,
                "zero_fallbacks": True,
                "required_protocol": (
                    "freeze_router_after_calibration_then_evaluate_on_disjoint_held_out_prompts"
                ),
                "global_replication": [
                    "at_least_two_model_scales_including_27B",
                    "independent_hardware_run",
                    "sustained_load_and_concurrency",
                ],
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay Mio DSpark/DFlash benchmark data through the isolated router."
    )
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extra-exploration", type=int, default=0)
    parser.add_argument("--exploration-interval", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    calibration = len(SUPPORTED_ARMS)
    config = RouterConfig(
        max_exploration_decisions=calibration + max(0, args.extra_exploration),
        exploration_interval=max(0, args.exploration_interval),
    )
    report = replay_benchmark(args.benchmark_json, config=config)
    encoded = json.dumps(report, indent=2 if args.pretty else None, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
