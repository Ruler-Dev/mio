from __future__ import annotations

from dataclasses import replace
import math

import pytest

from experimental.effort.statistics_v2 import (
    EffortStatisticsRow,
    GateFailure,
    PreregisteredGatePolicy,
    RunProvenance,
    analyze_paired_rows,
    evaluate_preregistered_gate,
    exact_mcnemar,
)


def row(
    task: int,
    strategy: str,
    *,
    correct: bool,
    seconds: float = 1.0,
    fast_path: bool = True,
    deadline_violations: int = 0,
) -> EffortStatisticsRow:
    return EffortStatisticsRow(
        task_id=f"task-{task:03d}",
        strategy=strategy,
        correct=correct,
        e2e_seconds=seconds,
        fast_path=fast_path,
        deadline_violations=deadline_violations,
    )


def provenance(*, dirty: bool = False, leakage: bool = False) -> RunProvenance:
    digest = "a" * 64
    return RunProvenance(
        git_revision="b" * 40,
        git_dirty=dirty,
        model_revision="Qwen-test@0123456789abcdef",
        policy_sha256=digest,
        task_manifest_sha256=digest,
        scorer_sha256=digest,
        verifier_sha256=digest,
        preregistration_sha256=digest,
        test_split_id="heldout-v1",
        leakage_detected=leakage,
    )


def paired_rows(
    tasks: int = 100,
    *,
    rescues: int = 10,
    fast_path_seconds: float = 1.01,
    retry_seconds: float = 1.10,
    deadline_task: int | None = None,
) -> list[EffortStatisticsRow]:
    rows: list[EffortStatisticsRow] = []
    baseline_correct_from = 20
    rescued = set(range(baseline_correct_from - rescues, baseline_correct_from))
    for task in range(tasks):
        baseline_correct = task >= baseline_correct_from
        candidate_correct = baseline_correct or task in rescued
        is_fast_path = task not in rescued
        rows.extend(
            (
                row(task, "low", correct=baseline_correct),
                row(
                    task,
                    "medium",
                    correct=candidate_correct,
                    seconds=fast_path_seconds if is_fast_path else retry_seconds,
                    fast_path=is_fast_path,
                    deadline_violations=int(task == deadline_task),
                ),
            )
        )
    return rows


def analyze(rows, *, samples: int = 2_000, seed: int = 17):
    return analyze_paired_rows(
        rows,
        baseline_strategy="low",
        candidate_strategy="medium",
        bootstrap_samples=samples,
        seed=seed,
    )


def test_exact_two_sided_mcnemar_uses_only_discordant_pairs() -> None:
    baseline = [False] * 6 + [True] * 4
    candidate = [True] * 6 + [True] * 4
    result = exact_mcnemar(baseline, candidate)
    assert result.baseline_only_correct == 0
    assert result.candidate_only_correct == 6
    assert result.discordant_pairs == 6
    assert result.p_value == pytest.approx(0.03125)

    mixed = exact_mcnemar(
        [True, False, False, False, False, False],
        [False, True, True, True, True, True],
    )
    assert mixed.p_value == pytest.approx(0.21875)


def test_seeded_task_bootstrap_is_deterministic_and_reports_cost_metrics() -> None:
    rows = paired_rows()
    first = analyze(rows, samples=500, seed=99)
    second = analyze(reversed(rows), samples=500, seed=99)
    assert first == second
    assert first.tasks == 100
    assert first.baseline_accuracy == pytest.approx(0.8)
    assert first.candidate_accuracy == pytest.approx(0.9)
    assert first.accuracy_delta.point == pytest.approx(0.1)
    assert first.accuracy_delta.lower > 0.0
    assert first.e2e_latency_ratio.point == pytest.approx(1.019)
    assert first.correct_completions_per_second_ratio.point == pytest.approx(
        (0.9 / 101.9) / (0.8 / 100.0)
    )
    assert first.baseline_latency_seconds.p50 == pytest.approx(1.0)
    assert first.baseline_latency_seconds.p95 == pytest.approx(1.0)
    assert first.candidate_latency_seconds.p50 == pytest.approx(1.01)
    assert first.candidate_latency_seconds.p95 == pytest.approx(1.10)
    assert first.paired_latency_ratio.p50 == pytest.approx(1.01)
    assert first.paired_latency_ratio.p95 == pytest.approx(1.10)
    assert first.fast_path_tasks == 90
    assert first.fast_path_overhead_ratio == pytest.approx(0.01)


def test_bootstrap_seed_changes_resamples_not_point_estimates() -> None:
    first = analyze(paired_rows(), samples=200, seed=1)
    second = analyze(paired_rows(), samples=200, seed=2)
    assert first.accuracy_delta.point == second.accuracy_delta.point
    assert first.e2e_latency_ratio.point == second.e2e_latency_ratio.point
    assert first.accuracy_delta != second.accuracy_delta


@pytest.mark.parametrize(
    "bad_row, message",
    [
        (
            {
                "task_id": "task",
                "strategy": "low",
                "correct": True,
                "e2e_seconds": 1.0,
                "fast_path": True,
            },
            "missing required",
        ),
        (
            {
                "task_id": "task",
                "strategy": "low",
                "correct": True,
                "e2e_seconds": math.nan,
                "fast_path": True,
                "deadline_violations": 0,
            },
            "finite and positive",
        ),
    ],
)
def test_rows_fail_closed_on_missing_or_non_finite_values(bad_row, message) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_paired_rows(
            [bad_row],
            baseline_strategy="low",
            candidate_strategy="medium",
            bootstrap_samples=1,
        )


def test_analysis_rejects_duplicates_missing_pairs_and_unknown_strategies() -> None:
    baseline = row(1, "low", correct=True)
    candidate = row(1, "medium", correct=True)
    with pytest.raises(ValueError, match="duplicate"):
        analyze([baseline, baseline, candidate], samples=10)
    with pytest.raises(ValueError, match="complete strategy pair"):
        analyze([baseline], samples=10)
    with pytest.raises(ValueError, match="unexpected strategies"):
        analyze([baseline, candidate, row(2, "high", correct=True)], samples=10)


def test_preregistered_gate_accepts_only_a_clean_supported_gain() -> None:
    statistics = analyze(paired_rows(), samples=2_000)
    gate = evaluate_preregistered_gate(statistics, provenance())
    assert gate.passed is True
    assert gate.failures == ()
    assert gate.corrected_alpha == pytest.approx(0.05 / 3)


def test_gate_rejects_small_dirty_leaked_or_missing_provenance() -> None:
    small = analyze(paired_rows(tasks=50, rescues=10), samples=500)
    small_gate = evaluate_preregistered_gate(small, provenance(dirty=True, leakage=True))
    assert GateFailure.INSUFFICIENT_TASKS in small_gate.failures
    assert GateFailure.GIT_DIRTY in small_gate.failures
    assert GateFailure.LEAKAGE_DETECTED in small_gate.failures

    missing = evaluate_preregistered_gate(
        analyze(paired_rows(), samples=500),
        {"git_revision": "b" * 40},
    )
    assert GateFailure.MISSING_PROVENANCE in missing.failures
    assert missing.passed is False


def test_gate_rejects_non_significant_or_non_positive_quality() -> None:
    rows = paired_rows(rescues=0)
    statistics = analyze(rows, samples=500)
    gate = evaluate_preregistered_gate(statistics, provenance())
    assert GateFailure.ACCURACY_DELTA_TOO_SMALL in gate.failures
    assert GateFailure.QUALITY_CI_NOT_POSITIVE in gate.failures
    assert GateFailure.MCNEMAR_NOT_SIGNIFICANT in gate.failures


def test_gate_rejects_fast_path_e2e_and_deadline_regressions() -> None:
    fast_regression = analyze(
        paired_rows(fast_path_seconds=1.03, retry_seconds=1.10),
        samples=500,
    )
    fast_gate = evaluate_preregistered_gate(fast_regression, provenance())
    assert GateFailure.FAST_PATH_OVERHEAD in fast_gate.failures

    slow = analyze(
        paired_rows(fast_path_seconds=1.14, retry_seconds=1.40, deadline_task=0),
        samples=500,
    )
    slow_gate = evaluate_preregistered_gate(slow, provenance())
    assert GateFailure.E2E_LATENCY in slow_gate.failures
    assert GateFailure.DEADLINE_VIOLATIONS in slow_gate.failures


def test_gate_rejects_a_missing_fast_path_metric() -> None:
    statistics = analyze(paired_rows(rescues=20), samples=500)
    assert statistics.fast_path_tasks == 80
    without_fast_path = replace(
        statistics,
        fast_path_tasks=0,
        fast_path_overhead_ratio=None,
    )
    gate = evaluate_preregistered_gate(without_fast_path, provenance())
    assert GateFailure.FAST_PATH_METRIC_MISSING in gate.failures


def test_bonferroni_uses_the_number_of_planned_comparisons() -> None:
    statistics = analyze(paired_rows(rescues=6), samples=2_000)
    assert statistics.mcnemar.p_value == pytest.approx(0.03125)

    one_comparison = evaluate_preregistered_gate(
        statistics,
        provenance(),
        policy=PreregisteredGatePolicy(planned_comparisons=1),
    )
    three_comparisons = evaluate_preregistered_gate(
        statistics,
        provenance(),
        policy=PreregisteredGatePolicy(planned_comparisons=3),
    )
    assert GateFailure.MCNEMAR_NOT_SIGNIFICANT not in one_comparison.failures
    assert GateFailure.MCNEMAR_NOT_SIGNIFICANT in three_comparisons.failures
    assert three_comparisons.corrected_alpha == pytest.approx(0.05 / 3)


def test_gate_requires_five_point_accuracy_gain() -> None:
    statistics = analyze(paired_rows(rescues=4), samples=1_000)
    gate = evaluate_preregistered_gate(
        statistics,
        provenance(),
        policy=PreregisteredGatePolicy(alpha=0.5, planned_comparisons=1),
    )
    assert statistics.accuracy_delta.point == pytest.approx(0.04)
    assert GateFailure.ACCURACY_DELTA_TOO_SMALL in gate.failures


def test_low_baseline_bootstrap_is_conservative_instead_of_crashing() -> None:
    rows: list[EffortStatisticsRow] = []
    for task in range(100):
        baseline_correct = task == 99
        candidate_correct = baseline_correct or task < 10
        rows.extend(
            (
                row(task, "low", correct=baseline_correct),
                row(
                    task,
                    "medium",
                    correct=candidate_correct,
                    seconds=1.01,
                ),
            )
        )
    statistics = analyze(rows, samples=2_000, seed=11)
    assert statistics.zero_baseline_correct_bootstrap_samples > 0
    assert statistics.correct_completions_per_second_ratio.lower == 0.0
    gate = evaluate_preregistered_gate(statistics, provenance())
    assert GateFailure.CORRECT_COMPLETIONS_RATE_TOO_LOW in gate.failures


def test_gate_rejects_low_correct_completions_rate_lower_bound() -> None:
    statistics = analyze(paired_rows(), samples=1_000)
    degraded = replace(
        statistics,
        correct_completions_per_second_ratio=replace(
            statistics.correct_completions_per_second_ratio,
            lower=0.94,
        ),
    )
    gate = evaluate_preregistered_gate(degraded, provenance())
    assert GateFailure.CORRECT_COMPLETIONS_RATE_TOO_LOW in gate.failures
