from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from experimental.effort.uncertainty_statistics import (
    AURC_DEFINITION,
    PreregisteredUncertaintyGatePolicy,
    UncertaintyGateFailure,
    UncertaintyObservation,
    analyze_paired_uncertainty,
    analyze_uncertainty,
    area_under_risk_coverage_curve,
    brier_score,
    evaluate_preregistered_uncertainty_gate,
    tie_corrected_auroc,
)


def observations(
    tasks: int = 200,
    *,
    errors: int = 100,
    signal: str = "strong",
) -> list[UncertaintyObservation]:
    result: list[UncertaintyObservation] = []
    for task in range(tasks):
        is_error = task < errors
        if signal == "strong":
            probability = (0.8 + 0.1 * (task % 2)) if is_error else (0.1 + 0.1 * (task % 2))
        elif signal == "constant":
            probability = errors / tasks
        elif signal == "inverse":
            probability = 0.1 if is_error else 0.9
        elif signal == "overlap":
            probability = ((task * 37) % 101) / 100
        else:
            raise AssertionError(f"unknown test signal: {signal}")
        result.append(
            UncertaintyObservation(
                task_cluster_id=f"task-{task:04d}",
                predicted_error_probability=probability,
                is_error=is_error,
            )
        )
    return result


def policy_for_small_tests(**changes) -> PreregisteredUncertaintyGatePolicy:
    defaults = {
        "min_tasks": 1,
        "min_errors": 1,
        "min_correct": 1,
        "min_valid_auroc_bootstrap_fraction": 0.0,
    }
    defaults.update(changes)
    return PreregisteredUncertaintyGatePolicy(**defaults)


FROZEN_REFERENCE = {
    "reference_error_probability": 0.5,
    "reference_source": "calibration-split:v1:sha256:test-fixture",
    "reference_is_frozen": True,
}


def test_tie_corrected_auroc_is_exact_and_order_invariant() -> None:
    scores = [0.9, 0.5, 0.5, 0.1]
    outcomes = [True, True, False, False]
    assert tie_corrected_auroc(scores, outcomes) == pytest.approx(0.875)
    assert tie_corrected_auroc(reversed(scores), reversed(outcomes)) == pytest.approx(0.875)
    assert tie_corrected_auroc([0.5] * 4, outcomes) == pytest.approx(0.5)


def test_brier_and_tie_corrected_aurc_have_documented_values() -> None:
    outcomes = [True, True, False, False]
    assert brier_score([0.9, 0.8, 0.2, 0.1], outcomes) == pytest.approx(0.025)

    all_tied = area_under_risk_coverage_curve([0.5] * 4, outcomes)
    perfect = area_under_risk_coverage_curve([0.9, 0.9, 0.1, 0.1], outcomes)
    inverse = area_under_risk_coverage_curve([0.1, 0.1, 0.9, 0.9], outcomes)
    assert all_tied == pytest.approx(0.5)
    assert perfect == pytest.approx((0.0 + 0.0 + 1 / 3 + 1 / 2) / 4)
    assert inverse == pytest.approx((1.0 + 1.0 + 2 / 3 + 1 / 2) / 4)
    assert perfect < all_tied < inverse


def test_seeded_task_bootstrap_is_deterministic_and_order_invariant() -> None:
    rows = observations(signal="overlap")
    first = analyze_uncertainty(
        rows,
        signal_name="native",
        bootstrap_samples=300,
        seed=41,
        reliability_bin_count=8,
    )
    second = analyze_uncertainty(
        reversed(rows),
        signal_name="native",
        bootstrap_samples=300,
        seed=41,
        reliability_bin_count=8,
    )
    assert first == second
    assert first.task_count == 200
    assert first.error_count == first.correct_count == 100
    assert first.error_prevalence == pytest.approx(0.5)
    assert first.valid_auroc_bootstrap_samples == 300
    assert first.invalid_auroc_bootstrap_samples == 0
    assert sum(value.task_count for value in first.reliability_bins) == 200
    assert first.aurc_definition == AURC_DEFINITION


def test_reliability_bins_are_descriptive_only() -> None:
    rows = observations(signal="strong")
    two_bins = analyze_uncertainty(
        rows,
        signal_name="calibrated",
        bootstrap_samples=200,
        seed=9,
        reliability_bin_count=2,
    )
    twenty_bins = analyze_uncertainty(
        rows,
        signal_name="calibrated",
        bootstrap_samples=200,
        seed=9,
        reliability_bin_count=20,
    )
    assert len(two_bins.reliability_bins) == 2
    assert len(twenty_bins.reliability_bins) == 20
    assert two_bins.auroc == twenty_bins.auroc
    assert two_bins.brier == twenty_bins.brier
    assert two_bins.aurc == twenty_bins.aurc
    assert two_bins.brier_delta_vs_constant == twenty_bins.brier_delta_vs_constant
    assert two_bins.aurc_delta_vs_constant == twenty_bins.aurc_delta_vs_constant
    assert evaluate_preregistered_uncertainty_gate(two_bins) == evaluate_preregistered_uncertainty_gate(
        twenty_bins
    )


def test_empty_reliability_bins_are_explicitly_serialized_as_null() -> None:
    result = analyze_uncertainty(
        observations(tasks=20, errors=10, signal="strong"),
        signal_name="calibrated",
        bootstrap_samples=100,
        seed=1,
        reliability_bin_count=10,
    )
    empty = next(value for value in result.reliability_bins if value.task_count == 0)
    assert empty.mean_predicted_error_probability is None
    assert empty.observed_error_rate is None
    encoded = json.dumps(result.to_mapping(), sort_keys=True, allow_nan=False)
    assert '"schema": "mio.uncertainty-router-statistics.v1"' in encoded
    assert '"mean_predicted_error_probability": null' in encoded


def test_paired_bootstrap_uses_identical_tasks_and_candidate_minus_reference() -> None:
    reference = observations(signal="constant")
    candidate = observations(signal="strong")
    comparison = analyze_paired_uncertainty(
        reference,
        reversed(candidate),
        reference_signal="constant",
        candidate_signal="calibrated",
        bootstrap_samples=300,
        seed=77,
    )
    repeated = analyze_paired_uncertainty(
        reversed(reference),
        candidate,
        reference_signal="constant",
        candidate_signal="calibrated",
        bootstrap_samples=300,
        seed=77,
    )
    assert comparison == repeated
    assert comparison.reference_metrics.auroc == pytest.approx(0.5)
    assert comparison.candidate_metrics.auroc == pytest.approx(1.0)
    assert comparison.auroc_delta.point == pytest.approx(0.5)
    assert comparison.brier_delta.upper < 0.0
    assert comparison.aurc_delta.upper < 0.0
    encoded = json.dumps(comparison.to_mapping(), sort_keys=True, allow_nan=False)
    assert '"direction": "candidate_minus_reference"' in encoded


def test_strong_signal_passes_all_default_preregistered_gates() -> None:
    statistics = analyze_uncertainty(
        observations(signal="strong"),
        signal_name="calibrated",
        bootstrap_samples=500,
        seed=5,
        **FROZEN_REFERENCE,
    )
    gate = evaluate_preregistered_uncertainty_gate(statistics)
    assert gate.passed is True
    assert gate.failures == ()
    assert statistics.auroc.lower > 0.5
    assert statistics.brier_delta_vs_constant.upper < 0.0
    assert statistics.aurc_delta_vs_constant.upper <= 0.0
    assert json.loads(json.dumps(gate.to_mapping()))["passed"] is True


def test_gate_uses_brier_and_aurc_upper_bounds_not_optimistic_points() -> None:
    statistics = analyze_uncertainty(
        observations(signal="strong"),
        signal_name="calibrated",
        bootstrap_samples=300,
        seed=6,
        **FROZEN_REFERENCE,
    )
    uncertain_brier = replace(
        statistics,
        brier_delta_vs_constant=replace(
            statistics.brier_delta_vs_constant,
            upper=0.001,
        ),
    )
    brier_gate = evaluate_preregistered_uncertainty_gate(uncertain_brier)
    assert (
        UncertaintyGateFailure.BRIER_NOT_BETTER_THAN_CONSTANT_PREVALENCE
        in brier_gate.failures
    )

    uncertain_aurc = replace(
        statistics,
        aurc_delta_vs_constant=replace(
            statistics.aurc_delta_vs_constant,
            upper=0.001,
        ),
    )
    aurc_gate = evaluate_preregistered_uncertainty_gate(uncertain_aurc)
    assert UncertaintyGateFailure.AURC_WORSE_THAN_CONSTANT_PREVALENCE in aurc_gate.failures


def test_gate_fails_closed_for_small_or_class_sparse_samples() -> None:
    too_few_errors = analyze_uncertainty(
        observations(tasks=30, errors=10, signal="strong"),
        signal_name="calibrated",
        bootstrap_samples=300,
        seed=12,
    )
    first_gate = evaluate_preregistered_uncertainty_gate(too_few_errors)
    assert UncertaintyGateFailure.INSUFFICIENT_TASKS in first_gate.failures
    assert UncertaintyGateFailure.INSUFFICIENT_ERRORS in first_gate.failures

    too_few_correct = analyze_uncertainty(
        observations(tasks=30, errors=25, signal="strong"),
        signal_name="calibrated",
        bootstrap_samples=300,
        seed=12,
    )
    second_gate = evaluate_preregistered_uncertainty_gate(too_few_correct)
    assert UncertaintyGateFailure.INSUFFICIENT_TASKS in second_gate.failures
    assert UncertaintyGateFailure.INSUFFICIENT_CORRECT in second_gate.failures


def test_gate_rejects_chance_calibration_and_adverse_selective_risk() -> None:
    constant = analyze_uncertainty(
        observations(tasks=40, errors=20, signal="constant"),
        signal_name="constant",
        bootstrap_samples=300,
        seed=8,
        **FROZEN_REFERENCE,
    )
    constant_gate = evaluate_preregistered_uncertainty_gate(
        constant,
        policy=policy_for_small_tests(),
    )
    assert UncertaintyGateFailure.AUROC_LOWER_BOUND_NOT_ABOVE_CHANCE in constant_gate.failures
    assert (
        UncertaintyGateFailure.BRIER_NOT_BETTER_THAN_CONSTANT_PREVALENCE
        in constant_gate.failures
    )
    assert UncertaintyGateFailure.AURC_WORSE_THAN_CONSTANT_PREVALENCE not in constant_gate.failures

    inverse = analyze_uncertainty(
        observations(tasks=40, errors=20, signal="inverse"),
        signal_name="inverse",
        bootstrap_samples=300,
        seed=8,
        **FROZEN_REFERENCE,
    )
    inverse_gate = evaluate_preregistered_uncertainty_gate(
        inverse,
        policy=policy_for_small_tests(),
    )
    assert UncertaintyGateFailure.AURC_WORSE_THAN_CONSTANT_PREVALENCE in inverse_gate.failures


def test_gate_rejects_insufficient_class_valid_bootstrap_support() -> None:
    statistics = analyze_uncertainty(
        observations(tasks=2, errors=1, signal="strong"),
        signal_name="tiny",
        bootstrap_samples=400,
        seed=3,
        reliability_bin_count=2,
        **FROZEN_REFERENCE,
    )
    assert statistics.invalid_auroc_bootstrap_samples > 0
    gate = evaluate_preregistered_uncertainty_gate(
        statistics,
        policy=policy_for_small_tests(min_valid_auroc_bootstrap_fraction=0.9),
    )
    assert UncertaintyGateFailure.INSUFFICIENT_AUROC_BOOTSTRAP_SUPPORT in gate.failures


def test_observed_prevalence_is_descriptive_and_gate_requires_frozen_reference() -> None:
    rows = observations(tasks=40, errors=10, signal="strong")
    descriptive = analyze_uncertainty(
        rows,
        signal_name="descriptive",
        bootstrap_samples=200,
        seed=13,
    )
    assert descriptive.reference_error_probability == pytest.approx(0.25)
    assert descriptive.reference_source == "observed-evaluation-prevalence-descriptive-only"
    assert descriptive.reference_is_frozen is False
    gate = evaluate_preregistered_uncertainty_gate(
        descriptive,
        policy=policy_for_small_tests(),
    )
    assert UncertaintyGateFailure.REFERENCE_NOT_FROZEN in gate.failures

    frozen = analyze_uncertainty(
        rows,
        signal_name="confirmatory",
        bootstrap_samples=200,
        seed=13,
        reference_error_probability=0.4,
        reference_source="calibration-split:v2:manifest-deadbeef",
        reference_is_frozen=True,
    )
    assert frozen.reference_error_probability == pytest.approx(0.4)
    assert frozen.reference_source == "calibration-split:v2:manifest-deadbeef"
    expected_brier = 0.25 * 0.6**2 + 0.75 * 0.4**2
    assert frozen.constant_reference_metrics.brier == pytest.approx(expected_brier)
    assert UncertaintyGateFailure.REFERENCE_NOT_FROZEN not in evaluate_preregistered_uncertainty_gate(
        frozen,
        policy=policy_for_small_tests(),
    ).failures


@pytest.mark.parametrize(
    "probability",
    [math.nan, math.inf, -0.01, 1.01, True, "0.5"],
)
def test_observations_reject_invalid_probabilities(probability) -> None:
    with pytest.raises(ValueError, match="predicted_error_probability"):
        UncertaintyObservation("task", probability, True)


@pytest.mark.parametrize("outcome", [0, 1, None, "error"])
def test_observations_require_boolean_outcomes(outcome) -> None:
    with pytest.raises(ValueError, match="is_error must be a bool"):
        UncertaintyObservation("task", 0.5, outcome)


def test_mapping_schema_and_task_clusters_fail_closed() -> None:
    with pytest.raises(ValueError, match="fields do not match"):
        UncertaintyObservation.from_mapping(
            {
                "task_cluster_id": "task",
                "predicted_error_probability": 0.5,
                "is_error": True,
                "unexpected": "schema drift",
            }
        )

    duplicate = [
        UncertaintyObservation("task", 0.9, True),
        UncertaintyObservation("task", 0.1, False),
    ]
    with pytest.raises(ValueError, match="duplicate task_cluster_id"):
        analyze_uncertainty(
            duplicate,
            signal_name="bad",
            bootstrap_samples=10,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        analyze_uncertainty([], signal_name="bad", bootstrap_samples=10)
    with pytest.raises(ValueError, match="at least one error and one correct"):
        analyze_uncertainty(
            [UncertaintyObservation("task", 0.1, False)],
            signal_name="bad",
            bootstrap_samples=10,
        )


def test_paired_analysis_rejects_missing_tasks_but_supports_condition_outcomes() -> None:
    reference = observations(tasks=20, errors=10, signal="constant")
    candidate = observations(tasks=20, errors=10, signal="strong")
    with pytest.raises(ValueError, match="same task_cluster_ids"):
        analyze_paired_uncertainty(
            reference,
            candidate[:-1],
            reference_signal="raw",
            candidate_signal="calibrated",
            bootstrap_samples=10,
        )

    changed_condition = list(candidate)
    changed_condition[0] = replace(changed_condition[0], is_error=False)
    comparison = analyze_paired_uncertainty(
        reference,
        changed_condition,
        reference_signal="raw",
        candidate_signal="calibrated",
        bootstrap_samples=100,
        seed=4,
    )
    assert comparison.reference_error_count == 10
    assert comparison.candidate_error_count == 9
    assert comparison.reference_correct_count == 10
    assert comparison.candidate_correct_count == 11


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"signal_name": "", "bootstrap_samples": 10}, "signal_name"),
        ({"signal_name": "x", "bootstrap_samples": 0}, "bootstrap_samples"),
        ({"signal_name": "x", "bootstrap_samples": 10, "seed": True}, "seed"),
        ({"signal_name": "x", "bootstrap_samples": 10, "confidence": 1.0}, "confidence"),
        (
            {"signal_name": "x", "bootstrap_samples": 10, "reliability_bin_count": 0},
            "reliability_bin_count",
        ),
        (
            {
                "signal_name": "x",
                "bootstrap_samples": 10,
                "reference_error_probability": 0.4,
            },
            "reference_source",
        ),
        (
            {
                "signal_name": "x",
                "bootstrap_samples": 10,
                "reference_source": "calibration",
            },
            "requires an explicit",
        ),
        (
            {
                "signal_name": "x",
                "bootstrap_samples": 10,
                "reference_is_frozen": True,
            },
            "cannot be declared frozen",
        ),
        (
            {
                "signal_name": "x",
                "bootstrap_samples": 10,
                "reference_is_frozen": 1,
            },
            "reference_is_frozen must be a bool",
        ),
    ],
)
def test_analysis_arguments_fail_closed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_uncertainty(observations(tasks=20, errors=10), **kwargs)


def test_metric_functions_reject_malformed_or_single_class_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        brier_score([], [])
    with pytest.raises(ValueError, match="equal length"):
        area_under_risk_coverage_curve([0.1], [False, True])
    with pytest.raises(ValueError, match="bool"):
        brier_score([0.1], [0])
    with pytest.raises(ValueError, match="at least one error and one correct"):
        tie_corrected_auroc([0.1, 0.2], [False, False])


def test_policy_rejects_non_preregisterable_thresholds() -> None:
    with pytest.raises(ValueError, match="min_tasks"):
        PreregisteredUncertaintyGatePolicy(min_tasks=0)
    with pytest.raises(ValueError, match="min_brier_improvement"):
        PreregisteredUncertaintyGatePolicy(min_brier_improvement=-0.01)
    with pytest.raises(ValueError, match="max_aurc_regression"):
        PreregisteredUncertaintyGatePolicy(max_aurc_regression=-0.01)
    with pytest.raises(ValueError, match="min_valid_auroc_bootstrap_fraction"):
        PreregisteredUncertaintyGatePolicy(min_valid_auroc_bootstrap_fraction=1.01)
