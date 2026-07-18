from __future__ import annotations

import inspect
import json
import random

import pytest

from experimental.effort.calibration import (
    FrozenUncertaintyCalibrator,
    InsufficientCalibrationDataError,
    IsotonicCalibrationRow,
    TransitionCalibrationObservation,
    UncertaintyCalibrationObservation,
    build_frozen_transition_model,
    fit_isotonic_uncertainty,
    frozen_transition_model_from_mapping,
    frozen_transition_model_to_mapping,
)
from experimental.markov_effort_controller import (
    CalibrationIdentity,
    ControllerAction,
    Trigger,
)


IDENTITY = CalibrationIdentity(
    model="qwen-test",
    config="config-sha256:111",
    prompt="prompt-sha256:222",
    sampler="greedy-temp0",
    corpus="coding-suite-v1",
    split="calibration-v1",
    backend="mlx-lm-test",
)


def uncertainty_rows() -> list[UncertaintyCalibrationObservation]:
    return [
        UncertaintyCalibrationObservation("task-1", 0.10, False),
        UncertaintyCalibrationObservation("task-2", 0.20, True),
        UncertaintyCalibrationObservation("task-3", 0.30, False),
        UncertaintyCalibrationObservation("task-4", 0.40, True),
        UncertaintyCalibrationObservation("task-5", 0.50, True),
        UncertaintyCalibrationObservation("task-6", 0.60, True),
        UncertaintyCalibrationObservation("task-7", 0.70, True),
        UncertaintyCalibrationObservation("task-8", 0.80, True),
    ]


def transition_rows(count: int = 12) -> list[TransitionCalibrationObservation]:
    return [
        TransitionCalibrationObservation(
            task_cluster_id=f"task-{index:02d}",
            context_bucket="coding",
            trigger=Trigger.VALIDATOR_FAILURE,
            depth=1,
            action=ControllerAction.GENERATE_REPAIR,
            rescued=index % 3 != 0,
            extra_output_tokens=20 + index,
            direct_e2e_seconds=1.0 + index / 20,
            extra_e2e_seconds=0.30 + index / 100,
        )
        for index in range(count)
    ]


def test_weighted_pava_is_monotone_and_interpolates_without_labels() -> None:
    calibrator = fit_isotonic_uncertainty(IDENTITY, uncertainty_rows())
    probabilities = [row.error_probability for row in calibrator.rows]
    assert probabilities == sorted(probabilities)
    assert probabilities[:3] == pytest.approx([0.0, 0.5, 0.5])
    assert calibrator.transform(0.25) == pytest.approx(0.5)
    assert calibrator.transform(0.0) == pytest.approx(0.0)
    assert calibrator.transform(1.0) == pytest.approx(1.0)
    assert tuple(inspect.signature(calibrator.transform).parameters) == ("raw_uncertainty",)
    assert "is_error" not in calibrator.to_mapping()


def test_isotonic_fit_is_deterministic_across_input_order() -> None:
    rows = uncertainty_rows()
    shuffled = rows.copy()
    random.Random(81).shuffle(shuffled)
    assert fit_isotonic_uncertainty(IDENTITY, rows) == fit_isotonic_uncertainty(
        IDENTITY,
        shuffled,
    )


def test_calibrator_json_round_trip_preserves_exact_identity_and_rows() -> None:
    fitted = fit_isotonic_uncertainty(IDENTITY, uncertainty_rows())
    payload = json.loads(json.dumps(fitted.to_mapping(), sort_keys=True))
    restored = FrozenUncertaintyCalibrator.from_mapping(payload)
    assert restored == fitted
    assert restored.identity == IDENTITY
    assert restored.transform(0.35) == fitted.transform(0.35)


def test_calibrator_rejects_non_monotone_or_underpowered_inputs() -> None:
    with pytest.raises(ValueError, match="monotone"):
        FrozenUncertaintyCalibrator(
            identity=IDENTITY,
            rows=(
                IsotonicCalibrationRow(0.2, 0.8, 1, 1),
                IsotonicCalibrationRow(0.8, 0.2, 1, 1),
            ),
            observation_count=2,
            task_cluster_count=2,
        )
    with pytest.raises(InsufficientCalibrationDataError, match="requires 8"):
        fit_isotonic_uncertainty(IDENTITY, uncertainty_rows()[:7])
    with pytest.raises(ValueError, match="one observation"):
        fit_isotonic_uncertainty(
            IDENTITY,
            [*uncertainty_rows(), UncertaintyCalibrationObservation("task-1", 0.9, True)],
        )


def test_transition_builder_produces_one_sided_cluster_bounds() -> None:
    rows = transition_rows()
    model = build_frozen_transition_model(
        IDENTITY,
        rows,
        resamples=1_000,
        seed=19,
    )
    estimate, source = model.lookup(
        context_bucket="coding",
        trigger=Trigger.VALIDATOR_FAILURE,
        depth=1,
        action=ControllerAction.GENERATE_REPAIR,
    )
    assert source == "coding"
    assert estimate is not None
    rescue_point = sum(row.rescued for row in rows) / len(rows)
    token_point = sum(row.extra_output_tokens for row in rows) / len(rows)
    latency_point = sum(row.extra_e2e_latency_ratio for row in rows) / len(rows)
    assert estimate.conservative_success_lcb <= rescue_point
    assert estimate.extra_output_tokens_ucb >= token_point
    assert estimate.extra_e2e_latency_ratio_ucb >= latency_point
    assert estimate.bootstrap.task_cluster_count == len(rows)
    assert estimate.bootstrap.resamples == 1_000
    assert estimate.bootstrap.confidence_level == 0.95
    assert estimate.bootstrap.method == "task-cluster-percentile-one-sided-v1"


def test_transition_builder_counts_regressions_in_net_quality_lcb() -> None:
    rows = [
        TransitionCalibrationObservation(
            task_cluster_id=f"net-{index}",
            context_bucket="coding",
            trigger=Trigger.CALIBRATED_UNCERTAINTY,
            depth=1,
            action=ControllerAction.GENERATE_ALTERNATIVE,
            rescued=index < 6,
            quality_delta=1.0 if index < 6 else -1.0,
            extra_output_tokens=20,
            direct_e2e_seconds=1.0,
            extra_e2e_seconds=0.3,
        )
        for index in range(12)
    ]
    model = build_frozen_transition_model(
        IDENTITY,
        rows,
        resamples=1_000,
        seed=41,
    )
    estimate, _ = model.lookup(
        context_bucket="coding",
        trigger=Trigger.CALIBRATED_UNCERTAINTY,
        depth=1,
        action=ControllerAction.GENERATE_ALTERNATIVE,
    )
    assert estimate is not None
    assert estimate.conservative_success_lcb > 0.0
    assert estimate.quality_gain_lcb < 0.0


def test_transition_builder_is_seeded_and_input_order_independent() -> None:
    rows = transition_rows()
    shuffled = rows.copy()
    random.Random(99).shuffle(shuffled)
    first = build_frozen_transition_model(IDENTITY, rows, resamples=250, seed=23)
    second = build_frozen_transition_model(
        IDENTITY,
        shuffled,
        resamples=250,
        seed=23,
    )
    lookup = {
        "context_bucket": "coding",
        "trigger": Trigger.VALIDATOR_FAILURE,
        "depth": 1,
        "action": ControllerAction.GENERATE_REPAIR,
    }
    assert first.identity == second.identity == IDENTITY
    assert first.lookup(**lookup) == second.lookup(**lookup)


def test_transition_model_json_round_trip_preserves_bounds_and_identity() -> None:
    fitted = build_frozen_transition_model(
        IDENTITY,
        transition_rows(),
        resamples=100,
        seed=31,
    )
    payload = json.loads(
        json.dumps(frozen_transition_model_to_mapping(fitted), sort_keys=True)
    )
    restored = frozen_transition_model_from_mapping(payload)
    assert restored.identity == fitted.identity == IDENTITY
    assert restored.estimates == fitted.estimates

    legacy = json.loads(json.dumps(payload))
    legacy["schema"] = "mio.markov-transition-calibration.v1"
    for row in legacy["estimates"]:
        row.pop("conservative_quality_gain_lcb")
    restored_legacy = frozen_transition_model_from_mapping(legacy)
    assert all(
        estimate.conservative_quality_gain_lcb is None
        for estimate in restored_legacy.estimates
    )


def test_transition_builder_fails_closed_for_cluster_pseudoreplication() -> None:
    rows = transition_rows(8)
    duplicate = TransitionCalibrationObservation(
        task_cluster_id=rows[0].task_cluster_id,
        context_bucket=rows[0].context_bucket,
        trigger=rows[0].trigger,
        depth=rows[0].depth,
        action=rows[0].action,
        rescued=True,
        extra_output_tokens=99,
        direct_e2e_seconds=1.0,
        extra_e2e_seconds=0.5,
    )
    with pytest.raises(ValueError, match="duplicate task cluster"):
        build_frozen_transition_model(IDENTITY, [*rows, duplicate], resamples=10)


def test_transition_builder_fails_closed_for_any_underpowered_stratum() -> None:
    rows = transition_rows(8)
    underpowered = [
        TransitionCalibrationObservation(
            task_cluster_id=f"alt-{index}",
            context_bucket="coding",
            trigger=Trigger.CALIBRATED_UNCERTAINTY,
            depth=1,
            action=ControllerAction.GENERATE_ALTERNATIVE,
            rescued=True,
            extra_output_tokens=12,
            direct_e2e_seconds=1.0,
            extra_e2e_seconds=0.2,
        )
        for index in range(7)
    ]
    with pytest.raises(InsufficientCalibrationDataError, match="requires 8"):
        build_frozen_transition_model(
            IDENTITY,
            [*rows, *underpowered],
            resamples=10,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_uncertainty", float("nan")),
        ("raw_uncertainty", 1.1),
        ("is_error", 1),
    ],
)
def test_uncertainty_observation_validation(field: str, value: object) -> None:
    kwargs = {
        "task_cluster_id": "task",
        "raw_uncertainty": 0.5,
        "is_error": False,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        UncertaintyCalibrationObservation(**kwargs)


def test_transition_observation_rejects_runtime_or_invalid_actions() -> None:
    kwargs = {
        "task_cluster_id": "task",
        "context_bucket": "coding",
        "trigger": Trigger.VALIDATOR_FAILURE,
        "depth": 1,
        "action": ControllerAction.GENERATE_DIRECT,
        "rescued": False,
        "extra_output_tokens": 10,
        "direct_e2e_seconds": 1.0,
        "extra_e2e_seconds": 0.2,
    }
    with pytest.raises(ValueError, match="extra-generation"):
        TransitionCalibrationObservation(**kwargs)
    kwargs["action"] = "generate_repair"
    with pytest.raises(ValueError, match="extra-generation"):
        TransitionCalibrationObservation(
            **kwargs,
        )
