"""Fail-closed validation statistics for uncertainty-based effort routing.

The independent sampling unit in this module is a task.  Each condition must
therefore contain exactly one predicted probability of error and one hidden
binary outcome per task.  Repeated generations are not accepted as if they
were independent observations.

Three complementary properties are measured:

* AUROC tests discrimination: a larger score must rank errors above correct
  answers.  Equal scores receive half credit.
* Brier score tests probability calibration and sharpness.
* AURC tests selective prediction.  Tasks are accepted in ascending predicted
  error order and AURC is the arithmetic mean of prefix risks.  Within an
  equal-score block, the contribution is the exact expectation under uniform
  random tie-breaking.  Thus an all-tied constant-prevalence router has AURC
  equal to the observed error prevalence.

Confidence intervals use a seeded, ordinary non-parametric task-cluster
percentile bootstrap.  A paired comparison reuses the same resampled task
indices for both signals or conditions; condition outcomes may differ because
the generated answers may differ.  Reliability bins are deliberately
descriptive and never participate in the pre-registered gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import random
from typing import Any, Iterable, Mapping, Sequence


_OBSERVATION_FIELDS = frozenset(
    {"task_cluster_id", "predicted_error_probability", "is_error"}
)
_STATISTICS_SCHEMA = "mio.uncertainty-router-statistics.v1"
_PAIRED_SCHEMA = "mio.paired-uncertainty-router-statistics.v1"
_GATE_SCHEMA = "mio.uncertainty-router-gate.v1"
_BOOTSTRAP_METHOD = "ordinary-task-cluster-percentile-paired-v1"
AURC_DEFINITION = "mean-prefix-risk-ascending-p-error-uniform-tie-expectation-v1"


def _probability(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return result


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_interval_range(
    interval: ConfidenceInterval,
    *,
    label: str,
    lower: float,
    upper: float,
) -> None:
    for field in ("point", "lower", "upper"):
        value = getattr(interval, field)
        if not lower <= value <= upper:
            raise ValueError(f"{label} {field} must be in [{lower}, {upper}]")


@dataclass(frozen=True)
class UncertaintyObservation:
    """One router-visible probability and one offline-only task outcome."""

    task_cluster_id: str
    predicted_error_probability: float
    is_error: bool

    def __post_init__(self) -> None:
        if not isinstance(self.task_cluster_id, str) or not self.task_cluster_id.strip():
            raise ValueError("task_cluster_id must be a non-empty string")
        object.__setattr__(
            self,
            "predicted_error_probability",
            _probability(
                self.predicted_error_probability,
                label="predicted_error_probability",
            ),
        )
        if type(self.is_error) is not bool:
            raise ValueError("is_error must be a bool")

    def to_mapping(self) -> dict[str, str | float | bool]:
        return {
            "task_cluster_id": self.task_cluster_id,
            "predicted_error_probability": self.predicted_error_probability,
            "is_error": self.is_error,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> UncertaintyObservation:
        if not isinstance(value, Mapping) or set(value) != _OBSERVATION_FIELDS:
            raise ValueError("uncertainty observation fields do not match the schema")
        return cls(**{field: value[field] for field in _OBSERVATION_FIELDS})


@dataclass(frozen=True)
class ConfidenceInterval:
    """Point estimate and two-sided percentile confidence interval."""

    point: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        point = _finite(self.point, label="confidence interval point")
        lower = _finite(self.lower, label="confidence interval lower")
        upper = _finite(self.upper, label="confidence interval upper")
        if lower > upper:
            raise ValueError("confidence interval lower bound exceeds upper bound")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_mapping(self) -> dict[str, float]:
        return {"point": self.point, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class PointMetrics:
    """Uncertainty metrics without confidence intervals."""

    auroc: float
    brier: float
    aurc: float

    def __post_init__(self) -> None:
        for field in ("auroc", "brier", "aurc"):
            value = _probability(getattr(self, field), label=field)
            object.__setattr__(self, field, value)

    def to_mapping(self) -> dict[str, float]:
        return {"auroc": self.auroc, "brier": self.brier, "aurc": self.aurc}


@dataclass(frozen=True)
class ReliabilityBin:
    """One descriptive equal-width predicted-error bin."""

    index: int
    lower_bound: float
    upper_bound: float
    upper_inclusive: bool
    task_count: int
    mean_predicted_error_probability: float | None
    observed_error_rate: float | None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("reliability bin index must be a non-negative integer")
        lower = _probability(self.lower_bound, label="reliability bin lower_bound")
        upper = _probability(self.upper_bound, label="reliability bin upper_bound")
        if lower >= upper:
            raise ValueError("reliability bin lower_bound must be below upper_bound")
        if type(self.upper_inclusive) is not bool:
            raise ValueError("reliability bin upper_inclusive must be a bool")
        if type(self.task_count) is not int or self.task_count < 0:
            raise ValueError("reliability bin task_count must be non-negative")
        if self.task_count == 0:
            if self.mean_predicted_error_probability is not None or self.observed_error_rate is not None:
                raise ValueError("empty reliability bins must have null summaries")
        else:
            if self.mean_predicted_error_probability is None or self.observed_error_rate is None:
                raise ValueError("non-empty reliability bins require summaries")
            _probability(
                self.mean_predicted_error_probability,
                label="mean_predicted_error_probability",
            )
            _probability(self.observed_error_rate, label="observed_error_rate")

    def to_mapping(self) -> dict[str, int | float | bool | None]:
        return {
            "index": self.index,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "upper_inclusive": self.upper_inclusive,
            "task_count": self.task_count,
            "mean_predicted_error_probability": self.mean_predicted_error_probability,
            "observed_error_rate": self.observed_error_rate,
        }


@dataclass(frozen=True)
class UncertaintyStatistics:
    """Serializable task-level validation report for one uncertainty signal."""

    signal_name: str
    task_count: int
    error_count: int
    correct_count: int
    error_prevalence: float
    reference_error_probability: float
    reference_source: str
    reference_is_frozen: bool
    auroc: ConfidenceInterval
    brier: ConfidenceInterval
    aurc: ConfidenceInterval
    constant_reference_metrics: PointMetrics
    brier_delta_vs_constant: ConfidenceInterval
    aurc_delta_vs_constant: ConfidenceInterval
    reliability_bins: tuple[ReliabilityBin, ...]
    bootstrap_samples: int
    valid_auroc_bootstrap_samples: int
    invalid_auroc_bootstrap_samples: int
    bootstrap_seed: int
    confidence: float
    bootstrap_method: str = _BOOTSTRAP_METHOD
    aurc_definition: str = AURC_DEFINITION

    def __post_init__(self) -> None:
        if not isinstance(self.signal_name, str) or not self.signal_name.strip():
            raise ValueError("signal_name must be a non-empty string")
        _positive_integer(self.task_count, label="task_count")
        _positive_integer(self.error_count, label="error_count")
        _positive_integer(self.correct_count, label="correct_count")
        if self.error_count + self.correct_count != self.task_count:
            raise ValueError("error_count and correct_count must sum to task_count")
        _probability(self.error_prevalence, label="error_prevalence")
        if not all(
            isinstance(value, ConfidenceInterval)
            for value in (
                self.auroc,
                self.brier,
                self.aurc,
                self.brier_delta_vs_constant,
                self.aurc_delta_vs_constant,
            )
        ):
            raise ValueError("metric intervals must be ConfidenceInterval values")
        for field in ("auroc", "brier", "aurc"):
            _validate_interval_range(
                getattr(self, field),
                label=field,
                lower=0.0,
                upper=1.0,
            )
        for field in ("brier_delta_vs_constant", "aurc_delta_vs_constant"):
            _validate_interval_range(
                getattr(self, field),
                label=field,
                lower=-1.0,
                upper=1.0,
            )
        if not isinstance(self.constant_reference_metrics, PointMetrics):
            raise ValueError("constant_reference_metrics must be PointMetrics")
        expected_prevalence = self.error_count / self.task_count
        if not math.isclose(self.error_prevalence, expected_prevalence, abs_tol=1e-15):
            raise ValueError("error_prevalence does not match task outcomes")
        reference_probability = _probability(
            self.reference_error_probability,
            label="reference_error_probability",
        )
        if not isinstance(self.reference_source, str) or not self.reference_source.strip():
            raise ValueError("reference_source must be a non-empty string")
        if type(self.reference_is_frozen) is not bool:
            raise ValueError("reference_is_frozen must be a bool")
        expected_constant_brier = (
            expected_prevalence * (1.0 - reference_probability) ** 2
            + (1.0 - expected_prevalence) * reference_probability**2
        )
        if not math.isclose(self.constant_reference_metrics.auroc, 0.5, abs_tol=1e-15):
            raise ValueError("constant-reference AUROC must equal 0.5")
        if not math.isclose(
            self.constant_reference_metrics.brier,
            expected_constant_brier,
            abs_tol=1e-15,
        ):
            raise ValueError("constant-reference Brier score does not match outcomes")
        if not math.isclose(
            self.constant_reference_metrics.aurc,
            expected_prevalence,
            abs_tol=1e-15,
        ):
            raise ValueError("constant-reference AURC does not match prevalence")
        if not math.isclose(
            self.brier_delta_vs_constant.point,
            self.brier.point - self.constant_reference_metrics.brier,
            abs_tol=1e-15,
        ):
            raise ValueError("Brier delta point does not match reported metrics")
        if not math.isclose(
            self.aurc_delta_vs_constant.point,
            self.aurc.point - self.constant_reference_metrics.aurc,
            abs_tol=1e-15,
        ):
            raise ValueError("AURC delta point does not match reported metrics")
        object.__setattr__(self, "reliability_bins", tuple(self.reliability_bins))
        if not self.reliability_bins or any(
            not isinstance(value, ReliabilityBin) for value in self.reliability_bins
        ):
            raise ValueError("reliability_bins must contain ReliabilityBin values")
        bin_count = len(self.reliability_bins)
        if sum(value.task_count for value in self.reliability_bins) != self.task_count:
            raise ValueError("reliability bin task counts must sum to task_count")
        for index, value in enumerate(self.reliability_bins):
            if value.index != index:
                raise ValueError("reliability bin indices must be contiguous")
            if not math.isclose(value.lower_bound, index / bin_count, abs_tol=1e-15):
                raise ValueError("reliability bin lower bounds must be equal-width")
            if not math.isclose(
                value.upper_bound,
                (index + 1) / bin_count,
                abs_tol=1e-15,
            ):
                raise ValueError("reliability bin upper bounds must be equal-width")
            if value.upper_inclusive != (index == bin_count - 1):
                raise ValueError("only the last reliability bin may include its upper bound")
        _positive_integer(self.bootstrap_samples, label="bootstrap_samples")
        _positive_integer(
            self.valid_auroc_bootstrap_samples,
            label="valid_auroc_bootstrap_samples",
        )
        if type(self.invalid_auroc_bootstrap_samples) is not int or self.invalid_auroc_bootstrap_samples < 0:
            raise ValueError("invalid_auroc_bootstrap_samples must be non-negative")
        if self.valid_auroc_bootstrap_samples + self.invalid_auroc_bootstrap_samples != self.bootstrap_samples:
            raise ValueError("AUROC bootstrap sample counts must sum to bootstrap_samples")
        if type(self.bootstrap_seed) is not int:
            raise ValueError("bootstrap_seed must be an integer")
        if not math.isfinite(self.confidence) or not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be finite and in (0, 1)")
        if self.bootstrap_method != _BOOTSTRAP_METHOD:
            raise ValueError("unsupported bootstrap method")
        if self.aurc_definition != AURC_DEFINITION:
            raise ValueError("unsupported AURC definition")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": _STATISTICS_SCHEMA,
            "signal_name": self.signal_name,
            "task_count": self.task_count,
            "error_count": self.error_count,
            "correct_count": self.correct_count,
            "error_prevalence": self.error_prevalence,
            "constant_reference": {
                "predicted_error_probability": self.reference_error_probability,
                "source": self.reference_source,
                "frozen_before_evaluation": self.reference_is_frozen,
                "metrics": self.constant_reference_metrics.to_mapping(),
            },
            "auroc": self.auroc.to_mapping(),
            "brier": self.brier.to_mapping(),
            "aurc": self.aurc.to_mapping(),
            "brier_delta_vs_constant": self.brier_delta_vs_constant.to_mapping(),
            "aurc_delta_vs_constant": self.aurc_delta_vs_constant.to_mapping(),
            "reliability_bins": [value.to_mapping() for value in self.reliability_bins],
            "bootstrap": {
                "method": self.bootstrap_method,
                "samples": self.bootstrap_samples,
                "valid_auroc_samples": self.valid_auroc_bootstrap_samples,
                "invalid_auroc_samples": self.invalid_auroc_bootstrap_samples,
                "seed": self.bootstrap_seed,
                "confidence": self.confidence,
            },
            "aurc_definition": self.aurc_definition,
        }


@dataclass(frozen=True)
class PairedUncertaintyStatistics:
    """Serializable paired candidate-minus-reference signal comparison."""

    reference_signal: str
    candidate_signal: str
    task_count: int
    reference_error_count: int
    reference_correct_count: int
    candidate_error_count: int
    candidate_correct_count: int
    reference_metrics: PointMetrics
    candidate_metrics: PointMetrics
    auroc_delta: ConfidenceInterval
    brier_delta: ConfidenceInterval
    aurc_delta: ConfidenceInterval
    bootstrap_samples: int
    valid_auroc_bootstrap_samples: int
    invalid_auroc_bootstrap_samples: int
    bootstrap_seed: int
    confidence: float
    bootstrap_method: str = _BOOTSTRAP_METHOD
    aurc_definition: str = AURC_DEFINITION

    def __post_init__(self) -> None:
        for field in ("reference_signal", "candidate_signal"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if self.reference_signal == self.candidate_signal:
            raise ValueError("reference_signal and candidate_signal must differ")
        _positive_integer(self.task_count, label="task_count")
        for condition in ("reference", "candidate"):
            errors = getattr(self, f"{condition}_error_count")
            correct = getattr(self, f"{condition}_correct_count")
            _positive_integer(errors, label=f"{condition}_error_count")
            _positive_integer(correct, label=f"{condition}_correct_count")
            if errors + correct != self.task_count:
                raise ValueError(
                    f"{condition}_error_count and {condition}_correct_count "
                    "must sum to task_count"
                )
        if not isinstance(self.reference_metrics, PointMetrics) or not isinstance(
            self.candidate_metrics, PointMetrics
        ):
            raise ValueError("paired point metrics must be PointMetrics values")
        if not all(
            isinstance(value, ConfidenceInterval)
            for value in (self.auroc_delta, self.brier_delta, self.aurc_delta)
        ):
            raise ValueError("paired deltas must be ConfidenceInterval values")
        for metrics_label, metrics in (
            ("reference", self.reference_metrics),
            ("candidate", self.candidate_metrics),
        ):
            for field in ("auroc", "brier", "aurc"):
                _probability(getattr(metrics, field), label=f"{metrics_label}_{field}")
        for field in ("auroc_delta", "brier_delta", "aurc_delta"):
            _validate_interval_range(
                getattr(self, field),
                label=field,
                lower=-1.0,
                upper=1.0,
            )
        for field in ("auroc", "brier", "aurc"):
            reported = getattr(self, f"{field}_delta").point
            expected = getattr(self.candidate_metrics, field) - getattr(
                self.reference_metrics,
                field,
            )
            if not math.isclose(reported, expected, abs_tol=1e-15):
                raise ValueError(f"{field} delta point does not match reported metrics")
        _positive_integer(self.bootstrap_samples, label="bootstrap_samples")
        _positive_integer(
            self.valid_auroc_bootstrap_samples,
            label="valid_auroc_bootstrap_samples",
        )
        if type(self.invalid_auroc_bootstrap_samples) is not int or self.invalid_auroc_bootstrap_samples < 0:
            raise ValueError("invalid_auroc_bootstrap_samples must be non-negative")
        if self.valid_auroc_bootstrap_samples + self.invalid_auroc_bootstrap_samples != self.bootstrap_samples:
            raise ValueError("AUROC bootstrap sample counts must sum to bootstrap_samples")
        if type(self.bootstrap_seed) is not int:
            raise ValueError("bootstrap_seed must be an integer")
        if not math.isfinite(self.confidence) or not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be finite and in (0, 1)")
        if self.bootstrap_method != _BOOTSTRAP_METHOD:
            raise ValueError("unsupported bootstrap method")
        if self.aurc_definition != AURC_DEFINITION:
            raise ValueError("unsupported AURC definition")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": _PAIRED_SCHEMA,
            "reference_signal": self.reference_signal,
            "candidate_signal": self.candidate_signal,
            "task_count": self.task_count,
            "reference_error_count": self.reference_error_count,
            "reference_correct_count": self.reference_correct_count,
            "candidate_error_count": self.candidate_error_count,
            "candidate_correct_count": self.candidate_correct_count,
            "reference_metrics": self.reference_metrics.to_mapping(),
            "candidate_metrics": self.candidate_metrics.to_mapping(),
            "deltas": {
                "direction": "candidate_minus_reference",
                "auroc": self.auroc_delta.to_mapping(),
                "brier": self.brier_delta.to_mapping(),
                "aurc": self.aurc_delta.to_mapping(),
            },
            "bootstrap": {
                "method": self.bootstrap_method,
                "samples": self.bootstrap_samples,
                "valid_auroc_samples": self.valid_auroc_bootstrap_samples,
                "invalid_auroc_samples": self.invalid_auroc_bootstrap_samples,
                "seed": self.bootstrap_seed,
                "confidence": self.confidence,
            },
            "aurc_definition": self.aurc_definition,
        }


@dataclass(frozen=True)
class PreregisteredUncertaintyGatePolicy:
    """Thresholds that must be frozen before evaluating a held-out split."""

    min_tasks: int = 100
    min_errors: int = 20
    min_correct: int = 20
    min_auroc_lower_bound: float = 0.5
    min_brier_improvement: float = 0.0
    max_aurc_regression: float = 0.0
    min_valid_auroc_bootstrap_fraction: float = 0.95

    def __post_init__(self) -> None:
        _positive_integer(self.min_tasks, label="min_tasks")
        _positive_integer(self.min_errors, label="min_errors")
        _positive_integer(self.min_correct, label="min_correct")
        _probability(self.min_auroc_lower_bound, label="min_auroc_lower_bound")
        _probability(self.min_brier_improvement, label="min_brier_improvement")
        _probability(self.max_aurc_regression, label="max_aurc_regression")
        _probability(
            self.min_valid_auroc_bootstrap_fraction,
            label="min_valid_auroc_bootstrap_fraction",
        )

    def to_mapping(self) -> dict[str, int | float]:
        return {
            "min_tasks": self.min_tasks,
            "min_errors": self.min_errors,
            "min_correct": self.min_correct,
            "min_auroc_lower_bound": self.min_auroc_lower_bound,
            "min_brier_improvement": self.min_brier_improvement,
            "max_aurc_regression": self.max_aurc_regression,
            "min_valid_auroc_bootstrap_fraction": self.min_valid_auroc_bootstrap_fraction,
        }


class UncertaintyGateFailure(StrEnum):
    REFERENCE_NOT_FROZEN = "reference_not_frozen"
    INSUFFICIENT_TASKS = "insufficient_tasks"
    INSUFFICIENT_ERRORS = "insufficient_errors"
    INSUFFICIENT_CORRECT = "insufficient_correct"
    INSUFFICIENT_AUROC_BOOTSTRAP_SUPPORT = "insufficient_auroc_bootstrap_support"
    AUROC_LOWER_BOUND_NOT_ABOVE_CHANCE = "auroc_lower_bound_not_above_chance"
    BRIER_NOT_BETTER_THAN_CONSTANT_PREVALENCE = "brier_not_better_than_constant_prevalence"
    AURC_WORSE_THAN_CONSTANT_PREVALENCE = "aurc_worse_than_constant_prevalence"


@dataclass(frozen=True)
class PreregisteredUncertaintyGateResult:
    passed: bool
    failures: tuple[UncertaintyGateFailure, ...]
    policy: PreregisteredUncertaintyGatePolicy

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("passed must be a bool")
        object.__setattr__(self, "failures", tuple(self.failures))
        if any(not isinstance(value, UncertaintyGateFailure) for value in self.failures):
            raise ValueError("failures must contain UncertaintyGateFailure values")
        if self.passed != (not self.failures):
            raise ValueError("passed must agree with failures")
        if not isinstance(self.policy, PreregisteredUncertaintyGatePolicy):
            raise ValueError("policy must be a PreregisteredUncertaintyGatePolicy")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": _GATE_SCHEMA,
            "passed": self.passed,
            "failures": [value.value for value in self.failures],
            "policy": self.policy.to_mapping(),
        }


def _validate_metric_inputs(
    predicted_error_probabilities: Iterable[float],
    error_outcomes: Iterable[bool],
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    probabilities = tuple(
        _probability(value, label="predicted_error_probability")
        for value in predicted_error_probabilities
    )
    outcomes = tuple(error_outcomes)
    if not probabilities:
        raise ValueError("metric inputs must not be empty")
    if len(probabilities) != len(outcomes):
        raise ValueError("metric inputs must have equal length")
    if any(type(value) is not bool for value in outcomes):
        raise ValueError("error outcomes must be bool values")
    return probabilities, outcomes


def tie_corrected_auroc(
    predicted_error_probabilities: Iterable[float],
    error_outcomes: Iterable[bool],
) -> float:
    """Return error-detection AUROC, awarding half credit to score ties."""

    probabilities, outcomes = _validate_metric_inputs(
        predicted_error_probabilities,
        error_outcomes,
    )
    errors = sum(outcomes)
    correct = len(outcomes) - errors
    if errors == 0 or correct == 0:
        raise ValueError("AUROC requires at least one error and one correct task")

    ordered = sorted(zip(probabilities, outcomes), key=lambda value: value[0])
    correct_below = 0
    concordance = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        block = ordered[index:end]
        block_errors = sum(outcome for _, outcome in block)
        block_correct = len(block) - block_errors
        concordance += block_errors * correct_below
        concordance += 0.5 * block_errors * block_correct
        correct_below += block_correct
        index = end
    return concordance / (errors * correct)


def brier_score(
    predicted_error_probabilities: Iterable[float],
    error_outcomes: Iterable[bool],
) -> float:
    """Return the mean squared error of predicted error probabilities."""

    probabilities, outcomes = _validate_metric_inputs(
        predicted_error_probabilities,
        error_outcomes,
    )
    return sum((probability - float(outcome)) ** 2 for probability, outcome in zip(probabilities, outcomes)) / len(
        outcomes
    )


def area_under_risk_coverage_curve(
    predicted_error_probabilities: Iterable[float],
    error_outcomes: Iterable[bool],
) -> float:
    """Return mean prefix risk after accepting tasks from low to high error P.

    The discrete curve has one coverage point per accepted task, including
    full coverage.  For a tied score block, every within-block prefix uses its
    exact expected error count under uniform random tie-breaking.  This avoids
    dependence on input order or task identifiers.
    """

    probabilities, outcomes = _validate_metric_inputs(
        predicted_error_probabilities,
        error_outcomes,
    )
    ordered = sorted(zip(probabilities, outcomes), key=lambda value: value[0])
    accepted_before = 0
    errors_before = 0
    risk_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        block = ordered[index:end]
        block_size = len(block)
        block_errors = sum(outcome for _, outcome in block)
        for offset in range(1, block_size + 1):
            expected_errors = errors_before + offset * block_errors / block_size
            risk_sum += expected_errors / (accepted_before + offset)
        accepted_before += block_size
        errors_before += block_errors
        index = end
    return risk_sum / len(ordered)


def _point_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
) -> PointMetrics:
    return PointMetrics(
        auroc=tie_corrected_auroc(probabilities, outcomes),
        brier=brier_score(probabilities, outcomes),
        aurc=area_under_risk_coverage_curve(probabilities, outcomes),
    )


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _interval(point: float, samples: Sequence[float], confidence: float) -> ConfidenceInterval:
    if not samples:
        raise ValueError("confidence interval requires at least one valid bootstrap sample")
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        point=point,
        lower=_percentile(samples, tail),
        upper=_percentile(samples, 1.0 - tail),
    )


def _coerce_observation(
    value: UncertaintyObservation | Mapping[str, Any],
) -> UncertaintyObservation:
    if isinstance(value, UncertaintyObservation):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("observations must be UncertaintyObservation objects or mappings")
    return UncertaintyObservation.from_mapping(value)


def _index_observations(
    observations: Iterable[UncertaintyObservation | Mapping[str, Any]],
) -> tuple[UncertaintyObservation, ...]:
    indexed: dict[str, UncertaintyObservation] = {}
    for raw in observations:
        observation = _coerce_observation(raw)
        if observation.task_cluster_id in indexed:
            raise ValueError(
                f"duplicate task_cluster_id: {observation.task_cluster_id!r}; "
                "task-level metrics require exactly one row per task"
            )
        indexed[observation.task_cluster_id] = observation
    if not indexed:
        raise ValueError("observations must not be empty")
    ordered = tuple(indexed[key] for key in sorted(indexed))
    errors = sum(value.is_error for value in ordered)
    if errors == 0 or errors == len(ordered):
        raise ValueError("uncertainty validation requires at least one error and one correct task")
    return ordered


def _validate_analysis_arguments(
    *,
    signal_name: str,
    bootstrap_samples: int,
    seed: int,
    confidence: float,
    reliability_bin_count: int,
) -> None:
    if not isinstance(signal_name, str) or not signal_name.strip():
        raise ValueError("signal_name must be a non-empty string")
    _positive_integer(bootstrap_samples, label="bootstrap_samples")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")
    _positive_integer(reliability_bin_count, label="reliability_bin_count")


def _reliability_bins(
    observations: Sequence[UncertaintyObservation],
    bin_count: int,
) -> tuple[ReliabilityBin, ...]:
    members: list[list[UncertaintyObservation]] = [[] for _ in range(bin_count)]
    for observation in observations:
        index = min(
            int(observation.predicted_error_probability * bin_count),
            bin_count - 1,
        )
        members[index].append(observation)
    result: list[ReliabilityBin] = []
    for index, values in enumerate(members):
        count = len(values)
        result.append(
            ReliabilityBin(
                index=index,
                lower_bound=index / bin_count,
                upper_bound=(index + 1) / bin_count,
                upper_inclusive=index == bin_count - 1,
                task_count=count,
                mean_predicted_error_probability=(
                    sum(value.predicted_error_probability for value in values) / count
                    if count
                    else None
                ),
                observed_error_rate=(
                    sum(value.is_error for value in values) / count if count else None
                ),
            )
        )
    return tuple(result)


def analyze_uncertainty(
    observations: Iterable[UncertaintyObservation | Mapping[str, Any]],
    *,
    signal_name: str,
    bootstrap_samples: int = 10_000,
    seed: int = 20260718,
    confidence: float = 0.95,
    reliability_bin_count: int = 10,
    reference_error_probability: float | None = None,
    reference_source: str | None = None,
    reference_is_frozen: bool = False,
) -> UncertaintyStatistics:
    """Analyze one calibrated router signal against a constant reference.

    Confirmatory use must pass a probability learned outside the evaluation
    split, a non-empty provenance string, and ``reference_is_frozen=True``.
    Omitting the probability uses observed evaluation prevalence only for
    descriptive analysis; that report is explicitly marked non-frozen and is
    rejected by :func:`evaluate_preregistered_uncertainty_gate`.
    """

    _validate_analysis_arguments(
        signal_name=signal_name,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        confidence=confidence,
        reliability_bin_count=reliability_bin_count,
    )
    if type(reference_is_frozen) is not bool:
        raise ValueError("reference_is_frozen must be a bool")
    if reference_error_probability is None:
        if reference_source is not None:
            raise ValueError(
                "reference_source requires an explicit reference_error_probability"
            )
        if reference_is_frozen:
            raise ValueError(
                "an observed evaluation prevalence cannot be declared frozen"
            )
    else:
        reference_error_probability = _probability(
            reference_error_probability,
            label="reference_error_probability",
        )
        if not isinstance(reference_source, str) or not reference_source.strip():
            raise ValueError(
                "explicit reference_error_probability requires a non-empty reference_source"
            )
    ordered = _index_observations(observations)
    probabilities = tuple(value.predicted_error_probability for value in ordered)
    outcomes = tuple(value.is_error for value in ordered)
    count = len(ordered)
    errors = sum(outcomes)
    prevalence = errors / count
    if reference_error_probability is None:
        resolved_reference_probability = prevalence
        resolved_reference_source = "observed-evaluation-prevalence-descriptive-only"
    else:
        resolved_reference_probability = reference_error_probability
        assert reference_source is not None
        resolved_reference_source = reference_source
    point = _point_metrics(probabilities, outcomes)
    constant_probabilities = (resolved_reference_probability,) * count
    constant = _point_metrics(constant_probabilities, outcomes)

    rng = random.Random(seed)
    auroc_samples: list[float] = []
    brier_samples: list[float] = []
    aurc_samples: list[float] = []
    brier_delta_samples: list[float] = []
    aurc_delta_samples: list[float] = []
    invalid_auroc_samples = 0
    for _ in range(bootstrap_samples):
        indices = tuple(rng.randrange(count) for _ in range(count))
        sample_probabilities = tuple(probabilities[index] for index in indices)
        sample_outcomes = tuple(outcomes[index] for index in indices)
        sample_errors = sum(sample_outcomes)
        sample_brier = brier_score(sample_probabilities, sample_outcomes)
        sample_aurc = area_under_risk_coverage_curve(
            sample_probabilities,
            sample_outcomes,
        )
        sample_constant_probabilities = (resolved_reference_probability,) * count
        constant_brier = brier_score(sample_constant_probabilities, sample_outcomes)
        constant_aurc = area_under_risk_coverage_curve(
            sample_constant_probabilities,
            sample_outcomes,
        )
        brier_samples.append(sample_brier)
        aurc_samples.append(sample_aurc)
        brier_delta_samples.append(sample_brier - constant_brier)
        aurc_delta_samples.append(sample_aurc - constant_aurc)
        if sample_errors == 0 or sample_errors == count:
            invalid_auroc_samples += 1
        else:
            auroc_samples.append(
                tie_corrected_auroc(sample_probabilities, sample_outcomes)
            )
    if not auroc_samples:
        raise ValueError("bootstrap produced no class-valid AUROC samples")

    return UncertaintyStatistics(
        signal_name=signal_name,
        task_count=count,
        error_count=errors,
        correct_count=count - errors,
        error_prevalence=prevalence,
        reference_error_probability=resolved_reference_probability,
        reference_source=resolved_reference_source,
        reference_is_frozen=reference_is_frozen,
        auroc=_interval(point.auroc, auroc_samples, confidence),
        brier=_interval(point.brier, brier_samples, confidence),
        aurc=_interval(point.aurc, aurc_samples, confidence),
        constant_reference_metrics=constant,
        brier_delta_vs_constant=_interval(
            point.brier - constant.brier,
            brier_delta_samples,
            confidence,
        ),
        aurc_delta_vs_constant=_interval(
            point.aurc - constant.aurc,
            aurc_delta_samples,
            confidence,
        ),
        reliability_bins=_reliability_bins(ordered, reliability_bin_count),
        bootstrap_samples=bootstrap_samples,
        valid_auroc_bootstrap_samples=len(auroc_samples),
        invalid_auroc_bootstrap_samples=invalid_auroc_samples,
        bootstrap_seed=seed,
        confidence=confidence,
    )


def analyze_paired_uncertainty(
    reference_observations: Iterable[UncertaintyObservation | Mapping[str, Any]],
    candidate_observations: Iterable[UncertaintyObservation | Mapping[str, Any]],
    *,
    reference_signal: str,
    candidate_signal: str,
    bootstrap_samples: int = 10_000,
    seed: int = 20260718,
    confidence: float = 0.95,
) -> PairedUncertaintyStatistics:
    """Compare two signals or conditions with paired candidate-minus-reference deltas.

    The task manifests must match exactly.  Each condition is scored against
    its own outcome, allowing a decoding condition to change correctness as
    well as the uncertainty signal.  Shared bootstrap indices preserve the
    task-level pairing.
    """

    _validate_analysis_arguments(
        signal_name=reference_signal,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        confidence=confidence,
        reliability_bin_count=1,
    )
    if not isinstance(candidate_signal, str) or not candidate_signal.strip():
        raise ValueError("candidate_signal must be a non-empty string")
    if reference_signal == candidate_signal:
        raise ValueError("reference_signal and candidate_signal must differ")
    reference = _index_observations(reference_observations)
    candidate = _index_observations(candidate_observations)
    reference_by_task = {value.task_cluster_id: value for value in reference}
    candidate_by_task = {value.task_cluster_id: value for value in candidate}
    if set(reference_by_task) != set(candidate_by_task):
        raise ValueError("paired signals must contain exactly the same task_cluster_ids")

    task_ids = tuple(sorted(reference_by_task))
    reference_probabilities: list[float] = []
    candidate_probabilities: list[float] = []
    reference_outcomes: list[bool] = []
    candidate_outcomes: list[bool] = []
    for task_id in task_ids:
        reference_value = reference_by_task[task_id]
        candidate_value = candidate_by_task[task_id]
        reference_probabilities.append(reference_value.predicted_error_probability)
        candidate_probabilities.append(candidate_value.predicted_error_probability)
        reference_outcomes.append(reference_value.is_error)
        candidate_outcomes.append(candidate_value.is_error)

    reference_scores = tuple(reference_probabilities)
    candidate_scores = tuple(candidate_probabilities)
    reference_outcome_values = tuple(reference_outcomes)
    candidate_outcome_values = tuple(candidate_outcomes)
    reference_point = _point_metrics(reference_scores, reference_outcome_values)
    candidate_point = _point_metrics(candidate_scores, candidate_outcome_values)
    count = len(task_ids)

    rng = random.Random(seed)
    auroc_delta_samples: list[float] = []
    brier_delta_samples: list[float] = []
    aurc_delta_samples: list[float] = []
    invalid_auroc_samples = 0
    for _ in range(bootstrap_samples):
        indices = tuple(rng.randrange(count) for _ in range(count))
        sample_reference = tuple(reference_scores[index] for index in indices)
        sample_candidate = tuple(candidate_scores[index] for index in indices)
        sample_reference_outcomes = tuple(
            reference_outcome_values[index] for index in indices
        )
        sample_candidate_outcomes = tuple(
            candidate_outcome_values[index] for index in indices
        )
        sample_reference_errors = sum(sample_reference_outcomes)
        sample_candidate_errors = sum(sample_candidate_outcomes)
        brier_delta_samples.append(
            brier_score(sample_candidate, sample_candidate_outcomes)
            - brier_score(sample_reference, sample_reference_outcomes)
        )
        aurc_delta_samples.append(
            area_under_risk_coverage_curve(
                sample_candidate,
                sample_candidate_outcomes,
            )
            - area_under_risk_coverage_curve(
                sample_reference,
                sample_reference_outcomes,
            )
        )
        if (
            sample_reference_errors in (0, count)
            or sample_candidate_errors in (0, count)
        ):
            invalid_auroc_samples += 1
        else:
            auroc_delta_samples.append(
                tie_corrected_auroc(sample_candidate, sample_candidate_outcomes)
                - tie_corrected_auroc(sample_reference, sample_reference_outcomes)
            )
    if not auroc_delta_samples:
        raise ValueError("bootstrap produced no class-valid paired AUROC samples")

    return PairedUncertaintyStatistics(
        reference_signal=reference_signal,
        candidate_signal=candidate_signal,
        task_count=count,
        reference_error_count=sum(reference_outcome_values),
        reference_correct_count=count - sum(reference_outcome_values),
        candidate_error_count=sum(candidate_outcome_values),
        candidate_correct_count=count - sum(candidate_outcome_values),
        reference_metrics=reference_point,
        candidate_metrics=candidate_point,
        auroc_delta=_interval(
            candidate_point.auroc - reference_point.auroc,
            auroc_delta_samples,
            confidence,
        ),
        brier_delta=_interval(
            candidate_point.brier - reference_point.brier,
            brier_delta_samples,
            confidence,
        ),
        aurc_delta=_interval(
            candidate_point.aurc - reference_point.aurc,
            aurc_delta_samples,
            confidence,
        ),
        bootstrap_samples=bootstrap_samples,
        valid_auroc_bootstrap_samples=len(auroc_delta_samples),
        invalid_auroc_bootstrap_samples=invalid_auroc_samples,
        bootstrap_seed=seed,
        confidence=confidence,
    )


def evaluate_preregistered_uncertainty_gate(
    statistics: UncertaintyStatistics,
    *,
    policy: PreregisteredUncertaintyGatePolicy | None = None,
) -> PreregisteredUncertaintyGateResult:
    """Evaluate a held-out router signal against frozen fail-closed gates.

    Brier and AURC deltas are signal minus the all-tied constant reference.
    Consequently, negative values are improvements.  Brier requires a
    strictly negative upper confidence bound; AURC permits equality.  The
    reference probability and its provenance must have been frozen before the
    evaluation split was opened.
    """

    if not isinstance(statistics, UncertaintyStatistics):
        raise ValueError("statistics must be an UncertaintyStatistics result")
    selected = policy or PreregisteredUncertaintyGatePolicy()
    if not isinstance(selected, PreregisteredUncertaintyGatePolicy):
        raise ValueError("policy must be a PreregisteredUncertaintyGatePolicy")
    failures: list[UncertaintyGateFailure] = []
    if not statistics.reference_is_frozen:
        failures.append(UncertaintyGateFailure.REFERENCE_NOT_FROZEN)
    if statistics.task_count < selected.min_tasks:
        failures.append(UncertaintyGateFailure.INSUFFICIENT_TASKS)
    if statistics.error_count < selected.min_errors:
        failures.append(UncertaintyGateFailure.INSUFFICIENT_ERRORS)
    if statistics.correct_count < selected.min_correct:
        failures.append(UncertaintyGateFailure.INSUFFICIENT_CORRECT)
    valid_fraction = statistics.valid_auroc_bootstrap_samples / statistics.bootstrap_samples
    if valid_fraction < selected.min_valid_auroc_bootstrap_fraction:
        failures.append(UncertaintyGateFailure.INSUFFICIENT_AUROC_BOOTSTRAP_SUPPORT)
    if statistics.auroc.lower <= selected.min_auroc_lower_bound:
        failures.append(UncertaintyGateFailure.AUROC_LOWER_BOUND_NOT_ABOVE_CHANCE)
    if statistics.brier_delta_vs_constant.upper >= -selected.min_brier_improvement:
        failures.append(
            UncertaintyGateFailure.BRIER_NOT_BETTER_THAN_CONSTANT_PREVALENCE
        )
    if statistics.aurc_delta_vs_constant.upper > selected.max_aurc_regression:
        failures.append(UncertaintyGateFailure.AURC_WORSE_THAN_CONSTANT_PREVALENCE)
    return PreregisteredUncertaintyGateResult(
        passed=not failures,
        failures=tuple(failures),
        policy=selected,
    )


__all__ = [
    "AURC_DEFINITION",
    "ConfidenceInterval",
    "PairedUncertaintyStatistics",
    "PointMetrics",
    "PreregisteredUncertaintyGatePolicy",
    "PreregisteredUncertaintyGateResult",
    "ReliabilityBin",
    "UncertaintyGateFailure",
    "UncertaintyObservation",
    "UncertaintyStatistics",
    "analyze_paired_uncertainty",
    "analyze_uncertainty",
    "area_under_risk_coverage_curve",
    "brier_score",
    "evaluate_preregistered_uncertainty_gate",
    "tie_corrected_auroc",
]
