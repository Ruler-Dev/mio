"""Offline calibration for the deterministic Markov effort controller.

The functions in this module are deliberately separated from request-time
routing.  Outcome labels are inputs to the two ``fit_*``/``build_*``
functions only; the frozen runtime objects contain calibrated probabilities
and conservative resource bounds, never task outcomes.

No optional statistics dependency is required.  Uncertainty calibration uses
weighted pool-adjacent-violators (PAVA), and transition bounds use a seeded
one-sided task-cluster percentile bootstrap.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Iterable, Mapping

from experimental.markov_effort_controller import (
    BootstrapMetadata,
    CalibrationIdentity,
    ControllerAction,
    EXTRA_ACTIONS,
    FrozenTransitionModel,
    TransitionEstimate,
    Trigger,
)


_IDENTITY_FIELDS = (
    "model",
    "config",
    "prompt",
    "sampler",
    "corpus",
    "split",
    "backend",
)
_CALIBRATOR_SCHEMA = "mio.isotonic-error-calibration.v1"
_TRANSITION_MODEL_SCHEMA = "mio.markov-transition-calibration.v1"
_BOOTSTRAP_METHOD = "task-cluster-percentile-one-sided-v1"


class InsufficientCalibrationDataError(ValueError):
    """Raised instead of publishing an underpowered calibration artifact."""


def _finite_probability(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return result


def _positive_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _identity_to_mapping(identity: CalibrationIdentity) -> dict[str, str]:
    return {field: getattr(identity, field) for field in _IDENTITY_FIELDS}


def _identity_from_mapping(value: Any) -> CalibrationIdentity:
    if not isinstance(value, Mapping):
        raise ValueError("calibration identity must be a mapping")
    if set(value) != set(_IDENTITY_FIELDS):
        raise ValueError("calibration identity fields do not match the schema")
    if any(not isinstance(value[field], str) for field in _IDENTITY_FIELDS):
        raise ValueError("calibration identity values must be strings")
    return CalibrationIdentity(**{field: value[field] for field in _IDENTITY_FIELDS})


@dataclass(frozen=True)
class UncertaintyCalibrationObservation:
    """Offline-only row containing the hidden error outcome."""

    task_cluster_id: str
    raw_uncertainty: float
    is_error: bool

    def __post_init__(self) -> None:
        if not isinstance(self.task_cluster_id, str) or not self.task_cluster_id.strip():
            raise ValueError("task_cluster_id must be a non-empty string")
        object.__setattr__(
            self,
            "raw_uncertainty",
            _finite_probability(self.raw_uncertainty, label="raw_uncertainty"),
        )
        if type(self.is_error) is not bool:
            raise ValueError("is_error must be a bool")


@dataclass(frozen=True)
class IsotonicCalibrationRow:
    """One immutable runtime knot in the monotone calibration curve."""

    raw_uncertainty: float
    error_probability: float
    observation_count: int
    task_cluster_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_uncertainty",
            _finite_probability(self.raw_uncertainty, label="raw_uncertainty"),
        )
        object.__setattr__(
            self,
            "error_probability",
            _finite_probability(self.error_probability, label="error_probability"),
        )
        if type(self.observation_count) is not int or self.observation_count < 1:
            raise ValueError("observation_count must be a positive integer")
        if type(self.task_cluster_count) is not int or self.task_cluster_count < 1:
            raise ValueError("task_cluster_count must be a positive integer")
        if self.task_cluster_count > self.observation_count:
            raise ValueError("task_cluster_count cannot exceed observation_count")

    def to_mapping(self) -> dict[str, int | float]:
        return {
            "raw_uncertainty": self.raw_uncertainty,
            "error_probability": self.error_probability,
            "observation_count": self.observation_count,
            "task_cluster_count": self.task_cluster_count,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> IsotonicCalibrationRow:
        fields = {
            "raw_uncertainty",
            "error_probability",
            "observation_count",
            "task_cluster_count",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("isotonic row fields do not match the schema")
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True)
class FrozenUncertaintyCalibrator:
    """Serializable label-free runtime transform from uncertainty to P(error)."""

    identity: CalibrationIdentity
    rows: tuple[IsotonicCalibrationRow, ...]
    observation_count: int
    task_cluster_count: int
    method: str = "weighted-pava-linear-interpolation-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CalibrationIdentity):
            raise ValueError("identity must be a CalibrationIdentity")
        object.__setattr__(self, "rows", tuple(self.rows))
        if not self.rows:
            raise ValueError("calibration rows must not be empty")
        if any(not isinstance(row, IsotonicCalibrationRow) for row in self.rows):
            raise ValueError("rows must contain IsotonicCalibrationRow values")
        raw_values = tuple(row.raw_uncertainty for row in self.rows)
        probabilities = tuple(row.error_probability for row in self.rows)
        if any(left >= right for left, right in zip(raw_values, raw_values[1:])):
            raise ValueError("raw uncertainty knots must be strictly increasing")
        if any(left > right for left, right in zip(probabilities, probabilities[1:])):
            raise ValueError("error probabilities must be monotone non-decreasing")
        if type(self.observation_count) is not int or self.observation_count < 1:
            raise ValueError("observation_count must be a positive integer")
        if type(self.task_cluster_count) is not int or self.task_cluster_count < 2:
            raise ValueError("task_cluster_count must be at least two")
        if sum(row.observation_count for row in self.rows) != self.observation_count:
            raise ValueError("row observation counts do not match observation_count")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty string")

    def transform(self, raw_uncertainty: float) -> float:
        """Return calibrated P(error) using only controller-visible uncertainty."""

        raw = _finite_probability(raw_uncertainty, label="raw_uncertainty")
        knots = tuple(row.raw_uncertainty for row in self.rows)
        index = bisect_left(knots, raw)
        if index == 0:
            return self.rows[0].error_probability
        if index == len(self.rows):
            return self.rows[-1].error_probability
        right = self.rows[index]
        if right.raw_uncertainty == raw:
            return right.error_probability
        left = self.rows[index - 1]
        weight = (raw - left.raw_uncertainty) / (right.raw_uncertainty - left.raw_uncertainty)
        return left.error_probability + weight * (right.error_probability - left.error_probability)

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact with exact experiment identity."""

        return {
            "schema": _CALIBRATOR_SCHEMA,
            "identity": _identity_to_mapping(self.identity),
            "method": self.method,
            "observation_count": self.observation_count,
            "task_cluster_count": self.task_cluster_count,
            "rows": [row.to_mapping() for row in self.rows],
        }

    @classmethod
    def from_mapping(cls, value: Any) -> FrozenUncertaintyCalibrator:
        fields = {
            "schema",
            "identity",
            "method",
            "observation_count",
            "task_cluster_count",
            "rows",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("calibrator fields do not match the schema")
        if value["schema"] != _CALIBRATOR_SCHEMA:
            raise ValueError("unsupported calibrator schema")
        rows = value["rows"]
        if not isinstance(rows, list):
            raise ValueError("calibrator rows must be a list")
        return cls(
            identity=_identity_from_mapping(value["identity"]),
            rows=tuple(IsotonicCalibrationRow.from_mapping(row) for row in rows),
            observation_count=value["observation_count"],
            task_cluster_count=value["task_cluster_count"],
            method=value["method"],
        )


@dataclass
class _PavaBlock:
    start: int
    stop: int
    error_count: int
    observation_count: int

    @property
    def mean(self) -> float:
        return self.error_count / self.observation_count


def fit_isotonic_uncertainty(
    identity: CalibrationIdentity,
    observations: Iterable[UncertaintyCalibrationObservation],
    *,
    min_task_clusters: int = 8,
) -> FrozenUncertaintyCalibrator:
    """Fit a deterministic monotone uncertainty transform offline.

    PAVA is weighted by the number of observations at each unique raw score.
    Task clusters are used as the minimum independent-data gate, not as
    pseudo-replicates.  Hidden ``is_error`` labels are discarded after this
    function returns.
    """

    if not isinstance(identity, CalibrationIdentity):
        raise ValueError("identity must be a CalibrationIdentity")
    if type(min_task_clusters) is not int or min_task_clusters < 2:
        raise ValueError("min_task_clusters must be an integer of at least two")
    materialized = tuple(observations)
    if not materialized:
        raise InsufficientCalibrationDataError("uncertainty calibration observations are empty")
    if any(not isinstance(row, UncertaintyCalibrationObservation) for row in materialized):
        raise ValueError("observations must contain UncertaintyCalibrationObservation values")
    cluster_count = len({row.task_cluster_id for row in materialized})
    if cluster_count != len(materialized):
        raise ValueError(
            "uncertainty calibration accepts exactly one observation per task cluster"
        )
    if cluster_count < min_task_clusters:
        raise InsufficientCalibrationDataError(
            f"uncertainty calibration requires {min_task_clusters} task clusters; got {cluster_count}"
        )

    grouped: dict[float, list[UncertaintyCalibrationObservation]] = {}
    for observation in materialized:
        grouped.setdefault(observation.raw_uncertainty, []).append(observation)
    raw_values = sorted(grouped)
    blocks: list[_PavaBlock] = []
    for index, raw in enumerate(raw_values):
        group = grouped[raw]
        blocks.append(
            _PavaBlock(
                start=index,
                stop=index + 1,
                error_count=sum(observation.is_error for observation in group),
                observation_count=len(group),
            )
        )
        while len(blocks) >= 2 and blocks[-2].mean > blocks[-1].mean:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                _PavaBlock(
                    start=left.start,
                    stop=right.stop,
                    error_count=left.error_count + right.error_count,
                    observation_count=left.observation_count + right.observation_count,
                )
            )

    fitted = [0.0] * len(raw_values)
    for block in blocks:
        fitted[block.start : block.stop] = [block.mean] * (block.stop - block.start)
    rows = tuple(
        IsotonicCalibrationRow(
            raw_uncertainty=raw,
            error_probability=fitted[index],
            observation_count=len(grouped[raw]),
            task_cluster_count=len({row.task_cluster_id for row in grouped[raw]}),
        )
        for index, raw in enumerate(raw_values)
    )
    return FrozenUncertaintyCalibrator(
        identity=identity,
        rows=rows,
        observation_count=len(materialized),
        task_cluster_count=cluster_count,
    )


@dataclass(frozen=True)
class TransitionCalibrationObservation:
    """Offline-only clustered outcome and cost for one attempted transition."""

    task_cluster_id: str
    context_bucket: str
    trigger: Trigger
    depth: int
    action: ControllerAction
    rescued: bool
    extra_output_tokens: int
    direct_e2e_seconds: float
    extra_e2e_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.task_cluster_id, str) or not self.task_cluster_id.strip():
            raise ValueError("task_cluster_id must be a non-empty string")
        if not isinstance(self.context_bucket, str) or not self.context_bucket.strip():
            raise ValueError("context_bucket must be a non-empty string")
        if not isinstance(self.trigger, Trigger):
            raise ValueError("trigger must be a Trigger")
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError("depth must be a positive integer")
        if not isinstance(self.action, ControllerAction) or self.action not in EXTRA_ACTIONS:
            raise ValueError("action must be an extra-generation action")
        if type(self.rescued) is not bool:
            raise ValueError("rescued must be a bool")
        if type(self.extra_output_tokens) is not int or self.extra_output_tokens < 1:
            raise ValueError("extra_output_tokens must be a positive integer")
        object.__setattr__(
            self,
            "direct_e2e_seconds",
            _positive_number(self.direct_e2e_seconds, label="direct_e2e_seconds"),
        )
        object.__setattr__(
            self,
            "extra_e2e_seconds",
            _positive_number(self.extra_e2e_seconds, label="extra_e2e_seconds"),
        )

    @property
    def extra_e2e_latency_ratio(self) -> float:
        return self.extra_e2e_seconds / self.direct_e2e_seconds


_TransitionKey = tuple[str, Trigger, int, ControllerAction]


def _transition_key(observation: TransitionCalibrationObservation) -> _TransitionKey:
    return (
        observation.context_bucket,
        observation.trigger,
        observation.depth,
        observation.action,
    )


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _group_seed(seed: int, key: _TransitionKey) -> int:
    payload = (f"{seed}\0{key[0]}\0{key[1].value}\0{key[2]}\0{key[3].value}").encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


def _bootstrap_transition(
    key: _TransitionKey,
    observations: tuple[TransitionCalibrationObservation, ...],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> TransitionEstimate:
    rng = random.Random(seed)
    cluster_count = len(observations)
    rescue_samples: list[float] = []
    token_samples: list[float] = []
    latency_samples: list[float] = []
    for _ in range(resamples):
        sample = tuple(observations[rng.randrange(cluster_count)] for _ in observations)
        rescue_samples.append(sum(row.rescued for row in sample) / cluster_count)
        token_samples.append(sum(row.extra_output_tokens for row in sample) / cluster_count)
        latency_samples.append(sum(row.extra_e2e_latency_ratio for row in sample) / cluster_count)

    rescue_point = sum(row.rescued for row in observations) / cluster_count
    token_point = sum(row.extra_output_tokens for row in observations) / cluster_count
    latency_point = sum(row.extra_e2e_latency_ratio for row in observations) / cluster_count
    lower_tail = 1.0 - confidence_level
    rescue_lcb = min(rescue_point, _percentile(rescue_samples, lower_tail))
    token_ucb = max(token_point, _percentile(token_samples, confidence_level))
    latency_ucb = max(latency_point, _percentile(latency_samples, confidence_level))
    context_bucket, trigger, depth, action = key
    return TransitionEstimate(
        context_bucket=context_bucket,
        trigger=trigger,
        depth=depth,
        action=action,
        conservative_success_lcb=rescue_lcb,
        extra_output_tokens_ucb=token_ucb,
        extra_e2e_latency_ratio_ucb=latency_ucb,
        bootstrap=BootstrapMetadata(
            task_cluster_count=cluster_count,
            resamples=resamples,
            confidence_level=confidence_level,
            method=_BOOTSTRAP_METHOD,
            seed=seed,
        ),
    )


def build_frozen_transition_model(
    identity: CalibrationIdentity,
    observations: Iterable[TransitionCalibrationObservation],
    *,
    min_task_clusters: int = 8,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260718,
) -> FrozenTransitionModel:
    """Build immutable transition bounds from independent task clusters.

    Each task cluster may contribute at most one observation to a transition
    stratum.  Every emitted stratum must independently satisfy the cluster
    gate; the whole build fails if any requested stratum is underpowered, so a
    partially calibrated table cannot be published by accident.
    """

    if not isinstance(identity, CalibrationIdentity):
        raise ValueError("identity must be a CalibrationIdentity")
    if type(min_task_clusters) is not int or min_task_clusters < 2:
        raise ValueError("min_task_clusters must be an integer of at least two")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    confidence = _finite_probability(confidence_level, label="confidence_level")
    if confidence <= 0.5 or confidence >= 1.0:
        raise ValueError("confidence_level must be in (0.5, 1)")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    materialized = tuple(observations)
    if not materialized:
        raise InsufficientCalibrationDataError("transition calibration observations are empty")
    if any(not isinstance(row, TransitionCalibrationObservation) for row in materialized):
        raise ValueError("observations must contain TransitionCalibrationObservation values")

    grouped: dict[_TransitionKey, dict[str, TransitionCalibrationObservation]] = {}
    for observation in materialized:
        key = _transition_key(observation)
        clusters = grouped.setdefault(key, {})
        if observation.task_cluster_id in clusters:
            raise ValueError(f"duplicate task cluster in transition stratum: {observation.task_cluster_id}/{key}")
        clusters[observation.task_cluster_id] = observation

    insufficient = [(key, len(clusters)) for key, clusters in grouped.items() if len(clusters) < min_task_clusters]
    if insufficient:
        details = ", ".join(
            f"{key[0]}/{key[1].value}/d{key[2]}/{key[3].value}={count}"
            for key, count in sorted(
                insufficient,
                key=lambda item: (
                    item[0][0],
                    item[0][1].value,
                    item[0][2],
                    item[0][3].value,
                ),
            )
        )
        raise InsufficientCalibrationDataError(
            f"every transition stratum requires {min_task_clusters} task clusters; got {details}"
        )

    estimates: list[TransitionEstimate] = []
    for key in sorted(
        grouped,
        key=lambda item: (item[0], item[1].value, item[2], item[3].value),
    ):
        rows = tuple(grouped[key][cluster] for cluster in sorted(grouped[key]))
        derived_seed = _group_seed(seed, key)
        estimates.append(
            _bootstrap_transition(
                key,
                rows,
                resamples=resamples,
                confidence_level=confidence,
                seed=derived_seed,
            )
        )
    return FrozenTransitionModel(identity, estimates)


def frozen_transition_model_to_mapping(
    model: FrozenTransitionModel,
) -> dict[str, Any]:
    """Serialize a frozen transition model without private-state access."""

    if not isinstance(model, FrozenTransitionModel):
        raise ValueError("model must be a FrozenTransitionModel")
    rows: list[dict[str, Any]] = []
    for estimate in model.estimates:
        rows.append(
            {
                "context_bucket": estimate.context_bucket,
                "trigger": estimate.trigger.value,
                "depth": estimate.depth,
                "action": estimate.action.value,
                "conservative_success_lcb": estimate.conservative_success_lcb,
                "extra_output_tokens_ucb": estimate.extra_output_tokens_ucb,
                "extra_e2e_latency_ratio_ucb": estimate.extra_e2e_latency_ratio_ucb,
                "bootstrap": {
                    "task_cluster_count": estimate.bootstrap.task_cluster_count,
                    "resamples": estimate.bootstrap.resamples,
                    "confidence_level": estimate.bootstrap.confidence_level,
                    "method": estimate.bootstrap.method,
                    "seed": estimate.bootstrap.seed,
                },
            }
        )
    return {
        "schema": _TRANSITION_MODEL_SCHEMA,
        "identity": _identity_to_mapping(model.identity),
        "estimates": rows,
    }


def frozen_transition_model_from_mapping(value: Any) -> FrozenTransitionModel:
    """Restore a transition table while re-running every dataclass invariant."""

    fields = {"schema", "identity", "estimates"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("transition model fields do not match the schema")
    if value["schema"] != _TRANSITION_MODEL_SCHEMA:
        raise ValueError("unsupported transition model schema")
    rows = value["estimates"]
    if not isinstance(rows, list):
        raise ValueError("transition estimates must be a list")
    estimates: list[TransitionEstimate] = []
    estimate_fields = {
        "context_bucket",
        "trigger",
        "depth",
        "action",
        "conservative_success_lcb",
        "extra_output_tokens_ucb",
        "extra_e2e_latency_ratio_ucb",
        "bootstrap",
    }
    bootstrap_fields = {
        "task_cluster_count",
        "resamples",
        "confidence_level",
        "method",
        "seed",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != estimate_fields:
            raise ValueError("transition estimate fields do not match the schema")
        bootstrap = row["bootstrap"]
        if not isinstance(bootstrap, Mapping) or set(bootstrap) != bootstrap_fields:
            raise ValueError("bootstrap fields do not match the schema")
        estimates.append(
            TransitionEstimate(
                context_bucket=row["context_bucket"],
                trigger=Trigger(row["trigger"]),
                depth=row["depth"],
                action=ControllerAction(row["action"]),
                conservative_success_lcb=row["conservative_success_lcb"],
                extra_output_tokens_ucb=row["extra_output_tokens_ucb"],
                extra_e2e_latency_ratio_ucb=row["extra_e2e_latency_ratio_ucb"],
                bootstrap=BootstrapMetadata(
                    **{field: bootstrap[field] for field in bootstrap_fields}
                ),
            )
        )
    return FrozenTransitionModel(_identity_from_mapping(value["identity"]), estimates)
