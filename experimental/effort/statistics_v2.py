"""Fail-closed paired statistics for adaptive-effort experiments.

The task, rather than a repeated generation, is the resampling unit.  Every
task must have exactly one row for the baseline and one for the candidate
strategy.  This keeps paired comparisons explicit and prevents repetitions of
the same task from being mistaken for independent evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import random
import re
from typing import Any, Iterable, Mapping


_ROW_FIELDS = frozenset(
    {
        "task_id",
        "strategy",
        "correct",
        "e2e_seconds",
        "fast_path",
        "deadline_violations",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "git_revision",
        "git_dirty",
        "model_revision",
        "policy_sha256",
        "task_manifest_sha256",
        "scorer_sha256",
        "verifier_sha256",
        "preregistration_sha256",
        "test_split_id",
        "leakage_detected",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{7,64}")


@dataclass(frozen=True)
class EffortStatisticsRow:
    """One strategy result for one independent task."""

    task_id: str
    strategy: str
    correct: bool
    e2e_seconds: float
    fast_path: bool
    deadline_violations: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.strategy, str) or not self.strategy.strip():
            raise ValueError("strategy must be a non-empty string")
        if type(self.correct) is not bool:
            raise ValueError("correct must be a bool")
        if type(self.fast_path) is not bool:
            raise ValueError("fast_path must be a bool")
        if not isinstance(self.e2e_seconds, (int, float)) or isinstance(
            self.e2e_seconds, bool
        ):
            raise ValueError("e2e_seconds must be numeric")
        if not math.isfinite(float(self.e2e_seconds)) or self.e2e_seconds <= 0.0:
            raise ValueError("e2e_seconds must be finite and positive")
        if type(self.deadline_violations) is not int or self.deadline_violations < 0:
            raise ValueError("deadline_violations must be a non-negative integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EffortStatisticsRow:
        missing = _ROW_FIELDS.difference(value)
        if missing:
            raise ValueError(f"row is missing required fields: {', '.join(sorted(missing))}")
        return cls(**{field: value[field] for field in _ROW_FIELDS})


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.point, self.lower, self.upper)):
            raise ValueError("confidence interval values must be finite")
        if self.lower > self.upper:
            raise ValueError("confidence interval lower bound exceeds upper bound")


@dataclass(frozen=True)
class DistributionSummary:
    p50: float
    p95: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.p50) or not math.isfinite(self.p95):
            raise ValueError("distribution summary values must be finite")
        if self.p50 > self.p95:
            raise ValueError("p50 must not exceed p95")


@dataclass(frozen=True)
class McNemarResult:
    baseline_only_correct: int
    candidate_only_correct: int
    discordant_pairs: int
    p_value: float


@dataclass(frozen=True)
class PairedEffortStatistics:
    baseline_strategy: str
    candidate_strategy: str
    tasks: int
    baseline_accuracy: float
    candidate_accuracy: float
    accuracy_delta: ConfidenceInterval
    e2e_latency_ratio: ConfidenceInterval
    correct_completions_per_second_ratio: ConfidenceInterval
    baseline_latency_seconds: DistributionSummary
    candidate_latency_seconds: DistributionSummary
    paired_latency_ratio: DistributionSummary
    fast_path_tasks: int
    fast_path_overhead_ratio: float | None
    candidate_deadline_violations: int
    mcnemar: McNemarResult
    bootstrap_samples: int
    bootstrap_seed: int
    zero_baseline_correct_bootstrap_samples: int


@dataclass(frozen=True)
class RunProvenance:
    """Minimum immutable provenance required by a confirmatory quality gate."""

    git_revision: str
    git_dirty: bool
    model_revision: str
    policy_sha256: str
    task_manifest_sha256: str
    scorer_sha256: str
    verifier_sha256: str
    preregistration_sha256: str
    test_split_id: str
    leakage_detected: bool

    def __post_init__(self) -> None:
        if not isinstance(self.git_revision, str) or not _GIT_REVISION_RE.fullmatch(
            self.git_revision.casefold()
        ):
            raise ValueError("git_revision must be a 7-64 character hexadecimal revision")
        if type(self.git_dirty) is not bool:
            raise ValueError("git_dirty must be a bool")
        if type(self.leakage_detected) is not bool:
            raise ValueError("leakage_detected must be a bool")
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("model_revision must be present")
        if not isinstance(self.test_split_id, str) or not self.test_split_id.strip():
            raise ValueError("test_split_id must be present")
        for field in (
            "policy_sha256",
            "task_manifest_sha256",
            "scorer_sha256",
            "verifier_sha256",
            "preregistration_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.casefold()):
                raise ValueError(f"{field} must be a complete SHA-256 digest")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RunProvenance:
        missing = _PROVENANCE_FIELDS.difference(value)
        if missing:
            raise ValueError(
                f"provenance is missing required fields: {', '.join(sorted(missing))}"
            )
        return cls(**{field: value[field] for field in _PROVENANCE_FIELDS})


@dataclass(frozen=True)
class PreregisteredGatePolicy:
    min_tasks: int = 100
    alpha: float = 0.05
    planned_comparisons: int = 3
    min_accuracy_delta: float = 0.05
    min_correct_completions_per_second_ratio: float = 0.95
    max_fast_path_overhead_ratio: float = 0.02
    max_e2e_latency_ratio: float = 1.15

    def __post_init__(self) -> None:
        if self.min_tasks < 1:
            raise ValueError("min_tasks must be positive")
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be finite and in (0, 1)")
        if type(self.planned_comparisons) is not int or self.planned_comparisons < 1:
            raise ValueError("planned_comparisons must be a positive integer")
        if not math.isfinite(self.min_accuracy_delta) or not 0.0 <= self.min_accuracy_delta <= 1.0:
            raise ValueError("min_accuracy_delta must be finite and in [0, 1]")
        if (
            not math.isfinite(self.min_correct_completions_per_second_ratio)
            or self.min_correct_completions_per_second_ratio < 0.0
        ):
            raise ValueError(
                "min_correct_completions_per_second_ratio must be finite and non-negative"
            )
        if (
            not math.isfinite(self.max_fast_path_overhead_ratio)
            or self.max_fast_path_overhead_ratio < 0.0
        ):
            raise ValueError("max_fast_path_overhead_ratio must be finite and non-negative")
        if not math.isfinite(self.max_e2e_latency_ratio) or self.max_e2e_latency_ratio < 1.0:
            raise ValueError("max_e2e_latency_ratio must be finite and at least one")

    @property
    def corrected_alpha(self) -> float:
        """Bonferroni-corrected alpha for the planned comparisons."""

        return self.alpha / self.planned_comparisons


class GateFailure(StrEnum):
    INSUFFICIENT_TASKS = "insufficient_tasks"
    MISSING_PROVENANCE = "missing_provenance"
    GIT_DIRTY = "git_dirty"
    LEAKAGE_DETECTED = "leakage_detected"
    ACCURACY_DELTA_TOO_SMALL = "accuracy_delta_too_small"
    QUALITY_CI_NOT_POSITIVE = "quality_ci_not_positive"
    MCNEMAR_NOT_SIGNIFICANT = "mcnemar_not_significant"
    CORRECT_COMPLETIONS_RATE_TOO_LOW = "correct_completions_rate_too_low"
    FAST_PATH_METRIC_MISSING = "fast_path_metric_missing"
    FAST_PATH_OVERHEAD = "fast_path_overhead"
    E2E_LATENCY = "e2e_latency"
    DEADLINE_VIOLATIONS = "deadline_violations"


@dataclass(frozen=True)
class PreregisteredGateResult:
    passed: bool
    failures: tuple[GateFailure, ...]
    corrected_alpha: float


def exact_mcnemar(
    baseline_correct: Iterable[bool],
    candidate_correct: Iterable[bool],
) -> McNemarResult:
    """Return the exact two-sided McNemar test for paired binary outcomes."""

    baseline = tuple(baseline_correct)
    candidate = tuple(candidate_correct)
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("McNemar inputs must be non-empty and have equal length")
    if any(type(value) is not bool for value in (*baseline, *candidate)):
        raise ValueError("McNemar outcomes must be bool values")
    baseline_only = sum(base and not contender for base, contender in zip(baseline, candidate))
    candidate_only = sum(not base and contender for base, contender in zip(baseline, candidate))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(baseline_only, candidate_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return McNemarResult(
        baseline_only_correct=baseline_only,
        candidate_only_correct=candidate_only,
        discordant_pairs=discordant,
        p_value=p_value,
    )


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _summary(values: Iterable[float]) -> DistributionSummary:
    materialized = tuple(values)
    return DistributionSummary(
        p50=_percentile(materialized, 0.50),
        p95=_percentile(materialized, 0.95),
    )


def _ratio(numerator: float, denominator: float, *, label: str) -> float:
    if denominator <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    value = numerator / denominator
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _metrics_for_pairs(
    pairs: tuple[tuple[EffortStatisticsRow, EffortStatisticsRow], ...],
    *,
    zero_baseline_rate_ratio: float | None = None,
) -> tuple[float, float, float, bool]:
    count = len(pairs)
    accuracy_delta = sum(int(candidate.correct) - int(baseline.correct) for baseline, candidate in pairs) / count
    baseline_seconds = sum(baseline.e2e_seconds for baseline, _ in pairs)
    candidate_seconds = sum(candidate.e2e_seconds for _, candidate in pairs)
    latency_ratio = _ratio(candidate_seconds, baseline_seconds, label="latency ratio")
    baseline_correct = sum(baseline.correct for baseline, _ in pairs)
    candidate_correct = sum(candidate.correct for _, candidate in pairs)
    if baseline_correct == 0:
        if zero_baseline_rate_ratio is None:
            raise ValueError(
                "correct-completions rate ratio requires at least one correct baseline task"
            )
        return accuracy_delta, latency_ratio, zero_baseline_rate_ratio, True
    baseline_rate = _ratio(baseline_correct, baseline_seconds, label="baseline correct-completions rate")
    candidate_rate = _ratio(candidate_correct, candidate_seconds, label="candidate correct-completions rate")
    rate_ratio = _ratio(candidate_rate, baseline_rate, label="correct-completions rate ratio")
    return accuracy_delta, latency_ratio, rate_ratio, False


def _coerce_row(value: EffortStatisticsRow | Mapping[str, Any]) -> EffortStatisticsRow:
    if isinstance(value, EffortStatisticsRow):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("rows must be EffortStatisticsRow objects or mappings")
    return EffortStatisticsRow.from_mapping(value)


def analyze_paired_rows(
    rows: Iterable[EffortStatisticsRow | Mapping[str, Any]],
    *,
    baseline_strategy: str,
    candidate_strategy: str,
    bootstrap_samples: int = 10_000,
    seed: int = 20260716,
    confidence: float = 0.95,
) -> PairedEffortStatistics:
    """Analyze paired task rows with a seeded task-cluster bootstrap."""

    if not isinstance(baseline_strategy, str) or not baseline_strategy.strip():
        raise ValueError("baseline_strategy must be present")
    if not isinstance(candidate_strategy, str) or not candidate_strategy.strip():
        raise ValueError("candidate_strategy must be present")
    if baseline_strategy == candidate_strategy:
        raise ValueError("baseline and candidate strategies must differ")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")

    materialized = tuple(_coerce_row(row) for row in rows)
    if not materialized:
        raise ValueError("rows must not be empty")
    permitted = {baseline_strategy, candidate_strategy}
    unexpected = sorted({row.strategy for row in materialized}.difference(permitted))
    if unexpected:
        raise ValueError(f"unexpected strategies: {', '.join(unexpected)}")

    indexed: dict[tuple[str, str], EffortStatisticsRow] = {}
    task_ids: set[str] = set()
    for row in materialized:
        key = (row.task_id, row.strategy)
        if key in indexed:
            raise ValueError(f"duplicate task/strategy row: {row.task_id}/{row.strategy}")
        indexed[key] = row
        task_ids.add(row.task_id)

    pairs: list[tuple[EffortStatisticsRow, EffortStatisticsRow]] = []
    for task_id in sorted(task_ids):
        baseline = indexed.get((task_id, baseline_strategy))
        candidate = indexed.get((task_id, candidate_strategy))
        if baseline is None or candidate is None:
            raise ValueError(f"task {task_id!r} does not have a complete strategy pair")
        pairs.append((baseline, candidate))
    paired = tuple(pairs)
    if len(materialized) != 2 * len(paired):
        raise ValueError("every task must have exactly one row per strategy")

    point_accuracy, point_latency, point_rate, _ = _metrics_for_pairs(paired)
    rng = random.Random(seed)
    accuracy_samples: list[float] = []
    latency_samples: list[float] = []
    rate_samples: list[float] = []
    zero_baseline_correct_samples = 0
    for _ in range(bootstrap_samples):
        sample = tuple(paired[rng.randrange(len(paired))] for _ in paired)
        # A resample with no correct baseline completions has an undefined
        # rate ratio. Infinity would make its interval optimistic, so the
        # pre-registered fail-closed contribution is zero.
        accuracy, latency, rate, zero_baseline = _metrics_for_pairs(
            sample,
            zero_baseline_rate_ratio=0.0,
        )
        zero_baseline_correct_samples += int(zero_baseline)
        accuracy_samples.append(accuracy)
        latency_samples.append(latency)
        rate_samples.append(rate)

    tail = (1.0 - confidence) / 2.0
    baseline_latency = tuple(baseline.e2e_seconds for baseline, _ in paired)
    candidate_latency = tuple(candidate.e2e_seconds for _, candidate in paired)
    per_task_latency_ratio = tuple(
        _ratio(candidate.e2e_seconds, baseline.e2e_seconds, label="paired task latency ratio")
        for baseline, candidate in paired
    )
    fast_path_pairs = tuple(pair for pair in paired if pair[1].fast_path)
    fast_path_overhead = None
    if fast_path_pairs:
        fast_path_baseline = sum(pair[0].e2e_seconds for pair in fast_path_pairs)
        fast_path_candidate = sum(pair[1].e2e_seconds for pair in fast_path_pairs)
        fast_path_overhead = _ratio(
            fast_path_candidate,
            fast_path_baseline,
            label="fast-path latency ratio",
        ) - 1.0

    baseline_outcomes = tuple(baseline.correct for baseline, _ in paired)
    candidate_outcomes = tuple(candidate.correct for _, candidate in paired)
    return PairedEffortStatistics(
        baseline_strategy=baseline_strategy,
        candidate_strategy=candidate_strategy,
        tasks=len(paired),
        baseline_accuracy=sum(baseline_outcomes) / len(paired),
        candidate_accuracy=sum(candidate_outcomes) / len(paired),
        accuracy_delta=ConfidenceInterval(
            point=point_accuracy,
            lower=_percentile(accuracy_samples, tail),
            upper=_percentile(accuracy_samples, 1.0 - tail),
        ),
        e2e_latency_ratio=ConfidenceInterval(
            point=point_latency,
            lower=_percentile(latency_samples, tail),
            upper=_percentile(latency_samples, 1.0 - tail),
        ),
        correct_completions_per_second_ratio=ConfidenceInterval(
            point=point_rate,
            lower=_percentile(rate_samples, tail),
            upper=_percentile(rate_samples, 1.0 - tail),
        ),
        baseline_latency_seconds=_summary(baseline_latency),
        candidate_latency_seconds=_summary(candidate_latency),
        paired_latency_ratio=_summary(per_task_latency_ratio),
        fast_path_tasks=len(fast_path_pairs),
        fast_path_overhead_ratio=fast_path_overhead,
        candidate_deadline_violations=sum(candidate.deadline_violations for _, candidate in paired),
        mcnemar=exact_mcnemar(baseline_outcomes, candidate_outcomes),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed,
        zero_baseline_correct_bootstrap_samples=zero_baseline_correct_samples,
    )


def _coerce_provenance(
    value: RunProvenance | Mapping[str, Any] | None,
) -> RunProvenance | None:
    if isinstance(value, RunProvenance):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        return RunProvenance.from_mapping(value)
    except (TypeError, ValueError):
        return None


def evaluate_preregistered_gate(
    statistics: PairedEffortStatistics,
    provenance: RunProvenance | Mapping[str, Any] | None,
    *,
    policy: PreregisteredGatePolicy | None = None,
) -> PreregisteredGateResult:
    """Evaluate the pre-registered quality/cost gate without silent defaults."""

    if not isinstance(statistics, PairedEffortStatistics):
        raise ValueError("statistics must be a PairedEffortStatistics result")
    selected_policy = policy or PreregisteredGatePolicy()
    resolved_provenance = _coerce_provenance(provenance)
    failures: list[GateFailure] = []
    if statistics.tasks < selected_policy.min_tasks:
        failures.append(GateFailure.INSUFFICIENT_TASKS)
    if resolved_provenance is None:
        failures.append(GateFailure.MISSING_PROVENANCE)
    else:
        if resolved_provenance.git_dirty:
            failures.append(GateFailure.GIT_DIRTY)
        if resolved_provenance.leakage_detected:
            failures.append(GateFailure.LEAKAGE_DETECTED)
    if statistics.accuracy_delta.point < selected_policy.min_accuracy_delta:
        failures.append(GateFailure.ACCURACY_DELTA_TOO_SMALL)
    if statistics.accuracy_delta.lower <= 0.0:
        failures.append(GateFailure.QUALITY_CI_NOT_POSITIVE)
    if statistics.mcnemar.p_value >= selected_policy.corrected_alpha:
        failures.append(GateFailure.MCNEMAR_NOT_SIGNIFICANT)
    if (
        statistics.correct_completions_per_second_ratio.lower
        < selected_policy.min_correct_completions_per_second_ratio
    ):
        failures.append(GateFailure.CORRECT_COMPLETIONS_RATE_TOO_LOW)
    if statistics.fast_path_overhead_ratio is None:
        failures.append(GateFailure.FAST_PATH_METRIC_MISSING)
    elif statistics.fast_path_overhead_ratio > selected_policy.max_fast_path_overhead_ratio:
        failures.append(GateFailure.FAST_PATH_OVERHEAD)
    if statistics.e2e_latency_ratio.point > selected_policy.max_e2e_latency_ratio:
        failures.append(GateFailure.E2E_LATENCY)
    if statistics.candidate_deadline_violations:
        failures.append(GateFailure.DEADLINE_VIOLATIONS)
    return PreregisteredGateResult(
        passed=not failures,
        failures=tuple(failures),
        corrected_alpha=selected_policy.corrected_alpha,
    )


__all__ = [
    "ConfidenceInterval",
    "DistributionSummary",
    "EffortStatisticsRow",
    "GateFailure",
    "McNemarResult",
    "PairedEffortStatistics",
    "PreregisteredGatePolicy",
    "PreregisteredGateResult",
    "RunProvenance",
    "analyze_paired_rows",
    "evaluate_preregistered_gate",
    "exact_mcnemar",
]
