#!/usr/bin/env python3
"""MioCodeBench v1: paired, leakage-resistant coding-quality benchmark harness.

The harness deliberately separates generation from hidden evaluation.  It does
not know how Mio produces a patch and it does not know how a private evaluator
scores one; callers inject both operations.  This keeps the protocol testable
without importing the inference stack and makes it possible to freeze a suite
before any model run.

Only :class:`SourceFreeAggregate` is intended for publication.  Per-fixture
records remain in memory for local diagnostics and are never serialized by the
public serializer in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol, Sequence


SCHEMA_VERSION = "miocodebench-v1"
GATE_OFF = "gate_off"
GATE_ON = "gate_on"
CONDITIONS = (GATE_OFF, GATE_ON)
_SAFE_FIXTURE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")
_CONDITION_METRIC_KEYS = {
    "run_count",
    "passed_count",
    "passed_rate",
    "regression_free_count",
    "regression_free_rate",
    "generation_completed_count",
    "generation_completed_rate",
    "validation_succeeded_count",
    "validation_succeeded_rate",
    "mean_mutation_count",
    "mean_tool_calls",
    "mean_output_tokens",
    "mean_model_seconds",
    "mean_wall_seconds",
}
_CONTINGENCY_KEYS = {"both_pass", "gate_off_only", "gate_on_only", "neither_pass"}
_PRIMARY_STATISTIC_KEYS = {
    "gate_on_minus_gate_off",
    "bootstrap_interval_low",
    "bootstrap_interval_high",
    "exact_discordant_p",
}
_INTEGRITY_KEYS = {
    "suite_digest_matched",
    "generation_precedes_evaluation",
    "pair_complete",
    "run_count_complete",
    "all_generation_completed",
    "expected_run_count",
    "observed_run_count",
    "eligible",
}
_CLAIM_KEYS = {
    "eligible",
    "status",
    "minimum_pairs_met",
    "positive_direction",
    "bootstrap_interval_positive",
    "discordant_p_within_alpha",
}
_CLAIM_STATUSES = {
    "quality_improvement_supported",
    "no_claim_integrity_gate",
    "no_claim_insufficient_pairs",
    "no_claim_nonpositive_delta",
    "no_claim_interval_crosses_zero",
    "no_claim_discordant_p_above_alpha",
}


class BenchmarkProtocolError(ValueError):
    """Raised before generation when the frozen protocol is not satisfied."""


@dataclass(frozen=True)
class PublicFile:
    """One source file visible to both benchmark conditions."""

    relative_name: str
    content: str

    def __post_init__(self) -> None:
        _validate_relative_name(self.relative_name)
        if not isinstance(self.content, str):
            raise TypeError("public file content must be text")


@dataclass(frozen=True)
class CodingFixture:
    """Frozen public inputs for one paired coding task.

    Private checks intentionally are not a field of this class.  The hidden
    evaluator receives only ``fixture_id`` and resolves its oracle separately.
    """

    fixture_id: str
    instruction: str
    public_files: tuple[PublicFile, ...]

    def __post_init__(self) -> None:
        if not _SAFE_FIXTURE_ID.fullmatch(self.fixture_id):
            raise ValueError("fixture_id must be a short lowercase stable identifier")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("fixture instruction cannot be empty")
        if not isinstance(self.public_files, tuple) or any(
            not isinstance(item, PublicFile) for item in self.public_files
        ):
            raise TypeError("public_files must be a tuple of PublicFile values")
        names = [item.relative_name for item in self.public_files]
        if not names:
            raise ValueError("a fixture must expose at least one public file")
        if len(names) != len(set(names)):
            raise ValueError("public file names must be unique within a fixture")


@dataclass(frozen=True)
class MaterializedFixture:
    """Public fixture copied into an isolated condition workspace."""

    fixture_id: str
    instruction: str
    workspace: Path


@dataclass(frozen=True)
class ScheduledRun:
    fixture: CodingFixture
    condition: str
    schedule_index: int


@dataclass(frozen=True)
class GenerationRequest:
    """Input passed to the injected model/agent runner."""

    fixture_id: str
    instruction: str
    condition: str
    workspace: Path
    schedule_index: int


@dataclass(frozen=True)
class GenerationObservation:
    """Content-free measurements returned by the injected runner."""

    completed: bool
    mutation_count: int = 0
    tool_calls: int = 0
    output_tokens: int = 0
    validation_attempted: bool = False
    validation_succeeded: bool = False
    model_seconds: float = 0.0
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("completed", "validation_attempted", "validation_succeeded"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        for field_name in ("mutation_count", "tool_calls", "output_tokens"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in ("model_seconds", "wall_seconds"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a finite non-negative number")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{field_name} must be a finite non-negative number")
        if self.validation_succeeded and not self.validation_attempted:
            raise ValueError("a successful validation requires a validation attempt")


@dataclass(frozen=True)
class EvaluationRequest:
    """Workspace handed to the private evaluator after generation is sealed."""

    fixture_id: str
    condition: str
    workspace: Path
    schedule_index: int


@dataclass(frozen=True)
class HiddenEvaluation:
    """Binary private outcomes retained only for paired aggregation."""

    passed: bool
    regression_free: bool

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool) or not isinstance(self.regression_free, bool):
            raise ValueError("hidden evaluation outcomes must be boolean")


class GenerationRunner(Protocol):
    def __call__(self, request: GenerationRequest) -> GenerationObservation:
        """Run exactly one condition in the supplied isolated workspace."""


class HiddenEvaluator(Protocol):
    def __call__(self, request: EvaluationRequest) -> HiddenEvaluation:
        """Evaluate a sealed workspace without exposing private checks."""


@dataclass(frozen=True)
class Preregistration:
    """Frozen decision rules checked before the first generation call."""

    expected_suite_sha256: str
    seed: int = 20260718
    bootstrap_samples: int = 10_000
    alpha: float = 0.05
    minimum_pairs_for_claim: int = 16

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.expected_suite_sha256):
            raise ValueError("expected_suite_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.bootstrap_samples, bool)
            or not isinstance(self.bootstrap_samples, int)
            or self.bootstrap_samples < 1
        ):
            raise ValueError("bootstrap_samples must be at least one")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        if (
            isinstance(self.minimum_pairs_for_claim, bool)
            or not isinstance(self.minimum_pairs_for_claim, int)
            or self.minimum_pairs_for_claim < 1
        ):
            raise ValueError("minimum_pairs_for_claim must be at least one")


@dataclass(frozen=True)
class RunRecord:
    """Private in-memory record; do not publish or serialize."""

    fixture_id: str
    condition: str
    schedule_index: int
    workspace: Path
    generation: GenerationObservation
    evaluation: HiddenEvaluation


@dataclass(frozen=True)
class SourceFreeAggregate:
    """The only benchmark representation approved for public serialization."""

    suite_sha256: str
    seed: int
    pair_count: int
    bootstrap_samples: int
    alpha: float
    condition_metrics: Mapping[str, Mapping[str, int | float]]
    paired_contingency: Mapping[str, int]
    primary_statistics: Mapping[str, int | float]
    integrity_gate: Mapping[str, bool | int]
    claim_gate: Mapping[str, bool | str]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.suite_sha256):
            raise ValueError("suite_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for field_name in ("pair_count", "bootstrap_samples"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        if tuple(self.condition_metrics) != CONDITIONS:
            raise ValueError("public condition metrics must use the frozen condition order")
        for metrics in self.condition_metrics.values():
            _require_fixed_numeric_mapping("condition_metrics", metrics, _CONDITION_METRIC_KEYS)
        _require_fixed_integer_mapping("paired_contingency", self.paired_contingency, _CONTINGENCY_KEYS)
        _require_fixed_numeric_mapping("primary_statistics", self.primary_statistics, _PRIMARY_STATISTIC_KEYS)
        if set(self.integrity_gate) != _INTEGRITY_KEYS:
            raise ValueError("integrity_gate does not match the public schema")
        if set(self.claim_gate) != _CLAIM_KEYS:
            raise ValueError("claim_gate does not match the public schema")
        if self.claim_gate["status"] not in _CLAIM_STATUSES:
            raise ValueError("claim_gate status is not an approved public enum")
        _require_bool_keys(
            "integrity_gate",
            self.integrity_gate,
            _INTEGRITY_KEYS - {"expected_run_count", "observed_run_count"},
        )
        _require_bool_keys("claim_gate", self.claim_gate, _CLAIM_KEYS - {"status"})
        for key in ("expected_run_count", "observed_run_count"):
            value = self.integrity_gate[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"integrity_gate.{key} must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        """Return a fixed-schema tree containing no per-fixture material."""

        return {
            "schema_version": SCHEMA_VERSION,
            "suite_sha256": self.suite_sha256,
            "protocol": {
                "seed": self.seed,
                "pair_count": self.pair_count,
                "bootstrap_samples": self.bootstrap_samples,
                "alpha": self.alpha,
                "generation_precedes_evaluation": True,
            },
            "condition_metrics": {condition: dict(self.condition_metrics[condition]) for condition in CONDITIONS},
            "paired_contingency": dict(self.paired_contingency),
            "primary_statistics": dict(self.primary_statistics),
            "integrity_gate": dict(self.integrity_gate),
            "claim_gate": dict(self.claim_gate),
        }


@dataclass(frozen=True)
class BenchmarkExecution:
    """Completed execution with private records and a publishable aggregate."""

    records: tuple[RunRecord, ...]
    aggregate: SourceFreeAggregate


def _require_fixed_numeric_mapping(
    name: str,
    values: Mapping[str, object],
    expected_keys: set[str],
) -> None:
    if set(values) != expected_keys:
        raise ValueError(f"{name} does not match the public schema")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name}.{key} must be a finite number")


def _require_fixed_integer_mapping(
    name: str,
    values: Mapping[str, object],
    expected_keys: set[str],
) -> None:
    if set(values) != expected_keys:
        raise ValueError(f"{name} does not match the public schema")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}.{key} must be a non-negative integer")


def _require_bool_keys(name: str, values: Mapping[str, object], keys: set[str]) -> None:
    for key in keys:
        if not isinstance(values[key], bool):
            raise ValueError(f"{name}.{key} must be boolean")


def _validate_relative_name(value: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("public file name must be a non-empty POSIX relative name")
    raw_parts = value.split("/")
    name = PurePosixPath(value)
    if name.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("public file name must stay inside the fixture workspace")


def _canonical_suite_payload(fixtures: Sequence[CodingFixture]) -> bytes:
    ordered = sorted(fixtures, key=lambda fixture: fixture.fixture_id)
    payload = [
        {
            "fixture_id": fixture.fixture_id,
            "instruction": fixture.instruction,
            "public_files": [
                {"relative_name": item.relative_name, "content": item.content}
                for item in sorted(fixture.public_files, key=lambda item: item.relative_name)
            ],
        }
        for fixture in ordered
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def fixture_suite_sha256(fixtures: Sequence[CodingFixture]) -> str:
    """Hash all public benchmark inputs in a stable, order-independent form."""

    _validate_fixture_set(fixtures)
    return hashlib.sha256(_canonical_suite_payload(fixtures)).hexdigest()


def _validate_fixture_set(fixtures: Sequence[CodingFixture]) -> None:
    if not fixtures:
        raise BenchmarkProtocolError("the benchmark suite cannot be empty")
    identifiers = [fixture.fixture_id for fixture in fixtures]
    if len(identifiers) != len(set(identifiers)):
        raise BenchmarkProtocolError("fixture identifiers must be unique")


def materialize_public_fixture(fixture: CodingFixture, workspace: Path) -> MaterializedFixture:
    """Create an isolated workspace containing public fixture files only."""

    workspace = Path(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise BenchmarkProtocolError("fixture workspace must be empty")
    workspace.mkdir(parents=True, exist_ok=True)
    root = workspace.resolve()

    for item in sorted(fixture.public_files, key=lambda public_file: public_file.relative_name):
        target = workspace.joinpath(*PurePosixPath(item.relative_name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.parent.resolve().relative_to(root)
        except ValueError as exc:
            raise BenchmarkProtocolError("public file escaped the fixture workspace") from exc
        target.write_text(item.content, encoding="utf-8")

    return MaterializedFixture(
        fixture_id=fixture.fixture_id,
        instruction=fixture.instruction,
        workspace=workspace,
    )


def make_balanced_schedule(
    fixtures: Sequence[CodingFixture],
    *,
    seed: int,
) -> tuple[ScheduledRun, ...]:
    """Build a SHA-derived deterministic AB/BA schedule balanced within one."""

    _validate_fixture_set(fixtures)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    def seeded_rank(fixture: CodingFixture) -> tuple[bytes, str]:
        payload = f"{seed}\x00{fixture.fixture_id}".encode()
        return hashlib.sha256(payload).digest(), fixture.fixture_id

    ordered = sorted(fixtures, key=seeded_rank)

    schedule: list[ScheduledRun] = []
    schedule_index = 0
    for pair_index, fixture in enumerate(ordered):
        condition_order = CONDITIONS if pair_index % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in condition_order:
            schedule.append(
                ScheduledRun(
                    fixture=fixture,
                    condition=condition,
                    schedule_index=schedule_index,
                )
            )
            schedule_index += 1
    return tuple(schedule)


def exact_discordant_p(gate_off_only: int, gate_on_only: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant pairs."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (gate_off_only, gate_on_only)
    ):
        raise ValueError("discordant counts must be non-negative integers")
    discordant = gate_off_only + gate_on_only
    if discordant == 0:
        return 1.0
    smaller = min(gate_off_only, gate_on_only)
    tail_numerator = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail_numerator / (2**discordant))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def paired_bootstrap_delta(
    gate_off: Sequence[float | int | bool],
    gate_on: Sequence[float | int | bool],
    *,
    samples: int,
    seed: int,
    alpha: float,
) -> tuple[float, float, float]:
    """Observed paired delta and deterministic percentile bootstrap interval."""

    if len(gate_off) != len(gate_on) or not gate_off:
        raise ValueError("paired bootstrap inputs must have equal non-zero length")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be at least one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")

    deltas = [float(on) - float(off) for off, on in zip(gate_off, gate_on, strict=True)]
    if any(not math.isfinite(value) for value in deltas):
        raise ValueError("paired bootstrap values must be finite")
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    bootstrapped = [sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(samples)]
    return (
        observed,
        _percentile(bootstrapped, alpha / 2.0),
        _percentile(bootstrapped, 1.0 - alpha / 2.0),
    )


def _mean(values: Iterable[int | float | bool]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized)


def _condition_aggregate(records: Sequence[RunRecord]) -> Mapping[str, int | float]:
    count = len(records)
    passed = sum(record.evaluation.passed for record in records)
    regression_free = sum(record.evaluation.regression_free for record in records)
    completed = sum(record.generation.completed for record in records)
    validation_succeeded = sum(record.generation.validation_succeeded for record in records)
    return MappingProxyType(
        {
            "run_count": count,
            "passed_count": passed,
            "passed_rate": passed / count,
            "regression_free_count": regression_free,
            "regression_free_rate": regression_free / count,
            "generation_completed_count": completed,
            "generation_completed_rate": completed / count,
            "validation_succeeded_count": validation_succeeded,
            "validation_succeeded_rate": validation_succeeded / count,
            "mean_mutation_count": _mean(record.generation.mutation_count for record in records),
            "mean_tool_calls": _mean(record.generation.tool_calls for record in records),
            "mean_output_tokens": _mean(record.generation.output_tokens for record in records),
            "mean_model_seconds": _mean(record.generation.model_seconds for record in records),
            "mean_wall_seconds": _mean(record.generation.wall_seconds for record in records),
        }
    )


def _build_aggregate(
    records: Sequence[RunRecord],
    preregistration: Preregistration,
    suite_sha256: str,
) -> SourceFreeAggregate:
    by_fixture: dict[str, dict[str, RunRecord]] = {}
    by_condition: dict[str, list[RunRecord]] = {condition: [] for condition in CONDITIONS}
    for record in records:
        if record.condition not in by_condition:
            raise BenchmarkProtocolError("unexpected benchmark condition")
        if record.condition in by_fixture.setdefault(record.fixture_id, {}):
            raise BenchmarkProtocolError("duplicate fixture-condition result")
        by_fixture[record.fixture_id][record.condition] = record
        by_condition[record.condition].append(record)

    pair_complete = all(set(pair) == set(CONDITIONS) for pair in by_fixture.values())
    expected_run_count = len(by_fixture) * len(CONDITIONS)
    run_count_complete = len(records) == expected_run_count
    if not pair_complete or not run_count_complete:
        raise BenchmarkProtocolError("cannot aggregate incomplete paired records")

    gate_off_outcomes: list[bool] = []
    gate_on_outcomes: list[bool] = []
    both_pass = gate_off_only = gate_on_only = neither = 0
    for fixture_id in sorted(by_fixture):
        gate_off_passed = by_fixture[fixture_id][GATE_OFF].evaluation.passed
        gate_on_passed = by_fixture[fixture_id][GATE_ON].evaluation.passed
        gate_off_outcomes.append(gate_off_passed)
        gate_on_outcomes.append(gate_on_passed)
        if gate_off_passed and gate_on_passed:
            both_pass += 1
        elif gate_off_passed:
            gate_off_only += 1
        elif gate_on_passed:
            gate_on_only += 1
        else:
            neither += 1

    delta, interval_low, interval_high = paired_bootstrap_delta(
        gate_off_outcomes,
        gate_on_outcomes,
        samples=preregistration.bootstrap_samples,
        seed=preregistration.seed ^ 0xB00757,
        alpha=preregistration.alpha,
    )
    discordant_p = exact_discordant_p(gate_off_only, gate_on_only)
    enough_pairs = len(by_fixture) >= preregistration.minimum_pairs_for_claim
    direction_positive = delta > 0.0
    interval_positive = interval_low > 0.0
    discordant_significant = discordant_p < preregistration.alpha
    all_generation_completed = all(record.generation.completed for record in records)
    integrity_eligible = pair_complete and run_count_complete and all_generation_completed
    claim_eligible = (
        integrity_eligible and enough_pairs and direction_positive and interval_positive and discordant_significant
    )

    if claim_eligible:
        claim_status = "quality_improvement_supported"
    elif not integrity_eligible:
        claim_status = "no_claim_integrity_gate"
    elif not enough_pairs:
        claim_status = "no_claim_insufficient_pairs"
    elif not direction_positive:
        claim_status = "no_claim_nonpositive_delta"
    elif not interval_positive:
        claim_status = "no_claim_interval_crosses_zero"
    else:
        claim_status = "no_claim_discordant_p_above_alpha"

    condition_metrics = MappingProxyType(
        {condition: _condition_aggregate(by_condition[condition]) for condition in CONDITIONS}
    )
    return SourceFreeAggregate(
        suite_sha256=suite_sha256,
        seed=preregistration.seed,
        pair_count=len(by_fixture),
        bootstrap_samples=preregistration.bootstrap_samples,
        alpha=preregistration.alpha,
        condition_metrics=condition_metrics,
        paired_contingency=MappingProxyType(
            {
                "both_pass": both_pass,
                "gate_off_only": gate_off_only,
                "gate_on_only": gate_on_only,
                "neither_pass": neither,
            }
        ),
        primary_statistics=MappingProxyType(
            {
                "gate_on_minus_gate_off": delta,
                "bootstrap_interval_low": interval_low,
                "bootstrap_interval_high": interval_high,
                "exact_discordant_p": discordant_p,
            }
        ),
        integrity_gate=MappingProxyType(
            {
                "suite_digest_matched": True,
                "generation_precedes_evaluation": True,
                "pair_complete": pair_complete,
                "run_count_complete": run_count_complete,
                "all_generation_completed": all_generation_completed,
                "expected_run_count": expected_run_count,
                "observed_run_count": len(records),
                "eligible": integrity_eligible,
            }
        ),
        claim_gate=MappingProxyType(
            {
                "eligible": claim_eligible,
                "status": claim_status,
                "minimum_pairs_met": enough_pairs,
                "positive_direction": direction_positive,
                "bootstrap_interval_positive": interval_positive,
                "discordant_p_within_alpha": discordant_significant,
            }
        ),
    )


def run_benchmark(
    *,
    fixtures: Sequence[CodingFixture],
    preregistration: Preregistration,
    runner: GenerationRunner | Callable[[GenerationRequest], GenerationObservation],
    hidden_evaluator: HiddenEvaluator | Callable[[EvaluationRequest], HiddenEvaluation],
    work_root: Path,
) -> BenchmarkExecution:
    """Execute every generation before invoking the first hidden evaluation."""

    _validate_fixture_set(fixtures)
    suite_sha256 = fixture_suite_sha256(fixtures)
    if suite_sha256 != preregistration.expected_suite_sha256:
        raise BenchmarkProtocolError("fixture suite does not match the preregistered digest")

    work_root = Path(work_root)
    if work_root.exists() and any(work_root.iterdir()):
        raise BenchmarkProtocolError("benchmark work root must be empty")
    work_root.mkdir(parents=True, exist_ok=True)
    schedule = make_balanced_schedule(fixtures, seed=preregistration.seed)

    # Phase 1 is deliberately completed and sealed before phase 2 starts.
    generated: list[tuple[ScheduledRun, Path, GenerationObservation]] = []
    for item in schedule:
        workspace = work_root / f"run-{item.schedule_index:04d}"
        materialized = materialize_public_fixture(item.fixture, workspace)
        request = GenerationRequest(
            fixture_id=materialized.fixture_id,
            instruction=materialized.instruction,
            condition=item.condition,
            workspace=materialized.workspace,
            schedule_index=item.schedule_index,
        )
        observation = runner(request)
        if not isinstance(observation, GenerationObservation):
            raise TypeError("runner must return GenerationObservation")
        generated.append((item, workspace, observation))

    # Phase 2 may inspect mutated workspaces but cannot affect another run.
    records: list[RunRecord] = []
    for item, workspace, observation in generated:
        evaluation = hidden_evaluator(
            EvaluationRequest(
                fixture_id=item.fixture.fixture_id,
                condition=item.condition,
                workspace=workspace,
                schedule_index=item.schedule_index,
            )
        )
        if not isinstance(evaluation, HiddenEvaluation):
            raise TypeError("hidden_evaluator must return HiddenEvaluation")
        records.append(
            RunRecord(
                fixture_id=item.fixture.fixture_id,
                condition=item.condition,
                schedule_index=item.schedule_index,
                workspace=workspace,
                generation=observation,
                evaluation=evaluation,
            )
        )

    aggregate = _build_aggregate(records, preregistration, suite_sha256)
    return BenchmarkExecution(records=tuple(records), aggregate=aggregate)


def serialize_source_free_aggregate(aggregate: SourceFreeAggregate, *, indent: int | None = 2) -> str:
    """Serialize only the fixed aggregate schema; private records are rejected."""

    if not isinstance(aggregate, SourceFreeAggregate):
        raise TypeError("only SourceFreeAggregate may be serialized for publication")
    # Revalidate at the boundary in case a caller manually constructed the
    # dataclass with mutable mappings and changed them after initialization.
    aggregate.__post_init__()
    return json.dumps(aggregate.to_dict(), sort_keys=True, indent=indent, allow_nan=False) + "\n"
