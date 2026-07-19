"""Pure protocol for an exploratory four-arm repository-quality pilot.

The module contains no inference or model imports.  It freezes scheduling,
routing, allocation, selection, accounting, and serialization while leaving
workspace execution and hidden evaluation to a later adapter.  Hidden outcomes
enter only after every routing and terminal-selection decision is immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import re
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "mio.repository-quality-four-arm-pilot.v2"
FROZEN_SEED = 20260719
ROOT_SCHEDULE_REVISION = "mio.repository-quality.root-order.v1"
STATIC_ALLOCATION_REVISION = "mio.repository-quality.static-order.v1"
EXTRA_SCHEDULE_REVISION = "mio.repository-quality.extra-order.v1"
RECOVERY_PROMPT_REVISION = "mio.repository-quality.recovery-prompt.v1"
SMOKE_SUITE_SHA256 = "d0fef6c7ccfcccbf6dcbc70f973d931f5dba023f45f4335482d72f626c824afc"
DEVELOPMENT_SUITE_SHA256 = "3b9cd3611486e5b3d20a5786249fdc9446af3aecf3a648d507f46a5f5c3208e5"
ALL_SUITE_SHA256 = "32f4a59ab1831b5130fccbcdc3a9affcfe5e03204f7de0423aec106d4251857c"
SUITE_COHORTS = {
    SMOKE_SUITE_SHA256: "smoke",
    DEVELOPMENT_SUITE_SHA256: "development",
    ALL_SUITE_SHA256: "all",
}
SUITE_COUNTS = {
    SMOKE_SUITE_SHA256: 4,
    DEVELOPMENT_SUITE_SHA256: 8,
    ALL_SUITE_SHA256: 12,
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_DOMAIN = b"mio.repository-quality.bootstrap.v1\0"
BOOTSTRAP_LOWER_INDEX = 499
CLASSIFIER_PRECEDENCE = (
    "root_incomplete",
    "scope_invalid",
    "public_fail",
    "public_unknown",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PilotProtocolError(ValueError):
    """Raised before execution when the frozen pilot protocol is invalid."""


class LogicalArm(StrEnum):
    PLAIN = "plain"
    QUALITY = "quality"
    QUALITY_STATIC_EXTRA = "quality_static_extra"
    MARKOV_QUALITY = "markov_quality"


LOGICAL_ARMS = (
    LogicalArm.PLAIN,
    LogicalArm.QUALITY,
    LogicalArm.QUALITY_STATIC_EXTRA,
    LogicalArm.MARKOV_QUALITY,
)


class PhysicalRoot(StrEnum):
    PLAIN = "plain"
    QUALITY_SHARED = "quality_shared"


class PublicState(StrEnum):
    ROOT_INCOMPLETE = "root_incomplete"
    SCOPE_INVALID = "scope_invalid"
    PUBLIC_FAIL = "public_fail"
    PUBLIC_UNKNOWN = "public_unknown"


class VisibleCheckOutcome(StrEnum):
    NOT_RUN = "not_run"
    FAIL = "fail"
    PASS = "pass"


class ExtraAction(StrEnum):
    REFINE = "refine"


class RouterMode(StrEnum):
    EXPLORATORY = "exploratory"
    CALIBRATED = "calibrated"


class CandidateChoice(StrEnum):
    ROOT = "root"
    CHILD = "child"


@dataclass(frozen=True)
class CandidateBudget:
    max_rounds: int = 12
    max_tool_calls: int = 32
    max_output_tokens: int = 2048
    max_wall_seconds: float = 120.0
    max_context_tokens: int = 8192

    def __post_init__(self) -> None:
        for name in ("max_rounds", "max_tool_calls", "max_output_tokens", "max_context_tokens"):
            value = getattr(self, name)
            minimum = 0 if name == "max_tool_calls" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (
            isinstance(self.max_wall_seconds, bool)
            or not isinstance(self.max_wall_seconds, (int, float))
            or not math.isfinite(float(self.max_wall_seconds))
            or self.max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be finite and positive")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "max_rounds": self.max_rounds,
            "max_tool_calls": self.max_tool_calls,
            "max_output_tokens": self.max_output_tokens,
            "max_wall_seconds": float(self.max_wall_seconds),
            "max_context_tokens": self.max_context_tokens,
        }


DIRECT_CANDIDATE_BUDGET = CandidateBudget(
    max_rounds=12,
    max_tool_calls=32,
    max_output_tokens=2048,
    max_wall_seconds=120.0,
    max_context_tokens=8192,
)
EXTRA_CANDIDATE_BUDGET = CandidateBudget(
    max_rounds=4,
    max_tool_calls=8,
    max_output_tokens=384,
    max_wall_seconds=20.0,
    max_context_tokens=8192,
)


@dataclass(frozen=True)
class ExtraSpec:
    action: ExtraAction = ExtraAction.REFINE
    prompt_revision: str = RECOVERY_PROMPT_REVISION
    budget: CandidateBudget = EXTRA_CANDIDATE_BUDGET
    effort: str = "high"

    def __post_init__(self) -> None:
        if not isinstance(self.action, ExtraAction):
            raise TypeError("extra action must be an ExtraAction")
        if self.prompt_revision != RECOVERY_PROMPT_REVISION:
            raise PilotProtocolError("extra prompt revision differs from the frozen preregistration")
        if not isinstance(self.budget, CandidateBudget):
            raise TypeError("extra budget must be a CandidateBudget")
        if self.budget != EXTRA_CANDIDATE_BUDGET:
            raise PilotProtocolError("extra budget differs from the frozen preregistration")
        if self.effort != "high":
            raise ValueError("the frozen extra effort must be high")


@dataclass(frozen=True)
class TransitionIdentity:
    model: str
    config: str
    prompt: str
    corpus: str
    split: str
    backend: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in self._values()):
            raise ValueError("transition identity fields must be non-empty strings")

    def _values(self) -> tuple[str, ...]:
        return (self.model, self.config, self.prompt, self.corpus, self.split, self.backend)

    @property
    def sha256(self) -> str:
        return hashlib.sha256("\0".join(self._values()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CalibratedTransition:
    public_state: PublicState
    coverage_debt: bool
    route: bool
    action: ExtraAction = ExtraAction.REFINE

    def __post_init__(self) -> None:
        if not isinstance(self.public_state, PublicState):
            raise TypeError("public_state must be a PublicState")
        if type(self.coverage_debt) is not bool or type(self.route) is not bool:
            raise TypeError("coverage_debt and route must be bool")
        if not isinstance(self.action, ExtraAction):
            raise TypeError("action must be an ExtraAction")


@dataclass(frozen=True)
class CalibratedTransitionTable:
    identity: TransitionIdentity
    estimates: tuple[CalibratedTransition, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TransitionIdentity):
            raise TypeError("transition table identity must be a TransitionIdentity")
        if not isinstance(self.estimates, tuple) or any(
            not isinstance(item, CalibratedTransition) for item in self.estimates
        ):
            raise TypeError("transition estimates must be a tuple of CalibratedTransition values")
        keys = [(item.public_state, item.coverage_debt) for item in self.estimates]
        if len(keys) != len(set(keys)):
            raise ValueError("transition table contains a duplicate observable state")


@dataclass(frozen=True)
class PublicEvidence:
    """Typed raw public evidence; every controller field is derived here."""

    initial_snapshot_complete: bool = True
    current_snapshot_complete: bool = True
    tool_telemetry_complete: bool = True
    budget_exhausted: bool = False
    deadline_violated: bool = False
    quality_decision: str = "pass"
    terminal_reason: str = "model_final"
    scope_valid: bool = True
    visible_check: VisibleCheckOutcome = VisibleCheckOutcome.NOT_RUN
    mutation_epoch: int = 0
    trusted_test_or_build_attempt_count: int = 0
    trusted_test_count: int = 0
    trusted_build_count: int = 0
    trusted_static_count: int = 0
    trusted_diff_count: int = 0

    def __post_init__(self) -> None:
        flags = (
            self.initial_snapshot_complete,
            self.current_snapshot_complete,
            self.tool_telemetry_complete,
            self.budget_exhausted,
            self.deadline_violated,
            self.scope_valid,
        )
        if any(type(flag) is not bool for flag in flags):
            raise TypeError("public evidence flags must be bool")
        if self.quality_decision not in {"pass", "incomplete", "not_applicable"}:
            raise ValueError("quality_decision is outside the frozen schema")
        if not isinstance(self.terminal_reason, str) or not self.terminal_reason:
            raise TypeError("terminal_reason must be a non-empty string")
        if not isinstance(self.visible_check, VisibleCheckOutcome):
            raise TypeError("visible_check must be a VisibleCheckOutcome")
        for name in (
            "mutation_epoch",
            "trusted_test_or_build_attempt_count",
            "trusted_test_count",
            "trusted_build_count",
            "trusted_static_count",
            "trusted_diff_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"{name} must be a non-negative integer")
        attempts = self.trusted_test_or_build_attempt_count
        successes = self.trusted_test_count + self.trusted_build_count
        if successes > attempts:
            raise PilotProtocolError("trusted test/build successes exceed attempts")
        expected_visible = (
            VisibleCheckOutcome.NOT_RUN
            if attempts == 0
            else VisibleCheckOutcome.PASS
            if successes == attempts
            else VisibleCheckOutcome.FAIL
        )
        if self.visible_check is not expected_visible:
            raise PilotProtocolError("visible check contradicts trusted test/build attempts")

    @property
    def state(self) -> PublicState:
        return classify_public_state(
            initial_snapshot_complete=self.initial_snapshot_complete,
            current_snapshot_complete=self.current_snapshot_complete,
            tool_telemetry_complete=self.tool_telemetry_complete,
            budget_exhausted=self.budget_exhausted,
            deadline_violated=self.deadline_violated,
            quality_decision=self.quality_decision,
            terminal_reason=self.terminal_reason,
            scope_valid=self.scope_valid,
            visible_check=self.visible_check,
        )

    @property
    def snapshot_and_telemetry_complete(self) -> bool:
        return bool(self.initial_snapshot_complete and self.current_snapshot_complete and self.tool_telemetry_complete)

    @property
    def coverage_debt(self) -> bool:
        return bool(self.mutation_epoch >= 2 and self.trusted_static_count == 0 and self.trusted_diff_count == 0)

    @property
    def quality_decision_is_pass(self) -> bool:
        return self.quality_decision == "pass"

    @property
    def terminal_reason_is_model_final(self) -> bool:
        return self.terminal_reason == "model_final"

    @property
    def trusted_test_or_build_present(self) -> bool:
        return self.trusted_test_count + self.trusted_build_count > 0

    @property
    def trusted_static_or_diff_present(self) -> bool:
        return self.trusted_static_count + self.trusted_diff_count > 0

    @property
    def selection_admissible(self) -> bool:
        return bool(
            self.snapshot_and_telemetry_complete
            and self.scope_valid
            and not self.budget_exhausted
            and not self.deadline_violated
        )

    def trajectory_valid(self, *, quality_derived: bool) -> bool:
        return bool(
            self.selection_admissible
            and self.terminal_reason_is_model_final
            and (not quality_derived or self.quality_decision_is_pass)
        )


def classify_public_state(
    *,
    initial_snapshot_complete: bool,
    current_snapshot_complete: bool,
    tool_telemetry_complete: bool,
    budget_exhausted: bool,
    deadline_violated: bool,
    quality_decision: str,
    terminal_reason: str,
    scope_valid: bool,
    visible_check: VisibleCheckOutcome | str,
) -> PublicState:
    """Classify public evidence with total, preregistered precedence.

    A visible pass is deliberately ``PUBLIC_UNKNOWN``: public checks cannot
    certify the hidden coding endpoint.  No hidden label is accepted here.
    """

    flags = (
        initial_snapshot_complete,
        current_snapshot_complete,
        tool_telemetry_complete,
        budget_exhausted,
        deadline_violated,
        scope_valid,
    )
    if any(type(flag) is not bool for flag in flags):
        raise TypeError("public-state classifier flags must be bool")
    if not isinstance(quality_decision, str) or not isinstance(terminal_reason, str):
        raise TypeError("quality_decision and terminal_reason must be strings")
    outcome = VisibleCheckOutcome(visible_check)
    if (
        not initial_snapshot_complete
        or not current_snapshot_complete
        or not tool_telemetry_complete
        or budget_exhausted
        or deadline_violated
        or quality_decision != "pass"
        or terminal_reason != "model_final"
    ):
        return PublicState.ROOT_INCOMPLETE
    if not scope_valid:
        return PublicState.SCOPE_INVALID
    if outcome is VisibleCheckOutcome.FAIL:
        return PublicState.PUBLIC_FAIL
    return PublicState.PUBLIC_UNKNOWN


@dataclass(frozen=True)
class PilotRouter:
    mode: RouterMode = RouterMode.EXPLORATORY
    expected_identity: TransitionIdentity | None = None
    transition_table: CalibratedTransitionTable | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RouterMode):
            raise TypeError("router mode must be a RouterMode")
        if self.mode is RouterMode.EXPLORATORY:
            if self.expected_identity is not None or self.transition_table is not None:
                raise PilotProtocolError("exploratory routing cannot consume calibrated transition state")
            return
        if self.expected_identity is None or self.transition_table is None:
            raise PilotProtocolError("calibrated routing requires an expected identity and transition table")
        if self.transition_table.identity != self.expected_identity:
            raise PilotProtocolError("calibrated transition identity is incompatible with this pilot")
        if not self.transition_table.estimates:
            raise PilotProtocolError("calibrated transition table is empty")

    def should_route(self, evidence: PublicEvidence) -> bool:
        if not isinstance(evidence, PublicEvidence):
            raise TypeError("router accepts only PublicEvidence")
        if not evidence.snapshot_and_telemetry_complete:
            return False
        if self.mode is RouterMode.EXPLORATORY:
            return bool(
                evidence.state
                in {
                    PublicState.ROOT_INCOMPLETE,
                    PublicState.SCOPE_INVALID,
                    PublicState.PUBLIC_FAIL,
                }
                or evidence.coverage_debt
            )
        assert self.transition_table is not None
        for estimate in self.transition_table.estimates:
            if (estimate.public_state, estimate.coverage_debt) == (evidence.state, evidence.coverage_debt):
                return estimate.route
        return False

    @property
    def identity_sha256(self) -> str | None:
        return self.expected_identity.sha256 if self.expected_identity is not None else None


def _fixture_ids(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError("fixture identifiers must be non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError("fixture identifiers must be unique")
    return result


def _rank_digest(seed: int, fixture_id: str, namespace: str) -> bytes:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TypeError("seed must be a non-negative integer")
    payload = namespace.encode("ascii") + b"\0" + str(seed).encode("ascii") + b"\0" + fixture_id.encode("utf-8")
    return hashlib.sha256(payload).digest()


@dataclass(frozen=True)
class RootScheduledRun:
    fixture_id: str
    root: PhysicalRoot
    schedule_index: int


def make_root_schedule(fixture_ids: Iterable[str], *, seed: int) -> tuple[RootScheduledRun, ...]:
    """Counterbalance the two physical roots; Quality is shared by three arms."""

    fixtures = _fixture_ids(fixture_ids)
    ordered = sorted(fixtures, key=lambda item: (_rank_digest(seed, item, ROOT_SCHEDULE_REVISION), item))
    schedule: list[RootScheduledRun] = []
    for pair_index, fixture_id in enumerate(ordered):
        roots = (
            (PhysicalRoot.PLAIN, PhysicalRoot.QUALITY_SHARED)
            if pair_index % 2 == 0
            else (PhysicalRoot.QUALITY_SHARED, PhysicalRoot.PLAIN)
        )
        for root in roots:
            schedule.append(RootScheduledRun(fixture_id, root, len(schedule)))
    return tuple(schedule)


def select_static_fixture_ids(
    fixture_ids: Iterable[str],
    *,
    k: int,
    seed: int,
) -> tuple[str, ...]:
    """Select exactly K Static tasks from seed and IDs, never task evidence."""

    fixtures = _fixture_ids(fixture_ids)
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= len(fixtures):
        raise ValueError("k must be an integer in [0, fixture_count]")
    ordered = sorted(fixtures, key=lambda item: (_rank_digest(seed, item, STATIC_ALLOCATION_REVISION), item))
    return tuple(ordered[:k])


@dataclass(frozen=True)
class ExtraAllocation:
    fixture_ids: tuple[str, ...]
    markov_fixture_ids: tuple[str, ...]
    static_fixture_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        fixtures = _fixture_ids(self.fixture_ids)
        if fixtures != self.fixture_ids:
            raise ValueError("fixture_ids must be a tuple")
        if fixtures != tuple(sorted(fixtures)):
            raise PilotProtocolError("allocation fixture_ids must use canonical Unicode order")
        for name in ("markov_fixture_ids", "static_fixture_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != len(set(values)) or not set(values) <= set(fixtures):
                raise ValueError(f"{name} must be a unique tuple drawn from fixture_ids")
        if len(self.markov_fixture_ids) != len(self.static_fixture_ids):
            raise ValueError("Static and Markov must receive exactly the same number of extras")

    @property
    def k(self) -> int:
        return len(self.markov_fixture_ids)

    @property
    def promotable(self) -> bool:
        return 0 < self.k < len(self.fixture_ids)

    @property
    def nonpromotable_reason(self) -> str | None:
        if self.k == 0:
            return "no_markov_routes"
        if self.k == len(self.fixture_ids):
            return "all_tasks_routed"
        return None


def allocate_extras(
    fixture_ids: Iterable[str],
    evidence_by_fixture: Mapping[str, PublicEvidence],
    *,
    router: PilotRouter,
    seed: int,
) -> ExtraAllocation:
    fixtures = tuple(sorted(_fixture_ids(fixture_ids)))
    if set(evidence_by_fixture) != set(fixtures):
        raise PilotProtocolError("public evidence must cover every fixture exactly once")
    if not isinstance(router, PilotRouter):
        raise TypeError("router must be a PilotRouter")
    routed = tuple(fixture_id for fixture_id in fixtures if router.should_route(evidence_by_fixture[fixture_id]))
    static = select_static_fixture_ids(fixtures, k=len(routed), seed=seed)
    return ExtraAllocation(fixtures, routed, static)


@dataclass(frozen=True)
class ExtraScheduledRun:
    fixture_id: str
    arms: tuple[LogicalArm, ...]
    action: ExtraAction
    budget: CandidateBudget
    schedule_index: int


def make_extra_schedule(
    allocation: ExtraAllocation,
    *,
    spec: ExtraSpec,
    seed: int,
) -> tuple[ExtraScheduledRun, ...]:
    """Order unique physical extras and retain every logical arm reference."""

    if not isinstance(allocation, ExtraAllocation) or not isinstance(spec, ExtraSpec):
        raise TypeError("allocation and spec have the wrong type")
    static = set(allocation.static_fixture_ids)
    markov = set(allocation.markov_fixture_ids)
    unique = sorted(static | markov, key=lambda item: (_rank_digest(seed, item, EXTRA_SCHEDULE_REVISION), item))
    return tuple(
        ExtraScheduledRun(
            fixture_id=fixture_id,
            arms=tuple(
                arm
                for arm in (LogicalArm.QUALITY_STATIC_EXTRA, LogicalArm.MARKOV_QUALITY)
                if (arm is LogicalArm.QUALITY_STATIC_EXTRA and fixture_id in static)
                or (arm is LogicalArm.MARKOV_QUALITY and fixture_id in markov)
            ),
            action=spec.action,
            budget=spec.budget,
            schedule_index=index,
        )
        for index, fixture_id in enumerate(unique)
    )


@dataclass(frozen=True)
class CandidateCost:
    model_rounds: int = 0
    tool_calls: int = 0
    output_tokens: int = 0
    model_seconds: float = 0.0
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("model_rounds", "tool_calls", "output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("model_seconds", "wall_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")

    def __add__(self, other: CandidateCost) -> CandidateCost:
        if not isinstance(other, CandidateCost):
            return NotImplemented
        return CandidateCost(
            model_rounds=self.model_rounds + other.model_rounds,
            tool_calls=self.tool_calls + other.tool_calls,
            output_tokens=self.output_tokens + other.output_tokens,
            model_seconds=self.model_seconds + other.model_seconds,
            wall_seconds=self.wall_seconds + other.wall_seconds,
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "model_rounds": self.model_rounds,
            "tool_calls": self.tool_calls,
            "output_tokens": self.output_tokens,
            "model_seconds": float(self.model_seconds),
            "wall_seconds": float(self.wall_seconds),
        }


@dataclass(frozen=True)
class CandidateObservation:
    """Private candidate record.  Its physical ID is never serialized."""

    physical_candidate_id: str
    terminal_artifact_id: str
    public_evidence: PublicEvidence
    cost: CandidateCost
    attempt_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.physical_candidate_id, str) or not self.physical_candidate_id:
            raise ValueError("physical_candidate_id must be a non-empty private identifier")
        if not isinstance(self.terminal_artifact_id, str) or not _SHA256.fullmatch(self.terminal_artifact_id):
            raise ValueError("terminal_artifact_id must be an exact lowercase tree SHA-256")
        if not isinstance(self.public_evidence, PublicEvidence) or not isinstance(self.cost, CandidateCost):
            raise TypeError("candidate evidence and cost have the wrong type")
        if self.attempt_count != 1:
            raise PilotProtocolError("pilot candidates are terminal and cannot be retried")


_PUBLIC_RANK = {
    PublicState.SCOPE_INVALID: 0,
    PublicState.ROOT_INCOMPLETE: 1,
    PublicState.PUBLIC_FAIL: 2,
    PublicState.PUBLIC_UNKNOWN: 3,
}


def _public_score(candidate: CandidateObservation) -> tuple[int, int, int, int, int]:
    evidence = candidate.public_evidence
    return (
        _PUBLIC_RANK[evidence.state],
        int(evidence.quality_decision_is_pass),
        int(evidence.terminal_reason_is_model_final),
        int(evidence.trusted_test_or_build_present),
        int(evidence.trusted_static_or_diff_present),
    )


def select_candidate(root: CandidateObservation, child: CandidateObservation | None) -> CandidateChoice:
    """Accept a child only for strict public improvement; ties retain root."""

    if not isinstance(root, CandidateObservation):
        raise TypeError("root must be a CandidateObservation")
    if child is None:
        return CandidateChoice.ROOT
    if not isinstance(child, CandidateObservation):
        raise TypeError("child must be a CandidateObservation or None")
    return (
        CandidateChoice.CHILD
        if child.public_evidence.selection_admissible and _public_score(child) > _public_score(root)
        else CandidateChoice.ROOT
    )


@dataclass(frozen=True)
class HiddenOutcome:
    """Reusable verdict determined only by exact terminal workspace bytes."""

    evaluator_passed: bool
    regression_free: bool

    def __post_init__(self) -> None:
        if type(self.evaluator_passed) is not bool or type(self.regression_free) is not bool:
            raise TypeError("hidden outcomes must be bool")
        if self.evaluator_passed and not self.regression_free:
            raise PilotProtocolError("evaluator_passed implies regression_free")


@dataclass(frozen=True)
class ArmHiddenOutcome:
    """Logical-arm outcome with trajectory validity kept arm-specific."""

    arm: LogicalArm
    outcome: HiddenOutcome
    trajectory_valid: bool

    def __post_init__(self) -> None:
        if not isinstance(self.arm, LogicalArm):
            raise TypeError("arm must be a LogicalArm")
        if not isinstance(self.outcome, HiddenOutcome):
            raise TypeError("outcome must be a HiddenOutcome")
        if type(self.trajectory_valid) is not bool:
            raise TypeError("trajectory_valid must be bool")

    @property
    def hidden_task_success(self) -> bool:
        return self.outcome.evaluator_passed and self.trajectory_valid


@dataclass(frozen=True)
class FixturePilotRecord:
    """Private terminal record built only after routing and selection are sealed."""

    fixture_id: str
    plain_root: CandidateObservation
    quality_root: CandidateObservation
    static_child: CandidateObservation | None
    markov_child: CandidateObservation | None
    static_selection: CandidateChoice
    markov_selection: CandidateChoice
    outcomes: tuple[ArmHiddenOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fixture_id, str) or not self.fixture_id:
            raise ValueError("fixture_id must be non-empty")
        if not isinstance(self.plain_root, CandidateObservation) or not isinstance(
            self.quality_root, CandidateObservation
        ):
            raise TypeError("root candidates must be CandidateObservation values")
        if self.static_child is not None and not isinstance(self.static_child, CandidateObservation):
            raise TypeError("static_child must be a CandidateObservation or None")
        if self.markov_child is not None and not isinstance(self.markov_child, CandidateObservation):
            raise TypeError("markov_child must be a CandidateObservation or None")
        if not isinstance(self.outcomes, tuple) or any(
            not isinstance(item, ArmHiddenOutcome) for item in self.outcomes
        ):
            raise TypeError("outcomes must be a tuple of ArmHiddenOutcome values")
        if self.plain_root.physical_candidate_id == self.quality_root.physical_candidate_id:
            raise PilotProtocolError("Plain and shared Quality roots must be distinct physical candidates")
        root_ids = {
            self.plain_root.physical_candidate_id,
            self.quality_root.physical_candidate_id,
        }
        for child in (self.static_child, self.markov_child):
            if child is not None and child.physical_candidate_id in root_ids:
                raise PilotProtocolError("an allocated child must be a distinct physical generation")
        expected_static = select_candidate(self.quality_root, self.static_child)
        expected_markov = select_candidate(self.quality_root, self.markov_child)
        if self.static_selection is not expected_static or self.markov_selection is not expected_markov:
            raise PilotProtocolError("terminal selection does not match the strict public selector")
        if tuple(item.arm for item in self.outcomes) != LOGICAL_ARMS:
            raise PilotProtocolError("hidden outcomes must use the frozen logical-arm order")
        for item in self.outcomes:
            candidate = self.candidate_for_arm(item.arm)
            expected_validity = candidate.public_evidence.trajectory_valid(
                quality_derived=item.arm is not LogicalArm.PLAIN
            )
            if item.trajectory_valid is not expected_validity:
                raise PilotProtocolError("trajectory validity does not match the selected typed evidence")

    def candidate_for_arm(self, arm: LogicalArm) -> CandidateObservation:
        if arm is LogicalArm.PLAIN:
            return self.plain_root
        if arm is LogicalArm.QUALITY:
            return self.quality_root
        if arm is LogicalArm.QUALITY_STATIC_EXTRA:
            return self.static_child if self.static_selection is CandidateChoice.CHILD else self.quality_root
        if arm is LogicalArm.MARKOV_QUALITY:
            return self.markov_child if self.markov_selection is CandidateChoice.CHILD else self.quality_root
        raise ValueError("unknown logical arm")

    def cost_for_arm(self, arm: LogicalArm) -> CandidateCost:
        if arm is LogicalArm.PLAIN:
            return self.plain_root.cost
        if arm is LogicalArm.QUALITY:
            return self.quality_root.cost
        if arm is LogicalArm.QUALITY_STATIC_EXTRA:
            child = self.static_child
        elif arm is LogicalArm.MARKOV_QUALITY:
            child = self.markov_child
        else:
            raise ValueError("unknown logical arm")
        return self.quality_root.cost + (child.cost if child is not None else CandidateCost())

    def outcome_for_arm(self, arm: LogicalArm) -> ArmHiddenOutcome:
        try:
            return next(item for item in self.outcomes if item.arm is arm)
        except StopIteration as error:
            raise ValueError("unknown logical arm") from error


@dataclass(frozen=True)
class PilotProtocol:
    suite_sha256: str
    seed: int
    root_budget: CandidateBudget = DIRECT_CANDIDATE_BUDGET
    extra_spec: ExtraSpec = ExtraSpec()
    router: PilotRouter = PilotRouter()

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.suite_sha256) or self.suite_sha256 not in SUITE_COHORTS:
            raise ValueError("suite_sha256 is not a frozen MioCodeBench cohort")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be the frozen integer")
        if self.seed != FROZEN_SEED:
            raise PilotProtocolError("pilot seed differs from the frozen preregistration")
        if not isinstance(self.root_budget, CandidateBudget):
            raise TypeError("root_budget must be a CandidateBudget")
        if self.root_budget != DIRECT_CANDIDATE_BUDGET:
            raise PilotProtocolError("root budget differs from the frozen preregistration")
        if not isinstance(self.extra_spec, ExtraSpec) or not isinstance(self.router, PilotRouter):
            raise TypeError("extra_spec and router have the wrong type")
        if self.router.mode is not RouterMode.EXPLORATORY:
            raise PilotProtocolError("this protocol revision permits only the exploratory router")

    @property
    def cohort(self) -> str:
        return SUITE_COHORTS[self.suite_sha256]

    @property
    def expected_fixture_count(self) -> int:
        return SUITE_COUNTS[self.suite_sha256]


@dataclass(frozen=True)
class ArmAggregate:
    run_count: int
    passed_count: int
    workspace_evaluator_passed_count: int
    regression_free_count: int
    terminal_completion_count: int
    selected_child_count: int
    extra_candidate_count: int
    logical_candidate_count: int
    logical_cost: CandidateCost

    def to_dict(self) -> dict[str, object]:
        return {
            "run_count": self.run_count,
            "passed_count": self.passed_count,
            "passed_rate": self.passed_count / self.run_count,
            "workspace_evaluator_passed_count": self.workspace_evaluator_passed_count,
            "workspace_evaluator_passed_rate": (self.workspace_evaluator_passed_count / self.run_count),
            "regression_free_count": self.regression_free_count,
            "regression_free_rate": self.regression_free_count / self.run_count,
            "terminal_completion_count": self.terminal_completion_count,
            "terminal_completion_rate": self.terminal_completion_count / self.run_count,
            "selected_child_count": self.selected_child_count,
            "extra_candidate_count": self.extra_candidate_count,
            "logical_candidate_count": self.logical_candidate_count,
            "logical_cost": self.logical_cost.to_dict(),
        }


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


@dataclass(frozen=True)
class PairedContrast:
    baseline: LogicalArm
    candidate: LogicalArm
    both_pass: int
    baseline_only: int
    candidate_only: int
    neither_pass: int
    pass_rate_delta: float
    workspace_evaluator_both_pass: int
    workspace_evaluator_baseline_only: int
    workspace_evaluator_candidate_only: int
    workspace_evaluator_neither_pass: int
    workspace_evaluator_pass_rate_delta: float
    output_token_ratio: float | None
    model_seconds_ratio: float | None
    wall_seconds_ratio: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.value,
            "candidate": self.candidate.value,
            "paired_contingency": {
                "both_pass": self.both_pass,
                "baseline_only": self.baseline_only,
                "candidate_only": self.candidate_only,
                "neither_pass": self.neither_pass,
            },
            "pass_rate_delta": self.pass_rate_delta,
            "workspace_evaluator_paired_contingency": {
                "both_pass": self.workspace_evaluator_both_pass,
                "baseline_only": self.workspace_evaluator_baseline_only,
                "candidate_only": self.workspace_evaluator_candidate_only,
                "neither_pass": self.workspace_evaluator_neither_pass,
            },
            "workspace_evaluator_pass_rate_delta": (self.workspace_evaluator_pass_rate_delta),
            "logical_cost_ratios": {
                "output_tokens": self.output_token_ratio,
                "model_seconds": self.model_seconds_ratio,
                "wall_seconds": self.wall_seconds_ratio,
            },
        }


@dataclass(frozen=True)
class PhysicalCostAggregate:
    unique_candidate_count: int
    logical_candidate_reference_count: int
    shared_or_deduplicated_reference_count: int
    plain_root_count: int
    shared_quality_root_count: int
    static_extra_count: int
    markov_extra_count: int
    unique_extra_count: int
    cost: CandidateCost

    def to_dict(self) -> dict[str, object]:
        return {
            "unique_candidate_count": self.unique_candidate_count,
            "logical_candidate_reference_count": self.logical_candidate_reference_count,
            "shared_or_deduplicated_reference_count": self.shared_or_deduplicated_reference_count,
            "plain_root_count": self.plain_root_count,
            "shared_quality_root_count": self.shared_quality_root_count,
            "static_extra_count": self.static_extra_count,
            "markov_extra_count": self.markov_extra_count,
            "unique_extra_count": self.unique_extra_count,
            "cost": self.cost.to_dict(),
        }


@dataclass(frozen=True)
class GenerationCompletionReceipt:
    fixture_count: int
    expected_root_generation_count: int
    completed_root_generation_count: int
    expected_unique_extra_generation_count: int
    completed_unique_extra_generation_count: int
    root_schedule_sealed_before_first_generation: bool
    allocation_sealed_after_all_roots: bool
    extra_schedule_sealed_before_first_extra: bool

    def __post_init__(self) -> None:
        for name in (
            "fixture_count",
            "expected_root_generation_count",
            "completed_root_generation_count",
            "expected_unique_extra_generation_count",
            "completed_unique_extra_generation_count",
        ):
            value = getattr(self, name)
            minimum = 1 if name == "fixture_count" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise PilotProtocolError(f"{name} is not a valid generation count")
        if self.expected_root_generation_count != self.fixture_count * 2:
            raise PilotProtocolError("root generation count is not exactly two per fixture")
        if self.completed_root_generation_count != self.expected_root_generation_count:
            raise PilotProtocolError("not every scheduled root generation completed")
        if self.completed_unique_extra_generation_count != self.expected_unique_extra_generation_count:
            raise PilotProtocolError("not every scheduled unique extra generation completed")
        for name in (
            "root_schedule_sealed_before_first_generation",
            "allocation_sealed_after_all_roots",
            "extra_schedule_sealed_before_first_extra",
        ):
            if getattr(self, name) is not True:
                raise PilotProtocolError(f"generation receipt does not attest {name}")


@dataclass(frozen=True)
class EvaluationBarrierReceipt:
    expected_logical_selection_count: int
    registered_logical_selection_count: int
    unique_terminal_artifact_count: int
    hidden_evaluation_count: int
    all_generation_complete_before_seal: bool
    selection_sealed_before_hidden: bool
    hidden_evaluation_single_use: bool

    def __post_init__(self) -> None:
        for name in (
            "expected_logical_selection_count",
            "registered_logical_selection_count",
            "unique_terminal_artifact_count",
            "hidden_evaluation_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PilotProtocolError(f"{name} must be a positive integer")
        if self.registered_logical_selection_count != self.expected_logical_selection_count:
            raise PilotProtocolError("selection barrier did not register every logical terminal")
        if self.hidden_evaluation_count != self.unique_terminal_artifact_count:
            raise PilotProtocolError("hidden evaluator calls do not match unique terminal artifacts")
        for name in (
            "all_generation_complete_before_seal",
            "selection_sealed_before_hidden",
            "hidden_evaluation_single_use",
        ):
            if getattr(self, name) is not True:
                raise PilotProtocolError(f"barrier receipt does not attest {name}")


@dataclass(frozen=True)
class PromotionAnalysis:
    cohort: str
    fixture_count: int
    route_count: int
    quality_gain_point: float
    quality_gain_lcb: float
    workspace_evaluator_gain_point: float
    workspace_evaluator_gain_lcb: float
    rescue_numerator: int
    rescue_denominator: int
    rescue_probability_point: float | None
    rescue_probability_lcb: float
    workspace_evaluator_rescue_count: int
    trajectory_only_rescue_count: int
    changed_terminal_rescue_count: int
    same_terminal_rescue_count: int
    byte_changed_selected_child_count: int
    quality_to_markov_regressions: int
    workspace_evaluator_regressions: int
    quality_pass_count: int
    static_pass_count: int
    markov_pass_count: int
    quality_workspace_evaluator_pass_count: int
    static_workspace_evaluator_pass_count: int
    markov_workspace_evaluator_pass_count: int
    wall_point_ratio: float | None
    model_seconds_point_ratio: float | None
    output_tokens_point_ratio: float | None
    budget_deadline_snapshot_or_telemetry_violation_count: int
    bootstrap_samples: int
    promotion_eligible: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "complete_frozen_analysis",
            "population": {
                "cohort": self.cohort,
                "fixture_count": self.fixture_count,
                "pooled_cohorts": False,
            },
            "route_count": self.route_count,
            "quality_gain": {
                "point": self.quality_gain_point,
                "lcb_95_one_sided": self.quality_gain_lcb,
            },
            "workspace_evaluator_gain": {
                "point": self.workspace_evaluator_gain_point,
                "lcb_95_one_sided": self.workspace_evaluator_gain_lcb,
            },
            "rescue_probability": {
                "numerator": self.rescue_numerator,
                "denominator": self.rescue_denominator,
                "point": self.rescue_probability_point,
                "lcb_95_one_sided": self.rescue_probability_lcb,
            },
            "rescue_mechanism_counts": {
                "workspace_evaluator_rescue": self.workspace_evaluator_rescue_count,
                "trajectory_only_rescue": self.trajectory_only_rescue_count,
                "changed_terminal": self.changed_terminal_rescue_count,
                "same_terminal": self.same_terminal_rescue_count,
            },
            "byte_changed_selected_child_count": self.byte_changed_selected_child_count,
            "quality_to_markov_regressions": self.quality_to_markov_regressions,
            "workspace_evaluator_regressions": self.workspace_evaluator_regressions,
            "composite_pass_counts": {
                "quality": self.quality_pass_count,
                "quality_static_extra": self.static_pass_count,
                "markov_quality": self.markov_pass_count,
            },
            "workspace_evaluator_pass_counts": {
                "quality": self.quality_workspace_evaluator_pass_count,
                "quality_static_extra": self.static_workspace_evaluator_pass_count,
                "markov_quality": self.markov_workspace_evaluator_pass_count,
            },
            "logical_markov_over_quality_point_ratios": {
                "wall_seconds": self.wall_point_ratio,
                "model_seconds": self.model_seconds_point_ratio,
                "output_tokens": self.output_tokens_point_ratio,
            },
            "budget_deadline_snapshot_or_telemetry_violation_count": (
                self.budget_deadline_snapshot_or_telemetry_violation_count
            ),
            "bootstrap": {
                "samples": self.bootstrap_samples,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "seed": BOOTSTRAP_SEED,
                "index_generator": "sha256-counter-modulo-n-v1",
                "lower_order_statistic_zero_based": BOOTSTRAP_LOWER_INDEX,
            },
            "promotion_eligible": self.promotion_eligible,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class PilotAggregate:
    protocol: PilotProtocol
    allocation: ExtraAllocation
    arm_metrics: tuple[tuple[LogicalArm, ArmAggregate], ...]
    contrasts: tuple[tuple[str, PairedContrast], ...]
    physical_costs: PhysicalCostAggregate
    analysis: PromotionAnalysis
    barrier_receipt: EvaluationBarrierReceipt
    fallback_to_root_count: int
    selected_outputs_consistent: bool

    def __post_init__(self) -> None:
        if tuple(arm for arm, _metrics in self.arm_metrics) != LOGICAL_ARMS:
            raise ValueError("arm metrics do not use the frozen arm order")
        expected_contrasts = (
            "quality_vs_plain",
            "static_vs_quality",
            "markov_vs_quality",
            "markov_vs_static",
        )
        if tuple(name for name, _contrast in self.contrasts) != expected_contrasts:
            raise ValueError("paired contrasts do not use the frozen order")
        if type(self.selected_outputs_consistent) is not bool:
            raise TypeError("selected_outputs_consistent must be bool")
        if self.barrier_receipt.expected_logical_selection_count != len(self.allocation.fixture_ids) * len(
            LOGICAL_ARMS
        ):
            raise PilotProtocolError("barrier receipt logical count does not match the cohort")

    def to_dict(self) -> dict[str, object]:
        nonpromotable = self.allocation.nonpromotable_reason
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": {
                "suite_sha256": self.protocol.suite_sha256,
                "cohort": self.protocol.cohort,
                "seed": self.protocol.seed,
                "logical_arms": [arm.value for arm in LOGICAL_ARMS],
                "shared_quality_root_arms": [
                    LogicalArm.QUALITY.value,
                    LogicalArm.QUALITY_STATIC_EXTRA.value,
                    LogicalArm.MARKOV_QUALITY.value,
                ],
                "root_schedule_revision": ROOT_SCHEDULE_REVISION,
                "static_allocation_revision": STATIC_ALLOCATION_REVISION,
                "extra_schedule_revision": EXTRA_SCHEDULE_REVISION,
                "classifier_precedence": list(CLASSIFIER_PRECEDENCE),
                "visible_pass_semantics": PublicState.PUBLIC_UNKNOWN.value,
                "selector": "strict_public_improvement_else_root",
                "retry_policy": "none_terminal_public_state_v1",
                "root_budget": self.protocol.root_budget.to_dict(),
                "extra": {
                    "action": self.protocol.extra_spec.action.value,
                    "prompt_revision": self.protocol.extra_spec.prompt_revision,
                    "effort": self.protocol.extra_spec.effort,
                    "budget": self.protocol.extra_spec.budget.to_dict(),
                },
                "root_effort": "medium",
                "router_mode": self.protocol.router.mode.value,
                "transition_identity_sha256": self.protocol.router.identity_sha256,
            },
            "integrity": {
                "fixture_count": len(self.allocation.fixture_ids),
                "all_generation_before_hidden_evaluation": (self.barrier_receipt.all_generation_complete_before_seal),
                "terminal_selection_precedes_hidden_evaluation": (self.barrier_receipt.selection_sealed_before_hidden),
                "hidden_evaluation_single_use": self.barrier_receipt.hidden_evaluation_single_use,
                "shared_quality_root": True,
                "static_markov_k_equal": True,
                "static_extra_count": self.allocation.k,
                "markov_extra_count": self.allocation.k,
                "route_count": self.allocation.k,
                "fallback_to_root_count": self.fallback_to_root_count,
                "unique_terminal_artifact_count": self.barrier_receipt.unique_terminal_artifact_count,
                "hidden_evaluation_count": self.barrier_receipt.hidden_evaluation_count,
                "budget_deadline_snapshot_or_telemetry_violation_count": (
                    self.analysis.budget_deadline_snapshot_or_telemetry_violation_count
                ),
                "selected_outputs_consistent": self.selected_outputs_consistent,
            },
            "arm_metrics": {arm.value: metrics.to_dict() for arm, metrics in self.arm_metrics},
            "paired_contrasts": {name: contrast.to_dict() for name, contrast in self.contrasts},
            "physical_costs": self.physical_costs.to_dict(),
            "analysis": self.analysis.to_dict(),
            "claim": {
                "status": "exploratory_no_claim",
                "scientific_claim_eligible": False,
                "advance_to_new_calibration_eligible": self.analysis.promotion_eligible,
                "nonpromotable_reason": nonpromotable,
                "promotion_failures": list(self.analysis.failures),
            },
            "hidden_labels_serialized": False,
        }


def _sum_costs(values: Iterable[CandidateCost]) -> CandidateCost:
    total = CandidateCost()
    for value in values:
        total += value
    return total


def _arm_aggregate(records: Sequence[FixturePilotRecord], arm: LogicalArm) -> ArmAggregate:
    outcomes = [record.outcome_for_arm(arm) for record in records]
    costs = [record.cost_for_arm(arm) for record in records]
    if arm is LogicalArm.QUALITY_STATIC_EXTRA:
        extras = sum(record.static_child is not None for record in records)
        selected = sum(record.static_selection is CandidateChoice.CHILD for record in records)
    elif arm is LogicalArm.MARKOV_QUALITY:
        extras = sum(record.markov_child is not None for record in records)
        selected = sum(record.markov_selection is CandidateChoice.CHILD for record in records)
    else:
        extras = selected = 0
    return ArmAggregate(
        run_count=len(records),
        passed_count=sum(item.hidden_task_success for item in outcomes),
        workspace_evaluator_passed_count=sum(item.outcome.evaluator_passed for item in outcomes),
        regression_free_count=sum(item.outcome.regression_free for item in outcomes),
        terminal_completion_count=sum(item.trajectory_valid for item in outcomes),
        selected_child_count=selected,
        extra_candidate_count=extras,
        logical_candidate_count=len(records) + extras,
        logical_cost=_sum_costs(costs),
    )


def _contrast(
    records: Sequence[FixturePilotRecord],
    metrics: Mapping[LogicalArm, ArmAggregate],
    baseline: LogicalArm,
    candidate: LogicalArm,
) -> PairedContrast:
    both = baseline_only = candidate_only = neither = 0
    raw_both = raw_baseline_only = raw_candidate_only = raw_neither = 0
    for record in records:
        left_outcome = record.outcome_for_arm(baseline)
        right_outcome = record.outcome_for_arm(candidate)
        left = left_outcome.hidden_task_success
        right = right_outcome.hidden_task_success
        if left and right:
            both += 1
        elif left:
            baseline_only += 1
        elif right:
            candidate_only += 1
        else:
            neither += 1
        raw_left = left_outcome.outcome.evaluator_passed
        raw_right = right_outcome.outcome.evaluator_passed
        if raw_left and raw_right:
            raw_both += 1
        elif raw_left:
            raw_baseline_only += 1
        elif raw_right:
            raw_candidate_only += 1
        else:
            raw_neither += 1
    left_cost = metrics[baseline].logical_cost
    right_cost = metrics[candidate].logical_cost
    return PairedContrast(
        baseline=baseline,
        candidate=candidate,
        both_pass=both,
        baseline_only=baseline_only,
        candidate_only=candidate_only,
        neither_pass=neither,
        pass_rate_delta=(candidate_only - baseline_only) / len(records),
        workspace_evaluator_both_pass=raw_both,
        workspace_evaluator_baseline_only=raw_baseline_only,
        workspace_evaluator_candidate_only=raw_candidate_only,
        workspace_evaluator_neither_pass=raw_neither,
        workspace_evaluator_pass_rate_delta=(raw_candidate_only - raw_baseline_only) / len(records),
        output_token_ratio=_ratio(right_cost.output_tokens, left_cost.output_tokens),
        model_seconds_ratio=_ratio(right_cost.model_seconds, left_cost.model_seconds),
        wall_seconds_ratio=_ratio(right_cost.wall_seconds, left_cost.wall_seconds),
    )


def _bootstrap_index(*, fixture_count: int, sample_index: int, draw_index: int) -> int:
    payload = (
        BOOTSTRAP_DOMAIN
        + str(BOOTSTRAP_SEED).encode("ascii")
        + b"\0"
        + sample_index.to_bytes(8, "big")
        + draw_index.to_bytes(8, "big")
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % fixture_count


def _conservative_lcb(point: float, samples: Sequence[float]) -> float:
    if len(samples) != BOOTSTRAP_SAMPLES:
        raise PilotProtocolError("bootstrap sample count drifted")
    ordered = sorted(samples)
    return min(point, ordered[BOOTSTRAP_LOWER_INDEX])


def _build_promotion_analysis(
    *,
    protocol: PilotProtocol,
    allocation: ExtraAllocation,
    records: Sequence[FixturePilotRecord],
    metrics: Mapping[LogicalArm, ArmAggregate],
    physical_candidates: Mapping[str, CandidateObservation],
    selected_outputs_consistent: bool,
) -> PromotionAnalysis:
    fixture_count = len(records)
    quality = tuple(record.outcome_for_arm(LogicalArm.QUALITY).hidden_task_success for record in records)
    markov = tuple(record.outcome_for_arm(LogicalArm.MARKOV_QUALITY).hidden_task_success for record in records)
    quality_workspace = tuple(record.outcome_for_arm(LogicalArm.QUALITY).outcome.evaluator_passed for record in records)
    markov_workspace = tuple(
        record.outcome_for_arm(LogicalArm.MARKOV_QUALITY).outcome.evaluator_passed for record in records
    )
    routed = frozenset(allocation.markov_fixture_ids)
    routed_quality_failures = tuple(
        index for index, record in enumerate(records) if record.fixture_id in routed and not quality[index]
    )
    rescue_numerator = sum(markov[index] for index in routed_quality_failures)
    rescue_denominator = len(routed_quality_failures)
    rescue_point = rescue_numerator / rescue_denominator if rescue_denominator > 0 else None
    quality_gain_point = (
        sum(int(right) - int(left) for left, right in zip(quality, markov, strict=True)) / fixture_count
    )
    workspace_evaluator_gain_point = (
        sum(int(right) - int(left) for left, right in zip(quality_workspace, markov_workspace, strict=True))
        / fixture_count
    )

    composite_rescue_indices = tuple(index for index in routed_quality_failures if markov[index])
    workspace_evaluator_rescue_count = sum(
        not quality_workspace[index] and markov_workspace[index] for index in composite_rescue_indices
    )
    trajectory_only_rescue_count = len(composite_rescue_indices) - workspace_evaluator_rescue_count
    changed_terminal_rescue_count = sum(
        records[index].candidate_for_arm(LogicalArm.QUALITY).terminal_artifact_id
        != records[index].candidate_for_arm(LogicalArm.MARKOV_QUALITY).terminal_artifact_id
        for index in composite_rescue_indices
    )
    same_terminal_rescue_count = len(composite_rescue_indices) - changed_terminal_rescue_count
    byte_changed_selected_child_count = sum(
        record.markov_selection is CandidateChoice.CHILD
        and record.markov_child is not None
        and record.markov_child.terminal_artifact_id != record.quality_root.terminal_artifact_id
        for record in records
    )

    gain_samples: list[float] = []
    workspace_gain_samples: list[float] = []
    rescue_samples: list[float] = []
    for sample_index in range(BOOTSTRAP_SAMPLES):
        indices = tuple(
            _bootstrap_index(
                fixture_count=fixture_count,
                sample_index=sample_index,
                draw_index=draw_index,
            )
            for draw_index in range(fixture_count)
        )
        gain_samples.append(sum(int(markov[index]) - int(quality[index]) for index in indices) / fixture_count)
        workspace_gain_samples.append(
            sum(int(markov_workspace[index]) - int(quality_workspace[index]) for index in indices) / fixture_count
        )
        rescue_indices = tuple(index for index in indices if records[index].fixture_id in routed and not quality[index])
        rescue_samples.append(
            sum(markov[index] for index in rescue_indices) / len(rescue_indices) if rescue_indices else 0.0
        )
    quality_gain_lcb = _conservative_lcb(quality_gain_point, gain_samples)
    workspace_evaluator_gain_lcb = _conservative_lcb(
        workspace_evaluator_gain_point,
        workspace_gain_samples,
    )
    rescue_probability_lcb = _conservative_lcb(rescue_point or 0.0, rescue_samples)

    quality_metrics = metrics[LogicalArm.QUALITY]
    static_metrics = metrics[LogicalArm.QUALITY_STATIC_EXTRA]
    markov_metrics = metrics[LogicalArm.MARKOV_QUALITY]
    wall_ratio = _ratio(markov_metrics.logical_cost.wall_seconds, quality_metrics.logical_cost.wall_seconds)
    model_ratio = _ratio(markov_metrics.logical_cost.model_seconds, quality_metrics.logical_cost.model_seconds)
    token_ratio = _ratio(markov_metrics.logical_cost.output_tokens, quality_metrics.logical_cost.output_tokens)
    regressions = sum(left and not right for left, right in zip(quality, markov, strict=True))
    workspace_evaluator_regressions = sum(
        left and not right for left, right in zip(quality_workspace, markov_workspace, strict=True)
    )
    violation_count = sum(
        (not candidate.public_evidence.snapshot_and_telemetry_complete)
        or candidate.public_evidence.budget_exhausted
        or candidate.public_evidence.deadline_violated
        for candidate in physical_candidates.values()
    )

    failures: list[str] = []
    if protocol.cohort != "all" or fixture_count != 12:
        failures.append("analysis_population_not_single_all_cohort")
    if allocation.k < 8:
        failures.append("markov_route_count_below_8")
    if allocation.k >= fixture_count:
        failures.append("markov_route_count_is_all_tasks")
    if rescue_denominator < 1:
        failures.append("rescue_denominator_below_1")
    if quality_gain_lcb < 0.01:
        failures.append("quality_gain_lcb_below_0.01")
    if workspace_evaluator_gain_lcb < 0.01:
        failures.append("workspace_evaluator_gain_lcb_below_0.01")
    if rescue_probability_lcb < 0.10:
        failures.append("rescue_probability_lcb_below_0.10")
    if regressions != 0:
        failures.append("quality_to_markov_regressions_nonzero")
    if workspace_evaluator_regressions != 0:
        failures.append("workspace_evaluator_regressions_nonzero")
    if markov_metrics.passed_count < quality_metrics.passed_count:
        failures.append("markov_pass_below_quality")
    if markov_metrics.passed_count < static_metrics.passed_count:
        failures.append("markov_pass_below_static")
    if markov_metrics.workspace_evaluator_passed_count < quality_metrics.workspace_evaluator_passed_count:
        failures.append("markov_workspace_evaluator_pass_below_quality")
    if markov_metrics.workspace_evaluator_passed_count < static_metrics.workspace_evaluator_passed_count:
        failures.append("markov_workspace_evaluator_pass_below_static")
    for label, ratio, maximum in (
        ("wall", wall_ratio, 1.25),
        ("model_seconds", model_ratio, 1.25),
        ("output_tokens", token_ratio, 1.10),
    ):
        if ratio is None or not math.isfinite(ratio) or ratio > maximum:
            failures.append(f"markov_over_quality_{label}_point_ratio_invalid_or_above_limit")
    if violation_count:
        failures.append("budget_deadline_snapshot_or_telemetry_violation")
    if not selected_outputs_consistent:
        failures.append("workspace_evaluator_verdict_inconsistent")

    return PromotionAnalysis(
        cohort=protocol.cohort,
        fixture_count=fixture_count,
        route_count=allocation.k,
        quality_gain_point=quality_gain_point,
        quality_gain_lcb=quality_gain_lcb,
        workspace_evaluator_gain_point=workspace_evaluator_gain_point,
        workspace_evaluator_gain_lcb=workspace_evaluator_gain_lcb,
        rescue_numerator=rescue_numerator,
        rescue_denominator=rescue_denominator,
        rescue_probability_point=rescue_point,
        rescue_probability_lcb=rescue_probability_lcb,
        workspace_evaluator_rescue_count=workspace_evaluator_rescue_count,
        trajectory_only_rescue_count=trajectory_only_rescue_count,
        changed_terminal_rescue_count=changed_terminal_rescue_count,
        same_terminal_rescue_count=same_terminal_rescue_count,
        byte_changed_selected_child_count=byte_changed_selected_child_count,
        quality_to_markov_regressions=regressions,
        workspace_evaluator_regressions=workspace_evaluator_regressions,
        quality_pass_count=quality_metrics.passed_count,
        static_pass_count=static_metrics.passed_count,
        markov_pass_count=markov_metrics.passed_count,
        quality_workspace_evaluator_pass_count=(quality_metrics.workspace_evaluator_passed_count),
        static_workspace_evaluator_pass_count=(static_metrics.workspace_evaluator_passed_count),
        markov_workspace_evaluator_pass_count=(markov_metrics.workspace_evaluator_passed_count),
        wall_point_ratio=wall_ratio,
        model_seconds_point_ratio=model_ratio,
        output_tokens_point_ratio=token_ratio,
        budget_deadline_snapshot_or_telemetry_violation_count=violation_count,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        promotion_eligible=not failures,
        failures=tuple(failures),
    )


def build_aggregate(
    *,
    protocol: PilotProtocol,
    allocation: ExtraAllocation,
    records: Sequence[FixturePilotRecord],
    barrier_receipt: EvaluationBarrierReceipt,
) -> PilotAggregate:
    """Aggregate already-sealed terminal records without exposing private rows."""

    if (
        not isinstance(protocol, PilotProtocol)
        or not isinstance(allocation, ExtraAllocation)
        or not isinstance(barrier_receipt, EvaluationBarrierReceipt)
    ):
        raise TypeError("protocol, allocation, and barrier receipt have the wrong type")
    materialized = tuple(records)
    if not materialized or any(not isinstance(item, FixturePilotRecord) for item in materialized):
        raise TypeError("records must be a non-empty FixturePilotRecord sequence")
    by_fixture = {record.fixture_id: record for record in materialized}
    if len(by_fixture) != len(materialized) or set(by_fixture) != set(allocation.fixture_ids):
        raise PilotProtocolError("terminal records must cover the allocation fixtures exactly once")
    if len(materialized) != protocol.expected_fixture_count:
        raise PilotProtocolError("terminal record count does not match the frozen cohort")
    static_set = set(allocation.static_fixture_ids)
    markov_set = set(allocation.markov_fixture_ids)
    expected_markov = tuple(
        fixture_id
        for fixture_id in allocation.fixture_ids
        if protocol.router.should_route(by_fixture[fixture_id].quality_root.public_evidence)
    )
    expected_static = select_static_fixture_ids(
        allocation.fixture_ids,
        k=len(expected_markov),
        seed=protocol.seed,
    )
    if allocation.markov_fixture_ids != expected_markov or allocation.static_fixture_ids != expected_static:
        raise PilotProtocolError("extra allocation does not match the sealed router and static hash")
    for fixture_id, record in by_fixture.items():
        if (record.static_child is not None) != (fixture_id in static_set):
            raise PilotProtocolError("Static child presence does not match the sealed allocation")
        if (record.markov_child is not None) != (fixture_id in markov_set):
            raise PilotProtocolError("Markov child presence does not match the sealed allocation")
        if fixture_id in static_set & markov_set and record.static_child != record.markov_child:
            raise PilotProtocolError("overlapping Static and Markov extras must reuse one physical child")

    ordered = tuple(by_fixture[fixture_id] for fixture_id in sorted(allocation.fixture_ids))
    metrics_pairs = tuple((arm, _arm_aggregate(ordered, arm)) for arm in LOGICAL_ARMS)
    metrics = dict(metrics_pairs)
    contrasts = (
        ("quality_vs_plain", _contrast(ordered, metrics, LogicalArm.PLAIN, LogicalArm.QUALITY)),
        (
            "static_vs_quality",
            _contrast(ordered, metrics, LogicalArm.QUALITY, LogicalArm.QUALITY_STATIC_EXTRA),
        ),
        ("markov_vs_quality", _contrast(ordered, metrics, LogicalArm.QUALITY, LogicalArm.MARKOV_QUALITY)),
        (
            "markov_vs_static",
            _contrast(ordered, metrics, LogicalArm.QUALITY_STATIC_EXTRA, LogicalArm.MARKOV_QUALITY),
        ),
    )

    physical: dict[str, CandidateObservation] = {}
    physical_owner: dict[str, str] = {}
    logical_references = 0
    selected_outcomes: dict[tuple[str, str], HiddenOutcome] = {}
    selected_outputs_consistent = True
    for record in ordered:
        candidates = (record.plain_root, record.quality_root, record.static_child, record.markov_child)
        for candidate in candidates:
            if candidate is None:
                continue
            previous = physical.setdefault(candidate.physical_candidate_id, candidate)
            if previous != candidate:
                raise PilotProtocolError("one physical candidate ID describes conflicting observations")
            owner = physical_owner.setdefault(candidate.physical_candidate_id, record.fixture_id)
            if owner != record.fixture_id:
                raise PilotProtocolError("one physical candidate cannot belong to two fixtures")
        logical_references += 4 + int(record.static_child is not None) + int(record.markov_child is not None)
        for arm in LOGICAL_ARMS:
            candidate = record.candidate_for_arm(arm)
            artifact_key = (record.fixture_id, candidate.terminal_artifact_id)
            # Exact workspace evaluator verdicts may be reused for identical
            # fixture/workspace bytes. Logical trajectory validity must not be.
            outcome = record.outcome_for_arm(arm).outcome
            previous_outcome = selected_outcomes.setdefault(artifact_key, outcome)
            selected_outputs_consistent = selected_outputs_consistent and previous_outcome == outcome

    physical_costs = PhysicalCostAggregate(
        unique_candidate_count=len(physical),
        logical_candidate_reference_count=logical_references,
        shared_or_deduplicated_reference_count=logical_references - len(physical),
        plain_root_count=len(ordered),
        shared_quality_root_count=len(ordered),
        static_extra_count=allocation.k,
        markov_extra_count=allocation.k,
        unique_extra_count=len(static_set | markov_set),
        cost=_sum_costs(candidate.cost for candidate in physical.values()),
    )
    if barrier_receipt.expected_logical_selection_count != len(ordered) * len(LOGICAL_ARMS):
        raise PilotProtocolError("barrier receipt expected count does not match terminal records")
    if barrier_receipt.unique_terminal_artifact_count != len(selected_outcomes):
        raise PilotProtocolError("barrier receipt artifact count does not match terminal records")
    analysis = _build_promotion_analysis(
        protocol=protocol,
        allocation=allocation,
        records=ordered,
        metrics=metrics,
        physical_candidates=physical,
        selected_outputs_consistent=selected_outputs_consistent,
    )
    fallback_to_root_count = sum(
        record.static_child is not None and record.static_selection is CandidateChoice.ROOT for record in ordered
    ) + sum(record.markov_child is not None and record.markov_selection is CandidateChoice.ROOT for record in ordered)
    return PilotAggregate(
        protocol=protocol,
        allocation=allocation,
        arm_metrics=metrics_pairs,
        contrasts=contrasts,
        physical_costs=physical_costs,
        analysis=analysis,
        barrier_receipt=barrier_receipt,
        fallback_to_root_count=fallback_to_root_count,
        selected_outputs_consistent=selected_outputs_consistent,
    )


_SENSITIVE_KEYS = {
    "assistant_text",
    "completion",
    "content",
    "fixture_id",
    "hidden_checks",
    "instruction",
    "path",
    "physical_candidate_id",
    "terminal_artifact_id",
    "prompt",
    "record",
    "records",
    "tool_output",
    "workspace",
}


def _assert_source_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or key.casefold() in _SENSITIVE_KEYS:
                raise ValueError("public pilot artifact contains a sensitive key")
            _assert_source_free(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_source_free(item)


def serialize_source_free_aggregate(aggregate: PilotAggregate, *, indent: int | None = 2) -> str:
    if not isinstance(aggregate, PilotAggregate):
        raise TypeError("only PilotAggregate can cross the public serialization boundary")
    aggregate.__post_init__()
    payload = aggregate.to_dict()
    _assert_source_free(payload)
    return json.dumps(payload, sort_keys=True, indent=indent, allow_nan=False) + "\n"
