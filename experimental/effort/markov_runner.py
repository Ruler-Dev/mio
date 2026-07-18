"""Backend-independent, two-channel runner for Markov effort experiments.

The public channel contains the task prompt, candidate text, cheap validation
feedback, generation metrics, and uncertainty.  The hidden channel is an
opaque terminal evaluator.  It is invoked exactly once, after the controller
has irreversibly selected and finished a terminal node, so hidden evidence
cannot affect routing, prompts, or selection.

The runner intentionally knows nothing about MLX or any other inference
backend.  Adapters must report their own :class:`GenerationMetrics` and obey
the token/deadline allocations included in :class:`PublicGenerationFeedback`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Callable, Protocol

from experimental.effort.humaneval import (
    PublicHumanEvalCase,
    PublicValidationResult,
)
from experimental.markov_effort_controller import (
    CandidateEvidence,
    ControllerAction,
    ControllerState,
    EffortTier,
    GenerationMetrics,
    MarkovTreeEffortController,
    ValidationOutcome,
)


@dataclass(frozen=True)
class PublicGenerationFeedback:
    """Controller-visible context passed to a generation backend.

    ``parent_completion`` and validator fields are absent on the direct path.
    They contain only evidence already emitted on the public channel for extra
    generations; hidden tests and hidden scores have no representation here.
    """

    action: ControllerAction
    parent_node_id: int | None
    parent_completion: str | None
    validator_status: str | None
    validator_feedback: str
    max_output_tokens: int
    max_additional_e2e_seconds: float | None
    seed: int

    def __post_init__(self) -> None:
        if not self.action.generates_candidate:
            raise ValueError("generation feedback requires a generation action")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.max_additional_e2e_seconds is not None and (
            self.max_additional_e2e_seconds <= 0.0 or not math.isfinite(self.max_additional_e2e_seconds)
        ):
            raise ValueError("generation deadline must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.validator_feedback, str):
            raise TypeError("validator_feedback must be a string")
        if self.parent_node_id is None:
            if self.parent_completion is not None or self.validator_status is not None:
                raise ValueError("direct feedback cannot contain parent evidence")
        elif self.parent_completion is None or self.validator_status is None:
            raise ValueError("child feedback requires complete parent evidence")


@dataclass(frozen=True)
class GeneratedCandidate:
    """One backend result before any public or hidden evaluation."""

    completion: str
    metrics: GenerationMetrics
    raw_uncertainty: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.completion, str):
            raise TypeError("candidate completion must be a string")
        if not isinstance(self.metrics, GenerationMetrics):
            raise TypeError("candidate metrics must be GenerationMetrics")
        if self.raw_uncertainty is not None and (
            not isinstance(self.raw_uncertainty, (int, float))
            or isinstance(self.raw_uncertainty, bool)
            or not math.isfinite(float(self.raw_uncertainty))
            or not 0.0 <= self.raw_uncertainty <= 1.0
        ):
            raise ValueError("raw_uncertainty must be finite and in [0, 1]")


@dataclass(frozen=True)
class HiddenEvaluationResult:
    """Terminal-only result returned by an experiment's hidden evaluator."""

    score: float
    passed: bool
    status: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or not math.isfinite(float(self.score))
            or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("hidden score must be finite and in [0, 1]")
        if type(self.passed) is not bool:
            raise TypeError("hidden passed flag must be a bool")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("hidden status must be a non-empty string")
        if self.elapsed_seconds < 0.0 or not math.isfinite(self.elapsed_seconds):
            raise ValueError("hidden evaluation time must be finite and non-negative")


class CandidateGenerator(Protocol):
    """Backend adapter that can see only a public case and public feedback."""

    def __call__(
        self,
        case: PublicHumanEvalCase,
        feedback: PublicGenerationFeedback,
        /,
    ) -> GeneratedCandidate: ...


class PublicValidator(Protocol):
    """Cheap validator that has no access to hidden tests."""

    def __call__(
        self,
        case: PublicHumanEvalCase,
        completion: str,
        /,
    ) -> PublicValidationResult: ...


class HiddenEvaluator(Protocol):
    """Opaque scorer called once for the already-selected terminal output.

    Implementations receive only the public case and completion through this
    interface.  They may close over a corresponding full case internally, but
    no hidden field can flow back through the call signature.
    """

    def __call__(
        self,
        case: PublicHumanEvalCase,
        completion: str,
        /,
    ) -> HiddenEvaluationResult: ...


class UncertaintyCalibrator(Protocol):
    """Map one raw backend signal to a calibrated probability of error."""

    def __call__(self, raw_uncertainty: float, /) -> float: ...


@dataclass(frozen=True)
class CandidateTrace:
    """Auditable public record for one node in the generated candidate tree."""

    node_id: int
    parent_id: int | None
    action: ControllerAction
    completion: str
    generation_feedback: PublicGenerationFeedback
    generation_metrics: GenerationMetrics
    raw_uncertainty: float | None
    calibrated_uncertainty: float | None
    public_validation: PublicValidationResult
    routing_validation: PublicValidationResult
    controller_seconds: float
    allocated_output_tokens: int
    allocated_e2e_seconds: float | None
    deadline_exceeded: bool


@dataclass(frozen=True)
class MarkovEffortRun:
    """Terminal output of a complete two-channel effort run."""

    tier: EffortTier
    public_case: PublicHumanEvalCase
    controller_state: ControllerState
    tree: tuple[CandidateTrace, ...]
    terminal_output: str
    hidden_terminal_score: float
    hidden_evaluation: HiddenEvaluationResult
    controller_seconds: float
    deadline_overshoot_node_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.controller_state.terminal:
            raise ValueError("an effort run requires a terminal controller state")
        if not self.tree:
            raise ValueError("an effort run requires at least one candidate")
        if self.controller_state.selected_node_id not in {node.node_id for node in self.tree}:
            raise ValueError("selected node is missing from the runner tree")
        if self.hidden_terminal_score != self.hidden_evaluation.score:
            raise ValueError("terminal score does not match hidden evaluation")


class OutputTokenAllocationExceeded(RuntimeError):
    """Raised before validation/scoring when a backend exceeds its allocation."""

    def __init__(self, *, node_id: int, allowed: int, reported: int) -> None:
        self.node_id = node_id
        self.allowed = allowed
        self.reported = reported
        super().__init__(
            f"candidate {node_id} exceeded output-token allocation: reported {reported}, allowed {allowed}"
        )


def _elapsed(clock: Callable[[], float], started: float) -> float:
    elapsed = clock() - started
    if elapsed < 0.0 or not math.isfinite(elapsed):
        raise RuntimeError("runner clock must be finite and monotonic")
    return elapsed


def _generation_feedback(
    decision,
    traces: tuple[CandidateTrace, ...],
) -> PublicGenerationFeedback:
    if decision.max_output_tokens is None or decision.seed is None or decision.node_id is None:
        raise RuntimeError("controller emitted an incomplete generation decision")
    parent = (
        next((node for node in traces if node.node_id == decision.parent_id), None)
        if decision.parent_id is not None
        else None
    )
    if decision.parent_id is not None and parent is None:
        raise RuntimeError("controller selected a parent absent from the public tree")
    return PublicGenerationFeedback(
        action=decision.action,
        parent_node_id=decision.parent_id,
        parent_completion=parent.completion if parent is not None else None,
        validator_status=(parent.routing_validation.status if parent is not None else None),
        validator_feedback=(parent.routing_validation.feedback if parent is not None else ""),
        max_output_tokens=decision.max_output_tokens,
        max_additional_e2e_seconds=decision.max_additional_e2e_seconds,
        seed=decision.seed,
    )


def _deadline_validation(
    validation: PublicValidationResult,
    *,
    e2e_seconds: float,
    allocation: float | None,
) -> PublicValidationResult:
    if allocation is None or e2e_seconds <= allocation:
        return validation
    return replace(
        validation,
        outcome=ValidationOutcome.FAIL,
        status="deadline_exceeded",
        feedback="candidate_e2e_deadline_exceeded",
    )


def _calibrated_uncertainty(
    raw_uncertainty: float | None,
    calibrator: UncertaintyCalibrator | None,
) -> float | None:
    if raw_uncertainty is None:
        return None
    if calibrator is None:
        raise ValueError("calibrate_uncertainty is required when a backend emits raw uncertainty")
    calibrated = calibrator(raw_uncertainty)
    if (
        not isinstance(calibrated, (int, float))
        or isinstance(calibrated, bool)
        or not math.isfinite(float(calibrated))
        or not 0.0 <= calibrated <= 1.0
    ):
        raise ValueError("calibrated uncertainty must be finite and in [0, 1]")
    return float(calibrated)


def run_markov_effort(
    *,
    case: PublicHumanEvalCase,
    controller: MarkovTreeEffortController,
    generator: CandidateGenerator,
    public_validator: PublicValidator,
    hidden_evaluator: HiddenEvaluator,
    calibrate_uncertainty: UncertaintyCalibrator | None = None,
    context_bucket: str = "coding",
    clock: Callable[[], float] = time.perf_counter,
) -> MarkovEffortRun:
    """Drive one controller trace, then score its terminal output exactly once.

    Hidden evaluation occurs only after :meth:`MarkovTreeEffortController.finish`.
    Any exception or output-token allocation violation before that point aborts
    the run without invoking the hidden evaluator.  A synchronous generic
    runner cannot forcibly preempt an arbitrary backend at its deadline; it
    therefore passes the deadline to the adapter, converts a post-hoc overshoot
    into a public routing failure, and retains both raw and routing validation
    records for audit.
    """

    if not isinstance(case, PublicHumanEvalCase):
        raise TypeError("runner accepts only PublicHumanEvalCase")
    state = controller.initial_state(case.task_id, context_bucket=context_bucket)
    traces: tuple[CandidateTrace, ...] = ()
    total_controller_seconds = 0.0

    while True:
        decision_started = clock()
        decision = controller.decide(state)
        decision_seconds = _elapsed(clock, decision_started)
        total_controller_seconds += decision_seconds

        if not decision.action.generates_candidate:
            finish_started = clock()
            state = controller.finish(state, decision)
            finish_seconds = _elapsed(clock, finish_started)
            total_controller_seconds += finish_seconds
            break

        feedback = _generation_feedback(decision, traces)
        generated = generator(case, feedback)
        if not isinstance(generated, GeneratedCandidate):
            raise TypeError("generator must return GeneratedCandidate")
        if generated.metrics.output_tokens > feedback.max_output_tokens:
            raise OutputTokenAllocationExceeded(
                node_id=decision.node_id,
                allowed=feedback.max_output_tokens,
                reported=generated.metrics.output_tokens,
            )
        calibrated_uncertainty = _calibrated_uncertainty(
            generated.raw_uncertainty,
            calibrate_uncertainty,
        )

        raw_validation = public_validator(case, generated.completion)
        if not isinstance(raw_validation, PublicValidationResult):
            raise TypeError("public validator must return PublicValidationResult")

        preliminary_evidence = CandidateEvidence(
            metrics=generated.metrics,
            validator=raw_validation.outcome,
            evaluation_score=None,
            uncertainty=calibrated_uncertainty,
            validation_seconds=raw_validation.elapsed_seconds,
            controller_seconds=decision_seconds,
        )
        observe_started = clock()
        observed_state = controller.observe(state, decision, preliminary_evidence)
        observe_seconds = _elapsed(clock, observe_started)
        total_controller_seconds += observe_seconds
        candidate_controller_seconds = decision_seconds + observe_seconds

        reported_e2e_seconds = (
            generated.metrics.total_seconds + raw_validation.elapsed_seconds + candidate_controller_seconds
        )
        routing_validation = _deadline_validation(
            raw_validation,
            e2e_seconds=reported_e2e_seconds,
            allocation=feedback.max_additional_e2e_seconds,
        )
        final_evidence = replace(
            preliminary_evidence,
            validator=routing_validation.outcome,
            controller_seconds=candidate_controller_seconds,
        )
        final_node = replace(observed_state.nodes[-1], evidence=final_evidence)
        state = replace(
            observed_state,
            nodes=(*observed_state.nodes[:-1], final_node),
        )
        trace = CandidateTrace(
            node_id=final_node.node_id,
            parent_id=final_node.parent_id,
            action=final_node.action,
            completion=generated.completion,
            generation_feedback=feedback,
            generation_metrics=generated.metrics,
            raw_uncertainty=generated.raw_uncertainty,
            calibrated_uncertainty=calibrated_uncertainty,
            public_validation=raw_validation,
            routing_validation=routing_validation,
            controller_seconds=candidate_controller_seconds,
            allocated_output_tokens=feedback.max_output_tokens,
            allocated_e2e_seconds=feedback.max_additional_e2e_seconds,
            deadline_exceeded=final_node.deadline_exceeded,
        )
        traces = (*traces, trace)

    selected_id = state.selected_node_id
    selected = next(node for node in traces if node.node_id == selected_id)

    # This is the sole hidden-channel call.  It is deliberately below finish
    # and receives no tree/controller object through which it could alter the
    # already-terminal selection.
    hidden_evaluation = hidden_evaluator(case, selected.completion)
    if not isinstance(hidden_evaluation, HiddenEvaluationResult):
        raise TypeError("hidden evaluator must return HiddenEvaluationResult")

    return MarkovEffortRun(
        tier=controller.tier,
        public_case=case,
        controller_state=state,
        tree=traces,
        terminal_output=selected.completion,
        hidden_terminal_score=hidden_evaluation.score,
        hidden_evaluation=hidden_evaluation,
        controller_seconds=total_controller_seconds,
        deadline_overshoot_node_ids=tuple(node.node_id for node in traces if node.deadline_exceeded),
    )
