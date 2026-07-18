from __future__ import annotations

from dataclasses import replace

import pytest

from experimental.effort.humaneval import (
    HumanEvalCase,
    PublicHumanEvalCase,
    PublicValidationResult,
)
from experimental.effort.markov_runner import (
    GeneratedCandidate,
    HiddenEvaluationResult,
    OutputTokenAllocationExceeded,
    PublicGenerationFeedback,
    run_markov_effort,
)
from experimental.markov_effort_controller import (
    BootstrapMetadata,
    CalibrationIdentity,
    ControllerAction,
    ControllerState,
    EffortProfile,
    EffortTier,
    FrozenTransitionModel,
    GenerationMetrics,
    MarkovTreeEffortController,
    TransitionEstimate,
    Trigger,
    ValidationOutcome,
)


IDENTITY = CalibrationIdentity(
    model="runner-test-model",
    config="runner-test-config",
    prompt="runner-test-prompt",
    sampler="greedy",
    corpus="runner-test-corpus",
    split="calibration",
    backend="backend-independent-test",
)
PUBLIC_CASE = PublicHumanEvalCase(
    task_id="HumanEval/0",
    prompt="def add(a, b):\n",
    entry_point="add",
)


def _metrics(
    *,
    output_tokens: int = 10,
    prefill_seconds: float = 0.05,
    decode_seconds: float = 0.05,
) -> GenerationMetrics:
    return GenerationMetrics(
        prompt_tokens=20,
        output_tokens=output_tokens,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
    )


def _validation(
    outcome: ValidationOutcome,
    *,
    status: str,
    feedback: str,
    elapsed_seconds: float = 0.001,
) -> PublicValidationResult:
    return PublicValidationResult(
        outcome=outcome,
        status=status,
        feedback=feedback,
        elapsed_seconds=elapsed_seconds,
        source_sha256="0" * 64,
    )


def _repair_estimate() -> TransitionEstimate:
    return TransitionEstimate(
        context_bucket="coding",
        trigger=Trigger.VALIDATOR_FAILURE,
        depth=1,
        action=ControllerAction.GENERATE_REPAIR,
        conservative_success_lcb=0.8,
        extra_output_tokens_ucb=10.0,
        extra_e2e_latency_ratio_ucb=0.5,
        bootstrap=BootstrapMetadata(
            task_cluster_count=32,
            resamples=1_000,
            confidence_level=0.95,
            method="task-cluster-percentile",
            seed=7,
        ),
    )


def _uncertainty_estimate() -> TransitionEstimate:
    return replace(
        _repair_estimate(),
        trigger=Trigger.CALIBRATED_UNCERTAINTY,
        action=ControllerAction.GENERATE_ALTERNATIVE,
    )


def _identity_calibrator(raw_uncertainty: float) -> float:
    return raw_uncertainty


class _FinishTrackingController(MarkovTreeEffortController):
    finished = False

    def finish(self, state: ControllerState, decision) -> ControllerState:
        result = super().finish(state, decision)
        self.finished = True
        return result


def _controller(
    tier: EffortTier | str,
    *,
    profile: EffortProfile | None = None,
    tracking: bool = False,
) -> MarkovTreeEffortController:
    controller_type = _FinishTrackingController if tracking else MarkovTreeEffortController
    return controller_type(
        tier=tier,
        transition_model=FrozenTransitionModel(
            IDENTITY,
            (_repair_estimate(), _uncertainty_estimate()),
        ),
        calibration_identity=IDENTITY,
        initial_max_output_tokens=64,
        profile=profile,
    )


class _PublicGenerator:
    def __init__(self) -> None:
        self.feedback: list[PublicGenerationFeedback] = []

    def __call__(
        self,
        case: PublicHumanEvalCase,
        feedback: PublicGenerationFeedback,
    ) -> GeneratedCandidate:
        assert type(case) is PublicHumanEvalCase
        assert not hasattr(case, "test")
        assert not hasattr(feedback, "hidden_score")
        self.feedback.append(feedback)
        if feedback.action is ControllerAction.GENERATE_DIRECT:
            assert feedback.parent_completion is None
            return GeneratedCandidate("direct", _metrics(), raw_uncertainty=0.9)
        assert feedback.parent_completion == "direct"
        assert feedback.validator_status == "public_failure"
        assert feedback.validator_feedback == "visible_feedback"
        return GeneratedCandidate("repair", _metrics(), raw_uncertainty=0.1)


class _PublicValidator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(
        self,
        case: PublicHumanEvalCase,
        completion: str,
    ) -> PublicValidationResult:
        assert type(case) is PublicHumanEvalCase
        self.calls.append(completion)
        if completion == "direct":
            return _validation(
                ValidationOutcome.FAIL,
                status="public_failure",
                feedback="visible_feedback",
            )
        return _validation(
            ValidationOutcome.PASS,
            status="public_pass",
            feedback="visible_pass",
        )


class _TerminalEvaluator:
    def __init__(self, controller: MarkovTreeEffortController, score: float = 1.0) -> None:
        self.controller = controller
        self.score = score
        self.calls: list[str] = []

    def __call__(
        self,
        case: PublicHumanEvalCase,
        completion: str,
    ) -> HiddenEvaluationResult:
        # This raises if the runner ever evaluates before terminal finish.
        if not getattr(self.controller, "finished", False):
            raise AssertionError("hidden evaluator called before terminal finish")
        assert type(case) is PublicHumanEvalCase
        self.calls.append(completion)
        return HiddenEvaluationResult(
            score=self.score,
            passed=self.score == 1.0,
            status="hidden_result",
            elapsed_seconds=0.01,
        )


@pytest.mark.parametrize("tier", tuple(EffortTier))
def test_all_five_tiers_drive_terminal_two_channel_runs(tier: EffortTier) -> None:
    controller = _controller(tier, tracking=True)
    generator = _PublicGenerator()
    validator = _PublicValidator()
    evaluator = _TerminalEvaluator(controller)

    result = run_markov_effort(
        case=PUBLIC_CASE,
        controller=controller,
        generator=generator,
        public_validator=validator,
        hidden_evaluator=evaluator,
        calibrate_uncertainty=_identity_calibrator,
    )

    expected_outputs = ["direct"] if tier is EffortTier.LOW else ["direct", "repair"]
    assert [node.completion for node in result.tree] == expected_outputs
    assert validator.calls == expected_outputs
    assert evaluator.calls == [expected_outputs[-1]]
    assert result.tier is tier
    assert result.terminal_output == expected_outputs[-1]
    assert result.hidden_terminal_score == 1.0
    assert result.controller_state.terminal is True
    assert result.controller_state.selected_node_id == len(expected_outputs) - 1
    assert result.controller_seconds >= 0.0
    assert all(node.evidence.evaluation_score is None for node in result.controller_state.nodes)


def test_tree_captures_public_evidence_metrics_uncertainty_and_parent_feedback() -> None:
    controller = _controller(EffortTier.HIGH, tracking=True)
    generator = _PublicGenerator()
    result = run_markov_effort(
        case=PUBLIC_CASE,
        controller=controller,
        generator=generator,
        public_validator=_PublicValidator(),
        hidden_evaluator=_TerminalEvaluator(controller),
        calibrate_uncertainty=_identity_calibrator,
    )

    direct, repair = result.tree
    assert (direct.node_id, direct.parent_id) == (0, None)
    assert (repair.node_id, repair.parent_id) == (1, 0)
    assert direct.generation_metrics == _metrics()
    assert direct.raw_uncertainty == 0.9
    assert direct.calibrated_uncertainty == 0.9
    assert direct.public_validation.status == "public_failure"
    assert direct.routing_validation == direct.public_validation
    assert repair.generation_feedback.parent_completion == "direct"
    assert repair.allocated_output_tokens > 0
    assert repair.allocated_e2e_seconds is not None
    assert repair.controller_seconds >= 0.0


def test_routing_uses_calibrated_probability_and_calibrator_sees_no_label() -> None:
    calibrator_calls: list[float] = []

    def execute(calibrated_probability: float):
        controller = _controller(EffortTier.HIGH, tracking=True)

        def generator(_case, feedback):
            if feedback.action is ControllerAction.GENERATE_DIRECT:
                return GeneratedCandidate("direct", _metrics(), raw_uncertainty=0.95)
            return GeneratedCandidate("alternative", _metrics(), raw_uncertainty=None)

        def validator(_case, _completion):
            return _validation(
                ValidationOutcome.UNKNOWN,
                status="parseable",
                feedback="parseable",
            )

        def calibrator(raw_uncertainty):
            # The one-argument signature makes task labels and hidden results
            # unavailable to calibration on the request path.
            calibrator_calls.append(raw_uncertainty)
            return calibrated_probability

        return run_markov_effort(
            case=PUBLIC_CASE,
            controller=controller,
            generator=generator,
            public_validator=validator,
            hidden_evaluator=_TerminalEvaluator(controller),
            calibrate_uncertainty=calibrator,
        )

    below_threshold = execute(0.1)
    above_threshold = execute(0.9)

    assert [node.completion for node in below_threshold.tree] == ["direct"]
    assert below_threshold.tree[0].raw_uncertainty == 0.95
    assert below_threshold.tree[0].calibrated_uncertainty == 0.1
    assert [node.completion for node in above_threshold.tree] == [
        "direct",
        "alternative",
    ]
    assert calibrator_calls == [0.95, 0.95]
    assert below_threshold.controller_state.nodes[0].evidence.uncertainty == 0.1
    assert above_threshold.controller_state.nodes[0].evidence.uncertainty == 0.9


def test_raw_uncertainty_requires_an_explicit_calibrator() -> None:
    controller = _controller(EffortTier.LOW, tracking=True)

    with pytest.raises(ValueError, match="calibrate_uncertainty is required"):
        run_markov_effort(
            case=PUBLIC_CASE,
            controller=controller,
            generator=_PublicGenerator(),
            public_validator=_PublicValidator(),
            hidden_evaluator=_TerminalEvaluator(controller),
        )

    assert controller.finished is False


def test_hidden_score_cannot_change_routing_or_terminal_selection() -> None:
    def execute(score: float):
        controller = _controller(EffortTier.MEDIUM, tracking=True)
        return run_markov_effort(
            case=PUBLIC_CASE,
            controller=controller,
            generator=_PublicGenerator(),
            public_validator=_PublicValidator(),
            hidden_evaluator=_TerminalEvaluator(controller, score),
            calibrate_uncertainty=_identity_calibrator,
        )

    low_score = execute(0.0)
    high_score = execute(1.0)

    assert low_score.terminal_output == high_score.terminal_output == "repair"
    assert [node.action for node in low_score.tree] == [node.action for node in high_score.tree]
    assert low_score.controller_state.selected_node_id == (high_score.controller_state.selected_node_id)
    assert low_score.hidden_terminal_score == 0.0
    assert high_score.hidden_terminal_score == 1.0


def test_output_token_overshoot_aborts_before_validation_and_hidden_evaluation() -> None:
    controller = _controller(EffortTier.LOW, tracking=True)
    validator_calls = 0
    evaluator_calls = 0

    def generator(_case, _feedback):
        return GeneratedCandidate("oversized", _metrics(output_tokens=65))

    def validator(_case, _completion):
        nonlocal validator_calls
        validator_calls += 1
        return _validation(
            ValidationOutcome.PASS,
            status="pass",
            feedback="pass",
        )

    def evaluator(_case, _completion):
        nonlocal evaluator_calls
        evaluator_calls += 1
        raise AssertionError("hidden evaluator must not run after allocation failure")

    with pytest.raises(OutputTokenAllocationExceeded) as captured:
        run_markov_effort(
            case=PUBLIC_CASE,
            controller=controller,
            generator=generator,
            public_validator=validator,
            hidden_evaluator=evaluator,
            calibrate_uncertainty=_identity_calibrator,
        )

    assert captured.value.allowed == 64
    assert captured.value.reported == 65
    assert validator_calls == 0
    assert evaluator_calls == 0
    assert controller.finished is False


def test_posthoc_deadline_overshoot_fails_closed_and_remains_visible() -> None:
    profile = EffortProfile(
        max_candidates=2,
        max_extra_output_tokens=64,
        max_output_tokens_per_candidate=64,
        max_latency_ratio=1.75,
        uncertainty_threshold=1.0,
    )
    controller = _controller(EffortTier.MEDIUM, profile=profile, tracking=True)

    def generator(_case, feedback):
        if feedback.action is ControllerAction.GENERATE_DIRECT:
            return GeneratedCandidate("direct", _metrics(), raw_uncertainty=0.9)
        return GeneratedCandidate(
            "slow-repair",
            _metrics(prefill_seconds=0.15, decode_seconds=0.15),
            raw_uncertainty=0.1,
        )

    def validator(_case, completion):
        if completion == "direct":
            return _validation(
                ValidationOutcome.FAIL,
                status="public_failure",
                feedback="visible_feedback",
            )
        return _validation(
            ValidationOutcome.PASS,
            status="raw_public_pass",
            feedback="raw_visible_pass",
        )

    evaluator = _TerminalEvaluator(controller)
    result = run_markov_effort(
        case=PUBLIC_CASE,
        controller=controller,
        generator=generator,
        public_validator=validator,
        hidden_evaluator=evaluator,
        calibrate_uncertainty=_identity_calibrator,
    )

    slow = result.tree[1]
    assert slow.public_validation.outcome is ValidationOutcome.PASS
    assert slow.public_validation.status == "raw_public_pass"
    assert slow.routing_validation.outcome is ValidationOutcome.FAIL
    assert slow.routing_validation.status == "deadline_exceeded"
    assert slow.deadline_exceeded is True
    assert result.deadline_overshoot_node_ids == (1,)
    assert result.controller_state.terminal_action is ControllerAction.STOP
    assert evaluator.calls == [result.terminal_output]


def test_full_humaneval_case_is_rejected_before_any_callback() -> None:
    full_case = HumanEvalCase(
        task_id=PUBLIC_CASE.task_id,
        prompt=PUBLIC_CASE.prompt,
        test="def check(candidate): assert candidate(1, 1) == 2",
        entry_point=PUBLIC_CASE.entry_point,
    )
    callback_calls = 0

    def forbidden(*_args):
        nonlocal callback_calls
        callback_calls += 1
        raise AssertionError("callback must not run for a hidden-bearing case")

    with pytest.raises(TypeError, match="only PublicHumanEvalCase"):
        run_markov_effort(
            case=full_case,  # type: ignore[arg-type]
            controller=_controller(EffortTier.LOW),
            generator=forbidden,
            public_validator=forbidden,
            hidden_evaluator=forbidden,
        )
    assert callback_calls == 0


@pytest.mark.parametrize(
    "uncertainty",
    [-0.01, 1.01, float("inf"), float("nan"), True],
)
def test_invalid_raw_uncertainty_fails_closed(uncertainty) -> None:
    with pytest.raises(ValueError, match="raw_uncertainty"):
        GeneratedCandidate("candidate", _metrics(), raw_uncertainty=uncertainty)


def test_hidden_evaluation_result_is_validated() -> None:
    with pytest.raises(ValueError, match="hidden score"):
        HiddenEvaluationResult(
            score=1.1,
            passed=True,
            status="invalid",
            elapsed_seconds=0.1,
        )

    with pytest.raises(ValueError, match="evaluation time"):
        replace(
            HiddenEvaluationResult(1.0, True, "valid", 0.1),
            elapsed_seconds=-0.1,
        )
