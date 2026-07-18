from __future__ import annotations

from dataclasses import replace

import pytest

from experimental.markov_effort_controller import (
    BootstrapMetadata,
    CalibrationIdentity,
    CandidateEvidence,
    ControllerAction,
    EFFORT_PROFILES,
    EXTRA_ACTIONS,
    EffortProfile,
    EffortTier,
    FrozenTransitionModel,
    GenerationMetrics,
    MarkovTreeEffortController,
    TransitionEstimate,
    Trigger,
    ValidationOutcome,
    trace_metrics,
)


CALIBRATION_IDENTITY = CalibrationIdentity(
    model="qwen-test",
    config="config-sha256:111",
    prompt="prompt-sha256:222",
    sampler="greedy-temp0",
    corpus="coding-suite-v1",
    split="calibration",
    backend="mlx-lm-test",
)


def estimate(
    action: ControllerAction,
    *,
    depth: int = 1,
    trigger: Trigger = Trigger.VALIDATOR_FAILURE,
    context: str = "global",
    task_clusters: int = 20,
    success_lcb: float = 0.50,
    tokens: float = 24.0,
    latency_ratio: float = 0.4,
    quality_gain_lcb: float | None = None,
) -> TransitionEstimate:
    return TransitionEstimate(
        context_bucket=context,
        trigger=trigger,
        depth=depth,
        action=action,
        conservative_success_lcb=success_lcb,
        extra_output_tokens_ucb=tokens,
        extra_e2e_latency_ratio_ucb=latency_ratio,
        bootstrap=BootstrapMetadata(
            task_cluster_count=task_clusters,
            resamples=10_000,
            confidence_level=0.95,
            method="task-cluster-percentile",
            seed=17,
        ),
        conservative_quality_gain_lcb=quality_gain_lcb,
    )


def evidence(
    outcome: ValidationOutcome,
    *,
    prompt_tokens: int = 100,
    output_tokens: int = 20,
    prefill_seconds: float = 0.1,
    decode_seconds: float = 0.2,
    other_seconds: float = 0.0,
    evaluation: float | None = None,
    uncertainty: float | None = None,
    validation_seconds: float = 0.0,
    controller_seconds: float = 0.0,
) -> CandidateEvidence:
    return CandidateEvidence(
        metrics=GenerationMetrics(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            other_seconds=other_seconds,
        ),
        validator=outcome,
        evaluation_score=evaluation,
        uncertainty=uncertainty,
        validation_seconds=validation_seconds,
        controller_seconds=controller_seconds,
    )


def controller(
    *rows: TransitionEstimate,
    tier: str = "ultra",
    profile: EffortProfile | None = None,
) -> MarkovTreeEffortController:
    return MarkovTreeEffortController(
        tier=tier,
        transition_model=FrozenTransitionModel(CALIBRATION_IDENTITY, rows),
        calibration_identity=CALIBRATION_IDENTITY,
        initial_max_output_tokens=64,
        profile=profile,
    )


def add_direct(ctrl, state, direct_evidence):
    decision = ctrl.decide(state)
    assert decision.action is ControllerAction.GENERATE_DIRECT
    return ctrl.observe(state, decision, direct_evidence)


def test_public_effort_tiers_are_exact_and_budgets_are_monotone() -> None:
    assert [tier.value for tier in EffortTier] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "ultra",
    ]
    profiles = [EFFORT_PROFILES[tier] for tier in EffortTier]
    assert [profile.max_candidates for profile in profiles] == sorted(
        profile.max_candidates for profile in profiles
    )
    assert [profile.max_extra_output_tokens for profile in profiles] == sorted(
        profile.max_extra_output_tokens for profile in profiles
    )
    assert [profile.max_latency_ratio for profile in profiles] == sorted(
        profile.max_latency_ratio for profile in profiles
    )
    assert [profile.uncertainty_threshold for profile in profiles] == sorted(
        (profile.uncertainty_threshold for profile in profiles), reverse=True
    )


@pytest.mark.parametrize("tier", [tier.value for tier in EffortTier])
def test_every_effort_tier_preserves_the_direct_fast_path(tier: str) -> None:
    ctrl = controller(tier=tier)
    state = ctrl.initial_state(f"fast-path-{tier}")
    state = add_direct(ctrl, state, evidence(ValidationOutcome.PASS))
    terminal = ctrl.decide(state)
    assert terminal.action is ControllerAction.ACCEPT
    assert terminal.node_id == 0
    assert terminal.reason == "validator_pass"


def test_direct_fast_path_is_stable_and_needs_no_calibration() -> None:
    first = controller()
    second = controller()
    state_a = first.initial_state("request-7", context_bucket="coding")
    state_b = second.initial_state("request-7", context_bucket="coding")
    assert first.decide(state_a) == second.decide(state_b)

    state_a = add_direct(
        first,
        state_a,
        evidence(ValidationOutcome.PASS, evaluation=1.0),
    )
    terminal = first.decide(state_a)
    assert terminal.action is ControllerAction.ACCEPT
    assert terminal.reason == "validator_pass"
    finished = first.finish(state_a, terminal)
    assert finished.selected_node_id == 0
    assert finished.terminal is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other-model"),
        ("config", "config-sha256:other"),
        ("prompt", "prompt-sha256:other"),
        ("sampler", "temperature-0.2"),
        ("corpus", "other-corpus"),
        ("split", "held-out"),
        ("backend", "other-backend"),
    ],
)
def test_controller_rejects_any_calibration_identity_mismatch(field, value) -> None:
    mismatched = replace(CALIBRATION_IDENTITY, **{field: value})
    model = FrozenTransitionModel(CALIBRATION_IDENTITY)
    with pytest.raises(ValueError, match="identity mismatch"):
        MarkovTreeEffortController(
            tier="high",
            transition_model=model,
            calibration_identity=mismatched,
            initial_max_output_tokens=64,
        )


def test_failed_direct_uses_best_conservative_gain_per_latency() -> None:
    repair = estimate(
        ControllerAction.GENERATE_REPAIR,
        success_lcb=0.70,
        latency_ratio=0.8,
    )
    alternative = estimate(
        ControllerAction.GENERATE_ALTERNATIVE,
        success_lcb=0.55,
        latency_ratio=0.3,
    )
    ctrl = controller(repair, alternative)
    state = add_direct(
        ctrl,
        ctrl.initial_state("quality-route"),
        evidence(ValidationOutcome.FAIL, evaluation=0.0),
    )
    decision = ctrl.decide(state)
    assert decision.action is ControllerAction.GENERATE_ALTERNATIVE
    assert decision.parent_id == 0
    assert decision.predicted_success_lcb is not None


def test_tree_contains_repair_child_and_alternative_sibling() -> None:
    ctrl = controller(
        estimate(ControllerAction.GENERATE_REPAIR, depth=1),
        estimate(ControllerAction.GENERATE_ALTERNATIVE, depth=2),
    )
    state = add_direct(
        ctrl,
        ctrl.initial_state("tree"),
        evidence(ValidationOutcome.FAIL, evaluation=0.1),
    )
    repair_decision = ctrl.decide(state)
    assert repair_decision.action is ControllerAction.GENERATE_REPAIR
    state = ctrl.observe(
        state,
        repair_decision,
        evidence(ValidationOutcome.FAIL, evaluation=0.4),
    )
    alternative_decision = ctrl.decide(state)
    assert alternative_decision.action is ControllerAction.GENERATE_ALTERNATIVE
    state = ctrl.observe(
        state,
        alternative_decision,
        evidence(ValidationOutcome.PASS, evaluation=1.0),
    )

    assert [node.parent_id for node in state.nodes] == [None, 0, 0]
    terminal = ctrl.decide(state)
    finished = ctrl.finish(state, terminal)
    assert terminal.action is ControllerAction.ACCEPT
    assert finished.selected_node_id == 2


def test_evaluation_score_is_never_used_for_selection_or_decisions() -> None:
    profile = EffortProfile(
        max_candidates=2,
        max_extra_output_tokens=64,
        max_output_tokens_per_candidate=64,
        max_latency_ratio=2.0,
        uncertainty_threshold=1.0,
    )
    ctrl = controller(
        estimate(ControllerAction.GENERATE_REPAIR),
        profile=profile,
    )
    state = add_direct(
        ctrl,
        ctrl.initial_state("evaluation-segregation"),
        evidence(
            ValidationOutcome.FAIL,
            evaluation=0.0,
            uncertainty=0.1,
        ),
    )
    repair = ctrl.decide(state)
    state = ctrl.observe(
        state,
        repair,
        evidence(
            ValidationOutcome.FAIL,
            evaluation=1.0,
            uncertainty=0.9,
        ),
    )

    terminal = ctrl.decide(state)
    assert terminal.action is ControllerAction.STOP
    assert terminal.node_id == 0


def test_uncertainty_requires_both_threshold_and_frozen_evidence() -> None:
    row = estimate(
        ControllerAction.GENERATE_ALTERNATIVE,
        trigger=Trigger.CALIBRATED_UNCERTAINTY,
    )
    ctrl = controller(row, tier="high")
    low = add_direct(
        ctrl,
        ctrl.initial_state("certain"),
        evidence(ValidationOutcome.UNKNOWN, uncertainty=0.79),
    )
    assert ctrl.decide(low).action is ControllerAction.ACCEPT

    high = add_direct(
        ctrl,
        ctrl.initial_state("uncertain"),
        evidence(ValidationOutcome.UNKNOWN, uncertainty=0.80),
    )
    assert ctrl.decide(high).action is ControllerAction.GENERATE_ALTERNATIVE

    uncalibrated = controller(tier="high")
    state = add_direct(
        uncalibrated,
        uncalibrated.initial_state("no-table"),
        evidence(ValidationOutcome.UNKNOWN, uncertainty=1.0),
    )
    decision = uncalibrated.decide(state)
    assert decision.action is ControllerAction.ACCEPT
    assert decision.reason == "no_calibrated_transition_within_budget"


def test_insufficient_task_clusters_or_offline_lcb_never_spends_compute() -> None:
    ctrl = controller(
        estimate(ControllerAction.GENERATE_REPAIR, task_clusters=7, success_lcb=0.9),
        estimate(ControllerAction.GENERATE_ALTERNATIVE, success_lcb=0.01),
    )
    state = add_direct(
        ctrl,
        ctrl.initial_state("guard"),
        evidence(ValidationOutcome.FAIL),
    )
    decision = ctrl.decide(state)
    assert decision.action is ControllerAction.STOP
    assert decision.reason == "no_calibrated_transition_within_budget"


def test_high_rescue_rate_cannot_hide_negative_net_quality_gain() -> None:
    ctrl = controller(
        estimate(
            ControllerAction.GENERATE_REPAIR,
            success_lcb=0.8,
            quality_gain_lcb=-0.1,
        )
    )
    state = add_direct(
        ctrl,
        ctrl.initial_state("net-quality-gate"),
        evidence(ValidationOutcome.FAIL),
    )
    decision = ctrl.decide(state)
    assert decision.action is ControllerAction.STOP
    assert decision.reason == "no_calibrated_transition_within_budget"


def test_action_choice_uses_net_quality_gain_not_rescue_rate() -> None:
    repair = estimate(
        ControllerAction.GENERATE_REPAIR,
        success_lcb=0.9,
        quality_gain_lcb=0.1,
        latency_ratio=0.5,
    )
    alternative = estimate(
        ControllerAction.GENERATE_ALTERNATIVE,
        success_lcb=0.4,
        quality_gain_lcb=0.3,
        latency_ratio=0.5,
    )
    ctrl = controller(repair, alternative)
    state = add_direct(
        ctrl,
        ctrl.initial_state("net-quality-choice"),
        evidence(ValidationOutcome.FAIL),
    )
    decision = ctrl.decide(state)
    assert decision.action is ControllerAction.GENERATE_ALTERNATIVE
    assert decision.predicted_success_lcb == pytest.approx(0.4)
    assert decision.predicted_quality_gain_lcb == pytest.approx(0.3)


def test_context_specific_transition_precedes_global_fallback() -> None:
    ctrl = controller(
        estimate(ControllerAction.GENERATE_REPAIR, context="global"),
        estimate(
            ControllerAction.GENERATE_REPAIR,
            context="coding",
            success_lcb=0.80,
        ),
    )
    coding = add_direct(
        ctrl,
        ctrl.initial_state("context", context_bucket="coding"),
        evidence(ValidationOutcome.FAIL),
    )
    decision = ctrl.decide(coding)
    assert decision.transition_source == "coding"

    math_state = add_direct(
        ctrl,
        ctrl.initial_state("fallback", context_bucket="math"),
        evidence(ValidationOutcome.FAIL),
    )
    assert ctrl.decide(math_state).transition_source == "global"


def test_frozen_transition_estimates_have_stable_public_order() -> None:
    alternative = estimate(ControllerAction.GENERATE_ALTERNATIVE, context="zeta")
    repair = estimate(ControllerAction.GENERATE_REPAIR, context="alpha")
    model = FrozenTransitionModel(CALIBRATION_IDENTITY, (alternative, repair))
    assert model.estimates == (repair, alternative)


def test_tiers_expose_distinct_action_sets() -> None:
    assert EFFORT_PROFILES[EffortTier.LOW].allowed_actions == ()
    assert EFFORT_PROFILES[EffortTier.MEDIUM].allowed_actions == (
        ControllerAction.GENERATE_REPAIR,
    )
    assert EFFORT_PROFILES[EffortTier.HIGH].allowed_actions == (
        ControllerAction.GENERATE_REPAIR,
        ControllerAction.GENERATE_ALTERNATIVE,
    )
    assert EFFORT_PROFILES[EffortTier.XHIGH].allowed_actions == EXTRA_ACTIONS
    assert EFFORT_PROFILES[EffortTier.ULTRA].allowed_actions == EXTRA_ACTIONS


def test_medium_cannot_branch_and_high_cannot_refine() -> None:
    alternative = estimate(ControllerAction.GENERATE_ALTERNATIVE)
    medium = controller(alternative, tier="medium")
    state = add_direct(
        medium,
        medium.initial_state("medium-action-envelope"),
        evidence(ValidationOutcome.FAIL),
    )
    assert medium.decide(state).action is ControllerAction.STOP

    refine = estimate(
        ControllerAction.GENERATE_REFINE,
        trigger=Trigger.CALIBRATED_UNCERTAINTY,
    )
    high = controller(refine, tier="high")
    state = add_direct(
        high,
        high.initial_state("high-action-envelope"),
        evidence(ValidationOutcome.UNKNOWN, uncertainty=0.9),
    )
    assert high.decide(state).action is ControllerAction.ACCEPT

    xhigh = controller(refine, tier="xhigh")
    state = add_direct(
        xhigh,
        xhigh.initial_state("xhigh-action-envelope"),
        evidence(ValidationOutcome.UNKNOWN, uncertainty=0.9),
    )
    assert xhigh.decide(state).action is ControllerAction.GENERATE_REFINE


def test_ultra_can_repeat_a_calibrated_action_at_a_later_depth() -> None:
    ctrl = controller(
        estimate(ControllerAction.GENERATE_REPAIR, depth=1),
        estimate(ControllerAction.GENERATE_REPAIR, depth=2),
        tier="ultra",
    )
    state = add_direct(
        ctrl,
        ctrl.initial_state("repeat-action"),
        evidence(ValidationOutcome.FAIL),
    )
    first = ctrl.decide(state)
    state = ctrl.observe(state, first, evidence(ValidationOutcome.FAIL))
    second = ctrl.decide(state)
    assert first.action is second.action is ControllerAction.GENERATE_REPAIR
    assert second.parent_id == 1


def test_token_and_latency_budgets_gate_before_generation() -> None:
    token_limited = EffortProfile(
        max_candidates=2,
        max_extra_output_tokens=16,
        max_output_tokens_per_candidate=16,
        max_latency_ratio=3.0,
        uncertainty_threshold=1.0,
    )
    ctrl = controller(
        estimate(ControllerAction.GENERATE_REPAIR, tokens=17),
        profile=token_limited,
    )
    state = add_direct(
        ctrl,
        ctrl.initial_state("tokens"),
        evidence(ValidationOutcome.FAIL),
    )
    assert ctrl.decide(state).action is ControllerAction.STOP

    latency_limited = replace(token_limited, max_extra_output_tokens=64, max_latency_ratio=1.2)
    ctrl = controller(
        estimate(
            ControllerAction.GENERATE_REPAIR,
            tokens=16,
            latency_ratio=0.21,
        ),
        profile=latency_limited,
    )
    state = add_direct(
        ctrl,
        ctrl.initial_state("latency"),
        evidence(ValidationOutcome.FAIL),
    )
    assert ctrl.decide(state).action is ControllerAction.STOP


def test_zero_e2e_baseline_stops_instead_of_issuing_extra_compute() -> None:
    ctrl = controller(estimate(ControllerAction.GENERATE_REPAIR))
    state = add_direct(
        ctrl,
        ctrl.initial_state("zero-baseline"),
        evidence(
            ValidationOutcome.FAIL,
            prompt_tokens=0,
            output_tokens=0,
            prefill_seconds=0.0,
            decode_seconds=0.0,
        ),
    )
    decision = ctrl.decide(state)
    assert decision.action is ControllerAction.STOP
    assert decision.reason == "zero_baseline_e2e_latency"


def test_generation_adapter_cannot_exceed_token_allocation() -> None:
    ctrl = controller(estimate(ControllerAction.GENERATE_REPAIR))
    state = add_direct(
        ctrl,
        ctrl.initial_state("allocation"),
        evidence(ValidationOutcome.FAIL),
    )
    decision = ctrl.decide(state)
    too_many = evidence(
        ValidationOutcome.PASS,
        output_tokens=(decision.max_output_tokens or 0) + 1,
    )
    with pytest.raises(ValueError, match="allocation"):
        ctrl.observe(state, decision, too_many)


def test_forged_or_stale_decision_is_rejected_by_replay() -> None:
    ctrl = controller(estimate(ControllerAction.GENERATE_REPAIR))
    state = ctrl.initial_state("replay")
    direct = ctrl.decide(state)
    forged = replace(direct, seed=(direct.seed or 0) + 1)
    with pytest.raises(ValueError, match="replay"):
        ctrl.observe(state, forged, evidence(ValidationOutcome.PASS))


def test_trace_reports_backend_rates_and_effective_cost_separately() -> None:
    ctrl = controller(estimate(ControllerAction.GENERATE_REPAIR))
    state = add_direct(
        ctrl,
        ctrl.initial_state("metrics"),
        evidence(
            ValidationOutcome.FAIL,
            prompt_tokens=100,
            output_tokens=20,
            prefill_seconds=0.1,
            decode_seconds=0.2,
            evaluation=0.0,
            validation_seconds=0.02,
            controller_seconds=0.005,
        ),
    )
    repair = ctrl.decide(state)
    state = ctrl.observe(
        state,
        repair,
        evidence(
            ValidationOutcome.PASS,
            prompt_tokens=200,
            output_tokens=20,
            prefill_seconds=0.2,
            decode_seconds=0.2,
            evaluation=1.0,
            validation_seconds=0.03,
            controller_seconds=0.005,
        ),
    )
    state = ctrl.finish(state, ctrl.decide(state))
    metrics = trace_metrics(state)

    assert metrics.baseline_prefill_tokens_per_second == pytest.approx(1000.0)
    assert metrics.aggregate_prefill_tokens_per_second == pytest.approx(1000.0)
    assert metrics.prefill_speed_ratio == pytest.approx(1.0)
    assert metrics.baseline_decode_tokens_per_second == pytest.approx(100.0)
    assert metrics.aggregate_decode_tokens_per_second == pytest.approx(100.0)
    assert metrics.decode_speed_ratio == pytest.approx(1.0)
    assert metrics.generation_latency_ratio == pytest.approx(0.7 / 0.3)
    assert metrics.e2e_latency_ratio == pytest.approx(0.76 / 0.325)
    assert metrics.effective_selected_tokens_per_second == pytest.approx(20 / 0.76)
    assert metrics.controller_overhead_fraction == pytest.approx(0.01 / 0.76)
    assert metrics.validation_overhead_fraction == pytest.approx(0.05 / 0.76)
    assert metrics.evaluation_delta == pytest.approx(1.0)


def test_deadline_overshoot_is_visible_instead_of_hidden() -> None:
    profile = EffortProfile(
        max_candidates=2,
        max_extra_output_tokens=64,
        max_output_tokens_per_candidate=64,
        max_latency_ratio=1.5,
        uncertainty_threshold=1.0,
    )
    ctrl = controller(
        estimate(
            ControllerAction.GENERATE_REPAIR,
            tokens=20,
            latency_ratio=0.4,
        ),
        profile=profile,
    )
    state = add_direct(
        ctrl,
        ctrl.initial_state("overshoot"),
        evidence(
            ValidationOutcome.FAIL,
            prefill_seconds=0.1,
            decode_seconds=0.1,
            validation_seconds=0.05,
            controller_seconds=0.05,
        ),
    )
    repair = ctrl.decide(state)
    assert repair.max_additional_e2e_seconds == pytest.approx(0.15)
    state = ctrl.observe(
        state,
        repair,
        evidence(
            ValidationOutcome.PASS,
            prefill_seconds=0.06,
            decode_seconds=0.06,
            validation_seconds=0.02,
            controller_seconds=0.02,
        ),
    )
    assert trace_metrics(state).deadline_violations == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"output_tokens": -1}, "token"),
        ({"prefill_seconds": -0.1}, "timing"),
    ],
)
def test_invalid_generation_metrics_fail_closed(kwargs, message) -> None:
    values = {
        "prompt_tokens": 1,
        "output_tokens": 1,
        "prefill_seconds": 0.1,
        "decode_seconds": 0.1,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        GenerationMetrics(**values)
