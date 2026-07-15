from __future__ import annotations

import pytest

from experimental.mixture import (
    DrafterObservation,
    OnlineDrafterRouter,
    RouteContext,
    RouterConfig,
)
from experimental.mixture.model import ArmCurve


def _context(index: int = 1, *, requested_tokens: int = 64) -> RouteContext:
    return RouteContext(
        request_id=f"request-{index}",
        prompt_tokens=48,
        requested_tokens=requested_tokens,
        workload="test",
    )


def _observation(
    arm: str,
    wall_seconds: float,
    *,
    output_tokens: int = 64,
    fallback: bool = False,
) -> DrafterObservation:
    rounds = 16
    return DrafterObservation(
        arm=arm,
        wall_seconds=wall_seconds,
        ttft_seconds=min(0.05, wall_seconds / 2),
        output_tokens=output_tokens,
        rounds=rounds,
        accepted_per_round=output_tokens / rounds,
        verify_seconds=wall_seconds * 0.2,
        fallback=fallback,
        parity=True,
    )


def test_acceptance_round_curve_drives_projection() -> None:
    curve = ArmCurve("draft")
    curve.update(
        DrafterObservation(
            arm="draft",
            wall_seconds=1.1,
            ttft_seconds=0.1,
            output_tokens=100,
            rounds=10,
            accepted_per_round=10.0,
            verify_seconds=0.2,
        )
    )

    prediction = curve.predict_seconds(_context(requested_tokens=200))

    assert prediction == pytest.approx(2.1)
    assert curve.verify_seconds_per_round.mean == pytest.approx(0.02)


def _run_deterministic_sequence() -> tuple[list[str], OnlineDrafterRouter]:
    router = OnlineDrafterRouter(("dflash", "dspark"))
    choices: list[str] = []
    for index in range(1, 9):
        context = _context(index)
        decision = router.route(context)
        choices.append(decision.arm)
        wall = 1.0 if decision.arm == "dflash" else 0.5
        router.observe(context, decision, _observation(decision.arm, wall))
    return choices, router


def test_router_is_deterministic_and_converges_after_bounded_calibration() -> None:
    first_choices, first = _run_deterministic_sequence()
    second_choices, second = _run_deterministic_sequence()

    assert first_choices == second_choices
    assert first_choices == ["dflash", "dspark"] + ["dspark"] * 6
    assert first.exploration_count == second.exploration_count == 2
    assert first.best_static_arm() == second.best_static_arm() == "dspark"
    assert all(
        len(event["online_update_arms"]) == 1 for event in first.telemetry
    )


def test_periodic_exploration_never_exceeds_hard_budget() -> None:
    config = RouterConfig(
        max_exploration_decisions=4,
        exploration_interval=2,
    )
    router = OnlineDrafterRouter(("dflash", "dspark"), config=config)
    decisions = []
    for index in range(1, 13):
        context = _context(index)
        decision = router.route(context)
        decisions.append(decision)
        wall = 1.0 if decision.arm == "dflash" else 0.5
        router.observe(context, decision, _observation(decision.arm, wall))

    assert router.exploration_count == 4
    assert sum(decision.exploration for decision in decisions) == 4
    assert [decision.arm for decision in decisions[-5:]] == ["dspark"] * 5


def test_hysteresis_requires_repeated_evidence_before_switching() -> None:
    config = RouterConfig(
        max_exploration_decisions=3,
        exploration_interval=2,
        switch_margin=0.05,
        switch_patience=2,
    )
    router = OnlineDrafterRouter(("a", "b"), config=config)

    context = _context(1)
    first = router.route(context)
    router.observe(context, first, _observation("a", 1.0))
    context = _context(2)
    second = router.route(context)
    router.observe(context, second, _observation("b", 1.4))

    context = _context(3)
    establish = router.route(context)
    assert establish.arm == "a"
    router.observe(context, establish, _observation("a", 1.0))

    context = _context(4)
    explore = router.route(context)
    assert explore.arm == "b" and explore.reason == "bounded_exploration"
    router.observe(context, explore, _observation("b", 0.1))

    context = _context(5)
    hold = router.route(context)
    assert hold.arm == "a" and hold.reason == "hysteresis_hold"
    router.observe(context, hold, _observation("a", 1.0))

    context = _context(6)
    switch = router.route(context)
    assert switch.arm == "b" and switch.reason == "hysteresis_switch"


def test_regression_guard_latches_to_best_static_without_dual_execution() -> None:
    config = RouterConfig(
        max_exploration_decisions=3,
        exploration_interval=2,
        guard_regression_margin=0.0,
        guard_patience=1,
    )
    router = OnlineDrafterRouter(("a", "b"), config=config)
    for index, wall in ((1, 0.5), (2, 1.0), (3, 0.5)):
        context = _context(index)
        decision = router.route(context)
        router.observe(context, decision, _observation(decision.arm, wall))

    context = _context(4)
    explored = router.route(context)
    assert explored.arm == "b"
    router.observe(context, explored, _observation("b", 2.0))
    assert router.fallback_latched is True

    context = _context(5)
    guarded = router.route(context)
    assert guarded.arm == "a"
    assert guarded.reason == "guarded_static_fallback"


def test_feedback_for_an_unselected_arm_is_rejected() -> None:
    router = OnlineDrafterRouter(("a", "b"))
    context = _context()
    decision = router.route(context)

    with pytest.raises(ValueError, match="selected arm"):
        router.observe(context, decision, _observation("b", 1.0))


def test_router_rejects_a_zero_sample_calibration() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RouterConfig(warmup_samples_per_arm=0)
