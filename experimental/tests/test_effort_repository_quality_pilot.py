from __future__ import annotations

import hashlib
import json

import pytest

from experimental.effort import repository_quality_pilot as pilot
from experimental.effort.repository_quality_pilot import (
    ALL_SUITE_SHA256,
    BOOTSTRAP_SAMPLES,
    CLASSIFIER_PRECEDENCE,
    DIRECT_CANDIDATE_BUDGET,
    EvaluationBarrierReceipt,
    EXTRA_CANDIDATE_BUDGET,
    EXTRA_SCHEDULE_REVISION,
    FROZEN_SEED,
    LOGICAL_ARMS,
    ROOT_SCHEDULE_REVISION,
    SCHEMA_VERSION,
    SMOKE_SUITE_SHA256,
    STATIC_ALLOCATION_REVISION,
    ArmHiddenOutcome,
    CalibratedTransition,
    CalibratedTransitionTable,
    CandidateBudget,
    CandidateChoice,
    CandidateCost,
    CandidateObservation,
    ExtraAction,
    ExtraAllocation,
    ExtraSpec,
    FixturePilotRecord,
    HiddenOutcome,
    LogicalArm,
    PhysicalRoot,
    PilotProtocol,
    PilotProtocolError,
    PilotRouter,
    PublicEvidence,
    PublicState,
    RouterMode,
    TransitionIdentity,
    VisibleCheckOutcome,
    allocate_extras,
    build_aggregate,
    classify_public_state,
    make_extra_schedule,
    make_root_schedule,
    select_candidate,
    select_static_fixture_ids,
    serialize_source_free_aggregate,
)


def _digest(seed: int, fixture_id: str, namespace: str) -> bytes:
    return hashlib.sha256(
        namespace.encode("ascii") + b"\0" + str(seed).encode("ascii") + b"\0" + fixture_id.encode("utf-8")
    ).digest()


def _evidence(
    state: PublicState = PublicState.PUBLIC_UNKNOWN,
    *,
    coverage_debt: bool = False,
    snapshot_and_telemetry_complete: bool = True,
    quality_decision_is_pass: bool = True,
    terminal_reason_is_model_final: bool = True,
    trusted_test_or_build_present: bool = False,
    trusted_static_or_diff_present: bool = False,
    budget_exhausted: bool = False,
    deadline_violated: bool = False,
) -> PublicEvidence:
    initial_complete = snapshot_and_telemetry_complete
    current_complete = snapshot_and_telemetry_complete
    telemetry_complete = snapshot_and_telemetry_complete
    scope_valid = state is not PublicState.SCOPE_INVALID
    quality_decision = "pass" if quality_decision_is_pass else "incomplete"
    terminal_reason = "model_final" if terminal_reason_is_model_final else "budget"
    attempts = int(trusted_test_or_build_present or state is PublicState.PUBLIC_FAIL)
    test_successes = int(trusted_test_or_build_present and state is not PublicState.PUBLIC_FAIL)
    visible_check = (
        VisibleCheckOutcome.NOT_RUN
        if attempts == 0
        else VisibleCheckOutcome.PASS
        if test_successes == attempts
        else VisibleCheckOutcome.FAIL
    )
    if state is PublicState.ROOT_INCOMPLETE and all(
        (
            snapshot_and_telemetry_complete,
            quality_decision_is_pass,
            terminal_reason_is_model_final,
            not budget_exhausted,
            not deadline_violated,
        )
    ):
        quality_decision = "incomplete"
    evidence = PublicEvidence(
        initial_snapshot_complete=initial_complete,
        current_snapshot_complete=current_complete,
        tool_telemetry_complete=telemetry_complete,
        budget_exhausted=budget_exhausted,
        deadline_violated=deadline_violated,
        quality_decision=quality_decision,
        terminal_reason=terminal_reason,
        scope_valid=scope_valid,
        visible_check=visible_check,
        mutation_epoch=2 if coverage_debt else 0,
        trusted_test_or_build_attempt_count=attempts,
        trusted_test_count=test_successes,
        trusted_static_count=int(trusted_static_or_diff_present),
    )
    assert evidence.state is state
    assert evidence.coverage_debt is coverage_debt
    return evidence


def _candidate(
    identifier: str,
    state: PublicState,
    *,
    coverage_debt: bool = False,
    snapshot_and_telemetry_complete: bool = True,
    quality_decision_is_pass: bool = True,
    terminal_reason_is_model_final: bool = True,
    trusted_test_or_build_present: bool = False,
    trusted_static_or_diff_present: bool = False,
    budget_exhausted: bool = False,
    deadline_violated: bool = False,
    terminal_artifact_id: str | None = None,
    cost: CandidateCost | None = None,
) -> CandidateObservation:
    return CandidateObservation(
        physical_candidate_id=identifier,
        terminal_artifact_id=terminal_artifact_id or hashlib.sha256(identifier.encode()).hexdigest(),
        public_evidence=_evidence(
            state,
            coverage_debt=coverage_debt,
            snapshot_and_telemetry_complete=snapshot_and_telemetry_complete,
            quality_decision_is_pass=quality_decision_is_pass,
            terminal_reason_is_model_final=terminal_reason_is_model_final,
            trusted_test_or_build_present=trusted_test_or_build_present,
            trusted_static_or_diff_present=trusted_static_or_diff_present,
            budget_exhausted=budget_exhausted,
            deadline_violated=deadline_violated,
        ),
        cost=cost or CandidateCost(1, 2, 10, 1.0, 2.0),
    )


def _outcomes(
    *values: tuple[bool, bool] | tuple[bool, bool, bool],
) -> tuple[ArmHiddenOutcome, ...]:
    assert len(values) == len(LOGICAL_ARMS)
    result = []
    for arm, value in zip(LOGICAL_ARMS, values, strict=True):
        evaluator_passed, regression_free = value[:2]
        trajectory_valid = value[2] if len(value) == 3 else True
        result.append(
            ArmHiddenOutcome(
                arm,
                HiddenOutcome(evaluator_passed, regression_free),
                trajectory_valid,
            )
        )
    return tuple(result)


def _barrier_receipt(records: tuple[FixturePilotRecord, ...]) -> EvaluationBarrierReceipt:
    artifacts = {
        (record.fixture_id, record.candidate_for_arm(arm).terminal_artifact_id)
        for record in records
        for arm in LOGICAL_ARMS
    }
    logical_count = len(records) * len(LOGICAL_ARMS)
    return EvaluationBarrierReceipt(
        expected_logical_selection_count=logical_count,
        registered_logical_selection_count=logical_count,
        unique_terminal_artifact_count=len(artifacts),
        hidden_evaluation_count=len(artifacts),
        all_generation_complete_before_seal=True,
        selection_sealed_before_hidden=True,
        hidden_evaluation_single_use=True,
    )


def test_public_classifier_is_total_and_uses_frozen_precedence() -> None:
    assert CLASSIFIER_PRECEDENCE == (
        "root_incomplete",
        "scope_invalid",
        "public_fail",
        "public_unknown",
    )
    common = {
        "initial_snapshot_complete": True,
        "current_snapshot_complete": True,
        "tool_telemetry_complete": True,
        "budget_exhausted": False,
        "deadline_violated": False,
        "quality_decision": "pass",
        "terminal_reason": "model_final",
        "scope_valid": True,
    }

    assert classify_public_state(**common, visible_check=VisibleCheckOutcome.PASS) is PublicState.PUBLIC_UNKNOWN
    assert classify_public_state(**common, visible_check=VisibleCheckOutcome.FAIL) is PublicState.PUBLIC_FAIL
    assert (
        classify_public_state(**{**common, "scope_valid": False}, visible_check=VisibleCheckOutcome.FAIL)
        is PublicState.SCOPE_INVALID
    )
    assert (
        classify_public_state(
            **{**common, "initial_snapshot_complete": False, "scope_valid": False},
            visible_check=VisibleCheckOutcome.FAIL,
        )
        is PublicState.ROOT_INCOMPLETE
    )
    for incomplete in (
        "initial_snapshot_complete",
        "current_snapshot_complete",
        "tool_telemetry_complete",
    ):
        assert (
            classify_public_state(**{**common, incomplete: False}, visible_check=VisibleCheckOutcome.PASS)
            is PublicState.ROOT_INCOMPLETE
        )
    for field, value in (
        ("budget_exhausted", True),
        ("deadline_violated", True),
        ("quality_decision", "incomplete"),
        ("terminal_reason", "budget"),
    ):
        assert (
            classify_public_state(**{**common, field: value}, visible_check=VisibleCheckOutcome.PASS)
            is PublicState.ROOT_INCOMPLETE
        )
    assert classify_public_state(**common, visible_check=VisibleCheckOutcome.NOT_RUN) is PublicState.PUBLIC_UNKNOWN


def test_root_schedule_is_input_order_independent_and_counterbalanced() -> None:
    fixtures = tuple(f"case-{index}" for index in range(7))
    first = make_root_schedule(fixtures, seed=19)
    second = make_root_schedule(reversed(fixtures), seed=19)

    assert first == second
    assert len(first) == 2 * len(fixtures)
    first_positions = [first[index].root for index in range(0, len(first), 2)]
    assert abs(first_positions.count(PhysicalRoot.PLAIN) - first_positions.count(PhysicalRoot.QUALITY_SHARED)) <= 1
    assert [item.schedule_index for item in first] == list(range(len(first)))
    for index in range(0, len(first), 2):
        block = first[index : index + 2]
        assert block[0].fixture_id == block[1].fixture_id
        assert {item.root for item in block} == {PhysicalRoot.PLAIN, PhysicalRoot.QUALITY_SHARED}

    expected_order = sorted(fixtures, key=lambda item: (_digest(19, item, ROOT_SCHEDULE_REVISION), item))
    assert [first[index].fixture_id for index in range(0, len(first), 2)] == expected_order


def test_exploratory_router_depends_only_on_public_state_and_coverage_debt() -> None:
    router = PilotRouter()
    assert router.should_route(_evidence(PublicState.ROOT_INCOMPLETE)) is True
    assert router.should_route(_evidence(PublicState.SCOPE_INVALID)) is True
    assert router.should_route(_evidence(PublicState.PUBLIC_FAIL)) is True
    assert router.should_route(_evidence(PublicState.PUBLIC_UNKNOWN)) is False
    assert router.should_route(_evidence(PublicState.PUBLIC_UNKNOWN, coverage_debt=True)) is True

    # Snapshot/telemetry incompleteness is terminal evidence, not a safe branch
    # point. Other root-incomplete causes remain eligible for one recovery.
    assert (
        router.should_route(
            _evidence(
                PublicState.ROOT_INCOMPLETE,
                snapshot_and_telemetry_complete=False,
            )
        )
        is False
    )
    assert (
        router.should_route(
            _evidence(
                PublicState.ROOT_INCOMPLETE,
                snapshot_and_telemetry_complete=True,
            )
        )
        is True
    )
    assert (
        router.should_route(
            _evidence(
                PublicState.ROOT_INCOMPLETE,
                coverage_debt=True,
                snapshot_and_telemetry_complete=False,
            )
        )
        is False
    )

    # Hidden outcomes have no place in the router interface and cannot change it.
    evidence = _evidence(PublicState.PUBLIC_UNKNOWN, coverage_debt=True)
    before = router.should_route(evidence)
    _hidden_a = HiddenOutcome(False, False)
    _hidden_b = HiddenOutcome(True, True)
    assert router.should_route(evidence) is before


def test_calibrated_router_rejects_empty_or_incompatible_transition_identity() -> None:
    expected = TransitionIdentity("model", "config", "prompt", "corpus", "split", "backend")
    other = TransitionIdentity("other", "config", "prompt", "corpus", "split", "backend")

    with pytest.raises(PilotProtocolError, match="empty"):
        PilotRouter(
            mode=RouterMode.CALIBRATED,
            expected_identity=expected,
            transition_table=CalibratedTransitionTable(expected, ()),
        )
    with pytest.raises(PilotProtocolError, match="incompatible"):
        PilotRouter(
            mode=RouterMode.CALIBRATED,
            expected_identity=expected,
            transition_table=CalibratedTransitionTable(
                other,
                (CalibratedTransition(PublicState.PUBLIC_FAIL, False, True),),
            ),
        )

    router = PilotRouter(
        mode=RouterMode.CALIBRATED,
        expected_identity=expected,
        transition_table=CalibratedTransitionTable(
            expected,
            (CalibratedTransition(PublicState.PUBLIC_FAIL, False, True),),
        ),
    )
    assert router.should_route(_evidence(PublicState.PUBLIC_FAIL)) is True
    assert router.should_route(_evidence(PublicState.PUBLIC_UNKNOWN)) is False
    assert router.identity_sha256 == expected.sha256


def test_static_allocation_is_exact_k_and_uses_only_the_preregistered_hash() -> None:
    fixtures = tuple(f"fixture-{index}" for index in range(9))
    selected = select_static_fixture_ids(fixtures, k=4, seed=20260718)
    expected = tuple(
        sorted(
            fixtures,
            key=lambda item: (_digest(20260718, item, STATIC_ALLOCATION_REVISION), item),
        )[:4]
    )
    assert selected == expected
    assert select_static_fixture_ids(reversed(fixtures), k=4, seed=20260718) == expected

    evidence = {
        fixture_id: _evidence(PublicState.PUBLIC_FAIL if index < 3 else PublicState.PUBLIC_UNKNOWN)
        for index, fixture_id in enumerate(fixtures)
    }
    allocation = allocate_extras(fixtures, evidence, router=PilotRouter(), seed=20260718)
    assert allocation.k == 3
    assert len(allocation.static_fixture_ids) == len(allocation.markov_fixture_ids) == 3
    assert allocation.static_fixture_ids == select_static_fixture_ids(fixtures, k=3, seed=20260718)


def test_extra_schedule_deduplicates_overlap_and_preserves_logical_references() -> None:
    allocation = ExtraAllocation(
        fixture_ids=("a", "b", "c", "d"),
        markov_fixture_ids=("a", "c", "d"),
        static_fixture_ids=("a", "b", "d"),
    )
    spec = ExtraSpec(action=ExtraAction.REFINE)
    schedule = make_extra_schedule(allocation, spec=spec, seed=55)

    assert len(schedule) == len(set(allocation.static_fixture_ids) | set(allocation.markov_fixture_ids))
    assert [item.schedule_index for item in schedule] == list(range(len(schedule)))
    assert {item.action for item in schedule} == {ExtraAction.REFINE}
    assert {item.budget for item in schedule} == {spec.budget}
    assert sum(LogicalArm.MARKOV_QUALITY in item.arms for item in schedule) == allocation.k
    assert sum(LogicalArm.QUALITY_STATIC_EXTRA in item.arms for item in schedule) == allocation.k
    assert {item.fixture_id: item.arms for item in schedule if item.fixture_id in {"a", "d"}} == {
        "a": (LogicalArm.QUALITY_STATIC_EXTRA, LogicalArm.MARKOV_QUALITY),
        "d": (LogicalArm.QUALITY_STATIC_EXTRA, LogicalArm.MARKOV_QUALITY),
    }
    expected_order = sorted(
        {"a", "b", "c", "d"},
        key=lambda item: (_digest(55, item, EXTRA_SCHEDULE_REVISION), item),
    )
    assert [item.fixture_id for item in schedule] == expected_order
    assert EXTRA_SCHEDULE_REVISION == "mio.repository-quality.extra-order.v1"


def test_frozen_root_and_extra_budgets_cannot_drift() -> None:
    assert PilotProtocol(suite_sha256=SMOKE_SUITE_SHA256, seed=FROZEN_SEED).root_budget == DIRECT_CANDIDATE_BUDGET
    assert ExtraSpec().budget == EXTRA_CANDIDATE_BUDGET
    with pytest.raises(PilotProtocolError, match="root budget"):
        PilotProtocol(
            suite_sha256=SMOKE_SUITE_SHA256,
            seed=FROZEN_SEED,
            root_budget=CandidateBudget(max_output_tokens=2049),
        )
    with pytest.raises(PilotProtocolError, match="extra budget"):
        ExtraSpec(budget=CandidateBudget(max_rounds=5, max_tool_calls=8, max_output_tokens=384, max_wall_seconds=20.0))


def test_selector_requires_strict_public_improvement_and_ties_keep_root() -> None:
    root = _candidate("root", PublicState.PUBLIC_FAIL)
    assert select_candidate(root, None) is CandidateChoice.ROOT
    assert select_candidate(root, _candidate("tie", PublicState.PUBLIC_FAIL)) is CandidateChoice.ROOT
    assert select_candidate(root, _candidate("better", PublicState.PUBLIC_UNKNOWN)) is CandidateChoice.CHILD
    assert (
        select_candidate(
            _candidate("debt", PublicState.PUBLIC_UNKNOWN, coverage_debt=True),
            _candidate("covered", PublicState.PUBLIC_UNKNOWN, coverage_debt=False),
        )
        is CandidateChoice.ROOT
    )
    assert (
        select_candidate(
            _candidate("root-static", PublicState.PUBLIC_UNKNOWN),
            _candidate(
                "child-static",
                PublicState.PUBLIC_UNKNOWN,
                trusted_static_or_diff_present=True,
            ),
        )
        is CandidateChoice.CHILD
    )
    assert (
        select_candidate(
            _candidate("invalid-root", PublicState.SCOPE_INVALID),
            _candidate(
                "inadmissible",
                PublicState.ROOT_INCOMPLETE,
                budget_exhausted=True,
            ),
        )
        is CandidateChoice.ROOT
    )
    assert (
        select_candidate(
            _candidate("good", PublicState.PUBLIC_UNKNOWN),
            _candidate("worse", PublicState.PUBLIC_FAIL),
        )
        is CandidateChoice.ROOT
    )


def _aggregate_fixture() -> tuple[object, ExtraAllocation, tuple[FixturePilotRecord, ...]]:
    fixtures = (
        "SECRET_FIXTURE_ALPHA",
        "SECRET_FIXTURE_BETA",
        "SECRET_FIXTURE_GAMMA",
        "SECRET_FIXTURE_DELTA",
    )
    static_id = select_static_fixture_ids(fixtures, k=1, seed=FROZEN_SEED)[0]
    evidence = {
        fixture_id: _evidence(PublicState.PUBLIC_FAIL if fixture_id == static_id else PublicState.PUBLIC_UNKNOWN)
        for fixture_id in fixtures
    }
    allocation = allocate_extras(fixtures, evidence, router=PilotRouter(), seed=FROZEN_SEED)
    assert allocation.static_fixture_ids == allocation.markov_fixture_ids == (static_id,)

    records: list[FixturePilotRecord] = []
    for fixture_id in fixtures:
        plain = _candidate(f"PRIVATE_PLAIN_{fixture_id}", PublicState.PUBLIC_UNKNOWN)
        quality = _candidate(
            f"PRIVATE_QUALITY_{fixture_id}",
            evidence[fixture_id].state,
        )
        if fixture_id == static_id:
            # Static and Markov logically reference one physically deduplicated
            # child. Logical costs still charge that child to both policies.
            child = _candidate(f"PRIVATE_SHARED_CHILD_{fixture_id}", PublicState.PUBLIC_UNKNOWN)
            static_child = markov_child = child
            static_choice = markov_choice = CandidateChoice.CHILD
            outcomes = _outcomes((False, True), (False, True), (True, True), (True, True))
        else:
            static_child = markov_child = None
            static_choice = markov_choice = CandidateChoice.ROOT
            outcomes = _outcomes((True, True), (True, True), (True, True), (True, True))
        records.append(
            FixturePilotRecord(
                fixture_id=fixture_id,
                plain_root=plain,
                quality_root=quality,
                static_child=static_child,
                markov_child=markov_child,
                static_selection=static_choice,
                markov_selection=markov_choice,
                outcomes=outcomes,
            )
        )
    protocol = PilotProtocol(suite_sha256=SMOKE_SUITE_SHA256, seed=FROZEN_SEED)
    materialized = tuple(records)
    aggregate = build_aggregate(
        protocol=protocol,
        allocation=allocation,
        records=materialized,
        barrier_receipt=_barrier_receipt(materialized),
    )
    return aggregate, allocation, materialized


def test_shared_quality_root_logical_cost_and_physical_dedup_are_distinct() -> None:
    aggregate, allocation, _records = _aggregate_fixture()
    metrics = dict(aggregate.arm_metrics)

    assert allocation.k == 1
    assert metrics[LogicalArm.PLAIN].logical_candidate_count == 4
    assert metrics[LogicalArm.QUALITY].logical_candidate_count == 4
    assert metrics[LogicalArm.QUALITY_STATIC_EXTRA].logical_candidate_count == 5
    assert metrics[LogicalArm.MARKOV_QUALITY].logical_candidate_count == 5
    assert metrics[LogicalArm.PLAIN].logical_cost.output_tokens == 40
    assert metrics[LogicalArm.QUALITY].logical_cost.output_tokens == 40
    assert metrics[LogicalArm.QUALITY_STATIC_EXTRA].logical_cost.output_tokens == 50
    assert metrics[LogicalArm.MARKOV_QUALITY].logical_cost.output_tokens == 50

    # Four Plain roots + four shared Quality roots + one physically shared child.
    assert aggregate.physical_costs.unique_candidate_count == 9
    assert aggregate.physical_costs.logical_candidate_reference_count == 18
    assert aggregate.physical_costs.shared_or_deduplicated_reference_count == 9
    assert aggregate.physical_costs.cost.output_tokens == 90
    assert aggregate.selected_outputs_consistent is True


def test_paired_contrasts_use_terminal_hidden_outcomes_and_logical_costs() -> None:
    aggregate, _allocation, _records = _aggregate_fixture()
    contrasts = dict(aggregate.contrasts)

    static = contrasts["static_vs_quality"]
    assert (static.both_pass, static.baseline_only, static.candidate_only, static.neither_pass) == (3, 0, 1, 0)
    assert static.pass_rate_delta == 0.25
    assert static.output_token_ratio == 1.25
    markov_static = contrasts["markov_vs_static"]
    assert markov_static.pass_rate_delta == 0.0
    assert markov_static.output_token_ratio == 1.0


def test_hidden_success_reuses_verdict_but_derives_physical_trajectory_validity() -> None:
    fixtures = ("fixture-a", "fixture-b", "fixture-c", "fixture-d")
    evidence = {fixture_id: _evidence(PublicState.PUBLIC_UNKNOWN) for fixture_id in fixtures}
    allocation = allocate_extras(fixtures, evidence, router=PilotRouter(), seed=FROZEN_SEED)
    records: list[FixturePilotRecord] = []
    for index, fixture_id in enumerate(fixtures):
        shared_artifact = hashlib.sha256(f"same-terminal-bytes:{fixture_id}".encode()).hexdigest()
        plain = _candidate(
            f"plain:{fixture_id}",
            PublicState.ROOT_INCOMPLETE if index == 0 else PublicState.PUBLIC_UNKNOWN,
            terminal_reason_is_model_final=index != 0,
            terminal_artifact_id=shared_artifact,
        )
        shared_quality = _candidate(
            f"quality:{fixture_id}",
            PublicState.PUBLIC_UNKNOWN,
            terminal_artifact_id=shared_artifact,
        )
        records.append(
            FixturePilotRecord(
                fixture_id=fixture_id,
                plain_root=plain,
                quality_root=shared_quality,
                static_child=None,
                markov_child=None,
                static_selection=CandidateChoice.ROOT,
                markov_selection=CandidateChoice.ROOT,
                outcomes=_outcomes(
                    (True, True, index != 0),
                    (True, True, True),
                    (True, True, True),
                    (True, True, True),
                ),
            )
        )
    materialized = tuple(records)
    aggregate = build_aggregate(
        protocol=PilotProtocol(suite_sha256=SMOKE_SUITE_SHA256, seed=FROZEN_SEED),
        allocation=allocation,
        records=materialized,
        barrier_receipt=_barrier_receipt(materialized),
    )
    metrics = dict(aggregate.arm_metrics)

    # Independent physical trajectories may end in identical bytes and share
    # one evaluator verdict, but keep derived trajectory validity separate.
    assert aggregate.selected_outputs_consistent is True
    assert metrics[LogicalArm.PLAIN].passed_count == 3
    assert metrics[LogicalArm.QUALITY].passed_count == 4
    assert metrics[LogicalArm.QUALITY_STATIC_EXTRA].passed_count == 4
    assert metrics[LogicalArm.MARKOV_QUALITY].passed_count == 4
    assert metrics[LogicalArm.PLAIN].workspace_evaluator_passed_count == 4
    assert metrics[LogicalArm.QUALITY].workspace_evaluator_passed_count == 4
    quality_plain = dict(aggregate.contrasts)["quality_vs_plain"]
    assert (quality_plain.baseline_only, quality_plain.candidate_only, quality_plain.pass_rate_delta) == (
        0,
        1,
        0.25,
    )
    assert (
        quality_plain.workspace_evaluator_both_pass,
        quality_plain.workspace_evaluator_baseline_only,
        quality_plain.workspace_evaluator_candidate_only,
        quality_plain.workspace_evaluator_neither_pass,
        quality_plain.workspace_evaluator_pass_rate_delta,
    ) == (4, 0, 0, 0, 0.0)
    assert "workspace_evaluator_gain_lcb_below_0.01" in aggregate.analysis.failures


def test_source_free_schema_contains_no_private_rows_ids_or_hidden_labels() -> None:
    aggregate, allocation, records = _aggregate_fixture()
    serialized = serialize_source_free_aggregate(aggregate)
    payload = json.loads(serialized)

    assert set(payload) == {
        "analysis",
        "arm_metrics",
        "claim",
        "hidden_labels_serialized",
        "integrity",
        "paired_contrasts",
        "physical_costs",
        "protocol",
        "schema_version",
    }
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["hidden_labels_serialized"] is False
    assert payload["claim"]["status"] == "exploratory_no_claim"
    assert payload["claim"]["advance_to_new_calibration_eligible"] is False
    assert payload["analysis"]["status"] == "complete_frozen_analysis"
    assert payload["analysis"]["bootstrap"]["samples"] == BOOTSTRAP_SAMPLES
    assert payload["analysis"]["population"] == {
        "cohort": "smoke",
        "fixture_count": 4,
        "pooled_cohorts": False,
    }
    assert payload["analysis"]["promotion_eligible"] is False
    assert "analysis_population_not_single_all_cohort" in payload["analysis"]["failures"]
    assert payload["protocol"]["shared_quality_root_arms"] == [
        "quality",
        "quality_static_extra",
        "markov_quality",
    ]
    assert payload["integrity"]["static_extra_count"] == allocation.k
    for record in records:
        assert record.fixture_id not in serialized
        for candidate in (record.plain_root, record.quality_root, record.static_child, record.markov_child):
            if candidate is not None:
                assert candidate.physical_candidate_id not in serialized
    assert 'passed": true' not in serialized.casefold()


@pytest.mark.parametrize(
    ("states", "reason"),
    [
        ((PublicState.PUBLIC_UNKNOWN,) * 3, "no_markov_routes"),
        ((PublicState.PUBLIC_FAIL,) * 3, "all_tasks_routed"),
    ],
)
def test_k_zero_and_k_n_are_explicitly_nonpromotable(states, reason) -> None:
    fixtures = ("a", "b", "c")
    evidence = {fixture_id: _evidence(state) for fixture_id, state in zip(fixtures, states, strict=True)}
    allocation = allocate_extras(fixtures, evidence, router=PilotRouter(), seed=4)
    assert allocation.promotable is False
    assert allocation.nonpromotable_reason == reason


def test_candidate_retries_and_allocation_drift_fail_closed() -> None:
    with pytest.raises(PilotProtocolError, match="cannot be retried"):
        CandidateObservation(
            physical_candidate_id="candidate",
            terminal_artifact_id=hashlib.sha256(b"candidate").hexdigest(),
            public_evidence=_evidence(PublicState.ROOT_INCOMPLETE),
            cost=CandidateCost(),
            attempt_count=2,
        )
    with pytest.raises(ValueError, match="same number"):
        ExtraAllocation(("a", "b"), ("a",), ())


@pytest.mark.parametrize(
    ("attempts", "test_successes", "visible"),
    [
        (0, 1, VisibleCheckOutcome.NOT_RUN),
        (1, 0, VisibleCheckOutcome.PASS),
        (1, 1, VisibleCheckOutcome.FAIL),
        (0, 0, VisibleCheckOutcome.PASS),
    ],
)
def test_public_evidence_rejects_contradictory_visible_telemetry(
    attempts: int,
    test_successes: int,
    visible: VisibleCheckOutcome,
) -> None:
    with pytest.raises(PilotProtocolError):
        PublicEvidence(
            visible_check=visible,
            trusted_test_or_build_attempt_count=attempts,
            trusted_test_count=test_successes,
        )


def test_frozen_protocol_rejects_seed_suite_prompt_and_router_drift() -> None:
    with pytest.raises(PilotProtocolError, match="seed"):
        PilotProtocol(suite_sha256=SMOKE_SUITE_SHA256, seed=FROZEN_SEED + 1)
    with pytest.raises(ValueError, match="cohort"):
        PilotProtocol(suite_sha256="a" * 64, seed=FROZEN_SEED)
    with pytest.raises(PilotProtocolError, match="prompt revision"):
        ExtraSpec(prompt_revision="different")
    identity = TransitionIdentity("m", "c", "p", "corpus", "all", "dflash")
    calibrated = PilotRouter(
        mode=RouterMode.CALIBRATED,
        expected_identity=identity,
        transition_table=CalibratedTransitionTable(
            identity,
            (CalibratedTransition(PublicState.PUBLIC_FAIL, False, True),),
        ),
    )
    with pytest.raises(PilotProtocolError, match="exploratory"):
        PilotProtocol(
            suite_sha256=ALL_SUITE_SHA256,
            seed=FROZEN_SEED,
            router=calibrated,
        )


def test_bootstrap_counter_has_independent_golden_vectors() -> None:
    vectors = {
        (0, 0): ("b474384edae2ac677866e1e1a65118bff1c5b442f89e1b8e769bd5a49cea1ac1", 1),
        (0, 1): ("1d42f2fd627ef5b5e6e226d18cfdd18884d0b923894723189976b2dfdff58338", 8),
        (1, 0): ("488769d8521dbcd50bf9a8ec26f31c1d966cfcebeaefa0ea2b52b5d94a15e266", 10),
        (9999, 11): ("d9a4c23a464e22aa2ed6636a4b72dc66d1496fadc1c33d65a97569184d36b42d", 5),
    }
    for (sample_index, draw_index), (expected_digest, expected_index) in vectors.items():
        payload = (
            b"mio.repository-quality.bootstrap.v1\0"
            + b"20260719\0"
            + sample_index.to_bytes(8, "big")
            + draw_index.to_bytes(8, "big")
        )
        assert hashlib.sha256(payload).hexdigest() == expected_digest
        assert (
            pilot._bootstrap_index(
                fixture_count=12,
                sample_index=sample_index,
                draw_index=draw_index,
            )
            == expected_index
        )
    assert pilot.BOOTSTRAP_LOWER_INDEX == 499


def _promotion_fixture() -> tuple[ExtraAllocation, tuple[FixturePilotRecord, ...]]:
    fixtures = tuple(f"case-{index:02d}" for index in range(12))
    routed = set(fixtures[:8])
    evidence = {
        fixture_id: _evidence(PublicState.PUBLIC_FAIL if fixture_id in routed else PublicState.PUBLIC_UNKNOWN)
        for fixture_id in fixtures
    }
    allocation = allocate_extras(fixtures, evidence, router=PilotRouter(), seed=FROZEN_SEED)
    assert allocation.k == 8
    root_cost = CandidateCost(1, 2, 100, 10.0, 10.0)
    child_cost = CandidateCost(1, 2, 10, 1.0, 1.0)
    records: list[FixturePilotRecord] = []
    for fixture_id in fixtures:
        root_evaluator_pass = fixture_id not in routed
        plain = _candidate(
            f"plain:{fixture_id}",
            PublicState.PUBLIC_UNKNOWN,
            cost=root_cost,
        )
        quality = _candidate(
            f"quality:{fixture_id}",
            evidence[fixture_id].state,
            trusted_test_or_build_present=fixture_id not in routed,
            cost=root_cost,
        )
        allocated = fixture_id in set(allocation.static_fixture_ids) | set(allocation.markov_fixture_ids)
        child = (
            _candidate(
                f"child:{fixture_id}",
                PublicState.PUBLIC_UNKNOWN,
                trusted_test_or_build_present=True,
                trusted_static_or_diff_present=True,
                cost=child_cost,
            )
            if allocated
            else None
        )
        static_child = child if fixture_id in allocation.static_fixture_ids else None
        markov_child = child if fixture_id in allocation.markov_fixture_ids else None
        static_choice = select_candidate(quality, static_child)
        markov_choice = select_candidate(quality, markov_child)
        static_pass = True if static_choice is CandidateChoice.CHILD else root_evaluator_pass
        markov_pass = True if markov_choice is CandidateChoice.CHILD else root_evaluator_pass
        records.append(
            FixturePilotRecord(
                fixture_id=fixture_id,
                plain_root=plain,
                quality_root=quality,
                static_child=static_child,
                markov_child=markov_child,
                static_selection=static_choice,
                markov_selection=markov_choice,
                outcomes=_outcomes(
                    (True, True),
                    (root_evaluator_pass, True),
                    (static_pass, True),
                    (markov_pass, True),
                ),
            )
        )
    return allocation, tuple(records)


def test_exact_all_cohort_bootstrap_and_promotion_gate_are_executable() -> None:
    allocation, records = _promotion_fixture()
    aggregate = build_aggregate(
        protocol=PilotProtocol(suite_sha256=ALL_SUITE_SHA256, seed=FROZEN_SEED),
        allocation=allocation,
        records=records,
        barrier_receipt=_barrier_receipt(records),
    )
    analysis = aggregate.analysis

    assert analysis.bootstrap_samples == BOOTSTRAP_SAMPLES
    assert analysis.route_count == 8
    assert analysis.rescue_numerator == analysis.rescue_denominator == 8
    assert analysis.quality_gain_point == pytest.approx(8 / 12)
    assert analysis.quality_gain_lcb >= 0.01
    assert analysis.workspace_evaluator_gain_point == pytest.approx(8 / 12)
    assert analysis.workspace_evaluator_gain_lcb >= 0.01
    assert analysis.rescue_probability_lcb >= 0.10
    assert analysis.workspace_evaluator_rescue_count == 8
    assert analysis.trajectory_only_rescue_count == 0
    assert analysis.changed_terminal_rescue_count == 8
    assert analysis.same_terminal_rescue_count == 0
    assert analysis.byte_changed_selected_child_count == 8
    assert analysis.quality_to_markov_regressions == 0
    assert analysis.workspace_evaluator_regressions == 0
    assert analysis.quality_workspace_evaluator_pass_count == 4
    assert analysis.static_workspace_evaluator_pass_count >= 4
    assert analysis.markov_workspace_evaluator_pass_count == 12
    assert analysis.output_tokens_point_ratio == pytest.approx((1200 + 80) / 1200)
    assert analysis.promotion_eligible is True
    assert analysis.failures == ()


def test_barrier_receipt_and_trajectory_validity_fail_closed() -> None:
    allocation, records = _promotion_fixture()
    record = records[0]
    forged_outcomes = tuple(
        ArmHiddenOutcome(item.arm, item.outcome, not item.trajectory_valid) if item.arm is LogicalArm.QUALITY else item
        for item in record.outcomes
    )
    with pytest.raises(PilotProtocolError, match="trajectory validity"):
        FixturePilotRecord(
            fixture_id=record.fixture_id,
            plain_root=record.plain_root,
            quality_root=record.quality_root,
            static_child=record.static_child,
            markov_child=record.markov_child,
            static_selection=record.static_selection,
            markov_selection=record.markov_selection,
            outcomes=forged_outcomes,
        )
    with pytest.raises(PilotProtocolError, match="artifact count"):
        build_aggregate(
            protocol=PilotProtocol(suite_sha256=ALL_SUITE_SHA256, seed=FROZEN_SEED),
            allocation=allocation,
            records=records,
            barrier_receipt=EvaluationBarrierReceipt(
                expected_logical_selection_count=48,
                registered_logical_selection_count=48,
                unique_terminal_artifact_count=47,
                hidden_evaluation_count=47,
                all_generation_complete_before_seal=True,
                selection_sealed_before_hidden=True,
                hidden_evaluation_single_use=True,
            ),
        )
