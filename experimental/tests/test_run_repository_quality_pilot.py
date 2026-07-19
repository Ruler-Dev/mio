from __future__ import annotations

import hashlib
import inspect
import json
import platform
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import experimental.effort.run_repository_quality_pilot as pilot_runner
from experimental.effort.bench_repository_quality_pilot import (
    DIRECT_EXECUTION_BUDGET,
    EXTRA_EXECUTION_BUDGET,
    HiddenEvaluationBarrier,
    HiddenEvaluationBatch,
    ImmutableWorkspaceArchive,
    PublicRepositoryState,
    RepositoryPilotProtocolError,
    RetainedAgentStage,
    SelectedFixtureCandidates,
    logical_terminal_key,
    regular_tree_sha256,
    to_protocol_candidate_cost,
    to_protocol_public_evidence,
)
from experimental.effort.repository_quality_pilot import (
    FROZEN_SEED,
    LOGICAL_ARMS,
    SMOKE_SUITE_SHA256,
    CandidateChoice,
    CandidateObservation,
    GenerationCompletionReceipt,
    HiddenOutcome,
    LogicalArm,
    PilotProtocol,
    make_root_schedule,
)
from experimental.effort.run_repository_quality_pilot import (
    PilotRunPhase,
    RepositoryQualityPilotOrchestrator,
    bind_records_from_hidden_batch,
    execute_repository_quality_pilot,
)
from mio.agent import AgentRoundTrace, AgentToolTrace, AgentTurnResult
from mio.agent_policy import AgentToolPolicy
from mio.coding_quality import (
    CodingEffort,
    CodingQualityGate,
    RequestIntent,
    ValidationKind,
    snapshot_workspaces,
)
from scripts.run_coding_quality_benchmark import select_cases
from scripts.bench_coding_quality import HiddenEvaluation
from scripts.run_coding_quality_benchmark import (
    DRAFT_CONTENT_IDENTITY,
    DRAFT_REPOSITORY_LABEL,
    FROZEN_SOFTWARE_VERSIONS,
    TARGET_CONTENT_IDENTITY,
    TARGET_REPOSITORY_LABEL,
    CleanSourceLock,
    LocalModelLock,
    RuntimeIdentity,
)


_DRAFTER_REF = "fake/dflash"
_PLAIN_TOOLS = ("bash", "read", "write", "edit")
_QUALITY_TOOLS = ("validate", *_PLAIN_TOOLS)


def _tool_surface(names: tuple[str, ...]) -> tuple[dict[str, object], tuple[dict, ...]]:
    registry = {name: object() for name in names}
    specs = tuple({"type": "function", "function": {"name": name, "parameters": {}}} for name in names)
    return registry, specs


def _round_trace(*, completion_tokens: int = 1) -> AgentRoundTrace:
    return AgentRoundTrace(
        round_index=0,
        prompt_tokens=8,
        completion_tokens=completion_tokens,
        total_time_s=0.5,
        prompt_tps=16.0,
        generation_tps=2.0,
        generation_backend="dflash",
        fallback_ar=False,
        prefill_ns=200_000_000,
        decode_ns=300_000_000,
        model_total_ns=500_000_000,
        logical_prompt_tokens=8,
        physical_prefill_tokens=8,
        physical_decode_tokens=completion_tokens,
        warm_offset=0,
        warm_offset_tokens=0,
        timing_source="runtime_raw_ns",
        drafter_requested="dflash",
        drafter_selected="dflash",
        drafter_ref=_DRAFTER_REF,
        deadline_hit=False,
    )


def _validate_trace(sequence: int) -> AgentToolTrace:
    return AgentToolTrace(
        sequence=sequence,
        round_index=0,
        tool_name="validate",
        operation="validate",
        permission="shell",
        allowed=True,
        outcome="ok",
        target_sha256=f"{sequence + 1:064x}",
        duration_ns=1,
        output_chars=1,
        audit_count=1,
        audit_sha256=f"{sequence + 10:064x}",
        telemetry_complete=True,
    )


def _editable_module(root: Path) -> Path:
    candidates = tuple(path for path in sorted(root.glob("*.py")) if not path.name.startswith("test_public_"))
    if len(candidates) != 1:
        raise AssertionError("fake corpus executor expected one editable top-level Python module")
    return candidates[0]


def _repair_workspace(root: Path) -> None:
    module = _editable_module(root)
    original = module.read_text(encoding="utf-8")
    module.write_text(original.rstrip() + "\n\n# repaired by fake executor\n", encoding="utf-8")


def _state(
    *,
    root: Path,
    quality_enabled: bool,
    budget: object,
    effort: str,
) -> dict[str, object]:
    names = _QUALITY_TOOLS if quality_enabled else _PLAIN_TOOLS
    registry, specs = _tool_surface(names)
    return {
        "tool_policy": AgentToolPolicy.coding_workspace(root, allow_network=False),
        "tool_registry": registry,
        "tool_specs": specs,
        "execution_budget": budget,
        "coding_effort": effort,
        "quality_gate_enabled": quality_enabled,
        "quality_gate_require_change": quality_enabled,
        "messages": [],
    }


class _FakeRetainedExecutor:
    """Produces strict retained telemetry while doing no model inference."""

    def __init__(self) -> None:
        self.direct_calls: list[tuple[str, bool]] = []
        self.recovery_calls: list[str] = []

    def run_direct(
        self,
        *,
        fixture_id: str,
        instruction: str,
        workspace: Path,
        quality_enabled: bool,
        effort: str = "medium",
    ) -> RetainedAgentStage:
        assert effort == CodingEffort.MEDIUM.value
        root = workspace.resolve(strict=True)
        pristine_tree = regular_tree_sha256(root)
        pristine = snapshot_workspaces((root,))
        gate: CodingQualityGate | None = None
        report: dict[str, object] | None = None
        if quality_enabled:
            gate = CodingQualityGate(
                roots=(root,),
                effort=CodingEffort.MEDIUM,
                enabled=True,
                intent=RequestIntent.CODE_CHANGE_REQUESTED,
                require_net_workspace_change=True,
                request_sha256="a" * 64,
                initial_snapshot=pristine,
                current_snapshot=pristine,
            )
            report = gate.report()
        else:
            _repair_workspace(root)
        current = snapshot_workspaces((root,))
        result = AgentTurnResult(
            assistant_text="private fake direct output",
            quality_gate=report,
            terminal_reason="model_final",
            budget_exhaustion=None,
            tool_telemetry_complete=True,
            tool_calls=0,
            tool_events=(),
            tool_result_chars=0,
            wall_time_s=0.75,
            rounds=(_round_trace(),),
            completion_tokens=1,
        )
        self.direct_calls.append((fixture_id, quality_enabled))
        return RetainedAgentStage(
            stage="direct",
            fixture_id=fixture_id,
            instruction=instruction,
            workspace=root,
            pristine_tree_sha256=pristine_tree,
            terminal_tree_sha256=regular_tree_sha256(root),
            pristine_snapshot=pristine,
            current_snapshot=current,
            execution_budget=DIRECT_EXECUTION_BUDGET,
            coding_effort=CodingEffort.MEDIUM.value,
            drafter_ref=_DRAFTER_REF,
            state=_state(
                root=root,
                quality_enabled=quality_enabled,
                budget=DIRECT_EXECUTION_BUDGET,
                effort=CodingEffort.MEDIUM.value,
            ),
            result=result,
            trusted_quality_gate=gate,
            quality_enabled=quality_enabled,
        )

    def run_recovery(
        self,
        *,
        direct: RetainedAgentStage,
        archive: ImmutableWorkspaceArchive,
        branch_root: Path,
        containment_root: Path,
    ) -> RetainedAgentStage:
        root = archive.clone_to(branch_root, containment_root=containment_root)
        current = snapshot_workspaces((root,))
        gate = CodingQualityGate(
            roots=(root,),
            effort=CodingEffort.HIGH,
            enabled=True,
            intent=RequestIntent.CODE_CHANGE_REQUESTED,
            require_net_workspace_change=True,
            request_sha256="b" * 64,
            initial_snapshot=direct.pristine_snapshot,
            current_snapshot=current,
        )
        _repair_workspace(root)
        terminal = gate.refresh()
        for index, kind in enumerate((ValidationKind.TEST, ValidationKind.STATIC)):
            gate.record_validation(
                kind,
                command_sha256=f"{index + 30:064x}",
                allowed=True,
                outcome="ok",
                snapshot=terminal,
            )
            gate.validate_invocations += 1
        traces = tuple(_validate_trace(index) for index in range(2))
        result = AgentTurnResult(
            assistant_text="private fake recovery output",
            quality_gate=gate.report(),
            terminal_reason="model_final",
            budget_exhaustion=None,
            tool_telemetry_complete=True,
            tool_calls=2,
            tool_events=traces,
            tool_result_chars=2,
            wall_time_s=0.75,
            rounds=(_round_trace(),),
            completion_tokens=1,
        )
        self.recovery_calls.append(direct.fixture_id)
        return RetainedAgentStage(
            stage="recovery",
            fixture_id=direct.fixture_id,
            instruction=direct.instruction,
            workspace=root,
            pristine_tree_sha256=direct.pristine_tree_sha256,
            terminal_tree_sha256=regular_tree_sha256(root),
            pristine_snapshot=direct.pristine_snapshot,
            current_snapshot=terminal,
            execution_budget=EXTRA_EXECUTION_BUDGET,
            coding_effort=CodingEffort.HIGH.value,
            drafter_ref=_DRAFTER_REF,
            state=_state(
                root=root,
                quality_enabled=True,
                budget=EXTRA_EXECUTION_BUDGET,
                effort=CodingEffort.HIGH.value,
            ),
            result=result,
            trusted_quality_gate=gate,
            quality_enabled=True,
        )


def _smoke_protocol() -> PilotProtocol:
    return PilotProtocol(suite_sha256=SMOKE_SUITE_SHA256, seed=FROZEN_SEED)


def test_fake_smoke_executes_sealed_roots_deduplicated_extras_and_hidden_barrier(
    tmp_path: Path,
) -> None:
    cases = select_cases("smoke")
    protocol = _smoke_protocol()
    executor = _FakeRetainedExecutor()
    events: list[str] = []
    hidden_calls: list[tuple[str, str]] = []

    def verify() -> None:
        events.append("verify")

    def factory():
        events.append("factory")

        def evaluate(fixture_id: str, root: Path) -> HiddenOutcome:
            events.append("hidden")
            tree = regular_tree_sha256(root)
            hidden_calls.append((fixture_id, tree))
            repaired = "# repaired by fake executor" in _editable_module(root).read_text(encoding="utf-8")
            return HiddenOutcome(evaluator_passed=repaired, regression_free=True)

        return evaluate

    work_root = tmp_path / "private-run"
    execution = execute_repository_quality_pilot(
        cases=cases,
        protocol=protocol,
        executor=executor,
        work_root=work_root,
        verify_frozen_inputs=verify,
        hidden_evaluator_factory=factory,
    )

    expected_schedule = make_root_schedule(
        tuple(case.fixture.fixture_id for case in cases),
        seed=FROZEN_SEED,
    )
    assert executor.direct_calls == [
        (item.fixture_id, item.root.value == "quality_shared") for item in expected_schedule
    ]
    assert len(executor.direct_calls) == 8
    assert len(executor.recovery_calls) == 4
    assert len(set(executor.recovery_calls)) == 4
    assert execution.allocation.k == 4
    assert set(execution.allocation.static_fixture_ids) == set(execution.allocation.markov_fixture_ids)
    assert execution.generation_receipt.expected_unique_extra_generation_count == 4
    assert execution.aggregate.barrier_receipt.expected_logical_selection_count == 16
    assert execution.aggregate.barrier_receipt.hidden_evaluation_count == 8
    assert len(hidden_calls) == 8
    assert len(set(hidden_calls)) == 8
    assert [item for item in events if item == "verify"] == ["verify", "verify", "verify"]
    assert events.index("factory") > [index for index, item in enumerate(events) if item == "verify"][1]
    assert events[-1] == "verify"
    assert all(record.static_child is record.markov_child for record in execution.records)
    assert all(record.static_selection is CandidateChoice.CHILD for record in execution.records)
    assert all(record.markov_selection is CandidateChoice.CHILD for record in execution.records)
    assert dict(execution.aggregate.arm_metrics)[LogicalArm.PLAIN].passed_count == 4
    assert dict(execution.aggregate.arm_metrics)[LogicalArm.QUALITY].passed_count == 0
    assert dict(execution.aggregate.arm_metrics)[LogicalArm.MARKOV_QUALITY].passed_count == 4
    assert not work_root.exists()


def _public_state(*, quality: bool) -> PublicRepositoryState:
    validation_counts = (
        ("test", 1 if quality else 0),
        ("build", 0),
        ("static", 0),
        ("diff", 0),
        ("review", 0),
    )
    return PublicRepositoryState(
        scope_valid=True,
        public_test_attempted=quality,
        public_test_passed=quality,
        public_test_status="passed" if quality else "not_run",
        gate_present=quality,
        gate_decision="pass" if quality else "not_applicable",
        gate_phase="complete" if quality else "experiment_disabled",
        gate_satisfied=True,
        initial_snapshot_complete=True,
        current_snapshot_complete=True,
        net_workspace_changed=True,
        mutation_epoch=1 if quality else 0,
        trusted_test_or_build_attempt_count=1 if quality else 0,
        validation_counts=validation_counts,
        terminal_reason="model_final",
        budget_exhausted=False,
        deadline_violated=False,
        tool_telemetry_complete=True,
        round_count=0,
        tool_calls=0,
        output_tokens=0,
        model_seconds=0.0,
        wall_seconds=0.0,
    )


def _sealed_one_fixture_batch(
    tmp_path: Path,
) -> tuple[
    HiddenEvaluationBarrier,
    tuple[SelectedFixtureCandidates, ...],
    HiddenEvaluationBatch,
]:
    plain_root = tmp_path / "plain"
    quality_root = tmp_path / "quality"
    plain_root.mkdir()
    quality_root.mkdir()
    (plain_root / "module.py").write_text("PLAIN = 1\n")
    (quality_root / "module.py").write_text("QUALITY = 1\n")
    plain_state = _public_state(quality=False)
    quality_state = _public_state(quality=True)
    plain = CandidateObservation(
        physical_candidate_id="fixture:plain",
        terminal_artifact_id=regular_tree_sha256(plain_root),
        public_evidence=to_protocol_public_evidence(plain_state),
        cost=to_protocol_candidate_cost(plain_state),
    )
    quality = CandidateObservation(
        physical_candidate_id="fixture:quality",
        terminal_artifact_id=regular_tree_sha256(quality_root),
        public_evidence=to_protocol_public_evidence(quality_state),
        cost=to_protocol_candidate_cost(quality_state),
    )
    selected = (
        SelectedFixtureCandidates(
            fixture_id="fixture",
            plain_root=plain,
            quality_root=quality,
            static_child=None,
            markov_child=None,
            static_selection=CandidateChoice.ROOT,
            markov_selection=CandidateChoice.ROOT,
        ),
    )
    barrier = HiddenEvaluationBarrier.for_fixtures(("fixture",))
    for arm in LOGICAL_ARMS:
        root = plain_root if arm is LogicalArm.PLAIN else quality_root
        state = plain_state if arm is LogicalArm.PLAIN else quality_state
        observation = plain if arm is LogicalArm.PLAIN else quality
        barrier.register(
            logical_terminal_key("fixture", arm),
            root,
            state,
            fixture_id="fixture",
            observation=observation,
        )
    receipt = GenerationCompletionReceipt(
        fixture_count=1,
        expected_root_generation_count=2,
        completed_root_generation_count=2,
        expected_unique_extra_generation_count=0,
        completed_unique_extra_generation_count=0,
        root_schedule_sealed_before_first_generation=True,
        allocation_sealed_after_all_roots=True,
        extra_schedule_sealed_before_first_extra=True,
    )
    barrier.seal(generation_receipt=receipt)
    batch = barrier.evaluate(
        lambda _fixture_id, root: HiddenOutcome(
            evaluator_passed=root == plain_root.resolve(),
            regression_free=True,
        )
    )
    return barrier, selected, batch


@pytest.mark.parametrize("attack", ("flip", "swap"))
def test_forged_or_swapped_hidden_outcomes_cannot_enter_record_builder(
    tmp_path: Path,
    attack: str,
) -> None:
    barrier, selections, batch = _sealed_one_fixture_batch(tmp_path)
    assert bind_records_from_hidden_batch(
        barrier=barrier,
        selections=selections,
        batch=batch,
    )

    plain_key = logical_terminal_key("fixture", LogicalArm.PLAIN)
    quality_keys = tuple(logical_terminal_key("fixture", arm) for arm in LOGICAL_ARMS if arm is not LogicalArm.PLAIN)
    forged_results = dict(batch.results)
    if attack == "flip":
        forged_results[plain_key] = HiddenOutcome(False, True)
        for key in quality_keys:
            forged_results[key] = HiddenOutcome(True, True)
    else:
        plain_outcome = forged_results[plain_key]
        quality_outcome = forged_results[quality_keys[0]]
        forged_results[plain_key] = quality_outcome
        for key in quality_keys:
            forged_results[key] = plain_outcome
    forged = HiddenEvaluationBatch(
        MappingProxyType(forged_results),
        batch.terminal_bindings,
        batch.receipt,
    )

    with pytest.raises(RepositoryPilotProtocolError, match="not emitted by this barrier"):
        bind_records_from_hidden_batch(
            barrier=barrier,
            selections=selections,
            batch=forged,
        )


def test_end_to_end_api_has_no_caller_supplied_receipt_records_or_outcomes(tmp_path: Path) -> None:
    parameters = inspect.signature(execute_repository_quality_pilot).parameters
    assert "generation_receipt" not in parameters
    assert "records" not in parameters
    assert "outcomes" not in parameters

    forged_receipt = GenerationCompletionReceipt(
        fixture_count=4,
        expected_root_generation_count=8,
        completed_root_generation_count=8,
        expected_unique_extra_generation_count=0,
        completed_unique_extra_generation_count=0,
        root_schedule_sealed_before_first_generation=True,
        allocation_sealed_after_all_roots=True,
        extra_schedule_sealed_before_first_extra=True,
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        execute_repository_quality_pilot(  # type: ignore[call-arg]
            cases=select_cases("smoke"),
            protocol=_smoke_protocol(),
            executor=_FakeRetainedExecutor(),
            work_root=tmp_path / "never-created",
            verify_frozen_inputs=lambda: None,
            hidden_evaluator_factory=lambda: lambda _fixture_id, _root: HiddenOutcome(False, True),
            generation_receipt=forged_receipt,
        )
    assert not (tmp_path / "never-created").exists()


def test_pre_hidden_verification_failure_aborts_cleans_and_never_constructs_evaluator(
    tmp_path: Path,
) -> None:
    verification_count = 0
    factory_called = False

    def verify() -> None:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            raise RuntimeError("frozen input drift")

    def factory():
        nonlocal factory_called
        factory_called = True
        return lambda _fixture_id, _root: HiddenOutcome(False, True)

    work_root = tmp_path / "aborted-run"
    orchestrator = RepositoryQualityPilotOrchestrator(
        cases=select_cases("smoke"),
        protocol=_smoke_protocol(),
        executor=_FakeRetainedExecutor(),
        work_root=work_root,
        verify_frozen_inputs=verify,
        hidden_evaluator_factory=factory,
    )
    with pytest.raises(RuntimeError, match="frozen input drift"):
        orchestrator.run()

    assert orchestrator.phase is PilotRunPhase.ABORTED
    assert verification_count == 2
    assert factory_called is False
    assert not work_root.exists()


def test_hidden_evaluator_wrong_type_aborts_without_aggregate_and_cleans(tmp_path: Path) -> None:
    work_root = tmp_path / "wrong-hidden-type"
    orchestrator = RepositoryQualityPilotOrchestrator(
        cases=select_cases("smoke"),
        protocol=_smoke_protocol(),
        executor=_FakeRetainedExecutor(),
        work_root=work_root,
        verify_frozen_inputs=lambda: None,
        hidden_evaluator_factory=lambda: lambda _fixture_id, _root: object(),  # type: ignore[return-value]
    )
    with pytest.raises(RepositoryPilotProtocolError, match="exact HiddenOutcome"):
        orchestrator.run()

    assert orchestrator.phase is PilotRunPhase.ABORTED
    assert not work_root.exists()


def test_batch_identity_cannot_be_replaced_even_with_identical_values(tmp_path: Path) -> None:
    barrier, selections, batch = _sealed_one_fixture_batch(tmp_path)
    copied = replace(batch)
    assert copied == batch
    assert copied is not batch

    with pytest.raises(RepositoryPilotProtocolError, match="not emitted by this barrier"):
        bind_records_from_hidden_batch(
            barrier=barrier,
            selections=selections,
            batch=copied,
        )


def test_preregistration_v3_exact_bytes_and_source_scope_are_sealed(tmp_path: Path) -> None:
    assert pilot_runner._assert_preregistration_seal() == pilot_runner.PREREGISTRATION_SHA256
    document = json.loads(
        (pilot_runner._REPOSITORY_ROOT / pilot_runner.PREREGISTRATION_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    integrity = document["protocol_integrity"]
    revision = document["revision_history"]
    assert tuple(integrity["source_lock_must_include"]) == pilot_runner.PILOT_SOURCE_LOCK_FILES
    assert integrity["result_envelope_schema"] == pilot_runner.RESULT_ENVELOPE_SCHEMA
    assert integrity["abort_envelope_schema"] == pilot_runner.ABORT_ENVELOPE_SCHEMA
    assert integrity["attempt_start_schema"] == pilot_runner.ATTEMPT_START_SCHEMA
    assert integrity["source_lock_schema"] == pilot_runner.SOURCE_LOCK_SCHEMA
    assert revision["predecessor_sha256"] == pilot_runner.PREDECESSOR_PREREGISTRATION_SHA256
    assert revision["post_hoc_incident_record_sha256"] == pilot_runner.V2_INCIDENT_SHA256
    pilot_runner._assert_predecessor_and_incident_seals()

    tampered = tmp_path / "tampered-preregistration.json"
    source = (pilot_runner._REPOSITORY_ROOT / pilot_runner.PREREGISTRATION_RELATIVE_PATH).read_bytes()
    tampered.write_bytes(source + b"\n")
    with pytest.raises(RepositoryPilotProtocolError, match="frozen SHA-256"):
        pilot_runner._assert_preregistration_seal(tampered)


class _FakeNativeManager:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.unload_count = 0

    def unload_all(self) -> None:
        self.unload_count += 1
        self.events.append("manager_unloaded")


def _install_fake_native_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    retained_executor: object | None = None,
) -> SimpleNamespace:
    """Install a source-free native harness without loading an MLX model."""

    target = tmp_path / "target-model"
    draft = tmp_path / "draft-model"
    target.mkdir()
    draft.mkdir()
    events: list[str] = []
    load_calls: list[dict[str, object]] = []
    verification_counts = {"source": 0, "models": 0, "runtime": 0}
    manager = _FakeNativeManager(events)
    executor = retained_executor or _FakeRetainedExecutor()
    source_lock = CleanSourceLock(
        repo_root=pilot_runner._REPOSITORY_ROOT,
        git_revision="a" * 40,
        source_sha256="b" * 64,
        source_file_count=len(pilot_runner.PILOT_SOURCE_LOCK_FILES),
    )
    model_locks = (
        LocalModelLock(
            role="target",
            repository_label=TARGET_REPOSITORY_LABEL,
            content_identity=TARGET_CONTENT_IDENTITY,
            resolved_path=target,
        ),
        LocalModelLock(
            role="drafter",
            repository_label=DRAFT_REPOSITORY_LABEL,
            content_identity=DRAFT_CONTENT_IDENTITY,
            resolved_path=draft,
        ),
    )
    runtime = RuntimeIdentity(
        python_version=platform.python_version(),
        software_versions=FROZEN_SOFTWARE_VERSIONS,
        hardware_label="darwin-arm64-fake-10cpu-1b",
    )

    def capture_source(repo_root: Path, *, source_files: tuple[str, ...]) -> CleanSourceLock:
        assert repo_root == pilot_runner._REPOSITORY_ROOT
        assert source_files == pilot_runner.PILOT_SOURCE_LOCK_FILES
        events.append("source_captured")
        return source_lock

    def verify_source(
        observed: CleanSourceLock,
        *,
        source_files: tuple[str, ...],
    ) -> None:
        assert observed is source_lock
        assert source_files == pilot_runner.PILOT_SOURCE_LOCK_FILES
        verification_counts["source"] += 1

    def verify_models(observed: object) -> None:
        assert observed is model_locks
        verification_counts["models"] += 1

    def verify_runtime_identity(observed: object) -> None:
        assert observed is runtime
        verification_counts["runtime"] += 1

    native_shell = SimpleNamespace(
        config=object(),
        manager=manager,
        engine=object(),
        tier="small",
    )

    def load_native(**kwargs: object):
        load_calls.append(dict(kwargs))
        events.append("model_loaded")
        return native_shell, manager

    class FakeCorpusHiddenEvaluator:
        def __init__(self, _cases: object) -> None:
            assert manager.unload_count == 1
            events.append("hidden_wrapper_created")

        def __call__(self, request: object) -> HiddenEvaluation:
            assert manager.unload_count == 1
            workspace = getattr(request, "workspace")
            repaired = "# repaired by fake executor" in _editable_module(workspace).read_text(encoding="utf-8")
            return HiddenEvaluation(passed=repaired, regression_free=True)

    monkeypatch.setattr(pilot_runner, "_assert_frozen_environment", lambda: None)
    monkeypatch.setattr(pilot_runner, "_assert_gate_profile_seal", lambda: None)
    monkeypatch.setattr(
        pilot_runner,
        "_assert_preregistration_seal",
        lambda _path=None: pilot_runner.PREREGISTRATION_SHA256,
    )
    monkeypatch.setattr(pilot_runner, "capture_clean_source_lock", capture_source)
    monkeypatch.setattr(pilot_runner, "collect_runtime_identity", lambda: runtime)
    monkeypatch.setattr(pilot_runner, "bind_frozen_local_models", lambda _target, _draft: model_locks)
    monkeypatch.setattr(pilot_runner, "verify_clean_source_lock", verify_source)
    monkeypatch.setattr(pilot_runner, "verify_frozen_local_models", verify_models)
    monkeypatch.setattr(pilot_runner, "verify_runtime_identity", verify_runtime_identity)
    monkeypatch.setattr(pilot_runner, "_load_native_executor", load_native)
    monkeypatch.setattr(pilot_runner, "RetainedNativeAgentExecutor", lambda **_kwargs: executor)
    monkeypatch.setattr(pilot_runner, "CorpusHiddenEvaluator", FakeCorpusHiddenEvaluator)
    return SimpleNamespace(
        target=target,
        draft=draft,
        events=events,
        load_calls=load_calls,
        verification_counts=verification_counts,
        manager=manager,
        executor=executor,
        source_lock=source_lock,
        model_locks=model_locks,
        runtime=runtime,
    )


def test_native_wrapper_attests_unloads_before_hidden_and_emits_source_free_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_fake_native_runtime(tmp_path, monkeypatch)
    work_root = tmp_path / "native-private-run"
    attempt_root = tmp_path / "native-attempt"
    envelope = pilot_runner.run_native_repository_quality_pilot(
        split="smoke",
        tier="small",
        config_path=None,
        target_path=harness.target,
        draft_path=harness.draft,
        work_root=work_root,
        attempt_root=attempt_root,
    )

    assert harness.manager.unload_count == 1
    assert harness.events.index("manager_unloaded") < harness.events.index("hidden_wrapper_created")
    assert envelope.publication_receipt.successful_verifications == 4
    assert envelope.publication_receipt.manager_unloaded_before_hidden is True
    assert harness.verification_counts == {"source": 7, "models": 7, "runtime": 7}
    assert not work_root.exists()
    start_bytes = (attempt_root / pilot_runner._ATTEMPT_START_FILENAME).read_bytes()
    result_bytes = (attempt_root / pilot_runner._ATTEMPT_RESULT_FILENAME).read_bytes()
    assert not (attempt_root / pilot_runner._ATTEMPT_ABORT_FILENAME).exists()

    serialized = pilot_runner.serialize_repository_quality_pilot_result(envelope)
    payload = json.loads(serialized)
    assert payload["schema_version"] == pilot_runner.RESULT_ENVELOPE_SCHEMA
    assert payload["attempt"]["start_sha256"] == hashlib.sha256(start_bytes).hexdigest()
    assert payload["protocol"]["preregistration_sha256"] == pilot_runner.PREREGISTRATION_SHA256
    assert payload["protocol"]["private_evaluator_bundle_sha256"]
    assert payload["implementation"]["source_file_count"] == len(pilot_runner.PILOT_SOURCE_LOCK_FILES)
    assert payload["runtime"]["sampler_seed"] is None
    assert payload["runtime"]["network_enabled"] is False
    assert payload["hidden_labels_serialized"] is False
    assert str(tmp_path) not in serialized
    assert '"fixture_id"' not in serialized
    assert result_bytes.decode("utf-8") == serialized
    assert harness.verification_counts == {"source": 8, "models": 8, "runtime": 8}

    copied_execution = replace(envelope.execution)
    with pytest.raises(ValueError, match="another execution"):
        pilot_runner.RepositoryQualityPilotResultEnvelope(
            provenance=envelope.provenance,
            attempt_start=envelope.attempt_start,
            execution=copied_execution,
            publication_receipt=envelope.publication_receipt,
            post_run_verified=True,
        )
    with pytest.raises(TypeError, match="native publication receipt"):
        pilot_runner.RepositoryQualityPilotResultEnvelope(
            provenance=envelope.provenance,
            attempt_start=envelope.attempt_start,
            execution=envelope.execution,
            publication_receipt=object(),  # type: ignore[arg-type]
            post_run_verified=True,
        )


class _DerivedTimingExecutor(_FakeRetainedExecutor):
    def run_direct(self, **kwargs: object) -> RetainedAgentStage:
        stage = super().run_direct(**kwargs)  # type: ignore[arg-type]
        derived_round = replace(stage.result.rounds[0], timing_source="derived_legacy_us")
        return replace(stage, result=replace(stage.result, rounds=(derived_round,)))


class _LeakyFailureExecutor(_FakeRetainedExecutor):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    def run_direct(self, **_kwargs: object) -> RetainedAgentStage:
        raise RuntimeError(self.secret)


def _run_fake_native(harness: SimpleNamespace, *, work_root: Path, attempt_root: Path):
    return pilot_runner.run_native_repository_quality_pilot(
        split="smoke",
        tier="small",
        config_path=None,
        target_path=harness.target,
        draft_path=harness.draft,
        work_root=work_root,
        attempt_root=attempt_root,
    )


def test_derived_legacy_telemetry_emits_abort_no_result_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_fake_native_runtime(
        tmp_path,
        monkeypatch,
        retained_executor=_DerivedTimingExecutor(),
    )
    work_root = tmp_path / "private-run"
    attempt_root = tmp_path / "attempt"

    with pytest.raises(pilot_runner.NativePilotAborted) as raised:
        _run_fake_native(harness, work_root=work_root, attempt_root=attempt_root)

    assert str(raised.value) == "native pilot aborted: dflash_raw_phase_telemetry_missing"
    start_bytes = (attempt_root / pilot_runner._ATTEMPT_START_FILENAME).read_bytes()
    abort_text = (attempt_root / pilot_runner._ATTEMPT_ABORT_FILENAME).read_text(encoding="utf-8")
    payload = json.loads(abort_text)
    assert not (attempt_root / pilot_runner._ATTEMPT_RESULT_FILENAME).exists()
    assert payload["schema_version"] == pilot_runner.ABORT_ENVELOPE_SCHEMA
    assert payload["status"] == "aborted_no_result"
    assert payload["attempt"]["start_sha256"] == hashlib.sha256(start_bytes).hexdigest()
    assert payload["abort"]["failure_boundary"] == "root_generation_telemetry_validation"
    assert payload["abort"]["reason_code"] == "dflash_raw_phase_telemetry_missing"
    assert payload["abort"]["generation_receipt_issued"] is False
    assert payload["abort"]["completed_root_generation_count"] == 0
    assert payload["abort"]["completed_unique_extra_generation_count"] == 0
    assert payload["abort"]["hidden_evaluator_constructed"] is False
    assert payload["abort"]["hidden_evaluation_started"] is False
    assert payload["abort"]["manager_state"] == "unloaded"
    assert payload["abort"]["work_root_cleanup"] == "complete"
    assert payload["abort"]["post_abort_frozen_input_verification"] == "passed"
    assert payload["abort"]["aggregate_produced"] is False
    assert payload["publication"] == {
        "breakthrough_claim_allowed": False,
        "partial_generation_reuse_allowed": False,
        "quality_claim_allowed": False,
        "result_envelope_created": False,
        "speed_claim_allowed": False,
    }
    assert payload["hidden_labels_serialized"] is False
    assert not work_root.exists()
    assert harness.manager.unload_count == 1


def test_pre_attempt_integrity_failure_creates_no_start_and_loads_no_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_fake_native_runtime(tmp_path, monkeypatch)
    attempt_root = tmp_path / "attempt"

    def fail_source_verification(*_args: object, **_kwargs: object) -> None:
        raise RepositoryPilotProtocolError("pre-attempt source drift")

    monkeypatch.setattr(pilot_runner, "verify_clean_source_lock", fail_source_verification)
    with pytest.raises(RepositoryPilotProtocolError, match="pre-attempt source drift"):
        _run_fake_native(
            harness,
            work_root=tmp_path / "private-run",
            attempt_root=attempt_root,
        )

    assert not attempt_root.exists()
    assert harness.load_calls == []
    assert harness.manager.unload_count == 0


def test_abort_serialization_digests_exception_without_path_or_prompt_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_prompt = f"PROMPT_SECRET::{tmp_path / 'private' / 'candidate.py'}"
    harness = _install_fake_native_runtime(
        tmp_path,
        monkeypatch,
        retained_executor=_LeakyFailureExecutor(secret_prompt),
    )
    attempt_root = tmp_path / "attempt"

    with pytest.raises(pilot_runner.NativePilotAborted) as raised:
        _run_fake_native(
            harness,
            work_root=tmp_path / "private-run",
            attempt_root=attempt_root,
        )

    serialized = (attempt_root / pilot_runner._ATTEMPT_ABORT_FILENAME).read_text(encoding="utf-8")
    payload = json.loads(serialized)
    assert secret_prompt not in serialized
    assert str(tmp_path) not in serialized
    assert secret_prompt not in str(raised.value)
    assert payload["abort"]["failure_message_sha256"] == hashlib.sha256(secret_prompt.encode("utf-8")).hexdigest()
    assert payload["abort"]["reason_code"] == "infrastructure_failure"


def test_success_terminalization_failure_emits_abort_instead_of_stranding_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_fake_native_runtime(tmp_path, monkeypatch)
    attempt_root = tmp_path / "attempt"

    def fail_result_serialization(_envelope: object, *, indent: int | None = 2) -> str:
        del indent
        raise RuntimeError(f"terminal-secret::{tmp_path}")

    monkeypatch.setattr(
        pilot_runner,
        "serialize_repository_quality_pilot_result",
        fail_result_serialization,
    )
    with pytest.raises(pilot_runner.NativePilotAborted):
        _run_fake_native(
            harness,
            work_root=tmp_path / "private-run",
            attempt_root=attempt_root,
        )

    assert (attempt_root / pilot_runner._ATTEMPT_START_FILENAME).is_file()
    assert (attempt_root / pilot_runner._ATTEMPT_ABORT_FILENAME).is_file()
    assert not (attempt_root / pilot_runner._ATTEMPT_RESULT_FILENAME).exists()
    serialized = (attempt_root / pilot_runner._ATTEMPT_ABORT_FILENAME).read_text(encoding="utf-8")
    payload = json.loads(serialized)
    assert str(tmp_path) not in serialized
    assert payload["abort"]["failure_boundary"] == "result_terminalization"
    assert payload["abort"]["reason_code"] == "infrastructure_failure"
    assert payload["abort"]["manager_state"] == "unloaded"
    assert payload["abort"]["work_root_cleanup"] == "complete"


def test_start_publication_failure_after_link_emits_abort_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_fake_native_runtime(tmp_path, monkeypatch)
    attempt_root = tmp_path / "attempt"
    atomic_create = pilot_runner._atomic_create_result

    def fail_after_start_link(path: Path, content: str) -> None:
        atomic_create(path, content)
        if path.name == pilot_runner._ATTEMPT_START_FILENAME:
            raise OSError("simulated post-link directory-fsync failure")

    monkeypatch.setattr(pilot_runner, "_atomic_create_result", fail_after_start_link)
    with pytest.raises(pilot_runner.NativePilotAborted):
        _run_fake_native(
            harness,
            work_root=tmp_path / "private-run",
            attempt_root=attempt_root,
        )

    start_bytes = (attempt_root / pilot_runner._ATTEMPT_START_FILENAME).read_bytes()
    abort_text = (attempt_root / pilot_runner._ATTEMPT_ABORT_FILENAME).read_text(encoding="utf-8")
    payload = json.loads(abort_text)
    assert not (attempt_root / pilot_runner._ATTEMPT_RESULT_FILENAME).exists()
    assert payload["attempt"]["start_sha256"] == hashlib.sha256(start_bytes).hexdigest()
    assert payload["abort"]["failure_boundary"] == "attempt_start_publication"
    assert payload["abort"]["reason_code"] == "infrastructure_failure"
    assert payload["abort"]["manager_state"] == "never_loaded"
    assert payload["abort"]["hidden_evaluator_constructed"] is False
    assert payload["abort"]["hidden_evaluation_started"] is False
    assert harness.load_calls == []
    assert harness.manager.unload_count == 0


def test_second_use_of_same_attempt_root_fails_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_fake_native_runtime(tmp_path, monkeypatch)
    attempt_root = tmp_path / "attempt"
    work_root = tmp_path / "private-run"
    _run_fake_native(harness, work_root=work_root, attempt_root=attempt_root)
    original_files = {path.name: path.read_bytes() for path in attempt_root.iterdir()}

    with pytest.raises(RepositoryPilotProtocolError, match="create-once"):
        _run_fake_native(harness, work_root=work_root, attempt_root=attempt_root)

    assert len(harness.load_calls) == 1
    assert {path.name: path.read_bytes() for path in attempt_root.iterdir()} == original_files


def test_native_work_locations_fail_fast_on_source_or_model_overlap(tmp_path: Path) -> None:
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    locks = (
        LocalModelLock(
            "target",
            TARGET_REPOSITORY_LABEL,
            TARGET_CONTENT_IDENTITY,
            target,
        ),
        LocalModelLock(
            "drafter",
            DRAFT_REPOSITORY_LABEL,
            DRAFT_CONTENT_IDENTITY,
            draft,
        ),
    )
    with pytest.raises(RepositoryPilotProtocolError, match="disjoint"):
        pilot_runner._validate_native_work_location(
            pilot_runner._REPOSITORY_ROOT / "private-run",
            source_root=pilot_runner._REPOSITORY_ROOT,
            model_locks=locks,
            label="pilot work root",
            must_exist=False,
        )
    with pytest.raises(RepositoryPilotProtocolError, match="disjoint"):
        pilot_runner._validate_native_work_location(
            tmp_path,
            source_root=pilot_runner._REPOSITORY_ROOT,
            model_locks=locks,
            label="work parent",
            must_exist=True,
        )
    safe = tmp_path.parent / f"{tmp_path.name}-safe-run"
    assert pilot_runner._validate_native_work_location(
        safe,
        source_root=pilot_runner._REPOSITORY_ROOT,
        model_locks=locks,
        label="pilot work root",
        must_exist=False,
    ) == safe.resolve(strict=False)


def test_result_output_is_atomic_create_once_and_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    for directory in (source, target, draft):
        directory.mkdir()
    locks = (
        LocalModelLock(
            "target",
            TARGET_REPOSITORY_LABEL,
            TARGET_CONTENT_IDENTITY,
            target,
        ),
        LocalModelLock(
            "drafter",
            DRAFT_REPOSITORY_LABEL,
            DRAFT_CONTENT_IDENTITY,
            draft,
        ),
    )
    output = tmp_path / "result.json"
    assert pilot_runner._validate_new_output_path(
        output,
        source_root=source,
        model_locks=locks,
    ) == output.resolve(strict=False)
    pilot_runner._atomic_create_result(output, "first\n")
    assert output.read_text(encoding="utf-8") == "first\n"

    with pytest.raises(RepositoryPilotProtocolError, match="create-once"):
        pilot_runner._validate_new_output_path(
            output,
            source_root=source,
            model_locks=locks,
        )
    with pytest.raises(RepositoryPilotProtocolError, match="create-once"):
        pilot_runner._atomic_create_result(output, "replacement\n")
    assert output.read_text(encoding="utf-8") == "first\n"


def test_native_cli_exposes_no_quality_or_protocol_knobs() -> None:
    parameters = inspect.signature(pilot_runner.run_native_repository_quality_pilot).parameters
    assert "verify_frozen_inputs" not in parameters
    assert "seed" not in parameters
    assert "effort" not in parameters
    assert "router" not in parameters
    args = pilot_runner._parse_args(
        [
            "--target-path",
            "/tmp/target",
            "--draft-path",
            "/tmp/draft",
            "--attempt-root",
            "/tmp/attempt",
        ]
    )
    assert args.split == "smoke"
    assert args.attempt_root == Path("/tmp/attempt")

    with pytest.raises(SystemExit):
        pilot_runner._parse_args(
            [
                "--target-path",
                "/tmp/target",
                "--draft-path",
                "/tmp/draft",
                "--attempt-root",
                "/tmp/attempt",
                "--effort",
                "high",
            ]
        )
    with pytest.raises(SystemExit):
        pilot_runner._parse_args(
            [
                "--split",
                "all",
                "--target-path",
                "/tmp/target",
                "--draft-path",
                "/tmp/draft",
                "--attempt-root",
                "/tmp/attempt",
            ]
        )
