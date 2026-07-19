from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experimental.effort.bench_repository_quality_pilot import (
    DIRECT_EXECUTION_BUDGET,
    EXTRA_EXECUTION_BUDGET,
    FROZEN_RUNNER_SETTINGS,
    FORBIDDEN_ENVIRONMENT_OVERRIDES,
    RECOVERY_PROMPT,
    RESET_MANIFEST,
    HiddenEvaluationBarrier,
    HiddenEvaluationBatch,
    ImmutableWorkspaceArchive,
    PublicRepositoryState,
    PublicScopeContract,
    PublicScopeVerdict,
    RepositoryPilotProtocolError,
    RetainedAgentStage,
    RetainedNativeAgentExecutor,
    SelectedFixtureCandidates,
    VisiblePublicTestResult,
    extract_public_repository_state,
    prefer_recovery_publicly,
    prepare_pristine_direct_roots,
    public_state_json,
    regular_tree_sha256,
    safe_clone_workspace,
    logical_terminal_key,
    to_protocol_candidate_cost,
    to_protocol_public_evidence,
)
from experimental.effort.repository_quality_pilot import (
    CandidateChoice,
    CandidateObservation,
    EvaluationBarrierReceipt,
    FixturePilotRecord,
    GenerationCompletionReceipt,
    HiddenOutcome,
    LOGICAL_ARMS,
    LogicalArm,
    PublicState,
    VisibleCheckOutcome,
)
from mio.agent import (
    AgentExecutionBudget,
    AgentRoundTrace,
    AgentToolTrace,
    AgentTurnResult,
)
from mio.agent_policy import (
    AgentPathViolation,
    AgentToolPermission,
    AgentToolPolicy,
    resolve_workspace_path,
)
from mio.coding_quality import (
    CodingEffort,
    CodingQualityGate,
    RequestIntent,
    ValidationEvidence,
    ValidationKind,
    snapshot_workspaces,
)


def _write_fixture(root: Path, text: str = "VALUE = 1\n") -> Path:
    root.mkdir(parents=True)
    (root / "module.py").write_text(text)
    (root / "tests").mkdir()
    (root / "tests" / "test_module.py").write_text("from module import VALUE\n")
    return root


def _result(
    *,
    report: dict[str, object] | None,
    terminal_reason: str = "model_final",
) -> AgentTurnResult:
    return AgentTurnResult(
        assistant_text="never serialize this assistant text",
        quality_gate=report,
        terminal_reason=terminal_reason,
        budget_exhaustion=None,
        tool_telemetry_complete=True,
        tool_calls=0,
        tool_events=(),
        tool_result_chars=0,
        wall_time_s=1.5,
        rounds=(),
        completion_tokens=0,
    )


def _fake_agent(process):
    names = ("validate", "bash", "read", "write", "edit")
    registry = {name: {"fn": lambda: None, "args": []} for name in names}
    specs = [{"type": "function", "function": {"name": name, "parameters": {}}} for name in names]
    return SimpleNamespace(
        AGENT_TOOLS=registry,
        AGENT_TOOLS_SPEC=specs,
        PromptPolicy=lambda: SimpleNamespace(label="test"),
        console=object(),
        _process_user_input=process,
    )


class _FakeEngine:
    def __init__(self) -> None:
        self.reset_count = 0
        self.tier_config = SimpleNamespace(**dict(FROZEN_RUNNER_SETTINGS))
        self.drafter_status = {
            "requested": "dflash",
            "selected": "dflash",
            "ref": "test/dflash",
            "fallback_used": False,
            "strict": True,
        }
        self._prefix_cache = []
        self._last_prompt_tokens = [1]
        self._pending_assistant_prefill = "stale"
        self._dspark_runtime = None

    def _prefix_cache_invalidate(self) -> None:
        self.reset_count += 1


class _FakeManager:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine

    def get_engine(self, _tier: str) -> _FakeEngine:
        return self.engine


def _record_gate_change_and_validations(
    gate: CodingQualityGate,
    root: Path,
    *,
    value: int,
    kinds: tuple[ValidationKind, ...],
) -> None:
    (root / "module.py").write_text(f"VALUE = {value}\n")
    current = gate.refresh()
    for kind in kinds:
        gate.record_validation(
            kind,
            command_sha256=(str(kind.value).encode().hex() + "0" * 64)[:64],
            allowed=True,
            outcome="ok",
            snapshot=current,
        )
        gate.validate_invocations += 1


def _round_trace(*, completion_tokens: int = 7, deadline_hit: bool = False) -> AgentRoundTrace:
    return AgentRoundTrace(
        round_index=0,
        prompt_tokens=10,
        completion_tokens=completion_tokens,
        total_time_s=1.0,
        prompt_tps=10.0,
        generation_tps=7.0,
        generation_backend="dflash",
        fallback_ar=False,
        prefill_ns=400_000_000,
        decode_ns=600_000_000,
        model_total_ns=1_000_000_000,
        logical_prompt_tokens=10,
        physical_prefill_tokens=10,
        physical_decode_tokens=completion_tokens,
        warm_offset=0,
        warm_offset_tokens=0,
        timing_source="runtime_raw_ns",
        drafter_requested="dflash",
        drafter_selected="dflash",
        drafter_ref="test/dflash",
        deadline_hit=deadline_hit,
    )


def _tool_trace(
    sequence: int,
    *,
    allowed: bool = True,
    outcome: str = "ok",
    telemetry_complete: bool = True,
) -> AgentToolTrace:
    return AgentToolTrace(
        sequence=sequence,
        round_index=0,
        tool_name="validate",
        operation="validate",
        permission="shell",
        allowed=allowed,
        outcome=outcome,
        target_sha256=f"{sequence + 1:064x}",
        duration_ns=1,
        output_chars=1,
        audit_count=1,
        audit_sha256=f"{sequence + 2:064x}",
        telemetry_complete=telemetry_complete,
    )


def _quality_stage(
    tmp_path: Path,
    *,
    validation_specs: tuple[tuple[ValidationKind, bool, str], ...] = (
        (ValidationKind.TEST, True, "ok"),
        (ValidationKind.STATIC, True, "ok"),
    ),
    stale_validation_specs: tuple[tuple[ValidationKind, bool, str], ...] = (),
    mutate_noneditable: bool = False,
) -> tuple[RetainedAgentStage, PublicScopeContract, PublicScopeVerdict]:
    root = _write_fixture(tmp_path / "workspace")
    contract = PublicScopeContract.capture(
        "fixture-1",
        root,
        editable_names=("module.py",),
    )
    pristine_tree = regular_tree_sha256(root)
    pristine = snapshot_workspaces((root,))
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
    if stale_validation_specs:
        (root / "module.py").write_text("VALUE = 9\n")
        stale = gate.refresh()
        for kind, allowed, outcome in stale_validation_specs:
            gate.record_validation(
                kind,
                command_sha256=f"{len(gate.validations) + 10:064x}",
                allowed=allowed,
                outcome=outcome,
                snapshot=stale,
            )
            gate.validate_invocations += 1
    (root / "module.py").write_text("VALUE = 2\n")
    if mutate_noneditable:
        (root / "tests" / "test_module.py").write_text("raise AssertionError('tampered')\n")
    current = gate.refresh()
    for kind, allowed, outcome in validation_specs:
        gate.record_validation(
            kind,
            command_sha256=f"{len(gate.validations) + 10:064x}",
            allowed=allowed,
            outcome=outcome,
            snapshot=current,
        )
        gate.validate_invocations += 1
    terminal_tree = regular_tree_sha256(root)
    all_specs = (*stale_validation_specs, *validation_specs)
    traces = tuple(
        _tool_trace(index, allowed=allowed, outcome=outcome)
        for index, (_kind, allowed, outcome) in enumerate(all_specs)
    )
    result = AgentTurnResult(
        assistant_text="private assistant output",
        terminal_reason="model_final",
        rounds=(_round_trace(),),
        tool_events=traces,
        tool_calls=len(traces),
        tool_result_chars=len(traces),
        wall_time_s=1.5,
        quality_gate=gate.report(),
        completion_tokens=7,
        budget_exhaustion=None,
        tool_telemetry_complete=True,
    )
    agent_surface = _fake_agent(lambda *_args: None)
    state = {
        "tool_policy": AgentToolPolicy.coding_workspace(root, allow_network=False),
        "tool_registry": agent_surface.AGENT_TOOLS,
        "tool_specs": tuple(agent_surface.AGENT_TOOLS_SPEC),
        "execution_budget": DIRECT_EXECUTION_BUDGET,
        "coding_effort": CodingEffort.MEDIUM.value,
        "quality_gate_enabled": True,
        "quality_gate_require_change": True,
    }
    stage = RetainedAgentStage(
        stage="direct",
        fixture_id="fixture-1",
        instruction="secret prompt",
        workspace=root,
        pristine_tree_sha256=pristine_tree,
        terminal_tree_sha256=terminal_tree,
        pristine_snapshot=pristine,
        current_snapshot=current,
        execution_budget=DIRECT_EXECUTION_BUDGET,
        coding_effort=CodingEffort.MEDIUM.value,
        drafter_ref="test/dflash",
        state=state,
        result=result,
        trusted_quality_gate=gate,
    )
    verdict = contract.assess_terminal(
        fixture_id=stage.fixture_id,
        terminal_root=root,
        terminal_tree_sha256=terminal_tree,
    )
    return stage, contract, verdict


def _plain_stage(
    tmp_path: Path,
    *,
    fixture_id: str = "fixture-plain",
    value: int = 2,
) -> tuple[RetainedAgentStage, PublicScopeContract, PublicScopeVerdict]:
    root = _write_fixture(tmp_path / "plain-workspace")
    contract = PublicScopeContract.capture(
        fixture_id,
        root,
        editable_names=("module.py",),
    )
    pristine_tree = regular_tree_sha256(root)
    pristine = snapshot_workspaces((root,))
    (root / "module.py").write_text(f"VALUE = {value}\n")
    current = snapshot_workspaces((root,))
    terminal_tree = regular_tree_sha256(root)
    result = AgentTurnResult(
        assistant_text="private plain output",
        terminal_reason="model_final",
        rounds=(_round_trace(),),
        tool_events=(),
        tool_calls=0,
        tool_result_chars=0,
        wall_time_s=1.5,
        quality_gate=None,
        completion_tokens=7,
        budget_exhaustion=None,
        tool_telemetry_complete=True,
    )
    agent_surface = _fake_agent(lambda *_args: None)
    plain_names = ("bash", "read", "write", "edit")
    registry = {name: agent_surface.AGENT_TOOLS[name] for name in plain_names}
    specs = tuple(spec for spec in agent_surface.AGENT_TOOLS_SPEC if spec["function"]["name"] in plain_names)
    stage = RetainedAgentStage(
        stage="direct",
        fixture_id=fixture_id,
        instruction="plain prompt",
        workspace=root,
        pristine_tree_sha256=pristine_tree,
        terminal_tree_sha256=terminal_tree,
        pristine_snapshot=pristine,
        current_snapshot=current,
        execution_budget=DIRECT_EXECUTION_BUDGET,
        coding_effort=CodingEffort.MEDIUM.value,
        drafter_ref="test/dflash",
        state={
            "tool_policy": AgentToolPolicy.coding_workspace(root, allow_network=False),
            "tool_registry": registry,
            "tool_specs": specs,
            "execution_budget": DIRECT_EXECUTION_BUDGET,
            "coding_effort": CodingEffort.MEDIUM.value,
            "quality_gate_enabled": False,
            "quality_gate_require_change": False,
        },
        result=result,
        trusted_quality_gate=None,
        quality_enabled=False,
    )
    verdict = contract.assess_terminal(
        fixture_id=stage.fixture_id,
        terminal_root=root,
        terminal_tree_sha256=terminal_tree,
    )
    return stage, contract, verdict


def _public_state(**changes) -> PublicRepositoryState:
    base = PublicRepositoryState(
        scope_valid=True,
        public_test_attempted=True,
        public_test_passed=True,
        public_test_status="passed",
        gate_present=True,
        gate_decision="pass",
        gate_phase="passed",
        gate_satisfied=True,
        initial_snapshot_complete=True,
        current_snapshot_complete=True,
        net_workspace_changed=True,
        mutation_epoch=1,
        trusted_test_or_build_attempt_count=1,
        validation_counts=(("test", 1), ("build", 0), ("static", 1), ("diff", 0), ("review", 0)),
        terminal_reason="model_final",
        budget_exhausted=False,
        deadline_violated=False,
        tool_telemetry_complete=True,
        round_count=1,
        tool_calls=2,
        output_tokens=7,
        model_seconds=1.0,
        wall_seconds=1.5,
    )
    return replace(base, **changes)


def _generation_receipt(
    *,
    fixture_count: int = 1,
    unique_extra_count: int = 0,
) -> GenerationCompletionReceipt:
    return GenerationCompletionReceipt(
        fixture_count=fixture_count,
        expected_root_generation_count=fixture_count * 2,
        completed_root_generation_count=fixture_count * 2,
        expected_unique_extra_generation_count=unique_extra_count,
        completed_unique_extra_generation_count=unique_extra_count,
        root_schedule_sealed_before_first_generation=True,
        allocation_sealed_after_all_roots=True,
        extra_schedule_sealed_before_first_extra=True,
    )


def _register_test_terminal(
    barrier: HiddenEvaluationBarrier,
    key: str,
    root: Path,
    *,
    fixture_id: str = "fixture-1",
    physical_candidate_id: str = "test-physical",
    public_state: PublicRepositoryState | None = None,
) -> CandidateObservation:
    state = _public_state() if public_state is None else public_state
    observation = CandidateObservation(
        physical_candidate_id=physical_candidate_id,
        terminal_artifact_id=regular_tree_sha256(root),
        public_evidence=to_protocol_public_evidence(state),
        cost=to_protocol_candidate_cost(state),
    )
    barrier.register(
        key,
        root,
        state,
        fixture_id=fixture_id,
        observation=observation,
    )
    return observation


def test_execution_budgets_are_exact_frozen_and_stage_owned() -> None:
    assert DIRECT_EXECUTION_BUDGET == AgentExecutionBudget(12, 32, 2_048, 120.0, 8_192)
    assert EXTRA_EXECUTION_BUDGET == AgentExecutionBudget(4, 8, 384, 20.0, 8_192)
    with pytest.raises(FrozenInstanceError):
        DIRECT_EXECUTION_BUDGET.max_rounds = 99  # type: ignore[misc]


def test_runtime_and_scope_constants_align_with_preregistration() -> None:
    repository = Path(__file__).resolve().parents[2]
    preregistration = json.loads(
        (repository / "benchmarks" / "repository-quality-four-arm-preregistration-v2.json").read_text()
    )

    assert tuple(preregistration["runtime"]["forbidden_environment_overrides"]) == (FORBIDDEN_ENVIRONMENT_OVERRIDES)
    assert preregistration["budgets"]["direct_per_turn"] == {
        **{
            "max_rounds": DIRECT_EXECUTION_BUDGET.max_rounds,
            "max_tool_calls": DIRECT_EXECUTION_BUDGET.max_tool_calls,
            "max_output_tokens": DIRECT_EXECUTION_BUDGET.max_output_tokens,
            "max_wall_seconds": DIRECT_EXECUTION_BUDGET.max_wall_seconds,
            "max_context_tokens": DIRECT_EXECUTION_BUDGET.max_context_tokens,
        },
        "effort": CodingEffort.MEDIUM.value,
    }
    assert "host-only public contract" in preregistration["public_state"]["scope_valid_semantics"]


def test_recovery_prompt_is_byte_identical_to_preregistration() -> None:
    repository = Path(__file__).resolve().parents[2]
    preregistration = json.loads(
        (repository / "benchmarks" / "repository-quality-four-arm-preregistration-v2.json").read_text()
    )
    assert RECOVERY_PROMPT == preregistration["budgets"]["extra_prompt_template"]
    assert RECOVERY_PROMPT.count("{instruction}") == 1


def test_pristine_plain_and_quality_roots_are_independent(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "fixture")
    source_digest = regular_tree_sha256(source)
    roots = prepare_pristine_direct_roots(source, tmp_path / "runs")

    roots.verify_pristine()
    assert roots.plain != roots.quality
    assert roots.pristine_sha256 == source_digest
    plain_policy = AgentToolPolicy.coding_workspace(roots.plain, allow_network=False)
    with pytest.raises(AgentPathViolation, match="outside"):
        resolve_workspace_path(roots.quality / "module.py", plain_policy)
    with pytest.raises(AgentPathViolation, match="outside"):
        resolve_workspace_path(source / "module.py", plain_policy)
    (roots.plain / "module.py").write_text("VALUE = 2\n")

    assert regular_tree_sha256(source) == source_digest
    assert regular_tree_sha256(roots.quality) == source_digest
    with pytest.raises(RepositoryPilotProtocolError, match="no longer matches"):
        roots.verify_pristine()


def test_safe_clone_rejects_symlinks_and_destination_escape(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "fixture")
    containment = tmp_path / "contained"
    containment.mkdir()
    (source / "alias.py").symlink_to(source / "module.py")

    with pytest.raises(RepositoryPilotProtocolError, match="symlink"):
        safe_clone_workspace(source, containment / "clone", containment_root=containment)

    (source / "alias.py").unlink()
    with pytest.raises(RepositoryPilotProtocolError, match="escapes"):
        safe_clone_workspace(source, tmp_path / "outside", containment_root=containment)


def test_regular_tree_and_clone_reject_hardlinks(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "fixture")
    os.link(source / "module.py", source / "module_alias.py")

    with pytest.raises(RepositoryPilotProtocolError, match="hard-linked"):
        regular_tree_sha256(source)
    containment = tmp_path / "contained"
    containment.mkdir()
    with pytest.raises(RepositoryPilotProtocolError, match="hard-linked"):
        safe_clone_workspace(source, containment / "clone", containment_root=containment)


def test_regular_tree_and_contract_reject_symlink_workspace_root(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "fixture")
    alias = tmp_path / "fixture-alias"
    alias.symlink_to(source, target_is_directory=True)

    with pytest.raises(RepositoryPilotProtocolError, match="real directory"):
        regular_tree_sha256(alias)
    with pytest.raises(RepositoryPilotProtocolError, match="real directory"):
        PublicScopeContract.capture(
            "fixture-1",
            alias,
            editable_names=("module.py",),
        )


def test_public_scope_contract_accepts_only_exact_declared_editable_change(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "fixture")
    contract = PublicScopeContract.capture(
        "fixture-1",
        root,
        editable_names=("module.py",),
    )
    (root / "module.py").write_text("VALUE = 2\n")
    terminal = regular_tree_sha256(root)

    verdict = contract.assess_terminal(
        fixture_id="fixture-1",
        terminal_root=root,
        terminal_tree_sha256=terminal,
    )

    assert verdict.scope_valid is True
    assert verdict.reason == "valid"
    assert verdict.fixture_id == contract.fixture_id
    assert verdict.contract_sha256 == contract.contract_sha256
    assert verdict.pristine_manifest_sha256 == contract.pristine_manifest_sha256
    assert verdict.terminal_tree_sha256 == terminal


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (
        ("unchanged", "no_editable_bytes_changed"),
        ("add", "name_set_changed"),
        ("delete", "name_set_changed"),
        ("rename", "name_set_changed"),
        ("mode", "kind_or_mode_changed"),
        ("directory_mode", "kind_or_mode_changed"),
        ("noneditable", "noneditable_bytes_changed"),
    ),
)
def test_public_scope_contract_rejects_manifest_and_byte_attacks(
    tmp_path: Path,
    case: str,
    expected_reason: str,
) -> None:
    root = _write_fixture(tmp_path / case)
    contract = PublicScopeContract.capture(
        f"fixture-{case}",
        root,
        editable_names=("module.py",),
    )
    if case == "add":
        (root / "added.py").write_text("VALUE = 9\n")
        (root / "module.py").write_text("VALUE = 2\n")
    elif case == "delete":
        (root / "tests" / "test_module.py").unlink()
        (root / "module.py").write_text("VALUE = 2\n")
    elif case == "rename":
        (root / "tests" / "test_module.py").rename(root / "tests" / "renamed.py")
        (root / "module.py").write_text("VALUE = 2\n")
    elif case == "mode":
        os.chmod(root / "module.py", 0o755)
        (root / "module.py").write_text("VALUE = 2\n")
    elif case == "directory_mode":
        os.chmod(root / "tests", 0o700)
        (root / "module.py").write_text("VALUE = 2\n")
    elif case == "noneditable":
        (root / "module.py").write_text("VALUE = 2\n")
        (root / "tests" / "test_module.py").write_text("tampered\n")

    terminal = regular_tree_sha256(root)
    verdict = contract.assess_terminal(
        fixture_id=f"fixture-{case}",
        terminal_root=root,
        terminal_tree_sha256=terminal,
    )

    assert verdict.scope_valid is False
    assert verdict.reason == expected_reason


@pytest.mark.parametrize("case", ("symlink", "special", "hardlink"))
def test_public_scope_contract_rejects_aliases_and_special_files(tmp_path: Path, case: str) -> None:
    root = _write_fixture(tmp_path / case)
    contract = PublicScopeContract.capture(
        f"fixture-{case}",
        root,
        editable_names=("module.py",),
    )
    (root / "module.py").write_text("VALUE = 2\n")
    if case == "symlink":
        (root / "alias.py").symlink_to("module.py")
    elif case == "special":
        os.mkfifo(root / "pipe")
    else:
        os.link(root / "module.py", root / "alias.py")

    with pytest.raises(RepositoryPilotProtocolError):
        contract.assess_terminal(
            fixture_id=f"fixture-{case}",
            terminal_root=root,
            terminal_tree_sha256="0" * 64,
        )


def test_public_scope_contract_rejects_wrong_fixture_and_stale_digest(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "fixture")
    contract = PublicScopeContract.capture(
        "fixture-1",
        root,
        editable_names=("module.py",),
    )
    (root / "module.py").write_text("VALUE = 2\n")
    terminal = regular_tree_sha256(root)

    with pytest.raises(RepositoryPilotProtocolError, match="wrong fixture"):
        contract.assess_terminal(
            fixture_id="fixture-2",
            terminal_root=root,
            terminal_tree_sha256=terminal,
        )
    (root / "module.py").write_text("VALUE = 3\n")
    with pytest.raises(RepositoryPilotProtocolError, match="stale"):
        contract.assess_terminal(
            fixture_id="fixture-1",
            terminal_root=root,
            terminal_tree_sha256=terminal,
        )


def test_archive_detects_mutation_and_clones_without_changing_source(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "source")
    containment = tmp_path / "work"
    containment.mkdir()
    source_digest = regular_tree_sha256(source)
    archive = ImmutableWorkspaceArchive.capture(
        source,
        containment / "archive",
        containment_root=containment,
    )
    branch = archive.clone_to(containment / "branch", containment_root=containment)
    (branch / "module.py").write_text("VALUE = 3\n")

    archive.verify_unchanged()
    assert regular_tree_sha256(source) == source_digest
    (archive.root / "module.py").write_text("tampered\n")
    with pytest.raises(RepositoryPilotProtocolError, match="archive changed"):
        archive.verify_unchanged()


def test_retained_executor_reconstructs_pristine_gate_and_rebinds_clone_policy(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "fixture")
    roots = prepare_pristine_direct_roots(source, tmp_path / "direct")
    engine = _FakeEngine()
    observed: list[dict[str, object]] = []
    expected_pristine: dict[str, object] = {}

    def process(instruction, _engine, _manager, _config, state):
        root = state["tool_policy"].workspace_roots[0]
        observed.append(
            {
                "instruction": instruction,
                "root": root,
                "budget": state["execution_budget"],
                "gate": state.get("_quality_gate"),
                "messages": list(state["messages"]),
            }
        )
        if len(observed) == 1:
            gate = state["_quality_gate"]
            _record_gate_change_and_validations(
                gate,
                root,
                value=2,
                kinds=(ValidationKind.TEST,),
            )
            state["messages"].extend(
                [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": "direct complete"},
                ]
            )
            return _result(report=gate.report())
        gate = state["_quality_gate"]
        assert gate.roots == (root,)
        assert gate.effort is CodingEffort.HIGH
        assert gate.require_net_workspace_change is True
        assert gate.validations == []
        assert gate.initial_snapshot == expected_pristine["snapshot"]
        assert gate.initial_snapshot.content_sha256 != gate.current_snapshot.content_sha256
        assert state["messages"][0]["content"] == "implement the change"
        _record_gate_change_and_validations(
            gate,
            root,
            value=3,
            kinds=(ValidationKind.TEST, ValidationKind.STATIC),
        )
        return _result(report=gate.report())

    executor = RetainedNativeAgentExecutor(
        config=object(),
        manager=_FakeManager(engine),
        engine=engine,
        tier="small",
        agent_module=_fake_agent(process),
    )
    direct = executor.run_direct(
        fixture_id="fixture-1",
        instruction="implement the change",
        workspace=roots.quality,
        quality_enabled=True,
    )
    expected_pristine["snapshot"] = direct.pristine_snapshot
    direct_digest = regular_tree_sha256(direct.workspace)
    # A caller cannot relax the recovery stage by mutating retained state.
    direct.state["execution_budget"] = AgentExecutionBudget(99, 99, 99_999, 999.0, 99_999)
    containment = tmp_path / "branches"
    containment.mkdir()
    archive = ImmutableWorkspaceArchive.capture(
        direct.workspace,
        containment / "direct-archive",
        containment_root=containment,
    )
    with pytest.raises(RepositoryPilotProtocolError, match="state execution budget"):
        executor.run_recovery(
            direct=direct,
            archive=archive,
            branch_root=containment / "rejected-recovery",
            containment_root=containment,
        )
    direct.state["execution_budget"] = DIRECT_EXECUTION_BUDGET
    recovery = executor.run_recovery(
        direct=direct,
        archive=archive,
        branch_root=containment / "recovery",
        containment_root=containment,
    )

    assert observed[0]["budget"] == DIRECT_EXECUTION_BUDGET
    assert observed[1]["budget"] == EXTRA_EXECUTION_BUDGET
    assert observed[1]["instruction"] == RECOVERY_PROMPT.format(instruction="implement the change")
    assert observed[1]["root"] == recovery.workspace
    assert observed[1]["root"] != direct.workspace
    assert AgentToolPermission.NETWORK not in recovery.state["tool_policy"].permissions
    assert regular_tree_sha256(direct.workspace) == direct_digest
    assert regular_tree_sha256(archive.root) == direct_digest
    assert (recovery.workspace / "module.py").read_text() == "VALUE = 3\n"
    assert recovery.reset_manifest == RESET_MANIFEST
    with pytest.raises(AgentPathViolation, match="outside"):
        resolve_workspace_path(archive.root / "module.py", recovery.state["tool_policy"])
    with pytest.raises(AgentPathViolation, match="outside"):
        resolve_workspace_path(direct.workspace / "module.py", recovery.state["tool_policy"])
    assert engine.reset_count == 2
    assert engine._last_prompt_tokens == []
    assert engine._pending_assistant_prefill == ""


def test_direct_executor_rejects_nonmedium_effort_wrong_engine_and_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_fixture(tmp_path / "fixture")
    roots = prepare_pristine_direct_roots(source, tmp_path / "direct")
    engine = _FakeEngine()
    process_called = False

    def process(_instruction, _engine, _manager, _config, _state):
        nonlocal process_called
        process_called = True
        return _result(report=None)

    executor = RetainedNativeAgentExecutor(
        config=object(),
        manager=_FakeManager(engine),
        engine=engine,
        tier="small",
        agent_module=_fake_agent(process),
    )
    with pytest.raises(RepositoryPilotProtocolError, match="exactly medium"):
        executor.run_direct(
            fixture_id="fixture-1",
            instruction="change",
            workspace=roots.plain,
            quality_enabled=False,
            effort="high",
        )
    assert process_called is False

    wrong_engine = _FakeEngine()
    wrong_manager_executor = RetainedNativeAgentExecutor(
        config=object(),
        manager=_FakeManager(wrong_engine),
        engine=engine,
        tier="small",
        agent_module=_fake_agent(process),
    )
    with pytest.raises(RepositoryPilotProtocolError, match="engine identity"):
        wrong_manager_executor.run_direct(
            fixture_id="fixture-1",
            instruction="change",
            workspace=roots.plain,
            quality_enabled=False,
        )
    assert process_called is False

    monkeypatch.setenv("MIO_PREFILL_CHUNK", "4096")
    with pytest.raises(RepositoryPilotProtocolError, match="environment override"):
        executor.run_direct(
            fixture_id="fixture-1",
            instruction="change",
            workspace=roots.plain,
            quality_enabled=False,
        )
    assert process_called is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("mutation_epoch", True, "mutation epoch"),
        ("mutation_epoch", -1, "mutation epoch"),
        ("mutation_epoch", 0, "contradict"),
        ("changed_kinds", None, "string list"),
        ("changed_kinds", "code", "string list"),
        ("changed_kinds", ["code", "code"], "malformed"),
        ("changed_kinds", ["cyber"], "malformed"),
        ("changed_kinds", [], "contradict"),
    ),
)
def test_recovery_rejects_malformed_parent_epoch_and_changed_kinds(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source = _write_fixture(tmp_path / "fixture")
    roots = prepare_pristine_direct_roots(source, tmp_path / "direct")
    engine = _FakeEngine()
    invoked = 0

    def process(_instruction, _engine, _manager, _config, state):
        nonlocal invoked
        invoked += 1
        if invoked != 1:
            raise AssertionError("malformed parent must be rejected before recovery generation")
        root = state["tool_policy"].workspace_roots[0]
        gate = state["_quality_gate"]
        _record_gate_change_and_validations(
            gate,
            root,
            value=2,
            kinds=(ValidationKind.TEST,),
        )
        return _result(report=gate.report())

    executor = RetainedNativeAgentExecutor(
        config=object(),
        manager=_FakeManager(engine),
        engine=engine,
        tier="small",
        agent_module=_fake_agent(process),
    )
    direct = executor.run_direct(
        fixture_id="fixture-1",
        instruction="implement",
        workspace=roots.quality,
        quality_enabled=True,
    )
    containment = tmp_path / "branches"
    containment.mkdir()
    archive = ImmutableWorkspaceArchive.capture(
        direct.workspace,
        containment / "archive",
        containment_root=containment,
    )
    malformed = dict(direct.result.quality_gate or {})
    malformed[field] = value
    direct.result = replace(direct.result, quality_gate=malformed)

    with pytest.raises(RepositoryPilotProtocolError, match=message):
        executor.run_recovery(
            direct=direct,
            archive=archive,
            branch_root=containment / "recovery",
            containment_root=containment,
        )
    assert invoked == 1
    assert not (containment / "recovery").exists()


def test_recovery_accepts_coherent_no_edit_quality_root_and_can_earn_real_change(
    tmp_path: Path,
) -> None:
    source = _write_fixture(tmp_path / "fixture")
    roots = prepare_pristine_direct_roots(source, tmp_path / "direct")
    engine = _FakeEngine()
    observed_gates: list[CodingQualityGate] = []

    def process(_instruction, _engine, _manager, _config, state):
        root = state["tool_policy"].workspace_roots[0]
        gate = state["_quality_gate"]
        observed_gates.append(gate)
        if len(observed_gates) == 1:
            assert gate.effort is CodingEffort.MEDIUM
            assert gate.mutation_epoch == 0
            assert gate.changed_kinds == set()
            assert gate.initial_snapshot.content_sha256 == gate.current_snapshot.content_sha256
            return _result(report=gate.report(), terminal_reason="quality_incomplete")
        assert gate.effort is CodingEffort.HIGH
        assert gate.mutation_epoch == 0
        assert gate.changed_kinds == set()
        assert gate.initial_snapshot.content_sha256 == gate.current_snapshot.content_sha256
        _record_gate_change_and_validations(
            gate,
            root,
            value=2,
            kinds=(ValidationKind.TEST, ValidationKind.STATIC),
        )
        return _result(report=gate.report())

    executor = RetainedNativeAgentExecutor(
        config=object(),
        manager=_FakeManager(engine),
        engine=engine,
        tier="small",
        agent_module=_fake_agent(process),
    )
    direct = executor.run_direct(
        fixture_id="fixture-1",
        instruction="implement",
        workspace=roots.quality,
        quality_enabled=True,
    )
    assert direct.pristine_snapshot.content_sha256 == direct.current_snapshot.content_sha256
    containment = tmp_path / "branches"
    containment.mkdir()
    archive = ImmutableWorkspaceArchive.capture(
        direct.workspace,
        containment / "archive",
        containment_root=containment,
    )

    recovery = executor.run_recovery(
        direct=direct,
        archive=archive,
        branch_root=containment / "recovery",
        containment_root=containment,
    )

    assert len(observed_gates) == 2
    assert recovery.trusted_quality_gate is observed_gates[1]
    assert recovery.current_snapshot.content_sha256 != recovery.pristine_snapshot.content_sha256
    assert recovery.result.quality_gate["decision"] == "pass"


def test_public_state_ignores_text_hashes_and_hidden_fields(tmp_path: Path) -> None:
    stage, contract, verdict = _quality_stage(tmp_path)
    public = extract_public_repository_state(
        stage,
        scope_contract=contract,
        scope_verdict=verdict,
    )
    encoded = public_state_json(public)
    preregistration = json.loads(
        (
            Path(__file__).resolve().parents[2] / "benchmarks" / "repository-quality-four-arm-preregistration-v2.json"
        ).read_text()
    )
    payload = json.loads(encoded)

    assert public.scope_valid is True
    assert public.high_coverage is True
    assert public.net_workspace_changed is True
    assert "secret prompt" not in encoded
    assert "assistant text" not in encoded
    assert "hidden" not in encoded
    assert "sha256" not in encoded
    assert str(stage.workspace) not in encoded
    assert tuple(payload) == tuple(sorted(preregistration["public_state"]["allowed_features"]))
    assert "public_test_status" not in payload
    assert "review" not in payload


def test_plain_direct_extraction_is_strict_typed_and_bridges_to_protocol(tmp_path: Path) -> None:
    stage, contract, verdict = _plain_stage(tmp_path)

    public = extract_public_repository_state(
        stage,
        scope_contract=contract,
        scope_verdict=verdict,
    )
    evidence = to_protocol_public_evidence(public)
    cost = to_protocol_candidate_cost(public)

    assert public.scope_valid is True
    assert public.gate_present is False
    assert public.gate_decision == "not_applicable"
    assert public.gate_satisfied is True
    assert public.public_test_status == "not_run"
    assert public.trusted_test_or_build_attempt_count == 0
    assert public.validation_counts == (
        ("test", 0),
        ("build", 0),
        ("static", 0),
        ("diff", 0),
        ("review", 0),
    )
    assert evidence.quality_decision == "not_applicable"
    assert evidence.visible_check is VisibleCheckOutcome.NOT_RUN
    assert evidence.trajectory_valid(quality_derived=False) is True
    assert evidence.state is PublicState.ROOT_INCOMPLETE
    assert cost.model_rounds == 1
    assert cost.tool_calls == 0
    assert cost.output_tokens == 7


def test_candidate_cost_uses_raw_model_phase_ns_not_round_time_with_overhead(tmp_path: Path) -> None:
    stage, contract, verdict = _plain_stage(tmp_path)
    result = stage.result
    assert isinstance(result, AgentTurnResult)
    round_trace = replace(
        result.rounds[0],
        total_time_s=1.25,
        prefill_ns=250_000_000,
        decode_ns=500_000_000,
        model_total_ns=750_000_000,
    )
    stage.result = replace(result, rounds=(round_trace,))

    public = extract_public_repository_state(
        stage,
        scope_contract=contract,
        scope_verdict=verdict,
    )
    cost = to_protocol_candidate_cost(public)

    assert round_trace.total_time_s > cost.model_seconds
    assert public.model_seconds == 0.75
    assert cost.model_seconds == 0.75
    assert cost.wall_seconds == 1.5


def test_plain_direct_rejects_quality_report_gate_or_validate_trace(tmp_path: Path) -> None:
    stage, contract, verdict = _plain_stage(tmp_path)
    result = stage.result
    assert isinstance(result, AgentTurnResult)
    stage.result = replace(result, quality_gate={"schema": "forged"})
    with pytest.raises(RepositoryPilotProtocolError, match="Quality report"):
        extract_public_repository_state(
            stage,
            scope_contract=contract,
            scope_verdict=verdict,
        )

    stage.result = result
    stage.trusted_quality_gate = CodingQualityGate(
        roots=(stage.workspace,),
        initial_snapshot=stage.pristine_snapshot,
        current_snapshot=stage.current_snapshot,
    )
    with pytest.raises(RepositoryPilotProtocolError, match="retained a Quality gate"):
        extract_public_repository_state(
            stage,
            scope_contract=contract,
            scope_verdict=verdict,
        )

    stage.trusted_quality_gate = None
    validate_trace = _tool_trace(0)
    stage.result = replace(
        result,
        tool_events=(validate_trace,),
        tool_calls=1,
        tool_result_chars=1,
    )
    with pytest.raises(RepositoryPilotProtocolError, match="forbidden validate"):
        extract_public_repository_state(
            stage,
            scope_contract=contract,
            scope_verdict=verdict,
        )


def test_public_extraction_requires_fresh_exact_scope_binding(tmp_path: Path) -> None:
    stage, contract, verdict = _quality_stage(tmp_path)

    with pytest.raises(RepositoryPilotProtocolError, match="requires a host scope"):
        extract_public_repository_state(
            stage,
            scope_contract=None,  # type: ignore[arg-type]
            scope_verdict=verdict,
        )
    wrong_fixture = replace(stage, fixture_id="fixture-2")
    with pytest.raises(RepositoryPilotProtocolError, match="fixture binding"):
        extract_public_repository_state(
            wrong_fixture,
            scope_contract=contract,
            scope_verdict=verdict,
        )

    (stage.workspace / "module.py").write_text("VALUE = 3\n")
    with pytest.raises(RepositoryPilotProtocolError, match="stale"):
        extract_public_repository_state(
            stage,
            scope_contract=contract,
            scope_verdict=verdict,
        )


def test_public_extraction_exposes_invalid_noneditable_scope_without_content(tmp_path: Path) -> None:
    stage, contract, verdict = _quality_stage(tmp_path, mutate_noneditable=True)

    public = extract_public_repository_state(
        stage,
        scope_contract=contract,
        scope_verdict=verdict,
    )

    assert verdict.reason == "noneditable_bytes_changed"
    assert public.scope_valid is False
    assert public.state_label == "scope_invalid"
    assert "test_module.py" not in public_state_json(public)


def test_visible_test_aggregate_uses_only_trusted_current_epoch_evidence(tmp_path: Path) -> None:
    stage, contract, verdict = _quality_stage(
        tmp_path,
        stale_validation_specs=((ValidationKind.TEST, False, "denied"),),
    )

    public = extract_public_repository_state(
        stage,
        scope_contract=contract,
        scope_verdict=verdict,
    )

    assert public.public_test_attempted is True
    assert public.public_test_passed is True
    assert public.public_test_status == "passed"
    assert public.validation_count("test") == 1


def test_visible_test_aggregate_ignores_same_epoch_wrong_revision_evidence(tmp_path: Path) -> None:
    stage, contract, verdict = _quality_stage(tmp_path)
    gate = stage.trusted_quality_gate
    result = stage.result
    assert isinstance(gate, CodingQualityGate)
    assert isinstance(result, AgentTurnResult)
    gate.validations.append(
        ValidationEvidence(
            kind=ValidationKind.BUILD,
            epoch=gate.mutation_epoch,
            revision_sha256=stage.pristine_snapshot.revision_sha256,
            command_sha256="e" * 64,
            allowed=False,
            outcome="denied",
        )
    )
    gate.validate_invocations += 1
    stage.result = replace(
        result,
        tool_events=(*result.tool_events, _tool_trace(2, allowed=False, outcome="denied")),
        tool_calls=3,
        tool_result_chars=3,
        quality_gate=gate.report(),
    )

    public = extract_public_repository_state(
        stage,
        scope_contract=contract,
        scope_verdict=verdict,
    )

    assert public.trusted_test_or_build_attempt_count == 1
    assert public.public_test_passed is True
    assert public.public_test_status == "passed"


def test_visible_test_aggregate_preserves_exact_not_run_attempt_count(tmp_path: Path) -> None:
    stage, contract, verdict = _quality_stage(
        tmp_path,
        validation_specs=((ValidationKind.STATIC, True, "ok"),),
    )

    public = extract_public_repository_state(
        stage,
        scope_contract=contract,
        scope_verdict=verdict,
    )
    evidence = to_protocol_public_evidence(public)

    assert public.public_test_attempted is False
    assert public.public_test_status == "not_run"
    assert public.trusted_test_or_build_attempt_count == 0
    assert evidence.trusted_test_or_build_attempt_count == 0
    assert evidence.visible_check is VisibleCheckOutcome.NOT_RUN


def test_visible_test_aggregate_fails_on_any_current_test_or_build_failure(tmp_path: Path) -> None:
    stage, contract, verdict = _quality_stage(
        tmp_path,
        validation_specs=(
            (ValidationKind.TEST, True, "ok"),
            (ValidationKind.BUILD, False, "denied"),
            (ValidationKind.STATIC, True, "ok"),
        ),
    )

    public = extract_public_repository_state(
        stage,
        scope_contract=contract,
        scope_verdict=verdict,
    )
    evidence = to_protocol_public_evidence(public)

    assert public.gate_decision == "pass"
    assert public.public_test_attempted is True
    assert public.public_test_passed is False
    assert public.public_test_status == "error"
    assert public.trusted_test_or_build_attempt_count == 2
    assert evidence.trusted_test_or_build_attempt_count == 2
    assert evidence.trusted_test_count == 1
    assert evidence.trusted_build_count == 0
    assert evidence.visible_check is VisibleCheckOutcome.FAIL
    assert public.state_label == "public_fail"


@pytest.mark.parametrize(
    ("attempted", "passed", "status"),
    (
        (False, True, "not_run"),
        (False, False, "passed"),
        (True, False, "passed"),
        (True, True, "failed"),
        (True, True, "error"),
        (True, False, "not_run"),
    ),
)
def test_visible_public_test_result_rejects_every_contradictory_shape(
    attempted: bool,
    passed: bool,
    status: str,
) -> None:
    with pytest.raises(ValueError, match="contradictory"):
        VisiblePublicTestResult(attempted, passed, status)


def test_public_state_rejects_attempt_success_and_status_contradictions() -> None:
    with pytest.raises(ValueError, match="successes exceed attempts"):
        _public_state(trusted_test_or_build_attempt_count=0)
    with pytest.raises(ValueError, match="pass flag contradicts"):
        _public_state(trusted_test_or_build_attempt_count=2)
    with pytest.raises(TypeError, match="must be bool"):
        _public_state(tool_telemetry_complete=1)


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("telemetry_string", "must be bool"),
        ("tool_count", "tool calls contradict"),
        ("completion_total", "completion-token total"),
        ("output_budget", "completion tokens exceed"),
        ("round_budget", "model rounds exceed"),
        ("state_budget", "state execution budget"),
        ("state_effort", "state effort"),
        ("round_bool", "round deadline_hit must be bool"),
        ("backend", "backend is not frozen DFlash"),
        ("fallback", "autoregressive fallback"),
        ("drafter", "selection is not strict DFlash"),
        ("drafter_ref", "reference differs"),
        ("prefill_accounting", "physical prefill tokens contradict"),
        ("phase_accounting", "model total nanoseconds contradict"),
        ("model_wall", "model seconds exceed"),
        ("gate_content_hash", "Quality report contradicts"),
        ("gate_count", "Quality report contradicts"),
    ),
)
def test_public_extraction_rejects_malformed_or_contradictory_telemetry(
    tmp_path: Path,
    attack: str,
    message: str,
) -> None:
    stage, contract, verdict = _quality_stage(tmp_path)
    result = stage.result
    assert isinstance(result, AgentTurnResult)
    if attack == "telemetry_string":
        stage.result = replace(result, tool_telemetry_complete="true")  # type: ignore[arg-type]
    elif attack == "tool_count":
        stage.result = replace(result, tool_calls=result.tool_calls + 1)
    elif attack == "completion_total":
        stage.result = replace(result, completion_tokens=result.completion_tokens + 1)
    elif attack == "output_budget":
        oversized = replace(
            result.rounds[0],
            completion_tokens=2_049,
            physical_decode_tokens=2_049,
        )
        stage.result = replace(result, rounds=(oversized,), completion_tokens=2_049)
    elif attack == "round_budget":
        rounds = tuple(replace(result.rounds[0], round_index=index, completion_tokens=1) for index in range(13))
        stage.result = replace(result, rounds=rounds, completion_tokens=13)
    elif attack == "state_budget":
        stage.state["execution_budget"] = AgentExecutionBudget(99, 99, 99_999, 999.0, 99_999)
    elif attack == "state_effort":
        stage.state["coding_effort"] = "high"
    elif attack == "round_bool":
        malformed_round = replace(result.rounds[0], deadline_hit=1)  # type: ignore[arg-type]
        stage.result = replace(result, rounds=(malformed_round,))
    elif attack == "backend":
        stage.result = replace(
            result,
            rounds=(replace(result.rounds[0], generation_backend="baseline"),),
        )
    elif attack == "fallback":
        stage.result = replace(
            result,
            rounds=(replace(result.rounds[0], fallback_ar=True),),
        )
    elif attack == "drafter":
        stage.result = replace(
            result,
            rounds=(replace(result.rounds[0], drafter_selected="baseline"),),
        )
    elif attack == "drafter_ref":
        stage.result = replace(
            result,
            rounds=(replace(result.rounds[0], drafter_ref="other/dflash"),),
        )
    elif attack == "prefill_accounting":
        stage.result = replace(
            result,
            rounds=(replace(result.rounds[0], physical_prefill_tokens=9),),
        )
    elif attack == "phase_accounting":
        stage.result = replace(
            result,
            rounds=(replace(result.rounds[0], model_total_ns=999_999_999),),
        )
    elif attack == "model_wall":
        stage.result = replace(result, wall_time_s=0.5)
    elif attack == "gate_content_hash":
        report = dict(result.quality_gate or {})
        report["current_content_sha256"] = "f" * 64
        stage.result = replace(result, quality_gate=report)
    else:
        report = dict(result.quality_gate or {})
        counts = dict(report["validation_counts"])
        counts["test"] = 99
        report["validation_counts"] = counts
        stage.result = replace(result, quality_gate=report)

    with pytest.raises(RepositoryPilotProtocolError, match=message):
        extract_public_repository_state(
            stage,
            scope_contract=contract,
            scope_verdict=verdict,
        )


def test_public_selector_requires_strict_admissible_recovery() -> None:
    direct = _public_state(
        gate_decision="incomplete",
        gate_phase="dirty",
        gate_satisfied=False,
        validation_counts=(("test", 1), ("build", 0), ("static", 0), ("diff", 0), ("review", 0)),
    )
    recovery = _public_state()
    assert prefer_recovery_publicly(direct, recovery) is True
    assert prefer_recovery_publicly(recovery, recovery) is False
    assert prefer_recovery_publicly(direct, replace(recovery, tool_telemetry_complete=False)) is False


def test_public_state_classifier_matches_frozen_total_order() -> None:
    complete = _public_state()
    assert complete.state_label == "public_unknown"
    failed_counts = (
        ("test", 0),
        ("build", 0),
        ("static", 1),
        ("diff", 0),
        ("review", 0),
    )
    assert (
        replace(
            complete,
            public_test_passed=False,
            public_test_status="failed",
            validation_counts=failed_counts,
        ).state_label
        == "public_fail"
    )
    assert replace(complete, scope_valid=False).state_label == "scope_invalid"
    assert (
        replace(
            complete,
            gate_decision="incomplete",
            gate_satisfied=False,
        ).state_label
        == "root_incomplete"
    )
    assert replace(complete, scope_valid=False, deadline_violated=True).state_label == "root_incomplete"


def test_hidden_evaluator_is_unreachable_before_complete_selection_barrier(tmp_path: Path) -> None:
    first = _write_fixture(tmp_path / "first", "VALUE = 1\n")
    second = _write_fixture(tmp_path / "second", "VALUE = 2\n")
    keys = {arm: logical_terminal_key("fixture-1", arm) for arm in LOGICAL_ARMS}
    barrier = HiddenEvaluationBarrier.for_fixtures(("fixture-1",))
    calls: list[str] = []

    def hidden(key: str, _root: Path) -> HiddenOutcome:
        calls.append(key)
        return HiddenOutcome(True, True)

    with pytest.raises(RepositoryPilotProtocolError, match="blocked"):
        barrier.evaluate(hidden)
    assert calls == []

    _register_test_terminal(
        barrier,
        keys[LogicalArm.PLAIN],
        first,
        physical_candidate_id="first-physical",
    )
    with pytest.raises(RepositoryPilotProtocolError, match="all terminal"):
        barrier.seal(generation_receipt=_generation_receipt())
    assert calls == []

    for arm in (
        LogicalArm.QUALITY,
        LogicalArm.QUALITY_STATIC_EXTRA,
        LogicalArm.MARKOV_QUALITY,
    ):
        _register_test_terminal(
            barrier,
            keys[arm],
            second,
            physical_candidate_id="second-physical",
        )
    sealed = barrier.seal(generation_receipt=_generation_receipt())
    assert set(sealed) == set(keys.values())
    batch = barrier.evaluate(hidden)
    assert isinstance(batch, HiddenEvaluationBatch)
    assert dict(batch.results) == {key: HiddenOutcome(True, True) for key in keys.values()}
    assert isinstance(batch.receipt, EvaluationBarrierReceipt)
    assert batch.receipt.expected_logical_selection_count == 4
    assert batch.receipt.registered_logical_selection_count == 4
    assert batch.receipt.unique_terminal_artifact_count == 2
    assert batch.receipt.hidden_evaluation_count == 2
    assert batch.receipt.all_generation_complete_before_seal is True
    assert batch.receipt.selection_sealed_before_hidden is True
    assert batch.receipt.hidden_evaluation_single_use is True
    assert barrier.receipt is batch.receipt
    assert calls == ["fixture-1", "fixture-1"]
    with pytest.raises(RepositoryPilotProtocolError, match="single-use"):
        barrier.evaluate(hidden)


def test_hidden_barrier_requires_generation_attestation_and_strict_outcome(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "terminal")
    barrier = HiddenEvaluationBarrier.for_fixtures(("fixture-1",))
    for key in tuple(logical_terminal_key("fixture-1", arm) for arm in LOGICAL_ARMS):
        _register_test_terminal(barrier, key, root)
    with pytest.raises(RepositoryPilotProtocolError, match="exact GenerationCompletionReceipt"):
        barrier.seal(generation_receipt=True)  # type: ignore[arg-type]
    with pytest.raises(RepositoryPilotProtocolError, match="four per fixture"):
        barrier.seal(generation_receipt=_generation_receipt(fixture_count=2))
    with pytest.raises(ValueError, match="not every scheduled root"):
        GenerationCompletionReceipt(
            fixture_count=1,
            expected_root_generation_count=2,
            completed_root_generation_count=1,
            expected_unique_extra_generation_count=0,
            completed_unique_extra_generation_count=0,
            root_schedule_sealed_before_first_generation=True,
            allocation_sealed_after_all_roots=True,
            extra_schedule_sealed_before_first_extra=True,
        )
    with pytest.raises(RepositoryPilotProtocolError, match="unavailable"):
        _ = barrier.receipt

    barrier.seal(generation_receipt=_generation_receipt())
    calls = 0

    def malformed(_fixture_id: str, _root: Path):
        nonlocal calls
        calls += 1
        return {"evaluator_passed": True, "regression_free": True}

    with pytest.raises(RepositoryPilotProtocolError, match="exact HiddenOutcome"):
        barrier.evaluate(malformed)  # type: ignore[arg-type]
    assert calls == 1
    with pytest.raises(RepositoryPilotProtocolError, match="unavailable"):
        _ = barrier.receipt
    with pytest.raises(RepositoryPilotProtocolError, match="single-use"):
        barrier.evaluate(lambda _fixture, _root: HiddenOutcome(True, True))

    with pytest.raises(TypeError, match="hidden outcomes must be bool"):
        HiddenOutcome(True, 1)  # type: ignore[arg-type]


def test_hidden_barrier_rejects_evaluator_workspace_mutation(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "terminal")
    barrier = HiddenEvaluationBarrier.for_fixtures(("fixture-1",))
    for key in tuple(logical_terminal_key("fixture-1", arm) for arm in LOGICAL_ARMS):
        _register_test_terminal(barrier, key, root)
    barrier.seal(generation_receipt=_generation_receipt())
    calls = 0

    def malicious(_fixture_id: str, workspace: Path) -> HiddenOutcome:
        nonlocal calls
        calls += 1
        (workspace / "module.py").write_text("VALUE = 99\n")
        return HiddenOutcome(True, True)

    with pytest.raises(RepositoryPilotProtocolError, match="mutated its sealed"):
        barrier.evaluate(malicious)
    assert calls == 1
    with pytest.raises(RepositoryPilotProtocolError, match="unavailable"):
        _ = barrier.receipt
    with pytest.raises(RepositoryPilotProtocolError, match="single-use"):
        barrier.evaluate(lambda _fixture, _root: HiddenOutcome(True, True))


def test_hidden_barrier_detects_post_selection_workspace_mutation(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "terminal")
    barrier = HiddenEvaluationBarrier.for_fixtures(("fixture-1",))
    for key in tuple(logical_terminal_key("fixture-1", arm) for arm in LOGICAL_ARMS):
        _register_test_terminal(barrier, key, root)
    barrier.seal(generation_receipt=_generation_receipt())
    (root / "module.py").write_text("VALUE = 99\n")
    called = False

    def hidden(_key: str, _root: Path) -> HiddenOutcome:
        nonlocal called
        called = True
        return HiddenOutcome(True, True)

    with pytest.raises(RepositoryPilotProtocolError, match="changed after"):
        barrier.evaluate(hidden)
    assert called is False


def test_hidden_barrier_distinguishes_logical_selection_from_physical_dedup(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "shared")
    keys = {arm: logical_terminal_key("fixture-1", arm) for arm in LOGICAL_ARMS}
    barrier = HiddenEvaluationBarrier.for_fixtures(("fixture-1",))
    for key in keys.values():
        _register_test_terminal(barrier, key, root)
    barrier.seal(generation_receipt=_generation_receipt(unique_extra_count=1))
    calls: list[str] = []

    batch = barrier.evaluate(lambda key, _root: calls.append(key) or HiddenOutcome(True, True))

    assert barrier.logical_selection_count == 4
    assert barrier.physical_evaluation_count == 1
    assert calls == ["fixture-1"]
    assert dict(batch.results) == {key: HiddenOutcome(True, True) for key in keys.values()}
    assert batch.receipt.unique_terminal_artifact_count == 1
    assert batch.receipt.hidden_evaluation_count == 1


def test_barrier_builds_records_only_from_exact_logical_terminal_outcomes(tmp_path: Path) -> None:
    plain_stage, plain_contract, plain_verdict = _plain_stage(
        tmp_path / "plain",
        fixture_id="fixture-1",
        value=3,
    )
    quality_stage, quality_contract, quality_verdict = _quality_stage(tmp_path / "quality")
    plain_state = extract_public_repository_state(
        plain_stage,
        scope_contract=plain_contract,
        scope_verdict=plain_verdict,
    )
    quality_state = extract_public_repository_state(
        quality_stage,
        scope_contract=quality_contract,
        scope_verdict=quality_verdict,
    )
    plain_candidate = CandidateObservation(
        physical_candidate_id="plain-physical",
        terminal_artifact_id=plain_stage.terminal_tree_sha256,
        public_evidence=to_protocol_public_evidence(plain_state),
        cost=to_protocol_candidate_cost(plain_state),
    )
    quality_candidate = CandidateObservation(
        physical_candidate_id="quality-physical",
        terminal_artifact_id=quality_stage.terminal_tree_sha256,
        public_evidence=to_protocol_public_evidence(quality_state),
        cost=to_protocol_candidate_cost(quality_state),
    )
    selected = SelectedFixtureCandidates(
        fixture_id="fixture-1",
        plain_root=plain_candidate,
        quality_root=quality_candidate,
        static_child=None,
        markov_child=None,
        static_selection=CandidateChoice.ROOT,
        markov_selection=CandidateChoice.ROOT,
    )
    barrier = HiddenEvaluationBarrier.for_fixtures(("fixture-1",))
    for arm in LOGICAL_ARMS:
        stage = plain_stage if arm is LogicalArm.PLAIN else quality_stage
        state = plain_state if arm is LogicalArm.PLAIN else quality_state
        candidate = plain_candidate if arm is LogicalArm.PLAIN else quality_candidate
        barrier.register(
            logical_terminal_key("fixture-1", arm),
            stage.workspace,
            state,
            fixture_id="fixture-1",
            observation=candidate,
        )
    barrier.seal(generation_receipt=_generation_receipt())

    batch = barrier.evaluate(
        lambda _fixture, root: (
            HiddenOutcome(False, True) if root == plain_stage.workspace else HiddenOutcome(True, True)
        )
    )
    records = barrier.bind_fixture_records((selected,), batch)

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, FixturePilotRecord)
    assert record.outcome_for_arm(LogicalArm.PLAIN).outcome == HiddenOutcome(False, True)
    for arm in (
        LogicalArm.QUALITY,
        LogicalArm.QUALITY_STATIC_EXTRA,
        LogicalArm.MARKOV_QUALITY,
    ):
        assert record.outcome_for_arm(arm).outcome == HiddenOutcome(True, True)
        assert record.outcome_for_arm(arm).trajectory_valid is True

    reversed_results = dict(reversed(tuple(batch.results.items())))
    reversed_bindings = dict(reversed(tuple(batch.terminal_bindings.items())))
    forged_batch = HiddenEvaluationBatch(
        reversed_results,
        reversed_bindings,
        batch.receipt,
    )
    with pytest.raises(RepositoryPilotProtocolError, match="not emitted by this barrier"):
        barrier.bind_fixture_records((selected,), forged_batch)

    swapped_plain = replace(
        plain_candidate,
        terminal_artifact_id=quality_candidate.terminal_artifact_id,
    )
    swapped = replace(selected, plain_root=swapped_plain)
    with pytest.raises(RepositoryPilotProtocolError, match="observation differs"):
        barrier.bind_fixture_records((swapped,), batch)

    zero_cost = replace(
        plain_candidate.cost,
        model_rounds=0,
        tool_calls=0,
        output_tokens=0,
        model_seconds=0.0,
        wall_seconds=0.0,
    )
    for forged_plain in (
        replace(plain_candidate, cost=zero_cost),
        replace(plain_candidate, physical_candidate_id="forged-physical"),
    ):
        forged = replace(selected, plain_root=forged_plain)
        with pytest.raises(RepositoryPilotProtocolError, match="observation differs"):
            barrier.bind_fixture_records((forged,), batch)

    (quality_stage.workspace / "module.py").write_text("VALUE = 99\n")
    with pytest.raises(RepositoryPilotProtocolError, match="changed before record binding"):
        barrier.bind_fixture_records((selected,), batch)
