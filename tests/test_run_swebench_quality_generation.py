from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import bench_swebench_quality as protocol
from scripts import run_swebench_quality_generation as runner


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _source_repo(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "Mio Test")
    _git(source, "config", "user.email", "mio@example.invalid")
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", "module.py")
    _git(source, "commit", "--quiet", "-m", "base")
    return source, _git(source, "rev-parse", "HEAD")


def _instances(base_commit: str) -> tuple[protocol.PublicInstance, ...]:
    return tuple(
        protocol.PublicInstance(
            instance_id=f"owner__repository-{index}",
            repo="owner/repository",
            base_commit=base_commit,
            problem_statement=f"Change VALUE for task {index}.",
        )
        for index in (1, 2)
    )


def _schedule_document(instances: tuple[protocol.PublicInstance, ...]):
    schedule = protocol.make_balanced_schedule(
        [instance.instance_id for instance in instances],
        require_full=False,
    )
    document = protocol.private_schedule_document(instances, evidence_run=False)
    return document, schedule


def _tier(**changes):
    values = {
        "drafter_backend": "target_ar",
        "context_window": 32_768,
        "max_output_tokens": 4_096,
        "tq_bits": 16,
        "pq_bits": 16,
        "bmp_paths": 1,
        "ddtree_budget": 0,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _agent_module():
    tools = {name: {"fn": lambda: None, "args": []} for name in runner.TOOL_SURFACE}
    specs = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in runner.TOOL_SURFACE
    ]
    return SimpleNamespace(AGENT_TOOLS=tools, AGENT_TOOLS_SPEC=specs)


def _binding() -> runner.GenerationBinding:
    return runner.GenerationBinding(
        mio_commit="b" * 40,
        model_identity=protocol.EXPECTED_MODEL_IDENTITY,
        runtime_digest="d" * 64,
    )


class RecordingExecutor:
    def __init__(self) -> None:
        self.requests: list[runner.ArmRunRequest] = []

    def __call__(self, request: runner.ArmRunRequest) -> runner.ArmRunOutcome:
        self.requests.append(request)
        assert not any(path.name.casefold() == ".git" for path in request.workspace.rglob("*"))
        assert not any(request.cache_directory.iterdir())
        (request.workspace / "module.py").write_text(
            f"VALUE = {2 if request.quality_gate_enabled else 3}\n",
            encoding="utf-8",
        )
        return runner.ArmRunOutcome(
            status="completed",
            quality_gate_decision=("satisfied" if request.quality_gate_enabled else "not_applicable"),
            output_tokens=17,
            tool_calls=2,
            wall_seconds=0.25,
        )


def test_pair_runner_uses_balanced_order_identical_tools_and_fresh_state(tmp_path: Path) -> None:
    source, base_commit = _source_repo(tmp_path)
    instances = _instances(base_commit)
    document, schedule = _schedule_document(instances)
    layout = runner.GenerationLayout.create(tmp_path / "generation")
    executor = RecordingExecutor()
    agent_module = _agent_module()

    observed_factor = runner.run_generation_pairs(
        schedule_document=document,
        schedule=schedule,
        layout=layout,
        workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
        executor=executor,
        binding=_binding(),
        tier_config=_tier(),
        agent_module=agent_module,
    )

    assert [request.entry for request in executor.requests] == list(schedule)
    assert len({request.workspace for request in executor.requests}) == len(schedule)
    assert len({request.cache_directory for request in executor.requests}) == len(schedule)
    assert all(tuple(request.tool_registry) == runner.TOOL_SURFACE for request in executor.requests)
    assert all(
        request.tool_policy.command_timeout_s == runner.FROZEN_COMMAND_TIMEOUT_SECONDS for request in executor.requests
    )
    assert (
        len({protocol.sha256_bytes(protocol.canonical_json_bytes(request.tool_specs)) for request in executor.requests})
        == 1
    )
    assert all("owner__repository" not in request.instruction for request in executor.requests)
    assert all(request.coding_effort == "medium" for request in executor.requests)
    for first, second in zip(executor.requests[::2], executor.requests[1::2], strict=True):
        assert first.seed == second.seed
        assert first.entry.instance_id == second.entry.instance_id
        assert {first.quality_gate_enabled, second.quality_gate_enabled} == {False, True}
    _registry, _specs, surface_digest = runner.build_identical_tool_surface(agent_module)
    assert observed_factor == runner.factor_digest(surface_digest)
    with pytest.raises(TypeError):
        executor.requests[0].tool_registry["read"]["args"] = ("changed",)
    assert isinstance(executor.requests[0].tool_registry["read"]["args"], tuple)

    records = protocol.AttemptLedger(layout.ledger, protocol.schedule_digest(schedule)).read()
    assert [record["event"] for record in records] == [
        "started",
        "completed",
        "started",
        "completed",
    ]
    assert len(list(layout.canonical.glob("*.json"))) == len(schedule)
    assert all("diff --git" in protocol.CheckpointStore(layout.canonical).load(entry).model_patch for entry in schedule)


def test_resume_skips_only_complete_pairs_and_never_calls_executor(tmp_path: Path) -> None:
    source, base_commit = _source_repo(tmp_path)
    instances = _instances(base_commit)
    document, schedule = _schedule_document(instances)
    layout = runner.GenerationLayout.create(tmp_path / "generation")
    first = RecordingExecutor()
    kwargs = {
        "schedule_document": document,
        "schedule": schedule,
        "layout": layout,
        "workspace_factory": runner.ExternalGitWorkspaceFactory(lambda _instance: source),
        "binding": _binding(),
        "tier_config": _tier(),
        "agent_module": _agent_module(),
    }
    runner.run_generation_pairs(executor=first, **kwargs)
    second = RecordingExecutor()

    runner.run_generation_pairs(executor=second, **kwargs)

    assert second.requests == []
    assert runner.pending_pairs(schedule, layout) == ()
    assert len(protocol.AttemptLedger(layout.ledger, protocol.schedule_digest(schedule)).read()) == 4


def test_interrupted_arm_requires_whole_pair_abort_and_retry(tmp_path: Path) -> None:
    source, base_commit = _source_repo(tmp_path)
    instances = _instances(base_commit)
    document, schedule = _schedule_document(instances)
    layout = runner.GenerationLayout.create(tmp_path / "generation")
    calls = 0
    requests: list[runner.ArmRunRequest] = []

    def interrupted(request: runner.ArmRunRequest) -> runner.ArmRunOutcome:
        nonlocal calls
        calls += 1
        requests.append(request)
        if calls == 2:
            raise RuntimeError("simulated host loss")
        (request.workspace / "module.py").write_text("VALUE = 4\n", encoding="utf-8")
        return runner.ArmRunOutcome(
            status="completed",
            quality_gate_decision=("satisfied" if request.quality_gate_enabled else "not_applicable"),
        )

    with pytest.raises(RuntimeError, match="host loss"):
        runner.run_generation_pairs(
            schedule_document=document,
            schedule=schedule,
            layout=layout,
            workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
            executor=interrupted,
            binding=_binding(),
            tier_config=_tier(),
            agent_module=_agent_module(),
        )
    assert not list(layout.canonical.glob("*.json"))
    with pytest.raises(protocol.ProtocolError, match="interrupted attempt"):
        runner.pending_pairs(schedule, layout)
    with pytest.raises(protocol.ProtocolError, match="interrupted attempt"):
        runner.run_generation_pairs(
            schedule_document=document,
            schedule=schedule,
            layout=layout,
            workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
            executor=RecordingExecutor(),
            binding=_binding(),
            tier_config=_tier(),
            agent_module=_agent_module(),
        )

    runner.abort_interrupted_pair(
        schedule,
        layout,
        pair_index=schedule[0].pair_index,
        reason_code="infrastructure_host_loss",
    )
    retry_start = len(requests)
    runner.run_generation_pairs(
        schedule_document=document,
        schedule=schedule,
        layout=layout,
        workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
        executor=interrupted,
        binding=_binding(),
        tier_config=_tier(),
        agent_module=_agent_module(),
    )
    assert len(requests[retry_start:]) == len(schedule)
    records = protocol.AttemptLedger(layout.ledger, protocol.schedule_digest(schedule)).read()
    assert [record["event"] for record in records[:4]] == [
        "started",
        "aborted",
        "started",
        "completed",
    ]
    assert records[2]["attempt_index"] == 1


def test_visible_git_created_by_executor_is_rejected_case_insensitively(tmp_path: Path) -> None:
    source, base_commit = _source_repo(tmp_path)
    instances = _instances(base_commit)
    document, schedule = _schedule_document(instances)
    layout = runner.GenerationLayout.create(tmp_path / "generation")

    def malicious(request: runner.ArmRunRequest) -> runner.ArmRunOutcome:
        (request.workspace / ".GiT").mkdir()
        return runner.ArmRunOutcome(
            status="completed",
            quality_gate_decision=("satisfied" if request.quality_gate_enabled else "not_applicable"),
        )

    with pytest.raises(protocol.ProtocolError, match="forbidden Git metadata"):
        runner.run_generation_pairs(
            schedule_document=document,
            schedule=schedule,
            layout=layout,
            workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
            executor=malicious,
            binding=_binding(),
            tier_config=_tier(),
            agent_module=_agent_module(),
        )
    assert not list(layout.canonical.glob("*.json"))


def test_workspace_factory_cannot_reuse_state_outside_exclusive_arm_root(
    tmp_path: Path,
) -> None:
    source, base_commit = _source_repo(tmp_path)
    instances = _instances(base_commit)
    document, schedule = _schedule_document(instances)
    layout = runner.GenerationLayout.create(tmp_path / "generation")
    factory = runner.ExternalGitWorkspaceFactory(lambda _instance: source)

    def misplaced_factory(*, instance, entry, destination):
        del destination
        return factory(
            instance=instance,
            entry=entry,
            destination=tmp_path / "reused-outside-attempt",
        )

    with pytest.raises(protocol.ProtocolError, match="exclusive arm"):
        runner.run_generation_pairs(
            schedule_document=document,
            schedule=schedule,
            layout=layout,
            workspace_factory=misplaced_factory,
            executor=RecordingExecutor(),
            binding=_binding(),
            tier_config=_tier(),
            agent_module=_agent_module(),
        )
    assert not list(layout.canonical.glob("*.json"))


def test_generation_receipt_recomputes_factor_ledger_and_canonical_hashes(tmp_path: Path) -> None:
    source, base_commit = _source_repo(tmp_path)
    instances = _instances(base_commit)
    document, schedule = _schedule_document(instances)
    layout = runner.GenerationLayout.create(tmp_path / "generation")
    agent_module = _agent_module()
    runner.run_generation_pairs(
        schedule_document=document,
        schedule=schedule,
        layout=layout,
        workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
        executor=RecordingExecutor(),
        binding=_binding(),
        tier_config=_tier(),
        agent_module=agent_module,
    )
    _registry, _specs, surface_digest = runner.build_identical_tool_surface(agent_module)

    with pytest.raises(protocol.ProtocolError, match="immutable run header"):
        runner.seal_generation_receipt(
            schedule=schedule,
            layout=layout,
            binding=runner.GenerationBinding(
                mio_commit="c" * 40,
                model_identity=protocol.EXPECTED_MODEL_IDENTITY,
                runtime_digest="e" * 64,
            ),
            tool_surface_sha256=surface_digest,
            observed_model_identity_before=protocol.EXPECTED_MODEL_IDENTITY,
            observed_model_identity_after=protocol.EXPECTED_MODEL_IDENTITY,
        )

    receipt_sha256 = runner.seal_generation_receipt(
        schedule=schedule,
        layout=layout,
        binding=_binding(),
        tool_surface_sha256=surface_digest,
        observed_model_identity_before=protocol.EXPECTED_MODEL_IDENTITY,
        observed_model_identity_after=protocol.EXPECTED_MODEL_IDENTITY,
    )

    assert receipt_sha256 == runner.verify_generation_receipt(
        receipt_path=layout.receipt,
        schedule=schedule,
        layout=layout,
        binding=_binding(),
        tool_surface_sha256=surface_digest,
    )
    receipt = json.loads(layout.receipt.read_text(encoding="utf-8"))
    assert receipt["pair_count"] == 2
    assert receipt["arm_count"] == 4
    assert receipt["factor_sha256"] == runner.factor_digest(surface_digest)
    assert receipt["contains_model_text_or_evaluator_output"] is False
    assert receipt["evidence_class"] == "non_evidence_smoke"
    assert receipt["confirmatory_evidence_admissible"] is False
    assert set(receipt["confirmatory_blockers"]) == set(runner.CONFIRMATORY_BLOCKERS)
    assert receipt["run_header_sha256"] == protocol.sha256_file(layout.run_header)
    header = json.loads(layout.run_header.read_text(encoding="utf-8"))
    assert header["factor_sha256"] == receipt["factor_sha256"]
    assert header["generation_binding"] == receipt["generation_binding"]
    assert layout.ledger.parent == layout.root
    assert layout.run_header.parent == layout.root
    assert layout.canonical.parent == layout.root
    assert layout.ledger not in layout.canonical.parents

    checkpoint = protocol.CheckpointStore(layout.canonical).path_for(schedule[0])
    checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
    with pytest.raises((protocol.ProtocolError, json.JSONDecodeError)):
        runner.verify_generation_receipt(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=layout,
            binding=_binding(),
            tool_surface_sha256=surface_digest,
        )


def test_receipt_verification_does_not_repair_deleted_canonical_output(tmp_path: Path) -> None:
    source, base_commit = _source_repo(tmp_path)
    instances = _instances(base_commit)
    document, schedule = _schedule_document(instances)
    layout = runner.GenerationLayout.create(tmp_path / "generation")
    agent_module = _agent_module()
    runner.run_generation_pairs(
        schedule_document=document,
        schedule=schedule,
        layout=layout,
        workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
        executor=RecordingExecutor(),
        binding=_binding(),
        tier_config=_tier(),
        agent_module=agent_module,
    )
    _registry, _specs, surface_digest = runner.build_identical_tool_surface(agent_module)
    runner.seal_generation_receipt(
        schedule=schedule,
        layout=layout,
        binding=_binding(),
        tool_surface_sha256=surface_digest,
        observed_model_identity_before=protocol.EXPECTED_MODEL_IDENTITY,
        observed_model_identity_after=protocol.EXPECTED_MODEL_IDENTITY,
    )
    deleted = protocol.CheckpointStore(layout.canonical).path_for(schedule[0])
    deleted.unlink()

    with pytest.raises(protocol.ProtocolError, match="canonical pair is incomplete"):
        runner.verify_generation_receipt(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=layout,
            binding=_binding(),
            tool_surface_sha256=surface_digest,
        )
    assert not deleted.exists()


def test_target_only_tier_and_exact_model_identity_are_mandatory() -> None:
    runner.validate_target_only_tier(_tier())
    with pytest.raises(protocol.ProtocolError, match="target-only"):
        runner.validate_target_only_tier(_tier(drafter_backend="dflash"))
    with pytest.raises(protocol.ProtocolError, match="Qwen 3.6 27B"):
        runner.GenerationBinding(
            mio_commit="b" * 40,
            model_identity="local-sha256-v1:" + "a" * 64,
            runtime_digest="d" * 64,
        )


def test_recomputed_schedule_summary_cannot_hide_reordered_pairs(tmp_path: Path) -> None:
    _source, base_commit = _source_repo(tmp_path)
    instances = _instances(base_commit)
    document, schedule = _schedule_document(instances)
    reordered = tuple((*schedule[2:4], *schedule[0:2]))
    document["source_free_summary"] = protocol.source_free_schedule_summary(reordered)

    with pytest.raises(protocol.ProtocolError, match="deterministic balanced schedule"):
        runner.run_generation_pairs(
            schedule_document=document,
            schedule=reordered,
            layout=runner.GenerationLayout.create(tmp_path / "generation"),
            workspace_factory=lambda **_kwargs: pytest.fail("workspace must not be created"),
            executor=lambda _request: pytest.fail("model must not run"),
            binding=_binding(),
            tier_config=_tier(),
            agent_module=_agent_module(),
        )


def test_confirmatory_schedule_is_hard_blocked_before_workspace_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, base_commit = _source_repo(tmp_path)
    instances = _instances(base_commit)
    document, schedule = _schedule_document(instances)
    document["evidence_class"] = "confirmatory"
    monkeypatch.setattr(
        protocol,
        "PUBLIC_SNAPSHOT_SHA256",
        document["dataset_public_snapshot_sha256"],
    )
    layout = runner.GenerationLayout.create(tmp_path / "generation")
    executor = RecordingExecutor()

    with pytest.raises(protocol.ProtocolError, match="confirmatory SWE-bench is blocked"):
        runner.run_generation_pairs(
            schedule_document=document,
            schedule=schedule,
            layout=layout,
            workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
            executor=executor,
            binding=_binding(),
            tier_config=_tier(),
            agent_module=_agent_module(),
        )
    assert executor.requests == []
    assert not layout.ledger.exists()


def test_confirmatory_manifest_cannot_rebind_altered_task_with_same_ids(
    tmp_path: Path,
) -> None:
    _source, base_commit = _source_repo(tmp_path)
    instances = _instances(base_commit)
    document, schedule = _schedule_document(instances)
    document["evidence_class"] = "confirmatory"
    document["public_instances"][0]["problem_statement"] = "Altered after sealing."
    altered = tuple(protocol.PublicInstance.from_mapping(row) for row in document["public_instances"])
    document["dataset_public_snapshot_sha256"] = protocol.sha256_bytes(
        protocol.canonical_jsonl_bytes(
            [instance.as_dict() for instance in sorted(altered, key=lambda item: item.instance_id)]
        )
    )

    with pytest.raises(protocol.ProtocolError, match="exact official public snapshot"):
        runner._validate_schedule_document(document, schedule)


def test_native_executor_resets_cache_conversation_and_injects_exact_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent
    from mio.agent_policy import AgentToolPolicy

    class FakeEngine:
        tier_config = _tier()
        _draft_model = None
        _dspark_runtime = None
        _prefix_cache = {"dirty": object()}
        _last_prompt_tokens = [1, 2]
        _pending_assistant_prefill = "<"
        resets = 0

        def _prefix_cache_invalidate(self) -> None:
            self.resets += 1
            self._prefix_cache.clear()

    engine = FakeEngine()
    native = runner.NativeMioArmExecutor(
        engine=engine,
        manager=object(),
        config=object(),
        tier="large",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    registry, specs, _digest = runner.build_identical_tool_surface(_agent_module())
    message_lists: list[list] = []
    budgets = []

    def fake_process(_instruction, _engine, _manager, _config, state):
        assert not engine._prefix_cache
        assert engine._last_prompt_tokens == []
        assert engine._pending_assistant_prefill == ""
        message_lists.append(state["messages"])
        budgets.append(state["execution_budget"])
        return SimpleNamespace(
            terminal_reason="model_final",
            rounds=(SimpleNamespace(completion_tokens=7),),
            completion_tokens=7,
            quality_gate={"decision": "satisfied"},
            tool_calls=2,
            wall_time_s=0.5,
        )

    monkeypatch.setattr(agent, "_process_user_input", fake_process)

    for index, enabled in enumerate((False, True)):
        request = runner.ArmRunRequest(
            entry=protocol.ScheduleEntry(
                pair_index=0,
                execution_index=index,
                instance_id="owner__repository-1",
                instance_digest=protocol._instance_digest("owner__repository-1"),
                condition="gate_on" if enabled else "gate_off",
                position_in_pair=index,
            ),
            instruction="safe public task",
            workspace=workspace,
            cache_directory=cache,
            tool_registry=registry,
            tool_specs=specs,
            tool_policy=AgentToolPolicy.coding_workspace(workspace, allow_network=False),
            quality_gate_enabled=enabled,
            coding_effort="medium",
            seed=1,
        )
        outcome = native(request)
        assert outcome.status == "completed"
        assert outcome.quality_gate_decision == ("satisfied" if enabled else "not_applicable")
        engine._prefix_cache["dirty"] = object()

    assert engine.resets == 2
    assert len(message_lists) == 2 and message_lists[0] is not message_lists[1]
    assert all(budget.max_rounds == 12 for budget in budgets)
    assert all(budget.max_tool_calls == 32 for budget in budgets)
    assert all(budget.max_output_tokens == 24_576 for budget in budgets)
    assert all(budget.max_context_tokens == 32_768 for budget in budgets)
    assert all(budget.max_wall_seconds == 1_800 for budget in budgets)


def test_native_model_exception_is_sealed_as_nonretryable_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent
    from mio.agent_policy import AgentToolPolicy

    engine = SimpleNamespace(
        tier_config=_tier(),
        _draft_model=None,
        _dspark_runtime=None,
        _prefix_cache={},
        _last_prompt_tokens=[],
        _pending_assistant_prefill="",
        _prefix_cache_invalidate=lambda: None,
    )
    native = runner.NativeMioArmExecutor(
        engine=engine,
        manager=object(),
        config=object(),
        tier="large",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    registry, specs, _digest = runner.build_identical_tool_surface(_agent_module())
    monkeypatch.setattr(
        agent,
        "_process_user_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model failed")),
    )

    outcome = native(
        runner.ArmRunRequest(
            entry=protocol.ScheduleEntry(
                pair_index=0,
                execution_index=0,
                instance_id="owner__repository-1",
                instance_digest=protocol._instance_digest("owner__repository-1"),
                condition="gate_on",
                position_in_pair=0,
            ),
            instruction="safe public task",
            workspace=workspace,
            cache_directory=cache,
            tool_registry=registry,
            tool_specs=specs,
            tool_policy=AgentToolPolicy.coding_workspace(workspace, allow_network=False),
            quality_gate_enabled=True,
            coding_effort="medium",
            seed=1,
        )
    )

    assert outcome.status == "model_error"
    assert outcome.quality_gate_decision == "incomplete"
    assert 0 <= outcome.wall_seconds <= protocol.MAX_AGENT_WALL_SECONDS
