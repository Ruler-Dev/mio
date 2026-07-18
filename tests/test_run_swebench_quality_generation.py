from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
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


def _attested_mio_repo(tmp_path: Path, name: str = "attested-mio") -> Path:
    repository = tmp_path / name
    (repository / "mio").mkdir(parents=True, mode=0o700)
    (repository / "scripts").mkdir()
    (repository / "mio" / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text('[project]\nname = "mio"\n', encoding="utf-8")
    (repository / "scripts" / "run_swebench_quality_generation.py").write_text(
        "# attested test runner\n",
        encoding="utf-8",
    )
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Mio Test")
    _git(repository, "config", "user.email", "mio@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "attested source")
    return repository.resolve(strict=True)


def _local_model_bundle(tmp_path: Path, name: str = "attested-model") -> Path:
    model = tmp_path / name
    model.mkdir(mode=0o700)
    (model / "config.json").write_text('{"model_type":"qwen-test"}\n', encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"complete-test-weight-bytes")
    (model / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    return model.resolve(strict=True)


def _runtime_document(marker: str = "stable") -> dict:
    return {
        "schema": runner.RUNTIME_ATTESTATION_SCHEMA,
        "python": {"implementation": "CPython", "version": "test", "marker": marker},
        "installed_distributions": [{"name": "mlx", "version": "test"}],
        "critical_distribution_contents": [{"name": "mlx", "version": "test", "content_sha256": "a" * 64}],
        "absolute_paths_serialized": False,
        "environment_values_serialized": False,
        "environment_value_hashes_serialized": True,
        "full_package_inventory_serialized": True,
    }


def _automatic_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "automatic",
) -> tuple[runner.GenerationBinding, Path, Path]:
    from experimental.effort.model_identity import fingerprint_local_model

    repository = _attested_mio_repo(tmp_path, f"{name}-mio")
    model = _local_model_bundle(tmp_path, f"{name}-model")
    identity = fingerprint_local_model(model).revision
    monkeypatch.setattr(protocol, "EXPECTED_MODEL_IDENTITY", identity)
    monkeypatch.setattr(runner, "_collect_runtime_document", _runtime_document)
    monkeypatch.setattr(runner, "_assert_executing_mio_tree", lambda _repository: None)
    binding = runner.GenerationBinding.automatic_local(
        repository_root=repository,
        model_root=model,
    )
    return binding, repository, model


def _native_executor_for_model(
    model: Path,
    *,
    tier_name: str = "large",
    require_raw_target_telemetry: bool = False,
) -> runner.NativeMioArmExecutor:
    from mio.config import MioConfig, TierConfig
    from mio.engine import MioEngine
    from mio.model_manager import ModelManager

    tier_config = TierConfig(
        name=tier_name,
        target_model=str(model),
        draft_model="disabled",
        context_window=32_768,
        max_output_tokens=4_096,
        drafter_backend="target_ar",
        tq_bits=16,
        pq_bits=16,
        bmp_paths=1,
        ddtree_budget=0,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )
    engine = MioEngine(tier_config=tier_config)
    engine._loaded = True
    engine._target_model = object()
    engine._tokenizer = object()
    engine._target_meta = {"resolved_model_ref": str(model)}
    manager = ModelManager(MioConfig(tiers={tier_name: tier_config}, active_tiers=[]))
    manager._engines[tier_name] = engine
    return runner.NativeMioArmExecutor(
        engine=engine,
        manager=manager,
        config=manager.config,
        tier=tier_name,
        require_raw_target_telemetry=require_raw_target_telemetry,
    )


def _duck_typed_native_executor_for_model(
    model: Path,
    *,
    tier_name: str = "large",
) -> runner.NativeMioArmExecutor:
    tier_config = _tier()
    tier_config.target_model = str(model)
    engine = SimpleNamespace(
        tier_config=tier_config,
        is_loaded=True,
        _target_model=object(),
        _tokenizer=object(),
        _target_meta={"resolved_model_ref": str(model)},
    )

    class Manager:
        def loaded_tiers(self):
            return [tier_name]

        def get_engine(self, requested_tier):
            assert requested_tier == tier_name
            return engine

    return runner.NativeMioArmExecutor(
        engine=engine,
        manager=Manager(),
        config=object(),
        tier=tier_name,
    )


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
    return runner.GenerationBinding.for_non_evidence_smoke(
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
            executor=interrupted,
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
            binding=runner.GenerationBinding.for_non_evidence_smoke(
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

    assert receipt_sha256 == runner.verify_legacy_generation_artifacts(
        receipt_path=layout.receipt,
        schedule=schedule,
        layout=layout,
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
        runner.verify_legacy_generation_artifacts(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=layout,
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
        runner.verify_legacy_generation_artifacts(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=layout,
            tool_surface_sha256=surface_digest,
        )
    assert not deleted.exists()


def test_receipt_verification_rejects_hardlinked_receipt_and_run_header(tmp_path: Path) -> None:
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

    receipt_alias = tmp_path / "receipt-alias.json"
    os.link(layout.receipt, receipt_alias)
    with pytest.raises(protocol.ProtocolError, match="single-link"):
        runner.verify_legacy_generation_artifacts(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=layout,
            tool_surface_sha256=surface_digest,
        )
    receipt_alias.unlink()

    header_alias = tmp_path / "run-header-alias.json"
    os.link(layout.run_header, header_alias)
    with pytest.raises(protocol.ProtocolError, match="single-link"):
        runner.verify_legacy_generation_artifacts(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=layout,
            tool_surface_sha256=surface_digest,
        )


def test_automatic_binding_attests_exact_git_model_and_runtime_without_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, repository, model = _automatic_binding(tmp_path, monkeypatch)

    binding.validate_for_run(evidence_run=True)
    attestation = binding.attestation_dict()
    serialized = json.dumps(attestation, sort_keys=True)
    assert binding.binding_source == runner.AUTOMATIC_BINDING_SOURCE
    assert binding.mio_commit == _git(repository, "rev-parse", "HEAD")
    assert binding.model_identity == protocol.EXPECTED_MODEL_IDENTITY
    assert attestation["git"]["worktree_clean"] is True
    assert attestation["model"]["complete_file_bytes_hashed"] is True
    assert attestation["model"]["manifest_sha256"] == binding.model_identity.removeprefix("local-sha256-v1:")
    assert attestation["runtime"]["digest"] == binding.runtime_digest
    assert attestation["runtime"]["critical_versions"] == [{"name": "mlx", "version": "test"}]
    assert "manifest" not in attestation["runtime"]
    assert "installed_distributions" not in serialized
    assert "value_sha256" not in serialized
    assert "content_sha256" not in serialized
    assert attestation["privacy"] == {
        "absolute_local_paths_serialized": False,
        "environment_values_serialized": False,
        "environment_value_hashes_serialized": False,
        "full_package_inventory_serialized": False,
        "private_runtime_manifest_retained_in_memory": True,
    }
    assert attestation["end_to_end_confirmatory_chain_of_custody_proven"] is False
    assert any("chain_of_custody" in blocker for blocker in runner.CONFIRMATORY_BLOCKERS)
    assert str(repository) not in serialized
    assert str(model) not in serialized

    executor = _native_executor_for_model(model)
    header = runner.build_run_header(
        schedule_document={
            "evidence_class": "non_evidence_smoke",
            "dataset_public_snapshot_sha256": "a" * 64,
        },
        schedule=(),
        binding=binding,
        tool_surface_sha256="b" * 64,
        executor=executor,
        workspace_factory=lambda **_kwargs: pytest.fail("header construction must not create a workspace"),
        tier_config=executor.engine.tier_config,
    )
    header_serialized = json.dumps(header, sort_keys=True)
    assert "installed_distributions" not in header_serialized
    assert "value_sha256" not in header_serialized
    assert "content_sha256" not in header_serialized


@pytest.mark.parametrize("change", ["tracked", "untracked"])
def test_automatic_binding_rejects_dirty_or_untracked_mio_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from experimental.effort.model_identity import fingerprint_local_model

    repository = _attested_mio_repo(tmp_path)
    model = _local_model_bundle(tmp_path)
    monkeypatch.setattr(protocol, "EXPECTED_MODEL_IDENTITY", fingerprint_local_model(model).revision)
    monkeypatch.setattr(runner, "_collect_runtime_document", _runtime_document)
    monkeypatch.setattr(runner, "_assert_executing_mio_tree", lambda _repository: None)
    if change == "tracked":
        (repository / "mio" / "__init__.py").write_text("changed = True\n", encoding="utf-8")
    else:
        (repository / "untracked.py").write_text("changed = True\n", encoding="utf-8")

    with pytest.raises(protocol.ProtocolError, match="clean Mio worktree"):
        runner.GenerationBinding.automatic_local(
            repository_root=repository,
            model_root=model,
        )


def test_automatic_binding_detects_git_model_and_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, repository, model = _automatic_binding(tmp_path, monkeypatch)
    (repository / "untracked.py").write_text("drift = True\n", encoding="utf-8")
    with pytest.raises(protocol.ProtocolError, match="clean Mio worktree"):
        binding.validate_for_run(evidence_run=False)
    (repository / "untracked.py").unlink()

    (model / "model-00001-of-00001.safetensors").write_bytes(b"mutated-weight-bytes")
    with pytest.raises(protocol.ProtocolError, match="does not match the frozen"):
        binding.validate_for_run(evidence_run=False)
    (model / "model-00001-of-00001.safetensors").write_bytes(b"complete-test-weight-bytes")

    monkeypatch.setattr(runner, "_collect_runtime_document", lambda: _runtime_document("drifted"))
    with pytest.raises(protocol.ProtocolError, match="runtime/dependency environment.*drifted"):
        binding.validate_for_run(evidence_run=False)


def test_automatic_binding_rejects_model_path_alias_and_hardlinked_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experimental.effort.model_identity import fingerprint_local_model

    repository = _attested_mio_repo(tmp_path)
    model = _local_model_bundle(tmp_path)
    monkeypatch.setattr(protocol, "EXPECTED_MODEL_IDENTITY", fingerprint_local_model(model).revision)
    monkeypatch.setattr(runner, "_collect_runtime_document", _runtime_document)
    monkeypatch.setattr(runner, "_assert_executing_mio_tree", lambda _repository: None)
    alias = tmp_path / "model-alias"
    alias.symlink_to(model, target_is_directory=True)

    with pytest.raises(protocol.ProtocolError, match="symlink component"):
        runner.GenerationBinding.automatic_local(
            repository_root=repository,
            model_root=alias,
        )

    alias.unlink()
    os.link(model / "model-00001-of-00001.safetensors", tmp_path / "weight-alias")
    with pytest.raises(protocol.ProtocolError, match="single-link"):
        runner.GenerationBinding.automatic_local(
            repository_root=repository,
            model_root=model,
        )


def test_automatic_binding_rejects_wrong_full_model_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _attested_mio_repo(tmp_path)
    model = _local_model_bundle(tmp_path)
    monkeypatch.setattr(protocol, "EXPECTED_MODEL_IDENTITY", "local-sha256-v1:" + "0" * 64)
    monkeypatch.setattr(runner, "_collect_runtime_document", _runtime_document)
    monkeypatch.setattr(runner, "_assert_executing_mio_tree", lambda _repository: None)

    with pytest.raises(protocol.ProtocolError, match="does not match the frozen"):
        runner.GenerationBinding.automatic_local(
            repository_root=repository,
            model_root=model,
        )


def test_automatic_binding_rejects_clean_clone_other_than_executing_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experimental.effort.model_identity import fingerprint_local_model

    repository = _attested_mio_repo(tmp_path)
    model = _local_model_bundle(tmp_path)
    monkeypatch.setattr(protocol, "EXPECTED_MODEL_IDENTITY", fingerprint_local_model(model).revision)
    monkeypatch.setattr(runner, "_collect_runtime_document", _runtime_document)

    with pytest.raises(protocol.ProtocolError, match="differs from the executing runner"):
        runner.GenerationBinding.automatic_local(
            repository_root=repository,
            model_root=model,
        )


def test_automatic_binding_rejects_ignored_runtime_source_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experimental.effort.model_identity import fingerprint_local_model

    repository = _attested_mio_repo(tmp_path)
    model = _local_model_bundle(tmp_path)
    (repository / ".gitignore").write_text("mio/shadow.py\n", encoding="utf-8")
    _git(repository, "add", ".gitignore")
    _git(repository, "commit", "--quiet", "-m", "ignore shadow")
    (repository / "mio" / "shadow.py").write_text("SHADOW = True\n", encoding="utf-8")
    monkeypatch.setattr(protocol, "EXPECTED_MODEL_IDENTITY", fingerprint_local_model(model).revision)
    monkeypatch.setattr(runner, "_collect_runtime_document", _runtime_document)
    monkeypatch.setattr(runner, "_assert_executing_mio_tree", lambda _repository: None)

    with pytest.raises(protocol.ProtocolError, match="ignored runtime-relevant"):
        runner.GenerationBinding.automatic_local(
            repository_root=repository,
            model_root=model,
        )


def test_automatic_binding_requires_loaded_native_engine_for_same_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, _repository, model = _automatic_binding(tmp_path, monkeypatch)

    with pytest.raises(protocol.ProtocolError, match="native Mio executor"):
        binding.validate_for_run(
            evidence_run=False,
            executor=RecordingExecutor(),
            tier_config=_tier(),
            require_executor_binding=True,
        )

    duck_typed = _duck_typed_native_executor_for_model(model)
    with pytest.raises(protocol.ProtocolError, match="exact production engine and manager"):
        binding.validate_for_run(
            evidence_run=False,
            executor=duck_typed,
            tier_config=duck_typed.engine.tier_config,
            require_executor_binding=True,
        )

    other_model = _local_model_bundle(tmp_path, "other-loaded-model")
    other_executor = _native_executor_for_model(other_model)
    with pytest.raises(protocol.ProtocolError, match="differs from the automatically fingerprinted model"):
        binding.validate_for_run(
            evidence_run=False,
            executor=other_executor,
            tier_config=other_executor.engine.tier_config,
            require_executor_binding=True,
        )

    executor = _native_executor_for_model(model)
    with pytest.raises(protocol.ProtocolError, match="not the exact loaded engine tier config"):
        binding.validate_for_run(
            evidence_run=False,
            executor=executor,
            tier_config=_tier(),
            require_executor_binding=True,
        )

    executor.engine.tier_config.temperature = 0.5
    with pytest.raises(protocol.ProtocolError, match="differs from frozen controls"):
        binding.validate_for_run(
            evidence_run=False,
            executor=executor,
            tier_config=executor.engine.tier_config,
            require_executor_binding=True,
        )
    executor.engine.tier_config.temperature = 0.0

    observed = binding.validate_for_run(
        evidence_run=False,
        executor=executor,
        tier_config=executor.engine.tier_config,
        require_executor_binding=True,
    )
    assert observed["automatic"] is True
    assert observed["model_identity"] == binding.model_identity


def test_target_only_tier_and_exact_model_identity_are_mandatory() -> None:
    runner.validate_target_only_tier(_tier())
    with pytest.raises(protocol.ProtocolError, match="target-only"):
        runner.validate_target_only_tier(_tier(drafter_backend="dflash"))
    with pytest.raises(protocol.ProtocolError, match="Qwen 3.6 27B"):
        runner.GenerationBinding.for_non_evidence_smoke(
            mio_commit="b" * 40,
            model_identity="local-sha256-v1:" + "a" * 64,
            runtime_digest="d" * 64,
        )
    with pytest.raises(TypeError):
        runner.GenerationBinding(  # type: ignore[call-arg]
            mio_commit="b" * 40,
            model_identity=protocol.EXPECTED_MODEL_IDENTITY,
            runtime_digest="d" * 64,
        )


def test_generation_runner_script_entrypoint_is_importable() -> None:
    completed = subprocess.run(
        ["python3", str(runner.ROOT / "scripts" / "run_swebench_quality_generation.py")],
        cwd=runner.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "confirmatory evidence remains hard-blocked" in completed.stdout


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


def test_confirmatory_schedule_rejects_manual_binding_before_workspace_or_model(
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

    with pytest.raises(protocol.ProtocolError, match="automatic preflight fingerprints"):
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


def test_automatically_attested_confirmatory_schedule_remains_hard_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, _repository, model = _automatic_binding(tmp_path, monkeypatch, name="confirmatory")
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
    executor = _native_executor_for_model(model)

    with pytest.raises(protocol.ProtocolError, match="confirmatory SWE-bench is blocked"):
        runner.run_generation_pairs(
            schedule_document=document,
            schedule=schedule,
            layout=layout,
            workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
            executor=executor,
            binding=binding,
            tier_config=executor.engine.tier_config,
            agent_module=_agent_module(),
        )
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


def _raw_target_round(index: int, completion_tokens: int = 2, **changes):
    prompt_tokens = 10 + index
    prefill_ns = 100
    decode_ns = 200
    physical_decode_tokens = completion_tokens + 1
    values = {
        "round_index": index,
        "prompt_tokens": prompt_tokens,
        "total_time_s": 0.0000003,
        "prompt_tps": prompt_tokens * 1_000_000_000 / prefill_ns,
        "generation_tps": physical_decode_tokens * 1_000_000_000 / decode_ns,
        "generation_backend": "baseline",
        "fallback_ar": False,
        "drafter_requested": "target_ar",
        "drafter_selected": "baseline",
        "drafter_ref": None,
        "timing_source": "runtime_raw_ns",
        "prefill_ns": prefill_ns,
        "decode_ns": decode_ns,
        "model_total_ns": 300,
        "completion_tokens": completion_tokens,
        "logical_prompt_tokens": prompt_tokens,
        "physical_prefill_tokens": prompt_tokens,
        "physical_decode_tokens": physical_decode_tokens,
        "warm_offset": 0,
        "warm_offset_tokens": 0,
        "phase_censored": False,
        "deadline_hit": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _raw_target_result():
    def tool_trace(sequence: int, round_index: int, tool_name: str = "read"):
        is_command = tool_name == "validate"
        return SimpleNamespace(
            sequence=sequence,
            round_index=round_index,
            tool_name=tool_name,
            operation=tool_name,
            permission="shell" if is_command else "read",
            allowed=True,
            outcome="ok",
            target_sha256="a" * 64,
            duration_ns=10,
            effective_timeout_ns=(300_000_000_000 if is_command else 30_000_000_000),
            exit_code_or_signal=0 if is_command else None,
            output_chars=10,
            audit_count=1,
            audit_sha256="b" * 64,
            timeout_enforced=not is_command,
            telemetry_complete=True,
            effect_unknown=False,
        )

    return SimpleNamespace(
        terminal_reason="model_final",
        rounds=(_raw_target_round(0, 2), _raw_target_round(1, 3)),
        completion_tokens=5,
        quality_gate=None,
        tool_calls=2,
        tool_events=(
            tool_trace(0, 0),
            tool_trace(1, 0, "validate"),
        ),
        tool_telemetry_complete=True,
        tool_result_chars=20,
        wall_time_s=0.0,
        budget_exhaustion=None,
    )


def _passing_quality_report(instruction: str, initial, current) -> dict[str, object]:
    return {
        "schema": "mio.coding-quality-gate.v1",
        "enabled": True,
        "effort": "medium",
        "intent": "code_change_requested",
        "request_sha256": protocol.sha256_bytes(instruction.encode("utf-8")),
        "decision": "pass",
        "phase": "passed",
        "activated": True,
        "satisfied": True,
        "mutation_epoch": 1,
        "changed_kinds": ["code"],
        "snapshot_complete": current.complete,
        "snapshot_method": current.method,
        "snapshot_error_codes": list(current.error_codes),
        "initial_revision_sha256": initial.revision_sha256,
        "current_revision_sha256": current.revision_sha256,
        "required": ["test_or_build"],
        "missing": [],
        "validation_counts": {"test": 1, "build": 0, "static": 0, "diff": 0, "review": 0},
        "validation_attempts": 1,
        "successful_reads": 1,
    }


def _observing_quality_report(instruction: str, snapshot) -> dict[str, object]:
    return {
        "schema": "mio.coding-quality-gate.v1",
        "enabled": True,
        "effort": "medium",
        "intent": "inspect",
        "request_sha256": protocol.sha256_bytes(instruction.encode("utf-8")),
        "decision": "not_applicable",
        "phase": "observing",
        "activated": False,
        "satisfied": True,
        "mutation_epoch": 0,
        "changed_kinds": [],
        "snapshot_complete": snapshot.complete,
        "snapshot_method": snapshot.method,
        "snapshot_error_codes": list(snapshot.error_codes),
        "initial_revision_sha256": snapshot.revision_sha256,
        "current_revision_sha256": snapshot.revision_sha256,
        "required": [],
        "missing": [],
        "validation_counts": {"test": 1, "build": 0, "static": 0, "diff": 0, "review": 0},
        "validation_attempts": 1,
        "successful_reads": 1,
    }


def _sealed_portable_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from mio import agent

    binding, _repository, model = _automatic_binding(tmp_path, monkeypatch, name="portable")
    source, base_commit = _source_repo(tmp_path)
    document, schedule = _schedule_document(_instances(base_commit))
    layout = runner.GenerationLayout.create(tmp_path / "generation", portable_artifacts=True)
    executor = _native_executor_for_model(
        model,
        require_raw_target_telemetry=True,
    )
    agent_module = _agent_module()

    def fake_process(_instruction, _engine, _manager, _config, state):
        from mio.coding_quality import snapshot_workspaces

        workspace = state["tool_policy"].workspace_roots[0]
        initial = snapshot_workspaces((workspace,))
        (workspace / "module.py").write_text(
            f"VALUE = {2 if state['quality_gate_enabled'] else 3}\n",
            encoding="utf-8",
        )
        result = _raw_target_result()
        if state["quality_gate_enabled"]:
            current = snapshot_workspaces((workspace,))
            result.quality_gate = _passing_quality_report(_instruction, initial, current)
        return result

    monkeypatch.setattr(agent, "_process_user_input", fake_process)
    runner.run_generation_pairs(
        schedule_document=document,
        schedule=schedule,
        layout=layout,
        workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
        executor=executor,
        binding=binding,
        tier_config=executor.engine.tier_config,
        agent_module=agent_module,
        require_portable_artifacts=True,
    )
    _registry, _specs, surface_sha256 = runner.build_identical_tool_surface(agent_module)
    receipt_sha256 = runner.seal_generation_receipt(
        schedule=schedule,
        layout=layout,
        binding=binding,
        tool_surface_sha256=surface_sha256,
        observed_model_identity_before=binding.model_identity,
        observed_model_identity_after=binding.model_identity,
    )
    return layout, schedule, binding, surface_sha256, receipt_sha256


def _raw_target_native_request(tmp_path: Path):
    from mio.agent_policy import AgentToolPolicy

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    registry, specs, _digest = runner.build_identical_tool_surface(_agent_module())
    return runner.ArmRunRequest(
        entry=protocol.ScheduleEntry(
            pair_index=0,
            execution_index=0,
            instance_id="owner__repository-1",
            instance_digest=protocol._instance_digest("owner__repository-1"),
            condition="gate_off",
            position_in_pair=0,
        ),
        instruction="safe public task",
        workspace=workspace,
        cache_directory=cache,
        tool_registry=registry,
        tool_specs=specs,
        tool_policy=AgentToolPolicy.coding_workspace(workspace, allow_network=False),
        quality_gate_enabled=False,
        coding_effort="medium",
        seed=1,
    )


def test_native_raw_target_telemetry_admits_complete_contiguous_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent

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
        require_raw_target_telemetry=True,
    )
    result = _raw_target_result()
    monkeypatch.setattr(agent, "_process_user_input", lambda *_args, **_kwargs: result)

    outcome = native(_raw_target_native_request(tmp_path))

    assert outcome.status == "completed"
    assert outcome.output_tokens == 5
    assert outcome.tool_calls == 2


def test_native_raw_target_telemetry_derives_timeout_status_from_terminal_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent

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
        require_raw_target_telemetry=True,
    )
    result = _raw_target_result()
    timeout = result.tool_events[-1]
    timeout.round_index = 1
    timeout.tool_name = "read"
    timeout.operation = "read"
    timeout.permission = "read"
    timeout.outcome = "timeout"
    timeout.duration_ns = 1
    timeout.effective_timeout_ns = 1
    timeout.exit_code_or_signal = None
    timeout.timeout_enforced = True
    timeout.telemetry_complete = False
    timeout.effect_unknown = True
    result.terminal_reason = "tool_timeout"
    result.tool_telemetry_complete = False
    monkeypatch.setattr(agent, "_process_user_input", lambda *_args, **_kwargs: result)

    outcome = native(_raw_target_native_request(tmp_path))
    turn, quality, _rounds, tools = outcome.telemetry.document()

    assert outcome.status == "timeout"
    assert outcome.quality_gate_decision == "not_applicable"
    assert turn["terminal_reason"] == "tool_timeout"
    assert turn["status"] == "timeout"
    assert turn["tool_telemetry_complete"] is False
    assert quality["enabled"] is False
    assert tools[-1]["effect_unknown"] is True


def test_native_raw_target_telemetry_accepts_denied_unrecognized_validate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent

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
        require_raw_target_telemetry=True,
    )
    result = _raw_target_result()
    rejected = result.tool_events[-1]
    rejected.allowed = False
    rejected.outcome = "unrecognized"
    rejected.exit_code_or_signal = None
    monkeypatch.setattr(agent, "_process_user_input", lambda *_args, **_kwargs: result)

    outcome = native(_raw_target_native_request(tmp_path))

    assert outcome.status == "completed"
    assert outcome.telemetry.document()[3][-1]["outcome"] == "unrecognized"


def test_native_gate_on_observation_is_completed_and_satisfied_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent
    from mio.coding_quality import snapshot_workspaces

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
        require_raw_target_telemetry=True,
    )
    request = _raw_target_native_request(tmp_path)
    request = replace(
        request,
        entry=replace(request.entry, condition="gate_on"),
        quality_gate_enabled=True,
    )
    result = _raw_target_result()
    result.quality_gate = _observing_quality_report(
        request.instruction,
        snapshot_workspaces((request.workspace,)),
    )
    monkeypatch.setattr(agent, "_process_user_input", lambda *_args, **_kwargs: result)

    outcome = native(request)
    turn, quality, _rounds, _tools = outcome.telemetry.document()

    assert outcome.status == "completed"
    assert outcome.quality_gate_decision == "satisfied"
    assert turn["status"] == "completed"
    assert quality["decision"] == "not_applicable"
    assert quality["phase"] == "observing"
    assert quality["activated"] is False
    assert quality["satisfied"] is True


def test_native_budget_exhaustion_is_sealed_as_bounded_incomplete_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent

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
        require_raw_target_telemetry=True,
    )
    result = _raw_target_result()
    result.rounds = tuple(_raw_target_round(index, 0) for index in range(runner.TARGET_MAX_ROUNDS))
    result.completion_tokens = 0
    result.tool_calls = 0
    result.tool_events = ()
    result.tool_result_chars = 0
    result.terminal_reason = "budget_exhausted"
    result.budget_exhaustion = f"model round limit {runner.TARGET_MAX_ROUNDS} reached"
    monkeypatch.setattr(agent, "_process_user_input", lambda *_args, **_kwargs: result)

    outcome = native(_raw_target_native_request(tmp_path))
    turn, _quality, rounds, _tools = outcome.telemetry.document()

    assert outcome.status == "incomplete"
    assert len(rounds) == runner.TARGET_MAX_ROUNDS
    assert turn["budget_exhaustion_kind"] == "model_rounds"


def test_quality_deadline_serializes_complete_wall_overhead_and_capped_checkpoint_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent
    from mio.coding_quality import snapshot_workspaces

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
        require_raw_target_telemetry=True,
    )
    request = _raw_target_native_request(tmp_path)
    request = replace(request, entry=replace(request.entry, condition="gate_on"), quality_gate_enabled=True)

    def deadline_result(instruction, _engine, _manager, _config, _state):
        initial = snapshot_workspaces((request.workspace,))
        (request.workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
        current = snapshot_workspaces((request.workspace,))
        result = _raw_target_result()
        result.rounds = (_raw_target_round(0, 2, deadline_hit=True, phase_censored=True),)
        result.completion_tokens = 2
        result.tool_calls = 0
        result.tool_events = ()
        result.tool_result_chars = 0
        result.wall_time_s = protocol.MAX_AGENT_WALL_SECONDS
        result.terminal_reason = "quality_incomplete"
        result.budget_exhaustion = f"wall time limit {protocol.MAX_AGENT_WALL_SECONDS}s reached"
        result.quality_gate = {
            "schema": "mio.coding-quality-gate.v1",
            "enabled": True,
            "effort": "medium",
            "intent": "code_change_requested",
            "request_sha256": protocol.sha256_bytes(instruction.encode("utf-8")),
            "decision": "incomplete",
            "phase": "dirty",
            "activated": True,
            "satisfied": False,
            "mutation_epoch": 1,
            "changed_kinds": ["code"],
            "snapshot_complete": current.complete,
            "snapshot_method": current.method,
            "snapshot_error_codes": list(current.error_codes),
            "initial_revision_sha256": initial.revision_sha256,
            "current_revision_sha256": current.revision_sha256,
            "required": ["test_or_build"],
            "missing": ["test_or_build"],
            "validation_counts": {"test": 0, "build": 0, "static": 0, "diff": 0, "review": 0},
            "validation_attempts": 0,
            "successful_reads": 0,
        }
        return result

    monotonic = iter((0.0, protocol.MAX_AGENT_WALL_SECONDS + 0.25))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(monotonic))
    monkeypatch.setattr(agent, "_process_user_input", deadline_result)

    outcome = native(request)
    turn, quality, rounds, _tools = outcome.telemetry.document()

    assert outcome.status == "incomplete"
    assert outcome.wall_seconds == protocol.MAX_AGENT_WALL_SECONDS
    assert turn["wall_elapsed_ns"] == (protocol.MAX_AGENT_WALL_SECONDS * 1_000_000_000) + 250_000_000
    assert turn["terminal_reason"] == "quality_incomplete"
    assert turn["budget_exhaustion_kind"] == "wall_time"
    assert rounds[-1]["deadline_hit"] is True
    assert quality["satisfied"] is False
    checkpoint = protocol.ArmCheckpoint.for_entry(
        request.entry,
        schedule_sha256="a" * 64,
        status=outcome.status,
        model_patch="",
        mio_commit="b" * 40,
        model_identity=protocol.EXPECTED_MODEL_IDENTITY,
        runtime_digest="d" * 64,
        quality_gate_decision=outcome.quality_gate_decision,
        output_tokens=outcome.output_tokens,
        tool_calls=outcome.tool_calls,
        wall_seconds=outcome.wall_seconds,
    )
    runner._telemetry_sidecar_document(request.entry, checkpoint, "e" * 64, outcome.telemetry)


def test_native_gate_on_rejects_quality_report_without_snapshot_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent
    from mio.coding_quality import snapshot_workspaces

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
        require_raw_target_telemetry=True,
    )
    request = _raw_target_native_request(tmp_path)
    request = replace(request, entry=replace(request.entry, condition="gate_on"), quality_gate_enabled=True)
    result = _raw_target_result()
    report = _observing_quality_report(request.instruction, snapshot_workspaces((request.workspace,)))
    del report["snapshot_method"]
    result.quality_gate = report
    monkeypatch.setattr(agent, "_process_user_input", lambda *_args, **_kwargs: result)

    with pytest.raises(protocol.ProtocolError, match="quality report fields"):
        native(request)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("empty_rounds", "at least one model round"),
        ("noncontiguous_rounds", "zero-based and contiguous"),
        ("backend", "target_ar/baseline/no-drafter"),
        ("requested", "target_ar/baseline/no-drafter"),
        ("selected", "target_ar/baseline/no-drafter"),
        ("drafter_ref", "target_ar/baseline/no-drafter"),
        ("timing_source", "runtime_raw_ns"),
        ("phase_sum", "prefill plus decode"),
        ("physical_decode", "physical decode work"),
        ("tool_incomplete", "tool_telemetry_complete differs"),
        ("tool_count", "exactly one trace per tool call"),
        ("tool_sequence", "tool traces must be zero-based and contiguous"),
        ("tool_trace_incomplete", "incomplete file telemetry"),
        ("delivered_total", "delivered-token total"),
        ("prompt_alias", "prompt-token aliases disagree"),
        ("warm_alias", "warm-offset aliases disagree"),
        ("total_time", "exceeds total model-call time"),
        ("tool_name_vocab", "outside the sealed vocabulary"),
        ("tool_outcome_vocab", "outside the sealed vocabulary"),
        ("tool_target_digest", "lowercase SHA-256"),
        ("tool_round_order", "invalid or reordered"),
        ("prompt_throughput", "prompt throughput"),
        ("operation_mismatch", "name/operation/permission"),
        ("permission_mismatch", "name/operation/permission"),
        ("allowed_mismatch", "allowed flag and outcome"),
        ("audit_count", "audit_count"),
        ("timeout_flag", "terminable watchdog"),
        ("duration_bound", "effective timeout plus parent bound"),
        ("effect_unknown", "effect_unknown"),
        ("completed_last_round_tool", "tool-free final round"),
        ("complete_wall", "durations exceed complete arm wall"),
    ),
)
def test_native_raw_target_telemetry_rejects_incomplete_or_mismatched_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    from mio import agent

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
        require_raw_target_telemetry=True,
    )
    result = _raw_target_result()
    if case == "empty_rounds":
        result.rounds = ()
    elif case == "noncontiguous_rounds":
        result.rounds = (result.rounds[0], _raw_target_round(2, 3))
    elif case == "backend":
        result.rounds[0].generation_backend = "dflash"
    elif case == "requested":
        result.rounds[0].drafter_requested = "auto"
    elif case == "selected":
        result.rounds[0].drafter_selected = "dflash"
    elif case == "drafter_ref":
        result.rounds[0].drafter_ref = "local-draft"
    elif case == "timing_source":
        result.rounds[0].timing_source = "derived_legacy_us"
    elif case == "phase_sum":
        result.rounds[0].model_total_ns = 301
    elif case == "physical_decode":
        result.rounds[0].physical_decode_tokens = 1
    elif case == "tool_incomplete":
        result.tool_telemetry_complete = False
    elif case == "tool_count":
        result.tool_events = result.tool_events[:1]
    elif case == "tool_sequence":
        result.tool_events[1].sequence = 2
    elif case == "tool_trace_incomplete":
        result.tool_events[0].telemetry_complete = False
    elif case == "delivered_total":
        result.completion_tokens = 6
    elif case == "prompt_alias":
        result.rounds[0].prompt_tokens += 1
    elif case == "warm_alias":
        result.rounds[0].warm_offset_tokens = 1
    elif case == "total_time":
        result.rounds[0].total_time_s = 0.0
    elif case == "tool_name_vocab":
        result.tool_events[0].tool_name = "read:/private/secret"
    elif case == "tool_outcome_vocab":
        result.tool_events[0].outcome = "secret-output"
    elif case == "tool_target_digest":
        result.tool_events[0].target_sha256 = "not-a-digest"
    elif case == "tool_round_order":
        result.tool_events[0].round_index = 1
        result.tool_events[1].round_index = 0
    elif case == "prompt_throughput":
        result.rounds[0].prompt_tps = 1.0
    elif case == "operation_mismatch":
        result.tool_events[0].operation = "write"
    elif case == "permission_mismatch":
        result.tool_events[0].permission = "shell"
    elif case == "allowed_mismatch":
        result.tool_events[0].allowed = False
    elif case == "audit_count":
        result.tool_events[0].audit_count = 0
    elif case == "timeout_flag":
        result.tool_events[0].timeout_enforced = False
    elif case == "duration_bound":
        result.tool_events[0].duration_ns = 35_000_000_001
    elif case == "effect_unknown":
        result.tool_events[0].effect_unknown = True
    elif case == "completed_last_round_tool":
        result.tool_events[1].round_index = 1
    elif case == "complete_wall":
        result.rounds[0].total_time_s = 1.0
    monkeypatch.setattr(agent, "_process_user_input", lambda *_args, **_kwargs: result)

    with pytest.raises(protocol.ProtocolError, match=message):
        native(_raw_target_native_request(tmp_path))


def test_native_raw_target_telemetry_censors_unstructured_model_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent

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
        require_raw_target_telemetry=True,
    )
    monkeypatch.setattr(
        agent,
        "_process_user_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model failed")),
    )

    outcome = native(_raw_target_native_request(tmp_path))
    turn, quality, rounds, tools = outcome.telemetry.document()

    assert outcome.status == "model_error"
    assert outcome.output_tokens == 0
    assert outcome.tool_calls == 0
    assert turn["terminal_reason"] == "model_error"
    assert turn["trajectory_complete"] is False
    assert turn["counters_observed"] is False
    assert turn["tool_telemetry_complete"] is False
    assert quality["enabled"] is False
    assert rounds == ()
    assert tools == ()


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


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (protocol.ProtocolError("bad protocol"), "bad protocol"),
        (OSError("host filesystem failed"), "infrastructure failure"),
        (MemoryError("host memory failed"), "infrastructure failure"),
    ],
)
def test_native_infrastructure_and_protocol_exceptions_remain_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    match: str,
) -> None:
    from mio import agent

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
        require_raw_target_telemetry=True,
    )

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(agent, "_process_user_input", fail)

    with pytest.raises(protocol.ProtocolError, match=match):
        native(_raw_target_native_request(tmp_path))


def test_portable_model_exceptions_complete_pair_and_preserve_workspace_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mio import agent

    binding, _repository, model = _automatic_binding(tmp_path, monkeypatch, name="portable-error")
    source, base_commit = _source_repo(tmp_path)
    document, schedule = _schedule_document(_instances(base_commit))
    layout = runner.GenerationLayout.create(tmp_path / "generation", portable_artifacts=True)
    executor = _native_executor_for_model(model, require_raw_target_telemetry=True)

    def model_failure(_instruction, _engine, _manager, _config, state):
        workspace = state["tool_policy"].workspace_roots[0]
        (workspace / "module.py").write_text("VALUE = 99\n", encoding="utf-8")
        raise RuntimeError("model stream failed after mutation")

    monkeypatch.setattr(agent, "_process_user_input", model_failure)
    runner.run_generation_pairs(
        schedule_document=document,
        schedule=schedule,
        layout=layout,
        workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
        executor=executor,
        binding=binding,
        tier_config=executor.engine.tier_config,
        agent_module=_agent_module(),
        require_portable_artifacts=True,
    )

    assert runner.pending_pairs(schedule, layout, require_telemetry=True) == ()
    records = protocol.AttemptLedger(layout.ledger, protocol.schedule_digest(schedule)).read()
    assert [record["event"] for record in records] == ["started", "completed", "started", "completed"]
    store = protocol.CheckpointStore(layout.canonical)
    for entry in schedule:
        checkpoint = store.load(entry)
        assert checkpoint.status == "model_error"
        assert "VALUE = 99" in checkpoint.model_patch
        checkpoint_sha256 = protocol._immutable_file_sha256(store.path_for(entry))
        telemetry, _digest = runner._load_telemetry_sidecar(
            layout.telemetry,
            entry,
            checkpoint,
            checkpoint_sha256,
        )
        assert telemetry["turn"]["trajectory_complete"] is False
        assert telemetry["turn"]["counters_observed"] is False
        assert telemetry["round_count"] == 0
        assert telemetry["tool_trace_count"] == 0
        if entry.condition == "gate_on":
            assert telemetry["quality_gate"]["phase"] == "model_error"
            assert telemetry["quality_gate"]["satisfied"] is False
        else:
            assert telemetry["quality_gate"]["phase"] == "experiment_disabled"


def test_portable_artifacts_support_cross_process_audit_and_separate_current_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, schedule, binding, surface_sha256, receipt_sha256 = _sealed_portable_run(tmp_path, monkeypatch)

    assert (
        runner.verify_sealed_generation_artifacts(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=runner.GenerationLayout.open(layout.root),
            tool_surface_sha256=surface_sha256,
        )
        == receipt_sha256
    )
    receipt = json.loads(layout.receipt.read_text(encoding="utf-8"))
    audit = receipt["sealed_artifact_audit"]
    assert audit["portable"] is True
    assert audit["cross_process_sealed_artifact_verification_supported"] is True
    assert audit["current_environment_reattestation_is_separate"] is True
    assert audit["runtime_manifest"]["sha256"] == binding.runtime_digest
    assert audit["telemetry_manifest"]["arm_count"] == len(schedule)
    assert len(receipt["canonical_manifest"]) == len(schedule)
    for path in (layout.artifact_profile, layout.runtime_manifest, layout.receipt):
        metadata = path.stat()
        assert metadata.st_mode & 0o777 == 0o600
        assert metadata.st_nlink == 1
    for path in layout.telemetry.glob("*.json"):
        metadata = path.stat()
        assert metadata.st_mode & 0o777 == 0o600
        assert metadata.st_nlink == 1
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(sidecar, sort_keys=True)
        assert "problem_statement" not in serialized
        assert "assistant_text" not in serialized
        assert "tool_arguments" not in serialized
        assert "model_patch" not in serialized
        assert "/private/" not in serialized
        assert sidecar["round_count"] == 2
        assert sidecar["tool_trace_count"] == 2

    monkeypatch.setattr(runner, "_collect_runtime_document", lambda: _runtime_document("current-host-drift"))
    assert (
        runner.verify_sealed_generation_artifacts(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=runner.GenerationLayout.open(layout.root),
            tool_surface_sha256=surface_sha256,
        )
        == receipt_sha256
    )
    with pytest.raises(protocol.ProtocolError, match="runtime/dependency environment.*drifted"):
        runner.reattest_current_generation_environment(layout=layout, binding=binding)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("missing_sidecar", "missing"),
        ("symlink_sidecar", "aliases"),
        ("sidecar_mode", "private permissions"),
        ("hardlink_sidecar", "single-link"),
        ("runtime_tamper", "digest binding mismatch"),
        ("receipt_digest", "differs from the retained sealed artifacts"),
        ("sidecar_content_field", "fields or schema"),
        ("sidecar_status", "not derivable"),
        ("sidecar_quality_phase", "passing quality semantics"),
    ),
)
def test_portable_artifact_audit_rejects_tamper_missing_alias_and_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    layout, schedule, _binding_value, surface_sha256, _receipt_sha256 = _sealed_portable_run(tmp_path, monkeypatch)
    sidecar = sorted(layout.telemetry.glob("*.json"))[0]
    if case == "missing_sidecar":
        sidecar.unlink()
    elif case == "symlink_sidecar":
        outside = tmp_path / "outside-telemetry.json"
        outside.write_bytes(sidecar.read_bytes())
        outside.chmod(0o600)
        sidecar.unlink()
        sidecar.symlink_to(outside)
    elif case == "sidecar_mode":
        sidecar.chmod(0o644)
    elif case == "hardlink_sidecar":
        os.link(sidecar, tmp_path / "telemetry-alias.json")
    elif case == "runtime_tamper":
        runtime = json.loads(layout.runtime_manifest.read_text(encoding="utf-8"))
        runtime["python"]["marker"] = "tampered"
        layout.runtime_manifest.write_bytes(protocol.canonical_json_bytes(runtime))
    elif case == "receipt_digest":
        receipt = json.loads(layout.receipt.read_text(encoding="utf-8"))
        receipt["sealed_artifact_audit"]["runtime_manifest"]["sha256"] = "0" * 64
        layout.receipt.write_bytes(protocol.canonical_json_bytes(receipt))
    elif case == "sidecar_content_field":
        telemetry = json.loads(sidecar.read_text(encoding="utf-8"))
        telemetry["prompt"] = "must never be admitted"
        sidecar.write_bytes(protocol.canonical_json_bytes(telemetry))
    elif case == "sidecar_status":
        telemetry = json.loads(sidecar.read_text(encoding="utf-8"))
        telemetry["turn"]["status"] = "incomplete"
        sidecar.write_bytes(protocol.canonical_json_bytes(telemetry))
    elif case == "sidecar_quality_phase":
        sidecar = next(path for path in layout.telemetry.glob("*.json") if "gate_on" in path.name)
        telemetry = json.loads(sidecar.read_text(encoding="utf-8"))
        telemetry["quality_gate"]["phase"] = "dirty"
        sidecar.write_bytes(protocol.canonical_json_bytes(telemetry))
    else:  # pragma: no cover - exhaustive parametrization guard
        raise AssertionError(case)

    with pytest.raises(protocol.ProtocolError, match=message):
        runner.verify_sealed_generation_artifacts(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=layout,
            tool_surface_sha256=surface_sha256,
        )


def test_legacy_layout_is_explicitly_nonportable_for_original_artifact_audit(tmp_path: Path) -> None:
    layout = runner.GenerationLayout.create(tmp_path / "legacy-generation")
    schedule = protocol.make_balanced_schedule(
        ["owner__repository-1", "owner__repository-2"],
        require_full=False,
    )

    with pytest.raises(protocol.ProtocolError, match="legacy generation layout is non-portable"):
        runner.verify_sealed_generation_artifacts(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=layout,
            tool_surface_sha256="a" * 64,
        )


def test_legacy_layout_cannot_claim_current_environment_reattestation(tmp_path: Path) -> None:
    layout = runner.GenerationLayout.create(tmp_path / "legacy-generation")

    with pytest.raises(protocol.ProtocolError, match="cannot support current-environment reattestation"):
        runner.reattest_current_generation_environment(layout=layout, binding=_binding())
    with pytest.raises(protocol.ProtocolError, match="cannot support current-environment reattestation"):
        runner.verify_generation_receipt(
            receipt_path=layout.receipt,
            schedule=(),
            layout=layout,
            binding=_binding(),
            tool_surface_sha256="a" * 64,
        )


def test_portable_completed_ledger_binds_checkpoint_and_sidecar_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, schedule, _binding_value, _surface_sha256, _receipt_sha256 = _sealed_portable_run(
        tmp_path,
        monkeypatch,
    )
    ledger = protocol.AttemptLedger(layout.ledger, protocol.schedule_digest(schedule))
    completed = [record for record in ledger.read() if record["event"] == "completed"]
    canonical = protocol.CheckpointStore(layout.canonical)

    for record, pair in zip(
        completed, (schedule[index : index + 2] for index in range(0, len(schedule), 2)), strict=True
    ):
        for entry in pair:
            checkpoint_sha256 = protocol._immutable_file_sha256(canonical.path_for(entry))
            telemetry_sha256 = protocol._immutable_file_sha256(runner._telemetry_path(layout.telemetry, entry))
            expected = runner._pair_artifact_binding_sha256(checkpoint_sha256, telemetry_sha256)
            assert record["checkpoint_sha256s"][entry.condition] == expected
            assert expected != checkpoint_sha256

    first_pair = schedule[:2]
    record = completed[0]
    entry = first_pair[0]
    canonical_sidecar = runner._telemetry_path(layout.telemetry, entry)
    canonical_sidecar.unlink()
    attempt_store = protocol.pair_attempt_store(
        layout.attempts,
        entry.pair_index,
        int(record["attempt_index"]),
    )
    attempt_sidecar = runner._telemetry_path(attempt_store.root / "telemetry", entry)
    telemetry = json.loads(attempt_sidecar.read_text(encoding="utf-8"))
    telemetry["tools"][0]["target_sha256"] = "c" * 64
    attempt_sidecar.write_bytes(protocol.canonical_json_bytes(telemetry))

    with pytest.raises(protocol.ProtocolError, match="checkpoint/telemetry pair"):
        runner.pending_pairs(schedule, layout, repair_completed_promotions=True)
    assert not canonical_sidecar.exists()


def test_portable_resume_missing_runtime_rejects_without_mutating_any_retained_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, schedule, binding, _surface_sha256, _receipt_sha256 = _sealed_portable_run(tmp_path, monkeypatch)
    source = tmp_path / "source"
    base_commit = _git(source, "rev-parse", "HEAD")
    document, reconstructed_schedule = _schedule_document(_instances(base_commit))
    assert reconstructed_schedule == schedule
    executor = _native_executor_for_model(
        tmp_path / "portable-model",
        require_raw_target_telemetry=True,
    )
    layout.runtime_manifest.unlink()

    def retained_bytes() -> dict[str, tuple[bytes, int]]:
        return {
            str(path.relative_to(layout.root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in layout.root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }

    before = retained_bytes()
    with pytest.raises(protocol.ProtocolError, match="missing"):
        runner.run_generation_pairs(
            schedule_document=document,
            schedule=schedule,
            layout=layout,
            workspace_factory=runner.ExternalGitWorkspaceFactory(lambda _instance: source),
            executor=executor,
            binding=binding,
            tier_config=executor.engine.tier_config,
            agent_module=_agent_module(),
            require_portable_artifacts=True,
        )

    assert retained_bytes() == before
    assert not layout.runtime_manifest.exists()
