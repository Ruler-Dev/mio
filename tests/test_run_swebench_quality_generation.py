from __future__ import annotations

import json
import os
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


def _native_executor_for_model(model: Path, *, tier_name: str = "large") -> runner.NativeMioArmExecutor:
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
        runner.verify_generation_receipt(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=layout,
            binding=_binding(),
            tool_surface_sha256=surface_digest,
        )
    receipt_alias.unlink()

    header_alias = tmp_path / "run-header-alias.json"
    os.link(layout.run_header, header_alias)
    with pytest.raises(protocol.ProtocolError, match="single-link"):
        runner.verify_generation_receipt(
            receipt_path=layout.receipt,
            schedule=schedule,
            layout=layout,
            binding=_binding(),
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
    values = {
        "round_index": index,
        "generation_backend": "baseline",
        "fallback_ar": False,
        "drafter_requested": "target_ar",
        "drafter_selected": "baseline",
        "drafter_ref": None,
        "timing_source": "runtime_raw_ns",
        "prefill_ns": 100,
        "decode_ns": 200,
        "model_total_ns": 300,
        "completion_tokens": completion_tokens,
        "physical_decode_tokens": completion_tokens + 1,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _raw_target_result():
    return SimpleNamespace(
        terminal_reason="model_final",
        rounds=(_raw_target_round(0, 2), _raw_target_round(1, 3)),
        completion_tokens=5,
        quality_gate=None,
        tool_calls=2,
        tool_events=(
            SimpleNamespace(sequence=0, telemetry_complete=True),
            SimpleNamespace(sequence=1, telemetry_complete=True),
        ),
        tool_telemetry_complete=True,
        wall_time_s=0.5,
    )


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
        ("tool_incomplete", "tool telemetry is incomplete"),
        ("tool_count", "exactly one trace per tool call"),
        ("tool_sequence", "tool traces must be zero-based and contiguous"),
        ("tool_trace_incomplete", "tool trace is incomplete"),
        ("delivered_total", "delivered-token total"),
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
    monkeypatch.setattr(agent, "_process_user_input", lambda *_args, **_kwargs: result)

    with pytest.raises(protocol.ProtocolError, match=message):
        native(_raw_target_native_request(tmp_path))


def test_native_raw_target_telemetry_rejects_unstructured_model_error(
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

    with pytest.raises(protocol.ProtocolError, match="without a structured result"):
        native(_raw_target_native_request(tmp_path))


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
