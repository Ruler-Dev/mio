from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.bench_coding_quality import (
    GATE_OFF,
    GATE_ON,
    EvaluationRequest,
    GenerationObservation,
    HiddenEvaluation,
    materialize_public_fixture,
)
from scripts.run_coding_quality_benchmark import (
    CORPUS,
    ALL_PROTOCOL_SHA256,
    ALL_SUITE_SHA256,
    DRAFT_CONTENT_IDENTITY,
    DRAFT_REPOSITORY_LABEL,
    DEVELOPMENT_SUITE_SHA256,
    DEVELOPMENT_PROTOCOL_SHA256,
    FROZEN_ALPHA,
    FROZEN_BOOTSTRAP_SAMPLES,
    FROZEN_EVALUATOR_TIMEOUT_S,
    FROZEN_SOFTWARE_VERSIONS,
    FROZEN_SEED,
    GATE_PROFILE_SCHEMA,
    GATE_PROFILE_SHA256,
    RESULT_ENVELOPE_SCHEMA,
    SMOKE_PROTOCOL_SHA256,
    SMOKE_SUITE_SHA256,
    SOURCE_LOCK_SCHEMA,
    TARGET_CONTENT_IDENTITY,
    TARGET_REPOSITORY_LABEL,
    BenchmarkResultEnvelope,
    CleanSourceLock,
    CorpusHiddenEvaluator,
    LocalModelLock,
    RealMioGenerationRunner,
    RuntimeIdentity,
    _FROZEN_ENVIRONMENT_VARIABLES,
    _assert_source_free_artifact,
    _assert_frozen_environment,
    _load_native_executor,
    _parse_args,
    _SOURCE_LOCK_FILES,
    bind_frozen_local_models,
    capture_clean_source_lock,
    agent_turn_to_observation,
    build_agent_tool_surface,
    execute_corpus,
    fixture_suite_sha256,
    fixture_tree_sha256,
    protocol_suite_sha256,
    select_cases,
    sealed_suite_sha256,
    sealed_protocol_sha256,
    serialize_source_free_aggregate,
    serialize_source_free_result,
    verify_clean_source_lock,
    verify_frozen_local_models,
    validate_output_path,
)


def _runtime_identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        python_version="3.12.0",
        software_versions=FROZEN_SOFTWARE_VERSIONS,
        hardware_label="darwin-arm64-mac16-5-16cpu-51539607552b",
    )


def test_cli_requires_explicit_model_paths_and_defaults_to_small() -> None:
    with pytest.raises(SystemExit):
        _parse_args([])

    arguments = _parse_args(["--target-path", "target", "--draft-path", "draft"])

    assert arguments.target_path == Path("target")
    assert arguments.draft_path == Path("draft")
    assert arguments.tier == "small"


def test_executable_cli_bootstraps_repository_imports(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_coding_quality_benchmark.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--target-path" in completed.stdout
    assert "--draft-path" in completed.stdout


def test_local_model_lock_checks_exact_identities_before_and_after_run(tmp_path: Path) -> None:
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()
    revisions = {
        target.resolve(): TARGET_CONTENT_IDENTITY,
        drafter.resolve(): DRAFT_CONTENT_IDENTITY,
    }
    calls: list[Path] = []

    def fake_fingerprint(path: Path):
        resolved = path.resolve()
        calls.append(resolved)
        return SimpleNamespace(revision=revisions[resolved])

    locks = bind_frozen_local_models(target, drafter, fingerprint=fake_fingerprint)
    verify_frozen_local_models(locks, fingerprint=fake_fingerprint)

    assert calls == [target.resolve(), drafter.resolve(), target.resolve(), drafter.resolve()]
    assert [(lock.role, lock.repository_label, lock.content_identity) for lock in locks] == [
        ("target", TARGET_REPOSITORY_LABEL, TARGET_CONTENT_IDENTITY),
        ("drafter", DRAFT_REPOSITORY_LABEL, DRAFT_CONTENT_IDENTITY),
    ]

    revisions[drafter.resolve()] = "local-sha256-v1:" + "0" * 64
    with pytest.raises(RuntimeError, match="drafter local model changed"):
        verify_frozen_local_models(locks, fingerprint=fake_fingerprint)


def test_local_model_lock_rejects_wrong_preload_identity(tmp_path: Path) -> None:
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()

    with pytest.raises(RuntimeError, match="target local model"):
        bind_frozen_local_models(
            target,
            drafter,
            fingerprint=lambda _path: SimpleNamespace(revision="local-sha256-v1:" + "0" * 64),
        )


def test_native_loader_overrides_persisted_tier_and_requires_strict_dflash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MIO_DDTREE_BUDGET", raising=False)
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()
    tier = SimpleNamespace(
        target_model="wrong-target",
        draft_model="wrong-draft",
        draft_fallback_model="wrong-fallback",
        drafter_backend="auto",
        drafter_strict=False,
        tq_bits=4,
        pq_bits=4,
        bmp_paths=7,
        ddtree_budget=12,
        temperature=0.9,
        top_p=0.5,
        top_k=99,
        dspark_prefix_cache=True,
    )
    config = SimpleNamespace(tiers={"small": tier}, active_tiers=["wrong"])

    class FakeManager:
        def __init__(self, loaded_config):
            self.config = loaded_config
            self.loaded: list[str] = []
            self.unloaded = False
            self.engine = SimpleNamespace(
                drafter_status={
                    "selected": "dflash",
                    "fallback_used": False,
                    "strict": True,
                    "ref": str(drafter.resolve()),
                }
            )

        def load_tier(self, tier_name: str) -> None:
            self.loaded.append(tier_name)

        def get_engine(self, _tier_name: str):
            return self.engine

        def unload_all(self) -> None:
            self.unloaded = True

    executor, manager = _load_native_executor(
        tier="small",
        config_path=None,
        target_path=target,
        draft_path=drafter,
        config_loader=lambda _path: config,
        manager_factory=FakeManager,
    )

    assert executor.tier == "small"
    assert manager.loaded == ["small"]
    assert config.active_tiers == ["small"]
    assert tier.target_model == str(target.resolve())
    assert tier.draft_model == str(drafter.resolve())
    assert tier.draft_fallback_model is None
    assert tier.drafter_backend == "dflash"
    assert tier.drafter_strict is True
    assert (tier.tq_bits, tier.pq_bits, tier.bmp_paths, tier.ddtree_budget) == (16, 16, 1, 0)
    assert (tier.context_window, tier.max_output_tokens) == (8192, 2048)
    assert (tier.temperature, tier.top_p, tier.top_k, tier.dspark_prefix_cache) == (0.0, 1.0, 0, False)


def test_native_loader_rejects_backend_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIO_DDTREE_BUDGET", raising=False)
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()
    config = SimpleNamespace(tiers={"small": SimpleNamespace()}, active_tiers=[])

    class FakeManager:
        def __init__(self, _config):
            self.unloaded = False
            self.engine = SimpleNamespace(
                drafter_status={
                    "selected": "dflash",
                    "fallback_used": True,
                    "strict": True,
                    "ref": str(drafter.resolve()),
                }
            )

        def load_tier(self, _tier_name: str) -> None:
            return None

        def get_engine(self, _tier_name: str):
            return self.engine

        def unload_all(self) -> None:
            self.unloaded = True

    manager = FakeManager(config)
    with pytest.raises(RuntimeError, match="strict DFlash primary"):
        _load_native_executor(
            tier="small",
            config_path=None,
            target_path=target,
            draft_path=drafter,
            config_loader=lambda _path: config,
            manager_factory=lambda _config: manager,
        )
    assert manager.unloaded is True


@pytest.mark.parametrize("name", _FROZEN_ENVIRONMENT_VARIABLES)
def test_frozen_environment_rejects_every_native_dflash_override(name: str) -> None:
    for value in ("1", ""):
        with pytest.raises(RuntimeError, match=name):
            _assert_frozen_environment({name: value})

    _assert_frozen_environment({})


def test_output_path_cannot_overlap_or_follow_certified_inputs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    draft_root = tmp_path / "draft"
    result_root = tmp_path / "results"
    for root in (source_root, target_root, draft_root, result_root):
        root.mkdir()
    locks = (
        LocalModelLock(
            role="target",
            repository_label=TARGET_REPOSITORY_LABEL,
            content_identity=TARGET_CONTENT_IDENTITY,
            resolved_path=target_root,
        ),
        LocalModelLock(
            role="drafter",
            repository_label=DRAFT_REPOSITORY_LABEL,
            content_identity=DRAFT_CONTENT_IDENTITY,
            resolved_path=draft_root,
        ),
    )

    outside = result_root / "result.json"
    assert validate_output_path(
        outside,
        source_root=source_root,
        model_locks=locks,
    ) == outside.resolve()
    for protected in (source_root, target_root, draft_root):
        with pytest.raises(RuntimeError, match="outside source and model roots"):
            validate_output_path(
                protected / "result.json",
                source_root=source_root,
                model_locks=locks,
            )

    destination = result_root / "destination.json"
    destination.write_text("existing", encoding="utf-8")
    symlink = result_root / "result-link.json"
    symlink.symlink_to(destination)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        validate_output_path(
            symlink,
            source_root=source_root,
            model_locks=locks,
        )


def test_clean_source_lock_rejects_dirty_tree_and_post_run_drift(tmp_path: Path) -> None:
    source_files = ("a.py", "nested/b.py")
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "nested/b.py").write_text("b = 2\n", encoding="utf-8")
    dirty = False

    def fake_git_probe(root: Path, arguments: tuple[str, ...]) -> str:
        assert root == tmp_path.resolve()
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(tmp_path.resolve()) + "\n"
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M a.py\n" if dirty else ""
        if arguments[:3] == ("ls-files", "-z", "--"):
            return "\x00".join(source_files) + "\x00"
        raise AssertionError(arguments)

    source_lock = capture_clean_source_lock(
        tmp_path,
        git_probe=fake_git_probe,
        source_files=source_files,
    )
    assert source_lock.git_revision == "a" * 40
    assert source_lock.source_file_count == 2

    (tmp_path / "a.py").write_text("a = 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source drifted"):
        verify_clean_source_lock(
            source_lock,
            git_probe=fake_git_probe,
            source_files=source_files,
        )

    dirty = True
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        capture_clean_source_lock(
            tmp_path,
            git_probe=fake_git_probe,
            source_files=source_files,
        )


def test_corpus_has_frozen_smoke_and_development_splits() -> None:
    smoke = select_cases("smoke")
    development = select_cases("development")

    assert len(CORPUS) == 12
    assert [case.fixture.fixture_id for case in smoke] == ["s01", "s02", "s03", "s04"]
    assert [case.fixture.fixture_id for case in development] == [
        "d01",
        "d02",
        "d03",
        "d04",
        "d05",
        "d06",
        "d07",
        "d08",
    ]
    assert fixture_suite_sha256(tuple(case.fixture for case in smoke)) == SMOKE_SUITE_SHA256
    assert fixture_suite_sha256(tuple(case.fixture for case in development)) == DEVELOPMENT_SUITE_SHA256
    assert fixture_suite_sha256(tuple(case.fixture for case in CORPUS)) == ALL_SUITE_SHA256
    assert sealed_suite_sha256(smoke) == SMOKE_SUITE_SHA256
    assert sealed_protocol_sha256(smoke) == SMOKE_PROTOCOL_SHA256
    assert sealed_protocol_sha256(development) == DEVELOPMENT_PROTOCOL_SHA256
    assert sealed_protocol_sha256(CORPUS) == ALL_PROTOCOL_SHA256


def test_hidden_oracles_are_not_materialized_as_public_files() -> None:
    for case in CORPUS:
        public_text = "\n".join(item.content for item in case.fixture.public_files)
        assert case.oracle.hidden_checks not in public_text
        assert case.oracle.public_regression not in public_text
        assert all("hidden" not in item.relative_name for item in case.fixture.public_files)
        assert len(case.fixture.public_files) == 2
        for public_file in case.fixture.public_files:
            compile(public_file.content, public_file.relative_name, "exec")


def test_explicit_corpus_seal_fails_closed_before_execution() -> None:
    smoke = select_cases("smoke")
    changed_fixture = replace(smoke[0].fixture, instruction=smoke[0].fixture.instruction + " changed")
    changed = (replace(smoke[0], fixture=changed_fixture), *smoke[1:])

    with pytest.raises(RuntimeError, match="explicit suite seal"):
        sealed_suite_sha256(changed)


def test_private_protocol_seal_covers_oracle_scope_timeout_and_analysis() -> None:
    smoke = select_cases("smoke")
    baseline = protocol_suite_sha256(smoke)
    changed_oracle = replace(
        smoke[0],
        oracle=replace(
            smoke[0].oracle,
            hidden_checks=smoke[0].oracle.hidden_checks + "\nassert True\n",
        ),
    )
    changed_scope = replace(
        smoke[0],
        editable_names=(smoke[0].fixture.public_files[1].relative_name,),
    )

    assert baseline == SMOKE_PROTOCOL_SHA256
    assert fixture_suite_sha256(tuple(case.fixture for case in smoke)) == fixture_suite_sha256(
        tuple(case.fixture for case in (changed_oracle, *smoke[1:]))
    )
    assert protocol_suite_sha256((changed_oracle, *smoke[1:])) != baseline
    assert protocol_suite_sha256((changed_scope, *smoke[1:])) != baseline
    assert protocol_suite_sha256(
        smoke,
        evaluator_timeout_s=FROZEN_EVALUATOR_TIMEOUT_S + 1.0,
    ) != baseline
    assert protocol_suite_sha256(smoke, seed=FROZEN_SEED + 1) != baseline
    assert protocol_suite_sha256(
        smoke,
        bootstrap_samples=FROZEN_BOOTSTRAP_SAMPLES + 1,
    ) != baseline
    assert protocol_suite_sha256(smoke, alpha=FROZEN_ALPHA / 2) != baseline
    with pytest.raises(RuntimeError, match="private protocol"):
        sealed_protocol_sha256((changed_oracle, *smoke[1:]))
    with pytest.raises(ValueError, match="execution seed is frozen"):
        execute_corpus(
            cases=smoke,
            runner=lambda _request: GenerationObservation(completed=True),
            hidden_evaluator=lambda _request: HiddenEvaluation(
                passed=False,
                regression_free=True,
            ),
            work_root=Path("unused"),
            seed=FROZEN_SEED + 1,
        )


def test_tool_surfaces_exclude_mcp_and_skills_and_only_gate_on_gets_validate() -> None:
    fake_agent = SimpleNamespace(
        AGENT_TOOLS={
            name: {"fn": object()}
            for name in [
                "bash",
                "read",
                "write",
                "edit",
                "validate",
                "list_mio_skills",
                "read_mio_skill",
                "list_mcp_tools",
                "call_mcp_tool",
            ]
        },
        AGENT_TOOLS_SPEC=[
            {"type": "function", "function": {"name": name}}
            for name in [
                "bash",
                "read",
                "write",
                "edit",
                "validate",
                "list_mio_skills",
                "read_mio_skill",
                "list_mcp_tools",
                "call_mcp_tool",
            ]
        ],
    )

    off_registry, off_specs = build_agent_tool_surface(GATE_OFF, fake_agent)
    on_registry, on_specs = build_agent_tool_surface(GATE_ON, fake_agent)

    assert tuple(off_registry) == ("bash", "read", "write", "edit")
    assert tuple(on_registry) == ("bash", "read", "write", "edit", "validate")
    assert [item["function"]["name"] for item in off_specs] == list(off_registry)
    assert [item["function"]["name"] for item in on_specs] == list(on_registry)


def test_agent_turn_adapter_uses_content_free_metrics_and_fails_closed_without_gate_record() -> None:
    rounds = (
        SimpleNamespace(completion_tokens=7, total_time_s=0.4),
        SimpleNamespace(completion_tokens=5, total_time_s=0.3),
    )
    events = (
        SimpleNamespace(operation="write", allowed=True, outcome="ok"),
        SimpleNamespace(operation="validate", allowed=True, outcome="ok"),
    )
    result = SimpleNamespace(
        terminal_reason="model_final",
        rounds=rounds,
        tool_events=events,
        tool_calls=2,
        wall_time_s=0.9,
        quality_gate={"decision": "satisfied"},
        assistant_text="PRIVATE MODEL OUTPUT",
    )

    observation = agent_turn_to_observation(result, GATE_ON)

    assert observation.completed is True
    assert observation.mutation_count == 1
    assert observation.tool_calls == 2
    assert observation.output_tokens == 12
    assert observation.validation_attempted is True
    assert observation.validation_succeeded is True
    assert observation.model_seconds == pytest.approx(0.7)
    assert observation.wall_seconds == 0.9
    result.quality_gate = None
    assert agent_turn_to_observation(result, GATE_ON).completed is False
    assert agent_turn_to_observation(result, GATE_OFF).completed is True


def test_real_runner_resets_identical_fixture_bytes_and_disables_network(tmp_path: Path) -> None:
    case = select_cases("smoke")[0]
    calls = []

    class FakeExecutor:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                terminal_reason="model_final",
                rounds=(),
                tool_events=(),
                tool_calls=0,
                wall_time_s=0.1,
                quality_gate={"decision": "not_applicable"},
            )

    runner = RealMioGenerationRunner(
        executor=FakeExecutor(),
        fixtures=(case.fixture,),
        effort="medium",
        agent_module=SimpleNamespace(
            AGENT_TOOLS={name: {"fn": object()} for name in ["bash", "read", "write", "edit", "validate"]},
            AGENT_TOOLS_SPEC=[
                {"type": "function", "function": {"name": name}}
                for name in ["bash", "read", "write", "edit", "validate"]
            ],
        ),
    )
    for condition in (GATE_OFF, GATE_ON):
        workspace = tmp_path / condition
        workspace.mkdir()
        for public_file in case.fixture.public_files:
            (workspace / public_file.relative_name).write_text(public_file.content)
        request = SimpleNamespace(
            fixture_id=case.fixture.fixture_id,
            instruction=case.fixture.instruction,
            condition=condition,
            workspace=workspace,
            schedule_index=len(calls),
        )
        observation = runner(request)
        assert observation.completed is True

    assert [call["quality_gate_enabled"] for call in calls] == [False, True]
    assert [tuple(call["tool_registry"]) for call in calls] == [
        ("bash", "read", "write", "edit"),
        ("bash", "read", "write", "edit", "validate"),
    ]
    assert all("network" not in {permission.value for permission in call["tool_policy"].permissions} for call in calls)

    # A byte change before generation is an integrity error, not a task result.
    changed = tmp_path / "changed"
    changed.mkdir()
    for public_file in case.fixture.public_files:
        (changed / public_file.relative_name).write_text(public_file.content)
    (changed / case.fixture.public_files[0].relative_name).write_text("tampered")
    with pytest.raises(RuntimeError, match="frozen initial fixture bytes"):
        runner(
            SimpleNamespace(
                fixture_id=case.fixture.fixture_id,
                instruction=case.fixture.instruction,
                condition=GATE_OFF,
                workspace=changed,
                schedule_index=2,
            )
        )


def test_execute_corpus_uses_two_phases_and_serializes_only_aggregate(tmp_path: Path) -> None:
    cases = select_cases("smoke")
    events: list[str] = []
    initial_digests = {case.fixture.fixture_id: fixture_tree_sha256(case.fixture) for case in cases}

    def fake_runner(request):
        events.append("generate")
        assert (
            fixture_tree_sha256(next(case.fixture for case in cases if case.fixture.fixture_id == request.fixture_id))
            == initial_digests[request.fixture_id]
        )
        return GenerationObservation(completed=True)

    def fake_hidden(_request):
        events.append("evaluate")
        return HiddenEvaluation(passed=False, regression_free=True)

    execution = execute_corpus(
        cases=cases,
        runner=fake_runner,
        hidden_evaluator=fake_hidden,
        work_root=tmp_path / "runs",
    )
    artifact = serialize_source_free_aggregate(execution.aggregate)
    parsed = json.loads(artifact)

    assert events == ["generate"] * 8 + ["evaluate"] * 8
    assert parsed["protocol"]["pair_count"] == 4
    assert parsed["claim_gate"]["status"] == "no_claim_insufficient_pairs"
    for case in cases:
        assert case.fixture.fixture_id not in artifact
        assert case.fixture.instruction not in artifact
        assert case.oracle.hidden_checks not in artifact
    assert str(tmp_path) not in artifact


def test_result_envelope_has_fixed_provenance_and_no_paths_or_hidden_labels(tmp_path: Path) -> None:
    cases = select_cases("smoke")
    execution = execute_corpus(
        cases=cases,
        runner=lambda _request: GenerationObservation(completed=True),
        hidden_evaluator=lambda _request: HiddenEvaluation(passed=False, regression_free=True),
        work_root=tmp_path / "runs",
    )
    model_locks = (
        LocalModelLock(
            role="target",
            repository_label=TARGET_REPOSITORY_LABEL,
            content_identity=TARGET_CONTENT_IDENTITY,
            resolved_path=tmp_path / "private-target-path",
        ),
        LocalModelLock(
            role="drafter",
            repository_label=DRAFT_REPOSITORY_LABEL,
            content_identity=DRAFT_CONTENT_IDENTITY,
            resolved_path=tmp_path / "private-draft-path",
        ),
    )
    source_lock = CleanSourceLock(
        repo_root=tmp_path,
        git_revision="a" * 40,
        source_sha256="b" * 64,
        source_file_count=len(_SOURCE_LOCK_FILES),
    )

    artifact = serialize_source_free_result(
        BenchmarkResultEnvelope(
            source_lock=source_lock,
            model_locks=model_locks,
            runtime_identity=_runtime_identity(),
            aggregate=execution.aggregate,
            split="smoke",
            tier="small",
            effort="medium",
            protocol_sha256=SMOKE_PROTOCOL_SHA256,
        )
    )
    parsed = json.loads(artifact)

    assert set(parsed) == {
        "aggregate",
        "hidden_labels_serialized",
        "hardware",
        "implementation",
        "models",
        "protocol",
        "runtime",
        "schema_version",
        "software",
    }
    assert parsed["schema_version"] == RESULT_ENVELOPE_SCHEMA
    assert parsed["implementation"] == {
        "git_clean": True,
        "git_revision": "a" * 40,
        "post_run_source_stable": True,
        "source_file_count": len(_SOURCE_LOCK_FILES),
        "source_lock_schema": SOURCE_LOCK_SCHEMA,
        "source_sha256": "b" * 64,
    }
    assert parsed["models"]["target"]["content_identity"] == TARGET_CONTENT_IDENTITY
    assert parsed["models"]["drafter"]["content_identity"] == DRAFT_CONTENT_IDENTITY
    assert parsed["models"]["post_run_identities_stable"] is True
    assert parsed["protocol"] == {
        "gate_profile_schema": GATE_PROFILE_SCHEMA,
        "gate_profile_sha256": GATE_PROFILE_SHA256,
        "protocol_sha256": SMOKE_PROTOCOL_SHA256,
    }
    assert parsed["software"] == {
        "python_version": "3.12.0",
        "packages": dict(FROZEN_SOFTWARE_VERSIONS),
    }
    assert parsed["hardware"] == {
        "label": "darwin-arm64-mac16-5-16cpu-51539607552b"
    }
    assert parsed["runtime"] == {
        "bmp_paths": 1,
        "cold_arm_state": True,
        "context_window": 8192,
        "ddtree_budget": 0,
        "drafter_backend": "dflash",
        "dflash_draft_sink": 64,
        "dflash_draft_window": 1024,
        "dflash_exact_commit_oracle": False,
        "dflash_exact_components": "gdn,attention,mlp,head",
        "dflash_max_context": 131072,
        "dflash_qmv_staging": False,
        "dflash_qmv_vectors": "auto",
        "dflash_quantize_draft": False,
        "dflash_verify_len_override": False,
        "effort": "medium",
        "environment_overrides": False,
        "max_output_tokens": 2048,
        "network_enabled": False,
        "pq_bits": 16,
        "prefill_chunk": 2048,
        "split": "smoke",
        "temperature": 0.0,
        "tier": "small",
        "top_k": 0,
        "top_p": 1.0,
        "tq_bits": 16,
    }
    assert parsed["hidden_labels_serialized"] is False
    assert str(tmp_path) not in artifact
    assert "private-target-path" not in artifact
    assert "private-draft-path" not in artifact
    for case in cases:
        assert case.fixture.fixture_id not in artifact
        assert case.fixture.instruction not in artifact
        assert case.oracle.hidden_checks not in artifact


def test_result_envelope_rejects_cross_split_and_runtime_provenance(tmp_path: Path) -> None:
    cases = select_cases("smoke")
    execution = execute_corpus(
        cases=cases,
        runner=lambda _request: GenerationObservation(completed=True),
        hidden_evaluator=lambda _request: HiddenEvaluation(passed=False, regression_free=True),
        work_root=tmp_path / "runs",
    )
    model_locks = (
        LocalModelLock(
            role="target",
            repository_label=TARGET_REPOSITORY_LABEL,
            content_identity=TARGET_CONTENT_IDENTITY,
            resolved_path=tmp_path / "target",
        ),
        LocalModelLock(
            role="drafter",
            repository_label=DRAFT_REPOSITORY_LABEL,
            content_identity=DRAFT_CONTENT_IDENTITY,
            resolved_path=tmp_path / "draft",
        ),
    )
    source_lock = CleanSourceLock(
        repo_root=tmp_path,
        git_revision="a" * 40,
        source_sha256="b" * 64,
        source_file_count=len(_SOURCE_LOCK_FILES),
    )
    common = {
        "source_lock": source_lock,
        "model_locks": model_locks,
        "runtime_identity": _runtime_identity(),
        "aggregate": execution.aggregate,
        "tier": "small",
        "effort": "medium",
    }

    with pytest.raises(ValueError, match="frozen split protocol"):
        BenchmarkResultEnvelope(
            **common,
            split="development",
            protocol_sha256=DEVELOPMENT_PROTOCOL_SHA256,
        )
    with pytest.raises(ValueError, match="frozen split protocol"):
        BenchmarkResultEnvelope(
            **common,
            split="smoke",
            protocol_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="frozen software lock"):
        RuntimeIdentity(
            python_version="3.12.0",
            software_versions=(("mlx", "/private/path"),),
            hardware_label="darwin-arm64",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"records": []},
        {"safe": {"fixture_id": "private"}},
        {"target_path": "relative"},
        {"hidden_labels_serialized": True},
        {"safe": "/private/absolute/path"},
    ],
)
def test_public_artifact_boundary_rejects_sensitive_keys_and_paths(payload: object) -> None:
    with pytest.raises(ValueError):
        _assert_source_free_artifact(payload)


def test_hidden_evaluator_requires_scoped_nonempty_edit_and_immutable_public_test(tmp_path: Path) -> None:
    case = select_cases("smoke")[0]
    evaluator = CorpusHiddenEvaluator((case,))
    evaluator._run_oracle = lambda _workspace, _source: True  # type: ignore[method-assign]

    def workspace(name: str) -> Path:
        root = tmp_path / name
        root.mkdir()
        for public_file in case.fixture.public_files:
            (root / public_file.relative_name).write_text(public_file.content)
        return root

    def evaluate(root: Path) -> HiddenEvaluation:
        return evaluator(
            SimpleNamespace(
                fixture_id=case.fixture.fixture_id,
                condition=GATE_ON,
                workspace=root,
                schedule_index=0,
            )
        )

    unchanged = workspace("unchanged")
    assert evaluate(unchanged).passed is False

    valid = workspace("valid")
    (valid / case.editable_names[0]).write_text("def normalize_whitespace(text):\n    return ' '.join(text.split())\n")
    assert evaluate(valid).passed is True

    changed_test = workspace("changed-test")
    (changed_test / case.editable_names[0]).write_text("def normalize_whitespace(text):\n    return text\n")
    (changed_test / "test_public_text_utils.py").write_text("# weakened")
    assert evaluate(changed_test).passed is False

    extra = workspace("extra")
    (extra / case.editable_names[0]).write_text("def normalize_whitespace(text):\n    return text\n")
    (extra / "notes.txt").write_text("unexpected")
    assert evaluate(extra).passed is False


@pytest.mark.skipif(platform.system() != "Darwin", reason="native evaluator uses the macOS process sandbox")
def test_hidden_evaluator_runs_oracles_in_read_only_sandbox(tmp_path: Path) -> None:
    case = select_cases("smoke")[0]
    workspace = materialize_public_fixture(case.fixture, tmp_path / "workspace").workspace
    (workspace / case.editable_names[0]).write_text(
        "def normalize_whitespace(text):\n    return ' '.join(text.split())\n",
        encoding="utf-8",
    )

    outcome = CorpusHiddenEvaluator((case,))(
        EvaluationRequest(
            fixture_id=case.fixture.fixture_id,
            condition=GATE_ON,
            workspace=workspace,
            schedule_index=0,
        )
    )

    assert outcome == HiddenEvaluation(passed=True, regression_free=True)
    assert not (workspace / "__pycache__").exists()
