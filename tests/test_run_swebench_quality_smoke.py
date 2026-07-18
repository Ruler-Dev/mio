from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mio.config import MioConfig, TierConfig
from scripts import bench_swebench_quality as protocol
from scripts import run_swebench_quality_generation as generation
from scripts import run_swebench_quality_smoke as smoke


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _bare_source(tmp_path: Path) -> tuple[Path, str]:
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    _git("init", "--quiet", cwd=work)
    _git("config", "user.name", "Mio Test", cwd=work)
    _git("config", "user.email", "mio@example.invalid", cwd=work)
    (work / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git("add", "module.py", cwd=work)
    _git("commit", "--quiet", "-m", "base", cwd=work)
    commit = _git("rev-parse", "HEAD", cwd=work)
    source = tmp_path / "source.git"
    _git("clone", "--quiet", "--mirror", str(work), str(source))
    return source.resolve(strict=True), commit


def _instances(base_commit: str) -> tuple[protocol.PublicInstance, ...]:
    return tuple(
        protocol.PublicInstance(
            instance_id=f"owner__repository-{index}",
            repo="owner/repository",
            base_commit=base_commit,
            problem_statement=f"Change VALUE for case {index} without exposing this text.",
        )
        for index in (1, 2)
    )


def _private_schedule(tmp_path: Path, instances: tuple[protocol.PublicInstance, ...]) -> Path:
    path = tmp_path / "private-schedule.json"
    document = protocol.private_schedule_document(instances, evidence_run=False)
    path.write_bytes(protocol.canonical_json_bytes(document))
    path.chmod(0o600)
    return path.resolve(strict=True)


def _model_root(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir(mode=0o700)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    return model.resolve(strict=True)


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text("{}\n", encoding="utf-8")
    path.chmod(0o600)
    return path.resolve(strict=True)


def _config(model: Path) -> MioConfig:
    tier = TierConfig(
        name="large",
        target_model=str(model),
        draft_model="unused",
        context_window=99,
        max_output_tokens=99,
    )
    return MioConfig(tiers={"large": tier}, active_tiers=["small"], tandem=True)


def test_stable_config_requires_private_permissions(tmp_path: Path) -> None:
    path = _config_path(tmp_path)
    path.chmod(0o644)

    with pytest.raises(protocol.ProtocolError, match="0600"):
        smoke._load_stable_config(path, lambda _path: MioConfig.default())


class FakeBinding:
    model_identity = protocol.EXPECTED_MODEL_IDENTITY

    def __init__(self) -> None:
        self.validations: list[dict] = []

    def validate_for_run(self, **kwargs):
        self.validations.append(kwargs)
        return {"automatic": True}


class FakeManager:
    def __init__(self, config: MioConfig) -> None:
        self.config = config
        self.engine = SimpleNamespace(
            tier_config=config.tiers["large"],
            is_loaded=False,
            _target_model=None,
            _tokenizer=None,
            _draft_model=None,
            _dspark_runtime=None,
            _drafter_requested="target_ar",
            _drafter_selected="baseline",
            _drafter_ref=None,
            last_metrics=SimpleNamespace(
                generation_backend="baseline",
                drafter_requested="target_ar",
                drafter_selected="baseline",
                drafter_ref=None,
                fallback_ar=False,
                timing_source="runtime_raw_ns",
                completion_tokens=3,
                physical_decode_tokens=5,
                logical_prompt_tokens=11,
                physical_prefill_tokens=7,
                warm_offset=4,
                prefill_ns=100,
                decode_ns=200,
                model_total_ns=300,
            ),
        )
        self.unloaded = False

    def load_tier(self, tier_name: str) -> None:
        assert tier_name == "large"
        self.engine.is_loaded = True
        self.engine._target_model = object()
        self.engine._tokenizer = object()

    def get_engine(self, tier_name: str):
        assert tier_name == "large"
        return self.engine

    def loaded_tiers(self) -> list[str]:
        return ["large"] if self.engine.is_loaded else []

    def unload_all(self) -> None:
        self.unloaded = True
        self.engine.is_loaded = False


def _dependencies(
    *,
    config: MioConfig,
    binding: FakeBinding,
    managers: list[FakeManager],
    calls: list[str],
    fail_run: bool = False,
) -> smoke.SmokeDependencies:
    surface = "c" * 64
    factor = generation.factor_digest(surface)

    def manager_factory(loaded_config):
        assert loaded_config is config
        manager = FakeManager(loaded_config)
        managers.append(manager)
        return manager

    def run_pairs(**kwargs):
        calls.append("run")
        if fail_run:
            raise protocol.ProtocolError("immutable resume header mismatch")
        assert kwargs["tier_config"] is config.tiers["large"]
        assert kwargs["executor"].engine.tier_config is kwargs["tier_config"]
        instance = protocol.PublicInstance.from_mapping(kwargs["schedule_document"]["public_instances"][0])
        assert kwargs["workspace_factory"].source_for(instance).name == "source.git"
        return factor

    def seal_receipt(**_kwargs):
        calls.append("seal")
        return "d" * 64

    def verify_receipt(**_kwargs):
        calls.append("verify")
        return "d" * 64

    return smoke.SmokeDependencies(
        load_schedule=protocol.load_private_schedule,
        load_config=lambda _path: config,
        binding_factory=lambda **_kwargs: binding,
        manager_factory=manager_factory,
        executor_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        workspace_factory=generation.ExternalGitWorkspaceFactory,
        run_pairs=run_pairs,
        build_tool_surface=lambda: ({}, (), surface),
        seal_receipt=seal_receipt,
        verify_receipt=verify_receipt,
    )


def _options(
    *,
    tmp_path: Path,
    schedule: Path,
    source: Path,
    model: Path,
    config_path: Path,
    mode: str = "new",
    layout: Path | None = None,
) -> smoke.SmokeOptions:
    return smoke.SmokeOptions(
        schedule_path=schedule,
        layout_root=layout or (tmp_path / "generation"),
        layout_mode=mode,
        model_root=model,
        config_path=config_path,
        tier_name="large",
        repo_source_arguments=(f"owner/repository={source}",),
    )


def test_run_smoke_forces_exact_loaded_tier_seals_and_unloads(tmp_path: Path) -> None:
    source, commit = _bare_source(tmp_path)
    schedule = _private_schedule(tmp_path, _instances(commit))
    model = _model_root(tmp_path)
    config_path = _config_path(tmp_path)
    config = _config(model)
    binding = FakeBinding()
    managers: list[FakeManager] = []
    calls: list[str] = []
    deps = _dependencies(config=config, binding=binding, managers=managers, calls=calls)

    result = smoke.run_smoke(
        _options(
            tmp_path=tmp_path,
            schedule=schedule,
            source=source,
            model=model,
            config_path=config_path,
        ),
        dependencies=deps,
    )

    assert calls == ["run", "seal", "verify"]
    assert managers[0].unloaded is True
    assert binding.validations[0]["require_executor_binding"] is True
    tier = config.tiers["large"]
    assert tier.target_model == str(model)
    assert tier.drafter_backend == "target_ar"
    assert tier.draft_model == "disabled-for-target-ar-smoke"
    assert tier.context_window == generation.TARGET_CONTEXT_TOKENS
    assert tier.max_output_tokens == generation.TARGET_MAX_OUTPUT_TOKENS_PER_ROUND
    assert tier.tq_bits == tier.pq_bits == 16
    assert tier.bmp_paths == 1
    assert tier.ddtree_budget == 0
    assert tier.temperature == 0.0
    assert tier.top_p == 1.0
    assert tier.top_k == 0
    assert result.factor_sha256 == generation.factor_digest("c" * 64)
    assert result.receipt_sha256 == "d" * 64
    assert result.as_dict()["contains_issue_model_or_patch_text"] is False
    assert result.as_dict()["confirmatory_evidence_admissible"] is False


def test_resume_mismatch_fails_closed_before_seal_and_still_unloads(tmp_path: Path) -> None:
    source, commit = _bare_source(tmp_path)
    schedule = _private_schedule(tmp_path, _instances(commit))
    model = _model_root(tmp_path)
    config_path = _config_path(tmp_path)
    layout = generation.GenerationLayout.create(tmp_path / "generation")
    config = _config(model)
    managers: list[FakeManager] = []
    calls: list[str] = []
    deps = _dependencies(
        config=config,
        binding=FakeBinding(),
        managers=managers,
        calls=calls,
        fail_run=True,
    )

    with pytest.raises(protocol.ProtocolError, match="resume header mismatch"):
        smoke.run_smoke(
            _options(
                tmp_path=tmp_path,
                schedule=schedule,
                source=source,
                model=model,
                config_path=config_path,
                mode="resume",
                layout=layout.root,
            ),
            dependencies=deps,
        )

    assert calls == ["run"]
    assert managers[0].unloaded is True


def test_repo_mapping_requires_every_repo_once_and_base_commit_present(tmp_path: Path) -> None:
    source, commit = _bare_source(tmp_path)
    document = protocol.private_schedule_document(_instances(commit), evidence_run=False)

    with pytest.raises(protocol.ProtocolError, match="incomplete"):
        smoke.resolve_repo_sources((), document)
    with pytest.raises(protocol.ProtocolError, match="duplicate repository"):
        smoke.resolve_repo_sources(
            (f"owner/repository={source}", f"owner/repository={source}"),
            document,
        )

    wrong = dict(document)
    wrong["public_instances"] = [
        {**document["public_instances"][0], "base_commit": "a" * 40},
    ]
    with pytest.raises(protocol.ProtocolError, match="Git preflight"):
        smoke.resolve_repo_sources((f"owner/repository={source}",), wrong)


def test_repo_source_rejects_non_bare_and_path_alias(tmp_path: Path) -> None:
    source, commit = _bare_source(tmp_path)
    document = protocol.private_schedule_document(_instances(commit), evidence_run=False)
    ordinary = tmp_path / "ordinary"
    _git("clone", "--quiet", str(source), str(ordinary))
    with pytest.raises(protocol.ProtocolError, match="bare or mirror"):
        smoke.resolve_repo_sources((f"owner/repository={ordinary}",), document)

    alias = tmp_path / "source-alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(protocol.ProtocolError, match="symlink"):
        smoke.resolve_repo_sources((f"owner/repository={alias}",), document)


def test_layout_mode_is_explicit_and_never_overwrites(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(protocol.ProtocolError, match="already exists"):
        smoke._new_layout_root(existing)
    with pytest.raises(protocol.ProtocolError, match="does not exist"):
        smoke._prepare_layout_root(
            smoke.SmokeOptions(
                schedule_path=tmp_path / "schedule",
                layout_root=tmp_path / "missing",
                layout_mode="resume",
                model_root=tmp_path / "model",
                config_path=tmp_path / "config",
                tier_name="large",
                repo_source_arguments=("owner/repository=/tmp/source.git",),
            )
        )
    with pytest.raises(protocol.ProtocolError, match="explicitly new or resume"):
        smoke.SmokeOptions(
            schedule_path=tmp_path / "schedule",
            layout_root=tmp_path / "layout",
            layout_mode="automatic",
            model_root=tmp_path / "model",
            config_path=tmp_path / "config",
            tier_name="large",
            repo_source_arguments=("owner/repository=/tmp/source.git",),
        )


def test_confirmatory_schedule_is_rejected_before_binding_or_model_load(tmp_path: Path) -> None:
    source, commit = _bare_source(tmp_path)
    document = protocol.private_schedule_document(_instances(commit), evidence_run=False)
    document["evidence_class"] = "confirmatory"
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_bytes(protocol.canonical_json_bytes(document))
    schedule_path.chmod(0o600)
    model = _model_root(tmp_path)
    config_path = _config_path(tmp_path)
    called = False

    def binding_factory(**_kwargs):
        nonlocal called
        called = True
        return FakeBinding()

    config = _config(model)
    deps = _dependencies(
        config=config,
        binding=FakeBinding(),
        managers=[],
        calls=[],
    )
    deps = smoke.SmokeDependencies(
        **{
            **deps.__dict__,
            "load_schedule": lambda _path: (
                document,
                protocol.make_balanced_schedule(
                    tuple(instance.instance_id for instance in _instances(commit)),
                    require_full=False,
                ),
            ),
            "binding_factory": binding_factory,
        }
    )

    with pytest.raises(protocol.ProtocolError, match="non-evidence smoke schedules only"):
        smoke.run_smoke(
            _options(
                tmp_path=tmp_path,
                schedule=schedule_path,
                source=source,
                model=model,
                config_path=config_path,
            ),
            dependencies=deps,
        )
    assert called is False


def test_dirty_attestation_failure_creates_no_layout_or_manager(tmp_path: Path) -> None:
    source, commit = _bare_source(tmp_path)
    schedule = _private_schedule(tmp_path, _instances(commit))
    model = _model_root(tmp_path)
    config_path = _config_path(tmp_path)
    config = _config(model)
    managers: list[FakeManager] = []
    deps = _dependencies(
        config=config,
        binding=FakeBinding(),
        managers=managers,
        calls=[],
    )

    def reject_dirty(**_kwargs):
        raise protocol.ProtocolError("automatic attestation requires a clean Mio worktree")

    deps = smoke.SmokeDependencies(**{**deps.__dict__, "binding_factory": reject_dirty})
    layout = tmp_path / "generation"
    with pytest.raises(protocol.ProtocolError, match="clean Mio worktree"):
        smoke.run_smoke(
            _options(
                tmp_path=tmp_path,
                schedule=schedule,
                source=source,
                model=model,
                config_path=config_path,
                layout=layout,
            ),
            dependencies=deps,
        )
    assert managers == []
    assert not layout.exists()


def test_offline_environment_is_forced_and_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous")
    monkeypatch.delenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", raising=False)
    with smoke._offline_environment():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "1"
        assert os.environ["GIT_NO_LAZY_FETCH"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "previous"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "TRANSFORMERS_NO_ADVISORY_WARNINGS" not in os.environ


def test_raw_metrics_keep_decode_work_separate_from_delivered_budget() -> None:
    metrics = SimpleNamespace(
        generation_backend="baseline",
        drafter_requested="target_ar",
        drafter_selected="baseline",
        drafter_ref=None,
        fallback_ar=False,
        timing_source="runtime_raw_ns",
        completion_tokens=2,
        physical_decode_tokens=7,
        logical_prompt_tokens=9,
        physical_prefill_tokens=6,
        warm_offset=3,
        prefill_ns=10,
        decode_ns=20,
        model_total_ns=30,
    )
    smoke._validate_last_raw_metrics(SimpleNamespace(last_metrics=metrics))
    metrics.timing_source = "derived_legacy_us"
    with pytest.raises(protocol.ProtocolError, match="runtime_raw_ns"):
        smoke._validate_last_raw_metrics(SimpleNamespace(last_metrics=metrics))


def test_parser_requires_explicit_layout_mode_and_repeatable_sources() -> None:
    parser = smoke.build_parser()
    args = parser.parse_args(
        [
            "--schedule",
            "/tmp/schedule.json",
            "--layout",
            "/tmp/layout",
            "--new-layout",
            "--model",
            "/tmp/model",
            "--config",
            "/tmp/config.json",
            "--tier",
            "large",
            "--repo-source",
            "a/b=/tmp/a.git",
            "--repo-source",
            "c/d=/tmp/c.git",
        ]
    )
    assert args.layout_mode == "new"
    assert args.repo_source == ["a/b=/tmp/a.git", "c/d=/tmp/c.git"]


def test_result_json_has_no_private_problem_or_patch_text() -> None:
    result = smoke.SmokeResult(
        layout_mode="resume",
        schedule_sha256="a" * 64,
        factor_sha256="b" * 64,
        receipt_sha256="c" * 64,
        pair_count=1,
        arm_count=2,
    )
    rendered = json.dumps(result.as_dict(), sort_keys=True)
    assert "problem_statement" not in rendered
    assert "model_patch" not in rendered
    assert "/tmp/" not in rendered
    assert result.as_dict()["evaluator_invoked"] is False
    assert result.as_dict()["implicit_network_or_download_invoked"] is False
