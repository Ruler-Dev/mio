from __future__ import annotations

from argparse import Namespace
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import bench_swebench_quality as benchmark


def _instance_id(index: int) -> str:
    return f"owner__repository-{index + 1}"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Mio Test")
    _git(repo, "config", "user.email", "mio@example.invalid")
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "module.py")
    _git(repo, "commit", "--quiet", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _checkpoint(entry: benchmark.ScheduleEntry, schedule_sha256: str, **changes):
    values = {
        "schedule_sha256": schedule_sha256,
        "status": "completed",
        "model_patch": "",
        "mio_commit": "b" * 40,
        "model_identity": benchmark.EXPECTED_MODEL_IDENTITY,
        "runtime_digest": "d" * 64,
        "quality_gate_decision": "satisfied" if entry.condition == "gate_on" else "not_applicable",
        "output_tokens": 17,
        "tool_calls": 3,
        "wall_seconds": 1.25,
    }
    values.update(changes)
    return benchmark.ArmCheckpoint.for_entry(entry, **values)


def _official_report(ids: list[str], resolved: set[str], **changes) -> dict:
    unresolved = sorted(set(ids) - resolved)
    report = {
        "schema_version": 2,
        "total_instances": len(ids),
        "submitted_instances": len(ids),
        "completed_instances": len(ids),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": 0,
        "error_instances": 0,
        "completed_ids": sorted(ids),
        "incomplete_ids": [],
        "empty_patch_ids": [],
        "submitted_ids": sorted(ids),
        "resolved_ids": sorted(resolved),
        "unresolved_ids": unresolved,
        "error_ids": [],
    }
    report.update(changes)
    return report


def _binding() -> dict[str, str]:
    return {
        "mio_commit": "b" * 40,
        "model_identity": benchmark.EXPECTED_MODEL_IDENTITY,
        "runtime_digest": "d" * 64,
    }


def test_preregistration_freezes_full_verified_and_official_harness() -> None:
    protocol = json.loads(benchmark.PREREGISTRATION_PATH.read_text(encoding="utf-8"))

    assert protocol["confirmatory_scope"]["required_unique_instances"] == 500
    assert protocol["confirmatory_scope"]["required_generation_arms"] == 1000
    assert protocol["confirmatory_scope"]["smoke_runs_are_evidence"] is False
    assert protocol["official_harness"]["version"] == benchmark.HARNESS_VERSION
    assert protocol["official_harness"]["git_commit"] == benchmark.HARNESS_COMMIT
    assert protocol["dataset_artifacts"]["parquet_sha256"] == benchmark.DATASET_PARQUET_SHA256
    assert protocol["dataset_artifacts"]["full_snapshot_sha256"] == benchmark.FULL_SNAPSHOT_SHA256
    assert protocol["dataset_artifacts"]["public_snapshot_sha256"] == benchmark.PUBLIC_SNAPSHOT_SHA256
    assert protocol["model"]["content_identity"] == benchmark.EXPECTED_MODEL_IDENTITY
    assert protocol["model"]["role"] == "target_only_autoregressive_control"
    assert protocol["model"]["dflash"] is False
    assert protocol["model"]["dspark"] is False


def test_public_manifest_rejects_gold_and_requires_full_for_evidence(tmp_path: Path) -> None:
    row = {
        "instance_id": _instance_id(0),
        "repo": "owner/repository",
        "base_commit": "a" * 40,
        "problem_statement": "Fix the public behavior.",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([row]), encoding="utf-8")

    with pytest.raises(benchmark.ProtocolError, match="official public"):
        benchmark.load_public_manifest(path)

    assert benchmark.load_public_manifest(path, expected_count=1, evidence_run=False)[0].repo == "owner/repository"
    row["test_patch"] = "SECRET GOLD TEST"
    path.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(benchmark.ProtocolError, match="forbidden evaluator fields"):
        benchmark.load_public_manifest(path, expected_count=1, evidence_run=False)


def test_evidence_manifest_requires_exact_preregistered_public_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "instance_id": _instance_id(0),
        "repo": "owner/repository",
        "base_commit": "a" * 40,
        "problem_statement": "Fix the public behavior.",
    }
    path = tmp_path / "manifest.jsonl"
    path.write_bytes(benchmark.canonical_json_bytes(row))
    path.chmod(0o600)
    monkeypatch.setattr(benchmark, "PUBLIC_SNAPSHOT_SHA256", benchmark.sha256_file(path))
    monkeypatch.setattr(benchmark, "EXPECTED_INSTANCES", 1)

    assert benchmark.load_public_manifest(path, expected_count=1)[0].instance_id == row["instance_id"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(benchmark.ProtocolError, match="SHA-256 mismatch"):
        benchmark.load_public_manifest(path, expected_count=1)


def test_prepare_verifies_parquet_and_emits_exact_canonical_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    rows = []
    for index in (1, 0):
        values = {key: "" for key in benchmark.FULL_INSTANCE_KEYS}
        values.update(
            {
                "repo": "owner/repository",
                "instance_id": _instance_id(index),
                "base_commit": f"{index + 1:040x}",
                "problem_statement": f"Fix behavior {index}.",
            }
        )
        rows.append(values)
    source = tmp_path / "verified.parquet"
    parquet.write_table(pyarrow.Table.from_pylist(rows), source)
    source.chmod(0o600)
    ordered = sorted(rows, key=lambda row: row["instance_id"])
    full_payload = benchmark.canonical_jsonl_bytes(ordered)
    public_payload = benchmark.canonical_jsonl_bytes(
        [{key: row[key] for key in ("instance_id", "repo", "base_commit", "problem_statement")} for row in ordered]
    )
    monkeypatch.setattr(benchmark, "EXPECTED_INSTANCES", 2)
    monkeypatch.setattr(benchmark, "DATASET_PARQUET_SHA256", benchmark.sha256_file(source))
    monkeypatch.setattr(benchmark, "FULL_SNAPSHOT_SHA256", benchmark.sha256_bytes(full_payload))
    monkeypatch.setattr(benchmark, "PUBLIC_SNAPSHOT_SHA256", benchmark.sha256_bytes(public_payload))

    paths = benchmark.prepare_official_snapshots(source, tmp_path / "snapshots")

    assert paths["full"].read_bytes() == full_payload
    assert paths["public"].read_bytes() == public_payload
    assert paths["full"].stat().st_mode & 0o777 == 0o600
    assert paths["public"].stat().st_mode & 0o777 == 0o600
    assert paths["full"].parent.stat().st_mode & 0o777 == 0o700
    source.write_bytes(source.read_bytes() + b"tamper")
    with pytest.raises(benchmark.ProtocolError, match="parquet SHA-256 mismatch"):
        benchmark.prepare_official_snapshots(source, tmp_path / "other")


def test_private_artifacts_reject_repo_paths_symlink_parents_and_reused_directories(
    tmp_path: Path,
) -> None:
    with pytest.raises(benchmark.ProtocolError, match="outside the Mio repository"):
        benchmark.require_private_path(benchmark.ROOT / "private-gold.jsonl", must_exist=False)
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(benchmark.ProtocolError, match="symlink component"):
        benchmark.require_private_path(linked / "artifact.json", must_exist=False)

    evaluation = benchmark.create_private_directory(tmp_path / "evaluation")
    assert evaluation.stat().st_mode & 0o777 == 0o700
    with pytest.raises(benchmark.ProtocolError, match="new and exclusive"):
        benchmark.create_private_directory(evaluation)


def test_full_schedule_is_deterministic_adjacent_and_exactly_balanced() -> None:
    identifiers = [_instance_id(index) for index in range(500)]
    first = benchmark.make_balanced_schedule(identifiers)
    second = benchmark.make_balanced_schedule(tuple(reversed(identifiers)))

    assert first == second
    assert len(first) == 1000
    summary = benchmark.source_free_schedule_summary(first)
    assert summary == benchmark.source_free_schedule_summary(second)
    assert summary["pairs"] == 500
    assert summary["gate_off_first_pairs"] == 250
    assert summary["gate_on_first_pairs"] == 250
    assert summary["pair_arms_adjacent"] is True
    assert not any(identifier in json.dumps(summary) for identifier in identifiers)


def test_partial_schedule_is_explicitly_nonconfirmatory() -> None:
    ids = [_instance_id(index) for index in range(4)]
    with pytest.raises(benchmark.ProtocolError, match="exactly 500"):
        benchmark.make_balanced_schedule(ids)
    smoke = benchmark.make_balanced_schedule(ids, require_full=False)
    assert len(smoke) == 8
    assert benchmark.source_free_schedule_summary(smoke)["pairs"] == 4


def test_private_schedule_is_regenerated_not_trusted_from_recomputed_summary(tmp_path: Path) -> None:
    instances = tuple(
        benchmark.PublicInstance(
            instance_id=_instance_id(index),
            repo="owner/repository",
            base_commit=f"{index + 1:040x}",
            problem_statement=f"Fix {index}.",
        )
        for index in range(2)
    )
    document = benchmark.private_schedule_document(instances, evidence_run=False)
    document["schedule"][1]["condition"] = "gate_off"
    tampered = tuple(benchmark.ScheduleEntry(**row) for row in document["schedule"])
    document["source_free_summary"] = benchmark.source_free_schedule_summary(tampered)
    path = tmp_path / "schedule.json"
    path.write_bytes(benchmark.canonical_json_bytes(document))
    path.chmod(0o600)

    with pytest.raises(benchmark.ProtocolError, match="frozen balanced schedule"):
        benchmark.load_private_schedule(path)


def test_patch_adapter_captures_tracked_and_untracked_without_model_prose(tmp_path: Path) -> None:
    repo, base_commit = _init_repo(tmp_path)
    (repo / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "new_file.py").write_text("NEW = True\n", encoding="utf-8")

    patch = benchmark.capture_git_patch(repo, expected_base_commit=base_commit)

    assert patch.startswith("diff --git ")
    assert "module.py" in patch
    assert "new_file.py" in patch
    assert "VALUE = 2" in patch
    assert "NEW = True" in patch
    assert "```" not in patch
    with pytest.raises(benchmark.ProtocolError, match="raw git diff"):
        benchmark.official_prediction(_instance_id(0), "gate_off", "I changed the code successfully.")


def test_patch_adapter_fails_on_wrong_base_commit(tmp_path: Path) -> None:
    repo, _base_commit = _init_repo(tmp_path)
    with pytest.raises(benchmark.ProtocolError, match="HEAD differs"):
        benchmark.capture_git_patch(repo, expected_base_commit="f" * 40)


def test_patch_capture_ignores_host_git_redirects_and_repo_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base_commit = _init_repo(tmp_path)
    marker = tmp_path / "executed"
    command = tmp_path / "host-command.sh"
    command.write_text(f'#!/bin/sh\ntouch {marker}\ncat "$1"\n', encoding="utf-8")
    command.chmod(0o755)
    (repo / ".gitattributes").write_text("*.bin diff=hostile\n", encoding="utf-8")
    (repo / "payload.bin").write_bytes(b"\x00old")
    _git(repo, "add", ".gitattributes", "payload.bin")
    _git(repo, "commit", "--quiet", "-m", "binary base")
    base_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "config", "core.fsmonitor", str(command))
    _git(repo, "config", "diff.hostile.textconv", str(command))
    (repo / "payload.bin").write_bytes(b"\x00new")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong-git-dir"))
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", str(command))

    patch = benchmark.capture_git_patch(repo, expected_base_commit=base_commit)

    assert "payload.bin" in patch
    assert not marker.exists()


@pytest.mark.parametrize(
    "patch",
    [
        "diff --git a/x.bin b/x.bin\nnew file mode 100644\nindex 0000000..1111111\nGIT binary patch\nliteral 1\nAcmZQz\n",
        "diff --git a/old b/new\nsimilarity index 100%\nrename from old\nrename to new\n",
        "diff --git a/tool b/tool\nold mode 100644\nnew mode 100755\n",
    ],
)
def test_patch_validator_accepts_git_binary_rename_and_mode_sections(patch: str) -> None:
    benchmark.validate_patch_only(patch)


def test_patch_validator_preserves_markdown_fences_inside_raw_git_diff() -> None:
    patch = (
        "diff --git a/README.md b/README.md\n"
        "index 1111111..2222222 100644\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -0,0 +1,3 @@\n"
        "+```python\n"
        "+print('documented')\n"
        "+```\n"
    )

    benchmark.validate_patch_only(patch)


def test_checkpoint_store_is_atomic_idempotent_and_immutable(tmp_path: Path) -> None:
    schedule = benchmark.make_balanced_schedule([_instance_id(0), _instance_id(1)], require_full=False)
    digest = benchmark.schedule_digest(schedule)
    store = benchmark.CheckpointStore(tmp_path / "checkpoints")
    checkpoint = _checkpoint(schedule[0], digest)

    path = store.save(checkpoint)
    assert store.save(checkpoint) == path
    assert store.load(schedule[0]) == checkpoint
    with pytest.raises(benchmark.ProtocolError, match="different bytes"):
        store.save(replace(checkpoint, output_tokens=18))


def test_checkpoint_rejects_prefix_only_or_wrong_27b_identity() -> None:
    schedule = benchmark.make_balanced_schedule([_instance_id(0), _instance_id(1)], require_full=False)
    with pytest.raises(benchmark.ProtocolError, match="full local model identity"):
        _checkpoint(schedule[0], benchmark.schedule_digest(schedule), model_identity="local-sha256-v1:x")
    with pytest.raises(benchmark.ProtocolError, match="frozen Qwen"):
        _checkpoint(
            schedule[0],
            benchmark.schedule_digest(schedule),
            model_identity=f"local-sha256-v1:{'a' * 64}",
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"output_tokens": benchmark.MAX_OUTPUT_TOKENS_PER_ARM + 1},
        {"tool_calls": benchmark.MAX_TOOL_CALLS_PER_ARM + 1},
        {"wall_seconds": benchmark.MAX_AGENT_WALL_SECONDS + 0.001},
    ],
)
def test_checkpoint_enforces_frozen_agent_budgets(changes: dict[str, float | int]) -> None:
    schedule = benchmark.make_balanced_schedule([_instance_id(0), _instance_id(1)], require_full=False)
    with pytest.raises(benchmark.ProtocolError, match="invalid checkpoint metrics"):
        _checkpoint(schedule[0], benchmark.schedule_digest(schedule), **changes)


def test_completed_gate_checkpoint_cannot_be_incomplete() -> None:
    schedule = benchmark.make_balanced_schedule([_instance_id(0), _instance_id(1)], require_full=False)
    gate_on = next(entry for entry in schedule if entry.condition == "gate_on")
    with pytest.raises(benchmark.ProtocolError, match="completed gate_on"):
        _checkpoint(
            gate_on,
            benchmark.schedule_digest(schedule),
            status="completed",
            quality_gate_decision="incomplete",
        )


def test_resume_and_attempt_ledger_retain_whole_pair_retries(tmp_path: Path) -> None:
    schedule = benchmark.make_balanced_schedule([_instance_id(0), _instance_id(1)], require_full=False)
    digest = benchmark.schedule_digest(schedule)
    store = benchmark.CheckpointStore(tmp_path / "selected")
    store.save(_checkpoint(schedule[0], digest))
    assert benchmark.resume_entries(schedule, store) == schedule[1:]

    ledger = benchmark.AttemptLedger(tmp_path / "attempts.jsonl", digest)
    ledger.append(pair_index=0, attempt_index=0, event="started", reason_code="initial")
    ledger.append(
        pair_index=0,
        attempt_index=0,
        event="aborted",
        reason_code="infrastructure_host_loss",
    )
    ledger.append(
        pair_index=0,
        attempt_index=1,
        event="started",
        reason_code="infrastructure_host_loss",
    )
    hashes = {"gate_off": "a" * 64, "gate_on": "b" * 64}
    ledger.append(
        pair_index=0,
        attempt_index=1,
        event="completed",
        reason_code="completed",
        checkpoint_sha256s=hashes,
    )

    assert len(ledger.read()) == 4
    assert (
        benchmark.pair_attempt_store(tmp_path / "stores", 0, 0).root
        != benchmark.pair_attempt_store(tmp_path / "stores", 0, 1).root
    )
    with pytest.raises(benchmark.ProtocolError, match="forbidden after a completed"):
        ledger.append(
            pair_index=0,
            attempt_index=2,
            event="started",
            reason_code="infrastructure_process_crash",
        )
    with pytest.raises(benchmark.ProtocolError, match="event fields are invalid"):
        ledger.append(pair_index=1, attempt_index=0, event="started", reason_code="retry")


def test_export_requires_every_checkpoint_and_one_runtime_binding(tmp_path: Path) -> None:
    schedule = benchmark.make_balanced_schedule([_instance_id(0), _instance_id(1)], require_full=False)
    digest = benchmark.schedule_digest(schedule)
    store = benchmark.CheckpointStore(tmp_path / "checkpoints")
    for entry in schedule:
        store.save(_checkpoint(entry, digest))

    paths = benchmark.export_official_predictions(schedule, store, tmp_path / "predictions")

    assert set(paths) == set(benchmark.CONDITIONS)
    for condition, path in paths.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert all(set(row) == {"instance_id", "model_name_or_path", "model_patch"} for row in rows)
        assert all(row["model_name_or_path"] == benchmark.MODEL_LABELS[condition] for row in rows)

    conflicting_store = benchmark.CheckpointStore(tmp_path / "conflicting")
    for index, entry in enumerate(schedule):
        runtime = ("d" if index == 0 else "e") * 64
        conflicting_store.save(_checkpoint(entry, digest, runtime_digest=runtime))
    with pytest.raises(benchmark.ProtocolError, match="one Mio/model/runtime identity"):
        benchmark.export_official_predictions(schedule, conflicting_store, tmp_path / "bad")


def test_official_harness_commands_use_only_verified_local_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = tmp_path / "full.jsonl"
    full.write_text("pinned\n", encoding="utf-8")
    full.chmod(0o600)
    monkeypatch.setattr(benchmark, "FULL_SNAPSHOT_SHA256", benchmark.sha256_file(full))
    predictions = tmp_path / "predictions"
    predictions.mkdir(mode=0o700)
    for condition in benchmark.CONDITIONS:
        (predictions / f"{condition}.jsonl").write_text("{}\n", encoding="utf-8")
    distribution = "e" * 64
    commands = benchmark.official_harness_commands(
        predictions,
        full,
        schedule_sha256="a" * 64,
        max_workers=4,
        harness_distribution_sha256=distribution,
    )

    assert len(commands) == 2
    for command in commands:
        assert "swebench.harness.run_evaluation" in command
        assert str(full.resolve()) in command
        assert benchmark.DATASET_NAME not in command
        assert "--cache_level" in command
        assert "env" in command
        assert "--timeout" in command
        assert "1800" in command
        run_id = command[command.index("--run_id") + 1]
        prediction_path = Path(command[command.index("--predictions_path") + 1])
        condition = "gate_off" if "gate_off" in run_id else "gate_on"
        assert run_id == benchmark.evaluation_run_id(
            condition,
            benchmark.sha256_file(prediction_path),
            "a" * 64,
            1800,
            distribution,
        )


def test_run_id_binds_schedule_timeout_and_harness_distribution() -> None:
    base = benchmark.evaluation_run_id("gate_off", "a" * 64, "b" * 64, 1800, "c" * 64)
    assert base != benchmark.evaluation_run_id("gate_off", "a" * 64, "d" * 64, 1800, "c" * 64)
    assert base != benchmark.evaluation_run_id("gate_off", "a" * 64, "b" * 64, 60, "c" * 64)
    assert base != benchmark.evaluation_run_id("gate_off", "a" * 64, "b" * 64, 1800, "e" * 64)


def test_evaluation_receipt_binds_schedule_predictions_dataset_reports_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = benchmark.make_balanced_schedule([_instance_id(0), _instance_id(1)], require_full=False)
    schedule_sha256 = benchmark.schedule_digest(schedule)
    store = benchmark.CheckpointStore(tmp_path / "checkpoints")
    for entry in schedule:
        store.save(_checkpoint(entry, schedule_sha256))
    predictions = tmp_path / "predictions"
    benchmark.export_official_predictions(schedule, store, predictions)
    full = tmp_path / "full.jsonl"
    full.write_text("pinned evaluator fields\n", encoding="utf-8")
    full.chmod(0o600)
    monkeypatch.setattr(benchmark, "FULL_SNAPSHOT_SHA256", benchmark.sha256_file(full))
    seal = benchmark.build_evaluation_seal(
        schedule,
        store,
        predictions,
        full,
        max_workers=2,
        timeout_seconds=60,
        harness_distribution_sha256="e" * 64,
    )
    ids = list(benchmark._expected_instance_ids(schedule))
    reports = {}
    for condition in benchmark.CONDITIONS:
        path = tmp_path / f"{condition}.report.json"
        path.write_text(json.dumps(_official_report(ids, {ids[0]})), encoding="utf-8")
        path.chmod(0o600)
        reports[condition] = path
    receipt = benchmark.build_evaluation_receipt(
        seal,
        schedule,
        reports,
        observed_model_identity_before=benchmark.EXPECTED_MODEL_IDENTITY,
        observed_model_identity_after=benchmark.EXPECTED_MODEL_IDENTITY,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(benchmark.canonical_json_bytes(receipt))
    receipt_path.chmod(0o600)

    verified, digest = benchmark.verify_evaluation_receipt(
        receipt_path,
        schedule,
        store,
        predictions,
        full,
        reports,
    )

    assert verified == receipt
    assert digest == benchmark.sha256_file(receipt_path)
    reports["gate_on"].write_text(
        json.dumps(_official_report(ids, {ids[0], ids[1]})),
        encoding="utf-8",
    )
    with pytest.raises(benchmark.ProtocolError, match="differs from the immutable receipt"):
        benchmark.verify_evaluation_receipt(
            receipt_path,
            schedule,
            store,
            predictions,
            full,
            reports,
        )


def test_source_free_paired_aggregate_uses_official_resolved_sets(tmp_path: Path) -> None:
    ids = [_instance_id(index) for index in range(4)]
    off = _official_report(ids, {ids[0], ids[1]})
    on = _official_report(ids, {ids[0], ids[2], ids[3]})
    off_path = tmp_path / "off.json"
    on_path = tmp_path / "on.json"
    off_path.write_text(json.dumps(off), encoding="utf-8")
    on_path.write_text(json.dumps(on), encoding="utf-8")
    off_path.chmod(0o600)
    on_path.chmod(0o600)

    result = benchmark.aggregate_official_reports(
        off_path,
        on_path,
        expected_ids=ids,
        evidence_run=False,
        schedule_sha256="a" * 64,
        evaluation_receipt_sha256="b" * 64,
        generation_binding=_binding(),
    )

    assert result["status"] == "non_evidence_smoke"
    assert result["paired"]["both_resolved"] == 1
    assert result["paired"]["gate_off_only"] == 1
    assert result["paired"]["gate_on_only"] == 2
    assert result["paired"]["neither_resolved"] == 0
    assert result["paired"]["resolution_difference"] == pytest.approx(0.25)
    assert result["claim_gate"]["quality_improvement"] is False
    serialized = json.dumps(result)
    assert not any(identifier in serialized for identifier in ids)
    assert "resolved_ids" not in serialized


def test_confirmatory_evaluate_and_aggregate_are_blocked_without_generation_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confirmatory_schedule = tuple(object() for _ in range(2 * benchmark.EXPECTED_INSTANCES))
    monkeypatch.setattr(
        benchmark,
        "load_private_schedule",
        lambda _path: ({"evidence_class": "confirmatory"}, confirmatory_schedule),
    )
    with pytest.raises(benchmark.ProtocolError, match="generation runner"):
        benchmark._command_evaluate(
            Namespace(schedule=tmp_path / "schedule.json", timeout=benchmark.CONFIRMATORY_TIMEOUT_SECONDS)
        )
    with pytest.raises(benchmark.ProtocolError, match="generation runner"):
        benchmark._command_aggregate(Namespace(schedule=tmp_path / "schedule.json"))
    with pytest.raises(benchmark.ProtocolError, match="generation runner"):
        benchmark.aggregate_official_reports(
            tmp_path / "synthetic-off.json",
            tmp_path / "synthetic-on.json",
            expected_ids=[_instance_id(index) for index in range(benchmark.EXPECTED_INSTANCES)],
            evidence_run=True,
            schedule_sha256="a" * 64,
            evaluation_receipt_sha256="b" * 64,
            generation_binding=_binding(),
        )


def test_confirmatory_timeout_is_frozen_before_generation_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confirmatory_schedule = tuple(object() for _ in range(2 * benchmark.EXPECTED_INSTANCES))
    monkeypatch.setattr(
        benchmark,
        "load_private_schedule",
        lambda _path: ({"evidence_class": "confirmatory"}, confirmatory_schedule),
    )
    with pytest.raises(benchmark.ProtocolError, match="timeout is frozen at 1800"):
        benchmark._command_evaluate(Namespace(schedule=tmp_path / "schedule.json", timeout=60))
    with pytest.raises(benchmark.ProtocolError, match="timeout is frozen at 1800"):
        benchmark._command_commands(Namespace(schedule=tmp_path / "schedule.json", timeout=60))


def test_official_errors_fail_closed_before_aggregation(tmp_path: Path) -> None:
    ids = [_instance_id(index) for index in range(2)]
    off = _official_report(ids, set(), error_instances=1, error_ids=[ids[0]])
    on = _official_report(ids, set())
    off_path = tmp_path / "off.json"
    on_path = tmp_path / "on.json"
    off_path.write_text(json.dumps(off), encoding="utf-8")
    on_path.write_text(json.dumps(on), encoding="utf-8")
    off_path.chmod(0o600)
    on_path.chmod(0o600)

    with pytest.raises(benchmark.ProtocolError, match="block confirmatory aggregation"):
        benchmark.aggregate_official_reports(
            off_path,
            on_path,
            expected_ids=ids,
            evidence_run=False,
            schedule_sha256="a" * 64,
            evaluation_receipt_sha256="b" * 64,
            generation_binding=_binding(),
        )


def test_evaluation_host_preflight_fails_closed_on_macos_or_missing_docker() -> None:
    blocked = benchmark.assess_evaluation_host(
        machine="arm64",
        docker_cli_present=False,
        docker_daemon_ready=False,
        swebench_version=None,
        swebench_distribution_sha256=None,
        free_storage_gib=590,
    )
    ready = benchmark.assess_evaluation_host(
        machine="x86_64",
        docker_cli_present=True,
        docker_daemon_ready=True,
        swebench_version="4.1.0",
        swebench_distribution_sha256="a" * 64,
        free_storage_gib=200,
    )

    assert blocked["ready"] is False
    assert "confirmatory_evaluation_requires_x86_64" in blocked["blockers"]
    assert "docker_cli_missing" in blocked["blockers"]
    assert ready["ready"] is True
    assert ready["blockers"] == []


def test_model_tree_identity_binds_names_sizes_and_complete_bytes(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights-v1")
    first = benchmark.model_tree_identity(model)
    second = benchmark.model_tree_identity(model)
    (model / "model.safetensors").write_bytes(b"weights-v2")

    assert first == second
    assert first.startswith("local-sha256-v1:")
    assert benchmark.model_tree_identity(model) != first
