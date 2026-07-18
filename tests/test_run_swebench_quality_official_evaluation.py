from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import bench_swebench_quality as protocol
from scripts import run_swebench_quality_official_evaluation as evaluator


INSTANCE_IDS = ("django__django-15268", "matplotlib__matplotlib-24149")


def _instances() -> tuple[protocol.PublicInstance, ...]:
    return (
        protocol.PublicInstance(
            instance_id=INSTANCE_IDS[0],
            repo="django/django",
            base_commit="1" * 40,
            problem_statement="Fix the public Django behavior.",
        ),
        protocol.PublicInstance(
            instance_id=INSTANCE_IDS[1],
            repo="matplotlib/matplotlib",
            base_commit="2" * 40,
            problem_statement="Fix the public Matplotlib behavior.",
        ),
    )


def _patch(value: int) -> str:
    return (
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 0\n"
        f"+VALUE = {value}\n"
    )


def _write_private(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    path.chmod(0o600)
    return path.resolve(strict=True)


def _schedule(tmp_path: Path) -> tuple[Path, tuple[protocol.ScheduleEntry, ...]]:
    document = protocol.private_schedule_document(_instances(), evidence_run=False)
    path = _write_private(tmp_path / "private-schedule.json", protocol.canonical_json_bytes(document))
    _loaded, schedule = protocol.load_private_schedule(path)
    return path, schedule


def _fake_layout(
    tmp_path: Path,
    schedule: tuple[protocol.ScheduleEntry, ...],
    *,
    empty_condition: str | None = None,
    missing_condition: str | None = None,
    status_by_condition: dict[str, str] | None = None,
) -> SimpleNamespace:
    root = tmp_path / "generation"
    root.mkdir(mode=0o700)
    canonical = root / "canonical"
    canonical.mkdir(mode=0o700)
    receipt = _write_private(root / "generation-receipt.json", b"sealed-generation\n")
    digest = protocol.schedule_digest(schedule)
    store = protocol.CheckpointStore(canonical)
    statuses = status_by_condition or {}
    for index, entry in enumerate(schedule):
        if entry.condition == missing_condition:
            continue
        status = statuses.get(entry.condition, "completed")
        checkpoint = protocol.ArmCheckpoint.for_entry(
            entry,
            schedule_sha256=digest,
            status=status,
            model_patch="" if entry.condition == empty_condition else _patch(index + 1),
            mio_commit="b" * 40,
            model_identity=protocol.EXPECTED_MODEL_IDENTITY,
            runtime_digest="d" * 64,
            quality_gate_decision=(
                ("satisfied" if status == "completed" else "incomplete")
                if entry.condition == "gate_on"
                else "not_applicable"
            ),
            output_tokens=10,
            tool_calls=1,
            wall_seconds=1.0,
        )
        store.save(checkpoint)
    return SimpleNamespace(
        root=root.resolve(strict=True),
        canonical=canonical.resolve(strict=True),
        receipt=receipt,
        portable_artifacts=True,
    )


def _image_manifest(tmp_path: Path, ids: tuple[str, ...] = INSTANCE_IDS) -> tuple[Path, dict[str, str]]:
    digests = {instance_id: f"sha256:{index + 3:064x}" for index, instance_id in enumerate(ids)}
    document = {
        "schema": evaluator.IMAGE_MANIFEST_SCHEMA,
        "namespace": evaluator.NAMESPACE,
        "instance_image_tag": evaluator.INSTANCE_IMAGE_TAG,
        "images": [
            {
                "instance_id": instance_id,
                "repository": (f"swebench/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}"),
                "manifest_digest": digests[instance_id],
            }
            for instance_id in ids
        ],
    }
    path = _write_private(tmp_path / "images.json", protocol.canonical_json_bytes(document))
    return path, digests


def _harness_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "harness"
    entrypoint = root / "swebench" / "harness" / "run_evaluation.py"
    entrypoint.parent.mkdir(parents=True, mode=0o700)
    entrypoint.write_text("# pinned fake entry point\n", encoding="utf-8")
    (root / "swebench" / "__init__.py").write_text("", encoding="utf-8")
    base = tmp_path / "base-python"
    base_python = base / "bin" / "python"
    base_python.parent.mkdir(parents=True, mode=0o700)
    base_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    base_python.chmod(0o700)
    stdlib = base / "lib" / "python3.11"
    (stdlib / "lib-dynload").mkdir(parents=True, mode=0o700)
    (stdlib / "os.py").write_text("# fake pinned stdlib\n", encoding="utf-8")
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True, mode=0o700)
    python.symlink_to(base_python)
    (root / ".venv" / "pyvenv.cfg").write_text(f"home = {base / 'bin'}\n", encoding="utf-8")
    (root / ".venv" / "lib" / "python3.11" / "site-packages").mkdir(parents=True, mode=0o700)
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    docker.chmod(0o700)
    return root.resolve(strict=True), python.absolute(), docker.resolve(strict=True)


def _tests_status(resolved: bool) -> dict[str, dict[str, list[str]]]:
    return {
        "FAIL_TO_PASS": {
            "success": ["test_fix"] if resolved else [],
            "failure": [] if resolved else ["test_fix"],
        },
        "PASS_TO_PASS": {"success": ["test_regression"], "failure": []},
        "FAIL_TO_FAIL": {"success": [], "failure": []},
        "PASS_TO_FAIL": {"success": [], "failure": []},
    }


class FakeProcessRunner:
    def __init__(
        self,
        *,
        harness_root: Path,
        docker_host: str,
        image_digests: dict[str, str],
        aggregate_mutation: dict[str, Any] | None = None,
        instance_mutation: dict[str, Any] | None = None,
        nonzero_arm: str | None = None,
        corrupt_image: bool = False,
        harness_status: bytes = b"",
        harness_tree: str = evaluator.OFFICIAL_HARNESS_TREE,
        mutate_venv_arm: str | None = None,
        mutate_base_arm: str | None = None,
        report_for_empty: bool = False,
    ) -> None:
        self.harness_root = harness_root
        self.docker_host = docker_host
        self.image_digests = image_digests
        self.aggregate_mutation = aggregate_mutation or {}
        self.instance_mutation = instance_mutation or {}
        self.nonzero_arm = nonzero_arm
        self.corrupt_image = corrupt_image
        self.harness_status = harness_status
        self.harness_tree = harness_tree
        self.mutate_venv_arm = mutate_venv_arm
        self.mutate_base_arm = mutate_base_arm
        self.report_for_empty = report_for_empty
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self._tracked_files = {
            path.relative_to(self.harness_root).as_posix(): (path.read_bytes(), path.stat().st_mode)
            for path in (
                self.harness_root / "swebench" / "__init__.py",
                self.harness_root / "swebench" / "harness" / "run_evaluation.py",
            )
        }
        tree_rows = []
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
            for relative, (payload, mode) in sorted(self._tracked_files.items()):
                git_mode = "100755" if mode & 0o111 else "100644"
                tree_rows.append(f"{git_mode} blob {evaluator._git_blob_sha1(payload)}\t{relative}".encode() + b"\0")
                member = tarfile.TarInfo(relative)
                member.size = len(payload)
                member.mode = 0o755 if git_mode == "100755" else 0o644
                archive.addfile(member, io.BytesIO(payload))
        self._tree_payload = b"".join(tree_rows)
        self._archive_payload = archive_buffer.getvalue()

    @staticmethod
    def _completed(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> SimpleNamespace:
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    def __call__(self, command, **kwargs):  # noqa: ANN001, ANN204 - mirrors subprocess.run
        command = tuple(str(value) for value in command)
        self.calls.append((command, kwargs))
        if "rev-parse" in command and command[-1] == "HEAD":
            return self._completed(stdout=(evaluator.OFFICIAL_HARNESS_COMMIT + "\n").encode())
        if "rev-parse" in command and command[-1] == "HEAD^{tree}":
            return self._completed(stdout=(self.harness_tree + "\n").encode())
        if "status" in command and "--porcelain=v1" in command:
            return self._completed(stdout=self.harness_status)
        if "ls-tree" in command:
            return self._completed(stdout=self._tree_payload)
        if "archive" in command and "--format=tar" in command:
            return self._completed(stdout=self._archive_payload)
        if evaluator._ISOLATED_PROBE_CODE in command:
            base = self.harness_root.parent / "base-python"
            site_packages = self.harness_root / ".venv" / "lib" / "python3.11" / "site-packages"
            probe = {
                "base_prefix": str(base.resolve()),
                "distributions": [
                    {"name": "docker", "version": "7.1.0"},
                    {"name": "swebench", "version": "4.1.0"},
                ],
                "executable": str(Path(command[0]).resolve()),
                "flags": {
                    "dont_write_bytecode": 1,
                    "ignore_environment": 1,
                    "isolated": 1,
                    "no_site": 1,
                    "no_user_site": 1,
                },
                "module": str((self.harness_root / "swebench" / "__init__.py").resolve()),
                "platstdlib": str(base / "lib" / "python3.11"),
                "python": "3.11.15",
                "site_packages": str(site_packages),
                "stdlib": str(base / "lib" / "python3.11"),
                "sys_path": [
                    str(self.harness_root),
                    str(site_packages),
                    str(base / "lib" / "python311.zip"),
                    str(base / "lib" / "python3.11"),
                    str(base / "lib" / "python3.11" / "lib-dynload"),
                ],
            }
            return self._completed(stdout=json.dumps(probe).encode())
        if "context" in command and "inspect" in command:
            context = command[-1]
            payload = [{"Name": context, "Endpoints": {"docker": {"Host": self.docker_host}}}]
            return self._completed(stdout=json.dumps(payload).encode())
        if "version" in command and "{{json .Server}}" in command:
            payload = {
                "Version": "29.5.2",
                "ApiVersion": "1.54",
                "GitCommit": "engine-commit",
                "Os": "linux",
                "Arch": "amd64",
            }
            return self._completed(stdout=json.dumps(payload).encode())
        if "image" in command and "inspect" in command:
            tagged_reference, digest_reference = command[-2:]
            instance_id = next(
                identifier
                for identifier in self.image_digests
                if identifier.lower().replace("__", "_1776_") in tagged_reference
            )
            manifest_digest = self.image_digests[instance_id]
            repository = tagged_reference.rsplit(":", 1)[0]
            image_id = "sha256:" + "f" * 64 if self.corrupt_image else manifest_digest
            row = {
                "Id": image_id,
                "Architecture": "amd64",
                "Os": "linux",
                "RepoDigests": [f"{repository}@{manifest_digest}"],
            }
            digest_row = dict(row)
            if self.corrupt_image:
                digest_row["Id"] = manifest_digest
            return self._completed(stdout=json.dumps([row, digest_row]).encode())
        if "ps" in command and "--filter" in command:
            return self._completed(stdout=b"")
        if "swebench.harness.run_evaluation" in command:
            return self._run_harness(command, kwargs)
        raise AssertionError(f"unexpected subprocess command: {command}")

    def _run_harness(self, command: tuple[str, ...], kwargs: dict[str, Any]) -> SimpleNamespace:
        cwd = Path(kwargs["cwd"])
        arm = cwd.name
        if self.nonzero_arm == arm:
            kwargs["stdout"].write(b"private harness failure\n")
            return self._completed(returncode=7)
        run_id = command[command.index("--run_id") + 1]
        prediction_path = Path(command[command.index("--predictions_path") + 1])
        rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
        model_label = rows[0]["model_name_or_path"]
        start = command.index("--instance_ids") + 1
        end = command.index("--max_workers")
        instance_ids = list(command[start:end])
        assert instance_ids == sorted(INSTANCE_IDS)
        assert {row["instance_id"] for row in rows} == set(instance_ids)

        empty = {row["instance_id"] for row in rows if not row["model_patch"]}
        evaluated = set(instance_ids) - empty
        resolved = ({instance_ids[0]} & evaluated) if instance_ids else set()
        if arm == "quality":
            resolved = set(evaluated)
        unresolved = evaluated - resolved
        report = {
            "schema_version": 2,
            "total_instances": len(instance_ids),
            "submitted_instances": len(instance_ids),
            "completed_instances": len(evaluated),
            "resolved_instances": len(resolved),
            "unresolved_instances": len(unresolved),
            "empty_patch_instances": len(empty),
            "error_instances": 0,
            "completed_ids": sorted(evaluated),
            "incomplete_ids": [],
            "empty_patch_ids": sorted(empty),
            "submitted_ids": sorted(instance_ids),
            "resolved_ids": sorted(resolved),
            "unresolved_ids": sorted(unresolved),
            "error_ids": [],
        }
        report.update(self.aggregate_mutation)
        (cwd / f"{model_label}.{run_id}.json").write_text(json.dumps(report), encoding="utf-8")
        report_root = cwd / "logs" / "run_evaluation" / run_id / model_label
        for instance_id in sorted(evaluated):
            resolved_value = instance_id in resolved
            row = {
                "patch_is_None": False,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "resolved": resolved_value,
                "tests_status": _tests_status(resolved_value),
            }
            row.update(self.instance_mutation)
            directory = report_root / instance_id
            directory.mkdir(parents=True, mode=0o755)
            (directory / "report.json").write_text(json.dumps({instance_id: row}), encoding="utf-8")
        if self.report_for_empty and empty:
            instance_id = sorted(empty)[0]
            directory = report_root / instance_id
            directory.mkdir(parents=True, mode=0o755)
            (directory / "report.json").write_text(json.dumps({instance_id: {}}), encoding="utf-8")
        kwargs["stdout"].write(b"private official harness log\n")
        if self.mutate_venv_arm == arm:
            injected = self.harness_root / ".venv" / "lib" / "python3.11" / "site-packages" / "late.pth"
            injected.parent.mkdir(parents=True, exist_ok=True)
            injected.write_text("import late_drift\n", encoding="utf-8")
            self.mutate_venv_arm = None
        if self.mutate_base_arm == arm:
            stdlib = self.harness_root.parent / "base-python" / "lib" / "python3.11" / "os.py"
            stdlib.write_text("# mutated fake stdlib\n", encoding="utf-8")
            self.mutate_base_arm = None
        return self._completed()


class EvalContext:
    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        empty_condition: str | None = None,
        missing_condition: str | None = None,
        status_by_condition: dict[str, str] | None = None,
        **runner_options: Any,
    ) -> None:
        self.schedule_path, self.schedule = _schedule(tmp_path)
        self.layout = _fake_layout(
            tmp_path,
            self.schedule,
            empty_condition=empty_condition,
            missing_condition=missing_condition,
            status_by_condition=status_by_condition,
        )
        self.dataset = _write_private(tmp_path / "verified.parquet", b"pinned parquet bytes\n")
        monkeypatch.setattr(evaluator, "DATASET_PARQUET_SHA256", protocol.sha256_file(self.dataset))
        self.image_manifest, image_digests = _image_manifest(tmp_path)
        self.harness_root, self.python, self.docker = _harness_tree(tmp_path)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket_path = Path("/tmp") / f"mio-swe-{os.getpid()}-{id(self):x}.sock"
        self.socket.bind(str(self.socket_path))
        self.runner = FakeProcessRunner(
            harness_root=self.harness_root,
            docker_host=f"unix://{self.socket_path}",
            image_digests=image_digests,
            **runner_options,
        )
        self.verify_calls: list[dict[str, Any]] = []

        def verify_generation(**kwargs):
            self.verify_calls.append(kwargs)
            return "a" * 64

        self.dependencies = evaluator.EvaluationDependencies(
            run_process=self.runner,
            load_schedule=protocol.load_private_schedule,
            open_layout=lambda root: self.layout,
            verify_generation=verify_generation,
            tool_surface_digest=lambda: "e" * 64,
        )
        self.options = evaluator.EvaluationOptions(
            schedule_path=self.schedule_path,
            generation_layout=self.layout.root,
            dataset_path=self.dataset,
            harness_root=self.harness_root,
            python_executable=self.python,
            docker_executable=self.docker,
            docker_context="test-x86",
            image_manifest=self.image_manifest,
            output_root=tmp_path / "official-output",
        )

    def close(self) -> None:
        self.socket.close()
        self.socket_path.unlink(missing_ok=True)


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    value = EvalContext(tmp_path, monkeypatch)
    try:
        yield value
    finally:
        value.close()


def _harness_calls(runner: FakeProcessRunner) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    return [call for call in runner.calls if "swebench.harness.run_evaluation" in call[0]]


def test_official_evaluation_seals_exact_commands_reports_and_private_receipt(context: EvalContext) -> None:
    result = evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)

    assert result.status == "sealed_official_evaluation"
    assert result.plain_resolved == 1
    assert result.quality_resolved == 2
    assert len(context.verify_calls) == 2
    assert context.verify_calls[0]["receipt_path"] == context.layout.receipt
    assert context.verify_calls[0]["tool_surface_sha256"] == "e" * 64
    calls = _harness_calls(context.runner)
    assert len(calls) == 2
    assert {Path(kwargs["cwd"]).name for _command, kwargs in calls} == {"plain", "quality"}
    assert len({Path(kwargs["cwd"]) for _command, kwargs in calls}) == 2
    for command, kwargs in calls:
        assert command[1:6] == ("-I", "-B", "-S", "-c", evaluator._ISOLATED_LAUNCHER_CODE)
        assert command[9] == "swebench.harness.run_evaluation"
        assert command[command.index("--dataset_name") + 1] == str(context.dataset)
        assert command[command.index("--max_workers") + 1] == "1"
        assert command[command.index("--open_file_limit") + 1] == "4096"
        assert command[command.index("--timeout") + 1] == "1800"
        assert command[command.index("--force_rebuild") + 1] == "false"
        assert command[command.index("--cache_level") + 1] == "instance"
        assert command[command.index("--clean") + 1] == "false"
        assert command[command.index("--namespace") + 1] == "swebench"
        assert command[command.index("--instance_image_tag") + 1] == "v2"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert "PYTHONPATH" not in kwargs["env"]
        assert kwargs["env"]["DOCKER_HOST"].startswith("unix://")

    output = context.options.output_root
    receipt_path = output / "evaluation-receipt.json"
    payload = receipt_path.read_bytes()
    receipt = json.loads(payload)
    assert protocol.canonical_json_bytes(receipt) == payload
    assert protocol.sha256_file(receipt_path) == result.evaluation_receipt_sha256
    assert receipt["harness"]["git_commit"] == evaluator.OFFICIAL_HARNESS_COMMIT
    assert receipt["harness"]["git_tree"] == evaluator.OFFICIAL_HARNESS_TREE
    assert receipt["harness"]["isolated_no_site_execution"] is True
    assert receipt["harness"]["python_bytecode_writes_disabled"] is True
    assert receipt["harness"]["pth_policy"]["executed_or_added_to_sys_path"] is False
    assert receipt["harness"]["filesystem_manifest"]["unexpected_entry_count"] == 0
    assert receipt["dataset"]["parquet_sha256"] == evaluator.DATASET_PARQUET_SHA256
    assert receipt["parameters"] == evaluator._command_parameters()
    assert receipt["paired_outcomes"]["quality_minus_plain"] == 1
    assert len(receipt["docker"]["images"]) == 2
    assert all("prediction_sha256" in receipt["arms"][arm] for arm in evaluator.ARM_CONDITIONS)
    assert all("sha256" in receipt["arms"][arm]["aggregate_report"] for arm in evaluator.ARM_CONDITIONS)
    assert all(
        receipt["arms"][arm]["generation_outcomes"]["scheduled_terminal_checkpoints"] == len(INSTANCE_IDS)
        for arm in evaluator.ARM_CONDITIONS
    )
    assert receipt["paired_outcomes"]["all_preregistered_terminal_outcomes_included"] is True
    assert receipt["selection_by_terminal_status_or_patch_availability"] is False
    assert _patch(1).encode() not in payload
    assert "private official harness log" not in payload.decode()
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert all(path.stat().st_mode & 0o077 == 0 for path in output.rglob("*") if not path.is_symlink())


def test_dry_run_verifies_exports_and_plans_without_invoking_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch)
    try:
        options = evaluator.EvaluationOptions(**{**context.options.__dict__, "dry_run": True})
        result = evaluator.run_official_evaluation(options, dependencies=context.dependencies)

        assert result.status == "dry_run_preflight_complete"
        assert not _harness_calls(context.runner)
        assert (options.output_root / "evaluation-plan.json").is_file()
        assert not (options.output_root / "evaluation-receipt.json").exists()
        assert {path.name for path in (options.output_root / "predictions").iterdir()} == {
            "plain.predictions.jsonl",
            "quality.predictions.jsonl",
        }
        assert all(path.read_text() for path in (options.output_root / "predictions").iterdir())
    finally:
        context.close()


def test_empty_terminal_predictions_remain_in_denominator_as_effective_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(
        tmp_path,
        monkeypatch,
        empty_condition="gate_on",
    )
    try:
        result = evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)

        assert result.quality_resolved == 0
        receipt = json.loads((context.options.output_root / "evaluation-receipt.json").read_bytes())
        quality = receipt["arms"]["quality"]
        assert quality["empty_prediction_count"] == len(INSTANCE_IDS)
        assert quality["effective_unresolved_count"] == len(INSTANCE_IDS)
        assert quality["unresolved_count"] == 0
        assert quality["per_instance_reports"] == []
        assert quality["all_scheduled_outcomes_accounted_without_selection"] is True
        assert receipt["paired_outcomes"]["quality_effective_unresolved"] == len(INSTANCE_IDS)
        rows = [
            json.loads(line)
            for line in (context.options.output_root / "predictions" / "quality.predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == len(INSTANCE_IDS)
        assert all(row["model_patch"] == "" for row in rows)
    finally:
        context.close()


def test_empty_prediction_must_not_gain_a_per_instance_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, empty_condition="gate_on", report_for_empty=True)
    try:
        with pytest.raises(protocol.ProtocolError, match="report for an empty prediction"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not (context.options.output_root / "evaluation-receipt.json").exists()
    finally:
        context.close()


def test_missing_terminal_checkpoint_still_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, missing_condition="gate_on")
    try:
        with pytest.raises(protocol.ProtocolError, match="canonical generation directory has missing or extra entries"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not _harness_calls(context.runner)
        assert not (context.options.output_root / "evaluation-receipt.json").exists()
    finally:
        context.close()


def test_canonical_generation_directory_rejects_extra_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch)
    try:
        _write_private(context.layout.canonical / "ignored-extra.json", b"{}")
        with pytest.raises(protocol.ProtocolError, match="missing or extra entries"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert context.runner.calls == []
    finally:
        context.close()


def test_output_root_must_not_overlap_generation_or_harness_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch)
    try:
        overlapping = context.layout.root / "ignored-output"
        options = evaluator.EvaluationOptions(**{**context.options.__dict__, "output_root": overlapping})
        with pytest.raises(protocol.ProtocolError, match="overlaps an immutable input"):
            evaluator.run_official_evaluation(options, dependencies=context.dependencies)
        assert not overlapping.exists()
    finally:
        context.close()


def test_output_root_rejects_real_case_alias_on_case_insensitive_filesystem(tmp_path: Path) -> None:
    protected = tmp_path / "ProtectedCase"
    protected.mkdir()
    alias = tmp_path / "pROTECTEDcASE"
    try:
        if not alias.exists() or not os.path.samefile(alias, protected):
            pytest.skip("requires a case-insensitive filesystem")
    except OSError:
        pytest.skip("requires a case-insensitive filesystem")

    with pytest.raises(protocol.ProtocolError, match="filesystem spelling alias|overlaps an immutable input"):
        evaluator._exclusive_output_destination(alias / "official-output", [protected])


def test_expected_ids_rejects_duplicate_instance_pairs() -> None:
    schedule = protocol.make_balanced_schedule([INSTANCE_IDS[0], INSTANCE_IDS[1]], require_full=False)
    duplicate = tuple(
        protocol.ScheduleEntry(
            pair_index=entry.pair_index,
            execution_index=entry.execution_index,
            instance_id=INSTANCE_IDS[0],
            instance_digest=protocol._instance_digest(INSTANCE_IDS[0]),
            condition=entry.condition,
            position_in_pair=entry.position_in_pair,
        )
        for entry in schedule
    )

    with pytest.raises(protocol.ProtocolError, match="duplicate instance pairs"):
        evaluator._expected_ids(duplicate)


def test_image_manifest_rejects_missing_instance(tmp_path: Path) -> None:
    path, _digests = _image_manifest(tmp_path, (INSTANCE_IDS[0],))

    with pytest.raises(protocol.ProtocolError, match="missing, or extra"):
        evaluator._load_image_manifest(path, INSTANCE_IDS)


def test_local_image_tag_must_resolve_to_pinned_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, corrupt_image=True)
    try:
        with pytest.raises(protocol.ProtocolError, match="tag/digest/platform"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not context.options.output_root.exists()
    finally:
        context.close()


def test_harness_untracked_source_drift_fails_before_docker_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(
        tmp_path,
        monkeypatch,
        harness_status=b"?? sitecustomize.py\n",
    )
    try:
        with pytest.raises(protocol.ProtocolError, match="tracked or untracked source drift"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not context.options.output_root.exists()
        assert not any("context" in command for command, _kwargs in context.runner.calls)
    finally:
        context.close()


def test_physically_present_ignored_pycache_fails_before_docker_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch)
    try:
        cache = context.harness_root / "swebench" / "harness" / "__pycache__"
        cache.mkdir()
        (cache / "run_evaluation.cpython-311.pyc").write_bytes(b"real ignored runtime bytes")

        with pytest.raises(protocol.ProtocolError, match="entry outside the pinned tree"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not context.options.output_root.exists()
        assert not any("context" in command for command, _kwargs in context.runner.calls)
    finally:
        context.close()


def test_even_non_runtime_auxiliary_checkout_bytes_fail_exact_tree_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch)
    try:
        (context.harness_root / "LOCAL-NOTES.txt").write_text("not trusted\n", encoding="utf-8")

        with pytest.raises(protocol.ProtocolError, match="entry outside the pinned tree"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not context.options.output_root.exists()
    finally:
        context.close()


def test_harness_tree_is_exactly_pinned_and_replace_objects_are_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, harness_tree="0" * 40)
    try:
        with pytest.raises(protocol.ProtocolError, match="differs from the pinned tree"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        git_calls = [(command, kwargs) for command, kwargs in context.runner.calls if "rev-parse" in command]
        assert git_calls
        assert all(kwargs["env"]["GIT_NO_REPLACE_OBJECTS"] == "1" for _command, kwargs in git_calls)
        assert not context.options.output_root.exists()
    finally:
        context.close()


def test_tracked_head_manifest_detects_bytes_hidden_from_git_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch)
    try:
        entrypoint = context.harness_root / "swebench" / "harness" / "run_evaluation.py"
        entrypoint.write_text("# drift hidden by assume-unchanged\n", encoding="utf-8")

        with pytest.raises(protocol.ProtocolError, match="tracked harness"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not context.options.output_root.exists()
        assert not any("context" in command for command, _kwargs in context.runner.calls)
    finally:
        context.close()


def test_recursive_venv_manifest_hashes_pth_and_sitecustomize_content(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    site_packages = venv / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /trusted/python\n", encoding="utf-8")
    (site_packages / "runtime.pth").write_text("/trusted/harness\n", encoding="utf-8")
    customization = site_packages / "sitecustomize.py"
    customization.write_text("VALUE = 1\n", encoding="utf-8")

    before = evaluator._venv_manifest(venv)
    customization.write_text("VALUE = 2\n", encoding="utf-8")
    after = evaluator._venv_manifest(venv)

    assert before["manifest_sha256"] != after["manifest_sha256"]
    assert before["includes_package_code_metadata_pth_and_sitecustomize"] is True
    assert before["absolute_paths_serialized"] is False


def test_isolated_no_site_probe_ignores_external_pth_target_bytes(tmp_path: Path) -> None:
    root = tmp_path / "harness"
    package = root / "swebench"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    venv = tmp_path / "venv"
    site_packages = venv / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    target = external / "payload.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    marker = tmp_path / "pth-executed"
    (site_packages / "escape.pth").write_text(
        f"{external}\nimport pathlib;pathlib.Path({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    command = (
        sys.executable,
        "-I",
        "-B",
        "-S",
        "-c",
        evaluator._ISOLATED_PROBE_CODE,
        str(root),
        str(site_packages),
        str(tmp_path / "nonexistent-pycache"),
    )

    policy_before = evaluator._pth_policy_document(venv, site_packages)
    before = subprocess.run(command, check=True, capture_output=True).stdout
    target.write_text("VALUE = 2\n", encoding="utf-8")
    policy_after = evaluator._pth_policy_document(venv, site_packages)
    after = subprocess.run(command, check=True, capture_output=True).stdout
    probe = json.loads(after)

    assert before == after
    assert policy_before == policy_after
    assert policy_after["disabled_by_python_isolated_no_site"] is True
    assert not marker.exists()
    assert str(external) not in probe["sys_path"]
    assert probe["flags"] == {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
    }


@pytest.mark.skipif(sys.implementation.name != "cpython", reason="official evaluator pins CPython")
def test_bytecode_disable_flag_propagates_to_spawned_python_worker(tmp_path: Path) -> None:
    payload = tmp_path / "payload.py"
    payload.write_text("VALUE = 1\n", encoding="utf-8")
    probe = tmp_path / "spawn_probe.py"
    probe.write_text(
        "import multiprocessing as mp\n"
        "import pathlib\n"
        "import sys\n"
        "\n"
        "def worker(root):\n"
        "    if sys.flags.dont_write_bytecode != 1:\n"
        "        raise RuntimeError('spawned interpreter lost -B')\n"
        "    sys.path.insert(0, root)\n"
        "    import payload\n"
        "    if payload.VALUE != 1:\n"
        "        raise RuntimeError('payload import failed')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    process = mp.get_context('spawn').Process(target=worker, args=(sys.argv[1],))\n"
        "    process.start()\n"
        "    process.join(30)\n"
        "    if process.is_alive():\n"
        "        process.kill()\n"
        "        process.join()\n"
        "        raise RuntimeError('spawned interpreter timed out')\n"
        "    raise SystemExit(process.exitcode)\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        (sys.executable, "-I", "-B", "-S", str(probe), str(tmp_path)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=45,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert not tuple(tmp_path.rglob("*.pyc"))
    assert not tuple(tmp_path.rglob("__pycache__"))


def test_venv_drift_during_evaluation_fails_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, mutate_venv_arm="plain")
    try:
        with pytest.raises(protocol.ProtocolError, match="harness checkout or Python changed"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not (context.options.output_root / "evaluation-receipt.json").exists()
    finally:
        context.close()


def test_python_base_prefix_drift_during_evaluation_fails_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, mutate_base_arm="plain")
    try:
        with pytest.raises(protocol.ProtocolError, match="harness checkout or Python changed"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not (context.options.output_root / "evaluation-receipt.json").exists()
    finally:
        context.close()


@pytest.mark.parametrize("status", ["incomplete", "model_error", "timeout"])
def test_every_terminal_status_with_a_captured_patch_is_exported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, status_by_condition={"gate_on": status})
    try:
        options = evaluator.EvaluationOptions(**{**context.options.__dict__, "dry_run": True})
        evaluator.run_official_evaluation(options, dependencies=context.dependencies)

        rows = [
            json.loads(line)
            for line in (options.output_root / "predictions" / "quality.predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        plan = json.loads((options.output_root / "evaluation-plan.json").read_bytes())
        outcomes = plan["arms"]["quality"]["generation_outcomes"]
        assert len(rows) == len(INSTANCE_IDS)
        assert all(row["model_patch"] for row in rows)
        assert outcomes["checkpoint_status_counts"][status] == len(INSTANCE_IDS)
        assert outcomes["empty_prediction_count"] == 0
        assert outcomes["all_scheduled_terminal_outcomes_exported"] is True
    finally:
        context.close()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"error_instances": 1, "error_ids": [INSTANCE_IDS[0]]}, "error_instances"),
        ({"completed_instances": 1}, "completed_instances"),
        ({"submitted_ids": [INSTANCE_IDS[0]]}, "IDs are incomplete"),
        ({"resolved_ids": INSTANCE_IDS, "unresolved_ids": [INSTANCE_IDS[1]]}, "sets are incoherent"),
    ],
)
def test_zero_exit_is_rejected_when_aggregate_report_is_incoherent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, Any],
    match: str,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, aggregate_mutation=mutation)
    try:
        with pytest.raises(protocol.ProtocolError, match=match):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not (context.options.output_root / "evaluation-receipt.json").exists()
    finally:
        context.close()


def test_zero_exit_is_rejected_when_patch_was_not_applied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, instance_mutation={"patch_successfully_applied": False})
    try:
        with pytest.raises(protocol.ProtocolError, match="patch application fields"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not (context.options.output_root / "evaluation-receipt.json").exists()
    finally:
        context.close()


def test_zero_exit_is_rejected_when_resolution_disagrees_with_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, instance_mutation={"resolved": True})
    try:
        with pytest.raises(protocol.ProtocolError, match="resolution differs"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
    finally:
        context.close()


def test_nonzero_harness_exit_fails_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch, nonzero_arm="plain")
    try:
        with pytest.raises(protocol.ProtocolError, match="process failed"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert not (context.options.output_root / "evaluation-receipt.json").exists()
        assert (context.options.output_root / "plain" / "harness.log").read_bytes() == b"private harness failure\n"
    finally:
        context.close()


def test_dataset_digest_mismatch_stops_before_any_external_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = EvalContext(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(evaluator, "DATASET_PARQUET_SHA256", "0" * 64)
        with pytest.raises(protocol.ProtocolError, match="parquet SHA-256 mismatch"):
            evaluator.run_official_evaluation(context.options, dependencies=context.dependencies)
        assert context.runner.calls == []
    finally:
        context.close()


def test_public_result_is_content_free() -> None:
    result = evaluator.EvaluationResult(
        status="sealed_official_evaluation",
        schedule_sha256="a" * 64,
        generation_receipt_sha256="b" * 64,
        plan_sha256="c" * 64,
        pair_count=2,
        evaluation_receipt_sha256="d" * 64,
        plain_resolved=1,
        quality_resolved=2,
    ).public_dict()

    assert result["plain_resolved"] == 1
    assert result["quality_resolved"] == 2
    assert result["contains_issue_model_patch_or_evaluator_text"] is False
    assert "path" not in json.dumps(result)
    assert "patch" in "contains_issue_model_patch_or_evaluator_text"
