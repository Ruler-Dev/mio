from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import mio.coding_quality as coding_quality
from mio.agent_policy import AgentAuditEvent
from mio.coding_quality import (
    CodingEffort,
    CodingQualityGate,
    GateStatus,
    RequestIntent,
    ValidationKind,
    classify_request_intent,
    infer_validation_kind,
    snapshot_workspaces,
)


def _audit(
    operation: str,
    *,
    outcome: str = "ok",
    allowed: bool = True,
    target: str = "content-free-target",
) -> AgentAuditEvent:
    return AgentAuditEvent(
        timestamp=1.0,
        operation=operation,
        permission="write" if operation in {"write", "edit"} else "shell",
        target=target,
        allowed=allowed,
        outcome=outcome,
        detail="",
    )


def _record_edit(
    gate: CodingQualityGate,
    path: Path,
    content: str,
    *,
    operation: str = "edit",
) -> None:
    before = gate.before_tool(operation, {"path": str(path)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    gate.after_tool(
        operation,
        {"path": str(path)},
        before=before,
        audit_events=[_audit(operation, target=str(path))],
    )


def _record_validation(
    gate: CodingQualityGate,
    kind: ValidationKind,
    argv: tuple[str, ...],
    *,
    allowed: bool = True,
    outcome: str = "ok",
) -> None:
    gate.record_validation(
        kind,
        argv=argv,
        allowed=allowed,
        outcome=outcome,
    )


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("pytest", "-q"), ValidationKind.TEST),
        (("python3", "-m", "pytest", "-q"), ValidationKind.TEST),
        (("python", "-m", "unittest", "discover"), ValidationKind.TEST),
        (("npm", "test", "--", "--runInBand"), ValidationKind.TEST),
        (("pnpm", "run", "test:unit"), ValidationKind.TEST),
        (("cargo", "test", "--workspace"), ValidationKind.TEST),
        (("go", "test", "./..."), ValidationKind.TEST),
        (("ruff", "check", "."), ValidationKind.STATIC),
        (("python3", "-m", "compileall", "mio"), ValidationKind.STATIC),
        (("tsc", "--noEmit"), ValidationKind.STATIC),
        (("cargo", "clippy", "--all-targets"), ValidationKind.STATIC),
        (("npm", "run", "build"), ValidationKind.BUILD),
        (("python3", "-m", "build"), ValidationKind.BUILD),
        (("git", "diff", "--check"), ValidationKind.DIFF),
    ],
)
def test_validation_argv_classifier_accepts_direct_recognized_commands(
    argv: tuple[str, ...],
    expected: ValidationKind,
) -> None:
    assert infer_validation_kind(argv) is expected


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("echo", "pytest"),
        ("bash", "-c", "pytest -q"),
        ("sh", "-c", "cargo test"),
        ("env", "pytest", "-q"),
        ("timeout", "10", "pytest"),
        ("uv", "run", "pytest"),
        ("python3", "-c", "import pytest; pytest.main()"),
        ("python3", "script.py"),
        ("pytest", "||", "true"),
        ("pytest", "&&", "ruff", "check", "."),
        ("pytest", "|", "tee", "result.txt"),
        ("pytest", ";", "true"),
        ("pytest", ">", "result.txt"),
        ("pytest\ntrue",),
    ],
)
def test_validation_argv_classifier_rejects_shell_wrappers_inline_code_and_grammar(
    argv: tuple[str, ...],
) -> None:
    assert infer_validation_kind(argv) is None


def test_validation_argv_classifier_rejects_non_argv_inputs() -> None:
    assert infer_validation_kind("pytest -q") is None
    assert infer_validation_kind([]) is None
    assert infer_validation_kind(["pytest", ""]) is None
    assert infer_validation_kind(["pytest", "bad\x00arg"]) is None


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Implement the missing parser", RequestIntent.CODE_CHANGE_REQUESTED),
        ("Correggi il bug nel parser", RequestIntent.CODE_CHANGE_REQUESTED),
        ("Inspect the parser architecture", RequestIntent.INSPECT),
        ("Spiegami come funziona", RequestIntent.INSPECT),
        ("Hello Mio", RequestIntent.GENERAL),
    ],
)
def test_request_intent_is_advisory_and_multilingual(
    prompt: str,
    expected: RequestIntent,
) -> None:
    assert classify_request_intent(prompt) is expected


def _init_git_repo(root: Path) -> Path:
    root.mkdir()
    tracked = root / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Mio Test",
            "-c",
            "user.email=mio@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return tracked


def test_git_snapshot_preserves_preexisting_dirty_baseline_and_detects_new_and_deleted_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    tracked = _init_git_repo(root)
    tracked.write_text("VALUE = 2  # preexisting dirty\n", encoding="utf-8")

    dirty_baseline = snapshot_workspaces([root])
    repeated = snapshot_workspaces([root])

    assert dirty_baseline.complete is True
    assert dirty_baseline.method == "git"
    assert repeated.revision_sha256 == dirty_baseline.revision_sha256
    assert repeated.entries == dirty_baseline.entries

    new_file = root / "new.txt"
    new_file.write_text("new untracked file\n", encoding="utf-8")
    with_new_file = snapshot_workspaces([root])
    assert with_new_file.revision_sha256 != dirty_baseline.revision_sha256
    assert {entry.suffix for entry in with_new_file.entries} == {".py", ".txt"}

    tracked.unlink()
    with_deletion = snapshot_workspaces([root])
    assert with_deletion.revision_sha256 != with_new_file.revision_sha256
    assert hashlib.sha256(b"deleted").hexdigest() in {entry.state_sha256 for entry in with_deletion.entries}


def test_snapshot_falls_back_to_bounded_manifest_when_git_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "plain-workspace"
    root.mkdir()
    source = root / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (root / "README.md").write_text("documentation\n", encoding="utf-8")
    monkeypatch.setattr(
        coding_quality,
        "_run_git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )

    before = snapshot_workspaces([root])
    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = snapshot_workspaces([root])

    assert before.complete is True
    assert before.method == "manifest"
    assert len(before.entries) == 2
    assert after.revision_sha256 != before.revision_sha256
    serialized = json.dumps(
        {
            "revision": after.revision_sha256,
            "entries": [entry.__dict__ for entry in after.entries],
        }
    )
    assert "module.py" not in serialized
    assert "VALUE = 2" not in serialized


def test_requested_change_without_a_workspace_diff_is_incomplete(tmp_path: Path) -> None:
    gate = CodingQualityGate.start(
        [tmp_path],
        "Implement a parser change, then stop without editing.",
        effort=CodingEffort.LOW,
    )

    decision = gate.decision()

    assert gate.mutation_epoch == 0
    assert decision.status is GateStatus.INCOMPLETE
    assert decision.phase == "change_required"
    assert decision.missing == ("workspace_mutation",)


def test_successful_edit_audit_activates_gate_even_when_snapshot_has_no_delta(
    tmp_path: Path,
) -> None:
    target = tmp_path / "same.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    gate = CodingQualityGate.start([tmp_path], "hello", effort=CodingEffort.LOW)
    before = gate.before_tool("edit", {"path": str(target)})

    gate.after_tool(
        "edit",
        {"path": str(target)},
        before=before,
        audit_events=[_audit("edit", target=str(target))],
    )

    assert gate.mutation_epoch == 1
    assert gate.decision().status is GateStatus.INCOMPLETE
    assert gate.decision().missing == ("any_validation",)


def test_denied_or_failed_edit_does_not_create_a_mutation_epoch(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")

    for event in (
        _audit("edit", allowed=False, outcome="denied", target=str(target)),
        _audit("edit", allowed=True, outcome="old_string_not_found", target=str(target)),
        _audit("write", allowed=True, outcome="error", target=str(target)),
    ):
        gate = CodingQualityGate.start([tmp_path], "hello", effort=CodingEffort.LOW)
        before = gate.before_tool(event.operation, {"path": str(target)})
        gate.after_tool(
            event.operation,
            {"path": str(target)},
            before=before,
            audit_events=[event],
        )
        assert gate.mutation_epoch == 0
        assert gate.decision().status is GateStatus.NOT_APPLICABLE


def test_new_mutation_epoch_invalidates_previously_passing_validation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    gate = CodingQualityGate.start([tmp_path], "hello", effort=CodingEffort.HIGH)
    _record_edit(gate, target, "VALUE = 2\n")
    _record_validation(gate, ValidationKind.TEST, ("pytest", "-q"))
    _record_validation(gate, ValidationKind.DIFF, ("git", "diff", "--check"))

    assert gate.mutation_epoch == 1
    assert gate.decision().status is GateStatus.PASS

    _record_edit(gate, target, "VALUE = 3\n")

    decision = gate.decision()
    assert gate.mutation_epoch == 2
    assert decision.status is GateStatus.INCOMPLETE
    assert decision.missing == ("test", "static_or_diff")
    assert gate.report()["validation_counts"] == {
        "test": 0,
        "build": 0,
        "static": 0,
        "diff": 0,
        "review": 0,
    }


def test_validation_bound_to_a_stale_snapshot_cannot_satisfy_current_revision(
    tmp_path: Path,
) -> None:
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    gate = CodingQualityGate.start([tmp_path], "hello", effort=CodingEffort.LOW)
    stale_snapshot = gate.current_snapshot
    assert stale_snapshot is not None
    _record_edit(gate, target, "VALUE = 2\n")

    gate.record_validation(
        ValidationKind.TEST,
        argv=("pytest", "-q"),
        allowed=True,
        outcome="ok",
        snapshot=stale_snapshot,
    )

    assert gate.mutation_epoch == 1
    assert gate.decision().status is GateStatus.INCOMPLETE
    assert gate.decision().missing == ("any_validation",)


@pytest.mark.parametrize("outcome", ["nonzero", "timeout", "output_limit"])
def test_unsuccessful_validation_outcomes_never_satisfy_gate(
    tmp_path: Path,
    outcome: str,
) -> None:
    target = tmp_path / f"module-{outcome}.py"
    gate = CodingQualityGate.start([tmp_path], "hello", effort=CodingEffort.LOW)
    _record_edit(gate, target, "VALUE = 1\n")
    before = gate.before_tool("validate", {"argv": ["pytest", "-q"]})
    gate.after_tool(
        "validate",
        {"argv": ["pytest", "-q"]},
        before=before,
        audit_events=[_audit("validate", outcome=outcome)],
    )

    decision = gate.decision()
    assert decision.status is GateStatus.INCOMPLETE
    assert decision.phase == "validation_failed"
    assert decision.missing == ("any_validation",)
    assert gate.report()["validation_attempts"] == 1
    assert gate.report()["validation_counts"]["test"] == 0


def test_denied_validation_is_recorded_but_never_satisfies_gate(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    gate = CodingQualityGate.start([tmp_path], "hello", effort=CodingEffort.LOW)
    _record_edit(gate, target, "VALUE = 1\n")
    gate.record_validation(
        ValidationKind.TEST,
        argv=("pytest", "-q"),
        allowed=False,
        outcome="denied",
    )

    assert gate.decision().status is GateStatus.INCOMPLETE
    assert gate.report()["validation_attempts"] == 1
    assert gate.report()["validation_counts"]["test"] == 0


@pytest.mark.parametrize("tool_name", ["bash", "call_mcp_tool"])
def test_bash_and_mcp_snapshot_delta_create_fail_closed_code_mutation(
    tmp_path: Path,
    tool_name: str,
) -> None:
    root = tmp_path / tool_name
    root.mkdir()
    gate = CodingQualityGate.start([root], "hello", effort=CodingEffort.MEDIUM)
    before = gate.before_tool(tool_name, {})
    (root / "README.md").write_text("changed indirectly\n", encoding="utf-8")

    gate.after_tool(
        tool_name,
        {},
        before=before,
        audit_events=[_audit(tool_name, target="opaque")],
    )

    decision = gate.decision()
    assert gate.mutation_epoch == 1
    assert gate.changed_kinds == {"code"}
    assert decision.status is GateStatus.INCOMPLETE
    assert decision.missing == ("test_or_build",)


def test_report_is_content_free_even_with_secret_request_paths_source_and_argv(
    tmp_path: Path,
) -> None:
    secret_root_marker = "NEVER_SERIALIZE_ROOT_6d57"
    secret_request_marker = "NEVER_SERIALIZE_REQUEST_f21a"
    secret_source_marker = "NEVER_SERIALIZE_SOURCE_b0bd"
    secret_command_marker = "NEVER_SERIALIZE_COMMAND_9be2"
    root = tmp_path / secret_root_marker
    root.mkdir()
    target = root / "NEVER_SERIALIZE_PATH_22b8.py"
    gate = CodingQualityGate.start(
        [root],
        f"Implement {secret_request_marker}",
        effort=CodingEffort.LOW,
    )
    _record_edit(gate, target, f"VALUE = '{secret_source_marker}'\n")
    gate.record_validation(
        ValidationKind.TEST,
        argv=("pytest", secret_command_marker),
        allowed=True,
        outcome="ok",
    )

    serialized = json.dumps(gate.report(), sort_keys=True)

    assert gate.decision().status is GateStatus.PASS
    assert gate.request_sha256 in serialized
    assert str(root) not in serialized
    assert secret_root_marker not in serialized
    assert secret_request_marker not in serialized
    assert secret_source_marker not in serialized
    assert secret_command_marker not in serialized
    assert target.name not in serialized
    assert "pytest" not in serialized


def _gate_with_code_mutation(tmp_path: Path, effort: CodingEffort) -> CodingQualityGate:
    root = tmp_path / effort.value
    root.mkdir()
    target = root / "module.py"
    gate = CodingQualityGate.start([root], "hello", effort=effort)
    _record_edit(gate, target, "VALUE = 1\n")
    return gate


def test_effort_profiles_are_monotonic_for_the_same_test_evidence(tmp_path: Path) -> None:
    statuses: dict[CodingEffort, GateStatus] = {}
    for effort in CodingEffort:
        gate = _gate_with_code_mutation(tmp_path, effort)
        _record_validation(gate, ValidationKind.TEST, ("pytest", "-q"))
        statuses[effort] = gate.decision().status

    assert statuses == {
        CodingEffort.LOW: GateStatus.PASS,
        CodingEffort.MEDIUM: GateStatus.PASS,
        CodingEffort.HIGH: GateStatus.INCOMPLETE,
        CodingEffort.XHIGH: GateStatus.INCOMPLETE,
        CodingEffort.ULTRA: GateStatus.INCOMPLETE,
    }


def test_medium_docs_relaxation_does_not_apply_to_code(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs-root"
    docs_root.mkdir()
    docs_gate = CodingQualityGate.start([docs_root], "hello", effort=CodingEffort.MEDIUM)
    _record_edit(docs_gate, docs_root / "README.md", "updated docs\n")
    _record_validation(docs_gate, ValidationKind.DIFF, ("git", "diff", "--check"))

    code_gate = _gate_with_code_mutation(tmp_path, CodingEffort.MEDIUM)
    _record_validation(code_gate, ValidationKind.DIFF, ("git", "diff", "--check"))

    assert docs_gate.changed_kinds == {"docs"}
    assert docs_gate.decision().status is GateStatus.PASS
    assert code_gate.changed_kinds == {"code"}
    assert code_gate.decision().status is GateStatus.INCOMPLETE
    assert code_gate.decision().missing == ("test_or_build",)


def test_high_and_xhigh_require_distinct_additional_categories(tmp_path: Path) -> None:
    high = _gate_with_code_mutation(tmp_path, CodingEffort.HIGH)
    _record_validation(high, ValidationKind.TEST, ("pytest", "-q"))
    assert high.decision().missing == ("static_or_diff",)
    _record_validation(high, ValidationKind.DIFF, ("git", "diff", "--check"))
    assert high.decision().status is GateStatus.PASS

    xhigh = _gate_with_code_mutation(tmp_path, CodingEffort.XHIGH)
    _record_validation(xhigh, ValidationKind.TEST, ("pytest", "-q"))
    _record_validation(xhigh, ValidationKind.DIFF, ("git", "diff", "--check"))
    assert xhigh.decision().missing == ("static",)
    _record_validation(xhigh, ValidationKind.STATIC, ("ruff", "check", "."))
    assert xhigh.decision().status is GateStatus.PASS


def test_ultra_requires_review_or_second_distinct_successful_test(tmp_path: Path) -> None:
    gate = _gate_with_code_mutation(tmp_path, CodingEffort.ULTRA)
    _record_validation(gate, ValidationKind.TEST, ("pytest", "-q"))
    _record_validation(gate, ValidationKind.STATIC, ("ruff", "check", "."))
    _record_validation(gate, ValidationKind.DIFF, ("git", "diff", "--check"))

    assert gate.decision().missing == ("review_or_second_distinct_test",)

    _record_validation(gate, ValidationKind.TEST, ("pytest", "-q"))
    assert gate.decision().missing == ("review_or_second_distinct_test",)

    _record_validation(gate, ValidationKind.TEST, ("pytest", "tests/test_other.py", "-q"))
    assert gate.decision().status is GateStatus.PASS


def test_ultra_accepts_distinct_trusted_review_alternative(tmp_path: Path) -> None:
    gate = _gate_with_code_mutation(tmp_path, CodingEffort.ULTRA)
    _record_validation(gate, ValidationKind.TEST, ("pytest", "-q"))
    _record_validation(gate, ValidationKind.STATIC, ("ruff", "check", "."))
    _record_validation(gate, ValidationKind.DIFF, ("git", "diff", "--check"))
    _record_validation(
        gate,
        ValidationKind.REVIEW,
        ("trusted-review-adapter", "opaque"),
    )

    assert gate.decision().status is GateStatus.PASS
    assert gate.report()["validation_counts"]["review"] == 1


def test_standalone_gate_defaults_to_high_and_disabled_control_is_explicit(
    tmp_path: Path,
) -> None:
    default_gate = CodingQualityGate.start([tmp_path], "hello")
    disabled = CodingQualityGate.start(
        [tmp_path],
        "Implement a change",
        effort=CodingEffort.ULTRA,
        enabled=False,
    )

    assert default_gate.effort is CodingEffort.HIGH
    assert disabled.decision().status is GateStatus.NOT_APPLICABLE
    assert disabled.decision().phase == "disabled"
    assert disabled.decision().satisfied is True


def test_unknown_effort_is_rejected_instead_of_silently_aliased(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CodingQualityGate.start([tmp_path], "hello", effort="impossible")
