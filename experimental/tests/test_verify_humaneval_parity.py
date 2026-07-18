from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from experimental.effort.humaneval import (
    HumanEvalCase,
    HumanEvalReference,
    VerificationResult,
)
import experimental.effort.verify_humaneval_parity as parity_module
from experimental.effort.verify_humaneval_parity import (
    EXPECTED_HUMANEVAL_TASKS,
    GitState,
    build_parity_report,
    git_repository_root,
    inspect_git_state,
    main,
    output_is_inside_repository,
    write_json_atomic,
)


def reference(index: int = 0) -> HumanEvalReference:
    return HumanEvalReference(
        case=HumanEvalCase(
            task_id=f"HumanEval/{index}",
            prompt=f"def solve_{index}(value):\n",
            test=f"def check(candidate):\n    assert candidate(1) == {index + 1}\n",
            entry_point=f"solve_{index}",
        ),
        canonical_solution=f"    return value + {index}\n",
    )


def passing_verifier(
    case: HumanEvalCase,
    solution: str,
    *,
    timeout_s: float,
) -> VerificationResult:
    source = (case.prompt + solution).encode("utf-8")
    return VerificationResult(
        passed=True,
        status="passed",
        feedback="passed",
        elapsed_seconds=timeout_s / 100.0,
        source_sha256=hashlib.sha256(source).hexdigest(),
        output_sha256="0" * 64,
        output_chars=0,
    )


def test_report_is_source_free_and_claim_requires_clean_exact_parity() -> None:
    item = HumanEvalReference(
        case=HumanEvalCase(
            task_id="HumanEval/secret",
            prompt="def solve(value):\n",
            test="HIDDEN_TEST_SENTINEL = 'do not serialize'\n",
            entry_point="solve",
        ),
        canonical_solution="    return 'CANONICAL_SOURCE_SENTINEL'\n",
    )
    report = build_parity_report(
        (item,),
        timeout_s=10.0,
        git_state=GitState(revision="a" * 40, dirty=False),
        verifier_sha256="b" * 64,
        verifier=passing_verifier,
        expected_tasks=1,
    )

    serialized = json.dumps(report, sort_keys=True)
    assert "HIDDEN_TEST_SENTINEL" not in serialized
    assert "CANONICAL_SOURCE_SENTINEL" not in serialized
    assert report["claim"] == {
        "eligible": True,
        "parity": True,
        "passed": 1,
        "total": 1,
        "expected": 1,
        "ineligibility_reasons": [],
    }
    task = report["tasks"][0]
    assert set(task) == {
        "task_id",
        "status",
        "passed",
        "elapsed_seconds",
        "canonical_solution_sha256",
        "verified_source_sha256",
    }


def test_report_rejects_dirty_tree_even_when_all_references_pass() -> None:
    report = build_parity_report(
        (reference(),),
        timeout_s=10.0,
        git_state=GitState(revision="a" * 40, dirty=True),
        verifier_sha256="b" * 64,
        verifier=passing_verifier,
        expected_tasks=1,
    )
    claim = report["claim"]
    assert claim["parity"] is True
    assert claim["eligible"] is False
    assert claim["ineligibility_reasons"] == ["git_tree_dirty_before_verification"]


def test_report_rejects_source_changes_during_verification() -> None:
    report = build_parity_report(
        (reference(),),
        timeout_s=10.0,
        git_state=GitState(revision="a" * 40, dirty=False),
        verifier_sha256="b" * 64,
        verifier=passing_verifier,
        expected_tasks=1,
        post_verification_git_probe=lambda: GitState(
            revision="c" * 40,
            dirty=True,
        ),
    )
    claim = report["claim"]
    assert claim["parity"] is True
    assert claim["eligible"] is False
    assert claim["ineligibility_reasons"] == [
        "git_tree_dirty_after_verification",
        "git_revision_changed_during_verification",
    ]


def test_report_rejects_wrong_task_count() -> None:
    report = build_parity_report(
        (reference(),),
        timeout_s=10.0,
        git_state=GitState(revision="a" * 40, dirty=False),
        verifier_sha256="b" * 64,
        verifier=passing_verifier,
        expected_tasks=2,
    )
    claim = report["claim"]
    assert claim["parity"] is False
    assert claim["eligible"] is False
    assert claim["ineligibility_reasons"] == [
        "expected_2_tasks_but_loaded_1",
        "expected_2_passes_but_observed_1",
    ]


def test_atomic_writer_replaces_complete_json_without_temporary_files(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "parity.json"
    output.parent.mkdir()
    output.write_text("old", encoding="utf-8")
    report = {"schema": "test", "claim": {"eligible": True}}

    write_json_atomic(output, report)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_git_inspection_fails_closed_and_reports_clean_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "c" * 40

    def clean_git(
        arguments: list[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        stdout = revision + "\n" if arguments == ["rev-parse", "HEAD"] else ""
        return subprocess.CompletedProcess(["git"], 0, stdout=stdout, stderr="")

    monkeypatch.setattr(parity_module, "_run_git", clean_git)
    assert inspect_git_state(tmp_path) == GitState(revision=revision, dirty=False)

    def missing_git(
        arguments: list[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        del arguments, cwd
        raise FileNotFoundError

    monkeypatch.setattr(parity_module, "_run_git", missing_git)
    assert inspect_git_state(tmp_path) == GitState(revision="unknown", dirty=True)
    assert git_repository_root(tmp_path) is None


def test_output_path_must_resolve_outside_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    outside = tmp_path / "reports" / "parity.json"

    assert output_is_inside_repository(repository / "parity.json", repository)
    assert output_is_inside_repository(repository, repository)
    assert not output_is_inside_repository(outside, repository)


def test_main_rejects_output_inside_repository_before_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    output = repository / "parity.json"
    fetch_called = False

    def unexpected_fetch(_path: Path) -> Path:
        nonlocal fetch_called
        fetch_called = True
        raise AssertionError("corpus fetch must not start")

    monkeypatch.setattr(parity_module, "git_repository_root", lambda: repository)
    monkeypatch.setattr(parity_module, "fetch_humaneval", unexpected_fetch)

    assert main(["--quiet", "--output", str(output)]) == 2
    assert fetch_called is False
    assert not output.exists()


def test_main_writes_164_rows_and_exit_reflects_git_cleanliness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = tuple(reference(index) for index in range(EXPECTED_HUMANEVAL_TASKS))
    monkeypatch.setattr(parity_module, "fetch_humaneval", lambda path: path)
    monkeypatch.setattr(
        parity_module,
        "load_humaneval_references",
        lambda _path: references,
    )
    monkeypatch.setattr(parity_module, "verify_candidate", passing_verifier)
    monkeypatch.setattr(parity_module, "verifier_source_sha256", lambda: "d" * 64)
    monkeypatch.setattr(
        parity_module,
        "inspect_git_state",
        lambda: GitState(revision="e" * 40, dirty=False),
    )
    clean_output = tmp_path / "clean.json"

    assert main(["--quiet", "--output", str(clean_output)]) == 0
    clean_report = json.loads(clean_output.read_text(encoding="utf-8"))
    assert clean_report["claim"]["eligible"] is True
    assert len(clean_report["tasks"]) == EXPECTED_HUMANEVAL_TASKS

    monkeypatch.setattr(
        parity_module,
        "inspect_git_state",
        lambda: GitState(revision="e" * 40, dirty=True),
    )
    dirty_output = tmp_path / "dirty.json"
    assert main(["--quiet", "--output", str(dirty_output)]) == 1
    dirty_report = json.loads(dirty_output.read_text(encoding="utf-8"))
    assert dirty_report["claim"]["parity"] is True
    assert dirty_report["claim"]["eligible"] is False
