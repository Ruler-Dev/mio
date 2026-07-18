from __future__ import annotations

import hashlib
import json
import os
import shlex
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
    infer_misrouted_validation_command,
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
        (("python3", "-B", "-m", "unittest", "discover"), ValidationKind.TEST),
        (("python3", "-I", "-m", "pytest", "-q"), ValidationKind.TEST),
        (("python", "-m", "unittest", "discover"), ValidationKind.TEST),
        (("npm", "test", "--", "--runInBand"), ValidationKind.TEST),
        (("pnpm", "run", "test:unit"), ValidationKind.TEST),
        (("cargo", "test", "--workspace"), ValidationKind.TEST),
        (("go", "test", "./..."), ValidationKind.TEST),
        (("ruff", "check", "."), ValidationKind.STATIC),
        (("ruff", "format", "--check", "."), ValidationKind.STATIC),
        (("node", "--check", "module.js"), ValidationKind.STATIC),
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
        ("python3", "-V", "-m", "compileall", "mio"),
        ("python3", "-VV", "-m", "unittest", "discover"),
        ("python3", "script.py"),
        ("pytest", "||", "true"),
        ("pytest", "&&", "ruff", "check", "."),
        ("pytest", "|", "tee", "result.txt"),
        ("pytest", ";", "true"),
        ("pytest", ">", "result.txt"),
        ("pytest\ntrue",),
        ("pytest", "--help"),
        ("pytest", "--co"),
        ("pytest", "--collect-only"),
        ("pytest", "--setup-plan"),
        ("pytest", "--fixtures-per-test"),
        ("pytest", "-V"),
        ("pytest", "-VV"),
        ("python3", "-m", "pytest", "-V"),
        ("tox", "-l"),
        ("tox", "-a"),
        ("tox", "-av"),
        ("tox", "config"),
        ("tox", "devenv", ".venv"),
        ("tox", "exec", "python", "-V"),
        ("tox", "--notest"),
        ("nox", "-l"),
        ("nox", "--install-only"),
        ("ctest", "--show-only"),
        ("ctest", "--show-only=json-v1"),
        ("ctest", "--help-command", "add_test"),
        ("ctest", "--help-full"),
        ("ctest", "--help-variable", "CTEST_COMMAND"),
        ("ctest", "--print-labels"),
        ("ctest", "-S", "dashboard.cmake"),
        ("ctest", "-T", "Start"),
        ("npm", "test", "--", "--listTests"),
        ("npm", "test", "--", "--passWithNoTests"),
        ("npm", "--prefix", "test", "exec", "true"),
        ("swift", "test", "list"),
        ("python3", "-m", "ruff", "clean"),
        ("python3", "-m", "ruff", "check", "--exit-zero", "."),
        ("ruff", "format", "."),
        ("ruff", "check", "--exit-zero", "."),
        ("ruff", "check", "--fix-only", "."),
        ("ruff", "check", "--show-files"),
        ("ruff", "check", "-"),
        ("ruff", "format", "--check", "-"),
        ("python3", "-m", "ruff", "check", "--show-settings", "."),
        ("python3", "-m", "compileall", "-q", "mio"),
        ("node", "--check"),
        ("go", "build", "-n", "./..."),
        ("tsc", "--showConfig"),
        ("tsc", "--init"),
        ("tsc", "--build", "--clean"),
        ("tsc", "--listFilesOnly"),
        ("tsc", "--build", "--dry"),
        ("tsc", "-v"),
        ("eslint", "--print-config", "module.js"),
        ("eslint", "--env-info"),
        ("eslint", "-v"),
        ("stylelint", "-v"),
        ("mypy", "-V"),
        ("python3", "-m", "mypy", "-V"),
        ("pyright", "--createstub", "module"),
        ("./pytest", "-q"),
        ("cargo", "test", "--no-run"),
        ("cargo", "build", "--build-plan"),
        ("go", "test", "-list", "."),
        ("go", "test", "-c"),
        ("go", "test", "-exec", "echo"),
        ("ctest", "-N"),
        ("make", "-n", "test"),
        ("make", "-q", "test"),
        ("make", "-t", "test"),
        ("make", "-sn", "test"),
        ("mvn", "test", "-DskipTests"),
        ("mvn", "-f", "test", "help:evaluate"),
        ("gradle", "-m", "test"),
        ("dotnet", "test", "-t"),
        ("swift", "test", "-l"),
        ("swift", "build", "--show-bin-path"),
        ("git", "diff", "HEAD", "HEAD", "--check"),
        ("git", "diff", "--no-index", "--check", "/dev/null", "/dev/null"),
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
        ("Aggiorna e ottimizza il parser", RequestIntent.CODE_CHANGE_REQUESTED),
        ("Installa e configura il parser", RequestIntent.CODE_CHANGE_REQUESTED),
        ("Do not modify it; explain how to implement it", RequestIntent.INSPECT),
        ("Non modificare nulla; spiega come implementarlo", RequestIntent.INSPECT),
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


def test_git_snapshot_never_invokes_repository_fsmonitor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_git_repo(root)
    marker = tmp_path / "fsmonitor-was-invoked"
    hook = root / "fsmonitor-hook.sh"
    hook.write_text(
        f"#!/bin/sh\nprintf invoked > {shlex.quote(str(marker))}\nexit 0\n",
        encoding="utf-8",
    )
    hook.chmod(hook.stat().st_mode | 0o100)
    subprocess.run(
        ["git", "-C", str(root), "config", "core.fsmonitor", str(hook)],
        check=True,
    )

    snapshot = snapshot_workspaces([root])

    assert snapshot.complete is True
    assert snapshot.method in {"git", "manifest"}
    assert not marker.exists()


@pytest.mark.parametrize("index_hint", ["--assume-unchanged", "--skip-worktree"])
def test_git_snapshot_hashes_tracked_content_hidden_by_index_hints(
    tmp_path: Path,
    index_hint: str,
) -> None:
    root = tmp_path / "hinted-repo"
    _init_git_repo(root)
    source = root / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "module.py"], check=True)
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
            "tracked fixture",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "update-index", index_hint, "module.py"],
        check=True,
    )
    gate = CodingQualityGate.start([root], "hello", effort=CodingEffort.ULTRA)
    assert gate.initial_snapshot is not None
    assert gate.initial_snapshot.method == "git"
    before = gate.before_tool("bash", {})
    source.write_text("VALUE = 2\n", encoding="utf-8")

    gate.after_tool("bash", {}, before=before, audit_events=[])

    assert gate.mutation_epoch == 1
    assert gate.changed_kinds == {"code"}
    assert gate.decision().status is GateStatus.INCOMPLETE


def test_git_snapshot_detects_ignored_document_mutated_by_unsafe_tool(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ignored-repo"
    _init_git_repo(root)
    (root / ".gitignore").write_text("ignored.md\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".gitignore"], check=True)
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
            "ignore fixture",
        ],
        check=True,
    )
    gate = CodingQualityGate.start([root], "hello", effort=CodingEffort.MEDIUM)
    before = gate.before_tool("bash", {})
    (root / "ignored.md").write_text("changed outside native tools\n", encoding="utf-8")

    gate.after_tool("bash", {}, before=before, audit_events=[])

    assert gate.mutation_epoch == 1
    assert gate.changed_kinds == {"code"}
    assert gate.decision().status is GateStatus.INCOMPLETE


def test_git_snapshot_detects_ignored_build_file_mutated_by_unsafe_tool(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ignored-build-repo"
    _init_git_repo(root)
    (root / ".gitignore").write_text("BUILD\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".gitignore"], check=True)
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
            "ignore build fixture",
        ],
        check=True,
    )
    gate = CodingQualityGate.start([root], "hello", effort=CodingEffort.MEDIUM)
    before = gate.before_tool("bash", {})
    (root / "BUILD").write_text("target(name='mio')\n", encoding="utf-8")

    gate.after_tool("bash", {}, before=before, audit_events=[])

    assert gate.mutation_epoch == 1
    assert gate.changed_kinds == {"code"}
    assert gate.decision().status is GateStatus.INCOMPLETE


def test_manifest_hash_rejects_file_swapped_to_external_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    outside = tmp_path / "outside-secret.py"
    outside.write_text("SECRET = 'must-not-hash'\n", encoding="utf-8")
    resolved_root = root.resolve()
    original_open = coding_quality.os.open
    calls = 0

    def swap_before_file_open(path, flags, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.unlink()
            target.symlink_to(outside)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(coding_quality.os, "open", swap_before_file_open)

    with pytest.raises(OSError):
        coding_quality._hash_file(target, root=resolved_root, byte_budget=[0])


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


@pytest.mark.parametrize(
    "links",
    [
        {
            "docs/_theme/epub/static/note.png": "../../main/static/note.png",
            "docs/_theme/epub/static/warning.png": "../../main/static/warning.png",
        },
        {
            "lib/data/images/back-symbolic.svg": "back.svg",
            "lib/data/images/save-symbolic.svg": "save.svg",
        },
    ],
    ids=("django-relative-links", "matplotlib-relative-links"),
)
def test_manifest_snapshot_attests_swe_style_relative_file_symlinks_without_following(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    links: dict[str, str],
) -> None:
    root = tmp_path / "external-git-worktree"
    root.mkdir()
    for relative, target in links.items():
        link = root / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        target_path = link.parent / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("safe target\n", encoding="utf-8")
        link.symlink_to(target)
    monkeypatch.setattr(
        coding_quality,
        "_run_git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("external metadata unavailable")),
    )

    first = snapshot_workspaces([root])
    second = snapshot_workspaces([root])

    assert first.complete is True
    assert first.method == "manifest"
    assert len(first.symlinks) == len(links)
    assert {item.relative for item in first.symlinks} == set(links)
    assert {item.target for item in first.symlinks} == set(links.values())
    assert first.revision_sha256 == second.revision_sha256


def test_git_snapshot_attests_tracked_relative_symlink_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "git-symlink"
    root.mkdir()
    (root / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "current.py").symlink_to("target.py")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
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
            "tracked symlink",
        ],
        check=True,
    )

    def raw_git(repository, *args, probe=None):
        del probe
        completed = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            stdout=subprocess.PIPE,
        )
        return completed.stdout

    monkeypatch.setattr(coding_quality, "_prepare_git_probe", lambda _root: (("git",), {}))
    monkeypatch.setattr(coding_quality, "_run_git", raw_git)

    snapshot = snapshot_workspaces([root])

    assert snapshot.complete is True
    assert snapshot.method == "git"
    assert tuple(item.relative for item in snapshot.symlinks) == ("current.py",)
    assert snapshot.symlinks[0].target == "target.py"
    expected_state = hashlib.sha256(
        b"symlink\0target.py" + f"\0mode:{os.lstat(root / 'current.py').st_mode & 0o777}".encode()
    ).hexdigest()
    assert snapshot.symlinks[0].state_sha256 == expected_state


def test_manifest_snapshot_accepts_only_independently_attestable_directory_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "directory-links"
    (root / "real" / "nested").mkdir(parents=True)
    (root / "real" / "nested" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "alias").symlink_to("real", target_is_directory=True)
    monkeypatch.setattr(
        coding_quality,
        "_run_git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )

    snapshot = snapshot_workspaces([root])

    assert snapshot.complete is True
    assert tuple(item.relative for item in snapshot.symlinks) == ("alias",)

    (root / "alias").unlink()
    (root / ".git").mkdir()
    (root / "alias").symlink_to(".git", target_is_directory=True)
    rejected = snapshot_workspaces([root])
    assert rejected.complete is False
    assert rejected.method == "incomplete"

    variant_root = tmp_path / "case-variant-directory-link"
    (variant_root / ".GIT").mkdir(parents=True)
    (variant_root / ".GIT" / "hidden.py").write_text("TRAILING = 1  \n", encoding="utf-8")
    (variant_root / "alias").symlink_to(".GIT", target_is_directory=True)
    case_variant = snapshot_workspaces([variant_root])
    assert case_variant.complete is False


def test_manifest_snapshot_rejects_multi_link_directory_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "directory-cycle"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "a" / "to-b").symlink_to("../b", target_is_directory=True)
    (root / "b" / "to-a").symlink_to("../a", target_is_directory=True)
    monkeypatch.setattr(
        coding_quality,
        "_run_git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )

    snapshot = snapshot_workspaces([root])

    assert snapshot.complete is False
    assert snapshot.method == "incomplete"


def test_manifest_snapshot_rejects_case_variant_ancestor_directory_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "case-cycle"
    (root / "Real" / "sub").mkdir(parents=True)
    # On case-insensitive APFS this names the ancestor itself. On a
    # case-sensitive filesystem the separate directory keeps the regression
    # deterministic and the conservative case-folded rule still rejects it.
    try:
        (root / "REAL").mkdir()
    except FileExistsError:
        pass
    (root / "Real" / "sub" / "loop").symlink_to("../../REAL", target_is_directory=True)
    monkeypatch.setattr(
        coding_quality,
        "_run_git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )

    snapshot = snapshot_workspaces([root])

    assert snapshot.complete is False


def test_manifest_snapshot_deep_symlink_chain_fails_closed_without_recursion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "deep-chain"
    root.mkdir()
    (root / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    depth = coding_quality._MAX_SYMLINK_DEPTH + 2
    for index in range(depth):
        target = f"link-{index + 1:04d}" if index + 1 < depth else "target.py"
        (root / f"link-{index:04d}").symlink_to(target)
    monkeypatch.setattr(
        coding_quality,
        "_run_git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )

    snapshot = snapshot_workspaces([root])

    assert snapshot.complete is False
    assert snapshot.error_codes == ("snapshot_incomplete",)


def test_manifest_snapshot_propagates_walk_errors_as_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "walk-error"
    (root / "target").mkdir(parents=True)
    (root / "alias").symlink_to("target", target_is_directory=True)
    monkeypatch.setattr(
        coding_quality,
        "_run_git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )

    def failed_walk(*_args, onerror, **_kwargs):
        onerror(PermissionError("simulated incomplete traversal"))
        yield  # pragma: no cover

    monkeypatch.setattr(coding_quality.os, "walk", failed_walk)

    snapshot = snapshot_workspaces([root])

    assert snapshot.complete is False
    assert snapshot.method == "incomplete"


@pytest.mark.parametrize(
    "target_kind",
    ("absolute", "escape", "dangling", "cycle", "hardlink-target"),
)
def test_manifest_snapshot_rejects_unsafe_symlink_topologies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    root = tmp_path / target_kind
    root.mkdir()
    outside = tmp_path / f"outside-{target_kind}.py"
    outside.write_text("SECRET = 1\n", encoding="utf-8")
    link = root / "link.py"
    if target_kind == "absolute":
        link.symlink_to(outside)
    elif target_kind == "escape":
        link.symlink_to("../outside-escape.py")
    elif target_kind == "dangling":
        link.symlink_to("missing.py")
    elif target_kind == "cycle":
        link.symlink_to("other.py")
        (root / "other.py").symlink_to("link.py")
    else:
        hardlinked = root / "hardlinked.py"
        os.link(outside, hardlinked)
        link.symlink_to("hardlinked.py")
    monkeypatch.setattr(
        coding_quality,
        "_run_git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )

    snapshot = snapshot_workspaces([root])

    assert snapshot.complete is False
    assert snapshot.method == "incomplete"
    assert snapshot.error_codes == ("snapshot_incomplete",)


def test_symlink_lstat_readlink_lstat_race_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "race"
    root.mkdir()
    (root / "one.py").write_text("ONE = 1\n", encoding="utf-8")
    (root / "second-target.py").write_text("TWO = 2\n", encoding="utf-8")
    link = root / "current.py"
    link.symlink_to("one.py")
    monkeypatch.setattr(
        coding_quality,
        "_run_git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )
    original_readlink = coding_quality.os.readlink
    swapped = False

    def swap_during_readlink(path, *args, **kwargs):
        nonlocal swapped
        if not swapped and path == "current.py":
            swapped = True
            link.unlink()
            link.symlink_to("second-target.py")
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(coding_quality.os, "readlink", swap_during_readlink)

    snapshot = snapshot_workspaces([root])

    assert swapped is True
    assert snapshot.complete is False


def test_only_unchanged_baseline_symlinks_receive_hygiene_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "baseline"
    root.mkdir()
    (root / "one.py").write_text("ONE = 1\n", encoding="utf-8")
    (root / "two.py").write_text("TWO = 2\n", encoding="utf-8")
    link = root / "current.py"
    link.symlink_to("one.py")
    monkeypatch.setattr(
        coding_quality,
        "_run_git",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )
    gate = CodingQualityGate.start([root], "change", effort=CodingEffort.XHIGH)

    assert tuple(item.relative for item in gate.trusted_unchanged_symlinks()) == ("current.py",)

    link.unlink()
    link.symlink_to("two.py")
    gate.refresh()

    assert gate.trusted_unchanged_symlinks() == ()


def test_only_unchanged_baseline_regular_files_receive_hygiene_authority(
    tmp_path: Path,
) -> None:
    unchanged = tmp_path / "unchanged.py"
    changed = tmp_path / "changed.py"
    unchanged.write_text("UNCHANGED = 1\n", encoding="utf-8")
    changed.write_text("CHANGED = 1\n", encoding="utf-8")
    gate = CodingQualityGate.start([tmp_path], "change", effort=CodingEffort.XHIGH)
    initial_trust = gate.trusted_unchanged_regular_path_hashes()

    assert coding_quality._revision_path_sha256(0, "unchanged.py") in initial_trust
    assert coding_quality._revision_path_sha256(0, "changed.py") in initial_trust

    changed.write_text("CHANGED = 2\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("NEW = 1\n", encoding="utf-8")
    gate.refresh()
    current_trust = gate.trusted_unchanged_regular_path_hashes()

    assert coding_quality._revision_path_sha256(0, "unchanged.py") in current_trust
    assert coding_quality._revision_path_sha256(0, "changed.py") not in current_trust
    assert coding_quality._revision_path_sha256(0, "new.py") not in current_trust


def test_trusted_builtin_read_can_defer_snapshot_but_unknown_or_unaudited_cannot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    gate = CodingQualityGate.start([tmp_path], "inspect", effort=CodingEffort.MEDIUM)
    current = gate.current_snapshot
    assert current is not None
    snapshots = 0

    def counted_snapshot(_roots):
        nonlocal snapshots
        snapshots += 1
        return current

    monkeypatch.setattr(coding_quality, "snapshot_workspaces", counted_snapshot)
    read_event = AgentAuditEvent(
        timestamp=1.0,
        operation="read",
        permission="read",
        target=str(target),
        allowed=True,
        outcome="ok",
    )

    gate.after_tool(
        "read",
        {"path": "module.py"},
        before=current,
        audit_events=(read_event,),
        trusted_non_mutating=True,
    )
    assert snapshots == 0
    assert gate.successful_reads == 1

    gate.after_tool("read", {"path": "module.py"}, before=current, audit_events=())
    gate.after_tool("unknown", {}, before=current, audit_events=())
    assert snapshots == 2


def test_quality_report_exposes_only_closed_snapshot_attestation_fields(tmp_path: Path) -> None:
    gate = CodingQualityGate.start([tmp_path], "inspect", effort=CodingEffort.LOW)

    report = gate.report()

    assert report["snapshot_method"] in {"git", "manifest"}
    assert report["snapshot_error_codes"] == []

    incomplete = coding_quality.WorkspaceSnapshot(
        revision_sha256="a" * 64,
        entries=(),
        complete=False,
        root_count=1,
        method="incomplete",
        error_codes=("snapshot_incomplete",),
    )
    failed = CodingQualityGate(
        roots=(tmp_path,),
        initial_snapshot=incomplete,
        current_snapshot=incomplete,
    ).report()
    assert failed["snapshot_method"] == "incomplete"
    assert failed["snapshot_error_codes"] == ["snapshot_incomplete"]


def test_feedback_signature_changes_with_revision_epoch_phase_and_obligation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    gate = CodingQualityGate.start([tmp_path], "change", effort=CodingEffort.HIGH)
    observing = gate.feedback_signature()
    _record_edit(gate, target, "VALUE = 2\n")
    dirty = gate.feedback_signature()
    assert dirty != observing
    assert dirty == gate.feedback_signature()

    gate.record_validation(
        ValidationKind.TEST,
        argv=("pytest", "-q"),
        allowed=True,
        outcome="nonzero",
    )
    failed = gate.feedback_signature()
    assert failed != dirty

    _record_edit(gate, target, "VALUE = 3\n")
    next_epoch = gate.feedback_signature()
    assert next_epoch not in {observing, dirty, failed}


def test_requested_change_without_a_workspace_diff_is_not_certified(tmp_path: Path) -> None:
    gate = CodingQualityGate.start(
        [tmp_path],
        "Implement a parser change, then stop without editing.",
        effort=CodingEffort.LOW,
    )

    decision = gate.decision()

    assert gate.mutation_epoch == 0
    assert decision.status is GateStatus.INCOMPLETE
    assert decision.activated is False
    assert decision.phase == "awaiting_change"
    assert decision.required == ("net_workspace_change",)
    assert decision.missing == ("net_workspace_change",)
    assert gate.should_persist() is False


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
    assert gate.decision().phase == "no_net_change"
    assert gate.decision().missing == ("net_workspace_change",)


def test_edit_then_revert_cannot_pass_with_stale_validation(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    gate = CodingQualityGate.start(
        [tmp_path],
        "Change module.py.",
        effort=CodingEffort.LOW,
        require_net_workspace_change=True,
    )

    _record_edit(gate, target, "VALUE = 2\n")
    _record_validation(gate, ValidationKind.TEST, ("pytest", "-q"))
    _record_edit(gate, target, "VALUE = 1\n")
    _record_validation(gate, ValidationKind.TEST, ("pytest", "-q"))

    decision = gate.decision()
    assert gate.mutation_epoch == 2
    assert gate.net_workspace_changed is False
    assert decision.status is GateStatus.INCOMPLETE
    assert decision.phase == "no_net_change"
    assert decision.missing == ("net_workspace_change",)
    assert gate.should_persist() is False


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python3 -m pytest -q", ValidationKind.TEST),
        ("git diff --check", ValidationKind.DIFF),
        ("python3 -m pytest -q || true", None),
        ("python3 -m pytest -q | tee result.txt", None),
        ("python3 -m pytest -q > result.txt", None),
        ("python3 -m pytest 'tests/test one.py'", None),
        ("$(which pytest) -q", None),
    ],
)
def test_misrouted_validation_classifier_is_narrow_and_never_evidence(
    command: str,
    expected: ValidationKind | None,
) -> None:
    assert infer_misrouted_validation_command(command) is expected


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


def test_terminal_refresh_invalidates_late_workspace_change(tmp_path: Path) -> None:
    root = tmp_path / "late-change"
    root.mkdir()
    target = root / "module.py"
    gate = CodingQualityGate.start([root], "hello", effort=CodingEffort.MEDIUM)
    _record_edit(gate, target, "VALUE = 1\n")
    _record_validation(gate, ValidationKind.TEST, ("pytest", "-q"))
    assert gate.decision().status is GateStatus.PASS

    target.write_text("VALUE = 2\n", encoding="utf-8")
    gate.refresh()

    assert gate.mutation_epoch == 2
    assert gate.changed_kinds == {"code"}
    assert gate.decision().status is GateStatus.INCOMPLETE


def test_before_unsafe_tool_reconciles_background_code_change(tmp_path: Path) -> None:
    root = tmp_path / "background-change"
    root.mkdir()
    readme = root / "README.md"
    gate = CodingQualityGate.start([root], "hello", effort=CodingEffort.MEDIUM)
    _record_edit(gate, readme, "docs\n")
    _record_validation(gate, ValidationKind.DIFF, ("git", "diff", "--check"))
    assert gate.changed_kinds == {"docs"}
    assert gate.decision().status is GateStatus.PASS

    (root / "requirements.txt").write_text("package==1\n", encoding="utf-8")
    gate.before_tool("validate", {"argv": ["pytest", "-q"]})

    assert gate.mutation_epoch == 2
    assert gate.changed_kinds == {"code", "docs"}
    assert gate.decision().status is GateStatus.INCOMPLETE
    assert gate.decision().missing == ("test_or_build",)


def test_unsafe_tool_activates_fail_closed_gate_when_snapshots_stay_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = coding_quality.WorkspaceSnapshot(
        revision_sha256="a" * 64,
        entries=(),
        complete=False,
        root_count=1,
        method="incomplete",
        error_codes=("snapshot_incomplete",),
    )
    gate = CodingQualityGate(
        roots=(tmp_path,),
        effort=CodingEffort.MEDIUM,
        initial_snapshot=incomplete,
        current_snapshot=incomplete,
    )
    monkeypatch.setattr(coding_quality, "snapshot_workspaces", lambda _roots: incomplete)

    gate.after_tool("bash", {}, before=incomplete, audit_events=[])

    decision = gate.decision()
    assert gate.mutation_epoch == 1
    assert decision.status is GateStatus.INCOMPLETE
    assert "complete_workspace_snapshot" in decision.missing


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


@pytest.mark.parametrize("name", ["requirements.txt", "CMakeLists.txt"])
def test_source_named_txt_files_never_receive_docs_relaxation(
    tmp_path: Path,
    name: str,
) -> None:
    root = tmp_path / "source-name"
    root.mkdir()
    gate = CodingQualityGate.start([root], "hello", effort=CodingEffort.MEDIUM)
    _record_edit(gate, root / name, "changed\n")
    _record_validation(gate, ValidationKind.DIFF, ("git", "diff", "--check"))

    assert gate.changed_kinds == {"code"}
    assert gate.decision().status is GateStatus.INCOMPLETE
    assert gate.decision().missing == ("test_or_build",)


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

    _record_validation(gate, ValidationKind.TEST, ("pytest", "-qq"))
    assert gate.decision().missing == ("review_or_second_distinct_test",)

    _record_validation(
        gate,
        ValidationKind.TEST,
        ("python3", "-B", "-m", "pytest", "--tb=long", "-q"),
    )
    assert gate.decision().missing == ("review_or_second_distinct_test",)

    _record_validation(gate, ValidationKind.TEST, ("pytest", "-q", "."))
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
