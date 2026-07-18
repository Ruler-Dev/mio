"""Certify Mio's verifier against all pinned HumanEval reference solutions.

The JSON report contains only identifiers, hashes, statuses, and timings.  It
never serializes hidden tests, canonical source, verifier output, or local
filesystem paths.  A successful process exit is deliberately stricter than a
passing corpus: the repository must remain clean at the same Git revision for
the whole run so provenance identifies the exact implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from experimental.effort.humaneval import (
    HUMANEVAL_REVISION,
    HUMANEVAL_SHA256,
    HUMANEVAL_URL,
    HumanEvalReference,
    VerificationResult,
    corpus_manifest,
    default_corpus_path,
    fetch_humaneval,
    load_humaneval_references,
    reference_manifest,
    verify_candidate,
)


REPORT_SCHEMA = "mio.humaneval-verifier-parity.v2"
EXPECTED_HUMANEVAL_TASKS = 164
DEFAULT_TIMEOUT_SECONDS = 10.0
PRIMARY_VERIFIER_SOURCE = "experimental/effort/humaneval.py"
VERIFIER_SOURCE_PATHS = (
    PRIMARY_VERIFIER_SOURCE,
    "experimental/markov_effort_controller.py",
    "mio/agent.py",
    "mio/agent_policy.py",
)


@dataclass(frozen=True)
class GitState:
    revision: str
    dirty: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_git(arguments: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed executable and arguments
        ["git", "-C", os.fspath(cwd), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )


def inspect_git_state(repository_hint: Path | None = None) -> GitState:
    """Return fail-closed Git provenance without exposing local paths."""

    hint = (repository_hint or Path(__file__).resolve().parent).resolve()
    try:
        revision_result = _run_git(["rev-parse", "HEAD"], cwd=hint)
        status_result = _run_git(
            ["status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=hint,
        )
    except (OSError, subprocess.TimeoutExpired):
        return GitState(revision="unknown", dirty=True)
    revision = revision_result.stdout.strip()
    if (
        revision_result.returncode != 0
        or len(revision) not in {40, 64}
        or any(character not in "0123456789abcdefABCDEF" for character in revision)
    ):
        return GitState(revision="unknown", dirty=True)
    dirty = status_result.returncode != 0 or bool(status_result.stdout)
    return GitState(revision=revision, dirty=dirty)


def git_repository_root(repository_hint: Path | None = None) -> Path | None:
    """Resolve the containing worktree root, or fail closed with ``None``."""

    hint = (repository_hint or Path(__file__).resolve().parent).resolve()
    try:
        result = _run_git(["rev-parse", "--show-toplevel"], cwd=hint)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        root = Path(result.stdout.strip()).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return root if root.is_dir() else None


def output_is_inside_repository(output: Path, repository_root: Path) -> bool:
    """Return whether an output path resolves inside the source worktree."""

    candidate = output.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve(strict=False)
    root = repository_root.expanduser().resolve(strict=True)
    return candidate == root or root in candidate.parents


def verifier_source_sha256() -> str:
    """Hash the complete source module that implements ``verify_candidate``."""

    source_path_text = inspect.getsourcefile(verify_candidate)
    if source_path_text is None:
        raise RuntimeError("cannot locate verifier source")
    return _sha256(Path(source_path_text).read_bytes())


def verifier_source_hashes() -> dict[str, str]:
    """Hash the verifier and every repository module it delegates to."""

    repository_root = Path(__file__).resolve().parents[2]
    return {
        relative_path: _sha256((repository_root / relative_path).read_bytes())
        for relative_path in VERIFIER_SOURCE_PATHS
    }


def _source_bundle_sha256(source_files: Mapping[str, str]) -> str:
    canonical = json.dumps(
        dict(sorted(source_files.items())),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(canonical)


def _invoke_verifier(
    verifier: Callable[..., VerificationResult],
    reference: HumanEvalReference,
    *,
    timeout_s: float,
) -> VerificationResult:
    return verifier(
        reference.case,
        reference.canonical_solution,
        timeout_s=timeout_s,
    )


def build_parity_report(
    references: tuple[HumanEvalReference, ...],
    *,
    timeout_s: float,
    git_state: GitState,
    verifier_sha256: str,
    verifier_source_files: Mapping[str, str] | None = None,
    verifier: Callable[..., VerificationResult] | None = None,
    expected_tasks: int = EXPECTED_HUMANEVAL_TASKS,
    progress: Callable[[int, int, str, VerificationResult], None] | None = None,
    post_verification_git_probe: Callable[[], GitState] | None = None,
) -> dict[str, object]:
    """Run every reference and return a source-free parity report."""

    if not 0.1 <= timeout_s <= 30.0:
        raise ValueError("timeout_s must be between 0.1 and 30 seconds")
    if expected_tasks <= 0:
        raise ValueError("expected_tasks must be positive")
    source_files = dict(
        verifier_source_files
        or {PRIMARY_VERIFIER_SOURCE: verifier_sha256}
    )
    if source_files.get(PRIMARY_VERIFIER_SOURCE) != verifier_sha256:
        raise ValueError("primary verifier hash does not match the source bundle")
    if any(
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for path, digest in source_files.items()
    ):
        raise ValueError("verifier source bundle is malformed")
    active_verifier = verifier or verify_candidate
    tasks: list[dict[str, object]] = []
    passed = 0
    for index, reference in enumerate(references, start=1):
        result = _invoke_verifier(
            active_verifier,
            reference,
            timeout_s=timeout_s,
        )
        task_passed = result.passed is True and result.status == "passed"
        passed += int(task_passed)
        tasks.append(
            {
                "task_id": reference.case.task_id,
                "status": result.status,
                "passed": task_passed,
                "elapsed_seconds": round(result.elapsed_seconds, 9),
                "canonical_solution_sha256": reference.canonical_solution_sha256,
                "verified_source_sha256": result.source_sha256,
            }
        )
        if progress is not None:
            progress(index, len(references), reference.case.task_id, result)

    final_git_state = post_verification_git_probe() if post_verification_git_probe is not None else git_state
    revision_stable = final_git_state.revision == git_state.revision
    exact_corpus = len(references) == expected_tasks
    parity = exact_corpus and passed == expected_tasks
    reasons: list[str] = []
    if not exact_corpus:
        reasons.append(f"expected_{expected_tasks}_tasks_but_loaded_{len(references)}")
    if passed != expected_tasks:
        reasons.append(f"expected_{expected_tasks}_passes_but_observed_{passed}")
    if git_state.dirty:
        reasons.append("git_tree_dirty_before_verification")
    if final_git_state.dirty and not git_state.dirty:
        reasons.append("git_tree_dirty_after_verification")
    if not revision_stable:
        reasons.append("git_revision_changed_during_verification")
    claim_eligible = parity and not git_state.dirty and not final_git_state.dirty and revision_stable

    cases = tuple(reference.case for reference in references)
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "claim": {
            "eligible": claim_eligible,
            "parity": parity,
            "passed": passed,
            "total": len(references),
            "expected": expected_tasks,
            "ineligibility_reasons": reasons,
        },
        "corpus": {
            "name": "openai/human-eval",
            "revision": HUMANEVAL_REVISION,
            "archive_sha256": HUMANEVAL_SHA256,
            "source_url": HUMANEVAL_URL,
            "manifest": corpus_manifest(cases),
            "reference_manifest": reference_manifest(references),
        },
        "verifier": {
            "callable": "experimental.effort.humaneval.verify_candidate",
            "source_path": PRIMARY_VERIFIER_SOURCE,
            "source_sha256": verifier_sha256,
            "source_hash_scope": "complete_module_files",
            "source_files": dict(sorted(source_files.items())),
            "source_bundle_sha256": _source_bundle_sha256(source_files),
            "timeout_seconds_per_task": timeout_s,
        },
        "git": {
            "revision": git_state.revision,
            "dirty_before_verification": git_state.dirty,
            "revision_after_verification": final_git_state.revision,
            "dirty_after_verification": final_git_state.dirty,
            "revision_stable": revision_stable,
        },
        "tasks": tasks,
    }


def write_json_atomic(path: Path, report: dict[str, object]) -> None:
    """Publish one complete JSON report with an atomic same-directory rename."""

    destination = path.expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    destination = destination.absolute()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _progress(
    index: int,
    total: int,
    task_id: str,
    result: VerificationResult,
) -> None:
    print(
        f"[{index:03d}/{total:03d}] {task_id}: {result.status} ({result.elapsed_seconds:.3f}s)",
        file=sys.stderr,
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all 164 SHA-pinned HumanEval canonical solutions and write a source-free JSON provenance report."
        )
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=default_corpus_path(),
        help="cache path for the pinned official HumanEval archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination JSON path outside the Git worktree (published atomically)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-task verifier timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-task progress on stderr",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not 0.1 <= arguments.timeout <= 30.0:
        print("error: --timeout must be between 0.1 and 30 seconds", file=sys.stderr)
        return 2

    repository_root = git_repository_root()
    if repository_root is not None and output_is_inside_repository(
        arguments.output,
        repository_root,
    ):
        print(
            "error: --output must be outside the Git worktree so the final clean-tree attestation remains true",
            file=sys.stderr,
        )
        return 2

    # The report is required to live outside the worktree, so the post-run Git
    # probe remains true through atomic publication and successful process exit.
    git_state = inspect_git_state()
    corpus_path = fetch_humaneval(arguments.corpus)
    references = load_humaneval_references(corpus_path)
    source_files = verifier_source_hashes()
    report = build_parity_report(
        references,
        timeout_s=arguments.timeout,
        git_state=git_state,
        verifier_sha256=source_files[PRIMARY_VERIFIER_SOURCE],
        verifier_source_files=source_files,
        progress=None if arguments.quiet else _progress,
        post_verification_git_probe=inspect_git_state,
    )
    write_json_atomic(arguments.output, report)
    claim = report["claim"]
    if not isinstance(claim, dict) or claim.get("eligible") is not True:
        reasons = claim.get("ineligibility_reasons", []) if isinstance(claim, dict) else []
        print(
            "parity claim rejected: " + ", ".join(str(reason) for reason in reasons),
            file=sys.stderr,
        )
        return 1
    print(
        f"certified {EXPECTED_HUMANEVAL_TASKS}/{EXPECTED_HUMANEVAL_TASKS}; report: {arguments.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
