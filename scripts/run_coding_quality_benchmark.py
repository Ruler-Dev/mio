#!/usr/bin/env python3
"""Run MioCodeBench v1 smoke/development arms with the native Mio agent.

This module contains only the non-confirmatory 4-task smoke and 8-task
development corpus.  Public files are materialized by ``bench_coding_quality``;
private evaluators stay in the host process and are never copied under an agent
workspace root.  The command writes only the source-free aggregate schema.

The MLX stack is imported lazily.  Unit tests can inject a callback runner and
exercise every protocol boundary without loading a model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

# ``python scripts/run_coding_quality_benchmark.py`` sets ``sys.path[0]`` to
# ``scripts/`` rather than the repository root.  Bootstrap the checkout before
# importing Mio-local packages so the executable form and ``python -m`` resolve
# the exact same source tree.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from experimental.effort.model_identity import fingerprint_local_model  # noqa: E402

from scripts.bench_coding_quality import (  # noqa: E402
    GATE_OFF,
    GATE_ON,
    BenchmarkExecution,
    CodingFixture,
    EvaluationRequest,
    GenerationObservation,
    GenerationRequest,
    HiddenEvaluation,
    Preregistration,
    PublicFile,
    SourceFreeAggregate,
    fixture_suite_sha256,
    run_benchmark,
    serialize_source_free_aggregate,
)


_PUBLIC_TOOL_NAMES = ("bash", "read", "write", "edit")
_GATE_TOOL_NAMES = (*_PUBLIC_TOOL_NAMES, "validate")

RESULT_ENVELOPE_SCHEMA = "mio.coding-quality-result-envelope.v1"
SOURCE_LOCK_SCHEMA = "mio.coding-quality-source-lock.v1"
TARGET_REPOSITORY_LABEL = "mlx-community/Qwen3.5-4B-4bit"
TARGET_CONTENT_IDENTITY = (
    "local-sha256-v1:7d7ea69d09ada4f1d2f49f6ca651441ac279b95b6d280f259da04fbde504376f"
)
DRAFT_REPOSITORY_LABEL = "z-lab/Qwen3.5-4B-DFlash"
DRAFT_CONTENT_IDENTITY = (
    "local-sha256-v1:4b60bced36f602da85a6447d3648e7ac37a0c5cce68d505a49664252c0586b98"
)
FROZEN_CONTEXT_WINDOW = 8192
FROZEN_MAX_OUTPUT_TOKENS = 2048
FROZEN_SEED = 20260718
FROZEN_BOOTSTRAP_SAMPLES = 10_000
FROZEN_ALPHA = 0.05
FROZEN_MINIMUM_PAIRS_FOR_CLAIM = 16
FROZEN_EVALUATOR_TIMEOUT_S = 5.0
FROZEN_SOFTWARE_VERSIONS = (
    ("dflash-mlx", "0.1.8"),
    ("huggingface-hub", "1.24.0"),
    ("mlx", "0.32.0"),
    ("mlx-audio", "0.4.4"),
    ("mlx-dspark", "0.5.0"),
    ("mlx-lm", "0.31.3"),
    ("mlx-vlm", "0.6.5"),
    ("transformers", "5.14.1"),
)
GATE_PROFILE_SCHEMA = "mio.coding-quality-effort-profiles.v1"
_GATE_PROFILE_MANIFEST = {
    "schema": GATE_PROFILE_SCHEMA,
    "profiles": {
        "low": {"code": ["any_validation"], "docs": ["any_validation"]},
        "medium": {"code": ["test_or_build"], "docs": ["diff"]},
        "high": {"code": ["test", "static_or_diff"], "docs": ["test", "static_or_diff"]},
        "xhigh": {"code": ["test", "static", "diff"], "docs": ["test", "static", "diff"]},
        "ultra": {
            "code": ["test", "static", "diff", "review_or_second_distinct_test"],
            "docs": ["test", "static", "diff", "review_or_second_distinct_test"],
        },
    },
}
_COMPUTED_GATE_PROFILE_SHA256 = hashlib.sha256(
    json.dumps(
        _GATE_PROFILE_MANIFEST,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
GATE_PROFILE_SHA256 = "f522ac26d7b49c55e0c048e119e42802ff4b35da7223bf12b1f5e200fbb5208b"

# These are every behavior- or timing-affecting environment override consulted
# by the native DFlash path.  A scientific run uses the checked-in defaults;
# accepting even a seemingly equivalent override would make provenance depend
# on unreported caller state.
_FROZEN_ENVIRONMENT_VARIABLES = (
    "DDTREE_EXACT_COMMIT",
    "DFLASH_DRAFT_SINK",
    "DFLASH_DRAFT_WINDOW",
    "DFLASH_MAX_CTX",
    "DFLASH_QUANTIZE_DRAFT",
    "DFLASH_VERIFY_LEN",
    "MIO_DDTREE_BUDGET",
    "MIO_DEBUG_LOG",
    "MIO_DEBUG_LOG_PATH",
    "MIO_DFLASH_EXACT_COMMIT_ORACLE",
    "MIO_DFLASH_EXACT_COMPONENTS",
    "MIO_DFLASH_QMV_STAGING",
    "MIO_DFLASH_QMV_VECTORS",
    "MIO_PREFILL_CHUNK",
)
_SOURCE_LOCK_FILES = (
    "pyproject.toml",
    "mio/agent.py",
    "mio/agent_policy.py",
    "mio/coding_quality.py",
    "mio/config.py",
    "mio/model_manager.py",
    "mio/prompt_policy.py",
    "scripts/bench_coding_quality.py",
    "scripts/run_coding_quality_benchmark.py",
)
_SENSITIVE_ARTIFACT_KEYS = frozenset(
    {
        "assistant_text",
        "completion",
        "content",
        "draft_path",
        "fixture_id",
        "hidden_checks",
        "hidden_labels",
        "instruction",
        "model_path",
        "oracle",
        "path",
        "prompt",
        "public_files",
        "public_regression",
        "record",
        "records",
        "request",
        "response",
        "target_path",
        "tool_output",
        "workspace",
    }
)
_FULL_GIT_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SAFE_PUBLIC_LABEL = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_SAFE_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z._+!-]{0,127}\Z")
_SAFE_HARDWARE_LABEL = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class CleanSourceLock:
    """Content-bound clean implementation state retained only in memory."""

    repo_root: Path
    git_revision: str
    source_sha256: str
    source_file_count: int


@dataclass(frozen=True)
class LocalModelLock:
    """One exact local model snapshot; its absolute path is never public."""

    role: str
    repository_label: str
    content_identity: str
    resolved_path: Path


@dataclass(frozen=True)
class RuntimeIdentity:
    """Source-free identity of the interpreter, MLX stack, and host class."""

    python_version: str
    software_versions: tuple[tuple[str, str], ...]
    hardware_label: str

    def __post_init__(self) -> None:
        if not _SAFE_VERSION.fullmatch(self.python_version):
            raise ValueError("runtime identity contains an unsafe or unavailable python_version")
        if self.software_versions != FROZEN_SOFTWARE_VERSIONS:
            raise ValueError("runtime identity does not match the frozen software lock")
        if any(
            not _SAFE_PUBLIC_LABEL.fullmatch(name) or not _SAFE_VERSION.fullmatch(version)
            for name, version in self.software_versions
        ):
            raise ValueError("runtime identity contains an unsafe software version")
        if not _SAFE_HARDWARE_LABEL.fullmatch(self.hardware_label):
            raise ValueError("runtime identity contains an unsafe or unavailable hardware label")


@dataclass(frozen=True)
class BenchmarkResultEnvelope:
    """Fixed-schema, source-free result approved for public serialization."""

    source_lock: CleanSourceLock
    model_locks: tuple[LocalModelLock, LocalModelLock]
    runtime_identity: RuntimeIdentity
    aggregate: SourceFreeAggregate
    split: str
    tier: str
    effort: str
    protocol_sha256: str
    gate_profile_sha256: str = GATE_PROFILE_SHA256
    post_run_verified: bool = True

    def __post_init__(self) -> None:
        expected_models = {
            "target": (TARGET_REPOSITORY_LABEL, TARGET_CONTENT_IDENTITY),
            "drafter": (DRAFT_REPOSITORY_LABEL, DRAFT_CONTENT_IDENTITY),
        }
        observed_models = {
            lock.role: (lock.repository_label, lock.content_identity) for lock in self.model_locks
        }
        if observed_models != expected_models or len(self.model_locks) != len(expected_models):
            raise ValueError("result envelope model identities do not match the frozen protocol")
        if not self.post_run_verified:
            raise ValueError("result envelope requires successful post-run verification")
        if not _FULL_GIT_REVISION.fullmatch(self.source_lock.git_revision):
            raise ValueError("result envelope requires a full lowercase Git revision")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_lock.source_sha256):
            raise ValueError("result envelope requires a lowercase source SHA-256")
        if self.source_lock.source_file_count != len(_SOURCE_LOCK_FILES):
            raise ValueError("result envelope source scope does not match the frozen protocol")
        expected_splits = {
            "smoke": (SMOKE_SUITE_SHA256, SMOKE_PROTOCOL_SHA256, 4),
            "development": (DEVELOPMENT_SUITE_SHA256, DEVELOPMENT_PROTOCOL_SHA256, 8),
            "all": (ALL_SUITE_SHA256, ALL_PROTOCOL_SHA256, 12),
        }
        if self.split not in expected_splits:
            raise ValueError("result envelope split is not supported by this runner")
        expected_suite, expected_protocol, expected_pairs = expected_splits[self.split]
        if (
            self.aggregate.suite_sha256 != expected_suite
            or self.protocol_sha256 != expected_protocol
            or self.aggregate.pair_count != expected_pairs
            or self.aggregate.seed != FROZEN_SEED
            or self.aggregate.bootstrap_samples != FROZEN_BOOTSTRAP_SAMPLES
            or self.aggregate.alpha != FROZEN_ALPHA
        ):
            raise ValueError("result envelope aggregate does not match its frozen split protocol")
        if self.effort not in {"low", "medium", "high", "xhigh", "ultra"}:
            raise ValueError("result envelope effort is not supported")
        if self.gate_profile_sha256 != GATE_PROFILE_SHA256:
            raise ValueError("result envelope gate profile does not match the frozen protocol")
        _assert_gate_profile_seal()
        if not _SAFE_PUBLIC_LABEL.fullmatch(self.tier):
            raise ValueError("result envelope tier must be a source-free public label")
        if not isinstance(self.runtime_identity, RuntimeIdentity):
            raise ValueError("result envelope requires a verified runtime identity")
        self.runtime_identity.__post_init__()

    def to_dict(self) -> dict[str, object]:
        models = {lock.role: lock for lock in self.model_locks}
        return {
            "schema_version": RESULT_ENVELOPE_SCHEMA,
            "implementation": {
                "source_lock_schema": SOURCE_LOCK_SCHEMA,
                "git_revision": self.source_lock.git_revision,
                "git_clean": True,
                "source_sha256": self.source_lock.source_sha256,
                "source_file_count": self.source_lock.source_file_count,
                "post_run_source_stable": True,
            },
            "models": {
                "target": {
                    "repository_label": models["target"].repository_label,
                    "content_identity": models["target"].content_identity,
                },
                "drafter": {
                    "backend": "dflash",
                    "repository_label": models["drafter"].repository_label,
                    "content_identity": models["drafter"].content_identity,
                },
                "post_run_identities_stable": True,
            },
            "protocol": {
                "protocol_sha256": self.protocol_sha256,
                "gate_profile_schema": GATE_PROFILE_SCHEMA,
                "gate_profile_sha256": self.gate_profile_sha256,
            },
            "software": {
                "python_version": self.runtime_identity.python_version,
                "packages": dict(self.runtime_identity.software_versions),
            },
            "hardware": {"label": self.runtime_identity.hardware_label},
            "runtime": {
                "split": self.split,
                "tier": self.tier,
                "effort": self.effort,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "context_window": FROZEN_CONTEXT_WINDOW,
                "max_output_tokens": FROZEN_MAX_OUTPUT_TOKENS,
                "tq_bits": 16,
                "pq_bits": 16,
                "bmp_paths": 1,
                "ddtree_budget": 0,
                "drafter_backend": "dflash",
                "dflash_quantize_draft": False,
                "dflash_verify_len_override": False,
                "dflash_max_context": 131072,
                "dflash_draft_sink": 64,
                "dflash_draft_window": 1024,
                "dflash_exact_commit_oracle": False,
                "dflash_exact_components": "gdn,attention,mlp,head",
                "dflash_qmv_vectors": "auto",
                "dflash_qmv_staging": False,
                "prefill_chunk": 2048,
                "environment_overrides": False,
                "cold_arm_state": True,
                "network_enabled": False,
            },
            "aggregate": json.loads(serialize_source_free_aggregate(self.aggregate)),
            "hidden_labels_serialized": False,
        }


GitProbe = Callable[[Path, tuple[str, ...]], str]
ModelFingerprint = Callable[[Path], Any]


def _assert_frozen_environment(environ: Mapping[str, str] | None = None) -> None:
    """Reject caller state that can alter the frozen DFlash execution path."""

    active = os.environ if environ is None else environ
    # Presence itself is an override: for example an explicitly empty
    # MIO_DFLASH_EXACT_COMPONENTS disables the non-empty checked-in default.
    overridden = sorted(name for name in _FROZEN_ENVIRONMENT_VARIABLES if name in active)
    if overridden:
        raise RuntimeError(
            "MioCodeBench requires frozen DFlash environment defaults; unset: "
            + ", ".join(overridden)
        )


def _assert_gate_profile_seal() -> None:
    from mio.coding_quality import CodingQualityGate, WorkspaceSnapshot

    observed_profiles: dict[str, dict[str, list[str]]] = {}
    synthetic_snapshot = WorkspaceSnapshot(
        revision_sha256="0" * 64,
        entries=(),
        complete=True,
        root_count=1,
        method="protocol_contract",
    )
    for effort in ("low", "medium", "high", "xhigh", "ultra"):
        observed_profiles[effort] = {}
        for change_kind in ("code", "docs"):
            gate = CodingQualityGate(
                roots=(_REPOSITORY_ROOT,),
                effort=effort,
                initial_snapshot=synthetic_snapshot,
                current_snapshot=synthetic_snapshot,
            )
            gate.mutation_epoch = 1
            gate.changed_kinds = {change_kind}
            required, _missing = gate._requirements()
            observed_profiles[effort][change_kind] = list(required)
    observed_manifest = {
        "schema": GATE_PROFILE_SCHEMA,
        "profiles": observed_profiles,
    }
    observed_sha256 = hashlib.sha256(
        json.dumps(
            observed_manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        _COMPUTED_GATE_PROFILE_SHA256 != GATE_PROFILE_SHA256
        or observed_sha256 != GATE_PROFILE_SHA256
    ):
        raise RuntimeError("coding-quality effort profiles no longer match their explicit seal")


def _sysctl_value(name: str) -> str:
    executable = Path("/usr/sbin/sysctl")
    if not executable.is_file():
        raise RuntimeError("hardware identity is unavailable")
    completed = subprocess.run(
        [os.fspath(executable), "-n", name],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        raise RuntimeError("hardware identity is unavailable")
    return value


def _physical_memory_bytes() -> int:
    if platform.system() == "Darwin":
        try:
            value = int(_sysctl_value("hw.memsize"))
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("hardware memory identity is unavailable") from exc
    else:
        try:
            value = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise RuntimeError("hardware memory identity is unavailable") from exc
    if value <= 0:
        raise RuntimeError("hardware memory identity is unavailable")
    return value


def _hardware_model() -> str:
    if platform.system() == "Darwin":
        return _sysctl_value("hw.model")
    value = platform.processor().strip() or platform.machine().strip()
    if not value:
        raise RuntimeError("hardware model identity is unavailable")
    return value


def _label_segment(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", value.casefold()).strip("-._")
    if not normalized:
        raise RuntimeError("hardware identity contains no safe public label")
    return normalized


def collect_runtime_identity() -> RuntimeIdentity:
    """Collect a path-free identity for the effective native MLX runtime."""

    try:
        software_versions = tuple(
            (name, importlib.metadata.version(name))
            for name, _expected in FROZEN_SOFTWARE_VERSIONS
        )
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("the installed MLX runtime identity is unavailable") from exc
    if software_versions != FROZEN_SOFTWARE_VERSIONS:
        raise RuntimeError("the installed MLX stack does not match the frozen software lock")
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        raise RuntimeError("hardware CPU identity is unavailable")
    hardware_label = "-".join(
        (
            _label_segment(platform.system()),
            _label_segment(platform.machine()),
            _label_segment(_hardware_model()),
            f"{cpu_count}cpu",
            f"{_physical_memory_bytes()}b",
        )
    )
    return RuntimeIdentity(
        python_version=platform.python_version(),
        software_versions=software_versions,
        hardware_label=hardware_label,
    )


def verify_runtime_identity(expected: RuntimeIdentity) -> None:
    """Fail if software or hardware identity changed during execution."""

    if collect_runtime_identity() != expected:
        raise RuntimeError("benchmark runtime identity drifted during execution")


def _path_is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_output_path(
    output: Path | None,
    *,
    source_root: Path,
    model_locks: Sequence[LocalModelLock],
) -> Path | None:
    """Reject result destinations that could invalidate certified inputs."""

    if output is None:
        return None
    candidate = Path(output).expanduser()
    if candidate.is_symlink():
        raise RuntimeError("benchmark output must not be a symlink")
    resolved = candidate.resolve(strict=False)
    protected_roots = (
        Path(source_root).resolve(strict=True),
        *(lock.resolved_path.resolve(strict=True) for lock in model_locks),
    )
    if any(_path_is_within(resolved, root) for root in protected_roots):
        raise RuntimeError("benchmark output must stay outside source and model roots")
    if candidate.exists() and not candidate.is_file():
        raise RuntimeError("benchmark output must be a regular file destination")
    return resolved


def _atomic_write_result(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _git_probe(repo_root: Path, arguments: tuple[str, ...]) -> str:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repo_root), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot establish benchmark Git provenance")
    return completed.stdout


def _source_tree_sha256(repo_root: Path, source_files: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative_name in sorted(source_files):
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("source lock contains a non-relative file name")
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("source lock input is not a regular file")
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity:
            raise RuntimeError("source changed while its benchmark digest was computed")
        encoded_name = relative.as_posix().encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def capture_clean_source_lock(
    repo_root: Path | None = None,
    *,
    git_probe: GitProbe = _git_probe,
    source_files: Sequence[str] = _SOURCE_LOCK_FILES,
) -> CleanSourceLock:
    """Bind the run to one full clean Git revision and one source digest."""

    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    top_level = Path(git_probe(root, ("rev-parse", "--show-toplevel")).strip()).resolve(strict=True)
    if top_level != root:
        raise RuntimeError("benchmark source root is not the Git top level")
    revision = git_probe(root, ("rev-parse", "HEAD")).strip().casefold()
    if not _FULL_GIT_REVISION.fullmatch(revision):
        raise RuntimeError("benchmark requires a full immutable Git revision")
    status = git_probe(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    if status:
        raise RuntimeError("benchmark requires a clean Git worktree")
    tracked_output = git_probe(root, ("ls-files", "-z", "--", *source_files))
    tracked = tuple(item for item in tracked_output.split("\x00") if item)
    if set(tracked) != set(source_files) or len(tracked) != len(source_files):
        raise RuntimeError("benchmark source-lock scope is not fully tracked")
    return CleanSourceLock(
        repo_root=root,
        git_revision=revision,
        source_sha256=_source_tree_sha256(root, source_files),
        source_file_count=len(source_files),
    )


def verify_clean_source_lock(
    expected: CleanSourceLock,
    *,
    git_probe: GitProbe = _git_probe,
    source_files: Sequence[str] = _SOURCE_LOCK_FILES,
) -> None:
    """Reject a dirty tree, revision change, or source drift after execution."""

    observed = capture_clean_source_lock(
        expected.repo_root,
        git_probe=git_probe,
        source_files=source_files,
    )
    if observed != expected:
        raise RuntimeError("benchmark implementation source drifted during execution")


def bind_frozen_local_models(
    target_path: Path,
    draft_path: Path,
    *,
    fingerprint: ModelFingerprint = fingerprint_local_model,
) -> tuple[LocalModelLock, LocalModelLock]:
    """Fail closed unless both local paths exactly match preregistered bytes."""

    target = Path(target_path).expanduser().resolve(strict=True)
    drafter = Path(draft_path).expanduser().resolve(strict=True)
    if target == drafter:
        raise RuntimeError("target and drafter must be distinct local model snapshots")
    locks: list[LocalModelLock] = []
    for role, label, identity, path in (
        ("target", TARGET_REPOSITORY_LABEL, TARGET_CONTENT_IDENTITY, target),
        ("drafter", DRAFT_REPOSITORY_LABEL, DRAFT_CONTENT_IDENTITY, drafter),
    ):
        observed = fingerprint(path)
        if getattr(observed, "revision", None) != identity:
            raise RuntimeError(f"{role} local model does not match the frozen content identity")
        locks.append(
            LocalModelLock(
                role=role,
                repository_label=label,
                content_identity=identity,
                resolved_path=path,
            )
        )
    return locks[0], locks[1]


def verify_frozen_local_models(
    expected: Sequence[LocalModelLock],
    *,
    fingerprint: ModelFingerprint = fingerprint_local_model,
) -> None:
    """Re-fingerprint both model bundles after all paired generations."""

    if tuple(lock.role for lock in expected) != ("target", "drafter"):
        raise RuntimeError("post-run model verification received an invalid lock set")
    for lock in expected:
        observed = fingerprint(lock.resolved_path)
        if getattr(observed, "revision", None) != lock.content_identity:
            raise RuntimeError(f"{lock.role} local model changed during benchmark execution")


def _assert_source_free_artifact(value: object) -> None:
    """Reject private records, content fields, and absolute paths recursively."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("public benchmark artifact keys must be strings")
            normalized = key.casefold()
            path_key = normalized.endswith(("_path", "_paths")) and normalized != "bmp_paths"
            if normalized in _SENSITIVE_ARTIFACT_KEYS or path_key:
                raise ValueError("public benchmark artifact contains a sensitive key")
            if normalized == "hidden_labels_serialized" and item is not False:
                raise ValueError("hidden labels must never be serialized")
            _assert_source_free_artifact(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_source_free_artifact(item)
        return
    if isinstance(value, str) and (
        value.startswith(("/", "~/")) or _WINDOWS_ABSOLUTE_PATH.match(value)
    ):
        raise ValueError("public benchmark artifact contains an absolute path")


def serialize_source_free_result(
    envelope: BenchmarkResultEnvelope,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize the fixed provenance envelope and no private run records."""

    if not isinstance(envelope, BenchmarkResultEnvelope):
        raise TypeError("only BenchmarkResultEnvelope may be serialized for publication")
    envelope.__post_init__()
    payload = envelope.to_dict()
    _assert_source_free_artifact(payload)
    return json.dumps(payload, sort_keys=True, indent=indent, allow_nan=False) + "\n"


@dataclass(frozen=True)
class HiddenOracle:
    """Evaluator programs that never enter an agent-visible workspace."""

    public_regression: str
    hidden_checks: str


@dataclass(frozen=True)
class CorpusCase:
    split: str
    fixture: CodingFixture
    oracle: HiddenOracle
    editable_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split not in {"smoke", "development"}:
            raise ValueError("corpus split must be smoke or development")
        public_names = {item.relative_name for item in self.fixture.public_files}
        if not self.editable_names or not set(self.editable_names) <= public_names:
            raise ValueError("editable names must be a non-empty subset of public files")
        compile(self.oracle.public_regression, f"<{self.fixture.fixture_id}-public>", "exec")
        compile(self.oracle.hidden_checks, f"<{self.fixture.fixture_id}-hidden>", "exec")


def _public_test(module: str, body: str) -> str:
    return f"""import unittest

from {module} import *


class PublicContractTests(unittest.TestCase):
{body}


if __name__ == "__main__":
    unittest.main()
"""


def _case(
    *,
    split: str,
    fixture_id: str,
    instruction: str,
    module: str,
    source: str,
    public_test_body: str,
    public_regression: str,
    hidden_checks: str,
) -> CorpusCase:
    return CorpusCase(
        split=split,
        fixture=CodingFixture(
            fixture_id=fixture_id,
            instruction=(
                f"{instruction} Work only in this workspace. Preserve the public API and use only "
                "the Python standard library. Do not edit the public test or create extra files. "
                "Before finishing run: python3 -B -m unittest discover -s . -p test_public_*.py"
            ),
            public_files=(
                PublicFile(relative_name=f"{module}.py", content=source),
                PublicFile(
                    relative_name=f"test_public_{module}.py",
                    content=_public_test(module, public_test_body),
                ),
            ),
        ),
        oracle=HiddenOracle(public_regression=public_regression, hidden_checks=hidden_checks),
        editable_names=(f"{module}.py",),
    )


CORPUS: tuple[CorpusCase, ...] = (
    _case(
        split="smoke",
        fixture_id="s01",
        instruction="Implement normalize_whitespace(text) in text_utils.py.",
        module="text_utils",
        source='''"""Small text normalization helpers."""\n\n\ndef normalize_whitespace(text):\n    """Collapse Unicode whitespace runs and strip the ends."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_words_and_newline(self):
        self.assertEqual(normalize_whitespace("  hello   local\\nworld  "), "hello local world")

    def test_empty(self):
        self.assertEqual(normalize_whitespace("   "), "")""",
        public_regression="""from text_utils import normalize_whitespace
assert normalize_whitespace("  hello   local\\nworld  ") == "hello local world"
assert normalize_whitespace("   ") == ""
""",
        hidden_checks="""from text_utils import normalize_whitespace
assert normalize_whitespace("a\\tb\\r\\nc") == "a b c"
assert normalize_whitespace("\u2003alpha\u00a0beta\u2009") == "alpha beta"
""",
    ),
    _case(
        split="smoke",
        fixture_id="s02",
        instruction="Implement clamp(value, lower, upper) in math_utils.py; reject an inverted interval.",
        module="math_utils",
        source='''"""Numeric boundary helpers."""\n\n\ndef clamp(value, lower, upper):\n    """Return value constrained to the inclusive [lower, upper] interval."""\n    return value\n''',
        public_test_body="""    def test_inside_and_edges(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-1, 0, 10), 0)
        self.assertEqual(clamp(12, 0, 10), 10)""",
        public_regression="""from math_utils import clamp
assert clamp(5, 0, 10) == 5
assert clamp(-1, 0, 10) == 0
assert clamp(12, 0, 10) == 10
""",
        hidden_checks="""from math_utils import clamp
assert clamp(0.25, 0.5, 1.0) == 0.5
assert clamp(1.0, 1.0, 1.0) == 1.0
try:
    clamp(1, 4, 2)
except ValueError:
    pass
else:
    raise AssertionError("inverted intervals must fail")
""",
    ),
    _case(
        split="smoke",
        fixture_id="s03",
        instruction="Implement batched(iterable, size) in iter_utils.py as a lazy iterator of tuples.",
        module="iter_utils",
        source='''"""Iterator helpers."""\n\n\ndef batched(iterable, size):\n    """Yield tuples of at most size items without materializing iterable."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_full_and_partial_batches(self):
        self.assertEqual(list(batched([1, 2, 3, 4, 5], 2)), [(1, 2), (3, 4), (5,)])""",
        public_regression="""from iter_utils import batched
assert list(batched([1, 2, 3, 4, 5], 2)) == [(1, 2), (3, 4), (5,)]
""",
        hidden_checks="""from iter_utils import batched
assert list(batched((value for value in range(4)), 3)) == [(0, 1, 2), (3,)]
assert list(batched([], 2)) == []
try:
    list(batched([1], 0))
except ValueError:
    pass
else:
    raise AssertionError("non-positive size must fail")
""",
    ),
    _case(
        split="smoke",
        fixture_id="s04",
        instruction="Implement deep_get(mapping, dotted_path, default=None) in mapping_utils.py.",
        module="mapping_utils",
        source='''"""Nested mapping helpers."""\n\n\ndef deep_get(mapping, dotted_path, default=None):\n    """Follow dot-separated mapping keys; return default when traversal fails."""\n    return default\n''',
        public_test_body="""    def test_present_and_missing(self):
        payload = {"user": {"profile": {"name": "Mio"}}}
        self.assertEqual(deep_get(payload, "user.profile.name"), "Mio")
        self.assertEqual(deep_get(payload, "user.id", 7), 7)""",
        public_regression="""from mapping_utils import deep_get
payload = {"user": {"profile": {"name": "Mio"}}}
assert deep_get(payload, "user.profile.name") == "Mio"
assert deep_get(payload, "user.id", 7) == 7
""",
        hidden_checks="""from mapping_utils import deep_get
marker = object()
assert deep_get({"a": {"b": None}}, "a.b", marker) is None
assert deep_get({"a": 3}, "a.b", marker) is marker
assert deep_get({"": {"x": 1}}, ".x", marker) == 1
""",
    ),
    _case(
        split="development",
        fixture_id="d01",
        instruction=(
            "Implement backoff_delays(attempts, base=1.0, factor=2.0, cap=None) in retry.py "
            "with validation and per-delay capping."
        ),
        module="retry",
        source='''"""Retry scheduling utilities."""\n\n\ndef backoff_delays(attempts, base=1.0, factor=2.0, cap=None):\n    """Return deterministic exponential delays for attempts after the first call."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_exponential_and_cap(self):
        self.assertEqual(backoff_delays(4), [1.0, 2.0, 4.0, 8.0])
        self.assertEqual(backoff_delays(4, cap=3.0), [1.0, 2.0, 3.0, 3.0])""",
        public_regression="""from retry import backoff_delays
assert backoff_delays(4) == [1.0, 2.0, 4.0, 8.0]
assert backoff_delays(4, cap=3.0) == [1.0, 2.0, 3.0, 3.0]
""",
        hidden_checks="""from retry import backoff_delays
assert backoff_delays(0) == []
assert backoff_delays(3, base=0.5, factor=3, cap=2) == [0.5, 1.5, 2]
for args in [(-1,), (2, -1), (2, 1, 0), (2, 1, 2, -1)]:
    try:
        backoff_delays(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(args)
""",
    ),
    _case(
        split="development",
        fixture_id="d02",
        instruction=(
            "Implement topological_sort(graph) in graph.py. Include neighbor-only nodes, use lexical "
            "tie-breaking, do not mutate input, and reject cycles."
        ),
        module="graph",
        source='''"""Deterministic directed-graph algorithms."""\n\n\ndef topological_sort(graph):\n    """Return a deterministic ordering for a mapping of node to dependencies."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_dependency_order(self):
        graph = {"deploy": {"test"}, "test": {"build"}, "build": set()}
        self.assertEqual(topological_sort(graph), ["build", "test", "deploy"])""",
        public_regression="""from graph import topological_sort
graph = {"deploy": {"test"}, "test": {"build"}, "build": set()}
assert topological_sort(graph) == ["build", "test", "deploy"]
""",
        hidden_checks="""from graph import topological_sort
graph = {"z": {"a"}, "b": set()}
snapshot = {key: set(value) for key, value in graph.items()}
assert topological_sort(graph) == ["a", "b", "z"]
assert graph == snapshot
try:
    topological_sort({"a": {"b"}, "b": {"a"}})
except ValueError:
    pass
else:
    raise AssertionError("cycle must fail")
""",
    ),
    _case(
        split="development",
        fixture_id="d03",
        instruction="Implement the bounded LRUCache API in cache.py using Python standard-library data structures.",
        module="cache",
        source='''"""Small in-memory caches."""\n\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n\n    def get(self, key, default=None):\n        return default\n\n    def put(self, key, value):\n        pass\n\n    def __len__(self):\n        return 0\n''',
        public_test_body="""    def test_eviction(self):
        cache = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(cache.get("a"), 1)
        cache.put("c", 3)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(len(cache), 2)""",
        public_regression="""from cache import LRUCache
cache = LRUCache(2)
cache.put("a", 1); cache.put("b", 2)
assert cache.get("a") == 1
cache.put("c", 3)
assert cache.get("b") is None and len(cache) == 2
""",
        hidden_checks="""from cache import LRUCache
cache = LRUCache(2)
cache.put("a", 1); cache.put("b", 2); cache.put("a", 9); cache.put("c", 3)
assert cache.get("a") == 9 and cache.get("b", "missing") == "missing"
try:
    LRUCache(0)
except ValueError:
    pass
else:
    raise AssertionError("zero capacity must fail")
""",
    ),
    _case(
        split="development",
        fixture_id="d04",
        instruction=(
            "Implement redact_secrets(value, keys, replacement='[REDACTED]') in redaction.py. "
            "Recursively copy dict/list/tuple containers and match dictionary keys case-insensitively."
        ),
        module="redaction",
        source='''"""Structured-data redaction."""\n\n\ndef redact_secrets(value, keys, replacement="[REDACTED]"):\n    """Return a redacted deep copy of built-in containers."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_nested_mapping(self):
        value = {"token": "abc", "nested": [{"PASSWORD": "xyz", "ok": 1}]}
        expected = {"token": "***", "nested": [{"PASSWORD": "***", "ok": 1}]}
        self.assertEqual(redact_secrets(value, {"token", "password"}, "***"), expected)""",
        public_regression="""from redaction import redact_secrets
value = {"token": "abc", "nested": [{"PASSWORD": "xyz", "ok": 1}]}
expected = {"token": "***", "nested": [{"PASSWORD": "***", "ok": 1}]}
assert redact_secrets(value, {"token", "password"}, "***") == expected
""",
        hidden_checks="""from redaction import redact_secrets
source = {"Auth": ("keep", {"secret": 4}), "items": [1, 2]}
result = redact_secrets(source, {"auth", "secret"})
assert result == {"Auth": "[REDACTED]", "items": [1, 2]}
assert source == {"Auth": ("keep", {"secret": 4}), "items": [1, 2]}
assert result is not source and result["items"] is not source["items"]
""",
    ),
    _case(
        split="development",
        fixture_id="d05",
        instruction=(
            "Implement group_totals(rows, group_key, value_key) in records.py using Decimal for exact "
            "accumulation; accept int/float/str/Decimal values and return Decimal totals."
        ),
        module="records",
        source='''"""Tabular record aggregation."""\n\n\ndef group_totals(rows, group_key, value_key):\n    """Aggregate numeric record values without binary floating-point drift."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_grouping(self):
        from decimal import Decimal
        rows = [{"team": "a", "amount": "0.1"}, {"team": "a", "amount": 0.2}, {"team": "b", "amount": 2}]
        self.assertEqual(group_totals(rows, "team", "amount"), {"a": Decimal("0.3"), "b": Decimal("2")})""",
        public_regression="""from decimal import Decimal
from records import group_totals
rows = [{"team": "a", "amount": "0.1"}, {"team": "a", "amount": 0.2}, {"team": "b", "amount": 2}]
assert group_totals(rows, "team", "amount") == {"a": Decimal("0.3"), "b": Decimal("2")}
""",
        hidden_checks="""from decimal import Decimal
from records import group_totals
assert group_totals([], "k", "v") == {}
assert group_totals([{"k": None, "v": Decimal("1.25")}, {"k": None, "v": "2.75"}], "k", "v") == {None: Decimal("4.00")}
rows = ({"k": index % 2, "v": index} for index in range(4))
assert group_totals(rows, "k", "v") == {0: Decimal("2"), 1: Decimal("4")}
""",
    ),
    _case(
        split="development",
        fixture_id="d06",
        instruction=(
            "Implement canonicalize_url(url) in urls.py: lowercase scheme/host, remove fragments and "
            "default ports, normalize an empty path to '/', and sort query pairs while preserving duplicates/blanks."
        ),
        module="urls",
        source='''"""Stable URL canonicalization."""\n\n\ndef canonicalize_url(url):\n    """Return a conservative canonical representation of an HTTP(S) URL."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_basic_http_url(self):
        value = "HTTPS://Example.COM:443/path?z=2&a=1#section"
        self.assertEqual(canonicalize_url(value), "https://example.com/path?a=1&z=2")""",
        public_regression="""from urls import canonicalize_url
value = "HTTPS://Example.COM:443/path?z=2&a=1#section"
assert canonicalize_url(value) == "https://example.com/path?a=1&z=2"
""",
        hidden_checks="""from urls import canonicalize_url
assert canonicalize_url("http://EXAMPLE.com:80") == "http://example.com/"
assert canonicalize_url("https://example.com/?b=&a=2&a=1") == "https://example.com/?a=1&a=2&b="
assert canonicalize_url("https://user:pass@EXAMPLE.com:444/x") == "https://user:pass@example.com:444/x"
""",
    ),
    _case(
        split="development",
        fixture_id="d07",
        instruction=(
            "Implement rolling_mean(values, window) in window.py as a single-pass function returning floats; "
            "support generators, reject non-positive windows, and return [] when the window is too large."
        ),
        module="window",
        source='''"""Streaming numeric windows."""\n\n\ndef rolling_mean(values, window):\n    """Return the arithmetic mean of each complete consecutive window."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_three_value_window(self):
        self.assertEqual(rolling_mean([1, 2, 3, 6], 3), [2.0, 11 / 3])""",
        public_regression="""from window import rolling_mean
assert rolling_mean([1, 2, 3, 6], 3) == [2.0, 11 / 3]
""",
        hidden_checks="""from window import rolling_mean
assert rolling_mean((value for value in [2, 4, 8]), 1) == [2.0, 4.0, 8.0]
assert rolling_mean([1, 2], 3) == []
try:
    rolling_mean([1], 0)
except ValueError:
    pass
else:
    raise AssertionError("invalid window")
""",
    ),
    _case(
        split="development",
        fixture_id="d08",
        instruction=(
            "Implement deduplicate_events(events) in events.py. Keep first-seen id order but retain the last "
            "full event for each id, accept any iterable, copy outputs, and reject events missing id."
        ),
        module="events",
        source='''"""Event-stream normalization."""\n\n\ndef deduplicate_events(events):\n    """Deduplicate mapping events by id without mutating input mappings."""\n    raise NotImplementedError("TODO")\n''',
        public_test_body="""    def test_last_value_first_order(self):
        events = [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "a", "v": 3}]
        self.assertEqual(deduplicate_events(events), [{"id": "a", "v": 3}, {"id": "b", "v": 2}])""",
        public_regression="""from events import deduplicate_events
events = [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "a", "v": 3}]
assert deduplicate_events(events) == [{"id": "a", "v": 3}, {"id": "b", "v": 2}]
""",
        hidden_checks="""from events import deduplicate_events
source = [{"id": 0, "v": []}, {"id": 0, "v": [1]}]
result = deduplicate_events(iter(source))
assert result == [{"id": 0, "v": [1]}]
assert result[0] is not source[1]
try:
    deduplicate_events([{"value": 1}])
except (KeyError, ValueError):
    pass
else:
    raise AssertionError("missing id must fail")
""",
    ),
)


# Explicit seals make corpus edits visible during review.  Update them only as
# a preregistered protocol revision, never in response to benchmark outcomes.
SMOKE_SUITE_SHA256 = "d0fef6c7ccfcccbf6dcbc70f973d931f5dba023f45f4335482d72f626c824afc"
DEVELOPMENT_SUITE_SHA256 = "3b9cd3611486e5b3d20a5786249fdc9446af3aecf3a648d507f46a5f5c3208e5"
ALL_SUITE_SHA256 = "32f4a59ab1831b5130fccbcdc3a9affcfe5e03204f7de0423aec106d4251857c"


def select_cases(split: str) -> tuple[CorpusCase, ...]:
    if split == "all":
        return CORPUS
    if split not in {"smoke", "development"}:
        raise ValueError("split must be smoke, development, or all")
    return tuple(case for case in CORPUS if case.split == split)


def sealed_suite_sha256(cases: Sequence[CorpusCase]) -> str:
    """Return the explicit split seal and reject a silently changed corpus."""

    identifiers = tuple(case.fixture.fixture_id for case in cases)
    seals = {
        tuple(case.fixture.fixture_id for case in select_cases("smoke")): SMOKE_SUITE_SHA256,
        tuple(case.fixture.fixture_id for case in select_cases("development")): DEVELOPMENT_SUITE_SHA256,
        tuple(case.fixture.fixture_id for case in select_cases("all")): ALL_SUITE_SHA256,
    }
    expected = seals.get(identifiers)
    if expected is None:
        raise ValueError("cases must be one complete frozen MioCodeBench split")
    actual = fixture_suite_sha256(tuple(case.fixture for case in cases))
    if actual != expected:
        raise RuntimeError("MioCodeBench corpus no longer matches its explicit suite seal")
    return expected


def fixture_tree_sha256(fixture: CodingFixture) -> str:
    digest = hashlib.sha256()
    for public_file in sorted(fixture.public_files, key=lambda item: item.relative_name):
        encoded_name = public_file.relative_name.encode("utf-8")
        encoded_content = public_file.content.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(b"F")
        digest.update(len(encoded_content).to_bytes(8, "big"))
        digest.update(encoded_content)
    return digest.hexdigest()


def protocol_suite_sha256(
    cases: Sequence[CorpusCase],
    *,
    evaluator_timeout_s: float = FROZEN_EVALUATOR_TIMEOUT_S,
    seed: int = FROZEN_SEED,
    bootstrap_samples: int = FROZEN_BOOTSTRAP_SAMPLES,
    alpha: float = FROZEN_ALPHA,
    minimum_pairs_for_claim: int = FROZEN_MINIMUM_PAIRS_FOR_CLAIM,
) -> str:
    """Seal public inputs, private evaluators, edit scope, and analysis knobs."""

    ordered = sorted(cases, key=lambda case: case.fixture.fixture_id)
    if not ordered:
        raise ValueError("protocol seal requires at least one corpus case")
    payload = {
        "schema": "mio.coding-quality-corpus-protocol.v1",
        "gate_profile_sha256": GATE_PROFILE_SHA256,
        "public_suite_sha256": fixture_suite_sha256(
            tuple(case.fixture for case in ordered)
        ),
        "evaluator": {
            "implementation": "CorpusHiddenEvaluator.v1",
            "timeout_s": evaluator_timeout_s,
        },
        "analysis": {
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "alpha": alpha,
            "minimum_pairs_for_claim": minimum_pairs_for_claim,
        },
        "cases": [
            {
                "fixture_id": case.fixture.fixture_id,
                "split": case.split,
                "editable_names": list(case.editable_names),
                "public_regression": case.oracle.public_regression,
                "hidden_checks": case.oracle.hidden_checks,
            }
            for case in ordered
        ],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# Explicit private-protocol seals.  Updating a hidden oracle, edit scope,
# timeout, schedule seed, or analysis threshold requires a reviewed protocol
# revision and new constants before any generation can start.
SMOKE_PROTOCOL_SHA256 = "3d595244487a82471a1ac596fd0f29be495bf3b0e63d8a42e1cfad0985553cf3"
DEVELOPMENT_PROTOCOL_SHA256 = "f93ca2da59a4e1a3f4cd74bc146b508dfbb68e3e1b4c96eecca5880dcd0ccd0f"
ALL_PROTOCOL_SHA256 = "4077241ce159be68f755c5d083f5e4d99f7d786c075c2e5de23a24ff304b147d"


def sealed_protocol_sha256(cases: Sequence[CorpusCase]) -> str:
    _assert_gate_profile_seal()
    identifiers = tuple(case.fixture.fixture_id for case in cases)
    seals = {
        tuple(case.fixture.fixture_id for case in select_cases("smoke")): SMOKE_PROTOCOL_SHA256,
        tuple(case.fixture.fixture_id for case in select_cases("development")): DEVELOPMENT_PROTOCOL_SHA256,
        tuple(case.fixture.fixture_id for case in select_cases("all")): ALL_PROTOCOL_SHA256,
    }
    expected = seals.get(identifiers)
    if expected is None:
        raise ValueError("cases must be one complete frozen MioCodeBench split")
    actual = protocol_suite_sha256(cases)
    if actual != expected:
        raise RuntimeError("MioCodeBench private protocol no longer matches its explicit seal")
    return expected


def workspace_tree_sha256(workspace: Path) -> str:
    digest = hashlib.sha256()
    root = workspace.resolve()
    entries = sorted(root.rglob("*"))
    for path in entries:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if path.is_symlink():
            digest.update(b"L")
        elif path.is_dir():
            digest.update(b"D")
        elif path.is_file():
            content = path.read_bytes()
            digest.update(b"F")
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        else:
            digest.update(b"O")
    return digest.hexdigest()


def build_agent_tool_surface(
    condition: str, agent_module: Any | None = None
) -> tuple[Mapping[str, Any], tuple[dict, ...]]:
    """Return the frozen benchmark-only tool registry for one arm."""

    if condition not in {GATE_OFF, GATE_ON}:
        raise ValueError("unknown benchmark condition")
    if agent_module is None:
        from mio import agent as agent_module

    names = _PUBLIC_TOOL_NAMES if condition == GATE_OFF else _GATE_TOOL_NAMES
    registry = MappingProxyType({name: agent_module.AGENT_TOOLS[name] for name in names})
    specs_by_name = {
        spec["function"]["name"]: spec
        for spec in agent_module.AGENT_TOOLS_SPEC
        if isinstance(spec, dict) and isinstance(spec.get("function"), dict)
    }
    specs = tuple(specs_by_name[name] for name in names)
    return registry, specs


def agent_turn_to_observation(result: Any, condition: str) -> GenerationObservation:
    """Adapt a content-free AgentTurnResult to the benchmark runner contract."""

    if condition not in {GATE_OFF, GATE_ON}:
        raise ValueError("unknown benchmark condition")
    rounds = tuple(getattr(result, "rounds", ()) or ())
    events = tuple(getattr(result, "tool_events", ()) or ())
    mutations = [
        event
        for event in events
        if getattr(event, "operation", "") in {"write", "edit"}
        and bool(getattr(event, "allowed", False))
        and getattr(event, "outcome", "") == "ok"
    ]
    validations = [event for event in events if getattr(event, "operation", "") == "validate"]
    gate_record = getattr(result, "quality_gate", None)
    terminal_complete = getattr(result, "terminal_reason", "") == "model_final"
    if condition == GATE_ON:
        if not isinstance(gate_record, Mapping):
            terminal_complete = False
        else:
            decision = gate_record.get("decision", gate_record.get("status"))
            terminal_complete = terminal_complete and decision in {
                "satisfied",
                "not_applicable",
                "complete",
                "pass",
            }

    return GenerationObservation(
        completed=bool(terminal_complete),
        mutation_count=len(mutations),
        tool_calls=int(getattr(result, "tool_calls", len(events)) or 0),
        output_tokens=sum(int(getattr(item, "completion_tokens", 0) or 0) for item in rounds),
        validation_attempted=bool(validations),
        validation_succeeded=any(
            bool(getattr(event, "allowed", False)) and getattr(event, "outcome", "") == "ok" for event in validations
        ),
        model_seconds=sum(float(getattr(item, "total_time_s", 0.0) or 0.0) for item in rounds),
        wall_seconds=float(getattr(result, "wall_time_s", 0.0) or 0.0),
    )


class AgentTurnExecutor(Protocol):
    def __call__(
        self,
        *,
        request: GenerationRequest,
        tool_registry: Mapping[str, Any],
        tool_specs: Sequence[dict],
        tool_policy: Any,
        quality_gate_enabled: bool,
        effort: str,
    ) -> Any: ...


class NativeAgentTurnExecutor:
    """One loaded Mio engine with fresh conversation and cache state per arm."""

    def __init__(self, *, config: Any, manager: Any, engine: Any, tier: str) -> None:
        self.config = config
        self.manager = manager
        self.engine = engine
        self.tier = tier

    def _reset_engine_state(self) -> None:
        invalidator = getattr(self.engine, "_prefix_cache_invalidate", None)
        if callable(invalidator):
            invalidator()
        if hasattr(self.engine, "_last_prompt_tokens"):
            self.engine._last_prompt_tokens = []
        if hasattr(self.engine, "_pending_assistant_prefill"):
            self.engine._pending_assistant_prefill = ""
        dspark = getattr(self.engine, "_dspark_runtime", None)
        prefix_cache = getattr(dspark, "_prefix_cache", None)
        executor = getattr(dspark, "_executor", None)
        if prefix_cache is not None and executor is not None:
            executor.submit(prefix_cache.reset).result()

    def __call__(
        self,
        *,
        request: GenerationRequest,
        tool_registry: Mapping[str, Any],
        tool_specs: Sequence[dict],
        tool_policy: Any,
        quality_gate_enabled: bool,
        effort: str,
    ) -> Any:
        from mio import agent
        from mio.prompt_policy import PromptPolicy

        self._reset_engine_state()
        state = {
            "tier": self.tier,
            "prompt_policy": PromptPolicy(),
            "tool_policy": tool_policy,
            "tool_registry": tool_registry,
            "tool_specs": tuple(tool_specs),
            "messages": [],
            "quality_gate_enabled": quality_gate_enabled,
            "coding_effort": effort,
        }
        previous_console = agent.console
        try:
            from rich.console import Console

            agent.console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
            return agent._process_user_input(
                request.instruction,
                self.engine,
                self.manager,
                self.config,
                state,
            )
        finally:
            agent.console = previous_console


class RealMioGenerationRunner:
    """Benchmark callback enforcing identical bytes and a network-free policy."""

    def __init__(
        self,
        *,
        executor: AgentTurnExecutor,
        fixtures: Sequence[CodingFixture],
        effort: str,
        agent_module: Any | None = None,
    ) -> None:
        self.executor = executor
        self.effort = effort
        self.agent_module = agent_module
        self._initial_digests = {fixture.fixture_id: fixture_tree_sha256(fixture) for fixture in fixtures}

    def __call__(self, request: GenerationRequest) -> GenerationObservation:
        expected = self._initial_digests.get(request.fixture_id)
        if expected is None or workspace_tree_sha256(request.workspace) != expected:
            raise RuntimeError("agent workspace does not match the frozen initial fixture bytes")

        from mio.agent_policy import AgentToolPermission, AgentToolPolicy

        policy = AgentToolPolicy.coding_workspace(request.workspace, allow_network=False)
        if AgentToolPermission.NETWORK in policy.permissions:
            raise RuntimeError("coding benchmark policy unexpectedly grants network access")
        registry, specs = build_agent_tool_surface(request.condition, self.agent_module)
        result = self.executor(
            request=request,
            tool_registry=registry,
            tool_specs=specs,
            tool_policy=policy,
            quality_gate_enabled=request.condition == GATE_ON,
            effort=self.effort,
        )
        return agent_turn_to_observation(result, request.condition)


class CorpusHiddenEvaluator:
    """Run pristine public and private oracles after all agent arms are sealed."""

    def __init__(
        self,
        cases: Sequence[CorpusCase],
        *,
        timeout_s: float = FROZEN_EVALUATOR_TIMEOUT_S,
    ) -> None:
        self._cases = {case.fixture.fixture_id: case for case in cases}
        self.timeout_s = timeout_s

    @staticmethod
    def _scope_is_valid(case: CorpusCase, workspace: Path) -> bool:
        root = workspace.resolve()
        initial = {item.relative_name: item.content.encode("utf-8") for item in case.fixture.public_files}
        entries = list(root.rglob("*"))
        if any(path.is_symlink() or not path.is_file() for path in entries):
            return False
        observed_names = {path.relative_to(root).as_posix() for path in entries}
        if observed_names != set(initial):
            return False

        edited = False
        editable = set(case.editable_names)
        for relative_name, original in initial.items():
            current = (root / relative_name).read_bytes()
            if relative_name in editable:
                edited = edited or current != original
            elif current != original:
                return False
        return edited

    def _run_oracle(self, workspace: Path, source: str) -> bool:
        from mio.agent_policy import AgentToolPolicy, sandboxed_command

        bootstrap = "import sys; sys.path.insert(0, '.')\n" + source
        argv = [sys.executable, "-I", "-B", "-c", bootstrap]
        policy = AgentToolPolicy.read_only(workspace)
        sandboxed_argv, environment = sandboxed_command(
            argv,
            policy,
            allow_process_fork=False,
        )
        environment = dict(environment)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                sandboxed_argv,
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0

    def __call__(self, request: EvaluationRequest) -> HiddenEvaluation:
        case = self._cases.get(request.fixture_id)
        if case is None:
            raise RuntimeError("hidden oracle is missing for a corpus fixture")
        scope_valid = self._scope_is_valid(case, request.workspace)
        regression_free = self._run_oracle(request.workspace, case.oracle.public_regression)
        hidden_passed = self._run_oracle(request.workspace, case.oracle.hidden_checks)
        # ``passed`` is the preregistered composite primary outcome, not merely
        # the private assertion bit.
        return HiddenEvaluation(
            passed=scope_valid and regression_free and hidden_passed,
            regression_free=regression_free,
        )


def execute_corpus(
    *,
    cases: Sequence[CorpusCase],
    runner: Callable[[GenerationRequest], GenerationObservation],
    work_root: Path,
    hidden_evaluator: Callable[[EvaluationRequest], HiddenEvaluation] | None = None,
    seed: int = FROZEN_SEED,
) -> BenchmarkExecution:
    """Execute a non-confirmatory split through the shared two-phase harness."""

    if not cases:
        raise ValueError("at least one corpus case is required")
    if seed != FROZEN_SEED:
        raise ValueError("MioCodeBench execution seed is frozen by the private protocol seal")
    fixtures = tuple(case.fixture for case in cases)
    sealed_protocol_sha256(cases)
    return run_benchmark(
        fixtures=fixtures,
        preregistration=Preregistration(
            expected_suite_sha256=sealed_suite_sha256(cases),
            seed=seed,
            bootstrap_samples=FROZEN_BOOTSTRAP_SAMPLES,
            alpha=FROZEN_ALPHA,
            minimum_pairs_for_claim=FROZEN_MINIMUM_PAIRS_FOR_CLAIM,
        ),
        runner=runner,
        hidden_evaluator=hidden_evaluator or CorpusHiddenEvaluator(cases),
        work_root=work_root,
    )


def _load_native_executor(
    *,
    tier: str,
    config_path: Path | None,
    target_path: Path,
    draft_path: Path,
    config_loader: Callable[[Path | None], Any] | None = None,
    manager_factory: Callable[[Any], Any] | None = None,
) -> tuple[NativeAgentTurnExecutor, Any]:
    if config_loader is None:
        from mio.config import load_config

        config_loader = load_config
    if manager_factory is None:
        from mio.model_manager import ModelManager

        manager_factory = ModelManager

    config = config_loader(config_path)
    if tier not in config.tiers:
        raise ValueError(f"unknown configured tier: {tier}")
    config.active_tiers = [tier]
    tier_config = config.tiers[tier]
    # The command-line snapshots are authoritative.  Never allow a persisted
    # tier, registry fallback, or alternate drafter backend to change an arm.
    tier_config.target_model = os.fspath(Path(target_path).resolve(strict=True))
    tier_config.draft_model = os.fspath(Path(draft_path).resolve(strict=True))
    tier_config.draft_fallback_model = None
    tier_config.drafter_backend = "dflash"
    tier_config.drafter_strict = True
    tier_config.tq_bits = 16
    tier_config.pq_bits = 16
    tier_config.bmp_paths = 1
    tier_config.ddtree_budget = 0
    tier_config.context_window = FROZEN_CONTEXT_WINDOW
    tier_config.max_output_tokens = FROZEN_MAX_OUTPUT_TOKENS
    tier_config.temperature = 0.0
    tier_config.top_p = 1.0
    tier_config.top_k = 0
    # Both arms are cold by protocol. Disable DSpark's private prefix slots;
    # MioEngine's own cache is explicitly invalidated before every arm.
    tier_config.dspark_prefix_cache = False
    manager = manager_factory(config)
    manager.load_tier(tier)
    engine = manager.get_engine(tier)
    drafter_status = getattr(engine, "drafter_status", None)
    try:
        loaded_drafter_path = Path(str(drafter_status.get("ref"))).expanduser().resolve(strict=True)
    except (AttributeError, OSError, RuntimeError):
        loaded_drafter_path = None
    if not isinstance(drafter_status, Mapping) or (
        drafter_status.get("selected") != "dflash"
        or drafter_status.get("fallback_used") is not False
        or drafter_status.get("strict") is not True
        or loaded_drafter_path != Path(draft_path).resolve(strict=True)
    ):
        manager.unload_all()
        raise RuntimeError("benchmark engine did not load the exact strict DFlash primary")
    return NativeAgentTurnExecutor(config=config, manager=manager, engine=engine, tier=tier), manager


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["smoke", "development", "all"], default="smoke")
    parser.add_argument("--tier", default="small")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "ultra"], default="medium")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--target-path",
        type=Path,
        required=True,
        help=f"exact local {TARGET_REPOSITORY_LABEL} snapshot",
    )
    parser.add_argument(
        "--draft-path",
        type=Path,
        required=True,
        help=f"exact local {DRAFT_REPOSITORY_LABEL} snapshot",
    )
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _assert_frozen_environment()
    _assert_gate_profile_seal()
    cases = select_cases(args.split)
    fixtures = tuple(case.fixture for case in cases)
    protocol_sha256 = sealed_protocol_sha256(cases)
    source_lock = capture_clean_source_lock()
    runtime_identity = collect_runtime_identity()
    model_locks = bind_frozen_local_models(args.target_path, args.draft_path)
    output_path = validate_output_path(
        args.output,
        source_root=source_lock.repo_root,
        model_locks=model_locks,
    )
    manager = None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        # Model loaders are verbose; reserve stdout exclusively for the JSON
        # artifact and keep content-bearing agent output in an in-memory sink.
        with redirect_stdout(sys.stderr):
            executor, manager = _load_native_executor(
                tier=args.tier,
                config_path=args.config,
                target_path=model_locks[0].resolved_path,
                draft_path=model_locks[1].resolved_path,
            )
        runner = RealMioGenerationRunner(
            executor=executor,
            fixtures=fixtures,
            effort=args.effort,
        )
        if args.work_root is None:
            temporary = tempfile.TemporaryDirectory(prefix="mio-codebench-")
            work_root = Path(temporary.name)
        else:
            work_root = args.work_root
        with redirect_stdout(sys.stderr):
            execution = execute_corpus(cases=cases, runner=runner, work_root=work_root)
        manager.unload_all()
        manager = None
        verify_frozen_local_models(model_locks)
        verify_clean_source_lock(source_lock)
        verify_runtime_identity(runtime_identity)
        serialized = serialize_source_free_result(
            BenchmarkResultEnvelope(
                source_lock=source_lock,
                model_locks=model_locks,
                runtime_identity=runtime_identity,
                aggregate=execution.aggregate,
                split=args.split,
                tier=args.tier,
                effort=args.effort,
                protocol_sha256=protocol_sha256,
            )
        )
        if output_path is None:
            sys.stdout.write(serialized)
        else:
            # Re-resolve after the long-running generation to catch a target or
            # parent changed into a symlink while the benchmark was executing.
            verified_output = validate_output_path(
                output_path,
                source_root=source_lock.repo_root,
                model_locks=model_locks,
            )
            if verified_output is None:  # pragma: no cover - guarded above
                raise RuntimeError("benchmark output path disappeared")
            _atomic_write_result(verified_output, serialized)
    finally:
        if manager is not None:
            manager.unload_all()
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
