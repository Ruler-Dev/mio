#!/usr/bin/env python3
"""Sealed paired-generation runner for Mio's SWE-bench quality study.

This module contains no dataset downloader, evaluator, or model auto-loader.
Generation is dependency injected and confirmatory runs remain subject to the
hard gates in :mod:`scripts.bench_swebench_quality`.  The implementation keeps
gold data out of the model boundary, runs an adjacent *whole pair* as the
smallest resumable unit, and promotes checkpoints only after both arms exist.

The model-visible checkout never contains ``.git`` (in any casing).  Trusted
patch capture uses external, private Git metadata and assistant prose is never
accepted as a prediction.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import bench_swebench_quality as protocol  # noqa: E402

GENERATION_SCHEMA = f"{protocol.SCHEMA}.paired-generation-runner.v1"
GENERATION_RUN_HEADER_SCHEMA = f"{GENERATION_SCHEMA}.run-header"
GENERATION_RECEIPT_SCHEMA = f"{GENERATION_SCHEMA}.receipt"
PORTABLE_LAYOUT_PROFILE_SCHEMA = f"{GENERATION_SCHEMA}.portable-layout.v1"
SEALED_RUNTIME_BINDING_SCHEMA = f"{GENERATION_SCHEMA}.sealed-runtime-binding.v1"
PAIR_ARTIFACT_BINDING_SCHEMA = f"{GENERATION_SCHEMA}.pair-artifact-binding.v1"
ARM_TELEMETRY_SCHEMA = f"{GENERATION_SCHEMA}.arm-telemetry.v1"
TELEMETRY_MANIFEST_SCHEMA = f"{GENERATION_SCHEMA}.telemetry-manifest.v1"
TOOL_SURFACE = ("validate", "bash", "read", "write", "edit")
TARGET_CONTEXT_TOKENS = 32_768
TARGET_MAX_OUTPUT_TOKENS_PER_ROUND = 4_096
TARGET_MAX_ROUNDS = 12
TARGET_TQ_BITS = 16
TARGET_PQ_BITS = 16
TARGET_BMP_PATHS = 1
TARGET_DDTREE_BUDGET = 0
TARGET_TEMPERATURE = 0.0
TARGET_TOP_P = 1.0
TARGET_TOP_K = 0
FROZEN_COMMAND_TIMEOUT_SECONDS = 300.0
CONFIRMATORY_GENERATION_ENABLED = False
CONFIRMATORY_BLOCKERS = (
    "v2_efficiency_guardrail_and_content_free_round_telemetry",
    "end_to_end_automatic_chain_of_custody_requires_clean_subprocess_or_in_memory_provenance",
    "v2_wall_overrun_adjudication",
    "v2_storage_bounded_external_object_strategy",
    "pinned_x86_64_official_evaluation_image_digests",
)
AUTOMATIC_ATTESTATION_SCHEMA = f"{GENERATION_SCHEMA}.automatic-attestation.v1"
RUNTIME_ATTESTATION_SCHEMA = f"{AUTOMATIC_ATTESTATION_SCHEMA}.runtime"
AUTOMATIC_BINDING_SOURCE = "automatic_preflight_local_v1"
NON_EVIDENCE_BINDING_SOURCE = "caller_supplied_non_evidence_smoke_v1"
_MIO_REQUIRED_TRACKED_PATHS = (
    "mio/__init__.py",
    "pyproject.toml",
    "scripts/run_swebench_quality_generation.py",
)
_RUNTIME_RELEVANT_SOURCE_SUFFIXES = frozenset({".dylib", ".jinja", ".json", ".metal", ".py", ".pyi", ".so", ".toml"})
_CRITICAL_RUNTIME_DISTRIBUTIONS = (
    "dflash-mlx",
    "huggingface-hub",
    "jinja2",
    "mlx",
    "mlx-dspark",
    "mlx-lm",
    "numpy",
    "rich",
    "safetensors",
    "tokenizers",
    "transformers",
)
_RUNTIME_ENVIRONMENT_NAMES = frozenset(
    {
        "ACCELERATE_USE_CPU",
        "DYLD_LIBRARY_PATH",
        "HOME",
        "LD_LIBRARY_PATH",
        "OMP_NUM_THREADS",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONHOME",
        "PYTHONPATH",
        "SHELL",
        "TMPDIR",
        "VECLIB_MAXIMUM_THREADS",
    }
)
_RUNTIME_ENVIRONMENT_PREFIXES = (
    "ACCELERATE_",
    "HF_",
    "METAL_",
    "MIO_",
    "MLX_",
    "TOKENIZERS_",
    "TRANSFORMERS_",
)
_DISTRIBUTION_NAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\Z")
_AUTOMATIC_ATTESTATION_SEAL = object()
_VALIDATED_TELEMETRY_SEAL = object()
_PORTABLE_LAYOUT_PROFILE = {
    "schema": PORTABLE_LAYOUT_PROFILE_SCHEMA,
    "portable_cross_process_artifact_audit": True,
    "runtime_manifest_required": True,
    "per_arm_telemetry_required": True,
    "current_environment_reattestation_is_separate": True,
}
_ROUND_TRACE_FIELDS = (
    "round_index",
    "prompt_tokens",
    "completion_tokens",
    "total_time_s",
    "prompt_tps",
    "generation_tps",
    "generation_backend",
    "fallback_ar",
    "prefill_ns",
    "decode_ns",
    "model_total_ns",
    "logical_prompt_tokens",
    "physical_prefill_tokens",
    "physical_decode_tokens",
    "warm_offset",
    "warm_offset_tokens",
    "timing_source",
    "drafter_requested",
    "drafter_selected",
    "drafter_ref",
    "phase_censored",
    "deadline_hit",
)
_TOOL_TRACE_FIELDS = (
    "sequence",
    "round_index",
    "tool_name",
    "operation",
    "permission",
    "allowed",
    "outcome",
    "target_sha256",
    "duration_ns",
    "effective_timeout_ns",
    "exit_code_or_signal",
    "output_chars",
    "audit_count",
    "audit_sha256",
    "timeout_enforced",
    "telemetry_complete",
    "effect_unknown",
)
_TOOL_NAMES = frozenset((*TOOL_SURFACE, "unknown"))
_TOOL_OPERATIONS = frozenset((*TOOL_SURFACE, "unknown", "multiple"))
_TOOL_PERMISSIONS = frozenset({"read", "write", "shell", "network", "none", "multiple"})
_TOOL_OUTCOMES_BY_NAME = {
    "bash": frozenset({"ok", "nonzero", "timeout", "output_limit", "denied", "error"}),
    "validate": frozenset(
        {
            "ok",
            "nonzero",
            "timeout",
            "output_limit",
            "denied",
            "error",
            "no_work",
            "unrecognized",
            "unscoped",
            "untrusted_executable",
        }
    ),
    "read": frozenset({"ok", "not_found", "denied", "error", "timeout"}),
    "write": frozenset({"ok", "denied", "error", "timeout"}),
    "edit": frozenset({"ok", "not_found", "old_string_not_found", "denied", "error", "timeout"}),
    "unknown": frozenset({"unrecognized"}),
}
_TOOL_PERMISSION_BY_NAME = {
    "bash": frozenset({"shell"}),
    "validate": frozenset({"shell"}),
    "read": frozenset({"read"}),
    "write": frozenset({"write"}),
    "edit": frozenset({"read", "write", "multiple"}),
    "unknown": frozenset({"none"}),
}
_TERMINAL_REASONS = frozenset(
    {
        "model_final",
        "model_error",
        "tool_timeout",
        "quality_incomplete",
        "budget_exhausted",
        "budget_finalization",
    }
)
_BUDGET_EXHAUSTION_KINDS = frozenset(
    {"none", "wall_time", "completion_tokens", "context_tokens", "model_rounds", "tool_calls", "tool_output"}
)
_QUALITY_DECISIONS = frozenset({"not_applicable", "incomplete", "pass"})
_QUALITY_PHASES = frozenset(
    {
        "disabled",
        "observing",
        "awaiting_change",
        "no_net_change",
        "dirty",
        "validation_failed",
        "passed",
        "model_error",
    }
)
_QUALITY_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "ultra"})
_QUALITY_INTENTS = frozenset({"general", "inspect", "code_change_requested"})
_QUALITY_OBLIGATIONS = frozenset(
    {
        "any_validation",
        "diff",
        "test_or_build",
        "test",
        "static_or_diff",
        "static",
        "review_or_second_distinct_test",
        "complete_workspace_snapshot",
        "net_workspace_change",
    }
)
_VALIDATION_KINDS = ("test", "build", "static", "diff", "review")
_SNAPSHOT_METHODS = frozenset({"git", "manifest", "incomplete"})
_SNAPSHOT_ERROR_CODES = frozenset({"snapshot_incomplete", "snapshot_limit"})
_MAX_TOOL_OUTPUT_CHARS = 24_000
_TOOL_PARENT_OVERHEAD_NS = 5_000_000_000
# The agent budget is exactly 1,800 seconds.  The outer executor clock also
# includes Python return/serialization overhead, which is evidence rather than
# additional model time.  Bound that overhead explicitly instead of making a
# legitimate deadline outcome impossible to seal.
_EXECUTOR_WALL_OVERHEAD_NS = 5_000_000_000
MODEL_INSTRUCTION_TEMPLATE = (
    "Repository: {repo}\n"
    "Base commit: {base_commit}\n\n"
    "Problem statement:\n{problem_statement}\n\n"
    "Modify the provided workspace to solve the problem. Use only the declared local tools. "
    "Do not use network access. Run the narrowest relevant validation after edits."
)


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise protocol.ProtocolError(f"{label} must be a lowercase SHA-256")


def _require_commit(value: str, label: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise protocol.ProtocolError(f"{label} must be a lowercase Git commit")


def _mkdir_private(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(path, 0o700)
    return path


def _assert_no_visible_git(workspace: Path) -> None:
    """Reject every visible .git spelling without following symlinks."""

    root = workspace.resolve(strict=True)
    errors: list[OSError] = []
    entries = 0
    for directory, dirnames, filenames in os.walk(
        root,
        followlinks=False,
        onerror=errors.append,
    ):
        entries += len(dirnames) + len(filenames)
        if entries > 100_000:
            raise protocol.ProtocolError("model-visible workspace exceeded traversal bound")
        if len(Path(directory).relative_to(root).parts) > 256:
            raise protocol.ProtocolError("model-visible workspace exceeded traversal depth")
        if any(name.casefold() == ".git" for name in (*dirnames, *filenames)):
            raise protocol.ProtocolError("model-visible workspace contains forbidden Git metadata")
    if errors:
        raise protocol.ProtocolError("model-visible workspace scan was incomplete")


def _canonical_local_directory(path: Path, label: str) -> Path:
    """Resolve one caller path without accepting symlink or spelling aliases."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise protocol.ProtocolError(f"{label} must be an absolute canonical path")
    protocol._reject_symlink_path_components(candidate)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise protocol.ProtocolError(f"{label} does not exist") from exc
    if candidate != resolved or not resolved.is_dir():
        raise protocol.ProtocolError(f"{label} must be an ordinary canonical directory, not an alias")

    # Case-insensitive filesystems can resolve a differently-cased spelling to
    # the same inode without reporting a symlink.  Require every supplied path
    # component to match the directory entry byte-for-byte as well.
    current = Path(resolved.anchor)
    for component in resolved.parts[1:]:
        try:
            names = {entry.name for entry in os.scandir(current)}
        except OSError as exc:
            raise protocol.ProtocolError(f"cannot inspect canonical {label}") from exc
        if component not in names:
            raise protocol.ProtocolError(f"{label} uses a filesystem spelling alias")
        current /= component
    return resolved


def _hash_regular_attestation_file(path: Path, label: str) -> tuple[int, str]:
    """Hash a stable regular file descriptor and reject link-swap races."""

    descriptor = -1
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise protocol.ProtocolError(f"{label} is not a regular file")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    except protocol.ProtocolError:
        raise
    except OSError as exc:
        raise protocol.ProtocolError(f"cannot hash {label}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    named_identity = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
    )
    if before_identity != after_identity or after_identity != named_identity:
        raise protocol.ProtocolError(f"{label} changed while it was fingerprinted")
    return before.st_size, digest.hexdigest()


def _clean_git_document(repo_root: Path) -> dict[str, Any]:
    _assert_executing_mio_tree(repo_root)
    top_level = protocol._run_git(repo_root, ["rev-parse", "--show-toplevel"]).decode().strip()
    try:
        observed_root = Path(top_level).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise protocol.ProtocolError("Mio Git root is unavailable") from exc
    if observed_root != repo_root:
        raise protocol.ProtocolError("attested Mio path is not the exact Git worktree root")

    def snapshot() -> tuple[str, str]:
        status = protocol._run_git(
            repo_root,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
        )
        if status:
            raise protocol.ProtocolError("automatic attestation requires a clean Mio worktree with no untracked files")
        head = protocol._run_git(repo_root, ["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip()
        tree = protocol._run_git(repo_root, ["rev-parse", "--verify", "HEAD^{tree}"]).decode().strip()
        _require_commit(head, "attested Mio HEAD")
        _require_commit(tree, "attested Mio tree")
        return head, tree

    before = snapshot()
    tracked = {
        os.fsdecode(value)
        for value in protocol._run_git(
            repo_root,
            ["ls-files", "-z", "--", *_MIO_REQUIRED_TRACKED_PATHS],
        ).split(b"\0")
        if value
    }
    if tracked != set(_MIO_REQUIRED_TRACKED_PATHS):
        raise protocol.ProtocolError("automatic attestation target is not the required Mio source tree")
    ignored = {
        os.fsdecode(value)
        for value in protocol._run_git(
            repo_root,
            [
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
                "--",
                "mio",
                "scripts",
                "experimental",
            ],
        ).split(b"\0")
        if value
    }
    runtime_relevant_ignored = sorted(
        path
        for path in ignored
        if Path(path).suffix.casefold() in _RUNTIME_RELEVANT_SOURCE_SUFFIXES and "__pycache__" not in Path(path).parts
    )
    if runtime_relevant_ignored:
        raise protocol.ProtocolError("Mio source tree contains ignored runtime-relevant files")
    for relative_path in _MIO_REQUIRED_TRACKED_PATHS:
        candidate = repo_root / relative_path
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise protocol.ProtocolError("required Mio source file is unavailable") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise protocol.ProtocolError("required Mio source file must be regular and single-link")
    after = snapshot()
    if after != before:
        raise protocol.ProtocolError("Mio HEAD changed while it was automatically attested")
    return {
        "head_commit": before[0],
        "head_tree": before[1],
        "required_tracked_paths": list(_MIO_REQUIRED_TRACKED_PATHS),
        "worktree_clean": True,
        "untracked_files": 0,
        "ignored_runtime_relevant_files": 0,
    }


def _assert_executing_mio_tree(repo_root: Path) -> None:
    """Bind the clean checkout to the Python source that is actually running."""

    executing_root = Path(__file__).resolve(strict=True).parents[1]
    if executing_root != repo_root:
        raise protocol.ProtocolError("attested Mio repository differs from the executing runner source tree")

    from experimental.effort import model_identity
    from mio import agent, agent_policy, coding_quality, engine

    modules = (
        sys.modules[__name__],
        protocol,
        model_identity,
        agent,
        agent_policy,
        coding_quality,
        engine,
    )
    relative_origins = []
    for module in modules:
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str):
            raise protocol.ProtocolError("critical Mio runtime module has no filesystem origin")
        try:
            origin = Path(raw_origin).resolve(strict=True)
            relative = origin.relative_to(repo_root).as_posix()
        except (OSError, RuntimeError, ValueError) as exc:
            raise protocol.ProtocolError("critical Mio runtime module was loaded outside the attested tree") from exc
        metadata = origin.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise protocol.ProtocolError("critical Mio runtime module must be regular and single-link")
        relative_origins.append(relative)
    tracked_origins = {
        os.fsdecode(value)
        for value in protocol._run_git(
            repo_root,
            ["ls-files", "-z", "--", *sorted(relative_origins)],
        ).split(b"\0")
        if value
    }
    if tracked_origins != set(relative_origins):
        raise protocol.ProtocolError("critical executing Mio runtime module is not tracked by attested HEAD")


def _local_model_document(model_root: Path) -> dict[str, Any]:
    from experimental.effort.model_identity import ModelIdentityError, fingerprint_local_model

    try:
        fingerprint = fingerprint_local_model(model_root)
    except ModelIdentityError as exc:
        raise protocol.ProtocolError(f"cannot automatically fingerprint local MLX target: {exc}") from exc
    if fingerprint.revision != protocol.EXPECTED_MODEL_IDENTITY:
        raise protocol.ProtocolError(
            "automatic local model fingerprint does not match the frozen Qwen 3.6 27B identity"
        )
    if not fingerprint.files or fingerprint.total_bytes <= 0:
        raise protocol.ProtocolError("automatic local model fingerprint is incomplete")
    for item in fingerprint.files:
        relative = Path(item.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise protocol.ProtocolError("automatic local model manifest contains an unsafe path")
        candidate = model_root / relative
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise protocol.ProtocolError("automatic local model manifest changed after hashing") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise protocol.ProtocolError("automatic local model files must be regular and single-link")
        if metadata.st_size != item.size_bytes:
            raise protocol.ProtocolError("automatic local model file size changed after hashing")
    return {
        "fingerprint_schema": fingerprint.schema,
        "identity": fingerprint.revision,
        "manifest_sha256": fingerprint.digest,
        "file_count": len(fingerprint.files),
        "total_bytes": fingerprint.total_bytes,
        "complete_file_bytes_hashed": True,
        "canonical_local_directory": True,
        "single_link_files": True,
    }


def _canonical_distribution_name(value: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", value.strip().casefold())
    if not _DISTRIBUTION_NAME_RE.fullmatch(normalized):
        raise protocol.ProtocolError("installed distribution has an invalid canonical name")
    return normalized


def _installed_distribution_environment() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    buckets: dict[str, list[Any]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(raw_name, str) or not raw_name.strip() or not isinstance(version, str) or not version.strip():
            raise protocol.ProtocolError("installed distribution metadata is incomplete")
        name = _canonical_distribution_name(raw_name)
        buckets.setdefault(name, []).append(distribution)
    if not buckets:
        raise protocol.ProtocolError("runtime distribution environment is empty")
    critical: dict[str, Any] = {}
    for name in _CRITICAL_RUNTIME_DISTRIBUTIONS:
        candidates = buckets.get(name, [])
        if len(candidates) != 1:
            raise protocol.ProtocolError(f"critical runtime distribution must have one installation: {name}")
        critical[name] = candidates[0]
    installed = [
        {
            "name": name,
            "installations": len(candidates),
            "versions": sorted(distribution.version for distribution in candidates),
        }
        for name, candidates in sorted(buckets.items())
    ]
    return critical, installed


def _distribution_content_document(name: str, distribution: Any) -> dict[str, Any]:
    raw_files = distribution.files
    if raw_files is None:
        raise protocol.ProtocolError(f"critical runtime distribution lacks a file manifest: {name}")
    files = sorted(raw_files, key=lambda item: item.as_posix())
    if not files or len(files) > 25_000:
        raise protocol.ProtocolError(f"critical runtime distribution file count is invalid: {name}")
    rows = []
    seen: set[str] = set()
    total_bytes = 0
    for item in files:
        relative = item.as_posix()
        if not relative or "\x00" in relative or "\\" in relative or relative in seen:
            raise protocol.ProtocolError(f"critical runtime distribution manifest is ambiguous: {name}")
        seen.add(relative)
        candidate = Path(distribution.locate_file(item))
        size_bytes, file_sha256 = _hash_regular_attestation_file(
            candidate,
            f"{name} runtime distribution file",
        )
        total_bytes += size_bytes
        rows.append({"path": relative, "size_bytes": size_bytes, "sha256": file_sha256})
    content_sha256 = protocol.sha256_bytes(
        protocol.canonical_json_bytes(
            {
                "schema": f"{RUNTIME_ATTESTATION_SCHEMA}.distribution-content",
                "name": name,
                "version": distribution.version,
                "files": rows,
            }
        )
    )
    return {
        "name": name,
        "version": distribution.version,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "content_sha256": content_sha256,
    }


def _collect_runtime_document() -> dict[str, Any]:
    critical_distributions, installed = _installed_distribution_environment()
    critical_content = [
        _distribution_content_document(name, critical_distributions[name]) for name in _CRITICAL_RUNTIME_DISTRIBUTIONS
    ]
    executable = Path(sys.executable).resolve(strict=True)
    executable_size, executable_sha256 = _hash_regular_attestation_file(executable, "Python executable")
    environment = [
        {
            "name": name,
            "value_sha256": protocol.sha256_bytes(os.environ[name].encode("utf-8", errors="surrogateescape")),
        }
        for name in sorted(os.environ)
        if name in _RUNTIME_ENVIRONMENT_NAMES
        or any(name.startswith(prefix) for prefix in _RUNTIME_ENVIRONMENT_PREFIXES)
    ]
    return {
        "schema": RUNTIME_ATTESTATION_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": str(getattr(sys.implementation, "cache_tag", "")),
            "soabi": str(sysconfig.get_config_var("SOABI") or ""),
            "build": list(platform.python_build()),
            "executable_size_bytes": executable_size,
            "executable_sha256": executable_sha256,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "environment": environment,
        "installed_distributions": installed,
        "critical_distribution_contents": critical_content,
        "absolute_paths_serialized": False,
        "environment_values_serialized": False,
        "environment_value_hashes_serialized": True,
        "full_package_inventory_serialized": True,
    }


@dataclass(frozen=True)
class AutomaticGenerationAttestation:
    """Preflight source/model/runtime state; not end-to-end evidence."""

    repository_root: Path = field(repr=False, compare=False)
    model_root: Path = field(repr=False, compare=False)
    payload: bytes = field(repr=False)
    private_runtime_payload: bytes = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _AUTOMATIC_ATTESTATION_SEAL:
            raise protocol.ProtocolError("automatic generation attestation must be collected by the trusted path")
        try:
            document = json.loads(self.payload)
        except (UnicodeDecodeError, ValueError) as exc:
            raise protocol.ProtocolError("automatic generation attestation payload is invalid") from exc
        if protocol.canonical_json_bytes(document) != self.payload:
            raise protocol.ProtocolError("automatic generation attestation is not canonical")
        if (
            document.get("schema") != AUTOMATIC_ATTESTATION_SCHEMA
            or document.get("binding_source") != AUTOMATIC_BINDING_SOURCE
            or document.get("automatic") is not True
        ):
            raise protocol.ProtocolError("automatic generation attestation schema is invalid")
        try:
            private_runtime = json.loads(self.private_runtime_payload)
        except (UnicodeDecodeError, ValueError) as exc:
            raise protocol.ProtocolError("private runtime attestation payload is invalid") from exc
        if protocol.canonical_json_bytes(private_runtime) != self.private_runtime_payload:
            raise protocol.ProtocolError("private runtime attestation is not canonical")
        if protocol.sha256_bytes(self.private_runtime_payload) != document.get("runtime", {}).get("digest"):
            raise protocol.ProtocolError("public runtime digest differs from the private runtime manifest")

    @classmethod
    def collect(cls, *, repository_root: Path, model_root: Path) -> "AutomaticGenerationAttestation":
        repository = _canonical_local_directory(repository_root, "Mio repository root")
        model = _canonical_local_directory(model_root, "local model root")
        git_document = _clean_git_document(repository)
        model_document = _local_model_document(model)
        runtime_document = _collect_runtime_document()
        private_runtime_payload = protocol.canonical_json_bytes(runtime_document)
        runtime_digest = protocol.sha256_bytes(private_runtime_payload)
        critical_versions = [
            {"name": row["name"], "version": row["version"]}
            for row in runtime_document["critical_distribution_contents"]
        ]
        document = {
            "schema": AUTOMATIC_ATTESTATION_SCHEMA,
            "binding_source": AUTOMATIC_BINDING_SOURCE,
            "automatic": True,
            "git": git_document,
            "model": model_document,
            "runtime": {
                "digest": runtime_digest,
                "python": {
                    "implementation": runtime_document["python"].get("implementation", "unavailable"),
                    "version": runtime_document["python"].get("version", "unavailable"),
                },
                "critical_versions": critical_versions,
            },
            "privacy": {
                "absolute_local_paths_serialized": False,
                "environment_values_serialized": False,
                "environment_value_hashes_serialized": False,
                "full_package_inventory_serialized": False,
                "private_runtime_manifest_retained_in_memory": True,
            },
            "end_to_end_confirmatory_chain_of_custody_proven": False,
        }
        return cls(
            repository_root=repository,
            model_root=model,
            payload=protocol.canonical_json_bytes(document),
            private_runtime_payload=private_runtime_payload,
            _seal=_AUTOMATIC_ATTESTATION_SEAL,
        )

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.payload)

    @property
    def mio_commit(self) -> str:
        return str(self.as_dict()["git"]["head_commit"])

    @property
    def model_identity(self) -> str:
        return str(self.as_dict()["model"]["identity"])

    @property
    def runtime_digest(self) -> str:
        return str(self.as_dict()["runtime"]["digest"])

    def verify_current(self) -> None:
        current = type(self).collect(
            repository_root=self.repository_root,
            model_root=self.model_root,
        )
        if current.payload == self.payload and current.private_runtime_payload == self.private_runtime_payload:
            return
        before = self.as_dict()
        after = current.as_dict()
        if before["git"] != after["git"]:
            label = "Mio Git"
        elif before["model"] != after["model"]:
            label = "local model"
        else:
            label = "runtime/dependency environment"
        raise protocol.ProtocolError(f"automatic {label} attestation drifted")


@dataclass(frozen=True)
class GenerationBinding:
    """Preflight identities shared by all arms; not end-to-end evidence."""

    mio_commit: str
    model_identity: str
    runtime_digest: str
    binding_source: str
    automatic_attestation: AutomaticGenerationAttestation | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_commit(self.mio_commit, "Mio commit")
        if self.model_identity != protocol.EXPECTED_MODEL_IDENTITY:
            raise protocol.ProtocolError("generation must use the frozen Qwen 3.6 27B identity")
        _require_sha256(self.runtime_digest, "runtime digest")
        if self.binding_source == AUTOMATIC_BINDING_SOURCE:
            if not isinstance(self.automatic_attestation, AutomaticGenerationAttestation):
                raise protocol.ProtocolError("automatic preflight binding requires an automatic attestation")
            if (
                self.mio_commit != self.automatic_attestation.mio_commit
                or self.model_identity != self.automatic_attestation.model_identity
                or self.runtime_digest != self.automatic_attestation.runtime_digest
            ):
                raise protocol.ProtocolError("generation binding differs from its automatic attestation")
        elif self.binding_source == NON_EVIDENCE_BINDING_SOURCE:
            if self.automatic_attestation is not None:
                raise protocol.ProtocolError("caller-supplied smoke binding cannot claim automatic attestation")
        else:
            raise protocol.ProtocolError("generation binding source is not trusted or explicitly non-evidence")

    @classmethod
    def automatic_local(
        cls,
        *,
        repository_root: Path,
        model_root: Path,
    ) -> "GenerationBinding":
        attestation = AutomaticGenerationAttestation.collect(
            repository_root=repository_root,
            model_root=model_root,
        )
        return cls(
            mio_commit=attestation.mio_commit,
            model_identity=attestation.model_identity,
            runtime_digest=attestation.runtime_digest,
            binding_source=AUTOMATIC_BINDING_SOURCE,
            automatic_attestation=attestation,
        )

    @classmethod
    def for_non_evidence_smoke(
        cls,
        *,
        mio_commit: str,
        model_identity: str,
        runtime_digest: str,
    ) -> "GenerationBinding":
        return cls(
            mio_commit=mio_commit,
            model_identity=model_identity,
            runtime_digest=runtime_digest,
            binding_source=NON_EVIDENCE_BINDING_SOURCE,
        )

    def validate_for_run(
        self,
        *,
        evidence_run: bool,
        executor: ArmExecutor | None = None,
        tier_config: Any | None = None,
        require_executor_binding: bool = False,
    ) -> dict[str, Any]:
        if self.binding_source != AUTOMATIC_BINDING_SOURCE:
            if evidence_run:
                raise protocol.ProtocolError("confirmatory generation requires automatic preflight fingerprints")
            return _executor_model_binding_document(self, executor, tier_config)
        assert self.automatic_attestation is not None
        self.automatic_attestation.verify_current()
        if executor is None:
            if require_executor_binding:
                raise protocol.ProtocolError("automatic generation requires a bound native Mio executor")
            return {
                "automatic": True,
                "environment_reverified": True,
                "model_identity": self.model_identity,
            }
        return _executor_model_binding_document(self, executor, tier_config)

    def attestation_dict(self) -> dict[str, Any]:
        if self.automatic_attestation is not None:
            return self.automatic_attestation.as_dict()
        return {
            "schema": AUTOMATIC_ATTESTATION_SCHEMA,
            "binding_source": NON_EVIDENCE_BINDING_SOURCE,
            "automatic": False,
            "confirmatory_evidence_admissible": False,
            "caller_supplied_values": ["mio_commit", "model_identity", "runtime_digest"],
        }

    def as_dict(self) -> dict[str, str]:
        return {
            "mio_commit": self.mio_commit,
            "model_identity": self.model_identity,
            "runtime_digest": self.runtime_digest,
        }


def _executor_model_binding_document(
    binding: GenerationBinding,
    executor: ArmExecutor | None,
    supplied_tier_config: Any | None,
) -> dict[str, Any]:
    if binding.binding_source != AUTOMATIC_BINDING_SOURCE:
        return {
            "automatic": False,
            "confirmatory_evidence_admissible": False,
            "reason": "caller_supplied_non_evidence_smoke_binding",
        }
    from mio.config import MioConfig, TierConfig
    from mio.engine import MioEngine
    from mio.model_manager import ModelManager

    if executor is None or type(executor) is not NativeMioArmExecutor:
        raise protocol.ProtocolError("automatic generation requires the exact native Mio executor")
    assert binding.automatic_attestation is not None
    engine = executor.engine
    if type(engine) is not MioEngine or type(executor.manager) is not ModelManager:
        raise protocol.ProtocolError("automatic generation requires exact production engine and manager classes")
    if type(executor.config) is not MioConfig or executor.config is not executor.manager.config:
        raise protocol.ProtocolError("automatic generation requires the exact production Mio configuration")
    if (
        getattr(engine, "is_loaded", False) is not True
        or getattr(engine, "_target_model", None) is None
        or getattr(engine, "_tokenizer", None) is None
    ):
        raise protocol.ProtocolError("automatic generation requires an already loaded target engine")
    tier_config = getattr(engine, "tier_config", None)
    if type(tier_config) is not TierConfig or supplied_tier_config is not tier_config:
        raise protocol.ProtocolError("supplied tier config is not the exact loaded engine tier config")
    if executor.tier != tier_config.name or executor.config.tiers.get(executor.tier) is not tier_config:
        raise protocol.ProtocolError("loaded engine tier is not bound to the production Mio configuration")
    validate_target_only_tier(tier_config)
    target_reference = getattr(tier_config, "target_model", None)
    target_metadata = getattr(engine, "_target_meta", None)
    resolved_reference = target_metadata.get("resolved_model_ref") if isinstance(target_metadata, Mapping) else None
    if not isinstance(target_reference, str) or not isinstance(resolved_reference, (str, Path)):
        raise protocol.ProtocolError("loaded target engine does not expose a verifiable local model reference")
    configured_root = _canonical_local_directory(Path(target_reference), "configured target model root")
    loaded_root = _canonical_local_directory(Path(resolved_reference), "loaded target model root")
    if configured_root != binding.automatic_attestation.model_root or loaded_root != configured_root:
        raise protocol.ProtocolError("loaded target engine differs from the automatically fingerprinted model")
    loaded_tiers = getattr(executor.manager, "loaded_tiers", None)
    get_engine = getattr(executor.manager, "get_engine", None)
    if (
        not callable(loaded_tiers)
        or not callable(get_engine)
        or executor.tier not in loaded_tiers()
        or get_engine(executor.tier) is not engine
    ):
        raise protocol.ProtocolError("automatic generation manager is not bound to the attested target engine")
    return {
        "automatic": True,
        "preflight_only": True,
        "end_to_end_confirmatory_chain_of_custody_proven": False,
        "executor": _implementation_identity(executor),
        "engine": _implementation_identity(engine),
        "manager": _implementation_identity(executor.manager),
        "tier": executor.tier,
        "engine_loaded": True,
        "configured_and_loaded_model_paths_identical": True,
        "model_identity": binding.model_identity,
        "raw_target_telemetry_required": executor.require_raw_target_telemetry,
        "raw_target_telemetry_receipt_bound": False,
    }


@dataclass(frozen=True)
class ArmWorkspace:
    """Fresh, mutually isolated state allocated for exactly one arm."""

    workspace: Path
    external_git_directory: Path
    cache_directory: Path

    def validated(self) -> "ArmWorkspace":
        protocol._reject_symlink_path_components(self.workspace)
        workspace = self.workspace.resolve(strict=True)
        git_directory = protocol.require_private_directory(self.external_git_directory)
        cache_directory = protocol.require_private_directory(self.cache_directory)
        for candidate, label in (
            (workspace, "workspace"),
            (git_directory, "external Git directory"),
            (cache_directory, "cache directory"),
        ):
            if not candidate.is_dir():
                raise protocol.ProtocolError(f"arm {label} must be a directory")
        paths = (workspace, git_directory, cache_directory)
        for index, first in enumerate(paths):
            for second in paths[index + 1 :]:
                if protocol._is_within(first, second) or protocol._is_within(second, first):
                    raise protocol.ProtocolError("arm workspace, Git metadata, and cache must be separate")
        _assert_no_visible_git(workspace)
        if any(cache_directory.iterdir()):
            raise protocol.ProtocolError("fresh arm cache directory must start empty")
        return ArmWorkspace(workspace, git_directory, cache_directory)


@dataclass(frozen=True)
class ArmRunRequest:
    """Trusted request passed to a model executor.

    ``instruction`` is the complete model-facing task.  The instance identifier
    remains runner-private and is intentionally absent from that string.
    """

    entry: protocol.ScheduleEntry
    instruction: str
    workspace: Path
    cache_directory: Path
    tool_registry: Mapping[str, Any]
    tool_specs: tuple[dict[str, Any], ...]
    tool_policy: Any
    quality_gate_enabled: bool
    coding_effort: str
    seed: int


def _exact_keys(raw: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    if set(raw) != set(expected):
        raise protocol.ProtocolError(f"{label} fields differ from the sealed schema")


def _nonnegative_integer(value: Any, label: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise protocol.ProtocolError(f"{label} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise protocol.ProtocolError(f"{label} exceeds its frozen maximum")
    return value


def _finite_nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise protocol.ProtocolError(f"{label} must be a finite non-negative number")
    observed = float(value)
    if not math.isfinite(observed) or observed < 0:
        raise protocol.ProtocolError(f"{label} must be a finite non-negative number")
    return observed


def _validate_round_trace_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(raw, _ROUND_TRACE_FIELDS, "round telemetry")
    round_index = _nonnegative_integer(raw["round_index"], "round_index", maximum=TARGET_MAX_ROUNDS - 1)
    prompt_tokens = _nonnegative_integer(raw["prompt_tokens"], "prompt_tokens", maximum=TARGET_CONTEXT_TOKENS)
    completion_tokens = _nonnegative_integer(
        raw["completion_tokens"],
        "completion_tokens",
        maximum=TARGET_MAX_OUTPUT_TOKENS_PER_ROUND,
    )
    logical_prompt_tokens = _nonnegative_integer(
        raw["logical_prompt_tokens"],
        "logical_prompt_tokens",
        maximum=TARGET_CONTEXT_TOKENS,
    )
    warm_offset = _nonnegative_integer(raw["warm_offset"], "warm_offset", maximum=logical_prompt_tokens)
    warm_offset_tokens = _nonnegative_integer(
        raw["warm_offset_tokens"],
        "warm_offset_tokens",
        maximum=logical_prompt_tokens,
    )
    physical_prefill_tokens = _nonnegative_integer(raw["physical_prefill_tokens"], "physical_prefill_tokens")
    physical_decode_tokens = _nonnegative_integer(raw["physical_decode_tokens"], "physical_decode_tokens")
    prefill_ns = _nonnegative_integer(raw["prefill_ns"], "prefill_ns")
    decode_ns = _nonnegative_integer(raw["decode_ns"], "decode_ns")
    model_total_ns = _nonnegative_integer(raw["model_total_ns"], "model_total_ns")
    total_time_s = _finite_nonnegative_number(raw["total_time_s"], "total_time_s")
    prompt_tps = _finite_nonnegative_number(raw["prompt_tps"], "prompt_tps")
    generation_tps = _finite_nonnegative_number(raw["generation_tps"], "generation_tps")
    if prompt_tokens != logical_prompt_tokens:
        raise protocol.ProtocolError("round prompt-token aliases disagree")
    if warm_offset != warm_offset_tokens:
        raise protocol.ProtocolError("round warm-offset aliases disagree")
    if physical_prefill_tokens != logical_prompt_tokens - warm_offset:
        raise protocol.ProtocolError("round physical prefill accounting is inconsistent")
    if physical_decode_tokens < completion_tokens:
        raise protocol.ProtocolError("round physical decode work is below delivered completion tokens")
    if model_total_ns != prefill_ns + decode_ns:
        raise protocol.ProtocolError("round raw model time differs from prefill plus decode")
    if model_total_ns > math.ceil(total_time_s * 1_000_000_000):
        raise protocol.ProtocolError("round raw model time exceeds total model-call time")
    expected_prompt_tps = logical_prompt_tokens * 1_000_000_000 / prefill_ns if prefill_ns else 0.0
    expected_generation_tps = physical_decode_tokens * 1_000_000_000 / decode_ns if decode_ns else 0.0
    if not math.isclose(prompt_tps, expected_prompt_tps, rel_tol=1e-9, abs_tol=1e-9):
        raise protocol.ProtocolError("round prompt throughput differs from raw token/time accounting")
    if not math.isclose(generation_tps, expected_generation_tps, rel_tol=1e-9, abs_tol=1e-9):
        raise protocol.ProtocolError("round generation throughput differs from raw token/time accounting")
    if logical_prompt_tokens + completion_tokens > TARGET_CONTEXT_TOKENS:
        raise protocol.ProtocolError("round prompt plus completion exceeds the frozen context")
    if (
        raw["generation_backend"] != "baseline"
        or raw["fallback_ar"] is not False
        or raw["timing_source"] != "runtime_raw_ns"
        or raw["drafter_requested"] != "target_ar"
        or raw["drafter_selected"] != "baseline"
        or raw["drafter_ref"] is not None
    ):
        raise protocol.ProtocolError("round telemetry differs from target_ar/baseline/no-drafter raw timing")
    if not isinstance(raw["phase_censored"], bool) or not isinstance(raw["deadline_hit"], bool):
        raise protocol.ProtocolError("round censoring fields must be boolean")
    if raw["deadline_hit"] and not raw["phase_censored"]:
        raise protocol.ProtocolError("a round deadline hit must mark phase telemetry censored")
    return {
        "round_index": round_index,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time_s": total_time_s,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "generation_backend": "baseline",
        "fallback_ar": False,
        "prefill_ns": prefill_ns,
        "decode_ns": decode_ns,
        "model_total_ns": model_total_ns,
        "logical_prompt_tokens": logical_prompt_tokens,
        "physical_prefill_tokens": physical_prefill_tokens,
        "physical_decode_tokens": physical_decode_tokens,
        "warm_offset": warm_offset,
        "warm_offset_tokens": warm_offset_tokens,
        "timing_source": "runtime_raw_ns",
        "drafter_requested": "target_ar",
        "drafter_selected": "baseline",
        "drafter_ref": None,
        "phase_censored": raw["phase_censored"],
        "deadline_hit": raw["deadline_hit"],
    }


def _validate_tool_trace_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(raw, _TOOL_TRACE_FIELDS, "tool telemetry")
    sequence = _nonnegative_integer(raw["sequence"], "tool sequence", maximum=protocol.MAX_TOOL_CALLS_PER_ARM - 1)
    round_index = _nonnegative_integer(raw["round_index"], "tool round_index", maximum=TARGET_MAX_ROUNDS - 1)
    tool_name = raw["tool_name"]
    operation = raw["operation"]
    permission = raw["permission"]
    outcome = raw["outcome"]
    if tool_name not in _TOOL_NAMES or operation not in _TOOL_OPERATIONS or permission not in _TOOL_PERMISSIONS:
        raise protocol.ProtocolError("tool telemetry name, operation, or permission is outside the sealed vocabulary")
    if operation != tool_name or permission not in _TOOL_PERMISSION_BY_NAME[str(tool_name)]:
        raise protocol.ProtocolError("tool telemetry name/operation/permission combination is invalid")
    if outcome not in _TOOL_OUTCOMES_BY_NAME[str(tool_name)]:
        raise protocol.ProtocolError("tool telemetry name/outcome is outside the sealed vocabulary")
    if not isinstance(raw["allowed"], bool):
        raise protocol.ProtocolError("tool allowed must be boolean")
    for name in ("timeout_enforced", "telemetry_complete", "effect_unknown"):
        if not isinstance(raw[name], bool):
            raise protocol.ProtocolError(f"tool {name} must be boolean")
    target_sha256 = raw["target_sha256"]
    audit_sha256 = raw["audit_sha256"]
    if not isinstance(target_sha256, str):
        raise protocol.ProtocolError("tool target_sha256 is malformed")
    _require_sha256(target_sha256, "tool target digest")
    if not isinstance(audit_sha256, str):
        raise protocol.ProtocolError("tool audit_sha256 is malformed")
    _require_sha256(audit_sha256, "tool audit digest")
    duration_ns = _nonnegative_integer(raw["duration_ns"], "tool duration_ns")
    timeout_ns = raw["effective_timeout_ns"]
    if timeout_ns is not None:
        timeout_ns = _nonnegative_integer(timeout_ns, "tool effective timeout ns")
        if timeout_ns == 0:
            raise protocol.ProtocolError("known tool effective timeout must be positive")
    if tool_name != "unknown" and timeout_ns is None:
        raise protocol.ProtocolError("known tool telemetry lacks its effective timeout")
    maximum_timeout_ns = int(
        (FROZEN_COMMAND_TIMEOUT_SECONDS if tool_name in {"bash", "validate"} else 30.0) * 1_000_000_000
    )
    if timeout_ns is not None and timeout_ns > maximum_timeout_ns:
        raise protocol.ProtocolError("tool effective timeout exceeds the frozen bound")
    denied_outcomes = {"denied", "unrecognized", "unscoped", "untrusted_executable"}
    if (raw["allowed"] and outcome in denied_outcomes) or (not raw["allowed"] and outcome not in denied_outcomes):
        raise protocol.ProtocolError("tool allowed flag and outcome are inconsistent")
    if tool_name == "unknown":
        if (
            raw["allowed"] is not False
            or outcome != "unrecognized"
            or timeout_ns is not None
            or duration_ns > _TOOL_PARENT_OVERHEAD_NS
            or raw["timeout_enforced"]
            or not raw["telemetry_complete"]
            or raw["effect_unknown"]
        ):
            raise protocol.ProtocolError("unknown tool telemetry must use the bounded denied sentinel")
    else:
        if outcome == "timeout":
            if timeout_ns is None or not (timeout_ns <= duration_ns <= timeout_ns + _TOOL_PARENT_OVERHEAD_NS):
                raise protocol.ProtocolError("tool timeout duration is outside its frozen bound")
        elif timeout_ns is not None and duration_ns > timeout_ns + _TOOL_PARENT_OVERHEAD_NS:
            raise protocol.ProtocolError("non-timeout tool duration exceeds its effective timeout plus parent bound")
        if tool_name in {"bash", "validate"}:
            if raw["timeout_enforced"]:
                raise protocol.ProtocolError("command tool cannot claim a full-invocation watchdog")
            if not raw["telemetry_complete"] or raw["effect_unknown"]:
                raise protocol.ProtocolError("command tool telemetry must be complete with known effect")
        else:
            if not raw["timeout_enforced"]:
                raise protocol.ProtocolError("file tool telemetry must attest its terminable watchdog")
            incomplete_timeout = outcome == "timeout" and raw["effect_unknown"] and not raw["telemetry_complete"]
            if not raw["telemetry_complete"] and not incomplete_timeout:
                raise protocol.ProtocolError("incomplete file telemetry is allowed only for an unknown-effect timeout")
            if raw["effect_unknown"] != incomplete_timeout:
                raise protocol.ProtocolError("file tool effect_unknown disagrees with terminal timeout telemetry")
    exit_value = raw["exit_code_or_signal"]
    valid_exit = (
        exit_value is None
        or (isinstance(exit_value, int) and not isinstance(exit_value, bool) and -64 <= exit_value <= 255)
        or (
            isinstance(exit_value, str)
            and exit_value.startswith("signal:")
            and exit_value.removeprefix("signal:").isdigit()
            and 1 <= int(exit_value.removeprefix("signal:")) <= 64
        )
    )
    if not valid_exit:
        raise protocol.ProtocolError("tool exit_code_or_signal is malformed")
    if tool_name in {"read", "write", "edit", "unknown"} and exit_value is not None:
        raise protocol.ProtocolError("file and unknown tool telemetry cannot carry a process exit status")
    if outcome in denied_outcomes | {"error"} and exit_value is not None:
        raise protocol.ProtocolError("denied or error tool telemetry cannot carry a process exit status")
    if tool_name in {"bash", "validate"} and outcome in {"ok", "no_work"} and exit_value != 0:
        raise protocol.ProtocolError("successful command telemetry must record exit code zero")
    if outcome == "nonzero" and not (
        (isinstance(exit_value, int) and not isinstance(exit_value, bool) and exit_value != 0)
        or (isinstance(exit_value, str) and exit_value.startswith("signal:"))
    ):
        raise protocol.ProtocolError("nonzero command telemetry lacks a nonzero exit status")
    output_chars = _nonnegative_integer(raw["output_chars"], "tool output chars", maximum=_MAX_TOOL_OUTPUT_CHARS)
    audit_count = _nonnegative_integer(raw["audit_count"], "tool audit count")
    if (tool_name == "unknown" and audit_count != 0) or (tool_name != "unknown" and audit_count < 1):
        raise protocol.ProtocolError("tool audit_count is inconsistent with dispatcher admission")
    return {
        "sequence": sequence,
        "round_index": round_index,
        "tool_name": str(tool_name),
        "operation": str(operation),
        "permission": str(permission),
        "allowed": raw["allowed"],
        "outcome": str(outcome),
        "target_sha256": target_sha256,
        "duration_ns": duration_ns,
        "effective_timeout_ns": timeout_ns,
        "exit_code_or_signal": exit_value,
        "output_chars": output_chars,
        "audit_count": audit_count,
        "audit_sha256": audit_sha256,
        "timeout_enforced": raw["timeout_enforced"],
        "telemetry_complete": raw["telemetry_complete"],
        "effect_unknown": raw["effect_unknown"],
    }


def _budget_exhaustion_kind(value: Any, terminal_reason: str) -> str:
    if value is None:
        if terminal_reason in {"budget_exhausted", "budget_finalization"}:
            raise protocol.ProtocolError("budget terminal reason lacks an exhaustion classification")
        return "none"
    if not isinstance(value, str) or terminal_reason not in {
        "budget_exhausted",
        "budget_finalization",
        "quality_incomplete",
    }:
        raise protocol.ProtocolError("budget exhaustion detail disagrees with terminal reason")
    patterns = (
        (r"wall time limit 1800(?:\.0+)?s reached", "wall_time"),
        (rf"completion token limit {protocol.MAX_OUTPUT_TOKENS_PER_ARM} reached", "completion_tokens"),
        (rf"context token limit {TARGET_CONTEXT_TOKENS} reached", "context_tokens"),
        (rf"model round limit {TARGET_MAX_ROUNDS} reached", "model_rounds"),
        (rf"tool call limit {protocol.MAX_TOOL_CALLS_PER_ARM} reached", "tool_calls"),
        (r"tool result limit 100000 characters reached", "tool_output"),
    )
    for pattern, kind in patterns:
        if re.fullmatch(pattern, value):
            return kind
    raise protocol.ProtocolError("budget exhaustion detail is outside the sealed vocabulary")


def _validate_snapshot_observation(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"revision_sha256", "complete", "method", "error_codes"}
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise protocol.ProtocolError("quality snapshot observation fields are invalid")
    revision = raw["revision_sha256"]
    if not isinstance(revision, str):
        raise protocol.ProtocolError("quality snapshot revision is malformed")
    _require_sha256(revision, "quality snapshot revision")
    complete = raw["complete"]
    method = raw["method"]
    errors = raw["error_codes"]
    if not isinstance(complete, bool) or not isinstance(method, str) or not isinstance(errors, list):
        raise protocol.ProtocolError("quality snapshot status is malformed")
    methods = method.split("+")
    if not methods or len(methods) > 8 or any(item not in _SNAPSHOT_METHODS for item in methods):
        raise protocol.ProtocolError("quality snapshot method is outside the sealed vocabulary")
    if (
        len(errors) != len(set(errors))
        or errors != sorted(errors)
        or any(item not in _SNAPSHOT_ERROR_CODES for item in errors)
    ):
        raise protocol.ProtocolError("quality snapshot error codes are outside the sealed vocabulary")
    if complete and ("incomplete" in methods or errors):
        raise protocol.ProtocolError("complete quality snapshot carries an incomplete method or error")
    if not complete and ("incomplete" not in methods or not errors):
        raise protocol.ProtocolError("incomplete quality snapshot lacks a method/error attestation")
    return {
        "revision_sha256": revision,
        "complete": complete,
        "method": method,
        "error_codes": errors,
    }


def _validate_quality_gate_report(
    raw: Any,
    *,
    quality_gate_enabled: bool,
) -> dict[str, Any]:
    zero_counts = {name: 0 for name in _VALIDATION_KINDS}
    if not quality_gate_enabled:
        if raw is not None:
            raise protocol.ProtocolError("gate_off arm unexpectedly carries a quality-gate report")
        return {
            "enabled": False,
            "effort": "not_applicable",
            "intent": "not_applicable",
            "decision": "not_applicable",
            "status": "not_applicable",
            "phase": "experiment_disabled",
            "activated": False,
            "satisfied": True,
            "require_net_workspace_change": False,
            "mutation_epoch": 0,
            "request_sha256": None,
            "initial_revision_sha256": None,
            "current_revision_sha256": None,
            "initial_snapshot_complete": None,
            "initial_content_sha256": None,
            "current_content_sha256": None,
            "snapshot": None,
            "changed_kinds": [],
            "required": [],
            "missing": [],
            "validation_counts": zero_counts,
            "validate_invocations": 0,
            "recognized_validation_attempts": 0,
            "validation_attempts": 0,
            "misrouted_validation_commands": 0,
            "successful_reads": 0,
        }
    expected_keys = {
        "schema",
        "enabled",
        "effort",
        "intent",
        "request_sha256",
        "decision",
        "phase",
        "activated",
        "satisfied",
        "require_net_workspace_change",
        "mutation_epoch",
        "changed_kinds",
        "snapshot_complete",
        "snapshot_method",
        "snapshot_error_codes",
        "initial_revision_sha256",
        "current_revision_sha256",
        "initial_snapshot_complete",
        "initial_content_sha256",
        "current_content_sha256",
        "required",
        "missing",
        "validation_counts",
        "validate_invocations",
        "recognized_validation_attempts",
        "validation_attempts",
        "misrouted_validation_commands",
        "successful_reads",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise protocol.ProtocolError("gate_on quality report fields differ from the sealed schema")
    if raw.get("schema") != "mio.coding-quality-gate.v3" or raw.get("enabled") is not True:
        raise protocol.ProtocolError("gate_on quality report schema or enabled flag is invalid")
    effort = raw["effort"]
    intent = raw["intent"]
    decision = raw["decision"]
    phase = raw["phase"]
    if effort != "medium" or effort not in _QUALITY_EFFORTS or intent not in _QUALITY_INTENTS:
        raise protocol.ProtocolError("quality effort or intent differs from the frozen experiment")
    if decision not in _QUALITY_DECISIONS or phase not in _QUALITY_PHASES:
        raise protocol.ProtocolError("quality decision or phase is outside the sealed vocabulary")
    for name in (
        "activated",
        "satisfied",
        "require_net_workspace_change",
        "initial_snapshot_complete",
        "snapshot_complete",
    ):
        if not isinstance(raw[name], bool):
            raise protocol.ProtocolError(f"quality {name} must be boolean")
    if raw["require_net_workspace_change"] is not True:
        raise protocol.ProtocolError("gate_on quality report lacks the mandatory net-change contract")
    mutation_epoch = _nonnegative_integer(raw["mutation_epoch"], "quality mutation_epoch", maximum=128)
    request_sha256 = raw["request_sha256"]
    initial_revision = raw["initial_revision_sha256"]
    current_revision = raw["current_revision_sha256"]
    initial_content = raw["initial_content_sha256"]
    current_content = raw["current_content_sha256"]
    for value, label in (
        (request_sha256, "quality request digest"),
        (initial_revision, "quality initial revision"),
        (current_revision, "quality current revision"),
        (initial_content, "quality initial content"),
        (current_content, "quality current content"),
    ):
        if not isinstance(value, str):
            raise protocol.ProtocolError(f"{label} is malformed")
        _require_sha256(value, label)
    changed_kinds = raw["changed_kinds"]
    if (
        not isinstance(changed_kinds, list)
        or changed_kinds != sorted(set(changed_kinds))
        or any(item not in {"code", "docs"} for item in changed_kinds)
    ):
        raise protocol.ProtocolError("quality changed_kinds is outside the sealed vocabulary")
    required = raw["required"]
    missing = raw["missing"]
    if not isinstance(required, list) or not isinstance(missing, list):
        raise protocol.ProtocolError("quality obligations must be lists")
    if (
        required != list(dict.fromkeys(required))
        or missing != list(dict.fromkeys(missing))
        or any(item not in _QUALITY_OBLIGATIONS for item in (*required, *missing))
        or not set(missing).issubset(set(required) | {"complete_workspace_snapshot"})
    ):
        raise protocol.ProtocolError("quality obligations are outside the sealed vocabulary or inconsistent")
    counts = raw["validation_counts"]
    if not isinstance(counts, Mapping) or set(counts) != set(_VALIDATION_KINDS):
        raise protocol.ProtocolError("quality validation counts have the wrong schema")
    validation_counts = {
        name: _nonnegative_integer(counts[name], f"quality {name} validations", maximum=protocol.MAX_TOOL_CALLS_PER_ARM)
        for name in _VALIDATION_KINDS
    }
    validate_invocations = _nonnegative_integer(
        raw["validate_invocations"],
        "quality validate invocations",
        maximum=protocol.MAX_TOOL_CALLS_PER_ARM,
    )
    recognized_validation_attempts = _nonnegative_integer(
        raw["recognized_validation_attempts"],
        "quality recognized validation attempts",
        maximum=protocol.MAX_TOOL_CALLS_PER_ARM,
    )
    validation_attempts = _nonnegative_integer(
        raw["validation_attempts"],
        "quality validation-attempt alias",
        maximum=protocol.MAX_TOOL_CALLS_PER_ARM,
    )
    misrouted_validation_commands = _nonnegative_integer(
        raw["misrouted_validation_commands"],
        "quality misrouted validation commands",
        maximum=protocol.MAX_TOOL_CALLS_PER_ARM,
    )
    if validation_attempts != recognized_validation_attempts:
        raise protocol.ProtocolError("quality validation-attempt alias differs from recognized attempts")
    if recognized_validation_attempts > validate_invocations:
        raise protocol.ProtocolError("quality recognized validation attempts exceed validate invocations")
    if sum(validation_counts.values()) > recognized_validation_attempts:
        raise protocol.ProtocolError("quality successful validation counts exceed attempts")
    successful_reads = _nonnegative_integer(
        raw["successful_reads"],
        "quality successful reads",
        maximum=protocol.MAX_TOOL_CALLS_PER_ARM,
    )
    snapshot = _validate_snapshot_observation(
        {
            "revision_sha256": current_revision,
            "complete": raw["snapshot_complete"],
            "method": raw["snapshot_method"],
            "error_codes": raw["snapshot_error_codes"],
        }
    )
    activated = raw["activated"]
    satisfied = raw["satisfied"]
    require_net_workspace_change = raw["require_net_workspace_change"]
    snapshot_comparison_complete = raw["initial_snapshot_complete"] and snapshot["complete"]
    content_changed = initial_content != current_content
    revision_changed = initial_revision != current_revision
    if snapshot_comparison_complete and content_changed and not revision_changed:
        raise protocol.ProtocolError("quality revision/content delta is inconsistent")
    net_workspace_change = snapshot_comparison_complete and content_changed
    if activated != (mutation_epoch > 0):
        raise protocol.ProtocolError("quality activation differs from its mutation epoch")
    if activated != bool(changed_kinds):
        raise protocol.ProtocolError("quality changed kinds differ from its activation state")
    if not activated:
        expected_missing = ["net_workspace_change"]
        if not snapshot_comparison_complete:
            expected_missing.append("complete_workspace_snapshot")
        if (
            decision != "incomplete"
            or satisfied
            or phase != "awaiting_change"
            or net_workspace_change
            or list(required) != ["net_workspace_change"]
            or list(missing) != expected_missing
        ):
            raise protocol.ProtocolError("awaiting-change quality report semantics are inconsistent")
    elif not net_workspace_change:
        expected_missing = ["net_workspace_change"]
        if not snapshot_comparison_complete:
            expected_missing.append("complete_workspace_snapshot")
        if (
            decision != "incomplete"
            or satisfied
            or phase != "no_net_change"
            or list(required) != ["net_workspace_change"]
            or list(missing) != expected_missing
        ):
            raise protocol.ProtocolError("no-net-change quality report semantics are inconsistent")
    elif decision == "pass":
        if (
            not satisfied
            or phase != "passed"
            or missing
            or not snapshot_comparison_complete
            or mutation_epoch == 0
            or not changed_kinds
        ):
            raise protocol.ProtocolError("passing quality report semantics are inconsistent")
    elif decision == "incomplete":
        if (
            satisfied
            or phase not in {"dirty", "validation_failed"}
            or not missing
            or mutation_epoch == 0
            or not changed_kinds
        ):
            raise protocol.ProtocolError("incomplete quality report semantics are inconsistent")
    else:
        raise protocol.ProtocolError("activated quality report cannot be not_applicable")
    if activated and net_workspace_change:
        expected_required = ["diff"] if changed_kinds == ["docs"] else ["test_or_build"]
        if require_net_workspace_change:
            expected_required.insert(0, "net_workspace_change")
        expected_missing = []
        validation_required = expected_required[-1]
        if validation_required == "diff" and validation_counts["diff"] == 0:
            expected_missing.append("diff")
        if validation_required == "test_or_build" and not (validation_counts["test"] or validation_counts["build"]):
            expected_missing.append("test_or_build")
        if not snapshot_comparison_complete:
            expected_missing.append("complete_workspace_snapshot")
        if list(required) != expected_required or list(missing) != expected_missing:
            raise protocol.ProtocolError("quality decision is not derivable from its obligations and evidence counts")
    return {
        "enabled": True,
        "effort": effort,
        "intent": intent,
        "decision": decision,
        "status": decision,
        "phase": phase,
        "activated": activated,
        "satisfied": satisfied,
        "require_net_workspace_change": require_net_workspace_change,
        "mutation_epoch": mutation_epoch,
        "request_sha256": request_sha256,
        "initial_revision_sha256": initial_revision,
        "current_revision_sha256": current_revision,
        "initial_snapshot_complete": raw["initial_snapshot_complete"],
        "initial_content_sha256": initial_content,
        "current_content_sha256": current_content,
        "snapshot": snapshot,
        "changed_kinds": list(changed_kinds),
        "required": list(required),
        "missing": list(missing),
        "validation_counts": validation_counts,
        "validate_invocations": validate_invocations,
        "recognized_validation_attempts": recognized_validation_attempts,
        "validation_attempts": validation_attempts,
        "misrouted_validation_commands": misrouted_validation_commands,
        "successful_reads": successful_reads,
    }


def _validate_normalized_quality_document(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact content-free Quality document persisted in a sidecar."""

    expected = {
        "enabled",
        "effort",
        "intent",
        "decision",
        "status",
        "phase",
        "activated",
        "satisfied",
        "require_net_workspace_change",
        "mutation_epoch",
        "request_sha256",
        "initial_revision_sha256",
        "current_revision_sha256",
        "initial_snapshot_complete",
        "initial_content_sha256",
        "current_content_sha256",
        "snapshot",
        "changed_kinds",
        "required",
        "missing",
        "validation_counts",
        "validate_invocations",
        "recognized_validation_attempts",
        "validation_attempts",
        "misrouted_validation_commands",
        "successful_reads",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected or not isinstance(raw.get("enabled"), bool):
        raise protocol.ProtocolError("normalized quality document fields are invalid")
    if raw["enabled"] is False:
        expected_disabled = {
            "enabled": False,
            "effort": "not_applicable",
            "intent": "not_applicable",
            "decision": "not_applicable",
            "status": "not_applicable",
            "phase": "experiment_disabled",
            "activated": False,
            "satisfied": True,
            "require_net_workspace_change": False,
            "mutation_epoch": 0,
            "request_sha256": None,
            "initial_revision_sha256": None,
            "current_revision_sha256": None,
            "initial_snapshot_complete": None,
            "initial_content_sha256": None,
            "current_content_sha256": None,
            "snapshot": None,
            "changed_kinds": [],
            "required": [],
            "missing": [],
            "validation_counts": {name: 0 for name in _VALIDATION_KINDS},
            "validate_invocations": 0,
            "recognized_validation_attempts": 0,
            "validation_attempts": 0,
            "misrouted_validation_commands": 0,
            "successful_reads": 0,
        }
        if dict(raw) != expected_disabled:
            raise protocol.ProtocolError("disabled quality document differs from its sealed sentinel")
        return expected_disabled

    # A model exception can occur before the quality tracker returns its normal
    # revision-bound report.  Preserve that absence as a unique, content-free
    # terminal sentinel; never fabricate a workspace snapshot or validation.
    if raw.get("phase") == "model_error":
        expected_model_error = {
            "enabled": True,
            "effort": "medium",
            "intent": "general",
            "decision": "incomplete",
            "status": "incomplete",
            "phase": "model_error",
            "activated": False,
            "satisfied": False,
            "require_net_workspace_change": True,
            "mutation_epoch": 0,
            "request_sha256": raw.get("request_sha256"),
            "initial_revision_sha256": None,
            "current_revision_sha256": None,
            "initial_snapshot_complete": None,
            "initial_content_sha256": None,
            "current_content_sha256": None,
            "snapshot": None,
            "changed_kinds": [],
            "required": [],
            "missing": [],
            "validation_counts": {name: 0 for name in _VALIDATION_KINDS},
            "validate_invocations": 0,
            "recognized_validation_attempts": 0,
            "validation_attempts": 0,
            "misrouted_validation_commands": 0,
            "successful_reads": 0,
        }
        request_sha256 = raw.get("request_sha256")
        if not isinstance(request_sha256, str):
            raise protocol.ProtocolError("model-error quality request digest is malformed")
        _require_sha256(request_sha256, "model-error quality request digest")
        if dict(raw) != expected_model_error:
            raise protocol.ProtocolError("model-error quality document differs from its sealed sentinel")
        return expected_model_error

    effort = raw["effort"]
    intent = raw["intent"]
    decision = raw["decision"]
    phase = raw["phase"]
    if effort != "medium" or effort not in _QUALITY_EFFORTS or intent not in _QUALITY_INTENTS:
        raise protocol.ProtocolError("normalized quality effort or intent is invalid")
    if decision not in _QUALITY_DECISIONS or raw["status"] != decision or phase not in _QUALITY_PHASES:
        raise protocol.ProtocolError("normalized quality decision, status, or phase is invalid")
    if any(
        not isinstance(raw[name], bool)
        for name in ("activated", "satisfied", "require_net_workspace_change", "initial_snapshot_complete")
    ):
        raise protocol.ProtocolError("normalized quality booleans are invalid")
    if raw["require_net_workspace_change"] is not True:
        raise protocol.ProtocolError("normalized gate_on quality lacks the mandatory net-change contract")
    mutation_epoch = _nonnegative_integer(raw["mutation_epoch"], "quality mutation_epoch", maximum=128)
    for name in (
        "request_sha256",
        "initial_revision_sha256",
        "current_revision_sha256",
        "initial_content_sha256",
        "current_content_sha256",
    ):
        value = raw[name]
        if not isinstance(value, str):
            raise protocol.ProtocolError(f"normalized quality {name} is malformed")
        _require_sha256(value, f"normalized quality {name}")
    snapshot = _validate_snapshot_observation(raw["snapshot"])
    if snapshot["revision_sha256"] != raw["current_revision_sha256"]:
        raise protocol.ProtocolError("normalized quality snapshot revision is inconsistent")
    changed_kinds = raw["changed_kinds"]
    if (
        not isinstance(changed_kinds, list)
        or changed_kinds != sorted(set(changed_kinds))
        or any(item not in {"code", "docs"} for item in changed_kinds)
    ):
        raise protocol.ProtocolError("normalized quality changed_kinds is invalid")
    required = raw["required"]
    missing = raw["missing"]
    if (
        not isinstance(required, list)
        or not isinstance(missing, list)
        or required != list(dict.fromkeys(required))
        or missing != list(dict.fromkeys(missing))
        or any(item not in _QUALITY_OBLIGATIONS for item in (*required, *missing))
        or not set(missing).issubset(set(required) | {"complete_workspace_snapshot"})
    ):
        raise protocol.ProtocolError("normalized quality obligations are invalid")
    counts = raw["validation_counts"]
    if not isinstance(counts, Mapping) or set(counts) != set(_VALIDATION_KINDS):
        raise protocol.ProtocolError("normalized quality validation counts are invalid")
    validated_counts = {
        name: _nonnegative_integer(counts[name], f"quality {name} validations", maximum=protocol.MAX_TOOL_CALLS_PER_ARM)
        for name in _VALIDATION_KINDS
    }
    validate_invocations = _nonnegative_integer(
        raw["validate_invocations"], "quality validate invocations", maximum=protocol.MAX_TOOL_CALLS_PER_ARM
    )
    recognized_attempts = _nonnegative_integer(
        raw["recognized_validation_attempts"],
        "quality recognized validation attempts",
        maximum=protocol.MAX_TOOL_CALLS_PER_ARM,
    )
    attempts = _nonnegative_integer(
        raw["validation_attempts"], "quality validation-attempt alias", maximum=protocol.MAX_TOOL_CALLS_PER_ARM
    )
    misrouted_commands = _nonnegative_integer(
        raw["misrouted_validation_commands"],
        "quality misrouted validation commands",
        maximum=protocol.MAX_TOOL_CALLS_PER_ARM,
    )
    successful_reads = _nonnegative_integer(
        raw["successful_reads"], "quality successful reads", maximum=protocol.MAX_TOOL_CALLS_PER_ARM
    )
    if attempts != recognized_attempts:
        raise protocol.ProtocolError("normalized validation-attempt alias differs from recognized attempts")
    if recognized_attempts > validate_invocations:
        raise protocol.ProtocolError("normalized recognized attempts exceed validate invocations")
    if sum(validated_counts.values()) > recognized_attempts:
        raise protocol.ProtocolError("normalized quality successes exceed validation attempts")
    activated = raw["activated"]
    satisfied = raw["satisfied"]
    require_net_workspace_change = raw["require_net_workspace_change"]
    snapshot_comparison_complete = raw["initial_snapshot_complete"] and snapshot["complete"]
    content_changed = raw["initial_content_sha256"] != raw["current_content_sha256"]
    revision_changed = raw["initial_revision_sha256"] != raw["current_revision_sha256"]
    if snapshot_comparison_complete and content_changed and not revision_changed:
        raise protocol.ProtocolError("normalized quality revision/content delta is inconsistent")
    net_workspace_change = snapshot_comparison_complete and content_changed
    if activated != (mutation_epoch > 0):
        raise protocol.ProtocolError("normalized quality activation differs from its mutation epoch")
    if activated != bool(changed_kinds):
        raise protocol.ProtocolError("normalized quality changed kinds differ from activation")
    if not activated:
        expected_missing = ["net_workspace_change"]
        if not snapshot_comparison_complete:
            expected_missing.append("complete_workspace_snapshot")
        if (
            decision != "incomplete"
            or satisfied
            or phase != "awaiting_change"
            or net_workspace_change
            or list(required) != ["net_workspace_change"]
            or list(missing) != expected_missing
        ):
            raise protocol.ProtocolError("normalized awaiting-change quality semantics are inconsistent")
    elif not net_workspace_change:
        expected_missing = ["net_workspace_change"]
        if not snapshot_comparison_complete:
            expected_missing.append("complete_workspace_snapshot")
        if (
            decision != "incomplete"
            or satisfied
            or phase != "no_net_change"
            or list(required) != ["net_workspace_change"]
            or list(missing) != expected_missing
        ):
            raise protocol.ProtocolError("normalized no-net-change quality semantics are inconsistent")
    elif decision == "pass":
        if (
            not satisfied
            or phase != "passed"
            or missing
            or not snapshot_comparison_complete
            or mutation_epoch == 0
            or not changed_kinds
        ):
            raise protocol.ProtocolError("normalized passing quality semantics are inconsistent")
    elif decision == "incomplete":
        if (
            satisfied
            or phase not in {"dirty", "validation_failed"}
            or not missing
            or mutation_epoch == 0
            or not changed_kinds
        ):
            raise protocol.ProtocolError("normalized incomplete quality semantics are inconsistent")
    else:
        raise protocol.ProtocolError("normalized activated quality decision is inconsistent")
    if activated and net_workspace_change:
        expected_required = ["diff"] if changed_kinds == ["docs"] else ["test_or_build"]
        if require_net_workspace_change:
            expected_required.insert(0, "net_workspace_change")
        expected_missing = []
        validation_required = expected_required[-1]
        if validation_required == "diff" and validated_counts["diff"] == 0:
            expected_missing.append("diff")
        if validation_required == "test_or_build" and not (validated_counts["test"] or validated_counts["build"]):
            expected_missing.append("test_or_build")
        if not snapshot_comparison_complete:
            expected_missing.append("complete_workspace_snapshot")
        if list(required) != expected_required or list(missing) != expected_missing:
            raise protocol.ProtocolError("normalized quality decision is not derivable from its evidence")
    return {
        **dict(raw),
        "snapshot": snapshot,
        "validation_counts": validated_counts,
        "changed_kinds": list(changed_kinds),
        "required": list(required),
        "missing": list(missing),
        "validate_invocations": validate_invocations,
        "recognized_validation_attempts": recognized_attempts,
        "validation_attempts": attempts,
        "misrouted_validation_commands": misrouted_commands,
        "successful_reads": successful_reads,
    }


def _validate_turn_and_topology(
    raw: Mapping[str, Any],
    rounds: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "terminal_reason",
        "budget_exhaustion_kind",
        "trajectory_complete",
        "counters_observed",
        "tool_telemetry_complete",
        "wall_elapsed_ns",
        "completion_tokens",
        "tool_calls",
        "tool_result_chars",
        "status",
        "quality_gate_decision",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise protocol.ProtocolError("turn telemetry fields differ from the sealed schema")
    terminal_reason = raw["terminal_reason"]
    budget_kind = raw["budget_exhaustion_kind"]
    status = raw["status"]
    quality_decision = raw["quality_gate_decision"]
    if terminal_reason not in _TERMINAL_REASONS or budget_kind not in _BUDGET_EXHAUSTION_KINDS:
        raise protocol.ProtocolError("turn terminal or budget reason is outside the sealed vocabulary")
    if status not in {"completed", "incomplete", "model_error", "timeout"}:
        raise protocol.ProtocolError("turn status is outside the sealed vocabulary")
    if quality_decision not in {"not_applicable", "satisfied", "incomplete"}:
        raise protocol.ProtocolError("checkpoint quality decision is outside the sealed vocabulary")
    if any(
        not isinstance(raw[name], bool)
        for name in ("trajectory_complete", "counters_observed", "tool_telemetry_complete")
    ):
        raise protocol.ProtocolError("turn telemetry-completeness flags must be boolean")
    wall_ns = _nonnegative_integer(
        raw["wall_elapsed_ns"],
        "turn wall_elapsed_ns",
        maximum=int(protocol.MAX_AGENT_WALL_SECONDS * 1_000_000_000) + _EXECUTOR_WALL_OVERHEAD_NS,
    )
    completion_tokens = _nonnegative_integer(
        raw["completion_tokens"],
        "turn completion_tokens",
        maximum=protocol.MAX_OUTPUT_TOKENS_PER_ARM,
    )
    tool_calls = _nonnegative_integer(raw["tool_calls"], "turn tool_calls", maximum=protocol.MAX_TOOL_CALLS_PER_ARM)
    tool_result_chars = _nonnegative_integer(raw["tool_result_chars"], "turn tool_result_chars", maximum=100_000)
    if completion_tokens != sum(item["completion_tokens"] for item in rounds) or tool_calls != len(tools):
        raise protocol.ProtocolError("turn token/tool totals differ from raw streams")
    if raw["trajectory_complete"] and raw["tool_telemetry_complete"] != all(
        item["telemetry_complete"] for item in tools
    ):
        raise protocol.ProtocolError("turn tool_telemetry_complete differs from raw tool traces")
    if quality["enabled"] and quality["phase"] != "model_error":
        successful_reads = sum(
            item["tool_name"] == "read" and item["allowed"] and item["outcome"] == "ok" for item in tools
        )
        validate_calls = sum(item["tool_name"] == "validate" for item in tools)
        bash_calls = sum(item["tool_name"] == "bash" for item in tools)
        successful_validations = sum(
            item["tool_name"] == "validate" and item["allowed"] and item["outcome"] == "ok" for item in tools
        )
        if quality["successful_reads"] != successful_reads:
            raise protocol.ProtocolError("quality successful-read count differs from admitted tool telemetry")
        if quality["validate_invocations"] != validate_calls:
            raise protocol.ProtocolError("quality validate invocations differ from admitted validate tool calls")
        if quality["recognized_validation_attempts"] > validate_calls:
            raise protocol.ProtocolError("quality recognized attempts exceed admitted validate tool calls")
        if quality["misrouted_validation_commands"] > bash_calls:
            raise protocol.ProtocolError("quality misrouted validation commands exceed admitted bash tool calls")
        if sum(quality["validation_counts"].values()) > successful_validations:
            raise protocol.ProtocolError("quality validation successes exceed successful validate tool traces")
    round_total_ns = sum(math.ceil(item["total_time_s"] * 1_000_000_000) for item in rounds)
    tool_total_ns = sum(item["duration_ns"] for item in tools)
    if round_total_ns + tool_total_ns > wall_ns:
        raise protocol.ProtocolError("round and tool total durations exceed complete arm wall time")
    timeout_tools = [item for item in tools if item["outcome"] == "timeout"]
    deadline_rounds = [item for item in rounds if item["deadline_hit"]]
    if terminal_reason == "model_error":
        if (
            status != "model_error"
            or rounds
            or tools
            or budget_kind != "none"
            or raw["trajectory_complete"] is not False
            or raw["counters_observed"] is not False
            or raw["tool_telemetry_complete"] is not False
            or (quality["enabled"] and quality["phase"] != "model_error")
        ):
            raise protocol.ProtocolError("model_error terminal topology is inconsistent")
    elif raw["trajectory_complete"] is not True or raw["counters_observed"] is not True:
        raise protocol.ProtocolError("structured terminal outcome lacks a complete observed trajectory")
    elif terminal_reason == "tool_timeout":
        if (
            status != "timeout"
            or len(timeout_tools) != 1
            or timeout_tools[0] is not tools[-1]
            or timeout_tools[0]["round_index"] != rounds[-1]["round_index"]
            or deadline_rounds
        ):
            raise protocol.ProtocolError("tool_timeout terminal topology is inconsistent")
    elif timeout_tools:
        raise protocol.ProtocolError("non-timeout turn contains a tool timeout")
    if deadline_rounds:
        if (
            len(deadline_rounds) != 1
            or deadline_rounds[0] is not rounds[-1]
            or terminal_reason not in {"budget_exhausted", "budget_finalization", "quality_incomplete"}
            or any(item["round_index"] >= rounds[-1]["round_index"] for item in tools)
        ):
            raise protocol.ProtocolError("model deadline topology is inconsistent")
    expected_status = (
        "model_error"
        if terminal_reason == "model_error"
        else (
            "timeout"
            if terminal_reason == "tool_timeout"
            else ("completed" if terminal_reason == "model_final" and quality["satisfied"] else "incomplete")
        )
    )
    expected_quality_decision = (
        "not_applicable" if not quality["enabled"] else ("satisfied" if quality["satisfied"] else "incomplete")
    )
    if status != expected_status or quality_decision != expected_quality_decision:
        raise protocol.ProtocolError(
            "turn status or checkpoint quality decision is not derivable from terminal evidence"
        )
    if terminal_reason == "model_final" and tools and tools[-1]["round_index"] >= rounds[-1]["round_index"]:
        raise protocol.ProtocolError("completed model_final must end with a tool-free final round")
    if (
        quality["enabled"]
        and not quality["satisfied"]
        and terminal_reason not in {"model_error", "quality_incomplete", "tool_timeout"}
    ):
        raise protocol.ProtocolError("unsatisfied quality gate must produce quality_incomplete termination")
    if terminal_reason == "quality_incomplete" and (not quality["enabled"] or quality["satisfied"]):
        raise protocol.ProtocolError("quality_incomplete termination lacks an unsatisfied gate")
    if terminal_reason in {"budget_exhausted", "budget_finalization"} and budget_kind == "none":
        raise protocol.ProtocolError("turn budget classification disagrees with terminal reason")
    if (
        terminal_reason not in {"budget_exhausted", "budget_finalization", "quality_incomplete"}
        and budget_kind != "none"
    ):
        raise protocol.ProtocolError("non-budget turn unexpectedly carries a budget classification")
    if budget_kind == "completion_tokens" and completion_tokens < protocol.MAX_OUTPUT_TOKENS_PER_ARM:
        raise protocol.ProtocolError("completion-token exhaustion is below its frozen limit")
    if budget_kind == "context_tokens" and (
        rounds[-1]["prompt_tokens"] + rounds[-1]["completion_tokens"] < TARGET_CONTEXT_TOKENS
    ):
        raise protocol.ProtocolError("context exhaustion is below its frozen limit")
    if budget_kind == "model_rounds" and len(rounds) != TARGET_MAX_ROUNDS:
        raise protocol.ProtocolError("model-round exhaustion differs from its frozen limit")
    if budget_kind == "tool_calls" and tool_calls != protocol.MAX_TOOL_CALLS_PER_ARM:
        raise protocol.ProtocolError("tool-call exhaustion differs from its frozen limit")
    if budget_kind == "tool_output" and tool_result_chars != 100_000:
        raise protocol.ProtocolError("tool-output exhaustion differs from its frozen limit")
    if budget_kind == "wall_time" and wall_ns < int(protocol.MAX_AGENT_WALL_SECONDS * 1_000_000_000):
        raise protocol.ProtocolError("wall-time exhaustion is below its frozen limit")
    return dict(raw)


@dataclass(frozen=True)
class ValidatedArmTelemetry:
    """Canonical content-free traces admitted only by the native executor."""

    payload: bytes = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _VALIDATED_TELEMETRY_SEAL:
            raise protocol.ProtocolError("arm telemetry must be validated by the native executor")
        try:
            raw = json.loads(self.payload)
        except (UnicodeDecodeError, ValueError) as exc:
            raise protocol.ProtocolError("validated arm telemetry is not JSON") from exc
        if protocol.canonical_json_bytes(raw) != self.payload or set(raw) != {
            "turn",
            "quality_gate",
            "rounds",
            "tools",
        }:
            raise protocol.ProtocolError("validated arm telemetry is not canonical")
        self._validated_document(raw)

    @staticmethod
    def _validated_document(
        raw: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        raw_rounds = raw.get("rounds")
        raw_tools = raw.get("tools")
        if not isinstance(raw_rounds, list):
            raise protocol.ProtocolError("arm round telemetry must be a list")
        if not isinstance(raw_tools, list):
            raise protocol.ProtocolError("arm tool telemetry must be a list")
        rounds = tuple(_validate_round_trace_record(item) for item in raw_rounds)
        tools = tuple(_validate_tool_trace_record(item) for item in raw_tools)
        if tuple(item["round_index"] for item in rounds) != tuple(range(len(rounds))):
            raise protocol.ProtocolError("round telemetry indices must be contiguous and ordered")
        if tuple(item["sequence"] for item in tools) != tuple(range(len(tools))):
            raise protocol.ProtocolError("tool telemetry sequence must be contiguous and ordered")
        tool_rounds = tuple(item["round_index"] for item in tools)
        if any(index >= len(rounds) for index in tool_rounds) or tool_rounds != tuple(sorted(tool_rounds)):
            raise protocol.ProtocolError("tool telemetry round references are invalid or reordered")
        quality = raw.get("quality_gate")
        turn = raw.get("turn")
        if not isinstance(quality, Mapping) or not isinstance(turn, Mapping):
            raise protocol.ProtocolError("turn or quality telemetry is malformed")
        validated_quality = _validate_normalized_quality_document(quality)
        validated_turn = _validate_turn_and_topology(turn, rounds, tools, validated_quality)
        return validated_turn, validated_quality, rounds, tools

    @classmethod
    def from_result(
        cls,
        result: Any,
        *,
        quality_gate_enabled: bool,
        wall_elapsed_s: float,
    ) -> "ValidatedArmTelemetry":
        rounds = [{name: getattr(trace, name, None) for name in _ROUND_TRACE_FIELDS} for trace in result.rounds]
        tools = [{name: getattr(trace, name, None) for name in _TOOL_TRACE_FIELDS} for trace in result.tool_events]
        validated_rounds = tuple(_validate_round_trace_record(item) for item in rounds)
        validated_tools = tuple(_validate_tool_trace_record(item) for item in tools)
        quality = _validate_quality_gate_report(
            getattr(result, "quality_gate", None),
            quality_gate_enabled=quality_gate_enabled,
        )
        terminal_reason = getattr(result, "terminal_reason", None)
        if terminal_reason not in _TERMINAL_REASONS:
            raise protocol.ProtocolError("agent terminal reason is outside the sealed vocabulary")
        reported_wall_seconds = _finite_nonnegative_number(getattr(result, "wall_time_s", None), "agent wall time")
        wall_seconds = _finite_nonnegative_number(wall_elapsed_s, "complete executor wall time")
        if reported_wall_seconds > wall_seconds:
            raise protocol.ProtocolError("agent-reported wall time exceeds complete executor wall time")
        wall_ns = math.ceil(wall_seconds * 1_000_000_000)
        budget_kind = _budget_exhaustion_kind(getattr(result, "budget_exhaustion", None), terminal_reason)
        status = (
            "timeout"
            if terminal_reason == "tool_timeout"
            else ("completed" if terminal_reason == "model_final" and quality["satisfied"] else "incomplete")
        )
        quality_decision = (
            "not_applicable" if not quality_gate_enabled else ("satisfied" if quality["satisfied"] else "incomplete")
        )
        turn = {
            "terminal_reason": terminal_reason,
            "budget_exhaustion_kind": budget_kind,
            "trajectory_complete": True,
            "counters_observed": True,
            "tool_telemetry_complete": getattr(result, "tool_telemetry_complete", None),
            "wall_elapsed_ns": wall_ns,
            "completion_tokens": getattr(result, "completion_tokens", None),
            "tool_calls": getattr(result, "tool_calls", None),
            "tool_result_chars": getattr(result, "tool_result_chars", None),
            "status": status,
            "quality_gate_decision": quality_decision,
        }
        validated_turn = _validate_turn_and_topology(turn, validated_rounds, validated_tools, quality)
        return cls(
            payload=protocol.canonical_json_bytes(
                {
                    "turn": validated_turn,
                    "quality_gate": quality,
                    "rounds": list(validated_rounds),
                    "tools": list(validated_tools),
                }
            ),
            _seal=_VALIDATED_TELEMETRY_SEAL,
        )

    @classmethod
    def for_model_error(
        cls,
        *,
        quality_gate_enabled: bool,
        request_sha256: str,
        wall_elapsed_s: float,
    ) -> "ValidatedArmTelemetry":
        """Seal a content-free terminal record when model execution raises.

        This constructor is intentionally separate from ``from_result``: no
        round, tool, quality snapshot, or token counter is invented after an
        unstructured model exception.
        """

        _require_sha256(request_sha256, "model-error request digest")
        wall_seconds = _finite_nonnegative_number(wall_elapsed_s, "complete executor wall time")
        wall_ns = math.ceil(wall_seconds * 1_000_000_000)
        if quality_gate_enabled:
            quality = {
                "enabled": True,
                "effort": "medium",
                "intent": "general",
                "decision": "incomplete",
                "status": "incomplete",
                "phase": "model_error",
                "activated": False,
                "satisfied": False,
                "require_net_workspace_change": True,
                "mutation_epoch": 0,
                "request_sha256": request_sha256,
                "initial_revision_sha256": None,
                "current_revision_sha256": None,
                "initial_snapshot_complete": None,
                "initial_content_sha256": None,
                "current_content_sha256": None,
                "snapshot": None,
                "changed_kinds": [],
                "required": [],
                "missing": [],
                "validation_counts": {name: 0 for name in _VALIDATION_KINDS},
                "validate_invocations": 0,
                "recognized_validation_attempts": 0,
                "validation_attempts": 0,
                "misrouted_validation_commands": 0,
                "successful_reads": 0,
            }
        else:
            quality = _validate_quality_gate_report(None, quality_gate_enabled=False)
        turn = {
            "terminal_reason": "model_error",
            "budget_exhaustion_kind": "none",
            "trajectory_complete": False,
            "counters_observed": False,
            "tool_telemetry_complete": False,
            "wall_elapsed_ns": wall_ns,
            "completion_tokens": 0,
            "tool_calls": 0,
            "tool_result_chars": 0,
            "status": "model_error",
            "quality_gate_decision": "incomplete" if quality_gate_enabled else "not_applicable",
        }
        validated_quality = _validate_normalized_quality_document(quality)
        validated_turn = _validate_turn_and_topology(turn, (), (), validated_quality)
        return cls(
            payload=protocol.canonical_json_bytes(
                {
                    "turn": validated_turn,
                    "quality_gate": validated_quality,
                    "rounds": [],
                    "tools": [],
                }
            ),
            _seal=_VALIDATED_TELEMETRY_SEAL,
        )

    def document(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        return self._validated_document(json.loads(self.payload))


@dataclass(frozen=True)
class ArmRunOutcome:
    """Content-free terminal outcome returned by an executor."""

    status: str
    quality_gate_decision: str
    output_tokens: int = 0
    tool_calls: int = 0
    wall_seconds: float = 0.0
    telemetry: ValidatedArmTelemetry | None = field(default=None, repr=False, compare=False)


class ArmExecutor(Protocol):
    def __call__(self, request: ArmRunRequest) -> ArmRunOutcome: ...


class WorkspaceFactory(Protocol):
    def __call__(
        self,
        *,
        instance: protocol.PublicInstance,
        entry: protocol.ScheduleEntry,
        destination: Path,
    ) -> ArmWorkspace: ...


def validate_target_only_tier(tier: Any) -> None:
    """Enforce the frozen target-only AR control before model loading."""

    expected = {
        "drafter_backend": "target_ar",
        "context_window": TARGET_CONTEXT_TOKENS,
        "max_output_tokens": TARGET_MAX_OUTPUT_TOKENS_PER_ROUND,
        "tq_bits": TARGET_TQ_BITS,
        "pq_bits": TARGET_PQ_BITS,
        "bmp_paths": TARGET_BMP_PATHS,
        "ddtree_budget": TARGET_DDTREE_BUDGET,
        "temperature": TARGET_TEMPERATURE,
        "top_p": TARGET_TOP_P,
        "top_k": TARGET_TOP_K,
    }
    differences = {
        name: (getattr(tier, name, None), wanted)
        for name, wanted in expected.items()
        if getattr(tier, name, None) != wanted
    }
    if differences:
        raise protocol.ProtocolError(f"target-only 27B tier differs from frozen controls: {differences}")


def build_identical_tool_surface(
    agent_module: Any | None = None,
) -> tuple[Mapping[str, Any], tuple[dict[str, Any], ...], str]:
    """Build one immutable five-tool surface shared byte-for-byte by both arms."""

    if agent_module is None:
        from mio import agent as agent_module

    try:
        registry = {}
        for name in TOOL_SURFACE:
            definition = dict(agent_module.AGENT_TOOLS[name])
            definition["args"] = tuple(definition.get("args", ()))
            registry[name] = MappingProxyType(definition)
        raw_specs = {
            item["function"]["name"]: item
            for item in agent_module.AGENT_TOOLS_SPEC
            if isinstance(item, dict) and isinstance(item.get("function"), dict)
        }
        specs = tuple(copy.deepcopy(raw_specs[name]) for name in TOOL_SURFACE)
    except (KeyError, TypeError) as exc:
        raise protocol.ProtocolError("native agent lacks the frozen SWE-bench tool surface") from exc
    if tuple(registry) != TOOL_SURFACE or tuple(item["function"]["name"] for item in specs) != TOOL_SURFACE:
        raise protocol.ProtocolError("SWE-bench tool surface order or schema differs")
    document = _tool_surface_document(registry, specs)
    digest = protocol.sha256_bytes(protocol.canonical_json_bytes(document))
    return MappingProxyType(registry), specs, digest


def _tool_surface_document(
    registry: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dispatcher = []
    for name in TOOL_SURFACE:
        definition = registry[name]
        function = definition.get("fn")
        permission = definition.get("permission")
        dispatcher.append(
            {
                "name": name,
                "args": list(definition.get("args", ())),
                "permission": getattr(permission, "value", None),
                "inject_policy": bool(definition.get("inject_policy", False)),
                "callable_module": str(getattr(function, "__module__", "")),
                "callable_qualname": str(getattr(function, "__qualname__", "")),
            }
        )
    return {
        "names": list(TOOL_SURFACE),
        "dispatcher": dispatcher,
        "specs": list(specs),
    }


def factor_document(tool_surface_sha256: str) -> dict[str, Any]:
    """Describe the single intended experimental factor and all hard equalities."""

    _require_sha256(tool_surface_sha256, "tool surface digest")
    return {
        "schema": f"{GENERATION_SCHEMA}.factor",
        "experimental_factor": "mandatory_coding_quality_gate",
        "common": {
            "model_identity": protocol.EXPECTED_MODEL_IDENTITY,
            "model_role": "target_only_autoregressive_control",
            "drafter_backend": "target_ar",
            "dflash": False,
            "dspark": False,
            "bmp": False,
            "turboquant": False,
            "polarquant": False,
            "temperature": TARGET_TEMPERATURE,
            "top_p": TARGET_TOP_P,
            "top_k": TARGET_TOP_K,
            "context_tokens": TARGET_CONTEXT_TOKENS,
            "max_output_tokens_per_round": TARGET_MAX_OUTPUT_TOKENS_PER_ROUND,
            "max_output_tokens_per_arm": protocol.MAX_OUTPUT_TOKENS_PER_ARM,
            "max_rounds_per_arm": TARGET_MAX_ROUNDS,
            "max_tool_calls_per_arm": protocol.MAX_TOOL_CALLS_PER_ARM,
            "max_wall_seconds_per_arm": protocol.MAX_AGENT_WALL_SECONDS,
            "command_timeout_seconds": FROZEN_COMMAND_TIMEOUT_SECONDS,
            "tool_names": list(TOOL_SURFACE),
            "tool_surface_sha256": tool_surface_sha256,
            "instruction_template_sha256": protocol.sha256_bytes(MODEL_INSTRUCTION_TEMPLATE.encode("utf-8")),
            "seed_policy": (
                "pair_seed_recorded_in_runner_request_but_not_claimed_as_native_engine_rng_enforcement;"
                "decode_is_frozen_greedy"
            ),
            "network": False,
            "fresh_workspace_and_conversation_per_arm": True,
            "fresh_empty_runner_cache_directory_allocated_per_arm": True,
            "runtime_cache_directory_enforcement": False,
        },
        "arms": {
            "gate_off": {
                "quality_gate_enabled": False,
                "quality_gate_effort": "not_applicable",
            },
            "gate_on": {
                "quality_gate_enabled": True,
                "quality_gate_effort": "medium",
            },
        },
        "allowed_difference": "quality_gate_enabled_and_its_preregistered_feedback",
    }


def factor_digest(tool_surface_sha256: str) -> str:
    return protocol.sha256_bytes(protocol.canonical_json_bytes(factor_document(tool_surface_sha256)))


def _implementation_identity(value: Any) -> dict[str, str]:
    target = value if isinstance(value, type) else type(value)
    if callable(value) and getattr(value, "__module__", None) and getattr(value, "__qualname__", None):
        target = value
    return {
        "module": str(getattr(target, "__module__", "")),
        "qualname": str(getattr(target, "__qualname__", "")),
    }


def build_run_header(
    *,
    schedule_document: Mapping[str, Any],
    schedule: Sequence[protocol.ScheduleEntry],
    binding: GenerationBinding,
    tool_surface_sha256: str,
    executor: ArmExecutor,
    workspace_factory: WorkspaceFactory,
    tier_config: Any,
    sealed_runtime_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind smoke execution inputs before the first pair is admitted.

    Caller-supplied identities are marked smoke-only.  The automatic path
    records recomputable preflight digests, but does not claim clean-subprocess
    or in-memory end-to-end provenance.  Resume and receipt creation cannot
    re-declare a different factor or binding.
    """

    factor = factor_document(tool_surface_sha256)
    if sealed_runtime_manifest is None:
        artifact_audit = {
            "portable": False,
            "status": "legacy_or_dependency_injected_layout_without_persisted_runtime_and_telemetry",
            "cross_process_verifiable": False,
            "current_environment_reattestation_is_separate": True,
        }
    else:
        if binding.binding_source != AUTOMATIC_BINDING_SOURCE:
            raise protocol.ProtocolError("portable artifact audit requires an automatic generation binding")
        if sealed_runtime_manifest.get("sha256") != binding.runtime_digest:
            raise protocol.ProtocolError("run header runtime manifest differs from the generation binding")
        artifact_audit = {
            "portable": True,
            "status": "sealed_original_artifacts_cross_process_verifiable",
            "cross_process_verifiable": True,
            "current_environment_reattestation_is_separate": True,
            "runtime_manifest": dict(sealed_runtime_manifest),
            "arm_telemetry_schema": ARM_TELEMETRY_SCHEMA,
            "telemetry_manifest_schema": TELEMETRY_MANIFEST_SCHEMA,
            "content_free_telemetry": True,
        }
    loaded_target_binding = _executor_model_binding_document(binding, executor, tier_config)
    if sealed_runtime_manifest is not None:
        if loaded_target_binding.get("raw_target_telemetry_required") is not True:
            raise protocol.ProtocolError("portable run header requires native raw target telemetry")
        loaded_target_binding["raw_target_telemetry_receipt_bound"] = True
    return {
        "schema": GENERATION_RUN_HEADER_SCHEMA,
        "evidence_class": str(schedule_document["evidence_class"]),
        "confirmatory_evidence_admissible": False,
        "confirmatory_blockers": list(CONFIRMATORY_BLOCKERS),
        "preregistration_sha256": protocol.preregistration_digest(),
        "schedule_sha256": protocol.schedule_digest(schedule),
        "schedule_document_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(dict(schedule_document))),
        "dataset_public_snapshot_sha256": str(schedule_document["dataset_public_snapshot_sha256"]),
        "generation_binding": binding.as_dict(),
        "generation_binding_attestation": binding.attestation_dict(),
        "loaded_target_binding": loaded_target_binding,
        "tool_surface_sha256": tool_surface_sha256,
        "factor_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(factor)),
        "factor": factor,
        "runner_source_sha256": protocol.sha256_file(Path(__file__)),
        "executor": _implementation_identity(executor),
        "workspace_factory": _implementation_identity(workspace_factory),
        "sealed_artifact_audit": artifact_audit,
    }


def _model_instruction(instance: protocol.PublicInstance) -> str:
    return MODEL_INSTRUCTION_TEMPLATE.format(
        repo=instance.repo,
        base_commit=instance.base_commit,
        problem_statement=instance.problem_statement,
    )


def _arm_seed(entry: protocol.ScheduleEntry) -> int:
    # Both arms in a pair receive the same seed.  Greedy decoding makes this a
    # redundant control, but recording it closes a future sampling ambiguity.
    material = f"mio-swebench-arm-seed-v1\0{protocol.SCHEDULE_SEED}\0{entry.instance_digest}"
    return int(protocol.sha256_bytes(material.encode())[:16], 16)


class ExternalGitWorkspaceFactory:
    """Materialize clean local clones with Git metadata outside model reach.

    Workspaces are deliberately retained with their attempts for smoke-debug
    auditability.  This implementation is not storage-bounded for 1,000 arms;
    confirmatory generation remains blocked until v2 freezes an immutable
    shared-object or attested post-capture cleanup design.
    """

    def __init__(self, source_for: Callable[[protocol.PublicInstance], Path]) -> None:
        self.source_for = source_for

    @staticmethod
    def _clone(source: Path, workspace: Path) -> None:
        environment = {
            "HOME": "/var/empty",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "TMPDIR": "/tmp",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        result = subprocess.run(
            [
                protocol._trusted_git(),
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "clone",
                "--quiet",
                "--no-hardlinks",
                "--no-checkout",
                "--",
                str(source),
                str(workspace),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300,
            env=environment,
        )
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace")[:500]
            raise protocol.ProtocolError(f"trusted local clone failed: {detail}")

    def __call__(
        self,
        *,
        instance: protocol.PublicInstance,
        entry: protocol.ScheduleEntry,
        destination: Path,
    ) -> ArmWorkspace:
        del entry
        source = protocol.require_private_path(self.source_for(instance), must_exist=True)
        if not source.is_dir():
            raise protocol.ProtocolError("trusted source mirror must be a local directory")
        destination = _mkdir_private(destination)
        workspace = destination / "workspace"
        cache_directory = _mkdir_private(destination / "cache")
        metadata_parent = _mkdir_private(destination / "trusted-metadata")
        git_directory = metadata_parent / "git"
        self._clone(source, workspace)
        embedded_git = workspace / ".git"
        if not embedded_git.is_dir() or embedded_git.is_symlink():
            raise protocol.ProtocolError("trusted clone did not create ordinary Git metadata")
        os.replace(embedded_git, git_directory)
        os.chmod(git_directory, 0o700)
        protocol._run_git(
            workspace,
            ["checkout", "--force", "--detach", instance.base_commit, "--"],
            timeout_s=300,
            git_directory=git_directory,
            work_tree=workspace,
        )
        result = ArmWorkspace(workspace, git_directory, cache_directory).validated()
        head = (
            protocol._run_git(
                workspace,
                ["rev-parse", "HEAD"],
                git_directory=git_directory,
                work_tree=workspace,
            )
            .decode()
            .strip()
        )
        if head != instance.base_commit:
            raise protocol.ProtocolError("fresh workspace does not match dataset base_commit")
        return result


@dataclass(frozen=True)
class GenerationLayout:
    """Private layout separating attempts, canonical output, ledger, and receipt."""

    root: Path
    attempts: Path
    canonical: Path
    telemetry: Path
    ledger: Path
    run_header: Path
    runtime_manifest: Path
    artifact_profile: Path
    receipt: Path
    portable_artifacts: bool

    def validated(self) -> "GenerationLayout":
        root = protocol.require_private_directory(self.root)
        attempts = protocol.require_private_directory(self.attempts)
        canonical = protocol.require_private_directory(self.canonical)
        if attempts.parent != root or canonical.parent != root:
            raise protocol.ProtocolError("generation layout directories must be direct private children")
        if not isinstance(self.portable_artifacts, bool):
            raise protocol.ProtocolError("generation layout portability flag must be boolean")
        if self.portable_artifacts:
            telemetry = protocol.require_private_directory(self.telemetry)
            if telemetry.parent != root:
                raise protocol.ProtocolError("telemetry directory must be a direct private layout child")
            profile_payload = protocol._read_immutable_file(self.artifact_profile)
            assert profile_payload is not None
            try:
                profile = json.loads(profile_payload)
            except (UnicodeDecodeError, ValueError) as exc:
                raise protocol.ProtocolError("portable layout profile is not valid JSON") from exc
            if profile != _PORTABLE_LAYOUT_PROFILE or protocol.canonical_json_bytes(profile) != profile_payload:
                raise protocol.ProtocolError("portable layout profile differs from the sealed schema")
        else:
            telemetry = self.telemetry.absolute()
            if any(os.path.lexists(path) for path in (self.telemetry, self.runtime_manifest, self.artifact_profile)):
                raise protocol.ProtocolError("legacy layout contains an ambiguous portable-artifact marker")
        for path, label in (
            (self.ledger, "attempt ledger"),
            (self.run_header, "run header"),
            (self.runtime_manifest, "runtime manifest"),
            (self.artifact_profile, "portable layout profile"),
            (self.receipt, "generation receipt"),
        ):
            if path.absolute().parent.resolve(strict=True) != root:
                raise protocol.ProtocolError(f"{label} must remain separate beside canonical output")
            if path.is_symlink():
                raise protocol.ProtocolError(f"{label} must not be a symlink")
        if (
            self.ledger.name != "pair-attempt-ledger.jsonl"
            or self.run_header.name != "generation-run-header.json"
            or self.runtime_manifest.name != "sealed-runtime-manifest.json"
            or self.artifact_profile.name != "portable-layout-profile.json"
            or self.receipt.name != "generation-receipt.json"
            or self.telemetry.name != "sealed-telemetry"
        ):
            raise protocol.ProtocolError("generation layout artifact names differ from the sealed design")
        return GenerationLayout(
            root,
            attempts,
            canonical,
            telemetry,
            self.ledger.absolute(),
            self.run_header.absolute(),
            self.runtime_manifest.absolute(),
            self.artifact_profile.absolute(),
            self.receipt.absolute(),
            self.portable_artifacts,
        )

    @classmethod
    def create(cls, root: Path, *, portable_artifacts: bool = False) -> "GenerationLayout":
        if not isinstance(portable_artifacts, bool):
            raise protocol.ProtocolError("portable_artifacts must be boolean")
        root = protocol.create_private_directory(root)
        attempts = _mkdir_private(root / "attempts")
        canonical = _mkdir_private(root / "canonical")
        telemetry = root / "sealed-telemetry"
        artifact_profile = root / "portable-layout-profile.json"
        runtime_manifest = root / "sealed-runtime-manifest.json"
        if portable_artifacts:
            telemetry = _mkdir_private(telemetry)
            protocol._atomic_write(artifact_profile, protocol.canonical_json_bytes(_PORTABLE_LAYOUT_PROFILE))
        return cls(
            root=root,
            attempts=attempts,
            canonical=canonical,
            telemetry=telemetry,
            ledger=root / "pair-attempt-ledger.jsonl",
            run_header=root / "generation-run-header.json",
            runtime_manifest=runtime_manifest,
            artifact_profile=artifact_profile,
            receipt=root / "generation-receipt.json",
            portable_artifacts=portable_artifacts,
        ).validated()

    @classmethod
    def open(cls, root: Path) -> "GenerationLayout":
        root = protocol.require_private_directory(root)
        attempts = protocol.require_private_directory(root / "attempts")
        canonical = protocol.require_private_directory(root / "canonical")
        artifact_profile = root / "portable-layout-profile.json"
        runtime_manifest = root / "sealed-runtime-manifest.json"
        telemetry = root / "sealed-telemetry"
        portable_artifacts = os.path.lexists(artifact_profile)
        if not portable_artifacts and (os.path.lexists(runtime_manifest) or os.path.lexists(telemetry)):
            raise protocol.ProtocolError("legacy layout contains incomplete portable-artifact state")
        return cls(
            root=root,
            attempts=attempts,
            canonical=canonical,
            telemetry=telemetry,
            ledger=root / "pair-attempt-ledger.jsonl",
            run_header=root / "generation-run-header.json",
            runtime_manifest=runtime_manifest,
            artifact_profile=artifact_profile,
            receipt=root / "generation-receipt.json",
            portable_artifacts=portable_artifacts,
        ).validated()


def _load_run_header(
    layout: GenerationLayout,
    *,
    verify_current_runner: bool = True,
) -> dict[str, Any]:
    layout = layout.validated()
    path = protocol.require_private_path(layout.run_header, must_exist=True)
    if path.stat().st_mode & 0o077:
        raise protocol.ProtocolError("private generation run header must use 0600 permissions")
    payload = protocol._read_immutable_file(path)
    assert payload is not None
    try:
        import json

        header = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise protocol.ProtocolError("generation run header is not valid JSON") from exc
    if not isinstance(header, dict) or protocol.canonical_json_bytes(header) != payload:
        raise protocol.ProtocolError("generation run header is not canonical JSON")
    if header.get("schema") != GENERATION_RUN_HEADER_SCHEMA:
        raise protocol.ProtocolError("unexpected generation run header schema")
    if verify_current_runner and header.get("runner_source_sha256") != protocol.sha256_file(Path(__file__)):
        raise protocol.ProtocolError("current runner source differs from the immutable run header")
    return header


def _sealed_runtime_descriptor(layout: GenerationLayout, binding: GenerationBinding) -> dict[str, Any]:
    layout = layout.validated()
    if not layout.portable_artifacts:
        raise protocol.ProtocolError("legacy generation layout is non-portable and has no sealed runtime manifest")
    attestation = binding.automatic_attestation
    if binding.binding_source != AUTOMATIC_BINDING_SOURCE or attestation is None:
        raise protocol.ProtocolError("portable smoke artifacts require an automatic runtime attestation")
    payload = attestation.private_runtime_payload
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise protocol.ProtocolError("private runtime manifest is not valid JSON") from exc
    if protocol.canonical_json_bytes(document) != payload or document.get("schema") != RUNTIME_ATTESTATION_SCHEMA:
        raise protocol.ProtocolError("private runtime manifest is not canonical or has the wrong schema")
    digest = protocol.sha256_bytes(payload)
    if digest != binding.runtime_digest:
        raise protocol.ProtocolError("private runtime manifest differs from the automatic runtime digest")
    protocol._atomic_write(layout.runtime_manifest, payload)
    profile_payload = protocol._read_immutable_file(layout.artifact_profile)
    observed = protocol._read_immutable_file(layout.runtime_manifest)
    assert profile_payload is not None and observed is not None
    if observed != payload:
        raise protocol.ProtocolError("sealed runtime manifest publication changed bytes")
    return {
        "schema": SEALED_RUNTIME_BINDING_SCHEMA,
        "filename": layout.runtime_manifest.name,
        "sha256": digest,
        "size_bytes": len(payload),
        "runtime_schema": RUNTIME_ATTESTATION_SCHEMA,
        "layout_profile_sha256": protocol.sha256_bytes(profile_payload),
    }


def _verify_sealed_runtime_manifest(
    layout: GenerationLayout,
    descriptor: Mapping[str, Any],
    *,
    expected_runtime_digest: str,
) -> dict[str, Any]:
    layout = layout.validated()
    if not layout.portable_artifacts:
        raise protocol.ProtocolError(
            "legacy generation layout is non-portable and cannot attest original runtime bytes"
        )
    expected_keys = {
        "schema",
        "filename",
        "sha256",
        "size_bytes",
        "runtime_schema",
        "layout_profile_sha256",
    }
    if not isinstance(descriptor, Mapping) or set(descriptor) != expected_keys:
        raise protocol.ProtocolError("sealed runtime descriptor fields are invalid")
    if (
        descriptor.get("schema") != SEALED_RUNTIME_BINDING_SCHEMA
        or descriptor.get("filename") != layout.runtime_manifest.name
        or descriptor.get("runtime_schema") != RUNTIME_ATTESTATION_SCHEMA
    ):
        raise protocol.ProtocolError("sealed runtime descriptor differs from the portable layout")
    payload = protocol._read_immutable_file(layout.runtime_manifest)
    profile_payload = protocol._read_immutable_file(layout.artifact_profile)
    assert payload is not None and profile_payload is not None
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise protocol.ProtocolError("sealed runtime manifest is not valid JSON") from exc
    digest = protocol.sha256_bytes(payload)
    if protocol.canonical_json_bytes(document) != payload or document.get("schema") != RUNTIME_ATTESTATION_SCHEMA:
        raise protocol.ProtocolError("sealed runtime manifest is not canonical or has the wrong schema")
    if (
        descriptor.get("sha256") != digest
        or descriptor.get("size_bytes") != len(payload)
        or descriptor.get("layout_profile_sha256") != protocol.sha256_bytes(profile_payload)
        or digest != expected_runtime_digest
    ):
        raise protocol.ProtocolError("sealed runtime manifest digest binding mismatch")
    return dict(descriptor)


def _seal_run_header(layout: GenerationLayout, expected: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload = protocol.canonical_json_bytes(dict(expected))
    existing = protocol._read_immutable_file(layout.run_header, allow_missing=True)
    if existing is not None:
        observed = _load_run_header(layout)
        if observed != dict(expected):
            raise protocol.ProtocolError("generation run header differs from current execution inputs")
        return observed, protocol.sha256_bytes(existing)
    protocol._atomic_write(layout.run_header, payload)
    observed = _load_run_header(layout)
    if observed != dict(expected):
        raise protocol.ProtocolError("generation run header differs from current execution inputs")
    return observed, protocol.sha256_bytes(payload)


def _require_pristine_layout_before_first_header(layout: GenerationLayout) -> None:
    """Reject ambiguous recovery instead of creating a header over prior bytes."""

    artifacts = (layout.ledger, layout.runtime_manifest, layout.receipt)
    if any(os.path.lexists(path) for path in artifacts):
        raise protocol.ProtocolError("headerless generation layout already contains run artifacts")
    for directory in (layout.attempts, layout.canonical, layout.telemetry if layout.portable_artifacts else None):
        if directory is not None and any(directory.iterdir()):
            raise protocol.ProtocolError("headerless generation layout already contains run state")


def _pairs(schedule: Sequence[protocol.ScheduleEntry]) -> tuple[tuple[protocol.ScheduleEntry, ...], ...]:
    protocol._expected_instance_ids(schedule)
    return tuple(tuple(schedule[index : index + 2]) for index in range(0, len(schedule), 2))


def _pair_records(
    ledger: protocol.AttemptLedger,
    pair_index: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(record for record in ledger.read() if record["pair_index"] == pair_index)


def _attempt_index(records: Sequence[Mapping[str, Any]]) -> int:
    starts = [int(record["attempt_index"]) for record in records if record["event"] == "started"]
    return max(starts, default=-1) + 1


def _completed_event(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    completed = [record for record in records if record["event"] == "completed"]
    if len(completed) > 1:
        raise protocol.ProtocolError("pair has multiple completed attempts")
    return completed[0] if completed else None


def _has_open_attempt(records: Sequence[Mapping[str, Any]]) -> bool:
    if not records:
        return False
    key = (records[-1]["attempt_index"], records[-1]["event"])
    return key[1] == "started"


def _checkpoint_hashes(
    store: protocol.CheckpointStore,
    pair: Sequence[protocol.ScheduleEntry],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for entry in pair:
        path = store.path_for(entry)
        if not path.is_file():
            raise protocol.ProtocolError("completed pair attempt lacks an arm checkpoint")
        store.load(entry)
        hashes[entry.condition] = protocol._immutable_file_sha256(path)
    return hashes


def _pair_artifact_binding_sha256(checkpoint_sha256: str, telemetry_sha256: str) -> str:
    _require_sha256(checkpoint_sha256, "pair checkpoint digest")
    _require_sha256(telemetry_sha256, "pair telemetry digest")
    return protocol.sha256_bytes(
        protocol.canonical_json_bytes(
            {
                "schema": PAIR_ARTIFACT_BINDING_SCHEMA,
                "checkpoint_sha256": checkpoint_sha256,
                "telemetry_sha256": telemetry_sha256,
            }
        )
    )


def _completed_pair_hashes(
    store: protocol.CheckpointStore,
    pair: Sequence[protocol.ScheduleEntry],
    *,
    require_telemetry: bool,
    telemetry_root: Path | None = None,
) -> dict[str, str]:
    """Return ledger commitments, binding sidecars for portable attempts."""

    checkpoint_hashes = _checkpoint_hashes(store, pair)
    if not require_telemetry:
        if telemetry_root is not None:
            raise protocol.ProtocolError("legacy pair cannot bind a telemetry root")
        return checkpoint_hashes
    if telemetry_root is None:
        raise protocol.ProtocolError("portable completed pair lacks a telemetry root")
    hashes: dict[str, str] = {}
    for entry in pair:
        checkpoint = store.load(entry)
        _document, telemetry_sha256 = _load_telemetry_sidecar(
            telemetry_root,
            entry,
            checkpoint,
            checkpoint_hashes[entry.condition],
        )
        hashes[entry.condition] = _pair_artifact_binding_sha256(
            checkpoint_hashes[entry.condition],
            telemetry_sha256,
        )
    return hashes


def _telemetry_path(root: Path, entry: protocol.ScheduleEntry) -> Path:
    return root / f"{entry.execution_index:04d}-{entry.condition}.telemetry.json"


def _telemetry_sidecar_document(
    entry: protocol.ScheduleEntry,
    checkpoint: protocol.ArmCheckpoint,
    checkpoint_sha256: str,
    telemetry: ValidatedArmTelemetry,
) -> dict[str, Any]:
    if not isinstance(telemetry, ValidatedArmTelemetry) or telemetry._seal is not _VALIDATED_TELEMETRY_SEAL:
        raise protocol.ProtocolError("portable arm outcome lacks native validated telemetry")
    turn, quality, rounds, tools = telemetry.document()
    document = {
        "schema": ARM_TELEMETRY_SCHEMA,
        "execution_index": entry.execution_index,
        "pair_index": entry.pair_index,
        "position_in_pair": entry.position_in_pair,
        "condition": entry.condition,
        "instance_digest": entry.instance_digest,
        "checkpoint_sha256": checkpoint_sha256,
        "turn": turn,
        "quality_gate": quality,
        "round_count": len(rounds),
        "tool_trace_count": len(tools),
        "rounds": list(rounds),
        "tools": list(tools),
        "privacy": {
            "content_free": True,
            "only_counters_vocabulary_and_sha256_commitments": True,
        },
    }
    _validate_telemetry_sidecar(document, entry, checkpoint, checkpoint_sha256)
    return document


def _validate_telemetry_sidecar(
    raw: Mapping[str, Any],
    entry: protocol.ScheduleEntry,
    checkpoint: protocol.ArmCheckpoint,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "execution_index",
        "pair_index",
        "position_in_pair",
        "condition",
        "instance_digest",
        "checkpoint_sha256",
        "turn",
        "quality_gate",
        "round_count",
        "tool_trace_count",
        "rounds",
        "tools",
        "privacy",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys or raw.get("schema") != ARM_TELEMETRY_SCHEMA:
        raise protocol.ProtocolError("arm telemetry sidecar fields or schema are invalid")
    expected_binding = {
        "execution_index": entry.execution_index,
        "pair_index": entry.pair_index,
        "position_in_pair": entry.position_in_pair,
        "condition": entry.condition,
        "instance_digest": entry.instance_digest,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if any(raw.get(name) != value for name, value in expected_binding.items()):
        raise protocol.ProtocolError("arm telemetry sidecar differs from its schedule or checkpoint")
    if checkpoint.execution_index != entry.execution_index or checkpoint.condition != entry.condition:
        raise protocol.ProtocolError("arm telemetry checkpoint object differs from its schedule")
    raw_rounds = raw.get("rounds")
    raw_tools = raw.get("tools")
    if not isinstance(raw_rounds, list) or not isinstance(raw_tools, list):
        raise protocol.ProtocolError("arm telemetry sidecar streams are malformed")
    rounds = tuple(_validate_round_trace_record(item) for item in raw_rounds)
    tools = tuple(_validate_tool_trace_record(item) for item in raw_tools)
    if tuple(item["round_index"] for item in rounds) != tuple(range(len(rounds))):
        raise protocol.ProtocolError("arm telemetry round indices are not contiguous")
    if tuple(item["sequence"] for item in tools) != tuple(range(len(tools))):
        raise protocol.ProtocolError("arm telemetry tool sequences are not contiguous")
    tool_rounds = tuple(item["round_index"] for item in tools)
    if any(index >= len(rounds) for index in tool_rounds) or tool_rounds != tuple(sorted(tool_rounds)):
        raise protocol.ProtocolError("arm telemetry tool round topology is invalid")
    if raw.get("round_count") != len(rounds) or raw.get("tool_trace_count") != len(tools):
        raise protocol.ProtocolError("arm telemetry sidecar counts differ from its streams")
    quality_raw = raw.get("quality_gate")
    turn_raw = raw.get("turn")
    if not isinstance(quality_raw, Mapping) or not isinstance(turn_raw, Mapping):
        raise protocol.ProtocolError("arm telemetry turn or quality document is malformed")
    quality = _validate_normalized_quality_document(quality_raw)
    turn = _validate_turn_and_topology(turn_raw, rounds, tools, quality)
    checkpoint_wall_ns = round(checkpoint.wall_seconds * 1_000_000_000)
    expected_checkpoint_wall_ns = min(
        turn["wall_elapsed_ns"],
        int(protocol.MAX_AGENT_WALL_SECONDS * 1_000_000_000),
    )
    if (
        turn["completion_tokens"] != checkpoint.output_tokens
        or turn["tool_calls"] != checkpoint.tool_calls
        or turn["status"] != checkpoint.status
        or turn["quality_gate_decision"] != checkpoint.quality_gate_decision
        or abs(expected_checkpoint_wall_ns - checkpoint_wall_ns) > 1
    ):
        raise protocol.ProtocolError("arm telemetry terminal evidence differs from checkpoint")
    if quality["enabled"] != (entry.condition == "gate_on"):
        raise protocol.ProtocolError("arm telemetry Quality condition differs from the schedule")
    expected_privacy = {
        "content_free": True,
        "only_counters_vocabulary_and_sha256_commitments": True,
    }
    if raw.get("privacy") != expected_privacy:
        raise protocol.ProtocolError("arm telemetry privacy declaration is invalid")
    return dict(raw)


def _save_telemetry_sidecar(
    root: Path,
    entry: protocol.ScheduleEntry,
    checkpoint: protocol.ArmCheckpoint,
    checkpoint_sha256: str,
    telemetry: ValidatedArmTelemetry,
) -> Path:
    protocol._reject_symlink_path_components(root)
    if root.exists():
        root = protocol.require_private_directory(root)
    else:
        root = _mkdir_private(root)
    payload = protocol.canonical_json_bytes(
        _telemetry_sidecar_document(entry, checkpoint, checkpoint_sha256, telemetry)
    )
    destination = _telemetry_path(root, entry)
    protocol._atomic_write(destination, payload)
    if protocol._read_immutable_file(destination) != payload:
        raise protocol.ProtocolError("arm telemetry sidecar publication changed bytes")
    return destination


def _load_telemetry_sidecar(
    root: Path,
    entry: protocol.ScheduleEntry,
    checkpoint: protocol.ArmCheckpoint,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], str]:
    path = _telemetry_path(root, entry)
    payload = protocol._read_immutable_file(path)
    assert payload is not None
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise protocol.ProtocolError("arm telemetry sidecar is not valid JSON") from exc
    if protocol.canonical_json_bytes(document) != payload:
        raise protocol.ProtocolError("arm telemetry sidecar is not canonical JSON")
    validated = _validate_telemetry_sidecar(document, entry, checkpoint, checkpoint_sha256)
    return validated, protocol.sha256_bytes(payload)


def _promote_completed_pair(
    layout: GenerationLayout,
    pair: Sequence[protocol.ScheduleEntry],
    completed: Mapping[str, Any],
    *,
    require_telemetry: bool,
) -> None:
    attempt_index = int(completed["attempt_index"])
    attempt_store = protocol.pair_attempt_store(layout.attempts, pair[0].pair_index, attempt_index)
    checkpoint_hashes = _checkpoint_hashes(attempt_store, pair)
    completed_hashes = _completed_pair_hashes(
        attempt_store,
        pair,
        require_telemetry=require_telemetry,
        telemetry_root=(attempt_store.root / "telemetry" if require_telemetry else None),
    )
    if completed_hashes != completed["checkpoint_sha256s"]:
        raise protocol.ProtocolError("completed ledger event differs from retained checkpoint/telemetry pair")
    canonical_store = protocol.CheckpointStore(layout.canonical)
    for entry in pair:
        checkpoint = attempt_store.load(entry)
        destination = canonical_store.save(checkpoint)
        if protocol._immutable_file_sha256(destination) != checkpoint_hashes[entry.condition]:
            raise protocol.ProtocolError("canonical promotion changed checkpoint bytes")
        if require_telemetry:
            source_document, source_sha256 = _load_telemetry_sidecar(
                attempt_store.root / "telemetry",
                entry,
                checkpoint,
                checkpoint_hashes[entry.condition],
            )
            telemetry_payload = protocol.canonical_json_bytes(source_document)
            telemetry_destination = _telemetry_path(layout.telemetry, entry)
            protocol._atomic_write(telemetry_destination, telemetry_payload)
            _document, promoted_sha256 = _load_telemetry_sidecar(
                layout.telemetry,
                entry,
                checkpoint,
                checkpoint_hashes[entry.condition],
            )
            if promoted_sha256 != source_sha256:
                raise protocol.ProtocolError("canonical promotion changed arm telemetry bytes")


def pending_pairs(
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    *,
    repair_completed_promotions: bool = True,
    require_telemetry: bool | None = None,
) -> tuple[tuple[protocol.ScheduleEntry, ...], ...]:
    """Resume only complete pairs; never continue after one arm."""

    layout = layout.validated()
    if require_telemetry is None:
        require_telemetry = layout.portable_artifacts
    if not isinstance(require_telemetry, bool) or require_telemetry != layout.portable_artifacts:
        raise protocol.ProtocolError("generation layout portability differs from telemetry requirements")
    digest = protocol.schedule_digest(schedule)
    ledger = protocol.AttemptLedger(layout.ledger, digest)
    canonical_store = protocol.CheckpointStore(layout.canonical)
    pending: list[tuple[protocol.ScheduleEntry, ...]] = []
    expected_paths = {canonical_store.path_for(entry).name for entry in schedule}
    observed_paths = {path.name for path in layout.canonical.glob("*.json")}
    if observed_paths - expected_paths:
        raise protocol.ProtocolError("canonical store contains an unexpected checkpoint")
    for pair in _pairs(schedule):
        records = _pair_records(ledger, pair[0].pair_index)
        if _has_open_attempt(records):
            raise protocol.ProtocolError(
                "pair has an interrupted attempt; append a blinded infrastructure abort before retry"
            )
        completed = _completed_event(records)
        existing = [canonical_store.path_for(entry).exists() for entry in pair]
        if completed is None:
            if any(existing):
                raise protocol.ProtocolError("canonical arm exists without a completed pair ledger event")
            pending.append(pair)
            continue
        if repair_completed_promotions:
            _promote_completed_pair(layout, pair, completed, require_telemetry=require_telemetry)
        elif not all(existing):
            raise protocol.ProtocolError("sealed canonical pair is incomplete")
        expected_hashes = completed["checkpoint_sha256s"]
        observed_hashes = _completed_pair_hashes(
            canonical_store,
            pair,
            require_telemetry=require_telemetry,
            telemetry_root=(layout.telemetry if require_telemetry else None),
        )
        if observed_hashes != expected_hashes:
            raise protocol.ProtocolError("canonical checkpoint/telemetry pair differs from completed attempt")
    return tuple(pending)


def abort_interrupted_pair(
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    *,
    pair_index: int,
    reason_code: str,
) -> None:
    """Declare one interrupted pair retry using a frozen blinded reason."""

    layout = layout.validated()
    pairs = _pairs(schedule)
    if pair_index < 0 or pair_index >= len(pairs):
        raise protocol.ProtocolError("interrupted pair index is outside the schedule")
    ledger = protocol.AttemptLedger(layout.ledger, protocol.schedule_digest(schedule))
    records = _pair_records(ledger, pair_index)
    if not _has_open_attempt(records):
        raise protocol.ProtocolError("pair has no open attempt to abort")
    ledger.append(
        pair_index=pair_index,
        attempt_index=int(records[-1]["attempt_index"]),
        event="aborted",
        reason_code=reason_code,
    )


def _validate_schedule_document(
    document: Mapping[str, Any],
    schedule: Sequence[protocol.ScheduleEntry],
) -> tuple[dict[str, protocol.PublicInstance], bool]:
    if document.get("schema") != protocol.SCHEMA:
        raise protocol.ProtocolError("generation schedule schema differs from the frozen adapter")
    if document.get("preregistration_sha256") != protocol.preregistration_digest():
        raise protocol.ProtocolError("generation schedule preregistration binding mismatch")
    if (
        document.get("dataset") != protocol.DATASET_NAME
        or document.get("dataset_revision") != protocol.DATASET_REVISION
        or document.get("dataset_full_snapshot_sha256") != protocol.FULL_SNAPSHOT_SHA256
        or document.get("expected_model_identity") != protocol.EXPECTED_MODEL_IDENTITY
    ):
        raise protocol.ProtocolError("generation schedule dataset or model binding mismatch")
    if document.get("source_free_summary") != protocol.source_free_schedule_summary(schedule):
        raise protocol.ProtocolError("generation schedule document does not bind the supplied schedule")
    raw_instances = document.get("public_instances")
    if not isinstance(raw_instances, list):
        raise protocol.ProtocolError("generation schedule lacks public instances")
    instances = tuple(protocol.PublicInstance.from_mapping(row) for row in raw_instances)
    by_id = {instance.instance_id: instance for instance in instances}
    if len(by_id) != len(instances) or set(by_id) != {entry.instance_id for entry in schedule}:
        raise protocol.ProtocolError("generation instances differ from scheduled pairs")
    expected_schedule = protocol.make_balanced_schedule(
        [instance.instance_id for instance in instances],
        require_full=False,
    )
    if tuple(schedule) != expected_schedule:
        raise protocol.ProtocolError("generation order differs from the deterministic balanced schedule")
    if document.get("schedule") != [entry.private_dict() for entry in schedule]:
        raise protocol.ProtocolError("generation schedule entries differ from the sealed document")
    public_rows = [instance.as_dict() for instance in sorted(instances, key=lambda item: item.instance_id)]
    observed_public_sha256 = protocol.sha256_bytes(protocol.canonical_jsonl_bytes(public_rows))
    if document.get("dataset_public_snapshot_sha256") != observed_public_sha256:
        raise protocol.ProtocolError("generation schedule public snapshot binding mismatch")
    evidence_class = document.get("evidence_class")
    if evidence_class not in {"confirmatory", "non_evidence_smoke"}:
        raise protocol.ProtocolError("generation schedule evidence class is invalid")
    if evidence_class == "confirmatory" and observed_public_sha256 != protocol.PUBLIC_SNAPSHOT_SHA256:
        raise protocol.ProtocolError("confirmatory generation requires the exact official public snapshot")
    return by_id, evidence_class == "confirmatory"


def run_generation_pairs(
    *,
    schedule_document: Mapping[str, Any],
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    workspace_factory: WorkspaceFactory,
    executor: ArmExecutor,
    binding: GenerationBinding,
    tier_config: Any,
    agent_module: Any | None = None,
    require_portable_artifacts: bool = False,
) -> str:
    """Run every missing *whole pair* and return the frozen factor digest.

    Confirmatory execution is intentionally delegated to the adapter's global
    readiness gate.  With protocol v1 this currently raises before any model
    call, preventing accidental unblinding while controls remain pending.
    """

    layout = layout.validated()
    if not isinstance(require_portable_artifacts, bool) or require_portable_artifacts != layout.portable_artifacts:
        raise protocol.ProtocolError("runner portability requirement differs from the generation layout")
    by_id, evidence_run = _validate_schedule_document(schedule_document, schedule)
    binding.validate_for_run(
        evidence_run=evidence_run,
        executor=executor,
        tier_config=tier_config,
        require_executor_binding=True,
    )
    if evidence_run and not CONFIRMATORY_GENERATION_ENABLED:
        raise protocol.ProtocolError(
            "confirmatory SWE-bench is blocked: this runner is smoke-only until every v2 control is frozen"
        )
    protocol.require_confirmatory_generation_attestation(evidence_run)
    if evidence_run and len(schedule) != protocol.EXPECTED_INSTANCES * 2:
        raise protocol.ProtocolError("confirmatory generation requires all 500 complete pairs")
    validate_target_only_tier(tier_config)
    if require_portable_artifacts and (
        type(executor) is not NativeMioArmExecutor or executor.require_raw_target_telemetry is not True
    ):
        raise protocol.ProtocolError("portable smoke artifacts require the native raw-telemetry executor")
    registry, specs, surface_sha256 = build_identical_tool_surface(agent_module)
    header_exists = os.path.lexists(layout.run_header)
    if header_exists:
        retained_header = _load_run_header(layout)
        runtime_descriptor = retained_header.get("sealed_artifact_audit", {}).get("runtime_manifest")
        if require_portable_artifacts:
            _verify_sealed_runtime_manifest(
                layout,
                runtime_descriptor,
                expected_runtime_digest=binding.runtime_digest,
            )
        elif runtime_descriptor is not None:
            raise protocol.ProtocolError("legacy generation header unexpectedly binds a portable runtime manifest")
    else:
        _require_pristine_layout_before_first_header(layout)
        runtime_descriptor = _sealed_runtime_descriptor(layout, binding) if require_portable_artifacts else None
    run_header, _run_header_sha256 = _seal_run_header(
        layout,
        build_run_header(
            schedule_document=schedule_document,
            schedule=schedule,
            binding=binding,
            tool_surface_sha256=surface_sha256,
            executor=executor,
            workspace_factory=workspace_factory,
            tier_config=tier_config,
            sealed_runtime_manifest=runtime_descriptor,
        ),
    )
    # Only a byte-for-byte header/runtime match may authorize idempotent
    # canonical promotion repair or a new ledger append.
    pending = pending_pairs(
        schedule,
        layout,
        require_telemetry=require_portable_artifacts,
    )
    study_factor_sha256 = str(run_header["factor_sha256"])
    schedule_sha256 = protocol.schedule_digest(schedule)
    ledger = protocol.AttemptLedger(layout.ledger, schedule_sha256)
    seen_workspaces: set[Path] = set()
    seen_git_directories: set[Path] = set()
    seen_cache_directories: set[Path] = set()

    for pair in pending:
        pair_index = pair[0].pair_index
        records = _pair_records(ledger, pair_index)
        attempt_index = _attempt_index(records)
        ledger.append(
            pair_index=pair_index,
            attempt_index=attempt_index,
            event="started",
            reason_code="initial" if attempt_index == 0 else str(records[-1]["reason_code"]),
        )
        attempt_store = protocol.pair_attempt_store(layout.attempts, pair_index, attempt_index)
        for entry in pair:
            instance = by_id[entry.instance_id]
            arm_root = attempt_store.root / f"arm-{entry.position_in_pair}-{entry.condition}"
            arm = workspace_factory(
                instance=instance,
                entry=entry,
                destination=arm_root,
            ).validated()
            if arm_root.is_symlink():
                raise protocol.ProtocolError("arm destination must not be a symlink")
            try:
                resolved_arm_root = arm_root.resolve(strict=True)
            except OSError as exc:
                raise protocol.ProtocolError("workspace factory did not create its exclusive arm destination") from exc
            expected_paths = {
                "workspace": resolved_arm_root / "workspace",
                "Git metadata": resolved_arm_root / "trusted-metadata" / "git",
                "cache": resolved_arm_root / "cache",
            }
            observed_paths = {
                "workspace": arm.workspace,
                "Git metadata": arm.external_git_directory,
                "cache": arm.cache_directory,
            }
            if observed_paths != expected_paths:
                raise protocol.ProtocolError("workspace factory returned state outside its exclusive arm destination")
            for path, seen, label in (
                (arm.workspace, seen_workspaces, "workspace"),
                (arm.external_git_directory, seen_git_directories, "Git metadata"),
                (arm.cache_directory, seen_cache_directories, "cache"),
            ):
                resolved = path.resolve(strict=True)
                if resolved in seen:
                    raise protocol.ProtocolError(f"arm reused a supposedly fresh {label}")
                seen.add(resolved)

            from mio.agent_policy import AgentToolPermission, AgentToolPolicy

            policy = AgentToolPolicy.coding_workspace(
                arm.workspace,
                command_timeout_s=FROZEN_COMMAND_TIMEOUT_SECONDS,
                allow_network=False,
            )
            if AgentToolPermission.NETWORK in policy.permissions:
                raise protocol.ProtocolError("SWE-bench generation unexpectedly grants network access")
            request = ArmRunRequest(
                entry=entry,
                instruction=_model_instruction(instance),
                workspace=arm.workspace,
                cache_directory=arm.cache_directory,
                tool_registry=registry,
                tool_specs=tuple(copy.deepcopy(specs)),
                tool_policy=policy,
                quality_gate_enabled=entry.condition == "gate_on",
                coding_effort="medium",
                seed=_arm_seed(entry),
            )
            outcome = executor(request)
            if not isinstance(outcome, ArmRunOutcome):
                raise protocol.ProtocolError("arm executor returned an untrusted outcome type")
            observed_surface = _tool_surface_document(
                request.tool_registry,
                request.tool_specs,
            )
            if (
                tuple(request.tool_registry) != TOOL_SURFACE
                or protocol.sha256_bytes(protocol.canonical_json_bytes(observed_surface)) != surface_sha256
            ):
                raise protocol.ProtocolError("arm executor mutated the frozen tool surface")
            _assert_no_visible_git(arm.workspace)
            patch = protocol.capture_git_patch(
                arm.workspace,
                expected_base_commit=instance.base_commit,
                external_git_directory=arm.external_git_directory,
            )
            checkpoint = protocol.ArmCheckpoint.for_entry(
                entry,
                schedule_sha256=schedule_sha256,
                status=outcome.status,
                model_patch=patch,
                mio_commit=binding.mio_commit,
                model_identity=binding.model_identity,
                runtime_digest=binding.runtime_digest,
                quality_gate_decision=outcome.quality_gate_decision,
                output_tokens=outcome.output_tokens,
                tool_calls=outcome.tool_calls,
                wall_seconds=outcome.wall_seconds,
            )
            checkpoint_path = attempt_store.save(checkpoint)
            if require_portable_artifacts:
                if outcome.telemetry is None:
                    raise protocol.ProtocolError("portable smoke arm lacks native validated telemetry")
                _save_telemetry_sidecar(
                    attempt_store.root / "telemetry",
                    entry,
                    checkpoint,
                    protocol._immutable_file_sha256(checkpoint_path),
                    outcome.telemetry,
                )

        hashes = _completed_pair_hashes(
            attempt_store,
            pair,
            require_telemetry=require_portable_artifacts,
            telemetry_root=(attempt_store.root / "telemetry" if require_portable_artifacts else None),
        )
        completed = ledger.append(
            pair_index=pair_index,
            attempt_index=attempt_index,
            event="completed",
            reason_code="completed",
            checkpoint_sha256s=hashes,
        )
        _promote_completed_pair(
            layout,
            pair,
            completed,
            require_telemetry=require_portable_artifacts,
        )
    return study_factor_sha256


def _generation_manifest(
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    run_header: Mapping[str, Any],
) -> list[dict[str, Any]]:
    layout = layout.validated()
    if pending_pairs(schedule, layout, repair_completed_promotions=False):
        raise protocol.ProtocolError("generation receipt requires every pair to be complete")
    store = protocol.CheckpointStore(layout.canonical)
    if layout.portable_artifacts:
        expected_telemetry = {_telemetry_path(layout.telemetry, entry).name for entry in schedule}
        observed_telemetry = {path.name for path in layout.telemetry.iterdir()}
        if observed_telemetry != expected_telemetry:
            raise protocol.ProtocolError("sealed telemetry directory is missing an arm or contains an extra artifact")
    rows = []
    expected_binding = run_header.get("generation_binding")
    expected_schedule_sha256 = str(run_header.get("schedule_sha256", ""))
    for entry in schedule:
        checkpoint = store.load(entry)
        checkpoint_binding = {
            "mio_commit": checkpoint.mio_commit,
            "model_identity": checkpoint.model_identity,
            "runtime_digest": checkpoint.runtime_digest,
        }
        if (
            checkpoint.preregistration_sha256 != run_header.get("preregistration_sha256")
            or checkpoint.schedule_sha256 != expected_schedule_sha256
            or checkpoint_binding != expected_binding
        ):
            raise protocol.ProtocolError("canonical checkpoint differs from the immutable run header")
        checkpoint_sha256 = protocol._immutable_file_sha256(store.path_for(entry))
        row = {
            "execution_index": entry.execution_index,
            "pair_index": entry.pair_index,
            "position_in_pair": entry.position_in_pair,
            "condition": entry.condition,
            "instance_digest": checkpoint.instance_digest,
            "checkpoint_sha256": checkpoint_sha256,
        }
        if layout.portable_artifacts:
            telemetry, telemetry_sha256 = _load_telemetry_sidecar(
                layout.telemetry,
                entry,
                checkpoint,
                checkpoint_sha256,
            )
            row.update(
                {
                    "telemetry_filename": _telemetry_path(layout.telemetry, entry).name,
                    "telemetry_sha256": telemetry_sha256,
                    "completed_artifact_binding_sha256": _pair_artifact_binding_sha256(
                        checkpoint_sha256,
                        telemetry_sha256,
                    ),
                    "round_count": telemetry["round_count"],
                    "tool_trace_count": telemetry["tool_trace_count"],
                }
            )
        rows.append(row)
    return rows


def _telemetry_manifest(manifest: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries = []
    for row in manifest:
        required = {
            "execution_index",
            "condition",
            "checkpoint_sha256",
            "telemetry_filename",
            "telemetry_sha256",
            "completed_artifact_binding_sha256",
            "round_count",
            "tool_trace_count",
        }
        if not required.issubset(row):
            raise protocol.ProtocolError("portable canonical manifest lacks arm telemetry bindings")
        entries.append({name: row[name] for name in sorted(required)})
    document = {
        "schema": TELEMETRY_MANIFEST_SCHEMA,
        "arm_count": len(entries),
        "entries": entries,
        "content_free": True,
    }
    document["entries_sha256"] = protocol.sha256_bytes(protocol.canonical_json_bytes(entries))
    return document


def _build_generation_receipt_from_artifacts(
    *,
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    run_header: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute a receipt from sealed bytes without inspecting the host."""

    if run_header.get("schedule_sha256") != protocol.schedule_digest(schedule):
        raise protocol.ProtocolError("receipt schedule differs from the immutable run header")
    manifest = _generation_manifest(schedule, layout, run_header)
    ledger = protocol.AttemptLedger(layout.ledger, protocol.schedule_digest(schedule))
    records = ledger.read()
    completed_pairs = {record["pair_index"] for record in records if record["event"] == "completed"}
    if completed_pairs != {pair[0].pair_index for pair in _pairs(schedule)}:
        raise protocol.ProtocolError("generation ledger lacks exactly one completion per pair")
    ledger_payload = protocol._read_immutable_file(layout.ledger)
    run_header_payload = protocol._read_immutable_file(layout.run_header)
    assert ledger_payload is not None and run_header_payload is not None
    run_header_sha256 = protocol.sha256_bytes(run_header_payload)
    artifact_header = run_header.get("sealed_artifact_audit")
    if not isinstance(artifact_header, Mapping) or not isinstance(artifact_header.get("portable"), bool):
        raise protocol.ProtocolError("run header lacks an explicit sealed-artifact audit mode")
    if artifact_header["portable"] != layout.portable_artifacts:
        raise protocol.ProtocolError("run header portability differs from the retained layout")
    if layout.portable_artifacts:
        binding = run_header.get("generation_binding")
        if not isinstance(binding, Mapping) or not isinstance(binding.get("runtime_digest"), str):
            raise protocol.ProtocolError("portable run header lacks its runtime binding")
        runtime_descriptor = artifact_header.get("runtime_manifest")
        _verify_sealed_runtime_manifest(
            layout,
            runtime_descriptor,
            expected_runtime_digest=str(binding["runtime_digest"]),
        )
        telemetry_manifest = _telemetry_manifest(manifest)
        receipt_artifact_audit = {
            "portable": True,
            "cross_process_sealed_artifact_verification_supported": True,
            "runtime_manifest": dict(runtime_descriptor),
            "telemetry_manifest": telemetry_manifest,
            "telemetry_manifest_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(telemetry_manifest)),
            "current_environment_reattestation_is_separate": True,
            "sealed_artifact_verification_does_not_claim_current_environment_identity": True,
        }
    else:
        receipt_artifact_audit = {
            "portable": False,
            "cross_process_sealed_artifact_verification_supported": False,
            "legacy_non_portable": True,
            "current_environment_reattestation_is_separate": True,
        }
    model_identity = str(run_header["generation_binding"]["model_identity"])
    return {
        "schema": GENERATION_RECEIPT_SCHEMA,
        "preregistration_sha256": run_header["preregistration_sha256"],
        "schedule_sha256": run_header["schedule_sha256"],
        "run_header_sha256": run_header_sha256,
        "factor_sha256": run_header["factor_sha256"],
        "factor": run_header["factor"],
        "generation_binding": run_header["generation_binding"],
        "generation_binding_attestation": run_header["generation_binding_attestation"],
        "loaded_target_binding": run_header["loaded_target_binding"],
        "model_identity_checks": {
            "before_first_generation": model_identity,
            "after_last_generation": model_identity,
        },
        "attempt_ledger": {
            "sha256": protocol.sha256_bytes(ledger_payload),
            "records": len(records),
            "head_sha256": records[-1]["record_sha256"] if records else "",
        },
        "pair_count": len(schedule) // 2,
        "arm_count": len(schedule),
        "canonical_manifest_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(manifest)),
        "canonical_manifest": manifest,
        "sealed_artifact_audit": receipt_artifact_audit,
        "contains_model_text_or_evaluator_output": False,
        "evidence_class": run_header["evidence_class"],
        "confirmatory_evidence_admissible": False,
        "confirmatory_blockers": list(run_header["confirmatory_blockers"]),
    }


def build_generation_receipt(
    *,
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    binding: GenerationBinding,
    tool_surface_sha256: str,
    observed_model_identity_before: str,
    observed_model_identity_after: str,
) -> dict[str, Any]:
    """Build a receipt after separately re-attesting the current environment."""

    if {observed_model_identity_before, observed_model_identity_after} != {protocol.EXPECTED_MODEL_IDENTITY}:
        raise protocol.ProtocolError("generation receipt target identity checks differ")
    run_header = _load_run_header(layout, verify_current_runner=True)
    if run_header.get("generation_binding") != binding.as_dict():
        raise protocol.ProtocolError("receipt binding differs from the immutable run header")
    evidence_run = run_header.get("evidence_class") == "confirmatory"
    binding.validate_for_run(evidence_run=evidence_run)
    if run_header.get("generation_binding_attestation") != binding.attestation_dict():
        raise protocol.ProtocolError("receipt attestation differs from the immutable run header")
    if run_header.get("tool_surface_sha256") != tool_surface_sha256:
        raise protocol.ProtocolError("receipt tool surface differs from the immutable run header")
    if {observed_model_identity_before, observed_model_identity_after} != {
        str(run_header["generation_binding"]["model_identity"])
    }:
        raise protocol.ProtocolError("receipt identity observations differ from the immutable run header")
    return _build_generation_receipt_from_artifacts(
        schedule=schedule,
        layout=layout,
        run_header=run_header,
    )


def seal_generation_receipt(
    *,
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    binding: GenerationBinding,
    tool_surface_sha256: str,
    observed_model_identity_before: str,
    observed_model_identity_after: str,
) -> str:
    layout = layout.validated()
    # A crash may occur after the atomic ledger completion and before both
    # idempotent canonical publishes. Repair is allowed only before sealing.
    pending_pairs(schedule, layout, repair_completed_promotions=True)
    receipt = build_generation_receipt(
        schedule=schedule,
        layout=layout,
        binding=binding,
        tool_surface_sha256=tool_surface_sha256,
        observed_model_identity_before=observed_model_identity_before,
        observed_model_identity_after=observed_model_identity_after,
    )
    payload = protocol.canonical_json_bytes(receipt)
    protocol._atomic_write(layout.receipt, payload)
    return protocol.sha256_bytes(payload)


def verify_generation_receipt(
    *,
    receipt_path: Path,
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    binding: GenerationBinding,
    tool_surface_sha256: str,
) -> str:
    """Verify sealed bytes, then independently re-attest the current host."""

    layout = layout.validated()
    if not layout.portable_artifacts:
        raise protocol.ProtocolError("legacy generation receipt cannot support current-environment reattestation")
    digest = _verify_generation_artifacts(
        receipt_path=receipt_path,
        schedule=schedule,
        layout=layout,
        tool_surface_sha256=tool_surface_sha256,
        require_portable=True,
    )
    reattest_current_generation_environment(layout=layout, binding=binding)
    return digest


def _verify_generation_artifacts(
    *,
    receipt_path: Path,
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    tool_surface_sha256: str,
    require_portable: bool,
) -> str:
    """Verify only retained bytes; never compare them with the current host."""

    layout = layout.validated()
    if require_portable and not layout.portable_artifacts:
        raise protocol.ProtocolError(
            "legacy generation layout is non-portable; original runtime and telemetry were not retained"
        )
    path = protocol.require_private_path(receipt_path, must_exist=True)
    if path != layout.receipt.resolve(strict=True):
        raise protocol.ProtocolError("generation receipt path differs from the sealed layout receipt")
    if path.stat().st_mode & 0o077:
        raise protocol.ProtocolError("private generation receipt must use 0600 permissions")
    payload = protocol._read_immutable_file(path)
    assert payload is not None
    try:
        import json

        observed = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise protocol.ProtocolError("generation receipt is not valid JSON") from exc
    if protocol.canonical_json_bytes(observed) != payload:
        raise protocol.ProtocolError("generation receipt is not canonical JSON")
    if observed.get("schema") != GENERATION_RECEIPT_SCHEMA:
        raise protocol.ProtocolError("unexpected generation receipt schema")
    run_header = _load_run_header(layout, verify_current_runner=False)
    runner_source_sha256 = run_header.get("runner_source_sha256")
    if not isinstance(runner_source_sha256, str):
        raise protocol.ProtocolError("run header runner source digest is malformed")
    _require_sha256(runner_source_sha256, "sealed runner source digest")
    if run_header.get("tool_surface_sha256") != tool_surface_sha256:
        raise protocol.ProtocolError("sealed tool surface differs from the requested artifact audit")
    expected = _build_generation_receipt_from_artifacts(
        schedule=schedule,
        layout=layout,
        run_header=run_header,
    )
    if observed != expected:
        raise protocol.ProtocolError("generation receipt differs from the retained sealed artifacts")
    return protocol.sha256_bytes(payload)


def verify_sealed_generation_artifacts(
    *,
    receipt_path: Path,
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    tool_surface_sha256: str,
) -> str:
    """Cross-process audit of original sealed bytes, with no current-env claim."""

    return _verify_generation_artifacts(
        receipt_path=receipt_path,
        schedule=schedule,
        layout=layout,
        tool_surface_sha256=tool_surface_sha256,
        require_portable=True,
    )


def verify_legacy_generation_artifacts(
    *,
    receipt_path: Path,
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    tool_surface_sha256: str,
) -> str:
    """Audit retained legacy bytes without claiming runtime reattestation."""

    layout = layout.validated()
    if layout.portable_artifacts:
        raise protocol.ProtocolError("portable generation must use the sealed cross-process verifier")
    return _verify_generation_artifacts(
        receipt_path=receipt_path,
        schedule=schedule,
        layout=layout,
        tool_surface_sha256=tool_surface_sha256,
        require_portable=False,
    )


def reattest_current_generation_environment(
    *,
    layout: GenerationLayout,
    binding: GenerationBinding,
) -> dict[str, Any]:
    """Fail closed if the *current* source/model/runtime differs from capture."""

    layout = layout.validated()
    if (
        not layout.portable_artifacts
        or binding.binding_source != AUTOMATIC_BINDING_SOURCE
        or binding.automatic_attestation is None
    ):
        raise protocol.ProtocolError(
            "legacy or caller-supplied generation binding cannot support current-environment reattestation"
        )
    run_header = _load_run_header(layout, verify_current_runner=True)
    if run_header.get("generation_binding") != binding.as_dict():
        raise protocol.ProtocolError("current binding differs from the immutable run header")
    evidence_run = run_header.get("evidence_class") == "confirmatory"
    current = binding.validate_for_run(evidence_run=evidence_run)
    if run_header.get("generation_binding_attestation") != binding.attestation_dict():
        raise protocol.ProtocolError("current attestation differs from the immutable run header")
    return {
        "sealed_artifact_verification_not_performed_here": True,
        "current_environment_reverified": True,
        "binding": binding.as_dict(),
        "attestation_result": current,
    }


class NativeMioArmExecutor:
    """Adapter for one loaded target-only Mio engine with fresh state per arm.

    ``require_raw_target_telemetry`` is opt-in so existing non-benchmark callers
    remain compatible.  When enabled, every arm must expose contiguous,
    content-free raw round/tool traces before an outcome can be returned.  The
    portable smoke layout persists those validated traces beside the unchanged
    v1 checkpoint; this still does not make a smoke result confirmatory evidence.
    """

    def __init__(
        self,
        *,
        engine: Any,
        manager: Any,
        config: Any,
        tier: str,
        require_raw_target_telemetry: bool = False,
    ) -> None:
        if not isinstance(require_raw_target_telemetry, bool):
            raise protocol.ProtocolError("raw target telemetry flag must be boolean")
        validate_target_only_tier(getattr(engine, "tier_config", None))
        self.engine = engine
        self.manager = manager
        self.config = config
        self.tier = tier
        self.require_raw_target_telemetry = require_raw_target_telemetry

    @staticmethod
    def _trace_nonnegative_int(trace: Any, name: str) -> int:
        value = getattr(trace, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise protocol.ProtocolError(f"raw target round {name} must be a non-negative integer")
        return value

    def _validate_raw_target_result(
        self,
        result: Any,
        request: ArmRunRequest,
        *,
        wall_elapsed_s: float,
    ) -> ValidatedArmTelemetry | None:
        if not self.require_raw_target_telemetry:
            return None
        rounds = tuple(getattr(result, "rounds", ()) or ())
        if not rounds:
            raise protocol.ProtocolError("raw target telemetry requires at least one model round")
        for expected_index, trace in enumerate(rounds):
            if getattr(trace, "round_index", None) != expected_index:
                raise protocol.ProtocolError("raw target round indexes must be zero-based and contiguous")
            if (
                getattr(trace, "generation_backend", None) != "baseline"
                or getattr(trace, "fallback_ar", None) is not False
                or getattr(trace, "drafter_requested", None) != "target_ar"
                or getattr(trace, "drafter_selected", None) != "baseline"
                or getattr(trace, "drafter_ref", None) is not None
            ):
                raise protocol.ProtocolError("raw target round differs from target_ar/baseline/no-drafter")
            if getattr(trace, "timing_source", None) != "runtime_raw_ns":
                raise protocol.ProtocolError("raw target round timing source is not runtime_raw_ns")
            prefill_ns = self._trace_nonnegative_int(trace, "prefill_ns")
            decode_ns = self._trace_nonnegative_int(trace, "decode_ns")
            model_total_ns = self._trace_nonnegative_int(trace, "model_total_ns")
            completion_tokens = self._trace_nonnegative_int(trace, "completion_tokens")
            physical_decode_tokens = self._trace_nonnegative_int(trace, "physical_decode_tokens")
            if model_total_ns != prefill_ns + decode_ns:
                raise protocol.ProtocolError("raw target model time differs from prefill plus decode")
            if physical_decode_tokens < completion_tokens:
                raise protocol.ProtocolError("raw target physical decode work is below delivered completion tokens")

        tool_calls = getattr(result, "tool_calls", None)
        if isinstance(tool_calls, bool) or not isinstance(tool_calls, int) or tool_calls < 0:
            raise protocol.ProtocolError("raw target tool-call count is invalid")
        tool_events = tuple(getattr(result, "tool_events", ()) or ())
        if len(tool_events) != tool_calls:
            raise protocol.ProtocolError("raw target telemetry requires exactly one trace per tool call")
        for sequence, trace in enumerate(tool_events):
            if getattr(trace, "sequence", None) != sequence:
                raise protocol.ProtocolError("raw target tool traces must be zero-based and contiguous")

        delivered = getattr(result, "completion_tokens", None)
        if isinstance(delivered, bool) or not isinstance(delivered, int) or delivered < 0:
            raise protocol.ProtocolError("raw target delivered-token count is invalid")
        if delivered != sum(self._trace_nonnegative_int(trace, "completion_tokens") for trace in rounds):
            raise protocol.ProtocolError("raw target delivered-token total differs from round completion tokens")
        return ValidatedArmTelemetry.from_result(
            result,
            quality_gate_enabled=request.quality_gate_enabled,
            wall_elapsed_s=wall_elapsed_s,
        )

    def _assert_manager_engine_identity(self) -> None:
        loaded_tiers = getattr(self.manager, "loaded_tiers", None)
        get_engine = getattr(self.manager, "get_engine", None)
        if not callable(loaded_tiers) or not callable(get_engine):
            return
        if self.tier in loaded_tiers() and get_engine(self.tier) is not self.engine:
            raise protocol.ProtocolError("manager would substitute an unverified engine for this arm")

    def _reset_engine_state(self) -> None:
        invalidator = getattr(self.engine, "_prefix_cache_invalidate", None)
        if callable(invalidator):
            invalidator()
        if getattr(self.engine, "_prefix_cache", None):
            raise protocol.ProtocolError("target engine prefix cache did not reset between arms")
        if hasattr(self.engine, "_last_prompt_tokens"):
            self.engine._last_prompt_tokens = []
        if hasattr(self.engine, "_pending_assistant_prefill"):
            self.engine._pending_assistant_prefill = ""
        if getattr(self.engine, "_draft_model", None) is not None:
            raise protocol.ProtocolError("target-only executor unexpectedly loaded a draft model")
        if getattr(self.engine, "_dspark_runtime", None) is not None:
            raise protocol.ProtocolError("target-only executor unexpectedly loaded DSpark")

    def __call__(self, request: ArmRunRequest) -> ArmRunOutcome:
        import io

        from mio import agent
        from mio.prompt_policy import PromptPolicy
        from rich.console import Console

        self._assert_manager_engine_identity()
        self._reset_engine_state()
        state = {
            "tier": self.tier,
            "prompt_policy": PromptPolicy(),
            "tool_policy": request.tool_policy,
            "tool_registry": request.tool_registry,
            "tool_specs": request.tool_specs,
            "messages": [],
            "quality_gate_enabled": request.quality_gate_enabled,
            "quality_gate_require_change": request.quality_gate_enabled,
            "coding_effort": request.coding_effort,
            "execution_budget": agent.AgentExecutionBudget(
                max_rounds=TARGET_MAX_ROUNDS,
                max_tool_calls=protocol.MAX_TOOL_CALLS_PER_ARM,
                max_output_tokens=protocol.MAX_OUTPUT_TOKENS_PER_ARM,
                max_wall_seconds=protocol.MAX_AGENT_WALL_SECONDS,
                max_context_tokens=TARGET_CONTEXT_TOKENS,
            ),
        }
        previous_console = agent.console
        started = time.perf_counter()
        try:
            agent.console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
            try:
                result = agent._process_user_input(
                    request.instruction,
                    self.engine,
                    self.manager,
                    self.config,
                    state,
                )
            except protocol.ProtocolError:
                # Protocol violations are never model outcomes and must leave
                # the pair open for explicit fail-closed adjudication.
                raise
            except (MemoryError, OSError) as exc:
                raise protocol.ProtocolError("model execution suffered an infrastructure failure") from exc
            except Exception:
                # Ordinary Python exceptions from target generation are sealed
                # as a non-retryable model outcome. Host/process loss never
                # reaches this branch and leaves the whole pair explicitly open
                # for blinded infrastructure adjudication.
                elapsed = time.perf_counter() - started
                maximum_complete_wall = protocol.MAX_AGENT_WALL_SECONDS + (_EXECUTOR_WALL_OVERHEAD_NS / 1_000_000_000)
                if elapsed > maximum_complete_wall:
                    raise protocol.ProtocolError(
                        "model exception exceeded the bounded executor wall cap; overrun adjudication is required"
                    ) from None
                telemetry = None
                if self.require_raw_target_telemetry:
                    telemetry = ValidatedArmTelemetry.for_model_error(
                        quality_gate_enabled=request.quality_gate_enabled,
                        request_sha256=protocol.sha256_bytes(request.instruction.encode("utf-8")),
                        wall_elapsed_s=elapsed,
                    )
                return ArmRunOutcome(
                    status="model_error",
                    quality_gate_decision=("incomplete" if request.quality_gate_enabled else "not_applicable"),
                    wall_seconds=min(elapsed, protocol.MAX_AGENT_WALL_SECONDS),
                    telemetry=telemetry,
                )
        finally:
            agent.console = previous_console
        complete_wall_seconds = time.perf_counter() - started
        telemetry = self._validate_raw_target_result(
            result,
            request,
            wall_elapsed_s=complete_wall_seconds,
        )
        if telemetry is not None:
            turn, _quality, _rounds, _tools = telemetry.document()
            return ArmRunOutcome(
                status=turn["status"],
                quality_gate_decision=turn["quality_gate_decision"],
                output_tokens=turn["completion_tokens"],
                tool_calls=turn["tool_calls"],
                wall_seconds=min(
                    turn["wall_elapsed_ns"] / 1_000_000_000,
                    protocol.MAX_AGENT_WALL_SECONDS,
                ),
                telemetry=telemetry,
            )
        rounds = tuple(getattr(result, "rounds", ()) or ())
        output_tokens = int(
            getattr(result, "completion_tokens", 0)
            or sum(int(getattr(item, "completion_tokens", 0) or 0) for item in rounds)
        )
        terminal = getattr(result, "terminal_reason", "") == "model_final"
        if request.quality_gate_enabled:
            gate = getattr(result, "quality_gate", None)
            decision = gate.get("decision", gate.get("status")) if isinstance(gate, Mapping) else None
            satisfied = decision in {"satisfied", "complete", "pass"}
            quality_decision = "satisfied" if satisfied else "incomplete"
            terminal = terminal and satisfied
        else:
            quality_decision = "not_applicable"
        return ArmRunOutcome(
            status="completed" if terminal else "incomplete",
            quality_gate_decision=quality_decision,
            output_tokens=output_tokens,
            tool_calls=int(getattr(result, "tool_calls", 0) or 0),
            wall_seconds=min(
                float(getattr(result, "wall_time_s", 0.0) or 0.0),
                protocol.MAX_AGENT_WALL_SECONDS,
            ),
            telemetry=telemetry,
        )


def main() -> int:
    # Deliberately no implicit execution path: callers must inject a loaded
    # engine, private source resolver, and frozen bindings programmatically.
    print(
        "SWE-bench paired smoke-generation runner installed; confirmatory evidence remains "
        "hard-blocked and no model, dataset, or evaluator runs implicitly."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
