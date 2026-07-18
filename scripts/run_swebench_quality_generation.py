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
TOOL_SURFACE = ("bash", "validate", "read", "write", "edit")
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


@dataclass(frozen=True)
class ArmRunOutcome:
    """Content-free terminal outcome returned by an executor."""

    status: str
    quality_gate_decision: str
    output_tokens: int = 0
    tool_calls: int = 0
    wall_seconds: float = 0.0


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
) -> dict[str, Any]:
    """Bind smoke execution inputs before the first pair is admitted.

    Caller-supplied identities are marked smoke-only.  The automatic path
    records recomputable preflight digests, but does not claim clean-subprocess
    or in-memory end-to-end provenance.  Resume and receipt creation cannot
    re-declare a different factor or binding.
    """

    factor = factor_document(tool_surface_sha256)
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
        "loaded_target_binding": _executor_model_binding_document(binding, executor, tier_config),
        "tool_surface_sha256": tool_surface_sha256,
        "factor_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(factor)),
        "factor": factor,
        "runner_source_sha256": protocol.sha256_file(Path(__file__)),
        "executor": _implementation_identity(executor),
        "workspace_factory": _implementation_identity(workspace_factory),
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
    ledger: Path
    run_header: Path
    receipt: Path

    def validated(self) -> "GenerationLayout":
        root = protocol.require_private_directory(self.root)
        attempts = protocol.require_private_directory(self.attempts)
        canonical = protocol.require_private_directory(self.canonical)
        if attempts.parent != root or canonical.parent != root:
            raise protocol.ProtocolError("generation layout directories must be direct private children")
        for path, label in (
            (self.ledger, "attempt ledger"),
            (self.run_header, "run header"),
            (self.receipt, "generation receipt"),
        ):
            if path.absolute().parent.resolve(strict=True) != root:
                raise protocol.ProtocolError(f"{label} must remain separate beside canonical output")
            if path.is_symlink():
                raise protocol.ProtocolError(f"{label} must not be a symlink")
        if (
            self.ledger.name != "pair-attempt-ledger.jsonl"
            or self.run_header.name != "generation-run-header.json"
            or self.receipt.name != "generation-receipt.json"
        ):
            raise protocol.ProtocolError("generation layout artifact names differ from the sealed design")
        return GenerationLayout(
            root,
            attempts,
            canonical,
            self.ledger.absolute(),
            self.run_header.absolute(),
            self.receipt.absolute(),
        )

    @classmethod
    def create(cls, root: Path) -> "GenerationLayout":
        root = protocol.create_private_directory(root)
        attempts = _mkdir_private(root / "attempts")
        canonical = _mkdir_private(root / "canonical")
        return cls(
            root=root,
            attempts=attempts,
            canonical=canonical,
            ledger=root / "pair-attempt-ledger.jsonl",
            run_header=root / "generation-run-header.json",
            receipt=root / "generation-receipt.json",
        )

    @classmethod
    def open(cls, root: Path) -> "GenerationLayout":
        root = protocol.require_private_directory(root)
        attempts = protocol.require_private_directory(root / "attempts")
        canonical = protocol.require_private_directory(root / "canonical")
        return cls(
            root=root,
            attempts=attempts,
            canonical=canonical,
            ledger=root / "pair-attempt-ledger.jsonl",
            run_header=root / "generation-run-header.json",
            receipt=root / "generation-receipt.json",
        )


def _load_run_header(layout: GenerationLayout) -> dict[str, Any]:
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
    if header.get("runner_source_sha256") != protocol.sha256_file(Path(__file__)):
        raise protocol.ProtocolError("current runner source differs from the immutable run header")
    return header


def _seal_run_header(layout: GenerationLayout, expected: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload = protocol.canonical_json_bytes(dict(expected))
    protocol._atomic_write(layout.run_header, payload)
    observed = _load_run_header(layout)
    if observed != dict(expected):
        raise protocol.ProtocolError("generation run header differs from current execution inputs")
    return observed, protocol.sha256_bytes(payload)


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


def _promote_completed_pair(
    layout: GenerationLayout,
    pair: Sequence[protocol.ScheduleEntry],
    completed: Mapping[str, Any],
) -> None:
    attempt_index = int(completed["attempt_index"])
    attempt_store = protocol.pair_attempt_store(layout.attempts, pair[0].pair_index, attempt_index)
    hashes = _checkpoint_hashes(attempt_store, pair)
    if hashes != completed["checkpoint_sha256s"]:
        raise protocol.ProtocolError("completed ledger event differs from retained attempt")
    canonical_store = protocol.CheckpointStore(layout.canonical)
    for entry in pair:
        checkpoint = attempt_store.load(entry)
        destination = canonical_store.save(checkpoint)
        if protocol._immutable_file_sha256(destination) != hashes[entry.condition]:
            raise protocol.ProtocolError("canonical promotion changed checkpoint bytes")


def pending_pairs(
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    *,
    repair_completed_promotions: bool = True,
) -> tuple[tuple[protocol.ScheduleEntry, ...], ...]:
    """Resume only complete pairs; never continue after one arm."""

    layout = layout.validated()
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
            _promote_completed_pair(layout, pair, completed)
        elif not all(existing):
            raise protocol.ProtocolError("sealed canonical pair is incomplete")
        expected_hashes = completed["checkpoint_sha256s"]
        if _checkpoint_hashes(canonical_store, pair) != expected_hashes:
            raise protocol.ProtocolError("canonical pair differs from completed attempt")
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
) -> str:
    """Run every missing *whole pair* and return the frozen factor digest.

    Confirmatory execution is intentionally delegated to the adapter's global
    readiness gate.  With protocol v1 this currently raises before any model
    call, preventing accidental unblinding while controls remain pending.
    """

    layout = layout.validated()
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
    registry, specs, surface_sha256 = build_identical_tool_surface(agent_module)
    pending = pending_pairs(schedule, layout)
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
        ),
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
            attempt_store.save(checkpoint)

        hashes = _checkpoint_hashes(attempt_store, pair)
        completed = ledger.append(
            pair_index=pair_index,
            attempt_index=attempt_index,
            event="completed",
            reason_code="completed",
            checkpoint_sha256s=hashes,
        )
        _promote_completed_pair(layout, pair, completed)
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
        rows.append(
            {
                "execution_index": entry.execution_index,
                "pair_index": entry.pair_index,
                "position_in_pair": entry.position_in_pair,
                "condition": entry.condition,
                "instance_digest": checkpoint.instance_digest,
                "checkpoint_sha256": protocol._immutable_file_sha256(store.path_for(entry)),
            }
        )
    return rows


def build_generation_receipt(
    *,
    schedule: Sequence[protocol.ScheduleEntry],
    layout: GenerationLayout,
    binding: GenerationBinding,
    tool_surface_sha256: str,
    observed_model_identity_before: str,
    observed_model_identity_after: str,
) -> dict[str, Any]:
    """Build a canonical, content-free receipt after all pairs are sealed."""

    if {
        observed_model_identity_before,
        observed_model_identity_after,
    } != {protocol.EXPECTED_MODEL_IDENTITY}:
        raise protocol.ProtocolError("generation receipt target identity checks differ")
    run_header = _load_run_header(layout)
    if run_header.get("generation_binding") != binding.as_dict():
        raise protocol.ProtocolError("receipt binding differs from the immutable run header")
    evidence_run = run_header.get("evidence_class") == "confirmatory"
    binding.validate_for_run(evidence_run=evidence_run)
    if run_header.get("generation_binding_attestation") != binding.attestation_dict():
        raise protocol.ProtocolError("receipt attestation differs from the immutable run header")
    if run_header.get("tool_surface_sha256") != tool_surface_sha256:
        raise protocol.ProtocolError("receipt tool surface differs from the immutable run header")
    if run_header.get("schedule_sha256") != protocol.schedule_digest(schedule):
        raise protocol.ProtocolError("receipt schedule differs from the immutable run header")
    if {
        observed_model_identity_before,
        observed_model_identity_after,
    } != {str(run_header["generation_binding"]["model_identity"])}:
        raise protocol.ProtocolError("receipt identity observations differ from the immutable run header")
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
            "before_first_generation": observed_model_identity_before,
            "after_last_generation": observed_model_identity_after,
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
        "contains_model_text_or_evaluator_output": False,
        "evidence_class": run_header["evidence_class"],
        "confirmatory_evidence_admissible": False,
        "confirmatory_blockers": list(CONFIRMATORY_BLOCKERS),
    }


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
    """Recompute every ledger/checkpoint/factor binding without evaluation data."""

    layout = layout.validated()
    path = protocol.require_private_path(receipt_path, must_exist=True)
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
    expected = build_generation_receipt(
        schedule=schedule,
        layout=layout,
        binding=binding,
        tool_surface_sha256=tool_surface_sha256,
        observed_model_identity_before=protocol.EXPECTED_MODEL_IDENTITY,
        observed_model_identity_after=protocol.EXPECTED_MODEL_IDENTITY,
    )
    if observed != expected:
        raise protocol.ProtocolError("generation receipt differs from current sealed artifacts")
    return protocol.sha256_bytes(payload)


class NativeMioArmExecutor:
    """Adapter for one loaded target-only Mio engine with fresh state per arm."""

    def __init__(self, *, engine: Any, manager: Any, config: Any, tier: str) -> None:
        validate_target_only_tier(getattr(engine, "tier_config", None))
        self.engine = engine
        self.manager = manager
        self.config = config
        self.tier = tier

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
            except Exception:
                # Ordinary Python exceptions from target generation are sealed
                # as a non-retryable model outcome. Host/process loss never
                # reaches this branch and leaves the whole pair explicitly open
                # for blinded infrastructure adjudication.
                elapsed = time.perf_counter() - started
                if elapsed > protocol.MAX_AGENT_WALL_SECONDS:
                    raise protocol.ProtocolError(
                        "model exception exceeded the frozen wall cap; v2 overrun adjudication is required"
                    ) from None
                return ArmRunOutcome(
                    status="model_error",
                    quality_gate_decision=("incomplete" if request.quality_gate_enabled else "not_applicable"),
                    wall_seconds=elapsed,
                )
        finally:
            agent.console = previous_console
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
            wall_seconds=float(getattr(result, "wall_time_s", 0.0) or 0.0),
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
