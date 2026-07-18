#!/usr/bin/env python3
"""Fail-closed local evaluator for sealed paired SWE-bench quality runs.

This wrapper is deliberately evaluation-only.  It never downloads a dataset,
image, model, or harness checkout.  It verifies a portable sealed generation
layout, exports the two official prediction streams into a new private output
tree, and invokes one pinned official harness process from a distinct working
directory for each arm.  Patch and evaluator text stay in that private tree;
stdout receives hashes and counts only.

The official harness can exit successfully while individual instances failed.
Consequently, process exit status is only the first admission check.  The
aggregate report and every per-instance report must form one coherent outcome
partition before this command writes its immutable evaluation receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import bench_swebench_quality as protocol  # noqa: E402
from scripts import run_swebench_quality_generation as generation  # noqa: E402

EVALUATION_SCHEMA = f"{generation.GENERATION_SCHEMA}.official-evaluation.v1"
IMAGE_MANIFEST_SCHEMA = f"{EVALUATION_SCHEMA}.image-manifest"
PLAN_SCHEMA = f"{EVALUATION_SCHEMA}.plan"
RECEIPT_SCHEMA = f"{EVALUATION_SCHEMA}.receipt"

# This post-v4.1.0 official commit fixes evaluation of patches that only add
# files.  A tag alone is therefore insufficient to identify the harness used.
OFFICIAL_HARNESS_COMMIT = "f7bbbb2ccdf479001d6467c9e34af59e44a840f9"
OFFICIAL_HARNESS_TREE = "81083caddb04c76896805b38eaa4e43ca3ce2d63"
DATASET_PARQUET_SHA256 = protocol.DATASET_PARQUET_SHA256

MAX_WORKERS = 1
OPEN_FILE_LIMIT = 4096
TIMEOUT_SECONDS = 1800
NAMESPACE = "swebench"
INSTANCE_IMAGE_TAG = "v2"
CACHE_LEVEL = "instance"

ARM_CONDITIONS = MappingProxyType({"plain": "gate_off", "quality": "gate_on"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOCKER_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCKER_CONTEXT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_IMAGE_REPOSITORY_RE = re.compile(r"^swebench/sweb\.eval\.x86_64\.[a-z0-9_.-]+$")
_RUN_ID_RE = re.compile(r"^mio-qwen36-27b-official-(?:plain|quality)-[0-9a-f]{16}$")
_MAX_HARNESS_TRACKED_FILES = 10_000
_MAX_HARNESS_TRACKED_BYTES = 1_000_000_000
_MAX_HARNESS_VENV_ENTRIES = 250_000
_MAX_HARNESS_VENV_BYTES = 8_000_000_000
_MAX_PYTHON_BASE_ENTRIES = 100_000
_MAX_PYTHON_BASE_BYTES = 2_000_000_000
_MAX_HARNESS_FILESYSTEM_ENTRIES = 100_000
_MAX_ATTESTED_RELATIVE_PATH_BYTES = 4_096
_ISOLATED_PROBE_CODE = (
    "import importlib.metadata as m,json,pathlib,platform,sys,sysconfig;"
    "root,site,cache=sys.argv[1:4];base=list(sys.path);"
    "sys.path[:]=[root,site,*base];sys.pycache_prefix=cache;sys.dont_write_bytecode=True;"
    "import swebench;"
    "d=sorted([{'name':(x.metadata.get('Name') or '').lower().replace('_','-'),'version':x.version} "
    "for x in m.distributions(path=[site])],key=lambda x:(x['name'],x['version']));"
    "print(json.dumps({'base_prefix':str(pathlib.Path(sys.base_prefix).resolve()),'distributions':d,"
    "'executable':str(pathlib.Path(sys.executable).resolve()),'flags':{'ignore_environment':sys.flags.ignore_environment,"
    "'isolated':sys.flags.isolated,'no_site':sys.flags.no_site,'no_user_site':sys.flags.no_user_site},"
    "'module':str(pathlib.Path(swebench.__file__).resolve()),'platstdlib':sysconfig.get_path('platstdlib'),"
    "'python':platform.python_version(),'site_packages':site,'stdlib':sysconfig.get_path('stdlib'),"
    "'sys_path':sys.path},sort_keys=True,separators=(',',':')))"
)
_ISOLATED_LAUNCHER_CODE = (
    "import runpy,sys;root,site,cache,module,*args=sys.argv[1:];base=list(sys.path);"
    "sys.path[:]=[root,site,*base];sys.pycache_prefix=cache;sys.dont_write_bytecode=True;"
    "sys.argv=[module,*args];runpy.run_module(module,run_name='__main__')"
)

_OFFLINE_ENVIRONMENT = MappingProxyType(
    {
        "DO_NOT_TRACK": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
)
_HOST_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "NO_PROXY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "TMPDIR",
        "TZ",
    }
)


class _DuplicateJSONKey(ValueError):
    pass


def _strict_json_loads(payload: str | bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJSONKey(key)
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, ValueError) as exc:
        raise protocol.ProtocolError(f"{label} is not unambiguous valid JSON") from exc


@dataclass(frozen=True)
class EvaluationOptions:
    schedule_path: Path
    generation_layout: Path
    dataset_path: Path
    harness_root: Path
    python_executable: Path
    docker_executable: Path
    docker_context: str
    image_manifest: Path
    output_root: Path
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dry_run, bool):
            raise protocol.ProtocolError("dry_run must be boolean")
        if not _DOCKER_CONTEXT_RE.fullmatch(self.docker_context):
            raise protocol.ProtocolError("Docker context name is not canonical")


@dataclass(frozen=True)
class EvaluationDependencies:
    """Dependency seam: production uses existing protocol APIs and subprocess."""

    run_process: Callable[..., Any]
    load_schedule: Callable[[Path], tuple[dict[str, Any], tuple[protocol.ScheduleEntry, ...]]]
    open_layout: Callable[[Path], generation.GenerationLayout]
    verify_generation: Callable[..., str]
    tool_surface_digest: Callable[[], str]


@dataclass(frozen=True)
class ImageBinding:
    instance_id: str
    repository: str
    manifest_digest: str

    @property
    def tagged_reference(self) -> str:
        return f"{self.repository}:{INSTANCE_IMAGE_TAG}"

    @property
    def digest_reference(self) -> str:
        return f"{self.repository}@{self.manifest_digest}"


@dataclass(frozen=True)
class PredictionArtifact:
    arm: str
    condition: str
    path: Path
    sha256: str
    rows: int
    model_label: str
    empty_prediction_ids: tuple[str, ...]
    outcome_summary: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    schedule_sha256: str
    generation_receipt_sha256: str
    plan_sha256: str
    pair_count: int
    evaluation_receipt_sha256: str | None = None
    plain_resolved: int | None = None
    quality_resolved: int | None = None

    def public_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": EVALUATION_SCHEMA,
            "status": self.status,
            "schedule_sha256": self.schedule_sha256,
            "generation_receipt_sha256": self.generation_receipt_sha256,
            "plan_sha256": self.plan_sha256,
            "pair_count": self.pair_count,
            "contains_issue_model_patch_or_evaluator_text": False,
            "network_or_download_requested_by_wrapper": False,
            "confirmatory_evidence_admissible": False,
        }
        if self.evaluation_receipt_sha256 is not None:
            result.update(
                {
                    "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
                    "plain_resolved": self.plain_resolved,
                    "quality_resolved": self.quality_resolved,
                }
            )
        return result


def production_dependencies() -> EvaluationDependencies:
    def surface_digest() -> str:
        _registry, _specs, digest = generation.build_identical_tool_surface()
        return digest

    return EvaluationDependencies(
        run_process=subprocess.run,
        load_schedule=protocol.load_private_schedule,
        open_layout=generation.GenerationLayout.open,
        verify_generation=generation.verify_sealed_generation_artifacts,
        tool_surface_digest=surface_digest,
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise protocol.ProtocolError(f"{label} is not a lowercase SHA-256")
    return value


def _require_absolute_spelling(path: Path, label: str, *, allow_final_symlink: bool = False) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or any(part in {".", ".."} for part in candidate.parts):
        raise protocol.ProtocolError(f"{label} must be an absolute canonical path")
    protocol._reject_symlink_path_components(candidate.parent if allow_final_symlink else candidate)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise protocol.ProtocolError(f"{label} does not exist") from exc
    if not allow_final_symlink and candidate != resolved:
        raise protocol.ProtocolError(f"{label} must not use a filesystem alias")
    return candidate if allow_final_symlink else resolved


def _require_private_regular_file(path: Path, label: str, *, suffix: str | None = None) -> Path:
    resolved = protocol.require_private_path(path, must_exist=True)
    if path.expanduser().absolute() != resolved:
        raise protocol.ProtocolError(f"{label} must use its canonical absolute spelling")
    try:
        metadata = resolved.lstat()
    except OSError as exc:
        raise protocol.ProtocolError(f"cannot inspect {label}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise protocol.ProtocolError(f"{label} must be a regular single-link file")
    if metadata.st_mode & 0o077:
        raise protocol.ProtocolError(f"{label} must use 0600 permissions")
    if suffix is not None and resolved.suffix != suffix:
        raise protocol.ProtocolError(f"{label} must use the {suffix} suffix")
    return resolved


def _stable_private_bytes(path: Path, label: str) -> bytes:
    payload = protocol._read_immutable_file(path)
    if payload is None:
        raise protocol.ProtocolError(f"{label} disappeared during verification")
    return payload


def _load_stable_schedule(
    path: Path,
    loader: Callable[[Path], tuple[dict[str, Any], tuple[protocol.ScheduleEntry, ...]]],
) -> tuple[dict[str, Any], tuple[protocol.ScheduleEntry, ...]]:
    schedule_path = _require_private_regular_file(path, "private schedule", suffix=".json")
    before = _stable_private_bytes(schedule_path, "private schedule")
    document, schedule = loader(schedule_path)
    after = _stable_private_bytes(schedule_path, "private schedule")
    if before != after or protocol.canonical_json_bytes(document) != before:
        raise protocol.ProtocolError("private schedule changed or is not canonical JSON")
    return document, schedule


def _expected_ids(schedule: Sequence[protocol.ScheduleEntry]) -> tuple[str, ...]:
    # Reuse the protocol's paired-structure validation rather than treating a
    # pair of duplicate prediction IDs as a harmless harness overwrite.
    identifiers = protocol._expected_instance_ids(schedule)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise protocol.ProtocolError("evaluation schedule has missing or duplicate instance IDs")
    return identifiers


def _load_and_verify_generation(
    options: EvaluationOptions,
    dependencies: EvaluationDependencies,
) -> tuple[dict[str, Any], tuple[protocol.ScheduleEntry, ...], generation.GenerationLayout, str]:
    document, schedule = _load_stable_schedule(options.schedule_path, dependencies.load_schedule)
    _expected_ids(schedule)
    layout_root = protocol.require_private_directory(options.generation_layout)
    layout = dependencies.open_layout(layout_root)
    if not getattr(layout, "portable_artifacts", False):
        raise protocol.ProtocolError("official evaluation requires a portable sealed generation layout")
    tool_surface_sha256 = _require_sha256(dependencies.tool_surface_digest(), "tool surface digest")
    receipt_sha256 = dependencies.verify_generation(
        receipt_path=layout.receipt,
        schedule=schedule,
        layout=layout,
        tool_surface_sha256=tool_surface_sha256,
    )
    _require_sha256(receipt_sha256, "sealed generation receipt digest")
    store = protocol.CheckpointStore(layout.canonical)
    expected_names = {store.path_for(entry).name for entry in schedule}
    observed_names = {path.name for path in layout.canonical.iterdir()}
    if observed_names != expected_names:
        raise protocol.ProtocolError("canonical generation directory has missing or extra entries")
    return document, schedule, layout, receipt_sha256


def _verify_dataset(path: Path) -> tuple[Path, str]:
    dataset = _require_private_regular_file(path, "SWE-bench Verified parquet", suffix=".parquet")
    digest = protocol.sha256_bytes(_stable_private_bytes(dataset, "SWE-bench Verified parquet"))
    if digest != DATASET_PARQUET_SHA256:
        raise protocol.ProtocolError("SWE-bench Verified parquet SHA-256 mismatch")
    return dataset, digest


def _normalized_path_key(path: Path) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", part).casefold() for part in path.parts)


def _require_existing_canonical_spelling(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise protocol.ProtocolError(f"{label} must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        if component != unicodedata.normalize("NFC", component):
            raise protocol.ProtocolError(f"{label} is not NFC-normalized")
        try:
            names = os.listdir(current)
        except OSError as exc:
            raise protocol.ProtocolError(f"cannot inspect {label} spelling") from exc
        key = component.casefold()
        aliases = [name for name in names if unicodedata.normalize("NFC", name).casefold() == key]
        if aliases != [component]:
            raise protocol.ProtocolError(f"{label} uses a filesystem spelling alias")
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise protocol.ProtocolError(f"{label} disappeared during spelling verification") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise protocol.ProtocolError(f"{label} contains a symlink")
    return current


def _existing_ancestor_identities(path: Path) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    current = path
    while True:
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise protocol.ProtocolError("official output ancestor must not be a symlink")
        identities.add((metadata.st_dev, metadata.st_ino))
        if current == current.parent:
            break
        current = current.parent
    return identities


def _exclusive_output_destination(path: Path, protected: Sequence[Path]) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute() or any(part in {".", ".."} for part in candidate.parts):
        raise protocol.ProtocolError("official output root must use an absolute canonical spelling")
    if candidate.name != unicodedata.normalize("NFC", candidate.name):
        raise protocol.ProtocolError("official output root is not NFC-normalized")
    parent = _require_existing_canonical_spelling(candidate.parent, "official output parent")
    canonical = parent / candidate.name
    candidate_key = unicodedata.normalize("NFC", candidate.name).casefold()
    try:
        aliases = [
            name for name in os.listdir(parent) if unicodedata.normalize("NFC", name).casefold() == candidate_key
        ]
    except OSError as exc:
        raise protocol.ProtocolError("cannot inspect official output destination") from exc
    if aliases or os.path.lexists(canonical):
        raise protocol.ProtocolError("official output root must be a new exclusive path")
    ancestor_identities = _existing_ancestor_identities(parent)
    canonical_key = _normalized_path_key(canonical)
    for raw in protected:
        protected_path = raw.resolve(strict=True)
        protected_key = _normalized_path_key(protected_path)
        lexical_overlap = (
            canonical_key[: len(protected_key)] == protected_key or protected_key[: len(canonical_key)] == canonical_key
        )
        protected_metadata = protected_path.lstat()
        identity_overlap = (
            stat.S_ISDIR(protected_metadata.st_mode)
            and (protected_metadata.st_dev, protected_metadata.st_ino) in ancestor_identities
        )
        if lexical_overlap or identity_overlap:
            raise protocol.ProtocolError("official output root overlaps an immutable input")
    return canonical


def _stable_regular_bytes(
    path: Path,
    label: str,
    *,
    maximum_bytes: int | None = None,
) -> tuple[bytes, os.stat_result]:
    """Read one no-follow regular file and reject replacement during hashing."""

    descriptor = -1
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise protocol.ProtocolError(f"{label} is not a regular file")
        if maximum_bytes is not None and (maximum_bytes < 0 or before.st_size > maximum_bytes):
            raise protocol.ProtocolError(f"{label} exceeds its attestation byte bound")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    except protocol.ProtocolError:
        raise
    except OSError as exc:
        raise protocol.ProtocolError(f"cannot read {label}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = lambda item: (  # noqa: E731 - compact immutable race tuple
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
        stat.S_IFMT(item.st_mode),
    )
    if identity(before) != identity(after) or identity(after) != identity(named):
        raise protocol.ProtocolError(f"{label} changed while it was fingerprinted")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise protocol.ProtocolError(f"{label} size changed while it was fingerprinted")
    return payload, before


def _safe_relative_path(value: str, label: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise protocol.ProtocolError(f"{label} contains an unsafe path")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise protocol.ProtocolError(f"{label} path is not canonical UTF-8") from exc
    parts = value.split("/")
    if value.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise protocol.ProtocolError(f"{label} contains an unsafe path")
    if len(encoded) > _MAX_ATTESTED_RELATIVE_PATH_BYTES:
        raise protocol.ProtocolError(f"{label} path exceeds its attestation bound")
    return value


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()  # noqa: S324


def _tracked_head_manifest(
    harness_root: Path,
    *,
    run_process: Callable[..., Any],
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], frozenset[str]]:
    """Compare every tracked worktree byte with an archive of pinned HEAD.

    ``git status`` trusts mutable index flags such as ``assume-unchanged`` and
    ``skip-worktree``.  The HEAD tree plus archive path is independent of those
    flags and therefore detects the underlying byte drift directly.
    """

    git = protocol._trusted_git()
    tree_result = _capture_process(
        run_process,
        (git, "-C", str(harness_root), "ls-tree", "-r", "-z", "HEAD"),
        cwd=harness_root,
        environment=environment,
        label="harness HEAD tree enumeration",
    )
    tree_payload = getattr(tree_result, "stdout", None)
    if not isinstance(tree_payload, bytes) or not tree_payload:
        raise protocol.ProtocolError("harness HEAD tree enumeration is empty or malformed")
    expected: dict[str, tuple[str, str]] = {}
    for record in tree_payload.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            raw_mode, raw_type, raw_object = header.split(b" ", 2)
            relative = raw_path.decode("utf-8", errors="strict")
            mode = raw_mode.decode("ascii", errors="strict")
            object_type = raw_type.decode("ascii", errors="strict")
            object_id = raw_object.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise protocol.ProtocolError("harness HEAD tree enumeration is malformed") from exc
        relative = _safe_relative_path(relative, "harness HEAD tree")
        if mode not in {"100644", "100755"} or object_type != "blob" or not re.fullmatch(r"[0-9a-f]{40}", object_id):
            raise protocol.ProtocolError("harness HEAD contains a symlink or unsupported tracked object")
        if relative in expected:
            raise protocol.ProtocolError("harness HEAD tree contains duplicate paths")
        expected[relative] = (mode, object_id)
    if not expected or len(expected) > _MAX_HARNESS_TRACKED_FILES:
        raise protocol.ProtocolError("harness tracked-file count exceeds its attestation bound")

    archive_result = _capture_process(
        run_process,
        (git, "-C", str(harness_root), "archive", "--format=tar", "HEAD"),
        cwd=harness_root,
        environment=environment,
        label="harness HEAD byte archive",
    )
    archive_payload = getattr(archive_result, "stdout", None)
    if not isinstance(archive_payload, bytes) or not archive_payload:
        raise protocol.ProtocolError("harness HEAD archive is empty or malformed")
    if len(archive_payload) > _MAX_HARNESS_TRACKED_BYTES + 128 * 1024 * 1024:
        raise protocol.ProtocolError("harness HEAD archive exceeds its attestation bound")

    rows: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    total_bytes = 0
    try:
        archive = tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:")
    except tarfile.TarError as exc:
        raise protocol.ProtocolError("harness HEAD archive is not a plain tar stream") from exc
    with archive:
        for member in archive:
            if member.isdir():
                continue
            relative = _safe_relative_path(member.name.rstrip("/"), "harness HEAD archive")
            if not member.isfile() or member.issym() or member.islnk():
                raise protocol.ProtocolError("harness HEAD archive contains an unsupported tracked object")
            expected_entry = expected.get(relative)
            if expected_entry is None or relative in observed_paths:
                raise protocol.ProtocolError("harness HEAD archive paths differ from its tree")
            observed_paths.add(relative)
            expected_mode, expected_object_id = expected_entry
            archive_mode = "100755" if member.mode & 0o111 else "100644"
            if archive_mode != expected_mode:
                raise protocol.ProtocolError("harness HEAD archive mode differs from its tree")
            handle = archive.extractfile(member)
            if handle is None:
                raise protocol.ProtocolError("harness HEAD archive member has no bytes")
            expected_bytes = handle.read()
            total_bytes += len(expected_bytes)
            if total_bytes > _MAX_HARNESS_TRACKED_BYTES:
                raise protocol.ProtocolError("harness tracked bytes exceed their attestation bound")
            if _git_blob_sha1(expected_bytes) != expected_object_id:
                raise protocol.ProtocolError("harness HEAD archive bytes differ from its tree object")
            candidate = harness_root.joinpath(*relative.split("/"))
            protocol._reject_symlink_path_components(candidate)
            observed_bytes, metadata = _stable_regular_bytes(
                candidate,
                "tracked harness file",
                maximum_bytes=len(expected_bytes),
            )
            observed_mode = "100755" if metadata.st_mode & 0o111 else "100644"
            if observed_mode != expected_mode or observed_bytes != expected_bytes:
                raise protocol.ProtocolError("tracked harness bytes differ from pinned HEAD")
            rows.append(
                {
                    "path": relative,
                    "mode": expected_mode,
                    "size_bytes": len(expected_bytes),
                    "sha256": protocol.sha256_bytes(expected_bytes),
                }
            )
    if observed_paths != set(expected):
        raise protocol.ProtocolError("harness HEAD archive is missing a tracked path")
    rows.sort(key=lambda row: row["path"])
    manifest_sha256 = protocol.sha256_bytes(protocol.canonical_json_bytes(rows))
    return (
        {
            "file_count": len(rows),
            "total_bytes": total_bytes,
            "expected_manifest_sha256": manifest_sha256,
            "observed_manifest_sha256": manifest_sha256,
            "all_tracked_bytes_match_head": True,
            "index_flags_not_trusted": True,
        },
        frozenset(expected),
    )


def _recursive_tree_manifest(
    tree_root: Path,
    *,
    label: str,
    maximum_entries: int,
    maximum_bytes: int,
) -> dict[str, Any]:
    """Recursively fingerprint every file, directory, and symlink target."""

    try:
        root_metadata = tree_root.lstat()
    except OSError as exc:
        raise protocol.ProtocolError(f"{label} root is unavailable") from exc
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise protocol.ProtocolError(f"{label} root must be an ordinary directory")
    rows: list[dict[str, Any]] = []
    pending = [tree_root]
    total_bytes = 0
    file_count = 0
    directory_count = 0
    symlink_count = 0
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name))
        except OSError as exc:
            raise protocol.ProtocolError(f"cannot enumerate {label}") from exc
        spellings = [entry.name.casefold() for entry in entries]
        if len(spellings) != len(set(spellings)):
            raise protocol.ProtocolError(f"{label} contains case-folding path aliases")
        for entry in entries:
            candidate = Path(entry.path)
            try:
                relative = candidate.relative_to(tree_root).as_posix()
                metadata = candidate.lstat()
            except (OSError, ValueError) as exc:
                raise protocol.ProtocolError(f"{label} changed during enumeration") from exc
            relative = _safe_relative_path(relative, label)
            if len(rows) >= maximum_entries:
                raise protocol.ProtocolError(f"{label} entry count exceeds its attestation bound")
            permissions = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                directory_count += 1
                rows.append({"path": relative, "kind": "directory", "mode": permissions})
                pending.append(candidate)
                continue
            if stat.S_ISREG(metadata.st_mode):
                payload, stable_metadata = _stable_regular_bytes(
                    candidate,
                    f"{label} file",
                    maximum_bytes=maximum_bytes - total_bytes,
                )
                total_bytes += len(payload)
                file_count += 1
                rows.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "mode": permissions,
                        "link_count": stable_metadata.st_nlink,
                        "size_bytes": len(payload),
                        "sha256": protocol.sha256_bytes(payload),
                    }
                )
            elif stat.S_ISLNK(metadata.st_mode):
                try:
                    target_spelling = os.readlink(candidate)
                    resolved_target = candidate.resolve(strict=True)
                    target_metadata = resolved_target.lstat()
                except (OSError, RuntimeError) as exc:
                    raise protocol.ProtocolError(f"{label} contains a dangling or cyclic symlink") from exc
                symlink_count += 1
                target_row: dict[str, Any] = {
                    "path": relative,
                    "kind": "symlink",
                    "mode": permissions,
                    "target_sha256": protocol.sha256_bytes(os.fsencode(target_spelling)),
                }
                if stat.S_ISDIR(target_metadata.st_mode):
                    try:
                        resolved_target.relative_to(tree_root)
                    except ValueError as exc:
                        raise protocol.ProtocolError(f"{label} directory symlink escapes its tree") from exc
                    target_row["target_kind"] = "directory"
                elif stat.S_ISREG(target_metadata.st_mode):
                    target_payload, _stable_target = _stable_regular_bytes(
                        resolved_target,
                        f"{label} symlink target",
                        maximum_bytes=maximum_bytes - total_bytes,
                    )
                    total_bytes += len(target_payload)
                    target_row.update(
                        {
                            "target_kind": "file",
                            "target_size_bytes": len(target_payload),
                            "target_content_sha256": protocol.sha256_bytes(target_payload),
                        }
                    )
                else:
                    raise protocol.ProtocolError(f"{label} symlink targets an unsafe filesystem object")
                rows.append(target_row)
            else:
                raise protocol.ProtocolError(f"{label} contains an unsafe filesystem object")
            if total_bytes > maximum_bytes:
                raise protocol.ProtocolError(f"{label} bytes exceed their attestation bound")
    rows.sort(key=lambda row: row["path"])
    return {
        "entry_count": len(rows),
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "total_hashed_bytes": total_bytes,
        "manifest_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(rows)),
        "complete_recursive_content_hash": True,
        "absolute_paths_serialized": False,
    }


def _venv_manifest(venv_root: Path) -> dict[str, Any]:
    document = _recursive_tree_manifest(
        venv_root,
        label="harness venv",
        maximum_entries=_MAX_HARNESS_VENV_ENTRIES,
        maximum_bytes=_MAX_HARNESS_VENV_BYTES,
    )
    document.update(
        {
            "includes_package_code_metadata_pth_and_sitecustomize": True,
            "pth_execution_disabled_by_isolated_no_site": True,
        }
    )
    return document


def _venv_site_packages(venv_root: Path) -> Path:
    candidates = sorted(venv_root.glob("lib/python*/site-packages"))
    if len(candidates) != 1:
        raise protocol.ProtocolError("harness venv must contain exactly one site-packages directory")
    candidate = candidates[0]
    protocol._reject_symlink_path_components(candidate)
    if not candidate.is_dir():
        raise protocol.ProtocolError("harness venv site-packages is unavailable")
    return candidate.resolve(strict=True)


def _pth_policy_document(venv_root: Path, site_packages: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    executable_lines = 0
    path_lines = 0
    for candidate in sorted(site_packages.rglob("*.pth")):
        protocol._reject_symlink_path_components(candidate)
        payload, _metadata = _stable_regular_bytes(
            candidate,
            "harness venv .pth file",
            maximum_bytes=1_000_000,
        )
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise protocol.ProtocolError("harness venv .pth file is not canonical UTF-8") from exc
        relative = candidate.relative_to(venv_root).as_posix()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(("import ", "import\t")):
                executable_lines += 1
            else:
                path_lines += 1
        rows.append(
            {
                "path": _safe_relative_path(relative, "harness venv .pth"),
                "size_bytes": len(payload),
                "sha256": protocol.sha256_bytes(payload),
            }
        )
    return {
        "file_count": len(rows),
        "files_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(rows)),
        "executable_line_count": executable_lines,
        "path_line_count": path_lines,
        "executed_or_added_to_sys_path": False,
        "disabled_by_python_isolated_no_site": True,
        "external_target_bytes_can_not_affect_execution": True,
    }


def _harness_auxiliary_manifest(
    harness_root: Path,
    *,
    tracked_paths: frozenset[str],
    selected_venv_root: Path,
) -> dict[str, Any]:
    """Require the exact pinned tree outside ``.git`` and one selected venv.

    Git status intentionally does not report ignored files.  Rather than try
    to classify which ignored bytes might become executable, enumerate the
    real filesystem and reject *every* entry not implied by the pinned tree.
    This also excludes stale egg-info, alternate virtual environments, empty
    directories, bytecode caches, and customization hooks from the trust base.
    """

    try:
        venv_relative = selected_venv_root.relative_to(harness_root).as_posix()
    except ValueError as exc:
        raise protocol.ProtocolError("selected harness venv is outside the checkout") from exc
    venv_relative = _safe_relative_path(venv_relative, "selected harness venv")
    if any(
        path == venv_relative or path.startswith(f"{venv_relative}/") or venv_relative.startswith(f"{path}/")
        for path in tracked_paths
    ):
        raise protocol.ProtocolError("selected harness venv overlaps the pinned tree")

    allowed_directories: set[str] = set()
    for tracked in tracked_paths:
        parts = tracked.split("/")
        allowed_directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    if ".git" in allowed_directories or ".git" in tracked_paths:
        raise protocol.ProtocolError("pinned harness tree unexpectedly contains .git")

    rows: list[dict[str, Any]] = []
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    pending = [harness_root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name))
        except OSError as exc:
            raise protocol.ProtocolError("cannot enumerate harness auxiliary state") from exc
        spellings = [unicodedata.normalize("NFC", entry.name).casefold() for entry in entries]
        if len(spellings) != len(set(spellings)):
            raise protocol.ProtocolError("harness filesystem contains path spelling aliases")
        for entry in entries:
            candidate = Path(entry.path)
            if candidate == harness_root / ".git" or candidate == selected_venv_root:
                continue
            try:
                relative = candidate.relative_to(harness_root).as_posix()
                metadata = candidate.lstat()
            except (OSError, ValueError) as exc:
                raise protocol.ProtocolError("harness auxiliary state changed during enumeration") from exc
            relative = _safe_relative_path(relative, "harness auxiliary state")
            if relative in tracked_paths:
                if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                    raise protocol.ProtocolError("pinned harness file is not an ordinary file")
                observed_files.add(relative)
                rows.append(
                    {
                        "path": relative,
                        "kind": "tracked_file",
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "size_bytes": metadata.st_size,
                    }
                )
                continue
            if stat.S_ISDIR(metadata.st_mode) and relative in allowed_directories:
                observed_directories.add(relative)
                rows.append(
                    {
                        "path": relative,
                        "kind": "tracked_prefix_directory",
                        "mode": stat.S_IMODE(metadata.st_mode),
                    }
                )
                pending.append(candidate)
                continue
            raise protocol.ProtocolError("harness filesystem contains an entry outside the pinned tree")

        if len(rows) > _MAX_HARNESS_FILESYSTEM_ENTRIES:
            raise protocol.ProtocolError("harness filesystem entry count exceeds its bound")
    if observed_files != set(tracked_paths) or observed_directories != allowed_directories:
        raise protocol.ProtocolError("harness filesystem differs from the pinned tree structure")
    rows.sort(key=lambda row: row["path"])
    return {
        "entry_count": len(rows),
        "tracked_file_count": len(observed_files),
        "tracked_prefix_directory_count": len(observed_directories),
        "manifest_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(rows)),
        "excluded_roots": [".git", venv_relative],
        "unexpected_entry_count": 0,
        "complete_outside_git_and_selected_virtual_environment": True,
        "absolute_paths_serialized": False,
    }


def _verify_harness_checkout(
    root: Path,
    python_executable: Path,
    *,
    run_process: Callable[..., Any],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    harness_root = protocol.require_private_path(root, must_exist=True)
    if root.expanduser().absolute() != harness_root or not harness_root.is_dir():
        raise protocol.ProtocolError("official harness root must be a canonical local directory")
    source = harness_root / "swebench" / "harness" / "run_evaluation.py"
    if not source.is_file() or source.is_symlink():
        raise protocol.ProtocolError("pinned harness entry point is missing")

    git = protocol._trusted_git()
    head = _capture_text(
        run_process,
        (git, "-C", str(harness_root), "rev-parse", "HEAD"),
        cwd=harness_root,
        environment=environment,
        label="harness Git head",
    ).strip()
    if head != OFFICIAL_HARNESS_COMMIT:
        raise protocol.ProtocolError("official harness checkout differs from the pinned commit")
    tree = _capture_text(
        run_process,
        (git, "-C", str(harness_root), "rev-parse", "HEAD^{tree}"),
        cwd=harness_root,
        environment=environment,
        label="harness Git tree",
    ).strip()
    if tree != OFFICIAL_HARNESS_TREE:
        raise protocol.ProtocolError("official harness Git tree differs from the pinned tree")
    status_text = _capture_text(
        run_process,
        (git, "-C", str(harness_root), "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=harness_root,
        environment=environment,
        label="harness Git status",
    )
    if status_text:
        raise protocol.ProtocolError("official harness checkout has tracked or untracked source drift")
    tracked_manifest, tracked_paths = _tracked_head_manifest(
        harness_root,
        run_process=run_process,
        environment=environment,
    )

    python_path = _require_absolute_spelling(
        python_executable,
        "harness Python executable",
        allow_final_symlink=True,
    )
    if python_path.parent.name != "bin" or python_path.parent.parent.parent != harness_root:
        raise protocol.ProtocolError("harness Python executable must belong to a venv inside the pinned checkout")
    target = python_path.resolve(strict=True)
    if not target.is_file() or not os.access(python_path, os.X_OK):
        raise protocol.ProtocolError("harness Python executable is not executable")
    venv_root = python_path.parent.parent
    site_packages = _venv_site_packages(venv_root)
    filesystem_manifest = _harness_auxiliary_manifest(
        harness_root,
        tracked_paths=tracked_paths,
        selected_venv_root=venv_root,
    )
    venv_manifest = _venv_manifest(venv_root)
    pth_policy = _pth_policy_document(venv_root, site_packages)
    probe_cache = venv_root / ".mio-isolated-probe-pycache"
    if os.path.lexists(probe_cache):
        raise protocol.ProtocolError("isolated harness probe cache path must not already exist")
    probe = _capture_json(
        run_process,
        (
            str(python_path),
            "-I",
            "-S",
            "-c",
            _ISOLATED_PROBE_CODE,
            str(harness_root),
            str(site_packages),
            str(probe_cache),
        ),
        cwd=harness_root,
        environment=environment,
        label="harness Python import probe",
    )
    if set(probe) != {
        "base_prefix",
        "distributions",
        "executable",
        "flags",
        "module",
        "platstdlib",
        "python",
        "site_packages",
        "stdlib",
        "sys_path",
    }:
        raise protocol.ProtocolError("harness Python import probe has unexpected fields")
    module = probe.get("module")
    version = probe.get("python")
    executable = probe.get("executable")
    distributions = probe.get("distributions")
    base_prefix = probe.get("base_prefix")
    flags = probe.get("flags")
    probe_site = probe.get("site_packages")
    stdlib = probe.get("stdlib")
    platstdlib = probe.get("platstdlib")
    sys_path = probe.get("sys_path")
    if (
        not isinstance(module, str)
        or not isinstance(version, str)
        or not isinstance(executable, str)
        or not isinstance(distributions, list)
        or not distributions
        or not isinstance(base_prefix, str)
        or not isinstance(probe_site, str)
        or not isinstance(stdlib, str)
        or not isinstance(platstdlib, str)
        or not isinstance(sys_path, list)
        or len(sys_path) < 4
    ):
        raise protocol.ProtocolError("harness Python import probe is malformed")
    if flags != {"ignore_environment": 1, "isolated": 1, "no_site": 1, "no_user_site": 1}:
        raise protocol.ProtocolError("harness Python probe was not isolated with site disabled")
    if Path(executable).resolve(strict=True) != target:
        raise protocol.ProtocolError("harness Python probe executed another interpreter")
    try:
        observed_site = Path(probe_site).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise protocol.ProtocolError("harness Python probe site-packages is unavailable") from exc
    if observed_site != site_packages:
        raise protocol.ProtocolError("harness Python probe used another site-packages directory")

    try:
        base_root = Path(base_prefix).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise protocol.ProtocolError("harness Python base prefix is unavailable") from exc
    if base_prefix != str(base_root) or not base_root.is_dir() or base_root == venv_root:
        raise protocol.ProtocolError("harness Python base prefix is not canonical")
    try:
        target.relative_to(base_root)
    except ValueError as exc:
        raise protocol.ProtocolError("harness Python executable target is outside its base prefix") from exc
    if sys_path[:2] != [str(harness_root), str(site_packages)] or not all(
        isinstance(item, str) and item for item in sys_path
    ):
        raise protocol.ProtocolError("harness Python isolated sys.path is malformed")
    for raw_path in [*sys_path[2:], stdlib, platstdlib]:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            raise protocol.ProtocolError("harness Python base path is not absolute")
        try:
            candidate.resolve(strict=False).relative_to(base_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise protocol.ProtocolError("harness Python base path escapes its attested prefix") from exc
    for raw_directory in (stdlib, platstdlib):
        if not Path(raw_directory).is_dir():
            raise protocol.ProtocolError("harness Python standard-library path is unavailable")

    normalized_distributions: list[dict[str, str]] = []
    for row in distributions:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "version"}
            or not isinstance(row["name"], str)
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", row["name"])
            or not isinstance(row["version"], str)
            or not row["version"]
            or len(row["version"]) > 128
        ):
            raise protocol.ProtocolError("harness Python distribution identity is malformed")
        normalized_distributions.append({"name": row["name"], "version": row["version"]})
    if normalized_distributions != sorted(
        normalized_distributions,
        key=lambda row: (row["name"], row["version"]),
    ):
        raise protocol.ProtocolError("harness Python distributions are not canonical")
    try:
        module_relative = Path(module).resolve(strict=True).relative_to(harness_root).as_posix()
    except (OSError, ValueError) as exc:
        raise protocol.ProtocolError("harness Python imports SWE-bench from another checkout") from exc
    if module_relative not in tracked_paths:
        raise protocol.ProtocolError("harness Python imports untracked SWE-bench package bytes")
    python_base_manifest = _recursive_tree_manifest(
        base_root,
        label="Python base prefix",
        maximum_entries=_MAX_PYTHON_BASE_ENTRIES,
        maximum_bytes=_MAX_PYTHON_BASE_BYTES,
    )
    return {
        "git_commit": head,
        "git_tree": tree,
        "root": str(harness_root),
        "entrypoint_sha256": protocol.sha256_file(source),
        "python_executable": str(python_path),
        "python_target_sha256": protocol.sha256_file(target),
        "python_version": version,
        "python_base_prefix": str(base_root),
        "python_base_manifest": python_base_manifest,
        "site_packages": str(site_packages),
        "distribution_count": len(normalized_distributions),
        "distributions_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(normalized_distributions)),
        "tracked_head_manifest": tracked_manifest,
        "venv_manifest": venv_manifest,
        "pth_policy": pth_policy,
        "filesystem_manifest": filesystem_manifest,
        "isolated_no_site_execution": True,
        "tracked_worktree_clean": True,
        "untracked_and_ignored_entries_absent": True,
    }


def _load_image_manifest(path: Path, expected_ids: Sequence[str]) -> tuple[tuple[ImageBinding, ...], str]:
    manifest_path = _require_private_regular_file(path, "official image manifest", suffix=".json")
    payload = _stable_private_bytes(manifest_path, "official image manifest")
    document = _strict_json_loads(payload, "official image manifest")
    expected_keys = {"schema", "namespace", "instance_image_tag", "images"}
    if (
        not isinstance(document, dict)
        or set(document) != expected_keys
        or protocol.canonical_json_bytes(document) != payload
    ):
        raise protocol.ProtocolError("official image manifest is not canonical or has unexpected fields")
    if (
        document["schema"] != IMAGE_MANIFEST_SCHEMA
        or document["namespace"] != NAMESPACE
        or document["instance_image_tag"] != INSTANCE_IMAGE_TAG
        or not isinstance(document["images"], list)
    ):
        raise protocol.ProtocolError("official image manifest policy differs from the evaluator")

    bindings: list[ImageBinding] = []
    for raw in document["images"]:
        if not isinstance(raw, dict) or set(raw) != {"instance_id", "repository", "manifest_digest"}:
            raise protocol.ProtocolError("official image record fields are invalid")
        instance_id = raw["instance_id"]
        repository = raw["repository"]
        manifest_digest = raw["manifest_digest"]
        if not isinstance(instance_id, str) or not isinstance(repository, str) or not isinstance(manifest_digest, str):
            raise protocol.ProtocolError("official image record values are invalid")
        expected_repository = f"{NAMESPACE}/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}"
        if repository != expected_repository or not _IMAGE_REPOSITORY_RE.fullmatch(repository):
            raise protocol.ProtocolError("official image repository does not match its instance ID")
        if not _DOCKER_DIGEST_RE.fullmatch(manifest_digest):
            raise protocol.ProtocolError("official image manifest digest is malformed")
        bindings.append(ImageBinding(instance_id, repository, manifest_digest))
    observed_ids = [binding.instance_id for binding in bindings]
    if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) != set(expected_ids):
        raise protocol.ProtocolError("official image manifest has duplicate, missing, or extra instance IDs")
    return tuple(sorted(bindings, key=lambda item: item.instance_id)), protocol.sha256_bytes(payload)


def _capture_process(
    run_process: Callable[..., Any],
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    label: str,
) -> Any:
    try:
        result = run_process(
            tuple(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise protocol.ProtocolError(f"{label} could not be executed") from exc
    if not isinstance(getattr(result, "returncode", None), int) or result.returncode != 0:
        raise protocol.ProtocolError(f"{label} failed")
    return result


def _capture_text(
    run_process: Callable[..., Any],
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    label: str,
) -> str:
    result = _capture_process(run_process, command, cwd=cwd, environment=environment, label=label)
    stdout = getattr(result, "stdout", None)
    if not isinstance(stdout, bytes):
        raise protocol.ProtocolError(f"{label} did not return byte output")
    try:
        return stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise protocol.ProtocolError(f"{label} output is not UTF-8") from exc


def _capture_json(
    run_process: Callable[..., Any],
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    label: str,
) -> Any:
    text = _capture_text(run_process, command, cwd=cwd, environment=environment, label=label)
    return _strict_json_loads(text, label)


def _docker_host(
    docker_executable: Path,
    context: str,
    *,
    run_process: Callable[..., Any],
    cwd: Path,
    environment: Mapping[str, str],
) -> tuple[Path, str]:
    docker = _require_absolute_spelling(docker_executable, "Docker executable", allow_final_symlink=True)
    if not docker.resolve(strict=True).is_file() or not os.access(docker, os.X_OK):
        raise protocol.ProtocolError("Docker executable is not executable")
    raw = _capture_json(
        run_process,
        (str(docker), "context", "inspect", context),
        cwd=cwd,
        environment=environment,
        label="Docker context inspection",
    )
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict) or raw[0].get("Name") != context:
        raise protocol.ProtocolError("Docker context inspection returned another context")
    try:
        host = raw[0]["Endpoints"]["docker"]["Host"]
    except (KeyError, TypeError) as exc:
        raise protocol.ProtocolError("Docker context lacks a local engine endpoint") from exc
    if not isinstance(host, str) or not host.startswith("unix://"):
        raise protocol.ProtocolError("official evaluation requires a local Unix Docker endpoint")
    socket_path = Path(host.removeprefix("unix://"))
    if not socket_path.is_absolute() or socket_path.is_symlink():
        raise protocol.ProtocolError("Docker Unix endpoint path is not canonical")
    try:
        socket_metadata = socket_path.lstat()
    except OSError as exc:
        raise protocol.ProtocolError("Docker Unix endpoint is unavailable") from exc
    if not stat.S_ISSOCK(socket_metadata.st_mode):
        raise protocol.ProtocolError("Docker endpoint is not a Unix socket")
    return docker, host


def _docker_cli_prefix(docker: Path, host: str) -> tuple[str, ...]:
    return (str(docker), "--host", host)


def _inspect_docker(
    docker: Path,
    host: str,
    bindings: Sequence[ImageBinding],
    *,
    run_process: Callable[..., Any],
    cwd: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    prefix = _docker_cli_prefix(docker, host)
    server = _capture_json(
        run_process,
        (*prefix, "version", "--format", "{{json .Server}}"),
        cwd=cwd,
        environment=environment,
        label="Docker server inspection",
    )
    if not isinstance(server, dict) or server.get("Os") != "linux" or server.get("Arch") != "amd64":
        raise protocol.ProtocolError("official images require a linux/amd64 Docker engine")
    if any(
        not isinstance(server.get(name), str) or not server[name] for name in ("Version", "ApiVersion", "GitCommit")
    ):
        raise protocol.ProtocolError("Docker server identity fields are malformed")

    images: list[dict[str, str]] = []
    for binding in bindings:
        inspected = _capture_json(
            run_process,
            (*prefix, "image", "inspect", binding.tagged_reference, binding.digest_reference),
            cwd=cwd,
            environment=environment,
            label="local official image inspection",
        )
        if (
            not isinstance(inspected, list)
            or len(inspected) != 2
            or not all(isinstance(row, dict) for row in inspected)
        ):
            raise protocol.ProtocolError("local official image inspection is malformed")
        tagged, digested = inspected
        image_id = tagged.get("Id")
        if (
            not isinstance(image_id, str)
            or not _DOCKER_DIGEST_RE.fullmatch(image_id)
            or digested.get("Id") != image_id
            or tagged.get("Architecture") != "amd64"
            or tagged.get("Os") != "linux"
            or digested.get("Architecture") != "amd64"
            or digested.get("Os") != "linux"
        ):
            raise protocol.ProtocolError("local official image tag/digest/platform binding differs")
        repo_digests = tagged.get("RepoDigests")
        if not isinstance(repo_digests, list) or binding.digest_reference not in repo_digests:
            raise protocol.ProtocolError("local official image lacks its pinned repository digest")
        images.append(
            {
                "instance_id": binding.instance_id,
                "repository": binding.repository,
                "tagged_reference": binding.tagged_reference,
                "manifest_digest": binding.manifest_digest,
                "local_image_id": image_id,
                "platform": "linux/amd64",
            }
        )
    return {
        "host": host,
        "server_version": server.get("Version"),
        "server_api_version": server.get("ApiVersion"),
        "server_git_commit": server.get("GitCommit"),
        "platform": "linux/amd64",
        "images": images,
    }


def _ensure_no_run_containers(
    docker: Path,
    host: str,
    run_id: str,
    *,
    run_process: Callable[..., Any],
    cwd: Path,
    environment: Mapping[str, str],
) -> None:
    output = _capture_text(
        run_process,
        (*_docker_cli_prefix(docker, host), "ps", "-a", "--filter", f"name={run_id}", "--format", "{{.ID}}"),
        cwd=cwd,
        environment=environment,
        label="Docker container cleanup inspection",
    )
    if output.strip():
        raise protocol.ProtocolError("official harness left a run container behind")


def _new_private_subdirectory(parent: Path, name: str) -> Path:
    path = parent / name
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise protocol.ProtocolError(f"private {name} destination already exists") from exc
    os.chmod(path, 0o700)
    return path


def _export_predictions(
    schedule: Sequence[protocol.ScheduleEntry],
    store: protocol.CheckpointStore,
    output_directory: Path,
) -> tuple[dict[str, PredictionArtifact], dict[str, str]]:
    expected_ids = set(_expected_ids(schedule))
    expected_schedule_sha256 = protocol.schedule_digest(schedule)
    arm_by_condition = {condition: arm for arm, condition in ARM_CONDITIONS.items()}
    rows: dict[str, list[dict[str, str]]] = {arm: [] for arm in ARM_CONDITIONS}
    status_counts: dict[str, dict[str, int]] = {
        arm: {status: 0 for status in ("completed", "incomplete", "model_error", "timeout")} for arm in ARM_CONDITIONS
    }
    decision_counts: dict[str, dict[str, int]] = {
        arm: {decision: 0 for decision in ("incomplete", "not_applicable", "satisfied")} for arm in ARM_CONDITIONS
    }
    empty_reasons: dict[str, dict[str, int]] = {arm: {"empty_model_patch": 0} for arm in ARM_CONDITIONS}
    empty_ids: dict[str, list[str]] = {arm: [] for arm in ARM_CONDITIONS}
    bindings: set[tuple[str, str, str]] = set()
    for entry in schedule:
        checkpoint = store.load(entry)
        if (
            checkpoint.preregistration_sha256 != protocol.preregistration_digest()
            or checkpoint.schedule_sha256 != expected_schedule_sha256
            or checkpoint.model_identity != protocol.EXPECTED_MODEL_IDENTITY
        ):
            raise protocol.ProtocolError("official prediction checkpoint binding differs from the frozen study")
        arm = arm_by_condition.get(entry.condition)
        if arm is None:
            raise protocol.ProtocolError("scheduled condition has no official evaluation arm")
        if checkpoint.status not in status_counts[arm] or checkpoint.quality_gate_decision not in decision_counts[arm]:
            raise protocol.ProtocolError("official prediction checkpoint has a non-terminal outcome vocabulary")
        status_counts[arm][checkpoint.status] += 1
        decision_counts[arm][checkpoint.quality_gate_decision] += 1
        # Export the trusted Git capture for every terminal status.  Status is
        # retained independently in the outcome counters; selecting patches by
        # status would create a second, post-hoc exclusion rule.
        published_patch = checkpoint.model_patch
        if not published_patch:
            empty_ids[arm].append(entry.instance_id)
            empty_reasons[arm]["empty_model_patch"] += 1
        rows[arm].append(protocol.official_prediction(entry.instance_id, entry.condition, published_patch))
        bindings.add((checkpoint.mio_commit, checkpoint.model_identity, checkpoint.runtime_digest))
    if len(bindings) != 1:
        raise protocol.ProtocolError("paired predictions do not share one Mio/model/runtime binding")

    artifacts: dict[str, PredictionArtifact] = {}
    for arm, condition in ARM_CONDITIONS.items():
        arm_rows = sorted(rows[arm], key=lambda row: row["instance_id"])
        identifiers = [row["instance_id"] for row in arm_rows]
        if len(set(identifiers)) != len(identifiers) or set(identifiers) != expected_ids:
            raise protocol.ProtocolError(f"{arm} predictions have duplicate, missing, or extra instance IDs")
        path = output_directory / f"{arm}.predictions.jsonl"
        payload = protocol.canonical_jsonl_bytes(arm_rows)
        protocol._atomic_write(path, payload)
        observed = _stable_private_bytes(path, f"{arm} predictions")
        if observed != payload:
            raise protocol.ProtocolError(f"{arm} prediction publication changed bytes")
        artifacts[arm] = PredictionArtifact(
            arm=arm,
            condition=condition,
            path=path,
            sha256=protocol.sha256_bytes(payload),
            rows=len(arm_rows),
            model_label=protocol.MODEL_LABELS[condition],
            empty_prediction_ids=tuple(sorted(empty_ids[arm])),
            outcome_summary={
                "scheduled_terminal_checkpoints": len(arm_rows),
                "checkpoint_status_counts": status_counts[arm],
                "quality_gate_decision_counts": decision_counts[arm],
                "empty_prediction_reason_counts": empty_reasons[arm],
                "empty_prediction_count": len(empty_ids[arm]),
                "nonempty_prediction_count": len(arm_rows) - len(empty_ids[arm]),
                "empty_prediction_ids_sha256": protocol.sha256_bytes(
                    protocol.canonical_json_bytes(sorted(empty_ids[arm]))
                ),
                "all_scheduled_terminal_outcomes_exported": True,
                "terminal_outcomes_not_selected_by_patch_availability": True,
            },
        )
    mio_commit, model_identity, runtime_digest = bindings.pop()
    return artifacts, {
        "mio_commit": mio_commit,
        "model_identity": model_identity,
        "runtime_digest": runtime_digest,
    }


def _run_id(arm: str, schedule_sha256: str, prediction_sha256: str) -> str:
    if arm not in ARM_CONDITIONS:
        raise protocol.ProtocolError("unknown official evaluation arm")
    material = {
        "arm": arm,
        "schedule_sha256": _require_sha256(schedule_sha256, "schedule digest"),
        "prediction_sha256": _require_sha256(prediction_sha256, "prediction digest"),
        "dataset_sha256": DATASET_PARQUET_SHA256,
        "harness_commit": OFFICIAL_HARNESS_COMMIT,
        "parameters": _command_parameters(),
    }
    suffix = protocol.sha256_bytes(protocol.canonical_json_bytes(material))[:16]
    return f"mio-qwen36-27b-official-{arm}-{suffix}"


def _command_parameters() -> dict[str, Any]:
    return {
        "split": "test",
        "max_workers": MAX_WORKERS,
        "open_file_limit": OPEN_FILE_LIMIT,
        "timeout_seconds": TIMEOUT_SECONDS,
        "force_rebuild": False,
        "cache_level": CACHE_LEVEL,
        "clean": False,
        "namespace": NAMESPACE,
        "instance_image_tag": INSTANCE_IMAGE_TAG,
    }


def _harness_command(
    python_executable: Path,
    harness_root: Path,
    site_packages: Path,
    isolated_cache: Path,
    dataset: Path,
    prediction: PredictionArtifact,
    expected_ids: Sequence[str],
    run_id: str,
) -> tuple[str, ...]:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise protocol.ProtocolError("official evaluation run ID is malformed")
    return (
        str(python_executable),
        "-I",
        "-S",
        "-c",
        _ISOLATED_LAUNCHER_CODE,
        str(harness_root),
        str(site_packages),
        str(isolated_cache),
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(dataset),
        "--split",
        "test",
        "--predictions_path",
        str(prediction.path),
        "--instance_ids",
        *expected_ids,
        "--max_workers",
        str(MAX_WORKERS),
        "--open_file_limit",
        str(OPEN_FILE_LIMIT),
        "--timeout",
        str(TIMEOUT_SECONDS),
        "--force_rebuild",
        "false",
        "--cache_level",
        CACHE_LEVEL,
        "--clean",
        "false",
        "--run_id",
        run_id,
        "--namespace",
        NAMESPACE,
        "--instance_image_tag",
        INSTANCE_IMAGE_TAG,
    )


def _privatize_tree(root: Path) -> None:
    for current, directory_names, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path.is_symlink():
            raise protocol.ProtocolError("official harness output contains a directory symlink")
        os.chmod(current_path, 0o700)
        for name in (*directory_names, *filenames):
            candidate = current_path / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise protocol.ProtocolError("official harness output contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                os.chmod(candidate, 0o700)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                os.chmod(candidate, 0o600)
            else:
                raise protocol.ProtocolError("official harness output contains an unsafe filesystem object")


def _validated_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value) or len(set(value)) != len(value):
        raise protocol.ProtocolError(f"official report {label} is not a unique string list")
    return value


def _validate_test_status(status: Any) -> bool:
    expected_categories = {"FAIL_TO_PASS", "PASS_TO_PASS", "FAIL_TO_FAIL", "PASS_TO_FAIL"}
    if not isinstance(status, dict) or set(status) != expected_categories:
        raise protocol.ProtocolError("per-instance tests_status is missing")
    resolution_categories = {"FAIL_TO_PASS", "PASS_TO_PASS"}
    failures = False
    for category in sorted(expected_categories):
        row = status.get(category)
        if not isinstance(row, dict) or set(row) != {"success", "failure"}:
            raise protocol.ProtocolError("per-instance tests_status category is malformed")
        success = _validated_string_list(row["success"], f"{category}.success")
        failure = _validated_string_list(row["failure"], f"{category}.failure")
        if set(success) & set(failure):
            raise protocol.ProtocolError("per-instance tests_status partitions overlap")
        if category in resolution_categories:
            failures = failures or bool(failure)
    return not failures


def _validate_reports(
    arm_directory: Path,
    prediction: PredictionArtifact,
    run_id: str,
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    expected = set(expected_ids)
    empty_expected = set(prediction.empty_prediction_ids)
    if not empty_expected.issubset(expected):
        raise protocol.ProtocolError("official empty-prediction set is outside the schedule")
    evaluated_expected = expected - empty_expected
    report_path = arm_directory / f"{prediction.model_label}.{run_id}.json"
    payload = _stable_private_bytes(report_path, "official aggregate report")
    report = _strict_json_loads(payload, "official aggregate report")
    expected_report_fields = {
        "schema_version",
        "total_instances",
        "submitted_instances",
        "completed_instances",
        "resolved_instances",
        "unresolved_instances",
        "empty_patch_instances",
        "error_instances",
        "completed_ids",
        "incomplete_ids",
        "empty_patch_ids",
        "submitted_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
    }
    if not isinstance(report, dict) or set(report) != expected_report_fields or report.get("schema_version") != 2:
        raise protocol.ProtocolError("official aggregate report must use schema_version 2")
    count = len(expected_ids)
    count_fields = {
        "total_instances": count,
        "submitted_instances": count,
        "completed_instances": len(evaluated_expected),
        "empty_patch_instances": len(empty_expected),
        "error_instances": 0,
    }
    for name, wanted in count_fields.items():
        value = report.get(name)
        if isinstance(value, bool) or value != wanted:
            raise protocol.ProtocolError(f"official aggregate report {name} is not admissible")
    completed = set(_validated_string_list(report.get("completed_ids"), "completed_ids"))
    submitted = set(_validated_string_list(report.get("submitted_ids"), "submitted_ids"))
    incomplete = set(_validated_string_list(report.get("incomplete_ids"), "incomplete_ids"))
    empty = set(_validated_string_list(report.get("empty_patch_ids"), "empty_patch_ids"))
    errors = set(_validated_string_list(report.get("error_ids"), "error_ids"))
    resolved = set(_validated_string_list(report.get("resolved_ids"), "resolved_ids"))
    unresolved = set(_validated_string_list(report.get("unresolved_ids"), "unresolved_ids"))
    if completed != evaluated_expected or submitted != expected or incomplete or empty != empty_expected or errors:
        raise protocol.ProtocolError("official aggregate report IDs are incomplete or erroneous")
    if resolved & unresolved or resolved | unresolved != evaluated_expected:
        raise protocol.ProtocolError("official aggregate resolved/unresolved sets are incoherent")
    resolved_count = report.get("resolved_instances")
    unresolved_count = report.get("unresolved_instances")
    if (
        isinstance(resolved_count, bool)
        or isinstance(unresolved_count, bool)
        or resolved_count != len(resolved)
        or unresolved_count != len(unresolved)
    ):
        raise protocol.ProtocolError("official aggregate outcome counts differ from their ID sets")

    instance_reports: list[dict[str, Any]] = []
    report_root = arm_directory / "logs" / "run_evaluation" / run_id / prediction.model_label
    absent_empty_reports: list[str] = []
    for instance_id in expected_ids:
        instance_path = report_root / instance_id / "report.json"
        if instance_id in empty_expected:
            if os.path.lexists(instance_path):
                raise protocol.ProtocolError("official harness produced a report for an empty prediction")
            absent_empty_reports.append(instance_id)
            continue
        instance_payload = _stable_private_bytes(instance_path, "official per-instance report")
        document = _strict_json_loads(instance_payload, "official per-instance report")
        if (
            not isinstance(document, dict)
            or set(document) != {instance_id}
            or not isinstance(document[instance_id], dict)
        ):
            raise protocol.ProtocolError("official per-instance report is bound to another instance")
        row = document[instance_id]
        if set(row) != {
            "patch_is_None",
            "patch_exists",
            "patch_successfully_applied",
            "resolved",
            "tests_status",
        }:
            raise protocol.ProtocolError("official per-instance report fields differ from the pinned harness")
        if (
            row.get("patch_is_None") is not False
            or row.get("patch_exists") is not True
            or row.get("patch_successfully_applied") is not True
            or not isinstance(row.get("resolved"), bool)
        ):
            raise protocol.ProtocolError("official per-instance patch application fields are not admissible")
        recomputed_resolved = _validate_test_status(row.get("tests_status"))
        if row["resolved"] is not recomputed_resolved or row["resolved"] != (instance_id in resolved):
            raise protocol.ProtocolError("official per-instance resolution differs from tests_status or aggregate")
        instance_reports.append(
            {
                "instance_id": instance_id,
                "sha256": protocol.sha256_bytes(instance_payload),
                "resolved": row["resolved"],
            }
        )
    return {
        "aggregate_report": {
            "filename": report_path.name,
            "sha256": protocol.sha256_bytes(payload),
            "schema_version": 2,
        },
        "per_instance_reports": instance_reports,
        "per_instance_reports_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(instance_reports)),
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(unresolved),
        "empty_prediction_ids": sorted(empty_expected),
        "empty_prediction_count": len(empty_expected),
        "empty_prediction_reports_absent": sorted(absent_empty_reports),
        "effective_unresolved_ids": sorted(unresolved | empty_expected),
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "effective_unresolved_count": len(unresolved | empty_expected),
        "effective_outcome_count": len(resolved | unresolved | empty_expected),
        "all_scheduled_outcomes_accounted_without_selection": ((resolved | unresolved | empty_expected) == expected),
    }


def _run_harness(
    command: Sequence[str],
    *,
    arm_directory: Path,
    environment: Mapping[str, str],
    run_process: Callable[..., Any],
    process_timeout_seconds: int,
) -> str:
    if process_timeout_seconds <= TIMEOUT_SECONDS:
        raise protocol.ProtocolError("official harness process timeout is not above the per-instance timeout")
    log_path = arm_directory / "harness.log"
    try:
        handle = log_path.open("xb")
    except FileExistsError as exc:
        raise protocol.ProtocolError("private harness log already exists") from exc
    process_error: BaseException | None = None
    result: Any = None
    with handle:
        os.fchmod(handle.fileno(), 0o600)
        try:
            result = run_process(
                tuple(command),
                cwd=arm_directory,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=process_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            process_error = exc
        handle.flush()
        os.fsync(handle.fileno())
    _privatize_tree(arm_directory)
    if process_error is not None:
        raise protocol.ProtocolError("official harness process could not be executed") from process_error
    if not isinstance(getattr(result, "returncode", None), int) or result.returncode != 0:
        raise protocol.ProtocolError("official harness process failed; private log retained")
    return protocol.sha256_bytes(_stable_private_bytes(log_path, "private harness log"))


def _host_environment() -> dict[str, str]:
    environment = {name: value for name, value in os.environ.items() if name in _HOST_ENVIRONMENT_ALLOWLIST}
    environment.update(_OFFLINE_ENVIRONMENT)
    return environment


def _base_environment(docker_host: str) -> dict[str, str]:
    environment = _host_environment()
    environment["DOCKER_HOST"] = docker_host
    environment.pop("DOCKER_CONTEXT", None)
    return environment


def run_official_evaluation(
    options: EvaluationOptions,
    *,
    dependencies: EvaluationDependencies | None = None,
) -> EvaluationResult:
    deps = dependencies or production_dependencies()
    schedule_document, schedule, layout, generation_receipt_sha256 = _load_and_verify_generation(options, deps)
    expected_ids = _expected_ids(schedule)
    schedule_sha256 = protocol.schedule_digest(schedule)
    dataset, dataset_sha256 = _verify_dataset(options.dataset_path)

    # Preflight commands run without a Docker host first; the context probe
    # resolves one immutable local Unix endpoint that all later commands use.
    bootstrap_environment = _host_environment()
    harness = _verify_harness_checkout(
        options.harness_root,
        options.python_executable,
        run_process=deps.run_process,
        environment=bootstrap_environment,
    )
    images, image_manifest_sha256 = _load_image_manifest(options.image_manifest, expected_ids)
    output_destination = _exclusive_output_destination(
        options.output_root,
        (
            layout.root,
            dataset,
            Path(harness["root"]),
            options.image_manifest,
            options.schedule_path,
        ),
    )
    docker, docker_host = _docker_host(
        options.docker_executable,
        options.docker_context,
        run_process=deps.run_process,
        cwd=Path(harness["root"]),
        environment=bootstrap_environment,
    )
    environment = _base_environment(docker_host)
    docker_attestation = _inspect_docker(
        docker,
        docker_host,
        images,
        run_process=deps.run_process,
        cwd=Path(harness["root"]),
        environment=environment,
    )

    output_root = protocol.create_private_directory(output_destination)
    predictions_directory = _new_private_subdirectory(output_root, "predictions")
    artifacts, generation_binding = _export_predictions(
        schedule,
        protocol.CheckpointStore(layout.canonical),
        predictions_directory,
    )
    arm_directories = {arm: _new_private_subdirectory(output_root, arm) for arm in ARM_CONDITIONS}
    run_ids = {arm: _run_id(arm, schedule_sha256, artifact.sha256) for arm, artifact in artifacts.items()}
    commands = {
        arm: _harness_command(
            Path(harness["python_executable"]),
            Path(harness["root"]),
            Path(harness["site_packages"]),
            arm_directories[arm] / "isolated-python-cache",
            dataset,
            artifacts[arm],
            expected_ids,
            run_ids[arm],
        )
        for arm in ARM_CONDITIONS
    }
    process_timeout_seconds = TIMEOUT_SECONDS * len(expected_ids) + 600
    plan = {
        "schema": PLAN_SCHEMA,
        "dry_run": options.dry_run,
        "generation": {
            "layout": str(layout.root),
            "receipt_sha256": generation_receipt_sha256,
            "schedule_sha256": schedule_sha256,
            "binding": generation_binding,
        },
        "dataset": {
            "name": protocol.DATASET_NAME,
            "revision": protocol.DATASET_REVISION,
            "path": str(dataset),
            "parquet_sha256": dataset_sha256,
        },
        "harness": harness,
        "docker": {
            "context": options.docker_context,
            "image_manifest_sha256": image_manifest_sha256,
            **docker_attestation,
        },
        "parameters": _command_parameters(),
        "scheduled_instance_ids": list(expected_ids),
        "scheduled_instance_ids_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(list(expected_ids))),
        "arms": {
            arm: {
                "condition": artifacts[arm].condition,
                "prediction_filename": artifacts[arm].path.name,
                "prediction_sha256": artifacts[arm].sha256,
                "prediction_rows": artifacts[arm].rows,
                "model_label": artifacts[arm].model_label,
                "generation_outcomes": dict(artifacts[arm].outcome_summary),
                "run_id": run_ids[arm],
                "working_directory": str(arm_directories[arm]),
                "command": list(commands[arm]),
                "command_sha256": protocol.sha256_bytes(protocol.canonical_json_bytes(list(commands[arm]))),
                "wrapper_process_timeout_seconds": process_timeout_seconds,
            }
            for arm in ARM_CONDITIONS
        },
        "network_or_download_requested_by_wrapper": False,
        "all_dataset_harness_and_image_inputs_preexisting": True,
        "subprocess_environment_allowlist": sorted(_HOST_ENVIRONMENT_ALLOWLIST),
        "offline_environment_keys": sorted(_OFFLINE_ENVIRONMENT),
        "contains_patch_or_evaluator_text": False,
        "evidence_class": schedule_document["evidence_class"],
        "confirmatory_evidence_admissible": False,
    }
    plan_path = output_root / "evaluation-plan.json"
    plan_payload = protocol.canonical_json_bytes(plan)
    protocol._atomic_write(plan_path, plan_payload)
    if _stable_private_bytes(plan_path, "immutable evaluation plan") != plan_payload:
        raise protocol.ProtocolError("immutable evaluation plan publication changed bytes")
    plan_sha256 = protocol.sha256_bytes(plan_payload)
    if options.dry_run:
        return EvaluationResult(
            status="dry_run_preflight_complete",
            schedule_sha256=schedule_sha256,
            generation_receipt_sha256=generation_receipt_sha256,
            plan_sha256=plan_sha256,
            pair_count=len(expected_ids),
        )

    arm_results: dict[str, dict[str, Any]] = {}
    for arm in ARM_CONDITIONS:
        if (
            _inspect_docker(
                docker,
                docker_host,
                images,
                run_process=deps.run_process,
                cwd=arm_directories[arm],
                environment=environment,
            )
            != docker_attestation
        ):
            raise protocol.ProtocolError("Docker engine or official image bindings changed before an arm")
        _ensure_no_run_containers(
            docker,
            docker_host,
            run_ids[arm],
            run_process=deps.run_process,
            cwd=arm_directories[arm],
            environment=environment,
        )
        log_sha256 = _run_harness(
            commands[arm],
            arm_directory=arm_directories[arm],
            environment=environment,
            run_process=deps.run_process,
            process_timeout_seconds=process_timeout_seconds,
        )
        _ensure_no_run_containers(
            docker,
            docker_host,
            run_ids[arm],
            run_process=deps.run_process,
            cwd=arm_directories[arm],
            environment=environment,
        )
        report = _validate_reports(arm_directories[arm], artifacts[arm], run_ids[arm], expected_ids)
        arm_results[arm] = {
            **plan["arms"][arm],
            "harness_log_sha256": log_sha256,
            **report,
        }

    post_docker = _inspect_docker(
        docker,
        docker_host,
        images,
        run_process=deps.run_process,
        cwd=output_root,
        environment=environment,
    )
    if post_docker != docker_attestation:
        raise protocol.ProtocolError("Docker engine or official image bindings changed during evaluation")
    if (
        protocol.sha256_bytes(_stable_private_bytes(options.image_manifest, "post-evaluation image manifest"))
        != image_manifest_sha256
    ):
        raise protocol.ProtocolError("official image manifest changed during evaluation")
    post_harness = _verify_harness_checkout(
        options.harness_root,
        options.python_executable,
        run_process=deps.run_process,
        environment=bootstrap_environment,
    )
    if post_harness != harness:
        raise protocol.ProtocolError("official harness checkout or Python changed during evaluation")
    post_dataset_sha256 = protocol.sha256_bytes(
        _stable_private_bytes(dataset, "post-evaluation SWE-bench Verified parquet")
    )
    if post_dataset_sha256 != dataset_sha256:
        raise protocol.ProtocolError("SWE-bench Verified parquet changed during evaluation")
    for artifact in artifacts.values():
        if (
            protocol.sha256_bytes(_stable_private_bytes(artifact.path, "post-evaluation predictions"))
            != artifact.sha256
        ):
            raise protocol.ProtocolError("official predictions changed during evaluation")
    post_document, post_schedule = _load_stable_schedule(options.schedule_path, deps.load_schedule)
    if post_document != schedule_document or post_schedule != schedule:
        raise protocol.ProtocolError("private schedule changed during evaluation")
    post_generation_receipt_sha256 = deps.verify_generation(
        receipt_path=layout.receipt,
        schedule=schedule,
        layout=layout,
        tool_surface_sha256=_require_sha256(deps.tool_surface_digest(), "post-evaluation tool surface digest"),
    )
    if post_generation_receipt_sha256 != generation_receipt_sha256:
        raise protocol.ProtocolError("sealed generation artifacts changed during evaluation")
    if protocol.sha256_bytes(_stable_private_bytes(plan_path, "post-evaluation immutable plan")) != plan_sha256:
        raise protocol.ProtocolError("immutable evaluation plan changed during evaluation")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "plan_sha256": plan_sha256,
        "generation": plan["generation"],
        "dataset": plan["dataset"],
        "harness": plan["harness"],
        "docker": plan["docker"],
        "parameters": plan["parameters"],
        "scheduled_instance_ids_sha256": plan["scheduled_instance_ids_sha256"],
        "scheduled_instances": len(expected_ids),
        "arms": arm_results,
        "paired_outcomes": {
            "plain_resolved": arm_results["plain"]["resolved_count"],
            "quality_resolved": arm_results["quality"]["resolved_count"],
            "quality_minus_plain": arm_results["quality"]["resolved_count"] - arm_results["plain"]["resolved_count"],
            "plain_raw_harness_unresolved": arm_results["plain"]["unresolved_count"],
            "quality_raw_harness_unresolved": arm_results["quality"]["unresolved_count"],
            "plain_empty_predictions": arm_results["plain"]["empty_prediction_count"],
            "quality_empty_predictions": arm_results["quality"]["empty_prediction_count"],
            "plain_effective_unresolved": arm_results["plain"]["effective_unresolved_count"],
            "quality_effective_unresolved": arm_results["quality"]["effective_unresolved_count"],
            "empty_predictions_count_as_effective_unresolved": True,
            "all_preregistered_terminal_outcomes_included": True,
        },
        "official_harness_process_exit_zero_is_not_sufficient": True,
        "all_reports_and_patch_application_fields_validated": True,
        "selection_by_terminal_status_or_patch_availability": False,
        "network_or_download_requested_by_wrapper": False,
        "contains_patch_or_evaluator_text": False,
        "evidence_class": plan["evidence_class"],
        "confirmatory_evidence_admissible": False,
    }
    receipt_path = output_root / "evaluation-receipt.json"
    receipt_payload = protocol.canonical_json_bytes(receipt)
    protocol._atomic_write(receipt_path, receipt_payload)
    if _stable_private_bytes(receipt_path, "immutable evaluation receipt") != receipt_payload:
        raise protocol.ProtocolError("immutable evaluation receipt publication changed bytes")
    receipt_sha256 = protocol.sha256_bytes(receipt_payload)
    return EvaluationResult(
        status="sealed_official_evaluation",
        schedule_sha256=schedule_sha256,
        generation_receipt_sha256=generation_receipt_sha256,
        plan_sha256=plan_sha256,
        pair_count=len(expected_ids),
        evaluation_receipt_sha256=receipt_sha256,
        plain_resolved=arm_results["plain"]["resolved_count"],
        quality_resolved=arm_results["quality"]["resolved_count"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True, help="Canonical private paired schedule (0600).")
    parser.add_argument("--generation-layout", type=Path, required=True, help="Portable sealed generation layout.")
    parser.add_argument("--dataset", type=Path, required=True, help="Pinned local Verified parquet (0600).")
    parser.add_argument("--harness-root", type=Path, required=True, help="Clean pinned official SWE-bench checkout.")
    parser.add_argument("--python-executable", type=Path, required=True, help="Python from a venv inside harness root.")
    parser.add_argument("--docker-executable", type=Path, required=True, help="Local Docker CLI executable.")
    parser.add_argument("--docker-context", required=True, help="Local linux/amd64 Docker context name.")
    parser.add_argument("--image-manifest", type=Path, required=True, help="Canonical private image digest manifest.")
    parser.add_argument("--output-root", type=Path, required=True, help="New exclusive private evaluation root.")
    parser.add_argument("--dry-run", action="store_true", help="Verify/export/plan without invoking the harness.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = EvaluationOptions(
        schedule_path=args.schedule,
        generation_layout=args.generation_layout,
        dataset_path=args.dataset,
        harness_root=args.harness_root,
        python_executable=args.python_executable,
        docker_executable=args.docker_executable,
        docker_context=args.docker_context,
        image_manifest=args.image_manifest,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    try:
        result = run_official_evaluation(options)
    except protocol.ProtocolError as exc:
        print(f"official SWE-bench evaluation blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.public_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
