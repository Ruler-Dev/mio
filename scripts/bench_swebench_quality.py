#!/usr/bin/env python3
"""Frozen adapter for Mio's paired SWE-bench Verified quality experiment.

This file deliberately does not download SWE-bench, start a 27B model, or run
Docker when imported.  It provides the protocol-critical, dependency-free
parts of the study: input redaction, balanced scheduling, immutable arm
checkpoints, workspace-to-patch conversion, official prediction export,
environment preflight, and source-free paired aggregation.

The model-facing runner remains separate.  It must materialize a fresh checkout
at ``base_commit``, expose only the redacted public instance fields, call Mio
with the frozen arm configuration, and then invoke :func:`capture_git_patch`.
Assistant prose is never accepted as a patch.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shlex
import shutil
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PREREGISTRATION_PATH = ROOT / "benchmarks" / "swebench-quality-preregistration-v1.json"
SCHEMA = "mio.swebench-verified-quality-run.v1"
AGGREGATE_SCHEMA = "mio.swebench-verified-quality-source-free-aggregate.v1"
DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
DATASET_PARQUET_SHA256 = "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
FULL_SNAPSHOT_SHA256 = "52ccbc6ec0e03085f95191b261e0ed881cd6a0752a3c5247c1aba258ec2993da"
PUBLIC_SNAPSHOT_SHA256 = "9116deb4b3b24346a278373cf1551bd8cee4e0677776105f62b4474ca50dfaba"
FULL_SNAPSHOT_FILENAME = "swebench-verified-full-v1.jsonl"
PUBLIC_SNAPSHOT_FILENAME = "swebench-verified-public-v1.jsonl"
HARNESS_VERSION = "4.1.0"
HARNESS_COMMIT = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
EXPECTED_INSTANCES = 500
SCHEDULE_SEED = 20260718
BOOTSTRAP_SAMPLES = 10_000
CONFIRMATORY_TIMEOUT_SECONDS = 1800
MAX_OUTPUT_TOKENS_PER_ARM = 24_576
MAX_TOOL_CALLS_PER_ARM = 32
MAX_AGENT_WALL_SECONDS = 1800.0
CONFIRMATORY_GENERATION_ATTESTATION_IMPLEMENTED = False
CONFIRMATORY_RETRY_LEDGER_INTEGRATION_IMPLEMENTED = False
CONFIRMATORY_EFFICIENCY_GUARDRAIL_IMPLEMENTED = False
CONFIRMATORY_DOCKER_IMAGE_DIGEST_CAPTURE_IMPLEMENTED = False
CONDITIONS = ("gate_off", "gate_on")
MODEL_LABELS = {
    "gate_off": "mio-qwen36-27b-gate-off",
    "gate_on": "mio-qwen36-27b-gate-on",
}
EXPECTED_MODEL_IDENTITY = "local-sha256-v1:ba3975accc6b6398f47f82ff7640b39f5541abb49f1d3c6f34113aa7fb040c87"
PUBLIC_INSTANCE_KEYS = frozenset({"instance_id", "repo", "base_commit", "problem_statement"})
FULL_INSTANCE_KEYS = (
    "repo",
    "instance_id",
    "base_commit",
    "patch",
    "test_patch",
    "problem_statement",
    "hints_text",
    "created_at",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "environment_setup_commit",
    "difficulty",
)
FORBIDDEN_INSTANCE_KEYS = frozenset(
    {
        "patch",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "hints_text",
        "difficulty",
        "official_reports",
        "peer_arm_patch",
        "peer_arm_trajectory",
    }
)
_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_IDENTITY_RE = re.compile(r"^local-sha256-v1:[0-9a-f]{64}$")
_TERMINAL_STATUSES = frozenset({"completed", "incomplete", "model_error", "timeout"})
_INFRASTRUCTURE_REASONS = frozenset(
    {
        "infrastructure_process_crash",
        "infrastructure_host_loss",
        "infrastructure_telemetry_corruption",
        "infrastructure_evaluator_failure",
    }
)


class ProtocolError(RuntimeError):
    """Raised when an observation cannot be admitted to the frozen study."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlink_path_components(path: Path) -> None:
    absolute = path.expanduser().absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ProtocolError("private artifact path contains a symlink component")


def require_private_path(path: Path, *, must_exist: bool) -> Path:
    """Require private/gold state outside the repository and outside symlink trees."""

    _reject_symlink_path_components(path)
    absolute = path.expanduser().absolute()
    if must_exist:
        resolved = absolute.resolve(strict=True)
    else:
        parent = absolute.parent.resolve(strict=True)
        resolved = parent / absolute.name
    if _is_within(resolved, ROOT.resolve(strict=True)):
        raise ProtocolError("private and gold artifacts must remain outside the Mio repository")
    return resolved


def create_private_directory(path: Path) -> Path:
    destination = require_private_path(path, must_exist=False)
    try:
        destination.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise ProtocolError("private destination must be new and exclusive") from exc
    os.chmod(destination, 0o700)
    if any(destination.iterdir()):
        raise ProtocolError("new private destination is unexpectedly non-empty")
    return destination


def require_private_directory(path: Path) -> Path:
    directory = require_private_path(path, must_exist=True)
    if not directory.is_dir() or directory.stat().st_mode & 0o077:
        raise ProtocolError("private artifact directory must use 0700 permissions")
    return directory


def require_confirmatory_generation_attestation(evidence_run: bool) -> None:
    pending = []
    if not CONFIRMATORY_GENERATION_ATTESTATION_IMPLEMENTED:
        pending.append("isolated generation runner and generation receipt")
    if not CONFIRMATORY_RETRY_LEDGER_INTEGRATION_IMPLEMENTED:
        pending.append("generation retry-ledger integration")
    if not CONFIRMATORY_EFFICIENCY_GUARDRAIL_IMPLEMENTED:
        pending.append("efficiency guardrail")
    if not CONFIRMATORY_DOCKER_IMAGE_DIGEST_CAPTURE_IMPLEMENTED:
        pending.append("Docker image-digest capture")
    if evidence_run and pending:
        raise ProtocolError(
            "confirmatory SWE-bench is blocked until these controls are implemented: " + ", ".join(pending)
        )


def canonical_json_bytes(value: Any) -> bytes:
    """Return one deterministic JSON representation, rejecting NaN/Infinity."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _public_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {key: raw[key] for key in ("instance_id", "repo", "base_commit", "problem_statement")}


def verify_full_snapshot(path: Path) -> Path:
    resolved = require_private_path(path, must_exist=True)
    if not resolved.is_file() or resolved.suffix != ".jsonl":
        raise ProtocolError("official full SWE-bench snapshot must be a JSONL file")
    if resolved.stat().st_mode & 0o077:
        raise ProtocolError("official full SWE-bench snapshot must use private 0600 permissions")
    if sha256_file(resolved) != FULL_SNAPSHOT_SHA256:
        raise ProtocolError("official full SWE-bench snapshot SHA-256 mismatch")
    return resolved


def verify_public_snapshot(path: Path) -> Path:
    resolved = require_private_path(path, must_exist=True)
    if not resolved.is_file() or resolved.suffix != ".jsonl":
        raise ProtocolError("official public SWE-bench snapshot must be a JSONL file")
    if resolved.stat().st_mode & 0o077:
        raise ProtocolError("official public SWE-bench snapshot must use private 0600 permissions")
    if sha256_file(resolved) != PUBLIC_SNAPSHOT_SHA256:
        raise ProtocolError("official public SWE-bench snapshot SHA-256 mismatch")
    return resolved


def prepare_official_snapshots(parquet_path: Path, output_directory: Path) -> dict[str, Path]:
    """Redact the exact official parquet into two canonical immutable snapshots."""

    source = require_private_path(parquet_path, must_exist=True)
    if not source.is_file() or sha256_file(source) != DATASET_PARQUET_SHA256:
        raise ProtocolError("official SWE-bench Verified parquet SHA-256 mismatch")
    if source.stat().st_mode & 0o077:
        raise ProtocolError("gold parquet must use private 0600 permissions")
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ProtocolError("prepare requires pyarrow; no dataset fallback is permitted") from exc

    table = parquet.read_table(source)
    if tuple(table.column_names) != FULL_INSTANCE_KEYS or table.num_rows != EXPECTED_INSTANCES:
        raise ProtocolError("official parquet schema or 500-row cardinality mismatch")
    rows = table.to_pylist()
    if any(set(row) != set(FULL_INSTANCE_KEYS) for row in rows):
        raise ProtocolError("official parquet row keys mismatch")
    if any(not all(isinstance(value, str) for value in row.values()) for row in rows):
        raise ProtocolError("official parquet contains a non-string task field")
    rows.sort(key=lambda row: row["instance_id"])
    public_rows = [_public_row(row) for row in rows]
    public_instances = tuple(PublicInstance.from_mapping(row) for row in public_rows)
    if len({instance.instance_id for instance in public_instances}) != EXPECTED_INSTANCES:
        raise ProtocolError("official parquet contains duplicate task identifiers")

    full_payload = canonical_jsonl_bytes(rows)
    public_payload = canonical_jsonl_bytes(public_rows)
    if sha256_bytes(full_payload) != FULL_SNAPSHOT_SHA256:
        raise ProtocolError("canonical full snapshot differs from the preregistered SHA-256")
    if sha256_bytes(public_payload) != PUBLIC_SNAPSHOT_SHA256:
        raise ProtocolError("canonical public snapshot differs from the preregistered SHA-256")

    destination = create_private_directory(output_directory)
    paths = {
        "full": destination / FULL_SNAPSHOT_FILENAME,
        "public": destination / PUBLIC_SNAPSHOT_FILENAME,
    }
    _atomic_write(paths["full"], full_payload)
    _atomic_write(paths["public"], public_payload)
    return paths


def preregistration_digest(path: Path = PREREGISTRATION_PATH) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "mio.swebench-verified-quality-preregistration.v1":
        raise ProtocolError("unexpected SWE-bench quality preregistration schema")
    return sha256_bytes(canonical_json_bytes(payload))


@dataclass(frozen=True)
class PublicInstance:
    """The only SWE-bench fields allowed to cross the model input firewall."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PublicInstance":
        keys = set(raw)
        forbidden = keys & FORBIDDEN_INSTANCE_KEYS
        if forbidden:
            raise ProtocolError(f"manifest contains forbidden evaluator fields: {sorted(forbidden)}")
        extra = keys - PUBLIC_INSTANCE_KEYS
        missing = PUBLIC_INSTANCE_KEYS - keys
        if extra or missing:
            raise ProtocolError(f"public manifest keys differ: missing={sorted(missing)}, extra={sorted(extra)}")
        instance = cls(
            instance_id=str(raw["instance_id"]),
            repo=str(raw["repo"]),
            base_commit=str(raw["base_commit"]),
            problem_statement=str(raw["problem_statement"]),
        )
        if not _INSTANCE_ID_RE.fullmatch(instance.instance_id):
            raise ProtocolError("invalid SWE-bench instance_id")
        if not _REPO_RE.fullmatch(instance.repo):
            raise ProtocolError("invalid SWE-bench repo identifier")
        if not _COMMIT_RE.fullmatch(instance.base_commit):
            raise ProtocolError("base_commit must be a lowercase 40-character SHA-1")
        if not instance.problem_statement.strip():
            raise ProtocolError("problem_statement must be non-empty")
        return instance

    def as_dict(self) -> dict[str, str]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
        }


def _load_json_or_jsonl(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("instances"), list):
            value = value["instances"]
        if not isinstance(value, list):
            raise ProtocolError("manifest must be a JSON list, {instances: [...]}, or JSONL")
        rows = value
    return rows


def load_public_manifest(
    path: Path,
    *,
    expected_count: int = EXPECTED_INSTANCES,
    evidence_run: bool = True,
) -> tuple[PublicInstance, ...]:
    path = require_private_path(path, must_exist=True)
    if evidence_run:
        verify_public_snapshot(path)
    rows = _load_json_or_jsonl(path)
    instances = tuple(PublicInstance.from_mapping(row) for row in rows)
    identifiers = [instance.instance_id for instance in instances]
    if len(set(identifiers)) != len(identifiers):
        raise ProtocolError("public instance manifest contains duplicate instance_id values")
    if evidence_run and len(instances) != EXPECTED_INSTANCES:
        raise ProtocolError("confirmatory evidence requires all 500 SWE-bench Verified instances")
    if len(instances) != expected_count:
        raise ProtocolError(f"expected {expected_count} instances, found {len(instances)}")
    return instances


@dataclass(frozen=True)
class ScheduleEntry:
    pair_index: int
    execution_index: int
    instance_id: str
    instance_digest: str
    condition: str
    position_in_pair: int

    def private_dict(self) -> dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "execution_index": self.execution_index,
            "instance_id": self.instance_id,
            "instance_digest": self.instance_digest,
            "condition": self.condition,
            "position_in_pair": self.position_in_pair,
        }


def _instance_rank(instance_id: str, seed: int) -> str:
    return sha256_bytes(f"mio-swebench-order-v1\0{seed}\0{instance_id}".encode())


def _instance_digest(instance_id: str) -> str:
    return sha256_bytes(f"mio-swebench-instance-v1\0{instance_id}".encode())


def make_balanced_schedule(
    instance_ids: Sequence[str],
    *,
    seed: int = SCHEDULE_SEED,
    require_full: bool = True,
) -> tuple[ScheduleEntry, ...]:
    """Create adjacent pairs with deterministic, exactly balanced arm order."""

    identifiers = tuple(str(value) for value in instance_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ProtocolError("schedule requires non-empty unique instance IDs")
    if any(not _INSTANCE_ID_RE.fullmatch(value) for value in identifiers):
        raise ProtocolError("schedule contains an invalid SWE-bench instance_id")
    if require_full and len(identifiers) != EXPECTED_INSTANCES:
        raise ProtocolError("confirmatory schedule requires exactly 500 pairs")
    if len(identifiers) % 2:
        raise ProtocolError("balanced schedule requires an even number of instances")

    ranked = sorted(identifiers, key=lambda value: (_instance_rank(value, seed), value))
    entries: list[ScheduleEntry] = []
    for pair_index, instance_id in enumerate(ranked):
        order = CONDITIONS if pair_index % 2 == 0 else tuple(reversed(CONDITIONS))
        for position, condition in enumerate(order):
            entries.append(
                ScheduleEntry(
                    pair_index=pair_index,
                    execution_index=len(entries),
                    instance_id=instance_id,
                    instance_digest=_instance_digest(instance_id),
                    condition=condition,
                    position_in_pair=position,
                )
            )
    return tuple(entries)


def schedule_digest(schedule: Sequence[ScheduleEntry]) -> str:
    return sha256_bytes(canonical_json_bytes([entry.private_dict() for entry in schedule]))


def source_free_schedule_summary(schedule: Sequence[ScheduleEntry]) -> dict[str, Any]:
    if len(schedule) % 2:
        raise ProtocolError("schedule has an incomplete pair")
    off_first = sum(entry.condition == "gate_off" and entry.position_in_pair == 0 for entry in schedule)
    on_first = sum(entry.condition == "gate_on" and entry.position_in_pair == 0 for entry in schedule)
    return {
        "schedule_sha256": schedule_digest(schedule),
        "pairs": len(schedule) // 2,
        "arms": len(schedule),
        "gate_off_first_pairs": off_first,
        "gate_on_first_pairs": on_first,
        "pair_arms_adjacent": all(
            schedule[index].pair_index == schedule[index + 1].pair_index for index in range(0, len(schedule), 2)
        ),
    }


def private_schedule_document(
    instances: Sequence[PublicInstance],
    *,
    evidence_run: bool,
) -> dict[str, Any]:
    schedule = make_balanced_schedule(
        [instance.instance_id for instance in instances],
        require_full=evidence_run,
    )
    by_id = {instance.instance_id: instance for instance in instances}
    public_rows = [by_id[identifier].as_dict() for identifier in sorted(by_id)]
    public_snapshot_sha256 = sha256_bytes(canonical_jsonl_bytes(public_rows))
    if evidence_run and public_snapshot_sha256 != PUBLIC_SNAPSHOT_SHA256:
        raise ProtocolError("confirmatory schedule requires the exact official public snapshot")
    return {
        "schema": SCHEMA,
        "evidence_class": "confirmatory" if evidence_run else "non_evidence_smoke",
        "preregistration_sha256": preregistration_digest(),
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "dataset_full_snapshot_sha256": FULL_SNAPSHOT_SHA256,
        "dataset_public_snapshot_sha256": public_snapshot_sha256,
        "expected_model_identity": EXPECTED_MODEL_IDENTITY,
        "schedule": [entry.private_dict() for entry in schedule],
        "public_instances": public_rows,
        "source_free_summary": source_free_schedule_summary(schedule),
    }


def load_private_schedule(path: Path) -> tuple[dict[str, Any], tuple[ScheduleEntry, ...]]:
    path = require_private_path(path, must_exist=True)
    if path.stat().st_mode & 0o077:
        raise ProtocolError("private schedule must use 0600 permissions")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ProtocolError("unexpected private schedule schema")
    if document.get("preregistration_sha256") != preregistration_digest():
        raise ProtocolError("private schedule is bound to another preregistration")
    if document.get("dataset") != DATASET_NAME or document.get("dataset_revision") != DATASET_REVISION:
        raise ProtocolError("private schedule dataset identity mismatch")
    if document.get("dataset_full_snapshot_sha256") != FULL_SNAPSHOT_SHA256:
        raise ProtocolError("private schedule full snapshot binding mismatch")
    if document.get("expected_model_identity") != EXPECTED_MODEL_IDENTITY:
        raise ProtocolError("private schedule target model binding mismatch")
    evidence_class = document.get("evidence_class")
    if evidence_class not in {"confirmatory", "non_evidence_smoke"}:
        raise ProtocolError("private schedule evidence class is invalid")
    evidence_run = evidence_class == "confirmatory"
    raw_instances = document.get("public_instances")
    if not isinstance(raw_instances, list):
        raise ProtocolError("private schedule lacks its public instance snapshot")
    instances = tuple(PublicInstance.from_mapping(row) for row in raw_instances)
    if len({instance.instance_id for instance in instances}) != len(instances):
        raise ProtocolError("private schedule contains duplicate public instances")
    public_rows = [instance.as_dict() for instance in sorted(instances, key=lambda item: item.instance_id)]
    public_digest = sha256_bytes(canonical_jsonl_bytes(public_rows))
    if document.get("dataset_public_snapshot_sha256") != public_digest:
        raise ProtocolError("private schedule public snapshot digest mismatch")
    if evidence_run and public_digest != PUBLIC_SNAPSHOT_SHA256:
        raise ProtocolError("confirmatory schedule is not the official public snapshot")
    try:
        entries = tuple(ScheduleEntry(**row) for row in document.get("schedule", []))
    except (TypeError, ValueError) as exc:
        raise ProtocolError("private schedule contains malformed entries") from exc
    expected = make_balanced_schedule(
        [instance.instance_id for instance in instances],
        require_full=evidence_run,
    )
    if entries != expected:
        raise ProtocolError("private schedule differs from the frozen balanced schedule")
    if document.get("source_free_summary") != source_free_schedule_summary(entries):
        raise ProtocolError("private schedule digest or summary mismatch")
    return document, entries


def _trusted_git() -> str:
    for candidate in ("/usr/bin/git", "/opt/homebrew/bin/git"):
        if Path(candidate).is_file():
            return candidate
    resolved = shutil.which("git", path="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")
    if resolved is None:
        raise ProtocolError("trusted Git executable is unavailable")
    return resolved


def _run_git(
    repo: Path,
    argv: Sequence[str],
    *,
    timeout_s: float = 60.0,
    allowed_returncodes: frozenset[int] = frozenset({0}),
    git_directory: Path | None = None,
    work_tree: Path | None = None,
) -> bytes:
    if (git_directory is None) != (work_tree is None):
        raise ProtocolError("external Git commands require both git_directory and work_tree")
    environment = {
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "TMPDIR": "/tmp",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    location = ["-C", str(repo)]
    if git_directory is not None and work_tree is not None:
        private_git = require_private_directory(git_directory)
        resolved_tree = work_tree.resolve(strict=True)
        if (
            not resolved_tree.is_dir()
            or _is_within(private_git, resolved_tree)
            or _is_within(resolved_tree, private_git)
        ):
            raise ProtocolError("external Git metadata and worktree must be separate directories")
        location = [
            "-C",
            str(resolved_tree),
            f"--git-dir={private_git}",
            f"--work-tree={resolved_tree}",
        ]
    result = subprocess.run(
        [
            _trusted_git(),
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "diff.external=",
            "-c",
            "submodule.recurse=false",
            *location,
            *argv,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_s,
        env=environment,
    )
    if result.returncode not in allowed_returncodes:
        detail = result.stderr.decode("utf-8", errors="replace")[:500]
        raise ProtocolError(f"git command failed ({argv[0]}): {detail}")
    return result.stdout


def capture_git_patch(
    workspace: Path,
    *,
    expected_base_commit: str,
    max_patch_bytes: int = 32 * 1024 * 1024,
    external_git_directory: Path | None = None,
) -> str:
    """Capture a patch from an isolated checkout, including untracked files.

    The adapter reads repository state only.  It never scans assistant text for
    a diff or accepts Markdown/code-fence output as an official prediction.
    """

    root = workspace.resolve(strict=True)
    embedded_git = root / ".git"
    git_options: dict[str, Path] = {}
    if external_git_directory is None:
        if embedded_git.is_symlink() or not embedded_git.is_dir():
            raise ProtocolError("generation workspace is not an isolated Git checkout")
    else:
        traversal_errors: list[OSError] = []
        entry_count = 0
        for directory, dirnames, filenames in os.walk(
            root,
            followlinks=False,
            onerror=traversal_errors.append,
        ):
            entry_count += len(dirnames) + len(filenames)
            if entry_count > 100_000:
                raise ProtocolError("generation workspace traversal exceeded its entry bound")
            names = {*dirnames, *filenames}
            if any(name.casefold() == ".git" for name in names):
                raise ProtocolError("model-visible generation workspace contains forbidden Git metadata")
            if len(Path(directory).relative_to(root).parts) > 256:
                raise ProtocolError("generation workspace traversal exceeded its depth bound")
        if traversal_errors:
            raise ProtocolError("generation workspace Git-metadata scan was incomplete")
        private_git = require_private_directory(external_git_directory)
        if _is_within(private_git, root) or _is_within(root, private_git):
            raise ProtocolError("external Git metadata must be outside the model-visible workspace")
        git_options = {"git_directory": private_git, "work_tree": root}
    top_level = _run_git(root, ["rev-parse", "--show-toplevel"], **git_options).decode().strip()
    if Path(top_level).resolve(strict=True) != root:
        raise ProtocolError("generation workspace Git root differs from the isolated checkout")
    head = _run_git(root, ["rev-parse", "HEAD"], **git_options).decode().strip()
    if head != expected_base_commit:
        raise ProtocolError("generation workspace HEAD differs from dataset base_commit")

    untracked_raw = _run_git(
        root,
        ["ls-files", "--others", "--exclude-standard", "-z", "--"],
        **git_options,
    )
    patch_parts = [
        _run_git(
            root,
            [
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
                "--",
            ],
            **git_options,
        )
    ]
    for raw_name in (value for value in untracked_raw.split(b"\0") if value):
        name = os.fsdecode(raw_name)
        patch_parts.append(
            _run_git(
                root,
                [
                    "diff",
                    "--no-index",
                    "--binary",
                    "--full-index",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--",
                    "/dev/null",
                    name,
                ],
                allowed_returncodes=frozenset({0, 1}),
                **git_options,
            )
        )
    patch_bytes = b"".join(patch_parts)
    if len(patch_bytes) > max_patch_bytes:
        raise ProtocolError("captured model patch exceeds the preregistered size limit")
    patch = patch_bytes.decode("utf-8", errors="strict")
    validate_patch_only(patch, max_patch_bytes=max_patch_bytes)
    return patch


def _valid_git_diff_section(section: str) -> bool:
    unified = "\n--- " in section and "\n+++ " in section
    binary = "\nGIT binary patch\n" in section or "\nBinary files " in section
    rename = "\nrename from " in section and "\nrename to " in section
    mode_change = "\nold mode " in section and "\nnew mode " in section
    empty_file_change = "\nnew file mode " in section or "\ndeleted file mode " in section
    return unified or binary or rename or mode_change or empty_file_change


def validate_patch_only(patch: str, *, max_patch_bytes: int = 32 * 1024 * 1024) -> None:
    if not isinstance(patch, str):
        raise ProtocolError("model_patch must be a string")
    encoded = patch.encode("utf-8")
    if len(encoded) > max_patch_bytes:
        raise ProtocolError("model_patch exceeds size limit")
    if "\x00" in patch:
        raise ProtocolError("model_patch contains non-patch framing")
    if not patch:
        return
    if not patch.startswith("diff --git "):
        raise ProtocolError("non-empty model_patch must be a raw git diff")
    sections = [section for section in re.split(r"(?m)(?=^diff --git )", patch) if section]
    if not sections or any(not section.startswith("diff --git ") for section in sections):
        raise ProtocolError("model_patch contains data outside git diff sections")
    if any(not _valid_git_diff_section(section) for section in sections):
        raise ProtocolError("model_patch contains an unsupported or malformed git diff section")


def official_prediction(instance_id: str, condition: str, patch: str) -> dict[str, str]:
    if not _INSTANCE_ID_RE.fullmatch(instance_id):
        raise ProtocolError("invalid prediction instance_id")
    if condition not in CONDITIONS:
        raise ProtocolError("unknown benchmark condition")
    validate_patch_only(patch)
    return {
        "instance_id": instance_id,
        "model_name_or_path": MODEL_LABELS[condition],
        "model_patch": patch,
    }


@dataclass(frozen=True)
class ArmCheckpoint:
    preregistration_sha256: str
    schedule_sha256: str
    execution_index: int
    pair_index: int
    instance_id: str
    instance_digest: str
    condition: str
    status: str
    model_patch: str
    mio_commit: str
    model_identity: str
    runtime_digest: str
    quality_gate_decision: str
    output_tokens: int = 0
    tool_calls: int = 0
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS or self.status not in _TERMINAL_STATUSES:
            raise ProtocolError("invalid terminal arm checkpoint")
        if self.instance_digest != _instance_digest(self.instance_id):
            raise ProtocolError("checkpoint instance digest mismatch")
        if not _COMMIT_RE.fullmatch(self.mio_commit):
            raise ProtocolError("checkpoint must bind a clean Mio commit")
        if not _MODEL_IDENTITY_RE.fullmatch(self.model_identity):
            raise ProtocolError("checkpoint must bind a full local model identity")
        if self.model_identity != EXPECTED_MODEL_IDENTITY:
            raise ProtocolError("checkpoint target model differs from frozen Qwen 3.6 27B")
        if not _SHA256_RE.fullmatch(self.runtime_digest):
            raise ProtocolError("checkpoint runtime_digest must be SHA-256")
        if self.condition == "gate_on" and self.quality_gate_decision not in {"satisfied", "incomplete"}:
            raise ProtocolError("gate_on checkpoint lacks an authoritative gate decision")
        if self.condition == "gate_on" and self.status == "completed" and self.quality_gate_decision == "incomplete":
            raise ProtocolError("completed gate_on checkpoint cannot have an incomplete gate decision")
        if self.condition == "gate_off" and self.quality_gate_decision != "not_applicable":
            raise ProtocolError("gate_off checkpoint must use not_applicable gate decision")
        if (
            self.output_tokens < 0
            or self.output_tokens > MAX_OUTPUT_TOKENS_PER_ARM
            or self.tool_calls < 0
            or self.tool_calls > MAX_TOOL_CALLS_PER_ARM
            or self.wall_seconds < 0
            or self.wall_seconds > MAX_AGENT_WALL_SECONDS
            or not math.isfinite(self.wall_seconds)
        ):
            raise ProtocolError("invalid checkpoint metrics")
        validate_patch_only(self.model_patch)

    @classmethod
    def for_entry(cls, entry: ScheduleEntry, **kwargs: Any) -> "ArmCheckpoint":
        return cls(
            preregistration_sha256=preregistration_digest(),
            schedule_sha256=str(kwargs.pop("schedule_sha256")),
            execution_index=entry.execution_index,
            pair_index=entry.pair_index,
            instance_id=entry.instance_id,
            instance_digest=entry.instance_digest,
            condition=entry.condition,
            **kwargs,
        )

    def private_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA}.arm-checkpoint",
            **self.__dict__,
        }


def _read_immutable_file(path: Path, *, allow_missing: bool = False) -> bytes | None:
    """Read one stable, single-link regular file without following aliases."""

    _reject_symlink_path_components(path.parent)
    try:
        parent = path.parent.resolve(strict=True)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise ProtocolError(f"immutable artifact parent is missing: {path.name}") from None
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(parent, directory_flags)
    fd = -1
    try:
        try:
            fd = os.open(
                path.name,
                os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            if allow_missing:
                return None
            raise ProtocolError(f"immutable artifact is missing: {path.name}") from None
        except OSError as exc:
            raise ProtocolError(f"cannot open immutable artifact without following aliases: {path.name}") from exc

        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ProtocolError("immutable artifact must be a single-link regular file")
        if before.st_mode & 0o077:
            raise ProtocolError("immutable artifact must use private permissions")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
        try:
            named = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise ProtocolError("immutable artifact path changed during read") from exc
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields) or (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_nlink,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        ) != tuple(getattr(after, field) for field in stable_fields):
            raise ProtocolError("immutable artifact changed during read")
        return b"".join(chunks)
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(directory_fd)


def _immutable_file_sha256(path: Path) -> str:
    payload = _read_immutable_file(path)
    assert payload is not None
    return sha256_bytes(payload)


class CheckpointStore:
    """Immutable per-arm checkpoints with crash-safe, exclusive publication."""

    def __init__(self, root: Path) -> None:
        self.root = root
        _reject_symlink_path_components(self.root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def path_for(self, entry: ScheduleEntry) -> Path:
        return self.root / f"{entry.execution_index:04d}-{entry.condition}.json"

    def save(self, checkpoint: ArmCheckpoint) -> Path:
        entry = ScheduleEntry(
            pair_index=checkpoint.pair_index,
            execution_index=checkpoint.execution_index,
            instance_id=checkpoint.instance_id,
            instance_digest=checkpoint.instance_digest,
            condition=checkpoint.condition,
            position_in_pair=checkpoint.execution_index % 2,
        )
        destination = self.path_for(entry)
        payload = canonical_json_bytes(checkpoint.private_dict())
        existing = _read_immutable_file(destination, allow_missing=True)
        if existing is not None:
            if existing != payload:
                raise ProtocolError("immutable checkpoint already exists with different bytes")
            return destination
        try:
            _atomic_write(destination, payload)
        except ProtocolError as exc:
            # A second writer may win after our initial absence check.  Re-read
            # through the same no-alias path and accept only byte identity.
            observed = _read_immutable_file(destination, allow_missing=True)
            if observed is None or observed != payload:
                raise ProtocolError("concurrent checkpoint publication conflict") from exc
        if _read_immutable_file(destination) != payload:
            raise ProtocolError("checkpoint publication changed bytes")
        return destination

    def load(self, entry: ScheduleEntry) -> ArmCheckpoint:
        payload = _read_immutable_file(self.path_for(entry))
        assert payload is not None
        try:
            raw = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("checkpoint is not valid UTF-8 JSON") from exc
        if raw.pop("schema", None) != f"{SCHEMA}.arm-checkpoint":
            raise ProtocolError("unexpected arm checkpoint schema")
        checkpoint = ArmCheckpoint(**raw)
        if (
            checkpoint.execution_index != entry.execution_index
            or checkpoint.pair_index != entry.pair_index
            or checkpoint.instance_id != entry.instance_id
            or checkpoint.condition != entry.condition
        ):
            raise ProtocolError("checkpoint does not match its scheduled arm")
        return checkpoint


def resume_entries(
    schedule: Sequence[ScheduleEntry],
    store: CheckpointStore,
) -> tuple[ScheduleEntry, ...]:
    """Return only missing arms after validating every immutable terminal checkpoint."""

    digest = schedule_digest(schedule)
    pending: list[ScheduleEntry] = []
    for entry in schedule:
        path = store.path_for(entry)
        if not path.exists():
            pending.append(entry)
            continue
        checkpoint = store.load(entry)
        if checkpoint.preregistration_sha256 != preregistration_digest():
            raise ProtocolError("resume checkpoint preregistration binding mismatch")
        if checkpoint.schedule_sha256 != digest:
            raise ProtocolError("resume checkpoint schedule binding mismatch")
    return tuple(pending)


def pair_attempt_store(root: Path, pair_index: int, attempt_index: int) -> CheckpointStore:
    if pair_index < 0 or attempt_index < 0:
        raise ProtocolError("pair and attempt indices must be non-negative")
    return CheckpointStore(root / f"pair-{pair_index:04d}" / f"attempt-{attempt_index:03d}")


class AttemptLedger:
    """Hash-chained append-only events retaining every whole-pair attempt."""

    _SCHEMA = f"{SCHEMA}.pair-attempt-event"
    _EVENTS = frozenset({"started", "completed", "aborted"})
    _REASONS = frozenset({"initial", "completed"}) | _INFRASTRUCTURE_REASONS

    def __init__(self, path: Path, schedule_sha256: str) -> None:
        if not _SHA256_RE.fullmatch(schedule_sha256):
            raise ProtocolError("attempt ledger requires a schedule SHA-256")
        self.path = path
        self.schedule_sha256 = schedule_sha256

    @classmethod
    def _record_digest(cls, record: Mapping[str, Any]) -> str:
        payload = {key: value for key, value in record.items() if key != "record_sha256"}
        return sha256_bytes(canonical_json_bytes(payload))

    def _parse(self, payload: bytes) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        previous = ""
        states: dict[tuple[int, int], str] = {}
        latest_attempt: dict[int, int] = {}
        for sequence, line in enumerate(payload.splitlines()):
            if not line:
                raise ProtocolError("attempt ledger contains an empty record")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError("attempt ledger contains invalid JSON") from exc
            if not isinstance(record, dict) or canonical_json_bytes(record).rstrip(b"\n") != line:
                raise ProtocolError("attempt ledger record is not canonical JSON")
            if record.get("schema") != self._SCHEMA or record.get("sequence") != sequence:
                raise ProtocolError("attempt ledger schema or sequence mismatch")
            if record.get("schedule_sha256") != self.schedule_sha256:
                raise ProtocolError("attempt ledger schedule binding mismatch")
            if record.get("previous_sha256") != previous:
                raise ProtocolError("attempt ledger hash chain mismatch")
            digest = record.get("record_sha256")
            if not isinstance(digest, str) or digest != self._record_digest(record):
                raise ProtocolError("attempt ledger record digest mismatch")
            pair_index = record.get("pair_index")
            attempt_index = record.get("attempt_index")
            event = record.get("event")
            reason_code = record.get("reason_code")
            checkpoints = record.get("checkpoint_sha256s")
            if (
                not isinstance(pair_index, int)
                or pair_index < 0
                or not isinstance(attempt_index, int)
                or attempt_index < 0
                or event not in self._EVENTS
                or not isinstance(reason_code, str)
                or reason_code not in self._REASONS
                or not isinstance(checkpoints, dict)
            ):
                raise ProtocolError("attempt ledger event fields are invalid")
            key = (pair_index, attempt_index)
            if event == "started":
                if key in states or attempt_index != latest_attempt.get(pair_index, -1) + 1:
                    raise ProtocolError("attempt ledger start order is invalid")
                if attempt_index == 0 and reason_code != "initial":
                    raise ProtocolError("first pair attempt must use the initial reason")
                if attempt_index > 0 and reason_code not in _INFRASTRUCTURE_REASONS:
                    raise ProtocolError("only blinded infrastructure reasons permit pair retry")
                if attempt_index > 0 and states.get((pair_index, attempt_index - 1)) != "aborted":
                    raise ProtocolError("pair retry is forbidden after a completed attempt")
                if checkpoints:
                    raise ProtocolError("started attempt cannot already contain checkpoint hashes")
                states[key] = "started"
                latest_attempt[pair_index] = attempt_index
            else:
                if states.get(key) != "started":
                    raise ProtocolError("attempt terminal event lacks one unmatched start")
                if event == "completed":
                    if reason_code != "completed":
                        raise ProtocolError("completed attempt must use the completed reason")
                    if set(checkpoints) != set(CONDITIONS) or any(
                        not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in checkpoints.values()
                    ):
                        raise ProtocolError("completed pair attempt lacks both checkpoint hashes")
                else:
                    if reason_code not in _INFRASTRUCTURE_REASONS:
                        raise ProtocolError("aborted attempt requires an infrastructure reason")
                    if checkpoints:
                        raise ProtocolError("aborted attempt cannot claim checkpoint hashes")
                states[key] = event
            records.append(record)
            previous = digest
        return tuple(records)

    def read(self) -> tuple[dict[str, Any], ...]:
        payload = _read_immutable_file(self.path, allow_missing=True)
        if payload is None:
            return ()
        return self._parse(payload)

    def append(
        self,
        *,
        pair_index: int,
        attempt_index: int,
        event: str,
        reason_code: str,
        checkpoint_sha256s: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        _reject_symlink_path_components(self.path.parent)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.resolve(strict=True)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(parent, directory_flags)
        fd = -1
        try:
            try:
                fd = os.open(
                    self.path.name,
                    os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise ProtocolError("attempt ledger cannot follow an alias") from exc
            os.fchmod(fd, 0o600)
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise ProtocolError("attempt ledger must be a single-link regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                chunks: list[bytes] = []
                while True:
                    block = os.read(fd, 1024 * 1024)
                    if not block:
                        break
                    chunks.append(block)
                payload = b"".join(chunks)
                records = self._parse(payload)
                previous = records[-1]["record_sha256"] if records else ""
                record: dict[str, Any] = {
                    "schema": self._SCHEMA,
                    "sequence": len(records),
                    "previous_sha256": previous,
                    "schedule_sha256": self.schedule_sha256,
                    "pair_index": pair_index,
                    "attempt_index": attempt_index,
                    "event": event,
                    "reason_code": reason_code,
                    "checkpoint_sha256s": dict(checkpoint_sha256s or {}),
                }
                record["record_sha256"] = self._record_digest(record)
                candidate = canonical_json_bytes(record)
                self._parse(payload + candidate)
                offset = 0
                while offset < len(candidate):
                    written = os.write(fd, candidate[offset:])
                    if written <= 0:
                        raise ProtocolError("attempt ledger append made no progress")
                    offset += written
                os.fsync(fd)
                after = os.fstat(fd)
                try:
                    named = os.stat(self.path.name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    raise ProtocolError("attempt ledger path changed during append") from exc
                if (
                    not stat.S_ISREG(after.st_mode)
                    or after.st_nlink != 1
                    or (named.st_dev, named.st_ino, named.st_nlink) != (after.st_dev, after.st_ino, after.st_nlink)
                ):
                    raise ProtocolError("attempt ledger changed identity during append")
                os.fsync(directory_fd)
                return record
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            if fd >= 0:
                os.close(fd)
            os.close(directory_fd)


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    _reject_symlink_path_components(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    existing = _read_immutable_file(parent / path.name, allow_missing=True)
    if existing is not None:
        if existing == payload:
            return
        raise ProtocolError(f"refusing to overwrite immutable artifact: {path.name}")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(parent, directory_flags)
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary_fd = -1
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=directory_fd,
        )
        os.fchmod(temporary_fd, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(temporary_fd, payload[offset:])
            if written <= 0:
                raise ProtocolError("immutable artifact write made no progress")
            offset += written
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            observed = _read_immutable_file(parent / path.name)
            if observed != payload:
                raise ProtocolError(f"concurrent immutable artifact conflict: {path.name}")
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            pass
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)

    if _read_immutable_file(parent / path.name) != payload:
        raise ProtocolError("immutable artifact publication changed bytes")


def export_official_predictions(
    schedule: Sequence[ScheduleEntry],
    store: CheckpointStore,
    output_directory: Path,
) -> dict[str, Path]:
    if output_directory.exists():
        output_directory = require_private_directory(output_directory)
    else:
        output_directory = create_private_directory(output_directory)
    predictions, _binding = _prediction_payloads_and_binding(schedule, store)
    pair_count = len(schedule) // 2
    if pair_count == EXPECTED_INSTANCES and any(len(rows) != EXPECTED_INSTANCES for rows in predictions.values()):
        raise ProtocolError("full export requires 500 predictions in each arm")
    paths: dict[str, Path] = {}
    for condition, rows in predictions.items():
        path = output_directory / f"{condition}.jsonl"
        payload = b"".join(canonical_json_bytes(row) for row in rows)
        _atomic_write(path, payload)
        paths[condition] = path
    return paths


def _prediction_payloads_and_binding(
    schedule: Sequence[ScheduleEntry],
    store: CheckpointStore,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    expected_schedule_digest = schedule_digest(schedule)
    predictions: dict[str, list[dict[str, str]]] = {condition: [] for condition in CONDITIONS}
    bindings: set[tuple[str, str, str]] = set()
    for entry in schedule:
        checkpoint = store.load(entry)
        if checkpoint.preregistration_sha256 != preregistration_digest():
            raise ProtocolError("checkpoint preregistration binding mismatch")
        if checkpoint.schedule_sha256 != expected_schedule_digest:
            raise ProtocolError("checkpoint schedule binding mismatch")
        bindings.add((checkpoint.mio_commit, checkpoint.model_identity, checkpoint.runtime_digest))
        predictions[entry.condition].append(
            official_prediction(entry.instance_id, entry.condition, checkpoint.model_patch)
        )
    if len(bindings) != 1:
        raise ProtocolError("paired arms do not share one Mio/model/runtime identity")
    mio_commit, model_identity, runtime_digest = bindings.pop()
    return predictions, {
        "mio_commit": mio_commit,
        "model_identity": model_identity,
        "runtime_digest": runtime_digest,
    }


def official_harness_commands(
    predictions_directory: Path,
    full_snapshot: Path,
    *,
    schedule_sha256: str,
    max_workers: int = 6,
    timeout_seconds: int = 1800,
    harness_distribution_sha256: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    if max_workers < 1 or timeout_seconds < 1:
        raise ProtocolError("harness worker and timeout values must be positive")
    if not _SHA256_RE.fullmatch(schedule_sha256):
        raise ProtocolError("harness command schedule SHA-256 is malformed")
    dataset_path = verify_full_snapshot(full_snapshot)
    predictions_root = require_private_directory(predictions_directory)
    distribution_identity = harness_distribution_sha256 or harness_distribution_identity()
    if not _SHA256_RE.fullmatch(distribution_identity):
        raise ProtocolError("evaluation harness distribution identity is malformed")
    commands = []
    for condition in CONDITIONS:
        prediction_path = predictions_root / f"{condition}.jsonl"
        if not prediction_path.is_file():
            raise ProtocolError(f"missing {condition} official prediction artifact")
        run_id = evaluation_run_id(
            condition,
            sha256_file(prediction_path),
            schedule_sha256,
            timeout_seconds,
            distribution_identity,
        )
        commands.append(
            (
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                str(dataset_path),
                "--split",
                "test",
                "--predictions_path",
                str(prediction_path),
                "--max_workers",
                str(max_workers),
                "--run_id",
                run_id,
                "--cache_level",
                "env",
                "--clean",
                "true",
                "--timeout",
                str(timeout_seconds),
            )
        )
    return tuple(commands)


def evaluation_run_id(
    condition: str,
    prediction_sha256: str,
    schedule_sha256: str,
    timeout_seconds: int,
    harness_distribution_sha256: str,
) -> str:
    if condition not in CONDITIONS:
        raise ProtocolError("unknown evaluation run condition")
    if (
        not _SHA256_RE.fullmatch(prediction_sha256)
        or not _SHA256_RE.fullmatch(schedule_sha256)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
        or not _SHA256_RE.fullmatch(harness_distribution_sha256)
    ):
        raise ProtocolError("evaluation run identity input is malformed")
    material = {
        "condition": condition,
        "prediction_sha256": prediction_sha256,
        "preregistration_sha256": preregistration_digest(),
        "schedule_sha256": schedule_sha256,
        "timeout_seconds": timeout_seconds,
        "full_snapshot_sha256": FULL_SNAPSHOT_SHA256,
        "harness_commit": HARNESS_COMMIT,
        "harness_distribution_sha256": harness_distribution_sha256,
    }
    suffix = sha256_bytes(canonical_json_bytes(material))[:16]
    return f"mio-qwen36-27b-{condition}-v1-{suffix}"


def _expected_instance_ids(schedule: Sequence[ScheduleEntry]) -> tuple[str, ...]:
    if not schedule or len(schedule) % 2:
        raise ProtocolError("evaluation schedule contains an incomplete pair")
    identifiers: list[str] = []
    for index in range(0, len(schedule), 2):
        first, second = schedule[index : index + 2]
        if (
            first.pair_index != second.pair_index
            or first.instance_id != second.instance_id
            or {first.condition, second.condition} != set(CONDITIONS)
            or {first.position_in_pair, second.position_in_pair} != {0, 1}
            or first.execution_index != index
            or second.execution_index != index + 1
        ):
            raise ProtocolError("evaluation schedule pair structure is invalid")
        identifiers.append(first.instance_id)
    if len(set(identifiers)) != len(identifiers):
        raise ProtocolError("evaluation schedule contains duplicate instance pairs")
    return tuple(sorted(identifiers))


def harness_distribution_identity() -> str:
    try:
        distribution = importlib.metadata.distribution("swebench")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ProtocolError("pinned swebench distribution is not installed") from exc
    if distribution.version != HARNESS_VERSION:
        raise ProtocolError("installed swebench version differs from the pinned harness")
    digest = hashlib.sha256()
    digest.update(b"mio-swebench-distribution-v1\0")
    files = sorted(
        (item for item in (distribution.files or ()) if "__pycache__" not in item.parts and item.suffix != ".pyc"),
        key=str,
    )
    if not files:
        raise ProtocolError("installed swebench distribution has no identity files")
    for relative in files:
        candidate = Path(distribution.locate_file(relative))
        if not candidate.is_file():
            continue
        name = str(relative).encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def build_evaluation_seal(
    schedule: Sequence[ScheduleEntry],
    store: CheckpointStore,
    predictions_directory: Path,
    full_snapshot: Path,
    *,
    max_workers: int,
    timeout_seconds: int,
    harness_distribution_sha256: str,
) -> dict[str, Any]:
    identifiers = _expected_instance_ids(schedule)
    if len(identifiers) == EXPECTED_INSTANCES and timeout_seconds != CONFIRMATORY_TIMEOUT_SECONDS:
        raise ProtocolError("confirmatory evaluation timeout is frozen at 1800 seconds")
    frozen_schedule_sha256 = schedule_digest(schedule)
    predictions, binding = _prediction_payloads_and_binding(schedule, store)
    if binding["model_identity"] != EXPECTED_MODEL_IDENTITY:
        raise ProtocolError("evaluation checkpoint model identity mismatch")
    if not _SHA256_RE.fullmatch(harness_distribution_sha256):
        raise ProtocolError("evaluation harness distribution identity is malformed")
    dataset_path = verify_full_snapshot(full_snapshot)
    predictions_root = require_private_directory(predictions_directory)
    prediction_rows: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        path = predictions_root / f"{condition}.jsonl"
        expected_payload = b"".join(canonical_json_bytes(row) for row in predictions[condition])
        if path.read_bytes() != expected_payload:
            raise ProtocolError(f"{condition} prediction artifact differs from immutable checkpoints")
        prediction_sha256 = sha256_bytes(expected_payload)
        prediction_rows[condition] = {
            "path": str(path),
            "sha256": prediction_sha256,
            "rows": len(predictions[condition]),
            "model_label": MODEL_LABELS[condition],
            "run_id": evaluation_run_id(
                condition,
                prediction_sha256,
                frozen_schedule_sha256,
                timeout_seconds,
                harness_distribution_sha256,
            ),
        }
    commands = official_harness_commands(
        predictions_root,
        dataset_path,
        schedule_sha256=frozen_schedule_sha256,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        harness_distribution_sha256=harness_distribution_sha256,
    )
    return {
        "schema": f"{SCHEMA}.evaluation-seal",
        "preregistration_sha256": preregistration_digest(),
        "schedule_sha256": schedule_digest(schedule),
        "scheduled_instance_ids_sha256": sha256_bytes(canonical_json_bytes(list(identifiers))),
        "scheduled_instances": len(identifiers),
        "dataset": {
            "name": DATASET_NAME,
            "revision": DATASET_REVISION,
            "path": str(dataset_path),
            "parquet_sha256": DATASET_PARQUET_SHA256,
            "full_snapshot_sha256": FULL_SNAPSHOT_SHA256,
            "public_snapshot_sha256": PUBLIC_SNAPSHOT_SHA256,
        },
        "generation_binding": binding,
        "predictions": prediction_rows,
        "harness": {
            "version": HARNESS_VERSION,
            "commit": HARNESS_COMMIT,
            "distribution_sha256": harness_distribution_sha256,
            "max_workers": max_workers,
            "timeout_seconds": timeout_seconds,
        },
        "commands": [list(command) for command in commands],
    }


def evaluation_seal_digest(seal: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(seal)))


def official_report_path(evaluation_directory: Path, condition: str, run_id: str) -> Path:
    if condition not in CONDITIONS:
        raise ProtocolError("unknown report condition")
    if not re.fullmatch(r"mio-qwen36-27b-(?:gate_off|gate_on)-v1-[0-9a-f]{16}", run_id):
        raise ProtocolError("official report run ID is malformed")
    return evaluation_directory / f"{MODEL_LABELS[condition]}.{run_id}.json"


def _exact_one_sided_mcnemar(gate_on_only: int, gate_off_only: int) -> float:
    discordant = gate_on_only + gate_off_only
    if discordant == 0:
        return 1.0
    numerator = sum(math.comb(discordant, value) for value in range(gate_on_only, discordant + 1))
    return numerator / (2**discordant)


def _paired_bootstrap_interval(
    differences: Sequence[int],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SCHEDULE_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not differences or samples < 1 or not 0 < alpha < 1:
        raise ProtocolError("invalid paired bootstrap configuration")
    rng = random.Random(seed)
    count = len(differences)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(differences[rng.randrange(count)] for _ in range(count)) / count)
    estimates.sort()
    low_index = max(0, math.floor((alpha / 2) * samples))
    high_index = min(samples - 1, math.ceil((1 - alpha / 2) * samples) - 1)
    return estimates[low_index], estimates[high_index]


def _load_official_report(
    path: Path,
    *,
    expected_ids: Sequence[str],
) -> tuple[dict[str, Any], set[str]]:
    path = require_private_path(path, must_exist=True)
    if path.stat().st_mode & 0o077:
        raise ProtocolError("private official report must use 0600 permissions")
    expected = set(expected_ids)
    expected_count = len(expected)
    if expected_count != len(tuple(expected_ids)):
        raise ProtocolError("expected official report IDs are duplicated")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 2:
        raise ProtocolError("official SWE-bench report must use schema_version 2")
    if report.get("total_instances") != expected_count or report.get("submitted_instances") != expected_count:
        raise ProtocolError("official report is not a complete submitted dataset")
    submitted_list = report.get("submitted_ids", [])
    submitted = set(submitted_list)
    if len(submitted) != len(submitted_list) or submitted != expected:
        raise ProtocolError("official report submitted_ids differ from the frozen schedule")
    if report.get("error_instances", 0) or report.get("error_ids") or report.get("incomplete_ids"):
        raise ProtocolError("official harness errors/incomplete instances block confirmatory aggregation")
    resolved = set(report.get("resolved_ids", []))
    if not resolved <= submitted or len(resolved) != report.get("resolved_instances"):
        raise ProtocolError("official report resolved_ids mismatch")
    unresolved = set(report.get("unresolved_ids", []))
    completed = set(report.get("completed_ids", []))
    empty = set(report.get("empty_patch_ids", []))
    if (
        len(unresolved) != report.get("unresolved_instances")
        or len(completed) != report.get("completed_instances")
        or len(empty) != report.get("empty_patch_instances")
        or not unresolved <= submitted
        or not completed <= submitted
        or not empty <= submitted
        or resolved & unresolved
        or completed != resolved | unresolved
        or submitted != completed | empty
        or completed & empty
    ):
        raise ProtocolError("official report outcome partitions are inconsistent")
    return report, resolved


def build_evaluation_receipt(
    seal: Mapping[str, Any],
    schedule: Sequence[ScheduleEntry],
    report_paths: Mapping[str, Path],
    *,
    observed_model_identity_before: str,
    observed_model_identity_after: str,
) -> dict[str, Any]:
    if (
        observed_model_identity_before != EXPECTED_MODEL_IDENTITY
        or observed_model_identity_after != EXPECTED_MODEL_IDENTITY
    ):
        raise ProtocolError("evaluation receipt model identity checks differ from frozen target")
    expected_ids = _expected_instance_ids(schedule)
    reports: dict[str, dict[str, str]] = {}
    for condition in CONDITIONS:
        path = report_paths[condition].expanduser().resolve(strict=True)
        _load_official_report(path, expected_ids=expected_ids)
        reports[condition] = {"path": str(path), "sha256": sha256_file(path)}
    return {
        "schema": f"{SCHEMA}.evaluation-receipt",
        "seal": dict(seal),
        "seal_sha256": evaluation_seal_digest(seal),
        "reports": reports,
        "model_identity_checks": {
            "post_generation_pre_evaluation": observed_model_identity_before,
            "post_evaluation": observed_model_identity_after,
        },
    }


def verify_evaluation_receipt(
    receipt_path: Path,
    schedule: Sequence[ScheduleEntry],
    store: CheckpointStore,
    predictions_directory: Path,
    full_snapshot: Path,
    report_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], str]:
    receipt_path = require_private_path(receipt_path, must_exist=True)
    if receipt_path.stat().st_mode & 0o077:
        raise ProtocolError("private evaluation receipt must use 0600 permissions")
    payload = receipt_path.read_bytes()
    receipt = json.loads(payload)
    if canonical_json_bytes(receipt) != payload:
        raise ProtocolError("evaluation receipt is not canonical JSON")
    if receipt.get("schema") != f"{SCHEMA}.evaluation-receipt":
        raise ProtocolError("unexpected evaluation receipt schema")
    seal = receipt.get("seal")
    if not isinstance(seal, dict) or receipt.get("seal_sha256") != evaluation_seal_digest(seal):
        raise ProtocolError("evaluation receipt seal digest mismatch")
    if receipt.get("model_identity_checks") != {
        "post_generation_pre_evaluation": EXPECTED_MODEL_IDENTITY,
        "post_evaluation": EXPECTED_MODEL_IDENTITY,
    }:
        raise ProtocolError("evaluation receipt model identity checks mismatch")
    harness = seal.get("harness")
    if not isinstance(harness, dict):
        raise ProtocolError("evaluation receipt lacks its harness binding")
    observed_seal = build_evaluation_seal(
        schedule,
        store,
        predictions_directory,
        full_snapshot,
        max_workers=int(harness.get("max_workers", 0)),
        timeout_seconds=int(harness.get("timeout_seconds", 0)),
        harness_distribution_sha256=str(harness.get("distribution_sha256", "")),
    )
    if observed_seal != seal:
        raise ProtocolError("current evaluation inputs differ from the immutable seal")
    expected_ids = _expected_instance_ids(schedule)
    receipt_reports = receipt.get("reports")
    if not isinstance(receipt_reports, dict) or set(receipt_reports) != set(CONDITIONS):
        raise ProtocolError("evaluation receipt report bindings are incomplete")
    for condition in CONDITIONS:
        path = report_paths[condition].expanduser().resolve(strict=True)
        report_binding = receipt_reports[condition]
        if not isinstance(report_binding, dict):
            raise ProtocolError("evaluation receipt report binding is malformed")
        if report_binding != {"path": str(path), "sha256": sha256_file(path)}:
            raise ProtocolError("current official report differs from the immutable receipt")
        _load_official_report(path, expected_ids=expected_ids)
    return receipt, sha256_bytes(payload)


def aggregate_official_reports(
    gate_off_report: Path,
    gate_on_report: Path,
    *,
    expected_ids: Sequence[str],
    evidence_run: bool = True,
    schedule_sha256: str,
    evaluation_receipt_sha256: str,
    generation_binding: Mapping[str, str],
) -> dict[str, Any]:
    require_confirmatory_generation_attestation(evidence_run)
    expected_identifiers = tuple(expected_ids)
    expected_count = len(expected_identifiers)
    if not _SHA256_RE.fullmatch(schedule_sha256) or not _SHA256_RE.fullmatch(evaluation_receipt_sha256):
        raise ProtocolError("aggregate seal digests are malformed")
    if set(generation_binding) != {"mio_commit", "model_identity", "runtime_digest"}:
        raise ProtocolError("aggregate generation binding fields mismatch")
    if (
        not _COMMIT_RE.fullmatch(generation_binding["mio_commit"])
        or generation_binding["model_identity"] != EXPECTED_MODEL_IDENTITY
        or not _SHA256_RE.fullmatch(generation_binding["runtime_digest"])
    ):
        raise ProtocolError("aggregate generation binding is invalid")
    off_report, off_resolved = _load_official_report(
        gate_off_report,
        expected_ids=expected_identifiers,
    )
    on_report, on_resolved = _load_official_report(
        gate_on_report,
        expected_ids=expected_identifiers,
    )
    submitted_off = set(off_report["submitted_ids"])
    submitted_on = set(on_report["submitted_ids"])
    if submitted_off != submitted_on:
        raise ProtocolError("official arm reports do not contain identical paired instances")
    if evidence_run and expected_count != EXPECTED_INSTANCES:
        raise ProtocolError("only the full 500-pair run can produce confirmatory evidence")

    both = len(off_resolved & on_resolved)
    off_only = len(off_resolved - on_resolved)
    on_only = len(on_resolved - off_resolved)
    neither = expected_count - both - off_only - on_only
    differences = [
        int(identifier in on_resolved) - int(identifier in off_resolved) for identifier in sorted(submitted_off)
    ]
    delta = sum(differences) / expected_count
    interval_low, interval_high = _paired_bootstrap_interval(differences)
    p_value = _exact_one_sided_mcnemar(on_only, off_only)
    improvement = evidence_run and interval_low > 0 and p_value < 0.05
    aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "status": "confirmatory_complete" if evidence_run else "non_evidence_smoke",
        "preregistration_sha256": preregistration_digest(),
        "schedule_sha256": schedule_sha256,
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "dataset": {
            "name": DATASET_NAME,
            "revision": DATASET_REVISION,
            "pairs": expected_count,
        },
        "harness": {
            "version": HARNESS_VERSION,
            "commit": HARNESS_COMMIT,
            "docker_isolation_required": True,
        },
        "generation_binding": dict(generation_binding),
        "arms": {
            "gate_off": {
                "submitted": expected_count,
                "resolved": len(off_resolved),
                "resolution_rate": len(off_resolved) / expected_count,
                "empty_patch_count": int(off_report.get("empty_patch_instances", 0)),
            },
            "gate_on": {
                "submitted": expected_count,
                "resolved": len(on_resolved),
                "resolution_rate": len(on_resolved) / expected_count,
                "empty_patch_count": int(on_report.get("empty_patch_instances", 0)),
            },
        },
        "paired": {
            "both_resolved": both,
            "gate_off_only": off_only,
            "gate_on_only": on_only,
            "neither_resolved": neither,
            "resolution_difference": delta,
            "paired_bootstrap_95_percent": [interval_low, interval_high],
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": SCHEDULE_SEED,
            "exact_one_sided_mcnemar_p": p_value,
        },
        "claim_gate": {
            "quality_improvement": improvement,
            "lower_confidence_bound_gt_zero": interval_low > 0,
            "exact_p_lt_0_05": p_value < 0.05,
            "full_500_pairs": evidence_run and expected_count == EXPECTED_INSTANCES,
            "scope": "single_model_single_runtime_swebench_verified",
        },
        "content_policy": "source_free_aggregate_no_per_instance_rows",
    }
    assert_source_free_aggregate(aggregate)
    return aggregate


def assert_source_free_aggregate(value: Any) -> None:
    forbidden_keys = {
        "instance_id",
        "submitted_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
        "model_patch",
        "problem_statement",
        "repo",
        "base_commit",
        "assistant_text",
        "stdout",
        "stderr",
        "test_name",
        "absolute_path",
    }

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            overlap = set(node) & forbidden_keys
            if overlap:
                raise ProtocolError(f"aggregate leaks forbidden fields: {sorted(overlap)}")
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)


def model_tree_identity(path: Path) -> str:
    """Use Mio's shared MLX fingerprint implementation for the frozen target."""

    from experimental.effort.model_identity import ModelIdentityError, fingerprint_local_model

    try:
        return fingerprint_local_model(path).revision
    except ModelIdentityError as exc:
        raise ProtocolError(f"cannot fingerprint local MLX target: {exc}") from exc


def verify_expected_model(path: Path) -> str:
    identity = model_tree_identity(path)
    if identity != EXPECTED_MODEL_IDENTITY:
        raise ProtocolError("local model does not match the frozen Qwen 3.6 27B identity")
    return identity


def git_clean_head(repo: Path) -> str:
    head = _run_git(repo, ["rev-parse", "HEAD"]).decode().strip()
    if not _COMMIT_RE.fullmatch(head):
        raise ProtocolError("Mio HEAD is not a full commit")
    status = _run_git(repo, ["status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        raise ProtocolError("confirmatory run requires a clean Mio worktree")
    return head


def assess_evaluation_host(
    *,
    machine: str,
    docker_cli_present: bool,
    docker_daemon_ready: bool,
    swebench_version: str | None,
    swebench_distribution_sha256: str | None,
    free_storage_gib: float,
) -> dict[str, Any]:
    blockers = []
    if machine.lower() not in {"x86_64", "amd64"}:
        blockers.append("confirmatory_evaluation_requires_x86_64")
    if not docker_cli_present:
        blockers.append("docker_cli_missing")
    elif not docker_daemon_ready:
        blockers.append("docker_daemon_unavailable")
    if swebench_version != HARNESS_VERSION:
        blockers.append("pinned_swebench_4_1_0_missing")
    if swebench_version == HARNESS_VERSION and (
        swebench_distribution_sha256 is None or not _SHA256_RE.fullmatch(swebench_distribution_sha256)
    ):
        blockers.append("swebench_distribution_identity_unavailable")
    if free_storage_gib < 120:
        blockers.append("less_than_120_gib_free_storage")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "machine": machine,
        "docker_cli_present": docker_cli_present,
        "docker_daemon_ready": docker_daemon_ready,
        "swebench_version": swebench_version,
        "swebench_distribution_sha256": swebench_distribution_sha256,
        "free_storage_gib": free_storage_gib,
    }


def probe_evaluation_host() -> dict[str, Any]:
    docker_path = shutil.which("docker")
    docker_ready = False
    if docker_path:
        try:
            result = subprocess.run(
                [docker_path, "info", "--format", "{{.ServerVersion}}"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=20,
            )
            docker_ready = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            docker_ready = False
    try:
        swebench_version = importlib.metadata.version("swebench")
    except importlib.metadata.PackageNotFoundError:
        swebench_version = None
    swebench_distribution_sha256 = None
    if swebench_version == HARNESS_VERSION:
        try:
            swebench_distribution_sha256 = harness_distribution_identity()
        except ProtocolError:
            swebench_distribution_sha256 = None
    free = shutil.disk_usage(ROOT).free / (1024**3)
    return assess_evaluation_host(
        machine=platform.machine(),
        docker_cli_present=docker_path is not None,
        docker_daemon_ready=docker_ready,
        swebench_version=swebench_version,
        swebench_distribution_sha256=swebench_distribution_sha256,
        free_storage_gib=free,
    )


def _command_plan(args: argparse.Namespace) -> int:
    evidence_run = not args.non_evidence_smoke
    expected = EXPECTED_INSTANCES if evidence_run else args.smoke_count
    if not evidence_run and (not 2 <= expected <= 10 or expected % 2):
        raise ProtocolError("balanced non-evidence smoke must contain 2, 4, 6, 8, or 10 instances")
    instances = load_public_manifest(args.instances, expected_count=expected, evidence_run=evidence_run)
    document = private_schedule_document(instances, evidence_run=evidence_run)
    state_directory = create_private_directory(args.state_dir)
    output = state_directory / "private-schedule.json"
    _atomic_write(output, canonical_json_bytes(document))
    print(json.dumps(document["source_free_summary"], sort_keys=True))
    return 0


def _command_prepare(args: argparse.Namespace) -> int:
    paths = prepare_official_snapshots(args.parquet, args.output_directory)
    print(
        json.dumps(
            {
                "parquet_sha256": DATASET_PARQUET_SHA256,
                "full_snapshot_sha256": sha256_file(paths["full"]),
                "public_snapshot_sha256": sha256_file(paths["public"]),
            },
            sort_keys=True,
        )
    )
    return 0


def _command_export(args: argparse.Namespace) -> int:
    _document, schedule = load_private_schedule(args.schedule)
    checkpoints = require_private_path(args.checkpoints, must_exist=True)
    output_directory = create_private_directory(args.output_directory)
    paths = export_official_predictions(schedule, CheckpointStore(checkpoints), output_directory)
    print(json.dumps({condition: str(path) for condition, path in paths.items()}, sort_keys=True))
    return 0


def _command_commands(args: argparse.Namespace) -> int:
    document, schedule = load_private_schedule(args.schedule)
    if document.get("evidence_class") == "confirmatory" and args.timeout != CONFIRMATORY_TIMEOUT_SECONDS:
        raise ProtocolError("confirmatory evaluation timeout is frozen at 1800 seconds")
    for command in official_harness_commands(
        args.predictions_directory,
        args.full_snapshot,
        schedule_sha256=schedule_digest(schedule),
        max_workers=args.max_workers,
        timeout_seconds=args.timeout,
    ):
        print(shlex.join(command))
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    document, schedule = load_private_schedule(args.schedule)
    evidence_run = document.get("evidence_class") == "confirmatory"
    if evidence_run:
        if len(schedule) != 2 * EXPECTED_INSTANCES:
            raise ProtocolError("confirmatory evaluation requires all 500 paired tasks")
        if args.timeout != CONFIRMATORY_TIMEOUT_SECONDS:
            raise ProtocolError("confirmatory evaluation timeout is frozen at 1800 seconds")
    require_confirmatory_generation_attestation(evidence_run)
    host = probe_evaluation_host()
    if not host["ready"]:
        raise ProtocolError(f"official evaluation host preflight failed: {host['blockers']}")
    checkpoints = require_private_path(args.checkpoints, must_exist=True)
    require_private_path(args.predictions_directory, must_exist=True)
    seal_path = require_private_path(args.seal, must_exist=False)
    receipt_path = require_private_path(args.receipt, must_exist=False)
    evaluation_directory = create_private_directory(args.evaluation_directory)
    model_before = verify_expected_model(args.model_path)
    seal = build_evaluation_seal(
        schedule,
        CheckpointStore(checkpoints),
        args.predictions_directory,
        args.full_snapshot,
        max_workers=args.max_workers,
        timeout_seconds=args.timeout,
        harness_distribution_sha256=str(host["swebench_distribution_sha256"]),
    )
    if git_clean_head(ROOT) != seal["generation_binding"]["mio_commit"]:
        raise ProtocolError("current clean Mio commit differs from generation checkpoints")
    _atomic_write(seal_path, canonical_json_bytes(seal))
    commands = tuple(tuple(value) for value in seal["commands"])
    for condition, command in zip(CONDITIONS, commands, strict=True):
        log_path = evaluation_directory / f"{condition}.harness.log"
        with log_path.open("xb") as log:
            os.fchmod(log.fileno(), 0o600)
            result = subprocess.run(
                command,
                cwd=evaluation_directory,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.flush()
            os.fsync(log.fileno())
        if result.returncode:
            raise ProtocolError(f"official {condition} harness process failed")

    report_paths = {
        condition: official_report_path(
            evaluation_directory,
            condition,
            str(seal["predictions"][condition]["run_id"]),
        )
        for condition in CONDITIONS
    }
    for report_path in report_paths.values():
        verified_report = require_private_path(report_path, must_exist=True)
        if not verified_report.is_file():
            raise ProtocolError("official harness did not produce the expected bound report")
        os.chmod(verified_report, 0o600)
    model_after = verify_expected_model(args.model_path)
    receipt = build_evaluation_receipt(
        seal,
        schedule,
        report_paths,
        observed_model_identity_before=model_before,
        observed_model_identity_after=model_after,
    )
    _atomic_write(receipt_path, canonical_json_bytes(receipt))
    print(json.dumps({"evaluation_receipt_sha256": sha256_file(receipt_path)}, sort_keys=True))
    return 0


def _command_aggregate(args: argparse.Namespace) -> int:
    document, schedule = load_private_schedule(args.schedule)
    evidence_run = document.get("evidence_class") == "confirmatory"
    require_confirmatory_generation_attestation(evidence_run)
    checkpoints = require_private_path(args.checkpoints, must_exist=True)
    report_paths = {
        "gate_off": args.gate_off_report,
        "gate_on": args.gate_on_report,
    }
    receipt, receipt_sha256 = verify_evaluation_receipt(
        args.receipt,
        schedule,
        CheckpointStore(checkpoints),
        args.predictions_directory,
        args.full_snapshot,
        report_paths,
    )
    aggregate = aggregate_official_reports(
        args.gate_off_report,
        args.gate_on_report,
        expected_ids=_expected_instance_ids(schedule),
        evidence_run=evidence_run,
        schedule_sha256=schedule_digest(schedule),
        evaluation_receipt_sha256=receipt_sha256,
        generation_binding=receipt["seal"]["generation_binding"],
    )
    _atomic_write(args.output, canonical_json_bytes(aggregate), mode=0o644)
    print(json.dumps(aggregate["claim_gate"], sort_keys=True))
    return 0


def _command_preflight(_args: argparse.Namespace) -> int:
    report = probe_evaluation_host()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="fail-closed official evaluation-host check")
    preflight.set_defaults(handler=_command_preflight)

    prepare = subparsers.add_parser("prepare", help="redact the exact pinned official parquet")
    prepare.add_argument("--parquet", type=Path, required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    prepare.set_defaults(handler=_command_prepare)

    plan = subparsers.add_parser("plan", help="seal a redacted paired generation schedule")
    plan.add_argument("--instances", type=Path, required=True)
    plan.add_argument("--state-dir", type=Path, required=True)
    plan.add_argument("--non-evidence-smoke", action="store_true")
    plan.add_argument("--smoke-count", type=int, default=10)
    plan.set_defaults(handler=_command_plan)

    export = subparsers.add_parser("export", help="export complete immutable checkpoints as official JSONL")
    export.add_argument("--schedule", type=Path, required=True)
    export.add_argument("--checkpoints", type=Path, required=True)
    export.add_argument("--output-directory", type=Path, required=True)
    export.set_defaults(handler=_command_export)

    commands = subparsers.add_parser("commands", help="print pinned official Docker harness commands")
    commands.add_argument("--schedule", type=Path, required=True)
    commands.add_argument("--predictions-directory", type=Path, required=True)
    commands.add_argument("--full-snapshot", type=Path, required=True)
    commands.add_argument("--max-workers", type=int, default=6)
    commands.add_argument("--timeout", type=int, default=1800)
    commands.set_defaults(handler=_command_commands)

    evaluate = subparsers.add_parser("evaluate", help="run and seal the pinned official Docker harness")
    evaluate.add_argument("--schedule", type=Path, required=True)
    evaluate.add_argument("--checkpoints", type=Path, required=True)
    evaluate.add_argument("--predictions-directory", type=Path, required=True)
    evaluate.add_argument("--full-snapshot", type=Path, required=True)
    evaluate.add_argument("--model-path", type=Path, required=True)
    evaluate.add_argument("--evaluation-directory", type=Path, required=True)
    evaluate.add_argument("--seal", type=Path, required=True)
    evaluate.add_argument("--receipt", type=Path, required=True)
    evaluate.add_argument("--max-workers", type=int, default=6)
    evaluate.add_argument("--timeout", type=int, default=1800)
    evaluate.set_defaults(handler=_command_evaluate)

    aggregate = subparsers.add_parser("aggregate", help="create a source-free paired official-result aggregate")
    aggregate.add_argument("--schedule", type=Path, required=True)
    aggregate.add_argument("--checkpoints", type=Path, required=True)
    aggregate.add_argument("--predictions-directory", type=Path, required=True)
    aggregate.add_argument("--full-snapshot", type=Path, required=True)
    aggregate.add_argument("--receipt", type=Path, required=True)
    aggregate.add_argument("--gate-off-report", type=Path, required=True)
    aggregate.add_argument("--gate-on-report", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(handler=_command_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ProtocolError, subprocess.TimeoutExpired) as exc:
        print(f"SWE-bench quality protocol blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
