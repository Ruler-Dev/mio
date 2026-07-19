"""Repository-level depth-one quality-effort pilot primitives.

This module deliberately contains no benchmark corpus and no hidden evaluator.
It provides the trust-boundary plumbing needed by a later four-arm pilot:

* explicit, immutable direct and recovery budgets;
* regular-file-only workspace cloning under an explicit containment root;
* a retained native-agent session with a cold engine reset at every stage;
* reconstruction of a recovery quality gate against the pristine baseline;
* content-free public-state extraction; and
* a barrier that makes hidden evaluation unreachable until every terminal
  workspace has been selected and sealed.

The recovery stage is depth one.  It is not a production effort controller and
does not claim a Markov, quality, or throughput improvement.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mio.agent import (
    AgentExecutionBudget,
    AgentRoundTrace,
    AgentToolTrace,
    AgentTurnResult,
)
from mio.agent_policy import AgentToolPermission, AgentToolPolicy
from mio.coding_quality import (
    CodingEffort,
    CodingQualityGate,
    RequestIntent,
    ValidationEvidence,
    ValidationKind,
    WorkspaceSnapshot,
    snapshot_workspaces,
)
from experimental.effort.repository_quality_pilot import (
    LOGICAL_ARMS,
    ArmHiddenOutcome,
    CandidateChoice,
    CandidateCost as ProtocolCandidateCost,
    CandidateObservation,
    EvaluationBarrierReceipt,
    FixturePilotRecord,
    GenerationCompletionReceipt,
    HiddenOutcome,
    LogicalArm,
    PublicEvidence as ProtocolPublicEvidence,
    VisibleCheckOutcome,
    select_candidate,
)


DIRECT_EXECUTION_BUDGET = AgentExecutionBudget(
    max_rounds=12,
    max_tool_calls=32,
    max_output_tokens=2_048,
    max_wall_seconds=120.0,
    max_context_tokens=8_192,
)
EXTRA_EXECUTION_BUDGET = AgentExecutionBudget(
    max_rounds=4,
    max_tool_calls=8,
    max_output_tokens=384,
    max_wall_seconds=20.0,
    max_context_tokens=8_192,
)

RECOVERY_PROMPT = (
    "Perform one bounded review and repair pass over the current attempted implementation. "
    "Inspect the existing changes and visible tests for edge cases, API-contract errors, and "
    "regressions. Make only justified corrections. Use the trusted validate tool; do not make "
    "cosmetic edits merely to satisfy the gate. The original task remains: {instruction}"
)

FROZEN_RUNNER_SETTINGS: Mapping[str, object] = MappingProxyType(
    {
        "context_window": 8_192,
        "max_output_tokens": 2_048,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "drafter_backend": "dflash",
        "drafter_strict": True,
        "draft_fallback_model": None,
        "tq_bits": 16,
        "pq_bits": 16,
        "bmp_paths": 1,
        "ddtree_budget": 0,
        "dspark_prefix_cache": False,
    }
)
RESET_MANIFEST = (
    "target_prefix_cache",
    "last_prompt_tokens",
    "pending_assistant_prefill",
    "dspark_prefix_cache_if_present",
)

FORBIDDEN_ENVIRONMENT_OVERRIDES = (
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

_PLAIN_TOOLS = ("bash", "read", "write", "edit")
_QUALITY_TOOLS = ("validate", *_PLAIN_TOOLS)
_VALIDATION_KINDS = ("test", "build", "static", "diff", "review")
_PUBLIC_TEST_STATUSES = frozenset({"not_run", "passed", "failed", "error"})
_SHA256_HEX = frozenset("0123456789abcdef")
_GATE_DECISIONS = frozenset({"pass", "incomplete", "not_applicable"})
_CHANGED_KINDS = frozenset({"code", "docs"})


class RepositoryPilotProtocolError(RuntimeError):
    """Raised when a repository-pilot trust boundary cannot be established."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _regular_tree_entries(root: Path) -> tuple[tuple[str, str, int, bytes], ...]:
    """Return a deterministic tree representation while rejecting aliases."""

    try:
        canonical = root.expanduser().resolve(strict=True)
        root_stat = root.lstat()
    except (OSError, RuntimeError) as exc:
        raise RepositoryPilotProtocolError(f"workspace root is unavailable: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RepositoryPilotProtocolError("workspace root must be a real directory")

    rows: list[tuple[str, str, int, bytes]] = []

    def visit(directory: Path, relative_prefix: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise RepositoryPilotProtocolError("workspace tree cannot be enumerated") from exc
        for entry in entries:
            relative = relative_prefix / entry.name
            relative_name = relative.as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RepositoryPilotProtocolError("workspace entry cannot be inspected") from exc
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISLNK(metadata.st_mode):
                raise RepositoryPilotProtocolError("workspace trees may not contain symlinks")
            if stat.S_ISDIR(metadata.st_mode):
                rows.append((relative_name, "D", mode, b""))
                visit(Path(entry.path), relative)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RepositoryPilotProtocolError("workspace trees may contain only regular files")
            if metadata.st_nlink != 1:
                raise RepositoryPilotProtocolError("workspace trees may not contain hard-linked files")
            try:
                payload = Path(entry.path).read_bytes()
            except OSError as exc:
                raise RepositoryPilotProtocolError("workspace file cannot be read") from exc
            try:
                after = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RepositoryPilotProtocolError("workspace file changed during attestation") from exc
            before_identity = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before_identity != after_identity:
                raise RepositoryPilotProtocolError("workspace file changed during attestation")
            rows.append((relative_name, "F", mode, payload))

    visit(canonical, Path())
    return tuple(rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise RepositoryPilotProtocolError(f"{label} must be a lowercase SHA-256")
    return value


def _scope_manifest_sha256(
    entries: Sequence[tuple[str, str, int, str | None]],
) -> str:
    digest = hashlib.sha256(b"mio.repository-public-scope-manifest.v1\0")
    for relative, kind, mode, content_sha256 in entries:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(kind.encode("ascii"))
        digest.update(mode.to_bytes(4, "big"))
        digest.update((content_sha256 or "").encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PublicScopeVerdict:
    """Host-only scope result sealed to one fixture and terminal tree."""

    fixture_id: str
    contract_sha256: str
    pristine_manifest_sha256: str
    terminal_tree_sha256: str
    scope_valid: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self.fixture_id) is not str or not self.fixture_id:
            raise ValueError("scope verdict fixture_id must be a non-empty string")
        for label, value in (
            ("scope verdict contract", self.contract_sha256),
            ("scope verdict pristine manifest", self.pristine_manifest_sha256),
            ("scope verdict terminal tree", self.terminal_tree_sha256),
        ):
            _require_sha256(value, label)
        if type(self.scope_valid) is not bool:
            raise TypeError("scope_valid must be bool")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("scope verdict reason must be a non-empty string")
        if self.scope_valid != (self.reason == "valid"):
            raise ValueError("scope verdict validity contradicts its reason")


@dataclass(frozen=True)
class PublicScopeContract:
    """Host-only pristine manifest and editable-name allowlist.

    The contract is intentionally never placed in agent state or serialized as
    controller input.  Only its boolean verdict crosses the public-state
    boundary after the exact terminal tree has been re-attested.
    """

    fixture_id: str
    pristine_tree_sha256: str
    pristine_manifest_sha256: str
    editable_names: tuple[str, ...]
    contract_sha256: str
    _pristine_entries: tuple[tuple[str, str, int, str | None], ...] = field(
        repr=False,
    )

    @classmethod
    def capture(
        cls,
        fixture_id: str,
        pristine_root: Path,
        *,
        editable_names: Sequence[str],
    ) -> PublicScopeContract:
        if type(fixture_id) is not str or not fixture_id:
            raise ValueError("scope contract fixture_id must be a non-empty string")
        if type(editable_names) not in {tuple, list}:
            raise TypeError("editable_names must be a sequence of relative file names")
        editable = tuple(editable_names)
        if not editable or any(type(name) is not str or not name for name in editable):
            raise ValueError("editable_names must contain non-empty strings")
        if len(editable) != len(set(editable)):
            raise ValueError("editable_names must be unique")
        for name in editable:
            path = Path(name)
            if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise ValueError("editable_names must be normalized relative paths")

        root = Path(pristine_root).expanduser()
        tree_sha256, entries = _regular_tree_attestation(root)
        entry_by_name = {entry[0]: entry for entry in entries}
        if any(name not in entry_by_name or entry_by_name[name][1] != "F" for name in editable):
            raise ValueError("every editable name must identify a pristine regular file")
        manifest_sha256 = _scope_manifest_sha256(entries)
        digest = hashlib.sha256(b"mio.repository-public-scope-contract.v1\0")
        digest.update(fixture_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tree_sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(manifest_sha256.encode("ascii"))
        for name in editable:
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
        return cls(
            fixture_id=fixture_id,
            pristine_tree_sha256=tree_sha256,
            pristine_manifest_sha256=manifest_sha256,
            editable_names=editable,
            contract_sha256=digest.hexdigest(),
            _pristine_entries=entries,
        )

    def assess_terminal(
        self,
        *,
        fixture_id: str,
        terminal_root: Path,
        terminal_tree_sha256: str,
    ) -> PublicScopeVerdict:
        """Return a digest-bound public verdict or reject a stale binding."""

        if fixture_id != self.fixture_id:
            raise RepositoryPilotProtocolError("scope contract was used for the wrong fixture")
        declared_terminal = _require_sha256(terminal_tree_sha256, "declared terminal tree")
        root = Path(terminal_root).expanduser()
        actual_terminal, current_entries = _regular_tree_attestation(root)
        if actual_terminal != declared_terminal:
            raise RepositoryPilotProtocolError("scope verdict terminal digest is stale")
        pristine_by_name = {entry[0]: entry for entry in self._pristine_entries}
        current_by_name = {entry[0]: entry for entry in current_entries}

        reason = "valid"
        if tuple(pristine_by_name) != tuple(current_by_name):
            reason = "name_set_changed"
        elif any(
            (current_by_name[name][1], current_by_name[name][2])
            != (pristine_by_name[name][1], pristine_by_name[name][2])
            for name in pristine_by_name
        ):
            reason = "kind_or_mode_changed"
        elif any(
            current_by_name[name][3] != pristine_by_name[name][3]
            for name in pristine_by_name
            if pristine_by_name[name][1] == "F" and name not in self.editable_names
        ):
            reason = "noneditable_bytes_changed"
        elif not any(current_by_name[name][3] != pristine_by_name[name][3] for name in self.editable_names):
            reason = "no_editable_bytes_changed"
        return PublicScopeVerdict(
            fixture_id=self.fixture_id,
            contract_sha256=self.contract_sha256,
            pristine_manifest_sha256=self.pristine_manifest_sha256,
            terminal_tree_sha256=actual_terminal,
            scope_valid=reason == "valid",
            reason=reason,
        )


def regular_tree_sha256(root: Path) -> str:
    """Hash names, kinds, modes, and bytes of a safe regular-file tree."""

    return _regular_tree_rows_sha256(_regular_tree_entries(Path(root)))


def _regular_tree_rows_sha256(rows: Sequence[tuple[str, str, int, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, kind, mode, payload in rows:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(kind.encode("ascii"))
        digest.update(mode.to_bytes(4, "big"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _regular_tree_attestation(
    root: Path,
) -> tuple[str, tuple[tuple[str, str, int, str | None], ...]]:
    first = _regular_tree_entries(Path(root))
    digest = _regular_tree_rows_sha256(first)
    manifest = tuple(
        (
            relative,
            kind,
            mode,
            _sha256_bytes(payload) if kind == "F" else None,
        )
        for relative, kind, mode, payload in first
    )
    if _regular_tree_entries(Path(root)) != first:
        raise RepositoryPilotProtocolError("workspace tree changed during terminal attestation")
    return digest, manifest


def safe_clone_workspace(source: Path, destination: Path, *, containment_root: Path) -> str:
    """Clone a regular tree without allowing a destination escape or alias."""

    source_path = Path(source).expanduser()
    regular_tree_sha256(source_path)
    source_root = source_path.resolve(strict=True)
    containment = Path(containment_root).expanduser().resolve(strict=True)
    if not containment.is_dir():
        raise RepositoryPilotProtocolError("clone containment root must be a directory")
    destination_path = Path(destination).expanduser()
    if destination_path.exists() or destination_path.is_symlink():
        raise RepositoryPilotProtocolError("clone destination must not already exist")
    try:
        destination_parent = destination_path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RepositoryPilotProtocolError("clone destination parent is unavailable") from exc
    canonical_destination = destination_parent / destination_path.name
    if canonical_destination == containment or not _is_within(canonical_destination, containment):
        raise RepositoryPilotProtocolError("clone destination escapes its containment root")
    if _is_within(canonical_destination, source_root) or _is_within(source_root, canonical_destination):
        raise RepositoryPilotProtocolError("clone source and destination may not overlap")

    source_digest = regular_tree_sha256(source_root)
    canonical_destination.mkdir(mode=0o700)
    try:
        for relative, kind, mode, payload in _regular_tree_entries(source_root):
            target = canonical_destination / relative
            if kind == "D":
                target.mkdir(mode=mode)
                os.chmod(target, mode, follow_symlinks=False)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                finally:
                    os.close(descriptor)
                os.chmod(target, mode, follow_symlinks=False)
    except Exception:
        # A partial destination is never a valid candidate.  Leave cleanup to
        # the caller's disposable work root, but make the failure explicit.
        raise

    if regular_tree_sha256(source_root) != source_digest:
        raise RepositoryPilotProtocolError("clone source changed during the copy")
    if regular_tree_sha256(canonical_destination) != source_digest:
        raise RepositoryPilotProtocolError("clone bytes or modes do not match the source")
    return source_digest


@dataclass(frozen=True)
class DirectWorkspaceRoots:
    """Independent pristine roots for the Plain and Quality direct arms."""

    plain: Path
    quality: Path
    pristine_sha256: str

    def verify_pristine(self) -> None:
        for root in (self.plain, self.quality):
            if regular_tree_sha256(root) != self.pristine_sha256:
                raise RepositoryPilotProtocolError("a direct arm no longer matches the pristine fixture")


def prepare_pristine_direct_roots(source: Path, work_root: Path) -> DirectWorkspaceRoots:
    """Materialize isolated Plain and Quality roots from one public fixture."""

    source_path = Path(source).expanduser()
    regular_tree_sha256(source_path)
    source_root = source_path.resolve(strict=True)
    target_root = Path(work_root).expanduser()
    if target_root.exists():
        if target_root.is_symlink() or not target_root.is_dir() or any(target_root.iterdir()):
            raise RepositoryPilotProtocolError("direct-arm work root must be an empty real directory")
    else:
        target_root.mkdir(parents=True, mode=0o700)
    target_root = target_root.resolve(strict=True)
    if _is_within(target_root, source_root) or _is_within(source_root, target_root):
        raise RepositoryPilotProtocolError("fixture source and direct-arm work root may not overlap")

    pristine = regular_tree_sha256(source_root)
    plain = target_root / "plain"
    quality = target_root / "quality"
    safe_clone_workspace(source_root, plain, containment_root=target_root)
    safe_clone_workspace(source_root, quality, containment_root=target_root)
    if regular_tree_sha256(source_root) != pristine:
        raise RepositoryPilotProtocolError("pristine fixture changed while direct roots were prepared")
    roots = DirectWorkspaceRoots(plain=plain, quality=quality, pristine_sha256=pristine)
    roots.verify_pristine()
    return roots


@dataclass(frozen=True)
class ImmutableWorkspaceArchive:
    """A digest-sealed clone whose immutability is verified at every use."""

    root: Path
    tree_sha256: str
    snapshot: WorkspaceSnapshot

    @classmethod
    def capture(
        cls,
        source: Path,
        destination: Path,
        *,
        containment_root: Path,
    ) -> ImmutableWorkspaceArchive:
        digest = safe_clone_workspace(source, destination, containment_root=containment_root)
        snapshot = snapshot_workspaces((destination,))
        if not snapshot.complete:
            raise RepositoryPilotProtocolError("archive workspace snapshot is incomplete")
        return cls(Path(destination).resolve(strict=True), digest, snapshot)

    def verify_unchanged(self) -> None:
        if regular_tree_sha256(self.root) != self.tree_sha256:
            raise RepositoryPilotProtocolError("immutable workspace archive changed")
        current = snapshot_workspaces((self.root,))
        if not current.complete or current.content_sha256 != self.snapshot.content_sha256:
            raise RepositoryPilotProtocolError("immutable workspace archive snapshot changed")

    def clone_to(self, destination: Path, *, containment_root: Path) -> Path:
        self.verify_unchanged()
        safe_clone_workspace(self.root, destination, containment_root=containment_root)
        self.verify_unchanged()
        return Path(destination).resolve(strict=True)


@dataclass
class RetainedAgentStage:
    """One native agent stage plus the state needed for a safe continuation."""

    stage: str
    fixture_id: str
    instruction: str
    workspace: Path
    pristine_tree_sha256: str
    terminal_tree_sha256: str
    pristine_snapshot: WorkspaceSnapshot
    current_snapshot: WorkspaceSnapshot
    execution_budget: AgentExecutionBudget
    coding_effort: str
    drafter_ref: str
    state: dict[str, Any] = field(repr=False)
    result: Any = field(repr=False)
    trusted_quality_gate: CodingQualityGate | None = field(default=None, repr=False)
    quality_enabled: bool = True
    reset_manifest: tuple[str, ...] = RESET_MANIFEST


class _BootstrapRetainedCodingQualityGate(CodingQualityGate):
    """Retain the exact gate object created by this host adapter.

    The native loop normally refuses a pristine pending gate because ordinary
    interactive state should not pressure a later turn to edit.  This adapter
    creates the gate for the *same* physical turn solely so trusted validation
    evidence remains host-inspectable.  Exactly one bootstrap reuse check is
    overridden; every later persistence decision keeps native semantics.
    """

    _bootstrap_reuse_pending = True

    def should_persist(self) -> bool:
        if self._bootstrap_reuse_pending:
            self._bootstrap_reuse_pending = False
            return True
        return super().should_persist()


def _tool_surface(agent_module: Any, *, quality_enabled: bool) -> tuple[Mapping[str, Any], tuple[dict, ...]]:
    names = _QUALITY_TOOLS if quality_enabled else _PLAIN_TOOLS
    registry_source = getattr(agent_module, "AGENT_TOOLS", None)
    specs_source = getattr(agent_module, "AGENT_TOOLS_SPEC", None)
    if not isinstance(registry_source, Mapping) or not isinstance(specs_source, (list, tuple)):
        raise RepositoryPilotProtocolError("agent module has no valid native tool surface")
    try:
        registry = MappingProxyType({name: registry_source[name] for name in names})
    except KeyError as exc:
        raise RepositoryPilotProtocolError("agent module is missing a required native tool") from exc
    by_name = {
        spec.get("function", {}).get("name"): spec
        for spec in specs_source
        if isinstance(spec, dict) and isinstance(spec.get("function"), dict)
    }
    try:
        specs = tuple(by_name[name] for name in names)
    except KeyError as exc:
        raise RepositoryPilotProtocolError("agent module is missing a required native tool schema") from exc
    return registry, specs


class RetainedNativeAgentExecutor:
    """Run cold, isolated direct stages and one copied-history recovery stage."""

    def __init__(
        self,
        *,
        config: Any,
        manager: Any,
        engine: Any,
        tier: str,
        agent_module: Any | None = None,
    ) -> None:
        if agent_module is None:
            from mio import agent as agent_module

        self.config = config
        self.manager = manager
        self.engine = engine
        self.tier = tier
        self.agent_module = agent_module

    def _assert_frozen_runner_settings(self) -> None:
        getter = getattr(self.manager, "get_engine", None)
        if not callable(getter) or getter(self.tier) is not self.engine:
            raise RepositoryPilotProtocolError("manager tier does not resolve to the retained engine identity")
        active_overrides = tuple(name for name in FORBIDDEN_ENVIRONMENT_OVERRIDES if name in os.environ)
        if active_overrides:
            raise RepositoryPilotProtocolError(
                "forbidden benchmark environment override is present: " + ", ".join(active_overrides)
            )
        tier_config = getattr(self.engine, "tier_config", None)
        if tier_config is None:
            raise RepositoryPilotProtocolError("engine has no inspectable frozen tier settings")
        for name, expected in FROZEN_RUNNER_SETTINGS.items():
            if getattr(tier_config, name, object()) != expected:
                raise RepositoryPilotProtocolError(f"engine runner setting {name} is not frozen")
        drafter_status = getattr(self.engine, "drafter_status", None)
        if not isinstance(drafter_status, Mapping) or (
            drafter_status.get("requested") != "dflash"
            or drafter_status.get("selected") != "dflash"
            or drafter_status.get("fallback_used") is not False
            or drafter_status.get("strict") is not True
        ):
            raise RepositoryPilotProtocolError("engine did not retain the strict DFlash primary")
        _strict_string(drafter_status.get("ref"), "strict DFlash reference")

    def _drafter_ref(self) -> str:
        status = getattr(self.engine, "drafter_status", None)
        if not isinstance(status, Mapping):
            raise RepositoryPilotProtocolError("engine has no typed DFlash status")
        return _strict_string(status.get("ref"), "strict DFlash reference")

    def _cold_reset(self) -> tuple[str, ...]:
        self._assert_frozen_runner_settings()
        invalidator = getattr(self.engine, "_prefix_cache_invalidate", None)
        if not callable(invalidator):
            raise RepositoryPilotProtocolError("engine has no target-prefix reset primitive")
        invalidator()
        if getattr(self.engine, "_prefix_cache", None):
            raise RepositoryPilotProtocolError("target prefix cache did not reset")
        if not hasattr(self.engine, "_last_prompt_tokens") or not hasattr(self.engine, "_pending_assistant_prefill"):
            raise RepositoryPilotProtocolError("engine lacks resettable prompt state")
        self.engine._last_prompt_tokens = []
        self.engine._pending_assistant_prefill = ""
        dspark = getattr(self.engine, "_dspark_runtime", None)
        prefix_cache = getattr(dspark, "_prefix_cache", None)
        dspark_executor = getattr(dspark, "_executor", None)
        if prefix_cache is not None and dspark_executor is not None:
            dspark_executor.submit(prefix_cache.reset).result()
        self._assert_frozen_runner_settings()
        return RESET_MANIFEST

    def _invoke(self, instruction: str, state: dict[str, Any]) -> tuple[Any, tuple[str, ...]]:
        process = getattr(self.agent_module, "_process_user_input", None)
        if not callable(process):
            raise RepositoryPilotProtocolError("agent module has no native processing entry point")
        reset_manifest = self._cold_reset()
        previous_console = getattr(self.agent_module, "console", None)
        try:
            from rich.console import Console

            if hasattr(self.agent_module, "console"):
                self.agent_module.console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
            result = process(instruction, self.engine, self.manager, self.config, state)
            self._assert_frozen_runner_settings()
            return result, reset_manifest
        finally:
            if hasattr(self.agent_module, "console"):
                self.agent_module.console = previous_console

    @staticmethod
    def _coding_policy(workspace: Path) -> AgentToolPolicy:
        policy = AgentToolPolicy.coding_workspace(workspace, allow_network=False)
        if AgentToolPermission.NETWORK in policy.permissions or policy.workspace_roots != (workspace.resolve(),):
            raise RepositoryPilotProtocolError("native stage policy escaped its single local workspace")
        return policy

    def run_direct(
        self,
        *,
        fixture_id: str,
        instruction: str,
        workspace: Path,
        quality_enabled: bool,
        effort: str = "medium",
    ) -> RetainedAgentStage:
        if type(fixture_id) is not str or not fixture_id:
            raise ValueError("fixture_id must be a non-empty string")
        if type(quality_enabled) is not bool:
            raise TypeError("quality_enabled must be bool")
        if type(effort) is not str or effort != CodingEffort.MEDIUM.value:
            raise RepositoryPilotProtocolError("direct effort must be exactly medium")
        workspace_path = Path(workspace).expanduser()
        pristine_tree_sha256 = regular_tree_sha256(workspace_path)
        root = workspace_path.resolve(strict=True)
        pristine = snapshot_workspaces((root,))
        if not pristine.complete:
            raise RepositoryPilotProtocolError("direct pristine snapshot is incomplete")
        registry, specs = _tool_surface(self.agent_module, quality_enabled=quality_enabled)
        trusted_gate: CodingQualityGate | None = None
        if quality_enabled:
            trusted_gate = _BootstrapRetainedCodingQualityGate(
                roots=(root,),
                effort=CodingEffort.MEDIUM,
                enabled=True,
                intent=RequestIntent.CODE_CHANGE_REQUESTED,
                require_net_workspace_change=True,
                request_sha256=hashlib.sha256(instruction.encode("utf-8", errors="replace")).hexdigest(),
                initial_snapshot=pristine,
                current_snapshot=pristine,
            )
        state: dict[str, Any] = {
            "tier": self.tier,
            "prompt_policy": self.agent_module.PromptPolicy()
            if hasattr(self.agent_module, "PromptPolicy")
            else _default_prompt_policy(),
            "tool_policy": self._coding_policy(root),
            "tool_registry": registry,
            "tool_specs": specs,
            "messages": [],
            "quality_gate_enabled": quality_enabled,
            "quality_gate_require_change": quality_enabled,
            "coding_effort": CodingEffort.MEDIUM.value,
            "execution_budget": DIRECT_EXECUTION_BUDGET,
        }
        if trusted_gate is not None:
            state["_quality_gate"] = trusted_gate
            state["quality_gate_pending"] = True
        result, reset_manifest = self._invoke(instruction, state)
        terminal_tree_sha256 = regular_tree_sha256(root)
        current = snapshot_workspaces((root,))
        if not current.complete:
            raise RepositoryPilotProtocolError("direct terminal snapshot is incomplete")
        return RetainedAgentStage(
            stage="direct",
            fixture_id=fixture_id,
            instruction=instruction,
            workspace=root,
            pristine_tree_sha256=pristine_tree_sha256,
            terminal_tree_sha256=terminal_tree_sha256,
            pristine_snapshot=pristine,
            current_snapshot=current,
            execution_budget=DIRECT_EXECUTION_BUDGET,
            coding_effort=CodingEffort.MEDIUM.value,
            drafter_ref=self._drafter_ref(),
            state=state,
            result=result,
            trusted_quality_gate=trusted_gate,
            quality_enabled=quality_enabled,
            reset_manifest=reset_manifest,
        )

    def run_recovery(
        self,
        *,
        direct: RetainedAgentStage,
        archive: ImmutableWorkspaceArchive,
        branch_root: Path,
        containment_root: Path,
    ) -> RetainedAgentStage:
        if direct.stage != "direct" or not direct.quality_enabled:
            raise RepositoryPilotProtocolError("recovery requires a retained Quality direct stage")
        _validate_stage_policy_and_budget(direct)
        if direct.coding_effort != CodingEffort.MEDIUM.value or direct.execution_budget != DIRECT_EXECUTION_BUDGET:
            raise RepositoryPilotProtocolError("recovery parent does not have the frozen direct effort and budget")
        if direct.drafter_ref != self._drafter_ref():
            raise RepositoryPilotProtocolError("recovery parent and engine DFlash identities differ")
        if regular_tree_sha256(direct.workspace) != direct.terminal_tree_sha256:
            raise RepositoryPilotProtocolError("retained direct workspace changed before recovery")
        if archive.snapshot.content_sha256 != direct.current_snapshot.content_sha256:
            raise RepositoryPilotProtocolError("archive is not the retained direct workspace")
        if archive.tree_sha256 != direct.terminal_tree_sha256:
            raise RepositoryPilotProtocolError("archive tree digest is not the retained direct terminal")
        if not isinstance(direct.result, AgentTurnResult):
            raise RepositoryPilotProtocolError("recovery parent result is not a typed AgentTurnResult")
        if not isinstance(direct.trusted_quality_gate, CodingQualityGate):
            raise RepositoryPilotProtocolError("recovery parent has no retained trusted Quality gate")
        if direct.trusted_quality_gate.roots != (direct.workspace.resolve(),):
            raise RepositoryPilotProtocolError("recovery parent gate is bound to another workspace")
        if direct.trusted_quality_gate.effort is not CodingEffort.MEDIUM:
            raise RepositoryPilotProtocolError("recovery parent gate effort is not medium")
        direct_report = _required_attr(direct.result, "quality_gate", "recovery parent result")
        if type(direct_report) is not dict:
            raise RepositoryPilotProtocolError("recovery parent has no strict Quality report")
        reported_epoch = _nonnegative_int(direct_report.get("mutation_epoch"), "direct mutation epoch")
        raw_kinds = direct_report.get("changed_kinds")
        if type(raw_kinds) is not list or any(type(value) is not str for value in raw_kinds):
            raise RepositoryPilotProtocolError("direct changed_kinds must be a string list")
        if len(raw_kinds) != len(set(raw_kinds)) or any(value not in _CHANGED_KINDS for value in raw_kinds):
            raise RepositoryPilotProtocolError("direct changed_kinds is malformed")
        changed_kinds = set(raw_kinds)
        direct_changed = direct.pristine_snapshot.content_sha256 != direct.current_snapshot.content_sha256
        if reported_epoch == 0:
            coherent_epoch = not changed_kinds and not direct_changed
        else:
            coherent_epoch = len(changed_kinds) > 0
        if not coherent_epoch:
            raise RepositoryPilotProtocolError("direct epoch, changed_kinds, and terminal snapshot contradict")
        if direct_report.get("current_content_sha256") != direct.current_snapshot.content_sha256:
            raise RepositoryPilotProtocolError("direct gate content hash does not bind its terminal snapshot")
        if direct_report != direct.trusted_quality_gate.report():
            raise RepositoryPilotProtocolError("recovery parent Quality report contradicts its trusted gate")
        branch = archive.clone_to(branch_root, containment_root=containment_root)
        current = snapshot_workspaces((branch,))
        if not current.complete or current.content_sha256 != direct.current_snapshot.content_sha256:
            raise RepositoryPilotProtocolError("recovery branch does not match its direct parent")
        gate = _BootstrapRetainedCodingQualityGate(
            roots=(branch,),
            effort=CodingEffort.HIGH,
            enabled=True,
            intent=RequestIntent.CODE_CHANGE_REQUESTED,
            require_net_workspace_change=True,
            request_sha256=hashlib.sha256(direct.instruction.encode("utf-8", errors="replace")).hexdigest(),
            initial_snapshot=direct.pristine_snapshot,
            current_snapshot=current,
            mutation_epoch=reported_epoch,
            changed_kinds=changed_kinds,
            validations=[],
            validate_invocations=0,
            misrouted_validation_commands=0,
            successful_reads=0,
            snapshot_failed_closed=False,
        )
        registry, specs = _tool_surface(self.agent_module, quality_enabled=True)
        child_state = {
            key: value
            for key, value in direct.state.items()
            if key
            not in {
                "_quality_gate",
                "execution_budget",
                "quality_gate_pending",
                "tool_policy",
                "tool_registry",
                "tool_specs",
            }
        }
        child_state.update(
            {
                "tool_policy": self._coding_policy(branch),
                "tool_registry": registry,
                "tool_specs": specs,
                "messages": copy.deepcopy(list(direct.state.get("messages", ()))),
                "quality_gate_enabled": True,
                "quality_gate_require_change": True,
                "coding_effort": CodingEffort.HIGH.value,
                "execution_budget": EXTRA_EXECUTION_BUDGET,
                "_quality_gate": gate,
                "quality_gate_pending": True,
            }
        )
        recovery_prompt = RECOVERY_PROMPT.format(instruction=direct.instruction)
        result, reset_manifest = self._invoke(recovery_prompt, child_state)
        regular_tree_sha256(branch)
        terminal = snapshot_workspaces((branch,))
        if not terminal.complete:
            raise RepositoryPilotProtocolError("recovery terminal snapshot is incomplete")
        archive.verify_unchanged()
        if regular_tree_sha256(direct.workspace) != regular_tree_sha256(archive.root):
            raise RepositoryPilotProtocolError("retained direct root changed during recovery")
        return RetainedAgentStage(
            stage="recovery",
            fixture_id=direct.fixture_id,
            instruction=direct.instruction,
            workspace=branch,
            pristine_tree_sha256=direct.pristine_tree_sha256,
            terminal_tree_sha256=regular_tree_sha256(branch),
            pristine_snapshot=direct.pristine_snapshot,
            current_snapshot=terminal,
            execution_budget=EXTRA_EXECUTION_BUDGET,
            coding_effort=CodingEffort.HIGH.value,
            drafter_ref=self._drafter_ref(),
            state=child_state,
            result=result,
            trusted_quality_gate=gate,
            quality_enabled=True,
            reset_manifest=reset_manifest,
        )


def _default_prompt_policy() -> Any:
    from mio.prompt_policy import PromptPolicy

    return PromptPolicy()


@dataclass(frozen=True)
class VisiblePublicTestResult:
    attempted: bool
    passed: bool
    status: str

    def __post_init__(self) -> None:
        if type(self.attempted) is not bool or type(self.passed) is not bool:
            raise TypeError("visible public-test flags must be bool")
        if type(self.status) is not str or self.status not in _PUBLIC_TEST_STATUSES:
            raise ValueError("visible public-test status is not allowlisted")
        legal = {
            (False, False, "not_run"),
            (True, True, "passed"),
            (True, False, "failed"),
            (True, False, "error"),
        }
        if (self.attempted, self.passed, self.status) not in legal:
            raise ValueError("visible public-test fields are contradictory")


@dataclass(frozen=True)
class PublicRepositoryState:
    """Content-free controller input; it has no hidden-label field."""

    scope_valid: bool
    public_test_attempted: bool
    public_test_passed: bool
    public_test_status: str
    gate_present: bool
    gate_decision: str
    gate_phase: str
    gate_satisfied: bool
    initial_snapshot_complete: bool
    current_snapshot_complete: bool
    net_workspace_changed: bool
    mutation_epoch: int
    trusted_test_or_build_attempt_count: int
    validation_counts: tuple[tuple[str, int], ...]
    terminal_reason: str
    budget_exhausted: bool
    deadline_violated: bool
    tool_telemetry_complete: bool
    round_count: int
    tool_calls: int
    output_tokens: int
    model_seconds: float
    wall_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "scope_valid",
            "public_test_attempted",
            "public_test_passed",
            "gate_present",
            "gate_satisfied",
            "initial_snapshot_complete",
            "current_snapshot_complete",
            "net_workspace_changed",
            "budget_exhausted",
            "deadline_violated",
            "tool_telemetry_complete",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        VisiblePublicTestResult(
            self.public_test_attempted,
            self.public_test_passed,
            self.public_test_status,
        )
        if type(self.gate_decision) is not str or self.gate_decision not in _GATE_DECISIONS:
            raise ValueError("gate_decision is not allowlisted")
        if type(self.gate_phase) is not str or not self.gate_phase:
            raise ValueError("gate_phase must be a non-empty string")
        if self.gate_satisfied != (self.gate_decision in {"pass", "not_applicable"}):
            raise ValueError("gate satisfaction contradicts gate decision")
        if self.net_workspace_changed and not self.snapshot_complete:
            raise ValueError("net workspace change requires complete snapshots")
        _nonnegative_int(self.mutation_epoch, "mutation epoch")
        _nonnegative_int(
            self.trusted_test_or_build_attempt_count,
            "trusted test/build attempt count",
        )
        if type(self.validation_counts) is not tuple:
            raise TypeError("validation_counts must be a tuple")
        if tuple(name for name, _count in self.validation_counts) != _VALIDATION_KINDS:
            raise ValueError("validation_counts must use the exact frozen names and order")
        for name, count in self.validation_counts:
            _nonnegative_int(count, f"validation count {name}")
        successful_test_or_build = self.validation_count("test") + self.validation_count("build")
        if successful_test_or_build > self.trusted_test_or_build_attempt_count:
            raise ValueError("trusted test/build successes exceed attempts")
        if self.public_test_attempted != (self.trusted_test_or_build_attempt_count > 0):
            raise ValueError("public-test attempted flag contradicts trusted attempt count")
        if self.public_test_attempted and self.public_test_passed != (
            successful_test_or_build == self.trusted_test_or_build_attempt_count
        ):
            raise ValueError("public-test pass flag contradicts trusted attempts and successes")
        _strict_string(self.terminal_reason, "terminal reason")
        for name in ("round_count", "tool_calls", "output_tokens"):
            _nonnegative_int(getattr(self, name), name)
        for name in ("model_seconds", "wall_seconds"):
            _finite_nonnegative(getattr(self, name), name)

    def validation_count(self, kind: str) -> int:
        return dict(self.validation_counts).get(kind, 0)

    @property
    def snapshot_complete(self) -> bool:
        return self.initial_snapshot_complete and self.current_snapshot_complete

    @property
    def high_coverage(self) -> bool:
        return self.validation_count("test") > 0 and (
            self.validation_count("static") > 0 or self.validation_count("diff") > 0
        )

    @property
    def state_label(self) -> str:
        """Frozen total classifier used by the preregistered selector."""

        if (
            not self.snapshot_complete
            or not self.tool_telemetry_complete
            or self.budget_exhausted
            or self.deadline_violated
            or self.gate_decision != "pass"
            or self.terminal_reason != "model_final"
        ):
            return "root_incomplete"
        if not self.scope_valid:
            return "scope_invalid"
        if self.public_test_attempted and not self.public_test_passed:
            return "public_fail"
        return "public_unknown"


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RepositoryPilotProtocolError(f"{label} must be a non-negative integer")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepositoryPilotProtocolError(f"{label} must be a finite non-negative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise RepositoryPilotProtocolError(f"{label} must be a finite non-negative number")
    return converted


def _strict_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise RepositoryPilotProtocolError(f"{label} must be bool")
    return value


def _strict_string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise RepositoryPilotProtocolError(f"{label} must be {qualifier}")
    return value


def _required_attr(value: object, name: str, label: str) -> object:
    if not hasattr(value, name):
        raise RepositoryPilotProtocolError(f"{label} is missing required field {name}")
    return getattr(value, name)


def _validate_scope_binding(
    stage: RetainedAgentStage,
    contract: PublicScopeContract,
    verdict: PublicScopeVerdict,
) -> bool:
    if not isinstance(contract, PublicScopeContract) or not isinstance(verdict, PublicScopeVerdict):
        raise RepositoryPilotProtocolError("public state requires a host scope contract and verdict")
    if stage.fixture_id != contract.fixture_id or verdict.fixture_id != stage.fixture_id:
        raise RepositoryPilotProtocolError("public scope fixture binding is wrong")
    _require_sha256(stage.pristine_tree_sha256, "stage pristine tree")
    _require_sha256(stage.terminal_tree_sha256, "stage terminal tree")
    if stage.pristine_tree_sha256 != contract.pristine_tree_sha256:
        raise RepositoryPilotProtocolError("scope contract does not bind the stage pristine tree")
    if _scope_manifest_sha256(contract._pristine_entries) != contract.pristine_manifest_sha256:
        raise RepositoryPilotProtocolError("scope contract pristine manifest is malformed")

    digest = hashlib.sha256(b"mio.repository-public-scope-contract.v1\0")
    digest.update(contract.fixture_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(contract.pristine_tree_sha256.encode("ascii"))
    digest.update(b"\0")
    digest.update(contract.pristine_manifest_sha256.encode("ascii"))
    for name in contract.editable_names:
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
    if digest.hexdigest() != contract.contract_sha256:
        raise RepositoryPilotProtocolError("scope contract commitment is malformed")

    expected = contract.assess_terminal(
        fixture_id=stage.fixture_id,
        terminal_root=stage.workspace,
        terminal_tree_sha256=stage.terminal_tree_sha256,
    )
    if verdict != expected:
        raise RepositoryPilotProtocolError("scope verdict is stale or does not match the host contract")
    return verdict.scope_valid


def _validate_stage_policy_and_budget(stage: RetainedAgentStage) -> AgentExecutionBudget:
    if stage.stage == "direct":
        expected_budget = DIRECT_EXECUTION_BUDGET
        expected_effort = CodingEffort.MEDIUM.value
    elif stage.stage == "recovery":
        expected_budget = EXTRA_EXECUTION_BUDGET
        expected_effort = CodingEffort.HIGH.value
    else:
        raise RepositoryPilotProtocolError("public extraction received an unknown physical stage")
    if stage.execution_budget != expected_budget:
        raise RepositoryPilotProtocolError("stage execution budget differs from the frozen protocol")
    if type(stage.coding_effort) is not str or stage.coding_effort != expected_effort:
        raise RepositoryPilotProtocolError("stage effort differs from the frozen protocol")
    if stage.state.get("execution_budget") != expected_budget:
        raise RepositoryPilotProtocolError("agent state execution budget was changed")
    if stage.state.get("coding_effort") != expected_effort:
        raise RepositoryPilotProtocolError("agent state effort was changed")
    if type(stage.quality_enabled) is not bool:
        raise RepositoryPilotProtocolError("stage quality flag must be bool")
    if stage.state.get("quality_gate_enabled") is not stage.quality_enabled:
        raise RepositoryPilotProtocolError("agent state quality flag contradicts the physical stage")
    if stage.state.get("quality_gate_require_change") is not stage.quality_enabled:
        raise RepositoryPilotProtocolError("agent state net-change flag contradicts the physical stage")
    policy = stage.state.get("tool_policy")
    if not isinstance(policy, AgentToolPolicy):
        raise RepositoryPilotProtocolError("stage has no strict native tool policy")
    if policy.workspace_roots != (stage.workspace.resolve(),):
        raise RepositoryPilotProtocolError("stage policy is not bound to its exact workspace")
    if AgentToolPermission.NETWORK in policy.permissions:
        raise RepositoryPilotProtocolError("stage policy unexpectedly enables network access")
    expected_names = _QUALITY_TOOLS if stage.quality_enabled else _PLAIN_TOOLS
    registry = stage.state.get("tool_registry")
    specs = stage.state.get("tool_specs")
    if not isinstance(registry, Mapping) or tuple(registry) != expected_names:
        raise RepositoryPilotProtocolError("stage tool registry differs from the frozen tool surface")
    if type(specs) is not tuple:
        raise RepositoryPilotProtocolError("stage tool schemas must be an immutable tuple")
    spec_names = tuple(
        spec.get("function", {}).get("name") if type(spec) is dict and type(spec.get("function")) is dict else None
        for spec in specs
    )
    if spec_names != expected_names:
        raise RepositoryPilotProtocolError("stage tool schemas differ from the frozen tool surface")
    return expected_budget


def _validate_round_trace(
    trace: object,
    index: int,
    *,
    drafter_ref: str,
) -> AgentRoundTrace:
    if not isinstance(trace, AgentRoundTrace):
        raise RepositoryPilotProtocolError("round telemetry must use AgentRoundTrace")
    if _nonnegative_int(trace.round_index, "round index") != index:
        raise RepositoryPilotProtocolError("round indices must be contiguous")
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "prefill_ns",
        "decode_ns",
        "model_total_ns",
        "logical_prompt_tokens",
        "physical_prefill_tokens",
        "physical_decode_tokens",
        "warm_offset",
        "warm_offset_tokens",
    ):
        _nonnegative_int(getattr(trace, name), f"round {name}")
    for name in ("total_time_s", "prompt_tps", "generation_tps"):
        _finite_nonnegative(getattr(trace, name), f"round {name}")
    for name in (
        "generation_backend",
        "timing_source",
        "drafter_requested",
        "drafter_selected",
    ):
        _strict_string(getattr(trace, name), f"round {name}")
    if trace.drafter_ref is not None:
        _strict_string(trace.drafter_ref, "round drafter_ref")
    for name in ("fallback_ar", "phase_censored", "deadline_hit"):
        _strict_bool(getattr(trace, name), f"round {name}")
    if trace.generation_backend != "dflash":
        raise RepositoryPilotProtocolError("round generation backend is not frozen DFlash")
    if trace.fallback_ar:
        raise RepositoryPilotProtocolError("round unexpectedly used autoregressive fallback")
    if trace.drafter_requested != "dflash" or trace.drafter_selected != "dflash":
        raise RepositoryPilotProtocolError("round drafter selection is not strict DFlash")
    if trace.drafter_ref != drafter_ref:
        raise RepositoryPilotProtocolError("round DFlash reference differs from the sealed engine identity")
    if trace.timing_source != "runtime_raw_ns":
        raise RepositoryPilotProtocolError("round timing source is not exact runtime nanoseconds")
    if trace.logical_prompt_tokens != trace.prompt_tokens:
        raise RepositoryPilotProtocolError("logical prompt tokens contradict prompt tokens")
    if trace.warm_offset_tokens != trace.warm_offset or trace.warm_offset > trace.logical_prompt_tokens:
        raise RepositoryPilotProtocolError("round warm-offset telemetry is contradictory")
    if trace.physical_prefill_tokens != trace.logical_prompt_tokens - trace.warm_offset:
        raise RepositoryPilotProtocolError("physical prefill tokens contradict the warm offset")
    if trace.model_total_ns != trace.prefill_ns + trace.decode_ns:
        raise RepositoryPilotProtocolError("model total nanoseconds contradict prefill and decode")
    if trace.model_total_ns > math.ceil(trace.total_time_s * 1e9):
        raise RepositoryPilotProtocolError("model phase timing exceeds round wall time")
    if trace.physical_decode_tokens < trace.completion_tokens:
        raise RepositoryPilotProtocolError("physical decode tokens are below completion tokens")
    return trace


def _validate_tool_trace(trace: object, index: int, round_count: int) -> AgentToolTrace:
    if not isinstance(trace, AgentToolTrace):
        raise RepositoryPilotProtocolError("tool telemetry must use AgentToolTrace")
    if _nonnegative_int(trace.sequence, "tool sequence") != index:
        raise RepositoryPilotProtocolError("tool trace sequences must be contiguous")
    round_index = _nonnegative_int(trace.round_index, "tool round index")
    if round_index >= round_count:
        raise RepositoryPilotProtocolError("tool trace references an unobserved model round")
    for name in ("tool_name", "operation", "permission", "outcome"):
        _strict_string(getattr(trace, name), f"tool {name}")
    _require_sha256(trace.target_sha256, "tool target commitment")
    _require_sha256(trace.audit_sha256, "tool audit commitment")
    for name in ("duration_ns", "output_chars", "audit_count"):
        _nonnegative_int(getattr(trace, name), f"tool {name}")
    if trace.effective_timeout_ns is not None:
        timeout_ns = _nonnegative_int(trace.effective_timeout_ns, "tool effective timeout")
        if timeout_ns == 0:
            raise RepositoryPilotProtocolError("tool effective timeout must be positive")
    if trace.exit_code_or_signal is not None and (
        isinstance(trace.exit_code_or_signal, bool) or not isinstance(trace.exit_code_or_signal, (int, str))
    ):
        raise RepositoryPilotProtocolError("tool exit code or signal is malformed")
    for name in ("allowed", "timeout_enforced", "telemetry_complete", "effect_unknown"):
        _strict_bool(getattr(trace, name), f"tool {name}")
    if trace.effect_unknown and trace.telemetry_complete:
        raise RepositoryPilotProtocolError("unknown tool effect cannot have complete telemetry")
    return trace


@dataclass(frozen=True)
class _ValidatedTelemetry:
    rounds: tuple[AgentRoundTrace, ...]
    tool_events: tuple[AgentToolTrace, ...]
    terminal_reason: str
    budget_exhausted: bool
    deadline_violated: bool
    telemetry_complete: bool
    output_tokens: int
    model_seconds: float
    wall_seconds: float


def _validate_result_telemetry(
    stage: RetainedAgentStage,
    budget: AgentExecutionBudget,
) -> _ValidatedTelemetry:
    result = stage.result
    if not isinstance(result, AgentTurnResult):
        raise RepositoryPilotProtocolError("stage result must be a typed AgentTurnResult")
    rounds_value = _required_attr(result, "rounds", "agent result")
    tools_value = _required_attr(result, "tool_events", "agent result")
    if type(rounds_value) is not tuple or type(tools_value) is not tuple:
        raise RepositoryPilotProtocolError("rounds and tool events must be immutable tuples")
    drafter_ref = _strict_string(stage.drafter_ref, "stage DFlash reference")
    rounds = tuple(
        _validate_round_trace(trace, index, drafter_ref=drafter_ref) for index, trace in enumerate(rounds_value)
    )
    tools = tuple(_validate_tool_trace(trace, index, len(rounds)) for index, trace in enumerate(tools_value))
    if len(rounds) > budget.max_rounds:
        raise RepositoryPilotProtocolError("observed model rounds exceed the stage budget")
    tool_calls = _nonnegative_int(_required_attr(result, "tool_calls", "agent result"), "tool calls")
    if tool_calls != len(tools) or tool_calls > budget.max_tool_calls:
        raise RepositoryPilotProtocolError("observed tool calls contradict traces or budget")
    result_chars = _nonnegative_int(
        _required_attr(result, "tool_result_chars", "agent result"),
        "tool result characters",
    )
    if result_chars < sum(trace.output_chars for trace in tools):
        raise RepositoryPilotProtocolError("tool result character count is smaller than observed tool traces")
    output_tokens = sum(trace.completion_tokens for trace in rounds)
    reported_tokens = _nonnegative_int(
        _required_attr(result, "completion_tokens", "agent result"),
        "completion tokens",
    )
    if reported_tokens != output_tokens:
        raise RepositoryPilotProtocolError("completion-token total contradicts round traces")
    if budget.max_output_tokens is not None and output_tokens > budget.max_output_tokens:
        raise RepositoryPilotProtocolError("observed completion tokens exceed the stage budget")
    if budget.max_context_tokens is not None and any(
        trace.prompt_tokens + trace.completion_tokens > budget.max_context_tokens for trace in rounds
    ):
        raise RepositoryPilotProtocolError("observed context tokens exceed the stage budget")
    wall_seconds = _finite_nonnegative(
        _required_attr(result, "wall_time_s", "agent result"),
        "wall seconds",
    )
    # Raw prefill + decode is the validated model-phase endpoint; round time may
    # additionally include orchestration overhead and therefore is not model cost.
    model_seconds = sum(trace.model_total_ns for trace in rounds) / 1_000_000_000
    if model_seconds > wall_seconds:
        raise RepositoryPilotProtocolError("summed model seconds exceed observed stage wall time")
    terminal_reason = _strict_string(
        _required_attr(result, "terminal_reason", "agent result"),
        "terminal reason",
    )
    raw_exhaustion = _required_attr(result, "budget_exhaustion", "agent result")
    if raw_exhaustion is not None:
        _strict_string(raw_exhaustion, "budget exhaustion")
    budget_exhausted = raw_exhaustion is not None
    if terminal_reason == "model_final" and budget_exhausted:
        raise RepositoryPilotProtocolError("model_final contradicts a budget exhaustion")
    if terminal_reason in {"budget_exhausted", "budget_finalization"} and not budget_exhausted:
        raise RepositoryPilotProtocolError("budget terminal has no exhaustion evidence")
    if budget.max_wall_seconds is not None and wall_seconds >= budget.max_wall_seconds and not budget_exhausted:
        raise RepositoryPilotProtocolError("observed wall time reaches or exceeds the budget without exhaustion")
    telemetry_complete = _strict_bool(
        _required_attr(result, "tool_telemetry_complete", "agent result"),
        "tool telemetry complete",
    )
    expected_complete = len(tools) == tool_calls and all(trace.telemetry_complete for trace in tools)
    if telemetry_complete != expected_complete:
        raise RepositoryPilotProtocolError("aggregate tool telemetry contradicts tool traces")
    deadline_violated = any(trace.deadline_hit for trace in rounds) or any(
        trace.outcome == "timeout" for trace in tools
    )
    tool_timeout = any(trace.outcome == "timeout" for trace in tools)
    if (terminal_reason == "tool_timeout") != tool_timeout:
        raise RepositoryPilotProtocolError("tool-timeout terminal contradicts tool traces")
    if any(trace.deadline_hit for trace in rounds) and not budget_exhausted:
        raise RepositoryPilotProtocolError("generation deadline evidence lacks budget exhaustion")
    return _ValidatedTelemetry(
        rounds=rounds,
        tool_events=tools,
        terminal_reason=terminal_reason,
        budget_exhausted=budget_exhausted,
        deadline_violated=deadline_violated,
        telemetry_complete=telemetry_complete,
        output_tokens=output_tokens,
        model_seconds=model_seconds,
        wall_seconds=wall_seconds,
    )


def _validate_evidence(evidence: object) -> ValidationEvidence:
    if not isinstance(evidence, ValidationEvidence):
        raise RepositoryPilotProtocolError("trusted validation ledger contains an untyped entry")
    if not isinstance(evidence.kind, ValidationKind):
        raise RepositoryPilotProtocolError("trusted validation kind is malformed")
    _nonnegative_int(evidence.epoch, "trusted validation epoch")
    _require_sha256(evidence.revision_sha256, "trusted validation revision")
    _require_sha256(evidence.command_sha256, "trusted validation command")
    _strict_bool(evidence.allowed, "trusted validation allowed")
    _strict_string(evidence.outcome, "trusted validation outcome")
    return evidence


def _validate_gate_and_derive_visible_test(
    stage: RetainedAgentStage,
    telemetry: _ValidatedTelemetry,
) -> tuple[dict[str, object], dict[str, int], int, VisiblePublicTestResult]:
    gate = stage.trusted_quality_gate
    if not isinstance(gate, CodingQualityGate):
        raise RepositoryPilotProtocolError("Quality stage has no retained trusted gate")
    if gate.roots != (stage.workspace.resolve(),):
        raise RepositoryPilotProtocolError("trusted gate roots do not bind the terminal workspace")
    expected_effort = CodingEffort(stage.coding_effort)
    if gate.effort is not expected_effort or gate.enabled is not True:
        raise RepositoryPilotProtocolError("trusted gate effort or enabled flag is malformed")
    if gate.require_net_workspace_change is not True:
        raise RepositoryPilotProtocolError("trusted gate lost its net-change contract")
    if gate.initial_snapshot is None or gate.current_snapshot is None:
        raise RepositoryPilotProtocolError("trusted gate snapshots are missing")
    if (
        not gate.initial_snapshot.complete
        or gate.initial_snapshot.content_sha256 != stage.pristine_snapshot.content_sha256
    ):
        raise RepositoryPilotProtocolError("trusted gate initial snapshot is not pristine")
    if (
        not gate.current_snapshot.complete
        or gate.current_snapshot.content_sha256 != stage.current_snapshot.content_sha256
        or gate.current_snapshot.revision_sha256 != stage.current_snapshot.revision_sha256
    ):
        raise RepositoryPilotProtocolError("trusted gate current snapshot does not bind the terminal")

    raw_report = _required_attr(stage.result, "quality_gate", "agent result")
    if type(raw_report) is not dict:
        raise RepositoryPilotProtocolError("Quality result must contain a strict gate report")
    report: dict[str, object] = raw_report
    expected_report = gate.report()
    if report != expected_report:
        raise RepositoryPilotProtocolError("serialized Quality report contradicts the retained gate")
    if report.get("schema") != "mio.coding-quality-gate.v3":
        raise RepositoryPilotProtocolError("Quality report schema is not frozen")
    decision = _strict_string(report.get("decision"), "gate decision")
    if decision not in _GATE_DECISIONS:
        raise RepositoryPilotProtocolError("gate decision is not allowlisted")
    _strict_string(report.get("phase"), "gate phase")
    _strict_bool(report.get("satisfied"), "gate satisfied")
    _strict_bool(report.get("initial_snapshot_complete"), "gate initial snapshot complete")
    _strict_bool(report.get("snapshot_complete"), "gate current snapshot complete")
    if report.get("current_content_sha256") != stage.current_snapshot.content_sha256:
        raise RepositoryPilotProtocolError("gate current content hash does not equal the terminal snapshot")
    if report.get("current_revision_sha256") != stage.current_snapshot.revision_sha256:
        raise RepositoryPilotProtocolError("gate current revision does not equal the terminal snapshot")
    mutation_epoch = _nonnegative_int(report.get("mutation_epoch"), "mutation epoch")
    changed_kinds = report.get("changed_kinds")
    if type(changed_kinds) is not list or any(type(value) is not str for value in changed_kinds):
        raise RepositoryPilotProtocolError("gate changed_kinds must be a string list")
    if len(changed_kinds) != len(set(changed_kinds)) or any(value not in _CHANGED_KINDS for value in changed_kinds):
        raise RepositoryPilotProtocolError("gate changed_kinds is malformed")
    net_changed = stage.pristine_snapshot.content_sha256 != stage.current_snapshot.content_sha256
    if (mutation_epoch == 0) != (len(changed_kinds) == 0) or (net_changed and mutation_epoch == 0):
        raise RepositoryPilotProtocolError("gate epoch, changed kinds, and terminal change contradict")

    evidence = tuple(_validate_evidence(item) for item in gate.validations)
    current = tuple(
        item
        for item in evidence
        if item.epoch == mutation_epoch and item.revision_sha256 == stage.current_snapshot.revision_sha256
    )
    successful = tuple(item for item in current if item.allowed and item.outcome == "ok")
    counts = {kind.value: sum(item.kind is kind for item in successful) for kind in ValidationKind}
    raw_counts = report.get("validation_counts")
    if type(raw_counts) is not dict or tuple(raw_counts) != _VALIDATION_KINDS:
        raise RepositoryPilotProtocolError("gate validation counts have the wrong exact names or order")
    reported_counts = {
        name: _nonnegative_int(raw_counts[name], f"validation count {name}") for name in _VALIDATION_KINDS
    }
    if reported_counts != counts:
        raise RepositoryPilotProtocolError("gate validation counts contradict trusted current-revision evidence")
    for name, expected in (
        ("validation_attempts", len(evidence)),
        ("recognized_validation_attempts", len(evidence)),
        ("validate_invocations", gate.validate_invocations),
        ("misrouted_validation_commands", gate.misrouted_validation_commands),
        ("successful_reads", gate.successful_reads),
    ):
        if _nonnegative_int(report.get(name), f"gate {name}") != expected:
            raise RepositoryPilotProtocolError(f"gate {name} contradicts retained evidence")
    observed_validate_calls = sum(item.tool_name == "validate" for item in telemetry.tool_events)
    if observed_validate_calls != gate.validate_invocations:
        raise RepositoryPilotProtocolError("gate validate invocation count contradicts tool telemetry")
    validation_traces = tuple(item for item in telemetry.tool_events if item.tool_name == "validate")
    trace_cursor = 0
    for item in evidence:
        while trace_cursor < len(validation_traces) and (
            validation_traces[trace_cursor].allowed != item.allowed
            or validation_traces[trace_cursor].outcome != item.outcome
        ):
            trace_cursor += 1
        if trace_cursor == len(validation_traces):
            raise RepositoryPilotProtocolError("trusted validation evidence contradicts validate tool traces")
        trace_cursor += 1

    visible_attempts = tuple(item for item in current if item.kind in {ValidationKind.TEST, ValidationKind.BUILD})
    if not visible_attempts:
        visible = VisiblePublicTestResult(False, False, "not_run")
    elif all(item.allowed and item.outcome == "ok" for item in visible_attempts):
        visible = VisiblePublicTestResult(True, True, "passed")
    else:
        error = any(not item.allowed or item.outcome in {"error", "timeout", "denied"} for item in visible_attempts)
        visible = VisiblePublicTestResult(True, False, "error" if error else "failed")
    return report, reported_counts, len(visible_attempts), visible


def extract_public_repository_state(
    stage: RetainedAgentStage,
    *,
    scope_contract: PublicScopeContract,
    scope_verdict: PublicScopeVerdict,
) -> PublicRepositoryState:
    """Extract strict host-attested public evidence with no caller summary."""

    if not isinstance(stage, RetainedAgentStage):
        raise TypeError("stage must be RetainedAgentStage")
    budget = _validate_stage_policy_and_budget(stage)
    scope_valid = _validate_scope_binding(stage, scope_contract, scope_verdict)
    live_snapshot = snapshot_workspaces((stage.workspace,))
    if (
        not live_snapshot.complete
        or not stage.pristine_snapshot.complete
        or not stage.current_snapshot.complete
        or live_snapshot.content_sha256 != stage.current_snapshot.content_sha256
        or live_snapshot.revision_sha256 != stage.current_snapshot.revision_sha256
    ):
        raise RepositoryPilotProtocolError("stage snapshot is incomplete or stale")
    telemetry = _validate_result_telemetry(stage, budget)
    if stage.quality_enabled:
        report, counts, test_or_build_attempts, public_test = _validate_gate_and_derive_visible_test(
            stage,
            telemetry,
        )
        gate_present = True
        gate_decision = _strict_string(report["decision"], "gate decision")
        gate_phase = _strict_string(report["phase"], "gate phase")
        gate_satisfied = _strict_bool(report["satisfied"], "gate satisfied")
        mutation_epoch = _nonnegative_int(report["mutation_epoch"], "mutation epoch")
    else:
        if stage.trusted_quality_gate is not None:
            raise RepositoryPilotProtocolError("Plain stage unexpectedly retained a Quality gate")
        if _required_attr(stage.result, "quality_gate", "Plain result") is not None:
            raise RepositoryPilotProtocolError("Plain stage unexpectedly serialized a Quality report")
        if any(event.tool_name == "validate" for event in telemetry.tool_events):
            raise RepositoryPilotProtocolError("Plain stage executed a forbidden validate tool")
        counts = {name: 0 for name in _VALIDATION_KINDS}
        test_or_build_attempts = 0
        public_test = VisiblePublicTestResult(False, False, "not_run")
        gate_present = False
        gate_decision = "not_applicable"
        gate_phase = "experiment_disabled"
        gate_satisfied = True
        mutation_epoch = 0
    return PublicRepositoryState(
        scope_valid=scope_valid,
        public_test_attempted=public_test.attempted,
        public_test_passed=public_test.passed,
        public_test_status=public_test.status,
        gate_present=gate_present,
        gate_decision=gate_decision,
        gate_phase=gate_phase,
        gate_satisfied=gate_satisfied,
        initial_snapshot_complete=True,
        current_snapshot_complete=True,
        net_workspace_changed=stage.pristine_snapshot.content_sha256 != stage.current_snapshot.content_sha256,
        mutation_epoch=mutation_epoch,
        trusted_test_or_build_attempt_count=test_or_build_attempts,
        validation_counts=tuple((name, counts[name]) for name in _VALIDATION_KINDS),
        terminal_reason=telemetry.terminal_reason,
        budget_exhausted=telemetry.budget_exhausted,
        deadline_violated=telemetry.deadline_violated,
        tool_telemetry_complete=telemetry.telemetry_complete,
        round_count=len(telemetry.rounds),
        tool_calls=len(telemetry.tool_events),
        output_tokens=telemetry.output_tokens,
        model_seconds=telemetry.model_seconds,
        wall_seconds=telemetry.wall_seconds,
    )


def to_protocol_public_evidence(state: PublicRepositoryState) -> ProtocolPublicEvidence:
    """Bridge one strict adapter state into the pure four-arm protocol."""

    if not isinstance(state, PublicRepositoryState):
        raise TypeError("state must be PublicRepositoryState")
    visible = {
        "not_run": VisibleCheckOutcome.NOT_RUN,
        "passed": VisibleCheckOutcome.PASS,
        "failed": VisibleCheckOutcome.FAIL,
        "error": VisibleCheckOutcome.FAIL,
    }[state.public_test_status]
    return ProtocolPublicEvidence(
        initial_snapshot_complete=state.initial_snapshot_complete,
        current_snapshot_complete=state.current_snapshot_complete,
        tool_telemetry_complete=state.tool_telemetry_complete,
        budget_exhausted=state.budget_exhausted,
        deadline_violated=state.deadline_violated,
        quality_decision=state.gate_decision,
        terminal_reason=state.terminal_reason,
        scope_valid=state.scope_valid,
        visible_check=visible,
        mutation_epoch=state.mutation_epoch,
        trusted_test_or_build_attempt_count=state.trusted_test_or_build_attempt_count,
        trusted_test_count=state.validation_count("test"),
        trusted_build_count=state.validation_count("build"),
        trusted_static_count=state.validation_count("static"),
        trusted_diff_count=state.validation_count("diff"),
    )


def to_protocol_candidate_cost(state: PublicRepositoryState) -> ProtocolCandidateCost:
    """Bridge raw model-phase and wall costs without normalization or imputation."""

    if not isinstance(state, PublicRepositoryState):
        raise TypeError("state must be PublicRepositoryState")
    return ProtocolCandidateCost(
        model_rounds=state.round_count,
        tool_calls=state.tool_calls,
        output_tokens=state.output_tokens,
        model_seconds=state.model_seconds,
        wall_seconds=state.wall_seconds,
    )


def prefer_recovery_publicly(direct: PublicRepositoryState, recovery: PublicRepositoryState) -> bool:
    """Choose a child only for a strict, hidden-free public-evidence gain."""

    state_rank = {
        "scope_invalid": 0,
        "root_incomplete": 1,
        "public_fail": 2,
        "public_unknown": 3,
    }

    def rank(value: PublicRepositoryState) -> tuple[int, ...]:
        return (
            state_rank[value.state_label],
            int(value.gate_decision == "pass"),
            int(value.terminal_reason == "model_final"),
            int(value.validation_count("test") > 0 or value.validation_count("build") > 0),
            int(value.validation_count("static") > 0 or value.validation_count("diff") > 0),
        )

    recovery_admissible = bool(
        recovery.scope_valid
        and recovery.tool_telemetry_complete
        and recovery.snapshot_complete
        and not recovery.budget_exhausted
        and not recovery.deadline_violated
    )
    return recovery_admissible and rank(recovery) > rank(direct)


@dataclass(frozen=True)
class TerminalWorkspace:
    key: str
    fixture_id: str
    root: Path
    tree_sha256: str
    public_state: PublicRepositoryState
    observation: CandidateObservation

    def __post_init__(self) -> None:
        if type(self.observation) is not CandidateObservation:
            raise TypeError("terminal workspace requires an exact CandidateObservation")
        if self.observation.terminal_artifact_id != self.tree_sha256:
            raise RepositoryPilotProtocolError("terminal observation digest differs from its workspace")
        if self.observation.public_evidence != to_protocol_public_evidence(self.public_state):
            raise RepositoryPilotProtocolError("terminal observation evidence differs from its public state")
        if self.observation.cost != to_protocol_candidate_cost(self.public_state):
            raise RepositoryPilotProtocolError("terminal observation cost differs from its public state")

    @property
    def physical_evaluation_key(self) -> tuple[str, str]:
        """Private dedup identity: exact fixture plus exact terminal bytes."""

        return (self.fixture_id, self.tree_sha256)


def logical_terminal_key(fixture_id: str, arm: LogicalArm) -> str:
    """Return the one canonical host-only key for a fixture/logical arm."""

    if type(fixture_id) is not str or not fixture_id:
        raise ValueError("fixture_id must be a non-empty string")
    if not isinstance(arm, LogicalArm):
        raise TypeError("arm must be LogicalArm")
    return json.dumps(
        {"arm": arm.value, "fixture_id": fixture_id},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class SelectedFixtureCandidates:
    """Hidden-free selected candidates awaiting barrier-owned outcomes."""

    fixture_id: str
    plain_root: CandidateObservation
    quality_root: CandidateObservation
    static_child: CandidateObservation | None
    markov_child: CandidateObservation | None
    static_selection: CandidateChoice
    markov_selection: CandidateChoice

    def __post_init__(self) -> None:
        if type(self.fixture_id) is not str or not self.fixture_id:
            raise ValueError("fixture_id must be a non-empty string")
        if not isinstance(self.plain_root, CandidateObservation) or not isinstance(
            self.quality_root,
            CandidateObservation,
        ):
            raise TypeError("selected roots must be CandidateObservation values")
        for child in (self.static_child, self.markov_child):
            if child is not None and not isinstance(child, CandidateObservation):
                raise TypeError("selected child must be CandidateObservation or None")
        if self.static_selection is not select_candidate(self.quality_root, self.static_child):
            raise RepositoryPilotProtocolError("Static selection contradicts the frozen selector")
        if self.markov_selection is not select_candidate(self.quality_root, self.markov_child):
            raise RepositoryPilotProtocolError("Markov selection contradicts the frozen selector")

    def candidate_for_arm(self, arm: LogicalArm) -> CandidateObservation:
        if arm is LogicalArm.PLAIN:
            return self.plain_root
        if arm is LogicalArm.QUALITY:
            return self.quality_root
        if arm is LogicalArm.QUALITY_STATIC_EXTRA:
            return self.static_child if self.static_selection is CandidateChoice.CHILD else self.quality_root
        if arm is LogicalArm.MARKOV_QUALITY:
            return self.markov_child if self.markov_selection is CandidateChoice.CHILD else self.quality_root
        raise ValueError("unknown logical arm")


@dataclass(frozen=True)
class HiddenEvaluationBatch:
    """Strict hidden verdicts plus their protocol-integrity receipt."""

    results: Mapping[str, HiddenOutcome]
    terminal_bindings: Mapping[str, TerminalWorkspace]
    receipt: EvaluationBarrierReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.results, Mapping) or any(
            type(key) is not str or type(value) is not HiddenOutcome for key, value in self.results.items()
        ):
            raise TypeError("hidden evaluation results must map strings to exact HiddenOutcome values")
        if not isinstance(self.terminal_bindings, Mapping) or any(
            type(key) is not str or not isinstance(value, TerminalWorkspace)
            for key, value in self.terminal_bindings.items()
        ):
            raise TypeError("hidden evaluation terminal bindings are malformed")
        if tuple(self.results) != tuple(self.terminal_bindings):
            raise RepositoryPilotProtocolError("hidden results and terminal bindings use different logical keys")
        if not isinstance(self.receipt, EvaluationBarrierReceipt):
            raise TypeError("hidden evaluation batch requires an EvaluationBarrierReceipt")
        if self.receipt.expected_logical_selection_count != len(self.results):
            raise RepositoryPilotProtocolError("hidden batch result count contradicts its receipt")
        physical: dict[tuple[str, str], HiddenOutcome] = {}
        for key, terminal in self.terminal_bindings.items():
            previous = physical.setdefault(terminal.physical_evaluation_key, self.results[key])
            if previous != self.results[key]:
                raise RepositoryPilotProtocolError("one physical terminal has conflicting hidden outcomes")
        if len(physical) != self.receipt.unique_terminal_artifact_count:
            raise RepositoryPilotProtocolError("hidden batch physical count contradicts its receipt")


class HiddenEvaluationBarrier:
    """Make the hidden callback unreachable until all selections are sealed."""

    def __init__(self, expected_keys: Sequence[str]) -> None:
        expected = tuple(expected_keys)
        if not expected or any(not isinstance(key, str) or not key for key in expected):
            raise ValueError("selection barrier keys must be non-empty strings")
        if len(expected) != len(set(expected)):
            raise ValueError("selection barrier keys must be unique")
        self._expected = expected
        self._registered: dict[str, TerminalWorkspace] = {}
        self._sealed = False
        self._evaluated = False
        self._all_generation_complete_before_seal = False
        self._generation_receipt: GenerationCompletionReceipt | None = None
        self._hidden_call_count = 0
        self._receipt: EvaluationBarrierReceipt | None = None
        self._batch: HiddenEvaluationBatch | None = None

    @classmethod
    def for_fixtures(cls, fixture_ids: Sequence[str]) -> HiddenEvaluationBarrier:
        fixtures = tuple(fixture_ids)
        if not fixtures or any(type(fixture_id) is not str or not fixture_id for fixture_id in fixtures):
            raise ValueError("fixture IDs must be non-empty strings")
        if len(fixtures) != len(set(fixtures)):
            raise ValueError("fixture IDs must be unique")
        return cls(tuple(logical_terminal_key(fixture_id, arm) for fixture_id in fixtures for arm in LOGICAL_ARMS))

    def register(
        self,
        key: str,
        root: Path,
        public_state: PublicRepositoryState,
        *,
        fixture_id: str,
        observation: CandidateObservation,
    ) -> None:
        if self._sealed or self._evaluated:
            raise RepositoryPilotProtocolError("terminal selection barrier is already closed")
        if key not in self._expected or key in self._registered:
            raise RepositoryPilotProtocolError("terminal selection key is unexpected or duplicated")
        if not isinstance(public_state, PublicRepositoryState):
            raise RepositoryPilotProtocolError("terminal selection requires a typed public state")
        if type(observation) is not CandidateObservation:
            raise RepositoryPilotProtocolError("terminal selection requires an exact CandidateObservation")
        if type(fixture_id) is not str or not fixture_id:
            raise RepositoryPilotProtocolError("physical fixture id must be a non-empty string")
        root_path = Path(root).expanduser()
        tree_sha256 = regular_tree_sha256(root_path)
        canonical = root_path.resolve(strict=True)
        self._registered[key] = TerminalWorkspace(
            key=key,
            fixture_id=fixture_id,
            root=canonical,
            tree_sha256=tree_sha256,
            public_state=public_state,
            observation=observation,
        )

    def seal(
        self,
        *,
        generation_receipt: GenerationCompletionReceipt,
    ) -> Mapping[str, TerminalWorkspace]:
        if self._sealed:
            raise RepositoryPilotProtocolError("terminal selection barrier is already sealed")
        if type(generation_receipt) is not GenerationCompletionReceipt:
            raise RepositoryPilotProtocolError("selection seal requires an exact GenerationCompletionReceipt")
        if len(self._expected) != generation_receipt.fixture_count * 4:
            raise RepositoryPilotProtocolError("logical selection count is not exactly four per fixture")
        fixture_ids = tuple(dict.fromkeys(terminal.fixture_id for terminal in self._registered.values()))
        if len(fixture_ids) != generation_receipt.fixture_count:
            raise RepositoryPilotProtocolError("registered terminal fixtures contradict the generation receipt")
        canonical_keys = {logical_terminal_key(fixture_id, arm) for fixture_id in fixture_ids for arm in LOGICAL_ARMS}
        if set(self._expected) != canonical_keys:
            raise RepositoryPilotProtocolError("expected logical keys are not the canonical four arms per fixture")
        if tuple(sorted(self._registered)) != tuple(sorted(self._expected)):
            raise RepositoryPilotProtocolError("all terminal workspaces must be selected before sealing")
        for terminal in self._registered.values():
            if regular_tree_sha256(terminal.root) != terminal.tree_sha256:
                raise RepositoryPilotProtocolError("terminal workspace changed before the selection seal")
        self._generation_receipt = generation_receipt
        self._all_generation_complete_before_seal = True
        self._sealed = True
        return MappingProxyType(dict(self._registered))

    def evaluate(
        self,
        hidden_evaluator: Callable[[str, Path], HiddenOutcome],
    ) -> HiddenEvaluationBatch:
        if not self._sealed:
            raise RepositoryPilotProtocolError("hidden evaluation is blocked until selection is sealed")
        if self._evaluated:
            raise RepositoryPilotProtocolError("hidden evaluation barrier is single-use")
        if not callable(hidden_evaluator):
            raise TypeError("hidden evaluator must be callable")
        for terminal in self._registered.values():
            if regular_tree_sha256(terminal.root) != terminal.tree_sha256:
                raise RepositoryPilotProtocolError("terminal workspace changed after the selection seal")
        self._evaluated = True
        physical_results: dict[tuple[str, str], HiddenOutcome] = {}
        for key in self._expected:
            terminal = self._registered[key]
            if terminal.physical_evaluation_key not in physical_results:
                if regular_tree_sha256(terminal.root) != terminal.tree_sha256:
                    raise RepositoryPilotProtocolError(
                        "terminal workspace changed immediately before hidden evaluation"
                    )
                try:
                    outcome = hidden_evaluator(
                        terminal.fixture_id,
                        terminal.root,
                    )
                finally:
                    if regular_tree_sha256(terminal.root) != terminal.tree_sha256:
                        raise RepositoryPilotProtocolError("hidden evaluator mutated its sealed terminal workspace")
                self._hidden_call_count += 1
                if type(outcome) is not HiddenOutcome:
                    raise RepositoryPilotProtocolError("hidden evaluator must return an exact HiddenOutcome")
                physical_results[terminal.physical_evaluation_key] = outcome
        for terminal in self._registered.values():
            if regular_tree_sha256(terminal.root) != terminal.tree_sha256:
                raise RepositoryPilotProtocolError("terminal workspace changed during hidden evaluation")
        logical_results = {
            key: physical_results[self._registered[key].physical_evaluation_key] for key in self._expected
        }
        receipt = EvaluationBarrierReceipt(
            expected_logical_selection_count=len(self._expected),
            registered_logical_selection_count=len(self._registered),
            unique_terminal_artifact_count=self.physical_evaluation_count,
            hidden_evaluation_count=self._hidden_call_count,
            all_generation_complete_before_seal=self._all_generation_complete_before_seal,
            selection_sealed_before_hidden=True,
            hidden_evaluation_single_use=self._evaluated,
        )
        self._receipt = receipt
        batch = HiddenEvaluationBatch(
            MappingProxyType(logical_results),
            MappingProxyType(dict(self._registered)),
            receipt,
        )
        self._batch = batch
        return batch

    def bind_fixture_records(
        self,
        candidates: Sequence[SelectedFixtureCandidates],
        batch: HiddenEvaluationBatch,
    ) -> tuple[FixturePilotRecord, ...]:
        """Bind pure records only to this barrier's exact selected outcomes."""

        if batch is not self._batch or self._batch is None:
            raise RepositoryPilotProtocolError("hidden batch was not emitted by this barrier")
        selected = tuple(candidates)
        if not selected or any(not isinstance(item, SelectedFixtureCandidates) for item in selected):
            raise TypeError("candidates must be a non-empty SelectedFixtureCandidates sequence")
        if len(selected) != batch.receipt.expected_logical_selection_count // len(LOGICAL_ARMS):
            raise RepositoryPilotProtocolError("selected fixture count contradicts the hidden receipt")
        if len({item.fixture_id for item in selected}) != len(selected):
            raise RepositoryPilotProtocolError("selected fixture candidates are duplicated")
        fixture_order = tuple(dict.fromkeys(self._registered[key].fixture_id for key in self._expected))
        if tuple(item.fixture_id for item in selected) != fixture_order:
            raise RepositoryPilotProtocolError("selected fixture candidates are out of barrier order")
        expected_keys = tuple(logical_terminal_key(item.fixture_id, arm) for item in selected for arm in LOGICAL_ARMS)
        if set(expected_keys) != set(batch.results) or set(expected_keys) != set(batch.terminal_bindings):
            raise RepositoryPilotProtocolError("selected candidates do not cover the exact hidden logical keys")

        records: list[FixturePilotRecord] = []
        for item in selected:
            outcomes: list[ArmHiddenOutcome] = []
            for arm in LOGICAL_ARMS:
                key = logical_terminal_key(item.fixture_id, arm)
                candidate = item.candidate_for_arm(arm)
                terminal = batch.terminal_bindings[key]
                if regular_tree_sha256(terminal.root) != terminal.tree_sha256:
                    raise RepositoryPilotProtocolError("hidden terminal changed before record binding")
                if terminal.fixture_id != item.fixture_id:
                    raise RepositoryPilotProtocolError("hidden terminal is bound to the wrong fixture")
                if type(candidate) is not CandidateObservation or candidate != terminal.observation:
                    raise RepositoryPilotProtocolError(
                        "selected candidate observation differs from the sealed terminal binding"
                    )
                outcomes.append(
                    ArmHiddenOutcome(
                        arm=arm,
                        outcome=batch.results[key],
                        trajectory_valid=candidate.public_evidence.trajectory_valid(
                            quality_derived=arm is not LogicalArm.PLAIN,
                        ),
                    )
                )
            records.append(
                FixturePilotRecord(
                    fixture_id=item.fixture_id,
                    plain_root=item.plain_root,
                    quality_root=item.quality_root,
                    static_child=item.static_child,
                    markov_child=item.markov_child,
                    static_selection=item.static_selection,
                    markov_selection=item.markov_selection,
                    outcomes=tuple(outcomes),
                )
            )
        return tuple(records)

    @property
    def logical_selection_count(self) -> int:
        return len(self._registered)

    @property
    def physical_evaluation_count(self) -> int:
        return len({terminal.physical_evaluation_key for terminal in self._registered.values()})

    @property
    def receipt(self) -> EvaluationBarrierReceipt:
        if self._receipt is None:
            raise RepositoryPilotProtocolError("evaluation receipt is unavailable before successful hidden evaluation")
        return self._receipt


def public_state_json(state: PublicRepositoryState) -> str:
    """Serialize the allowlisted state without prompts, paths, hashes, or labels."""

    if not isinstance(state, PublicRepositoryState):
        raise TypeError("state must be PublicRepositoryState")
    payload = {
        "quality_decision": state.gate_decision,
        "quality_phase": state.gate_phase,
        "terminal_reason": state.terminal_reason,
        "budget_exhaustion_present": state.budget_exhausted,
        "deadline_violation_present": state.deadline_violated,
        "initial_snapshot_complete": state.initial_snapshot_complete,
        "current_snapshot_complete": state.current_snapshot_complete,
        "net_content_change_present": state.net_workspace_changed,
        "mutation_epoch": state.mutation_epoch,
        "trusted_test_or_build_attempt_count": state.trusted_test_or_build_attempt_count,
        "trusted_test_count": state.validation_count("test"),
        "trusted_build_count": state.validation_count("build"),
        "trusted_static_count": state.validation_count("static"),
        "trusted_diff_count": state.validation_count("diff"),
        "tool_telemetry_complete": state.tool_telemetry_complete,
        "visible_test_pass": (state.public_test_passed if state.public_test_attempted else None),
        "scope_valid": state.scope_valid,
        "round_count": state.round_count,
        "tool_call_count": state.tool_calls,
        "output_token_count": state.output_tokens,
        "model_seconds": state.model_seconds,
        "wall_seconds": state.wall_seconds,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
