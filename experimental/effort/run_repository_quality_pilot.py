"""Fail-closed orchestration for the repository-quality four-arm pilot.

The core remains model-loader agnostic for deterministic testing, while the
native CLI below binds the committed source, preregistration, exact model
snapshots, package/runtime identity, and host before loading the retained Mio
executor.  Unit tests exercise that path without running MLX inference.

The only path from hidden evaluator outputs to ``FixturePilotRecord`` values is
``bind_records_from_hidden_batch``.  Callers cannot supply generation receipts,
terminal outcomes, or pre-built records to the end-to-end API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from experimental.effort.bench_repository_quality_pilot import (
    HiddenEvaluationBarrier,
    HiddenEvaluationBatch,
    ImmutableWorkspaceArchive,
    PublicRepositoryState,
    PublicScopeContract,
    RepositoryPilotProtocolError,
    RetainedAgentStage,
    RetainedNativeAgentExecutor,
    SelectedFixtureCandidates,
    extract_public_repository_state,
    logical_terminal_key,
    prepare_pristine_direct_roots,
    regular_tree_sha256,
    to_protocol_candidate_cost,
    to_protocol_public_evidence,
)
from experimental.effort.repository_quality_pilot import (
    FROZEN_SEED,
    LOGICAL_ARMS,
    CandidateObservation,
    ExtraAllocation,
    ExtraScheduledRun,
    FixturePilotRecord,
    GenerationCompletionReceipt,
    HiddenOutcome,
    LogicalArm,
    PhysicalRoot,
    PilotAggregate,
    PilotProtocol,
    RootScheduledRun,
    allocate_extras,
    build_aggregate,
    make_extra_schedule,
    make_root_schedule,
    select_candidate,
    serialize_source_free_aggregate,
)
from scripts.bench_coding_quality import (
    EvaluationRequest,
    HiddenEvaluation,
    materialize_public_fixture,
)
from scripts.run_coding_quality_benchmark import (
    ALL_PROTOCOL_SHA256,
    DEVELOPMENT_PROTOCOL_SHA256,
    DRAFT_CONTENT_IDENTITY,
    DRAFT_REPOSITORY_LABEL,
    FROZEN_CONTEXT_WINDOW,
    FROZEN_MAX_OUTPUT_TOKENS,
    GATE_PROFILE_SCHEMA,
    GATE_PROFILE_SHA256,
    SMOKE_PROTOCOL_SHA256,
    TARGET_CONTENT_IDENTITY,
    TARGET_REPOSITORY_LABEL,
    CleanSourceLock,
    CorpusCase,
    CorpusHiddenEvaluator,
    LocalModelLock,
    RuntimeIdentity,
    _assert_frozen_environment,
    _assert_gate_profile_seal,
    _assert_source_free_artifact,
    _load_native_executor,
    bind_frozen_local_models,
    capture_clean_source_lock,
    collect_runtime_identity,
    sealed_protocol_sha256,
    sealed_suite_sha256,
    select_cases,
    validate_output_path,
    verify_clean_source_lock,
    verify_frozen_local_models,
    verify_runtime_identity,
)


RESULT_ENVELOPE_SCHEMA = "mio.repository-quality-four-arm-result-envelope.v3"
ABORT_ENVELOPE_SCHEMA = "mio.repository-quality-four-arm-abort-envelope.v3"
ATTEMPT_START_SCHEMA = "mio.repository-quality-four-arm-attempt-start.v3"
SOURCE_LOCK_SCHEMA = "mio.repository-quality-source-lock.v1"
PREREGISTRATION_SCHEMA = "mio.repository-quality-four-arm-preregistration.v3"
PREREGISTRATION_REVISION = "mio-repository-quality-four-arm-pilot-v3"
PREREGISTRATION_SHA256 = "d3ddbfa29bc99f2b480797fadf6686cbc200f973e6fa6325805855494d600d3d"
PREREGISTRATION_RELATIVE_PATH = "benchmarks/repository-quality-four-arm-preregistration-v3.json"
PREDECESSOR_PREREGISTRATION_SCHEMA = "mio.repository-quality-four-arm-preregistration.v2"
PREDECESSOR_PREREGISTRATION_REVISION = "mio-repository-quality-four-arm-pilot-v2"
PREDECESSOR_PREREGISTRATION_SHA256 = "9192463d8afa08a23296e9079291dd0dfcf52910a21eebf8ad2292ddaec69610"
PREDECESSOR_PREREGISTRATION_RELATIVE_PATH = "benchmarks/repository-quality-four-arm-preregistration-v2.json"
V2_INCIDENT_SCHEMA = "mio.repository-quality-four-arm-incident-record.v1"
V2_INCIDENT_STATUS = "post_hoc_incident_record_no_result"
V2_INCIDENT_SHA256 = "4e32325739bbcd35554bb43bc5bdddb205ecbe5453cad8d47cec24b876eac157"
V2_INCIDENT_RELATIVE_PATH = "benchmarks/incidents/repository-quality-four-arm-v2-smoke-aborted-8bf6e6e.json"
PILOT_SOURCE_LOCK_FILES = (
    "pyproject.toml",
    "uv.lock",
    "mio/__init__.py",
    "mio/agent.py",
    "mio/agent_policy.py",
    "mio/coding_quality.py",
    "mio/config.py",
    "mio/drafter_selection.py",
    "mio/engine.py",
    "mio/model_manager.py",
    "mio/prompt_policy.py",
    "mio/tool_calls.py",
    "mio/dflash/__init__.py",
    "mio/dflash/kernels.py",
    "mio/dflash/model.py",
    "mio/dflash/recurrent_rollback_cache.py",
    "mio/dflash/runtime.py",
    "mio/dflash/verify_linear.py",
    "scripts/bench_coding_quality.py",
    "scripts/run_coding_quality_benchmark.py",
    "experimental/effort/model_identity.py",
    "experimental/effort/repository_quality_pilot.py",
    "experimental/effort/bench_repository_quality_pilot.py",
    "experimental/effort/run_repository_quality_pilot.py",
    PREDECESSOR_PREREGISTRATION_RELATIVE_PATH,
    V2_INCIDENT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    "docs/22-markov-quality-pilot.md",
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SAFE_TIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_PRIVATE_PROTOCOL_SEALS = {
    "smoke": SMOKE_PROTOCOL_SHA256,
    "development": DEVELOPMENT_PROTOCOL_SHA256,
    "all": ALL_PROTOCOL_SHA256,
}


class PilotRunPhase(StrEnum):
    CREATED = "created"
    PRECHECKED = "prechecked"
    ROOT_SCHEDULE_SEALED = "root_schedule_sealed"
    ROOTS_COMPLETE = "roots_complete"
    ALLOCATION_SEALED = "allocation_sealed"
    EXTRAS_COMPLETE = "extras_complete"
    SELECTIONS_SEALED = "selections_sealed"
    PRE_HIDDEN_VERIFIED = "pre_hidden_verified"
    HIDDEN_COMPLETE = "hidden_complete"
    POST_HIDDEN_VERIFIED = "post_hidden_verified"
    COMPLETE = "complete"
    ABORTED = "aborted"


class NativeAbortReason(StrEnum):
    DFLASH_RAW_PHASE_TELEMETRY_MISSING = "dflash_raw_phase_telemetry_missing"
    MODEL_LOAD_FAILURE = "model_load_failure"
    PROTOCOL_FAILURE = "protocol_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    MANAGER_UNLOAD_FAILURE = "manager_unload_failure"


class NativeManagerState(StrEnum):
    NEVER_LOADED = "never_loaded"
    LOAD_FAILED = "load_failed"
    LOADED = "loaded"
    UNLOADED = "unloaded"
    UNLOAD_FAILED = "unload_failed"


class NativeCleanupState(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"


class NativeVerificationState(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


class RetainedStageExecutor(Protocol):
    """Exact executor surface consumed by the core pilot."""

    def run_direct(
        self,
        *,
        fixture_id: str,
        instruction: str,
        workspace: Path,
        quality_enabled: bool,
        effort: str = "medium",
    ) -> RetainedAgentStage: ...

    def run_recovery(
        self,
        *,
        direct: RetainedAgentStage,
        archive: ImmutableWorkspaceArchive,
        branch_root: Path,
        containment_root: Path,
    ) -> RetainedAgentStage: ...


HiddenEvaluator = Callable[[str, Path], HiddenOutcome]
HiddenEvaluatorFactory = Callable[[], HiddenEvaluator]
FrozenInputVerifier = Callable[[], None]
_NATIVE_PUBLICATION_AUTHORITY = object()
_NATIVE_ATTEMPT_AUTHORITY = object()
_NATIVE_ABORT_AUTHORITY = object()


def _read_exact_json_seal(
    path: Path,
    *,
    expected_sha256: str,
    unavailable_message: str,
) -> dict[str, object]:
    """Read one regular JSON file only when its complete bytes match a seal."""

    candidate = Path(path).expanduser()
    try:
        metadata = candidate.lstat()
        payload = candidate.read_bytes()
    except OSError as exc:
        raise RepositoryPilotProtocolError(unavailable_message) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RepositoryPilotProtocolError(f"{unavailable_message}: not a regular file")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RepositoryPilotProtocolError(f"{unavailable_message}: frozen SHA-256 changed")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepositoryPilotProtocolError(f"{unavailable_message}: invalid JSON") from exc
    if type(document) is not dict:
        raise RepositoryPilotProtocolError(f"{unavailable_message}: root must be an object")
    return document


def _assert_predecessor_and_incident_seals() -> None:
    predecessor = _read_exact_json_seal(
        _REPOSITORY_ROOT / PREDECESSOR_PREREGISTRATION_RELATIVE_PATH,
        expected_sha256=PREDECESSOR_PREREGISTRATION_SHA256,
        unavailable_message="pilot predecessor preregistration is unavailable",
    )
    if (
        predecessor.get("schema") != PREDECESSOR_PREREGISTRATION_SCHEMA
        or predecessor.get("protocol_revision") != PREDECESSOR_PREREGISTRATION_REVISION
    ):
        raise RepositoryPilotProtocolError("pilot predecessor preregistration anchors changed")

    incident = _read_exact_json_seal(
        _REPOSITORY_ROOT / V2_INCIDENT_RELATIVE_PATH,
        expected_sha256=V2_INCIDENT_SHA256,
        unavailable_message="pilot v2 incident record is unavailable",
    )
    incident_protocol = incident.get("protocol")
    if (
        incident.get("schema") != V2_INCIDENT_SCHEMA
        or incident.get("status") != V2_INCIDENT_STATUS
        or type(incident_protocol) is not dict
        or incident_protocol.get("schema") != PREDECESSOR_PREREGISTRATION_SCHEMA
        or incident_protocol.get("revision") != PREDECESSOR_PREREGISTRATION_REVISION
        or incident_protocol.get("preregistration_sha256") != PREDECESSOR_PREREGISTRATION_SHA256
    ):
        raise RepositoryPilotProtocolError("pilot v2 incident anchors changed")


def _assert_preregistration_seal(
    path: Path | None = None,
) -> str:
    """Verify the exact reviewed preregistration bytes and semantic anchors."""

    candidate = Path(path or (_REPOSITORY_ROOT / PREREGISTRATION_RELATIVE_PATH)).expanduser()
    document = _read_exact_json_seal(
        candidate,
        expected_sha256=PREREGISTRATION_SHA256,
        unavailable_message="pilot preregistration is unavailable",
    )
    integrity = document.get("protocol_integrity")
    revision_history = document.get("revision_history")
    if type(integrity) is not dict or type(revision_history) is not dict:
        raise RepositoryPilotProtocolError("pilot preregistration has no protocol-integrity object")
    if (
        document.get("schema") != PREREGISTRATION_SCHEMA
        or document.get("protocol_revision") != PREREGISTRATION_REVISION
        or integrity.get("result_envelope_schema") != RESULT_ENVELOPE_SCHEMA
        or integrity.get("abort_envelope_schema") != ABORT_ENVELOPE_SCHEMA
        or integrity.get("attempt_start_schema") != ATTEMPT_START_SCHEMA
        or integrity.get("source_lock_schema") != SOURCE_LOCK_SCHEMA
        or tuple(integrity.get("source_lock_must_include", ())) != PILOT_SOURCE_LOCK_FILES
        or integrity.get("quality_profile_sha256") != GATE_PROFILE_SHA256
        or revision_history.get("predecessor_path") != PREDECESSOR_PREREGISTRATION_RELATIVE_PATH
        or revision_history.get("predecessor_schema") != PREDECESSOR_PREREGISTRATION_SCHEMA
        or revision_history.get("predecessor_revision") != PREDECESSOR_PREREGISTRATION_REVISION
        or revision_history.get("predecessor_sha256") != PREDECESSOR_PREREGISTRATION_SHA256
        or revision_history.get("post_hoc_incident_record_path") != V2_INCIDENT_RELATIVE_PATH
        or revision_history.get("post_hoc_incident_record_status") != V2_INCIDENT_STATUS
        or revision_history.get("post_hoc_incident_record_sha256") != V2_INCIDENT_SHA256
        or revision_history.get("v2_rerun_forbidden") is not True
    ):
        raise RepositoryPilotProtocolError("pilot preregistration semantic anchors changed")
    expected_models = {
        "target": {
            "repository_label": TARGET_REPOSITORY_LABEL,
            "content_identity": TARGET_CONTENT_IDENTITY,
        },
        "drafter": {
            "repository_label": DRAFT_REPOSITORY_LABEL,
            "content_identity": DRAFT_CONTENT_IDENTITY,
            "backend": "dflash",
        },
        "same_exact_stack_for_every_physical_generation": True,
        "identity_verified_before_and_after": True,
    }
    if document.get("models") != expected_models:
        raise RepositoryPilotProtocolError("pilot preregistration model identities changed")
    _assert_predecessor_and_incident_seals()
    return PREREGISTRATION_SHA256


@dataclass(frozen=True)
class FrozenPilotProvenance:
    """Host-only identity bundle reverified at every core integrity barrier."""

    source_lock: CleanSourceLock
    model_locks: tuple[LocalModelLock, LocalModelLock]
    runtime_identity: RuntimeIdentity
    cases: tuple[CorpusCase, ...]
    split: str
    tier: str
    suite_sha256: str
    private_protocol_sha256: str
    preregistration_sha256: str = PREREGISTRATION_SHA256

    def __post_init__(self) -> None:
        if not isinstance(self.source_lock, CleanSourceLock):
            raise TypeError("pilot provenance requires a CleanSourceLock")
        if self.source_lock.source_file_count != len(PILOT_SOURCE_LOCK_FILES):
            raise ValueError("pilot source-lock count differs from the preregistration")
        if not isinstance(self.runtime_identity, RuntimeIdentity):
            raise TypeError("pilot provenance requires a RuntimeIdentity")
        if not self.cases or any(not isinstance(case, CorpusCase) for case in self.cases):
            raise TypeError("pilot provenance requires one frozen CorpusCase cohort")
        if self.split not in _PRIVATE_PROTOCOL_SEALS:
            raise ValueError("pilot provenance split is unsupported")
        if not _SAFE_TIER.fullmatch(self.tier):
            raise ValueError("pilot provenance tier is not a safe public label")
        if self.preregistration_sha256 != PREREGISTRATION_SHA256:
            raise ValueError("pilot provenance preregistration SHA-256 changed")
        if self.private_protocol_sha256 != _PRIVATE_PROTOCOL_SEALS[self.split]:
            raise ValueError("pilot provenance private protocol seal changed")
        if self.suite_sha256 != sealed_suite_sha256(self.cases):
            raise ValueError("pilot provenance public suite seal changed")
        if self.private_protocol_sha256 != sealed_protocol_sha256(self.cases):
            raise ValueError("pilot provenance corpus/private-evaluator seal changed")
        expected_models = {
            "target": (TARGET_REPOSITORY_LABEL, TARGET_CONTENT_IDENTITY),
            "drafter": (DRAFT_REPOSITORY_LABEL, DRAFT_CONTENT_IDENTITY),
        }
        observed_models = {lock.role: (lock.repository_label, lock.content_identity) for lock in self.model_locks}
        if observed_models != expected_models or len(self.model_locks) != 2:
            raise ValueError("pilot provenance model identities changed")

    def verify(self) -> None:
        """Fail if any source, protocol, model, package, or host lock drifted."""

        _assert_frozen_environment()
        _assert_gate_profile_seal()
        if _assert_preregistration_seal() != self.preregistration_sha256:
            raise RepositoryPilotProtocolError("pilot preregistration drifted")
        if sealed_suite_sha256(self.cases) != self.suite_sha256:
            raise RepositoryPilotProtocolError("pilot public suite drifted")
        if sealed_protocol_sha256(self.cases) != self.private_protocol_sha256:
            raise RepositoryPilotProtocolError("pilot private evaluator protocol drifted")
        verify_clean_source_lock(
            self.source_lock,
            source_files=PILOT_SOURCE_LOCK_FILES,
        )
        verify_frozen_local_models(self.model_locks)
        verify_runtime_identity(self.runtime_identity)


@dataclass
class _NativeVerificationLedger:
    provenance: FrozenPilotProvenance
    successful_verifications: int = 0

    def verify(self) -> None:
        self.provenance.verify()
        self.successful_verifications += 1


def _provenance_payload(provenance: FrozenPilotProvenance) -> dict[str, object]:
    """Return the common source-free provenance carried by every native receipt."""

    models = {lock.role: lock for lock in provenance.model_locks}
    return {
        "implementation": {
            "git_revision": provenance.source_lock.git_revision,
            "source_lock_schema": SOURCE_LOCK_SCHEMA,
            "source_sha256": provenance.source_lock.source_sha256,
            "source_file_count": provenance.source_lock.source_file_count,
        },
        "protocol": {
            "preregistration_schema": PREREGISTRATION_SCHEMA,
            "preregistration_revision": PREREGISTRATION_REVISION,
            "preregistration_sha256": provenance.preregistration_sha256,
            "predecessor_preregistration_sha256": PREDECESSOR_PREREGISTRATION_SHA256,
            "v2_incident_record_sha256": V2_INCIDENT_SHA256,
            "public_suite_sha256": provenance.suite_sha256,
            "private_evaluator_bundle_sha256": provenance.private_protocol_sha256,
            "quality_profile_schema": GATE_PROFILE_SCHEMA,
            "quality_profile_sha256": GATE_PROFILE_SHA256,
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
        },
        "software": {
            "python_version": provenance.runtime_identity.python_version,
            "packages": dict(provenance.runtime_identity.software_versions),
        },
        "hardware": {"label": provenance.runtime_identity.hardware_label},
        "runtime": {
            "split": provenance.split,
            "tier": provenance.tier,
        },
    }


@dataclass(frozen=True)
class _NativeAttemptStartReceipt:
    """Private, source-free authority showing that this v3 attempt was claimed."""

    authority: object
    provenance: FrozenPilotProvenance
    started_at_utc: str

    def __post_init__(self) -> None:
        if self.authority is not _NATIVE_ATTEMPT_AUTHORITY:
            raise RepositoryPilotProtocolError("native attempt-start authority is invalid")
        if not isinstance(self.provenance, FrozenPilotProvenance):
            raise TypeError("native attempt-start requires FrozenPilotProvenance")
        try:
            parsed = datetime.fromisoformat(self.started_at_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("native attempt-start timestamp is invalid") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ValueError("native attempt-start timestamp must be UTC")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ATTEMPT_START_SCHEMA,
            "status": "started",
            "started_at_utc": self.started_at_utc,
            "provenance": _provenance_payload(self.provenance),
            "hidden_labels_serialized": False,
        }

    def serialize(self) -> str:
        self.__post_init__()
        payload = self.to_dict()
        _assert_source_free_artifact(payload)
        return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _NativeAbortReceipt:
    authority: object
    provenance: FrozenPilotProvenance
    attempt_start: _NativeAttemptStartReceipt
    failure_boundary: str
    reason_code: NativeAbortReason
    failure_message_sha256: str
    generation_receipt_issued: bool
    completed_root_generation_count: int
    completed_unique_extra_generation_count: int
    hidden_evaluator_constructed: bool
    hidden_evaluation_started: bool
    manager_state: NativeManagerState
    work_root_cleanup: NativeCleanupState
    post_abort_verification: NativeVerificationState
    successful_verifications: int

    def __post_init__(self) -> None:
        if self.authority is not _NATIVE_ABORT_AUTHORITY:
            raise RepositoryPilotProtocolError("native abort authority is invalid")
        if self.attempt_start.provenance is not self.provenance:
            raise RepositoryPilotProtocolError("native abort is bound to another attempt")
        self.attempt_start.__post_init__()
        if not _SAFE_TIER.fullmatch(self.failure_boundary):
            raise ValueError("native abort failure boundary is not a safe label")
        if not isinstance(self.reason_code, NativeAbortReason):
            raise TypeError("native abort reason code is not typed")
        if not re.fullmatch(r"[0-9a-f]{64}", self.failure_message_sha256):
            raise ValueError("native abort failure-message digest is invalid")
        for name in (
            "generation_receipt_issued",
            "hidden_evaluator_constructed",
            "hidden_evaluation_started",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"native abort {name} must be boolean")
        for name in (
            "completed_root_generation_count",
            "completed_unique_extra_generation_count",
            "successful_verifications",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"native abort {name} must be a non-negative integer")
        if not isinstance(self.manager_state, NativeManagerState):
            raise TypeError("native abort manager state is not typed")
        if not isinstance(self.work_root_cleanup, NativeCleanupState):
            raise TypeError("native abort cleanup state is not typed")
        if not isinstance(self.post_abort_verification, NativeVerificationState):
            raise TypeError("native abort verification state is not typed")
        if self.hidden_evaluation_started and not self.hidden_evaluator_constructed:
            raise ValueError("hidden evaluation cannot start before evaluator construction")


@dataclass(frozen=True)
class RepositoryQualityPilotAbortEnvelope:
    """Non-result terminal envelope issued only by a claimed native v3 attempt."""

    provenance: FrozenPilotProvenance
    attempt_start: _NativeAttemptStartReceipt
    abort_receipt: _NativeAbortReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, FrozenPilotProvenance):
            raise TypeError("abort envelope requires FrozenPilotProvenance")
        if type(self.attempt_start) is not _NativeAttemptStartReceipt:
            raise TypeError("abort envelope requires a native attempt-start receipt")
        if type(self.abort_receipt) is not _NativeAbortReceipt:
            raise TypeError("abort envelope requires a native abort receipt")
        self.abort_receipt.__post_init__()
        if (
            self.attempt_start.provenance is not self.provenance
            or self.abort_receipt.provenance is not self.provenance
            or self.abort_receipt.attempt_start is not self.attempt_start
        ):
            raise ValueError("abort envelope receipts are bound to another attempt")

    def to_dict(self) -> dict[str, object]:
        receipt = self.abort_receipt
        return {
            "schema_version": ABORT_ENVELOPE_SCHEMA,
            "status": "aborted_no_result",
            "attempt": {
                "start_schema": ATTEMPT_START_SCHEMA,
                "start_sha256": self.attempt_start.sha256,
            },
            "provenance": _provenance_payload(self.provenance),
            "abort": {
                "failure_boundary": receipt.failure_boundary,
                "reason_code": receipt.reason_code.value,
                "failure_message_sha256": receipt.failure_message_sha256,
                "generation_receipt_issued": receipt.generation_receipt_issued,
                "completed_root_generation_count": receipt.completed_root_generation_count,
                "completed_unique_extra_generation_count": (receipt.completed_unique_extra_generation_count),
                "hidden_evaluator_constructed": receipt.hidden_evaluator_constructed,
                "hidden_evaluation_started": receipt.hidden_evaluation_started,
                "manager_state": receipt.manager_state.value,
                "work_root_cleanup": receipt.work_root_cleanup.value,
                "post_abort_frozen_input_verification": receipt.post_abort_verification.value,
                "successful_verification_count": receipt.successful_verifications,
                "aggregate_produced": False,
            },
            "publication": {
                "result_envelope_created": False,
                "partial_generation_reuse_allowed": False,
                "quality_claim_allowed": False,
                "speed_claim_allowed": False,
                "breakthrough_claim_allowed": False,
            },
            "hidden_labels_serialized": False,
        }


def serialize_repository_quality_pilot_abort(
    envelope: RepositoryQualityPilotAbortEnvelope,
    *,
    indent: int | None = 2,
) -> str:
    if not isinstance(envelope, RepositoryQualityPilotAbortEnvelope):
        raise TypeError("only RepositoryQualityPilotAbortEnvelope can be serialized")
    envelope.__post_init__()
    payload = envelope.to_dict()
    _assert_source_free_artifact(payload)
    return json.dumps(payload, sort_keys=True, indent=indent, allow_nan=False) + "\n"


class NativePilotAborted(RepositoryPilotProtocolError):
    """Raised after a source-free abort envelope has been durably published."""

    def __init__(self, envelope: RepositoryQualityPilotAbortEnvelope) -> None:
        if not isinstance(envelope, RepositoryQualityPilotAbortEnvelope):
            raise TypeError("NativePilotAborted requires an abort envelope")
        self.envelope = envelope
        super().__init__(f"native pilot aborted: {envelope.abort_receipt.reason_code.value}")


@dataclass(frozen=True)
class PilotExecution:
    """Private completed execution; only ``aggregate`` is publishable."""

    protocol: PilotProtocol
    allocation: ExtraAllocation
    generation_receipt: GenerationCompletionReceipt
    records: tuple[FixturePilotRecord, ...]
    aggregate: PilotAggregate

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, PilotProtocol):
            raise TypeError("execution protocol must be PilotProtocol")
        if not isinstance(self.allocation, ExtraAllocation):
            raise TypeError("execution allocation must be ExtraAllocation")
        if not isinstance(self.generation_receipt, GenerationCompletionReceipt):
            raise TypeError("execution generation receipt has the wrong type")
        if not self.records or any(not isinstance(item, FixturePilotRecord) for item in self.records):
            raise TypeError("execution records must be FixturePilotRecord values")
        if not isinstance(self.aggregate, PilotAggregate):
            raise TypeError("execution aggregate must be PilotAggregate")


@dataclass(frozen=True)
class _NativePublicationReceipt:
    authority: object
    provenance: FrozenPilotProvenance
    attempt_start: _NativeAttemptStartReceipt
    execution: PilotExecution
    successful_verifications: int
    manager_unloaded_before_hidden: bool

    def __post_init__(self) -> None:
        if self.authority is not _NATIVE_PUBLICATION_AUTHORITY:
            raise RepositoryPilotProtocolError("native publication authority is invalid")
        if self.attempt_start.provenance is not self.provenance:
            raise RepositoryPilotProtocolError("native publication is bound to another attempt")
        self.attempt_start.__post_init__()
        if self.successful_verifications < 4:
            raise RepositoryPilotProtocolError("native publication verification ledger is incomplete")
        if self.manager_unloaded_before_hidden is not True:
            raise RepositoryPilotProtocolError("native manager was not unloaded before hidden evaluation")


@dataclass(frozen=True)
class RepositoryQualityPilotResultEnvelope:
    """Fixed-schema source-free envelope for a verified native execution."""

    provenance: FrozenPilotProvenance
    attempt_start: _NativeAttemptStartReceipt
    execution: PilotExecution
    publication_receipt: _NativePublicationReceipt
    post_run_verified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, FrozenPilotProvenance):
            raise TypeError("result envelope requires FrozenPilotProvenance")
        if not isinstance(self.execution, PilotExecution):
            raise TypeError("result envelope requires PilotExecution")
        if type(self.attempt_start) is not _NativeAttemptStartReceipt:
            raise TypeError("result envelope requires a native attempt-start receipt")
        if type(self.publication_receipt) is not _NativePublicationReceipt:
            raise TypeError("result envelope requires a native publication receipt")
        self.publication_receipt.__post_init__()
        if (
            self.publication_receipt.provenance is not self.provenance
            or self.publication_receipt.attempt_start is not self.attempt_start
            or self.publication_receipt.execution is not self.execution
        ):
            raise ValueError("native publication receipt is bound to another execution")
        if self.post_run_verified is not True:
            raise ValueError("result envelope requires successful post-run verification")
        if self.execution.protocol.suite_sha256 != self.provenance.suite_sha256:
            raise ValueError("result aggregate and provenance suites differ")
        if self.execution.protocol.cohort != self.provenance.split:
            raise ValueError("result aggregate and provenance splits differ")
        if self.execution.aggregate.protocol != self.execution.protocol:
            raise ValueError("result aggregate protocol differs from its execution")
        if self.execution.aggregate.allocation != self.execution.allocation:
            raise ValueError("result aggregate allocation differs from its execution")
        # Never trust the caller-provided post_run_verified bit.  Re-attest the
        # complete source/model/runtime/protocol bundle at construction and at
        # every public serialization boundary.
        self.provenance.verify()

    def to_dict(self) -> dict[str, object]:
        models = {lock.role: lock for lock in self.provenance.model_locks}
        receipt = self.execution.generation_receipt
        return {
            "schema_version": RESULT_ENVELOPE_SCHEMA,
            "attempt": {
                "start_schema": ATTEMPT_START_SCHEMA,
                "start_sha256": self.attempt_start.sha256,
            },
            "implementation": {
                "git_revision": self.provenance.source_lock.git_revision,
                "git_clean": True,
                "source_lock_schema": SOURCE_LOCK_SCHEMA,
                "source_sha256": self.provenance.source_lock.source_sha256,
                "source_file_count": self.provenance.source_lock.source_file_count,
                "post_run_source_stable": True,
            },
            "protocol": {
                "preregistration_schema": PREREGISTRATION_SCHEMA,
                "preregistration_revision": PREREGISTRATION_REVISION,
                "preregistration_sha256": self.provenance.preregistration_sha256,
                "predecessor_preregistration_sha256": PREDECESSOR_PREREGISTRATION_SHA256,
                "v2_incident_record_sha256": V2_INCIDENT_SHA256,
                "public_suite_sha256": self.provenance.suite_sha256,
                "private_evaluator_bundle_sha256": self.provenance.private_protocol_sha256,
                "quality_profile_schema": GATE_PROFILE_SCHEMA,
                "quality_profile_sha256": GATE_PROFILE_SHA256,
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
            "software": {
                "python_version": self.provenance.runtime_identity.python_version,
                "packages": dict(self.provenance.runtime_identity.software_versions),
            },
            "hardware": {"label": self.provenance.runtime_identity.hardware_label},
            "runtime": {
                "split": self.provenance.split,
                "tier": self.provenance.tier,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "sampler_seed": None,
                "schedule_and_bootstrap_seed": self.execution.protocol.seed,
                "context_window": FROZEN_CONTEXT_WINDOW,
                "max_output_tokens": FROZEN_MAX_OUTPUT_TOKENS,
                "tq_bits": 16,
                "pq_bits": 16,
                "bmp_paths": 1,
                "ddtree_budget": 0,
                "drafter_backend": "dflash",
                "drafter_strict": True,
                "draft_fallback_model": None,
                "dspark_prefix_cache": False,
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
                "cold_physical_generation_state": True,
                "network_enabled": False,
                "mcp_tools_exposed": False,
            },
            "generation_integrity": {
                "fixture_count": receipt.fixture_count,
                "expected_root_generation_count": receipt.expected_root_generation_count,
                "completed_root_generation_count": receipt.completed_root_generation_count,
                "expected_unique_extra_generation_count": (receipt.expected_unique_extra_generation_count),
                "completed_unique_extra_generation_count": (receipt.completed_unique_extra_generation_count),
                "root_schedule_sealed_before_first_generation": (receipt.root_schedule_sealed_before_first_generation),
                "allocation_sealed_after_all_roots": receipt.allocation_sealed_after_all_roots,
                "extra_schedule_sealed_before_first_extra": (receipt.extra_schedule_sealed_before_first_extra),
            },
            "aggregate": json.loads(serialize_source_free_aggregate(self.execution.aggregate)),
            "hidden_labels_serialized": False,
        }


def serialize_repository_quality_pilot_result(
    envelope: RepositoryQualityPilotResultEnvelope,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize one verified envelope without private rows, paths, or labels."""

    if not isinstance(envelope, RepositoryQualityPilotResultEnvelope):
        raise TypeError("only RepositoryQualityPilotResultEnvelope can be serialized")
    envelope.__post_init__()
    payload = envelope.to_dict()
    _assert_source_free_artifact(payload)
    return json.dumps(payload, sort_keys=True, indent=indent, allow_nan=False) + "\n"


@dataclass(frozen=True)
class _SealedManifest:
    path: Path
    sha256: str

    def verify(self) -> None:
        try:
            metadata = self.path.lstat()
            payload = self.path.read_bytes()
        except OSError as exc:
            raise RepositoryPilotProtocolError("a private schedule seal is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RepositoryPilotProtocolError("a private schedule seal is not a regular file")
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise RepositoryPilotProtocolError("a private schedule seal changed after creation")


@dataclass(frozen=True)
class _BoundCandidate:
    observation: CandidateObservation
    public_state: PublicRepositoryState
    archive: ImmutableWorkspaceArchive
    stage: RetainedAgentStage

    def verify(self) -> None:
        self.archive.verify_unchanged()
        if self.archive.tree_sha256 != self.observation.terminal_artifact_id:
            raise RepositoryPilotProtocolError("candidate observation and archive terminal bytes differ")
        if to_protocol_public_evidence(self.public_state) != self.observation.public_evidence:
            raise RepositoryPilotProtocolError("candidate observation and public state differ")


@dataclass
class _FixtureContext:
    case: CorpusCase
    fixture_root: Path
    contract: PublicScopeContract
    plain_workspace: Path
    quality_workspace: Path
    archive_root: Path
    branch_root: Path
    plain: _BoundCandidate | None = None
    quality: _BoundCandidate | None = None
    extra: _BoundCandidate | None = None


def bind_records_from_hidden_batch(
    *,
    barrier: HiddenEvaluationBarrier,
    selections: Sequence[SelectedFixtureCandidates],
    batch: HiddenEvaluationBatch,
) -> tuple[FixturePilotRecord, ...]:
    """Build records only from the exact batch emitted by ``barrier``.

    No outcome mapping is accepted separately.  The barrier verifies logical
    key, fixture, selected artifact digest, public evidence, physical reuse,
    and batch object identity before constructing any ``ArmHiddenOutcome``.
    """

    if type(barrier) is not HiddenEvaluationBarrier:
        raise TypeError("barrier must be the exact HiddenEvaluationBarrier type")
    if type(batch) is not HiddenEvaluationBatch:
        raise TypeError("batch must be the exact HiddenEvaluationBatch type")
    return barrier.bind_fixture_records(tuple(selections), batch)


class RepositoryQualityPilotOrchestrator:
    """Execute one complete frozen cohort through a strict phase machine."""

    def __init__(
        self,
        *,
        cases: Sequence[CorpusCase],
        protocol: PilotProtocol,
        executor: RetainedStageExecutor,
        work_root: Path,
        verify_frozen_inputs: FrozenInputVerifier,
        hidden_evaluator_factory: HiddenEvaluatorFactory,
    ) -> None:
        self.cases = tuple(cases)
        self.protocol = protocol
        self.executor = executor
        self.work_root = Path(work_root).expanduser()
        self.verify_frozen_inputs = verify_frozen_inputs
        self.hidden_evaluator_factory = hidden_evaluator_factory
        self.phase = PilotRunPhase.CREATED
        self.abort_from_phase: PilotRunPhase | None = None
        self.cleanup_complete = False

        self._fixture_ids: tuple[str, ...] = ()
        self._contexts: dict[str, _FixtureContext] = {}
        self._root_schedule: tuple[RootScheduledRun, ...] = ()
        self._extra_schedule: tuple[ExtraScheduledRun, ...] = ()
        self._root_manifest: _SealedManifest | None = None
        self._extra_manifest: _SealedManifest | None = None
        self._allocation: ExtraAllocation | None = None
        self._completed_roots: set[tuple[str, PhysicalRoot]] = set()
        self._completed_extras: set[str] = set()
        self._generation_receipt: GenerationCompletionReceipt | None = None
        self._barrier: HiddenEvaluationBarrier | None = None
        self._selections: tuple[SelectedFixtureCandidates, ...] = ()
        self._owns_work_root = False
        self._work_identity: tuple[int, int] | None = None

    def _require_phase(self, expected: PilotRunPhase) -> None:
        if self.phase is not expected:
            raise RepositoryPilotProtocolError(f"pilot phase {self.phase.value} cannot execute {expected.value} work")

    def _transition(self, expected: PilotRunPhase, target: PilotRunPhase) -> None:
        self._require_phase(expected)
        self.phase = target

    def _precheck(self) -> None:
        self._require_phase(PilotRunPhase.CREATED)
        if not isinstance(self.protocol, PilotProtocol):
            raise TypeError("protocol must be PilotProtocol")
        if not self.cases or any(not isinstance(case, CorpusCase) for case in self.cases):
            raise TypeError("cases must be a non-empty CorpusCase sequence")
        if not callable(getattr(self.executor, "run_direct", None)) or not callable(
            getattr(self.executor, "run_recovery", None)
        ):
            raise TypeError("executor does not implement the retained stage API")
        if not callable(self.verify_frozen_inputs) or not callable(self.hidden_evaluator_factory):
            raise TypeError("verifier and hidden evaluator factory must be callable")

        suite_sha256 = sealed_suite_sha256(self.cases)
        sealed_protocol_sha256(self.cases)
        if suite_sha256 != self.protocol.suite_sha256:
            raise RepositoryPilotProtocolError("corpus suite differs from the pilot protocol")
        if len(self.cases) != self.protocol.expected_fixture_count:
            raise RepositoryPilotProtocolError("corpus size differs from the frozen cohort")
        fixture_ids = tuple(case.fixture.fixture_id for case in self.cases)
        if len(fixture_ids) != len(set(fixture_ids)):
            raise RepositoryPilotProtocolError("corpus fixture IDs are duplicated")
        self.verify_frozen_inputs()
        self._fixture_ids = fixture_ids
        self._transition(PilotRunPhase.CREATED, PilotRunPhase.PRECHECKED)

    def _create_owned_work_root(self) -> None:
        if self.work_root.exists() or self.work_root.is_symlink():
            raise RepositoryPilotProtocolError("pilot work root must not already exist")
        try:
            parent = self.work_root.parent.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RepositoryPilotProtocolError("pilot work-root parent is unavailable") from exc
        if not parent.is_dir():
            raise RepositoryPilotProtocolError("pilot work-root parent must be a directory")
        destination = parent / self.work_root.name
        destination.mkdir(mode=0o700)
        metadata = destination.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RepositoryPilotProtocolError("pilot work root is not a real directory")
        self.work_root = destination
        self._owns_work_root = True
        self._work_identity = (metadata.st_dev, metadata.st_ino)

    @staticmethod
    def _seal_json(path: Path, payload: Mapping[str, object]) -> _SealedManifest:
        encoded = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        seal = _SealedManifest(path.resolve(strict=True), hashlib.sha256(encoded).hexdigest())
        seal.verify()
        return seal

    def _prepare_and_seal_root_schedule(self) -> None:
        self._require_phase(PilotRunPhase.PRECHECKED)
        self._create_owned_work_root()
        seals_root = self.work_root / "host-seals"
        fixtures_root = self.work_root / "fixtures"
        seals_root.mkdir(mode=0o700)
        fixtures_root.mkdir(mode=0o700)

        for index, case in enumerate(self.cases):
            fixture_root = fixtures_root / f"fixture-{index:04d}"
            source_root = fixture_root / "source"
            direct_root = fixture_root / "direct"
            archive_root = fixture_root / "archives"
            branch_root = fixture_root / "branches"
            fixture_root.mkdir(mode=0o700)
            materialized = materialize_public_fixture(case.fixture, source_root)
            contract = PublicScopeContract.capture(
                case.fixture.fixture_id,
                materialized.workspace,
                editable_names=case.editable_names,
            )
            roots = prepare_pristine_direct_roots(materialized.workspace, direct_root)
            archive_root.mkdir(mode=0o700)
            branch_root.mkdir(mode=0o700)
            self._contexts[case.fixture.fixture_id] = _FixtureContext(
                case=case,
                fixture_root=fixture_root,
                contract=contract,
                plain_workspace=roots.plain,
                quality_workspace=roots.quality,
                archive_root=archive_root,
                branch_root=branch_root,
            )

        self._root_schedule = make_root_schedule(self._fixture_ids, seed=self.protocol.seed)
        self._root_manifest = self._seal_json(
            seals_root / "root-schedule.json",
            {
                "schema": "mio.repository-quality-private-root-schedule.v1",
                "suite_sha256": self.protocol.suite_sha256,
                "seed": self.protocol.seed,
                "runs": [
                    {
                        "fixture_id": item.fixture_id,
                        "physical_root": item.root.value,
                        "schedule_index": item.schedule_index,
                    }
                    for item in self._root_schedule
                ],
            },
        )
        self._transition(PilotRunPhase.PRECHECKED, PilotRunPhase.ROOT_SCHEDULE_SEALED)

    def _bind_stage(
        self,
        *,
        context: _FixtureContext,
        stage: RetainedAgentStage,
        expected_workspace: Path,
        expected_stage: str,
        quality_enabled: bool,
        physical_candidate_id: str,
        archive_name: str,
    ) -> _BoundCandidate:
        if not isinstance(stage, RetainedAgentStage):
            raise RepositoryPilotProtocolError("executor returned an untyped retained stage")
        if stage.fixture_id != context.case.fixture.fixture_id:
            raise RepositoryPilotProtocolError("executor stage is bound to the wrong fixture")
        if stage.stage != expected_stage or stage.quality_enabled is not quality_enabled:
            raise RepositoryPilotProtocolError("executor stage kind or Quality flag is wrong")
        if stage.workspace.resolve(strict=True) != expected_workspace.resolve(strict=True):
            raise RepositoryPilotProtocolError("executor stage is bound to the wrong workspace")
        if regular_tree_sha256(stage.workspace) != stage.terminal_tree_sha256:
            raise RepositoryPilotProtocolError("executor terminal tree commitment is stale")

        verdict = context.contract.assess_terminal(
            fixture_id=stage.fixture_id,
            terminal_root=stage.workspace,
            terminal_tree_sha256=stage.terminal_tree_sha256,
        )
        public_state = extract_public_repository_state(
            stage,
            scope_contract=context.contract,
            scope_verdict=verdict,
        )
        archive = ImmutableWorkspaceArchive.capture(
            stage.workspace,
            context.archive_root / archive_name,
            containment_root=context.fixture_root,
        )
        if archive.tree_sha256 != stage.terminal_tree_sha256:
            raise RepositoryPilotProtocolError("terminal archive differs from its retained stage")
        observation = CandidateObservation(
            physical_candidate_id=physical_candidate_id,
            terminal_artifact_id=archive.tree_sha256,
            public_evidence=to_protocol_public_evidence(public_state),
            cost=to_protocol_candidate_cost(public_state),
        )
        bound = _BoundCandidate(observation, public_state, archive, stage)
        bound.verify()
        return bound

    def _run_roots(self) -> None:
        self._require_phase(PilotRunPhase.ROOT_SCHEDULE_SEALED)
        if self._root_manifest is None:
            raise RepositoryPilotProtocolError("root schedule has no private seal")
        self._root_manifest.verify()
        for item in self._root_schedule:
            key = (item.fixture_id, item.root)
            if key in self._completed_roots:
                raise RepositoryPilotProtocolError("root schedule attempted a duplicate physical generation")
            context = self._contexts[item.fixture_id]
            if item.root is PhysicalRoot.PLAIN:
                workspace = context.plain_workspace
                quality_enabled = False
                archive_name = "plain-root"
            else:
                workspace = context.quality_workspace
                quality_enabled = True
                archive_name = "quality-root"
            if regular_tree_sha256(workspace) != context.contract.pristine_tree_sha256:
                raise RepositoryPilotProtocolError("scheduled direct workspace is no longer pristine")
            stage = self.executor.run_direct(
                fixture_id=item.fixture_id,
                instruction=context.case.fixture.instruction,
                workspace=workspace,
                quality_enabled=quality_enabled,
                effort="medium",
            )
            bound = self._bind_stage(
                context=context,
                stage=stage,
                expected_workspace=workspace,
                expected_stage="direct",
                quality_enabled=quality_enabled,
                physical_candidate_id=f"{item.fixture_id}:root:{item.root.value}",
                archive_name=archive_name,
            )
            if item.root is PhysicalRoot.PLAIN:
                if context.plain is not None:
                    raise RepositoryPilotProtocolError("Plain root was already recorded")
                context.plain = bound
            else:
                if context.quality is not None:
                    raise RepositoryPilotProtocolError("Quality root was already recorded")
                context.quality = bound
            self._completed_roots.add(key)
        self._root_manifest.verify()
        expected = {(item.fixture_id, item.root) for item in self._root_schedule}
        if self._completed_roots != expected or len(expected) != len(self._fixture_ids) * 2:
            raise RepositoryPilotProtocolError("root execution count differs from the sealed schedule")
        if any(context.plain is None or context.quality is None for context in self._contexts.values()):
            raise RepositoryPilotProtocolError("a fixture is missing a sealed direct root")
        self._transition(PilotRunPhase.ROOT_SCHEDULE_SEALED, PilotRunPhase.ROOTS_COMPLETE)

    def _seal_allocation(self) -> None:
        self._require_phase(PilotRunPhase.ROOTS_COMPLETE)
        evidence = {
            fixture_id: context.quality.observation.public_evidence
            for fixture_id, context in self._contexts.items()
            if context.quality is not None
        }
        if set(evidence) != set(self._fixture_ids):
            raise RepositoryPilotProtocolError("Quality evidence is incomplete before allocation")
        allocation = allocate_extras(
            self._fixture_ids,
            evidence,
            router=self.protocol.router,
            seed=self.protocol.seed,
        )
        schedule = make_extra_schedule(
            allocation,
            spec=self.protocol.extra_spec,
            seed=self.protocol.seed,
        )
        self._allocation = allocation
        self._extra_schedule = schedule
        self._extra_manifest = self._seal_json(
            self.work_root / "host-seals" / "extra-schedule.json",
            {
                "schema": "mio.repository-quality-private-extra-schedule.v1",
                "markov_fixture_ids": list(allocation.markov_fixture_ids),
                "static_fixture_ids": list(allocation.static_fixture_ids),
                "runs": [
                    {
                        "fixture_id": item.fixture_id,
                        "logical_arms": [arm.value for arm in item.arms],
                        "action": item.action.value,
                        "schedule_index": item.schedule_index,
                    }
                    for item in schedule
                ],
            },
        )
        self._transition(PilotRunPhase.ROOTS_COMPLETE, PilotRunPhase.ALLOCATION_SEALED)

    def _run_extras(self) -> None:
        self._require_phase(PilotRunPhase.ALLOCATION_SEALED)
        if self._allocation is None or self._extra_manifest is None:
            raise RepositoryPilotProtocolError("extra allocation has no private seal")
        self._extra_manifest.verify()
        for item in self._extra_schedule:
            if item.fixture_id in self._completed_extras:
                raise RepositoryPilotProtocolError("extra schedule attempted a duplicate physical generation")
            context = self._contexts[item.fixture_id]
            if context.quality is None:
                raise RepositoryPilotProtocolError("extra generation has no shared Quality root")
            context.quality.verify()
            branch = context.branch_root / "recovery"
            stage = self.executor.run_recovery(
                direct=context.quality.stage,
                archive=context.quality.archive,
                branch_root=branch,
                containment_root=context.fixture_root,
            )
            context.extra = self._bind_stage(
                context=context,
                stage=stage,
                expected_workspace=branch,
                expected_stage="recovery",
                quality_enabled=True,
                physical_candidate_id=f"{item.fixture_id}:extra:refine",
                archive_name="extra-refine",
            )
            self._completed_extras.add(item.fixture_id)
        self._extra_manifest.verify()
        expected = {item.fixture_id for item in self._extra_schedule}
        if self._completed_extras != expected:
            raise RepositoryPilotProtocolError("extra execution count differs from the sealed schedule")
        for context in self._contexts.values():
            if context.plain is None or context.quality is None:
                raise RepositoryPilotProtocolError("direct candidate disappeared during extra generation")
            context.plain.verify()
            context.quality.verify()
            if context.extra is not None:
                context.extra.verify()
        self._transition(PilotRunPhase.ALLOCATION_SEALED, PilotRunPhase.EXTRAS_COMPLETE)

    def _issue_generation_receipt(self) -> GenerationCompletionReceipt:
        self._require_phase(PilotRunPhase.EXTRAS_COMPLETE)
        if self._root_manifest is None or self._extra_manifest is None:
            raise RepositoryPilotProtocolError("generation schedules are not sealed")
        self._root_manifest.verify()
        self._extra_manifest.verify()
        expected_roots = {(item.fixture_id, item.root) for item in self._root_schedule}
        expected_extras = {item.fixture_id for item in self._extra_schedule}
        if self._completed_roots != expected_roots or self._completed_extras != expected_extras:
            raise RepositoryPilotProtocolError("observed generation sets differ from sealed schedules")
        receipt = GenerationCompletionReceipt(
            fixture_count=len(self._fixture_ids),
            expected_root_generation_count=len(self._root_schedule),
            completed_root_generation_count=len(self._completed_roots),
            expected_unique_extra_generation_count=len(self._extra_schedule),
            completed_unique_extra_generation_count=len(self._completed_extras),
            root_schedule_sealed_before_first_generation=True,
            allocation_sealed_after_all_roots=True,
            extra_schedule_sealed_before_first_extra=True,
        )
        self._generation_receipt = receipt
        return receipt

    @staticmethod
    def _selected_bound(
        context: _FixtureContext,
        selected: SelectedFixtureCandidates,
        arm: LogicalArm,
    ) -> _BoundCandidate:
        candidate = selected.candidate_for_arm(arm)
        choices = tuple(item for item in (context.plain, context.quality, context.extra) if item is not None)
        matches = tuple(item for item in choices if item.observation is candidate)
        if len(matches) != 1:
            raise RepositoryPilotProtocolError("logical selection does not bind one physical candidate")
        return matches[0]

    def _seal_selections(self) -> None:
        self._require_phase(PilotRunPhase.EXTRAS_COMPLETE)
        if self._allocation is None:
            raise RepositoryPilotProtocolError("selection has no sealed allocation")
        receipt = self._issue_generation_receipt()
        static = set(self._allocation.static_fixture_ids)
        markov = set(self._allocation.markov_fixture_ids)
        selections: list[SelectedFixtureCandidates] = []
        for fixture_id in self._fixture_ids:
            context = self._contexts[fixture_id]
            if context.plain is None or context.quality is None:
                raise RepositoryPilotProtocolError("selection is missing a direct candidate")
            should_have_extra = fixture_id in static or fixture_id in markov
            if (context.extra is not None) is not should_have_extra:
                raise RepositoryPilotProtocolError("physical extra presence contradicts the sealed allocation")
            static_child = context.extra.observation if fixture_id in static and context.extra is not None else None
            markov_child = context.extra.observation if fixture_id in markov and context.extra is not None else None
            selections.append(
                SelectedFixtureCandidates(
                    fixture_id=fixture_id,
                    plain_root=context.plain.observation,
                    quality_root=context.quality.observation,
                    static_child=static_child,
                    markov_child=markov_child,
                    static_selection=select_candidate(context.quality.observation, static_child),
                    markov_selection=select_candidate(context.quality.observation, markov_child),
                )
            )

        barrier = HiddenEvaluationBarrier.for_fixtures(self._fixture_ids)
        for selected in selections:
            context = self._contexts[selected.fixture_id]
            for arm in LOGICAL_ARMS:
                bound = self._selected_bound(context, selected, arm)
                bound.verify()
                barrier.register(
                    logical_terminal_key(selected.fixture_id, arm),
                    bound.archive.root,
                    bound.public_state,
                    fixture_id=selected.fixture_id,
                    observation=bound.observation,
                )
        barrier.seal(generation_receipt=receipt)
        self._barrier = barrier
        self._selections = tuple(selections)
        self._transition(PilotRunPhase.EXTRAS_COMPLETE, PilotRunPhase.SELECTIONS_SEALED)

    def _evaluate_and_aggregate(self) -> PilotExecution:
        self._require_phase(PilotRunPhase.SELECTIONS_SEALED)
        if self._barrier is None or self._allocation is None or self._generation_receipt is None:
            raise RepositoryPilotProtocolError("sealed execution state is incomplete")
        self.verify_frozen_inputs()
        self._transition(PilotRunPhase.SELECTIONS_SEALED, PilotRunPhase.PRE_HIDDEN_VERIFIED)

        hidden_evaluator = self.hidden_evaluator_factory()
        if not callable(hidden_evaluator):
            raise TypeError("hidden evaluator factory must return a callable")
        batch = self._barrier.evaluate(hidden_evaluator)
        self._transition(PilotRunPhase.PRE_HIDDEN_VERIFIED, PilotRunPhase.HIDDEN_COMPLETE)

        self.verify_frozen_inputs()
        self._transition(PilotRunPhase.HIDDEN_COMPLETE, PilotRunPhase.POST_HIDDEN_VERIFIED)
        records = bind_records_from_hidden_batch(
            barrier=self._barrier,
            selections=self._selections,
            batch=batch,
        )
        aggregate = build_aggregate(
            protocol=self.protocol,
            allocation=self._allocation,
            records=records,
            barrier_receipt=batch.receipt,
        )
        execution = PilotExecution(
            protocol=self.protocol,
            allocation=self._allocation,
            generation_receipt=self._generation_receipt,
            records=records,
            aggregate=aggregate,
        )
        self._transition(PilotRunPhase.POST_HIDDEN_VERIFIED, PilotRunPhase.COMPLETE)
        return execution

    def _cleanup_owned_work_root(self) -> None:
        if not self._owns_work_root:
            return
        try:
            metadata = self.work_root.lstat()
        except FileNotFoundError:
            raise RepositoryPilotProtocolError("owned pilot work root disappeared before cleanup") from None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RepositoryPilotProtocolError("owned pilot work root changed kind before cleanup")
        if self._work_identity != (metadata.st_dev, metadata.st_ino):
            raise RepositoryPilotProtocolError("owned pilot work root identity changed before cleanup")
        shutil.rmtree(self.work_root)
        self._owns_work_root = False

    def run(self) -> PilotExecution:
        """Run the complete cohort once and remove every private workspace."""

        failure: BaseException | None = None
        try:
            self._precheck()
            self._prepare_and_seal_root_schedule()
            self._run_roots()
            self._seal_allocation()
            self._run_extras()
            self._seal_selections()
            return self._evaluate_and_aggregate()
        except BaseException as exc:
            failure = exc
            self.abort_from_phase = self.phase
            self.phase = PilotRunPhase.ABORTED
            raise
        finally:
            try:
                self._cleanup_owned_work_root()
                self.cleanup_complete = True
            except BaseException:
                self.cleanup_complete = False
                if self.abort_from_phase is None:
                    self.abort_from_phase = self.phase
                self.phase = PilotRunPhase.ABORTED
                if failure is None:
                    raise
            finally:
                self._contexts.clear()
                self._barrier = None
                self._selections = ()


def execute_repository_quality_pilot(
    *,
    cases: Sequence[CorpusCase],
    protocol: PilotProtocol,
    executor: RetainedStageExecutor,
    work_root: Path,
    verify_frozen_inputs: FrozenInputVerifier,
    hidden_evaluator_factory: HiddenEvaluatorFactory,
) -> PilotExecution:
    """Convenience entry point with no caller-controlled receipts or outcomes."""

    return RepositoryQualityPilotOrchestrator(
        cases=cases,
        protocol=protocol,
        executor=executor,
        work_root=work_root,
        verify_frozen_inputs=verify_frozen_inputs,
        hidden_evaluator_factory=hidden_evaluator_factory,
    ).run()


def _capture_native_provenance(
    *,
    cases: tuple[CorpusCase, ...],
    split: str,
    tier: str,
    target_path: Path,
    draft_path: Path,
) -> FrozenPilotProvenance:
    """Capture every frozen native identity before model loading."""

    _assert_frozen_environment()
    _assert_gate_profile_seal()
    preregistration_sha256 = _assert_preregistration_seal()
    suite_sha256 = sealed_suite_sha256(cases)
    private_protocol_sha256 = sealed_protocol_sha256(cases)
    source_lock = capture_clean_source_lock(
        _REPOSITORY_ROOT,
        source_files=PILOT_SOURCE_LOCK_FILES,
    )
    runtime_identity = collect_runtime_identity()
    model_locks = bind_frozen_local_models(target_path, draft_path)
    return FrozenPilotProvenance(
        source_lock=source_lock,
        model_locks=model_locks,
        runtime_identity=runtime_identity,
        cases=cases,
        split=split,
        tier=tier,
        suite_sha256=suite_sha256,
        private_protocol_sha256=private_protocol_sha256,
        preregistration_sha256=preregistration_sha256,
    )


def _validate_native_work_location(
    candidate: Path,
    *,
    source_root: Path,
    model_locks: Sequence[LocalModelLock],
    label: str,
    must_exist: bool,
) -> Path:
    """Reject work locations inside, equal to, or containing protected roots."""

    raw = Path(candidate).expanduser()
    if raw.is_symlink():
        raise RepositoryPilotProtocolError(f"{label} must not be a symlink")
    try:
        resolved = raw.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise RepositoryPilotProtocolError(f"{label} is unavailable") from exc
    protected = (
        Path(source_root).resolve(strict=True),
        *(lock.resolved_path.resolve(strict=True) for lock in model_locks),
    )
    for root in protected:
        try:
            resolved.relative_to(root)
            overlaps = True
        except ValueError:
            try:
                root.relative_to(resolved)
                overlaps = True
            except ValueError:
                overlaps = False
        if overlaps:
            raise RepositoryPilotProtocolError(f"{label} must stay disjoint from source and model roots")
    if must_exist and not resolved.is_dir():
        raise RepositoryPilotProtocolError(f"{label} must be a directory")
    return resolved


def _validate_new_output_path(
    output: Path | None,
    *,
    source_root: Path,
    model_locks: Sequence[LocalModelLock],
) -> Path | None:
    """Require a new result name outside every protected input root."""

    validated = validate_output_path(
        output,
        source_root=source_root,
        model_locks=model_locks,
    )
    if validated is None:
        return None
    if validated.exists() or validated.is_symlink():
        raise RepositoryPilotProtocolError("pilot output is create-once and must not exist")
    try:
        parent = validated.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RepositoryPilotProtocolError("pilot output parent is unavailable") from exc
    if not parent.is_dir():
        raise RepositoryPilotProtocolError("pilot output parent must be a directory")
    return validated


def _atomic_create_result(path: Path, content: str) -> None:
    """Publish complete bytes exactly once, without an overwrite-capable rename."""

    destination = Path(path)
    parent = destination.parent.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise RepositoryPilotProtocolError("pilot output is create-once and already exists") from exc
        temporary.unlink()
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


_ATTEMPT_START_FILENAME = "attempt-start.json"
_ATTEMPT_RESULT_FILENAME = "result.json"
_ATTEMPT_ABORT_FILENAME = "abort.json"


class _NativeAbortSignal(RuntimeError):
    def __init__(self, reason_code: NativeAbortReason, failure_boundary: str) -> None:
        self.reason_code = reason_code
        self.failure_boundary = failure_boundary
        super().__init__(reason_code.value)


@dataclass
class _RawTimingAttestingExecutor:
    """Reject non-raw native DFlash telemetry with a typed abort reason."""

    delegate: RetainedStageExecutor

    @staticmethod
    def _attest(stage: RetainedAgentStage) -> RetainedAgentStage:
        rounds = getattr(getattr(stage, "result", None), "rounds", None)
        if type(rounds) is tuple and any(
            getattr(round_trace, "timing_source", None) != "runtime_raw_ns" for round_trace in rounds
        ):
            raise _NativeAbortSignal(
                NativeAbortReason.DFLASH_RAW_PHASE_TELEMETRY_MISSING,
                "root_generation_telemetry_validation",
            )
        return stage

    def run_direct(self, **kwargs: Any) -> RetainedAgentStage:
        return self._attest(self.delegate.run_direct(**kwargs))

    def run_recovery(self, **kwargs: Any) -> RetainedAgentStage:
        return self._attest(self.delegate.run_recovery(**kwargs))


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        try:
            right.relative_to(left)
            return True
        except ValueError:
            return False


def _claim_native_attempt_root(
    candidate: Path,
    *,
    source_root: Path,
    model_locks: Sequence[LocalModelLock],
    work_root: Path,
) -> Path:
    """Atomically claim one persistent, empty receipt root for this v3 attempt."""

    resolved = _validate_native_work_location(
        candidate,
        source_root=source_root,
        model_locks=model_locks,
        label="native attempt root",
        must_exist=False,
    )
    try:
        parent = resolved.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RepositoryPilotProtocolError("native attempt-root parent is unavailable") from exc
    if not parent.is_dir():
        raise RepositoryPilotProtocolError("native attempt-root parent must be a directory")
    resolved_work_root = Path(work_root).expanduser().resolve(strict=False)
    if _paths_overlap(resolved, resolved_work_root):
        raise RepositoryPilotProtocolError("native attempt root and private work root must be disjoint")
    try:
        resolved.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise RepositoryPilotProtocolError("native v3 attempt root is create-once and already exists") from exc
    except OSError as exc:
        raise RepositoryPilotProtocolError("native v3 attempt root could not be claimed") from exc
    metadata = resolved.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RepositoryPilotProtocolError("native v3 attempt root is not a real directory")
    return resolved


def _issue_native_attempt_start(provenance: FrozenPilotProvenance) -> _NativeAttemptStartReceipt:
    provenance.verify()
    return _NativeAttemptStartReceipt(
        authority=_NATIVE_ATTEMPT_AUTHORITY,
        provenance=provenance,
        started_at_utc=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )


def _verify_attempt_start_file(root: Path, receipt: _NativeAttemptStartReceipt) -> None:
    path = root / _ATTEMPT_START_FILENAME
    try:
        metadata = path.lstat()
        payload = path.read_bytes()
    except OSError as exc:
        raise RepositoryPilotProtocolError("native attempt-start receipt is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RepositoryPilotProtocolError("native attempt-start receipt is not a regular file")
    if hashlib.sha256(payload).hexdigest() != receipt.sha256:
        raise RepositoryPilotProtocolError("native attempt-start receipt changed after creation")


def _native_abort_reason(
    failure: BaseException,
    *,
    failure_boundary: str,
) -> NativeAbortReason:
    if isinstance(failure, _NativeAbortSignal):
        return failure.reason_code
    if failure_boundary == "model_load":
        return NativeAbortReason.MODEL_LOAD_FAILURE
    if failure_boundary == "manager_unload":
        return NativeAbortReason.MANAGER_UNLOAD_FAILURE
    if isinstance(failure, RepositoryPilotProtocolError):
        return NativeAbortReason.PROTOCOL_FAILURE
    return NativeAbortReason.INFRASTRUCTURE_FAILURE


def run_native_repository_quality_pilot(
    *,
    split: str,
    tier: str,
    config_path: Path | None,
    target_path: Path,
    draft_path: Path,
    work_root: Path,
    attempt_root: Path,
    output: Path | None = None,
) -> RepositoryQualityPilotResultEnvelope:
    """Run the authorized native smoke cohort with non-injectable provenance."""

    if split != "smoke":
        raise RepositoryPilotProtocolError(
            "native pilot v3 authorizes smoke only; all requires a later integrity-authorized wrapper"
        )
    cases = select_cases(split)
    provenance = _capture_native_provenance(
        cases=cases,
        split=split,
        tier=tier,
        target_path=target_path,
        draft_path=draft_path,
    )
    _validate_new_output_path(
        output,
        source_root=provenance.source_lock.repo_root,
        model_locks=provenance.model_locks,
    )
    validated_work_root = _validate_native_work_location(
        work_root,
        source_root=provenance.source_lock.repo_root,
        model_locks=provenance.model_locks,
        label="pilot work root",
        must_exist=False,
    )
    attempt_start = _issue_native_attempt_start(provenance)
    claimed_attempt_root = _claim_native_attempt_root(
        attempt_root,
        source_root=provenance.source_lock.repo_root,
        model_locks=provenance.model_locks,
        work_root=validated_work_root,
    )

    manager: Any | None = None
    manager_state = NativeManagerState.NEVER_LOADED
    unload_attempted = False
    verification_ledger = _NativeVerificationLedger(provenance)
    hidden_evaluator_constructed = False
    hidden_evaluation_started = False
    orchestrator: RepositoryQualityPilotOrchestrator | None = None
    execution: PilotExecution | None = None
    result_envelope: RepositoryQualityPilotResultEnvelope | None = None
    failure: BaseException | None = None
    failure_boundary = "attempt_start_publication"
    attempt_start_published = False

    def unload_manager() -> None:
        nonlocal manager_state, unload_attempted
        if manager is None or manager_state is NativeManagerState.UNLOADED or unload_attempted:
            return
        unload_attempted = True
        try:
            manager.unload_all()
        except BaseException:
            manager_state = NativeManagerState.UNLOAD_FAILED
            raise
        manager_state = NativeManagerState.UNLOADED

    try:
        try:
            _atomic_create_result(
                claimed_attempt_root / _ATTEMPT_START_FILENAME,
                attempt_start.serialize(),
            )
        except BaseException:
            # os.link() may have installed the create-once receipt even when a
            # later temporary-file unlink or directory fsync fails. Treat that
            # exact, verifiable path as a consumed attempt and terminalize it
            # with an abort instead of stranding a start-only directory.
            _verify_attempt_start_file(claimed_attempt_root, attempt_start)
            attempt_start_published = True
            raise
        _verify_attempt_start_file(claimed_attempt_root, attempt_start)
        attempt_start_published = True

        failure_boundary = "model_load"
        with redirect_stdout(sys.stderr):
            native_executor, manager = _load_native_executor(
                tier=tier,
                config_path=config_path,
                target_path=provenance.model_locks[0].resolved_path,
                draft_path=provenance.model_locks[1].resolved_path,
            )
        manager_state = NativeManagerState.LOADED
        failure_boundary = "native_executor_construction"
        executor = _RawTimingAttestingExecutor(
            RetainedNativeAgentExecutor(
                config=native_executor.config,
                manager=native_executor.manager,
                engine=native_executor.engine,
                tier=native_executor.tier,
            )
        )

        def hidden_evaluator_factory() -> HiddenEvaluator:
            nonlocal hidden_evaluator_constructed
            # Model memory and mutable runtime state are gone before the private
            # evaluator object or callback can be constructed.
            unload_manager()
            corpus_evaluator = CorpusHiddenEvaluator(cases)
            hidden_evaluator_constructed = True

            def evaluate(fixture_id: str, root: Path) -> HiddenOutcome:
                nonlocal hidden_evaluation_started
                hidden_evaluation_started = True
                evaluation = corpus_evaluator(
                    EvaluationRequest(
                        fixture_id=fixture_id,
                        condition="sealed_terminal",
                        workspace=root,
                        schedule_index=0,
                    )
                )
                if not isinstance(evaluation, HiddenEvaluation):
                    raise RepositoryPilotProtocolError("corpus hidden evaluator returned an invalid result")
                return HiddenOutcome(
                    evaluator_passed=evaluation.passed,
                    regression_free=evaluation.regression_free,
                )

            return evaluate

        orchestrator = RepositoryQualityPilotOrchestrator(
            cases=cases,
            protocol=PilotProtocol(
                suite_sha256=provenance.suite_sha256,
                seed=FROZEN_SEED,
            ),
            executor=executor,
            work_root=validated_work_root,
            verify_frozen_inputs=verification_ledger.verify,
            hidden_evaluator_factory=hidden_evaluator_factory,
        )
        failure_boundary = "cohort_execution"
        with redirect_stdout(sys.stderr):
            execution = orchestrator.run()
        # The core verifies after hidden evaluation and before aggregation.
        # Reverify once more after aggregation so the published envelope covers
        # the complete execution interval.
        failure_boundary = "post_run_verification"
        verification_ledger.verify()
    except BaseException as exc:
        failure = exc
        if isinstance(exc, _NativeAbortSignal):
            failure_boundary = exc.failure_boundary
        elif manager_state is NativeManagerState.UNLOAD_FAILED:
            failure_boundary = "manager_unload"
        elif orchestrator is not None and orchestrator.abort_from_phase is not None:
            failure_boundary = orchestrator.abort_from_phase.value
    finally:
        try:
            unload_manager()
        except BaseException as exc:
            if failure is None:
                failure = exc
                failure_boundary = "manager_unload"

    if failure is not None and not attempt_start_published:
        # No verifiable start receipt means no scientific attempt exists and
        # therefore no receipt-bound abort can be authored. Reclaim the empty
        # directory when possible, while preserving any unexpected evidence.
        try:
            claimed_attempt_root.rmdir()
        except OSError:
            pass
        raise RepositoryPilotProtocolError(
            "native attempt-start publication failed before a verifiable receipt"
        ) from failure

    if failure is None:
        try:
            failure_boundary = "result_terminalization"
            if execution is None:  # pragma: no cover - defensive state-machine guard
                raise RepositoryPilotProtocolError("native pilot ended without execution or failure")
            publication_receipt = _NativePublicationReceipt(
                authority=_NATIVE_PUBLICATION_AUTHORITY,
                provenance=provenance,
                attempt_start=attempt_start,
                execution=execution,
                successful_verifications=verification_ledger.successful_verifications,
                manager_unloaded_before_hidden=manager_state is NativeManagerState.UNLOADED,
            )
            result_envelope = RepositoryQualityPilotResultEnvelope(
                provenance=provenance,
                attempt_start=attempt_start,
                execution=execution,
                publication_receipt=publication_receipt,
                post_run_verified=True,
            )
            serialized_result = serialize_repository_quality_pilot_result(result_envelope)
            _verify_attempt_start_file(claimed_attempt_root, attempt_start)
            _atomic_create_result(
                claimed_attempt_root / _ATTEMPT_RESULT_FILENAME,
                serialized_result,
            )
        except BaseException as exc:
            failure = exc

    if failure is not None:
        # A create-once terminal path wins even if the final directory fsync
        # raised after linking it. Never publish contradictory result+abort
        # siblings for one attempt.
        result_path = claimed_attempt_root / _ATTEMPT_RESULT_FILENAME
        if result_path.exists() or result_path.is_symlink():
            raise RepositoryPilotProtocolError(
                "native result terminalization failed after the result path became occupied"
            ) from None
        post_abort_verification = NativeVerificationState.NOT_ATTEMPTED
        try:
            verification_ledger.verify()
        except BaseException:
            post_abort_verification = NativeVerificationState.FAILED
        else:
            post_abort_verification = NativeVerificationState.PASSED

        cleanup_state = NativeCleanupState.COMPLETE
        if orchestrator is not None and not orchestrator.cleanup_complete:
            cleanup_state = NativeCleanupState.FAILED
        roots_completed = 0 if orchestrator is None else len(orchestrator._completed_roots)
        extras_completed = 0 if orchestrator is None else len(orchestrator._completed_extras)
        generation_receipt_issued = bool(orchestrator is not None and orchestrator._generation_receipt is not None)
        if manager is None and manager_state is NativeManagerState.NEVER_LOADED and failure_boundary == "model_load":
            manager_state = NativeManagerState.LOAD_FAILED
        abort_receipt = _NativeAbortReceipt(
            authority=_NATIVE_ABORT_AUTHORITY,
            provenance=provenance,
            attempt_start=attempt_start,
            failure_boundary=failure_boundary,
            reason_code=_native_abort_reason(failure, failure_boundary=failure_boundary),
            failure_message_sha256=hashlib.sha256(str(failure).encode("utf-8")).hexdigest(),
            generation_receipt_issued=generation_receipt_issued,
            completed_root_generation_count=roots_completed,
            completed_unique_extra_generation_count=extras_completed,
            hidden_evaluator_constructed=hidden_evaluator_constructed,
            hidden_evaluation_started=hidden_evaluation_started,
            manager_state=manager_state,
            work_root_cleanup=cleanup_state,
            post_abort_verification=post_abort_verification,
            successful_verifications=verification_ledger.successful_verifications,
        )
        abort_envelope = RepositoryQualityPilotAbortEnvelope(
            provenance=provenance,
            attempt_start=attempt_start,
            abort_receipt=abort_receipt,
        )
        _verify_attempt_start_file(claimed_attempt_root, attempt_start)
        _atomic_create_result(
            claimed_attempt_root / _ATTEMPT_ABORT_FILENAME,
            serialize_repository_quality_pilot_abort(abort_envelope),
        )
        raise NativePilotAborted(abort_envelope) from None

    if result_envelope is None:  # pragma: no cover - defensive state-machine guard
        raise RepositoryPilotProtocolError("native pilot produced no terminal result envelope")
    return result_envelope


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("smoke",),
        default="smoke",
        help="v3 authorizes only the four-task harness-validation smoke cohort",
    )
    parser.add_argument("--tier", default="small")
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
    parser.add_argument(
        "--work-parent",
        type=Path,
        default=None,
        help="optional parent for an automatically removed private run directory",
    )
    parser.add_argument(
        "--attempt-root",
        type=Path,
        required=True,
        help="new persistent directory for create-once start and terminal receipts",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    temporary_parent = None if args.work_parent is None else args.work_parent.expanduser()
    if temporary_parent is not None:
        try:
            resolved_parent = temporary_parent.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RepositoryPilotProtocolError("work parent is unavailable") from exc
        if temporary_parent.is_symlink() or not resolved_parent.is_dir():
            raise RepositoryPilotProtocolError("work parent must be a real directory")
        preliminary_locks = (
            LocalModelLock(
                role="target",
                repository_label=TARGET_REPOSITORY_LABEL,
                content_identity=TARGET_CONTENT_IDENTITY,
                resolved_path=args.target_path.expanduser().resolve(strict=True),
            ),
            LocalModelLock(
                role="drafter",
                repository_label=DRAFT_REPOSITORY_LABEL,
                content_identity=DRAFT_CONTENT_IDENTITY,
                resolved_path=args.draft_path.expanduser().resolve(strict=True),
            ),
        )
        temporary_parent = _validate_native_work_location(
            resolved_parent,
            source_root=_REPOSITORY_ROOT,
            model_locks=preliminary_locks,
            label="work parent",
            must_exist=True,
        )

    with tempfile.TemporaryDirectory(
        prefix="mio-repository-quality-pilot-",
        dir=temporary_parent,
    ) as private_parent:
        envelope = run_native_repository_quality_pilot(
            split=args.split,
            tier=args.tier,
            config_path=args.config,
            target_path=args.target_path,
            draft_path=args.draft_path,
            work_root=Path(private_parent) / "run",
            attempt_root=args.attempt_root,
            output=args.output,
        )
    serialized = serialize_repository_quality_pilot_result(envelope)
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        verified_output = _validate_new_output_path(
            args.output,
            source_root=envelope.provenance.source_lock.repo_root,
            model_locks=envelope.provenance.model_locks,
        )
        if verified_output is None:  # pragma: no cover - guarded by args.output
            raise RepositoryPilotProtocolError("pilot output path disappeared")
        _atomic_create_result(verified_output, serialized)
    return 0


__all__ = (
    "FrozenPilotProvenance",
    "PilotExecution",
    "PilotRunPhase",
    "NativePilotAborted",
    "RepositoryQualityPilotAbortEnvelope",
    "RepositoryQualityPilotResultEnvelope",
    "RepositoryQualityPilotOrchestrator",
    "bind_records_from_hidden_batch",
    "execute_repository_quality_pilot",
    "main",
    "run_native_repository_quality_pilot",
    "serialize_repository_quality_pilot_abort",
    "serialize_repository_quality_pilot_result",
)


if __name__ == "__main__":
    raise SystemExit(main())
