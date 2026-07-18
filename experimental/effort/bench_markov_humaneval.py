#!/usr/bin/env python3
"""Two-phase HumanEval harness for Mio's Markov effort controller.

The protocol has a hard channel boundary:

* ``calibrate`` explores candidates using only :class:`PublicHumanEvalCase`,
  public validation, and deterministic action/depth schedules.  Hidden tests
  are not run until generation for *all* calibration tasks is complete.
* ``evaluate`` loads the frozen calibration artifact, verifies its exact
  experiment identity, and runs the real :class:`MarkovTreeEffortController`
  for ``low``, ``medium``, ``high``, ``xhigh``, and ``ultra``.  The hidden
  evaluator is called exactly once after terminal selection for each strategy.

Calibration outcomes never appear in prompts or request-time state.  A
transition's quality label is net terminal change (+1 rescue, -1 regression,
0 unchanged), and every underpowered action/depth/trigger stratum is omitted
from the published transition table and reported explicitly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from experimental.effort.calibration import (
    FrozenUncertaintyCalibrator,
    TransitionCalibrationObservation,
    UncertaintyCalibrationObservation,
    build_frozen_transition_model,
    fit_isotonic_uncertainty,
    frozen_transition_model_from_mapping,
    frozen_transition_model_to_mapping,
)
from experimental.effort.humaneval import (
    CALIBRATION_TASKS,
    HUMANEVAL_REVISION,
    HUMANEVAL_SHA256,
    SPLIT_SALT,
    HumanEvalCase,
    PublicHumanEvalCase,
    PublicValidationResult,
    corpus_manifest,
    fetch_humaneval,
    load_humaneval,
    split_humaneval,
    validate_candidate_public,
    verify_candidate,
)
from experimental.effort.markov_runner import (
    CandidateGenerator,
    GeneratedCandidate,
    HiddenEvaluationResult,
    MarkovEffortRun,
    PublicGenerationFeedback,
    run_markov_effort,
)
from experimental.effort.model_identity import (
    ModelIdentityError,
    ResolvedModelReference,
    resolve_model_reference,
)
from experimental.effort.statistics_v2 import (
    EffortStatisticsRow,
    PreregisteredGatePolicy,
    RunProvenance,
    analyze_paired_rows,
    evaluate_preregistered_gate,
)
from experimental.markov_effort_controller import (
    CalibrationIdentity,
    ControllerAction,
    EFFORT_PROFILES,
    EXTRA_ACTIONS,
    EffortTier,
    FrozenTransitionModel,
    MarkovTreeEffortController,
    Trigger,
    ValidationOutcome,
    deterministic_generation_seed,
)


CALIBRATION_ARTIFACT_SCHEMA = "mio.markov-effort-calibration.v1"
EVALUATION_SCHEMA = "mio.markov-effort-humaneval-evaluation.v1"
PROTOCOL_REVISION = "mio-markov-humaneval-two-phase-v2"
EXPECTED_HUMANEVAL_TASKS = 164
EXPECTED_HELDOUT_TASKS = EXPECTED_HUMANEVAL_TASKS - CALIBRATION_TASKS
OFFICIAL_FULL_MANIFEST_SHA256 = "8a99055becc53543c0553b340b5dc1c3a964f37e4b7c2f8d581dca73de92d79d"
OFFICIAL_CALIBRATION_MANIFEST_SHA256 = "a3e588c4f625d4a7f911ce108eca03d886cd5cafd86f9452ae2f13ba8243fefb"
OFFICIAL_HELDOUT_MANIFEST_SHA256 = "cfbcdb420dd9d269b184dbb8f2c97d9c0994270c6828757fc0d30270e8b2c3ef"
VERIFIER_PARITY_CERTIFICATE_SCHEMA = "mio.humaneval-verifier-parity.v3"
VERIFIER_PARITY_CERTIFICATE_RELATIVE_PATH = (
    "benchmarks/results/humaneval-verifier-parity-962ad90.json"
)
VERIFIER_PARITY_CERTIFICATE_PATH = (
    Path(__file__).resolve().parents[2] / VERIFIER_PARITY_CERTIFICATE_RELATIVE_PATH
)
VERIFIER_PARITY_CERTIFICATE_SHA256 = (
    "cf83439e7dbdbe9f07a91f506f38e58f9fb8eedc367fcb42d9c21af7484ff982"
)
VERIFIER_PARITY_TIMEOUT_SECONDS = 10.0
VERIFIER_PARITY_CORPUS_MANIFEST_SHA256 = (
    "8a99055becc53543c0553b340b5dc1c3a964f37e4b7c2f8d581dca73de92d79d"
)
VERIFIER_PARITY_REFERENCE_MANIFEST_SHA256 = (
    "f88802dcce08968c3e76fa214334b9f08b8226144dd5dc50b5dbc4b321234664"
)
VERIFIER_PARITY_SOURCE_PATHS = frozenset(
    {
        "experimental/effort/humaneval.py",
        "experimental/effort/verify_humaneval_parity.py",
        "experimental/markov_effort_controller.py",
        "mio/agent.py",
        "mio/agent_policy.py",
    }
)
DEFAULT_MODEL = "models/Qwen3.6-27B-UD-Q4_K_XL-mlx"
DEFAULT_CONTEXT_BUCKET = "coding"
TIERS = (
    EffortTier.LOW,
    EffortTier.MEDIUM,
    EffortTier.HIGH,
    EffortTier.XHIGH,
    EffortTier.ULTRA,
)
PREREGISTRATION = {
    "protocol_revision": PROTOCOL_REVISION,
    "primary_endpoint": "paired_pass_at_1_delta_vs_low",
    "planned_comparisons": 4,
    "familywise_alpha": 0.05,
    "minimum_confirmatory_tasks": 100,
    "minimum_accuracy_delta": 0.05,
    "minimum_correct_completions_per_second_ratio": 0.95,
    "maximum_fast_path_overhead_ratio": 0.02,
    "maximum_e2e_latency_ratio": 1.15,
    "deadline_violations_allowed": 0,
    "resampling_unit": "task",
    "calibration_split_tasks": CALIBRATION_TASKS,
    "confirmatory_heldout_tasks": EXPECTED_HELDOUT_TASKS,
    "controller_visible": [
        "public_prompt",
        "public_validator_status",
        "public_validator_feedback",
        "calibrated_uncertainty",
        "frozen_transition_bounds",
    ],
    "evaluation_only": ["hidden_tests", "hidden_pass_fail", "aggregate_accuracy"],
    "verifier_parity_certificate_required": True,
}

_IDENTITY_FIELDS = {
    "model",
    "config",
    "prompt",
    "sampler",
    "corpus",
    "split",
    "backend",
}


class BenchmarkProtocolError(ValueError):
    """Raised when an experiment would violate the frozen protocol."""


class CalibrationHiddenEvaluator(Protocol):
    """Offline evaluator; the full case exists only behind this boundary."""

    def __call__(
        self,
        case: HumanEvalCase,
        completion: str,
        /,
    ) -> HiddenEvaluationResult: ...


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_regular_file_bounded(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise BenchmarkProtocolError("verifier parity certificate is not a regular file")
        if file_stat.st_size > max_bytes:
            raise BenchmarkProtocolError("verifier parity certificate exceeds the size limit")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read(max_bytes + 1)
    except BenchmarkProtocolError:
        raise
    except FileNotFoundError as exc:
        raise BenchmarkProtocolError("verifier parity certificate is missing") from exc
    except OSError as exc:
        raise BenchmarkProtocolError("cannot read verifier parity certificate") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > max_bytes:
        raise BenchmarkProtocolError("verifier parity certificate exceeds the size limit")
    return payload


def verifier_parity_certificate_identity(
    path: Path | None = None,
) -> dict[str, Any]:
    """Validate the committed 164/164 certificate against current sources.

    The exact report bytes are pinned, then its semantic claims and every
    transitive verifier source hash are checked again.  Missing, stale, or
    tampered evidence raises before calibration or evaluation can proceed.
    """

    certificate = path or VERIFIER_PARITY_CERTIFICATE_PATH
    try:
        payload = _read_regular_file_bounded(certificate, max_bytes=1_000_000)
        certificate_sha256 = hashlib.sha256(payload).hexdigest()
        if certificate_sha256 != VERIFIER_PARITY_CERTIFICATE_SHA256:
            raise BenchmarkProtocolError("verifier parity certificate digest mismatch")
        report = json.loads(payload)
    except BenchmarkProtocolError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkProtocolError("cannot read verifier parity certificate") from exc
    if not isinstance(report, Mapping):
        raise BenchmarkProtocolError("verifier parity certificate must be a mapping")
    if (
        set(report) != {"schema", "schema_version", "claim", "corpus", "verifier", "git", "tasks"}
        or report.get("schema") != VERIFIER_PARITY_CERTIFICATE_SCHEMA
        or report.get("schema_version") != 2
    ):
        raise BenchmarkProtocolError("verifier parity certificate schema mismatch")

    claim = report.get("claim")
    expected_claim = {
        "eligible": True,
        "parity": True,
        "passed": EXPECTED_HUMANEVAL_TASKS,
        "total": EXPECTED_HUMANEVAL_TASKS,
        "expected": EXPECTED_HUMANEVAL_TASKS,
        "ineligibility_reasons": [],
    }
    if claim != expected_claim:
        raise BenchmarkProtocolError("verifier parity certificate claim is not eligible 164/164")

    expected_task_ids = [f"HumanEval/{index}" for index in range(EXPECTED_HUMANEVAL_TASKS)]
    corpus = report.get("corpus")
    if not isinstance(corpus, Mapping):
        raise BenchmarkProtocolError("verifier parity certificate corpus is missing")
    manifest = corpus.get("manifest")
    reference_manifest = corpus.get("reference_manifest")
    if (
        corpus.get("revision") != HUMANEVAL_REVISION
        or corpus.get("archive_sha256") != HUMANEVAL_SHA256
        or not isinstance(manifest, Mapping)
        or manifest.get("tasks") != EXPECTED_HUMANEVAL_TASKS
        or manifest.get("task_ids") != expected_task_ids
        or manifest.get("manifest_sha256")
        != VERIFIER_PARITY_CORPUS_MANIFEST_SHA256
        or not isinstance(reference_manifest, Mapping)
        or reference_manifest.get("tasks") != EXPECTED_HUMANEVAL_TASKS
        or reference_manifest.get("task_ids") != expected_task_ids
        or reference_manifest.get("manifest_sha256")
        != VERIFIER_PARITY_REFERENCE_MANIFEST_SHA256
    ):
        raise BenchmarkProtocolError("verifier parity certificate corpus identity mismatch")

    verifier = report.get("verifier")
    if not isinstance(verifier, Mapping):
        raise BenchmarkProtocolError("verifier parity certificate source identity is missing")
    source_files = verifier.get("source_files")
    if (
        verifier.get("callable") != "experimental.effort.humaneval.verify_candidate"
        or verifier.get("source_path") != "experimental/effort/humaneval.py"
        or verifier.get("source_hash_scope") != "complete_module_files"
        or verifier.get("timeout_seconds_per_task")
        != VERIFIER_PARITY_TIMEOUT_SECONDS
        or not isinstance(source_files, Mapping)
        or set(source_files) != VERIFIER_PARITY_SOURCE_PATHS
        or any(not _is_sha256(digest) for digest in source_files.values())
        or verifier.get("source_sha256") != source_files.get("experimental/effort/humaneval.py")
        or verifier.get("source_bundle_sha256") != _canonical_sha256(dict(source_files))
    ):
        raise BenchmarkProtocolError("verifier parity certificate source bundle is malformed")
    repository_root = Path(__file__).resolve().parents[2]
    try:
        for relative_path, certified_digest in source_files.items():
            source_path = (repository_root / str(relative_path)).resolve(strict=True)
            if (
                repository_root not in source_path.parents
                or _file_sha256(source_path) != certified_digest
            ):
                raise BenchmarkProtocolError(
                    "verifier parity certificate source bundle is stale"
                )
    except BenchmarkProtocolError:
        raise
    except (OSError, RuntimeError) as exc:
        raise BenchmarkProtocolError(
            "verifier parity certificate source bundle cannot be verified"
        ) from exc

    git = report.get("git")
    if not isinstance(git, Mapping):
        raise BenchmarkProtocolError("verifier parity certificate Git identity is missing")
    certified_revision = git.get("revision")
    if (
        not isinstance(certified_revision, str)
        or len(certified_revision) != 40
        or any(character not in "0123456789abcdef" for character in certified_revision)
        or git.get("revision_after_verification") != certified_revision
        or git.get("dirty_before_verification") is not False
        or git.get("dirty_after_verification") is not False
        or git.get("revision_stable") is not True
    ):
        raise BenchmarkProtocolError("verifier parity certificate Git attestation is invalid")

    tasks = report.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_HUMANEVAL_TASKS:
        raise BenchmarkProtocolError("verifier parity certificate task rows are incomplete")
    reference_rows: list[dict[str, str]] = []
    expected_task_row_fields = {
        "task_id",
        "status",
        "passed",
        "prepared_source_match",
        "elapsed_seconds",
        "canonical_solution_sha256",
        "verified_source_sha256",
    }
    for expected_task_id, row in zip(expected_task_ids, tasks, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_task_row_fields
            or row.get("task_id") != expected_task_id
            or row.get("status") != "passed"
            or row.get("passed") is not True
            or row.get("prepared_source_match") is not True
            or not _is_sha256(row.get("canonical_solution_sha256"))
            or not _is_sha256(row.get("verified_source_sha256"))
            or not isinstance(row.get("elapsed_seconds"), (int, float))
            or isinstance(row.get("elapsed_seconds"), bool)
            or not math.isfinite(float(row["elapsed_seconds"]))
            or float(row["elapsed_seconds"]) < 0.0
        ):
            raise BenchmarkProtocolError("verifier parity certificate contains an invalid task row")
        reference_rows.append(
            {
                "task_id": expected_task_id,
                "canonical_solution_sha256": str(row["canonical_solution_sha256"]),
            }
        )
    if _canonical_sha256(reference_rows) != VERIFIER_PARITY_REFERENCE_MANIFEST_SHA256:
        raise BenchmarkProtocolError("verifier parity reference manifest does not match task rows")

    return {
        "validated": True,
        "schema": VERIFIER_PARITY_CERTIFICATE_SCHEMA,
        "path": VERIFIER_PARITY_CERTIFICATE_RELATIVE_PATH,
        "certificate_sha256": certificate_sha256,
        "certified_git_revision": certified_revision,
        "corpus_manifest_sha256": VERIFIER_PARITY_CORPUS_MANIFEST_SHA256,
        "reference_manifest_sha256": VERIFIER_PARITY_REFERENCE_MANIFEST_SHA256,
        "source_bundle_sha256": verifier["source_bundle_sha256"],
        "timeout_seconds_per_task": VERIFIER_PARITY_TIMEOUT_SECONDS,
        "passed": EXPECTED_HUMANEVAL_TASKS,
        "total": EXPECTED_HUMANEVAL_TASKS,
    }


def _identity_to_mapping(identity: CalibrationIdentity) -> dict[str, str]:
    return {field: getattr(identity, field) for field in sorted(_IDENTITY_FIELDS)}


def _identity_from_mapping(value: Any) -> CalibrationIdentity:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_FIELDS:
        raise BenchmarkProtocolError("calibration identity fields do not match the schema")
    if any(not isinstance(value[field], str) for field in _IDENTITY_FIELDS):
        raise BenchmarkProtocolError("calibration identity values must be strings")
    return CalibrationIdentity(**{field: value[field] for field in _IDENTITY_FIELDS})


def _provenance_to_mapping(provenance: RunProvenance) -> dict[str, Any]:
    if not isinstance(provenance, RunProvenance):
        raise TypeError("provenance must be RunProvenance")
    return asdict(provenance)


def _provenance_from_mapping(value: Any) -> RunProvenance:
    if not isinstance(value, Mapping):
        raise BenchmarkProtocolError("provenance must be a mapping")
    try:
        return RunProvenance.from_mapping(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkProtocolError("invalid experiment provenance") from exc


@dataclass(frozen=True)
class ExcludedTransitionStratum:
    """One expected stratum intentionally absent from the frozen model."""

    context_bucket: str
    trigger: Trigger
    depth: int
    action: ControllerAction
    task_clusters: int
    minimum_task_clusters: int
    reason: str = "underpowered"

    def __post_init__(self) -> None:
        if not self.context_bucket:
            raise ValueError("context_bucket must not be empty")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if self.action not in EXTRA_ACTIONS:
            raise ValueError("excluded action must generate an extra candidate")
        if self.task_clusters < 0 or self.minimum_task_clusters < 2:
            raise ValueError("invalid task-cluster counts")
        if self.reason == "underpowered" and self.task_clusters >= self.minimum_task_clusters:
            raise ValueError("an excluded stratum must be underpowered")
        if self.reason not in {"underpowered", "state_aliasing_off_policy"}:
            raise ValueError("unsupported excluded-stratum reason")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "context_bucket": self.context_bucket,
            "trigger": self.trigger.value,
            "depth": self.depth,
            "action": self.action.value,
            "task_clusters": self.task_clusters,
            "minimum_task_clusters": self.minimum_task_clusters,
            "reason": self.reason,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> ExcludedTransitionStratum:
        fields = {
            "context_bucket",
            "trigger",
            "depth",
            "action",
            "task_clusters",
            "minimum_task_clusters",
            "reason",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise BenchmarkProtocolError("excluded stratum fields do not match the schema")
        return cls(
            context_bucket=value["context_bucket"],
            trigger=Trigger(value["trigger"]),
            depth=value["depth"],
            action=ControllerAction(value["action"]),
            task_clusters=value["task_clusters"],
            minimum_task_clusters=value["minimum_task_clusters"],
            reason=value["reason"],
        )


@dataclass(frozen=True)
class CalibrationArtifact:
    """Frozen label-free objects and exact provenance for evaluation."""

    identity: CalibrationIdentity
    uncertainty_calibrator: FrozenUncertaintyCalibrator
    transition_model: FrozenTransitionModel
    provenance: RunProvenance
    calibration_manifest: Mapping[str, Any]
    protocol: Mapping[str, Any]
    excluded_strata: tuple[ExcludedTransitionStratum, ...]
    summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.uncertainty_calibrator.identity != self.identity:
            raise BenchmarkProtocolError("uncertainty calibrator identity mismatch")
        if self.transition_model.identity != self.identity:
            raise BenchmarkProtocolError("transition model identity mismatch")
        manifest_digest = self.calibration_manifest.get("manifest_sha256")
        if manifest_digest != self.provenance.task_manifest_sha256:
            raise BenchmarkProtocolError("calibration manifest/provenance mismatch")
        if self.provenance.leakage_detected:
            raise BenchmarkProtocolError("calibration artifact reports protocol leakage")
        if self.provenance.git_dirty:
            raise BenchmarkProtocolError("calibration artifact was produced from a dirty tree")
        if self.protocol.get("revision") != PROTOCOL_REVISION:
            raise BenchmarkProtocolError("calibration protocol revision mismatch")
        if self.protocol.get("phase") != "calibrate":
            raise BenchmarkProtocolError("calibration artifact has the wrong phase")
        parity_identity = verifier_parity_certificate_identity()
        if self.protocol.get("verifier_parity_certificate") != parity_identity:
            raise BenchmarkProtocolError("calibration verifier parity certificate mismatch")
        split_proof = self.protocol.get("pinned_split_proof")
        if not isinstance(split_proof, Mapping) or split_proof.get("verified") is not True:
            raise BenchmarkProtocolError("calibration artifact lacks a verified pinned split")
        if split_proof.get("disjoint") is not True:
            raise BenchmarkProtocolError("calibration artifact split is not disjoint")
        if split_proof.get("calibration") != dict(self.calibration_manifest):
            raise BenchmarkProtocolError("calibration artifact split manifest mismatch")
        source_hashes = self.protocol.get("source_sha256")
        if source_hashes != _source_hashes():
            raise BenchmarkProtocolError("calibration artifact source hashes are stale or tampered")
        config = self.protocol.get("config")
        if not isinstance(config, Mapping):
            raise BenchmarkProtocolError("calibration artifact config is missing")
        try:
            settings = CalibrationConfig(**dict(config))
        except (TypeError, ValueError) as exc:
            raise BenchmarkProtocolError("calibration artifact config is invalid") from exc
        if (
            settings.hidden_evaluator_timeout_seconds
            != parity_identity["timeout_seconds_per_task"]
        ):
            raise BenchmarkProtocolError(
                "calibration timeout does not match verifier parity certificate"
            )
        if self.provenance.policy_sha256 != calibration_policy_sha256(settings):
            raise BenchmarkProtocolError("calibration policy provenance mismatch")
        static_hashes = expected_static_provenance_hashes()
        if self.provenance.scorer_sha256 != static_hashes["scorer_sha256"]:
            raise BenchmarkProtocolError("calibration scorer provenance mismatch")
        if self.provenance.verifier_sha256 != static_hashes["verifier_sha256"]:
            raise BenchmarkProtocolError("calibration verifier provenance mismatch")
        if self.provenance.preregistration_sha256 != static_hashes["preregistration_sha256"]:
            raise BenchmarkProtocolError("calibration preregistration provenance mismatch")
        object.__setattr__(self, "excluded_strata", tuple(self.excluded_strata))

    def to_mapping(self) -> dict[str, Any]:
        protocol = json.loads(json.dumps(self.protocol, sort_keys=True))
        return {
            "schema": CALIBRATION_ARTIFACT_SCHEMA,
            "identity": _identity_to_mapping(self.identity),
            "uncertainty_calibrator": self.uncertainty_calibrator.to_mapping(),
            "transition_model": frozen_transition_model_to_mapping(self.transition_model),
            "provenance": _provenance_to_mapping(self.provenance),
            "calibration_manifest": dict(self.calibration_manifest),
            "protocol": protocol,
            "excluded_strata": [row.to_mapping() for row in self.excluded_strata],
            "summary": dict(self.summary),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> CalibrationArtifact:
        fields = {
            "schema",
            "identity",
            "uncertainty_calibrator",
            "transition_model",
            "provenance",
            "calibration_manifest",
            "protocol",
            "excluded_strata",
            "summary",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise BenchmarkProtocolError("calibration artifact fields do not match the schema")
        if value["schema"] != CALIBRATION_ARTIFACT_SCHEMA:
            raise BenchmarkProtocolError("unsupported calibration artifact schema")
        if not isinstance(value["calibration_manifest"], Mapping):
            raise BenchmarkProtocolError("calibration manifest must be a mapping")
        if not isinstance(value["protocol"], Mapping):
            raise BenchmarkProtocolError("calibration protocol must be a mapping")
        if not isinstance(value["summary"], Mapping):
            raise BenchmarkProtocolError("calibration summary must be a mapping")
        excluded = value["excluded_strata"]
        if not isinstance(excluded, list):
            raise BenchmarkProtocolError("excluded strata must be a list")
        try:
            calibrator = FrozenUncertaintyCalibrator.from_mapping(
                value["uncertainty_calibrator"]
            )
            transition_model = frozen_transition_model_from_mapping(
                value["transition_model"]
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkProtocolError("invalid frozen calibration payload") from exc
        return cls(
            identity=_identity_from_mapping(value["identity"]),
            uncertainty_calibrator=calibrator,
            transition_model=transition_model,
            provenance=_provenance_from_mapping(value["provenance"]),
            calibration_manifest=dict(value["calibration_manifest"]),
            protocol=dict(value["protocol"]),
            excluded_strata=tuple(
                ExcludedTransitionStratum.from_mapping(row) for row in excluded
            ),
            summary=dict(value["summary"]),
        )


def save_calibration_artifact(artifact: CalibrationArtifact, path: Path) -> None:
    """Atomically write a canonical, human-readable calibration artifact."""

    if not isinstance(artifact, CalibrationArtifact):
        raise TypeError("artifact must be CalibrationArtifact")
    _write_json(path, artifact.to_mapping())


def load_calibration_artifact(
    path: Path,
    *,
    expected_identity: CalibrationIdentity,
) -> CalibrationArtifact:
    """Load an artifact and fail closed on any experiment-identity drift."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkProtocolError("cannot read calibration artifact") from exc
    artifact = CalibrationArtifact.from_mapping(value)
    if artifact.identity != expected_identity:
        raise BenchmarkProtocolError("calibration artifact identity mismatch")
    return artifact


@dataclass(frozen=True)
class CalibrationConfig:
    initial_max_output_tokens: int = 256
    min_task_clusters: int = 8
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    seed: int = 20260718
    context_bucket: str = DEFAULT_CONTEXT_BUCKET
    hidden_evaluator_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        positive_integers = (
            self.initial_max_output_tokens,
            self.bootstrap_resamples,
        )
        if any(type(value) is not int or value < 1 for value in positive_integers):
            raise ValueError("token limits and bootstrap resamples must be positive")
        if type(self.min_task_clusters) is not int or self.min_task_clusters < 2:
            raise ValueError("min_task_clusters must be at least two")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.context_bucket:
            raise ValueError("context_bucket must not be empty")
        if not 0.1 <= self.hidden_evaluator_timeout_seconds <= 30.0:
            raise ValueError("hidden evaluator timeout must be in [0.1, 30]")


@dataclass(frozen=True)
class _ExplorationCandidate:
    node_id: int
    depth: int
    action: ControllerAction
    tier: EffortTier | None
    context_bucket: str
    parent_node_id: int | None
    generated: GeneratedCandidate
    validation: PublicValidationResult
    observed_e2e_seconds: float
    action_history: tuple[ControllerAction, ...]


@dataclass(frozen=True)
class _CaseExploration:
    case: HumanEvalCase
    candidates: tuple[_ExplorationCandidate, ...]


def _generation_seed(
    task_id: str,
    node_id: int,
    action: ControllerAction,
) -> int:
    return deterministic_generation_seed(
        seed_salt="mio-markov-effort-v1",
        request_id=task_id,
        node_id=node_id,
        action=action,
    )


def _direct_feedback(case: PublicHumanEvalCase, config: CalibrationConfig) -> PublicGenerationFeedback:
    return PublicGenerationFeedback(
        action=ControllerAction.GENERATE_DIRECT,
        parent_node_id=None,
        parent_completion=None,
        validator_status=None,
        validator_feedback="",
        max_output_tokens=config.initial_max_output_tokens,
        max_additional_e2e_seconds=None,
        seed=_generation_seed(case.task_id, 0, ControllerAction.GENERATE_DIRECT),
    )


def _trigger_for_validation(validation: PublicValidationResult) -> Trigger | None:
    if validation.outcome is ValidationOutcome.FAIL:
        return Trigger.VALIDATOR_FAILURE
    if validation.outcome is ValidationOutcome.UNKNOWN:
        return Trigger.CALIBRATED_UNCERTAINTY
    return None


def _action_supported(trigger: Trigger, action: ControllerAction) -> bool:
    if trigger is Trigger.VALIDATOR_FAILURE:
        return action in EXTRA_ACTIONS
    return action in {
        ControllerAction.GENERATE_ALTERNATIVE,
        ControllerAction.GENERATE_REFINE,
    }


def _profile_context(context_bucket: str, tier: EffortTier) -> str:
    profile_digest = _canonical_sha256(asdict(EFFORT_PROFILES[tier]))
    return f"{context_bucket}:{tier.value}:{profile_digest}"


def _generate_calibration_exploration(
    cases: Sequence[HumanEvalCase],
    *,
    generator: CandidateGenerator,
    public_validator: Callable[[PublicHumanEvalCase, str], PublicValidationResult],
    config: CalibrationConfig,
) -> tuple[_CaseExploration, ...]:
    """Generate every public exploration before any hidden label is requested."""

    explorations: list[_CaseExploration] = []
    for case in cases:
        public = case.public
        direct_started = time.perf_counter()
        direct = generator(public, _direct_feedback(public, config))
        if not isinstance(direct, GeneratedCandidate):
            raise TypeError("generator must return GeneratedCandidate")
        if direct.metrics.output_tokens > config.initial_max_output_tokens:
            raise BenchmarkProtocolError("direct calibration candidate exceeded its token allocation")
        direct_validation = public_validator(public, direct.completion)
        if not isinstance(direct_validation, PublicValidationResult):
            raise TypeError("public_validator must return PublicValidationResult")
        direct_wall_seconds = time.perf_counter() - direct_started
        candidates: list[_ExplorationCandidate] = [
            _ExplorationCandidate(
                node_id=0,
                depth=0,
                action=ControllerAction.GENERATE_DIRECT,
                tier=None,
                context_bucket=config.context_bucket,
                parent_node_id=None,
                generated=direct,
                validation=direct_validation,
                observed_e2e_seconds=max(
                    direct_wall_seconds,
                    direct.metrics.total_seconds + direct_validation.elapsed_seconds,
                ),
                action_history=(),
            )
        ]
        direct_candidate = candidates[0]
        for tier in TIERS[1:]:
            profile = EFFORT_PROFILES[tier]
            context = _profile_context(config.context_bucket, tier)
            token_allocation = min(
                profile.max_extra_output_tokens,
                profile.max_output_tokens_per_candidate,
            )
            deadline = direct_candidate.observed_e2e_seconds * (
                profile.max_latency_ratio - 1.0
            )
            for action in profile.allowed_actions:
                feedback = PublicGenerationFeedback(
                    action=action,
                    parent_node_id=direct_candidate.node_id,
                    parent_completion=direct_candidate.generated.completion,
                    validator_status=direct_candidate.validation.status,
                    validator_feedback=direct_candidate.validation.feedback,
                    max_output_tokens=token_allocation,
                    max_additional_e2e_seconds=deadline,
                    seed=_generation_seed(case.task_id, 1, action),
                )
                candidate_started = time.perf_counter()
                generated = generator(public, feedback)
                if not isinstance(generated, GeneratedCandidate):
                    raise TypeError("generator must return GeneratedCandidate")
                if generated.metrics.output_tokens > token_allocation:
                    raise BenchmarkProtocolError(
                        "extra calibration candidate exceeded its token allocation"
                    )
                validation = public_validator(public, generated.completion)
                if not isinstance(validation, PublicValidationResult):
                    raise TypeError("public_validator must return PublicValidationResult")
                candidate_wall_seconds = time.perf_counter() - candidate_started
                observed_e2e_seconds = max(
                    candidate_wall_seconds,
                    generated.metrics.total_seconds + validation.elapsed_seconds,
                )
                if observed_e2e_seconds > deadline:
                    validation = replace(
                        validation,
                        outcome=ValidationOutcome.FAIL,
                        status="deadline_exceeded",
                        feedback="candidate_e2e_deadline_exceeded",
                    )
                candidate = _ExplorationCandidate(
                    node_id=len(candidates),
                    depth=1,
                    action=action,
                    tier=tier,
                    context_bucket=context,
                    parent_node_id=direct_candidate.node_id,
                    generated=generated,
                    validation=validation,
                    observed_e2e_seconds=observed_e2e_seconds,
                    action_history=(action,),
                )
                candidates.append(candidate)
        explorations.append(_CaseExploration(case=case, candidates=tuple(candidates)))
    return tuple(explorations)


def _best_public_candidate(
    candidates: Sequence[_ExplorationCandidate],
    calibrator: FrozenUncertaintyCalibrator,
) -> _ExplorationCandidate:
    validation_rank = {
        ValidationOutcome.FAIL: 0,
        ValidationOutcome.UNKNOWN: 1,
        ValidationOutcome.PASS: 2,
    }

    def rank(candidate: _ExplorationCandidate) -> tuple[float, ...]:
        raw_uncertainty = candidate.generated.raw_uncertainty
        uncertainty = (
            calibrator.transform(raw_uncertainty)
            if raw_uncertainty is not None
            else 1.0
        )
        return (
            float(validation_rank[candidate.validation.outcome]),
            -float(uncertainty),
            -float(candidate.generated.metrics.output_tokens),
            -float(candidate.node_id),
        )

    return max(candidates, key=rank)


def _expected_transition_keys(
    config: CalibrationConfig,
) -> tuple[tuple[str, Trigger, int, ControllerAction], ...]:
    keys: list[tuple[str, Trigger, int, ControllerAction]] = []
    for tier in TIERS[1:]:
        profile = EFFORT_PROFILES[tier]
        context = _profile_context(config.context_bucket, tier)
        for trigger in Trigger:
            for action in profile.allowed_actions:
                if _action_supported(trigger, action):
                    keys.append((context, trigger, 1, action))
    return tuple(keys)


def filter_underpowered_transition_observations(
    observations: Iterable[TransitionCalibrationObservation],
    *,
    expected_keys: Iterable[tuple[str, Trigger, int, ControllerAction]],
    min_task_clusters: int,
) -> tuple[
    tuple[TransitionCalibrationObservation, ...],
    tuple[ExcludedTransitionStratum, ...],
]:
    """Keep only independently powered strata and report every omission."""

    if type(min_task_clusters) is not int or min_task_clusters < 2:
        raise ValueError("min_task_clusters must be at least two")
    grouped: dict[
        tuple[str, Trigger, int, ControllerAction],
        list[TransitionCalibrationObservation],
    ] = defaultdict(list)
    for row in observations:
        key = (row.context_bucket, row.trigger, row.depth, row.action)
        grouped[key].append(row)

    included: list[TransitionCalibrationObservation] = []
    excluded: list[ExcludedTransitionStratum] = []
    expected = tuple(dict.fromkeys(expected_keys))
    unexpected = sorted(
        set(grouped).difference(expected),
        key=lambda item: (item[0], item[1].value, item[2], item[3].value),
    )
    if unexpected:
        raise BenchmarkProtocolError(f"unexpected transition calibration stratum: {unexpected[0]}")
    for context, trigger, depth, action in expected:
        rows = grouped.get((context, trigger, depth, action), [])
        clusters = {row.task_cluster_id for row in rows}
        if len(clusters) != len(rows):
            raise BenchmarkProtocolError(
                f"duplicate task cluster in {context}/{trigger.value}/d{depth}/{action.value}"
            )
        if len(clusters) < min_task_clusters:
            excluded.append(
                ExcludedTransitionStratum(
                    context_bucket=context,
                    trigger=trigger,
                    depth=depth,
                    action=action,
                    task_clusters=len(clusters),
                    minimum_task_clusters=min_task_clusters,
                )
            )
        else:
            included.extend(rows)
    return tuple(included), tuple(excluded)


def _candidate_e2e(candidate: _ExplorationCandidate) -> float:
    value = candidate.observed_e2e_seconds
    if value <= 0.0 or not math.isfinite(value):
        raise BenchmarkProtocolError("calibration candidate E2E latency must be positive")
    return value


def _pinned_split_proof(
    cases: Sequence[HumanEvalCase],
) -> dict[str, Any]:
    full = tuple(cases)
    if len(full) != EXPECTED_HUMANEVAL_TASKS:
        raise BenchmarkProtocolError(
            f"pinned HumanEval corpus must contain {EXPECTED_HUMANEVAL_TASKS} tasks"
        )
    if len({case.task_id for case in full}) != len(full):
        raise BenchmarkProtocolError("pinned HumanEval corpus has duplicate task ids")
    calibration = split_humaneval(full, "calibration")
    heldout = split_humaneval(full, "heldout")
    calibration_ids = {case.task_id for case in calibration}
    heldout_ids = {case.task_id for case in heldout}
    if calibration_ids & heldout_ids or calibration_ids | heldout_ids != {
        case.task_id for case in full
    }:
        raise BenchmarkProtocolError("pinned HumanEval split is not disjoint and exhaustive")
    proof = {
        "verified": True,
        "full": corpus_manifest(full),
        "calibration": corpus_manifest(calibration),
        "heldout": corpus_manifest(heldout),
        "disjoint": True,
    }
    expected = {
        "full": OFFICIAL_FULL_MANIFEST_SHA256,
        "calibration": OFFICIAL_CALIBRATION_MANIFEST_SHA256,
        "heldout": OFFICIAL_HELDOUT_MANIFEST_SHA256,
    }
    for split_name, digest in expected.items():
        if proof[split_name]["manifest_sha256"] != digest:
            raise BenchmarkProtocolError(
                f"{split_name} cases do not match the official pinned manifest"
            )
    return proof


def calibrate_markov_humaneval(
    *,
    cases: Sequence[HumanEvalCase],
    pinned_corpus: Sequence[HumanEvalCase],
    identity: CalibrationIdentity,
    provenance: RunProvenance,
    generator: CandidateGenerator,
    hidden_evaluator: CalibrationHiddenEvaluator,
    public_validator: Callable[
        [PublicHumanEvalCase, str], PublicValidationResult
    ] = validate_candidate_public,
    config: CalibrationConfig | None = None,
) -> CalibrationArtifact:
    """Fit frozen effort artifacts on the exact 32-task calibration split.

    The generation and hidden-evaluation loops are intentionally separate.
    A hidden result is never available while any model prompt is being built.
    """

    parity_identity = verifier_parity_certificate_identity()
    selected = tuple(cases)
    settings = config or CalibrationConfig()
    if (
        settings.hidden_evaluator_timeout_seconds
        != parity_identity["timeout_seconds_per_task"]
    ):
        raise BenchmarkProtocolError(
            "calibration timeout does not match verifier parity certificate"
        )
    split_proof = _pinned_split_proof(pinned_corpus)
    expected_calibration = split_humaneval(tuple(pinned_corpus), "calibration")
    if selected != expected_calibration:
        raise BenchmarkProtocolError("calibration cases do not match the pinned split exactly")
    if len(selected) != CALIBRATION_TASKS:
        raise BenchmarkProtocolError(
            f"calibration requires exactly {CALIBRATION_TASKS} pinned-split tasks"
        )
    if len({case.task_id for case in selected}) != len(selected):
        raise BenchmarkProtocolError("calibration task ids must be unique")
    expected_split_identity = f"{SPLIT_SALT}:calibration:{CALIBRATION_TASKS}"
    if identity.split != expected_split_identity:
        raise BenchmarkProtocolError("calibration identity does not name the pinned split")
    expected_corpus_identity = f"HumanEval@{HUMANEVAL_REVISION}:{HUMANEVAL_SHA256}"
    if identity.corpus != expected_corpus_identity:
        raise BenchmarkProtocolError("calibration identity does not name pinned HumanEval")
    manifest = corpus_manifest(selected)
    if provenance.task_manifest_sha256 != manifest["manifest_sha256"]:
        raise BenchmarkProtocolError("calibration provenance has the wrong task manifest")
    if provenance.leakage_detected:
        raise BenchmarkProtocolError("calibration provenance reports leakage")
    if provenance.git_dirty:
        raise BenchmarkProtocolError("calibration requires a clean git tree")
    if not provenance.test_split_id.startswith("HumanEval:calibration:"):
        raise BenchmarkProtocolError("calibration provenance has the wrong split id")
    if provenance.policy_sha256 != calibration_policy_sha256(settings):
        raise BenchmarkProtocolError("calibration provenance has the wrong policy hash")
    static_hashes = expected_static_provenance_hashes()
    for field, expected_digest in static_hashes.items():
        if getattr(provenance, field) != expected_digest:
            raise BenchmarkProtocolError(f"calibration provenance has the wrong {field}")

    # Phase A: no hidden evaluator is reachable from this function.
    explorations = _generate_calibration_exploration(
        selected,
        generator=generator,
        public_validator=public_validator,
        config=settings,
    )

    # Phase B starts only after generation for every case has completed.
    hidden: dict[tuple[str, int], HiddenEvaluationResult] = {}
    hidden_calls = 0
    for exploration in explorations:
        for candidate in exploration.candidates:
            result = hidden_evaluator(
                exploration.case,
                candidate.generated.completion,
            )
            if not isinstance(result, HiddenEvaluationResult):
                raise TypeError("hidden_evaluator must return HiddenEvaluationResult")
            key = (exploration.case.task_id, candidate.node_id)
            if key in hidden:
                raise BenchmarkProtocolError("calibration candidate was evaluated twice")
            hidden[key] = result
            hidden_calls += 1

    uncertainty_rows: list[UncertaintyCalibrationObservation] = []
    for exploration in explorations:
        by_id = {candidate.node_id: candidate for candidate in exploration.candidates}
        direct = by_id[0]
        raw_uncertainty = direct.generated.raw_uncertainty
        if raw_uncertainty is None:
            raise BenchmarkProtocolError(
                f"direct candidate {exploration.case.task_id} has no raw uncertainty"
            )
        direct_hidden = hidden[(exploration.case.task_id, 0)]
        uncertainty_rows.append(
            UncertaintyCalibrationObservation(
                task_cluster_id=exploration.case.task_id,
                raw_uncertainty=raw_uncertainty,
                is_error=not direct_hidden.passed,
            )
        )

    calibrator = fit_isotonic_uncertainty(
        identity,
        uncertainty_rows,
        min_task_clusters=settings.min_task_clusters,
    )

    transition_rows: list[TransitionCalibrationObservation] = []
    skipped_zero_token_rows = 0
    selection_counts: dict[str, int] = defaultdict(int)
    for exploration in explorations:
        by_id = {candidate.node_id: candidate for candidate in exploration.candidates}
        direct = by_id[0]
        direct_e2e = _candidate_e2e(direct)
        for candidate in exploration.candidates[1:]:
            if candidate.tier is None or candidate.parent_node_id != direct.node_id:
                raise BenchmarkProtocolError("tier calibration candidate has an invalid root")
            profile = EFFORT_PROFILES[candidate.tier]
            trigger = _trigger_for_validation(direct.validation)
            if trigger is Trigger.CALIBRATED_UNCERTAINTY:
                direct_raw = direct.generated.raw_uncertainty
                if direct_raw is None:
                    raise BenchmarkProtocolError("direct uncertainty disappeared after fitting")
                if calibrator.transform(direct_raw) < profile.uncertainty_threshold:
                    continue
            if (
                trigger is None
                or candidate.action not in profile.allowed_actions
                or not _action_supported(trigger, candidate.action)
            ):
                continue
            if candidate.generated.metrics.output_tokens < 1:
                skipped_zero_token_rows += 1
                continue
            selected_before = _best_public_candidate((direct,), calibrator)
            selected_after = _best_public_candidate((direct, candidate), calibrator)
            before_hidden = hidden[(exploration.case.task_id, selected_before.node_id)]
            after_hidden = hidden[(exploration.case.task_id, selected_after.node_id)]
            quality_delta = float(int(after_hidden.passed) - int(before_hidden.passed))
            selection_counts[
                f"{candidate.context_bucket}/{trigger.value}/{candidate.action.value}/"
                f"{selected_before.node_id}->{selected_after.node_id}"
            ] += 1
            transition_rows.append(
                TransitionCalibrationObservation(
                    task_cluster_id=exploration.case.task_id,
                    context_bucket=candidate.context_bucket,
                    trigger=trigger,
                    depth=candidate.depth,
                    action=candidate.action,
                    rescued=(quality_delta == 1.0),
                    quality_delta=quality_delta,
                    extra_output_tokens=candidate.generated.metrics.output_tokens,
                    direct_e2e_seconds=direct_e2e,
                    extra_e2e_seconds=_candidate_e2e(candidate),
                )
            )
    expected_keys = _expected_transition_keys(settings)
    publishable_rows = tuple(row for row in transition_rows if row.depth == 1)
    included, underpowered = filter_underpowered_transition_observations(
        publishable_rows,
        expected_keys=(key for key in expected_keys if key[2] == 1),
        min_task_clusters=settings.min_task_clusters,
    )
    deeper_groups: dict[
        tuple[str, Trigger, int, ControllerAction], set[str]
    ] = defaultdict(set)
    for row in transition_rows:
        if row.depth > 1:
            deeper_groups[
                (row.context_bucket, row.trigger, row.depth, row.action)
            ].add(row.task_cluster_id)
    structural = tuple(
        ExcludedTransitionStratum(
            context_bucket=context,
            trigger=trigger,
            depth=depth,
            action=action,
            task_clusters=len(deeper_groups.get((context, trigger, depth, action), set())),
            minimum_task_clusters=settings.min_task_clusters,
            reason="state_aliasing_off_policy",
        )
        for context, trigger, depth, action in expected_keys
        if depth > 1
    )
    excluded = (*underpowered, *structural)
    if included:
        mean_bound_model = build_frozen_transition_model(
            identity,
            included,
            min_task_clusters=settings.min_task_clusters,
            resamples=settings.bootstrap_resamples,
            confidence_level=settings.confidence_level,
            seed=settings.seed,
        )
        included_by_key: dict[
            tuple[str, Trigger, int, ControllerAction],
            list[TransitionCalibrationObservation],
        ] = defaultdict(list)
        for row in included:
            included_by_key[
                (row.context_bucket, row.trigger, row.depth, row.action)
            ].append(row)
        transition_model = FrozenTransitionModel(
            identity,
            (
                replace(
                    estimate,
                    # Runtime budget checks require a per-observation
                    # conservative envelope, not a CI for the mean cost.
                    extra_output_tokens_ucb=max(
                        row.extra_output_tokens
                        for row in included_by_key[
                            (
                                estimate.context_bucket,
                                estimate.trigger,
                                estimate.depth,
                                estimate.action,
                            )
                        ]
                    ),
                    extra_e2e_latency_ratio_ucb=max(
                        row.extra_e2e_latency_ratio
                        for row in included_by_key[
                            (
                                estimate.context_bucket,
                                estimate.trigger,
                                estimate.depth,
                                estimate.action,
                            )
                        ]
                    ),
                )
                for estimate in mean_bound_model.estimates
            ),
        )
    else:
        transition_model = FrozenTransitionModel(identity)

    protocol = {
        "revision": PROTOCOL_REVISION,
        "phase": "calibrate",
        "source_sha256": _source_hashes(),
        "pinned_split_proof": split_proof,
        "public_generation_only": True,
        "hidden_evaluation_after_all_generation": True,
        "hidden_labels_serialized": False,
        "verifier_parity_certificate": parity_identity,
        "quality_delta": "+1 rescue, -1 regression, 0 unchanged",
        "behavior_policy": "tier-specific_depth1_counterfactuals",
        "parent_rules": {
            "generate_repair": "direct_root_at_depth1",
            "generate_alternative": "direct_root",
            "generate_refine": "public_best_is_direct_at_depth1",
        },
        "action_history_recorded_during_calibration": True,
        "published_transition_depths": [1],
        "deeper_transition_status": (
            "diagnostic_only: controller state key omits action history, so depth>1 "
            "behavior-policy estimates are not published"
        ),
        "profile_contexts": {
            tier.value: _profile_context(settings.context_bucket, tier)
            for tier in TIERS[1:]
        },
        "profiles": {
            tier.value: asdict(EFFORT_PROFILES[tier]) for tier in TIERS
        },
        "cost_envelope": "maximum_observed_task_cost_per_stratum",
        "config": asdict(settings),
    }
    summary = {
        "tasks": len(selected),
        "generated_candidates": sum(len(run.candidates) for run in explorations),
        "hidden_evaluations": hidden_calls,
        "uncertainty_observations": len(uncertainty_rows),
        "transition_observations_observed": len(transition_rows),
        "transition_observations_state_aliasing_excluded": sum(
            row.depth > 1 for row in transition_rows
        ),
        "transition_observations_published": len(included),
        "transition_strata_published": len(transition_model.estimates),
        "transition_strata_excluded": len(excluded),
        "zero_output_transition_rows_excluded": skipped_zero_token_rows,
        "public_selector_transitions": dict(sorted(selection_counts.items())),
    }
    return CalibrationArtifact(
        identity=identity,
        uncertainty_calibrator=calibrator,
        transition_model=transition_model,
        provenance=provenance,
        calibration_manifest=manifest,
        protocol=protocol,
        excluded_strata=excluded,
        summary=summary,
    )


def _feedback_cache_key(
    case: PublicHumanEvalCase,
    feedback: PublicGenerationFeedback,
) -> tuple[Any, ...]:
    parent_digest = (
        hashlib.sha256(feedback.parent_completion.encode("utf-8")).hexdigest()
        if feedback.parent_completion is not None
        else None
    )
    return (
        case.task_id,
        hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
        case.entry_point,
        feedback.action.value,
        feedback.parent_node_id,
        parent_digest,
        feedback.validator_status,
        feedback.validator_feedback,
        feedback.max_output_tokens,
        feedback.max_additional_e2e_seconds,
        (
            None
            if feedback.action is ControllerAction.GENERATE_DIRECT
            else feedback.seed
        ),
    )


class MemoizingDirectGenerator:
    """Share deterministic direct generations across all five paired tiers."""

    def __init__(self, generator: CandidateGenerator, *, deterministic: bool) -> None:
        if not deterministic:
            raise BenchmarkProtocolError(
                "paired direct sharing requires a deterministic generator"
            )
        self._generator = generator
        self._cache: dict[tuple[Any, ...], GeneratedCandidate] = {}
        self._timing: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.backend_calls = 0
        self.direct_cache_hits = 0

    @property
    def wrapped(self) -> CandidateGenerator:
        return self._generator

    def _audit_length(self) -> int | None:
        records = getattr(self._generator, "audit_records", None)
        return len(records) if isinstance(records, tuple) else None

    def _capture_audit(self, key: tuple[Any, ...], before: int | None) -> None:
        records = getattr(self._generator, "audit_records", None)
        if before is None or not isinstance(records, tuple) or len(records) != before + 1:
            return
        record = records[-1]
        self._timing[key] = {
            "ttft_seconds": getattr(record, "ttft_seconds", None),
            "timing_method": getattr(record, "timing_method", "backend_audit"),
            "finish_reason": getattr(record, "finish_reason", None),
            "peak_memory_bytes": getattr(record, "peak_memory_bytes", None),
        }

    def __call__(
        self,
        case: PublicHumanEvalCase,
        feedback: PublicGenerationFeedback,
        /,
    ) -> GeneratedCandidate:
        key = _feedback_cache_key(case, feedback)
        if feedback.action is ControllerAction.GENERATE_DIRECT and key in self._cache:
            self.direct_cache_hits += 1
            return self._cache[key]
        before = self._audit_length()
        generated = self._generator(case, feedback)
        if not isinstance(generated, GeneratedCandidate):
            raise TypeError("generator must return GeneratedCandidate")
        self.backend_calls += 1
        self._capture_audit(key, before)
        if feedback.action is ControllerAction.GENERATE_DIRECT:
            self._cache[key] = generated
        return generated

    def timing_for(
        self,
        case: PublicHumanEvalCase,
        feedback: PublicGenerationFeedback,
    ) -> Mapping[str, Any]:
        return self._timing.get(
            _feedback_cache_key(case, feedback),
            {
                "ttft_seconds": None,
                "timing_method": "unavailable",
                "finish_reason": None,
                "peak_memory_bytes": None,
            },
        )


def _terminal_evaluator(
    case: HumanEvalCase,
    evaluator: CalibrationHiddenEvaluator,
) -> tuple[Callable[[PublicHumanEvalCase, str], HiddenEvaluationResult], Callable[[], int]]:
    calls = 0

    def evaluate(public: PublicHumanEvalCase, completion: str) -> HiddenEvaluationResult:
        nonlocal calls
        if public != case.public:
            raise BenchmarkProtocolError("hidden evaluator received the wrong public case")
        calls += 1
        if calls > 1:
            raise BenchmarkProtocolError("hidden evaluator called more than once for a strategy")
        result = evaluator(case, completion)
        if not isinstance(result, HiddenEvaluationResult):
            raise TypeError("hidden_evaluator must return HiddenEvaluationResult")
        return result

    return evaluate, lambda: calls


def _finite_rate(tokens: int, seconds: float) -> float | None:
    if tokens <= 0 or seconds <= 0.0:
        return None
    return tokens / seconds


def _serialize_run(
    run: MarkovEffortRun,
    generator: MemoizingDirectGenerator,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for trace in run.tree:
        timing = generator.timing_for(run.public_case, trace.generation_feedback)
        metrics = trace.generation_metrics
        node_e2e = metrics.total_seconds + trace.public_validation.elapsed_seconds + trace.controller_seconds
        nodes.append(
            {
                "node_id": trace.node_id,
                "parent_id": trace.parent_id,
                "action": trace.action.value,
                "completion": trace.completion,
                "completion_sha256": hashlib.sha256(
                    trace.completion.encode("utf-8", errors="replace")
                ).hexdigest(),
                "generation_feedback": {
                    "seed": trace.generation_feedback.seed,
                    "parent_node_id": trace.generation_feedback.parent_node_id,
                    "validator_status": trace.generation_feedback.validator_status,
                    "validator_feedback": trace.generation_feedback.validator_feedback,
                    "max_output_tokens": trace.generation_feedback.max_output_tokens,
                    "max_additional_e2e_seconds": (
                        trace.generation_feedback.max_additional_e2e_seconds
                    ),
                },
                "public_validation": {
                    "outcome": trace.public_validation.outcome.value,
                    "status": trace.public_validation.status,
                    "feedback": trace.public_validation.feedback,
                    "elapsed_seconds": trace.public_validation.elapsed_seconds,
                    "source_sha256": trace.public_validation.source_sha256,
                },
                "routing_validation": {
                    "outcome": trace.routing_validation.outcome.value,
                    "status": trace.routing_validation.status,
                    "feedback": trace.routing_validation.feedback,
                },
                "raw_uncertainty": trace.raw_uncertainty,
                "calibrated_uncertainty": trace.calibrated_uncertainty,
                "prompt_tokens": metrics.prompt_tokens,
                "output_tokens": metrics.output_tokens,
                "timed_decode_tokens": metrics.timed_decode_tokens,
                "ttft_seconds": timing["ttft_seconds"],
                "prefill_seconds": metrics.prefill_seconds,
                "decode_seconds": metrics.decode_seconds,
                "other_seconds": metrics.other_seconds,
                "generation_seconds": metrics.total_seconds,
                "controller_seconds": trace.controller_seconds,
                "runtime_e2e_seconds": node_e2e,
                "prefill_tokens_per_second": _finite_rate(
                    metrics.prompt_tokens, metrics.prefill_seconds
                ),
                "decode_tokens_per_second": _finite_rate(
                    metrics.timed_decode_tokens, metrics.decode_seconds
                ),
                "allocated_output_tokens": trace.allocated_output_tokens,
                "allocated_e2e_seconds": trace.allocated_e2e_seconds,
                "deadline_exceeded": trace.deadline_exceeded,
                "timing_method": timing["timing_method"],
                "finish_reason": timing["finish_reason"],
                "peak_memory_bytes": timing["peak_memory_bytes"],
            }
        )

    generation_seconds = sum(node.generation_metrics.total_seconds for node in run.tree)
    validation_seconds = sum(node.public_validation.elapsed_seconds for node in run.tree)
    runtime_e2e = generation_seconds + validation_seconds + run.controller_seconds
    if runtime_e2e <= 0.0:
        raise BenchmarkProtocolError("evaluation runtime E2E latency must be positive")
    prompt_tokens = sum(node.generation_metrics.prompt_tokens for node in run.tree)
    output_tokens = sum(node.generation_metrics.output_tokens for node in run.tree)
    timed_decode_tokens = sum(
        node.generation_metrics.timed_decode_tokens for node in run.tree
    )
    prefill_seconds = sum(node.generation_metrics.prefill_seconds for node in run.tree)
    decode_seconds = sum(node.generation_metrics.decode_seconds for node in run.tree)
    return {
        "tier": run.tier.value,
        "terminal_action": run.controller_state.terminal_action.value,
        "terminal_reason": run.controller_state.terminal_reason,
        "selected_node_id": run.controller_state.selected_node_id,
        "terminal_output_sha256": hashlib.sha256(
            run.terminal_output.encode("utf-8", errors="replace")
        ).hexdigest(),
        "hidden_terminal": {
            "passed": run.hidden_evaluation.passed,
            "score": run.hidden_evaluation.score,
            "status": run.hidden_evaluation.status,
            "elapsed_seconds": run.hidden_evaluation.elapsed_seconds,
        },
        "tree": nodes,
        "metrics": {
            "candidates": len(run.tree),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "timed_decode_tokens": timed_decode_tokens,
            "ttft_seconds": nodes[0]["ttft_seconds"],
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "generation_seconds": generation_seconds,
            "public_validation_seconds": validation_seconds,
            "controller_seconds": run.controller_seconds,
            "runtime_e2e_seconds": runtime_e2e,
            "benchmark_e2e_seconds": runtime_e2e + run.hidden_evaluation.elapsed_seconds,
            "prefill_tokens_per_second": _finite_rate(prompt_tokens, prefill_seconds),
            "decode_tokens_per_second": _finite_rate(
                timed_decode_tokens, decode_seconds
            ),
            "deadline_violations": len(run.deadline_overshoot_node_ids),
            "deadline_overshoot_node_ids": list(run.deadline_overshoot_node_ids),
        },
    }


def _claim_failures(
    *,
    split: str,
    limited: bool,
    task_count: int,
    provenance: RunProvenance,
    calibration_provenance: RunProvenance,
) -> tuple[str, ...]:
    failures: list[str] = []
    if split != "heldout":
        failures.append("not_heldout")
    if limited or task_count != EXPECTED_HELDOUT_TASKS:
        failures.append("limited_run")
    if provenance.git_dirty:
        failures.append("git_dirty")
    if calibration_provenance.git_dirty:
        failures.append("calibration_git_dirty")
    if provenance.leakage_detected:
        failures.append("leakage_detected")
    return tuple(failures)


def evaluate_markov_humaneval(
    *,
    cases: Sequence[HumanEvalCase],
    pinned_corpus: Sequence[HumanEvalCase],
    split: str,
    limited: bool,
    artifact: CalibrationArtifact,
    expected_identity: CalibrationIdentity,
    provenance: RunProvenance,
    generator: CandidateGenerator,
    hidden_evaluator: CalibrationHiddenEvaluator,
    generator_deterministic: bool,
    public_validator: Callable[
        [PublicHumanEvalCase, str], PublicValidationResult
    ] = validate_candidate_public,
    initial_max_output_tokens: int = 256,
    context_bucket: str = DEFAULT_CONTEXT_BUCKET,
    bootstrap_samples: int = 10_000,
    seed: int = 20260718,
    hidden_evaluator_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run five paired effort strategies against a frozen calibration artifact."""

    parity_identity = verifier_parity_certificate_identity()
    if artifact.protocol.get("verifier_parity_certificate") != parity_identity:
        raise BenchmarkProtocolError("evaluation verifier parity certificate mismatch")
    if (
        hidden_evaluator_timeout_seconds
        != parity_identity["timeout_seconds_per_task"]
    ):
        raise BenchmarkProtocolError(
            "evaluation timeout does not match verifier parity certificate"
        )
    selected = tuple(cases)
    pinned = tuple(pinned_corpus)
    split_proof = _pinned_split_proof(pinned)
    if split not in {"calibration", "heldout"}:
        raise ValueError("split must be calibration or heldout")
    if not selected or len({case.task_id for case in selected}) != len(selected):
        raise BenchmarkProtocolError("evaluation cases must be non-empty and unique")
    expected_split = split_humaneval(pinned, split)
    expected_by_id = {case.task_id: case for case in expected_split}
    if any(expected_by_id.get(case.task_id) != case for case in selected):
        raise BenchmarkProtocolError("evaluation cases are outside the pinned requested split")
    if not limited and selected != expected_split:
        raise BenchmarkProtocolError("full evaluation does not match the pinned split manifest")
    if artifact.calibration_manifest != split_proof["calibration"]:
        raise BenchmarkProtocolError("artifact calibration split differs from pinned corpus")
    if split == "heldout" and set(artifact.calibration_manifest["task_ids"]) & {
        case.task_id for case in selected
    }:
        raise BenchmarkProtocolError("heldout evaluation overlaps calibration tasks")
    if artifact.identity != expected_identity:
        raise BenchmarkProtocolError("calibration artifact identity mismatch")
    if artifact.uncertainty_calibrator.identity != expected_identity:
        raise BenchmarkProtocolError("uncertainty calibrator identity mismatch")
    if artifact.transition_model.identity != expected_identity:
        raise BenchmarkProtocolError("transition model identity mismatch")
    if initial_max_output_tokens < 1:
        raise ValueError("initial_max_output_tokens must be positive")
    artifact_config = artifact.protocol.get("config")
    if not isinstance(artifact_config, Mapping):
        raise BenchmarkProtocolError("calibration artifact has no protocol config")
    if artifact_config.get("initial_max_output_tokens") != initial_max_output_tokens:
        raise BenchmarkProtocolError("calibration/evaluation direct token budget mismatch")
    if artifact_config.get("context_bucket") != context_bucket:
        raise BenchmarkProtocolError("calibration/evaluation context bucket mismatch")
    if (
        artifact_config.get("hidden_evaluator_timeout_seconds")
        != hidden_evaluator_timeout_seconds
    ):
        raise BenchmarkProtocolError("calibration/evaluation hidden timeout mismatch")
    manifest = corpus_manifest(selected)
    if provenance.task_manifest_sha256 != manifest["manifest_sha256"]:
        raise BenchmarkProtocolError("evaluation provenance has the wrong task manifest")
    expected_policy_sha256 = evaluation_policy_sha256(
        initial_max_output_tokens=initial_max_output_tokens,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    if provenance.policy_sha256 != expected_policy_sha256:
        raise BenchmarkProtocolError("evaluation provenance has the wrong policy hash")
    static_hashes = expected_static_provenance_hashes()
    for field, expected_digest in static_hashes.items():
        if getattr(provenance, field) != expected_digest:
            raise BenchmarkProtocolError(f"evaluation provenance has the wrong {field}")

    memoized = MemoizingDirectGenerator(
        generator,
        deterministic=generator_deterministic,
    )
    task_rows: list[dict[str, Any]] = []
    statistics_rows: list[EffortStatisticsRow] = []
    pending: list[tuple[HumanEvalCase, EffortTier, MarkovEffortRun]] = []

    def deferred_terminal(
        _case: PublicHumanEvalCase,
        _completion: str,
    ) -> HiddenEvaluationResult:
        # This is a type-compatible sentinel, not a hidden evaluator.  The
        # selected output is immutable once run_markov_effort returns.
        return HiddenEvaluationResult(
            score=0.0,
            passed=False,
            status="deferred_hidden_evaluation",
            elapsed_seconds=0.0,
        )

    # Generate and terminally select every strategy before the first hidden
    # test.  Hidden scoring therefore cannot affect routing, prompt order, or
    # model thermals for a later tier.
    for case in selected:
        for tier in TIERS:
            controller = MarkovTreeEffortController(
                tier=tier,
                transition_model=artifact.transition_model,
                calibration_identity=expected_identity,
                initial_max_output_tokens=initial_max_output_tokens,
            )
            run = run_markov_effort(
                case=case.public,
                controller=controller,
                generator=memoized,
                public_validator=public_validator,
                hidden_evaluator=deferred_terminal,
                calibrate_uncertainty=artifact.uncertainty_calibrator.transform,
                context_bucket=_profile_context(context_bucket, tier),
            )
            pending.append((case, tier, run))

    expected_direct_cache_hits = len(selected) * (len(TIERS) - 1)
    if memoized.direct_cache_hits != expected_direct_cache_hits:
        raise BenchmarkProtocolError(
            "paired tiers did not share exactly one deterministic direct generation per task"
        )

    tier_rows_by_task: dict[str, dict[str, Any]] = {
        case.task_id: {} for case in selected
    }
    hidden_calls = 0
    for case, tier, deferred_run in pending:
        hidden_result = hidden_evaluator(case, deferred_run.terminal_output)
        if not isinstance(hidden_result, HiddenEvaluationResult):
            raise TypeError("hidden_evaluator must return HiddenEvaluationResult")
        hidden_calls += 1
        run = replace(
            deferred_run,
            hidden_terminal_score=hidden_result.score,
            hidden_evaluation=hidden_result,
        )
        serialized = _serialize_run(run, memoized)
        tier_rows_by_task[case.task_id][tier.value] = serialized
        metrics = serialized["metrics"]
        statistics_rows.append(
            EffortStatisticsRow(
                task_id=case.task_id,
                strategy=tier.value,
                correct=run.hidden_evaluation.passed,
                e2e_seconds=metrics["runtime_e2e_seconds"],
                fast_path=(len(run.tree) == 1),
                deadline_violations=metrics["deadline_violations"],
            )
        )

    for case in selected:
        task_rows.append(
            {"task_id": case.task_id, "tiers": tier_rows_by_task[case.task_id]}
        )

    gate_policy = PreregisteredGatePolicy(planned_comparisons=4)
    claim_failures = _claim_failures(
        split=split,
        limited=limited,
        task_count=len(selected),
        provenance=provenance,
        calibration_provenance=artifact.provenance,
    )
    comparisons: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for comparison_index, tier in enumerate(TIERS[1:], start=1):
        selected_rows = [
            row for row in statistics_rows if row.strategy in {EffortTier.LOW.value, tier.value}
        ]
        statistics = analyze_paired_rows(
            selected_rows,
            baseline_strategy=EffortTier.LOW.value,
            candidate_strategy=tier.value,
            bootstrap_samples=bootstrap_samples,
            seed=seed + comparison_index,
        )
        gate = evaluate_preregistered_gate(
            statistics,
            provenance,
            policy=gate_policy,
        )
        comparisons[tier.value] = asdict(statistics)
        gates[tier.value] = {
            "statistical_gate": {
                "passed": gate.passed,
                "failures": [failure.value for failure in gate.failures],
                "corrected_alpha": gate.corrected_alpha,
            },
            "heldout_claim_passed": gate.passed and not claim_failures,
            "heldout_claim_failures": list(claim_failures),
        }

    artifact_mapping = artifact.to_mapping()
    return {
        "schema": EVALUATION_SCHEMA,
        "protocol_revision": PROTOCOL_REVISION,
        "phase": "evaluate",
        "split": split,
        "limited": limited,
        "tiers": [tier.value for tier in TIERS],
        "planned_comparisons": 4,
        "artifact_sha256": _canonical_sha256(artifact_mapping),
        "calibration_identity": _identity_to_mapping(expected_identity),
        "calibration_artifact_provenance": _provenance_to_mapping(
            artifact.provenance
        ),
        "evaluation_manifest": manifest,
        "pinned_split_proof": split_proof,
        "verifier_parity_certificate": parity_identity,
        "provenance": _provenance_to_mapping(provenance),
        "claim": {
            "eligible": not claim_failures,
            "failures": list(claim_failures),
            "requires_full_heldout_tasks": EXPECTED_HELDOUT_TASKS,
        },
        "summary": {
            "tasks": len(selected),
            "strategy_runs": len(selected) * len(TIERS),
            "hidden_terminal_evaluations": hidden_calls,
            "backend_generation_calls": memoized.backend_calls,
            "shared_direct_cache_hits": memoized.direct_cache_hits,
        },
        "comparisons_vs_low": comparisons,
        "gates": gates,
        "tasks": task_rows,
    }


def _verification_evaluator(timeout_s: float) -> CalibrationHiddenEvaluator:
    def evaluate(case: HumanEvalCase, completion: str) -> HiddenEvaluationResult:
        result = verify_candidate(case, completion, timeout_s=timeout_s)
        return HiddenEvaluationResult(
            score=float(result.passed),
            passed=result.passed,
            status=result.status,
            elapsed_seconds=result.elapsed_seconds,
        )

    return evaluate


def _git_revision() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BenchmarkProtocolError("git revision is unavailable") from exc
    return value


def _git_dirty() -> bool:
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return True
    return bool(output.strip())


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _runtime_identity(
    *,
    resolved_model: ResolvedModelReference,
    initial_max_output_tokens: int,
    renormalize_uncertainty_logprobs: bool = False,
    uncertainty_logprob_stride: int = 1,
) -> CalibrationIdentity:
    # Importing the adapter is cheap; model loading happens only after this
    # content-bound identity has been constructed from the shared resolution.
    from experimental.effort.mlx_backend import (
        MLXSamplerSettings,
        PROMPT_REVISION,
        TIMING_METHOD,
    )

    settings = MLXSamplerSettings(
        renormalize_uncertainty_logprobs=renormalize_uncertainty_logprobs,
        uncertainty_logprob_stride=uncertainty_logprob_stride,
    )
    return CalibrationIdentity(
        model=resolved_model.canonical_model_id,
        config=_canonical_sha256(
            {
                "protocol": PROTOCOL_REVISION,
                "initial_max_output_tokens": initial_max_output_tokens,
                "source_sha256": _source_hashes(),
            }
        ),
        prompt=PROMPT_REVISION,
        sampler=_canonical_sha256(asdict(settings)),
        corpus=f"HumanEval@{HUMANEVAL_REVISION}:{HUMANEVAL_SHA256}",
        split=f"{SPLIT_SALT}:calibration:{CALIBRATION_TASKS}",
        backend=(
            f"mlx={_package_version('mlx')};mlx-lm={_package_version('mlx-lm')};"
            f"timing={TIMING_METHOD}"
        ),
    )


def _build_generator(
    resolved_model: ResolvedModelReference,
    *,
    renormalize_uncertainty_logprobs: bool = False,
    uncertainty_logprob_stride: int = 1,
) -> CandidateGenerator:
    from experimental.effort.mlx_backend import MLXEffortGenerator, MLXSamplerSettings

    return MLXEffortGenerator.from_pretrained(
        resolved_model.load_model_id,
        revision=resolved_model.load_revision,
        audited_model_id=resolved_model.canonical_model_id,
        lazy=False,
        settings=MLXSamplerSettings(
            renormalize_uncertainty_logprobs=renormalize_uncertainty_logprobs,
            uncertainty_logprob_stride=uncertainty_logprob_stride,
        ),
    )


def _source_hashes() -> dict[str, str]:
    files = {
        "harness": Path(__file__),
        "controller": Path(__file__).parents[1] / "markov_effort_controller.py",
        "calibration": Path(__file__).with_name("calibration.py"),
        "runner": Path(__file__).with_name("markov_runner.py"),
        "humaneval": Path(__file__).with_name("humaneval.py"),
        "mlx_backend": Path(__file__).with_name("mlx_backend.py"),
        "model_identity": Path(__file__).with_name("model_identity.py"),
        "statistics": Path(__file__).with_name("statistics_v2.py"),
        "verifier_parity_certificate": VERIFIER_PARITY_CERTIFICATE_PATH,
    }
    return {name: _file_sha256(path) for name, path in files.items()}


def calibration_policy_sha256(config: CalibrationConfig) -> str:
    """Return the exact clean-source policy identity required by artifacts."""

    if not isinstance(config, CalibrationConfig):
        raise TypeError("config must be CalibrationConfig")
    sources = _source_hashes()
    return _canonical_sha256(
        {
            "policy": {"phase": "calibrate", "config": asdict(config)},
            "sources": sources,
        }
    )


def expected_static_provenance_hashes() -> dict[str, str]:
    humaneval_sha256 = _source_hashes()["humaneval"]
    return {
        "scorer_sha256": humaneval_sha256,
        "verifier_sha256": humaneval_sha256,
        "preregistration_sha256": _canonical_sha256(PREREGISTRATION),
    }


def evaluation_policy_sha256(
    *,
    initial_max_output_tokens: int,
    bootstrap_samples: int,
    seed: int,
) -> str:
    policy = {
        "phase": "evaluate",
        "tiers": [tier.value for tier in TIERS],
        "initial_max_output_tokens": initial_max_output_tokens,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
    return _canonical_sha256({"policy": policy, "sources": _source_hashes()})


def _make_provenance(
    *,
    model_revision: str,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    split: str,
    leakage_detected: bool,
) -> RunProvenance:
    hashes = _source_hashes()
    return RunProvenance(
        git_revision=_git_revision(),
        git_dirty=_git_dirty(),
        model_revision=model_revision,
        policy_sha256=_canonical_sha256({"policy": policy, "sources": hashes}),
        task_manifest_sha256=str(manifest["manifest_sha256"]),
        scorer_sha256=hashes["humaneval"],
        verifier_sha256=hashes["humaneval"],
        preregistration_sha256=_canonical_sha256(PREREGISTRATION),
        test_split_id=f"HumanEval:{split}:{manifest['manifest_sha256']}",
        leakage_detected=leakage_detected,
    )


def _write_json(path: Path, value: Any) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    calibrate = subparsers.add_parser("calibrate", help="fit frozen calibration artifacts")
    calibrate.add_argument("--model", default=DEFAULT_MODEL)
    calibrate.add_argument(
        "--model-revision",
        required=True,
        help="full Hugging Face commit or local-sha256-v1:<digest>",
    )
    calibrate.add_argument("--corpus", type=Path)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--initial-max-output-tokens", type=int, default=256)
    calibrate.add_argument("--min-task-clusters", type=int, default=8)
    calibrate.add_argument("--bootstrap-resamples", type=int, default=10_000)
    calibrate.add_argument("--seed", type=int, default=20260718)
    calibrate.add_argument("--validator-timeout", type=float, default=10.0)
    calibrate.add_argument(
        "--renormalize-uncertainty-logprobs",
        action="store_true",
        help="opt in to FP32 full-vocabulary uncertainty renormalization",
    )
    calibrate.add_argument(
        "--uncertainty-logprob-stride",
        type=int,
        default=1,
        help="sample every Nth selected-token log probability (FP32 pilot: 8)",
    )

    evaluate = subparsers.add_parser("evaluate", help="run paired five-tier evaluation")
    evaluate.add_argument("--model", default=DEFAULT_MODEL)
    evaluate.add_argument(
        "--model-revision",
        required=True,
        help="must exactly match the frozen full commit or local fingerprint",
    )
    evaluate.add_argument("--corpus", type=Path)
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("calibration", "heldout"), default="heldout")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--initial-max-output-tokens", type=int, default=256)
    evaluate.add_argument("--bootstrap-samples", type=int, default=10_000)
    evaluate.add_argument("--seed", type=int, default=20260718)
    evaluate.add_argument("--validator-timeout", type=float, default=10.0)
    evaluate.add_argument(
        "--renormalize-uncertainty-logprobs",
        action="store_true",
        help="must exactly match the frozen calibration sampler",
    )
    evaluate.add_argument(
        "--uncertainty-logprob-stride",
        type=int,
        default=1,
        help="must exactly match the frozen calibration sampler",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _base_parser().parse_args(argv)
    parity_identity = verifier_parity_certificate_identity()
    if args.validator_timeout != parity_identity["timeout_seconds_per_task"]:
        raise SystemExit(
            "--validator-timeout must match the certified verifier timeout "
            f"({parity_identity['timeout_seconds_per_task']:g} seconds)"
        )
    try:
        resolved_model = resolve_model_reference(args.model, args.model_revision)
    except ModelIdentityError as exc:
        raise SystemExit(f"model identity error: {exc}") from exc
    corpus_path = args.corpus or fetch_humaneval()
    all_cases = load_humaneval(corpus_path)
    identity = _runtime_identity(
        resolved_model=resolved_model,
        initial_max_output_tokens=args.initial_max_output_tokens,
        renormalize_uncertainty_logprobs=args.renormalize_uncertainty_logprobs,
        uncertainty_logprob_stride=args.uncertainty_logprob_stride,
    )
    if args.phase == "calibrate":
        generator = _build_generator(
            resolved_model,
            renormalize_uncertainty_logprobs=args.renormalize_uncertainty_logprobs,
            uncertainty_logprob_stride=args.uncertainty_logprob_stride,
        )
        hidden_evaluator = _verification_evaluator(args.validator_timeout)
        cases = split_humaneval(all_cases, "calibration")
        config = CalibrationConfig(
            initial_max_output_tokens=args.initial_max_output_tokens,
            min_task_clusters=args.min_task_clusters,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
            hidden_evaluator_timeout_seconds=args.validator_timeout,
        )
        manifest = corpus_manifest(cases)
        provenance = _make_provenance(
            model_revision=resolved_model.canonical_model_id,
            manifest=manifest,
            policy={"phase": "calibrate", "config": asdict(config)},
            split="calibration",
            leakage_detected=False,
        )
        artifact = calibrate_markov_humaneval(
            cases=cases,
            pinned_corpus=all_cases,
            identity=identity,
            provenance=provenance,
            generator=generator,
            hidden_evaluator=hidden_evaluator,
            config=config,
        )
        save_calibration_artifact(artifact, args.output)
        print(
            f"[calibrate] artifact={args.output} tasks={len(cases)} "
            f"strata={len(artifact.transition_model.estimates)} dirty={provenance.git_dirty}",
            flush=True,
        )
        return 0

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    cases = split_humaneval(all_cases, args.split)
    if args.limit is not None:
        cases = cases[: args.limit]
    artifact = load_calibration_artifact(
        args.artifact,
        expected_identity=identity,
    )
    generator = _build_generator(
        resolved_model,
        renormalize_uncertainty_logprobs=args.renormalize_uncertainty_logprobs,
        uncertainty_logprob_stride=args.uncertainty_logprob_stride,
    )
    hidden_evaluator = _verification_evaluator(args.validator_timeout)
    manifest = corpus_manifest(cases)
    limited = args.limit is not None
    provenance = _make_provenance(
        model_revision=resolved_model.canonical_model_id,
        manifest=manifest,
        policy={
            "phase": "evaluate",
            "tiers": [tier.value for tier in TIERS],
            "initial_max_output_tokens": args.initial_max_output_tokens,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        split=args.split,
        leakage_detected=(args.split != "heldout"),
    )
    result = evaluate_markov_humaneval(
        cases=cases,
        pinned_corpus=all_cases,
        split=args.split,
        limited=limited,
        artifact=artifact,
        expected_identity=identity,
        provenance=provenance,
        generator=generator,
        hidden_evaluator=hidden_evaluator,
        generator_deterministic=True,
        initial_max_output_tokens=args.initial_max_output_tokens,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        hidden_evaluator_timeout_seconds=args.validator_timeout,
    )
    result["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.output, result)
    print(
        f"[evaluate] output={args.output} split={args.split} tasks={len(cases)} "
        f"claim_eligible={result['claim']['eligible']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
