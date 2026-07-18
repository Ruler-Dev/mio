#!/usr/bin/env python3
"""Source-free paired HumanEval ablation for MLX uncertainty scoring.

This benchmark changes only the uncertainty-scoring path: native selected-token
log probabilities at stride 1 are compared with FP32-renormalized log
probabilities at stride 8.  Both conditions receive the exact same public case,
greedy sampler, direct prompt, seed, and output-token budget.  All 64 candidates
are generated, their text/token/count/finish decode identity is confirmed, and
only then is the identical output verified once per task (32 hidden calls).

The JSON report contains aggregate statistics, timing, and hash commitments.
It never serializes prompts, completions, hidden tests, or per-task outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

from experimental.effort.bench_markov_humaneval import (
    BenchmarkProtocolError,
    VERIFIER_PARITY_CERTIFICATE_PATH,
    _git_dirty,
    _git_revision,
    _write_json,
    verifier_parity_certificate_identity,
)
from experimental.effort.humaneval import (
    CALIBRATION_TASKS,
    HUMANEVAL_REVISION,
    HUMANEVAL_SHA256,
    SPLIT_SALT,
    HumanEvalCase,
    corpus_manifest,
    fetch_humaneval,
    load_humaneval,
    split_humaneval,
    verify_candidate,
)
from experimental.effort.markov_runner import (
    GeneratedCandidate,
    HiddenEvaluationResult,
    PublicGenerationFeedback,
)
from experimental.effort.mlx_backend import (
    MLXEffortGenerator,
    MLXSamplerSettings,
    PROMPT_REVISION,
    TIMING_METHOD,
    UNCERTAINTY_METHOD_NATIVE,
    UNCERTAINTY_METHOD_RENORMALIZED,
)
from experimental.effort.model_identity import (
    LocalModelFingerprint,
    ModelIdentityError,
    ResolvedModelReference,
    fingerprint_local_model,
    resolve_model_reference,
)
from experimental.effort.uncertainty_statistics import (
    area_under_risk_coverage_curve,
    tie_corrected_auroc,
)
from experimental.markov_effort_controller import (
    ControllerAction,
    deterministic_generation_seed,
)


REPORT_SCHEMA = "mio.paired-uncertainty-humaneval.v2"
PROTOCOL_REVISION = "mio-paired-uncertainty-humaneval-v2"
OFFICIAL_CALIBRATION_MANIFEST_SHA256 = (
    "a3e588c4f625d4a7f911ce108eca03d886cd5cafd86f9452ae2f13ba8243fefb"
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
NATIVE_SIGNAL = "native-stride1"
RENORMALIZED_SIGNAL = "fp32-renormalized-stride8"
DEFAULT_MODEL = "models/Qwen3.6-27B-UD-Q4_K_XL-mlx"


class AuditedGenerator(Protocol):
    model_id: str
    settings: MLXSamplerSettings

    @property
    def audit_records(self) -> tuple[object, ...]: ...

    def __call__(
        self,
        case,
        feedback: PublicGenerationFeedback,
        /,
    ) -> GeneratedCandidate: ...


class HiddenEvaluator(Protocol):
    def __call__(
        self,
        case: HumanEvalCase,
        completion: str,
        /,
    ) -> HiddenEvaluationResult: ...


class IntegrityProbe(Protocol):
    def __call__(self) -> IntegritySnapshot: ...


@dataclass(frozen=True)
class IntegritySnapshot:
    """Content identity that must remain stable across model load and scoring."""

    git_revision: str
    git_dirty: bool
    source_sha256: Mapping[str, str]
    model_identity: str

    def __post_init__(self) -> None:
        if (
            len(self.git_revision) != 40
            or any(character not in "0123456789abcdef" for character in self.git_revision)
        ):
            raise BenchmarkProtocolError("Git revision must be a full commit digest")
        if type(self.git_dirty) is not bool:
            raise BenchmarkProtocolError("Git dirty state must be boolean")
        if not isinstance(self.source_sha256, Mapping) or not self.source_sha256:
            raise BenchmarkProtocolError("source identity must be a non-empty mapping")
        if any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in self.source_sha256.items()
        ):
            raise BenchmarkProtocolError("source identity contains an invalid digest")
        if not isinstance(self.model_identity, str) or not self.model_identity:
            raise BenchmarkProtocolError("model identity must be non-empty")
        object.__setattr__(self, "source_sha256", dict(self.source_sha256))


@dataclass(frozen=True)
class PairedUncertaintyConfig:
    max_output_tokens: int = 256
    seed: int = 20260718
    verifier_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(self.verifier_timeout_seconds, bool)
            or not isinstance(self.verifier_timeout_seconds, (int, float))
            or not math.isfinite(float(self.verifier_timeout_seconds))
            or self.verifier_timeout_seconds <= 0.0
        ):
            raise ValueError("verifier_timeout_seconds must be finite and positive")


@dataclass(frozen=True)
class _GeneratedCondition:
    generated: GeneratedCandidate
    audit: object
    output_sha256: str


@dataclass(frozen=True)
class _GeneratedPair:
    case: HumanEvalCase
    feedback: PublicGenerationFeedback
    native: _GeneratedCondition
    renormalized: _GeneratedCondition


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _source_hashes() -> dict[str, str]:
    here = Path(__file__)
    files = {
        "harness": here,
        "benchmark_protocol": here.with_name("bench_markov_humaneval.py"),
        "controller": here.parents[1] / "markov_effort_controller.py",
        "humaneval": here.with_name("humaneval.py"),
        "markov_runner": here.with_name("markov_runner.py"),
        "mlx_backend": here.with_name("mlx_backend.py"),
        "model_identity": here.with_name("model_identity.py"),
        "uncertainty_statistics": here.with_name("uncertainty_statistics.py"),
        "verifier_parity_certificate": VERIFIER_PARITY_CERTIFICATE_PATH,
    }
    return {name: _file_sha256(path) for name, path in files.items()}


def _capture_integrity_snapshot(model: str, revision: str) -> IntegritySnapshot:
    """Re-resolve every mutable input instead of trusting a prior label."""

    try:
        resolved = resolve_model_reference(model, revision)
    except ModelIdentityError as exc:
        raise BenchmarkProtocolError("model identity drift detected") from exc
    return IntegritySnapshot(
        git_revision=_git_revision(),
        git_dirty=_git_dirty(),
        source_sha256=_source_hashes(),
        model_identity=resolved.canonical_model_id,
    )


def _require_integrity(
    expected: IntegritySnapshot,
    probe: IntegrityProbe,
    *,
    phase: str,
) -> None:
    if not isinstance(expected, IntegritySnapshot):
        raise BenchmarkProtocolError("expected integrity snapshot is missing")
    observed = probe()
    if not isinstance(observed, IntegritySnapshot):
        raise BenchmarkProtocolError(f"integrity probe returned an invalid value at {phase}")
    if observed.git_dirty:
        raise BenchmarkProtocolError(f"Git tree became dirty at {phase}")
    if observed != expected:
        raise BenchmarkProtocolError(f"experiment integrity drift detected at {phase}")


def _validate_model_identity(resolved: ResolvedModelReference) -> None:
    if not isinstance(resolved, ResolvedModelReference):
        raise BenchmarkProtocolError("resolved_model must be content-bound")
    canonical = resolved.canonical_model_id
    if resolved.source_kind == "local":
        prefix = "local-mlx@local-sha256-v1:"
        digest = canonical.removeprefix(prefix)
        if not canonical.startswith(prefix) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise BenchmarkProtocolError("local model identity is not content-bound")
    elif resolved.source_kind == "huggingface":
        revision = canonical.rsplit("@", 1)[-1]
        if not canonical.startswith("hf://") or len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise BenchmarkProtocolError("remote model identity is not commit-bound")
    else:  # pragma: no cover - guarded by ResolvedModelReference's type
        raise BenchmarkProtocolError("unsupported model source")


def _expected_settings() -> tuple[MLXSamplerSettings, MLXSamplerSettings]:
    return (
        MLXSamplerSettings(
            uncertainty_logprob_stride=1,
            renormalize_uncertainty_logprobs=False,
        ),
        MLXSamplerSettings(
            uncertainty_logprob_stride=8,
            renormalize_uncertainty_logprobs=True,
        ),
    )


def _validate_generators(
    native: AuditedGenerator,
    renormalized: AuditedGenerator,
    *,
    model_id: str,
) -> None:
    expected_native, expected_renormalized = _expected_settings()
    if native.model_id != model_id or renormalized.model_id != model_id:
        raise BenchmarkProtocolError("both generators must use the resolved model identity")
    if native.settings != expected_native:
        raise BenchmarkProtocolError("native condition must be greedy stride-1 native scoring")
    if renormalized.settings != expected_renormalized:
        raise BenchmarkProtocolError(
            "candidate condition must differ only by FP32 renormalization at stride 8"
        )
    native_mapping = asdict(native.settings)
    candidate_mapping = asdict(renormalized.settings)
    differences = {
        key for key in native_mapping if native_mapping[key] != candidate_mapping[key]
    }
    if differences != {
        "uncertainty_logprob_stride",
        "renormalize_uncertainty_logprobs",
    }:
        raise BenchmarkProtocolError("condition sampler settings are not a focused ablation")
    if not native.settings.deterministic or not renormalized.settings.deterministic:
        raise BenchmarkProtocolError("paired uncertainty ablation requires greedy generation")


def _direct_feedback(case: HumanEvalCase, config: PairedUncertaintyConfig) -> PublicGenerationFeedback:
    seed = deterministic_generation_seed(
        seed_salt=f"{PROTOCOL_REVISION}:{config.seed}",
        request_id=case.task_id,
        node_id=0,
        action=ControllerAction.GENERATE_DIRECT,
    )
    return PublicGenerationFeedback(
        action=ControllerAction.GENERATE_DIRECT,
        parent_node_id=None,
        parent_completion=None,
        validator_status=None,
        validator_feedback="",
        max_output_tokens=config.max_output_tokens,
        max_additional_e2e_seconds=None,
        seed=seed,
    )


def _call_generator(
    generator: AuditedGenerator,
    case: HumanEvalCase,
    feedback: PublicGenerationFeedback,
    *,
    expected_method: str,
    expected_stride: int,
    expected_renormalized: bool,
) -> _GeneratedCondition:
    before = generator.audit_records
    generated = generator(case.public, feedback)
    after = generator.audit_records
    if not isinstance(generated, GeneratedCandidate):
        raise TypeError("generator must return GeneratedCandidate")
    if generated.raw_uncertainty is None:
        raise BenchmarkProtocolError("generator did not emit an uncertainty signal")
    if generated.metrics.output_tokens > feedback.max_output_tokens:
        raise BenchmarkProtocolError("generator exceeded the paired output-token budget")
    if len(after) != len(before) + 1 or after[:-1] != before:
        raise BenchmarkProtocolError("generator audit trail is not append-only one-record-per-call")
    audit = after[-1]
    output_sha256 = _text_sha256(generated.completion)
    checks = {
        "task_id": case.task_id,
        "action": ControllerAction.GENERATE_DIRECT,
        "model_id": generator.model_id,
        "seed": feedback.seed,
        "max_output_tokens": feedback.max_output_tokens,
        "deterministic_sampler": True,
        "output_text_sha256": output_sha256,
        "output_tokens": generated.metrics.output_tokens,
        "prompt_tokens": generated.metrics.prompt_tokens,
        "prefill_seconds": generated.metrics.prefill_seconds,
        "decode_seconds": generated.metrics.decode_seconds,
        "other_seconds": generated.metrics.other_seconds,
        "uncertainty_logprob_stride": expected_stride,
        "uncertainty_logprobs_renormalized": expected_renormalized,
        "uncertainty_method": expected_method,
    }
    for field, expected in checks.items():
        if getattr(audit, field, None) != expected:
            raise BenchmarkProtocolError(f"generator audit mismatch: {field}")
    if getattr(audit, "raw_uncertainty", None) != generated.raw_uncertainty:
        raise BenchmarkProtocolError("generator audit uncertainty mismatch")
    for field in (
        "prompt_source_sha256",
        "prompt_token_ids_sha256",
        "output_token_ids_sha256",
    ):
        digest = getattr(audit, field, None)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BenchmarkProtocolError(f"generator audit lacks {field}")
    finish_reason = getattr(audit, "finish_reason", None)
    if not isinstance(finish_reason, str) or not finish_reason:
        raise BenchmarkProtocolError("generator audit lacks finish_reason")
    return _GeneratedCondition(
        generated=generated,
        audit=audit,
        output_sha256=output_sha256,
    )


def _aggregate_generation(rows: Sequence[_GeneratedCondition]) -> dict[str, Any]:
    metrics = tuple(row.generated.metrics for row in rows)
    prefill = sum(value.prefill_seconds for value in metrics)
    decode = sum(value.decode_seconds for value in metrics)
    other = sum(value.other_seconds for value in metrics)
    prompt_tokens = sum(value.prompt_tokens for value in metrics)
    output_tokens = sum(value.output_tokens for value in metrics)
    timed_decode_tokens = sum(value.timed_decode_tokens for value in metrics)
    scoring = sum(float(getattr(row.audit, "uncertainty_scoring_seconds")) for row in rows)
    return {
        "calls": len(rows),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "timed_decode_tokens": timed_decode_tokens,
        "prefill_seconds": prefill,
        "decode_seconds": decode,
        "other_seconds": other,
        "generation_seconds": prefill + decode + other,
        "uncertainty_scoring_seconds": scoring,
        "mean_uncertainty_scoring_seconds": scoring / len(rows),
        "median_uncertainty_scoring_seconds": statistics.median(
            float(getattr(row.audit, "uncertainty_scoring_seconds")) for row in rows
        ),
        "prefill_tokens_per_second": prompt_tokens / prefill if prefill else None,
        "decode_tokens_per_second": timed_decode_tokens / decode if decode else None,
    }


def _aggregate_verification(rows: Sequence[HiddenEvaluationResult]) -> dict[str, Any]:
    elapsed = sum(row.elapsed_seconds for row in rows)
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.status] = statuses.get(row.status, 0) + 1
    return {
        "calls": len(rows),
        "passed": sum(row.passed for row in rows),
        "failed": sum(not row.passed for row in rows),
        "elapsed_seconds": elapsed,
        "mean_elapsed_seconds": elapsed / len(rows),
        "status_counts": dict(sorted(statuses.items())),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise BenchmarkProtocolError("cannot summarize an empty bootstrap sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _interval(point: float, values: Sequence[float]) -> dict[str, float]:
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "point": point,
        "lower": _percentile(values, tail),
        "upper": _percentile(values, 1.0 - tail),
    }


def _ordered_rank_rows(
    rows: Sequence[tuple[str, float, bool]],
) -> tuple[tuple[str, float, bool], ...]:
    indexed: dict[str, tuple[str, float, bool]] = {}
    for task_id, score, is_error in rows:
        if task_id in indexed:
            raise BenchmarkProtocolError("duplicate task in raw-rank observations")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise BenchmarkProtocolError("raw rank score must be finite and in [0, 1]")
        if type(is_error) is not bool:
            raise BenchmarkProtocolError("hidden outcome must be boolean")
        indexed[task_id] = (task_id, float(score), is_error)
    ordered = tuple(indexed[task_id] for task_id in sorted(indexed))
    errors = sum(row[2] for row in ordered)
    if not ordered or errors in {0, len(ordered)}:
        raise BenchmarkProtocolError(
            "ranking analysis requires at least one error and one correct task"
        )
    return ordered


def _analyze_rank_signal(
    rows: Sequence[tuple[str, float, bool]],
    *,
    signal_name: str,
    seed: int,
) -> dict[str, Any]:
    ordered = _ordered_rank_rows(rows)
    scores = tuple(row[1] for row in ordered)
    outcomes = tuple(row[2] for row in ordered)
    count = len(ordered)
    point_auroc = tie_corrected_auroc(scores, outcomes)
    point_aurc = area_under_risk_coverage_curve(scores, outcomes)
    rng = random.Random(seed)
    auroc_samples: list[float] = []
    aurc_samples: list[float] = []
    invalid_auroc = 0
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = tuple(rng.randrange(count) for _ in range(count))
        sample_scores = tuple(scores[index] for index in indices)
        sample_outcomes = tuple(outcomes[index] for index in indices)
        aurc_samples.append(
            area_under_risk_coverage_curve(sample_scores, sample_outcomes)
        )
        errors = sum(sample_outcomes)
        if errors in {0, count}:
            invalid_auroc += 1
        else:
            auroc_samples.append(tie_corrected_auroc(sample_scores, sample_outcomes))
    if not auroc_samples:
        raise BenchmarkProtocolError("bootstrap produced no class-valid AUROC sample")
    errors = sum(outcomes)
    return {
        "schema": "mio.raw-rank-uncertainty-statistics.v1",
        "signal_name": signal_name,
        "score_semantics": "raw_rank_score_not_calibrated_probability",
        "task_count": count,
        "error_count": errors,
        "correct_count": count - errors,
        "auroc": _interval(point_auroc, auroc_samples),
        "aurc": _interval(point_aurc, aurc_samples),
        "bootstrap": {
            "method": "ordinary-task-cluster-percentile-v1",
            "samples": BOOTSTRAP_SAMPLES,
            "valid_auroc_samples": len(auroc_samples),
            "invalid_auroc_samples": invalid_auroc,
            "seed": seed,
            "confidence": BOOTSTRAP_CONFIDENCE,
        },
    }


def _analyze_paired_rank(
    reference_rows: Sequence[tuple[str, float, bool]],
    candidate_rows: Sequence[tuple[str, float, bool]],
    *,
    seed: int,
) -> dict[str, Any]:
    reference = _ordered_rank_rows(reference_rows)
    candidate = _ordered_rank_rows(candidate_rows)
    if tuple(row[0] for row in reference) != tuple(row[0] for row in candidate):
        raise BenchmarkProtocolError("paired rank task manifests differ")
    if tuple(row[2] for row in reference) != tuple(row[2] for row in candidate):
        raise BenchmarkProtocolError("paired rank labels diverged despite shared verification")
    reference_scores = tuple(row[1] for row in reference)
    candidate_scores = tuple(row[1] for row in candidate)
    outcomes = tuple(row[2] for row in reference)
    count = len(outcomes)
    reference_auroc = tie_corrected_auroc(reference_scores, outcomes)
    candidate_auroc = tie_corrected_auroc(candidate_scores, outcomes)
    reference_aurc = area_under_risk_coverage_curve(reference_scores, outcomes)
    candidate_aurc = area_under_risk_coverage_curve(candidate_scores, outcomes)
    rng = random.Random(seed)
    auroc_deltas: list[float] = []
    aurc_deltas: list[float] = []
    invalid_auroc = 0
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = tuple(rng.randrange(count) for _ in range(count))
        sample_reference = tuple(reference_scores[index] for index in indices)
        sample_candidate = tuple(candidate_scores[index] for index in indices)
        sample_outcomes = tuple(outcomes[index] for index in indices)
        aurc_deltas.append(
            area_under_risk_coverage_curve(sample_candidate, sample_outcomes)
            - area_under_risk_coverage_curve(sample_reference, sample_outcomes)
        )
        errors = sum(sample_outcomes)
        if errors in {0, count}:
            invalid_auroc += 1
        else:
            auroc_deltas.append(
                tie_corrected_auroc(sample_candidate, sample_outcomes)
                - tie_corrected_auroc(sample_reference, sample_outcomes)
            )
    if not auroc_deltas:
        raise BenchmarkProtocolError("paired bootstrap produced no class-valid AUROC sample")
    return {
        "schema": "mio.paired-raw-rank-uncertainty-statistics.v1",
        "reference_signal": NATIVE_SIGNAL,
        "candidate_signal": RENORMALIZED_SIGNAL,
        "score_semantics": "raw_rank_score_not_calibrated_probability",
        "task_count": count,
        "shared_outcome_per_task": True,
        "reference": {"auroc": reference_auroc, "aurc": reference_aurc},
        "candidate": {"auroc": candidate_auroc, "aurc": candidate_aurc},
        "deltas": {
            "direction": "candidate_minus_reference",
            "auroc": _interval(candidate_auroc - reference_auroc, auroc_deltas),
            "aurc": _interval(candidate_aurc - reference_aurc, aurc_deltas),
        },
        "bootstrap": {
            "method": "ordinary-task-cluster-percentile-paired-v1",
            "samples": BOOTSTRAP_SAMPLES,
            "valid_auroc_samples": len(auroc_deltas),
            "invalid_auroc_samples": invalid_auroc,
            "seed": seed,
            "confidence": BOOTSTRAP_CONFIDENCE,
        },
    }


def run_paired_uncertainty_ablation(
    *,
    cases: Sequence[HumanEvalCase],
    pinned_corpus: Sequence[HumanEvalCase],
    resolved_model: ResolvedModelReference,
    native_generator: AuditedGenerator,
    renormalized_generator: AuditedGenerator,
    hidden_evaluator: HiddenEvaluator,
    expected_integrity: IntegritySnapshot,
    integrity_probe: IntegrityProbe,
    config: PairedUncertaintyConfig | None = None,
) -> dict[str, Any]:
    """Run the pre-registered 32-task ablation and return a source-free report."""

    settings = config or PairedUncertaintyConfig()
    _validate_model_identity(resolved_model)
    if expected_integrity.git_dirty:
        raise BenchmarkProtocolError("paired uncertainty benchmark requires a clean Git tree")
    if expected_integrity.model_identity != resolved_model.canonical_model_id:
        raise BenchmarkProtocolError("integrity snapshot/model identity mismatch")
    _require_integrity(expected_integrity, integrity_probe, phase="pre_generation")
    parity = verifier_parity_certificate_identity()
    if settings.verifier_timeout_seconds != parity["timeout_seconds_per_task"]:
        raise BenchmarkProtocolError("verifier timeout must match the parity certificate")

    full = tuple(pinned_corpus)
    selected = tuple(cases)
    expected = split_humaneval(full, "calibration")
    manifest = corpus_manifest(selected)
    if selected != expected or len(selected) != CALIBRATION_TASKS:
        raise BenchmarkProtocolError("benchmark requires the exact 32-task calibration split")
    if manifest["manifest_sha256"] != OFFICIAL_CALIBRATION_MANIFEST_SHA256:
        raise BenchmarkProtocolError("calibration split does not match the pinned official manifest")
    _validate_generators(
        native_generator,
        renormalized_generator,
        model_id=resolved_model.canonical_model_id,
    )

    # Counterbalance warm-order effects deterministically: 16 tasks per order.
    ranked = sorted(
        selected,
        key=lambda case: hashlib.sha256(
            f"{PROTOCOL_REVISION}:condition-order\0{case.task_id}".encode("utf-8")
        ).digest(),
    )
    native_first_ids = {case.task_id for case in ranked[: len(ranked) // 2]}
    generated_pairs: list[_GeneratedPair] = []
    for case in selected:
        feedback = _direct_feedback(case, settings)
        if case.task_id in native_first_ids:
            native = _call_generator(
                native_generator,
                case,
                feedback,
                expected_method=UNCERTAINTY_METHOD_NATIVE,
                expected_stride=1,
                expected_renormalized=False,
            )
            renormalized = _call_generator(
                renormalized_generator,
                case,
                feedback,
                expected_method=UNCERTAINTY_METHOD_RENORMALIZED,
                expected_stride=8,
                expected_renormalized=True,
            )
        else:
            renormalized = _call_generator(
                renormalized_generator,
                case,
                feedback,
                expected_method=UNCERTAINTY_METHOD_RENORMALIZED,
                expected_stride=8,
                expected_renormalized=True,
            )
            native = _call_generator(
                native_generator,
                case,
                feedback,
                expected_method=UNCERTAINTY_METHOD_NATIVE,
                expected_stride=1,
                expected_renormalized=False,
            )
        if (
            getattr(native.audit, "prompt_source_sha256")
            != getattr(renormalized.audit, "prompt_source_sha256")
            or getattr(native.audit, "prompt_token_ids_sha256")
            != getattr(renormalized.audit, "prompt_token_ids_sha256")
        ):
            raise BenchmarkProtocolError("paired conditions did not receive the same prompt")
        native_decode_identity = (
            native.output_sha256,
            getattr(native.audit, "output_token_ids_sha256"),
            native.generated.metrics.output_tokens,
            getattr(native.audit, "finish_reason"),
        )
        renormalized_decode_identity = (
            renormalized.output_sha256,
            getattr(renormalized.audit, "output_token_ids_sha256"),
            renormalized.generated.metrics.output_tokens,
            getattr(renormalized.audit, "finish_reason"),
        )
        if native_decode_identity != renormalized_decode_identity:
            raise BenchmarkProtocolError(
                "focused ablation decode identity mismatch before hidden evaluation"
            )
        generated_pairs.append(
            _GeneratedPair(
                case=case,
                feedback=feedback,
                native=native,
                renormalized=renormalized,
            )
        )

    _require_integrity(expected_integrity, integrity_probe, phase="pre_hidden")

    # Hard phase boundary: identical outputs are verified once, then that one
    # immutable task label is reused by both ranking signals.
    shared_hidden: list[HiddenEvaluationResult] = []
    for pair in generated_pairs:
        result = hidden_evaluator(pair.case, pair.native.generated.completion)
        if not isinstance(result, HiddenEvaluationResult):
            raise TypeError("hidden evaluator must return HiddenEvaluationResult")
        shared_hidden.append(result)

    native_rank_rows = tuple(
        (
            pair.case.task_id,
            float(pair.native.generated.raw_uncertainty),
            not result.passed,
        )
        for pair, result in zip(generated_pairs, shared_hidden, strict=True)
    )
    renormalized_rank_rows = tuple(
        (
            pair.case.task_id,
            float(pair.renormalized.generated.raw_uncertainty),
            not result.passed,
        )
        for pair, result in zip(generated_pairs, shared_hidden, strict=True)
    )
    native_statistics = _analyze_rank_signal(
        native_rank_rows,
        signal_name=NATIVE_SIGNAL,
        seed=settings.seed,
    )
    renormalized_statistics = _analyze_rank_signal(
        renormalized_rank_rows,
        signal_name=RENORMALIZED_SIGNAL,
        seed=settings.seed,
    )
    paired_statistics = _analyze_paired_rank(
        native_rank_rows,
        renormalized_rank_rows,
        seed=settings.seed,
    )

    native_outputs = [
        {
            "task_id": pair.case.task_id,
            "output_text_sha256": pair.native.output_sha256,
            "output_token_ids_sha256": getattr(
                pair.native.audit,
                "output_token_ids_sha256",
            ),
            "output_tokens": pair.native.generated.metrics.output_tokens,
            "finish_reason": getattr(pair.native.audit, "finish_reason"),
        }
        for pair in generated_pairs
    ]
    renormalized_outputs = [
        {
            "task_id": pair.case.task_id,
            "output_text_sha256": pair.renormalized.output_sha256,
            "output_token_ids_sha256": getattr(
                pair.renormalized.audit,
                "output_token_ids_sha256",
            ),
            "output_tokens": pair.renormalized.generated.metrics.output_tokens,
            "finish_reason": getattr(pair.renormalized.audit, "finish_reason"),
        }
        for pair in generated_pairs
    ]
    paired_outputs = [
        {
            "task_id": pair.case.task_id,
            "output_text_sha256": pair.native.output_sha256,
            "output_token_ids_sha256": getattr(
                pair.native.audit,
                "output_token_ids_sha256",
            ),
            "output_tokens": pair.native.generated.metrics.output_tokens,
            "finish_reason": getattr(pair.native.audit, "finish_reason"),
            "conditions_equal": True,
        }
        for pair in generated_pairs
    ]
    prompt_rows = [
        {
            "task_id": pair.case.task_id,
            "prompt_source_sha256": getattr(pair.native.audit, "prompt_source_sha256"),
            "prompt_token_ids_sha256": getattr(pair.native.audit, "prompt_token_ids_sha256"),
            "seed": pair.feedback.seed,
            "max_output_tokens": pair.feedback.max_output_tokens,
        }
        for pair in generated_pairs
    ]
    private_result_rows = [
        {
            "task_id": pair.case.task_id,
            "native": {
                "raw_rank_score": pair.native.generated.raw_uncertainty,
                "generation_metrics": asdict(pair.native.generated.metrics),
                "uncertainty_scoring_seconds": getattr(
                    pair.native.audit,
                    "uncertainty_scoring_seconds",
                ),
            },
            "renormalized": {
                "raw_rank_score": pair.renormalized.generated.raw_uncertainty,
                "generation_metrics": asdict(pair.renormalized.generated.metrics),
                "uncertainty_scoring_seconds": getattr(
                    pair.renormalized.audit,
                    "uncertainty_scoring_seconds",
                ),
            },
            "shared_hidden": {
                "score": result.score,
                "passed": result.passed,
                "status": result.status,
                "elapsed_seconds": result.elapsed_seconds,
            },
            "decode_identity": paired_output,
        }
        for pair, result, paired_output in zip(
            generated_pairs,
            shared_hidden,
            paired_outputs,
            strict=True,
        )
    ]
    native_rows = tuple(pair.native for pair in generated_pairs)
    renormalized_rows = tuple(pair.renormalized for pair in generated_pairs)
    _require_integrity(expected_integrity, integrity_probe, phase="post_analysis")
    return {
        "schema": REPORT_SCHEMA,
        "protocol": {
            "revision": PROTOCOL_REVISION,
            "phase_order": "generate-all-confirm-identical-then-verify-once-per-task",
            "score_semantics": "raw_rank_score_not_calibrated_probability",
            "probability_calibration_metrics_included": False,
            "conditions": {
                NATIVE_SIGNAL: asdict(native_generator.settings),
                RENORMALIZED_SIGNAL: asdict(renormalized_generator.settings),
            },
            "bootstrap": {
                "samples": BOOTSTRAP_SAMPLES,
                "seed": settings.seed,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "resampling_unit": "task",
                "paired_indices": True,
            },
        },
        "provenance": {
            "git_revision": expected_integrity.git_revision,
            "git_dirty": False,
            "model": resolved_model.canonical_model_id,
            "corpus_revision": HUMANEVAL_REVISION,
            "corpus_archive_sha256": HUMANEVAL_SHA256,
            "split_salt": SPLIT_SALT,
            "prompt_revision": PROMPT_REVISION,
            "timing_method": TIMING_METHOD,
            "packages": {
                "mlx": _package_version("mlx"),
                "mlx-lm": _package_version("mlx-lm"),
            },
            "source_sha256": dict(expected_integrity.source_sha256),
            "verifier_parity_certificate": parity,
        },
        "corpus": manifest,
        "pairing": {
            "tasks": len(generated_pairs),
            "native_first_tasks": len(native_first_ids),
            "renormalized_first_tasks": len(generated_pairs) - len(native_first_ids),
            "prompt_and_budget_manifest_sha256": _canonical_sha256(prompt_rows),
            "native_output_manifest_sha256": _canonical_sha256(native_outputs),
            "renormalized_output_manifest_sha256": _canonical_sha256(
                renormalized_outputs
            ),
            "paired_output_manifest_sha256": _canonical_sha256(paired_outputs),
            "identical_outputs": len(generated_pairs),
            "mismatched_outputs": 0,
            "all_outputs_identical": True,
            "decode_identity_fields": [
                "output_text_sha256",
                "output_token_ids_sha256",
                "output_tokens",
                "finish_reason",
            ],
            "shared_hidden_labels": len(generated_pairs),
        },
        "row_commitment": {
            "schema": "mio.paired-uncertainty-private-row-commitment.v1",
            "rows": len(private_result_rows),
            "manifest_sha256": _canonical_sha256(private_result_rows),
            "preimage_serialized": False,
            "committed_fields": [
                "raw_rank_scores",
                "generation_timing",
                "uncertainty_scoring_timing",
                "shared_hidden_score_outcome_status_timing",
                "decode_identity",
            ],
        },
        "timing": {
            NATIVE_SIGNAL: {
                "generation": _aggregate_generation(native_rows),
            },
            RENORMALIZED_SIGNAL: {
                "generation": _aggregate_generation(renormalized_rows),
            },
            "shared_verification": _aggregate_verification(shared_hidden),
        },
        "statistics": {
            NATIVE_SIGNAL: native_statistics,
            RENORMALIZED_SIGNAL: renormalized_statistics,
            "paired": paired_statistics,
        },
        "claim": {
            "eligible_as_focused_scoring_ablation": True,
            "raw_score_is_calibrated_probability": False,
            "probability_calibration_claim": False,
            "quality_claim": "descriptive-calibration-split-only",
            "heldout_opened": False,
        },
    }


def _verification_evaluator(timeout_seconds: float) -> HiddenEvaluator:
    def evaluate(case: HumanEvalCase, completion: str) -> HiddenEvaluationResult:
        result = verify_candidate(case, completion, timeout_s=timeout_seconds)
        return HiddenEvaluationResult(
            score=float(result.passed),
            passed=result.passed,
            status=result.status,
            elapsed_seconds=result.elapsed_seconds,
        )

    return evaluate


def _copy_model_tree(source: Path, destination: Path) -> str:
    """Clone the complete APFS tree when available, else copy every entry."""

    try:
        subprocess.run(
            ["cp", "-cR", os.fspath(source), os.fspath(destination)],
            check=True,
            capture_output=True,
            timeout=300.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source, destination, symlinks=True)
    return os.fspath(destination)


def _snapshot_tree_paths(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    directories: list[Path] = [root]
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            mode = child.lstat().st_mode
            if not stat.S_ISDIR(mode):
                raise BenchmarkProtocolError("model snapshot contains a non-directory entry")
            directories.append(child)
        for name in file_names:
            child = directory_path / name
            mode = child.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise BenchmarkProtocolError("model snapshot contains a non-regular file")
            files.append(child)
    return tuple(directories), tuple(files)


@dataclass
class LocalModelSnapshot:
    """Private immutable copy that remains alive while MLX owns mapped weights."""

    _temporary_directory: tempfile.TemporaryDirectory
    path: Path
    canonical_model_id: str
    requested_revision: str
    directories: tuple[Path, ...]
    files: tuple[Path, ...]
    initial_fingerprint: LocalModelFingerprint
    _closed: bool = False

    @property
    def resolved_reference(self) -> ResolvedModelReference:
        if self._closed:
            raise BenchmarkProtocolError("model snapshot is already closed")
        return ResolvedModelReference(
            source_kind="local",
            canonical_model_id=self.canonical_model_id,
            load_model_id=os.fspath(self.path),
            load_revision=None,
            requested_model="private-read-only-snapshot",
            requested_revision=self.requested_revision,
        )

    def verify(self, *, phase: str) -> LocalModelFingerprint:
        if self._closed:
            raise BenchmarkProtocolError("model snapshot is already closed")
        try:
            observed = fingerprint_local_model(self.path)
        except ModelIdentityError as exc:
            raise BenchmarkProtocolError(
                f"private model snapshot cannot be fingerprinted at {phase}"
            ) from exc
        observed_identity = f"local-mlx@{observed.revision}"
        if (
            observed_identity != self.canonical_model_id
            or observed.digest != self.initial_fingerprint.digest
        ):
            raise BenchmarkProtocolError(
                f"private model snapshot integrity drift detected at {phase}"
            )
        return observed

    def close(self) -> None:
        if self._closed:
            return
        # Deletion needs write permission on directories.  Restore modes only
        # after all MLX-backed objects have left the enclosing context.
        for directory in self.directories:
            try:
                directory.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            except FileNotFoundError:
                pass
        for file_path in self.files:
            try:
                file_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except FileNotFoundError:
                pass
        self._closed = True
        self._temporary_directory.cleanup()

    def __enter__(self) -> LocalModelSnapshot:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _snapshot_local_model(resolved: ResolvedModelReference) -> LocalModelSnapshot:
    """Clone a content-bound local bundle and freeze the private copy."""

    if resolved.source_kind != "local":
        raise BenchmarkProtocolError(
            "remote Hugging Face models are not supported by the evidence CLI: "
            "a content-verified private remote snapshot is not implemented"
        )
    source = Path(resolved.load_model_id).resolve(strict=True)
    temporary = tempfile.TemporaryDirectory(prefix="mio-mlx-model-snapshot-")
    root = Path(temporary.name)
    root.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    destination = root / "model"
    try:
        _copy_model_tree(source, destination)
        fingerprint = fingerprint_local_model(destination)
        snapshot_identity = f"local-mlx@{fingerprint.revision}"
        if snapshot_identity != resolved.canonical_model_id:
            raise BenchmarkProtocolError(
                "private model snapshot does not match the requested content identity"
            )
        directories, files = _snapshot_tree_paths(destination)
        for file_path in files:
            file_path.chmod(stat.S_IRUSR)
        for directory in reversed(directories):
            directory.chmod(stat.S_IRUSR | stat.S_IXUSR)
        return LocalModelSnapshot(
            _temporary_directory=temporary,
            path=destination,
            canonical_model_id=resolved.canonical_model_id,
            requested_revision=resolved.requested_revision,
            directories=directories,
            files=files,
            initial_fingerprint=fingerprint,
        )
    except Exception:
        # No tree has been published to MLX yet, so cleanup is safe here.
        for directory, _directory_names, _file_names in os.walk(
            destination,
            topdown=True,
        ):
            try:
                Path(directory).chmod(
                    stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
                )
            except FileNotFoundError:
                pass
        temporary.cleanup()
        raise


def _load_generators(
    snapshot: LocalModelSnapshot,
    *,
    loader: Callable[..., tuple[object, object]] | None = None,
) -> tuple[MLXEffortGenerator, MLXEffortGenerator]:
    if not isinstance(snapshot, LocalModelSnapshot):
        raise BenchmarkProtocolError("MLX generators require a private model snapshot")
    snapshot.verify(phase="loader_preflight")
    resolved = snapshot.resolved_reference
    if loader is None:
        from mlx_lm.utils import load

        selected_loader = load
    else:
        selected_loader = loader

    model, tokenizer = selected_loader(
        resolved.load_model_id,
        revision=resolved.load_revision,
        lazy=False,
    )
    native_settings, renormalized_settings = _expected_settings()
    return (
        MLXEffortGenerator(
            model,
            tokenizer,
            model_id=resolved.canonical_model_id,
            settings=native_settings,
        ),
        MLXEffortGenerator(
            model,
            tokenizer,
            model_id=resolved.canonical_model_id,
            settings=renormalized_settings,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-revision",
        required=True,
        help="full Hugging Face commit or local-sha256-v1:<digest>",
    )
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    if output == repository_root or repository_root in output.parents:
        raise SystemExit("--output must be outside the Git worktree")
    try:
        resolved = resolve_model_reference(args.model, args.model_revision)
    except ModelIdentityError as exc:
        raise SystemExit(f"model identity error: {exc}") from exc
    if resolved.source_kind != "local":
        raise SystemExit(
            "remote Hugging Face models are disabled for this evidence CLI: "
            "a content-verified private remote snapshot is not implemented"
        )
    initial_integrity = IntegritySnapshot(
        git_revision=_git_revision(),
        git_dirty=_git_dirty(),
        source_sha256=_source_hashes(),
        model_identity=resolved.canonical_model_id,
    )
    if initial_integrity.git_dirty:
        raise SystemExit("paired uncertainty benchmark requires a clean Git tree")
    integrity_probe = lambda: _capture_integrity_snapshot(  # noqa: E731
        args.model,
        args.model_revision,
    )
    corpus_path = args.corpus or fetch_humaneval()
    pinned = load_humaneval(corpus_path)
    cases = split_humaneval(pinned, "calibration")
    config = PairedUncertaintyConfig(
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
    )
    with _snapshot_local_model(resolved) as model_snapshot:
        model_snapshot.verify(phase="pre_model_load")
        native, renormalized = _load_generators(
            model_snapshot,
        )
        post_load_fingerprint = model_snapshot.verify(phase="post_model_load")
        _require_integrity(initial_integrity, integrity_probe, phase="post_model_load")
        started_at_utc = datetime.now(timezone.utc).isoformat()
        report = run_paired_uncertainty_ablation(
            cases=cases,
            pinned_corpus=pinned,
            resolved_model=resolved,
            native_generator=native,
            renormalized_generator=renormalized,
            hidden_evaluator=_verification_evaluator(
                config.verifier_timeout_seconds
            ),
            expected_integrity=initial_integrity,
            integrity_probe=integrity_probe,
            config=config,
        )
        completed_at_utc = datetime.now(timezone.utc).isoformat()
        final_snapshot_fingerprint = model_snapshot.verify(phase="post_run")
        _require_integrity(
            initial_integrity,
            integrity_probe,
            phase="post_run_before_write",
        )
        report["provenance"]["model_snapshot"] = {
            "method": "private-full-bundle-copy-cow-preferred-v1",
            "loaded_from_snapshot_only": True,
            "snapshot_path_serialized": False,
            "read_only_during_load_and_run": True,
            "fingerprint_schema": post_load_fingerprint.schema,
            "fingerprint_digest": post_load_fingerprint.digest,
            "fingerprinted_files": len(post_load_fingerprint.files),
            "fingerprinted_bytes": post_load_fingerprint.total_bytes,
            "post_load_verified": True,
            "post_run_verified": (
                final_snapshot_fingerprint.digest
                == post_load_fingerprint.digest
            ),
        }
        report["started_at_utc"] = started_at_utc
        report["completed_at_utc"] = completed_at_utc
        _write_json(output, report)
    print(
        f"[paired-uncertainty] output={output} tasks={report['pairing']['tasks']} "
        f"outputs_identical={report['pairing']['all_outputs_identical']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
