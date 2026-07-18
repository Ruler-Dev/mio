#!/usr/bin/env python3
"""Source-free paired HumanEval ablation for MLX uncertainty scoring.

This benchmark changes only the uncertainty-scoring path: native selected-token
log probabilities at stride 1 are compared with FP32-renormalized log
probabilities at stride 8.  Both conditions receive the exact same public case,
greedy sampler, direct prompt, seed, and output-token budget.  All 64 candidates
are generated before the hidden verifier is opened, then each condition is
verified exactly once per task.

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
from pathlib import Path
import statistics
from typing import Any, Protocol, Sequence

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
    ModelIdentityError,
    ResolvedModelReference,
    resolve_model_reference,
)
from experimental.effort.uncertainty_statistics import (
    UncertaintyObservation,
    analyze_paired_uncertainty,
    analyze_uncertainty,
)
from experimental.markov_effort_controller import (
    ControllerAction,
    deterministic_generation_seed,
)


REPORT_SCHEMA = "mio.paired-uncertainty-humaneval.v1"
PROTOCOL_REVISION = "mio-paired-uncertainty-humaneval-v1"
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
        "controller": here.parents[1] / "markov_effort_controller.py",
        "humaneval": here.with_name("humaneval.py"),
        "markov_runner": here.with_name("markov_runner.py"),
        "mlx_backend": here.with_name("mlx_backend.py"),
        "model_identity": here.with_name("model_identity.py"),
        "uncertainty_statistics": here.with_name("uncertainty_statistics.py"),
        "verifier_parity_certificate": VERIFIER_PARITY_CERTIFICATE_PATH,
    }
    return {name: _file_sha256(path) for name, path in files.items()}


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
        "uncertainty_logprob_stride": expected_stride,
        "uncertainty_logprobs_renormalized": expected_renormalized,
        "uncertainty_method": expected_method,
    }
    for field, expected in checks.items():
        if getattr(audit, field, None) != expected:
            raise BenchmarkProtocolError(f"generator audit mismatch: {field}")
    if getattr(audit, "raw_uncertainty", None) != generated.raw_uncertainty:
        raise BenchmarkProtocolError("generator audit uncertainty mismatch")
    for field in ("prompt_source_sha256", "prompt_token_ids_sha256"):
        digest = getattr(audit, field, None)
        if not isinstance(digest, str) or len(digest) != 64:
            raise BenchmarkProtocolError(f"generator audit lacks {field}")
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


def run_paired_uncertainty_ablation(
    *,
    cases: Sequence[HumanEvalCase],
    pinned_corpus: Sequence[HumanEvalCase],
    resolved_model: ResolvedModelReference,
    native_generator: AuditedGenerator,
    renormalized_generator: AuditedGenerator,
    hidden_evaluator: HiddenEvaluator,
    git_revision: str,
    git_dirty: bool,
    config: PairedUncertaintyConfig | None = None,
) -> dict[str, Any]:
    """Run the pre-registered 32-task ablation and return a source-free report."""

    settings = config or PairedUncertaintyConfig()
    _validate_model_identity(resolved_model)
    if git_dirty:
        raise BenchmarkProtocolError("paired uncertainty benchmark requires a clean Git tree")
    if len(git_revision) != 40 or any(character not in "0123456789abcdef" for character in git_revision):
        raise BenchmarkProtocolError("Git revision must be a full commit digest")
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
        generated_pairs.append(
            _GeneratedPair(
                case=case,
                feedback=feedback,
                native=native,
                renormalized=renormalized,
            )
        )

    # Hard phase boundary: no hidden call appears above this line.
    native_hidden: list[HiddenEvaluationResult] = []
    renormalized_hidden: list[HiddenEvaluationResult] = []
    for pair in generated_pairs:
        native_result = hidden_evaluator(pair.case, pair.native.generated.completion)
        renormalized_result = hidden_evaluator(
            pair.case,
            pair.renormalized.generated.completion,
        )
        if not isinstance(native_result, HiddenEvaluationResult) or not isinstance(
            renormalized_result, HiddenEvaluationResult
        ):
            raise TypeError("hidden evaluator must return HiddenEvaluationResult")
        native_hidden.append(native_result)
        renormalized_hidden.append(renormalized_result)

    native_observations = tuple(
        UncertaintyObservation(
            task_cluster_id=pair.case.task_id,
            predicted_error_probability=float(pair.native.generated.raw_uncertainty),
            is_error=not result.passed,
        )
        for pair, result in zip(generated_pairs, native_hidden, strict=True)
    )
    renormalized_observations = tuple(
        UncertaintyObservation(
            task_cluster_id=pair.case.task_id,
            predicted_error_probability=float(pair.renormalized.generated.raw_uncertainty),
            is_error=not result.passed,
        )
        for pair, result in zip(generated_pairs, renormalized_hidden, strict=True)
    )
    native_statistics = analyze_uncertainty(
        native_observations,
        signal_name=NATIVE_SIGNAL,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=settings.seed,
        confidence=BOOTSTRAP_CONFIDENCE,
    )
    renormalized_statistics = analyze_uncertainty(
        renormalized_observations,
        signal_name=RENORMALIZED_SIGNAL,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=settings.seed,
        confidence=BOOTSTRAP_CONFIDENCE,
    )
    paired_statistics = analyze_paired_uncertainty(
        native_observations,
        renormalized_observations,
        reference_signal=NATIVE_SIGNAL,
        candidate_signal=RENORMALIZED_SIGNAL,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=settings.seed,
        confidence=BOOTSTRAP_CONFIDENCE,
    )

    native_outputs = [
        {"task_id": pair.case.task_id, "output_sha256": pair.native.output_sha256}
        for pair in generated_pairs
    ]
    renormalized_outputs = [
        {
            "task_id": pair.case.task_id,
            "output_sha256": pair.renormalized.output_sha256,
        }
        for pair in generated_pairs
    ]
    paired_outputs = [
        {
            "task_id": pair.case.task_id,
            "native_output_sha256": pair.native.output_sha256,
            "renormalized_output_sha256": pair.renormalized.output_sha256,
            "equal": pair.native.output_sha256 == pair.renormalized.output_sha256,
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
    identical = sum(row["equal"] for row in paired_outputs)
    native_rows = tuple(pair.native for pair in generated_pairs)
    renormalized_rows = tuple(pair.renormalized for pair in generated_pairs)
    return {
        "schema": REPORT_SCHEMA,
        "protocol": {
            "revision": PROTOCOL_REVISION,
            "phase_order": "generate-all-then-verify-each-condition-once",
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
            "git_revision": git_revision,
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
            "source_sha256": _source_hashes(),
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
            "identical_outputs": identical,
            "mismatched_outputs": len(generated_pairs) - identical,
            "all_outputs_identical": identical == len(generated_pairs),
        },
        "timing": {
            NATIVE_SIGNAL: {
                "generation": _aggregate_generation(native_rows),
                "verification": _aggregate_verification(native_hidden),
            },
            RENORMALIZED_SIGNAL: {
                "generation": _aggregate_generation(renormalized_rows),
                "verification": _aggregate_verification(renormalized_hidden),
            },
        },
        "statistics": {
            NATIVE_SIGNAL: native_statistics.to_mapping(),
            RENORMALIZED_SIGNAL: renormalized_statistics.to_mapping(),
            "paired": paired_statistics.to_mapping(),
        },
        "claim": {
            "eligible_as_focused_scoring_ablation": identical == len(generated_pairs),
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


def _load_generators(
    resolved: ResolvedModelReference,
) -> tuple[MLXEffortGenerator, MLXEffortGenerator]:
    from mlx_lm.utils import load

    model, tokenizer = load(
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
    revision = _git_revision()
    if _git_dirty():
        raise SystemExit("paired uncertainty benchmark requires a clean Git tree")
    try:
        resolved = resolve_model_reference(args.model, args.model_revision)
    except ModelIdentityError as exc:
        raise SystemExit(f"model identity error: {exc}") from exc
    corpus_path = args.corpus or fetch_humaneval()
    pinned = load_humaneval(corpus_path)
    cases = split_humaneval(pinned, "calibration")
    config = PairedUncertaintyConfig(
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
    )
    native, renormalized = _load_generators(resolved)
    report = run_paired_uncertainty_ablation(
        cases=cases,
        pinned_corpus=pinned,
        resolved_model=resolved,
        native_generator=native,
        renormalized_generator=renormalized,
        hidden_evaluator=_verification_evaluator(config.verifier_timeout_seconds),
        git_revision=revision,
        git_dirty=False,
        config=config,
    )
    report["launched_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output, report)
    print(
        f"[paired-uncertainty] output={output} tasks={report['pairing']['tasks']} "
        f"outputs_identical={report['pairing']['all_outputs_identical']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
