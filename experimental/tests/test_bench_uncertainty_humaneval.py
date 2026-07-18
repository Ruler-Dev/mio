from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import pytest

import experimental.effort.bench_uncertainty_humaneval as benchmark
from experimental.effort.humaneval import HumanEvalCase, corpus_manifest, split_humaneval
from experimental.effort.markov_runner import GeneratedCandidate, HiddenEvaluationResult
from experimental.effort.mlx_backend import (
    MLXSamplerSettings,
    UNCERTAINTY_METHOD_NATIVE,
    UNCERTAINTY_METHOD_RENORMALIZED,
)
from experimental.effort.model_identity import ResolvedModelReference
from experimental.markov_effort_controller import ControllerAction, GenerationMetrics


@dataclass(frozen=True)
class _Audit:
    task_id: str
    action: ControllerAction
    model_id: str
    seed: int
    max_output_tokens: int
    deterministic_sampler: bool
    output_text_sha256: str
    prompt_source_sha256: str
    prompt_token_ids_sha256: str
    uncertainty_logprob_stride: int
    uncertainty_logprobs_renormalized: bool
    uncertainty_method: str
    raw_uncertainty: float
    uncertainty_scoring_seconds: float


class _FakeGenerator:
    def __init__(self, *, candidate: bool, events: list[tuple[str, str, str]]) -> None:
        digest = "a" * 64
        self.model_id = f"local-mlx@local-sha256-v1:{digest}"
        self.settings = MLXSamplerSettings(
            uncertainty_logprob_stride=8 if candidate else 1,
            renormalize_uncertainty_logprobs=candidate,
        )
        self._candidate = candidate
        self._events = events
        self._audit_records: list[_Audit] = []

    @property
    def audit_records(self) -> tuple[_Audit, ...]:
        return tuple(self._audit_records)

    def __call__(self, case, feedback, /) -> GeneratedCandidate:
        condition = benchmark.RENORMALIZED_SIGNAL if self._candidate else benchmark.NATIVE_SIGNAL
        self._events.append(("generate", case.task_id, condition))
        index = int(case.task_id.split("/")[-1])
        completion = f"completion-secret-{case.task_id}"
        # Both conditions emit identical text; only uncertainty scoring differs.
        uncertainty = (
            (0.88 if index % 4 == 0 else 0.12)
            if self._candidate
            else (0.62 if index % 4 == 0 else 0.38)
        )
        output_sha256 = hashlib.sha256(completion.encode()).hexdigest()
        prompt_sha256 = hashlib.sha256(f"same-prompt:{case.task_id}".encode()).hexdigest()
        token_sha256 = hashlib.sha256(f"same-tokens:{case.task_id}".encode()).hexdigest()
        self._audit_records.append(
            _Audit(
                task_id=case.task_id,
                action=feedback.action,
                model_id=self.model_id,
                seed=feedback.seed,
                max_output_tokens=feedback.max_output_tokens,
                deterministic_sampler=True,
                output_text_sha256=output_sha256,
                prompt_source_sha256=prompt_sha256,
                prompt_token_ids_sha256=token_sha256,
                uncertainty_logprob_stride=8 if self._candidate else 1,
                uncertainty_logprobs_renormalized=self._candidate,
                uncertainty_method=(
                    UNCERTAINTY_METHOD_RENORMALIZED
                    if self._candidate
                    else UNCERTAINTY_METHOD_NATIVE
                ),
                raw_uncertainty=uncertainty,
                uncertainty_scoring_seconds=0.002 if self._candidate else 0.01,
            )
        )
        return GeneratedCandidate(
            completion=completion,
            metrics=GenerationMetrics(
                prompt_tokens=40,
                output_tokens=5,
                prefill_seconds=0.04,
                decode_seconds=0.02,
                other_seconds=0.001,
            ),
            raw_uncertainty=uncertainty,
        )


def _fixture_corpus() -> tuple[HumanEvalCase, ...]:
    return tuple(
        HumanEvalCase(
            task_id=f"HumanEval/{index}",
            prompt=f"prompt-secret-{index}\ndef solve_{index}():\n",
            test=f"test-secret-{index}",
            entry_point=f"solve_{index}",
        )
        for index in range(164)
    )


def _resolved_model() -> ResolvedModelReference:
    digest = "a" * 64
    return ResolvedModelReference(
        source_kind="local",
        canonical_model_id=f"local-mlx@local-sha256-v1:{digest}",
        load_model_id="/not-used-in-unit-test",
        load_revision=None,
        requested_model="/not-serialized",
        requested_revision=f"local-sha256-v1:{digest}",
    )


def _parity_fixture() -> dict[str, object]:
    return {
        "validated": True,
        "schema": "test-certificate",
        "certificate_sha256": "b" * 64,
        "timeout_seconds_per_task": 10.0,
        "passed": 164,
        "total": 164,
    }


def test_paired_ablation_keeps_hidden_channel_closed_and_report_source_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _fixture_corpus()
    cases = split_humaneval(corpus, "calibration")
    monkeypatch.setattr(
        benchmark,
        "OFFICIAL_CALIBRATION_MANIFEST_SHA256",
        corpus_manifest(cases)["manifest_sha256"],
    )
    monkeypatch.setattr(
        benchmark,
        "verifier_parity_certificate_identity",
        _parity_fixture,
    )
    events: list[tuple[str, str, str]] = []
    native = _FakeGenerator(candidate=False, events=events)
    renormalized = _FakeGenerator(candidate=True, events=events)

    def hidden(case: HumanEvalCase, completion: str, /) -> HiddenEvaluationResult:
        assert completion == f"completion-secret-{case.task_id}"
        events.append(("verify", case.task_id, hashlib.sha256(completion.encode()).hexdigest()))
        passed = int(case.task_id.split("/")[-1]) % 4 != 0
        return HiddenEvaluationResult(
            score=float(passed),
            passed=passed,
            status="passed" if passed else "failed",
            elapsed_seconds=0.005,
        )

    report = benchmark.run_paired_uncertainty_ablation(
        cases=cases,
        pinned_corpus=corpus,
        resolved_model=_resolved_model(),
        native_generator=native,
        renormalized_generator=renormalized,
        hidden_evaluator=hidden,
        git_revision="c" * 40,
        git_dirty=False,
    )

    assert len(events) == 128
    assert all(event[0] == "generate" for event in events[:64])
    assert all(event[0] == "verify" for event in events[64:])
    assert len([event for event in events if event[0] == "verify"]) == 64
    assert report["pairing"]["native_first_tasks"] == 16
    assert report["pairing"]["renormalized_first_tasks"] == 16
    assert report["pairing"]["identical_outputs"] == 32
    assert report["pairing"]["all_outputs_identical"] is True
    assert report["timing"][benchmark.NATIVE_SIGNAL]["verification"]["calls"] == 32
    assert report["timing"][benchmark.RENORMALIZED_SIGNAL]["verification"]["calls"] == 32
    assert report["statistics"]["paired"]["bootstrap"]["samples"] == 10_000
    assert report["statistics"]["paired"]["bootstrap"]["method"].startswith(
        "ordinary-task-cluster"
    )

    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    assert "prompt-secret" not in encoded
    assert "completion-secret" not in encoded
    assert "test-secret" not in encoded
    assert '"tasks": [' not in encoded
    assert '"is_error"' not in encoded
    assert '"completion"' not in encoded


def test_ablation_fails_before_generation_for_dirty_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str, str]] = []
    with pytest.raises(benchmark.BenchmarkProtocolError, match="clean Git"):
        benchmark.run_paired_uncertainty_ablation(
            cases=(),
            pinned_corpus=(),
            resolved_model=_resolved_model(),
            native_generator=_FakeGenerator(candidate=False, events=events),
            renormalized_generator=_FakeGenerator(candidate=True, events=events),
            hidden_evaluator=lambda *_: pytest.fail("hidden verifier was opened"),
            git_revision="c" * 40,
            git_dirty=True,
        )
    assert events == []


def test_condition_drift_fails_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = _fixture_corpus()
    cases = split_humaneval(corpus, "calibration")
    monkeypatch.setattr(
        benchmark,
        "OFFICIAL_CALIBRATION_MANIFEST_SHA256",
        corpus_manifest(cases)["manifest_sha256"],
    )
    monkeypatch.setattr(
        benchmark,
        "verifier_parity_certificate_identity",
        _parity_fixture,
    )
    events: list[tuple[str, str, str]] = []
    native = _FakeGenerator(candidate=False, events=events)
    renormalized = _FakeGenerator(candidate=True, events=events)
    renormalized.settings = MLXSamplerSettings(
        temperature=0.1,
        uncertainty_logprob_stride=8,
        renormalize_uncertainty_logprobs=True,
    )
    with pytest.raises(benchmark.BenchmarkProtocolError, match="candidate condition"):
        benchmark.run_paired_uncertainty_ablation(
            cases=cases,
            pinned_corpus=corpus,
            resolved_model=_resolved_model(),
            native_generator=native,
            renormalized_generator=renormalized,
            hidden_evaluator=lambda *_: pytest.fail("hidden verifier was opened"),
            git_revision="c" * 40,
            git_dirty=False,
        )
    assert events == []
