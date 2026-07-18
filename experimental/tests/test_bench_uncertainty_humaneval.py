from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat

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
    output_token_ids_sha256: str
    output_tokens: int
    prompt_tokens: int
    prefill_seconds: float
    decode_seconds: float
    other_seconds: float
    finish_reason: str
    prompt_source_sha256: str
    prompt_token_ids_sha256: str
    uncertainty_logprob_stride: int
    uncertainty_logprobs_renormalized: bool
    uncertainty_method: str
    raw_uncertainty: float
    uncertainty_scoring_seconds: float


class _FakeGenerator:
    def __init__(
        self,
        *,
        candidate: bool,
        events: list[tuple[str, str, str]],
        token_mismatch: bool = False,
    ) -> None:
        digest = "a" * 64
        self.model_id = f"local-mlx@local-sha256-v1:{digest}"
        self.settings = MLXSamplerSettings(
            uncertainty_logprob_stride=8 if candidate else 1,
            renormalize_uncertainty_logprobs=candidate,
        )
        self._candidate = candidate
        self._token_mismatch = token_mismatch
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
        output_token_ids_sha256 = hashlib.sha256(
            f"output-tokens:{case.task_id}:{self._token_mismatch}".encode()
        ).hexdigest()
        self._audit_records.append(
            _Audit(
                task_id=case.task_id,
                action=feedback.action,
                model_id=self.model_id,
                seed=feedback.seed,
                max_output_tokens=feedback.max_output_tokens,
                deterministic_sampler=True,
                output_text_sha256=output_sha256,
                output_token_ids_sha256=output_token_ids_sha256,
                output_tokens=5,
                prompt_tokens=40,
                prefill_seconds=0.04,
                decode_seconds=0.02,
                other_seconds=0.001,
                finish_reason="stop",
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


def _integrity(*, dirty: bool = False, source_digest: str = "d") -> benchmark.IntegritySnapshot:
    return benchmark.IntegritySnapshot(
        git_revision="c" * 40,
        git_dirty=dirty,
        source_sha256={"test-fixture": source_digest * 64},
        model_identity=_resolved_model().canonical_model_id,
    )


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
    seen_hidden: set[str] = set()

    def hidden(case: HumanEvalCase, completion: str, /) -> HiddenEvaluationResult:
        assert case.task_id not in seen_hidden, "identical output must be verified only once"
        seen_hidden.add(case.task_id)
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
        expected_integrity=_integrity(),
        integrity_probe=lambda: _integrity(),
    )

    assert len(events) == 96
    assert all(event[0] == "generate" for event in events[:64])
    assert all(event[0] == "verify" for event in events[64:])
    assert len([event for event in events if event[0] == "verify"]) == 32
    assert len(seen_hidden) == 32
    assert report["pairing"]["native_first_tasks"] == 16
    assert report["pairing"]["renormalized_first_tasks"] == 16
    assert report["pairing"]["identical_outputs"] == 32
    assert report["pairing"]["all_outputs_identical"] is True
    assert report["timing"]["shared_verification"]["calls"] == 32
    assert report["pairing"]["shared_hidden_labels"] == 32
    assert "paired_label_manifest_sha256" not in report["pairing"]
    assert report["row_commitment"]["rows"] == 32
    assert report["statistics"][benchmark.NATIVE_SIGNAL]["error_count"] == report[
        "statistics"
    ][benchmark.RENORMALIZED_SIGNAL]["error_count"]
    assert report["statistics"]["paired"]["bootstrap"]["samples"] == 10_000
    assert report["statistics"]["paired"]["shared_outcome_per_task"] is True
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
    assert '"brier"' not in encoded
    assert '"reliability_bins"' not in encoded
    assert '"raw_score_is_calibrated_probability": false' in encoded


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
            expected_integrity=_integrity(dirty=True),
            integrity_probe=lambda: pytest.fail("probe should not run for dirty snapshot"),
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
            expected_integrity=_integrity(),
            integrity_probe=lambda: _integrity(),
        )
    assert events == []


def test_output_token_identity_mismatch_fails_before_hidden_labels(
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
    with pytest.raises(benchmark.BenchmarkProtocolError, match="decode identity mismatch"):
        benchmark.run_paired_uncertainty_ablation(
            cases=cases,
            pinned_corpus=corpus,
            resolved_model=_resolved_model(),
            native_generator=_FakeGenerator(candidate=False, events=events),
            renormalized_generator=_FakeGenerator(
                candidate=True,
                events=events,
                token_mismatch=True,
            ),
            hidden_evaluator=lambda *_: pytest.fail("hidden verifier was opened"),
            expected_integrity=_integrity(),
            integrity_probe=lambda: _integrity(),
        )
    assert all(event[0] == "generate" for event in events)


def test_integrity_drift_after_hidden_scoring_blocks_report(
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
    probes = 0

    def integrity_probe() -> benchmark.IntegritySnapshot:
        nonlocal probes
        probes += 1
        return _integrity() if probes < 3 else _integrity(source_digest="e")

    def hidden(case: HumanEvalCase, completion: str, /) -> HiddenEvaluationResult:
        events.append(("verify", case.task_id, hashlib.sha256(completion.encode()).hexdigest()))
        passed = int(case.task_id.split("/")[-1]) % 4 != 0
        return HiddenEvaluationResult(
            score=float(passed),
            passed=passed,
            status="passed" if passed else "failed",
            elapsed_seconds=0.005,
        )

    with pytest.raises(benchmark.BenchmarkProtocolError, match="post_analysis"):
        benchmark.run_paired_uncertainty_ablation(
            cases=cases,
            pinned_corpus=corpus,
            resolved_model=_resolved_model(),
            native_generator=_FakeGenerator(candidate=False, events=events),
            renormalized_generator=_FakeGenerator(candidate=True, events=events),
            hidden_evaluator=hidden,
            expected_integrity=_integrity(),
            integrity_probe=integrity_probe,
        )
    assert probes == 3
    assert len([event for event in events if event[0] == "verify"]) == 32


def test_private_snapshot_is_content_bound_read_only_and_is_the_only_load_path(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original-model"
    original.mkdir()
    (original / "config.json").write_text('{"model_type":"fixture"}\n')
    (original / "model.safetensors").write_bytes(b"tiny-weight-fixture")
    (original / "tokenizer.json").write_text('{"fixture":true}\n')
    (original / "copied-entire-bundle.bin").write_bytes(b"non-identity-extra")
    fingerprint = benchmark.fingerprint_local_model(original)
    resolved = benchmark.resolve_model_reference(original.as_posix(), fingerprint.revision)
    loaded: dict[str, object] = {}

    def fake_loader(
        model_id: str,
        *,
        revision: str | None,
        lazy: bool,
    ) -> tuple[object, object]:
        loaded.update(model_id=model_id, revision=revision, lazy=lazy)
        return object(), object()

    with benchmark._snapshot_local_model(resolved) as snapshot:
        snapshot_path = snapshot.path
        assert snapshot_path != original
        assert snapshot_path.is_dir()
        assert (snapshot_path / "copied-entire-bundle.bin").read_bytes() == b"non-identity-extra"
        assert snapshot.verify(phase="unit_test").digest == fingerprint.digest
        assert all(
            not (path.stat().st_mode & stat.S_IWUSR)
            for path in (*snapshot.directories, *snapshot.files)
        )

        native, renormalized = benchmark._load_generators(
            snapshot,
            loader=fake_loader,
        )
        assert Path(str(loaded["model_id"])) == snapshot_path
        assert Path(str(loaded["model_id"])) != original
        assert loaded["revision"] is None
        assert loaded["lazy"] is False
        assert native.model_id == resolved.canonical_model_id
        assert renormalized.model_id == resolved.canonical_model_id

    assert not snapshot_path.exists()


def test_private_snapshot_rejects_remote_model_reference() -> None:
    remote = ResolvedModelReference(
        source_kind="huggingface",
        canonical_model_id=f"hf://org/model@{'a' * 40}",
        load_model_id="org/model",
        load_revision="a" * 40,
        requested_model="org/model",
        requested_revision="a" * 40,
    )
    with pytest.raises(benchmark.BenchmarkProtocolError, match="remote Hugging Face"):
        benchmark._snapshot_local_model(remote)
