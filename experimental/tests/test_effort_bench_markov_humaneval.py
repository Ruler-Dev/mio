from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import experimental.effort.bench_markov_humaneval as benchmark_module

from experimental.effort.bench_markov_humaneval import (
    CALIBRATION_ARTIFACT_SCHEMA,
    EVALUATION_SCHEMA,
    PROTOCOL_REVISION,
    VERIFIER_PARITY_CERTIFICATE_PATH,
    BenchmarkProtocolError,
    CalibrationArtifact,
    CalibrationConfig,
    calibration_policy_sha256,
    calibrate_markov_humaneval,
    evaluation_policy_sha256,
    evaluate_markov_humaneval,
    expected_static_provenance_hashes,
    filter_underpowered_transition_observations,
    load_calibration_artifact,
    save_calibration_artifact,
    verifier_parity_certificate_identity,
)
from experimental.effort.calibration import TransitionCalibrationObservation
from experimental.effort.humaneval import (
    CALIBRATION_TASKS,
    HUMANEVAL_REVISION,
    HUMANEVAL_SHA256,
    SPLIT_SALT,
    HumanEvalCase,
    PublicHumanEvalCase,
    PublicValidationResult,
    corpus_manifest,
    split_humaneval,
)
from experimental.effort.markov_runner import (
    GeneratedCandidate,
    HiddenEvaluationResult,
    PublicGenerationFeedback,
)
from experimental.effort.statistics_v2 import RunProvenance
from experimental.markov_effort_controller import (
    CalibrationIdentity,
    ControllerAction,
    EFFORT_PROFILES,
    EffortTier,
    GenerationMetrics,
    Trigger,
    ValidationOutcome,
)


IDENTITY = CalibrationIdentity(
    model="fake-model@0123456789abcdef",
    config="fake-config-v1",
    prompt="fake-public-prompt-v1",
    sampler="greedy-fake-v1",
    corpus=f"HumanEval@{HUMANEVAL_REVISION}:{HUMANEVAL_SHA256}",
    split=f"{SPLIT_SALT}:calibration:{CALIBRATION_TASKS}",
    backend="fake-backend-v1",
)
ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64
TWO_SHA = "2" * 64
THREE_SHA = "3" * 64


def _case(index: int, *, prefix: str = "HumanEval") -> HumanEvalCase:
    return HumanEvalCase(
        task_id=f"{prefix}/{index}",
        prompt=f"def solve_{index}(value):\n    \"\"\"Return a value.\"\"\"\n",
        test="def check(candidate):\n    assert candidate(1) == 1\n",
        entry_point=f"solve_{index}",
    )


def _provenance(
    cases: tuple[HumanEvalCase, ...],
    *,
    dirty: bool = False,
    leakage: bool = False,
    split: str = "calibration",
    policy_sha256: str = ZERO_SHA,
) -> RunProvenance:
    manifest = corpus_manifest(cases)
    static_hashes = expected_static_provenance_hashes()
    return RunProvenance(
        git_revision="abcdef0123456789",
        git_dirty=dirty,
        model_revision="0123456789abcdef",
        policy_sha256=policy_sha256,
        task_manifest_sha256=str(manifest["manifest_sha256"]),
        scorer_sha256=static_hashes["scorer_sha256"],
        verifier_sha256=static_hashes["verifier_sha256"],
        preregistration_sha256=static_hashes["preregistration_sha256"],
        test_split_id=f"HumanEval:{split}:{manifest['manifest_sha256']}",
        leakage_detected=leakage,
    )


def _task_index(task_id: str) -> int:
    return int(task_id.rsplit("/", 1)[-1])


class _FakeGenerator:
    """Deterministic public-only generator with cheap extra candidates."""

    def __init__(self, hidden_calls: list[tuple[str, str]] | None = None) -> None:
        self.hidden_calls = hidden_calls
        self.calls: list[tuple[PublicHumanEvalCase, PublicGenerationFeedback]] = []

    def __call__(
        self,
        case: PublicHumanEvalCase,
        feedback: PublicGenerationFeedback,
    ) -> GeneratedCandidate:
        assert type(case) is PublicHumanEvalCase
        assert not hasattr(case, "test")
        assert not hasattr(feedback, "hidden_score")
        if self.hidden_calls is not None:
            # Calibration must finish every generation before the first hidden call.
            assert self.hidden_calls == []
        self.calls.append((case, feedback))
        index = _task_index(case.task_id)
        direct = feedback.action is ControllerAction.GENERATE_DIRECT
        if direct:
            assert feedback.parent_completion is None
            metrics = GenerationMetrics(
                prompt_tokens=40,
                output_tokens=20,
                prefill_seconds=0.05,
                decode_seconds=0.05,
            )
            uncertainty = 0.2 if index % 2 == 0 else 0.9
        else:
            assert feedback.parent_completion is not None
            assert feedback.validator_status == "public_failure"
            metrics = GenerationMetrics(
                prompt_tokens=48,
                output_tokens=8,
                prefill_seconds=0.005,
                decode_seconds=0.005,
            )
            uncertainty = 0.1
        completion = (
            f"task={case.task_id};action={feedback.action.value};seed={feedback.seed}"
        )
        return GeneratedCandidate(
            completion=completion,
            metrics=metrics,
            raw_uncertainty=uncertainty,
        )


def _public_failure(
    case: PublicHumanEvalCase,
    completion: str,
) -> PublicValidationResult:
    assert type(case) is PublicHumanEvalCase
    return PublicValidationResult(
        outcome=ValidationOutcome.FAIL,
        status="public_failure",
        feedback="syntax_only_failure",
        elapsed_seconds=0.001,
        source_sha256=(completion.encode("utf-8").hex() + ZERO_SHA)[:64],
    )


def _public_success(
    case: PublicHumanEvalCase,
    completion: str,
) -> PublicValidationResult:
    assert type(case) is PublicHumanEvalCase
    return PublicValidationResult(
        outcome=ValidationOutcome.PASS,
        status="public_success",
        feedback="syntax_only_success",
        elapsed_seconds=0.001,
        source_sha256=hashlib.sha256(completion.encode("utf-8")).hexdigest(),
    )


class _FakeHiddenEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, case: HumanEvalCase, completion: str) -> HiddenEvaluationResult:
        self.calls.append((case.task_id, completion))
        is_direct = f"action={ControllerAction.GENERATE_DIRECT.value}" in completion
        passed = (not is_direct) or _task_index(case.task_id) % 2 == 0
        return HiddenEvaluationResult(
            score=float(passed),
            passed=passed,
            status="passed" if passed else "assertion_failed",
            elapsed_seconds=0.002,
        )


class _RelabelingHiddenEvaluator:
    """Would relabel a timeout as passing when the same artifact is retried."""

    def __init__(self, first_passing_task: str) -> None:
        self.first_passing_task = first_passing_task
        self.calls: list[tuple[str, str]] = []
        self.calls_by_task: dict[str, int] = {}

    def __call__(self, case: HumanEvalCase, completion: str) -> HiddenEvaluationResult:
        self.calls.append((case.task_id, completion))
        count = self.calls_by_task.get(case.task_id, 0) + 1
        self.calls_by_task[case.task_id] = count
        first_passed = case.task_id == self.first_passing_task
        passed = first_passed if count == 1 else not first_passed
        return HiddenEvaluationResult(
            score=float(passed),
            passed=passed,
            status="passed" if passed else "timeout",
            elapsed_seconds=0.002 if passed else 10.0,
        )


class _ConstantGenerator:
    """Emit byte-identical artifacts for distinct tasks without hidden access."""

    def __init__(self, hidden_calls: list[tuple[str, str]]) -> None:
        self.hidden_calls = hidden_calls
        self.calls: list[str] = []

    def __call__(
        self,
        case: PublicHumanEvalCase,
        feedback: PublicGenerationFeedback,
    ) -> GeneratedCandidate:
        assert self.hidden_calls == []
        assert feedback.action is ControllerAction.GENERATE_DIRECT
        self.calls.append(case.task_id)
        return GeneratedCandidate(
            completion="identical terminal artifact",
            metrics=GenerationMetrics(
                prompt_tokens=40,
                output_tokens=4,
                prefill_seconds=0.05,
                decode_seconds=0.05,
            ),
            raw_uncertainty=0.0,
        )


class _TaskSensitiveHiddenEvaluator:
    def __init__(self, passing_task: str) -> None:
        self.passing_task = passing_task
        self.calls: list[tuple[str, str]] = []

    def __call__(self, case: HumanEvalCase, completion: str) -> HiddenEvaluationResult:
        self.calls.append((case.task_id, completion))
        passed = case.task_id == self.passing_task
        return HiddenEvaluationResult(
            score=float(passed),
            passed=passed,
            status="passed" if passed else "assertion_failed",
            elapsed_seconds=0.002,
        )


@pytest.fixture(scope="module")
def calibration_bundle() -> tuple[
    CalibrationArtifact,
    _FakeGenerator,
    _FakeHiddenEvaluator,
    tuple[HumanEvalCase, ...],
    tuple[HumanEvalCase, ...],
]:
    full_cases = tuple(_case(index) for index in range(164))
    cases = split_humaneval(full_cases, "calibration")
    heldout = split_humaneval(full_cases, "heldout")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        benchmark_module,
        "OFFICIAL_FULL_MANIFEST_SHA256",
        corpus_manifest(full_cases)["manifest_sha256"],
    )
    monkeypatch.setattr(
        benchmark_module,
        "OFFICIAL_CALIBRATION_MANIFEST_SHA256",
        corpus_manifest(cases)["manifest_sha256"],
    )
    monkeypatch.setattr(
        benchmark_module,
        "OFFICIAL_HELDOUT_MANIFEST_SHA256",
        corpus_manifest(heldout)["manifest_sha256"],
    )
    hidden = _FakeHiddenEvaluator()
    generator = _FakeGenerator(hidden.calls)
    config = CalibrationConfig(
        initial_max_output_tokens=64,
        min_task_clusters=8,
        bootstrap_resamples=64,
        seed=17,
    )
    artifact = calibrate_markov_humaneval(
        cases=cases,
        pinned_corpus=full_cases,
        identity=IDENTITY,
        provenance=_provenance(
            cases,
            policy_sha256=calibration_policy_sha256(config),
        ),
        generator=generator,
        hidden_evaluator=hidden,
        public_validator=_public_failure,
        config=config,
    )
    yield artifact, generator, hidden, cases, full_cases
    monkeypatch.undo()


def test_calibration_defers_hidden_evaluation_and_discards_task_labels(
    calibration_bundle,
) -> None:
    artifact, generator, hidden, cases, _ = calibration_bundle
    candidates_per_task = 10

    assert len(generator.calls) == len(cases) * candidates_per_task
    assert len(hidden.calls) == len(generator.calls)
    assert artifact.summary["generated_candidates"] == len(generator.calls)
    assert artifact.summary["hidden_evaluations"] == len(hidden.calls)
    assert artifact.summary["transition_strata_published"] == 9
    assert artifact.summary["transition_strata_excluded"] == 5
    assert artifact.summary["transition_observations_state_aliasing_excluded"] == 0
    assert all(row.reason == "underpowered" for row in artifact.excluded_strata)
    assert all(row.depth == 1 for row in artifact.transition_model.estimates)

    serialized = json.dumps(artifact.to_mapping(), sort_keys=True)
    assert "is_error" not in serialized
    assert "task_cluster_id" not in serialized
    assert "completion" not in serialized
    assert artifact.protocol["hidden_evaluation_after_all_generation"] is True
    assert artifact.protocol["hidden_labels_serialized"] is False
    assert artifact.protocol["published_transition_depths"] == [1]
    assert artifact.protocol["verifier_parity_certificate"] == (
        verifier_parity_certificate_identity()
    )


def test_artifact_round_trip_and_exact_identity_rejection(
    calibration_bundle,
    tmp_path: Path,
) -> None:
    artifact, _, _, _, _ = calibration_bundle
    path = tmp_path / "calibration.json"
    save_calibration_artifact(artifact, path)

    restored = load_calibration_artifact(path, expected_identity=IDENTITY)
    assert restored.to_mapping() == artifact.to_mapping()
    assert restored.to_mapping()["schema"] == CALIBRATION_ARTIFACT_SCHEMA

    mismatch = replace(IDENTITY, sampler="temperature-0.2")
    with pytest.raises(BenchmarkProtocolError, match="identity mismatch"):
        load_calibration_artifact(path, expected_identity=mismatch)

    tampered = artifact.to_mapping()
    tampered["uncertainty_calibrator"]["identity"]["backend"] = "other-backend"
    with pytest.raises(BenchmarkProtocolError, match="identity mismatch"):
        CalibrationArtifact.from_mapping(tampered)

    stale_certificate = artifact.to_mapping()
    stale_certificate["protocol"]["verifier_parity_certificate"][
        "certificate_sha256"
    ] = ZERO_SHA
    with pytest.raises(BenchmarkProtocolError, match="parity certificate mismatch"):
        CalibrationArtifact.from_mapping(stale_certificate)


def test_verifier_parity_certificate_fails_closed_on_missing_tampered_or_stale_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(BenchmarkProtocolError, match="missing"):
        verifier_parity_certificate_identity(missing)

    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(VERIFIER_PARITY_CERTIFICATE_PATH.read_bytes() + b"\n")
    with pytest.raises(BenchmarkProtocolError, match="digest mismatch"):
        verifier_parity_certificate_identity(tampered)

    semantic_tamper = json.loads(
        VERIFIER_PARITY_CERTIFICATE_PATH.read_text(encoding="utf-8")
    )
    semantic_tamper["schema_version"] = 999
    semantic_path = tmp_path / "semantic-tamper.json"
    semantic_payload = (
        json.dumps(semantic_tamper, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    semantic_path.write_bytes(semantic_payload)
    monkeypatch.setattr(
        benchmark_module,
        "VERIFIER_PARITY_CERTIFICATE_SHA256",
        hashlib.sha256(semantic_payload).hexdigest(),
    )
    with pytest.raises(BenchmarkProtocolError, match="schema mismatch"):
        verifier_parity_certificate_identity(semantic_path)

    symlink = tmp_path / "certificate-link.json"
    symlink.symlink_to(VERIFIER_PARITY_CERTIFICATE_PATH)
    with pytest.raises(BenchmarkProtocolError, match="cannot read"):
        verifier_parity_certificate_identity(symlink)

    monkeypatch.setattr(
        benchmark_module,
        "VERIFIER_PARITY_CERTIFICATE_SHA256",
        benchmark_module._file_sha256(VERIFIER_PARITY_CERTIFICATE_PATH),
    )

    original_file_sha256 = benchmark_module._file_sha256

    def stale_source(path: Path) -> str:
        if path.name == "agent.py":
            return ZERO_SHA
        return original_file_sha256(path)

    monkeypatch.setattr(benchmark_module, "_file_sha256", stale_source)
    with pytest.raises(BenchmarkProtocolError, match="source bundle is stale"):
        verifier_parity_certificate_identity()


def test_calibration_and_evaluation_require_the_certified_verifier_timeout(
    calibration_bundle,
) -> None:
    artifact, _, _, _, _ = calibration_bundle
    wrong_timeout = 5.0

    with pytest.raises(BenchmarkProtocolError, match="calibration timeout"):
        calibrate_markov_humaneval(
            cases=(),
            pinned_corpus=(),
            identity=IDENTITY,
            provenance=_provenance(()),
            generator=_FakeGenerator(),
            hidden_evaluator=_FakeHiddenEvaluator(),
            config=CalibrationConfig(
                hidden_evaluator_timeout_seconds=wrong_timeout,
            ),
        )

    with pytest.raises(BenchmarkProtocolError, match="evaluation timeout"):
        evaluate_markov_humaneval(
            cases=(),
            pinned_corpus=(),
            split="heldout",
            limited=True,
            artifact=artifact,
            expected_identity=IDENTITY,
            provenance=_provenance((), split="heldout"),
            generator=_FakeGenerator(),
            hidden_evaluator=_FakeHiddenEvaluator(),
            generator_deterministic=True,
            hidden_evaluator_timeout_seconds=wrong_timeout,
        )


def _transition_row(
    task: int,
    *,
    action: ControllerAction,
) -> TransitionCalibrationObservation:
    return TransitionCalibrationObservation(
        task_cluster_id=f"task-{task}",
        context_bucket="coding",
        trigger=Trigger.VALIDATOR_FAILURE,
        depth=1,
        action=action,
        rescued=(task % 2 == 0),
        quality_delta=1.0 if task % 2 == 0 else 0.0,
        extra_output_tokens=8,
        direct_e2e_seconds=1.0,
        extra_e2e_seconds=0.5,
    )


def test_underpowered_strata_are_dropped_instead_of_published() -> None:
    repair_key = (
        "coding",
        Trigger.VALIDATOR_FAILURE,
        1,
        ControllerAction.GENERATE_REPAIR,
    )
    refine_key = (
        "coding",
        Trigger.VALIDATOR_FAILURE,
        1,
        ControllerAction.GENERATE_REFINE,
    )
    observations = [
        *(
            _transition_row(task, action=ControllerAction.GENERATE_REPAIR)
            for task in range(8)
        ),
        *(
            _transition_row(task, action=ControllerAction.GENERATE_REFINE)
            for task in range(3)
        ),
    ]

    included, excluded = filter_underpowered_transition_observations(
        observations,
        expected_keys=(repair_key, refine_key),
        min_task_clusters=8,
    )

    assert len(included) == 8
    assert {row.action for row in included} == {ControllerAction.GENERATE_REPAIR}
    assert len(excluded) == 1
    assert excluded[0].action is ControllerAction.GENERATE_REFINE
    assert excluded[0].task_clusters == 3


def test_evaluation_runs_all_five_tiers_once_and_shares_direct_generation(
    calibration_bundle,
) -> None:
    artifact, _, _, _, full_cases = calibration_bundle
    cases = split_humaneval(full_cases, "calibration")[:4]
    hidden = _FakeHiddenEvaluator()
    generator = _FakeGenerator(hidden.calls)

    result = evaluate_markov_humaneval(
        cases=cases,
        pinned_corpus=full_cases,
        split="calibration",
        limited=True,
        artifact=artifact,
        expected_identity=IDENTITY,
        provenance=_provenance(
            cases,
            leakage=True,
            split="calibration-pilot",
            policy_sha256=evaluation_policy_sha256(
                initial_max_output_tokens=64,
                bootstrap_samples=64,
                seed=19,
            ),
        ),
        generator=generator,
        hidden_evaluator=hidden,
        generator_deterministic=True,
        public_validator=_public_failure,
        initial_max_output_tokens=64,
        bootstrap_samples=64,
        seed=19,
    )

    assert result["tiers"] == [tier.value for tier in EffortTier]
    assert result["summary"]["strategy_runs"] == len(cases) * 5
    distinct_terminal_artifacts = sum(
        len(
            {
                tier["terminal_output_sha256"]
                for tier in task["tiers"].values()
            }
        )
        for task in result["tasks"]
    )
    assert result["summary"]["hidden_terminal_evaluations"] == (
        distinct_terminal_artifacts
    )
    assert result["summary"]["distinct_terminal_artifacts"] == (
        distinct_terminal_artifacts
    )
    assert result["summary"]["hidden_terminal_verdict_reuses"] == (
        len(cases) * 5 - distinct_terminal_artifacts
    )
    assert len(hidden.calls) == distinct_terminal_artifacts
    assert len(set(hidden.calls)) == len(hidden.calls)
    assert result["summary"]["shared_direct_cache_hits"] == len(cases) * 4
    # One direct plus one repair for each of the four non-low tiers.
    assert result["summary"]["backend_generation_calls"] == len(cases) * 5

    for task in result["tasks"]:
        assert set(task["tiers"]) == {tier.value for tier in EffortTier}
        low = task["tiers"][EffortTier.LOW.value]
        assert len(low["tree"]) == 1
        for tier in tuple(EffortTier)[1:]:
            assert len(task["tiers"][tier.value]["tree"]) == 2
            assert task["tiers"][tier.value]["tree"][1]["action"] in {
                action.value for action in EFFORT_PROFILES[tier].allowed_actions
            }
        direct_hashes = {
            task["tiers"][tier.value]["tree"][0]["completion_sha256"]
            for tier in EffortTier
        }
        assert len(direct_hashes) == 1


def test_identical_terminal_artifact_reuses_one_nondeterministic_verdict_per_task(
    calibration_bundle,
) -> None:
    artifact, _, _, _, full_cases = calibration_bundle
    cases = split_humaneval(full_cases, "calibration")[:3]
    hidden = _RelabelingHiddenEvaluator(cases[0].task_id)
    generator = _FakeGenerator(hidden.calls)

    result = evaluate_markov_humaneval(
        cases=cases,
        pinned_corpus=full_cases,
        split="calibration",
        limited=True,
        artifact=artifact,
        expected_identity=IDENTITY,
        provenance=_provenance(
            cases,
            leakage=True,
            split="calibration-deduplication",
            policy_sha256=evaluation_policy_sha256(
                initial_max_output_tokens=64,
                bootstrap_samples=32,
                seed=29,
            ),
        ),
        generator=generator,
        hidden_evaluator=hidden,
        generator_deterministic=True,
        public_validator=_public_success,
        initial_max_output_tokens=64,
        bootstrap_samples=32,
        seed=29,
    )

    assert len(generator.calls) == len(cases)
    assert len(hidden.calls) == len(cases)
    assert hidden.calls_by_task == {case.task_id: 1 for case in cases}
    assert result["summary"]["strategy_runs"] == len(cases) * 5
    assert result["summary"]["hidden_terminal_evaluations"] == len(cases)
    assert result["summary"]["distinct_terminal_artifacts"] == len(cases)
    assert result["summary"]["hidden_terminal_verdict_reuses"] == len(cases) * 4
    assert result["summary"]["shared_direct_cache_hits"] == len(cases) * 4
    for task in result["tasks"]:
        assert len(
            {
                tier["terminal_output_sha256"]
                for tier in task["tiers"].values()
            }
        ) == 1
        observed_verdicts = {
            (
                tier["hidden_terminal"]["passed"],
                tier["hidden_terminal"]["status"],
                tier["hidden_terminal"]["elapsed_seconds"],
            )
            for tier in task["tiers"].values()
        }
        expected_verdict = (
            {(True, "passed", 0.002)}
            if task["task_id"] == cases[0].task_id
            else {(False, "timeout", 10.0)}
        )
        assert observed_verdicts == expected_verdict


def test_identical_output_from_different_tasks_keeps_task_scoped_verdicts(
    calibration_bundle,
) -> None:
    artifact, _, _, _, full_cases = calibration_bundle
    cases = split_humaneval(full_cases, "calibration")[:2]
    hidden = _TaskSensitiveHiddenEvaluator(cases[0].task_id)
    generator = _ConstantGenerator(hidden.calls)

    result = evaluate_markov_humaneval(
        cases=cases,
        pinned_corpus=full_cases,
        split="calibration",
        limited=True,
        artifact=artifact,
        expected_identity=IDENTITY,
        provenance=_provenance(
            cases,
            leakage=True,
            split="calibration-task-scoped-deduplication",
            policy_sha256=evaluation_policy_sha256(
                initial_max_output_tokens=64,
                bootstrap_samples=32,
                seed=31,
            ),
        ),
        generator=generator,
        hidden_evaluator=hidden,
        generator_deterministic=True,
        public_validator=_public_success,
        initial_max_output_tokens=64,
        bootstrap_samples=32,
        seed=31,
    )

    assert len(generator.calls) == len(cases)
    assert len(hidden.calls) == len(cases)
    assert {task_id for task_id, _completion in hidden.calls} == {
        case.task_id for case in cases
    }
    assert {completion for _task_id, completion in hidden.calls} == {
        "identical terminal artifact"
    }
    assert result["summary"]["hidden_terminal_evaluations"] == len(cases)
    assert result["summary"]["hidden_terminal_verdict_reuses"] == len(cases) * 4
    by_task = {row["task_id"]: row for row in result["tasks"]}
    assert {
        tier["hidden_terminal"]["passed"]
        for tier in by_task[cases[0].task_id]["tiers"].values()
    } == {True}
    assert {
        tier["hidden_terminal"]["passed"]
        for tier in by_task[cases[1].task_id]["tiers"].values()
    } == {False}


def test_evaluation_schema_metrics_provenance_and_claims_fail_closed(
    calibration_bundle,
) -> None:
    artifact, _, _, _, full_cases = calibration_bundle
    cases = split_humaneval(full_cases, "heldout")[:4]
    result = evaluate_markov_humaneval(
        cases=cases,
        pinned_corpus=full_cases,
        split="heldout",
        limited=True,
        artifact=artifact,
        expected_identity=IDENTITY,
        provenance=_provenance(
            cases,
            dirty=True,
            leakage=True,
            split="heldout-limited",
            policy_sha256=evaluation_policy_sha256(
                initial_max_output_tokens=64,
                bootstrap_samples=32,
                seed=23,
            ),
        ),
        generator=_FakeGenerator(),
        hidden_evaluator=_FakeHiddenEvaluator(),
        generator_deterministic=True,
        public_validator=_public_failure,
        initial_max_output_tokens=64,
        bootstrap_samples=32,
        seed=23,
    )

    assert EVALUATION_SCHEMA == "mio.markov-effort-humaneval-evaluation.v2"
    assert result["schema"] == EVALUATION_SCHEMA
    assert result["protocol_revision"] == PROTOCOL_REVISION
    assert result["planned_comparisons"] == 4
    assert result["hidden_evaluation_policy"] == {
        "unit": "task_id_and_exact_terminal_output",
        "verdict_reused_across_tiers": True,
        "all_strategy_generation_completed_before_evaluation": True,
        "serialized_cache_keys": False,
    }
    assert benchmark_module.PREREGISTRATION[
        "hidden_terminal_evaluation_unit"
    ] == "task_id_and_exact_terminal_output"
    assert benchmark_module.PREREGISTRATION[
        "reuse_hidden_terminal_verdict_across_tiers"
    ] is True
    assert set(result["comparisons_vs_low"]) == {"medium", "high", "xhigh", "ultra"}
    assert result["claim"] == {
        "eligible": False,
        "failures": [
            "limited_run",
            "git_dirty",
            "leakage_detected",
        ],
        "requires_full_heldout_tasks": 132,
    }
    assert result["verifier_parity_certificate"] == (
        verifier_parity_certificate_identity()
    )
    assert result["provenance"]["git_dirty"] is True
    assert result["provenance"]["leakage_detected"] is True
    assert all(
        gate["heldout_claim_passed"] is False for gate in result["gates"].values()
    )

    node = result["tasks"][0]["tiers"]["low"]["tree"][0]
    assert {
        "ttft_seconds",
        "prefill_seconds",
        "decode_seconds",
        "runtime_e2e_seconds",
        "prompt_tokens",
        "output_tokens",
        "allocated_output_tokens",
        "deadline_exceeded",
    }.issubset(node)
    assert result["tasks"][0]["tiers"]["low"]["metrics"]["deadline_violations"] == 0


def test_evaluation_rejects_identity_drift_before_generation(
    calibration_bundle,
) -> None:
    artifact, _, _, _, full_cases = calibration_bundle
    cases = split_humaneval(full_cases, "calibration")[:2]
    generator = _FakeGenerator()

    with pytest.raises(BenchmarkProtocolError, match="identity mismatch"):
        evaluate_markov_humaneval(
            cases=cases,
            pinned_corpus=full_cases,
            split="calibration",
            limited=True,
            artifact=artifact,
            expected_identity=replace(IDENTITY, backend="changed-backend"),
            provenance=_provenance(cases, leakage=True),
            generator=generator,
            hidden_evaluator=_FakeHiddenEvaluator(),
            generator_deterministic=True,
            public_validator=_public_failure,
            initial_max_output_tokens=64,
            bootstrap_samples=8,
        )

    assert generator.calls == []
