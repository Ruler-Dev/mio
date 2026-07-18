from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.bench_coding_quality import (
    GATE_OFF,
    GATE_ON,
    BenchmarkProtocolError,
    CodingFixture,
    GenerationObservation,
    HiddenEvaluation,
    Preregistration,
    PublicFile,
    exact_discordant_p,
    fixture_suite_sha256,
    make_balanced_schedule,
    materialize_public_fixture,
    paired_bootstrap_delta,
    run_benchmark,
    serialize_source_free_aggregate,
)


def _fixtures(count: int = 4) -> tuple[CodingFixture, ...]:
    return tuple(
        CodingFixture(
            fixture_id=f"case-{index}",
            instruction=f"Repair public behavior marker {index}: SECRET_INSTRUCTION_{index}",
            public_files=(
                PublicFile(
                    relative_name="src/module.py",
                    content=f"VALUE = {index}  # SECRET_SOURCE_{index}\n",
                ),
            ),
        )
        for index in range(count)
    )


def _preregistration(fixtures: tuple[CodingFixture, ...], **overrides: object) -> Preregistration:
    values: dict[str, object] = {
        "expected_suite_sha256": fixture_suite_sha256(fixtures),
        "seed": 17,
        "bootstrap_samples": 500,
        "alpha": 0.05,
        "minimum_pairs_for_claim": 4,
    }
    values.update(overrides)
    return Preregistration(**values)  # type: ignore[arg-type]


def test_materialize_public_fixture_copies_only_declared_public_files(tmp_path: Path) -> None:
    fixture = _fixtures(1)[0]
    materialized = materialize_public_fixture(fixture, tmp_path / "workspace")

    assert materialized.fixture_id == fixture.fixture_id
    assert materialized.instruction == fixture.instruction
    assert (materialized.workspace / "src/module.py").read_text() == fixture.public_files[0].content
    assert sorted(
        path.relative_to(materialized.workspace).as_posix() for path in materialized.workspace.rglob("*")
    ) == [
        "src",
        "src/module.py",
    ]


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "a/../../escape.py", "a\\b.py", "./same.py"])
def test_public_file_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError):
        PublicFile(relative_name=name, content="x")


def test_balanced_schedule_is_deterministic_paired_and_counterbalanced() -> None:
    fixtures = _fixtures(7)
    first = make_balanced_schedule(fixtures, seed=9182)
    second = make_balanced_schedule(tuple(reversed(fixtures)), seed=9182)

    assert first == second
    assert len(first) == len(fixtures) * 2
    first_positions = [first[index].condition for index in range(0, len(first), 2)]
    assert abs(first_positions.count(GATE_OFF) - first_positions.count(GATE_ON)) <= 1
    for index in range(0, len(first), 2):
        pair = first[index : index + 2]
        assert pair[0].fixture.fixture_id == pair[1].fixture.fixture_id
        assert {pair[0].condition, pair[1].condition} == {GATE_OFF, GATE_ON}


def test_generation_for_all_runs_precedes_any_hidden_evaluation(tmp_path: Path) -> None:
    fixtures = _fixtures(3)
    events: list[tuple[str, str, str]] = []

    def runner(request):
        events.append(("generate", request.fixture_id, request.condition))
        (request.workspace / "agent-output.txt").write_text(request.condition)
        return GenerationObservation(
            completed=True,
            mutation_count=1,
            tool_calls=3,
            output_tokens=20,
            validation_attempted=request.condition == GATE_ON,
            validation_succeeded=request.condition == GATE_ON,
            model_seconds=0.4,
            wall_seconds=0.5,
        )

    def evaluator(request):
        events.append(("evaluate", request.fixture_id, request.condition))
        return HiddenEvaluation(
            passed=(request.workspace / "agent-output.txt").read_text() == GATE_ON,
            regression_free=True,
        )

    execution = run_benchmark(
        fixtures=fixtures,
        preregistration=_preregistration(fixtures),
        runner=runner,
        hidden_evaluator=evaluator,
        work_root=tmp_path / "runs",
    )

    event_kinds = [event[0] for event in events]
    assert event_kinds == ["generate"] * 6 + ["evaluate"] * 6
    assert len(execution.records) == 6
    assert execution.aggregate.integrity_gate["eligible"] is True


def test_aggregate_has_paired_statistics_and_contains_no_source_material(tmp_path: Path) -> None:
    fixtures = _fixtures(4)
    outcomes = {
        "case-0": {GATE_OFF: True, GATE_ON: True},
        "case-1": {GATE_OFF: True, GATE_ON: False},
        "case-2": {GATE_OFF: False, GATE_ON: True},
        "case-3": {GATE_OFF: False, GATE_ON: False},
    }

    def runner(request):
        return GenerationObservation(completed=True, model_seconds=1.0, wall_seconds=1.2)

    def evaluator(request):
        passed = outcomes[request.fixture_id][request.condition]
        return HiddenEvaluation(passed=passed, regression_free=passed)

    execution = run_benchmark(
        fixtures=fixtures,
        preregistration=_preregistration(fixtures),
        runner=runner,
        hidden_evaluator=evaluator,
        work_root=tmp_path / "runs",
    )
    serialized = serialize_source_free_aggregate(execution.aggregate)
    artifact = json.loads(serialized)

    assert artifact["paired_contingency"] == {
        "gate_off_only": 1,
        "both_pass": 1,
        "neither_pass": 1,
        "gate_on_only": 1,
    }
    assert artifact["primary_statistics"]["gate_on_minus_gate_off"] == 0.0
    assert artifact["primary_statistics"]["exact_discordant_p"] == 1.0
    assert artifact["claim_gate"]["eligible"] is False
    assert artifact["claim_gate"]["status"] == "no_claim_nonpositive_delta"
    for fixture in fixtures:
        assert fixture.fixture_id not in serialized
        assert fixture.instruction not in serialized
        assert fixture.public_files[0].content not in serialized
    assert str(tmp_path) not in serialized
    assert "agent-output.txt" not in serialized


def test_exact_discordant_p_matches_small_binomial_cases() -> None:
    assert exact_discordant_p(0, 0) == 1.0
    assert exact_discordant_p(0, 5) == pytest.approx(0.0625)
    assert exact_discordant_p(0, 6) == pytest.approx(0.03125)
    assert exact_discordant_p(2, 2) == 1.0


def test_paired_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    gate_off = [0, 0, 1, 1, 0, 1]
    gate_on = [1, 1, 1, 0, 1, 1]

    first = paired_bootstrap_delta(gate_off, gate_on, samples=1000, seed=44, alpha=0.05)
    second = paired_bootstrap_delta(gate_off, gate_on, samples=1000, seed=44, alpha=0.05)

    assert first == second
    assert first[0] == pytest.approx(1 / 3)
    assert first[1] <= first[0] <= first[2]


def test_suite_integrity_mismatch_fails_before_any_callback(tmp_path: Path) -> None:
    fixtures = _fixtures(2)
    callback_count = 0

    def forbidden_callback(_request):
        nonlocal callback_count
        callback_count += 1
        raise AssertionError("callback must not run")

    preregistration = Preregistration(expected_suite_sha256="0" * 64)
    with pytest.raises(BenchmarkProtocolError, match="preregistered digest"):
        run_benchmark(
            fixtures=fixtures,
            preregistration=preregistration,
            runner=forbidden_callback,
            hidden_evaluator=forbidden_callback,
            work_root=tmp_path / "runs",
        )

    assert callback_count == 0
    assert not (tmp_path / "runs").exists()


def test_claim_gate_requires_all_preregistered_evidence(tmp_path: Path) -> None:
    fixtures = _fixtures(8)

    def runner(_request):
        return GenerationObservation(completed=True)

    def evaluator(request):
        return HiddenEvaluation(passed=request.condition == GATE_ON, regression_free=True)

    execution = run_benchmark(
        fixtures=fixtures,
        preregistration=_preregistration(fixtures, minimum_pairs_for_claim=8, bootstrap_samples=1000),
        runner=runner,
        hidden_evaluator=evaluator,
        work_root=tmp_path / "runs",
    )

    assert execution.aggregate.primary_statistics["gate_on_minus_gate_off"] == 1.0
    assert execution.aggregate.primary_statistics["bootstrap_interval_low"] == 1.0
    assert execution.aggregate.primary_statistics["exact_discordant_p"] == pytest.approx(0.0078125)
    assert execution.aggregate.claim_gate["eligible"] is True
    assert execution.aggregate.claim_gate["status"] == "quality_improvement_supported"


def test_claim_gate_blocks_an_underpowered_positive_result(tmp_path: Path) -> None:
    fixtures = _fixtures(6)

    execution = run_benchmark(
        fixtures=fixtures,
        preregistration=_preregistration(fixtures, minimum_pairs_for_claim=20),
        runner=lambda _request: GenerationObservation(completed=True),
        hidden_evaluator=lambda request: HiddenEvaluation(
            passed=request.condition == GATE_ON,
            regression_free=True,
        ),
        work_root=tmp_path / "runs",
    )

    assert execution.aggregate.claim_gate["eligible"] is False
    assert execution.aggregate.claim_gate["status"] == "no_claim_insufficient_pairs"


def test_incomplete_generation_closes_integrity_and_claim_gates(tmp_path: Path) -> None:
    fixtures = _fixtures(8)

    execution = run_benchmark(
        fixtures=fixtures,
        preregistration=_preregistration(fixtures, minimum_pairs_for_claim=8),
        runner=lambda request: GenerationObservation(completed=request.schedule_index != 0),
        hidden_evaluator=lambda request: HiddenEvaluation(
            passed=request.condition == GATE_ON,
            regression_free=True,
        ),
        work_root=tmp_path / "runs",
    )

    assert execution.aggregate.integrity_gate["all_generation_completed"] is False
    assert execution.aggregate.integrity_gate["eligible"] is False
    assert execution.aggregate.claim_gate["eligible"] is False
    assert execution.aggregate.claim_gate["status"] == "no_claim_integrity_gate"


def test_public_serializer_rejects_private_execution(tmp_path: Path) -> None:
    fixtures = _fixtures(1)
    execution = run_benchmark(
        fixtures=fixtures,
        preregistration=_preregistration(fixtures, minimum_pairs_for_claim=1),
        runner=lambda _request: GenerationObservation(completed=True),
        hidden_evaluator=lambda _request: HiddenEvaluation(passed=True, regression_free=True),
        work_root=tmp_path / "runs",
    )

    with pytest.raises(TypeError, match="only SourceFreeAggregate"):
        serialize_source_free_aggregate(execution)  # type: ignore[arg-type]


def test_public_serializer_revalidates_mutable_mappings(tmp_path: Path) -> None:
    fixtures = _fixtures(1)
    execution = run_benchmark(
        fixtures=fixtures,
        preregistration=_preregistration(fixtures, minimum_pairs_for_claim=1),
        runner=lambda _request: GenerationObservation(completed=True),
        hidden_evaluator=lambda _request: HiddenEvaluation(passed=True, regression_free=True),
        work_root=tmp_path / "runs",
    )
    # Public instances built by the harness use read-only mappings.  Simulate a
    # hostile post-construction mutation to prove the serializer boundary does
    # not trust a dataclass instance solely because its type is correct.
    object.__setattr__(
        execution.aggregate,
        "claim_gate",
        {**execution.aggregate.claim_gate, "private_material": "SECRET_SOURCE"},
    )

    with pytest.raises(ValueError, match="claim_gate does not match"):
        serialize_source_free_aggregate(execution.aggregate)
