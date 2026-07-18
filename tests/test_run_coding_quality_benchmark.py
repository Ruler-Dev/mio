from __future__ import annotations

import json
import platform
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.bench_coding_quality import (
    GATE_OFF,
    GATE_ON,
    EvaluationRequest,
    GenerationObservation,
    HiddenEvaluation,
    materialize_public_fixture,
)
from scripts.run_coding_quality_benchmark import (
    CORPUS,
    ALL_SUITE_SHA256,
    DEVELOPMENT_SUITE_SHA256,
    SMOKE_SUITE_SHA256,
    CorpusHiddenEvaluator,
    RealMioGenerationRunner,
    agent_turn_to_observation,
    build_agent_tool_surface,
    execute_corpus,
    fixture_suite_sha256,
    fixture_tree_sha256,
    select_cases,
    sealed_suite_sha256,
    serialize_source_free_aggregate,
)


def test_corpus_has_frozen_smoke_and_development_splits() -> None:
    smoke = select_cases("smoke")
    development = select_cases("development")

    assert len(CORPUS) == 12
    assert [case.fixture.fixture_id for case in smoke] == ["s01", "s02", "s03", "s04"]
    assert [case.fixture.fixture_id for case in development] == [
        "d01",
        "d02",
        "d03",
        "d04",
        "d05",
        "d06",
        "d07",
        "d08",
    ]
    assert fixture_suite_sha256(tuple(case.fixture for case in smoke)) == SMOKE_SUITE_SHA256
    assert fixture_suite_sha256(tuple(case.fixture for case in development)) == DEVELOPMENT_SUITE_SHA256
    assert fixture_suite_sha256(tuple(case.fixture for case in CORPUS)) == ALL_SUITE_SHA256
    assert sealed_suite_sha256(smoke) == SMOKE_SUITE_SHA256


def test_hidden_oracles_are_not_materialized_as_public_files() -> None:
    for case in CORPUS:
        public_text = "\n".join(item.content for item in case.fixture.public_files)
        assert case.oracle.hidden_checks not in public_text
        assert case.oracle.public_regression not in public_text
        assert all("hidden" not in item.relative_name for item in case.fixture.public_files)
        assert len(case.fixture.public_files) == 2
        for public_file in case.fixture.public_files:
            compile(public_file.content, public_file.relative_name, "exec")


def test_explicit_corpus_seal_fails_closed_before_execution() -> None:
    smoke = select_cases("smoke")
    changed_fixture = replace(smoke[0].fixture, instruction=smoke[0].fixture.instruction + " changed")
    changed = (replace(smoke[0], fixture=changed_fixture), *smoke[1:])

    with pytest.raises(RuntimeError, match="explicit suite seal"):
        sealed_suite_sha256(changed)


def test_tool_surfaces_exclude_mcp_and_skills_and_only_gate_on_gets_validate() -> None:
    fake_agent = SimpleNamespace(
        AGENT_TOOLS={
            name: {"fn": object()}
            for name in [
                "bash",
                "read",
                "write",
                "edit",
                "validate",
                "list_mio_skills",
                "read_mio_skill",
                "list_mcp_tools",
                "call_mcp_tool",
            ]
        },
        AGENT_TOOLS_SPEC=[
            {"type": "function", "function": {"name": name}}
            for name in [
                "bash",
                "read",
                "write",
                "edit",
                "validate",
                "list_mio_skills",
                "read_mio_skill",
                "list_mcp_tools",
                "call_mcp_tool",
            ]
        ],
    )

    off_registry, off_specs = build_agent_tool_surface(GATE_OFF, fake_agent)
    on_registry, on_specs = build_agent_tool_surface(GATE_ON, fake_agent)

    assert tuple(off_registry) == ("bash", "read", "write", "edit")
    assert tuple(on_registry) == ("bash", "read", "write", "edit", "validate")
    assert [item["function"]["name"] for item in off_specs] == list(off_registry)
    assert [item["function"]["name"] for item in on_specs] == list(on_registry)


def test_agent_turn_adapter_uses_content_free_metrics_and_fails_closed_without_gate_record() -> None:
    rounds = (
        SimpleNamespace(completion_tokens=7, total_time_s=0.4),
        SimpleNamespace(completion_tokens=5, total_time_s=0.3),
    )
    events = (
        SimpleNamespace(operation="write", allowed=True, outcome="ok"),
        SimpleNamespace(operation="validate", allowed=True, outcome="ok"),
    )
    result = SimpleNamespace(
        terminal_reason="model_final",
        rounds=rounds,
        tool_events=events,
        tool_calls=2,
        wall_time_s=0.9,
        quality_gate={"decision": "satisfied"},
        assistant_text="PRIVATE MODEL OUTPUT",
    )

    observation = agent_turn_to_observation(result, GATE_ON)

    assert observation.completed is True
    assert observation.mutation_count == 1
    assert observation.tool_calls == 2
    assert observation.output_tokens == 12
    assert observation.validation_attempted is True
    assert observation.validation_succeeded is True
    assert observation.model_seconds == pytest.approx(0.7)
    assert observation.wall_seconds == 0.9
    result.quality_gate = None
    assert agent_turn_to_observation(result, GATE_ON).completed is False
    assert agent_turn_to_observation(result, GATE_OFF).completed is True


def test_real_runner_resets_identical_fixture_bytes_and_disables_network(tmp_path: Path) -> None:
    case = select_cases("smoke")[0]
    calls = []

    class FakeExecutor:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                terminal_reason="model_final",
                rounds=(),
                tool_events=(),
                tool_calls=0,
                wall_time_s=0.1,
                quality_gate={"decision": "not_applicable"},
            )

    runner = RealMioGenerationRunner(
        executor=FakeExecutor(),
        fixtures=(case.fixture,),
        effort="medium",
        agent_module=SimpleNamespace(
            AGENT_TOOLS={name: {"fn": object()} for name in ["bash", "read", "write", "edit", "validate"]},
            AGENT_TOOLS_SPEC=[
                {"type": "function", "function": {"name": name}}
                for name in ["bash", "read", "write", "edit", "validate"]
            ],
        ),
    )
    for condition in (GATE_OFF, GATE_ON):
        workspace = tmp_path / condition
        workspace.mkdir()
        for public_file in case.fixture.public_files:
            (workspace / public_file.relative_name).write_text(public_file.content)
        request = SimpleNamespace(
            fixture_id=case.fixture.fixture_id,
            instruction=case.fixture.instruction,
            condition=condition,
            workspace=workspace,
            schedule_index=len(calls),
        )
        observation = runner(request)
        assert observation.completed is True

    assert [call["quality_gate_enabled"] for call in calls] == [False, True]
    assert [tuple(call["tool_registry"]) for call in calls] == [
        ("bash", "read", "write", "edit"),
        ("bash", "read", "write", "edit", "validate"),
    ]
    assert all("network" not in {permission.value for permission in call["tool_policy"].permissions} for call in calls)

    # A byte change before generation is an integrity error, not a task result.
    changed = tmp_path / "changed"
    changed.mkdir()
    for public_file in case.fixture.public_files:
        (changed / public_file.relative_name).write_text(public_file.content)
    (changed / case.fixture.public_files[0].relative_name).write_text("tampered")
    with pytest.raises(RuntimeError, match="frozen initial fixture bytes"):
        runner(
            SimpleNamespace(
                fixture_id=case.fixture.fixture_id,
                instruction=case.fixture.instruction,
                condition=GATE_OFF,
                workspace=changed,
                schedule_index=2,
            )
        )


def test_execute_corpus_uses_two_phases_and_serializes_only_aggregate(tmp_path: Path) -> None:
    cases = select_cases("smoke")
    events: list[str] = []
    initial_digests = {case.fixture.fixture_id: fixture_tree_sha256(case.fixture) for case in cases}

    def fake_runner(request):
        events.append("generate")
        assert (
            fixture_tree_sha256(next(case.fixture for case in cases if case.fixture.fixture_id == request.fixture_id))
            == initial_digests[request.fixture_id]
        )
        return GenerationObservation(completed=True)

    def fake_hidden(_request):
        events.append("evaluate")
        return HiddenEvaluation(passed=False, regression_free=True)

    execution = execute_corpus(
        cases=cases,
        runner=fake_runner,
        hidden_evaluator=fake_hidden,
        work_root=tmp_path / "runs",
    )
    artifact = serialize_source_free_aggregate(execution.aggregate)
    parsed = json.loads(artifact)

    assert events == ["generate"] * 8 + ["evaluate"] * 8
    assert parsed["protocol"]["pair_count"] == 4
    assert parsed["claim_gate"]["status"] == "no_claim_insufficient_pairs"
    for case in cases:
        assert case.fixture.fixture_id not in artifact
        assert case.fixture.instruction not in artifact
        assert case.oracle.hidden_checks not in artifact
    assert str(tmp_path) not in artifact


def test_hidden_evaluator_requires_scoped_nonempty_edit_and_immutable_public_test(tmp_path: Path) -> None:
    case = select_cases("smoke")[0]
    evaluator = CorpusHiddenEvaluator((case,))
    evaluator._run_oracle = lambda _workspace, _source: True  # type: ignore[method-assign]

    def workspace(name: str) -> Path:
        root = tmp_path / name
        root.mkdir()
        for public_file in case.fixture.public_files:
            (root / public_file.relative_name).write_text(public_file.content)
        return root

    def evaluate(root: Path) -> HiddenEvaluation:
        return evaluator(
            SimpleNamespace(
                fixture_id=case.fixture.fixture_id,
                condition=GATE_ON,
                workspace=root,
                schedule_index=0,
            )
        )

    unchanged = workspace("unchanged")
    assert evaluate(unchanged).passed is False

    valid = workspace("valid")
    (valid / case.editable_names[0]).write_text("def normalize_whitespace(text):\n    return ' '.join(text.split())\n")
    assert evaluate(valid).passed is True

    changed_test = workspace("changed-test")
    (changed_test / case.editable_names[0]).write_text("def normalize_whitespace(text):\n    return text\n")
    (changed_test / "test_public_text_utils.py").write_text("# weakened")
    assert evaluate(changed_test).passed is False

    extra = workspace("extra")
    (extra / case.editable_names[0]).write_text("def normalize_whitespace(text):\n    return text\n")
    (extra / "notes.txt").write_text("unexpected")
    assert evaluate(extra).passed is False


@pytest.mark.skipif(platform.system() != "Darwin", reason="native evaluator uses the macOS process sandbox")
def test_hidden_evaluator_runs_oracles_in_read_only_sandbox(tmp_path: Path) -> None:
    case = select_cases("smoke")[0]
    workspace = materialize_public_fixture(case.fixture, tmp_path / "workspace").workspace
    (workspace / case.editable_names[0]).write_text(
        "def normalize_whitespace(text):\n    return ' '.join(text.split())\n",
        encoding="utf-8",
    )

    outcome = CorpusHiddenEvaluator((case,))(
        EvaluationRequest(
            fixture_id=case.fixture.fixture_id,
            condition=GATE_ON,
            workspace=workspace,
            schedule_index=0,
        )
    )

    assert outcome == HiddenEvaluation(passed=True, regression_free=True)
    assert not (workspace / "__pycache__").exists()
