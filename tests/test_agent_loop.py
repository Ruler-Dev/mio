"""End-to-end contracts for the native coding-agent tool loop."""

from __future__ import annotations

from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from mio import agent
from mio.agent_policy import AgentToolPolicy
from mio.config import MioConfig
from mio.prompt_policy import PromptPolicy


def _tool_call(name: str, **arguments: str) -> str:
    parameters = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items())
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


class _ScriptedEngine:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.requests: list[list[dict]] = []
        self.tool_specs: list[list[dict] | None] = []
        self.max_token_limits: list[int | None] = []
        self.deadlines: list[float | None] = []
        self.last_metrics = SimpleNamespace(
            generation_tps=0.0,
            completion_tokens=0,
            total_time_s=0.0,
        )

    def generate_stream(
        self,
        messages,
        *,
        tools,
        max_tokens=None,
        deadline_monotonic=None,
    ):
        self.requests.append(deepcopy(messages))
        self.tool_specs.append(tools)
        self.max_token_limits.append(max_tokens)
        self.deadlines.append(deadline_monotonic)
        response = next(self.responses)
        # Deliberately split XML delimiters across chunks to exercise the
        # incremental terminal filter as well as whole-call dispatch.
        for offset in range(0, len(response), 7):
            yield response[offset : offset + 7], None


class _MetricScriptedEngine(_ScriptedEngine):
    def __init__(
        self,
        responses: list[str],
        *,
        prompt_tokens: list[int],
        completion_tokens: list[int],
    ) -> None:
        super().__init__(responses)
        self._prompt_tokens = iter(prompt_tokens)
        self._preflight_prompt_tokens = iter(prompt_tokens)
        self._completion_tokens = iter(completion_tokens)

    def prompt_token_count(self, _messages, *, tools=None):
        del tools
        return next(self._preflight_prompt_tokens)

    def generate_stream(
        self,
        messages,
        *,
        tools,
        max_tokens=None,
        deadline_monotonic=None,
    ):
        self.last_metrics = SimpleNamespace(
            prompt_tokens=next(self._prompt_tokens),
            generation_tps=0.0,
            completion_tokens=next(self._completion_tokens),
            total_time_s=0.0,
        )
        yield from super().generate_stream(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            deadline_monotonic=deadline_monotonic,
        )


class _Manager:
    def loaded_tiers(self):
        return []

    def get_engine(self, _tier):  # pragma: no cover - guarded by loaded_tiers
        raise AssertionError("unexpected engine reload")


def _state(tmp_path):
    return {
        "tier": "test",
        "prompt_policy": PromptPolicy(),
        "tool_policy": AgentToolPolicy.coding_workspace(tmp_path),
        "messages": [],
    }


def _enable_quality_gate(state: dict, effort: str = "medium") -> dict:
    state["quality_gate_enabled"] = True
    state["coding_effort"] = effort
    return state


def _audited_test_validation(argv, *, policy):
    from mio.coding_quality import infer_validation_kind

    kind = infer_validation_kind(argv)
    assert kind is not None
    policy.audit(
        operation="validate",
        permission=agent.AgentToolPermission.SHELL,
        target=f"{kind.value}:python3 sha256:test-only",
        allowed=True,
        outcome="ok",
        detail=f"kind={kind.value}; returncode=0",
    )
    return "(validation test: PASS; returncode=0)"


def test_coding_agent_replays_structured_tool_transcript_and_finishes(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "stats.py").write_text(
        "def mean_even(values):\n    raise NotImplementedError('TODO')\n",
        encoding="utf-8",
    )
    output = StringIO()
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )
    monkeypatch.setitem(
        agent.AGENT_TOOLS,
        "bash",
        {
            "fn": lambda command, *, policy: "Ran 1 test in 0.001s\nOK",
            "args": ["command"],
            "permission": agent.AgentToolPermission.SHELL,
        },
    )
    engine = _ScriptedEngine(
        [
            _tool_call("read", path="stats.py"),
            _tool_call(
                "edit",
                path="stats.py",
                old="raise NotImplementedError('TODO')",
                new="return sum(value for value in values if value % 2 == 0)",
            ),
            _tool_call("bash", command="python3 -m unittest -v"),
            "Implemented stats.py. Tests: 1 passed.",
        ]
    )
    state = _state(tmp_path)

    result = agent._process_user_input(
        "Implement mean_even and run tests.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert "return sum(value for value in values" in (tmp_path / "stats.py").read_text()
    assert len(engine.requests) == 4
    replay = engine.requests[1]
    assert replay[-2]["role"] == "assistant"
    assert replay[-2]["content"] is None
    assert replay[-2]["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": {"path": "stats.py"},
    }
    assert replay[-1]["role"] == "tool"
    assert replay[-1]["name"] == "read"
    assert replay[-1]["tool_call_id"].startswith("call_")
    assert "NotImplementedError" in replay[-1]["content"]
    rendered = output.getvalue()
    assert "<tool_call>" not in rendered
    assert "<function=" not in rendered
    assert "◆ read(path)" in rendered
    assert "Implemented stats.py. Tests: 1 passed." in rendered
    assert state["messages"][-1] == {
        "role": "assistant",
        "content": "Implemented stats.py. Tests: 1 passed.",
    }
    assert result.assistant_text == "Implemented stats.py. Tests: 1 passed."
    assert result.terminal_reason == "model_final"
    assert len(result.rounds) == 4
    assert result.tool_calls == 3
    assert result.tool_result_chars > 0
    assert all(not trace.target_sha256.endswith("stats.py") for trace in result.tool_events)


def test_coding_agent_allows_final_synthesis_after_more_than_five_tool_rounds(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "note.txt").write_text("ready\n", encoding="utf-8")
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    engine = _ScriptedEngine(
        [_tool_call("read", path="note.txt") for _ in range(6)] + ["Final synthesis after six bounded tool rounds."],
    )
    state = _state(tmp_path)

    agent._process_user_input(
        "Inspect repeatedly, then summarize.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert len(engine.requests) == 7
    assert all(spec is agent.AGENT_TOOLS_SPEC for spec in engine.tool_specs)
    assert state["messages"][-1]["content"] == "Final synthesis after six bounded tool rounds."


def test_coding_agent_replays_multiple_calls_as_one_structured_tool_group(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two\n", encoding="utf-8")
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    engine = _ScriptedEngine(
        [
            _tool_call("read", path="one.txt") + _tool_call("read", path="two.txt"),
            "Read both files; no changes were needed.",
        ]
    )
    state = _state(tmp_path)

    agent._process_user_input(
        "Read both files and summarize.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    replay_group = engine.requests[1][-3:]
    assert replay_group[0]["role"] == "assistant"
    assert len(replay_group[0]["tool_calls"]) == 2
    assert [message["role"] for message in replay_group[1:]] == ["tool", "tool"]
    assert [message["content"] for message in replay_group[1:]] == ["one\n", "two\n"]
    assert replay_group[1]["tool_call_id"] != replay_group[2]["tool_call_id"]


def test_coding_agent_drops_unterminated_tool_xml_from_output_and_history(
    monkeypatch,
    tmp_path,
):
    output = StringIO()
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )
    engine = _ScriptedEngine(
        [
            "I inspected the request.\n<tool_call>\n<function=read>\n<parameter=path>\nsecret.txt\n",
        ]
    )
    state = _state(tmp_path)

    agent._process_user_input(
        "Inspect safely.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    rendered = output.getvalue()
    assert "I inspected the request." in rendered
    assert "<tool_call>" not in rendered
    assert state["messages"][-1]["content"] == "I inspected the request."


def test_coding_agent_reserves_twelfth_round_for_budget_synthesis(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "note.txt").write_text("ready\n", encoding="utf-8")
    output = StringIO()
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )
    engine = _ScriptedEngine(
        [_tool_call("read", path="note.txt") for _ in range(11)]
        + ["Stopped cleanly: no files changed; latest read succeeded; work remains."],
    )
    state = _state(tmp_path)

    agent._process_user_input(
        "Inspect until the bounded loop stops.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert len(engine.requests) == agent._MAX_AGENT_ROUNDS_PER_TURN == 12
    assert all(spec is agent.AGENT_TOOLS_SPEC for spec in engine.tool_specs[:-1])
    assert engine.tool_specs[-1] is None
    assert "Tool execution must stop now (model round limit 12 reached)" in (engine.requests[-1][-1]["content"])
    assert state["messages"][-1]["content"].endswith("work remains.")
    assert "<tool_call>" not in output.getvalue()


def test_coding_agent_never_dispatches_memorized_tool_xml_on_final_round(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "note.txt").write_text("ready\n", encoding="utf-8")
    output = StringIO()
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )
    calls = []
    original_read = agent.AGENT_TOOLS["read"]
    monkeypatch.setitem(
        agent.AGENT_TOOLS,
        "read",
        {
            **original_read,
            "fn": lambda path, *, policy: calls.append(path) or "ready",
        },
    )
    engine = _ScriptedEngine(
        [_tool_call("read", path="note.txt") for _ in range(12)],
    )
    state = _state(tmp_path)

    agent._process_user_input(
        "Keep reading.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert len(engine.requests) == 12
    assert len(calls) == 11
    assert engine.tool_specs[-1] is None
    assert "no additional operation was executed" in state["messages"][-1]["content"]
    assert "<tool_call>" not in output.getvalue()


@pytest.mark.parametrize(("call_cap", "requested_calls"), [(2, 3), (32, 33)])
def test_trusted_execution_budget_blocks_before_call_beyond_cap(
    monkeypatch,
    tmp_path,
    call_cap,
    requested_calls,
):
    (tmp_path / "note.txt").write_text("ready\n", encoding="utf-8")
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    dispatched: list[str] = []
    state = _state(tmp_path)
    state["execution_budget"] = agent.AgentExecutionBudget(
        max_rounds=3,
        max_tool_calls=call_cap,
    )
    state["tool_registry"] = {
        "read": {
            "fn": lambda path, *, policy: dispatched.append(path) or "ready\n",
            "args": ["path"],
            "permission": agent.AgentToolPermission.READ,
        }
    }
    engine = _ScriptedEngine(
        [
            "".join(_tool_call("read", path=f"{index}.txt") for index in range(requested_calls)),
            "Stopped after the trusted tool budget.",
        ]
    )

    result = agent._process_user_input(
        "Read at most the caller-authorized number of files.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert dispatched == [f"{index}.txt" for index in range(call_cap)]
    assert result.tool_calls == call_cap
    assert result.terminal_reason == "budget_finalization"
    assert result.budget_exhaustion == f"tool call limit {call_cap} reached"
    assert engine.tool_specs == [agent.AGENT_TOOLS_SPEC, None]


def test_tool_budget_exhaustion_preserves_incomplete_quality_report(
    monkeypatch,
    tmp_path,
):
    target = tmp_path / "stats.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    state = _enable_quality_gate(_state(tmp_path))
    state["execution_budget"] = agent.AgentExecutionBudget(
        max_rounds=3,
        max_tool_calls=1,
    )
    engine = _ScriptedEngine(
        [
            _tool_call("edit", path="stats.py", old="VALUE = 1", new="VALUE = 2")
            + _tool_call("validate", argv='["python3", "-m", "pytest", "-q"]'),
            "The edit was made, but validation could not run within budget.",
        ]
    )

    result = agent._process_user_input(
        "Change VALUE to 2 and validate it.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert result.tool_calls == 1
    assert result.budget_exhaustion == "tool call limit 1 reached"
    assert result.terminal_reason == "quality_incomplete"
    assert result.quality_gate is not None
    assert result.quality_gate["decision"] == "incomplete"
    assert state["quality_gate_pending"] is True


def test_completion_budget_is_cumulative_and_stops_before_later_dispatch(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    dispatched: list[str] = []
    state = _state(tmp_path)
    state["execution_budget"] = agent.AgentExecutionBudget(
        max_rounds=4,
        max_tool_calls=4,
        max_output_tokens=5,
    )
    state["tool_registry"] = {
        "read": {
            "fn": lambda path, *, policy: dispatched.append(path) or "ok",
            "args": ["path"],
            "permission": agent.AgentToolPermission.READ,
        }
    }
    engine = _MetricScriptedEngine(
        [_tool_call("read", path="first.txt"), _tool_call("read", path="second.txt")],
        prompt_tokens=[10, 12],
        completion_tokens=[3, 2],
    )

    result = agent._process_user_input(
        "Use the bounded token budget.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert dispatched == ["first.txt"]
    assert result.tool_calls == 1
    assert result.completion_tokens == 5
    assert result.budget_exhaustion == "completion token limit 5 reached"
    assert result.terminal_reason == "budget_exhausted"
    assert len(result.rounds) == 2
    assert engine.max_token_limits == [5, 2]


def test_completion_budget_never_raises_engine_configured_output_limit(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    state = _state(tmp_path)
    state["execution_budget"] = agent.AgentExecutionBudget(max_output_tokens=100)
    engine = _MetricScriptedEngine(
        ["Finished within the engine ceiling."],
        prompt_tokens=[10],
        completion_tokens=[4],
    )
    engine.tier_config = SimpleNamespace(max_output_tokens=4)

    result = agent._process_user_input(
        "Respect both limits.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert engine.max_token_limits == [4]
    assert result.completion_tokens == 4
    assert result.terminal_reason == "model_final"


def test_context_budget_uses_reported_prompt_plus_completion_without_dispatch(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    state = _state(tmp_path)
    state["execution_budget"] = agent.AgentExecutionBudget(max_context_tokens=8)
    engine = _MetricScriptedEngine(
        [_tool_call("read", path="never.txt")],
        prompt_tokens=[7],
        completion_tokens=[1],
    )

    result = agent._process_user_input(
        "Do not exceed context budget.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert result.tool_calls == 0
    assert result.budget_exhaustion == "context token limit 8 reached"
    assert result.terminal_reason == "budget_exhausted"
    assert engine.max_token_limits == [1]


def test_context_budget_blocks_before_generation_when_prompt_fills_window(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    state = _state(tmp_path)
    state["execution_budget"] = agent.AgentExecutionBudget(max_context_tokens=8)
    engine = _MetricScriptedEngine(
        ["must not be consumed"],
        prompt_tokens=[8],
        completion_tokens=[1],
    )

    result = agent._process_user_input(
        "Stop before an over-context round.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert result.rounds == ()
    assert result.tool_calls == 0
    assert result.completion_tokens == 0
    assert result.budget_exhaustion == "context token limit 8 reached"
    assert result.terminal_reason == "budget_exhausted"
    assert engine.requests == []


def test_wall_deadline_reduces_injected_command_timeout_and_blocks_next_call(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    clock = {"now": 0.0}
    monkeypatch.setattr(agent.time, "perf_counter", lambda: clock["now"])
    observed_timeouts: list[float] = []

    def bounded_bash(command, *, policy):
        observed_timeouts.append(policy.command_timeout_s)
        clock["now"] = 0.6
        return command

    state = _state(tmp_path)
    state["execution_budget"] = agent.AgentExecutionBudget(
        max_rounds=4,
        max_tool_calls=4,
        max_wall_seconds=0.5,
    )
    state["tool_registry"] = {
        "bash": {
            "fn": bounded_bash,
            "args": ["command"],
            "permission": agent.AgentToolPermission.SHELL,
        }
    }
    engine = _ScriptedEngine([_tool_call("bash", command="first") + _tool_call("bash", command="second")])

    result = agent._process_user_input(
        "Run only within the caller deadline.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert observed_timeouts == [0.5]
    assert result.tool_calls == 1
    assert result.wall_time_s == pytest.approx(0.6)
    assert result.budget_exhaustion == "wall time limit 0.5s reached"
    assert result.terminal_reason == "budget_exhausted"
    assert engine.deadlines == [0.5]


def test_execution_budget_state_rejects_untrusted_mapping(tmp_path):
    state = _state(tmp_path)
    state["execution_budget"] = {"max_tool_calls": 10_000}

    with pytest.raises(TypeError, match="AgentExecutionBudget"):
        agent._process_user_input(
            "Ignore model-selected budgets.",
            _ScriptedEngine(["unused"]),
            _Manager(),
            MioConfig.default(),
            state,
        )


def test_quality_gate_reprompts_after_edit_until_trusted_validation(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "stats.py").write_text("VALUE = 1\n", encoding="utf-8")
    output = StringIO()
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )
    monkeypatch.setitem(
        agent.AGENT_TOOLS,
        "validate",
        {
            "fn": _audited_test_validation,
            "args": ["argv"],
            "permission": agent.AgentToolPermission.SHELL,
        },
    )
    engine = _ScriptedEngine(
        [
            _tool_call("edit", path="stats.py", old="VALUE = 1", new="VALUE = 2"),
            "Implemented and complete.",
            _tool_call("validate", argv='["python3", "-m", "pytest", "-q"]'),
            "Implemented stats.py. Trusted tests passed.",
        ]
    )
    state = _enable_quality_gate(_state(tmp_path))

    result = agent._process_user_input(
        "Update stats.py to set VALUE to 2.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert (tmp_path / "stats.py").read_text() == "VALUE = 2\n"
    assert len(engine.requests) == 4
    assert "Coding-quality gate incomplete" in engine.requests[2][-1]["content"]
    assert result.terminal_reason == "model_final"
    assert result.quality_gate is not None
    assert result.quality_gate["decision"] == "pass"
    assert result.quality_gate["mutation_epoch"] == 1
    assert result.quality_gate["validation_counts"]["test"] == 1
    assert state["quality_gate_pending"] is False


def test_quality_gate_invalidates_validation_after_a_later_edit(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "stats.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    monkeypatch.setitem(
        agent.AGENT_TOOLS,
        "validate",
        {
            "fn": _audited_test_validation,
            "args": ["argv"],
            "permission": agent.AgentToolPermission.SHELL,
        },
    )
    test_call = _tool_call("validate", argv='["python3", "-m", "pytest", "-q"]')
    engine = _ScriptedEngine(
        [
            _tool_call("edit", path="stats.py", old="VALUE = 1", new="VALUE = 2"),
            test_call,
            _tool_call("edit", path="stats.py", old="VALUE = 2", new="VALUE = 3"),
            "Everything is done.",
            test_call,
            "VALUE is 3 and the current revision passed tests.",
        ]
    )
    state = _enable_quality_gate(_state(tmp_path))

    result = agent._process_user_input(
        "Change VALUE to 3 and test it.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert len(engine.requests) == 6
    assert "Coding-quality gate incomplete" in engine.requests[4][-1]["content"]
    assert result.quality_gate is not None
    assert result.quality_gate["decision"] == "pass"
    assert result.quality_gate["mutation_epoch"] == 2
    assert result.quality_gate["validation_attempts"] == 2
    assert result.quality_gate["validation_counts"]["test"] == 1


def test_quality_obligation_survives_generation_exception_after_edit(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "stats.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    monkeypatch.setitem(
        agent.AGENT_TOOLS,
        "validate",
        {
            "fn": _audited_test_validation,
            "args": ["argv"],
            "permission": agent.AgentToolPermission.SHELL,
        },
    )
    state = _enable_quality_gate(_state(tmp_path))
    interrupted = _ScriptedEngine([_tool_call("edit", path="stats.py", old="VALUE = 1", new="VALUE = 2")])

    with pytest.raises(RuntimeError, match="generator raised StopIteration"):
        agent._process_user_input(
            "Update stats.py to set VALUE to 2.",
            interrupted,
            _Manager(),
            MioConfig.default(),
            state,
        )

    pending = state.get("_quality_gate")
    assert pending is not None
    assert pending.mutation_epoch == 1
    assert state["quality_gate_pending"] is True

    resumed = _ScriptedEngine(
        [
            _tool_call("validate", argv='["python3", "-m", "pytest", "-q"]'),
            "The interrupted edit is now validated.",
        ]
    )
    result = agent._process_user_input(
        "Continue and validate the pending edit.",
        resumed,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert result.quality_gate is not None
    assert result.quality_gate["decision"] == "pass"
    assert state["quality_gate_pending"] is False


def test_passed_gate_is_refreshed_before_late_mutation_can_be_discarded(
    monkeypatch,
    tmp_path,
):
    from mio.coding_quality import CodingEffort, CodingQualityGate, ValidationKind

    target = tmp_path / "stats.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    monkeypatch.setitem(
        agent.AGENT_TOOLS,
        "validate",
        {
            "fn": _audited_test_validation,
            "args": ["argv"],
            "permission": agent.AgentToolPermission.SHELL,
        },
    )
    gate = CodingQualityGate.start([tmp_path], "change", effort=CodingEffort.MEDIUM)
    before = gate.before_tool("edit", {"path": "stats.py"})
    target.write_text("VALUE = 2\n", encoding="utf-8")
    gate.after_tool(
        "edit",
        {"path": "stats.py"},
        before=before,
        audit_events=[
            agent.AgentAuditEvent(
                timestamp=1.0,
                operation="edit",
                permission="write",
                target="stats.py",
                allowed=True,
                outcome="ok",
            )
        ],
    )
    gate.record_validation(
        ValidationKind.TEST,
        argv=("pytest", "-q"),
        allowed=True,
        outcome="ok",
    )
    assert gate.decision().satisfied is True
    target.write_text("VALUE = 3\n", encoding="utf-8")

    state = _enable_quality_gate(_state(tmp_path))
    state["_quality_gate"] = gate
    state["quality_gate_pending"] = False
    engine = _ScriptedEngine(
        [
            "The prior validation is still enough.",
            _tool_call("validate", argv='["python3", "-m", "pytest", "-q"]'),
            "The late revision is now validated.",
        ]
    )

    result = agent._process_user_input(
        "Continue.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert len(engine.requests) == 3
    assert "Coding-quality gate incomplete" in engine.requests[1][-1]["content"]
    assert result.quality_gate is not None
    assert result.quality_gate["mutation_epoch"] == 2
    assert result.quality_gate["decision"] == "pass"


def test_terminal_refresh_overrides_budget_finalization_after_late_mutation(
    monkeypatch,
    tmp_path,
):
    from mio.coding_quality import CodingQualityGate

    target = tmp_path / "note.txt"
    target.write_text("stable\n", encoding="utf-8")
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    original_refresh = CodingQualityGate.refresh
    refresh_calls = 0

    def refresh_then_mutate(gate):
        nonlocal refresh_calls
        refresh_calls += 1
        snapshot = original_refresh(gate)
        if refresh_calls == 1:
            target.write_text("late mutation\n", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(CodingQualityGate, "refresh", refresh_then_mutate)
    engine = _ScriptedEngine(
        [
            *[_tool_call("read", path="note.txt") for _ in range(11)],
            "Final status after the bounded loop.",
        ]
    )
    state = _enable_quality_gate(_state(tmp_path))

    result = agent._process_user_input(
        "Inspect repeatedly.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert refresh_calls == 2
    assert result.terminal_reason == "quality_incomplete"
    assert "Coding-quality gate: INCOMPLETE" in result.assistant_text
    assert result.quality_gate is not None
    assert result.quality_gate["decision"] == "incomplete"
