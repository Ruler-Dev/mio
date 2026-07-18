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


def test_round_trace_propagates_raw_model_telemetry():
    trace = agent._round_trace(
        3,
        SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=4,
            total_time_s=0.5,
            prompt_tps=24.0,
            generation_tps=8.0,
            generation_backend="baseline",
            fallback_ar=False,
            prefill_ns=100,
            decode_ns=300,
            model_total_ns=400,
            logical_prompt_tokens=12,
            physical_prefill_tokens=7,
            physical_decode_tokens=4,
            warm_offset=5,
            timing_source="runtime_raw_ns",
        ),
    )

    assert trace.prefill_ns == 100
    assert trace.decode_ns == 300
    assert trace.model_total_ns == 400
    assert trace.logical_prompt_tokens == 12
    assert trace.physical_prefill_tokens == 7
    assert trace.physical_decode_tokens == 4
    assert trace.warm_offset == 5
    assert trace.warm_offset_tokens == 5
    assert trace.timing_source == "runtime_raw_ns"


def test_round_trace_rejects_instead_of_rewriting_incoherent_zero_values():
    with pytest.raises(ValueError, match="logical_prompt_tokens"):
        agent._round_trace(
            0,
            SimpleNamespace(
                prompt_tokens=3,
                logical_prompt_tokens=0,
                physical_prefill_tokens=0,
                warm_offset=0,
                completion_tokens=0,
                prefill_ns=0,
                decode_ns=0,
                model_total_ns=0,
                total_time_s=0.0,
            ),
        )


def test_builtin_validate_is_complete_but_does_not_claim_full_timeout_enforcement(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    clock = iter([100, 160])
    monkeypatch.setattr(agent.time, "perf_counter_ns", lambda: next(clock))
    state = _state(tmp_path)
    state["tool_registry"] = {
        "validate": agent.AGENT_TOOLS["validate"],
    }
    engine = _ScriptedEngine(
        [
            _tool_call("validate", argv='["git", "diff", "--check"]'),
            "Validation completed.",
        ]
    )

    result = agent._process_user_input(
        "Validate the workspace.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert len(result.tool_events) == 1
    event = result.tool_events[0]
    assert event.tool_name == "validate"
    assert event.duration_ns == 60
    assert event.effective_timeout_ns == 30_000_000_000
    assert event.exit_code_or_signal == 0
    assert event.output_chars > 0
    assert event.audit_count == 1
    assert event.timeout_enforced is False
    assert event.telemetry_complete is True
    assert result.tool_telemetry_complete is True


def test_tool_exception_without_audit_marks_turn_telemetry_incomplete(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )

    def unaudited_failure(argv, *, policy):
        del argv, policy
        raise RuntimeError("failed before audit")

    state = _state(tmp_path)
    state["tool_registry"] = {
        "validate": {
            "fn": unaudited_failure,
            "args": ["argv"],
            "permission": agent.AgentToolPermission.SHELL,
        }
    }
    engine = _ScriptedEngine(
        [
            _tool_call("validate", argv='["python3", "-m", "pytest", "-q"]'),
            "Validation could not be attested.",
        ]
    )

    result = agent._process_user_input(
        "Try validation.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert result.tool_calls == 1
    assert len(result.tool_events) == 1
    event = result.tool_events[0]
    assert event.sequence == 0
    assert event.tool_name == "validate"
    assert event.outcome == "error"
    assert event.audit_count == 0
    assert event.telemetry_complete is False
    assert result.tool_telemetry_complete is False


def test_multiple_audits_fold_into_one_contiguous_tool_record(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    clock = iter([1_000, 1_120, 2_000, 2_170])
    monkeypatch.setattr(agent.time, "perf_counter_ns", lambda: next(clock))

    def multiply_audited(command, *, policy):
        for target in ("first", "second"):
            policy.audit(
                operation="bash",
                permission=agent.AgentToolPermission.SHELL,
                target=target,
                allowed=True,
                outcome="ok",
                detail="returncode=0",
            )
        return command

    state = _state(tmp_path)
    state["tool_registry"] = {
        "bash": {
            "fn": multiply_audited,
            "args": ["command"],
            "permission": agent.AgentToolPermission.SHELL,
        }
    }
    engine = _ScriptedEngine(
        [
            _tool_call("bash", command="one") + _tool_call("bash", command="second"),
            "Both calls completed.",
        ]
    )

    result = agent._process_user_input(
        "Run both.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert result.tool_calls == 2
    assert len(result.tool_events) == 2
    assert [event.sequence for event in result.tool_events] == [0, 1]
    assert [event.audit_count for event in result.tool_events] == [2, 2]
    assert [event.duration_ns for event in result.tool_events] == [120, 170]
    assert [event.exit_code_or_signal for event in result.tool_events] == [0, 0]
    assert all(event.telemetry_complete is True for event in result.tool_events)
    assert result.tool_telemetry_complete is True


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ("returncode=7; output_chars=6", 7),
        ("returncode=-15; output_chars=6", "signal:15"),
        ("signal=9; output_chars=6", "signal:9"),
    ],
)
def test_tool_trace_normalizes_exit_code_or_signal(detail, expected):
    event = agent.AgentAuditEvent(
        timestamp=1.0,
        operation="bash",
        permission="shell",
        target="zsh sha256:test",
        allowed=True,
        outcome="nonzero",
        detail=detail,
    )

    trace = agent._tool_trace(
        sequence=0,
        round_index=0,
        tool_name="bash",
        args={"command": "hidden"},
        events=(event,),
        result="result",
        fallback_outcome="error",
        duration_ns=10,
        effective_timeout_ns=300_000_000_000,
        timeout_enforced=False,
        telemetry_complete=True,
        known_tool=True,
        permission_fallback="shell",
    )

    assert trace.exit_code_or_signal == expected
    assert trace.output_chars == 6


def test_unknown_tool_uses_sentinel_and_argument_commitment(monkeypatch, tmp_path):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    engine = _ScriptedEngine(
        [
            _tool_call("secret-unregistered-tool", path="private.py"),
            "Unknown tool was rejected.",
        ]
    )

    result = agent._process_user_input(
        "Try an unknown tool.",
        engine,
        _Manager(),
        MioConfig.default(),
        _state(tmp_path),
    )

    assert result.tool_calls == 1
    assert len(result.tool_events) == 1
    event = result.tool_events[0]
    assert event.sequence == 0
    assert event.tool_name == "unknown"
    assert event.operation == "unknown"
    assert event.allowed is False
    assert event.outcome == "unrecognized"
    assert len(event.target_sha256) == 64
    assert "secret" not in event.target_sha256
    assert event.audit_count == 0
    assert event.telemetry_complete is True
    assert result.tool_telemetry_complete is True


def test_denied_file_invocation_has_one_terminable_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    state = _state(tmp_path)
    state["tool_policy"] = AgentToolPolicy.read_only(tmp_path)
    engine = _ScriptedEngine(
        [
            _tool_call("write", path="blocked.py", content="VALUE = 1\n"),
            "Write was denied.",
        ]
    )

    result = agent._process_user_input(
        "Try writing.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert result.tool_calls == 1
    assert len(result.tool_events) == 1
    event = result.tool_events[0]
    assert event.sequence == 0
    assert event.tool_name == "write"
    assert event.allowed is False
    assert event.outcome == "denied"
    assert event.effective_timeout_ns == 30_000_000_000
    assert event.timeout_enforced is True
    assert event.output_chars > 0
    assert event.audit_count == 1
    assert result.tool_telemetry_complete is True


def test_file_worker_timeout_includes_spawn_and_confirms_termination(tmp_path):
    (tmp_path / "value.txt").write_text("value", encoding="utf-8")
    policy = AgentToolPolicy.read_only(tmp_path)

    result = agent._run_terminable_file_tool(
        "read",
        {"path": "value.txt"},
        policy,
        timeout_ns=1,
    )

    assert result.outcome == "timeout"
    assert result.timed_out is True
    assert result.duration_ns >= 1
    assert result.termination_confirmed is True
    assert result.telemetry_complete is False
    assert result.events == ()


def test_supervised_file_timeout_gets_one_parent_audit_record(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )

    calls = 0

    def supervised_timeout(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return agent._TerminableToolResult(
            result="(tool timed out)",
            events=(),
            outcome="timeout",
            timed_out=True,
            duration_ns=30_000_000_010,
            termination_confirmed=True,
            telemetry_complete=True,
        )

    monkeypatch.setattr(agent, "_run_terminable_file_tool", supervised_timeout)
    clock = iter([100, 700])
    monkeypatch.setattr(agent.time, "perf_counter_ns", lambda: next(clock))
    engine = _ScriptedEngine(
        [
            _tool_call("read", path="blocked.txt") + _tool_call("read", path="must-not-run.txt"),
            "Read timed out.",
        ]
    )

    result = agent._process_user_input(
        "Read once.",
        engine,
        _Manager(),
        MioConfig.default(),
        _state(tmp_path),
    )

    assert result.tool_calls == 1
    assert calls == 1
    assert len(result.tool_events) == 1
    event = result.tool_events[0]
    assert event.outcome == "timeout"
    assert event.allowed is True
    assert event.duration_ns == 600
    assert event.effective_timeout_ns == 30_000_000_000
    assert event.timeout_enforced is True
    assert event.audit_count == 1
    assert event.effect_unknown is True
    assert event.telemetry_complete is False
    assert result.tool_telemetry_complete is False
    assert result.terminal_reason == "tool_timeout"
    assert len(engine.requests) == 1


def test_command_timeout_is_terminal_but_telemetry_fields_remain_complete(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    (tmp_path / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    invocations = 0

    def timed_validation(argv, *, policy):
        nonlocal invocations
        del argv
        invocations += 1
        policy.audit(
            operation="validate",
            permission=agent.AgentToolPermission.SHELL,
            target="test:python3 sha256:preflight-error",
            allowed=True,
            outcome="error",
            detail="preflight cleanup error",
        )
        policy.audit(
            operation="validate",
            permission=agent.AgentToolPermission.SHELL,
            target="test:python3 sha256:timeout",
            allowed=True,
            outcome="timeout",
            detail="returncode=-15; output_chars=0",
        )
        return "(validation timed out)"

    state = _enable_quality_gate(_state(tmp_path))
    state["tool_registry"] = {
        "edit": agent.AGENT_TOOLS["edit"],
        "validate": {
            "fn": timed_validation,
            "args": ["argv"],
            "permission": agent.AgentToolPermission.SHELL,
        },
    }
    engine = _ScriptedEngine(
        [
            _tool_call(
                "edit",
                path="value.py",
                old="VALUE = 1",
                new="VALUE = 2",
            )
            + _tool_call("validate", argv='["python3", "-m", "pytest", "-q"]')
            + _tool_call("validate", argv='["python3", "-m", "pytest", "-q"]'),
            "Must not be generated.",
        ]
    )

    result = agent._process_user_input(
        "Change value.py and validate it.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert invocations == 1
    assert len(engine.requests) == 1
    assert result.tool_calls == 2
    assert result.terminal_reason == "tool_timeout"
    event = result.tool_events[1]
    assert event.outcome == "timeout"
    assert event.timeout_enforced is False
    assert event.effect_unknown is False
    assert event.audit_count == 2
    assert event.telemetry_complete is True
    assert result.tool_telemetry_complete is True
    assert result.quality_gate is not None
    assert result.quality_gate["decision"] == "incomplete"


def test_known_tool_without_audit_is_incomplete_even_without_permission_field(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    state = _state(tmp_path)
    state["tool_registry"] = {
        "catalog": {
            "fn": lambda: "catalog result",
            "args": [],
        }
    }
    engine = _ScriptedEngine([_tool_call("catalog"), "Catalog inspection completed."])

    result = agent._process_user_input(
        "Inspect catalog.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert result.tool_calls == 1
    event = result.tool_events[0]
    assert event.tool_name == "catalog"
    assert event.audit_count == 0
    assert event.telemetry_complete is False
    assert result.tool_telemetry_complete is False


def test_file_trace_duration_uses_full_parent_interval_not_worker_metric(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    worker_audit = agent.AgentAuditEvent(
        timestamp=1.0,
        operation="read",
        permission="read",
        target="bounded.txt",
        allowed=True,
        outcome="ok",
        detail="output_chars=2",
    )

    def completed_worker(*_args, **_kwargs):
        return agent._TerminableToolResult(
            result="ok",
            events=(worker_audit,),
            outcome="ok",
            timed_out=False,
            duration_ns=1,
            termination_confirmed=True,
            telemetry_complete=True,
        )

    monkeypatch.setattr(agent, "_run_terminable_file_tool", completed_worker)
    clock = iter([1_000, 1_900])
    monkeypatch.setattr(agent.time, "perf_counter_ns", lambda: next(clock))
    engine = _ScriptedEngine([_tool_call("read", path="bounded.txt"), "Read completed."])

    result = agent._process_user_input(
        "Read once.",
        engine,
        _Manager(),
        MioConfig.default(),
        _state(tmp_path),
    )

    event = result.tool_events[0]
    assert event.duration_ns == 900
    assert event.duration_ns != 1
    assert event.timeout_enforced is True
    assert event.telemetry_complete is True


@pytest.mark.parametrize(
    ("permissions", "expected_allowed"),
    [
        (frozenset({agent.AgentToolPermission.WRITE}), False),
        (
            frozenset(
                {
                    agent.AgentToolPermission.READ,
                    agent.AgentToolPermission.WRITE,
                }
            ),
            True,
        ),
    ],
)
def test_synthetic_edit_timeout_requires_read_and_write_grants(
    monkeypatch,
    tmp_path,
    permissions,
    expected_allowed,
):
    monkeypatch.setattr(
        agent,
        "console",
        Console(file=StringIO(), force_terminal=False, color_system=None),
    )

    def supervised_timeout(*_args, **_kwargs):
        return agent._TerminableToolResult(
            result="(tool timed out)",
            events=(),
            outcome="timeout",
            timed_out=True,
            duration_ns=30_000_000_001,
            termination_confirmed=True,
            telemetry_complete=False,
        )

    monkeypatch.setattr(agent, "_run_terminable_file_tool", supervised_timeout)
    state = _state(tmp_path)
    state["tool_policy"] = AgentToolPolicy(
        workspace_roots=(tmp_path,),
        permissions=permissions,
    )
    engine = _ScriptedEngine(
        [
            _tool_call("edit", path="value.py", old="one", new="two"),
            "Must not be generated.",
        ]
    )

    result = agent._process_user_input(
        "Edit once.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    event = result.tool_events[0]
    assert event.allowed is expected_allowed
    assert event.outcome == "timeout"
    assert event.effect_unknown is True
    assert event.telemetry_complete is False
    assert result.terminal_reason == "tool_timeout"
    assert len(engine.requests) == 1


def test_worker_terminate_and_kill_share_one_second_grace(monkeypatch):
    clock = {"now": 0.0}
    joins: list[float] = []

    class StubbornProcess:
        def is_alive(self):
            return True

        def terminate(self):
            return None

        def kill(self):
            return None

        def join(self, timeout):
            joins.append(timeout)
            clock["now"] += timeout

    monkeypatch.setattr(agent.time, "monotonic", lambda: clock["now"])
    process = StubbornProcess()

    assert (
        agent._terminate_tool_worker(
            process,
            deadline_monotonic=1.0,
        )
        is False
    )
    assert sum(joins) == pytest.approx(1.0)
    assert clock["now"] == pytest.approx(1.0)

    # A cleanup retry receives the same exhausted deadline, not another grace.
    agent._terminate_tool_worker(process, deadline_monotonic=1.0)
    assert clock["now"] == pytest.approx(1.0)


def test_file_timeout_cap_uses_remaining_arm_wall_and_blocks_custom_worker(
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

    invoked = False

    def custom_read(path, *, policy):
        nonlocal invoked
        del path, policy
        invoked = True
        return "must not run"

    state = _state(tmp_path)
    state["execution_budget"] = agent.AgentExecutionBudget(
        max_rounds=2,
        max_wall_seconds=0.25,
    )
    state["tool_registry"] = {
        "read": {
            "fn": custom_read,
            "args": ["path"],
            "permission": agent.AgentToolPermission.READ,
        }
    }
    engine = _ScriptedEngine([_tool_call("read", path="x.py"), "Read completed."])

    result = agent._process_user_input(
        "Read once.",
        engine,
        _Manager(),
        MioConfig.default(),
        state,
    )

    assert result.tool_calls == 1
    event = result.tool_events[0]
    assert invoked is False
    assert event.effective_timeout_ns == 250_000_000
    assert event.timeout_enforced is False
    assert event.outcome == "error"
    assert event.telemetry_complete is False
    assert result.tool_telemetry_complete is False


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
    state["allow_unterminable_custom_file_tools"] = True

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
    state["allow_unterminable_custom_file_tools"] = True
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
    state["allow_unterminable_custom_file_tools"] = True
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
    assert len(result.tool_events) == 1
    assert result.tool_events[0].effective_timeout_ns == 500_000_000
    assert result.tool_events[0].timeout_enforced is False
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
    assert "Coding-quality gate incomplete" in engine.requests[1][-1]["content"]
    assert "Coding-quality gate incomplete" in engine.requests[2][-1]["content"]
    assert engine.requests[3][-1]["role"] == "tool"
    assert "Coding-quality gate incomplete" not in engine.requests[3][-1]["content"]
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
