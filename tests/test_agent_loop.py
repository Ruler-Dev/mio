"""End-to-end contracts for the native coding-agent tool loop."""

from __future__ import annotations

from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from mio import agent
from mio.agent_policy import AgentToolPolicy
from mio.config import MioConfig
from mio.prompt_policy import PromptPolicy


def _tool_call(name: str, **arguments: str) -> str:
    parameters = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n"
        for key, value in arguments.items()
    )
    return (
        "<tool_call>\n"
        f"<function={name}>\n"
        f"{parameters}"
        "</function>\n"
        "</tool_call>"
    )


class _ScriptedEngine:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.requests: list[list[dict]] = []
        self.tool_specs: list[list[dict] | None] = []
        self.last_metrics = SimpleNamespace(
            generation_tps=0.0,
            completion_tokens=0,
            total_time_s=0.0,
        )

    def generate_stream(self, messages, *, tools):
        self.requests.append(deepcopy(messages))
        self.tool_specs.append(tools)
        response = next(self.responses)
        # Deliberately split XML delimiters across chunks to exercise the
        # incremental terminal filter as well as whole-call dispatch.
        for offset in range(0, len(response), 7):
            yield response[offset : offset + 7], None


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
    engine = _ScriptedEngine([
        _tool_call("read", path="stats.py"),
        _tool_call(
            "edit",
            path="stats.py",
            old="raise NotImplementedError('TODO')",
            new="return sum(value for value in values if value % 2 == 0)",
        ),
        _tool_call("bash", command="python3 -m unittest -v"),
        "Implemented stats.py. Tests: 1 passed.",
    ])
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
        [_tool_call("read", path="note.txt") for _ in range(6)]
        + ["Final synthesis after six bounded tool rounds."],
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
    engine = _ScriptedEngine([
        _tool_call("read", path="one.txt") + _tool_call("read", path="two.txt"),
        "Read both files; no changes were needed.",
    ])
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
    engine = _ScriptedEngine([
        "I inspected the request.\n<tool_call>\n<function=read>\n"
        "<parameter=path>\nsecret.txt\n",
    ])
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
    assert "Tool execution must stop now (model round limit 12 reached)" in (
        engine.requests[-1][-1]["content"]
    )
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
    engine = _ScriptedEngine([
        _tool_call("edit", path="stats.py", old="VALUE = 1", new="VALUE = 2"),
        "Implemented and complete.",
        _tool_call("validate", argv='["python3", "-m", "pytest", "-q"]'),
        "Implemented stats.py. Trusted tests passed.",
    ])
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
    engine = _ScriptedEngine([
        _tool_call("edit", path="stats.py", old="VALUE = 1", new="VALUE = 2"),
        test_call,
        _tool_call("edit", path="stats.py", old="VALUE = 2", new="VALUE = 3"),
        "Everything is done.",
        test_call,
        "VALUE is 3 and the current revision passed tests.",
    ])
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
