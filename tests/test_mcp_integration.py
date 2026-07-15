from __future__ import annotations

import asyncio
import json
import sys

from mio import agent
from mio.prompt_policy import PromptMode, PromptPolicy


def test_native_agent_exposes_bounded_mcp_bridge(monkeypatch):
    monkeypatch.setattr("mio.mcp.list_mcp_tools", lambda server: {"server": server, "tools": []})
    monkeypatch.setattr(
        "mio.mcp.call_mcp_tool",
        lambda server, name, arguments: {"server": server, "tool": name, "result": arguments},
    )
    assert json.loads(agent.tool_list_mcp_tools("ponytail"))["server"] == "ponytail"
    result = json.loads(agent.tool_call_mcp_tool("ponytail", "instructions", {"mode": "lite"}))
    assert result["result"] == {"mode": "lite"}
    spec_names = {spec["function"]["name"] for spec in agent.AGENT_TOOLS_SPEC}
    assert {"list_mcp_tools", "call_mcp_tool"} <= spec_names


def test_webui_skills_dispatch_mcp_bridge(monkeypatch):
    from mio.webui import skills

    monkeypatch.setattr("mio.mcp.list_mcp_tools", lambda server: {"server": server, "tools": []})
    monkeypatch.setattr(
        "mio.mcp.call_mcp_tool",
        lambda server, name, arguments: {"server": server, "tool": name, "result": arguments},
    )
    assert skills.execute_skill("list_mcp_tools", {"server": "llm-wiki"})["server"] == "llm-wiki"
    result = skills.execute_skill(
        "call_mcp_tool",
        {"server": "llm-wiki", "name": "llm_wiki_search", "arguments": {"query": "mlx"}},
    )
    assert result["result"] == {"query": "mlx"}
    tool_names = {spec["function"]["name"] for spec in skills.get_tools_spec()}
    assert {"list_mcp_tools", "call_mcp_tool"} <= tool_names


def test_native_slash_command_switches_ponytail_policy():
    state = {"prompt_policy": PromptPolicy()}
    result = agent.handle_slash_command("/ponytail ultra", None, None, state)
    assert result == "Prompt policy: **ponytail/ultra**"
    assert state["prompt_policy"].label == "ponytail/ultra"
    agent.handle_slash_command("/ponytail off", None, None, state)
    assert state["prompt_policy"].mode is PromptMode.NONE


def test_top_level_and_serve_cli_keep_separate_prompt_destinations(monkeypatch):
    import mio.main as main_module

    captured = []
    monkeypatch.setattr(main_module, "_cmd_native_agent", lambda args: captured.append(args.prompt_policy))
    monkeypatch.setattr(sys, "argv", ["mio", "--ponytail", "ultra"])
    main_module.main()
    assert captured.pop().label == "ponytail/ultra"

    monkeypatch.setattr(main_module, "_cmd_serve", lambda args: captured.append(args.prompt_policy))
    monkeypatch.setattr(sys, "argv", ["mio", "serve", "--ponytail", "lite"])
    main_module.main()
    assert captured.pop().label == "ponytail/lite"


def test_webui_mount_preserves_full_ponytail_policy(monkeypatch, tmp_path):
    from mio.webui import router, scheduler

    class Manager:
        config = type("Config", (), {"tiers": {}})()

        def loaded_tiers(self):
            return []

    monkeypatch.setattr(scheduler, "init", lambda manager, **kwargs: None)
    policy = PromptPolicy.resolve(ponytail="ultra")
    router.mount_webui(Manager(), prompt_policy=policy, sessions_dir=tmp_path / "sessions")

    assert router._prompt_policy.label == "ponytail/ultra"
    assert router._caveman_level == "off"
    # Legacy clients echo caveman=off; that must not erase a mounted Ponytail policy.
    assert router._resolve_chat_prompt_policy({"caveman": "off"}).label == "ponytail/ultra"
    config = asyncio.run(router.get_config())
    assert config["prompt_policy"] == "ponytail/ultra"

    asyncio.run(router.update_config({"prompt_mode": "none"}))
    assert router._resolve_chat_prompt_policy({"caveman": "off"}).mode is PromptMode.NONE
