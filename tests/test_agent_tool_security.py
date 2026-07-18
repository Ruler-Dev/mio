from __future__ import annotations

import json
import os
import platform
import shlex
import time
from pathlib import Path

import pytest

from mio import agent
from mio.agent_policy import (
    AgentAuditEvent,
    AgentPathViolation,
    AgentToolPermission,
    AgentToolPolicy,
    is_broad_workspace_root,
    resolve_workspace_path,
)


def _policy(
    workspace: Path,
    permissions: set[AgentToolPermission],
    events: list[AgentAuditEvent],
    *,
    output_limit_chars: int = 10_000,
    file_limit_chars: int = 4_000_000,
    command_timeout_s: float = 30.0,
) -> AgentToolPolicy:
    return AgentToolPolicy(
        workspace_roots=(workspace,),
        permissions=frozenset(permissions),
        output_limit_chars=output_limit_chars,
        file_limit_chars=file_limit_chars,
        command_timeout_s=command_timeout_s,
        audit_sink=events.append,
    )


def test_direct_mutation_and_shell_require_explicit_caller_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = tmp_path / "note.txt"
    original.write_text("before", encoding="utf-8")

    # A missing policy is a declared read-only compatibility policy, never an
    # implicit mutation/process grant.
    assert agent.tool_read("note.txt") == "before"
    assert "permission denied" in agent.tool_write("created.txt", "content")
    assert "permission denied" in agent.tool_edit("note.txt", "before", "after")
    assert "permission denied" in agent.tool_bash("python3 -c 'print(1)'")
    assert not (tmp_path / "created.txt").exists()
    assert original.read_text(encoding="utf-8") == "before"


def test_permissions_are_independent_and_every_operation_is_audited(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events: list[AgentAuditEvent] = []
    write_only = _policy(workspace, {AgentToolPermission.WRITE}, events)

    assert "written" in agent.tool_write("note.txt", "alpha", policy=write_only)
    assert "permission denied" in agent.tool_read("note.txt", policy=write_only)
    assert "permission denied" in agent.tool_edit(
        "note.txt", "alpha", "beta", policy=write_only
    )
    assert "permission denied" in agent.tool_bash("python3 -V", policy=write_only)
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "alpha"

    assert [(event.operation, event.outcome) for event in events] == [
        ("write", "ok"),
        ("read", "denied"),
        ("edit", "denied"),
        ("bash", "denied"),
    ]
    assert events[0].allowed is True
    assert all(event.allowed is False for event in events[1:])
    assert events[2].permission == AgentToolPermission.READ.value


def test_confined_paths_reject_absolute_escape_prefix_collision_and_traversal(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    outside = tmp_path / "work-elsewhere"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    events: list[AgentAuditEvent] = []
    policy = _policy(
        workspace,
        {AgentToolPermission.READ, AgentToolPermission.WRITE},
        events,
    )

    for escaped in ("../work-elsewhere/secret.txt", str(secret)):
        assert "permission denied" in agent.tool_read(escaped, policy=policy)
        assert "permission denied" in agent.tool_write(escaped, "changed", policy=policy)
    assert secret.read_text(encoding="utf-8") == "secret"
    assert all(event.allowed is False for event in events)

    with pytest.raises(ValueError, match="parent traversal"):
        resolve_workspace_path("nested/../file.txt", policy)


def test_symlink_components_and_final_symlinks_are_never_followed(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    (workspace / "secret-link.txt").symlink_to(secret)
    events: list[AgentAuditEvent] = []
    policy = _policy(
        workspace,
        {AgentToolPermission.READ, AgentToolPermission.WRITE},
        events,
    )

    assert "symlink" in agent.tool_read("secret-link.txt", policy=policy)
    assert "symlink" in agent.tool_write("escape/new.txt", "new", policy=policy)
    assert "symlink" in agent.tool_edit(
        "secret-link.txt", "secret", "changed", policy=policy
    )
    assert secret.read_text(encoding="utf-8") == "secret"
    assert not (outside / "new.txt").exists()


def test_granted_file_tools_support_nested_write_edit_and_preserve_mode(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events: list[AgentAuditEvent] = []
    policy = _policy(
        workspace,
        {AgentToolPermission.READ, AgentToolPermission.WRITE},
        events,
    )

    assert "written" in agent.tool_write("nested/script.sh", "echo old\n", policy=policy)
    script = workspace / "nested" / "script.sh"
    script.chmod(0o755)
    assert "edited" in agent.tool_edit("nested/script.sh", "old", "new", policy=policy)
    assert agent.tool_read("nested/script.sh", policy=policy) == "echo new\n"
    assert script.stat().st_mode & 0o777 == 0o755
    assert not list(workspace.rglob(".mio-agent-*.tmp"))


def test_bash_uses_argv_without_shell_caps_output_and_redacts_audit(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events: list[AgentAuditEvent] = []
    policy = _policy(
        workspace,
        {AgentToolPermission.SHELL},
        events,
        output_limit_chars=64,
    )
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return agent._BoundedCommandResult(output="x" * 200, returncode=0)

    monkeypatch.setattr(agent, "_run_bounded_process", fake_run)
    result = agent.tool_bash(
        "python3 -c \"print('audit-secret')\"", policy=policy
    )

    assert captured["argv"][:2] == ["/usr/bin/sandbox-exec", "-p"]
    profile = captured["argv"][2]
    assert "(deny default)" in profile
    assert "(allow default)" not in profile
    assert "SYS_setsid SYS_setpgid" in profile
    assert "(allow network*)" not in profile
    assert captured["argv"][-5] == "/bin/zsh"
    assert captured["argv"][-4] == "-dfc"
    assert captured["argv"][-1] == "python3 -c \"print('audit-secret')\""
    assert captured["cwd"] == workspace.resolve()
    assert captured["env"]["HOME"] == str(workspace.resolve())
    assert captured["env"]["ZDOTDIR"] == "/var/empty"
    assert captured["env"]["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert "HF_TOKEN" not in captured["env"]
    assert len(result) == policy.output_limit_chars
    assert result.endswith("... (output truncated)")
    assert events[-1].outcome == "ok"
    assert "audit-secret" not in events[-1].target
    assert events[-1].target.startswith("zsh sha256:")


def test_agent_registry_declares_trusted_policy_injection():
    assert agent.AGENT_TOOLS["read"]["permission"] is AgentToolPermission.READ
    assert agent.AGENT_TOOLS["write"]["permission"] is AgentToolPermission.WRITE
    assert agent.AGENT_TOOLS["edit"]["permission"] is AgentToolPermission.WRITE
    assert agent.AGENT_TOOLS["bash"]["permission"] is AgentToolPermission.SHELL
    assert agent.AGENT_TOOLS["list_mcp_tools"]["inject_policy"] is True
    assert agent.AGENT_TOOLS["call_mcp_tool"]["inject_policy"] is True


def test_native_cli_explicitly_declares_the_coding_policy(monkeypatch, tmp_path):
    import mio.main as main_module

    captured = {}

    class Manager:
        def __init__(self, _config):
            pass

        def load_active_tiers(self):
            pass

        def unload_all(self):
            pass

    config = type(
        "Config",
        (),
        {
            "tiers": {"large-moe": object()},
            "active_tiers": ["large-moe"],
            "tandem": False,
        },
    )()
    args = type(
        "Args",
        (),
        {
            "tier": "large-moe",
            "tandem": False,
            "tq4": False,
            "mpath": 1,
            "context": None,
            "prompt": [],
            "prompt_policy": None,
        },
    )()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mio.config.load_config", lambda: config)
    monkeypatch.setattr("mio.model_manager.ModelManager", Manager)
    monkeypatch.setattr("mio.agent.run_agent", lambda *_args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(main_module, "_apply_tq4_flag", lambda *_args: None)
    monkeypatch.setattr(main_module, "_apply_mpath_flag", lambda *_args: None)
    monkeypatch.setattr(main_module, "_apply_context_flag", lambda *_args: None)
    monkeypatch.setattr(main_module, "_configured_tier_name", lambda *_args: "large-moe")

    main_module._cmd_native_agent(args)

    policy = captured["tool_policy"]
    assert policy.workspace_roots == (tmp_path.resolve(),)
    assert policy.permissions == frozenset(
        {
            AgentToolPermission.READ,
            AgentToolPermission.WRITE,
            AgentToolPermission.SHELL,
        }
    )


def test_agent_workspace_prefers_vcs_root_and_rejects_broad_roots(monkeypatch, tmp_path):
    import mio.main as main_module

    project = tmp_path / "project"
    nested = project / "src" / "package"
    nested.mkdir(parents=True)
    (project / ".git").mkdir()
    monkeypatch.chdir(nested)

    assert main_module._agent_workspace_root(None, unsafe_broad=False) == project
    with pytest.raises(ValueError, match="refusing broad agent workspace"):
        main_module._agent_workspace_root(str(Path.home()), unsafe_broad=False)
    assert main_module._agent_workspace_root(
        str(Path.home()), unsafe_broad=True
    ) == Path.home().resolve()

    monkeypatch.chdir(Path.home())
    assert agent._default_read_policy().permissions == frozenset()


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/tmp",
        "/var",
        "/private/tmp",
        "/private/var",
        "/System",
        "/Library",
        "/Applications",
        "/usr",
        "/opt",
        "/Volumes",
        "/Network",
    ],
)
def test_system_and_alias_roots_are_always_classified_as_broad(path):
    assert is_broad_workspace_root(path) is True


def test_account_home_remains_broad_when_home_environment_is_forged(monkeypatch, tmp_path):
    try:
        import pwd
    except ImportError:
        pytest.skip("POSIX account database unavailable")
    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    monkeypatch.setenv("HOME", str(tmp_path))

    assert is_broad_workspace_root(account_home) is True


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS inherited sandbox")
def test_real_shell_sandbox_cannot_read_outside_workspace(tmp_path):
    workspace = tmp_path / 'spazio "quoted" 🚀'
    workspace.mkdir()
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("must-not-leak", encoding="utf-8")
    policy = AgentToolPolicy.coding_workspace(workspace)

    output = agent.tool_bash(
        f"python3 -c \"print(open({str(outside)!r}).read())\"",
        policy=policy,
    )

    assert "must-not-leak" not in output
    assert "Operation not permitted" in output

    # Full shell semantics remain available for coding agents and skill
    # runners, but descendants inherit the same filesystem boundary.
    piped = agent.tool_bash(
        "printf 'hermes\\n' | tr '[:lower:]' '[:upper:]' > result.txt && cat result.txt",
        policy=policy,
    )
    assert piped == "HERMES"
    assert (workspace / "result.txt").read_text(encoding="utf-8") == "HERMES\n"


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS inherited sandbox")
def test_real_shell_rejects_preexisting_hardlink_aliases(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("must-not-leak", encoding="utf-8")
    os.link(outside, workspace / "hardlink.txt")
    events: list[AgentAuditEvent] = []
    policy = _policy(
        workspace,
        {
            AgentToolPermission.READ,
            AgentToolPermission.WRITE,
            AgentToolPermission.SHELL,
        },
        events,
    )

    output = agent.tool_bash(
        "cat hardlink.txt; printf MUTATED > hardlink.txt",
        policy=policy,
    )

    assert "hard-linked workspace files are not allowed" in output
    assert "must-not-leak" not in output
    assert outside.read_text(encoding="utf-8") == "must-not-leak"
    assert events[-1].allowed is False
    assert events[-1].outcome == "denied"


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS inherited sandbox")
def test_real_shell_hardlink_audit_fails_closed_on_unlistable_directory(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    locked = workspace / "locked"
    locked.mkdir()
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("must-not-leak", encoding="utf-8")
    os.link(outside, locked / "known-hardlink.txt")
    locked.chmod(0o111)
    policy = AgentToolPolicy.coding_workspace(workspace)
    try:
        output = agent.tool_bash("cat locked/known-hardlink.txt", policy=policy)
    finally:
        locked.chmod(0o755)

    assert "cannot audit sandbox links" in output
    assert "must-not-leak" not in output


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS inherited sandbox")
def test_sandbox_runtime_roots_are_readable_but_never_writable(tmp_path):
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    workspace.mkdir()
    runtime.mkdir()
    runtime_file = runtime / "provider.txt"
    runtime_file.write_text("trusted-runtime\n", encoding="utf-8")
    policy = AgentToolPolicy.coding_workspace(workspace)
    command = (
        f"cat {shlex.quote(str(runtime_file))}; "
        f"python3 -c {shlex.quote(f'import os; print(os.path.realpath({str(runtime_file)!r}))')}; "
        f"printf MUTATED > {shlex.quote(str(runtime_file))}; "
        f"ls {shlex.quote(str(tmp_path))} >/dev/null 2>&1 && printf ANCESTOR_LISTED; "
        "printf workspace-ok > result.txt"
    )
    argv, environment = agent.sandboxed_command(
        agent._shell_argv(command, timeout_s=policy.command_timeout_s),
        policy,
        read_only_roots=(runtime,),
    )

    result = agent._run_bounded_process(
        argv,
        cwd=workspace,
        env=environment,
        timeout_s=policy.command_timeout_s,
        output_limit_chars=policy.output_limit_chars,
    )

    assert "trusted-runtime" in result.output
    assert str(runtime_file.resolve()) in result.output
    assert "ANCESTOR_LISTED" not in result.output
    assert runtime_file.read_text(encoding="utf-8") == "trusted-runtime\n"
    assert (workspace / "result.txt").read_text(encoding="utf-8") == "workspace-ok"


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS inherited sandbox")
def test_sandbox_can_exec_direct_command_while_denying_child_forks(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events: list[AgentAuditEvent] = []
    policy = _policy(
        workspace,
        {AgentToolPermission.READ, AgentToolPermission.SHELL},
        events,
    )

    direct_argv, environment = agent.sandboxed_command(
        ["/bin/sh", "-c", "exec /usr/bin/printf direct-ok"],
        policy,
        allow_process_fork=False,
    )
    direct = agent._run_bounded_process(
        direct_argv,
        cwd=workspace,
        env=environment,
        timeout_s=policy.command_timeout_s,
        output_limit_chars=policy.output_limit_chars,
    )
    assert direct.returncode == 0
    assert direct.output == "direct-ok"
    assert "(allow process-exec)" in direct_argv[2]
    assert "process-fork" not in direct_argv[2]

    child_argv, environment = agent.sandboxed_command(
        [
            "/bin/sh",
            "-c",
            "/usr/bin/printf subprocess-allowed | /usr/bin/cat",
        ],
        policy,
        allow_process_fork=False,
    )
    child = agent._run_bounded_process(
        child_argv,
        cwd=workspace,
        env=environment,
        timeout_s=policy.command_timeout_s,
        output_limit_chars=policy.output_limit_chars,
    )

    assert child.returncode != 0
    assert "Operation not permitted" in child.output
    assert "subprocess-allowed" not in child.output
    assert not list(workspace.iterdir())


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS inherited sandbox")
def test_sandbox_runtime_roots_share_the_hardlink_preflight(tmp_path):
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    workspace.mkdir()
    runtime.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.link(outside, runtime / "alias.txt")
    policy = AgentToolPolicy.coding_workspace(workspace)
    with pytest.raises(AgentPathViolation, match="hard-linked workspace files"):
        agent.sandboxed_command(
            ["/bin/echo", "ok"],
            policy,
            read_only_roots=(runtime,),
        )


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS inherited sandbox")
def test_real_shell_permissions_startup_files_and_host_channels_are_confined(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "readable.txt").write_text("workspace-secret", encoding="utf-8")
    (workspace / ".zshenv").write_text("print sourced > startup-marker\n", encoding="utf-8")
    events: list[AgentAuditEvent] = []
    shell_only = _policy(
        workspace,
        {AgentToolPermission.SHELL},
        events,
        command_timeout_s=2,
    )

    result = agent.tool_bash(
        "cat readable.txt 2>/dev/null; printf leaked > created.txt; "
        "if kill -0 $PPID 2>/dev/null; then printf SIGNAL_ALLOWED; "
        "else printf SIGNAL_DENIED; fi",
        policy=shell_only,
    )

    assert "workspace-secret" not in result
    assert "SIGNAL_DENIED" in result
    assert not (workspace / "created.txt").exists()
    assert not (workspace / "startup-marker").exists()


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS inherited sandbox")
def test_real_shell_bounds_output_closes_stdin_and_reaps_background_group(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events: list[AgentAuditEvent] = []
    policy = _policy(
        workspace,
        {
            AgentToolPermission.READ,
            AgentToolPermission.WRITE,
            AgentToolPermission.SHELL,
        },
        events,
        output_limit_chars=96,
        command_timeout_s=2,
    )

    started = time.monotonic()
    overflow = agent.tool_bash("yes mio", policy=policy)
    assert time.monotonic() - started < 2
    assert len(overflow) <= policy.output_limit_chars
    assert "output limit exceeded" in overflow

    assert agent.tool_bash("read line || printf stdin-closed", policy=policy) == "stdin-closed"
    background = agent.tool_bash(
        "sleep 30 & child=$!; printf '%s' $child > child.json; printf launched",
        policy=policy,
    )
    assert background == "launched"
    child_pid = int(json.loads((workspace / "child.json").read_text(encoding="utf-8")))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_file_limits_fifo_and_hard_links_fail_closed(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events: list[AgentAuditEvent] = []
    policy = _policy(
        workspace,
        {AgentToolPermission.READ, AgentToolPermission.WRITE},
        events,
        file_limit_chars=1_024,
    )

    assert "file-size policy limit" in agent.tool_write(
        "large.txt", "x" * 1_025, policy=policy
    )
    (workspace / "large.txt").write_text("x" * 1_025, encoding="utf-8")
    assert "editable-size policy limit" in agent.tool_edit(
        "large.txt", "x", "y", policy=policy
    )

    fifo = workspace / "input.pipe"
    os.mkfifo(fifo)
    started = time.monotonic()
    assert "not a regular file" in agent.tool_read("input.pipe", policy=policy)
    assert time.monotonic() - started < 1

    outside = tmp_path / "hardlink-secret.txt"
    outside.write_text("hardlink-secret", encoding="utf-8")
    os.link(outside, workspace / "hardlink.txt")
    hardlink_result = agent.tool_read("hardlink.txt", policy=policy)
    assert "hard-linked files are not allowed" in hardlink_result
    assert "hardlink-secret" not in hardlink_result


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -1.0, 601.0])
def test_policy_rejects_unbounded_or_invalid_timeouts(tmp_path, timeout):
    with pytest.raises(ValueError, match="command_timeout_s"):
        AgentToolPolicy(
            workspace_roots=(tmp_path,),
            permissions=frozenset(),
            command_timeout_s=timeout,
        )
