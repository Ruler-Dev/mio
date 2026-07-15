from __future__ import annotations

import json
import os
import platform
import signal
from pathlib import Path

import pytest

from mio.agent_policy import AgentToolPermission, AgentToolPolicy, sandboxed_command
from mio.mcp import (
    MCPConfigError,
    MCPPermission,
    MCPPermissionError,
    MCPProtocolError,
    MCPRegistry,
    MCPServerConfig,
    MCPTransport,
    load_registry,
)
from mio.mcp.client import HTTPResponse, StdioProvider, _default_http_sender
from mio.mcp.config import builtin_configs, default_config_path
from mio.mcp.hub import MCPHub, MCPHubError, MCPHubPolicy


def _agent_policy(
    workspace: Path,
    permissions: set[AgentToolPermission],
) -> AgentToolPolicy:
    return AgentToolPolicy(
        workspace_roots=(workspace,),
        permissions=frozenset(permissions),
    )


def test_local_servers_default_enabled_remote_and_auth_opt_in():
    local = MCPServerConfig(name="local", transport=MCPTransport.STDIO, command=("tool",))
    remote = MCPServerConfig(name="remote", transport=MCPTransport.HTTP, url="https://example.test/mcp")
    authenticated_local = MCPServerConfig(
        name="auth-local",
        transport=MCPTransport.HTTP,
        url="http://127.0.0.1:9999/mcp",
        header_env={"Authorization": "TOKEN"},
    )
    authenticated_stdio = MCPServerConfig(
        name="auth-stdio",
        transport=MCPTransport.STDIO,
        command=("tool",),
        environment_env={"API_TOKEN": "TOKEN"},
    )
    assert local.enabled is True
    assert remote.enabled is False
    assert authenticated_local.enabled is False
    assert authenticated_stdio.enabled is False
    assert MCPPermission.PROCESS in local.permissions
    assert MCPPermission.NETWORK in remote.permissions
    assert MCPPermission.SECRETS in authenticated_local.permissions
    assert MCPPermission.SECRETS in authenticated_stdio.permissions

    with pytest.raises(MCPConfigError, match="must use https"):
        MCPServerConfig(
            name="unsafe-auth",
            transport=MCPTransport.HTTP,
            url="http://example.test/mcp",
            header_env={"Authorization": "TOKEN"},
        )


def test_builtin_registry_is_mio_local_and_enabled(tmp_path):
    registry = load_registry(tmp_path / "missing.json")
    assert {config.name for config in registry.list()} == {"headroom", "llm-wiki", "ponytail"}
    assert all(config.enabled and config.is_local for config in registry.list())
    assert registry.get("headroom").command[0].endswith("/.mio/bin/headroom")
    assert registry.get("headroom").environment["HEADROOM_WORKSPACE_DIR"].endswith("/.mio/headroom")
    assert registry.get("headroom").environment["HEADROOM_PROXY_URL"] == "http://127.0.0.1:8787"
    assert MCPPermission.NETWORK in registry.get("headroom").permissions
    assert registry.get("llm-wiki").command[-1] == "mio.mcp.llm_wiki_server"
    assert registry.get("ponytail").command[-1].endswith("/ponytail/ponytail-mcp/index.js")


def test_mcp_paths_honor_mio_home(monkeypatch, tmp_path):
    home = tmp_path / "custom-mio-home"
    monkeypatch.setenv("MIO_HOME", str(home))
    monkeypatch.delenv("MIO_MCP_CONFIG", raising=False)

    configs = {config.name: config for config in builtin_configs()}
    assert default_config_path() == home / "mcp.json"
    assert configs["headroom"].command[0] == str(home / "bin" / "headroom")
    assert configs["headroom"].environment["HEADROOM_WORKSPACE_DIR"] == str(home / "headroom")
    assert configs["llm-wiki"].environment["MIO_WIKI_ROOT"] == str(home / "wiki")
    assert configs["ponytail"].command[-1].startswith(str(home / "tools"))


def test_registry_override_and_atomic_save(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "headroom",
                        "transport": "stdio",
                        "command": ["custom-headroom", "mcp", "serve"],
                        "enabled": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = load_registry(path)
    assert registry.get("headroom").enabled is False
    assert registry.get("headroom").command[0] == "custom-headroom"
    registry.set_enabled("headroom", True)
    assert registry.save() == path
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    assert path.stat().st_mode & 0o777 == 0o600


def test_provider_creation_requires_every_explicit_grant():
    config = MCPServerConfig(
        name="local",
        transport=MCPTransport.STDIO,
        command=("tool",),
        permissions=frozenset({MCPPermission.READ}),
    )
    registry = MCPRegistry([config])
    with pytest.raises(MCPPermissionError, match="process, read"):
        registry.create_provider("local")


class _Writer:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, data):
        self.data.extend(data)

    async def drain(self):
        return None

    def close(self):
        self.closed = True


class _Reader:
    def __init__(self, lines):
        self.lines = list(lines)

    async def readline(self):
        return self.lines.pop(0) if self.lines else b""


class _Process:
    def __init__(self, lines):
        self.stdin = _Writer()
        self.stdout = _Reader(lines)
        self.returncode = None

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_stdio_provider_is_lazy_and_handles_notifications():
    calls = []
    process = _Process(
        [
            b'{"jsonrpc":"2.0","method":"notifications/progress"}\n',
            b'{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n',
        ]
    )

    async def factory(*command, **kwargs):
        calls.append((command, kwargs))
        return process

    config = MCPServerConfig(name="test", transport=MCPTransport.STDIO, command=("fake", "serve"))
    registry = MCPRegistry([config])
    provider = registry.create_provider(
        "test",
        granted_permissions=config.permissions,
        process_factory=factory,
    )
    assert calls == []
    assert await provider.list_tools() == {"tools": []}
    sent = json.loads(bytes(process.stdin.data).decode().strip())
    assert sent["method"] == "tools/list"
    assert len(calls) == 1
    await provider.close()


@pytest.mark.asyncio
async def test_stdio_provider_enforces_output_limit():
    process = _Process([b"x" * 1025])

    async def factory(*command, **kwargs):
        return process

    config = MCPServerConfig(
        name="test",
        transport=MCPTransport.STDIO,
        command=("fake",),
        max_output_bytes=1024,
    )
    provider = MCPRegistry([config]).create_provider(
        "test", granted_permissions=config.permissions, process_factory=factory
    )
    with pytest.raises(MCPProtocolError, match="output limit"):
        await provider.list_tools()


@pytest.mark.asyncio
async def test_stdio_provider_discards_process_after_invalid_utf8():
    process = _Process([b'\xff\n'])

    async def factory(*command, **kwargs):
        return process

    config = MCPServerConfig(name="test", transport=MCPTransport.STDIO, command=("fake",))
    provider = StdioProvider(config, config.permissions, process_factory=factory)
    with pytest.raises(MCPProtocolError, match="invalid JSON-RPC"):
        await provider.list_tools()
    assert provider._process is None
    assert process.returncode == 0


@pytest.mark.asyncio
async def test_http_provider_parses_sse_and_keeps_session_header():
    seen = []

    async def sender(url, body, headers, timeout_s, max_output_bytes):
        seen.append((url, json.loads(body), dict(headers), timeout_s, max_output_bytes))
        request_id = json.loads(body).get("id")
        payload = json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"tools": []}})
        return HTTPResponse(
            200,
            {"Content-Type": "text/event-stream", "Mcp-Session-Id": "session-1"},
            f"event: message\ndata: {payload}\n\n".encode(),
        )

    config = MCPServerConfig(
        name="local-http",
        transport=MCPTransport.HTTP,
        url="http://127.0.0.1:8123/mcp",
    )
    provider = MCPRegistry([config]).create_provider(
        "local-http", granted_permissions=config.permissions, http_sender=sender
    )
    assert await provider.list_tools() == {"tools": []}
    assert await provider.list_tools() == {"tools": []}
    assert "Mcp-Session-Id" not in seen[0][2]
    assert seen[1][2]["Mcp-Session-Id"] == "session-1"


class _FakeProvider:
    def __init__(self, *, result=None):
        self.result = result or {"content": [{"type": "text", "text": "ok"}]}
        self.initialize_calls = 0
        self.list_calls = 0
        self.call_calls = 0
        self.close_calls = 0

    async def initialize(self):
        self.initialize_calls += 1
        return {"serverInfo": {"name": "fake"}}

    async def list_tools(self):
        self.list_calls += 1
        return {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}

    async def call_tool(self, name, arguments):
        self.call_calls += 1
        return self.result

    async def close(self):
        self.close_calls += 1


def test_hub_initializes_once_caches_schema_dispatches_and_closes():
    config = MCPServerConfig(name="fake", transport=MCPTransport.STDIO, command=("fake",))
    fake = _FakeProvider()
    factory_calls = []

    def factory(name, selected_config, granted):
        factory_calls.append((name, selected_config, granted))
        return fake

    hub = MCPHub(MCPRegistry([config]), provider_factory=factory, tool_cache_ttl_s=60)
    try:
        assert hub.list_tools("fake")["tools"][0]["name"] == "echo"
        assert hub.list_tools("fake")["tools"][0]["name"] == "echo"
        result = hub.call_tool("fake", "echo", {"value": 1})
        assert result["result"]["content"][0]["text"] == "ok"
        assert len(factory_calls) == 1
        assert fake.initialize_calls == 1
        assert fake.list_calls == 1
        assert fake.call_calls == 1
    finally:
        hub.close()
    assert fake.close_calls == 1


@pytest.mark.asyncio
async def test_hub_serializes_concurrent_first_provider_initialization():
    import asyncio

    config = MCPServerConfig(name="fake", transport=MCPTransport.STDIO, command=("fake",))
    fake = _FakeProvider()
    factory_calls = 0

    async def factory(_name, _config, _granted):
        nonlocal factory_calls
        factory_calls += 1
        await asyncio.sleep(0.01)
        return fake

    hub = MCPHub(MCPRegistry([config]), provider_factory=factory)
    providers = await asyncio.gather(*(hub._provider("fake") for _ in range(8)))

    assert all(provider is fake for provider in providers)
    assert factory_calls == 1
    assert fake.initialize_calls == 1
    await hub._close_all()
    assert fake.close_calls == 1


def test_hub_denies_disabled_and_remote_servers_before_factory():
    disabled = MCPServerConfig(
        name="disabled", transport=MCPTransport.STDIO, command=("fake",), enabled=False
    )
    remote = MCPServerConfig(
        name="remote", transport=MCPTransport.HTTP, url="https://example.test/mcp", enabled=True
    )
    called = []

    def factory(*args):
        called.append(args)
        return _FakeProvider()

    hub = MCPHub(MCPRegistry([disabled, remote]), provider_factory=factory)
    try:
        with pytest.raises(MCPHubError, match="disabled"):
            hub.list_tools("disabled")
        with pytest.raises(MCPHubError, match="remote"):
            hub.list_tools("remote")
    finally:
        hub.close()
    assert called == []


def test_authenticated_remote_requires_remote_and_auth_opt_ins():
    config = MCPServerConfig(
        name="remote-auth",
        transport=MCPTransport.HTTP,
        url="https://example.test/mcp",
        enabled=True,
        header_env={"Authorization": "MCP_TOKEN"},
    )
    grants = {config.name: config.permissions}
    with pytest.raises(MCPHubError, match="remote"):
        MCPHubPolicy(
            allow_authenticated=True,
            explicit_grants=grants,
        ).grants_for(config)
    assert (
        MCPHubPolicy(
            allow_remote=True,
            allow_authenticated=True,
            explicit_grants=grants,
        ).grants_for(config)
        == config.permissions
    )


def test_hub_rechecks_policy_before_serving_cached_tool_schema():
    config = MCPServerConfig(name="fake", transport=MCPTransport.STDIO, command=("fake",))
    registry = MCPRegistry([config])
    fake = _FakeProvider()
    hub = MCPHub(registry, provider_factory=lambda *args: fake, tool_cache_ttl_s=60)
    try:
        assert hub.list_tools("fake")["tools"][0]["name"] == "echo"
        registry.set_enabled("fake", False)
        with pytest.raises(MCPHubError, match="disabled"):
            hub.list_tools("fake")
    finally:
        hub.close()


def test_native_agent_policy_is_a_conservative_ceiling_for_every_mcp_permission(tmp_path):
    config = MCPServerConfig(
        name="capabilities",
        transport=MCPTransport.STDIO,
        command=("fake",),
        permissions=frozenset(
            {
                MCPPermission.READ,
                MCPPermission.WRITE,
                MCPPermission.FILESYSTEM_READ,
                MCPPermission.FILESYSTEM_WRITE,
                MCPPermission.NETWORK,
            }
        ),
    )
    fake = _FakeProvider()
    factory_calls = []
    hub = MCPHub(
        MCPRegistry([config]),
        provider_factory=lambda *args: factory_calls.append(args) or fake,
    )
    all_permissions = {
        AgentToolPermission.READ,
        AgentToolPermission.WRITE,
        AgentToolPermission.SHELL,
        AgentToolPermission.NETWORK,
    }
    try:
        for missing in sorted(all_permissions, key=lambda item: item.value):
            policy = _agent_policy(tmp_path, all_permissions - {missing})
            with pytest.raises(MCPHubError, match=rf"{missing.value} not granted"):
                hub.list_tools("capabilities", agent_policy=policy)
            assert factory_calls == []

        full_policy = _agent_policy(tmp_path, all_permissions)
        assert hub.list_tools("capabilities", agent_policy=full_policy)["tools"][0]["name"] == "echo"
        assert len(factory_calls) == 1

        # A less-privileged caller cannot reuse a provider/schema cached by a
        # more-privileged agent invocation.
        with pytest.raises(MCPHubError, match="network not granted"):
            hub.list_tools(
                "capabilities",
                agent_policy=_agent_policy(
                    tmp_path,
                    all_permissions - {AgentToolPermission.NETWORK},
                ),
            )
    finally:
        hub.close()


def test_native_agent_policy_never_grants_mcp_secrets(tmp_path):
    config = MCPServerConfig(
        name="credentialed",
        transport=MCPTransport.STDIO,
        command=("fake",),
        enabled=True,
        environment_env={"TOKEN": "MCP_TEST_TOKEN"},
    )
    factory_calls = []
    hub = MCPHub(
        MCPRegistry([config]),
        policy=MCPHubPolicy(
            allow_authenticated=True,
            explicit_grants={config.name: config.permissions},
        ),
        provider_factory=lambda *args: factory_calls.append(args) or _FakeProvider(),
    )
    policy = _agent_policy(
        tmp_path,
        {
            AgentToolPermission.READ,
            AgentToolPermission.WRITE,
            AgentToolPermission.SHELL,
            AgentToolPermission.NETWORK,
        },
    )
    try:
        with pytest.raises(MCPHubError, match="requires secrets"):
            hub.list_tools("credentialed", agent_policy=policy)
    finally:
        hub.close()
    assert factory_calls == []


def test_native_agent_denies_http_provider_even_with_network_grant(tmp_path):
    config = MCPServerConfig(
        name="remote",
        transport=MCPTransport.HTTP,
        url="https://example.test/mcp",
        enabled=True,
    )
    factory_calls = []
    hub = MCPHub(
        MCPRegistry([config]),
        policy=MCPHubPolicy(
            allow_remote=True,
            explicit_grants={
                config.name: frozenset(
                    {MCPPermission.NETWORK, MCPPermission.WRITE}
                )
            },
        ),
        provider_factory=lambda *args: factory_calls.append(args) or _FakeProvider(),
    )
    try:
        policy = _agent_policy(tmp_path, {AgentToolPermission.NETWORK})
        with pytest.raises(MCPHubError, match="confined stdio transport"):
            hub.list_tools("remote", agent_policy=policy)
    finally:
        hub.close()

    assert factory_calls == []


def test_hub_checks_advertised_name_and_result_limit():
    config = MCPServerConfig(
        name="fake",
        transport=MCPTransport.STDIO,
        command=("fake",),
        max_output_bytes=1024,
    )
    fake = _FakeProvider(result={"value": "x" * 2000})
    hub = MCPHub(MCPRegistry([config]), provider_factory=lambda *args: fake, max_output_bytes=1024)
    try:
        with pytest.raises(MCPHubError, match="not advertised"):
            hub.call_tool("fake", "unknown")
        with pytest.raises(MCPHubError, match="output limit"):
            hub.call_tool("fake", "echo")
    finally:
        hub.close()


@pytest.mark.asyncio
async def test_stdio_timeout_discards_process_before_next_request():
    class SlowReader:
        async def readline(self):
            import asyncio

            await asyncio.sleep(1)
            return b""

    first = _Process([])
    first.stdout = SlowReader()
    second = _Process([b'{"jsonrpc":"2.0","id":2,"result":{"ok":true}}\n'])
    processes = [first, second]

    async def factory(*command, **kwargs):
        return processes.pop(0)

    config = MCPServerConfig(
        name="test",
        transport=MCPTransport.STDIO,
        command=("fake",),
        timeout_s=0.01,
    )
    provider = StdioProvider(config, config.permissions, process_factory=factory)
    with pytest.raises(TimeoutError):
        await provider.request("first")
    assert provider._process is None
    assert first.returncode == 0
    assert await provider.request("second") == {"ok": True}
    await provider.close()


@pytest.mark.asyncio
async def test_default_http_sender_disables_redirects(monkeypatch):
    import io
    import urllib.error

    captured = {}

    class FakeOpener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                302,
                "redirect",
                {"Location": "https://other.example/mcp"},
                io.BytesIO(b"redirect blocked"),
            )

    def fake_build_opener(handler):
        captured["handler"] = handler
        return FakeOpener()

    monkeypatch.setattr("urllib.request.build_opener", fake_build_opener)
    response = await _default_http_sender(
        "https://secure.example/mcp",
        b"{}",
        {"Authorization": "secret"},
        1.0,
        1024,
    )
    assert response.status == 302
    assert captured["handler"].redirect_request(None, None, 302, "", {}, "http://evil.test") is None


@pytest.mark.asyncio
async def test_native_agent_stdio_uses_narrow_sandbox_and_sanitized_environment(
    monkeypatch,
    tmp_path,
):
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "mio-runtime"
    workspace.mkdir()
    runtime.mkdir()
    runtime_bin = runtime / "bin"
    runtime_bin.mkdir()
    executable = runtime_bin / "provider"
    executable.write_text("test", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    captured = {}
    process = _Process([b'{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n'])

    def fake_sandbox(command, policy, *, read_only_roots=()):
        captured["sandbox_command"] = command
        captured["sandbox_policy"] = policy
        captured["read_only_roots"] = tuple(read_only_roots)
        return ["sandbox-wrapper", *command], {
            "PATH": "/safe/bin",
            "HOME": str(workspace),
            "TMPDIR": str(workspace),
            "ZDOTDIR": "/var/empty",
        }

    async def factory(*command, **kwargs):
        captured["launched_command"] = command
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr("mio.mcp.client.sandboxed_command", fake_sandbox)
    config = MCPServerConfig(
        name="native",
        transport=MCPTransport.STDIO,
        command=(str(executable), "serve"),
        permissions=frozenset({MCPPermission.READ}),
        environment={"PROVIDER_MODE": "readonly"},
    )
    agent_policy = _agent_policy(
        workspace,
        {
            AgentToolPermission.READ,
            AgentToolPermission.WRITE,
            AgentToolPermission.SHELL,
            AgentToolPermission.NETWORK,
        },
    )
    provider = MCPRegistry([config]).create_provider(
        "native",
        granted_permissions=config.permissions,
        process_factory=factory,
        agent_policy=agent_policy,
        mio_runtime_root=runtime,
    )

    assert await provider.list_tools() == {"tools": []}
    assert captured["sandbox_command"] == [str(executable), "serve"]
    assert captured["launched_command"] == ("sandbox-wrapper", str(executable), "serve")
    child_policy = captured["sandbox_policy"]
    assert child_policy.workspace_roots == (workspace.resolve(),)
    assert captured["read_only_roots"] == (runtime_bin.resolve(),)
    assert child_policy.permissions == frozenset(
        {AgentToolPermission.READ, AgentToolPermission.SHELL}
    )
    assert captured["kwargs"]["start_new_session"] is True
    environment = captured["kwargs"]["env"]
    assert environment["PATH"] == "/safe/bin"
    assert environment["HOME"] == str(workspace.resolve())
    assert environment["TMPDIR"] == str(workspace.resolve())
    assert environment["MIO_HOME"] == str(runtime.resolve())
    assert environment["PROVIDER_MODE"] == "readonly"
    assert "HF_TOKEN" not in environment
    assert captured["kwargs"]["cwd"] == str(workspace.resolve())
    await provider.close()


def test_native_agent_stdio_rejects_reserved_environment_override(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "mio-runtime"
    workspace.mkdir()
    runtime.mkdir()
    monkeypatch.setattr(
        "mio.mcp.client.sandboxed_command",
        lambda command, _policy, **_kwargs: (command, {"PATH": "/safe/bin"}),
    )
    config = MCPServerConfig(
        name="native",
        transport=MCPTransport.STDIO,
        command=("provider",),
        environment={"PYTHONPATH": str(tmp_path / "injection")},
    )
    provider = StdioProvider(
        config,
        config.permissions,
        agent_policy=_agent_policy(workspace, {AgentToolPermission.SHELL}),
        mio_runtime_root=runtime,
    )

    with pytest.raises(MCPPermissionError, match="reserved variable 'PYTHONPATH'"):
        provider._launch_spec()


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"OPENAI_API_KEY": "literal-secret"}, "literal credential 'OPENAI_API_KEY'"),
        ({"PROXY_URL": "https://user:password@example.test"}, "URL cannot contain userinfo"),
    ],
)
def test_native_agent_stdio_rejects_literal_credentials(
    monkeypatch,
    tmp_path,
    environment,
    message,
):
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "mio-runtime"
    workspace.mkdir()
    runtime.mkdir()
    sandbox_calls = 0

    def fake_sandbox(command, _policy, **_kwargs):
        nonlocal sandbox_calls
        sandbox_calls += 1
        return command, {"PATH": "/safe/bin"}

    monkeypatch.setattr("mio.mcp.client.sandboxed_command", fake_sandbox)
    config = MCPServerConfig(
        name="native",
        transport=MCPTransport.STDIO,
        command=("provider",),
        environment=environment,
    )
    provider = StdioProvider(
        config,
        config.permissions,
        agent_policy=_agent_policy(workspace, {AgentToolPermission.SHELL}),
        mio_runtime_root=runtime,
    )

    with pytest.raises(MCPPermissionError, match=message):
        provider._launch_spec()
    assert sandbox_calls == 1


def test_hub_recreates_provider_when_native_agent_policy_fingerprint_changes(tmp_path):
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    config = MCPServerConfig(name="fake", transport=MCPTransport.STDIO, command=("fake",))
    providers = []
    received_policies = []

    def factory(_name, _config, _granted, agent_policy):
        provider = _FakeProvider()
        providers.append(provider)
        received_policies.append(agent_policy)
        return provider

    first_policy = _agent_policy(first_workspace, {AgentToolPermission.SHELL})
    second_policy = _agent_policy(second_workspace, {AgentToolPermission.SHELL})
    hub = MCPHub(MCPRegistry([config]), provider_factory=factory, tool_cache_ttl_s=60)
    try:
        assert hub.list_tools("fake", agent_policy=first_policy)["tools"]
        assert hub.list_tools("fake", agent_policy=second_policy)["tools"]
        assert hub.list_tools("fake", agent_policy=second_policy)["tools"]
    finally:
        hub.close()

    assert received_policies == [first_policy, second_policy]
    assert len(providers) == 2
    assert providers[0].close_calls == 1
    assert providers[1].close_calls == 1


@pytest.mark.asyncio
async def test_stdio_close_finishes_kill_and_reap_before_propagating_cancellation():
    import asyncio

    class BlockingProcess(_Process):
        def __init__(self):
            super().__init__([])
            self.terminated = asyncio.Event()
            self.reaped = asyncio.Event()
            self.killed = False
            self.wait_calls = 0

        def terminate(self):
            self.terminated.set()

        def kill(self):
            self.killed = True
            self.returncode = -9
            self.reaped.set()

        async def wait(self):
            self.wait_calls += 1
            await self.reaped.wait()
            return self.returncode

    process = BlockingProcess()
    config = MCPServerConfig(
        name="test",
        transport=MCPTransport.STDIO,
        command=("fake",),
        timeout_s=0.01,
    )
    provider = StdioProvider(config, config.permissions)
    provider._process = process

    close_task = asyncio.create_task(provider.close())
    await process.terminated.wait()
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert provider._process is None
    assert process.killed is True
    assert process.reaped.is_set()
    assert process.wait_calls >= 2


def test_builtin_native_agent_launch_specs_separate_data_and_code_roots(monkeypatch, tmp_path):
    runtime = tmp_path / "mio"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    release = runtime / "tools" / "mcp-releases" / "current" / "headroom"
    headroom_executable = release / "bin" / "headroom"
    headroom_executable.parent.mkdir(parents=True)
    headroom_executable.write_text("test", encoding="utf-8")
    runtime_bin = runtime / "bin"
    runtime_bin.mkdir(parents=True)
    (runtime_bin / "headroom").symlink_to(headroom_executable)
    ponytail_root = runtime / "tools" / "sources" / "ponytail" / "ponytail-mcp"
    ponytail_root.mkdir(parents=True)
    (ponytail_root / "index.js").write_text("test", encoding="utf-8")
    (ponytail_root.parent / "hooks").mkdir()

    monkeypatch.setenv("MIO_HOME", str(runtime))
    captured = []

    def fake_sandbox(command, policy, *, read_only_roots=()):
        captured.append((tuple(command), policy, tuple(read_only_roots)))
        return ["sandbox", *command], {"PATH": "/safe/bin"}

    monkeypatch.setattr("mio.mcp.client.sandboxed_command", fake_sandbox)
    configs = {config.name: config for config in builtin_configs()}
    permissions = {
        AgentToolPermission.READ,
        AgentToolPermission.WRITE,
        AgentToolPermission.SHELL,
        AgentToolPermission.NETWORK,
    }
    agent_policy = _agent_policy(workspace, permissions)

    for name in ("headroom", "llm-wiki", "ponytail"):
        StdioProvider(
            configs[name],
            configs[name].permissions,
            agent_policy=agent_policy,
            mio_runtime_root=runtime,
        )._launch_spec()

    (
        (headroom_command, headroom_policy, headroom_read_only),
        (wiki_command, wiki_policy, wiki_read_only),
        (ponytail_command, ponytail_policy, ponytail_read_only),
    ) = captured

    assert headroom_command[0] == str(headroom_executable.resolve())
    assert wiki_command == configs["llm-wiki"].command
    assert ponytail_command[-1] == str((ponytail_root / "index.js").resolve())
    assert headroom_policy.workspace_roots == (workspace.resolve(), (runtime / "headroom").resolve())
    assert headroom_read_only == (release.resolve(),)
    assert wiki_policy.workspace_roots == (workspace.resolve(), (runtime / "wiki").resolve())
    assert Path(__file__).resolve().parents[1] in wiki_read_only
    assert ponytail_policy.workspace_roots == (workspace.resolve(), (runtime / "config").resolve())
    assert ponytail_read_only == (ponytail_root.parent.resolve(),)
    assert runtime.resolve() not in {
        *headroom_policy.workspace_roots,
        *headroom_read_only,
        *wiki_policy.workspace_roots,
        *wiki_read_only,
        *ponytail_policy.workspace_roots,
        *ponytail_read_only,
    }
    assert (runtime / "headroom" / "config").is_dir()
    assert (runtime / "wiki").is_dir()
    assert (runtime / "config").is_dir()


@pytest.mark.skipif(platform.system() != "Darwin", reason="Apple sandbox profile test")
def test_sandbox_profile_never_grants_write_to_mcp_code_roots(tmp_path):
    workspace = tmp_path / "workspace"
    code_root = tmp_path / "runtime-code"
    workspace.mkdir()
    code_root.mkdir()
    (code_root / "provider.py").write_text("pass", encoding="utf-8")
    policy = _agent_policy(
        workspace,
        {
            AgentToolPermission.READ,
            AgentToolPermission.WRITE,
            AgentToolPermission.SHELL,
        },
    )

    command, _environment = sandboxed_command(
        ["/usr/bin/true"],
        policy,
        read_only_roots=(code_root,),
    )
    profile = command[2]
    allow_write_lines = [
        line for line in profile.splitlines() if line.startswith("(allow file-write*")
    ]
    deny_write_lines = [
        line for line in profile.splitlines() if line.startswith("(deny file-write*")
    ]
    assert all(str(code_root.resolve()) not in line for line in allow_write_lines)
    assert any(str(code_root.resolve()) in line for line in deny_write_lines)


@pytest.mark.asyncio
async def test_stdio_reaps_dead_leader_group_before_replacing_process(monkeypatch):
    class DeadProcess(_Process):
        def __init__(self):
            super().__init__([])
            self.pid = 424242
            self.returncode = 0
            self.waited = False

        async def wait(self):
            self.waited = True
            return self.returncode

    old_process = DeadProcess()
    new_process = _Process([b'{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n'])
    factory_calls = 0

    async def factory(*_command, **_kwargs):
        nonlocal factory_calls
        factory_calls += 1
        return new_process

    signals = []
    monkeypatch.setattr(
        "mio.mcp.client.os.killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )
    config = MCPServerConfig(name="test", transport=MCPTransport.STDIO, command=("fake",))
    provider = StdioProvider(config, config.permissions, process_factory=factory)
    provider._process = old_process

    assert await provider.list_tools() == {"tools": []}
    assert old_process.waited is True
    assert (old_process.pid, signal.SIGKILL) in signals
    assert factory_calls == 1
    await provider.close()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
@pytest.mark.asyncio
async def test_stdio_close_kills_real_child_group_after_leader_exits(tmp_path):
    import asyncio

    pid_file = tmp_path / "child.pid"
    process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        '/bin/sleep 30 </dev/null >/dev/null 2>&1 & echo $! > "$PID_FILE"; exit 0',
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env={"PATH": "/usr/bin:/bin", "PID_FILE": str(pid_file)},
        start_new_session=True,
    )
    assert await process.wait() == 0
    child_pid = int(pid_file.read_text(encoding="utf-8").strip())
    config = MCPServerConfig(name="test", transport=MCPTransport.STDIO, command=("fake",))
    provider = StdioProvider(config, config.permissions)
    provider._process = process

    try:
        await provider.close()
        for _ in range(100):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("background child survived provider.close()")
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
