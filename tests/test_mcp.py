from __future__ import annotations

import json

import pytest

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
