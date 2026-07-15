"""Bounded synchronous bridge from Mio agents/UI to enabled local MCPs.

The hub owns one background asyncio loop, initializes each provider once,
caches tool discovery briefly, and closes every child at shutdown.  Mio's
default policy auto-grants only enabled, unauthenticated local providers;
remote or credential-bearing providers remain explicit CLI/application opt-in.
"""

from __future__ import annotations

import atexit
import asyncio
import concurrent.futures
import inspect
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from mio.mcp.client import MCPError, MCPProtocolError, MCPProvider
from mio.mcp.config import MCPPermission, MCPServerConfig
from mio.mcp.registry import MCPRegistry, load_registry

MAX_ARGUMENT_BYTES = 1024 * 1024
MAX_BRIDGE_OUTPUT_BYTES = 4 * 1024 * 1024


class MCPHubError(MCPError):
    pass


@dataclass(frozen=True)
class MCPHubPolicy:
    """Execution policy separate from the registry's enabled/disabled state."""

    allow_local: bool = True
    allow_remote: bool = False
    allow_authenticated: bool = False
    explicit_grants: Mapping[str, frozenset[MCPPermission]] = field(default_factory=dict)

    def grants_for(self, config: MCPServerConfig) -> frozenset[MCPPermission]:
        if not config.enabled:
            raise MCPHubError(f"MCP server {config.name!r} is disabled")

        explicit = frozenset(
            MCPPermission(value) for value in self.explicit_grants.get(config.name, frozenset())
        )
        if config.is_local:
            if not self.allow_local:
                raise MCPHubError(f"local MCP server {config.name!r} is not allowed by policy")
        elif not self.allow_remote:
            raise MCPHubError(f"remote MCP server {config.name!r} is not allowed by policy")

        if config.uses_auth:
            if not self.allow_authenticated:
                raise MCPHubError(f"authenticated MCP server {config.name!r} is not allowed by policy")
            granted = explicit
        elif config.is_local:
            # Enabling a local, unauthenticated declaration in Mio's registry
            # is the explicit local policy decision. Remote/auth never inherit it.
            granted = config.permissions
        else:
            granted = explicit

        missing = config.permissions - granted
        if missing:
            names = ", ".join(sorted(permission.value for permission in missing))
            raise MCPHubError(f"MCP server {config.name!r} still needs grants: {names}")
        return granted


ProviderFactory = Callable[[str, MCPServerConfig, frozenset[MCPPermission]], MCPProvider]


def _default_provider_factory(
    name: str,
    config: MCPServerConfig,
    granted: frozenset[MCPPermission],
) -> MCPProvider:
    registry = MCPRegistry([config])
    return registry.create_provider(name, granted_permissions=granted)


class MCPHub:
    def __init__(
        self,
        registry: MCPRegistry | None = None,
        *,
        policy: MCPHubPolicy | None = None,
        provider_factory: ProviderFactory = _default_provider_factory,
        tool_cache_ttl_s: float = 60.0,
        max_output_bytes: int = MAX_BRIDGE_OUTPUT_BYTES,
    ) -> None:
        self.registry = registry or load_registry()
        self.policy = policy or MCPHubPolicy()
        self._provider_factory = provider_factory
        self._tool_cache_ttl_s = max(0.0, float(tool_cache_ttl_s))
        self._max_output_bytes = max(1024, min(int(max_output_bytes), MAX_BRIDGE_OUTPUT_BYTES))
        self._providers: dict[str, MCPProvider] = {}
        self._provider_locks: dict[str, asyncio.Lock] = {}
        self._tools_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._start_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether this hub has permanently released its providers and loop."""
        return self._closed

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._closed:
            raise MCPHubError("MCP hub is closed")
        if self._loop is not None and self._thread is not None and self._thread.is_alive():
            return self._loop
        with self._start_lock:
            if self._closed:
                raise MCPHubError("MCP hub is closed")
            if self._loop is not None and self._thread is not None and self._thread.is_alive():
                return self._loop
            self._loop_ready.clear()

            def run_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._loop_ready.set()
                loop.run_forever()
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()

            self._thread = threading.Thread(target=run_loop, name="mio-mcp-hub", daemon=True)
            self._thread.start()
            if not self._loop_ready.wait(5.0) or self._loop is None:
                raise MCPHubError("MCP hub event loop failed to start")
        return self._loop

    def _run(self, coroutine, timeout_s: float):
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=max(0.1, timeout_s))
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise MCPHubError(f"MCP bridge timed out after {timeout_s:.1f}s") from exc
        except MCPError:
            raise
        except Exception as exc:
            raise MCPHubError(f"MCP bridge failed: {type(exc).__name__}: {exc}") from exc

    def _timeout_for(self, server: str) -> float:
        return self.registry.get(server).timeout_s + 2.0

    def _bounded(self, server: str, value: Any) -> Any:
        config = self.registry.get(server)
        limit = min(config.max_output_bytes, self._max_output_bytes)
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise MCPHubError("MCP result is not JSON serializable") from exc
        if len(encoded) > limit:
            raise MCPHubError(f"MCP result exceeds bridge output limit ({limit} bytes)")
        return value

    async def _provider(self, server: str) -> MCPProvider:
        config = self.registry.get(server)
        granted = self.policy.grants_for(config)
        provider = self._providers.get(server)
        if provider is not None:
            return provider
        lock = self._provider_locks.get(server)
        if lock is None:
            lock = asyncio.Lock()
            self._provider_locks[server] = lock
        async with lock:
            provider = self._providers.get(server)
            if provider is not None:
                return provider
            created = self._provider_factory(server, config, granted)
            provider = await created if inspect.isawaitable(created) else created
            try:
                await provider.initialize()
            except Exception:
                await provider.close()
                raise
            self._providers[server] = provider
            return provider

    async def _list_tools(self, server: str) -> list[dict[str, Any]]:
        self.policy.grants_for(self.registry.get(server))
        loop = asyncio.get_running_loop()
        cached = self._tools_cache.get(server)
        if cached and cached[0] > loop.time():
            return cached[1]
        provider = await self._provider(server)
        payload = await provider.list_tools()
        tools = payload.get("tools") if isinstance(payload, dict) else None
        if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
            raise MCPHubError("MCP tools/list returned an invalid payload")
        self._bounded(server, tools)
        self._tools_cache[server] = (loop.time() + self._tool_cache_ttl_s, tools)
        return tools

    def list_tools(self, server: str) -> dict[str, Any]:
        tools = self._run(self._list_tools(server), self._timeout_for(server))
        result = {"server": server, "tools": tools}
        self._bounded(server, result)
        return json.loads(json.dumps(result, ensure_ascii=False))

    async def _call_tool(self, server: str, name: str, arguments: Mapping[str, Any]) -> Any:
        tools = await self._list_tools(server)
        known_names = {str(tool.get("name")) for tool in tools}
        if name not in known_names:
            raise MCPHubError(f"MCP tool {name!r} is not advertised by server {server!r}")
        provider = await self._provider(server)
        try:
            return await provider.call_tool(name, arguments)
        except MCPProtocolError:
            # A dead/restarted stdio session must be reinitialized on the next
            # explicit request. Never retry a possibly mutating call.
            await self._drop_provider(server)
            raise

    def call_tool(self, server: str, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise MCPHubError("MCP tool name is required")
        arguments = dict(arguments or {})
        try:
            arg_bytes = len(json.dumps(arguments, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise MCPHubError("MCP tool arguments must be JSON serializable") from exc
        if arg_bytes > MAX_ARGUMENT_BYTES:
            raise MCPHubError(f"MCP tool arguments exceed {MAX_ARGUMENT_BYTES} bytes")
        result = self._run(self._call_tool(server, name, arguments), self._timeout_for(server))
        return self._bounded(server, {"server": server, "tool": name, "result": result})

    async def _drop_provider(self, server: str) -> None:
        lock = self._provider_locks.get(server)
        if lock is None:
            lock = asyncio.Lock()
            self._provider_locks[server] = lock
        async with lock:
            provider = self._providers.pop(server, None)
            self._tools_cache.pop(server, None)
            if provider is not None:
                await provider.close()

    def close_server(self, server: str) -> None:
        self._run(self._drop_provider(server), self._timeout_for(server))

    async def _close_all(self) -> None:
        await asyncio.gather(
            *(self._drop_provider(server) for server in list(self._providers)),
            return_exceptions=True,
        )

    def close(self) -> None:
        # Lifespan shutdown and the atexit fallback may race. Claim ownership
        # once, then do the potentially blocking provider cleanup outside the
        # lock so repeated callers return immediately.
        with self._close_lock:
            if self._closed:
                return
            with self._start_lock:
                self._closed = True
                loop = self._loop
                thread = self._thread
        if loop is not None and thread is not None and thread.is_alive():
            future = asyncio.run_coroutine_threadsafe(self._close_all(), loop)
            try:
                future.result(timeout=5.0)
            except Exception:
                future.cancel()
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
            thread.join(timeout=5.0)
        self._loop = None
        self._thread = None


_default_hub: MCPHub | None = None
_default_hub_lock = threading.Lock()


def get_default_hub() -> MCPHub:
    global _default_hub
    with _default_hub_lock:
        if _default_hub is None or _default_hub.closed:
            _default_hub = MCPHub()
        return _default_hub


def configure_default_hub(registry: MCPRegistry) -> MCPHub:
    """Bind agent/UI MCP calls to the server's selected Mio registry."""
    global _default_hub
    with _default_hub_lock:
        previous = _default_hub
        if previous is not None and not previous.closed and previous.registry is registry:
            return previous
        created = MCPHub(registry)
        _default_hub = created
    if previous is not None:
        previous.close()
    return created


def list_mcp_tools(server: str) -> dict[str, Any]:
    try:
        return get_default_hub().list_tools(server)
    except (MCPError, ValueError) as exc:
        return {"server": server, "error": f"{type(exc).__name__}: {exc}"}


def call_mcp_tool(server: str, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        return get_default_hub().call_tool(server, name, arguments)
    except (MCPError, ValueError) as exc:
        return {"server": server, "tool": name, "error": f"{type(exc).__name__}: {exc}"}


def close_default_hub() -> None:
    """Detach and close Mio's process-wide MCP hub, idempotently.

    Detaching first makes a subsequent application lifespan create a fresh hub
    while the old provider processes are being reaped.
    """
    global _default_hub
    with _default_hub_lock:
        hub = _default_hub
        _default_hub = None
    if hub is not None:
        hub.close()


atexit.register(close_default_hub)
