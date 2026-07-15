"""Permission-gated MCP clients for stdio and Streamable HTTP/SSE."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from mio.mcp.config import MCPPermission, MCPServerConfig, MCPTransport

MCP_PROTOCOL_VERSION = "2025-11-25"


class MCPError(RuntimeError):
    pass


class MCPDisabledError(MCPError):
    pass


class MCPPermissionError(MCPError):
    pass


class MCPProtocolError(MCPError):
    pass


class MCPRemoteError(MCPError):
    def __init__(self, error: Any) -> None:
        super().__init__(f"MCP server error: {error}")
        self.error = error


def _check_access(config: MCPServerConfig, granted: frozenset[MCPPermission]) -> None:
    if not config.enabled:
        raise MCPDisabledError(f"MCP server {config.name!r} is disabled")
    missing = config.permissions - granted
    if missing:
        names = ", ".join(sorted(permission.value for permission in missing))
        raise MCPPermissionError(f"MCP server {config.name!r} requires explicit grants: {names}")


def _safe_process_env(config: MCPServerConfig) -> dict[str, str]:
    # Do not leak the caller's tokens or cloud credentials into MCP children.
    allowed = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(config.environment)
    for child_name, source_name in config.environment_env.items():
        value = os.environ.get(source_name)
        if not value:
            raise MCPPermissionError(f"required MCP credential environment variable {source_name!r} is missing")
        env[child_name] = value
    return env


def _decode_jsonrpc(payload: bytes, expected_id: int) -> Any:
    try:
        message = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MCPProtocolError(f"invalid JSON-RPC response: {exc}") from exc
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        raise MCPProtocolError("response is not a JSON-RPC 2.0 object")
    if message.get("id") != expected_id:
        raise MCPProtocolError(f"unexpected JSON-RPC id {message.get('id')!r}")
    if "error" in message:
        raise MCPRemoteError(message["error"])
    if "result" not in message:
        raise MCPProtocolError("JSON-RPC response has neither result nor error")
    return message["result"]


class MCPProvider:
    def __init__(self, config: MCPServerConfig, granted: frozenset[MCPPermission]) -> None:
        _check_access(config, granted)
        self.config = config
        self.granted = granted
        self._next_id = 0

    def _request_payload(self, method: str, params: Mapping[str, Any] | None) -> tuple[int, bytes]:
        self._next_id += 1
        request_id = self._next_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = dict(params)
        return request_id, json.dumps(message, separators=(",", ":")).encode("utf-8")

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        raise NotImplementedError

    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        raise NotImplementedError

    async def initialize(self) -> Any:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "mio", "version": "0.1.0"},
            },
        )
        await self.notify("notifications/initialized")
        return result

    async def list_tools(self) -> Any:
        return await self.request("tools/list")

    async def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        # Dispatch is always mediated by the hub's policy, schema discovery,
        # argument limit, timeout, and output limit.
        return await self.request("tools/call", {"name": name, "arguments": dict(arguments or {})})

    async def close(self) -> None:
        return None


ProcessFactory = Callable[..., Awaitable[Any]]


class StdioProvider(MCPProvider):
    def __init__(
        self,
        config: MCPServerConfig,
        granted: frozenset[MCPPermission],
        *,
        process_factory: ProcessFactory = asyncio.create_subprocess_exec,
    ) -> None:
        super().__init__(config, granted)
        self._process_factory = process_factory
        self._process: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_process(self) -> Any:
        if self._process is None or self._process.returncode is not None:
            try:
                self._process = await self._process_factory(
                    *self.config.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=_safe_process_env(self.config),
                    limit=self.config.max_output_bytes + 1,
                )
            except (FileNotFoundError, OSError) as exc:
                raise MCPError(f"cannot start MCP server {self.config.name!r}: {exc}") from exc
        if self._process.stdin is None or self._process.stdout is None:
            raise MCPError("MCP child process has no stdio pipes")
        return self._process

    async def _stop_process(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), min(self.config.timeout_s, 3.0))
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        request_id, payload = self._request_payload(method, params)
        async with self._lock:
            try:
                process = await self._ensure_process()
                process.stdin.write(payload + b"\n")
                await asyncio.wait_for(process.stdin.drain(), self.config.timeout_s)

                consumed = 0
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self.config.timeout_s
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError(f"MCP server {self.config.name!r} timed out")
                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), remaining)
                    except (ValueError, asyncio.LimitOverrunError) as exc:
                        raise MCPProtocolError("MCP stdio response exceeded output limit") from exc
                    if not line:
                        raise MCPProtocolError("MCP stdio server closed before responding")
                    consumed += len(line)
                    if consumed > self.config.max_output_bytes:
                        raise MCPProtocolError("MCP stdio response exceeded output limit")
                    try:
                        message = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise MCPProtocolError(f"invalid JSON-RPC response: {exc}") from exc
                    # Notifications may legally arrive while a request is pending.
                    if isinstance(message, dict) and "id" not in message:
                        continue
                    return _decode_jsonrpc(line, request_id)
            except (TimeoutError, MCPProtocolError, asyncio.CancelledError):
                await self._stop_process()
                raise
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                await self._stop_process()
                raise MCPError(f"MCP stdio transport failed: {exc}") from exc

    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = dict(params)
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        async with self._lock:
            try:
                process = await self._ensure_process()
                process.stdin.write(payload + b"\n")
                await asyncio.wait_for(process.stdin.drain(), self.config.timeout_s)
            except (TimeoutError, asyncio.CancelledError, BrokenPipeError, ConnectionResetError, OSError):
                await self._stop_process()
                raise

    async def close(self) -> None:
        async with self._lock:
            await self._stop_process()


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


HTTPSender = Callable[[str, bytes, Mapping[str, str], float, int], Awaitable[HTTPResponse]]


async def _default_http_sender(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout_s: float,
    max_output_bytes: int,
) -> HTTPResponse:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        # Redirects can forward Authorization or custom secret headers to a
        # different host or downgrade HTTPS. MCP endpoints must be explicit.
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    def send() -> HTTPResponse:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        opener = urllib.request.build_opener(NoRedirect())
        try:
            with opener.open(request, timeout=timeout_s) as response:
                payload = response.read(max_output_bytes + 1)
                return HTTPResponse(response.status, dict(response.headers.items()), payload)
        except urllib.error.HTTPError as exc:
            return HTTPResponse(exc.code, dict(exc.headers.items()), exc.read(max_output_bytes + 1))

    return await asyncio.wait_for(asyncio.to_thread(send), timeout_s + 1.0)


def _sse_data(body: bytes, expected_id: int, max_output_bytes: int) -> bytes:
    if len(body) > max_output_bytes:
        raise MCPProtocolError("MCP HTTP response exceeded output limit")
    try:
        text = body.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise MCPProtocolError("MCP SSE response is not valid UTF-8") from exc
    for event in text.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
        if not data:
            continue
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and message.get("id") == expected_id:
            return data.encode("utf-8")
    raise MCPProtocolError("SSE stream did not contain the matching JSON-RPC response")


class HTTPProvider(MCPProvider):
    def __init__(
        self,
        config: MCPServerConfig,
        granted: frozenset[MCPPermission],
        *,
        sender: HTTPSender = _default_http_sender,
    ) -> None:
        super().__init__(config, granted)
        self._sender = sender
        self._session_id: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        for header, env_name in self.config.header_env.items():
            value = os.environ.get(env_name)
            if not value:
                raise MCPPermissionError(f"required MCP credential environment variable {env_name!r} is missing")
            headers[header] = value
        return headers

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        request_id, payload = self._request_payload(method, params)
        response = await self._sender(
            self.config.url or "",
            payload,
            self._headers(),
            self.config.timeout_s,
            self.config.max_output_bytes,
        )
        if len(response.body) > self.config.max_output_bytes:
            raise MCPProtocolError("MCP HTTP response exceeded output limit")
        if not 200 <= response.status < 300:
            raise MCPProtocolError(f"MCP HTTP server returned status {response.status}")
        headers = {key.lower(): value for key, value in response.headers.items()}
        self._session_id = headers.get("mcp-session-id", self._session_id)
        content_type = headers.get("content-type", "")
        body = (
            _sse_data(response.body, request_id, self.config.max_output_bytes)
            if "text/event-stream" in content_type or self.config.transport is MCPTransport.SSE
            else response.body
        )
        return _decode_jsonrpc(body, request_id)

    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = dict(params)
        response = await self._sender(
            self.config.url or "",
            json.dumps(message, separators=(",", ":")).encode("utf-8"),
            self._headers(),
            self.config.timeout_s,
            self.config.max_output_bytes,
        )
        if len(response.body) > self.config.max_output_bytes:
            raise MCPProtocolError("MCP HTTP response exceeded output limit")
        if not 200 <= response.status < 300:
            raise MCPProtocolError(f"MCP HTTP server returned status {response.status}")
