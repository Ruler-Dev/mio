"""Permission-gated MCP clients for stdio and Streamable HTTP/SSE."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from mio.agent_policy import (
    AgentPermissionDenied,
    AgentPolicyError,
    AgentToolPermission,
    AgentToolPolicy,
    sandboxed_command,
)
from mio.mcp.config import MCPPermission, MCPServerConfig, MCPTransport
from mio.paths import mio_home

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


_AGENT_RESERVED_ENV = frozenset(
    {
        "BASH_ENV",
        "CDPATH",
        "ENV",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "HOME",
        "IFS",
        "MIO_HOME",
        "NODE_OPTIONS",
        "PATH",
        "PERL5OPT",
        "RUBYOPT",
        "SHELLOPTS",
        "SSH_AUTH_SOCK",
        "TMPDIR",
        "ZDOTDIR",
    }
)
_AGENT_RESERVED_ENV_PREFIXES = ("DYLD_", "GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_", "LD_", "PYTHON")
_AGENT_SECRET_ENV_MARKERS = (
    "ACCESS_KEY",
    "APIKEY",
    "API_KEY",
    "AUTHORIZATION",
    "CREDENTIAL",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)


def _reject_native_secret_environment(name: str, value: str) -> None:
    normalized_name = name.upper().replace("-", "_")
    if any(marker in normalized_name for marker in _AGENT_SECRET_ENV_MARKERS):
        raise MCPPermissionError(
            f"native-agent MCP environment cannot contain literal credential {name!r}"
        )
    if "://" not in value:
        return
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return
    if parsed.username is not None or parsed.password is not None:
        raise MCPPermissionError(
            f"native-agent MCP environment URL cannot contain userinfo in {name!r}"
        )


def _safe_process_env(
    config: MCPServerConfig,
    *,
    base_environment: Mapping[str, str] | None = None,
    native_agent: bool = False,
) -> dict[str, str]:
    # Do not leak the caller's tokens or cloud credentials into MCP children.
    if base_environment is None:
        allowed = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")
        env = {key: os.environ[key] for key in allowed if key in os.environ}
    else:
        env = dict(base_environment)
    if native_agent and config.environment_env:
        raise MCPPermissionError("native-agent MCP processes cannot receive credential environment variables")
    for name, value in config.environment.items():
        upper_name = name.upper()
        if native_agent:
            _reject_native_secret_environment(name, value)
        if native_agent and (
            upper_name in _AGENT_RESERVED_ENV
            or upper_name.startswith(_AGENT_RESERVED_ENV_PREFIXES)
        ):
            raise MCPPermissionError(
                f"native-agent MCP environment cannot override reserved variable {name!r}"
            )
        env[name] = value
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


_MCP_AGENT_PERMISSION_MAP: Mapping[MCPPermission, AgentToolPermission | None] = {
    MCPPermission.READ: AgentToolPermission.READ,
    MCPPermission.FILESYSTEM_READ: AgentToolPermission.READ,
    MCPPermission.WRITE: AgentToolPermission.WRITE,
    MCPPermission.FILESYSTEM_WRITE: AgentToolPermission.WRITE,
    MCPPermission.PROCESS: AgentToolPermission.SHELL,
    MCPPermission.NETWORK: AgentToolPermission.NETWORK,
    MCPPermission.SECRETS: None,
}
_MIO_DATA_DIRECTORY_ENV = frozenset(
    {
        "HEADROOM_CONFIG_DIR",
        "HEADROOM_WORKSPACE_DIR",
        "MIO_WIKI_ROOT",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    }
)


def _native_stdio_policy(
    config: MCPServerConfig,
    agent_policy: AgentToolPolicy,
    data_roots: tuple[Path, ...],
) -> AgentToolPolicy:
    """Narrow an agent policy to one stdio MCP declaration and Mio state."""

    permissions: set[AgentToolPermission] = set()
    for permission in config.permissions:
        mapped = _MCP_AGENT_PERMISSION_MAP[permission]
        if mapped is None:
            raise MCPPermissionError("native-agent MCP processes cannot receive secrets")
        permissions.add(mapped)
    missing = permissions - agent_policy.permissions
    if missing:
        names = ", ".join(sorted(permission.value for permission in missing))
        raise MCPPermissionError(f"native-agent MCP process exceeds agent policy: {names}")

    roots = list(agent_policy.workspace_roots)
    for declared_root in data_roots:
        if not any(declared_root == root or declared_root.is_relative_to(root) for root in roots):
            roots.append(declared_root)
    return AgentToolPolicy(
        workspace_roots=tuple(roots),
        permissions=frozenset(permissions),
        output_limit_chars=agent_policy.output_limit_chars,
        file_limit_chars=agent_policy.file_limit_chars,
        command_timeout_s=min(agent_policy.command_timeout_s, config.timeout_s),
        audit_sink=agent_policy.audit_sink,
    )


def _mio_candidate_path(raw_value: str, canonical_runtime: Path) -> Path | None:
    candidate_text = raw_value
    if "=" in candidate_text:
        _, possible_path = candidate_text.split("=", 1)
        if Path(possible_path).expanduser().is_absolute():
            candidate_text = possible_path
    candidate = Path(candidate_text).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(canonical_runtime)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved_candidate


def _ensure_declared_mio_data_directories(
    config: MCPServerConfig,
    runtime_root: Path,
) -> Path:
    """Create only exact, trusted Mio-owned data directories from config."""

    declared = [
        value
        for name, value in config.environment.items()
        if name.upper() in _MIO_DATA_DIRECTORY_ENV and Path(value).expanduser().is_absolute()
    ]
    if not declared:
        return runtime_root.expanduser().resolve(strict=False)
    try:
        expanded_runtime = runtime_root.expanduser()
        expanded_runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        canonical_runtime = expanded_runtime.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MCPPermissionError(f"cannot prepare Mio runtime root: {runtime_root}") from exc
    for raw_directory in declared:
        candidate = _mio_candidate_path(raw_directory, canonical_runtime)
        if candidate is None or candidate == canonical_runtime:
            raise MCPPermissionError(
                f"MCP data directory must be a strict child of MIO_HOME: {raw_directory}"
            )
        existed = candidate.exists()
        try:
            candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
            canonical_candidate = candidate.resolve(strict=True)
            canonical_candidate.relative_to(canonical_runtime)
            if not canonical_candidate.is_dir():
                raise MCPPermissionError(f"MCP data path is not a directory: {raw_directory}")
            if not existed:
                canonical_candidate.chmod(0o700)
        except MCPPermissionError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise MCPPermissionError(
                f"cannot prepare declared MCP data directory: {raw_directory}"
            ) from exc
    return canonical_runtime


def _declared_mio_runtime_access(
    config: MCPServerConfig,
    runtime_root: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...], Path]:
    """Separate writable provider data from read-only executables/packages.

    Adding all of ``~/.mio`` would expose unrelated sessions, skills and model
    state and make the inherited-sandbox hard-link audit scan a very large
    tree. Only exact data directories declared by known runtime variables are
    writable; executable/package roots and other declared paths are read-only.
    """

    canonical_runtime = _ensure_declared_mio_data_directories(config, runtime_root)
    data_roots: list[Path] = []
    read_only_roots: list[Path] = []

    def add_root(collection: list[Path], directory: Path) -> None:
        if any(directory == existing or directory.is_relative_to(existing) for existing in collection):
            return
        collection[:] = [
            existing for existing in collection if not existing.is_relative_to(directory)
        ]
        collection.append(directory)

    for name, raw_candidate in config.environment.items():
        resolved_candidate = _mio_candidate_path(raw_candidate, canonical_runtime)
        if resolved_candidate is None:
            continue
        directory = resolved_candidate if resolved_candidate.is_dir() else resolved_candidate.parent
        while directory != canonical_runtime and not directory.is_dir():
            directory = directory.parent
        if directory == canonical_runtime or not directory.is_dir():
            continue
        try:
            canonical_directory = directory.resolve(strict=True)
            canonical_directory.relative_to(canonical_runtime)
        except (OSError, RuntimeError, ValueError):
            continue
        if name.upper() in _MIO_DATA_DIRECTORY_ENV:
            add_root(data_roots, canonical_directory)
        else:
            add_root(read_only_roots, canonical_directory)

    for raw_candidate in config.command:
        resolved_candidate = _mio_candidate_path(raw_candidate, canonical_runtime)
        if resolved_candidate is None:
            continue
        directory = resolved_candidate if resolved_candidate.is_dir() else resolved_candidate.parent
        # Console scripts in a Mio-owned virtualenv need their sibling lib/
        # packages, not writable access to the environment.
        if directory.name == "bin" and directory.parent != canonical_runtime:
            directory = directory.parent
        # Ponytail's MCP entrypoint imports its repository-level hooks/ sibling.
        if directory.name == "ponytail-mcp" and (directory.parent / "hooks").is_dir():
            directory = directory.parent
        if directory != canonical_runtime and directory.is_dir():
            add_root(read_only_roots, directory.resolve(strict=True))

    try:
        module_index = config.command.index("-m")
        module_name = config.command[module_index + 1]
    except (ValueError, IndexError):
        module_name = ""
    if module_name == "mio" or module_name.startswith("mio."):
        # `python -m mio...` must import Mio even when the user's selected
        # coding workspace is a different repository.
        add_root(read_only_roots, Path(__file__).resolve().parents[2])

    return tuple(data_roots), tuple(read_only_roots), canonical_runtime


def _canonicalize_mio_command(command: list[str], canonical_runtime: Path) -> list[str]:
    """Execute Mio-owned symlinked entrypoints by their confined real paths."""

    canonical: list[str] = []
    for argument in command:
        prefix = ""
        path_text = argument
        if "=" in argument:
            option, possible_path = argument.split("=", 1)
            if Path(possible_path).expanduser().is_absolute():
                prefix = option + "="
                path_text = possible_path
        path = Path(path_text).expanduser()
        if not path.is_absolute() or not path.exists():
            canonical.append(argument)
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(canonical_runtime)
        except (OSError, RuntimeError, ValueError):
            canonical.append(argument)
            continue
        canonical.append(prefix + str(resolved))
    return canonical


class StdioProvider(MCPProvider):
    def __init__(
        self,
        config: MCPServerConfig,
        granted: frozenset[MCPPermission],
        *,
        process_factory: ProcessFactory = asyncio.create_subprocess_exec,
        agent_policy: AgentToolPolicy | None = None,
        mio_runtime_root: Path | None = None,
    ) -> None:
        super().__init__(config, granted)
        self._process_factory = process_factory
        self._agent_policy = agent_policy
        self._mio_runtime_root = Path(mio_runtime_root) if mio_runtime_root is not None else mio_home()
        self._process: Any = None
        self._lock = asyncio.Lock()

    @property
    def _effective_timeout_s(self) -> float:
        if self._agent_policy is None:
            return self.config.timeout_s
        return min(self.config.timeout_s, self._agent_policy.command_timeout_s)

    def _launch_spec(self) -> tuple[list[str], dict[str, str]]:
        command = list(self.config.command)
        if self._agent_policy is None:
            return command, _safe_process_env(self.config)
        data_roots, read_only_roots, runtime_root = _declared_mio_runtime_access(
            self.config,
            self._mio_runtime_root,
        )
        command = _canonicalize_mio_command(command, runtime_root)
        child_policy = _native_stdio_policy(
            self.config,
            self._agent_policy,
            data_roots,
        )
        try:
            command, environment = sandboxed_command(
                command,
                child_policy,
                read_only_roots=read_only_roots,
            )
        except AgentPermissionDenied as exc:
            raise MCPPermissionError(f"native-agent MCP sandbox is unavailable: {exc}") from exc
        except AgentPolicyError as exc:
            raise MCPPermissionError(f"native-agent MCP sandbox rejected runtime roots: {exc}") from exc
        # Never expose the caller's real home. HOME/TMPDIR stay inside the
        # already-authorized workspace; MIO_HOME only names the application
        # base while declared subdirectories remain the actual SBPL grants.
        environment.update(
            {
                "HOME": str(child_policy.primary_workspace),
                "MIO_HOME": str(runtime_root),
                "TMPDIR": str(child_policy.primary_workspace),
            }
        )
        return command, _safe_process_env(
            self.config,
            base_environment=environment,
            native_agent=True,
        )

    async def _ensure_process(self) -> Any:
        if self._process is not None and self._process.returncode is not None:
            # A dead leader may still have live descendants in its dedicated
            # group. Reap/kill that group before replacing the session.
            await self._stop_process()
        if self._process is None:
            try:
                command, environment = self._launch_spec()
                cwd = (
                    str(self._agent_policy.primary_workspace)
                    if self._agent_policy is not None
                    else None
                )
                self._process = await self._process_factory(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=environment,
                    limit=self.config.max_output_bytes + 1,
                    start_new_session=True,
                    cwd=cwd,
                )
            except (FileNotFoundError, OSError) as exc:
                raise MCPError(f"cannot start MCP server {self.config.name!r}: {exc}") from exc
        if self._process.stdin is None or self._process.stdout is None:
            raise MCPError("MCP child process has no stdio pipes")
        return self._process

    @staticmethod
    def _signal_process(process: Any, sig: signal.Signals) -> None:
        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, sig)
                return
            except ProcessLookupError:
                return
            except OSError:
                # Test doubles and unusual subprocess implementations may not
                # expose a real session even though production children do.
                pass
        action = process.terminate if sig is signal.SIGTERM else process.kill
        try:
            action()
        except ProcessLookupError:
            pass

    async def _terminate_and_reap(self, process: Any) -> None:
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            self._signal_process(process, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), min(self._effective_timeout_s, 3.0))
            except asyncio.TimeoutError:
                self._signal_process(process, signal.SIGKILL)
                await process.wait()
            else:
                # The server may have left descendants in its dedicated
                # process group. Ensure none outlive the provider leader.
                pid = getattr(process, "pid", None)
                if isinstance(pid, int) and pid > 0:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        else:
            # The leader is already finished, but descendants may still be
            # alive in the dedicated session/process group.
            pid = getattr(process, "pid", None)
            if isinstance(pid, int) and pid > 0:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await process.wait()

    async def _stop_process(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        cleanup = asyncio.create_task(self._terminate_and_reap(process))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # Do not let request/health cancellation strand a process or
            # zombie. Preserve cancellation only after TERM -> KILL -> reap.
            try:
                await cleanup
            finally:
                raise

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        request_id, payload = self._request_payload(method, params)
        async with self._lock:
            try:
                process = await self._ensure_process()
                process.stdin.write(payload + b"\n")
                await asyncio.wait_for(process.stdin.drain(), self._effective_timeout_s)

                consumed = 0
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self._effective_timeout_s
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
                await asyncio.wait_for(process.stdin.drain(), self._effective_timeout_s)
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
