"""Configuration primitives for Mio-owned MCP servers.

Mio keeps user overrides in ``~/.mio/mcp.json`` (or ``MIO_MCP_CONFIG``).
Local stdio providers are enabled by default; remote and authenticated HTTP
providers require an explicit ``"enabled": true`` in that file.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from mio.paths import mio_home


class MCPTransport(str, Enum):
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


class MCPPermission(str, Enum):
    """Capabilities a caller must grant before a provider can do work."""

    READ = "read"
    WRITE = "write"
    PROCESS = "process"
    NETWORK = "network"
    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    SECRETS = "secrets"


class MCPConfigError(ValueError):
    """Raised for invalid or unsafe MCP configuration."""


def default_config_path() -> Path:
    override = os.environ.get("MIO_MCP_CONFIG")
    return Path(override).expanduser() if override else mio_home() / "mcp.json"


def _is_loopback_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "::1",
        "localhost",
    }


@dataclass(frozen=True)
class MCPServerConfig:
    """A validated MCP server declaration.

    ``enabled=None`` applies the safe default: local unauthenticated servers
    are enabled, while remote or authenticated servers are opt-in.
    """

    name: str
    transport: MCPTransport
    command: tuple[str, ...] = ()
    url: str | None = None
    enabled: bool | None = None
    timeout_s: float = 30.0
    max_output_bytes: int = 4 * 1024 * 1024
    permissions: frozenset[MCPPermission] = field(default_factory=frozenset)
    environment: Mapping[str, str] = field(default_factory=dict)
    environment_env: Mapping[str, str] = field(default_factory=dict)
    header_env: Mapping[str, str] = field(default_factory=dict)
    description: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name or any(ch.isspace() for ch in self.name):
            raise MCPConfigError("MCP server name must be non-empty and contain no whitespace")
        if self.timeout_s <= 0 or self.timeout_s > 600:
            raise MCPConfigError("MCP timeout must be in (0, 600] seconds")
        if self.max_output_bytes < 1024 or self.max_output_bytes > 64 * 1024 * 1024:
            raise MCPConfigError("MCP max_output_bytes must be between 1 KiB and 64 MiB")

        permissions = frozenset(MCPPermission(value) for value in self.permissions)
        if self.transport is MCPTransport.STDIO:
            if not self.command or self.url:
                raise MCPConfigError("stdio MCP servers require command and forbid url")
            permissions |= {MCPPermission.PROCESS}
            local = True
        else:
            if self.command or not self.url:
                raise MCPConfigError("HTTP/SSE MCP servers require url and forbid command")
            parsed = urlparse(self.url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise MCPConfigError("MCP URL must use http or https")
            permissions |= {MCPPermission.NETWORK}
            local = _is_loopback_url(self.url)

        if self.header_env:
            if self.url and not self.is_local and not self.url.startswith("https://"):
                raise MCPConfigError("authenticated remote MCP servers must use https")
            permissions |= {MCPPermission.SECRETS}
        if self.environment_env:
            permissions |= {MCPPermission.SECRETS}
        enabled = self.enabled
        if enabled is None:
            enabled = local and not self.header_env and not self.environment_env

        object.__setattr__(self, "permissions", frozenset(permissions))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "environment", dict(self.environment))
        object.__setattr__(self, "environment_env", dict(self.environment_env))
        object.__setattr__(self, "header_env", dict(self.header_env))

    @property
    def is_local(self) -> bool:
        return self.transport is MCPTransport.STDIO or bool(self.url and _is_loopback_url(self.url))

    @property
    def uses_auth(self) -> bool:
        return bool(self.header_env or self.environment_env)

    def with_enabled(self, enabled: bool) -> "MCPServerConfig":
        return replace(self, enabled=enabled)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "transport": self.transport.value,
            "enabled": self.enabled,
            "timeout_s": self.timeout_s,
            "max_output_bytes": self.max_output_bytes,
            "permissions": sorted(permission.value for permission in self.permissions),
        }
        if self.command:
            data["command"] = list(self.command)
        if self.url:
            data["url"] = self.url
        if self.environment:
            data["environment"] = dict(self.environment)
        if self.environment_env:
            data["environment_env"] = dict(self.environment_env)
        if self.header_env:
            data["header_env"] = dict(self.header_env)
        if self.description:
            data["description"] = self.description
        if self.source:
            data["source"] = self.source
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MCPServerConfig":
        if not isinstance(data, Mapping):
            raise MCPConfigError("MCP server config must be an object")
        try:
            return cls(
                name=str(data["name"]),
                transport=MCPTransport(data["transport"]),
                command=tuple(str(part) for part in data.get("command", ())),
                url=str(data["url"]) if data.get("url") else None,
                enabled=data.get("enabled"),
                timeout_s=float(data.get("timeout_s", 30.0)),
                max_output_bytes=int(data.get("max_output_bytes", 4 * 1024 * 1024)),
                permissions=frozenset(MCPPermission(value) for value in data.get("permissions", ())),
                environment={str(k): str(v) for k, v in data.get("environment", {}).items()},
                environment_env={str(k): str(v) for k, v in data.get("environment_env", {}).items()},
                header_env={str(k): str(v) for k, v in data.get("header_env", {}).items()},
                description=str(data.get("description", "")),
                source=str(data.get("source", "")),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise MCPConfigError(f"invalid MCP server config: {exc}") from exc


def builtin_configs() -> tuple[MCPServerConfig, ...]:
    """Return Mio's local, enabled-by-default MCP presets."""
    home = mio_home()
    ponytail_server = home / "tools" / "sources" / "ponytail" / "ponytail-mcp" / "index.js"
    return (
        MCPServerConfig(
            name="headroom",
            transport=MCPTransport.STDIO,
            command=(str(home / "bin" / "headroom"), "mcp", "serve"),
            enabled=True,
            timeout_s=60.0,
            permissions=frozenset(
                {
                    MCPPermission.READ,
                    MCPPermission.WRITE,
                    MCPPermission.NETWORK,
                    MCPPermission.FILESYSTEM_READ,
                    MCPPermission.FILESYSTEM_WRITE,
                }
            ),
            environment={
                "HEADROOM_WORKSPACE_DIR": str(home / "headroom"),
                "HEADROOM_CONFIG_DIR": str(home / "headroom" / "config"),
                "HEADROOM_PROXY_URL": "http://127.0.0.1:8787",
                "HEADROOM_MCP_READ": "off",
                "HEADROOM_TELEMETRY": "off",
            },
            description="Local reversible context compression; compression uses the loopback Headroom proxy.",
            source="https://github.com/headroomlabs-ai/headroom",
        ),
        MCPServerConfig(
            name="llm-wiki",
            transport=MCPTransport.STDIO,
            command=(sys.executable, "-m", "mio.mcp.llm_wiki_server"),
            enabled=True,
            permissions=frozenset(
                {
                    MCPPermission.READ,
                    MCPPermission.WRITE,
                    MCPPermission.FILESYSTEM_READ,
                    MCPPermission.FILESYSTEM_WRITE,
                }
            ),
            environment={"MIO_WIKI_ROOT": str(home / "wiki")},
            description="Local cumulative evidence wiki for Mio agents.",
            source="https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f",
        ),
        MCPServerConfig(
            name="ponytail",
            transport=MCPTransport.STDIO,
            command=(shutil.which("node") or "node", str(ponytail_server)),
            enabled=True,
            permissions=frozenset({MCPPermission.READ, MCPPermission.FILESYSTEM_READ}),
            environment={
                "XDG_CONFIG_HOME": str(home / "config"),
                "PONYTAIL_DEFAULT_MODE": "full",
            },
            description="Read-only Ponytail engineering instructions for Mio.",
            source="https://github.com/DietrichGebert/ponytail",
        ),
    )


def read_config_file(path: Path) -> list[MCPServerConfig]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPConfigError(f"cannot read {path}: {exc}") from exc
    items = raw.get("servers") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise MCPConfigError("MCP config must be a list or an object with a servers list")
    return [MCPServerConfig.from_dict(item) for item in items]
