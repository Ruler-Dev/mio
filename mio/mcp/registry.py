"""Mio MCP registry with persistent, user-owned configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from mio.mcp.client import HTTPProvider, MCPProvider, StdioProvider
from mio.mcp.config import (
    MCPConfigError,
    MCPPermission,
    MCPServerConfig,
    MCPTransport,
    builtin_configs,
    default_config_path,
    read_config_file,
)
from mio.persistence import atomic_write_json


class MCPRegistry:
    def __init__(self, configs: Iterable[MCPServerConfig] = (), *, config_path: Path | None = None) -> None:
        self.config_path = config_path or default_config_path()
        self._configs: dict[str, MCPServerConfig] = {}
        for config in configs:
            self.register(config)

    def register(self, config: MCPServerConfig, *, replace: bool = False) -> None:
        if config.name in self._configs and not replace:
            raise MCPConfigError(f"MCP server {config.name!r} is already registered")
        self._configs[config.name] = config

    def get(self, name: str) -> MCPServerConfig:
        try:
            return self._configs[name]
        except KeyError as exc:
            raise MCPConfigError(f"unknown MCP server {name!r}") from exc

    def list(self) -> list[MCPServerConfig]:
        return [self._configs[name] for name in sorted(self._configs)]

    def set_enabled(self, name: str, enabled: bool) -> MCPServerConfig:
        config = self.get(name).with_enabled(enabled)
        self._configs[name] = config
        return config

    def create_provider(
        self,
        name: str,
        *,
        granted_permissions: Iterable[MCPPermission | str] = (),
        process_factory=None,
        http_sender=None,
    ) -> MCPProvider:
        """Create a provider after explicit permission grants.

        Creation does not launch a process or perform network I/O. The caller
        must explicitly invoke ``initialize``/``request``/``call_tool``.
        """
        config = self.get(name)
        granted = frozenset(MCPPermission(value) for value in granted_permissions)
        if config.transport is MCPTransport.STDIO:
            kwargs = {"process_factory": process_factory} if process_factory is not None else {}
            return StdioProvider(config, granted, **kwargs)
        kwargs = {"sender": http_sender} if http_sender is not None else {}
        return HTTPProvider(config, granted, **kwargs)

    def as_dict(self) -> dict:
        return {"version": 1, "servers": [config.as_dict() for config in self.list()]}

    def save(self, path: Path | None = None) -> Path:
        destination = path or self.config_path
        atomic_write_json(destination, self.as_dict())
        return destination


def load_registry(path: Path | str | None = None) -> MCPRegistry:
    config_path = Path(path).expanduser() if path is not None else default_config_path()
    registry = MCPRegistry(builtin_configs(), config_path=config_path)
    for config in read_config_file(config_path):
        registry.register(config, replace=True)
    return registry
