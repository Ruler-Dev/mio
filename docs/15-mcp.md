# Mio-owned MCP

MCP providers used by Mio belong to Mio. Their configuration and optional
executables live under `~/.mio`, never under Codex, Claude Code, or another
agent's home.

## Default providers

`mio.mcp.config.builtin_configs()` declares three local stdio providers:

| Name | Command | Default | Declared access |
|---|---|---|---|
| `llm-wiki` | current Python `-m mio.mcp.llm_wiki_server` | enabled | read/write Mio wiki files |
| `headroom` | `~/.mio/bin/headroom mcp serve` | enabled | local process/workspace plus loopback proxy network |
| `ponytail` | Node + `~/.mio/tools/sources/ponytail/ponytail-mcp/index.js` | enabled | read-only instructions |

Local unauthenticated stdio and loopback providers default to enabled. Remote
URLs and providers that consume credential environment variables default to
disabled unless explicitly configured otherwise.

```bash
mio mcp list
```

The server also exposes declarations at `GET /v1/mcp/servers`. That endpoint
never starts or calls a provider.

## What “enabled by default” means

Enabled means an unauthenticated local provider is eligible in Mio's registry,
receives its declared permission set from the default local hub policy, and is
discoverable through two bounded generic tools. It does not
mean:

- a process starts merely because `mio mcp list` ran;
- the model receives every MCP schema in every prompt;
- every calling surface consents to model-driven use;
- remote, authenticated, or undeclared permissions are granted;
- an optional missing executable prevents the inference engine from starting.

Providers are created lazily on the first `list_mcp_tools` or `call_mcp_tool`.
The generic bridge discovers the provider schema before a call, rejects tools
the provider did not advertise, caps argument/result bytes, applies timeouts,
caches discovery briefly, and closes child processes at shutdown. Remote or
credential-bearing providers require explicit policy and per-server grants.

The native agent includes the generic bridges and uses the default local hub
policy. Mio UI adds a stricter layer: `list_mcp_tools` and `call_mcp_tool` are
classified as sensitive orchestration and are omitted from the model tool set
unless their exact names are present both in `MIO_WEBUI_SKILL_GRANTS` and in
that WebSocket request's `skill_grants`. A direct sensitive UI invocation also
requires per-call confirmation. Thus a provider can be enabled locally without
granting automatic browser-model use. Disable any provider whose declared
filesystem/write/process scope is unacceptable; per-provider inner-tool
allow-lists remain release work. `mio mcp doctor` and `mio mcp check --json`
verify the managed runtimes, pins and digests offline.

For example, an operator who intentionally permits the two Web UI bridges can
start Mio with:

```bash
MIO_WEBUI_SKILL_GRANTS=list_mcp_tools,call_mcp_tool \
  mio serve --tier large --webui
```

The client must still include the same bridge name in `skill_grants` for each
model request; the environment variable alone is not request consent.

## Install Headroom and Ponytail inside Mio

Use Mio's packaged installer for both external runtimes:

```bash
mio mcp install-tools
mio mcp doctor
mio mcp check --json
```

`python3 scripts/install_mio_mcp_tools.py` remains a thin source-checkout
wrapper around the same packaged implementation.

The installer pins Headroom 0.31.0, the reviewed Ponytail commit, and the
vendored Ponytail npm lockfile. It builds a fresh release below
`~/.mio/tools/mcp-releases`, validates both MCP entrypoints, writes
`~/.mio/tools/mcp-tools.json`, and only then atomically switches the
`~/.mio/tools/mcp-current` symlink. `--check` is offline and verifies the
active release and manifest. Compatibility shims remain at
`~/.mio/bin/headroom`, `~/.mio/tools/headroom-ai`, and
`~/.mio/tools/sources/ponytail`.

The installer never invokes `headroom mcp install`, never edits Codex or
Claude configuration, and disables npm lifecycle scripts. Mio calls the
validated servers directly from its own registry.

The preset sets:

```text
HEADROOM_WORKSPACE_DIR=~/.mio/headroom
HEADROOM_CONFIG_DIR=~/.mio/headroom/config
HEADROOM_PROXY_URL=http://127.0.0.1:8787
HEADROOM_MCP_READ=off
HEADROOM_TELEMETRY=off
```

The MCP server itself starts lazily through Mio. Headroom's compression proxy
is a separate local service; start it only when routing traffic through the
proxy/API is required:

```bash
HEADROOM_WORKSPACE_DIR=~/.mio/headroom \
HEADROOM_CONFIG_DIR=~/.mio/headroom/config \
HEADROOM_TELEMETRY=off \
~/.mio/bin/headroom proxy --host 127.0.0.1 --port 8787
```

`headroom doctor` reports the proxy unavailable when this process is not
running. Mio does not route its model API through the Headroom proxy by
default; doing so is a separate experiment. Direct `headroom_compress` MCP
calls can run without the proxy.

One direct operational smoke used a synthetic 300-line JSON document. The
tool reported 7,862 input tokens, 2,484 output tokens, 5,378 saved (68.4%), and
the `smart_crusher` transform. In the same smoke, discovery/calls for Headroom,
LLM Wiki and Ponytail returned without MCP errors, while port 8787 remained
unreachable. This is evidence that the direct integration path executes; it is
not a versioned Qwen 3.6 benchmark and has no corpus, repetitions, quality
control, latency comparison, or retrieval-fidelity measurement. Do not
generalize the 68.4% figure to Mio workloads.

The MCP provider exposes a read-only Ponytail prompt/tool. Mio also implements
its own small inference-time `ponytail` prompt policy; the two surfaces are
related but not identical:

- `--prompt-mode ponytail` injects Mio's policy in the native agent, chat, or
  server;
- the Ponytail MCP provider returns upstream instructions when explicitly
  requested by an MCP client.

Nothing is installed into Codex or Claude configuration.

## LLM Wiki

LLM Wiki ships with Mio and stores local cumulative evidence under
`~/.mio/wiki` by default. It exposes:

- `llm_wiki_list`;
- `llm_wiki_search`;
- `llm_wiki_read`;
- `llm_wiki_write`;
- `llm_wiki_ingest`;
- `llm_wiki_lint`.

Paths are confined to the wiki root, writes are local, and the provider speaks
JSON-RPC over stdio. The current implementation is a cumulative sourced JSON
page store inspired by Karpathy's LLM Wiki direction. It does **not** yet
implement the proposed immutable-raw/compiled-Markdown three-layer layout;
that remains a development-plan item.

## Toggle providers

```bash
mio mcp disable headroom
mio mcp enable headroom
mio mcp disable ponytail
```

Discover or call a local provider directly:

```bash
mio mcp tools llm-wiki
mio mcp call llm-wiki llm_wiki_search --args '{"query":"mlx"}'
mio mcp tools ponytail
mio mcp call ponytail ponytail_instructions --args '{"mode":"full"}'
```

For a remote/authenticated entry, direct CLI calls additionally require
`--allow-remote` and/or `--allow-auth` plus one or more
`--grant read|write|process|network|filesystem_read|filesystem_write|secrets`
flags. Enabling the entry alone is insufficient.

Toggles save a complete `~/.mio/mcp.json` with file mode `0600`. A custom file
can replace built-in entries by name:

```bash
MIO_MCP_CONFIG=/path/to/mcp.json mio mcp list
mio serve --mcp-config /path/to/mcp.json
```

## Configuration schema

Example loopback HTTP provider:

```json
{
  "version": 1,
  "servers": [
    {
      "name": "example-local",
      "transport": "http",
      "url": "http://127.0.0.1:8765/mcp",
      "enabled": true,
      "timeout_s": 20,
      "max_output_bytes": 1048576,
      "permissions": ["read"]
    }
  ]
}
```

Supported transports are `stdio`, `http`, and `sse`. Bounds:

- timeout: greater than zero and at most 600 seconds;
- response: 1 KiB through 64 MiB;
- remote authenticated endpoint: HTTPS required;
- stdio: command required and URL forbidden;
- HTTP/SSE: URL required and command forbidden.

Permission names are `read`, `write`, `process`, `network`,
`filesystem_read`, `filesystem_write`, and `secrets`. Stdio automatically
requires `process`; HTTP/SSE automatically requires `network`; credential
mappings automatically require `secrets`.

## Secret handling

Stdio children inherit only `PATH`, `HOME`, `TMPDIR`, `LANG`, and `LC_ALL`,
plus explicitly declared variables. `environment_env` maps a named parent
variable to a child variable; `header_env` maps a parent variable to an HTTP
header. The secret value is not written into the JSON config.

Do not place Hugging Face tokens, API keys, or bearer values directly in
`environment`, repository files, or benchmark artifacts.

## Embedding API

Low-level provider creation is explicit:

```python
import asyncio
from mio.mcp import MCPPermission, load_registry

async def inspect_wiki():
    registry = load_registry()
    provider = registry.create_provider(
        "llm-wiki",
        granted_permissions={
            MCPPermission.PROCESS,
            MCPPermission.READ,
            MCPPermission.WRITE,
            MCPPermission.FILESYSTEM_READ,
            MCPPermission.FILESYSTEM_WRITE,
        },
    )
    try:
        await provider.initialize()
        return await provider.list_tools()
    finally:
        await provider.close()

print(asyncio.run(inspect_wiki()))
```

The low-level client's `call_tool` is never invoked by discovery itself. The
higher-level `MCPHub` is what applies Mio's default local policy and powers the
native-agent/Web UI tools.

## Verification

```bash
mio mcp doctor
mio mcp check --json
mio mcp list
~/.mio/bin/headroom --version
test -f ~/.mio/tools/sources/ponytail/ponytail-mcp/index.js
python3 -m pytest -q \
  tests/test_mcp.py tests/test_mcp_integration.py tests/test_llm_wiki.py
```

The current host passed live initialization and tool discovery for all three
installed local providers. Keep that smoke, plus the existing unavailable-
provider isolation test, in every release gate; an optional provider failure
must never prevent model generation or hide the other providers.
