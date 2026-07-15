# Deployment and local security

Mio is designed first for a single-user Apple Silicon workstation. The safe
default is a loopback-only process, not an internet-facing service.

## Local terminal agent

```bash
python3 -m pip install -e .
mio pull large
mio --tier large --workspace .
# Opt in only when this agent session really needs networked shell/MCP calls.
mio --tier large --workspace . --agent-network
```

The native CLI deliberately grants read, write, and shell capabilities to its
current workspace. Shell calls use a real `zsh`, so pipes, redirects, and
workspace scripts used by integrations such as Hermes or OpenClaw keep their
ordinary semantics. The child process runs inside Mio's inherited macOS
workspace sandbox: user-home and mounted-volume reads are denied outside the
declared roots, writes are confined to those roots, the environment is reduced
to a small non-secret allow-list, runtime/output are capped, and network access
requires a separate trusted-caller grant. The sandbox is default-deny for
Mach/XPC, Apple Events, IOKit, other-process inspection/signals, and network;
zsh startup files and stdin are disabled, and descendants cannot detach from
the supervised process group. A fail-closed tree preflight rejects hard-linked
regular files and unreadable subdirectories before every child launch.
Library callers that do not supply a tool policy are read-only, and unsupported
hosts fail closed instead of running an unrestricted shell.

Start the agent from the intended workspace and review mutations. Additional
workspace roots and network access must be granted explicitly by a trusted Mio
caller; model-generated tool arguments cannot widen the policy. External
instruction skills are read-only when discovered: the catalog never executes
repository scripts merely because their instructions were loaded.
Mio refuses filesystem root, account home, and broad system/volume roots unless
`--unsafe-broad-workspace` is supplied explicitly.

## Local API and UI

```bash
mio serve --tier large
mio serve --tier large --webui
```

Defaults:

- listen address: `127.0.0.1`;
- port: `9090`;
- CORS: loopback origins for the configured port;
- UI: disabled unless `--webui` is supplied;
- prompt policy: `caveman/full`;
- local MCP declarations: enabled in Mio's registry.

OpenAI clients use `http://127.0.0.1:9090/v1` and any non-empty local API-key
placeholder. That key is not authentication; it merely satisfies client SDKs.

## LAN or remote access

```bash
# Administrative opt-in: do not use alone.
mio serve --host 0.0.0.0 --port 9090 --unsafe-remote-bind
```

Mio rejects non-loopback binds without `--unsafe-remote-bind` (or
`MIO_UNSAFE_REMOTE_BIND=1`). The opt-in adds no bearer authentication. A
non-loopback bind must sit behind a trusted firewall and reverse proxy or
tunnel that supplies, at minimum:

1. authentication and TLS;
2. an explicit origin allow-list;
3. request/body and rate limits;
4. connection timeouts and cancellation;
5. access logging that redacts prompts, tokens, and secrets;
6. network policy preventing unintended access to local files/services.

Never expose the Web UI's local file, execution, skill, flow, schedule, or
webhook surfaces to untrusted users.

## Persistent state

User state is primarily under `~/.mio`:

```text
~/.mio/
├── config.json
├── mcp.json
├── skills/
├── tools/
├── wiki/
├── sessions/
└── ... UI databases and JSON stores
```

Target/draft checkpoints are repository-local under `models/` and `spd/` by
default. Back up durable user data before upgrades. Prompts, memory, projects,
schedules, and webhooks use schema-checked, locked read/modify/write operations
with atomic replacement. A malformed existing store fails closed with HTTP
409 and is left untouched instead of being treated as an empty list. Complete
versioned migrations and the remaining legacy-store audit are still tracked in
[13 — Development plan](13-development-plan.md).

## MCP isolation

Headroom is installed in a Mio-owned Python environment, Ponytail source under
Mio's tool root, and LLM Wiki runs from the Mio package. Child process
environments inherit only a small allow-list plus declared variables; cloud
credentials are not copied wholesale.

Provider configuration records permissions, timeout, and maximum response
bytes. For an unauthenticated local provider, enabling the declaration also
authorizes exactly that declared set in Mio's default hub, but each calling
surface can be stricter. In Mio UI, the model may auto-use public read-only
tools only; MCP orchestration and other sensitive skills require an exact
`MIO_WEBUI_SKILL_GRANTS` operator grant plus the same name in the request's
`skill_grants`. Direct sensitive UI runs additionally require per-call
confirmation. Remote/authenticated providers require separate explicit policy
and grants. See [15 — MCP](15-mcp.md).

Native-agent MCP bridges are capped by the same READ/WRITE/SHELL/NETWORK policy
and never grant credential injection. Stdio children inherit the same OS
sandbox and workspace roots. Provider data directories are exact writable Mio
roots, while executable/package roots are separately read-only; cached
providers are replaced when the caller policy changes. HTTP/SSE providers are
denied in the native-agent bridge because an already-running service cannot
inherit this process boundary. Provider declarations still need review: the OS
sandbox limits authority but cannot make a malicious tool semantically safe.

The Settings health panel uses CSRF-protected `POST /v1/mcp/health`; a GET can
never launch provider processes. The endpoint briefly probes only enabled,
local, unauthenticated providers under fixed timeout, concurrency,
server-count, and response-size budgets. Remote or credential-bearing entries
and HTTP/SSE entries are skipped. Every stdio probe receives a separate
least-authority workspace plus only its exact Mio data/runtime roots. It keeps
declared read access and confined process launch while stripping write and
network capabilities. The response omits commands, URLs, environment, tool
names, secrets, and raw exception messages.

## UI security boundaries

The current integration line adds:

- exact Host validation and loopback-origin checks resistant to DNS rebinding;
- session/CSRF checks on browser mutations and WebSocket handshakes;
- validated session identifiers and path confinement;
- a 32 MiB global HTTP request-body cap;
- a 25 MiB upload limit with chunked reads;
- sanitization for rendered chat/Markdown surfaces;
- local vendored Marked/Prism assets, so the main UI does not depend on a CDN
  at boot;
- artifact iframes without `allow-same-origin`;
- a Content Security Policy and other security headers (legacy inline UI
  styles/scripts still require CSP compatibility allowances);
- fail-closed grants and per-call confirmation for sensitive Web UI skills;
- public-IP-only, redirect-revalidated external fetching;
- atomic, schema-checked prompt/memory/project/schedule/webhook mutations that
  preserve malformed stores for recovery;
- a shared `window.Mio` module namespace and loopback server/CORS defaults.

Generated artifacts can intentionally execute JavaScript inside sandboxed
iframes. Treat them as untrusted content. These controls are defense in depth,
not multi-user isolation: bearer authentication, elimination of legacy CSP
inline allowances, executable-artifact consent, and browser QA across primary
views remain release work.

## Operations checklist

Before a local release:

```bash
python3 -m pip check
python3 -m pytest -q
mio mcp doctor
python3 -m mio.model_check
git status --short
```

Also run a real 4B API/UI smoke and the Qwen 3.6 benchmark matrix on the merge
candidate. Check that the benchmark records the intended clean commit rather
than a dirty worktree.

## Tandem and batch caveats

Tandem routing is currently heuristic. CLI/file `mio batch` uses MLX-LM
continuous batching for groups with the same temperature/top-p/top-k/seed
configuration, with independent session caches and shared weights. A singleton
uses the normal DSpark, DFlash, or baseline latency path selected for that
request; stochastic generation uses target-only MLX only when the selected
path requires that fallback. `/v1/batch` uses the same grouping logic within each resolved tier
and restores original item order, but the whole request still holds a
process-wide generation lock. Cross-request queueing/batching, cancellation,
fairness, and controlled concurrent-throughput evaluation remain open; the
existing functional smoke is not a batching-speedup measurement.
