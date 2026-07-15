# Deployment and local security

Mio is designed first for a single-user Apple Silicon workstation. The safe
default is a loopback-only process, not an internet-facing service.

## Local terminal agent

```bash
python3 -m pip install -e .
mio pull large
mio --tier large
```

The agent can read, write, edit, and run commands. Start it from an intended
workspace and review mutations. External instruction skills are read-only by
default; the catalog does not automatically execute repository scripts.

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
default. Back up durable user data before upgrades. Several older stores do
not yet have a unified migration/locking layer; interrupted-write hardening is
tracked in [13 — Development plan](13-development-plan.md).

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

## UI security boundaries

The current integration line adds:

- exact Host validation and loopback-origin checks resistant to DNS rebinding;
- session/CSRF checks on browser mutations and WebSocket handshakes;
- validated session identifiers and path confinement;
- a 32 MiB global HTTP request-body cap;
- a 25 MiB upload limit with chunked reads;
- sanitization for rendered chat/Markdown surfaces;
- artifact iframes without `allow-same-origin`;
- a Content Security Policy and other security headers (legacy inline UI
  styles/scripts still require CSP compatibility allowances);
- fail-closed grants and per-call confirmation for sensitive Web UI skills;
- public-IP-only, redirect-revalidated external fetching;
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
