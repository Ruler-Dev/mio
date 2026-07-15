# CLI and API reference

Run `mio --help` and `mio <command> --help` for the parser-authoritative
reference. This page explains the behavior and safety implications.

## `mio` — native coding agent

With no subcommand Mio starts the terminal agent.

```bash
mio
mio --tier large
mio --context 32k
mio --tq4
mio --mpath 2
mio --tandem
mio --workspace .
mio --agent-root ../shared-read-write
mio --agent-network
mio "inspect the current repository"
```

| Flag | Meaning |
|---|---|
| `--tier NAME` | select one configured tier |
| `--tandem` | load configured tiers and route requests |
| `--paro` | use the PARO tier table |
| `--context SIZE` | override context (`32k`, `131072`, and similar forms) |
| `--tq4` | select TQ4 and disable mutually exclusive PQ for this process |
| `--mpath K` | set BMP path count |
| `--workspace PATH` | set the primary agent root (default: nearest Git root, otherwise current directory) |
| `--agent-root PATH` | add an explicit workspace root; repeatable |
| `--agent-network` | grant shell/MCP network authority for this agent session |
| `--unsafe-broad-workspace` | explicitly permit `/`, home, or another broad root that Mio otherwise refuses |
| `prompt` | optional initial task |

The native agent exposes filesystem/shell tools and Mio's
`list_mio_skills`/`read_mio_skill` catalog tools. Its CLI trust boundary grants
READ/WRITE/SHELL only to the displayed roots. A real no-startup-file zsh runs
inside Mio's default-deny macOS process sandbox; network remains a separate
grant. Mio refuses account home, filesystem root, and broad system/volume roots unless the
unsafe acknowledgement is explicit. Before launch, Mio rejects hard-linked
regular files anywhere below the sandbox roots so an allowed pathname cannot
alias an outside inode. Library callers without a policy are read-only.

Current slash commands include `/model`, `/tier`, `/context`, `/caveman`,
`/ponytail`, `/tq`, `/status`, `/models`, `/configure`, `/clear`, `/help`, and
`/quit`. Top-level agent flags also accept `--prompt-mode`, `--prompt-level`,
`--caveman`, and `--ponytail`.

## `mio chat`

Starts a terminal chat without native agent tools.

```bash
mio chat --tier large
mio chat --prompt-mode none
mio chat --prompt-mode caveman --prompt-level full
mio chat --prompt-mode ponytail --prompt-level full
mio chat --no-caveman              # legacy alias for none
```

Prompt selectors are mutually exclusive. Legacy forms remain:

```bash
mio chat --caveman lite
mio chat --ponytail ultra
```

## `mio serve`

Starts the OpenAI-compatible server.

```bash
mio serve --tier large
mio serve --tier large --webui
mio serve --tiers large,medium
mio serve --tandem
mio serve --port 9100
mio serve --host 127.0.0.1
mio serve --host 0.0.0.0 --unsafe-remote-bind  # administrative opt-in only
mio serve --validate
mio serve --context 64k
mio serve --compact-threshold 0.75 --compact-target 0.50
mio serve --no-compact-summarize
mio serve --mcp-config ~/.mio/mcp.json
```

The effective host/port come from explicit flags, then persisted config, then
`127.0.0.1:9090`. `--webui` mounts Mio UI under `/ui`.

Prompt policy flags are the same as `mio chat`:

```bash
mio serve --prompt-mode none
mio serve --prompt-mode caveman --prompt-level lite
mio serve --prompt-mode ponytail --prompt-level full
```

Mio refuses a non-loopback address unless `--unsafe-remote-bind` (or
`MIO_UNSAFE_REMOTE_BIND=1`) is supplied. The opt-in does not add
authentication; use it only behind a trusted firewall or authenticated reverse
proxy with explicit origin policy.

### Core endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | process/model health |
| `GET` | `/v1/models` | loaded model list |
| `POST` | `/v1/models/load` | load a configured tier |
| `POST` | `/v1/models/unload` | unload a tier |
| `POST` | `/v1/chat/completions` | streaming or complete chat generation |
| `POST` | `/v1/batch` | bounded HTTP batch routed through tier/temperature groups |
| `GET` | `/v1/mcp/servers` | MCP declarations only; does not launch/call providers |
| `GET` | `/metrics` | runtime metrics |
| `GET` | `/dashboard` | lightweight metrics page |
| `WS` | `/ws/metrics` | live metric events |

Model identifiers are `mio-large-moe`, `mio-large`, `mio-medium`,
`mio-small`, and `mio-auto` when tandem routing is active.

## `mio pull`

Downloads the complete speculative stack into repository-local stable
directories. For Qwen 3.6 27B the default is target + preferred DSpark + a
separate compatible DFlash fallback.

```bash
mio pull                         # list tiers and raw keys
mio pull large                   # Qwen 3.6 27B target + DSpark + DFlash fallback
mio pull large --no-dspark       # target + DFlash only
mio pull large --no-fallback     # target + DSpark only; does not persist strict mode
mio pull qwen3.6-27b-unsloth
mio pull medium
```

`mio pull` validates each checkpoint and every declared shard independently.
Re-running a complete stack is a no-op; interrupted directories are resumed.

## `mio download`

Downloads target/draft references from the persisted tier configuration into
the Hugging Face cache. For predictable repository-local Qwen 3.6 directories,
prefer `mio pull`.

```bash
mio download
mio download --tier large
```

## `mio configure`

Runs the interactive configuration wizard and writes `~/.mio/config.json`.

```bash
mio configure
```

Persisted tiers overlay current registry defaults. A configured TQ bit width
wins over a legacy file that accidentally enabled both TQ and PQ.

## `mio mcp`

Lists, toggles, discovers, or explicitly calls Mio-owned providers.

```bash
mio mcp list
mio mcp disable headroom
mio mcp enable headroom
mio mcp tools llm-wiki
mio mcp call llm-wiki llm_wiki_search --args '{"query":"mlx"}'
mio mcp install-tools              # pinned Headroom + Ponytail inside $MIO_HOME
mio mcp check --json               # offline pins/digests/runtime verification
mio mcp doctor                     # human-readable alias for check
mio mcp list --config /path/to/mcp.json
```

Enabled unauthenticated local presets receive their declared permissions and
are eligible to the native agent/Web UI. Processes initialize lazily. Mio UI
still treats MCP orchestration as sensitive: model auto-use requires exact
operator and request grants, while a direct sensitive call additionally
requires confirmation. Remote or authenticated entries require
`--allow-remote`/`--allow-auth` and repeated `--grant PERMISSION` flags for
direct CLI calls. See [15 — MCP](15-mcp.md).

## `mio batch`

```bash
mio batch --input prompts.jsonl --output results.jsonl --tier large
```

Each input line is a JSON object containing `prompt` or `messages`, with
optional `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, and `stop`.
Requests are grouped by the complete sampler configuration:

- groups of two or more use `MioEngine.generate_batch` and MLX-LM continuous
  batching with shared weights and independent KV caches;
- a one-request group uses the ordinary latency path selected by the engine:
  DSpark, DFlash, or baseline after any per-request capability fallback;
- sampling uses unbiased target-only MLX only when the selected path requires
  it; DSpark sampling remains on the DSpark latency path;
- result order is restored to input order and records `backend` as
  `mlx-continuous`, `dspark-latency`, `dflash-latency`, `baseline-latency`,
  `mlx-target-sampling`, or `error`. Other configured latency modes use their
  normalized generation backend followed by `-latency`.

`/v1/chat/completions` validates the same sampling fields. Its `tools` and
`tool_choice` fields are typed: `none` disables tool exposure, `auto` exposes
tools without forcing a call, while `required` and a named function choice are
enforced and produce an explicit error if the model does not comply.

`temperature` accepts 0 through 2, `top_p` is in `(0, 1]`, `top_k` and `seed`
are non-negative, and `stop` is one string or a list of one to four non-empty
strings. Omitted or zero temperature retains exact greedy speculation. With
DSpark selected, a positive value remains on its exact speculative sampler;
greedy-only DFlash/DDTree instead use target-only MLX sampling so verification
cannot bias the distribution. Textual stop matches are hidden across chunk boundaries and
trim complete responses, with usage adjusted to exposed output. They do not
yet stop the underlying generation loop early, so generation may continue up
to EOS or the token cap after output has been suppressed.

A Qwen 3.5 4B smoke with prompts `alpha` and `beta` completed in 0.734 s via
`mlx-continuous`. Because no sequential control was recorded, this is a
functional result, not a speedup claim. `/v1/batch` now uses this grouping path
per resolved tier, limits a request to 64 items, applies the configured prompt
policy, preserves item order, and reports the selected backend. It still runs
the whole HTTP batch under the process-wide Metal lock; separate HTTP calls
are not merged into a continuously scheduled batch.

## `mio bench`

Runs the built-in tier smoke benchmark:

```bash
mio bench
```

For a versioned parity/performance result, use
`scripts/bench_qwen36_matrix.py`; `mio bench` is an operational smoke, not the
research harness.

## `mio status` and `mio menu`

```bash
mio status
mio menu
```

`status` prints persisted tier configuration and probes the default local
server. `menu` provides a numbered wrapper over common commands.
