# Mio

> *Ci sono tanti engine, ma questo è Mio — e quindi anche tuo.*

Mio is a local-first MLX inference engine, coding agent, OpenAI-compatible
server, and browser UI for Apple Silicon. The current development line adds a
tested Qwen 3.6 27B target with DSpark and DFlash drafters, Mio-owned MCP
providers, a reviewed managed snapshot of 916 external instruction skills,
and two mutually exclusive prompt policies: Caveman and Ponytail.

The project is deliberately evidence-driven. Performance numbers below come
from versioned raw results; broader claims such as "4.1x", "10-20x end to
end", or unmeasured coding-quality improvements are not Mio results and are
not claimed here.

## Verified snapshot

The checked-in Qwen 3.6 measurements use:

- target: `Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx`;
- draft: `z-lab/Qwen3.6-27B-DFlash`;
- Apple M4 Max, 48 GB unified memory, macOS 26.5.1;
- Python 3.12.0, MLX 0.32.0, mlx-lm 0.31.3, dflash-mlx 0.1.8;
- 256 prompt tokens, 32 generated tokens, one warm-up, two measured runs;
- benchmark commit `d49dec26dbd6053526027e013d5580e9cf5c10f4`.

| Mode | Prefill tok/s | Decode tok/s | End-to-end tok/s | Greedy token parity |
|---|---:|---:|---:|---|
| Target autoregressive baseline | 234.77 | 19.31 | 11.64 | control |
| DFlash | 232.92 | 33.64 | 15.61 | yes |

On this short workload DFlash improved decode throughput by **1.74x** and
end-to-end throughput by **1.34x**. Prefill was effectively neutral
(`0.992x`). This is not evidence of a prefill speedup, long-context behavior,
or a coding-quality gain. See [the benchmark guide](docs/16-benchmarks.md),
[raw results](benchmarks/results/qwen36-core-256.json), and the
[research paper](papers/mio-qwen36-research.md).

## Architecture

```text
terminal agent     OpenAI clients       Mio UI
      |                  |                 |
      +------------------+-----------------+
                         |
                  API / orchestration
                         |
     prompt policy · tools · Mio skills · local MCP
                         |
                      MioEngine
          +--------------+---------------+
          |              |               |
      baseline AR   DSpark/DFlash    DDTree / BMP
          +--------------+---------------+
                         |
           prefix state · KV-cache modes
                         |
                  MLX / Apple Metal
```

The detailed implementation map and subsystem boundaries are in
[docs/12-architecture.md](docs/12-architecture.md). The staged improvement
plan is in [docs/13-development-plan.md](docs/13-development-plan.md).

## Install

Requirements:

- an Apple Silicon Mac;
- Python 3.10 or newer;
- enough unified memory and disk for the selected model;
- Git for the optional external skill and Ponytail source installs;
- Node.js only for the optional local Ponytail MCP provider.

```bash
git clone https://github.com/Ruler-Dev/mio.git
cd mio
python3 -m pip install -e .
mio --help
```

The MLX runtime, server, UI, and built-in tools are Python dependencies. Model
weights and reviewed external skill repositories are intentionally installed
separately: they are large, independently licensed, and live outside the
Python package.

## Qwen 3.6 27B speculative stack

The `large` tier uses the Qwen 3.6 27B MLX target, prefers its compatible
DSpark checkpoint, and keeps a distinct DFlash checkpoint as the automatic
runtime fallback. If the Qwen 3.6 target/DFlash pair is incomplete, tier
resolution safely falls back to the registered Qwen 3.5 pair.

```bash
# Downloads target, preferred DSpark, and DFlash fallback into models/ and spd/.
mio pull large

# Confirm target weights/tokenizer/template and every draft shard.
python3 -m mio.model_check
```

Equivalent explicit downloads are:

```bash
hf download Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --local-dir models/Qwen3.6-27B-UD-Q4_K_XL-mlx
hf download Avesed/Qwen3.6-27B-DSpark \
  --local-dir spd/Qwen3.6-27B-DSpark
hf download z-lab/Qwen3.6-27B-DFlash \
  --local-dir spd/Qwen3.6-27B-DFlash
```

Do not put Hugging Face tokens in shell history, repository files, benchmark
JSON, or issue reports. Authenticate with `hf auth login` or a scoped
environment in your password manager.

Other tier shortcuts remain available:

| Tier | Preferred pair | Registered context |
|---|---|---:|
| `large-moe` | complete local Qwen 3.6 35B-A3B pair, otherwise Qwen 3.5 35B-A3B | 256K or 128K |
| `large` | Qwen 3.6 27B target + DSpark + DFlash fallback, otherwise Qwen 3.5 27B | 256K or 32K |
| `medium` | Qwen 3.5 9B UD-Q4_K_XL + matching draft | 16K |
| `small` | Qwen 3.5 4B 4-bit + matching draft | 8K |

These are registry limits, not promises that every context length fits every
Mac. Actual memory depends on model, cache mode, prompt length, and concurrent
applications.

## Run

```bash
# Native coding agent. The first persisted active tier is used unless set.
mio --tier large --workspace .

# Headless chat.
mio chat --tier large --prompt-mode none

# Local OpenAI-compatible API.
mio serve --tier large

# API plus browser UI at http://127.0.0.1:9090/ui.
mio serve --tier large --webui
```

Native agent file and shell tools are governed by Mio's own tool policy. The
CLI grants its displayed workspace read/write/shell access and preserves a real
`zsh` for pipes, redirects, and scripts needed by agent ecosystems such as
Hermes or OpenClaw. Child commands run inside a workspace-confined inherited
macOS default-deny sandbox with a sanitized environment, bounded descendants,
timeout/output caps, and network disabled unless `--agent-network` grants it
for that session. A fail-closed hard-link preflight prevents allowed pathnames
from aliasing outside inodes. `--agent-root` adds a deliberate extra root;
home, `/`, and broad system/volume roots require
`--unsafe-broad-workspace`. Library callers that omit a policy are read-only;
the model cannot expand its own roots or permissions.

The main chat shell also serves its versioned Marked/Prism runtime locally, so
Markdown and code highlighting do not need a CDN at boot. Optional artifact
renderers may still load their separately sandboxed libraries on demand. The
redistributed asset licenses are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The server binds to `127.0.0.1` by default and refuses a non-loopback address
unless `--unsafe-remote-bind` (or `MIO_UNSAFE_REMOTE_BIND=1`) is supplied.
That flag is only an administrative acknowledgement: it does not add
authentication. Use it only behind a trusted firewall or authenticated reverse
proxy with an explicit origin policy.

OpenAI SDK example:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9090/v1", api_key="local")
response = client.chat.completions.create(
    model="mio-large",
    messages=[{"role": "user", "content": "Write a small Python parser."}],
    stream=True,
)
```

The API validates `temperature`, `top_p`, `top_k`, `seed`, up to four textual
`stop` strings, typed function tools, and `tool_choice` values
`none`/`auto`/`required`/named. Omitted or zero temperature keeps exact greedy
speculation. DSpark implements exact speculative sampling for a positive
temperature; the greedy-only DFlash/DDTree paths deliberately use target-only
MLX sampling instead of biasing the requested distribution.
Textual stops are withheld/trimmed from complete and streamed responses, but
now also close the active stream and signal cancellation to the backend. The
observed savings still depend on decode chunk size and backend cancellation
latency, so stops are not presented as a fixed performance multiplier.

See [docs/02-commands.md](docs/02-commands.md) for the CLI and API surface.

## Mio UI and the native Artifact Lab

`mio serve --webui` exposes the browser application at
`http://127.0.0.1:9090/ui`. In addition to the general artifact renderers, the
UI ships three MLX-oriented Artifact Lab renderers as local JavaScript and
CSS assets:

| MIME type | Required payload shape | Purpose |
|---|---|---|
| `application/vnd.pimio.benchmark+json` | `{runs: [...]}` with at least one prefill, decode, or TTFT value | Compare matched MLX runs |
| `application/vnd.pimio.model-card+json` | `{name: "..."}` or `{model: "..."}` | Record checkpoint and compatibility metadata |
| `application/vnd.pimio.inference-trace+json` | `{spans: [...]}` with non-negative start and duration values | Inspect a measured request timeline |

These three renderers have no CDN dependency, do not execute artifact content,
and insert payload values into the page as text. Each JSON document is limited
to 512 KiB; benchmark payloads accept at most 48 runs and trace payloads at
most 256 spans. Invalid input produces an error card and remains editable in
the Source tab. Downloads are deterministic, pretty-printed JSON when the
payload parses successfully.

The model emits an artifact in the same envelope used by the rest of Mio UI:

```xml
<antArtifact identifier="mlx.run-01"
  type="application/vnd.pimio.benchmark+json"
  title="Matched MLX runs">
{"runs":[{"label":"baseline","prefill_tps":234.77,"decode_tps":19.31}]}
</antArtifact>
```

The values in that example are taken from the verified snapshot above; an
Artifact Lab renderer only presents supplied data and does not measure or
validate a performance claim. The model prompt explicitly tells the model not
to invent benchmark values or trace spans.

Artifact MIME aliases are normalized to the key used by the actual renderer.
Normalization is idempotent: applying it to an already-canonical React, code,
or Mermaid type does not change the type a second time. Artifact identifiers
may contain letters, digits, dots, underscores, and hyphens after an initial
alphanumeric character, so identifiers such as `mlx.run-01` survive parsing
and session reloads.

Session auto-save and JSON export retain complete artifact history in the
versioned `artifact_state` document:

```json
{
  "schema_version": 2,
  "active_artifact_id": "mlx.run-01",
  "chains": [
    {
      "id": "mlx.run-01",
      "active_index": 0,
      "revisions": [
        {
          "id": "mlx.run-01",
          "type": "application/vnd.pimio.benchmark+json",
          "title": "Matched MLX runs",
          "language": "",
          "content": "{\"runs\":[{\"label\":\"baseline\",\"decode_tps\":19.31}]}",
          "created_at": "2026-07-16T00:00:00.000Z",
          "content_id": "fnv1a32:31dc2a55",
          "provenance": {"producer": "chat"}
        }
      ]
    }
  ]
}
```

The `content_id` is a deterministic, non-cryptographic revision identifier,
not an integrity signature. Source edits append a revision and record the
prior `content_id` as provenance. Version 2 restores every bounded revision
chain and its selected `active_index`; imports without `artifact_state` still
accept the legacy `artifacts` list. JSON exports also include messages, title,
project association, and the per-chat system prompt.

On narrow viewports the artifact panel becomes a fixed sheet beside the
navigation rail rather than a fourth grid column. Preview and Source are real
tab buttons with selected-state metadata; dialogs, artifact regions, artifact
cards, gallery cards, and the send control expose labels or roles, and keyboard
focus receives a visible outline. These changes improve the current keyboard
and mobile structure, but live cross-browser and assistive-technology QA is
still part of the release gate. The complete renderer matrix and payload
examples are in [docs/11-mio-ui.md](docs/11-mio-ui.md).

## Prompt policies

`mio chat` and `mio serve` expose one prompt policy at a time:

```bash
mio serve --prompt-mode none
mio serve --prompt-mode caveman --prompt-level lite
mio serve --prompt-mode ponytail --prompt-level full
```

| Mode | Purpose | Evidence status |
|---|---|---|
| `none` | No Mio policy injection | control |
| `caveman` | Encourage concise responses | output/task impact not yet benchmarked on Qwen 3.6 |
| `ponytail` | Prefer the smallest sufficient engineering change | coding quality and token impact not yet benchmarked |

Levels are `lite`, `full`, and `ultra`; the default for agent/chat/server is
`caveman/full`. Legacy `--caveman ...` and `--ponytail ...` aliases remain.
Policy injection is skipped for known exact XML tool protocols. The native
agent accepts the same top-level flags and provides both `/caveman` and
`/ponytail` runtime commands.

## Mio-owned MCP

MCP configuration belongs to Mio at `~/.mio/mcp.json`, not to Codex or another
agent. Local unauthenticated providers are registered as enabled by default;
remote or credential-bearing providers are opt-in.

```bash
mio mcp install-tools
mio mcp doctor
mio mcp check --json
mio mcp list
mio mcp disable ponytail
mio mcp enable ponytail
mio serve --mcp-config ~/.mio/mcp.json
```

Mio UI Settings reports provider readiness through the CSRF-protected
`POST /v1/mcp/health`. The POST boundary matters because a probe can launch a
local provider process. It contacts only enabled local unauthenticated
stdio providers under fixed budgets and a dedicated least-authority sandbox;
remote, authenticated and unisolatable HTTP/SSE entries are skipped. Commands,
URLs, tool names, secrets, and raw errors are never returned to the browser.

The built-in presets are:

- `llm-wiki`: Mio's local Karpathy-style evidence wiki;
- `headroom`: a Mio-isolated Headroom command under `~/.mio/bin`;
- `ponytail`: a read-only local provider under `~/.mio/tools/sources`.

Enabled local providers are eligible through bounded
`list_mcp_tools`/`call_mcp_tool` bridges in the native agent and Web UI, but
registry availability is not blanket consent for a browser model loop. In Mio UI,
public read-only tools are the only automatic default; local reads/writes,
processes, private-network access, flows, and MCP orchestration require an
exact operator grant plus per-request consent. Direct sensitive UI calls also
require confirmation. Provider processes start lazily on first use, and an
enabled unauthenticated local declaration receives exactly its declared
permissions only after the calling surface's policy permits the call. Remote
or credential-bearing providers remain blocked without explicit
per-command/application policy. Missing optional executables do not prevent
Mio from starting. Setup and permission semantics are in
[docs/15-mcp.md](docs/15-mcp.md). The source-tree script remains only a thin
compatibility wrapper around `mio mcp install-tools`.

An operational MCP smoke discovered and called all three providers. On one
synthetic 300-line JSON input, Headroom's direct `headroom_compress` call
reported 7,862 to 2,484 tokens (5,378 saved, 68.4%) with `smart_crusher`.
This one input is not a Qwen benchmark or evidence of general task-quality,
latency, or compression gains; the local proxy on port 8787 was not running.

## Managed external skills, inside Mio

The reviewed catalog is installed under `~/.mio/skills` (or
`$MIO_HOME/skills`) and is never installed into Codex or Claude directories.

```bash
python3 scripts/install_mio_skills.py
```

| Source | Included skills |
|---|---:|
| Nutlope/hallmark | 1 |
| mattpocock/skills (active set) | 26 |
| Ruler-Dev/Anthropic-Cybersecurity-Skills | 817 |
| Ruler-Dev/Claude-Code-Game-Studios | 72 |
| **Expected managed snapshot** | **916** |

Mio integrates the catalog through `list_mio_skills` and `read_mio_skill` in
the native agent and Web UI. The model retrieves only relevant instructions
instead of receiving one schema per managed skill on every request. The 916
figure is the installer-verified count at the pinned reviewed revisions, not a
hard-coded live total: unmanaged Mio-local skills may make discovery return a
different number. Repository code is not executed during reading; executable
runners require a separate explicit trust path. When an authorized Mio agent
does need a skill's shell workflow, it receives the same real-zsh semantics
inside the agent workspace sandbox; shell, network, and extra roots remain
separate trusted-caller grants. See
[docs/14-external-skills.md](docs/14-external-skills.md).

## Cache modes and speculative paths

`TierConfig` currently defaults to PolarQuant 4-bit (`pq_bits=4`) and
TurboQuant off (`tq_bits=16`). `--tq4` selects TurboQuant and disables
PolarQuant because the formats are mutually exclusive. `--mpath K` opts into
BMP verification; DDTree has its own compatibility gates. DSpark owns its
upstream exact cache and therefore bypasses Mio's PolarQuant/TurboQuant cache
for that request; the selected policy is exposed in generation telemetry.

The current short Qwen 3.6 cache ablation is not strong enough to justify a
universal cache recommendation:

- DFlash + PQ4 changed the deterministic output in both measured runs;
- DFlash + TQ4 preserved parity but reduced end-to-end throughput by 7.7% and
  increased reported peak memory by 2.34 GB on the 256+32-token workload;
- the run is too short to measure long-context KV-memory savings.

Use the unquantized baseline/DFlash path when exact greedy parity is required,
and benchmark cache modes on the intended context and task distribution.

## Benchmark reproduction

```bash
python3 scripts/bench_qwen36_matrix.py \
  --tier large \
  --prompt-tokens 512 \
  --max-tokens 64 \
  --warmup 1 \
  --reps 3 \
  --modes baseline,dflash,pq4,tq4
```

Results include raw timings, package versions, model references, token hashes,
parity, commit, and dirty-tree state. The benchmark is a microbenchmark, not a
coding-agent evaluation. Full methods and limitations are in
[docs/16-benchmarks.md](docs/16-benchmarks.md).

## Batch inference

`mio batch` groups requests by temperature/top-p/top-k/seed and uses MLX-LM
continuous batching for groups of two or more, with shared model weights and
independent per-session KV caches. A singleton uses the engine's normal
latency path—DSpark, DFlash, or baseline according to runtime selection and
per-request fallback. Its reported backend reflects the path that actually ran;
sampling falls back to unbiased target-only MLX only when the selected path
requires it.

```bash
mio batch --input prompts.jsonl --output results.jsonl --tier large
```

A real Qwen 3.5 4B two-prompt smoke completed through the `mlx-continuous`
backend in 0.734 s. There is no sequential control for that run, so it proves
the path works but not that it is faster. `/v1/batch` routes each model tier
through the same sampler-grouped path and restores input order. It still
runs under the process-wide Metal lock and does not continuously combine
separate HTTP calls, so it is not yet a multi-request service scheduler.

## Documentation

- [Documentation index](docs/00-index.md)
- [Getting started](docs/01-getting-started.md)
- [Mio UI and artifact formats](docs/11-mio-ui.md)
- [Architecture](docs/12-architecture.md)
- [Development plan](docs/13-development-plan.md)
- [External skills](docs/14-external-skills.md)
- [MCP](docs/15-mcp.md)
- [Benchmarks](docs/16-benchmarks.md)
- [Mio Qwen 3.6 research paper](papers/mio-qwen36-research.md)

## Research status

The current working-tree R&D includes matched Qwen 3.6 27B DSpark and upstream
DFlash experiments. DSpark cap 2/cap 3 and upstream DFlash preserved all 12
paired outputs, but every candidate regressed TTFT; cap 4 and the full DSpark
block also failed parity. A separate short fused-cold-prefill pilot improved
TTFT by 1.1555x and end-to-end time by 1.0794x with 12/12 parity, but remains a
single-thread, short-context prototype. None is a breakthrough or production
speed claim.

Mio has not yet measured SWE-bench-style task success, edit correctness,
tool-call accuracy, Caveman/Ponytail quality, long-context scaling, or
multi-user throughput. A result will be called a breakthrough only if
independent reruns show a Pareto improvement without unacceptable regressions
in parity, quality, TTFT, memory, or reliability. Negative experiments remain
publishable results.

## License and upstream work

Mio is MIT licensed. Model checkpoints retain their own licenses. The project
builds on [MLX](https://github.com/ml-explore/mlx),
[mlx-lm](https://github.com/ml-explore/mlx-lm),
[DFlash](https://github.com/z-lab/dflash),
[Headroom](https://github.com/headroomlabs-ai/headroom),
[Ponytail](https://github.com/DietrichGebert/ponytail), and the external skill
sources listed above. See the paper references for the complete provenance
used by this research snapshot.
