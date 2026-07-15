# Mio architecture

> Status: living architecture reference for the `codex/qwen36-mlx-engine`
> development line. This document describes the implementation, not a target
> diagram. Planned changes live in [13-development-plan.md](13-development-plan.md).

## 1. System purpose and invariants

Mio is a local-first inference and agent stack for Apple Silicon. Its primary
responsibility is to run an MLX target model efficiently while preserving an
OpenAI-compatible API and usable local agent/UI surfaces.

The implementation has five non-negotiable invariants:

1. Model tensors and prompts stay on the machine unless a user explicitly
   invokes a network skill or remote MCP provider.
2. MLX work is serialized at the HTTP boundary. Concurrent Metal encoders on
   the same model are not assumed to be safe.
3. Every accelerated decode path must retain a baseline autoregressive path
   and report when it falls back.
4. Quantized cache modes are mutually exclusive and must be selected
   explicitly in configuration.
5. Performance claims must identify the model, prompt, token budget, cache
   mode, commit, machine class, warm/cold state, and parity result.

## 2. Component map

```text
CLI / native agent / HTTP clients / browser UI
                    |
        +-----------+------------+
        |                        |
    mio.main                  mio.server
        |                        |
    mio.agent             FastAPI + WebUI router
        |                        |
        +----------+-------------+
                   |
            ModelManager
                   |
              MioEngine
       +-----------+------------+-------------+
       |           |            |             |
   baseline      DSpark      DFlash      DDTree / BMP
       |           |            |             |
       +-----------+------------+-------------+
                   |
         target model + caches
                   |
                  MLX
```

Cross-cutting services are configuration, model discovery/downloads, prompt
policy, context compaction, prefix-state reuse, tools/skills, MCP providers,
metrics, validation, scheduling, and user-data persistence.

## 3. Module ownership

| Area | Primary modules | Responsibility |
|---|---|---|
| CLI | `mio/main.py`, `mio/menu.py`, `mio/configure.py` | Parse commands, apply one-shot overrides, launch the selected surface. |
| Configuration | `mio/config.py` | Typed tier settings, safe defaults, migration and JSON persistence. |
| Model catalog | `mio/models/registry.py`, `mio/model_check.py`, `mio/pull.py` | Known repositories, tier resolution, completeness checks, resumable download. |
| Lifecycle | `mio/model_manager.py` | Own loaded engines and map logical tier names to instances. |
| Inference facade | `mio/engine.py` | Chat templating, dispatch, cache selection, metrics and prefix reuse. |
| Drafter selection | `mio/drafter_selection.py` | Read-only metadata classification, strict mode and compatible DSpark/DFlash fallback planning. |
| DSpark runtime | `mio/dspark_runtime.py` | Dedicated MLX worker, bounded streaming bridge, cancellation and upstream exact prefix-cache ownership. |
| Speculative runtime | `mio/dflash/` | Target/draft loading, Qwen hooks, prefill, verification, rollback and streaming. |
| Tree speculation | `mio/ddtree/` | Candidate-tree construction, tree-aware forward pass and cache commit. |
| KV compression | `mio/polarquant/`, `mio/turboquant/` | Quantized cache formats, kernels, attention adapters and patching. |
| API | `mio/server.py`, `mio/dashboard.py`, `mio/tool_calls.py` | OpenAI-compatible endpoints, SSE, metrics and tool-call normalization. |
| Context | `mio/compactor.py`, `mio/prefix_cache.py`, `mio/cache_store.py` | Prompt compaction and reusable conversation state. |
| Native agent | `mio/agent.py`, `mio/validator.py` | REPL, filesystem/shell tools, policy prompts and optional code validation. |
| Web app | `mio/webui/router.py`, `mio/webui/mio_ui.html`, `mio/webui/assets/` | Chat, artifacts, projects, workflows, settings and browser UX. |
| Skills | `mio/webui/skills*.py` | Built-in tools and the Mio user-skill catalog. |
| Automation | `mio/webui/scheduler.py`, `mio/webui/webhooks.py`, `mio/webui/flow_runner.py` | Local scheduled prompts, webhook templates and workflow execution. |
| Batch | `mio/batch.py`, `MioEngine.generate_batch` | Temperature grouping and MLX-LM continuous sessions with shared weights/independent caches. |
| Benchmarks | `scripts/bench_*.py`, `benchmarks/results/` | Reproducible performance and parity evidence. |

## 4. Configuration and model resolution

`MioConfig` owns server settings, active tiers and a `TierConfig` per tier. A
tier contains target/draft references, context/output limits, sampling values,
cache mode and speculative settings. `load_config()` begins with a deep copy of
the current registry defaults, then overlays `~/.mio/config.json`. Unknown
fields are ignored and missing new fields inherit current defaults.

The registry separates logical tiers from concrete repositories. Resolution
prefers a complete local model directory. Every checkpoint needs configuration
plus all declared weight shards; target checkpoints additionally need a
loadable tokenizer vocabulary and chat template. Draft checkpoints use the
weights-only contract because they do not own tokenization. This prevents a
partially interrupted Hugging Face download from being treated as loadable.

`mio pull` resolves tier shortcuts to checkpoint stacks and downloads into
`models/` and `spd/`. The Qwen 3.6 27B default stack contains the target,
preferred DSpark checkpoint, and a distinct DFlash fallback. Each component is
validated independently, downloads are resumable, and no model weight is committed to Git.

## 5. Model lifecycle

`ModelManager` creates one `MioEngine` per active tier. The engine loads the
target first, classifies the requested drafter metadata, then loads DSpark or
DFlash. Standard targets go through the MLX loader
and Mio's target hooks; PARO targets use their specialized loader before the
same runtime hooks are attached.

Draft loading is intentionally stricter than an ordinary optional feature:

- a missing or unreadable DSpark draft may fall back to a distinct compatible
  DFlash checkpoint, then to baseline decoding if both fail;
- a draft that loads but declares an incompatible target architecture is a
  configuration error, not a silent fallback;
- DSpark and hybrid checkpoints are never reused as pure DFlash fallbacks;
- the selected draft is bound to the loaded target after metadata compatibility checks.

Unloading releases model references, cached conversation state and MLX cache
objects owned by the engine. The manager remains the only public lifecycle
owner so callers do not share half-loaded instances.

## 6. Request path

For a chat request, the flow is:

1. Select a tier directly or through `TandemRouter`.
2. Normalize OpenAI message content into Mio's internal message shape.
3. Compact context if the configured occupancy threshold is exceeded.
4. Apply the selected Mio prompt policy (`none`, `caveman`, or `ponytail`).
5. Render the model-native chat template, including function schemas when
   supplied, and tokenize it.
6. Look for a reusable prefix state when that decode mode supports it.
7. Resolve the one active KV-cache mode.
8. Dispatch to DSpark, DDTree, BMP-DFlash, vanilla DFlash, or baseline AR.
9. Apply textual-stop output filtering and normalize usage. Stop matches are
   withheld across stream chunks; a match raises the shared cancellation
   signal and closes the backend generator.
10. Normalize model-native tool calls and enforce `tool_choice` requirements.
11. Stream SSE chunks or return a complete response, then record metrics.

The server holds a process-wide GPU lock around model generation. HTTP clients
may connect concurrently, but only one request encodes work for a shared MLX
model at a time. This is a correctness constraint; throughput scheduling is a
separate subsystem in the development plan.

## 7. Prefill

Prefill prepares target cache state for the entire prompt and captures the
hidden representation needed by the draft. Long prompts are chunked to bound
peak transient memory. For Qwen 3.6 sliding-window layers, the runtime applies
the model's effective attention window while retaining absolute token
positions.

Two optimizations avoid unnecessary vocabulary projection:

- baseline prefill projects only the final prompt position because generation
  only needs the next-token logits;
- DFlash projects captured context in chunks instead of materializing a
  prompt-length vocabulary tensor.

The benchmark harness reports prefill separately from decode; a decode gain is
never presented as a prefill gain.

## 8. DFlash speculative decode

DFlash predicts a block with a lightweight draft conditioned on target hidden
state. The target verifies the block in one forward pass. The longest accepted
prefix is committed, the first mismatch is corrected from target logits, and
both target and draft caches are rolled back or advanced to the same absolute
position.

Important runtime details:

- target weight packing fuses compatible linear projections but validates the
  packed output before replacing the reference path;
- hybrid Gated DeltaNet layers use dedicated state/tape handling, while full
  attention layers use KV caches;
- verification length is capped to a safe model-specific block size;
- EOS suppression for tool calls is temporary and relaxed after a bounded
  prefix so requests still terminate naturally;
- streaming and non-streaming implementations must return identical token IDs
  under deterministic sampling;
- summary events carry token IDs, timing, acceptance, cycles, fallback state
  and optional final cache state.

Qwen 3.6 support adds sliding-attention-aware draft execution and explicit
target/draft architecture validation. The project currently uses the 27B
UD-Q4_K_XL MLX target with a block-7 DSpark checkpoint and a separate block-16
DFlash fallback.

### 8.1 DSpark execution boundary

DSpark is selected before DFlash when its complete local checkpoint matches
the target. `DSparkRuntime` owns one single-thread executor because MLX streams
and lazily evaluated arrays are thread-affine in the exercised upstream
runtime. Model loading, generation, prefix-cache operations and final array
materialization all happen on that worker. Concurrent callers are serialized
there even when they enter Mio from different application threads.

Streaming crosses the worker boundary through a bounded queue. Consumer
closure sets a cancellation event, closes the upstream generator and joins the
producer; this prevents an abandoned HTTP stream from continuing to allocate
tokens or filling an unbounded buffer. Upstream prefix-cache failures are
fail-open: the cache is reset or disabled, while token generation remains the
correctness path. Cache status is observability only and cannot invalidate a
successfully loaded runtime.

The default Qwen 3.6 profile caps DSpark proposals at three tokens and disables
lookup drafts because proposal depths of four or more failed the current
strict parity corpus. These values are guarded runtime settings, not a general
claim that three is optimal. A required-tool request temporarily uses the
target-only streaming path so EOS can be suppressed for the opening tool-call
window and then restored exactly.

## 9. Alternative decode paths

BMP-DFlash expands multiple draft paths and verifies them together. It is
enabled only for compatible cache modes because batched rollback semantics are
more complex than the single-path case.

DDTree builds a candidate tree from draft probabilities, compiles parent/depth
metadata, runs tree-aware target verification and commits the accepted path.
Its cache policy is isolated from PolarQuant/TurboQuant and currently uses an
MLX quantized cache where supported.

Baseline AR remains the control path and the operational fallback. Runtime
selection is `complete local DSpark -> complete local compatible DFlash ->
target AR`; startup never downloads an absent optional drafter. Explicit
strict mode turns either drafter failure into an error for reproducible
research runs. Any new speculative strategy must demonstrate exact greedy
token parity against target AR before performance results are accepted.

For independent prompt batches, `MioEngine.generate_batch` uses MLX-LM's
continuous `BatchGenerator` path rather than forcing unrelated sessions into a
speculative stream. It shares target weights, owns a separate cache per
session, removes completed sequences from the active batch, and invalidates
speculative prefix snapshots afterward. `mio.batch` groups requests by the
complete temperature/top-p/top-k/seed sampler configuration because one active
MLX batch uses one sampler. A singleton follows the engine's normal selected
latency path (DSpark, DFlash, or baseline after any capability fallback), and
its batch result reports the backend that actually generated. Sampling uses
target-only MLX only when that selected path requires it. The DFlash model hook
accepts vector KV offsets so batched cache positions remain valid. This
establishes a functional batching path, not a measured throughput gain or
cross-request service scheduler.

## 10. Cache architecture

Mio distinguishes four kinds of state:

1. target autoregressive state: KV entries and recurrent/hybrid state;
2. draft state: draft KV/recurrent state plus target-context features;
3. rollback snapshots used inside one speculative cycle;
4. prefix-cache entries retained across requests.

PolarQuant and TurboQuant are storage/attention formats for target KV state.
They are mutually exclusive. Quantized caches must implement the MLX cache
contract (`state`, trimming, masking, size reporting) and speculative rollback
semantics; a smaller allocation alone is insufficient.

The prefix cache stores the post-generation final state under the complete
prompt-plus-generation token sequence. Lookup uses the longest common prefix,
truncates state to the reusable position and rents the entry so concurrent
callers cannot mutate it. Eviction is bounded by both entry count and an
approximate cumulative token budget.

## 11. Prompt policy and tool calling

Prompt policy is orthogonal to decoding. Agent, chat, and server expose
Caveman and Ponytail as mutually exclusive modes with `lite`, `full`, and
`ultra` levels, plus `none`. Policy text is merged with an existing system
message, but skipped when the client system prompt declares an exact external
tool protocol whose syntax must remain untouched.

The agent and API preserve exact tool names. `mio/tool_calls.py` recognizes
model-native XML/function forms, converts values to JSON-compatible types and
supports incremental parsing for SSE responses. API sampling validates
temperature/top-p/top-k/seed. Positive temperature stays speculative when
DSpark is selected; greedy-only DFlash/DDTree use an observable target-only
fallback. `tool_choice` supports `none`, `auto`, `required`, and a named
function, with explicit errors for unmet required/named choices. Textual stops
trim exposed output and signal early cancellation to the active streaming
backend.

## 12. Native agent

The native agent owns conversation history and the interactive tool loop. It
exposes bounded filesystem inspection/editing, search, shell execution and
project status tools. Tool results are appended as structured messages before
the next model turn. Optional validation runs syntax and configured linters on
generated code.

Instruction skills are not registered as hundreds of independent function
schemas. Mio exposes catalog search and bounded skill reading so the model can
load only the relevant instructions. Executable skill scripts are a separate,
explicit trust path.

## 13. MCP boundary

Mio's MCP client treats each provider as a process or endpoint with an explicit
transport, timeout, output limit and permission policy. Local stdio providers
are enabled by default. Remote/authenticated providers are opt-in.

The built-in local declarations are:

- Mio's Karpathy-style LLM Wiki server, backed by `~/.mio/wiki`;
- Headroom, installed in an isolated environment under `~/.mio/tools`;
- Ponytail's local prompt/tool server when Node dependencies are available.

Provider failures are isolated: one unavailable provider must not prevent Mio
from starting or hide built-in tools.

Enabled declarations are loaded into a shared hub and provider processes start
lazily. Enabled, unauthenticated local providers receive their declared
permissions by default; remote or authenticated providers require explicit
policy and grants. The native agent and Web UI expose bounded generic
`list_mcp_tools`/`call_mcp_tool` bridges. `/v1/mcp/servers` remains a
declaration/status endpoint and never launches or calls a provider.

Provider enablement and model consent are separate layers. The Web UI marks
both MCP bridges as sensitive orchestration: model auto-use requires an exact
operator grant and the same per-request grant, while a direct sensitive run
also requires confirmation. Public read-only Web UI tools are the only
automatic default.

## 14. HTTP server

`mio/server.py` implements model listing, health, metrics, tier lifecycle,
single and streaming chat completions and batch requests. It mounts the WebUI
router only when requested. A live console panel consumes the same generation
metrics recorded for HTTP responses.

The safe default is loopback binding. The server refuses a non-loopback host
unless `--unsafe-remote-bind` or `MIO_UNSAFE_REMOTE_BIND=1` explicitly
acknowledges the risk. Cross-origin access is restricted to allowed origins;
the opt-in adds no authentication and must be paired with a trusted firewall
or authentication/reverse-proxy policy.

## 15. Web application

Mio UI is a server-rendered HTML shell plus modular JavaScript assets. The
backend router provides sessions, projects, attachments, artifacts, skills,
RAG, workflows, schedules, webhooks and chat streaming. Persistent user data
lives under `~/.mio`; repository code and user content are deliberately
separate.

Security boundaries are enforced server-side:

- Host validation rejects untrusted hostnames and DNS-rebinding attempts;
- browser mutations and WebSockets require same-origin session/CSRF state;
- session IDs and user-supplied paths are normalized and confined;
- uploads have bounded, chunked reads and confined storage paths;
- untrusted Markdown/HTML is sanitized before insertion;
- generated artifacts render in a restricted iframe sandbox;
- external fetches reject non-global targets and revalidate redirects;
- CSP and security headers are emitted, with documented legacy inline
  compatibility allowances;
- sensitive built-in skills require exact operator/request grants, and direct
  invocations require per-call confirmation;
- external catalog runners are not exposed by the default agent/UI tools.

Executable built-in skills and generated artifacts remain a trusted-local-user
surface. These controls do not provide bearer authentication or hostile
multi-user isolation.

Client modules share state through a single documented `window.Mio` namespace
rather than depending on accidental top-level `let` bindings.

## 16. Automations and flows

Schedules and webhooks persist as JSON/JSONL under `~/.mio`. The scheduler is
an in-process asyncio task started with the WebUI lifecycle. Each run resolves
an available tier, streams a bounded response and records its result.

Flow execution validates a graph, topologically orders ready nodes and dispatches
typed handlers such as LLM calls, skills, transforms, conditionals, RAG and
artifact emission. User-input nodes are resolved from run input; they are not
silently replaced with empty values. Run state is streamed to the UI and saved
for later inspection.

Artifact nodes emit structured, bounded events that the client validates,
adds to the ordinary artifact gallery/version store, and opens in the artifact
panel. Limits are 16 artifacts per run, 512 KiB per artifact, and 2 MiB total;
there is no placeholder JSONL handoff.

Saved flows can be published persistently behind two stable built-in tools:
`list_flow_skills` and `run_flow_skill`. Mio deliberately does not add one
function schema per published graph. Save/run limits include 200 graph nodes,
200 execution hops, a 2 MiB graph document, 64 KiB of arguments, a 256 KiB
published result and a 120-second published-run timeout. Recursive flow-skill
dispatch is rejected. The present executor is serial within a graph and has no
retry/backoff or intra-flow parallelism.

## 17. Persistence map

| Data | Default location | Format |
|---|---|---|
| Runtime configuration | `~/.mio/config.json` | JSON |
| MCP providers | `~/.mio/mcp.json` | JSON |
| Installed skills | `~/.mio/skills/` | Agent Skills directories + lock metadata |
| Mio-managed tools | `~/.mio/tools/`, `~/.mio/bin/` | Isolated runtimes/binaries |
| LLM Wiki | `~/.mio/wiki/` | Markdown + indexes/log |
| Sessions/projects/prompts | `~/.mio/` | JSON |
| Schedules/webhooks/runs | `~/.mio/` | JSON/JSONL |
| RAG metadata | `~/.mio/rag.sqlite` | SQLite |
| Downloaded models | repository `models/`, `spd/` by default | HF/MLX checkpoint files |
| Benchmark evidence | `benchmarks/results/` | Versioned JSON |

## 18. Verification layers

Unit tests cover configuration migration, server response shape, compaction,
prefix reuse, cache formats, DFlash compatibility and tree construction.
Model-marked tests exercise real local checkpoints. A release benchmark should
add three gates that unit tests cannot replace:

1. exact deterministic token parity;
2. warm and cold performance distributions, not a single run;
3. peak unified-memory measurement.

Release browser QA must exercise the built WebUI against a live local server, including
mobile layout, settings persistence, session operations, skill discovery,
flow execution, artifact isolation and streaming error states.

## 19. Known architectural pressure points

The current implementation is functional but several files exceed a healthy
ownership boundary: `mio/webui/router.py`, `mio/webui/skills.py`,
`mio/webui/mio_ui.html`, `mio/dflash/runtime.py`, and `mio/server.py`. Their
decomposition is a maintainability task, not a cosmetic refactor: it enables
isolated security policy, transport tests, cache invariants and performance
experiments without cross-subsystem regressions.

See [13-development-plan.md](13-development-plan.md) for the staged extraction
and the quantitative acceptance criteria for each subsystem.
