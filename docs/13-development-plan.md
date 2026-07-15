# Mio development plan

> Goal: turn Mio into a fast, evidence-driven MLX engine and a coherent local
> agent product without trading correctness or local security for benchmark
> numbers.

This plan is organized by acceptance gates rather than calendar estimates.
Items marked **landed** are implemented on the active development branch;
items marked **current** are part of the present integration checkpoint.

## 1. Definition of done

A change is complete only when all applicable conditions hold:

- deterministic token parity is tested against the appropriate control;
- failure and fallback behavior are observable;
- unit tests and real-model smoke tests pass;
- user-visible configuration is persisted and documented;
- security-sensitive inputs have size, path, origin and timeout bounds;
- a benchmark result records commit, dirty state, hardware class, model refs,
  parameters, repetitions and raw per-run values;
- the relevant README/docs are updated in the same checkpoint;
- the checkpoint is committed and pushed before another risky layer begins.

## 2. Priority map

| Priority | Meaning | Exit condition |
|---|---|---|
| P0 | Correctness/security blocker | No silent corruption, exposed local service or arbitrary path escape. |
| P1 | Product-completeness blocker | Requested surface works end to end and has tests. |
| P2 | Performance/maintainability | Measured gain or materially smaller change surface. |
| P3 | Research/optional ecosystem | Reproducible experiment with an honest result, positive or negative. |

## 3. Inference core

### 3.1 Target and draft compatibility

- **Landed:** Qwen 3.6 27B target and DFlash registry entries, role-aware
  completeness validation (target shards + tokenizer/template; draft shards),
  resumable pull and matching target/draft validation.
- **Landed:** sliding-window-aware draft execution and effective-window
  resolution for Qwen 3.6.
- Add a compatibility manifest keyed by architecture, hidden size, vocabulary,
  rope settings, layer layout and training target hash.
- Refuse approximate compatibility unless the operator opts into an experiment.
- Add a model-marked test for each supported target/draft family.

Acceptance: an incompatible draft fails before generation; every supported
pair produces the same greedy tokens as target AR on the parity corpus.

### 3.2 Prefill

- **Landed:** last-position-only LM-head projection for baseline, BMP and
  DDTree prefill.
- **Landed:** chunked draft-context projection to avoid prompt-length
  vocabulary materialization.
- Profile chunk size by prompt length and available unified memory instead of
  using one global value.
- Fuse capture/projection paths where it reduces command-buffer boundaries.
- Evaluate paged or streaming prompt ingestion for 32K-256K contexts.
- Add separate time series for tokenization, target forward, projection,
  draft-context build and first-token latency.

Acceptance: no token-parity regression, no higher peak memory, and a median
prefill improvement with a 95% bootstrap interval above zero on at least three
prompt lengths. A neutral experiment is documented and not enabled.

### 3.3 DFlash decode

- **Landed:** target packing, speculative hooks, rollback, prefix-aware state,
  streaming and Qwen 3.6 support.
- Replace fixed proposal length with an online controller using recent
  acceptance, verify cost and memory pressure.
- Cache compiled MLX graphs by shape bucket to reduce first-cycle overhead.
- Profile every synchronization and remove host reads from the hot path.
- Compare single-path, BMP and adaptive tree strategies per workload class.
- Investigate vocabulary pruning only behind an exact-correction mechanism.

Acceptance: exact deterministic parity and higher median completion tok/s than
the current DFlash control on code, chat and structured-tool workloads. Report
p1 latency and peak memory as well as mean throughput.

### 3.4 DDTree and BMP

- Make strategy selection data-driven rather than a static CLI choice.
- Add a common proposal/verification protocol so DFlash, BMP, DDTree and
  DSpark can share metrics and cache contracts.
- Bound tree width/depth by predicted utility per verified node.
- Add property tests for parent maps, DFS ordering and rollback after every
  possible acceptance length.
- Do not combine quantized caches with a speculative path until rollback tests
  cover the exact cache implementation.

Acceptance: strategy auto-selection never underperforms the best fixed safe
default by more than 5% on the calibration suite and never changes output.

### 3.5 DSpark research and guarded runtime integration

- **Landed on the development branch:** metadata-driven DSpark selection,
  strict mode, a distinct DFlash load-failure fallback, selection telemetry,
  and complete-stack `mio pull` support for Qwen 3.6 27B.
- Keep model-specific caps behind parity evidence. The current 27B profile uses
  cap 3 without lookup; cap 4 and the full block failed the strict parity gate.
- Establish controls: target AR, current DFlash, DDTree and BMP.
- Test a known compatible draft before training or converting a new one.
- If target compatibility fails, record it and build a conversion/training
  experiment on an isolated branch/worktree.
- Run ablations on proposal depth, tree construction, verifier batching,
  quantization and hybrid DFlash/DSpark routing.
- Promote a result only after independent reruns and parity validation.

Breakthrough criterion: a reproducible Pareto improvement over target AR and
the best currently safe speculative backend in both prefill/TTFT and decode,
or a decode improvement without a statistically or operationally meaningful
regression in TTFT, memory, parity, reliability or quality. Beating a slower
experimental path is insufficient. Iteration alone is not a scientific
result.

## 4. Cache subsystem

### 4.1 Prefix cache

- **Landed:** position-aware prefix truncation, final-state storage and
  token-budget eviction.
- Add explicit cache-state schemas instead of loosely typed dictionaries.
- Hash immutable model/policy/tool-template inputs into the cache key.
- Add per-entry byte estimates and unified-memory pressure eviction.
- Test divergent multi-turn branches and simultaneous rented entries.

Acceptance: warm requests skip exactly the reported token count, match cold
tokens and do not grow memory beyond the configured budget.

### 4.2 PolarQuant and TurboQuant

- **Landed:** deterministic configuration migration and mutual exclusion.
- Implement/verify complete `state` restore contracts for every cache version;
  do not leave mutating setters as placeholders.
- Consolidate versioned cache implementations behind one tested protocol.
- Add quantization-error, rollback and long-context drift tests.
- Select cache mode from a measured memory/throughput/quality policy, not a
  marketing default.

Acceptance: restore/trim round trips preserve subsequent greedy tokens; stated
memory savings match allocated bytes; speed claims include prefill impact.

## 5. Scheduling and throughput

### 5.1 HTTP request scheduler

- Replace the process-wide lock as the scheduling abstraction with a bounded
  request queue that still serializes unsafe MLX encoder access.
- Add cancellation, deadlines, backpressure and per-tier queues.
- Report queue time separately from inference time.
- Prevent a long batch or 256K prefill from starving interactive traffic.

Acceptance: overload returns a bounded error, cancelled clients release work,
and interactive p95 latency remains within the configured service objective.

### 5.2 Continuous batching

- **Landed:** `MioEngine.generate_batch` uses MLX-LM continuous batching with
  shared weights, independent session caches, completed-sequence removal, and
  vector KV-offset support in the target hook.
- **Landed:** file/CLI batches are grouped by temperature/top-p/top-k/seed;
  single-item groups use the normal selected DSpark, DFlash, or baseline
  latency path and report the backend that actually ran. Stochastic requests
  use the unbiased target-only sampler only when the selected path requires
  that fallback; per-request stops apply and order is stable.
- **Current limitation:** textual stops filter/trim exposed output but do not
  yet interrupt the underlying generation loop, so they are not a compute
  optimization.
- **Current:** `/v1/batch` resolves tiers, applies prompt policy, limits each
  request to 64 items, and routes each tier through the same temperature-grouped
  path while restoring original order.
- Extend continuous batching across independent HTTP requests; the current
  handler batches only the items submitted together and holds the Metal lock
  for the whole request.
- Add fair queueing/cancellation around continuous batches before treating the
  mechanism as a service scheduler.
- Group prefills by length buckets and decode cycles by verifier shape.
- Add throughput-vs-latency benchmarks at concurrency 1, 2, 4 and 8.

Acceptance: total throughput improves over sequential processing without token
changes, cache cross-talk or unbounded latency.

Current evidence: a real Qwen 3.5 4B two-prompt smoke (`alpha`, `beta`)
completed with backend `mlx-continuous` in 0.734 s. It has no sequential
control and therefore establishes functionality only, not throughput gain.

### 5.3 Tandem router

- Fix routing from a static keyword heuristic to observable task/model policy.
- Add model availability, context length, queue depth and memory headroom.
- Record routing decisions and counterfactual quality/latency samples.
- Allow deterministic rules for deployments that do not want learning.

Acceptance: every decision is explainable and fallback cannot select an
unloaded or context-incompatible tier.

## 6. Server and API

### 6.1 Local security

- **Current:** bind to `127.0.0.1` by default.
- **Current:** restrict CORS to loopback origins with explicit override.
- **Landed:** reject non-loopback binds unless `--unsafe-remote-bind` (or the
  matching environment opt-in) explicitly acknowledges the unauthenticated
  deployment risk.
- Add optional bearer authentication before supporting non-loopback binds.
- **Landed:** cap every HTTP request body at 32 MiB and uploads at 25 MiB.
- Add tighter message- and tool-schema-specific limits below the global cap.
- **Landed:** normalize and validate stop/temperature/top-p/top-k/seed, retain
  greedy DFlash by default, and route explicit stochastic requests to the
  scientifically compatible target-only sampler.
- Add structured error codes and redact secrets from debug logs.

Acceptance: security tests cover origin rejection, traversal, oversize input,
auth and log redaction; LAN exposure requires an explicit operator action.

### 6.2 OpenAI compatibility

- **Landed:** complete `stop`, `top_p`, `top_k`, seed and trimmed usage semantics
  across streaming, non-streaming and submitted batches.
- **Landed:** type `tools` plus none/auto/required/named `tool_choice`; only
  required/named choices add forcing, and violations fail explicitly.
- Match tool-call chunking and finish reasons used by current OpenAI clients.
- Stop backend generation as soon as a textual stop is confirmed rather than
  only suppressing subsequent output.
- Add a contract suite driven by captured, secret-free SDK requests.
- Version Mio-specific extensions under a separate namespace.

Acceptance: supported clients pass the contract suite; unsupported fields
produce actionable validation errors.

### 6.3 Observability

- Unify console, dashboard and JSON metrics around one event schema.
- Add TTFT, queue time, prefill, verify, decode, acceptance distribution,
  fallback reason, cache hit and peak memory.
- Export optional Prometheus/OpenTelemetry locally.
- Keep raw prompts and generated text out of telemetry by default.

## 7. Prompt policy, tools and harness

### 7.1 Caveman and Ponytail

- **Current:** `none`, `caveman` and `ponytail` are first-class mutually
  exclusive Mio modes across native agent, CLI, API and Web UI policy state.
- Persist selected mode/level and show it in UI/server status.
- Keep policy text versioned and include its hash in benchmark metadata.
- Evaluate task completion and tool accuracy, not only output-token count.

Acceptance: policy selection is identical across agent, chat and server; exact
external tool-protocol system prompts are not corrupted.

### 7.2 Native tools

- Split filesystem reads, writes and shell execution into permission classes.
- Add workspace roots, command allow/deny policy, timeouts and output limits.
- Make every mutation visible in the transcript and audit log.
- Replace regex-only tool parsing with model-template-aware structured parsing
  while keeping a compatibility fallback.

### 7.3 Coding harness evaluation

- Build a local task corpus with unit-test outcomes, edit correctness, tool
  calls, retries, elapsed time and total generated tokens.
- Compare base target, DFlash target, Caveman and Ponytail separately.
- Include held-out repositories to avoid tuning the harness to Mio itself.
- Report uncertainty and failures; never infer coding quality from tok/s.

## 8. MCP subsystem

- **Landed:** implement stdio and HTTP transports with initialization,
  capability discovery, calls, cancellation and clean shutdown.
- **Landed:** enable local providers by default; remote/auth providers opt-in.
- **Landed:** ship Mio presets for Headroom, local LLM Wiki and Ponytail.
- **Landed:** expose bounded generic discovery/call tools in the native agent
  and Web UI without adding every provider schema to every prompt.
- **Landed:** enforce per-provider timeouts/result bounds and a constrained
  child environment with explicit secret mappings.
- **Landed:** treat MCP bridges as sensitive Web UI orchestration; model
  auto-use needs exact operator and per-request grants, and direct UI runs need
  confirmation.
- Add per-provider tool allow-lists and user confirmation for risky calls.
- Treat tool descriptions/results as untrusted content and delimit them in the
  prompt to reduce tool poisoning.
- Add health/status UI.
- **Landed:** add `mio mcp doctor`, offline `mio mcp check`, and the packaged
  `mio mcp install-tools` installer.
- **Landed:** cover stdio/HTTP lifecycle, timeouts, malformed frames, bounds,
  policy and bridge behavior with fake providers.

Acceptance: Mio starts when an optional provider is unavailable, never passes
undeclared secrets, bounds every result and can disable a provider immediately.

## 9. Skills subsystem

- **Landed:** integrate the requested Hallmark, Matt Pocock, cybersecurity and
  game-studio managed snapshot under `~/.mio/skills`, with pinned provenance
  and an installer-verified expected count of 916 at those revisions.
- **Landed:** support instruction-only Agent Skills; do not require `run.py`.
- **Landed:** expose catalog search/read tools in the native agent and Web UI
  instead of 900+ function schemas.
- **Landed:** resolve collisions with stable aliases and record original
  source/name/revision/digest provenance.
- **Landed:** validate and stage a complete catalog, publish it atomically under
  an inter-process lock, and retain a rollback path.
- **Landed:** treat executable skill scripts as untrusted and disabled unless
  both persistent policy and the individual call grant execution.
- Add catalog refresh, rollback and integrity verification commands/UI.

Acceptance: every requested skill has valid frontmatter and is searchable and
readable by the model; missing/invalid skills are reported, not silently lost.

## 10. LLM Wiki

- **Landed:** provide an offline, path-confined JSON page store with sourced
  ingest, lexical search, list/read/write and provenance/link lint tools over
  Mio's local MCP provider.
- Implement the Karpathy three-layer pattern under `~/.mio/wiki`: immutable
  raw sources, compiled Markdown wiki and explicit schema/instructions.
- Track source provenance and contradictions on every compiled page.
- Add embeddings only when corpus size and retrieval evaluation justify it.

Acceptance: ingest/query/lint work offline, answers can cite local source/page
paths, and link/provenance checks detect broken or orphaned content.

## 11. Headroom

- Install Headroom in an isolated Mio-owned environment, not in the project
  MLX environment and not in Codex.
- Run its local MCP provider by default; keep proxying/compression configurable
  until evaluated against Mio workloads.
- Measure prompt-token savings, retrieval rate, latency, factual/tool parity
  and local resource cost.
- Provide a bypass on any compression/retrieval failure.

Acceptance: a local health check passes, MCP tools initialize from Mio, and no
compression claim is adopted before a controlled Mio evaluation.

## 12. Web UI

### 12.1 Architecture and state

- **Current:** expose one stable `window.Mio` state/API namespace.
- Split the monolithic router into sessions, chat, artifacts, files, skills,
  knowledge, automation and settings routers.
- Split the HTML shell into buildable templates/components while retaining a
  no-cloud production bundle.
- Remove CDN runtime dependencies and add an asset manifest.

### 12.2 Security

- **Landed:** validate Host/Origin/session/CSRF state, confine paths, cap
  uploads, sanitize Markdown/HTML and restrict artifact iframe permissions.
- **Landed:** emit a Content Security Policy and security headers; remove the
  remaining inline compatibility allowances as the UI is modularized.
- **Landed:** fail closed for sensitive Web UI skills and require operator plus
  request grants for model auto-use and per-call confirmation for direct runs.
- Add explicit consent for executable artifacts and continue tightening
  external-content rendering.
- Add stored-XSS and sandbox-escape regression fixtures.

### 12.3 Product coherence

- Define tokens for color, typography, spacing, radius, elevation, focus and
  motion; remove one-off inline styles.
- Use the same empty/loading/error/success patterns in every view.
- Make navigation, settings and model/policy status consistent across desktop
  and mobile.
- Meet keyboard, screen-reader, contrast and reduced-motion requirements.
- Keep artifact/skill counts derived from runtime data, not hard-coded copy.

### 12.4 Missing behavior

- **Landed:** replace the Flow Run placeholder with a serial DAG executor,
  complete node inspector, SSE event stream and persistent flow documents.
- **Landed:** publish flows behind the stable bounded `list_flow_skills` and
  `run_flow_skill` schemas rather than adding one model schema per graph.
- Add retry/backoff and safe intra-flow parallelism where graph dependencies
  permit it; retain the current 200-node/200-hop, 2 MiB graph, 64 KiB argument,
  256 KiB result and 120-second published-run bounds.
- **Landed:** start the scheduler on the live WebUI event loop and shut it down
  through the FastAPI lifespan.
- Complete dashboard data parsing and live error recovery.
- **Landed:** deliver bounded `artifact_emitted` events to the ordinary gallery
  and artifact panel through the stable `window.Mio.artifacts` API.
- Add undo/retry semantics to destructive session/project actions.

Acceptance: browser tests cover all primary views at 320, 375, 414, 768 and
desktop widths with no console errors, broken controls or horizontal overflow.

## 13. Persistence and data integrity

- Centralize `MIO_HOME` resolution instead of repeating `Path.home()/.mio`.
- Add schemas and migrations for every JSON/SQLite store.
- Use file locks plus atomic replace for concurrent writers.
- Add export/import and a documented backup strategy.
- Separate cache/derived files from user-authored durable content.
- Never store tokens or provider secrets in versioned configuration.

Acceptance: simulated interruption cannot truncate a durable store; migration
tests cover every published schema version.

## 14. Packaging and dependency management

- Keep MLX runtime dependencies compatible with the newest verified releases.
- Add a reproducible lock for development/benchmark environments while
  retaining sensible package ranges for library installation.
- Keep the production DSpark dependency version-gated and isolate only
  experimental adapters/harnesses behind the R&D package boundary.
- Package every WebUI asset and validate installed-wheel behavior.
- Provide `mio doctor` for Python, MLX, Metal, model shards, MCP tools, Node
  requirements and writable Mio data directories.

Acceptance: clean editable and wheel installs pass smoke tests on a supported
Apple Silicon host; `pip check`/resolver checks are clean.

## 15. Tests and CI

- Unit: pure configuration, parsers, cache state, policy and graph behavior.
- Integration: fake models/providers, API contract, persistence and scheduler.
- Model: local 4B smoke on every merge candidate; 27B performance gate on
  designated hardware.
- Browser: primary UX, security fixtures, mobile and accessibility.
- Soak: long context, multi-turn prefix reuse, cancellation and repeated model
  load/unload.
- Research: benchmark matrix with parity and raw distributions.

Required merge gate:

1. formatting/lint clean;
2. full non-model tests green;
3. package build/install smoke green;
4. local MCP initialization and skill-catalog integrity green;
5. real 4B UI/API smoke green;
6. Qwen 3.6 27B DSpark selection/fallback and DFlash parity/performance reruns green;
7. documentation and benchmark provenance current;
8. clean working tree and pushed branch.

## 16. Documentation and research paper

- **Current:** update README claims to measured Qwen 3.6 results and label
  historical or external numbers explicitly.
- **Current:** document install, models, prompt policies, MCP, skills, security,
  UI and benchmark reproduction in [15-mcp.md](15-mcp.md) and
  [16-benchmarks.md](16-benchmarks.md).
- **Current:** publish
  [the repository research paper](../papers/mio-qwen36-research.md) covering
  architecture, harnessing, the missing coding-quality experiment,
  prefill/decode methods, ablations, limitations and threats to validity.
- Include negative experiments and distinguish engineering observations from
  scientific evidence.
- Generate PDF only from the versioned paper source.

## 17. Staged delivery order

1. Close prompt policy, MCP, skills and local security checkpoints.
2. Close UI security/stub checkpoint and browser QA.
3. Complete API semantics, cache state placeholders and batch scheduler.
4. Rerun unit, package, model and benchmark gates.
5. Update all docs and publish the paper source/results.
6. Merge to `main` only if every required merge gate is green.
7. Continue isolated DSpark/DFlash/prefill research loops and retained negative ablations.
8. Merge a research result only if it meets the breakthrough criterion; retain
   negative findings in `experimental/` or the paper appendix.
