# Mio on Qwen 3.6: Local Harnessing, Prefill, and Speculative Decode on Apple Silicon

**Mio Project Contributors**

**Research snapshot:** 15 July 2026

**Status:** working engineering report; not peer reviewed; not a final paper

## Abstract

Mio is a local-first inference and agent stack built on Apple's MLX runtime.
This working report records a matched Qwen 3.6 27B experiment in which target
AR, upstream `mlx-dspark` 0.4.1, and upstream `dflash-mlx` 0.1.8 share one
loaded target. Four prompts were each measured three times at 64 output tokens
after one warm-up block, with seeded Latin rotation and 10,000
prompt-clustered bootstrap resamples. All 12 DSpark cap-2 and all 12 upstream
DFlash pairs preserved exact target-token parity and used no fallback.

The cap-2 DSpark path produced a median paired TTFT speedup of 0.755765 (95% CI
[0.647886, 0.803553]), decode speedup of 1.072969 ([1.031231, 1.312283]), and
end-to-end speedup of 1.029057 ([0.992400, 1.187073]). Upstream DFlash produced
0.909765 TTFT ([0.849269, 0.947602]), 2.372774 decode ([1.614444, 2.647570]),
and 2.002901 end to end ([1.474298, 2.206704]). Both candidates fail the
declared workload gate because TTFT regresses; neither result is a
“breakthrough.” A DSpark cap-3 ablation remained parity-safe and raised its
decode point estimate to 1.111549, but its decode and end-to-end lower bounds
fell below 1.0. Cap 4 and the unrestricted block each reached only 75% parity
and were rejected.

A separate fused cold-prefill prototype removed an avoidable singleton target
forward from upstream DFlash's cold path. On the same four prompts, three
repetitions, and 16 output tokens it preserved parity in 12/12 pairs and
measured 1.155496 TTFT ([1.112174, 1.161389]) and 1.079385 end-to-end
([1.065719, 1.093801]) speedups. The projection-only ablation was neutral
(1.004332 TTFT, 0.996899 end to end). The positive fused result remains a
single-thread prototype implemented through a temporary global patch; it has
not been tested with concurrency, prefix snapshots, PQ/TQ caches, or dynamic
tool/EOS policy, and therefore is not promotion evidence.

The same schema-v2 protocol on Qwen3-4B is negative for correctness:
DSpark parity was 75% and DFlash parity 50%, including with DSpark lookup.
Likewise, an isolated mixture-of-drafters replay did not beat the best static
arm: the router achieved 0.9684x of static DSpark at 4B and 0.9589x of static
DFlash at 27B. This supports scale-aware static selection as an engineering
hypothesis, not request-level mixture superiority.

These upstream measurements are intentionally separate from Mio's vendored
exact-verifier diagnostic. That path ran at about 11.6-11.9 tokens/s versus
18.7-19.8 tokens/s for target AR, with roughly 90% of request time in target
verification. It diagnoses a Mio implementation bottleneck; it does not
contradict or substitute for the direct upstream `dflash-mlx` benchmark.
Historical schema-v1 artifacts at commit `d49dec26` are retained only as
historical evidence.

The report also documents Mio's wider harness: model compatibility checks,
local prompt policies, Mio-owned MCP, an on-demand instruction-skill catalog,
an OpenAI-compatible server, and a browser UI. Those capabilities are not
performance or coding-quality evidence. No held-out coding corpus currently
establishes improved correctness, tool accuracy, or semantic quality. This is
a research snapshot with explicit acceptance criteria, not a final paper or a
claim of scientific discovery.

All schema-v2, cap-sweep, mixture, and fused-prefill artifacts discussed here
were generated from revision `9b9bb14` with `git_dirty=true` and were untracked
at the time of this snapshot. They are preliminary local artifacts, not a
reviewed release, an independently replicated result, or a published paper.

## 1. Research questions

This report asks six questions:

1. Can Mio load Qwen 3.6 27B with a locally complete DSpark draft, fall back
   to a separately compatible DFlash draft, and finally degrade to target AR?
2. Under one shared target and a paired protocol, do upstream DSpark and
   DFlash preserve deterministic tokens and improve TTFT, decode, and total
   request time?
3. How does DSpark proposal depth trade speed against exact parity at 27B?
4. Can cold-prefill target work and draft-context projection be reorganized
   without changing the target token stream?
5. Do 4B replication and a mixture-of-drafters router generalize the 27B
   result or instead falsify broader claims?
6. Which claims about Mio's vendored verifier, cache experiments, coding
   harness, MCP, skills, and UI are justified by current evidence?

The paper intentionally separates implemented mechanisms from validated
effects. A feature existing in code is not evidence that it improves speed,
memory, quality, or task success.

## 2. Terminology and control

### 2.1 What “base” means here

The experimental control is the same local quantized target checkpoint,
`Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx`, decoded autoregressively without a
speculative draft or quantized KV cache. We call this **target AR**. The direct
upstream experiment constructs this control through
`dflash_mlx.stream_baseline_generate`; the separate Mio-vendored diagnostic
uses Mio's own target-AR path.

It is not:

- the upstream Qwen 3.6 27B BF16 checkpoint;
- an external serving engine such as vLLM or SGLang;
- a quality comparison between quantization recipes;
- a cloud API baseline;
- a coding-agent benchmark.

The schema-v2 matched experiment compares direct upstream DSpark and DFlash
generation against the direct upstream DFlash package's target baseline while
sharing the exact same loaded target instance. Its ratios are therefore
within-process workload ratios, not comparisons with BF16 Qwen, another
quantization, or another serving engine.

Two other controls must not be conflated with that result. The historical
schema-v1 “1.74x decode” is a Mio measurement at commit `d49dec26`. The newer
Mio-vendored exact-verifier diagnostic observes about 11.6-11.9 tokens/s versus
18.7-19.8 for its target AR control. Neither value describes the direct
upstream `dflash-mlx` arm in the schema-v2 matched experiment.

### 2.2 Prefill, decode, and completion rate

**Prefill** processes prompt tokens and builds model state before generation.
**Decode** emits completion tokens after prefill. The harness reports:

```text
prefill throughput = prompt tokens / prefill seconds
decode throughput  = generated tokens / decode seconds
completion rate    = generated tokens / total request seconds
```

The last value is labeled end-to-end tokens/s in the JSON. Because its
numerator contains only generated tokens, it is not total processed tokens/s.

### 2.3 Exact deterministic token parity

For every candidate labeled exact, correctness is checked by equality of the
complete normalized generated token-ID list with its paired target AR run.
This is stronger than comparing decoded strings and detects
tokenizer-equivalent or hidden whitespace differences. The schema-v2 harness
normalizes upstream final-block overshoot to the configured 64-token budget
while conservatively retaining the full call time. Exactness remains a finite
test, not a proof over every prompt or stochastic distribution.

## 3. Background

### 3.1 MLX on unified memory

MLX provides array execution and Metal acceleration on Apple Silicon. Model
weights, recurrent state, KV caches, draft features, and application memory
share unified memory. This reduces explicit device transfer complexity but
makes peak allocation, application contention, and cache ownership important
engineering constraints.

Dense 27B decode is often dominated by repeatedly reading weights and model
state. A speculative method can improve completion throughput when one target
verification pass validates several draft tokens for less cost than the same
number of AR steps.

### 3.2 DFlash

DFlash uses a lightweight block-diffusion draft conditioned on target hidden
features. The draft proposes a block; the target verifies it; the longest
accepted prefix is committed; and target/draft state is rolled back or
advanced to an aligned absolute position. The central performance variable is
not acceptance alone, but accepted tokens per target verification relative to
draft and rollback overhead.

Mio retains target AR as the operational fallback and experimental control.
This matters because a speculative path that silently falls back may appear
correct while its reported “DFlash” timing no longer measures DFlash.

### 3.3 DSpark

DSpark is a semi-autoregressive drafter with confidence-scheduled speculative
generation. In Mio's Qwen 3.6 integration it is the preferred local drafter
when its checkpoint is complete and compatible. The runtime confines its
mutable target wrapper, drafter context, and prefix-cache state to one worker
thread because moving already-materialized MLX state across execution streams
can fail or silently invalidate timing ownership.

Proposal depth is a correctness parameter as well as a performance parameter.
The 27B cap sweep in this report preserves exact parity at caps 2 and 3 but not
at cap 4 or the unrestricted seven-token block. Mio therefore uses a guarded
Qwen 3.6 profile rather than assuming the upstream maximum is exact for every
target/drafter/quantization combination.

### 3.4 Qwen 3.6 architectural change

The tested target declares 64 layers, hidden size 5120, vocabulary size
248,320, and an effective 2048-token sliding window for relevant attention
layers. The DFlash draft proposes blocks of up to 16 tokens. Compared with
earlier supported Qwen paths, Qwen 3.6 requires correct absolute positions and
causal sliding-attention masks while hybrid recurrent and full-attention state
remain aligned.

### 3.5 Cache quantization

Quantizing KV state can reduce long-context storage and sometimes change
attention cost. It can also introduce logit perturbations, conversion overhead,
larger transient allocations, or incomplete snapshot/rollback behavior. A
cache claim must therefore include output parity or quality, context length,
allocated bytes, prefill cost, decode cost, and speculative compatibility.

## 4. Mio architecture

Mio combines four user surfaces with one model lifecycle:

```text
native coding agent     OpenAI API     batch/bench     Mio UI
         \                 |              |             /
          +----------------+--------------+------------+
                           |
                    request orchestration
                           |
     compaction · policy · tools · skills · MCP registry
                           |
                        MioEngine
                           |
                  drafter/capability plan
           +---------------+----------------+
           |               |                |
      local DSpark    DFlash fallback    target AR
           |               |                |
           +------- DFlash-only DDTree / BMP
                           |
             selected speculative backend
                           |
       target/draft state · prefix reuse · cache format
                           |
                         MLX
```

The implementation-level module map is maintained in
[`docs/12-architecture.md`](../docs/12-architecture.md).

### 4.1 Model discovery and compatibility

The registry maps logical tiers to target/draft repositories and stable local
directories. A local checkpoint is selected only when its configuration and
every indexed SafeTensors shard are present. This prevents a partial download
from being mistaken for a loadable model.

The target loads before the draft. Metadata checks classify a draft as DSpark,
DFlash, or incompatible before binding. For Qwen 3.6, automatic selection uses
DSpark only when its local checkpoint is complete. A compatible, already local
DFlash checkpoint is the next fallback; target-only AR is the final safe
degradation. A strict mode converts drafter load/capability failure into an
error instead of silently changing the measured backend. Metrics expose the
requested, detected, selected, and fallback backend.

`mio pull large` is the provisioning complement to that startup policy: by
default it downloads the target, preferred DSpark checkpoint, and compatible
DFlash fallback. `--no-dspark` skips the DSpark download and `--no-fallback`
skips the DFlash fallback download for that invocation. The flags control
download contents; they do not make an incomplete remote checkpoint an
implicit startup dependency and do not persist strict runtime policy.

### 4.2 Request path

The server normalizes OpenAI messages, optionally compacts context, applies
one prompt policy, renders the native chat template, tokenizes, attempts
prefix-state reuse, selects one cache mode, dispatches generation, normalizes
tool calls, and records metrics. A process-wide lock serializes unsafe MLX
model work at the HTTP boundary.

### 4.3 Prefill changes

Mio's current production code contains two relevant prefill optimizations:

1. target baseline/BMP/DDTree paths project only the final prompt position
   through the LM head when only next-token logits are needed;
2. DFlash projects captured target context in chunks instead of materializing
   a prompt-length-by-vocabulary tensor.

These changes reduce unnecessary work or peak intermediates by construction,
but the checked-in historical matrix does not include a pre-change build. Their
isolated contribution is therefore unknown.

The fused cold-prefill R&D adapter goes further: when there is no reusable
prefix snapshot, it replaces the upstream `prompt_len-1` plus singleton seam
with one full cold target prefill and evaluates the exact draft-context
projection on first use after the native early token event. Its positive pilot
is reported in Section 7.4. The mechanism is not in the production path because
its current temporary global patch is not concurrency-safe and its cache/tool
contracts are not yet certified.

### 4.4 Prefix-state reuse

Mio can retain final target/draft state across requests, find a longest common
token prefix, trim state to the reusable absolute position, and prefill only
the suffix. Entries are rented during mutation and evicted under bounded entry
and token budgets. Quantized and multi-path modes are gated unless their state
contracts support safe reuse.

Historical prefix-cache results exist, but they were not generated by the
current Qwen 3.6 benchmark schema. They are treated as prior engineering
observations rather than current scientific evidence.

### 4.5 Alternative speculative strategies

BMP verifies parallel draft continuations. DDTree verifies a structured
candidate tree with architecture-specific cache/state rules. Both are opt-in
because additional verifier width can cost more than higher acceptance saves.
Historical Mio measurements already contain negative BMP cases. The current
Qwen 3.6 artifact measures neither strategy.

### 4.6 Continuous batch path

Mio separates single-stream latency from multi-session throughput.
`MioEngine.generate_batch` uses MLX-LM continuous batching for two or more
independent prompts: model weights are shared, each session owns its KV cache,
and completed sequences leave the active batch. File/CLI requests are grouped
by temperature because one MLX batch shares a sampler; a group of one retains
the normal single-request latency path and reports the backend actually used.
Supporting this required the speculative target hook to accept vector KV
offsets rather than assuming one scalar cache position.

A real Qwen 3.5 4B smoke generated for prompts `alpha` and `beta` through the
`mlx-continuous` backend in 0.734 seconds. The smoke had no sequential control,
uses a different model from the Qwen 3.6 experiment, and is not stored in the
Qwen 3.6 benchmark JSON. It demonstrates execution and cache separation, not a
speedup. The HTTP `/v1/batch` handler now applies the same tier/temperature
grouping to items inside one bounded request, but it holds a process-wide
Metal lock and does not batch across independent HTTP requests.

## 5. Agent harness and local integrations

### 5.1 Native tools

The native agent provides filesystem inspection/editing, shell execution,
conversation history, and optional validation. These capabilities can improve
the range of tasks a model can attempt, but they also introduce permission,
timeout, transcript, and evaluation requirements.

No current artifact measures whether the agent solves more coding tasks than
the target model in a tool-free prompt. Tool availability must not be equated
with correctness.

### 5.2 Prompt policies

Mio exposes `none`, `caveman`, and `ponytail` modes in agent/chat/server. Caveman
asks for concise communication; Ponytail asks the model to choose the smallest
sufficient engineering solution. Levels are `lite`, `full`, and `ultra`, and
the modes are mutually exclusive.

The policies transform system messages only. They do not grant tools or alter
model weights. Known exact XML tool protocols bypass injection. No Qwen 3.6
corpus currently measures their effect on output tokens, coding success,
security, regressions, or tool-call accuracy.

### 5.3 916 Mio-local instruction skills

The external catalog contains 916 pinned skills:

| Source | Count |
|---|---:|
| Nutlope/hallmark | 1 |
| mattpocock/skills active set | 26 |
| Ruler-Dev/Anthropic-Cybersecurity-Skills | 817 |
| Ruler-Dev/Claude-Code-Game-Studios | 72 |

They install under `~/.mio/skills`, not into Codex or Claude homes. Mio exposes
catalog search and bounded instruction reading rather than 916 independent
function schemas. This is a prompt-headroom design decision: retrieve relevant
instructions only when needed. Reading does not execute repository code;
execution requires a separate explicit trust path.

This architecture is implemented and catalog integrity is testable. Its
effect on coding quality, token usage, latency, and prompt selection accuracy
is not yet measured.

### 5.4 Mio-owned MCP

The registry supports stdio and HTTP/SSE JSON-RPC providers with declared
permissions, timeouts, output limits, constrained child environments, and
explicit secret mappings. Local unauthenticated providers are enabled by
default; remote/authenticated providers are opt-in.

The default declarations are local LLM Wiki, Mio-isolated Headroom, and
read-only Ponytail. The native agent and Web UI expose bounded discovery/call
tools, and enabled unauthenticated local providers receive their declared
permissions. Processes initialize lazily; remote/authenticated providers are
blocked without explicit policy.

An operational, non-versioned smoke discovered and called all three providers
without MCP errors. A direct Headroom compression call on one synthetic
300-line JSON input reported 7,862 to 2,484 tokens, or 5,378 saved (68.4%), via
`smart_crusher`, even though the optional HTTP proxy on port 8787 was not
running. This shows the direct MCP path executed on that input. It is not part
of the Qwen 3.6 artifacts, has no repetitions or quality control, and does not
support a general Headroom compression, latency, retrieval, or task-success
claim.

### 5.5 Browser UI and local security

Mio UI surfaces chat, artifacts, built-in tools, external skill discovery,
projects, knowledge, flows, schedules, and dashboards. The current integration
adds loopback binding/CORS, identifier/path validation, upload bounds,
sanitization, sandboxed artifact frames, and a shared client state namespace.

Flow Mode now executes saved DAGs server-side and streams per-node events. A
complete inspector edits the shipped node types, and persistent publication
exposes any selected graph through two stable bounded tools,
`list_flow_skills` and `run_flow_skill`. Limits are 200 graph nodes and 200
execution hops, a 2 MiB graph file, 64 KiB arguments, 256 KiB results, and a
120-second published-run timeout; recursive flow-skill dispatch is rejected.
Execution within a graph remains serial and lacks retry/backoff.

The UI is part of the harness, not part of the Qwen performance experiment.
Browser coherence, accessibility, content-security policy, dangerous-action
confirmation, and stored-XSS coverage require separate acceptance tests.

## 6. Experimental method and evidence classes

### 6.1 Hardware and software

The schema-v2 27B experiment and fused-prefill pilot were obtained on:

| Component | Value |
|---|---|
| system | MacBook Pro, Apple M4 Max, 16 CPU cores |
| unified memory | 48 GB |
| OS | macOS 26.5.1 arm64 |
| Python | 3.12.0 |
| MLX | 0.32.0 |
| mlx-lm | 0.31.3 |
| mlx-dspark | 0.4.1 |
| dflash-mlx | 0.1.8 |
| target | Qwen3.6-27B-UD-Q4_K_XL-mlx |
| DSpark draft | Qwen3.6-27B-DSpark, block 7, W4 group 64 |
| DFlash draft | Qwen3.6-27B-DFlash, W4 group 64 |

The new JSON provenance records `Apple M4 Max`, 51,539,607,552 bytes of unified
memory, the MLX device identifier, package versions, and SHA-256 hashes of all
three model configurations. The older schema-v1 artifacts record the platform
but not the same complete hardware provenance.

### 6.2 Evidence status and historical commit

This report keeps six evidence classes separate:

1. **Versioned historical artifacts.** The two schema-v1 JSON files support
   claims only about their recorded Mio implementation and workload.
2. **Mio-vendored diagnostics.** Narrow 27B checks exercise Mio's corrected
   exact verifier and phase telemetry; they are not the upstream benchmark.
3. **Preliminary schema-v2 matched artifacts.** The 27B baseline/DSpark/DFlash
   matrix and cap sweep have paired prompt-cluster intervals, exact-token
   checks, and fallback checks, but were produced from a dirty tree and have
   not been reviewed or independently replicated.
4. **Isolated R&D prototypes.** Fused cold-prefill and mixture routing live
   outside the production runtime and carry narrower validity contracts.
5. **Negative replication.** The Qwen3-4B v0.4.1 artifacts fail exact parity
   and therefore cannot support performance promotion even when speed ratios
   look favorable.
6. **Non-evidence coding smoke.** A clean two-pair Qwen 3.6 27B Verified run
   validates the paired generation/evaluation chain, but is far below the
   preregistered 500-pair quality study and cannot support promotion.

The versioned historical artifacts identify:

```text
git revision: d49dec26dbd6053526027e013d5580e9cf5c10f4
git dirty:    false
```

This commit contains the last-position baseline projection and the historical
Qwen 3.6 matrix harness. Later integration work does not retroactively change
these numbers, and the artifact does not describe the later exact verifier.

The new artifacts instead identify:

```text
git revision: 9b9bb142f97958f720e26f29233b27c5d2f06978
git dirty:    true
tracking:     untracked at the research snapshot
```

Their intervals quantify the measured local workload but do not convert a
dirty, single-machine experiment into release or publication evidence.

### 6.3 Historical prompt and sampling

The harness repeats a fixed software-engineering sentence until tokenization
produces exactly 256 target tokens. Each mode generates 32 tokens. It performs
one warm-up and two measured repetitions. The token control is deterministic.

### 6.4 Historical modes

Core artifact:

- target AR, no quantized KV cache;
- DFlash, no quantized KV cache.

Cache artifact:

- target AR, no quantized KV cache;
- DFlash + PQ4;
- DFlash + TQ4.

The cache artifact does not include unquantized DFlash, so it is not a clean
cache-only factorial design.

### 6.5 Historical metrics and aggregation

Each repetition records elapsed, prefill, derived decode time, prompt and
generation tokens, throughput values, acceptance, tokens/cycle, peak MLX
memory, fallback state, token IDs, and SHA-256 token hash. Tables report the
median of two measured repetitions.

No confidence interval or hypothesis test is meaningful with two
repetitions. Ratios are descriptive.

### 6.6 Schema-v2 paired protocol

The current harness emits schema v2. Every measured repetition contains every
requested mode. Seeded randomized Latin-rotation blocks balance execution
position, and both warm-up and measured order are persisted. A run can
reproduce the schedule with `--seed`.

Each repetition stores a paired candidate/baseline throughput ratio and
normalized timings for:

```text
prefill · draft · draft_prefill · draft_incremental
verify · replay · rebuild · commit
```

Schema v2 also records cache commit mode, rebuilt target-token count, exact
acceptance-correction count, p95 TTFT/decode latency, execution peak memory,
fallback, model hashes, and baseline determinism. Every exact mode must match
its paired target AR token list, and strict mode exits nonzero on execution,
timing, determinism, fallback, length, or parity failure.

The estimand for TTFT, decode, and end-to-end speed is the median paired
baseline/candidate ratio. Confidence intervals use 10,000 cluster-bootstrap
resamples: one sampled unit is a prompt and always carries all three of its
repetitions. Repetitions are not treated as independent observations. The
candidate gate additionally requires p95 TTFT/decode non-regression, 100%
parity, zero fallbacks, enough pairs and prompt clusters, deterministic
baseline output, and p95 peak-memory non-regression when complete positive
measurements exist.

### 6.7 Matched 27B protocol and cap sweep

The primary artifact uses four built-in prompts: Python refactoring,
concurrency debugging, structured JSON, and copy-heavy continuation. Chat
templating produces prompt lengths of 71, 49, 40, and 57 tokens. Each prompt
receives one warm-up block and three measured blocks; every measured block
contains target AR, DSpark, and DFlash in a seeded Latin rotation. All arms
share one target object and generate a normalized budget of 64 tokens.

The main DSpark arm uses maximum proposal depth 2 with lookup disabled. Three
additional full-matrix artifacts change only that depth to 3, 4, or 0
(unrestricted block). Direct upstream DFlash uses its block-16 drafter with
`dflash` verification. Both drafts use W4 group-64 quantization. Upstream
final-block overshoot is truncated to exactly 64 token IDs for parity and the
throughput numerator, while wall time conservatively retains the overshoot.

### 6.8 Qwen3-4B replication protocol

The 4B v0.4.1 runs use the same four-prompt, three-repetition, 64-token,
10,000-resample schema-v2 protocol with a matched 8-bit target, block-7 DSpark
draft, and block-16 DFlash draft. A second artifact adds DSpark lookup as a
separately labeled candidate rather than changing the pure DSpark arm. These
runs are correctness falsification evidence: strict mode failed because
multiple prompt clusters diverged from target AR.

### 6.9 Fused cold-prefill and mixture protocols

The fused-prefill pilot compares standard upstream DFlash cold prefill with an
isolated adapter that performs one full target prefill and defers exact
draft-context projection until first access. It uses the same four prompts,
three repetitions, balanced arm order, 10,000 prompt-cluster resamples, and 16
output tokens. It excludes prefix snapshots and runs through a temporary
single-process global patch.

The mixture experiment is an offline replay over already measured 4B and 27B
matrices. Each online decision sees only the selected arm's outcome; the full
matrix is revealed afterward solely to compute static and oracle comparators.
The router pays one calibration selection per arm, then uses cost curves,
hysteresis, and a regression guard. Because calibration and evaluation reuse
the same prompt set, this is a falsification/debugging replay, not held-out
router evidence.

## 7. Results

### 7.1 Matched Qwen 3.6 27B result

All 24 candidate/baseline pairs in the main artifact were complete, timing
valid, deterministic on the baseline, free of fallback, and parity-safe. A
speedup ratio above 1.0 favors the candidate.

| 27B arm | TTFT speedup (95% CI) | Decode speedup (95% CI) | End-to-end speedup (95% CI) | Exact parity |
|---|---:|---:|---:|---:|
| target AR | 1.000000 | 1.000000 | 1.000000 | control |
| DSpark, cap 2 | 0.755765 [0.647886, 0.803553] | 1.072969 [1.031231, 1.312283] | 1.029057 [0.992400, 1.187073] | 12/12 |
| upstream DFlash | 0.909765 [0.849269, 0.947602] | 2.372774 [1.614444, 2.647570] | 2.002901 [1.474298, 2.206704] | 12/12 |

The direct upstream DFlash arm has a large, confidence-bounded decode and
end-to-end gain on this workload. Its TTFT nevertheless regresses by about 9%
at the point estimate and its entire TTFT interval is below 1.0. The DSpark
cap-2 decode lower bound is above 1.0, but its TTFT is substantially worse and
its end-to-end interval crosses 1.0. Both also fail the configured p95 TTFT
non-regression gate. Therefore `research_claim.any_workload_candidate` is
false for the artifact even before asking whether a global breakthrough is
evaluable.

This distinction is central: “DFlash decoded 2.37x faster on the measured
27B workload” is supported as a scoped direct-upstream observation. “Mio is
2.37x faster,” “prefill is faster,” and “a breakthrough was discovered” are
not supported.

### 7.2 DSpark proposal-depth sweep

| Maximum draft tokens | TTFT | Decode | End to end | Parity | Strict result |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.755765 [0.647886, 0.803553] | 1.072969 [1.031231, 1.312283] | 1.029057 [0.992400, 1.187073] | 1.00 | pass |
| 3 | 0.718655 [0.709748, 0.780929] | 1.111549 [0.966516, 1.301636] | 1.058017 [0.937663, 1.213975] | 1.00 | pass |
| 4 | 0.711030 | 0.964488 | 0.936517 | 0.75 | fail |
| unrestricted block | 0.718345 | 0.684193 | 0.691023 | 0.75 | fail |

Cap 3 is the largest tested parity-safe setting and has the strongest DSpark
decode and end-to-end point estimates. Its intervals do not establish a
decode or total-request gain, and TTFT remains a large regression. Mio uses
cap 3 with lookup disabled as a guarded Qwen 3.6 runtime profile, not as a
scientifically promoted speed claim. Cap 4 and unrestricted generation each
diverged on all three `python-refactor` repetitions, so their timings are
reported only as rejected ablations.

### 7.3 Direct upstream DFlash versus Mio's vendored verifier

The schema-v2 DFlash arm calls upstream `dflash-mlx` 0.1.8 directly. Mio's
vendored exact-verifier diagnostic is a different implementation and corpus:

| Measurement | Target AR | DFlash candidate | Interpretation |
|---|---:|---:|---|
| direct upstream matched matrix | ratio control | decode 2.372774x; E2E 2.002901x | fast decode, TTFT regression |
| Mio-vendored exact diagnostic | about 18.7-19.8 tok/s | about 11.6-11.9 tok/s | candidate is roughly 36-41% slower |

In the Mio diagnostic, target verification consumed about 5.06 seconds of a
roughly 5.65-second request, or approximately 90%. Drafting, replay, rebuild,
and commit were secondary. That bottleneck explains where the vendored path
must improve; it must not be averaged with or used to discount the upstream
measurement. The two implementations were not run as one paired randomized
matrix.

MLX arrays are lazily evaluated, so posterior logits, hidden state, replayed
recurrent state, and rebuilt cache state must be materialized inside the phase
being timed. Otherwise work can migrate to a later synchronization and make
the verifier appear cheaper than the real request.

### 7.4 Fused cold-prefill pilot

| Candidate versus standard upstream DFlash | Point estimate | 95% prompt-cluster CI | Parity |
|---|---:|---:|---:|
| TTFT speedup | 1.155496 | [1.112174, 1.161389] | 12/12 |
| end-to-end speedup | 1.079385 | [1.065719, 1.093801] | 12/12 |
| projection-only TTFT ablation | 1.004332 | not promotion-tested | 12/12 |
| projection-only E2E ablation | 0.996899 | not promotion-tested | 12/12 |

Every fused pair was above 1.0 for both TTFT and end-to-end time. The
projection-only ablation shows that deferred projection is not the source of
the gain; the useful mechanism is removing the avoidable singleton full-model
forward and synchronization boundary from cold prefill.

This is the strongest positive prefill hypothesis in the current snapshot,
but its scope is deliberately narrow: prompt lengths are 40-71 tokens,
generation stops at 16 tokens, there is one machine and one model pair, the
tree was dirty, and the prototype patches a process-global upstream function.
It is single-thread research code with no concurrency, prefix-snapshot,
PQ/TQ, dynamic tool-required/EOS, cancellation, sustained-load, or long-prompt
certification. It is not enabled in production and is not a breakthrough.

### 7.5 Qwen3-4B v0.4.1 replication is negative

| 4B arm | TTFT point estimate | Decode point estimate | E2E point estimate | Parity |
|---|---:|---:|---:|---:|
| DSpark cap 2, lookup off | 0.433818 | 1.668989 | 1.413527 | **0.75** |
| upstream DFlash | 0.828232 | 1.018143 | 1.003655 | **0.50** |

DSpark diverged on all three `python-refactor` repetitions. DFlash diverged on
all `python-refactor` and all `structured-json` repetitions. The separate
lookup artifact retains the same 75% DSpark and 50% DFlash parity; lookup does
not repair correctness. Strict mode therefore fails both 4B artifacts. Their
speed ratios are useful for diagnosis but invalid for promotion because a
faster different token stream is not exact speculative acceleration.

This supersedes the earlier exploratory 4B narrative. The current v0.4.1
four-prompt artifact is the applicable 4B result.

### 7.6 Mixture-of-drafters replay

| Replay scale | Best static arm | Static-arm advantage | Router / best static | Router parity |
|---|---|---:|---:|---:|
| Qwen3-4B | DSpark | 1.4257x over DFlash | 0.9684x | 0.75 |
| Qwen 3.6 27B | DFlash | 1.7301x over DSpark | 0.9589x | 1.00 |

At 4B, static DSpark takes 6.5138 seconds versus 9.2867 for static DFlash. The
router pays one DFlash calibration request and finishes in 6.7263 seconds, a
3.16% regression. At 27B, static DFlash takes 26.0635 seconds versus 45.0937
for static DSpark. The router pays one DSpark calibration request and finishes
in 27.1818 seconds, a 4.11% regression.

No prompt-level mixture opportunity appears in either measured corpus: the
same arm wins every request within one scale. The scale reversal is useful
evidence for static scale-aware selection, which can avoid online calibration.
It is not evidence that a request-level mixture beats the best drafter, and
the same-corpus replay is not held-out evaluation.

### 7.7 Historical schema-v1 results

The versioned commit-`d49dec26` core artifact remains historical context:

| Historical mode | Prefill tok/s | Decode tok/s | Completion tok/s | Acceptance | Tokens/cycle | Parity |
|---|---:|---:|---:|---:|---:|---:|
| target AR | 234.77 | 19.31 | 11.64 | 0 | 0 | control |
| DFlash | 232.92 | 33.64 | 15.61 | 0.8125 | 5.33 | yes |

Its descriptive DFlash/target ratios are 0.992x prefill, 1.743x decode, and
1.340x total completion. It has only two measured repetitions and predates
the current exact verifier, direct-upstream adapter, and prompt-cluster
protocol. The 1.743x value is reproducible for that artifact but cannot be
used as the current Mio or upstream result.

The historical cache artifact is also negative/mixed. DFlash+PQ4 measured
1.269x decode but changed deterministic tokens. DFlash+TQ4 preserved tokens,
measured only 1.043x decode, regressed prefill to 0.785x, and regressed total
completion to 0.923x. The 256-token run did not demonstrate a peak-memory
reduction. Neither cache mode is evidence of a universal speed or memory gain.

### 7.8 Metal weight-staging ablation

The cooperative threadgroup weight-staging QMV implementation is exact in its
exercised 4/5/6/8-bit, vector-width, and tail checks. On real Qwen 3.6 27B
projections it increased T16 kernel time by 9.3% for `up`, 11.8% for `down`,
and 13.3% for `qkv`; T5 regressions were approximately 12-20%.

The ablation remains available behind `MIO_DFLASH_QMV_STAGING=1` and is
disabled by default. Extra cooperative loads and synchronization did not pay
for the exercised shapes, so it is retained as negative evidence rather than
selected as an optimization.

### 7.9 Synthesis of positive and negative evidence

The current evidence supports four narrow statements:

1. Direct upstream DFlash materially accelerates 27B decode and total request
   time on the measured four-prompt workload while regressing TTFT.
2. DSpark caps 2 and 3 preserve 27B parity; cap 2 has a positive
   confidence-bounded decode gain, while neither cap passes the TTFT/total
   promotion gate.
3. Fusing the cold target prefill is a promising parity-safe prefill mechanism
   in an isolated short-request pilot.
4. The best drafter reverses by scale in replay, supporting static scale-aware
   policy as a hypothesis.

It also establishes important negatives: cap 4/full DSpark and both 4B
speculative paths fail parity; online mixture loses to a static arm; Mio's
vendored verifier is slower than its target control; weight staging regresses;
PQ4 changes tokens; and TQ4 loses end to end. None of the artifacts establishes
coding-quality improvement, universal MLX superiority, or a breakthrough.

## 8. What changed relative to earlier Mio documentation

Earlier project copy included figures such as 4.1x DFlash, 10-20x task time,
fixed Caveman token-reduction percentages, and universal cache-speed/memory
claims. Those figures were not supported by controlled Qwen 3.6 evidence and
have been removed from the canonical README. The checked-in 1.74x schema-v1
ratio is now labeled historical so it is not confused with either the slower
Mio-vendored exact diagnostic or the faster direct-upstream schema-v2 DFlash
arm. The newer 2.372774x decode result is itself scoped to one dirty-tree,
single-machine, four-prompt workload and is not substituted for the old
universal claims.

Historical Qwen 3/3.5 experiment notes remain in the repository with explicit
status banners. They may motivate hypotheses, but their models, prompts,
commits, and metrics must not be pooled with this result.

## 9. Coding and harnessing evaluation

### 9.1 What exact parity implies

The primary 27B matrix produced exactly the same 64 normalized token IDs as
its paired target AR control in all 12 DSpark cap-2 and all 12 upstream DFlash
runs. The DSpark cap-3 matrix also passed 12/12. On those finite trajectories,
speculative execution changed latency, not the response. Cap 4, unrestricted
DSpark, and the 4B candidates show why that property must be measured rather
than assumed.

It does not establish that Mio's agent solves coding tasks better. An
identical token stream has the same semantic content for that prompt whether
the exact engine is faster or slower, while the broader harness may change
prompts, tools, retries, context, and outputs.

### 9.2 Initial quality-gate smoke and missing full study

Mio completed a counterbalanced two-pair SWE-bench Verified smoke with the
target-only Qwen 3.6 27B backend. All four arms reached their 12-round,
11-tool-call terminal budget. Plain produced two empty predictions. Quality
produced one empty prediction and one 716-character Matplotlib patch. The
pinned official harness resolved 0/2 for Plain and 0/2 for Quality; the
non-empty Quality patch was not correct enough to resolve the issue.

Quality used 665.001 aggregate wall seconds versus 772.876 for Plain, but the
pair-level ratios pointed in opposite directions (1.1425 Django, 0.7074
Matplotlib) and the dynamic prompt streams differed. With only two pairs and
zero resolution difference, this is neither coding-quality nor speed evidence.
It is a useful null result: enforcing the current gate made the agent produce a
patch where Plain did not, but did not make that patch successful. The
source-free artifact is
[`benchmarks/results/swebench-quality-27b-smoke-0fd8389.json`](../benchmarks/results/swebench-quality-27b-smoke-0fd8389.json).

Post-smoke trace analysis found that this intervention had two avoidable
procedural failure modes. A requested change with no workspace mutation was
treated as observational and therefore satisfied, while a late edit left only
one tool-enabled round; the model selected Bash for a test-like command, and
the reserved synthesis round made trusted validation impossible. Quality Gate
v2 consequently requires a net revision delta under a trusted change contract,
rejects identical writes and edit-revert trajectories, prioritizes `validate`
over Bash, and makes the final bounded round a restricted recovery round. It
keeps the 12-round ceiling and does not modify MLX inference kernels. Thus its
expected cost, if any, is agent-level prompt/tool wall time rather than an
intrinsic change in prefill or decode tok/s. These design corrections are not
retroactive evidence: the v2 policy must pass paired MioCodeBench calibration
and internal holdout cost gates before another 27B smoke, and only the frozen
500-pair Verified experiment can support a quality claim.

A proper coding study should compare at least:

1. target AR with tool-free prompts;
2. DSpark and DFlash with identical prompts and no policy;
3. native agent tools without an added prompt policy;
4. Caveman at each level;
5. Ponytail at each level;
6. instruction-skill retrieval on/off;
7. Headroom compression/retrieval on/off;
8. effort/quality policy on/off, independently of the decode backend.

Tasks should come from held-out repositories and record:

- tests passed and patch correctness;
- files/lines changed and unnecessary churn;
- tool-call validity and correct arguments;
- retries, recovery from failed tools, and context loss;
- wall time, TTFT, generated tokens, and total prompt tokens;
- security-policy violations and data-loss incidents;
- human review for maintainability where automated tests are insufficient.

Until the full held-out corpus exists, the paper reports harness capabilities
and this null smoke, not coding improvement over the base control.

### 9.3 Harness-token accounting

The 916-skill catalog is intentionally searched/read on demand. A future
evaluation should count:

- catalog-search tokens;
- loaded instruction tokens;
- avoided always-on schema tokens;
- selection precision/recall for relevant skills;
- task outcome with and without the selected instruction;
- latency and failure introduced by retrieval.

Similarly, Headroom must be evaluated for compression ratio, retrieval
fidelity, added latency, missed facts, tool accuracy, and resource cost. Token
reduction alone is not sufficient if it removes evidence needed for a correct
edit. The single 68.4% synthetic-JSON smoke in Section 5.4 is an integration
observation, not this evaluation.

## 10. Threats to validity

### 10.1 Statistical validity

The primary matrix has 12 pairs per candidate but only four independent prompt
clusters. Its 10,000 bootstrap resamples preserve within-prompt repetition and
therefore avoid the worst pseudo-replication, but four clusters still produce
fragile intervals and cannot characterize broad prompt variation, thermal
drift, or rare tails. Cap-sweep artifacts reuse the same corpus. Historical
schema-v1 artifacts have only two measured repetitions. The fused pilot has
the same four clusters at a shorter output length.

### 10.2 Workload validity

The built-in corpus spans refactoring, debugging, structured JSON, and a
copy-heavy continuation, but it is authored for this harness and is not a
held-out coding benchmark. Prompts are only 40-71 tokens and outputs 64 tokens;
the fused pilot stops at 16. It does not cover long context, multilingual text,
real repository state, dynamic tool calls, adversarial prompts, or production
conversation histories. Acceptance and verifier efficiency are content
dependent.

### 10.3 Hardware validity

The result comes from one M4 Max 48 GB system. Memory bandwidth, power mode,
thermal state, OS/Metal version, and concurrent applications can change
throughput. The new schema records the device and memory, but no independent
machine or implementation has replicated the result.

### 10.4 Measurement validity

TTFT is an external-call-to-first-output proxy: upstream DFlash exposes a
native token event, while DSpark reports its first text callback. Decode time
is wall time minus that proxy. Upstream final-block overshoot is removed from
token IDs and the throughput numerator but retained in wall time. Peak MLX
memory is measured with all three models resident in one process, so it
compares execution peaks rather than standalone deployment footprint. Graph
compilation, lazy synchronization, tokenizer work, and queueing require
further isolation.

### 10.5 Control validity

The schema-v2 control is target AR from the direct upstream DFlash package,
using the same quantized MLX target instance as both candidate arms. It does
not measure upstream BF16 Qwen, another quantization, vLLM, SGLang, or Mio's
production HTTP path. Mio's vendored verifier and historical cache modes are
separate controls and cannot be pooled with it. The 4B target/drafters are a
different model family/scale combination and do not validate 27B by transfer.

### 10.6 Correctness validity

Exact first-64 tokens provide a strong finite control for caps 2/3 and direct
upstream DFlash at 27B. Cap 4/full and the 4B artifacts demonstrate finite
divergence. Normalization is necessary because upstream speculative functions
may overshoot the requested final block. The checks do not test stochastic
distributions, long-run numerical drift, semantic quality, dynamic EOS/tool
protocols, or adversarial prompts.

### 10.7 Integration validity

The schema-v2 matrices and prototypes were generated from a dirty revision and
were untracked at this snapshot. Production auto-selection, DSpark worker
ownership, bounded streaming/cancellation, local fallback, `mio pull`, MCP,
skills, prompt policies, UI security, and flows have independent tests; none
is exercised by the direct-upstream performance matrix. Release gates and a
clean rerun are required before merge or publication.

## 11. Research roadmap

### 11.1 Prefill experiments

Run 256, 2K, 8K, 32K, and maximum-safe prompts. Instrument:

- tokenization;
- target forward by chunk;
- final-position versus full LM-head projection;
- draft-context capture/projection;
- graph compilation and synchronization;
- TTFT and peak transient allocation.

Promote the fused cold-prefill mechanism from a temporary patch to an explicit
runtime context only after concurrency, cancellation, prefix snapshot
restore/publish, PQ/TQ, tool-required/EOS, and long-prompt contracts pass.
Repeat at 64 and 256 output tokens and on another Apple Silicon machine.

A prefill change is promoted only when exact output is retained, peak memory
and tail latency do not regress, its point estimate exceeds +5%, and its paired
bootstrap lower bound exceeds 1.0 at multiple lengths.

### 11.2 Decode experiments

Profile and reduce the exact verifier cost before tuning acceptance alone.
Evaluate proposal length as a guarded, parity-gated control based on acceptance,
verifier cost, and memory; cap 4/full remain forbidden for the current 27B pair.
Compare direct upstream DFlash, the Mio verifier, DSpark, BMP, and DDTree on
code, prose, JSON, and tool calls. Record p1/p50/p95 rather than only medians
and eliminate hidden fallbacks. Keep the slower weight-staging path as a
negative control.

### 11.3 Cache experiments

Add unquantized DFlash to the same randomized cache run. Measure cache bytes
per layer and prompt length, trim/restore/rollback round trips, deterministic
drift length, semantic quality, and long-context allocation. Profile why TQ4
prefill and transient memory regressed.

### 11.4 Service experiments

Use the landed MLX continuous batch path and the same-request HTTP grouping as
the implementation baseline. Add a bounded cross-request scheduler before
claiming service throughput. Measure sequential versus continuous at
concurrency 1/2/4/8, queue time, cancellation, fairness, interactive p95, and
cache isolation.

### 11.5 Coding harness

Build the held-out corpus described in Section 9. Separate engine acceleration
from policy/harness effects. Exact DSpark/DFlash candidates must match the
target under deterministic decoding; Caveman, Ponytail, skills, Headroom, and
future effort/quality policies need independent quality and token ablations.

## 12. DSpark research protocol

The matched 27B study establishes a bounded DSpark operating region for the
tested pair. Caps 2 and 3 preserve all 12 paired 64-token trajectories; cap 2
has a 1.072969 decode speedup with a confidence lower bound of 1.031231, while
cap 3 has a 1.111549 point estimate whose lower bound is 0.966516. Both regress
TTFT. Cap 4 and the unrestricted block fail parity at 75%.

The 4B v0.4.1 replication is a correctness failure, not positive transfer:
DSpark reaches 75% parity and DFlash 50%. Mixture replay also fails to beat the
best static arm at either scale. Production therefore uses metadata/local
completeness to choose a guarded static DSpark profile, a separately compatible
local DFlash fallback, or target-only AR. It does not deploy the R&D router.

The next sequence is:

1. repeat the exact artifacts from a clean, committed revision;
2. freeze a target AR/direct-upstream DFlash/DSpark/Mio-verifier control matrix;
3. validate architecture, tokenizer, target training, quantization, and
   proposal-depth compatibility before each run;
4. enforce the requested output budget while separately recording overshoot;
5. require token parity and zero fallback for every exact mode;
6. measure TTFT/prefill, decode, p95 latency, and memory in seeded Latin blocks;
7. certify the fused cold-prefill mechanism under concurrency and cache/tool
   contracts;
8. run a disjoint held-out corpus and retain prompt-cluster bootstrap intervals;
9. replicate independently on matched 4B and 27B pairs and another machine;
10. publish incompatible, neutral, or slower results as negative evidence.

A drafter that merely loads is not necessarily compatible. The cap and 4B
failures show that model identity, proposal depth, quantization, lookup, and
runtime version must be part of the compatibility contract.

### 12.1 Breakthrough criterion

Mio will use “breakthrough” only when one candidate satisfies every gate:

1. 100% deterministic token parity on exact modes and zero fallback;
2. point estimates of at least +5% for both TTFT/prefill and decode;
3. paired-bootstrap lower confidence bounds greater than 1.0 for both gains;
4. no material peak-memory, tail-latency, reliability, or quality regression;
5. success on a held-out corpus spanning code, prose, structured output, tool
   calls, and multiple prompt/output lengths;
6. independent replication on matched 4B and 27B target/draft pairs.

Direct upstream DFlash passes the 27B decode/total interval gates but fails
TTFT. DSpark fails TTFT and total-request gates; cap 4/full and the 4B artifacts
also fail parity. The fused-prefill pilot passes its narrow short-request
intervals but lacks the scope, runtime safety, and independent replication
required above. Repeated search without all six conditions is experimentation,
not discovery.

## 13. Reproducibility

Provision the complete three-model local stack with:

```bash
mio pull large
```

The command downloads the target, preferred DSpark draft, and compatible
DFlash fallback by default. `mio pull large --no-dspark` and
`mio pull large --no-fallback` intentionally omit the named download.

Reproduce the primary cap-2 matched matrix from a clean revision with:

```bash
python3 scripts/bench_speculative_matched.py \
  --model models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --dspark-draft spd/Qwen3.6-27B-DSpark \
  --dflash-draft spd/Qwen3.6-27B-DFlash \
  --max-tokens 64 --warmup 1 --reps 3 \
  --seed 20260715 --bootstrap-samples 10000 \
  --dspark-max-draft-tokens 2 --no-dspark-lookup \
  --strict \
  --output benchmarks/results/speculative-matched-qwen36-27b-rerun.json
```

Repeat the largest parity-safe DSpark cap as a separate artifact:

```bash
python3 scripts/bench_speculative_matched.py \
  --model models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --dspark-draft spd/Qwen3.6-27B-DSpark \
  --dflash-draft spd/Qwen3.6-27B-DFlash \
  --max-tokens 64 --warmup 1 --reps 3 \
  --seed 20260715 --bootstrap-samples 10000 \
  --dspark-max-draft-tokens 3 --no-dspark-lookup \
  --strict \
  --output benchmarks/results/speculative-matched-qwen36-27b-cap3-rerun.json
```

The fused cold-prefill pilot and mixture replays are isolated experiments:

```bash
python3 -m experimental.upstream_dflash.bench_deferred_priming \
  --model models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --draft spd/Qwen3.6-27B-DFlash \
  --max-tokens 16 --repetitions 3 \
  --seed 20260715 --bootstrap-samples 10000 \
  > experimental/upstream_dflash/results/qwen36-27b-fused-rerun.json

python3 -m experimental.mixture.replay \
  benchmarks/results/speculative-matched-qwen3-4b-20260715-v041.json \
  --pretty

python3 -m experimental.mixture.replay \
  benchmarks/results/speculative-matched-qwen36-27b-20260715.json \
  --pretty
```

The versioned historical schema-v1 artifacts were generated at their recorded
commit with:

```bash
python3 scripts/bench_qwen36_matrix.py \
  --tier large --prompt-tokens 256 --max-tokens 32 \
  --warmup 1 --reps 2 --modes baseline,dflash \
  --output benchmarks/results/qwen36-core-256.json

python3 scripts/bench_qwen36_matrix.py \
  --tier large --prompt-tokens 256 --max-tokens 32 \
  --warmup 1 --reps 2 --modes baseline,pq4,tq4 \
  --output benchmarks/results/qwen36-cache-256.json
```

Do not overwrite any source artifact. Before interpreting a rerun, verify a
clean tree, package versions, complete indexed model shards, exact-mode parity,
baseline determinism, and absence of fallback. The expanded protocol and gates
are in
[`docs/16-benchmarks.md`](../docs/16-benchmarks.md).

## 14. Conclusion

The matched Qwen 3.6 27B matrix is the strongest current decode evidence.
Direct upstream DFlash preserves all 12 paired target trajectories and measures
2.372774x decode and 2.002901x end-to-end speedups with confidence lower bounds
above 1.0. It also regresses TTFT to 0.909765x, so the complete promotion gate
fails. DSpark cap 2 preserves 12/12 and has a smaller confidence-bounded decode
gain, but substantially worse TTFT and an inconclusive total-request interval.
Cap 3 is the largest parity-safe tested profile; cap 4 and the unrestricted
block are rejected at 75% parity.

The fused cold-prefill pilot identifies a plausible new mechanism: remove one
avoidable full-model singleton seam and defer exact draft-context projection.
It measures 1.155496x TTFT and 1.079385x end to end with 12/12 parity and lower
bounds above 1.0, whereas projection deferral alone is neutral. Its temporary
global patch, 16-token outputs, short prompts, dirty tree, and missing
concurrency/cache/tool certification prevent production promotion.

Negative evidence constrains the next design. Qwen3-4B v0.4.1 fails parity for
both candidates; online mixture loses to the best static arm at both scales;
Mio's vendored exact verifier is slower than its target control; cooperative
weight staging regresses; historical PQ4 changes tokens; and TQ4 is slower end
to end. The apparent reversal between static DSpark at 4B and static DFlash at
27B motivates scale-aware static selection, not a mixture breakthrough.

Production Mio now has the operational skeleton required to keep experiments
honest: use a complete local DSpark checkpoint when selected, fall back to an
independently compatible local DFlash checkpoint, then degrade to target-only
AR with observable telemetry. `mio pull large` provisions target, DSpark, and
DFlash unless an explicit download flag omits one. This reliability work is
not itself performance evidence.

Mio's wider contribution remains a testable local harness around the engine:
model validation, explicit fallback, prompt-policy separation, on-demand
instruction skills, permission-gated local MCP, and multiple user surfaces.
The first clean paired Verified smoke is null at 0/2 versus 0/2, so no current
experiment shows that these layers improve coding-task quality over target AR.
The report ends with a falsifiable six-gate protocol and a clean full-500
requirement, not a declaration of breakthrough.

## Appendix A. Raw result pointers

- [`benchmarks/results/speculative-matched-qwen36-27b-20260715.json`](../benchmarks/results/speculative-matched-qwen36-27b-20260715.json)
- [`benchmarks/results/speculative-matched-qwen36-27b-dspark-cap3-20260715.json`](../benchmarks/results/speculative-matched-qwen36-27b-dspark-cap3-20260715.json)
- [`benchmarks/results/speculative-matched-qwen36-27b-dspark-cap4-20260715.json`](../benchmarks/results/speculative-matched-qwen36-27b-dspark-cap4-20260715.json)
- [`benchmarks/results/speculative-matched-qwen36-27b-dspark-full-20260715.json`](../benchmarks/results/speculative-matched-qwen36-27b-dspark-full-20260715.json)
- [`benchmarks/results/speculative-matched-qwen3-4b-20260715-v041.json`](../benchmarks/results/speculative-matched-qwen3-4b-20260715-v041.json)
- [`benchmarks/results/speculative-matched-qwen3-4b-lookup-20260715-v041.json`](../benchmarks/results/speculative-matched-qwen3-4b-lookup-20260715-v041.json)
- [`experimental/upstream_dflash/results/qwen36-27b-fused-cold-prefill-20260715.json`](../experimental/upstream_dflash/results/qwen36-27b-fused-cold-prefill-20260715.json)
- [`experimental/mixture/README.md`](../experimental/mixture/README.md)
- [`scripts/bench_speculative_matched.py`](../scripts/bench_speculative_matched.py)
- [`benchmarks/results/qwen36-core-256.json`](../benchmarks/results/qwen36-core-256.json)
- [`benchmarks/results/qwen36-cache-256.json`](../benchmarks/results/qwen36-cache-256.json)
- [`benchmarks/results/swebench-quality-27b-smoke-0fd8389.json`](../benchmarks/results/swebench-quality-27b-smoke-0fd8389.json)
- [`scripts/bench_qwen36_matrix.py`](../scripts/bench_qwen36_matrix.py)
- [`benchmarks/results/qwen36-20260715-192941.json`](../benchmarks/results/qwen36-20260715-192941.json)
- [`benchmarks/results/qwen36-20260715-193332.json`](../benchmarks/results/qwen36-20260715-193332.json)
- [`docs/12-architecture.md`](../docs/12-architecture.md)
- [`docs/13-development-plan.md`](../docs/13-development-plan.md)

The first seven R&D artifacts and the mixture prototype above were untracked
and generated from a dirty tree at this snapshot. Listing them makes the local
evidence auditable; it does not make them reviewed, released, or published.
The two timestamped `qwen36-20260715-*` files are likewise dirty-tree,
single-prompt diagnostics for the vendored verifier. The two named schema-v1
core/cache JSON files are the older versioned evidence and retain their own
narrower provenance.

## Appendix B. Relevant implementation checkpoints

| Commit | Role |
|---|---|
| `1724e16` | Qwen 3.6 registry/model support and MLX stack update |
| `4c17a0a` | deterministic persisted configuration/cache selection |
| `3c66c33` | Qwen 3.6 sliding-attention DFlash and state fixes |
| `d49dec2` | last-logit prefill optimization and benchmark matrix |
| `1e82978` | benchmark provenance correction |
| `d937b1c` | complete TurboQuant cache-state restore |
| `3733d1a` | pinned 916-skill catalog integrated into Mio |
| `d8c42ba` | MLX-LM continuous batch path and real 4B smoke |
| `9b9bb14` | runtime hardening and reproducible R&D harness base revision |
| `0fd8389` | clean two-pair Qwen 3.6 27B Quality generation |
| `9aec54f` | bytecode-immutable official evaluator and sealed rerun |

The versioned schema-v1 performance JSON records
`d49dec26dbd6053526027e013d5580e9cf5c10f4` with a clean tree. The new
schema-v2 and fused-prefill artifacts record
`9b9bb142f97958f720e26f29233b27c5d2f06978` with a dirty tree. Later commits
are engineering context unless a clean rerun explicitly records them.

## Appendix C. References

1. J. Chen, Y. Liang, and Z. Liu, “DFlash: Block Diffusion for Flash
   Speculative Decoding,” arXiv:2602.06036, 2026.
   <https://arxiv.org/abs/2602.06036>
2. Apple ML Explore, “MLX.” <https://github.com/ml-explore/mlx>
3. Apple ML Explore, “mlx-lm.” <https://github.com/ml-explore/mlx-lm>
4. z-lab, “Qwen3.6-27B-DFlash.”
   <https://huggingface.co/z-lab/Qwen3.6-27B-DFlash>
5. Brooooooklyn, “Qwen3.6-27B-UD-Q4_K_XL-mlx.”
   <https://huggingface.co/Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx>
6. Headroom Labs, “Headroom.”
   <https://github.com/headroomlabs-ai/headroom>
7. D. Gebert, “Ponytail.” <https://github.com/DietrichGebert/ponytail>
8. A. Karpathy, “LLM Wiki” pattern.
   <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
9. Nutlope, “Hallmark.” <https://github.com/Nutlope/hallmark>
10. M. Pocock, “Skills.” <https://github.com/mattpocock/skills>
11. Ruler-Dev, “Anthropic Cybersecurity Skills.”
    <https://github.com/Ruler-Dev/Anthropic-Cybersecurity-Skills>
12. Ruler-Dev, “Claude Code Game Studios.”
    <https://github.com/Ruler-Dev/Claude-Code-Game-Studios>
13. X. Cheng et al., “DSpark: Confidence-Scheduled Speculative Decoding with
    Semi-Autoregressive Generation,” arXiv:2607.05147, 2026.
    <https://arxiv.org/abs/2607.05147>
14. A. Rahim, “mlx-dspark.” <https://github.com/ARahim3/mlx-dspark>
15. bstnxbt, “dflash-mlx.” <https://github.com/bstnxbt/dflash-mlx>
16. Avesed, “Qwen3.6-27B-DSpark.”
    <https://huggingface.co/Avesed/Qwen3.6-27B-DSpark>
