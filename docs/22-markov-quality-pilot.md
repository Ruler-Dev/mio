# Repository-level Markov quality pilot

> Status: the single v4 smoke is complete with an exploratory no-claim result.
> The
> machine-readable source of truth is
> [`benchmarks/repository-quality-four-arm-preregistration-v4.json`](../benchmarks/repository-quality-four-arm-preregistration-v4.json).
> The exact source-free [result](../benchmarks/results/repository-quality-four-arm-v4-smoke-f5d04dc.json)
> and [start receipt](../benchmarks/results/repository-quality-four-arm-v4-smoke-attempt-start-f5d04dc.json)
> are committed separately.
> V2 and v3 are immutable failed attempts, not benchmark results. V2 has a
> [post-hoc incident record](../benchmarks/incidents/repository-quality-four-arm-v2-smoke-aborted-8bf6e6e.json).
> V3 has the exact native [start receipt](../benchmarks/incidents/repository-quality-four-arm-v3-smoke-attempt-start-16213e2.json),
> [abort receipt](../benchmarks/incidents/repository-quality-four-arm-v3-smoke-abort-16213e2.json),
> and [incident analysis](../benchmarks/incidents/repository-quality-four-arm-v3-smoke-incident-16213e2.json).

## Why this experiment is necessary

Quality Gate v2 improved completion integrity but did not improve hidden task
success in its one-use development run. Plain and Quality both passed `2/8`,
while Quality used `2.5444x` wall time, `2.6413x` model time, and `2.1858x`
output tokens. Increasing effort indiscriminately is therefore rejected.

The existing HumanEval Markov controller cannot be transferred into this
experiment. It operates on completion strings rather than persistent agent
workspaces, its calibration identity is different, and both committed 4B
calibration artifacts contain zero eligible transition rows. A faithful use of
those artifacts would simply accept Quality or stop; inventing a transition
would invalidate the experiment.

This pilot tests the missing repository-level mechanism: whether one bounded
recovery turn can be allocated from a sealed, public workspace state more
efficiently than assigning the same number of recovery turns statically. It is
a failure-routing engineering pilot, not evidence of a calibrated MDP, provider
reasoning equivalence, improved coding quality, or a breakthrough.

## Four logical arms, two shared roots

Each task physically generates one Plain root and one Quality-medium root. The
Quality root is archived and reused byte-for-byte by three logical arms:

| arm | root | possible extra turn |
| --- | --- | --- |
| `plain` | Quality off | none |
| `quality` | shared Quality-medium | none |
| `quality_static_extra` | shared Quality-medium | seeded static allocation |
| `markov_quality` | shared Quality-medium | public-state allocation |

Regenerating separate Quality roots for Static and Markov is forbidden because
sampling, runtime noise, or thermal order could otherwise be mistaken for a
routing effect. Plain and the shared Quality root use a deterministic AB/BA
counterbalance. Every physical generation starts with cold mutable engine
state.

The frozen seed is `20260719`, encoded as unpadded base-10 ASCII. Fixture IDs
are UTF-8 and fields are separated by one NUL byte. Tasks are ordered by raw
SHA-256 bytes using the domain `mio.repository-quality.root-order.v1\0`, then
Plain/Quality is used at even positions and Quality/Plain at odd positions.
Static and extra ordering use the distinct domains recorded in the
machine-readable preregistration. Schedules are serialized before generation.

## Public state and depth-one action

After every root is complete, the harness seals only controller-visible
evidence. A root is classified as `root_incomplete`, `scope_invalid`,
`public_fail`, or `public_unknown`. The last label means that public evidence
has not established a failure; it deliberately does not say the solution is
semantically correct.

The classifier is total and ordered. It first emits `root_incomplete` when a
snapshot or tool telemetry is incomplete, a budget/deadline limit was hit, the
Quality decision is not `pass`, or the terminal reason is not `model_final`.
Only if that rule is false can invalid scope emit `scope_invalid`. Only if both
earlier rules are false can a failed visible test emit `public_fail`; every
remaining state is `public_unknown`. A visible-test value is three-valued:
`false` when any trusted current-revision/current-epoch `validate` test or build
is denied or non-OK, `true` when at least one exists and all are allowed and
OK, and null when none exists. Ordinary shell output never counts. Malformed or
contradictory telemetry aborts the cohort instead of being guessed.

`scope_valid` is also public and host-computed, not an agent assertion. It is
bound to the sealed fixture ID, pristine public manifest, editable-name set,
and exact terminal tree digest. Names and modes must remain identical, every
non-editable byte must remain unchanged, and at least one editable file must
change. Add/delete/rename, symlink, special-file, hard-link, alias, stale
verdict, or wrong-fixture verdict fails scope before routing or child selection;
the hidden evaluator is never consulted for this check.

The exploratory depth-one router may schedule one recovery when:

```text
snapshot_complete && telemetry_complete && (hard_debt || coverage_debt)
```

`hard_debt` covers an incomplete terminal/gate, invalid scope, or visible-test
failure. `coverage_debt` covers a multi-mutation trajectory without trusted
static/diff evidence. Hidden pass/fail, hidden output, peer-arm results, and
request/artifact hashes are forbidden routing features.

The recovery has a fixed High-effort envelope of four model rounds, eight tool
calls, 384 output tokens, 20 seconds, and 8,192 context tokens. Its prompt is
sealed as `mio.repository-quality.recovery-prompt.v1`, and it continues the
root's agent history on a verified workspace clone. Its reconstructed Quality
gate retains the pristine task snapshot as baseline and the cloned root as the
current revision, but inherits no validation evidence. This prevents a valid
root edit from becoming a false `no_net_change` while still requiring the final
revision to earn fresh trusted evidence.

## Frozen runtime and isolation

The sampler is greedy (`temperature=0`, `top_p=1`, `top_k=0`) and therefore
receives no RNG seed. `20260719` is used only for schedules, static allocation,
and bootstrap indices. Target and strict DFlash identities, MLX package versions, context,
quantization, DFlash geometry, prefill chunk, and every environment variable
that can override that path are frozen in the JSON. Before every physical root
or child generation the runner invalidates the engine prefix cache, clears the
last-prompt and pending-prefill state, and resets the DSpark prefix cache when
present. Direct roots receive fresh conversations; a recovery deep-copies the
Quality conversation but receives the same cold engine reset.

V2 stopped fail-closed during root generation because the normal DFlash stream
did not export raw nanosecond phase timings. The engine correctly labeled its
microsecond conversion `derived_legacy_us`, which the protocol rejected:
stream elapsed time can include generator suspension and downstream consumer
work. V3 and v4 do not relax that check. DFlash now accumulates disjoint intervals
between explicit active-runtime probes: timing stops before each event
dictionary is built and restarts only after the generator itself resumes.
Event construction, telemetry serialization, and consumer suspension are not
charged to decode. Warm-prefix draft-context synchronization is charged to
prefill; token materialization and final-state cache synchronization complete
before decode timing closes. `elapsed_us` is the separate runtime-phase wall
interval from immediately before prefill through final-state synchronization;
it excludes earlier prompt/cache setup and never substitutes for
`model_total_ns`.

The only v3 smoke completed and bound all eight direct roots, then aborted
fail-closed in phase `allocation_sealed` before a generation receipt, hidden
evaluator, hidden outcome, or aggregate existed. Its native abort reports zero
completed extras; that does not assert whether an in-flight first extra had
begun. The source-free exception digest matches the committed static invariant
`observed wall time exceeds the budget without exhaustion`. The strict
validator was correct: `AgentTurnResult.wall_time_s` included terminal Quality
refresh/report/bookkeeping, but that post-loop path could cross the deadline
without recording exhaustion.

V4 changes only that accounting. After terminal Quality refresh, state
persistence, and telemetry reconciliation, the agent takes one final
`time.perf_counter` sample. The same sample defines `wall_time_s` and any newly
crossed wall-limit exhaustion. Only `model_final` becomes `budget_exhausted`;
`quality_incomplete`, `tool_timeout`, and existing budget terminal reasons keep
precedence. Assistant text and persisted history are unchanged. The resulting
trajectory is typed but `root_incomplete`, so a recovery falls back to its root
instead of malformed telemetry aborting the cohort. Budgets remain exactly 120
seconds for direct turns and 20 seconds for recovery; no grace was added and
the strict validator remains fail-closed while now using the same inclusive
at-or-beyond boundary as the runtime.

An agent receives exactly one active task workspace and the fixed local tool
surface. Network and MCP access are disabled for this benchmark. Hidden and
public evaluator programs, task/arm schedules, peer workspaces, archived roots,
and earlier results remain host-only and outside its allowed roots. The shell
stays inside the workspace sandbox. Symlinks, special files, aliases, scope
escapes, and policy-boundary failures fail closed. Existing public-suite,
private-evaluator-bundle, and Quality-profile digests are checked alongside a
clean committed source lock before and after execution.

There is exactly one attempt per physical generation and per unique hidden
evaluation. A model budget/deadline terminal is an observed invalid trajectory,
not infrastructure and not retried. A visible or hidden assertion failure, or
the evaluator's frozen timeout, is likewise a verdict. Engine/reset exceptions,
process crashes, malformed telemetry, snapshot/archive failures, evaluator
exceptions, or source/model drift abort the entire cohort without an aggregate.
This revision permits no discretionary rerun; a failed cohort requires a
documented, committed protocol revision.

## Native executable and publication boundary

The v4 smoke runner is the only path authorized to publish this experiment's
result envelope:

```bash
uv run --locked python -m experimental.effort.run_repository_quality_pilot \
  --split smoke \
  --tier small \
  --target-path <exact-Qwen3.5-4B-4bit-directory> \
  --draft-path <exact-Qwen3.5-4B-DFlash-directory> \
  --attempt-root <new-persistent-create-once-directory> \
  --output <path-outside-the-repository-and-model-directories>
```

It accepts no effort, router, seed, prompt, budget, sampler, or hidden-evaluator
override. Before model loading it requires a clean committed tree and verifies
the expanded critical-source manifest, v4 and predecessor preregistration
SHA-256 values, both v3 native receipt digests, the v3 incident digest,
corpus/private-evaluator and
Quality-profile seals, local model fingerprints, Python/package versions, and
hardware identity. The private work and output locations must be disjoint from
source and model roots. Result publication is create-once and refuses to
overwrite an existing artifact. The model manager is unloaded before the hidden
evaluator is constructed. Verification runs again after generation and before
publication. A bare core aggregate produced with an injected test executor is
not a publishable v4 result; only the native, receipt-bound, source-free
envelope is.

After provenance and destination checks—but before model loading—the runner
atomically creates `attempt-start.json` under the new attempt root. A terminal
attempt then creates exactly one sibling: `result.json` on success or
`abort.json` on failure. Both bind the start-receipt SHA. The abort envelope
contains only typed phase/state fields, content-free progress counts, and an
exception-message digest; it excludes paths, prompts, fixture identifiers,
candidate bytes, tracebacks, and hidden outcomes. Failure before provenance or
destination acceptance does not claim a scientific attempt. If start
publication reports a failure after installing the exact bytes, the runner
verifies them and terminalizes with a bound abort before model loading. The one-attempt
authorization remains a procedural research rule because an operator could
delete external receipt storage; an abort nevertheless consumes v4 and any
later attempt requires a committed v5.

This protocol revision authorizes only `smoke`. An `all` run needs a later
wrapper that consumes a valid smoke integrity artifact without consulting its
quality, route, rescue, regression, or cost outcomes.

## V3 release and disposition

The final pre-attempt release candidate was checked on 2026-07-19 with the
locked environment and no model inference:

```text
PYTHONPATH=. uv run --locked pytest -q
1596 passed, 2 skipped, 1 pre-existing Starlette/httpx deprecation warning

uv run --locked ruff check <five changed Python files>
All checks passed

uv run --locked ruff format --check <five changed Python files>
5 files already formatted
```

The historical v3 preregistration SHA-256 is
`d3ddbfa29bc99f2b480797fadf6686cbc200f973e6fa6325805855494d600d3d`;
the v2 predecessor and post-hoc incident seals are verified by the runner.
This attestation records a procedural release gate, not a quality, speed, or
breakthrough result. The native v3 start bound Git revision
`16213e264d38993f8e5b074588d424a199269dbe` and the complete 28-file
critical-source digest. Its abort receipt is bound to that start by SHA-256.
V3 produced no hidden evaluation and no aggregate, so its completed roots and
any unbound in-flight extra cannot be reused regardless of individual validity.

## V4 release attestation

V4 is frozen under preregistration SHA-256
`ae49deb27e1929c76b032a65ee1515e3a0ac270d78cdb49fa791aa6d9ca93381`.
The final pre-attempt candidate was checked on 2026-07-19 without model
inference:

```text
PYTHONPATH=. uv run --locked pytest -q
1609 passed, 2 skipped, 1 pre-existing Starlette/httpx deprecation warning

uv run --locked ruff check <seven changed Python files>
All checks passed

uv run --locked ruff format --check <seven changed Python files>
7 files already formatted
```

Because `mio/agent.py` belongs to the transitive HumanEval verifier source
bundle, the old certificate correctly failed closed after the accounting
change. The public parity harness was rerun from clean, stable Git revision
`ca3cbcb2d3693f4e451d42e6d634e1952fcb1d7d`: all `164/164` canonical
solutions passed. The new source-free
[certificate](../benchmarks/results/humaneval-verifier-parity-ca3cbcb.json)
has SHA-256
`43c36131409f8edb132ab2fada88d17bcf9e203c3d6dfacadca1d70f0e8e4c6b`.
JSON parsing, source-free/privacy scans, exact v3 receipt bindings, the 34-file
manifest, and `git diff --check` also pass. These are integrity attestations,
not quality, speed, or breakthrough results. The v4 native start subsequently
bound clean Git revision `f5d04dc2accff53a909fa4c11fe8a448754124b9`
and the complete 34-file source digest.

## V4 smoke result

The create-once attempt completed normally. Its start SHA-256 is
`2523a4a9849f4c49e36a964961f074460003adc365c2fe9c2e9e328ed38b67cc`;
the result SHA-256 is
`f34a58a375fe5a392a140664e3659e1d788a2d2e11d74fe6a5cdf0e799d84d88`.
All eight scheduled roots and the one unique scheduled extra completed before
eight single-use hidden evaluations. Selection was sealed first, the model was
unloaded before hidden evaluation, source/model/runtime identity remained
stable, and no hidden label is serialized.

| arm | composite pass | terminal complete | selected child | model seconds | wall seconds | output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain | 3/4 | 3/4 | — | 42.6149 | 79.0953 | 2,230 |
| Quality | 3/4 | 3/4 | — | 39.6131 | 75.9603 | 1,376 |
| Static extra | 3/4 | 3/4 | 0/1 | 44.8145 | 96.0713 | 1,484 |
| Markov Quality | 3/4 | 3/4 | 0/1 | 44.8145 | 96.0713 | 1,484 |

Quality versus Plain was neutral on correctness (`0.75` each) in this four-task
smoke. The router selected one task; the allocation-matched static control
selected the same task, so they shared one physical recovery. That child did
not strictly improve public state and was not selected. Markov versus Quality
therefore had zero composite and workspace-evaluator gain while costing
`1.1313x` model time, `1.0785x` output tokens, and `1.2648x` wall time. Three
physical candidate observations contained a budget/deadline/snapshot/telemetry violation
and correctly remained incomplete.

The frozen analysis is not promotion-eligible: this is not the required all
cohort, there was only one independent route, both gain lower bounds and rescue
probability were zero, wall cost exceeded the `1.25x` gate, and integrity-cost
violations were nonzero. Consequently the result supports no quality, speed,
Markov, or breakthrough claim and does not authorize the `all` cohort. Its
useful result is narrower: the raw DFlash timing, sandboxed coding harness,
allocation barrier, recovery fallback, hidden-evaluation isolation, and
source-free publication path now complete end to end.

## Allocation-matched static control

Let `K` be the number of Markov routes computed after all roots are sealed and
before any hidden evaluation. Static receives exactly `K` recovery allocations.
Its tasks are the lowest deterministic hashes of
`mio.repository-quality.static-order.v1 || NUL || seed || NUL || fixture_id`;
neither public difficulty nor hidden labels can influence them. Overlapping
Static/Markov allocations generate one exact physical child and reuse it.

Static and Markov use the exact same prompt, tool surface, sandbox, effort, and
resource caps. If both allocate the same task, one exact physical child is
reused. Actual cost remains reported rather than normalized away. `K=0` and
`K=all tasks` are valid negative feasibility outcomes but cannot promote the
policy.

## Selection and leakage barrier

The child replaces its root only on a strict lexicographic improvement in this
frozen integer tuple (larger is better):

```text
(state_rank,
 quality_decision_is_pass,
 terminal_reason_is_model_final,
 trusted_test_or_build_present,
 trusted_static_or_diff_present)
```

State ranks are `scope_invalid=0`, `root_incomplete=1`, `public_fail=2`, and
`public_unknown=3`. A child is inadmissible if it has a budget/deadline
violation, incomplete telemetry or snapshots, or invalid scope. Ties retain
the root; latency, tokens, hashes, and hidden outcomes never break a tie. Every
logical terminal selection is immutable before the hidden evaluator becomes
callable.

Logical selection and physical deduplication are separate. A no-edit child may
still be logically selected if fresh trusted evidence makes its public score
strictly better; identical bytes do not force root retention. Conversely,
identical bytes never copy gate, completion, or telemetry validity from one
trajectory to another.

Hidden evaluation then runs once per exact `(fixture, terminal workspace)` and
reuses only the workspace-determined evaluator verdict across arms selecting
identical bytes. Each logical arm separately earns trajectory validity from
complete snapshots/telemetry, valid scope, no budget/deadline violation,
`model_final`, and—except for Plain—a `pass` Quality decision. The primary
`hidden_task_success` is evaluator pass AND that arm's trajectory validity.
Generation, routing, static allocation, recovery, and selection all finish
before the first hidden call. No hidden feedback can trigger a repair or choose
a candidate. Before sealing, a typed generation receipt must account for exactly
`2N` roots and every scheduled unique extra; a plain caller boolean is rejected.
The aggregate is then constructible only with a typed barrier receipt attesting
that every expected logical terminal was registered after generation,
selection was sealed before the first callback, evaluation was single-use, and
the number of unique `(fixture, terminal bytes)` pairs equals the callback
count; these integrity fields are not hard-coded report claims.

The protocol reports the raw `workspace_evaluator_passed` bit separately from
the composite success endpoint. This distinction is mandatory: an identical
terminal artifact can become a valid completed trajectory after fresh trusted
validation, but its code bytes did not thereby become more correct. Every
paired contrast therefore reports both composite and raw evaluator
contingencies. The analysis also separates raw evaluator rescues from
trajectory-only rescues and changed-terminal rescues from same-terminal
rescues. Advancement requires a positive one-sided lower bound for both the
composite gain and the raw workspace-evaluator gain, zero regressions under
both endpoints, and raw Markov pass count at least as high as Quality and
Static. Thus validation-only improvement can be useful harness evidence but
cannot satisfy the workspace-correctness gate.

Cost endpoints are equally explicit. Output tokens are the sum of typed round
completion tokens. `model_seconds` is the sum of raw
`model_total_ns = prefill_ns + decode_ns`, divided by one billion; round
`total_time_s` is checked but is not substituted for this model-phase measure.
`wall_seconds` is the complete agent-turn wall time, including orchestration
and tools. Logical arm cost charges every allocated child even on fallback;
physical cost deduplicates the shared work separately.

## Interpretation and stop rules

The four predeclared contrasts are Quality−Plain, Static−Quality,
Markov−Quality, and Markov−Static. Reports include paired rescues/regressions,
logical token/model/wall cost, physical experiment cost, route/selection
counts, unique terminal artifacts, and evaluation reuse.

The first authorized cohort is the four-task smoke, solely to validate the
harness. Standalone smoke and development runs can never pass promotion and
cannot be pooled. The only promotion-eligible exploratory population under
this revision is a separately regenerated `all` cohort: all 12 tasks, all
roots/extras/selections behind one barrier before its first hidden call. A
later `all` run may follow smoke only when every smoke integrity boundary
passes; smoke quality, rescue, route, regression, and cost outcomes are ignored
for that continuation decision. All of these tasks are already observed and
none can support a quality or speed claim.

The population is deliberately narrow: 12 small Python standard-library
fixtures, each with one editable implementation file and public `unittest`
checks, run on the frozen Qwen3.5 4B target. It contains no JavaScript,
TypeScript, React, CSS, browser rendering, C, C++, C#, dependency installation,
large multi-file repository work, SWE-bench Verified task, or 27B execution.
Consequently this pilot cannot establish frontend aesthetics, multi-language
coding quality, repository-scale agent quality, or 27B generalization. Those
questions require separately preregistered held-out suites and validators.

For the 12-task exploratory gate, net quality is the paired mean of
`Markov success - Quality success`. Rescue probability is calculated only over
Markov-routed tasks on which Quality composite success is false: the numerator
is those becoming successful under Markov and the denominator is exactly that
set. The route count must be at least eight and less than 12, so `K=all` remains
non-promotable. Zero-denominator rescue samples score zero rather than being
dropped.

One-sided 95% lower bounds use 10,000 paired, task-level resamples. Indices come
from the frozen SHA-256 counter generator and seed `20260719` in the JSON; the
500th sorted value (zero-based index 499) is conservatively clamped by the
observed point. Before indexing, analysis rows are sorted by fixture ID in
ascending Unicode code-point order. Missing/invalid rows are never imputed or deleted: they fail the
gate. Cost limits are explicitly point ratios of total logical Markov cost to
total logical Quality-root cost; an allocated child is charged even when it
falls back, while physically deduplicated cost is reported separately.

Progress to a newly authored untouched calibration additionally requires
positive conservative rescue and net-quality bounds, zero Quality-success
regressions, Markov success at least equal to both Quality and Static, the
frozen point cost ratios, allocation parity, and no deadline, snapshot,
budget, telemetry, isolation, or protocol violation. Any future quality claim requires
a newly created untouched repository-task split and a new committed protocol.
