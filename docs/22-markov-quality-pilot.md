# Repository-level Markov quality pilot

> Status: preregistered exploratory protocol; no result yet. The machine-readable
> source of truth is
> [`benchmarks/repository-quality-four-arm-preregistration-v2.json`](../benchmarks/repository-quality-four-arm-preregistration-v2.json).

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

The v2 smoke runner is the only path authorized to publish this experiment's
result envelope:

```bash
uv run --locked python -m experimental.effort.run_repository_quality_pilot \
  --split smoke \
  --tier small \
  --target-path <exact-Qwen3.5-4B-4bit-directory> \
  --draft-path <exact-Qwen3.5-4B-DFlash-directory> \
  --output <path-outside-the-repository-and-model-directories>
```

It accepts no effort, router, seed, prompt, budget, sampler, or hidden-evaluator
override. Before model loading it requires a clean committed tree and verifies
the exact source list, v2 preregistration SHA-256, corpus/private-evaluator and
Quality-profile seals, local model fingerprints, Python/package versions, and
hardware identity. The private work and output locations must be disjoint from
source and model roots. Result publication is create-once and refuses to
overwrite an existing artifact. The model manager is unloaded before the hidden
evaluator is constructed. Verification runs again after generation and before
publication. A bare core aggregate produced with an injected test executor is
not a publishable v2 result; only the native, receipt-bound, source-free
envelope is.

This protocol revision authorizes only `smoke`. An `all` run needs a later
wrapper that consumes a valid smoke integrity artifact without consulting its
quality, route, rescue, regression, or cost outcomes.

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

Protocol v2 reports the raw `workspace_evaluator_passed` bit separately from
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
