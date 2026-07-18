# Coding quality gate and MioCodeBench v1 preregistration

> Status: preregistered protocol; no benchmark result is reported on this
> page. The machine-readable source of truth is
> [`benchmarks/coding-quality-preregistration-v1.json`](../benchmarks/coding-quality-preregistration-v1.json).

Mio's coding quality gate is a deterministic control around the native agent
loop. It does not claim to make the base model intrinsically more capable. It
requires evidence after a workspace mutation, can return the agent to the tool
loop when evidence is missing, and prevents an unverified change from being
reported as successfully completed. This experiment asks whether that control
improves end-to-end repository-task success and measures its cost.

The protocol is frozen before the first scored run. An implementation change,
prompt change, threshold change, task substitution, hidden-evaluator change,
or model change starts a new protocol version; it is not an amendment to v1.

## 1. Questions and hypotheses

The primary estimand is the paired difference in held-out hidden task success:

```text
mean(success_gate_on - success_gate_off)
```

`success` means that the produced workspace passes the external hidden
evaluator and all regression checks. The primary null hypothesis is that the
paired difference is at most zero. The directional alternative is that the
mandatory gate improves success.

Secondary questions are:

- how often a mutation is followed by valid evidence from the latest mutation
  epoch;
- whether the gate reduces incomplete edit-without-test trajectories;
- whether it changes valid-patch, regression-free, and scoped-edit rates;
- what it costs in model rounds, generated tokens, tool calls, wall time and
  task completions per hour;
- whether raw inference throughput changes even though both arms use the same
  inference stack.

No result from this study may be used as a DFlash-versus-target speed or token
parity claim. Both arms use the same target and DFlash identities; the only
experimental factor is the coding quality gate.

## 2. Frozen identities and conditions

The run must fail closed unless the installed model contents still match these
full audited identities:

| Role | Repository label | Preregistered content identity |
|---|---|---|
| target | `mlx-community/Qwen3.5-4B-4bit` | `local-sha256-v1:7d7ea69d09ada4f1d2f49f6ca651441ac279b95b6d280f259da04fbde504376f` |
| DFlash drafter | `z-lab/Qwen3.5-4B-DFlash` | `local-sha256-v1:4b60bced36f602da85a6447d3648e7ac37a0c5cce68d505a49664252c0586b98` |

These values identify preregistered local content. They are not benchmark
results. The runner must recompute both identities immediately before and
after the study and serialize only the identities, never absolute local model
paths.

The two conditions are built from one clean Mio commit:

- `gate_off`: the deterministic quality gate is disabled by the benchmark-only
  switch;
- `gate_on`: the same agent, prompt policy, model stack, sampler, permissions,
  round budget and tools run with the mandatory quality gate enabled.

The off switch is an experimental control, not a production default. It must
not widen tool permissions or bypass the existing sandbox and audit policy.
Any condition-specific difference beyond gate instructions, gate state,
evidence collection and deterministic gate feedback invalidates the study.

The run records the full clean implementation commit and a digest of all
gate, agent-loop, prompt and benchmark sources. An arm observed on another
commit is not paired evidence.

## 3. What the gate certifies

A successful `write` or `edit` begins a new mutation epoch. Evidence from an
earlier epoch no longer certifies the workspace. Only an authoritative tool
audit event with an allowed operation and successful outcome can satisfy the
gate; assistant prose and text that merely resembles a command cannot.

The gate recognizes validation categories conservatively:

- `test`: a project test runner or a narrowly targeted test command;
- `build`: a recognized project build command;
- `static`: a parser, compiler, type checker or linter check;
- `diff`: repository-integrity checks such as malformed-diff or whitespace
  validation;
- `review`: a distinct review/remediation stage when an effort profile
  requires one.

The exact classifier and effort profile are source-bound by the run manifest.
Unknown commands never count as evidence. A nonzero, timed-out, denied or
output-limited command is recorded but does not satisfy a success requirement.
Another mutation invalidates the evidence and requires validation again.

If a task ends without a workspace mutation, the gate does not fabricate a
quality certificate. If a task mutates the workspace but cannot meet its
configured profile within the round or tool budget, its terminal state is
`incomplete`; it must not silently report success. Likewise, an effort tier
whose distinct capabilities are not implemented is unavailable and fails
closed. In particular, a higher tier is never silently aliased to a lower
tier.

## 4. Effort profiles

The primary benchmark profile is the production CLI/config default, `medium`.
The standalone gate object has a defensive `high` constructor default, but the
native agent passes the persisted `medium` setting explicitly. Every run
records the effective profile and the profile-table digest.

| Effort | Evidence required after the latest mutation |
|---|---|
| `low` | any one successful trusted validation |
| `medium` | code/unknown mutation: `test` or `build`; documentation-only mutation: `diff` |
| `high` | `test` and at least one of `static` or `diff` |
| `xhigh` | `test`, `static`, and `diff` |
| `ultra` | all `xhigh` evidence, plus trusted `review` or at least two successful `test` commands with distinct command digests |

Mutations observed through unknown, Bash or MCP paths are classified as code
for fail-closed evidence requirements; they do not receive the documentation-
only relaxation. The profiles are monotonic. A tier that cannot satisfy its
distinct requirement within the available capabilities or budget ends
`incomplete`; it is never aliased or downgraded to another tier.

## 5. Corpus and split discipline

MioCodeBench v1 contains 28 local repository-edit tasks:

| Split | Count | Permitted use |
|---|---:|---|
| smoke | 4 | verify runner, workspace reset and evaluator plumbing only |
| development | 8 | debug the implementation and choose documented defaults |
| held-out | 16 | one sealed confirmatory comparison; no tuning |

The task manifest assigns stable opaque IDs and freezes the initial public
tree digest, prompt digest, hidden-evaluator digest, language, task family,
timeout and allowed edit scope. The sealed manifest digest must be published
before any held-out generation. The task bodies and hidden tests are not
embedded in the public result artifact.

Smoke results are never quality evidence. Development results may find defects
but may not be combined with held-out rows. Once the release candidate,
manifest and schedule are sealed, no default, prompt, classifier, tier,
timeout or task can be changed in response to development or held-out scores.

Each arm starts from a separate copy of the same read-only fixture snapshot.
The agent can read only the public workspace. Hidden evaluators, expected
patches, reference solutions, task labels and the other arm's workspace are
outside every allowed root.

## 6. Leakage firewall

The confirmatory phase uses a generate-then-evaluate firewall:

1. verify the clean code revision, package lock, model identities, corpus seal
   and precomputed arm schedule;
2. generate all 32 held-out arms without invoking or reading any hidden
   evaluator;
3. seal final tree digests, content-free trajectories and execution status;
4. verify that no model, code, task or schedule identity changed;
5. run hidden evaluation once for every valid arm;
6. aggregate with the preregistered analysis and write a new immutable result
   artifact.

The runner and the agent process must not receive hidden test output. Hidden
evaluation cannot trigger remediation, another model turn, a prompt edit, or
selection between candidate patches. The hidden label for one task remains
unavailable until generation for all held-out tasks and both arms is complete.

An infrastructure-invalid arm may be rerun only for a reason declared before
unblinding: process crash, host sleep/restart, model identity mismatch,
fixture-copy failure, evaluator infrastructure failure, or telemetry loss.
The whole pair is rerun from the original fixture with the same seed; the
reason and both attempts remain in the audit artifact. A model/tool mistake,
test failure, timeout within the task budget, incomplete gate, bad patch or
failed hidden test is an outcome, not an infrastructure retry.

## 7. Pairing and execution order

`gate_off` and `gate_on` form a pair for each task. Prompt, initial tree,
target/DFlash contents, sampling parameters, tool policy, system resources,
token budget, wall timeout and maximum tool rounds are identical. Agent
history, workspace, prefix cache and mutable runtime state are reset between
arms.

Order is frozen from seed `20260718` using a task-ID hash and is balanced
within each split: half of held-out tasks run off then on and half run on then
off. Execution order is serialized before generation. Warm-up workloads are
fixed, excluded from analysis and cannot expose hidden fixtures. Thermal and
memory telemetry are recorded so an order effect can be reported, never
silently corrected after inspection.

## 8. Outcomes and timing

The primary binary outcome is `hidden_task_success`. It requires all of:

- the hidden evaluator passes;
- declared public/regression checks pass;
- the final tree is a syntactically valid patch relative to the initial tree;
- edits remain inside the task's allowed scope;
- no hidden evaluator, credential, forbidden path or peer-arm data was read;
- the run did not terminate as infrastructure-invalid.

Secondary quality outcomes include hidden-test pass, public-test pass,
regression-free status, valid patch, scoped edit, successful mutation followed
by latest-epoch validation, incomplete terminal state, and baseline-pass to
gate-fail discordance.

Efficiency outcomes include model rounds, input/output tokens, tool calls,
mutation count, validation attempts by category, time to first token, decode
tokens per second, model-active seconds, tool seconds, evaluator seconds,
end-to-end task seconds, peak unified memory, and successful tasks per host
hour. Evaluator time is reported separately and excluded from agent task time.
Raw inference throughput and end-to-end task efficiency are never conflated.

Timers must synchronize lazy MLX work inside the measured phase. The result
records the timing method, warm-up policy, package versions, hardware class and
raw per-arm timing values needed for paired reanalysis.

## 9. Frozen analysis

All confirmatory estimates use only the 16 held-out task pairs. Invalid pairs
are reported and excluded from the primary estimate; the benchmark is
non-promotable if any pair is missing after the single declared infrastructure
rerun procedure.

For binary quality outcomes, report:

- both arm counts and rates;
- the paired percentage-point difference;
- the four paired outcome cells;
- a two-sided exact McNemar test over discordant pairs;
- a 95% task-paired bootstrap interval with 10,000 resamples and seed
  `20260718`.

For positive continuous outcomes, report paired per-task ratios, the median
ratio and a 95% task-paired bootstrap interval with the same seed and resample
count. Also publish p50 and p95 raw values by arm. Zero denominators and
timeouts remain explicit and are not replaced with favorable finite values.

The family-wise primary decision has one outcome and one comparison. Secondary
p-values are descriptive and receive Holm correction within the quality and
efficiency families. Development and smoke observations have no p-values and
are never pooled into a held-out interval.

## 10. Claim and promotion gates

The following language is permitted only when every corresponding condition
holds:

- **Mandatory-gate correctness:** deterministic unit and integration tests
  prove that successful mutation invalidates old evidence, failed validation
  does not satisfy the gate, a final answer is rejected while evidence is
  missing, and unavailable upper tiers fail closed.
- **Held-out quality improvement:** the held-out point estimate for
  `hidden_task_success` is positive, its paired-bootstrap lower bound is above
  zero, exact McNemar `p < 0.05`, all 16 pairs are present, and no security or
  evaluator-isolation violation occurred.
- **No material raw-throughput regression:** the lower bound of the paired
  gate-on/gate-off decode-throughput ratio is at least `0.95`.
- **No material task-efficiency regression:** the lower bound of the paired
  gate-off/gate-on end-to-end task-time ratio is at least `0.80`, equivalent to
  ruling out more than a 25% task-time increase under this ratio definition.
- **Coding-quality breakthrough:** held-out quality improvement passes, the
  point improvement is at least 12.5 percentage points, both no-material-
  regression gates pass, there are no baseline-pass/gate-fail task regressions,
  and an independent clean rerun reproduces the direction and all safety gates.

Passing deterministic gate tests does not establish model-quality improvement.
A positive point estimate with an interval crossing zero is inconclusive. A
failed speed gate does not erase a quality result, but it prohibits claims of
quality improvement “without sacrificing speed.” With only 16 held-out tasks,
the study is a bounded local benchmark rather than evidence of universal coding
ability.

## 11. Content-free evidence contract

Public telemetry and the result artifact must not contain raw user prompts,
fixture source, generated source, diffs, assistant prose, shell command text,
stdout/stderr, test names, plaintext paths, expected patches, hidden-test
output, credentials, tokens, usernames or absolute home paths.

Per-arm evidence may contain only:

- versioned schema and full clean code/model identities;
- split and opaque task/arm identifiers derived with run-scoped HMAC;
- HMACs of request, initial tree, final tree, changed paths and commands;
- path extensions and aggregate byte/file counts;
- tool category, allow/deny state, outcome class and bounded numeric detail;
- mutation epoch and validation category/outcome/epoch;
- gate profile, profile digest, decision and incomplete reason enum;
- token, round, call, mutation and validation counts;
- synchronized timing, memory and evaluator pass/fail booleans;
- aggregate paired statistics and preregistered claim-gate decisions.

The HMAC key and plaintext mapping remain outside the repository. Unsalted
hashes are insufficient for low-entropy path or command names. Audit detail
must use enums and bounded numbers rather than truncated raw output. Published
artifacts are append-only: reruns use a new filename and preserve negative or
inconclusive results.

## 12. Interpretation boundary

This protocol can support a claim about one Mio revision, one local Qwen3.5-4B
target/DFlash stack, one task corpus, one tool policy and one host class. It
cannot by itself establish that the gate improves every model, language,
repository, prompt policy or effort tier. It also cannot establish a new
inference algorithm, intrinsic base-model improvement, DFlash parity, or a
universal speedup.

The benchmark report must lead with the gate decision and all failed claim
gates, not only favorable point estimates. Negative, neutral and incomplete
outcomes remain publishable scientific results.
