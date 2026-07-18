# Qwen 3.6 27B quality gate: SWE-bench Verified preregistration

> Status: protocol, trusted dataset adapter, hardened patch capture, immutable
> checkpoints, paired non-evidence smoke runner, portable smoke artifact audit,
> cross-process receipt verifier, and fail-closed official smoke evaluator are
> implemented. Confirmatory `evaluate` and `aggregate` remain deliberately
> **hard-blocked** by the unresolved v2 controls and protocol revision. No
> quality result is reported here. The Mac now has Docker CLI 29.6.2, a prepared
> QEMU/Colima `linux/amd64` profile, the two pinned smoke images, and a clean
> official harness checkout; that VM is intentionally stopped while MLX owns
> host memory and is restarted only for evaluation. The machine-readable
> protocol is
> [`benchmarks/swebench-quality-preregistration-v1.json`](../benchmarks/swebench-quality-preregistration-v1.json).

This is the required external benchmark for Mio's coding-quality gate. It
compares the same local Qwen 3.6 27B target, Mio commit, MLX runtime, prompt,
tool surface, budgets, and fresh repository state under two conditions:

- `gate_off`: the deterministic quality obligation is disabled;
- `gate_on`: the production gate is enabled at `medium` effort.

The study asks whether the gate changes end-to-end issue resolution, not
whether it changes the base model's intrinsic knowledge. No result from a
local toy corpus, smoke subset, raw tok/s test, or incomplete Verified subset
answers that question.

## Official benchmark boundary

SWE-bench gives a system a real repository at a pre-fix commit and a GitHub
issue, then evaluates the generated patch against repository tests. The
original paper defines the issue-resolution task over real repositories and
the official dataset card states that Verified contains 500 human-validated
test instances. The card also defines `base_commit`, `problem_statement`, gold
`patch`, `test_patch`, `FAIL_TO_PASS`, and `PASS_TO_PASS` fields. Only the first
two public task inputs plus `repo` cross Mio's model-input firewall; gold and
evaluator fields never do.

Primary sources:

- [SWE-bench paper](https://arxiv.org/abs/2310.06770)
- [official SWE-bench repository](https://github.com/SWE-bench/SWE-bench)
- [official SWE-bench Verified dataset card](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
- [official evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/)
- [official harness reference](https://www.swebench.com/SWE-bench/reference/harness/)
- [official Docker setup guide](https://www.swebench.com/SWE-bench/guides/docker_setup/)

Confirmatory protocol v1 is pinned to official `swebench` 4.1.0, Git tag
`v4.1.0`, commit `726c5461e2ef52d83cf1ea2107870a8bb3328d57`. The
non-evidence smoke wrapper instead pins official commit
`f7bbbb2ccdf479001d6467c9e34af59e44a840f9`, which includes the later fix for
patches that add only new files; it cannot be relabelled as v1 confirmatory
evidence. The Verified dataset is pinned to revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. An update to any confirmatory
identity requires a new protocol version before labels are inspected.

The adapter never lets the official harness resolve a mutable remote dataset.
`prepare` accepts only the official parquet with SHA-256
`a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd`,
validates its exact schema and 500 rows, and writes canonical JSONL snapshots:

- full evaluator snapshot:
  `52ccbc6ec0e03085f95191b261e0ed881cd6a0752a3c5247c1aba258ec2993da`;
- redacted public snapshot:
  `9116deb4b3b24346a278373cf1551bd8cee4e0677776105f62b4474ca50dfaba`.

The confirmatory planner rejects every other public byte stream. Evaluation
accepts only the exact local full snapshot, so a later Hugging Face revision
cannot silently enter the study.

## Frozen comparison

The target is `Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx` in plain,
target-only autoregressive mode. DFlash, DSpark, BMP, and TurboQuant are off in
both arms. This intentionally removes speculative-runtime behavior from the
causal comparison. A full byte identity of the local model tree must be sealed
before the first generation and recomputed after the last generation; a path,
repository label, or `config.json` hash alone is not sufficient.

The exact shared Mio fingerprint is
`local-sha256-v1:ba3975accc6b6398f47f82ff7640b39f5541abb49f1d3c6f34113aa7fb040c87`.
The adapter calls `experimental.effort.model_identity.fingerprint_local_model`
instead of maintaining another identity algorithm. Checkpoints reject any
other digest. The pending generation runner must check it before generation;
`evaluate` checks it after generation and again before publishing its receipt.

Both arms use:

- greedy decoding, 32,768-token context;
- at most 4,096 output tokens per round, 24,576 per arm, 12 rounds, and 1,800
  wall seconds, with at most 32 tool calls per arm;
- identical `bash`, `read`, `write`, `edit`, and `validate` schemas;
- the same confined workspace permissions with generation network disabled;
- one clean Mio commit and one dependency/runtime digest;
- a fresh checkout at the instance's exact `base_commit`.

The gate arm may differ only in gate instructions, gate state, deterministic
feedback, and latest-mutation evidence enforcement. In particular, the off arm
keeps the `validate` tool so the experiment does not confound enforcement with
tool availability.

## Mandatory full-500 paired design

All 500 Verified instances are required, producing 1,000 generation arms.
There is one pair per instance. Pair order is frozen with seed `20260718`:
instances are ranked by a SHA-256 transform of seed and ID, paired arms remain
adjacent, and first-arm order alternates. Exactly 250 pairs run off→on and 250
run on→off.

A smoke may use at most ten instances to test repository setup, patch
application, checkpoints, and official harness compatibility. It must be
labeled `non_evidence_smoke`; it cannot support quality, speed, or promotion
claims, and it cannot be pooled with the 500-pair run. Any partial run is also
non-evidence. Synthetic checkpoints and reports can exercise plumbing, but
cannot produce `confirmatory_complete` while the generation attestation is
unimplemented.

The primary outcome is the official harness `resolved` result. The primary
estimand is:

```text
mean(resolved_gate_on - resolved_gate_off), n = 500 pairs
```

The report includes the four paired cells, the percentage-point difference,
a seeded 10,000-resample task-paired bootstrap interval, and a one-sided exact
McNemar/binomial test. A quality-improvement claim requires the 95% bootstrap
lower bound to exceed zero and the preregistered one-sided exact p-value to be
below 0.05. Practical promotion additionally requires a point estimate of at
least two percentage points and no frozen efficiency guardrail failure.

## Leakage and isolation

Generation and evaluation are separated:

1. A trusted preparer reduces the pinned dataset to exactly
   `instance_id`, `repo`, `base_commit`, and `problem_statement`.
2. The adapter rejects any manifest that includes gold patch, test patch,
   expected test names, hints, difficulty, peer-arm data, or evaluator output.
3. Each arm gets a separate fresh checkout. The other arm is not mounted and
   no Git worktree or mutable prefix state is reused.
4. All 1,000 generations and their content-free trajectories are sealed before
   official evaluation begins.
5. Mio's official prediction is captured from repository state with
   `git diff --binary --full-index`; untracked files and valid binary-only,
   rename-only, and mode-only sections are included. Model prose, Markdown
   fences, and diff-looking assistant text are never parsed.
6. The adapter exports the official three-field JSONL schema:
   `instance_id`, `model_name_or_path`, `model_patch`.
7. The pinned official Docker harness applies and evaluates those predictions
   in isolated instance environments.

Official documentation says the harness uses Docker images to create
reproducible per-task environments, applies each prediction, runs tests, and
grades resolution. It also identifies `run_evaluation` as the main entry point
and documents the three-field prediction schema. Those official operations,
not a Mio-specific evaluator, decide the primary outcome.

Generation repositories are treated as hostile input during patch capture.
Host Git receives a secret-free environment whitelist: inherited `GIT_DIR`,
`GIT_WORK_TREE`, credentials, global/system configuration, optional hooks,
`core.fsmonitor`, external diff, and `textconv` execution are unavailable. The
generation runner must make `.git` and all equivalent repository metadata
completely invisible to the model and every model-facing tool. Trusted commit
and diff capture runs in a separate, non-model-visible namespace; it is never
exposed through the agent shell, even read-only.

The adapter now supports this boundary directly: `capture_git_patch` accepts a
private external Git directory, requires it outside the model workspace with
`0700` permissions, rejects any case-insensitive `.git` entry created under the
visible tree, and invokes Git with explicit `--git-dir`/`--work-tree` paths.
The paired smoke runner provisions that separation and rejects a workspace if
the factory returns state outside its exclusive arm destination or a
case-insensitive `.git` entry appears. Clean-subprocess/container provenance
for confirmatory generation remains pending.

All gold, full, and private artifacts live outside the Mio repository root.
Symlinked path components are rejected. Private directories use mode `0700`
and private files use `0600` where the producing tool permits it. Every official
evaluation gets a new, exclusive, initially empty directory, so an old log or
report cannot be mistaken for output from the current attempt.

## Crash safety and retry policy

Each generation arm is an immutable checkpoint bound to the preregistration,
schedule, Mio commit, full model identity, and runtime digest. Checkpoints are
written to a temporary file, flushed, and published exclusively. Resume skips
only a byte-verified terminal checkpoint. A conflicting checkpoint is a hard
protocol error rather than an overwrite.

Model errors, malformed or empty patches, tool failures, timeouts, and an
incomplete gate are outcomes and are not retried. A process crash, host loss,
telemetry corruption, or verified evaluator-infrastructure failure may be
classified without looking at task success; the entire pair is then rerun with
the same seed and order, and every attempt remains in the private audit ledger.
Official harness errors block confirmatory aggregation until that blinded
classification is complete.

`resume_entries` validates every existing terminal checkpoint before returning
only missing arms. The implemented ledger primitive uses separate
`pair-N/attempt-N` checkpoint directories plus a locked, hash-chained,
append-only event ledger. Retry reasons are restricted to process crash, host
loss, telemetry corruption, or evaluator-infrastructure failure. A retry can
begin only after the prior whole-pair attempt was aborted for one of those
reasons; retry after a completed attempt is forbidden. Legacy completion binds
both checkpoint hashes. Portable completion instead records a domain-separated
digest of each checkpoint SHA-256 and telemetry-sidecar SHA-256. Promotion and
receipt verification recompute that binding, so neither artifact can be
swapped independently. This primitive is integrated in the smoke runner;
confirmatory provenance remains blocked.

Before Docker starts, `evaluate` writes an immutable seal binding the schedule,
exact dataset snapshot, both prediction hashes, clean Mio commit, exact model
identity, runtime digest, content-derived run IDs, command vectors, harness
version/commit, and installed harness-distribution hash. Its receipt adds both
official report hashes and the repeated model check. Aggregation recomputes the
current dataset, prediction, checkpoint and report bindings, requires report
IDs to equal the frozen schedule exactly, and rejects stale or unrelated runs.
Each run ID binds condition, prediction, preregistration, schedule, frozen
1,800-second timeout, full snapshot, harness commit, and installed harness
distribution. Confirmatory timeout changes are protocol errors.

The adapter does not yet attest how checkpoints were generated. Consequently,
confirmatory `evaluate` and `aggregate` fail before host preflight or report
loading. A generation receipt must bind the isolated runner, `.git`-invisibility
proof, all checkpoint and ledger hashes, enforced budgets, and runtime identity
before that block can be removed. The exact efficiency guardrail and official
Docker image digests also remain undefined/unrecorded; both must be frozen and
implemented before the first confirmatory generation. The existing smoke path
remains usable solely as `non_evidence_smoke`.

Raw schedules, IDs, patches, model text, trajectories, harness logs, and
per-instance labels remain private. The publishable result contains aggregate
counts and statistics only. Hashing a known instance ID is not treated as
anonymization, so even per-instance digests are omitted from the public result.

### Portable non-evidence smoke artifacts

Every newly created smoke CLI layout now has an immutable `0600` layout
profile, the exact canonical private runtime manifest captured by automatic
attestation, and one canonical `0600` telemetry sidecar per arm. A sidecar is
bound to the schedule execution index and immutable checkpoint SHA-256. It
contains validated raw round fields, sanitized fixed-vocabulary/hash tool
fields, a closed terminal document, and a closed content-free Quality report.
These attest terminal reason, bounded budget kind, telemetry completeness,
complete wall time, derived status/gate decision, Quality phase and activation,
mutation/revision/snapshot state, obligation vocabulary, and bounded validation
counts. An unstructured model exception is sealed as a non-retryable
`model_error`; its unavailable trajectory and counters are explicitly marked
incomplete instead of being invented. A deadline may exceed the 1,800-second
agent budget by at most the sealed five-second executor-overhead allowance. The
sidecar retains that observed wall value while the legacy checkpoint field is
capped at 1,800 seconds. Larger overruns still require the blocked v2 watchdog
adjudication.

A sidecar contains no task text, assistant text, tool arguments or output,
filesystem path, or patch. Throughput must agree with raw token/nanosecond
accounting whenever counters are observed; censored arms are not throughput
observations. Tool admission, audit, timeout, effect, and final-round topology
are validated against closed rules. The generation receipt binds the runtime
manifest, checkpoint/sidecar pair bindings, and complete telemetry manifest.
The retained SHA-256 commitments are not keyed encryption or anonymization;
the sidecars remain private `0600` artifacts and their digests may still permit
correlation when an input has a small enumerable domain.

`verify_sealed_generation_artifacts` audits those retained bytes without
comparing them with the machine running the audit. This makes a later
cross-process integrity audit possible without falsely claiming that the
current Python, MLX packages, model tree, or Mio checkout is the original
environment. `reattest_current_generation_environment` is a separate,
fail-closed operation; `verify_generation_receipt` performs the sealed-byte
audit and then that current-environment re-attestation. Missing files,
non-canonical JSON, digest drift, permissive modes, symlinks, hard links,
unknown trace fields, and invalid timing/vocabulary all fail closed.

Layouts created before this profile are left untouched and are explicitly
reported as legacy/non-portable. `verify_legacy_generation_artifacts` audits
their retained bytes without claiming current-environment re-attestation.
They cannot be resumed by the smoke CLI or upgraded by inventing the absent
runtime and per-arm trace bytes. Portable resume compares the immutable header
and retained runtime before promotion repair; a missing runtime is rejected
without recreation or an earlier artifact mutation. None of these
integrity guarantees changes the smoke's `non_evidence_smoke` classification
or removes a confirmatory v2 blocker.

## Adapter commands

The adapter is dependency-free on import and does not download a dataset,
load MLX, or start Docker. It deliberately stops at explicit boundaries:

```bash
# Convert the already downloaded, pinned official parquet into exact snapshots.
python3 scripts/bench_swebench_quality.py prepare \
  --parquet /private/test-00000-of-00001.parquet \
  --output-directory /private/mio-swe-dataset-v1

# This must fail closed on the current Mac; run it on the intended x86_64
# official-evaluation host.
python3 scripts/bench_swebench_quality.py preflight

# Build the schedule only from the exact redacted snapshot.
python3 scripts/bench_swebench_quality.py plan \
  --instances /private/mio-swe-dataset-v1/swebench-verified-public-v1.jsonl \
  --state-dir /private/mio-swe-run-v1

# After an external Mio generation runner has atomically recorded every arm,
# export exactly 500 official predictions per condition.
python3 scripts/bench_swebench_quality.py export \
  --schedule /private/mio-swe-run-v1/private-schedule.json \
  --checkpoints /private/mio-swe-run-v1/checkpoints \
  --output-directory /private/mio-swe-run-v1/predictions

# Future confirmatory command: this intentionally fails closed until the
# isolated generation runner and generation receipt are implemented.
python3 scripts/bench_swebench_quality.py evaluate \
  --schedule /private/mio-swe-run-v1/private-schedule.json \
  --checkpoints /private/mio-swe-run-v1/checkpoints \
  --predictions-directory /private/mio-swe-run-v1/predictions \
  --full-snapshot /private/mio-swe-dataset-v1/swebench-verified-full-v1.jsonl \
  --model-path /private/models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --evaluation-directory /private/mio-swe-run-v1/evaluation-attempt-001 \
  --seal /private/mio-swe-run-v1/evaluation-seal.json \
  --receipt /private/mio-swe-run-v1/evaluation-receipt.json

# Aggregate only through the verified receipt. Exact report names contain the
# content-derived run ID printed in the private evaluation seal.
python3 scripts/bench_swebench_quality.py aggregate \
  --schedule /private/mio-swe-run-v1/private-schedule.json \
  --checkpoints /private/mio-swe-run-v1/checkpoints \
  --predictions-directory /private/mio-swe-run-v1/predictions \
  --full-snapshot /private/mio-swe-dataset-v1/swebench-verified-full-v1.jsonl \
  --receipt /private/mio-swe-run-v1/evaluation-receipt.json \
  --gate-off-report /private/mio-swe-run-v1/evaluation-attempt-001/mio-qwen36-27b-gate-off.RUN_ID.json \
  --gate-on-report /private/mio-swe-run-v1/evaluation-attempt-001/mio-qwen36-27b-gate-on.RUN_ID.json \
  --output benchmarks/results/swebench-verified-quality-v1.json
```

`prepare`, `plan`, `export`, and each evaluation attempt require new output
directories. Reusing a directory, placing private data under the repository,
or traversing a symlink parent fails closed.

For a completed portable smoke layout, the official wrapper performs its own
artifact-only generation audit and exports separate private prediction
streams. Every scheduled terminal checkpoint stays in the denominator:
non-empty captured patches are evaluated regardless of terminal status, while
genuinely empty predictions are retained as effective unresolved outcomes.
The receipt preserves both the harness's raw empty/unresolved counts and this
derived all-scheduled denominator, preventing post-hoc failure exclusion.

The wrapper compares every tracked harness byte with the pinned Git tree
without trusting index hints such as `skip-worktree`, disables Git replacement
objects, pins the exact tree object, and rejects every filesystem entry outside
that tree except `.git` and the one selected venv. It fingerprints both the
complete venv and the external Python base prefix recursively (package code,
metadata, standard library, `.pth`, `sitecustomize`, binaries and symlink
targets), then verifies all manifests again after evaluation. Probe and harness
execution use Python `-I -S` with an explicit attested path, so `.pth` files and
user-site customizations are recorded but never executed.

The wrapper also checks local image tag-to-digest bindings and requires the
output directory to be disjoint from every immutable input, using canonical
filesystem spelling and ancestor identities rather than lexical paths alone.
Each arm runs in a fresh working directory. An exit-zero harness result is
rejected unless aggregate and per-instance reports agree on exact IDs, patch
application, completion, test partitions, empty predictions, and resolution.
`--dry-run` executes every preflight and writes the immutable plan without
starting the harness:

```bash
python3 scripts/run_swebench_quality_official_evaluation.py \
  --schedule /private/mio-swe-run-v1/private-schedule.json \
  --generation-layout /private/mio-swe-run-v1/generation \
  --dataset /private/test-00000-of-00001.parquet \
  --harness-root /private/SWE-bench-f7bbbb2 \
  --python-executable /private/SWE-bench-f7bbbb2/.venv311/bin/python \
  --docker-executable /opt/homebrew/bin/docker \
  --docker-context colima-swebench-x86 \
  --image-manifest /private/mio-swe-run-v1/official-images.json \
  --output-root /private/mio-swe-run-v1/official-evaluation-001 \
  --dry-run
```

Remove `--dry-run` only after reviewing that private plan. The wrapper itself
neither downloads nor pulls; missing dataset, checkout, daemon, image, digest,
or generation bytes stop before evaluation. Its offline environment is not a
kernel-enforced network sandbox: the pinned upstream harness currently starts
containers on Docker's configured network. This remains a reproducibility
limitation of the non-evidence smoke and must be closed before a confirmatory
claim. Public stdout contains only hashes and aggregate counts, never patch or
evaluator text.

`commands` remains a diagnostic printer and requires both `--schedule` and
`--full-snapshot`. Manually executing its output cannot create the receipt
required by the confirmatory aggregator. Run IDs are derived from the prediction, dataset,
preregistration, schedule, timeout, harness commit, and installed
harness-distribution identities, preventing old harness logs from being reused
when any evaluated input changes.

The non-evidence runner provides a checkout-confined offline tool bridge,
cumulative token/wall limits, authoritative gate evidence, an integrated
attempt ledger, `ArmCheckpoint` creation, and a verifiable portable receipt.
It does not replace the still-blocked confirmatory clean-subprocess and official
evaluator chain. The adapter fails rather than silently substituting assistant
text, synthetic evidence, or an unisolated host repository.

## Resource and feasibility estimate

The full study has 1,000 agent arms. At the frozen 24,576-token hard cap this
is at most 24.576 million generated tokens. The prior short-workload target-only
measurement was about 18.9 decode tokens/s; dividing only by that rate gives a
361-hour decode ceiling before prefill, tools, repository I/O, and tests. The
aggregate of the frozen per-arm wall caps is 500 host-hours. A planning range
of 50–180 generation host-hours is more plausible if most tasks terminate far below the
token cap, but it is not evidence and must be updated only as an operational
budget after a non-evidence smoke—not by changing the frozen analysis.

For evaluation, the official docs recommend x86_64, at least 120 GB free
storage, 16 GB RAM, and eight CPU cores; arm64 support is described as
experimental. The current machine has ample host disk and 48 GiB unified
memory. A dedicated QEMU/Colima profile supplies an eight-CPU, 32 GiB
`linux/amd64` Docker engine with the two smoke images already pinned by
repository digest. The official checkout and Python 3.11 environment are also
prepared outside the repository. The profile is stopped before 27B generation
to avoid contaminating memory-pressure measurements, then restarted for the
official smoke evaluation. This makes a local non-evidence evaluation
possible; it does not remove the confirmatory protocol blockers or make two
smoke instances representative of all 500 Verified tasks.

## Interpretation boundary

A positive result would support one narrow claim: on one frozen Qwen 3.6 27B
target-only Mio configuration, the mandatory gate improved paired resolution
on this pinned SWE-bench Verified revision. It would not prove a universal
coding breakthrough, intrinsic model improvement, speculative-decoding speed,
or generalization beyond these repositories. A null or negative result is also
publishable and must not trigger task removal, threshold changes, or a hidden
second analysis under protocol v1.
