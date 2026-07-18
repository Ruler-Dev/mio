# SWE-bench Verified quality protocol v2

Protocol v2 is a machine-readable, **hard-blocked** extension of the v1 paired
Qwen 3.6 27B quality-gate study. It inherits v1 by canonical SHA-256
`834b205733c02a81adaa8ad1cbfd3ab66bdb65575fc162742c53af246422d708`.
Dataset, model, 500-pair schedule, target-only comparison, and existing quality
claim gates are unchanged.
The complete canonical v2 preregistration is sealed as
`bf9429105651d5c06d14720ba9fe096c78197319072e7b9fef6de86984e0b1e2`;
every top-level block is validated exactly.

The v2 preregistration adds controls that must exist before any confirmatory
generation or official evaluation:

- a 1,800-second administratively censored hard wall per arm;
- exact per-operation timeouts and no retry for model/tool/test timeouts;
- distinct termination reasons for 30/300-second tool timeouts, the 600-second
  model-round timeout, and the separately censored 1,800-second arm wall;
- at most three whole-pair attempts, with no retry after completion;
- integer-nanosecond, content-free wall/prefill/decode/tool telemetry;
- supervisor-observed chain-of-custody from the complete model-tree fingerprint
  to the target actually loaded by `NativeMioArmExecutor`, and from the clean
  repository commit to the code and import origins actually executed;
- paired-bootstrap upper-95% efficiency limits of 1.25 wall, 1.10 prefill
  cost/token, and 1.05 decode cost/token;
- a 500-record offline `linux/amd64` Docker image lock, with registry manifest,
  config, layer, local image ID, and RootFS identities;
- official harness image reuse through `cache_level=instance`, `clean=false`,
  `force_rebuild=false`, and the local `mio-swe-v2-locked` tag.

The machine-readable source is
[`benchmarks/swebench-quality-preregistration-v2.json`](../benchmarks/swebench-quality-preregistration-v2.json).
Pure offline validators and guardrail aggregation live in
[`scripts/bench_swebench_quality_v2.py`](../scripts/bench_swebench_quality_v2.py).
They do not load MLX, invoke Docker, generate patches, run the official harness,
or produce benchmark evidence.

The retry helper only classifies an allowlisted reason. It always returns
`admissible_for_retry=false`; authority requires both a trusted-supervisor
incident receipt and a sealed pair-attempt ledger. Likewise, offline Docker
functions are named and returned as `syntactic_non_evidence`. They can bind an
exact expected instance-digest population, but cannot attest daemon state,
materialization, or evaluation-time image reuse.

## Promotion rule

Promotion requires every inherited v1 quality condition plus every v2
efficiency condition:

```text
paired quality bootstrap lower bound > 0
AND exact one-sided McNemar p < 0.05
AND resolution difference >= 0.02
AND paired-bootstrap U95(wall ratio) <= 1.25
AND paired-bootstrap U95(prefill cost/token ratio) <= 1.10
AND paired-bootstrap U95(decode cost/token ratio) <= 1.05
```

Missing or phase-censored prefill/decode telemetry fails efficiency promotion
closed. It does not erase a valid frozen patch or transform a model timeout into
an infrastructure retry.

Round `completion_tokens` counts physically decoded tokens, including a token
computed before a censored deadline but not delivered to the agent loop. Arm
`output_tokens` is the delivered/budget count and may consequently be lower;
the decode-cost guardrail uses physical `decode_tokens`. Round phase durations
must have `timing_source=runtime_raw_ns` and a positive effective timeout no
larger than the frozen 600 seconds. The adapter mapping is normative:
`AgentRoundTrace.physical_decode_tokens` becomes
`RoundMetricRecord.completion_tokens`, while the arm's delivered/budget
`output_tokens` is the sum of `AgentRoundTrace.completion_tokens`. The similar
names must never be mapped in the opposite direction. Each tool telemetry row
represents exactly one admitted invocation, never one row per surrounding audit event.
Requests rejected before admission by the dispatcher budget produce no tool
record because no invocation occurred. `output_chars` counts the visible,
post-cap result, including denial, unknown-tool, and error messages. The frozen
outcome vocabulary includes `untrusted_executable` for validation runners that
resolve inside a writable workspace. `not_found` and `old_string_not_found` are
frozen as allowed, visible, nonterminal results rather than dispatcher failures.
Name/outcome and exit-status combinations are frozen bidirectionally; file
tools carry no process exit status, and `unrecognized` is exclusive to the
`unknown` sentinel. No arbitrary outcome string is admissible. Tool targets use
HMAC-SHA-256. The protocol requires a private random per-run key that is never
serialized, but the pure adapter only checks that caller bytes have sufficient
length and cannot attest randomness, privacy, uniqueness, or supervisor
custody. Therefore `supervisor_private_per_run_hmac_key_not_attested` remains a
blocker. Tags are intended for within-run grouping and are neither public
commitments nor authenticity evidence.
The adapter derives the tag as
`HMAC_SHA256(private_per_run_key, AgentToolTrace.target_sha256)`. Unknown tool
names are serialized only as `tool_name=unknown`, `outcome=unrecognized`, and
`allowed=false`; the requested name never enters frozen telemetry.

The adapter is explicitly shape-only and non-authoritative. Before mapping a
known bounded file tool, it requires
`AgentToolTrace.timeout_enforced=true` and separately requires complete
telemetry. The flag itself is not serialized and is not evidence. `bash` and
`validate` are rejected unconditionally by this adapter while their preflight
and hygiene work is not entirely inside the terminable watchdog, so
`known_tool_full_invocation_watchdog_not_attested` remains an activation
blocker and those traces must fail closed.

The pure bundle validator checks counts, terminal timeout relationships, the
one-second process kill grace, nondecreasing tool-round indices, terminal tools
in the final round, no event after a terminal timeout, and physically possible
sequential duration sums. It cannot reconstruct start/end ordering without a
trusted supervisor timeline, so `supervisor_monotonic_timeline_not_attested`
remains a blocker.

The current `promotion_decision` helper validates shape and may expose formula
outputs only under a `shape_only_formula_outputs` label. Its caller-supplied
counts, bootstrap metadata, and checksum are not authority or receipt binding,
so every admissible criterion, `mathematical_criteria_met_unverified`, and
`promote` remain false. A future activation must recompute and bind every input
inside a trusted receipt path before a promotion verdict can become admissible.
The aggregate's canonical SHA-256 field is only an accidental-corruption
checksum, not a signature, commitment, receipt, or evidence. A caller can
relabel metadata and recalculate that checksum; the hard blocker
`promotion_input_authority_and_receipt_binding_not_implemented` prevents such
shape from becoming evidence.

## Activation boundary

`confirmatory_activation.enabled` is false. The implementation always raises
before confirmatory use. Activation requires a future clean protocol commit,
an independently observed model/code/import chain-of-custody, attested hard
watchdog and telemetry, completed whole-pair retry integration, a real
non-placeholder x86_64 Docker lock, locked-image harness flags, and a
deterministic evaluator timeout classifier. The existing
`v2_clean_commit_and_runtime_not_sealed` blocker remains unresolved. This
document reports no generation, evaluation, quality, or performance result.
