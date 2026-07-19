# Mio benchmark artifacts

This directory contains published historical measurements and preliminary R&D
artifacts. The canonical methods, formulas, result status, and claim boundary are in
[`docs/16-benchmarks.md`](../docs/16-benchmarks.md).

## Coding quality gate

[`coding-quality-preregistration-v1.json`](coding-quality-preregistration-v1.json)
freezes the gate-off/gate-on MioCodeBench v1 comparison before execution. It
uses one implementation commit and one target/DFlash stack for both arms, 4
smoke tasks, 8 development tasks and 16 held-out pairs, generate-then-evaluate
leakage isolation, paired statistics, explicit claim gates, and content-free
evidence. The readable protocol is
[`docs/19-coding-quality-gate.md`](../docs/19-coding-quality-gate.md).

The protocol has now been executed for its 4-pair smoke and one-use 8-pair
development split. Published source-free results:

- [`results/miocodebench-quality-v2-smoke-278f294.json`](results/miocodebench-quality-v2-smoke-278f294.json): `3/4` versus `3/4`, neutral;
- [`results/miocodebench-quality-v2-development-278f294.json`](results/miocodebench-quality-v2-development-278f294.json): `2/8` versus `2/8`, with Quality cost ratios of `2.5444x` wall, `2.6413x` model time, and `2.1858x` output tokens.

The development result fails the frozen go/no-go gates. No coding-quality or
speed improvement is supported, and the planned follow-up 27B smoke is stopped.

The next repository-level allocation experiment is separately frozen in
[`repository-quality-four-arm-preregistration-v4.json`](repository-quality-four-arm-preregistration-v4.json).
It compares a shared Quality root, an allocation-matched static recovery, and
an exploratory depth-one public-state router. The smoke and already observed
development tasks remain calibration-only; this preregistration is not a
result or a Markov quality claim. The protocol reports raw workspace-evaluator
correctness separately from trajectory/compliance success, so validation-only
improvements cannot satisfy its engineering advancement gate.

The first v2 smoke attempt aborted before routing, hidden evaluation, or result
publication because DFlash streaming lacked the preregistered raw phase-time
provenance. Its [post-hoc incident record](incidents/repository-quality-four-arm-v2-smoke-aborted-8bf6e6e.json)
is not a result. V3 retained that check, added yield-exclusive raw DFlash
timing, and completed eight direct roots before aborting at the sealed
allocation boundary: terminal agent bookkeeping had crossed a wall budget
without recording exhaustion. The exact v3
[start](incidents/repository-quality-four-arm-v3-smoke-attempt-start-16213e2.json),
[abort](incidents/repository-quality-four-arm-v3-smoke-abort-16213e2.json), and
[incident](incidents/repository-quality-four-arm-v3-smoke-incident-16213e2.json)
artifacts contain no hidden outcome or aggregate. V4 freezes a same-sample
terminal wall-time/exhaustion fix without increasing budgets or weakening the
validator; it is still a preregistration, not a result.

The terminal-accounting change touches the transitive HumanEval verifier
source bundle, so its parity certificate was regenerated rather than silently
reused. The clean source-bound
[`164/164` certificate](results/humaneval-verifier-parity-ca3cbcb.json) has
SHA-256 `43c36131409f8edb132ab2fada88d17bcf9e203c3d6dfacadca1d70f0e8e4c6b`;
it certifies verifier parity only, not model accuracy or adaptive-policy gain.

The single v4 smoke is now published as an exact source-free
[result](results/repository-quality-four-arm-v4-smoke-f5d04dc.json) with its
[create-once start receipt](results/repository-quality-four-arm-v4-smoke-attempt-start-f5d04dc.json).
All four arms passed `3/4`; the single routed/static recovery was not selected,
and Markov had zero gain over Quality at `1.2648x` logical wall cost. The frozen
promotion gate failed, so this is a harness-validation result only: no larger
cohort and no quality, speed, or breakthrough claim are authorized.

## Qwen 3.6 matrix harness

`scripts/bench_qwen36_matrix.py` compares:

- `baseline`: exact target autoregressive decoding;
- `dflash`: exact unquantized DFlash;
- `pq4`: DFlash with PolarQuant 4-bit target KV cache;
- `tq4`: DFlash with TurboQuant 4-bit target KV cache.

Schema v2 runs seeded randomized Latin-rotation mode blocks, persists warm-up
and execution order, verifies baseline determinism and every mode marked
`exact`, and stores paired candidate/baseline throughput ratios. It also
records normalized phase timings, cache commit mode, rebuilt target tokens,
and exact-acceptance corrections.

Example:

```bash
python3 scripts/bench_qwen36_matrix.py \
  --tier large \
  --prompt-tokens 512 \
  --max-tokens 64 \
  --warmup 1 \
  --reps 8 \
  --seed 20260715 \
  --modes baseline,dflash,pq4,tq4
```

Exact modes fail strict parity when their token IDs differ from the paired
target AR result. Lossy cache modes record parity without being classified as
exact; their divergence must still be reported.

## Published schema-v1 artifacts

- `results/qwen36-core-256.json`: historical target AR versus DFlash at commit
  `d49dec26`; exact token parity and 1.74x median decode throughput for that
  implementation and two-repetition short workload.
- `results/qwen36-cache-256.json`: historical cache ablation; PQ4 diverged,
  while TQ4 preserved tokens but was slower end to end.

Both files record the full clean commit
`d49dec26dbd6053526027e013d5580e9cf5c10f4`. They predate the current exact
verification path and schema-v2 paired protocol. Do not use their 1.74x ratio
as a claim about the current exact engine.

## Current research status

The exercised Qwen 3.6 27B exact path preserved all 64 target tokens but
decoded at approximately 11.6-11.9 tok/s versus 18.7-19.8 tok/s for target AR.
Verification consumed about 5.06 seconds of a roughly 5.65-second request.
The opt-in `MIO_DFLASH_QMV_STAGING=1` Metal ablation remained exact but slowed
the measured T16 kernels by 9.3-13.3% and T5 kernels by about 12-20%.

The current matched Qwen 3.6 27B study uses four prompts and 12 pairs. DSpark
cap 2/cap 3 and upstream DFlash preserve 12/12 paired outputs, but each
regresses TTFT; cap 4/full DSpark preserve only 9/12. Upstream DFlash's direct
decode/E2E gains do not describe Mio's vendored runtime. The Qwen3-4B v0.4.1
reruns also fail strict parity (DSpark 0.75, DFlash 0.50). None is a
breakthrough claim.

## Review policy

Do not overwrite a published artifact with a rerun. Write a new file, review
its provenance, and update the docs in the same checkpoint. Never commit
credentials, absolute home paths, secret-bearing prompts, or terminal-only
numbers without raw repetitions.

A breakthrough claim additionally requires 100% parity, zero fallbacks, at
least +5% point estimates for both TTFT/prefill and decode, paired-bootstrap
lower bounds above 1.0, no material memory or tail-latency regression, a
held-out corpus, and independent 4B and 27B replication.
