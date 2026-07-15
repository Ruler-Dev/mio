# Reproducible Mio benchmarks

Mio stores reviewed raw performance evidence as JSON in
`benchmarks/results/`. This page distinguishes published artifacts, current
diagnostic observations, and preliminary research probes. A number observed in
an exploratory run is not a release claim.

## Evidence status

| Evidence | Status | Valid use |
|---|---|---|
| `qwen36-core-256.json` | published schema-v1 artifact | reproduce the historical commit and workload |
| `qwen36-cache-256.json` | published schema-v1 artifact | reproduce the historical cache ablation |
| `qwen36-20260715-192941.json`, `qwen36-20260715-193332.json` | preliminary dirty-tree single-prompt diagnostics | reproduce the vendored exact-verifier bottleneck only |
| `speculative-matched-qwen36-27b-20260715.json` | preliminary dirty-tree 12-pair/4-prompt artifact | compare target AR, upstream DSpark and upstream DFlash |
| Qwen 3.6 DSpark cap-3/cap-4/full artifacts | preliminary dirty-tree ablations | choose a guarded candidate cap; reject unsafe settings |
| fused cold-prefill result under `experimental/` | preliminary dirty-tree 12-pair pilot | motivate stronger TTFT experiments only |
| Qwen3-4B v0.4.1 artifacts | preliminary dirty-tree strict gate failure | diagnose quantized numerical parity; never support a speed claim |

The two published files record commit
`d49dec26dbd6053526027e013d5580e9cf5c10f4` with `git_dirty=false`. They use
schema v1 and predate the current exact verification path and schema-v2 paired
protocol. They remain reproducible evidence for that commit, but their 1.74x
historical DFlash decode ratio must not be presented as the performance of the
current exact engine.

All 15 July matched, cap-sweep, 4B and fused-prefill artifacts were generated
from a dirty working tree and remain untracked at this checkpoint. They are
reported for review and reproduction, but they are not published release
evidence until rerun from the exact clean commit that contains their harness.

## Tested 27B system

- host: Apple M4 Max, 48 GB unified memory;
- OS: macOS 26.5.1 arm64;
- Python: 3.12.0;
- MLX: 0.32.0;
- mlx-lm: 0.31.3;
- dflash-mlx: 0.1.8;
- mlx-dspark: 0.4.1;
- target: `Qwen3.6-27B-UD-Q4_K_XL-mlx`;
- DSpark: `Qwen3.6-27B-DSpark`;
- DFlash: `Qwen3.6-27B-DFlash`.

The hardware block in the existing artifacts records platform and machine but
not the chip name or unified-memory capacity. Those two values were recorded
separately on the benchmark host.

## Current exact 27B status

The exact path now reproduces all 64 baseline token IDs in the exercised
Qwen 3.6 27B checks. It does not currently accelerate decode:

| Path | Observed decode range | Exact token parity |
|---|---:|---:|
| target AR | about 18.7-19.8 tok/s | control |
| current exact DFlash | about 11.6-11.9 tok/s | 64/64 |

These are narrow diagnostic observations, not confidence-bounded benchmark
results. They show that correctness has been restored and that the present
exact implementation is slower than target AR on the exercised trajectory.
They supersede any claim that the historical schema-v1 1.74x ratio describes
the current exact implementation.

Phase telemetry attributes about 5.06 seconds of a roughly 5.65-second DFlash
request to target verification. Verification therefore accounts for about
90% of the observed wall time and is the primary optimization target. Replay,
rebuild, and commit must remain inside their measured phases; moving lazy MLX
evaluation outside a timer would make the attribution invalid.

### Exact Metal weight-staging ablation

The cooperative threadgroup weight-staging QMV kernel remains available only
for controlled reproduction:

```bash
MIO_DFLASH_QMV_STAGING=1 python3 scripts/bench_qwen36_matrix.py ...
```

It preserves exact arithmetic in the exercised bit-width, vector-width, and
tail checks, but it is disabled by default because it regressed real Qwen 3.6
27B kernel time. At T16, the observed increases were 9.3% for `up`, 11.8% for
`down`, and 13.3% for `qkv`; at T5, regressions were approximately 12-20%.
This is a negative ablation, not an optimization claim.

## Matched Qwen 3.6 27B speculative study

The current matched harness loads one target plus both upstream drafters once,
uses four chat-templated workloads, warms every mode, and runs three balanced
repetitions per prompt. Every result below uses 64 generated tokens, 12 paired
observations, prompt-cluster bootstrap intervals, and zero fallback executions.
These are direct `mlx-dspark`/`dflash-mlx` measurements; the DFlash result is
not the throughput of Mio's slower vendored `mio.dflash` production path.

| Candidate | TTFT speedup | Decode speedup | End-to-end speedup | Parity | Gate |
|---|---:|---:|---:|---:|---|
| DSpark cap 2, lookup off | 0.7558x | 1.0730x | 1.0291x | 12/12 | reject: TTFT and E2E lower CI |
| DSpark cap 3, lookup off | 0.7187x | 1.1115x | 1.0580x | 12/12 | reject: TTFT and decode/E2E lower CI |
| upstream DFlash | 0.9098x | 2.3728x | 2.0029x | 12/12 | reject: TTFT regression |
| DSpark cap 4 | 0.7110x | 0.9645x | 0.9365x | 9/12 | reject: parity and speed |
| DSpark full block | 0.7183x | 0.6842x | 0.6910x | 9/12 | reject: parity and speed |

For cap 2, the decode cluster-bootstrap 95% interval is
`1.0312x–1.3123x`, while the E2E interval is `0.9924x–1.1871x`. For cap 3,
the decode and E2E intervals both cross 1.0. Cap 3 is the best parity-safe
point estimate and is therefore the guarded Qwen 3.6 runtime profile, not a
breakthrough claim. Caps of four or more are rejected by the measured parity
gate.

The 4B v0.4.1 reruns are negative controls: DSpark parity is 0.75 and upstream
DFlash parity is 0.50 on 12 pairs, with or without DSpark lookup. Their speed
ratios are invalid for promotion. The divergence is associated with
width-dependent quantized MLX numerics and must not be described as exact
speculative acceleration.

## Fused cold-prefill pilot

The upstream DFlash path normally splits a cold prompt into `prompt_len - 1`
plus a singleton seam even when no prefix snapshot is active. An isolated
prototype uses the existing full-tail path and defers the draft-context
projection until first drafter access. Across four short prompts and three
balanced repetitions (16 generated tokens), it measured:

| Metric | Point estimate | Prompt-cluster bootstrap 95% CI |
|---|---:|---:|
| token parity | 12/12 | descriptive |
| TTFT speedup | 1.1555x | 1.1122x–1.1614x |
| end-to-end speedup | 1.0794x | 1.0657x–1.0938x |

Projection deferral alone was neutral (`1.0043x` TTFT, `0.9969x` E2E); the
measured gain comes from removing the cold singleton seam. This is a
single-thread global patch, covers only 40–71 prompt tokens and 16 output
tokens, and excludes prefix snapshots, PQ/TQ, dynamic tool EOS and concurrency.
It is a promising candidate, **not a production or global breakthrough**.

## Historical schema-v1 workload

The two checked-in artifacts tokenize and repeat a fixed software-engineering
seed until it has exactly 256 tokens, request 32 output tokens, and run one
warm-up plus two measured repetitions per mode in one process. DFlash token IDs
matched target AR in the core artifact.

Timing definitions:

```text
prefill tok/s    = prompt_tokens / prefill_seconds
decode tok/s     = generated_tokens / (elapsed - prefill)_seconds
end-to-end tok/s = generated_tokens / elapsed_seconds
```

End-to-end throughput is a completion-rate metric whose numerator contains
only generated tokens.

### Historical core artifact

| Mode | Prefill tok/s | Decode tok/s | End-to-end tok/s | Acceptance | Peak GB | Parity |
|---|---:|---:|---:|---:|---:|---:|
| target AR | 234.77 | 19.31 | 11.64 | 0 | 25.23 | control |
| DFlash at recorded commit | 232.92 | 33.64 | 15.61 | 0.8125 | 25.25 | yes |

The historical median ratios were 0.992x prefill, 1.743x decode, and 1.340x
end to end. These values describe only the recorded schema-v1 commit and its
short workload.

### Historical cache artifact

| Mode | Prefill tok/s | Decode tok/s | End-to-end tok/s | Acceptance | Peak GB | Parity |
|---|---:|---:|---:|---:|---:|---:|
| target AR | 235.03 | 19.25 | 11.63 | 0 | 25.23 | control |
| DFlash + PQ4 | 231.18 | 24.43 | 13.24 | 0.7813 | 25.24 | **no** |
| DFlash + TQ4 | 184.56 | 20.08 | 10.73 | 0.8125 | 27.57 | yes |

The cache artifact omits unquantized DFlash, so it cannot isolate the cache
format's incremental cost. PQ4 changed the deterministic output. TQ4 preserved
tokens but was slower end to end and did not reduce peak memory on this short
context.

## Benchmark schema v2

The current harness emits `schema_version: 2` and adds the controls needed for
paired analysis:

- a user-visible `--seed`;
- seeded randomized Latin-rotation blocks, so every repetition contains every
  mode and every mode occupies every execution position in a complete block;
- persisted warm-up and measured execution order;
- baseline self-determinism and parity for every mode marked `exact`;
- strict failure when any exact mode differs from its paired baseline;
- per-repetition candidate/baseline ratios and median paired ratios for
  prefill, decode, and end-to-end throughput;
- normalized phase timings for `prefill`, `draft`, `draft_prefill`,
  `draft_incremental`, `verify`, `replay`, `rebuild`, and `commit`;
- cache commit mode, rebuilt-target-token count, and exact-acceptance
  correction count.

The general schema-v2 matrix records paired point estimates. The matched
speculative harness extends that contract with 10,000-sample prompt-cluster
bootstrap intervals, p95 latency and memory gates, explicit fallback counts,
and a nonzero strict exit when any exact candidate loses parity.

## Reproduction

From the repository root with a complete local Qwen 3.6 stack:

```bash
python3 scripts/bench_qwen36_matrix.py \
  --tier large \
  --prompt-tokens 256 \
  --max-tokens 64 \
  --warmup 1 \
  --reps 6 \
  --seed 20260715 \
  --modes baseline,dflash \
  --output benchmarks/results/qwen36-exact-local.json
```

Use a new `-local` filename until the result has been reviewed. Do not
overwrite the published schema-v1 artifacts.

Reproduce the matched target/DSpark/DFlash protocol with:

```bash
python3 scripts/bench_speculative_matched.py \
  --model models/Qwen3.6-27B-UD-Q4_K_XL-mlx \
  --dspark-draft spd/Qwen3.6-27B-DSpark \
  --dflash-draft spd/Qwen3.6-27B-DFlash \
  --max-tokens 64 \
  --warmup 1 \
  --reps 3 \
  --bootstrap-samples 10000 \
  --dspark-max-draft-tokens 3 \
  --output benchmarks/results/speculative-matched-qwen36-local.json
```

Pre-run controls:

```bash
git status --short
python3 -m pip check
python3 -m mio.model_check
python3 - <<'PY'
import importlib.metadata as m
for name in ("mlx", "mlx-lm", "mlx-vlm", "dflash-mlx", "mlx-dspark"):
    print(name, m.version(name))
PY
```

Keep power and thermal state stable, close unrelated GPU-heavy applications,
and record deviations. A clean commit is required for a release claim.

## Breakthrough acceptance criterion

Mio will use the word **breakthrough** only when one candidate satisfies all
of the following:

1. 100% deterministic token parity on exact modes and zero fallbacks;
2. point estimates of at least +5% for both TTFT/prefill and decode;
3. paired bootstrap lower confidence bounds greater than 1.0 for both gains;
4. no material peak-memory, tail-latency, reliability, or quality regression;
5. success on a held-out corpus spanning code, prose, structured output, tool
   calls, and multiple prompt/output lengths;
6. independent replication on matched 4B and 27B target/draft pairs.

Until every condition holds, faster observations are hypotheses or
engineering progress, not a scientific discovery.

## Recommended research matrix

| Dimension | Minimum useful coverage |
|---|---|
| prompt length | 256, 2K, 8K, 32K, largest safe length |
| output length | 64, 128, 512 or workload termination |
| workloads | held-out code, prose, JSON/tool calls, long retrieval context |
| modes | target AR, exact DFlash, DSpark, BMP, DDTree; caches as separate ablations |
| repetitions | balanced Latin blocks sufficient for paired bootstrap intervals |
| state | cold process, warm graph, prefix miss, prefix hit |
| metrics | tokenize, prefill, TTFT, all v2 phases, decode, queue, p50/p95, memory, parity, fallback |

For cache studies, keep the speculative method constant and vary only the
cache format. For prompt policies or Headroom, measure task quality and
retrieved facts in addition to token count.

## Claim boundary

Currently supported:

- Qwen 3.6 27B loads with DSpark preferred and a distinct DFlash fallback in Mio;
- the exercised current exact path preserved all 64 baseline tokens;
- the current exact path is slower than target AR on that diagnostic run;
- target verification dominates the observed exact-path wall time;
- the opt-in Metal weight-staging ablation is exact but slower;
- upstream DFlash has a large direct decode/E2E advantage but regresses TTFT
  and is not yet production-compatible with all Mio semantics;
- the fused cold-prefill pilot improves short-prompt TTFT/E2E with 12/12
  parity and warrants stronger validation.

Not currently supported:

- a current speedup from Mio's vendored exact DFlash path;
- a Qwen 3.6 prefill breakthrough;
- a universal DFlash or DSpark factor;
- PolarQuant zero overhead or exact parity;
- TurboQuant memory savings on long context;
- improved coding-agent quality from MCP, skills, Caveman, Ponytail, or
  Headroom;
- a scientific breakthrough.

## Adding a result

Commit raw JSON rather than terminal summaries. Review that:

- model references are portable and contain no credentials or home paths;
- `git_revision` identifies clean tested code;
- raw repetitions, execution order, token IDs/hashes, and paired ratios exist;
- exact parity and fallback checks pass;
- phase timers include lazy MLX materialization;
- negative and neutral results are documented beside gains.
