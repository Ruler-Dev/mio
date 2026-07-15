# KV-cache modes: PolarQuant and TurboQuant

The target cache grows with attended context and can dominate long-context
memory. Mio contains two experimental quantized formats, PolarQuant (PQ) and
TurboQuant (TQ), plus the unquantized MLX cache. They are mutually exclusive.

## Configuration

`TierConfig` currently initializes:

```text
pq_bits = 4
tq_bits = 16   # off
```

`--tq4` changes the selected tier to:

```text
pq_bits = 16   # off
tq_bits = 4
```

Examples:

```bash
mio --tier large --tq4
mio chat --tier large --tq4
mio serve --tier large --tq4
```

For explicit baseline research, construct a tier with both `pq_bits=16` and
`tq_bits=16`. Do not run both quantizers at once.

## What must be validated

A cache implementation is correct only if it supports the full runtime
contract, including:

- update/fetch and attention masks;
- absolute offsets and sliding windows;
- trim/restore state;
- speculative rollback;
- prefix-state reuse where enabled;
- serialization or explicit exclusion from reusable state;
- accurate memory accounting.

Smaller allocated tensors alone do not prove correctness or speed.

## Qwen 3.6 short-context ablation

The checked-in cache artifact uses the Qwen 3.6 27B pair, 256 prompt tokens,
32 generated tokens, one warm-up, and two measured repetitions.

| Mode | Prefill tok/s | Decode tok/s | End-to-end tok/s | Peak GB | Matches baseline |
|---|---:|---:|---:|---:|---|
| target AR, unquantized | 235.03 | 19.25 | 11.63 | 25.23 | control |
| DFlash + PQ4 | 231.18 | 24.43 | 13.24 | 25.24 | **no** |
| DFlash + TQ4 | 184.56 | 20.08 | 10.73 | 27.57 | yes |

Against the baseline in this artifact:

- `DFlash + PQ4`: prefill `0.984x`, decode `1.269x`, end to end `1.138x`;
- `DFlash + TQ4`: prefill `0.785x`, decode `1.043x`, end to end `0.923x`.

These are composite modes: the artifact does not contain unquantized DFlash in
the same run, so it cannot isolate the incremental effect of PQ4 or TQ4 from
DFlash. The separate core artifact measures unquantized DFlash at 33.64 decode
tok/s, but cross-artifact comparisons are not a randomized paired ablation.

## Interpretation

The present evidence does not support "zero overhead", a universal speedup,
or a memory-saving claim for Qwen 3.6:

- PQ4 produced a different deterministic token sequence in both repetitions;
- TQ4 preserved tokens but was 7.7% slower end to end and reported 2.34 GB
  higher peak memory on this short run;
- 256 prompt tokens are far too few for KV storage to dominate a 21+ GB model,
  so peak process memory cannot establish long-context compression;
- two repetitions provide no useful confidence interval.

The negative TQ4 result is still useful: TQ4 should not become the blanket
performance default based on this workload. The PQ4 divergence requires a
quality/parity policy before it can be used where exact greedy output matters.

## Reproduce

```bash
python3 scripts/bench_qwen36_matrix.py \
  --tier large \
  --prompt-tokens 256 \
  --max-tokens 32 \
  --warmup 1 \
  --reps 2 \
  --modes baseline,pq4,tq4 \
  --output benchmarks/results/qwen36-cache-256-local.json
```

For an actual cache study, repeat at 2K, 8K, 32K, and the largest safe context;
add unquantized DFlash to the same process; record allocated cache bytes per
layer; and evaluate token drift and task quality beyond 32 generated tokens.

Raw evidence: [qwen36-cache-256.json](../benchmarks/results/qwen36-cache-256.json).
