# Phase 0 analysis — measured vs predicted

- git SHA at run: `4e44f9c88ae6` (pre-profiler-commits)
- hardware: `Mac16,5` (M4 Max 128 GB)
- target: `Qwen3.6-35B-A3B-UD-Q4_K_XL-mlx` (21 GB weights + 913 MB DFlash draft)
- config: pq_bits=4 (PolarQuant), tq_bits=16 (off), DFlash draft loaded
- timestamp: 1776774871

## Prediction check

The hypothesis pre-registration (`experiments/phase0_baselines/hypothesis.md`) made four predictions **before** any measurement. Results:

- **P1 (linear share ≥55% below 8K):** PASS — measured avg **70.8%** across {512, 1024, 2048, 4096} warm runs.
- **P2 (attention share ≥40% at 32K):** PASS — measured **43.6%** warm at N=32K.
- **P3 (ms(32K)/ms(8K) > 4 — super-linear):** PASS — measured **8.42×** (exact linear would be 4.00).
- **P4 (cold-warm gap ≤20% at N≥8K):** PASS — N=8192: −1.2%, N=16384: −3.0%, N=32768: −9.7% (warm slightly slower, consistent with thermal/allocation effects at the top of the range).

All four predictions pass. The baseline stands.

## Measurement methodology caveat

My per-layer profiler calls `mx.eval(out)` after each decoder layer to force GPU sync and produce meaningful per-layer wall-clock attributions. This adds ~11 s at N=32K cold (about 23% of total) and ~1.5 s at N=32K warm (3%). Numbers below are from the sync-on run and therefore include this overhead. The natural-service wall-clock (sync-off) at 32K is **35.9 s cold / 50.5 s warm** rather than the sync-on 47.1 / 52.1 s.

Layer-3 attention shows up as a ~10 s outlier at 32K under sync-on, with the other 15 attention layers at ~1.3 s each. This is partly a backlog-drain artifact: layers 0–2 (linear GatedDelta) defer some evaluation, and layer 3 — the first attention layer — is where `mx.eval` forces the drain. **Read per-layer attributions qualitatively, not as independent per-layer costs.** What's reliable: aggregate linear ms vs aggregate attention ms, and total ms.

## Baseline prefill table — warm (sync-on, per-rep-min)

| ctx | tokens | total ms | linear ms | attn ms | linear % | attn % |
|----:|-------:|---------:|----------:|--------:|---------:|-------:|
| 512 | 496 | 672.9 | 528.8 | 141.9 | 78.6% | 21.1% |
| 1024 | 1006 | 698.6 | 526.3 | 170.0 | 75.3% | 24.3% |
| 2048 | 2046 | 1487.4 | 974.7 | 339.2 | 65.5% | 22.8% |
| 4096 | 4098 | 2804.2 | 1784.8 | 662.5 | 63.6% | 23.6% |
| 8192 | 8194 | 6191.0 | 3815.3 | 1648.4 | 61.6% | 26.6% |
| 16384 | 16386 | 14422.7 | 8003.0 | 4773.3 | 55.5% | 33.1% |
| 32768 | 32770 | 52115.3 | 22526.5 | 22717.0 | 43.2% | 43.6% |

## Data-driven attack-vector ranking

Ranked by **absolute ms each block consumes at the contexts mio serves**. Fractional speedups only matter if they attack the biggest block.

| ctx | linear ms | attention ms | other ms | top vector by ms |
|-----|----------:|-------------:|---------:|:-----------------|
| 4K | 1,785 | 662 | 357 | linear (3× more) |
| 16K | 8,003 | 4,773 | 1,646 | linear (1.7× more) |
| 32K | 22,527 | 22,717 | 6,872 | **attention = linear** |

"Other" (6.9 s at 32K) is embedding + per-layer norms + final norm + LM head + MoE router overhead. Worth investigating but ~14% of total is a secondary target.

### Key finding — the plan's prior assumption does not hold on this model

The research program document's FLOPs math is for **Llama-3.1-8B** (L=32, dense attention, dense MLP). It predicts MLP = 77% of linear work. **On Qwen3.6-35B-A3B, measured reality is different:**

- MLP is sparse MoE (3B active of 35B total weights per token). It's not the bottleneck.
- The dominant block is **GatedDeltaNet "linear_attn"** — 48 of 64 layers, per-token recurrent state-space op. **55–79% of prefill at mio's operating contexts.**
- Pure-attention layers are 16 of 64. Quadratic scaling brings them from 21% (N=512) to 44% (N=32K).

**This is a different optimization surface than the plan's E-series was designed for.** The plan's E3 "Simdgroup-Fused SwiGLU" attacks MLP; MLP is not this model's bottleneck. For the mio production tier, the highest-leverage targets in priority order are:

1. **GatedDeltaNet per-layer recurrence kernel.** 22.5 s at 32K. No published work optimizes this kernel on Apple Silicon. Even a 20% improvement = 4.5 s saved.
2. **Attention at long context.** 22.7 s at 32K, growing quadratically. The C-series plan applies — LowRank-QK (C1), BlockSparseLearned (C3), StaticMask (C4). Or: better attention kernels for Qwen3Next's specific head-layout.
3. **A-series (skip layers).** 55-65% of prefill is across 48 linear layers; skipping half is still 4-6 s saved at 4-8K. But requires distillation (weeks of cold-path training).

**For Phase 1, I propose attacking GatedDeltaNet first.** It's the biggest block at realistic contexts (4-16K), and nobody's optimized it for MLX on Apple Silicon. This is the Apple-Silicon-first research lane the plan emphasizes as novel, applied to the actual model we ship.

## Open questions for Federico gate-review before Phase 1

1. **Switch target model to Qwen2.5-7B-Instruct?** The research plan recommends 7B for iteration speed. Sticking with large-moe costs ~30 min per per-theory bench but uses the production tier. Trade-off: real numbers on the serving model, versus faster research loop.
2. **Which block to attack first?** My proposal above is GatedDeltaNet. But the plan's priority matrix has E3 (fused MLP) and E1 (AMX co-prefill) in Phase 1. On this model the plan's E3 attacks a minor block (sparse MoE MLP — already cheap); I'd rather spend Phase 1 effort where the data points.
3. **Phase 0 quality eval.** The plan calls for MMLU, GSM8K, HumanEval, LongBench, RULER. Running all five on Qwen3.6-35B-A3B is substantial compute (several hours each). Recommend restricting Phase 0 quality-lock to: (a) GSM8K 8-shot CoT, 500 problems, ~1-2 h; (b) HumanEval 164 problems, ~30 min. Defer MMLU and LongBench until a theory claims ≥15% speedup.

**Phase 0 cannot self-promote to Phase 1; these need explicit Federico approval.**
