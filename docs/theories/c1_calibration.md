# C1 LowRank-QK — calibration results (first pass)

**Hypothesis.** For each attention head h in each layer l, there exists a rank r_{h,l} ∈ [4, 32] such that attention computed via low-rank QK factorization recovers ≥98% of full attention's Frobenius energy.

**Result.** Partial confirmation. Low-rank structure exists, but ranks are larger than the plan's prediction — **normalized to d_head, Q and K need ~27-33% of nominal dimension for 98% energy retention** (not 4-32 / 128 = 3-25% as predicted for Llama).

## Setup

- Target: `large-moe` (Qwen3.6-35B-A3B-UD-Q4_K_XL).
- **d_head = 256** (not 128). 16 Q heads per attention layer, 2 KV heads per attention layer (GQA 8:1).
- 10 attention layers captured during calibration. Model appears to have fewer attention layers than the full 64-layer count suggested elsewhere — need to verify.
- 1 calibration sample, ctx ≈ 4256 tokens. Real coding-corpus-like prompt.

## Method

1. Hook Qwen3NextAttention class `__call__` to invoke `q_proj(x)` and `k_proj(x)` alongside the normal forward, stashing outputs.
2. Reshape to (L, n_heads, d_head) — for Q, take first d_head per head (skip the gate half).
3. Per-head SVD on the (L × d_head) activation matrix.
4. Cumulative Frobenius energy: `cum = cumsum(σ²) / sum(σ²)`.
5. Minimum rank r such that `cum[r-1] ≥ threshold`.

## Raw rank distribution

160 Q heads (16 heads × 10 layers) and 20 K heads (2 heads × 10 layers).

| | @95% | @98% | @99% |
|---|---|---|---|
| **Q**  min/median/p90/max | 11 / 44 / 60 / 71 | 28 / 69 / 87 / 94 | 45 / 88 / 104 / 111 |
| **K**  min/median/p90/max | 30 / 56 / 66 / 68 | 55 / 83 / 91 / 94 | 75 / 100 / 108 / 110 |

Median savings at 98% energy: Q **3.7×**, K **3.1×**.

## Expected prefill speedup

Assuming a per-head-rank-selected implementation retains 98% Frobenius energy and that the quality drop from rank-reduced attention is small (untested here), the attention compute drops by the median savings factor.

**Using the conservative bound (K @ 3.1×):**

| ctx | baseline total ms | baseline attn ms | attn after C1 | saved | % speedup |
|----:|------------------:|-----------------:|--------------:|------:|----------:|
| 4K  | 2,869 ms | 662 ms (23%) | 213 ms | 449 ms | **16%** |
| 8K  | 6,056 ms | 1,648 ms (27%) | 531 ms | 1,117 ms | **18%** |
| 16K | 13,441 ms | 4,773 ms (36%) | 1,538 ms | 3,235 ms | **24%** |
| 32K | 52,115 ms | 22,717 ms (44%) | 7,328 ms | 15,389 ms | **30%** |

These are upper bounds assuming:
- Per-head rank selection works without quality degradation.
- Implementation overhead is negligible (low-rank matmul should be a net win at r=83 vs d=256, but kernel launch costs can eat into the theoretical 3.1×).
- No interaction with RoPE — Q/K are rotated before the QK^T op, and rank reduction would happen pre-rotation. SVD is on pre-RoPE Q/K but attention uses rotated values. This needs verification.

## Caveats before claiming a win

1. **Sample size.** 1 calibration run is insufficient. Multi-sample (at least 3 diverse prompts) needed to see if rank is stable or prompt-dependent.
2. **Context dependence.** Calibrated at 4K. Ranks may shift at 16K or 32K — need to check.
3. **Per-head variance is wide.** Q ranks at 98% span 28-94. Uniform-rank implementation at median=69 loses energy on 50% of heads. A per-head-rank scheme is feasible (just more bookkeeping) but adds implementation complexity.
4. **No quality benchmark yet.** Frobenius energy retention ≠ output token quality. Need to replace attention with low-rank version and measure perplexity / code-quality.
5. **Kernel dispatch.** Low-rank attention requires a fused Metal kernel to actually hit the 3.1× theoretical — a naive two-matmul decomposition has extra memory traffic that kills the win. This is the engineering hurdle.
6. **RoPE interaction.** Above note — rank reduction pre-RoPE may not compose cleanly with post-RoPE attention compute.

## Next steps

- Run multi-sample calibration (3 samples at ctx=4K, 3 at 8K, 3 at 16K) to check rank stability.
- Investigate RoPE composition — can we rank-reduce the *rotated* Q/K instead?
- If ranks stabilize, prototype low-rank attention in MLX (no custom kernel yet — just two consecutive matmuls) and measure quality on GSM8K or HumanEval.
- If quality holds, write a fused Metal kernel to unlock the 3× speed.
