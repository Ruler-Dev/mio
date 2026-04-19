# DGSA (Draft-Guided Sparse Attention) — A Negative Result on Qwen3.5

## TL;DR

We invented and implemented **DGSA**: an attention-only adaptation of
SpecPrefill that preserves SSM correctness on hybrid (gated delta-net +
attention) models like Qwen3.5. It works correctly. **It does not produce
meaningful speedup**, because attention is no longer the bottleneck on this
architecture. Maximum theoretical speedup ≤ 1.08×.

This is a research-grade negative result with practical implications:
**sparse-attention prefill techniques (SpecPrefill, MInference, FlexPrefill,
DGSA) cannot accelerate Qwen3.5 prefill** — by design.

## What DGSA does

Per prefill step on a hybrid_gdn model:

1. Score per-token importance (by draft attention pattern, by position
   heuristic, or by the target's first attention layer's pattern).
2. Pick `keep_indices` = top-K + first F sinks + last W recent.
3. Forward target on the FULL prompt, but **monkey-patch attention layers**:
   - Compute Q from all positions
   - Compute K, V from all positions, apply RoPE at all positions
   - **Slice K, V to keep_indices BEFORE attention SDPA**
   - Output is dense over query positions
4. SSM (GatedDeltaNet) layers are **NOT patched** — they process the full
   sequence and maintain correct recurrent state.

The KV cache after sparse prefill contains only K entries per attention layer.
Decode appends to it normally.

## Why it doesn't help on Qwen3.5

Profile of Qwen3.5-9B-4bit at L=2000 prompt tokens:

| Component | Per-layer time | Layers | Total |
|-----------|----------------|--------|-------|
| Attention layer (full) | 72 ms | 8  | 576 ms (25%) |
| SSM layer | 72 ms | 24 | 1728 ms (75%) |
| **Total** | | 32 | 2304 ms |

The "attention layer" time INCLUDES q/k/v/o projections, q_norm/k_norm, RoPE,
gate sigmoid, AND the MLP that follows. Of those, the actual `softmax(QK^T) @ V`
softmax+matmul block is a small fraction.

Upper-bound experiments: zero out each sub-component to measure its share of
prefill time on Qwen3.5-9B at L=2000:

| Configuration | Prefill (ms) | Component |
|---------------|--------------|-----------|
| Baseline (everything on) | 2625 | — |
| Zero attention sub-layer | 2427 | Attention = **8%** (198 ms) |
| Zero MLP sub-layer | 1116 | **MLP = 57%** (1509 ms) |
| Implied SSM contribution | ~917 ms | **SSM = ~35%** |

So MLP and SSM dominate. Even ELIMINATING the entire attention sub-layer
compute saves only **8%** of total prefill time. DGSA, which only sparsifies
the inner softmax+matmul (maybe 30-40% of attention sub-layer time), would
save at most ~3% of total prefill in the best case.

Empirical DGSA results on Qwen3.5-9B at varying prompt lengths
(strategy=anchor_strided, keep_first=4, keep_last=64, stride=4):

| Prompt | Kept | Dense (ms) | DGSA (ms) | Speedup | Output correct? |
|--------|------|------------|-----------|---------|---|
| 463    | 167 (36%) | 645  | 620  | 1.04× | ✓ Canberra |
| 913    | 280 (31%) | 1204 | 1198 | 1.01× | ✓ Canberra |
| 1813   | 505 (28%) | 2372 | 2373 | 1.00× | ✓ Canberra |
| 3613   | 955 (26%) | 4827 | 4859 | 0.99× | ✓ Canberra |

Quality is preserved across all sizes (correct answers). Speedup is at
1.0×, exactly as predicted by the upper-bound analysis.

## Why hybrid SSM resists sparse-attention optimization

The Qwen3.5 architecture choice (`full_attention_interval = 4` — only every
4th layer is full attention) was made specifically so that attention is NOT
the dominant compute. SSM layers (GatedDeltaNet) and MLP layers absorb the
budget instead. Sparse attention techniques target softmax/QK^T cost; on
SSM-hybrid models that cost is already small.

This generalizes: any architecture that uses linear-attention or SSM
substitutes (Mamba-2, Jamba, RWKV, Gated DeltaNet, Mamba-3) will exhibit
this same property. Sparse prefill techniques designed for pure-attention
transformers won't transfer.

## What WOULD work on Qwen3.5

Compute is **57% MLP, 35% SSM, 8% attention**. To meaningfully accelerate
Qwen3.5 prefill, attack the MLP+SSM majority:

1. **Hidden-state activation quantization during prefill**: cast `h` to int8
   between layers, MLP and SSM projections in int8. Could give ~2× on the
   compute-bound MLP and SSM linear projections. Requires Metal kernel work.

2. **MLP gating sparsity**: SwiGLU produces ~50% sparse activations. Skip
   the down-projection rows that gate to ~0. Some MLX MoE kernels do this
   for experts; doing it for non-MoE MLP needs custom kernels.

3. **Token-level layer routing** (Mixture-of-Depths): route easy tokens
   through fewer layers. Requires retraining; out of scope.

4. **Prefix cache** (already shipping in mio): store KV state for shared
   system-prompt prefixes. **Delivers 4.7-8.6× wall-clock speedup on warm
   hits across all 4 default tiers** (`papers/prefill-speedups.md` §1.3).
   This is the actual practical win for Qwen3.5.

## Files

```
experimental/dgsa/
├── __init__.py
├── state.py             # thread-local DGSA-active state
├── attention_patch.py   # monkey-patch Qwen3NextAttention with sparse-K path
├── selection.py         # anchor_strided + attention_scored
└── session.py           # DGSASession orchestrator
experimental/notes/
└── dgsa_findings.md     (this file)
```

## Production wiring decision

**Do not wire DGSA into mio production.** The implementation is correct
but the speedup ceiling is 1.08× under the most generous assumption, and
empirically delivers ~1.0×. Keeping the code in `experimental/` for
reference and as a worked example of how attention-side optimisation
interacts with hybrid SSM architectures.

The actual prefill optimisation that lands for Qwen3.5 is **prefix cache +
LM-head slicing** (already production), giving 4-8× warm-hit speedup and
+13-15% cold-prefill speedup respectively.

## Lesson for future work

Before implementing a sparse-attention or attention-only optimization:
**profile what fraction of prefill compute attention actually consumes**.
On hybrid SSM architectures (Qwen3.5, Jamba, Granite-MoE-Hybrid), that
fraction is small by design — and any attention-targeting optimization
will have a low ceiling regardless of how clever the algorithm is.
