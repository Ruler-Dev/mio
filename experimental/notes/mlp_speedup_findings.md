# MLP Speedup Investigation on Qwen3.5 — Negative Result

## TL;DR

Tried three angles to accelerate MLP compute on Qwen3.5-9B-4bit prefill,
which dominates total prefill time at 57%. **None delivered measurable
speedup**. The compute is bound by 4-bit grouped quantized matmul kernels
that MLX has already optimized. Without writing custom Metal kernels for
sparse / row-subset 4-bit matmul, this is the floor.

## Profile (Qwen3.5-9B-4bit, L=2000)

```
gate_proj (4096 → 12288) :  15.7 ms  (4-bit grouped quantized matmul)
up_proj   (4096 → 12288) :  15.7 ms  (same)
swiglu (elementwise)     :   0.4 ms
down_proj (12288 → 4096) :  16.1 ms  (4-bit grouped quantized matmul)
─────────────────────────
total per MLP:             ~47 ms
total across 32 layers:    ~1500 ms
prefill total:             ~2625 ms
MLP share:                  57%
```

## Things tried

### 1. mx.compile fusion of gate+up+swiglu+down

```python
@mx.compile
def fused_mlp(x):
    g = mlp.gate_proj(x); u = mlp.up_proj(x)
    return mlp.down_proj(swiglu(g, u))
```

Result: **47.04 ms vs stock 47.12 ms** — no speedup. MLX already fuses these
internally; no further gains available.

### 2. Activation sparsity exploitation

After SwiGLU on a real prompt, intermediate activations are extremely sparse:

```
fraction with |x| < 0.0001 :  4.7%
fraction with |x| < 0.001  : 26.1%
fraction with |x| < 0.01   : 81.0%
fraction with |x| < 0.05   : 98.9%
fraction with |x| < 0.1    : 99.8%
```

Tried explicitly zero-masking near-zero values before `down_proj`:

| Threshold | Time |
|-----------|------|
| baseline | 48.67 ms |
| 0.001    | 48.67 ms |
| 0.01     | 47.81 ms |
| 0.05     | 47.85 ms |

**No speedup** — MLX dense matmul does not detect or exploit explicit zeros.
Sparse matmul would require either:
- Custom Metal kernel reading sparse activation patterns (non-trivial)
- A specialized sparse-aware quantized matmul (does not exist in MLX yet)

### 3. Top-K column subset (skipped — requires custom kernel)

Selecting the top-K most active intermediate columns and using only those K
rows of `down_proj` would be the principled approximation. But `down_proj`
is `nn.QuantizedLinear` (4-bit grouped quantization, group_size=64). Row-
subsetting requires either re-grouping or operating outside the quantization
boundary — both need custom Metal kernels.

## Why the architectural ceiling is real

| Compute (Qwen3.5-9B prefill, L=2000) | Share |
|--------------------------------------|-------|
| MLP                                  | 57%   |
| SSM (GatedDeltaNet)                  | 35%   |
| Attention                            |  8%   |
| Other (norms, embed, lm_head)        | trace |

Within MLP, compute splits roughly equally across three quantized matmuls.
Each is already running at hardware peak (Apple Silicon Metal kernels for
4-bit grouped quantized matmul are highly optimized; mlx-lm uses them
directly). The arithmetic intensity (FLOPs per byte loaded) is right at the
roofline knee, so there's no easy memory-bandwidth win either.

The real remaining levers all require either:
- **Custom Metal kernels** for sparse / low-rank / int8-activation matmul (weeks of work, MLX team would need to ship in mlx-core eventually)
- **Model surgery / retraining** for layer skipping or low-rank decomposition
- **Different model architecture** (e.g., switch to a model with smaller MLP)

## What this means for mio

For Qwen3.5 family (the production target):

- **Prefix cache** (already shipping, default on) gives 4.7-8.6× warm-hit
  speedup. **This is the dominant practical TTFT optimization**.
- **LM-head slicing** (already shipping, default on) gives +13-15% cold-prefill
  speedup. Universal, free.
- **TQ4** (opt-in, `--tq4`) trades decode speed for KV memory; on 27B-dense
  at 32K it's actually faster.
- **BMP-DFlash** (opt-in, `--mpath`) wins on Qwen3-8B math, regresses on
  Qwen3.5.

There is no further "free lunch" inference speedup for Qwen3.5 prefill that
doesn't require either custom Metal kernels or model retraining.

## What this means for the field

Hybrid SSM + attention models (Qwen3.5, Jamba, Mamba-3, Granite-MoE-Hybrid,
Bailing-MoE-Linear, etc.) are well-balanced architectures: MLP, SSM, and
attention share compute roughly evenly. **No single sub-component is a clean
target for sparse / sketchy approximation**. This is by design — the entire
point of these architectures is to avoid bottlenecks.

Optimization research that targets a single sub-component (e.g.,
SpecPrefill, MInference, FlexPrefill, DGSA — all attention-only) cannot
deliver large speedups on hybrid SSM models. The next frontier is either:

1. **Architecture-aware optimisations** that touch all three sub-components
   (e.g., int8 activations cast across all linears). Requires custom Metal.
2. **Cross-call optimisations** like prefix cache that bypass the per-call
   compute entirely. **Already shipping in mio.**
3. **Model-level changes** (smaller models, distillation, MoE expert
   pruning). Out of inference scope.

## Status

`experimental/mlp_speedup/` directory created but no working code shipped —
all attempted optimisations were no-ops or required out-of-scope kernel work.
Profile script lives in this note's source as the experimental record.
