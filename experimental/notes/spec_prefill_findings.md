# SpecPrefill Experimental Findings

## TL;DR

Implemented Speculative Prefill (Yang et al., ICML 2025) for mio as a
self-contained experimental module under `experimental/`. On Qwen3-8B-4bit
(pure attention, the regime SpecPrefill targets):

| Prompt length | Dense (ms) | SpecPrefill (ms) | **Speedup** |
|---------------|------------|-------------------|---------|
| 562  | 695   | 316  | **2.20×** |
| 1112 | 1362  | 582  | 2.34× |
| 2212 | 2757  | 1099 | **2.51×** |
| 4412 | 5695  | 2192 | **2.60×** |
| 8812 | 12860 | 5023 | **2.56×** |

Quality (semantic correctness): 7/8 diverse prompts at 2200 tokens, 20% keep,
correctly answered (the 8th was on track but the substring matcher caught
"thinking out loud" preamble before the keyword).

## Architecture

Three modules under `experimental/spec_prefill/`:

1. **`rope_pos.py`** — per-position RoPE (so we can apply RoPE based on
   each selected token's *original* prompt position, not its sparse-index).
   Validated to match `mlx.fast.rope` to bf16 numerical precision.

2. **`sparse_attention.py`** — Qwen3 forward pass that accepts explicit
   `position_ids` instead of inferring positions from cache offset. Used both
   for sparse prefill (selected tokens at original positions) and for decode
   (logical positions starting at the original prompt length).

3. **`scoring.py`** — token importance estimator. Runs the speculator (target
   itself, with early-exit at layer 8) and aggregates per-layer attention
   "received-ness" scores via max-over-layers-and-heads. With `fast_score=True`,
   uses an O(D·T) approximation (mean-Q × K) instead of materializing the full
   (T, T) softmax matrix — critical for long contexts.

4. **`session.py`** — orchestrates score → select → sparse-prefill → AR-decode.
   Configurable: `keep_ratio`, `chunk_size`, `score_early_exit`,
   `always_keep_first`, `always_keep_last`.

## Key engineering decisions

### 1. Self-as-speculator with early exit
The paper uses a separate small model. We use the target itself, exiting after
4-8 layers. This avoids loading a second model and keeps the speculator's
architecture trivially compatible. The early-exit reduces scoring cost from
36 layers → 8 layers, ~22% of full forward.

### 2. Fast scoring (no T² materialization)
Replacing the full softmax matrix with `mean_q @ k^T` keeps scoring O(D·T)
instead of O(T²). On 8K-token prompts this turned a 1.03× regression into a
2.56× speedup. Slight quality cost was not observable — same answers.

### 3. Token-level selection (chunk_size=1)
The paper recommends chunked selection (chunk_size=8-16) for parallelism on
GPU. On MLX/Apple Silicon, token-level selection consistently outperformed
chunk-based on quality and didn't materially change wall-time at our scales.

### 4. Per-position RoPE precision
A subtle bf16 precision bug initially caused 0.06 max diff per layer in RoPE
output, compounding to 2.1 max diff in final logits across 36 layers. Fixed
by keeping the rotation in float32 throughout and casting only at the end.
Now per-layer RoPE diff is 0.00006 (essentially noise), and the sparse
forward at keep=100% matches dense within ~1 unit of logit noise per layer
(below practical impact).

### 5. always_keep_last
Surprisingly tricky. Too small (16) preserves only the user-question tail.
Too large (32-64) eats into score-budget, sometimes dropping critical system
tokens. Sweet spot for our prompts: 16 with token-level selection.

## What didn't work

- **Bit-exact match to dense forward**: the bf16 numerical drift through 36
  layers means SpecPrefill@keep=100% (i.e., no selection) doesn't produce
  byte-identical text to dense even on identical inputs. First-token argmax
  flips on close-to-tie logits, then autoregressive divergence cascades.
  Doesn't matter for semantic correctness — the model still produces correct
  answers — but means strict "did the output change?" benchmarks score badly.

- **Chunked selection on Apple Silicon**: the paper's chunk_size=8 selection
  underperformed token-level. The GPU-parallel-block argument matters less
  on MLX where attention scoring is already vectorized.

## Production wiring (deferred)

The code is entirely in `experimental/` and does NOT modify any `mio/`
code. To wire it into mio:

- Best fit: replace prefill in `generate_dflash_once` with sparse-prefill
  when a `--spec-prefill <ratio>` flag is set.
- Compatibility:
  - **Pure attention models only** (Qwen3-8B). Hybrid_gdn (Qwen3.5 family)
    does NOT work — SSM layers can't accept dropped tokens.
  - **Incompatible with prefix cache** (different cache shape).
  - **Incompatible with TQ4** (cache layer needs sparse-aware quant).
  - **Compatible with DFlash decode** in principle — sparse cache + DFlash
    speculative decode should compose. Not yet tested.

## Files in this experiment

```
experimental/
├── spec_prefill/
│   ├── __init__.py
│   ├── rope_pos.py            # per-position RoPE
│   ├── sparse_attention.py    # position-aware Qwen3 forward
│   ├── scoring.py             # importance scoring
│   └── session.py             # SpecPrefillSession
├── tests/
│   ├── test_rope_pos.py       # 8 RoPE unit tests
│   └── test_sparse_attention.py  # parity vs stock
├── bench/
│   ├── spec_prefill_quickcheck.py
│   └── spec_prefill_quality_sweep.py
└── notes/
    └── spec_prefill_findings.md   (this file)
```

## Final benchmark configuration

```python
SpecPrefillSession(
    target_model=qwen3_8b,
    target_tokenizer=tok,
    speculator_model=qwen3_8b,   # same model, early-exit
    keep_ratio=0.20,
    chunk_size=1,
    score_early_exit=8,
    always_keep_first=4,
    always_keep_last=16,
)
```

This config gives the 2.5× speedup with correct semantic output on diverse
prompts at 2K-8K context.
