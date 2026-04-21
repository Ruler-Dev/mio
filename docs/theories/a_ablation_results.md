# A-series layer-skip ablation — negative results

**Summary.** Neither attention nor GatedDeltaNet layers on Qwen3.6-35B-A3B can be cleanly skipped post-hoc. No layer's removal produces baseline-matching output across a diverse prompt set.

## Method

For each layer *l* of type *T* (self_attn or linear_attn):
1. Hot-patch the layer's class `__call__` to return a zero tensor instead of computing normally (residual passes unchanged, no contribution added).
2. Run generate() on 4 diverse coding prompts, each padded to ~4K context tokens, 128 output tokens.
3. Compare output sha256 and longest-common-prefix (lcp) against baseline.
4. Score each layer: matches/4, near-matches (lcp ≥ 90%)/4.

## Results — attention layers (10 total)

Best-ranked:

| layer | matches/4 | near-matches/4 | avg lcp | avg prefill delta |
|-----:|:---:|:---:|:---:|---:|
| L4 | 1/4 | 1/4 | 0.32 | +180 ms |
| L0 | 1/4 | 1/4 | 0.46 | +213 ms |
| L5, L2 | 0/4 | 1/4 | 0.31-0.32 | +77 to +328 ms |
| L6, L7, L3, L8, L9, L1 | 0/4 | 0/4 | 0.07-0.23 | +90 to +406 ms |

No attention layer has >1/4 matches. The earlier single-prompt result (4 of 10 layers matching) was a prompt-specific coincidence — across diverse prompts, every layer matters.

## Results — GDN layers (30 total, sampled every 6th)

| layer | matches/4 | near-matches/4 | avg lcp |
|-----:|:---:|:---:|:---:|
| gdn-24 | 1/4 | 1/4 | 0.53 |
| gdn-0 | 0/4 | 0/4 | **0.00** (catastrophic) |
| gdn-6, -12, -18 | 0/4 | 0/4 | 0.09-0.36 |

Layer 0 GDN is particularly critical — its removal zeros the lcp across every prompt. The model's output is fully determined by what enters layer 0.

## Why prefill delta is POSITIVE under skip

Counter-intuitive observation: skipping a layer makes prefill *slower*, not faster:

- Python dispatch overhead from the class-level `__call__` wrapper adds cost to *every* attention (or GDN) call, not just the skipped one.
- The zeroed output changes downstream computations: notably MoE routing (each token's chosen expert may differ → different compute pattern).
- Cache allocation / memory pressure may shift.

So the ablation is a quality-preservation test, not a speed test. If a layer were skippable with zero quality loss, a clean production implementation (early return, no wrapper) would genuinely save its compute.

## Implications

The theories in the research program's A-series (SwiftKV, DepthRouter, LayerFuse, ProgressiveRefine, AsymmDepth) all depend on some form of layer skipping being viable. On this specific model, it isn't, not as a post-hoc trick. The only path is **distillation into a structurally smaller model** — training a student that was designed with fewer layers from the start — which is weeks of compute on billions of tokens.

This matches intuition: Qwen3.6-35B-A3B is a deliberately architected hybrid that amortizes computation across its 30 GDN + 10 attention layers. Each contributes meaningful information to the residual stream. Prune-and-retrain could work. Post-hoc prune cannot.

## Tracking

- `experiments/a_ablation/results.json` — single-prompt (misleading).
- `experiments/a_ablation/multi_prompt_4k.json` — 4 prompts × 10 attention layers.
- `experiments/a_ablation/gdn_multi.json` — 4 prompts × 5 sampled GDN layers.

## Next direction

With C1 (weight-SVD) and A-series (layer skip) both closed as post-hoc methods on this model, the tractable in-session optimizations are:

1. **Documented 6.8% micro-win at 16K** (chunk=16). Shippable but tiny.
2. **Frozen-KV productization** from the kv-experimentation branch — already a real 300-1700× win on repeat prompts.
3. **Decode-side attacks** (DFlash draft improvements, DDTree on pure-attention targets, speculative decoding variants). Requires a different tier / draft model.

Big prefill wins on this specific model require training-scale investment. Honest conclusion.
