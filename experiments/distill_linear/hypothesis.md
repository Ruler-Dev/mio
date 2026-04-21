# Week 1 — Linear Replacement Probe: pre-registration

**Status:** registered before data collection. Committed to
`prefill-research` branch at the current HEAD before any harvest or
regression runs. Results-driven edits forbidden.

## Context

Phase 0/1 closed three post-hoc theories (C1 weight-rank, A-attention-skip, A-GDN-skip) as dead on Qwen3.6-35B-A3B. Every layer contributes meaningfully; no layer can be zeroed without quality loss.

Open question: **can some layers' function be approximated by a cheap learned replacement?** Specifically, for each layer *l*, does there exist a matrix W_l ∈ ℝ^{d×d} such that

    layer_l(x) ≈ x + W_l @ LayerNorm(x)

with high fidelity on real activations? If yes, those layers can be replaced with a single matmul at inference — much cheaper than full attention (4 projections + SDPA + output projection) or full GDN (conv1d + projections + recurrence + output projection).

## Method

1. **Harvest.** Hook each decoder layer to capture `(x_in, x_out)` where `x_in` is the residual stream at the layer input and `x_out` is the residual stream at the layer output. During a prefill pass on 30 diverse prompts at ctx≈2K each, record pairs. Total: ~60K tokens × 40 layers = 2.4M (x_in, x_out) training examples.
2. **Fit.** Per layer, split 80/20 train/test. Fit `W = (X_train^T X_train)⁻¹ X_train^T Y_train` by ordinary least squares. Compute residual `r = y_out - x_in - W @ LN(x_in)`.
3. **Report.** Per layer:
   - R² = 1 − ||r||² / ||y_out − mean(y_out)||² on the test split.
   - MSE on test.
   - Rank at 95% singular energy of W.
4. **Ablation.** Take the top-K layers by R². Replace their `__call__` with `x + W @ LN(x)` at inference. Run the same 4-prompt multi-context quality test used for A-series ablation.

## Predictions

**P1.** Attention layer(s) near the residual stream boundary (first and last) have higher R² than middle layers. Rationale: early layers do "input shaping" which is structurally closer to a linear transform; middle layers do complex token interactions.

**P2.** GDN layers are HARDER to approximate linearly than attention. Rationale: GDN is a recurrent state-space operator — its function integrates over the sequence. A linear projection can't capture position-dependent integration.

**P3.** At least 3 of 10 attention layers will have R² > 0.9. If zero layers cross this bar, linear replacement is dead on this model and we move to MLP replacement next week.

**P4.** Prefill speedup from replacing the top-K layers (where K = count of layers with R² > 0.9) will be proportional to K × per_layer_fraction, roughly 2-15% depending on how many layers qualify.

## Pass criteria

The experiment **passes** and moves to Week 2 in "linear works" mode iff:
- At least 3 layers have test R² > 0.9.
- Replacement ablation on those layers shows ≥ 1 of 4 prompt matches and avg lcp ≥ 0.5.

It **fails** (triggering MLP upgrade) iff:
- Fewer than 2 layers have R² > 0.9, OR
- Replacement ablation shows avg lcp < 0.3 across prompts for top-K layers.

## Negative control

Fit an identity mapping (W = 0, output = x_in) and report its R² on the same test set as a baseline. A learned W must beat this floor. If identity R² is already high (say 0.7+), then layers are mostly residual passthrough and "replacement" is near-trivial but also near-useless for compute savings (since identity == no compute).

## Compute budget for this week

- Harvest: 1 prefill pass × 30 prompts × ~30s each = 15 min on M4 Max.
- Regression: per-layer OLS on 80K samples × 40 layers ≈ 5 min CPU.
- Ablation: (K+1) × 4 prompts × 1 generate = 10-30 min.

**Total: ~1 hour of compute.** Session-compatible. If results are positive, Week 2 uses the harvest data to train better projectors.
