# Week 3 — Compensatory Fine-Tuning Probe: pre-registration

**Status:** registered before training. Commit at `7e3c587` HEAD. No
results-driven edits.

## Context

Week 1-2 established that post-hoc OLS and OLS+MLP replacement of a layer
produces high per-layer R² (up to 0.993) but destroys generation quality
at K=1 (0/4 prompt matches, avg lcp 0.06). The 2% residual per layer is
context-dependent (attention/GDN state) and not capturable by any
per-position function of the layer's input.

The path forward is end-to-end distillation: downstream layers learn to
compensate for upstream approximation errors. But full student training
is weeks of compute. Before committing to that, I need to know: **does
compensation work at all on this model?**

## Experiment

Install the **fitted OLS W at a single early layer (L1)** as in Week 2.
Then **fine-tune a small downstream parameter set** (post-attention
layernorms across the remaining 38 layers, ~40K params — tiny) using a
**logit-matching KL loss** against the frozen original model on a small
calibration corpus.

If the downstream fine-tune recovers quality on the 4 held-out test
prompts (avg lcp ≥ 0.5, or matches ≥ 1/4), then compensatory training
works and full Path B is viable. If quality stays broken (lcp < 0.2),
the compensation mechanism is structurally limited and we need full
joint training of many parameters.

## Method

1. Load large-moe target model.
2. Install OLS-fitted W at layer 1 (the top-R² layer from Week 1).
3. Load the teacher target as a separate model (same weights, no patch)
   for the distillation loss comparison.

   Actually: simpler to avoid loading two full models in 128 GB. Use a
   **single model, captured baseline outputs**: for each calibration
   prompt, pre-compute and cache the teacher's final logits (or full
   hidden state pre-LM-head) before installing the patch. Fine-tune
   against the cached outputs.

4. Select trainable parameters: **only** `post_attention_layernorm.weight`
   on layers 2..39. That's (40 - 2) × 2048 = ~78K scalar params total.
5. Training:
   - Loss: MSE on pre-LM-head hidden state (proxy for logit-KL), or KL
     on final token logits across the prompt sequence. Pick hidden-state
     MSE for speed; it's a strong proxy.
   - Corpus: the 30 calibration prompts from Week 1 (same as the
     harvested data — no held-out contamination because the quality test
     uses a separate set of 4 prompts).
   - Optimizer: Adam, lr 1e-4.
   - Epochs: 3-10. Stop early if loss plateaus.
6. Quality test: run generate() on the same 4 prompts (fib_memo,
   binsearch, list_dedupe, class_bst) used in Week 2 swap ablation.
   Compare sha / lcp to teacher baseline.

## Predictions

**P1.** Loss drops monotonically during fine-tuning (sanity check — if
   loss doesn't decrease, the gradient pipeline is broken).

**P2.** Post-LN-only fine-tuning will provide **some** quality recovery
   but probably not full. Prediction: at best 1-2 of 4 prompts match
   after fine-tune, avg lcp 0.3-0.5. Reason: layernorm parameters are a
   small compensation surface; they can rescale but not restructure.

**P3.** Fine-tuning *later* layers (closer to LM head) will contribute
   more compensation than early layers. Measurable via layer-wise loss
   attribution after training.

**P4.** If P2 holds (some recovery), then fine-tuning a LARGER
   parameter set (e.g., post-attention MLP down_proj weights on
   layers 2..39) will recover more quality. We don't test this in this
   experiment but note the extrapolation.

## Pass criteria

Pass (→ Week 4: scale up to full parameter fine-tune):
- Loss descends at least 50% from initial value.
- Quality test: avg lcp ≥ 0.3 AND ≥ 1 of 4 prompts shows lcp ≥ 0.8.

Fail (→ abandon compensation-style distillation, design a full joint
training from scratch):
- Loss plateaus without descending, OR
- Quality test: avg lcp < 0.2, 0/4 matches.

## Negative control

**Random initialization control:** before training, randomly perturb the
post-LN weights by a small amount. If the quality is no worse than the
pre-training state, then the layernorm parameters aren't actually
meaningful for compensation — we're just getting lucky with layernorm
invariance. The learned fine-tune must beat random perturbation.

## Compute budget

- Pre-compute teacher outputs on 30 prompts: ~1 min (already have
  them from Week 1 harvest, just need the final hidden).
- Training: 30 prompts × 2K tokens × 3-10 epochs. Per-iter: forward
  pass + backprop on LN params only. MLX can handle this on M4 Max in
  seconds per epoch. **Total: 5-15 min.**
- Quality test: 4 prompts × 1 generate = ~10 min.

**Session-compatible.** One session's work.

## Out of scope

- Full student training (distinct experiment; do only if this passes).
- KV cache handling in the distillation loss (use logits / final hidden
  as target, no cache path changes).
- Quality benchmarks beyond the 4-prompt lcp test (MMLU/GSM8K deferred
  until the method shows signal).
