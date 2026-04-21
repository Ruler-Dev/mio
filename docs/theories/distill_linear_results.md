# Week 1-2: post-hoc layer replacement — final negative

## Pre-registered hypothesis (Week 1)

"For some subset of the 40 layers, the layer's function can be approximated
by `layer'(x) ≈ x + W @ x` with R² > 0.9 against real activations."

## Pre-registered pass criteria

1. ≥3 layers with R² > 0.9 on test split — **PASSED** (12 of 40 qualify).
2. Replacement ablation ≥1/4 prompt matches AND avg lcp ≥ 0.5 — **FAILED** (0/4, avg lcp 0.06).

Overall: fit signal strong, deployment signal dead.

## Experimental results

### Linear OLS per-layer fit (Week 1)

Test-set R² on delta prediction `(Y - X) ≈ X @ W`:

| rank | layer | kind | delta R² |
|---:|---:|---|---:|
| 1 | L1 | GDN | 0.993 |
| 2 | L2 | GDN | 0.991 |
| 3 | L3 | attn | 0.980 |
| 4 | L10 | GDN | 0.979 |
| 5 | L4 | GDN | 0.973 |
| ... | ... | ... | ... |
| 40 | L25 | GDN | 0.601 |

Twelve layers at R² > 0.9. Strong per-layer signal.

### Deployment test: replace K top layers with their fitted W

| K | matches (of 4) | avg lcp | prefill delta (ms) |
|---:|---:|---:|---:|
| 1 | 0 | 0.06 | +80 (overhead dominates) |
| 2 | 0 | 0.01 | +125 |
| 3 | 0 | 0.01 | +203 |
| 5 | 0 | 0.00 | +142 |
| 8 | 0 | 0.00 | **−61** |
| 13 | 0 | 0.00 | **−390** (14% prefill win) |

Compute saving is real (−390 ms at K=13). Quality is 100% destroyed at every K. Linear fit's 0.7-2% residual destabilizes DFlash/greedy decoding at the very first token.

Qualitative output inspection:
- K=1 (L1 only): coherent Python code, different wording. *Semantic equivalence, byte diff.*
- K=3: grammar-level collapse ("n-th purposeosest number").
- K=8: total degeneration ("I have a comprehensive LLM (202 202 202").

There is a narrow K=1 band where output is *usable* but byte-different, but prefill savings there (~200 ms per replaced layer) are masked by Python dispatch overhead in our wrapper.

### Nonlinear attempt (Week 2 Path A)

Tried: MLP with SiLU (d=2048, bottleneck=512, 6.3M params), architecture = linear skip + MLP residual, skip warm-started from OLS.

| layer | linear R² | linear-skip + MLP R² after 10 epochs |
|---:|---:|---:|
| L1 | 0.993 | 0.993 |
| L2 | 0.991 | 0.991 |
| L3 | 0.980 | 0.980 |
| L10 | 0.979 | 0.979 |

The MLP residual branch learns **nothing** on top of the OLS solution. The 0.7-2% residual is noise with respect to x alone — there's no nonlinear structure for a wider/deeper model to capture.

### Why no post-hoc replacement can work

A layer's 2% residual depends on *context* — neighboring tokens, sequence position, RoPE rotation state, GDN recurrent state. None of that is captured by `f(x_i)` where x_i is the layer's input at position i.

This is what attention is for. You cannot approximate attention with a per-position function. Rank-99% is as good as it gets.

For generation, that's not good enough.

## Why the prefill savings don't translate

At K=13 we save 390 ms prefill. Output is gibberish. The greedy/DFlash decode:
1. Takes the target's post-prefill logits.
2. Argmax for the first bonus token.
3. Draft proposes K more; target verifies.

If the post-prefill logits are off by even 5% on the wrong dimension, the argmax token flips. Once the first token is wrong, the context diverges, and everything downstream is random.

Replacing layers with W changes the target's logit distribution at the last position. Even a 1% KL shift between baseline logits and replaced-model logits reliably changes argmax when the top two token probabilities are close (which they are, commonly).

## What remains

Two real paths for the "big win":

1. **End-to-end knowledge distillation** (Week 3+). Train a student with fewer layers by minimizing KL(teacher_logits, student_logits) across a corpus. Downstream layers learn to compensate for upstream approximation errors. Standard SwiftKV / MiniLLM approach. Needs: training harness in MLX or PyTorch, 1-10B tokens of calibration corpus, days-weeks of compute.

2. **Drop the post-hoc theory family entirely** and invest in Metal kernel fusion of existing ops (GDN in_proj + conv + recurrence + out_proj fused into one dispatch). No quality risk, ~10-20% on the dominant block, weeks of kernel work but no training.

Post-hoc fitting cannot close the gap. This branch's prefill-research lane ends here without end-to-end distillation or kernel engineering.

## Deliverables committed

- `mio/theories/distill_linear/harvest.py` — activation harvester (30 prompts × 2K × 40 layers)
- `mio/theories/distill_linear/regress.py` — per-layer OLS + SVD + identity baselines
- `mio/theories/distill_linear/swap_ablation.py` — install W into attention/GDN class and test quality
- `mio/theories/distill_linear/swap_sweep.py` — K-sweep (1, 2, 3, 5, 8, 13)
- `mio/theories/distill_linear/mlp_fit.py` — linear-skip + MLP-residual with OLS warm start
- `experiments/distill_linear/harvest/` — 19 GB harvested activations, 40 layers
- `experiments/distill_linear/regression.json` — per-layer rankings
- `experiments/distill_linear/swap_sweep.json` — K-sweep results
- `experiments/distill_linear/mlps2/report.json` — nonlinear attempt, no improvement
- `experiments/distill_linear/hypothesis.md` — pre-registration (pass criteria: fail)
