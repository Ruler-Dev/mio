# Phase 0 — Baseline Prefill Profile: pre-registration

**Status:** registered before data collection. Written 2026-04-21 on branch
`prefill-research` at commit `4e44f9c` (prior to any profiling runs on
large-moe). Any result-driven edits are forbidden; a post-hoc deviation
from these predictions must be called out in the writeup, not quietly
amended here.

## Goal

Lock baseline prefill time for `large-moe` (Qwen3.6-35B-A3B-UD-Q4_K_XL)
under mio's default stack (PolarQuant-4 KV + DFlash draft, MLX standard
kernels) at every context size Phase 0 cares about. Produce an honest
breakdown of where wall-clock is spent — linear (GatedDeltaNet) vs.
attention — so subsequent theory work attacks the right thing.

## Concrete measurement

- Tier: `large-moe`.
- Contexts: N ∈ {512, 1024, 2048, 4096, 8192, 16384, 32768}.
- Repeats: 2 per N (rep0 = cold, rep1 = warm).
- Tool: `tools/profile_prefill.py`.
- Output: `experiments/phase0_baselines/results.json` + `docs/theories/baselines.md`.
- Both files stamp the git SHA and hardware model.

## Predictions (to be validated or refuted by the results)

The plan's FLOP math was done for Llama-3.1-8B (L=32, dense attention).
Qwen3.6-35B-A3B is structurally different: 48 GatedDeltaNet layers +
16 full-attention layers (hybrid). Predictions are adapted for this
architecture.

1. **Linear share dominates below 8 K context.** At N ∈ {512, 1024, 2048, 4096},
   the 48 GatedDeltaNet layers should account for ≥ 55% of wall-clock.
   (Attention is quadratic in N but only on 16/64 layers; linear is
   per-token recurrent across 48/64 layers.)

2. **Crossover with attention occurs somewhere in [8 K, 32 K].** By N=32K,
   attention share should reach ≥ 40% and could exceed 50%.

3. **Total prefill scales super-linearly with N past 8 K.** The quadratic
   attention term shows up as a super-linear trend in total ms. Observe:
   ms(32K) / ms(8K) should exceed 4× (if it were purely linear, 4×
   would be exact match).

4. **Cold vs warm gap is ≤ 20% at N ≥ 8 K.** After the first run loads
   Metal kernels and pages weights, warm runs stabilize. The cold-to-warm
   delta should be modest at large N (kernel launch is amortized).

## Pass criteria

This experiment "passes" (i.e. produces numbers we'll rely on in later
phases) when:

- All 14 cells (7 contexts × 2 reps) complete without error.
- Per-layer timing sums to between 85% and 105% of total wall-clock
  (i.e. the instrumentation doesn't lose or overcount significant time).
- Cold and warm reps are both recorded and distinguished.
- JSON and markdown outputs are committed.

## Prediction evaluation

After results land, I will:

1. Check each of (1)–(4) against measured data, state pass/fail for each
   prediction explicitly.
2. Identify the 1–2 most leverage-dense attack vectors based on data,
   not the plan's a priori scoring.
3. Present to Federico for Phase 1 gate approval.

## Negative control

None applicable — this is a measurement experiment, not a theory test.
The "control" is methodological: the same code path without profiling
patches produces equivalent total prefill times (± measurement noise).
Tested implicitly by running the bench harness from `kv-experimentation`
on the same model for the same contexts (baseline numbers in
`docs/kv-experimentation-results.md` at `4e44f9c`).
