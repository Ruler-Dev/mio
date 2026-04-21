# Mio Prefill Research — Status

**Single source of truth.** Updated at every session end. Read at session start.

## Current phase

**Phase 0 — Foundation.** Non-skippable. Must complete before any theory code.

## What's done

- Repository layout created: `mio/theories/`, `experiments/`, `docs/theories/`, `tools/`.
- Branch: `prefill-research` (forked from `kv-experimentation` at 4e44f9c).
- Target model locked: `large-moe` = Qwen3.6-35B-A3B-UD-Q4_K_XL (21 GB target + 913 MB DFlash draft). Federico-approved via earlier benchmark work.
- Prior numbers (from kv-experimentation branch, documented in `docs/kv-experimentation-results.md`):
  - Baseline decode tok/s: 127.9 (4K) / 91.7 (16K) / 36.0 (32K), mean across 4 coding prompts.
  - Baseline prefill ms: 3102 (4K) / 12966 (16K) / 39325 (32K), mean across 4 coding prompts.
  - DFlash acceptance: 4.0–11.3 tokens/cycle.
  - Frozen KV (C3): 326× (4K) → 1734× (32K) warm-vs-cold prefill speedup on sort_bug.
  - DDTree (ddtree-mlx port): 0.03–0.42× vs baseline — does not work on hybrid model, killed after 3 cells.

## Latest state

**Phase 4 (Path C productization) in progress.** All Phase 1-3 research landed positive: 5 of 10 attention layers in Qwen3.6-35B-A3B are fully spliceable with byte-exact output preservation. Projected prefill speedup: **12% at 4K → 22% at 32K**, scaling with context size.

Branch: `prefill-research`. Never merged to main (user direction).

## Path C results summary

- **Phase 1 (K_base context-robustness):** 4/4 candidate chunks pass splice-safety threshold. Early attention layers (L3-L11) nearly context-invariant.
- **Phase 2 (RoPE math):** implementation sanity bit-exact (rel RMSE 0.0005). Splice error scales with layer depth, matching Phase 1 variance under sqrt transformation.
- **Phase 3 end-to-end:** K=1 single-layer splice preserves semantic output (paraphrase drift on tied logits). **K=5 splice [L3,L7,L11,L15,L19] produces byte-exact match with fresh baseline.** K=6+ adding L23 breaks it — structural boundary at mid-point of attention stack.

Full writeup: `docs/theories/path_c_results.md`.

## Prior phase history

**Phase 0 complete — baselines locked.**

**Phase 0 deliverables shipped:**
- `tools/profile_prefill.py` — per-layer wall-clock profiler (class-level patching with id-keyed slot map so it works under MLX's method-resolution).
- `tools/analyze_phase0.py` — prediction-check + attack-vector analysis from JSON.
- `experiments/phase0_baselines/hypothesis.md` — 4 predictions pre-registered before any run.
- `experiments/phase0_baselines/results.json` — full 7 contexts × 2 reps, per-layer timings.
- `docs/theories/baselines.md` — locked baseline numbers.
- `docs/theories/phase0_analysis.md` — prediction check (4/4 PASS), attack-vector ranking, key findings.

**Baselines locked** for large-moe (Qwen3.6-35B-A3B, PQ-4, DFlash):
- Warm prefill: 673 ms @ 512 → 2.8 s @ 4K → 14.4 s @ 16K → 52.1 s @ 32K (sync-on).
- Linear (GatedDelta) share: 79% @ 512 → 64% @ 4K → 56% @ 16K → 43% @ 32K.
- Attention share: 21% → 24% → 33% → 44%. Crosses linear at ~32K.

**Key finding (unplanned discovery):** the research program document's FLOP math assumes Llama-8B-style dense MLP at 77% of work. On Qwen3.6-35B-A3B MoE hybrid, **MLP is not the bottleneck** (sparse MoE, 3B active). The dominant block is **GatedDeltaNet** (48 of 64 layers, 55-79% of prefill at realistic contexts). The plan's E3 "fused SwiGLU" attacks a minor block on this model.

**Proposal for Phase 1 (pending Federico approval):** attack GatedDeltaNet. Biggest block at mio's 4K-16K operating contexts, zero published Apple-Silicon work on it. Rationale in `docs/theories/phase0_analysis.md`.

## Open Federico gates

1. Target model swap to Qwen2.5-7B-Instruct? (default: stay on large-moe).
2. Phase 1 first attack: GatedDeltaNet (my proposal) vs plan's E3 (fused MLP — attacks wrong block for this model) vs plan's E1 (AMX).
3. Phase 0 quality lock — full MMLU/LongBench/RULER or proposed reduced set (GSM8K + HumanEval).
3. Commit results to `docs/theories/baselines.md` with hardware stamp and git SHA.
4. Pick highest-leverage attack vector from the breakdown data. No theory implementation until profile is in hand.

## Operating principles (carried from Antigravity CLAUDE.md even though the file isn't in the repo)

1. **Never report a measurement I didn't run.** Every number in docs/ is tied to a specific run with a timestamp and git SHA.
2. **Phase gates are Federico-approved.** I do not self-promote from Phase 0 to Phase 1. I present numbers and wait.
3. **Hypotheses pre-registered.** `experiments/{id}/hypothesis.md` exists and is committed *before* any implementation code in `mio/theories/{id}/`.
4. **STATUS.md updated every session.** No exceptions.
5. **Repository layout is fixed.** `mio/theories/{id}/`, `experiments/{id}/`, `docs/theories/{id}.md`. No improvisation.
6. **Negative controls ship with every theory.** A version that should not work. If the learned version doesn't beat the control, the theory fails.

## Open decisions

- **Target model.** Prior plan recommended Qwen2.5-7B-Instruct for faster iteration. I'm sticking with large-moe (Qwen3.6-35B-A3B) for continuity; switching would cost ~1 session for new baselines. Federico call if we want to revisit.
- **Base benchmarks for quality.** Phase 0 plan lists MMLU, GSM8K, HumanEval, LongBench, RULER. Running all of these cold on a 35B MoE is substantial compute. Recommend restricting to: GSM8K (8-shot CoT, 500 problems), HumanEval (164 problems), RULER-8K needle retention (single pass). MMLU + LongBench deferred until a theory claims to deliver enough speedup to justify the eval cost. Federico call.
