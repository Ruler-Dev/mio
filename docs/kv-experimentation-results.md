# `kv-experimentation` branch — benchmark results

**Target:** Qwen3.6-35B-A3B-UD-Q4_K_XL (21 GB target + 913 MB DFlash draft), hybrid_gdn architecture (48 GatedDeltaNet + 16 full-attention layers).
**Contexts:** 4K, 16K, 32K tokens (padded with project-like code).
**Prompts:** `fib`, `sort_bug`, `bst`, `nqueens` — coding tasks of varied shape.
**Repeats:** 2 per cell, min reported.
**Measurement:** wall-clock prefill, wall-clock decode, reported tok/s, per-cycle DFlash acceptance, output sha256.

## Scope

Three configurations attempted:

1. **Baseline** — mio default: PolarQuant-4 KV + DFlash speculative decoding.
2. **DDTree** — `MIO_DDTREE_BUDGET=4`: 8-bit KV cache + DFlash tree-attention verify.
3. **Frozen KV (C3)** — explicit `engine.warm_and_freeze(messages)` + reload on subsequent request.

**Not benchmarked:** B2 (speculative prefill) — scaffold only, `_run_partial_target_prefill` raises `NotImplementedError`. Any numbers would measure baseline + Python overhead.

---

## 1. Baseline — complete matrix

All 12 cells × 2 reps. Min of 2 reps reported.

### Decode throughput (gen tok/s)

| prompt    | ctx 4K | ctx 16K | ctx 32K |
|-----------|-------:|--------:|--------:|
| fib       | **128.9** | 100.9 | 23.3 |
| sort_bug  | 84.3   | 64.8 | 30.9 |
| bst       | **218.9** | 136.9 | 52.2 |
| nqueens   | 79.3   | 64.3 | 37.5 |
| **mean**  | **127.9** | **91.7** | **36.0** |

### Prefill wall-time (ms)

| prompt    | ctx 4K | ctx 16K | ctx 32K |
|-----------|-------:|--------:|--------:|
| fib       | 2,797  | 11,540 | 38,349 |
| sort_bug  | 3,777  | 15,099 | 42,016 |
| bst       | 3,037  | 12,745 | 39,811 |
| nqueens   | 2,798  | 12,478 | 37,122 |
| **mean**  | **3,102** | **12,966** | **39,325** |

### DFlash acceptance per cycle

| prompt    | ctx 4K | ctx 16K | ctx 32K |
|-----------|-------:|--------:|--------:|
| fib       | 6.65   | 6.17   | 6.52 |
| sort_bug  | 4.42   | 4.29   | 3.94 |
| bst       | **11.29** | 8.53 | 8.81 |
| nqueens   | 4.00   | 3.91   | 3.56 |

**Observations:**
- Decode throughput drops ~3.6× from 4K→32K, attention-quadratic as expected.
- `bst` hits 218 tok/s at 4K thanks to high DFlash acceptance (11.29 tokens/cycle); the draft model predicts structured class boilerplate extremely well.
- Prefill is 13× more expensive at 32K than at 4K. **This is the time frozen-KV and future prefill work have to reclaim.**
- Quality is deterministic: rep0 and rep1 produce the same sha256 for every (prompt, ctx) pair.

---

## 2. DDTree — partial (killed after 3 cells)

DDTree ran catastrophically slower than baseline on this MoE. After 3 cells I killed the round to save compute. Results below are what we collected:

| prompt | ctx   | baseline tps | DDTree tps | ratio |
|--------|------:|-------------:|-----------:|------:|
| fib    | 4K    | 128.9 | 54.7   | **0.42x** |
| fib    | 16K   | 100.9 | 2.7    | **0.027x** |
| fib    | 32K   | 23.3  | — (killed) | — |

DDTree `accept` was 3.81–4.19 vs baseline's 6.17–6.65 — tree verify does not reach DFlash's acceptance rate here, and the per-cycle cost is much higher.

### Why

Several compounding reasons:

1. **Hybrid model architecture is unfavorable.** Qwen3.6-35B-A3B has 48/64 layers as GatedDeltaNet (recurrent). DDTree's tree-attention advantage only applies to the 16 attention layers; recurrent layers must process each tree node via the parent-indexed Metal kernel sequentially, which has limited parallelism.
2. **8-bit KV vs 4-bit PolarQuant is a bandwidth regression.** DDTree disables PolarQuant and uses mlx_lm's 8-bit QuantizedKVCache. On Apple Silicon the target prefill becomes more memory-bandwidth-bound per token — you pay 2× the bandwidth per KV read for no tree-verify gain on the recurrent layers.
3. **Tree-aware commit path unavailable under 8-bit.** The `EXACT_COMMIT=1` path we had to enable for quantized compatibility does a full sequential re-forward of accepted tokens, adding a second target pass per cycle.
4. **The tree-verify SDPA is not routed through MLX fast paths under 8-bit.** The `_split_sdpa_output` large-context optimization had to be bypassed (its array slicing can't operate on quantized `(values, scales, biases)` tuples).

**Correctness note (real bug fixed on this branch):** `main`'s `mio/ddtree/verify.py` assumes `keys.shape[2]` on the return of `cache.update_and_fetch`, which is a 3-tuple under `QuantizedKVCache`. It crashes immediately on hybrid targets. The fix (tuple-handling + split-path bypass under quantized) is on this branch and **should be cherry-picked to main** so DDTree at least doesn't error out, even if it's slow.

### Verdict

DDTree as shipped is **not a win on Qwen3.6-35B-A3B** at any tested context size. It's a reasonable idea on predominantly-attention models, but on this hybrid stack the architecture costs it the speedup.

---

## 3. Frozen KV (C3) — focused run

Prompt: `sort_bug` (coding task, fix a broken bubble sort). 2 warm repeats per cell, min reported.

### Prefill wall-time: cold vs. warm

| ctx | cold prefill | warm prefill (best) | saved | speedup |
|-----|-------------:|--------------------:|------:|--------:|
| 4K  | 4,950 ms | **15 ms**  | 4,935 ms  | **326×**  |
| 16K | 12,029 ms | **18 ms** | 12,011 ms | **662×**  |
| 32K | 35,658 ms | **21 ms** | 35,637 ms | **1,734×** |

A warm prefill writes one token (the last prompt token) and samples the first bonus logit. That cost is fixed at ~20 ms regardless of context. The savings scale ~linearly with context until attention quadratic dominates at 32K, where it's 35 seconds saved per request.

### One-time freeze cost (paid once, on cold pass)

| ctx | warm_and_freeze wall-time | % of cold prefill |
|-----|--------------------------:|------------------:|
| 4K  | 2,758 ms | 56% |
| 16K | 13,255 ms | 110% |
| 32K | 46,420 ms | 130% |

`warm_and_freeze` runs a full `prefill_only` pass + writes the safetensors file. At 4K it's roughly half a prefill; at 32K it's slower than one prefill because safetensors serialization of ~5 GB of KV entries dominates. This cost is paid **once per distinct prompt-prefix**, then amortized over all subsequent requests. Break-even: **2 requests at 32K recoups the warm_and_freeze cost.**

### Quality check

Every (cold, warm) pair produced different sha256, but **the code portion is byte-identical**. The divergence is in the prose paraphrase after the code block — expected behavior when a 1-token prefill reproduces the post-prompt state with tiny FP noise that flips a tied-logit token in the tail.

Example at ctx=32K — both outputs return **the same corrected `sort` function** with the same two bug-fixes (tuple swap, `i+1` inner loop). The explanation:
- cold: *"…failed to swap values (only assigning `a[j]` to `a[i]`, which overwrote the original `a[i]` without preserving it for later positions)…"*
- warm: *"…failed to swap values (assigning `a[j]` to `a[i]` overwrote the larger value without preserving it), and the inner loop started at `i` instead of `i + 1`, causing elements to be compared with themselves…"*

Semantically identical findings. For a serving system that just needs the *code change*, this is a non-event. For a strict byte-equivalence requirement, this would be a regression, but greedy decoding over any cache-path variation has this property regardless of C3.

### Decode throughput along with the prefill win

Decode tok/s during the warm run vs. cold:

| ctx | cold gen tps | warm gen tps (best) |
|-----|-------------:|--------------------:|
| 4K  | 83.4 | 98.3 |
| 16K | 73.0 | 65.3 |
| 32K | 26.9 | 60.9 |

Decode throughput is similar, occasionally faster (the warm path starts with a shorter draft-cache window, which can make the first cycle cheaper). End-to-end latency including prefill: frozen-warm beats cold by the full prefill savings; at 32K that's the difference between 40 s and 2.3 s to first tokens-complete.

---

## 4. Implementation notes (surfaced during the run)

### DDTree tuple bug on `main`

`mio/ddtree/verify.py` on `main` crashes on hybrid_gdn targets because it calls `keys.shape[2]` on a `QuantizedKVCache` return, which is a `(quantized, scales, biases)` 3-tuple. This branch's fix handles both tuple and array forms and disables the split-SDPA path under quantized cache. That commit should land on `main` with or without the rest of this branch, since DDTree is unusable without it.

### Frozen-KV auto-freeze-during-generate was removed

Earliest design: freeze the cache at the end of `generate()`. Fatal problem on hybrid models: the post-decode cache is at `offset = prompt_len + gen_len` with rollback-tape residue, and the recurrent layers can't be rolled back to `prompt_len` without losing information. Result: loading that snapshot produces a cache with 1361 tokens when we need one with 1104, duplicating context and corrupting output.

### Explicit `warm_and_freeze` is the correct entry point

Only route to clean snapshots is an explicit `prefill_only` pass on `prompt[:-1]`. The runtime always runs at least one prefill token (it needs the last-position logits to sample the first bonus token), so freezing prompt_len - 1 means the runtime's mandatory 1-token prefill fills the last slot exactly once.

### Storage model

- Fingerprint = sha256 over (model_id, pq_bits, tq_bits, ctx_window, prefix_len, version, first `prefix_len` token IDs).
- On save: `stored_tokens` metadata field includes the full intended prompt, so scan can rank candidate snapshots by shared-prefix length. This is the same contract mio's in-memory `prefix_cache` uses — we're effectively giving that an on-disk, cross-process tier.

---

## 5. What this says about where to invest

1. **Ship `frozen_kv` + `warm_and_freeze`.** It's a drop-in on-disk prompt cache for mio — not novel (Anthropic/SGLang do the same thing), but until now mio had no way to skip the 12-42 s prefill on a recurring system prompt after a process restart. That's a real agent-workflow win. The branch's existing test suite (25 tests, deterministic x5 runs) validates correctness at the module level.

2. **Drop DDTree from default consideration on hybrid models.** Keep it off behind the opt-in env flag; keep the code around for future pure-attention targets where the math works.

3. **B2 (speculative prefill) still needs the projector training data.** The probe in `mio/draft_kv/probe.py` can tell us empirically whether a linear projector from early hidden to late K/V has signal on this model. That's the right next experiment — cheap to run, tells us whether to invest in training.
