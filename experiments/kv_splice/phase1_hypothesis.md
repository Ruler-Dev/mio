# Path C Phase 1 — Chunk K_base Context-Robustness Study: pre-registration

**Status:** registered before data collection. Committed to `prefill-research` HEAD before any capture runs.

## Context

Path C's core idea is splicing KV for a shared "chunk" of tokens across prompts that have that chunk at different absolute positions. The math is clean for RoPE rotation — K_base can be rotated to any new position. But K_base itself depends on context via the transformer stack: `K_base_l = k_proj_l(x_l)` where x_l at position p in the chunk has been shaped by all preceding attention and GDN ops.

So: the fundamental empirical question is **how much does K_base at a fixed chunk-of-tokens vary when the preceding context changes?**

If K_base is mostly context-invariant, splicing produces good approximations and Path C works. If K_base varies wildly with context, splicing gives wrong attention outputs downstream — Path C dies.

## Measurement

- Target: Qwen3.6-35B-A3B (10 attention layers, 2 KV heads per layer, d_head=256).
- Candidate chunks (4 distinct, each 200-400 tokens, chosen for "self-containment"):
  1. Python imports block (~150 tokens of `import X`, `from Y import Z`).
  2. A standard class skeleton (docstring + 3 methods stub).
  3. A markdown section (heading + list + code snippet).
  4. A JSON-format tool definition (`{"name": ..., "description": ..., "parameters": ...}`).
- Wrappers (8 per chunk): 8 different preceding contexts of varying length (500-2000 tokens), varying topic, each ending with the chunk.
- Capture: at each attention layer, for each token position IN the chunk, record k_proj(x) output — a (n_kv_heads, d_head) vector per position per wrapper.

## Metric

For each (layer l, position p within chunk, head h):
    μ = mean of k_proj output across wrappers
    var_chunk = mean squared deviation from μ
    var_total = total variance of K_base across ALL positions (not just the chunk) in a single wrapper
    ratio = var_chunk / var_total

**ratio < 0.1** → essentially context-invariant; splice-safe.
**0.1 ≤ ratio < 0.3** → mostly context-invariant; splicing introduces small perturbation, likely within DFlash's tolerance.
**ratio ≥ 0.3** → substantially context-dependent; splicing produces wrong attention; not splice-safe.

Aggregate by layer index and by chunk type. Report worst-case ratio, median ratio.

## Predictions

**P1.** Variance RATIO is smaller at earlier attention layers (e.g., layer 3) than later (e.g., layer 39), because context-propagation has had fewer hops to amplify differences.

**P2.** The Python imports chunk will be MOST robust (tokens are essentially a vocabulary sampler — each line is independent).

**P3.** At least one (layer, chunk) combination will have ratio < 0.2 — enough to demonstrate splice-safety exists on this model.

**P4.** The JSON tool definition chunk will be the LEAST robust because its tokens interact semantically (keys reference values) — attention within the chunk depends on what was seen before, which depends on context.

## Pass criteria

Pass Phase 1 and proceed to Phase 2 (RoPE-rewriting prototype) iff:
- At least 2 of 4 chunks show median ratio < 0.3 across attention layers.
- At least 1 chunk × layer combination achieves ratio < 0.1.

Fail (→ Path C dead on this model) iff:
- All chunks show median ratio > 0.5 across layers.

## Negative control

For a "random" chunk (500 tokens of random vocab), expect high ratio (~1.0 since the chunk has no semantic structure to "anchor" the model). If even the random chunk has low ratio, the measurement is measuring the wrong thing.

## Compute budget

- 4 chunks × 8 wrappers = 32 prefill passes at ctx~1500 tokens each.
- Each prefill: ~1 s on M4 Max.
- Capture hook overhead: negligible.
- Total: ~1 min of compute.
- Analysis: CPU regression, seconds.

**Session-compatible.** Produces a go/no-go decision on Path C before investing weeks.
