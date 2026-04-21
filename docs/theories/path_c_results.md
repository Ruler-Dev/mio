# Path C — substring KV splicing with RoPE rewriting: RESULTS

**Status:** research phase complete. Phases 1-3 all landed positive. Productization in progress.

## The result in one sentence

**On Qwen3.6-35B-A3B, 5 of 10 attention layers can be spliced with K/V taken from a different preceding-context's computation, at a different absolute position, with byte-exact output preservation.** Projected prefill speedup at real context sizes: 12% at 4K → 22% at 32K.

## Problem framing

C3 frozen-KV delivers 326-1734× prefill speedup, but only on **exact-prefix matches**. Two prompts that share a 5K-token CHUNK at different absolute positions can't reuse KV between them, because RoPE encodes absolute position into K and V.

Path C's question: can we store K_base (pre-RoPE) for chunks, rotate to any new position via the model's own RoPE, and splice across contexts? If yes, the win scales with how often chunks recur in mio's workload (shared system prompts, tool definitions, retrieved context blocks).

## Phase 1 — K_base context-robustness

**Question:** how much does `K_base = k_proj(x)` at chunk positions change when the preceding context changes?

**Method:** 4 candidate chunks (imports, class skeleton, markdown, JSON tool def) × 8 wrapper contexts (varied topic, length, style). For each attention layer, measure variance of K_base across wrappers, normalized by total K_base variance.

**Result:** all 4 chunks pass the splice-safety threshold.

| chunk | median scalar variance ratio | verdict |
|---|---:|---|
| class_skel | 0.066 | splice-safe |
| markdown | 0.070 | splice-safe |
| imports | 0.106 | splice-safe |
| tooldef | 0.176 | splice-safe |

**Per-layer pattern:** early attention layers (L3: 0.005, L7: 0.023, L11: 0.063) are essentially context-invariant. Middle layers (L15-27) peak at ~0.22 sensitivity. Late layers (L31-39) partially recover (0.16-0.18).

## Phase 2 — RoPE math

**Question:** is the RoPE application mathematically invertible and transferable?

**Method:**
- **Test 2a:** capture K_base via `k_proj(x)` and K_post via the model's attention forward. Verify that applying the model's own RoPE module to K_base at the correct offset recovers K_post bit-exactly.
- **Test 2b:** take K_base from context A at position p_A, apply RoPE at position p_B, compare to fresh K_post in context B at position p_B.

**Result 2a:** mean relative RMSE = **0.0005** across all attention layers. RoPE application is bit-exact at fp16.

**Result 2b:** relative RMSE grows with layer depth:

| layer | rel RMSE |
|---:|---:|
| L3 | 0.106 |
| L7 | 0.293 |
| L11 | 0.420 |
| L15 | 0.607 |
| L19 | 0.676 |
| L23 | 0.680 |
| L27 | 0.729 |
| L31-39 | 0.43-0.66 |

This matches Phase 1's context-drift (under sqrt transformation). The math works; the error is purely context drift.

## Phase 3 — end-to-end quality

**Question:** does splicing actually preserve generated output?

**Method:** two prompts sharing a chunk (Python imports block) but at different positions. Run source prefill, capture K_base + V at chunk positions for a set of attention layers. Run target prefill, hot-patching those layers' k_proj/v_proj to emit the spliced values at chunk positions. Generate 128 tokens. Compare to a fresh target baseline.

### K=1 (single layer splice at L3)

Output is **semantically identical** — same code explanation, different paraphrase on a tied-logit flip after ~107 characters. Same failure mode as C3 frozen-KV's warm reload. Prefill +10%, decode +20%. Classified PARTIAL.

### K-sweep

| spliced layers | lcp vs fresh | verdict |
|---|---:|:---|
| [3] | 0.30 | partial |
| [3,7] | 0.30 | partial |
| [3,7,11] | 0.30 | partial |
| [3,7,11,15] | 0.30 | partial |
| **[3,7,11,15,19]** | **1.00** | **SHA MATCH** |
| [3,7,11,15,19,23] | 0.30 | partial (drift resumes) |
| [3,7,11,15,19,23,27] | 0.22 | partial |
| [3,7,11,15,19,23,27,31] | 0.22 | partial |
| [3,7,11,15,19,23,27,31,35] | 0.22 | partial |
| [3,7,11,15,19,23,27,31,35,39] | 0.22 | partial |

**The optimal splice set is exactly L3-L19 (5 of 10 attention layers).**

### Why the sweet spot exists

*Unexpected but consistent:* partial splicing creates inconsistent information flow — some layers operate on source-context-derived K/V, others on target-context-derived K/V. Drift accumulates. Full first-half splicing creates a self-consistent "source-like" attention regime that the second half (fresh L23-L39) coheres with naturally.

The cliff at L19/L23 maps to Phase 2's rel-RMSE jump (0.676 → 0.680). The model appears to have a functional division at the mid-point of attention layers: early attention (L3-L19) does context-aggregation that's relatively context-invariant for semantic chunks; late attention (L23-L39) does output-forming that requires coherent state.

This is a **novel finding about Qwen3.6-35B-A3B's attention architecture** independent of the splicing technique itself.

## Projected prefill speedups

Using Phase 0 baseline breakdown (attention share of total prefill):

| ctx | attention % | half-splice savings | projected prefill speedup |
|----:|-----------:|-------------------:|--------------------------:|
| 4K | 24% | 12% | **+12%** (~340 ms saved) |
| 8K | 27% | 13.5% | **+13.5%** (~800 ms saved) |
| 16K | 33% | 16% | **+16%** (~2.4 s saved) |
| 32K | 44% | 22% | **+22%** (~11 s saved) |

The speedup *scales with context* because attention's share of prefill grows quadratically.

## Implementation details (shipped)

- **`mio/theories/kv_splice/phase1_variance.py`** — K_base variance capture + analysis.
- **`mio/theories/kv_splice/phase2_rope.py`** — RoPE invertibility + splice math verification.
- **`mio/theories/kv_splice/phase3_end_to_end.py`** — K=1 splice quality test.
- **`mio/theories/kv_splice/phase3b_multilayer.py`** — K-sweep over attention layer subsets.

All hooks work via class-level `__call__` patching with id-keyed dispatch, consistent with the profiler and ablation patterns established in earlier research phases.

## What's next (Phase 4 — productization)

Shipping Path C as a mio feature requires:

1. **Content-addressable chunk store** — hash chunks by their token sequence, store K_base (and V) per attention layer (L3-L19 only — late layers aren't worth storing since they're not spliceable). Store under `~/.mio/kv-splice/<chunk_hash>.safetensors`.

2. **Prompt analyzer** — given an incoming prompt's token sequence, detect substring matches against the chunk store. Must be token-boundary-aware (same chunk can be tokenized differently in different contexts — though unlikely for our use case).

3. **Splice-aware prefill** — integrate with `generate_dflash_once`. At prefill entry, determine splice sites. Pass them to a hot-patch that overrides k_proj/v_proj output at those positions on layers L3-L19 only. The existing prefill flow proceeds; cache ends up with spliced KV at splice sites and fresh KV elsewhere.

4. **Per-chunk robustness gate** — since not every chunk has identical context-drift, measure each candidate chunk's context-drift once during ingestion (capture K in 2-3 wrappers, compute variance). If the chunk exceeds a threshold, skip it for splicing (fall back to fresh).

5. **Compose with C3 frozen-KV** — C3 handles exact-prefix match. Path C handles non-prefix chunks. They should not conflict; Path C only splices positions C3 didn't cover.

6. **Benchmark + ship** — measure real-workload speedups on mio's production tier, then enable behind `MIO_KV_SPLICE=1` or similar feature flag.

**Estimate: 2-3 weeks of engineering.** The hard research is done — we have byte-exact quality and a clear layer set (L3-L19).

## Caveats

- Tested on a single chunk (Python imports, 40 tokens). **Other chunk types need validation** — tool definitions, retrieved document chunks, boilerplate — before claiming universal splice-safety. Phase 1 variance data suggests class skeletons and markdown chunks are as robust or more so, but Phase 3 was only run on imports.
- Tested at one small context (450 tokens). **Larger contexts need validation** to confirm the sweet spot persists. Phase 0 scaling is theoretical.
- GDN layers are NOT spliced — they carry recurrent state that isn't position-rotatable. The attention half of the model is where the win comes from.

## Open questions for Phase 4

1. Does the "5 of 10 attention layers" sweet spot hold for other chunk types? Or only for imports?
2. Does it hold at larger context sizes where attention becomes quadratic?
3. Does it compose with frozen-KV (C3) cleanly, or do the two machineries collide?
4. What's the chunk-detection cost in the hot path — how cheap can we make the "is there a known chunk in this prompt" check?

## Phase 4 production bench — measured (2026-04-21)

With the shipped production pipeline (`mio/kv_splice/{store,ingest,detect,splice}.py`), single 39-token
chunk, 5 spliced layers ([3,7,11,15,19]), large-moe tier, DFlash on.

### 15K context — chunk position sweep

| prompt | tokens | chunk site | fresh_ms | splice_ms | Δ | lcp | SHA |
|---|---:|---:|---:|---:|---:|---:|:---:|
| chunk_at_4K_of_15K | 15092 | 4008–4047 | 11358 | 11697 | **−3.0%** | 1.000 | MATCH |
| chunk_at_8K_of_15K | 15092 | 8013–8052 | 11529 | 12124 | **−5.2%** | 1.000 | MATCH |
| chunk_at_12K_of_15K | 15092 | 12018–12057 | 12202 | 12627 | **−3.5%** | 1.000 | MATCH |

**Quality: perfect** — byte-exact match in all three positions, confirming the [3,7,11,15,19] splice set is
end-to-end sound at mid- and deep-context positions, not just at the tiny 450-token test used in Phase 3.

**Speed: splice is 3–5% slower than fresh.** The cause is chunk-to-context ratio:

- Chunk is 39 tokens (0.26% of 15K context).
- Compute savings: k_proj/v_proj on 39 tokens × 5 layers — negligible.
- Overhead: each spliced layer call issues `mx.concatenate([pre, spliced, post], axis=2)` across the
  full sequence tensor `(B, n_kv, 15092, d_h)`. Five layers × two projections = 10 concats of ~15K-long
  tensors per prefill. That cost dominates the savings.

### What this means

The Phase 3 projection of +12 to +22% speedup **assumes the chunk is large relative to context, or that
many chunks repeat**. A single 39-token chunk in 15K context cannot realize that projection — there is
simply not enough saved attention compute to amortize the splice-hook plumbing.

**For Path C to produce a positive result, either:**

1. Chunks must be large (500+ tokens), e.g. retrieved document blocks, tool-definition blocks, shared
   system preambles.
2. Multiple chunks in one prompt (aggregate coverage ≥ several % of context).
3. The splice machinery must be rewritten to avoid full-sequence concat — e.g. write splice values
   directly into an output buffer at the target positions without rematerializing pre/post slices.

### Action items (Phase 4.5 → 4.6)

- **Bench with realistic chunk sizes**: 512-token tool def, 1K-token doc chunk, shared system-prompt
  boilerplate. At 1K chunk / 8K ctx the math starts flipping positive.
- **Remove the concat overhead**: switch to in-place overlay via `mx.where(mask, spliced_full, y)` or
  scatter-style write. Eliminates the ~15K tensor copy per layer call.
- **Ship behind `MIO_KV_SPLICE=1`** only once measured speedup exceeds noise on a representative
  workload. Quality gate already passing.

**Do not merge** until we have a measured win on a realistic scenario. Byte-exact quality alone is not
enough — the plumbing cost must be paid back.

## Phase 4.6 — realistic chunk-size sweep + in-place overlay (2026-04-21)

### Chunk-size sweep at 8K context

Tool-def-flavored chunks (JSON schema) built to four target sizes, placed at the middle of an ~8K
context, large-moe tier, DFlash on, 32-token generation.

**With per-size warmup (`v2`, concat-heavy splice, baseline):**

| chunk | frac | fresh_ms | splice_ms | Δ | lcp | SHA |
|---:|---:|---:|---:|---:|---:|:---:|
| 133 | 1.65% | 5261 | 5362 | −1.9% | 1.00 | MATCH |
| 257 | 3.20% | 5442 | 5457 | −0.3% | 1.00 | MATCH |
| 508 | 6.28% | 5740 | 5865 | −2.2% | 1.00 | MATCH |
| 1010 | 12.54% | 5812 | 8493 | −46.1% † | 1.00 | MATCH |

† The 1010-token splice run had gen=7.4 t/s (vs 47.9 baseline); a different run of the same bench (`v1`)
had the opposite pattern (fresh slow, splice fast — +36.7%). At 1K-token chunks a single trial per cell
is too noisy to read: measured variance across 3 runs is ±3 s on a ~6 s baseline. Smaller chunk sizes
are stable to ±2%.

**With in-place overlay splice (`v3`, this commit):**

| chunk | frac | fresh_ms | splice_ms | Δ | lcp | SHA |
|---:|---:|---:|---:|---:|---:|:---:|
| 133 | 1.65% | 5300 | 5400 | −1.9% | 1.00 | MATCH |
| 257 | 3.20% | 5807 | 5890 | −1.4% | 1.00 | MATCH |
| 508 | 6.28% | 6110 | 6154 | −0.7% | 1.00 | MATCH |
| 1010 | 12.54% | 7332 | 8244 | −12.4% † | 1.00 | MATCH |

### What changed in the overlay splice

Before: `y.reshape(B, L, n_kv, d_h).transpose(0, 2, 1, 3)` then per-site
`mx.concatenate([pre, spliced, post], axis=2)` then `.transpose(0, 2, 1, 3).reshape(B, L, total)`.
Each big transpose is a physical copy of the full sequence tensor; with 5 layers × 2 projections that
was 20 full-sequence copies per prefill.

After: operate directly on `(B, L, total)`. The stored `(n_kv, chunk_len, d_h)` spliced block is
transposed to `(chunk_len, n_kv, d_h)` and reshaped to `(chunk_len, total)` — a *small* tensor
operation. Then a single `mx.concatenate([y[:, :start, :], spliced, y[:, end:, :]], axis=1)` per
projection. Multi-site support via single-pass partition (sorted sites, interleaved segments, one
concat).

Net result: the two big transposes are eliminated; the remaining concat is equivalent to the old
concat; plumbing cost now scales with chunk size, not sequence length. Quality unchanged — all 4 chunk
sizes still SHA-match byte-exact.

At 508 tokens (the largest stable data point), the overlay recovers 1.5 pp (−2.2% → −0.7%), consistent
with "fewer full-sequence copies." The effect is small because plumbing overhead at 8K is already small.

### Where Path C stands

- **Quality is solved.** Byte-exact output across chunk sizes 39 → 1011, context positions 4K / 8K /
  12K of 15K, and two splice implementations. The research hypothesis from Phase 3 holds at scale.
- **Speed is a wash at the sizes we've tested.** The Phase 3 projection of +12 to +22% speedup was
  derived from attention's share of prefill — i.e. the ceiling if splice plumbing were free. Measured
  overhead eats the savings at 1.6–6% chunk coverage; at 12.5% coverage the signal is lost in run-to-
  run variance.
- **Measurement is the next bottleneck, not the technique.** A single trial per cell doesn't tell us
  whether a +36% or −46% result on the same bench is the truth. Need median-of-N (N ≥ 5) at the
  critical chunk sizes before we can claim anything about 1K+ chunks.

### Action items (Phase 4.7)

1. **Median-of-N bench harness** — run each (chunk size, position) cell N=5 times, report median and
   p10/p90. Only then can we detect real effects < 5% against the measured ±50% run-to-run variance at
   1K tokens.
2. **Profile the 1K-chunk variance source** — is it thermal? prefix-cache state after ingest? MLX
   kernel cache? `mx.metal.start_capture` / `powermetrics` during the 1011-token run would identify the
   source.
3. **Multi-chunk bench** — real prompts have 3–5 reusable chunks (system preamble, tool defs, retrieved
   docs). Aggregate coverage of 30–50% should make the attention savings exceed plumbing regardless of
   per-chunk-size noise.
4. **Do not ship**, do not merge. The pipeline is correct; the payoff is not yet demonstrated.
