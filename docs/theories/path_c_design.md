# Path C — Substring KV splicing with RoPE rewriting: design

**Status:** design only. Implementation is research-grade multi-week work.

## Problem

C3 frozen-KV gives 326-1734× prefill speedup **when the entire prompt prefix matches exactly**. In practice, agent prompts have a fixed system prompt (10-40 K tokens) followed by variable user content. C3 handles that perfectly.

**Path C's scope:** what if TWO prompts share a 5 K-token CHUNK in the *middle*, not a prefix? Example:

- Prompt A: `[system: 10K tokens][tool_def: 5K shared tokens][question_1: 500 tokens]`
- Prompt B: `[system: 10K tokens][tool_def: 5K shared tokens][question_2: 500 tokens]`

Exact-prefix match catches the 10 K system. The 5 K tool_def is also shared, but if A had been cached with system + tool_def + question_1 concatenated, the cached KV has position encodings for `question_1` starting at position 15,000. Prompt B's version of tool_def is at same positions 10,000-15,000 → prefix-match catches it.

**Where C3 fails:** if prompts A and B have DIFFERENT system prompts but identical tool_defs, the tool_def chunk appears at DIFFERENT absolute positions. RoPE encodes position into K and V — a K vector computed for position 10,000 is NOT valid at position 8,000. C3 can't splice tool_def from a different-positioned cache.

**Path C's contribution:** store KV for "content chunks" *without* their RoPE applied (or with a neutralization that can be reversed). At splice time, apply fresh RoPE for the chunk's new absolute position.

## Math

RoPE on query/key applies a position-dependent rotation R(θ_pos) ∈ SO(d_head) as a block-diagonal matrix of 2×2 rotations:

    K_pos = K_base × R(θ_pos)

where K_base is the pre-RoPE key (output of k_proj of x, which is position-invariant). R(θ_pos) is orthogonal, so R(θ_pos)^T × R(θ_pos) = I.

**Key insight:** if we store K_base (pre-RoPE) instead of K_pos (post-RoPE), we can rotate to any new position at splice time with a single matmul per layer per token:

    K_new_pos = K_base × R(θ_new_pos)

Cost: O(L × d_head) per layer. Cheap.

## Implementation sketch

1. **Chunk-aware capture:** during prefill, in addition to writing the final post-RoPE K/V into the KV cache, also write a pre-RoPE copy to a side buffer. (MLX's attention layers apply RoPE just before cache.update_and_fetch; hook there.)

2. **Storage:** content-addressable store keyed by chunk token hash (say 1024-token chunks). Each entry: `{chunk_hash: (K_base per_layer_per_head, V per_layer_per_head)}`.

3. **Retrieval:** on a new prompt, tokenize, split into 1024-token chunks (overlapping or not), hash each. Look up matching chunks in the store.

4. **Splice:** construct the prompt's KV cache by:
   - For each token position p in the new prompt:
     - Find which stored chunk (if any) owns this position.
     - If owned: load K_base from the chunk's store, apply `R(θ_p)`, write into the cache.
     - If not: compute normally via a partial forward on that span.

5. **Run full target prefill over only the NON-cached spans** to compute the missing KV. The attention computation at each cached-but-misaligned position reads from the newly-rotated K_new_pos; attention across cached + fresh spans works out because we've rotated every K into the right position.

## The catches

**Catch 1 — Attention is not just KV.** During prefill, each layer's OUTPUT depends on attention over ALL prior positions. A token at position p sees positions 0..p-1. If we splice chunk tokens [p=500..1500] from another context where they were at positions [100..1100], the splicing changes what they "saw" during their original computation. Their final hidden state encoding is context-specific.

In other words: we can splice the *output KV of the chunk* at a new position, but the chunk's *attention input* was computed under a different context. If the downstream-of-chunk tokens read attention over those (now-relocated) K's, they get the right answer. But IF the chunk's own computation depended on seeing particular earlier tokens (which it almost always does), that context is lost.

**This is the hard problem.** Fully lossless chunk splicing across contexts requires the chunk to have been computed attending to the SAME preceding tokens in both scenarios. That rarely happens.

Approximation approach: chunks that are "context-robust" (i.e., self-contained, like a tool definition) produce similar enough KV regardless of preceding context. Empirically measurable — capture a chunk at 100 different contexts, measure KV variance, flag low-variance chunks as splice-safe.

**Catch 2 — RoPE rewriting is not instantaneous.** At prefill time, rotating K_base for L tokens × 16 layers × 2 KV-heads × d_head = 2 K × 16 × 2 × 256 = 16 M elements per layer × 16 layers = 256 M element-wise rotation ops. At MLX's matmul throughput, negligible. Verified.

**Catch 3 — Cache format must be modified.** Current mio KVCache (including PolarQuant variants) stores post-RoPE K. Adding a pre-RoPE capture hook requires touching the attention layer forward. Achievable via existing mio hook infrastructure (similar to speculative-linear-cache patching in mio/dflash/runtime.py), but non-trivial.

**Catch 4 — Only attention layers have RoPE.** GatedDeltaNet has its own recurrent state that is not position-rotatable. Splicing GDN state across contexts is a different problem — recurrent state accumulates through time, and "position shift" isn't a meaningful op. So chunk-splicing for this hybrid model is ATTENTION-ONLY, which means we save only the attention layer's KV recomputation (not the GDN recurrence).

On Qwen3.6-35B-A3B at 16K context:
- Attention: 4.7 s (33% of prefill)
- If we could splice a 5 K chunk: save ~1.5 s of attention for that chunk.
- GDN cost for that chunk: 2.5 s (not savable via splicing).

Path C on this model: save ~15-20% of prefill ON THE SPLICED FRACTION, which itself is some fraction of the prompt. Maybe 5-15% total prefill win at realistic chunk overlap rates.

## Scope for a first implementation

1. **Week 1:** RoPE rewriting prototype. Given a (K_base, V, pos) tuple, show that applying R(θ_new_pos) and then running attention produces the same output as computing K fresh at the new position. Bit-exact or ≤1e-4 MSE.
2. **Week 2:** Pre-RoPE capture hook for Qwen3NextAttention layers. Verify stored KV round-trips through RoPE.
3. **Week 3:** Chunk-level content-addressable store. Hash, save, load. Tests.
4. **Week 4:** Splice-aware prefill runtime. Partial forward over uncovered spans + KV injection for covered chunks. Attention math correctness tested.
5. **Week 5:** End-to-end benchmark: prompts with shared chunks at varying positions. Measure quality preservation + prefill speedup.

**Total: ~5 weeks** assuming the math works out and RoPE rewriting is bit-exact. It's genuinely novel — nobody has published chunk-level KV splicing for a LOCAL inference stack. Mio would be first.

## Why defer from this session

Path C requires:
- Deep changes to the attention layer forward path.
- Custom cache format with pre-RoPE + post-RoPE storage.
- Splice-aware prefill path that handles mixed cached/uncached spans.
- Quality validation at multiple chunk positions and contexts.

All of this is research-grade engineering. Not session-compatible; 5 weeks is the realistic floor.

## Composition with Paths A and B

- **Path A + Path C:** Path A's cluster retrieval becomes truly useful — a "similar but not identical" prompt can recycle whole chunks of the prototype's KV via Path C's splicing. Path A selects the candidate, Path C makes it reusable.
- **Path B is independent.** Per-user LoRA is orthogonal; it affects which model runs, not how its KV cache is structured.

## Bottom line for Path C

Highest-ceiling "big win" of the three, and the only one that would produce a genuinely novel research result. Also the hardest. **Pick this for a serious multi-week project** when there's a reliable research week / month of focused time.
