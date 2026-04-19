# Prefill Speedups in Mio: Prefix Cache + LM-Head Slicing

**Technical note, mio project, 2026**

## Abstract

Two complementary optimizations to reduce prefill latency and time-to-first-token
(TTFT) in mio's DFlash runtime:

1. **Prefix Cache** — cache per-layer KV (+ SSM) state after the first
   prefill pass, keyed by token prefix. On subsequent calls that share a
   common prefix (system prompt, agent directives, multi-turn history),
   skip prefill for that range entirely. Measured **4.4–7.8× wall-clock
   speedup** on warm hits across all four mio default tiers.

2. **LM-Head Slicing** — during prefill, only the last token's logits are
   used (to sample the first bonus token). Project only the final hidden
   state through the vocab-sized LM head instead of every position.
   Measured **+13–15% pure prefill tok/s** across small/medium tiers.

Both optimizations compose multiplicatively: warm hits benefit from both
(a shorter prefill + a smaller LM-head matmul on the small suffix).

## 1. Prefix Cache

### 1.1 Mechanism

`generate_dflash_once` gained two optional parameters:
- `warm_state: dict | None` — pre-populated `{target_cache, draft_cache, target_hidden, offset}`
- `prefill_only: bool` — short-circuit after prefill, skip the decode loop
- `return_final_state: bool` — attach the final cache dict to the result

On engine-level `generate()`:
1. Look up the longest cached prefix that matches the start of the incoming prompt.
2. If found, pass the cached state as `warm_state`. The runtime skips forward on
   those cached tokens and prefills only `prompt_tokens[offset:]`.
3. After generate, if the previous prompt and the current share a ≥`min_tokens`
   prefix (default 64), and that prefix isn't already cached, run a
   `prefill_only` pass on the common prefix (minus a `margin` of 32 tokens for
   tokenization noise) and store the resulting caches.
4. LRU-evict at `max_entries` (default 4).

### 1.2 Target_hidden reconstruction

The subtle gotcha: the DFlash draft uses `target_hidden` — per-layer feature
extracts of the target's hidden states — for cross-attention. If the prefill
only ran on the novel suffix, `target_hidden` covers only that suffix, starving
the draft of prompt context and dropping acceptance length.

Solution: we persist `target_hidden` in the cache entry (captured during the
`prefill_only` pass). On hit, concatenate the cached `target_hidden` (for the
warm range) with the fresh `target_hidden` (from novel-suffix prefill) before
handing off to the draft. This keeps the draft's context unchanged.

### 1.3 Measurements (M4 Max, default mio tiers)

1784-token system prompt, 5-call session (call 1 cold, call 2 miss+warm-up,
calls 3-5 warm hits). Reported as `cold_wall → warm_hit_avg`.

| Tier              | Cold wall (ms) | Warm-hit avg (ms) | Speedup |
|-------------------|----------------|-------------------|---------|
| small (Qwen3.5-4B) | 1490           | 246               | **6.05×** |
| medium (Qwen3.5-9B) | 2489          | 339               | **7.34×** |
| large (Qwen3.5-27B) | 8720          | 1014              | **8.60×** |
| large-moe (35B-A3B) | 1174          | 250               | **4.71×** |

Best single hit: small tier call 4, **1490ms → 114ms** (13×).

On isolated prefill-only benchmarks (bypassing the engine), prefill time drops
**19.8×** (482 → 24 ms) for a 645-token prefix on small tier.

### 1.4 Gating

Prefix cache is disabled (falls back to full prefill) when:
- `tq_bits ∈ {2, 3, 4}` (TurboQuantKVCacheV2 uses pre-allocated buffers that
  don't snapshot via simple dict storage).
- `bmp_paths ≥ 2` (BMP-DFlash requires fresh caches for batch expand/filter).
- PARO quantization path (untested, gated off).

Implementation: `MioEngine._prefix_cache_enabled()`.

### 1.5 Limits & future work

- **Memory**: each entry holds a full KV + SSM state. Small tier at 1000 tokens ≈ 150 MB.
- **Radix-tree indexing**: current lookup is linear over ≤4 entries (O(N·L));
  SGLang-style radix tree would be O(L).
- **Persistence**: disk-backed cache across sessions not yet implemented.
- **Extend to TQ4**: give `TurboQuantKVCacheV2` a snapshot/restore API to
  lift the gating condition.

### 1.6 Relaxed lookup with KV truncation (tool-calling agents)

The strict-prefix lookup (`cached_tokens` must be a verbatim prefix of new
prompt) works for plain chat but misses every cross-turn lookup when Qwen's
template adds a `<think>\n\n</think>\n\n` wrapper to the *current* assistant
turn but not to prior assistant turns rendered in history. The stored key
diverges from the next turn's prompt by ~10 tokens in the middle →
strict-prefix match fails → 0% hit rate on multi-turn agent sessions.

Fix: longest-common-prefix lookup + runtime KV truncation to the match
length. Implementation:

1. `_prefix_cache_lookup(prompt_tokens)` scans all entries, picks the one
   with the longest common prefix (not strict prefix), returns it with
   `offset = match_length`. Entries below `_prefix_cache_min_tokens` are
   still gated to avoid negative speedups on tiny matches.

2. `_truncate_warm_state(entry, length)` trims each cached structure to
   `length` positions *before* the runtime warm-start:
   - `target_cache` → `trim_cache_to` (wraps `mlx_lm.models.cache.trim_prompt_cache`).
   - `draft_cache` (`ContextOnlyDraftKVCache`) → slice `keys`/`values`
     tensors to `[:, :, :length, :]`, set `offset = length`.
   - `target_hidden` → slice to `[:, :length, :]`.

3. Matched entries are removed from the cache map on lookup (rented out).
   The post-generation state of the current request is re-stored under
   a fresh key. This avoids shared-state mutation when multiple requests
   would otherwise compete for the same entry.

### 1.7 Production measurements (Kilo agent, 50-turn session)

Real workload: Kilo Code coding agent running a backend code-audit task
against the Qwen3.5-35B-A3B large-moe tier. Session covers 50 HTTP
requests with growing tool-call conversation history and ~88K-token
context at steady state.

| Metric | Value |
|---|---|
| Cache hits | **47 / 50 (94%)** |
| Tokens saved (cumulative) | **1,679,640** |
| Peak prefill rate | **173,799 tok/s** (burst on near-full-prefix hit) |
| Avg prefill rate | 30,437 tok/s (vs ~1,000 tok/s cold) |
| DFlash acceptance | 0.38–0.54 on short tool-call responses (EOS-suppression window dominates) |
| Representative warm-hit wall | ~2.5 s total (prompt=8.9K, gen=388 tok) |

For comparison, on the same request shape *without* the relaxed lookup,
every turn is a miss and mio re-prefills the full 88K-token context
(~50 s wall per turn). The relaxed lookup makes the difference between
a usable agent and a dead one at this context length.

### 1.8 Integration with OpenAI function calling

The cache is the load-bearing optimization for multi-turn tool-calling
agent loops. For each hit to work end-to-end, `server.py` must round-trip
the assistant's `tool_calls` and the client's `role=tool` messages through
Qwen's chat template byte-identically — otherwise the rendered assistant
turn in the next request diverges from the cached key at the tool call
itself and the prefix match collapses. Implementation:

- Preserve `tool_calls`, `tool_call_id`, and `name` fields when unpacking
  `ChatCompletionRequest.messages` (Pydantic `extra: "allow"` tolerates
  them; `server._coerce_message_content` explicitly carries them).
- Convert OpenAI's `arguments: str` (JSON string) back to `arguments: dict`
  before calling `apply_chat_template`, since Qwen's template iterates
  arguments as a mapping.
- Let Qwen's template render `role=tool` as `<|im_start|>user\n<tool_response>...</tool_response><|im_end|>` — no custom handling needed; the template
  already handles it correctly once `tool_call_id` is present.

Combined with the relaxed lookup + EOS-suppression window (first 40
output tokens, see `mio/dflash/runtime.py:stream_dflash_generate`
`relax_suppress_after`), tool-call emission is now reliable, acceptance
stays ≥0.80 on longer responses, and the cache carries nearly all of
the conversation prefix across turns.

## 2. LM-Head Slicing (only_last_logit)

### 2.1 Rationale

During prefill, the only logit we consume is the last position's:
```python
staged_first = greedy_tokens_with_mask(prefill_logits[:, -1, :], ...)
```

Yet `target_forward_with_hidden_states` was projecting every position through
the LM head (`hidden_dim → vocab_size`). For Qwen3.5-4B at 2000 tokens,
that's 2000 × 3072 × 248,000 ≈ 1.5 TFLOPs of wasted computation.

### 2.2 Fix

Added `only_last_logit: bool = False` to `target_forward_with_hidden_states`.
When set, the function slices the post-layer-norm hidden state to
`h[:, -1:, :]` before passing through the LM head. Hidden states used by the
draft are fully captured regardless, so draft quality is unaffected.

Enabled on the prefill path in `generate_dflash_once`.

### 2.3 Measurements

Pure prefill tokens-per-second at representative prompt lengths:

| Tier    | Prompt | Before | After | Speedup |
|---------|--------|--------|-------|---------|
| small   | 512    | 1403   | 1612  | **+14.9%** |
| small   | 2048   | 1385   | 1594  | **+15.1%** |
| medium  | 512    | 777    | 879   | **+13.1%** |
| medium  | 2048   | 777    | 880   | **+13.2%** |

The speedup is roughly constant across prompt lengths because the LM-head
compute grows linearly in prompt tokens. Remaining prefill time is dominated
by attention + MLP compute in the transformer stack.

### 2.4 Why this wasn't already the default

The mlx-lm reference implementation does project all positions — that's needed
for training-style loss computation (logits at every position become labels for
next-token cross-entropy). For inference with greedy decoding + DFlash-style
speculative verify, we only consume last-position logits at prefill; during
verify the shape differs anyway (multi-token input over a short block, all
positions matter).

## 3. Combined effect on interactive chat

Typical coding agent turn: 1000-token system prompt + 50-token user message.
With both optimizations:

- **Turn 1 (cold)**: baseline prefill × (1 − 0.13) ≈ 87% of old prefill time.
- **Turn 2 (miss + warm-up)**: same as baseline + one-time amortized cost.
- **Turn 3+ (warm hits)**: prefill time reduced to the 50-token suffix × 1.15
  ≈ 5% of the cold prefill time. That's **~20× effective TTFT reduction**
  on the stable portion of the conversation.

## 4. Files

- `mio/engine.py` — `_prefix_cache_*` methods.
- `mio/dflash/runtime.py`:
  - `target_forward_with_hidden_states` — `only_last_logit` parameter.
  - `generate_dflash_once` — `warm_state`, `return_final_state`, `prefill_only`.
- `tests/test_prefix_cache.py` — 5 unit tests (lookup/store/LCP/margin/gating).
- `scripts/bench_prefill.py` — cold-path microbench.
- `scripts/bench_prefix_cache.py` — 4-tier warm-hit bench.
- `docs/09-prefix-cache.md` — user-facing guide.
