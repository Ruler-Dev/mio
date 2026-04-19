# Prefix Cache: Skip Re-Prefilling Shared Prompt Prefixes

Mio's prefix cache automatically detects when consecutive prompts share a
long common prefix (system prompt, conversation history, agent directives)
and skips re-prefilling that portion on hit. Implementation is in
`mio/engine.py` (`_prefix_cache_*` methods) and
`mio/dflash/runtime.py` (`warm_state` parameter of `generate_dflash_once`).

## Production numbers (Kilo agent session)

A 50-turn Kilo agent session running code audits against the Qwen3.5-35B-A3B
large-moe tier produced:

- **47 / 50 cache hits** (94%)
- **1,679,640 tokens saved** — tokens that would have been re-prefilled
- **Peak prefill 173,799 tok/s** during burst-hit turns
- Average prefill 30,437 tok/s (vs ~1,000 tok/s cold)

Per-turn wall time on a warm hit collapses from ~15 seconds of prefill to
~5 seconds total (mostly decode of the novel tool-call output). At 88k-token
contexts, this is the difference between a usable agent and a dead one.

The hit rate came from the **longest-common-prefix lookup with KV
truncation** (see "Relaxed lookup" below) — strict-prefix matching missed
every cross-turn lookup on Qwen3.5 due to the template's `<think>` wrapper,
producing 0% hits before the fix.

## What it does

Interactive chat and agent sessions resend a large chunk of the prompt every
turn — a long system prompt, any tool-calling rules, and the running
conversation. That prefix hasn't changed, but every call's prefill pass
recomputes the full KV cache for it. On a 1000-token system prompt with the
small (4B) tier, that's ~700ms wasted per turn.

The prefix cache:

1. After each successful `generate()`, looks at the previous prompt and
   the current one, takes their **longest common prefix**, and if it's
   ≥ `_prefix_cache_min_tokens` (default 64), runs a `prefill_only` pass on
   that prefix. The resulting KV + SSM cache state is stored in-memory keyed
   by the token tuple.
2. Before the next `generate()`, looks up the longest cache entry whose
   tokens match the start of the incoming prompt.
3. On a hit, passes the cached state to `generate_dflash_once` as
   `warm_state`. The target and draft models skip forward on the cached
   range and prefill only the novel suffix.

## Measured speedup

From `scripts/bench_prefix_cache.py`, repeated calls sharing a ~1784-token
system prompt, calls 3-5 are warm hits (call 1 cold, call 2 = miss + one-time
warm-up), 16 generated tokens each:

| Tier | Prompt | Cold wall | Warm avg | **Speedup** |
|------|--------|-----------|----------|-------------|
| small (4B)       | 1784 | 1490 ms  | 246 ms  | **6.05×** |
| medium (9B)      | 1784 | 2489 ms  | 339 ms  | **7.34×** |
| large (27B)      | 1784 | 8720 ms  | 1014 ms | **8.60×** |
| large-moe (35B-A3B) | 1784 | 1174 ms  | 250 ms  | **4.71×** |

Best hits saw 12-15× speedup (e.g., small tier call 4: 114 ms vs 1490 ms cold).

On isolated prefill-only tests (not routed through the engine), warm cache
gave **19.8× prefill speedup** (482 ms → 24 ms) and **3.0× end-to-end**.

For a coding agent that fires a large system prompt + conversation history
on every user turn, this is by far the largest single TTFT win in mio.

## How it works (technical)

`generate_dflash_once` gained two parameters:

```python
generate_dflash_once(
    ...,
    warm_state: dict | None = None,  # {"target_cache": [...], "draft_cache": [...], "offset": N}
    return_final_state: bool = False,
    prefill_only: bool = False,      # short-circuit after prefill, skip decode loop
)
```

When `warm_state` is set, the target and draft caches are reused as-is and
only `prompt_tokens[offset:]` is fed to the target for prefill. The rest of
the loop is unchanged — the cache shapes and offsets line up because we
truncate the cached KV to `offset` before prefill.

`MioEngine` manages a small dict of (token_tuple → final_state) entries,
LRU-evicted at `_prefix_cache_max_entries` (default 4). Entries are cache-
populated only when two consecutive prompts share a ≥64-token prefix —
a one-shot cost amortized over subsequent hits.

## Relaxed lookup + KV truncation

The original strict-prefix lookup required `cached_key` to be a verbatim
prefix of the new prompt tokens. This worked for plain chat but broke on
Qwen3.5's tool-calling template:

- Turn N generates with `enable_thinking=False` → template prefills
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` before the assistant
  content.
- Turn N+1 renders the *prior* assistant turn as history (no
  `add_generation_prompt`) → the template emits `<|im_start|>assistant\n` +
  content, **without** the `<think>...</think>` wrapper (Qwen only wraps the
  current assistant turn).
- Result: cached key diverges from new prompt by ~10 tokens in the middle,
  strict-prefix match fails, and every cross-turn lookup is a miss.

The fix in `_prefix_cache_lookup` finds the **longest common prefix** across
all entries and returns the matching entry with `offset` set to the match
length. `_truncate_warm_state` then trims each cached KV structure to that
length before runtime warm-start:

- `target_cache`: `trim_cache_to` delegates to mlx_lm's `trim_prompt_cache`
  (the same primitive mlx_lm uses for its own prompt caching).
- `draft_cache` (`ContextOnlyDraftKVCache`): `keys`/`values` tensors sliced
  to `[:, :, :length, :]` and `offset` set to `length`.
- `target_hidden` (context feature tensor): sliced to `[:, :length, :]`.

Matched entries are **removed from the cache map** ("rented out") before
return so concurrent or later requests don't see the mutated state. The
new post-generation state is stored under a fresh key after the request
completes.

On a typical Kilo turn the match recovers ~99% of the cached prefix
(everything up to the `<think>` wrapper divergence point), leaving only
tens of tokens to prefill fresh. This is what the 94% hit rate in the
production numbers above comes from.

## Interaction with tool calling

The cache is the key to fast multi-turn agent loops. Each turn Kilo sends:

1. The same 8-9k-token system prompt (Kilo tool descriptions + project
   CLAUDE.md) — **never changes**.
2. The full message history with prior `tool_calls` + `role=tool` results —
   grows each turn, but the head stays stable.
3. Whatever came back from the last tool invocation.

The cache's relaxed lookup matches from position 0 up to wherever the new
turn diverges from the cached key, which is typically at the assistant
turn boundary (the `<think>` wrapper issue). For a 50-turn audit session
this means we re-prefill ~200 tokens per turn instead of 90k+.

`server.py` preserves `tool_calls`, `tool_call_id`, and `role=tool`
messages through to Qwen's chat template so the template re-renders prior
turns byte-identically to what we generated. Without that, the assistant
tokens diverge at the tool call itself (Kilo would store `content=""` but
Qwen's template expects `tool_calls: [...]` rendered as XML), which would
break the cache prefix almost immediately.

## Gating (when the cache is OFF)

The cache is automatically disabled when any of these are true:

- `tq_bits ∈ {2, 3, 4}` (TurboQuant KV quantization): the
  `TurboQuantKVCacheV2` state includes pre-allocated buffers and rollback
  tapes that don't snapshot via simple dict storage.
- `bmp_paths ≥ 2` (BMP-DFlash): the batch-expand/filter logic in BMP requires
  fresh caches per round.
- PARO-quantized models: not yet tested, untested path.

In those cases `generate()` falls back to vanilla DFlash with full prefill.

## Memory

Each cache entry holds one full KV + SSM state for its cached prompt. For
Qwen3.5-4B at 1000 tokens, that's ~150 MB. The `_prefix_cache_max_entries`
default is 4; adjust via `engine._prefix_cache_max_entries = N`. Entries are
evicted LRU.

## Invalidation

Call `engine._prefix_cache_invalidate()` to drop all entries. You should do
this when:

- Switching model (tier change).
- Changing sampler (temperature, stops) — currently the cache ignores
  sampler state because mio is greedy-only.
- Before benchmarks that measure cold-start TTFT.

## Future work

- **Radix tree indexing**: rather than iterating entries linearly for the
  lookup (`O(N*L)` token comparisons), SGLang-style radix tree would reduce
  lookup to `O(L)`. At N≤4 entries the current linear scan is negligible.
- **Chunked prefill with prefix-aware chunking**: aligning chunk boundaries
  to cached prefix lengths for better batching.
- **TQ-aware prefix cache**: extend `TurboQuantKVCacheV2` with snapshot/restore
  so the gating condition can be lifted.
- **Disk persistence**: write cache entries to disk at session end; reload
  on next session. Would give zero-TTFT on the first call of every session.
