# kv-experimentation branch — benchmark results

**Status:** benchmark in progress. This document will be populated from
`/tmp/bench-runs/large_moe_results.json` once the run completes.

## Scope

- Target tier: `large-moe` (Qwen3.6-35B-A3B-UD-Q4_K_XL, 21 GB target + 913 MB draft).
- Context sizes: 4K, 16K, 32K tokens.
- Coding prompts: `fib`, `sort_bug`, `bst`, `nqueens` (padded with project-like code to hit target context size).
- Repeats: 2 per cell, min reported (reduces warmup / jitter noise).
- Sampling: greedy via DFlash (deterministic at the token level *when the KV cache path is identical*).

## What is being measured

### 1. Baseline

Default mio config on `large-moe`: PolarQuant-4 KV cache + DFlash speculative decoding. This is what `mio serve` runs today.

### 2. DDTree

`MIO_DDTREE_BUDGET=4` — 8-bit KV cache (mlx_lm QuantizedKVCache) + DFlash tree-attention verify. Incompatible with PolarQuant, so it trades 4-bit KV compression for tree-speculation throughput.

### 3. Frozen KV (C3)

Explicit warm-and-freeze pass: engine runs `prefill_only` on the first N-1 prompt tokens, serializes `(target_cache, draft_cache, target_hidden)` via safetensors, then reloads on the next request and skips prefill for the covered portion.

### What is NOT benchmarked

- **Speculative Prefill (B2)** — the partial-target-forward hook is a `raise NotImplementedError`. The runtime always falls back. Measuring it would only show baseline + Python overhead.

## Expected outcome shape

(to be replaced with actual numbers once the bench completes)

### Baseline vs DDTree decode speed (gen tok/s)

| prompt   | ctx 4K | ctx 16K | ctx 32K |
|----------|-------:|--------:|--------:|
| fib      | TBD    | TBD     | TBD     |
| sort_bug | TBD    | TBD     | TBD     |
| bst      | TBD    | TBD     | TBD     |
| nqueens  | TBD    | TBD     | TBD     |

### Frozen KV prefill speedup (warm vs cold)

| ctx  | cold prefill (ms) | warm prefill (ms) | speedup |
|------|------------------:|------------------:|--------:|
| 4K   | TBD               | TBD               | TBD     |
| 16K  | TBD               | TBD               | TBD     |
| 32K  | TBD               | TBD               | TBD     |

### Quality check (hashes)

Exact output sha256 is reported per run. Three expected outcomes:

1. **baseline cold vs. baseline rep1**: same sha (deterministic greedy).
2. **frozen-KV cold vs. frozen-KV warm**: may differ in the *paraphrase tail* after a tied-logit token flips under minor FP noise, but the *code portion* should be equivalent. We verify semantic code equivalence by inspecting the first 800 chars.
3. **baseline vs. DDTree**: expected to differ — 8-bit KV vs. PolarQuant-4 gives different attention outputs. This is not a regression; it's a precision-trade.

## Implementation notes (from debugging)

### DDTree tuple bug (fixed on this branch)

`verify.py` at head of this branch now handles `QuantizedKVCache.update_and_fetch`'s 3-tuple return correctly — it previously assumed `keys.shape[2]`, which only works for plain `KVCache`. Also the long-context split-SDPA path is bypassed under quantized cache (the split code does raw array slicing that can't operate on the quantized tuple form).

Since `main` still ships DDTree with the old bug, **this fix should be cherry-picked to `main`** — DDTree on main can't actually run end-to-end on hybrid targets like the large-moe tier. That's a real, user-facing bug hidden behind `MIO_DDTREE_BUDGET=4`.

### Frozen-KV auto-freeze disabled in the generate path

The earliest wiring auto-froze post-`generate` state (cache at offset = prompt_len + gen_len with rollback tape residue on recurrent layers). Truncating that back to prompt_len is impossible on hybrid models — the recurrent layers have accumulated state that can't be rolled back past the trim boundary. Fix: automatic freeze is off; clean snapshots are only produced by the explicit `MioEngine.warm_and_freeze(messages)` entry point, which runs `prefill_only` (no decode mutations) and freezes exactly `prompt_len - 1` tokens of state so the runtime's mandatory 1-token prefill fills the last slot.

### Why prompt_len - 1

The runtime clamps `warm_offset >= prompt_len` down to `prompt_len - 1` to keep at least one token for the post-prefill logits sampling pass. If we froze a cache of prompt_len tokens, the runtime would append the last prompt token *again* at position prompt_len, producing a duplicate-context bug. Freezing prompt_len - 1 makes the runtime's minimum prefill correctly fill the last-token slot exactly once.
