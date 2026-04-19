"""DGSA — Draft-Guided Sparse Attention for Qwen3.5 hybrid_gdn prefill.

Novel technique invented for mio. Combines:

  1. SpecPrefill's idea of using a draft model to score per-token importance.
  2. Skip-Softmax-style attention sparsification (NVIDIA, 2026): skip
     attention compute on low-contribution blocks/tokens.
  3. Hybrid SSM compatibility: SSM (gated delta-net) layers process the FULL
     prompt — only ATTENTION layers compute against a sparse subset of keys.

Why it's needed: SpecPrefill drops tokens from the input. That works on
pure-attention transformers but breaks SSM layers (recurrent state cannot
have tokens skipped). Skip-Softmax skips attention compute but uses
heuristic in-target signals. DGSA gets the best of both: full-prompt SSM
state (correctness) + draft-scored sparse attention (speed).

Per-prefill-step:
  1. Score prompt tokens via DFlash draft (or self-early-exit) attention.
  2. Pick keep_indices = top-K + recent window + sinks.
  3. Forward target on FULL prompt with attention layers patched:
       Q, K, V computed for ALL positions
       RoPE applied to ALL positions (correct absolute positions)
       Then K, V SLICED to keep_indices before SDPA
       Output is dense over query positions
  4. KV cache stores the sliced K, V; cache.offset = full prompt length.

Decode runs unchanged — new tokens append to cache and attend to all
(sliced-prefill + full-decode) entries.

See `experimental/notes/dgsa_findings.md` for results.
"""
