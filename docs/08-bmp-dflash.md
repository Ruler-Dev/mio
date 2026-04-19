# BMP-DFlash: Batched Multi-Path Speculative Decoding on Qwen3.5

Mio's BMP-DFlash is a hybrid-architecture-compatible adaptation of DDTree
(Ringel & Romano, 2026). It extends vanilla DFlash speculative decoding by
verifying **K parallel continuations** of every draft block in a single target
forward pass — picking the continuation with the longest accepted prefix.

## Why this exists

The DDTree paper reports 1.30×–1.50× speedups over vanilla DFlash on Qwen3
models. Its trick is tree attention: B tree nodes in one target forward, each
node attending only to its ancestors. **That trick does not apply to models
with SSM / linear-attention layers**, because SSM state is recurrent — you
cannot branch it inside a single sequence forward without corrupting state
across branches.

mio's default targets are all **Qwen3.5 family** (gated-delta-net + full
attention hybrids). So the paper's tree-attention mechanism is architecturally
unavailable on the exact models we run.

BMP-DFlash sidesteps the problem by branching in the **batch dimension**
instead of the sequence dimension. Attention, SSM, and FFN all process each
batch row independently by construction, so correctness is preserved on any
architecture — including hybrid_gdn.

## How it works

Per decoding round, given the bonus token `b` produced by the previous target
step:

1. **Draft**: run the DFlash block-diffusion draft once. This produces `L`
   per-position marginal distributions `q₁, …, q_L` for the next L positions.

2. **Build K paths**: use the DDTree best-first heap (Algorithm 1 of the paper)
   on the draft's per-position top-K tokens to select the K highest-scoring
   root-to-leaf token sequences. Each path has length ≤ L; shorter paths are
   padded with the marginal-argmax continuation so every path has length
   `block_len`.

3. **Verify**: stack the K paths as a `(K, block_len)` batch. Expand the
   target model's caches along the batch dimension. Run ONE target forward
   pass — K parallel trajectories, each processed independently through every
   layer (SSM included).

4. **Pick winner**: for each row k, compute the acceptance length
   `α_k = max{i : verify[k, 1..i] all equal target_argmax[k, 0..i-1]}`.
   Winning row `k* = argmax α_k`. Ties break toward lower index so path 0
   (the vanilla DFlash argmax path) wins by default — gives a predictable
   fallback.

5. **Commit**: filter every cache's batch dim down to `[k*:k*+1, …]` and use
   DFlash's existing rollback mechanism to rewind the SSM tape from
   `block_len` steps back to the accepted prefix length. No replay forward
   needed.

## Guarantees and limits

- **Correctness is preserved**: each batch row is a standalone linear
  trajectory through the target model, so SSM state is valid per row.
- **Acceptance is monotone non-decreasing** in K: the K=1 path is the vanilla
  DFlash argmax trajectory, so BMP acceptance_length ≥ DFlash acceptance_length
  by construction (tie-break on equality favors path 0).
- **Not stackable with TurboQuant 4-bit today**: the engine disables BMP when
  `tq_bits ∈ {2, 3, 4}` and falls back to vanilla DFlash. Lifting this is
  pending cache-layer plumbing in `TurboQuantKVCacheV2`.

## Cost model

Per verify step on a hybrid_gdn model, the work is roughly:
- Vanilla DFlash: one batch=1 forward of `block_len` tokens.
- BMP K=2: one batch=2 forward of `block_len` tokens. MLX batches attention
  and MLPs efficiently, so the wall-clock is usually between 1.0× and 1.5×
  of vanilla — not 2×.

BMP wins when the K-path acceptance gain exceeds this K-batch overhead. In
practice on Qwen3.5:
- K=2 is often a near-wash unless acceptance is below ~70%; it helps on
  longer, lower-confidence decodes.
- K=3 consistently improves tokens/cycle but sometimes not wall-clock.
- K=4 rarely beats K=2 or K=3 because draft marginals past the top 2 alternatives
  at position 1 are usually very low probability.

Run `scripts/bench_bmp.py --tiers small,medium --max-tokens 128` to measure
on your hardware.

### Measured on M4 Max

**Qwen3-8B-4bit (pure attention, `mio pull qwen3-8b-4bit`)**:
- math: vanilla **74.0 tok/s** → BMP K=2 **93.3 tok/s** (**1.26×**); tpc
  5.12 → 6.74; accept 0.80 → 0.85.
- prose: vanilla 28.3 → K=2 29.6 (1.05×, marginal).
- code: K=2 regresses slightly (39.9 vs 43.9 vanilla).
- K=3/4: worse than K=2 on every prompt — verifier overhead dominates.

**Qwen3.5 family (default tiers, hybrid_gdn)**:
- Every tested (tier × prompt × K) regresses vs vanilla DFlash. DFlash
  acceptance on Qwen3.5 is already 0.77–0.88, leaving almost no headroom.

So the picture is: BMP K=2 is worth turning on for **pure-attention Qwen3 on
math-like prompts**, and off everywhere else. See `papers/bmp-dflash.md` §5
for full tables.

Usage:
```bash
# BMP K=2 on Qwen3-8B (after `mio pull qwen3-8b-4bit`)
# Currently requires a custom tier config; the default tiers are Qwen3.5.
```

**Don't turn `--mpath 2+` on Qwen3.5 tiers for throughput.** The
implementation is correct and opt-in; keep it for sampling mode / weaker
drafters / adaptive-K future work.

## Using it

CLI flag, available on all entrypoints:

```bash
mio --mpath 2                  # native agent with BMP K=2
mio chat --tier medium --mpath 3
mio serve --tier large-moe --mpath 2
```

Programmatic:

```python
from mio.config import MioConfig
from mio.engine import MioEngine

cfg = MioConfig.default()
cfg.tiers["medium"].bmp_paths = 3
eng = MioEngine(tier_config=cfg.tiers["medium"])
eng.load()
text, metrics = eng.generate([{"role": "user", "content": "..."}], max_tokens=256)
```

Default is K=1 (vanilla DFlash). Opt in explicitly — see `scripts/bench_bmp.py`
for per-tier bench numbers before turning it on in production.

## Files

- `mio/dflash/ddtree.py` — Algorithm 1 builder + path enumeration.
- `mio/dflash/bmp.py` — batch helpers (expand/filter/snapshot),
  per-row acceptance, path batch construction.
- `mio/dflash/bmp_runtime.py` — the `generate_bmp_dflash_once` entrypoint.
- `mio/dflash/recurrent_rollback_cache.py` — extended `filter()` that
  slices rollback tapes in lock-step with the cache state.
- `papers/bmp-dflash.md` — technical note with derivation and benchmark data.
