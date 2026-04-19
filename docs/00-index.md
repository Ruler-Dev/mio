# Mio Documentation Index

## User docs

- [01 — Getting Started](./01-getting-started.md) — install, download, first run
- [02 — Commands](./02-commands.md) — full CLI reference (incl. `--tq4`, `--mpath`, `--caveman`)
- [03 — Deployment](./03-deployment.md) — server, Docker, monitoring
- [04 — Customization](./04-customization.md) — system prompts, model wiring
- [05 — Models](./05-models.md) — tier matrix, model registry
- [06 — TurboQuant](./06-turboquant.md) — KV-cache quantization (TQ4 default OFF, opt-in)
- [08 — BMP-DFlash](./08-bmp-dflash.md) — Batched Multi-Path speculative decoding
- [09 — Prefix Cache](./09-prefix-cache.md) — automatic shared-prefix reuse
- [11 — Mio UI](./11-mio-ui.md) — Web interface: artifacts, skills, weather widget, document generation

## Technical notes / "papers"

- [BMP-DFlash technical note](../papers/bmp-dflash.md) — algorithm derivation, correctness proofs, full bench tables
- [Prefill speedups](../papers/prefill-speedups.md) — prefix cache + LM-head slicing, per-tier numbers

## Experimental (non-production)

Code under `experimental/` does not affect production. Wiring decisions deferred per item.

- [Speculative Prefill on Qwen3 (pure attention)](../experimental/notes/spec_prefill_findings.md) —
  2.5-2.6× prefill speedup on Qwen3-8B at 20% keep ratio. Does NOT work on
  Qwen3.5 hybrid_gdn (SSM layers can't drop tokens).
- [DGSA — Draft-Guided Sparse Attention (negative result)](../experimental/notes/dgsa_findings.md) —
  Novel attention-only adaptation for hybrid SSM models. Architecturally
  ceiling-bound to ≤1.08× because attention is only 8% of Qwen3.5 prefill
  compute (MLP 57%, SSM 35%). Documented as definitive negative result.
- [MLP speedup investigation (negative result)](../experimental/notes/mlp_speedup_findings.md) —
  Tried `mx.compile`, activation zero-masking, etc. on Qwen3.5 MLP. None
  deliver speedup; 4-bit grouped matmul kernels are already at hardware
  peak. Further wins require custom Metal sparse-matmul kernels or model
  retraining.

## What's currently optimised

| Optimization | Default | Opt-in flag | Status |
|--------------|---------|-------------|--------|
| Caveman system prompt (level: ultra) | **on** | `--no-caveman` to disable; `--caveman {off,lite,full,ultra}` to set level | Production |
| DFlash speculative decoding | **on** | (always on when draft loaded) | Production |
| LM-head slicing (`only_last_logit`) | **on** | (auto, no flag) | Production, +13-15% cold prefill |
| Prefix cache | **on** | (auto, no flag; gated off when TQ4 / BMP active) | Production, 4.7-8.6× warm hits |
| TurboQuant 4-bit KV cache | off | `--tq4` | Production, opt-in |
| BMP-DFlash multi-path verify | off | `--mpath K` | Production, opt-in (K=2 wins on Qwen3-8B math; not on Qwen3.5) |
| Speculative Prefill | off | (experimental only) | `experimental/spec_prefill/`, Qwen3-8B only |

## Latest measurements (M4 Max, default tiers)

`large-moe` (Qwen3.5-35B-A3B) baseline: 202 gen tok/s, 0.89 acceptance, 9.14 avg accept length.
Unchanged through every optimisation in this matrix when defaults are used.

For prefill TTFT: with prefix cache active on a long shared prompt (≈1700 tokens),
warm calls hit 4.7-8.6× faster than cold across all four tiers
(see `papers/prefill-speedups.md` §1.3).
