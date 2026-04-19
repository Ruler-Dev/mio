# TurboQuant KV-Cache Compression

## What Is TurboQuant?

TurboQuant compresses the Key-Value cache during inference. The KV cache stores attention state for all previous tokens -- it grows linearly with context length and is the main memory bottleneck for long contexts.

TurboQuant V2 uses hardware-accelerated affine quantization via MLX's `mx.quantized_matmul` Metal kernel. 3.6x cache compression at near-native speed.

## How It Stacks with PARO and DFlash

All three are orthogonal:
- **PARO**: better model weights (INT4 with rotation) -- loaded once
- **DFlash**: fewer forward passes (speculative decoding) -- during generation
- **TurboQuant**: smaller KV cache (quantized) -- during attention

Combined: better weights + faster decoding + less memory per token.

## Selecting TQ Mode

**TurboQuant is OFF by default.** Enable with the `--tq4` flag on any
entrypoint:

```bash
mio --tq4
mio chat --tq4
mio serve --tq4
```

Inside the agent (legacy interactive selector), type `/context` to switch.
Programmatically: set `TierConfig.tq_bits = 4` (or 3 / 2) before constructing
the engine.

```
> /tq
TurboQuant V2:
- Bits: 4
- Group size: 64
- Rotation: True
- Normalization: True
- QJL: False
```

## TQ Cache Options

| Setting | Compression | Decode (vs baseline) | When it wins |
|---------|-------------|---------------------|--------------|
| **OFF** (default) | 1.0× | 1.00× | Default; matches CLAUDE.md ~204 tok/s on large-moe |
| 4-bit | 3.6× | 0.7-0.9× small/medium/MoE; **1.67× on 27B-dense at 32K** | When KV bandwidth is the decode bottleneck (long context, dense attention) |
| 3-bit | 4.7× | ~0.6× | Memory-tight long-context |
| 2-bit | 5.5× | ~0.5× | 256K+ context where fp16 KV won't fit |

Measured TQ4 vs OFF on M4 Max (see `papers/prefill-speedups.md` and
`scripts/bench_tq4_context.py`):

| Tier | Context | OFF gen t/s | TQ4 gen t/s | KV (OFF) | KV (TQ4) |
|------|---------|-------------|-------------|---------|---------|
| small (4B)        | 7 680   | 94.6  | 83.5 (-12%) | 0.26 GB | 0.07 GB |
| medium (9B)       | 15 872  | 59.4  | 54.9 (-8%)  | 0.53 GB | 0.15 GB |
| large (27B-dense) | 32 256  | 10.3  | **17.2 (+67%)** | 2.13 GB | 0.60 GB |
| large-moe (35B-A3B) | 130 560 | 18.2  | 9.6 (-47%)  | 2.68 GB | 0.76 GB |

**When TQ4 beats baseline**: 27B-dense at 32K — attention is KV-bandwidth-
bound and shrinking KV is a clear win.

**When TQ4 loses**: MoE at 128K — only 3B params active per token, so KV
bandwidth isn't the bottleneck; per-step quantize/dequantize overhead
dominates. Stay on baseline.

## Context Impact (35B-A3B MoE Default)

| Context | fp16 KV | TQ 4-bit | TQ 2-bit | Total VRAM (TQ4) |
|---------|---------|----------|----------|-------------------|
| 32K | ~0.5 GB | ~0.1 GB | ~0.1 GB | ~18 GB |
| 128K (default) | ~2.0 GB | ~0.5 GB | ~0.3 GB | ~18.5 GB |
| 256K | ~4.0 GB | ~1.0 GB | ~0.5 GB | ~19 GB |
| 512K | ~8.0 GB | ~2.0 GB | ~1.0 GB | ~20 GB |
| 1M | ~16 GB | ~4.0 GB | ~2.0 GB | ~22 GB |

Without TQ, 1M context = 16 GB cache (tight). With TQ 2-bit, 1M = 2 GB (easy).
