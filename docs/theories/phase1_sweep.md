# Phase 1 knob sweep — runtime config tuning

**Question:** are there free speedups from tuning existing runtime knobs?
**Answer:** marginal — mio defaults are already well-tuned.

## Setup

- Target: `large-moe` (Qwen3.6-35B-A3B-UD-Q4_K_XL, hybrid 48 GatedDelta + 16 attention).
- Measurement: end-to-end generate with DFlash, 128 output tokens, warm rep (after 1 warmup run per cell). Prefill ms derived from metrics.prompt_tps.
- Knobs: `split_full_attention_sdpa` chunk_size ∈ {8, 16, 32, 64}, PolarQuant bits ∈ {4, 16}.
- Baseline: PQ=4, chunk_size=8 (mio default).

## Results — delta vs baseline (PQ=4, chunk=8)

### chunk_size effect at PQ=4 (keeping PolarQuant on)

| ctx | chunk=8 (base) | chunk=16 | chunk=32 | chunk=64 |
|----:|---------------:|---------:|---------:|---------:|
| 2K | **1,441 ms** | +18.1% | +7.0% | +15.1% |
| 4K | **2,869 ms** | +13.6% | +2.8% | +12.9% |
| 8K | **6,056 ms** | +5.9% | +0.1% | +7.6% |
| 16K | 13,441 ms | **−6.8%** | −4.8% | −2.5% |

**Read:** chunk=8 is best below 16K. At 16K, chunk=16 wins by 6.8% (~900 ms). The default is already optimal for mio's typical 1K–8K workload.

### PolarQuant on vs off at chunk=8

| ctx | PQ=4 (on, baseline) | PQ=16 (off) | slowdown |
|----:|--------------------:|------------:|---------:|
| 2K | 1,441 ms | 1,693 ms | **+17.5%** |
| 4K | 2,869 ms | 3,539 ms | **+23.4%** |
| 8K | 6,056 ms | 7,760 ms | **+28.1%** |
| 16K | 13,441 ms | 18,119 ms | **+34.8%** |

**Read:** disabling PolarQuant is *worse*, not better. PQ's Hadamard rotation + 4-bit quantize is cheap per write; the big saving is reading the cache with 4× less bandwidth during subsequent attention ops, which dominates prefill at long context.

Generation tok/s also drops with PQ off: 100 → 74 tok/s at 16K (26% slower).

## Takeaway

- **No meaningful free speedup** from these two knobs. Default config is near-optimal.
- **Finding worth noting**: PolarQuant accelerates prefill by 17-35%. It's been framed as a "compression with zero speed loss" feature; it's actually a prefill *accelerator* at long context, and the PQ-off path is genuinely slower to serve. (Documentation nit — CLAUDE.md says "zero speed overhead with DFlash"; for prefill specifically it's *negative* overhead.)
- **One micro-win available**: auto-select chunk_size=16 when N ≥ 16K, chunk=8 otherwise. +900 ms saved at 16K. Trivial to ship but not a research result.

Committed the full matrix at `experiments/phase0_sweep/results.json`.

## Next

Knob tuning exhausted. Moving to C1 LowRank-QK calibration — attention is 4.7-22.7 s of prefill at 16K-32K, a much bigger block to attack than what runtime config can touch.
