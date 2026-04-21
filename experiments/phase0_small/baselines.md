# Baseline prefill profile — small

- git: `bfa2738e588f`
- hardware: `Mac16,5`
- target: `/Users/ruler/Documents/pi-mio/models/Qwen3.5-4B-4bit`
- draft: `/Users/ruler/Documents/pi-mio/spd/Qwen3.5-4B-DFlash`
- pq_bits=4, tq_bits=16
- timestamp: 1776775339

## Warm prefill (rep >= 1) by context

| ctx | tokens | total ms | linear ms | attn ms | linear % | attn % |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 673 | 444.5 | 334.7 | 108.4 | 75.3% | 24.4% |
| 1024 | 1206 | 762.3 | 571.6 | 189.1 | 75.0% | 24.8% |
| 2048 | 2212 | 1646.7 | 1065.6 | 371.6 | 64.7% | 22.6% |
| 4096 | 4097 | 3107.4 | 1954.8 | 730.4 | 62.9% | 23.5% |
| 8192 | 8286 | 6706.5 | 4077.8 | 1749.3 | 60.8% | 26.1% |

## Cold vs warm (first rep vs. best subsequent)

| ctx | cold ms | warm ms |
|---|---:|---:|
| 512 | 604 | 445 |
| 1024 | 771 | 762 |
| 2048 | 1668 | 1647 |
| 4096 | 3127 | 3107 |
| 8192 | 6614 | 6706 |
