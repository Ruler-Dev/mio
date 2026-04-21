# Baseline prefill profile — large-moe

- git: `4e44f9c88ae6`
- hardware: `Mac16,5`
- target: `/Users/ruler/Documents/pi-mio/models/Qwen3.6-35B-A3B-UD-Q4_K_XL-mlx`
- draft: `/Users/ruler/Documents/pi-mio/spd/Qwen3.6-35B-A3B-DFlash`
- pq_bits=4, tq_bits=16
- timestamp: 1776774871

## Warm prefill (rep >= 1) by context

| ctx | tokens | total ms | linear ms | attn ms | linear % | attn % |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 673 | 672.9 | 528.8 | 141.9 | 78.6% | 21.1% |
| 1024 | 1206 | 698.6 | 526.3 | 170.0 | 75.3% | 24.3% |
| 2048 | 2212 | 1487.4 | 974.7 | 339.2 | 65.5% | 22.8% |
| 4096 | 4097 | 2804.2 | 1784.8 | 662.5 | 63.6% | 23.6% |
| 8192 | 8286 | 6191.0 | 3815.3 | 1648.4 | 61.6% | 26.6% |
| 16384 | 16561 | 14422.7 | 8003.0 | 4773.3 | 55.5% | 33.1% |
| 32768 | 32780 | 52115.3 | 22526.5 | 22717.0 | 43.2% | 43.6% |

## Cold vs warm (first rep vs. best subsequent)

| ctx | cold ms | warm ms |
|---|---:|---:|
| 512 | 3616 | 673 |
| 1024 | 709 | 699 |
| 2048 | 1514 | 1487 |
| 4096 | 2849 | 2804 |
| 8192 | 6114 | 6191 |
| 16384 | 13990 | 14423 |
| 32768 | 47069 | 52115 |
