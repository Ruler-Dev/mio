# Mio benchmark artifacts

`scripts/bench_qwen36_matrix.py` compares greedy autoregressive inference with
DFlash, PolarQuant, and TurboQuant on the same loaded target/draft pair. It
stores raw repetitions, environment metadata, output-token hashes, parity, and
median performance in `benchmarks/results/`.

Example:

```bash
python scripts/bench_qwen36_matrix.py \
  --tier large --prompt-tokens 512 --max-tokens 64 \
  --warmup 1 --reps 3
```

Unquantized DFlash is required to match the greedy baseline token-for-token.
Quantized KV modes report parity but are not rejected automatically because
lossy cache quantization can legitimately change later logits.
