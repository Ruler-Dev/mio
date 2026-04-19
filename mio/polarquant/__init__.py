"""PolarQuant — lightweight KV-cache quantization for DFlash.

Hadamard rotation + MLX-native affine quantization. Designed for zero
speed overhead with DFlash speculative decoding:
  - No forced mx.eval() — preserves MLX lazy evaluation pipeline
  - No L2 normalization — one fewer op per token
  - No QJL residual correction — simpler, faster
  - 4-bit default: ~4x KV compression, cosine similarity >0.99
"""
