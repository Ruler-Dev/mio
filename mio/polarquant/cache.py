"""PolarQuantKVCache — Zero-overhead quantized KV cache for DFlash.

At 4-bit: uses MLX's native QuantizedKVCache directly (mx.quantized_matmul
Metal kernel — near FP16 speed parity). No rotation needed, 4-bit affine
already achieves ~0.998 cosine similarity.

At 3-bit or 2-bit: adds Hadamard rotation before quantization. This spreads
outliers uniformly, improving quality at low bit-widths (0.95+ cosine at 3-bit
vs ~0.82 without rotation). Requires SDPA patch for Q rotation.

Design: zero overhead at 4-bit, quality-preserving at sub-4-bit.
"""

import mlx.core as mx
from mlx_lm.models.cache import QuantizedKVCache

from mio.polarquant.hadamard import hadamard_matrix


class PolarQuantKVCache(QuantizedKVCache):
    """Quantized KV cache with optional Hadamard rotation.

    At 4-bit: identical to QuantizedKVCache (no rotation, zero overhead).
    At 3/2-bit: adds Hadamard rotation for quality (needs SDPA patch).
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 4,
        group_size: int = 64,
    ):
        super().__init__(group_size=group_size, bits=bits)
        self.head_dim = head_dim
        self._use_rotation = bits < 4  # only rotate at sub-4-bit
        if self._use_rotation:
            self.H = hadamard_matrix(head_dim)
            mx.eval(self.H)
        else:
            self.H = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Optionally rotate, then delegate to QuantizedKVCache."""
        if self._use_rotation:
            keys = keys @ self.H
            values = values @ self.H
        return super().update_and_fetch(keys, values)
