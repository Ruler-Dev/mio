"""DDTree — diffusion draft-tree speculative decoding for hybrid models.

Ported from humanrouter/ddtree-mlx. Extends DFlash with tree-attention
verification: the draft block's per-position logits feed a heap-built tree
of candidate paths, verified through the target model in one forward pass
using tree attention masks + parent-indexed GatedDelta recurrence.

Re-targets `dflash_mlx.*` imports to `mio.dflash.*` and routes the
attention SDPA through `mlx_lm.models.base.scaled_dot_product_attention`
so it works with `QuantizedKVCache` (8-bit KV compression).
"""

from mio.ddtree.runtime import generate_ddtree_once

__all__ = ["generate_ddtree_once"]
