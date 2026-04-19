"""PolarQuant SDPA — attention with Hadamard-rotated quantized KV cache.

Uses mx.quantized_matmul for both Q*K scoring and weighted-sum of V,
maximizing Metal hardware affinity. The Hadamard rotation is applied to
queries and inverse-rotated from the output.

Since the normalized Hadamard matrix is self-inverse (H = H^T = H^{-1}),
the same matrix is used for all transforms.
"""

import mlx.core as mx
from mlx.utils import tree_map


def polarquant_sdpa(
    queries: mx.array,
    q_keys: tuple,
    q_values: tuple,
    cache,
    scale: float,
    mask=None,
) -> mx.array:
    """Scaled dot-product attention with PolarQuant quantized KV cache.

    Args:
        queries: (B, n_q_heads, T_q, D) query vectors
        q_keys: (data, scales, biases) quantized keys from cache
        q_values: (data, scales, biases) quantized values from cache
        cache: PolarQuantKVCache instance
        scale: attention scale factor (1/sqrt(D))
        mask: None, "causal", or mx.array

    Returns:
        (B, n_q_heads, T_q, D) attention output
    """
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = q_keys[0].shape[1]
    n_repeats = n_q_heads // n_kv_heads

    # Rotate queries into Hadamard space and apply scale
    q_rot = (queries * scale) @ cache.H

    # GQA: reshape for grouped query attention
    if n_repeats > 1:
        q_rot = q_rot.reshape(B, n_kv_heads, n_repeats, T_q, D)
    else:
        q_rot = q_rot[:, :, None, :, :]
    q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
    q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    # Scores via hardware-accelerated quantized matmul
    scores = mx.quantized_matmul(
        q_rot, *q_keys,
        transpose=True, group_size=cache.group_size, bits=cache.bits,
    )

    # Apply mask
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    # Softmax + value output via quantized matmul
    weights = mx.softmax(scores, axis=-1, precise=True)
    output = mx.quantized_matmul(
        weights, *q_values,
        transpose=False, group_size=cache.group_size, bits=cache.bits,
    )

    # Inverse Hadamard rotation (H is self-inverse)
    output = output @ cache.H

    return output.reshape(B, n_q_heads, T_q, D)
