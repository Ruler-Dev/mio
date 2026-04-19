"""Walsh-Hadamard matrix generation for PolarQuant rotation.

The Hadamard matrix H_n (n = 2^k) is orthogonal: H @ H^T = n * I.
Normalized: (H / sqrt(n)) is a unitary rotation that is self-inverse,
meaning the same matrix is used for both forward and inverse transforms.

This spreads outlier values across all dimensions, making the distribution
more uniform and dramatically improving low-bit quantization quality.
"""

import mlx.core as mx


def hadamard_matrix(dim: int, *, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Generate a normalized Hadamard matrix of size dim x dim.

    Args:
        dim: Must be a power of 2.
        dtype: Output dtype.

    Returns:
        Orthonormal Hadamard matrix H where H @ H^T = I.
    """
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"dim must be a power of 2, got {dim}")

    # Build via Sylvester construction: H_1 = [1], H_{2n} = [[H_n, H_n], [H_n, -H_n]]
    h = mx.array([[1.0]], dtype=dtype)
    while h.shape[0] < dim:
        h = mx.concatenate(
            [
                mx.concatenate([h, h], axis=1),
                mx.concatenate([h, -h], axis=1),
            ],
            axis=0,
        )

    # Normalize to make it orthonormal: H @ H^T = I
    return h * (dim ** -0.5)
