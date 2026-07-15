import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .kernels.rotation import get_rotation_kernel


def _pack_pairs(pairs: mx.array, group_size: int) -> mx.array:
    """Pack int16 pair indices into int32 for the Metal kernel."""
    krot, hidden = int(pairs.shape[0]), int(pairs.shape[1])
    p = np.array(pairs, copy=False).reshape(krot, hidden // group_size, group_size).astype(np.int32, copy=False)
    return mx.array((p[:, :, 0::2] | (p[:, :, 1::2] << 16)).reshape(krot, -1))


def _precompute_scatter_orders(
    pairs: mx.array, group_size: int,
) -> list[tuple[mx.array, mx.array, mx.array]]:
    """Pre-compute gather/scatter indices for pure-MLX rotation.

    Pairs are group-local indices; this converts them to absolute indices
    and pre-computes the argsort for scattering results back.

    Returns a list of (gather_a, gather_b, scatter_order) per rotation round.
    """
    krot = int(pairs.shape[0])
    dim = int(pairs.shape[1])
    num_groups = dim // group_size
    p_np = np.array(pairs, copy=False).astype(np.int32)
    offsets = np.arange(num_groups).reshape(-1, 1) * group_size  # (num_groups, 1)

    tables = []
    for k in range(krot):
        # Reshape to (num_groups, group_size) — pairs are group-local
        pk = p_np[k].reshape(num_groups, group_size)
        local_a = pk[:, 0::2]  # (num_groups, half_gs) — first of each pair
        local_b = pk[:, 1::2]  # (num_groups, half_gs) — second of each pair
        # Convert to absolute indices
        abs_a = (local_a + offsets).reshape(-1)  # (dim//2,)
        abs_b = (local_b + offsets).reshape(-1)  # (dim//2,)
        combined = np.concatenate([abs_a, abs_b])
        order = np.argsort(combined).astype(np.int32)
        tables.append((
            mx.array(abs_a, dtype=mx.int32),
            mx.array(abs_b, dtype=mx.int32),
            mx.array(order, dtype=mx.int32),
        ))
    return tables


def _apply_rotation_mlx(
    x: mx.array,
    scatter_tables: list[tuple[mx.array, mx.array, mx.array]],
    cos: mx.array,
    sin: mx.array,
    scales_flat: mx.array,
) -> mx.array:
    """Pure MLX pairwise Givens rotation (no custom Metal kernel).

    Fully compatible with lazy evaluation — no graph breaks.
    """
    x = x * scales_flat
    for k, (ga, gb, scatter_order) in enumerate(scatter_tables):
        xa = x[..., ga]
        xb = x[..., gb]
        cos_k = cos[k]
        sin_k = sin[k]
        new_a = cos_k * xa + sin_k * xb
        new_b = cos_k * xb - sin_k * xa
        x = mx.concatenate([new_a, new_b], axis=-1)[..., scatter_order]
    return x


def _apply_rotation(
    x: mx.array,
    packed_pairs: mx.array,
    cos: mx.array,
    sin: mx.array,
    scales_flat: mx.array,
    dim: int,
    krot: int,
    group_size: int,
) -> mx.array:
    """Dispatch the Metal pairwise-rotation kernel on a 2-D (batch, dim) tensor."""
    batch = x.shape[0]
    if batch == 0:
        return x
    tile = 1 if batch <= 1 else 4
    half_group = group_size // 2
    num_groups = dim // group_size
    params = mx.array([batch, dim, krot, group_size], dtype=mx.int32)
    grid = (math.ceil(batch / tile) * half_group, num_groups, 1)
    return get_rotation_kernel(tile)(
        inputs=[x, packed_pairs, cos, sin, scales_flat, params],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=grid,
        threadgroup=(half_group, 1, 1),
    )[0]


class RotateQuantizedLinear(nn.Module):
    """Pairwise Givens rotation + quantized matmul (Metal kernel)."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 128,
        bits: int = 4,
        krot: int = 8,
    ):
        super().__init__()
        self.group_size = group_size
        self.bits = bits

        self.theta = mx.zeros((krot, input_dims // 2))
        self.pairs = mx.zeros((krot, input_dims), dtype=mx.int16)
        self.channel_scales = mx.ones((1, input_dims))

        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // group_size))
        self.biases = mx.zeros((output_dims, input_dims // group_size))

        if bias:
            self.bias = mx.zeros((output_dims,))

        self._cached = False

    def _cache_rotation(self):
        """Pre-compute sin/cos and pack pairs (called once on first forward)."""
        dim = self.theta.shape[1] * 2
        self._dim = dim
        self._krot = int(self.theta.shape[0])
        # Compute cos/sin via numpy to produce eagerly-evaluated MLX arrays.
        # The custom Metal kernel produces NaN when these are lazy MLX ops
        # in the computation graph (buffer aliasing during lazy eval).
        theta_np = np.array(self.theta)
        self._cos = mx.array(np.cos(theta_np))
        self._sin = mx.array(np.sin(theta_np))
        self._packed_pairs = _pack_pairs(self.pairs, self.group_size)
        self._scales_flat = mx.array(np.array(self.channel_scales).reshape(-1))
        self._cached = True

    def __call__(self, x: mx.array) -> mx.array:
        if not self._cached:
            self._cache_rotation()

        shape = x.shape
        rotated = _apply_rotation(
            x.reshape(-1, self._dim),
            self._packed_pairs,
            self._cos,
            self._sin,
            self._scales_flat,
            self._dim,
            self._krot,
            self.group_size,
        )
        mx.eval(rotated)

        y = mx.quantized_matmul(
            rotated.reshape(shape),
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class _CachedRotation:
    """Mixin-style helper that pre-computes sin/cos and packs pairs for a single rotation."""

    def _init_rotation(self, krot: int, dim: int, group_size: int, prefix: str = ""):
        pfx = f"{prefix}_" if prefix else ""
        setattr(self, f"{pfx}theta", mx.zeros((krot, dim // 2)))
        setattr(self, f"{pfx}pairs", mx.zeros((krot, dim), dtype=mx.int16))
        setattr(self, f"{pfx}channel_scales", mx.ones((1, dim)))
        self._rot_group_size = group_size

    def _cache_single_rotation(self, prefix: str = ""):
        pfx = f"{prefix}_" if prefix else ""
        theta = getattr(self, f"{pfx}theta")
        pairs = getattr(self, f"{pfx}pairs")
        dim = int(theta.shape[1]) * 2
        krot = int(theta.shape[0])
        theta_np = np.array(theta)
        cos = mx.array(np.cos(theta_np))
        sin = mx.array(np.sin(theta_np))
        packed_pairs = _pack_pairs(pairs, self._rot_group_size)
        scales_flat = mx.array(np.array(getattr(self, f"{pfx}channel_scales")).reshape(-1))
        tag = f"_{prefix}" if prefix else ""
        setattr(self, f"_rot{tag}_dim", dim)
        setattr(self, f"_rot{tag}_krot", krot)
        setattr(self, f"_rot{tag}_cos", cos)
        setattr(self, f"_rot{tag}_sin", sin)
        setattr(self, f"_rot{tag}_packed_pairs", packed_pairs)
        setattr(self, f"_rot{tag}_scales_flat", scales_flat)

    def _rotate(self, x: mx.array, prefix: str = "") -> mx.array:
        tag = f"_{prefix}" if prefix else ""
        dim = getattr(self, f"_rot{tag}_dim")
        shape = x.shape
        rotated = _apply_rotation(
            x.reshape(-1, dim),
            getattr(self, f"_rot{tag}_packed_pairs"),
            getattr(self, f"_rot{tag}_cos"),
            getattr(self, f"_rot{tag}_sin"),
            getattr(self, f"_rot{tag}_scales_flat"),
            dim,
            getattr(self, f"_rot{tag}_krot"),
            self._rot_group_size,
        )
        mx.eval(rotated)
        return rotated.reshape(shape)


class RotateSwitchGLU(nn.Module, _CachedRotation):
    """SwitchGLU with shared pairwise rotation injected before each sub-layer.

    All experts share a single set of rotation parameters per projection:
    ``gate_up_rot`` is applied to x before gate_proj/up_proj, and
    ``down_rot`` is applied to the activation output before down_proj.
    """

    def __init__(self, glu: nn.Module, group_size: int, krot: int):
        super().__init__()
        self.gate_proj = glu.gate_proj
        self.up_proj = glu.up_proj
        self.down_proj = glu.down_proj
        self.activation = glu.activation

        gate_up_dim = glu.gate_proj.input_dims
        down_dim = glu.down_proj.input_dims
        self._init_rotation(krot, gate_up_dim, group_size, prefix="gate_up_rot")
        self._init_rotation(krot, down_dim, group_size, prefix="down_rot")
        self._cached = False

    def _cache_rotation(self):
        self._cache_single_rotation("gate_up_rot")
        self._cache_single_rotation("down_rot")
        self._cached = True

    def __call__(self, x, indices) -> mx.array:
        if not self._cached:
            self._cache_rotation()

        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        x = self._rotate(x, "gate_up_rot")

        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)

        act = self.activation(x_up, x_gate)
        act = self._rotate(act, "down_rot")

        x = self.down_proj(act, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
