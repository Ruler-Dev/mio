"""Projectors from intermediate hidden states to target KV cache entries.

Contract: given the target model's hidden state at layer N_early, produce
(K, V) tensors for each of the remaining layers N_early..N_layers-1.

In production this would be a trained per-layer MLP. For the scaffold we
ship IdentityKVProjector which passes hidden through a linear layer (or
straight-through when dims match) and tiles across heads. It's known to
produce garbage output quality; its purpose is to exercise shape contracts
and the integration plumbing so a real projector (trained externally) can
drop in later without touching the runtime.

Shape contract (per layer):
    Input:  hidden     (B, L, D_model)
    Output: K, V       each (B, n_kv_heads, L, head_dim)

where D_model = n_kv_heads * head_dim * 2 (K and V share the channel
split). Callers supply head_dim and n_kv_heads; the projector stacks K and
V along a doubled last dim and reshapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx


@dataclass(frozen=True)
class KVShape:
    """Per-layer KV output geometry."""

    n_kv_heads: int
    head_dim: int

    @property
    def channel_dim(self) -> int:
        """Required input channel dim to split into K and V."""
        return 2 * self.n_kv_heads * self.head_dim


class KVProjector(Protocol):
    """Projects intermediate hidden → per-layer KV."""

    def project(
        self,
        hidden: mx.array,
        *,
        layer_idx: int,
        kv_shape: KVShape,
    ) -> tuple[mx.array, mx.array]:
        """Return (K, V) for `layer_idx`.

        Args:
            hidden: (B, L, D_in) float array; D_in is projector's input size.
            layer_idx: target-model layer index this projection is for.
            kv_shape: output n_kv_heads / head_dim for this layer.

        Returns:
            K: (B, n_kv_heads, L, head_dim)
            V: (B, n_kv_heads, L, head_dim)
        """
        ...


class IdentityKVProjector:
    """Reshape-only projector; uses input hidden directly as K/V channels.

    Requires `hidden.shape[-1] == kv_shape.channel_dim`. Splits the last
    dim in half for K and V, then reshapes to per-head. This produces
    correctly-shaped output and exercises the surrounding pipeline; it
    does NOT produce useful KV values. Intended for plumbing tests and
    as a baseline to measure improvements against.
    """

    def project(
        self,
        hidden: mx.array,
        *,
        layer_idx: int,
        kv_shape: KVShape,
    ) -> tuple[mx.array, mx.array]:
        del layer_idx
        B, L, D = hidden.shape
        if D != kv_shape.channel_dim:
            raise ValueError(
                f"IdentityKVProjector needs hidden.shape[-1]={kv_shape.channel_dim}, "
                f"got {D}. Use LinearKVProjector to bridge mismatched dims."
            )
        k_flat, v_flat = mx.split(hidden, 2, axis=-1)
        # (B, L, n_kv_heads * head_dim) → (B, n_kv_heads, L, head_dim)
        k = k_flat.reshape(B, L, kv_shape.n_kv_heads, kv_shape.head_dim)
        v = v_flat.reshape(B, L, kv_shape.n_kv_heads, kv_shape.head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        return k, v


class LinearKVProjector:
    """Per-layer linear projection from hidden to (K, V) channels.

    Shape: `weights[layer_idx]` is (2 * n_kv_heads * head_dim, D_in).
    This is the scaffold form of what a trained projector would use.
    Initial weights are zero so output is known-bad but shape-correct
    (the zeroed K/V triggers uniform attention, which DFlash will reject
    immediately, so fallback kicks in fast if this is used in production).
    """

    def __init__(
        self,
        num_layers: int,
        d_in: int,
        kv_shape: KVShape,
    ) -> None:
        self.num_layers = int(num_layers)
        self.d_in = int(d_in)
        self.kv_shape = kv_shape
        d_out = kv_shape.channel_dim
        # Zero-init: shape-correct but produces uniform K/V. A trained
        # checkpoint overrides this via `.load_weights(...)`.
        self._weights = mx.zeros((num_layers, d_out, d_in), dtype=mx.float32)

    @property
    def weights(self) -> mx.array:
        return self._weights

    def load_weights(self, weights: mx.array) -> None:
        """Replace weights in-place; shape checked."""
        if weights.shape != (self.num_layers, self.kv_shape.channel_dim, self.d_in):
            raise ValueError(
                f"weights shape {weights.shape} != "
                f"({self.num_layers}, {self.kv_shape.channel_dim}, {self.d_in})"
            )
        self._weights = weights

    def project(
        self,
        hidden: mx.array,
        *,
        layer_idx: int,
        kv_shape: KVShape,
    ) -> tuple[mx.array, mx.array]:
        if kv_shape != self.kv_shape:
            raise ValueError(
                f"projector configured for {self.kv_shape}, got {kv_shape}"
            )
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        B, L, D = hidden.shape
        if D != self.d_in:
            raise ValueError(
                f"projector expects D_in={self.d_in}, got {D}"
            )
        w = self._weights[layer_idx]  # (D_out, D_in)
        # (B, L, D_in) @ (D_in, D_out) → (B, L, D_out)
        out = hidden @ w.T
        k_flat, v_flat = mx.split(out, 2, axis=-1)
        k = k_flat.reshape(B, L, kv_shape.n_kv_heads, kv_shape.head_dim)
        v = v_flat.reshape(B, L, kv_shape.n_kv_heads, kv_shape.head_dim)
        return k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
