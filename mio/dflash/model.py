"""Compatibility facade for the upstream :mod:`dflash_mlx` model.

Pi-Mio used to carry a fork of the DFlash draft model in this module. The
upstream implementation now supports mixed full/sliding-attention drafts,
position-aware sink/window caches, and causal sliding-window masks. Keeping
the public imports here lets the rest of Pi-Mio use that implementation
without duplicating a rapidly evolving model definition.
"""

from __future__ import annotations

import mlx.core as mx
from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashDraftModel,
    DFlashDraftModelArgs,
    FullContextDraftKVCache,
    build_target_layer_ids,
)

__all__ = [
    "ContextOnlyDraftKVCache",
    "DFlashDraftModel",
    "DFlashDraftModelArgs",
    "FullContextDraftKVCache",
    "build_target_layer_ids",
    "extract_context_feature",
]


def extract_context_feature(
    hidden_states: list[mx.array],
    layer_ids: list[int],
) -> mx.array:
    """Concatenate the target features consumed by the DFlash projection."""

    selected = [hidden_states[layer_id + 1] for layer_id in layer_ids]
    return mx.concatenate(selected, axis=-1)
