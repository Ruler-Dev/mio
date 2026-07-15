"""Compatibility between DFlash hooks and MLX-LM continuous batches."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mio.dflash.runtime import _scalar_cache_offset


def test_scalar_cache_offset_accepts_single_stream_values():
    assert _scalar_cache_offset(None) == 0
    assert _scalar_cache_offset(SimpleNamespace(offset=7)) == 7
    assert _scalar_cache_offset(SimpleNamespace(offset=mx.array(9))) == 9


def test_scalar_cache_offset_detects_per_sequence_batch_offsets():
    assert _scalar_cache_offset(SimpleNamespace(offset=mx.array([3, 5]))) is None
