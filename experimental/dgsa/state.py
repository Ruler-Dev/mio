"""Thread-safe state container for DGSA prefill.

Patched attention layers consult this state to decide whether to slice K/V.
Outside DGSA-mode (decode, normal forward), the state is inactive and the
layers behave like stock attention.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class _DGSAState:
    active: bool = False
    keep_indices: Optional[mx.array] = None  # (K,) int32 — positions of important keys
    cache_offset_override: Optional[int] = None  # used when populating cache after sparse prefill


_TLS = threading.local()


def get_state() -> _DGSAState:
    s = getattr(_TLS, "state", None)
    if s is None:
        s = _DGSAState()
        _TLS.state = s
    return s


@contextmanager
def dgsa_active(keep_indices: mx.array, cache_offset: Optional[int] = None):
    """Context manager: activate DGSA for the duration. Patched attention layers
    read this and slice their K/V accordingly.
    """
    s = get_state()
    prev_active = s.active
    prev_idx = s.keep_indices
    prev_off = s.cache_offset_override
    s.active = True
    s.keep_indices = keep_indices
    s.cache_offset_override = cache_offset
    try:
        yield
    finally:
        s.active = prev_active
        s.keep_indices = prev_idx
        s.cache_offset_override = prev_off
