"""Experimental deferred DFlash context projection for lower TTFT.

Upstream already yields the target's first greedy token before the first
speculative cycle.  It nevertheless evaluates the draft model's full prompt
context projection before that yield.  This prototype keeps the exact same
projection graph but evaluates it on the first draft-context read, after the
first token has become observable.

Implementation note: this temporarily patches an upstream module global and is
therefore deliberately single-threaded and experimental.  It is not suitable
for production serving without an upstream session-level injection point.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass
class DeferredPrimingStats:
    projection_slices_deferred: int = 0
    flushes: int = 0
    flush_us: float = 0.0
    prompt_tokens: int = 0
    bypassed_long_prompt: bool = False
    fused_cold_prefill: bool = False


_PATCH_LOCK = threading.Lock()


def _deferred_store_class(base: type, stats: DeferredPrimingStats, max_prompt_tokens: int) -> type:
    import mlx.core as mx

    class DeferredTargetFeatureStore(base):
        _mio_deferred_priming = True

        def write_prompt_slice(
            self,
            *,
            start: int,
            end: int,
            features: Any,
        ) -> Any:
            stats.prompt_tokens = int(self.prompt_len)
            if int(self.prompt_len) > max_prompt_tokens:
                stats.bypassed_long_prompt = True
                return super().write_prompt_slice(start=start, end=end, features=features)

            projected = self.project_context(features) if self.project_context is not None else features
            if self._current_hidden is None:
                self._current_hidden = mx.zeros(
                    (projected.shape[0], int(self.prompt_len), projected.shape[-1]),
                    dtype=projected.dtype,
                )
            self._current_hidden[:, int(start):int(end), :] = projected
            self._mio_prefill_projection_pending = True
            stats.projection_slices_deferred += 1
            return self._current_hidden

        def require_current_hidden(self) -> Any:
            if self._current_hidden is None:
                raise RuntimeError("target hidden features are unavailable")
            if getattr(self, "_mio_prefill_projection_pending", False):
                started = time.perf_counter_ns()
                mx.eval(self._current_hidden)
                stats.flush_us += (time.perf_counter_ns() - started) / 1_000.0
                stats.flushes += 1
                self._mio_prefill_projection_pending = False
            return self._current_hidden

        def _project(self, features: Any) -> Any:
            if getattr(self, "_mio_prefill_projection_pending", False):
                return self.project_context(features) if self.project_context is not None else features
            return super()._project(features)

    DeferredTargetFeatureStore.__name__ = "DeferredTargetFeatureStore"
    return DeferredTargetFeatureStore


@contextmanager
def deferred_drafter_priming(
    *,
    stats: DeferredPrimingStats | None = None,
    max_prompt_tokens: int = 512,
    fuse_cold_prefill: bool = False,
) -> Iterator[DeferredPrimingStats]:
    """Install the short-prompt prototype for one non-concurrent generation.

    ``fuse_cold_prefill`` is valid only when no prefix snapshot/service is used.
    Upstream normally splits a cold prompt into ``prompt_len - 1`` plus a
    singleton seam even when snapshots are disabled.  Returning boundary zero
    routes that case through its existing full-tail implementation in one pass.
    """

    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if not _PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("deferred drafter priming cannot run concurrently")

    resolved = stats or DeferredPrimingStats()
    try:
        import dflash_mlx.engine.spec_epoch as spec_epoch

        original = spec_epoch.TargetFeatureStore
        original_boundary = spec_epoch.compute_snapshot_boundary
        if getattr(original, "_mio_deferred_priming", False):
            raise RuntimeError("deferred drafter priming patch is already installed")
        spec_epoch.TargetFeatureStore = _deferred_store_class(
            original,
            resolved,
            int(max_prompt_tokens),
        )
        if fuse_cold_prefill:
            def experimental_boundary(prompt_len: int, stable_prefix_len: int | None) -> int:
                if stable_prefix_len is None:
                    resolved.fused_cold_prefill = True
                    return 0
                return int(original_boundary(prompt_len, stable_prefix_len))

            spec_epoch.compute_snapshot_boundary = experimental_boundary
        try:
            yield resolved
        finally:
            spec_epoch.TargetFeatureStore = original
            spec_epoch.compute_snapshot_boundary = original_boundary
    finally:
        _PATCH_LOCK.release()


def stream_with_deferred_drafter_priming(
    stream_factory: Callable[[], Iterable[Any]],
    *,
    stats: DeferredPrimingStats | None = None,
    max_prompt_tokens: int = 512,
    fuse_cold_prefill: bool = False,
) -> Iterator[Any]:
    """Run an upstream stream factory under the deferred-priming prototype."""

    with deferred_drafter_priming(
        stats=stats,
        max_prompt_tokens=max_prompt_tokens,
        fuse_cold_prefill=fuse_cold_prefill,
    ):
        stream = stream_factory()
        try:
            yield from stream
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
