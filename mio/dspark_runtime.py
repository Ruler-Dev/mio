"""Thread-confined adapter for the optional :mod:`mlx_dspark` runtime.

``mlx_dspark`` wraps an already loaded MLX target and keeps mutable speculative
state around each request.  Mio confines construction and generation to one
worker so a streaming callback never moves that state between threads.  The
adapter intentionally exposes ordinary dictionaries matching Mio's DFlash
result schema; the engine can therefore report one honest telemetry contract
without teaching the HTTP layer about a third-party result class.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class DSparkRuntime:
    """Loaded DSpark drafter plus its target wrapper.

    Use :meth:`load` rather than instantiating this class directly.  A single
    worker is deliberate: MLX execution is asynchronous and both the target
    cache and the DSpark context are request-mutable.
    """

    _executor: ThreadPoolExecutor = field(repr=False)
    _target: Any = field(repr=False)
    _tokenizer: Any = field(repr=False)
    _drafter: Any = field(repr=False)
    draft_ref: str
    max_draft_tokens: int = 2
    lookup_drafts: bool = True
    _prefix_cache: Any = field(default=None, repr=False)
    _prefix_cache_reason: str = field(default="disabled_by_config", repr=False)
    _closed: bool = field(default=False, repr=False)

    @classmethod
    def load(
        cls,
        *,
        target_model: Any,
        tokenizer: Any,
        draft_ref: str,
        bits: int = 4,
        group_size: int = 64,
        max_draft_tokens: int = 2,
        lookup_drafts: bool = True,
        prefix_cache: bool = True,
        prefix_cache_slots: int = 2,
        prefix_cache_min_reuse: int = 64,
    ) -> "DSparkRuntime":
        """Strictly load and verify a DSpark checkpoint on its worker thread."""

        max_draft_tokens = int(max_draft_tokens)
        if not 1 <= max_draft_tokens <= 3:
            raise ValueError(
                "dspark_max_draft_tokens must be between 1 and 3; cap >=4 failed Mio's strict Qwen3.6-27B parity gate"
            )

        # ``load_target_bundle(..., lazy=True)`` leaves disk-backed CPU graphs
        # attached to the loading thread's stream.  Passing those unevaluated
        # graphs to the dedicated DSpark worker fails with ``Stream(cpu, 0)``.
        # Materialize and synchronize on the owner thread first; subsequent
        # target operations and every DSpark cache are then created only on the
        # worker.  This is an explicit ownership boundary, not a global-stream
        # override (which would merely hide the invalid cross-thread graph).
        import mlx.core as mx

        mx.eval(target_model.parameters())
        mx.synchronize()

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mio-dspark")

        def _load() -> tuple[Any, Any, Any, str]:
            from mlx_dspark.load import apply_wired_limit, load_drafter
            from mlx_dspark.target import Target

            apply_wired_limit()
            drafter, _config = load_drafter(
                str(draft_ref),
                quantize=True,
                bits=int(bits),
                group_size=int(group_size),
                strict=True,
            )
            target = Target(target_model, tokenizer)
            target.verify_tap()
            upstream_prefix = None
            prefix_reason = "disabled_by_config"
            if prefix_cache:
                try:
                    from mlx_dspark.prefix_cache import PrefixCache, target_cache_reusable

                    if target_cache_reusable(target.make_cache()):
                        upstream_prefix = PrefixCache(
                            target.make_cache,
                            drafter.make_ctx_cache,
                            min_reuse=max(1, int(prefix_cache_min_reuse)),
                            slots=max(1, int(prefix_cache_slots)),
                        )
                        prefix_reason = "enabled_upstream"
                    else:
                        prefix_reason = "unsupported_target_cache"
                except Exception as error:
                    prefix_reason = f"initialization_failed:{type(error).__name__}"
            return target, drafter, upstream_prefix, prefix_reason

        try:
            target, drafter, upstream_prefix, prefix_reason = executor.submit(_load).result()
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        return cls(
            _executor=executor,
            _target=target,
            _tokenizer=tokenizer,
            _drafter=drafter,
            draft_ref=str(draft_ref),
            max_draft_tokens=max_draft_tokens,
            lookup_drafts=bool(lookup_drafts),
            _prefix_cache=upstream_prefix,
            _prefix_cache_reason=prefix_reason,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DSpark runtime is closed")

    def _disable_prefix_cache(self, reason: str, *, reset: bool = True) -> None:
        """Best-effort invalidate and permanently downgrade prefix reuse.

        This helper is called only on the runtime's worker.  Prefix caching is
        an optimization boundary: an upstream cache failure must never discard
        an otherwise valid generation result or poison the next request.
        """

        prefix_cache = self._prefix_cache
        if reset and prefix_cache is not None:
            try:
                prefix_cache.reset()
            except Exception:
                pass
        self._prefix_cache = None
        self._prefix_cache_reason = reason

    def _run(
        self,
        *,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
        stop: list[str] | None,
        on_text: Callable[[str], None] | None,
        cancel: threading.Event | None,
    ) -> dict[str, Any]:
        from mlx_dspark import StopStreaming, speculative_generate

        started = time.perf_counter()
        first_text_at: float | None = None
        cache = None
        ctx_caches = None
        reuse_len = 0
        if self._prefix_cache is not None:
            try:
                cache, ctx_caches, reuse_len = self._prefix_cache.acquire(prompt_ids)
            except Exception as error:
                self._disable_prefix_cache(
                    f"acquire_failed:{type(error).__name__}",
                )

        def _on_text(chunk: str) -> None:
            nonlocal first_text_at
            if cancel is not None and cancel.is_set():
                raise StopStreaming()
            if chunk and first_text_at is None:
                first_text_at = time.perf_counter()
            if on_text is not None and chunk:
                on_text(chunk)

        try:
            result = speculative_generate(
                self._target,
                self._tokenizer,
                self._drafter,
                prompt_ids=list(prompt_ids),
                cache=cache,
                ctx_caches=ctx_caches,
                reuse_len=reuse_len,
                max_new_tokens=max(1, int(max_new_tokens)),
                max_draft_tokens=self.max_draft_tokens,
                lookup_drafts=self.lookup_drafts,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                seed=seed,
                stop=stop,
                on_text=_on_text,
                apply_chat_template=False,
                verbose=False,
            )
        except BaseException:
            # ``acquire`` checks a cache out of the LRU. A failed generation
            # may have mutated it without a valid token record, so mirror the
            # upstream Engine contract and invalidate every slot.
            if self._prefix_cache is not None:
                try:
                    self._prefix_cache.reset()
                except Exception as reset_error:
                    self._disable_prefix_cache(
                        f"reset_failed:{type(reset_error).__name__}",
                        reset=False,
                    )
            raise
        if self._prefix_cache is not None:
            try:
                self._prefix_cache.store(
                    cache,
                    ctx_caches,
                    list(prompt_ids),
                    list(result.token_ids),
                )
            except Exception as error:
                self._disable_prefix_cache(
                    f"store_failed:{type(error).__name__}",
                )
        finished = time.perf_counter()
        elapsed_us = max(0.0, (finished - started) * 1e6)
        prefill_us = max(0.0, (first_text_at - started) * 1e6) if first_text_at is not None else 0.0
        token_ids = [int(token_id) for token_id in result.token_ids]
        accept_lengths = [int(length) for length in result.accept_lengths]
        prefix_info = {"slots": [], "hits": 0, "reused_tokens": 0}
        if self._prefix_cache is not None:
            try:
                prefix_info = self._prefix_cache.info()
            except Exception as error:
                self._disable_prefix_cache(
                    f"info_failed:{type(error).__name__}",
                )
        return {
            "backend": "dspark",
            "text": str(result.text),
            "generated_token_ids": token_ids,
            "generation_tokens": int(result.num_tokens),
            "prompt_token_count": len(prompt_ids),
            "elapsed_us": elapsed_us,
            "prefill_us": min(prefill_us, elapsed_us),
            "phase_timings_us": {
                "prefill": min(prefill_us, elapsed_us),
                "decode": max(0.0, elapsed_us - prefill_us),
            },
            # DSpark exposes committed tokens per round, not the number of
            # proposals at every lookup/drafter round. Report its exact mean
            # commit length and mark the proposal acceptance ratio unavailable
            # rather than manufacturing a denominator.
            "tokens_per_cycle": float(result.mean_accept_len),
            "acceptance_ratio": None,
            "acceptance_ratio_available": False,
            "cycles_completed": int(result.num_rounds),
            "accept_lengths": accept_lengths,
            "target_forwards": int(result.target_forwards),
            "lookup_rounds": int(result.lookup_rounds),
            "dspark_max_draft_tokens": self.max_draft_tokens,
            "dspark_lookup_drafts": self.lookup_drafts,
            "warm_offset": int(reuse_len),
            "cache_entries": len(prefix_info.get("slots", [])),
            "prefix_cache_hits": int(prefix_info.get("hits", 0)),
            "prefix_cache_reused_tokens": int(prefix_info.get("reused_tokens", 0)),
            "finish_reason": str(result.finish_reason),
            "stopped_early": bool(cancel is not None and cancel.is_set()),
        }

    def generate(
        self,
        *,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run one lossless DSpark request and return Mio-style metrics data."""

        self._ensure_open()
        return self._executor.submit(
            self._run,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
            on_text=None,
            cancel=None,
        ).result()

    def stream(
        self,
        *,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        stop: list[str] | None = None,
        cancel: threading.Event | None = None,
    ) -> Generator[tuple[str, dict[str, Any] | None], None, None]:
        """Yield UTF-8-safe text chunks followed by one terminal result dict."""

        self._ensure_open()
        stop_event = cancel or threading.Event()
        # Bound queued SSE text so a slow/disconnected client cannot turn
        # generation into unbounded host-memory growth. The callback applies
        # backpressure in short intervals so cancellation remains responsive.
        chunks: queue.Queue[str] = queue.Queue(maxsize=128)

        def _enqueue(chunk: str) -> None:
            from mlx_dspark import StopStreaming

            while not stop_event.is_set():
                try:
                    chunks.put(chunk, timeout=0.05)
                    return
                except queue.Full:
                    continue
            raise StopStreaming()

        future = self._executor.submit(
            self._run,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
            on_text=_enqueue,
            cancel=stop_event,
        )
        future_consumed = False
        try:
            while not future.done() or not chunks.empty():
                try:
                    chunk = chunks.get(timeout=0.05)
                except queue.Empty:
                    continue
                yield chunk, None
            result = future.result()
            future_consumed = True
            yield "", result
        finally:
            # ``StopStreaming`` is handled by mlx-dspark at the next round
            # boundary, leaving its caches internally consistent.
            stop_event.set()
            if not future_consumed:
                # Consumer disconnects must not leave target work running after
                # the request/gpu lock is released. Wait for the next DSpark
                # round boundary and consume any terminal exception.
                try:
                    future.result()
                except BaseException:
                    pass

    def close(self) -> None:
        """Wait for any active request and release the worker and drafter."""

        if self._closed:
            return
        self._closed = True
        try:
            if self._prefix_cache is not None:
                prefix_cache = self._prefix_cache

                def _reset_for_close() -> None:
                    try:
                        prefix_cache.reset()
                    except Exception as error:
                        self._prefix_cache_reason = f"close_reset_failed:{type(error).__name__}"

                self._executor.submit(_reset_for_close).result()
        finally:
            # Cache cleanup is optional; executor shutdown is the ownership
            # boundary that must always run, even if upstream reset is broken.
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._target = None
            self._drafter = None
            self._prefix_cache = None

    @property
    def prefix_cache_status(self) -> dict[str, Any]:
        """Report whether mlx-dspark's exact prefix cache is active."""

        if self._closed:
            return {"enabled": False, "reason": "runtime_closed"}
        if self._prefix_cache is None:
            return {"enabled": False, "reason": self._prefix_cache_reason}

        def _status() -> dict[str, Any]:
            if self._prefix_cache is None:
                return {"enabled": False, "reason": self._prefix_cache_reason}
            try:
                info = self._prefix_cache.info()
            except Exception as error:
                self._disable_prefix_cache(
                    f"info_failed:{type(error).__name__}",
                )
                return {"enabled": False, "reason": self._prefix_cache_reason}
            return {**info, "enabled": True, "reason": self._prefix_cache_reason}

        return self._executor.submit(_status).result()
