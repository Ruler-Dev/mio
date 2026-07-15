"""Event and invocation adapter for upstream ``dflash_mlx``.

The adapter intentionally accepts an already-loaded upstream ``RuntimeBundle``.
Mio's vendored target/draft objects are not assumed to be ABI-compatible with
the upstream package.  This keeps the prototype honest and prevents accidental
production coupling while the fast path is being certified.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


@dataclass(frozen=True)
class UpstreamGenerationRequest:
    """Subset of Mio's request contract that upstream can represent exactly."""

    max_new_tokens: int
    prompt: str = ""
    use_chat_template: bool = False
    block_tokens: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    suppress_token_ids: tuple[int, ...] = ()
    prompt_tokens_override: tuple[int, ...] | None = None
    quantize_kv_cache: bool = False
    target_fa_window: int = 0
    prefill_step_size: int | None = None
    draft_sink_size: int | None = None
    draft_window_size: int | None = None
    verify_len_cap: int | None = None
    verify_mode: str = "dflash"
    prefix_snapshot: Any = None
    snapshot_service: Any = None
    stable_prefix_len: int | None = None
    prefix_cache_active: bool = False
    publish_generation_snapshot: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.block_tokens is not None and self.block_tokens <= 0:
            raise ValueError("block_tokens must be positive when present")
        if self.target_fa_window < 0:
            raise ValueError("target_fa_window must be non-negative")
        if self.verify_mode not in {"dflash", "adaptive", "ddtree", "off"}:
            raise ValueError("verify_mode must be dflash, adaptive, ddtree, or off")
        if self.prefix_cache_active and self.snapshot_service is None:
            raise ValueError("snapshot_service is required when prefix_cache_active is true")


_EVENT_KINDS = {
    "PrefillProgressEvent": "prefill_progress",
    "PrefillCompleteEvent": "prefill",
    "TokenEvent": "token",
    "SnapshotPublishedEvent": "snapshot",
    "CycleCompleteEvent": "cycle",
    "MemoryWaterfallEvent": "memory",
    "SummaryEvent": "summary",
}


def _payload(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    to_payload = getattr(event, "to_payload", None)
    if callable(to_payload):
        payload = to_payload()
        if not isinstance(payload, Mapping):
            raise TypeError("upstream event to_payload() must return a mapping")
        return dict(payload)
    if is_dataclass(event):
        return asdict(event)
    raise TypeError(f"unsupported upstream event payload: {type(event).__name__}")


class MioEventAdapter:
    """Stateful translation from upstream dataclasses to Mio event dictionaries."""

    def __init__(self, *, strict: bool = True, include_diagnostics: bool = True) -> None:
        self.strict = bool(strict)
        self.include_diagnostics = bool(include_diagnostics)
        self.prefill_us = 0.0
        self.warm_offset = 0

    def adapt(self, event: Any) -> dict[str, Any] | None:
        if isinstance(event, Mapping) and "event" in event:
            return dict(event)

        class_name = type(event).__name__
        kind = _EVENT_KINDS.get(class_name)
        if kind is None:
            if self.strict:
                raise TypeError(f"unknown upstream event type: {class_name}")
            return None

        payload = _payload(event)
        payload["event"] = kind

        if kind == "prefill":
            self.prefill_us = float(payload.get("prefill_us", 0.0) or 0.0)
            self.warm_offset = int(payload.get("prefill_tokens_restored", 0) or 0)
            payload.setdefault("warm_offset", self.warm_offset)
        elif kind == "summary":
            payload.setdefault("prefill_us", self.prefill_us)
            payload.setdefault("warm_offset", self.warm_offset)
            payload.setdefault("cache_commit_mode", "upstream_target_ops_rollback")
            if self.include_diagnostics:
                payload.setdefault("engine", "dflash_mlx.stream_dflash_generate")
                payload.setdefault("metrics_scope", "request")
        elif kind == "memory" and set(payload) == {"fields", "event"}:
            fields = payload.pop("fields")
            if isinstance(fields, Mapping):
                payload.update(fields)

        return payload


def adapt_upstream_stream(
    stream: Iterable[Any],
    *,
    adapter: MioEventAdapter | None = None,
) -> Iterator[dict[str, Any]]:
    """Translate and close an upstream event stream, including on cancellation."""

    resolved = adapter or MioEventAdapter()
    try:
        for event in stream:
            adapted = resolved.adapt(event)
            if adapted is not None:
                yield adapted
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def stream_bundle_as_mio(
    bundle: Any,
    request: UpstreamGenerationRequest,
    *,
    runtime_context: Any = None,
    adapter: MioEventAdapter | None = None,
) -> Iterator[dict[str, Any]]:
    """Run an upstream ``RuntimeBundle`` and expose Mio-shaped events.

    Imports are delayed so pure adapter/compatibility tests do not require MLX.
    PQ/TQ, sampling, dynamic suppression and Mio ``warm_state`` are intentionally
    absent; callers must pass the compatibility gate before using this runner.
    """

    from dflash_mlx.runtime import stream_dflash_generate
    from dflash_mlx.runtime.context import build_offline_runtime_context

    context = runtime_context or build_offline_runtime_context(
        target_fa_window=request.target_fa_window,
        prefill_step_size=request.prefill_step_size,
        draft_sink_size=request.draft_sink_size,
        draft_window_size=request.draft_window_size,
        verify_len_cap=request.verify_len_cap,
        verify_mode=request.verify_mode,
    )
    raw = stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        use_chat_template=request.use_chat_template,
        block_tokens=request.block_tokens,
        stop_token_ids=list(request.stop_token_ids) or None,
        suppress_token_ids=list(request.suppress_token_ids) or None,
        prompt_tokens_override=(
            list(request.prompt_tokens_override)
            if request.prompt_tokens_override is not None
            else None
        ),
        quantize_kv_cache=request.quantize_kv_cache,
        prefix_snapshot=request.prefix_snapshot,
        snapshot_service=request.snapshot_service,
        stable_prefix_len=request.stable_prefix_len,
        prefix_cache_active=request.prefix_cache_active,
        publish_generation_snapshot=request.publish_generation_snapshot,
        runtime_context=context,
    )
    yield from adapt_upstream_stream(raw, adapter=adapter)
