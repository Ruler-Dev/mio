"""Runtime and engine integration tests for DSpark selection."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mio.config import TierConfig
from mio.drafter_selection import DrafterKind, DrafterPlan
from mio.dspark_runtime import DSparkRuntime
from mio.engine import MioEngine


class _Tokenizer:
    eos_token_ids = [0]
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids if token_id)


def _tier(**overrides) -> TierConfig:
    values = {
        "name": "test",
        "target_model": "target",
        "draft_model": "hybrid",
        "draft_fallback_model": "pure-dflash",
        "context_window": 4096,
        "max_output_tokens": 64,
    }
    values.update(overrides)
    return TierConfig(**values)


def _dspark_plan(*, strict=False) -> DrafterPlan:
    return DrafterPlan(
        requested="auto",
        detected=DrafterKind.HYBRID_DFLASH_MARKOV,
        primary_backend="dspark",
        primary_ref="hybrid",
        fallback_ref="pure-dflash",
        strict=strict,
        reason="auto_detected_hybrid_dflash_markov",
    )


def test_engine_loads_metadata_selected_dspark(monkeypatch):
    runtime = SimpleNamespace(close=lambda: None)
    captured = {}

    def load(**kwargs):
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr("mio.drafter_selection.plan_drafter", lambda *_args: _dspark_plan())
    monkeypatch.setattr("mio.dspark_runtime.DSparkRuntime.load", load)
    engine = MioEngine(_tier())
    engine._target_model = object()
    engine._tokenizer = _Tokenizer()
    engine._target_meta = {"config": {"model_type": "qwen3_5"}}

    engine._load_draft(engine.tier_config)

    assert engine._dspark_runtime is runtime
    assert engine._draft_model is None
    assert captured["draft_ref"] == "hybrid"
    status = engine.drafter_status
    assert status["requested"] == "auto"
    assert status["detected"] == "hybrid_dflash_markov"
    assert status["selected"] == "dspark"
    assert status["reason"] == "auto_detected_hybrid_dflash_markov"
    assert status["ref"] == "hybrid"
    assert status["fallback_used"] is False
    assert status["strict"] is False
    assert status["dspark"]["max_draft_tokens"] == 2
    assert status["dspark"]["prefix_cache"]["reason"] == "runtime_status_unavailable"
    assert captured["max_draft_tokens"] == 2
    assert captured["lookup_drafts"] is True


def test_dspark_load_failure_uses_distinct_dflash_with_telemetry(monkeypatch):
    monkeypatch.setattr("mio.drafter_selection.plan_drafter", lambda *_args: _dspark_plan())
    monkeypatch.setattr(
        "mio.dspark_runtime.DSparkRuntime.load",
        lambda **_kwargs: (_ for _ in ()).throw(ImportError("mlx-dspark unavailable")),
    )
    engine = MioEngine(_tier())
    engine._target_model = object()
    engine._tokenizer = _Tokenizer()
    engine._target_meta = {"config": {}}
    fallback_model = object()

    def load_fallback(ref):
        assert ref == "pure-dflash"
        engine._draft_model = fallback_model
        return "/resolved/pure-dflash"

    monkeypatch.setattr(engine, "_load_dflash", load_fallback)

    engine._load_draft(engine.tier_config)

    assert engine._dspark_runtime is None
    assert engine._draft_model is fallback_model
    status = engine.drafter_status
    assert status["selected"] == "dflash"
    assert status["fallback_used"] is True
    assert status["ref"] == "/resolved/pure-dflash"
    assert "dspark_load_failed_using_compatible_dflash" in status["reason"]
    metrics = engine._metrics_from_result({"elapsed_us": 1, "prefill_us": 0})
    assert metrics.drafter_requested == "auto"
    assert metrics.drafter_selected == "dflash"
    assert metrics.drafter_fallback_used is True
    assert "mlx-dspark unavailable" in metrics.drafter_reason


def test_strict_dspark_load_failure_never_calls_dflash(monkeypatch):
    monkeypatch.setattr(
        "mio.drafter_selection.plan_drafter",
        lambda *_args: _dspark_plan(strict=True),
    )
    monkeypatch.setattr(
        "mio.dspark_runtime.DSparkRuntime.load",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bad tensors")),
    )
    engine = MioEngine(_tier(drafter_strict=True))
    engine._target_model = object()
    engine._tokenizer = _Tokenizer()
    monkeypatch.setattr(
        engine,
        "_load_dflash",
        lambda _ref: pytest.fail("strict mode must not load a fallback"),
    )

    with pytest.raises(RuntimeError, match="dspark_load_failed_strict"):
        engine._load_draft(engine.tier_config)

    assert engine.drafter_status["selected"] == "baseline"
    assert "bad tensors" in engine.drafter_status["reason"]


def test_nonstream_generation_uses_dspark_for_stochastic_sampling(monkeypatch):
    captured = {}

    class Runtime:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return {
                "backend": "dspark",
                "text": "sampled",
                "generated_token_ids": [1, 2],
                "generation_tokens": 2,
                "prompt_token_count": 3,
                "prefill_us": 100,
                "elapsed_us": 500,
                "tokens_per_cycle": 2.0,
                "cycles_completed": 1,
            }

    engine = MioEngine(_tier())
    engine._loaded = True
    engine._target_model = object()
    engine._tokenizer = _Tokenizer()
    engine._dspark_runtime = Runtime()
    engine._drafter_selected = "dspark"
    engine._drafter_detected = "hybrid_dflash_markov"
    engine._drafter_reason = "auto_detected_hybrid_dflash_markov"
    engine._apply_chat_template = lambda _messages, tools=None: [10, 11, 12]
    monkeypatch.setattr(engine, "_make_sampler", lambda *_args: object())

    text, metrics = engine.generate(
        [{"role": "user", "content": "x"}],
        max_tokens=9,
        temperature=0.7,
        top_p=0.8,
        top_k=4,
        seed=123,
        stop=["END"],
    )

    assert text == "sampled"
    assert captured == {
        "prompt_ids": [10, 11, 12],
        "max_new_tokens": 9,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 4,
        "seed": 123,
        "stop": ["END"],
    }
    assert metrics.drafter_selected == "dspark"
    assert metrics.fallback_ar is False
    assert metrics.avg_acceptance_length == 2.0
    assert metrics.generation_backend == "dspark"


def test_stream_generation_forwards_dspark_chunks_and_metrics():
    class Runtime:
        def stream(self, **kwargs):
            assert kwargs["prompt_ids"] == [10, 11]
            yield "hel", None
            yield "lo", None
            yield (
                "",
                {
                    "backend": "dspark",
                    "generated_token_ids": [1, 2],
                    "generation_tokens": 2,
                    "prompt_token_count": 2,
                    "prefill_us": 10,
                    "elapsed_us": 30,
                    "tokens_per_cycle": 2.0,
                    "cycles_completed": 1,
                },
            )

    engine = MioEngine(_tier())
    engine._loaded = True
    engine._tokenizer = _Tokenizer()
    engine._dspark_runtime = Runtime()
    engine._drafter_selected = "dspark"
    engine._apply_chat_template = lambda _messages, tools=None: [10, 11]

    output = list(engine.generate_stream([{"role": "user", "content": "x"}]))

    assert "".join(chunk for chunk, metrics in output if metrics is None) == "hello"
    assert output[-1][1] is not None
    assert output[-1][1].drafter_selected == "dspark"


def test_runtime_normalizes_generation_and_streams_on_worker(monkeypatch):
    worker_threads = []

    class Result:
        text = "hello"
        token_ids = [1, 2]
        num_tokens = 2
        num_rounds = 1
        accept_lengths = [2]
        target_forwards = 2
        seconds = 0.01
        finish_reason = "length"
        lookup_rounds = 0
        mean_accept_len = 2.0

    def generate(_target, _tokenizer, _drafter, **kwargs):
        worker_threads.append(threading.current_thread().name)
        kwargs["on_text"]("he")
        kwargs["on_text"]("llo")
        assert kwargs["apply_chat_template"] is False
        assert kwargs["max_draft_tokens"] == 2
        return Result()

    monkeypatch.setattr("mlx_dspark.speculative_generate", generate)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-dspark")
    runtime = DSparkRuntime(executor, object(), _Tokenizer(), object(), "draft")
    try:
        result = runtime.generate(prompt_ids=[4, 5, 6], max_new_tokens=2)
        streamed = list(runtime.stream(prompt_ids=[4, 5, 6], max_new_tokens=2))
    finally:
        runtime.close()

    assert result["generated_token_ids"] == [1, 2]
    assert result["prompt_token_count"] == 3
    assert result["tokens_per_cycle"] == 2.0
    assert result["acceptance_ratio"] is None
    assert result["acceptance_ratio_available"] is False
    assert [chunk for chunk, item in streamed if item is None] == ["he", "llo"]
    assert streamed[-1][1]["backend"] == "dspark"
    assert all(name.startswith("test-dspark") for name in worker_threads)


def test_tool_required_nonstream_always_routes_through_stream_without_stop(monkeypatch):
    engine = MioEngine(_tier())
    engine._loaded = True
    engine._tokenizer = _Tokenizer()
    final = engine._metrics_from_result(
        {
            "generation_tokens": 1,
            "prompt_token_count": 1,
            "elapsed_us": 20,
            "prefill_us": 10,
            "backend": "baseline",
            "fallback_ar": True,
        }
    )
    captured = {}

    def stream(_messages, **kwargs):
        captured.update(kwargs)
        yield "<tool_call>ok</tool_call>", None
        yield "", final

    monkeypatch.setattr(engine, "generate_stream", stream)
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    text, metrics = engine.generate(
        [{"role": "user", "content": "x"}],
        tools=tools,
        tool_required=True,
    )

    assert text == "<tool_call>ok</tool_call>"
    assert metrics is final
    assert captured["tool_required"] is True
    assert captured["tools"] is tools
    assert captured["stop"] is None


def test_tool_required_stream_uses_target_only_with_explicit_reason(monkeypatch):
    class Runtime:
        def stream(self, **_kwargs):
            pytest.fail("required-tool streaming must not silently enter DSpark")

    class Detokenizer:
        last_segment = ""

        def reset(self):
            pass

        def add_token(self, _token):
            pass

        def finalize(self):
            pass

    engine = MioEngine(_tier())
    engine._loaded = True
    engine._target_model = object()
    engine._tokenizer = _Tokenizer()
    engine._dspark_runtime = Runtime()
    engine._drafter_selected = "dspark"
    engine._apply_chat_template = lambda _messages, tools=None: [1, 2, 3]
    engine._new_streaming_detokenizer = Detokenizer
    monkeypatch.setattr(engine, "_make_sampler", lambda *_args: object())
    captured = {}

    def baseline(**kwargs):
        captured.update(kwargs)
        yield {"event": "prefill", "prefill_us": 10, "prompt_token_count": 3}
        yield {
            "event": "summary",
            "generated_token_ids": [],
            "generation_tokens": 0,
            "prompt_token_count": 3,
            "prefill_us": 10,
            "elapsed_us": 20,
            "fallback_ar": True,
            "fallback_reason": kwargs["fallback_reason"],
        }

    monkeypatch.setattr("mio.dflash.runtime.stream_baseline_generate", baseline)
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    output = list(
        engine.generate_stream(
            [{"role": "user", "content": "x"}],
            tools=tools,
            tool_required=True,
        )
    )

    metrics = output[-1][1]
    assert captured["suppress_token_ids"] == [0]
    assert captured["fallback_reason"] == ("dspark_tool_required_uses_target_only_for_eos_suppression")
    assert metrics is not None
    assert metrics.fallback_ar is True
    assert metrics.generation_backend == "baseline"
    assert metrics.fallback_reason == captured["fallback_reason"]


def test_tool_required_ddtree_uses_target_stream_with_dynamic_eos(monkeypatch):
    class Detokenizer:
        last_segment = ""

        def add_token(self, _token):
            pass

        def finalize(self):
            pass

    engine = MioEngine(_tier(ddtree_budget=8))
    engine._loaded = True
    engine._target_model = object()
    engine._draft_model = object()
    engine._tokenizer = _Tokenizer()
    engine._target_meta = {"target_family": "hybrid_gdn"}
    engine._apply_chat_template = lambda _messages, tools=None: [1, 2, 3]
    engine._new_streaming_detokenizer = Detokenizer
    monkeypatch.setattr(engine, "_make_sampler", lambda *_args: object())
    monkeypatch.setattr(
        "mio.ddtree.runtime.stream_ddtree_generate",
        lambda **_kwargs: pytest.fail("required-tool generation must bypass DDTree's fixed EOS mask"),
    )
    captured = {}

    def baseline(**kwargs):
        captured.update(kwargs)
        yield {"event": "prefill", "prefill_us": 10, "prompt_token_count": 3}
        yield {
            "event": "summary",
            "generated_token_ids": [],
            "generation_tokens": 0,
            "prompt_token_count": 3,
            "prefill_us": 10,
            "elapsed_us": 20,
            "fallback_ar": True,
            "fallback_reason": kwargs["fallback_reason"],
        }

    monkeypatch.setattr("mio.dflash.runtime.stream_baseline_generate", baseline)
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    _text, metrics = engine.generate(
        [{"role": "user", "content": "x"}],
        tools=tools,
        tool_required=True,
    )

    assert captured["suppress_token_ids"] == [0]
    assert captured["relax_suppress_after"] == 40
    assert captured["relax_suppress_token_ids"] == [0]
    assert captured["fallback_reason"] == ("ddtree_tool_required_uses_target_only_for_dynamic_eos_suppression")
    assert metrics.generation_backend == "baseline"
    assert metrics.fallback_ar is True
    assert metrics.fallback_reason == captured["fallback_reason"]


def test_tool_required_nonstream_with_text_stop_routes_through_stream(monkeypatch):
    engine = MioEngine(_tier())
    engine._loaded = True
    engine._tokenizer = _Tokenizer()
    engine._dspark_runtime = object()
    final = engine._metrics_from_result(
        {
            "generation_tokens": 1,
            "prompt_token_count": 1,
            "elapsed_us": 20,
            "prefill_us": 10,
            "backend": "baseline",
            "fallback_ar": True,
            "fallback_reason": ("dspark_tool_required_uses_target_only_for_eos_suppression"),
        }
    )
    captured = {}

    def stream(_messages, **kwargs):
        captured.update(kwargs)
        yield "safe", None
        yield "", final

    monkeypatch.setattr(engine, "generate_stream", stream)
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    text, metrics = engine.generate(
        [{"role": "user", "content": "x"}],
        tools=tools,
        tool_required=True,
        stop=["STOP"],
    )

    assert text == "safe"
    assert metrics is final
    assert captured["tool_required"] is True
    assert captured["stop"] == ["STOP"]


def test_bmp_configuration_selects_compatible_dflash_before_dspark(monkeypatch):
    monkeypatch.setattr("mio.drafter_selection.plan_drafter", lambda *_args: _dspark_plan())
    monkeypatch.setattr(
        "mio.dspark_runtime.DSparkRuntime.load",
        lambda **_kwargs: pytest.fail("BMP must select its DFlash-capable backend"),
    )
    engine = MioEngine(_tier(bmp_paths=2))
    engine._target_model = object()
    engine._tokenizer = _Tokenizer()
    fallback_model = object()

    def load_fallback(ref):
        assert ref == "pure-dflash"
        engine._draft_model = fallback_model
        return ref

    monkeypatch.setattr(engine, "_load_dflash", load_fallback)

    engine._load_draft(engine.tier_config)

    assert engine._dspark_runtime is None
    assert engine._draft_model is fallback_model
    assert engine.drafter_status["selected"] == "dflash"
    assert engine.drafter_status["fallback_used"] is True
    assert "bmp_paths=2" in engine.drafter_status["reason"]
    assert any("requires_dflash" in item for item in engine.drafter_status["capability_policy"])


def test_runtime_rejects_cap_above_validated_parity_boundary():
    with pytest.raises(ValueError, match="cap >=4 failed"):
        DSparkRuntime.load(
            target_model=object(),
            tokenizer=object(),
            draft_ref="unused",
            max_draft_tokens=4,
        )


def test_stream_consumer_close_signals_worker_cancellation(monkeypatch):
    finished = threading.Event()

    class Result:
        text = "partial"
        token_ids = [1]
        num_tokens = 1
        num_rounds = 1
        accept_lengths = [1]
        target_forwards = 1
        seconds = 0.01
        finish_reason = "stop"
        lookup_rounds = 0
        mean_accept_len = 1.0

    def generate(_target, _tokenizer, _drafter, **kwargs):
        from mlx_dspark import StopStreaming

        try:
            kwargs["on_text"]("first")
            while True:
                kwargs["on_text"]("later")
        except StopStreaming:
            return Result()
        finally:
            finished.set()

    monkeypatch.setattr("mlx_dspark.speculative_generate", generate)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cancel-dspark")
    runtime = DSparkRuntime(executor, object(), _Tokenizer(), object(), "draft")
    stream = runtime.stream(prompt_ids=[1], max_new_tokens=16)
    try:
        assert next(stream) == ("first", None)
        stream.close()
        assert finished.wait(timeout=2.0)
    finally:
        runtime.close()


def test_runtime_resets_checked_out_prefix_cache_after_generation_error(monkeypatch):
    class Prefix:
        reset_called = False

        def acquire(self, _prompt_ids):
            return object(), object(), 2

        def reset(self):
            self.reset_called = True

        def info(self):
            return {"slots": []}

    prefix = Prefix()
    monkeypatch.setattr(
        "mlx_dspark.speculative_generate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("generation failed")),
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefix-dspark")
    runtime = DSparkRuntime(
        executor,
        object(),
        _Tokenizer(),
        object(),
        "draft",
        _prefix_cache=prefix,
    )
    try:
        with pytest.raises(RuntimeError, match="generation failed"):
            runtime.generate(prompt_ids=[1, 2, 3], max_new_tokens=2)
        assert prefix.reset_called is True
    finally:
        runtime.close()


def test_runtime_passes_upstream_prefix_state_and_reports_reuse(monkeypatch):
    cache = object()
    ctx = object()
    captured = {}

    class Prefix:
        def acquire(self, prompt_ids):
            assert prompt_ids == [1, 2, 3]
            return cache, ctx, 2

        def store(self, got_cache, got_ctx, prompt_ids, token_ids):
            captured["store"] = (got_cache, got_ctx, prompt_ids, token_ids)

        def reset(self):
            pass

        def info(self):
            return {
                "slots": [{"tokens": 4}],
                "hits": 1,
                "reused_tokens": 2,
            }

    class Result:
        text = "ok"
        token_ids = [7]
        num_tokens = 1
        num_rounds = 1
        accept_lengths = [1]
        target_forwards = 1
        seconds = 0.01
        finish_reason = "length"
        lookup_rounds = 0
        mean_accept_len = 1.0

    def generate(_target, _tokenizer, _drafter, **kwargs):
        captured["generate"] = kwargs
        kwargs["on_text"]("ok")
        return Result()

    monkeypatch.setattr("mlx_dspark.speculative_generate", generate)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefix-dspark")
    runtime = DSparkRuntime(
        executor,
        object(),
        _Tokenizer(),
        object(),
        "draft",
        max_draft_tokens=3,
        lookup_drafts=False,
        _prefix_cache=Prefix(),
        _prefix_cache_reason="enabled_upstream",
    )
    try:
        result = runtime.generate(prompt_ids=[1, 2, 3], max_new_tokens=2)
    finally:
        runtime.close()

    assert captured["generate"]["cache"] is cache
    assert captured["generate"]["ctx_caches"] is ctx
    assert captured["generate"]["reuse_len"] == 2
    assert captured["generate"]["max_draft_tokens"] == 3
    assert captured["generate"]["lookup_drafts"] is False
    assert captured["store"] == (cache, ctx, [1, 2, 3], [7])
    assert result["warm_offset"] == 2
    assert result["cache_entries"] == 1
    assert result["prefix_cache_hits"] == 1


def test_runtime_prefix_acquire_failure_falls_back_to_cold_generation(monkeypatch):
    captured = {}

    class Prefix:
        def acquire(self, _prompt_ids):
            raise RuntimeError("broken acquire")

        def reset(self):
            pass

    class Result:
        text = "ok"
        token_ids = [7]
        num_tokens = 1
        num_rounds = 1
        accept_lengths = [1]
        target_forwards = 1
        finish_reason = "length"
        lookup_rounds = 0
        mean_accept_len = 1.0

    def generate(_target, _tokenizer, _drafter, **kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr("mlx_dspark.speculative_generate", generate)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefix-acquire")
    runtime = DSparkRuntime(
        executor,
        object(),
        _Tokenizer(),
        object(),
        "draft",
        _prefix_cache=Prefix(),
        _prefix_cache_reason="enabled_upstream",
    )
    try:
        result = runtime.generate(prompt_ids=[1, 2], max_new_tokens=1)
        status = runtime.prefix_cache_status
    finally:
        runtime.close()

    assert result["text"] == "ok"
    assert captured["cache"] is None
    assert captured["ctx_caches"] is None
    assert captured["reuse_len"] == 0
    assert status == {"enabled": False, "reason": "acquire_failed:RuntimeError"}


def test_runtime_prefix_store_failure_preserves_valid_output(monkeypatch):
    class Prefix:
        def acquire(self, _prompt_ids):
            return object(), object(), 1

        def store(self, *_args):
            raise ValueError("broken store")

        def reset(self):
            pass

    class Result:
        text = "ok"
        token_ids = [7]
        num_tokens = 1
        num_rounds = 1
        accept_lengths = [1]
        target_forwards = 1
        finish_reason = "length"
        lookup_rounds = 0
        mean_accept_len = 1.0

    monkeypatch.setattr(
        "mlx_dspark.speculative_generate",
        lambda *_args, **_kwargs: Result(),
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefix-store")
    runtime = DSparkRuntime(
        executor,
        object(),
        _Tokenizer(),
        object(),
        "draft",
        _prefix_cache=Prefix(),
        _prefix_cache_reason="enabled_upstream",
    )
    try:
        result = runtime.generate(prompt_ids=[1, 2], max_new_tokens=1)
        status = runtime.prefix_cache_status
    finally:
        runtime.close()

    assert result["text"] == "ok"
    assert status == {"enabled": False, "reason": "store_failed:ValueError"}


def test_runtime_prefix_info_failure_returns_disabled_status(monkeypatch):
    class Prefix:
        def info(self):
            raise RuntimeError("broken info")

        def reset(self):
            pass

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefix-info")
    runtime = DSparkRuntime(
        executor,
        object(),
        _Tokenizer(),
        object(),
        "draft",
        _prefix_cache=Prefix(),
        _prefix_cache_reason="enabled_upstream",
    )
    try:
        status = runtime.prefix_cache_status
    finally:
        runtime.close()

    assert status == {"enabled": False, "reason": "info_failed:RuntimeError"}


def test_runtime_close_shuts_executor_when_prefix_reset_fails():
    class Prefix:
        def reset(self):
            raise RuntimeError("broken reset")

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefix-close")
    runtime = DSparkRuntime(
        executor,
        object(),
        _Tokenizer(),
        object(),
        "draft",
        _prefix_cache=Prefix(),
    )

    runtime.close()

    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        executor.submit(lambda: None)


def test_engine_prefix_status_failure_is_nonfatal_and_runtime_is_owned(monkeypatch):
    class Runtime:
        closed = False

        @property
        def prefix_cache_status(self):
            raise RuntimeError("broken status")

        def close(self):
            self.closed = True

    runtime = Runtime()
    monkeypatch.setattr("mio.drafter_selection.plan_drafter", lambda *_args: _dspark_plan())
    monkeypatch.setattr(
        "mio.dspark_runtime.DSparkRuntime.load",
        lambda **_kwargs: runtime,
    )
    engine = MioEngine(_tier())
    engine._target_model = object()
    engine._tokenizer = _Tokenizer()

    engine._load_draft(engine.tier_config)

    assert engine.drafter_status["selected"] == "dspark"
    assert engine.drafter_status["dspark"]["prefix_cache"] == {
        "enabled": False,
        "reason": "status_failed:RuntimeError",
    }
    engine.unload()
    assert runtime.closed is True


def test_engine_unload_invalidates_prefix_arrays_even_after_partial_load():
    engine = MioEngine(_tier())
    engine._target_model = object()
    engine._prefix_cache = {(1, 2): {"target_cache": object()}}
    engine._last_prompt_tokens = [1, 2]

    engine.unload()

    assert engine._prefix_cache == {}
    assert engine._last_prompt_tokens == []
    assert engine._target_model is None
