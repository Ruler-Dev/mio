"""Concurrency regressions for model lifecycle and HTTP/UI generation paths."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mio import model_manager as manager_module
from mio import server
from mio.engine import GenerationMetrics
from mio.webui import router as ui_router
from mio.webui import webhooks


@pytest.mark.asyncio
async def test_model_load_and_unload_run_in_worker_under_gpu_lock(monkeypatch):
    main_thread = threading.get_ident()
    calls: list[tuple[str, int, bool]] = []

    class Manager:
        def load_tier(self, tier: str) -> None:
            calls.append((f"load:{tier}", threading.get_ident(), server._GPU_LOCK.locked()))

        def unload_tier(self, tier: str) -> None:
            calls.append((f"unload:{tier}", threading.get_ident(), server._GPU_LOCK.locked()))

    monkeypatch.setattr(server, "_manager", Manager())

    assert await server.load_model(server.TierLoadRequest(tier="small")) == {
        "status": "loaded",
        "tier": "small",
    }
    assert await server.unload_model(server.TierLoadRequest(tier="small")) == {
        "status": "unloaded",
        "tier": "small",
    }

    assert [call[0] for call in calls] == ["load:small", "unload:small"]
    assert all(thread_id != main_thread and locked for _, thread_id, locked in calls)


@pytest.mark.asyncio
async def test_chat_compaction_runs_off_loop_without_prelocking_gpu(monkeypatch):
    main_thread = threading.get_ident()
    compact_calls: list[tuple[int, bool, bool]] = []

    class Engine:
        def generate(self, _messages, **_kwargs):
            return "ok", GenerationMetrics(prompt_tokens=2, completion_tokens=1)

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    def compact(messages, _engine, *, gpu_lock, **_kwargs):
        compact_calls.append(
            (threading.get_ident(), server._GPU_LOCK.locked(), gpu_lock is None)
        )
        return messages, SimpleNamespace(triggered=False)

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_compact_enabled", True)
    monkeypatch.setattr(server, "_validate_enabled", False)
    monkeypatch.setattr(server, "_apply_policy", lambda messages, **_kwargs: messages)
    monkeypatch.setattr("mio.compactor.compact", compact)

    response = await server.chat_completions(
        server.ChatCompletionRequest(
            model="mio-small",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert compact_calls == [(compact_calls[0][0], True, True)]
    assert compact_calls[0][0] != main_thread


@pytest.mark.asyncio
async def test_webhook_generation_runs_off_loop_under_gpu_lock(monkeypatch):
    main_thread = threading.get_ident()
    gpu_lock = threading.Lock()
    calls: list[tuple[int, bool, list[dict], int]] = []
    logs: list[dict] = []

    class Engine:
        def generate(self, messages, *, max_tokens):
            calls.append((threading.get_ident(), gpu_lock.locked(), messages, max_tokens))
            return "webhook output", GenerationMetrics()

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    monkeypatch.setattr(ui_router, "_manager", Manager())
    monkeypatch.setattr(ui_router, "_gpu_lock", gpu_lock)
    monkeypatch.setattr(
        webhooks,
        "load_webhooks",
        lambda: [{
            "slug": "build",
            "prompt": "Build {{target}}",
            "tier": "small",
            "secret": "test-secret",
        }],
    )
    monkeypatch.setattr(
        webhooks,
        "render_prompt",
        lambda prompt, payload: prompt.replace("{{target}}", payload["target"]),
    )
    monkeypatch.setattr(webhooks, "append_log", lambda _slug, _payload, result: logs.append(result))

    request = SimpleNamespace(headers={"x-mio-webhook-secret": "test-secret"})
    result = await ui_router.webhook_fire("build", request, {"target": "Mio"})

    assert result == {
        "ok": True,
        "slug": "build",
        "tier": "small",
        "output": "webhook output",
    }
    assert calls == [
        (
            calls[0][0],
            True,
            [{"role": "user", "content": "Build Mio"}],
            2048,
        )
    ]
    assert calls[0][0] != main_thread
    assert logs == [result]


@pytest.mark.asyncio
async def test_generation_holds_lifecycle_gpu_lock_against_unload(monkeypatch):
    generation_started = threading.Event()
    allow_generation_finish = threading.Event()
    unload_started = threading.Event()

    class Engine:
        is_loaded = True

        def generate(self, _messages, **_kwargs):
            assert server._GPU_LOCK.locked()
            assert self.is_loaded
            generation_started.set()
            assert allow_generation_finish.wait(timeout=1.0)
            assert self.is_loaded
            return "safe", GenerationMetrics(prompt_tokens=1, completion_tokens=1)

    engine = Engine()

    class Manager:
        def __init__(self):
            self.lookups = 0

        def loaded_tiers(self):
            return ["small"] if engine.is_loaded else []

        def get_engine(self, tier):
            assert tier == "small"
            self.lookups += 1
            if not engine.is_loaded:
                raise RuntimeError("engine unloaded")
            return engine

        def unload_tier(self, tier):
            assert tier == "small"
            assert server._GPU_LOCK.locked()
            unload_started.set()
            engine.is_loaded = False

    manager = Manager()
    monkeypatch.setattr(server, "_manager", manager)
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_compact_enabled", False)
    monkeypatch.setattr(server, "_validate_enabled", False)
    monkeypatch.setattr(server, "_GPU_LOCK", threading.Lock())
    monkeypatch.setattr(server, "_apply_policy", lambda messages, **_kwargs: messages)

    chat_task = asyncio.create_task(
        server.chat_completions(
            server.ChatCompletionRequest(
                model="mio-small",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
    )
    assert await asyncio.to_thread(generation_started.wait, 1.0)

    unload_task = asyncio.create_task(
        server.unload_model(server.TierLoadRequest(tier="small"))
    )
    await asyncio.sleep(0.03)
    assert not unload_started.is_set()
    assert engine.is_loaded

    allow_generation_finish.set()
    response = await chat_task
    assert response["choices"][0]["message"]["content"] == "safe"
    assert await unload_task == {"status": "unloaded", "tier": "small"}
    assert unload_started.is_set()
    assert not engine.is_loaded
    # One route validation plus a second lookup inside the lifecycle lock.
    assert manager.lookups == 2


@pytest.mark.asyncio
async def test_rest_cancelled_while_waiting_never_runs_abandoned_generation(monkeypatch):
    gpu_lock = threading.Lock()
    gpu_lock.acquire()
    generation_started = threading.Event()
    lookups = 0

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            generation_started.set()
            yield "abandoned", GenerationMetrics(completion_tokens=1)

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            nonlocal lookups
            assert tier == "small"
            lookups += 1
            return Engine()

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_compact_enabled", False)
    monkeypatch.setattr(server, "_validate_enabled", False)
    monkeypatch.setattr(server, "_GPU_LOCK", gpu_lock)
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})
    monkeypatch.setattr(server, "_apply_policy", lambda messages, **_kwargs: messages)

    task = asyncio.create_task(
        server.chat_completions(
            server.ChatCompletionRequest(
                model="mio-small",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gpu_lock.release()

    # The route performs one eager validation lookup; the cancelled worker
    # must never perform the second, under-lock lookup or begin generation.
    assert lookups == 1
    assert not generation_started.is_set()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_rest_cancelled_while_compaction_waits_never_runs_it(monkeypatch):
    gpu_lock = threading.Lock()
    gpu_lock.acquire()
    compact_called = threading.Event()

    class Engine:
        pass

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    def compact(*_args, **_kwargs):
        compact_called.set()
        return [], SimpleNamespace(triggered=False)

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_compact_enabled", True)
    monkeypatch.setattr(server, "_GPU_LOCK", gpu_lock)
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})
    monkeypatch.setattr("mio.compactor.compact", compact)

    task = asyncio.create_task(
        server.chat_completions(
            server.ChatCompletionRequest(
                model="mio-small",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gpu_lock.release()

    assert not compact_called.is_set()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_rest_disconnect_while_waiting_returns_499_without_generation(monkeypatch):
    gpu_lock = threading.Lock()
    gpu_lock.acquire()
    disconnected = threading.Event()
    generation_started = threading.Event()

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            generation_started.set()
            yield "abandoned", GenerationMetrics(completion_tokens=1)

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    class Request:
        async def is_disconnected(self):
            return disconnected.is_set()

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_compact_enabled", False)
    monkeypatch.setattr(server, "_validate_enabled", False)
    monkeypatch.setattr(server, "_GPU_LOCK", gpu_lock)
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})
    monkeypatch.setattr(server, "_apply_policy", lambda messages, **_kwargs: messages)

    task = asyncio.create_task(
        server.chat_completions(
            server.ChatCompletionRequest(
                model="mio-small",
                messages=[{"role": "user", "content": "hello"}],
            ),
            Request(),
        )
    )
    await asyncio.sleep(0.05)
    disconnected.set()
    with pytest.raises(server.HTTPException) as closed:
        await task
    gpu_lock.release()

    assert closed.value.status_code == 499
    assert not generation_started.is_set()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_rest_cancellation_closes_active_stream_before_releasing_gpu(monkeypatch):
    gpu_lock = threading.Lock()
    generation_started = threading.Event()
    allow_inflight_step = threading.Event()
    source_closed = threading.Event()

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            try:
                generation_started.set()
                yield "first", None
                assert allow_inflight_step.wait(timeout=1.0)
                yield "discarded", GenerationMetrics(completion_tokens=2)
            finally:
                assert gpu_lock.locked()
                source_closed.set()

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_compact_enabled", False)
    monkeypatch.setattr(server, "_validate_enabled", False)
    monkeypatch.setattr(server, "_GPU_LOCK", gpu_lock)
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})
    monkeypatch.setattr(server, "_apply_policy", lambda messages, **_kwargs: messages)

    task = asyncio.create_task(
        server.chat_completions(
            server.ChatCompletionRequest(
                model="mio-small",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
    )
    assert await asyncio.to_thread(generation_started.wait, 1.0)
    task.cancel()
    await asyncio.sleep(0.02)
    assert not task.done()
    allow_inflight_step.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert source_closed.is_set()
    assert not gpu_lock.locked()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_validation_retry_cancelled_on_gpu_queue_never_starts(monkeypatch):
    gpu_lock = threading.Lock()
    validation_started = threading.Event()
    attempts = 0

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            nonlocal attempts
            attempts += 1
            yield "first attempt", GenerationMetrics(
                prompt_tokens=2,
                completion_tokens=2,
            )

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    def validate_response(_text):
        # The initial generation has released the lock. Hold it while the retry
        # worker is enqueued, then cancel the owning request.
        gpu_lock.acquire()
        validation_started.set()
        return SimpleNamespace(passed=False, errors=["retry"])

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_compact_enabled", False)
    monkeypatch.setattr(server, "_validate_enabled", True)
    monkeypatch.setattr(server, "_GPU_LOCK", gpu_lock)
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})
    monkeypatch.setattr(server, "_apply_policy", lambda messages, **_kwargs: messages)
    monkeypatch.setattr("mio.validator.validate_response", validate_response)
    monkeypatch.setattr("mio.validator.build_retry_message", lambda _errors: "retry")

    task = asyncio.create_task(
        server.chat_completions(
            server.ChatCompletionRequest(
                model="mio-small",
                messages=[{"role": "user", "content": "hello"}],
                validate=True,
            )
        )
    )
    assert await asyncio.to_thread(validation_started.wait, 1.0)
    for _ in range(20):
        with server._STREAM_PRODUCERS_LOCK:
            retry_queued = any("retry" in thread.name for thread in server._STREAM_PRODUCERS)
        if retry_queued:
            break
        await asyncio.sleep(0.01)
    assert retry_queued

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gpu_lock.release()

    assert attempts == 1
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


def test_model_manager_serializes_load_and_lifecycle_snapshots(monkeypatch):
    instances = []
    load_count = 0
    count_lock = threading.Lock()
    unload_started = threading.Event()
    allow_unload = threading.Event()

    class Engine:
        def __init__(self, tier_config):
            self.tier_config = tier_config
            self.is_loaded = False
            self.last_metrics = GenerationMetrics()
            instances.append(self)

        def load(self):
            nonlocal load_count
            with count_lock:
                load_count += 1
            time.sleep(0.03)
            self.is_loaded = True

        def unload(self):
            unload_started.set()
            assert allow_unload.wait(timeout=1.0)
            self.is_loaded = False

    tier = SimpleNamespace(
        target_model="target",
        draft_model="draft",
        context_window=4096,
    )
    config = SimpleNamespace(tiers={"small": tier}, active_tiers=["small"], tandem=False)
    monkeypatch.setattr(manager_module, "MioEngine", Engine)
    manager = manager_module.ModelManager(config)
    monkeypatch.setattr(manager, "total_vram_gb", lambda: 0.0)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _index: manager.load_tier("small"), range(4)))
    assert load_count == 1
    assert len(instances) == 1

    with ThreadPoolExecutor(max_workers=3) as pool:
        unload_future = pool.submit(manager.unload_tier, "small")
        assert unload_started.wait(timeout=1.0)
        get_future = pool.submit(manager.get_engine, "small")
        status_future = pool.submit(manager.status)
        time.sleep(0.03)
        assert not get_future.done()
        assert not status_future.done()

        allow_unload.set()
        unload_future.result(timeout=1.0)
        with pytest.raises(RuntimeError, match="not loaded"):
            get_future.result(timeout=1.0)
        assert status_future.result(timeout=1.0)["loaded_tiers"] == []

    # Nested snapshot helpers use the RLock and must not self-deadlock.
    assert manager.get_model_names() == []


def test_model_manager_prefers_current_mlx_peak_memory_api():
    calls: list[str] = []
    fake_mx = SimpleNamespace(
        get_peak_memory=lambda: calls.append("top-level") or 123.0,
        metal=SimpleNamespace(
            get_peak_memory=lambda: calls.append("deprecated") or 456.0
        ),
    )

    assert manager_module.ModelManager._peak_memory_bytes(fake_mx) == 123.0
    assert calls == ["top-level"]


def _decode_sse_data(chunk: str) -> dict | str | None:
    if not chunk.startswith("data: "):
        return None
    payload = chunk.removeprefix("data: ").strip()
    if payload == "[DONE]":
        return payload
    return json.loads(payload)


class _RecordingWebSocket:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.events.append(payload)


@pytest.mark.asyncio
async def test_webui_round_resolves_engine_under_lock_and_closes_source(monkeypatch):
    gpu_lock = threading.Lock()
    source_closed = threading.Event()
    lookups = 0
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            try:
                yield "hello", GenerationMetrics(prompt_tokens=2, completion_tokens=1)
            finally:
                source_closed.set()

    class Manager:
        def get_engine(self, tier):
            nonlocal lookups
            assert tier == "small"
            assert gpu_lock.locked()
            lookups += 1
            return Engine()

    websocket = _RecordingWebSocket()
    result = await ui_router._stream_webui_round(
        websocket,
        manager=Manager(),
        tier="small",
        gpu_lock=gpu_lock,
        messages=[{"role": "user", "content": "go"}],
        max_tokens=8,
        temperature=0.0,
        tools=None,
        is_first=True,
    )

    assert result.completed and not result.failed
    assert result.text == "hello"
    assert result.metrics == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "prompt_tps": 0.0,
        "generation_tps": 0.0,
        "acceptance_ratio": 0.0,
    }
    assert lookups == 1
    assert source_closed.is_set()
    assert not gpu_lock.locked()
    assert [event["type"] for event in websocket.events] == ["token"]
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_webui_round_slow_socket_applies_bounded_backpressure(monkeypatch):
    gpu_lock = threading.Lock()
    produced = 0
    source_closed = threading.Event()
    first_send = asyncio.Event()
    hold_send = asyncio.Event()
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            nonlocal produced
            try:
                for index in range(10_000):
                    produced += 1
                    yield str(index), None
            finally:
                source_closed.set()

    class Manager:
        def get_engine(self, _tier):
            assert gpu_lock.locked()
            return Engine()

    class SlowWebSocket(_RecordingWebSocket):
        async def send_json(self, payload: dict) -> None:
            self.events.append(payload)
            if payload.get("type") == "token":
                first_send.set()
                await hold_send.wait()

    task = asyncio.create_task(
        ui_router._stream_webui_round(
            SlowWebSocket(),
            manager=Manager(),
            tier="small",
            gpu_lock=gpu_lock,
            messages=[{"role": "user", "content": "go"}],
            max_tokens=10_000,
            temperature=0.0,
            tools=None,
            is_first=True,
        )
    )
    await asyncio.wait_for(first_send.wait(), timeout=1.0)
    await asyncio.sleep(0.15)
    assert produced <= ui_router._WS_STREAM_QUEUE_MAXSIZE + 2

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source_closed.is_set()
    assert not gpu_lock.locked()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_webui_round_cancelled_while_waiting_never_runs_abandoned_work(monkeypatch):
    gpu_lock = threading.Lock()
    gpu_lock.acquire()
    generation_started = threading.Event()
    lookups = 0
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            generation_started.set()
            yield "should not run", None

    class Manager:
        def get_engine(self, _tier):
            nonlocal lookups
            lookups += 1
            return Engine()

    task = asyncio.create_task(
        ui_router._stream_webui_round(
            _RecordingWebSocket(),
            manager=Manager(),
            tier="small",
            gpu_lock=gpu_lock,
            messages=[],
            max_tokens=8,
            temperature=0.0,
            tools=None,
            is_first=True,
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gpu_lock.release()

    assert lookups == 0
    assert not generation_started.is_set()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_webui_round_rechecks_unloaded_tier_after_lock_wait(monkeypatch):
    gpu_lock = threading.Lock()
    gpu_lock.acquire()
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})

    class Manager:
        def get_engine(self, tier):
            assert tier == "small"
            assert gpu_lock.locked()
            raise RuntimeError("tier 'small' is not loaded")

    websocket = _RecordingWebSocket()
    task = asyncio.create_task(
        ui_router._stream_webui_round(
            websocket,
            manager=Manager(),
            tier="small",
            gpu_lock=gpu_lock,
            messages=[],
            max_tokens=8,
            temperature=0.0,
            tools=None,
            is_first=True,
        )
    )
    await asyncio.sleep(0.05)
    gpu_lock.release()
    result = await task

    assert result.failed and not result.completed
    assert [event["type"] for event in websocket.events] == ["error"]
    assert "not loaded" in websocket.events[0]["message"]
    assert not gpu_lock.locked()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_webui_send_failure_cancels_and_closes_producer(monkeypatch):
    gpu_lock = threading.Lock()
    source_closed = threading.Event()
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            try:
                for index in range(10_000):
                    yield str(index), None
            finally:
                source_closed.set()

    class Manager:
        def get_engine(self, _tier):
            return Engine()

    class ClosedWebSocket:
        async def send_json(self, _payload: dict) -> None:
            from starlette.websockets import WebSocketDisconnect

            raise WebSocketDisconnect(code=1006)

    with pytest.raises(Exception) as disconnected:
        await ui_router._stream_webui_round(
            ClosedWebSocket(),
            manager=Manager(),
            tier="small",
            gpu_lock=gpu_lock,
            messages=[],
            max_tokens=10_000,
            temperature=0.0,
            tools=None,
            is_first=True,
        )
    assert disconnected.value.__class__.__name__ == "WebSocketDisconnect"
    assert source_closed.is_set()
    assert not gpu_lock.locked()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_webui_backend_error_never_emits_done_success(monkeypatch):
    gpu_lock = threading.Lock()
    source_closed = threading.Event()
    monkeypatch.setattr(server, "_STREAM_PRODUCERS", {})

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            try:
                yield "partial", None
                raise RuntimeError("backend exploded")
            finally:
                source_closed.set()

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            assert gpu_lock.locked()
            return Engine()

    websocket = _RecordingWebSocket()
    monkeypatch.setattr(ui_router, "_manager", Manager())
    monkeypatch.setattr(ui_router, "_gpu_lock", gpu_lock)
    monkeypatch.setattr(ui_router, "_load_memory", lambda: [])

    await ui_router._handle_chat(
        websocket,
        {
            "messages": [{"role": "user", "content": "go"}],
            "skills": False,
            "max_tokens": 8,
            "temperature": 0.0,
        },
    )

    event_types = [event["type"] for event in websocket.events]
    assert event_types == ["start", "token", "error"]
    assert "backend exploded" in websocket.events[-1]["message"]
    assert "done" not in event_types
    assert source_closed.is_set()
    assert not gpu_lock.locked()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_sse_slow_consumer_applies_bounded_backpressure(monkeypatch):
    produced = 0
    source_closed = threading.Event()
    monkeypatch.setattr(server, "_manager", None)
    monkeypatch.setattr(server, "_GPU_LOCK", threading.Lock())

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            nonlocal produced
            try:
                for index in range(10_000):
                    produced += 1
                    yield f"{index}", None
            finally:
                source_closed.set()

    stream = server._stream_response(
        Engine(),
        [{"role": "user", "content": "go"}],
        "chatcmpl-backpressure",
        1,
        "mio-small",
        "small",
    )

    role = _decode_sse_data(await anext(stream))
    assert isinstance(role, dict)
    first_content = _decode_sse_data(await anext(stream))
    assert isinstance(first_content, dict)
    await asyncio.sleep(0.15)

    # One yielded item, a full bounded queue, and at most one producer item
    # currently waiting for queue capacity.
    assert produced <= server._STREAM_QUEUE_MAXSIZE + 2

    await stream.aclose()
    assert source_closed.is_set()
    assert not server._GPU_LOCK.locked()


@pytest.mark.asyncio
async def test_sse_disconnect_cancels_producer_waiting_for_gpu(monkeypatch):
    gpu_lock = threading.Lock()
    gpu_lock.acquire()
    generation_started = threading.Event()
    monkeypatch.setattr(server, "_manager", None)
    monkeypatch.setattr(server, "_GPU_LOCK", gpu_lock)

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            generation_started.set()
            yield "should not run", None

    stream = server._stream_response(
        Engine(),
        [{"role": "user", "content": "go"}],
        "chatcmpl-disconnect",
        1,
        "mio-small",
        "small",
    )
    await anext(stream)  # role chunk; producer starts on the next pull
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    gpu_lock.release()
    await asyncio.sleep(0.05)
    assert not generation_started.is_set()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS


@pytest.mark.asyncio
async def test_sse_error_has_no_success_trailer_and_cleans_producer(monkeypatch):
    monkeypatch.setattr(server, "_manager", None)
    monkeypatch.setattr(server, "_GPU_LOCK", threading.Lock())

    class Engine:
        def generate_stream(self, _messages, **_kwargs):
            yield "partial", None
            raise RuntimeError("backend exploded")

    chunks = [
        chunk
        async for chunk in server._stream_response(
            Engine(),
            [{"role": "user", "content": "go"}],
            "chatcmpl-error",
            1,
            "mio-small",
            "small",
        )
    ]
    decoded = [_decode_sse_data(chunk) for chunk in chunks]

    errors = [item for item in decoded if isinstance(item, dict) and "error" in item]
    assert len(errors) == 1
    assert "backend exploded" in errors[0]["error"]["message"]
    assert "[DONE]" not in decoded
    assert not any(
        isinstance(item, dict)
        and any(choice.get("finish_reason") for choice in item.get("choices", []))
        for item in decoded
    )
    assert not server._GPU_LOCK.locked()
    with server._STREAM_PRODUCERS_LOCK:
        assert not server._STREAM_PRODUCERS
