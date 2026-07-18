"""Regression tests for the explicit OpenAI sampling/tool contract."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from mio import server
from mio.config import TierConfig
from mio.engine import GenerationMetrics, MioEngine
from mio.webui import router as webui_router


class _Tokenizer:
    eos_token_ids = [0]
    eos_token_id = 0
    all_special_ids = [0]
    clean_up_tokenization_spaces = False

    def decode(self, token_ids, skip_special_tokens=True):
        return {7: "hello STOP ignored", 8: "greedy"}.get(token_ids[0], "")

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class _ByteFallbackTokenizer:
    eos_token_ids = [0]
    eos_token_id = 0
    all_special_ids = [0]
    clean_up_tokenization_spaces = False

    _bytes = {1: b"\xf0", 2: b"\x9f", 3: b"\x9a", 4: b"\x80", 5: b"!"}

    def decode(self, token_ids, skip_special_tokens=True):
        data = b"".join(self._bytes.get(int(token_id), b"") for token_id in token_ids)
        return data.decode("utf-8", errors="replace")

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())


def _engine() -> MioEngine:
    engine = MioEngine(
        TierConfig(
            name="test",
            target_model="unused",
            draft_model="unused",
            context_window=4096,
            max_output_tokens=64,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
    )
    engine._loaded = True
    engine._target_model = object()
    engine._draft_model = object()
    engine._tokenizer = _Tokenizer()
    engine._apply_chat_template = lambda messages, tools=None: [1, 2, 3]
    engine._prefix_cache_enabled = lambda: False
    return engine


def test_positive_temperature_uses_unbiased_target_only_sampler(monkeypatch):
    engine = _engine()
    sampler = object()
    captured = {}
    monkeypatch.setattr(engine, "_make_sampler", lambda *args: sampler)

    def baseline_stream(**kwargs):
        captured.update(kwargs)
        yield {"event": "prefill", "prefill_us": 500, "prompt_token_count": 3}
        yield {
            "event": "token",
            "token_id": 7,
            "generated_tokens": 1,
            "acceptance_ratio": 0.0,
            "cycles_completed": 1,
        }
        yield {
            "event": "summary",
            "generation_tokens": 1,
            "prompt_token_count": 3,
            "elapsed_us": 1000,
            "prefill_us": 500,
        }

    monkeypatch.setattr("mio.dflash.runtime.stream_baseline_generate", baseline_stream)
    monkeypatch.setattr(
        "mio.dflash.runtime.generate_dflash_once",
        lambda **kwargs: pytest.fail("stochastic sampling must not use greedy DFlash"),
    )

    text, metrics = engine.generate(
        [{"role": "user", "content": "x"}],
        temperature=0.7,
        top_p=0.8,
        top_k=9,
        seed=123,
        stop=["STOP"],
    )

    assert captured["sampler"] is sampler
    assert text == "hello "
    assert metrics.fallback_ar is True
    assert metrics.fallback_reason == "stochastic_sampling_requires_target_only"
    assert metrics.completion_tokens == len("hello ".encode())


def test_unspecified_temperature_keeps_fast_greedy_dflash(monkeypatch):
    engine = _engine()
    captured = {}
    monkeypatch.setattr(engine, "_make_sampler", lambda *args: object())
    monkeypatch.setattr(
        "mio.dflash.runtime.generate_baseline_once",
        lambda **kwargs: pytest.fail("default generation must retain DFlash"),
    )

    def dflash(**kwargs):
        captured.update(kwargs)
        return {
            "generated_token_ids": [8],
            "generation_tokens": 1,
            "prompt_token_count": 3,
            "elapsed_us": 1000,
            "prefill_us": 500,
        }

    monkeypatch.setattr("mio.dflash.runtime.generate_dflash_once", dflash)

    text, metrics = engine.generate([{"role": "user", "content": "x"}])

    assert text == "greedy"
    assert metrics.fallback_reason is None
    assert captured["draft_model"] is engine._draft_model


def test_auto_tools_do_not_suppress_eos_but_required_tools_relax_in_stream(monkeypatch):
    engine = _engine()
    captured_once: list[dict] = []
    captured_stream: dict = {}
    monkeypatch.setattr(engine, "_make_sampler", lambda *args: object())

    def dflash(**kwargs):
        captured_once.append(kwargs)
        return {
            "generated_token_ids": [8],
            "generation_tokens": 1,
            "prompt_token_count": 3,
            "elapsed_us": 1000,
            "prefill_us": 500,
        }

    def dflash_stream(**kwargs):
        captured_stream.update(kwargs)
        yield {"event": "prefill", "prefill_us": 500, "prompt_token_count": 3}
        yield {
            "event": "summary",
            "generated_token_ids": [],
            "generation_tokens": 0,
            "prompt_token_count": 3,
            "elapsed_us": 1000,
            "prefill_us": 500,
        }

    class Detokenizer:
        last_segment = ""

        def add_token(self, _token):
            pass

        def finalize(self):
            pass

    monkeypatch.setattr("mio.dflash.runtime.generate_dflash_once", dflash)
    monkeypatch.setattr("mio.dflash.runtime.stream_dflash_generate", dflash_stream)
    engine._new_streaming_detokenizer = Detokenizer
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    engine.generate([{"role": "user", "content": "x"}], tools=tools)
    engine.generate(
        [{"role": "user", "content": "x"}],
        tools=tools,
        tool_required=True,
    )

    assert captured_once[0]["suppress_token_ids"] is None
    assert captured_stream["suppress_token_ids"] == [0]
    assert captured_stream["relax_suppress_after"] == 40
    assert captured_stream["relax_suppress_token_ids"] == [0]


def test_stream_stop_never_leaks_split_stop_sequence():
    engine = _engine()
    metrics = GenerationMetrics(prompt_tokens=2, completion_tokens=20)
    engine._generate_stream_raw = lambda *args, **kwargs: iter([("before ST", None), ("OP after", None), ("", metrics)])

    chunks = list(
        engine.generate_stream(
            [{"role": "user", "content": "x"}],
            stop=["STOP"],
        )
    )

    assert "".join(text for text, item_metrics in chunks if item_metrics is None) == "before "
    assert chunks[-1][1] is metrics
    assert metrics.completion_tokens == len("before ".encode())


def test_stream_stop_closes_raw_generation_immediately():
    engine = _engine()
    metrics = GenerationMetrics(prompt_tokens=2, completion_tokens=20)
    hidden_chunks = 0

    def raw(*_args, stop_signal, **_kwargs):
        nonlocal hidden_chunks
        yield "answer STOP", None
        while not stop_signal.is_set():
            hidden_chunks += 1
            yield "hidden", None
        yield "", metrics

    engine._generate_stream_raw = raw
    chunks = list(
        engine.generate_stream(
            [{"role": "user", "content": "x"}],
            stop=["STOP"],
        )
    )

    assert "".join(text for text, item_metrics in chunks if item_metrics is None) == "answer "
    assert hidden_chunks == 0
    assert chunks[-1][1] is metrics


def test_prompt_token_count_uses_exact_rendered_tools():
    engine = _engine()
    captured = {}

    def render(messages, tools=None):
        captured["messages"] = messages
        captured["tools"] = tools
        return [1, 2, 3, 4]

    engine._apply_chat_template = render
    messages = [{"role": "user", "content": "x"}]
    tools = [{"type": "function", "function": {"name": "read"}}]

    assert engine.prompt_token_count(messages, tools=tools) == 4
    assert captured == {"messages": messages, "tools": tools}


def test_expired_stream_deadline_stops_before_decode(monkeypatch):
    engine = _engine()
    requested_events = 0

    class Detokenizer:
        last_segment = ""

        def add_token(self, _token):
            pass

        def finalize(self):
            pass

        def reset(self):
            pass

    def dflash_stream(**_kwargs):
        nonlocal requested_events
        requested_events += 1
        yield {
            "event": "prefill",
            "prefill_us": 0.001,
            "prefill_ns": 1,
            "decode_ns": 0,
            "model_total_ns": 1,
            "prompt_token_count": 3,
            "logical_prompt_tokens": 3,
            "physical_prefill_tokens": 3,
            "warm_offset": 0,
        }
        requested_events += 1
        yield {
            "event": "token",
            "token_id": 7,
            "generated_tokens": 1,
        }

    engine._new_streaming_detokenizer = Detokenizer
    monkeypatch.setattr("mio.dflash.runtime.stream_dflash_generate", dflash_stream)

    output = list(
        engine.generate_stream(
            [{"role": "user", "content": "x"}],
            deadline_monotonic=0.0,
        )
    )

    assert requested_events == 1
    assert [chunk for chunk, metrics in output if chunk] == []
    assert output[-1][1] is not None
    assert output[-1][1].completion_tokens == 0
    assert output[-1][1].prefill_ns == 1
    assert output[-1][1].decode_ns == 0
    assert output[-1][1].timing_source == "runtime_raw_ns"
    assert output[-1][1].phase_censored is True
    assert output[-1][1].deadline_hit is True


def test_deadline_after_decode_keeps_physical_token_cost_without_emitting_token(
    monkeypatch,
):
    engine = _engine()
    stop_signal = threading.Event()

    class Detokenizer:
        last_segment = ""

        def add_token(self, _token):
            raise AssertionError("deadline-censored token must not be detokenized")

        def finalize(self):
            pass

        def reset(self):
            pass

    def baseline_stream(**_kwargs):
        yield {
            "event": "prefill",
            "prefill_us": 0.001,
            "prefill_ns": 1,
            "decode_ns": 0,
            "model_total_ns": 1,
            "prompt_token_count": 3,
            "logical_prompt_tokens": 3,
            "physical_prefill_tokens": 3,
            "physical_decode_tokens": 0,
            "warm_offset": 0,
        }
        stop_signal.set()
        yield {
            "event": "token",
            "token_id": 7,
            "generated_tokens": 1,
            "physical_decode_tokens": 1,
            "prefill_ns": 1,
            "decode_ns": 2,
            "model_total_ns": 3,
            "logical_prompt_tokens": 3,
            "physical_prefill_tokens": 3,
            "warm_offset": 0,
        }

    engine._new_streaming_detokenizer = Detokenizer
    monkeypatch.setattr("mio.dflash.runtime.stream_baseline_generate", baseline_stream)

    output = list(
        engine._generate_stream_raw(
            [{"role": "user", "content": "x"}],
            temperature=0.5,
            stop_signal=stop_signal,
            decode_chunk_tokens=1,
            deadline_monotonic=0.0,
        )
    )

    assert [chunk for chunk, metrics in output if chunk] == []
    metrics = output[-1][1]
    assert metrics is not None
    assert metrics.completion_tokens == 0
    assert metrics.physical_decode_tokens == 1
    assert metrics.decode_ns == 2
    assert metrics.phase_censored is True
    assert metrics.deadline_hit is True


def test_raw_stream_detokenizes_split_utf8_without_replacement(monkeypatch):
    engine = _engine()
    engine._tokenizer = _ByteFallbackTokenizer()

    def baseline_stream(**_kwargs):
        yield {"event": "prefill", "prefill_us": 10, "prompt_token_count": 3}
        for index, token_id in enumerate([1, 2, 3, 4, 5], start=1):
            yield {
                "event": "token",
                "token_id": token_id,
                "generated_tokens": index,
                "acceptance_ratio": 0.0,
                "cycles_completed": index,
            }
        yield {
            "event": "summary",
            "generation_tokens": 5,
            "prompt_token_count": 3,
            "elapsed_us": 100,
            "prefill_us": 10,
        }

    monkeypatch.setattr("mio.dflash.runtime.stream_baseline_generate", baseline_stream)
    chunks = list(
        engine._generate_stream_raw(
            [{"role": "user", "content": "x"}],
            temperature=0.5,
            decode_chunk_tokens=1,
        )
    )
    text = "".join(chunk for chunk, metrics in chunks if metrics is None)
    assert text == "🚀!"
    assert "�" not in text


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_p", 0.0),
        ("top_p", 1.1),
        ("top_k", -1),
        ("seed", -1),
        ("stop", []),
        ("stop", [""]),
    ],
)
def test_chat_schema_rejects_invalid_sampling_instead_of_ignoring(field, value):
    payload = {"messages": [{"role": "user", "content": "x"}], field: value}
    with pytest.raises(ValidationError):
        server.ChatCompletionRequest(**payload)


def test_chat_schema_normalizes_scalar_stop_and_types_tools():
    request = server.ChatCompletionRequest(
        messages=[{"role": "user", "content": "x"}],
        stop="END",
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
    )

    assert request.stop == ["END"]
    tools, requirement = server._resolve_request_tools(request)
    assert [tool["function"]["name"] for tool in tools] == ["lookup"]
    assert requirement == "lookup"


def test_max_completion_tokens_is_honoured_and_conflicts_fail():
    request = server.ChatCompletionRequest(
        messages=[{"role": "user", "content": "x"}],
        max_completion_tokens=17,
    )
    assert request.max_tokens == 17

    same = server.ChatCompletionRequest(
        messages=[{"role": "user", "content": "x"}],
        max_tokens=9,
        max_completion_tokens=9,
    )
    assert same.max_tokens == 9

    with pytest.raises(ValidationError, match="must match"):
        server.ChatCompletionRequest(
            messages=[{"role": "user", "content": "x"}],
            max_tokens=8,
            max_completion_tokens=9,
        )


def test_tool_choice_auto_and_none_do_not_force_tool_calls(monkeypatch):
    monkeypatch.setattr(
        server,
        "apply_prompt_policy",
        lambda messages, *_args, **_kwargs: [dict(message) for message in messages],
    )
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    messages = [{"role": "user", "content": "answer if no lookup is needed"}]

    auto = server._apply_policy(messages, tools=tools)
    none = server._apply_policy(messages, tools=None)
    required = server._apply_policy(messages, tools=tools, tool_requirement="required")

    assert auto == messages
    assert none == messages
    assert "respond ONLY with a single <tool_call>" in required[-1]["content"]


def test_named_choice_must_reference_a_declared_tool():
    request = server.ChatCompletionRequest(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"type": "function", "function": {"name": "one"}}],
        tool_choice={"type": "function", "function": {"name": "two"}},
    )
    with pytest.raises(HTTPException) as exc:
        server._resolve_request_tools(request)
    assert exc.value.status_code == 400


def test_chat_endpoint_forwards_every_sampling_field_without_auto_forcing(monkeypatch):
    captured = {}

    class Engine:
        def generate(self, messages, **kwargs):
            captured.update(messages=messages, kwargs=kwargs)
            return "plain answer", GenerationMetrics(prompt_tokens=2, completion_tokens=2)

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_compact_enabled", False)
    monkeypatch.setattr(
        server,
        "apply_prompt_policy",
        lambda messages, *_args, **_kwargs: [dict(message) for message in messages],
    )
    request = server.ChatCompletionRequest(
        model="mio-small",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.4,
        top_p=0.8,
        top_k=12,
        seed=99,
        stop="END",
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice="auto",
    )

    response = asyncio.run(server.chat_completions(request))

    assert response["choices"][0]["message"]["content"] == "plain answer"
    assert captured["kwargs"] == {
        "max_tokens": None,
        "temperature": 0.4,
        "stop": ["END"],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {}},
            }
        ],
        "tool_required": False,
        "top_p": 0.8,
        "top_k": 12,
        "seed": 99,
    }
    assert "respond ONLY" not in captured["messages"][-1]["content"]


def test_required_tool_choice_fails_explicitly_when_model_emits_no_call(monkeypatch):
    class Engine:
        def generate(self, messages, **kwargs):
            return "plain answer", GenerationMetrics(prompt_tokens=2, completion_tokens=2)

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            return Engine()

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_compact_enabled", False)
    request = server.ChatCompletionRequest(
        model="mio-small",
        messages=[{"role": "user", "content": "x"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice="required",
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.chat_completions(request))
    assert exc.value.status_code == 502


def test_nonstream_finish_reason_is_length_at_requested_cap(monkeypatch):
    class Engine:
        def generate(self, messages, **kwargs):
            return "two tokens", GenerationMetrics(prompt_tokens=1, completion_tokens=2)

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            return Engine()

    monkeypatch.setattr(server, "_manager", Manager())
    monkeypatch.setattr(server, "_router", None)
    monkeypatch.setattr(server, "_compact_enabled", False)
    response = asyncio.run(
        server.chat_completions(
            server.ChatCompletionRequest(
                model="mio-small",
                messages=[{"role": "user", "content": "x"}],
                max_tokens=2,
            )
        )
    )
    assert response["choices"][0]["finish_reason"] == "length"


def test_stream_finish_reason_is_length_at_requested_cap(monkeypatch):
    monkeypatch.setattr(server, "_manager", None)

    class Engine:
        def generate_stream(self, messages, **kwargs):
            yield "done", None
            yield "", GenerationMetrics(prompt_tokens=1, completion_tokens=2)

    async def collect():
        return [
            chunk
            async for chunk in server._stream_response(
                Engine(),
                [{"role": "user", "content": "x"}],
                "chatcmpl-length",
                1,
                "mio-small",
                "small",
                max_tokens=2,
            )
        ]

    chunks = asyncio.run(collect())
    payloads = [
        item
        for item in (chunk.removeprefix("data: ").strip() for chunk in chunks if chunk.startswith("data: "))
        if item != "[DONE]"
    ]
    decoded = [server.json.loads(item) for item in payloads]
    assert decoded[-1]["choices"][0]["finish_reason"] == "length"


def test_stream_required_tool_violation_has_no_success_trailer(monkeypatch):
    monkeypatch.setattr(server, "_manager", None)

    class Engine:
        def generate_stream(self, messages, **kwargs):
            yield "plain answer", None
            yield "", GenerationMetrics(prompt_tokens=1, completion_tokens=2)

    async def collect():
        return [
            chunk
            async for chunk in server._stream_response(
                Engine(),
                [{"role": "user", "content": "x"}],
                "chatcmpl-required",
                1,
                "mio-small",
                "small",
                tools=[{"type": "function", "function": {"name": "lookup"}}],
                tool_requirement="required",
            )
        ]

    chunks = asyncio.run(collect())
    assert any("tool_choice_violation" in chunk for chunk in chunks)
    assert not any("[DONE]" in chunk for chunk in chunks)


def test_webui_default_preserves_dflash_and_config_rejects_invalid_temperature(monkeypatch):
    monkeypatch.setattr(webui_router, "_temperature", 0.0)
    monkeypatch.setattr(webui_router, "_manager", None)

    config = asyncio.run(webui_router.get_config())
    assert config["temperature"] == 0.0
    with pytest.raises(HTTPException) as exc:
        asyncio.run(webui_router.update_config({"temperature": "nan"}))
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException):
        asyncio.run(webui_router.update_config({"temperature": 2.1}))


def test_webui_html_uses_zero_not_truthy_fallback_for_temperature():
    html = (Path(__file__).parents[1] / "mio" / "webui" / "mio_ui.html").read_text(encoding="utf-8")
    assert "temperature: 0.0" in html
    assert "currentConfig.temperature ?? 0.0" in html
    assert "currentConfig.temperature || 0.6" not in html
