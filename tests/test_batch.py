"""Continuous-batch scheduling and engine metric regressions."""

from __future__ import annotations

from types import SimpleNamespace

import mlx_lm

from mio.batch import BatchRequest, process_batch
from mio.config import TierConfig
from mio.engine import GenerationMetrics, MioEngine


class _FakeEngine:
    def __init__(self):
        self.tier_config = SimpleNamespace(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_output_tokens=32,
        )
        self.calls = []

    def generate_batch(
        self,
        messages,
        *,
        max_tokens,
        temperature,
        top_p,
        top_k,
        seed,
        stop,
    ):
        self.calls.append((messages, max_tokens, temperature, top_p, top_k, seed, stop))
        return [
            (
                request[-1]["content"] + "!",
                GenerationMetrics(
                    prompt_tokens=3,
                    completion_tokens=1,
                    generation_tps=42.0,
                    metrics_scope="batch" if len(messages) > 1 else "request",
                    batch_size=len(messages),
                    fallback_reason=(
                        "stochastic_sampling_requires_target_only"
                        if temperature > 0
                        else None
                    ),
                ),
            )
            for request in messages
        ]


class _FakeManager:
    def __init__(self, engine):
        self.engine = engine

    def get_engine(self, tier):
        assert tier == "small"
        return self.engine


def _request(text: str, temperature: float | None = None) -> BatchRequest:
    return BatchRequest(
        messages=[{"role": "user", "content": text}],
        max_tokens=7,
        temperature=temperature,
    )


def test_process_batch_groups_by_sampler_and_preserves_input_order():
    engine = _FakeEngine()
    requests = [_request("one"), _request("two", 0.2), _request("three")]

    results = process_batch(requests, _FakeManager(engine), tier="small")

    assert [result.index for result in results] == [0, 1, 2]
    assert [result.text for result in results] == ["one!", "two!", "three!"]
    assert results[0].backend == "mlx-continuous"
    assert results[0].metrics_scope == "batch"
    assert results[0].generation_tps is None
    assert results[0].batch_generation_tps == 42.0
    assert results[0].batch_size == 2
    assert results[1].backend == "mlx-target-sampling"
    assert results[1].metrics_scope == "request"
    assert results[1].generation_tps == 42.0
    assert len(engine.calls) == 2
    assert [message[-1]["content"] for message in engine.calls[0][0]] == ["one", "three"]
    assert engine.calls[0][2] == 0.0
    assert engine.calls[1][2] == 0.2


def test_generate_batch_uses_mlx_batch_stats(monkeypatch):
    tier = TierConfig(
        name="test",
        target_model="unused",
        draft_model="unused",
        context_window=4096,
        max_output_tokens=64,
        temperature=0.6,
        top_p=0.9,
        top_k=10,
    )
    engine = MioEngine(tier)
    engine._loaded = True
    engine._target_model = object()
    engine._tokenizer = SimpleNamespace(
        eos_token_ids=[0],
        encode=lambda text, add_special_tokens=False: [1] * len(text.split()),
    )
    prompts = iter([[1, 2, 3], [4, 5]])
    monkeypatch.setattr(engine, "_apply_chat_template", lambda _messages: next(prompts))
    captured = {}

    def fake_batch_generate(model, tokenizer, prompt_tokens, **kwargs):
        captured.update(model=model, tokenizer=tokenizer, prompts=prompt_tokens, kwargs=kwargs)
        stats = SimpleNamespace(
            prompt_tokens=5,
            prompt_tps=100.0,
            prompt_time=0.05,
            generation_tokens=4,
            generation_tps=40.0,
            generation_time=0.1,
            peak_memory=1.25,
        )
        return SimpleNamespace(texts=["hello world", "ok"], stats=stats)

    monkeypatch.setattr(mlx_lm, "batch_generate", fake_batch_generate)
    results = engine.generate_batch(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
        max_tokens=[5, 9],
        temperature=0.4,
        top_p=0.8,
        top_k=7,
        seed=None,
        stop=[["END"], None],
        prefill_batch_size=2,
        completion_batch_size=3,
    )

    assert captured["prompts"] == [[1, 2, 3], [4, 5]]
    assert captured["kwargs"]["max_tokens"] == [5, 9]
    assert captured["kwargs"]["prefill_batch_size"] == 2
    assert captured["kwargs"]["completion_batch_size"] == 3
    assert [metrics.prompt_tokens for _, metrics in results] == [3, 2]
    assert [metrics.completion_tokens for _, metrics in results] == [2, 1]
    assert results[0][1].generation_tps == 40.0
    assert results[0][1].metrics_scope == "batch"
    assert results[0][1].batch_size == 2
    assert results[0][1].peak_memory_gb == 1.25
    assert engine.last_metrics is results[-1][1]


def test_seeded_sampling_matches_independent_singleton_calls(monkeypatch):
    tier = TierConfig(
        name="test",
        target_model="unused",
        draft_model="unused",
        context_window=4096,
        max_output_tokens=64,
        temperature=0.6,
    )
    engine = MioEngine(tier)
    engine._loaded = True
    calls: list[tuple[str, int | None, int | None]] = []

    def fake_generate(messages, *, max_tokens, seed, **_kwargs):
        text = messages[-1]["content"]
        calls.append((text, max_tokens, seed))
        return f"{text}:{seed}", GenerationMetrics(completion_tokens=1)

    monkeypatch.setattr(engine, "generate", fake_generate)
    monkeypatch.setattr(
        mlx_lm,
        "batch_generate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("seeded sampling must not share the batch RNG")
        ),
    )

    results = engine.generate_batch(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
        max_tokens=[5, 9],
        temperature=0.4,
        seed=42,
        stop=[None, None],
    )

    assert [text for text, _metrics in results] == ["a:42", "b:42"]
    assert calls == [("a", 5, 42), ("b", 9, 42)]
