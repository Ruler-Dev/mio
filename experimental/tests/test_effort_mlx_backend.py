from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from experimental.effort.humaneval import (
    HumanEvalCase,
    PublicHumanEvalCase,
)
from experimental.effort.markov_runner import PublicGenerationFeedback
from experimental.effort.mlx_backend import (
    MLXEffortGenerator,
    MLXSamplerSettings,
    public_prompt_messages,
    raw_uncertainty_from_surprisal,
)
from experimental.markov_effort_controller import ControllerAction


PUBLIC_CASE = PublicHumanEvalCase(
    task_id="HumanEval/0",
    prompt='def add(a, b):\n    """Return a + b."""\n',
    entry_point="add",
)


def _feedback(
    action: ControllerAction,
    *,
    max_output_tokens: int = 8,
    deadline: float | None = None,
) -> PublicGenerationFeedback:
    direct = action is ControllerAction.GENERATE_DIRECT
    return PublicGenerationFeedback(
        action=action,
        parent_node_id=None if direct else 0,
        parent_completion=None if direct else "    return a - b",
        validator_status=None if direct else "format_error",
        validator_feedback="" if direct else "candidate is not parseable",
        max_output_tokens=max_output_tokens,
        max_additional_e2e_seconds=deadline,
        seed=1729,
    )


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [101, 102, 103]


@dataclass
class _Event:
    text: str
    token: int
    logprobs: list[float]
    prompt_tokens: int
    generation_tokens: int
    peak_memory: float
    finish_reason: str | None = None


class _ClosableEvents:
    def __init__(self, events: list[_Event]) -> None:
        self._events = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._events)

    def close(self) -> None:
        self.closed = True


class _Runtime:
    def __init__(
        self,
        events: list[_Event],
        *,
        reset_memory: bool = True,
        process_peak_memory: int | None = 1_400_000_000,
    ) -> None:
        self.events = events
        self.reset_memory = reset_memory
        self.process_peak_memory = process_peak_memory
        self.stream: _ClosableEvents | None = None
        self.sampler_calls: list[tuple[MLXSamplerSettings, int]] = []
        self.stream_calls: list[dict[str, object]] = []

    def reset_peak_memory(self) -> bool:
        return self.reset_memory

    def peak_memory_bytes(self) -> int | None:
        return self.process_peak_memory

    def make_sampler(self, settings: MLXSamplerSettings, seed: int) -> object:
        self.sampler_calls.append((settings, seed))
        return ("frozen-sampler", seed)

    def stream_generate(self, **kwargs):
        self.stream_calls.append(kwargs)
        self.stream = _ClosableEvents(list(self.events))
        return self.stream


class _Clock:
    def __init__(self, values: list[float] | None = None) -> None:
        self._values = iter(values) if values is not None else None
        self._next = 0.0

    def __call__(self) -> float:
        if self._values is not None:
            return next(self._values)
        value = self._next
        self._next += 0.01
        return value


def _logprobs(token: int, selected: float) -> list[float]:
    values = [-100.0] * max(token + 1, 2)
    values[token] = selected
    remaining_probability = 1.0 - math.exp(selected)
    runner_up = 0 if token != 0 else 1
    values[runner_up] = math.log(remaining_probability)
    return values


def test_all_generation_actions_have_distinct_public_only_prompts() -> None:
    prompts: dict[ControllerAction, str] = {}
    for action in (
        ControllerAction.GENERATE_DIRECT,
        ControllerAction.GENERATE_REPAIR,
        ControllerAction.GENERATE_ALTERNATIVE,
        ControllerAction.GENERATE_REFINE,
    ):
        messages = public_prompt_messages(PUBLIC_CASE, _feedback(action))
        assert [message["role"] for message in messages] == ["system", "user"]
        prompts[action] = messages[-1]["content"]

    assert "strongest direct" in prompts[ControllerAction.GENERATE_DIRECT]
    assert "public_parent_candidate" not in prompts[ControllerAction.GENERATE_DIRECT]
    assert "Repair the parent" in prompts[ControllerAction.GENERATE_REPAIR]
    assert "independent alternative" in prompts[ControllerAction.GENERATE_ALTERNATIVE]
    assert "Refine the parent" in prompts[ControllerAction.GENERATE_REFINE]
    for action in (
        ControllerAction.GENERATE_REPAIR,
        ControllerAction.GENERATE_ALTERNATIVE,
        ControllerAction.GENERATE_REFINE,
    ):
        assert "return a - b" in prompts[action]
        assert "format_error" in prompts[action]
        assert "candidate is not parseable" in prompts[action]
    assert len(set(prompts.values())) == 4


def test_generator_returns_honest_split_metrics_uncertainty_and_audit_hashes() -> None:
    tokenizer = _Tokenizer()
    runtime = _Runtime(
        [
            _Event("    return ", 4, _logprobs(4, -0.1), 3, 1, 1.2),
            _Event("a + b", 5, _logprobs(5, -0.2), 3, 2, 1.5, "stop"),
        ]
    )
    generator = MLXEffortGenerator(
        object(),
        tokenizer,
        model_id="fake/model@frozen",
        settings=MLXSamplerSettings(
            uncertainty_logprob_stride=1,
            renormalize_uncertainty_logprobs=True,
        ),
        runtime=runtime,
        clock=_Clock(),
    )

    result = generator(PUBLIC_CASE, _feedback(ControllerAction.GENERATE_DIRECT))

    assert result.completion == "    return a + b"
    assert result.metrics.prompt_tokens == 3
    assert result.metrics.output_tokens == 2
    assert result.metrics.prefill_seconds == pytest.approx(0.01)
    assert result.metrics.decode_seconds == pytest.approx(0.06)
    assert result.metrics.other_seconds == pytest.approx(0.02)
    expected_uncertainty = 1.0 - math.exp(-0.15)
    assert result.raw_uncertainty == pytest.approx(expected_uncertainty)

    assert len(generator.audit_records) == 1
    record = generator.audit_records[0]
    assert record.model_id == "fake/model@frozen"
    assert record.seed == 1729
    assert record.deterministic_sampler is True
    assert record.finish_reason == "stop"
    assert record.ttft_seconds == pytest.approx(0.02)
    assert record.peak_memory_bytes == 1_500_000_000
    assert record.peak_memory_scope == "generation_reset"
    assert record.selected_logprob_tokens == 2
    assert record.uncertainty_logprob_stride == 1
    assert record.uncertainty_logprobs_renormalized is True
    assert record.uncertainty_scoring_seconds == pytest.approx(0.02)
    assert record.mean_selected_token_surprisal == pytest.approx(0.15)
    assert record.backend_reported_prompt_tokens == 3
    assert record.backend_reported_generation_tokens == 2
    assert len(record.prompt_source_sha256) == 64
    assert len(record.prompt_token_ids_sha256) == 64
    assert len(record.output_text_sha256) == 64
    assert len(record.output_token_ids_sha256) == 64

    assert runtime.sampler_calls == [(generator.settings, 1729)]
    assert runtime.stream is not None and runtime.stream.closed is True
    assert runtime.stream_calls[0]["prompt_token_ids"] == (101, 102, 103)
    assert runtime.stream_calls[0]["max_output_tokens"] == 8
    _, template_kwargs = tokenizer.calls[0]
    assert template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_deadline_stops_stream_and_marks_length_like_output_maximally_uncertain() -> None:
    runtime = _Runtime(
        [
            _Event("one", 1, _logprobs(1, -0.01), 3, 1, 0.5),
            _Event("two", 2, _logprobs(2, -0.01), 3, 2, 0.6),
            _Event("three", 3, _logprobs(3, -0.01), 3, 3, 0.7, "stop"),
        ]
    )
    clock = _Clock([0.00, 0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.13, 0.14, 0.15])
    generator = MLXEffortGenerator(
        object(),
        _Tokenizer(),
        model_id="fake/model",
        settings=MLXSamplerSettings(uncertainty_logprob_stride=8),
        runtime=runtime,
        clock=clock,
    )

    result = generator(
        PUBLIC_CASE,
        _feedback(
            ControllerAction.GENERATE_DIRECT,
            max_output_tokens=8,
            deadline=0.12,
        ),
    )

    assert result.completion == "onetwo"
    assert result.metrics.output_tokens == 2
    assert result.raw_uncertainty == 1.0
    assert runtime.stream is not None and runtime.stream.closed is True
    record = generator.audit_records[0]
    assert record.finish_reason == "deadline"
    assert record.deadline_exceeded is True
    assert record.allocated_e2e_seconds == 0.12


def test_max_output_tokens_is_defensively_enforced_and_closes_stream() -> None:
    runtime = _Runtime(
        [
            _Event("one", 1, _logprobs(1, -0.1), 3, 1, 0.2),
            _Event("two", 2, _logprobs(2, -0.1), 3, 2, 0.3),
        ]
    )
    generator = MLXEffortGenerator(
        object(),
        _Tokenizer(),
        model_id="fake/model",
        runtime=runtime,
        clock=_Clock(),
    )

    result = generator(
        PUBLIC_CASE,
        _feedback(ControllerAction.GENERATE_DIRECT, max_output_tokens=1),
    )

    assert result.completion == "one"
    assert result.metrics.output_tokens == 1
    assert result.raw_uncertainty == pytest.approx(0.75)
    assert runtime.stream is not None and runtime.stream.closed is True
    assert generator.audit_records[0].finish_reason == "length"


def test_generator_rejects_full_case_before_touching_runtime() -> None:
    runtime = _Runtime([])
    generator = MLXEffortGenerator(
        object(),
        _Tokenizer(),
        model_id="fake/model",
        runtime=runtime,
        clock=_Clock(),
    )
    full_case = HumanEvalCase(
        task_id=PUBLIC_CASE.task_id,
        prompt=PUBLIC_CASE.prompt,
        test="def check(candidate):\n    assert candidate(1, 2) == 3\n",
        entry_point=PUBLIC_CASE.entry_point,
    )

    with pytest.raises(TypeError, match="only PublicHumanEvalCase"):
        generator(full_case, _feedback(ControllerAction.GENERATE_DIRECT))  # type: ignore[arg-type]

    assert runtime.sampler_calls == []
    assert runtime.stream_calls == []
    assert generator.audit_records == ()


def test_optional_fp32_renormalization_recovers_bfloat16_zero_collapse() -> None:
    event = _Event(
        "return 1",
        0,
        [0.0, -2.0],
        prompt_tokens=3,
        generation_tokens=1,
        peak_memory=0.5,
        finish_reason="stop",
    )

    def execute(renormalize: bool):
        generator = MLXEffortGenerator(
            object(),
            _Tokenizer(),
            model_id="fake/model",
            settings=MLXSamplerSettings(
                uncertainty_logprob_stride=1,
                renormalize_uncertainty_logprobs=renormalize,
            ),
            runtime=_Runtime([event]),
            clock=_Clock(),
        )
        result = generator(
            PUBLIC_CASE,
            _feedback(ControllerAction.GENERATE_DIRECT),
        )
        return result.raw_uncertainty, generator.audit_records[0]

    native_uncertainty, native_record = execute(False)
    renormalized_uncertainty, renormalized_record = execute(True)

    assert native_uncertainty == 0.0
    assert renormalized_uncertainty == pytest.approx(1.0 - 1.0 / (1.0 + math.exp(-2.0)))
    assert renormalized_uncertainty > native_uncertainty
    assert native_record.uncertainty_logprobs_renormalized is False
    assert renormalized_record.uncertainty_logprobs_renormalized is True
    assert "native" in native_record.uncertainty_method
    assert "fp32-renormalized" in renormalized_record.uncertainty_method
    assert renormalized_record.uncertainty_scoring_seconds >= 0.0


@pytest.mark.parametrize(
    ("logprobs", "reason", "expected"),
    [
        ([-0.1], "stop", 1.0 - math.exp(-0.1)),
        ([-0.1], "length", 0.75),
        ([-0.1], "deadline", 1.0),
        ([], "stop", 1.0),
    ],
)
def test_raw_uncertainty_is_bounded_and_explicitly_finish_aware(
    logprobs: list[float],
    reason: str,
    expected: float,
) -> None:
    uncertainty, _ = raw_uncertainty_from_surprisal(logprobs, reason)
    assert uncertainty == pytest.approx(expected)
    assert 0.0 <= uncertainty <= 1.0


def test_sampler_settings_reject_non_integral_limits() -> None:
    with pytest.raises((TypeError, ValueError)):
        MLXSamplerSettings(top_k=1.5)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        MLXSamplerSettings(prefill_step_size=2.5)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        MLXSamplerSettings(uncertainty_logprob_stride=0)
