"""Auditable MLX-LM generation adapter for public effort experiments.

The adapter deliberately accepts only the runner's public case and public
feedback types.  Model loading and MLX imports are lazy, so importing this
module (and running its unit tests) never loads model weights.  Hidden tests,
hidden evaluators, and terminal scores have no representation here.

Timing is measured at the adapter boundary.  ``prefill_seconds`` is wall time
from starting the MLX stream through the first yielded token; it therefore
includes first-token sampling.  ``decode_seconds`` is wall time after that
first token until the stream is stopped.  Tokenization, prompt construction,
hashing, and audit work are reported as ``other_seconds``.  This split is
reproducible and honest about what a synchronous Python stream can observe; it
does not pretend to be a kernel-only profiler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from experimental.effort.humaneval import PublicHumanEvalCase
from experimental.effort.markov_runner import (
    GeneratedCandidate,
    PublicGenerationFeedback,
)
from experimental.markov_effort_controller import (
    ControllerAction,
    GenerationMetrics,
)


PROMPT_REVISION = "mio-public-humaneval-effort-v1"
TIMING_METHOD = "adapter-wall-ttft-split-v1"
UNCERTAINTY_METHOD_NATIVE = "strided-native-selected-token-surprisal-finish-penalty-v1"
UNCERTAINTY_METHOD_RENORMALIZED = "strided-fp32-renormalized-selected-token-surprisal-finish-penalty-v2"
_SYSTEM_PROMPT = (
    "You are solving a public Python programming task. Use only the task and "
    "public validator feedback below. Return Python code only: either the full "
    "entry-point function or the exact continuation of the supplied function. "
    "Do not use Markdown fences or explanatory prose."
)


@dataclass(frozen=True)
class MLXSamplerSettings:
    """Frozen sampler and prompt settings for one experiment condition."""

    temperature: float = 0.0
    top_p: float = 1.0
    min_p: float = 0.0
    top_k: int = 0
    prefill_step_size: int = 2048
    use_chat_template: bool = True
    enable_thinking: bool = False
    # Native selected-token lookup is cheap enough to preserve the v1 every-
    # token baseline.  FP32 renormalization experiments should explicitly use
    # a wider stride (the preregistered pilot uses 8).
    uncertainty_logprob_stride: int = 1
    # Opt-in until the benchmark quantifies its extra full-vocabulary
    # logsumexp cost against the native bfloat16 signal.
    renormalize_uncertainty_logprobs: bool = False

    def __post_init__(self) -> None:
        numeric = (self.temperature, self.top_p, self.min_p)
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("sampler values must be finite numbers")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if (
            isinstance(self.prefill_step_size, bool)
            or not isinstance(self.prefill_step_size, int)
            or self.prefill_step_size < 1
        ):
            raise ValueError("prefill_step_size must be a positive integer")
        if (
            isinstance(self.uncertainty_logprob_stride, bool)
            or not isinstance(self.uncertainty_logprob_stride, int)
            or self.uncertainty_logprob_stride < 1
        ):
            raise ValueError("uncertainty_logprob_stride must be a positive integer")
        switches = (
            self.use_chat_template,
            self.enable_thinking,
            self.renormalize_uncertainty_logprobs,
        )
        if any(type(value) is not bool for value in switches):
            raise TypeError("prompt and uncertainty switches must be bool values")

    @property
    def deterministic(self) -> bool:
        """Whether sampling is greedy and independent of the RNG stream."""

        return self.temperature == 0.0


@dataclass(frozen=True)
class MLXGenerationAuditRecord:
    """Hash-addressed public record for one adapter invocation."""

    task_id: str
    action: ControllerAction
    model_id: str
    prompt_revision: str
    seed: int
    sampler: Mapping[str, object]
    deterministic_sampler: bool
    max_output_tokens: int
    allocated_e2e_seconds: float | None
    prompt_source_sha256: str
    prompt_token_ids_sha256: str
    output_text_sha256: str
    output_token_ids_sha256: str
    prompt_tokens: int
    output_tokens: int
    backend_reported_prompt_tokens: int | None
    backend_reported_generation_tokens: int | None
    ttft_seconds: float | None
    prefill_seconds: float
    decode_seconds: float
    other_seconds: float
    timing_method: str
    finish_reason: str
    deadline_exceeded: bool
    peak_memory_bytes: int | None
    peak_memory_scope: str
    selected_logprob_tokens: int
    uncertainty_logprob_stride: int
    uncertainty_logprobs_renormalized: bool
    uncertainty_scoring_seconds: float
    mean_selected_token_surprisal: float | None
    raw_uncertainty: float
    uncertainty_method: str

    def __post_init__(self) -> None:
        digests = (
            self.prompt_source_sha256,
            self.prompt_token_ids_sha256,
            self.output_text_sha256,
            self.output_token_ids_sha256,
        )
        if any(len(digest) != 64 for digest in digests):
            raise ValueError("audit digests must be SHA-256 hex strings")
        if not 0.0 <= self.raw_uncertainty <= 1.0:
            raise ValueError("raw uncertainty must be in [0, 1]")
        if self.uncertainty_scoring_seconds < 0.0 or not math.isfinite(self.uncertainty_scoring_seconds):
            raise ValueError("uncertainty scoring time must be finite and non-negative")


class MLXRuntime(Protocol):
    """Small injectable boundary around MLX-LM's process-global runtime."""

    def reset_peak_memory(self) -> bool: ...

    def peak_memory_bytes(self) -> int | None: ...

    def make_sampler(self, settings: MLXSamplerSettings, seed: int) -> object: ...

    def stream_generate(
        self,
        *,
        model: object,
        tokenizer: object,
        prompt_token_ids: Sequence[int],
        max_output_tokens: int,
        sampler: object,
        prefill_step_size: int,
    ) -> Iterable[object]: ...


class _DefaultMLXRuntime:
    """Lazy bridge to the installed ``mlx`` and ``mlx_lm`` packages."""

    def reset_peak_memory(self) -> bool:
        import mlx.core as mx

        reset = getattr(mx, "reset_peak_memory", None)
        if reset is None:
            return False
        reset()
        return True

    def peak_memory_bytes(self) -> int | None:
        import mlx.core as mx

        getter = getattr(mx, "get_peak_memory", None)
        if getter is None:
            return None
        value = float(getter())
        if not math.isfinite(value) or value < 0.0:
            return None
        return int(round(value))

    def make_sampler(self, settings: MLXSamplerSettings, seed: int) -> object:
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        return make_sampler(
            temp=settings.temperature,
            top_p=settings.top_p,
            min_p=settings.min_p,
            top_k=settings.top_k,
        )

    def stream_generate(
        self,
        *,
        model: object,
        tokenizer: object,
        prompt_token_ids: Sequence[int],
        max_output_tokens: int,
        sampler: object,
        prefill_step_size: int,
    ) -> Iterable[object]:
        from mlx_lm import stream_generate

        return stream_generate(
            model,
            tokenizer,
            list(prompt_token_ids),
            max_tokens=max_output_tokens,
            sampler=sampler,
            prefill_step_size=prefill_step_size,
        )


def public_prompt_messages(
    case: PublicHumanEvalCase,
    feedback: PublicGenerationFeedback,
) -> tuple[dict[str, str], ...]:
    """Build an action-specific prompt solely from public-channel values."""

    if type(case) is not PublicHumanEvalCase:
        raise TypeError("MLX effort generation accepts only PublicHumanEvalCase")
    if type(feedback) is not PublicGenerationFeedback:
        raise TypeError("MLX effort generation accepts only PublicGenerationFeedback")

    task = (
        f"Public task id: {case.task_id}\n"
        f"Required entry point: {case.entry_point}\n"
        "<public_task>\n"
        f"{case.prompt}"
        "\n</public_task>"
    )
    if feedback.action is ControllerAction.GENERATE_DIRECT:
        instruction = "Produce the strongest direct implementation."
    else:
        parent = feedback.parent_completion
        status = feedback.validator_status
        if parent is None or status is None:
            raise ValueError("non-direct generation requires public parent evidence")
        evidence = (
            "\n<public_parent_candidate>\n"
            f"{parent}"
            "\n</public_parent_candidate>\n"
            f"Public validator status: {status}\n"
            "<public_validator_feedback>\n"
            f"{feedback.validator_feedback}"
            "\n</public_validator_feedback>"
        )
        if feedback.action is ControllerAction.GENERATE_REPAIR:
            instruction = (
                "Repair the parent candidate using the public validator evidence. "
                "Return a complete corrected candidate."
            )
        elif feedback.action is ControllerAction.GENERATE_ALTERNATIVE:
            instruction = (
                "Produce an independent alternative implementation. Do not merely rephrase the parent candidate."
            )
        elif feedback.action is ControllerAction.GENERATE_REFINE:
            instruction = "Refine the parent candidate for correctness and edge cases while preserving any sound parts."
        else:  # pragma: no cover - dataclass and enum guard this upstream
            raise ValueError(f"unsupported generation action: {feedback.action}")
        task += evidence

    return (
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"{task}\n\n{instruction}"},
    )


def _prompt_source(messages: tuple[dict[str, str], ...]) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _coerce_token_ids(value: object) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("tokenizer must return a one-dimensional token sequence")
    result: list[int] = []
    for token in value:
        if hasattr(token, "item"):
            token = token.item()
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise TypeError("token ids must be non-negative integers")
        result.append(token)
    if not result:
        raise ValueError("tokenized effort prompt must not be empty")
    return tuple(result)


def _tokenize_prompt(
    tokenizer: object,
    messages: tuple[dict[str, str], ...],
    settings: MLXSamplerSettings,
) -> tuple[int, ...]:
    if settings.use_chat_template:
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_template is None:
            raise TypeError("chat-template mode requires tokenizer.apply_chat_template")
        encoded = apply_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=settings.enable_thinking,
        )
    else:
        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            raise TypeError("raw-prompt mode requires tokenizer.encode")
        rendered = (
            "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
            + "\n\nASSISTANT:\n"
        )
        encoded = encode(rendered, add_special_tokens=True)
    return _coerce_token_ids(encoded)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    canonical = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"mio-token-ids-v1\0" + canonical).hexdigest()


def _event_token_id(event: object) -> int:
    value = getattr(event, "token")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("MLX stream token must be a non-negative integer")
    return value


def _renormalized_selected_log_probability(
    logprobs: object,
    token_id: int,
) -> float | None:
    """Re-normalize a sampled log-probability vector in float32.

    MLX-LM may expose bfloat16 log probabilities.  For a confident token the
    selected value can round to exactly zero even though the runner-up mass is
    non-zero.  A float32 logsumexp over the already-produced vector recovers
    that residual mass without another model forward.  This helper is called
    only at the configured stride.
    """

    if isinstance(logprobs, (list, tuple)):
        try:
            values = [float(value) for value in logprobs]
        except (TypeError, ValueError):
            return None
        if token_id >= len(values) or not values or any(not math.isfinite(value) for value in values):
            return None
        maximum = max(values)
        normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in values))
        return min(0.0, values[token_id] - normalizer)

    # Keep MLX lazy at module import and avoid copying a vocabulary-sized
    # vector to the CPU.  Both scalar values are materialized in one eval.
    try:
        import mlx.core as mx

        if not isinstance(logprobs, mx.array):
            return None
        float_logprobs = logprobs.astype(mx.float32)
        selected = float_logprobs[token_id]
        normalizer = mx.logsumexp(float_logprobs)
        mx.eval(selected, normalizer)
        result = float(selected.item()) - float(normalizer.item())
    except (IndexError, TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return min(0.0, result)


def _selected_log_probability(
    event: object,
    token_id: int,
    *,
    renormalize: bool,
) -> float | None:
    logprobs = getattr(event, "logprobs", None)
    if logprobs is None:
        return None
    if renormalize:
        return _renormalized_selected_log_probability(logprobs, token_id)
    try:
        value = logprobs[token_id]
        if hasattr(value, "item"):
            value = value.item()
        result = float(value)
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    # Normalized log probabilities should be <= 0.  Clamp tiny positive
    # numerical noise, but reject values that cannot represent a probability.
    if result > 1e-6:
        return None
    return min(0.0, result)


def raw_uncertainty_from_surprisal(
    selected_log_probabilities: Sequence[float],
    finish_reason: str,
) -> tuple[float, float | None]:
    """Return an uncalibrated error signal and its mean token surprisal.

    ``1 - exp(-mean_surprisal)`` is one minus the geometric mean selected-token
    probability.  A length stop has a conservative 0.75 floor and a deadline
    stop is maximally uncertain.  These constants are frozen experiment
    heuristics, not probabilities of task failure; calibration is mandatory in
    the runner before routing on the value.
    """

    usable = [
        -float(log_probability)
        for log_probability in selected_log_probabilities
        if math.isfinite(float(log_probability)) and float(log_probability) <= 0.0
    ]
    mean_surprisal = sum(usable) / len(usable) if usable else None
    uncertainty = 1.0 if mean_surprisal is None else -math.expm1(-mean_surprisal)
    if finish_reason == "length":
        uncertainty = max(0.75, uncertainty)
    elif finish_reason == "deadline":
        uncertainty = 1.0
    return min(1.0, max(0.0, uncertainty)), mean_surprisal


def _optional_non_negative_int(value: object) -> int | None:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _event_peak_memory_bytes(event: object) -> int | None:
    value = getattr(event, "peak_memory", None)
    if value is None:
        return None
    try:
        gigabytes = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(gigabytes) or gigabytes < 0.0:
        return None
    return int(round(gigabytes * 1_000_000_000))


def _elapsed(end: float, start: float) -> float:
    value = end - start
    if value < 0.0 or not math.isfinite(value):
        raise RuntimeError("MLX adapter clock must be finite and monotonic")
    return value


class MLXEffortGenerator:
    """Callable MLX-LM adapter for :func:`run_markov_effort`.

    The loaded model and tokenizer are opaque objects.  Use
    :meth:`from_pretrained` for normal MLX-LM loading or inject a small runtime
    in unit tests.  Audit records are append-only for the lifetime of the
    adapter instance and contain hashes rather than prompt/candidate contents.
    """

    def __init__(
        self,
        model: object,
        tokenizer: object,
        *,
        model_id: str,
        settings: MLXSamplerSettings | None = None,
        runtime: MLXRuntime | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        self._model = model
        self._tokenizer = tokenizer
        self.model_id = model_id
        self.settings = settings or MLXSamplerSettings()
        self._runtime = runtime or _DefaultMLXRuntime()
        self._clock = clock
        self._audit_records: list[MLXGenerationAuditRecord] = []

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        audited_model_id: str | None = None,
        adapter_path: str | None = None,
        lazy: bool = False,
        settings: MLXSamplerSettings | None = None,
    ) -> MLXEffortGenerator:
        """Load weights lazily with MLX-LM only when explicitly requested."""

        from mlx_lm.utils import load

        model, tokenizer = load(
            model_id,
            revision=revision,
            adapter_path=adapter_path,
            lazy=lazy,
        )
        resolved_audit_id = (
            audited_model_id
            if audited_model_id is not None
            else (f"{model_id}@{revision}" if revision else model_id)
        )
        return cls(
            model,
            tokenizer,
            model_id=resolved_audit_id,
            settings=settings,
        )

    @property
    def audit_records(self) -> tuple[MLXGenerationAuditRecord, ...]:
        return tuple(self._audit_records)

    def __call__(
        self,
        case: PublicHumanEvalCase,
        feedback: PublicGenerationFeedback,
        /,
    ) -> GeneratedCandidate:
        if type(case) is not PublicHumanEvalCase:
            raise TypeError("MLX effort generation accepts only PublicHumanEvalCase")
        if type(feedback) is not PublicGenerationFeedback:
            raise TypeError("MLX effort generation accepts only PublicGenerationFeedback")

        call_started = self._clock()
        messages = public_prompt_messages(case, feedback)
        prompt_source = _prompt_source(messages)
        prompt_token_ids = _tokenize_prompt(self._tokenizer, messages, self.settings)
        memory_was_reset = self._runtime.reset_peak_memory()
        sampler = self._runtime.make_sampler(self.settings, feedback.seed)
        generation_started = self._clock()
        allocation = feedback.max_additional_e2e_seconds
        deadline = call_started + allocation if allocation is not None else None
        token_ids: list[int] = []
        text_segments: list[str] = []
        selected_logprobs: list[float] = []
        uncertainty_scoring_seconds = 0.0
        first_token_at: float | None = None
        finish_reason = "stream_exhausted"
        event_peak_memory_bytes: int | None = None
        backend_prompt_tokens: int | None = None
        backend_generation_tokens: int | None = None
        iterator = None

        if deadline is not None and generation_started >= deadline:
            finish_reason = "deadline"
        else:
            stream = self._runtime.stream_generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt_token_ids=prompt_token_ids,
                max_output_tokens=feedback.max_output_tokens,
                sampler=sampler,
                prefill_step_size=self.settings.prefill_step_size,
            )
            iterator = iter(stream)
            try:
                while len(token_ids) < feedback.max_output_tokens:
                    if deadline is not None and self._clock() >= deadline:
                        finish_reason = "deadline"
                        break
                    try:
                        event = next(iterator)
                    except StopIteration:
                        break
                    observed_at = self._clock()
                    if first_token_at is None:
                        first_token_at = observed_at

                    token_id = _event_token_id(event)
                    text = getattr(event, "text", None)
                    if not isinstance(text, str):
                        raise TypeError("MLX stream text segment must be a string")
                    token_ids.append(token_id)
                    text_segments.append(text)
                    sample_index = len(token_ids) - 1
                    if sample_index % self.settings.uncertainty_logprob_stride == 0:
                        uncertainty_started = self._clock()
                        selected = _selected_log_probability(
                            event,
                            token_id,
                            renormalize=self.settings.renormalize_uncertainty_logprobs,
                        )
                        uncertainty_scoring_seconds += _elapsed(
                            self._clock(),
                            uncertainty_started,
                        )
                        if selected is not None:
                            selected_logprobs.append(selected)

                    event_memory = _event_peak_memory_bytes(event)
                    if event_memory is not None:
                        event_peak_memory_bytes = max(
                            event_peak_memory_bytes or 0,
                            event_memory,
                        )
                    reported_prompt = _optional_non_negative_int(getattr(event, "prompt_tokens", None))
                    reported_generation = _optional_non_negative_int(getattr(event, "generation_tokens", None))
                    if reported_prompt is not None:
                        backend_prompt_tokens = reported_prompt
                    if reported_generation is not None:
                        backend_generation_tokens = reported_generation

                    reported_finish = getattr(event, "finish_reason", None)
                    if deadline is not None and observed_at >= deadline:
                        finish_reason = "deadline"
                        break
                    if reported_finish is not None:
                        finish_reason = str(reported_finish)
                        break
                else:
                    finish_reason = "length"
            finally:
                # ``stream_generate`` yields its terminal record before the
                # generator frame returns.  Close even a normal ``stop`` so
                # caches and generator-local references are released now,
                # and also close on malformed backend events/exceptions.
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

        generation_finished = self._clock()
        process_peak_memory = self._runtime.peak_memory_bytes()
        peak_values = [value for value in (event_peak_memory_bytes, process_peak_memory) if value is not None]
        peak_memory_bytes = max(peak_values) if peak_values else None
        if peak_memory_bytes is None:
            peak_memory_scope = "unavailable"
        elif memory_was_reset:
            peak_memory_scope = "generation_reset"
        else:
            peak_memory_scope = "process_lifetime"

        completion = "".join(text_segments)
        raw_uncertainty, mean_surprisal = raw_uncertainty_from_surprisal(
            selected_logprobs,
            finish_reason,
        )
        output_text_sha256 = _sha256_text(completion)
        output_tokens_sha256 = _token_ids_sha256(token_ids)
        prompt_tokens_sha256 = _token_ids_sha256(prompt_token_ids)
        finalization_finished = self._clock()

        if first_token_at is None:
            prefill_seconds = _elapsed(generation_finished, generation_started)
            decode_seconds = 0.0
            ttft_seconds = None
        else:
            prefill_seconds = _elapsed(first_token_at, generation_started)
            decode_seconds = _elapsed(generation_finished, first_token_at)
            ttft_seconds = _elapsed(first_token_at, call_started)
        other_seconds = _elapsed(generation_started, call_started) + _elapsed(
            finalization_finished,
            generation_finished,
        )
        metrics = GenerationMetrics(
            prompt_tokens=len(prompt_token_ids),
            output_tokens=len(token_ids),
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            other_seconds=other_seconds,
        )
        deadline_exceeded = bool(allocation is not None and _elapsed(finalization_finished, call_started) > allocation)
        record = MLXGenerationAuditRecord(
            task_id=case.task_id,
            action=feedback.action,
            model_id=self.model_id,
            prompt_revision=PROMPT_REVISION,
            seed=feedback.seed,
            sampler=asdict(self.settings),
            deterministic_sampler=self.settings.deterministic,
            max_output_tokens=feedback.max_output_tokens,
            allocated_e2e_seconds=allocation,
            prompt_source_sha256=_sha256_text(prompt_source),
            prompt_token_ids_sha256=prompt_tokens_sha256,
            output_text_sha256=output_text_sha256,
            output_token_ids_sha256=output_tokens_sha256,
            prompt_tokens=len(prompt_token_ids),
            output_tokens=len(token_ids),
            backend_reported_prompt_tokens=backend_prompt_tokens,
            backend_reported_generation_tokens=backend_generation_tokens,
            ttft_seconds=ttft_seconds,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            other_seconds=other_seconds,
            timing_method=TIMING_METHOD,
            finish_reason=finish_reason,
            deadline_exceeded=deadline_exceeded,
            peak_memory_bytes=peak_memory_bytes,
            peak_memory_scope=peak_memory_scope,
            selected_logprob_tokens=len(selected_logprobs),
            uncertainty_logprob_stride=self.settings.uncertainty_logprob_stride,
            uncertainty_logprobs_renormalized=(self.settings.renormalize_uncertainty_logprobs),
            uncertainty_scoring_seconds=uncertainty_scoring_seconds,
            mean_selected_token_surprisal=mean_surprisal,
            raw_uncertainty=raw_uncertainty,
            uncertainty_method=(
                UNCERTAINTY_METHOD_RENORMALIZED
                if self.settings.renormalize_uncertainty_logprobs
                else UNCERTAINTY_METHOD_NATIVE
            ),
        )
        self._audit_records.append(record)
        return GeneratedCandidate(
            completion=completion,
            metrics=metrics,
            raw_uncertainty=raw_uncertainty,
        )
