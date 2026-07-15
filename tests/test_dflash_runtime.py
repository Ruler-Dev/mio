"""Regression tests for the DFlash/Qwen compatibility layer."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import mlx.core as mx
import pytest


def test_baseline_runtime_exposes_and_applies_sampler_contract():
    from mio.dflash import runtime

    assert "sampler" in inspect.signature(runtime.generate_baseline_once).parameters
    assert "sampler" in inspect.signature(runtime.stream_baseline_generate).parameters

    observed: list[list[float]] = []

    def sampler(logprobs: mx.array) -> mx.array:
        observed.append(logprobs.tolist()[0])
        return mx.array([1], dtype=mx.uint32)

    token = runtime.sample_tokens_with_mask(
        mx.array([[1.0, 3.0, 2.0]]),
        sampler=sampler,
    )

    assert token.tolist() == [1]
    assert len(observed) == 1
    assert max(observed[0]) <= 0.0


def test_stream_baseline_relaxes_eos_suppression_after_configured_tokens(monkeypatch):
    from mio.dflash import runtime

    assert "relax_suppress_after" in inspect.signature(runtime.stream_baseline_generate).parameters
    masks = []
    sampled = iter([1, 1, 0])
    monkeypatch.setattr(runtime, "make_target_cache", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runtime,
        "chunked_prefill",
        lambda *_args, **_kwargs: (mx.zeros((1, 1, 3)), None),
    )
    monkeypatch.setattr(
        runtime,
        "build_suppress_token_mask",
        lambda _vocab, ids: ("mask", tuple(ids or [])),
    )

    def sample(_logits, _sampler, mask):
        masks.append(mask)
        return mx.array([next(sampled)], dtype=mx.uint32)

    monkeypatch.setattr(runtime, "sample_tokens_with_mask", sample)

    class Target:
        def __call__(self, _tokens, cache):
            assert cache is not None
            return mx.zeros((1, 1, 3))

    events = list(
        runtime.stream_baseline_generate(
            target_model=Target(),
            tokenizer=object(),
            prompt="",
            prompt_tokens_override=[2],
            max_new_tokens=5,
            stop_token_ids=[0],
            suppress_token_ids=[0],
            relax_suppress_after=2,
            relax_suppress_token_ids=[0],
        )
    )

    summary = events[-1]
    assert summary["generated_token_ids"] == [1, 1, 0]
    assert masks == [("mask", (0,)), ("mask", (0,)), ("mask", ())]


def _compatible_configs() -> tuple[dict, dict]:
    target = {
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
        }
    }
    draft = {
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "num_target_layers": 64,
        "vocab_size": 248320,
        "block_size": 16,
        "sliding_window": 2048,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "dflash_config": {
            "mask_token_id": 248070,
            "target_layer_ids": [1, 16, 31, 46, 61],
        },
    }
    return target, draft


def test_qwen36_dflash_configs_are_compatible():
    from mio.dflash.runtime import validate_draft_target_compatibility

    target, draft = _compatible_configs()
    result = validate_draft_target_compatibility(target, draft)
    assert result["num_target_layers"] == 64
    assert result["sliding_window"] == 2048


def test_dflash_config_mismatch_fails_fast():
    from mio.dflash.runtime import validate_draft_target_compatibility

    target, draft = _compatible_configs()
    target["text_config"]["hidden_size"] = 4096
    with pytest.raises(ValueError, match="hidden_size"):
        validate_draft_target_compatibility(target, draft)


def test_effective_draft_window_never_undershoots_checkpoint():
    from mio.dflash.runtime import _effective_draft_window

    model = SimpleNamespace(args=SimpleNamespace(sliding_window=2048))
    assert _effective_draft_window(model, 1024) == 2048
    assert _effective_draft_window(model, 4096) == 4096


def test_verify_len_cap_is_clamped_to_the_dflash_block(monkeypatch):
    from mio.dflash.runtime import _resolve_verify_len_cap

    monkeypatch.setenv("DFLASH_VERIFY_LEN", "5")
    assert _resolve_verify_len_cap(object(), 16) == 5
    monkeypatch.setenv("DFLASH_VERIFY_LEN", "99")
    assert _resolve_verify_len_cap(object(), 16) == 16
    monkeypatch.setenv("DFLASH_VERIFY_LEN", "1")
    assert _resolve_verify_len_cap(object(), 16) == 1


def test_fully_verified_draft_clears_rollback_without_replay():
    from mio.dflash.runtime import _restore_target_cache_after_acceptance

    class Cache:
        _armed = True
        _tape = object()
        _tape_k = object()
        _tape_g = object()
        _tape_qkv = object()
        _snapshot = object()

        def __init__(self):
            self.rollback_calls = []

        def rollback(self, accepted):
            self.rollback_calls.append(accepted)

    cache = Cache()
    replay_ns = _restore_target_cache_after_acceptance(
        [cache],
        target_len=10,
        acceptance_length=4,
        drafted_tokens=4,
    )

    assert replay_ns == 0
    assert cache.rollback_calls == []
    assert cache._armed is False
    assert cache._tape is None

    partial = Cache()
    _restore_target_cache_after_acceptance(
        [partial],
        target_len=8,
        acceptance_length=2,
        drafted_tokens=4,
    )
    assert partial.rollback_calls == [2]


def test_exact_target_cache_snapshot_restores_recurrent_state_and_kv_offset():
    from mio.dflash.runtime import (
        _restore_target_cache_exact,
        _snapshot_target_cache_exact,
    )

    class Recurrent:
        def __init__(self):
            self.cache = [mx.array([1.0]), mx.array([2.0])]
            self.lengths = mx.array([4])
            self.left_padding = None
            self._armed = True
            self._tape = object()
            self._snapshot = object()

    class KV:
        def __init__(self):
            self.offset = 4

    recurrent = Recurrent()
    original_state = list(recurrent.cache)
    kv = KV()
    snapshot = _snapshot_target_cache_exact([recurrent, kv])

    recurrent.cache = [mx.array([9.0]), mx.array([8.0])]
    recurrent.lengths = mx.array([0])
    kv.offset = 12
    _restore_target_cache_exact(
        [recurrent, kv],
        snapshot,
        expected_offset=4,
    )

    assert recurrent.cache[0] is original_state[0]
    assert recurrent.cache[1] is original_state[1]
    assert recurrent.lengths.tolist() == [4]
    assert recurrent._armed is False
    assert recurrent._tape is None
    assert recurrent._snapshot is None
    assert kv.offset == 4


def test_exact_rebuild_rejects_a_block_only_acceptance(monkeypatch):
    import mio.dflash.runtime as runtime

    calls: list[int] = []
    exact_next = {10: 11, 11: 99, 12: 13}

    def fake_forward(
        _model,
        *,
        input_ids,
        cache,
        capture_layer_ids,
        only_last_logit,
    ):
        del cache
        assert capture_layer_ids == {1}
        assert only_last_logit is True
        token = int(input_ids.item())
        calls.append(token)
        logits = mx.full((1, 1, 128), -10.0)
        logits[..., exact_next[token]] = 10.0
        hidden = {1: mx.array([[[float(token)]]])}
        return logits, hidden

    monkeypatch.setattr(runtime, "target_forward_with_hidden_states", fake_forward)
    result = runtime._rebuild_verified_prefix_exact(
        target_model=object(),
        target_cache=[],
        verify_token_ids=mx.array([[10, 11, 12]], dtype=mx.uint32),
        candidate_count=3,
        capture_layer_ids={1},
        suppress_token_mask=None,
        stop_token_ids=set(),
    )

    assert calls == [10, 11]
    assert result["commit_count"] == 2
    assert result["acceptance_length"] == 1
    assert result["stop_hit"] is False
    assert result["next_token"].tolist() == [99]
    assert result["committed_hidden"][1].tolist() == [[[10.0], [11.0]]]


def test_exact_rebuild_stops_at_first_committed_stop(monkeypatch):
    import mio.dflash.runtime as runtime

    def fake_forward(_model, *, input_ids, **_kwargs):
        token = int(input_ids.item())
        logits = mx.zeros((1, 1, 32))
        logits[..., token + 1] = 1.0
        return logits, {1: mx.array([[[float(token)]]])}

    monkeypatch.setattr(runtime, "target_forward_with_hidden_states", fake_forward)
    result = runtime._rebuild_verified_prefix_exact(
        target_model=object(),
        target_cache=[],
        verify_token_ids=mx.array([[10, 11]], dtype=mx.uint32),
        candidate_count=2,
        capture_layer_ids={1},
        suppress_token_mask=None,
        stop_token_ids={10},
    )

    assert result["commit_count"] == 1
    assert result["acceptance_length"] == 0
    assert result["stop_hit"] is True


@pytest.mark.parametrize(
    "stop_ids,expected",
    [
        ([10], (1, True)),
        ([12], (3, True)),
        ([13], (4, True)),
        ([99], (4, False)),
        ([], (4, False)),
    ],
)
def test_commit_prefix_stops_at_first_stop_token(stop_ids, expected):
    from mio.dflash.runtime import _commit_prefix_length

    tokens = mx.array([10, 11, 12, 13], dtype=mx.uint32)
    stops = mx.array(stop_ids, dtype=mx.uint32) if stop_ids else None
    assert _commit_prefix_length(tokens, stops) == expected


def test_chunked_dflash_prefill_projects_each_chunk(monkeypatch):
    import mio.dflash.runtime as runtime

    calls: list[bool] = []

    def fake_target_forward(
        _model,
        *,
        input_ids,
        cache,
        capture_layer_ids,
        only_last_logit,
    ):
        del cache
        calls.append(only_last_logit)
        length = int(input_ids.shape[1])
        assert capture_layer_ids == {1, 2}
        first = mx.ones((1, length, 2), dtype=mx.float32)
        second = mx.full((1, length, 2), 2, dtype=mx.float32)
        logits = mx.zeros((1, 1 if only_last_logit else length, 7))
        return logits, {1: first, 2: second}

    class Draft:
        target_layer_ids = [0, 1]

        @staticmethod
        def project_target_hidden(features):
            return features[..., :2] + features[..., 2:]

    monkeypatch.setattr(runtime, "target_forward_with_hidden_states", fake_target_forward)
    logits, context = runtime.chunked_dflash_prefill(
        object(),
        Draft(),
        input_ids=mx.arange(5, dtype=mx.uint32)[None],
        cache=[],
        chunk_size=2,
        only_last_logit=False,
    )

    assert calls == [True, True, False]
    assert logits.shape == (1, 1, 7)
    assert context.shape == (1, 5, 2)
    assert context.tolist() == [[[3.0, 3.0]] * 5]


def test_one_token_cycle_preserves_unconsumed_draft_context():
    from mio.dflash.runtime import _next_pending_draft_context

    class Draft:
        @staticmethod
        def project_target_hidden(features):
            return features * 10

    previous = mx.array([[[1.0], [2.0]]])
    committed = mx.array([[[3.0]]])

    pending = _next_pending_draft_context(
        Draft(),
        previous=previous,
        committed_hidden=committed,
        previous_was_consumed=False,
    )
    assert pending.tolist() == [[[1.0], [2.0], [30.0]]]

    replaced = _next_pending_draft_context(
        Draft(),
        previous=previous,
        committed_hidden=committed,
        previous_was_consumed=True,
    )
    assert replaced.tolist() == [[[30.0]]]


def test_final_draft_context_is_flushed_when_model_supports_it():
    from mio.dflash.runtime import _advance_draft_context_cache

    class Cache:
        keys = mx.array([1.0])
        values = mx.array([2.0])
        positions = mx.array([3])

    class Draft:
        seen = None

        def advance_projected_context_cache(self, *, draft_context, cache):
            self.seen = (draft_context.tolist(), cache)

    model = Draft()
    cache = [Cache()]
    context = mx.array([[[4.0], [5.0]]])
    assert _advance_draft_context_cache(model, context, cache) is True
    assert model.seen == (context.tolist(), cache)
    assert _advance_draft_context_cache(object(), context, cache) is False
