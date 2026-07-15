"""Regression tests for the DFlash/Qwen compatibility layer."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest


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
