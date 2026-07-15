from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experimental.dgsa.selection import attention_scored


class _RecordingLinearLayer:
    is_linear = True

    def __init__(self) -> None:
        self.mask = None

    def __call__(self, hidden, *, mask, cache):
        self.mask = mask
        assert cache is None
        return hidden


def test_attention_scoring_uses_qwen_ssm_mask_for_linear_layers(monkeypatch):
    """The scorer must mirror Qwen3NextModel's hybrid mask routing.

    A full-attention causal mask has different shape and semantics from the
    boolean sequence mask consumed by GatedDeltaNet. Passing the former can
    corrupt or fail scoring before the selected full-attention layer.
    """
    fa_mask = object()
    ssm_mask = object()
    monkeypatch.setattr(
        "mlx_lm.models.base.create_attention_mask",
        lambda hidden, cache: fa_mask,
    )
    monkeypatch.setattr(
        "mlx_lm.models.qwen3_next.create_ssm_mask",
        lambda hidden, cache: ssm_mask,
    )

    linear = _RecordingLinearLayer()
    inner = SimpleNamespace(
        embed_tokens=lambda ids: mx.zeros((1, int(ids.shape[1]), 4)),
        layers=[linear],
    )
    model = SimpleNamespace(model=inner)

    kept = attention_scored(
        model,
        mx.array([[1, 2, 3, 4]], dtype=mx.uint32),
        score_layer=99,
        keep_ratio=0.5,
        keep_first=1,
        keep_last=1,
    )

    assert linear.mask is ssm_mask
    assert kept.tolist() == [0, 2, 3]
