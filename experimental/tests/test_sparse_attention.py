"""Validate sparse_model_forward matches mlx-lm's stock Qwen3 forward
on dense input (positions=arange(prompt_len)). If they match, our
position-aware path is correct for sparse positions too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experimental.spec_prefill.sparse_attention import sparse_model_forward  # type: ignore


@pytest.fixture(scope="module")
def loaded_qwen3():
    """Load Qwen3-8B once for all tests."""
    from mlx_lm.utils import load
    target_path = "/Users/ruler/Documents/mio/models/Qwen3-8B-4bit"
    model, tokenizer = load(target_path)
    mx.eval(model.parameters())
    return model, tokenizer


def test_sparse_dense_logits_close(loaded_qwen3):
    """Sparse forward with positions=arange should produce close logits to stock.

    Compounding bf16 precision drift across 36 layers means the absolute diff
    grows; we accept a few units of diff but argmax (next-token) must agree.
    """
    model, tok = loaded_qwen3
    prompt_text = "The quick brown fox jumps over the lazy dog. " * 4
    ids = mx.array(tok.encode(prompt_text), dtype=mx.uint32)[None]
    L = int(ids.shape[1])
    positions = mx.arange(L, dtype=mx.int32)

    sparse_logits = sparse_model_forward(model, ids, positions=positions)
    stock_logits = model(ids)
    mx.eval(sparse_logits, stock_logits)

    diff = mx.abs(sparse_logits - stock_logits).max().item()
    print(f"\nL={L}  abs_logit_diff={diff:.4f}")
    # Within 4× expected bf16 precision compounding (36 layers × 0.06 baseline rope diff).
    assert diff < 4.0, f"abs_diff too large: {diff}"


def test_sparse_dense_argmax_matches_stock(loaded_qwen3):
    """Even with float noise, the next-token argmax should be identical."""
    model, tok = loaded_qwen3
    prompts = [
        "Write a Python function that computes the factorial of n. The function should ",
        "The capital of France is",
        "Once upon a time, there was a small village nestled in the mountains. ",
    ]
    for prompt_text in prompts:
        ids = mx.array(tok.encode(prompt_text), dtype=mx.uint32)[None]
        L = int(ids.shape[1])
        positions = mx.arange(L, dtype=mx.int32)
        sparse_logits = sparse_model_forward(model, ids, positions=positions)
        stock_logits = model(ids)
        mx.eval(sparse_logits, stock_logits)
        sparse_next = int(mx.argmax(sparse_logits[:, -1, :], axis=-1).item())
        stock_next = int(mx.argmax(stock_logits[:, -1, :], axis=-1).item())
        assert sparse_next == stock_next, (
            f"prompt: {prompt_text[:40]}... next-token mismatch: "
            f"sparse={sparse_next} ({tok.decode([sparse_next])}) "
            f"stock={stock_next} ({tok.decode([stock_next])})"
        )
