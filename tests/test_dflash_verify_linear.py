"""Exactness tests for timewise speculative-verification linears."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mio.dflash.verify_linear import (
    _can_custom_qmv,
    target_verify_linear,
    target_verify_mode,
)


def _singleton_reference(linear, inputs: mx.array) -> mx.array:
    return mx.concatenate(
        [linear(inputs[:, index : index + 1]) for index in range(inputs.shape[1])],
        axis=1,
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal kernels")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_quantized_target_verify_is_bit_exact_with_singleton_decode(bits: int):
    mx.random.seed(bits)
    dense = nn.Linear(512, 32, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    linear = nn.QuantizedLinear.from_linear(
        dense,
        group_size=64,
        bits=bits,
        mode="affine",
    )
    inputs = mx.sin(mx.arange(4 * 512, dtype=mx.float32) * 0.003).reshape(
        1, 4, 512
    ).astype(mx.bfloat16)

    expected = _singleton_reference(linear, inputs)
    with target_verify_mode():
        actual = target_verify_linear(linear, inputs)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal kernels")
def test_dense_target_verify_is_bit_exact_with_singleton_decode():
    mx.random.seed(7)
    linear = nn.Linear(512, 32, bias=False)
    linear.weight = linear.weight.astype(mx.bfloat16)
    inputs = mx.cos(mx.arange(4 * 512, dtype=mx.float32) * 0.005).reshape(
        1, 4, 512
    ).astype(mx.bfloat16)

    expected = _singleton_reference(linear, inputs)
    with target_verify_mode():
        actual = target_verify_linear(linear, inputs)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal kernels")
@pytest.mark.parametrize("vectors_per_group", [1, 2, 3, 4, 8])
def test_quantized_target_verify_handles_partial_vector_groups(
    monkeypatch: pytest.MonkeyPatch,
    vectors_per_group: int,
):
    mx.random.seed(11)
    dense = nn.Linear(512, 32, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    linear = nn.QuantizedLinear.from_linear(
        dense,
        group_size=64,
        bits=6,
        mode="affine",
    )
    inputs = mx.sin(mx.arange(2 * 5 * 512, dtype=mx.float32) * 0.002).reshape(
        2, 5, 512
    ).astype(mx.bfloat16)
    monkeypatch.setenv("MIO_DFLASH_QMV_VECTORS", str(vectors_per_group))

    expected = _singleton_reference(linear, inputs)
    with target_verify_mode():
        actual = target_verify_linear(linear, inputs)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal kernels")
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("vectors_per_group", [1, 2, 3, 4, 8])
def test_staged_quantized_target_verify_is_exact_for_padded_vector_groups(
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    vectors_per_group: int,
):
    mx.random.seed(bits * 100 + vectors_per_group)
    dense = nn.Linear(512, 32, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    linear = nn.QuantizedLinear.from_linear(
        dense,
        group_size=64,
        bits=bits,
        mode="affine",
    )
    inputs = mx.cos(mx.arange(5 * 512, dtype=mx.float32) * 0.0017).reshape(
        1, 5, 512
    ).astype(mx.bfloat16)
    monkeypatch.setenv("MIO_DFLASH_QMV_STAGING", "1")
    monkeypatch.setenv("MIO_DFLASH_QMV_VECTORS", str(vectors_per_group))

    expected = _singleton_reference(linear, inputs)
    with target_verify_mode():
        actual = target_verify_linear(linear, inputs)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal kernels")
def test_custom_qmv_rejects_group_sizes_that_cross_reduction_blocks():
    dense = nn.Linear(1536, 32, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    linear = nn.QuantizedLinear.from_linear(
        dense,
        group_size=64,
        bits=4,
        mode="affine",
    )
    # MLX currently exposes only aligned production group sizes. Mutating the
    # metadata exercises the defensive guard for imported/custom checkpoints.
    linear.group_size = 192
    inputs = mx.zeros((1, 3, 1536), dtype=mx.bfloat16)

    assert not _can_custom_qmv(linear, inputs)
