"""Numerical parity tests for speculative recurrent-state kernels."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models import gated_delta

from mio.dflash.kernels import gated_delta_kernel_with_tape, tape_replay_kernel


def _wave(size: int, scale: float) -> mx.array:
    return mx.sin(mx.arange(size, dtype=mx.float32) * scale)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires MLX Metal kernels")
@pytest.mark.parametrize("vectorized_gate", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_taped_gated_delta_is_bit_exact_with_mlx(
    vectorized_gate: bool,
    masked: bool,
):
    batch, steps, key_heads, key_dim = 1, 12, 2, 32
    value_heads, value_dim = 4, 8
    q = _wave(batch * steps * key_heads * key_dim, 0.017).reshape(
        batch, steps, key_heads, key_dim
    ).astype(mx.bfloat16)
    k = _wave(batch * steps * key_heads * key_dim, 0.013).reshape(
        batch, steps, key_heads, key_dim
    ).astype(mx.bfloat16)
    v = _wave(batch * steps * value_heads * value_dim, 0.019).reshape(
        batch, steps, value_heads, value_dim
    ).astype(mx.bfloat16)
    beta = ((_wave(batch * steps * value_heads, 0.023) + 1) * 0.35 + 0.1).reshape(
        batch, steps, value_heads
    ).astype(mx.bfloat16)
    state = (_wave(batch * value_heads * value_dim * key_dim, 0.007) * 0.1).reshape(
        batch, value_heads, value_dim, key_dim
    )
    if vectorized_gate:
        gate = (0.92 + 0.03 * _wave(batch * steps * value_heads * key_dim, 0.011)).reshape(
            batch, steps, value_heads, key_dim
        ).astype(mx.bfloat16)
    else:
        gate = (0.92 + 0.03 * _wave(batch * steps * value_heads, 0.011)).reshape(
            batch, steps, value_heads
        ).astype(mx.bfloat16)

    mask = None
    if masked:
        mask = mx.array([[True] * 5 + [False] + [True] * 6])

    expected_y, expected_state = gated_delta.gated_delta_kernel(
        q, k, v, gate, beta, state, mask
    )
    actual_y, actual_state, tape = gated_delta_kernel_with_tape(
        q, k, v, gate, beta, state, mask
    )

    accepted_steps = 7
    replayed_state = tape_replay_kernel(
        tape[:, :accepted_steps],
        k[:, :accepted_steps],
        gate[:, :accepted_steps],
        state,
        None if mask is None else mask[:, :accepted_steps],
    )
    _, expected_replayed_state = gated_delta.gated_delta_kernel(
        q[:, :accepted_steps],
        k[:, :accepted_steps],
        v[:, :accepted_steps],
        gate[:, :accepted_steps],
        beta[:, :accepted_steps],
        state,
        None if mask is None else mask[:, :accepted_steps],
    )
    continuation = slice(-3, None)
    expected_next_y, expected_next_state = gated_delta.gated_delta_kernel(
        q[:, continuation],
        k[:, continuation],
        v[:, continuation],
        gate[:, continuation],
        beta[:, continuation],
        expected_state,
    )
    actual_next_y, actual_next_state, _ = gated_delta_kernel_with_tape(
        q[:, continuation],
        k[:, continuation],
        v[:, continuation],
        gate[:, continuation],
        beta[:, continuation],
        actual_state,
    )
    mx.eval(
        expected_y,
        expected_state,
        actual_y,
        actual_state,
        replayed_state,
        expected_replayed_state,
        expected_next_y,
        expected_next_state,
        actual_next_y,
        actual_next_state,
    )

    assert actual_state.dtype == expected_state.dtype == mx.float32
    assert mx.array_equal(actual_y, expected_y).item()
    assert mx.array_equal(actual_state, expected_state).item()
    assert mx.array_equal(replayed_state, expected_replayed_state).item()
    assert mx.array_equal(actual_next_y, expected_next_y).item()
    assert mx.array_equal(actual_next_state, expected_next_state).item()
