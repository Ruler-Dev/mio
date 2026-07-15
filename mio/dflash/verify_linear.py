"""Timewise-exact linear kernels for speculative target verification.

MLX may select a different reduction kernel for a multi-token quantized
matrix multiplication than for autoregressive matrix-vector decoding. Tiny
rounding differences can change an argmax and invalidate greedy speculative
decoding. These helpers evaluate all verify positions in one Metal dispatch
while preserving the singleton reduction order.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import os
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn


_TARGET_VERIFY_ACTIVE: ContextVar[bool] = ContextVar(
    "mio_dflash_target_verify_active",
    default=False,
)


@contextmanager
def target_verify_mode() -> Iterator[None]:
    token = _TARGET_VERIFY_ACTIVE.set(True)
    try:
        yield
    finally:
        _TARGET_VERIFY_ACTIVE.reset(token)


def target_verify_active() -> bool:
    return _TARGET_VERIFY_ACTIVE.get()


@lru_cache(maxsize=16)
def _component_set(raw: str) -> frozenset[str]:
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def target_verify_component_enabled(component: str) -> bool:
    raw = os.environ.get(
        "MIO_DFLASH_EXACT_COMPONENTS",
        "gdn,attention,mlp,head",
    )
    return component.lower() in _component_set(raw)


def _qlinear_header(bits: int, group_size: int) -> str:
    if bits == 4:
        pack_factor, bytes_per_pack, packs_per_thread = 8, 4, 2
    elif bits == 5:
        pack_factor, bytes_per_pack, packs_per_thread = 8, 5, 2
    elif bits == 6:
        pack_factor, bytes_per_pack, packs_per_thread = 4, 3, 2
    elif bits == 8:
        pack_factor, bytes_per_pack, packs_per_thread = 4, 4, 2
    else:
        raise ValueError(f"unsupported Mio exact-QMV bit width: {bits}")

    return (
        r"""
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int BITS = __BITS__;
    constant constexpr int GS = __GS__;
    constant constexpr int PACK_FACTOR = __PACK_FACTOR__;
    constant constexpr int BYTES_PER_PACK = __BYTES_PER_PACK__;
    constant constexpr int PACKS_PER_THREAD = __PACKS_PER_THREAD__;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int WEIGHT_BYTES_PER_THREAD =
        PACKS_PER_THREAD * BYTES_PER_PACK;
    constant constexpr int WEIGHT_BYTES_PER_BLOCK =
        BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    constant constexpr int GROUPS_PER_BLOCK = BLOCK_SIZE / GS;
    constant constexpr int STAGED_BLOCKS = 8;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    template <typename T>
    inline float load_vector_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      if (BITS == 4) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 16.0f;
          x_thread[i + 2] = x[i + 2] / 256.0f;
          x_thread[i + 3] = x[i + 3] / 4096.0f;
        }
      } else if (BITS == 5) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 32.0f;
          x_thread[i + 2] = x[i + 2] / 4.0f;
          x_thread[i + 3] = x[i + 3] / 128.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 2.0f;
          x_thread[i + 6] = x[i + 6] / 64.0f;
          x_thread[i + 7] = x[i + 7] / 8.0f;
        }
      } else if (BITS == 6) {
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 64.0f;
          x_thread[i + 2] = x[i + 2] / 16.0f;
          x_thread[i + 3] = x[i + 3] / 4.0f;
        }
      } else {
        for (int i = 0; i < VALUES_PER_THREAD; ++i) {
          float value = static_cast<float>(x[i]);
          x_thread[i] = value;
          sum += value;
        }
      }
      return sum;
    }

    inline float qdot_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      float accum = 0.0f;
      if (BITS == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
          accum +=
              (x_thread[4 * i] * (ws[i] & 0x000f) +
               x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
               x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
               x_thread[4 * i + 3] * (ws[i] & 0xf000));
        }
      } else if (BITS == 5) {
        for (int i = 0; i < (VALUES_PER_THREAD / 8); ++i) {
          const thread float* xt = x_thread + 8 * i;
          const device uint8_t* wb = w + 5 * i;
          accum += (wb[0] & 0x1f) * xt[0];
          accum += (wb[0] & 0xe0) * xt[1];
          accum += (wb[1] & 0x03) * (xt[1] * 256.0f);
          accum += (wb[1] & 0x7c) * xt[2];
          accum += (wb[1] & 0x80) * xt[3];
          accum += (wb[2] & 0x0f) * (xt[3] * 256.0f);
          accum += (wb[2] & 0xf0) * xt[4];
          accum += (wb[3] & 0x01) * (xt[4] * 256.0f);
          accum += (wb[3] & 0x3e) * xt[5];
          accum += (wb[3] & 0xc0) * xt[6];
          accum += (wb[4] & 0x07) * (xt[6] * 256.0f);
          accum += (wb[4] & 0xf8) * xt[7];
        }
      } else if (BITS == 6) {
        for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
          const thread float* xt = x_thread + 4 * i;
          const device uint8_t* wb = w + 3 * i;
          accum += (wb[0] & 0x3f) * xt[0];
          accum += (wb[0] & 0xc0) * xt[1];
          accum += (wb[1] & 0x0f) * (xt[1] * 256.0f);
          accum += (wb[1] & 0xf0) * xt[2];
          accum += (wb[2] & 0x03) * (xt[2] * 256.0f);
          accum += (wb[2] & 0xfc) * xt[3];
        }
      } else if (BITS == 8) {
        for (int i = 0; i < VALUES_PER_THREAD; ++i) {
          accum += x_thread[i] * float(w[i]);
        }
      }
      return scale * accum + sum * bias;
    }

    inline float qdot_exact_staged(
        const threadgroup uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      float accum = 0.0f;
      if (BITS == 4) {
        const threadgroup uint16_t* ws =
            (const threadgroup uint16_t*)w;
        for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
          accum +=
              (x_thread[4 * i] * (ws[i] & 0x000f) +
               x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
               x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
               x_thread[4 * i + 3] * (ws[i] & 0xf000));
        }
      } else if (BITS == 5) {
        for (int i = 0; i < (VALUES_PER_THREAD / 8); ++i) {
          const thread float* xt = x_thread + 8 * i;
          const threadgroup uint8_t* wb = w + 5 * i;
          accum += (wb[0] & 0x1f) * xt[0];
          accum += (wb[0] & 0xe0) * xt[1];
          accum += (wb[1] & 0x03) * (xt[1] * 256.0f);
          accum += (wb[1] & 0x7c) * xt[2];
          accum += (wb[1] & 0x80) * xt[3];
          accum += (wb[2] & 0x0f) * (xt[3] * 256.0f);
          accum += (wb[2] & 0xf0) * xt[4];
          accum += (wb[3] & 0x01) * (xt[4] * 256.0f);
          accum += (wb[3] & 0x3e) * xt[5];
          accum += (wb[3] & 0xc0) * xt[6];
          accum += (wb[4] & 0x07) * (xt[6] * 256.0f);
          accum += (wb[4] & 0xf8) * xt[7];
        }
      } else if (BITS == 6) {
        for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
          const thread float* xt = x_thread + 4 * i;
          const threadgroup uint8_t* wb = w + 3 * i;
          accum += (wb[0] & 0x3f) * xt[0];
          accum += (wb[0] & 0xc0) * xt[1];
          accum += (wb[1] & 0x0f) * (xt[1] * 256.0f);
          accum += (wb[1] & 0xf0) * xt[2];
          accum += (wb[2] & 0x03) * (xt[2] * 256.0f);
          accum += (wb[2] & 0xfc) * xt[3];
        }
      } else if (BITS == 8) {
        for (int i = 0; i < VALUES_PER_THREAD; ++i) {
          accum += x_thread[i] * float(w[i]);
        }
      }
      return scale * accum + sum * bias;
    }
"""
        .replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__PACK_FACTOR__", str(pack_factor))
        .replace("__BYTES_PER_PACK__", str(bytes_per_pack))
        .replace("__PACKS_PER_THREAD__", str(packs_per_thread))
    )


_QMV_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint vector_group = threadgroup_position_in_grid.z;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int first_vector = int(vector_group) * VECTORS_PER_GROUP;

    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws_base =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_base =
        x + first_vector * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[VECTORS_PER_GROUP][RESULTS_PER_SIMDGROUP];
    float x_thread[VECTORS_PER_GROUP][VALUES_PER_THREAD];
    for (int t = 0; t < VECTORS_PER_GROUP; ++t) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        result[t][row] = 0.0f;
      }
    }

    const device uint8_t* ws = ws_base;
    const device T* sc = scales_base;
    const device T* bs = biases_base;
    const device T* xk = x_base;

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sums[VECTORS_PER_GROUP];
      for (int t = 0; t < VECTORS_PER_GROUP; ++t) {
        if (first_vector + t < TOTAL_VECTORS) {
          sums[t] = load_vector_exact<T>(xk + t * K_SIZE, x_thread[t]);
        }
      }

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        float s = float(sl[0]);
        float b = float(bl[0]);
        for (int t = 0; t < VECTORS_PER_GROUP; ++t) {
          if (first_vector + t < TOTAL_VECTORS) {
            result[t][row] += qdot_exact(wl, x_thread[t], s, b, sums[t]);
          }
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xk += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      for (int t = 0; t < VECTORS_PER_GROUP; ++t) {
        if (first_vector + t < TOTAL_VECTORS) {
          float reduced = simd_sum(result[t][row]);
          if (simd_lid == 0) {
            y[(first_vector + t) * N_SIZE + n] = T(reduced);
          }
        }
      }
    }
"""


_QMV_PARALLEL_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint vector_group = threadgroup_position_in_grid.z;
    uint output_half = thread_position_in_threadgroup.y;
    uint vector_lane = thread_position_in_threadgroup.z;
    uint simd_lid = thread_index_in_simdgroup;

    int vector_idx = int(vector_group) * VECTORS_PER_GROUP + int(vector_lane);
    if (vector_idx >= TOTAL_VECTORS) {
      return;
    }

    int out_row = int(n_tile) * BN + int(output_half) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_ptr =
        scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_ptr =
        biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_ptr =
        x + vector_idx * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    float result[RESULTS_PER_SIMDGROUP] = {0.0f};
    float x_thread[VALUES_PER_THREAD];

    for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
      float sum = load_vector_exact<T>(x_ptr, x_thread);
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        const device uint8_t* weight_row = ws + row * in_vec_size_w;
        const device T* scale_row = scales_ptr + row * in_vec_size_g;
        const device T* bias_row = biases_ptr + row * in_vec_size_g;
        result[row] += qdot_exact(
            weight_row,
            x_thread,
            float(scale_row[0]),
            float(bias_row[0]),
            sum);
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      scales_ptr += BLOCK_SIZE / GS;
      biases_ptr += BLOCK_SIZE / GS;
      x_ptr += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      float reduced = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[vector_idx * N_SIZE + out_row + row] = T(reduced);
      }
    }
"""


_QMV_STAGED_SOURCE = r"""
    uint n_tile = threadgroup_position_in_grid.y;
    uint vector_group = threadgroup_position_in_grid.z;
    uint output_half = thread_position_in_threadgroup.y;
    uint vector_lane = thread_position_in_threadgroup.z;
    uint simd_lid = thread_index_in_simdgroup;

    int vector_idx = int(vector_group) * VECTORS_PER_GROUP + int(vector_lane);
    bool active_vector = vector_idx < TOTAL_VECTORS;
    int safe_vector_idx = min(vector_idx, TOTAL_VECTORS - 1);
    int out_row = int(n_tile) * BN + int(output_half) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K_SIZE / GS;

    const device uint8_t* ws =
        (const device uint8_t*)w + out_row * in_vec_size_w +
        int(simd_lid) * WEIGHT_BYTES_PER_THREAD;
    const device T* scales_ptr = scales + out_row * in_vec_size_g;
    const device T* biases_ptr = biases + out_row * in_vec_size_g;
    const device T* x_ptr =
        x + safe_vector_idx * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

    threadgroup uint8_t
        staged_weights[STAGED_BLOCKS * BN * WEIGHT_BYTES_PER_BLOCK];
    threadgroup T staged_scales[STAGED_BLOCKS * BN * GROUPS_PER_BLOCK];
    threadgroup T staged_biases[STAGED_BLOCKS * BN * GROUPS_PER_BLOCK];
    float result[RESULTS_PER_SIMDGROUP] = {0.0f};
    float x_thread[VALUES_PER_THREAD];

    for (int k = 0; k < K_SIZE; k += STAGED_BLOCKS * BLOCK_SIZE) {
      int blocks_in_chunk =
          min(STAGED_BLOCKS, (K_SIZE - k) / BLOCK_SIZE);
      for (
          int block = int(vector_lane);
          block < blocks_in_chunk;
          block += VECTORS_PER_GROUP) {
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
          int staged_row = int(output_half) * RESULTS_PER_SIMDGROUP + row;
          int staged_block_row = block * BN + staged_row;
          int staged_weight_offset =
              staged_block_row * WEIGHT_BYTES_PER_BLOCK +
              int(simd_lid) * WEIGHT_BYTES_PER_THREAD;
          const device uint8_t* weight_row =
              ws + row * in_vec_size_w + block * WEIGHT_BYTES_PER_BLOCK;
          for (int byte = 0; byte < WEIGHT_BYTES_PER_THREAD; ++byte) {
            staged_weights[staged_weight_offset + byte] = weight_row[byte];
          }
          if (int(simd_lid) < GROUPS_PER_BLOCK) {
            int staged_group =
                staged_block_row * GROUPS_PER_BLOCK + int(simd_lid);
            staged_scales[staged_group] =
                scales_ptr[
                    row * in_vec_size_g + block * GROUPS_PER_BLOCK +
                    int(simd_lid)];
            staged_biases[staged_group] =
                biases_ptr[
                    row * in_vec_size_g + block * GROUPS_PER_BLOCK +
                    int(simd_lid)];
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (active_vector) {
        int lane_group = int(simd_lid) / SCALE_STEP_PER_THREAD;
        for (int block = 0; block < blocks_in_chunk; ++block) {
          float sum = load_vector_exact<T>(
              x_ptr + block * BLOCK_SIZE,
              x_thread);
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int staged_row = int(output_half) * RESULTS_PER_SIMDGROUP + row;
            int staged_block_row = block * BN + staged_row;
            const threadgroup uint8_t* weight_row =
                staged_weights + staged_block_row * WEIGHT_BYTES_PER_BLOCK +
                int(simd_lid) * WEIGHT_BYTES_PER_THREAD;
            int staged_group =
                staged_block_row * GROUPS_PER_BLOCK + lane_group;
            result[row] += qdot_exact_staged(
                weight_row,
                x_thread,
                float(staged_scales[staged_group]),
                float(staged_biases[staged_group]),
                sum);
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      ws += blocks_in_chunk * WEIGHT_BYTES_PER_BLOCK;
      scales_ptr += blocks_in_chunk * GROUPS_PER_BLOCK;
      biases_ptr += blocks_in_chunk * GROUPS_PER_BLOCK;
      x_ptr += blocks_in_chunk * BLOCK_SIZE;
    }

    if (active_vector) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        float reduced = simd_sum(result[row]);
        if (simd_lid == 0) {
          y[vector_idx * N_SIZE + out_row + row] = T(reduced);
        }
      }
    }
"""


@lru_cache(maxsize=None)
def _qmv_kernel(
    bits: int,
    group_size: int,
    dtype: mx.Dtype,
    verify_tokens: int,
    input_size: int,
    output_size: int,
    vectors_per_group: int,
    staging: bool,
):
    dtype_name = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            "mio_target_verify_qmv_"
            f"b{bits}_gs{group_size}_t{verify_tokens}_"
            f"v{vectors_per_group}_k{input_size}_n{output_size}_{dtype_name}_"
            f"{'staged' if staging else 'parallel'}"
        ),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_qlinear_header(bits, group_size),
        source=_QMV_STAGED_SOURCE if staging else _QMV_PARALLEL_SOURCE,
    )


def _can_custom_qmv(linear: Any, x: mx.array) -> bool:
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or not isinstance(linear, nn.QuantizedLinear)
        or x.ndim != 3
        or x.shape[1] <= 1
        or linear.bits not in (4, 5, 6, 8)
        or linear.mode != "affine"
        or linear.biases is None
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.scales.dtype != x.dtype
        or linear.biases.dtype != x.dtype
    ):
        return False
    _, _, input_size = x.shape
    output_size = linear.weight.shape[0]
    packed_input = linear.weight.shape[1] * 32 // linear.bits
    values_per_thread = 16 if linear.bits in (4, 5) else 8
    block_size = values_per_thread * 32
    return (
        input_size == packed_input
        and input_size % block_size == 0
        and input_size % linear.group_size == 0
        and output_size % 8 == 0
        and linear.group_size % values_per_thread == 0
        and block_size % linear.group_size == 0
    )


def _custom_quantized_linear(linear: Any, x: mx.array) -> mx.array | None:
    if not _can_custom_qmv(linear, x):
        return None
    batch, verify_tokens, input_size = x.shape
    output_size = linear.weight.shape[0]
    x = mx.contiguous(x)
    raw_group = os.environ.get("MIO_DFLASH_QMV_VECTORS", "").strip()
    if raw_group:
        try:
            vectors_per_group = int(raw_group)
        except ValueError:
            vectors_per_group = 1
        if vectors_per_group not in (1, 2, 3, 4, 8):
            vectors_per_group = 1
    else:
        vectors_per_group = 4 if int(output_size) >= 65_536 else 2
    total_vectors = int(batch * verify_tokens)
    staging = os.environ.get("MIO_DFLASH_QMV_STAGING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    kernel = _qmv_kernel(
        linear.bits,
        linear.group_size,
        x.dtype,
        int(verify_tokens),
        int(input_size),
        int(output_size),
        vectors_per_group,
        staging,
    )
    (out,) = kernel(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[
            ("T", x.dtype),
            ("VERIFY_T", int(verify_tokens)),
            ("K_SIZE", int(input_size)),
            ("N_SIZE", int(output_size)),
            ("VECTORS_PER_GROUP", vectors_per_group),
            ("TOTAL_VECTORS", total_vectors),
        ],
        grid=(
            32,
            2 * (output_size // 8),
            (
                (total_vectors + vectors_per_group - 1)
                // vectors_per_group
                * vectors_per_group
            ),
        ),
        threadgroup=(32, 2, vectors_per_group),
        output_shapes=[(batch, verify_tokens, output_size)],
        output_dtypes=[x.dtype],
    )
    if "bias" in linear:
        out = out + linear["bias"]
    return out


def _timewise(linear: Any, x: mx.array) -> mx.array:
    return mx.concatenate(
        [linear(x[:, index : index + 1]) for index in range(x.shape[1])],
        axis=1,
    )


def target_verify_linear(linear: Any, x: mx.array) -> mx.array:
    if not target_verify_active() or x.ndim != 3 or x.shape[1] <= 1:
        return linear(x)

    if isinstance(linear, nn.QuantizedLinear):
        out = _custom_quantized_linear(linear, x)
        if out is not None:
            return out
        return _timewise(linear, x)

    if isinstance(linear, nn.Linear):
        from mlx_vlm.models.qwen3_5.language import _target_verify_weight

        if "bias" not in linear:
            out = _target_verify_weight(linear.weight, x)
            if out is not None:
                return out
        return _timewise(linear, x)

    return _timewise(linear, x)


def target_verify_linears(linears: tuple[Any, ...], x: mx.array) -> tuple[mx.array, ...]:
    return tuple(target_verify_linear(linear, x) for linear in linears)


def target_verify_embedding_as_linear(embedding: Any, x: mx.array) -> mx.array:
    if not target_verify_active() or x.ndim != 3 or x.shape[1] <= 1:
        return embedding.as_linear(x)
    from mlx_vlm.models.qwen3_5.language import _target_verify_weight

    out = _target_verify_weight(embedding.weight, x)
    if out is not None:
        return out
    return mx.concatenate(
        [embedding.as_linear(x[:, index : index + 1]) for index in range(x.shape[1])],
        axis=1,
    )
