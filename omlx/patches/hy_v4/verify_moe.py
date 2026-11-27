# SPDX-License-Identifier: Apache-2.0
"""Fused split affine expert GEMVs and clamped SwiGLU for Hy4 verify."""

import logging
from functools import cache

import mlx.core as mx

from ...custom_kernels.quant_matmul import fast as quant_fast
from ..deepseek_v4.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    _sort_threshold,
)
from ..moe_verify_gather import gather_qmv, pair_order, projection

logger = logging.getLogger(__name__)


# MLX's 3- and 4-bit qmv_fast load_vector / qdot_n helpers for the fused split
# gate/up kernel below.
_AFFINE_HEADER = r"""
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int BITS = __BITS__;
constant constexpr int GS = __GS__;
constant constexpr int PACK_FACTOR = (BITS == 3) ? 8 : (32 / BITS);
constant constexpr int BYTES_PER_PACK = (BITS == 3) ? 3 : 4;
constant constexpr int PACKS_PER_THREAD = 2;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;

template <typename T>
inline float load_vector(const device T* x, thread float* x_thread) {
  float sum = 0;
  if (BITS == 3) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
          float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  } else {
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }
  return sum;
}

template <typename T>
inline float qdot_n(
    const device uint8_t* w,
    const thread float* x_thread,
    T scale,
    T bias,
    float sum,
    int N) {
  float accum = 0;
  if (BITS == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  } else {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }
  return scale * accum + sum * bias;
}
"""


_GATE_UP = r"""
    const int p = int(order[threadgroup_position_in_grid.y]);
    const int lane = int(thread_index_in_simdgroup);
    const int out = int(threadgroup_position_in_grid.z) * COLS * GROUPS
        + int(simdgroup_index_in_threadgroup) * COLS;
    const size_t e = size_t(rhs[p]);
    const int ws = K * BYTES_PER_PACK / PACK_FACTOR;
    const int ss = K / GS;
    const size_t offset = e * N + out;
    const size_t wp = offset * ws + lane * PACKS_PER_THREAD * BYTES_PER_PACK;
    const size_t sp = offset * ss + lane / SCALE_STEP_PER_THREAD;
    const device T* xp = x + (p / TOP) * K + lane * VALUES_PER_THREAD;
    float gate[COLS] = {0}, up[COLS] = {0};

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        float values[VALUES_PER_THREAD];
        const float sum = load_vector(xp + k, values);
        for (int c = 0; c < COLS; ++c) {
            const size_t wi = wp + c * ws + k * BYTES_PER_PACK / PACK_FACTOR;
            const size_t si = sp + c * ss + k / GS;
            gate[c] += qdot_n(
                (const device uchar*)gw + wi,
                values, gs[si], gb[si], sum, VALUES_PER_THREAD);
            up[c] += qdot_n(
                (const device uchar*)uw + wi,
                values, us[si], ub[si], sum, VALUES_PER_THREAD);
        }
    }
    for (int c = 0; c < COLS; ++c) {
        const float g1 = simd_sum(gate[c]);
        const float u1 = simd_sum(up[c]);
        if (lane == 0) {
            const T g = metal::min(T(g1), limit[0]);
            const T u = metal::min(metal::max(T(u1), T(-limit[0])), limit[0]);
            const auto a = 1 / (1 + metal::precise::exp(metal::abs(g)));
            const T sigmoid = (g < 0) ? a : 1 - a;
            y[p * N + out + c] = T(T(g * sigmoid) * u);
        }
    }
"""


@cache
def _gate_up_kernel(bits, group):
    header = _AFFINE_HEADER.replace("__BITS__", str(bits)).replace("__GS__", str(group))
    logger.debug(
        "Hy4 short verify: fused affine expert gate/up (bits=%d, group=%d)", bits, group
    )
    return mx.fast.metal_kernel(
        name=f"hy4_decode_split_gate_up_b{bits}_g{group}",
        input_names=["x", "gw", "gs", "gb", "uw", "us", "ub", "rhs", "order", "limit"],
        output_names=["y"],
        header=header,
        source=_GATE_UP,
    )


def _gate_up(x, gate, up, rhs, order, top, limit):
    cols, groups = 2, 2

    def operands(proj):
        return [proj["weight"], proj["scales"], proj["biases"]]

    n = gate.output_dims
    return _gate_up_kernel(gate.bits, gate.group_size)(
        inputs=[
            x,
            *operands(gate),
            *operands(up),
            rhs,
            order,
            mx.array([limit], dtype=x.dtype),
        ],
        template=[
            ("T", x.dtype),
            ("K", x.shape[-1]),
            ("N", n),
            ("TOP", top),
            ("COLS", cols),
            ("GROUPS", groups),
        ],
        grid=(32 * groups, rhs.size, n // (cols * groups)),
        threadgroup=(32 * groups, 1, 1),
        output_shapes=[(rhs.size, n)],
        output_dtypes=[x.dtype],
    )[0]


def routed_experts(switch, routed, indices, scores, limit):
    """Preserve router slot order and the stock expert reduction."""
    if (
        type(switch) is not SwitchGLU
        or switch.training
        or routed.shape[-1] != 1536
        or indices.shape != (routed.shape[0], 16)
        or "gate_up_proj" in switch
        or "oqh_experts" in switch
        or routed.dtype not in (mx.bfloat16, mx.float16)
    ):
        return None
    projections = switch.projections()
    if not all(
        type(proj) is QuantizedSwitchLinear
        and proj.bits in (3, 4)
        and proj.group_size in (32, 64, 128)
        and proj.input_dims == proj.output_dims == 1536
        and proj._has_affine_metadata_dtype(routed.dtype)
        and "bias" not in proj
        and "global_scale" not in proj
        for proj in projections
    ):
        return None
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    if (gate.bits, gate.group_size, gate.weight.shape) != (
        up.bits,
        up.group_size,
        up.weight.shape,
    ):
        return None

    do_sort = indices.size >= _sort_threshold(*projections)
    # MLX 0.32.4 GatherQMM::eval_gpu switches sorted one-row gathers to a
    # matrix kernel at four routes per expert bank entry. Its rounding is
    # different from the GEMV arithmetic reproduced below.
    if do_sort and any(indices.size >= 4 * proj.num_experts for proj in projections):
        return None

    down_proj = projection(down, down.input_dims)
    if down_proj is None or not down_proj.face.exact_for(routed.dtype):
        return None
    rhs = indices.astype(mx.uint32).reshape(-1)
    order = pair_order(rhs)
    hidden = _gate_up(routed, gate, up, rhs, order, indices.shape[-1], limit)
    values = gather_qmv(down_proj, hidden, rhs, 1, order)
    if (
        scores.dtype == mx.float32
        and do_sort
        and scores.shape[-1] in quant_fast.moe_weighted_sum_top_ks()
        and quant_fast.has_symbol("moe_weighted_sum")
    ):
        identity = mx.arange(rhs.size, dtype=mx.uint32)
        return quant_fast.moe_weighted_sum(
            values.reshape(rhs.size, 1, -1), identity, scores
        )
    return values.reshape(*indices.shape, down.output_dims)
