# SPDX-License-Identifier: Apache-2.0
"""Routed-expert projection for short verify blocks, grouped by expert.

A Lightning MTP verify block routes every row to its own top-k experts, and
``gather_qmm`` runs one quantized mat-vec per (row, expert) pair in routing
order. Consecutive tokens often pick the same experts, so the same expert
weights are fetched from DRAM once per row that selects them.

This kernel runs the same pairs, one threadgroup per pair and output tile,
but in expert order: each threadgroup finds the pair at its sorted position,
and the grid covers all pairs of one output tile before the next. Pairs that
share an expert then run back to back and read the expert tile from cache.
Per pair, the arithmetic is MLX's ``qmv_fast`` or ``qmv`` traversal (chosen by
``qmv_fast_layout``, MLX's rule), in the MLX 0.32.4 ``quantized.h``
transcription of ``omlx.formats.affine``, so each output equals the
``gather_qmm`` output bit for bit.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from ..formats.affine import _BITS, _GROUP_SIZES, _HEADER, qmv_fast_layout

MAX_ROWS = 8

# Each threadgroup takes the pair at sorted position ``y`` (pairs ordered by
# expert, then by pair index) and output tile ``z``. The grid walks every
# pair of one tile before the next tile, so pairs sharing an expert run back
# to back and read that expert tile from cache. Per pair this follows
# qmv_fast_impl (FAST) or the full-tile branch of qmv_impl.
_SOURCE = r"""
    const int pos = int(threadgroup_position_in_grid.y);
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    threadgroup int chosen;
    const int first = int(simd_gid * SIMD_SIZE + simd_lid);
    for (int p = first; p < N_PAIRS; p += 2 * SIMD_SIZE) {
      const uint e = rhs[p];
      int rank = 0;
      for (int q = 0; q < N_PAIRS; ++q) {
        const uint f = rhs[q];
        rank += (f < e || (f == e && q < p)) ? 1 : 0;
      }
      if (rank == pos) {
        chosen = p;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int pair = chosen;
    const int expert = int(rhs[pair]);

    const int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K_SIZE / GS;
    const int out_row = int(threadgroup_position_in_grid.z) * BN +
        int(simd_gid) * RESULTS_PER_SIMDGROUP;

    const device uint8_t* ws = (const device uint8_t*)w +
        expert * (N_SIZE * in_vec_size_w) + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc = scales + expert * (N_SIZE * in_vec_size_g) +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs = biases + expert * (N_SIZE * in_vec_size_g) +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xp = x + (pair / LHS_DIV) * K_SIZE +
        int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[RESULTS_PER_SIMDGROUP] = {0};

    int k = 0;
    const int full_limit = FAST ? K_SIZE : K_SIZE - BLOCK_SIZE;
    for (; k < full_limit; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float s = sc[row * in_vec_size_g];
        float b = bs[row * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }
    if (!FAST) {
      const int remaining = clamp(
          int(K_SIZE - k - int(simd_lid) * VALUES_PER_THREAD),
          0,
          VALUES_PER_THREAD);
      if (remaining > 0) {
        float sum = load_vector_safe<T>(xp, x_thread, remaining);
        for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          const device uint8_t* wl = ws + row * in_vec_size_w;
          float s = sc[row * in_vec_size_g];
          float b = bs[row * in_vec_size_g];
          result[row] += qdot_n(wl, x_thread, s, b, sum, remaining);
        }
      }
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
      float v = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[pair * N_SIZE + out_row + row] = static_cast<T>(v);
      }
    }
"""


@cache
def _kernel(bits: int, group_size: int, fast: bool):
    header = (
        _HEADER.replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__FAST__", "1" if fast else "0")
    )
    return mx.fast.metal_kernel(
        name=f"omlx_moe_verify_gather_b{bits}_gs{group_size}_{int(fast)}",
        input_names=["x", "w", "scales", "biases", "rhs"],
        output_names=["y"],
        header=header,
        source=_SOURCE,
    )


def supported(linear, dtype) -> bool:
    """True when ``linear`` is a quantized switch linear this kernel matches."""
    biases = linear.get("biases") if hasattr(linear, "get") else None
    weight = linear.get("weight") if hasattr(linear, "get") else None
    scales = linear.get("scales") if hasattr(linear, "get") else None
    if weight is None or scales is None or biases is None or "bias" in linear:
        return False
    if getattr(linear, "mode", "affine") != "affine":
        return False
    bits, group_size = getattr(linear, "bits", None), getattr(
        linear, "group_size", None
    )
    if bits not in _BITS or group_size not in _GROUP_SIZES:
        return False
    n = int(weight.shape[-2])
    k = int(scales.shape[-1]) * group_size
    values_per_thread = (8 if bits == 5 else (4 if bits == 6 else 32 // bits)) * (
        2 if qmv_fast_layout(k, n, bits, gathered=True) else 1
    )
    return (
        weight.ndim == 3
        and n % 8 == 0
        and k % 8 == 0
        and group_size % values_per_thread == 0
        and scales.dtype == dtype
        and biases.dtype == dtype
        and dtype in (mx.bfloat16, mx.float16)
    )


def fused_switch(switch_mlp, x: mx.array, indices: mx.array):
    """Fused gate/up then down routed projections for 2..MAX_ROWS rows.

    Returns ``[B, L, top_k, D]`` like the verifier's gather path, or None when
    the block size or weight format is outside this kernel.
    """
    batch, length, width = x.shape
    rows = batch * length
    gate_up = getattr(switch_mlp, "gate_up_proj", None)
    down = getattr(switch_mlp, "down_proj", None)
    if not (
        1 < rows <= MAX_ROWS
        and gate_up is not None
        and down is not None
        and supported(gate_up, x.dtype)
        and supported(down, x.dtype)
    ):
        return None
    top_k = int(indices.shape[-1])
    pairs = indices.reshape(rows * top_k)
    gate_up_out = gather_qmv(gate_up, x.reshape(rows, width), pairs, top_k)
    gate, up = mx.split(gate_up_out, 2, axis=-1)
    out = gather_qmv(down, switch_mlp.activation(up, gate), pairs, 1)
    return out.reshape(batch, length, top_k, -1)


def gather_qmv(linear, x: mx.array, rhs: mx.array, pairs_per_row: int) -> mx.array:
    """``gather_qmm`` over P (row, expert) pairs, pairs grouped by expert.

    ``rhs`` holds the expert of each pair; pair ``p`` reads row
    ``p // pairs_per_row`` of ``x`` (``[R, K]``). Returns ``[P, N]``.
    """
    weight, scales, biases = linear["weight"], linear["scales"], linear["biases"]
    k = int(x.shape[-1])
    n = int(weight.shape[-2])
    pairs = int(rhs.shape[0])
    fast = qmv_fast_layout(k, n, int(linear.bits), gathered=True)
    kernel = _kernel(int(linear.bits), int(linear.group_size), fast)
    return kernel(
        inputs=[x, weight, scales, biases, rhs],
        template=[
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("N_PAIRS", pairs),
            ("LHS_DIV", int(pairs_per_row)),
        ],
        grid=(32, 2 * pairs, n // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[(pairs, n)],
        output_dtypes=[x.dtype],
    )[0]
