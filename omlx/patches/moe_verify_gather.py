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
Per pair, the rows are the dot face of the weights' format in the mat-vec
skeleton (``omlx.formats``). The affine face is MLX's ``qmv_fast`` or ``qmv``
traversal, so each of its outputs equals the ``gather_qmm`` output bit for
bit.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import Any, NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .. import formats
from ..formats import Dot, kernels
from .module_cache import cached_per_module

logger = logging.getLogger(__name__)

MAX_ROWS = 8

# Each threadgroup takes the pair at sorted position ``y`` (pairs ordered by
# expert, then by pair index) and output tile ``z``. The grid walks every
# pair of one tile before the next tile, so pairs sharing an expert run back
# to back and read that expert tile from cache. A tile is 8 of a pair's
# output rows: 4 for each of the threadgroup's two simdgroups.
_SOURCE = r"""
    constexpr int LANES = 32;  // of a simdgroup
    constexpr int RPS = 4;     // output rows of a simdgroup
    static_assert(
        N_SIZE % (2 * RPS) == 0,
        "verify gather: N must be whole tiles of 8 outputs");
    static_assert(
        N_PAIRS % LHS_DIV == 0,
        "verify gather: pairs are whole rows of LHS_DIV experts");
    const int pos = int(threadgroup_position_in_grid.y);
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    threadgroup int chosen;
    const int first = int(simd_gid * LANES + simd_lid);
    for (int p = first; p < N_PAIRS; p += 2 * LANES) {
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
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const int pair = chosen;

    const int out_row =
        int(threadgroup_position_in_grid.z) * (2 * RPS) + int(simd_gid) * RPS;
    float result[RPS];
    qmv_rows<T, K_SIZE, RPS, 0>(
        rt, size_t(rhs[pair]) * N_SIZE + out_row, rt, 0,
        x + (pair / LHS_DIV) * K_SIZE, simd_lid, result);
    if (simd_lid == 0) {
      store_rows(omlx_iota<RPS>(), result, y + pair * N_SIZE + out_row);
    }
"""


@cache
def _kernel(face: Dot):
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("rt", face), "rhs"]
    return mx.fast.metal_kernel(
        name=f"omlx_moe_verify_gather_{face.key}",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header + kernels.STORE_ROWS,
        source=matrices.declarations + _SOURCE,
    )


class Projection(NamedTuple):
    """Stacked expert weights as this kernel runs them."""

    face: Dot
    kernel: Any
    operands: tuple  # the weight, then the face's tensors
    n: int


class _Plan(NamedTuple):
    gate_up: Projection
    down: Projection
    proof: kernels.Proof  # of a block's two launches


def projection(linear, k: int) -> Projection | None:
    """``linear``'s experts (``[E, N, K]``, no bias) as this kernel runs
    them, or None: their format has a dot face for them, and N is whole
    tiles of 8 outputs."""
    if not isinstance(linear, nn.Module) or "bias" in linear:
        return None
    weight = linear.get("weight")
    if not isinstance(weight, mx.array) or weight.ndim != 3 or weight.shape[1] % 8:
        return None
    n = weight.shape[1]
    face = formats.dot(linear, k, n)
    if face is None:
        return None
    operands = tuple(linear[name] for name in ("weight", *face.tensors))
    return Projection(face, _kernel(face), operands, n)


def _build_plan(switch_mlp, width: int) -> _Plan | None:
    gate_up = projection(switch_mlp.get("gate_up_proj"), width)
    if gate_up is None:
        return None
    down = projection(switch_mlp.get("down_proj"), gate_up.n // 2)
    if down is None:
        return None
    # What the launches are compiled from, beside a block's rows and routing.
    proof = kernels.proof(
        "verify_gather",
        gate_up.face,
        down.face,
        tuple(a.dtype for a in gate_up.operands + down.operands),
        width,
        gate_up.n,
        down.n,
    )
    return _Plan(gate_up, down, proof)


def fused_switch(switch_mlp, x: mx.array, indices: mx.array):
    """Fused gate/up then down routed projections for 2..MAX_ROWS rows.

    Returns ``[B, L, top_k, D]`` like the verifier's gather path, or None when
    the block size is outside this kernel, a projection has no dot face that
    is exact for ``x``'s dtype, the activation gives another dtype or width
    than the gate's and up's, or a block of these faces and shapes failed.
    """
    batch, length, width = x.shape
    rows = batch * length
    if not 1 < rows <= MAX_ROWS:
        return None
    plan = cached_per_module(
        switch_mlp,
        "_omlx_verify_gather_plan",
        lambda switch: _build_plan(switch, width),
        flags=(width,),
    )
    if (
        plan is None
        or plan.proof.failed
        or not plan.gate_up.face.exact_for(x.dtype)
        or not plan.down.face.exact_for(x.dtype)
    ):
        return None
    top_k = indices.shape[-1]
    pairs = indices.reshape(rows * top_k)
    launch = (rows, top_k, x.dtype, indices.dtype)
    proven = launch in plan.proof.ran
    if not proven:
        # What fails ahead of the block is not the launches' failure.
        mx.eval(x, indices)
    try:
        gate_up = gather_qmv(plan.gate_up, x.reshape(rows, width), pairs, top_k)
        gate, up = mx.split(gate_up, 2, axis=-1)
        hidden = switch_mlp.activation(up, gate)
        if hidden.dtype != x.dtype or hidden.shape != up.shape:
            # Not the rows that the down launch is planned and proven for.
            return None
        out = gather_qmv(plan.down, hidden, pairs, 1)
        if not proven:
            # Surface a kernel build failure while the call can still fall
            # back.
            mx.eval(out)
            plan.proof.ran.add(launch)
            logger.info("MoE expert-ordered verify gather engaged (%d rows)", rows)
    except Exception:
        plan.proof.failed = True
        logger.warning(
            "expert-ordered verify gather failed; composed fallback", exc_info=True
        )
        return None
    return out.reshape(batch, length, top_k, -1)


def gather_qmv(
    projection: Projection, x: mx.array, rhs: mx.array, pairs_per_row: int
) -> mx.array:
    """``gather_qmm`` over P (row, expert) pairs, pairs grouped by expert.

    ``rhs`` holds the expert of each pair; pair ``p`` reads row
    ``p // pairs_per_row`` of ``x`` (``[R, K]``). Returns ``[P, N]``.
    """
    n = projection.n
    pairs = rhs.shape[0]
    return projection.kernel(
        inputs=[x, *projection.operands, rhs],
        template=[
            ("T", x.dtype),
            ("K_SIZE", x.shape[-1]),
            ("N_SIZE", n),
            ("N_PAIRS", pairs),
            ("LHS_DIV", pairs_per_row),
        ],
        grid=(32, 2 * pairs, n // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[(pairs, n)],
        output_dtypes=[x.dtype],
    )[0]
