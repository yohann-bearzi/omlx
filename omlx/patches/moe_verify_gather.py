# SPDX-License-Identifier: Apache-2.0
"""Routed-expert projection for short verify blocks, grouped by expert.

A Lightning MTP verify block routes every row to its own top-k experts, and
``gather_qmm`` runs one quantized mat-vec per (row, expert) pair in routing
order. Consecutive tokens often pick the same experts, so the same expert
weights are fetched from DRAM once per row that selects them.

This kernel runs the same pairs, one threadgroup per pair and output tile,
but in expert order: each threadgroup takes the pair at its sorted position
(read from the block's order from ORDERED_PAIRS pairs, searched for below),
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

# A block of 2 to MAX_ROWS rows takes the gather, but one of FEW_PAIRS
# pairs or fewer over MANY_EXPERTS experts or more. Chained against the
# composed path on both Macs (Flash-Next's block, the 35B-A3B's, one of 8
# experts; 2, 3, 4 and 8 bits; independent and correlated routing), the
# gather is faster in every block but 2 rows (16 pairs) of the 35B-A3B's
# 256 experts with independent routing on the M5 Max: 1.011 at 4 bits,
# above the bar of 1.005 whatever correlated routing gains there. Few
# pairs over many experts share few experts: little for the gather to
# save. The 8-expert block wins at 8 to 16 pairs.
MAX_ROWS = 8
FEW_PAIRS = 16
MANY_EXPERTS = 64
# From this many pairs a block's pairs are put in expert order once, by a
# launch of their own (_ORDER) ahead of the block's two (_ORDERED_SOURCE);
# below it, each threadgroup ranks its pair among all of them (_SOURCE).
# Measured with 16 instances a side in 3 processes on both Macs: the extra
# launch costs 0.3-0.8% at 16 and 20 pairs and gains from 24 pairs.
ORDERED_PAIRS = 24
# MLX takes signed int32 grid dimensions. Two simdgroups are launched per pair.
_MAX_PAIRS = (2**31 - 1) // 2

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
        LHS_DIV >= 1,
        "verify gather: LHS_DIV is the experts of a row, 1 or more");
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
        x + size_t(pair / LHS_DIV) * K_SIZE, simd_lid, result);
    if (simd_lid == 0) {
      store_rows(omlx_iota<RPS>(), result, y + size_t(pair) * N_SIZE + out_row);
    }
"""


# _SOURCE with the pair at position ``y`` read from the block's order.
_ORDERED_SOURCE = r"""
    constexpr int RPS = 4;     // output rows of a simdgroup
    static_assert(
        N_SIZE % (2 * RPS) == 0,
        "verify gather: N must be whole tiles of 8 outputs");
    static_assert(
        LHS_DIV >= 1,
        "verify gather: LHS_DIV is the experts of a row, 1 or more");
    const uint pair = order[threadgroup_position_in_grid.y];
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint out_row = threadgroup_position_in_grid.z * (2 * RPS) + simd_gid * RPS;
    float result[RPS];
    qmv_rows<T, K_SIZE, RPS, 0>(
        rt, size_t(rhs[pair]) * N_SIZE + out_row, rt, 0,
        x + OFFSET(pair / uint(LHS_DIV)) * K_SIZE, thread_index_in_simdgroup, result);
    if (thread_index_in_simdgroup == 0) {
      store_rows(omlx_iota<RPS>(), result, y + OFFSET(pair) * N_SIZE + out_row);
    }
"""

# Thread p ranks pair p by (expert, pair index), the order _SOURCE's search
# uses, and writes p at that place.
_ORDER = r"""
    const uint p = thread_position_in_grid.x;
    const uint e = rhs[p];
    uint rank = 0;
    for (uint q = 0; q < uint(N_PAIRS); ++q) {
      const uint f = rhs[q];
      rank += (f < e || (f == e && q < p)) ? 1 : 0;
    }
    order[rank] = p;
"""


@cache
def _order_kernel():
    return mx.fast.metal_kernel(
        name="omlx_moe_verify_gather_order",
        input_names=["rhs"],
        output_names=["order"],
        source=_ORDER,
    )


def _order(rhs: mx.array) -> mx.array:
    """Pair indices in expert order (the experts as 32-bit unsigned
    integers), ties by pair: ``rhs``'s stable argsort for expert ids."""
    pairs = rhs.shape[0]
    if not pairs:
        return rhs.astype(mx.uint32)
    return _order_kernel()(
        inputs=[rhs],
        template=[("N_PAIRS", pairs)],
        grid=(pairs, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(pairs,)],
        output_dtypes=[mx.uint32],
    )[0]


@cache
def _ordered_kernel(face: Dot, small_offsets: bool = False):
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("rt", face), "rhs", "order"]
    offset_type = "uint" if small_offsets else "size_t"
    suffix = "_u32" if small_offsets else ""
    body = _ORDERED_SOURCE.replace("OFFSET", offset_type)
    return mx.fast.metal_kernel(
        name=f"omlx_moe_verify_gather_ordered_{face.key}{suffix}",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header + kernels.STORE_ROWS,
        source=matrices.declarations + body,
    )


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
    kernel: Any  # the pairs ranked in each threadgroup
    ordered: Any  # the pairs read from the block's order
    operands: tuple  # the weight, then the face's tensors
    n: int


class _Plan(NamedTuple):
    gate_up: Projection
    down: Projection
    proof: kernels.Proof  # of a block's launches (and its order's)
    width: int  # the K of gate_up: a row of x
    few_pairs: int  # blocks of this many pairs or fewer keep the composed path


def projection(linear, k: int) -> Projection | None:
    """``linear``'s experts (``[E, N, K]``, no bias) as this kernel runs
    them, or None: their format has a dot face for them, and N is whole
    tiles of 8 outputs."""
    return _projection(linear, k, None)


def _projection(
    linear, k: int, row_bounds: tuple[int, int] | None
) -> Projection | None:
    if not isinstance(linear, nn.Module) or "bias" in linear:
        return None
    names = formats.tensor_names(linear)
    # A format layer that still holds a load's placeholder has nothing to
    # read: refused here, where the plan is built (D11).
    refuse = getattr(linear, "refuse_placeholders", None)
    if refuse is not None:
        refuse()
    weight = linear.get(names[0])
    if not isinstance(weight, mx.array) or weight.ndim != 3 or weight.shape[1] % 8:
        return None
    n = weight.shape[1]
    face = formats.dot(linear, k, n)
    if face is None:
        return None
    operands = tuple(linear[name] for name in (names[0], *face.tensors))
    # The last input/output element is one less than its element count.
    # Direct gathers accept any valid pair grid and keep wide offsets.
    small_offsets = False
    if row_bounds is not None:
        input_rows, output_rows = row_bounds
        small_offsets = max(input_rows * k, output_rows * n) <= 2**32
    ordered = _ordered_kernel(face, small_offsets)
    return Projection(face, _kernel(face), ordered, operands, n)


def _build_plan(switch_mlp, width: int, top_k: int) -> _Plan | None:
    layers = switch_mlp.get("gate_up_proj"), switch_mlp.get("down_proj")
    max_pairs = MAX_ROWS * top_k
    gate_up = _projection(layers[0], width, (MAX_ROWS, max_pairs))
    if gate_up is None:
        return None
    down = _projection(layers[1], gate_up.n // 2, (max_pairs, max_pairs))
    # A pair's expert indexes both stacks.
    if down is None or down.operands[0].shape[0] != gate_up.operands[0].shape[0]:
        return None
    few_pairs = FEW_PAIRS if gate_up.operands[0].shape[0] >= MANY_EXPERTS else 0
    # What the launches are compiled from, beside a block's rows and routing,
    # and the bound (a Metal in ran passed it).
    proof = kernels.proof(
        "verify_gather",
        gate_up.face,
        down.face,
        tuple(a.dtype for a in gate_up.operands + down.operands),
        width,
        gate_up.n,
        down.n,
        few_pairs,
    )
    return _Plan(gate_up, down, proof, width, few_pairs)


_FAILED = "expert-ordered verify gather failed; composed fallback"


def _stand_in(plan: _Plan, metal):
    """The smallest launches of a block that MLX generates the Metal of
    ``metal`` for, over zeros. The pairs and a row's experts are template
    values, so x has the block's rows and the indices its pairs, of their
    dtypes; every pair is with expert 0 of stacks of the fewest experts that
    leave each tensor 8 elements. The down launch reads zeros of the rows
    that the activation gives, not the gate/up launch's output."""
    rows, top_k, dtype, index = metal
    pairs = rows * top_k
    stacks = [
        ((_experts(projection), *a.shape[1:]), a.dtype)
        for projection in (plan.gate_up, plan.down)
        for a in projection.operands
    ]
    specs = (
        ((pairs,), index),
        ((rows, plan.width), dtype),
        ((pairs, plan.gate_up.n // 2), dtype),
        *stacks,
    )
    cut = len(plan.gate_up.operands)

    def launch(rhs, x, hidden, *stacks):
        gate_up = plan.gate_up._replace(operands=stacks[:cut])
        down = plan.down._replace(operands=stacks[cut:])
        order = _order(rhs) if pairs >= ORDERED_PAIRS else None
        return [
            _gather_qmv(gate_up, x, rhs, top_k, order),
            _gather_qmv(down, hidden, rhs, 1, order),
        ]

    return kernels.zeros(*specs), launch


def _experts(projection: Projection) -> int:
    """The fewest experts that leave each of ``projection``'s tensors 8
    elements, as a stand-in's stacks have."""
    return max(-(-8 * a.shape[0] // a.size) for a in projection.operands)


def fused_switch(switch_mlp, x: mx.array, indices: mx.array):
    """Fused gate/up then down routed projections for 2..MAX_ROWS rows.

    Returns ``[B, L, top_k, D]`` like the verifier's gather path, or None when
    the block is outside this kernel (its rows, FEW_PAIRS pairs or fewer
    over MANY_EXPERTS experts or more, indices that are not integers or not
    top_k >= 1 of them for each row of x, x that is not float16, bfloat16
    or float32 or has fewer than 8 values), a projection has no dot face
    that is exact for ``x``'s dtype, the stacks have different expert counts,
    the pairs exceed the grid limit, the activation gives another dtype or
    shape than the up's, or a block of these faces and shapes failed.

    The first block of each Metal proves it on a stand-in (``_stand_in``):
    nothing of the call is evaluated here. Any of its arrays may belong to a
    function that MLX traces, and what fails in x, the indices, a tensor or
    the activation's arrays is raised where the caller evaluates the result.
    The activation itself runs in the call: what it raises, the call raises.
    """
    if x.ndim != 3 or indices.ndim != 3:
        return None
    batch, length, width = x.shape
    routed_batch, routed_length, top_k = indices.shape
    rows = batch * length
    # A pair is a row and one of its top_k experts.
    if not 1 < rows <= MAX_ROWS or routed_batch != batch or routed_length != length:
        return None
    plan = cached_per_module(
        switch_mlp,
        "_omlx_verify_gather_plan",
        lambda switch: _build_plan(switch, width, top_k),
        flags=(width, top_k),
    )
    if plan is None or plan.proof.failed:
        return None
    # What a block's launches are instantiated with beside the proof's
    # key. A Metal in ran passed the checks below when it was proven: a call
    # pays for neither, nor for the stand-in and the log line it would build.
    pair_count = rows * top_k
    metal = (rows, top_k, x.dtype, indices.dtype)
    if metal not in plan.proof.ran and (
        # MLX binds an input of fewer than 8 values as constants, and the
        # kernel reads x from device memory; like gather_qmm, it takes float
        # activations of the GPU and integer indices only.
        not plan.few_pairs < pair_count <= _MAX_PAIRS
        or not plan.gate_up.face.exact_for(x.dtype)
        or not plan.down.face.exact_for(x.dtype)
        or rows * width < 8
        or not mx.issubdtype(indices.dtype, mx.integer)
        or not plan.proof.prove(
            metal,
            lambda: _stand_in(plan, metal),
            logger,
            f"MoE expert-ordered verify gather engaged ({rows} rows)",
            _FAILED,
        )
    ):
        return None
    pairs = indices.reshape(pair_count)
    order = _order(pairs) if pair_count >= ORDERED_PAIRS else None
    gate_up = _gather_qmv(plan.gate_up, x.reshape(rows, width), pairs, top_k, order)
    gate, up = mx.split(gate_up, 2, axis=-1)
    hidden = switch_mlp.activation(up, gate)
    if hidden.dtype != x.dtype or hidden.shape != up.shape:
        # Not the rows that the down launch is planned and proven for.
        # Nothing is kept of it: the next block asks its activation again.
        return None
    down = _gather_qmv(plan.down, hidden, pairs, 1, order)
    return down.reshape(batch, length, top_k, -1)


def gather_qmv(
    projection: Projection,
    x: mx.array,
    rhs: mx.array,
    pairs_per_row: int,
    order: mx.array | None = None,
) -> mx.array:
    """``gather_qmm`` over P (row, expert) pairs, pairs grouped by expert.

    ``rhs`` holds the expert of each pair; pair ``p`` reads row
    ``p // pairs_per_row`` of ``x`` (``[R, K]``). ``order`` is the pairs in
    expert order (``_order``), or None for each threadgroup to find its
    pair. Returns ``[P, N]``.
    """
    if rhs.ndim != 1 or not mx.issubdtype(rhs.dtype, mx.integer):
        raise ValueError("verify gather: expert indices must be an integer vector")
    pairs = rhs.size
    if pairs > _MAX_PAIRS:
        raise ValueError("verify gather: pairs exceed the signed grid dimension")
    if pairs_per_row < 1:
        raise RuntimeError("verify gather: LHS_DIV is the experts of a row, 1 or more")
    if x.ndim != 2 or pairs % pairs_per_row or pairs > x.shape[0] * pairs_per_row:
        raise RuntimeError("verify gather: pairs are whole rows of LHS_DIV experts")
    if order is not None and (order.shape != (pairs,) or order.dtype != mx.uint32):
        raise ValueError("verify gather: order must be one uint32 per pair")
    return _gather_qmv(projection, x, rhs, pairs_per_row, order)


def _gather_qmv(projection, x, rhs, pairs_per_row, order):
    """Launch on the row and pair shapes admitted by the caller."""
    n = projection.n
    pairs = rhs.shape[0]
    template = [("T", x.dtype), ("K_SIZE", x.shape[-1]), ("N_SIZE", n)]
    if order is None:
        kernel, inputs = projection.kernel, [x, *projection.operands, rhs]
        template.append(("N_PAIRS", pairs))
    else:
        kernel, inputs = projection.ordered, [x, *projection.operands, rhs, order]
    template.append(("LHS_DIV", pairs_per_row))
    return kernel(
        inputs=inputs,
        template=template,
        grid=(32, 2 * pairs, n // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[(pairs, n)],
        output_dtypes=[x.dtype],
    )[0]
