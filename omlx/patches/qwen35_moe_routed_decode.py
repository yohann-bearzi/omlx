# SPDX-License-Identifier: Apache-2.0
"""Fused routed and shared experts for one-token Qwen3.5-MoE-family decode.

After the fused router, a one-token MoE block runs its routed experts as the
gate+up ``gather_qmm``, the compiled SwiGLU and the down ``gather_qmm`` (each
``gather_qmm`` behind an ``arange`` for its row indices), the shared expert
as three ``quantized_matmul`` launches and a SwiGLU, the shared-expert gate
as one more ``quantized_matmul``, and one combine launch for the
score-weighted sum plus the gated shared expert. At batch-one decode these
launches are short and mostly depend on each other, so this patch runs the
same arithmetic in two launches after the router:

1. gate+up with a SwiGLU epilogue, for the selected experts and the
   shared expert, plus the shared-expert gate row. Each simdgroup computes
   the gate rows and the matching up rows of one expert with MLX's
   ``qmv_fast`` or ``qmv`` lane partition and add order, rounds both to the
   activation dtype, then applies MLX's ``Sigmoid`` and the two multiplies of the
   compiled ``swiglu`` in its order. The gate row follows MLX's ordinary
   one-row QMV dispatch; its threadgroup comes first in the grid.
2. down with the combine. Simdgroup ``j`` of a threadgroup runs the stock
   ``qmv_fast`` or ``qmv`` work of selected expert ``j`` for
   the threadgroup's rows, one more simdgroup the shared expert's. Each row
   is rounded, then the combine of ``qwen35_moe_router.fused_moe_combine``
   follows: score products summed in MLX's ``col_reduce_small`` order, plus
   ``sigmoid(shared_gate) * shared``.

The router ahead of them (gate linear, precise softmax, fused top-k) runs
the gate linear as MLX's one-row gemv spread over one simdgroup per expert
(``qwen35_moe_router.router_gemv``; MLX's own launch has 32 threadgroups).
Its softmax and top-k run inside the gate+up launch where the shapes allow
(``qwen35_moe_router.softmax_topk_eligible``): the launch's first block
stores the row's selection and scores for the down launch, and every
simdgroup of an expert block recomputes the selection it serves from the
router logits with ``softmax_topk_row``'s arithmetic, so the block is three
dependent launches instead of four. Other shapes run the softmax and top-k
as their own launch (``qwen35_moe_router.softmax_topk_row``).

The result is bit-identical to the composed path. The quantized dot products
are the weight format's dot face (``omlx.formats``), one instantiation per
weight format. The affine format's is the MLX 0.32.4 transcription in
``omlx.formats.affine`` (2, 3, 4, 5, 6 and 8 bits; group size 32, 64 or
128). MLX picks ``qmv_fast`` when K is a multiple of its kernel block
(512 for 2- to 5-bit, 256 for 6/8-bit weights) and ``qmv`` otherwise. Gathered
expert projections also require N % 8 == 0; shared projections do not.
Routed experts are taken for one bf16 or fp16 token, top-k 8 or 10, experts
whose gate+up face is exact for the token's dtype, and whose down
face is exact for it (affine: scales in that dtype;
Qwen3.5/3.6-35B-A3B: hidden 2048, intermediate 512;
Qwen3.8-Flash-Next: 2560 and 640). Gate+up and down each take ``qmv_fast``
or ``qmv``. A shared expert or gate without such faces (unquantized,
packed, ...) runs as composed launches and only its outputs enter the
combine. Prefill, multi-row calls and every other shape keep the
original body.

One token's launches (router, gate+up, down) are one function of the token
and the block's arrays, which MLX's compile builds into a graph the first
time a thread meets the token's shape and dtype and the router's width on
blocks of the same faces, tensor dtypes and shapes; their proof keeps it for
that thread, and a
decode step replays it. MLX compiles a kernel when it first evaluates it, so
before the graph is first built its Metal is proven on a stand-in: the
smallest launches that MLX generates the same Metal for, over zeros, and
nothing of the call. If that fails, those blocks keep their composed body.

MTP verify windows (1..16 rows; row-exact MTP requires each row to equal
serial decode, and every armed verify runs them) run the same arithmetic for
the whole window in the same launches: the router gemv over all rows
(``qwen35_moe_router.router_gemv``), or the block's quantized gate one row at
a time through ``row_exact_qmv``, then
window variants of the launches above in which each threadgroup serves one
row with the one-token source verbatim (threadgroups of one weight block for
consecutive rows are adjacent, so shared experts are fetched once through
the cache). Up to three rows select inside the gate+up launch; from four
rows on the recomputed selections cost more than the routing launch
(``softmax_topk_rows``) they save, so it runs on its own. Every row is
bit-identical to the fused one-token call on that row. This replaces the
verifier's composed MoE (about fifteen launches, two of them binding the
whole stacked experts) in every armed verify, row-exact or not, when the
block folds its shared expert; one-row windows are one token's call. A
window's launches are one graph too, kept per thread in the window's proof
for the window's shape and dtype and the router's width; a quantized gate's
logits stay outside it, one row at a time. If its stand-in fails, the
verifier keeps its composed MoE for blocks of these faces and shapes.

MLX commits a command buffer once the inputs bound to it exceed its size cap
(50 MB by default), counting each input array whole. The stacked expert
weights are hundreds of MB, so every launch that binds them ends a command
buffer (~10-20 us of host CPU and a GPU gap per commit, and the host blocks
once too many are in flight). The kernels therefore bind a one-expert view
that shares the stacked array's buffer at offset 0 and index the other
experts from it. The weights are resident model parameters, so the cap has
nothing to bound here. Only scheduling changes.
"""

from __future__ import annotations

import logging
from functools import cache, partial, wraps
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .. import formats
from ..formats import Dot, kernels
from ..formats.linear import FormatLinear, FormatSwitchLinear
from .module_cache import cached_per_module
from .qwen35_moe_router import (
    fused_router_topk,
    router_eligible,
    router_gemv,
    softmax_topk_coalesce,
    softmax_topk_eligible,
    softmax_topk_rows,
    topk_header,
    topk_indices_header,
)
from .qwen35_verify_qmm import _is_armed as is_verify_armed
from .qwen35_verify_qmm import is_row_exact_armed
from .row_exact_qmv import quantized_linear as row_exact_linear

logger = logging.getLogger(__name__)

# The verify-window fast path (pair-coalesced gate+up and top-k, unrolled
# down) ships for the shapes it was measured to win on and no others:
# Flash-Next's 4-bit routed windows, ``(hidden, expert width, bits)``. The
# escalation measured the weighted window from 1.153x to 1.0167x and the
# served 4-prompt adaptive median to 1.0036 (M3 Ultra); every other shape
# keeps the kernels of f0968c13, because the same kernels cost on another
# family (35B-A3B drifts 1.0060-1.0176 on the M5 fan). A family that wants
# them is measured and added here.
_WINDOW_FAST_SHAPES = frozenset({(2560, 640, 4)})


# Top-k widths whose k-sum order in the combine matches MLX's reduction.
TOP_KS = (8, 10)
_GATE_UP_ROWS = 2  # gate rows (and as many up rows) per simdgroup
_GATE_UP_SIMDGROUPS = 2
_GATE_UP_COALESCE_TOPK_SIMDGROUPS = 4
_DOWN_ROWS = 4  # down rows per simdgroup unless tuned by window width
# Every gate+up simdgroup recomputes its row's selection, so the folded
# routing's ALU grows with the rows while the launch it saves does not: from
# four rows on the gate+up launch runs out of ALU headroom and the separate
# routing launch is as fast or faster (M5 Ultra, oQ5e shapes).
_TOPK_FOLD_MAX_ROWS = 3
# Pair-coalesced verify-window launches decode each shared expert block once
# for two window rows via the dot face's decode/apply contract (qmv_pair_rows).
_COALESCE_MAX_ROWS = 4
_COALESCE_MAX_PAIRS = 64
# Row-exact verify windows up to the hyper-connection and GDN verify
# kernels' 16-row ceiling; the kernels index rows from the grid.
WINDOW_MAX_ROWS = 16


def _down_rows(rows: int) -> int:
    """Down rows per simdgroup for a ``rows``-row launch. Every output row keeps
    its arithmetic under any grouping, so this only shapes the grid: one-token
    decode runs faster on more, smaller threadgroups (two rows); verify windows
    keep four, which served windows measured fastest on (M5 Ultra, oQ5e)."""
    return 2 if rows == 1 else _DOWN_ROWS


_COMMON = r"""
using namespace metal;

// MLX 0.32.3 Sigmoid, evaluated in T as the compiled swiglu does.
template <typename U>
inline U omlx_mlx_sigmoid(U x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

# The gate+up launches' epilogue. The down launches do not carry it: MLX
# hashes a kernel's whole source at every launch.
_SWIGLU = r"""
// swiglu(gate, up) of the compiled mlx-vlm activation: both rounded to T,
// silu(gate) = gate * sigmoid(gate), then times up, each op in T.
template <typename T, int RPS>
METAL_FUNC void swiglu_store(thread const float* result, device T* yp, uint simd_lid) {
  if (simd_lid == 0) {
    for (int row = 0; row < RPS; row++) {
      T g = static_cast<T>(result[row]);
      T u = static_cast<T>(result[row + RPS]);
      T t = g * omlx_mlx_sigmoid<T>(g);
      yp[row] = t * u;
    }
  }
}
"""

# One threadgroup of NSG simdgroups (each RPS gate + RPS up rows) per row
# block b = threadgroup y, blocks in order: [the shared-expert gate row, the
# NS / (NSG * RPS) shared-expert blocks,] then NI / (NSG * RPS) blocks per
# selected expert. Output y holds silu(gate) * up of the TOPK experts ([TOPK,
# NI]), then [those NS rows of the shared expert, then the gate row].
_GATE_UP_HEAD = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int ROWS = NSG * RPS;
    int b = int(threadgroup_position_in_grid.y);
    float result[2 * RPS];
"""

_GATE_UP_SHARED = r"""
    if (b == 0) {
      // shared_expert_gate: MLX's qmv for its one output row. First in the
      // grid, so its serial K walk overlaps the expert blocks.
      if (simd_gid == 0) {
        qmv_rows<T, K, 1, 0>(gt, 0, gt, 0, x, simd_lid, result);
        if (simd_lid == 0) {
          y[TOPK * NI + NS] = static_cast<T>(result[0]);
        }
      }
      return;
    }
    if (b <= NS / ROWS) {
      const int out_row = (b - 1) * ROWS + int(simd_gid) * RPS;
      qmv_rows<T, K, RPS, RPS>(sg, out_row, su, out_row, x, simd_lid, result);
      swiglu_store<T, RPS>(result, y + TOPK * NI + out_row, simd_lid);
      return;
    }
    b -= 1 + NS / ROWS;
"""

_GATE_UP_ROUTED = r"""
    const int slot = b / (NI / ROWS);
    const int out_row = (b % (NI / ROWS)) * ROWS + int(simd_gid) * RPS;
    const size_t expert = size_t(rhs[slot]);
    qmv_rows<T, K, RPS, RPS>(
        rt, expert * (2 * NI) + out_row, rt, expert * (2 * NI) + NI + out_row,
        x, simd_lid, result);
    swiglu_store<T, RPS>(result, y + size_t(slot) * NI + out_row, simd_lid);
"""

# Threadgroup (32, NPART): simdgroup j < TOPK computes RPS rows of selected
# expert j (simdgroup TOPK the shared expert's), then simdgroup 0 combines.
# Output is [N].
_DOWN_HEAD = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int out_row = int(tid.y) * RPS;
    const int slot = int(simd_gid);
    threadgroup T part[NPART * RPS];
    float result[RPS];
"""

_DOWN_SHARED_ROWS = r"""
    if (slot == TOPK) {
      qmv_rows<T, KS, RPS, 0>(sd, out_row, sd, out_row, x + TOPK * K, simd_lid, result);
    } else
"""

_DOWN_TAIL = r"""
    {
      const size_t expert = size_t(rhs[slot]);
      qmv_rows<T, K, RPS, 0>(
          rd, expert * N + out_row, rd, expert * N + out_row,
          x + slot * K, simd_lid, result);
    }
    if (simd_lid == 0) {
      for (int row = 0; row < RPS; row++) {
        part[slot * RPS + row] = static_cast<T>(result[row]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0 && int(simd_lid) < RPS) {
      // fused_moe_combine's arithmetic: each product and each add rounds to
      // T; the k-sum is MLX's one-row col_reduce_small (lane j % 8 folds
      // rows j, j + 8 onto +0, then lanes 1..7 add onto lane 0 in order).
      const int h = out_row + int(simd_lid);
      T lane[8];
      for (int l = 0; l < 8; ++l) {
        lane[l] = T(0.0f);
      }
      for (int j = 0; j < TOPK; ++j) {
        const T p = T(float(part[j * RPS + int(simd_lid)]) * float(scores[j]));
        lane[j % 8] = T(float(p) + float(lane[j % 8]));
      }
      T acc = lane[0];
      for (int l = 1; l < 8; ++l) {
        acc = T(float(lane[l]) + float(acc));
      }
      T sg;
      if constexpr (metal::is_same<T, half>::value) {
        sg = omlx_mlx_sigmoid<T>(GATE_VALUE);
      } else {
        const float g = float(GATE_VALUE);
        const T e = T(1.0f + float(T(metal::precise::exp(metal::abs(g)))));
        const T sy = T(metal::precise::divide(1.0f, float(e)));
        sg = g < 0.0f ? sy : T(1.0f - float(sy));
      }
      const T sh = T(float(sg) * float(SHARED_VALUE));
      y[h] = T(float(acc) + float(sh));
    }
"""

# Verify-window variants: row ``window_row`` of an M-row window runs the
# one-token source verbatim on its own input, routing and output (the kernel
# pointers are rebound in an inner scope). Threadgroup y is
# ``block * M + window_row``, so the rows of one weight block run back to
# back and share its fetch through the cache.
_GATE_UP_WINDOW_HEAD = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int ROWS = NSG * RPS;
    const int window_row = int(threadgroup_position_in_grid.y) % M;
    int b = int(threadgroup_position_in_grid.y) / M;
    float result[2 * RPS];
    const auto x_row = x + window_row * K;
    const auto rhs_row = rhs + window_row * TOPK;
    const auto y_row = y + window_row * (TOPK * NI + NS + 1);
    {
    const auto x = x_row;
    const auto rhs = rhs_row;
    const auto y = y_row;
"""

_DOWN_WINDOW_HEAD = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int window_row = int(tid.y) % M;
    const int out_row = int(tid.y) / M * RPS;
    const int slot = int(simd_gid);
    threadgroup T part[NPART * RPS];
    float result[RPS];
    const auto x_row = x + window_row * (TOPK * K + KS + 1);
    const auto rhs_row = rhs + window_row * TOPK;
    const auto scores_row = scores + window_row * TOPK;
    const auto y_row = y + window_row * N;
    {
    const auto x = x_row;
    const auto rhs = rhs_row;
    const auto scores = scores_row;
    const auto y = y_row;
"""

# Gate+up with the softmax + top-k folded in, for one-token decode (M = 1)
# and verify windows alike: threadgroup y = block * M + window_row serves row
# window_row with the blocks above, whose first block also selects that
# row's experts: its simdgroup 1 stores the TOPK experts and their scores
# from the row's router logits (softmax_topk_row's outputs, which the down
# launch reads). Every simdgroup of an expert block recomputes selection
# ``slot`` from the logits with the same arithmetic before streaming its
# rows, so no launch sits between the router gemv and the experts.
_GATE_UP_TOPK_HEAD = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int ROWS = NSG * RPS;
    const int window_row = int(threadgroup_position_in_grid.y) % M;
    int b = int(threadgroup_position_in_grid.y) / M;
    float result[2 * RPS];
    const auto x_row = x + window_row * K;
    const auto logits_row = logits + window_row * NE;
    const auto indices_row = indices + window_row * TOPK;
    const auto scores_row = scores + window_row * TOPK;
    const auto y_row = y + window_row * YW;
    {
    const auto x = x_row;
    const auto logits = logits_row;
    const auto indices = indices_row;
    const auto scores = scores_row;
    const auto y = y_row;
    if (b == 0 && simd_gid == 1) {
      omlx_router_topk<T, NE>::template write<TOPK>(logits, simd_lid, indices, scores);
      return;
    }
"""

# Without a folded shared expert the first block only selects.
_GATE_UP_TOPK_UNSHARED = r"""
    if (b == 0) {
      return;
    }
    b -= 1;
"""

_GATE_UP_TOPK_ROUTED = _GATE_UP_ROUTED.replace(
    "size_t(rhs[slot])", "size_t(omlx_router_topk<T, NE>::nth(logits, simd_lid, slot))"
)

# Pair-coalesced verify-window gate+up launch (after softmax_topk_coalesce):
# shared-expert rows are paired across consecutive window rows and routed
# experts are paired per meta[g] = (expert, count in {1, 2}, p0, p1), decoding
# each block once via qmv_pair_rows and applying it to both rows with the exact
# one-token arithmetic.
_GATE_UP_WINDOW_COALESCED = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int ROWS = NSG * RPS;
    constexpr int YW = TOPK * NI + NS + 1;
    int b = int(threadgroup_position_in_grid.y);
    float r0[2 * RPS], r1[2 * RPS];
    if (b == 0) {
      for (int m = int(simd_gid); m < M; m += NSG) {
        qmv_rows<T, K, 1, 0>(gt, 0, gt, 0, x + m * K, simd_lid, r0);
        if (simd_lid == 0) {
          y[m * YW + TOPK * NI + NS] = static_cast<T>(r0[0]);
        }
      }
      return;
    }
    b -= 1;
    constexpr int SH_TILES = NS / ROWS;
    constexpr int SH_PAIRS = (M + 1) / 2;
    if (b < SH_PAIRS * SH_TILES) {
      const int pair = b % SH_PAIRS;
      const int tile = b / SH_PAIRS;
      const int out_row = tile * ROWS + int(simd_gid) * RPS;
      const int m0 = pair * 2;
      const int m1 = m0 + 1;
      if (m1 < M) {
        qmv_pair_rows<T, K, RPS>(
            sg, out_row, su, out_row, x + m0 * K, x + m1 * K, simd_lid, r0, r1);
        swiglu_store<T, RPS>(r0, y + m0 * YW + TOPK * NI + out_row, simd_lid);
        swiglu_store<T, RPS>(r1, y + m1 * YW + TOPK * NI + out_row, simd_lid);
      } else {
        qmv_rows<T, K, RPS, RPS>(sg, out_row, su, out_row, x + m0 * K, simd_lid, r0);
        swiglu_store<T, RPS>(r0, y + m0 * YW + TOPK * NI + out_row, simd_lid);
      }
      return;
    }
    b -= SH_PAIRS * SH_TILES;
    constexpr int RT_TILES = NI / ROWS;
    const uint g = uint(b / RT_TILES);
    if (g >= ucount[0]) {
      return;
    }
    const int out_row = (b % RT_TILES) * ROWS + int(simd_gid) * RPS;
    const size_t e = size_t(meta[g * 4 + 0]);
    const uint cnt = meta[g * 4 + 1];
    const uint p0 = meta[g * 4 + 2];
    const int m0 = int(p0 / uint(TOPK));
    const int k0 = int(p0 % uint(TOPK));
    if (cnt == 2u) {
      const uint p1 = meta[g * 4 + 3];
      const int m1 = int(p1 / uint(TOPK));
      const int k1 = int(p1 % uint(TOPK));
      qmv_pair_rows<T, K, RPS>(
          rt, e * (2 * NI) + out_row, rt, e * (2 * NI) + NI + out_row,
          x + m0 * K, x + m1 * K, simd_lid, r0, r1);
      swiglu_store<T, RPS>(r0, y + m0 * YW + size_t(k0) * NI + out_row, simd_lid);
      swiglu_store<T, RPS>(r1, y + m1 * YW + size_t(k1) * NI + out_row, simd_lid);
    } else {
      qmv_rows<T, K, RPS, RPS>(
          rt, e * (2 * NI) + out_row, rt, e * (2 * NI) + NI + out_row,
          x + m0 * K, simd_lid, r0);
      swiglu_store<T, RPS>(r0, y + m0 * YW + size_t(k0) * NI + out_row, simd_lid);
    }
"""

# Folded 3-launch pair-coalesced verify-window gate+up launch (no separate
# routing launch): each routed threadgroup computes all M rows' top-k indices
# into threadgroup memory in parallel across its NSG simdgroups, pairs
# duplicate experts across rows via qmv_pair_rows, and returns immediately on
# the second member of each pair.
_GATE_UP_TOPK_COALESCED = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int ROWS = NSG * RPS;
    constexpr int YW = TOPK * NI + NS + 1;
    int b = int(threadgroup_position_in_grid.y);
    float r0[2 * RPS], r1[2 * RPS];
    if (b < M) {
      const int row = b;
      if (simd_gid == 0) {
        qmv_rows<T, K, 1, 0>(gt, 0, gt, 0, x + row * K, simd_lid, r0);
        if (simd_lid == 0) {
          y[row * YW + TOPK * NI + NS] = static_cast<T>(r0[0]);
        }
      } else if (simd_gid == 1) {
        omlx_router_topk<T, NE>::template write<TOPK>(
            logits + row * NE, simd_lid, indices + row * TOPK, scores + row * TOPK);
      }
      return;
    }
    b -= M;
    constexpr int SH_TILES = NS / ROWS;
    constexpr int SH_PAIRS = (M + 1) / 2;
    if (b < SH_PAIRS * SH_TILES) {
      const int pair = b % SH_PAIRS;
      const int tile = b / SH_PAIRS;
      const int out_row = tile * ROWS + int(simd_gid) * RPS;
      const int m0 = pair * 2;
      const int m1 = m0 + 1;
      if (m1 < M) {
        qmv_pair_rows<T, K, RPS>(
            sg, out_row, su, out_row, x + m0 * K, x + m1 * K, simd_lid, r0, r1);
        swiglu_store<T, RPS>(r0, y + m0 * YW + TOPK * NI + out_row, simd_lid);
        swiglu_store<T, RPS>(r1, y + m1 * YW + TOPK * NI + out_row, simd_lid);
      } else {
        qmv_rows<T, K, RPS, RPS>(sg, out_row, su, out_row, x + m0 * K, simd_lid, r0);
        swiglu_store<T, RPS>(r0, y + m0 * YW + TOPK * NI + out_row, simd_lid);
      }
      return;
    }
    b -= SH_PAIRS * SH_TILES;
    constexpr int RT_TILES = NI / ROWS;
    const int p = b / RT_TILES;
    const int tile = b % RT_TILES;
    const int m = p / TOPK;
    const int k = p % TOPK;
    threadgroup uint tg_ind[M * TOPK];
    for (int r = int(simd_gid); r < M; r += NSG) {
      omlx_router_topk_indices_tg<T, NE, TOPK>(
          logits + r * NE, simd_lid, tg_ind + r * TOPK);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint e = tg_ind[m * TOPK + k];
    int prev = 0;
    if (e < uint(NE)) {
      for (int r = 0; r < m; ++r) {
        for (int j = 0; j < TOPK; ++j) {
          if (tg_ind[r * TOPK + j] == e) {
            ++prev;
            break;
          }
        }
      }
      if ((prev & 1) == 1) {
        return;
      }
    }
    int m2 = -1, k2 = -1;
    if (e < uint(NE)) {
      for (int r = m + 1; r < M && m2 < 0; ++r) {
        for (int j = 0; j < TOPK; ++j) {
          if (tg_ind[r * TOPK + j] == e) {
            m2 = r;
            k2 = j;
            break;
          }
        }
      }
    }
    const int out_row = tile * ROWS + int(simd_gid) * RPS;
    if (m2 >= 0) {
      qmv_pair_rows<T, K, RPS>(
          rt, size_t(e) * (2 * NI) + out_row, rt, size_t(e) * (2 * NI) + NI + out_row,
          x + m * K, x + m2 * K, simd_lid, r0, r1);
      swiglu_store<T, RPS>(r0, y + m * YW + size_t(k) * NI + out_row, simd_lid);
      swiglu_store<T, RPS>(r1, y + m2 * YW + size_t(k2) * NI + out_row, simd_lid);
    } else {
      qmv_rows<T, K, RPS, RPS>(
          rt, size_t(e) * (2 * NI) + out_row, rt, size_t(e) * (2 * NI) + NI + out_row,
          x + m * K, simd_lid, r0);
      swiglu_store<T, RPS>(r0, y + m * YW + size_t(k) * NI + out_row, simd_lid);
    }
"""


def _add_shared_gate_up(
    matrices: kernels.Matrices, shared: Dot, gate: Dot
) -> list[str]:
    """The shared expert's gate (``sg``) and up (``su``) rows and the
    shared-expert gate row (``gt``)."""
    return (
        matrices.add("sg", shared)
        + matrices.add("su", shared)
        + matrices.add("gt", gate)
    )


@cache
def _gate_up_kernel(routed: Dot, shared: Dot | None, gate: Dot | None):
    """Gate+up/SwiGLU launch over the stacked gate+up rows (``rt``); with
    ``shared`` (and ``gate``) it also runs the shared expert's gate+up and
    the shared-expert gate row."""
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("rt", routed), "rhs"]
    source = _GATE_UP_HEAD
    name = f"omlx_qwen35_moe_gate_up_decode_{routed.key}"
    if shared is not None:
        inputs += _add_shared_gate_up(matrices, shared, gate)
        source += _GATE_UP_SHARED
        name += f"_shared_{shared.key}_gate_{gate.key}"
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=["y"],
        header=_COMMON + _SWIGLU + matrices.header,
        source=matrices.declarations + source + _GATE_UP_ROUTED,
    )


@cache
def _down_kernel(routed: Dot, shared: Dot | None):
    """Down/combine launch over the stacked down rows (``rd``); with
    ``shared`` it also runs the shared expert's down rows (``sd``),
    otherwise the shared expert's output is an input."""
    matrices = kernels.Matrices(unroll_pairs=True)
    routed_inputs = matrices.add("rd", routed)
    name = f"omlx_qwen35_moe_down_combine_decode_{routed.key}"
    if shared is None:
        inputs = ["shared", "gate", "x", *routed_inputs, "rhs", "scores"]
        source = _DOWN_HEAD + _DOWN_TAIL.replace("SHARED_VALUE", "shared[h]").replace(
            "GATE_VALUE", "gate[0]"
        )
    else:
        # x is the gate+up output: expert rows, shared rows, gate row.
        inputs = ["x", *routed_inputs, *matrices.add("sd", shared), "rhs", "scores"]
        source = (
            _DOWN_HEAD
            + _DOWN_SHARED_ROWS
            + _DOWN_TAIL.replace(
                "SHARED_VALUE", "part[TOPK * RPS + int(simd_lid)]"
            ).replace("GATE_VALUE", "x[TOPK * K + KS]")
        )
        name += f"_shared_{shared.key}"
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=["y"],
        header=_COMMON + matrices.header,
        source=matrices.declarations + source,
    )


@cache
def _gate_up_window_kernel(routed: Dot, shared: Dot, gate: Dot):
    """The folded gate+up/SwiGLU launch for every row of a verify window."""
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("rt", routed), "rhs"]
    inputs += _add_shared_gate_up(matrices, shared, gate)
    return mx.fast.metal_kernel(
        name=(
            f"omlx_qwen35_moe_gate_up_window_{routed.key}"
            f"_shared_{shared.key}_gate_{gate.key}"
        ),
        input_names=inputs,
        output_names=["y"],
        header=_COMMON + _SWIGLU + matrices.header,
        source=(
            matrices.declarations
            + _GATE_UP_WINDOW_HEAD
            + _GATE_UP_SHARED
            + _GATE_UP_ROUTED
            + "    }\n"
        ),
    )


@cache
def _down_window_kernel(routed: Dot, shared: Dot):
    """The folded down/combine launch for every row of a verify window."""
    matrices = kernels.Matrices()
    inputs = [
        "x",
        *matrices.add("rd", routed),
        *matrices.add("sd", shared),
        "rhs",
        "scores",
    ]
    tail = _DOWN_TAIL.replace(
        "SHARED_VALUE", "part[TOPK * RPS + int(simd_lid)]"
    ).replace("GATE_VALUE", "x[TOPK * K + KS]")
    return mx.fast.metal_kernel(
        name=f"omlx_qwen35_moe_down_combine_window_{routed.key}_shared_{shared.key}",
        input_names=inputs,
        output_names=["y"],
        header=_COMMON + matrices.header,
        source=matrices.declarations
        + _DOWN_WINDOW_HEAD
        + _DOWN_SHARED_ROWS
        + tail
        + "    }\n",
    )


@cache
def _gate_up_topk_kernel(
    routed: Dot, shared: Dot | None, gate: Dot | None, dtype: mx.Dtype
):
    """Gate+up/SwiGLU with the router softmax + top-k folded in, for M rows;
    with ``shared`` (and ``gate``) it also runs the shared expert's gate+up
    and the shared-expert gate row."""
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("rt", routed), "logits"]
    source = _GATE_UP_TOPK_HEAD
    # The selection is compiled for the activation dtype alone (topk_header).
    selection = "bf16" if dtype == mx.bfloat16 else "scan"
    name = f"omlx_qwen35_moe_gate_up_topk_{selection}_{routed.key}"
    if shared is None:
        source += _GATE_UP_TOPK_UNSHARED
    else:
        inputs += _add_shared_gate_up(matrices, shared, gate)
        source += _GATE_UP_SHARED
        name += f"_shared_{shared.key}_gate_{gate.key}"
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=["y", "indices", "scores"],
        header=_COMMON + _SWIGLU + topk_header(dtype) + matrices.header,
        source=matrices.declarations + source + _GATE_UP_TOPK_ROUTED + "    }\n",
    )


@cache
def _gate_up_window_coalesced_kernel(routed: Dot, shared: Dot, gate: Dot):
    """Pair-coalesced verify-window gate+up/SwiGLU launch reading expert pairs
    from ``softmax_topk_coalesce`` and applying each decoded block to two
    window rows via ``qmv_pair_rows``."""
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("rt", routed, pair=True), "meta", "ucount"]
    inputs += (
        matrices.add("sg", shared, pair=True)
        + matrices.add("su", shared, pair=True)
        + matrices.add("gt", gate)
    )
    return mx.fast.metal_kernel(
        name=(
            f"omlx_qwen35_moe_gate_up_window_coalesced_{routed.key}"
            f"_shared_{shared.key}_gate_{gate.key}"
        ),
        input_names=inputs,
        output_names=["y"],
        header=_COMMON + _SWIGLU + matrices.header,
        source=matrices.declarations + _GATE_UP_WINDOW_COALESCED,
    )


@cache
def _gate_up_topk_coalesced_kernel(
    routed: Dot, shared: Dot, gate: Dot, dtype: mx.Dtype
):
    """Folded 3-launch pair-coalesced verify-window gate+up/SwiGLU launch."""
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("rt", routed, pair=True), "logits"]
    inputs += (
        matrices.add("sg", shared, pair=True)
        + matrices.add("su", shared, pair=True)
        + matrices.add("gt", gate)
    )
    selection = "bf16" if dtype == mx.bfloat16 else "scan"
    return mx.fast.metal_kernel(
        name=(
            f"omlx_qwen35_moe_gate_up_topk_coalesced_{selection}_{routed.key}"
            f"_shared_{shared.key}_gate_{gate.key}"
        ),
        input_names=inputs,
        output_names=["y", "indices", "scores"],
        header=_COMMON + _SWIGLU + topk_indices_header(dtype) + matrices.header,
        source=matrices.declarations + _GATE_UP_TOPK_COALESCED,
    )


@cache
def _down_window_unrolled_kernel(routed: Dot, shared: Dot):
    """Verify-window down/combine launch with K-loop pairs unrolled."""
    # One unroll mechanism, the framework's: Matrices(unroll_pairs=True)
    # hands this kernel the rows skeleton with its K-loop pairs unrolled.
    matrices = kernels.Matrices(unroll_pairs=True)
    inputs = [
        "x",
        *matrices.add("rd", routed),
        *matrices.add("sd", shared),
        "rhs",
        "scores",
    ]
    shared_rows = _DOWN_SHARED_ROWS
    tail = _DOWN_TAIL.replace(
        "SHARED_VALUE", "part[TOPK * RPS + int(simd_lid)]"
    ).replace("GATE_VALUE", "x[TOPK * K + KS]")
    return mx.fast.metal_kernel(
        name=(
            f"omlx_qwen35_moe_down_combine_window_unrolled_{routed.key}"
            f"_shared_{shared.key}"
        ),
        input_names=inputs,
        output_names=["y"],
        header=_COMMON + matrices.header,
        source=matrices.declarations
        + _DOWN_WINDOW_HEAD
        + shared_rows
        + tail
        + "    }\n",
    )


def _plain(layer, *classes) -> bool:
    # Exactly a class whose call is a bare quantized_matmul / gather_qmm, or
    # a quant format's rows over its dot face; subclasses and repacked layers
    # may compute differently.
    return type(layer) in classes and "bias" not in layer


def _window_router_ok(gate, dtype: mx.Dtype) -> bool:
    """Whether a verify window takes the block's quantized gate through
    ``row_exact_linear``: exactly a plain quantized linear of an affine
    layout that tiles (its other layouts fall back to one stock call a
    row), without a bias, whose scales and biases are the dtype the
    launches are compiled for (a mixed dtype would promote in the tile and
    every window would decline after its gate work). These are the affine
    widths and group sizes ``row_exact_qmv`` tiles."""
    return (
        _plain(gate, nn.QuantizedLinear)
        and "biases" in gate
        and gate.mode == "affine"
        and gate.bits in (4, 5, 6, 8)
        and gate.group_size in (32, 64, 128)
        and gate["scales"].dtype == dtype
        and gate["biases"].dtype == gate["scales"].dtype
    )


def _operands(layer, face: Dot) -> tuple:
    names = formats.tensor_names(layer)
    return tuple(layer[name] for name in (names[0], *face.tensors))


def _address(a: mx.array) -> int:
    return np.frombuffer(memoryview(a), dtype=np.uint8).ctypes.data


def _expert_view(a: mx.array) -> mx.array:
    """``a[:1]`` when it shares ``a``'s buffer at offset 0, else ``a``.

    The kernels index every expert from the bound pointer, so a view that
    were copied (or offset) would read the wrong bytes; keep the whole array
    unless the first-expert slice provably aliases it. An expert of fewer
    than 8 elements keeps the whole array too: MLX binds so small an input
    as constants, not as the buffer a face reads. So does a placeholder of
    a function that MLX compiles over the block's state: MLX refuses to
    evaluate it (a ``ValueError``), and it has no buffer to share."""
    if a.ndim < 2 or a.size < 8 * a.shape[0]:
        return a
    view = a[:1]
    try:
        mx.eval(view)
    except ValueError:
        return a
    whole, first = memoryview(a), memoryview(view)
    if whole.c_contiguous and first.c_contiguous and _address(view) == _address(a):
        return view
    return a


class _Launches(NamedTuple):
    """What a block's launches are built from beside its arrays: kernels,
    template values, grids. It holds no array, so a graph built over it
    serves every block of its proof, each over its own arrays."""

    hidden: int
    dtype: mx.Dtype  # the token's dtype; both routed faces and any folded
    # shared faces are exact for it
    top_k: int
    fold: bool  # shared expert and its gate run inside the two launches
    router: object  # router_gemv's launcher; None: the block's gate runs as it is
    gate_up_kernel: object
    gate_up_template: list
    gate_up_grid: tuple
    gate_up_output: tuple
    down_kernel: object
    down_template: list
    down_threadgroup: tuple
    window_gate_up_kernel: object = None  # verify-window launches (fold)
    window_down_kernel: object = None  # the rolled down kernel (not built when fast)
    window_down: object = None  # the down kernel the window launches
    window_coalesced_kernel: object = None
    window_topk_coalesced_kernel: object = None
    window_down_unrolled_kernel: object = None
    window_inter: int = 0
    window_shared_width: int = 0
    window_fast: bool = False  # the measured fast path (for the plan's report)
    window_coalesced: bool = False  # launches.window_coalesced_kernel runs
    topk_kernel: object = None  # gate+up with the router top-k folded in
    topk_blocks: int = 0  # its blocks per row
    topk_width: int = 0  # its gate+up output per row


class _Plan(NamedTuple):
    launches: _Launches
    # The arrays the launches read: the router's weight (none when the gate
    # runs as it is), the routed gate+up, the routed down, then the shared
    # expert's gate+up with its gate and its down (none unless folded).
    operands: tuple
    experts: int  # the router's width: the experts it scores, the block's own
    outside: bool  # the block computes a router or shared expert outside them
    proof: kernels.Proof  # one token's launches; failed: nothing of the plan runs
    window: kernels.Proof | None = None  # theirs; failed: one token still runs
    quantized_router: bool = False  # a verify window's routing is the gate's rows


def _projection_dot(layer, k: int, n: int) -> Dot | None:
    # A checkpoint reader can describe the face of the layer it holds
    # without checking the dimensions the routed kernel is about to use.
    # On unmutated routed switch layers N comes from the same tensor, so K
    # refuses there; shared projections refuse on K or N.
    if isinstance(layer, (FormatLinear, FormatSwitchLinear)) and (
        layer.input_dims != k or layer.output_dims != n
    ):
        return None
    return formats.dot(layer, k, n)


def _shared_formats(block, hidden: int, dtype: mx.Dtype):
    """(gate+up, down, gate) dot faces and operands of a foldable shared
    expert and gate, or None."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    shared, gate = block.get("shared_expert"), block.get("shared_expert_gate")
    if type(shared) is not Qwen3_5MLP:
        return None
    layers = (
        shared.get("gate_proj"),
        shared.get("up_proj"),
        shared.get("down_proj"),
        gate,
    )
    if not all(_plain(layer, nn.QuantizedLinear, FormatLinear) for layer in layers):
        return None
    gate_proj, up_proj, down_proj, _ = layers
    width = gate_proj[formats.tensor_names(gate_proj)[0]].shape[0]
    gu_fmt = _projection_dot(gate_proj, hidden, width)
    d_fmt = _projection_dot(down_proj, width, hidden)
    g_fmt = _projection_dot(gate, hidden, 1)
    if (
        gu_fmt is None
        or d_fmt is None
        or g_fmt is None
        or _projection_dot(up_proj, hidden, width) != gu_fmt
        or not all(face.exact_for(dtype) for face in (gu_fmt, d_fmt, g_fmt))
        or width % (_GATE_UP_ROWS * _GATE_UP_SIMDGROUPS)
    ):
        return None
    return (
        width,
        gu_fmt,
        d_fmt,
        g_fmt,
        _operands(gate_proj, gu_fmt)
        + _operands(up_proj, gu_fmt)
        + _operands(gate, g_fmt),
        _operands(down_proj, d_fmt),
    )


def _window_fast(gate_up, gu_fmt, hidden: int, inter: int) -> bool:
    """Whether a folded block takes the measured verify-window fast path: a
    table-backed routed gate+up face and a ``(hidden, inter, bits)`` the
    allowlist names. The bits are the layer's own (never the face's name),
    and the allowlist is the only width filter: adding a measured width is
    editing the tuple, not a second condition."""
    bits = getattr(gate_up, "bits", None)
    return (
        bool(gu_fmt.tables)
        and type(bits) is int
        and (hidden, inter, bits) in _WINDOW_FAST_SHAPES
    )


def _build_plan(block, dtype: mx.Dtype = mx.bfloat16) -> _Plan | None:
    """Kernels and operands for one MoE block, or None outside the layout."""
    from mlx_vlm.models.switch_layers import QuantizedSwitchLinear, SwiGLU

    switch_mlp = block.get("switch_mlp")
    if dtype not in (mx.bfloat16, mx.float16):
        return None
    if block.training or switch_mlp is None or block.top_k not in TOP_KS:
        return None
    if type(switch_mlp.get("activation")) is not SwiGLU:
        return None
    gate_up = switch_mlp.get("gate_up_proj")
    down = switch_mlp.get("down_proj")
    if not (
        _plain(gate_up, QuantizedSwitchLinear, FormatSwitchLinear)
        and _plain(down, QuantizedSwitchLinear, FormatSwitchLinear)
    ):
        return None
    hidden = down[formats.tensor_names(down)[0]].shape[1]
    width = gate_up[formats.tensor_names(gate_up)[0]].shape[1]
    if width % 2:
        return None
    inter = width // 2
    gu_fmt = _projection_dot(gate_up, hidden, width)
    d_fmt = _projection_dot(down, inter, hidden)
    rows = _GATE_UP_ROWS * _GATE_UP_SIMDGROUPS
    if (
        gu_fmt is None
        or d_fmt is None
        or gate_up[formats.tensor_names(gate_up)[0]].shape[0]
        != down[formats.tensor_names(down)[0]].shape[0]
        or not gu_fmt.exact_for(dtype)
        or not d_fmt.exact_for(dtype)
        # The launches cover whole groups of rows.
        or inter % rows
        or any(hidden % _down_rows(window) for window in range(1, WINDOW_MAX_ROWS + 1))
    ):
        return None
    gate, experts = block.get("gate"), block.num_experts
    scores = (
        gate.get(formats.tensor_names(gate)[0]) if isinstance(gate, nn.Module) else None
    )
    # The router scores the experts the stacks hold, and the block says so.
    if gate_up[formats.tensor_names(gate_up)[0]].shape[0] != experts or (
        isinstance(scores, mx.array) and scores.shape[:1] != (experts,)
    ):
        return None
    top_k = block.top_k
    gate_up_operands = tuple(_expert_view(a) for a in _operands(gate_up, gu_fmt))
    down_operands = tuple(_expert_view(a) for a in _operands(down, d_fmt))
    shared = _shared_formats(block, hidden, dtype)
    if shared is not None and (
        kernels.table_bytes(gu_fmt, shared[1], shared[3]) > formats.TABLE_BYTES
        or kernels.table_bytes(d_fmt, shared[2]) > formats.TABLE_BYTES
    ):
        # One launch would hold more tables than a kernel holds: the shared
        # expert keeps its own launches.
        shared = None
    router, weight = None, ()
    if (
        type(gate) is nn.Linear
        and "bias" not in gate
        and gate["weight"].shape[-1] == hidden
        and gate["weight"].dtype == dtype
    ):
        router = router_gemv(gate["weight"])
    if router is not None:
        weight = (gate["weight"],)
    # A quantized gate's verify rows run its own one-row arithmetic
    # (``row_exact_linear``) outside the window's launches.
    quantized_router = router is None and _window_router_ok(gate, dtype)
    # What the launches are compiled from, beside a launch's rows and routing.
    operands = gate_up_operands + down_operands
    if shared is not None:
        operands += shared[4] + shared[5]
    key = (
        gu_fmt,
        d_fmt,
        shared[:4] if shared is not None else None,
        tuple(a.dtype for a in operands),
        hidden,
        inter,
        top_k,
        router is not None,
        dtype,
    )
    proof = kernels.proof("routed_decode", *key)
    gate_up_template = [
        ("T", dtype),
        ("K", hidden),
        ("NI", inter),
        ("RPS", _GATE_UP_ROWS),
        ("NSG", _GATE_UP_SIMDGROUPS),
        ("TOPK", top_k),
    ]
    down_template = [
        ("T", dtype),
        ("K", inter),
        ("N", hidden),
        ("TOPK", top_k),
    ]
    if shared is None:
        launches = _Launches(
            hidden=hidden,
            dtype=dtype,
            top_k=top_k,
            router=router,
            fold=False,
            gate_up_kernel=_gate_up_kernel(gu_fmt, None, None),
            gate_up_template=gate_up_template,
            gate_up_grid=(32, _GATE_UP_SIMDGROUPS * top_k * inter // rows, 1),
            gate_up_output=(top_k, inter),
            down_kernel=_down_kernel(d_fmt, None),
            down_template=down_template + [("NPART", top_k)],
            down_threadgroup=(32, top_k, 1),
            topk_kernel=_gate_up_topk_kernel(gu_fmt, None, None, dtype),
            topk_blocks=1 + top_k * inter // rows,
            topk_width=top_k * inter,
        )
        return _Plan(
            launches=launches,
            operands=(weight, gate_up_operands, down_operands, (), ()),
            experts=experts,
            outside=True,
            proof=proof,
        )
    width, sgu_fmt, sd_fmt, g_fmt, shared_gate_up, shared_down = shared
    can_coalesce = gu_fmt.vpt >= 1 and sgu_fmt.vpt >= 1
    # The fast path is decided once, from the layer's own bit width (never the
    # face's name: a key is a name) and the shapes it was measured on. Its
    # kernels are built only inside this decision: a block that never launches
    # them constructs no kernel object for them, which would otherwise move the
    # per-process placement draw of a path whose launches did not change.
    window_fast = _window_fast(gate_up, gu_fmt, hidden, inter)
    window_coalesced = can_coalesce and window_fast
    window_down_unrolled = (
        _down_window_unrolled_kernel(d_fmt, sd_fmt) if window_fast else None
    )
    window_down = (
        window_down_unrolled if window_fast else _down_window_kernel(d_fmt, sd_fmt)
    )
    launches = _Launches(
        hidden=hidden,
        dtype=dtype,
        top_k=top_k,
        router=router,
        fold=True,
        gate_up_kernel=_gate_up_kernel(gu_fmt, sgu_fmt, g_fmt),
        gate_up_template=gate_up_template + [("NS", width)],
        gate_up_grid=(
            32,
            _GATE_UP_SIMDGROUPS * (1 + width // rows + top_k * inter // rows),
            1,
        ),
        gate_up_output=(top_k * inter + width + 1,),
        down_kernel=_down_kernel(d_fmt, sd_fmt),
        down_template=down_template + [("KS", width), ("NPART", top_k + 1)],
        down_threadgroup=(32, top_k + 1, 1),
        window_gate_up_kernel=_gate_up_window_kernel(gu_fmt, sgu_fmt, g_fmt),
        # The window launch reads the chosen callable; the rolled kernel is
        # not built when the unrolled one runs.
        window_down_kernel=None if window_fast else window_down,
        window_down=window_down,
        window_coalesced_kernel=(
            _gate_up_window_coalesced_kernel(gu_fmt, sgu_fmt, g_fmt)
            if window_coalesced
            else None
        ),
        window_topk_coalesced_kernel=(
            _gate_up_topk_coalesced_kernel(gu_fmt, sgu_fmt, g_fmt, dtype)
            if window_coalesced
            else None
        ),
        window_down_unrolled_kernel=window_down_unrolled,
        window_inter=inter,
        window_shared_width=width,
        window_fast=window_fast,
        window_coalesced=window_coalesced,
        topk_kernel=_gate_up_topk_kernel(gu_fmt, sgu_fmt, g_fmt, dtype),
        topk_blocks=1 + width // rows + top_k * inter // rows,
        topk_width=top_k * inter + width + 1,
    )
    return _Plan(
        launches=launches,
        operands=(weight, gate_up_operands, down_operands, shared_gate_up, shared_down),
        experts=experts,
        outside=router is None,
        proof=proof,
        window=kernels.proof("routed_window", *key, window_fast),
        quantized_router=quantized_router,
    )


def _routes(plan: _Plan, x) -> bool:
    """Whether one token ``x`` runs through the plan's launches: a row of
    the plan's width, in the dtype they are compiled for, for a router of
    whole simdgroups, and none of their proof's launches failed."""
    launches = plan.launches
    return (
        x.dtype == launches.dtype
        and x.shape[-1] == launches.hidden
        and not plan.proof.failed
        and router_eligible(x, plan.experts)
    )


def _gate_up(launches: _Launches, x, indices, gate_up, shared_gate_up):
    """The one-token gate+up/SwiGLU launch on the routing ``indices``."""
    return launches.gate_up_kernel(
        inputs=[x, *gate_up, indices, *shared_gate_up],
        template=launches.gate_up_template,
        grid=launches.gate_up_grid,
        threadgroup=(32, _GATE_UP_SIMDGROUPS, 1),
        output_shapes=[launches.gate_up_output],
        output_dtypes=[launches.dtype],
    )[0]


def _down(launches: _Launches, h, indices, scores, shape, down, shared_down, *beside):
    """The one-token down/combine launch over gate+up output ``h``.
    ``beside`` is the composed shared expert's output and its gate's, where
    the launches do not fold them."""
    if launches.fold:
        inputs = [h, *down, *shared_down, indices, scores]
    else:
        # The shared expert comes first in the inputs, so its launches are
        # encoded (and can run) before the router's.
        inputs = [*beside, h, *down, indices, scores]
    rps = _down_rows(1)
    return launches.down_kernel(
        inputs=inputs,
        template=launches.down_template + [("RPS", rps)],
        grid=(32, launches.down_threadgroup[1] * launches.hidden // rps, 1),
        threadgroup=launches.down_threadgroup,
        output_shapes=[shape],
        output_dtypes=[launches.dtype],
    )[0]


def _topk_folds(launches: _Launches, logits) -> bool:
    return (
        launches.topk_kernel is not None
        # The launch reads the logits in the plan's dtype.
        and logits.dtype == launches.dtype
        and logits.size // logits.shape[-1] <= _TOPK_FOLD_MAX_ROWS
        and softmax_topk_eligible(logits)
    )


def _gate_up_topk(launches: _Launches, x, logits, gate_up, shared_gate_up):
    """Gate+up output, indices and scores of every row of ``x`` from its
    router ``logits`` (the rows' softmax + top-k run inside the launch)."""
    rows = x.size // launches.hidden
    top_k, width = launches.top_k, launches.topk_width
    return launches.topk_kernel(
        inputs=[x, *gate_up, logits, *shared_gate_up],
        template=launches.gate_up_template
        + [("NE", logits.shape[-1]), ("M", rows), ("YW", width)],
        grid=(32, _GATE_UP_SIMDGROUPS * launches.topk_blocks * rows, 1),
        threadgroup=(32, _GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(rows, width), (rows, top_k), (rows, top_k)],
        output_dtypes=[launches.dtype, mx.uint32, launches.dtype],
    )


def _routing(logits, top_k: int):
    """Each row's experts and scores from its router ``logits``, as
    ``fused_router_topk`` selects them over the precise softmax: in one
    launch where ``softmax_topk_rows`` takes the rows."""
    routing = softmax_topk_rows(logits, top_k, WINDOW_MAX_ROWS)
    if routing is None:
        # On the GPU's stream, as every launch: whatever the caller's device.
        scores = mx.softmax(logits, axis=-1, precise=True, stream=mx.gpu)
        routing = fused_router_topk(scores, top_k)
    return routing


def _window_gate_up(launches: _Launches, x, indices, gate_up, shared_gate_up):
    rows = x.shape[0]
    return launches.window_gate_up_kernel(
        inputs=[x, *gate_up, indices, *shared_gate_up],
        template=launches.gate_up_template + [("M", rows)],
        grid=(32, launches.gate_up_grid[1] * rows, 1),
        threadgroup=(32, _GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(rows, *launches.gate_up_output)],
        output_dtypes=[launches.dtype],
    )[0]


def _window_gate_up_coalesced(
    launches: _Launches, x, meta, ucount, gate_up, shared_gate_up
):
    rows = x.shape[0]
    nsg = _GATE_UP_SIMDGROUPS
    rows_per_tg = nsg * _GATE_UP_ROWS
    sh_pairs = (rows + 1) // 2
    n_blocks = (
        1
        + sh_pairs * (launches.window_shared_width // rows_per_tg)
        + rows * launches.top_k * (launches.window_inter // rows_per_tg)
    )
    return launches.window_coalesced_kernel(
        inputs=[x, *gate_up, meta, ucount, *shared_gate_up],
        template=launches.gate_up_template + [("M", rows)],
        grid=(32, nsg * n_blocks, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(rows, *launches.gate_up_output)],
        output_dtypes=[launches.dtype],
    )[0]


def _window_gate_up_topk_coalesced(
    launches: _Launches, x, logits, gate_up, shared_gate_up
):
    rows = x.shape[0]
    nsg = (
        _GATE_UP_COALESCE_TOPK_SIMDGROUPS
        if (
            launches.window_inter % (_GATE_UP_COALESCE_TOPK_SIMDGROUPS * _GATE_UP_ROWS)
            == 0
            and launches.window_shared_width
            % (_GATE_UP_COALESCE_TOPK_SIMDGROUPS * _GATE_UP_ROWS)
            == 0
        )
        else _GATE_UP_SIMDGROUPS
    )
    rows_per_tg = nsg * _GATE_UP_ROWS
    sh_pairs = (rows + 1) // 2
    n_blocks = (
        rows
        + sh_pairs * (launches.window_shared_width // rows_per_tg)
        + rows * launches.top_k * (launches.window_inter // rows_per_tg)
    )
    template = [
        ("NSG", nsg) if k == "NSG" else (k, v) for k, v in launches.gate_up_template
    ] + [("NE", logits.shape[-1]), ("M", rows)]
    top_k, width = launches.top_k, launches.topk_width
    return launches.window_topk_coalesced_kernel(
        inputs=[x, *gate_up, logits, *shared_gate_up],
        template=template,
        grid=(32, nsg * n_blocks, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(rows, width), (rows, top_k), (rows, top_k)],
        output_dtypes=[launches.dtype, mx.uint32, launches.dtype],
    )


def _window_down(launches: _Launches, h, indices, scores, down, shared_down):
    rows = indices.shape[0]
    rps = _down_rows(rows)
    return launches.window_down(
        inputs=[h, *down, *shared_down, indices, scores],
        template=launches.down_template + [("RPS", rps), ("M", rows)],
        grid=(32, launches.down_threadgroup[1] * launches.hidden // rps * rows, 1),
        threadgroup=launches.down_threadgroup,
        output_shapes=[(rows, launches.hidden)],
        output_dtypes=[launches.dtype],
    )[0]


def _experts(launches: _Launches, x, logits, operands, folded: bool, *beside):
    """``(switch_mlp(x, indices) * scores[..., None]).sum(axis=-2)
    + mx.sigmoid(shared_expert_gate(x)) * shared_expert(x)`` for each row
    of ``x`` (one token, or [M, hidden] for a window), on the routing of
    its router ``logits``, in two launches: gate+up (the routing inside it
    where it ``folded``: ``_topk_folds``) and down. Row r of a window runs
    the one-token arithmetic on its own routing. ``beside`` is the composed
    shared expert's output and its gate's, where the launches do not fold
    them (one token)."""
    _, gate_up, down, shared_gate_up, shared_down = operands
    one_token = x.size == launches.hidden
    if one_token:
        if folded:
            h, indices, scores = _gate_up_topk(
                launches, x, logits, gate_up, shared_gate_up
            )
        else:
            indices, scores = _routing(logits, launches.top_k)
            h = _gate_up(launches, x, indices, gate_up, shared_gate_up)
        return _down(launches, h, indices, scores, x.shape, down, shared_down, *beside)
    rows = x.shape[0]
    gen = kernels.gpu_generation()
    can_unfolded_coal = (
        launches.window_coalesced
        and 2 <= rows <= _COALESCE_MAX_ROWS
        and rows * launches.top_k <= _COALESCE_MAX_PAIRS
        and softmax_topk_eligible(logits, WINDOW_MAX_ROWS)
    )
    if can_unfolded_coal and ((gen >= 17 and rows >= 3) or rows == 4):
        indices, scores, meta, ucount = softmax_topk_coalesce(
            logits, launches.top_k, WINDOW_MAX_ROWS
        )
        h = _window_gate_up_coalesced(
            launches, x, meta, ucount, gate_up, shared_gate_up
        )
    elif launches.window_coalesced and folded and rows == 2:
        h, indices, scores = _window_gate_up_topk_coalesced(
            launches, x, logits, gate_up, shared_gate_up
        )
    elif folded:
        h, indices, scores = _gate_up_topk(launches, x, logits, gate_up, shared_gate_up)
    else:
        indices, scores = _routing(logits, launches.top_k)
        h = _window_gate_up(launches, x, indices, gate_up, shared_gate_up)
    return _window_down(launches, h, indices, scores, down, shared_down)


def _token(launches: _Launches, folded: bool):
    """One token's launches as one function of its arrays, for MLX's
    compile: ``x``, what the block computes outside the launches (its
    gate's logits where the gate runs as it is, then a shared expert's
    output and its gate's where the launches do not fold them) and the
    plan's operands. It closes over no array."""

    def graph(x, outside, operands):
        if launches.router is None:
            logits, *outside = outside
        else:
            logits = launches.router(x, *operands[0])
        return _experts(launches, x, logits, operands, folded, *outside)

    return graph


def _ranked(rows: int, experts: int, dtype) -> mx.array:
    """Router logits for ``rows`` rows that rank the experts in order: each
    row selects the first top-k. Made on the GPU's stream, as a stand-in's
    zeros are (``kernels.zeros``)."""
    gpu = mx.gpu
    column = mx.zeros((rows, 1), mx.float32, stream=gpu)
    ranked = mx.subtract(column, mx.arange(experts, stream=gpu), stream=gpu)
    return mx.astype(ranked, dtype, stream=gpu)


def _stand_in(launches: _Launches, logits, folded: bool, outside, operands):
    """The smallest launches of a block's rows that MLX generates the Metal
    of the call's for, over zeros: the rows, the router's weight, the whole
    of each shared matrix, what the block computes outside the launches,
    and as many matrices of each stack as a row selects. ``logits`` (of
    ``_ranked``) stand for the router's: each row selects the first
    matrices, which the stacks have. Nothing of the call is read but the
    shapes and dtypes of ``outside`` and of the operands.

    The logits and the zeros are the stand-in's inputs, made on the GPU's
    stream (``kernels.zeros``), which the proof evaluates ahead of its
    guard: what fails in making them (an allocation, an error an earlier
    evaluation left pending) is raised by the call and fails no proof."""
    rows, top_k = logits.shape[0], launches.top_k
    weight, gate_up, down, shared_gate_up, shared_down = operands
    if launches.router is None:
        outside = outside[1:]
    groups = (
        (((rows, launches.hidden), launches.dtype),),
        tuple((a.shape, a.dtype) for a in weight),
        tuple(((top_k, *a.shape[1:]), a.dtype) for a in gate_up),
        tuple(((top_k, *a.shape[1:]), a.dtype) for a in down),
        tuple((a.shape, a.dtype) for a in shared_gate_up),
        tuple((a.shape, a.dtype) for a in shared_down),
        tuple((a.shape, a.dtype) for a in outside),
    )

    def launch(logits, *made):
        made = iter(made)
        (x,), *stand_ins, outside = (
            tuple(next(made) for _ in group) for group in groups
        )
        launched = [_experts(launches, x, logits, stand_ins, folded, *outside)]
        if launches.router is not None:
            launched.append(launches.router(x, *stand_ins[0]))
        return launched

    zeros = kernels.zeros(*(spec for group in groups for spec in group))
    return [logits, *zeros], launch


def _outside(block, plan: _Plan, x):
    """What one token's launches take that the block computes outside them:
    the logits of a gate that runs as it is, then a shared expert's output
    and its gate's where the launches do not fold them. None where one is
    not what the launches read: an array in ``x``'s dtype (they are
    compiled for that one), a shared output of ``x``'s shape, one gate
    value of each of ``x``'s rows (the down launch reads ``gate[0]``),
    logits of the router's width for each row."""
    launches, outside = plan.launches, ()
    rows = x.shape[:-1]
    if not launches.fold:
        # Children by item: this runs for every MoE layer of every decode step.
        shared = block["shared_expert"](x)
        gate = block["shared_expert_gate"](x)
        if (
            shared.dtype != x.dtype
            or gate.dtype != x.dtype
            or shared.shape != x.shape
            or gate.shape != (*rows, 1)
        ):
            return None
        outside = (shared, gate)
    if launches.router is None:
        logits = block["gate"](x)
        if logits.dtype != x.dtype or logits.shape != (*rows, plan.experts):
            return None
        outside = (logits, *outside)
    return outside


_SAID = (
    "Qwen MoE fused routed-expert decode engaged",
    "fused routed-expert decode failed; composed fallback",
)


def _decode(block, plan: _Plan, x, graph):
    """One token's block output through the plan's launches, or None: ``x``
    does not run through them, or their launch for it does not compile.
    ``graph`` is the one the calling thread kept for ``x``'s signature, or
    None.

    The launches are one graph of ``x`` and the block's arrays, which MLX's
    compile builds the first time a thread meets the signature and the
    proof keeps for that thread. Its Metal is proven first, on the stand-in
    of ``_stand_in``: nothing of the call is evaluated. (Once, as the
    block's plan is built outside a trace, a view of each stack's first
    matrix is evaluated to learn whether it shares the stack's buffer:
    ``_expert_view``.)"""
    launches = plan.launches
    if graph is None and not _routes(plan, x):
        return None
    outside = _outside(block, plan, x)
    if outside is None:
        return None
    if graph is not None:
        return graph(x, outside, plan.operands)
    logits = _ranked(1, plan.experts, launches.dtype)
    folded = _topk_folds(launches, logits)
    stand_in = partial(_stand_in, launches, logits, folded, outside, plan.operands)
    if not plan.proof.prove((plan.experts, folded), stand_in, logger, *_SAID):
        return None
    graph = mx.compile(_token(launches, folded))
    y = graph(x, outside, plan.operands)
    plan.proof.keep((x.shape, x.dtype, plan.experts), graph)
    return y


def _window(launches: _Launches, folded: bool):
    """A window's launches as one function of its rows and the plan's
    operands, for MLX's compile: the router, gate+up and down of every row
    of ``x`` ([B, L, hidden]), the rows ([rows, hidden]) out. It closes over
    no array."""

    def graph(x, operands):
        x = mx.reshape(x, (-1, launches.hidden), stream=mx.gpu)
        logits = launches.router(x, *operands[0])
        return _experts(launches, x, logits, operands, folded)

    return graph


def _window_logits(launches: _Launches, folded: bool):
    """A window's launches as one function of its rows and the routing the
    block's quantized gate computed outside it, for MLX's compile: the rows
    of ``x`` ([B, L, hidden]), their logits ([rows, experts]) and the plan's
    operands, the rows ([rows, hidden]) out. It closes over no array."""

    def graph(x, logits, operands):
        x = mx.reshape(x, (-1, launches.hidden), stream=mx.gpu)
        return _experts(launches, x, logits, operands, folded)

    return graph


def _windows(plan: _Plan, x) -> bool:
    """Whether the rows of a window ``x`` ([B, L, hidden]) run through the
    plan's window launches: 2..``WINDOW_MAX_ROWS`` rows of the plan's
    width, in the dtype the launches are compiled for, for a router of
    whole simdgroups (the fused gemv, or the block's quantized gate's
    rows) that the launches run. (A window of one row is one token's
    call.)"""
    launches = plan.launches
    return (
        x.dtype == launches.dtype
        and x.shape[-1] == launches.hidden
        and 2 <= x.size // launches.hidden <= WINDOW_MAX_ROWS
        and (launches.router is not None or plan.quantized_router)
        and router_eligible(x, plan.experts, WINDOW_MAX_ROWS)
    )


def _cached_plan(block, dtype: mx.Dtype) -> _Plan | None:
    """The block's plan, kept in the block: depth 2 keys it on the expert
    and shared-expert weight arrays, the activation dtype, and the block attributes
    that it is planned for, with their types (8 and 8.0 differ). Keeps one plan
    per block; switching dtype rebuilds it while proofs persist per dtype."""
    top_k, experts = block.top_k, block.num_experts
    return cached_per_module(
        block,
        "_omlx_routed_decode_plan",
        partial(_build_plan, dtype=dtype),
        depth=2,
        flags=(dtype, type(top_k), top_k, type(experts), experts),
    )


def routed_verify_window(block, x):
    """The block's output for a verify window ``x`` ([B, L, hidden],
    1..``WINDOW_MAX_ROWS`` rows), each row bit-identical to the fused
    one-token call on that row, as [rows, hidden]; or None outside the
    fused layout.

    Router gemv (or a quantized gate's rows, through ``row_exact_linear``),
    gate+up (with the softmax + top-k folded in up to 3 rows, after the
    routing launch from 4) and down/combine each run once for the whole
    window, in one graph of the rows, their routing and the block's arrays
    that MLX's compile builds the first time a thread meets the window's
    shape and dtype and the router's width on blocks of these faces,
    tensor dtypes and shapes, and that the window's proof keeps for that
    thread. Its Metal is proven first on a stand-in (``_stand_in``):
    nothing of the call is evaluated (once, as the block's plan is built
    outside a trace, a view of each stack's first matrix is:
    ``_expert_view``). A window of one row is one token's call: its graph
    and its proof. A failed one-token proof keeps the windows composed
    too."""
    if x.ndim != 3:
        return None
    plan = _cached_plan(block, x.dtype)
    if plan is None or plan.window is None or plan.proof.failed:
        return None
    launches = plan.launches
    signature = (x.shape, x.dtype, plan.experts)
    if x.size == x.shape[-1]:
        # One row runs the one-token launches: their proof and graph.
        y = _decode(block, plan, x, plan.proof.graphs.get(signature))
        return None if y is None else mx.reshape(y, (1, -1), stream=mx.gpu)
    rows = x.size // x.shape[-1]
    # The window's graph is the plan's variant: the fused router's two-argument
    # call or a quantized gate's external logits. Plans of both can share the
    # proof (the router's absence is in its key), so the signature names the
    # variant.
    window_signature = (*signature, plan.quantized_router)
    # A verify step's block is the replay of the graph its thread kept.
    graph = plan.window.graphs.get(window_signature)
    if graph is not None:
        if plan.quantized_router:
            logits = row_exact_linear(block["gate"], x.reshape(rows, launches.hidden))
            if logits.dtype != launches.dtype or logits.shape != (rows, plan.experts):
                return None
            return graph(x, logits, plan.operands)
        return graph(x, plan.operands)
    if plan.window.failed or not _windows(plan, x):
        return None
    # A quantized gate's logits are its own one-row arithmetic per row
    # (``row_exact_linear``), computed outside the window's graph.
    logits = None
    if plan.quantized_router:
        logits = row_exact_linear(block["gate"], x.reshape(rows, launches.hidden))
        if logits.dtype != launches.dtype or logits.shape != (rows, plan.experts):
            return None
    ranked = _ranked(rows, plan.experts, launches.dtype)
    folded = _topk_folds(launches, ranked)
    outside = () if logits is None else (ranked,)
    stand_in = partial(_stand_in, launches, ranked, folded, outside, plan.operands)
    said = (
        f"Qwen MoE fused verify-window experts engaged ({rows} rows)",
        "fused verify-window experts failed; verifier fallback",
    )
    if not plan.window.prove((rows, plan.experts, folded), stand_in, logger, *said):
        return None
    if logits is None:
        graph = mx.compile(_window(launches, folded))
        y = graph(x, plan.operands)
    else:
        graph = mx.compile(_window_logits(launches, folded))
        y = graph(x, logits, plan.operands)
    plan.window.keep(window_signature, graph)
    return y


def _ensure_verify_window_patch(cls) -> None:
    """Serve verify MoE blocks through ``routed_verify_window``.

    Wraps the mlx-vlm verifier's ``_feed_forward`` (the router patch's verify
    entry) for ``cls`` blocks while a verify is armed; other blocks and shapes
    keep the wrapped body."""
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    original = Qwen3_5BatchInvariantForward._feed_forward
    if getattr(original, "_omlx_routed_verify_window", False):
        return

    @wraps(original)
    def verify_window(self, feed_forward, x):
        armed = is_row_exact_armed() or is_verify_armed()
        if armed and isinstance(feed_forward, cls):
            y = routed_verify_window(feed_forward, x)
            if y is not None:
                return mx.reshape(y, x.shape, stream=mx.gpu)
        return original(self, feed_forward, x)

    verify_window._omlx_routed_verify_window = True
    Qwen3_5BatchInvariantForward._feed_forward = verify_window


def apply_qwen35_moe_routed_decode_patch() -> bool:
    """Wrap the router-fused mlx-vlm ``Qwen3_5MoeSparseMoeBlock`` call.

    Needs ``qwen35_moe_router`` applied first: the fast arm reuses its fused
    routing launch, so it selects the same experts with the same scores as
    the body it replaces."""
    if not mx.metal.is_available():
        return False
    try:
        from mlx_vlm.models.qwen3_5_moe import language as vlm_moe
    except ImportError:
        return False
    cls = getattr(vlm_moe, "Qwen3_5MoeSparseMoeBlock", None)
    if cls is None or not getattr(cls, "_omlx_router_fused", False):
        return False
    if getattr(cls, "_omlx_routed_decode", False):
        return True
    # Read the weight formats with the model rather than in its first token.
    # A registry that cannot load leaves the block unpatched. The engine
    # applies its patches under a guard that logs at debug level.
    try:
        formats.load()
    except Exception:
        logger.exception("quant formats did not load; routed decode not applied")
        raise
    orig_call = cls.__call__

    def patched_call(self, x):
        shape = x.shape
        if shape and x.size == shape[-1]:
            plan = _cached_plan(self, x.dtype)
            if plan is not None:
                # A decode step is the replay of the graph its thread kept.
                graph = plan.proof.graphs.get((shape, x.dtype, plan.experts))
                if graph is not None and not plan.outside:
                    return graph(x, (), plan.operands)
                y = _decode(self, plan, x, graph)
                if y is not None:
                    return y
        elif x.ndim == 3 and shape[1] == 1:
            # Batched one-token decode: every row runs the one-token
            # arithmetic of its own stream in the window launches.
            y = routed_verify_window(self, x)
            if y is not None:
                return mx.reshape(y, shape, stream=mx.gpu)
        return orig_call(self, x)

    patched_call._omlx_routed_decode_original = orig_call
    cls.__call__ = patched_call
    cls._omlx_routed_decode = True
    _ensure_verify_window_patch(cls)
    logger.info("Qwen MoE fused routed-expert decode patch applied")
    return True
