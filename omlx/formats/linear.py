# SPDX-License-Identifier: Apache-2.0
"""Linear layers whose weights are kept in a registered format: what
``nn.QuantizedLinear`` and mlx-vlm's ``QuantizedSwitchLinear`` are to the
formats MLX knows.

A layer holds a matrix's tensors under the names its format gives them, and
asks the registry how to run them:

- dense layers pin a single row to their dot face and resolve every
  float16/bfloat16 call above one row through a construction-time per-chip
  row schedule (the Dot rows kernel up to ``MAX_ROWS``, MLX's matmul, wide
  mat-vec, verify MMA/MSG tiles, or steel/NAX dense tiles), replaying warm
  signatures through the native ``FormatLinearFast``;
- a switch layer's sorted prefill uses its tile face on tensor units or
  through MLX's steel matrix kernels, from 16 rows and four rows per expert
  on average. Below that threshold it uses its decode rows kernel;
- everything else (a layout the face does not cover, activations of a type
  it is not exact for, a format without the face, a face that did not
  compile) is the format's ``decode``, cast to the activations' type, then
  MLX's matmul (its ``gather_mm`` from as many indices as a switch layer's
  stack has matrices).
"""

from __future__ import annotations

import logging
import math
from copy import copy
from functools import cache
from threading import local
from typing import Any, NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .. import formats
from ..patches.module_cache import cached_per_module
from . import Dot, Tile, kernels, tile_dense

logger = logging.getLogger(__name__)


def packed_name(layer) -> str:
    """The name of ``layer``'s packed tensor: the format's declared codes
    name for a format layer, never a position in its tensor tuple;
    ``weight`` for MLX's own quantized layers (``formats.tensor_names``)."""
    fmt = getattr(layer, "fmt", None)
    if fmt is not None:
        return fmt.codes_name
    return formats.tensor_names(layer)[0]


# Maximum rows in one dense Dot rows launch (the fallback when a format
# has no legal tile route for 2..16 rows).
MAX_ROWS = 16

# C7 (2026-11-23) receipts: single-row decode cells on M3 Ultra where MLX's
# own qmv beats the Dot face (512x2560, 2560x2560 and 640x2560 at 1 row).
# Rows >= 2 are governed by the per-chip construction-time schedule and
# replayed through FormatLinearFast (VFAST Commit 3).
_DECODE_MLX_ROWS = frozenset()
_DECODE_MLX_CELLS = {
    ("applegpu_g15d", 512, 2560): frozenset({1}),
    ("applegpu_g15d", 2560, 2560): frozenset({1}),
    ("applegpu_g15d", 640, 2560): frozenset({1}),
}


def _decode_mlx_rows(architecture: str | None, n: int, k: int) -> frozenset[int]:
    """The decode row counts a stock layer of this (n, k) runs on MLX's qmv."""
    return _DECODE_MLX_CELLS.get((architecture, n, k), _DECODE_MLX_ROWS)


# Outside the rows kernel a switch layer decodes its whole stack at once for
# a call of at least _AT_ONCE_PAIRS pairs per matrix (where that is measured
# faster than pieces) when the stack is within _DECODE_AT_ONCE_BYTES (in the
# decode's dtype, and in x's when it differs); else a piece at a time,
# _DECODE_BYTES of matrices counted as float32 and in x's dtype. MLX keeps
# every piece of one evaluation in flight and frees a buffer when its command
# buffer completes: only many small pieces, which split its command buffers,
# hold less than at once.
_AT_ONCE_PAIRS = 20
_DECODE_AT_ONCE_BYTES = 1 << 31
_DECODE_BYTES = 1 << 28

# One threadgroup per pair (a row of x and the matrix it meets) and tile of 8
# of the pair's outputs: 4 for each of the threadgroup's two simdgroups.
# __MATRIX__ is the pair's matrix in the stack, __X_ROW__ its row of x.
_STACKED = r"""
    static_assert(
        PAIRS_PER_ROW >= 1,
        "format switch linear: PAIRS_PER_ROW is the matrices of a row of x, 1 or more");
"""
_SOURCE = r"""
    constexpr int RPS = 4;  // output rows of a simdgroup
    static_assert(
        N_SIZE % (2 * RPS) == 0,
        "format linear: N must be whole tiles of 8 outputs");
    const uint pair = threadgroup_position_in_grid.y;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const size_t out_row =
        size_t(threadgroup_position_in_grid.z) * (2 * RPS) + simd_gid * RPS;
    float result[RPS];
    qmv_rows<T, K_SIZE, RPS, 0>(
        m, size_t(__MATRIX__) * N_SIZE + out_row, m, 0,
        x + size_t(__X_ROW__) * K_SIZE, simd_lid, result);
    if (simd_lid == 0) {
      store_rows(omlx_iota<RPS>(), result, y + size_t(pair) * N_SIZE + out_row);
    }
"""


# Two to 16 rows of x in one launch, G of them a threadgroup: each block of
# the matrix decoded once for the G rows (the shared-rows skeleton), each
# row its one-row bits. RPS output rows a simdgroup, two simdgroups a tile
# of 8, as the one-row kernel. The skeleton is unrolled for a face whose one
# row is inside kernels.unrolled_tile_ok (groups of G rows stay within
# kernels.ROW_VALUES_BUDGET, inside it too), rolled for a wider face. Where N
# is not whole tiles (PARTIAL), the last tile's simdgroups hold the rows that
# are left, or none, as MLX's qmv does for its last tile.
_SHARED_SOURCE = r"""
    constexpr int RPS = __RPS__;
    constexpr bool PARTIAL = N_SIZE % (2 * RPS) != 0;
    const size_t row0 = size_t(threadgroup_position_in_grid.y) * ROWS;
    const int out_row = int(threadgroup_position_in_grid.z) * (2 * RPS) +
        int(simdgroup_index_in_threadgroup) * RPS;
    if (PARTIAL && out_row >= N_SIZE) {
      return;
    }
    const int valid = PARTIAL ? min(RPS, N_SIZE - out_row) : RPS;
    float result[ROWS * RPS];
    qmv_shared<T, K_SIZE, ROWS, RPS, VPT, PARTIAL>(
        m, size_t(out_row), valid, x + row0 * K_SIZE, thread_index_in_simdgroup, result);
    if (thread_index_in_simdgroup == 0) {
      for (int i = 0; i < ROWS; i++) {
        device T* out = y + (row0 + i) * N_SIZE + out_row;
        if (PARTIAL) {
          for (int r = 0; r < valid; r++) {
            out[r] = static_cast<T>(result[i * RPS + r]);
          }
        } else {
          store_rows(omlx_iota<RPS>(), result + i * RPS, out);
        }
      }
    }
"""
# The shared launch's outputs: RPS rows a simdgroup, two simdgroups a
# threadgroup's tile; its Metal and its grid read them from here.
_SHARED_RPS = 4
_SHARED_TILE = 2 * _SHARED_RPS
_SHARED_SOURCE = _SHARED_SOURCE.replace("__RPS__", str(_SHARED_RPS))
# A group of rows takes a launch's threadgroups down by its size; below a
# number of threadgroups a GPU core the cores idle and the group loses what
# sharing the weights saves. That number grows with the face's values a lane
# of a row (affine's vpt: 16 at 2 to 5 bits on qmv_fast, 8 at 6 and 8 bits
# on qmv_fast, 4 at 8 bits on qmv), whatever the group, and differs between
# GPU generations: for each, {vpt: threadgroups a core}. Measured
# (tools/main/unify/occupancy_bench.py, affine, K 2560, chains of 64
# dependent launches, the shared launch against the one-row launch; M3 Ultra
# 80 cores, M5 Max 40):
# - vpt 16 (2, 3 and 4 bits, groups of 2): both lose at 2 a core (1.010 to
#   1.093). The M3 Ultra loses at 3 a core (240 threadgroups: 1.064 / 1.027
#   at 3 / 4 bits); 3.2 (256) is the lowest measured threadgroup count that
#   wins for 4 bits (the gate's 35B shape, K 2048: refusing the group cost
#   0.5%), and 4 (the gate's flash shape, K 2560: refusing cost 0.9%; 2 / 4
#   bits 0.954 / 0.990), 6 and 8 (0.945 / 0.957) win; affine 3 bits at 320
#   stays at 1.017, the pre-bound behaviour, on no served path. The M5 Max
#   loses 1.6-4.5% at 6 a core and wins at 8 (0.960 / 0.997 / 0.969), 12
#   and 16.
# - vpt 8 (8 bits on qmv_fast, groups of 2, 3 and 4): the M3 Ultra breaks
#   even or wins at 2 a core (1.001 / 0.986 / 0.956), the M5 Max wins at 4
#   (0.968 to 0.922).
# A vpt not measured (4, 12, 32, ...) takes its generation's largest floor;
# a generation not measured takes the larger of each.
OCCUPANCY = {15: {8: 2, 16: 3.2}, 17: {8: 4, 16: 8}}
_UNMEASURED = {8: 4, 16: 8}


@cache
def _shared_kernel(face: Dot):
    matrices = kernels.Matrices(
        unrolled=kernels.unrolled_tile_ok(face.vpt, _SHARED_RPS, 1)
    )
    inputs = ["x", *matrices.add("m", face, shared=True)]
    return mx.fast.metal_kernel(
        name=f"omlx_format_linear_rows_{face.key}",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header + kernels.STORE_ROWS,
        source=matrices.declarations + _SHARED_SOURCE,
    )


def _group(rows: int, vpt: int, n: int) -> int:
    """The rows of x a threadgroup of the shared launch takes: the most, up
    to 4, that divide ``rows``, keep a lane's inputs of the group within
    kernels.ROW_VALUES_BUDGET (which keeps the unrolled tile in its
    envelope) and leave each GPU core the threadgroups that the face's vpt
    needs on this GPU (OCCUPANCY) over N ``n``; 1 where none does (the
    one-row launch then runs each row).
    Measured (u7 spike, both Macs): at 16 values a lane, groups of 3 and 4
    ran 1.15-1.70 of the one-row launch where groups of 2 ran 0.76-0.98."""
    limit = kernels.ROW_VALUES_BUDGET // vpt
    groups = [g for g in (4, 3, 2) if g <= limit and rows % g == 0]
    if not groups:
        # One row, a count no group divides, a wide face: nothing to ask
        # of the GPU.
        return 1
    floor = _occupancy(vpt) * kernels.gpu_cores()
    tiles = -(-n // _SHARED_TILE)
    return next((g for g in groups if rows // g * tiles >= floor), 1)


def _occupancy(vpt: int) -> float:
    """The threadgroups a GPU core needs of a shared launch over a face of
    ``vpt`` values a lane, on this GPU (OCCUPANCY)."""
    table = _occupancy_table(kernels.gpu_generation())
    return table.get(vpt, max(table.values()))


@cache
def _occupancy_table(generation: int) -> dict:
    """OCCUPANCY's table for a GPU ``generation``; a generation not measured
    takes the larger of each, logged once."""
    table = OCCUPANCY.get(generation)
    if table is None:
        logger.warning(
            "GPU generation %d: no measured occupancy; format linear row groups "
            "keep the larger floors",
            generation,
        )
        table = _UNMEASURED
    return table


@cache
def _kernel(face: Dot, stacked: bool):
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("m", face)]
    if stacked:
        inputs.append("rhs")
        matrix, x_row = "rhs[pair]", "pair / uint(PAIRS_PER_ROW)"
    else:
        matrix, x_row = "0", "pair"
    return mx.fast.metal_kernel(
        name=f"omlx_format_{'switch_linear' if stacked else 'linear'}_{face.key}",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header + kernels.STORE_ROWS,
        source=matrices.declarations
        + (_STACKED if stacked else "")
        + _SOURCE.replace("__MATRIX__", matrix).replace("__X_ROW__", x_row),
    )


class _Tile(NamedTuple):
    """The selected tile kernel, its proof and the layer's current operands."""

    face: Tile
    kernel: Any
    operands: tuple
    proof: kernels.Proof
    said: tuple[str, str]


class _Plan(NamedTuple):
    face: Dot
    kernel: Any
    shared: Any  # 2 to 16 rows of a matrix in one launch; None for a stack
    operands: tuple  # the weight, then the face's tensors
    proof: kernels.Proof
    k: int
    n: int
    said: tuple[str, str]  # what a proof logs: the launches engaged, or failed


def _tile(layer) -> _Tile | None:
    """Bind the sorted-gather tile backend for a switch layer on this GPU."""
    face = formats.tile(layer, layer.input_dims, layer.output_dims)
    if face is None:
        return None
    if _nax_host():
        from ..patches import m5_gather_qmm_nax as nax

        kernel = nax._face_kernel(face, "plain")
        route = "format_switch_tile_nax"
    else:
        from . import tile_gather

        kernel = tile_gather.kernel(face)
        route = "format_switch_tile_steel"
    if kernel is None:
        return None
    said = (
        f"quant format face {face.key}: tile kernel engaged",
        f"quant format face {face.key}: tile kernel failed; rows kernel fallback",
    )
    operands = tuple(layer[name] for name in (packed_name(layer), *face.tensors))
    dtypes = tuple(a.dtype for a in operands)
    proof = kernels.proof(route, face, dtypes)
    return _Tile(face, kernel, operands, proof, said)


@cache
def _metal_architecture():
    if mx.is_available(mx.gpu):
        return mx.device_info(mx.gpu).get("architecture")
    return None


@cache
def _nax_host() -> bool:
    """Whether this GPU has the tensor units oMLX's sorted gather runs on."""
    from ..custom_kernels.nax import is_nax_available

    try:
        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


def _plan(layer) -> _Plan | None:
    """How ``layer``'s rows run through its format's dot face, or None: the
    format has no face for the matrix, or a launch could not take it. A
    layer that still holds a load's placeholder is refused here, ahead of
    every route, once per plan and not per call."""
    layer.refuse_placeholders()
    k, n = layer.input_dims, layer.output_dims
    # MLX binds x as a buffer from 8 elements.
    if k < 8:
        return None
    face = formats.dot(layer, k, n)
    if face is None:
        return None
    operands = tuple(layer[name] for name in (packed_name(layer), *face.tensors))
    stacked = operands[0].ndim == 3
    # The one-row and the stack's launches are whole tiles of 8 outputs; a
    # matrix's N that is not runs in the shared-rows launch alone.
    if n % 8 and (stacked or face.vpt < 1):
        return None
    # What a launch is compiled from, beside what the layer keeps in ran.
    proof = kernels.proof(
        "format_switch_linear" if stacked else "format_linear",
        face,
        tuple(a.dtype for a in operands),
        k,
        n,
    )
    shared = None if stacked or face.vpt < 1 else _shared_kernel(face)
    kernel = _kernel(face, stacked)
    return _Plan(face, kernel, shared, operands, proof, k, n, _said(face.key))


@cache
def _said(key: str) -> tuple[str, str]:
    """What a proof of face ``key`` logs: the launches engaged, or failed."""
    return (
        f"quant format face {key}: rows kernel engaged",
        f"quant format face {key}: rows kernel failed; decode fallback",
    )


def _rows(
    kernel, k: int, n: int, x, operands, pairs, shape, rhs=None, per_row=1, tiles=None
):
    """One mat-vec per pair, ``pairs`` of them, as ``shape`` (which ends
    with N). Pair ``p`` is row ``p // per_row`` of ``x``, rows of K in any
    shape, with the matrix (``operands``), or with matrix ``rhs[p]`` of the
    stack. No reshape is put in the graph for the kernel's sake. A pair's
    outputs are ``tiles`` tiles of 8 from the first, or all of them: the
    grid is no part of the Metal, so a stand-in computes one."""
    dtype = x.dtype
    inputs = [x, *operands]
    template = [("T", dtype), ("K_SIZE", k), ("N_SIZE", n)]
    if rhs is not None:
        inputs.append(rhs)
        template.append(("PAIRS_PER_ROW", per_row))
    return kernel(
        inputs=inputs,
        template=template,
        grid=(32, 2 * pairs, n // 8 if tiles is None else tiles),
        threadgroup=(32, 2, 1),
        output_shapes=[shape],
        output_dtypes=[dtype],
    )[0]


class _Pairs(NamedTuple):
    """How a stack's launch pairs the rows of x with the indices."""

    pairs: int
    per_row: int  # the matrices of a row of x: its indices, or 1 if broadcast
    batch: tuple  # the leading axes of the result, ahead of x's rows (M) and N
    as_they_are: bool  # each row of x meets its own run of experts


def _pairs(k: int, x, indices) -> _Pairs | None:
    """How a stack's launch pairs ``x`` (rows of K in any shape) with
    ``indices``, from their shapes alone; None when there is no pair."""
    if not indices.size * x.size:
        return None
    shape, batch = x.shape, indices.shape
    lead, m = shape[:-2], shape[-2]
    own = len(lead)  # x's leading axes ahead of its trailing ones of 1
    while own and lead[own - 1] == 1:
        own -= 1
    # Each row of x meets its own run of experts: nothing is broadcast, in
    # memory or in shape. MLX binds indices of no axis as a scalar.
    as_they_are = m == 1 and len(batch) == len(lead) > 0 and batch[:own] == lead[:own]
    if as_they_are:
        pairs = indices.size
        return _Pairs(pairs, pairs // (x.size // k), batch, True)
    batch = mx.broadcast_shapes(lead, batch)
    return _Pairs(math.prod(batch) * m, 1, batch, False)


def _metal(x, pairs: _Pairs | None = None):
    """What a launch's Metal is instantiated with beside its proof's key:
    x's type, and for a stack the pairs of a row of x and whether there are
    fewer than 8 pairs (MLX binds fewer than 8 indices as constants: other
    Metal)."""
    return x.dtype if pairs is None else (x.dtype, pairs.per_row, pairs.pairs < 8)


def _linear(plan: _Plan, x):
    """A matrix's launch for ``x``: what its Metal is instantiated with
    beside the proof's key, and its graph, from ``x`` and the operands.
    Rows of x that a group of 2 to 4 divides run in the shared launch (x's
    type and the group), every other count in the one-row launch (x's
    type); a matrix whose N is not whole tiles of 8 runs every count in the
    shared launch (a group of 1 where none divides). The graph holds no
    array: it outlives the layers it is built for."""
    k, n = plan.k, plan.n
    pairs, shape = x.size // k, (*x.shape[:-1], n)
    metal = _linear_metal(plan, x)
    if isinstance(metal, tuple):
        shared, vpt, group = plan.shared, plan.face.vpt, metal[1]

        def graph(x, *operands):
            return _shared_rows(shared, k, n, vpt, x, operands, pairs, group, shape)

        return metal, graph
    kernel = plan.kernel

    def graph(x, *operands):
        return _rows(kernel, k, n, x, operands, pairs, shape)

    return metal, graph


def _linear_metal(plan: _Plan, x):
    """What a matrix's launch for ``x`` is instantiated with beside the
    proof's key: x's type and the group of rows a threadgroup takes for the
    shared launch, x's type alone for the one-row launch (``_linear``)."""
    group = (
        1 if plan.shared is None else _group(x.size // plan.k, plan.face.vpt, plan.n)
    )
    if group > 1 or plan.n % 8:
        return x.dtype, group
    return _metal(x)


def _shared_rows(
    kernel, k: int, n: int, vpt: int, x, operands, rows, group, shape, tiles=None
):
    """``rows`` rows of ``x`` with the matrix, ``group`` of them a
    threadgroup, as ``shape``; ``tiles`` tiles of 8 outputs from the first,
    or all of them."""
    return kernel(
        inputs=[x, *operands],
        template=[
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", group),
            ("VPT", vpt),
        ],
        grid=(
            32,
            2 * (rows // group),
            -(-n // _SHARED_TILE) if tiles is None else tiles,
        ),
        threadgroup=(32, 2, 1),
        output_shapes=[shape],
        output_dtypes=[x.dtype],
    )[0]


def _linear_stand_in(plan: _Plan, metal):
    """The smallest launch of a matrix that MLX generates the Metal of
    ``metal`` for (x's type), over zeros: one vector, and the whole of each
    tensor. What follows the rows in a matrix's tensors is the format's
    own, so no part of a tensor is known to be all that a row reads."""
    operands = ((a.shape, a.dtype) for a in plan.operands)
    if isinstance(metal, tuple):
        # The shared launch: one group of rows.
        dtype, group = metal
        k, n, vpt = plan.k, plan.n, plan.face.vpt

        def launch(x, *tensors):
            # Its first tile of outputs.
            shape = (group, n)
            return _shared_rows(
                plan.shared, k, n, vpt, x, tensors, group, group, shape, 1
            )

        return kernels.zeros(((group, k), dtype), *operands), launch

    def launch(x, *tensors):
        # Its first tile of outputs: the only ones its grid writes.
        return _rows(plan.kernel, plan.k, plan.n, x, tensors, 1, (8,), None, 1, 1)

    return kernels.zeros(((plan.k,), metal), *operands), launch


def _switch(plan: _Plan, x, indices):
    """A stack's launch for ``x`` and ``indices``, as ``_linear`` gives a
    matrix's, with the indices ahead of the operands; None when there is no
    pair to launch, or more than a grid takes. A negative index counts from
    the stack's end, as MLX's indexing does."""
    kernel, k, n = plan.kernel, plan.k, plan.n
    paired = _pairs(k, x, indices)
    # MLX's grid is int32, of 2 rows of threadgroups a pair.
    if paired is None or paired.pairs >= 2**30:
        return None
    pairs, per_row, batch, as_they_are = paired
    m = x.shape[-2]
    shape = (*batch, m, n)

    def graph(x, indices, *operands):
        if not as_they_are:
            x = mx.broadcast_to(x, (*batch, m, k))
            indices = mx.repeat(mx.broadcast_to(indices, batch), m)
        rhs = _wrapped(indices, operands[0].shape[0]).astype(mx.uint32)
        return _rows(kernel, k, n, x, operands, pairs, shape, rhs, per_row)

    return _metal(x, paired), graph


def _wrapped(indices, matrices: int):
    """``indices`` with each negative one counted from the end of a stack of
    ``matrices``, in a type wide enough to count them (a stack of 128 is no
    int8)."""
    if not mx.issubdtype(indices.dtype, mx.signedinteger):
        return indices
    if indices.dtype.size < 4:
        indices = indices.astype(mx.int32)
    return mx.where(indices < 0, indices + matrices, indices)


def _switch_stand_in(plan: _Plan, metal):
    """The smallest launch of a stack that MLX generates the Metal of
    ``metal`` for, over zeros. MLX binds an input of fewer than 8 elements
    as constants, which is other Metal: the stand-in has the fewest rows of
    x whose pairs are 8 or more, or fewer, as the call's are, and the
    fewest matrices that leave each tensor 8 elements. Every pair is with
    matrix 0."""
    dtype, per_row, few = metal
    rows = 1 if few else -(-8 // per_row)
    pairs = rows * per_row
    # Never more matrices than the stack has: formats.dot refuses a layer
    # with a tensor of fewer than 8 elements.
    matrices = max(-(-8 * a.shape[0] // a.size) for a in plan.operands)
    operands = (((matrices, *a.shape[1:]), a.dtype) for a in plan.operands)
    shape = (pairs, plan.n)

    def launch(x, *tensors):
        *tensors, rhs = tensors
        # Each pair's first tile of outputs.
        return _rows(
            plan.kernel, plan.k, plan.n, x, tensors, pairs, shape, rhs, per_row, 1
        )

    specs = (((rows, plan.k), dtype), *operands, ((pairs,), mx.uint32))
    return kernels.zeros(*specs), launch


# The activation dtypes oMLX serves a format's layers with: bfloat16, and
# float16 when activations are recast. A load builds only its checkpoint's
# floating dtype when known, falling back to both otherwise.
_WARM_DTYPES = (mx.bfloat16, mx.float16)
_NO_EPILOGUE = object()


def _infer_warm_dtypes(tensors, layout, dtypes) -> tuple[mx.Dtype, ...]:
    """The served floating dtypes to warm for one layer."""
    if dtypes is not None:
        return tuple(dict.fromkeys(dt for dt in dtypes if dt in _WARM_DTYPES))
    stored = (
        {name for name, (_, dt) in layout.items() if dt is None}
        if layout is not None
        else set()
    )
    found = [
        t.dtype
        for name, t in tensors.items()
        if name not in stored and t.dtype in _WARM_DTYPES
    ]
    if found:
        return tuple(dt for dt in _WARM_DTYPES if dt in found)
    return _WARM_DTYPES


def _switch_stand_in_specs(operands) -> tuple[tuple[tuple[int, ...], mx.Dtype], ...]:
    """Fewest matrices of a switch stack that preserve MLX input binding."""
    experts = operands[0].shape[0]
    matrices = max(
        (-(-8 * a.shape[0] // a.size)) if a.size >= 8 else a.shape[0] for a in operands
    )
    matrices = min(experts, matrices)
    return tuple(((matrices, *a.shape[1:]), a.dtype) for a in operands)


def _nax_face_plans(experts: int, k: int, n: int) -> tuple:
    """Every distinct NAX sorted-gather plan reachable for ``(experts, k, n)``."""
    from ..patches import m5_gather_qmm_nax as nax

    e = max(1, experts)
    samples = (16, 36 * e, 48 * e, 96 * e, 120 * e)
    return tuple(dict.fromkeys(nax.face_plan(r, experts, k, n) for r in samples))


def _record_proof(proofs: dict, inst, proof_obj, metal) -> None:
    targets = proofs.setdefault(inst, [])
    entry = (proof_obj, metal)
    if entry not in targets:
        targets.append(entry)


def _plan_switch_warm(
    stand_in,
    k: int,
    target_dtypes: tuple[mx.Dtype, ...],
    per_rows: tuple[int, ...],
    epilogue_limit,
    instances: dict,
    proofs: dict,
    proof_tasks: list,
) -> None:
    """Collect rows, tile_gather and optional NAX epilogue warms for a switch layer."""
    plan = _plan(stand_in)
    if plan is not None:
        op_specs = _switch_stand_in_specs(plan.operands)
        op_shapes = tuple(s for s, _ in op_specs)
        n = plan.n
        for dtype in target_dtypes:
            if not plan.face.exact_for(dtype):
                continue
            for per_row in dict.fromkeys(per_rows):
                few_opts = (False,) if per_row >= 8 else (False, True)
                for few in few_opts:
                    rows = 1 if few else -(-8 // per_row)
                    pairs = rows * per_row
                    in_specs = (((rows, k), dtype), *op_specs, ((pairs,), mx.uint32))
                    in_shapes = ((rows, k), *op_shapes, (pairs,))
                    inst = kernels.planned(
                        plan.kernel,
                        in_specs,
                        [dtype],
                        template=[
                            ("T", dtype),
                            ("K_SIZE", k),
                            ("N_SIZE", n),
                            ("PAIRS_PER_ROW", per_row),
                        ],
                    )

                    def switch_recipe(
                        in_shapes=in_shapes,
                        out_shapes=((pairs, n),),
                        grid=(32, 2 * pairs, 1),
                    ):
                        return kernels.StandIn(
                            input_shapes=in_shapes,
                            output_shapes=out_shapes,
                            grid=grid,
                            threadgroup=(32, 2, 1),
                        )

                    instances.setdefault(inst, switch_recipe)
                    _record_proof(proofs, inst, plan.proof, (dtype, per_row, few))

    tile = _tile(stand_in)
    if tile is None:
        return
    experts = tile.operands[0].shape[0]
    n = stand_in.output_dims
    op_specs = _switch_stand_in_specs(tile.operands)
    if not _nax_host():
        from . import tile_gather

        if tile.proof.failed or k % 32 or n <= 0 or experts <= 0:
            return
        mm = tile.kernel
        for dtype in target_dtypes:
            if not tile.face.exact_for(dtype):
                continue
            metal = (dtype, k, n % 32 == 0)

            def build_steel(stream, mm=mm, dtype=dtype, k=k, op_specs=op_specs):
                arrays = kernels.zeros(
                    ((16, 1, k), dtype), ((16,), mx.uint32), *op_specs, stream=stream
                )
                return arrays, lambda x, idx, *ops: tile_gather.launch(mm, x, ops, idx)

            proof_tasks.append((tile.proof, metal, build_steel))
        return

    from ..patches import m5_gather_qmm_nax as nax

    if not (0 < experts <= nax._MAX_EXPERTS and n > 0 and n % 32 == 0 and k % 32 == 0):
        return
    configs = _nax_face_plans(experts, k, n)
    mm = tile.kernel
    for dtype in target_dtypes:
        if not tile.face.exact_for(dtype):
            continue
        if not tile.proof.failed:
            for config in configs:
                metal = (dtype, k, config, n % 64 == 0)

                def build_nax(
                    stream, mm=mm, dtype=dtype, k=k, op_specs=op_specs, config=config
                ):
                    arrays = kernels.zeros(
                        ((16, 1, k), dtype),
                        ((16,), mx.uint32),
                        *op_specs,
                        stream=stream,
                    )
                    return (
                        arrays,
                        lambda x, idx, *ops: nax._launch_face(
                            mm, x, ops, idx, config, stream
                        ),
                    )

                proof_tasks.append((tile.proof, metal, build_nax))
        if epilogue_limit is not _NO_EPILOGUE and n % 64 == 0 and plan is not None:
            epi_mm = nax._face_kernel(tile.face, "act_map")
            if epi_mm is None:
                continue
            epi_proof = _epilogue_proof(tile, plan, mapped=True)
            if epi_proof.failed:
                continue
            epi = nax._EPI_SWIGLU if epilogue_limit is None else nax._EPI_CLAMPED
            limit = epilogue_limit
            for config in configs:
                metal = (dtype, k, config, epi, True)

                def build_epi(
                    stream,
                    epi_mm=epi_mm,
                    dtype=dtype,
                    k=k,
                    op_specs=op_specs,
                    config=config,
                    epi=epi,
                    limit=limit,
                ):
                    arrays = kernels.zeros(
                        ((1, 1, k), dtype),
                        ((16,), mx.uint32),
                        ((16,), mx.uint32),
                        *op_specs,
                        stream=stream,
                    )
                    return (
                        arrays,
                        lambda x, idx, rmap, *ops: nax._launch_face(
                            epi_mm, x, ops, idx, config, stream, epi, limit, rmap
                        ),
                    )

                proof_tasks.append((epi_proof, metal, build_epi))


def _plan_dense_warm(
    stand_in,
    fmt,
    bits: int,
    group_size: int,
    k: int,
    target_dtypes: tuple[mx.Dtype, ...],
    instances: dict,
    proofs: dict,
) -> None:
    """Collect rows, shared rows, and scheduled dense tile warms for a FormatLinear."""
    n = stand_in.output_dims
    is_stackable = fmt.mode == "affine" or (
        n > 16384 and formats.stackable(fmt, k, n, bits, group_size)
    )
    plan = _plan(stand_in)
    dot_face = None if plan is None else plan.face
    dense_plan = tile_dense.build_plan(
        stand_in, tensor_units=_nax_host(), dot_face=dot_face
    )
    if plan is not None:
        _, op_specs = tile_dense._stand_in_operands(
            plan.operands, n, min(n, 8), stackable=is_stackable
        )
        op_shapes = tuple(s for s, _ in op_specs)
        for dtype in target_dtypes:
            if not plan.face.exact_for(dtype):
                continue
            if n % 8 == 0:
                in_shapes = ((k,), *op_shapes)
                inst = kernels.planned(
                    plan.kernel,
                    (((k,), dtype), *op_specs),
                    [dtype],
                    template=[("T", dtype), ("K_SIZE", k), ("N_SIZE", n)],
                )

                def row_recipe(in_shapes=in_shapes):
                    # One tile of outputs: the rows kernel's first 8, one threadgroup.
                    return kernels.StandIn(
                        input_shapes=in_shapes,
                        output_shapes=((8,),),
                        grid=(32, 2, 1),
                        threadgroup=(32, 2, 1),
                    )

                instances.setdefault(inst, row_recipe)
                _record_proof(proofs, inst, plan.proof, dtype)
            if plan.shared is not None:
                groups = set()
                if n % 8:
                    groups.add(1)
                for r in range(2, MAX_ROWS + 1):
                    if (
                        dense_plan is None
                        or tile_dense.route_for(dense_plan, r, dtype).backend == "dot"
                    ):
                        g = _group(r, plan.face.vpt, n)
                        if g > 1:
                            groups.add(g)
                for group in sorted(groups):
                    in_shapes = ((group, k), *op_shapes)
                    inst = kernels.planned(
                        plan.shared,
                        (((group, k), dtype), *op_specs),
                        [dtype],
                        template=[
                            ("T", dtype),
                            ("K_SIZE", k),
                            ("N_SIZE", n),
                            ("ROWS", group),
                            ("VPT", plan.face.vpt),
                        ],
                    )

                    def shared_recipe(in_shapes=in_shapes, out_shapes=((group, n),)):
                        return kernels.StandIn(
                            input_shapes=in_shapes,
                            output_shapes=out_shapes,
                            grid=(32, 2, 1),
                            threadgroup=(32, 2, 1),
                        )

                    instances.setdefault(inst, shared_recipe)
                    _record_proof(proofs, inst, plan.proof, (dtype, group))

    if dense_plan is not None:
        td_instances, td_proofs = tile_dense.warm_specs(
            stand_in,
            target_dtypes,
            tensor_units=_nax_host(),
            stackable=is_stackable,
            plan=dense_plan,
        )
        for inst, recipe in td_instances.items():
            instances.setdefault(inst, recipe)
            for proof_obj, metal in td_proofs.get(inst, ()):
                _record_proof(proofs, inst, proof_obj, metal)


def warm_layer(
    layer_cls,
    fmt,
    bits,
    group_size,
    k,
    tensors,
    layout,
    *,
    dtypes=None,
    per_rows=(1,),
    epilogue_limit=_NO_EPILOGUE,
):
    """Build, in worker threads, all Metal kernels that ``layer_cls`` serves.

    Covers:
    - ``FormatLinear``: 1-row dot, shared-rows (unaligned N and verify groups),
      ``tile_wide`` (all reachable ``(V, PARTIAL_M)``), and ``tile_dense``;
    - ``FormatSwitchLinear``: switch ``rows``, sorted ``tile_gather`` (steel on
      M3, NAX on M5), and optional NAX mapped SwiGLU epilogue.

    Warms only the checkpoint's floating dtype when provided (or inferred from
    non-placeholder tensors), dropping the stand-in layer immediately so no
    checkpoint tensor is kept by a warm-up. Populates each ``Proof.ran`` once
    its build succeeds so the first served call pays zero compile proofs.

    Returns the submitted futures, or None when nothing applies.
    """
    if issubclass(layer_cls, FormatEmbedding) or k < 8:
        return None
    codes = tensors.get(fmt.codes_name)
    if codes is not None and codes.ndim >= 2 and codes.shape[-2] == 0:
        return None
    warm_dtypes = _infer_warm_dtypes(tensors, layout, dtypes)
    if not warm_dtypes:
        return None
    stored = tuple(name for name, (_, dtype) in (layout or {}).items() if dtype is None)
    probes = (
        tuple(
            (
                dtype,
                {
                    name: mx.zeros(t.shape, dtype) if name in stored else t
                    for name, t in tensors.items()
                },
            )
            for dtype in warm_dtypes
        )
        if stored
        else ((None, tensors),)
    )
    instances: dict = {}
    proofs: dict = {}
    proof_tasks: list = []
    stacked = bool(getattr(layer_cls, "_STACK", 0))
    for probe_dtype, probe in probes:
        stand_in = layer_cls(
            fmt,
            bits,
            group_size,
            k,
            probe,
            None,
            placeholders=False,
            _load_layout=layout,
        )
        target_dtypes = (probe_dtype,) if probe_dtype is not None else warm_dtypes
        if stacked:
            _plan_switch_warm(
                stand_in,
                k,
                target_dtypes,
                per_rows,
                epilogue_limit,
                instances,
                proofs,
                proof_tasks,
            )
        else:
            _plan_dense_warm(
                stand_in,
                fmt,
                bits,
                group_size,
                k,
                target_dtypes,
                instances,
                proofs,
            )
    futures = []
    if instances:
        futures.extend(kernels.warm(_attach_proofs(instances, proofs)))
    if proof_tasks:
        futures.extend(kernels.warm_proofs(proof_tasks))
    return tuple(futures) if futures else None


def _attach_proofs(instances: dict, proofs: dict) -> dict:
    return {
        inst: (kernels._WarmRecipe(recipe, proofs[inst]) if inst in proofs else recipe)
        for inst, recipe in instances.items()
    }


def warm_existing_switch(
    layer: FormatSwitchLinear,
    *,
    dtypes: tuple[mx.Dtype, ...],
    per_rows: tuple[int, ...] = (1,),
    epilogue_limit=_NO_EPILOGUE,
):
    """Submit warm futures for an already-loaded (e.g. post-fusion) switch layer."""
    warm_dtypes = tuple(dict.fromkeys(dt for dt in dtypes if dt in _WARM_DTYPES))
    if not warm_dtypes or layer.input_dims < 8 or layer.output_dims <= 0:
        return None
    instances: dict = {}
    proofs: dict = {}
    proof_tasks: list = []
    _plan_switch_warm(
        layer,
        layer.input_dims,
        warm_dtypes,
        per_rows,
        epilogue_limit,
        instances,
        proofs,
        proof_tasks,
    )
    futures = []
    if instances:
        futures.extend(kernels.warm(_attach_proofs(instances, proofs)))
    if proof_tasks:
        futures.extend(kernels.warm_proofs(proof_tasks))
    return tuple(futures) if futures else None


def warm_existing_dense(
    model,
    *,
    dtypes: tuple[mx.Dtype, ...] = (mx.float16,),
):
    """Submit warm futures for every already-loaded dense ``FormatLinear`` in ``model``."""
    warm_dtypes = tuple(dict.fromkeys(dt for dt in dtypes if dt in _WARM_DTYPES))
    if not warm_dtypes:
        return None
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return None
    instances: dict = {}
    proofs: dict = {}
    seen: set[tuple] = set()
    for _, mod in named_modules():
        if not isinstance(mod, FormatLinear):
            continue
        k, n = mod.input_dims, mod.output_dims
        if k < 8 or n <= 0:
            continue
        tensor_dtypes = tuple(
            (name, mod[name].dtype) for name in mod._tensors if name in mod
        )
        key = (
            mod.fmt.mode,
            mod.bits,
            mod.group_size,
            k,
            n,
            tensor_dtypes,
        )
        if key in seen:
            continue
        seen.add(key)
        _plan_dense_warm(
            mod,
            mod.fmt,
            mod.bits,
            mod.group_size,
            k,
            warm_dtypes,
            instances,
            proofs,
        )
    if not instances:
        return None
    return tuple(kernels.warm(_attach_proofs(instances, proofs)))


def _plan_existing_dense_warm(
    layer: FormatLinear,
    dtypes: tuple[mx.Dtype, ...],
    instances: dict,
    proofs: dict,
) -> None:
    """Collect load-time Metal warms for an already-loaded ``FormatLinear``."""
    warm_dtypes = tuple(dict.fromkeys(dt for dt in dtypes if dt in _WARM_DTYPES))
    k, n = layer.input_dims, layer.output_dims
    if not warm_dtypes or k < 8 or n <= 0:
        return
    _plan_dense_warm(
        layer,
        layer.fmt,
        layer.bits,
        layer.group_size,
        k,
        warm_dtypes,
        instances,
        proofs,
    )


def _prepare_linear_plan(
    mod: FormatLinear, tensor_units: bool, schedule_cache: dict
) -> None:
    kept = mod.__dict__.get(_KEPT)
    if kept is None or kept.weight is not mod[packed_name(mod)]:
        kept = mod._keep()
    if kept._slots.get("dense", _UNBUILT) is _UNBUILT:
        if mod.input_dims <= 0 or mod.output_dims <= 0:
            kept._slots.setdefault("dense", None)
        else:
            built = tile_dense.build_plan(
                mod,
                tensor_units=tensor_units,
                dot_face=None if kept.plan is None else kept.plan.face,
                _schedule_cache=schedule_cache,
            )
            kept._slots.setdefault("dense", built)


def prepare_layer_plans(model, *, extra_linears: tuple[FormatLinear, ...] = ()) -> None:
    """Resolve construction-time plans for every format layer in ``model``.

    Thread-independent: populates each ``FormatLinear``'s ``_Kept`` and dense
    schedule ``DensePlan`` (deduplicating per-chip schedule lookups across
    identical layer geometries within this pass), and each
    ``FormatSwitchLinear``'s ``_Kept`` and sorted ``_Tile``. Fails closed if
    any layer still holds an unloaded placeholder.
    """
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return
    tensor_units = _nax_host()
    schedule_cache: dict = {}
    for _, mod in named_modules():
        if isinstance(mod, FormatLinear):
            _prepare_linear_plan(mod, tensor_units, schedule_cache)
        elif isinstance(mod, FormatSwitchLinear):
            kept = mod.__dict__.get(_KEPT)
            if kept is None or kept.weight is not mod[packed_name(mod)]:
                kept = mod._keep()
            mod._sorted_tile(kept)
    for mod in extra_linears:
        if isinstance(mod, FormatLinear):
            _prepare_linear_plan(mod, tensor_units, schedule_cache)


def _infer_model_dtype(model) -> mx.Dtype | None:
    """First serving floating dtype (bf16/fp16) found on ``model``'s modules.

    Format layers' fixed-dtype storages (e.g. an fp16 ``scales``) are not the
    activation dtype: scan non-format module arrays and format layers' stored
    floats / bias first, falling back to other ``_tensors`` only if none match.
    """
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return None
    modules = list(named_modules())
    for _, mod in modules:
        if isinstance(mod, _FormatLayer):
            for name in (*mod._stored_floats, "bias"):
                arr = mod.get(name)
                if isinstance(arr, mx.array) and arr.dtype in _WARM_DTYPES:
                    return arr.dtype
        elif isinstance(mod, nn.Module):
            for val in dict.values(mod):
                if isinstance(val, mx.array) and val.dtype in _WARM_DTYPES:
                    return val.dtype
    for _, mod in modules:
        if isinstance(mod, _FormatLayer):
            for name in mod._tensors:
                arr = mod.get(name)
                if isinstance(arr, mx.array) and arr.dtype in _WARM_DTYPES:
                    return arr.dtype
    return None


def _layer_warm_dtypes(
    layer: _FormatLayer,
    model_dtype: mx.Dtype | None,
    explicit: tuple[mx.Dtype, ...] | None,
    *,
    fp16_recast: bool,
) -> tuple[mx.Dtype, ...]:
    if explicit is not None:
        return explicit
    if fp16_recast:
        return (mx.float16,)
    found = [
        arr.dtype
        for name in (*layer._stored_floats, "bias")
        if isinstance((arr := layer.get(name)), mx.array) and arr.dtype in _WARM_DTYPES
    ]
    if found:
        return tuple(dt for dt in _WARM_DTYPES if dt in found)
    if model_dtype is not None:
        return (model_dtype,)
    return _WARM_DTYPES


def _warm_linear_rows_graph(
    plan: _Plan | None,
    x_shape: tuple[int, ...],
    dtype: mx.Dtype,
    *,
    key: tuple,
) -> bool:
    """Compile and keep a rows graph on this thread if its Metal is warm."""
    if plan is None or plan.proof.failed or not plan.face.exact_for(dtype):
        return False
    if key in plan.proof.graphs:
        return True
    x_dummy = mx.zeros(x_shape, dtype)
    metal, graph = _linear(plan, x_dummy)
    if metal not in plan.proof.ran:
        return False
    compiled = mx.compile(graph)
    compiled(x_dummy, *plan.operands)
    plan.proof.keep(key, compiled)
    return True


def _warm_switch_prefill_graph(
    layer: FormatSwitchLinear,
    kept: _Kept,
    rows: int,
    dtype: mx.Dtype,
    *,
    seq_len: int | None = None,
    epilogue_limit=_NO_EPILOGUE,
) -> None:
    """Compile and keep sorted prefill switch graphs on this thread if warm."""
    plan = kept.plan
    tile = layer._sorted_tile(kept)
    if plan is None or tile is None or not tile.face.exact_for(dtype):
        return
    experts = plan.operands[0].shape[0]
    k, n = plan.k, plan.n
    if rows < 16 or rows < 4 * experts or k % 32 or n <= 0 or experts <= 0:
        return
    nax_host = _nax_host()
    if nax_host:
        from ..patches import m5_gather_qmm_nax as nax

        if experts > nax._MAX_EXPERTS or n % 32:
            return
    if not tile.proof.failed:
        x_shape = (rows, 1, k)
        idx_shape = (rows,)
        tile_sig = (x_shape, dtype, idx_shape, mx.uint32, n, experts)
        if tile_sig not in tile.proof.graphs:
            x_dummy = mx.zeros(x_shape, dtype)
            idx_dummy = mx.zeros(idx_shape, mx.uint32)
            metal, graph = _sorted(plan, tile.kernel, x_dummy, idx_dummy)
            if metal in tile.proof.ran:
                compiled = mx.compile(graph)
                compiled(x_dummy, idx_dummy, *tile.operands)
                tile.proof.keep(tile_sig, compiled)
    if (
        nax_host
        and epilogue_limit is not _NO_EPILOGUE
        and kept.bias is None
        and seq_len is not None
        and seq_len > 0
        and n % 64 == 0
        and seq_len * k < 2**32
    ):
        if epilogue_limit is not None and not (
            isinstance(epilogue_limit, (int, float))
            and not isinstance(epilogue_limit, bool)
            and math.isfinite(epilogue_limit)
        ):
            return
        epi_proof = _epilogue_proof(tile, plan, mapped=True)
        if epi_proof.failed:
            return
        tok_shape = (seq_len, 1, k)
        epi_sig = (
            "swiglu",
            tok_shape,
            dtype,
            (rows,),
            n,
            experts,
            epilogue_limit,
            True,
        )
        if epi_sig in epi_proof.graphs:
            return
        from ..patches import m5_gather_qmm_nax as nax

        kernel = nax._face_kernel(tile.face, "act_map")
        if kernel is None:
            return
        config = nax.face_plan(rows, experts, k, n)
        epi = nax._EPI_SWIGLU if epilogue_limit is None else nax._EPI_CLAMPED
        metal = (dtype, k, config, epi, True)
        if metal not in epi_proof.ran:
            return
        tok_dummy = mx.zeros(tok_shape, dtype)
        idx_dummy = mx.zeros((rows,), mx.uint32)
        rmap_dummy = mx.zeros((rows,), mx.uint32)

        def epi_graph(x, indices, row_map, *operands):
            return nax._launch_face(
                kernel, x, operands, indices, config, None, epi, epilogue_limit, row_map
            )

        compiled = mx.compile(epi_graph)
        compiled(tok_dummy, idx_dummy, rmap_dummy, *tile.operands)
        epi_proof.keep(epi_sig, compiled)


def _warm_scheduled_shape(
    mod: FormatLinear,
    kept: _Kept,
    x_shape: tuple[int, ...],
    rows: int,
    dtype: mx.Dtype,
) -> None:
    """Warm one multi-row (rows > 1) FormatLinear dispatch signature."""
    dense = kept._slots.get("dense", _UNBUILT)
    if dense is not None and dense is not _UNBUILT and rows < dense.row_limit:
        route = tile_dense.route_for(dense, rows, dtype)
        if route.backend == "dot":
            if rows <= MAX_ROWS and kept.plan is not None:
                sig = (x_shape, dtype)
                if _warm_linear_rows_graph(kept.plan, x_shape, dtype, key=sig):
                    mod._register_dot_fast(kept, kept.plan, sig)
            return
        if tile_dense.warm_graph(dense, x_shape, dtype):
            mod._register_fast(kept, dense, (x_shape, dtype))
        return
    if rows <= MAX_ROWS and kept.plan is not None:
        sig = (x_shape, dtype)
        if _warm_linear_rows_graph(kept.plan, x_shape, dtype, key=sig):
            mod._register_dot_fast(kept, kept.plan, sig)


def _warm_format_linear_dispatch(
    mod: FormatLinear,
    *,
    model_dtype: mx.Dtype | None,
    explicit: tuple[mx.Dtype, ...] | None,
    fp16_recast: bool,
    batches: tuple[int, ...],
    verifies: tuple[int, ...],
    prefills: tuple[int, ...],
) -> None:
    kept = mod.__dict__.get(_KEPT)
    if kept is None:
        return
    k, n = mod.input_dims, mod.output_dims
    if k < 8 or n <= 0:
        return
    layer_dtypes = _layer_warm_dtypes(
        mod, model_dtype, explicit, fp16_recast=fp16_recast
    )
    seq_rows = tuple(
        dict.fromkeys(
            (
                *batches,
                *verifies,
                *(b * s for b in batches for s in verifies if b * s <= MAX_ROWS),
            )
        )
    )
    for dtype in layer_dtypes:
        # 1. Single-row decode (1, 1, K): M3 C7 single-row mlx_rows bind on
        #    dense_fast; all other layers bind on dot_fast so warm 1-token
        #    decode hits native FormatLinearFast directly on entry.
        sig1 = ((1, 1, k), dtype)
        if 1 in kept.mlx_rows:
            dense = kept._slots.get("dense")
            if dense is not None:
                mod._bind_one_row_mlx_fast(kept, dense, sig1)
        elif kept.plan is not None:
            if _warm_linear_rows_graph(kept.plan, (1, 1, k), dtype, key=sig1):
                mod._register_dot_fast(kept, kept.plan, sig1)
        # 2. Multi-row decode (B, 1, K), batched verify (B, S, K), and
        #    single-stream / GDN-stacked verify (1, S, K) — every FormatLinear
        #    follows its construction-time schedule and registers on dense_fast
        #    or dot_fast (published into _fast/_fast2).
        for b in batches:
            _warm_scheduled_shape(mod, kept, (b, 1, k), b, dtype)
            for s in verifies:
                rows = b * s
                if rows <= MAX_ROWS:
                    _warm_scheduled_shape(mod, kept, (b, s, k), rows, dtype)
        for s in seq_rows:
            _warm_scheduled_shape(mod, kept, (1, s, k), s, dtype)
        # 3. Prefill (1, L, K) -> scheduled dense graph + native FormatLinearFast.
        for length in prefills:
            _warm_scheduled_shape(mod, kept, (1, length, k), length, dtype)


def warm_model_dispatch(
    model,
    *,
    dtypes: tuple[mx.Dtype, ...] | None = None,
    prefill_lengths: tuple[int, ...] = (2047, 2048),
    decode_batch_sizes: tuple[int, ...] = (1, 2, 4),
    mtp_enabled: bool = False,
    verify_rows: tuple[int, ...] = (2, 3, 4),
    extra_linears: tuple[FormatLinear, ...] = (),
) -> None:
    """Populate per-layer plans, compiled graphs and native fast dispatch on
    the calling thread so the first served request does none of that work.

    Must run on the thread that will serve requests (``Proof._kept`` and MLX's
    compile cache are thread-local). Only traces graphs whose Metals are
    already warmed in ``Proof.ran``; never inserts an unproven Metal.
    """
    if model is None or not mx.metal.is_available():
        return
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return

    prepare_layer_plans(model, extra_linears=extra_linears)

    explicit = (
        tuple(dict.fromkeys(dt for dt in dtypes if dt in _WARM_DTYPES))
        if dtypes is not None
        else None
    )
    if explicit is not None and not explicit:
        return
    fp16_recast = bool(getattr(model, "_omlx_fp16_recast", False))
    model_dtype = (
        None if explicit is not None or fp16_recast else _infer_model_dtype(model)
    )
    prefills = tuple(
        dict.fromkeys(r for r in prefill_lengths if type(r) is int and r > MAX_ROWS)
    )
    batches = tuple(
        dict.fromkeys(
            b for b in decode_batch_sizes if type(b) is int and 1 < b <= MAX_ROWS
        )
    )
    verifies = (
        tuple(
            dict.fromkeys(
                s for s in verify_rows if type(s) is int and 1 < s <= MAX_ROWS
            )
        )
        if mtp_enabled
        else ()
    )

    from ..patches import m5_gather_qmm

    for _, mod in named_modules():
        if isinstance(mod, FormatLinear):
            _warm_format_linear_dispatch(
                mod,
                model_dtype=model_dtype,
                explicit=explicit,
                fp16_recast=fp16_recast,
                batches=batches,
                verifies=verifies,
                prefills=prefills,
            )
            continue

        # 4. MoE FormatSwitchLinear sorted prefill (+ M5 NAX mapped SwiGLU).
        switch_mlp = getattr(mod, "switch_mlp", None)
        top_k = getattr(mod, "top_k", None)
        num_experts = getattr(mod, "num_experts", None)
        if (
            not isinstance(switch_mlp, nn.Module)
            or type(top_k) is not int
            or top_k <= 0
            or type(num_experts) is not int
            or num_experts <= 0
            or not prefills
        ):
            continue
        gate_up = switch_mlp.get("gate_up_proj")
        down = switch_mlp.get("down_proj")
        if isinstance(gate_up, FormatSwitchLinear):
            kept_gu = gate_up.__dict__.get(_KEPT)
            if kept_gu is not None:
                epi_limit = _NO_EPILOGUE
                if "bias" not in gate_up and "global_scale" not in gate_up:
                    lim = m5_gather_qmm._swiglu_limit(
                        getattr(switch_mlp, "activation", None)
                    )
                    if lim is not m5_gather_qmm._UNSUPPORTED and (
                        lim is None
                        or (
                            isinstance(lim, (int, float))
                            and not isinstance(lim, bool)
                            and math.isfinite(lim)
                        )
                    ):
                        epi_limit = lim
                gu_dtypes = _layer_warm_dtypes(
                    gate_up, model_dtype, explicit, fp16_recast=fp16_recast
                )
                for dtype in gu_dtypes:
                    for length in prefills:
                        _warm_switch_prefill_graph(
                            gate_up,
                            kept_gu,
                            length * top_k,
                            dtype,
                            seq_len=length,
                            epilogue_limit=epi_limit,
                        )
        elif gate_up is None:
            for proj_name in ("gate_proj", "up_proj"):
                proj = switch_mlp.get(proj_name)
                if isinstance(proj, FormatSwitchLinear):
                    kept_p = proj.__dict__.get(_KEPT)
                    if kept_p is not None:
                        p_dtypes = _layer_warm_dtypes(
                            proj, model_dtype, explicit, fp16_recast=fp16_recast
                        )
                        for dtype in p_dtypes:
                            for length in prefills:
                                _warm_switch_prefill_graph(
                                    proj, kept_p, length * top_k, dtype
                                )
        if isinstance(down, FormatSwitchLinear):
            kept_dn = down.__dict__.get(_KEPT)
            if kept_dn is not None:
                dn_dtypes = _layer_warm_dtypes(
                    down, model_dtype, explicit, fp16_recast=fp16_recast
                )
                for dtype in dn_dtypes:
                    for length in prefills:
                        _warm_switch_prefill_graph(
                            down,
                            kept_dn,
                            length * top_k,
                            dtype,
                        )

    # Extra runtime-stacked FormatLinear layers (e.g. Qwen4 GDN stacked
    # in-projections, used for 2..16-row batch decode and MTP verify, never prefill).
    for mod in extra_linears:
        if isinstance(mod, FormatLinear):
            _warm_format_linear_dispatch(
                mod,
                model_dtype=model_dtype,
                explicit=explicit,
                fp16_recast=fp16_recast,
                batches=batches,
                verifies=verifies,
                prefills=(),
            )


def _launch(plan: _Plan | None, signature: tuple, describe, stand_in, *arrays):
    """``arrays`` (x, then a stack's indices) through the rows kernel, the
    first time for ``signature``, or None: the layer has no plan, its proof
    has failed, the face is not exact for x's type, there is nothing to
    launch, or the launch does not compile.

    The launch's graph (``describe``) is compiled by MLX and kept under
    ``signature``, which is what it depends on (the arrays' shapes, x's
    type and the indices' type), by the thread that builds it: a layer's
    call asks for the graph its thread kept first, and replays it.

    The first time its Metal is built, it is proven on ``stand_in``: the
    smallest launch that MLX generates the same Metal for, over zeros in
    the place of x, of the indices and of the layer's tensors. Nothing of
    the call is evaluated here: not ``arrays``, not a tensor, not the
    result. Any of them may belong to a trace (a model calls its layers in
    functions that MLX compiles, over its state too), where it cannot be
    evaluated, and what fails in one is raised where the caller evaluates
    the result."""
    if plan is None or plan.proof.failed or not plan.face.exact_for(arrays[0].dtype):
        return None
    launch = describe(plan, *arrays)
    if launch is None:
        return None
    metal, graph = launch
    proof = plan.proof
    if not proof.prove(metal, lambda: stand_in(plan, metal), logger, *plan.said):
        return None
    graph = mx.compile(graph)
    y = graph(*arrays, *plan.operands)
    proof.keep(signature, graph)
    return y


def _sorts(plan: _Plan, x, indices) -> bool:
    """Check MLX's prefill threshold and the geometry of this GPU's kernel."""
    experts = plan.operands[0].shape[0]
    rows = indices.size
    if rows < 16 or rows < 4 * experts or not x.ndim or x.shape[-1] != plan.k:
        return False
    if _nax_host():
        from ..patches import m5_gather_qmm_nax as nax

        return nax.geometry(x, indices, experts, plan.n)
    return (
        x.shape == (rows, 1, plan.k)
        and indices.ndim == 1
        and indices.dtype == mx.uint32
        and x.dtype in (mx.float16, mx.bfloat16)
        and plan.k % 32 == 0
        and plan.n > 0
        and experts > 0
        and rows < 1 << 30
    )


def _sorted(plan: _Plan, kernel, x, indices):
    """A stack's launch of its sorted rows through the tile face, as
    ``_switch`` gives the rows kernel's. The Metal holds the call's dtype,
    K and tile plan."""
    if not _nax_host():
        from . import tile_gather

        def steel(x, indices, *operands):
            return tile_gather.launch(kernel, x, operands, indices)

        return (x.dtype, plan.k, plan.n % 32 == 0), steel

    from ..patches import m5_gather_qmm_nax as nax

    experts = plan.operands[0].shape[0]
    config = nax.face_plan(indices.size, experts, plan.k, plan.n)

    def graph(x, indices, *operands):
        return nax._launch_face(kernel, x, operands, indices, config, None)

    return (x.dtype, plan.k, config, plan.n % 64 == 0), graph


def _sorted_stand_in(tile: _Tile, metal, graph, mapped=False):
    """Prove the call's tile plan on 16 rows of expert zero. A mapped launch
    reads those rows from one token. Use the fewest matrices that leave
    eight elements in every operand."""
    dtype, k = metal[:2]
    matrices = max(-(-8 * a.shape[0] // a.size) for a in tile.operands)
    operands = (((matrices, *a.shape[1:]), a.dtype) for a in tile.operands)
    rows = 1 if mapped else 16
    specs = (((rows, 1, k), dtype), ((16,), mx.uint32))
    if mapped:
        specs += (((16,), mx.uint32),)
    return kernels.zeros(*specs, *operands), graph


def _launch_sorted(plan: _Plan, tile: _Tile, signature: tuple, x, indices):
    """``x`` and its sorted ``indices`` through the tile route, as
    ``_launch`` runs the rows kernel; None where the face is not exact for
    x's type or its proof has failed, for the rows kernel to take it."""
    if not tile.face.exact_for(x.dtype):
        return None
    metal, graph = _sorted(plan, tile.kernel, x, indices)
    return _keep_sorted(tile, tile.proof, signature, metal, graph, (x, indices))


def _epilogue_proof(tile: _Tile, plan: _Plan, mapped: bool) -> kernels.Proof:
    """The epilogue's own proof, apart from the tile gather's: one mapping
    failing to compile must not retire the other, or the gather. K is part
    of it: one face serves layers of different width."""
    route = "format_switch_tile_nax_act_map" if mapped else "format_switch_tile_nax_act"
    return kernels.proof(
        route, tile.face, tuple(a.dtype for a in tile.operands), plan.k
    )


def _keep_sorted(tile, proof, signature, metal, graph, arrays, mapped=False, said=None):
    if not proof.prove(
        metal,
        lambda: _sorted_stand_in(tile, metal, graph, mapped),
        logger,
        *(tile.said if said is None else said),
    ):
        return None
    graph = mx.compile(graph)
    y = graph(*arrays, *tile.operands)
    proof.keep(signature, graph)
    return y


def _swiglu(plan, tile, x, indices, limit, row_map=None):
    """Replay this activation and mapping, or prove and keep their graph."""
    mapped = row_map is not None
    proof = _epilogue_proof(tile, plan, mapped)
    if proof.failed:
        return None
    experts = plan.operands[0].shape[0]
    signature = (
        "swiglu",
        x.shape,
        x.dtype,
        indices.shape,
        plan.n,
        experts,
        limit,
        mapped,
    )
    arrays = (x, indices, row_map) if mapped else (x, indices)
    graph = proof.graphs.get(signature)
    if graph is not None:
        return graph(*arrays, *tile.operands)
    from ..patches import m5_gather_qmm_nax as nax

    kernel = nax._face_kernel(tile.face, "act_map" if mapped else "act")
    if kernel is None:
        return None
    config = nax.face_plan(indices.size, experts, plan.k, plan.n)
    epi = nax._EPI_SWIGLU if limit is None else nax._EPI_CLAMPED
    metal = (x.dtype, plan.k, config, epi, mapped)
    if mapped:

        def graph(x, indices, row_map, *operands):
            return nax._launch_face(
                kernel, x, operands, indices, config, None, epi, limit, row_map
            )

    else:

        def graph(x, indices, *operands):
            return nax._launch_face(
                kernel, x, operands, indices, config, None, epi, limit
            )

    kind = "mapped epilogue" if mapped else "epilogue"
    said = (
        f"quant format face {tile.face.key}: {kind} kernel engaged",
        f"quant format face {tile.face.key}: {kind} kernel failed; composed fallback",
    )
    return _keep_sorted(tile, proof, signature, metal, graph, arrays, mapped, said)


# Where a layer keeps what its calls work out of it once: its plan and its
# bias.
_KEPT = "_omlx_format_call"
_FAST = "_fast"
_FAST2 = "_fast2"
_NATIVE_FALLBACK_LOGGED = False
_UNBUILT = object()


class _Kept:
    """What one call of a layer works out, kept until a write retires it.

    ``plan`` is the Dot route a single row keeps; ``dense`` is the
    construction-time schedule every call above one row resolves through
    (``_UNBUILT`` until its first multi-row half/bfloat16 call binds it
    atomically into ``_slots``, None once the layer's format has no tile
    face). A layer can run both dense and Dot routes, so each operand group
    has its own native replay bound atomically into ``_slots``: ``dense_fast``
    for every scheduled dense route (including mma/msg, which inline
    ``N_size`` inside their compiled graph) and ``dot_fast`` for the Dot
    plan's route (single row and scheduled Dot rows). A switch layer keeps its sorted tile in
    ``sorted``. ``weight`` is the array the record was worked out over;
    another array retires it whole, as a write of any tensor or attribute
    does."""

    __slots__ = (
        "plan",
        "bias",
        "weight",
        "weight_key",
        "sorted",
        "mlx_rows",
        "_slots",
    )

    def __init__(
        self,
        plan,
        bias,
        weight,
        mlx_rows=frozenset(),
        *,
        weight_key: str = "weight",
    ):
        self.plan = plan
        self.bias = bias
        self.weight = weight
        self.weight_key = weight_key
        self.sorted = _UNBUILT
        self.mlx_rows = mlx_rows
        self._slots: dict[str, Any] = {}

    @property
    def dense(self):
        return self._slots.get("dense", _UNBUILT)

    @property
    def dense_fast(self):
        return self._slots.get("dense_fast")

    @property
    def dot_fast(self):
        return self._slots.get("dot_fast")


def _log_native_fallback() -> None:
    global _NATIVE_FALLBACK_LOGGED
    if not _NATIVE_FALLBACK_LOGGED:
        _NATIVE_FALLBACK_LOGGED = True
        logger.warning(
            "quant_matmul native FormatLinearFast unavailable; "
            "using Python warm dispatch fallback"
        )


def _unloaded(name: str, held, offered, expected) -> str:
    """A placeholder ``held`` of tensor ``name``, as a refusal names it: with
    the shape and dtype last offered in its place, or as never loaded."""
    if offered is None:
        return f"{name} (never loaded)"
    shape, dtype = offered
    if shape is None:
        return f"{name} (offered {dtype}, not an array)"
    return (
        f"{name} (offered {shape} {dtype}, not the layout's {held.shape} "
        f"{expected})"
    )


# Set only around the concrete constructor in an admitted oQ write. Copying
# a token or a context cannot copy another thread's pending construction.
_PENDING_WRITE = local()


def _take_write_permit(*args):
    expected = getattr(_PENDING_WRITE, "construction", None)
    if expected is None or any(a is not b for a, b in zip(args, expected, strict=True)):
        return False
    # An equality check, or even fmt.mode, could enter a plugin.
    _PENDING_WRITE.construction = None
    return True


class _FormatLayer(nn.Module):
    """What the two layers share: a format's tensors as frozen parameters,
    the attributes of MLX's quantized layers, and the format object that
    reads them (``fmt``).

    A layer works out its plan and finds its bias at its first call, and
    keeps them in one record in its ``__dict__`` with the weight they were
    found with, until one of its arrays or attributes is assigned or removed
    (``__setitem__``, ``__delitem__``, ``pop``: what MLX's loads,
    ``update``, ``set_dtype`` and attribute writes go through; a dict's
    own mutators and ``del`` of an attribute too) or its
    weight is another array: a function that ``mx.compile`` traces over the
    layer's state (``inputs=``) swaps every array of it for a trace's, and
    back, behind those methods. A call then reads one dict entry and the
    weight; the routes it proves are named entries of that record, which a
    write retires with it. Threads: a layer that is being loaded is not
    called meanwhile; two threads that make a layer's first call at once
    each work out the same plan, and either keeps it.

    The format's tensors keep the shapes and dtypes the layer is built
    with, which its face reads. MLX writes a module's arrays back one by one
    through ``__setitem__``, for a load (``load_weights``, ``update``) as for
    a recast (``set_dtype``, ``apply``, oMLX's float16 activations), and the
    two cannot be told apart there: an array of another shape or dtype is
    not taken, and raises nothing. A layer built for a load
    (``placeholders``) holds stand-ins that arrays of their shapes and
    dtypes replace, and refuses to compute while one is left."""

    _STACK = 0  # the axes ahead of one matrix's, in each of its tensors

    def __init__(
        self,
        fmt: formats.QuantFormat | str,
        bits: int,
        group_size: int,
        input_dims: int,
        tensors: dict[str, mx.array],
        bias: mx.array | None = None,
        *,
        placeholders: bool = False,
        _load_layout: dict | None = None,
        _write_permit: object | None = None,
    ) -> None:
        if _write_permit is None:
            if not isinstance(fmt, (str, formats.QuantFormat)):
                raise TypeError("a format layer needs a mode name or a QuantFormat")
            formats._may_build(fmt if isinstance(fmt, str) else fmt.mode)
        elif (
            bias is not None
            or placeholders is not False
            or _load_layout is not None
            or not _take_write_permit(
                _write_permit, type(self), fmt, bits, group_size, input_dims, tensors
            )
        ):
            # Even fmt.mode may run plugin code. A refused permit has no
            # trusted mode, and must not enter the plugin.
            raise formats.FormatRefusedError(
                "the write did not admit this layer", mode="unknown"
            )
        if isinstance(fmt, str):
            # A layer built by hand, of a registered mode: its format.
            mode, fmt = fmt, formats.get(fmt)
            if fmt is None:
                raise ValueError(f"no quant format is registered for mode {mode!r}")
        self._init_tensors(
            fmt,
            bits,
            group_size,
            input_dims,
            tensors,
            bias,
            placeholders=placeholders,
            layout=_load_layout,
        )

    def _init_tensors(
        self,
        fmt: formats.QuantFormat,
        bits: int,
        group_size: int,
        input_dims: int,
        tensors: dict[str, mx.array],
        bias: mx.array | None,
        *,
        placeholders: bool = False,
        layout: dict | None = None,
    ) -> None:
        """Initialize an admitted reader at the geometry of these tensors."""
        super().__init__()
        mode = fmt.mode
        names = (fmt.codes_name, *fmt.tensors)
        if set(tensors) != set(names):
            raise ValueError(
                f"quant format {mode!r} keeps a matrix as {', '.join(names)}, "
                f"not as {', '.join(sorted(tensors))}"
            )
        packed = tensors[fmt.codes_name]
        if packed.ndim != self._STACK + 2:
            raise ValueError(
                f"{type(self).__name__}: the codes ({fmt.codes_name!r}) have "
                f"{self._STACK + 2} axes, not the shape {packed.shape}"
            )
        # What the format lays out for the matrix the layer holds: a layer's
        # N is its codes' first axis.
        n = packed.shape[-2]
        if layout is None:
            layout = fmt.layout(input_dims, n, bits, group_size)
        if layout is not None:
            formats.check_layout(fmt, n, input_dims, layout)
        # What lets a stack's matrices be gathered, tensor by tensor.
        stack = packed.shape[: self._STACK]
        ragged = [name for name in names if tensors[name].shape[: self._STACK] != stack]
        if ragged:
            raise ValueError(
                f"{type(self).__name__}: {', '.join(ragged)} does not open with "
                f"the codes' ({fmt.codes_name!r}) stack of {stack}"
            )
        # The object, not a lookup of its mode: a checkpoint's own (what its
        # format's accept returned for its record) reads its layers. Kept out
        # of the module's items whatever its class.
        object.__setattr__(self, "fmt", fmt)
        self.mode, self.bits, self.group_size = mode, bits, group_size
        self.input_dims, self.output_dims = input_dims, n
        # The format's tensors, those a checkpoint stores in a floating dtype
        # of its own, and those that still hold a load's placeholders, each
        # with the shape and dtype last offered in its place (None: none
        # was). Kept beside the parameters, not among them.
        self.__dict__.update(
            _tensors=names,
            _stock=mode in formats._BUILTIN_MODES,
            _stored_floats=tuple(
                name for name, (_, dtype) in (layout or {}).items() if dtype is None
            ),
            _placeholders=dict.fromkeys(names if placeholders else ()),
        )
        for name in names:
            self[name] = tensors[name]
        if bias is not None:
            self["bias"] = bias
        self.freeze()

    def _load_dtype(self, name: str, expected):
        """The dtype ``name`` loads as: ``expected``, or None for a stored
        float, which any floating dtype may fill."""
        return None if name in self._stored_floats else expected

    def _matches_load_tensor(self, name: str, tensor, held) -> bool:
        """Whether ``tensor`` takes the placeholder ``held``: its shape, and
        its dtype (or any floating one, for a stored float)."""
        if tensor.shape != held.shape:
            return False
        expected = self._load_dtype(name, held.dtype)
        if expected is None:
            return mx.issubdtype(tensor.dtype, mx.floating)
        return tensor.dtype == expected

    def __setitem__(self, key: str, value: Any) -> None:
        # Over a tensor of the format the layer holds: an array of its shape
        # and dtype replaces it, any other is not taken, and the very array
        # it holds changes nothing. Only a load (a checkpoint loading on this
        # thread) changes what the layer holds as loaded: its array loads the
        # tensor, and one it does not take leaves the tensor unloaded, named
        # by what was offered. A recast (set_dtype, before or after a load)
        # hands arrays back outside a load: it changes no tensor's status.
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        self.__dict__.pop(_FAST2, None)
        held = self.get(key)
        if held is not None and key in self._tensors:
            if value is held:
                return
            checkpoint = formats._LOADING.get()
            loading = checkpoint is not None and checkpoint.active
            array = isinstance(value, mx.array)
            matches = array and self._matches_load_tensor(key, value, held)
            if not matches or (
                not loading and value.dtype != held.dtype and keeps_dtype(self, key)
            ):
                if loading:
                    unloads = key not in self._placeholders
                    # The dtype's name: a Dtype neither pickles nor copies.
                    self._placeholders[key] = (
                        (value.shape, str(value.dtype))
                        if array
                        else (None, type(value).__name__)
                    )
                    if unloads:
                        # A loaded tensor goes: a plan cached over it is
                        # built again, and refuses.
                        super().__setitem__(key, mx.zeros_like(held))
                return
            # Stored before its placeholder is cleared: a call on another
            # thread that finds no placeholder finds the tensor.
            super().__setitem__(key, value)
            if loading:
                self._placeholders.pop(key, None)
            return
        super().__setitem__(key, value)

    def __copy__(self) -> _FormatLayer:
        """A copy with a record of placeholders of its own: a load into one
        of the two loads nothing into the other."""
        clone = type(self).__new__(type(self))
        dict.update(clone, self)
        clone.__dict__.update(
            self.__dict__,
            _placeholders=dict(self._placeholders),
            # Frozen or not, each on its own.
            _no_grad=set(self._no_grad),
        )
        return clone

    def refuse_placeholders(self) -> None:
        """Raise while a tensor of the format still holds the load's
        placeholder: the layer never computes on one."""
        # A copy: a load on another thread may be emptying the set.
        pending = tuple(self._placeholders.items())
        if not pending:
            return
        said = []
        for name, offered in pending:
            expected = self._load_dtype(name, self[name].dtype)
            said.append(
                _unloaded(
                    name,
                    self[name],
                    offered,
                    "floating dtype" if expected is None else expected,
                )
            )
        held = ", ".join(said)
        raise ValueError(
            f"{self!r} computes on its checkpoint's tensors alone; it still "
            f"holds the load's placeholder of {held}"
        )

    def _stacked_outputs(
        self, other: _FormatLayer, *others: _FormatLayer, layout: dict | None = None
    ) -> _FormatLayer:
        """Join admitted projections along their output axis after layout checks."""
        sources = (self, other, *others)
        for source in sources:
            source.refuse_placeholders()
        tensors = {
            name: mx.concatenate([source[name] for source in sources], axis=self._STACK)
            for name in self._tensors
        }
        bias = (
            mx.concatenate([source["bias"] for source in sources], axis=-1)
            if "bias" in self
            else None
        )
        cls = FormatSwitchLinear if self._STACK else FormatLinear
        layer = cls.__new__(cls)
        layer._init_tensors(
            self.fmt,
            self.bits,
            self.group_size,
            self.input_dims,
            tensors,
            bias,
            layout=layout,
        )
        layer.train(self.training)
        layer._no_grad = set(self._no_grad)
        return layer

    def _keep(self) -> _Kept:
        """The layer's plan and bias, bound once into a record and published
        atomically. Two threads racing on a fresh layer's first call share the
        winning record (and its dense plan, proof binding, and native replays)
        with no lock in the hot path."""
        weight_key = packed_name(self)
        weight = self[weight_key]
        existing = self.__dict__.get(_KEPT)
        if existing is not None and existing.weight is weight:
            return existing
        bias = self.get("bias")
        if "bias" in self and not isinstance(bias, mx.array):
            # MLX's quantized layers raise as they add it.
            raise TypeError(f"the bias is an array, not {type(bias).__name__}")
        built = _Kept(
            _plan(self),
            bias,
            weight,
            (
                _decode_mlx_rows(
                    _metal_architecture(), self.output_dims, self.input_dims
                )
                if self._stock and not self._STACK
                else frozenset()
            ),
            weight_key=weight_key,
        )
        if existing is None:
            kept = self.__dict__.setdefault(_KEPT, built)
            if kept.weight is weight:
                return kept
        self.__dict__.pop(_FAST, None)
        self.__dict__.pop(_FAST2, None)
        self.__dict__[_KEPT] = built
        return built

    def _dense_tile(self, kept: _Kept) -> tile_dense.DensePlan | None:
        """The dense-prefill plan, built once with this chip's schedule and
        bound atomically onto ``kept`` (rule 28); None where the layer's
        format has no face this GPU's tile skeleton takes. Two threads racing
        on the first multi-row call keep the single winning plan and its proof
        binding with no lock in the hot path."""
        dense = kept._slots.get("dense", _UNBUILT)
        if dense is not _UNBUILT:
            return dense
        built = tile_dense.build_plan(
            self,
            tensor_units=_nax_host(),
            dot_face=None if kept.plan is None else kept.plan.face,
        )
        return kept._slots.setdefault("dense", built)

    def __delitem__(self, key: str) -> None:
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        self.__dict__.pop(_FAST2, None)
        super().__delitem__(key)

    def pop(self, *key_and_default):
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        self.__dict__.pop(_FAST2, None)
        return super().pop(*key_and_default)

    # A dict's own mutators, which do not go through the three above.
    def setdefault(self, key: str, default: Any = None) -> Any:
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        self.__dict__.pop(_FAST2, None)
        return super().setdefault(key, default)

    def popitem(self) -> tuple:
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        self.__dict__.pop(_FAST2, None)
        return super().popitem()

    def clear(self) -> None:
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        self.__dict__.pop(_FAST2, None)
        super().clear()

    def __ior__(self, other):
        # A dictionary union accepts mappings and iterables of key/value pairs.
        for key, value in dict(other).items():
            self[key] = value
        return self

    def __delattr__(self, name: str) -> None:
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        self.__dict__.pop(_FAST2, None)
        super().__delattr__(name)

    def _decoded(self, indices: mx.array | None = None) -> mx.array:
        """The layer's weights as floats: all of them, or the matrices at
        ``indices`` of the stack."""
        self.refuse_placeholders()
        fmt = self.fmt
        tensors = {name: self[name] for name in self._tensors}
        if indices is not None:
            tensors = {name: tensor[indices] for name, tensor in tensors.items()}
        return fmt.decode(tensors, self.bits, self.group_size)

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"bias={'bias' in self}, group_size={self.group_size}, "
            f"bits={self.bits}, mode={self.mode}"
        )


def _decode_dtype(layer: FormatSwitchLinear) -> tuple[str, int]:
    """The name and size of the dtype of the layer's decode (a Dtype
    neither copies nor pickles with the layer): read from the graph of one
    matrix's decode, never evaluated."""
    dtype = layer._decoded(slice(0, 1)).dtype
    return str(dtype), dtype.size


def _decoded_size(layer: FormatSwitchLinear, dtype: mx.Dtype) -> int:
    """The bytes of a weight the layer's decode holds for x of ``dtype``:
    the decode's own, and the cast's when x's dtype differs. The decode's
    dtype is read once for the layer."""
    name, size = cached_per_module(layer, "_omlx_decode_dtype", _decode_dtype, depth=0)
    return size + (dtype.size if str(dtype) != name else 0)


def keeps_dtype(module, key: str) -> bool:
    """Whether ``module``'s array ``key`` keeps its dtype through a recast
    of the model: a format layer's tensors (not its bias) are in the dtypes
    of the format's layout, which its face reads. The built-in affine's
    stored floats are the exception: the checkpoint fixes their dtype, and
    the face is exact for float16 activations once they are float16."""
    if not isinstance(module, _FormatLayer) or key not in module._tensors:
        return False
    return not (module.mode in formats._BUILTIN_MODES and key in module._stored_floats)


class FormatLinear(_FormatLayer):
    """``x @ W.T + bias`` for a matrix ``W`` kept in format ``mode``:
    ``tensors`` are its codes (``mode``'s format's ``codes_name``) and the
    format's tensors, by name."""

    _fast = None
    _fast2 = None

    def _stock_matmul(self, x):
        """MLX's own affine matmul over the layer's tensors, for a stock
        checkpoint's matrix: the route a stock layer ran before the switch.
        None for a plugin's format, whose tensors MLX cannot run."""
        if not self._stock or self._STACK:
            return None
        weight = self[self._tensors[0]]
        if weight.ndim != 2:
            return None
        # The caller adds the bias, as it does for every route here.
        return mx.quantized_matmul(
            x,
            weight,
            self["scales"],
            self["biases"],
            transpose=True,
            bits=self.bits,
            group_size=self.group_size,
            mode=self.mode,
        )

    def _scheduled(self, kept, plan, x, bias):
        """The plan's route for this call (with ``bias`` applied once): the
        native replay of the route's graph once it is built, else the route's
        own projection, which builds, proves and keeps the graph under this
        thread's signature. A single row keeps the Dot plan (``dot``)."""
        rows = x.size // plan.k
        if rows >= plan.row_limit:
            return None
        route = tile_dense.route_for(plan, rows, x.dtype)
        if route.backend == "dot":
            return self._dot_rows(kept, x, bias)
        if not plan.face.exact_for(x.dtype):
            return None
        fast = kept._slots.get("dense_fast")
        if fast is not None:
            y = fast(self, x)
            if y is not None:
                cur = self.__dict__.get(_FAST)
                if cur is None or (
                    cur is not fast and self.__dict__.get(_FAST2) is None
                ):
                    self._publish_fast(kept, fast)
                return y
        y = tile_dense.project(plan, x)
        if y is not None:
            self._register_fast(kept, plan, (x.shape, x.dtype))
            return y if bias is None else y + bias
        return None

    def _dot_rows(self, kept, x, bias):
        """A call of 2..MAX_ROWS rows whose route is the Dot plan (with
        ``bias`` applied once): its own native replay once the plan's graph is
        built, else the plan's rows kernel, which proves and keeps it. A
        longer call is no rows route (the chunked ordinary rows kernel it
        replaced is gone)."""
        plan = kept.plan
        if plan is None or not plan.face.exact_for(x.dtype):
            return None
        if x.size // plan.k > MAX_ROWS:
            return None
        signature = (x.shape, x.dtype)
        graph = plan.proof.graphs.get(signature)
        if graph is not None:
            if "no_dot_fast" not in kept._slots:
                self._register_dot_fast(kept, plan, signature)
            y = graph(x, *plan.operands)
            return y if bias is None else y + bias
        y = _launch(plan, signature, _linear, _linear_stand_in, x)
        if y is not None:
            self._register_dot_fast(kept, plan, signature)
            return y if bias is None else y + bias
        return None

    def _native_fast(self, kept, slot, operands, proof_state):
        """The native replay kept under ``slot``, or one bound over these
        operands and proof state and published atomically; None when the
        extension has none."""
        fast = kept._slots.get(slot)
        if fast is not None:
            return fast
        no_slot = f"no_{slot}"
        if no_slot in kept._slots:
            return None
        if len(operands) >= 16:
            kept._slots[no_slot] = True
            return None
        try:
            from ..custom_kernels.quant_matmul import fast as native
        except ImportError:
            _log_native_fallback()
            kept._slots[no_slot] = True
            return None
        if not native.has_symbol("FormatLinearFast"):
            _log_native_fallback()
            kept._slots[no_slot] = True
            return None
        fast = native.format_linear_fast(
            kept.weight, kept.bias, operands, proof_state, kept.weight_key
        )
        if fast is None:
            _log_native_fallback()
            kept._slots[no_slot] = True
            return None
        return kept._slots.setdefault(slot, fast)

    def _publish_fast(self, kept, fast) -> None:
        """Expose up to two distinct native replays on the layer without a
        lock: ``_fast`` holds the first route to warm and is never
        overwritten; ``_fast2`` holds a second route when a layer's schedule
        alternates between ``dense_fast`` and ``dot_fast``. Ignores and
        retracts publishes from a ``kept`` retired by concurrent mutation."""
        if self.__dict__.get(_KEPT) is not kept:
            return
        first = self.__dict__.setdefault(_FAST, fast)
        if first is fast:
            if self.__dict__.get(_FAST2) is fast:
                self.__dict__.pop(_FAST2, None)
        else:
            self.__dict__.setdefault(_FAST2, fast)
            if self.__dict__.get(_FAST) is fast and self.__dict__.get(_FAST2) is fast:
                self.__dict__.pop(_FAST2, None)
        if self.__dict__.get(_KEPT) is not kept:
            if self.__dict__.get(_FAST) is fast:
                self.__dict__.pop(_FAST, None)
            if self.__dict__.get(_FAST2) is fast:
                self.__dict__.pop(_FAST2, None)

    def _register_fast(self, kept, plan, signature: tuple):
        """Have the plan's native replay hold ``signature``: every scheduled
        dense route (including mma/msg, which inline ``N_size`` inside their
        compiled graph) shares ``dense_fast``. The replay reads this thread's
        graphs of the plan's proof binding, which a failed proof retires."""
        if "no_dense_fast" in kept._slots:
            return
        kept_local = plan.binding.state.get("_kept")
        if kept_local is None or kept_local.graphs.get(signature) is None:
            return
        fast = self._native_fast(kept, "dense_fast", plan.operands, plan.binding.state)
        if fast is None:
            return
        fast.__self__.register(signature)
        self._publish_fast(kept, fast)

    def _register_dot_fast(self, kept, plan, signature: tuple):
        """Have the Dot plan's native replay hold ``signature``: the plan's
        proof keeps this thread's graph, which a failed proof retires."""
        if "no_dot_fast" in kept._slots:
            return
        kept_local = plan.proof._kept
        if kept_local is None or kept_local.graphs.get(signature) is None:
            return
        fast = self._native_fast(kept, "dot_fast", plan.operands, vars(plan.proof))
        if fast is None:
            return
        fast.__self__.register(signature)
        self._publish_fast(kept, fast)

    def _bind_one_row_mlx_fast(
        self, kept: _Kept, dense: tile_dense.DensePlan, sig: tuple
    ) -> None:
        """Bind a single-row M3 C7 ``mlx_rows`` signature onto ``dense_fast``."""
        if dense.proof.failed:
            return
        kept_local = dense.binding.kept
        if kept_local is None or sig not in kept_local.graphs:
            tile_dense._keep_local(
                dense, sig, tile_dense._make_mlx_route(self.bits, self.group_size)
            )
        self._register_fast(kept, dense, sig)

    def __call__(self, x: mx.array) -> mx.array:
        fast = self._fast
        if fast is not None:
            y = fast(self, x)
            if y is not None:
                return y
            fast2 = self._fast2
            if fast2 is not None:
                y = fast2(self, x)
                if y is not None:
                    return y
        shape, dtype = x.shape, x.dtype
        kept = self.__dict__.get(_KEPT)
        if kept is None or kept.weight is not self[kept.weight_key]:
            kept = self._keep()
        plan, bias = kept.plan, kept.bias
        if plan is not None:
            # Fast single-row decode path on the already-fetched shape tuple:
            # avoids C++ x.size/x.ndim calls and integer division on every
            # 1-token step while still keeping the M3 C7 single-row mlx_rows.
            if (
                len(shape) == 2
                and shape[0] == 1
                or len(shape) == 3
                and shape[0] == 1
                and shape[1] == 1
            ):
                if kept.mlx_rows and len(shape) == 3 and shape[-1] == plan.k:
                    y = self._stock_matmul(x)
                    if dtype in (mx.float16, mx.bfloat16):
                        dense = kept._slots.get("dense")
                        if dense is not None:
                            self._bind_one_row_mlx_fast(kept, dense, (shape, dtype))
                    return y if bias is None else y + bias
                graph = plan.proof.graphs.get((shape, dtype))
                if graph is not None:
                    if "no_dot_fast" not in kept._slots:
                        self._register_dot_fast(kept, plan, (shape, dtype))
                    y = graph(x, *plan.operands)
                    return y if bias is None else y + bias
            if not shape or shape[-1] != plan.k:
                raise ValueError(f"x is (..., {plan.k}), not of shape {shape}")
            rows = x.size // plan.k
            if rows > 1 and dtype in (mx.float16, mx.bfloat16):
                dense = kept._slots.get("dense", _UNBUILT)
                if dense is _UNBUILT:
                    dense = self._dense_tile(kept)
                if dense is not None:
                    y = self._scheduled(kept, dense, x, bias)
                    if y is not None:
                        return y
        elif shape and shape[-1] == self.input_dims:
            rows = x.size // self.input_dims if self.input_dims else 0
            if rows > 1 and dtype in (mx.float16, mx.bfloat16):
                dense = kept._slots.get("dense", _UNBUILT)
                if dense is _UNBUILT:
                    dense = self._dense_tile(kept)
                if dense is not None:
                    y = self._scheduled(kept, dense, x, bias)
                    if y is not None:
                        return y
        # A single row is this: its Dot plan's kept graph and the bias.
        graph = plan.proof.graphs.get((shape, dtype)) if plan is not None else None
        if graph is not None:
            if "no_dot_fast" not in kept._slots:
                self._register_dot_fast(kept, plan, (shape, dtype))
            y = graph(x, *plan.operands)
        else:
            k = self.input_dims
            if not shape or shape[-1] != k:
                raise ValueError(f"x is (..., {k}), not of shape {shape}")
            y = None
            rows = x.size // k if k else 0
            if plan is not None and 0 < rows <= MAX_ROWS:
                y = _launch(plan, (shape, dtype), _linear, _linear_stand_in, x)
                if y is not None:
                    self._register_dot_fast(kept, plan, (shape, dtype))
            # No row materializes a matrix, stock or plugin.
            if y is None and rows == 0:
                y = mx.zeros((*shape[:-1], self.output_dims), dtype=dtype)
            if y is None and rows > 0 and self._stock and rows <= MAX_ROWS:
                # A short call whose matrix has no plan (qmv_quad's K 64 or
                # 128, too few scales for a face) keeps MLX's own matmul, as
                # it ran before the switch; a longer one with no scheduled
                # tile route falls to the decode below.
                y = self._stock_matmul(x)
            if y is None:
                y = x @ self._decoded().astype(dtype).T
        return y if bias is None else y + bias


class FormatEmbedding(_FormatLayer):
    """A token embedding whose matrix is a format's, looked up by row.

    The rows rule is the promise a lookup needs: a format whose layout
    declares N first (``formats.stackable``) keeps a row's weights in that
    row of each tensor, so a lookup decodes only the rows its ids name. A
    format that cannot promise it is refused here, before a checkpoint
    builds it: decoding the whole table to gather from it would be a silent
    slow path on a vocabulary's size (D11: the format's own names).
    """

    def _init_tensors(
        self,
        fmt,
        bits,
        group_size,
        input_dims,
        tensors,
        bias,
        *,
        placeholders=False,
        layout=None,
    ) -> None:
        super()._init_tensors(
            fmt,
            bits,
            group_size,
            input_dims,
            tensors,
            bias,
            placeholders=placeholders,
            layout=layout,
        )
        # A lookup decodes just the rows it names, which a format may only
        # promise through the rows rule (its layout with N first). Checked
        # after the tensor validation above, so a missing or 1-D codes
        # tensor is the ValueError it is for every format layer.
        if not formats.stackable(fmt, input_dims, self.output_dims, bits, group_size):
            raise ValueError(
                f"quant format {fmt.mode!r} cannot hold a token embedding: its "
                f"rows are not independent, so a lookup could not decode the "
                f"rows it names (a layout with N first promises it, "
                f"formats.stackable)"
            )
        # A dense view over the same reader and arrays, built at the first
        # as_linear and dropped by every write to the layer (the setters drop
        # the kept view), as a FormatLinear's plan is.
        prototype = FormatLinear.__new__(FormatLinear)
        prototype._init_tensors(
            fmt, bits, group_size, input_dims, tensors, bias, layout=layout
        )
        for name in tensors:
            del prototype[name]
        if bias is not None:
            del prototype.bias
        object.__setattr__(self, "_linear_prototype", prototype)

    @property
    def num_embeddings(self) -> int:
        return self.output_dims

    @property
    def dims(self) -> int:
        return self.input_dims

    def __call__(self, ids: mx.array) -> mx.array:
        rows = self._decoded(ids.reshape(-1))
        return rows.reshape(*ids.shape, self.dims)

    def as_linear(self, x: mx.array) -> mx.array:
        self.refuse_placeholders()
        view = self.__dict__.get(_KEPT)
        codes = self._tensors[0]
        if view is None or view.get(codes) is not self[codes]:
            view = copy(self._linear_prototype)
            for name in self._tensors:
                view[name] = self[name]
            if "bias" in self:
                view["bias"] = self["bias"]
            self.__dict__[_KEPT] = view
        return view(x)


class FormatSwitchLinear(_FormatLayer):
    """A stack of matrices kept in format ``mode`` (a switch layer's
    experts), called as mlx-vlm's ``QuantizedSwitchLinear``: ``x`` is
    ``(..., M, K)``, ``indices`` name a matrix for each of ``x``'s leading
    axes, broadcast together, and the result is ``(..., M, N)``."""

    _STACK = 1

    @property
    def num_experts(self) -> int:
        return self[self._tensors[0]].shape[0]

    def _through_decode(self, x, indices, sorted_indices):
        """``x`` times the matrices at ``indices``, decoded: what the rows
        kernel does not take, the indices' matrices or the stack, at once
        within a budget, else a piece at a time."""
        dtype = x.dtype
        batch = mx.broadcast_shapes(x.shape[:-2], indices.shape)
        if x.size == 0 or indices.size == 0:
            # No product: no matrix is decoded.
            return mx.zeros((*batch, x.shape[-2], self.output_dims), dtype)
        matrix = self.output_dims * self.input_dims
        piece = max(1, _DECODE_BYTES // max(1, matrix * (4 + dtype.size)))
        # One matrix for each pair of x and the indices broadcast together.
        pairs = math.prod(batch)
        if indices.size < self.num_experts:
            # A decode step's indices: their matrices alone are decoded, at
            # once (broadcast over x) as a piece holds them, else a pair at
            # a time while the pairs are fewer than the stack's matrices.
            if indices.size <= piece:
                return x @ self._decoded(indices).astype(dtype).swapaxes(-1, -2)
            if pairs < self.num_experts:
                return self._selected(x, indices, piece)
        indices = _wrapped(indices, self.num_experts)
        # A prefill's: the stack once, not a matrix for each index; whole
        # from enough pairs, if it is within the budget.
        if self.num_experts > piece and (
            # A pair's rows each a product.
            pairs * x.shape[-2] < _AT_ONCE_PAIRS * self.num_experts
            or self.num_experts * matrix * _decoded_size(self, dtype)
            > _DECODE_AT_ONCE_BYTES
        ):
            return self._stacked(x, indices, sorted_indices, piece)
        return mx.gather_mm(
            x,
            self._decoded().astype(dtype).swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )

    def _selected(self, x, indices, piece):
        """``x`` times the matrices at ``indices``, ``piece`` pairs at a
        time, each piece one matmul. With one row of x for each index (a
        model's call) and a decode that works element by element, each pair
        has the bits of the matmul of all of them at once; with more rows,
        or x broadcast over the indices, MLX may choose another kernel for a
        smaller batch."""
        dtype, (m, k) = x.dtype, x.shape[-2:]
        batch = mx.broadcast_shapes(x.shape[:-2], indices.shape)
        rows = mx.broadcast_to(x, (*batch, m, k)).reshape(-1, m, k)
        each = mx.broadcast_to(indices, batch).flatten()
        parts = [
            rows[p : p + piece]
            @ self._decoded(each[p : p + piece]).astype(dtype).swapaxes(-1, -2)
            for p in range(0, each.size, piece)
        ]
        return mx.concatenate(parts).reshape(*batch, m, self.output_dims)

    def _stacked(self, x, indices, sorted_indices, piece):
        """MLX's gather_mm of ``x`` with the stack, ``piece`` matrices at a
        time: each piece's gather_mm takes every pair (pieces x the
        gather's work), and each pair keeps the product of the piece that
        holds its matrix. With a decode that works element by element, that
        product has the bits of the gather over the whole stack."""
        # Clipped to a piece's bounds, which an index of a small integer
        # dtype may not hold.
        indices = indices.astype(mx.uint32)
        y = None
        for first in range(0, self.num_experts, piece):
            last = min(first + piece, self.num_experts)
            weights = self._decoded(slice(first, last)).astype(x.dtype)
            part = mx.gather_mm(
                x,
                weights.swapaxes(-1, -2),
                rhs_indices=mx.clip(indices, first, last - 1) - first,
                sorted_indices=sorted_indices,
            )
            if y is None:
                y = part
            else:
                inside = mx.expand_dims(indices >= first, (-2, -1))
                y = mx.where(inside, part, y)
        return y

    def _sorted_tile(self, kept: _Kept) -> _Tile | None:
        """The sorted tile, built once; None where the layer's format has no
        face this GPU's tile kernel takes. It dies with the record."""
        if kept.sorted is _UNBUILT:
            kept.sorted = _tile(self)
        return kept.sorted

    def _sorted_swiglu(self, x, indices, limit, token_rows):
        """The sorted NAX activation of a fused stack, optionally reading token rows."""
        if not _nax_host():
            return None
        from ..patches import m5_gather_qmm_nax as nax

        kept = self.__dict__.get(_KEPT)
        if kept is None or kept.weight is not self[kept.weight_key]:
            kept = self._keep()
        plan = kept.plan
        if plan is None or kept.bias is not None:
            return None
        experts, rows = plan.operands[0].shape[0], indices.size
        if (
            plan.n % 64
            or rows < 16
            or rows < 4 * experts
            or not x.ndim
            or x.shape[-1] != plan.k
            or not nax.geometry(x, indices, experts, plan.n)
        ):
            return None
        if limit is not None and not math.isfinite(limit):
            return None
        tile = self._sorted_tile(kept)
        if tile is None or not tile.face.exact_for(x.dtype):
            return None
        if token_rows is not None:
            tokens, row_map = token_rows
            if (
                isinstance(tokens, mx.array)
                and isinstance(row_map, mx.array)
                and tokens.ndim == 3
                and tokens.shape[0] > 0
                and tokens.shape[1:] == (1, plan.k)
                and tokens.dtype == x.dtype
                and row_map.ndim == 1
                and row_map.shape[0] == rows
                and row_map.dtype == mx.uint32
                and int(tokens.shape[0]) * plan.k < 2**32
            ):
                y = _swiglu(plan, tile, tokens, indices, limit, row_map)
                if y is not None:
                    return y
        return _swiglu(plan, tile, x, indices, limit)

    def __call__(
        self, x: mx.array, indices: mx.array, sorted_indices: bool = False
    ) -> mx.array:
        # sorted_indices is mlx-vlm's hint to MLX's gathers. Sorted rows of
        # a prefill run the tile route where it takes them, whatever ran
        # before (its graphs are its proof's alone: an unsorted call never
        # replays one); the rows kernel runs every pair as one mat-vec, in
        # any order; the decode passes the hint on to MLX's gather_mm.
        shape, dtype = x.shape, x.dtype
        signature = (shape, dtype, indices.shape, indices.dtype)
        kept = self.__dict__.get(_KEPT)
        if kept is None or kept.weight is not self[kept.weight_key]:
            kept = self._keep()
        plan, bias = kept.plan, kept.bias
        y = None
        if sorted_indices and plan is not None and _sorts(plan, x, indices):
            tile = self._sorted_tile(kept)
            # The graph captures the schedule, which depends on both stack
            # dimensions even for an identical activation and index shape.
            tile_signature = (*signature, plan.n, self.num_experts)
            graph = tile.proof.graphs.get(tile_signature) if tile is not None else None
            if graph is not None:
                y = graph(x, indices, *tile.operands)
            elif tile is not None:
                y = _launch_sorted(plan, tile, tile_signature, x, indices)
        if y is None:
            graph = plan.proof.graphs.get(signature) if plan is not None else None
            if graph is not None:
                y = graph(x, indices, *plan.operands)
            else:
                k = self.input_dims
                if len(shape) < 2 or shape[-1] != k:
                    raise ValueError(f"x is (..., M, {k}), not of shape {shape}")
                if not mx.issubdtype(indices.dtype, mx.integer):
                    raise ValueError(f"indices are integers, not {indices.dtype}")
                y = _launch(plan, signature, _switch, _switch_stand_in, x, indices)
                if y is None:
                    y = self._through_decode(x, indices, sorted_indices)
        if bias is not None:
            y = y + mx.expand_dims(bias[indices], -2)
        return y
