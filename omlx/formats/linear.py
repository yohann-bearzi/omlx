# SPDX-License-Identifier: Apache-2.0
"""Linear layers whose weights are kept in a registered format: what
``nn.QuantizedLinear`` and mlx-vlm's ``QuantizedSwitchLinear`` are to the
formats MLX knows.

A layer holds a matrix's tensors under the names its format gives them, and
asks the registry how to run them:

- ordinary Qwen dense projections select a single-row dot or wide mat-vec
  according to MLX's row count, layout and size rules. The wide kernel
  decodes weights through the format's tile face.
  Explicitly row-exact calls use the dot face for each input row;
- a switch layer's sorted prefill uses its tile face on tensor units or
  through MLX's steel matrix kernels, from 16 rows and four rows per expert
  on average. Below that threshold it uses its decode rows kernel;
- dense layers can fall back to the dot face for short calls. When a layer
  has no eligible launch it decodes to the activations' type, then uses
  MLX's matmul (its ``gather_mm`` from as many indices as a switch layer's
  stack has matrices).
"""

from __future__ import annotations

import logging
import math
from functools import cache
from typing import Any, NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .. import formats
from ..patches.module_cache import cached_per_module
from . import Dot, Tile, kernels, tile_wide
from .mlx import traversal

logger = logging.getLogger(__name__)

# Maximum number of serial-exact rows in one dense Dot launch. A longer
# explicitly row-exact Qwen call uses several such groups.
MAX_ROWS = 16

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


def _occupancy(vpt: int) -> int:
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
    """How a stack's sorted rows run through its format's tile face."""

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


def _tile(layer, plan: _Plan) -> _Tile | None:
    """Bind the tile backend for this GPU with its own proof and operands."""
    face = formats.tile(layer, plan.k, plan.n)
    if face is None:
        return None
    if _nax_host():
        from ..patches import m5_gather_qmm_nax as nax

        kernel = nax._face_kernel(face)
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
    operands = tuple(layer[name] for name in ("weight", *face.tensors))
    dtypes = tuple(a.dtype for a in operands)
    proof = kernels.proof(route, face, dtypes)
    return _Tile(face, kernel, operands, proof, said)


def _ordinary_tile(layer, k, n, route, build_kernel):
    face = formats.tile(layer, k, n)
    if face is None:
        return None
    kernel = build_kernel(face)
    if kernel is None:
        return None
    operands = tuple(layer[name] for name in ("weight", *face.tensors))
    proof = kernels.proof(
        f"format_linear_{route}", face, tuple(a.dtype for a in operands), k, n
    )
    said = (
        f"quant format face {face.key}: {route} kernel engaged",
        f"quant format face {face.key}: {route} kernel failed; rows kernel fallback",
    )
    return _Tile(face, kernel, operands, proof, said)


def _decode_pair(layer, plan):
    tile = formats.tile(layer, plan.k, plan.n)
    if tile is None or kernels.table_bytes(plan.face, tile) > formats.TABLE_BYTES:
        return None
    names, kernel = tile_wide.pair_kernel(plan.face, tile)
    # These weight buffers are fixed until a layer write retires the plan.
    operands = tuple(mx.contiguous(layer[name]) for name in names)
    proof = kernels.proof(
        "format_linear_decode_pair",
        plan.face,
        tile,
        tuple(a.dtype for a in operands),
        plan.k,
        plan.n,
    )
    said = (
        f"quant format face {tile.key}: two-request decode kernel engaged",
        f"quant format face {tile.key}: two-request decode kernel failed",
    )
    return _Tile(tile, kernel, operands, proof, said)


@cache
def _metal_architecture():
    if mx.is_available(mx.gpu):
        return mx.device_info(mx.gpu).get("architecture")
    return None


def _loading_qwen():
    checkpoint = formats._LOADING.get()
    if checkpoint is None or not checkpoint.active:
        return False
    try:
        config = checkpoint._read_config()
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict):
        return False
    from ..oq import _family

    for section in (config, config.get("text_config")):
        if isinstance(section, dict):
            name = section.get("model_type")
            if isinstance(name, str) and _family(name) == "qwen4_exp":
                return True
    return False


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
    operands = tuple(layer[name] for name in ("weight", *face.tensors))
    stacked = layer["weight"].ndim == 3
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
        # Its first tile of outputs.
        return _rows(plan.kernel, plan.k, plan.n, x, tensors, 1, (plan.n,), None, 1, 1)

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


def _sorted_stand_in(tile: _Tile, metal, graph):
    """The smallest launch of the tile Metal ``metal``, over zeros: 16 rows
    of x, all for matrix 0, and the fewest matrices that leave each tensor
    8 elements. The template values are the call's (its plan), not the
    stand-in's."""
    dtype, k = metal[:2]
    matrices = max(-(-8 * a.shape[0] // a.size) for a in tile.operands)
    operands = (((matrices, *a.shape[1:]), a.dtype) for a in tile.operands)
    specs = (((16, 1, k), dtype), ((16,), mx.uint32), *operands)

    return kernels.zeros(*specs), graph


def _launch_sorted(plan: _Plan, tile: _Tile, signature: tuple, x, indices):
    """``x`` and its sorted ``indices`` through the tile route, as
    ``_launch`` runs the rows kernel; None where the face is not exact for
    x's type or its proof has failed, for the rows kernel to take it."""
    if not tile.face.exact_for(x.dtype):
        return None
    metal, graph = _sorted(plan, tile.kernel, x, indices)
    if not tile.proof.prove(
        metal, lambda: _sorted_stand_in(tile, metal, graph), logger, *tile.said
    ):
        return None
    graph = mx.compile(graph)
    y = graph(x, indices, *tile.operands)
    tile.proof.keep(signature, graph)
    return y


# Where a layer keeps what its calls work out of it once: its plan and its
# bias.
_KEPT = "_omlx_format_call"
_FAST = "_fast"
_NATIVE_FALLBACK_LOGGED = False


def _log_native_fallback() -> None:
    global _NATIVE_FALLBACK_LOGGED
    if not _NATIVE_FALLBACK_LOGGED:
        _NATIVE_FALLBACK_LOGGED = True
        logger.warning(
            "quant_matmul native FormatLinearFast unavailable; "
            "using Python warm dispatch fallback"
        )



def _unloaded(name: str, held, offered) -> str:
    """A placeholder ``held`` of tensor ``name``, as a refusal names it: with
    the shape and dtype last offered in its place, or as never loaded."""
    if offered is None:
        return f"{name} (never loaded)"
    shape, dtype = offered
    if shape is None:
        return f"{name} (offered {dtype}, not an array)"
    return (
        f"{name} (offered {shape} {dtype}, not the layout's {held.shape} "
        f"{held.dtype})"
    )


class _FormatLayer(nn.Module):
    """What the two layers share: a format's tensors as frozen parameters,
    the attributes of MLX's quantized layers, and the format object that
    reads them (``fmt``).

    A layer works out its plan and finds its bias at its first call, and
    keeps them in its ``__dict__`` with the weight they were found with,
    until one of its arrays or attributes is assigned or removed
    (``__setitem__``, ``__delitem__``, ``pop``: what MLX's loads,
    ``update``, ``set_dtype`` and attribute writes go through; a dict's
    own mutators and ``del`` of an attribute too) or its
    weight is another array: a function that ``mx.compile`` traces over the
    layer's state (``inputs=``) swaps every array of it for a trace's, and
    back, behind those methods. A call then reads one dict entry and the
    weight. Threads: a layer that is being loaded is not called meanwhile;
    two threads that make a layer's first call at once each work out the
    same plan, and either keeps it.

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
    ) -> None:
        if not isinstance(fmt, (str, formats.QuantFormat)):
            raise TypeError("a format layer needs a mode name or a QuantFormat")
        formats._may_build(fmt if isinstance(fmt, str) else fmt.mode)
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
        names = ("weight", *fmt.tensors)
        if set(tensors) != set(names):
            raise ValueError(
                f"quant format {mode!r} keeps a matrix as {', '.join(names)}, "
                f"not as {', '.join(sorted(tensors))}"
            )
        weight = tensors["weight"]
        if weight.ndim != self._STACK + 2:
            raise ValueError(
                f"{type(self).__name__}: the weight has {self._STACK + 2} axes, "
                f"not the shape {weight.shape}"
            )
        # What the format lays out for the matrix the layer holds: a layer's
        # N is its weight's first axis.
        n = weight.shape[-2]
        if layout is None:
            layout = fmt.layout(input_dims, n, bits, group_size)
        if layout is not None:
            formats.check_layout(fmt, n, input_dims, layout)
        # What lets a stack's matrices be gathered, tensor by tensor.
        stack = weight.shape[: self._STACK]
        ragged = [name for name in names if tensors[name].shape[: self._STACK] != stack]
        if ragged:
            raise ValueError(
                f"{type(self).__name__}: {', '.join(ragged)} does not open with "
                f"the weight's stack of {stack}"
            )
        # The object, not a lookup of its mode: a checkpoint's own (what its
        # format's accept returned for its record) reads its layers. Kept out
        # of the module's items whatever its class.
        object.__setattr__(self, "fmt", fmt)
        self.mode, self.bits, self.group_size = mode, bits, group_size
        self.input_dims, self.output_dims = input_dims, n
        # The format's tensors, and those that still hold a load's
        # placeholders, each with the shape and dtype last offered in its
        # place (None: none was). Kept beside the parameters, not among them.
        self.__dict__.update(
            _tensors=names, _placeholders=dict.fromkeys(names if placeholders else ())
        )
        if not self._STACK and _loading_qwen():
            from ..patches.qwen35_verify_qmm import is_row_exact_armed

            object.__setattr__(self, "_qwen_row_exact", is_row_exact_armed)
        for name in names:
            self[name] = tensors[name]
        if bias is not None:
            self["bias"] = bias
        self.freeze()

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
        held = self.get(key)
        if held is not None and key in self._tensors:
            if value is held:
                return
            checkpoint = formats._LOADING.get()
            loading = checkpoint is not None and checkpoint.active
            array = isinstance(value, mx.array)
            if not array or value.shape != held.shape or value.dtype != held.dtype:
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
        held = ", ".join(
            _unloaded(name, self[name], offered) for name, offered in pending
        )
        raise ValueError(
            f"{self!r} computes on its checkpoint's tensors alone; it still "
            f"holds the load's placeholder of {held}"
        )

    def _keep(self) -> tuple:
        """The layer's plan and bias, worked out and kept."""
        self.__dict__.pop(_FAST, None)
        bias = self.get("bias")
        if "bias" in self and not isinstance(bias, mx.array):
            # MLX's quantized layers raise as they add it.
            raise TypeError(f"the bias is an array, not {type(bias).__name__}")
        kept = self.__dict__[_KEPT] = (_plan(self), bias, self["weight"])
        return kept

    def __delitem__(self, key: str) -> None:
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        super().__delitem__(key)

    def pop(self, *key_and_default):
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        return super().pop(*key_and_default)

    # A dict's own mutators, which do not go through the three above.
    def setdefault(self, key: str, default: Any = None) -> Any:
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        return super().setdefault(key, default)

    def popitem(self) -> tuple:
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        return super().popitem()

    def clear(self) -> None:
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        super().clear()

    def __ior__(self, other):
        # A dictionary union accepts mappings and iterables of key/value pairs.
        for key, value in dict(other).items():
            self[key] = value
        return self

    def __delattr__(self, name: str) -> None:
        self.__dict__.pop(_KEPT, None)
        self.__dict__.pop(_FAST, None)
        super().__delattr__(name)

    def _decoded(self, indices: mx.array | None = None) -> mx.array:
        """The layer's weights as floats: all of them, or the matrices at
        ``indices`` of the stack."""
        self.refuse_placeholders()
        fmt = self.fmt
        tensors = {name: self[name] for name in ("weight", *fmt.tensors)}
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
    of the format's layout, which its face reads."""
    return isinstance(module, _FormatLayer) and key in module._tensors


def _exact_rows(plan, x):
    """Keep long verify calls in groups of at most MAX_ROWS serial-exact rows."""
    if plan is None or plan.proof.failed or not plan.face.exact_for(x.dtype):
        return None
    proof = plan.proof
    signature = ("row_exact", x.shape, x.dtype)
    graph = proof.graphs.get(signature)
    if graph is not None:
        return graph(x, *plan.operands)

    k, n = plan.k, plan.n
    groups, tail = divmod(x.size // k, MAX_ROWS)
    flat = x.reshape(-1, k)

    def rows_graph(count):
        key = ((count, k), x.dtype)
        if key not in proof.graphs:
            _launch(plan, key, _linear, _linear_stand_in, flat[:count])
        return proof.graphs.get(key)

    full = rows_graph(MAX_ROWS)
    last = rows_graph(tail) if tail else None
    if full is None or (tail and last is None):
        return None

    def grouped(x, *operands):
        rows = x.reshape(-1, k)
        outputs = [
            full(rows[i * MAX_ROWS : (i + 1) * MAX_ROWS], *operands)
            for i in range(groups)
        ]
        if tail:
            outputs.append(last(rows[-tail:], *operands))
        return mx.concatenate(outputs).reshape(*x.shape[:-1], n)

    graph = mx.compile(grouped)
    result = graph(x, *plan.operands)
    proof.keep(signature, graph)
    return result


class FormatLinear(_FormatLayer):
    """``x @ W.T + bias`` for a matrix ``W`` kept in format ``mode``:
    ``tensors`` are its ``weight`` and the format's tensors, by name."""

    _qwen_row_exact = None
    _fast = None

    def _ordinary(self, kept, x, signature):
        plan = kept[0]
        architecture = _metal_architecture()
        if not architecture:
            return None
        rows = x.size // plan.k
        geometry = (plan.n, plan.k, self.group_size, architecture)
        folded = traversal(rows, *geometry, bits=self.bits)
        if folded[0] in ("fallback", "quad"):
            return None
        if len(kept) == 3:
            kept = self.__dict__[_KEPT] = (*kept, {})
        if (
            signature[0][:-1] == (2, 1)
            and folded[0] == "wide"
            and plan.n % 8 == 0
            and plan.face.exact_for(x.dtype)
        ):
            routes = kept[3]
            if "decode_pair" not in routes:
                routes["decode_pair"] = _decode_pair(self, plan)
            pair = routes["decode_pair"]
            if pair is not None:
                graph = pair.proof.graphs.get(signature)
                if graph is not None:
                    self._register_fast(kept, pair, signature)
                    return graph(x, *pair.operands)
                y = _launch(
                    pair, signature, tile_wide.describe, tile_wide.pair_stand_in, x
                )
                if y is not None:
                    self._register_fast(kept, pair, signature)
                    return y
        single = x.ndim < 3 or rows == x.shape[-2]
        route_sig = signature if single else ("batched", *signature)
        result = self._project_route(kept, x, route_sig, folded)
        if result is None or single:
            return result
        # MLX folds the leading axes if the input is row-contiguous or it
        # must copy each matrix. A gap between intact matrices keeps M per
        # matrix instead. The same trace can receive either physical layout.
        batched = traversal(x.shape[-2], *geometry, bits=self.bits, batched=True)
        if batched == folded:
            return result
        separate = self._project_route(kept, x, route_sig, batched)
        if separate is None:
            return None
        from .tile_layout import per_matrix

        return mx.where(per_matrix(x), separate, result)

    def _project_route(self, kept, x, signature, choice):
        plan, routes = kept[0], kept[3]
        route = choice[0]
        if route == "rows":
            key = ("ordinary_rows", *signature)
            graph = plan.proof.graphs.get(key)
            if graph is not None:
                return graph(x, *plan.operands)
            return _launch(plan, key, _linear, _linear_stand_in, x)
        if route != "wide":
            return None
        if route not in routes:
            tile = _ordinary_tile(self, plan.k, plan.n, route, tile_wide.kernel)
            routes[route] = tile
            if tile is not None:
                self.__dict__[_KEPT] = (*kept[:4], tile)
        tile = routes[route]
        if tile is None:
            return None
        graph = tile.proof.graphs.get(signature)
        if graph is not None:
            if len(signature) == 2:
                self._register_fast(kept, tile, signature)
            return graph(x, *tile.operands)
        y = _launch(tile, signature, tile_wide.describe, tile_wide.stand_in, x)
        if y is not None and len(signature) == 2:
            self._register_fast(kept, tile, signature)
        return y

    def _register_fast(self, kept, tile: _Tile, signature: tuple) -> None:
        graph = tile.proof.graphs.get(signature)
        if graph is None:
            return
        # Each plan binds its own operands and live proof. A layer can switch
        # between a prefill's wide graph and the two-request decode graph.
        binding = ("native", id(tile))
        fast = kept[3].get(binding)
        if fast is None:
            try:
                from ..custom_kernels.quant_matmul import fast as native
            except ImportError:
                _log_native_fallback()
                return

            if not native.has_symbol("FormatLinearFast"):
                _log_native_fallback()
                return
            fast = native.format_linear_fast(
                kept[2], kept[1], tile.operands, vars(tile.proof)
            )
            if fast is None:
                _log_native_fallback()
                return
            kept[3][binding] = fast
        fast.__self__.register(signature)
        self.__dict__[_FAST] = fast

    def __call__(self, x: mx.array) -> mx.array:
        fast = self._fast
        if fast is not None:
            y = fast(self, x)
            if y is not None:
                return y
        shape, dtype = x.shape, x.dtype
        kept = self.__dict__.get(_KEPT)
        if kept is None or kept[2] is not self["weight"]:
            kept = self._keep()
        plan, bias = kept[0], kept[1]
        row_exact = self._qwen_row_exact
        if row_exact is not None and plan is not None:
            if len(kept) > 4 and not row_exact():
                wide = kept[4]
                graph = wide.proof.graphs.get((shape, dtype))
                if graph is not None:
                    if fast is not None:
                        self._register_fast(kept, wide, (shape, dtype))
                    y = graph(x, *wide.operands)
                    return y if bias is None else y + bias
            if not shape or shape[-1] != plan.k:
                raise ValueError(f"x is (..., {plan.k}), not of shape {shape}")
            if x.size > plan.k:
                if row_exact():
                    if x.size > MAX_ROWS * plan.k:
                        y = _exact_rows(plan, x)
                        if y is not None:
                            return y if bias is None else y + bias
                else:
                    y = self._ordinary(kept, x, (shape, dtype))
                    if y is not None:
                        return y if bias is None else y + bias
        # A decode step is this, its kept graph's replay and the bias.
        graph = plan.proof.graphs.get((shape, dtype)) if plan is not None else None
        if graph is not None:
            y = graph(x, *plan.operands)
        else:
            k = self.input_dims
            if not shape or shape[-1] != k:
                raise ValueError(f"x is (..., {k}), not of shape {shape}")
            y = None
            if plan is not None and 0 < x.size // k <= MAX_ROWS:
                y = _launch(plan, (shape, dtype), _linear, _linear_stand_in, x)
            if y is None:
                y = x @ self._decoded().astype(dtype).T
        return y if bias is None else y + bias


class FormatSwitchLinear(_FormatLayer):
    """A stack of matrices kept in format ``mode`` (a switch layer's
    experts), called as mlx-vlm's ``QuantizedSwitchLinear``: ``x`` is
    ``(..., M, K)``, ``indices`` name a matrix for each of ``x``'s leading
    axes, broadcast together, and the result is ``(..., M, N)``."""

    _STACK = 1

    @property
    def num_experts(self) -> int:
        return self["weight"].shape[0]

    def _stacked_outputs(self, other: FormatSwitchLinear) -> FormatSwitchLinear:
        """Join two admitted, row-stackable projections after checkpoint load.

        The fusion has checked their layouts. Only the layers themselves
        supply the reader, dimensions and arrays for the derived projection.
        """
        self.refuse_placeholders()
        other.refuse_placeholders()
        tensors = {
            name: mx.concatenate([self[name], other[name]], axis=1)
            for name in self._tensors
        }
        bias = (
            mx.concatenate([self["bias"], other["bias"]], axis=-1)
            if "bias" in self
            else None
        )
        layer = FormatSwitchLinear.__new__(FormatSwitchLinear)
        layer._init_tensors(
            self.fmt, self.bits, self.group_size, self.input_dims, tensors, bias
        )
        layer.train(self.training)
        layer._no_grad = set(self._no_grad)
        return layer

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
        if kept is None or kept[2] is not self["weight"]:
            kept = self._keep()
        plan, bias = kept[0], kept[1]
        y = None
        if sorted_indices and plan is not None and _sorts(plan, x, indices):
            if len(kept) == 3:
                # The fourth entry is lazy and dies with the kept weight and plan.
                kept = self.__dict__[_KEPT] = (*kept, _tile(self, plan))
            tile = kept[3]
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
