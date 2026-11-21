# SPDX-License-Identifier: Apache-2.0
"""Multi-row quantized projection with one-row decode arithmetic per row.

Stock ``quantized_matmul`` switches kernels with the row count (``qmv_fast``
or ``qmv`` at one row, ``qmv_wide`` and then tiled qmm for more), so a
speculative verify row and the serial decode step for the same token get
different bits from the same projection. Here every row runs MLX 0.32.4's
one-row ``qmv_fast`` (K of the kernel block: 512 for
4/5-bit, 256 for 6/8-bit weights) or ``qmv`` traversal, transcribed in
``omlx.formats.affine``, so row ``r`` of the output equals
``quantized_matmul(x[r:r+1])`` bit for bit.

A threadgroup computes one ``2 * RPS``-column output tile for ``ROWS`` rows:
each weight block is decoded once and applied to all of them, and every
(row, column) accumulator still sums its K blocks in qmv order and finishes
with the same ``simd_sum`` (qmv's lane layout per column does not depend on
how many columns a simdgroup owns or how many rows share the threadgroup).

The tile runs over each matrix's dot face (``formats.dot``), through the
shared-rows skeleton ``kernels.qmv_shared``, so any format layer with a dot
face takes it too; one launch may run several projections of one input.
``one_row_qmv`` runs it for the one-row decode call itself, with a narrower
column tile than stock ``qmv_fast`` so more threadgroups stream the
weights; it replaces ``quantized_matmul`` bit for bit.
"""

from __future__ import annotations

import logging
from functools import cache

import mlx.core as mx
import mlx.nn as nn

from .. import formats
from ..formats import Dot, affine, kernels
from ..formats.affine import _GROUP_SIZES, qmv_fast_layout, values_per_thread
from ..formats.linear import FormatLinear
from .module_cache import cached_per_module

logger = logging.getLogger(__name__)

MAX_ROWS = 16
# The affine widths the row-exact verify tile decodes. 2- and 3-bit stock
# layers keep MLX's path here, as they did in upstream's verify gather
# (``_BITS`` 4/5/6/8 too): they are out of the PoC's served widths, not
# refused by ``affine.decoded`` (which takes them).
_BITS = (4, 5, 6, 8)
# Large 6- and 8-bit heads can amortize padding over a shared weight pass.
_PADDED_ROWS_MIN_WEIGHT_BYTES = 128 * 1024 * 1024

# One projection's tile: a threadgroup's two simdgroups own RPS output
# columns each of tile `tile` (2 * RPS columns), for ROWS rows from row0.
# Every K block of a column is decoded once and applied to the ROWS rows,
# their inputs loaded ahead; each (row, column) accumulator is the one-row
# call's. Simdgroups past N (a partial last tile) have nothing to compute.
_TILE = r"""
template <
    typename T, int K, int N, int ROWS, int RPS, int VPT,
    typename D, typename X, typename Y>
METAL_FUNC void omlx_row_exact(D m, X x, Y y, int tile, int row0, uint sg, uint sl) {
  constexpr bool PARTIAL = N % (2 * RPS) != 0;
  const int out_row = tile * (2 * RPS) + int(sg) * RPS;
  if (PARTIAL && out_row >= N) {
    return;
  }
  const int valid = PARTIAL ? min(RPS, N - out_row) : RPS;
  float result[ROWS * RPS];
  qmv_shared<T, K, ROWS, RPS, VPT, PARTIAL>(
      m, size_t(out_row), valid, x + row0 * K, sl, result);
  if (sl == 0) {
    for (int i = 0; i < ROWS; i++) {
      for (int r = 0; r < valid; r++) {
        y[(row0 + i) * N + out_row + r] = static_cast<T>(result[i * RPS + r]);
      }
    }
  }
}
"""


# Projection i's tile `tile` (of its own tiles).
_CALL = (
    "omlx_row_exact<T, K, N_{i}, ROWS, RPS, VPT>("
    "m{i}, x, y{i}, {tile}, row0, sg, sl);"
)


def _launch_source(count: int) -> str:
    """The body of a launch of ``count`` projections of one input: the tile
    axis walks each projection's tiles in turn, and each writes its own
    output."""
    lines = [
        "    const int tile = int(threadgroup_position_in_grid.z);",
        "    const int row0 = int(threadgroup_position_in_grid.y) * ROWS;",
        "    const uint sg = simdgroup_index_in_threadgroup;",
        "    const uint sl = thread_index_in_simdgroup;",
    ]
    if count == 1:
        return "\n".join([*lines, "    " + _CALL.format(i=0, tile="tile")]) + "\n"
    lines.append("    int start = 0;")
    for i in range(count):
        lines += [
            f"    constexpr int TILES_{i} = (N_{i} + 2 * RPS - 1) / (2 * RPS);",
            f"    if (tile < start + TILES_{i}) {{",
            "      " + _CALL.format(i=i, tile="tile - start"),
            "      return;",
            "    }",
            f"    start += TILES_{i};",
        ]
    return "\n".join(lines) + "\n"


def _source(faces: tuple, unrolled: bool) -> tuple[list[str], str, str]:
    """The inputs, header and source of a launch over ``faces`` (one a
    projection), the skeleton's loops unrolled in full if ``unrolled``."""
    matrices = kernels.Matrices(unrolled=unrolled)
    inputs = ["x"]
    for i, face in enumerate(faces):
        inputs += matrices.add(f"m{i}", face, shared=True)
    source = matrices.declarations + _launch_source(len(faces))
    return inputs, matrices.header + _TILE, source


@cache
def _kernel(faces: tuple, unrolled: bool, constants: tuple):
    inputs, header, source = _source(faces, unrolled)
    # MLX writes the template signature on each launch. Bind the integer
    # dimensions once for all layers and verify lengths with this tile.
    prefix = "".join(f"constexpr int {name} = {value};\n" for name, value in constants)
    return mx.fast.metal_kernel(
        name="omlx_row_exact_"
        + "_".join(face.key for face in faces)
        + ("_u_" if unrolled else "_")
        + "_".join(str(value) for _, value in constants),
        input_names=inputs,
        output_names=[f"y{i}" for i in range(len(faces))],
        header=header,
        source=prefix + source,
    )


class _Launch:
    """Static parameters of one launch of projections of one input (their
    faces share a vpt): ``rows`` rows, ``per_group`` of them a threadgroup
    (rows padded with zeros to whole groups), ``rps`` output columns a
    simdgroup, the skeleton unrolled if ``unrolled``."""

    __slots__ = (
        "kernel",
        "template",
        "grid",
        "output_shapes",
        "output_dtypes",
        "padding",
        "shape",
        "faces",
    )

    def __init__(self, faces, dtype, k, sizes, rows, per_group, rps, unrolled):
        self.faces = tuple(faces)
        self.padding = -rows % per_group
        padded = rows + self.padding
        constants = (
            ("K", k),
            ("ROWS", per_group),
            ("RPS", rps),
            ("VPT", self.faces[0].vpt),
            *((f"N_{i}", n) for i, n in enumerate(sizes)),
        )
        self.kernel = _kernel(self.faces, unrolled, constants)
        self.template = [("T", dtype)]
        tiles = sum((n + 2 * rps - 1) // (2 * rps) for n in sizes)
        self.grid = (32, 2 * (padded // per_group), tiles)
        self.output_shapes = [(padded, n) for n in sizes]
        self.output_dtypes = [dtype] * len(sizes)
        # What an unrolled launch's check is kept by (None: rolled).
        self.shape = (self.faces, *self.template, *constants) if unrolled else None

    def __call__(self, x: mx.array, operands) -> list:
        """The outputs for ``x`` of shape [rows, K] and each projection's
        operands (its weight, then its face's tensors)."""
        rows = x.shape[0]
        if self.padding:
            x = mx.pad(x, [(0, self.padding), (0, 0)])
        outputs = self.kernel(
            inputs=[x, *(a for matrix in operands for a in matrix)],
            template=self.template,
            grid=self.grid,
            threadgroup=(32, 2, 1),
            output_shapes=self.output_shapes,
            output_dtypes=self.output_dtypes,
        )
        return [y[:rows] for y in outputs] if self.padding else outputs


# The envelope of kernels.unrolled_tile_ok was measured on one GPU
# generation; another compiler or GPU can still get a tile inside it wrong (an
# M3 Ultra did at 4-bit gs64: whole rows off, not rounding). So the first
# launch of each unrolled shape is checked on the running GPU against stock
# one-row ``quantized_matmul`` of the same weights on a fixed input; a shape
# with any differing bit runs the rolled tile (same arithmetic, same
# geometry) from then on. Verdicts are per compiled shape: the bug is in code
# generation, not in the weight values.
_UNROLLED_VERDICTS: dict = {}


def unrolled_tile_verified(
    shape: tuple, run, reference, rows: int, k: int, dtype
) -> bool:
    """Whether the unrolled launch ``shape`` (a hashable key of every template
    parameter) gives the one-row calls' bits. ``run(x)`` launches it on ``x``
    [rows, k] and ``reference(x)`` makes each of its outputs one row at a
    time; checked once per shape."""
    verdict = _UNROLLED_VERDICTS.get(shape)
    if verdict is None:
        x = (mx.random.normal((rows, k), key=mx.random.key(rows)) * 0.5).astype(dtype)
        got, want = run(x), reference(x)
        mx.eval(got, want)
        verdict = _UNROLLED_VERDICTS[shape] = all(
            mx.array_equal(a.reshape(b.shape), b).item() for a, b in zip(got, want)
        )
        if not verdict:
            logger.warning(
                "row-exact unrolled tile %s differs from the one-row call on this "
                "GPU; it runs the rolled tile instead",
                shape,
            )
    return verdict


def _stock_rows(weights: list, bits: int, group_size: int):
    """The reference of a launch over affine tensors: stock one-row
    ``quantized_matmul`` of each ``(weight, scales, biases)``, row by row."""

    def reference(x: mx.array) -> list:
        return [
            mx.concatenate(
                [
                    mx.quantized_matmul(
                        x[r : r + 1],
                        w,
                        scales=s,
                        biases=b,
                        transpose=True,
                        group_size=group_size,
                        bits=bits,
                    )
                    for r in range(x.shape[0])
                ]
            )
            for w, s, b in weights
        ]

    return reference


def _face(linear, x: mx.array) -> Dot | None:
    """The dot face (``formats.dot``) ``linear``'s rows run over, or None.
    ``linear`` is an MLX affine layer of this module's widths (a subclass
    too: the prefill reclass makes these projections one) or a format layer,
    and its face is exact for ``x``'s dtype and reads K in whole lanes."""
    if isinstance(linear, nn.QuantizedLinear):
        if getattr(linear, "mode", "affine") != "affine" or linear.bits not in _BITS:
            return None
    elif not isinstance(linear, FormatLinear) or x.shape[-1] != linear.input_dims:
        return None
    k = int(x.shape[-1])
    face = formats.dot(linear, k, int(linear[formats.tensor_names(linear)[0]].shape[0]))
    if face is None or not face.exact_for(x.dtype) or face.vpt < 1 or k % face.vpt:
        return None
    return face


def _operands(linear, face: Dot) -> tuple:
    return (
        linear[formats.tensor_names(linear)[0]],
        *(linear[name] for name in face.tensors),
    )


def _padded(bits: int, fast: bool, n: int, k: int) -> bool:
    """Whether to pad a partial tile on an affine head's fast traversal."""
    return (
        bits in (6, 8) and fast and n * k * bits // 8 >= _PADDED_ROWS_MIN_WEIGHT_BYTES
    )


def _unrolled_ok(bits: int, rps: int, rows_per_group: int) -> bool:
    """kernels.unrolled_tile_ok for the qmv_fast tile of ``bits``-bit weights."""
    return kernels.unrolled_tile_ok(values_per_thread(bits, True), rps, rows_per_group)


def _module_launch(linears, faces, x: mx.array, rows: int, geometry=None):
    """The launch of ``linears`` (layers over ``faces``) for ``rows`` rows of
    ``x``, on ``geometry`` (rows per threadgroup, columns per simdgroup,
    unrolled) or the default one; None where ``geometry`` does not fit. A
    threadgroup of several rows runs the unrolled skeleton inside
    ``kernels.unrolled_tile_ok`` (by default, or as ``geometry`` asks), once
    this GPU's bits are checked against the layers' one-row calls."""
    k = int(x.shape[-1])
    sizes = [
        int(linear[formats.tensor_names(linear)[0]].shape[0]) for linear in linears
    ]
    vpt = faces[0].vpt
    if geometry is None:
        first = linears[0]
        bits = int(first.bits)
        # The padding rule is affine's: a format layer of the built-in mode
        # keeps it.
        pad = getattr(first, "mode", "affine") == "affine" and _padded(
            bits, qmv_fast_layout(k, max(sizes), bits), max(sizes), k
        )
        per_group, rps = kernels.shared_geometry(vpt, max(sizes), rows, pad)
        if bits == 6 and rows % per_group:
            # Three inputs share a 6-bit decode. Four pads a five-row window
            # to eight and regresses the 27B head on the M5 Max.
            per_group = 3
        unrolled = per_group > 1
    else:
        per_group, rps, unrolled = geometry
        if rows % per_group:
            return None
    unrolled = unrolled and kernels.unrolled_tile_ok(vpt, rps, per_group)
    launch = _Launch(faces, x.dtype, k, sizes, rows, per_group, rps, unrolled)
    if launch.shape is not None:

        def run(probe):
            outputs = launch(probe, [_operands(m, f) for m, f in zip(linears, faces)])
            return [_biased(m, y) for m, y in zip(linears, outputs)]

        def reference(probe):
            return [_one_row_calls(linear, probe) for linear in linears]

        if not unrolled_tile_verified(launch.shape, run, reference, rows, k, x.dtype):
            launch = _Launch(faces, x.dtype, k, sizes, rows, per_group, rps, False)
    return launch


def _format_plans(linear):
    # A checkpoint reader chooses its face from the layer's actual parameters.
    # Other projections in a group keep the token, without retaining the arrays.
    return cached_per_module(
        linear,
        "_omlx_format_row_exact_plans",
        lambda _: ({}, object()),
        depth=0,
        flags=(
            id(linear.fmt),
            linear.fmt,
            linear.input_dims,
            type(linear.bits),
            linear.bits,
            type(linear.group_size),
            linear.group_size,
            tuple(dict.keys(linear)),
        ),
    )


def _plans(linear) -> dict:
    if isinstance(linear, FormatLinear):
        return _format_plans(linear)[0]
    # Plans live on the (first) module, so they die with it; weights are
    # read per call.
    plans = linear.__dict__.get("_omlx_row_exact_plans")
    if plans is None:
        plans = {}
        object.__setattr__(linear, "_omlx_row_exact_plans", plans)
    return plans


def _plan(linear, x: mx.array, rows: int) -> _Launch | None:
    plans = _plans(linear)
    key = (rows, x.dtype, x.shape[-1])
    if key not in plans:
        face = _face(linear, x) if rows <= MAX_ROWS else None
        plans[key] = (
            None if face is None else _module_launch((linear,), (face,), x, rows)
        )
    return plans[key]


def _one_row_calls(linear, x: mx.array) -> mx.array:
    """``linear`` on each row of ``x`` ([rows, K]) alone: the decode call."""
    return mx.concatenate([linear(x[r : r + 1][None])[0] for r in range(x.shape[0])])


def _biased(linear, y: mx.array) -> mx.array:
    return y + linear["bias"] if "bias" in linear else y


def quantized_linear(linear, x: mx.array) -> mx.array:
    """``linear(x)`` for ``x`` of shape [..., K], each row with one-row arithmetic.

    ``linear`` is an MLX quantized layer or a format layer. Layers the kernel
    does not take (no dot face for ``x``, other modes or bit widths, more
    than MAX_ROWS rows) run one one-row call per row, which is the serial
    decode call itself.
    """
    lead = x.shape[:-1]
    rows = 1
    for size in lead:
        rows *= size
    if rows <= 1:
        return linear(x)
    flat = x.reshape(rows, x.shape[-1])
    launch = _plan(linear, x, rows)
    if launch is None:
        return _one_row_calls(linear, flat).reshape(*lead, -1)
    (y,) = launch(flat, [_operands(linear, launch.faces[0])])
    return _biased(linear, y).reshape(*lead, -1)


def _group_plan(linears, x: mx.array, rows: int, geometry=None) -> _Launch | None:
    plans = _plans(linears[0])
    key = (
        tuple(id(linear) for linear in linears),
        rows,
        x.dtype,
        x.shape[-1],
        geometry,
    )
    entry = plans.get(key)
    if entry is not None:
        launch, others = entry
        if not others or all(
            _format_plans(linears[index])[1] is token for index, token in others
        ):
            return launch
    faces = [_face(linear, x) for linear in linears] if rows <= MAX_ROWS else []
    grouped = (
        faces
        and all(face is not None for face in faces)
        and len({face.vpt for face in faces}) == 1
        and not any("bias" in linear for linear in linears)
        and kernels.table_bytes(*faces) <= formats.TABLE_BYTES
    )
    launch = _module_launch(linears, faces, x, rows, geometry) if grouped else None
    others = tuple(
        (index, _format_plans(linear)[1])
        for index, linear in enumerate(linears[1:], 1)
        if isinstance(linear, FormatLinear)
    )
    plans[key] = launch, others
    return launch


def quantized_linears(linears, x: mx.array) -> tuple:
    """``tuple(linear(x) for linear in linears)`` with one-row arithmetic per
    row, in a single launch when the projections have faces of one vpt and
    no bias, and their distinct tables fit in one launch."""
    lead = x.shape[:-1]
    rows = 1
    for size in lead:
        rows *= size
    plan = _group_plan(linears, x, rows) if 1 < rows and 1 < len(linears) <= 4 else None
    if plan is None:
        return tuple(quantized_linear(linear, x) for linear in linears)
    operands = [_operands(m, face) for m, face in zip(linears, plan.faces)]
    outputs = plan(x.reshape(rows, x.shape[-1]), operands)
    return tuple(y.reshape(*lead, -1) for y in outputs)


def quantized_linears_tiled(
    linears, x: mx.array, rows_per_group: int, rps: int, unrolled: bool = False
):
    """``quantized_linears`` in one launch for 1..MAX_ROWS rows on an explicit
    tile: each threadgroup applies ``2 * rps`` output columns to
    ``rows_per_group`` rows, on the unrolled tile if ``unrolled`` (inside
    ``kernels.unrolled_tile_ok``). Every tile gives each row its one-row bits
    (at one row, the layer's own call's); None where the launch does not
    take these projections."""
    lead = x.shape[:-1]
    rows = 1
    for size in lead:
        rows *= size
    plan = _group_plan(tuple(linears), x, rows, (rows_per_group, rps, unrolled))
    if plan is None:
        return None
    operands = [_operands(m, face) for m, face in zip(linears, plan.faces)]
    outputs = plan(x.reshape(rows, x.shape[-1]), operands)
    return tuple(y.reshape(*lead, -1) for y in outputs)


def format_rows_qmv(linear, dtype, geometry):
    """Plan a format matrix on the GDN's (columns, group) tile for each width."""
    k = linear.input_dims
    probe = mx.zeros((1, k), dtype)
    face = _face(linear, probe)
    if face is None or kernels.table_bytes(face) > formats.TABLE_BYTES:
        return None
    operands = _operands(linear, face)
    operand_dtypes = tuple(a.dtype for a in operands)
    n = linear.output_dims

    @cache
    def tile(rows):
        rps, group = geometry(rows)
        launch = _module_launch((linear,), (face,), probe, rows, (group, rps, rows > 1))
        # The kernel already binds the face, geometry and verified unroll choice.
        proof = kernels.proof("format_gdn_rows", launch.kernel, dtype, operand_dtypes)
        return launch, proof

    def project(x):
        rows = x.size // k
        launch, proof = tile(rows)
        signature = (x.shape, x.dtype)
        graph = proof.graphs.get(signature)
        if graph is not None:
            return graph(x, *operands)

        # Equal tiles share the graph on this thread, each with its own weights.
        @mx.compile
        def graph(value, *tensors):
            (y,) = launch(value.reshape(rows, k), (tensors,))
            return y.reshape(*value.shape[:-1], n)

        y = graph(x, *operands)
        proof.keep(signature, graph)
        return y

    return project


class OneRowQmv:
    """``mx.quantized_matmul`` of one row on a ``qmv_fast`` shape, bit for bit.

    Stock one-row ``quantized_matmul`` runs ``qmv_fast`` with 8 output columns
    per threadgroup (4 per simdgroup). A column's arithmetic (its lanes' K-block
    accumulation and the closing ``simd_sum``) does not depend on how many
    columns its simdgroup owns, so the same tile kernel with ``2 * rps``
    columns per threadgroup gives the same bits with more threadgroups in
    flight. At the Qwen4 GDN decode projections (1x2560 -> 16480 6-bit and
    1x6144 -> 2560 5-bit) that runs nearer the DRAM floor than the stock
    kernel, whose launch geometry is fixed.
    """

    __slots__ = ("_kernel", "_weights", "_template", "_grid", "_n", "_dtype")

    def __init__(self, weight, scales, biases, bits, group_size, dtype, rps):
        n = int(weight.shape[0])
        k = int(scales.shape[-1]) * group_size
        face = _affine_face(bits, group_size, dtype)
        launch = _Launch((face,), dtype, k, [n], 1, 1, rps, False)
        self._kernel, self._template, self._grid = (
            launch.kernel,
            launch.template,
            launch.grid,
        )
        self._weights = (weight, scales, biases)
        self._n = n
        self._dtype = dtype

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is one row ``[..., K]`` of the planned dtype."""
        return self._kernel(
            inputs=[x, *self._weights],
            template=self._template,
            grid=self._grid,
            threadgroup=(32, 2, 1),
            output_shapes=[(*x.shape[:-1], self._n)],
            output_dtypes=[self._dtype],
        )[0]


def _affine_face(bits: int, group_size: int, dtype) -> Dot:
    """The ``qmv_fast`` face of affine tensors taken by position, of scales
    and activations of ``dtype`` (float32 too: GLM's)."""
    if bits not in _BITS:
        raise ValueError(f"row-exact tile: no decode of {bits}-bit weights")
    return affine.face(bits, group_size, True, dtype)  # refuses a group size MLX lacks


def _qmv_fast_layout(weight, scales, biases, bits, group_size, mode, dtype, rps) -> bool:
    """Stock one-row ``quantized_matmul`` runs ``qmv_fast`` on this layout
    (``qmv_fast_layout``) and ``2 * rps`` divides N."""
    if (
        mode != "affine"
        or bits not in _BITS
        or group_size not in _GROUP_SIZES
        or biases is None
        or dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or weight.dtype != mx.uint32
        or weight.ndim != 2
        or scales.dtype != dtype
        or biases.dtype != dtype
        or scales.shape != biases.shape
    ):
        return False
    n = int(weight.shape[0])
    k = int(scales.shape[-1]) * group_size
    return not (
        not qmv_fast_layout(k, n, bits)
        or n % (2 * rps)
        or scales.shape != (n, k // group_size)
        or weight.shape[1] * 32 != k * bits
    )


def one_row_qmv(weight, scales, biases, bits, group_size, mode, dtype, rps):
    """A ``OneRowQmv`` on a ``_qmv_fast_layout``, else None."""
    if not _qmv_fast_layout(weight, scales, biases, bits, group_size, mode, dtype, rps):
        return None
    return OneRowQmv(weight, scales, biases, bits, group_size, dtype, rps)


class RowsQmv:
    """``OneRowQmv`` for a block of rows: every row of ``x`` gets one-row
    ``qmv_fast`` bits. ``geometry(rows)`` gives the output columns per
    simdgroup and the rows each threadgroup applies its decoded weight tile
    to (a divisor of ``rows``); launch parameters are kept per row count.
    ``unrolled`` runs the fully unrolled tile (same bits) on tiles inside
    ``kernels.unrolled_tile_ok`` that this GPU computes bit-exactly (checked once per
    tile shape), else the rolled tile."""

    __slots__ = (
        "_face",
        "_weights",
        "_k",
        "_n",
        "_dtype",
        "_geometry",
        "_launch",
        "_bits",
        "_group_size",
        "_unrolled",
    )

    def __init__(
        self, weight, scales, biases, bits, group_size, dtype, geometry, unrolled=False
    ):
        self._face = _affine_face(bits, group_size, dtype)
        self._weights = (weight, scales, biases)
        self._k = int(scales.shape[-1]) * group_size
        self._n = int(weight.shape[0])
        self._dtype = dtype
        self._geometry = geometry
        self._launch = {}
        self._bits = bits
        self._group_size = group_size
        self._unrolled = unrolled

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``[..., K]`` of the planned dtype, row-contiguous."""
        rows = x.size // self._k
        launch = self._launch.get(rows)
        if launch is None:
            rps, per_group = self._geometry(rows)
            if (
                rows % per_group
                or self._n % (2 * rps)
                or (self._unrolled and not _unrolled_ok(self._bits, rps, per_group))
            ):
                raise ValueError(
                    f"no {rps}x{per_group} tile for {rows} rows of {self._n}"
                )
            launch = _Launch(
                (self._face,),
                self._dtype,
                self._k,
                [self._n],
                rows,
                per_group,
                rps,
                self._unrolled,
            )
            if launch.shape is not None and not unrolled_tile_verified(
                launch.shape,
                lambda probe: launch(probe, [self._weights]),
                _stock_rows([self._weights], self._bits, self._group_size),
                rows,
                self._k,
                self._dtype,
            ):
                launch = _Launch(
                    (self._face,),
                    self._dtype,
                    self._k,
                    [self._n],
                    rows,
                    per_group,
                    rps,
                    False,
                )
            self._launch[rows] = launch
        (y,) = launch(x.reshape(rows, self._k), [self._weights])
        return y.reshape(*x.shape[:-1], self._n)


def rows_qmv(
    weight, scales, biases, bits, group_size, mode, dtype, geometry, unrolled=False
):
    """A ``RowsQmv`` on a ``_qmv_fast_layout`` (checked at one column per
    simdgroup; ``__call__`` checks the tile ``geometry`` picks), else None."""
    if not _qmv_fast_layout(weight, scales, biases, bits, group_size, mode, dtype, 1):
        return None
    return RowsQmv(weight, scales, biases, bits, group_size, dtype, geometry, unrolled)


__all__ = [
    "MAX_ROWS",
    "OneRowQmv",
    "RowsQmv",
    "one_row_qmv",
    "quantized_linear",
    "quantized_linears",
    "quantized_linears_tiled",
    "rows_qmv",
]
