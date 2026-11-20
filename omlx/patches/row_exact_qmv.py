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

``one_row_qmv`` runs the same tile for the one-row decode call itself, with
a narrower column tile than stock ``qmv_fast`` so more threadgroups stream
the weights; it replaces ``quantized_matmul`` bit for bit.
"""

from __future__ import annotations

import logging
from functools import cache
from textwrap import indent

import mlx.core as mx
import mlx.nn as nn

from ..formats import kernels
from ..formats.affine import (
    _GROUP_SIZES,
    decoded,
    header,
    qmv_fast_layout,
    values_per_thread,
)

logger = logging.getLogger(__name__)

MAX_ROWS = 16
# The affine widths the row-exact verify tile decodes. 2- and 3-bit stock
# layers keep MLX's path here, as they did in upstream's verify gather
# (``_BITS`` 4/5/6/8 too): they are out of the PoC's served widths, not
# refused by ``affine.decoded`` (which takes them).
_BITS = (4, 5, 6, 8)
# Padding a partial row tile pays for large 8-bit weight streams. Smaller
# matrices and 4/5/6-bit unpackers are faster with the existing divisor tile.
_PADDED_ROWS_MIN_WEIGHT_BYTES = 128 * 1024 * 1024

# Per row this is MLX's qmv_fast_impl (FAST) or qmv_impl; the row loop sits
# inside each K block so the block's weight bytes are fetched once. Columns
# past N (qmv's partial last tile) are skipped: each column's arithmetic does
# not depend on which tile computes it. ``ROW_EXACT_UNROLL`` precedes every
# row and column loop: empty, or a full unroll in the ``unrolled`` kernels
# (unrolling repeats each accumulator's steps in the same order).
_TILE = r"""
// Tiny scale/bias arrays (N < 8) arrive in the constant address space, so
// their pointer type is a template parameter.
template <typename T, int K_SIZE, int N_OUT, int ROWS, int RPS, typename SP>
METAL_FUNC void row_exact_tile(
    const device T* x,
    const device uint32_t* w,
    SP scales,
    SP biases,
    device T* y,
    int tile,
    int row0,
    uint simd_gid,
    uint simd_lid) {
  constexpr bool PARTIAL = (N_OUT % (2 * RPS)) != 0;
  const int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
  const int in_vec_size_g = K_SIZE / GS;
  const int out_row = tile * (2 * RPS) + int(simd_gid) * RPS;
  if (PARTIAL && out_row >= N_OUT) {
    return;
  }
  const int valid = PARTIAL ? min(RPS, N_OUT - out_row) : RPS;

  const device uint8_t* ws = (const device uint8_t*)w + out_row * in_vec_size_w +
      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  SP sc = scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
  SP bs = biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
  const device T* xp = x + row0 * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

  float result[ROWS][RPS];
  ROW_EXACT_UNROLL
  for (int i = 0; i < ROWS; i++) {
    ROW_EXACT_UNROLL
    for (int r = 0; r < RPS; r++) {
      result[i][r] = 0;
    }
  }

  int k = 0;
  const int full_limit = FAST ? K_SIZE : K_SIZE - BLOCK_SIZE;
  for (; k < full_limit; k += BLOCK_SIZE) {
__BLOCK__    ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    sc += BLOCK_SIZE / GS;
    bs += BLOCK_SIZE / GS;
    xp += BLOCK_SIZE;
  }
__LAST_BLOCK__
  ROW_EXACT_UNROLL
  for (int i = 0; i < ROWS; i++) {
    ROW_EXACT_UNROLL
    for (int r = 0; r < RPS; r++) {
      if (PARTIAL && r >= valid) {
        break;
      }
      float v = simd_sum(result[i][r]);
      if (simd_lid == 0) {
        y[(row0 + i) * N_OUT + out_row + r] = static_cast<T>(v);
      }
    }
  }
}
"""

# One K block of a lane: its inputs of every row, then each column's weights
# decoded once against them.
_BLOCK = r"""
    float xs[ROWS][VALUES_PER_THREAD];
    float sums[ROWS];
    ROW_EXACT_UNROLL
    for (int i = 0; i < ROWS; i++) {
      sums[i] = load_vector(xp + i * K_SIZE, xs[i]);
    }
    ROW_EXACT_UNROLL
    for (int r = 0; r < RPS; r++) {
      if (PARTIAL && r >= valid) {
        break;
      }
      float wq[W_TERMS];
      decode_w(ws + r * in_vec_size_w, wq);
      float s = sc[r * in_vec_size_g];
      float b = bs[r * in_vec_size_g];
      ROW_EXACT_UNROLL
      for (int i = 0; i < ROWS; i++) {
        result[i][r] += s * wdot(wq, xs[i]) + sums[i] * b;
      }
    }
"""

# qmv_impl's last block, in a qmv tile alone (qmv_fast's K is whole blocks):
# a lane has all of its inputs there or none (K is whole groups), and for
# all of them load_vector_safe and qdot_safe are the full block's code.
_LAST_BLOCK = (
    "  if (int(simd_lid) * VALUES_PER_THREAD < K_SIZE - k) {\n"
    + indent(_BLOCK[1:], "  ")
    + "  }\n"
)

_SOURCE = r"""
    row_exact_tile<T, K_SIZE, N_SIZE, ROWS, RPS>(
        x, w, scales, biases, y,
        int(threadgroup_position_in_grid.z),
        int(threadgroup_position_in_grid.y) * ROWS,
        simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""


def _group_source(count: int) -> str:
    """Same-input projections in one launch: the tile axis walks each
    projection's tiles in turn, and every projection writes its own output."""
    lines = [
        "    const int tile = int(threadgroup_position_in_grid.z);",
        "    const int row0 = int(threadgroup_position_in_grid.y) * ROWS;",
        "    const uint sg = simdgroup_index_in_threadgroup;",
        "    const uint sl = thread_index_in_simdgroup;",
        "    int start = 0;",
    ]
    for i in range(count):
        lines += [
            f"    constexpr int TILES_{i} = (N_{i} + 2 * RPS - 1) / (2 * RPS);",
            f"    if (tile < start + TILES_{i}) {{",
            f"      row_exact_tile<T, K_SIZE, N_{i}, ROWS, RPS>(",
            f"          x, w{i}, scales{i}, biases{i}, y{i}, tile - start, row0, sg, sl);",
            "      return;",
            "    }",
            f"    start += TILES_{i};",
        ]
    return "\n".join(lines) + "\n"


# The envelope of kernels.unrolled_tile_ok was measured on one GPU
# generation; another compiler or GPU can still get a tile inside it wrong (an
# M3 Ultra did at 4-bit gs64: whole rows off, not rounding). So the first
# launch of each unrolled shape is checked on the running GPU against stock
# one-row ``quantized_matmul`` of the same weights on a fixed input; a shape
# with any differing bit runs the rolled tile (same arithmetic, same
# geometry) from then on. Verdicts are per compiled shape: the bug is in code
# generation, not in the weight values.
_UNROLLED_VERDICTS: dict = {}


def unrolled_tile_verified(shape: tuple, run, weights: list, bits: int, group_size: int, rows: int, k: int, dtype) -> bool:
    """Whether the unrolled launch ``shape`` (a hashable key of every template
    parameter) matches stock one-row ``quantized_matmul`` bit for bit.
    ``run(x)`` launches it on ``x`` [rows, k] and returns one output per
    ``(weight, scales, biases)`` of ``weights``; checked once per shape."""
    verdict = _UNROLLED_VERDICTS.get(shape)
    if verdict is None:
        x = (mx.random.normal((rows, k), key=mx.random.key(rows)) * 0.5).astype(dtype)
        got = run(x)
        want = [
            mx.concatenate(
                [
                    mx.quantized_matmul(
                        x[r : r + 1], w, scales=s, biases=b,
                        transpose=True, group_size=group_size, bits=bits,
                    )
                    for r in range(rows)
                ]
            )
            for w, s, b in weights
        ]
        mx.eval(got, want)
        verdict = _UNROLLED_VERDICTS[shape] = all(
            mx.array_equal(a.reshape(b.shape), b).item() for a, b in zip(got, want)
        )
        if not verdict:
            logger.warning(
                "row-exact unrolled tile %s differs from one-row quantized_matmul "
                "on this GPU; it runs the rolled tile instead",
                shape,
            )
    return verdict


def _header(bits: int, group_size: int, fast: bool, unrolled: bool = False) -> str:
    """The tile's source for one bit width: MLX's code for that width, and
    qmv_impl's last block in a qmv tile alone."""
    if bits not in _BITS:
        raise ValueError(f"row-exact tile: no decode of {bits}-bit weights")
    start = header(bits, group_size, fast)  # refuses a group size MLX lacks
    unroll = '_Pragma("clang loop unroll(full)")' if unrolled else ""
    return (
        f"#define ROW_EXACT_UNROLL {unroll}\n"
        + start
        + decoded(bits)
        + _TILE.replace("__BLOCK__", _BLOCK[1:]).replace(
            "__LAST_BLOCK__", "" if fast else _LAST_BLOCK
        )
    )


@cache
def _kernel(bits: int, group_size: int, fast: bool, unrolled: bool = False):
    return mx.fast.metal_kernel(
        name=f"omlx_row_exact_qmv_b{bits}_gs{group_size}_{int(fast)}" + ("_u" if unrolled else ""),
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        header=_header(bits, group_size, fast, unrolled),
        source=_SOURCE,
    )


@cache
def _group_kernel(bits: int, group_size: int, fast: bool, count: int, unrolled: bool = False):
    inputs = ["x"]
    for i in range(count):
        inputs += [f"w{i}", f"scales{i}", f"biases{i}"]
    return mx.fast.metal_kernel(
        name=f"omlx_row_exact_qmv_group{count}_b{bits}_gs{group_size}_{int(fast)}"
        + ("_u" if unrolled else ""),
        input_names=inputs,
        output_names=[f"y{i}" for i in range(count)],
        header=_header(bits, group_size, fast, unrolled),
        source=_group_source(count),
    )


def _kernel_supported(linear: nn.QuantizedLinear, x: mx.array) -> bool:
    bits, group_size = linear.bits, linear.group_size
    if (
        getattr(linear, "mode", "affine") != "affine"
        or bits not in _BITS
        or group_size not in _GROUP_SIZES
        or linear.biases is None
        or x.dtype not in (mx.bfloat16, mx.float16)
        or linear.scales.dtype != x.dtype
        or linear.biases.dtype != x.dtype
    ):
        return False
    n = int(linear.weight.shape[0])
    k = int(linear.scales.shape[-1]) * group_size
    fast = qmv_fast_layout(k, n, bits)
    return (
        x.shape[-1] == k
        and k % 8 == 0
        and group_size % values_per_thread(bits, fast) == 0
    )


def _launch_geometry(
    bits: int, fast: bool, n: int, rows: int, k: int
) -> tuple[int, int]:
    """(rows per threadgroup, output columns per simdgroup) for one launch."""
    pad = bits == 8 and fast and n * k >= _PADDED_ROWS_MIN_WEIGHT_BYTES
    return kernels.shared_geometry(values_per_thread(bits, fast), n, rows, pad)


def _unrolled_ok(bits: int, rps: int, rows_per_group: int) -> bool:
    """kernels.unrolled_tile_ok for the qmv_fast tile of ``bits``-bit weights."""
    return kernels.unrolled_tile_ok(values_per_thread(bits, True), rps, rows_per_group)


class _Plan:
    """Static launch parameters of one (linear, row count, dtype) shape."""

    __slots__ = (
        "kernel",
        "template",
        "grid",
        "output_shapes",
        "output_dtypes",
        "bias",
        "padding",
    )

    def __init__(self, linear: nn.QuantizedLinear, x: mx.array, rows: int):
        k = int(x.shape[-1])
        n = int(linear.weight.shape[0])
        bits = int(linear.bits)
        fast = qmv_fast_layout(k, n, bits)
        rows_per_group, rps = _launch_geometry(bits, fast, n, rows, k)
        self.padding = -rows % rows_per_group
        padded_rows = rows + self.padding
        self.kernel = _kernel(bits, int(linear.group_size), fast)
        self.template = [
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", rows_per_group),
            ("RPS", rps),
        ]
        self.grid = (
            32,
            2 * (padded_rows // rows_per_group),
            (n + 2 * rps - 1) // (2 * rps),
        )
        self.output_shapes = [(padded_rows, n)]
        self.output_dtypes = [x.dtype]
        self.bias = "bias" in linear


def _plan(linear: nn.QuantizedLinear, x: mx.array, rows: int) -> _Plan | None:
    # Plans live on the module, so they die with it; weights are read per call.
    plans = linear.__dict__.get("_omlx_row_exact_plans")
    if plans is None:
        plans = {}
        object.__setattr__(linear, "_omlx_row_exact_plans", plans)
    key = (rows, x.dtype, x.shape[-1])
    if key not in plans:
        supported = rows <= MAX_ROWS and _kernel_supported(linear, x)
        plans[key] = _Plan(linear, x, rows) if supported else None
    return plans[key]


def quantized_linear(linear: nn.QuantizedLinear, x: mx.array) -> mx.array:
    """``linear(x)`` for ``x`` of shape [..., K], each row with one-row arithmetic.

    Layouts the kernel does not cover (non-affine modes, other bit widths or
    group sizes, more than MAX_ROWS rows) run one stock one-row call per row,
    which is the serial decode call itself.
    """
    lead = x.shape[:-1]
    rows = 1
    for size in lead:
        rows *= size
    if rows <= 1:
        return linear(x)
    flat = x.reshape(rows, x.shape[-1])
    plan = _plan(linear, x, rows)
    if plan is None:
        y = mx.concatenate(
            [linear(flat[r : r + 1][None])[0] for r in range(rows)], axis=0
        )
        return y.reshape(*lead, -1)
    if plan.padding:
        flat = mx.pad(flat, [(0, plan.padding), (0, 0)])
    y = plan.kernel(
        inputs=[flat, linear.weight, linear.scales, linear.biases],
        template=plan.template,
        grid=plan.grid,
        threadgroup=(32, 2, 1),
        output_shapes=plan.output_shapes,
        output_dtypes=plan.output_dtypes,
    )[0]
    if plan.padding:
        y = y[:rows]
    if plan.bias:
        y = y + linear["bias"]
    return y.reshape(*lead, -1)


class _GroupPlan:
    """Static launch parameters of one same-input projection group."""

    __slots__ = (
        "kernel",
        "template",
        "grid",
        "output_shapes",
        "output_dtypes",
        "padding",
    )

    def __init__(self, linears, x: mx.array, rows: int, geometry=None):
        k = int(x.shape[-1])
        sizes = [int(linear.weight.shape[0]) for linear in linears]
        first = linears[0]
        bits = int(first.bits)
        fast = qmv_fast_layout(k, sizes[0], bits)
        rows_per_group, rps, unrolled = geometry or (
            *_launch_geometry(bits, fast, max(sizes), rows, k),
            False,
        )
        self.padding = -rows % rows_per_group
        padded_rows = rows + self.padding
        self.kernel = _group_kernel(
            bits, int(first.group_size), fast, len(linears), unrolled
        )
        self.template = [("T", x.dtype), ("K_SIZE", k)]
        self.template += [(f"N_{i}", n) for i, n in enumerate(sizes)]
        self.template += [("ROWS", rows_per_group), ("RPS", rps)]
        tiles = sum((n + 2 * rps - 1) // (2 * rps) for n in sizes)
        self.grid = (32, 2 * (padded_rows // rows_per_group), tiles)
        self.output_shapes = [(padded_rows, n) for n in sizes]
        self.output_dtypes = [x.dtype] * len(sizes)

    def launch(self, linears, x: mx.array) -> list:
        """The group's outputs for ``x`` of shape [rows, K]."""
        rows = x.shape[0]
        if self.padding:
            x = mx.pad(x, [(0, self.padding), (0, 0)])
        inputs = [x]
        for linear in linears:
            inputs += [linear.weight, linear.scales, linear.biases]
        outputs = self.kernel(
            inputs=inputs,
            template=self.template,
            grid=self.grid,
            threadgroup=(32, 2, 1),
            output_shapes=self.output_shapes,
            output_dtypes=self.output_dtypes,
        )
        return [y[:rows] for y in outputs] if self.padding else outputs


def _group_plan(linears, x: mx.array, rows: int, geometry=None) -> _GroupPlan | None:
    first = linears[0]
    plans = first.__dict__.get("_omlx_row_exact_plans")
    if plans is None:
        plans = {}
        object.__setattr__(first, "_omlx_row_exact_plans", plans)
    key = (tuple(id(linear) for linear in linears), rows, x.dtype, x.shape[-1], geometry)
    if key not in plans:
        k = int(x.shape[-1])
        fast = [
            qmv_fast_layout(k, int(linear.weight.shape[0]), int(linear.bits))
            for linear in linears
        ]
        grouped = (
            rows <= MAX_ROWS
            and all(fast) == any(fast)
            and all(
                isinstance(linear, nn.QuantizedLinear)
                and linear.bits == first.bits
                and linear.group_size == first.group_size
                and "bias" not in linear
                and _kernel_supported(linear, x)
                for linear in linears
            )
            and (
                geometry is None
                or rows % geometry[0] == 0
                and (
                    not geometry[2]
                    or _unrolled_ok(first.bits, geometry[1], geometry[0])
                )
            )
        )
        plan = _GroupPlan(linears, x, rows, geometry) if grouped else None
        if plan is not None and geometry is not None and geometry[2]:
            k = int(x.shape[-1])
            shape = (
                "group", int(first.bits), int(first.group_size), fast[0], k,
                tuple(int(linear.weight.shape[0]) for linear in linears),
                rows, geometry[0], geometry[1], x.dtype,
            )
            weights = [(linear.weight, linear.scales, linear.biases) for linear in linears]
            if not unrolled_tile_verified(
                shape, lambda probe: plan.launch(linears, probe), weights,
                int(first.bits), int(first.group_size), rows, k, x.dtype,
            ):
                plan = _GroupPlan(linears, x, rows, (geometry[0], geometry[1], False))
        plans[key] = plan
    return plans[key]


def quantized_linears(linears, x: mx.array) -> tuple:
    """``tuple(linear(x) for linear in linears)`` with one-row arithmetic per
    row, in a single launch when the projections share their quantization."""
    lead = x.shape[:-1]
    rows = 1
    for size in lead:
        rows *= size
    plan = _group_plan(linears, x, rows) if 1 < rows and 1 < len(linears) <= 4 else None
    if plan is None:
        return tuple(quantized_linear(linear, x) for linear in linears)
    outputs = plan.launch(linears, x.reshape(rows, x.shape[-1]))
    return tuple(y.reshape(*lead, -1) for y in outputs)


def quantized_linears_tiled(
    linears, x: mx.array, rows_per_group: int, rps: int, unrolled: bool = False
):
    """``quantized_linears`` in one launch for 1..MAX_ROWS rows on an explicit
    tile: each threadgroup applies ``2 * rps`` output columns to
    ``rows_per_group`` rows, on the unrolled tile if ``unrolled`` (inside
    ``kernels.unrolled_tile_ok``). Every tile gives each row its one-row ``qmv`` bits
    (at one row, stock ``quantized_matmul``'s); None where the group kernel
    does not take these projections."""
    lead = x.shape[:-1]
    rows = 1
    for size in lead:
        rows *= size
    plan = _group_plan(tuple(linears), x, rows, (rows_per_group, rps, unrolled))
    if plan is None:
        return None
    outputs = plan.launch(linears, x.reshape(rows, x.shape[-1]))
    return tuple(y.reshape(*lead, -1) for y in outputs)


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
        self._kernel = _kernel(bits, group_size, True)
        self._weights = (weight, scales, biases)
        self._template = [
            ("T", dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", 1),
            ("RPS", rps),
        ]
        self._grid = (32, 2, n // (2 * rps))
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

    __slots__ = ("_kernel", "_weights", "_k", "_n", "_dtype", "_geometry", "_launch", "_bits", "_group_size", "_unrolled")

    def __init__(self, weight, scales, biases, bits, group_size, dtype, geometry, unrolled=False):
        self._kernel = _kernel(bits, group_size, True, False)
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
                raise ValueError(f"no {rps}x{per_group} tile for {rows} rows of {self._n}")
            template = [
                ("T", self._dtype),
                ("K_SIZE", self._k),
                ("N_SIZE", self._n),
                ("ROWS", per_group),
                ("RPS", rps),
            ]
            grid = (32, 2 * (rows // per_group), self._n // (2 * rps))
            kernel = self._kernel
            if self._unrolled:
                unrolled = _kernel(self._bits, self._group_size, True, True)
                shape = ("rows", self._bits, self._group_size, self._k, self._n, rows, per_group, rps, self._dtype)

                def run(probe):
                    return unrolled(
                        inputs=[probe, *self._weights],
                        template=template,
                        grid=grid,
                        threadgroup=(32, 2, 1),
                        output_shapes=[(rows, self._n)],
                        output_dtypes=[self._dtype],
                    )

                if unrolled_tile_verified(
                    shape, run, [self._weights], self._bits, self._group_size, rows, self._k, self._dtype
                ):
                    kernel = unrolled
            launch = self._launch[rows] = (kernel, template, grid)
        kernel, template, grid = launch
        return kernel(
            inputs=[x, *self._weights],
            template=template,
            grid=grid,
            threadgroup=(32, 2, 1),
            output_shapes=[(*x.shape[:-1], self._n)],
            output_dtypes=[self._dtype],
        )[0]


def rows_qmv(weight, scales, biases, bits, group_size, mode, dtype, geometry, unrolled=False):
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
