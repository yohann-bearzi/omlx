# SPDX-License-Identifier: Apache-2.0
"""Checks for a format's tests. ``dot_rows`` runs a face in the skeletons
oMLX's kernels are built from, outside any model, and ``tile_rows`` reads
every weight of rows through a tile face. ``no_dot_defects`` fails a test in
which a format's ``dot()`` or ``tile()`` was defective.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import cache

import mlx.core as mx

from .. import formats
from . import Dot, Tile, kernels

# One threadgroup of one simdgroup per NA + NB rows.
_ROWS = r"""
    const uint simd_lid = thread_index_in_simdgroup;
    const size_t row = size_t(threadgroup_position_in_grid.y) * (NA + NB);
    float result[NA + NB];
    qmv_rows<T, K, NA, NB>(m, row, m, row + NA, x, simd_lid, result);
    if (simd_lid == 0) {
      store_rows(omlx_iota<NA + NB>(), result, y + row);
    }
"""


@contextmanager
def no_dot_defects():
    """Fail on leaving the block when a format's ``dot()`` or ``tile()`` was
    defective in it.

    ``omlx.formats.dot`` and ``omlx.formats.tile`` log a defect and answer
    None: a model keeps its other kernels. A test wants the failure. oMLX
    runs each test under ``tests/`` in this block, and so can a plugin, from
    an autouse fixture. An exception of the block's body is raised in place
    of this failure.

    The block's value is the set of the defects met so far, each a format's
    mode and the error's ``repr``. A test that expects one removes it."""
    defects: set = set()
    kept, formats._DOT_DEFECTS = formats._DOT_DEFECTS, defects
    try:
        yield defects
    finally:
        formats._DOT_DEFECTS = kept
    if defects:
        raise AssertionError(
            f"a quant format's dot() or tile() is defective: {sorted(defects)}"
        )


@cache
def _dot_rows_kernel(face: Dot):
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("m", face)]
    return mx.fast.metal_kernel(
        name=f"omlx_format_dot_rows_{face.key}",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header + kernels.STORE_ROWS,
        source=matrices.declarations + _ROWS,
    )


def dot_rows(
    face: Dot, operands, x: mx.array, n: int, packs: tuple[int, int] = (1, 1)
) -> mx.array:
    """The products of ``x`` (K activations) with the first ``n`` rows of a
    matrix, through the row skeleton.

    ``operands`` are the matrix's weight and the face's tensors, in order.
    A launch takes ``packs[0]`` rows as the skeleton's first pack and the
    next ``packs[1]`` as its second, one row or more in all; ``n`` is one
    or more times their sum, and no more than the matrix's rows (which is
    not checked). ``x`` is one vector of a dtype the face is exact for; it
    and each operand have 8 elements or more. Anything else is refused with
    a ``ValueError``. The kernel is compiled for ``face`` and K.
    """
    na, nb = packs
    if min(na, nb) < 0 or na + nb == 0:
        raise ValueError(f"packs={packs}: two counts of rows, one row or more in all")
    if n <= 0 or n % (na + nb):
        raise ValueError(f"n={n} is not one launch or more of {na + nb} rows")
    if x.ndim != 1:
        raise ValueError(f"x is one vector of K activations, not of shape {x.shape}")
    operands, names = tuple(operands), ("weight", *face.tensors)
    if len(operands) != len(names):
        raise ValueError(
            f"face {face.key} reads {', '.join(names)}; operands given: {len(operands)}"
        )
    if not face.exact_for(x.dtype):
        exact = ", ".join(str(dtype) for dtype in face.dtypes or formats.FLOATS)
        raise ValueError(f"face {face.key} is exact for {exact}, x is {x.dtype}")
    for name, operand in zip(("x", *names), (x, *operands), strict=True):
        if operand.size < 8:
            raise ValueError(
                f"face {face.key}: {name} has {operand.size} elements; MLX binds an "
                "input of fewer than 8 as constants, not as a device buffer"
            )
    return _dot_rows_kernel(face)(
        inputs=[x, *operands],
        template=[("T", x.dtype), ("K", x.shape[-1]), ("NA", na), ("NB", nb)],
        grid=(32, n // (na + nb), 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )[0]


# Each weight of the rows, one a thread: weight w of row rows[r], of the
# kernel's K, the grid's width.
_TILE_ROWS = r"""
    const uint w = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    const size_t at = size_t(r) * size_t(K) + w;
    omlx_tile_rows<T, K>(m, size_t(rows[r]), int(w), w32 + at, wt + at);
"""

# Weight w of row `row`, as a tile kernel reads it: the code's run, its
# group's parameters, then the weight, stored as a float32 and rounded once
# to T. Its position in the run is not known when the kernel is compiled.
# As a kernel does, it calls params and fetch more than once on one cursor,
# into arrays of P and Codes, another group and run first, and keeps the
# second answers.
_TILE_ROWS_HEADER = r"""
template <typename T, int K, typename F>
METAL_FUNC void omlx_tile_rows(
    F m, size_t row, int w, device float* w32, device T* wt) {
  if constexpr (omlx_tile_face<F, K>::value) {
    using R = typename F::template Runs<K>;
    constexpr int GROUPS = (K + R::GROUP - 1) / R::GROUP;
    const R runs(m, row);
    typename R::P p[2];
    typename R::Codes c[2];
    p[0] = runs.params((w / R::GROUP + 1) % GROUPS);
    c[0] = runs.fetch((w / R::RUN + 1) % (K / R::RUN));
    p[1] = runs.params(w / R::GROUP);
    c[1] = runs.fetch(w / R::RUN);
    const float v = R::weight(p[1], c[1], w % R::RUN);
    *w32 = v;
    *wt = static_cast<T>(v);
  }
}
"""


@cache
def _tile_rows_kernel(face: Tile):
    matrices = kernels.Matrices()
    inputs = ["rows", *matrices.add("m", face)]
    return mx.fast.metal_kernel(
        name=f"omlx_format_tile_rows_{face.key}",
        input_names=inputs,
        output_names=["w32", "wt"],
        header=matrices.header + _TILE_ROWS_HEADER,
        source=matrices.declarations + _TILE_ROWS,
    )


def tile_rows(
    face: Tile, operands, k: int, rows, dtype: mx.Dtype
) -> tuple[mx.array, mx.array]:
    """Every weight of some rows of a matrix of K inputs, read through a tile
    face as a tile kernel reads it: ``weight(params(g), fetch(run), i)``.

    ``operands`` are the matrix's weight and the face's tensors, in order.
    ``rows`` are the rows' indices (in a stack, the row ``n`` of matrix
    ``e`` is ``e * N + n``), none past the matrix's (which is not checked).
    K is a whole number of runs of 8. Returns the weights twice, each array
    of shape ``(len(rows), k)``: as ``weight`` gives them, in float32, and
    rounded once to ``dtype``, a dtype the face is exact for, as a kernel
    stores them for activations of that dtype. Each operand has 8 elements
    or more. Anything else is refused with a ``ValueError``. The kernel is
    compiled for ``face``, ``dtype`` and K; the rows are values it is
    given.
    """
    if not isinstance(face, Tile):
        raise ValueError(f"{face!r} is not a Tile")
    if k <= 0 or k % 8:
        raise ValueError(f"k={k} is not one run or more of 8 inputs")
    rows = list(rows)
    if not rows or not all(
        isinstance(r, int) and not isinstance(r, bool) and r >= 0 for r in rows
    ):
        raise ValueError(f"rows={rows} are not one row index or more")
    operands, names = tuple(operands), ("weight", *face.tensors)
    if len(operands) != len(names):
        raise ValueError(
            f"face {face.key} reads {', '.join(names)}; operands given: {len(operands)}"
        )
    if not face.exact_for(dtype):
        exact = ", ".join(str(exact) for exact in face.dtypes)
        raise ValueError(f"face {face.key} is exact for {exact}, not {dtype}")
    for name, operand in zip(names, operands, strict=True):
        if operand.size < 8:
            raise ValueError(
                f"face {face.key}: {name} has {operand.size} elements; MLX binds an "
                "input of fewer than 8 as constants, not as a device buffer"
            )
    return tuple(
        _tile_rows_kernel(face)(
            inputs=[mx.array(rows, dtype=mx.uint64), *operands],
            template=[("T", dtype), ("K", k)],
            grid=(k, len(rows), 1),
            threadgroup=(min(k, 256), 1, 1),
            output_shapes=[(len(rows), k), (len(rows), k)],
            output_dtypes=[mx.float32, dtype],
        )
    )
