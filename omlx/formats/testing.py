# SPDX-License-Identifier: Apache-2.0
"""Checks for a format's faces, for the format's own tests: a face runs here
in the skeletons oMLX's kernels are built from, outside any model.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from . import Dot, kernels

_STORE = r"""
template <typename T, int... R>
METAL_FUNC void store_rows(omlx_ints<R...>, thread const float* result, device T* y) {
  ((y[R] = static_cast<T>(result[R])), ...);
}
"""

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


@cache
def _dot_rows_kernel(face: Dot):
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("m", face)]
    return mx.fast.metal_kernel(
        name=f"omlx_format_dot_rows_{face.key}",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header + _STORE,
        source=matrices.declarations + _ROWS,
    )


def dot_rows(
    face: Dot, operands, x: mx.array, n: int, packs: tuple[int, int] = (1, 1)
) -> mx.array:
    """The products of ``x`` (K activations) with the first ``n`` rows of a
    matrix, through the row skeleton.

    ``operands`` are the matrix's weight and the face's tensors, in order.
    A launch takes ``packs[0]`` rows as the skeleton's first pack and the
    next ``packs[1]`` as its second; ``n`` is a multiple of their sum. The
    kernel is compiled for ``face`` and K.
    """
    na, nb = packs
    if min(na, nb) < 0 or na + nb == 0:
        raise ValueError(f"packs={packs}: a launch takes one row or more")
    if n % (na + nb):
        raise ValueError(f"n={n} is not a multiple of the {na + nb} rows of a launch")
    if x.ndim != 1:
        raise ValueError(f"x is one vector of K activations, not of shape {x.shape}")
    operands, names = tuple(operands), ("weight", *face.tensors)
    if len(operands) != len(names):
        raise ValueError(
            f"face {face.key} reads {', '.join(names)}: {len(operands)} operands given"
        )
    if not face.exact_for(x.dtype):
        exact = ", ".join(str(dtype) for dtype in face.dtypes)
        raise ValueError(f"face {face.key} is exact for {exact}, x is {x.dtype}")
    for name, operand in zip(names, operands, strict=True):
        if operand.size < 8:
            raise ValueError(
                f"face {face.key}: {name} has {operand.size} elements; MLX binds an "
                "input of fewer than 8 as constants, not as the device pointer of a face"
            )
    return _dot_rows_kernel(face)(
        inputs=[x, *operands],
        template=[("T", x.dtype), ("K", x.shape[-1]), ("NA", na), ("NB", nb)],
        grid=(32, n // (na + nb), 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )[0]
