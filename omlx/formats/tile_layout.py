# SPDX-License-Identifier: Apache-2.0
"""MLX's dense QMM layout choice for inputs with several leading matrices."""

from functools import cache

import mlx.core as mx


@cache
def _layout_kernel():
    return mx.fast.metal_kernel(
        name="omlx_format_per_matrix",
        input_names=["x"],
        output_names=["selected"],
        ensure_row_contiguous=False,
        source="""
            size_t span = 1;
            bool row_contiguous = true;
            for (int i = x_ndim - 1; i >= 0; --i) {
              row_contiguous &= x_shape[i] == 1 || size_t(x_strides[i]) == span;
              span *= size_t(x_shape[i]);
            }
            const int last = x_ndim - 1;
            selected[0] = !row_contiguous && x_strides[last] == 1 &&
                x_strides[last - 1] == x_shape[last];
        """,
    )


def per_matrix(x):
    """Whether MLX keeps each matrix's rows when it sees these live strides."""
    return _layout_kernel()(
        inputs=[x],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[()],
        output_dtypes=[mx.bool_],
    )[0]
