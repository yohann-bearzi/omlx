# SPDX-License-Identifier: Apache-2.0
"""MLX's wide mat-vec reduction over a format's runs of eight weights."""

from functools import cache

import mlx.core as mx

from . import kernels

_HEADER = r"""
template <int K, typename R>
struct omlx_wide_groups {
  static_assert(K % R::RUN == 0, "wide input requires complete tile runs");
  static constant constexpr int groups = (K - 1) / R::GROUP + 1;
  static constant constexpr int runs = (K < R::GROUP ? K : R::GROUP) / R::RUN;

  static constexpr bool past_tail(int g, int r) {
    return g == groups - 1 && r > ((K - 1) % R::GROUP) / R::RUN;
  }

  // The cold proof compiles the actual face and K. Sample both ends of
  // its final group and the next run against the bounds used by the loop.
  static constexpr bool visits(int k) {
    const int g = k / R::GROUP;
    const int r = (k % R::GROUP) / R::RUN;
    return g < groups && r < runs && !past_tail(g, r);
  }
};

template <typename T, int K, int N, int V, bool PARTIAL_M, typename F>
inline void omlx_wide(
    const F matrix, const device T* x, device T* y, int rows,
    uint3 tile, ushort simd, ushort lane) {
  if constexpr (omlx_tile_face<F, K>::value) {
    using R = typename F::template Runs<K>;
    using Walk = omlx_wide_groups<K, R>;
    if constexpr (K % R::GROUP != 0) {
      constexpr int tail_start = ((K - 1) / R::GROUP) * R::GROUP;
      static_assert(Walk::visits(tail_start) && Walk::visits(K - R::RUN) &&
                    !Walk::visits(K), "wide traversal must cover the final group");
    }
    constexpr bool PARTIAL_N = (N % 8) != 0;
    const int col = int(tile.y) * 8 + int(simd) * 4 + lane / 8;
    const int first = int(tile.x) * V;
    const int row = PARTIAL_N ? min(col, N - 1) : col;
    const R runs(matrix, size_t(row));
    const device T* vectors[V];
    float result[V] = {};
    #pragma unroll
    for (int v = 0; v < V; ++v) {
      const int m = PARTIAL_M ? min(first + v, rows - 1) : first + v;
      vectors[v] = x + size_t(m) * K;
    }
    for (int g = lane % 8; g < Walk::groups; g += 8) {
      const auto params = runs.params(g);
      #pragma unroll
      for (int r = 0; r < Walk::runs; ++r) {
        if constexpr (K % R::GROUP != 0) {
          if (Walk::past_tail(g, r)) break;
        }
        const int k0 = g * R::GROUP + r * R::RUN;
        const auto codes = runs.fetch(k0 / R::RUN);
        float weights[R::RUN];
        #pragma unroll
        for (int i = 0; i < R::RUN; ++i) {
          weights[i] = R::weight(params, codes, i);
        }
        #pragma unroll
        for (int v = 0; v < V; ++v) {
          float part = 0;
          #pragma unroll
          for (int i = 0; i < R::RUN; ++i) {
            part += float(vectors[v][k0 + i]) * weights[i];
          }
          result[v] += part;
        }
      }
    }
    #pragma unroll
    for (int v = 0; v < V; ++v) {
      result[v] += simd_shuffle_down(result[v], 4);
      result[v] += simd_shuffle_down(result[v], 2);
      result[v] += simd_shuffle_down(result[v], 1);
    }
    if (lane % 8 == 0 && (!PARTIAL_N || col < N)) {
      #pragma unroll
      for (int v = 0; v < V; ++v) {
        if (!PARTIAL_M || first + v < rows) {
          y[size_t(first + v) * N + col] = T(result[v]);
        }
      }
    }
  }
}
"""
_SOURCE = """
    int rows = 0;
    if constexpr (PARTIAL_M) {
      rows = 1;
      for (int d = 0; d < x_ndim - 1; ++d) {
        rows *= x_shape[d];
      }
    }
    omlx_wide<T, K, N, V, PARTIAL_M>(
        w, x, y, rows,
        threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""


def kernel(face):
    return kernels.face_kernel(
        "omlx_format_linear_wide", ("x", "w"), {"w": face}, ("y",), _HEADER, _SOURCE
    )


# Two concurrent decode requests pass two singleton matrices. MLX folds them
# into one wide product unless intact rows have a gap between the matrices.
# The same compiled graph receives both layouts; decide on the actual GPU
# strides. The contiguous data input also covers a column-strided activation.
_PAIR_SOURCE = """
    if (layout_strides[0] != K && layout_strides[1] == K && layout_strides[2] == 1) {
        const uint out = threadgroup_position_in_grid.y * 8 +
                         simdgroup_index_in_threadgroup * 4;
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            float partial[4];
            qmv_rows<T, K, 4, 0>(d, out, d, 0, x + m * K,
                                 thread_index_in_simdgroup, partial);
            if (thread_index_in_simdgroup == 0) {
                store_rows(omlx_iota<4>(), partial, y + size_t(m) * N + out);
            }
        }
    } else {
        omlx_wide<T, K, N, 2, false>(w, x, y, 2, threadgroup_position_in_grid,
                                     simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    }
"""


@cache
def pair_kernel(dot, tile):
    matrices = kernels.Matrices()
    inputs = ["layout", "x", *matrices.add("w", tile), *matrices.add("d", dot)]
    mm = mx.fast.metal_kernel(
        name=f"omlx_format_decode_pair_{dot.key}_{tile.key}",
        input_names=inputs,
        output_names=["y"],
        ensure_row_contiguous=False,
        header=matrices.header + kernels.STORE_ROWS + _HEADER,
        source=matrices.declarations + _PAIR_SOURCE,
    )
    names = ("weight", *tile.tensors, "weight", *dot.tensors)
    unique = tuple(dict.fromkeys(names))
    indices = tuple(unique.index(name) for name in names)

    def launch(*, inputs, **kwargs):
        x, *weights = inputs
        return mm(
            inputs=[x, mx.contiguous(x), *(weights[i] for i in indices)], **kwargs
        )

    return unique, launch


def pair_stand_in(plan, metal):
    dtype, k, _, _, _ = metal
    arrays = kernels.zeros(
        ((2, 1, k), dtype), *((a.shape, a.dtype) for a in plan.operands)
    )
    _, graph = describe(plan, arrays[0])
    return arrays, graph


def describe(plan, x):
    k = x.shape[-1]
    n = plan.operands[0].shape[0]
    rows = x.size // k
    tiles = (rows + 4) // 5
    vectors = (rows + tiles - 1) // tiles
    partial_m = rows % vectors != 0
    grid = (32 * tiles, 2 * ((n + 7) // 8), 1)
    shape = (*x.shape[:-1], n)
    dtype = x.dtype
    template = [
        ("T", dtype),
        ("K", k),
        ("N", n),
        ("V", vectors),
        ("PARTIAL_M", partial_m),
    ]
    mm = plan.kernel

    def graph(x, *operands):
        return mm(
            inputs=[x, *operands],
            template=template,
            grid=grid,
            threadgroup=(32, 2, 1),
            output_shapes=[shape],
            output_dtypes=[dtype],
        )[0]

    return (dtype, k, n, vectors, partial_m), graph


def stand_in(plan, metal):
    dtype, k, _n, vectors, partial_m = metal
    rows = 2 * vectors - 1 if partial_m else vectors
    arrays = kernels.zeros(
        ((rows, k), dtype), *((a.shape, a.dtype) for a in plan.operands)
    )
    _, graph = describe(plan, arrays[0])
    return arrays, graph
