# SPDX-License-Identifier: Apache-2.0
"""MLX's wide mat-vec reduction over a format's runs of eight weights."""

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
