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

template <typename T, int K, int N, int V, bool PARTIAL_M, int FIXED_FIRST, typename F>
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
    constexpr bool PACK_X = FIXED_FIRST >= 0;
    const int col = int(tile.y) * 8 + int(simd) * 4 + lane / 8;
    const int first = PACK_X ? FIXED_FIRST : int(tile.x) * V;
    const int row = PARTIAL_N ? min(col, N - 1) : col;
    const R runs(matrix, size_t(row));
    const device T* vectors[V];
    float result[V] = {};
    const device T* x_base = x + size_t(first) * K;
    if constexpr (!PACK_X) {
      #pragma unroll
      for (int v = 0; v < V; ++v) {
        const int m = PARTIAL_M ? min(first + v, rows - 1) : first + v;
        vectors[v] = x + size_t(m) * K;
      }
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
          if constexpr (PACK_X) {
            // A contiguous view can start at an odd element offset.
            typedef metal::vec<T, 8> Vec8 __attribute__((aligned(alignof(T))));
            const auto values = *reinterpret_cast<const device Vec8*>(
                x_base + size_t(v) * K + k0);
            #pragma unroll
            for (int i = 0; i < R::RUN; ++i) {
              part += float(values[i]) * weights[i];
            }
          } else {
            #pragma unroll
            for (int i = 0; i < R::RUN; ++i) {
              part += float(vectors[v][k0 + i]) * weights[i];
            }
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
    const uint3 tile = threadgroup_position_in_grid;
    const ushort simd = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    if constexpr (EXACT_ROWS == 7) {
      if (tile.x == 0) {
        omlx_wide<T, K, N, 4, false, 0>(w, x, y, 7, tile, simd, lane);
      } else {
        omlx_wide<T, K, N, 3, false, 4>(w, x, y, 7, tile, simd, lane);
      }
    } else if constexpr (EXACT_ROWS == 5) {
      omlx_wide<T, K, N, 5, false, 0>(w, x, y, 5, tile, simd, lane);
    } else {
      int rows = 0;
      if constexpr (PARTIAL_M) {
        rows = 1;
        for (int d = 0; d < x_ndim - 1; ++d) rows *= x_shape[d];
      }
      omlx_wide<T, K, N, V, PARTIAL_M, -1>(w, x, y, rows, tile, simd, lane);
    }
"""


def kernel(face):
    return kernels.face_kernel(
        "omlx_format_linear_wide", ("x", "w"), {"w": face}, ("y",), _HEADER, _SOURCE
    )


def layout(rows):
    tiles = (rows + 4) // 5
    vectors = (rows + tiles - 1) // tiles
    return vectors, rows % vectors != 0, rows if rows in (5, 7) else 0


def template(dtype, k, n, vectors, partial_m, exact_rows):
    return [
        ("T", dtype),
        ("K", k),
        ("N", n),
        ("V", vectors),
        ("PARTIAL_M", partial_m),
        ("EXACT_ROWS", exact_rows),
    ]


def _graph(plan, x, metal):
    dtype, k, n, vectors, _, _ = metal
    rows = x.size // k
    tiles = (rows + vectors - 1) // vectors
    grid = (32 * tiles, 2 * ((n + 7) // 8), 1)
    shape = (*x.shape[:-1], n)
    params = template(*metal)
    mm = plan.kernel

    def graph(x, *operands):
        return mm(
            inputs=[x, *operands],
            template=params,
            grid=grid,
            threadgroup=(32, 2, 1),
            output_shapes=[shape],
            output_dtypes=[dtype],
        )[0]

    return graph


def describe(plan, x):
    k = x.shape[-1]
    metal = (x.dtype, k, plan.operands[0].shape[0], *layout(x.size // k))
    return metal, _graph(plan, x, metal)


def stand_in(plan, metal):
    dtype, k, _n, vectors, partial_m, exact_rows = metal
    rows = exact_rows or (2 * vectors - 1 if partial_m else vectors)
    arrays = kernels.zeros(
        ((rows, k), dtype), *((a.shape, a.dtype) for a in plan.operands)
    )
    return arrays, _graph(plan, arrays[0], metal)
