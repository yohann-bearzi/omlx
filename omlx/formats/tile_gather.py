# SPDX-License-Identifier: Apache-2.0
"""MLX's steel sorted gather over a format's runs of eight weights."""

from __future__ import annotations

import logging
import threading
from functools import cache

import mlx.core as mx

from ..patches.m5_gather_qmm_nax import _read_mlx_headers
from . import Tile, kernels

logger = logging.getLogger(__name__)
_lock = threading.Lock()


@cache
def _headers():
    text = _read_mlx_headers(
        (
            "mlx/backend/metal/kernels/steel/gemm/gemm.h",
            "mlx/backend/metal/kernels/quantized_utils.h",
        )
    )
    if text is None:
        logger.warning("MLX steel headers unavailable; sorted format layers use rows")
    return text


_HEADER = r"""
namespace omlx_steel_gather {
constexpr constant int BM = 16;
constexpr constant int BN = 32;
constexpr constant int BK = 32;

// Each of the 64 threads writes 16 weights from one matrix row. A thread
// covers two complete groups at GROUP=8 and part or all of one larger group.
template <typename T, typename F, int K, bool ALIGN_N>
struct WeightLoader {
  using R = typename F::template Runs<K>;
  static constant constexpr int values = 16;
  static constant constexpr int group_width = R::GROUP < values ? R::GROUP : values;
  const R runs;
  const short row;
  threadgroup T* dst;
  int col;

  WeightLoader(const F m, size_t first, short columns, threadgroup T* tile,
               ushort simd, ushort lane) thread
      : runs(m, first + ((ALIGN_N || (simd * 32 + lane) / 2 < columns)
                             ? (simd * 32 + lane) / 2 : 0)),
        row((simd * 32 + lane) / 2),
        dst(tile + row * (BK + 16 / sizeof(T)) + (lane % 2) * values),
        col((lane % 2) * values) {}

  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < values / group_width; ++g) {
      const auto p = runs.params((col + g * group_width) / R::GROUP);
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < group_width / R::RUN; ++r) {
        const short j = g * group_width + r * R::RUN;
        const auto codes = runs.fetch((col + j) / R::RUN);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < R::RUN; ++i) {
          dst[j + i] = T(R::weight(p, codes, int(i)));
        }
      }
    }
  }
  METAL_FUNC void load_safe(short2 shape) const thread {
    if (row < shape.y) {
      load_unsafe();
    } else {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < values; ++i) {
        dst[i] = T(0);
      }
    }
  }
  METAL_FUNC void next() thread { col += BK; }
};

template <typename T, int K, bool ALIGN_N, typename F>
METAL_FUNC void gather(
    const device T* x, F matrix, const device int* offsets, device T* y,
    int rows, int n, int experts, threadgroup T* xs, threadgroup T* ws,
    uint3 tile, ushort simd, ushort lane) {
  if constexpr (omlx_tile_face<F, K>::value) {
    static_assert(K > 0 && K % BK == 0, "steel gather: K is whole 32-wide tiles");
    constexpr int stride = BK + 16 / sizeof(T);
    using MMA = mlx::steel::BlockMMA<
        T, T, BM, BN, BK, 1, 2, false, true, stride, stride>;
    using X = mlx::steel::BlockLoader<T, BM, BK, stride, 1, 64>;
    int row, expert;
    short height;
    if (!schedule_row_tile<BM>(offsets, experts, rows, tile.y, lane,
                               row, expert, height)) {
      return;
    }
    const int col = tile.x * BN;
    const short width = ALIGN_N ? BN : min(BN, n - col);
    MMA mma(simd, lane);
    X a(x + size_t(row) * K, K, xs, simd, lane);
    WeightLoader<T, F, K, ALIGN_N> b(
        matrix, size_t(expert) * n + col, width, ws, simd, lane);
    y += size_t(row) * n + col;
    if (height == BM && (ALIGN_N || width == BN)) {
      gemm_loop_aligned(xs, ws, mma, a, b, K / BK);
      mma.store_result(y, n);
    } else if (ALIGN_N || width == BN) {
      gemm_loop_unaligned<false, true, true>(
          xs, ws, mma, a, b, K / BK, height, width, short(BK));
      mma.store_result_safe(y, n, short2(BN, height));
    } else if (height == BM) {
      gemm_loop_unaligned<true, false, true>(
          xs, ws, mma, a, b, K / BK, height, width, short(BK));
      mma.store_result_safe(y, n, short2(width, BM));
    } else {
      gemm_loop_unaligned<false, false, true>(
          xs, ws, mma, a, b, K / BK, height, width, short(BK));
      mma.store_result_safe(y, n, short2(width, height));
    }
  }
}
}  // namespace omlx_steel_gather
"""

_SOURCE = """
    constexpr int stride = omlx_steel_gather::BK + 16 / sizeof(T);
    threadgroup T xs[omlx_steel_gather::BM * stride];
    threadgroup T ws[omlx_steel_gather::BN * stride];
    omlx_steel_gather::gather<T, K, ALIGN_N>(
        x, w, offsets, y, meta[0], meta[1], meta[2], xs, ws,
        threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""


@cache
def _build(face: Tile):
    header = _headers()
    if header is None:
        return None
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("w", face), "offsets", "meta"]
    return mx.fast.metal_kernel(
        name=f"omlx_steel_gather_{face.key}",
        input_names=inputs,
        output_names=["y"],
        header=header + matrices.header + _HEADER,
        source=matrices.declarations + _SOURCE,
    )


def kernel(face: Tile):
    """Build the kernel once, even when two engine threads first need it."""
    with _lock:
        return _build(face)


def launch(mm, x, operands, indices):
    """One 16 × 32 tile per group; MLX schedules rows from expert offsets."""
    rows, _, k = x.shape
    experts, n = operands[0].shape[:2]
    counts = mx.zeros((max(8, experts),), mx.int32).at[indices].add(1)
    offsets = mx.cumsum(counts, inclusive=False)
    meta = mx.array([rows, n, experts], dtype=mx.int32)
    return mm(
        inputs=[x, *operands, offsets, meta],
        template=[("T", x.dtype), ("K", k), ("ALIGN_N", n % 32 == 0)],
        grid=(64 * ((n + 31) // 32), (rows + 15) // 16 + min(experts, rows), 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows, 1, n)],
        output_dtypes=[x.dtype],
    )[0]
