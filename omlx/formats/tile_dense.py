# SPDX-License-Identifier: Apache-2.0
"""Unified parameterized direct dense prefill skeleton over a format's tile face.

One steel emitter (Apple GPU without tensor units, e.g. M3 Ultra) and one NAX
emitter (Apple GPU with tensor units, e.g. M5 Max) serve every dense matrix
above ``linear.MAX_ROWS``. There is no MoE schedule, no expert offsets buffer,
and no per-call policy lookup: ``build_plan`` resolves the layer's per-chip
schedule once from ``omlx.formats.tuning`` at construction time.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from functools import cache
from typing import Any, NamedTuple

import mlx.core as mx

from .. import formats
from ..patches.m5_gather_qmm_nax import _read_mlx_headers
from . import Tile, affine, kernels, tile_wide, tuning
from .tuning import DenseTilePolicy, ParamStaging

logger = logging.getLogger(__name__)
_lock = threading.Lock()


@cache
def _steel_headers() -> str | None:
    text = _read_mlx_headers(
        (
            "mlx/backend/metal/kernels/steel/gemm/gemm.h",
            "mlx/backend/metal/kernels/quantized_utils.h",
        )
    )
    if text is None:
        logger.warning("MLX steel headers unavailable; dense format prefill decodes")
    return text


@cache
def _nax_headers() -> str | None:
    text = _read_mlx_headers(("mlx/backend/metal/kernels/steel/gemm/nax.h",))
    if text is None:
        logger.warning("MLX NAX headers unavailable; dense format prefill decodes")
    return text


_STEEL_HEADER = r"""
namespace omlx_steel_dense {

template <
    typename T,
    typename F,
    int K,
    int BN,
    int BK,
    int TGP_SIZE,
    bool ALIGN_N,
    bool VEC_STORE,
    bool CACHE_P>
struct WeightLoader {
  using R = typename F::template Runs<K>;
  using P = typename R::P;
  static constant constexpr int values = (BN * BK) / TGP_SIZE;
  static constant constexpr int tpr = values > 0 ? (BK / values) : 1;
  static constant constexpr int group_width =
      R::GROUP < values ? R::GROUP : values;
  static constant constexpr int n_groups =
      group_width > 0 ? (values / group_width) : 0;
  static constant constexpr int runs_per_group = group_width / R::RUN;
  static_assert(
      values > 0 && values * TGP_SIZE == BN * BK && tpr * values == BK,
      "steel loader tile split");
  static_assert(
      n_groups * group_width == values && runs_per_group > 0 &&
          runs_per_group * R::RUN == group_width,
      "steel loader group split");
  static constant constexpr bool can_cache_p =
      CACHE_P && (n_groups == 1) && (R::GROUP > BK) && (R::GROUP % BK == 0);
  static constant constexpr int group_steps =
      R::GROUP > BK ? (R::GROUP / BK) : 1;

  const R runs;
  const short row;
  threadgroup T* dst;
  uint col;
  P cached_p;
  short step_cnt;

  WeightLoader(
      const F m,
      size_t first,
      short columns,
      threadgroup T* tile,
      ushort simd,
      ushort lane,
      int k_start = 0) thread
      : runs(
            m,
            first +
                ((ALIGN_N || int(simd * 32 + lane) / tpr < int(columns))
                     ? int(simd * 32 + lane) / tpr
                     : 0)),
        row(short(int(simd * 32 + lane) / tpr)),
        dst(tile + row * (BK + 16 / sizeof(T)) +
            (int(simd * 32 + lane) % tpr) * values),
        col(uint(k_start) + uint((int(simd * 32 + lane) % tpr) * values)),
        step_cnt(short((k_start % R::GROUP) / BK)) {
    if constexpr (can_cache_p) {
      if (col < uint(K)) {
        cached_p = runs.params(int(col / uint(R::GROUP)));
      }
    }
  }

  template <int... I>
  METAL_FUNC static void store_vec_(
      threadgroup T* p_dst,
      thread const P& p,
      const typename R::Codes codes,
      omlx_ints<I...>) {
    metal::vec<T, R::RUN> v;
    ((v[I] = static_cast<T>(R::weight(p, codes, int(I)))), ...);
    *(threadgroup metal::vec<T, R::RUN>*)p_dst = v;
  }

  METAL_FUNC void load_unsafe() const thread {
    if constexpr (can_cache_p) {
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < runs_per_group; ++r) {
        const short j = r * R::RUN;
        const auto codes = runs.fetch(int((col + uint(j)) / uint(R::RUN)));
        if constexpr (VEC_STORE) {
          store_vec_(dst + j, cached_p, codes, omlx_iota<R::RUN>());
        } else {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < R::RUN; ++i) {
            dst[j + i] = T(R::weight(cached_p, codes, int(i)));
          }
        }
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short g = 0; g < n_groups; ++g) {
        const auto p =
            runs.params(int((col + uint(g * group_width)) / uint(R::GROUP)));
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < runs_per_group; ++r) {
          const short j = g * group_width + r * R::RUN;
          const auto codes = runs.fetch(int((col + uint(j)) / uint(R::RUN)));
          if constexpr (VEC_STORE) {
            store_vec_(dst + j, p, codes, omlx_iota<R::RUN>());
          } else {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < R::RUN; ++i) {
              dst[j + i] = T(R::weight(p, codes, int(i)));
            }
          }
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

  METAL_FUNC void next() thread {
    col += uint(BK);
    if constexpr (can_cache_p) {
      ++step_cnt;
      if (step_cnt == group_steps) {
        step_cnt = 0;
        if (col < uint(K)) {
          cached_p = runs.params(int(col / uint(R::GROUP)));
        }
      }
    }
  }
};

template <
    typename T,
    typename U,
    int K,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool ALIGN_N,
    bool VEC_STORE,
    bool CACHE_P,
    int X_NREADS,
    typename F>
METAL_FUNC void dense_steel(
    const device T* x,
    const F matrix,
    device U* y,
    int rows,
    int n,
    int k_start,
    int k_iters,
    threadgroup T* xs,
    threadgroup T* ws,
    uint3 tile,
    ushort simd,
    ushort lane) {
  if constexpr (omlx_tile_face<F, K>::value) {
    static_assert(K > 0 && K % BK == 0, "steel dense: K is whole BK tiles");
    constexpr int TGP = WM * WN * 32;
    constexpr int stride = BK + 16 / sizeof(T);
    using MMA = mlx::steel::BlockMMA<
        T, U, BM, BN, BK, WM, WN, false, true, stride, stride>;
    constexpr short n_reads =
        X_NREADS > 0 ? short(X_NREADS) : short((BM * BK) / TGP);
    constexpr short tcols = n_reads > 0 ? short(BK / n_reads) : short(0);
    constexpr short trows = tcols > 0 ? short(TGP / tcols) : short(0);
    static_assert(
        n_reads > 0 && tcols > 0 && tcols * n_reads == BK,
        "steel loader column tile");
    static_assert(
        trows > 0 && trows * tcols == TGP && BM % trows == 0,
        "steel loader row tile");
    using X = mlx::steel::BlockLoader<T, BM, BK, stride, 1, TGP, 1, n_reads>;
    const int row = int(tile.y) * BM;
    const int col = int(tile.x) * BN;
    const short height = short(metal::min(BM, rows - row));
    const short width = ALIGN_N ? short(BN) : short(metal::min(BN, n - col));
    MMA mma(simd, lane);
    X a(x + size_t(row) * K + k_start, K, xs, simd, lane);
    WeightLoader<T, F, K, BN, BK, TGP, ALIGN_N, VEC_STORE, CACHE_P> b(
        matrix, size_t(col), width, ws, simd, lane, k_start);
    y += size_t(row) * n + col;
    if (height == BM && (ALIGN_N || width == BN)) {
      gemm_loop_aligned(xs, ws, mma, a, b, k_iters);
      mma.store_result(y, n);
    } else if (ALIGN_N || width == BN) {
      gemm_loop_unaligned<false, true, true>(
          xs, ws, mma, a, b, k_iters, height, width, short(BK));
      mma.store_result_safe(y, n, short2(BN, height));
    } else if (height == BM) {
      gemm_loop_unaligned<true, false, true>(
          xs, ws, mma, a, b, k_iters, height, width, short(BK));
      mma.store_result_safe(y, n, short2(width, BM));
    } else {
      gemm_loop_unaligned<false, false, true>(
          xs, ws, mma, a, b, k_iters, height, width, short(BK));
      mma.store_result_safe(y, n, short2(width, height));
    }
  }
}

template <
    typename T,
    int K,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool ALIGN_N,
    bool VEC_STORE,
    bool CACHE_P,
    int X_NREADS,
    int SPLIT_K,
    int K_PART,
    typename F,
    typename U>
METAL_FUNC void dense_entry(
    const device T* x,
    const F matrix,
    device U* y,
    int rows,
    int n,
    threadgroup T* xs,
    threadgroup T* ws,
    uint3 tile,
    ushort simd,
    ushort lane) {
  if constexpr (SPLIT_K > 1) {
    static_assert(
        K > 0 && K % BK == 0 && K_PART > 0 && K_PART % BK == 0,
        "steel dense split_k: K and K_PART are whole BK tiles");
    const int part = int(tile.z);
    const int k_start = part * K_PART;
    const int k_rem = metal::max(0, metal::min(K_PART, K - k_start));
    if (k_rem <= 0) {
      return;
    }
    const size_t out_stride = size_t(rows) * size_t(n);
    dense_steel<
        T, U, K, BM, BN, BK, WM, WN, ALIGN_N, VEC_STORE, CACHE_P, X_NREADS>(
        x, matrix, y + size_t(part) * out_stride, rows, n, k_start, k_rem / BK,
        xs, ws, tile, simd, lane);
  } else {
    dense_steel<
        T, U, K, BM, BN, BK, WM, WN, ALIGN_N, VEC_STORE, CACHE_P, X_NREADS>(
        x, matrix, y, rows, n, 0, K / BK, xs, ws, tile, simd, lane);
  }
}

}  // namespace omlx_steel_dense
"""

_STEEL_SOURCE = """
    int rows = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
      rows *= x_shape[d];
    }
    constexpr int stride = BK + 16 / sizeof(T);
    threadgroup T xs[BM * stride];
    threadgroup T ws[BN * stride];
    omlx_steel_dense::dense_entry<
        T, K, BM, BN, BK, WM, WN, ALIGN_N, VEC_STORE, CACHE_P, X_NREADS,
        SPLIT_K, K_PART>(
        x, w, y, rows, int(w_weight_shape[0]), xs, ws,
        threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""

_NAX_HEADER = r"""
using namespace metal;
namespace omlx_nax_dense {
using namespace mlx::steel;

template <
    typename T,
    typename F,
    int K,
    int BN,
    int BK,
    int LT,
    bool ALIGN_N>
struct NaxWeightLoader {
  using R = typename F::template Runs<K>;
  using P = typename R::P;
  using Codes = typename R::Codes;
  static constant constexpr int kTPR = LT / BN;
  static constant constexpr int kVPT = BK / kTPR;
  static constant constexpr int kRuns = kVPT / R::RUN;
  static constant constexpr int kGV = kVPT < R::GROUP ? kVPT : R::GROUP;
  static constant constexpr int kNG = kVPT / kGV;
  static constant constexpr int kRPG = kGV / R::RUN;
  static constant constexpr int kBKP = BK + 16 / sizeof(T);
  static_assert(LT % BN == 0 && BK % kTPR == 0, "nax loader tile split");
  static_assert(kRuns * R::RUN == kVPT && kNG * kRPG == kRuns, "nax group split");

  const R runs;
  const short row;
  const short col;
  Codes codes[kRuns];
  P p[kNG];

  METAL_FUNC NaxWeightLoader(
      const F m,
      const size_t w_row,
      const short tgp_bn,
      const uint lid) thread
      : runs(
            m,
            w_row + size_t((lid < uint(LT) && (ALIGN_N || int(lid / kTPR) < int(tgp_bn)))
                               ? int(lid / kTPR)
                               : 0)),
        row(short(lid / kTPR)),
        col(short((lid % kTPR) * kVPT)) {}

  template <int... Rn, int... Gn>
  METAL_FUNC void fetch_(omlx_ints<Rn...>, omlx_ints<Gn...>, const int kb) thread {
    const int run0 = (kb * BK + col) / R::RUN;
    ((codes[Rn] = runs.fetch(run0 + Rn)), ...);
    ((p[Gn] = runs.params((kb * BK + col + Gn * kGV) / R::GROUP)), ...);
  }

  template <int... Rn, int... Gn>
  METAL_FUNC void fetch_tail_(
      omlx_ints<Rn...>, omlx_ints<Gn...>, const int kb, const int k_valid) thread {
    const int run0 = (kb * BK + col) / R::RUN;
    ((col + Rn * R::RUN < k_valid ? void(codes[Rn] = runs.fetch(run0 + Rn))
                                  : void()),
     ...);
    ((col + Gn * kGV < k_valid
          ? void(p[Gn] = runs.params((kb * BK + col + Gn * kGV) / R::GROUP))
          : void()),
     ...);
  }

  template <int r, int... I>
  METAL_FUNC void store_run_(threadgroup T* dst, omlx_ints<I...>) const thread {
    vec<T, R::RUN> v;
    ((v[I] = static_cast<T>(R::weight(p[r / kRPG], codes[r], int(I)))), ...);
    *(threadgroup vec<T, R::RUN>*)(dst + r * R::RUN) = v;
  }

  template <int... Rn>
  METAL_FUNC void store_(
      omlx_ints<Rn...>, threadgroup T* dst, const int k_valid) const thread {
    ((col + Rn * R::RUN < k_valid
          ? store_run_<Rn>(dst, omlx_iota<R::RUN>())
          : void()),
     ...);
  }

  METAL_FUNC void fetch(const int kb) thread {
    fetch_(omlx_iota<kRuns>(), omlx_iota<kNG>(), kb);
  }

  METAL_FUNC void fetch_tail(const int kb, const int k_valid) thread {
    fetch_tail_(omlx_iota<kRuns>(), omlx_iota<kNG>(), kb, k_valid);
  }

  METAL_FUNC void store_words(threadgroup T* Ws, const int k_valid) const thread {
    store_(omlx_iota<kRuns>(), Ws + row * kBKP + col, k_valid);
  }

  METAL_FUNC void store(threadgroup T* Ws) const thread {
    store_words(Ws, BK);
  }

  template <int... I>
  METAL_FUNC void zero_(omlx_ints<I...>, threadgroup T* dst) const thread {
    ((dst[I] = T(0)), ...);
  }

  METAL_FUNC void zero(threadgroup T* Ws) const thread {
    zero_(omlx_iota<kVPT>(), Ws + row * kBKP + col);
  }
};

template <typename T>
METAL_FUNC void mma_2m1n(
    thread BaseNAXFrag::dtype_frag_t<float>& Cm0,
    thread BaseNAXFrag::dtype_frag_t<float>& Cm1,
    const thread BaseNAXFrag::dtype_frag_t<T>& Am0,
    const thread BaseNAXFrag::dtype_frag_t<T>& Am1,
    const thread BaseNAXFrag::dtype_frag_t<T>& B) {
  // MLX's nax.h 2M x 1N BaseNAXFrag::mma declares matmul2d_descriptor(16, 32,
  // 16) instead of (32, 16, 16), zeroing Cm1; issue the 32x16x16 op directly.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      32,
      16,
      16,
      false,
      true,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<T, T, float>();
  auto ct_b =
      gemm_op.template get_right_input_cooperative_tensor<T, T, float>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = Am0[i];
    ct_a[BaseNAXFrag::kElemsPerFrag + i] = Am1[i];
    ct_b[i] = B[i];
    ct_c[i] = Cm0[i];
    ct_c[BaseNAXFrag::kElemsPerFrag + i] = Cm1[i];
  }
  gemm_op.run(ct_a, ct_b, ct_c);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    Cm0[i] = ct_c[i];
    Cm1[i] = ct_c[BaseNAXFrag::kElemsPerFrag + i];
  }
}

template <typename T, short kTM, short kTN, int BKP, bool FULL>
METAL_FUNC void sub_step(
    thread NAXTile<float, kTM, kTN>& Dtile,
    const device T* xn,
    const threadgroup T* ws,
    const int K,
    const short sgp_sm) {
  constexpr short kSK = 32;
  constexpr short kTK = 2;
  NAXTile<T, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<T, kTM, kTK> Atile;
    volatile int compiler_barrier;
    Atile.load(xn, K);
    Btile.template load<T, BKP, 1>(ws);
    if constexpr (kTN == 1) {
      STEEL_PRAGMA_UNROLL
      for (short mm = 0; mm < kTM; mm += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < kTK; kk++) {
          mma_2m1n<T>(
              Dtile.frag_at(mm, 0),
              Dtile.frag_at(mm + 1, 0),
              Atile.frag_at(mm, kk),
              Atile.frag_at(mm + 1, kk),
              Btile.frag_at(0, kk));
        }
      }
    } else {
      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});
    }
    (void)compiler_barrier;
  } else {
    Btile.template load<T, BKP, 1>(ws);
    if constexpr (kTN == 1) {
      STEEL_PRAGMA_UNROLL
      for (short mm = 0; mm < kTM; mm += 2) {
        if (mm * 16 < sgp_sm) {
          NAXTile<T, 2, kTK> Apair;
          Apair.load_safe(xn + mm * 16 * K, K, short2(kSK, sgp_sm - mm * 16));
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) {
            mma_2m1n<T>(
                Dtile.frag_at(mm, 0),
                Dtile.frag_at(mm + 1, 0),
                Apair.frag_at(0, kk),
                Apair.frag_at(1, kk),
                Btile.frag_at(0, kk));
          }
        }
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short mm = 0; mm < kTM; mm++) {
        if (mm * 16 < sgp_sm) {
          NAXTile<T, 1, kTK> Arow;
          Arow.load_safe(xn + mm * 16 * K, K, short2(kSK, sgp_sm - mm * 16));
          STEEL_PRAGMA_UNROLL
          for (short nn = 0; nn < kTN; nn += 2) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < kTK; kk++) {
              BaseNAXFrag::mma(
                  Dtile.frag_at(mm, nn),
                  Dtile.frag_at(mm, nn + 1),
                  Arow.frag_at(0, kk),
                  metal::bool_constant<false>{},
                  Btile.frag_at(nn, kk),
                  Btile.frag_at(nn + 1, kk),
                  metal::bool_constant<true>{});
            }
          }
        }
      }
    }
  }
}

template <
    typename T,
    typename U,
    int K,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool ALIGN_N,
    int SCHED,
    int GX,
    int SPLIT_K,
    int K_PART,
    typename F>
METAL_FUNC void dense_nax(
    const device T* x,
    const F w,
    device U* y,
    const int rows,
    const int n,
    threadgroup T* Ws,
    const uint3 group,
    const uint sgid,
    const uint lane) {
  if constexpr (omlx_tile_face<F, K>::value) {
    constexpr int kSK = 32;
    constexpr int kTGP = WM * WN * 32;
    constexpr int kSM = BM / WM;
    constexpr int kSN = BN / WN;
    constexpr short kTM = kSM / 16;
    constexpr short kTN = kSN / 16;
    constexpr int kBKP = BK + 16 / sizeof(T);
    constexpr int kLT = (kTGP >= BN * 4 && BK >= 128)
        ? (BN * 4)
        : ((kTGP >= BN * 2) ? (BN * 2) : BN);
    constexpr bool ALIGN_K = (K % BK == 0);
    static_assert(K > 0 && K % kSK == 0, "nax dense: K is whole 32-deep steps");
    static_assert(
        BM % (16 * WM) == 0 && BN % (16 * WN) == 0 && kTM > 0 && kTN > 0,
        "nax dense: whole 16x16 warp tiles");
    static_assert(
        kTN % 2 == 0 || (kTN == 1 && kTM % 2 == 0),
        "nax dense: tile_matmad_nax TN/TM parity");
    static_assert(
        SCHED == 0 || (ALIGN_N && ALIGN_K && kTN % 2 == 0),
        "nax db requires aligned N, K and even TN");
    static_assert(
        kLT <= kTGP,
        "nax dense: loader threads exceed threadgroup size");

    int k_start = 0;
    int kb_start = 0;
    int kb_end = K / BK;
    bool has_tail = !ALIGN_K;
    if constexpr (SPLIT_K > 1) {
      static_assert(
          K_PART > 0 && K_PART % BK == 0,
          "nax dense split_k: K_PART is whole BK tiles");
      const int part = int(group.z);
      k_start = part * K_PART;
      if (k_start >= K) {
        return;
      }
      const int k_end = min(K, k_start + K_PART);
      kb_start = k_start / BK;
      kb_end = k_end / BK;
      has_tail = (!ALIGN_K) && (k_end == K);
      y += size_t(part) * size_t(rows) * size_t(n);
    }

    uint tile_m = group.y;
    uint tile_n = group.x;
    const uint n_rows = (uint(rows) + uint(BM) - 1u) / uint(BM);
    if constexpr (GX > 0) {
      const uint n_cols = (uint(n) + uint(BN) - 1u) / uint(BN);
      tile_m = (group.y / n_cols) * uint(GX) + group.x;
      tile_n = group.y % n_cols;
      if (tile_m >= n_rows) {
        return;
      }
    }
    const int row0 = int(tile_m) * BM;
    const int col0 = int(tile_n) * BN;
    const int tile_rows = min(BM, rows - row0);
    const short tgp_bn = ALIGN_N ? short(BN) : short(min(BN, n - col0));

    const uint lid = sgid * 32u + lane;
    NaxWeightLoader<T, F, K, BN, BK, kLT, ALIGN_N> loader(
        w, size_t(col0), tgp_bn, lid);
    const bool loads = (kLT == kTGP) || (lid < uint(kLT));
    const bool row_live = ALIGN_N || (loader.row < tgp_bn);

    const short tm = short(kSM * int(sgid / uint(WN)));
    const short tn = short(kSN * int(sgid % uint(WN)));
    const short sgp_sm = short(min(int(kSM), max(0, tile_rows - int(tm))));
    const short sgp_sn =
        ALIGN_N ? short(kSN) : short(min(int(kSN), max(0, n - (col0 + int(tn)))));
    const bool sg_active = (sgp_sm > 0) && (ALIGN_N || sgp_sn > 0);

    if constexpr (SCHED == 1) {
      constexpr int kTile = BN * kBKP;
      const int k_iters = kb_end - kb_start;
      const int m0 = int(tm);
      const int rows_sg = min(int(kSM), tile_rows - m0);
      const device T* xs =
          x + size_t(row0 + max(0, min(m0, tile_rows - 1))) * K + size_t(k_start);
      const short2 sc = BaseNAXFrag::get_coord();
      int x_off[kTM][2];
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kTM; i++) {
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; h++) {
          const int r = min(int(i * 16 + sc.y + h * 8), max(rows_sg, 1) - 1);
          x_off[i][h] = r * K + sc.x;
        }
      }
      const short m_frags = rows_sg > 0 ? short((rows_sg + 15) / 16) : short(0);
      const threadgroup T* wsg = Ws + int(tn) * kBKP;
      NAXTile<float, kTM, kTN> D;
      D.clear();
      if (loads && k_iters > 0) {
        loader.fetch(kb_start);
        loader.store(Ws);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int kb = 0; kb < k_iters; kb++) {
        const bool more = (kb + 1) < k_iters;
        threadgroup T* next = Ws + ((kb + 1) & 1) * kTile;
        if (more && loads) {
          loader.fetch(kb_start + kb + 1);
        }
        const threadgroup T* wb = wsg + (kb & 1) * kTile;
        STEEL_PRAGMA_UNROLL
        for (short kk1 = 0; kk1 < BK; kk1 += kSK) {
          NAXTile<T, kTN, 2> Btile;
          Btile.template load<T, kBKP, 1>(wb + kk1);
          const int k = kb * BK + kk1;
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < kTM; i++) {
            if (i < m_frags) {
              NAXTile<T, 1, 2> Atile;
              STEEL_PRAGMA_UNROLL
              for (short h = 0; h < 2; h++) {
                const device T* xp = xs + x_off[i][h] + k;
                const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
                const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
                STEEL_PRAGMA_UNROLL
                for (short c = 0; c < 4; c++) {
                  Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
                  Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
                }
              }
              STEEL_PRAGMA_UNROLL
              for (short kk = 0; kk < 2; kk++) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kTN; j += 2) {
                  BaseNAXFrag::mma(
                      D.frag_at(i, j),
                      D.frag_at(i, j + 1),
                      Atile.frag_at(0, kk),
                      metal::bool_constant<false>{},
                      Btile.frag_at(j, kk),
                      Btile.frag_at(j + 1, kk),
                      metal::bool_constant<true>{});
                }
              }
            }
          }
        }
        if (more && loads) {
          loader.store(next);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      device U* yb = y + size_t(row0 + m0) * n + col0 + int(tn);
      if (rows_sg >= kSM) {
        D.store(yb, n);
      } else if (rows_sg > 0) {
        D.store_safe(yb, n, short2(kSN, short(rows_sg)));
      }
    } else {
      x += size_t(row0) * K + size_t(k_start);
      y += size_t(row0) * n + col0;
      NAXTile<float, kTM, kTN> Dtile;
      Dtile.clear();
      const device T* xn = x + int(tm) * K;
      const threadgroup T* ws = Ws + int(tn) * kBKP;
      const bool sg_full = sg_active && (sgp_sm == kSM);
      for (int kb = kb_start; kb < kb_end; kb++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (loads) {
          if (row_live) {
            loader.fetch(kb);
            loader.store(Ws);
          } else {
            loader.zero(Ws);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_full) {
          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += kSK) {
            sub_step<T, kTM, kTN, kBKP, true>(
                Dtile, xn + kk1, ws + kk1, K, sgp_sm);
          }
        } else if (sg_active) {
          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += kSK) {
            sub_step<T, kTM, kTN, kBKP, false>(
                Dtile, xn + kk1, ws + kk1, K, sgp_sm);
          }
        }
        xn += BK;
      }
      if constexpr (!ALIGN_K) {
        if (has_tail) {
          constexpr int k_rem = K - (K / BK) * BK;
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (loads) {
            if (row_live) {
              loader.fetch_tail(K / BK, k_rem);
              loader.store_words(Ws, k_rem);
            } else {
              loader.zero(Ws);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (sg_full) {
            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < k_rem; kk1 += kSK) {
              sub_step<T, kTM, kTN, kBKP, true>(
                  Dtile, xn + kk1, ws + kk1, K, sgp_sm);
            }
          } else if (sg_active) {
            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < k_rem; kk1 += kSK) {
              sub_step<T, kTM, kTN, kBKP, false>(
                  Dtile, xn + kk1, ws + kk1, K, sgp_sm);
            }
          }
        }
      }
      if (sg_full && sgp_sn == kSN) {
        Dtile.store(y + size_t(tm) * n + int(tn), n);
      } else if (sg_active) {
        Dtile.store_safe(y + size_t(tm) * n + int(tn), n, short2(sgp_sn, sgp_sm));
      }
    }
  }
}

}  // namespace omlx_nax_dense
"""

_NAX_SOURCE = """
    int rows = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
      rows *= x_shape[d];
    }
    constexpr int kBKP = BK + 16 / sizeof(T);
    threadgroup T Ws[(SCHED == 1 ? 2 : 1) * BN * kBKP + PAD / sizeof(T)];
    omlx_nax_dense::dense_nax<
        T, metal::conditional_t<(SPLIT_K > 1), float, T>,
        K, BM, BN, BK, WM, WN, ALIGN_N, SCHED, GX, SPLIT_K, K_PART>(
        x, w, y, rows, int(w_weight_shape[0]), Ws,
        threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
"""


_SKELETON_TEXT = (
    _STEEL_HEADER + "\0" + _STEEL_SOURCE + "\0" + _NAX_HEADER + "\0" + _NAX_SOURCE
)
tuning.register_skeleton("dense_tile", _SKELETON_TEXT)


class _WideBinding(NamedTuple):
    kernel: Any
    operands: tuple


class DensePlan(NamedTuple):
    """Construction-time dense prefill plan for one FormatLinear layer."""

    face: Tile
    operands: tuple
    proof: kernels.Proof
    said: tuple[str, str]
    schedule: tuple[tuple[int | None, DenseTilePolicy], ...]
    schedules: dict[mx.Dtype, tuple[tuple[int | None, DenseTilePolicy], ...]]
    k: int
    n: int
    bits: int
    group_size: int
    row_limit: int
    binding: kernels.ProofBinding


@cache
def _build(face: Tile, backend: str):
    if backend == "nax":
        header = _nax_headers()
        body_hdr, body_src, prefix = _NAX_HEADER, _NAX_SOURCE, "omlx_nax_dense"
    elif backend == "steel":
        header = _steel_headers()
        body_hdr, body_src, prefix = _STEEL_HEADER, _STEEL_SOURCE, "omlx_steel_dense"
    else:
        raise ValueError(f"unsupported dense tile backend: {backend!r}")
    if header is None:
        return None
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("w", face)]
    source = matrices.declarations + body_src
    staged = matrices.staged_bytes
    if backend == "nax" and staged:
        # Staged tables occupy threadgroup memory: subtract them from PAD so
        # the occupancy budget stays unchanged (owner rule 20).
        source = source.replace(
            "PAD / sizeof(T)];",
            f"(PAD > {staged} ? PAD - {staged} : 0) / sizeof(T)];",
        )
    return mx.fast.metal_kernel(
        name=f"{prefix}_{face.key}",
        input_names=inputs,
        output_names=["y"],
        header=header + matrices.header + body_hdr,
        source=source,
    )


def _kernel(face: Tile, policy: DenseTilePolicy):
    """Build (once) the unified steel or NAX dense kernel for ``face``."""
    with _lock:
        return _build(face, policy.backend)


def source_digest() -> str:
    """CPU-only SHA-256 digest of the unified steel and NAX dense Metal text
    over every table-free affine tile face (owner rule 20)."""
    digest = hashlib.sha256()
    count = 0
    for backend, body_hdr, body_src in (
        ("steel", _STEEL_HEADER, _STEEL_SOURCE),
        ("nax", _NAX_HEADER, _NAX_SOURCE),
    ):
        for bits in (2, 3, 4, 8):
            for gs in (32, 64, 128):
                for dtype in (mx.bfloat16, mx.float16):
                    face = affine.tile_face(bits, gs, dtype)
                    matrices = kernels.Matrices()
                    matrices.add("w", face)
                    for part in (
                        f"omlx_{backend}_dense_{face.key}",
                        matrices.header + body_hdr,
                        matrices.declarations + body_src,
                    ):
                        digest.update(part.encode() + b"\0")
                    count += 1
    return f"{count} {digest.hexdigest()}"


def _default_steel_policy(rows: int, n: int, k: int) -> DenseTilePolicy:
    """Measured M3 Ultra (applegpu_g15d) steel parity floor."""
    align_n = n % 32 == 0
    if rows <= 32:
        return DenseTilePolicy(
            "steel",
            32,
            32,
            32,
            2,
            2,
            align_n,
            x_nreads=8,
            param_staging=ParamStaging.PER_STEP,
        )
    if rows <= 48:
        return DenseTilePolicy(
            "steel",
            16,
            32,
            32,
            1,
            2,
            align_n,
            x_nreads=4,
            param_staging=ParamStaging.PER_STEP,
        )
    if n < 320:
        if n <= 32 and rows >= 1025:
            return DenseTilePolicy(
                "steel",
                16,
                32,
                32,
                1,
                2,
                align_n,
                x_nreads=4,
                param_staging=ParamStaging.PER_STEP,
            )
        return DenseTilePolicy(
            "steel",
            32,
            32,
            32,
            2,
            2,
            align_n,
            x_nreads=8,
            param_staging=ParamStaging.PER_STEP,
        )
    if n <= 512 and rows <= 512:
        return DenseTilePolicy(
            "steel",
            32,
            32,
            32,
            2,
            2,
            align_n,
            x_nreads=8,
            param_staging=ParamStaging.PER_STEP,
        )
    if rows <= 64 and n < 2560:
        return DenseTilePolicy(
            "steel",
            32,
            32,
            32,
            2,
            2,
            align_n,
            x_nreads=8,
            param_staging=ParamStaging.PER_STEP,
        )
    return DenseTilePolicy(
        "steel",
        64,
        32,
        32,
        2,
        2,
        align_n,
        x_nreads=8,
        param_staging=ParamStaging.PER_STEP,
    )


def _default_nax_policy(bits: int, rows: int, n: int, k: int) -> DenseTilePolicy:
    """Measured M5 Max (applegpu_g17s) NAX parity floor."""
    align_n = n % 64 == 0
    align_k64 = k % 64 == 0
    if rows <= 32:
        sched = "db" if (align_n and align_k64) else "seg"
        return DenseTilePolicy(
            "nax", 32, 64, 64, 2, 2, align_n, gx=0, pad_bytes=0, sched=sched
        )
    n_cols = (n + 63) // 64
    tgs_128 = ((rows + 127) // 128) * n_cols
    if k % 128 == 0 and k >= 640 and rows >= 128:
        if tgs_128 >= 160:
            return DenseTilePolicy(
                "nax",
                128,
                64,
                128,
                4,
                2,
                align_n,
                gx=32,
                pad_bytes=8192,
                sched="seg",
            )
        if rows >= 512 and tgs_128 >= 80:
            return DenseTilePolicy(
                "nax",
                128,
                64,
                128,
                4,
                2,
                align_n,
                gx=32,
                pad_bytes=0,
                sched="seg",
            )
    if k % 128 == 0 and k >= 1024 and (bits <= 4 or rows <= 128):
        return DenseTilePolicy(
            "nax", 64, 64, 128, 2, 2, align_n, gx=32, pad_bytes=0, sched="seg"
        )
    return DenseTilePolicy(
        "nax", 64, 64, 64, 2, 2, align_n, gx=0, pad_bytes=0, sched="seg"
    )


def _default_policy(
    tensor_units: bool, bits: int, rows: int, n: int, k: int
) -> DenseTilePolicy:
    """The skeleton's parity floor when the per-chip tuning table has no entry."""
    if tensor_units:
        return _default_nax_policy(bits, rows, n, k)
    return _default_steel_policy(rows, n, k)


def _rows_chunks(lo: int, hi: int | None, max_rows: int = 16) -> tuple[int, ...]:
    """Distinct chunk sizes ``_ordinary_rows`` can pass to ``_ordinary``
    for any row count in ``[lo, hi]`` (bounded by 1..max_rows)."""
    if hi is None or hi - lo + 1 >= max_rows:
        return tuple(range(1, max_rows + 1))
    chunks: set[int] = set()
    for r in range(lo, hi + 1):
        if r >= max_rows:
            chunks.add(max_rows)
        rem = r % max_rows
        if rem:
            chunks.add(rem)
    return tuple(sorted(chunks))


def _rows_bucket_ok(layer, lo: int, hi: int | None) -> bool:
    """Whether ``layer``'s ordinary rows route serves every chunk size a
    ``[lo, hi]`` bucket can issue (rule 28, fail closed)."""
    serves = getattr(layer, "_ordinary_serves", None)
    if serves is None:
        return False
    return all(serves(chunk) for chunk in _rows_chunks(lo, hi))


def _resolve_cell(
    layer,
    face: Tile,
    dtype: mx.Dtype,
    lo: int,
    hi: int | None,
    sample: int,
    *,
    tensor_units: bool,
    chip: str,
) -> DenseTilePolicy:
    k, n = layer.input_dims, layer.output_dims
    pol = tuning.lookup(
        "dense_tile",
        face,
        dtype=dtype,
        bits=layer.bits,
        group=layer.group_size,
        n=n,
        k=k,
        rows=sample,
        chip=chip,
    )
    if pol is not None and not tensor_units and pol.backend == "nax":
        pol = None
    if pol is None:
        return _default_policy(tensor_units, layer.bits, sample, n, k)
    if pol.backend == "mlx" and layer.mode != "affine":
        return _default_policy(tensor_units, layer.bits, sample, n, k)
    if pol.backend == "rows" and not _rows_bucket_ok(layer, lo, hi):
        return _default_policy(tensor_units, layer.bits, sample, n, k)
    return pol


def _build_schedule(
    layer,
    face: Tile,
    dtype: mx.Dtype,
    *,
    tensor_units: bool,
    chip: str,
) -> tuple[tuple[int | None, DenseTilePolicy], ...]:
    cuts = tuple(sorted({16, *tuning.row_bounds("dense_tile", chip=chip)}))
    schedule: list[tuple[int | None, DenseTilePolicy]] = []
    lo = 1
    for top in cuts:
        pol = _resolve_cell(
            layer, face, dtype, lo, top, top, tensor_units=tensor_units, chip=chip
        )
        if schedule and schedule[-1][1] == pol:
            schedule[-1] = (top, pol)
        else:
            schedule.append((top, pol))
        lo = top + 1
    tail_sample = max(4096, (cuts[-1] + 1) if cuts else 4096)
    tail_pol = _resolve_cell(
        layer,
        face,
        dtype,
        lo,
        None,
        tail_sample,
        tensor_units=tensor_units,
        chip=chip,
    )
    if schedule and schedule[-1][1] == tail_pol:
        schedule[-1] = (None, tail_pol)
    else:
        schedule.append((None, tail_pol))
    return tuple(schedule)


def build_plan(
    layer, *, tensor_units: bool, chip: str | None = None
) -> DensePlan | None:
    """Build a layer's construction-time dense prefill plan once."""
    k, n = layer.input_dims, layer.output_dims
    face = formats.tile(layer, k, n)
    if face is None:
        return None
    chip_id = tuning.current_chip() if chip is None else chip
    plan_dtypes = tuple(
        dt for dt in (mx.bfloat16, mx.float16) if face.exact_for(dt)
    ) or (mx.bfloat16, mx.float16)
    schedules = {
        dt: _build_schedule(layer, face, dt, tensor_units=tensor_units, chip=chip_id)
        for dt in plan_dtypes
    }
    schedule = schedules[plan_dtypes[0]]
    route = "format_linear_tile_nax" if tensor_units else "format_linear_tile_steel"
    packed = layer.fmt.codes_name
    operands = tuple(layer[name] for name in (packed, *face.tensors))
    dtypes = tuple(a.dtype for a in operands)
    proof = kernels.proof(route, face, dtypes)
    said = (
        f"quant format face {face.key}: tile kernel engaged",
        f"quant format face {face.key}: tile kernel failed; decode fallback",
    )
    return DensePlan(
        face=face,
        operands=operands,
        proof=proof,
        said=said,
        schedule=schedule,
        schedules=schedules,
        k=k,
        n=n,
        bits=layer.bits,
        group_size=layer.group_size,
        row_limit=2**31 if tensor_units else 2**30,
        binding=proof.bind(),
    )


def _policy_for_rows(
    schedule: tuple[tuple[int | None, DenseTilePolicy], ...], rows: int
) -> DenseTilePolicy:
    for hi, policy in schedule:
        if hi is None or rows <= hi:
            return policy
    return schedule[-1][1]


def policy_for(
    plan: DensePlan | None, rows: int, dtype: mx.Dtype | None = None
) -> DenseTilePolicy | None:
    """The plan's constructed policy for this many rows (rule 28)."""
    if plan is None:
        return None
    schedule = (
        plan.schedules.get(dtype, plan.schedule) if dtype is not None else plan.schedule
    )
    return _policy_for_rows(schedule, rows)


def _split_k_geometry(k: int, bk: int, requested: int) -> tuple[int, int]:
    """Effective (split_k, k_part) with k_part a multiple of bk and no empty
    trailing partitions."""
    if requested <= 1:
        return 1, k
    total_bk = (k + bk - 1) // bk
    k_part = ((total_bk + requested - 1) // requested) * bk
    return (k + k_part - 1) // k_part, k_part


def _launch(mm, policy: DenseTilePolicy, x: mx.array, operands: tuple) -> mx.array:
    """Launch the unified steel or NAX dense kernel with zero meta/params."""
    k = x.shape[-1]
    n = operands[0].shape[0]
    rows = x.size // k
    dtype = x.dtype
    out_shape = (*x.shape[:-1], n)
    bm, bn, bk = policy.bm, policy.bn, policy.bk
    wm, wn = policy.wm, policy.wn
    align_n = policy.align_n and (n % bn == 0)
    n_cols = (n + bn - 1) // bn
    n_rows = (rows + bm - 1) // bm

    if policy.backend == "nax":
        even_tn = (bn // (16 * wn)) % 2 == 0
        sched = (
            1 if (policy.sched == "db" and align_n and k % bk == 0 and even_tn) else 0
        )
        split_k, k_part = _split_k_geometry(k, bk, policy.split_k)
        gx = policy.gx
        if gx > 0:
            gx_eff = min(gx, n_rows)
            tg_x = gx_eff
            tg_y = ((n_rows + gx_eff - 1) // gx_eff) * n_cols
        else:
            tg_x = n_cols
            tg_y = n_rows
        out = mm(
            inputs=[x, *operands],
            template=[
                ("T", dtype),
                ("K", k),
                ("BM", bm),
                ("BN", bn),
                ("BK", bk),
                ("WM", wm),
                ("WN", wn),
                ("ALIGN_N", align_n),
                ("SCHED", sched),
                ("GX", gx),
                ("PAD", policy.pad_bytes),
                ("SPLIT_K", split_k),
                ("K_PART", k_part),
            ],
            grid=(tg_x * 32, tg_y * wn, wm * split_k),
            threadgroup=(32, wn, wm),
            output_shapes=[(split_k, rows, n) if split_k > 1 else out_shape],
            output_dtypes=[mx.float32 if split_k > 1 else dtype],
        )[0]
        if split_k > 1:
            return mx.sum(out, axis=0).astype(dtype).reshape(out_shape)
        return out

    tgp = wm * wn * 32
    cache_p = policy.param_staging is ParamStaging.GROUP_STEPS_COUNTER
    split_k, k_part = _split_k_geometry(k, bk, policy.split_k)
    out = mm(
        inputs=[x, *operands],
        template=[
            ("T", dtype),
            ("K", k),
            ("BM", bm),
            ("BN", bn),
            ("BK", bk),
            ("WM", wm),
            ("WN", wn),
            ("ALIGN_N", align_n),
            ("VEC_STORE", policy.vec_store),
            ("CACHE_P", cache_p),
            ("X_NREADS", policy.x_nreads),
            ("SPLIT_K", split_k),
            ("K_PART", k_part),
        ],
        grid=(tgp * n_cols, n_rows, split_k),
        threadgroup=(tgp, 1, 1),
        output_shapes=[(split_k, rows, n) if split_k > 1 else out_shape],
        output_dtypes=[mx.float32 if split_k > 1 else dtype],
    )[0]
    if split_k > 1:
        return mx.sum(out, axis=0).astype(dtype).reshape(out_shape)
    return out


def _keep_local(plan: DensePlan, sig: tuple[tuple[int, ...], mx.Dtype], graph) -> None:
    with kernels._KEEPING:
        kept = plan.binding.kept
        if plan.proof.failed or kept is None:
            return
        if len(kept.graphs) >= kernels.MAX_GRAPHS:
            kept.graphs = {}
            kept._replays = None
        elif kept._replays is not None:
            kept._replays.pop(sig, None)
        kept.graphs[sig] = graph


def _metal_key(plan, x, policy) -> tuple:
    """What one launch's Metal is built from: MLX binds an operand smaller
    than 8 elements, or a 0-d one, as a constant buffer (its address baked
    into the source), so those prove as a different Metal."""
    return (
        x.dtype,
        plan.k,
        policy,
        tuple((a.size < 8, a.ndim == 0) for a in plan.operands),
    )


def _project_cold(
    plan: DensePlan, x: mx.array, sig: tuple[tuple[int, ...], mx.Dtype]
) -> mx.array | None:
    if plan.proof.failed:
        return None
    if not plan.face.exact_for(x.dtype):
        return None
    rows = x.size // plan.k
    policy = _policy_for_rows(plan.schedules.get(x.dtype, plan.schedule), rows)

    if policy.backend == "mlx":
        bits, gs = plan.bits, plan.group_size

        def mlx_route(
            z,
            weight,
            scales,
            biases=None,
            _qmm=mx.quantized_matmul,
            _gs=gs,
            _bits=bits,
        ):
            return _qmm(z, weight, scales, biases, True, _gs, _bits, "affine")

        _keep_local(plan, sig, mlx_route)
        return mlx_route(x, *plan.operands)

    if policy.backend == "wide":
        shared_key = ("wide", x.shape, x.dtype, plan.n)
        graph = plan.proof.graphs.get(shared_key)
        if graph is not None:
            _keep_local(plan, sig, graph)
            return graph(x, *plan.operands)
        mm = tile_wide.kernel(plan.face)
        if mm is None:
            return None
        binding = _WideBinding(mm, plan.operands)
        metal, graph = tile_wide.describe(binding, x)
        if not plan.proof.prove(
            ("wide", *metal),
            lambda: tile_wide.stand_in(binding, metal),
            logger,
            *plan.said,
        ):
            return None
        compiled = mx.compile(graph)
        result = compiled(x, *plan.operands)
        plan.proof.keep(shared_key, compiled)
        _keep_local(plan, sig, compiled)
        return result

    align_n = policy.align_n and (plan.n % policy.bn == 0)
    even_tn = policy.backend == "nax" and (policy.bn // (16 * policy.wn)) % 2 == 0
    sched_eff = (
        "db"
        if (
            policy.backend == "nax"
            and policy.sched == "db"
            and align_n
            and plan.k % policy.bk == 0
            and even_tn
        )
        else "seg"
    )
    split_eff = (
        _split_k_geometry(plan.k, policy.bk, policy.split_k)[0]
        if policy.backend in ("steel", "nax")
        else 1
    )
    eff_policy = (
        policy
        if (
            align_n == policy.align_n
            and sched_eff == policy.sched
            and split_eff == policy.split_k
        )
        else DenseTilePolicy(
            policy.backend,
            policy.bm,
            policy.bn,
            policy.bk,
            policy.wm,
            policy.wn,
            align_n,
            split_eff,
            policy.x_nreads,
            policy.param_staging,
            policy.vec_store,
            policy.pad_bytes,
            policy.gx,
            sched_eff,
        )
    )
    shared_key = (x.shape, x.dtype, plan.n, eff_policy)
    graph = plan.proof.graphs.get(shared_key)
    if graph is not None:
        _keep_local(plan, sig, graph)
        return graph(x, *plan.operands)

    mm = _kernel(plan.face, eff_policy)
    if mm is None:
        return None
    metal = _metal_key(plan, x, eff_policy)

    def graph(z, *operands):
        return _launch(mm, eff_policy, z, operands)

    def stand_in():
        operands = ((a.shape, a.dtype) for a in plan.operands)
        return kernels.zeros(((16, plan.k), x.dtype), *operands), graph

    if not plan.proof.prove(metal, stand_in, logger, *plan.said):
        return None
    compiled = mx.compile(graph)
    result = compiled(x, *plan.operands)
    plan.proof.keep(shared_key, compiled)
    _keep_local(plan, sig, compiled)
    return result


def project(plan: DensePlan | None, x: mx.array) -> mx.array | None:
    """Replay this thread's compiled dense graph, proving once on cold miss."""
    if plan is None:
        return None
    sig = (x.shape, x.dtype)
    kept = plan.binding.kept
    graph = None if kept is None else kept.graphs.get(sig)
    if graph is not None:
        return graph(x, *plan.operands)
    return _project_cold(plan, x, sig)
