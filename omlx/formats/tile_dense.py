# SPDX-License-Identifier: Apache-2.0
"""Unified parameterized direct dense prefill skeleton over a format's tile face.

One steel emitter (Apple GPU without tensor units, e.g. M3 Ultra) and one NAX
emitter (Apple GPU with tensor units, e.g. M5 Max), together with the wide and
verify MMA/MSG routes, serve every dense matrix for rows >= 2. There is no MoE
schedule, no expert offsets buffer, and no per-call policy lookup: ``build_plan``
resolves the layer's per-chip schedule once from ``omlx.formats.tuning`` at
construction time.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from functools import cache, partial
from typing import Any, NamedTuple

import mlx.core as mx

from .. import formats
from ..patches.m5_gather_qmm_nax import _read_mlx_headers
from . import Tile, affine, kernels, tile_wide, tuning, verify_skeleton
from .tuning import DenseTilePolicy, MmaTilePolicy, MsgTilePolicy, ParamStaging

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

#ifdef OMLX_INTERLEAVED_WEIGHTS
// Group the eight-code runs of 16 output channels contiguously for NAX B reads.
template <int BK>
METAL_FUNC int interleaved_offset(int row, int k) {
  return (row / 16) * (16 * BK) + (k / 8) * (16 * 8) + (row % 16) * 8 + k % 8;
}
#endif

template <
    typename T,
    typename F,
    int K,
    int BN,
    int BK,
    int LT,
    bool ALIGN_N,
    int STRIDE = BK + 16 / sizeof(T)>
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
  static constant constexpr int kBKP = STRIDE;
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
#ifdef OMLX_INTERLEAVED_WEIGHTS
    *(threadgroup vec<T, R::RUN>*)(dst + interleaved_offset<BK>(row, col + r * R::RUN)) = v;
#else
    *(threadgroup vec<T, R::RUN>*)(dst + r * R::RUN) = v;
#endif
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
#ifdef OMLX_INTERLEAVED_WEIGHTS
    store_(omlx_iota<kRuns>(), Ws, k_valid);
#else
    store_(omlx_iota<kRuns>(), Ws + row * kBKP + col, k_valid);
#endif
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

// Contiguous MLX views may begin at any element offset. A vector-pointer
// cast promises vector alignment to Metal, even when the buffer is a view.
template <typename T, ushort N>
METAL_FUNC vec<T, N> load_activation(const device T* p) {
  vec<T, N> value;
  __builtin_memcpy(&value, p, sizeof(value));
  return value;
}

// An eight-code run fills one lane of a 16 x BN x 32 cooperative operation.
// All lanes apply the same K permutation to both operands. Eight SIMD groups
// cover disjoint K blocks and reduce FP32 fragments before the output cast.
template <typename T, int K, int BM, int BN, int BK, bool COMPENSATE, typename F>
METAL_FUNC void register_tile(
    const F w, const device T* x, device T* y, int rows, int n,
    threadgroup float4* partials, uint3 group, uint sg, uint lane) {
  using R = typename F::template Runs<K>;
  constexpr int tiles = BN / 16, WK = 8, values = BK / 4;
  static_assert(BM == 8 || BM == 16, "one row fragment");
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, BN, 32, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto a = op.template get_left_input_cooperative_tensor<T, T, float>();
  auto b = op.template get_right_input_cooperative_tensor<T, T, float>();
  auto residual = op.template get_right_input_cooperative_tensor<T, T, float>();
  auto acc = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(a)>, metal::remove_addrspace_t<decltype(b)>, float>();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 8 * tiles; ++i) acc[i] = 0.f;

  const short2 sc = BaseNAXFrag::get_coord();
  const int row0 = group.y * BM, col0 = group.x * BN;
  const int live = min(BM, rows - row0);
  const int lane_offset = (sc.x / 4) * values;
  for (int kb = int(sg) * BK; kb < K; kb += WK * BK) {
    const int k = kb + lane_offset;
    typename R::Codes codes[2 * tiles][values / R::RUN];
    constexpr short groups = (values + R::GROUP - 1) / R::GROUP;
    typename R::P params[2 * tiles][groups];
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2 * tiles; ++h) {
      const R reader(w, size_t(col0 + 8 * h + sc.y));
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < values / R::RUN; ++i)
        codes[h][i] = reader.fetch(k / R::RUN + i);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < groups; ++i)
        params[h][i] = reader.params(k / R::GROUP + i);
    }
    STEEL_PRAGMA_UNROLL
    for (short run = 0; run < values / R::RUN; ++run) {
      STEEL_PRAGMA_UNROLL
      for (short h = 0; h < 2; ++h) {
        const int row = 8 * h + sc.y;
        const vec<T, 8> xv = (8 * h < BM && row < live)
            ? load_activation<T, 8>(x + size_t(row0 + row) * K + k + run * 8)
            : vec<T, 8>(0);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 8; ++i) {
          const short e = (i / 4) * 8 + h * 4 + i % 4;
          a[e] = xv[i];
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < tiles; ++j) {
            const short wr = 2 * j + h;
            const short slot = (i / 4 * tiles + j) * 8 + h * 4 + i % 4;
            const float weight = R::weight(params[wr][run * 8 / R::GROUP], codes[wr][run], i);
            const T high = T(weight);
            b[slot] = high;
            if constexpr (COMPENSATE) residual[slot] = T(weight - float(high));
          }
        }
      }
      op.run(a, b, acc);
      if constexpr (COMPENSATE) op.run(a, residual, acc);
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short h = 0; h < BM / 8; ++h) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < tiles; ++j) {
      float4 v;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 4; ++i) v[i] = acc[8 * j + 4 * h + i];
      partials[(sg * BM + h * 8 + sc.y) * (BN / 4) + j * 4 + sc.x / 4] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint v = sg * 32 + lane;
  if (v < uint(live * (BN / 4))) {
    float4 sum = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < WK; ++i) sum += partials[i * BM * (BN / 4) + v];
    *(device vec<T, 4>*)(y + size_t(row0 + v / (BN / 4)) * n + col0 + (v % (BN / 4)) * 4) =
        vec<T, 4>(sum);
  }
}

// Each SIMD group owns 16 rows and the full width of the threadgroup tile.
// Keep two generations of A in registers. Dequantize the prefetched packed
// weights into the alternate shared buffer while the current tile is in use.
template <typename T, typename U, int K, int BM, int BN, int BK, int WM,
          int STRIDE, int SPLIT_K, typename F>
METAL_FUNC void prefetch_tile(
    const F w, const device T* x, device U* y, int rows, int n,
    threadgroup T* ws, uint3 group, uint sg, uint lane) {
  constexpr int blocks = K / (SPLIT_K * BK), tile_size = BN * STRIDE;
  constexpr int threads = WM * 32;
  constexpr short fragments = BK / 16;
  const int row0 = int(group.y) * BM, col0 = int(group.x) * BN;
  const int tile_rows = min(BM, rows - row0);
  const int live = min(16, tile_rows - int(sg) * 16);
  const short2 sc = BaseNAXFrag::get_coord();
  const int start = int(group.z) * blocks;
  NaxWeightLoader<T, F, K, BN, BK, threads, true, STRIDE> reader(
      w, size_t(col0), BN, sg * 32 + lane);
  const device T* xp[2];
  STEEL_PRAGMA_UNROLL
  for (short h = 0; h < 2; ++h) {
    const int r = int(sg) * 16 + h * 8 + int(sc.y);
    const int row = min(r, tile_rows - 1);
    xp[h] = x + size_t(row0 + row) * K + start * BK + sc.x;
  }

  NAXTile<float, 1, BN / 16> d;
  d.clear();
  vec<T, 4> current[fragments][2], next[fragments][2];
  reader.fetch(start);
  reader.store(ws);
  if (live > 0) {
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < fragments; ++f) {
      STEEL_PRAGMA_UNROLL
      for (short h = 0; h < 2; ++h)
        current[f][h] = load_activation<T, 4>(xp[h] + 16 * f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_NO_UNROLL
  for (int step = 0; step < blocks; ++step) {
    const bool more = step + 1 < blocks;
    if (more) reader.fetch(start + step + 1);
    if (live > 0) {
      const int next_k = (step + int(more)) * BK;
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < fragments; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; ++h)
          next[f][h] = load_activation<T, 4>(xp[h] + next_k + 16 * f);
      }
#ifdef OMLX_INTERLEAVED_WEIGHTS
      const threadgroup T* weights = ws + (step & 1) * tile_size;
#else
      const threadgroup T* weights = ws + (step & 1) * tile_size + sc.y * STRIDE + sc.x;
#endif
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < fragments; ++f) {
        BaseNAXFrag::dtype_frag_t<T> a;
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; ++h) {
          STEEL_PRAGMA_UNROLL
          for (short c = 0; c < 4; ++c) a[h * 4 + c] = current[f][h][c];
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < BN / 16; j += 2) {
          BaseNAXFrag::dtype_frag_t<T> b[2];
          STEEL_PRAGMA_UNROLL
          for (short col = 0; col < 2; ++col) {
            STEEL_PRAGMA_UNROLL
            for (short h = 0; h < 2; ++h) {
#ifdef OMLX_INTERLEAVED_WEIGHTS
              const vec<T, 4> v = *(const threadgroup vec<T, 4>*)(
                  weights + interleaved_offset<BK>(sc.y + (j + col) * 16 + h * 8, sc.x + 16 * f));
#else
              const vec<T, 4> v = *(const threadgroup vec<T, 4>*)(
                  weights + ((j + col) * 16 + h * 8) * STRIDE + 16 * f);
#endif
              STEEL_PRAGMA_UNROLL
              for (short c = 0; c < 4; ++c) b[col][h * 4 + c] = v[c];
            }
          }
          BaseNAXFrag::mma(d.frag_at(0, j), d.frag_at(0, j + 1),
              a, metal::bool_constant<false>{}, b[0], b[1], metal::bool_constant<true>{});
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < fragments; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; ++h) current[f][h] = next[f][h];
      }
    }
    if (more) reader.store(ws + ((step + 1) & 1) * tile_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (live > 0) {
    device U* out = y + (size_t(group.z) * rows + row0 + sg * 16) * n + col0;
    d.store_rows(out, n, live);
  }
}

}  // namespace omlx_nax_dense
"""

_NAX_SOURCE = """
    int rows = 1;
    for (int d = 0; d < x_ndim - 1; ++d) {
      rows *= x_shape[d];
    }
    if constexpr (SCHED == 2) {
      threadgroup float4 partials[8 * BM * (BN / 4)];
      if constexpr (omlx_tile_face<decltype(w), K>::value) {
        // Upstream's unaligned few-row matvec decodes weights in FP32.
        // Retain the rounding residual for that input layout.
        if (BM == 8 && rows < 8 &&
            (!__builtin_is_aligned(x, 16) || !__builtin_is_aligned(w_weight, 16))) {
          omlx_nax_dense::register_tile<T, K, BM, BN, BK, true>(
              w, x, y, rows, int(w_weight_shape[0]), partials, threadgroup_position_in_grid,
              simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        } else {
          omlx_nax_dense::register_tile<T, K, BM, BN, BK, false>(
              w, x, y, rows, int(w_weight_shape[0]), partials, threadgroup_position_in_grid,
              simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        }
      }
    } else if constexpr (SCHED == 3) {
      constexpr int stride = BK + ROW_PAD;
      threadgroup T ws[2 * BN * stride + PAD / sizeof(T)];
      uint3 gid = threadgroup_position_in_grid;
      if constexpr (GX > 0) {
        const uint cols = uint(w_weight_shape[0]) / BN;
        const uint m = gid.y / cols * GX + gid.x;
        gid = uint3(gid.y % cols, m, gid.z);
        if (m * BM >= uint(rows)) return;
      }
      if constexpr (omlx_tile_face<decltype(w), K>::value) {
        omlx_nax_dense::prefetch_tile<
            T, metal::conditional_t<(SPLIT_K > 1), float, T>, K, BM, BN, BK,
            WM, stride, SPLIT_K>(w, x, y, rows, int(w_weight_shape[0]), ws,
                gid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
      }
    } else {
      constexpr int kBKP = BK + 16 / sizeof(T);
      threadgroup T Ws[(SCHED == 1 ? 2 : 1) * BN * kBKP + PAD / sizeof(T)];
      omlx_nax_dense::dense_nax<
          T, metal::conditional_t<(SPLIT_K > 1), float, T>,
          K, BM, BN, BK, WM, WN, ALIGN_N, SCHED, GX, SPLIT_K, K_PART>(
          x, w, y, rows, int(w_weight_shape[0]), Ws,
          threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
          thread_index_in_simdgroup);
    }
"""


_SKELETON_TEXT = (
    _STEEL_HEADER + "\0" + _STEEL_SOURCE + "\0" + _NAX_HEADER + "\0" + _NAX_SOURCE
)
tuning.register_skeleton("dense_tile", _SKELETON_TEXT)

_SPLITK_ACCUM_SOURCE = """
    using Idx = metal::conditional_t<IDX64, size_t, uint>;
    const Idx gid = Idx(thread_position_in_grid.x);
    const Idx stride = Idx(part_strides[0]);
    float acc = part[gid];
    int i = 1;
    #pragma unroll
    for (; i + 3 < SPLIT_K; i += 4) {
      const float a0 = part[gid + Idx(i) * stride];
      const float a1 = part[gid + Idx(i + 1) * stride];
      const float a2 = part[gid + Idx(i + 2) * stride];
      const float a3 = part[gid + Idx(i + 3) * stride];
      acc = (((acc + a0) + a1) + a2) + a3;
    }
    #pragma unroll
    for (; i < SPLIT_K; ++i) {
      acc += part[gid + Idx(i) * stride];
    }
    y[gid] = static_cast<T>(acc);
"""


class _WideBinding(NamedTuple):
    kernel: Any
    operands: tuple


class DenseRoute(NamedTuple):
    """One construction-time schedule bucket: every row count up to ``hi``
    (None = the tail) takes ``backend``'s route with ``policy`` (None for an
    unmeasured Dot fallback). Backends: dot, mlx, wide, steel, nax, mma, msg."""

    hi: int | None
    backend: str
    policy: object


class DensePlan(NamedTuple):
    """Construction-time dense prefill plan for one FormatLinear layer."""

    face: Tile
    operands: tuple
    proof: kernels.Proof
    said: tuple[str, str]
    schedule: tuple[DenseRoute, ...]
    schedules: dict[mx.Dtype, tuple[DenseRoute, ...]]
    k: int
    n: int
    bits: int
    group_size: int
    row_limit: int
    binding: kernels.ProofBinding


def _nax_body(interleaved_weights: bool):
    if interleaved_weights:
        return (
            "omlx_nax_interleaved_dense",
            "#define OMLX_INTERLEAVED_WEIGHTS 1\n" + _NAX_HEADER,
        )
    return "omlx_nax_dense", _NAX_HEADER


@cache
def _build(face: Tile, backend: str, interleaved_weights: bool = False):
    if backend == "nax":
        header = _nax_headers()
        prefix, body_hdr = _nax_body(interleaved_weights)
        body_src = _NAX_SOURCE
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
        return _build(face, policy.backend, policy.interleaved_weights)


@cache
def _build_splitk_accum():
    return mx.fast.metal_kernel(
        name="omlx_dense_splitk_accum",
        input_names=["part"],
        output_names=["y"],
        source=_SPLITK_ACCUM_SOURCE,
    )


def _splitk_accum_kernel():
    """Build (once) the fused single-pass split-K FP32 reduce + cast kernel."""
    with _lock:
        return _build_splitk_accum()


def _splitk_accum(
    part: mx.array, out_shape: tuple[int, ...], dtype: mx.Dtype
) -> mx.array:
    """Single-pass FP32 split-K accumulation in ascending partition order."""
    split_k = part.shape[0]
    total = part.size // split_k
    if total == 0:
        return mx.zeros(out_shape, dtype=dtype)
    tg = 128 if total <= 32768 else 256
    idx64 = part.size >= (1 << 32)
    return _splitk_accum_kernel()(
        inputs=[part],
        template=[("T", dtype), ("SPLIT_K", split_k), ("IDX64", idx64)],
        grid=(total, 1, 1),
        threadgroup=(min(tg, total), 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[dtype],
    )[0]


def source_digest() -> str:
    """CPU-only SHA-256 digest of the unified steel and NAX dense Metal text
    over every table-free affine tile face plus the fused split-K accumulate
    (owner rule 20)."""
    digest = hashlib.sha256()
    count = 0
    for prefix, body_hdr, body_src in (
        ("omlx_steel_dense", _STEEL_HEADER, _STEEL_SOURCE),
        (*_nax_body(False), _NAX_SOURCE),
        (*_nax_body(True), _NAX_SOURCE),
    ):
        for bits in (2, 3, 4, 8):
            for gs in (32, 64, 128):
                for dtype in (mx.bfloat16, mx.float16):
                    face = affine.tile_face(bits, gs, dtype)
                    matrices = kernels.Matrices()
                    matrices.add("w", face)
                    for part in (
                        f"{prefix}_{face.key}",
                        matrices.header + body_hdr,
                        matrices.declarations + body_src,
                    ):
                        digest.update(part.encode() + b"\0")
                    count += 1
    digest.update(b"splitk_accum\0" + _SPLITK_ACCUM_SOURCE.encode() + b"\0")
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
    if bits == 4 and n == k and n in (2048, 2560) and 4 <= rows <= 512:
        if rows <= 16:
            return DenseTilePolicy(
                "nax",
                8 if rows <= 8 else 16,
                32 if n == 2048 and rows > 8 else 16,
                128 if n == 2048 else 64,
                8,
                1,
                sched="register",
            )
        if rows <= 128:
            split = 2 if rows <= 64 else 1
            return DenseTilePolicy(
                "nax", 64, 32, 64, 4, 1, split_k=split, sched="prefetch"
            )
        if n == 2048:
            if rows <= 224:
                return DenseTilePolicy(
                    "nax",
                    32,
                    32,
                    64,
                    2,
                    1,
                    gx=32,
                    sched="prefetch",
                    row_pad=0,
                    interleaved_weights=True,
                )
            bm, bn = (128, 64) if rows <= 256 else (64, 32)
            return DenseTilePolicy(
                "nax", bm, bn, 64, bm // 16, 1, gx=32, sched="prefetch"
            )
        if rows <= 320:
            return DenseTilePolicy(
                "nax",
                64,
                64,
                64,
                4,
                1,
                sched="prefetch",
                row_pad=0,
                interleaved_weights=True,
            )
        return DenseTilePolicy(
            "nax", 64, 64, 64, 4, 1, sched="prefetch", row_pad=16, gx=32, pad_bytes=4096
        )
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


_CHIP_RE = re.compile(r"^applegpu_g\d+[a-z]?(?:_|$)")


def _chip_recognized(chip: str) -> bool:
    """Whether the row schedule knows this host's chip family."""
    return bool(_CHIP_RE.match(chip or ""))


def _wide_legal(face: Tile, k: int, chip: str) -> bool:
    """The wide route's construction-time legality (rule 25: format faces
    only, no stock-only gate)."""
    return (
        _chip_recognized(chip)
        and k > 0
        and k % 8 == 0
        and kernels.table_bytes(face) <= 32768
    )


_UNSET = object()


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
    dot_ok: bool = True,
) -> DenseRoute:
    """One schedule bucket's route, resolved from the DATA once (rule 28).

    Measured entries win across dense_tile, mma_tile, and msg_tile (ranked by
    tuning.match_rank). At the measured M5 q4 widths the register schedule
    also serves faces without a table entry. Other small buckets retain
    their affine, wide, or Dot fallback.
    """
    k, n = layer.input_dims, layer.output_dims
    bound = sample if hi is None else hi
    candidates: list[tuple[int, str, str, object]] = []
    table_face = (
        affine.tile_face(layer.bits, layer.group_size, dtype)
        if layer.mode == "oqh_affine"
        else face
    )
    pol = tuning.lookup(
        "dense_tile",
        table_face,
        dtype=dtype,
        bits=layer.bits,
        group=layer.group_size,
        n=n,
        k=k,
        rows=sample,
        chip=chip,
    )
    if (
        isinstance(pol, DenseTilePolicy)
        and not (
            pol.backend == "nax"
            and (not tensor_units or pol.wm * pol.wn * 32 < 2 * pol.bn)
        )
        and not (pol.backend == "mlx" and layer.mode != "affine")
        and not (pol.backend == "wide" and not _wide_legal(face, k, chip))
        and not (
            pol.backend == "dot" and (not dot_ok or bound > tuning.schema._DOT_MAX_ROWS)
        )
    ):
        candidates.append((2, "dense_tile", pol.backend, pol))
    for prio, route_name, backend, cls in (
        (1, "mma_tile", "mma", MmaTilePolicy),
        (0, "msg_tile", "msg", MsgTilePolicy),
    ):
        data = tuning.lookup(
            route_name,
            table_face,
            dtype=dtype,
            bits=layer.bits,
            group=layer.group_size,
            n=n,
            k=k,
            rows=sample,
            chip=chip,
        )
        if isinstance(data, cls) and data.is_legal(face, k, n, bound, chip, dtype):
            candidates.append((prio, route_name, backend, data))
    if len(candidates) == 1:
        _, _, backend, chosen = candidates[0]
        return DenseRoute(hi, backend, chosen)
    if candidates:
        _, _, backend, chosen = max(
            candidates,
            key=lambda item: (
                tuning.match_rank(
                    item[1],
                    table_face,
                    dtype=dtype,
                    bits=layer.bits,
                    group=layer.group_size,
                    n=n,
                    k=k,
                    rows=sample,
                    chip=chip,
                ),
                item[0],
            ),
        )
        return DenseRoute(hi, backend, chosen)
    if tensor_units and 4 <= lo <= bound <= 16:
        default = _default_nax_policy(layer.bits, sample, n, k)
        if default.sched == "register" and default.is_legal(
            face, k, n, bound, chip, dtype
        ):
            return DenseRoute(hi, "nax", default)
    if hi is not None and hi <= 16:
        # Rule 28 / lead H4: unmeasured below-17 rows take MLX's matmul for
        # an affine face, the format-generic wide route for a plugin when
        # legal, else the Dot plan; never the prefill default tiles.
        if layer.mode == "affine":
            return DenseRoute(hi, "mlx", DenseTilePolicy("mlx"))
        if _wide_legal(face, k, chip):
            return DenseRoute(hi, "wide", DenseTilePolicy("wide"))
        return DenseRoute(hi, "dot", None)
    default = _default_policy(tensor_units, layer.bits, sample, n, k)
    if default.is_legal(face, k, n, bound, chip, dtype, tail=True):
        return DenseRoute(hi, default.backend, default)
    if layer.mode == "affine":
        return DenseRoute(hi, "mlx", DenseTilePolicy("mlx"))
    return DenseRoute(hi, "dot", None)


def _build_schedule(
    layer,
    face: Tile,
    dtype: mx.Dtype,
    *,
    tensor_units: bool,
    chip: str,
    dot_ok: bool = True,
) -> tuple[DenseRoute, ...]:
    cuts = tuple(
        sorted(
            {
                1,
                16,
                *tuning.row_bounds("dense_tile", chip=chip),
                *tuning.row_bounds("mma_tile", chip=chip),
                *tuning.row_bounds("msg_tile", chip=chip),
            }
        )
    )
    schedule: list[DenseRoute] = [DenseRoute(1, "dot", None)]
    lo = 2
    for top in cuts:
        if top < 2:
            continue
        route = _resolve_cell(
            layer,
            face,
            dtype,
            lo,
            top,
            top,
            tensor_units=tensor_units,
            chip=chip,
            dot_ok=dot_ok,
        )
        last = schedule[-1]
        if last.backend == route.backend and last.policy == route.policy:
            schedule[-1] = DenseRoute(top, route.backend, route.policy)
        else:
            schedule.append(DenseRoute(top, route.backend, route.policy))
        lo = top + 1
    tail_sample = max(4096, (cuts[-1] + 1) if cuts else 4096)
    tail = _resolve_cell(
        layer,
        face,
        dtype,
        lo,
        None,
        tail_sample,
        tensor_units=tensor_units,
        chip=chip,
        dot_ok=dot_ok,
    )
    last = schedule[-1]
    if last.backend == tail.backend and last.policy == tail.policy:
        schedule[-1] = DenseRoute(None, tail.backend, tail.policy)
    else:
        schedule.append(DenseRoute(None, tail.backend, tail.policy))
    return tuple(schedule)


def build_plan(
    layer,
    *,
    tensor_units: bool,
    chip: str | None = None,
    dot_face: Any = _UNSET,
    _schedule_cache: dict | None = None,
) -> DensePlan | None:
    """Build a layer's construction-time dense prefill plan once."""
    k, n = layer.input_dims, layer.output_dims
    face = formats.tile(layer, k, n)
    if face is None:
        return None
    if chip == "":
        raise ValueError("tuning: a dense plan names its chip")
    chip_id = tuning.current_chip() if chip is None else chip
    if dot_face is _UNSET:
        dot_face = None if k < 8 else formats.dot(layer, k, n)
        if dot_face is not None and n % 8 and dot_face.vpt < 1:
            dot_face = None
    plan_dtypes = tuple(
        dt for dt in (mx.bfloat16, mx.float16) if face.exact_for(dt)
    ) or (mx.bfloat16, mx.float16)
    if _schedule_cache is not None:
        schedules = {}
        for dt in plan_dtypes:
            dot_ok = dot_face is not None and dot_face.exact_for(dt)
            cache_key = (
                layer.mode,
                face,
                dt,
                layer.bits,
                layer.group_size,
                k,
                n,
                tensor_units,
                chip_id,
                dot_ok,
            )
            sched = _schedule_cache.get(cache_key)
            if sched is None:
                sched = _build_schedule(
                    layer,
                    face,
                    dt,
                    tensor_units=tensor_units,
                    chip=chip_id,
                    dot_ok=dot_ok,
                )
                _schedule_cache[cache_key] = sched
            schedules[dt] = sched
    else:
        schedules = {
            dt: _build_schedule(
                layer,
                face,
                dt,
                tensor_units=tensor_units,
                chip=chip_id,
                dot_ok=dot_face is not None and dot_face.exact_for(dt),
            )
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


def _route_for_rows(schedule: tuple[DenseRoute, ...], rows: int) -> DenseRoute:
    for route in schedule:
        if route.hi is None or rows <= route.hi:
            return route
    return schedule[-1]


def route_for(
    plan: DensePlan | None, rows: int, dtype: mx.Dtype | None = None
) -> DenseRoute:
    """The plan's constructed route for this many rows (rule 28)."""
    if plan is None:
        return DenseRoute(None, "dot", None)
    schedule = (
        plan.schedules.get(dtype, plan.schedule) if dtype is not None else plan.schedule
    )
    return _route_for_rows(schedule, rows)


def policy_for(
    plan: DensePlan | None, rows: int, dtype: mx.Dtype | None = None
) -> DenseTilePolicy | MmaTilePolicy | MsgTilePolicy | None:
    """The plan's constructed policy for this many rows (rule 28)."""
    return route_for(plan, rows, dtype).policy


def _split_k_geometry(k: int, bk: int, requested: int) -> tuple[int, int]:
    """Effective (split_k, k_part) with k_part a multiple of bk and no empty
    trailing partitions."""
    if requested <= 1:
        return 1, k
    total_bk = (k + bk - 1) // bk
    k_part = ((total_bk + requested - 1) // requested) * bk
    return (k + k_part - 1) // k_part, k_part


def _nax_spec(policy, k, n, rows, dtype):
    bm, bn, bk = policy.bm, policy.bn, policy.bk
    wm, wn = policy.wm, policy.wn
    aligned = policy.align_n and n % bn == 0
    split, part = _split_k_geometry(k, bk, policy.split_k)
    sched = {"seg": 0, "db": 1, "register": 2, "prefetch": 3}[policy.sched]
    if sched == 1 and (not aligned or k % bk or (bn // (16 * wn)) % 2):
        sched = 0
    n_rows, n_cols = (rows + bm - 1) // bm, (n + bn - 1) // bn
    gx = policy.gx
    if gx:
        x = min(gx, n_rows)
        y = (n_rows + x - 1) // x * n_cols
    else:
        x, y = n_cols, n_rows
    if sched in (2, 3):
        threads = 32 * wm
        grid, tg = (x * threads, y, split), (threads, 1, 1)
    else:
        grid, tg = (x * 32, y * wn, wm * split), (32, wn, wm)
    template = [
        ("T", dtype),
        ("K", k),
        ("BM", bm),
        ("BN", bn),
        ("BK", bk),
        ("WM", wm),
        ("WN", wn),
        ("ALIGN_N", aligned),
        ("SCHED", sched),
        ("GX", gx),
        ("PAD", policy.pad_bytes),
        ("SPLIT_K", split),
        ("K_PART", part),
        ("ROW_PAD", policy.row_pad),
    ]
    return template, grid, tg


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
        template, grid, threadgroup = _nax_spec(policy, k, n, rows, dtype)
        split_k, _ = _split_k_geometry(k, bk, policy.split_k)
        out = mm(
            inputs=[x, *operands],
            template=template,
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(split_k, rows, n) if split_k > 1 else out_shape],
            output_dtypes=[mx.float32 if split_k > 1 else dtype],
        )[0]
        if split_k > 1:
            return _splitk_accum(out, out_shape, dtype)
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
        return _splitk_accum(out, out_shape, dtype)
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


def _effective_policy(policy: DenseTilePolicy, k: int, n: int) -> DenseTilePolicy:
    if policy.sched in ("prefetch", "register"):
        return policy
    align_n = policy.align_n and (n % policy.bn == 0)
    even_tn = policy.backend == "nax" and (policy.bn // (16 * policy.wn)) % 2 == 0
    sched_eff = (
        "db"
        if (
            policy.backend == "nax"
            and policy.sched == "db"
            and align_n
            and k % policy.bk == 0
            and even_tn
        )
        else "seg"
    )
    split_eff = _split_k_geometry(k, policy.bk, policy.split_k)[0]
    if (
        align_n == policy.align_n
        and sched_eff == policy.sched
        and split_eff == policy.split_k
    ):
        return policy
    return DenseTilePolicy(
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


def _stand_in_operands(
    operands: tuple, n: int, need_rows: int, *, align: int = 1, stackable: bool = False
) -> tuple[int, tuple[tuple[tuple[int, ...], mx.Dtype], ...]]:
    """Smallest safe operand (shape, dtype) specs preserving MLX input binding."""
    if not stackable or not all(a.shape and a.shape[0] == n for a in operands):
        return n, tuple((a.shape, a.dtype) for a in operands)
    min_rows = max(
        need_rows,
        *(
            (-(-8 * a.shape[0] // a.size)) if a.size >= 8 else a.shape[0]
            for a in operands
        ),
    )
    if align > 1:
        min_rows = -(-min_rows // align) * align
    stand_n = min(n, min_rows)
    return stand_n, tuple(((stand_n, *a.shape[1:]), a.dtype) for a in operands)


def _mma_msg_spec(face: Tile, backend: str, k: int, n: int, m: int, pol):
    """Build the face kernel and launch grid for an mma or msg verify route."""
    if backend == "mma":
        ops = verify_skeleton.tile_mma_ops(face, policy=pol)
        kernel = kernels.face_kernel(
            f"omlx_fl_mma_rt{m // 8}_k{k}",
            ("x", "w", "N_size"),
            {"w": face},
            ("y",),
            ops.header,
            verify_skeleton.emit_mma_source(ops, m // 8, k),
        )
        grid = (32 * pol.nsg, n // pol.bn, 1)
    else:
        lifetime = getattr(pol, "load_lifetime", None)
        lifetime = getattr(lifetime, "value", lifetime) or "batch"
        ops = verify_skeleton.msg_tile_ops(face, m, pol.nsg, lifetime)
        kernel = verify_skeleton.msg_tile_kernel(ops)
        cols = 4 * pol.nsg
        grid = (32 * pol.nsg, (n + cols - 1) // cols, 1)
    return kernel, grid


def _make_mma_msg_graph(
    m: int,
    kernel,
    grid: tuple[int, int, int],
    k: int,
    n: int,
    is_mma: bool,
    nsg: int,
):
    """Compiled graph body for mma/msg: binds only scalars and the face kernel
    so global _PROOFS never pins a layer's DensePlan or GPU weights."""

    def graph(
        z,
        *tensors,
        _m=m,
        _kernel=kernel,
        _grid=grid,
        _k=k,
        _n=n,
        _is_mma=is_mma,
        _nsg=nsg,
    ):
        zz = z.reshape(-1, _k)
        if zz.shape[0] < _m:
            pad = mx.zeros((_m - zz.shape[0], _k), dtype=z.dtype)
            zz = mx.concatenate([zz, pad], axis=0)
        n_size = mx.array(_n, dtype=mx.uint32)
        (y,) = _kernel(
            inputs=[zz, *tensors, n_size],
            # The mma skeleton bakes K into its source; the msg skeleton
            # takes it as a template value.
            template=[("T", z.dtype)] + ([] if _is_mma else [("K", _k)]),
            grid=_grid,
            threadgroup=(32 * _nsg, 1, 1),
            output_shapes=[(_m, _n)],
            output_dtypes=[z.dtype],
        )
        return y[: z.size // _k].reshape(*z.shape[:-1], _n)

    return graph


def warm_specs(
    layer,
    dtypes: tuple[mx.Dtype, ...],
    *,
    tensor_units: bool,
    stackable: bool = False,
    dot_face: Any = _UNSET,
    plan: DensePlan | None = None,
) -> tuple[dict, dict]:
    """Plan load-time ``(instances, proofs)`` for every dense schedule bucket.

    Covers wide, mma, msg, steel, and NAX routes (dot is warmed by linear's
    rows/shared planner; mlx needs no custom Metal). Returns
    ``({instance: recipe}, {instance: [(proof, metal), ...]})``. Closures
    capture only plain shapes, dtypes and ints — never ``mx.array``.
    """
    if layer.input_dims <= 0 or layer.output_dims <= 0:
        return {}, {}
    if plan is None:
        plan = build_plan(layer, tensor_units=tensor_units, dot_face=dot_face)
    if plan is None:
        return {}, {}
    instances: dict = {}
    proofs: dict = {}
    op_flags = tuple((a.size < 8, a.ndim == 0) for a in plan.operands)
    stand_rows = 16
    for dtype in dtypes:
        if not plan.face.exact_for(dtype):
            continue
        schedule = plan.schedules.get(dtype, plan.schedule)
        lo = 1
        for route in schedule:
            hi, backend, policy = route
            bucket_lo = lo
            lo = (hi + 1) if hi is not None else lo
            if backend in ("dot", "mlx"):
                continue
            if backend == "wide":
                mm_wide = tile_wide.kernel(plan.face)
                if mm_wide is None:
                    continue
                _, wide_ops = _stand_in_operands(
                    plan.operands, plan.n, min(plan.n, 8), stackable=stackable
                )
                wide_shapes = tuple(s for s, _ in wide_ops)
                hi_row = min(hi, 64) if hi is not None else max(bucket_lo + 4, 32)
                modes = {
                    tile_wide.layout(rows)
                    for rows in range(max(2, bucket_lo), hi_row + 1)
                }
                for vectors, partial_m, exact_rows in sorted(modes):
                    rows = exact_rows or (2 * vectors - 1 if partial_m else vectors)
                    tiles = (rows + vectors - 1) // vectors
                    metal = (dtype, plan.k, plan.n, vectors, partial_m, exact_rows)
                    inst = kernels.planned(
                        mm_wide,
                        (((rows, plan.k), dtype), *wide_ops),
                        [dtype],
                        template=tile_wide.template(*metal),
                    )
                    in_shapes = ((rows, plan.k), *wide_shapes)
                    out_shapes = ((rows, plan.n),)
                    grid = (32 * tiles, 2, 1)

                    def wide_recipe(
                        in_shapes=in_shapes, out_shapes=out_shapes, grid=grid
                    ):
                        return kernels.StandIn(
                            input_shapes=in_shapes,
                            output_shapes=out_shapes,
                            grid=grid,
                            threadgroup=(32, 2, 1),
                        )

                    instances.setdefault(inst, wide_recipe)
                    proofs.setdefault(inst, []).append(
                        (
                            plan.proof,
                            ("wide", *metal),
                        )
                    )
                continue
            if backend == "mma":
                pol = policy
                hi_row = hi if hi is not None else max(bucket_lo, 24)
                m_vals = sorted(
                    {
                        ((rows + 7) // 8) * 8
                        for rows in range(max(2, bucket_lo), hi_row + 1)
                    }
                )
                stand_n, op_specs = _stand_in_operands(
                    plan.operands,
                    plan.n,
                    min(plan.n, pol.bn),
                    align=pol.bn,
                    stackable=stackable,
                )
                op_shapes = tuple(s for s, _ in op_specs)
                tg = (32 * pol.nsg, 1, 1)
                for m in m_vals:
                    kernel, _ = _mma_msg_spec(plan.face, "mma", plan.k, plan.n, m, pol)
                    inst = kernels.planned(
                        kernel,
                        (((m, plan.k), dtype), *op_specs, ((), mx.uint32)),
                        [dtype],
                        template=[("T", dtype)],
                    )
                    in_shapes = ((m, plan.k), *op_shapes, ())

                    def mma_recipe(
                        in_shapes=in_shapes,
                        out_shapes=((m, stand_n),),
                        grid=tg,
                        threadgroup=tg,
                    ):
                        return kernels.StandIn(
                            input_shapes=in_shapes,
                            output_shapes=out_shapes,
                            grid=grid,
                            threadgroup=threadgroup,
                        )

                    instances.setdefault(inst, mma_recipe)
                    proofs.setdefault(inst, []).append(
                        (plan.proof, ("mma", dtype, plan.k, m, pol, op_flags))
                    )
                continue
            if backend == "msg":
                pol = policy
                hi_row = min(hi, 8) if hi is not None else 8
                m_vals = sorted(
                    {
                        4 if rows <= 4 else 6 if rows <= 6 else 8
                        for rows in range(max(2, bucket_lo), hi_row + 1)
                    }
                )
                cols = 4 * pol.nsg
                stand_n, op_specs = _stand_in_operands(
                    plan.operands,
                    plan.n,
                    min(plan.n, cols),
                    align=cols,
                    stackable=stackable,
                )
                op_shapes = tuple(s for s, _ in op_specs)
                tg = (32 * pol.nsg, 1, 1)
                for m in m_vals:
                    kernel, _ = _mma_msg_spec(plan.face, "msg", plan.k, plan.n, m, pol)
                    inst = kernels.planned(
                        kernel,
                        (((m, plan.k), dtype), *op_specs, ((), mx.uint32)),
                        [dtype],
                        template=[("T", dtype), ("K", plan.k)],
                    )
                    in_shapes = ((m, plan.k), *op_shapes, ())

                    def msg_recipe(
                        in_shapes=in_shapes,
                        out_shapes=((m, stand_n),),
                        grid=tg,
                        threadgroup=tg,
                    ):
                        return kernels.StandIn(
                            input_shapes=in_shapes,
                            output_shapes=out_shapes,
                            grid=grid,
                            threadgroup=threadgroup,
                        )

                    instances.setdefault(inst, msg_recipe)
                    proofs.setdefault(inst, []).append(
                        (plan.proof, ("msg", dtype, plan.k, m, pol, op_flags))
                    )
                continue
            if backend not in ("steel", "nax"):
                continue
            eff = _effective_policy(policy, plan.k, plan.n)
            mm = _kernel(plan.face, eff)
            if mm is None:
                continue
            bm, bn, bk, wm, wn = eff.bm, eff.bn, eff.bk, eff.wm, eff.wn
            split_k, k_part = _split_k_geometry(plan.k, bk, eff.split_k)
            stand_n, op_specs = _stand_in_operands(
                plan.operands,
                plan.n,
                min(plan.n, bn),
                align=bn if eff.align_n else 1,
                stackable=stackable,
            )
            op_shapes = tuple(s for s, _ in op_specs)
            n_cols = (stand_n + bn - 1) // bn
            n_rows = (stand_rows + bm - 1) // bm
            out_shape = (
                (split_k, stand_rows, stand_n) if split_k > 1 else (stand_rows, stand_n)
            )
            out_dtype = mx.float32 if split_k > 1 else dtype
            if eff.backend == "nax":
                template, grid, threadgroup = _nax_spec(
                    eff, plan.k, stand_n, stand_rows, dtype
                )
            else:
                tgp = wm * wn * 32
                cache_p = eff.param_staging is ParamStaging.GROUP_STEPS_COUNTER
                template = [
                    ("T", dtype),
                    ("K", plan.k),
                    ("BM", bm),
                    ("BN", bn),
                    ("BK", bk),
                    ("WM", wm),
                    ("WN", wn),
                    ("ALIGN_N", eff.align_n),
                    ("VEC_STORE", eff.vec_store),
                    ("CACHE_P", cache_p),
                    ("X_NREADS", eff.x_nreads),
                    ("SPLIT_K", split_k),
                    ("K_PART", k_part),
                ]
                grid = (tgp * n_cols, n_rows, split_k)
                threadgroup = (tgp, 1, 1)
            inst = kernels.planned(
                mm,
                (((stand_rows, plan.k), dtype), *op_specs),
                [out_dtype],
                template=template,
            )
            in_shapes = ((stand_rows, plan.k), *op_shapes)

            def dense_recipe(
                in_shapes=in_shapes,
                out_shapes=(out_shape,),
                grid=grid,
                threadgroup=threadgroup,
            ):
                return kernels.StandIn(
                    input_shapes=in_shapes,
                    output_shapes=out_shapes,
                    grid=grid,
                    threadgroup=threadgroup,
                )

            instances.setdefault(inst, dense_recipe)
            proofs.setdefault(inst, []).append(
                (plan.proof, (dtype, plan.k, eff, op_flags))
            )
            if split_k > 1:
                accum_k = _splitk_accum_kernel()
                stand_total = stand_rows * stand_n
                accum_inst = kernels.planned(
                    accum_k,
                    (((split_k, stand_rows, stand_n), mx.float32),),
                    [dtype],
                    template=[
                        ("T", dtype),
                        ("SPLIT_K", split_k),
                        ("IDX64", False),
                    ],
                )
                accum_tg = (min(128, stand_total), 1, 1)

                def accum_recipe(
                    in_shapes=((split_k, stand_rows, stand_n),),
                    out_shapes=((stand_rows, stand_n),),
                    grid=(stand_total, 1, 1),
                    threadgroup=accum_tg,
                ):
                    return kernels.StandIn(
                        input_shapes=in_shapes,
                        output_shapes=out_shapes,
                        grid=grid,
                        threadgroup=threadgroup,
                    )

                instances.setdefault(accum_inst, accum_recipe)
    return instances, proofs


def _make_mlx_route(bits: int, gs: int):
    # Keep a C-level partial over mx.quantized_matmul directly (no mx.compile
    # wrapper around a single C++ primitive, which adds ~0.3 us per launch on
    # short rows), while still replaying natively via FormatLinearFast and
    # retiring cleanly on Proof.fail().
    return partial(
        mx.quantized_matmul,
        transpose=True,
        group_size=gs,
        bits=bits,
        mode="affine",
    )


def warm_graph(
    plan: DensePlan | None, x_shape: tuple[int, ...], dtype: mx.Dtype
) -> bool:
    """Pre-compile and keep this thread's dense graph for ``(x_shape, dtype)``.

    Traces on a lazy zero array (never evaluates a GPU kernel) and inserts into
    ``plan.proof.graphs`` / ``plan.binding.kept.graphs`` ONLY when the target
    Metal is already warmed in ``plan.proof.ran`` (or the backend is ``mlx``).
    Returns True when ``sig`` is populated on this thread and ready for fast
    dispatch, False otherwise.
    """
    if plan is None or plan.proof.failed or not plan.face.exact_for(dtype):
        return False
    if len(x_shape) < 2 or x_shape[-1] != plan.k:
        return False
    rows = math.prod(x_shape[:-1])
    if rows <= 0 or rows >= plan.row_limit:
        return False
    sig = (x_shape, dtype)
    kept = plan.binding.kept
    if kept is None:
        return False
    if sig in kept.graphs:
        return True
    route = _route_for_rows(plan.schedules.get(dtype, plan.schedule), rows)
    if route.backend == "dot":
        return False

    if route.backend in ("mma", "msg"):
        pol = route.policy
        if route.backend == "mma":
            m = ((rows + 7) // 8) * 8
        else:
            m = 4 if rows <= 4 else 6 if rows <= 6 else 8
        shared_key = (route.backend, x_shape, dtype, plan.n, m, pol)
        graph = plan.proof.graphs.get(shared_key)
        if graph is not None:
            _keep_local(plan, sig, graph)
            return True
        metal = (
            route.backend,
            dtype,
            plan.k,
            m,
            pol,
            tuple((a.size < 8, a.ndim == 0) for a in plan.operands),
        )
        if metal not in plan.proof.ran:
            return False
        kernel, grid = _mma_msg_spec(plan.face, route.backend, plan.k, plan.n, m, pol)
        graph_fn = _make_mma_msg_graph(
            m, kernel, grid, plan.k, plan.n, route.backend == "mma", pol.nsg
        )
        x_dummy = mx.zeros(x_shape, dtype)
        compiled = mx.compile(graph_fn)
        compiled(x_dummy, *plan.operands)
        plan.proof.keep(shared_key, compiled)
        _keep_local(plan, sig, compiled)
        return True

    policy = route.policy

    if policy.backend == "mlx":
        shared_key = ("mlx", plan.bits, plan.group_size)
        graph = plan.proof.graphs.get(shared_key)
        if graph is None:
            graph = _make_mlx_route(plan.bits, plan.group_size)
            plan.proof.keep(shared_key, graph)
        _keep_local(plan, sig, graph)
        return True

    if policy.backend == "wide":
        shared_key = ("wide", x_shape, dtype, plan.n)
        graph = plan.proof.graphs.get(shared_key)
        if graph is not None:
            _keep_local(plan, sig, graph)
            return True
        mm = tile_wide.kernel(plan.face)
        if mm is None:
            return False
        binding = _WideBinding(mm, plan.operands)
        x_dummy = mx.zeros(x_shape, dtype)
        metal, graph = tile_wide.describe(binding, x_dummy)
        if ("wide", *metal) not in plan.proof.ran:
            return False
        compiled = mx.compile(graph)
        compiled(x_dummy, *plan.operands)
        plan.proof.keep(shared_key, compiled)
        _keep_local(plan, sig, compiled)
        return True

    if policy.backend not in ("steel", "nax"):
        return False

    eff_policy = _effective_policy(policy, plan.k, plan.n)
    shared_key = (x_shape, dtype, plan.n, eff_policy)
    graph = plan.proof.graphs.get(shared_key)
    if graph is not None:
        _keep_local(plan, sig, graph)
        return True
    mm = _kernel(plan.face, eff_policy)
    if mm is None:
        return False
    x_dummy = mx.zeros(x_shape, dtype)
    metal = _metal_key(plan, x_dummy, eff_policy)
    if metal not in plan.proof.ran:
        return False

    def graph_fn(z, *operands):
        return _launch(mm, eff_policy, z, operands)

    compiled = mx.compile(graph_fn)
    compiled(x_dummy, *plan.operands)
    plan.proof.keep(shared_key, compiled)
    _keep_local(plan, sig, compiled)
    return True


def _project_cold(
    plan: DensePlan, x: mx.array, sig: tuple[tuple[int, ...], mx.Dtype]
) -> mx.array | None:
    if plan.proof.failed:
        return None
    if not plan.face.exact_for(x.dtype):
        return None
    rows = x.size // plan.k
    route = _route_for_rows(plan.schedules.get(x.dtype, plan.schedule), rows)
    if route.backend == "dot":
        return None
    if route.backend in ("mma", "msg"):
        pol = route.policy
        if route.backend == "mma":
            m = ((rows + 7) // 8) * 8
        else:
            m = 4 if rows <= 4 else 6 if rows <= 6 else 8
        shared_key = (route.backend, x.shape, x.dtype, plan.n, m, pol)
        graph = plan.proof.graphs.get(shared_key)
        if graph is not None:
            _keep_local(plan, sig, graph)
            return graph(x, *plan.operands)
        kernel, grid = _mma_msg_spec(plan.face, route.backend, plan.k, plan.n, m, pol)
        graph = _make_mma_msg_graph(
            m, kernel, grid, plan.k, plan.n, route.backend == "mma", pol.nsg
        )

        metal = (
            route.backend,
            x.dtype,
            plan.k,
            m,
            pol,
            tuple((a.size < 8, a.ndim == 0) for a in plan.operands),
        )

        def stand_in():
            arrays = kernels.zeros(
                ((m, plan.k), x.dtype),
                *((a.shape, a.dtype) for a in plan.operands),
            )
            return arrays, graph

        if not plan.proof.prove(metal, stand_in, logger, *plan.said):
            return None
        compiled = mx.compile(graph)
        result = compiled(x, *plan.operands)
        plan.proof.keep(shared_key, compiled)
        _keep_local(plan, sig, compiled)
        return result
    policy = route.policy

    if policy.backend == "mlx":
        shared_key = ("mlx", plan.bits, plan.group_size)
        graph = plan.proof.graphs.get(shared_key)
        if graph is not None:
            _keep_local(plan, sig, graph)
            return graph(x, *plan.operands)
        mlx_route = _make_mlx_route(plan.bits, plan.group_size)
        result = mlx_route(x, *plan.operands)
        plan.proof.keep(shared_key, mlx_route)
        _keep_local(plan, sig, mlx_route)
        return result

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

    if policy.backend not in ("steel", "nax"):
        return None

    eff_policy = _effective_policy(policy, plan.k, plan.n)
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
    """Replay this thread's compiled dense graph, proving once on cold miss.

    Every scheduled dense route (including mma/msg, which inline ``N_size``
    inside their compiled graph) takes ``plan.operands`` alone."""
    if plan is None:
        return None
    sig = (x.shape, x.dtype)
    if x.size > plan.k:
        kept = plan.binding.kept
        graph = None if kept is None else kept.graphs.get(sig)
        if graph is not None:
            return graph(x, *plan.operands)
    return _project_cold(plan, x, sig)
