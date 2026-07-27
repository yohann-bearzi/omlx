// Grouped TurboQuant MoE matmul on mlx::steel tiled GEMM.
//
// Mirrors deepseek_mxfp4_gather_blocks_rhs but sources the Ws tile from
// JANGTQ codebook weights. Rows must be sorted by expert; block_meta holds
// (row_start, expert, rows) triples so each threadgroup covers one row block
// of one expert and unpacks its weight tile once.

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "turboquant_loader.h"

#ifndef SIMD_SIZE
#define SIMD_SIZE 32
#endif

template <typename T, int BM, int BN, int BK, int WM, int WN, int bits>
[[kernel]] void turboquant_gather_blocks_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device half* norms [[buffer(2)]],
    const device float* codebook [[buffer(3)]],
    const device int32_t* block_meta [[buffer(4)]],
    const device int32_t* block_count [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& max_blocks [[buffer(7)]],
    const constant int& M [[buffer(8)]],
    const constant int& N [[buffer(9)]],
    const constant int& K [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)M;
  constexpr int vals_per_u32 = 32 / bits;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = TurboQuantBlockLoader<
      T, BN, BK, BK_padded, true, WM * WN * SIMD_SIZE, bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int block_id = int(tid.y);
  const int nblocks = block_count[0];
  if (block_id >= max_blocks || block_id >= nblocks) return;

  const int y_col = int(tid.x) * BN;
  if (y_col >= N) return;

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) return;

  const short tgp_bm = short(min(BM, rows));
  const short tgp_bn = short(min(BN, N - y_col));
  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);

  const int packed_cols = K / vals_per_u32;
  const size_t stride_w = size_t(N) * packed_cols;

  const device T* xl = x + size_t(row_start) * K;
  device T* yl = y + size_t(row_start) * N + y_col;
  const device uint32_t* wl =
      w + size_t(expert) * stride_w + size_t(y_col) * packed_cols;
  const device half* nl = norms + size_t(expert) * N + y_col;

  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(
      wl, nl, codebook, packed_cols, Ws, simd_group_id, simd_lane_id);

  if (rows == BM && tgp_bn == BN) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(yl, N);
  } else if (tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(BN, tgp_bm));
  } else if (rows == BM) {
    gemm_loop_unaligned<true, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, BM));
  } else {
    gemm_loop_unaligned<false, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, tgp_bm));
  }
}

#define instantiate_tq_blocks(type, bm, bn, bk, wm, wn, bits)          \
  instantiate_kernel(                                                  \
      "turboquant_gather_blocks_rhs_" #type "_bm_" #bm "_bn_" #bn      \
      "_bk_" #bk "_wm_" #wm "_wn_" #wn "_bits_" #bits,                 \
      turboquant_gather_blocks_rhs, type, bm, bn, bk, wm, wn, bits)

instantiate_tq_blocks(float16_t, 8, 32, 32, 1, 2, 2);
instantiate_tq_blocks(float16_t, 16, 32, 32, 1, 2, 2);
instantiate_tq_blocks(float16_t, 32, 32, 32, 1, 2, 2);
instantiate_tq_blocks(float16_t, 16, 64, 32, 1, 2, 2);
instantiate_tq_blocks(float16_t, 32, 64, 32, 1, 2, 2);
instantiate_tq_blocks(float16_t, 8, 32, 32, 1, 2, 4);
instantiate_tq_blocks(float16_t, 16, 32, 32, 1, 2, 4);
instantiate_tq_blocks(float16_t, 32, 32, 32, 1, 2, 4);
instantiate_tq_blocks(float16_t, 16, 64, 32, 1, 2, 4);
instantiate_tq_blocks(float16_t, 32, 64, 32, 1, 2, 4);

instantiate_tq_blocks(bfloat16_t, 8, 32, 32, 1, 2, 2);
instantiate_tq_blocks(bfloat16_t, 16, 32, 32, 1, 2, 2);
instantiate_tq_blocks(bfloat16_t, 32, 32, 32, 1, 2, 2);
instantiate_tq_blocks(bfloat16_t, 16, 64, 32, 1, 2, 2);
instantiate_tq_blocks(bfloat16_t, 32, 64, 32, 1, 2, 2);
instantiate_tq_blocks(bfloat16_t, 8, 32, 32, 1, 2, 4);
instantiate_tq_blocks(bfloat16_t, 16, 32, 32, 1, 2, 4);
instantiate_tq_blocks(bfloat16_t, 32, 32, 32, 1, 2, 4);
instantiate_tq_blocks(bfloat16_t, 16, 64, 32, 1, 2, 4);
instantiate_tq_blocks(bfloat16_t, 32, 64, 32, 1, 2, 4);
