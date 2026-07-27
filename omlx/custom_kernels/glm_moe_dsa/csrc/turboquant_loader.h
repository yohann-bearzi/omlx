// TurboQuant block loader for mlx::steel GEMM.
//
// Mirrors MLX's QuantizedBlockLoader (fp_quantized.h) but fills the Ws tile
// from JANGTQ codebook weights instead of affine scale/bias:
//
//   w[r][c] = codebook[(packed[r][c / vals_per_u32] >> shift) & mask] * norm[r]
//
// The Hadamard rotation is applied to x outside the kernel, so it never
// enters the loader. norm is per output row (not per K-group), so next()
// -- which walks the reduction dimension -- leaves it untouched.
#pragma once

#include <metal_stdlib>

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short bits>
struct TurboQuantBlockLoader {
  static constant constexpr const short vals_per_u32 = 32 / bits;
  static constant constexpr const short BCOLS_PACKED = BCOLS / vals_per_u32;
  static constant constexpr const short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  static constant constexpr const uint32_t mask = (1u << bits) - 1u;

  const int src_ld;            // packed_cols (uint32 words per row)
  const int tile_stride;       // words advanced per reduction step
  const short thread_idx;
  const short bi;              // row within the tile
  const short bj;              // packed-word column within the tile

  threadgroup T* dst;
  const device uint32_t* src;
  const device float* codebook;
  const device half* norms;    // per output row, indexed by bi

  TurboQuantBlockLoader(
      const device uint32_t* src_,
      const device half* norms_,
      const device float* codebook_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS_PACKED : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * vals_per_u32),
        src(src_ + bi * src_ld + bj),
        codebook(codebook_),
        norms(norms_ + bi) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }
    const T nrm = static_cast<T>(*norms);
    for (int i = 0; i < n_reads; i++) {
      uint32_t word = src[i];
      for (short v = 0; v < vals_per_u32; v++) {
        dst[i * vals_per_u32 + v] =
            static_cast<T>(codebook[(word >> (v * bits)) & mask]) * nrm;
      }
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }
    const bool oob = (reduction_dim == 1 && bi >= src_tile_dim.x) ||
                     (reduction_dim == 0 && bi >= src_tile_dim.y);
    if (oob) {
      for (int i = 0; i < n_reads * vals_per_u32; i++) {
        dst[i] = T(0);
      }
      return;
    }
    const T nrm = static_cast<T>(*norms);
    for (int i = 0; i < n_reads; i++) {
      uint32_t word = src[i];
      for (short v = 0; v < vals_per_u32; v++) {
        dst[i * vals_per_u32 + v] =
            static_cast<T>(codebook[(word >> (v * bits)) & mask]) * nrm;
      }
    }
  }

  void next() {
    src += tile_stride;   // norms are per-row: unchanged along the K axis
  }
};
