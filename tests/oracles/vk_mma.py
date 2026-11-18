# SPDX-License-Identifier: Apache-2.0
#
# The simdgroup-matrix tile kernel (``mma``) for 8..32 rows is adapted from
# dflash-mlx (dflash_mlx/verify_qmm.py, Copyright 2026 bstnxbt, Apache-2.0).
# Verbatim: the live code's names (K, M, N) are kept.
# ruff: noqa: N803, N806, SIM300
"""Frozen oracle of the legacy ``vk_mma`` kernel of
``omlx.patches.qwen35_verify_qmm`` (``vk_qmm_mma``: 7 to 24 rows, the
simdgroup-matrix tile kernel): its Python and Metal copied verbatim, so its
bits stay checkable after the kernel moves onto the tile face. The oracle
keeps a kernel cache of its own; its kernels' names are the live ones' (MLX
tells two kernels of one name apart by their source).
``tests/test_oracle_vk_mma.py`` proves the copy verbatim while the live
kernel exists, and compares the two bit for bit.
"""

from __future__ import annotations

_KERNEL_CACHE: dict = {}
_MMA_BN = 16
_MMA_BK = 32
_MMA_NSG = 8


def _build_mma_kernel(
    row_tiles: int, k_val: int, group_size: int, dtype, bits: int = 4
):
    """Rows padded to ``8 * row_tiles`` x 16 columns per threadgroup.

    Eight simdgroups each own one K chunk: they dequantize their BK x BN
    weight tile into a private threadgroup buffer and multiply it against
    every row tile with simdgroup matrices, so no cross-simdgroup barrier
    sits in the K loop. Partial sums are reduced once at the end. K is a
    template constant so the chunking folds into the code.
    """
    import mlx.core as mx

    key = ("mma", row_tiles, int(k_val), group_size, dtype, bits)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    if bits == 4:
        # One uint32 holds the pack's eight nibbles.
        unpack = """
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[sg_id][(dq_k * 8 + ki) * BN + dq_n] = T(float(nib) * s + b);
                }"""
    else:
        # MLX affine packs eight values into ``bits`` little-endian bytes.
        load = {
            2: "uint64_t packed = uint64_t(*(const device ushort*)wb);",
            6: "const device ushort* wh = (const device ushort*)wb;\n"
            "                uint64_t packed = uint64_t(wh[0]) | (uint64_t(wh[1]) << 16)"
            " | (uint64_t(wh[2]) << 32);",
            8: "uint2 w2 = *(const device uint2*)wb;\n"
            "                uint64_t packed = uint64_t(w2.x) | (uint64_t(w2.y) << 32);",
        }.get(
            bits,
            "uint64_t packed = 0;\n"
            '                _Pragma("unroll")\n'
            f"                for (int i = 0; i < {bits}; ++i) "
            "packed |= uint64_t(wb[i]) << (8 * i);",
        )
        unpack = f"""
                const device uchar* wb = ((const device uchar*)w_q)
                    + (long(n_global) * K_by_8 + (k_base >> 3)) * {bits};
                {load}
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = uint32_t((packed >> (ki * {bits})) & {(1 << bits) - 1}u);
                    B_tile[sg_id][(dq_k * 8 + ki) * BN + dq_n] = T(float(nib) * s + b);
                }}"""

    decl = "\n        ".join(
        f"simdgroup_matrix<T, 8, 8> a{r};"
        f" simdgroup_matrix<float, 8, 8> c{r}L = simdgroup_matrix<float, 8, 8>(0.0f);"
        f" simdgroup_matrix<float, 8, 8> c{r}R = simdgroup_matrix<float, 8, 8>(0.0f);"
        for r in range(row_tiles)
    )
    loads = "\n                ".join(
        f"simdgroup_load(a{r}, x + {r} * 8 * K + k0 + ks * BK_SUB, K);"
        for r in range(row_tiles)
    )
    macs = "\n                ".join(
        f"simdgroup_multiply_accumulate(c{r}L, a{r}, b_L, c{r}L);"
        f" simdgroup_multiply_accumulate(c{r}R, a{r}, b_R, c{r}R);"
        for r in range(row_tiles)
    )
    stores = "\n        ".join(
        f"simdgroup_store(c{r}L, tg_partials[sg_id] + {r} * 8 * BN, BN);"
        f" simdgroup_store(c{r}R, tg_partials[sg_id] + {r} * 8 * BN + 8, BN);"
        for r in range(row_tiles)
    )
    source = f"""
        using namespace metal;
        constexpr int BM = {8 * row_tiles};
        constexpr int BN = {_MMA_BN};
        constexpr int BK = {_MMA_BK};
        constexpr int BK_SUB = 8;
        constexpr int NSG = {_MMA_NSG};
        constexpr int GS = {group_size};
        constexpr int K = {int(k_val)};
        constexpr int K_by_8 = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;

        uint tid = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint lane = tid % 32;
        uint tg_n = threadgroup_position_in_grid.y;

        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        threadgroup T B_tile[NSG][BK * BN];
        threadgroup float tg_partials[NSG][BM * BN];

        simdgroup_matrix<T, 8, 8> b_L, b_R;
        {decl}

        int dq_n = int(lane) % BN;
        int dq_k_lane = int(lane) / BN;

        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            _Pragma("unroll")
            for (int pack_idx = 0; pack_idx < 2; ++pack_idx) {{
                int dq_k = pack_idx * 2 + dq_k_lane;
                int n_global = n0 + dq_n;
                int k_base = k0 + dq_k * 8;
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                {unpack}
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                {loads}
                simdgroup_load(b_L, B_tile[sg_id] + ks * BK_SUB * BN, BN);
                simdgroup_load(b_R, B_tile[sg_id] + ks * BK_SUB * BN + 8, BN);
                {macs}
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}

        {stores}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int off = int(tid); off < BM * BN; off += NSG * 32) {{
            float acc = 0.0f;
            _Pragma("unroll")
            for (int g = 0; g < NSG; ++g) {{
                acc += tg_partials[g][off];
            }}
            int row = off / BN;
            int col = off - row * BN;
            y[row * N + n0 + col] = T(acc);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"omlx_vk_mma_rt{row_tiles}_k{int(k_val)}_q{bits}_gs{group_size}_"
        f"{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _pad_rows(mx, x2, m: int):
    # metal_kernel copies non-contiguous inputs itself, so aligned rows pass
    # through without an extra dispatch.
    M = int(x2.shape[0])
    if M < m:
        pad = mx.zeros((m - M, x2.shape[1]), dtype=x2.dtype)
        return mx.concatenate([x2, pad], axis=0), M
    return x2, M


def vk_qmm_mma(x2, w_q, scales, biases, *, group_size: int, bits: int = 4):
    """(M, K) x (N, K)^T -> (M, N) for M in 7..24 through the mma tile kernel."""
    import mlx.core as mx

    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    row_tiles = (M + 7) // 8
    bm = 8 * row_tiles
    xm, M0 = _pad_rows(mx, x2, bm)
    kernel = _build_mma_kernel(row_tiles, K, group_size, x2.dtype, bits)
    (y,) = kernel(
        inputs=[xm, w_q, scales, biases, N],
        template=[("T", x2.dtype)],
        grid=(32 * _MMA_NSG, N // _MMA_BN, 1),
        threadgroup=(32 * _MMA_NSG, 1, 1),
        output_shapes=[(bm, N)],
        output_dtypes=[x2.dtype],
    )
    return y[:M0, :] if M0 < bm else y
