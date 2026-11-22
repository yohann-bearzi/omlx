# SPDX-License-Identifier: Apache-2.0
"""The verify kernels' skeletons: one parameterized Metal template per shape
family, with the face's code reading bound at compile time.

The MMA skeleton computes C = X W^T in 8-row tiles through
``simdgroup_multiply_accumulate``. An affine tile face binds its packed
byte pointers and scale/bias loads directly; a generic ``Tile`` face binds
the same skeleton through its ``Runs<K>`` cursor. Both emit the same
template, so a face swap changes the bound expressions only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MmaTileOps:
    """The compile-time face binding of the MMA skeleton."""

    header: str = ""
    operands: tuple[str, ...] = ("w_q", "scales", "biases")
    k_consts: str = ""
    cursor: str = ""
    close: str = ""
    load_params: str = ""
    unpack: str = ""


def affine_mma_ops(bits: int, group_size: int) -> MmaTileOps:
    """The affine tile binding: one packed run per lane read, a scale and a
    bias per group. 8-bit runs are two words; at 8-bit the MLX layout aligns
    every run to 8 bytes, so one uint2 load replaces two uint loads."""
    if bits == 4:
        unpack = """
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[sg_id][(dq_k * 8 + ki) * BN + dq_n] = T(float(nib) * s + b);
                }"""
    else:
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
    return MmaTileOps(
        header="",
        operands=("w_q", "scales", "biases"),
        k_consts=f"""        constexpr int GS = {group_size};
        constexpr int K = __K__;
        constexpr int K_by_8 = K / 8;
        constexpr int K_by_gs = K / GS;""",
        load_params="""                int n_global = n0 + dq_n;
                int k_base = k0 + dq_k * 8;
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);""",
        unpack=unpack,
    )


def tile_mma_ops(face) -> MmaTileOps:
    """The generic Tile binding: the face's own ``Runs<K>`` cursor, its
    ``params``/``fetch``/``weight`` calls, in the same skeleton as affine.
    The face type is dependent on the kernel's T so a face outside the tile
    contract fails at Metal compilation with the contract sentence instead of phase-1 compilation."""
    return MmaTileOps(
        header="template <typename, typename U> struct omlx_mma_dep { using type = U; };\n",
        operands=("w",),
        k_consts="        constexpr int K = __K__;",
        cursor=f"""        using F = typename omlx_mma_dep<T, {face.type}>::type;
        if constexpr (omlx_tile_face<F, K>::value) {{
        using R = typename F::template Runs<K>;
        const R runs(w, size_t(n0 + dq_n));""",
        close="\n        }",
        load_params="""                int k_base = k0 + dq_k * 8;""",
        unpack="""const auto p = runs.params(k_base / R::GROUP);
                const auto c = runs.fetch(k_base / 8);
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {
                    B_tile[sg_id][(dq_k * 8 + ki) * BN + dq_n] =
                        T(R::weight(p, c, ki));
                }""",
    )


def emit_mma_source(ops: MmaTileOps, row_tiles: int, k_val: int) -> str:
    """The MMA Metal source for ``row_tiles`` 8-row tiles over ``ops``."""
    bn, bk, nsg = 16, 32, 8
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
    k_consts = ops.k_consts.replace("__K__", str(int(k_val)))
    source = f"""
        using namespace metal;
        constexpr int BM = {8 * row_tiles};
        constexpr int BN = {bn};
        constexpr int BK = {bk};
        constexpr int BK_SUB = 8;
        constexpr int NSG = {nsg};
{k_consts}
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
{ops.cursor}
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            _Pragma("unroll")
            for (int pack_idx = 0; pack_idx < 2; ++pack_idx) {{
                int dq_k = pack_idx * 2 + dq_k_lane;
{ops.load_params}
                {ops.unpack}
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
        }}{ops.close}
    """
    return source


def emit_mma_kernel(ops: MmaTileOps, row_tiles: int, k_val: int, name: str) -> dict:
    """The MMA kernel spec for ``row_tiles`` 8-row tiles over ``ops``."""
    spec = dict(
        name=name,
        input_names=["x", *ops.operands, "N_size"],
        output_names=["y"],
        source=emit_mma_source(ops, row_tiles, k_val),
    )
    if ops.header:
        spec["header"] = ops.header
    return spec
