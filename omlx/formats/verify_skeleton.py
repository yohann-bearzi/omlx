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

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import tuning


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
    policy: tuning.MmaTilePolicy | None = None


_REGISTER_LOCK = threading.Lock()
_SKELETON_REGISTERED = False


class _StubFace:
    key = "stub"
    type = "F"
    source = ""
    tensors = ()
    tables = ()


def _skeleton_material() -> str:
    """Everything the route's emitted code is built from, for the digest."""
    from omlx.formats.tuning.schema import MmaTilePolicy

    parts = [
        emit_mma_source(MmaTileOps(), 1, 2560),
        emit_mma_source(MmaTileOps(policy=MmaTilePolicy(32, 64, 4)), 2, 2560),
        emit_mma_source(MmaTileOps(policy=MmaTilePolicy(8, 64, 4)), 2, 2560),
    ]
    for bits in (2, 3, 4, 5, 6, 8):
        parts.append(repr(affine_mma_ops(bits, 64)))
    parts.append(repr(tile_mma_ops(_StubFace())))
    return "\n".join(parts)


def _register_tuning() -> None:
    """Register this route's skeleton text with the tuner (once, thread-safe)."""
    global _SKELETON_REGISTERED
    if _SKELETON_REGISTERED:
        return
    with _REGISTER_LOCK:
        if _SKELETON_REGISTERED:
            return
        from omlx.formats import tuning

        tuning.register_skeleton("mma_tile", _skeleton_material())
        tuning.register_skeleton("msg_tile", _msg_skeleton_material())
        _SKELETON_REGISTERED = True


def mma_tile_policy(face, *, bits, group, n, k, rows, dtype):
    """The tuned tile for this face and shape, else the parity floor."""
    from omlx.formats.tuning.schema import MmaTilePolicy

    _register_tuning()
    from omlx.formats import tuning

    return (
        tuning.lookup(
            "mma_tile",
            face,
            dtype=dtype,
            bits=bits,
            group=group,
            n=n,
            k=k,
            rows=rows,
        )
        or MmaTilePolicy()
    )


def affine_mma_ops(bits: int, group_size: int, policy=None) -> MmaTileOps:
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
        policy=policy,
        load_params="""                int n_global = n0 + dq_n;
                int k_base = k0 + dq_k * 8;
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);""",
        unpack=unpack,
    )


def tile_mma_ops(face, policy=None) -> MmaTileOps:
    """The generic Tile binding: the face's own ``Runs<K>`` cursor, its
    ``params``/``fetch``/``weight`` calls, in the same skeleton as affine.
    The face type is dependent on the kernel's T so a face outside the tile
    contract fails at Metal compilation with the contract sentence instead of phase-1 compilation.
    """
    return MmaTileOps(
        header="template <typename, typename U> struct omlx_mma_dep { using type = U; };\n",
        operands=("w",),
        policy=policy,
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


def _emit_parity(ops: MmaTileOps, row_tiles: int, k_val: int) -> str:
    """The digest-pinned (BN 16, BK 32, NSG 8) body, verbatim."""
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


def _emit_general(
    ops: MmaTileOps, row_tiles: int, k_val: int, bn: int, bk: int, nsg: int
) -> str:
    """The tuned tile: BN/8 column tiles per row, BK/8 fragments per pass."""
    from omlx.formats.tuning.schema import mma_asserts

    tiles = max(1, bn // 8)
    k_lanes = 32 // bn if 0 < bn <= 32 else 1
    packs = max(1, (bk // 8) // k_lanes)
    declares = []
    for r in range(row_tiles):
        declares.append(f"simdgroup_matrix<T, 8, 8> a{r};")
        for t in range(tiles):
            declares.append(
                f"simdgroup_matrix<float, 8, 8> c{r}_{t} ="
                " simdgroup_matrix<float, 8, 8>(0.0f);"
            )
    decl = "\n        ".join(declares)
    b_decl = ", ".join(f"b_{t}" for t in range(tiles))
    loads = "\n                ".join(
        f"simdgroup_load(a{r}, x + {r} * 8 * K + k0 + ks * BK_SUB, K);"
        for r in range(row_tiles)
    )
    b_loads = "\n                ".join(
        f"simdgroup_load(b_{t}, B_tile[sg_id] + ks * BK_SUB * BN + {8 * t}, BN);"
        for t in range(tiles)
    )
    macs = "\n                ".join(
        " ".join(
            f"simdgroup_multiply_accumulate(c{r}_{t}, a{r}, b_{t}, c{r}_{t});"
            for t in range(tiles)
        )
        for r in range(row_tiles)
    )
    stores = "\n        ".join(
        " ".join(
            f"simdgroup_store(c{r}_{t}, tg_partials[sg_id] + {r} * 8 * BN"
            f" + {8 * t}, BN);"
            for t in range(tiles)
        )
        for r in range(row_tiles)
    )
    k_consts = ops.k_consts.replace("__K__", str(int(k_val)))
    asserts = "\n".join(
        f'        static_assert({expression}, "{message}");'
        for expression, message in mma_asserts()
    )
    source = f"""
        using namespace metal;
        constexpr int BM = {8 * row_tiles};
        constexpr int BN = {bn};
        constexpr int BK = {bk};
        constexpr int BK_SUB = 8;
        constexpr int NSG = {nsg};
{k_consts}
{asserts}
        constexpr int K_chunk = NSG > 0 ? K / NSG : 0;

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

        simdgroup_matrix<T, 8, 8> {b_decl};
        {decl}

        int dq_n = int(lane) % BN;
        int dq_k_lane = int(lane) / BN;
{ops.cursor}
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            _Pragma("unroll")
            for (int pack_idx = 0; pack_idx < {packs}; ++pack_idx) {{
                int dq_k = pack_idx * {k_lanes} + dq_k_lane;
{ops.load_params}
                {ops.unpack}
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                {loads}
                {b_loads}
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


def emit_mma_source(ops: MmaTileOps, row_tiles: int, k_val: int) -> str:
    """The MMA Metal source for ``row_tiles`` 8-row tiles over ``ops``.

    The parity tile (BN 16, BK 32, NSG 8) is the digest-pinned body
    verbatim; any other policy emits the same skeleton with BN/8 column
    tiles, BK/8 fragments and its compile-time asserts."""
    policy = ops.policy
    bn, bk, nsg = (
        (policy.bn, policy.bk, policy.nsg) if policy is not None else (16, 32, 8)
    )
    if (bn, bk, nsg) == (16, 32, 8):
        return _emit_parity(ops, row_tiles, k_val)
    return _emit_general(ops, row_tiles, k_val, bn, bk, nsg)


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


# ---------------------------------------------------------------------------
# msg tile — a huge-N (vocabulary head) verify kernel over any Tile face.
#
# NSG simdgroups a threadgroup, each four rows n0..n0 + 3 of the matrix with
# the full K walked lane-strided, a run of 8 inputs a lane and step, every
# input row against each row's weights in the run's order; then simd_sum and
# one lane per result stores. Each run reads one set of group parameters and
# packed codes per output column. The load layout — all four output columns'
# packed words at once, or one column streamed before the next fetch — is a
# DATA policy (``LoadLifetime``), never a face-kind branch: every face reads
# through its own ``Runs<K>`` cursor.
# ---------------------------------------------------------------------------

MSG_NSG = 8

_MSG_CURSORS = """\
      const R r0(w, size_t(n0));
      const R r1(w, size_t(n0 + 1));
      const R r2(w, size_t(n0 + 2));
      const R r3(w, size_t(n0 + 3));"""
_MSG_FOUR_ROWS = """\
        const auto p0 = r0.params(gi);
        const auto p1 = r1.params(gi);
        const auto p2 = r2.params(gi);
        const auto p3 = r3.params(gi);
        const auto c0 = r0.fetch(run);
        const auto c1 = r1.fetch(run);
        const auto c2 = r2.fetch(run);
        const auto c3 = r3.fetch(run);
        _Pragma("unroll")
        for (int ki = 0; ki < 8; ++ki) {
          float w0 = R::weight(p0, c0, ki);
          float w1 = R::weight(p1, c1, ki);
          float w2 = R::weight(p2, c2, ki);
          float w3 = R::weight(p3, c3, ki);
__FMAS__
        }"""
_MSG_ONE_ROW = """\
        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) {
          const R rj(w, size_t(n0 + j));
          const auto p = rj.params(gi);
          const auto c = rj.fetch(run);
          _Pragma("unroll")
          for (int ki = 0; ki < 8; ++ki) {
            const float wj = R::weight(p, c, ki);
__FMAS__
          }
        }"""
_MSG_SOURCE = """
    constexpr int NSG = __NSG__;
    constexpr int MROWS = __M__;
    using F = __FACE__;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint tg_n = threadgroup_position_in_grid.y;
    int N = int(N_size);
    int n0 = (int(tg_n) * NSG + int(sg)) * 4;
    if (n0 + 3 >= N) { return; }
    if constexpr (omlx_tile_face<F, K>::value) {
      using R = typename F::template Runs<K>;
__CURSORS__
      float acc[4 * MROWS];
      _Pragma("unroll")
      for (int i = 0; i < 4 * MROWS; ++i) {
        acc[i] = 0.0f;
      }
      using Vec8 = metal::vec<T, 8>;
      const device Vec8* xv = (const device Vec8*)x;
      for (int run = int(lane); run < K / 8; run += 32) {
        const int gi = run * 8 / R::GROUP;
__XLOADS__
__DECODE__
      }
      _Pragma("unroll")
      for (int i = 0; i < 4 * MROWS; ++i) {
        acc[i] = metal::simd_sum(acc[i]);
      }
      if (lane < 4 * MROWS) {
        int j = int(lane) / MROWS;
        int row = int(lane) - j * MROWS;
        y[row * N + n0 + j] = T(acc[int(lane)]);
      }
    }
"""


@dataclass(frozen=True)
class MsgTileOps:
    """The msg skeleton's compile-time binding: a Tile face, the padded
    activation rows its kernel walks (1 to 8), the simdgroups a threadgroup,
    and the weight load layout."""

    face: object
    rows: int
    nsg: int = MSG_NSG
    load_lifetime: str = "batch"


def msg_tile_ops(
    face, rows: int, nsg: int = MSG_NSG, load_lifetime: str = "batch"
) -> MsgTileOps:
    """The msg skeleton over ``face`` for ``rows`` activation rows."""
    rows = int(rows)
    nsg = int(nsg)
    if rows < 1 or rows > 8:
        raise ValueError(f"msg tile: rows is 1 to 8, not {rows}")
    if nsg < 1 or 32 * nsg > 1024:
        raise ValueError(f"msg tile: nsg * 32 threads at most 1024, not {nsg}")
    if load_lifetime not in ("batch", "column"):
        raise ValueError(
            f"msg tile: load lifetime is batch or column, not {load_lifetime!r}"
        )
    return MsgTileOps(face, rows, nsg, load_lifetime)


def _msg_source(face_type: str, m: int, nsg: int, load_lifetime: str) -> str:
    """The msg Metal source for a face type and its load layout."""
    xloads = "".join(
        f"        Vec8 v{r} = xv[{r} * (K / 8) + run];\n" for r in range(m)
    )
    if load_lifetime == "column":
        # One output column's packed words at a time: fewer live weights
        # while the accumulator count is high (m = 6 and 8).
        cursors = ""
        decode = _MSG_ONE_ROW
        fmas = "\n".join(
            f"            acc[j * {m} + {r}] += float(v{r}[ki]) * wj;" for r in range(m)
        )
    else:
        # All four columns' packed words at once: one fetch per run feeds
        # the four columns' arithmetic.
        cursors = _MSG_CURSORS
        decode = _MSG_FOUR_ROWS
        fmas = "\n".join(
            f"          acc[{j} * {m} + {r}] += float(v{r}[ki]) * w{j};"
            for j in range(4)
            for r in range(m)
        )
    return (
        _MSG_SOURCE.replace("__NSG__", str(nsg))
        .replace("__M__", str(m))
        .replace("__FACE__", face_type)
        .replace("__CURSORS__", cursors)
        .replace("__XLOADS__\n", xloads)
        .replace("__DECODE__", decode)
        .replace("__FMAS__", fmas)
    )


def emit_msg_source(ops: MsgTileOps) -> str:
    """The msg Metal source for ``ops`` (the vk msg kernel, generic faces)."""
    return _msg_source(ops.face.type, ops.rows, ops.nsg, ops.load_lifetime)


def msg_tile_kernel(ops: MsgTileOps):
    """The msg ``mx.fast.metal_kernel`` over ``ops`` (built once per spec)."""
    from . import kernels

    return kernels.face_kernel(
        f"omlx_vk_m{ops.rows}_nsg{ops.nsg}_{ops.load_lifetime}",
        ("x", "w", "N_size"),
        {"w": ops.face},
        ("y",),
        "",
        emit_msg_source(ops),
    )


def _msg_skeleton_material() -> str:
    """The msg skeleton's digest material: its emitted sources."""
    faces = [
        f"omlx_affine_tile_b{bits}g64::Tile<{metal}>"
        for bits in (8, 4)
        for metal in ("bfloat16_t", "float16_t")
    ]
    return "\n".join(
        _msg_source(face_type, rows, MSG_NSG, load_lifetime)
        for face_type in faces
        for load_lifetime in ("batch", "column")
        for rows in (4, 6, 8)
    )


# A route module registers its skeleton with the tuner when it loads (as
# tile_dense does for dense_tile): a lookup for mma_tile/msg_tile refuses a
# table whose digest it does not know, so this must run before the first plan.
_register_tuning()
