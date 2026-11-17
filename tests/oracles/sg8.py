# SPDX-License-Identifier: Apache-2.0
#
# The sg8 kernels' ``128 + q`` weight operand bits and per-group input sum
# correction are adapted from Splash
# (runtime/metal/kernels/decode/linear_q4_sgmatrix.metal, Apache-2.0,
# https://github.com/incoai/splash).
# Verbatim: the live code's names (K, M, N) are kept.
# ruff: noqa: N803, N806
"""Frozen oracle of the legacy ``sg8`` kernels of
``omlx.patches.qwen35_verify_qmm`` (single, gate/up swiglu, grouped, and the
per-64 input sums they can read): their Python and Metal copied verbatim, so
their bits stay checkable after the kernels move onto faces. The oracle
keeps a kernel cache of its own; its kernels' names are the live ones' (MLX
tells two kernels of one name apart by their source: measured, MLX 0.32.4).
``tests/test_oracle_sg8.py`` proves the copy verbatim while the live kernels
exist, and compares the two bit for bit.
"""

from __future__ import annotations

_KERNEL_CACHE: dict = {}


# One simdgroup computes C^T = W X^T for CT x 8 columns and all rows. A lane
# (fm, fn) owns W[n0 + fm][k + fn, k + fn + 1] (one byte of a 4-bit word), so
# the MLX layout feeds the matrix operand directly. The operand bits encode
# 128 + q exactly; each group then folds
# s * (acc - 128 * sum(x)) + b * sum(x) into the fp32 output. The two-matrix
# form reads gate and up weights in one pass and writes silu(gate) * up.


def _operand_bits(pair: str, half: bool) -> str:
    """Bits of 128 + q for a code pair holding one code per 16 bits.

    bf16 is 0x4300 | q. fp16 has a 10-bit mantissa, so at exponent 7 one code
    step is 8 ulp: 0x5800 | q << 3.
    """
    if half:
        return f"(({pair}) << 3) | 0x58005800u"
    return f"{pair} | 0x43004300u"


def _sg8_source(mats: int, half: bool = False) -> str:
    pair4 = _operand_bits("((wds[0] >> (4 * j)) & 0x000F000Fu)", half)
    pair_split = _operand_bits(
        "((wds[0] >> (BITS * j)) & MASK) | (((wds[1] >> (BITS * j)) & MASK) << 16)",
        half,
    )
    names = [("w_q", "scales", "biases"), ("w_q2", "scales2", "biases2")][:mats]
    decl = "\n".join(
        f"    float2 out{m}[CT];\n"
        f"    for (int c = 0; c < CT; ++c) out{m}[c] = float2(0.0f);"
        for m in range(mats)
    )
    acc_decl = "\n".join(
        f"        simdgroup_matrix<float, 8, 8> acc{m}[CT];\n"
        f"        for (int c = 0; c < CT; ++c) acc{m}[c] = "
        f"simdgroup_matrix<float, 8, 8>(0.0f);"
        for m in range(mats)
    )
    mma = "\n".join(f"""                {{
                    const device uchar* wg =
                        (const device uchar*){w} + long(n) * ROW_BYTES + long(g) * WB;
                    uint wds[2];
                    load_words(wg, qd, wds);
                    for (int j = 0; j < 4; ++j) {{
                        simdgroup_matrix<T, 8, 8> a, b;
                        uint pair = operand_pair(wds, j);
                        a.thread_elements()[0] = as_type<T>(ushort(pair));
                        a.thread_elements()[1] = as_type<T>(ushort(pair >> 16));
                        b.thread_elements()[0] = xa[j];
                        b.thread_elements()[1] = xb[j];
                        simdgroup_multiply_accumulate(acc{m}[c], a, b, acc{m}[c]);
                    }}
                }}""" for m, (w, _, _) in enumerate(names))
    fold = "\n".join(f"""            {{
                float s = float({s}[n * GROUPS + g]);
                float bias = float({b}[n * GROUPS + g]);
                float2 a2 = float2(acc{m}[c].thread_elements()[0], acc{m}[c].thread_elements()[1]);
                out{m}[c] += s * a2 + (bias - 128.0f * s) * xsum;
            }}""" for m, (_, s, b) in enumerate(names))
    parts = "\n".join(
        f"    for (int c = 0; c < CT; ++c) part[(({m} * CT + c) * NSG + sg) * 32 + lane] = out{m}[c];"
        for m in range(mats)
    )
    if mats == 1:
        epilogue = """
            float2 t = sum_parts(part, 0, c, lane);
            float2 v = t;"""
    else:
        # Round gate and up to T like the separate projections, then swiglu.
        epilogue = """
            float2 g2 = float2(T(sum_parts(part, 0, c, lane).x), T(sum_parts(part, 0, c, lane).y));
            float2 u2 = float2(T(sum_parts(part, 1, c, lane).x), T(sum_parts(part, 1, c, lane).y));
            float2 v = g2 / (1.0f + exp(-g2)) * u2;"""
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;
    uint qid = lane >> 2;
    uint fm = (qid & 4) | ((lane >> 1) & 3);
    uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
    constexpr int GROUPS = K / GS;
    constexpr int PER = GROUPS / NSG;
    constexpr int WB = GS * BITS / 8;
    constexpr long ROW_BYTES = long(K) * BITS / 8;
    int g_begin = int(sg) * PER;
    int nbase = int(threadgroup_position_in_grid.y) * 8 * CT;
    bool r0 = int(fn) < M;
    bool r1 = int(fn) + 1 < M;
    // Step j of a 32-wide k block: A columns 2i and 2i + 1 are values j and
    // j + 4 of the lane's eight packed values i * 8 .. i * 8 + 7, so both
    // operands come from one shift and mask. Its x values over j are adjacent.
    uint kx = (fm >> 1) * 8 + (fm & 1u) * 4;
    // Lanes past the last row read row M - 1 and never store, so the hot
    // loop has no row branches.
    const device T* x0 = x + min(int(fn), M - 1) * K + kx;
    const device T* x1 = x + min(int(fn) + 1, M - 1) * K + kx;
    auto load_words = [&](const device uchar* wg, int qd, thread uint* wds) {{
        if (BITS == 4) {{
            wds[0] = ((const device uint*)(wg + qd * 16))[fn >> 1];
        }} else {{
            // Pack i holds eight values in BITS bytes at byte BITS * i.
            // Read the second word only when the pack reaches it.
            // The last pack of a row then stays inside the block.
            const device uint* wp = (const device uint*)(wg + qd * 4 * BITS);
            uint at = (fn >> 1) * BITS;
            ulong v = ulong(wp[at / 4]);
            if ((at % 4) * 8 + 8 * BITS > 32)
                v |= ulong(wp[at / 4 + 1]) << 32;
            v >>= (at % 4) * 8;
            wds[0] = uint(v);
            wds[1] = uint(v >> (4 * BITS));
        }}
    }};
    constexpr uint MASK = (1u << BITS) - 1u;
    auto operand_pair = [&](thread uint* wds, int j) {{
        if (BITS == 4)
            return {pair4};
        return {pair_split};
    }};
{decl}
    for (int g = g_begin; g < g_begin + PER; ++g) {{
{acc_decl}
        float2 xsum = float2(0.0f);
        for (int qd = 0; qd < GS / 32; ++qd) {{
            int kq = g * GS + qd * 32;
            T xa[4], xb[4];
            vec<T, 4> va = *((const device vec<T, 4>*)(x0 + kq));
            vec<T, 4> vb = *((const device vec<T, 4>*)(x1 + kq));
            for (int j = 0; j < 4; ++j) {{
                xa[j] = va[j];
                xb[j] = vb[j];
                xsum += float2(float(xa[j]), float(xb[j]));
            }}
            for (int c = 0; c < CT; ++c) {{
                int n = nbase + c * 8 + int(fm);
{mma}
            }}
        }}
        // Lanes sharing fn differ only in the fm bits of the lane id.
        xsum += float2(simd_shuffle_xor(xsum.x, 2), simd_shuffle_xor(xsum.y, 2));
        xsum += float2(simd_shuffle_xor(xsum.x, 4), simd_shuffle_xor(xsum.y, 4));
        xsum += float2(simd_shuffle_xor(xsum.x, 16), simd_shuffle_xor(xsum.y, 16));
        for (int c = 0; c < CT; ++c) {{
            long n = nbase + c * 8 + int(fm);
{fold}
        }}
    }}
    threadgroup float2 part[{mats} * NSG * CT * 32];
{parts}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto sum_parts = [&](threadgroup float2* p, int m, int c, uint ln) {{
        float2 t = float2(0.0f);
        for (int i = 0; i < NSG; ++i) t += p[((m * CT + c) * NSG + i) * 32 + ln];
        return t;
    }};
    if (sg == 0) {{
        for (int c = 0; c < CT; ++c) {{
            int n = nbase + c * 8 + int(fm);{epilogue}
            if (r0) y[int(fn) * N + n] = T(v.x);
            if (r1) y[(int(fn) + 1) * N + n] = T(v.y);
        }}
    }}
"""


_XSUM_ACCUM = (
    "                xsum += float2(float(xa[j]), float(xb[j]));\n",
    "                xsum += float2(float(xa), float(xb));\n",
)


_XSUM_REDUCE = (
    "        xsum += float2(simd_shuffle_xor(xsum.x, 2), simd_shuffle_xor(xsum.y, 2));\n"
    "        xsum += float2(simd_shuffle_xor(xsum.x, 4), simd_shuffle_xor(xsum.y, 4));\n"
    "        xsum += float2(simd_shuffle_xor(xsum.x, 16), simd_shuffle_xor(xsum.y, 16));\n"
)


def _with_group_sums(source: str) -> str:
    """Read the per-group input sums from ``xs`` (M, K / GS) instead."""
    for line in _XSUM_ACCUM:
        source = source.replace(line, "")
    assert source.count(_XSUM_REDUCE) == 1
    source = source.replace(_XSUM_REDUCE, "")
    return source.replace(
        "        float2 xsum = float2(0.0f);\n",
        "        float2 xsum = float2(xs[min(int(fn), M - 1) * GROUPS + g],\n"
        "                             xs[min(int(fn) + 1, M - 1) * GROUPS + g]);\n",
    )


def _build_sg8_kernel(mats: int = 1, sums: bool = False, half: bool = False):
    import mlx.core as mx

    key = ("sg8", mats, sums, half)
    if key not in _KERNEL_CACHE:
        inputs = ["x", "w_q", "scales", "biases"]
        if mats == 2:
            inputs += ["w_q2", "scales2", "biases2"]
        source = _sg8_source(mats, half)
        if sums:
            inputs.append("xs")
            source = _with_group_sums(source)
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"omlx_vk_sg8_m{mats}"
            + ("_xs" if sums else "")
            + ("_fp16" if half else ""),
            input_names=inputs,
            output_names=["y"],
            source=source,
        )
    return _KERNEL_CACHE[key]


def _sg8_geometry(K: int, N: int) -> tuple[int, int]:
    """Simdgroups per threadgroup (K split) and 8-column tiles per simdgroup.

    Measured with serialized dependent projections on M3 Ultra: vocab-size N
    fills the GPU with four simdgroups and four tiles; everything else wants
    eight K partitions, and fat-K down projections two tiles to reuse x.
    """
    if N >= 65536:
        return 4, 4
    return 8, 2 if K >= 16384 else 1


def _sg8_call(x2, mats, weights, *, group_size: int, bits: int, sums=None):
    import mlx.core as mx

    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(weights[0].shape[0])
    nsg, ct = _sg8_geometry(K, N)
    if sums is not None and int(group_size) != _SUM_GROUP:
        sums = None
    extra = [] if sums is None else [sums]
    half = x2.dtype == mx.float16
    (y,) = _build_sg8_kernel(mats, sums is not None, half)(
        inputs=[x2, *weights, *extra],
        template=[
            ("T", x2.dtype),
            ("K", K),
            ("N", N),
            ("M", M),
            ("BITS", int(bits)),
            ("GS", int(group_size)),
            ("NSG", nsg),
            ("CT", ct),
        ],
        grid=(32 * nsg, N // (8 * ct), 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x2.dtype],
    )
    return y


def vk_qmm_sg8(x2, w_q, scales, biases, *, group_size: int, bits: int = 4, sums=None):
    """(M, K) x (N, K)^T -> (M, N) for M in 7..8 through the sg8 kernel."""
    return _sg8_call(
        x2, 1, (w_q, scales, biases), group_size=group_size, bits=bits, sums=sums
    )


def vk_swiglu_sg8(x2, gate, up, sums=None):
    """silu(x gate^T) * (x up^T) for M in 7..8 in one pass over both weights."""
    return _sg8_call(
        x2,
        2,
        (gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases),
        group_size=gate.group_size,
        bits=gate.bits,
        sums=sums,
    )


_SUM_GROUP = 64


_GROUP_SUMS_SOURCE = """
    uint e = thread_position_in_grid.x;
    const device T* p = x + long(e) * 64;
    float acc = 0.0f;
    for (int i = 0; i < 64; i += 4) {
        vec<T, 4> v = *((const device vec<T, 4>*)(p + i));
        acc += float(v[0]) + float(v[1]) + float(v[2]) + float(v[3]);
    }
    xs[e] = acc;
"""


def group_sums(x):
    """Per-64 sums of a row-major ``(..., K)`` input, as ``(rows, K / 64)``."""
    import mlx.core as mx

    if "group_sums" not in _KERNEL_CACHE:
        _KERNEL_CACHE["group_sums"] = mx.fast.metal_kernel(
            name="omlx_verify_group_sums",
            input_names=["x"],
            output_names=["xs"],
            source=_GROUP_SUMS_SOURCE,
        )
    K = int(x.shape[-1])
    rows = x.size // K
    (sums,) = _KERNEL_CACHE["group_sums"](
        inputs=[x],
        template=[("T", x.dtype)],
        grid=(rows * K // 64, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, K // 64)],
        output_dtypes=[mx.float32],
    )
    return sums


def _sg8_group_source(ns: tuple, bits: tuple, half: bool = False) -> str:
    """One launch for projections sharing x; matrix i owns tiles [E_{i-1}, E_i).

    Matrices may mix 4- and 5-bit storage; the branch is uniform per
    threadgroup.
    """
    pair4 = _operand_bits("((w0 >> (4 * j)) & 0x000F000Fu)", half)
    pair5 = _operand_bits(
        "((w0 >> (5 * j)) & 0x1Fu) | (((w1 >> (5 * j)) & 0x1Fu) << 16)", half
    )
    select, end = [], 0
    for i, (n, b) in enumerate(zip(ns, bits)):
        start, end = end, end + n // 8
        head = "if" if i == 0 else "} else if"
        select.append(
            f"    {head} (tile < {end}) {{\n"
            f"        wbase = (const device uchar*)w{i};\n"
            f"        sp = s{i};\n"
            f"        bp = b{i};\n"
            f"        yp = y{i};\n"
            f"        n_out = {n};\n"
            f"        wbits = {b};\n"
            f"        n = (tile - {start}) * 8 + int(fm);"
        )
    select.append("    }")
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;
    uint qid = lane >> 2;
    uint fm = (qid & 4) | ((lane >> 1) & 3);
    uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
    constexpr int GROUPS = K / GS;
    constexpr int PER = GROUPS / NSG;
    int tile = int(threadgroup_position_in_grid.y);
    const device uchar* wbase;
    const device T* sp;
    const device T* bp;
    device T* yp;
    int n_out = 0;
    int n = 0;
    int wbits = 4;
{chr(10).join(select)}
    const int WB = GS * wbits / 8;
    const long ROW_BYTES = long(K) * wbits / 8;
    int g_begin = int(sg) * PER;
    bool r0 = int(fn) < M;
    bool r1 = int(fn) + 1 < M;
    // Same operand order as the single-matrix kernel.
    uint kx = (fm >> 1) * 8 + (fm & 1u) * 4;
    // Lanes past the last row read row M - 1 and never store, so the hot
    // loop has no row branches.
    const device T* x0 = x + min(int(fn), M - 1) * K + kx;
    const device T* x1 = x + min(int(fn) + 1, M - 1) * K + kx;
    float2 out = float2(0.0f);
    for (int g = g_begin; g < g_begin + PER; ++g) {{
        simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0.0f);
        float2 xsum = float2(0.0f);
        const device uchar* wg = wbase + long(n) * ROW_BYTES + long(g) * WB;
        for (int qd = 0; qd < GS / 32; ++qd) {{
            int kq = g * GS + qd * 32;
            uint w0, w1 = 0;
            if (wbits == 4) {{
                w0 = ((const device uint*)(wg + qd * 16))[fn >> 1];
            }} else {{
                const device uint* wp = (const device uint*)(wg + qd * 20);
                uint at = (fn >> 1) * 5;
                ulong v = (ulong(wp[at / 4 + 1]) << 32) | ulong(wp[at / 4]);
                v >>= (at % 4) * 8;
                w0 = uint(v);
                w1 = uint(v >> 20);
            }}
            vec<T, 4> va = *((const device vec<T, 4>*)(x0 + kq));
            vec<T, 4> vb = *((const device vec<T, 4>*)(x1 + kq));
            for (int j = 0; j < 4; ++j) {{
                T xa = va[j];
                T xb = vb[j];
                xsum += float2(float(xa), float(xb));
                uint pair = wbits == 4
                    ? {pair4}
                    : {pair5};
                simdgroup_matrix<T, 8, 8> a, b;
                a.thread_elements()[0] = as_type<T>(ushort(pair));
                a.thread_elements()[1] = as_type<T>(ushort(pair >> 16));
                b.thread_elements()[0] = xa;
                b.thread_elements()[1] = xb;
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }}
        }}
        xsum += float2(simd_shuffle_xor(xsum.x, 2), simd_shuffle_xor(xsum.y, 2));
        xsum += float2(simd_shuffle_xor(xsum.x, 4), simd_shuffle_xor(xsum.y, 4));
        xsum += float2(simd_shuffle_xor(xsum.x, 16), simd_shuffle_xor(xsum.y, 16));
        float s = float(sp[long(n) * GROUPS + g]);
        float bias = float(bp[long(n) * GROUPS + g]);
        float2 a2 = float2(acc.thread_elements()[0], acc.thread_elements()[1]);
        out += s * a2 + (bias - 128.0f * s) * xsum;
    }}
    threadgroup float2 part[NSG * 32];
    part[sg * 32 + lane] = out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
        float2 t = float2(0.0f);
        for (int i = 0; i < NSG; ++i) t += part[i * 32 + lane];
        if (r0) yp[int(fn) * n_out + n] = T(t.x);
        if (r1) yp[(int(fn) + 1) * n_out + n] = T(t.y);
    }}
"""


_SG8_GROUP_NSG = 8


def vk_group_sg8(x2, linears, sums=None):
    """Every projection of ``x2`` (M, K) in one launch; returns their outputs."""
    import mlx.core as mx

    ns = tuple(int(linear.scales.shape[0]) for linear in linears)
    bits = tuple(int(linear.bits) for linear in linears)
    if sums is not None and int(linears[0].group_size) != _SUM_GROUP:
        sums = None
    half = x2.dtype == mx.float16
    key = ("sg8_group", ns, bits, sums is not None, half)
    if key not in _KERNEL_CACHE:
        inputs = ["x"]
        for i in range(len(ns)):
            inputs += [f"w{i}", f"s{i}", f"b{i}"]
        source = _sg8_group_source(ns, bits, half)
        if sums is not None:
            inputs.append("xs")
            source = _with_group_sums(source)
        _KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name="omlx_vk_sg8_group_"
            + "_".join(f"{n}q{b}" for n, b in zip(ns, bits))
            + ("_xs" if sums is not None else "")
            + ("_fp16" if half else ""),
            input_names=inputs,
            output_names=[f"y{i}" for i in range(len(ns))],
            source=source,
        )
    M = int(x2.shape[0])
    K = int(x2.shape[1])
    first = linears[0]
    args = [x2]
    for linear in linears:
        args += [linear.weight, linear.scales, linear.biases]
    if sums is not None:
        args.append(sums)
    return _KERNEL_CACHE[key](
        inputs=args,
        template=[
            ("T", x2.dtype),
            ("K", K),
            ("M", M),
            ("GS", int(first.group_size)),
            ("NSG", _SG8_GROUP_NSG),
        ],
        grid=(32 * _SG8_GROUP_NSG, sum(ns) // 8, 1),
        threadgroup=(32 * _SG8_GROUP_NSG, 1, 1),
        output_shapes=[(M, n) for n in ns],
        output_dtypes=[x2.dtype] * len(ns),
    )
