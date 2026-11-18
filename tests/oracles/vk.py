# SPDX-License-Identifier: Apache-2.0
#
# Adapted from MTPLX (mtplx/verify_kernels.py)
#   Copyright 2026 Youssof Altoukhi
#   Licensed under the Apache License, Version 2.0
#   https://github.com/youssofal/mtplx
#   "Powered by MTPLX by Youssof Altoukhi"
# Verbatim: the live code's names (K, M, N) are kept.
# ruff: noqa: N803, N806, SIM300
"""Frozen oracle of the legacy ``vk`` kernels of
``omlx.patches.qwen35_verify_qmm`` (``vk_qmm``: the multi-simdgroup ``msg``
kernel of huge N, the head, and the split-K kernel): their Python and Metal
copied verbatim, so their bits stay checkable after the kernels move onto
faces. The oracle keeps a kernel cache of its own; its kernels' names are
the live ones' (MLX tells two kernels of one name apart by their source).
``tests/test_oracle_vk.py`` proves the copy verbatim while the live kernels
exist, and compares the two bit for bit.
"""

from __future__ import annotations

_KERNEL_CACHE: dict = {}
_MSG_NSG = 8


def _fma_block(m: int, bits: int) -> str:
    per = 8 if bits == 4 else 4
    mask = "0xFu" if bits == 4 else "0xFFu"
    shift = 4 if bits == 4 else 8
    lines = [f"for (int ki = 0; ki < {per}; ++ki) {{"]
    for j in range(4):
        lines.append(
            f"    float w{j} = float((p{j} >> (ki * {shift})) & {mask}) * s{j} + b{j};"
        )
    for j in range(4):
        for r in range(m):
            lines.append(
                f"    acc[{j} * {m} + {r}] += "
                f"float(v{r}[ki{'' if bits == 4 else ' + koff'}]) * w{j};"
            )
    lines.append("}")
    return "\n            ".join(lines)


def _build_msg_kernel(m: int, bits: int, group_size: int, dtype, nsg: int):
    import mlx.core as mx

    key = ("msg", m, bits, group_size, dtype, nsg)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    xloads = "\n            ".join(
        f"Vec8 v{r} = xv[({r} * K + k_base) / 8];" for r in range(m)
    )
    if bits == 4:
        pack_setup = """
            int k_base = pack * 8;
            int gi = k_base / GS;
            uint32_t p0 = w_q[(n0 + 0) * K_by_p + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_p + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_p + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_p + pack];
        """
        body = f"""
        for (int pack = int(lane); pack < K_by_p; pack += 32) {{
            {pack_setup}
            {xloads}
            float s0 = float(scales[(n0 + 0) * K_by_gs + gi]);
            float s1 = float(scales[(n0 + 1) * K_by_gs + gi]);
            float s2 = float(scales[(n0 + 2) * K_by_gs + gi]);
            float s3 = float(scales[(n0 + 3) * K_by_gs + gi]);
            float b0 = float(biases[(n0 + 0) * K_by_gs + gi]);
            float b1 = float(biases[(n0 + 1) * K_by_gs + gi]);
            float b2 = float(biases[(n0 + 2) * K_by_gs + gi]);
            float b3 = float(biases[(n0 + 3) * K_by_gs + gi]);
            _Pragma("unroll")
            {_fma_block(m, 4)}
        }}
        """
    else:
        body = f"""
        for (int pair = int(lane); pair < K_by_p; pair += 32) {{
            int k_base = pair * 8;
            int gi = k_base / GS;
            {xloads}
            _Pragma("unroll")
            for (int wsel = 0; wsel < 2; ++wsel) {{
                int koff = wsel * 4;
                uint32_t p0 = w_q[(n0 + 0) * (K / 4) + pair * 2 + wsel];
                uint32_t p1 = w_q[(n0 + 1) * (K / 4) + pair * 2 + wsel];
                uint32_t p2 = w_q[(n0 + 2) * (K / 4) + pair * 2 + wsel];
                uint32_t p3 = w_q[(n0 + 3) * (K / 4) + pair * 2 + wsel];
                float s0 = float(scales[(n0 + 0) * K_by_gs + gi]);
                float s1 = float(scales[(n0 + 1) * K_by_gs + gi]);
                float s2 = float(scales[(n0 + 2) * K_by_gs + gi]);
                float s3 = float(scales[(n0 + 3) * K_by_gs + gi]);
                float b0 = float(biases[(n0 + 0) * K_by_gs + gi]);
                float b1 = float(biases[(n0 + 1) * K_by_gs + gi]);
                float b2 = float(biases[(n0 + 2) * K_by_gs + gi]);
                float b3 = float(biases[(n0 + 3) * K_by_gs + gi]);
                _Pragma("unroll")
                {_fma_block(m, 8)}
            }}
        }}
        """

    n_acc = 4 * m
    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int NSG = {nsg};
        constexpr int MROWS = {m};

        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_p = {"K / 8" if bits == 4 else "K / 8"};
        int K_by_gs = K / GS;
        int n0 = (int(tg_n) * NSG + int(sg)) * 4;
        if (n0 + 3 >= N) {{ return; }}

        float acc[{n_acc}];
        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        {body}

        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        if (lane < {n_acc}) {{
            int j = int(lane) / MROWS;
            int row = int(lane) - j * MROWS;
            y[row * N + n0 + j] = T(acc[int(lane)]);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"omlx_vk_m{m}_q{bits}_nsg{nsg}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _pack_block(m: int, bits: int, sfx: str) -> str:
    p = f"pack{sfx}"
    lines = [f"int k_base{sfx} = {p} * 8;", f"int gi{sfx} = k_base{sfx} / GS;"]
    for r in range(m):
        lines.append(f"Vec8 v{sfx}_{r} = xv[({r} * K + k_base{sfx}) / 8];")
    if bits == 4:
        for j in range(4):
            lines.append(f"uint32_t p{sfx}_{j} = w_q[(n0 + {j}) * K_by_p + {p}];")
        for j in range(4):
            lines.append(
                f"float s{sfx}_{j} = float(scales[(n0 + {j}) * K_by_gs + gi{sfx}]);"
                f" float b{sfx}_{j} = float(biases[(n0 + {j}) * K_by_gs + gi{sfx}]);"
            )
        for j in range(4):
            block = [
                "{",
                f"    uint32_t packed = p{sfx}_{j};",
                f"    float s = s{sfx}_{j};",
                f"    float b = b{sfx}_{j};",
                "    for (int ki = 0; ki < 8; ++ki) {",
                "        float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;",
            ]
            for r in range(m):
                block.append(
                    f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki]) * wv;"
                )
            block.extend(["    }", "}"])
            lines.extend(block)
    else:
        for j in range(4):
            lines.append(
                f"uint32_t pa{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 2];"
                f" uint32_t pb{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 2 + 1];"
            )
        for j in range(4):
            lines.append(
                f"float s{sfx}_{j} = float(scales[(n0 + {j}) * K_by_gs + gi{sfx}]);"
                f" float b{sfx}_{j} = float(biases[(n0 + {j}) * K_by_gs + gi{sfx}]);"
            )
        for j in range(4):
            block = [
                "{",
                f"    uint32_t pa = pa{sfx}_{j};",
                f"    uint32_t pb = pb{sfx}_{j};",
                f"    float s = s{sfx}_{j};",
                f"    float b = b{sfx}_{j};",
                "    for (int ki = 0; ki < 4; ++ki) {",
                "        float wa = float((pa >> (ki * 8)) & 0xFFu) * s + b;",
                "        float wb = float((pb >> (ki * 8)) & 0xFFu) * s + b;",
            ]
            for r in range(m):
                block.append(
                    f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki]) * wa;"
                )
                block.append(
                    f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki + 4]) * wb;"
                )
            block.extend(["    }", "}"])
            lines.extend(block)
    return "\n            ".join(lines)


def _build_ksplit_kernel(m: int, bits: int, group_size: int, dtype, *, k_parts: int):
    import mlx.core as mx

    key = ("ksplit", m, bits, group_size, dtype, k_parts)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    n_acc = 4 * m
    loop = f"""
        for (int packA = p_start + int(lane); packA < p_end; packA += 32) {{
            {_pack_block(m, bits, "A")}
        }}
    """

    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int K_PARTS = {k_parts};

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int K_by_p = K / 8;
        int K_by_w = K / 4;
        int K_by_gs = K / GS;
        int per_part = K_by_p / K_PARTS;
        int N = int(N_size);
        int n0 = int(tg_n) * 4;
        int p_start = int(part) * per_part;
        int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;

        float acc[{n_acc}];
        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        {loop}

        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        threadgroup float partials[K_PARTS * {n_acc}];
        if (lane == 0) {{
            _Pragma("unroll")
            for (int i = 0; i < {n_acc}; ++i) {{
                partials[int(part) * {n_acc} + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < {n_acc}) {{
            float total = 0.0f;
            _Pragma("unroll")
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partials[p * {n_acc} + int(lane)];
            }}
            int j = int(lane) / {m};
            int row = int(lane) - j * {m};
            y[row * N + n0 + j] = T(total);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"omlx_vk_ks_m{m}_q{bits}_kp{k_parts}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
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


def vk_qmm(x2, w_q, scales, biases, *, bits: int, group_size: int):
    """Verify-shape qmm: (M, K) x (N, K)^T -> (M, N), M in 2..6.

    Rows are padded up to the kernel's M template. split-K for regular
    shapes, msg tile for huge N (lm_head).
    """
    import mlx.core as mx

    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    m = 4 if M <= 4 else 6

    if N >= 100000 and N % (4 * _MSG_NSG) == 0:
        xm, M0 = _pad_rows(mx, x2, m)
        kernel = _build_msg_kernel(m, bits, group_size, x2.dtype, _MSG_NSG)
        cols = 4 * _MSG_NSG
        (y,) = kernel(
            inputs=[xm, w_q, scales, biases, K, N],
            template=[("T", x2.dtype)],
            grid=(32 * _MSG_NSG, (N + cols - 1) // cols, 1),
            threadgroup=(32 * _MSG_NSG, 1, 1),
            output_shapes=[(m, N)],
            output_dtypes=[x2.dtype],
        )
        return y[:M0, :] if M0 < m else y

    # Small M gets a dedicated template (fewer wasted accumulators).
    m = max(2, min(6, M))
    xm, M0 = _pad_rows(mx, x2, m)
    k_parts = 2 if N >= 4096 else 4
    kernel = _build_ksplit_kernel(m, bits, group_size, x2.dtype, k_parts=k_parts)
    (y,) = kernel(
        inputs=[xm, w_q, scales, biases, K, N],
        template=[("T", x2.dtype)],
        grid=(32 * k_parts, N // 4, 1),
        threadgroup=(32 * k_parts, 1, 1),
        output_shapes=[(m, N)],
        output_dtypes=[x2.dtype],
    )
    return y[:M0, :] if M0 < m else y
