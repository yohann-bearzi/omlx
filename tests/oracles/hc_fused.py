# SPDX-License-Identifier: Apache-2.0
"""Oracle of the Qwen4 hyper-connection kernels that read quantized weights.

``hc_fused.py`` runs six Metal sources over affine projections: the two-launch
decode (``norm_down`` and ``write_norm_down``, then ``decode_up``), the
three-launch decode (``fused_down``, ``fused_up``) and the prefill's
``tail_inject``. The hyper-connection ports (UNIFY-PLAN U16 to U21) move them
onto a format's dot faces and delete them there; they must keep their bits.
This module freezes them as at 7838ae6: each source, the two headers they use
(``hc_projection._HEADER`` and ``_TG_HEADER``), the weight-free stream norms
the three-launch decode and the prefill launch beside them, and their launch
code (geometry, templates, the choice of form). The prefill's projections are
MLX's ``quantized_matmul``, as the MLX layers compute them.

The launches name their kernels ``omlx_oracle_*``: MLX keys a kernel's
library by its name, source and options, and an oracle kernel's own name keeps
it apart from a live kernel whatever the live sources become.
``tests/test_qwen4_hc_oracle.py`` compares the live kernels with these, bit for
bit. A port that deletes a live source drops its byte comparison there and
keeps the output comparisons.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# hc_projection: MLX's load_vector and qdot of each width, per pack.
HEADER = r"""
using namespace metal;

template <int bits>
inline constexpr short hc_pack_factor() {
    return bits == 5 ? 8 : (bits == 6 ? 4 : 32 / bits);
}

template <int bits>
inline constexpr short hc_bytes_per_pack() {
    constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
    return power_of_2_bits ? 4 : (bits == 5 ? 5 : 3);
}

template <typename T, int N, int bits>
inline float hc_load_vector(const device T* x, thread float* xt) {
    float sum = 0.0f;
    if (bits == 4) {
        for (int i = 0; i < N; i += 4) {
            sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 16.0f;
            xt[i + 2] = x[i + 2] / 256.0f;
            xt[i + 3] = x[i + 3] / 4096.0f;
        }
    } else if (bits == 5) {
        for (int i = 0; i < N; i += 8) {
            sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3])
                + float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 32.0f;
            xt[i + 2] = x[i + 2] / 4.0f;
            xt[i + 3] = x[i + 3] / 128.0f;
            xt[i + 4] = x[i + 4] / 16.0f;
            xt[i + 5] = x[i + 5] / 2.0f;
            xt[i + 6] = x[i + 6] / 64.0f;
            xt[i + 7] = x[i + 7] / 8.0f;
        }
    } else if (bits == 6) {
        for (int i = 0; i < N; i += 4) {
            sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 64.0f;
            xt[i + 2] = x[i + 2] / 16.0f;
            xt[i + 3] = x[i + 3] / 4.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) {
            sum += x[i];
            xt[i] = x[i];
        }
    }
    return sum;
}

template <int N, int bits>
inline float hc_qdot(
    const device uint8_t* w,
    const thread float* xt,
    float scale,
    float bias,
    float sum) {
    float accum = 0.0f;
    if (bits == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < N / 4; ++i) {
            accum +=
                (xt[4 * i] * (ws[i] & 0x000f)
                 + xt[4 * i + 1] * (ws[i] & 0x00f0)
                 + xt[4 * i + 2] * (ws[i] & 0x0f00)
                 + xt[4 * i + 3] * (ws[i] & 0xf000));
        }
    } else if (bits == 5) {
        for (int i = 0; i < N / 8; ++i) {
            xt += 8 * i;
            w += 5 * i;
            accum += (w[0] & 0x1f) * xt[0];
            accum += (w[0] & 0xe0) * xt[1];
            accum += (w[1] & 0x3) * (xt[1] * 256.0f);
            accum += (w[1] & 0x7c) * xt[2];
            accum += (w[1] & 0x80) * xt[3];
            accum += (w[2] & 0xf) * (xt[3] * 256.0f);
            accum += (w[2] & 0xf0) * xt[4];
            accum += (w[3] & 0x1) * (xt[4] * 256.0f);
            accum += (w[3] & 0x3e) * xt[5];
            accum += (w[3] & 0xc0) * xt[6];
            accum += (w[4] & 0x7) * (xt[6] * 256.0f);
            accum += (w[4] & 0xf8) * xt[7];
        }
    } else if (bits == 6) {
        for (int i = 0; i < N / 4; ++i) {
            xt += 4 * i;
            w += 3 * i;
            accum += (w[0] & 0x3f) * xt[0];
            accum += (w[0] & 0xc0) * xt[1];
            accum += (w[1] & 0x0f) * (xt[1] * 256.0f);
            accum += (w[1] & 0xf0) * xt[2];
            accum += (w[2] & 0x03) * (xt[2] * 256.0f);
            accum += (w[2] & 0xfc) * xt[3];
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) accum += xt[i] * w[i];
    }
    return scale * accum + sum * bias;
}

"""

# hc_load_vector over a threadgroup-staged vector.
TG_HEADER = r"""
template <typename T, int N, int bits>
inline float hc_load_vector_tg(const threadgroup T* x, thread float* xt) {
    float sum = 0.0f;
    if (bits == 4) {
        for (int i = 0; i < N; i += 4) {
            sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 16.0f;
            xt[i + 2] = x[i + 2] / 256.0f;
            xt[i + 3] = x[i + 3] / 4096.0f;
        }
    } else if (bits == 5) {
        for (int i = 0; i < N; i += 8) {
            sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3])
                + float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 32.0f;
            xt[i + 2] = x[i + 2] / 4.0f;
            xt[i + 3] = x[i + 3] / 128.0f;
            xt[i + 4] = x[i + 4] / 16.0f;
            xt[i + 5] = x[i + 5] / 2.0f;
            xt[i + 6] = x[i + 6] / 64.0f;
            xt[i + 7] = x[i + 7] / 8.0f;
        }
    } else if (bits == 6) {
        for (int i = 0; i < N; i += 4) {
            sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 64.0f;
            xt[i + 2] = x[i + 2] / 16.0f;
            xt[i + 3] = x[i + 3] / 4.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) {
            sum += x[i];
            xt[i] = x[i];
        }
    }
    return sum;
}
"""

# The decode stream norm (omlx_qwen4_hc_fused_norm); reads no weights.
N_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    constexpr int PER = (H + 255) / 256;
    const device T* xp = x + (size_t)row * K + (size_t)s * H;
    const device T* wp = w + (size_t)s * H;
    device T* op = xn + (size_t)row * K + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(xp[k]) : 0.0f;
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) op[k] = T(v[i] * inv * (1.0f + float(wp[k])));
    }
"""

# The prefill norm with the pending write; reads no weights.
WN_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    constexpr int PER = (H + 255) / 256;
    const size_t base = (size_t)row * K + (size_t)s * H;
    const device T* bp = branch + (size_t)row * H;
    const float g = float(gate[(size_t)row * HC + s]);
    const device T* wp = w + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = 0.0f;
        if (k < H) {
            const T y = T(float(x[base + k]) + float(T(float(bp[k]) * g)));
            y_out[base + k] = y;
            v[i] = float(y);
        }
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::precise::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) xn[base + k] = T(v[i] * inv * (1.0f + float(wp[k])));
    }
"""

# omlx_qwen4_hc_fused_down: down and inject rows, 7 to 16 rows.
D_SOURCE = r"""
    const uint tg = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int KS = 4;
    constexpr int SLICE = K / KS;
    constexpr int GROUPS = K / GS_D;
    threadgroup float part[KS][8];
    const device T* x = xn + (size_t)row * K;
    const int rg = int(sg & 1);
    const int ks = int(sg >> 1);
    if (tg < R / 8) {
        constexpr int PF = hc_pack_factor<BITS_D>();
        constexpr int BP = hc_bytes_per_pack<BITS_D>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = GS_D / VPT;
        const int out_row = int(tg) * 8 + rg * 4;
        const device uint8_t* wp = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS + ks * (SLICE / GS_D)
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS + ks * (SLICE / GS_D)
            + int(lane) / SCALE_STEP;
        const device T* xp = x + ks * SLICE + int(lane) * VPT;
        float result[4] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < 4; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
            wp += BLOCK * BP / PF;
            sp += BLOCK / GS_D;
            bp += BLOCK / GS_D;
            xp += BLOCK;
        }
        // Load only lanes inside the tail; all lanes must join simd_sum.
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < 4; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
        }
        for (int r = 0; r < 4; ++r) {
            float v = simd_sum(result[r]);
            if (lane == 0) part[ks][rg * 4 + r] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < 2 && lane < 4) {
            const int r = int(lane);
            float v = part[0][sg * 4 + r] + part[1][sg * 4 + r]
                + part[2][sg * 4 + r] + part[3][sg * 4 + r];
            v = v / float(HC);
            act[(size_t)row * R + int(tg) * 8 + int(sg) * 4 + r]
                = T(v / (1.0f + metal::exp(-v)));
        }
        return;
    }
    if (INJ == 0 || rg != 0) return;
    {
        constexpr int PF = hc_pack_factor<BITS_I>();
        constexpr int BP = hc_bytes_per_pack<BITS_I>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 1;
        constexpr int VPT = PF;
        constexpr int BLOCK = VPT * 32;
        constexpr int GROUPS_I = K / GS_I;
        constexpr int SCALE_STEP = GS_I / VPT;
        const device uint8_t* wp = (const device uint8_t*)inject_w
            + ks * (SLICE * BP / PF) + int(lane) * BP;
        const device T* sp = inject_s + ks * (SLICE / GS_I) + int(lane) / SCALE_STEP;
        const device T* bp = inject_b + ks * (SLICE / GS_I) + int(lane) / SCALE_STEP;
        const device T* xp = x + ks * SLICE + int(lane) * VPT;
        float result[HC] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector<T, VPT, BITS_I>(xp, xv);
            for (int r = 0; r < HC; ++r) {
                result[r] += hc_qdot<VPT, BITS_I>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS_I]), float(bp[r * GROUPS_I]), sum);
            }
            wp += BLOCK * BP / PF;
            sp += BLOCK / GS_I;
            bp += BLOCK / GS_I;
            xp += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector<T, VPT, BITS_I>(xp, xv);
            for (int r = 0; r < HC; ++r) {
                result[r] += hc_qdot<VPT, BITS_I>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS_I]), float(bp[r * GROUPS_I]), sum);
            }
        }
        for (int r = 0; r < HC; ++r) {
            float v = simd_sum(result[r]);
            if (lane == 0) part[ks][r] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane < HC) {
            const int r = int(lane);
            float v = part[0][r] + part[1][r] + part[2][r] + part[3][r];
            v = v / float(HC);
            inj[(size_t)row * HC + r] = T(2.0f / (1.0f + metal::exp(-v)));
        }
    }
"""

# omlx_qwen4_hc_fused_up: up rows and the stream mix, 7 to 16 rows.
U_SOURCE = r"""
    const uint g = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const int h = int(g) * 64 + int(t >> 2);
    const int s = int(t & 3);
    const int n = s * H + h;
    constexpr int PF = hc_pack_factor<BITS_U>();
    constexpr int BP = hc_bytes_per_pack<BITS_U>();
    constexpr int ROW_BYTES = R * BP / PF;
    constexpr int GROUPS_R = R / GS_U;
    constexpr int CH = 2 * PF;
    constexpr int CPG = GS_U / CH;
    const device uint8_t* w = (const device uint8_t*)up_w + (size_t)n * ROW_BYTES;
    const device T* sp = up_s + (size_t)n * GROUPS_R;
    const device T* bp = up_b + (size_t)n * GROUPS_R;
    float xv[CH];
    const int r = threadgroup_position_in_grid.z;
    const device T* a = act + (size_t)r * R;
    float acc = 0.0f;
    for (int gq = 0; gq < GROUPS_R; ++gq) {
        const float sc = float(sp[gq]);
        const float bi = float(bp[gq]);
        for (int c = 0; c < CPG; ++c) {
            const int e = gq * GS_U + c * CH;
            float sum = hc_load_vector<T, CH, BITS_U>(a + e, xv);
            acc += hc_qdot<CH, BITS_U>(w + e * BP / PF, xv, sc, bi, sum);
        }
    }
    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * float(xn[(size_t)r * K + n]);
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
"""

# omlx_qwen4_hc_decode_[write_]norm_down: norm, down and inject, 1 to 6 rows.
NDN_SOURCE_TEMPLATE = r"""
    const uint gy = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint s = gy & 3;
    const uint g = gy >> 2;
    constexpr int PER = (H + 255) / 256;
    constexpr int KS = 4;
    constexpr int SLICE = K / KS;
    constexpr int GROUPS = K / GS_D;
    constexpr int GROUPS_I = K / GS_I;
    constexpr int PR = R + HC;
    constexpr int ROWS_TG = 8 * RPS;
    constexpr int NDG = R / ROWS_TG;
    static_assert(HC == 4 && K == 4 * H && R % ROWS_TG == 0, "down K slice ks is stream ks");
    static_assert(PER <= NDG + INJ, "one stored strip per threadgroup");
    threadgroup T xs[H];
    threadgroup float npart[8];
    float v[PER];
    T y_mine = T(0.0f);
    T n_mine = T(0.0f);
    const size_t base = (size_t)row * K + (size_t)s * H;
    // The norm weights do not depend on the sum of squares: load them with x.
    const device T* wp = w + (size_t)s * H;
    float wv[PER];
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        wv[i] = (k < H) ? float(wp[k]) : 0.0f;
    }
    const float e = eps[0];
    float ss = 0.0f;
    // LOAD
    ss = simd_sum(ss);
    if (lane == 0) npart[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += npart[i];
    const float inv = metal::rsqrt(tot / float(H) + e);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) {
            const T y = T(v[i] * inv * (1.0f + wv[i]));
            xs[k] = y;
            if (i == int(g)) n_mine = y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool down_tg = g < NDG;
    const int ks = int(s);
    device float* pp = parts + ((size_t)row * KS + ks) * PR;
    if (down_tg) {
        constexpr int PF = hc_pack_factor<BITS_D>();
        constexpr int BP = hc_bytes_per_pack<BITS_D>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = GS_D / VPT;
        const int out_row = int(g) * ROWS_TG + int(sg) * RPS;
        const device uint8_t* wq = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS + ks * (SLICE / GS_D)
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS + ks * (SLICE / GS_D)
            + int(lane) / SCALE_STEP;
        const threadgroup T* xp = xs + int(lane) * VPT;
        float result[RPS] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector_tg<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < RPS; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wq + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
            wq += BLOCK * BP / PF;
            sp += BLOCK / GS_D;
            bp += BLOCK / GS_D;
            xp += BLOCK;
        }
        // Load only lanes inside the tail; all lanes must join simd_sum.
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector_tg<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < RPS; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wq + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
        }
        for (int r = 0; r < RPS; ++r) {
            float v = simd_sum(result[r]);
            if (lane == 0) pp[out_row + r] = v;
        }
    } else if (INJ != 0 && int(sg) < HC) {
        constexpr int PF = hc_pack_factor<BITS_I>();
        constexpr int BP = hc_bytes_per_pack<BITS_I>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int VPT = PF;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = GS_I / VPT;
        const int r = int(sg);
        const device uint8_t* wq = (const device uint8_t*)inject_w
            + r * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * BP;
        const device T* sp = inject_s + r * GROUPS_I + ks * (SLICE / GS_I)
            + int(lane) / SCALE_STEP;
        const device T* bp = inject_b + r * GROUPS_I + ks * (SLICE / GS_I)
            + int(lane) / SCALE_STEP;
        const threadgroup T* xp = xs + int(lane) * VPT;
        float result = 0.0f;
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector_tg<T, VPT, BITS_I>(xp, xv);
            result += hc_qdot<VPT, BITS_I>(wq, xv, float(sp[0]), float(bp[0]), sum);
            wq += BLOCK * BP / PF;
            sp += BLOCK / GS_I;
            bp += BLOCK / GS_I;
            xp += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector_tg<T, VPT, BITS_I>(xp, xv);
            result += hc_qdot<VPT, BITS_I>(wq, xv, float(sp[0]), float(bp[0]), sum);
        }
        float v = simd_sum(result);
        if (lane == 0) pp[R + r] = v;
    }
    if (int(g) < PER && int(t) + int(g) * 256 < H) {
        const size_t at = base + int(t) + int(g) * 256;
        xn[at] = n_mine;
        // WRITE
    }
"""

NDN_LOAD = r"""
    const device T* xp = x + base;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(xp[k]) : 0.0f;
        ss += v[i] * v[i];
    }
"""

NDN_WRITE_LOAD = r"""
    const device T* bq = branch + (size_t)row * H;
    const float g_s = float(gate[(size_t)row * HC + s]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = 0.0f;
        if (k < H) {
            const T y = T(float(x[base + k]) + float(T(float(bq[k]) * g_s)));
            if (i == int(g)) y_mine = y;
            v[i] = float(y);
        }
        ss += v[i] * v[i];
    }
"""

# omlx_qwen4_hc_decode_up: slice sums, up rows and the mix, 1 to 6 rows.
U2_SOURCE = r"""
    constexpr int PF = hc_pack_factor<BITS_U>();
    constexpr int BP = hc_bytes_per_pack<BITS_U>();
    constexpr int ROW_BYTES = R * BP / PF;
    constexpr int GROUPS_R = R / GS_U;
    constexpr int CH = 2 * PF;
    constexpr int CPG = GS_U / CH;
    constexpr int NC = R / CH;
    constexpr int NCP = NC + 1;
    constexpr int HB = NO / HC;
    constexpr int JS = TGS / NC;
    constexpr int KS = 4;
    constexpr int PR = R + HC;
    static_assert(HC == 4 && NO % 32 == 0 && TGS % NC == 0 && JS > 0,
                  "up tiles: four streams, whole simdgroups, whole chunk rows");
    threadgroup T acts[R];
    threadgroup float q[NO * NCP];
    const uint t = thread_index_in_threadgroup;
    const int r = threadgroup_position_in_grid.z;
    const int h0 = int(threadgroup_position_in_grid.y) * HB;
    const int s = int(t & 3);
    const int h = h0 + int(t >> 2);
    const int n = s * H + h;
    // Epilogue threads fetch their normed element before the weights.
    const float xnv = int(t) < NO ? float(xn[(size_t)r * K + n]) : 0.0f;
    const device float* pp = parts + (size_t)r * KS * PR;
    for (int i = int(t); i < R; i += TGS) {
        float v = pp[i] + pp[PR + i] + pp[2 * PR + i] + pp[3 * PR + i];
        v = v / float(HC);
        acts[i] = T(v / (1.0f + metal::exp(-v)));
    }
    const bool inj_thread = INJ != 0 && threadgroup_position_in_grid.y == 0 && int(t) < HC;
    T injv = T(0.0f);
    if (inj_thread) {
        const int i = R + int(t);
        float v = pp[i] + pp[PR + i] + pp[2 * PR + i] + pp[3 * PR + i];
        v = v / float(HC);
        injv = T(2.0f / (1.0f + metal::exp(-v)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const int c = int(t) % NC;
        const int gq = c / CPG;
        const int e = c * CH;
        float xv[CH];
        float sum = hc_load_vector_tg<T, CH, BITS_U>(acts + e, xv);
        for (int j = int(t) / NC; j < NO; j += JS) {
            const int nj = (j / HB) * H + h0 + j % HB;
            const device uint8_t* w = (const device uint8_t*)up_w + (size_t)nj * ROW_BYTES;
            const float sc = float(up_s[(size_t)nj * GROUPS_R + gq]);
            const float bi = float(up_b[(size_t)nj * GROUPS_R + gq]);
            q[j * NCP + c] = hc_qdot<CH, BITS_U>(w + e * BP / PF, xv, sc, bi, sum);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) >= NO) return;
    const threadgroup float* qp = q + (s * HB + int(t >> 2)) * NCP;
    float acc = 0.0f;
    for (int c = 0; c < NC; ++c) {
        acc += qp[c];
    }
    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * xnv;
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
    if (inj_thread) inj[(size_t)r * HC + int(t)] = injv;
"""

# omlx_qwen4_hc_prefill_tail_inject: the prefill's mix and inject rows.
TI_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int PER_H = (H + 255) / 256;
    constexpr int PF = hc_pack_factor<BITS_I>();
    constexpr int BP = hc_bytes_per_pack<BITS_I>();
    constexpr int ROW_BYTES = K * BP / PF;
    constexpr int GROUPS = K / GS_I;
    constexpr int PER = K / 256;
    constexpr int VT = H / PER;
    static_assert(K == HC * H && H % PER == 0 && VT % 32 == 0 && HC * VT == 256,
                  "inject chunks must tile each stream");
    threadgroup T xs[H];
    threadgroup float part[HC][8];
    const device T* up_r = up + (size_t)row * K;
    const device T* xn_r = xn + (size_t)row * K;
    float acc[PER_H];
    for (int s = 0; s < HC; ++s) {
        for (int i = 0; i < PER_H; ++i) {
            const int h = int(t) + i * 256;
            if (h < H) {
                const int n = s * H + h;
                const T x = xn_r[n];
                xs[h] = x;
                const T g = T(1.0f / (1.0f + metal::exp(-float(up_r[n]))));
                const T p = T(float(g) * float(x));
                acc[i] = s == 0 ? float(p) : float(T(acc[i] + float(p)));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(t) < VT) {
            float res[HC] = {0.0f};
            float xv[PF];
            const int e0 = int(t) * PER;
            for (int e = e0; e < e0 + PER; e += PF) {
                const float sum = hc_load_vector_tg<T, PF, BITS_I>(xs + e, xv);
                const int ge = s * H + e;
                const int g = ge / GS_I;
                for (int r = 0; r < HC; ++r) {
                    res[r] += hc_qdot<PF, BITS_I>(
                        (const device uint8_t*)inject_w + r * ROW_BYTES + ge * BP / PF,
                        xv, float(inject_s[r * GROUPS + g]), float(inject_b[r * GROUPS + g]),
                        sum);
                }
            }
            for (int r = 0; r < HC; ++r) {
                const float v = simd_sum(res[r]);
                if (lane == 0) part[r][s * (VT / 32) + int(sg)] = v;
            }
        }
        // Stream s + 1 overwrites xs only after every inject read of stream s.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < PER_H; ++i) {
        const int h = int(t) + i * 256;
        if (h < H) mixed[(size_t)row * H + h] = T(acc[i] * (1.0f / float(HC)));
    }
    if (t < HC) {
        float v = 0.0f;
        for (int i = 0; i < 8; ++i) v += part[t][i];
        const float q = float(T(float(T(v)) / float(HC)));
        const float gate = float(T(1.0f / (1.0f + metal::exp(-q))));
        inj[(size_t)row * HC + t] = T(2.0f * gate);
    }
"""


def ndn_source(load: str, write: bool) -> str:
    return NDN_SOURCE_TEMPLATE.replace("    // LOAD\n", load).replace(
        "        // WRITE\n", "        y_out[at] = y_mine;\n" if write else ""
    )


NDN_SOURCE = ndn_source(NDN_LOAD, write=False)
NDNW_SOURCE = ndn_source(NDN_WRITE_LOAD, write=True)
NP_SOURCE = N_SOURCE.replace("metal::rsqrt", "metal::precise::rsqrt")
WND_SOURCE = WN_SOURCE.replace("metal::precise::rsqrt", "metal::rsqrt")
V2_HEADER = HEADER + TG_HEADER

# hc_fused's geometry: rows of the decode, the two-launch form's limits.
MAX_ROWS = 16
V2_MAX_ROWS = 6
V2_UP_THREADS = 320
TG_MEMORY_BYTES = 32768

_KERNELS: dict[str, object] = {}


def _kernel(name, input_names, output_names, source, header=""):
    name = f"omlx_oracle_{name}"
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            header=header,
            source=source,
            ensure_row_contiguous=True,
        )
        _KERNELS[name] = kernel
    return kernel


def pack_factor(bits: int) -> int:
    return 8 if bits == 5 else (4 if bits == 6 else 32 // bits)


def up_chunks(bits: int, lowrank: int) -> int:
    return lowrank // (2 * pack_factor(bits))


def down_rps(rows: int, gpu_class: str) -> int:
    if gpu_class == "s":
        return 1 if rows == 1 else (2 if rows < 4 else 4)
    return 1 if rows <= 2 else 2


def up_outputs(rows: int, gpu_class: str) -> int:
    return 128 if rows > 1 and gpu_class == "s" else 32


def down_row_groups(lowrank: int, inject: bool, rps: int) -> int:
    return lowrank // (8 * rps) + int(inject)


def two_launch_fits(rows, hc, hidden, lowrank, up_bits, inject, gpu_class) -> bool:
    """hc_fused's ``_decode_v2`` without its row limit and its switch: whether
    the two-launch kernels take ``rows`` rows."""
    rps = down_rps(rows, gpu_class)
    chunks = up_chunks(up_bits, lowrank)
    return (
        hc == 4
        and lowrank % (8 * rps) == 0
        and -(-hidden // 256) <= down_row_groups(lowrank, inject, rps)
        and hidden * 2 + 32 <= TG_MEMORY_BYTES
        and lowrank * 2 + up_outputs(rows, gpu_class) * (chunks + 1) * 4
        <= TG_MEMORY_BYTES
        and chunks <= V2_UP_THREADS
    )


def takes_two_launches(rows, hc, hidden, lowrank, up_bits, inject, gpu_class) -> bool:
    """hc_fused's choice of form: the two-launch kernels to ``V2_MAX_ROWS``
    rows where they fit, the three-launch kernels otherwise."""
    return rows <= V2_MAX_ROWS and two_launch_fits(
        rows, hc, hidden, lowrank, up_bits, inject, gpu_class
    )


def _eps(module) -> mx.array:
    return mx.array([float(module.hc_norm.eps)], dtype=mx.float32)


def _write_operands(hyper_input, write, hidden, hc):
    branch, gate = write
    lead = hyper_input.shape[:-1]
    dtype = hyper_input.dtype
    if not (
        branch.shape == (*lead, hidden)
        and gate.shape == (*lead, hc)
        and branch.dtype == dtype
        and gate.dtype == dtype
    ):
        raise ValueError("hc oracle: hc_fused takes no such pending write")
    rows = hyper_input.shape[0] * hyper_input.shape[1]
    return branch.reshape(rows, hidden), gate.reshape(rows, hc)


def _norm(module, flat, rows, hc, hidden, dtype, precise):
    width = hc * hidden
    return _kernel(
        "qwen4_hc_prefill_norm" if precise else "qwen4_hc_fused_norm",
        ["x", "w", "eps"],
        ["xn"],
        NP_SOURCE if precise else N_SOURCE,
    )(
        inputs=[flat, module.hc_norm.weight, _eps(module)],
        template=[("T", dtype), ("K", width), ("H", hidden)],
        grid=(256, hc, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[dtype],
    )[
        0
    ]


def _write_norm(module, flat, branch, gate, rows, hc, hidden, dtype, precise):
    width = hc * hidden
    return _kernel(
        "qwen4_hc_prefill_write_norm" if precise else "qwen4_hc_fused_write_norm",
        ["x", "branch", "gate", "w", "eps"],
        ["y_out", "xn"],
        WN_SOURCE if precise else WND_SOURCE,
    )(
        inputs=[flat, branch, gate, module.hc_norm.weight, _eps(module)],
        template=[("T", dtype), ("K", width), ("H", hidden), ("HC", hc)],
        grid=(256, hc, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width), (rows, width)],
        output_dtypes=[dtype, dtype],
    )


def _norm_down(
    module, flat, operands, rows, hc, hidden, lowrank, dtype, down, inject, gpu_class
):
    width = hc * hidden
    tensors = inject if inject is not None else down
    inputs = [
        module["hc_norm"]["weight"],
        _eps(module),
        down["weight"],
        down["scales"],
        down["biases"],
        tensors["weight"],
        tensors["scales"],
        tensors["biases"],
    ]
    names = [
        "w",
        "eps",
        "down_w",
        "down_s",
        "down_b",
        "inject_w",
        "inject_s",
        "inject_b",
    ]
    rps = down_rps(rows, gpu_class)
    launch = dict(
        template=[
            ("T", dtype),
            ("BITS_D", down.bits),
            ("BITS_I", tensors.bits),
            ("GS_D", down.group_size),
            ("GS_I", tensors.group_size),
            ("K", width),
            ("H", hidden),
            ("R", lowrank),
            ("HC", hc),
            ("INJ", 1 if inject is not None else 0),
            ("RPS", rps),
        ],
        grid=(256, hc * down_row_groups(lowrank, inject is not None, rps), rows),
        threadgroup=(256, 1, 1),
    )
    shapes = [(rows, width), (rows, hc, lowrank + hc)]
    types = [dtype, mx.float32]
    if operands is None:
        normed, parts = _kernel(
            "qwen4_hc_decode_norm_down",
            ["x", *names],
            ["xn", "parts"],
            NDN_SOURCE,
            header=V2_HEADER,
        )(inputs=[flat, *inputs], output_shapes=shapes, output_dtypes=types, **launch)
        return None, normed, parts
    return _kernel(
        "qwen4_hc_decode_write_norm_down",
        ["x", "branch", "gate", *names],
        ["y_out", "xn", "parts"],
        NDNW_SOURCE,
        header=V2_HEADER,
    )(
        inputs=[flat, *operands, *inputs],
        output_shapes=[(rows, width), *shapes],
        output_dtypes=[dtype, *types],
        **launch,
    )


def _up_mix(normed, parts, rows, hc, hidden, lowrank, dtype, up, inject, gpu_class):
    chunks = up_chunks(up.bits, lowrank)
    threads = chunks * (V2_UP_THREADS // chunks)
    outputs = up_outputs(rows, gpu_class)
    return _kernel(
        "qwen4_hc_decode_up",
        ["xn", "parts", "up_w", "up_s", "up_b"],
        ["mixed", "inj"],
        U2_SOURCE,
        header=V2_HEADER,
    )(
        inputs=[normed, parts, up["weight"], up["scales"], up["biases"]],
        template=[
            ("T", dtype),
            ("BITS_U", up.bits),
            ("GS_U", up.group_size),
            ("K", hc * hidden),
            ("R", lowrank),
            ("HC", hc),
            ("H", hidden),
            ("INJ", int(inject)),
            ("NO", outputs),
            ("TGS", threads),
        ],
        grid=(threads, hc * hidden // outputs, rows),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, hidden), (rows, hc)],
        output_dtypes=[dtype, dtype],
    )


def _down_up(normed, rows, hc, hidden, lowrank, dtype, down, up, inject):
    width = hc * hidden
    tensors = inject if inject is not None else down
    act, injection = _kernel(
        "qwen4_hc_fused_down",
        ["xn", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"],
        ["act", "inj"],
        D_SOURCE,
        header=HEADER,
    )(
        inputs=[
            normed,
            down["weight"],
            down["scales"],
            down["biases"],
            tensors["weight"],
            tensors["scales"],
            tensors["biases"],
        ],
        template=[
            ("T", dtype),
            ("BITS_D", down.bits),
            ("BITS_I", tensors.bits),
            ("GS_D", down.group_size),
            ("GS_I", tensors.group_size),
            ("K", width),
            ("R", lowrank),
            ("HC", hc),
            ("INJ", 1 if inject is not None else 0),
        ],
        grid=(32, 8 * (lowrank // 8 + 1), rows),
        threadgroup=(32, 8, 1),
        output_shapes=[(rows, lowrank), (rows, hc)],
        output_dtypes=[dtype, dtype],
    )
    mixed = _kernel(
        "qwen4_hc_fused_up",
        ["xn", "act", "up_w", "up_s", "up_b"],
        ["mixed"],
        U_SOURCE,
        header=HEADER,
    )(
        inputs=[normed, act, up["weight"], up["scales"], up["biases"]],
        template=[
            ("T", dtype),
            ("BITS_U", up.bits),
            ("GS_U", up.group_size),
            ("K", width),
            ("R", lowrank),
            ("HC", hc),
            ("H", hidden),
        ],
        grid=(256, hidden // 64, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[dtype],
    )[
        0
    ]
    return mixed, injection


def decode(module, hyper_input, write=None, *, gpu_class, two_launch=None):
    """What ``hc_fused.fused_forward`` returns for ``module`` (a
    Qwen4ExpGatedResidual) and ``hyper_input`` of at most 16 rows, on a GPU
    of class ``gpu_class`` (the last letter of its architecture). The form
    is hc_fused's choice, or the two-launch (``two_launch`` True, at any row
    count where it fits) or the three-launch kernels (False). For FP32 rows
    with a pending write, which hc_fused declines (None), it computes the
    kernels' outputs all the same."""
    hc, hidden, lowrank = module.hc_count, module.hidden_size, module.hc_lowrank
    batch, seq, _ = hyper_input.shape
    rows = batch * seq
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"hc oracle: the decode takes 1 to {MAX_ROWS} rows")
    down, up = module["input_mix_weight_down"], module["input_mix_weight_up"]
    inject = module.get("block_inject_weight")
    flat = hyper_input.reshape(rows, hc * hidden)
    dtype = hyper_input.dtype
    operands = (
        None if write is None else _write_operands(hyper_input, write, hidden, hc)
    )
    shape = (rows, hc, hidden, lowrank, up.bits, inject is not None, gpu_class)
    if two_launch is None:
        two_launch = takes_two_launches(*shape)
    elif two_launch and not two_launch_fits(*shape):
        raise ValueError("hc oracle: the two-launch kernels do not take this call")
    if two_launch:
        written, normed, parts = _norm_down(
            module,
            flat,
            operands,
            rows,
            hc,
            hidden,
            lowrank,
            dtype,
            down,
            inject,
            gpu_class,
        )
        mixed, injection = _up_mix(
            normed,
            parts,
            rows,
            hc,
            hidden,
            lowrank,
            dtype,
            up,
            inject is not None,
            gpu_class,
        )
    else:
        if operands is None:
            written, normed = None, _norm(module, flat, rows, hc, hidden, dtype, False)
        else:
            written, normed = _write_norm(
                module, flat, *operands, rows, hc, hidden, dtype, False
            )
        mixed, injection = _down_up(
            normed, rows, hc, hidden, lowrank, dtype, down, up, inject
        )
    if written is not None:
        hyper_input = written.reshape(hyper_input.shape)
    mixed = mixed.reshape(batch, seq, hidden)
    if inject is None:
        return mixed if write is None else (mixed, hyper_input)
    return mixed, hyper_input, injection.reshape(batch, seq, hc)


def _matmul(x, projection):
    """``projection(x)`` for an MLX affine layer: MLX's quantized matmul."""
    return mx.quantized_matmul(
        x,
        projection["weight"],
        projection["scales"],
        projection["biases"],
        transpose=True,
        group_size=projection.group_size,
        bits=projection.bits,
    )


_ACTS: dict[int, object] = {}
_TAILS: dict[tuple[int, int], object] = {}


def _act(hc: int):
    fn = _ACTS.get(hc)
    if fn is None:
        fn = mx.compile(lambda y: nn.silu(y / hc), shapeless=True)
        _ACTS[hc] = fn
    return fn


def _tail(hc: int, hidden: int):
    fn = _TAILS.get((hc, hidden))
    if fn is None:

        def tail(up, normed):
            gate = mx.sigmoid(up)
            acc = gate[..., :hidden] * normed[..., :hidden]
            for g in range(1, hc):
                lo = g * hidden
                acc = acc + gate[..., lo : lo + hidden] * normed[..., lo : lo + hidden]
            return acc * (1.0 / hc)

        fn = mx.compile(tail)
        _TAILS[(hc, hidden)] = fn
    return fn


def prefill(module, hyper_input, write=None):
    """What ``hc_fused.prefill_forward(..., nax=False)`` returns for more than
    16 rows: the stream norm, MLX's quantized matmuls, then the tail kernel
    (or, where it does not take the width, the compiled tail and MLX's
    inject)."""
    hc, hidden = module.hc_count, module.hidden_size
    width = hc * hidden
    dtype = hyper_input.dtype
    rows = hyper_input.shape[0] * hyper_input.shape[1]
    flat = hyper_input.reshape(rows, width)
    inject = module.get("block_inject_weight")
    if write is None:
        normed = _norm(module, flat, rows, hc, hidden, dtype, True)
    else:
        written, normed = _write_norm(
            module,
            flat,
            *_write_operands(hyper_input, write, hidden, hc),
            rows,
            hc,
            hidden,
            dtype,
            True,
        )
        hyper_input = written.reshape(hyper_input.shape)
    normed = normed.reshape(hyper_input.shape)
    mix = _act(hc)(_matmul(normed, module["input_mix_weight_down"]))
    up = _matmul(mix, module["input_mix_weight_up"])
    if inject is None or width % (256 * pack_factor(inject.bits)):
        mixed = _tail(hc, hidden)(up, normed)
        injection = (
            None if inject is None else 2 * mx.sigmoid(_matmul(normed, inject) / hc)
        )
    else:
        mixed, injection = _kernel(
            "qwen4_hc_prefill_tail_inject",
            ["up", "xn", "inject_w", "inject_s", "inject_b"],
            ["mixed", "inj"],
            TI_SOURCE,
            header=HEADER + TG_HEADER,
        )(
            inputs=[
                up.reshape(rows, width),
                normed.reshape(rows, width),
                inject["weight"],
                inject["scales"],
                inject["biases"],
            ],
            template=[
                ("T", dtype),
                ("BITS_I", inject.bits),
                ("GS_I", inject.group_size),
                ("K", width),
                ("H", hidden),
                ("HC", hc),
            ],
            grid=(256, 1, rows),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, hidden), (rows, hc)],
            output_dtypes=[dtype, dtype],
        )
        mixed = mixed.reshape(*hyper_input.shape[:-1], hidden)
        injection = injection.reshape(*hyper_input.shape[:-1], hc)
    if injection is None:
        return mixed if write is None else (mixed, hyper_input)
    return mixed, hyper_input, injection
