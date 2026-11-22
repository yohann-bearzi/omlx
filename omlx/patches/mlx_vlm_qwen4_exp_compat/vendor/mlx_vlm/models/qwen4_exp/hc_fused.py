# SPDX-License-Identifier: Apache-2.0
"""Metal kernels for Qwen4 hyper-connection inputs.

Decode (at most 16 BF16 or FP16 rows) fuses per-stream RMS norm, down/inject
projections with activation, and the up projection with stream mixing.
Prefill fuses the stream norm, silu(down / HC), and the mixing/inject
epilogue around the down/up projections; on GPUs with tensor units (M5) those
projections run in hc_prefill_nax's kernels with the epilogues folded in
(bit-identical to the MLX matmul path), elsewhere on MLX's quantized matmul.
Both need four streams and affine group-size-32 or -64 projections with 4/5/6/8-bit
weights. FP32 epilogues can round differently from the canonical 16-bit
operations. The rows, norm weight and projection scales share one dtype.

A pending residual write (previous block's branch times its injection gate)
can be applied inside the stream norm, which also stores the written
residual; it rounds exactly like the eager multiply and add.

Each kernel specialization is evaluated once to catch lazy compilation errors.
A failure falls back for the current call only; the NAX prefill projections
instead fall back to MLX for the process after a failure or a bitwise
mismatch on first use. Later evaluation errors propagate.
Where the geometry allows, small decode batches take two launches in place of
the three-launch kernels (norm, down/inject, up/mix); both produce the same bits.
"""

from __future__ import annotations

import functools
import logging
from itertools import chain
from operator import is_, itemgetter

import mlx.core as mx
import mlx.nn as nn

from omlx.formats import affine, kernels
_HEADER = r"""
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

from . import hc_prefill_nax

logger = logging.getLogger(__name__)

MAX_ROWS = 16
# Each projection's group size is a template parameter (GS_D, GS_I, GS_U): the
# traversal is the same for every group size, only the scale/bias index moves.
_GROUP_SIZES = (32, 64)
_SUPPORTED_BITS = (4, 5, 6, 8)
_DTYPES = (mx.bfloat16, mx.float16)
# Tests flip the flags below to reach the reference paths.
_DISABLED = False
_WRITE_DISABLED = False
# Prefill rows on the tensor units (hc_prefill_nax): three dispatches instead of
# six, bit-identical.
_NAX_DISABLED = False
_NAX_AVAILABLE: bool | None = None
_NAX_BROKEN = False
_DECODE_V2 = True
_KERNELS: dict[str, object] = {}
_VALIDATED: set[object] = set()
_FAILURE_LOGGED = False
_INELIGIBLE_LOGGED = False

_N_SOURCE = r"""
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

# The two down kernels traverse the same K slice. Its activations can be a
# stream staged in threadgroup memory or a slice of the device norm buffer.
_SLICE_HEADER = r"""
template <int K, int S, int N, typename Matrix, typename XP>
METAL_FUNC void hc_slice(Matrix m, int first, int slice, uint lane, XP x,
                         thread float (&result)[N]) {
    using L = typename Matrix::template Lanes<K>;
    static_assert(S % L::VPT == 0, "HC slice must hold whole lanes");
    constexpr int whole = S - S % L::BLOCK;
    constexpr int tail = S % L::BLOCK / L::VPT;
    L cursor(m, first, slice * (S / L::VPT) + lane);
    x += lane * L::VPT;
    float xt[L::VPT];
    for (int k = 0; k < whole; k += L::BLOCK) {
        const auto aux = L::load(x, xt);
        for (int r = 0; r < N; ++r)
            result[r] = L::madd(result[r], cursor.row(r), xt, aux);
        cursor.step();
        x += L::BLOCK;
    }
    if constexpr (tail > 0) {
        if (lane < uint(tail)) {
            const auto aux = L::load(x, xt);
            for (int r = 0; r < N; ++r)
                result[r] = L::madd(result[r], cursor.row(r), xt, aux);
        }
    }
    for (int r = 0; r < N; ++r)
        result[r] = cursor.finish(r, metal::simd_sum(result[r]));
}
"""

_DOWN = r"""
    const uint tg = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int KS = 4;
    constexpr int SLICE = K / KS;
    threadgroup float part[KS][8];
    const device T* x = xn + (size_t)row * K;
    const int rg = int(sg & 1);
    const int ks = int(sg >> 1);
    if (tg < R / 8) {
        const int out_row = int(tg) * 8 + rg * 4;
        float result[4] = {0.0f};
        hc_slice<K, SLICE>(down, out_row, ks, lane, x + ks * SLICE, result);
        for (int r = 0; r < 4; ++r) {
            if (lane == 0) part[ks][rg * 4 + r] = result[r];
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
        float result[HC] = {0.0f};
        hc_slice<K, SLICE>(inject, 0, ks, lane, x + ks * SLICE, result);
        for (int r = 0; r < HC; ++r) {
            if (lane == 0) part[ks][r] = result[r];
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

def _serial_up_source(dot, scratch=""):
    """Share the serial up kernel's indexing, sigmoid and stream mix."""
    return f"""
    const uint g = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const int h = int(g) * 64 + int(t >> 2);
    const int s = int(t & 3);
    const int n = s * H + h;
    using L = typename decltype(up)::template Lanes<R>;
    static_assert(R % L::VPT == 0, "HC up row must hold whole chunks");
{scratch}    const int r = threadgroup_position_in_grid.z;
    const device T* a = act + (size_t)r * R;
{dot}    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * float(xn[(size_t)r * K + n]);
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
"""


_UP = _serial_up_source(
    r"""    float acc = 0.0f;
    for (int c = 0; c < R / L::VPT; ++c) {
        L cursor(up, n, c);
        const auto aux = L::load(a + c * L::VPT, xv);
        acc += L::madd(0.0f, cursor.row(0), xv, aux);
    }
    acc = L(up, n, 0).finish(0, acc);
""",
    scratch="    float xv[L::VPT];\n",
)
_STREAM_UP = _serial_up_source("    float acc = hc_stream_up<R>(up, n, a);\n")

# Two-launch decode, bit-identical to _N_SOURCE/_WND_SOURCE -> _DOWN -> _UP.
# A dependent launch costs ~2 us of GPU time on its own, and the three-launch
# kernels ran on 4, 41 and 40 threadgroups of an 80-core GPU. With four streams
# the down projection's K slice ks is stream ks, so the stream norm folds into
# the down launch: one 256-thread threadgroup per (row group g, slice ks)
# normalizes only its own stream. Device stores come last in both kernels: a
# device load after a store the compiler cannot prove disjoint waits for it.
#
# _NORM_DOWN: threadgroup (g, ks) normalizes stream ks (and applies the pending
# write) exactly like the norm kernels (element k = t + i * 256, simd_sum, then
# the eight partials in order) into threadgroup memory. Simdgroup sg then runs the
# _DOWN lanes, block order and simd_sum of down rows g * 8 * RPS + sg * RPS + r
# over slice ks (RPS only sets how many rows share one pass over x); the extra row
# group g = R / (8 * RPS) runs the inject rows. Each simd_sum goes to
# parts[row][ks][out], and threadgroup (g, ks) stores strip i = g of stream ks.
_NORM_DOWN_TEMPLATE = r"""
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
        const int out_row = int(g) * ROWS_TG + int(sg) * RPS;
        float result[RPS] = {0.0f};
        hc_slice<K, SLICE>(down, out_row, ks, lane, xs, result);
        for (int r = 0; r < RPS; ++r) {
            if (lane == 0) pp[out_row + r] = result[r];
        }
    } else if (INJ != 0 && int(sg) < HC) {
        const int r = int(sg);
        float result[1] = {0.0f};
        hc_slice<K, SLICE>(inject, r, ks, lane, xs, result);
        if (lane == 0) pp[R + r] = result[0];
    }
    if (int(g) < PER && int(t) + int(g) * 256 < H) {
        const size_t at = base + int(t) + int(g) * 256;
        xn[at] = n_mine;
        // WRITE
    }
"""

_NDN_LOAD = r"""
    const device T* xp = x + base;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(xp[k]) : 0.0f;
        ss += v[i] * v[i];
    }
"""

_NDN_WRITE_LOAD = r"""
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


_NORM_DOWN = _NORM_DOWN_TEMPLATE.replace("// LOAD", _NDN_LOAD).replace("// WRITE", "")
_WRITE_NORM_DOWN = _NORM_DOWN_TEMPLATE.replace("// LOAD", _NDN_WRITE_LOAD).replace(
    "// WRITE", "y_out[at] = y_mine;"
)

# _DECODE_UP: threadgroup y owns up rows j = s * HB + hh (n = s * H + y * HB + hh).
# Every threadgroup combines the slice sums as _DOWN does (part[0] + part[1]
# + part[2] + part[3], then / HC and the BF16-rounded silu; 2 * sigmoid for the
# inject rows in threadgroup 0) into threadgroup memory. Thread t then owns chunk
# c = t % NC of rows j = t / NC, t / NC + TGS / NC, ... and stages each chunk's
# `madd` in q; one thread per output adds its NC chunks in chunk order, and the
# sigmoid gate / shuffle-down stream mix is _UP's.
_DECODE_UP = r"""
    using L = typename decltype(up)::template Lanes<R>;
    static_assert(R == NC * L::VPT, "HC up tile must hold whole chunks");
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
        float xv[L::VPT];
        const auto aux = L::load(acts + c * L::VPT, xv);
        for (int j = int(t) / NC; j < NO; j += JS) {
            const int nj = (j / HB) * H + h0 + j % HB;
            L cursor(up, nj, c);
            q[j * NCP + c] = L::madd(0.0f, cursor.row(0), xv, aux);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) >= NO) return;
    const threadgroup float* qp = q + (s * HB + int(t >> 2)) * NCP;
    float acc = 0.0f;
    for (int c = 0; c < NC; ++c) {
        acc += qp[c];
    }
    acc = L(up, n, 0).finish(0, acc);
    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * xnv;
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
    if (inj_thread) inj[(size_t)r * HC + int(t)] = injv;
"""


# The tail inject over the inject projection's dot face, one row per
# threadgroup: mixed = mean over streams of sigmoid(up) * normed, with BF16
# rounding after each gate, product and partial sum; inj = 2 * sigmoid of the
# inject dot, from the quantized four-row bank. The normed row is read once:
# each stream is staged in threadgroup memory and its inject dot runs over the
# face from there. The inject sum keeps the stock order: virtual thread
# v = s * VT + t (t < VT) sums its contiguous PER values in whole face lanes,
# then simd_sum per 32 virtual threads, then the 8 partials in order, and the
# row's face finishes the total once.
_TAIL_INJECT = r"""
    using L = typename decltype(inject)::template Lanes<K>;
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int PER_H = (H + 255) / 256;
    constexpr int PER = K / 256;
    constexpr int VT = H / PER;
    constexpr int CHUNKS = PER / L::VPT;
    constexpr int PARTS = VT / 32;
    static_assert(K == HC * H && H % PER == 0 && VT % 32 == 0 && HC * VT == 256,
                  "inject chunks must tile each stream");
    static_assert(PER % L::VPT == 0, "an inject chunk must hold whole lanes");
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
            float xv[L::VPT];
            const int e0 = int(t) * PER;
            const uint c0 = uint((s * H + e0) / L::VPT);
            for (int i = 0; i < CHUNKS; ++i) {
                const auto aux = L::load((const threadgroup T*)(xs + e0 + i * L::VPT), xv);
                const L cursor(inject, 0, c0 + uint(i));
                for (int r = 0; r < HC; ++r) {
                    res[r] = L::madd(res[r], cursor.row(r), xv, aux);
                }
            }
            for (int r = 0; r < HC; ++r) {
                const float v = metal::simd_sum(res[r]);
                if (lane == 0) part[r][s * PARTS + int(sg)] = v;
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
        v = L(inject, int(t), 0).finish(0, v);
        const float q = float(T(float(T(v)) / float(HC)));
        const float gate = float(T(1.0f / (1.0f + metal::exp(-q))));
        inj[(size_t)row * HC + t] = T(2.0f * gate);
    }
"""


def _kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
    header: str = "",
):
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


def enabled() -> bool:
    return not _DISABLED


def write_enabled() -> bool:
    """Whether residual writes may be deferred into the next fused stream norm."""
    return not (_DISABLED or _WRITE_DISABLED)


def _quantized_ok(projection, dtype) -> bool:
    return (
        isinstance(projection, nn.Module)
        and type(projection).__qualname__ in ("FormatLinear", "QuantizedLinear")
        and not getattr(projection, "_placeholders", None)
        and getattr(projection, "group_size", None) in _GROUP_SIZES
        and getattr(projection, "bits", None) in _SUPPORTED_BITS
        and getattr(projection, "mode", "affine") == "affine"
        and "bias" not in projection
        and isinstance(getattr(projection, "weight", None), mx.array)
        and projection.weight.dtype == mx.uint32
        and isinstance(getattr(projection, "scales", None), mx.array)
        and isinstance(getattr(projection, "biases", None), mx.array)
        and projection.scales.dtype == dtype
        and projection.biases.dtype == dtype
    )


def _ineligible(reason: str) -> bool:
    """Log the first unsupported model layout per process."""
    global _INELIGIBLE_LOGGED
    if not _INELIGIBLE_LOGGED:
        _INELIGIBLE_LOGGED = True
        logger.info(
            "Qwen4 fused hyper-connection kernels not used for this model (%s); "
            "the canonical path stays in effect",
            reason,
        )
    return False


def _rows_of(hyper_input) -> int | None:
    if not (
        isinstance(hyper_input, mx.array)
        and hyper_input.ndim == 3
        and hyper_input.dtype in _DTYPES
    ):
        return None
    return hyper_input.shape[0] * hyper_input.shape[1]


def compatible(module, hyper_input) -> bool:
    """Whether ``module`` (a Qwen4ExpGatedResidual) can take the fused path for ``hyper_input``."""
    # Runs for every hyper-connection of every decode step: the row gate
    # first (prefill rows must not reach the layout checks), then the
    # cached layout verdict.
    if _DISABLED or not isinstance(hyper_input, mx.array):
        return False
    shape = hyper_input.shape
    return (
        len(shape) == 3
        and 1 <= shape[0] * shape[1] <= MAX_ROWS
        and _layout_compatible(module, hyper_input)
    )


def prefill_compatible(module, hyper_input) -> bool:
    """Whether rows above MAX_ROWS can use the compiled prefill mean."""
    rows = _rows_of(hyper_input)
    return (
        enabled()
        and rows is not None
        and rows > MAX_ROWS
        and _layout_compatible(module, hyper_input)
    )


def _layout_compatible(module, hyper_input) -> bool:
    if _static_layout_dtype(module) != hyper_input.dtype:
        return False
    if hyper_input.shape[2] != module.hc_count * module.hidden_size:
        return _ineligible(f"input width {hyper_input.shape[2]}")
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _static_layout_dtype(module):
    """The module's compute dtype, or False for an unsupported layout.

    Cached per module; any reassigned child, weight, scale or bias re-checks.

    The key is the identity of every value of the module and of its child
    modules, compared with ``is`` (``id()`` raises an audit event per call).
    The cache keeps those objects alive so a match cannot come from a new
    object reusing a freed one's address.
    """
    cached = module.__dict__.get("_omlx_hc_layout")
    if cached is not None:
        children, refs, dtype = cached
        current = tuple(chain(dict.values(module), *map(dict.values, children)))
        if len(current) == len(refs) and all(map(is_, current, refs)):
            return dtype
    dtype = _check_static_layout(module)
    children = [value for value in dict.values(module) if isinstance(value, nn.Module)]
    refs = tuple(chain(dict.values(module), *map(dict.values, children)))
    module.__dict__["_omlx_hc_layout"] = (children, refs, dtype)
    return dtype


def _check_static_layout(module):
    """The hc_norm weight dtype when the whole layout is supported, else False."""
    if hasattr(module, "input_inject_weight"):
        return _ineligible("combined input projection layout")
    hc_count = getattr(module, "hc_count", None)
    hidden = getattr(module, "hidden_size", None)
    lowrank = getattr(module, "hc_lowrank", None)
    if not (
        isinstance(hc_count, int)
        and isinstance(hidden, int)
        and isinstance(lowrank, int)
        and hc_count == 4
        # The up grid and quantization groups require 64-element alignment.
        and hidden % 64 == 0
        and lowrank % 64 == 0
    ):
        return _ineligible(
            f"geometry hidden_size={hidden} hc_lowrank={lowrank} hc_count={hc_count}"
        )
    norm = getattr(module, "hc_norm", None)
    if not (
        norm is not None
        and getattr(norm, "group_size", None) == hidden
        and isinstance(getattr(norm, "weight", None), mx.array)
        and norm.weight.dtype in _DTYPES
        and norm.weight.shape == (hc_count * hidden,)
    ):
        return _ineligible("hc_norm layout")
    dtype = norm.weight.dtype
    if not (
        _quantized_ok(getattr(module, "input_mix_weight_down", None), dtype)
        and _quantized_ok(getattr(module, "input_mix_weight_up", None), dtype)
    ):
        return _ineligible(
            "projection quantisation (need affine group-size-32/64 4/5/6/8-bit "
            "with scales in the hc_norm dtype)"
        )
    if "block_inject_weight" in module and not _quantized_ok(
        module.block_inject_weight, dtype
    ):
        return _ineligible("block_inject_weight quantisation")
    return dtype


def _eps_array(module) -> mx.array:
    eps = getattr(module, "_omlx_hc_fused_eps", None)
    if eps is None:
        eps = mx.array([float(module.hc_norm.eps)], dtype=mx.float32)
        mx.eval(eps)
        module._omlx_hc_fused_eps = eps
    return eps


# Prefill keeps Qwen4ExpRMSNorm's precise rsqrt; only the f32 sum order differs.
_NP_SOURCE = _N_SOURCE.replace("metal::rsqrt", "metal::precise::rsqrt")

# The previous block's residual write fused into the stream norm:
# - y = T(x + T(branch * gate[s])) rounds like the eager multiply and add.
# - y is stored as the new residual stream and normalized as in _NP_SOURCE
#   (prefill) or _N_SOURCE (decode, _WND_SOURCE below).
_WN_SOURCE = r"""
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

# Decode keeps _N_SOURCE's fast rsqrt so rows match the unfused decode norm.
_WND_SOURCE = _WN_SOURCE.replace("metal::precise::rsqrt", "metal::rsqrt")


def _kernel_norm(module, flat, rows, hc, hidden, dtype, precise=False):
    width = hc * hidden
    name = "omlx_qwen4_hc_prefill_norm" if precise else "omlx_qwen4_hc_fused_norm"
    return _kernel(
        name, ["x", "w", "eps"], ["xn"], _NP_SOURCE if precise else _N_SOURCE
    )(
        inputs=[flat, module.hc_norm.weight, _eps_array(module)],
        template=[("T", dtype), ("K", width), ("H", hidden)],
        grid=(256, hc, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[dtype],
    )[0]


def _write_operands(hyper_input, write, hidden, hc):
    """Flattened (branch, gate) for the write-norm kernels, or None if they cannot take it.

    Mixed dtypes would promote the eager write, so only same-dtype operands qualify.
    """
    branch, gate = write
    lead = hyper_input.shape[:-1]
    dtype = hyper_input.dtype
    if not (
        isinstance(branch, mx.array)
        and isinstance(gate, mx.array)
        and branch.shape == (*lead, hidden)
        and gate.shape == (*lead, hc)
        and branch.dtype == dtype
        and gate.dtype == dtype
    ):
        return None
    rows = _rows_of(hyper_input)
    return branch.reshape(rows, hidden), gate.reshape(rows, hc)


def _kernel_write_norm(
    module, flat, branch, gate, rows, hc, hidden, dtype, precise=True
):
    width = hc * hidden
    return _kernel(
        "omlx_qwen4_hc_prefill_write_norm"
        if precise
        else "omlx_qwen4_hc_fused_write_norm",
        ["x", "branch", "gate", "w", "eps"],
        ["y_out", "xn"],
        _WN_SOURCE if precise else _WND_SOURCE,
    )(
        inputs=[flat, branch, gate, module.hc_norm.weight, _eps_array(module)],
        template=[("T", dtype), ("K", width), ("H", hidden), ("HC", hc)],
        grid=(256, hc, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width), (rows, width)],
        output_dtypes=[dtype, dtype],
    )


def _pack_factor(bits: int) -> int:
    return 8 if bits == 5 else (4 if bits == 6 else 32 // bits)


# Two-launch decode geometry, tuned on an 80-core M5 Ultra and a 40-core M5 Max
# (GPU class 's'); none of it changes an output bit. Down rows per simdgroup: 1
# for one or two rows (more threadgroups), 2 above; on the M5 Max 2 from two rows
# and 4 from four. Up rows per threadgroup: 32; on the M5 Max 128 from two rows, a
# quarter of the threadgroups, each combining the slice sums once for four times
# the rows. From 7 rows on, the per-row threadgroups re-read the weights more
# often than the three-launch kernels, which are then as fast or faster.
_V2_MAX_ROWS = 6
_V2_UP_THREADS = 320
_TG_MEMORY_BYTES = 32768


@functools.lru_cache(maxsize=None)
def _gpu_class() -> str:
    try:
        return str(mx.device_info().get("architecture", ""))[-1:]
    except Exception:
        return ""


def _decode_face(layer, inject):
    return affine.face(
        layer.bits,
        layer.group_size,
        not inject,
        layer["scales"].dtype,
        partial_blocks=not inject,
    )


def _matrix_reader(face):
    if not face.tensors:
        return lambda layer: (layer["weight"],)
    return itemgetter("weight", *face.tensors)


def _down_rps(rows: int) -> int:
    if _gpu_class() == "s":
        return 1 if rows == 1 else (2 if rows < 4 else 4)
    return 1 if rows <= 2 else 2


def _up_outputs(rows: int) -> int:
    return 128 if rows > 1 and _gpu_class() == "s" else 32


def _down_row_groups(lowrank: int, inject: bool, rps: int) -> int:
    return lowrank // (8 * rps) + int(inject)


def _decode_v2(
    rows: int,
    hc: int,
    hidden: int,
    lowrank: int,
    chunks: int,
    inject: bool,
    itemsize: int = 2,
    table_bytes_down: int = 0,
    table_bytes_up: int = 0,
) -> bool:
    """Whether the two-launch decode kernels apply: four streams, one stored
    256-element strip per norm/down threadgroup, and each kernel's threadgroup
    arrays within threadgroup memory."""
    rps = _down_rps(rows)
    return (
        _DECODE_V2
        and rows <= _V2_MAX_ROWS
        and hc == 4
        and lowrank % (8 * rps) == 0
        and -(-hidden // 256) <= _down_row_groups(lowrank, inject, rps)
        and hidden * itemsize + 32 + table_bytes_down <= _TG_MEMORY_BYTES
        and lowrank * itemsize + _up_outputs(rows) * (chunks + 1) * 4 + table_bytes_up
        <= _TG_MEMORY_BYTES
        and 0 < chunks <= _V2_UP_THREADS
    )


def _down_kernel(name, inputs, matrices, outputs, source):
    if "inject" not in matrices:
        # INJ is zero and the compiler elides the unused bank.
        source = "const auto inject = down;\n" + source
    return kernels.face_kernel(name, inputs, matrices, outputs, _SLICE_HEADER, source)


def _kernel_norm_down(launch, module, flat, operands, weights):
    norm = (module.hc_norm.weight, _eps_array(module))
    if operands is None:
        xn, parts = launch(inputs=[flat, *norm, *weights])
        return None, xn, parts
    return launch(inputs=[flat, *operands, *norm, *weights])


def _tail_inject_ok(width, hidden, hc, dtype, face) -> bool:
    """Whether the prefill tail inject's face tiles 256 threads and fits the
    threadgroup: its staged stream, the partials and its tables."""
    return (
        face is not None
        and face.vpt >= 1
        and hidden > 0
        and width > 0
        and width == hc * hidden
        and 8 % hc == 0
        and width % (256 * face.vpt) == 0
        and hidden * dtype.size + hc * 32 + kernels.table_bytes(face)
        <= _TG_MEMORY_BYTES
    )


def _tail_inject_launch(hc, hidden, dtype, inject_face):
    """The prefill tail inject's launch for one geometry and face, or None
    when the face does not fit. One call builds it; the module's plan holds
    it for that layer and dtype, so no global cache outlives its faces. Each
    call reads its own layer's arrays, including replacements after a load."""
    if inject_face is None or not _tail_inject_ok(
        hc * hidden, hidden, hc, dtype, inject_face
    ):
        return None
    width = hc * hidden
    read = _matrix_reader(inject_face)
    kernel = kernels.face_kernel(
        "omlx_qwen4_hc_prefill_tail_inject",
        ("up", "xn", "inject"),
        {"inject": inject_face},
        ("mixed", "inj"),
        "",
        _TAIL_INJECT,
    )
    template = [("T", dtype), ("K", width), ("H", hidden), ("HC", hc)]
    dtypes = [dtype, dtype]
    recorded = [False]  # the first call records the face kernel's instance

    def run(rows, up, normed, inject_layer):
        launch = dict(
            inputs=[up, normed, *read(inject_layer)],
            template=template,
            grid=(256, 1, rows),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, hidden), (rows, hc)],
            output_dtypes=dtypes,
        )
        if recorded[0]:
            return kernel.native(**launch)
        out = kernel(**launch)
        recorded[0] = True
        return out

    return run


@functools.cache
def _decode_launch(
    rows, hc, hidden, lowrank, dtype, down_face, up_face, inj_face, write
):
    """Choose the decode form and specialize its launches for one geometry and
    layout. This callable holds kernel arguments shared by layers and threads.
    Each call reads its own layer's arrays, including replacements after a load.
    """
    width = hc * hidden
    inject = inj_face is not None
    chunks = lowrank // up_face.vpt
    two = _decode_v2(
        rows,
        hc,
        hidden,
        lowrank,
        chunks,
        inject,
        dtype.size,
        table_bytes_down=kernels.table_bytes(
            down_face, *((inj_face,) if inject else ())
        ),
        table_bytes_up=kernels.table_bytes(up_face),
    )
    rps = _down_rps(rows) if two else 0
    no = _up_outputs(rows) if two else 0
    matrices = {"down": down_face}
    down_inputs = _matrix_reader(down_face)
    up_inputs = _matrix_reader(up_face)
    if inject:
        matrices["inject"] = inj_face
        inj_inputs = _matrix_reader(inj_face)

    shape = [("T", dtype), ("K", width), ("R", lowrank), ("HC", hc)]
    if two:
        names = ("x", "branch", "gate") if write else ("x",)
        results = ("y_out", "xn", "parts") if write else ("xn", "parts")
        shapes = [(rows, width)] * (1 + int(write)) + [(rows, hc, lowrank + hc)]
        name = "write_norm_down" if write else "norm_down"
        down_kernel = _down_kernel(
            f"omlx_qwen4_hc_decode_{name}",
            (*names, "w", "eps", *matrices),
            matrices,
            results,
            _WRITE_NORM_DOWN if write else _NORM_DOWN,
        )
        down = functools.partial(
            down_kernel,
            template=[*shape, ("H", hidden), ("INJ", int(inject)), ("RPS", rps)],
            grid=(256, hc * _down_row_groups(lowrank, inject, rps), rows),
            threadgroup=(256, 1, 1),
            output_shapes=shapes,
            output_dtypes=[dtype] * (1 + int(write)) + [mx.float32],
        )
        threads = chunks * (_V2_UP_THREADS // chunks)
        up_kernel = kernels.face_kernel(
            "omlx_qwen4_hc_decode_up",
            ("xn", "parts", "up"),
            {"up": up_face},
            ("mixed", "inj"),
            "",
            _DECODE_UP,
        )
        up = functools.partial(
            up_kernel,
            template=[
                *shape,
                ("H", hidden),
                ("INJ", int(inject)),
                ("NO", no),
                ("NC", chunks),
                ("TGS", threads),
            ],
            grid=(threads, width // no, rows),
            threadgroup=(threads, 1, 1),
            output_shapes=[(rows, hidden), (rows, hc)],
            output_dtypes=[dtype, dtype],
        )
    else:
        down_kernel = _down_kernel(
            "omlx_qwen4_hc_fused_down",
            ("xn", *matrices),
            matrices,
            ("act", "inj"),
            _DOWN,
        )
        down = functools.partial(
            down_kernel,
            template=[*shape, ("INJ", int(inject))],
            grid=(32, 8 * (lowrank // 8 + 1), rows),
            threadgroup=(32, 8, 1),
            output_shapes=[(rows, lowrank), (rows, hc)],
            output_dtypes=[dtype, dtype],
        )
        # Only the stock 8-bit g64 up shares these group scalars.
        streaming = 7 <= rows <= 16 and up_face == affine.face(
            8, 64, True, dtype, partial_blocks=True
        )
        up_kernel = kernels.face_kernel(
            "omlx_qwen4_hc_fused_up",
            ("xn", "act", "up"),
            {"up": up_face},
            ("mixed",),
            affine.serial_up_header(up_face) if streaming else "",
            _STREAM_UP if streaming else _UP,
        )
        up = functools.partial(
            up_kernel,
            template=[*shape, ("H", hidden)],
            grid=(256, hidden // 64, rows),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, hidden)],
            output_dtypes=[dtype],
        )

    def launch(module, flat, operands, down_layer, up_layer, inj_layer):
        weights = down_inputs(down_layer)
        if inject:
            weights += inj_inputs(inj_layer)
        if two:
            written, xn, parts = _kernel_norm_down(
                down, module, flat, operands, weights
            )
            mixed, injection = up(inputs=[xn, parts, *up_inputs(up_layer)])
        else:
            if write:
                written, xn = _kernel_write_norm(
                    module, flat, *operands, rows, hc, hidden, dtype, precise=False
                )
            else:
                written = None
                xn = _kernel_norm(module, flat, rows, hc, hidden, dtype)
            act, injection = down(inputs=[xn, *weights])
            mixed = up(inputs=[xn, act, *up_inputs(up_layer)])[0]
        return written, mixed, injection

    return launch


_TAILS: dict[tuple[int, int], object] = {}
_ACTS: dict[int, object] = {}


def _act(hc: int):
    """Compiled silu(down / hc): one launch that rounds like the two eager ops."""
    fn = _ACTS.get(hc)
    if fn is None:
        fn = mx.compile(lambda y: nn.silu(y / hc), shapeless=True)
        _ACTS[hc] = fn
    return fn


def _tail(hc: int, hidden: int):
    """Compiled mean over the streams as slice products: one fused pass instead of a strided reduce."""
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


def _nax_available() -> bool:
    global _NAX_AVAILABLE
    if _NAX_AVAILABLE is None:
        try:
            from omlx.custom_kernels.nax import is_nax_available

            _NAX_AVAILABLE = bool(is_nax_available())
        except Exception:  # noqa: BLE001 - no tensor units: keep MLX's kernels
            _NAX_AVAILABLE = False
    return _NAX_AVAILABLE


def _nax_prefill_ok(module, rows: int, width: int, inject) -> bool:
    """Whether the tensor-unit kernels reproduce this call bit for bit.

    They repeat MLX's unsplit NAX quantized matmul, so the up projection must
    take that kernel in MLX too, every projection must use the tile loop's
    group size, and the inject partials need the tail kernel's whole-pack
    thread slices.
    """
    return (
        not _NAX_DISABLED
        and not _NAX_BROKEN
        and inject is not None
        and width % (256 * _pack_factor(inject.bits)) == 0
        and all(
            projection.group_size == hc_prefill_nax.GROUP_SIZE
            for projection in (
                module.input_mix_weight_down,
                module.input_mix_weight_up,
                inject,
            )
        )
        and module.hidden_size % hc_prefill_nax.UP_COLS == 0
        and hc_prefill_nax.plain_qmm_nax(rows, width)
        and _nax_available()
    )


def _nax_prefill(module, hyper_input, flat, write, rows, hc, hidden, dtype, inject):
    """Three-dispatch prefill on the tensor units; None to use the MLX path."""
    global _NAX_BROKEN
    lead = hyper_input.shape[:-1]
    pending = None
    if write is not None:
        pending = _write_operands(hyper_input, write, hidden, hc)
        if pending is None:
            return None
    down, up = module.input_mix_weight_down, module.input_mix_weight_up
    try:
        written, normed, inj_part = hc_prefill_nax.norm_inject(
            module, flat, pending, rows, hc, hidden, dtype, _eps_array(module), inject
        )
        fused_down = hc_prefill_nax.plain_qmm_nax(rows, module.hc_lowrank)
        if fused_down:
            act = hc_prefill_nax.down_silu(normed, down, rows, hc)
        else:
            # MLX splits K for this few row tiles: keep its kernel.
            act = nn.silu(down(normed) / hc)
        mixed, injection = hc_prefill_nax.up_tail(
            act, up, normed, inj_part, rows, hc, hidden
        )
        passthrough = (
            hyper_input if written is None else written.reshape(hyper_input.shape)
        )
        out = (
            mixed.reshape(*lead, hidden),
            passthrough,
            injection.reshape(*lead, hc),
        )
        signature = (
            "prefill_nax",
            dtype,
            hc,
            hidden,
            module.hc_lowrank,
            down.bits,
            up.bits,
            inject.bits,
            write is not None,
            fused_down,
        )
        if signature not in _VALIDATED:
            # The kernels repeat MLX's quantized matmul arithmetic: check the
            # first call of each specialization against the MLX path bit for
            # bit (an MLX with a different qmm would fail here) and keep the
            # MLX path for good if they differ.
            reference = prefill_forward(module, hyper_input, write, nax=False)
            mx.eval(out, reference)
            if not all(
                mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()
                for a, b in zip(out, reference)
            ):
                _NAX_BROKEN = True
                logger.warning(
                    "Qwen4 tensor-unit hyper-connection prefill kernels differ "
                    "from this MLX's quantized matmul; using the MLX matmul path"
                )
                return reference
            _VALIDATED.add(signature)
    except Exception as exc:  # noqa: BLE001 - optional native path
        _NAX_BROKEN = True
        logger.warning(
            "Qwen4 tensor-unit hyper-connection prefill kernels failed; using "
            "the MLX matmul path: %s",
            exc,
        )
        return None
    return out


def prefill_forward(module, hyper_input, write=None, nax=True):
    """Prefill with the fused stream norm and tail/inject epilogue; None on failure.

    ``write`` is a pending ``(branch, gate)`` residual write onto ``hyper_input``.
    The norm kernel applies it and returns the written stream as the passthrough;
    a module without inject weights then returns ``(mixed, written)``. With
    ``nax`` the projections run in hc_prefill_nax's kernels where they
    reproduce this path bit for bit.
    """
    global _FAILURE_LOGGED
    try:
        hc, hidden = module.hc_count, module.hidden_size
        width = hc * hidden
        dtype = hyper_input.dtype
        rows = _rows_of(hyper_input)
        flat = hyper_input.reshape(rows, width)
        inject = module.block_inject_weight if "block_inject_weight" in module else None
        if nax and _nax_prefill_ok(module, rows, width, inject):
            fused = _nax_prefill(
                module, hyper_input, flat, write, rows, hc, hidden, dtype, inject
            )
            if fused is not None:
                return fused
        if write is None:
            normed = _kernel_norm(module, flat, rows, hc, hidden, dtype, precise=True)
        else:
            operands = _write_operands(hyper_input, write, hidden, hc)
            if operands is None:
                return None
            written, normed = _kernel_write_norm(
                module, flat, *operands, rows, hc, hidden, dtype
            )
            hyper_input = written.reshape(hyper_input.shape)
        normed = normed.reshape(hyper_input.shape)
        mix = _act(hc)(module.input_mix_weight_down(normed))
        up = module.input_mix_weight_up(mix)
        tail_launch = None
        tail_face = None
        if inject is not None:
            # The face and the launch depend on the layer's layout and the
            # input's dtype alone: resolve them once per module and dtype.
            plan = module.__dict__.get("_hc_tail_inject")
            if plan is None or plan[0] is not inject or plan[1] != dtype:
                face = _decode_face(inject, True)
                tail_launch = _tail_inject_launch(hc, hidden, dtype, face)
                tail_face = face if tail_launch is not None else None
                module.__dict__["_hc_tail_inject"] = (
                    inject,
                    dtype,
                    tail_launch,
                    tail_face,
                )
            else:
                tail_launch, tail_face = plan[2], plan[3]
        if tail_launch is None:
            mixed = _tail(hc, hidden)(up, normed)
            injection = None if inject is None else 2 * mx.sigmoid(inject(normed) / hc)
        else:
            mixed, injection = tail_launch(
                rows,
                up.reshape(rows, width),
                normed.reshape(rows, width),
                inject,
            )
            mixed = mixed.reshape(*hyper_input.shape[:-1], hidden)
            injection = injection.reshape(*hyper_input.shape[:-1], hc)
        signature = ("prefill", dtype, hc, hidden, write is not None, tail_face)
        if signature not in _VALIDATED:
            mx.eval(mixed) if injection is None else mx.eval(mixed, injection)
            _VALIDATED.add(signature)
        if injection is None:
            return mixed if write is None else (mixed, hyper_input)
        return mixed, hyper_input, injection
    except Exception as exc:  # noqa: BLE001 - optional native path
        if not _FAILURE_LOGGED:
            _FAILURE_LOGGED = True
            logger.warning(
                "Qwen4 fused hyper-connection kernels failed closed; using the "
                "canonical path: %s",
                exc,
            )
        return None


def fused_forward(module, hyper_input, write=None):
    """Return fused outputs, or None on construction or first-evaluation failure.

    ``write`` is a pending ``(branch, gate)`` residual write onto ``hyper_input``,
    applied inside the norm kernel as in :func:`prefill_forward`; the written
    stream is the passthrough, and a module without inject weights returns
    ``(mixed, written)``.
    """
    global _FAILURE_LOGGED
    try:
        hc, hidden, lowrank = module.hc_count, module.hidden_size, module.hc_lowrank
        width = hc * hidden
        batch, seq, _ = hyper_input.shape
        rows = batch * seq
        down, up = module["input_mix_weight_down"], module["input_mix_weight_up"]
        inject = module.get("block_inject_weight")
        flat = hyper_input.reshape(rows, width)
        dtype = hyper_input.dtype
        operands = None
        if write is not None:
            operands = _write_operands(hyper_input, write, hidden, hc)
            if operands is None:
                return None
        down_face = _decode_face(down, False)
        up_face = _decode_face(up, False)
        inject_face = _decode_face(inject, True) if inject is not None else None
        launch = _decode_launch(
            rows,
            hc,
            hidden,
            lowrank,
            dtype,
            down_face,
            up_face,
            inject_face,
            write is not None,
        )
        written, mixed, injection = launch(module, flat, operands, down, up, inject)
        if written is not None:
            hyper_input = written.reshape(hyper_input.shape)
        if launch not in _VALIDATED:
            # Metal compilation is lazy. Validate once, without synchronizing
            # subsequent layers or decode steps using the same specialization.
            if inject is None:
                mx.eval(mixed)
            else:
                mx.eval(mixed, injection)
            _VALIDATED.add(launch)
        mixed = mixed.reshape(batch, seq, hidden)
        if inject is None:
            return mixed if write is None else (mixed, hyper_input)
        return mixed, hyper_input, injection.reshape(batch, seq, hc)
    except Exception as exc:  # noqa: BLE001 - optional native path
        if not _FAILURE_LOGGED:
            _FAILURE_LOGGED = True
            logger.warning(
                "Qwen4 fused hyper-connection kernels failed closed; using the "
                "canonical path: %s",
                exc,
            )
        return None


__all__ = [
    "MAX_ROWS",
    "compatible",
    "enabled",
    "fused_forward",
    "prefill_compatible",
    "prefill_forward",
    "write_enabled",
]
