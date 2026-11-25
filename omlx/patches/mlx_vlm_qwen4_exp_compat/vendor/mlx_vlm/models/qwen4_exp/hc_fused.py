# SPDX-License-Identifier: Apache-2.0
"""Metal kernels for Qwen4 hyper-connection inputs.

Decode (at most 16 rows) fuses per-stream RMS norm, down/inject projections
with activation, and the up projection with stream mixing. Prefill fuses the
stream norm, silu(down / HC), and the mixing/inject epilogue around the
down/up projections; on GPUs with tensor units (M5) those projections run in
hc_prefill_nax's kernels with the epilogues folded in (currently unreachable
while NAX eligibility is False, DEFERRED 59), elsewhere each ``FormatLinear``
down/up projection runs its own route.

Both decode and prefill need four streams and ``FormatLinear`` projections:
built-in affine group-size-32 or -64 with 4/5/6/8-bit weights (scales/biases
in the norm dtype), or a plugin format's exact dot face for that dtype whose
``vpt`` divides ``hidden_size`` (down/inject) or ``hc_lowrank`` (up) and whose
combined staged tables fit ``formats.TABLE_BYTES``. Served compatibility
(``compatible`` / ``prefill_compatible``) and ``prefill_forward`` require
BF16 or FP16 rows matching the norm dtype; ``fused_forward`` also accepts
FP32 plans (including same-dtype pending writes) for direct oracle checks.

Closed-plan contract (rule 28): each layer's HC plan (faces, pre-read
weight tuples, norm weight, eps, prebuilt decode/prefill launches, and
compiled read checks) is built ONCE as an immutable ``_HCPlan`` tuple by the
load patch's post-eval hook (:func:`rebuild_plans` after the loader's
``model.eval()``), or lazily on first use under ``_PLAN_LOCK`` for directly
constructed modules and direct ``load_weights`` callers. Weights,
projections, norm, and geometry are
immutable after plan build; any post-load mutation requires calling
:func:`rebuild_plan` (or :func:`rebuild_plans` on the owning model), and
un-rebuilt post-load mutations to any recorded plan read fail closed with
``ValueError`` naming ``rebuild_plan``. ``fused_forward`` and
``prefill_forward`` also validate ``hyper_input``'s type, shape, dtype, and
row count against the plan up front and raise ``ValueError`` on mismatch.

A pending residual write (previous block's branch times its injection gate)
can be applied inside the stream norm, which also stores the written
residual; it rounds exactly like the eager multiply and add.

Each kernel specialization is evaluated once; a kernel that fails to build
or launch raises ``FaceKernelError`` naming its face (eligibility declines
return None before any kernel exists). Where the geometry allows, small
decode batches take two launches in place of the three-launch kernels (norm,
down/inject, up/mix); both produce the same bits.
"""

from __future__ import annotations

import functools
import logging
import math
import threading
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from omlx import formats
from omlx.formats import affine, kernels
from omlx.formats.linear import FormatLinear

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

from . import hc_prefill_nax  # noqa: E402 - hc_prefill_nax imports _HEADER

logger = logging.getLogger(__name__)

MAX_ROWS = 16
# Each projection's group size is a template parameter (GS_D, GS_I, GS_U): the
# traversal is the same for every group size, only the scale/bias index moves.
_GROUP_SIZES = (32, 64)
_SUPPORTED_BITS = (4, 5, 6, 8)
_DTYPES = (mx.bfloat16, mx.float16)
_PLAN_DTYPES = (mx.bfloat16, mx.float16, mx.float32)
_PLAN_ATTR = "_omlx_hc_plan"
_PLAN_LOCK = threading.Lock()
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
_VALIDATED_LOCK = threading.Lock()
_INELIGIBLE_LOGGED = False


class _HCPlan(NamedTuple):
    """Immutable per-layer hyper-connection execution plan (rule 28)."""

    dtype: mx.Dtype
    hc: int
    hidden: int
    lowrank: int
    width: int
    norm_weight: mx.array
    eps: mx.array
    down: FormatLinear
    up: FormatLinear
    inject: FormatLinear | None
    faces: tuple
    down_inj_weights: tuple[mx.array, ...]
    up_weights: tuple[mx.array, ...]
    inj_weights: tuple[mx.array, ...]
    decode_launches: tuple[tuple[object, object], ...]
    tail_launch: object
    tail_face: object
    reads: tuple
    id_attrs: tuple
    val_attrs: tuple
    id_items: tuple
    members: tuple
    operand_dtypes: tuple
    prefill_signatures: tuple


class FaceKernelError(RuntimeError):
    """A kernel over a face failed to build or launch: a defect, not an
    eligibility decision, so the fused path raises instead of routing around
    it. Eligibility (vpt, widths, tables) declines before any kernel exists."""


def _run_named(what: str, call):
    """Run a kernel build/launch, naming it and its Metal error on failure."""
    try:
        return call()
    except FaceKernelError:
        raise
    except Exception as error:
        raise FaceKernelError(f"{what} failed to build or launch: {error}") from error


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
    if constexpr (INJ != 0) {
        if (rg != 0) return;
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
    } else if constexpr (INJ != 0) {
        if (int(sg) < HC) {
            const int r = int(sg);
            float result[1] = {0.0f};
            hc_slice<K, SLICE>(inject, r, ks, lane, xs, result);
            if (lane == 0) pp[R + r] = result[0];
        }
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
    const bool inj_thread = threadgroup_position_in_grid.y == 0 && int(t) < HC;
    T injv = T(0.0f);
    if constexpr (INJ != 0) {
        if (inj_thread) {
            const int i = R + int(t);
            float v = pp[i] + pp[PR + i] + pp[2 * PR + i] + pp[3 * PR + i];
            v = v / float(HC);
            injv = T(2.0f / (1.0f + metal::exp(-v)));
        }
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
    if constexpr (INJ != 0) {
        if (inj_thread) inj[(size_t)r * HC + int(t)] = injv;
    }
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
        kernel = _KERNELS.setdefault(
            name,
            mx.fast.metal_kernel(
                name=name,
                input_names=input_names,
                output_names=output_names,
                header=header,
                source=source,
                ensure_row_contiguous=True,
            ),
        )
    return kernel


def enabled() -> bool:
    return not _DISABLED


def write_enabled() -> bool:
    """Whether residual writes may be deferred into the next fused stream norm."""
    return not (_DISABLED or _WRITE_DISABLED)


def _affine_tensors_ok(projection, dtype) -> bool:
    """MLX's own affine layout in the hc_norm dtype, at the widths H1 serves."""
    group_size = getattr(projection, "group_size", None)
    bits = getattr(projection, "bits", None)
    return (
        type(group_size) is int
        and group_size in _GROUP_SIZES
        and type(bits) is int
        and bits in _SUPPORTED_BITS
        and "bias" not in projection
        and isinstance(getattr(projection, "weight", None), mx.array)
        and projection.weight.dtype == mx.uint32
        and isinstance(getattr(projection, "scales", None), mx.array)
        and isinstance(getattr(projection, "biases", None), mx.array)
        and projection.scales.dtype == dtype
        and projection.biases.dtype == dtype
    )


def _projection_face(projection, k: int, n: int, dtype, inject: bool):
    """The dot face of one projection's ``(n, k)`` matrix, or None when the
    fused kernels cannot take it.

    A built-in affine layer keeps H1's lane face (not ``AffineFormat.dot``):
    the up and inject shapes need its exact traversal. A plugin format's own
    face is taken when it is exact in this dtype and has lanes; its codes are
    read by its own name (D11)."""
    if projection is None or type(projection) is not FormatLinear:
        return None
    projection.refuse_placeholders()
    if formats.input_transform(projection, refuse=False) is not None:
        return None
    if (
        getattr(projection, "input_dims", None) != k
        or getattr(projection, "output_dims", None) != n
    ):
        return None
    if getattr(projection, "mode", None) == "affine":
        if not _affine_tensors_ok(projection, dtype):
            return None
        return affine.face(
            projection.bits,
            projection.group_size,
            not inject,
            dtype,
            partial_blocks=not inject,
        )
    if "bias" in projection:
        return None
    face = formats.dot(projection, k, n)
    if face is None or not face.exact_for(dtype) or face.vpt < 1:
        return None
    return face


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


def compatible(module, hyper_input) -> bool:
    """Whether ``module`` (a Qwen4ExpGatedResidual) can take the fused decode
    path for ``hyper_input``."""
    # Runs for every hyper-connection of every decode step: the row gate
    # first (prefill rows must not reach the plan checks), then the cached
    # immutable plan verdict.
    if _DISABLED or not isinstance(hyper_input, mx.array):
        return False
    shape = hyper_input.shape
    if not (len(shape) == 3 and 1 <= shape[0] * shape[1] <= MAX_ROWS):
        return False
    return _plan_compatible(module, hyper_input)


def prefill_compatible(module, hyper_input) -> bool:
    """Whether rows above MAX_ROWS can use the compiled prefill path."""
    if _DISABLED or not isinstance(hyper_input, mx.array):
        return False
    shape = hyper_input.shape
    if not (len(shape) == 3 and shape[0] * shape[1] > MAX_ROWS):
        return False
    return _plan_compatible(module, hyper_input)


def _plan_compatible(module, hyper_input) -> bool:
    plan = _get_plan(module)
    if plan is False or plan.dtype not in _DTYPES or hyper_input.dtype != plan.dtype:
        return False
    if hyper_input.shape[2] != plan.width:
        return _ineligible(f"input width {hyper_input.shape[2]}")
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _geometry_norm_ok(hc, hidden, lowrank, norm_weight, norm_group, eps_value) -> bool:
    """Geometry and norm layout rules over values the build read, so every
    value they depend on is recorded for the warm check."""
    if not (
        type(hc) is int
        and type(hidden) is int
        and type(lowrank) is int
        and hc == 4
        and hidden > 0
        and lowrank > 0
        # The up grid and quantization groups require 64-element alignment.
        and hidden % 64 == 0
        and lowrank % 64 == 0
    ):
        return False
    return (
        type(norm_group) is int
        and norm_group == hidden
        and isinstance(eps_value, (int, float))
        and not isinstance(eps_value, bool)
        and math.isfinite(eps_value)
        and isinstance(norm_weight, mx.array)
        and norm_weight.dtype in _PLAN_DTYPES
        and norm_weight.shape == (hc * hidden,)
    )


def _layout_ok(
    hidden,
    lowrank,
    down_face,
    up_face,
    inj_face,
    has_inject: bool,
) -> bool:
    """Face layout validator: each face's presence and vpt divisibility, and
    the combined table budget."""
    if down_face is None or up_face is None or (inj_face is None and has_inject):
        return False
    if (
        hidden % down_face.vpt
        or lowrank % up_face.vpt
        or (inj_face is not None and hidden % inj_face.vpt)
    ):
        return False
    return (
        kernels.table_bytes(
            down_face, up_face, *((inj_face,) if inj_face is not None else ())
        )
        <= formats.TABLE_BYTES
    )


_EPS_CACHE: dict[float, mx.array] = {}


def _eps_array(module_or_value) -> mx.array:
    """The evaluated 1-element float32 eps tensor for one eps value (or a
    module's ``hc_norm.eps``).

    Shared across layers with the same eps (one GPU sync per distinct value),
    immutable once built."""
    value = (
        float(module_or_value)
        if isinstance(module_or_value, (int, float))
        else float(module_or_value.hc_norm.eps)
    )
    eps = _EPS_CACHE.get(value)
    if eps is None:
        fresh = mx.array([value], dtype=mx.float32)
        mx.eval(fresh)
        eps = _EPS_CACHE.setdefault(value, fresh)
    return eps


class _PlanReads:
    """Every layer value the plan build reads, in read order.

    Each read records ``(owner, kind, name, value)``; the warm check re-reads
    exactly this tuple (objects and bools by ``is``, int/float/str scalars by
    exact type and ``==``), so a value the build starts to depend on is
    covered without a hand-written list."""

    __slots__ = ("reads",)

    def __init__(self):
        self.reads = []

    def get(self, owner, name):
        value = getattr(owner, name, None)
        self.reads.append((owner, "get", name, value))
        return value

    def item(self, owner, name):
        value = owner[name]
        self.reads.append((owner, "item", name, value))
        return value

    def member(self, owner, name):
        value = name in owner
        self.reads.append((owner, "member", name, value))
        return value


_SCALAR_TYPES = (int, float, str)
_PLAN_STALE_MSG = (
    "the layer's HC projections, norm or geometry changed after "
    "its plan was built; call rebuild_plan(module) (the in-tree "
    "loaders do this at the end of their change)"
)


def _compile_plan_checks(recorded):
    """Split recorded plan reads into direct identity and typed-scalar tuples
    so warm forwards validate without per-entry lambdas or helper frames."""
    id_attrs = []
    val_attrs = []
    id_items = []
    members = []
    for owner, kind, name, value in recorded:
        if kind == "get":
            vtype = type(value)
            if vtype in _SCALAR_TYPES:
                val_attrs.append((owner, name, vtype, value))
            else:
                id_attrs.append((owner, name, value))
        elif kind == "item":
            id_items.append((owner, name, value))
        else:
            members.append((owner, name, value))
    item_keys = {(id(owner), name) for owner, name, _ in id_items}
    members = [
        entry
        for entry in members
        if not (entry[2] is True and (id(entry[0]), entry[1]) in item_keys)
    ]
    return (
        tuple(id_attrs),
        tuple(val_attrs),
        tuple(id_items),
        tuple(members),
    )


def _check_plan(plan: _HCPlan) -> None:
    """Fail closed after a post-load mutation: every recorded read of the
    build is re-read and compared (objects and bools by identity, int/float/str
    scalars by exact type and value). One sweep, then a clear error naming
    rebuild_plan."""
    stale = False
    try:
        for owner, name, expected in plan.id_attrs:
            if getattr(owner, name, None) is not expected:
                stale = True
                break
        if not stale:
            for owner, name, expected_type, expected in plan.val_attrs:
                now = getattr(owner, name, None)
                if type(now) is not expected_type or now != expected:
                    stale = True
                    break
        if not stale:
            for owner, name, expected in plan.id_items:
                if owner[name] is not expected:
                    stale = True
                    break
        if not stale:
            for owner, name, expected in plan.members:
                if (name in owner) is not expected:
                    stale = True
                    break
    except KeyError:
        stale = True
    if stale:
        raise ValueError(_PLAN_STALE_MSG)


def _read_tensors(reads, layer, face, codes_name):
    used = (codes_name, *face.tensors)
    if not all(reads.member(layer, name) for name in used):
        return None
    out = tuple(reads.item(layer, name) for name in used)
    for name in formats.tensor_names(layer):
        if name not in used:
            if not reads.member(layer, name):
                return None
            reads.item(layer, name)
    return out


def _build_plan(module) -> _HCPlan | bool:
    """Build the immutable ``_HCPlan`` for ``module``, or ``False`` when
    ineligible. Every layer value is read through ``_PlanReads`` so the warm
    check re-reads exactly what the build touched (rule 28); called once per
    layer at load time or lazily under ``_PLAN_LOCK``."""
    from .language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    reads = _PlanReads()
    if reads.get(module, "__class__") is not Qwen4ExpGatedResidual:
        return _ineligible("module layout (Qwen4ExpGatedResidual required)")
    inject_present = reads.member(module, "input_inject_weight")
    if inject_present:
        return _ineligible("combined input projection layout")
    hc_count = reads.get(module, "hc_count")
    hidden = reads.get(module, "hidden_size")
    lowrank = reads.get(module, "hc_lowrank")
    norm = reads.get(module, "hc_norm")
    if reads.get(norm, "__class__") is not Qwen4ExpRMSNorm:
        return _ineligible("hc_norm layout (Qwen4ExpRMSNorm required)")
    norm_weight = reads.get(norm, "weight")
    norm_group = reads.get(norm, "group_size")
    eps_value = reads.get(norm, "eps")
    if not _geometry_norm_ok(
        hc_count,
        hidden,
        lowrank,
        norm_weight,
        norm_group,
        eps_value,
    ):
        return _ineligible(
            f"geometry/hc_norm layout hidden_size={hidden} "
            f"hc_lowrank={lowrank} hc_count={hc_count}"
        )
    dtype = norm_weight.dtype
    width = hc_count * hidden
    down = reads.get(module, "input_mix_weight_down")
    up = reads.get(module, "input_mix_weight_up")
    inject = reads.get(module, "block_inject_weight")
    for layer in (down, up, *((inject,) if inject is not None else ())):
        if reads.get(layer, "__class__") is not FormatLinear:
            return _ineligible("projection layout (FormatLinear required)")
        layer.refuse_placeholders()
        if reads.member(layer, "bias"):
            return _ineligible("biased projections (the fused kernels add no bias)")
        for attr in (
            "input_dims",
            "output_dims",
            "mode",
            "bits",
            "group_size",
            "_tensors",
        ):
            reads.get(layer, attr)
        active = reads.get(layer, "fmt")
        if not isinstance(active, formats.QuantFormat):
            return _ineligible("projection format (QuantFormat required)")
        if active.training_sensitive:
            reads.get(layer, "_training")
    down_face = _projection_face(down, width, lowrank, dtype, False)
    up_face = _projection_face(up, lowrank, width, dtype, False)
    inj_face = (
        _projection_face(inject, width, hc_count, dtype, True)
        if inject is not None
        else None
    )
    if not _layout_ok(
        hidden,
        lowrank,
        down_face,
        up_face,
        inj_face,
        inject is not None,
    ):
        return _ineligible(
            "projection layout (FormatLinear affine 4/5/6/8-bit group-size-32/64 "
            "with scales in the hc_norm dtype, or a plugin face exact in it)"
        )
    down_weights = _read_tensors(reads, down, down_face, formats.tensor_names(down)[0])
    up_weights = _read_tensors(reads, up, up_face, formats.tensor_names(up)[0])
    inj_weights = (
        _read_tensors(reads, inject, inj_face, formats.tensor_names(inject)[0])
        if inject is not None
        else ()
    )
    if down_weights is None or up_weights is None or inj_weights is None:
        return _ineligible("missing projection tensor")
    down_inj_weights = down_weights + inj_weights
    operand_dtypes = tuple(
        value.dtype
        for owner, kind, name, value in reads.reads
        if kind == "item" and isinstance(value, mx.array)
    )
    if not mx.metal.is_available():
        return _ineligible("Metal unavailable")
    eps = _eps_array(float(eps_value))
    decode_launches = tuple(
        (
            _decode_launch(
                rows,
                hc_count,
                hidden,
                lowrank,
                dtype,
                down_face,
                up_face,
                inj_face,
                False,
                operand_dtypes,
            ),
            _decode_launch(
                rows,
                hc_count,
                hidden,
                lowrank,
                dtype,
                down_face,
                up_face,
                inj_face,
                True,
                operand_dtypes,
            ),
        )
        for rows in range(1, MAX_ROWS + 1)
    )
    tail_launch = (
        _tail_inject_launch(hc_count, hidden, dtype, inj_face)
        if dtype in _DTYPES
        else None
    )
    tail_face = inj_face if tail_launch is not None else None
    id_attrs, val_attrs, id_items, members = _compile_plan_checks(reads.reads)
    return _HCPlan(
        dtype=dtype,
        hc=hc_count,
        hidden=hidden,
        lowrank=lowrank,
        width=width,
        norm_weight=norm_weight,
        eps=eps,
        down=down,
        up=up,
        inject=inject,
        faces=(down_face, up_face, inj_face),
        down_inj_weights=down_inj_weights,
        up_weights=up_weights,
        inj_weights=inj_weights,
        decode_launches=decode_launches,
        tail_launch=tail_launch,
        tail_face=tail_face,
        reads=tuple(reads.reads),
        operand_dtypes=operand_dtypes,
        prefill_signatures=tuple(
            (
                "prefill",
                dtype,
                hc_count,
                hidden,
                write,
                tail_face,
                operand_dtypes if tail_face is not None else (),
            )
            for write in (False, True)
        ),
        id_attrs=id_attrs,
        val_attrs=val_attrs,
        id_items=id_items,
        members=members,
    )


def _get_plan(module) -> _HCPlan | bool:
    """Return ``module``'s immutable HC plan, building it once under
    ``_PLAN_LOCK`` if not yet built."""
    plan = module.__dict__.get(_PLAN_ATTR)
    if plan is not None:
        return plan
    with _PLAN_LOCK:
        plan = module.__dict__.get(_PLAN_ATTR)
        if plan is not None:
            return plan
        plan = _build_plan(module)
        module.__dict__[_PLAN_ATTR] = plan
        return plan


def rebuild_plan(module) -> _HCPlan | bool:
    """Build and store ``module``'s immutable HC plan under ``_PLAN_LOCK``.

    Weights, projections, norm, and geometry are immutable after plan build;
    call this after any intentional post-load mutation. Any previous plan is
    cleared before rebuilding so a failing build never leaves a stale plan.
    """
    with _PLAN_LOCK:
        module.__dict__.pop(_PLAN_ATTR, None)
        plan = _build_plan(module)
        module.__dict__[_PLAN_ATTR] = plan
        return plan


def _has_placeholders(module) -> bool:
    for name in (
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
    ):
        proj = getattr(module, name, None)
        if type(proj) is FormatLinear and getattr(proj, "_placeholders", None):
            return True
    return False


def rebuild_plans(model) -> int:
    """Build immutable HC plans for every hyper-connection layer in ``model``.

    Called by the load patch after the loader's ``model.eval()`` so every
    layer's plan is built once at the end of a load (rule 28). Layers whose
    projections still hold
    unfilled load placeholders have any stale plan cleared and are skipped so
    ``format_load`` reports the missing tensor; calling them directly still
    raises ``ValueError``.
    """
    candidates = [model]
    named = getattr(model, "named_modules", None)
    if named is not None:
        candidates.extend(mod for _, mod in named() if mod is not model)
    seen: set[int] = set()
    built = 0
    for mod in candidates:
        ident = id(mod)
        if ident in seen:
            continue
        seen.add(ident)
        if not (
            isinstance(mod, nn.Module)
            and hasattr(mod, "hc_norm")
            and hasattr(mod, "hc_lowrank")
        ):
            continue
        if _has_placeholders(mod):
            with _PLAN_LOCK:
                mod.__dict__.pop(_PLAN_ATTR, None)
            continue
        if rebuild_plan(mod) is not False:
            built += 1
    return built


def _validate_input(plan: _HCPlan, hyper_input, *, max_rows: int | None) -> int:
    """Validate ``hyper_input`` against ``plan`` up front; raise ValueError on
    any shape, dtype, or row-count mismatch."""
    if not isinstance(hyper_input, mx.array):
        raise ValueError(
            f"hyper_input must be an mx.array, got {type(hyper_input).__name__}"
        )
    shape = hyper_input.shape
    if len(shape) != 3:
        raise ValueError(
            f"hyper_input must be 3-D (batch, seq, width), got shape {tuple(shape)}"
        )
    if shape[2] != plan.width:
        raise ValueError(
            f"hyper_input width {shape[2]} does not match "
            f"plan width {plan.width} ({plan.hc} * {plan.hidden})"
        )
    if hyper_input.dtype != plan.dtype:
        raise ValueError(
            f"hyper_input dtype {hyper_input.dtype} does not match "
            f"plan dtype {plan.dtype}"
        )
    rows = shape[0] * shape[1]
    if rows < 1:
        raise ValueError(f"hyper_input must have at least 1 row, got {rows}")
    if max_rows is not None and rows > max_rows:
        raise ValueError(f"fused_forward accepts at most {max_rows} rows, got {rows}")
    return rows


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


def _kernel_norm(norm_weight, eps, flat, rows, hc, hidden, dtype, precise=False):
    width = hc * hidden
    name = "omlx_qwen4_hc_prefill_norm" if precise else "omlx_qwen4_hc_fused_norm"
    return _kernel(
        name, ["x", "w", "eps"], ["xn"], _NP_SOURCE if precise else _N_SOURCE
    )(
        inputs=[flat, norm_weight, eps],
        template=[("T", dtype), ("K", width), ("H", hidden)],
        grid=(256, hc, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[dtype],
    )[
        0
    ]


def _write_operands(hyper_input, write, rows: int, hidden: int, hc: int):
    """Flattened (branch, gate) for the write-norm kernels, or None if they
    cannot take it.

    Mixed dtypes would promote the eager write, so only same-dtype operands
    qualify (including BF16, FP16, and FP32).
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
    return branch.reshape(rows, hidden), gate.reshape(rows, hc)


def _kernel_write_norm(
    norm_weight, eps, flat, branch, gate, rows, hc, hidden, dtype, precise=True
):
    width = hc * hidden
    return _kernel(
        (
            "omlx_qwen4_hc_prefill_write_norm"
            if precise
            else "omlx_qwen4_hc_fused_write_norm"
        ),
        ["x", "branch", "gate", "w", "eps"],
        ["y_out", "xn"],
        _WN_SOURCE if precise else _WND_SOURCE,
    )(
        inputs=[flat, branch, gate, norm_weight, eps],
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


@functools.cache
def _gpu_class() -> str:
    try:
        return str(mx.device_info().get("architecture", ""))[-1:]
    except Exception:
        return ""


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


def _kernel_norm_down(launch, plan: _HCPlan, flat, operands):
    if operands is None:
        xn, parts = launch(
            inputs=[flat, plan.norm_weight, plan.eps, *plan.down_inj_weights]
        )
        return None, xn, parts
    return launch(
        inputs=[
            flat,
            operands[0],
            operands[1],
            plan.norm_weight,
            plan.eps,
            *plan.down_inj_weights,
        ]
    )


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
    when the face does not fit. Held inside the layer's immutable plan."""
    if inject_face is None or not _tail_inject_ok(
        hc * hidden, hidden, hc, dtype, inject_face
    ):
        return None
    width = hc * hidden
    template = [("T", dtype), ("K", width), ("H", hidden), ("HC", hc)]
    dtypes = [dtype, dtype]
    what = f"the {inject_face.key} face's prefill tail-inject kernel"
    kernel = _run_named(
        what,
        lambda: kernels.face_kernel(
            "omlx_qwen4_hc_prefill_tail_inject",
            ("up", "xn", "inject"),
            {"inject": inject_face},
            ("mixed", "inj"),
            "",
            _TAIL_INJECT,
        ),
    )
    # The fixed keys are bound once, so a warm call passes only what varies
    # with its rows (land review NIT-1).
    native = functools.partial(
        kernel.native,
        template=template,
        threadgroup=(256, 1, 1),
        output_dtypes=dtypes,
    )
    recorded = [False]  # the first call records the face kernel's instance

    def run(rows, up, normed, inject_weights):
        try:
            if recorded[0]:
                return native(
                    inputs=[up, normed, *inject_weights],
                    grid=(256, 1, rows),
                    output_shapes=[(rows, hidden), (rows, hc)],
                )
            out = kernel(
                inputs=[up, normed, *inject_weights],
                template=template,
                grid=(256, 1, rows),
                threadgroup=(256, 1, 1),
                output_shapes=[(rows, hidden), (rows, hc)],
                output_dtypes=dtypes,
            )
        except FaceKernelError:
            raise
        except Exception as error:
            raise FaceKernelError(
                f"{what} failed to build or launch: {error}"
            ) from error
        recorded[0] = True
        return out

    def _warm_entry(plan: _HCPlan):
        input_specs = (
            ((1, width), dtype),
            ((1, width), dtype),
            *((a.shape, a.dtype) for a in plan.inj_weights),
        )
        inst = kernel.instance(input_specs, dtypes, template=template)
        stand_in = kernels.StandIn(
            input_shapes=tuple(shape for shape, _ in input_specs),
            output_shapes=((1, hidden), (1, hc)),
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
        )
        return inst, (lambda s=stand_in: s), what

    def _mark_recorded() -> None:
        recorded[0] = True

    run._warm_entry = _warm_entry
    run._mark_recorded = _mark_recorded
    return run


@functools.cache
def _decode_launch(
    rows,
    hc,
    hidden,
    lowrank,
    dtype,
    down_face,
    up_face,
    inj_face,
    write,
    operand_dtypes,
):
    """Choose the decode form and specialize its prebuilt launches for one
    geometry and layout. Shared across layers and threads; warm calls invoke
    each kernel's ``native`` partial directly after first recording (OPT-1).
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
    if inject:
        matrices["inject"] = inj_face

    shape = [("T", dtype), ("K", width), ("R", lowrank), ("HC", hc)]
    faces = f"{down_face.key} / {up_face.key}"
    if inject:
        faces += f" / {inj_face.key}"

    norm_fn = None
    norm_kernel = None
    norm_kwargs = None
    try:
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
            down_kwargs = dict(
                template=[*shape, ("H", hidden), ("INJ", int(inject)), ("RPS", rps)],
                grid=(256, hc * _down_row_groups(lowrank, inject, rps), rows),
                threadgroup=(256, 1, 1),
                output_shapes=shapes,
                output_dtypes=[dtype] * (1 + int(write)) + [mx.float32],
            )
            down_cold = functools.partial(down_kernel, **down_kwargs)
            down_native = functools.partial(down_kernel.native, **down_kwargs)
            threads = chunks * (_V2_UP_THREADS // chunks)
            up_kernel = kernels.face_kernel(
                "omlx_qwen4_hc_decode_up",
                ("xn", "parts", "up"),
                {"up": up_face},
                ("mixed", "inj"),
                "",
                _DECODE_UP,
            )
            up_kwargs = dict(
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
            up_cold = functools.partial(up_kernel, **up_kwargs)
            up_native = functools.partial(up_kernel.native, **up_kwargs)
        else:
            if write:
                norm_kernel = _kernel(
                    "omlx_qwen4_hc_fused_write_norm",
                    ["x", "branch", "gate", "w", "eps"],
                    ["y_out", "xn"],
                    _WND_SOURCE,
                )
                norm_kwargs = dict(
                    template=[("T", dtype), ("K", width), ("H", hidden), ("HC", hc)],
                    grid=(256, hc, rows),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(rows, width), (rows, width)],
                    output_dtypes=[dtype, dtype],
                )
            else:
                norm_kernel = _kernel(
                    "omlx_qwen4_hc_fused_norm",
                    ["x", "w", "eps"],
                    ["xn"],
                    _N_SOURCE,
                )
                norm_kwargs = dict(
                    template=[("T", dtype), ("K", width), ("H", hidden)],
                    grid=(256, hc, rows),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(rows, width)],
                    output_dtypes=[dtype],
                )
            norm_fn = functools.partial(norm_kernel, **norm_kwargs)
            down_kernel = _down_kernel(
                "omlx_qwen4_hc_fused_down",
                ("xn", *matrices),
                matrices,
                ("act", "inj"),
                _DOWN,
            )
            down_kwargs = dict(
                template=[*shape, ("INJ", int(inject))],
                grid=(32, 8 * (lowrank // 8 + int(inject)), rows),
                threadgroup=(32, 8, 1),
                output_shapes=[(rows, lowrank), (rows, hc)],
                output_dtypes=[dtype, dtype],
            )
            down_cold = functools.partial(down_kernel, **down_kwargs)
            down_native = functools.partial(down_kernel.native, **down_kwargs)
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
            up_kwargs = dict(
                template=[*shape, ("H", hidden)],
                grid=(256, hidden // 64, rows),
                threadgroup=(256, 1, 1),
                output_shapes=[(rows, hidden)],
                output_dtypes=[dtype],
            )
            up_cold = functools.partial(up_kernel, **up_kwargs)
            up_native = functools.partial(up_kernel.native, **up_kwargs)

    except FaceKernelError:
        raise
    except Exception as error:
        raise FaceKernelError(
            f"the decode faces {faces} failed to build or launch: {error}"
        ) from error

    recorded = [False]  # first successful launch records both _FaceKernel instances

    def launch(plan: _HCPlan, flat, operands):
        try:
            warm = recorded[0]
            down = down_native if warm else down_cold
            up = up_native if warm else up_cold
            if two:
                written, xn, parts = _kernel_norm_down(down, plan, flat, operands)
                mixed, injection = up(inputs=[xn, parts, *plan.up_weights])
            else:
                if write:
                    written, xn = norm_fn(
                        inputs=[
                            flat,
                            operands[0],
                            operands[1],
                            plan.norm_weight,
                            plan.eps,
                        ]
                    )
                else:
                    written = None
                    xn = norm_fn(inputs=[flat, plan.norm_weight, plan.eps])[0]
                act, injection = down(inputs=[xn, *plan.down_inj_weights])
                mixed = up(inputs=[xn, act, *plan.up_weights])[0]
        except FaceKernelError:
            raise
        except Exception as error:
            raise FaceKernelError(
                f"the decode faces {faces} failed to build or launch: {error}"
            ) from error
        if not warm:
            recorded[0] = True
        return written, mixed, injection

    what = f"the decode faces {faces} kernels"

    def _warm_entries(plan: _HCPlan):
        entries = []
        if two:
            if write:
                down_specs = (
                    ((rows, width), dtype),
                    ((rows, hidden), dtype),
                    ((rows, hc), dtype),
                    (plan.norm_weight.shape, plan.norm_weight.dtype),
                    (plan.eps.shape, plan.eps.dtype),
                    *((a.shape, a.dtype) for a in plan.down_inj_weights),
                )
            else:
                down_specs = (
                    ((rows, width), dtype),
                    (plan.norm_weight.shape, plan.norm_weight.dtype),
                    (plan.eps.shape, plan.eps.dtype),
                    *((a.shape, a.dtype) for a in plan.down_inj_weights),
                )
            down_inst = down_kernel.instance(
                down_specs,
                down_kwargs["output_dtypes"],
                template=down_kwargs["template"],
            )
            down_si = kernels.StandIn(
                input_shapes=tuple(s for s, _ in down_specs),
                output_shapes=tuple(down_kwargs["output_shapes"]),
                grid=down_kwargs["grid"],
                threadgroup=down_kwargs["threadgroup"],
            )
            entries.append((down_inst, (lambda s=down_si: s), what))

            up_specs = (
                ((rows, width), dtype),
                ((rows, hc, lowrank + hc), mx.float32),
                *((a.shape, a.dtype) for a in plan.up_weights),
            )
            up_inst = up_kernel.instance(
                up_specs,
                up_kwargs["output_dtypes"],
                template=up_kwargs["template"],
            )
            up_si = kernels.StandIn(
                input_shapes=tuple(s for s, _ in up_specs),
                output_shapes=tuple(up_kwargs["output_shapes"]),
                grid=up_kwargs["grid"],
                threadgroup=up_kwargs["threadgroup"],
            )
            entries.append((up_inst, (lambda s=up_si: s), what))
        else:
            if write:
                norm_specs = (
                    ((rows, width), dtype),
                    ((rows, hidden), dtype),
                    ((rows, hc), dtype),
                    (plan.norm_weight.shape, plan.norm_weight.dtype),
                    (plan.eps.shape, plan.eps.dtype),
                )
            else:
                norm_specs = (
                    ((rows, width), dtype),
                    (plan.norm_weight.shape, plan.norm_weight.dtype),
                    (plan.eps.shape, plan.eps.dtype),
                )
            norm_inst = kernels.planned(
                norm_kernel,
                norm_specs,
                norm_kwargs["output_dtypes"],
                template=norm_kwargs["template"],
            )
            norm_si = kernels.StandIn(
                input_shapes=tuple(s for s, _ in norm_specs),
                output_shapes=tuple(norm_kwargs["output_shapes"]),
                grid=norm_kwargs["grid"],
                threadgroup=norm_kwargs["threadgroup"],
            )
            entries.append((norm_inst, (lambda s=norm_si: s), what))

            down_specs = (
                ((rows, width), dtype),
                *((a.shape, a.dtype) for a in plan.down_inj_weights),
            )
            down_inst = down_kernel.instance(
                down_specs,
                down_kwargs["output_dtypes"],
                template=down_kwargs["template"],
            )
            down_si = kernels.StandIn(
                input_shapes=tuple(s for s, _ in down_specs),
                output_shapes=tuple(down_kwargs["output_shapes"]),
                grid=down_kwargs["grid"],
                threadgroup=down_kwargs["threadgroup"],
            )
            entries.append((down_inst, (lambda s=down_si: s), what))

            up_specs = (
                ((rows, width), dtype),
                ((rows, lowrank), dtype),
                *((a.shape, a.dtype) for a in plan.up_weights),
            )
            up_inst = up_kernel.instance(
                up_specs,
                up_kwargs["output_dtypes"],
                template=up_kwargs["template"],
            )
            up_si = kernels.StandIn(
                input_shapes=tuple(s for s, _ in up_specs),
                output_shapes=tuple(up_kwargs["output_shapes"]),
                grid=up_kwargs["grid"],
                threadgroup=up_kwargs["threadgroup"],
            )
            entries.append((up_inst, (lambda s=up_si: s), what))
        return tuple(entries)

    def _mark_recorded() -> None:
        recorded[0] = True

    launch._warm_entries = _warm_entries
    launch._mark_recorded = _mark_recorded
    return launch


_TAILS: dict[tuple[int, int], object] = {}
_ACTS: dict[int, object] = {}


def _act(hc: int):
    """Compiled silu(down / hc): one launch that rounds like the two eager ops."""
    fn = _ACTS.get(hc)
    if fn is None:
        fn = _ACTS.setdefault(hc, mx.compile(lambda y: nn.silu(y / hc), shapeless=True))
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

        fn = _TAILS.setdefault((hc, hidden), mx.compile(tail))
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


def _nax_projection_eligible(projection) -> bool:
    """Whether the tensor-unit prefill may run this projection.

    False: the loader builds FormatLinear and the tensor-unit prefill has not
    been ported to faces (DEFERRED 59 / U29, hcnax, after U26/N2). DEL3-3
    keeps this single return line; the port replaces it."""
    return False


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
        # The projections' eligibility first: a plugin's bits need not be a
        # width _pack_factor divides (DEFERRED 59 path), so no arithmetic
        # on them before it.
        and all(
            _nax_projection_eligible(projection)
            and projection.group_size == hc_prefill_nax.GROUP_SIZE
            for projection in (
                module.input_mix_weight_down,
                module.input_mix_weight_up,
                inject,
            )
        )
        and width % (256 * _pack_factor(inject.bits)) == 0
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
        pending = _write_operands(hyper_input, write, rows, hidden, hc)
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
    """Prefill with the fused stream norm and tail/inject epilogue; None when
    this layer is not eligible. Raises ``ValueError`` if ``hyper_input`` does
    not match the layer's immutable plan (shape, width, dtype, or row count),
    or if the plan is FP32. A kernel that fails to build or launch raises
    ``FaceKernelError`` naming its face (a defect is not an eligibility
    decision).

    ``write`` is a pending ``(branch, gate)`` residual write onto ``hyper_input``.
    The norm kernel applies it and returns the written stream as the passthrough;
    a module without inject weights then returns ``(mixed, written)``. With
    ``nax`` the projections run in hc_prefill_nax's kernels where they
    reproduce this path bit for bit.
    """
    plan = module.__dict__.get(_PLAN_ATTR)
    if plan is None:
        plan = _get_plan(module)
    if plan is False or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    _check_plan(plan)
    if plan.dtype not in _DTYPES:
        raise ValueError(
            f"prefill_forward requires bfloat16 or float16, got {plan.dtype}"
        )
    rows = _validate_input(plan, hyper_input, max_rows=None)
    hc = plan.hc
    hidden = plan.hidden
    width = plan.width
    dtype = plan.dtype
    flat = hyper_input.reshape(rows, width)
    inject = plan.inject
    if nax and _nax_prefill_ok(module, rows, width, inject):
        fused = _nax_prefill(
            module, hyper_input, flat, write, rows, hc, hidden, dtype, inject
        )
        if fused is not None:
            return fused
    if write is None:
        normed = _run_named(
            "the prefill stream-norm kernel",
            lambda: _kernel_norm(
                plan.norm_weight,
                plan.eps,
                flat,
                rows,
                hc,
                hidden,
                dtype,
                precise=True,
            ),
        )
    else:
        operands = _write_operands(hyper_input, write, rows, hidden, hc)
        if operands is None:
            return None
        written, normed = _run_named(
            "the prefill write-norm kernel",
            lambda: _kernel_write_norm(
                plan.norm_weight,
                plan.eps,
                flat,
                *operands,
                rows,
                hc,
                hidden,
                dtype,
            ),
        )
        hyper_input = written.reshape(hyper_input.shape)
    normed = normed.reshape(hyper_input.shape)
    mix = _act(hc)(plan.down(normed))
    up = plan.up(mix)
    tail_launch = plan.tail_launch
    tail_face = plan.tail_face
    if tail_launch is None:
        mixed = _tail(hc, hidden)(up, normed)
        injection = None if inject is None else 2 * mx.sigmoid(inject(normed) / hc)
    else:
        mixed, injection = tail_launch(
            rows,
            up.reshape(rows, width),
            normed.reshape(rows, width),
            plan.inj_weights,
        )
        mixed = mixed.reshape(*hyper_input.shape[:-1], hidden)
        injection = injection.reshape(*hyper_input.shape[:-1], hc)
    signature = plan.prefill_signatures[write is not None]
    if signature not in _VALIDATED:
        _run_named(
            f"the {tail_face.key if tail_face is not None else 'compiled'} "
            "prefill kernels' first evaluation",
            lambda: mx.eval(mixed) if injection is None else mx.eval(mixed, injection),
        )
        with _VALIDATED_LOCK:
            _VALIDATED.add(signature)
    if injection is None:
        return mixed if write is None else (mixed, hyper_input)
    return mixed, hyper_input, injection


def _validate_decode_launch(plan: _HCPlan, launch, mixed, injection) -> None:
    down_face, up_face, inject_face = plan.faces
    faces = f"{down_face.key} / {up_face.key}"
    if inject_face is not None:
        faces += f" / {inject_face.key}"
    _run_named(
        f"the decode faces {faces} kernels' first evaluation",
        lambda: mx.eval(mixed) if plan.inject is None else mx.eval(mixed, injection),
    )
    with _VALIDATED_LOCK:
        _VALIDATED.add(launch)


def fused_forward(module, hyper_input, write=None):
    """Return fused decode outputs, or None when this layer is not eligible or
    ``write`` requires eager dtype promotion. Raises ``ValueError`` if
    ``hyper_input`` does not match the layer's immutable plan (shape, width,
    dtype, or row count 1..MAX_ROWS). A kernel that fails to build or launch
    raises ``FaceKernelError`` naming its faces (a defect is not an eligibility
    decision).

    ``write`` is a pending ``(branch, gate)`` residual write onto ``hyper_input``,
    applied inside the norm kernel as in :func:`prefill_forward`; the written
    stream is the passthrough, and a module without inject weights returns
    ``(mixed, written)``. Same-dtype pending writes are supported for BF16,
    FP16, and FP32 plans.
    """
    plan = module.__dict__.get(_PLAN_ATTR)
    if plan is None:
        plan = _get_plan(module)
    if plan is False or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    _check_plan(plan)
    rows = _validate_input(plan, hyper_input, max_rows=MAX_ROWS)
    operands = None
    if write is not None:
        operands = _write_operands(hyper_input, write, rows, plan.hidden, plan.hc)
        if operands is None:
            return None
    batch, seq, _ = hyper_input.shape
    flat = hyper_input.reshape(rows, plan.width)
    launch = plan.decode_launches[rows - 1][operands is not None]
    written, mixed, injection = launch(plan, flat, operands)
    if written is not None:
        hyper_input = written.reshape(hyper_input.shape)
    if launch not in _VALIDATED:
        # Metal compilation is lazy. Validate once, without synchronizing
        # subsequent layers or decode steps using the same specialization.
        _validate_decode_launch(plan, launch, mixed, injection)
    mixed = mixed.reshape(batch, seq, plan.hidden)
    if plan.inject is None:
        return mixed if operands is None else (mixed, hyper_input)
    return mixed, hyper_input, injection.reshape(batch, seq, plan.hc)


def _iter_plans(model):
    """Yield eligible ``_HCPlan``s for every hyper-connection in ``model``."""
    candidates = [model]
    named = getattr(model, "named_modules", None)
    if named is not None:
        candidates.extend(mod for _, mod in named() if mod is not model)
    seen: set[int] = set()
    for mod in candidates:
        ident = id(mod)
        if ident in seen:
            continue
        seen.add(ident)
        if not (
            isinstance(mod, nn.Module)
            and hasattr(mod, "hc_norm")
            and hasattr(mod, "hc_lowrank")
        ):
            continue
        if _has_placeholders(mod):
            continue
        plan = _get_plan(mod)
        if plan is not False:
            yield plan


def _prefill_norm_warm_entry(plan: _HCPlan, write: bool):
    """Return ``(instance, stand_in_recipe, label)`` for one prefill norm."""
    dtype = plan.dtype
    hc = plan.hc
    hidden = plan.hidden
    width = plan.width
    if write:
        # Prefill always runs with rows > MAX_ROWS, so gate has size >= 8
        # (device pointer, not constant); rows=2 gives gate.size == 8.
        kernel = _kernel(
            "omlx_qwen4_hc_prefill_write_norm",
            ["x", "branch", "gate", "w", "eps"],
            ["y_out", "xn"],
            _WN_SOURCE,
        )
        specs = (
            ((2, width), dtype),
            ((2, hidden), dtype),
            ((2, hc), dtype),
            (plan.norm_weight.shape, plan.norm_weight.dtype),
            (plan.eps.shape, plan.eps.dtype),
        )
        template = [("T", dtype), ("K", width), ("H", hidden), ("HC", hc)]
        out_dtypes = [dtype, dtype]
        stand_in = kernels.StandIn(
            input_shapes=tuple(s for s, _ in specs),
            output_shapes=((2, width), (2, width)),
            grid=(256, hc, 2),
            threadgroup=(256, 1, 1),
        )
        what = "the prefill write-norm kernel"
    else:
        kernel = _kernel(
            "omlx_qwen4_hc_prefill_norm",
            ["x", "w", "eps"],
            ["xn"],
            _NP_SOURCE,
        )
        specs = (
            ((1, width), dtype),
            (plan.norm_weight.shape, plan.norm_weight.dtype),
            (plan.eps.shape, plan.eps.dtype),
        )
        template = [("T", dtype), ("K", width), ("H", hidden)]
        out_dtypes = [dtype]
        stand_in = kernels.StandIn(
            input_shapes=tuple(s for s, _ in specs),
            output_shapes=((1, width),),
            grid=(256, hc, 1),
            threadgroup=(256, 1, 1),
        )
        what = "the prefill stream-norm kernel"
    inst = kernels.planned(kernel, specs, out_dtypes, template=template)
    return inst, (lambda s=stand_in: s), what


def warm_model(model) -> tuple:
    """Pre-compile every reachable fused HC Metal kernel for ``model`` on
    ``kernels._WARM_POOL`` and return one future per distinct unbuilt Metal.

    Marks each warmed decode/tail launch recorded and validated only after
    every Metal it depends on succeeds; fails closed with ``FaceKernelError``
    on any build or launch error. O(1) no-op returning ``()`` once warmed.
    """
    if (
        _DISABLED
        or model is None
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return ()

    write_modes = (False, True) if write_enabled() else (False,)
    tasks: dict[object, tuple] = {}
    barriers: list[tuple[tuple, object]] = []
    seen_launches: set[object] = set()
    seen_prefill: set[object] = set()
    tail_launches_by_inst: dict[object, list] = {}

    def register(inst, recipe, what) -> None:
        if inst not in tasks:
            tasks[inst] = (recipe, what)

    for plan in _iter_plans(model):
        for row_launches in plan.decode_launches:
            for write in write_modes:
                launch = row_launches[write]
                if launch in seen_launches:
                    continue
                seen_launches.add(launch)
                req = []
                for inst, recipe, what in launch._warm_entries(plan):
                    register(inst, recipe, what)
                    req.append(inst)
                if launch not in _VALIDATED:

                    def _mark_decode(fn=launch) -> None:
                        fn._mark_recorded()
                        with _VALIDATED_LOCK:
                            _VALIDATED.add(fn)

                    barriers.append((tuple(dict.fromkeys(req)), _mark_decode))
                else:
                    launch._mark_recorded()

        if plan.dtype not in _DTYPES:
            continue
        tail_inst = None
        if plan.tail_launch is not None:
            tail_inst, tail_recipe, tail_what = plan.tail_launch._warm_entry(plan)
            register(tail_inst, tail_recipe, tail_what)
            tail_launches_by_inst.setdefault(tail_inst, []).append(plan.tail_launch)

        for write in write_modes:
            sig = plan.prefill_signatures[write]
            if sig in seen_prefill:
                continue
            seen_prefill.add(sig)
            norm_inst, norm_recipe, norm_what = _prefill_norm_warm_entry(plan, write)
            register(norm_inst, norm_recipe, norm_what)
            req = [norm_inst]
            if tail_inst is not None:
                req.append(tail_inst)
            if sig not in _VALIDATED:

                def _mark_prefill(signature=sig) -> None:
                    with _VALIDATED_LOCK:
                        _VALIDATED.add(signature)

                barriers.append((tuple(dict.fromkeys(req)), _mark_prefill))

    for tail_inst, launches in tail_launches_by_inst.items():

        def _mark_tails(fns=tuple(launches)) -> None:
            for fn in fns:
                fn._mark_recorded()

        barriers.append(((tail_inst,), _mark_tails))

    if not tasks and not barriers:
        return ()

    # Snapshot unbuilt instances once so concurrent warm_model calls never
    # miss a barrier wakeup between checking _built and submitting futures.
    to_build = [
        (inst, recipe, what)
        for inst, (recipe, what) in tasks.items()
        if inst not in inst.kernel._built
    ]
    to_build_set = {inst for inst, _, _ in to_build}

    pending_by_inst: dict[object, list] = {}
    for req_insts, on_ready in barriers:
        unbuilt = [inst for inst in req_insts if inst in to_build_set]
        if not unbuilt:
            on_ready()
            continue
        state = [len(unbuilt), threading.Lock()]

        def _step(st=state, cb=on_ready) -> None:
            with st[1]:
                st[0] -= 1
                done = st[0] == 0
            if done:
                cb()

        for inst in unbuilt:
            pending_by_inst.setdefault(inst, []).append(_step)

    if not to_build:
        return ()

    def _build_one(inst, recipe, what, steps) -> None:
        _run_named(what, lambda: inst.build(recipe))
        for step in steps:
            step()

    return tuple(
        kernels._WARM_POOL.submit(
            _build_one,
            inst,
            recipe,
            what,
            tuple(pending_by_inst.get(inst, ())),
        )
        for inst, recipe, what in to_build
    )


def warm_model_dispatch(model, *, prefill_lengths: tuple[int, ...] = ()) -> None:
    """Pre-compile HC prefill ``mx.compile`` graphs (``_act`` and, when a layer
    has no tail-inject kernel, ``_tail``) on the engine's serving thread."""
    if (
        _DISABLED
        or model is None
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return
    lengths = tuple(
        dict.fromkeys(r for r in prefill_lengths if type(r) is int and r > MAX_ROWS)
    )
    seen_act: set[tuple] = set()
    seen_tail: set[tuple] = set()
    for plan in _iter_plans(model):
        if plan.dtype not in _DTYPES:
            continue
        act_key = (plan.hc, plan.dtype)
        if act_key not in seen_act:
            seen_act.add(act_key)
            mx.eval(_act(plan.hc)(mx.zeros((1, 1, plan.lowrank), dtype=plan.dtype)))
        if plan.tail_launch is None and lengths:
            for length in lengths:
                tail_key = (plan.hc, plan.hidden, length, plan.dtype)
                if tail_key not in seen_tail:
                    seen_tail.add(tail_key)
                    z = mx.zeros((1, length, plan.width), dtype=plan.dtype)
                    mx.eval(_tail(plan.hc, plan.hidden)(z, z))


__all__ = [
    "FaceKernelError",
    "MAX_ROWS",
    "compatible",
    "enabled",
    "fused_forward",
    "prefill_compatible",
    "prefill_forward",
    "rebuild_plan",
    "rebuild_plans",
    "warm_model",
    "warm_model_dispatch",
    "write_enabled",
]
