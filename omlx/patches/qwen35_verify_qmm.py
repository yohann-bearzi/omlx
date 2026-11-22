# SPDX-License-Identifier: Apache-2.0
#
# Adapted from MTPLX (mtplx/verify_kernels.py)
#   Copyright 2026 Youssof Altoukhi
#   Licensed under the Apache License, Version 2.0
#   https://github.com/youssofal/mtplx
#   "Powered by MTPLX by Youssof Altoukhi"
#
# The split-K and multi-simdgroup (msg) kernel morphologies and their Metal
# source generation below are ported from MTPLX. The simdgroup-matrix tile
# kernel (``mma``) for 8..32 rows is adapted from dflash-mlx
# (dflash_mlx/verify_qmm.py, Copyright 2026 bstnxbt, Apache-2.0). The oMLX
# integration — thread-local armed routing through ``nn.QuantizedLinear``,
# the hybrid dispatch, and the row/N-floor gating — is original to oMLX.
#
# The ``sg8`` kernels' ``128 + q`` weight operand bits and per-group input
# sum correction are adapted from Splash
# (runtime/metal/kernels/decode/linear_q4_sgmatrix.metal, Apache-2.0,
# https://github.com/incoai/splash).
"""Verify-shape quantized-matmul kernels for native MTP.

Speculative verify multiplies a skinny row batch (M = 1 + draft depth)
against the model's quantized weights. Stock MLX qmm is tuned for M=1
(decode) and large-M (prefill); at M=3..6 it pays a steep per-row penalty —
measured on Qwen3.6-27B/M3 Ultra the lm_head scales linearly (1.2ms at M=1 →
4.3ms at M=4) and the summed MLP/GDN projections add ~15ms per verify call.

Two kernel morphologies:

- split-K: grid y = N/4 column tiles, K reduction split across 2..4
  simdgroups with a threadgroup reduction. Wins in-context (mixed scheduling
  with attention/GDN kernels) — deep occupancy queues + latency hiding.
- msg (multi-simdgroup): one threadgroup carries NSG barrier-free
  simdgroups, each owning a BN=4 column tile with the full-K reduction
  lane-strided over pack-interleaved weight words. Wins on huge-N (lm_head)
  where the split-K tiny-tile grid thrashes the scheduler.

Dispatch: split-K everywhere, msg for N >= 100k, and for 7..24 rows (batched
verify: rows = requests x block) the ``mma`` tile kernel, which dequantizes
16-column weight tiles into threadgroup memory once and multiplies them
against every row with simdgroup matrices, so a batch of rows costs one pass
over the weights. Routing is armed only around the MTP verify forward via a
thread-local flag (set by
``omlx.patches.mlx_lm_mtp.batch_generator._call_backbone``) so nothing else
in the process sees the patched ``nn.QuantizedLinear``.

Numerics: fp32 accumulation in lane-strided K order differs from stock qmm
at bf16 tail-ULP level. Greedy outputs can therefore occasionally diverge
from the unrouted path (the token is still trunk-verified — the divergence
class is the same as any kernel change).

Supported: 4-bit and 8-bit affine, group_size in {32, 64, 128}, bf16/fp16
activations, M in 3..6, K % 64 == 0, N % 4 == 0.
The mma path takes 2-, 3-, 4-, 5-, 6- and 8-bit affine, M in 7..24.
It needs K % 256 == 0 and N % 16 == 0. Everything else falls back to stock.

On M5 (NAX) GPUs, 4-, 5-, 6- and 8-bit group-64 weights with N >= 1024 take the
skinny tensor-unit kernel (``omlx.custom_kernels.nax_skinny_qmm``) at ``mma``'s
9..16-row shapes and at 17..48 rows. It reads each weight once for all rows.

Blocks of four to eight rows (DFlash and Lightning MTP verifies) take
``sg8`` instead: bf16 or fp16 rows, 2- to 6-bit weights, N >= 1024. Each lane builds its simdgroup-matrix
operands in registers from one packed word per 32 inputs (bf16 ``128 + q``,
exact), so no threadgroup dequant tile is needed; the group fold removes
``128 * sum(x)`` and applies scale and bias. The verifier's projection tuples
share one launch (``vk_group_sg8``, mixed 4/5-bit widths allowed) and the dense
MLP's gate/up/SiLU is one launch (``vk_swiglu_sg8``). On tensor-unit GPUs a single
projection below six rows stays on stock qmm.

The armed verifier also fuses each residual add with the next RMSNorm
(bit-exact to the separate ops) and submits the partial graph every few
layers, so the GPU starts while the host still builds the verify.
"""

from __future__ import annotations

import logging
import re
import threading
from functools import cache, partial

from omlx.custom_kernels import nax_skinny_qmm
from omlx.custom_kernels.nax import is_nax_available

from . import m1_verify_mma, m1_verify_qmv

logger = logging.getLogger(__name__)

_KERNEL_CACHE: dict = {}


class _RouteState(threading.local):
    value: bool = False
    row_exact: bool = False
    layers: int = 0


_ROUTE_ARMED = _RouteState()
_ROW_EXACT_LOCK = threading.Lock()
_ROW_EXACT_ACTIVE = 0

_MSG_NSG = 8  # simdgroups per threadgroup for the msg (lm_head) kernel

# Only route projections with N >= this. The vk kernels beat stock qmm on GPU
# time at every eligible shape, but each ``mx.fast.metal_kernel`` invocation
# pays more Python/dispatch overhead than the built-in fast path. With ~330
# projections per verify forward, routing the small GDN/attention shapes
# (~1.0x GPU win) costs more on the CPU than it saves; the large-N shapes are
# few calls with real wins (lm_head 2.6-3.3x at M=3-4).
_MIN_ROUTE_N = 16384
# The mma path wins from much smaller N because stock qmm reads the weights
# more than once (qmv_wide tiles) or wastes half a 32-row tile at these M.
_MIN_MMA_ROUTE_N = 4096
# On NAX GPUs stock MLX's tensor-unit qmm wins at 24 rows; mma still wins at 16.
_NAX_MMA_MAX_ROWS = 16
# Draft k/v and conv projections (N 1024..1280) also gain from sg8.
_MIN_SG8_ROUTE_N = 1024
_SG8_MIN_ROWS = 4
# On tensor-unit GPUs stock qmm beats sg8 for a single projection below 6 rows.
_SG8_SINGLE_MIN_ROWS_NAX = 6


def set_verify_qmm_armed(flag: bool, *, row_exact: bool = False) -> None:
    """Arm/disarm verify-qmm routing (MTP verify forwards only).

    ``row_exact`` arms the row-exact mode instead: every multi-row
    ``nn.QuantizedLinear`` call runs ``row_exact_qmv`` (one-row decode
    arithmetic per row) and the fast verify kernels below stay disarmed.
    """
    global _ROW_EXACT_ACTIVE
    armed = bool(flag) and bool(row_exact)
    prev = _ROUTE_ARMED.row_exact
    _ROUTE_ARMED.value = bool(flag) and not row_exact
    _ROUTE_ARMED.row_exact = armed
    _ROUTE_ARMED.layers = 0
    if armed != prev:
        with _ROW_EXACT_LOCK:
            _ROW_EXACT_ACTIVE += 1 if armed else -1
    from omlx.custom_kernels.quant_matmul import fast as _native

    _native.set_row_exact_armed(armed)


def _is_armed() -> bool:
    return _ROUTE_ARMED.value


def is_row_exact_armed() -> bool:
    """True inside a verify forward whose rows must equal serial decode rows."""
    return _ROW_EXACT_ACTIVE > 0 and _ROUTE_ARMED.row_exact


# ---------------------------------------------------------------------------
# msg kernel — lm_head geometry.
# ---------------------------------------------------------------------------


# The multi-simdgroup (msg) kernel of huge N (a vocabulary head) over the
# tile face ``F`` of its matrix ``w`` (docs/quant-formats.md, The tile face):
# NSG simdgroups a threadgroup, each four rows n0..n0 + 3 of the matrix with
# the full K walked lane-strided, a run of 8 inputs a lane and step, every
# input row against each row's weights in the run's order; then simd_sum and
# one lane per result stores. Each run reads one set of group parameters
# and packed codes per output column.
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


@cache
def _affine_tile_faces(bits: int) -> frozenset:
    """The builtin affine tile faces of ``bits`` for every group and scale
    type. The msg kernel specializes those only; a tabled or plugin face
    keeps the generic loop."""
    import mlx.core as mx

    from omlx.formats import affine

    return frozenset(
        affine.tile_face(bits, group, dtype)
        for group in (32, 64, 128)
        for dtype in (mx.bfloat16, mx.float16)
    )


@cache
def _msg_kernel(m: int, face, nsg: int):
    """The msg kernel of ``m`` input rows over the tile face ``face``."""
    from omlx.formats import kernels

    xloads = "".join(
        f"        Vec8 v{r} = xv[{r} * (K / 8) + run];\n" for r in range(m)
    )
    if m == 6 and face in _affine_tile_faces(8):
        # Six activation rows need 24 accumulators. Consume both packed
        # words of one output column before fetching the next column.
        cursors = ""
        decode = _MSG_ONE_ROW
        fmas = "\n".join(
            f"            acc[j * {m} + {r}] += float(v{r}[ki]) * wj;" for r in range(m)
        )
    else:
        cursors = _MSG_CURSORS
        if face in _affine_tile_faces(4):
            # Keep full-width row arithmetic, rebasing at n0 only once.
            cursors = """\
      const R r0(w, size_t(n0));
      const F four{(const device uint8_t*)r0.c, r0.s, r0.b};
      const R r1(four, size_t(1));
      const R r2(four, size_t(2));
      const R r3(four, size_t(3));"""
        decode = _MSG_FOUR_ROWS
        fmas = "\n".join(
            f"          acc[{j} * {m} + {r}] += float(v{r}[ki]) * w{j};"
            for j in range(4)
            for r in range(m)
        )
    source = (
        _MSG_SOURCE.replace("__NSG__", str(nsg))
        .replace("__M__", str(m))
        .replace("__FACE__", face.type)
        .replace("__CURSORS__", cursors)
        .replace("__XLOADS__\n", xloads)
        .replace("__DECODE__", decode)
        .replace("__FMAS__", fmas)
    )
    return kernels.face_kernel(
        f"omlx_vk_m{m}_nsg{nsg}", ("x", "w", "N_size"), {"w": face}, ("y",), "", source
    )


# ---------------------------------------------------------------------------
# split-K kernel — default.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# mma kernel — 7..32 rows, simdgroup matrix tiles (adapted from dflash-mlx's
# combo_ktmpl morphology).
# ---------------------------------------------------------------------------

_MMA_BN = 16
_MMA_BK = 32
_MMA_NSG = 8


# The simdgroup-matrix tile kernel (mma) of 7 to 24 rows over the tile face
# ``F`` of its matrix ``w``: a threadgroup takes BN = 16 columns, its eight
# simdgroups a K chunk each. Per 32-deep step a lane decodes two runs of 8
# weights of its column into its simdgroup's tile, rounded to T; the
# simdgroup multiplies the tile against every row tile; the parts are summed
# once at the end, in simdgroup order.
@cache
def _mma_face(bits: int, group_size: int, dtype):
    """The affine tile face of one signature, built once per signature."""
    from omlx.formats import affine

    return affine.tile_face(bits, group_size, dtype)


# ---------------------------------------------------------------------------
# sg8 kernel - up to eight rows on exact (128 + q) simdgroup operands.
# ---------------------------------------------------------------------------


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
    pair = _operand_bits("R::pair(pack, j)", half)
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
                    using R = typename omlx_code_runs<decltype(m{m}), K>::type;
                    static_assert(R::GROUP == GS, "codes face: group size differs");
                    R cursor(m{m}, size_t(n));
                    const auto pack = cursor.fetch(kq / 8 + int(fn >> 1));
                    for (int j = 0; j < 4; ++j) {{
                        simdgroup_matrix<T, 8, 8> a, b;
                        uint pair = {pair};
                        a.thread_elements()[0] = as_type<T>(ushort(pair));
                        a.thread_elements()[1] = as_type<T>(ushort(pair >> 16));
                        b.thread_elements()[0] = xa[j];
                        b.thread_elements()[1] = xb[j];
                        simdgroup_multiply_accumulate(acc{m}[c], a, b, acc{m}[c]);
                    }}
                }}""" for m in range(mats))
    fold = "\n".join(f"""            {{
                using R = typename omlx_code_runs<decltype(m{m}), K>::type;
                const float2 map = R(m{m}, size_t(n)).params(g);
                float s = map.x, bias = map.y;
                float2 a2 = float2(acc{m}[c].thread_elements()[0], acc{m}[c].thread_elements()[1]);
                out{m}[c] += s * a2 + (bias - 128.0f * s) * xsum;
            }}""" for m in range(mats))
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
    int g_begin = int(sg) * PER;
    int nbase = int(threadgroup_position_in_grid.y) * 8 * CT;
    bool r0 = int(fn) < M;
    bool r1 = int(fn) + 1 < M;
    static_assert(M >= 1 && M <= 8 && GS >= 32 && GS % 32 == 0 &&
                  K > 0 && K % (GS * NSG) == 0 && N > 0 && N % (8 * CT) == 0,
                  "sg8 launch geometry: whole row, column and K tiles required");
    // Step j of a 32-wide k block: A columns 2i and 2i + 1 are values j and
    // j + 4 of the lane's eight packed values i * 8 .. i * 8 + 7, so both
    // operands come from one shift and mask. Its x values over j are adjacent.
    uint kx = (fm >> 1) * 8 + (fm & 1u) * 4;
    // Lanes past the last row read row M - 1 and never store, so the hot
    // loop has no row branches.
    const device T* x0 = x + min(int(fn), M - 1) * K + kx;
    const device T* x1 = x + min(int(fn) + 1, M - 1) * K + kx;
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


def _sg8_geometry(K: int, N: int) -> tuple[int, int]:
    """Simdgroups per threadgroup (K split) and 8-column tiles per simdgroup.

    Measured with serialized dependent projections on M3 Ultra: vocab-size N
    fills the GPU with four simdgroups and four tiles; everything else wants
    eight K partitions, and fat-K down projections two tiles to reuse x.
    """
    if N >= 65536:
        return 4, 4
    return 8, 2 if K >= 16384 else 1


def sg8_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    import mlx.core as mx

    # On M1-class GPUs the M1 kernels beat sg8 inside their row windows; the
    # fused swiglu path keys off this check, so it falls back per-linear.
    if _m1_route(M, K, N, bits, group_size, dtype):
        return False
    nsg, ct = _sg8_geometry(int(K), int(N))
    return (
        # The operand holds 128 + q exactly only up to 7-bit codes.
        int(bits) in (2, 3, 4, 5, 6)
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        # Two or three rows stay on split-K, which is faster there in context.
        and _SG8_MIN_ROWS <= int(M) <= 8
        and int(K) % (int(group_size) * nsg) == 0
        and int(N) % (8 * ct) == 0
        and int(N) >= _MIN_SG8_ROUTE_N
    )


@cache
def _sg8_launch(faces, rows, k, n, group_size, dtype, summed, operand_dtypes=None):
    """Bind the Metal constants and launch dimensions for one input signature."""
    import mlx.core as mx

    from omlx.formats import kernels

    nsg, ct = _sg8_geometry(k, n)
    half = dtype == mx.float16
    matrices = {f"m{i}": face for i, face in enumerate(faces)}
    inputs = ("x", *matrices)
    source = _sg8_source(len(faces), half)
    if summed:
        inputs += ("xs",)
        source = _with_group_sums(source)
    constants = (
        f"    constexpr int K = {k:d}, N = {n:d}, M = {rows:d};\n"
        f"    constexpr int GS = {group_size:d}, NSG = {nsg:d}, CT = {ct:d};\n"
    )
    name = f"omlx_codes_sg8_k{k:d}_n{n:d}_m{rows:d}_g{group_size:d}"
    if summed:
        name += "_xs"
    if half:
        name += "_fp16"
    kernel = kernels.face_kernel(name, inputs, matrices, ("y",), "", constants + source)
    recorded = [False]

    def launch(**call):
        # The first call records the Metal; the wrapper's per-launch record
        # is not paid on stock affine checkpoints' verify windows. Latch only
        # after kernel(**call) succeeds so a failed first call still records.
        if recorded[0]:
            return kernel.native(**call)
        result = kernel(**call)
        recorded[0] = True
        return result

    return partial(
        launch,
        template=[("T", dtype)],
        grid=(32 * nsg, n // (8 * ct), 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[dtype],
    )


def _sg8_call(x2, faces, weights, n, group_size, sums):
    rows, k = x2.shape
    summed = group_size == _SUM_GROUP and sums is not None
    inputs = [x2, *weights]
    if summed:
        inputs.append(sums)
    launch = _sg8_launch(
        faces,
        rows,
        k,
        n,
        group_size,
        x2.dtype,
        summed,
        operand_dtypes=tuple(a.dtype for a in inputs[1:]),
    )
    return launch(inputs=inputs)[0]


def sg8_codes(x2, face, operands, *, n, group_size, sums=None):
    """Launch one eligible (rows, K) projection over its matrix's codes face.
    The caller binds the weight and face tensors in order for (n, K)."""
    import mlx.core as mx

    from omlx.formats import FLOATS, Codes

    if not isinstance(face, Codes):
        raise ValueError(f"{face!r} is not a Codes face")
    if x2.dtype not in (mx.bfloat16, mx.float16):
        raise ValueError(f"sg8 reads bf16 or fp16 activations, not {x2.dtype}")
    if not face.exact_for(x2.dtype):
        exact = ", ".join(str(d) for d in face.dtypes or FLOATS)
        raise ValueError(f"face {face.key} is exact for {exact}, not {x2.dtype}")
    return _sg8_call(x2, (face,), operands, n, group_size, sums)


@cache
def _affine_sg8_launch(
    bits, group_size, scale_dtype, bias_dtype, rows, k, n, dtype, summed
):
    from omlx.formats.affine import codes_face

    face = codes_face(bits, group_size, scale_dtype, bias_dtype)
    return _sg8_launch((face,), rows, k, n, group_size, dtype, summed)


def vk_qmm_sg8(x2, w_q, scales, biases, *, group_size: int, bits: int = 4, sums=None):
    """An affine projection through the same integer-level kernel as a plugin."""
    rows, k = x2.shape
    summed = group_size == _SUM_GROUP and sums is not None
    launch = _affine_sg8_launch(
        bits,
        group_size,
        scales.dtype,
        biases.dtype,
        rows,
        k,
        w_q.shape[0],
        x2.dtype,
        summed,
    )
    inputs = [x2, w_q, scales, biases]
    if summed:
        inputs.append(sums)
    return launch(inputs=inputs)[0]


@cache
def _affine_swiglu_sg8_launch(
    bits, group_size, g_sd, g_bd, u_sd, u_bd, rows, k, n, dtype, summed
):
    from omlx.formats.affine import codes_face

    faces = (
        codes_face(bits, group_size, g_sd, g_bd),
        codes_face(bits, group_size, u_sd, u_bd),
    )
    return _sg8_launch(faces, rows, k, n, group_size, dtype, summed)


def vk_swiglu_sg8(x2, gate, up, sums=None):
    """silu(x gate^T) * (x up^T) over the same codes traversal in one launch."""
    rows, k = x2.shape
    if (gate.bits, gate.group_size) != (up.bits, up.group_size):
        raise ValueError("swiglu needs one bits and group size for its pair")
    summed = gate.group_size == _SUM_GROUP and sums is not None
    launch = _affine_swiglu_sg8_launch(
        gate.bits,
        gate.group_size,
        gate.scales.dtype,
        gate.biases.dtype,
        up.scales.dtype,
        up.biases.dtype,
        rows,
        k,
        gate.weight.shape[0],
        x2.dtype,
        summed,
    )
    inputs = [
        x2,
        gate.weight,
        gate.scales,
        gate.biases,
        up.weight,
        up.scales,
        up.biases,
    ]
    if summed:
        inputs.append(sums)
    return launch(inputs=inputs)[0]


# Inputs whose producer already wrote their per-64 group sums, by identity.
_SUM_GROUP = 64


def register_group_sums(x, sums) -> None:
    table = getattr(_ROUTE_ARMED, "sums", None)
    if table is None:
        table = _ROUTE_ARMED.sums = {}
    table[id(x)] = (x, sums)


def group_sums_for(x):
    entry = (getattr(_ROUTE_ARMED, "sums", None) or {}).get(id(x))
    return entry[1] if entry is not None and entry[0] is x else None


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


def clear_group_sums() -> None:
    _ROUTE_ARMED.sums = None


def sg8_swiglu_eligible(gate, up, x) -> bool:
    """Gate and up share geometry and quantization, and rows fit sg8."""
    import mlx.nn as nn

    if not (
        _is_armed()
        and isinstance(gate, nn.QuantizedLinear)
        and isinstance(up, nn.QuantizedLinear)
        and x.ndim == 3
        and getattr(gate, "mode", "affine") == "affine"
        and getattr(up, "mode", "affine") == "affine"
        and "bias" not in gate
        and "bias" not in up
        and gate.weight.shape == up.weight.shape
        and gate.bits == up.bits
        and gate.group_size == up.group_size
    ):
        return False
    rows = x.shape[0] * x.shape[1]
    K = x.shape[2]
    N = gate.scales.shape[0]
    return sg8_eligible(rows, K, N, gate.bits, gate.group_size, x.dtype)


def mma_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    import mlx.core as mx

    return (
        int(bits) in (2, 3, 4, 5, 6, 8)
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        # Above 24 rows stock qmm fills its 32-row tile and wins.
        and 7 <= int(M) <= (_NAX_MMA_MAX_ROWS if is_nax_available() else 24)
        and int(K) % (_MMA_BK * _MMA_NSG) == 0
        and int(N) % _MMA_BN == 0
        and int(N) >= _MIN_MMA_ROUTE_N
    )


_AFFINE_MMA = re.compile(r"affine_tile_b([234568])g(32|64|128)_(bf16|fp16)")


@cache
def _mma_plan(row_tiles: int, face, k: int, dtype):
    """``(kernel, template)`` for one mma signature, built at construction.
    The builtin affine faces take the byte-identical skeleton with K baked;
    every other face binds tile_mma_ops into the same skeleton with K baked."""
    import mlx.core as mx

    from omlx.formats import verify_skeleton

    match = _AFFINE_MMA.fullmatch(face.key)
    if match and face in _affine_tile_faces(int(match[1])):
        bits, group, tag = int(match[1]), int(match[2]), match[3]
        ops = verify_skeleton.affine_mma_ops(bits, group)
        name = f"omlx_vk_mma_rt{row_tiles}_k{k}_q{bits}_gs{group}_{tag}"
        spec = verify_skeleton.emit_mma_kernel(ops, row_tiles, k, name)
        return mx.fast.metal_kernel(**spec), [("T", dtype)]
    from omlx.formats import kernels

    ops = verify_skeleton.tile_mma_ops(face)
    source = verify_skeleton.emit_mma_source(ops, row_tiles, k)
    kernel = kernels.face_kernel(
        f"omlx_vk_mma_rt{row_tiles}_k{k}",
        ("x", "w", "N_size"),
        {"w": face},
        ("y",),
        ops.header,
        source,
    )
    return kernel, [("T", dtype)]


@cache
def _mma_launch(
    row_tiles: int, bits: int, group_size: int, scale_dtype, dtype, k: int, n: int
):
    """The whole launch descriptor for one signature: one cache lookup on
    the hot path, the face, kernel, partial and N scalar built once."""
    import mlx.core as mx

    if k <= 0 or n <= 0:
        raise ValueError(f"vk_qmm_mma takes K > 0 and N > 0, not {k} x {n}")
    face = _mma_face(bits, group_size, scale_dtype)
    kernel, template = _mma_plan(row_tiles, face, k, dtype)
    launch = partial(
        kernel,
        template=template,
        grid=(32 * _MMA_NSG, n // _MMA_BN, 1),
        threadgroup=(32 * _MMA_NSG, 1, 1),
        output_shapes=[(8 * row_tiles, n)],
        output_dtypes=[dtype],
    )
    return launch, mx.array(n, dtype=mx.int32)


def vk_qmm_mma(x2, w_q, scales, biases, *, group_size: int, bits: int = 4):
    """(M, K) x (N, K)^T -> (M, N) for M in 7..24 through the mma tile kernel
    over the affine tile face; None for scales no tile face reads."""
    import mlx.core as mx

    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    if not 1 <= M <= 24 or K % (_MMA_BK * _MMA_NSG) or N % _MMA_BN:
        raise ValueError(
            f"vk_qmm_mma takes 1 to 24 rows, K in {_MMA_BK * _MMA_NSG}s and N in "
            f"{_MMA_BN}s, not {M} x {K} x {N}"
        )
    if scales.dtype not in (mx.bfloat16, mx.float16) or biases.dtype != scales.dtype:
        return None  # no tile face reads such scales
    row_tiles = (M + 7) // 8
    bm = 8 * row_tiles
    xm, M0 = _pad_rows(mx, x2, bm)
    launch, n_scalar = _mma_launch(
        row_tiles, bits, group_size, scales.dtype, x2.dtype, K, N
    )
    (y,) = launch(inputs=[xm, w_q, scales, biases, n_scalar])
    return y[:M0, :] if M0 < bm else y


# ---------------------------------------------------------------------------
# Dispatch.
# ---------------------------------------------------------------------------


def _pad_rows(mx, x2, m: int):
    # metal_kernel copies non-contiguous inputs itself, so aligned rows pass
    # through without an extra dispatch.
    M = int(x2.shape[0])
    if M < m:
        pad = mx.zeros((m - M, x2.shape[1]), dtype=x2.dtype)
        return mx.concatenate([x2, pad], axis=0), M
    return x2, M


@cache
def _msg_launch(m: int, face, nsg: int, dtype, k: int, n: int):
    """The msg launch of ``(m, face, nsg, dtype, k, n)``, its Metal recorded
    on the first call: a stock affine head's verify steps then call the
    kernel directly."""
    kernel = _msg_kernel(m, face, nsg)
    cols = 4 * nsg
    recorded = [False]

    def launch(**call):
        if recorded[0]:
            return kernel.native(**call)
        result = kernel(**call)
        recorded[0] = True
        return result

    return partial(
        launch,
        template=[("T", dtype), ("K", k)],
        grid=(32 * nsg, (n + cols - 1) // cols, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[dtype],
    )


def vk_qmm(x2, w_q, scales, biases, *, bits: int, group_size: int):
    """Verify-shape qmm: (M, K) x (N, K)^T -> (M, N), M in 2..6.

    Rows are padded up to the kernel's M template. split-K for regular
    shapes, msg tile for huge N (lm_head). Its kernels take 6 rows at most
    and columns in fours: other shapes are refused. None for a head whose
    scales no tile face reads.
    """
    import mlx.core as mx

    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    if not 1 <= M <= 6 or N % 4:
        raise ValueError(f"vk_qmm takes 1 to 6 rows and N in fours, not {M} x {N}")
    m = 4 if M <= 4 else 6

    if N >= 100000 and N % (4 * _MSG_NSG) == 0:
        from omlx.formats import affine

        if (
            scales.dtype not in (mx.bfloat16, mx.float16)
            or biases.dtype != scales.dtype
        ):
            return None  # no tile face reads such scales
        xm, M0 = _pad_rows(mx, x2, m)
        face = affine.tile_face(bits, group_size, scales.dtype)
        launch = _msg_launch(m, face, _MSG_NSG, x2.dtype, K, N)
        (y,) = launch(inputs=[xm, w_q, scales, biases, N])
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


def vk_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    import mlx.core as mx

    # M >= 3: at M=2 (depth-1 verify) the dispatch overhead eats the GPU win
    # AND skipping it keeps depth-1 greedy output bit-identical to the
    # unrouted path. Depth >= 2 verifies at M >= 3 where the kernels pay.
    return (
        int(bits) in (4, 8)
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        and 3 <= int(M) <= 6
        and int(K) % 64 == 0
        and int(N) % 4 == 0
        and int(N) >= _MIN_ROUTE_N
    )


# ---------------------------------------------------------------------------
# QuantizedLinear routing patch.
# ---------------------------------------------------------------------------

_QL_PATCHED = False


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


def sg8_group_eligible(linears, x) -> bool:
    """Projections of one input with a shared layout that fit one sg8 launch."""
    import mlx.core as mx
    import mlx.nn as nn

    if not (
        _is_armed()
        and len(linears) > 1
        and x.ndim == 3
        and x.dtype in (mx.bfloat16, mx.float16)
    ):
        return False
    rows = x.shape[0] * x.shape[1]
    K = x.shape[2]
    first = linears[0]
    if any(
        isinstance(linear, nn.QuantizedLinear)
        and _m1_route(
            rows, K, linear.scales.shape[0], linear.bits, linear.group_size, x.dtype
        )
        for linear in linears
    ):
        # Per-linear calls let the M1 route take the large projections.
        return False
    for linear in linears:
        if not (
            isinstance(linear, nn.QuantizedLinear)
            and getattr(linear, "mode", "affine") == "affine"
            and "bias" not in linear
            and int(linear.bits) in (4, 5)
            and linear.group_size == first.group_size
            and linear.weight.shape[1] * 32 // linear.bits == K
            and linear.scales.shape[0] % 8 == 0
            # The kernel binds every matrix's parameters to one pointer type.
            and linear.scales.dtype == x.dtype
            and linear.biases.dtype == x.dtype
        ):
            return False
    return (
        _SG8_MIN_ROWS <= rows <= 8
        and int(first.group_size) in (32, 64, 128)
        and K % (int(first.group_size) * _SG8_GROUP_NSG) == 0
        and max(linear.scales.shape[0] for linear in linears) >= _MIN_MMA_ROUTE_N
    )


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


def _patch_verify_grouped_linears() -> None:
    """Serve the verifier's same-input projection tuples in one sg8 launch."""
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )
    except ImportError:
        return
    original = Qwen3_5BatchInvariantForward._linears
    if getattr(original, "_omlx_sg8_group", False):
        return

    def _linears(self, linears, x):
        linears = tuple(linears)
        if sg8_group_eligible(linears, x):
            batch, length, K = x.shape
            outs = vk_group_sg8(
                x.reshape(batch * length, K), linears, group_sums_for(x)
            )
            return tuple(out.reshape(batch, length, -1) for out in outs)
        return original(self, linears, x)

    _linears._omlx_sg8_group = True
    Qwen3_5BatchInvariantForward._linears = _linears


def _patch_verify_swiglu() -> None:
    """Fuse the dense verify MLP's gate/up projections and swiglu into sg8."""
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )
    except ImportError:
        return
    original = Qwen3_5BatchInvariantForward._feed_forward
    if getattr(original, "_omlx_sg8_swiglu", False):
        return

    def _feed_forward(self, feed_forward, x):
        gate = getattr(feed_forward, "gate_proj", None)
        up = getattr(feed_forward, "up_proj", None)
        down = getattr(feed_forward, "down_proj", None)
        if (
            down is not None
            and not hasattr(feed_forward, "switch_mlp")
            and sg8_swiglu_eligible(gate, up, x)
        ):
            batch, length, K = x.shape
            hidden = vk_swiglu_sg8(
                x.reshape(batch * length, K), gate, up, group_sums_for(x)
            )
            hidden = hidden.reshape(batch, length, -1)
            if down.group_size == _SUM_GROUP and sg8_eligible(
                batch * length,
                hidden.shape[-1],
                down.scales.shape[0],
                down.bits,
                down.group_size,
                hidden.dtype,
            ):
                register_group_sums(hidden, group_sums(hidden))
            return self._linear(down, hidden)
        return original(self, feed_forward, x)

    _feed_forward._omlx_sg8_swiglu = True
    Qwen3_5BatchInvariantForward._feed_forward = _feed_forward


_FLUSH_LAYERS = 8

# Residual add then MLX rms_norm in one launch, one threadgroup per row.
# The sum is rounded like the separate add, and the square sums follow the
# per-thread order of MLX's rms kernels (four reads per step, 1024 lanes).
_ADD_RMS_SOURCE = """
    uint lid = thread_position_in_threadgroup.x;
    uint lsize = threads_per_threadgroup.x;
    uint sg = lid / 32;
    uint lane = lid % 32;
    uint row = threadgroup_position_in_grid.x;
    constexpr int STEPS = (D + 4095) / 4096;
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    const device T* ap = a + long(row) * D;
    const device T* bp = b + long(row) * D;
    T vals[STEPS * 4];
    float acc = 0.0f;
    for (int st = 0; st < STEPS; ++st) {
        for (int i = 0; i < 4; ++i) {
            int idx = st * int(lsize) * 4 + int(lid) * 4 + i;
            T v = T(0);
            if (idx < D) {
                v = ap[idx] + bp[idx];
                s_out[long(row) * D + idx] = v;
                float xi = v;
                acc += xi * xi;
            }
            vals[st * 4 + i] = v;
        }
    }
    acc = simd_sum(acc);
    if (sg == 0)
        local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
        local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0)
            local_inv[0] = metal::precise::rsqrt(acc / D + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = local_inv[0];
    for (int st = 0; st < STEPS; ++st) {
        float part = 0.0f;
        int base = st * int(lsize) * 4 + int(lid) * 4;
        for (int i = 0; i < 4; ++i) {
            int idx = base + i;
            if (idx < D) {
                T nv = w[idx] * static_cast<T>(vals[st * 4 + i] * inv);
                n_out[long(row) * D + idx] = nv;
                part += float(nv);
            }
        }
        // Sixteen lanes cover one 64-wide group of the normed row.
        for (int off = 1; off < 16; off <<= 1)
            part += simd_shuffle_xor(part, ushort(off));
        if ((lid & 15) == 0 && base < D)
            xs_out[long(row) * (D / 64) + base / 64] = part;
    }
"""


def add_rms_norm(a, b, norm):
    """``(a + b, rms_norm(a + b), group sums)``; the first two match the
    separate MLX ops bit for bit, the sums are per 64 normed values."""
    import mlx.core as mx

    if "add_rms" not in _KERNEL_CACHE:
        _KERNEL_CACHE["add_rms"] = mx.fast.metal_kernel(
            name="omlx_verify_add_rms",
            input_names=["a", "b", "w", "eps"],
            output_names=["s_out", "n_out", "xs_out"],
            source=_ADD_RMS_SOURCE,
        )
    D = int(a.shape[-1])
    rows = a.size // D
    return _KERNEL_CACHE["add_rms"](
        inputs=[a, b, norm.weight, mx.array([norm.eps], dtype=mx.float32)],
        template=[("T", a.dtype), ("D", D)],
        grid=(1024 * rows, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[a.shape, a.shape, (rows, D // 64)],
        output_dtypes=[a.dtype, a.dtype, mx.float32],
    )


def _add_rms_eligible(a, b, norm) -> bool:
    import mlx.core as mx
    import mlx.nn as nn

    return (
        type(norm) is nn.RMSNorm
        and a.dtype in (mx.bfloat16, mx.float16)
        and b.dtype == a.dtype
        and a.shape == b.shape
        and norm.weight.dtype == a.dtype
        and norm.weight.shape == (a.shape[-1],)
        and 4096 < a.shape[-1] <= 16384
        and a.shape[-1] % 64 == 0
    )


def _patch_verify_layer_flush() -> None:
    """Fuse residual adds with the following RMSNorm and flush while building.

    An armed verify layer adds its mixer output and applies the MLP norm in
    one launch, and adds its MLP output together with the next layer's input
    norm. Every few layers the partial graph is submitted so the GPU starts
    while the host is still building the 27B verify.
    """
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )
    except ImportError:
        return
    original = Qwen3_5BatchInvariantForward._layer
    original_model = Qwen3_5BatchInvariantForward._model
    if getattr(original, "_omlx_flush", False):
        return
    import mlx.core as mx

    def _model(self, model, *args, **kwargs):
        layers = model.layers
        _ROUTE_ARMED.next_norm = {
            id(layer): nxt.input_layernorm for layer, nxt in zip(layers, layers[1:])
        }
        _ROUTE_ARMED.normed = None
        clear_group_sums()
        try:
            return original_model(self, model, *args, **kwargs)
        finally:
            _ROUTE_ARMED.next_norm = None
            _ROUTE_ARMED.normed = None
            clear_group_sums()

    def _mixed(self, layer, hidden, mask, cache, position_ids, position_embeddings):
        pending = getattr(_ROUTE_ARMED, "normed", None)
        if pending is not None and pending[0] is hidden:
            normed = pending[1]
        else:
            normed = layer.input_layernorm(hidden)
        if layer.is_linear:
            residual = self._gated_delta(layer.linear_attn, normed, mask, cache)
        else:
            residual = self._attention(
                layer.self_attn,
                normed,
                mask,
                cache,
                position_ids,
                position_embeddings,
            )
        post = layer.post_attention_layernorm
        if not _add_rms_eligible(hidden, residual, post):
            hidden = hidden + residual
            return hidden + self._feed_forward(layer.mlp, post(hidden))
        hidden, normed, sums = add_rms_norm(hidden, residual, post)
        register_group_sums(normed, sums)
        out = self._feed_forward(layer.mlp, normed)
        nxt = (getattr(_ROUTE_ARMED, "next_norm", None) or {}).get(id(layer))
        if nxt is None or not _add_rms_eligible(hidden, out, nxt):
            return hidden + out
        hidden, normed, sums = add_rms_norm(hidden, out, nxt)
        register_group_sums(normed, sums)
        _ROUTE_ARMED.normed = (hidden, normed)
        return hidden

    def _layer(self, layer, hidden, *args, **kwargs):
        if not _is_armed():
            return original(self, layer, hidden, *args, **kwargs)
        hidden = _mixed(self, layer, hidden, *args, **kwargs)
        count = getattr(_ROUTE_ARMED, "layers", 0) + 1
        _ROUTE_ARMED.layers = count
        if count % _FLUSH_LAYERS == 0:
            mx.async_eval(hidden)
        return hidden

    _model._omlx_flush = True
    _layer._omlx_flush = True
    Qwen3_5BatchInvariantForward._layer = _layer
    Qwen3_5BatchInvariantForward._model = _model


def _m1_route(rows: int, K: int, N: int, bits: int, group_size: int, dtype):
    """M1-class kernel for a verify projection ("m1", "m1_mma") or None."""
    if m1_verify_qmv.eligible(rows, K, N, bits, group_size, dtype):
        return "m1"
    if m1_verify_mma.eligible(rows, K, N, bits, group_size, dtype):
        return "m1_mma"
    return None


def skinny_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    """NAX skinny kernel (``nax_skinny_qmm``) for a 9..48-row verify projection.

    It rounds each dequantized weight to the row dtype like ``mma`` and stock's
    multi-row qmm, so it replaces only those: stock from 17 rows, ``mma`` at 9..16.
    Other 9..16-row calls keep stock's exact-code path, which rounds less. Outputs
    under 1024 columns are not faster.
    """
    if not nax_skinny_qmm.MIN_ROWS <= int(M) <= nax_skinny_qmm.MAX_ROWS:
        return False
    import mlx.core as mx

    if not (
        int(bits) in nax_skinny_qmm.BITS
        and int(group_size) == 64
        and dtype in (mx.bfloat16, mx.float16)
        and int(K) % 64 == 0
        and int(N) % 32 == 0
        and int(N) >= 1024
        and is_nax_available()
    ):
        return False
    return int(M) > _NAX_MMA_MAX_ROWS or mma_eligible(M, K, N, bits, group_size, dtype)


def _verify_route(rows: int, K: int, N: int, bits: int, group_size: int, dtype):
    """Kernel an armed verify projection runs ("m1", "m1_mma", "sg8", "skinny",
    "mma" or "vk"), or None for the stock qmm."""
    m1 = _m1_route(rows, K, N, bits, group_size, dtype)
    if m1:
        return m1
    if sg8_eligible(rows, K, N, bits, group_size, dtype):
        if rows < _SG8_SINGLE_MIN_ROWS_NAX and is_nax_available():
            return None
        return "sg8"
    if skinny_eligible(rows, K, N, bits, group_size, dtype):
        return "skinny"
    if mma_eligible(rows, K, N, bits, group_size, dtype):
        return "mma"
    if rows <= 6 and vk_eligible(rows, K, N, bits, group_size, dtype):
        return "vk"
    return None


def run_route(layer, x2, route: str, *, sums=None):
    """``layer`` applied to the (rows, K) input ``x2`` through the ``route`` kernel.

    Returns None when the route cannot read the layer's scales dtype.
    """
    if route in ("m1", "m1_mma", "skinny") and layer.scales.dtype != x2.dtype:
        return None
    args = (x2, layer.weight, layer.scales, layer.biases)
    if route == "m1":
        return m1_verify_qmv.m1_qmm(*args, bits=layer.bits, group_size=layer.group_size)
    if route == "m1_mma":
        return m1_verify_mma.m1_mma_qmm(
            *args, bits=layer.bits, group_size=layer.group_size
        )
    if route == "sg8":
        return vk_qmm_sg8(
            *args, group_size=layer.group_size, bits=layer.bits, sums=sums
        )
    if route == "skinny":
        K, N = int(x2.shape[1]), int(layer.scales.shape[0])
        return nax_skinny_qmm.skinny_qmm(*args, K, N, bits=layer.bits)
    if route == "mma":
        return vk_qmm_mma(*args, group_size=layer.group_size, bits=layer.bits)
    return vk_qmm(*args, bits=layer.bits, group_size=layer.group_size)


def takes_verify_route(layer, rows: int, dtype) -> bool:
    """Whether a ``rows``-row call of ``layer`` (an ``nn.QuantizedLinear``)
    now leaves the stock qmm: row-exact mode, or an armed verify route."""
    if rows < 2 or not getattr(type(layer), "_omlx_verify_qmm_patched", False):
        return False
    if is_row_exact_armed():
        return True
    if not _is_armed() or getattr(layer, "mode", "affine") != "affine":
        return False
    K = layer.weight.shape[-1] * 32 // layer.bits
    N = layer.scales.shape[0]
    return _verify_route(rows, K, N, layer.bits, layer.group_size, dtype) is not None


def apply_verify_qmm_patch() -> bool:
    """Route verify-shaped ``nn.QuantizedLinear`` calls to the vk kernels.

    Strictly gated: only while the MTP verify flag is armed
    (``set_verify_qmm_armed``), only rank-3 inputs whose rows (batch x block)
    fall in 3..6 (split-K) or 7..24 (mma), only affine layouts the kernels
    support. Everything else takes the stock path.
    """
    global _QL_PATCHED
    if _QL_PATCHED:
        return True

    import mlx.nn as nn

    cls = nn.QuantizedLinear
    if getattr(cls, "_omlx_verify_qmm_patched", False):
        _QL_PATCHED = True
        return True

    orig_call = cls.__call__

    from .row_exact_qmv import quantized_linear as row_exact_linear

    def patched_call(self, x):
        if is_row_exact_armed() and x.ndim >= 2 and x.size // x.shape[-1] > 1:
            return row_exact_linear(self, x)
        if (
            not _is_armed()
            or x.ndim != 3
            or getattr(self, "mode", "affine") != "affine"
            or (x.shape[0] == 1 and x.shape[1] < 2)
        ):
            return orig_call(self, x)
        batch, length, K = x.shape
        rows = batch * length
        N = self.scales.shape[0]
        route = _verify_route(rows, K, N, self.bits, self.group_size, x.dtype)
        if route is None:
            return orig_call(self, x)
        try:
            # Batched verify rows (requests x block) share one weight pass.
            y = run_route(self, x.reshape(rows, K), route, sums=group_sums_for(x))
            if y is None:
                return orig_call(self, x)
            if hasattr(self, "bias"):
                y = y + self.bias
            return y.reshape(batch, length, N)
        except Exception:
            logger.debug("verify qmm route failed; stock fallback", exc_info=True)
            return orig_call(self, x)

    cls.__call__ = patched_call
    cls._omlx_verify_qmm_patched = True
    _patch_verify_swiglu()
    _patch_verify_grouped_linears()
    _patch_verify_layer_flush()
    _QL_PATCHED = True
    logger.info(
        "MTP verify qmm patch applied (rows 2..6 split-K, 7..8 sg8, 9..24 mma%s%s)",
        "; NAX: skinny 9..48" if is_nax_available() else "",
        "; M1: qmv 2..4, mma 5..16" if m1_verify_qmv.enabled() else "",
    )
    return True
