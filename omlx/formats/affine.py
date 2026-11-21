# SPDX-License-Identifier: Apache-2.0
"""MLX's affine quantization (``mode="affine"``, the oQ and oQe checkpoints)
as a weight format.

The dot face is over the MLX 0.32.4 ``quantized.h`` transcription below: a
kernel instantiated with it computes each row as MLX's ``qmv_fast`` or ``qmv``
does, bit for bit, for bf16 or fp16 activations with scales in the same
dtype. MLX runs ``qmv_fast`` when K is a multiple of the kernel block, and
for one row through ``gather_qmm`` (stacked weights) when N is a multiple of
8 as well. One row of a 2-D matrix of K 64 or 128 at 2, 4 or 8 bits is a
third MLX kernel, ``qmv_quad``, which sums in another order: such a matrix
has no face.

The tile face gives each weight as ``scale * q + bias`` in float32, which a
kernel rounds once to the activations' dtype: MLX's ``dequantize`` of the
same weight over float32 copies of the scales and the biases (``scale * q``
is exact in float32 where it is finite), and what MLX's sorted
``gather_qmm`` loads into its tiles. MLX's ``dequantize`` over scales of
bf16 or fp16 computes in that type, and can differ. It reads runs of 4 and
8 bits only.

The codes face gives the packed integer levels and the group's map to the
single and shared SwiGLU sg8 traversal; MLX has no such route.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from . import Codes, Dot, QuantFormat, Tile

_BITS = (2, 3, 4, 5, 6, 8)
_GROUP_SIZES = (32, 64, 128)

# MLX 0.32.4 quantized.h's constants for a bit width, group size and
# traversal (qmv_fast's, FAST, or qmv's).
_CONSTANTS = r"""
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int BITS = __BITS__;
constant constexpr int GS = __GS__;
constant constexpr int FAST = __FAST__;
constant constexpr int PACK_FACTOR =
    __PACK_FACTOR__;
constant constexpr int BYTES_PER_PACK =
    ((BITS & (BITS - 1)) == 0) ? 4 : (BITS == 5 ? 5 : 3);
constant constexpr int PACKS_PER_THREAD = __PACKS__;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
"""

# MLX's pack factor (get_pack_factor) and the packs a lane reads per block
# (qmv_fast_impl's packs_per_thread: one at 2 bits; qmv_impl: one), as each
# width's source states them. The 4- to 8-bit text is the one their kernels
# have always been compiled from: a kernel of an existing checkpoint keeps
# its source byte for byte.
_PACKING = {
    2: ("32 / BITS", "1"),
    3: ("8", "FAST ? 2 : 1"),
    **dict.fromkeys(
        (4, 5, 6, 8), ("(BITS == 5) ? 8 : (BITS == 6 ? 4 : 32 / BITS)", "FAST ? 2 : 1")
    ),
}

# The branch of each bit width in MLX's load_vector and qdot, over a lane's
# VALUES_PER_THREAD values. A kernel's source holds its own width's branch
# alone: MLX hashes the whole source at every launch.
_LOAD = {
    2: r"""  for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
    sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 4.0f;
    x_thread[i + 2] = x[i + 2] / 16.0f;
    x_thread[i + 3] = x[i + 3] / 64.0f;
  }""",
    3: r"""  for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
    sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
        float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 8.0f;
    x_thread[i + 2] = x[i + 2] / 64.0f;
    x_thread[i + 3] = x[i + 3] / 2.0f;
    x_thread[i + 4] = x[i + 4] / 16.0f;
    x_thread[i + 5] = x[i + 5] / 128.0f;
    x_thread[i + 6] = x[i + 6] / 4.0f;
    x_thread[i + 7] = x[i + 7] / 32.0f;
  }""",
    4: r"""  for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
    sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }""",
    5: r"""  for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
    sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) +
        float(x[i + 4]) + float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 32.0f;
    x_thread[i + 2] = x[i + 2] / 4.0f;
    x_thread[i + 3] = x[i + 3] / 128.0f;
    x_thread[i + 4] = x[i + 4] / 16.0f;
    x_thread[i + 5] = x[i + 5] / 2.0f;
    x_thread[i + 6] = x[i + 6] / 64.0f;
    x_thread[i + 7] = x[i + 7] / 8.0f;
  }""",
    6: r"""  for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
    sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 64.0f;
    x_thread[i + 2] = x[i + 2] / 16.0f;
    x_thread[i + 3] = x[i + 3] / 4.0f;
  }""",
    8: r"""  for (int i = 0; i < VALUES_PER_THREAD; i++) {
    sum += x[i];
    x_thread[i] = x[i];
  }""",
}

_QDOT = {
    2: r"""  for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
    accum +=
        (x_thread[4 * i] * (w[i] & 0x03) +
         x_thread[4 * i + 1] * (w[i] & 0x0c) +
         x_thread[4 * i + 2] * (w[i] & 0x30) +
         x_thread[4 * i + 3] * (w[i] & 0xc0));
  }""",
    3: r"""  for (int i = 0; i < (VALUES_PER_THREAD / 8); i++) {
    x_thread += 8 * i;
    w += 3 * i;

    accum += (w[0] & 0x07) * x_thread[0];
    accum += (w[0] & 0x38) * x_thread[1];
    accum += (w[0] & 0xc0) * x_thread[2];
    accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

    accum += (w[1] & 0x0e) * x_thread[3];
    accum += (w[1] & 0x70) * x_thread[4];
    accum += (w[1] & 0x80) * x_thread[5];
    accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

    accum += (w[2] & 0x1c) * x_thread[6];
    accum += (w[2] & 0xe0) * x_thread[7];
  }""",
    4: r"""  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }""",
    5: r"""  for (int i = 0; i < (VALUES_PER_THREAD / 8); i++) {
    x_thread += 8 * i;
    w += 5 * i;

    accum += (w[0] & 0x1f) * x_thread[0];
    accum += (w[0] & 0xe0) * x_thread[1];
    accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
    accum += (w[1] & 0x7c) * x_thread[2];
    accum += (w[1] & 0x80) * x_thread[3];
    accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
    accum += (w[2] & 0xf0) * x_thread[4];
    accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
    accum += (w[3] & 0x3e) * x_thread[5];
    accum += (w[3] & 0xc0) * x_thread[6];
    accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
    accum += (w[4] & 0xf8) * x_thread[7];
  }""",
    6: r"""  for (int i = 0; i < (VALUES_PER_THREAD / 4); i++) {
    x_thread += 4 * i;
    w += 3 * i;

    accum += (w[0] & 0x3f) * x_thread[0];

    accum += (w[0] & 0xc0) * x_thread[1];
    accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

    accum += (w[1] & 0xf0) * x_thread[2];
    accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

    accum += (w[2] & 0xfc) * x_thread[3];
  }""",
    8: r"""  for (int i = 0; i < VALUES_PER_THREAD; i++) {
    accum += x_thread[i] * w[i];
  }""",
}

_LOAD_FUNCTION = r"""
template <typename XP>
inline float load_vector(XP x, thread float* x_thread) {
  float sum = 0;
__LOAD__
  return sum;
}
"""

_QDOT_FUNCTION = r"""
inline float qdot(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0;
__QDOT__
  return scale * accum + sum * bias;
}
"""


def header(bits: int, group_size: int, fast: bool) -> str:
    """MLX's constants and ``load_vector`` for ``bits``-bit weights in groups
    of ``group_size`` on the ``qmv_fast`` (``fast``) or ``qmv`` traversal:
    the start of a kernel's source over MLX's affine layout."""
    if bits not in _BITS or group_size not in _GROUP_SIZES:
        # MLX does not quantize to another group size, and the transcription
        # has no other bit width.
        raise ValueError(
            f"affine: no MLX transcription of {bits} bits in groups of {group_size}"
        )
    pack_factor, packs = _PACKING[bits]
    return (
        (_CONSTANTS + _LOAD_FUNCTION.replace("__LOAD__", _LOAD[bits]))
        .replace("__PACK_FACTOR__", pack_factor)
        .replace("__PACKS__", packs)
        .replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__FAST__", "1" if fast else "0")
    )


def qmv_fast_layout(k: int, n: int, bits: int, *, gathered: bool = False) -> bool:
    """MLX 0.32.4's one-row affine dispatch: K must align to the kernel block
    (512 for 2- to 5-bit, 256 for 6/8-bit). Only gathered QMV still requires N
    to be a multiple of 8; ordinary QMV supports partial output tiles."""
    pack_factor = 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)
    return (not gathered or n % 8 == 0) and k % (
        pack_factor * (1 if bits == 2 else 2) * 32
    ) == 0


# qmv's ``qdot`` split in two: the weight terms of a full pack run are decoded
# once per K block and output column, then every row accumulates them against
# its own inputs. Each term and each accumulation step is qdot's, in qdot's
# order (the 3/5/6-bit runs cover at most two packs, so qdot's cumulative
# pointer steps are the 8i/3i, 8i/5i and 4i/3i offsets used here). A tile holds the
# branch of its own width.
_DECODED_DOT = r"""
constant constexpr int W_TERMS = __W_TERMS__;

inline void decode_w(const device uint8_t* w, thread float* wq) {
__DECODE__
}

inline float wdot(const thread float* wq, const thread float* x) {
  float accum = 0;
__WDOT__
  return accum;
}
"""

_DECODE = {
    2: r"""  for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
    wq[4 * i] = w[i] & 0x03;
    wq[4 * i + 1] = w[i] & 0x0c;
    wq[4 * i + 2] = w[i] & 0x30;
    wq[4 * i + 3] = w[i] & 0xc0;
  }""",
    3: r"""  for (int i = 0; i < VALUES_PER_THREAD / 8; i++) {
    const device uint8_t* wb = w + 3 * i;
    thread float* wv = wq + 10 * i;
    wv[0] = wb[0] & 0x07;
    wv[1] = wb[0] & 0x38;
    wv[2] = wb[0] & 0xc0;
    wv[3] = wb[1] & 0x01;
    wv[4] = wb[1] & 0x0e;
    wv[5] = wb[1] & 0x70;
    wv[6] = wb[1] & 0x80;
    wv[7] = wb[2] & 0x03;
    wv[8] = wb[2] & 0x1c;
    wv[9] = wb[2] & 0xe0;
  }""",
    4: r"""  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
    wq[4 * i] = ws[i] & 0x000f;
    wq[4 * i + 1] = ws[i] & 0x00f0;
    wq[4 * i + 2] = ws[i] & 0x0f00;
    wq[4 * i + 3] = ws[i] & 0xf000;
  }""",
    5: r"""  for (int i = 0; i < VALUES_PER_THREAD / 8; i++) {
    const device uint8_t* wb = w + 5 * i;
    thread float* wv = wq + 12 * i;
    wv[0] = wb[0] & 0x1f;
    wv[1] = wb[0] & 0xe0;
    wv[2] = wb[1] & 0x3;
    wv[3] = wb[1] & 0x7c;
    wv[4] = wb[1] & 0x80;
    wv[5] = wb[2] & 0xf;
    wv[6] = wb[2] & 0xf0;
    wv[7] = wb[3] & 0x1;
    wv[8] = wb[3] & 0x3e;
    wv[9] = wb[3] & 0xc0;
    wv[10] = wb[4] & 0x7;
    wv[11] = wb[4] & 0xf8;
  }""",
    6: r"""  for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
    const device uint8_t* wb = w + 3 * i;
    thread float* wv = wq + 6 * i;
    wv[0] = wb[0] & 0x3f;
    wv[1] = wb[0] & 0xc0;
    wv[2] = wb[1] & 0x0f;
    wv[3] = wb[1] & 0xf0;
    wv[4] = wb[2] & 0x03;
    wv[5] = wb[2] & 0xfc;
  }""",
    8: r"""  for (int i = 0; i < VALUES_PER_THREAD; i++) {
    wq[i] = w[i];
  }""",
}

_WDOT = {
    2: r"""  for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
    accum +=
        (x[4 * i] * wq[4 * i] + x[4 * i + 1] * wq[4 * i + 1] +
         x[4 * i + 2] * wq[4 * i + 2] + x[4 * i + 3] * wq[4 * i + 3]);
  }""",
    3: r"""  for (int i = 0; i < VALUES_PER_THREAD / 8; i++) {
    const thread float* xv = x + 8 * i;
    const thread float* wv = wq + 10 * i;
    accum += wv[0] * xv[0];
    accum += wv[1] * xv[1];
    accum += wv[2] * xv[2];
    accum += wv[3] * (xv[2] * 256.0f);
    accum += wv[4] * xv[3];
    accum += wv[5] * xv[4];
    accum += wv[6] * xv[5];
    accum += wv[7] * (xv[5] * 256.0f);
    accum += wv[8] * xv[6];
    accum += wv[9] * xv[7];
  }""",
    4: r"""  for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
    accum +=
        (x[4 * i] * wq[4 * i] + x[4 * i + 1] * wq[4 * i + 1] +
         x[4 * i + 2] * wq[4 * i + 2] + x[4 * i + 3] * wq[4 * i + 3]);
  }""",
    5: r"""  for (int i = 0; i < VALUES_PER_THREAD / 8; i++) {
    const thread float* xv = x + 8 * i;
    const thread float* wv = wq + 12 * i;
    accum += wv[0] * xv[0];
    accum += wv[1] * xv[1];
    accum += wv[2] * (xv[1] * 256.0f);
    accum += wv[3] * xv[2];
    accum += wv[4] * xv[3];
    accum += wv[5] * (xv[3] * 256.0f);
    accum += wv[6] * xv[4];
    accum += wv[7] * (xv[4] * 256.0f);
    accum += wv[8] * xv[5];
    accum += wv[9] * xv[6];
    accum += wv[10] * (xv[6] * 256.0f);
    accum += wv[11] * xv[7];
  }""",
    6: r"""  for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
    const thread float* xv = x + 4 * i;
    const thread float* wv = wq + 6 * i;
    accum += wv[0] * xv[0];
    accum += wv[1] * xv[1];
    accum += wv[2] * (xv[1] * 256.0f);
    accum += wv[3] * xv[2];
    accum += wv[4] * (xv[2] * 256.0f);
    accum += wv[5] * xv[3];
  }""",
    8: r"""  for (int i = 0; i < VALUES_PER_THREAD; i++) {
    accum += x[i] * wq[i];
  }""",
}


# The weight terms a lane decodes of its VALUES_PER_THREAD inputs, by width:
# a 3-, 5- or 6-bit run's packs hold more terms than values.
_W_TERMS = {
    3: "10 * VALUES_PER_THREAD / 8",
    5: "12 * VALUES_PER_THREAD / 8",
    6: "6 * VALUES_PER_THREAD / 4",
}


def decoded(bits: int) -> str:
    """``decode_w`` and ``wdot`` of ``bits``-bit weights: qmv's ``qdot`` split
    in two, after ``header``."""
    return (
        _DECODED_DOT.replace("__W_TERMS__", _W_TERMS.get(bits, "VALUES_PER_THREAD"))
        .replace("__DECODE__", _DECODE[bits])
        .replace("__WDOT__", _WDOT[bits])
    )


# One lane of qmv_fast_impl (FAST) or qmv_impl: its pack(s) of each row's
# block, the group's scale and bias, and MLX's qdot on them. qmv_impl reads
# its last block through load_vector_safe and qdot_safe; a lane has all of
# its VALUES_PER_THREAD inputs there or none (K is a multiple of the group
# size), and for all of them the two are load_vector and qdot, operation for
# operation. Decoded, decode and apply are declared here and defined in the
# face's shared source (_SHARED), which only the kernels that apply a block
# to several inputs include: MLX hashes a kernel's whole source at every
# launch.
_QMV_LAYOUT_CHECK = r"""    static_assert(
        !FAST || K % BLOCK_SIZE == 0,
        "affine qmv_fast face: K must be whole blocks; AffineFormat.dot gives a K its face");"""

_PARTIAL_LAYOUT_CHECK = r"""    static_assert(
        K % VPT == 0 && K % GS == 0,
        "affine partial block: K must be whole lanes and groups");"""

_DOT = r"""
template <typename T>
struct Dot {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device T* scales;
  const device T* biases;

  template <int K>
  struct Lanes {
    static constant constexpr int VPT = VALUES_PER_THREAD;
    static constant constexpr int BLOCK = BLOCK_SIZE;
    static constant constexpr int ROW_W = K * BYTES_PER_PACK / PACK_FACTOR;
    static constant constexpr int ROW_G = K / GS;
__LAYOUT_CHECK__

    const device uint8_t* w;
    const device T* s;
    const device T* b;

    Lanes(Dot m, size_t row, uint lane)
        : w(m.codes + row * ROW_W + lane * PACKS_PER_THREAD * BYTES_PER_PACK),
          s(m.scales + row * ROW_G + lane / SCALE_STEP_PER_THREAD),
          b(m.biases + row * ROW_G + lane / SCALE_STEP_PER_THREAD) {}

    struct Row {
      const device uint8_t* w;
      float s;
      float b;
    };

    Row row(int r) const {
      return {w + r * ROW_W, float(s[r * ROW_G]), float(b[r * ROW_G])};
    }

    void step() {
      w += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      s += BLOCK_SIZE / GS;
      b += BLOCK_SIZE / GS;
    }

    template <typename XP>
    static float load(XP x, thread float* x_thread) {
      return load_vector(x, x_thread);
    }

    static float madd(float acc, Row r, const thread float* x_thread, float sum) {
      return acc + qdot(r.w, x_thread, r.s, r.b, sum);
    }

    struct Decoded;
    static Decoded decode(Row r);
    static float apply(
        float acc, thread const Decoded& d, const thread float* x_thread, float sum);

    float finish(int, float sum) const { return sum; }
  };
};
"""


# What the shared-rows kernels read of a face beyond the row skeleton: one
# row's weights of a block decoded once (qdot's terms, decode_w), applied to
# each input (wdot), as members the face's struct declares.
_SHARED = r"""
template <typename T>
template <int K>
struct Dot<T>::Lanes<K>::Decoded {
  float wq[W_TERMS];
  float s;
  float b;
};

template <typename T>
template <int K>
METAL_FUNC typename Dot<T>::template Lanes<K>::Decoded Dot<T>::Lanes<K>::decode(Row r) {
  Decoded d;
  decode_w(r.w, d.wq);
  d.s = r.s;
  d.b = r.b;
  return d;
}

template <typename T>
template <int K>
METAL_FUNC float Dot<T>::Lanes<K>::apply(
    float acc, thread const Decoded& d, const thread float* x_thread, float sum) {
  return acc + (d.s * wdot(d.wq, x_thread) + sum * d.b);
}
"""


def values_per_thread(bits: int, fast: bool) -> int:
    """VALUES_PER_THREAD of ``bits``-bit weights on the ``qmv_fast`` (``fast``)
    or ``qmv`` traversal: MLX's pack factor times the packs a lane reads."""
    pack_factor = 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)
    return pack_factor * (2 if fast and bits != 2 else 1)


@cache
def face(
    bits: int, group_size: int, fast: bool, dtype: mx.Dtype, *, partial_blocks=False
) -> Dot:
    """The dot face of ``bits``-bit weights with ``group_size`` groups, as
    MLX's ``qmv_fast`` (``fast``) or ``qmv`` traversal. A kernel that
    handles its own tail lanes can request ``partial_blocks`` to walk two
    packs per lane even when the last simdgroup's block is incomplete."""
    source = header(bits, group_size, fast)
    key = f"affine_b{bits}g{group_size}{'f' if fast else 's'}"
    if partial_blocks:
        key += "_partial"
    check = _PARTIAL_LAYOUT_CHECK if partial_blocks else _QMV_LAYOUT_CHECK
    dot = _DOT.replace("__LAYOUT_CHECK__", check)
    qdot = _QDOT_FUNCTION.replace("__QDOT__", _QDOT[bits])
    source += qdot
    return Dot(
        key=key,
        source=(
            f"namespace omlx_{key} {{\n{source}\n{dot}\n}}"
            f"  // namespace omlx_{key}\n"
        ),
        shared=(
            f"namespace omlx_{key} {{\n{decoded(bits)}{_SHARED}}}"
            f"  // namespace omlx_{key}\n"
        ),
        vpt=values_per_thread(bits, fast),
        type=f"omlx_{key}::Dot<T>",
        tensors=AffineFormat.tensors,
        dtypes=(dtype,),
    )


# The serial HC up walks adjacent lanes. Keep the group scalars in the row
# while advancing its packed codes, with one native lane of decoded weights
# live at a time. Each lane contributes its own rounded partial sum.
_STREAM_UP = r"""
template <int R, typename T, typename X>
METAL_FUNC float hc_stream_up(Dot<T> m, int n, X a) {
    using L = typename Dot<T>::template Lanes<R>;
    static_assert(R % GS == 0 && GS % L::VPT == 0, "whole affine groups");
    constexpr int chunks = GS / L::VPT;
    float acc = 0.0f;
    for (int g = 0; g < R / GS; ++g) {
        auto group = L(m, n, g * chunks).row(0);
        for (int chunk = 0; chunk < chunks; ++chunk) {
            float xv[L::VPT];
            const auto aux = L::load(a + g * GS + chunk * L::VPT, xv);
            const auto weights = L::decode(group);
            acc += L::apply(0.0f, weights, xv, aux);
            group.w += PACKS_PER_THREAD * BYTES_PER_PACK;
        }
    }
    return L(m, n, 0).finish(0, acc);
}
"""


@cache
def serial_up_header(face: Dot) -> str:
    """The shared affine face and its serial HC dot in the face's namespace."""
    namespace = f"omlx_{face.key}"
    return (
        f"{face.shared}\nnamespace {namespace} {{\n{_STREAM_UP}\n}}\n"
        f"using {namespace}::hc_stream_up;\n"
    )


_TILE_BITS = (4, 8)

# A run of 8 codes is BITS bytes: one 32-bit word at 4 bits, two at 8, read
# where they lie (an 8-bit run is two words, which a weight's slice aligns to
# 4 bytes, not 8: a packed_uint2). Each width's source has its run's type and
# the code at position i alone. K is asserted whole groups in the
# constructor, not in the class: a failed assert there would leave Runs<K>
# without members for the skeleton's checks, each then an error of its own.
_TILE = r"""
using namespace metal;

constant constexpr int BITS = __BITS__;
constant constexpr int GS = __GS__;

template <typename S>
struct Tile {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device S* scales;
  const device S* biases;

  template <int K>
  struct Runs {
    static constant constexpr int RUN = 8;
    static constant constexpr int GROUP = GS;
    static constant constexpr int ROW_BYTES = K / 8 * BITS;
    static constant constexpr int ROW_GROUPS = K / GS;
    struct P {
      float s;
      float b;
    };
__RUNS__
    const device S* s;
    const device S* b;

    Runs(Tile m, size_t row)
        : c((const device Stored*)(m.codes + row * ROW_BYTES)),
          s(m.scales + row * ROW_GROUPS),
          b(m.biases + row * ROW_GROUPS) {
      static_assert(
          K % GS == 0,
          "affine tile face: K must be whole groups; AffineFormat.tile gives a K "
          "its face");
    }

    P params(int g) const {
      return {float(s[g]), float(b[g])};
    }
    Codes fetch(int run) const {
      return Codes(c[run]);
    }
    static float weight(thread const P& p, Codes q, int i) {
      return p.s * float(code_at(q, i)) + p.b;
    }
  };
};
"""

# A run's type as a kernel holds it, as it is stored, and its code i, by width.
_RUNS = {
    4: r"""    using Codes = uint;
    using Stored = uint;
    const device Stored* c;
    static uint code_at(uint q, int i) {
      return (q >> (4 * i)) & 0xFu;
    }""",
    8: r"""    using Codes = uint2;
    using Stored = packed_uint2;
    const device Stored* c;
    static uint code_at(uint2 q, int i) {
      return ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xFFu;
    }""",
}

# Metal's names of the scales' types.
_METAL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t"}


def _tile_source(bits: int, group_size: int) -> str:
    """The Metal of the affine tile face of ``bits``-bit codes (4 or 8) in
    groups of ``group_size``: ``omlx_affine_tile_b{bits}g{group_size}::Tile<S>``,
    S the scales' type. One source serves both types of scales."""
    namespace = f"omlx_affine_tile_b{bits}g{group_size}"
    tile = (
        _TILE.replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__RUNS__", _RUNS[bits])
    )
    return f"namespace {namespace} {{\n{tile}\n}}  // namespace {namespace}\n"


@cache
def tile_face(bits: int, group_size: int, dtype: mx.Dtype) -> Tile:
    """The tile face of ``bits``-bit weights with ``group_size`` groups and
    scales and biases of ``dtype``, exact for activations of that dtype."""
    if bits not in _TILE_BITS or group_size not in _GROUP_SIZES:
        raise ValueError(
            f"affine tile face: no runs of {bits} bits in groups of {group_size}"
        )
    if dtype not in _METAL_TYPES:
        raise ValueError(f"affine tile face: no scales of {dtype}")
    suffix = "bf16" if dtype == mx.bfloat16 else "fp16"
    return Tile(
        key=f"affine_tile_b{bits}g{group_size}_{suffix}",
        source=_tile_source(bits, group_size),
        type=f"omlx_affine_tile_b{bits}g{group_size}::Tile<{_METAL_TYPES[dtype]}>",
        tensors=AffineFormat.tensors,
        dtypes=(dtype,),
    )


_CODE_BITS = (2, 3, 4, 5, 6)
_CODE_TYPES = {**_METAL_TYPES, mx.float32: "float"}

# Read exactly the words containing one eight-level run. The last run of a
# row can end at a word boundary and must not read into the following row.
_PACKED_RUN = """
    uint2 fetch(int run) const {
      const device uint* block = w + size_t(run >> 2) * BITS;
      uint at = uint(run & 3) * BITS;
      uint shift = (at % 4) * 8;
      ulong v = ulong(block[at / 4]);
      if (shift + 8 * BITS > 32) v |= ulong(block[at / 4 + 1]) << 32;
      v >>= shift;
      return uint2(uint(v), uint(v >> (4 * BITS)));
    }
    static uint pair(uint2 q, int j) {
      constexpr uint MASK = (1u << BITS) - 1;
      uint2 p = (q >> (BITS * j)) & MASK;
      return p.x | (p.y << 16);
    }"""

_NIBBLE_RUN = """
    uint fetch(int run) const { return w[run]; }
    static uint pair(uint q, int j) { return (q >> (4 * j)) & 0x000f000fu; }"""


@cache
def codes_face(bits, group_size, scale_dtype, bias_dtype) -> Codes:
    """The affine integer levels, with the parameter types stored on disk."""
    if bits not in _CODE_BITS or group_size not in _GROUP_SIZES:
        raise ValueError(f"affine codes face: {bits} bits in groups of {group_size}")
    if scale_dtype not in _CODE_TYPES or bias_dtype not in _CODE_TYPES:
        raise ValueError(
            f"affine codes face: {scale_dtype} scales and {bias_dtype} biases"
        )
    scale, bias = _CODE_TYPES[scale_dtype], _CODE_TYPES[bias_dtype]
    key = f"affine_codes_b{bits}g{group_size}_{scale}_{bias}"
    source = f"""
namespace omlx_{key} {{
struct Codes {{
  static constant constexpr int CONTRACT = 1;
  const device uchar* weight;
  const device {scale}* scales;
  const device {bias}* biases;
  template <int K>
  struct Runs {{
    static constant constexpr int BITS = {bits};
    static constant constexpr int GROUP = {group_size};
    const device uint* w;
    const device {scale}* s;
    const device {bias}* b;
    Runs(Codes m, size_t row)
      : w((const device uint*)m.weight + row * (K / 32 * BITS)),
        s(m.scales + row * (K / GROUP)), b(m.biases + row * (K / GROUP)) {{}}
    float2 params(int g) const {{ return float2(float(s[g]), float(b[g])); }}
    {_NIBBLE_RUN if bits == 4 else _PACKED_RUN}
  }};
}};
}}
"""
    return Codes(
        key,
        source,
        f"omlx_{key}::Codes",
        AffineFormat.tensors,
        (mx.bfloat16, mx.float16),
    )


class AffineFormat(QuantFormat):
    mode = "affine"
    tensors = ("scales", "biases")
    contract = 1

    def dot(self, layer, k: int, n: int) -> Dot | None:
        if (
            layer.bits not in _BITS
            or layer.group_size not in _GROUP_SIZES
            or "scales" not in layer
            or "biases" not in layer
            or layer["scales"].dtype not in (mx.bfloat16, mx.float16)
            or layer["biases"].dtype != layer["scales"].dtype
            # One matrix, or a stack of them (a switch layer's experts).
            or layer["weight"].ndim not in (2, 3)
            or layer["weight"].shape[-2:] != (n, k * layer.bits // 32)
            or k % layer.group_size
            # MLX binds an input of fewer than 8 elements as constants; the
            # scales and the biases are a matrix's smallest.
            or n * k // layer.group_size < 8
            # qmv_quad: MLX's kernel for one row of a 2-D matrix of K 64 or
            # 128 at a power-of-two bit width.
            or (
                layer["weight"].ndim == 2 and k in (64, 128) and layer.bits in (2, 4, 8)
            )
        ):
            return None
        # Stacked (3-D) weights are the ones MLX runs through gather_qmm.
        fast = qmv_fast_layout(k, n, layer.bits, gathered=layer["weight"].ndim == 3)
        return face(layer.bits, layer.group_size, fast, layer["scales"].dtype)

    def tile(self, layer, k: int, n: int) -> Tile | None:
        if (
            layer.bits not in _TILE_BITS
            or layer.group_size not in _GROUP_SIZES
            or not all(
                isinstance(layer.get(name), mx.array)
                for name in ("weight", "scales", "biases")
            )
        ):
            return None
        weight, scales, biases = layer["weight"], layer["scales"], layer["biases"]
        # A matrix, or a stack of them; codes in 32-bit words; one scale and
        # one bias per group of each row.
        groups = (*weight.shape[:-1], k // layer.group_size)
        if (
            weight.dtype != mx.uint32
            or weight.ndim not in (2, 3)
            or k % layer.group_size
            or weight.shape[-2:] != (n, k * layer.bits // 32)
            or scales.dtype not in _METAL_TYPES
            or biases.dtype != scales.dtype
            or scales.shape != groups
            or biases.shape != groups
            # MLX binds an input of fewer than 8 elements as constants.
            or scales.size < 8
        ):
            return None
        return tile_face(layer.bits, layer.group_size, scales.dtype)

    def codes(self, layer, k: int, n: int) -> Codes | None:
        bits, group = layer.bits, layer.group_size
        if bits not in _CODE_BITS or group not in _GROUP_SIZES or k <= 0 or k % group:
            return None
        weight, scales, biases = (layer.get(key) for key in ("weight", *self.tensors))
        if not all(isinstance(a, mx.array) for a in (weight, scales, biases)):
            return None
        if (
            weight.dtype != mx.uint32
            or weight.ndim not in (2, 3)
            or weight.shape[-2:] != (n, k // 32 * bits)
            or scales.shape != (*weight.shape[:-1], k // group)
            or biases.shape != scales.shape
            or scales.dtype not in _CODE_TYPES
            or biases.dtype not in _CODE_TYPES
            or min(weight.size, scales.size) < 8
        ):
            return None
        return codes_face(bits, group, scales.dtype, biases.dtype)

    def decode(self, tensors, bits: int, group_size: int) -> mx.array:
        return mx.dequantize(
            tensors["weight"],
            tensors["scales"],
            tensors["biases"],
            group_size=group_size,
            bits=bits,
            mode="affine",
        )


FORMAT = AffineFormat()
