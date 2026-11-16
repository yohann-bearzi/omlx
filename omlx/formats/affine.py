# SPDX-License-Identifier: Apache-2.0
"""MLX's affine quantization (``mode="affine"``, the oQ and oQe checkpoints)
as a weight format.

The dot face is over the MLX 0.32.4 ``quantized.h`` transcription below: a
kernel instantiated with it computes each row as MLX's ``qmv_fast`` or ``qmv``
does, bit for bit, for bf16 or fp16 activations with scales in the same
dtype. MLX runs ``qmv_fast`` when K is a multiple of the kernel block, and
for one row through ``gather_qmm`` (stacked weights) when N is a multiple of
8 as well. One row of a 2-D matrix of K 64 or 128 at 4 or 8 bits is a third
MLX kernel, ``qmv_quad``, which sums in another order: such a matrix has no
face.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from . import Dot, QuantFormat

_BITS = (4, 5, 6, 8)
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
    (BITS == 5) ? 8 : (BITS == 6 ? 4 : 32 / BITS);
constant constexpr int BYTES_PER_PACK =
    ((BITS & (BITS - 1)) == 0) ? 4 : (BITS == 5 ? 5 : 3);
constant constexpr int PACKS_PER_THREAD = FAST ? 2 : 1;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
"""

# The branch of each bit width in MLX's load_vector and qdot, over the first
# __N__ of a lane's values: all VALUES_PER_THREAD of them, or N in qmv_impl's
# last block (load_vector_safe, qdot_safe). A kernel's source holds its own
# width's branch alone: MLX hashes the whole source at every launch.
_LOAD = {
    4: r"""  for (int i = 0; i < __N__; i += 4) {
    sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }""",
    5: r"""  for (int i = 0; i < __N__; i += 8) {
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
    6: r"""  for (int i = 0; i < __N__; i += 4) {
    sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 64.0f;
    x_thread[i + 2] = x[i + 2] / 16.0f;
    x_thread[i + 3] = x[i + 3] / 4.0f;
  }""",
    8: r"""  for (int i = 0; i < __N__; i++) {
    sum += x[i];
    x_thread[i] = x[i];
  }""",
}

_QDOT = {
    4: r"""  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (__N__ / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }""",
    5: r"""  for (int i = 0; i < (__N__ / 8); i++) {
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
    6: r"""  for (int i = 0; i < (__N__ / 4); i++) {
    x_thread += 4 * i;
    w += 3 * i;

    accum += (w[0] & 0x3f) * x_thread[0];

    accum += (w[0] & 0xc0) * x_thread[1];
    accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

    accum += (w[1] & 0xf0) * x_thread[2];
    accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

    accum += (w[2] & 0xfc) * x_thread[3];
  }""",
    8: r"""  for (int i = 0; i < __N__; i++) {
    accum += x_thread[i] * w[i];
  }""",
}

_LOAD_FUNCTION = r"""
template <typename T>
inline float load_vector(const device T* x, thread float* x_thread) {
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
    return (
        (_CONSTANTS + _LOAD_FUNCTION.replace("__LOAD__", _LOAD[bits]))
        .replace("__N__", "VALUES_PER_THREAD")
        .replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__FAST__", "1" if fast else "0")
    )


def qmv_fast_layout(k: int, n: int, bits: int, *, gathered: bool = False) -> bool:
    """MLX 0.32.4's one-row affine dispatch: K must align to the kernel block
    (512 for 4/5-bit, 256 for 6/8-bit). Only gathered QMV still requires N
    to be a multiple of 8; ordinary QMV supports partial output tiles."""
    pack_factor = 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)
    return (not gathered or n % 8 == 0) and k % (
        pack_factor * (1 if bits == 2 else 2) * 32
    ) == 0


# One lane of qmv_fast_impl (FAST) or qmv_impl: its pack(s) of each row's
# block, the group's scale and bias, and MLX's qdot on them. qmv_impl reads
# its last block through load_vector_safe and qdot_safe; a lane has all of
# its VALUES_PER_THREAD inputs there or none (K is a multiple of the group
# size), and for all of them the two are load_vector and qdot, operation for
# operation.
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
    static_assert(
        !FAST || K % BLOCK_SIZE == 0,
        "affine qmv_fast face: K must be whole blocks; AffineFormat.dot gives a K its face");

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

    template <typename X>
    static float load(const device X* x, thread float* x_thread) {
      return load_vector<X>(x, x_thread);
    }

    static float madd(float acc, Row r, const thread float* x_thread, float sum) {
      return acc + qdot(r.w, x_thread, r.s, r.b, sum);
    }

    float finish(int, float sum) const { return sum; }
  };
};
"""


@cache
def face(bits: int, group_size: int, fast: bool, dtype: mx.Dtype) -> Dot:
    """The dot face of ``bits``-bit weights with ``group_size`` groups, as
    MLX's ``qmv_fast`` (``fast``) or ``qmv`` traversal."""
    if bits not in _BITS or group_size not in _GROUP_SIZES:
        # MLX does not quantize to another group size, and the transcription
        # has no other bit width.
        raise ValueError(
            f"affine face: no MLX transcription of {bits} bits in groups of "
            f"{group_size}"
        )
    key = f"affine_b{bits}g{group_size}{'f' if fast else 's'}"
    qdot = _QDOT_FUNCTION.replace("__QDOT__", _QDOT[bits])
    source = header(bits, group_size, fast) + qdot.replace("__N__", "VALUES_PER_THREAD")
    return Dot(
        key=key,
        source=(
            f"namespace omlx_{key} {{\n{source}\n{_DOT}\n}}"
            f"  // namespace omlx_{key}\n"
        ),
        type=f"omlx_{key}::Dot<T>",
        tensors=AffineFormat.tensors,
        dtypes=(dtype,),
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
            or (layer["weight"].ndim == 2 and k in (64, 128) and layer.bits in (4, 8))
        ):
            return None
        # Stacked (3-D) weights are the ones MLX runs through gather_qmm.
        fast = qmv_fast_layout(k, n, layer.bits, gathered=layer["weight"].ndim == 3)
        return face(layer.bits, layer.group_size, fast, layer["scales"].dtype)

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
