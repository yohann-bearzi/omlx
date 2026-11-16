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
same weight (``scale * q`` is exact in float32), and what MLX's sorted
``gather_qmm`` loads into its tiles. It reads runs of 4 and 8 bits only.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from . import Dot, QuantFormat, Tile

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
    source = header(bits, group_size, fast)
    key = f"affine_b{bits}g{group_size}{'f' if fast else 's'}"
    qdot = _QDOT_FUNCTION.replace("__QDOT__", _QDOT[bits])
    source += qdot
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


_TILE_BITS = (4, 8)

# A run of 8 codes is BITS bytes: one 32-bit word at 4 bits, two at 8. The
# code at position i comes from an overload on the run's type: BITS is not a
# template parameter of Tile, so an `if constexpr` on it would not discard
# the branch of the other width, which does not compile for this one.
_TILE = r"""
using namespace metal;

constant constexpr int BITS = __BITS__;
constant constexpr int GS = __GS__;
static_assert(
    BITS == 4 || BITS == 8,
    "affine tile face: runs of 4 and 8 bits only; 2- and 3-bit runs need a "
    "fetch of their own");

template <typename S>
struct Tile {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device S* scales;
  const device S* biases;

  struct Runs {
    static constant constexpr int RUN = 8;
    static constant constexpr int GROUP = GS;
    struct P {
      float s;
      float b;
    };
    using Codes = metal::conditional_t<BITS == 8, uint2, uint>;

    const device Codes* c;
    const device S* s;
    const device S* b;

    Runs(Tile m, size_t row, int k)
        : c((const device Codes*)(m.codes + row * size_t(k / 8 * BITS))),
          s(m.scales + row * size_t((k + GS - 1) / GS)),
          b(m.biases + row * size_t((k + GS - 1) / GS)) {}

    P params(int g) const {
      return {float(s[g]), float(b[g])};
    }
    Codes fetch(int run) const {
      return c[run];
    }
    static uint code_at(uint q, int i) {
      return (q >> (BITS * i)) & ((1u << BITS) - 1u);
    }
    static uint code_at(uint2 q, int i) {
      return ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xFFu;
    }
    static float weight(thread const P& p, Codes q, int i) {
      return p.s * float(code_at(q, i)) + p.b;
    }
  };
};
"""

# Metal's names of the scales' types.
_METAL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t"}


def _tile_source(bits: int, group_size: int) -> str:
    """The Metal of the affine tile face of ``bits``-bit codes in groups of
    ``group_size``: ``omlx_affine_tile_b{bits}g{group_size}::Tile<S>``, S the
    scales' type. One source serves both types of scales."""
    namespace = f"omlx_affine_tile_b{bits}g{group_size}"
    tile = _TILE.replace("__BITS__", str(bits)).replace("__GS__", str(group_size))
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
            or "scales" not in layer
            or "biases" not in layer
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
