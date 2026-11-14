# SPDX-License-Identifier: Apache-2.0
"""MLX's affine quantization (``mode="affine"``, the oQ and oQe checkpoints)
as a weight format.

The dot face is the MLX 0.32.4 ``quantized.h`` transcription in
``omlx.patches.moe_verify_gather``: a kernel instantiated with it computes
each row as MLX's ``qmv_fast`` or ``qmv`` does, bit for bit, for bf16 or fp16
activations with scales in the same dtype. MLX runs ``qmv_fast`` when K is a
multiple of the kernel block, and for one row through ``gather_qmm`` (stacked
weights) when N is a multiple of 8 as well. One row of a 2-D matrix of K 64
or 128 at 4 or 8 bits is a third MLX kernel, ``qmv_quad``, which sums in
another order: such a matrix has no face.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from ..patches.moe_verify_gather import _BITS, _GROUP_SIZES, _HEADER, qmv_fast_layout
from . import Dot, QuantFormat

# One lane of qmv_fast_impl (FAST) or qmv_impl: its pack(s) of each row's
# block, the group's scale and bias, and MLX's qdot on them. qmv_impl reads
# its last block through load_vector_safe and qdot_safe; a lane has all of
# its VALUES_PER_THREAD inputs there or none (K is a multiple of the group
# size), and for all of them the two are load_vector and qdot, operation for
# operation.
_DOT = r"""
template <typename T>
struct Dot {
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
      return acc + qdot_n(r.w, x_thread, r.s, r.b, sum, VALUES_PER_THREAD);
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
        # MLX does not quantize to another group size. For another bit width
        # the transcription compiles, and computes zeros.
        raise ValueError(
            f"affine face: no MLX transcription of {bits} bits in groups of "
            f"{group_size}"
        )
    key = f"affine_b{bits}g{group_size}{'f' if fast else 's'}"
    header = (
        _HEADER.replace("__BITS__", str(bits))
        .replace("__GS__", str(group_size))
        .replace("__FAST__", "1" if fast else "0")
    )
    return Dot(
        key=key,
        source=f"namespace omlx_{key} {{\n{header}\n{_DOT}\n}}  // namespace omlx_{key}\n",
        type=f"omlx_{key}::Dot<T>",
        tensors=AffineFormat.tensors,
        dtypes=(dtype,),
    )


class AffineFormat(QuantFormat):
    mode = "affine"
    tensors = ("scales", "biases")

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


FORMAT = AffineFormat()
