# SPDX-License-Identifier: Apache-2.0
"""oQh as a built-in format: rotated uniform codes with a row scale, or with
MLX affine group scales, in the basis of a signed block Walsh-Hadamard
transform of the input.

The group-scale format reads MLX's affine faces (the same skeletons and
per-chip tuning tables every stock affine layer uses). The row-scale format
adds one face member, ``rscale``: the dot and tile sources are MLX's affine
load and qdot with unit scale and ``-center`` bias, times the row's scale
where the row ends. Tile faces read runs of eight codes as ``BITS`` bytes in
the packing MLX writes, including the reinvested 5-bit experts.

A layer's input is rotated once by the format's ``InputTransform`` before
any face reads it; ``decode`` gives the weight in that rotated basis, and
``transforms.inverse_weight`` recovers the plain basis for references.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from functools import cache

import mlx.core as mx

from . import Codes, Dot, QuantFormat, Tile, affine, transforms
from .reads import (
    ApplyRead,
    CodesReads,
    DotReads,
    LoadRead,
    MaddRead,
    PairRead,
    TileReads,
)

_BITS = (2, 3, 4, 5, 6, 8)
_ROW_GROUP = 128
_BLOCKS = (128, 256)
_TILE_GROUPS = (32, 64, 128)
_METAL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t"}
_CENTER = {bits: ((1 << bits) - 1) / 2.0 for bits in _BITS}

# The row-scale dot: MLX's affine load and qdot with unit scale and the center
# as the bias; the row's rscale multiplies the finished sum. Read members are
# declared (reads=...) and emitted into the markers below, as every face's are.
_ROW_DOT = r"""
constant constexpr float CENTER = __CENTER__;

template <typename T>
struct Dot {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device T* rscale;

  template <int K>
  struct Lanes {
    static constant constexpr int VPT = VALUES_PER_THREAD;
    static constant constexpr int BLOCK = BLOCK_SIZE;
    static constant constexpr int ROW_W = K * BYTES_PER_PACK / PACK_FACTOR;
    static_assert(
        K % VPT == 0,
        "oQh dot face: K must be a whole number of a lane's inputs; OqhFormat.dot gives a K its face");

    const device uint8_t* w;
    const device T* rs;

    Lanes(Dot m, size_t row, uint lane)
        : w(m.codes + row * ROW_W + lane * PACKS_PER_THREAD * BYTES_PER_PACK),
          rs(m.rscale + row) {}

    struct Row {
      const device uint8_t* w;
    };

    Row row(int r) const { return {w + r * ROW_W}; }

    void step() {
      w += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    }

    __OMLX_READ_LOAD__ {
      return load_vector(x, x_thread);
    }

    __OMLX_READ_MADD__ {
      return acc + qdot(r.w, x_thread, 1.0f, -CENTER, sum);
    }

    float finish(int r, float sum) const { return sum * float(rs[r]); }

    struct Decoded;
    static Decoded decode(Row r);
    __OMLX_READ_APPLY__;
  };
};
"""

_ROW_DOT_SHARED = r"""
template <typename T>
template <int K>
struct Dot<T>::Lanes<K>::Decoded {
  float wq[W_TERMS];
};

template <typename T>
template <int K>
METAL_FUNC typename Dot<T>::template Lanes<K>::Decoded Dot<T>::Lanes<K>::decode(Row r) {
  Decoded d;
  decode_w(r.w, d.wq);
  return d;
}

__OMLX_READ_APPLY_SHARED__ {
  return acc + (wdot(d.wq, x_thread) - sum * CENTER);
}
"""

_ROW_DOT_READS = DotReads(
    load=LoadRead(params=("x", "x_thread")),
    madd=MaddRead(params=("acc", "r", "x_thread", "sum")),
    apply=ApplyRead(
        params=("acc", "d", "x_thread", "sum"),
        shared_slot="apply_shared",
        shared_owner="Dot<T>::Lanes<K>",
        shared_templates=("typename T", "int K"),
        wrap=True,
    ),
)


def _row_dot_namespace(bits: int, fast: bool) -> str:
    return f"oqh_b{bits}{'f' if fast else 's'}"


@cache
def _row_dot(bits: int, fast: bool, dtype: mx.Dtype) -> Dot:
    if bits not in _BITS or dtype not in _METAL_TYPES:
        raise ValueError(f"oQh dot face: no {bits} bits with {dtype} activations")
    # The affine constants carry a group size; the row face reads no group,
    # so it names one whose scale step is whole at every width.
    source = affine.header(bits, 32, fast)
    source += affine._QDOT_FUNCTION.replace("__QDOT__", affine._QDOT[bits])
    source += _ROW_DOT.replace("__CENTER__", repr(_CENTER[bits]))
    key = _row_dot_namespace(bits, fast)
    return Dot(
        key=key,
        source=f"namespace omlx_{key} {{\n{source}\n}}  // namespace omlx_{key}\n",
        shared=(
            f"namespace omlx_{key} {{\n{affine.decoded(bits)}{_ROW_DOT_SHARED}}}"
            f"  // namespace omlx_{key}\n"
        ),
        vpt=affine.values_per_thread(bits, fast),
        type=f"omlx_{key}::Dot<T>",
        tensors=("rscale",),
        dtypes=(dtype,),
        reads=_ROW_DOT_READS,
    )


# The row-scale tile: one scale for the row, the center subtracted by the
# reader. GROUP is 128 (a power of two that divides every oQh K), never a
# wide placeholder: the skeleton distributes groups over lanes.
_ROW_TILE = r"""
using namespace metal;

constant constexpr int BITS = __BITS__;
constant constexpr float CENTER = __CENTER__;

template <typename S>
struct Tile {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device S* rscale;

  template <int K>
  struct Runs {
    static constant constexpr int RUN = 8;
    static constant constexpr int GROUP = 128;
    static constant constexpr int ROW_BYTES = K / 8 * BITS;
    struct P {
      float s;
    };
__RUNS__
    // Rule 4 (docs/quant-formats.md): constructor computes pointers only;
    // rscale is read in params().
    const device S* r;

    Runs(Tile m, size_t row)
        : c((const device Stored*)(m.codes + row * ROW_BYTES)),
          r(m.rscale + row) {
      static_assert(
          K % GROUP == 0,
          "oQh tile face: K must be whole groups of 128; OqhFormat.tile gives a K its face");
    }

    P params(int) const { return {float(r[0])}; }
    Codes fetch(int run) const {
      __FETCH__
    }
    __OMLX_READ_WEIGHT__ {
      return p.s * (float(code_at(q, i)) - CENTER);
    }
  };
};
"""


@cache
def _row_tile(bits: int, dtype: mx.Dtype) -> Tile:
    if bits not in _BITS or dtype not in _METAL_TYPES:
        raise ValueError(f"oQh tile face: no {bits} bits with {dtype} scales")
    suffix = "bf16" if dtype == mx.bfloat16 else "fp16"
    namespace = f"omlx_oqh_tile_b{bits}_{suffix}"
    source = (
        _ROW_TILE.replace("__RUNS__", affine._RUNS[bits])
        .replace("__FETCH__", affine._FETCH.get(bits, "return Codes(c[run]);"))
        .replace("__BITS__", str(bits))
        .replace("__CENTER__", repr(_CENTER[bits]))
    )
    return Tile(
        key=f"oqh_tile_b{bits}_{suffix}",
        source=f"namespace {namespace} {{\n{source}\n}}  // namespace {namespace}\n",
        type=f"{namespace}::Tile<{_METAL_TYPES[dtype]}>",
        tensors=("rscale",),
        dtypes=(dtype,),
        reads=TileReads(),
    )


# The row-scale codes face, for the single and shared sg8 traversals: the
# levels, and the row's scale with the center folded into the bias.
_ROW_CODES = r"""
using namespace metal;

constant constexpr int BITS = __BITS__;
constant constexpr float CENTER = __CENTER__;

template <typename S>
struct Codes {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* weight;
  const device S* rscale;
  template <int K>
  struct Runs {
    static constant constexpr int BITS = __BITS__;
    static constant constexpr int GROUP = 128;
    // Rule 4 (docs/quant-formats.md): constructor computes pointers only;
    // rscale is read in params().
    const device uint* w;
    const device S* r;
    Runs(Codes m, size_t row)
      : w((const device uint*)m.weight + row * (K / 32 * __BITS__)), r(m.rscale + row) {}
    float2 params(int) const {
      const float s = float(r[0]);
      return float2(s, -s * CENTER);
    }
    __RUNS__
  };
};
"""


@cache
def _row_codes(bits: int, dtype: mx.Dtype) -> Codes:
    if bits != 4:
        raise ValueError(f"oQh codes face: {bits} bits has no eight-level run")
    if dtype not in _METAL_TYPES:
        raise ValueError(f"oQh codes face: no {dtype} scales")
    suffix = "bf16" if dtype == mx.bfloat16 else "fp16"
    key = f"oqh_codes_b{bits}_{suffix}"
    runs = affine._NIBBLE_RUN
    source = (
        _ROW_CODES.replace("__BITS__", str(bits))
        .replace("__CENTER__", repr(_CENTER[bits]))
        .replace("__RUNS__", runs)
    )
    return Codes(
        key,
        f"namespace omlx_{key} {{\n{source}\n}}  // namespace omlx_{key}\n",
        f"omlx_{key}::Codes<{_METAL_TYPES[dtype]}>",
        ("rscale",),
        (dtype,),
        reads=CodesReads(PairRead(pack="uint")),
    )


def _codes_shape_ok(weight, bits, n, k) -> bool:
    return weight.ndim in (2, 3) and weight.shape[-2:] == (n, k * bits // 32)


def _accept_record(fmt: QuantFormat, record: dict) -> QuantFormat:
    kept = dict(record or {})
    transform = kept.pop("transform", transforms._ABSENT)
    block = None
    if transform is not transforms._ABSENT:
        transforms.read_record(transform, kinds=("signed_wht",))
        extra = set(transform) - {"version", "kind", "block"}
        if extra:
            raise ValueError(
                f"quant format {fmt.mode!r}: unknown transform fields "
                f"{', '.join(sorted(extra))}"
            )
        if "block" in transform:
            block = transform["block"]
            if block not in _BLOCKS:
                raise ValueError(
                    f"quant format {fmt.mode!r}: transform block {block!r} is "
                    f"not one of {_BLOCKS}"
                )
    if kept:
        raise ValueError(
            f"quant format {fmt.mode!r}: unknown record fields "
            f"{', '.join(sorted(kept))}"
        )
    if block is not None:
        reader = copy.copy(fmt)
        reader.block = block
        return reader
    return fmt


def _input_transform(layer) -> transforms.InputTransform | None:
    signs = layer.get("oqh_signs")
    if not isinstance(signs, mx.array) or signs.ndim < 1 or signs.size == 0:
        return None
    block = int(signs.shape[-1])
    if block not in _BLOCKS or signs.size % block:
        return None
    return transforms.SignedWhtTransform(block=block, tensors=("oqh_signs",))


@cache
def _affine_dot(bits: int, group_size: int, fast: bool, dtype: mx.Dtype) -> Dot:
    face = affine.face(bits, group_size, fast, dtype)
    return replace(face, key=f"oqh_{face.key}", tensors=OqhAffineFormat.tensors)


@cache
def _affine_tile(bits: int, group_size: int, dtype: mx.Dtype) -> Tile:
    face = affine.tile_face(bits, group_size, dtype)
    return replace(face, key=f"oqh_{face.key}", tensors=OqhAffineFormat.tensors)


@cache
def _affine_codes(
    bits: int, group_size: int, s_dtype: mx.Dtype, b_dtype: mx.Dtype
) -> Codes:
    face = affine.codes_face(bits, group_size, s_dtype, b_dtype)
    return replace(face, key=f"oqh_{face.key}", tensors=OqhAffineFormat.tensors)


class OqhAffineFormat(QuantFormat):
    """Rotated codes with MLX affine group scales and biases: MLX's affine
    faces over the oQh tensor names, so the stock skeletons and per-chip
    dense tuning serve it unchanged."""

    mode = "oqh_affine"
    codes_name = "weight"
    tensors = ("oqh_scales", "oqh_biases")
    shared_tensors = ("oqh_signs",)
    contract = 1

    def layout(self, k: int, n: int, bits: int, group_size: int):
        """The tensors of one matrix; ``oqh_signs`` is a layer's own, one
        block vector. The block is 128 unless a checkpoint's record says
        otherwise (the stock oQh bridge builds from the stored shapes)."""
        block = getattr(self, "block", _ROW_GROUP)
        if (
            k <= 0
            or n <= 0
            or bits not in _BITS
            or group_size not in _TILE_GROUPS
            or block not in _BLOCKS
            or k % group_size
            or k % block
        ):
            return None
        return {
            "weight": ((n, k * bits // 32), mx.uint32),
            "oqh_scales": ((n, k // group_size), None),
            "oqh_biases": ((n, k // group_size), None),
            "oqh_signs": ((k // block, block), None),
        }

    def dot(self, layer, k: int, n: int) -> Dot | None:
        weight = layer.get("weight")
        scales, biases = layer.get("oqh_scales"), layer.get("oqh_biases")
        if (
            k <= 0
            or layer.bits not in _BITS
            or layer.group_size not in affine._GROUP_SIZES
            or not all(isinstance(a, mx.array) for a in (weight, scales, biases))
            or not _codes_shape_ok(weight, layer.bits, n, k)
            or weight.dtype != mx.uint32
            or scales.dtype not in _METAL_TYPES
            or biases.dtype != scales.dtype
            or k % layer.group_size
            or scales.shape != (*weight.shape[:-1], k // layer.group_size)
            or biases.shape != scales.shape
            or min(weight.size, scales.size) < 8
        ):
            return None
        fast = affine.qmv_fast_layout(k, n, layer.bits, gathered=weight.ndim == 3)
        return _affine_dot(layer.bits, layer.group_size, fast, scales.dtype)

    def tile(self, layer, k: int, n: int) -> Tile | None:
        weight = layer.get("weight")
        scales, biases = layer.get("oqh_scales"), layer.get("oqh_biases")
        if (
            k <= 0
            or layer.bits not in affine._TILE_FACE_BITS
            or layer.group_size not in _TILE_GROUPS
            or not all(isinstance(a, mx.array) for a in (weight, scales, biases))
            or not _codes_shape_ok(weight, layer.bits, n, k)
            or weight.dtype != mx.uint32
            or k % layer.group_size
            or scales.dtype not in _METAL_TYPES
            or biases.dtype != scales.dtype
            or scales.shape != (*weight.shape[:-1], k // layer.group_size)
            or biases.shape != scales.shape
            or min(weight.size, scales.size) < 8
        ):
            return None
        return _affine_tile(layer.bits, layer.group_size, scales.dtype)

    def codes(self, layer, k: int, n: int) -> Codes | None:
        bits, group = layer.bits, layer.group_size
        if k <= 0 or bits not in (2, 3, 4, 5, 6) or group not in affine._GROUP_SIZES:
            return None
        weight, scales, biases = (layer.get(name) for name in ("weight", *self.tensors))
        if not all(isinstance(a, mx.array) for a in (weight, scales, biases)):
            return None
        if (
            not _codes_shape_ok(weight, bits, n, k)
            or weight.dtype != mx.uint32
            or k % group
            or scales.shape != (*weight.shape[:-1], k // group)
            or biases.shape != scales.shape
            or scales.dtype not in affine._CODE_TYPES
            or biases.dtype not in affine._CODE_TYPES
            or min(weight.size, scales.size) < 8
        ):
            return None
        return _affine_codes(bits, group, scales.dtype, biases.dtype)

    def decode(self, tensors, bits: int, group_size: int) -> mx.array:
        return mx.dequantize(
            tensors["weight"],
            tensors["oqh_scales"],
            tensors["oqh_biases"],
            group_size=group_size,
            bits=bits,
            mode="affine",
        )

    def record(self) -> dict:
        return {
            "transform": {"version": transforms.TRANSFORM_VERSION, "kind": "signed_wht"}
        }

    def accept(self, record: dict) -> QuantFormat:
        return _accept_record(self, record)

    def input_transform(self, layer) -> transforms.InputTransform | None:
        return _input_transform(layer)


class OqhFormat(QuantFormat):
    """Rotated uniform codes with one scale per output row."""

    mode = "oqh"
    codes_name = "weight"
    tensors = ("rscale",)
    shared_tensors = ("oqh_signs",)
    contract = 1

    def layout(self, k: int, n: int, bits: int, group_size: int):
        """The tensors of one matrix; ``oqh_signs`` is a layer's own, one
        block vector (128 unless a record was accepted with another)."""
        block = getattr(self, "block", _ROW_GROUP)
        if (
            k <= 0
            or n <= 0
            or bits not in _BITS
            or block not in _BLOCKS
            or k % block
            or k % _ROW_GROUP
        ):
            return None
        return {
            "weight": ((n, k * bits // 32), mx.uint32),
            "rscale": ((n,), None),
            "oqh_signs": ((k // block, block), None),
        }

    def dot(self, layer, k: int, n: int) -> Dot | None:
        weight, rscale = layer.get("weight"), layer.get("rscale")
        if (
            k <= 0
            or layer.bits not in _BITS
            or not isinstance(weight, mx.array)
            or not isinstance(rscale, mx.array)
            or not _codes_shape_ok(weight, layer.bits, n, k)
            or weight.dtype != mx.uint32
            or rscale.dtype not in _METAL_TYPES
            or rscale.shape != weight.shape[:-1]
            or k % 32
            or min(weight.size, rscale.size) < 8
        ):
            return None
        # qmv_fast when K is whole kernel blocks: 512 at 2 to 5 bits, 256 at
        # 6 and 8 bits.
        pack_factor = (
            8 if layer.bits in (3, 5) else (4 if layer.bits == 6 else 32 // layer.bits)
        )
        block = pack_factor * (1 if layer.bits == 2 else 2) * 32
        fast = k % block == 0
        return _row_dot(layer.bits, fast, rscale.dtype)

    def tile(self, layer, k: int, n: int) -> Tile | None:
        weight, rscale = layer.get("weight"), layer.get("rscale")
        if (
            k <= 0
            or layer.bits not in _BITS
            or not all(isinstance(a, mx.array) for a in (weight, rscale))
            or not _codes_shape_ok(weight, layer.bits, n, k)
            or weight.dtype != mx.uint32
            or rscale.dtype not in _METAL_TYPES
            or rscale.shape != weight.shape[:-1]
            or k % _ROW_GROUP
            or min(weight.size, rscale.size) < 8
        ):
            return None
        return _row_tile(layer.bits, rscale.dtype)

    def codes(self, layer, k: int, n: int) -> Codes | None:
        bits = layer.bits
        # The codes contract's levels are 7-bit (sg8); MLX's 8-bit levels are
        # 8-bit, so only 4-bit row-scale codes have a codes face.
        if k <= 0 or bits != 4:
            return None
        weight, rscale = layer.get("weight"), layer.get("rscale")
        if (
            not all(isinstance(a, mx.array) for a in (weight, rscale))
            or not _codes_shape_ok(weight, bits, n, k)
            or weight.dtype != mx.uint32
            or rscale.dtype not in (mx.bfloat16, mx.float16)
            or rscale.shape != weight.shape[:-1]
            or k % _ROW_GROUP
            or min(weight.size, rscale.size) < 8
        ):
            return None
        return _row_codes(bits, rscale.dtype)

    def decode(self, tensors, bits: int, group_size: int) -> mx.array:
        weight, rscale = tensors["weight"], tensors["rscale"]
        k = weight.shape[-1] * 32 // bits
        gs = next(g for g in (128, 64, 32) if k % g == 0)
        center = _CENTER[bits]
        shape = (*weight.shape[:-1], k // gs)
        ones = mx.ones(shape, rscale.dtype)
        zeros = mx.zeros(shape, rscale.dtype)
        codes = mx.dequantize(
            weight, ones, zeros, group_size=gs, bits=bits, mode="affine"
        )
        rotated = (codes.astype(mx.float32) - center) * rscale.astype(mx.float32)[
            ..., None
        ]
        return rotated.astype(rscale.dtype)

    def record(self) -> dict:
        return {
            "transform": {"version": transforms.TRANSFORM_VERSION, "kind": "signed_wht"}
        }

    def accept(self, record: dict) -> QuantFormat:
        return _accept_record(self, record)

    def input_transform(self, layer) -> transforms.InputTransform | None:
        return _input_transform(layer)


OQH_AFFINE = OqhAffineFormat()
OQH = OqhFormat()

FORMAT = OQH_AFFINE
