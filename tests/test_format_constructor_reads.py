# SPDX-License-Identifier: Apache-2.0
"""Rule-4 constructor-read checks (``omlx.formats.testing.check_constructor_reads``).

A face's cursor constructor (``Runs(m, row)`` or ``Lanes(m, row, lane)``)
computes pointers from ``row`` and ``K`` and reads no operand data: kernels
build cursors for rows past the matrix and call ``params``/``fetch``/``row``/
``madd``/``finish`` only for rows they compute.
"""

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

from omlx import formats
from omlx.formats import Codes, Table, Tile, testing
from tests.test_formats_codes import _BYTE_CODES, _BYTE_READS
from tests.test_formats_kernels import _TOY as _DOT_TOY
from tests.test_formats_kernels import _matrix as _dot_matrix
from tests.test_formats_kernels import _toy as _dot_toy
from tests.test_formats_tile import (
    _TILE_READS,
    _TOY,
    _layer,
    _toy,
    _toy_matrix,
)

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_PAGE = (Path(__file__).parents[1] / "docs" / "quant-formats.md").read_text()

# Toy tile faces that violate Rule 4 by reading operand data in Runs(Tile, row).
_TOY_READS_ROW_SCALE = (
    _TOY.replace(
        "    const device float* r;\n",
        "    float rs;\n",
    )
    .replace(
        "          r(m.row_scales + row) {}",
        "          rs(m.row_scales[row]) {}",
    )
    .replace(
        "      return {g[group] * r[0]};",
        "      return {g[group] * rs};",
    )
)

_TOY_READS_GROUP_SCALE = (
    _TOY.replace(
        "    const device float* g;\n",
        "    const device float* g;\n    float g0;\n",
    )
    .replace(
        "          g(m.group_scales + row * GROUPS),\n",
        "          g(m.group_scales + row * GROUPS),\n"
        "          g0(m.group_scales[row * GROUPS]),\n",
    )
    .replace(
        "      return {g[group] * r[0]};",
        "      return {(group == 0 ? g0 : g[group]) * r[0]};",
    )
)

_TOY_READS_CODES = (
    _TOY.replace(
        "    const device packed_uint2* c;\n",
        "    const device packed_uint2* c;\n    uint2 c0;\n",
    )
    .replace(
        "        : c((const device packed_uint2*)(m.codes + row * K)),\n",
        "        : c((const device packed_uint2*)(m.codes + row * K)),\n"
        "          c0(uint2(((const device packed_uint2*)(m.codes + row * K))[0])),\n",
    )
    .replace(
        "      return Codes(c[run]);",
        "      return run == 0 ? c0 : Codes(c[run]);",
    )
)

_TOY_READS_BOTH_SCALES = (
    _TOY_READS_ROW_SCALE.replace(
        "    const device float* g;\n",
        "    const device float* g;\n    float g0;\n",
    )
    .replace(
        "          g(m.group_scales + row * GROUPS),\n",
        "          g(m.group_scales + row * GROUPS),\n"
        "          g0(m.group_scales[row * GROUPS]),\n",
    )
    .replace(
        "      return {g[group] * rs};",
        "      return {(group == 0 ? g0 : g[group]) * rs};",
    )
)

# A tile face with zero side tensors (only packed codes).
_NO_TENSORS_TILE = r"""namespace omlx_notensors_KEY {
struct Tile {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;

  template <int K>
  struct Runs {
    static constant constexpr int RUN = 8;
    static constant constexpr int GROUP = 32;
    struct P { float unused; };
    using Codes = uint2;
    const device packed_uint2* c;

    Runs(Tile m, size_t row)
        : c((const device packed_uint2*)(m.codes + row * K)) {}

    P params(int) const { return {1.0f}; }
    Codes fetch(int run) const { return Codes(c[run]); }
    __OMLX_READ_WEIGHT__ {
      const uint code = ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xFFu;
      return (float(code) - 128.0f) * p.unused;
    }
  };
};
}
"""

# A tile face with a staged lookup table.
_TABLED_TILE = r"""namespace omlx_tabled_KEY {
struct Tile {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* row_scales;
  omlx_table<float, 4> levels;

  template <int K>
  struct Runs {
    static constant constexpr int RUN = 8;
    static constant constexpr int GROUP = 32;
    struct P { float s; omlx_table<float, 4> lv; };
    using Codes = uint2;
    const device packed_uint2* c;
    const device float* r;
    omlx_table<float, 4> lv;

    Runs(Tile m, size_t row)
        : c((const device packed_uint2*)(m.codes + row * K)),
          r(m.row_scales + row),
          lv(m.levels) {}

    P params(int) const { return {r[0], lv}; }
    Codes fetch(int run) const { return Codes(c[run]); }
    __OMLX_READ_WEIGHT__ {
      const uint code = ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0x3u;
      return p.lv[code] * p.s;
    }
  };
};
}
"""

# A Dot face that reads its row scale in Lanes(Dot, row, lane).
_DOT_READS_SCALES = (
    _DOT_TOY.replace(
        "    const device float* s;\n",
        "    float s0;\n",
    )
    .replace(
        "s(m.scales + row) {}",
        "s0(m.scales[row]) {}",
    )
    .replace(
        "    float finish(int r, float sum) const { return sum * s[r]; }",
        "    float finish(int, float sum) const { return sum * s0; }",
    )
)

# A Codes face that reads its first map in Runs(ByteCodes, row).
_CODES_READS_MAPS = (
    _BYTE_CODES.replace(
        "    const device float* maps;\n",
        "    const device float* maps;\n    float2 m0;\n",
    )
    .replace(
        "      : values(m.values + row * K), maps(m.maps + row * (K / GROUP) * 2) {}",
        "      : values(m.values + row * K),\n"
        "        maps(m.maps + row * (K / GROUP) * 2),\n"
        "        m0(m.maps[row * (K / GROUP) * 2], m.maps[row * (K / GROUP) * 2 + 1]) {}",
    )
    .replace(
        "    float2 params(int g) const { return float2(maps[2*g], maps[2*g+1]); }",
        "    float2 params(int g) const {\n"
        "      return g == 0 ? m0 : float2(maps[2*g], maps[2*g+1]);\n"
        "    }",
    )
)


def _no_tensors_tile(key: str, *, reads_codes: bool = False) -> Tile:
    source = _NO_TENSORS_TILE.replace("KEY", key)
    if reads_codes:
        source = (
            source.replace(
                "    const device packed_uint2* c;\n",
                "    const device packed_uint2* c;\n    uint2 c0;\n",
            )
            .replace(
                "        : c((const device packed_uint2*)(m.codes + row * K)) {}",
                "        : c((const device packed_uint2*)(m.codes + row * K)),\n"
                "          c0(uint2(((const device packed_uint2*)"
                "(m.codes + row * K))[0])) {}",
            )
            .replace(
                "    Codes fetch(int run) const { return Codes(c[run]); }",
                "    Codes fetch(int run) const { return run == 0 ? c0 : Codes(c[run]); }",
            )
        )
    return Tile(
        key=f"notensors_{key}",
        source=source,
        type=f"omlx_notensors_{key}::Tile",
        tensors=(),
        reads=_TILE_READS,
    )


def _tabled_tile(key: str, *, reads_row_scale: bool = False) -> Tile:
    source = _TABLED_TILE.replace("KEY", key)
    if reads_row_scale:
        source = (
            source.replace("    const device float* r;\n", "    float rs;\n")
            .replace(
                "          r(m.row_scales + row),\n",
                "          rs(m.row_scales[row]),\n",
            )
            .replace(
                "    P params(int) const { return {r[0], lv}; }",
                "    P params(int) const { return {rs, lv}; }",
            )
        )
    return Tile(
        key=f"tabled_{key}",
        source=source,
        type=f"omlx_tabled_{key}::Tile",
        tensors=("row_scales",),
        tables=(Table("levels", mx.float32, (-1.5, -0.5, 0.5, 1.5)),),
        reads=_TILE_READS,
    )


@metal
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
def test_compliant_toy_tile_passes_with_no_false_positive_on_params_or_fetch(dtype):
    operands = _toy_matrix(8, 64, seed=1)
    testing.check_constructor_reads(_toy("ctor_ok"), operands, 64, [0, 3, 7], dtype)
    # Const-lvalue-only constructor also passes.
    const_only = _TOY.replace(
        "    Runs(Tile m, size_t row)",
        "    Runs(thread Tile&, size_t) = delete;\n"
        "    Runs(thread const Tile& m, size_t row)",
    )
    testing.check_constructor_reads(
        _toy("ctor_const_ok", const_only), operands, 64, [0, 7], dtype
    )


@metal
def test_toy_tile_reading_row_scales_in_constructor_fails_with_exact_message():
    operands = _toy_matrix(8, 64, seed=2)
    face = _toy("ctor_row", _TOY_READS_ROW_SCALE)
    with pytest.raises(AssertionError) as raised:
        testing.check_constructor_reads(face, operands, 64, [0, 5], mx.float32)
    assert str(raised.value) == (
        "face toy_ctor_row: constructor read row_scales "
        "(rule 4: the constructor computes pointers only and reads no data)"
    )


@metal
def test_toy_tile_reading_group_scales_in_constructor_fails_with_exact_message():
    operands = _toy_matrix(8, 64, seed=3)
    face = _toy("ctor_group", _TOY_READS_GROUP_SCALE)
    with pytest.raises(AssertionError) as raised:
        testing.check_constructor_reads(face, operands, 64, [0, 4], mx.float32)
    assert str(raised.value) == (
        "face toy_ctor_group: constructor read group_scales "
        "(rule 4: the constructor computes pointers only and reads no data)"
    )


@metal
@pytest.mark.parametrize("codes_name", ["weight", "codes"])
def test_toy_tile_reading_codes_in_constructor_names_codes_member(codes_name):
    operands = _toy_matrix(8, 64, seed=4)
    face = _toy(f"ctor_codes_{codes_name}", _TOY_READS_CODES)
    with pytest.raises(AssertionError) as raised:
        testing.check_constructor_reads(
            face, operands, 64, [0, 2], mx.float32, codes_name=codes_name
        )
    assert str(raised.value) == (
        f"face toy_ctor_codes_{codes_name}: constructor read {codes_name} "
        "(rule 4: the constructor computes pointers only and reads no data)"
    )


@metal
def test_toy_tile_reading_multiple_members_in_constructor_names_all_in_order():
    operands = _toy_matrix(8, 64, seed=5)
    face = _toy("ctor_both", _TOY_READS_BOTH_SCALES)
    with pytest.raises(AssertionError) as raised:
        testing.check_constructor_reads(face, operands, 64, [0, 7], mx.float32)
    assert str(raised.value) == (
        "face toy_ctor_both: constructor read group_scales, row_scales "
        "(rule 4: the constructor computes pointers only and reads no data)"
    )


@metal
def test_tile_with_zero_side_tensors_and_with_staged_tables():
    mx.random.seed(6)
    codes = mx.random.randint(0, 256, (8, 64)).astype(mx.uint8)
    row_scales = mx.random.normal((8,))
    testing.check_constructor_reads(
        _no_tensors_tile("ok"), (codes,), 64, [0, 7], mx.float32
    )
    with pytest.raises(AssertionError) as raised:
        testing.check_constructor_reads(
            _no_tensors_tile("bad", reads_codes=True), (codes,), 64, [0], mx.float32
        )
    assert str(raised.value) == (
        "face notensors_bad: constructor read weight "
        "(rule 4: the constructor computes pointers only and reads no data)"
    )
    testing.check_constructor_reads(
        _tabled_tile("ok"), (codes, row_scales), 64, [0, 3], mx.float32
    )
    with pytest.raises(AssertionError) as raised:
        testing.check_constructor_reads(
            _tabled_tile("bad", reads_row_scale=True),
            (codes, row_scales),
            64,
            [0, 3],
            mx.float32,
            codes_name="codes",
        )
    assert str(raised.value) == (
        "face tabled_bad: constructor read row_scales "
        "(rule 4: the constructor computes pointers only and reads no data)"
    )


@metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_builtin_affine_tile_faces_pass_constructor_reads(bits, group_size, dtype):
    layer = _layer(256, 16, bits, group_size, dtype, experts=2, seed=bits + group_size)
    face = formats.tile(layer, 256, 16)
    assert face is not None
    operands = [layer[name] for name in ("weight", *face.tensors)]
    testing.check_constructor_reads(face, operands, 256, [0, 15, 31], dtype)


@metal
def test_dot_face_constructor_reads_pass_and_fail_with_exact_message():
    operands, _, _ = _dot_matrix(8, 320)
    testing.check_constructor_reads(
        _dot_toy("ctor_dot_ok"), operands, 320, [0, 7], mx.float32, codes_name="codes"
    )
    bad = _dot_toy("ctor_dot_bad", _DOT_READS_SCALES)
    with pytest.raises(AssertionError) as raised:
        testing.check_constructor_reads(
            bad, operands, 320, [0, 7], mx.float32, codes_name="codes"
        )
    assert str(raised.value) == (
        "face toy_ctor_dot_bad: constructor read group_scales "
        "(rule 4: the constructor computes pointers only and reads no data)"
    )
    # Built-in affine dot faces (qmv_fast and qmv).
    for k, dtype in ((512, mx.bfloat16), (320, mx.float16)):
        layer = _layer(k, 16, 4, 64, dtype, seed=k)
        face = formats.dot(layer, k, 16)
        assert face is not None
        testing.check_constructor_reads(
            face, [layer.weight, layer.scales, layer.biases], k, [0, 15], dtype
        )


@metal
def test_codes_face_constructor_reads_pass_and_fail_with_exact_message():
    k, n = 128, 8
    mx.random.seed(9)
    storage = mx.random.randint(0, 128, (n, k)).astype(mx.uint8)
    maps = mx.random.normal((n, k // 64, 2))
    ok = Codes(
        "byte_codes_ctor_ok",
        _BYTE_CODES,
        "ByteCodes",
        ("maps",),
        (mx.float16,),
        reads=_BYTE_READS,
    )
    testing.check_constructor_reads(ok, (storage, maps), k, [0, 7], mx.float16)
    bad = Codes(
        "byte_codes_ctor_bad",
        _CODES_READS_MAPS,
        "ByteCodes",
        ("maps",),
        (mx.float16,),
        reads=_BYTE_READS,
    )
    with pytest.raises(AssertionError) as raised:
        testing.check_constructor_reads(bad, (storage, maps), k, [0, 7], mx.float16)
    assert str(raised.value) == (
        "face byte_codes_ctor_bad: constructor read maps "
        "(rule 4: the constructor computes pointers only and reads no data)"
    )
    # Built-in affine codes faces (4-bit nibble and 3-bit packed).
    for bits in (4, 3):
        layer = _layer(256, 16, bits, 64, mx.bfloat16, seed=bits)
        face = formats.codes(layer, 256, 16)
        assert face is not None
        testing.check_constructor_reads(
            face, [layer.weight, layer.scales, layer.biases], 256, [0, 15], mx.bfloat16
        )


@metal
def test_degenerate_all_zero_operands_fail_closed():
    # With group_scales == 0 and row_scales == 0, mutating codes produces 0 * 0
    # regardless of codes, so the probe refuses instead of failing open.
    zeros = (
        mx.zeros((8, 64), mx.uint8),
        mx.zeros((8, 2), mx.float32),
        mx.zeros((8,), mx.float32),
    )
    with pytest.raises(ValueError, match="had no observable effect on the probed rows"):
        testing.check_constructor_reads(_toy("ctor_zeros"), zeros, 64, [0], mx.float32)


def test_check_constructor_reads_refuses_invalid_arguments():
    operands = _toy_matrix(8, 64, seed=10)
    face = _toy("ctor_args")
    with pytest.raises(ValueError, match="is not a Dot, Tile, or Codes face"):
        testing.check_constructor_reads(object(), operands, 64, [0], mx.float32)
    for bad_k in (0, 12, -8, True, 1 << 31):
        with pytest.raises(ValueError, match="is not one run or more of 8"):
            testing.check_constructor_reads(face, operands, bad_k, [0], mx.float32)
    with pytest.raises(ValueError, match="are not one row index or more"):
        testing.check_constructor_reads(face, operands, 64, [], mx.float32)
    with pytest.raises(ValueError, match="exact for mlx.core.float16"):
        testing.check_constructor_reads(
            replace(face, dtypes=(mx.float16,)), operands, 64, [0], mx.bfloat16
        )
    with pytest.raises(ValueError, match="packed_bytes, group_scales, row_scales"):
        testing.check_constructor_reads(
            face, operands[:2], 64, [0], mx.float32, codes_name="packed_bytes"
        )
    # Dot refuses bool/<=0/2^31 and k that is not one run or more of 8.
    dot = _dot_toy("ctor_dot_args")
    dot_ops, _, _ = _dot_matrix(8, 64)
    for bad_k in (0, -1, True, False, 1 << 31):
        with pytest.raises(ValueError, match="is not a positive int below 2\\^31"):
            testing.check_constructor_reads(dot, dot_ops, bad_k, [0], mx.float32)
    for bad_k in (1, 4, 12):
        with pytest.raises(ValueError, match="is not one run or more of 8 inputs"):
            testing.check_constructor_reads(dot, dot_ops, bad_k, [0], mx.float32)


def test_dot_rejects_non_multiple_of_8_k():
    """A Dot k the Metal static assert would refuse (k % Lanes<K>::VPT) is a
    ValueError before the probe compiles: k = 1 is not one run of 8."""
    dot = _dot_toy("ctor_dot_k1")
    dot_ops, _, _ = _dot_matrix(8, 64)
    with pytest.raises(ValueError, match="k=1 is not one run or more of 8 inputs"):
        testing.check_constructor_reads(dot, dot_ops, 1, [0], mx.float32)


def test_docs_scope_rule4_to_constructor():
    page = _PAGE
    assert (
        "The constructor part of rule 4 is what "
        "`omlx.formats.testing.check_constructor_reads` checks"
    ) in page
    assert (
        "Rule 4 is what `omlx.formats.testing.check_constructor_reads` checks"
    ) not in page


def test_metal_mark_on_thread_safe_not_dot_k_validation():
    assert metal.mark in getattr(
        test_check_constructor_reads_is_thread_safe, "pytestmark", []
    )
    assert metal.mark not in getattr(
        test_dot_rejects_non_multiple_of_8_k, "pytestmark", []
    )
    assert metal.mark not in getattr(
        test_docs_scope_rule4_to_constructor, "pytestmark", []
    )


@metal
def test_check_constructor_reads_is_thread_safe():
    operands = _toy_matrix(16, 128, seed=11)
    mx.eval(*operands)
    face = _toy("ctor_threads")

    def run(_):
        testing.check_constructor_reads(face, operands, 128, [0, 7, 15], mx.bfloat16)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(4)))


def test_docs_name_check_constructor_reads_for_rule_4():
    assert "`omlx.formats.testing.check_constructor_reads`" in _PAGE
    assert "Nothing checks rules 3, 4, 5 and 6" not in _PAGE
    assert "Nothing checks rules 3, 5 and 6" in _PAGE
    versions = _PAGE.split("\n## Versions\n")[1].split("\n## ")[0]
    assert "`check_constructor_reads`" in versions


def test_ctor_probe_headers_avoid_signed_w_and_alt_bits_overflow():
    assert "for (int w =" not in testing._CTOR_TILE_HEADER
    assert "for (int w =" not in testing._CTOR_CODES_HEADER
    assert "alt_bits[2 * K" not in testing._CTOR_CODES_HEADER


def test_alternate_operand_edge_values():
    edges = [
        0.0,
        -0.0,
        0.25,
        1.0,
        -1.0,
        64.0,
        32768.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ]
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        array = mx.array(edges, dtype=dtype)
        out = testing._alternate_operand(array)
        assert out.dtype == array.dtype
        assert bool(mx.all(mx.isfinite(out)))
        assert bool(mx.all(out != 0))
        assert bool(mx.all(out != array))
    ints = mx.array([0, 1, 2, 255], dtype=mx.uint8)
    assert bool(mx.all(testing._alternate_operand(ints) == mx.bitwise_invert(ints)))
    signed = mx.array([-1, 0, 1, 1 << 30], dtype=mx.int32)
    assert bool(mx.all(testing._alternate_operand(signed) == mx.bitwise_invert(signed)))
    flags = mx.array([True, False])
    assert bool(mx.all(testing._alternate_operand(flags) == mx.logical_not(flags)))
    with pytest.raises(ValueError, match="unsupported operand dtype"):
        testing._alternate_operand(mx.array([1 + 1j], dtype=mx.complex64))
    assert "neg_two" not in inspect.getsource(testing._alternate_operand)
