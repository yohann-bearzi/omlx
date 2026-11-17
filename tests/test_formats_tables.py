# SPDX-License-Identifier: Apache-2.0
"""Tables a face declares as data: what a ``Table`` holds, what a face may
declare, and what a kernel's source gains (the staging) or does not (a face
without tables). The kit's ``staged_tables`` and the staging on the GPU."""

from __future__ import annotations

import hashlib
import struct
from itertools import product
from dataclasses import FrozenInstanceError, replace

import mlx.core as mx
import numpy as np
import pytest

from omlx import formats
from omlx.formats import Dot, Table, affine, kernels, testing
from omlx.formats import linear as format_linear
from omlx.patches import m5_gather_qmm_nax as nax
from omlx.patches import moe_verify_gather as gather
from omlx.patches import qwen35_moe_routed_decode as routed

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


# --- Table -----------------------------------------------------------------


def test_a_float32_table_holds_its_values_rounded_to_float32():
    table = Table("levels", mx.float32, (0.1, -2, np.float32(0.5), 3))
    assert table.values == (_f32(0.1), -2.0, 0.5, 3.0)
    assert table.bits == tuple(
        struct.unpack("<I", struct.pack("<f", v))[0] for v in table.values
    )


def test_half_and_bfloat16_tables_round_to_nearest_even():
    half = Table("t", mx.float16, (1 + 2**-11, 1 + 3 * 2**-11, 65504.0))
    assert half.values == (1.0, 1 + 2**-9, 65504.0)
    assert half.bits == (0x3C00, 0x3C02, 0x7BFF)
    bf16 = Table(
        "t", mx.bfloat16, (1 + 2**-8, 1 + 3 * 2**-8, -0.0, 2.0**-133, 3 * 2.0**-134)
    )
    assert bf16.values == (1.0, 1 + 2**-6, -0.0, 2.0**-133, 2.0**-132)
    assert bf16.bits == (0x3F80, 0x3F82, 0x8000, 0x0001, 0x0002)


def test_integer_tables_take_whole_numbers_in_range():
    assert Table("t", mx.int8, (-128, 127, 2.0)).bits == (0x80, 0x7F, 0x02)
    assert Table("t", mx.uint16, (0, 65535)).bits == (0, 0xFFFF)
    assert Table("t", mx.int32, (np.int64(-1),)).bits == (0xFFFFFFFF,)
    for dtype, value in (
        (mx.int8, 128),
        (mx.uint8, -1),
        (mx.uint32, 2**32),
        (mx.int16, 0.5),
    ):
        with pytest.raises(ValueError, match=f"is not a value of {dtype}"):
            Table("t", dtype, (value,))


@pytest.mark.parametrize(
    "dtype,value",
    [
        (mx.float32, float("nan")),
        (mx.float32, 1e39),
        (mx.float32, float("inf")),
        (mx.float16, 7e4),
        (mx.bfloat16, 3.4e38 * 2),
        (mx.bfloat16, float("-inf")),
        (mx.float16, float("inf")),
        (mx.int8, float("nan")),
        (mx.int8, float("inf")),
    ],
)
def test_a_value_its_type_cannot_hold_is_refused_as_a_value_error(dtype, value):
    with pytest.raises(ValueError, match="'t': .* is not a value of"):
        Table("t", dtype, (value,))


def test_a_table_is_compared_by_its_bits():
    assert Table("t", mx.float32, (0.1,)) == Table("t", mx.float32, (_f32(0.1),))
    assert hash(Table("t", mx.float32, (1,))) == hash(Table("t", mx.float32, (1.0,)))
    assert Table("t", mx.float32, (0.0,)) != Table("t", mx.float32, (-0.0,))
    assert Table("t", mx.float32, (1.0,)) != Table("t", mx.float16, (1.0,))
    assert Table("t", mx.float32, (1.0,)) != Table("u", mx.float32, (1.0,))
    with pytest.raises(FrozenInstanceError):
        Table("t", mx.float32, (1.0,)).values = (2.0,)


@pytest.mark.parametrize(
    "name,dtype,values,said",
    [
        ("2x", mx.float32, (1.0,), "not an identifier"),
        ("t", mx.float64, (1.0,), "a table holds one of"),
        ("t", mx.bool_, (1,), "a table holds one of"),
        ("t", mx.float32, [1.0], "values are a tuple of 1 to 1024"),
        ("t", mx.float32, (), "values are a tuple of 1 to 1024"),
        ("t", mx.float32, (0.0,) * 1025, "values are a tuple of 1 to 1024"),
        ("t", mx.float32, (True,), "True is not a number"),
        ("t", mx.float32, ("1",), "'1' is not a number"),
    ],
)
def test_a_malformed_table_is_refused(name, dtype, values, said):
    with pytest.raises(ValueError, match=said):
        Table(name, dtype, values)


# --- faces with tables ---------------------------------------------------------


def _face(*tables, tensors=("scales",)):
    return Dot(key="t", source="", type="T", tensors=tensors, tables=tables)


def test_tables_are_a_keyword_of_a_face_and_none_by_default():
    assert Dot("k", "", "T", ()).tables == ()
    with pytest.raises(TypeError):
        Dot("k", "", "T", (), (), (Table("t", mx.float32, (1.0,)),))


@pytest.mark.parametrize(
    "tables,tensors,said",
    [
        ([Table("t", mx.float32, (1.0,))], ("scales",), "tables are a tuple of Table"),
        (("t",), ("scales",), "tables are a tuple of Table"),
        (
            (Table("t", mx.float32, (1.0,)), Table("t", mx.int8, (1,))),
            ("scales",),
            "each named once",
        ),
        ((Table("scales", mx.float32, (1.0,)),), ("scales",), "not as the weight"),
        ((Table("weight", mx.float32, (1.0,)),), ("scales",), "not as the weight"),
        (
            (Table("a", mx.float32, (1.0,) * 1024), Table("b", mx.int8, (1,))),
            ("scales",),
            "4097 bytes",
        ),
    ],
)
def test_a_face_s_tables_are_checked(tables, tensors, said):
    with pytest.raises(ValueError, match=said):
        Dot(key="t", source="", type="T", tensors=tensors, tables=tables)


def test_a_face_may_hold_4096_bytes_of_tables():
    face = _face(Table("a", mx.float32, (1.0,) * 1023), Table("b", mx.int8, (1,) * 4))
    assert len(face.tables) == 2


def test_a_face_s_equality_holds_its_tables():
    a, b = Table("t", mx.float32, (1.0, 2.0)), Table("t", mx.float32, (1.0, 3.0))
    assert _face(a) == _face(Table("t", mx.float32, (1, 2)))
    assert hash(_face(a)) == hash(_face(Table("t", mx.float32, (1, 2))))
    assert _face(a) != _face(b) and _face(a) != _face()
    assert kernels.proof("k", _face(a)) is kernels.proof("k", replace(_face(a)))
    assert kernels.proof("k", _face(a)) is not kernels.proof("k", _face(b))


# --- the source a kernel gains -------------------------------------------------


def _staged(text):
    return [
        s
        for s in ("omlx_staged", "threadgroup_barrier", "threads_per_threadgroup")
        if s in text
    ]


def test_a_face_without_tables_adds_nothing_to_a_kernel():
    face = affine.face(4, 64, True, mx.bfloat16)
    matrices = kernels.Matrices()
    matrices.add("m", face)
    assert not _staged(matrices.header) and not _staged(matrices.declarations)
    assert matrices.declarations == (
        f"    const auto m = {face.type}{{(const device uint8_t*)m_weight, m_scales, "
        'm_biases};\n\n#line 1 "<kernel>"\n'
    )


def _faces():
    """Every affine dot face."""
    return [
        affine.face(bits, group, fast, dtype)
        for bits in (2, 3, 4, 5, 6, 8)
        for group in (32, 64, 128)
        for fast in (True, False)
        for dtype in (mx.bfloat16, mx.float16)
    ]


def _builder_digest(monkeypatch) -> str:
    """The digest of every Metal that the kernels over faces build for the
    affine faces: each kernel's name, header and body, in one order."""
    recorded = []

    def record(**kwargs):
        recorded.append((kwargs["name"], kwargs.get("header", ""), kwargs["source"]))

    monkeypatch.setattr(mx.fast, "metal_kernel", record)
    monkeypatch.setattr(nax, "_kernels", {})
    faces = _faces()
    for face in faces:
        format_linear._kernel.__wrapped__(face, False)
        format_linear._kernel.__wrapped__(face, True)
        gather._kernel.__wrapped__(face)
        gather._ordered_kernel.__wrapped__(face)
        testing._dot_rows_kernel.__wrapped__(face)
    for bits, group, dtype in product((4, 8), (32, 64, 128), (mx.bfloat16, mx.float16)):
        tile = affine.tile_face(bits, group, dtype)
        testing._tile_rows_kernel.__wrapped__(tile)
        nax._face_kernel(tile)
    shared, gate = affine.face(8, 128, True, mx.bfloat16), affine.face(
        8, 64, True, mx.bfloat16
    )
    for face in faces[:8]:
        for builder in (routed._gate_up_kernel, routed._gate_up_topk_kernel):
            builder.__wrapped__(face, shared, gate)
            builder.__wrapped__(face, None, None)
        routed._down_kernel.__wrapped__(face, shared)
        routed._down_kernel.__wrapped__(face, None)
        routed._gate_up_window_kernel.__wrapped__(face, shared, gate)
        routed._down_window_kernel.__wrapped__(face, shared)
    digest = hashlib.sha256()
    for record_ in recorded:
        for part in record_:
            digest.update(part.encode() + b"\0")
    return f"{len(recorded)} {digest.hexdigest()}"


# The digest of the parent commit's kernels (699a215): what a face without
# tables compiles is the same text, to the byte, so the same MLX library.
_PARENT_DIGEST = "448 c5dcac9158d8f5cee71653d45cb31820f815db085d228decb50e9b6c59797dd4"


def test_kernels_without_tables_are_the_same_text_as_before(monkeypatch):
    assert _builder_digest(monkeypatch) == _PARENT_DIGEST


_TOY = r"""
namespace omlx_tables_KEY {
struct Dot {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* scales;
MEMBERS
};
}
"""


def _toy(key, members, *tables):
    return Dot(
        key=f"tables_{key}",
        source=_TOY.replace("KEY", key).replace("MEMBERS", members),
        type=f"omlx_tables_{key}::Dot",
        tensors=("scales",),
        tables=tables,
    )


def test_a_kernel_stages_each_table_once_by_element_type():
    a = Table("a", mx.float32, (1.0, -0.0))
    b = Table("b", mx.int8, (-1, 2, 3))
    c = Table("c", mx.float32, (0.5,))
    one = _toy("one", "", a, b)
    two = _toy("two", "", c, Table("a", mx.float32, (1, -0.0)))
    matrices = kernels.Matrices()
    matrices.add("p", one)
    matrices.add("q", two)
    header = matrices.header
    assert header.count('#line 1 "<omlx staged tables>"') == 1
    assert (
        "constant uint32_t bits_float[3] = {0x3f800000u, 0x80000000u, 0x3f000000u};"
        in header
    )
    assert "constant uint8_t bits_int8_t[3] = {0xffu, 0x2u, 0x3u};" in header
    body = matrices.declarations
    assert body.startswith(
        "    threadgroup float omlx_tg_float[3];\n"
        "    if (thread_index_in_threadgroup < 3u) {\n"
    )
    assert body.count("threadgroup_barrier") == 1
    assert "threadgroup int8_t omlx_tg_int8_t[3];" in body
    assert "as_type<int8_t>(omlx_staged::bits_int8_t[i])" in body
    # Equal tables of two faces share one copy.
    assert "m_scales" not in body
    assert "p_scales, omlx_tg_float + 0, omlx_tg_int8_t + 0};" in body
    assert "q_scales, omlx_tg_float + 2, omlx_tg_float + 0};" in body
    assert "decltype(omlx_tables_one::Dot::b), const threadgroup int8_t*>" in body
    assert "table 'b' is int8 data" in body


def test_a_kernel_stages_at_most_4096_bytes_of_tables():
    matrices = kernels.Matrices()
    matrices.add("p", _toy("p", "", Table("a", mx.float32, (0.0,) * 1000)))
    matrices.add("q", _toy("q", "", Table("a", mx.float32, (0.0,) * 1000)))  # the same
    with pytest.raises(
        ValueError, match="'r': the kernel's tables would hold 4100 bytes"
    ):
        matrices.add("r", _toy("r", "", Table("b", mx.float32, (1.0,) * 25)))
    matrices.add("s", _toy("s", "", Table("b", mx.float32, (1.0,) * 24)))
    body = matrices.declarations
    assert (
        "omlx_tg_float[1024];\n    for (uint i = thread_index_in_threadgroup; i < 1024u;"
        in body
    )


@pytest.mark.parametrize("count,loop", [(32, False), (33, True)])
def test_up_to_32_values_of_a_type_are_one_copy_a_thread(count, loop):
    matrices = kernels.Matrices()
    matrices.add("p", _toy("p", "", Table("a", mx.int16, (0,) * count)))
    body = matrices.declarations
    assert ("for (uint i = thread_index_in_threadgroup" in body) == loop
    assert (f"if (thread_index_in_threadgroup < {count}u)" in body) == (not loop)


# --- on the GPU -----------------------------------------------------------------


_MEMBERS = """  const threadgroup float* a;
  const threadgroup int8_t* b;
  const threadgroup half* c;
  const threadgroup bfloat16_t* d;
  const threadgroup float* e;"""

_TABLES = (
    Table("a", mx.float32, (1.5, -0.0, 3.0e38, 2.0**-149)),
    Table("b", mx.int8, (-128, 0, 127)),
    Table("c", mx.float16, (0.1, -65504.0)),
    Table("d", mx.bfloat16, (0.1, -3.0)),
    Table("e", mx.float32, (7.0,) * 40),
)


def _operands():
    return mx.zeros((8, 8), mx.uint8), mx.ones((8,), mx.float32)


def _bits(array):
    width = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[array.dtype.size]
    return tuple(np.array(array.view(width)).tolist())


@metal
def test_staged_tables_reads_each_table_back_as_declared():
    face = _toy("all", _MEMBERS, *_TABLES)
    got = testing.staged_tables(face, _operands(), mx.float32)
    assert list(got) == ["a", "b", "c", "d", "e"]
    for table in _TABLES:
        assert got[table.name].dtype == table.dtype
        assert _bits(got[table.name]) == table.bits


@metal
def test_staged_tables_tells_two_members_in_another_order():
    members = _MEMBERS.replace("float* a;", "float* x;").replace(
        "float* e;", "float* a;"
    )
    members = members.replace("float* x;", "float* e;")
    got = testing.staged_tables(
        _toy("swapped", members, *_TABLES), _operands(), mx.float32
    )
    assert _bits(got["a"]) == _TABLES[4].bits[:4]


@metal
def test_a_member_of_another_type_fails_with_the_sentence():
    members = _MEMBERS.replace("int8_t* b;", "char* b;")
    with pytest.raises(RuntimeError, match="table 'b' is int8 data: the face declares"):
        mx.eval(
            testing.staged_tables(
                _toy("char", members, *_TABLES), _operands(), mx.float32
            )
        )


def test_staged_tables_refuses_what_it_cannot_launch():
    face = _toy("refused", _MEMBERS, *_TABLES)
    with pytest.raises(ValueError, match="has no tables"):
        testing.staged_tables(_toy("none", ""), _operands(), mx.float32)
    with pytest.raises(ValueError, match="operands given: 1"):
        testing.staged_tables(face, _operands()[:1], mx.float32)
    with pytest.raises(ValueError, match="scales has 4 elements"):
        testing.staged_tables(face, (_operands()[0], mx.ones((4,))), mx.float32)


@metal
def test_every_thread_of_a_partial_threadgroup_sees_the_whole_table():
    """MLX dispatches threads: a grid that is not whole threadgroups ends in
    a smaller one, which fills the table with the threads it has."""
    table = Table("t", mx.float32, tuple(float(i) for i in range(300)))
    tables = kernels.Tables()
    at = tables.add(table)
    kernel = mx.fast.metal_kernel(
        name="omlx_tables_partial",
        input_names=["x"],
        output_names=["y"],
        header=tables.header,
        source=tables.prologue + f"""
    const threadgroup float* t = {at};
    const uint g = threadgroup_position_in_grid.y;
    if (thread_index_in_threadgroup == 0) {{
      for (uint i = 0; i < 300u; ++i) {{
        y[g * 300 + i] = t[i] + x[0];
      }}
    }}
""",
    )
    (y,) = kernel(
        inputs=[mx.zeros((8,))],
        grid=(32, 6, 1),
        threadgroup=(32, 4, 1),
        output_shapes=[(600,)],
        output_dtypes=[mx.float32],
    )
    want = np.tile(np.arange(300, dtype=np.float32), 2)
    assert np.array_equal(np.array(y), want)


# The affine face whose 4-bit products read the weight values from a staged
# table: the values the masks give (j * 16^r), so MLX's bits. Every kernel
# over faces runs it in the tests of its own module.
LUT = Table(
    "lut", mx.float32, tuple(float(j * 16**r) for r in range(4) for j in range(16))
)

_TABLED = (
    (
        "  const device T* biases;\n",
        "  const device T* biases;\n  const threadgroup float* lut;\n",
    ),
    (
        "    const device T* b;\n",
        "    const device T* b;\n    const threadgroup float* l;\n",
    ),
    (
        "          b(m.biases + row * ROW_G + lane / SCALE_STEP_PER_THREAD) {}",
        "          b(m.biases + row * ROW_G + lane / SCALE_STEP_PER_THREAD),\n"
        "          l(m.lut) {}",
    ),
    (
        "      float b;\n    };",
        "      float b;\n      const threadgroup float* l;\n    };",
    ),
    (
        "      return {w + r * ROW_W, float(s[r * ROW_G]), float(b[r * ROW_G])};",
        "      return {w + r * ROW_W, float(s[r * ROW_G]), float(b[r * ROW_G]), l};",
    ),
    ("qdot(r.w, x_thread, r.s, r.b, sum)", "qdot(r.w, x_thread, r.s, r.b, sum, r.l)"),
    (
        "    float sum) {\n  float accum = 0;",
        "    float sum,\n    const threadgroup float* lut) {\n  float accum = 0;",
    ),
    ("(ws[i] & 0x000f)", "lut[ws[i] & 0xf]"),
    ("(ws[i] & 0x00f0)", "lut[16 + ((ws[i] >> 4) & 0xf)]"),
    ("(ws[i] & 0x0f00)", "lut[32 + ((ws[i] >> 8) & 0xf)]"),
    ("(ws[i] & 0xf000)", "lut[48 + ((ws[i] >> 12) & 0xf)]"),
)


def tabled(face: Dot) -> Dot:
    """``face`` (4-bit affine) reading its weight values from ``LUT``."""
    source = face.source
    for old, new in _TABLED:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    key = face.key + "_lut"
    return replace(
        face,
        key=key,
        source=source.replace(f"omlx_{face.key}", f"omlx_{key}"),
        type=face.type.replace(f"omlx_{face.key}", f"omlx_{key}"),
        tables=(LUT,),
    )


def tabled_dot(monkeypatch):
    """``formats.dot`` gives the tabled face of every 4-bit affine layer."""
    dot = formats.dot

    def patched(layer, k, n):
        face = dot(layer, k, n)
        if face is None or not face.key.startswith("affine_b4"):
            return face
        return tabled(face)

    monkeypatch.setattr(formats, "dot", patched)


@metal
@pytest.mark.parametrize("fast,k", [(True, 512), (False, 320)])
def test_the_row_skeleton_runs_a_tabled_face_to_the_bit(fast, k):
    mx.random.seed(3)
    w = mx.random.normal((64, k)).astype(mx.bfloat16)
    operands = mx.quantize(w, group_size=64, bits=4)
    x = mx.random.normal((k,)).astype(mx.bfloat16)
    face = affine.face(4, 64, fast, mx.bfloat16)
    want = testing.dot_rows(face, operands, x, 64, packs=(2, 2))
    got = testing.dot_rows(tabled(face), operands, x, 64, packs=(2, 2))
    assert np.array_equal(np.array(got.view(mx.uint16)), np.array(want.view(mx.uint16)))
    staged = testing.staged_tables(tabled(face), operands, mx.bfloat16)
    assert _bits(staged["lut"]) == LUT.bits


# The affine tile face whose 4-bit weights read their code's value from a
# staged table (0 to 15, the codes' exact values): the face's bits.
CODES = Table("codes", mx.float32, tuple(float(j) for j in range(16)))

_TILED = (
    (
        "  const device S* biases;\n\n  template <int K>",
        "  const device S* biases;\n  const threadgroup float* codes_lut;\n\n"
        "  template <int K>",
    ),
    (
        "      float b;\n    };",
        "      float b;\n      const threadgroup float* l;\n    };",
    ),
    (
        "    const device S* b;\n\n    Runs(",
        "    const device S* b;\n    const threadgroup float* l;\n\n    Runs(",
    ),
    (
        "          b(m.biases + row * ROW_GROUPS) {",
        "          b(m.biases + row * ROW_GROUPS),\n          l(m.codes_lut) {",
    ),
    (
        "      return {float(s[g]), float(b[g])};",
        "      return {float(s[g]), float(b[g]), l};",
    ),
    (
        "      return p.s * float(code_at(q, i)) + p.b;",
        "      return p.s * p.l[code_at(q, i)] + p.b;",
    ),
)


def tabled_tile(face):
    """``face`` (a 4-bit affine tile face) reading each code's value from
    ``CODES``, renamed ``codes_lut`` in its struct."""
    source = face.source
    for old, new in _TILED:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace = face.type.partition("::")[0]
    key = face.key + "_lut"
    return replace(
        face,
        key=key,
        source=source.replace(namespace, namespace + "_lut"),
        type=face.type.replace(namespace, namespace + "_lut"),
        tables=(Table("codes_lut", mx.float32, CODES.values),),
    )


@metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_a_tile_face_with_a_table_gives_its_weights_to_the_bit(dtype):
    mx.random.seed(5)
    operands = mx.quantize(mx.random.normal((64, 512)).astype(dtype), 64, 4)
    face = affine.tile_face(4, 64, dtype)
    rows = [0, 1, 17, 63]
    want = testing.tile_rows(face, operands, 512, rows, dtype)
    got = testing.tile_rows(tabled_tile(face), operands, 512, rows, dtype)
    for a, b in zip(got, want, strict=True):
        assert _bits(a) == _bits(b)
    staged = testing.staged_tables(tabled_tile(face), operands, dtype)
    assert _bits(staged["codes_lut"]) == CODES.bits


def test_a_tile_kernel_gives_its_pad_the_staged_bytes_back(monkeypatch):
    recorded = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: recorded.append(kw))
    monkeypatch.setattr(nax, "_kernels", {})
    face = affine.tile_face(4, 64, mx.bfloat16)
    nax._face_kernel(face)
    nax._face_kernel(tabled_tile(face))
    plain, staged = (kw["source"] for kw in recorded)
    assert "PAD / sizeof(T)];" in plain and "PAD > " not in plain
    assert "(PAD > 64 ? PAD - 64 : 0) / sizeof(T)];" in staged
    assert staged.startswith("    threadgroup float omlx_tg_float[16];\n")
