# SPDX-License-Identifier: Apache-2.0
"""Tables a face declares as data: what a ``Table`` holds, what a face may
declare, and what a kernel's source gains (the staging) or does not (a face
without tables). The kit's ``staged_tables`` and the staging on the GPU."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import FrozenInstanceError, replace
from itertools import product

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


def _capture_nax_without_mlx_headers(monkeypatch):
    """Capture oMLX's source even if an earlier NAX build found no MLX headers."""
    monkeypatch.setattr(nax, "_kernels", {})
    monkeypatch.setattr(nax, "_header_failed", False)
    monkeypatch.setattr(nax, "_read_mlx_headers", lambda names: "")


def _affine_sources(monkeypatch):
    """Name, header and body of every served affine builder in digest order."""
    recorded = []

    def record(**kwargs):
        recorded.append((kwargs["name"], kwargs.get("header", ""), kwargs["source"]))

    monkeypatch.setattr(mx.fast, "metal_kernel", record)
    _capture_nax_without_mlx_headers(monkeypatch)
    faces = _faces()
    for face in faces:
        format_linear._kernel.__wrapped__(face, False)
        format_linear._kernel.__wrapped__(face, True)
        format_linear._shared_kernel.__wrapped__(face)
        gather._kernel.__wrapped__(face)
        gather._ordered_kernel.__wrapped__(face)
    for bits, group, dtype in product((4, 8), (32, 64, 128), (mx.bfloat16, mx.float16)):
        tile = affine.tile_face(bits, group, dtype)
        for kind in ("plain", "act", "act_map"):
            nax._face_kernel(tile, kind)
    shared, gate = affine.face(8, 128, True, mx.bfloat16), affine.face(
        8, 64, True, mx.bfloat16
    )
    for face in faces[:8]:
        (dtype,) = face.dtypes
        routed._gate_up_kernel.__wrapped__(face, shared, gate)
        routed._gate_up_kernel.__wrapped__(face, None, None)
        routed._gate_up_topk_kernel.__wrapped__(face, shared, gate, dtype)
        routed._gate_up_topk_kernel.__wrapped__(face, None, None, dtype)
        routed._down_kernel.__wrapped__(face, shared)
        routed._down_kernel.__wrapped__(face, None)
        routed._gate_up_window_kernel.__wrapped__(face, shared, gate)
        routed._down_window_kernel.__wrapped__(face, shared)
    return recorded


def _builder_digest(monkeypatch) -> str:
    recorded = _affine_sources(monkeypatch)
    return _metal_digest(recorded)


def _metal_digest(recorded):
    digest = hashlib.sha256()
    for record_ in recorded:
        for part in record_:
            digest.update(part.encode() + b"\0")
    return f"{len(recorded)} {digest.hexdigest()}"


# B8's kernels on the compact skeletons (U3b), as served: the existing
# builders (the format linear's shared-rows kernels among them) and 36
# plain, SwiGLU and row-mapped tile kernels. Adding support for tables must
# preserve their Metal byte for byte when the face declares none.
_METAL_DIGEST = "460 b7a4a469a789fbfcd5b9547543b62d8a56e07524fa2b3376373679dbb1974033"


def test_kernels_without_tables_are_the_same_text_as_before(monkeypatch):
    assert _builder_digest(monkeypatch) == _METAL_DIGEST


def _steel_sources(monkeypatch):
    from omlx.formats import tile_gather

    records = []

    def record(**kw):
        records.append((kw["name"], kw["header"], kw["source"]))

    monkeypatch.setattr(mx.fast, "metal_kernel", record)
    monkeypatch.setattr(tile_gather, "_headers", lambda: "")
    for bits, group, dtype in product((4, 8), (32, 64, 128), (mx.bfloat16, mx.float16)):
        tile_gather._build.__wrapped__(affine.tile_face(bits, group, dtype))
    return records


def test_affine_steel_gathers_keep_the_table_free_face_text(monkeypatch):
    records = _steel_sources(monkeypatch)
    # B8's independently captured table-free face headers and declarations
    # with the steel loader and launch body, excluding installed MLX headers.
    expected = "0b6af24687c3b42aef8c56bd09c037fad68b6f3d2908570850d926bfe4035691"
    assert _metal_digest(records) == f"12 {expected}"


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


@pytest.fixture
def staged(monkeypatch):
    """Every table staged, as on a GPU with tensor units."""
    monkeypatch.setattr(kernels, "_small_tables_as_selects", lambda: False)


@pytest.fixture(params=["staged", "selects"])
def form(request, monkeypatch):
    """Small tables staged, or read as selects: each GPU builds one form,
    the tests build both."""
    selects = request.param == "selects"
    monkeypatch.setattr(kernels, "_small_tables_as_selects", lambda: selects)
    return request.param


def test_a_kernel_stages_each_table_once_by_element_type(staged):
    a = Table("a", mx.float32, (1.0, -0.0))
    b = Table("b", mx.int8, (-1, 2, 3))
    c = Table("c", mx.float32, (0.5,))
    one = _toy("one", "", a, b)
    two = _toy("two", "", c, Table("a", mx.float32, (1, -0.0)))
    matrices = kernels.Matrices()
    matrices.add("p", one)
    matrices.add("q", two)
    matrices.add("r", one)
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
    assert (
        "p_scales, omlx_table<float, 2>{omlx_tg_float + 0}, "
        "omlx_table<int8_t, 3>{omlx_tg_int8_t + 0}};" in body
    )
    assert (
        "q_scales, omlx_table<float, 1>{omlx_tg_float + 2}, "
        "omlx_table<float, 2>{omlx_tg_float + 0}};" in body
    )
    assert body.count("decltype(omlx_tables_one::Dot::b), omlx_table<int8_t, 3>>") == 1
    assert "table 'b' is 3 values of int8" in body
    assert "bool VALUES = (N <= 4 && false)>" in header


def test_small_tables_are_values_read_by_selects_where_the_gpu_reads_them_so(
    monkeypatch,
):
    monkeypatch.setattr(kernels, "_small_tables_as_selects", lambda: True)
    small = Table("b", mx.int8, (-1, 2, 3))
    big = Table("e", mx.float32, (1.0,) * 5)
    matrices = kernels.Matrices()
    matrices.add("p", _toy("small", "", small))
    assert "bool VALUES = (N <= 4 && true)>" in matrices.header
    assert "omlx_staged" not in matrices.header
    body = matrices.declarations
    assert "threadgroup" not in body.split("const auto")[0]
    assert (
        "p_scales, omlx_table<int8_t, 3>{as_type<int8_t>(uint8_t(0xffu)), "
        "as_type<int8_t>(uint8_t(0x2u)), as_type<int8_t>(uint8_t(0x3u)), "
        "as_type<int8_t>(uint8_t(0x0u))}};" in body
    )
    assert matrices.staged_bytes == 0
    # A larger table is staged beside it.
    matrices.add("q", _toy("big", "", big))
    assert matrices.declarations.startswith("    threadgroup float omlx_tg_float[5];\n")
    assert matrices.staged_bytes == 20
    # Four values are still values.
    four = kernels.Matrices()
    four.add("p", _toy("four", "", Table("t", mx.uint16, (1, 2, 3, 4))))
    assert "omlx_table<uint16_t, 4>{as_type<uint16_t>(uint16_t(0x1u))" in (
        four.declarations
    )
    assert four.staged_bytes == 0


def test_a_gpu_with_tensor_units_stages_small_tables_and_one_without_selects():
    # The GPU's own generation, whatever MLX's kernels use: the M3 family
    # (15) reads selects, the M5 family (17) stages, as does a GPU not named.
    assert kernels._selects_for("applegpu_g15d")
    assert kernels._selects_for("applegpu_g16s")
    assert not kernels._selects_for("applegpu_g17s")
    assert kernels._selects_for("applegpu_g17p")
    assert not kernels._selects_for("applegpu_g18p")
    assert not kernels._selects_for("")
    assert not kernels._selects_for("applegpu_g15")
    for unknown in (
        "applegpu_g15d2",
        "xapplegpu_g15d",
        "applegpu_g15ab",
        "applegpu_g15!",
    ):
        assert not kernels._selects_for(unknown)
    architecture = (
        mx.device_info(mx.gpu).get("architecture", "")
        if mx.is_available(mx.gpu)
        else ""
    )
    assert kernels._small_tables_as_selects.__wrapped__() is kernels._selects_for(
        architecture
    )


@pytest.mark.parametrize("available", [False, True])
def test_small_tables_probe_the_gpu_on_first_use_from_a_cpu_stream(
    monkeypatch, available
):
    from concurrent.futures import ThreadPoolExecutor

    def is_available(device):
        assert device == mx.gpu
        return available

    queries = []

    def device_info(device=None):
        queries.append(device)
        assert available, "queried device info without a GPU"
        assert device is not None and device == mx.gpu, "queried the default device"
        return {"architecture": "applegpu_g15d"}

    monkeypatch.setattr(mx, "is_available", is_available)
    monkeypatch.setattr(mx, "device_info", device_info)
    probe = kernels._small_tables_as_selects
    probe.cache_clear()

    def first_use():
        with mx.stream(mx.cpu):
            return probe()

    try:
        with ThreadPoolExecutor(max_workers=1) as worker:
            assert worker.submit(first_use).result() is available
        assert probe() is available
        assert queries == ([mx.gpu] if available else [])
    finally:
        probe.cache_clear()


@pytest.mark.parametrize("info", [{}, {"architecture": None}])
def test_unknown_gpu_metadata_stages_small_tables(monkeypatch, info):
    monkeypatch.setattr(mx, "is_available", lambda device: True)
    monkeypatch.setattr(mx, "device_info", lambda device: info)
    assert not kernels._small_tables_as_selects.__wrapped__()


def test_standalone_table_header_resumes_the_callers_source(form):
    tables = kernels.Tables()
    assert tables.header == ""
    tables.add(Table("t", mx.float32, (1.0,)))
    assert tables.header.splitlines()[-1] == '#line 1 "<kernel header>"'


@pytest.mark.parametrize("invalid", [None, []])
def test_standalone_tables_require_a_table_before_lookup(invalid):
    with pytest.raises(TypeError, match="expected a Table"):
        kernels.Tables().add(invalid)


@metal
def test_a_standalone_table_expression_indexes_the_second_staged_table():
    tables = kernels.Tables()
    tables.add(Table("first", mx.int32, (101,) * 8))
    second = Table("second", mx.int32, tuple(range(20, 28)))
    at = tables.add(second)
    kernel = mx.fast.metal_kernel(
        name="omlx_test_standalone_table_offset",
        input_names=["x"],
        output_names=["y"],
        header=tables.header,
        source=tables.prologue
        + "    const uint i = thread_position_in_grid.x;\n"
        + f"    y[i] = {at}[i] + x[i];\n",
    )
    (result,) = kernel(
        inputs=[mx.zeros((8,), mx.int32)],
        grid=(8, 1, 1),
        threadgroup=(8, 1, 1),
        output_shapes=[(8,)],
        output_dtypes=[mx.int32],
    )
    assert tuple(result.tolist()) == second.values


def test_a_kernel_without_tables_does_not_ask_the_gpu(monkeypatch):
    def asked():
        raise AssertionError("asked")

    monkeypatch.setattr(kernels, "_small_tables_as_selects", asked)
    matrices = kernels.Matrices()
    matrices.add("p", _toy("none", ""))
    assert "omlx_table<" not in matrices.header + matrices.declarations


def test_the_values_form_is_defined_only_in_a_kernel_that_holds_one(form):
    matrices = kernels.Matrices()
    matrices.add("p", _toy("small3", "", Table("t", mx.float32, (1.0, 2.0, 3.0))))
    assert ("struct omlx_table<T, N, true>" in matrices.header) is (form == "selects")


@metal
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16, mx.int8])
def test_selects_read_the_low_two_bits_including_unused_slots(monkeypatch, dtype):
    monkeypatch.setattr(kernels, "_small_tables_as_selects", lambda: True)
    tables = kernels.Tables()
    at = tables.add(Table("t", dtype, (1.0, 2.0, 3.0)))
    kernel = mx.fast.metal_kernel(
        name=f"omlx_test_past_{dtype.size}_{str(dtype)[-4:]}",
        input_names=["x"],
        output_names=["y"],
        header=tables.header,
        source=tables.prologue
        + f"    const auto t = {at};\n"
        + "    y[thread_position_in_grid.x] = t[thread_position_in_grid.x] + x[0];\n",
    )
    (y,) = kernel(
        inputs=[mx.zeros((8,), dtype=dtype)],
        grid=(8, 1, 1),
        threadgroup=(8, 1, 1),
        output_shapes=[(8,)],
        output_dtypes=[dtype],
    )
    for start in (0, 4):
        assert y[start : start + 3].tolist() == [1, 2, 3]
        if dtype == mx.int8:
            assert y[start + 3].item() == 0
        else:
            assert mx.isnan(y[start + 3]).item()


def test_a_kernel_stages_at_most_4096_bytes_of_tables(staged):
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


@pytest.mark.parametrize("count,one", [(32, True), (33, False)])
def test_up_to_32_values_of_a_type_are_one_copy_a_thread(count, one):
    """Up to 32 values, a thread copies one, and a threadgroup of fewer
    threads than values copies the rest in a loop that others skip."""
    matrices = kernels.Matrices()
    matrices.add("p", _toy("p", "", Table("a", mx.int16, (0,) * count)))
    body = matrices.declarations
    assert (f"if (thread_index_in_threadgroup < {count}u)" in body) == one
    assert ("for (uint i = thread_index_in_threadgroup;" in body) == (not one)
    assert ("i = thread_index_in_threadgroup + threads_per" in body) == one
    assert (
        "i += threads_per_threadgroup.x * threads_per_threadgroup.y"
        " * threads_per_threadgroup.z)" in body
    )


# --- on the GPU -----------------------------------------------------------------


_MEMBERS = """  omlx_table<float, 4> a;
  omlx_table<int8_t, 3> b;
  omlx_table<half, 2> c;
  omlx_table<bfloat16_t, 2> d;
  omlx_table<float, 40> e;
  omlx_table<uint32_t, 1> f;"""

_TABLES = (
    Table("a", mx.float32, (1.5, -0.0, 3.0e38, 2.0**-149)),
    Table("b", mx.int8, (-128, 0, 127)),
    Table("c", mx.float16, (0.1, -65504.0)),
    Table("d", mx.bfloat16, (0.1, -3.0)),
    Table("e", mx.float32, (7.0,) * 40),
    Table("f", mx.uint32, (0xDEADBEEF,)),
)


def _operands():
    return mx.zeros((8, 8), mx.uint8), mx.ones((8,), mx.float32)


def _bits(array):
    width = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[array.dtype.size]
    return tuple(np.array(array.view(width)).tolist())


@metal
def test_staged_tables_reads_each_table_back_as_declared(form):
    face = _toy(f"all_{form}", _MEMBERS, *_TABLES)
    got = testing.staged_tables(face, _operands(), mx.float32)
    assert list(got) == ["a", "b", "c", "d", "e", "f"]
    for table in _TABLES:
        assert got[table.name].dtype == table.dtype
        assert _bits(got[table.name]) == table.bits


@metal
def test_staged_tables_tells_two_members_in_another_order(form):
    x, y = Table("x", mx.float32, (1.0, 2.0)), Table("y", mx.float32, (3.0, 4.0))
    members = "  omlx_table<float, 2> y;\n  omlx_table<float, 2> x;"
    got = testing.staged_tables(
        _toy(f"swapped_{form}", members, x, y), _operands(), mx.float32
    )
    assert _bits(got["x"]) == y.bits and _bits(got["y"]) == x.bits


@metal
def test_a_member_of_another_type_fails_with_the_sentence():
    members = _MEMBERS.replace("omlx_table<int8_t, 3> b;", "omlx_table<char, 3> b;")
    with pytest.raises(
        RuntimeError, match="table 'b' is 3 values of int8: the face declares"
    ):
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


def test_staged_tables_checks_the_face_before_reading_it():
    with pytest.raises(ValueError, match="is not a Dot or a Tile"):
        testing.staged_tables(object(), (), mx.float32)


@pytest.mark.parametrize("dtype", [mx.int32, mx.float64])
def test_staged_tables_checks_the_activation_dtype_before_building(monkeypatch, dtype):
    face = _toy("bad_activation", _MEMBERS, *_TABLES)
    monkeypatch.setattr(testing, "_staged_tables_kernel", lambda face: pytest.fail())
    with pytest.raises(ValueError, match="is not a float type of the GPU"):
        testing.staged_tables(face, _operands(), dtype)


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
    const auto t = {at};
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
        "  const device T* biases;\n  omlx_table<float, 64> lut;\n",
    ),
    (
        "    const device T* b;\n",
        "    const device T* b;\n    omlx_table<float, 64> l;\n",
    ),
    (
        "          b(m.biases + row * ROW_G + lane / SCALE_STEP_PER_THREAD) {}",
        "          b(m.biases + row * ROW_G + lane / SCALE_STEP_PER_THREAD),\n"
        "          l(m.lut) {}",
    ),
    (
        "      float b;\n    };",
        "      float b;\n      omlx_table<float, 64> l;\n    };",
    ),
    (
        "      return {w + r * ROW_W, float(s[r * ROW_G]), float(b[r * ROW_G])};",
        "      return {w + r * ROW_W, float(s[r * ROW_G]), float(b[r * ROW_G]), l};",
    ),
    ("qdot(r.w, x_thread, r.s, r.b, sum)", "qdot(r.w, x_thread, r.s, r.b, sum, r.l)"),
    (
        "    float sum) {\n  float accum = 0;",
        "    float sum,\n    omlx_table<float, 64> lut) {\n  float accum = 0;",
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
        shared=face.shared.replace(f"omlx_{face.key}", f"omlx_{key}"),
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


@metal
def test_staged_inputs_fit_in_a_threadgroup_s_memory_with_the_tables(staged):
    """dot_rows stages x beside the face's tables: 32 KiB of bfloat16 inputs
    fit alone, not with a table of 64 floats."""
    k = 16384
    w = mx.zeros((8, k), mx.bfloat16)
    operands = mx.quantize(w, group_size=64, bits=4)
    x = mx.zeros((k,), mx.bfloat16)
    face = affine.face(4, 64, True, mx.bfloat16)
    testing.dot_rows(face, operands, x, 8, staged=True)
    with pytest.raises(ValueError, match="staged tables of 256 do not fit"):
        testing.dot_rows(tabled(face), operands, x, 8, staged=True)


# The affine tile face whose 4-bit weights read their code's value from a
# staged table (0 to 15, the codes' exact values): the face's bits.
CODES = Table("codes", mx.float32, tuple(float(j) for j in range(16)))

_TILED = (
    (
        "  const device S* biases;\n\n  template <int K>",
        "  const device S* biases;\n  omlx_table<float, 16> codes_lut;\n\n"
        "  template <int K>",
    ),
    (
        "      float b;\n    };",
        "      float b;\n      omlx_table<float, 16> l;\n    };",
    ),
    (
        "    const device S* b;\n\n    Runs(",
        "    const device S* b;\n    omlx_table<float, 16> l;\n\n    Runs(",
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
    _capture_nax_without_mlx_headers(monkeypatch)
    face = affine.tile_face(4, 64, mx.bfloat16)
    nax._face_kernel(face)
    nax._face_kernel(tabled_tile(face))
    plain, staged = (kw["source"] for kw in recorded)
    assert "PAD / sizeof(T)];" in plain and "PAD > " not in plain
    assert "(PAD > 64 ? PAD - 64 : 0) / sizeof(T)];" in staged
    assert staged.startswith("    threadgroup float omlx_tg_float[16];\n")


# The kernel tests' toy face, its weights a level of a 4-value table picked by
# the code's two low bits: a table each GPU may read either way.
LEVELS4 = Table("levels", mx.float32, (-3.0, -1.0, 0.5, 2.0))

_TOY4 = (
    (
        "  const device float* scales;\n",
        "  const device float* scales;\n  omlx_table<float, 4> levels;\n",
    ),
    (
        "    const device float* s;\n",
        "    const device float* s;\n    omlx_table<float, 4> lv;\n",
    ),
    ("s(m.scales + row) {}", "s(m.scales + row), lv(m.levels) {}"),
    (
        "      const device uint8_t* w;\n    };",
        "      const device uint8_t* w;\n      omlx_table<float, 4> lv;\n    };",
    ),
    ("return {w + r * K};", "return {w + r * K, lv};"),
    (
        "const device uint8_t* w, thread const float* xt) {",
        "const device uint8_t* w, thread const float* xt,\n"
        "                     omlx_table<float, 4> lv) {",
    ),
    ("(float(w[I]) - 128.0f)", "lv[w[I] & 3u]"),
    ("dot(omlx_iota<VPT>(), r.w, xt)", "dot(omlx_iota<VPT>(), r.w, xt, r.lv)"),
)


@metal
@pytest.mark.parametrize("k", [512, 320])
def test_a_four_value_table_gives_the_face_s_rows_in_either_form(form, k):
    from tests.test_formats_kernels import _TOY as ROWS_TOY
    from tests.test_formats_kernels import _matrix

    source = ROWS_TOY.replace("KEY", f"four_{form}")
    for old, new in _TOY4:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    face = Dot(
        key=f"toy_four_{form}",
        source=source,
        type=f"omlx_toy_four_{form}::Dot",
        tensors=("scales",),
        tables=(LEVELS4,),
    )
    (codes, scales), x, _ = _matrix(16, k)
    levels = np.array(LEVELS4.values, dtype=np.float32)[np.array(codes) & 3]
    want = (levels @ np.array(x)) * np.array(scales)
    got = testing.dot_rows(face, (codes, scales), x, 16, packs=(2, 2))
    assert np.array_equal(np.array(got), want)
    assert _bits(
        testing.staged_tables(face, (codes, scales), mx.float32)["levels"]
    ) == (LEVELS4.bits)


# --- the review of R1 (pre-rtables-90958cc) ------------------------------------


def test_every_table_type_s_bits_and_metal_names():
    assert Table("t", mx.uint8, (0, 255)).bits == (0, 0xFF)
    assert Table("t", mx.uint32, (0, 2**32 - 1)).bits == (0, 0xFFFFFFFF)
    assert Table("t", mx.int16, (-1, -32768, 32767)).bits == (0xFFFF, 0x8000, 0x7FFF)
    assert Table("t", mx.int32, (-(2**31), 2**31 - 1)).bits == (0x80000000, 0x7FFFFFFF)
    # bfloat16's normals start at 2^-126: just below, a subnormal's step.
    below = Table("t", mx.bfloat16, (2.0**-127, 2.0**-127 + 3 * 2.0**-134))
    assert below.bits == (0x0040, 0x0042)
    names = {
        mx.float32: ("float", "uint32_t"),
        mx.float16: ("half", "uint16_t"),
        mx.bfloat16: ("bfloat16_t", "uint16_t"),
        mx.int8: ("int8_t", "uint8_t"),
        mx.uint8: ("uint8_t", "uint8_t"),
        mx.int16: ("int16_t", "uint16_t"),
        mx.uint16: ("uint16_t", "uint16_t"),
        mx.int32: ("int32_t", "uint32_t"),
        mx.uint32: ("uint32_t", "uint32_t"),
    }
    tables = [Table(f"t{i}", dtype, (1,) * 5) for i, dtype in enumerate(names)]
    matrices = kernels.Matrices()
    matrices.add("p", _toy("types", "", *tables))
    header, body = matrices.header, matrices.declarations
    for metal, bits in names.values():
        assert f"constant {bits} bits_{metal}[5] = {{" in header
        assert f"threadgroup {metal} omlx_tg_{metal}[5];" in body


@pytest.mark.parametrize("name", ["tâble", "x y", ""])
def test_a_table_s_name_is_an_ascii_identifier(name):
    with pytest.raises(ValueError, match="its name is not an identifier"):
        Table(name, mx.float32, (1.0,))


def test_tables_hold_at_most_4096_bytes_in_a_kernel():
    tables = kernels.Tables()
    tables.add(Table("a", mx.float32, (0.0,) * 1024))
    tables.add(Table("a", mx.float32, (0,) * 1024))  # the same table
    with pytest.raises(ValueError, match="'b': the kernel's tables would hold 4097"):
        tables.add(Table("b", mx.int8, (1,)))
    assert (
        kernels.table_bytes(
            _toy("x", "", Table("a", mx.float32, (0.0,) * 8)),
            _toy(
                "y", "", Table("a", mx.float32, (0.0,) * 8), Table("b", mx.int8, (1,))
            ),
        )
        == 33
    )


def test_routed_fold_counts_the_faces_of_each_launch_at_the_limit(monkeypatch):
    from tests.test_qwen35_moe_routed_decode import _block

    block = _block(1024, 320, experts=16)
    shared = block.shared_expert
    gate_up_faces = (
        formats.dot(block.switch_mlp.gate_up_proj, 1024, 640),
        formats.dot(shared.gate_proj, 1024, 320),
        formats.dot(block.shared_expert_gate, 1024, 1),
    )
    down_faces = (
        formats.dot(block.switch_mlp.down_proj, 320, 1024),
        formats.dot(shared.down_proj, 320, 1024),
    )
    assert gate_up_faces[1] != down_faces[1]
    launches = []

    def bytes_at_limit(*faces):
        launches.append(faces)
        return formats.TABLE_BYTES

    monkeypatch.setattr(kernels, "table_bytes", bytes_at_limit)
    assert routed._build_plan(block).launches.fold
    assert launches == [gate_up_faces, down_faces]


def test_staged_tables_refuses_an_operand_of_7_elements():
    face = _toy("seven", _MEMBERS, *_TABLES)
    with pytest.raises(ValueError, match="scales has 7 elements"):
        testing.staged_tables(face, (_operands()[0], mx.ones((7,))), mx.float32)


# The row kit's tile kernel runs threadgroups of K threads up to 256: at K = 8
# a threadgroup has fewer threads than the table entries, and at K = 264 the
# grid's edge leaves a threadgroup of 8. The staging copies every entry.
def _tile_table_source(n):
    edits = (
        (
            "  const device float* row_scales;\n",
            f"  const device float* row_scales;\n  omlx_table<float, {n}> off;\n",
        ),
        (
            "      float s;  // the group's scale times the row's\n",
            "      float s;  // the group's scale times the row's\n"
            f"      omlx_table<float, {n}> t;\n",
        ),
        (
            "    const device float* r;\n",
            f"    const device float* r;\n    omlx_table<float, {n}> t;\n",
        ),
        (
            "          r(m.row_scales + row) {}",
            "          r(m.row_scales + row),\n          t(m.off) {}",
        ),
        ("      return {g[group] * r[0]};", "      return {g[group] * r[0], t};"),
        (
            "      return (float(code) - 128.0f) * p.s;",
            f"      return (float(code) - p.t[code % {n}u]) * p.s;",
        ),
    )
    from tests.test_formats_tile import _TOY

    source = _TOY

    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


@metal
@pytest.mark.parametrize("n", [16, 24, 64])
@pytest.mark.parametrize("k", [8, 264])
def test_staging_fills_every_entry_in_a_threadgroup_of_fewer_threads(k, n):
    from tests.test_formats_tile import _toy as tile_toy
    from tests.test_formats_tile import _toy_matrix

    # Entries that differ from each other and from the other cases' (a
    # kernel's threadgroup memory may hold what an earlier launch left).
    offsets = tuple(100.0 + i + k / 8 for i in range(n))
    face = replace(
        tile_toy(f"off{k}_{n}", _tile_table_source(n)),
        tables=(Table("off", mx.float32, offsets),),
    )
    codes, group_scales, row_scales = _toy_matrix(16, k, seed=k + n)
    got, _ = testing.tile_rows(
        face, (codes, group_scales, row_scales), k, range(16), mx.bfloat16
    )
    levels = codes.astype(mx.float32) - mx.array(offsets)[codes.astype(mx.uint32) % n]
    scales = group_scales * row_scales[..., None]
    want = levels * scales[..., mx.arange(k) // 32]
    assert _bits(got) == _bits(want)
