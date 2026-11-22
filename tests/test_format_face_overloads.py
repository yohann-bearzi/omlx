# SPDX-License-Identifier: Apache-2.0
"""Declared read signatures for format faces, and const skeleton reads.

A face declares its read members' signatures in Python (``DotReads``,
``TileReads``, ``CodesReads``) and oMLX emits the exact C++ signatures into
the face's source at explicit slots (``__OMLX_READ_<SLOT>__``). Nothing
parses C++: every unsafe form is refused at declaration time with a clear
``ValueError``. A face without ``reads`` is refused at construction: every
face declares its read members.
"""

from __future__ import annotations

import copy
import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

from omlx import formats
from omlx.formats import (
    ApplyRead,
    AuxType,
    Codes,
    CodesReads,
    Dot,
    DotReads,
    LoadRead,
    MaddRead,
    PairRead,
    Pointer,
    Tile,
    TileReads,
    TypeParam,
    WeightRead,
    affine,
    kernels,
    testing,
)
from tests import test_formats_kernels as dot
from tests import test_formats_tile as tile

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert mx.all(expected != 0).item()
    assert mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)).item()


def _dot_outputs(face, path):
    n, k = 8, 384  # One full block and a partial block.
    weights = (129 + mx.arange(n * k).reshape(n, k) % 7).astype(mx.uint8)
    scales = mx.arange(1, n + 1).astype(mx.float32)
    x = (1 + mx.arange(k) % 4).astype(mx.float32)
    w = (weights.astype(mx.float32) - 128) * scales[:, None]
    if path == "parts":
        stepped, direct = testing.lane_parts(face, (weights, scales), x, n)
        count = k // face.vpt
        expected = mx.sum(
            ((weights.astype(mx.float32) - 128) * x).reshape(n, count, face.vpt), axis=2
        )
        for output in (stepped, direct):
            _assert_bits(output[:, :count], expected)
            assert mx.all(output[:, count:] == 0).item()
        return direct[:, :count], expected
    if path in ("dot", "staged"):
        actual = testing.dot_rows(
            face, (weights, scales), x, n, staged=path == "staged"
        )
        return actual, mx.sum(w * x, axis=1)
    xs = mx.stack([x, x * 2])
    actual = testing.shared_rows(
        face, (weights, scales), xs, n, rps=2, unrolled=path == "unrolled"
    )
    return actual, mx.sum(xs[:, None, :] * w, axis=2)


# --- Const skeleton calls ------------------------------------------------


def _dot_reader(overloaded):
    source = dot._TOY
    if overloaded:
        anchor = "    float finish("
        assert source.count(anchor) == 1
        source = source.replace(
            anchor,
            """
    static float load(device float*, thread float* xt) {
      for (int i = 0; i < VPT; ++i) xt[i] = 0;
      return 0;
    }
    static float load(threadgroup float*, thread float* xt) {
      for (int i = 0; i < VPT; ++i) xt[i] = 0;
      return 0;
    }
    static float madd(float, Row, thread float*, float) { return 0; }
    template <typename A>
    static float apply(float, thread const Row&, thread float*, A) { return 0; }
""" + anchor,
        )
    return dot._toy(f"xt_{int(overloaded)}", source)


@pytest.mark.parametrize("path", ["dot", "staged", "shared", "unrolled", "parts"])
@pytest.mark.parametrize("overloaded", [False, True], ids=["const", "mutable"])
def test_dot_reads_const_activation_scratch(path, overloaded):
    actual, expected = _dot_outputs(_dot_reader(overloaded), path)
    _assert_bits(actual, expected)


@pytest.mark.parametrize("overload", ["const", "params", "codes", "short"])
@pytest.mark.parametrize("quad", [False, True], ids=["scalar", "fetch4"])
def test_tile_reads_const_weight_with_int_index(quad, overload):
    n, k = 8, 72  # Two complete quads and one scalar run.
    weights = (129 + mx.arange(n * k).reshape(n, k) % 17).astype(mx.uint8)
    group = mx.arange(1, n * 3 + 1).reshape(n, 3).astype(mx.float32)
    scale = mx.arange(1, n + 1).astype(mx.float32)
    source = tile._TOY
    slot = "    __OMLX_READ_WEIGHT__ {"
    assert source.count(slot) == 1
    extra = {
        "const": "",
        "params": (
            "    static float weight(thread P&, "
            "thread const Codes&, int) { return 0; }\n"
        ),
        "codes": (
            "    static float weight(thread const P&, "
            "thread Codes&, int) { return 0; }\n"
        ),
        "short": (
            "    static float weight(thread const P&, "
            "thread const Codes&, short) { return 0; }\n"
        ),
    }[overload]
    if quad:
        extra += """    void fetch4(int quad, thread Codes* out) const {
      for (int i = 0; i < 4; ++i) out[i] = fetch(4 * quad + i);
    }
"""
    reads = replace(
        tile._TILE_READS,
        weight=WeightRead(codes="thread const Codes&", params=("p", "q", "i")),
    )
    face = tile._toy(
        f"read_{overload}_{int(quad)}",
        source.replace(slot, extra + slot),
        reads,
    )
    rows = [7, 0, 3]
    want = tile._toy_decode(weights, group, scale)[rows]
    exact, rounded = testing.tile_rows(
        face, (weights, group, scale), k, rows, mx.bfloat16
    )
    _assert_bits(exact, want)
    _assert_bits(rounded, want.astype(mx.bfloat16))


@pytest.mark.parametrize("shared", [False, True], ids=["dot", "shared"])
@pytest.mark.parametrize("staged", [False, True], ids=["device", "threadgroup"])
def test_skeleton_qualifies_a_callers_mutable_activation_pointer(shared, staged):
    # Callers can own writable activation storage. The face only reads it.
    face = _dot_reader(True)
    matrices = kernels.Matrices()
    operands = matrices.add("m", face, shared=shared)
    storage = (
        "threadgroup float xs[256];"
        "for (int i = lane; i < 256; i += 32) xs[i] = x[i];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        if staged
        else "device float* xs = const_cast<device float*>(x);"
    )
    call = (
        "qmv_shared<float,256,1,1,8,false>(m,0,1,xs,lane,result);"
        if shared
        else "qmv_rows<float,256,1,0>(m,0,m,0,xs,lane,result);"
    )
    kernel = mx.fast.metal_kernel(
        name=f"kit_mutable_storage_{int(shared)}_{int(staged)}",
        input_names=["x", *operands],
        output_names=["out"],
        header=matrices.header,
        source=matrices.declarations
        + "uint lane = thread_index_in_simdgroup; float result[1];"
        + storage
        + call
        + "if (lane == 0) out[0] = result[0];",
    )
    (got,) = kernel(
        inputs=[
            mx.ones((256,)),
            mx.full((8, 256), 130, mx.uint8),
            mx.arange(1, 9, dtype=mx.float32),
        ],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
    )
    _assert_bits(got, mx.array([512.0]))


# --- Declaration-refusal tests (review bypass corpus) ----------------------


@pytest.mark.parametrize(
    ("name", "build", "fragment"),
    [
        # Pointer validation
        ("ptr_constant_space", lambda: Pointer("constant", "float"), "address space"),
        ("ptr_unknown_space", lambda: Pointer("generic", "float"), "address space"),
        (
            "ptr_const_in_space",
            lambda: Pointer("device const", "float"),
            "address space",
        ),
        ("ptr_non_bool_const", lambda: Pointer("device", "float", const=1), "const"),
        (
            "ptr_non_bool_const_first",
            lambda: Pointer("device", "float", const=True, const_first=1),
            "const_first",
        ),
        (
            "ptr_const_first_without_const",
            lambda: Pointer("thread", "float", const=False, const_first=True),
            "const_first",
        ),
        (
            "ptr_elem_ptr_const",
            lambda: Pointer("device", "float* const", const=True),
            "element",
        ),
        (
            "ptr_elem_ptr_const_ref",
            lambda: Pointer("device", "float* const&", const=True),
            "element",
        ),
        ("ptr_elem_ref", lambda: Pointer("thread", "float&", const=True), "element"),
        ("ptr_elem_short", lambda: Pointer("device", "short", const=True), "element"),
        (
            "ptr_elem_int_short",
            lambda: Pointer("device", "int short", const=True),
            "element",
        ),
        (
            "ptr_elem_default",
            lambda: Pointer("device", "float = 0", const=True),
            "element",
        ),
        # TypeParam validation (defaults, SFINAE, multi-params, keywords)
        ("tparam_empty", lambda: TypeParam(""), "identifier"),
        ("tparam_non_ascii", lambda: TypeParam("α"), "identifier"),
        ("tparam_default_short", lambda: TypeParam("typename = short"), "identifier"),
        ("tparam_default_val", lambda: TypeParam("X = float"), "identifier"),
        ("tparam_nontype_default", lambda: TypeParam("int N = 0"), "identifier"),
        ("tparam_bool_expr", lambda: TypeParam("bool B = (1 > 0)"), "identifier"),
        ("tparam_two", lambda: TypeParam("X, Y"), "identifier"),
        ("tparam_pack", lambda: TypeParam("X..."), "identifier"),
        (
            "tparam_sfinae",
            lambda: TypeParam("typename = metal::enable_if_t<!metal::is_const_v<X>>"),
            "identifier",
        ),
        ("tparam_keyword_short", lambda: TypeParam("short"), "reserved"),
        ("tparam_keyword_const", lambda: TypeParam("const"), "reserved"),
        ("tparam_reserved_row", lambda: TypeParam("Row"), "reserved"),
        ("tparam_reserved_codes", lambda: TypeParam("Codes"), "reserved"),
        # AuxType validation (by-value identifier; blocks refs/ptrs/SFINAE/keywords)
        ("aux_empty", lambda: AuxType(""), "identifier"),
        ("aux_const_ref", lambda: AuxType("thread const Aux&"), "identifier"),
        ("aux_const_ptr", lambda: AuxType("thread const Aux*"), "identifier"),
        ("aux_float_ref", lambda: AuxType("thread float&"), "identifier"),
        ("aux_float_ptr", lambda: AuxType("thread float*"), "identifier"),
        ("aux_star", lambda: AuxType("Aux*"), "identifier"),
        ("aux_amp", lambda: AuxType("Aux&"), "identifier"),
        (
            "aux_sfinae",
            lambda: AuxType("metal::enable_if_t<!metal::is_const_v<F>, float>"),
            "identifier",
        ),
        ("aux_float_builtin", lambda: AuxType("float"), "reserved"),
        ("aux_short_builtin", lambda: AuxType("short"), "reserved"),
        ("aux_half_builtin", lambda: AuxType("half"), "reserved"),
        ("aux_reserved_row", lambda: AuxType("Row"), "reserved"),
    ],
)
def test_primitives_refuse_unsafe_types(name, build, fragment):
    with pytest.raises(ValueError, match=fragment):
        build()


@pytest.mark.parametrize(
    ("name", "build", "fragment"),
    [
        # Review corpus: mutable device/threadgroup/generic, ptr-const, non-ptr
        (
            "load_mutable_device",
            lambda: LoadRead(input=Pointer("device", "float", const=False)),
            "const",
        ),
        (
            "load_mutable_threadgroup",
            lambda: LoadRead(input=Pointer("threadgroup", "float", const=False)),
            "const",
        ),
        (
            "load_generic_mutable_device",
            lambda: LoadRead(input=Pointer("device", TypeParam("X"), const=False)),
            "const",
        ),
        (
            "load_generic_mutable_staged",
            lambda: LoadRead(input=Pointer("threadgroup", TypeParam("X"), const=False)),
            "const",
        ),
        (
            "load_thread_space_input",
            lambda: LoadRead(input=Pointer("thread", "float", const=True)),
            "device",
        ),
        (
            "load_const_qualified_pointer_str",
            lambda: LoadRead(input="device float* const"),
            "load input",
        ),
        (
            "load_const_qualified_ref_str",
            lambda: LoadRead(input="device float* const&"),
            "load input",
        ),
        (
            "load_staged_const_qualified_str",
            lambda: LoadRead(input="threadgroup float* const"),
            "load input",
        ),
        (
            "load_non_pointer_value",
            lambda: LoadRead(input="device const float"),
            "load input",
        ),
        (
            "load_two_template_params",
            lambda: LoadRead(input=(TypeParam("X"), TypeParam("Y"))),
            "load input",
        ),
        (
            "load_const_scratch",
            lambda: LoadRead(scratch=Pointer("thread", "float", const=True)),
            "load scratch",
        ),
        (
            "load_device_scratch",
            lambda: LoadRead(scratch=Pointer("device", "float", const=False)),
            "load scratch",
        ),
        (
            "load_half_scratch",
            lambda: LoadRead(scratch=Pointer("thread", "half", const=False)),
            "load scratch",
        ),
        (
            "load_default_scratch_str",
            lambda: LoadRead(scratch="thread float* = (thread float*)0"),
            "load scratch",
        ),
        (
            "load_raw_aux_return",
            lambda: LoadRead(returns="Aux"),
            "load return",
        ),
        (
            "load_ref_return",
            lambda: LoadRead(returns="thread float&"),
            "load return",
        ),
        (
            "load_sfinae_return",
            lambda: LoadRead(returns="metal::enable_if_t<true, float>"),
            "load return",
        ),
        (
            "load_short_return",
            lambda: LoadRead(returns="short"),
            "load return",
        ),
        (
            "load_tparam_return",
            lambda: LoadRead(returns=TypeParam("R")),
            "load return",
        ),
        (
            "load_param_keyword",
            lambda: LoadRead(params=("short", "xt")),
            "parameter",
        ),
        (
            "load_param_default",
            lambda: LoadRead(params=("x", "xt = 0")),
            "parameter",
        ),
        (
            "load_param_duplicate",
            lambda: LoadRead(params=("x", "x")),
            "distinct",
        ),
        (
            "load_param_collides_tparam",
            lambda: LoadRead(input=TypeParam("x"), params=("x", "xt")),
            "collide",
        ),
        (
            "load_param_collides_aux",
            lambda: LoadRead(returns=AuxType("Aux"), params=("Aux", "xt")),
            "collide",
        ),
        (
            "load_param_arity",
            lambda: LoadRead(params=("x",)),
            "2 parameter",
        ),
        (
            "load_attr_qualifier",
            lambda: LoadRead(qualifiers=("__attribute__((always_inline))", "static")),
            "qualifier",
        ),
        (
            "load_missing_static",
            lambda: LoadRead(qualifiers=("inline",)),
            "static",
        ),
        (
            "load_dup_qualifier",
            lambda: LoadRead(qualifiers=("static", "static")),
            "duplicate",
        ),
        (
            "load_bad_slot",
            lambda: LoadRead(slot="LOAD"),
            "slot",
        ),
        (
            "load_aux_shadow_input_typeparam",
            lambda: LoadRead(input=TypeParam("Aux"), returns=AuxType("Aux")),
            "collide",
        ),
        (
            "load_aux_shadow_pointer_typeparam",
            lambda: LoadRead(
                input=Pointer("device", TypeParam("Aux"), const=True),
                returns=AuxType("Aux"),
            ),
            "collide",
        ),
    ],
)
def test_load_declaration_refuses_unsafe_forms(name, build, fragment):
    with pytest.raises(ValueError, match=fragment):
        build()


@pytest.mark.parametrize(
    ("name", "build", "fragment"),
    [
        # Review corpus: mutable scratch, ptr-const, template scratch, row, return
        (
            "madd_mutable_scratch",
            lambda: MaddRead(scratch=Pointer("thread", "float", const=False)),
            "madd scratch",
        ),
        (
            "madd_device_scratch",
            lambda: MaddRead(scratch=Pointer("device", "float", const=True)),
            "madd scratch",
        ),
        (
            "madd_half_scratch",
            lambda: MaddRead(scratch=Pointer("thread", "half", const=True)),
            "madd scratch",
        ),
        (
            "madd_template_elem_scratch",
            lambda: MaddRead(scratch=Pointer("thread", TypeParam("F"), const=True)),
            "madd scratch",
        ),
        (
            "madd_generic_scratch",
            lambda: MaddRead(scratch=TypeParam("XT")),
            "madd scratch",
        ),
        (
            "madd_ptr_const_scratch",
            lambda: MaddRead(scratch="thread float* const"),
            "madd scratch",
        ),
        (
            "madd_ptr_const_ref_scratch",
            lambda: MaddRead(scratch="thread float* const&"),
            "madd scratch",
        ),
        ("madd_const_row", lambda: MaddRead(row="thread const Row&"), "madd row"),
        (
            "madd_array_row",
            lambda: MaddRead(row="metal::array<uint, float>"),
            "madd row",
        ),
        ("madd_template_row", lambda: MaddRead(row=TypeParam("RR")), "madd row"),
        ("madd_short_return", lambda: MaddRead(returns="short"), "madd must return"),
        (
            "madd_sfinae_return",
            lambda: MaddRead(
                returns="metal::enable_if_t<!metal::is_const_v<F>, float>"
            ),
            "madd must return",
        ),
        ("madd_half_acc", lambda: MaddRead(acc="half"), "madd accumulator"),
        ("madd_aux_by_ref_str", lambda: MaddRead(aux="thread const Aux&"), "madd aux"),
        ("madd_aux_by_ptr_str", lambda: MaddRead(aux="thread const Aux*"), "madd aux"),
        ("madd_aux_default_str", lambda: MaddRead(aux="float = 0.0f"), "madd aux"),
        ("madd_aux_short", lambda: MaddRead(aux="short"), "madd aux"),
        ("madd_aux_raw_ident", lambda: MaddRead(aux="Aux"), "madd aux"),
        (
            "madd_param_default",
            lambda: MaddRead(params=("acc", "r", "xt", "sum = zero")),
            "parameter",
        ),
        (
            "madd_param_keyword",
            lambda: MaddRead(params=("acc", "r", "xt", "short")),
            "parameter",
        ),
        (
            "madd_param_collides_aux",
            lambda: MaddRead(aux=TypeParam("A"), params=("acc", "r", "xt", "A")),
            "collide",
        ),
        (
            "madd_attr_qualifier",
            lambda: MaddRead(qualifiers=("__attribute__((always_inline))", "static")),
            "qualifier",
        ),
    ],
)
def test_madd_declaration_refuses_unsafe_forms(name, build, fragment):
    with pytest.raises(ValueError, match=fragment):
        build()


@pytest.mark.parametrize(
    ("name", "build", "fragment"),
    [
        ("apply_row_instead_of_decoded", lambda: ApplyRead(decoded="Row"), "decoded"),
        (
            "apply_mutable_decoded",
            lambda: ApplyRead(decoded="thread Decoded&"),
            "decoded",
        ),
        (
            "apply_array_decoded",
            lambda: ApplyRead(decoded="thread const metal::array<float, 2>&"),
            "decoded",
        ),
        (
            "apply_template_decoded",
            lambda: ApplyRead(decoded=TypeParam("D")),
            "decoded",
        ),
        (
            "apply_mutable_scratch",
            lambda: ApplyRead(scratch=Pointer("thread", "float", const=False)),
            "apply scratch",
        ),
        (
            "apply_generic_scratch",
            lambda: ApplyRead(scratch=TypeParam("XT")),
            "apply scratch",
        ),
        (
            "apply_ptr_const_ref_scratch",
            lambda: ApplyRead(scratch="thread float* const&"),
            "apply scratch",
        ),
        ("apply_short_return", lambda: ApplyRead(returns="short"), "apply must return"),
        ("apply_half_acc", lambda: ApplyRead(acc="half"), "apply accumulator"),
        ("apply_bad_aux", lambda: ApplyRead(aux="thread const Aux&"), "apply aux"),
        ("apply_non_bool_wrap", lambda: ApplyRead(wrap=1), "wrap"),
        (
            "apply_bad_wrap_indent_newline",
            lambda: ApplyRead(wrap=True, wrap_indent="\n"),
            "wrap_indent",
        ),
        (
            "apply_bad_wrap_indent_code",
            lambda: ApplyRead(wrap=True, wrap_indent="short "),
            "wrap_indent",
        ),
        (
            "apply_owner_without_shared_slot",
            lambda: ApplyRead(shared_owner="Dot::Lanes<K>"),
            "shared_slot",
        ),
        (
            "apply_templates_without_shared_slot",
            lambda: ApplyRead(shared_templates=("int K",)),
            "shared_slot",
        ),
        (
            "apply_shared_slot_without_owner",
            lambda: ApplyRead(shared_slot="apply_shared"),
            "shared_owner",
        ),
        (
            "apply_shared_slot_equals_slot",
            lambda: ApplyRead(shared_slot="apply", shared_owner="Dot::Lanes"),
            "shared_slot",
        ),
        (
            "apply_shared_template_default",
            lambda: ApplyRead(
                shared_slot="apply_shared",
                shared_owner="Dot<T>::Lanes<K>",
                shared_templates=("typename T = float", "int K"),
            ),
            "shared_templates",
        ),
        (
            "apply_shared_template_comma",
            lambda: ApplyRead(
                shared_slot="apply_shared",
                shared_owner="Dot<T>::Lanes<K>",
                shared_templates=("typename T, int K",),
            ),
            "shared_templates",
        ),
        (
            "apply_shared_template_duplicate",
            lambda: ApplyRead(
                shared_slot="apply_shared",
                shared_owner="Dot<T>::Lanes<T>",
                shared_templates=("typename T", "int T"),
            ),
            "duplicate",
        ),
        (
            "apply_shared_template_collides_aux",
            lambda: ApplyRead(
                aux=TypeParam("T"),
                shared_slot="apply_shared",
                shared_owner="Dot<T>::Lanes<K>",
                shared_templates=("typename T", "int K"),
            ),
            "collide",
        ),
        (
            "apply_shared_owner_undeclared_arg",
            lambda: ApplyRead(
                shared_slot="apply_shared",
                shared_owner="Dot<U>::Lanes<K>",
                shared_templates=("typename T", "int K"),
            ),
            "shared_owner",
        ),
        (
            "apply_shared_owner_injection",
            lambda: ApplyRead(
                shared_slot="apply_shared",
                shared_owner=(
                    "Dot<T>::Lanes<K>; "
                    "static float madd(float, Row, thread float*, float) {} //"
                ),
                shared_templates=("typename T", "int K"),
            ),
            "shared_owner",
        ),
        (
            "apply_shared_qualifiers_static",
            lambda: ApplyRead(
                shared_slot="apply_shared",
                shared_owner="Dot<T>::Lanes<K>",
                shared_templates=("typename T", "int K"),
                shared_qualifiers=("static",),
            ),
            "shared_qualifiers",
        ),
        (
            "apply_shared_template_shadows_parameter",
            lambda: ApplyRead(
                params=("acc", "d", "xt", "K"),
                shared_slot="apply_shared",
                shared_owner="Dot<T>::Lanes<K>",
                shared_templates=("typename T", "int K"),
            ),
            "collide",
        ),
        (
            "apply_shared_template_shadows_aux",
            lambda: ApplyRead(
                aux=AuxType("K"),
                shared_slot="apply_shared",
                shared_owner="Dot<T>::Lanes<K>",
                shared_templates=("typename T", "int K"),
            ),
            "collide",
        ),
    ],
)
def test_apply_declaration_refuses_unsafe_forms(name, build, fragment):
    with pytest.raises(ValueError, match=fragment):
        build()


@pytest.mark.parametrize(
    ("name", "build", "fragment"),
    [
        ("dotreads_empty_loads", lambda: DotReads(load=()), "at least one LoadRead"),
        ("dotreads_bad_load", lambda: DotReads(load="load"), "LoadRead"),
        ("dotreads_bad_madd", lambda: DotReads(madd="madd"), "MaddRead"),
        ("dotreads_bad_apply", lambda: DotReads(apply="apply"), "ApplyRead"),
        (
            "dotreads_dup_load_slot",
            lambda: DotReads(
                load=(
                    LoadRead(Pointer("device", "float", const=True), slot="load"),
                    LoadRead(Pointer("threadgroup", "float", const=True), slot="load"),
                )
            ),
            "duplicate slot",
        ),
        (
            "dotreads_dup_generic_load",
            lambda: DotReads(
                load=(
                    LoadRead(TypeParam("XP"), slot="load_a"),
                    LoadRead(TypeParam("T"), slot="load_b"),
                )
            ),
            "duplicate load",
        ),
        (
            "dotreads_dup_device_load_const_order",
            lambda: DotReads(
                load=(
                    LoadRead(
                        Pointer("device", "float", const=True, const_first=True),
                        slot="load_a",
                    ),
                    LoadRead(
                        Pointer("device", "float", const=True, const_first=False),
                        slot="load_b",
                    ),
                )
            ),
            "duplicate load",
        ),
        (
            "dotreads_load_return_mismatch",
            lambda: DotReads(
                load=(
                    LoadRead(
                        Pointer("device", "float", const=True),
                        returns="float",
                        slot="load_a",
                    ),
                    LoadRead(
                        Pointer("threadgroup", "float", const=True),
                        returns=AuxType("Aux"),
                        slot="load_b",
                    ),
                )
            ),
            "same return type",
        ),
        (
            "dotreads_madd_aux_not_returned",
            lambda: DotReads(
                load=LoadRead(returns="float"),
                madd=MaddRead(aux=AuxType("Aux")),
            ),
            "load does not return",
        ),
        (
            "dotreads_madd_float_when_load_aux",
            lambda: DotReads(
                load=LoadRead(returns=AuxType("Aux")),
                madd=MaddRead(aux="float"),
            ),
            "load does not return",
        ),
        (
            "dotreads_apply_aux_not_returned",
            lambda: DotReads(
                load=LoadRead(returns=AuxType("Aux")),
                madd=MaddRead(aux=AuxType("Aux")),
                apply=ApplyRead(aux=AuxType("Other")),
            ),
            "load does not return",
        ),
        (
            "dotreads_slot_collision_load_madd",
            lambda: DotReads(
                load=LoadRead(slot="read"),
                madd=MaddRead(slot="read"),
            ),
            "duplicate slot",
        ),
    ],
)
def test_dot_reads_refuses_inconsistent_declarations(name, build, fragment):
    with pytest.raises(ValueError, match=fragment):
        build()


@pytest.mark.parametrize(
    ("name", "build", "fragment"),
    [
        # Review corpus: bad P/Codes/index, short, int short, refs, defaults, template
        (
            "weight_mutable_params",
            lambda: WeightRead(params_type="thread P&"),
            "params",
        ),
        ("weight_value_params", lambda: WeightRead(params_type="P"), "params"),
        ("weight_ptr_params", lambda: WeightRead(params_type="thread P*"), "params"),
        (
            "weight_ptr_const_params",
            lambda: WeightRead(params_type="thread P* const"),
            "params",
        ),
        (
            "weight_array_params",
            lambda: WeightRead(params_type="thread const metal::array<float, 4>&"),
            "params",
        ),
        (
            "weight_template_params",
            lambda: WeightRead(params_type=TypeParam("A")),
            "params",
        ),
        ("weight_mutable_codes", lambda: WeightRead(codes="thread Codes&"), "codes"),
        ("weight_moved_codes", lambda: WeightRead(codes="thread Codes&&"), "codes"),
        ("weight_ptr_codes", lambda: WeightRead(codes="Codes*"), "codes"),
        ("weight_template_codes", lambda: WeightRead(codes=TypeParam("Q")), "codes"),
        ("weight_short_index", lambda: WeightRead(index="short"), "index"),
        ("weight_int_short_index", lambda: WeightRead(index="int short"), "index"),
        (
            "weight_int_unsigned_index",
            lambda: WeightRead(index="int unsigned"),
            "index",
        ),
        (
            "weight_short_ref_index",
            lambda: WeightRead(index="thread const short&"),
            "index",
        ),
        ("weight_mutable_int_index", lambda: WeightRead(index="thread int&"), "index"),
        (
            "weight_const_int_ref_index",
            lambda: WeightRead(index="thread const int&"),
            "index",
        ),
        ("weight_moved_int_index", lambda: WeightRead(index="thread int&&"), "index"),
        ("weight_default_index", lambda: WeightRead(index="int = 0"), "index"),
        ("weight_template_index", lambda: WeightRead(index=TypeParam("I")), "index"),
        (
            "weight_short_return",
            lambda: WeightRead(returns="short"),
            "weight must return",
        ),
        (
            "weight_half_return",
            lambda: WeightRead(returns="half"),
            "weight must return",
        ),
        (
            "weight_param_keyword_short",
            lambda: WeightRead(params=("p", "q", "short")),
            "parameter",
        ),
        (
            "weight_param_keyword_unsigned",
            lambda: WeightRead(params=("p", "q", "unsigned")),
            "parameter",
        ),
        (
            "weight_param_default",
            lambda: WeightRead(params=("p", "q", "i = 0")),
            "parameter",
        ),
        (
            "weight_extra_param",
            lambda: WeightRead(params=("p", "q", "i", "extra")),
            "3 parameter",
        ),
        (
            "weight_bad_qualifier",
            lambda: WeightRead(qualifiers=("inline",)),
            "static",
        ),
        (
            "tilereads_bad_weight",
            lambda: TileReads(weight="weight"),
            "WeightRead",
        ),
    ],
)
def test_weight_declaration_refuses_unsafe_forms(name, build, fragment):
    with pytest.raises(ValueError, match=fragment):
        build()


@pytest.mark.parametrize(
    ("name", "build", "fragment"),
    [
        ("pair_short_return", lambda: PairRead(returns="ushort"), "pair must return"),
        ("pair_int_return", lambda: PairRead(returns="int"), "pair must return"),
        ("pair_float_return", lambda: PairRead(returns="float"), "pair must return"),
        ("pair_short_index", lambda: PairRead(index="short"), "index"),
        ("pair_int_short_index", lambda: PairRead(index="int short"), "index"),
        ("pair_mutable_int_index", lambda: PairRead(index="thread int&"), "index"),
        ("pair_template_index", lambda: PairRead(index=TypeParam("I")), "index"),
        ("pair_pack_ptr", lambda: PairRead(pack="uint*"), "pack"),
        ("pair_pack_ref", lambda: PairRead(pack="thread uint&"), "pack"),
        ("pair_pack_const_ref", lambda: PairRead(pack="thread const Pack&"), "pack"),
        ("pair_pack_short", lambda: PairRead(pack="short"), "pack"),
        ("pair_pack_float", lambda: PairRead(pack="float"), "pack"),
        ("pair_pack_template", lambda: PairRead(pack=TypeParam("Q")), "pack"),
        ("pair_param_keyword", lambda: PairRead(params=("q", "short")), "parameter"),
        ("codesreads_bad_pair", lambda: CodesReads(pair="pair"), "PairRead"),
    ],
)
def test_pair_declaration_refuses_unsafe_forms(name, build, fragment):
    with pytest.raises(ValueError, match=fragment):
        build()


# --- Face construction and slot validation -------------------------------


def _declared_dot_source() -> str:
    return dot._TOY


def _declared_dot_reads() -> DotReads:
    return dot._TOY_READS


def _declared_tile_source() -> str:
    return tile._TOY


def test_face_refuses_wrong_reads_kind():
    with pytest.raises(ValueError, match="reads must be DotReads, not TileReads"):
        Dot("k", _declared_dot_source(), "T", (), reads=TileReads())
    with pytest.raises(ValueError, match="reads must be DotReads, not str"):
        Dot("k", _declared_dot_source(), "T", (), reads="bad")
    with pytest.raises(ValueError, match="reads must be DotReads, not int"):
        Dot("k", _declared_dot_source(), "T", (), reads=0)
    with pytest.raises(ValueError, match="reads must be TileReads, not DotReads"):
        Tile("k", _declared_tile_source(), "T", (), reads=DotReads())
    with pytest.raises(ValueError, match="reads must be CodesReads, not TileReads"):
        Codes("k", "__OMLX_READ_PAIR__ {}", "T", (), reads=TileReads())


def test_face_refuses_missing_duplicate_midline_bad_suffix_and_leftover_slots():
    reads = _declared_dot_reads()
    src = _declared_dot_source()
    # Missing slot
    with pytest.raises(ValueError, match="missing read slot __OMLX_READ_MADD__"):
        Dot(
            "missing",
            src.replace("__OMLX_READ_MADD__", "int unused"),
            "T",
            (),
            reads=reads,
        )
    # Duplicate slot
    with pytest.raises(ValueError, match="repeats read slot __OMLX_READ_MADD__"):
        Dot(
            "dup",
            src.replace(
                "    __OMLX_READ_MADD__ {",
                "    __OMLX_READ_MADD__ {\n    __OMLX_READ_MADD__ {",
            ),
            "T",
            (),
            reads=reads,
        )
    # Commented / mid-line slot
    with pytest.raises(ValueError, match="must start its line"):
        Dot(
            "commented",
            src.replace("    __OMLX_READ_MADD__ {", "    // __OMLX_READ_MADD__ {"),
            "T",
            (),
            reads=reads,
        )
    with pytest.raises(ValueError, match="must start its line"):
        Dot(
            "quoted",
            src.replace(
                "    __OMLX_READ_MADD__ {",
                '    static_assert(true, "__OMLX_READ_MADD__");',
            ),
            "T",
            (),
            reads=reads,
        )
    # Bad trailing token on slot line (same line or next line)
    with pytest.raises(ValueError, match="must end with"):
        Dot(
            "deleted_slot",
            src.replace("    __OMLX_READ_MADD__ {", "    __OMLX_READ_MADD__ = delete;"),
            "T",
            (),
            reads=reads,
        )
    with pytest.raises(ValueError, match="followed by '\\{'"):
        Dot(
            "multiline_deleted_slot",
            src.replace(
                "    __OMLX_READ_MADD__ {", "    __OMLX_READ_MADD__\n    = delete;"
            ),
            "T",
            (),
            reads=reads,
        )
    with pytest.raises(ValueError, match="followed by '\\{'"):
        Dot(
            "multiline_trailing_ret",
            src.replace(
                "    __OMLX_READ_MADD__ {", "    __OMLX_READ_MADD__\n    -> short {"
            ),
            "T",
            (),
            reads=reads,
        )
    # A body brace on the next line stays valid.
    multiline_ok = Dot(
        "multiline_brace_ok",
        src.replace("    __OMLX_READ_MADD__ {", "    __OMLX_READ_MADD__\n    {"),
        "T",
        (),
        reads=reads,
    )
    assert "static float madd(" in multiline_ok.emitted_source
    # Unexpanded leftover slot (typo)
    with pytest.raises(ValueError, match="unexpanded read slot"):
        Dot(
            "leftover",
            src + "\n// __OMLX_READ_TYPO__\n",
            "T",
            (),
            reads=reads,
        )
    # A face without a declaration is refused before any slot is read.
    with pytest.raises(ValueError, match="declares its read members"):
        Dot("no_reads", src, "T", ())
    # shared_slot with empty face.shared
    shared_reads = DotReads(
        load=LoadRead(params=("x", "xt")),
        madd=MaddRead(params=("acc", "r", "xt", "")),
        apply=ApplyRead(
            params=("acc", "d", "xt", "aux"),
            shared_slot="apply_shared",
            shared_owner="Dot::Lanes<K>",
            shared_templates=("int K",),
        ),
    )
    with pytest.raises(ValueError, match="shared is empty"):
        Dot(
            "no_shared",
            src.replace("    __OMLX_READ_APPLY__ {", "    __OMLX_READ_APPLY__;"),
            "T",
            (),
            reads=shared_reads,
        )
    # shared_slot expects ';' in source, not '{'
    with pytest.raises(ValueError, match="must end with ';'"):
        Dot(
            "decl_with_brace",
            src,
            "T",
            (),
            shared="__OMLX_READ_APPLY_SHARED__ { return 0; }",
            reads=shared_reads,
        )
    # Missing shared_slot in shared
    with pytest.raises(
        ValueError, match="missing read slot __OMLX_READ_APPLY_SHARED__"
    ):
        Dot(
            "missing_shared",
            src.replace("    __OMLX_READ_APPLY__ {", "    __OMLX_READ_APPLY__;"),
            "T",
            (),
            shared="// no slot here\n",
            reads=shared_reads,
        )


def _dot_source_without_a_declared_apply() -> str:
    block = (
        "    __OMLX_READ_APPLY__ {\n" "      return madd(acc, d, xt, aux);\n" "    }\n"
    )
    assert dot._TOY.count(block) == 1
    return dot._TOY.replace(block, "")


def test_dot_with_vpt_or_shared_requires_declared_apply():
    reads = DotReads(load=LoadRead(), madd=MaddRead())
    source = _dot_source_without_a_declared_apply()
    with pytest.raises(ValueError, match="DotReads.apply"):
        Dot("vpt_without_apply", source, "T", (), vpt=8, reads=reads)
    with pytest.raises(ValueError, match="DotReads.apply"):
        Dot("shared_without_apply", source, "T", (), shared="int x;", reads=reads)
    # A dot-rows-only face needs no apply.
    face = Dot("plain_without_apply", source, "T", (), reads=reads)
    assert "static float madd(" in face.emitted_source


def test_expand_slot_handles_prefix_overlapping_slot_names():
    def source(load_marker: str, madd_marker: str) -> str:
        return (
            _declared_dot_source()
            .replace("    __OMLX_READ_LOAD__ {", f"    {load_marker} {{")
            .replace("    __OMLX_READ_MADD__ {", f"    {madd_marker} {{")
        )

    apply = _declared_dot_reads().apply
    first = Dot(
        "overlap_load_first",
        source("__OMLX_READ_LOAD__", "__OMLX_READ_LOAD__MADD__"),
        "T",
        (),
        reads=DotReads(
            load=LoadRead(slot="load"),
            madd=MaddRead(slot="load__madd"),
            apply=apply,
        ),
    )
    second = Dot(
        "overlap_madd_first",
        source("__OMLX_READ_LOAD__MADD__", "__OMLX_READ_LOAD__"),
        "T",
        (),
        reads=DotReads(
            load=LoadRead(slot="load__madd"),
            madd=MaddRead(slot="load"),
            apply=apply,
        ),
    )
    for face in (first, second):
        assert "static float load(" in face.emitted_source
        assert "static float madd(" in face.emitted_source


def test_a_face_without_declared_reads_is_refused():
    for kind, reads in ((Dot, "DotReads"), (Tile, "TileReads"), (Codes, "CodesReads")):
        with pytest.raises(
            ValueError, match=f"declares its read members.*{reads}"
        ):
            kind("no_reads", "struct F {};", "F", ())


# --- Acceptance & GPU execution with declared read signatures --------------


@pytest.mark.parametrize("path", ["dot", "staged", "shared", "unrolled", "parts"])
def test_declared_dot_face_emits_exact_metal_and_runs(path):
    src = _declared_dot_source().replace("KEY", f"decl_{path}")
    face = Dot(
        key=f"toy_decl_{path}",
        source=src,
        type=f"omlx_toy_decl_{path}::Dot",
        tensors=("group_scales",),
        vpt=8,
        reads=_declared_dot_reads(),
    )
    assert face.emitted_source == dot._TOY_LITERAL.replace("KEY", f"decl_{path}")
    actual, expected = _dot_outputs(face, path)
    _assert_bits(actual, expected)


def test_declared_dot_accepts_concrete_and_templated_const_loads():
    # Two concrete const loads (threadgroup + device)
    base = _declared_dot_source()
    concrete_src = base.replace(
        "    __OMLX_READ_LOAD__ {\n"
        "      copy(omlx_iota<VPT>(), x, xt);\n"
        "      return 0;\n"
        "    }",
        "    __OMLX_READ_LOAD_TG__ {\n"
        "      copy(omlx_iota<VPT>(), x, xt);\n"
        "      return 0;\n"
        "    }\n"
        "    __OMLX_READ_LOAD_DEV__ {\n"
        "      copy(omlx_iota<VPT>(), x, xt);\n"
        "      return 0;\n"
        "    }",
    ).replace("KEY", "concrete_decl")
    concrete_reads = DotReads(
        load=(
            LoadRead(
                Pointer("threadgroup", "float", const=True, const_first=True),
                params=("x", "xt"),
                slot="load_tg",
            ),
            LoadRead(
                Pointer("device", "float", const=True, const_first=True),
                params=("x", "xt"),
                slot="load_dev",
            ),
        ),
        madd=MaddRead(params=("acc", "r", "xt", "")),
        apply=ApplyRead(aux=TypeParam("A"), params=("acc", "d", "xt", "aux")),
    )
    face = Dot(
        key="toy_concrete_decl",
        source=concrete_src,
        type="omlx_toy_concrete_decl::Dot",
        tensors=("group_scales",),
        vpt=8,
        reads=concrete_reads,
    )
    for path in ("dot", "staged"):
        actual, expected = _dot_outputs(face, path)
        _assert_bits(actual, expected)

    # Two templated const pointer loads (const threadgroup X* and const device X*)
    tmpl_reads = DotReads(
        load=(
            LoadRead(
                Pointer("threadgroup", TypeParam("X"), const=True, const_first=True),
                params=("x", "xt"),
                slot="load_tg",
            ),
            LoadRead(
                Pointer("device", TypeParam("X"), const=True, const_first=True),
                params=("x", "xt"),
                slot="load_dev",
            ),
        ),
        madd=MaddRead(params=("acc", "r", "xt", "")),
        apply=ApplyRead(aux=TypeParam("A"), params=("acc", "d", "xt", "aux")),
    )
    tmpl_face = replace(
        face,
        key="toy_tmpl_const_decl",
        source=concrete_src.replace("concrete_decl", "tmpl_const_decl"),
        type="omlx_toy_tmpl_const_decl::Dot",
        reads=tmpl_reads,
    )
    for path in ("dot", "staged"):
        actual, expected = _dot_outputs(tmpl_face, path)
        _assert_bits(actual, expected)


@pytest.mark.parametrize(
    "declaration",
    ["struct Aux { float bias; };", "using Aux = float;", "typedef float Aux;"],
    ids=["struct", "using", "typedef"],
)
def test_declared_dot_accepts_aux_by_value_and_any_member_order(declaration):
    # Also places madd BEFORE load and includes C++17 digit separators and strings
    # with '//' to confirm zero C++ parsing happens.
    src = (
        _declared_dot_source()
        .replace(
            "    __OMLX_READ_LOAD__ {\n"
            "      copy(omlx_iota<VPT>(), x, xt);\n"
            "      return 0;\n"
            "    }\n"
            "    __OMLX_READ_MADD__ {\n"
            "      return acc + dot(omlx_iota<VPT>(), r.w, xt);\n"
            "    }",
            f"    static constant constexpr int MARK_A = 1'000;\n"
            f'    static_assert(MARK_A == 1000, "http://example/madd");\n'
            f"    {declaration}\n"
            "    __OMLX_READ_MADD__ {\n"
            "      return acc + dot(omlx_iota<VPT>(), r.w, xt);\n"
            "    }\n"
            "    __OMLX_READ_LOAD__ {\n"
            "      copy(omlx_iota<VPT>(), x, xt);\n"
            + (
                "      return {0.0f};\n"
                if declaration.startswith("struct")
                else "      return 0;\n"
            )
            + "    }\n"
            "    static constant constexpr int MARK_B = 2'000;",
        )
        .replace("KEY", "aux_decl")
    )
    aux = AuxType("Aux")
    reads = DotReads(
        load=LoadRead(
            input=TypeParam("XP"),
            returns=aux,
            params=("x", "xt"),
            qualifiers=("METAL_FUNC", "static"),
        ),
        madd=MaddRead(aux=aux, params=("acc", "r", "xt", "")),
        apply=ApplyRead(aux=aux, params=("acc", "d", "xt", "aux")),
    )
    face = Dot(
        key="toy_aux_decl",
        source=src,
        type="omlx_toy_aux_decl::Dot",
        tensors=("group_scales",),
        vpt=8,
        reads=reads,
    )
    for path in ("dot", "shared"):
        actual, expected = _dot_outputs(face, path)
        _assert_bits(actual, expected)


def test_declared_dot_supports_out_of_line_shared_apply():
    src = (
        _declared_dot_source()
        .replace(
            "    __OMLX_READ_APPLY__ {\n"
            "      return madd(acc, d, xt, aux);\n"
            "    }",
            "    __OMLX_READ_APPLY__;",
        )
        .replace("KEY", "ool_decl")
    )
    shared = (
        "namespace omlx_toy_ool_decl {\n"
        "__OMLX_READ_APPLY_SHARED__ {\n"
        "  return madd(acc, d, xt, aux);\n"
        "}\n"
        "}\n"
    )
    reads = DotReads(
        load=LoadRead(input=TypeParam("XP"), params=("x", "xt")),
        madd=MaddRead(
            scratch=Pointer("thread", "float", const=True, const_first=True),
            params=("acc", "r", "xt", "aux"),
        ),
        apply=ApplyRead(
            scratch=Pointer("thread", "float", const=True, const_first=True),
            params=("acc", "d", "xt", "aux"),
            shared_slot="apply_shared",
            shared_owner="Dot::Lanes<K>",
            shared_templates=("int K",),
            wrap=True,
        ),
    )
    face = Dot(
        key="toy_ool_decl",
        source=src,
        shared=shared,
        type="omlx_toy_ool_decl::Dot",
        tensors=("group_scales",),
        vpt=8,
        reads=reads,
    )
    assert "METAL_FUNC float Dot::Lanes<K>::apply(\n" in face.emitted_shared
    for path in ("dot", "shared", "unrolled"):
        actual, expected = _dot_outputs(face, path)
        _assert_bits(actual, expected)


@pytest.mark.parametrize("codes", ["Codes", "thread const Codes&"])
def test_declared_tile_face_emits_exact_metal_and_runs(codes):
    tag = "val" if codes == "Codes" else "ref"
    face = Tile(
        key=f"toy_decl_tile_{tag}",
        source=_declared_tile_source().replace("KEY", f"decl_tile_{tag}"),
        type=f"omlx_toy_decl_tile_{tag}::Tile",
        tensors=("group_scales", "row_scales"),
        reads=TileReads(WeightRead(codes=codes, params=("p", "q", "i"))),
    )
    expected_src = tile._TOY.replace("KEY", f"decl_tile_{tag}").replace(
        "    __OMLX_READ_WEIGHT__ {",
        f"    static float weight(thread const P& p, {codes} q, int i) {{",
    )
    assert face.emitted_source == expected_src
    n, k = 8, 72
    weights = (129 + mx.arange(n * k).reshape(n, k) % 17).astype(mx.uint8)
    group = mx.arange(1, n * 3 + 1).reshape(n, 3).astype(mx.float32)
    scale = mx.arange(1, n + 1).astype(mx.float32)
    rows = [7, 0, 3]
    want = tile._toy_decode(weights, group, scale)[mx.array(rows)]
    exact, rounded = testing.tile_rows(
        face, (weights, group, scale), k, rows, mx.bfloat16
    )
    _assert_bits(exact, want)
    _assert_bits(rounded, want.astype(mx.bfloat16))


@pytest.mark.parametrize("pack_decl", ["Pack", AuxType("Pack")], ids=["str", "aux"])
def test_declared_codes_face_emits_exact_metal_and_runs(pack_decl):
    src = """
struct DeclCodes {
  static constant constexpr int CONTRACT = 1;
  const device uchar* values;
  const device float* maps;
  template <int K>
  struct Runs {
    static constant constexpr int GROUP = 64;
    const device uchar* values;
    const device float* maps;
    Runs(DeclCodes m, size_t row)
      : values(m.values + row * K), maps(m.maps + row * (K / GROUP) * 2) {}
    float2 params(int g) const { return float2(maps[2*g], maps[2*g+1]); }
    struct Pack { uchar4 lo, hi; };
    Pack fetch(int run) const { return ((const device Pack*)values)[run]; }
    __OMLX_READ_PAIR__ {
      return uint(q.lo[j]) | (uint(q.hi[j]) << 16);
    }
  };
};
"""
    tag = "aux" if isinstance(pack_decl, AuxType) else "str"
    face = Codes(
        key=f"decl_codes_{tag}",
        source=src,
        type="DeclCodes",
        tensors=("maps",),
        reads=CodesReads(PairRead(pack=pack_decl, params=("q", "j"))),
    )
    assert "static uint pair(Pack q, int j)" in face.emitted_source
    k, n = 64, 8
    codes_arr = (1 + mx.arange(n * k).reshape(n, k) % 15).astype(mx.uint8)
    maps = mx.stack([mx.ones((n, 1)), mx.zeros((n, 1))], axis=-1).astype(mx.float32)
    levels, weights = testing.codes_rows(
        face, (codes_arr, maps), k, range(n), mx.float16
    )
    _assert_bits(levels, codes_arr.astype(mx.uint16))
    _assert_bits(weights, codes_arr.astype(mx.float32))


def test_declared_face_supports_replace_deepcopy_and_concurrent_construction():
    src = _declared_dot_source().replace("KEY", "copy_decl")
    face = Dot(
        key="toy_copy_decl",
        source=src,
        type="omlx_toy_copy_decl::Dot",
        tensors=("group_scales",),
        dtypes=(mx.float32,),
        vpt=8,
        reads=_declared_dot_reads(),
    )
    cloned = copy.deepcopy(face)
    assert cloned == face and cloned.emitted_source == face.emitted_source
    replaced = replace(face, dtypes=(mx.bfloat16,))
    assert replaced.dtypes == (mx.bfloat16,)
    assert replaced.emitted_source == face.emitted_source
    # Replacing reads re-emits from the template cleanly
    new_reads = replace(
        _declared_dot_reads(),
        madd=MaddRead(
            scratch=Pointer("thread", "float", const=True, const_first=True),
            params=("acc", "r", "xt", ""),
        ),
    )
    re_emitted = replace(face, reads=new_reads)
    assert "const thread float* xt" in re_emitted.emitted_source
    assert "thread const float* xt" in face.emitted_source

    def make(i):
        return Dot(
            key=f"toy_thread_{i}",
            source=_declared_dot_source().replace("KEY", f"thread_{i}"),
            type=f"omlx_toy_thread_{i}::Dot",
            tensors=("group_scales",),
            vpt=8,
            reads=_declared_dot_reads(),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        built = list(pool.map(make, range(8)))
    assert all(
        f.emitted_source == dot._TOY_LITERAL.replace("KEY", f"thread_{i}")
        for i, f in enumerate(built)
    )
    assert formats.DotReads is DotReads


# --- Shipped faces: built-in affine (219) and oQeL plugin (15) -------------


def _digest_records(records) -> str:
    h = hashlib.sha256()
    for rec in records:
        for part in rec:
            h.update(part.encode() + b"\0")
    return f"{len(records)} {h.hexdigest()}"


# Baseline digests captured from untouched wip/3 head ba421f5c, re-pinned on
# wip/a1-a13 (A13 shared-rows skeleton) and extended in T23 when
# affine._TILE_BITS gained 2 and 3 bits (+12 tile face records: 219 -> 231;
# +24 tile_wide/pair_kernel builders: 294 -> 318; all 219 existing face
# records and 294 existing builders verified byte-identical).
_AFFINE_ALL_DIGEST = (
    "231 39ddda59ca82a5835c9e339557fafe6b6cb936887c9465fb904e2d22d7f9826b"
)
_AFFINE_WIDE_AND_SG8_DIGEST = (
    "318 5905b3dfbb0349daa688a5c2f7384858cb4e5dd1d7aecfa97ff2a49e0eea39e5"
)
# Re-pinned on wip/a1-a13 together with the 219 affine records: the A13
# unrolled shared block adds three _Pragma("clang loop unroll(full)") lines
# (with the pack-fold for ROWS == 3), so its Matrices header differs from the
# baseline; the rolled block carries no placeholder line, so the rolled
# headers and the pair/codes citations are the baseline's. The plugin's own
# face Metal (emitted_source/emitted_shared) is unchanged; measured with the
# plugin staged at 238b9a9 on M5 Max.
_OQEL_15_DIGESTS = {
    False: "15 3a43bdfb1505022e273b2e2d4f5f3fc15799fb27838ea465b413bfa2f02696c3",
    True: "15 381372b7ba0d84f6589d06c1e881c5dde0e8fc96f20488b7a0e0b373474b3d43",
}
# Captured on release/3 (7e230438). The oQeL face sources are unchanged from
# ba421f5c (the 15-face digests above), but these 123 builder records include
# the gather and routed patch sources, which the release-2 fixes changed
# (moe_verify_gather, qwen35_moe_routed_decode, qwen35_moe_router). A base
# change re-pins this snapshot; the emission itself is pinned by the face
# digests inside this test, which also cover the emitted read signatures.
# Re-measured on wip/a1-a13 with the 15-face digests: 12 of these 123 builder
# records are the format_linear._shared_kernel builders that emit the unrolled
# shared skeleton (their header moves with it); the other 111, the rolled
# skeleton and the citations it sizes match the release-3 baseline.
_OQEL_BUILDER_DIGESTS = {
    False: "123 4f3543376bc4513c757ca46a6048f75378877611cc84154cc0622800aecad7fa",
    True: "123 0f3d0f0fcb8491b9f001d0bfd74c62a8f9f749d1ed99e4a7c2cbe5cc9b061908",
}


def test_the_serial_hc_up_header_carries_the_emitted_shared_metal():
    face = affine.face(4, 64, True, mx.bfloat16)
    header = affine.serial_up_header(face)
    assert header.startswith(f"{face.emitted_shared}\n")
    assert "__OMLX_READ_" not in header
    assert "hc_stream_up" in header


def test_every_shipped_affine_face_declares_reads_without_deprecation():
    affine.face.cache_clear()
    affine.tile_face.cache_clear()
    affine.codes_face.cache_clear()
    faces = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for bits in affine._BITS:
            for gs in affine._GROUP_SIZES:
                for fast in (False, True):
                    for dtype in (mx.bfloat16, mx.float16):
                        f = affine.face(bits, gs, fast, dtype)
                        assert isinstance(f.reads, DotReads)
                        assert "__OMLX_READ_LOAD__" in f.source
                        assert "__OMLX_READ_MADD__" in f.source
                        assert "__OMLX_READ_APPLY__" in f.source
                        assert "__OMLX_READ_APPLY_SHARED__" in f.shared
                        assert "static float load(" in f.emitted_source
                        assert "Dot<T>::Lanes<K>::apply(" in f.emitted_shared
                        faces += 1
        for bits in affine._TILE_BITS:
            for gs in affine._GROUP_SIZES:
                for dtype in (mx.bfloat16, mx.float16):
                    f = affine.tile_face(bits, gs, dtype)
                    assert isinstance(f.reads, TileReads)
                    assert "__OMLX_READ_WEIGHT__" in f.source
                    assert "static float weight(" in f.emitted_source
                    faces += 1
        for bits in affine._CODE_BITS:
            for gs in affine._GROUP_SIZES:
                for sd in affine._CODE_TYPES:
                    for bd in affine._CODE_TYPES:
                        f = affine.codes_face(bits, gs, sd, bd)
                        assert isinstance(f.reads, CodesReads)
                        assert "__OMLX_READ_PAIR__" in f.source
                        assert "static uint pair(" in f.emitted_source
                        faces += 1
    assert faces == 231
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_all_shipped_affine_faces_and_builders_are_byte_identical_to_ba421f5c(
    monkeypatch,
):
    from omlx.formats import tile_wide
    from omlx.patches import qwen35_verify_qmm as verify_qmm

    face_records = []
    for bits in affine._BITS:
        for gs in affine._GROUP_SIZES:
            for fast in (False, True):
                for dtype in (mx.bfloat16, mx.float16):
                    f = affine.face(bits, gs, fast, dtype)
                    m_plain = kernels.Matrices()
                    m_plain.add("m", f)
                    m_sh = kernels.Matrices()
                    m_sh.add("m", f, shared=True)
                    m_unr = kernels.Matrices(unrolled=True)
                    m_unr.add("m", f, shared=True)
                    m_pair = kernels.Matrices(unroll_pairs=True)
                    m_pair.add("m", f, pair=True)
                    face_records.append(
                        (
                            f.key,
                            f.emitted_source,
                            f.emitted_shared,
                            m_plain.header,
                            m_plain.declarations,
                            m_sh.header,
                            m_sh.declarations,
                            m_unr.header,
                            m_unr.declarations,
                            m_pair.header,
                            m_pair.declarations,
                        )
                    )
    for bits in affine._TILE_BITS:
        for gs in affine._GROUP_SIZES:
            for dtype in (mx.bfloat16, mx.float16):
                f = affine.tile_face(bits, gs, dtype)
                m = kernels.Matrices()
                m.add("m", f)
                face_records.append((f.key, f.emitted_source, m.header, m.declarations))
    for bits in affine._CODE_BITS:
        for gs in affine._GROUP_SIZES:
            for sd in affine._CODE_TYPES:
                for bd in affine._CODE_TYPES:
                    f = affine.codes_face(bits, gs, sd, bd)
                    m = kernels.Matrices()
                    m.add("m", f)
                    face_records.append(
                        (f.key, f.emitted_source, m.header, m.declarations)
                    )
    assert _digest_records(face_records) == _AFFINE_ALL_DIGEST

    builder_records = []

    def rec_mk(**kw):
        builder_records.append((kw["name"], kw.get("header", ""), kw["source"]))
        return lambda *a, **k: None

    monkeypatch.setattr(mx.fast, "metal_kernel", rec_mk)
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})
    for bits in affine._TILE_BITS:
        for gs in affine._GROUP_SIZES:
            for dtype in (mx.bfloat16, mx.float16):
                tf = affine.tile_face(bits, gs, dtype)
                tile_wide.kernel(tf)
                df = affine.face(bits, gs, True, dtype)
                tile_wide.pair_kernel.__wrapped__(df, tf)
    for bits in affine._CODE_BITS:
        for gs in affine._GROUP_SIZES:
            for sd in affine._CODE_TYPES:
                for bd in affine._CODE_TYPES:
                    cf = affine.codes_face(bits, gs, sd, bd)
                    for dtype in (mx.bfloat16, mx.float16):
                        verify_qmm._sg8_launch.__wrapped__(
                            (cf,), 4, 512, 512, gs, dtype, False
                        )
    assert _digest_records(builder_records) == _AFFINE_WIDE_AND_SG8_DIGEST


def _oqel():
    """The oQeL plugin's face module, from the installed package."""
    try:
        import omlx_oqel.face as face
        from omlx_oqel import codebook

        return face, codebook
    except ModuleNotFoundError as error:
        if error.name != "omlx_oqel":
            raise
        pytest.skip(
            "omlx-oqel is not installed (stage it with tools/integ/stage-plugin.sh)"
        )


def _oqel_digests(monkeypatch, face_mod, codebook, selects: bool):
    from omlx.formats import linear as format_linear
    from omlx.formats import tile_gather, tile_wide
    from omlx.patches import m5_gather_qmm_nax as nax
    from omlx.patches import moe_verify_gather as gather
    from omlx.patches import qwen35_moe_routed_decode as routed

    monkeypatch.setattr(kernels, "_small_tables_as_selects", lambda: selects)
    monkeypatch.setattr(nax, "_read_mlx_headers", lambda names: "")
    monkeypatch.setattr(nax, "_kernels", {})
    monkeypatch.setattr(nax, "_header_failed", False)
    monkeypatch.setattr(tile_gather, "_headers", lambda: "")
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})

    shared = affine.face(8, 128, True, mx.bfloat16)
    gate = affine.face(8, 64, True, mx.bfloat16)
    other = codebook.Layout(tables=tuple(reversed(codebook.FLASH_NEXT)))

    builder_records = []

    def rec_mk(**kw):
        builder_records.append((kw["name"], kw.get("header", ""), kw["source"]))
        return lambda *a, **k: None

    monkeypatch.setattr(mx.fast, "metal_kernel", rec_mk)

    face_records = []
    for bits in (2, 3, 4):
        for vpt in (8, 16):
            for layout in (codebook.DEFAULT, other):
                f = face_mod.dot(bits, layout, vpt)
                assert isinstance(f.reads, DotReads)
                m_plain = kernels.Matrices()
                m_plain.add("m", f)
                m_sh = kernels.Matrices()
                m_sh.add("m", f, shared=True)
                m_unr = kernels.Matrices(unrolled=True)
                m_unr.add("m", f, shared=True)
                m_pair = kernels.Matrices(unroll_pairs=True)
                m_pair.add("m", f, pair=True)
                face_records.append(
                    (
                        f.key,
                        f.emitted_source,
                        f.emitted_shared,
                        m_plain.header,
                        m_plain.declarations,
                        m_sh.header,
                        m_sh.declarations,
                        m_unr.header,
                        m_unr.declarations,
                        m_pair.header,
                        m_pair.declarations,
                    )
                )
                format_linear._kernel.__wrapped__(f, False)
                format_linear._kernel.__wrapped__(f, True)
                format_linear._shared_kernel.__wrapped__(f)
                gather._kernel.__wrapped__(f)
                gather._ordered_kernel.__wrapped__(f)
                routed._gate_up_kernel.__wrapped__(f, None, None)
                routed._down_kernel.__wrapped__(f, None)
                routed._gate_up_window_kernel.__wrapped__(f, shared, gate)
                routed._down_window_kernel.__wrapped__(f, shared)
        tf = face_mod.tile(bits, codebook.DEFAULT)
        assert isinstance(tf.reads, TileReads)
        m = kernels.Matrices()
        m.add("m", tf)
        face_records.append((tf.key, tf.emitted_source, m.header, m.declarations))
        tile_gather._build.__wrapped__(tf)
        tile_wide.kernel(tf)
        for kind in ("plain", "act", "act_map"):
            nax._face_kernel(tf, kind)
    return _digest_records(face_records), _digest_records(builder_records)


@pytest.mark.parametrize("selects", [False, True], ids=["staged", "selects"])
def test_every_oqel_plugin_face_emits_the_a6480a5_metal(monkeypatch, selects):
    """The plugin's 15 faces and the 123 served builders on them are
    byte-identical to a6480a5's literal read members: the declared
    signatures emit the same Metal at construction."""
    face_mod, codebook = _oqel()
    face_mod.dot.cache_clear()
    face_mod.tile.cache_clear()
    faces, builders = _oqel_digests(monkeypatch, face_mod, codebook, selects)
    assert faces == _OQEL_15_DIGESTS[selects]
    assert builders == _OQEL_BUILDER_DIGESTS[selects]


def test_plugin_overview_doc_builds_its_dot_face_with_reads():
    page = (
        Path(__file__).parents[1] / "docs" / "quant-format-plugins.md"
    ).read_text()
    (block,) = (
        text
        for language, text in dot._fenced(page)
        if language == "python" and "class MyFormat" in text
    )
    (source,) = (
        text
        for language, text in dot._BLOCKS
        if language == "cpp" and text.startswith("namespace omlx_myformat_b8 {")
    )
    scope = {"SOURCE": source}
    exec(block, scope)
    face = scope["FORMAT"].dot(None, 512, 8)
    assert isinstance(face.reads, DotReads)
    assert "static float load(" in face.emitted_source
    assert "__OMLX_READ_" not in face.emitted_source


def test_docs_specify_declared_read_signatures():
    page = (Path(__file__).parents[1] / "docs" / "quant-formats.md").read_text()
    assert "A face may also define a mutable overload." not in page
    assert "omlx/formats/reads.py" in page
    assert "A face without `reads` is refused at construction" in page
    assert "emitted `decode`" not in page
    for slot in (
        "__OMLX_READ_LOAD__",
        "__OMLX_READ_MADD__",
        "__OMLX_READ_APPLY__",
        "__OMLX_READ_WEIGHT__",
        "__OMLX_READ_PAIR__",
    ):
        assert slot in page
    assert "DeprecationWarning" not in page
    versions = page.split("\n## Versions\n")[1].split("\n## ")[0]
    for symbol in (
        "DotReads",
        "TileReads",
        "CodesReads",
        "LoadRead",
        "MaddRead",
        "ApplyRead",
        "WeightRead",
        "PairRead",
        "Pointer",
        "TypeParam",
        "AuxType",
    ):
        assert f"`{symbol}`" in versions
