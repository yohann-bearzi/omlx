# SPDX-License-Identifier: Apache-2.0
"""The parity skeletons' emitted Metal, pinned by digest against the released
kernels. CPU only: no GPU, no MLX (the reference builder is extracted from
the released source and run with a recording fake).

The row-exact skeleton's affine specs (below) are the per-shape kernels'
text, byte for byte (specs/parity-skeletons.md §3.1); its reference is
extracted from the release-3 commit eae63b6c itself, so the check compares
against the shipped generator, not a copy of it.

The verify (MMA) skeleton's 108 specs (after it) are compared with the
released builder the same way.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import importlib.util
import json
import logging
import re
import subprocess
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import (
    Dot,
    Table,
    affine,
    kernels,
    row_exact_skeleton,
    testing,
    tuning,
)
from omlx.formats.linear import FormatLinear
from omlx.patches import module_cache, qwen35_gdn_prework, row_exact_qmv

_REFERENCE = Path(__file__).parent / "data" / "parity_row_exact_a13.json"
_LAUNCH_19_REFERENCE = (
    Path(__file__).parent / "data" / "parity_row_exact_19_launches.json"
)
# The combined digest of the 240 pinned specs, captured from release-3
# eae63b6c's own builders (specs/parity-skeletons-evidence/a13/test_emit.py
# proved 240/240; the serialization here is the one below).
A13_DIGEST = "fd884b476cbabb5772730282d87b344412c95c853b007f7ba1ed51d665e02f9d"
# The combined digest of all 19 served launch captures from release-3 eae63b6c
# (spec, grid, threadgroup, template, input_shapes, output_shapes, output_dtypes).
A13_LAUNCH_19_DIGEST = (
    "6d39e45f0c311672076ceb86711ea67c964431e85c2dafba084d5b104b2d6359"
)


def _reference():
    return json.loads(_REFERENCE.read_text())


def _spec_digest(spec) -> str:
    parts = [spec["name"], spec["header"], spec["source"], *spec["inputs"], *spec["outputs"]]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def test_a13_affine_240_specs_byte_identical_to_eae63b6c():
    """240/240 (bits, group, fast, unroll, projections) specs equal the pinned
    release-3 builders exactly (names, Metal, inputs and outputs)."""
    reference = _reference()
    assert len(reference) == 240
    mismatches = []
    digest = hashlib.sha256()
    for want in reference:
        ops = row_exact_skeleton.affine_row_ops(
            want["bits"], want["group_size"], want["fast"]
        )
        spec = row_exact_skeleton.kernel_spec(
            ops, want["bits"], want["group_size"], want["fast"],
            want["unrolled"], want["count"],
        )
        got = {
            "name": spec["name"],
            "header": spec["header"],
            "source": spec["source"],
            "inputs": list(spec["input_names"]),
            "outputs": list(spec["output_names"]),
        }
        if got != {k: want[k] for k in got}:
            mismatches.append((want["bits"], want["group_size"], want["fast"],
                               want["unrolled"], want["count"]))
        digest.update(_spec_digest(got).encode())
    assert not mismatches, mismatches[:4]
    assert digest.hexdigest() == A13_DIGEST, digest.hexdigest()


def test_qmv_shared_unrolls_inner_block_and_folds_rows3():
    """The shared skeleton pragma-unrolls all three loops inside
    omlx_shared_block (rolled carries no pragma or fold helper), and keeps
    the pack-fold only for ROWS == 3 where pragma loses on M5 Max."""
    from omlx.formats import kernels

    pragma = '_Pragma("clang loop unroll(full)")'
    rolled = kernels._qmv_shared(False)
    unrolled = kernels._qmv_shared(True)
    assert pragma not in rolled
    assert "omlx_shared_folded" not in rolled

    block_start = unrolled.index("METAL_FUNC void omlx_shared_block(")
    block_end = unrolled.index("METAL_FUNC void omlx_qmv_shared(")
    block = unrolled[block_start:block_end]
    # All three loops in omlx_shared_block carry the full-unroll pragma.
    assert block.count(pragma) == 3, block
    assert f"{pragma}\nfor (int i = 0; i < ROWS; i++) {{" in block
    assert f"{pragma}\nfor (int r = 0; r < RPS; r++) {{" in block
    assert "if constexpr (ROWS == 3) {" in block
    assert "omlx_shared_folded<K, RPS, PARTIAL>(" in block


def test_row_index_width_is_chosen_at_construction():
    assert tuning.row_index_width(2560, 6144) is tuning.RowIndexWidth.INT32_CHECKED
    assert tuning.row_index_width(2**28, 16) is tuning.RowIndexWidth.SIZE_T64


_STAGED_BLOCK = re.compile(r"namespace omlx_staged \{[^}]*\}\n?")

_PLUGIN_DOT = r"""
namespace omlx_sk_KEY {
struct Dot {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* scales;

  template <int K>
  struct Lanes {
    static constant constexpr int VPT = 8;
    static constant constexpr int BLOCK = 32 * VPT;

    const device uint8_t* w;
    const device float* s;

    Lanes(Dot m, size_t row, uint lane)
        : w(m.codes + row * K + lane * VPT), s(m.scales + row) {}

    struct Row {
      const device uint8_t* w;
    };
    Row row(int r) const { return {w + r * K}; }
    void step() { w += BLOCK; }

    template <typename XP, int... I>
    static void copy(omlx_ints<I...>, XP x, thread float* xt) {
      ((xt[I] = float(x[I])), ...);
    }
    template <int... I>
    static float dot(
        omlx_ints<I...>, const device uint8_t* w, thread const float* xt) {
      return (... + ((float(w[I]) - 128.0f) * xt[I]));
    }

    __OMLX_READ_LOAD__ {
      copy(omlx_iota<VPT>(), x, xt);
      return 0.0f;
    }
    __OMLX_READ_MADD__ {
      return acc + dot(omlx_iota<VPT>(), r.w, xt);
    }
    float finish(int r, float sum) const { return sum * s[r]; }

    using Decoded = Row;
    static Decoded decode(Row r) { return r; }
    __OMLX_READ_APPLY__ {
      return madd(acc, d, xt, aux);
    }
  };
};
}  // namespace omlx_sk_KEY
"""


_PLUGIN_READS = formats.DotReads(
    load=formats.LoadRead(input=formats.TypeParam("XP"), params=("x", "xt")),
    madd=formats.MaddRead(params=("acc", "r", "xt", "")),
    apply=formats.ApplyRead(
        aux=formats.TypeParam("A"), params=("acc", "d", "xt", "aux")
    ),
)


def _plugin_dot(key: str, *, dtypes=(mx.float32,), vpt: int = 8) -> Dot:
    return Dot(
        key=f"sk_{key}",
        source=_PLUGIN_DOT.replace("KEY", key),
        type=f"omlx_sk_{key}::Dot",
        tensors=("group_scales",),
        dtypes=dtypes,
        vpt=vpt,
        reads=_PLUGIN_READS,
    )


def _tabled_plugin_dot(key: str, table_len: int) -> tuple[Dot, tuple[float, ...]]:
    values = tuple((i % 7 - 3) * 0.25 for i in range(table_len))
    table = Table("lut", mx.float32, values)
    mask = table_len - 1
    src = f"""
namespace omlx_sk_{key} {{
struct Dot {{
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* scales;
  omlx_table<float, {table_len}> lut;

  template <int K>
  struct Lanes {{
    static constant constexpr int VPT = 8;
    static constant constexpr int BLOCK = 32 * VPT;

    const device uint8_t* w;
    const device float* s;
    omlx_table<float, {table_len}> lut;

    Lanes(Dot m, size_t row, uint lane)
        : w(m.codes + row * K + lane * VPT), s(m.scales + row), lut(m.lut) {{}}

    struct Row {{
      const device uint8_t* w;
      omlx_table<float, {table_len}> lut;
    }};
    Row row(int r) const {{ return {{w + r * K, lut}}; }}
    void step() {{ w += BLOCK; }}

    __OMLX_READ_LOAD__ {{
      for (int i = 0; i < VPT; i++) xt[i] = float(x[i]);
      return 0.0f;
    }}
    __OMLX_READ_MADD__ {{ return 0.0f; }}
    using Decoded = Row;
    static Decoded decode(Row r) {{ return r; }}
    __OMLX_READ_APPLY__ {{
      for (int i = 0; i < VPT; i++) {{
        acc += d.lut[uint(d.w[i]) & {mask}u] * xt[i];
      }}
      return acc;
    }}
    float finish(int r, float sum) const {{ return sum * s[r]; }}
  }};
}};
}}  // namespace omlx_sk_{key}
"""
    face = Dot(
        key=f"sk_{key}",
        source=src,
        type=f"omlx_sk_{key}::Dot",
        tensors=("group_scales",),
        dtypes=(mx.float32,),
        vpt=8,
        tables=(table,),
        reads=formats.DotReads(apply=formats.ApplyRead()),
    )
    return face, values


def test_dot_row_ops_rejects_invalid_face_or_zero_vpt():
    with pytest.raises(TypeError):
        row_exact_skeleton.dot_row_ops("not_a_dot")
    zero_vpt = replace(_plugin_dot("zero_vpt"), vpt=0)
    with pytest.raises(ValueError, match="vpt"):
        row_exact_skeleton.dot_row_ops(zero_vpt)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_affine_row_ops_size_t64_widens_offsets_while_int32_stays_byte_identical():
    ops32 = row_exact_skeleton.affine_row_ops(
        8, 64, True, tuning.RowIndexWidth.INT32_CHECKED
    )
    ops64 = row_exact_skeleton.affine_row_ops(
        8, 64, True, tuning.RowIndexWidth.SIZE_T64
    )
    spec32 = row_exact_skeleton.kernel_spec(ops32, 8, 64, True, False, 0)
    spec64 = row_exact_skeleton.kernel_spec(ops64, 8, 64, True, False, 0)
    assert "const size_t in_vec_size_w" not in spec32["header"]
    assert "const size_t in_vec_size_w" in spec64["header"]
    assert "size_t(out_row)" in spec64["header"]
    assert kernels.hand_tables(spec64["header"]) == ()

    mx.random.seed(19)
    n, k, rows = 32, 256, 4
    w, s, b = mx.quantize(
        (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16), group_size=64, bits=8
    )
    x = (mx.random.normal((rows, k)) * 0.5).astype(mx.bfloat16)
    kern = mx.fast.metal_kernel(**spec64)
    (got,) = kern(
        inputs=[x, w, s, b],
        template=[
            ("T", mx.bfloat16),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", 2),
            ("RPS", 4),
        ],
        grid=(32, 2 * (rows // 2), n // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.bfloat16],
    )
    want = mx.concatenate(
        [
            mx.quantized_matmul(
                x[r : r + 1], w, scales=s, biases=b, transpose=True,
                group_size=64, bits=8,
            )
            for r in range(rows)
        ]
    )
    assert mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("k", [256, 320])
@pytest.mark.parametrize("n", [40, 12])
@pytest.mark.parametrize("unrolled", [False, True])
def test_dot_row_ops_compiles_rev1_plugin_face_through_same_skeleton(k, n, unrolled):
    face = _plugin_dot(f"plain_{k}_{n}_{int(unrolled)}")
    ops = row_exact_skeleton.dot_row_ops(face)
    fast = k % (32 * face.vpt) == 0
    spec = row_exact_skeleton.emit_row_exact_kernel(
        ops, fast, unrolled, 0, f"omlx_test_plugin_{k}_{n}_{int(unrolled)}"
    )
    assert "omlx_revision" not in spec["header"]
    assert '#line ' not in spec["header"]
    assert kernels.hand_tables(spec["header"]) == ()

    mx.random.seed(k + n)
    rows, per_group, rps = 4, 2, 2
    codes = mx.random.randint(0, 256, (n, k)).astype(mx.uint8)
    scales = mx.array([2.0 ** (r % 5 - 2) for r in range(n)], dtype=mx.float32)
    x = mx.random.randint(-8, 9, (rows, k)).astype(mx.float32)
    assert row_exact_skeleton.prove_dot_face(
        face,
        k,
        n,
        mx.float32,
        ((codes.shape, codes.dtype), (scales.shape, scales.dtype)),
    )
    kern = mx.fast.metal_kernel(**spec)
    (got,) = kern(
        inputs=[x, codes, scales],
        template=[
            ("T", mx.float32),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", per_group),
            ("RPS", rps),
        ],
        grid=(32, 2 * (rows // per_group), (n + 2 * rps - 1) // (2 * rps)),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.float32],
    )
    want = (x @ (codes.astype(mx.float32) - 128.0).T) * scales
    assert mx.array_equal(got.view(mx.uint32), want.view(mx.uint32)).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("table_len", [4, 64])
@pytest.mark.parametrize("selects", [False, True])
def test_dot_row_ops_compiles_tabled_plugin_face_through_same_skeleton(
    monkeypatch, table_len, selects
):
    monkeypatch.setattr(kernels, "_small_tables_as_selects", lambda: selects)
    face, values = _tabled_plugin_dot(f"tabled_{table_len}_{int(selects)}", table_len)
    ops = row_exact_skeleton.dot_row_ops(face)
    spec = row_exact_skeleton.emit_row_exact_kernel(
        ops, False, True, 0, f"omlx_test_tabled_{table_len}_{int(selects)}"
    )
    assert "omlx_table" in spec["header"]
    assert "omlx_revision" not in spec["header"]
    assert kernels.hand_tables(_STAGED_BLOCK.sub("", spec["header"])) == ()
    if table_len <= 4 and selects:
        assert "omlx_staged" not in spec["header"]
    else:
        assert "omlx_staged" in spec["header"]

    mx.random.seed(table_len)
    n, k, rows = 20, 320, 4
    codes = mx.random.randint(0, 256, (n, k)).astype(mx.uint8)
    scales = mx.array([2.0 ** (r % 4 - 1) for r in range(n)], dtype=mx.float32)
    x = mx.random.randint(-4, 5, (rows, k)).astype(mx.float32)
    specs = ((codes.shape, codes.dtype), (scales.shape, scales.dtype))
    assert row_exact_skeleton.prove_dot_face(face, k, n, mx.float32, specs)
    # Second call hits cached proof.ran without rebuilding the contract kernel.
    monkeypatch.setattr(
        row_exact_skeleton,
        "_dot_contract_kernel",
        lambda *_a, **_kw: pytest.fail("rebuilt proven contract kernel"),
    )
    assert row_exact_skeleton.prove_dot_face(face, k, n, mx.float32, specs)

    kern = mx.fast.metal_kernel(**spec)
    (got,) = kern(
        inputs=[x, codes, scales],
        template=[
            ("T", mx.float32),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", 2),
            ("RPS", 4),
        ],
        grid=(32, 2 * (rows // 2), (n + 7) // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.float32],
    )
    lut = mx.array(values, dtype=mx.float32)
    w_ref = lut[codes.astype(mx.uint32) & (table_len - 1)]
    want = (x @ w_ref.T) * scales
    assert mx.array_equal(got.view(mx.uint32), want.view(mx.uint32)).item()

    # Non-integer float32 activations: Rule 17(b) matches float32 reference.
    xf = (mx.random.normal((rows, k), key=mx.random.key(table_len + 11)) * 0.5).astype(
        mx.float32
    )
    (got_f,) = kern(
        inputs=[xf, codes, scales],
        template=[
            ("T", mx.float32),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", 2),
            ("RPS", 4),
        ],
        grid=(32, 2 * (rows // 2), (n + 7) // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.float32],
    )
    want_f = (xf @ w_ref.T) * scales
    max_err = mx.max(mx.abs(got_f - want_f)).item()
    assert max_err < 1e-4, max_err


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_dot_row_ops_mixed_affine_and_plugin_group_runs_same_skeleton():
    affine_face = affine.face(8, 64, True, mx.bfloat16)
    plugin_face = _plugin_dot("mixed_bf16", dtypes=(mx.bfloat16,))
    ops = (
        row_exact_skeleton.dot_row_ops(affine_face),
        row_exact_skeleton.dot_row_ops(plugin_face),
    )
    spec = row_exact_skeleton.emit_row_exact_kernel(
        ops, True, True, 2, "omlx_test_mixed_group"
    )
    assert "omlx_revision" not in spec["header"]
    assert kernels.hand_tables(spec["header"]) == ()

    mx.random.seed(7)
    k, n0, n1, rows = 256, 32, 24, 4
    w0, s0, b0 = mx.quantize(
        (mx.random.normal((n0, k)) * 0.05).astype(mx.bfloat16), group_size=64, bits=8
    )
    c1 = mx.random.randint(0, 256, (n1, k)).astype(mx.uint8)
    s1 = mx.array([2.0 ** (r % 4 - 2) for r in range(n1)], dtype=mx.float32)
    x = mx.random.randint(-4, 5, (rows, k)).astype(mx.bfloat16)
    assert row_exact_skeleton.prove_dot_face(
        plugin_face,
        k,
        n1,
        mx.bfloat16,
        ((c1.shape, c1.dtype), (s1.shape, s1.dtype)),
    )
    kern = mx.fast.metal_kernel(**spec)
    y0, y1 = kern(
        inputs=[x, w0, s0, b0, c1, s1],
        template=[
            ("T", mx.bfloat16),
            ("K_SIZE", k),
            ("N_0", n0),
            ("N_1", n1),
            ("ROWS", 2),
            ("RPS", 4),
        ],
        grid=(32, 2 * (rows // 2), (n0 + 7) // 8 + (n1 + 7) // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, n0), (rows, n1)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    want0 = mx.concatenate(
        [
            mx.quantized_matmul(
                x[r : r + 1], w0, scales=s0, biases=b0, transpose=True,
                group_size=64, bits=8,
            )
            for r in range(rows)
        ]
    )
    want1 = ((x.astype(mx.float32) @ (c1.astype(mx.float32) - 128.0).T) * s1).astype(
        mx.bfloat16
    )
    assert mx.array_equal(y0.view(mx.uint16), want0.view(mx.uint16)).item()
    assert mx.array_equal(y1.view(mx.uint16), want1.view(mx.uint16)).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    "case, mutate, expected_msg",
    [
        (
            "future_contract",
            lambda f: replace(
                f,
                source=f.source.replace(
                    "CONTRACT = 1", f"CONTRACT = {formats.CONTRACT + 1}"
                ),
            ),
            "written for a later revision of the contract",
        ),
        (
            "vpt_mismatch",
            lambda f: replace(f, vpt=4),
            "Lanes<K>::VPT must be the face's vpt",
        ),
        (
            "bad_block",
            lambda f: replace(
                f,
                source=f.source.replace("BLOCK = 32 * VPT", "BLOCK = 16 * VPT"),
            ),
            "Lanes<K>::BLOCK must be VPT inputs",
        ),
        (
            "bad_table",
            lambda f: replace(
                f,
                source=f.source.replace(
                    "const device float* scales;",
                    "const device float* scales;\n  omlx_table<float, 4> gain;",
                ),
                tables=(Table("gain", mx.float32, (1.0, 2.0)),),
            ),
            "table 'gain' is 2 values of float32",
        ),
    ],
)
def test_dot_row_ops_cold_proof_checks_contract_and_refuses_broken_faces(
    caplog, case, mutate, expected_msg
):
    bad = mutate(_plugin_dot(f"bad_{case}"))
    specs = (((16, 256), mx.uint8), ((16,), mx.float32))
    with caplog.at_level(logging.WARNING, logger="omlx.formats.row_exact_skeleton"):
        ok = row_exact_skeleton.prove_dot_face(bad, 256, 16, mx.float32, specs)
    assert ok is False
    text = "\n".join(
        r.getMessage() + ("" if not r.exc_info else str(r.exc_info[1]))
        for r in caplog.records
    )
    assert expected_msg in text
    # Second call short-circuits on the failed proof without recompiling.
    caplog.clear()
    assert row_exact_skeleton.prove_dot_face(bad, 256, 16, mx.float32, specs) is False
    assert caplog.records == []


def test_compile_kernel_takes_only_spec():
    """The unroll choice is part of the emitted spec (its name and header):
    compile_kernel cannot silently compile another variant."""
    import inspect

    assert list(inspect.signature(row_exact_skeleton.compile_kernel).parameters) == [
        "spec"
    ]


def test_row_exact_legacy_builders_deleted():
    """Rule 29: per-shape builders and legacy _Launch are deleted in the switch."""
    deleted = (
        "_TILE",
        "_BLOCK",
        "_LAST_BLOCK",
        "_SOURCE",
        "_group_source",
        "_header",
        "_kernel",
        "_group_kernel",
        "_source",
        "_launch_geometry",
        "_Launch",
        "_CALL",
        "_launch_source",
    )
    remaining = [name for name in deleted if hasattr(row_exact_qmv, name)]
    assert remaining == [], remaining


def test_row_exact_plan_pre_flattens_operands_and_chooses_index_width_at_build():
    """Rule 28: RowExactPlan resolves index width and pre-flattens operands once."""
    face = affine.face(8, 64, True, mx.bfloat16)
    w = mx.broadcast_to(mx.array(0, dtype=mx.uint32), (32, 64))
    s = mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (32, 4))
    b = mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (32, 4))
    plan32 = row_exact_skeleton.build_plan(
        (face,),
        (32,),
        256,
        4,
        mx.bfloat16,
        (2, 4, False),
        operands=(w, s, b),
        names=("weight",),
        operand_keys=(("weight", "scales", "biases"),),
    )
    assert isinstance(plan32, row_exact_skeleton.RowExactPlan)
    assert plan32.index_width is tuning.RowIndexWidth.INT32_CHECKED
    assert plan32.operands == (w, s, b)
    assert plan32.grid == (32, 4, 4)
    assert plan32.threadgroup == (32, 2, 1)
    assert "size_t(out_row)" not in plan32.spec["header"]

    plan64 = row_exact_skeleton.build_plan(
        (face,),
        (2**20,),
        4096,
        4,
        mx.bfloat16,
        (2, 4, False),
        row_bytes=4096,
    )
    assert isinstance(plan64, row_exact_skeleton.RowExactPlan)
    assert plan64.index_width is tuning.RowIndexWidth.SIZE_T64
    assert "size_t(out_row)" in plan64.spec["header"]


def test_row_exact_launch_plan_matches_eae63b6c_all_19_shapes(monkeypatch):
    """All 19 served shapes match eae63b6c byte-for-byte on spec, grid,
    threadgroup, template, input shapes and output shapes."""
    reference = json.loads(_LAUNCH_19_REFERENCE.read_text())
    assert len(reference) == 19

    captured = {}
    cur = [None]

    def fake_metal_kernel(**spec):
        def runner(**kw):
            captured[cur[0]] = {
                "spec": {
                    "name": spec["name"],
                    "input_names": list(spec["input_names"]),
                    "output_names": list(spec["output_names"]),
                    "header": spec.get("header", ""),
                    "source": spec["source"],
                },
                "grid": list(kw["grid"]),
                "threadgroup": list(kw["threadgroup"]),
                "template": [
                    [k, str(v) if isinstance(v, mx.Dtype) else v]
                    for k, v in kw.get("template", [])
                ],
                "input_shapes": [list(a.shape) for a in kw["inputs"]],
                "output_shapes": [list(s) for s in kw["output_shapes"]],
                "output_dtypes": [str(d) for d in kw["output_dtypes"]],
            }
            return [
                mx.broadcast_to(mx.array(0, dtype=d), s)
                for s, d in zip(kw["output_shapes"], kw["output_dtypes"])
            ]

        return runner

    monkeypatch.setattr(mx.fast, "metal_kernel", fake_metal_kernel)
    monkeypatch.setattr(row_exact_qmv, "unrolled_tile_verified", lambda *a, **kw: True)
    if hasattr(row_exact_skeleton, "_kernel_from_spec"):
        row_exact_skeleton._kernel_from_spec.cache_clear()
    if hasattr(row_exact_qmv, "_kernel"):
        row_exact_qmv._kernel.cache_clear()

    def mk_w(k, n, bits=8, gs=64, dtype=mx.bfloat16):
        w = mx.broadcast_to(mx.array(0, dtype=mx.uint32), (n, k * bits // 32))
        s = mx.broadcast_to(mx.array(0, dtype=dtype), (n, k // gs))
        b = mx.broadcast_to(mx.array(0, dtype=dtype), (n, k // gs))
        return w, s, b

    def mk_layer(k, n, bits=8, gs=64, dtype=mx.bfloat16):
        m = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
        nn.Module.__init__(m)
        m.bits, m.group_size, m.mode = bits, gs, "affine"
        m.weight, m.scales, m.biases = mk_w(k, n, bits, gs, dtype)
        return m

    try:
        w, s, b = mk_w(6144, 2560, 8, 64)
        p1 = row_exact_qmv.one_row_qmv(w, s, b, 8, 64, "affine", mx.bfloat16, 1)
        cur[0] = "gdn_out_m1"
        p1(mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (1, 1, 6144)))
        pm = row_exact_qmv.rows_qmv(
            w,
            s,
            b,
            8,
            64,
            "affine",
            mx.bfloat16,
            qwen35_gdn_prework._qwen4_verify_out_tile,
            unrolled=True,
        )
        for m in (2, 3, 4, 8):
            cur[0] = f"gdn_out_m{m}"
            pm(mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (1, m, 6144)))

        w, s, b = mk_w(2560, 16480, 8, 64)
        p1 = row_exact_qmv.one_row_qmv(w, s, b, 8, 64, "affine", mx.bfloat16, 2)
        cur[0] = "gdn_in_m1"
        p1(mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (1, 1, 2560)))
        pm = row_exact_qmv.rows_qmv(
            w,
            s,
            b,
            8,
            64,
            "affine",
            mx.bfloat16,
            qwen35_gdn_prework._qwen4_verify_in_tile,
            unrolled=True,
        )
        for m in (2, 3, 4):
            cur[0] = f"gdn_in_m{m}"
            pm(mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (1, m, 2560)))

        att_ls = [mk_layer(2560, n, 8, 64) for n in (12288, 512, 512, 640)]
        for m, geom in [(1, (1, 2, False)), (2, (2, 4, True)), (3, (1, 4, True))]:
            cur[0] = f"attn4_m{m}"
            row_exact_qmv.quantized_linears_tiled(
                att_ls,
                mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (1, m, 2560)),
                *geom,
            )

        o_l = [mk_layer(6144, 2560, 8, 64)]
        for m, geom in [(1, (1, 2, False)), (2, (2, 4, True))]:
            cur[0] = f"oproj_m{m}"
            row_exact_qmv.quantized_linears_tiled(
                o_l,
                mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (1, m, 6144)),
                *geom,
            )

        l4 = mk_layer(5120, 10240, 4, 64)
        for m in (2, 8):
            cur[0] = f"gdn27b_4b_m{m}"
            row_exact_qmv.quantized_linear(
                l4, mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (1, m, 5120))
            )

        qkv_ls = [mk_layer(2560, n, 8, 64) for n in (12288, 512, 512)]
        for m in (2, 8):
            cur[0] = f"qkv_ql_m{m}"
            row_exact_qmv.quantized_linears(
                qkv_ls, mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (1, m, 2560))
            )

        lh = mk_layer(2560, 151936, 8, 64)
        cur[0] = "head_8b_m5"
        row_exact_qmv.quantized_linear(
            lh, mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (1, 5, 2560))
        )
    finally:
        if hasattr(row_exact_skeleton, "_kernel_from_spec"):
            row_exact_skeleton._kernel_from_spec.cache_clear()
        if hasattr(row_exact_qmv, "_kernel"):
            row_exact_qmv._kernel.cache_clear()

    assert set(captured) == set(reference)
    mismatches = [k for k in reference if captured[k] != reference[k]]
    assert not mismatches, [
        (
            k,
            {
                f: (captured[k][f], reference[k][f])
                for f in reference[k]
                if captured[k][f] != reference[k][f]
            },
        )
        for k in mismatches[:3]
    ]
    digest = hashlib.sha256(json.dumps(captured, sort_keys=True).encode()).hexdigest()
    assert digest == A13_LAUNCH_19_DIGEST, digest


class _Bf16PluginFormat(formats.QuantFormat):
    mode = "sk_bf16_plugin"
    codes_name = "codes"
    tensors = ("group_scales",)
    contract = 1

    def __init__(self, face: Dot):
        self._face = face

    def dot(self, layer, k: int, n: int) -> Dot | None:
        return self._face

    def decode(self, tensors, bits: int, group_size: int) -> mx.array:
        return (tensors["codes"].astype(mx.float32) - 128.0) * tensors[
            "group_scales"
        ][:, None]


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@testing.verified("qwen4_exp")
def test_row_exact_supports_format_linear_plugin_face_and_mixed_group(monkeypatch):
    """Non-affine FormatLinear and a mixed affine+plugin group run through the
    same RowExactSkeleton / RowExactPlan with zero stock-only bypass."""
    plugin_face = _plugin_dot("mixed_group_layer", dtypes=(mx.bfloat16,))
    fmt = _Bf16PluginFormat(plugin_face)
    formats.load()
    monkeypatch.setitem(formats._FORMATS, fmt.mode, fmt)

    mx.random.seed(31)
    k, n_aff, n_plug, rows = 256, 32, 24, 4
    aff = nn.QuantizedLinear(k, n_aff, bias=False, group_size=64, bits=8)
    aff.weight, aff.scales, aff.biases = mx.quantize(
        (mx.random.normal((n_aff, k)) * 0.05).astype(mx.bfloat16),
        group_size=64,
        bits=8,
    )
    codes = mx.random.randint(0, 256, (n_plug, k)).astype(mx.uint8)
    scales = mx.array([2.0 ** (r % 4 - 2) for r in range(n_plug)], dtype=mx.float32)
    plug = FormatLinear(fmt, 8, 64, k, {"codes": codes, "group_scales": scales})
    x = mx.random.randint(-4, 5, (1, rows, k)).astype(mx.bfloat16)

    # Single plugin FormatLinear through quantized_linear -> RowExactPlan.
    plan_plug = row_exact_qmv._plan(plug, x, rows)
    assert isinstance(plan_plug, row_exact_skeleton.RowExactPlan)
    got_plug = row_exact_qmv.quantized_linear(plug, x)
    want_plug = mx.concatenate([plug(x[:, r : r + 1]) for r in range(rows)], axis=1)
    assert mx.array_equal(got_plug.view(mx.uint16), want_plug.view(mx.uint16)).item()

    # Mixed [affine, plugin] group in one RowExactPlan launch.
    gplan = row_exact_qmv._group_plan((aff, plug), x, rows, (2, 4, True))
    assert isinstance(gplan, row_exact_skeleton.RowExactPlan)
    assert len(gplan.faces) == 2
    y_aff, y_plug = row_exact_qmv.quantized_linears_tiled((aff, plug), x, 2, 4, True)
    want_aff = mx.concatenate([aff(x[:, r : r + 1]) for r in range(rows)], axis=1)
    assert mx.array_equal(y_aff.view(mx.uint16), want_aff.view(mx.uint16)).item()
    assert mx.array_equal(y_plug.view(mx.uint16), want_plug.view(mx.uint16)).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@testing.verified("qwen4_exp")
def test_row_exact_plan_cache_thread_safe_and_invalidates_on_weight_write(monkeypatch):
    """Rule 14 concurrency (4 threads + Barrier) and weight mutation
    invalidation on both nn.QuantizedLinear and FormatLinear."""
    plugin_face = _plugin_dot("concurrent_plan", dtypes=(mx.bfloat16,))
    fmt = _Bf16PluginFormat(plugin_face)
    formats.load()
    monkeypatch.setitem(formats._FORMATS, fmt.mode, fmt)

    mx.random.seed(41)
    k, n, rows = 256, 32, 4
    aff0 = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=8)
    aff0.weight, aff0.scales, aff0.biases = mx.quantize(
        (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16), group_size=64, bits=8
    )
    aff1 = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=8)
    aff1.weight, aff1.scales, aff1.biases = mx.quantize(
        (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16), group_size=64, bits=8
    )
    codes = mx.random.randint(0, 256, (n, k)).astype(mx.uint8)
    scales = mx.array([2.0 ** (r % 4 - 2) for r in range(n)], dtype=mx.float32)
    plug = FormatLinear(fmt, 8, 64, k, {"codes": codes, "group_scales": scales})
    x = mx.random.randint(-4, 5, (1, rows, k)).astype(mx.bfloat16)
    want0 = mx.concatenate([aff0(x[:, r : r + 1]) for r in range(rows)], axis=1)
    want1 = mx.concatenate([aff1(x[:, r : r + 1]) for r in range(rows)], axis=1)
    want_p = mx.concatenate([plug(x[:, r : r + 1]) for r in range(rows)], axis=1)
    mx.eval(
        x, want0, want1, want_p, aff0.parameters(), aff1.parameters(), plug.parameters()
    )

    gate = Barrier(4, timeout=20)

    def worker(_):
        gate.wait()
        y0, y1 = row_exact_qmv.quantized_linears((aff0, aff1), x)
        yp = row_exact_qmv.quantized_linear(plug, x)
        mx.eval(y0, y1, yp)
        return y0, y1, yp

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))
    for y0, y1, yp in results:
        assert mx.array_equal(y0.view(mx.uint16), want0.view(mx.uint16)).item()
        assert mx.array_equal(y1.view(mx.uint16), want1.view(mx.uint16)).item()
        assert mx.array_equal(yp.view(mx.uint16), want_p.view(mx.uint16)).item()

    # Mutate sibling nn.QuantizedLinear weight and FormatLinear codes: plans refresh.
    old_group = row_exact_qmv._group_plan((aff0, aff1), x, rows)
    aff1.weight, aff1.scales, aff1.biases = mx.quantize(
        (mx.random.normal((n, k), key=mx.random.key(99)) * 0.05).astype(mx.bfloat16),
        group_size=64,
        bits=8,
    )
    new_y0, new_y1 = row_exact_qmv.quantized_linears((aff0, aff1), x)
    new_group = row_exact_qmv._group_plan((aff0, aff1), x, rows)
    assert new_group is not old_group
    want1_new = mx.concatenate([aff1(x[:, r : r + 1]) for r in range(rows)], axis=1)
    assert mx.array_equal(new_y0.view(mx.uint16), want0.view(mx.uint16)).item()
    assert mx.array_equal(new_y1.view(mx.uint16), want1_new.view(mx.uint16)).item()
    assert not mx.array_equal(new_y1.view(mx.uint16), want1.view(mx.uint16)).item()

    old_plug_plan = row_exact_qmv._plan(plug, x, rows)
    plug["codes"] = mx.full(codes.shape, 132, codes.dtype)
    new_yp = row_exact_qmv.quantized_linear(plug, x)
    new_plug_plan = row_exact_qmv._plan(plug, x, rows)
    assert new_plug_plan is not old_plug_plan
    want_p_new = mx.concatenate([plug(x[:, r : r + 1]) for r in range(rows)], axis=1)
    assert mx.array_equal(new_yp.view(mx.uint16), want_p_new.view(mx.uint16)).item()
    assert not mx.array_equal(new_yp.view(mx.uint16), want_p.view(mx.uint16)).item()


def test_row_exact_plan_uses_slots_and_preflattened_checks():
    """RowExactPlan uses __slots__ and pre-flattens operand identity checks
    at construction so warm hits avoid zip/nested loops and __dict__ reads."""
    assert hasattr(row_exact_skeleton.RowExactPlan, "__slots__")
    face = affine.face(8, 64, True, mx.bfloat16)
    w, s, b = mx.quantize(
        mx.zeros((32, 256), dtype=mx.bfloat16), group_size=64, bits=8
    )
    single = row_exact_skeleton.build_plan(
        (face,),
        (32,),
        256,
        2,
        mx.bfloat16,
        (2, 4, False),
        operands=(w, s, b),
        names=("weight",),
        operand_keys=(("weight", "scales", "biases"),),
        grouped=False,
    )
    assert not hasattr(single, "__dict__")
    assert single.single_checks == (("weight", w), ("scales", s), ("biases", b))
    assert single.group_checks == (
        (0, "weight", w),
        (0, "scales", s),
        (0, "biases", b),
    )


def test_format_rows_qmv_bad_geometry_raises_value_error():
    """An unaligned geometry raises ValueError, never AttributeError on None."""
    mx.random.seed(51)
    k, n = 256, 32
    w, s, b = mx.quantize(
        (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16), group_size=64, bits=8
    )
    with testing.verified("qwen4_exp"):
        layer = FormatLinear(
            "affine", 8, 64, k, {"weight": w, "scales": s, "biases": b}
        )
    project = row_exact_qmv.format_rows_qmv(layer, mx.bfloat16, lambda rows: (4, 3))
    assert project is not None
    x = mx.zeros((1, 2, k), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="tile"):
        project(x)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_format_rows_qmv_isolates_graphs_across_output_widths_and_geometries(
    monkeypatch,
):
    """A plan's proof is keyed by its template (output width, geometry), grid and
    padding: two widths, or two geometries at one input shape, never replay a
    graph built for the other's launch."""
    mx.random.seed(57)
    k, rows = 256, 2
    x = mx.random.randint(-4, 5, (1, rows, k)).astype(mx.bfloat16)
    layers, projects = [], []
    with testing.verified("qwen4_exp"):
        for n in (32, 64):
            w, s, b = mx.quantize(
                (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16),
                group_size=64,
                bits=8,
            )
            layer = FormatLinear(
                "affine", 8, 64, k, {"weight": w, "scales": s, "biases": b}
            )
            layers.append(layer)
            projects.append(
                row_exact_qmv.format_rows_qmv(layer, mx.bfloat16, lambda _: (4, 1))
            )
    assert all(project is not None for project in projects)

    narrow, wide = projects
    got_narrow, got_wide = narrow(x), wide(x)
    assert got_narrow.shape == (1, rows, 32), got_narrow.shape
    assert got_wide.shape == (1, rows, 64), got_wide.shape
    for layer, got in zip(layers, (got_narrow, got_wide)):
        want = mx.concatenate(
            [layer(x[:, r : r + 1]) for r in range(rows)], axis=1
        )
        assert mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)).item()

    # Same width and input shape, a second geometry: its own graph, so the
    # second geometry's plan is the one traced and launched.
    monkeypatch.setattr(row_exact_qmv, "unrolled_tile_verified", lambda *a, **kw: True)
    calls = []
    traced = row_exact_skeleton.RowExactPlan.__call__

    def spy(self, value, operands=None):
        calls.append(dict(self.template)["RPS"])
        return traced(self, value, operands)

    monkeypatch.setattr(row_exact_skeleton.RowExactPlan, "__call__", spy)
    first = row_exact_qmv.format_rows_qmv(layers[0], mx.bfloat16, lambda _: (4, 2))
    second = row_exact_qmv.format_rows_qmv(layers[0], mx.bfloat16, lambda _: (2, 2))
    first(x)
    assert calls == [4], calls
    second(x)
    assert calls == [4, 2], calls


def test_clear_cached_drops_quantized_linear_row_exact_plans():
    """A stock layer's row-exact plans are a registered slot: clearing the
    module's cache drops the plans and their references to replaced weights."""
    mx.random.seed(59)
    k, n, rows = 256, 32, 2
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=8)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16), group_size=64, bits=8
    )
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    row_exact_qmv.quantized_linear(layer, x)
    assert "_omlx_row_exact_plans" in layer.__dict__
    module_cache.clear_cached(layer)
    assert "_omlx_row_exact_plans" not in layer.__dict__


def test_build_plan_rejects_direct_input_with_padding_or_group():
    """A direct-input plan launches ``x`` as it is: only an unpadded
    single-projection geometry can use it."""
    face = affine.face(8, 64, True, mx.bfloat16)
    common = dict(shapes=(32,), k=256, dtype=mx.bfloat16)
    assert (
        row_exact_skeleton.build_plan(
            (face,), rows=3, geometry=(2, 4, False), direct_input=True, **common
        )
        is None
    )
    assert (
        row_exact_skeleton.build_plan(
            (face,), rows=4, geometry=(2, 4, False), direct_input=True, grouped=True,
            **common,
        )
        is None
    )
    assert (
        row_exact_skeleton.build_plan(
            (face,), rows=4, geometry=(2, 4, False), direct_input=True, **common
        )
        is not None
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_format_rows_qmv_proof_graph_retains_no_weight_arrays(monkeypatch):
    """The graph a proof keeps is built from an operand-free plan: dropping the
    project releases the layer's tensors even though the proof outlives it."""
    monkeypatch.setattr(kernels, "_PROOFS", {})

    def helper():
        mx.random.seed(63)
        k, n = 256, 32
        w, s, b = mx.quantize(
            (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16),
            group_size=64,
            bits=8,
        )
        with testing.verified("qwen4_exp"):
            layer = FormatLinear(
                "affine", 8, 64, k, {"weight": w, "scales": s, "biases": b}
            )
        project = row_exact_qmv.format_rows_qmv(layer, mx.bfloat16, lambda _: (4, 1))
        return project, weakref.ref(layer["weight"])

    project, weight_ref = helper()
    out = project(mx.random.normal((1, 2, 256)).astype(mx.bfloat16))
    mx.eval(out)
    del out, project
    gc.collect()
    assert weight_ref() is None


def test_quantized_linears_tiled_recovers_after_bias_removed():
    """A bias-key refusal is cached with the state it was made under: removing
    the bias (even one stored as None) rebuilds the launch."""
    mx.random.seed(61)
    k, n, rows = 256, 32, 2
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=8)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16), group_size=64, bits=8
    )
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    want = mx.concatenate([layer(x[:, r : r + 1]) for r in range(rows)], axis=1)

    layer["bias"] = None
    assert row_exact_qmv.quantized_linears_tiled((layer,), x, 2, 4, True) is None
    del layer["bias"]
    (got,) = row_exact_qmv.quantized_linears_tiled((layer,), x, 2, 4, True)
    assert mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)).item()

    # A two-member group rebuilds its grouped launch after a bias is removed.
    second = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=8)
    second.weight, second.scales, second.biases = mx.quantize(
        (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16), group_size=64, bits=8
    )
    second["bias"] = None
    assert row_exact_qmv._group_plan((layer, second), x, rows) is None
    del second["bias"]
    assert row_exact_qmv._group_plan((layer, second), x, rows) is not None



def test_quantized_linears_tiled_single_layer_invalidates_on_weight_and_bias():
    """Single-member quantized_linears_tiled fast path invalidates cleanly on
    weight overwrite and bias addition."""
    mx.random.seed(53)
    k, n, rows = 256, 32, 2
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=8)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16), group_size=64, bits=8
    )
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    (y0,) = row_exact_qmv.quantized_linears_tiled((layer,), x, 2, 4, True)
    want0 = mx.concatenate([layer(x[:, r : r + 1]) for r in range(rows)], axis=1)
    assert mx.array_equal(y0.view(mx.uint16), want0.view(mx.uint16)).item()

    old_plan = row_exact_qmv._group_plan((layer,), x, rows, (2, 4, True))
    layer.weight, layer.scales, layer.biases = mx.quantize(
        (mx.random.normal((n, k), key=mx.random.key(77)) * 0.05).astype(mx.bfloat16),
        group_size=64,
        bits=8,
    )
    (y1,) = row_exact_qmv.quantized_linears_tiled((layer,), x, 2, 4, True)
    new_plan = row_exact_qmv._group_plan((layer,), x, rows, (2, 4, True))
    assert new_plan is not old_plan
    want1 = mx.concatenate([layer(x[:, r : r + 1]) for r in range(rows)], axis=1)
    assert mx.array_equal(y1.view(mx.uint16), want1.view(mx.uint16)).item()
    assert not mx.array_equal(y1.view(mx.uint16), y0.view(mx.uint16)).item()

    layer["bias"] = mx.ones((n,), dtype=mx.bfloat16)
    assert row_exact_qmv.quantized_linears_tiled((layer,), x, 2, 4, True) is None


# The verify (MMA) skeleton, its released builder recorded.

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "verify_skeleton_under_test", ROOT / "omlx" / "formats" / "verify_skeleton.py"
)
verify_skeleton = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = verify_skeleton
_spec.loader.exec_module(verify_skeleton)
RELEASE = "eae63b6c"
U34D_SHA256 = "a88dd195628a41c626754943c6245fc7cd9a4703ae61e1daa06c2f76a32b9e3a"


def _released_mma_builder():
    try:
        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{RELEASE}:omlx/patches/qwen35_verify_qmm.py",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the reference release source needs a git checkout")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_mma_kernel"
    )
    function.body = [s for s in function.body if not isinstance(s, ast.Import)]
    fake_mx = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        fast=SimpleNamespace(metal_kernel=lambda **kw: kw),
    )
    environment = {
        "_KERNEL_CACHE": {},
        "_MMA_BN": 16,
        "_MMA_BK": 32,
        "_MMA_NSG": 8,
        "mx": fake_mx,
    }
    exec(
        compile(ast.Module([function], type_ignores=[]), "<release>", "exec"),
        environment,
    )
    return environment["_build_mma_kernel"], fake_mx


def test_u34d_affine_108_mma_specs_byte_identical_to_the_release():
    builder, fake_mx = _released_mma_builder()
    digest = hashlib.sha256()
    count = 0
    for bits in (2, 3, 4, 5, 6, 8):
        for group in (32, 64, 128):
            ops = verify_skeleton.affine_mma_ops(bits, group)
            for row_tiles in (1, 2, 3):
                for dtype, tag in (
                    (fake_mx.bfloat16, "bf16"),
                    (fake_mx.float16, "fp16"),
                ):
                    name = f"omlx_vk_mma_rt{row_tiles}_k2560_q{bits}_gs{group}_{tag}"
                    got = verify_skeleton.emit_mma_kernel(ops, row_tiles, 2560, name)
                    want = builder(row_tiles, 2560, group, dtype, bits)
                    assert got == want, (bits, group, row_tiles, tag)
                    for part in (got["name"], got.get("header", ""), got["source"]):
                        digest.update(part.encode() + b"\0")
                    count += 1
    assert count == 108
    assert digest.hexdigest() == U34D_SHA256


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_the_mma_skeleton_has_no_reserved_keyword(bits):
    """8-bit once failed to compile on `int half` (Metal's half type)."""
    ops = verify_skeleton.affine_mma_ops(bits, 64)
    source = verify_skeleton.emit_mma_kernel(ops, 1, 2560, "s")["source"]
    assert "int half" not in source
    assert "for (int half" not in source


def test_mma_tile_ops_fields_are_all_used():
    import inspect

    source = inspect.getsource(verify_skeleton.emit_mma_source) + inspect.getsource(
        verify_skeleton.emit_mma_kernel
    )
    for field in verify_skeleton.MmaTileOps.__dataclass_fields__:
        assert field in source, field


def test_tile_mma_ops_keeps_the_contract_check():
    class Face:
        type = "F"

    source = verify_skeleton.emit_mma_source(
        verify_skeleton.tile_mma_ops(Face()), 1, 2560
    )
    assert "if constexpr (omlx_tile_face<F, K>::value)" in source
    assert "omlx_mma_dep<T, F>" in source


def test_tile_mma_ops_uses_emit_k_and_no_dead_n_global():
    class Face:
        type = "F"

    source = verify_skeleton.emit_mma_source(
        verify_skeleton.tile_mma_ops(Face()), 1, 5120
    )
    assert "constexpr int K = 5120;" in source
    assert "n_global" not in source


def test_emit_mma_source_does_not_rewrite_face_type():
    class Face:
        type = "omlx_foo__K__::Tile"

    source = verify_skeleton.emit_mma_source(
        verify_skeleton.tile_mma_ops(Face()), 1, 5120
    )
    assert "omlx_foo__K__::Tile" in source
    assert "constexpr int K = 5120;" in source
