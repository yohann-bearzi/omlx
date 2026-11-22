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
import hashlib
import importlib.util
import json
import logging
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx import formats
from omlx.formats import Dot, Table, affine, kernels, row_exact_skeleton, tuning

_REFERENCE = Path(__file__).parent / "data" / "parity_row_exact_a13.json"
# The combined digest of the 240 pinned specs, captured from release-3
# eae63b6c's own builders (specs/parity-skeletons-evidence/a13/test_emit.py
# proved 240/240; the serialization here is the one below).
A13_DIGEST = "fd884b476cbabb5772730282d87b344412c95c853b007f7ba1ed51d665e02f9d"


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


def test_qmv_shared_unrolls_by_pragma_not_fold_expression():
    """The shared skeleton's unrolled block is the looped one under a full
    pragma (SROA/LICM run first); the fold-expression form is gone."""
    from omlx.formats import kernels

    assert not hasattr(kernels, "_SHARED_FOLDED")
    unrolled = kernels._qmv_shared(True)
    assert '_Pragma("clang loop unroll(full)")' in unrolled
    assert "omlx_shared_folded" not in unrolled


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

    template <typename XP>
    static float load(XP x, thread float* xt) {
      copy(omlx_iota<VPT>(), x, xt);
      return 0.0f;
    }
    static float madd(float acc, Row r, thread const float* xt, float) {
      return acc + dot(omlx_iota<VPT>(), r.w, xt);
    }
    float finish(int r, float sum) const { return sum * s[r]; }

    using Decoded = Row;
    static Decoded decode(Row r) { return r; }
    template <typename A>
    static float apply(
        float acc, thread const Decoded& d, thread const float* xt, A aux) {
      return madd(acc, d, xt, aux);
    }
  };
};
}  // namespace omlx_sk_KEY
"""


def _plugin_dot(key: str, *, dtypes=(mx.float32,), vpt: int = 8) -> Dot:
    return Dot(
        key=f"sk_{key}",
        source=_PLUGIN_DOT.replace("KEY", key),
        type=f"omlx_sk_{key}::Dot",
        tensors=("group_scales",),
        dtypes=dtypes,
        vpt=vpt,
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

    template <typename XP>
    static float load(XP x, thread float* xt) {{
      for (int i = 0; i < VPT; i++) xt[i] = float(x[i]);
      return 0.0f;
    }}
    using Decoded = Row;
    static Decoded decode(Row r) {{ return r; }}
    static float apply(
        float acc, thread const Decoded& d, thread const float* xt, float) {{
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


# The verify (MMA) skeleton, its released builder recorded.

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
