# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.formats.debug (Milestone 1: layer factory & explain)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import types
from pathlib import Path

import mlx.core as mx
import pytest

from omlx import formats
from omlx.formats import (
    Dot,
    QuantFormat,
    Table,
    Tile,
    TileReads,
    affine,
    debug,
    kernels,
    testing,
    tile_dense,
    transforms,
    tuning,
)
from omlx.formats.linear import FormatEmbedding, FormatLinear


class _TabledToyFormat(QuantFormat):
    """Minimal plugin format with a small Table on its Tile face."""

    mode = "fmtdebug_tabled_toy"
    codes_name = "qcodes"
    tensors = ("qscale",)
    contract = 1

    def layout(self, k: int, n: int, bits: int, group_size: int):
        if k <= 0 or n <= 0 or bits != 4 or group_size != 64 or k % 64 != 0:
            return None
        return {
            "qcodes": ((n, k // 8), mx.uint32),
            "qscale": ((n, k // 64), None),
        }

    def tile(self, layer, k: int, n: int) -> Tile | None:
        if k % 64 != 0 or layer.bits != 4 or layer.group_size != 64:
            return None
        s_dt = layer["qscale"].dtype
        if s_dt not in (mx.bfloat16, mx.float16):
            return None
        mtype = "bfloat16_t" if s_dt == mx.bfloat16 else "float16_t"
        suffix = "bf16" if s_dt == mx.bfloat16 else "fp16"
        tbl = Table("lut", mx.float32, (0.0, 0.25, 0.5, 0.75))
        src = f"""
namespace omlx_fmtdebug_toy_{suffix} {{
struct Tile {{
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device {mtype}* scales;
  omlx_table<float, 4> lut;
  template <int K>
  struct Runs {{
    static constant constexpr int RUN = 8;
    static constant constexpr int GROUP = 64;
    struct P {{
      float s;
      omlx_table<float, 4> tbl;
    }};
    using Codes = uint32_t;
    const device uint32_t* c;
    const device {mtype}* sc;
    omlx_table<float, 4> tbl;
    Runs(Tile m, size_t row)
        : c((const device uint32_t*)(m.codes + row * (K / 2))),
          sc(m.scales + row * (K / 64)),
          tbl(m.lut) {{}}
    P params(int g) const {{ return {{float(sc[g]), tbl}}; }}
    Codes fetch(int run) const {{ return c[run]; }}
    __OMLX_READ_WEIGHT__ {{
      return p.s * p.tbl[(q >> (4 * i)) & 3u];
    }}
  }};
}};
}}
"""
        return Tile(
            key=f"fmtdebug_toy_{suffix}",
            source=src,
            type=f"omlx_fmtdebug_toy_{suffix}::Tile",
            tensors=("qscale",),
            dtypes=(s_dt,),
            tables=(tbl,),
            reads=TileReads(),
        )

    def decode(self, tensors, bits: int, group_size: int) -> mx.array:
        q = tensors["qcodes"]
        s = tensors["qscale"].astype(mx.float32)
        n = q.shape[-2]
        k = q.shape[-1] * 8
        return mx.zeros((*q.shape[:-2], n, k), dtype=mx.float32) + s[..., :1] * 0.0


class _DefectiveFaceFormat(QuantFormat):
    """Plugin format whose dot() raises to test pure defect capture."""

    mode = "fmtdebug_defective"
    codes_name = "qcodes"
    tensors = ("qscale",)
    contract = 1

    def layout(self, k: int, n: int, bits: int, group_size: int):
        if k <= 0 or n <= 0 or k % 32 != 0:
            return None
        return {"qcodes": ((n, k // 8), mx.uint32), "qscale": ((n, k // 32), None)}

    def dot(self, layer, k: int, n: int) -> Dot | None:
        raise ValueError("synthetic dot defect for explain test")

    def decode(self, tensors, bits: int, group_size: int) -> mx.array:
        q = tensors["qcodes"]
        return mx.zeros((*q.shape[:-2], q.shape[-2], q.shape[-1] * 8), dtype=mx.float32)


class _BadTransformFormat(QuantFormat):
    """Plugin format whose input_transform() returns a non-InputTransform."""

    mode = "fmtdebug_bad_tx"
    codes_name = "qcodes"
    tensors = ("qscale",)
    shared_tensors = ("qsigns",)
    contract = 1

    def layout(self, k: int, n: int, bits: int, group_size: int):
        if k <= 0 or n <= 0 or k % 32 != 0:
            return None
        return {
            "qcodes": ((n, k // 8), mx.uint32),
            "qscale": ((n, k // 32), None),
            "qsigns": ((k // 32, 32), None),
        }

    def input_transform(self, layer):
        return "not-an-input-transform"

    def decode(self, tensors, bits: int, group_size: int) -> mx.array:
        q = tensors["qcodes"]
        return mx.zeros((*q.shape[:-2], q.shape[-2], q.shape[-1] * 8), dtype=mx.float32)


_TOY_FMT = _TabledToyFormat()


def test_explain_parity_with_tile_dense_schedule_both_chips():
    rows = (2, 3, 4, 8, 16, 17, 32, 48, 64, 128, 512, 2048)
    chips = (
        ("applegpu_g15d_c80", False),
        ("applegpu_g17s_c40", True),
    )
    cases = [
        ("affine", 4, 64, 2560, 2560),
        ("affine", 8, 64, 2560, 2560),
        ("affine", 4, 64, 2560, 12288),
        ("affine", 8, 64, 2560, 12288),
        ("affine", 4, 64, 2560, 16480),
        ("affine", 8, 64, 2560, 16480),
        ("affine", 4, 64, 2560, 248320),
        ("affine", 8, 64, 2560, 248320),
        ("oqh_affine", 4, 64, 2560, 2560),
        ("oqh_affine", 8, 64, 2560, 12288),
        ("oqh", 4, 128, 2560, 2560),
        (_TOY_FMT, 4, 64, 512, 512),
    ]
    for fmt_or_mode, bits, gs, k, n in cases:
        layer = debug.make_layer(fmt_or_mode, bits=bits, group_size=gs, k=k, n=n)
        for chip, tu in chips:
            res = debug.explain(layer, rows=rows, dtype=mx.bfloat16, chip=chip)
            with testing.fresh_proofs():
                plan = tile_dense.build_plan(layer, tensor_units=tu, chip=chip)
            assert plan is not None
            for trace in res.row_traces:
                expected = tile_dense.route_for(plan, trace.rows, mx.bfloat16)
                if expected.backend == "dot":
                    if trace.rows <= 16 and any(
                        f.kind == "dot" and f.available for f in res.faces
                    ):
                        assert trace.selected_backend == "dot", (
                            fmt_or_mode,
                            bits,
                            k,
                            n,
                            chip,
                            trace.rows,
                            trace,
                        )
                    else:
                        assert trace.selected_backend == "decode"
                else:
                    assert trace.selected_backend == expected.backend, (
                        fmt_or_mode,
                        bits,
                        k,
                        n,
                        chip,
                        trace.rows,
                        trace.selected_backend,
                        expected.backend,
                    )
                    assert trace.policy_repr == repr(expected.policy)
        del layer


def test_explain_row1_m3_c7_override_3d_vs_2d():
    for n in (512, 2560, 640):
        layer = debug.make_layer("affine", bits=4, group_size=64, k=2560, n=n)
        res_3d = debug.explain(
            layer, rows=(1,), dtype=mx.bfloat16, chip="applegpu_g15d_c80", input_ndim=3
        )
        t3 = res_3d.row_traces[0]
        assert t3.selected_route == "mlx"
        assert t3.decision_source == "m3_c7_override"

        res_2d = debug.explain(
            layer, rows=(1,), dtype=mx.bfloat16, chip="applegpu_g15d_c80", input_ndim=2
        )
        t2 = res_2d.row_traces[0]
        assert t2.selected_route == "dot"
        assert t2.decision_source != "m3_c7_override"

        res_m5 = debug.explain(
            layer, rows=(1,), dtype=mx.bfloat16, chip="applegpu_g17s_c40", input_ndim=3
        )
        assert res_m5.row_traces[0].selected_route == "dot"


def test_explain_decline_reasons_and_stale_tuning_entry(tmp_path, monkeypatch):
    # 1. fp32 skips tile_dense (rows=1 stock affine falls back to mlx; rows=32 decode).
    layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    res_fp32 = debug.explain(
        layer, rows=(1, 32), dtype="fp32", chip="applegpu_g15d_c80"
    )
    assert res_fp32.row_traces[0].selected_route == "mlx"
    assert res_fp32.row_traces[1].selected_route == "decode"
    assert any(
        "fp32" in r.lower() or "dtype" in r.lower()
        for t in res_fp32.row_traces
        for r in t.decline_reasons
    )

    # 2. k=64 4b affine hits qmv_quad decline on dot face.
    layer_quad = debug.make_layer("affine", bits=4, group_size=64, k=64, n=64)
    res_quad = debug.explain(
        layer_quad, rows=(1,), dtype=mx.bfloat16, chip="applegpu_g15d_c80"
    )
    dot_face = next(f for f in res_quad.faces if f.kind == "dot")
    assert not dot_face.available
    assert dot_face.decline_reason is not None
    assert "qmv_quad" in dot_face.decline_reason
    assert res_quad.row_traces[0].selected_route == "mlx"

    # 3. Defective face captured cleanly without mutating formats._DOT_DEFECTS.
    bad_layer = debug.make_layer(
        _DefectiveFaceFormat(), bits=4, group_size=32, k=128, n=64
    )
    defects_before = set(formats._DOT_DEFECTS)
    res_bad = debug.explain(bad_layer, rows=(1,), chip="applegpu_g15d_c80")
    assert set(formats._DOT_DEFECTS) == defects_before
    bad_dot = next(f for f in res_bad.faces if f.kind == "dot")
    assert bad_dot.defective is True
    assert "synthetic dot defect" in (bad_dot.decline_reason or "")

    # 4. Stale skeleton_sha256 and face_sha256 entries reported in tuning_probes.
    chip = "applegpu_g15d_c80"
    face = affine.tile_face(4, 64, mx.bfloat16)
    entries = [
        {
            "key": {
                "route": "dense_tile",
                "face_id": face.key,
                "dtype": "bf16",
                "bits": 4,
                "group": 64,
                "n": 256,
                "k": 256,
                "rows_bucket": "25-32",
            },
            "face_sha256": tuning.face_digest(face),
            "skeleton_sha256": "000000000000",
            "measured_at": "2026-11-26T00:00:00Z",
            "params": {
                "backend": "steel",
                "bm": 32,
                "bn": 32,
                "bk": 32,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "x_nreads": 8,
            },
            "exactness": "bitwise",
        },
        {
            "key": {
                "route": "dense_tile",
                "face_id": face.key,
                "dtype": "bf16",
                "bits": 4,
                "group": 64,
                "n": 256,
                "k": 256,
                "rows_bucket": "25-32",
            },
            "face_sha256": "deadbeef0000",
            "skeleton_sha256": tuning.skeleton_digest("dense_tile"),
            "measured_at": "2026-11-26T00:00:00Z",
            "params": {
                "backend": "steel",
                "bm": 32,
                "bn": 32,
                "bk": 32,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "x_nreads": 8,
            },
            "exactness": "bitwise",
        },
    ]
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / f"{chip}.json").write_text(
        json.dumps({"schema_version": 1, "chip": chip, "entries": entries})
    )
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    tuning.reset_cache()
    try:
        res_stale = debug.explain(layer, rows=(32,), dtype=mx.bfloat16, chip=chip)
        trace = res_stale.row_traces[0]
        reasons = [r for p in trace.tuning_probes for r in p.rejected_reasons]
        assert any("skeleton_sha256" in r for r in reasons), reasons
        assert any("face_sha256" in r for r in reasons), reasons
        assert trace.decision_source == "parity_default"
    finally:
        tuning.reset_cache()


def test_explain_oqh_input_transform_and_switch_layer():
    oqh_layer = debug.make_layer("oqh", bits=4, group_size=128, k=256, n=256)
    res_oqh = debug.explain(oqh_layer, rows=(1, 32), chip="applegpu_g15d_c80")
    tx = res_oqh.input_transform
    assert tx.declared is True
    assert tx.active is True
    assert tx.kind == "signed_wht"
    assert tx.version == transforms.TRANSFORM_VERSION
    assert tx.block == 128
    assert tx.tensors == ("oqh_signs",)
    assert tx.native_fast_disabled is True
    assert tx.producer_spec is not None
    assert tx.producer_spec["gated_norm"] is True
    assert tx.producer_spec["act_pair"] is True
    assert all(t.native_fast_eligible is False for t in res_oqh.row_traces)

    # Switch layer: sorted tile vs switch_dot decline reasons.
    sw = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256, experts=8)
    res_m3 = debug.explain(
        sw, rows=(4, 16, 32), chip="applegpu_g15d_c80", sorted_indices=True
    )
    assert res_m3.experts == 8
    # rows=4 < 16 and < 4*8 -> switch_dot
    assert res_m3.row_traces[0].selected_route == "switch_dot"
    assert any(
        "16" in r or "4 * experts" in r for r in res_m3.row_traces[0].decline_reasons
    )
    # rows=16 < 4*8=32 -> switch_dot
    assert res_m3.row_traces[1].selected_route == "switch_dot"
    assert any("4 * experts" in r for r in res_m3.row_traces[1].decline_reasons)
    # rows=32 >= 4*8 -> switch_tile_steel on M3U
    assert res_m3.row_traces[2].selected_route == "switch_tile_steel"
    assert res_m3.row_traces[2].decision_source == "sorted_tile"

    # M5 Max takes switch_tile_nax for rows=32 sorted.
    res_m5 = debug.explain(
        sw, rows=(32,), chip="applegpu_g17s_c40", sorted_indices=True
    )
    assert res_m5.row_traces[0].selected_route == "switch_tile_nax"

    # sorted_indices=False declines sorted tile.
    res_unsorted = debug.explain(
        sw, rows=(32,), chip="applegpu_g15d_c80", sorted_indices=False
    )
    assert res_unsorted.row_traces[0].selected_route == "switch_dot"
    assert any(
        "sorted_indices" in r for r in res_unsorted.row_traces[0].decline_reasons
    )

    # FormatEmbedding summary and format_explain text/JSON round-trip.
    with testing.verified("qwen4_exp"):
        emb = FormatEmbedding(
            "affine",
            4,
            64,
            256,
            {
                "weight": mx.zeros((64, 32), dtype=mx.uint32),
                "scales": mx.ones((64, 4), dtype=mx.bfloat16),
                "biases": mx.zeros((64, 4), dtype=mx.bfloat16),
            },
        )
    res_emb = debug.explain(emb, rows=(1,), chip="applegpu_g15d_c80")
    assert res_emb.layer_type == "FormatEmbedding"
    assert res_emb.row_traces[0].selected_route == "embedding_lookup"
    text = debug.format_explain(res_oqh, as_json=False)
    assert "oqh" in text and "signed_wht" in text
    parsed = json.loads(debug.format_explain(res_oqh, as_json=True))
    assert parsed["mode"] == "oqh"
    assert parsed["input_transform"]["kind"] == "signed_wht"


def test_explain_is_pure_and_fail_closed(monkeypatch):
    layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    dict_before = dict(layer.__dict__)
    defects_before = set(formats._DOT_DEFECTS)
    proofs_before = dict(kernels._PROOFS)
    face_kernels_before = dict(kernels._FACE_KERNELS)

    _ = debug.explain(layer, rows=(1, 2, 16, 32, 128), chip="applegpu_g15d_c80")
    _ = debug.explain(layer, rows=(1, 2, 16, 32, 128), chip="applegpu_g17s_c40")

    assert dict(layer.__dict__) == dict_before
    assert set(formats._DOT_DEFECTS) == defects_before
    assert dict(kernels._PROOFS) == proofs_before
    assert dict(kernels._FACE_KERNELS) == face_kernels_before

    # Placeholder layer fails closed.
    with testing.verified("qwen4_exp"):
        placeholder = FormatLinear(
            "affine",
            4,
            64,
            256,
            {
                "weight": mx.zeros((64, 32), dtype=mx.uint32),
                "scales": mx.zeros((64, 4), dtype=mx.bfloat16),
                "biases": mx.zeros((64, 4), dtype=mx.bfloat16),
            },
            placeholders=True,
        )
    with pytest.raises(ValueError, match="placeholder"):
        debug.explain(placeholder, rows=(1,), chip="applegpu_g15d_c80")

    # Active formats._LOADING during make_layer fails closed.
    tok = formats._LOADING.set(object())
    try:
        with pytest.raises(RuntimeError, match="checkpoint load"):
            debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    finally:
        formats._LOADING.reset(tok)

    # Bad arguments fail closed.
    with pytest.raises((ValueError, TypeError)):
        debug.parse_dtype("float128")
    with pytest.raises((ValueError, TypeError)):
        debug.parse_dtype(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        debug.explain(layer, rows=(), chip="applegpu_g15d_c80")
    with pytest.raises(ValueError):
        debug.explain(layer, rows=(0,), chip="applegpu_g15d_c80")
    with pytest.raises((ValueError, TypeError)):
        debug.explain(layer, rows=(True,), chip="applegpu_g15d_c80")
    with pytest.raises(ValueError):
        debug.explain(layer, rows=(1,), chip="")
    with pytest.raises(ValueError):
        debug.explain(layer, rows=(1,), chip="applegpu_g15d_c80", input_ndim=4)
    with pytest.raises(ValueError):
        debug.load_format("nonexistent_mode_xyz")
    with pytest.raises(ValueError):
        debug.load_format("affine", plugin="bad_spec_no_colon")
    with pytest.raises(ValueError):
        debug.make_layer("affine", bits=4, group_size=64, k=30, n=64)

    # Malformed InputTransform propagates TypeError.
    bad_tx_layer = debug.make_layer(
        _BadTransformFormat(), bits=4, group_size=32, k=128, n=64
    )
    with pytest.raises(TypeError, match="InputTransform"):
        debug.explain(bad_tx_layer, rows=(1,), chip="applegpu_g15d_c80")

    # Plugin loading via --plugin pkg.mod:ATTR does not mutate formats._FORMATS.
    fake_mod = types.ModuleType("_fmtdebug_fake_plugin_mod")
    fake_mod.FORMAT = _TOY_FMT  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_fmtdebug_fake_plugin_mod", fake_mod)
    before_modes = set(formats._FORMATS)
    loaded = debug.load_format(
        "fmtdebug_tabled_toy", plugin="_fmtdebug_fake_plugin_mod:FORMAT"
    )
    assert loaded is _TOY_FMT
    assert set(formats._FORMATS) == before_modes
    assert "fmtdebug_tabled_toy" not in formats._FORMATS


def test_rule28_no_hot_path_import_and_rule30_no_env_vars():
    # Rule 28: importing runtime omlx.formats modules never imports debug.
    code = (
        "import sys; "
        "import omlx.formats, omlx.formats.linear, omlx.formats.tile_dense, "
        "omlx.formats.kernels; "
        "assert 'omlx.formats.debug' not in sys.modules; "
        "assert 'omlx.formats.__main__' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    # Rule 30: zero os.environ / os.getenv in debug.py.
    debug_path = Path(debug.__file__)
    tree = ast.parse(debug_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
            pytest.fail(
                f"Rule 30 violation: .{node.attr} in {debug_path}:{node.lineno}"
            )
        if isinstance(node, ast.Name) and node.id in ("environ", "getenv"):
            pytest.fail(f"Rule 30 violation: {node.id} in {debug_path}:{node.lineno}")
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in ("environ", "getenv"):
                    pytest.fail(
                        f"Rule 30 violation: import {alias.name} in "
                        f"{debug_path}:{node.lineno}"
                    )
