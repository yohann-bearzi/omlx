# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.formats.debug (Milestone 1: layer factory & explain)."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
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
    linear,
    testing,
    tile_dense,
    tile_gather,
    tile_wide,
    transforms,
    tuning,
)
from omlx.formats.linear import FormatEmbedding, FormatLinear
from omlx.patches import m5_gather_qmm_nax as nax


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
    assert res_fp32.schedule_summary == ()
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


# ---------------------------------------------------------------------------
# Milestone 2 — dump_metal tests
# ---------------------------------------------------------------------------


class _StagedTabledToyFormat(QuantFormat):
    """Plugin format with a 16-entry Table (always staged on M3 and M5)."""

    mode = "fmtdebug_staged_toy"
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
        tbl = Table("lut16", mx.float32, tuple(float(i) * 0.125 for i in range(16)))
        src = f"""
namespace omlx_fmtdebug_staged_{suffix} {{
struct Tile {{
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device {mtype}* scales;
  omlx_table<float, 16> lut16;
  template <int K>
  struct Runs {{
    static constant constexpr int RUN = 8;
    static constant constexpr int GROUP = 64;
    struct P {{
      float s;
      omlx_table<float, 16> tbl;
    }};
    using Codes = uint32_t;
    const device uint32_t* c;
    const device {mtype}* sc;
    omlx_table<float, 16> tbl;
    Runs(Tile m, size_t row)
        : c((const device uint32_t*)(m.codes + row * (K / 2))),
          sc(m.scales + row * (K / 64)),
          tbl(m.lut16) {{}}
    P params(int g) const {{ return {{float(sc[g]), tbl}}; }}
    Codes fetch(int run) const {{ return c[run]; }}
    __OMLX_READ_WEIGHT__ {{
      return p.s * p.tbl[(q >> (4 * i)) & 15u];
    }}
  }};
}};
}}
"""
        return Tile(
            key=f"fmtdebug_staged_{suffix}",
            source=src,
            type=f"omlx_fmtdebug_staged_{suffix}::Tile",
            tensors=("qscale",),
            dtypes=(s_dt,),
            tables=(tbl,),
            reads=TileReads(),
        )

    def decode(self, tensors, bits: int, group_size: int) -> mx.array:
        q = tensors["qcodes"]
        return mx.zeros((*q.shape[:-2], q.shape[-2], q.shape[-1] * 8), dtype=mx.float32)


_STAGED_TOY_FMT = _StagedTabledToyFormat()


def test_dump_metal_byte_identical_to_live_emitters(monkeypatch):
    captured: list[dict] = []
    real_mk = mx.fast.metal_kernel

    def spy_mk(**kwargs):
        captured.append(dict(kwargs))
        return real_mk(**kwargs)

    dense = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    switch = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256, experts=4)
    staged_dense = debug.make_layer(
        _STAGED_TOY_FMT, bits=4, group_size=64, k=256, n=256
    )
    staged_switch = debug.make_layer(
        _STAGED_TOY_FMT, bits=4, group_size=64, k=256, n=256, experts=4
    )

    dot_face = formats._checked_face(dense.fmt, dense, 256, 256, Dot, "dot")
    tile_face = formats._checked_face(dense.fmt, dense, 256, 256, Tile, "tile")
    sw_dot = formats._checked_face(switch.fmt, switch, 256, 256, Dot, "dot")
    sw_tile = formats._checked_face(switch.fmt, switch, 256, 256, Tile, "tile")
    st_tile = formats._checked_face(
        staged_dense.fmt, staged_dense, 256, 256, Tile, "tile"
    )
    st_sw_tile = formats._checked_face(
        staged_switch.fmt, staged_switch, 256, 256, Tile, "tile"
    )
    assert all(
        f is not None
        for f in (dot_face, tile_face, sw_dot, sw_tile, st_tile, st_sw_tile)
    )

    # Clear live caches inside monkeypatch scope so spy_mk captures every route.
    monkeypatch.setattr(mx.fast, "metal_kernel", spy_mk)
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})
    monkeypatch.setattr(nax, "_kernels", {})

    def _capture(fn):
        captured.clear()
        fn()
        assert len(captured) == 1, f"expected 1 metal_kernel call, got {len(captured)}"
        return captured[0]

    # 1. dot
    linear._kernel.cache_clear()
    cap_dot = _capture(lambda: linear._kernel(dot_face, False))
    res_dot = debug.dump_metal(dense, rows=1, route="dot", chip="applegpu_g15d_c80")

    # 2. dot_shared
    linear._shared_kernel.cache_clear()
    cap_sh = _capture(lambda: linear._shared_kernel(dot_face))
    res_sh = debug.dump_metal(
        dense, rows=2, route="dot_shared", chip="applegpu_g15d_c80"
    )

    # 3. steel
    tile_dense._build.cache_clear()
    cap_steel = _capture(lambda: tile_dense._build(tile_face, "steel"))
    res_steel = debug.dump_metal(
        dense, rows=32, route="steel", chip="applegpu_g15d_c80"
    )

    # 4. nax (affine)
    tile_dense._build.cache_clear()
    cap_nax = _capture(lambda: tile_dense._build(tile_face, "nax"))
    res_nax = debug.dump_metal(dense, rows=32, route="nax", chip="applegpu_g17s_c40")

    # 4b. nax with 16-entry staged Table (verifies PAD subtraction!)
    tile_dense._build.cache_clear()
    cap_nax_tbl = _capture(lambda: tile_dense._build(st_tile, "nax"))
    res_nax_tbl = debug.dump_metal(
        staged_dense, rows=32, route="nax", chip="applegpu_g17s_c40"
    )
    assert res_nax_tbl.launch.staged_table_bytes == 64
    assert "(PAD > 64 ? PAD - 64 : 0) / sizeof(T)];" in res_nax_tbl.raw_source

    # 5. wide
    cap_wide = _capture(lambda: tile_wide.kernel(tile_face))
    res_wide = debug.dump_metal(dense, rows=4, route="wide", chip="applegpu_g15d_c80")

    # 6. mma
    mma_pol = tuning.MmaTilePolicy()
    cap_mma = _capture(
        lambda: tile_dense._mma_msg_spec(tile_face, "mma", 256, 256, 8, mma_pol)
    )
    res_mma = debug.dump_metal(dense, rows=4, route="mma", chip="applegpu_g15d_c80")

    # 7. msg
    msg_pol = tuning.MsgTilePolicy()
    cap_msg = _capture(
        lambda: tile_dense._mma_msg_spec(tile_face, "msg", 256, 256, 4, msg_pol)
    )
    res_msg = debug.dump_metal(dense, rows=4, route="msg", chip="applegpu_g15d_c80")

    # 8. switch_dot
    linear._kernel.cache_clear()
    cap_sw_dot = _capture(lambda: linear._kernel(sw_dot, True))
    res_sw_dot = debug.dump_metal(
        switch, rows=4, route="switch_dot", chip="applegpu_g15d_c80"
    )

    # 9. switch_tile_steel
    tile_gather._build.cache_clear()
    cap_sw_steel = _capture(lambda: tile_gather._build(sw_tile))
    res_sw_steel = debug.dump_metal(
        switch, rows=32, route="switch_tile_steel", chip="applegpu_g15d_c80"
    )

    # 10. switch_tile_nax (affine + staged Table PAD subtraction)
    cap_sw_nax = _capture(lambda: nax._face_kernel(sw_tile, "plain"))
    res_sw_nax = debug.dump_metal(
        switch, rows=32, route="switch_tile_nax", chip="applegpu_g17s_c40"
    )
    cap_sw_nax_tbl = _capture(lambda: nax._face_kernel(st_sw_tile, "plain"))
    res_sw_nax_tbl = debug.dump_metal(
        staged_switch, rows=32, route="switch_tile_nax", chip="applegpu_g17s_c40"
    )
    assert "(PAD > 64 ? PAD - 64 : 0) / sizeof(T)];" in res_sw_nax_tbl.raw_source

    pairs = [
        ("dot", res_dot, cap_dot),
        ("dot_shared", res_sh, cap_sh),
        ("steel", res_steel, cap_steel),
        ("nax", res_nax, cap_nax),
        ("nax_tabled", res_nax_tbl, cap_nax_tbl),
        ("wide", res_wide, cap_wide),
        ("mma", res_mma, cap_mma),
        ("msg", res_msg, cap_msg),
        ("switch_dot", res_sw_dot, cap_sw_dot),
        ("switch_tile_steel", res_sw_steel, cap_sw_steel),
        ("switch_tile_nax", res_sw_nax, cap_sw_nax),
        ("switch_tile_nax_tabled", res_sw_nax_tbl, cap_sw_nax_tbl),
    ]
    for label, res, cap in pairs:
        assert res.kernel_name == cap["name"], label
        assert res.raw_header == cap["header"], label
        assert res.raw_source == cap["source"], label
        assert res.launch.input_names == tuple(cap["input_names"]), label
        assert res.launch.output_names == tuple(cap["output_names"]), label
        expected_raw = hashlib.sha256(
            f"{res.kernel_name}\0{res.raw_header}\0{res.raw_source}".encode()
        ).hexdigest()
        assert res.raw_digest == expected_raw, label

    # Cross-chip small-table selects vs staged (without mutating globals).
    small_layer = debug.make_layer(_TOY_FMT, bits=4, group_size=64, k=256, n=256)
    res_m3_small = debug.dump_metal(
        small_layer, rows=32, route="steel", chip="applegpu_g15d_c80"
    )
    res_m5_small = debug.dump_metal(
        small_layer, rows=32, route="nax", chip="applegpu_g17s_c40"
    )
    assert res_m3_small.table_mode == "selects"
    assert res_m3_small.launch.staged_table_bytes == 0
    assert "SELECTS" not in res_m3_small.raw_header
    assert "bool VALUES = (N <= 4 && true)" in res_m3_small.raw_header
    assert res_m5_small.table_mode == "staged"
    assert res_m5_small.launch.staged_table_bytes == 16
    assert "bool VALUES = (N <= 4 && false)" in res_m5_small.raw_header


def test_dump_metal_wide_template_matches_live_layout():
    dense = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    for rows in (4, 5, 7, 9):
        dumped = debug.dump_metal(
            dense, rows=rows, route="wide", chip="applegpu_g15d_c80"
        )
        expected = tuple(
            (tn, debug._template_record_value(tv))
            for tn, tv in tile_wide.template(
                mx.bfloat16, 256, 256, *tile_wide.layout(rows)
            )
        )
        assert dumped.launch.template == expected, f"rows={rows}"


def test_dump_metal_section_markers_and_deterministic_digests():
    dense = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    digest_before = tile_dense.source_digest()
    d1 = debug.dump_metal(dense, rows=32, route="steel", chip="applegpu_g15d_c80")
    d2 = debug.dump_metal(dense, rows=32, route="steel", chip="applegpu_g15d_c80")
    assert d1.raw_digest == d2.raw_digest
    assert d1.metal_digest == d2.metal_digest
    assert len(d1.raw_digest) == 64
    assert len(d1.metal_digest) == 64

    lines = d1.metal.splitlines()
    sec_names = [s.name for s in d1.sections]
    for expected in (
        "mlx_preamble",
        "mlx_route_headers",
        "face",
        "skeleton_header",
        "kernel_entry",
        "matrix_declarations",
        "skeleton_body",
    ):
        assert expected in sec_names, sec_names

    for sec in d1.sections:
        assert lines[sec.start_line - 1] == f"// >>> OMLX_SECTION: {sec.name} >>>"
        assert lines[sec.end_line - 1] == f"// <<< OMLX_SECTION: {sec.name} <<<"
        body = "\n".join(lines[sec.start_line : sec.end_line - 1])
        assert sec.sha256 == hashlib.sha256(body.encode()).hexdigest()

    # Check template formatting puts bool as true/false (not 1/0).
    assert "true" in d1.metal or "false" in d1.metal
    assert "w_weight_shape" in d1.metal
    assert "const constant int* w_weight_shape" in d1.metal

    # Summary formatting (text + JSON).
    txt = debug.format_dump_metal_summary(d1, as_json=False)
    assert d1.kernel_name in txt and d1.raw_digest in txt
    parsed = json.loads(debug.format_dump_metal_summary(d1, as_json=True))
    assert parsed["raw_digest"] == d1.raw_digest
    assert parsed["metal_digest"] == d1.metal_digest

    # Rule 20: dump_metal leaves the pinned dense skeleton source unchanged.
    from omlx.patches import qwen35_moe_routed_decode as routed
    from tests.test_formats_tile_dense_parity import _DENSE_DIGEST

    routed._ensure_tuning_registered()
    assert tile_dense.source_digest() == digest_before == _DENSE_DIGEST
    assert tuning.stale_entries() == []


def _has_xcrun_metal() -> bool:
    if shutil.which("xcrun") is None:
        return False
    proc = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


@pytest.mark.skipif(
    not _has_xcrun_metal(),
    reason="requires xcrun -sdk macosx metal toolchain",
)
def test_dump_metal_standalone_compiles_with_xcrun_metal(tmp_path):
    dense = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    switch = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256, experts=4)
    cases = [
        ("dot", dense, 1),
        ("dot_shared", dense, 2),
        ("steel", dense, 32),
        ("wide", dense, 4),
        ("wide", dense, 5),
        ("wide", dense, 7),
        ("mma", dense, 4),
        ("msg", dense, 4),
        ("switch_dot", switch, 4),
        ("switch_tile_steel", switch, 32),
    ]
    for route, layer, rows in cases:
        dumped = debug.dump_metal(
            layer, rows=rows, route=route, chip="applegpu_g15d_c80"
        )
        src_file = tmp_path / f"{route}_{rows}.metal"
        src_file.write_text(dumped.metal, encoding="utf-8")
        proc = subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-x",
                "metal",
                "-std=metal4.0",
                "-c",
                str(src_file),
                "-o",
                "/dev/null",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert (
            proc.returncode == 0
        ), f"xcrun metal failed for route={route} rows={rows}:\n{proc.stderr}"


def test_dump_metal_fail_closed_and_pure():
    dense = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    switch = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256, experts=4)

    # Purity check: dump_metal must not touch layer.__dict__, _FACE_KERNELS, _PROOFS.
    dict_before = dict(dense.__dict__)
    fk_before = dict(kernels._FACE_KERNELS)
    proofs_before = dict(kernels._PROOFS)
    _ = debug.dump_metal(dense, rows=32, route="steel", chip="applegpu_g15d_c80")
    assert dict(dense.__dict__) == dict_before
    assert dict(kernels._FACE_KERNELS) == fk_before
    assert dict(kernels._PROOFS) == proofs_before

    # Auto-selected mlx route (M3 C7 row 1 3-D) refuses without --route override.
    c7_layer = debug.make_layer("affine", bits=4, group_size=64, k=2560, n=2560)
    with pytest.raises(ValueError, match="mlx"):
        debug.dump_metal(c7_layer, rows=1, chip="applegpu_g15d_c80", input_ndim=3)

    # Auto-selected decode route (fp32 rows=32) refuses without --route override.
    with pytest.raises(ValueError, match="decode"):
        debug.dump_metal(dense, rows=32, dtype="fp32", chip="applegpu_g15d_c80")

    # Unknown route refuses.
    with pytest.raises(ValueError, match="unknown"):
        debug.dump_metal(dense, rows=1, route="not_a_route", chip="applegpu_g15d_c80")

    # Illegal forced routes refuse.
    ragged = debug.make_layer("affine", bits=4, group_size=64, k=256, n=260)
    with pytest.raises(ValueError):
        debug.dump_metal(ragged, rows=1, route="dot", chip="applegpu_g15d_c80")
    with pytest.raises(ValueError):
        debug.dump_metal(dense, rows=32, route="nax", chip="applegpu_g15d_c80")
    with pytest.raises(ValueError):
        debug.dump_metal(dense, rows=1, route="switch_dot", chip="applegpu_g15d_c80")
    with pytest.raises(ValueError):
        debug.dump_metal(switch, rows=32, route="steel", chip="applegpu_g15d_c80")
    with pytest.raises(ValueError):
        debug.dump_metal(dense, rows=0, route="dot", chip="applegpu_g15d_c80")
    with pytest.raises(ValueError):
        debug.dump_metal(switch, rows=1, per_row=0, route="switch_dot")

    # Placeholder layer refuses.
    with testing.verified("qwen4_exp"):
        ph = FormatLinear(
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
        debug.dump_metal(ph, rows=1, route="dot", chip="applegpu_g15d_c80")


# ---------------------------------------------------------------------------
# Milestone 3 Commit 3 — bench (Rule-31 chained + C2 fp32_ref + memory delta)
# ---------------------------------------------------------------------------


class _Group16ToyFormat(QuantFormat):
    """Toy format with non-MLX group_size=16 to test baseline fallback."""

    mode = "fmtdebug_g16_toy"
    codes_name = "qcodes"
    tensors = ("qscale",)
    contract = 1

    def layout(self, k: int, n: int, bits: int, group_size: int):
        if k <= 0 or n <= 0 or bits != 4 or group_size != 16 or k % 16 != 0:
            return None
        return {
            "qcodes": ((n, k // 8), mx.uint32),
            "qscale": ((n, k // 16), None),
        }

    def decode(self, tensors, bits: int, group_size: int) -> mx.array:
        q = tensors["qcodes"]
        s = tensors["qscale"].astype(mx.float32)
        n = q.shape[-2]
        words = q.shape[-1]
        k = words * 8
        # Unpack 8 nibbles per uint32 word, centered around 7.5.
        shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
        mask = mx.array(0xF, dtype=mx.uint32)
        nibbles = ((q[..., None] >> shifts) & mask).astype(mx.float32)
        vals = (nibbles.reshape(*q.shape[:-2], n, k) - 7.5) / 7.5
        scales_exp = mx.repeat(s, 16, axis=-1)
        return vals * scales_exp


def test_bench_math_helpers_cpu():
    # Exact balanced pair geometric mean and position factor.
    base = [10.0, 20.0, 12.0, 24.0]
    new = [5.0, 10.0, 6.0, 12.0]
    first = [True, True, False, False]
    ratios, positions = debug._both_orders(base, new, first, units=2)
    assert len(ratios) == 2
    assert pytest.approx(ratios[0], rel=1e-9) == 0.5
    assert pytest.approx(ratios[1], rel=1e-9) == 0.5
    assert pytest.approx(positions[0], rel=1e-9) == 1.0
    assert pytest.approx(positions[1], rel=1e-9) == 1.0

    # Same-order pair must fail closed with ValueError.
    with pytest.raises(ValueError, match="same order"):
        debug._both_orders([10.0, 10.0], [5.0, 5.0], [True, True], units=1)

    # _rel_l2 exact and zero-ref safe.
    ref = mx.array([3.0, 4.0], dtype=mx.float32)
    got = mx.array([3.0, 5.0], dtype=mx.float32)
    assert pytest.approx(debug._rel_l2(got, ref), rel=1e-6) == 0.2
    assert (
        debug._rel_l2(
            mx.zeros((4,), dtype=mx.float32), mx.zeros((4,), dtype=mx.float32)
        )
        == 0.0
    )

    # C2 threshold boundary: 1.05 * base + 1e-6, plus NaN fails closed.
    base_y = mx.array([3.0, 4.1], dtype=mx.float32)  # rel_l2 = 0.02
    good_y = mx.array([3.0, 4.104], dtype=mx.float32)  # rel_l2 = 0.0208 <= 0.021001
    bad_y = mx.array([3.0, 4.2], dtype=mx.float32)  # rel_l2 = 0.04 > 0.021001
    nan_y = mx.array([float("nan"), 4.0], dtype=mx.float32)

    fin, rl, _, rb, thr, passed = debug._c2_check(good_y, base_y, ref)
    assert fin is True
    assert rb is not None and thr is not None
    assert pytest.approx(thr, rel=1e-6) == 1.05 * rb + 1e-6
    assert rl <= thr and passed is True

    _, _, _, _, _, passed_bad = debug._c2_check(bad_y, base_y, ref)
    assert passed_bad is False

    fin_nan, _, _, _, _, passed_nan = debug._c2_check(nan_y, base_y, ref)
    assert fin_nan is False
    assert passed_nan is False


def test_bench_fail_closed_validation(monkeypatch):
    l1 = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    l2 = debug.make_layer("affine", bits=4, group_size=64, k=256, n=512)

    with pytest.raises(ValueError):
        debug.bench([], rows=1, rounds=4)
    with pytest.raises(ValueError):
        debug.bench([l1, l2], rows=1, rounds=4)
    with pytest.raises(ValueError):
        debug.bench(l1, rows=1, rounds=3)
    with pytest.raises(ValueError):
        debug.bench(l1, rows=1, rounds=0)
    with pytest.raises(ValueError):
        debug.bench(l1, rows=1, chain_len=0, rounds=4)
    with pytest.raises(ValueError):
        debug.bench(l1, rows=0, rounds=4)
    with pytest.raises(ValueError):
        debug.bench(l1, rows=1, warmup=-1, rounds=4)
    with pytest.raises(ValueError):
        debug.bench(l1, rows=1, input_ndim=4, rounds=4)

    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Metal"):
        debug.bench(l1, rows=1, rounds=4)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal GPU")
def test_bench_gpu_affine_square_and_rectangular():
    shapes = [
        (256, 256, 1),
        (256, 256, 64),
        (256, 512, 4),  # n > k (slice in fit_chain)
        (256, 128, 4),  # n < k (repeat-concat in fit_chain)
    ]
    for k, n, rows in shapes:
        layers = [
            debug.make_layer(
                "affine",
                bits=4,
                group_size=64,
                k=k,
                n=n,
                seed=i,
                realistic_weights=True,
            )
            for i in range(2)
        ]
        res = debug.bench(
            layers, rows=rows, dtype=mx.bfloat16, chain_len=4, rounds=4, warmup=1
        )
        assert res.c2_pass is True, (k, n, rows, res)
        assert res.layer_finite is True
        assert res.layer_median_us > 0
        assert res.baseline_available is True
        assert res.baseline_median_us is not None and res.baseline_median_us > 0
        assert res.paired_balanced_ratio is not None and res.paired_balanced_ratio > 0
        assert res.weight_bytes_delta == 0
        assert res.peak_memory_layer_bytes >= 0
        parsed = json.loads(debug.format_bench(res, as_json=True))
        assert parsed["c2_pass"] is True
        assert parsed["k"] == k and parsed["n"] == n
        assert "c2_pass" in debug.format_bench(res, as_json=False).lower() or (
            "PASS" in debug.format_bench(res, as_json=False)
        )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal GPU")
def test_bench_gpu_oqh_transform_and_non_mlx_group_size():
    # oqh with InputTransform active: C2 reference must apply inverse_weight!
    oqh_layers = [
        debug.make_layer(
            "oqh",
            bits=4,
            group_size=128,
            k=256,
            n=256,
            seed=10 + i,
            realistic_weights=True,
        )
        for i in range(2)
    ]
    res_oqh = debug.bench(
        oqh_layers, rows=4, dtype=mx.bfloat16, chain_len=4, rounds=4, warmup=1
    )
    assert res_oqh.layer_finite is True
    assert res_oqh.layer_rel_l2_fp32 < 0.25, res_oqh
    assert res_oqh.baseline_available is True
    assert res_oqh.c2_pass is True, res_oqh

    # Toy plugin with non-MLX group_size=16 -> fallback baseline_group_size=64 + note.
    g16_fmt = _Group16ToyFormat()
    g16_layers = [
        debug.make_layer(
            g16_fmt,
            bits=4,
            group_size=16,
            k=256,
            n=128,
            seed=20 + i,
            realistic_weights=True,
        )
        for i in range(2)
    ]
    res_g16 = debug.bench(
        g16_layers, rows=4, dtype=mx.bfloat16, chain_len=4, rounds=4, warmup=1
    )
    assert res_g16.layer_finite is True
    assert res_g16.baseline_available is True
    assert res_g16.baseline_group_size == 64
    assert res_g16.baseline_note is not None and "64" in res_g16.baseline_note


# ---------------------------------------------------------------------------
# Milestone 3 Commit 4 — warm_report tests
# ---------------------------------------------------------------------------


class _SmallWarmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
        self.l1 = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
        self.l2 = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
        self.l3 = debug.make_layer("affine", bits=4, group_size=64, k=256, n=512)
        self.sw0 = debug.make_layer(
            "affine", bits=4, group_size=64, k=256, n=256, experts=4
        )
        self.sw1 = debug.make_layer(
            "affine", bits=4, group_size=64, k=256, n=256, experts=4
        )


def test_warm_report_dry_run_dedup_dense_and_switch_model():
    model = _SmallWarmModel()
    rep = debug.warm_report(model, dtypes=(mx.bfloat16,), compile=False)
    assert rep.compile_executed is False
    assert rep.compile_ms == 0.0
    assert rep.compiled_now == 0
    assert rep.layers_scanned == 6
    assert rep.dense_layers == 4
    assert rep.switch_layers == 2
    assert rep.unique_pipelines > 0
    assert rep.total_requested > rep.unique_pipelines
    assert rep.deduped_count == rep.total_requested - rep.unique_pipelines
    assert rep.deduped_count > 0
    assert rep.dedup_ratio > 0.0
    for pipe in rep.pipelines:
        assert pipe.kernel_name.startswith("omlx_"), pipe

    # Verify proof_tasks deduplication across two identical switch layers.
    sw_model = nn.Module()
    sw_model.s0 = model.sw0
    sw_model.s1 = model.sw1
    sw_rep = debug.warm_report(sw_model, dtypes=(mx.bfloat16,), compile=False)
    proof_pipes = [p for p in sw_rep.pipelines if p.kind == "proof_task"]
    assert proof_pipes, sw_rep.pipelines
    assert all(p.requested_count == 2 for p in proof_pipes), proof_pipes
    assert sw_rep.deduped_count >= len(proof_pipes)

    # Formatting text & JSON.
    txt = debug.format_warm_report(rep, as_json=False)
    assert "unique_pipelines" in txt or "Pipelines" in txt
    parsed = json.loads(debug.format_warm_report(rep, as_json=True))
    assert parsed["layers_scanned"] == 6
    assert parsed["deduped_count"] == rep.deduped_count


def test_warm_report_compile_failure_awaits_all_futures(monkeypatch):
    awaited: list[str] = []

    class _TrackingFuture:
        def __init__(self, label: str, fail: bool = False):
            self.label = label
            self.fail = fail

        def result(self, timeout=None):
            awaited.append(self.label)
            if self.fail:
                raise RuntimeError(f"synthetic build error on {self.label}")
            return None

    def fake_warm(instances):
        return (
            _TrackingFuture("first", fail=True),
            _TrackingFuture("second", fail=False),
        )

    monkeypatch.setattr(kernels, "warm", fake_warm)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    with pytest.raises(RuntimeError, match="synthetic build error on first"):
        debug.warm_report(layer, dtypes=(mx.bfloat16,), compile=True)
    assert awaited == ["first", "second"]


def test_warm_report_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match="no FormatLinear"):
        debug.warm_report([], compile=False)

    plain_model = nn.Module()
    plain_model.fc = nn.Linear(64, 64)
    with pytest.raises(ValueError, match="no FormatLinear"):
        debug.warm_report(plain_model, compile=False)

    with testing.verified("qwen4_exp"):
        ph = FormatLinear(
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
        debug.warm_report(ph, compile=False)

    layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    with pytest.raises(ValueError):
        debug.warm_report(layer, dtypes=("fp32",), compile=False)
    with pytest.raises(ValueError):
        debug.warm_report(layer, per_rows=(0,), compile=False)

    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Metal"):
        debug.warm_report(layer, compile=True)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal GPU")
def test_warm_report_gpu_compiles_once_and_zero_runtime_proofs():
    # Use a distinct shape (k=384, n=384) to exercise real Metal compilation.
    layers = [
        debug.make_layer("affine", bits=4, group_size=64, k=384, n=384, seed=i)
        for i in range(2)
    ]

    # 1. isolated_proofs=True isolates global kernels._PROOFS while warming Metal.
    proofs_before = dict(kernels._PROOFS)
    rep_iso = debug.warm_report(
        layers, dtypes=(mx.bfloat16,), compile=True, isolated_proofs=True
    )
    assert dict(kernels._PROOFS) == proofs_before
    assert rep_iso.compile_executed is True
    assert rep_iso.unique_pipelines > 0
    assert rep_iso.compile_ms >= 0.0
    assert rep_iso.runtime_proof_delta == 0

    rep_iso2 = debug.warm_report(
        layers, dtypes=(mx.bfloat16,), compile=True, isolated_proofs=True
    )
    assert rep_iso2.already_built == rep_iso2.unique_pipelines
    assert rep_iso2.compiled_now == 0

    # 2. Within fresh_proofs, warm_report populates Proof.ran so forward has 0 proofs.
    with testing.fresh_proofs():
        rep1 = debug.warm_report(
            layers, dtypes=(mx.bfloat16,), compile=True, isolated_proofs=False
        )
        assert rep1.unique_pipelines > 0
        assert rep1.runtime_proof_delta == 0
        rep2 = debug.warm_report(
            layers, dtypes=(mx.bfloat16,), compile=True, isolated_proofs=False
        )
        assert rep2.already_built == rep2.unique_pipelines
        assert rep2.compiled_now == 0

        cnt_before = kernels.proof_count()
        x = mx.zeros((1, 1, 384), dtype=mx.bfloat16)
        y = layers[0](x)
        mx.eval(y)
        assert kernels.proof_count() == cnt_before


# ---------------------------------------------------------------------------
# Milestone 4 Commit 5 — python -m omlx.formats CLI tests
# ---------------------------------------------------------------------------


def _cli():
    from omlx.formats import __main__ as cli

    return cli


def test_cli_help_lists_all_five_subcommands(capsys):
    cli = _cli()
    rc = cli.main(["--help"])
    assert rc == 0
    out = capsys.readouterr().out
    for name in ("explain", "dump-metal", "bench", "warm-report", "audit"):
        assert name in out


def test_cli_explain_and_dump_metal_and_warm_report_dry_run_json(
    tmp_path, capsys, monkeypatch
):
    cli = _cli()

    # Validation errors exit 2 with a message on stderr.
    rc = cli.main(
        [
            "explain",
            "--mode",
            "affine",
            "--bits",
            "4",
            "--group-size",
            "64",
            "--k",
            "256",
            "--n",
            "256",
            "--rows",
            "0",
            "--json",
        ]
    )
    assert rc == 2
    assert "python -m omlx.formats" in capsys.readouterr().err

    # Runtime errors exit 1.
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic runtime failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(debug, "bench", boom)
        rc = cli.main(
            [
                "bench",
                "--mode",
                "affine",
                "--bits",
                "4",
                "--group-size",
                "64",
                "--k",
                "256",
                "--n",
                "256",
            ]
        )
        assert rc == 1
    assert "synthetic runtime failure" in capsys.readouterr().err

    rc = cli.main(
        [
            "explain",
            "--mode",
            "affine",
            "--bits",
            "4",
            "--group-size",
            "64",
            "--k",
            "256",
            "--n",
            "256",
            "--rows",
            "1,32",
            "--json",
        ]
    )
    assert rc == 0
    explained = json.loads(capsys.readouterr().out)
    assert explained["mode"] == "affine"
    assert [row["rows"] for row in explained["row_traces"]] == [1, 32]

    out_path = tmp_path / "affine-dot.metal"
    rc = cli.main(
        [
            "dump-metal",
            "--mode",
            "affine",
            "--bits",
            "4",
            "--group-size",
            "64",
            "--k",
            "256",
            "--n",
            "256",
            "--rows",
            "1",
            "--route",
            "dot",
            "--out",
            str(out_path),
            "--json",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["kernel_name"].startswith("omlx_")
    assert len(summary["raw_digest"]) == 64
    dumped = out_path.read_text(encoding="utf-8")
    assert "// >>> OMLX_SECTION: kernel_entry >>>" in dumped

    # Without --out or --json the raw .metal source goes to stdout.
    rc = cli.main(
        [
            "dump-metal",
            "--mode",
            "affine",
            "--bits",
            "4",
            "--group-size",
            "64",
            "--k",
            "256",
            "--n",
            "256",
            "--rows",
            "1",
            "--route",
            "dot",
            "--no-preamble",
        ]
    )
    assert rc == 0
    raw = capsys.readouterr().out
    assert "// >>> OMLX_SECTION: kernel_entry >>>" in raw
    assert "[[kernel]]" in raw

    rc = cli.main(
        [
            "warm-report",
            "--mode",
            "affine",
            "--bits",
            "4",
            "--group-size",
            "64",
            "--shapes",
            "256x256,256x256",
            "--dry-run",
            "--json",
        ]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["compile_executed"] is False
    assert report["layers_scanned"] == 2


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal GPU")
def test_cli_bench_gpu_json(capsys):
    cli = _cli()
    rc = cli.main(
        [
            "bench",
            "--mode",
            "affine",
            "--bits",
            "4",
            "--group-size",
            "64",
            "--k",
            "256",
            "--n",
            "256",
            "--rows",
            "1",
            "--units",
            "2",
            "--rounds",
            "4",
            "--chain-len",
            "4",
            "--warmup",
            "1",
            "--json",
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["c2_pass"] is True
    assert result["layer_rel_l2_fp32"] > 0.0
    assert result["layer_median_us"] > 0


def test_cli_lazy_audit_dispatch(monkeypatch, capsys):
    # Importing the CLI does not eagerly import the parallel lane's audit module.
    code = (
        "import sys; import omlx.formats.__main__ as cli; "
        "assert 'omlx.formats.audit' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    cli = _cli()

    def missing(name):
        raise ModuleNotFoundError(
            f"No module named {name!r}", name="omlx.formats.audit"
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(cli, "importlib", types.SimpleNamespace(import_module=missing))
        rc = cli.main(["audit", "--help"])
        assert rc == 2
        assert "not available in this build" in capsys.readouterr().err

    # A missing dependency *inside* the audit module must propagate, never be
    # reported as "audit is not available".
    def import_time_inner_missing(name):
        raise ModuleNotFoundError("inner_dep", name="inner_dep")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            cli,
            "importlib",
            types.SimpleNamespace(import_module=import_time_inner_missing),
        )
        with pytest.raises(ModuleNotFoundError) as excinfo:
            cli.main(["audit", "omlx/"])
        assert excinfo.value.name == "inner_dep"

    calls: list[list[str]] = []
    fake = types.ModuleType("omlx.formats.audit")

    def fake_main(argv):
        calls.append(list(argv))
        return 7

    fake.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omlx.formats.audit", fake)
    assert cli.main(["audit", "--help", "omlx/", "--strict"]) == 7
    assert calls == [["--help", "omlx/", "--strict"]]

    def raise_system_exit(argv):
        raise SystemExit(3)

    fake.main = raise_system_exit  # type: ignore[attr-defined]
    assert cli.main(["audit", "omlx/"]) == 3

    def raise_inner_missing(argv):
        raise ModuleNotFoundError("inner_dep", name="inner_dep")

    fake.main = raise_inner_missing  # type: ignore[attr-defined]
    with pytest.raises(ModuleNotFoundError) as excinfo:
        cli.main(["audit", "omlx/"])
    assert excinfo.value.name == "inner_dep"


def _assert_no_env_reads(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
            pytest.fail(f"Rule 30 violation: .{node.attr} in {path}:{node.lineno}")
        if isinstance(node, ast.Name) and node.id in ("environ", "getenv"):
            pytest.fail(f"Rule 30 violation: {node.id} in {path}:{node.lineno}")
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in ("environ", "getenv"):
                    pytest.fail(
                        f"Rule 30 violation: import {alias.name} in "
                        f"{path}:{node.lineno}"
                    )


def test_cli_subprocess_smoke_and_final_invariants():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "omlx.formats",
            "explain",
            "--mode",
            "affine",
            "--bits",
            "4",
            "--group-size",
            "64",
            "--k",
            "256",
            "--n",
            "256",
            "--rows",
            "1,32",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["mode"] == "affine"
    assert [row["rows"] for row in payload["row_traces"]] == [1, 32]

    # Rule 28: runtime modules import neither the debug API nor the CLI.
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

    # Rule 30: zero os.environ / os.getenv in either debug module.
    _assert_no_env_reads(Path(debug.__file__))
    _assert_no_env_reads(Path(_cli().__file__))

    # Rule 20: Metal emission sources and tuning tables are unchanged.
    from omlx.patches import qwen35_moe_routed_decode as routed
    from tests.test_formats_tile_dense_parity import _DENSE_DIGEST

    routed._ensure_tuning_registered()
    assert tile_dense.source_digest() == _DENSE_DIGEST
    assert tuning.stale_entries() == []


# ---------------------------------------------------------------------------
# Pre-land review fixes (pre-fmtdebug-ef1e42377)
# ---------------------------------------------------------------------------


def test_dump_metal_buffer_indices_match_mlx():
    layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    dumped = debug.dump_metal(layer, rows=32, route="steel", chip="applegpu_g15d_c80")
    expected_bindings = [
        "const device bfloat16_t* x [[buffer(0)]]",
        "const constant int* x_shape [[buffer(1)]]",
        "const constant int& x_ndim [[buffer(2)]]",
        "const device uint32_t* w_weight [[buffer(3)]]",
        "const constant int* w_weight_shape [[buffer(4)]]",
        "const device bfloat16_t* w_scales [[buffer(5)]]",
        "const device bfloat16_t* w_biases [[buffer(6)]]",
        "device bfloat16_t* y [[buffer(7)]]",
    ]
    positions = []
    for binding in expected_bindings:
        pos = dumped.metal.find(binding)
        assert pos != -1, f"missing binding {binding!r} in:\n{dumped.metal[-1500:]}"
        positions.append(pos)
    assert positions == sorted(positions), "buffer bindings out of MLX order"

    # 0-d scalar input (N_size on mma) must never emit _shape/_strides/_ndim.
    mma_dump = debug.dump_metal(layer, rows=4, route="mma", chip="applegpu_g15d_c80")
    assert "N_size_shape" not in mma_dump.metal
    assert "N_size_ndim" not in mma_dump.metal
    assert "const constant uint32_t& N_size [[buffer(" in mma_dump.metal


def test_dump_metal_forced_route_matches_auto_tuned_policy():
    layer_2560 = debug.make_layer("affine", bits=4, group_size=64, k=2560, n=2560)
    auto = debug.dump_metal(layer_2560, rows=48, chip="applegpu_g15d_c80")
    forced = debug.dump_metal(
        layer_2560, rows=48, chip="applegpu_g15d_c80", route="steel"
    )
    assert auto.route == "steel"
    assert forced.policy_repr == auto.policy_repr
    assert forced.launch.template == auto.launch.template
    assert forced.launch.grid == auto.launch.grid
    assert forced.launch.threadgroup == auto.launch.threadgroup
    assert forced.raw_digest == auto.raw_digest


def test_cli_bench_uses_realistic_weights(monkeypatch, capsys):
    cli = _cli()
    seen: list[tuple[int | None, bool | None]] = []
    real_make_layer = debug.make_layer

    def spy_make_layer(*args, **kwargs):
        seen.append((kwargs.get("seed"), kwargs.get("realistic_weights")))
        return real_make_layer(*args, **kwargs)

    monkeypatch.setattr(debug, "make_layer", spy_make_layer)
    monkeypatch.setattr(debug, "bench", lambda *a, **kw: object())
    monkeypatch.setattr(debug, "format_bench", lambda res, as_json=False: "{}")

    rc = cli.main(
        [
            "bench",
            "--mode",
            "affine",
            "--bits",
            "4",
            "--group-size",
            "64",
            "--k",
            "256",
            "--n",
            "256",
            "--units",
            "2",
            "--seed",
            "7",
            "--json",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "{}"
    assert seen == [(7, True), (8, True)]


class _SignFlipNoInverseFormat(QuantFormat):
    """Toy format with sign-flip InputTransform and inverse_weight=None."""

    mode = "fmtdebug_signflip_no_inv"
    codes_name = "qcodes"
    tensors = ("qscale",)
    shared_tensors = ("qsigns",)
    contract = 1

    def layout(self, k: int, n: int, bits: int, group_size: int):
        if k <= 0 or n <= 0 or bits != 4 or group_size != 64 or k % 64 != 0:
            return None
        return {
            "qcodes": ((n, k // 8), mx.uint32),
            "qscale": ((n, k // 64), None),
            "qsigns": ((k,), None),
        }

    def input_transform(self, layer):
        return transforms.InputTransform(
            version=transforms.TRANSFORM_VERSION,
            kind="sign_flip",
            tensors=("qsigns",),
            block=64,
            apply=lambda x, s: x * s.astype(x.dtype),
            inverse_weight=None,
        )

    def decode(self, tensors, bits: int, group_size: int) -> mx.array:
        q = tensors["qcodes"]
        s = tensors["qscale"].astype(mx.float32)
        n = q.shape[-2]
        k = q.shape[-1] * 8
        shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
        mask = mx.array(0xF, dtype=mx.uint32)
        nibbles = ((q[..., None] >> shifts) & mask).astype(mx.float32)
        vals = (nibbles.reshape(*q.shape[:-2], n, k) - 7.5) / 7.5
        scales_exp = mx.repeat(s, 64, axis=-1)
        return vals * scales_exp


def test_bench_transform_without_inverse_weight_skips_baseline():
    layer = debug.make_layer(
        _SignFlipNoInverseFormat(),
        bits=4,
        group_size=64,
        k=128,
        n=128,
        seed=3,
        realistic_weights=True,
    )
    layer["qsigns"] = mx.full((128,), -1.0, dtype=mx.bfloat16)

    bases, base_bits, base_gs, note = debug._resolve_baselines(
        (layer,), None, mx.bfloat16
    )
    assert (bases, base_bits, base_gs) == (None, None, None)
    assert note is not None and "inverse_weight" in note

    with pytest.raises(ValueError, match="inverse_weight"):
        debug._make_mlx_baseline(
            layer, dtype=mx.bfloat16, baseline_gs=64, exact_affine=False
        )

    if mx.metal.is_available():
        res = debug.bench(
            layer, rows=2, dtype=mx.bfloat16, chain_len=2, rounds=2, warmup=0
        )
        assert res.baseline_available is False
        assert res.baseline_rel_l2_fp32 is None
        assert res.c2_threshold is None
        assert res.layer_finite is True
        assert res.c2_pass is True
        assert res.baseline_note is not None and "inverse_weight" in res.baseline_note


def test_warm_report_dispatch_runs_for_single_layer_and_sequence(monkeypatch):
    dispatched: list[nn.Module] = []

    def fake_warm_dispatch(model, **kwargs):
        dispatched.append(model)

    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(kernels, "warm", lambda instances: ())
    monkeypatch.setattr(kernels, "warm_proofs", lambda tasks: ())
    monkeypatch.setattr(linear, "warm_model_dispatch", fake_warm_dispatch)

    layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    layer2 = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    debug.warm_report(layer, dtypes=(mx.bfloat16,), compile=True, warm_dispatch=True)
    debug.warm_report(
        [layer, layer2], dtypes=(mx.bfloat16,), compile=True, warm_dispatch=True
    )

    assert len(dispatched) == 2
    assert dispatched[0] is layer
    assert isinstance(dispatched[1], nn.Module)
    seq_mods = [mod for _, mod in dispatched[1].named_modules()]
    assert any(mod is layer for mod in seq_mods)
    assert any(mod is layer2 for mod in seq_mods)


class _AllDtypesTileToyFormat(_TabledToyFormat):
    """Toy format whose Tile face sets dtypes=() so tile_exact is True for fp32."""

    mode = "fmtdebug_all_dtypes_tile_toy"

    def tile(self, layer, k: int, n: int) -> Tile | None:
        base = super().tile(layer, k, n)
        if base is None:
            return None
        return Tile(
            key=f"{base.key}_alldt",
            source=base.source,
            type=base.type,
            tensors=base.tensors,
            dtypes=(),
            tables=base.tables,
            reads=base.reads,
        )


def test_explain_inexact_dtype_has_empty_schedule():
    bf16_layer = debug.make_layer(
        "affine", bits=4, group_size=64, k=256, n=256, dtype=mx.bfloat16
    )
    res_fp16 = debug.explain(
        bf16_layer, rows=(32,), dtype="fp16", chip="applegpu_g15d_c80"
    )
    assert res_fp16.schedule_summary == ()
    assert res_fp16.row_traces[0].selected_route == "decode"

    # When Tile.dtypes == (), tile_exact is True for fp32, yet tile_dense still
    # requires float16/bfloat16 so schedule_summary must remain empty.
    all_dt_layer = debug.make_layer(
        _AllDtypesTileToyFormat(),
        bits=4,
        group_size=64,
        k=256,
        n=256,
        dtype=mx.bfloat16,
    )
    res_fp32 = debug.explain(
        all_dt_layer, rows=(32,), dtype="fp32", chip="applegpu_g15d_c80"
    )
    tile_trace = next(f for f in res_fp32.faces if f.kind == "tile")
    assert tile_trace.available is True
    assert tile_trace.exact_for_dtype is True
    assert res_fp32.schedule_summary == ()


def test_warm_report_attributes_steel_on_nax_host(tmp_path, monkeypatch):
    chip = tuning.current_chip()
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
                "rows_bucket": "17-2048",
            },
            "face_sha256": tuning.face_digest(face),
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
        }
    ]
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / f"{chip}.json").write_text(
        json.dumps({"schema_version": 1, "chip": chip, "entries": entries})
    )
    monkeypatch.setattr(linear, "_nax_host", lambda: True)
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    tuning.reset_cache()
    try:
        layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
        rep = debug.warm_report(layer, dtypes=(mx.bfloat16,), compile=False)
        names = [p.kernel_name for p in rep.pipelines]
        assert any(name.startswith("omlx_steel_dense_") for name in names), names
        assert all(name != "omlx_unknown_kernel" for name in names), names
    finally:
        tuning.reset_cache()


# ---------------------------------------------------------------------------
# Pre-land review follow-ups (pre-fmtdebug-50a60c722)
# ---------------------------------------------------------------------------


def test_dump_metal_forced_route_uses_tuned_policy_when_auto_differs(
    tmp_path, monkeypatch
):
    chip = "applegpu_g15d_c80"
    face = affine.tile_face(4, 64, mx.bfloat16)
    face_sha = tuning.face_digest(face)
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
                "rows_bucket": "17-24",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": tuning.skeleton_digest("dense_tile"),
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "steel",
                "bm": 32,
                "bn": 32,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "x_nreads": 4,
                "param_staging": "group_steps",
                "vec_store": True,
            },
            "exactness": "bitwise",
        },
        {
            "key": {
                "route": "mma_tile",
                "face_id": face.key,
                "dtype": "bf16",
                "bits": 4,
                "group": 64,
                "n": 256,
                "k": 256,
                "rows_bucket": "20-20",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": tuning.skeleton_digest("mma_tile"),
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {"bn": 8, "bk": 32, "nsg": 8},
            "exactness": "bitwise",
        },
        {
            "key": {
                "route": "mma_tile",
                "face_id": face.key,
                "dtype": "bf16",
                "bits": 4,
                "group": 64,
                "n": 256,
                "k": 256,
                "rows_bucket": "17-24",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": tuning.skeleton_digest("mma_tile"),
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {"bn": 8, "bk": 32, "nsg": 8},
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
                "rows_bucket": "4-4",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": tuning.skeleton_digest("dense_tile"),
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "steel",
                "bm": 32,
                "bn": 32,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "x_nreads": 4,
                "param_staging": "group_steps",
                "vec_store": True,
            },
            "exactness": "bitwise",
        },
        {
            "key": {
                "route": "msg_tile",
                "face_id": face.key,
                "dtype": "bf16",
                "bits": 4,
                "group": 64,
                "n": 256,
                "k": 256,
                "rows_bucket": "4",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": tuning.skeleton_digest("msg_tile"),
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {"nsg": 4, "load_lifetime": "column"},
            "exactness": "bitwise",
        },
    ]
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / f"{chip}.json").write_text(
        json.dumps({"schema_version": 1, "chip": chip, "entries": entries})
    )
    # Also write an M5 table where dense_tile is NAX so forcing steel must NOT
    # adopt a NAX policy.
    chip_m5 = "applegpu_g17s_c40"
    m5_entries = [
        {
            "key": {
                "route": "dense_tile",
                "face_id": face.key,
                "dtype": "bf16",
                "bits": 4,
                "group": 64,
                "n": 256,
                "k": 256,
                "rows_bucket": "17-24",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": tuning.skeleton_digest("dense_tile"),
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "nax",
                "bm": 32,
                "bn": 64,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "sched": "db",
            },
            "exactness": "bitwise",
        },
        {
            "key": {
                "route": "mma_tile",
                "face_id": face.key,
                "dtype": "bf16",
                "bits": 4,
                "group": 64,
                "n": 256,
                "k": 256,
                "rows_bucket": "20-20",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": tuning.skeleton_digest("mma_tile"),
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {"bn": 8, "bk": 32, "nsg": 8},
            "exactness": "bitwise",
        },
    ]
    (v1 / f"{chip_m5}.json").write_text(
        json.dumps({"schema_version": 1, "chip": chip_m5, "entries": m5_entries})
    )

    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    tuning.reset_cache()
    try:
        layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
        # 1. Auto selects mma_tile (20-20); forced route="steel" must pick tuned
        # dense_tile (17-24, bk=64, vec_store=True), not default (bk=32, False).
        auto_20 = debug.dump_metal(layer, rows=20, chip=chip)
        assert auto_20.route == "mma"
        assert "bn=8" in (auto_20.policy_repr or "")

        forced_steel = debug.dump_metal(layer, rows=20, route="steel", chip=chip)
        assert "bk=64" in (forced_steel.policy_repr or ""), forced_steel.policy_repr
        assert "vec_store=True" in (forced_steel.policy_repr or "")
        tmpl_steel = dict(forced_steel.launch.template)
        assert tmpl_steel["BK"] == 64
        assert tmpl_steel["VEC_STORE"] is True

        # 1b. oqh_affine maps to affine table_face during forced-route lookup.
        oqh_aff = debug.make_layer("oqh_affine", bits=4, group_size=64, k=256, n=256)
        forced_oqh_steel = debug.dump_metal(oqh_aff, rows=20, route="steel", chip=chip)
        assert "bk=64" in (forced_oqh_steel.policy_repr or "")
        assert "vec_store=True" in (forced_oqh_steel.policy_repr or "")

        # 2. Auto selects steel (17-24, prio 2 > mma_tile prio 1 at rows=18);
        # forced route="mma" must pick tuned mma_tile (bn=8), not default bn=16.
        auto_18 = debug.dump_metal(layer, rows=18, chip=chip)
        assert auto_18.route == "steel"
        forced_mma = debug.dump_metal(layer, rows=18, route="mma", chip=chip)
        assert "bn=8" in (forced_mma.policy_repr or ""), forced_mma.policy_repr
        assert forced_mma.launch.grid == (32 * 8, 256 // 8, 1)

        # 3. Auto selects steel (4-4 outranks msg_tile "4"); forced route="msg"
        # must pick tuned msg_tile (nsg=4, ONE_COL_STREAMING), not default nsg=8.
        auto_4 = debug.dump_metal(layer, rows=4, chip=chip)
        assert auto_4.route == "steel"
        forced_msg = debug.dump_metal(layer, rows=4, route="msg", chip=chip)
        assert "nsg=4" in (forced_msg.policy_repr or ""), forced_msg.policy_repr
        assert "ONE_COL_STREAMING" in (forced_msg.policy_repr or "")
        assert forced_msg.launch.threadgroup == (32 * 4, 1, 1)

        # 4. On M5 where auto is mma (20-20) and dense_tile (17-24) is nax:
        # forcing nax uses tuned nax; forcing steel ignores nax entry and uses
        # default steel policy (bk=32, vec_store=False).
        forced_nax = debug.dump_metal(layer, rows=20, route="nax", chip=chip_m5)
        assert "backend='nax'" in (forced_nax.policy_repr or "")
        assert "sched='db'" in (forced_nax.policy_repr or "")
        forced_m5_steel = debug.dump_metal(layer, rows=20, route="steel", chip=chip_m5)
        assert "backend='steel'" in (forced_m5_steel.policy_repr or "")
        assert "bk=32" in (forced_m5_steel.policy_repr or "")
    finally:
        tuning.reset_cache()


def test_warm_report_registers_dense_names_under_tile_dense_lock(monkeypatch):
    class _TrackingLock:
        def __init__(self) -> None:
            self.enters = 0
            self.exits = 0
            self.depth = 0

        def __enter__(self) -> _TrackingLock:
            self.enters += 1
            self.depth += 1
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.depth -= 1
            self.exits += 1

    tracker = _TrackingLock()
    real_build = tile_dense._build
    build_calls: list[tuple[str, bool]] = []

    def spy_build(face, backend):
        build_calls.append((backend, tracker.depth > 0))
        return real_build(face, backend)

    monkeypatch.setattr(tile_dense, "_lock", tracker)
    monkeypatch.setattr(tile_dense, "_build", spy_build)

    layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
    raw_map: dict[int, str] = {}
    debug._register_raw_names_for_layer(layer, tensor_units=True, raw_name_map=raw_map)
    assert [b for b, _ in build_calls] == ["nax", "steel"]
    assert all(held for _, held in build_calls), build_calls
    # 2 tile_dense._build calls + 1 tile_dense._splitk_accum_kernel call.
    assert tracker.enters == 3 and tracker.exits == 3 and tracker.depth == 0

    build_calls.clear()
    debug.warm_report(layer, dtypes=(mx.bfloat16,), compile=False)
    assert len(build_calls) >= 1
    assert all(held for _, held in build_calls), build_calls
    assert tracker.depth == 0


def test_assemble_standalone_metal_scalar_guard_and_strides_binding():
    launch = debug.LaunchGeometry(
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        threadgroups=(1, 1, 1),
        template=(("T", "bfloat16_t"),),
        input_names=("x", "s"),
        output_names=("y",),
        input_specs=(
            ("x", "bfloat16_t", (4, 64), "device"),
            ("s", "uint32_t", (), "scalar"),
        ),
        output_specs=(("y", "bfloat16_t", (4, 64)),),
        staged_table_bytes=0,
        split_k=None,
        k_part=None,
    )
    raw_source = (
        "// uses x_shape x_strides x_ndim and mentions s_shape s_strides s_ndim\n"
        "y[0] = x[0];\n"
    )
    metal, _ = debug._assemble_standalone_metal(
        kernel_name="omlx_test_bindings",
        route="steel",
        mode="affine",
        bits=4,
        group_size=64,
        k=64,
        n=64,
        rows=4,
        chip="applegpu_g15d_c80",
        table_mode="selects",
        policy_repr=None,
        raw_digest="0" * 64,
        raw_source=raw_source,
        mlx_route_headers="",
        face_header="",
        skeleton_header="",
        matrix_declarations="",
        skeleton_body=raw_source,
        template=[("T", mx.bfloat16)],
        launch=launch,
        include_mlx_preamble=False,
    )
    sig_start = metal.index("[[kernel]] void omlx_test_bindings_tmpl(")
    sig_end = metal.index(") {", sig_start)
    sig = metal[sig_start:sig_end]
    expected_bindings = [
        "const device bfloat16_t* x [[buffer(0)]]",
        "const constant int* x_shape [[buffer(1)]]",
        "const constant int64_t* x_strides [[buffer(2)]]",
        "const constant int& x_ndim [[buffer(3)]]",
        "const constant uint32_t& s [[buffer(4)]]",
        "device bfloat16_t* y [[buffer(5)]]",
    ]
    positions = [sig.find(b) for b in expected_bindings]
    assert all(p != -1 for p in positions), sig
    assert positions == sorted(positions), sig
    assert "s_shape" not in sig
    assert "s_strides" not in sig
    assert "s_ndim" not in sig


def test_dump_metal_fp32_skips_tile_dense_schedule(tmp_path, monkeypatch):
    chip = "applegpu_g15d_c80"
    # 1. oqh_affine at fp32 must skip _build_schedule (which would fail inside
    # affine.tile_face(..., fp32)) and raise ValueError on inexact dtype.
    oqh_aff = debug.make_layer("oqh_affine", bits=4, group_size=64, k=256, n=256)
    with pytest.raises(ValueError, match="exact for fp32"):
        debug.dump_metal(oqh_aff, rows=32, dtype=mx.float32, route="steel", chip=chip)

    # 2. A Tile with dtypes=() is exact for fp32, yet dump_metal must not query
    # tile_dense._build_schedule or tuning tables for fp32.
    all_dt_layer = debug.make_layer(
        _AllDtypesTileToyFormat(),
        bits=4,
        group_size=64,
        k=256,
        n=256,
        dtype=mx.bfloat16,
    )
    tile_face = formats._checked_face(
        all_dt_layer.fmt, all_dt_layer, 256, 256, Tile, "tile"
    )
    assert tile_face is not None
    entries = [
        {
            "key": {
                "route": "dense_tile",
                "face_id": tile_face.key,
                "dtype": "fp32",
                "bits": 4,
                "group": 64,
                "n": 256,
                "k": 256,
                "rows_bucket": "25-32",
            },
            "face_sha256": tuning.face_digest(tile_face),
            "skeleton_sha256": tuning.skeleton_digest("dense_tile"),
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "steel",
                "bm": 32,
                "bn": 32,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "x_nreads": 4,
                "param_staging": "group_steps",
                "vec_store": True,
            },
            "exactness": "bitwise",
        }
    ]
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / f"{chip}.json").write_text(
        json.dumps({"schema_version": 1, "chip": chip, "entries": entries})
    )
    sched_calls: list[object] = []
    real_build_schedule = tile_dense._build_schedule

    def spy_build_schedule(*args, **kwargs):
        sched_calls.append(args)
        return real_build_schedule(*args, **kwargs)

    monkeypatch.setattr(tile_dense, "_build_schedule", spy_build_schedule)
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    tuning.reset_cache()
    try:
        dumped = debug.dump_metal(
            all_dt_layer, rows=32, dtype=mx.float32, route="steel", chip=chip
        )
        assert sched_calls == []
        assert "bk=32" in (dumped.policy_repr or ""), dumped.policy_repr
        assert "vec_store=False" in (dumped.policy_repr or "")
    finally:
        tuning.reset_cache()


class _SignFlipWithInverseFormat(_SignFlipNoInverseFormat):
    """Same mode as _SignFlipNoInverseFormat, but provides inverse_weight."""

    def input_transform(self, layer):
        return transforms.InputTransform(
            version=transforms.TRANSFORM_VERSION,
            kind="sign_flip",
            tensors=("qsigns",),
            block=64,
            apply=lambda x, s: x * s.astype(x.dtype),
            inverse_weight=lambda w, s: w * s.astype(w.dtype),
        )


def test_resolve_baselines_mixed_transform_sequence_and_no_redundant_decode(
    monkeypatch,
):
    layer_inv = debug.make_layer(
        _SignFlipWithInverseFormat(),
        bits=4,
        group_size=64,
        k=128,
        n=128,
        seed=1,
        realistic_weights=True,
    )
    layer_no_inv = debug.make_layer(
        _SignFlipNoInverseFormat(),
        bits=4,
        group_size=64,
        k=128,
        n=128,
        seed=2,
        realistic_weights=True,
    )

    # Mixed sequence where layers[0] has inverse_weight but layers[1] lacks it
    # must cleanly skip the baseline instead of raising inside _make_mlx_baseline.
    bases, base_bits, base_gs, note = debug._resolve_baselines(
        (layer_inv, layer_no_inv), None, mx.bfloat16
    )
    assert (bases, base_bits, base_gs) == (None, None, None)
    assert note is not None and "inverse_weight" in note

    # Resolving baselines for an invertible layer must not decode layer[0] twice.
    decode_calls: list[int] = []
    real_plain = debug._plain_weight_fp32

    def spy_plain(layer):
        decode_calls.append(id(layer))
        return real_plain(layer)

    monkeypatch.setattr(debug, "_plain_weight_fp32", spy_plain)
    bases_ok, bits_ok, gs_ok, _ = debug._resolve_baselines(
        (layer_inv,), None, mx.bfloat16
    )
    assert bases_ok is not None and len(bases_ok) == 1
    assert (bits_ok, gs_ok) == (4, 64)
    assert decode_calls == [id(layer_inv)]


# ---------------------------------------------------------------------------
# Pre-land review follow-ups (pre-fmtdebug-b6ef1b212)
# ---------------------------------------------------------------------------


def test_warm_report_names_splitk_accum_kernel(tmp_path, monkeypatch):
    chip = tuning.current_chip()
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
                "rows_bucket": "17-2048",
            },
            "face_sha256": tuning.face_digest(face),
            "skeleton_sha256": tuning.skeleton_digest("dense_tile"),
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "steel",
                "bm": 32,
                "bn": 32,
                "bk": 32,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "x_nreads": 8,
                "split_k": 2,
            },
            "exactness": "bitwise",
        }
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
        layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
        rep = debug.warm_report(layer, dtypes=(mx.bfloat16,), compile=False)
        names = [p.kernel_name for p in rep.pipelines]
        assert "omlx_dense_splitk_accum" in names, names
        assert all(n != "omlx_unknown_kernel" for n in names), names
    finally:
        tuning.reset_cache()


def test_dump_metal_forced_nax_finds_entry_behind_other_backend(tmp_path, monkeypatch):
    chip = "applegpu_g17s_c40"
    face = affine.tile_face(4, 64, mx.bfloat16)
    face_sha = tuning.face_digest(face)
    skel_sha = tuning.skeleton_digest("dense_tile")
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
                "rows_bucket": "20-20",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": skel_sha,
            "measured_at": "2026-11-27T00:00:00Z",
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
                "rows_bucket": "20-20",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": skel_sha,
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "nax",
                "bm": 32,
                "bn": 64,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "sched": "seg",
                "gx": 32,
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
        layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
        dumped = debug.dump_metal(layer, rows=20, route="nax", chip=chip)
        assert dict(dumped.launch.template)["GX"] == 32
    finally:
        tuning.reset_cache()


def test_dump_metal_forced_dense_skips_illegal_and_stale_entries(tmp_path, monkeypatch):
    chip = "applegpu_g17s_c40"
    face = affine.tile_face(4, 64, mx.bfloat16)
    face_sha = tuning.face_digest(face)
    skel_sha = tuning.skeleton_digest("dense_tile")
    # 1. Auto selects steel (20-20); higher-rank 20-20 nax entries have either
    #    stale skeleton/face digests or illegal pad_bytes (weight_bytes=9216 +
    #    pad_bytes=25000 > 32768). Forced nax must skip all three and select
    #    the lower-rank legal 17-24 nax entry (gx=32).
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
                "rows_bucket": "20-20",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": skel_sha,
            "measured_at": "2026-11-27T00:00:00Z",
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
                "rows_bucket": "20-20",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": "0" * 64,
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "nax",
                "bm": 32,
                "bn": 64,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "sched": "seg",
                "gx": 4,
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
                "rows_bucket": "20-20",
            },
            "face_sha256": "f" * 64,
            "skeleton_sha256": skel_sha,
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "nax",
                "bm": 32,
                "bn": 64,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "sched": "seg",
                "gx": 8,
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
                "rows_bucket": "20-20",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": skel_sha,
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "nax",
                "bm": 32,
                "bn": 64,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "sched": "seg",
                "gx": 16,
                "pad_bytes": 25000,
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
                "rows_bucket": "17-24",
            },
            "face_sha256": face_sha,
            "skeleton_sha256": skel_sha,
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "nax",
                "bm": 32,
                "bn": 64,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "sched": "seg",
                "gx": 32,
            },
            "exactness": "bitwise",
        },
        # 2. Entry legal for table_face (0 table_bytes: 18432 + 14304 = 32736
        #    <= 32768) but illegal for an actual_face carrying 64 table_bytes
        #    (32800 > 32768).
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
            "face_sha256": face_sha,
            "skeleton_sha256": skel_sha,
            "measured_at": "2026-11-27T00:00:00Z",
            "params": {
                "backend": "nax",
                "bm": 32,
                "bn": 64,
                "bk": 64,
                "wm": 2,
                "wn": 2,
                "align_n": True,
                "sched": "db",
                "gx": 24,
                "pad_bytes": 14304,
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
        layer = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
        dumped = debug.dump_metal(layer, rows=20, route="nax", chip=chip)
        assert dict(dumped.launch.template)["GX"] == 32

        # Direct actual_face vs table_face check: 64-byte table pushes tg over 32 KiB.
        big_face = Tile(
            key="fmtdebug_big_table_bf16",
            source=face.source,
            type=face.type,
            tensors=face.tensors,
            dtypes=face.dtypes,
            tables=(Table("lut", mx.float32, tuple(float(i) for i in range(16))),),
            reads=face.reads,
        )
        assert (
            debug._lookup_forced_dense_policy(
                face,
                face,
                backend="nax",
                dtype=mx.bfloat16,
                bits=4,
                group=64,
                n=256,
                k=256,
                rows=32,
                chip=chip,
            )
            is not None
        )
        assert (
            debug._lookup_forced_dense_policy(
                face,
                big_face,
                backend="nax",
                dtype=mx.bfloat16,
                bits=4,
                group=64,
                n=256,
                k=256,
                rows=32,
                chip=chip,
            )
            is None
        )
    finally:
        tuning.reset_cache()
