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


def test_dump_metal_section_markers_and_deterministic_digests():
    dense = debug.make_layer("affine", bits=4, group_size=64, k=256, n=256)
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

    # Rule 20: tile_dense.source_digest() unchanged and tuning.stale_entries() empty.
    from omlx.patches import qwen35_moe_routed_decode as routed

    routed._ensure_tuning_registered()
    assert (
        tile_dense.source_digest()
        == "48 895530b5126b4c0bef2de690245c34cdea1c5fb5b19bbfba85413d9d064b0898"
    )
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
        ("mma", dense, 4),
        ("msg", dense, 4),
        ("switch_dot", switch, 4),
        ("switch_tile_steel", switch, 32),
    ]
    for route, layer, rows in cases:
        dumped = debug.dump_metal(
            layer, rows=rows, route=route, chip="applegpu_g15d_c80"
        )
        src_file = tmp_path / f"{route}.metal"
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
        ), f"xcrun metal failed for route={route}:\n{proc.stderr}"


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
