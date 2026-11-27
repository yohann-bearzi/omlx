# SPDX-License-Identifier: Apache-2.0
"""Unified parameterized direct dense prefill skeleton (PFX-1): steel + NAX."""

from __future__ import annotations

import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.formats import affine, kernels, linear, testing, tile_dense, tuning
from omlx.formats.linear import FormatLinear
from omlx.formats.mlx import traversal as mlx_traversal
from omlx.patches import m5_gather_qmm_nax as nax
from tests.test_formats_tables import tabled_tile
from tests.test_formats_tile_dense import _no_decode, _one_expert
from tests.test_formats_tile_prefill import (
    _TILE,
    _stack,
    _tile_toy,
    needs_nax,
    needs_steel,
)
from tests.test_formats_tile_prefill import toy as toy
from tests.test_qwen35_moe_routed_decode import _same_bits

_STEEL_POLICIES = [
    tuning.DenseTilePolicy("steel", 16, 32, 32, 1, 2, True, x_nreads=8),
    tuning.DenseTilePolicy("steel", 32, 32, 32, 2, 2, True, x_nreads=8),
    tuning.DenseTilePolicy("steel", 64, 32, 32, 2, 2, True, x_nreads=8),
    tuning.DenseTilePolicy("steel", 64, 64, 32, 2, 2, True, x_nreads=8),
]

_NAX_SPECS = [
    (32, 64, 64, 2, 2, True, 0, 0, "db"),
    (32, 64, 64, 2, 2, True, 0, 0, "seg"),
    (64, 64, 64, 2, 2, True, 0, 0, "seg"),
    (64, 64, 128, 2, 2, True, 32, 0, "seg"),
    (128, 64, 128, 4, 2, True, 32, 8192, "seg"),
]


@pytest.fixture(autouse=True)
def by_hand():
    with getattr(testing, "verified", lambda *_: nullcontext())("qwen4_exp"):
        yield


def _layer(n, k, bits, gs, dtype=mx.bfloat16, seed=0, *, bias=False):
    mx.random.seed(seed)
    dense = nn.Linear(k, n, bias=bias)
    dense.set_dtype(dtype)
    quantized = dense.to_quantized(bits=bits, group_size=gs)
    layer = FormatLinear(
        "affine",
        bits,
        gs,
        k,
        {name: quantized[name] for name in ("weight", "scales", "biases")},
        bias=quantized.get("bias") if bias else None,
    )
    return quantized, layer


def _x(shape, dtype=mx.bfloat16, seed=7):
    return (mx.random.normal(shape, key=mx.random.key(seed)) * 0.5).astype(dtype)


def _dot(array: mx.array) -> str:
    buf = io.StringIO()
    mx.export_to_dot(buf, array)
    return buf.getvalue()


_DENSE_DIGEST = "49 e7a95a88fb60ed5685a812e25fb58c7bb4c43c883d030c68b1cc64a6e9109401"


def test_dense_skeleton_source_digest_is_pinned():
    assert tile_dense.source_digest() == _DENSE_DIGEST


def test_affine_tile_8b_uses_aligned_uint2_when_k_is_multiple_of_8():
    source = affine.tile_face(8, 64, mx.bfloat16).source
    assert "conditional_t<K % 8 == 0, uint2, packed_uint2>" in source


def test_old_b10_dual_dense_paths_are_deleted():
    assert not hasattr(tile_dense, "one_run")
    assert "one_run" not in nax._FACE_KINDS
    assert not hasattr(nax, "_ONE_RUN_HEADER")
    assert (
        kernels.hand_tables(tile_dense._STEEL_HEADER + tile_dense._STEEL_SOURCE) == ()
    )
    assert kernels.hand_tables(tile_dense._NAX_HEADER + tile_dense._NAX_SOURCE) == ()


def test_dense_tile_defaults_and_no_parallel_tuning_tables():
    for deleted in (
        "DENSE_ROW_RANGES",
        "M_BUCKETS",
        "m_bucket",
        "dense_tile_policy",
        "dense_tile_rule",
        "dense_tile_schedule",
    ):
        assert not hasattr(tuning, deleted), deleted
    assert "lookup" in tuning.__all__ and "row_bounds" in tuning.__all__
    assert tuning.skeleton_digest("dense_tile") is not None

    m3_small = tile_dense._default_steel_policy(33, 17408, 5120)
    assert (m3_small.backend, m3_small.bm, m3_small.bn) == ("steel", 16, 32)
    m3_wide = tile_dense._default_steel_policy(512, 17408, 5120)
    assert (m3_wide.backend, m3_wide.bm, m3_wide.bn) == ("steel", 64, 32)
    m3_ragged = tile_dense._default_steel_policy(512, 40, 2560)
    assert m3_ragged.backend == "steel" and not m3_ragged.align_n

    m5_small = tile_dense._default_nax_policy(4, 24, 17408, 5120)
    assert (m5_small.backend, m5_small.bm, m5_small.bn, m5_small.sched) == (
        "nax",
        32,
        64,
        "db",
    )
    m5_mid = tile_dense._default_nax_policy(4, 64, 5120, 17408)
    assert (m5_mid.backend, m5_mid.bm, m5_mid.bk) == ("nax", 64, 128)
    m5_big = tile_dense._default_nax_policy(4, 512, 17408, 5120)
    assert (m5_big.backend, m5_big.bm, m5_big.bk, m5_big.gx, m5_big.pad_bytes) == (
        "nax",
        128,
        128,
        32,
        8192,
    )
    m5_ragged = tile_dense._default_nax_policy(8, 512, 48, 2560)
    assert m5_ragged.backend == "nax" and not m5_ragged.align_n

    assert tuning.rows_bucket("dense_tile", 24) == "17-24"
    assert tuning.rows_bucket("dense_tile", 25) == "25-32"
    assert tuning.rows_bucket("dense_tile", 64) == "49-64"
    assert tile_dense._split_k_geometry(288, 64, 2) == (2, 192)


def test_dense_tile_json_entry_overrides_default_policy(tmp_path, monkeypatch):
    chip = "applegpu_g15d_c80"
    n, k = 128, 512
    _, layer = _layer(n, k, 4, 64)
    face = affine.tile_face(4, 64, mx.bfloat16)
    entry = {
        "key": {
            "route": "dense_tile",
            "face_id": face.key,
            "dtype": "bf16",
            "bits": 4,
            "group": 64,
            "n": n,
            "k": k,
            "rows_bucket": "257-512",
        },
        "face_sha256": tuning.face_digest(face),
        "skeleton_sha256": tuning.skeleton_digest("dense_tile"),
        "measured_at": "2026-11-22T11:30:00Z",
        "params": {
            "backend": "steel",
            "bm": 32,
            "bn": 32,
            "bk": 64,
            "wm": 2,
            "wn": 2,
            "align_n": True,
            "x_nreads": 4,
            "param_staging": "per_step",
            "vec_store": True,
        },
        "exactness": "bitwise",
    }
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / f"{chip}.json").write_text(
        json.dumps({"schema_version": 1, "chip": chip, "entries": [entry]})
    )
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    tuning.reset_cache()
    try:
        plan = tile_dense.build_plan(layer, tensor_units=False, chip=chip)
        assert plan is not None
        pol = tile_dense.policy_for(plan, 512, mx.bfloat16)
        assert (pol.backend, pol.bm, pol.bn, pol.bk, pol.vec_store) == (
            "steel",
            32,
            32,
            64,
            True,
        )
    finally:
        tuning.reset_cache()


def test_dense_plan_makes_zero_per_call_policy_or_device_queries(monkeypatch):
    n, k = 128, 512
    quantized, layer = _layer(n, k, 4, 64)
    x = _x((33, k))
    want = _one_expert(quantized, x)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    assert _same_bits(layer(x), want)

    def forbidden(*args, **kwargs):
        pytest.fail("hot dense prefill queried device or policy per call")

    monkeypatch.setattr(kernels, "gpu_generation", forbidden)
    monkeypatch.setattr(tuning, "lookup", forbidden)
    monkeypatch.setattr(tuning, "row_bounds", forbidden)
    monkeypatch.setattr(tuning, "current_chip", forbidden)
    assert _same_bits(layer(x), want)
    x2 = _x((65, k), seed=13)
    want2 = _one_expert(quantized, x2)
    assert _same_bits(layer(x2), want2)


def test_wide_and_mlx_table_backends_and_plugin_fallback(toy, monkeypatch):
    n, k, rows = 64, 512, 24
    quantized, layer = _layer(n, k, 8, 64)
    x = _x((rows, k))
    wide_pol = tuning.DenseTilePolicy("wide")
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: wide_pol)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        got_wide = layer(x)
        dot_wide = _dot(got_wide)
    assert "CustomKernel" in dot_wide or "Compiled" in dot_wide
    w32 = mx.dequantize(
        quantized["weight"],
        quantized["scales"].astype(mx.float32),
        quantized["biases"].astype(mx.float32),
        group_size=64,
        bits=8,
    )
    ref = x.astype(mx.float32) @ w32.T
    assert float(mx.abs(got_wide.astype(mx.float32) - ref).max().item()) <= 0.25

    _, layer_mlx = _layer(n, k, 8, 64)
    mlx_pol = tuning.DenseTilePolicy("mlx")
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: mlx_pol)
    with testing.fresh_proofs():
        got_mlx = layer_mlx(x)
    assert _same_bits(got_mlx, quantized(x))

    # A non-affine plugin format substitutes the chip's generic tile fallback
    # at construction time and never calls mx.quantized_matmul.
    fmt = toy(_tile_toy("no_mlx_bypass"))
    tensors = {key: a[0] for key, a in _stack(n, 256, seed=29).items()}
    plugin = FormatLinear("tiletoy", 8, 1, 256, tensors)
    xp = _x((rows, 256), mx.float16)
    ids = mx.zeros((rows,), mx.uint32)
    want_p = mx.gather_mm(
        xp[:, None, :],
        fmt.decode(tensors, 8, 1).astype(xp.dtype).T[None],
        rhs_indices=ids,
        sorted_indices=True,
    )[:, 0]

    def forbidden_qmm(*args, **kwargs):
        pytest.fail("non-affine plugin entered mx.quantized_matmul")

    monkeypatch.setattr(mx, "quantized_matmul", forbidden_qmm)
    with testing.fresh_proofs():
        assert _same_bits(plugin(xp), want_p)


def test_direct_dense_passes_zero_meta_or_params_arrays(monkeypatch):
    n, k = 128, 512
    quantized, layer = _layer(n, k, 4, 64)
    x = _x((33, k))
    want = _one_expert(quantized, x)
    seen = []
    real_build = tile_dense._kernel

    def observe(face, policy):
        mm = real_build(face, policy)

        def wrap(**kwargs):
            seen.append(len(kwargs["inputs"]))
            return mm(**kwargs)

        return wrap

    monkeypatch.setattr(tile_dense, "_kernel", observe)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        assert _same_bits(layer(x), want)
    assert seen and all(count == 4 for count in seen)


@needs_steel
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("policy", _STEEL_POLICIES, ids=lambda p: f"{p.bm}x{p.bn}")
def test_steel_direct_dense_geometries_match_mlx_bitwise(
    monkeypatch, policy, bits, dtype
):
    n, k, rows = 64, 512, 65
    quantized, layer = _layer(n, k, bits, 64, dtype)
    x = _x((rows, k), dtype)
    want = _one_expert(quantized, x)
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: policy)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        got = layer(x)
        dot = _dot(got)
    assert _same_bits(got, want)
    assert "CustomKernel" in dot or "Compiled" in dot


@needs_steel
@pytest.mark.parametrize("cache_p", [False, True])
@pytest.mark.parametrize("vec_store", [False, True])
@pytest.mark.parametrize("n", [1, 48, 80])
def test_steel_direct_dense_ragged_n_and_hooks_match_bitwise(
    monkeypatch, n, vec_store, cache_p
):
    k, rows = 512, 33
    quantized, layer = _layer(n, k, 8, 64)
    x = _x((rows, k))
    want = _one_expert(quantized, x)
    staging = (
        tuning.ParamStaging.GROUP_STEPS_COUNTER
        if cache_p
        else tuning.ParamStaging.PER_STEP
    )
    policy = tuning.DenseTilePolicy(
        "steel",
        32,
        32,
        32,
        2,
        2,
        False,
        x_nreads=8,
        param_staging=staging,
        vec_store=vec_store,
    )
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: policy)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        assert _same_bits(layer(x), want)


@needs_nax
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "spec",
    _NAX_SPECS,
    ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}_{s[8]}_gx{s[6]}_p{s[7]}",
)
def test_nax_direct_dense_geometries_match_mlx_bitwise(monkeypatch, spec, bits, dtype):
    bm, bn, bk, wm, wn, align_n, gx, pad, sched = spec
    policy = tuning.DenseTilePolicy(
        "nax", bm, bn, bk, wm, wn, align_n, gx=gx, pad_bytes=pad, sched=sched
    )
    n, k = 128, 512
    rows = 24 if policy.bm == 32 else (65 if policy.bm == 64 else 129)
    quantized, layer = _layer(n, k, bits, 64, dtype)
    x = _x((rows, k), dtype)
    want = _one_expert(quantized, x)
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: policy)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        got = layer(x)
        dot = _dot(got)
    assert _same_bits(got, want)
    assert "CustomKernel" in dot or "Compiled" in dot


@needs_nax
@pytest.mark.parametrize("n", [1, 40, 48, 96])
@pytest.mark.parametrize("k", [288, 512])
@pytest.mark.parametrize("rows", [17, 33, 65])
def test_nax_direct_dense_handles_ragged_n_and_k_bitwise(monkeypatch, n, k, rows):
    gs = 32 if k == 288 else 64
    quantized, layer = _layer(n, k, 4, gs)
    x = _x((rows, 2 * k))[..., ::2]
    want = _one_expert(quantized, x)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        assert _same_bits(layer(x), want)
    if n == 48 and k == 288 and rows == 17:
        forced_db = tuning.DenseTilePolicy(
            "nax", 32, 64, 64, 2, 2, True, gx=0, pad_bytes=0, sched="db"
        )
        monkeypatch.setattr(tuning, "lookup", lambda *a, **k: forced_db)
        _, layer_db = _layer(n, k, 4, gs)
        with testing.fresh_proofs():
            assert _same_bits(layer_db(x), want)


def test_direct_dense_runs_staged_table_and_subtracts_pad(monkeypatch):
    n, k, rows = 128, 512, 33
    quantized, _ = _layer(n, k, 4, 64)
    plain = affine.tile_face(4, 64, mx.bfloat16)
    staged = tabled_tile(plain)
    recorded = []
    real_mk = mx.fast.metal_kernel

    def spy(**kw):
        recorded.append(kw)
        return real_mk(**kw)

    monkeypatch.setattr(mx.fast, "metal_kernel", spy)
    if linear._nax_host():
        tile_dense._build.__wrapped__(plain, "nax")
        tile_dense._build.__wrapped__(staged, "nax")
        assert "PAD / sizeof(T)];" in recorded[-2]["source"]
        assert "(PAD > 64 ? PAD - 64 : 0) / sizeof(T)];" in recorded[-1]["source"]

    from omlx import formats

    class _TabledAffine(formats.QuantFormat):
        mode = "tabled_affine_test"
        tensors = ("scales", "biases")
        contract = formats.CONTRACT

        def layout(self, k_, n_, bits_, gs_):
            return affine.FORMAT.layout(k_, n_, bits_, gs_)

        def tile(self, layer_, k_, n_):
            return staged

        def decode(self, tensors_, bits_, gs_):
            return affine.FORMAT.decode(tensors_, bits_, gs_)

    fmt = _TabledAffine()
    layer = FormatLinear(
        fmt,
        4,
        64,
        k,
        {name: quantized[name] for name in ("weight", "scales", "biases")},
    )
    x = _x((rows, k))
    want = _one_expert(quantized, x)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        assert _same_bits(layer(x), want)


def test_direct_dense_respects_const_and_int_face_overloads(toy, monkeypatch):
    strict = (
        _TILE.replace(
            "Runs(Tile m, size_t row)",
            "Runs(thread Tile&, size_t) = delete;\n    Runs(const Tile m, size_t row)",
        )
        .replace(
            "P params(int) const {",
            "P params(int) = delete;\n    P params(int) const {",
        )
        .replace(
            "Codes fetch(int run) const {",
            "Codes fetch(int) = delete;\n    Codes fetch(int run) const {",
        )
        .replace(
            "    __OMLX_READ_WEIGHT__ {",
            "    static float weight(thread const P&, Codes, short) = delete;\n"
            "    __OMLX_READ_WEIGHT__ {",
        )
    )
    k, n = 256, 64
    fmt = toy(_tile_toy("strict_overloads", strict))
    tensors = {key: a[0] for key, a in _stack(n, k, seed=19).items()}
    layer = FormatLinear("tiletoy", 8, 1, k, tensors)
    x = _x((33, k), mx.float16)
    ids = mx.zeros((33,), mx.uint32)
    w = fmt.decode(tensors, 8, 1).astype(x.dtype)
    want = mx.gather_mm(x[:, None, :], w.T[None], rhs_indices=ids, sorted_indices=True)[
        :, 0
    ]
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        assert _same_bits(layer(x), want)


def test_direct_dense_enforces_contract_sentence_and_thread_safety(toy, caplog):
    k, n = 256, 64
    bad_src = _TILE.replace("static constant constexpr int CONTRACT = 1;", "")
    fmt = toy(_tile_toy("no_contract_dense", bad_src))
    tensors = {key: a[0] for key, a in _stack(n, k).items()}
    layer = FormatLinear("tiletoy", 8, 1, k, tensors)
    x = _x((33, k), mx.float16)
    want = x @ fmt.decode(tensors, 8, 1).astype(x.dtype).T
    with testing.fresh_proofs():
        assert _same_bits(layer(x), want)
    assert "CONTRACT" in caplog.text

    quantized, good = _layer(128, 512, 4, 64)
    x4 = _x((65, 512))
    expected = _one_expert(quantized, x4)
    mx.eval(expected, x4)
    ready = Barrier(4)

    def worker(_):
        ready.wait(timeout=30)
        return _same_bits(good(x4), expected)

    with testing.fresh_proofs(), ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(worker, range(4)))


@needs_steel
@pytest.mark.parametrize(
    "n,k,rows,split_k,bits,gs,cache_p",
    [
        (320, 10240, 17, 40, 8, 64, False),
        (320, 10240, 33, 20, 8, 64, False),
        (320, 10240, 128, 10, 8, 64, False),
        (48, 2560, 33, 40, 8, 64, False),
        (1, 2560, 512, 4, 8, 64, False),
        (64, 160, 33, 4, 4, 32, False),
        (64, 512, 33, 6, 8, 64, True),
    ],
    ids=[
        "hc_down_m17_sk40",
        "hc_down_m33_sk20",
        "hc_down_m128_sk10",
        "in_b_m33_sk40",
        "sh_exgate_m512_sk4",
        "ragged_k160_sk4_clamped",
        "non_group_aligned_kpart_cp",
    ],
)
def test_steel_direct_dense_split_k_matches_reference(
    monkeypatch, n, k, rows, split_k, bits, gs, cache_p
):
    quantized, layer = _layer(n, k, bits, gs)
    x = _x((1, rows, k))
    staging = (
        tuning.ParamStaging.GROUP_STEPS_COUNTER
        if cache_p
        else tuning.ParamStaging.PER_STEP
    )
    policy = tuning.DenseTilePolicy(
        "steel",
        32,
        32,
        32,
        2,
        2,
        n % 32 == 0,
        split_k=split_k,
        x_nreads=8,
        param_staging=staging,
    )
    seen = []
    real_k = tile_dense._kernel

    def observe(face, pol):
        mm = real_k(face, pol)

        def wrap(**kwargs):
            seen.append(
                (len(kwargs["inputs"]), kwargs["grid"][2], kwargs["output_dtypes"][0])
            )
            return mm(**kwargs)

        return wrap

    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: policy)
    monkeypatch.setattr(tile_dense, "_kernel", observe)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        got = layer(x)
    assert got.shape == (1, rows, n) and got.dtype == mx.bfloat16
    k_part = ((k // 32 + split_k - 1) // split_k) * 32
    expected_splits = (k + k_part - 1) // k_part
    assert seen and all(
        nin == 4 and gz == expected_splits and odt == mx.float32
        for nin, gz, odt in seen
    )
    stk = quantized(x)
    w32 = mx.dequantize(
        quantized["weight"],
        quantized["scales"].astype(mx.float32),
        quantized["biases"].astype(mx.float32),
        group_size=gs,
        bits=bits,
    )
    ref = x.astype(mx.float32) @ w32.T
    err_got = float(mx.abs(got.astype(mx.float32) - ref).max().item())
    err_stk = float(mx.abs(stk.astype(mx.float32) - ref).max().item())
    scale = max(1.0, float(mx.abs(ref).max().item()))
    assert err_got <= err_stk + 1e-3 * scale
    if (n, k, rows, split_k) == (48, 2560, 33, 40):
        assert _same_bits(got, stk)


@needs_steel
def test_steel_direct_dense_split_k_runs_tabled_tile(monkeypatch):
    from omlx import formats

    n, k, rows = 64, 512, 33
    quantized, _ = _layer(n, k, 4, 64)
    staged = tabled_tile(affine.tile_face(4, 64, mx.bfloat16))

    class _TabledSplitK(formats.QuantFormat):
        mode = "tabled_splitk_test"
        tensors = ("scales", "biases")
        contract = formats.CONTRACT

        def layout(self, k_, n_, bits_, gs_):
            return affine.FORMAT.layout(k_, n_, bits_, gs_)

        def tile(self, layer_, k_, n_):
            return staged

        def decode(self, tensors_, bits_, gs_):
            return affine.FORMAT.decode(tensors_, bits_, gs_)

    layer = FormatLinear(
        _TabledSplitK(),
        4,
        64,
        k,
        {name: quantized[name] for name in ("weight", "scales", "biases")},
    )
    x = _x((rows, k))
    policy = tuning.DenseTilePolicy(
        "steel", 32, 32, 32, 2, 2, True, split_k=3, x_nreads=8
    )
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: policy)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        got = layer(x)
    w32 = mx.dequantize(
        quantized["weight"],
        quantized["scales"].astype(mx.float32),
        quantized["biases"].astype(mx.float32),
        group_size=64,
        bits=4,
    )
    ref = x.astype(mx.float32) @ w32.T
    stk = quantized(x)
    err_got = float(mx.abs(got.astype(mx.float32) - ref).max().item())
    err_stk = float(mx.abs(stk.astype(mx.float32) - ref).max().item())
    scale = max(1.0, float(mx.abs(ref).max().item()))
    assert err_got <= err_stk + 1e-3 * scale


@needs_nax
@pytest.mark.parametrize(
    "n,k,rows,bm,bn,bk,wm,wn,split_k,sched,bits,gs,dtype",
    [
        (320, 10240, 17, 32, 64, 64, 2, 2, 8, "db", 8, 64, mx.bfloat16),
        (320, 10240, 33, 64, 64, 64, 2, 2, 8, "seg", 8, 64, mx.bfloat16),
        (512, 2560, 33, 64, 64, 64, 2, 2, 5, "seg", 8, 64, mx.bfloat16),
        (48, 2560, 128, 64, 64, 64, 2, 2, 10, "seg", 8, 64, mx.bfloat16),
        (1, 2560, 512, 64, 64, 64, 2, 2, 8, "seg", 8, 64, mx.float16),
        (96, 288, 33, 64, 64, 64, 2, 2, 3, "seg", 4, 32, mx.bfloat16),
        (128, 512, 33, 64, 64, 128, 2, 2, 2, "seg", 3, 64, mx.bfloat16),
        (128, 512, 33, 64, 64, 64, 2, 2, 4, "seg", 2, 64, mx.bfloat16),
    ],
    ids=[
        "hc_down_m17_db_sk8",
        "hc_down_m33_seg_sk8",
        "k_proj_m33_sk5",
        "in_b_m128_ragged_n_sk10",
        "exgate_m512_fp16_sk8",
        "ragged_nk_k288_sk3",
        "3b_bk128_sk2",
        "2b_sk4",
    ],
)
def test_nax_direct_dense_split_k_matches_reference(
    monkeypatch, n, k, rows, bm, bn, bk, wm, wn, split_k, sched, bits, gs, dtype
):
    quantized, layer = _layer(n, k, bits, gs, dtype)
    x = _x((1, rows, k), dtype)
    policy = tuning.DenseTilePolicy(
        "nax",
        bm,
        bn,
        bk,
        wm,
        wn,
        n % bn == 0,
        split_k=split_k,
        gx=0,
        pad_bytes=0,
        sched=sched,
    )
    seen = []
    real_k = tile_dense._kernel

    def observe(face, pol):
        mm = real_k(face, pol)

        def wrap(**kwargs):
            seen.append(
                (len(kwargs["inputs"]), kwargs["grid"][2], kwargs["output_dtypes"][0])
            )
            return mm(**kwargs)

        return wrap

    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: policy)
    monkeypatch.setattr(tile_dense, "_kernel", observe)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        got = layer(x)
    assert got.shape == (1, rows, n) and got.dtype == dtype
    total_bk = (k + bk - 1) // bk
    k_part = ((total_bk + split_k - 1) // split_k) * bk
    expected_splits = (k + k_part - 1) // k_part
    assert seen and all(
        nin == 4 and gz == wm * expected_splits and odt == mx.float32
        for nin, gz, odt in seen
    )
    stk = quantized(x)
    w32 = mx.dequantize(
        quantized["weight"],
        quantized["scales"].astype(mx.float32),
        quantized["biases"].astype(mx.float32),
        group_size=gs,
        bits=bits,
    )
    ref = x.astype(mx.float32) @ w32.T
    err_got = float(mx.abs(got.astype(mx.float32) - ref).max().item())
    err_stk = float(mx.abs(stk.astype(mx.float32) - ref).max().item())
    scale = max(1.0, float(mx.abs(ref).max().item()))
    assert err_got <= err_stk + 5e-3 * scale


def test_direct_dense_registers_native_fast_dispatch_and_respects_invalidation(
    monkeypatch,
):
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("native FormatLinearFast unavailable")
    quantized, layer = _layer(128, 512, 4, 64, bias=True)
    x = _x((1, 33, 512))
    want = quantized(x)
    with testing.fresh_proofs():
        y1 = layer(x)
        assert _same_bits(y1, want)
        assert layer._fast is not None
        y2 = layer(x)
        assert _same_bits(y2, want)
        # Weight mutation retires the native fast path cleanly.
        layer["weight"] = mx.array(layer["weight"])
        assert layer._fast is None
        assert _same_bits(layer(x), want)
        assert layer._fast is not None

    # MLX parity-default route also registers native fast dispatch.
    q_mlx, l_mlx = _layer(128, 512, 4, 64)
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: tuning.DenseTilePolicy("mlx"))
    with testing.fresh_proofs():
        assert _same_bits(l_mlx(x), q_mlx(x))
        assert l_mlx._fast is not None
        assert _same_bits(l_mlx(x), q_mlx(x))


def test_dense_proof_fail_retires_its_local_graphs(monkeypatch):
    n, k, rows = 64, 512, 33
    quantized, layer = _layer(n, k, 4, 64)
    x = mx.random.normal((rows, k), key=mx.random.key(12)).astype(mx.bfloat16)
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: _STEEL_POLICIES[1])
    plan = tile_dense.build_plan(layer, tensor_units=False)
    assert plan is not None
    first = tile_dense.project(plan, x)
    assert first is not None
    mx.eval(first)
    plan.proof.fail()
    assert tile_dense.project(plan, x) is None


def test_operand_binding_changes_the_proof_key():
    class _Plan:
        k = 8
        operands: tuple

    small, big = _Plan(), _Plan()
    small.operands = (mx.zeros((4,), mx.uint32),)
    big.operands = (mx.zeros((40,), mx.uint32),)
    x = mx.zeros((2, 8), mx.bfloat16)
    assert tile_dense._metal_key(small, x, _STEEL_POLICIES[1]) != tile_dense._metal_key(
        big, x, _STEEL_POLICIES[1]
    )
    zero_d, row = _Plan(), _Plan()
    zero_d.operands = (mx.zeros((), mx.uint32),)
    row.operands = (mx.zeros((8,), mx.uint32),)
    assert tile_dense._metal_key(
        zero_d, x, _STEEL_POLICIES[1]
    ) != tile_dense._metal_key(row, x, _STEEL_POLICIES[1])


def test_json_measured_span_subdivides_schedule_and_preserves_unbounded_tail(
    tmp_path, monkeypatch
):
    chip = "applegpu_g15d_c80"
    n, k = 256, 512
    _, layer = _layer(n, k, 4, 64)
    face = affine.tile_face(4, 64, mx.bfloat16)
    entry = {
        "key": {
            "route": "dense_tile",
            "face_id": face.key,
            "dtype": "bf16",
            "bits": 4,
            "group": 64,
            "n": n,
            "k": k,
            "rows_bucket": "44-52",
        },
        "face_sha256": tuning.face_digest(face),
        "skeleton_sha256": tuning.skeleton_digest("dense_tile"),
        "measured_at": "2026-11-22T11:30:00Z",
        "params": {"backend": "wide"},
        "exactness": "rule17_fp32",
    }
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / f"{chip}.json").write_text(
        json.dumps({"schema_version": 1, "chip": chip, "entries": [entry]})
    )
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    tuning.reset_cache()
    try:
        plan = tile_dense.build_plan(layer, tensor_units=False, chip=chip)
        assert plan is not None and plan.schedule[-1].hi is None
        assert tile_dense.policy_for(plan, 43, mx.bfloat16).backend == "steel"
        for r in range(44, 53):
            assert tile_dense.policy_for(plan, r, mx.bfloat16).backend == "wide"
        assert tile_dense.policy_for(plan, 53, mx.bfloat16).backend == "steel"
        assert tile_dense.policy_for(plan, 4096, mx.bfloat16).backend == "steel"
        assert tile_dense.policy_for(plan, 8192, mx.bfloat16).backend == "steel"
    finally:
        tuning.reset_cache()


def test_steel_weight_loader_guards_tile_and_group_split_in_source():
    assert "steel loader tile split" in tile_dense._STEEL_HEADER
    assert "steel loader group split" in tile_dense._STEEL_HEADER
    assert "steel loader column tile" in tile_dense._STEEL_HEADER
    assert "steel loader row tile" in tile_dense._STEEL_HEADER
    assert "steel dense split_k" in tile_dense._STEEL_HEADER
    assert "nax dense: tile_matmad_nax TN/TM parity" in tile_dense._NAX_HEADER


@needs_steel
def test_steel_weight_loader_refuses_underfilled_tile_split(monkeypatch, caplog):
    # BN=16, BK=32, WM=4, WN=1 -> TGP=128 -> values = (16*32)/128 = 4 < RUN=8.
    # Without the static_assert, runs_per_group truncates to 0 and skips loads.
    bad = tuning.DenseTilePolicy("steel", 64, 16, 32, 4, 1, True, x_nreads=8)
    n, k, rows = 64, 512, 64
    quantized, layer = _layer(n, k, 4, 64)
    x = _x((rows, k))
    face = affine.tile_face(4, 64, mx.bfloat16)
    ops = (quantized["weight"], quantized["scales"], quantized["biases"])
    mm = tile_dense._kernel(face, bad)
    with pytest.raises(RuntimeError, match="steel loader group split"):
        mx.eval(tile_dense._launch(mm, bad, x, ops))

    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: bad)
    want = quantized(x)
    with testing.fresh_proofs():
        got = layer(x)
    assert _same_bits(got, want)
    assert "steel loader group split" in caplog.text


@needs_steel
def test_steel_block_loader_refuses_bm_not_multiple_of_trows():
    # TGP=128, BK=32, x_nreads=8 -> TCOLS=4 -> TROWS=32; BM=16 % 32 != 0 OOBs xs.
    bad = tuning.DenseTilePolicy("steel", 16, 64, 32, 2, 2, True, x_nreads=8)
    n, k, rows = 64, 512, 32
    quantized, _ = _layer(n, k, 4, 64)
    x = _x((rows, k))
    face = affine.tile_face(4, 64, mx.bfloat16)
    ops = (quantized["weight"], quantized["scales"], quantized["biases"])
    mm = tile_dense._kernel(face, bad)
    with pytest.raises(RuntimeError, match="steel loader row tile"):
        mx.eval(tile_dense._launch(mm, bad, x, ops))


@needs_steel
def test_steel_block_loader_refuses_bk_not_multiple_of_nreads():
    # BK=32, x_nreads=6 -> TCOLS=5, TCOLS*n_reads=30 != BK=32 (truncated cols).
    bad = tuning.DenseTilePolicy("steel", 32, 32, 32, 2, 2, True, x_nreads=6)
    n, k, rows = 64, 512, 32
    quantized, _ = _layer(n, k, 4, 64)
    x = _x((rows, k))
    face = affine.tile_face(4, 64, mx.bfloat16)
    ops = (quantized["weight"], quantized["scales"], quantized["biases"])
    mm = tile_dense._kernel(face, bad)
    with pytest.raises(RuntimeError, match="steel loader column tile"):
        mx.eval(tile_dense._launch(mm, bad, x, ops))


@needs_steel
def test_steel_split_k_refuses_k_or_kpart_not_multiple_of_bk():
    n, k, rows = 64, 512, 32
    quantized, _ = _layer(n, k, 4, 64)
    x = _x((rows, k))
    face = affine.tile_face(4, 64, mx.bfloat16)
    ops = (quantized["weight"], quantized["scales"], quantized["biases"])
    mm = tile_dense._kernel(face, tuning.DenseTilePolicy("steel", 32, 32, 32))
    # K_PART=48 is not a multiple of BK=32 under SPLIT_K=2.
    with pytest.raises(RuntimeError, match="steel dense split_k"):
        mx.eval(
            mm(
                inputs=[x, *ops],
                template=[
                    ("T", mx.bfloat16),
                    ("K", k),
                    ("BM", 32),
                    ("BN", 32),
                    ("BK", 32),
                    ("WM", 2),
                    ("WN", 2),
                    ("ALIGN_N", True),
                    ("VEC_STORE", False),
                    ("CACHE_P", False),
                    ("X_NREADS", 8),
                    ("SPLIT_K", 2),
                    ("K_PART", 48),
                ],
                grid=(128 * 2, 1, 2),
                threadgroup=(128, 1, 1),
                output_shapes=[(2, rows, n)],
                output_dtypes=[mx.float32],
            )[0]
        )
    # K=544 with BK=64 (K % BK == 32 != 0) under SPLIT_K=2.
    q_odd, _ = _layer(n, 544, 4, 32)
    x_odd = _x((rows, 544))
    face32 = affine.tile_face(4, 32, mx.bfloat16)
    ops32 = (q_odd["weight"], q_odd["scales"], q_odd["biases"])
    mm32 = tile_dense._kernel(face32, tuning.DenseTilePolicy("steel", 32, 32, 64))
    with pytest.raises(RuntimeError, match="steel dense split_k"):
        mx.eval(
            mm32(
                inputs=[x_odd, *ops32],
                template=[
                    ("T", mx.bfloat16),
                    ("K", 544),
                    ("BM", 32),
                    ("BN", 32),
                    ("BK", 64),
                    ("WM", 2),
                    ("WN", 2),
                    ("ALIGN_N", True),
                    ("VEC_STORE", False),
                    ("CACHE_P", False),
                    ("X_NREADS", 4),
                    ("SPLIT_K", 2),
                    ("K_PART", 320),
                ],
                grid=(128 * 2, 1, 2),
                threadgroup=(128, 1, 1),
                output_shapes=[(2, rows, n)],
                output_dtypes=[mx.float32],
            )[0]
        )


@needs_nax
def test_nax_refuses_illegal_tn_tm_parity_and_runs_tn1_tm2(monkeypatch):
    n, k = 128, 512
    quantized, layer = _layer(n, k, 4, 64)
    face = affine.tile_face(4, 64, mx.bfloat16)
    ops = (quantized["weight"], quantized["scales"], quantized["biases"])
    x = _x((64, k))
    # (TM=1, TN=1) and (TM=2, TN=3): tile_matmad_nax silently no-ops without guard.
    for bm, bn in ((32, 32), (64, 96)):
        bad = tuning.DenseTilePolicy("nax", bm, bn, 64, 2, 2, False)
        mm = tile_dense._kernel(face, bad)
        with pytest.raises(RuntimeError, match="tile_matmad_nax TN/TM parity"):
            mx.eval(tile_dense._launch(mm, bad, x, ops))
    # Legal (TM=2, TN=1): BM=64, BN=32, WM=2, WN=2 must match bitwise aligned & tail.
    good = tuning.DenseTilePolicy("nax", 64, 32, 64, 2, 2, True)
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: good)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for rows in (64, 33):
        xr = _x((rows, k), seed=rows)
        with testing.fresh_proofs():
            assert _same_bits(layer(xr), _one_expert(quantized, xr))


@needs_steel
@pytest.mark.parametrize(
    "policy,rows",
    [
        (
            tuning.DenseTilePolicy(
                "steel", 16, 16, 128, 2, 2, True, x_nreads=8, vec_store=True
            ),
            33,
        ),
        (
            tuning.DenseTilePolicy("steel", 32, 16, 128, 2, 2, True, x_nreads=8),
            64,
        ),
        (
            tuning.DenseTilePolicy("steel", 32, 16, 64, 2, 1, True, x_nreads=8),
            65,
        ),
        (
            tuning.DenseTilePolicy(
                "steel", 32, 32, 64, 2, 2, True, x_nreads=4, vec_store=True
            ),
            65,
        ),
    ],
    ids=["16x16x128_vs", "32x16x128", "32x16x64_w2x1", "32x32x64_nr4_vs"],
)
def test_steel_bk64_bk128_dense_policies_match_reference(monkeypatch, policy, rows):
    n, k = 128, 512
    quantized, layer = _layer(n, k, 4, 64)
    x = _x((rows, k))
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: policy)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        got = layer(x)
    stk = quantized(x)
    w32 = mx.dequantize(
        quantized["weight"],
        quantized["scales"].astype(mx.float32),
        quantized["biases"].astype(mx.float32),
        group_size=64,
        bits=4,
    )
    ref = x.astype(mx.float32) @ w32.T
    err_got = float(mx.abs(got.astype(mx.float32) - ref).max().item())
    err_stk = float(mx.abs(stk.astype(mx.float32) - ref).max().item())
    scale = max(1.0, float(mx.abs(ref).max().item()))
    assert err_got <= err_stk + 1e-3 * scale


@needs_nax
def test_nax_128x32x128_dense_policy_matches_mlx_bitwise(monkeypatch):
    n, k, rows = 128, 512, 129
    policy = tuning.DenseTilePolicy(
        "nax", 128, 32, 128, 4, 1, True, gx=32, pad_bytes=0, sched="seg"
    )
    quantized, layer = _layer(n, k, 4, 64)
    x = _x((rows, k))
    want = _one_expert(quantized, x)
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: policy)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    with testing.fresh_proofs():
        got = layer(x)
    assert _same_bits(got, want)


def test_in_tree_tuning_tables_serve_m3_m5_gate2_shapes_and_44_52_rows_range():
    tuning.reset_cache()
    # M3 Ultra (applegpu_g15d_c80): Flash-Next 2560x2560 4b g64 includes the
    # 44-52 rows range between 16x16x128 (33..43) and 32x16x128 (53..64).
    _, flash = _layer(2560, 2560, 4, 64)
    flash_m3 = tile_dense.build_plan(
        flash, tensor_units=False, chip="applegpu_g15d_c80"
    )
    assert flash_m3 is not None
    p43 = tile_dense.policy_for(flash_m3, 43, mx.bfloat16)
    assert (p43.backend, p43.bm, p43.bn, p43.bk, p43.vec_store) == (
        "steel",
        16,
        16,
        128,
        True,
    )
    for r in (44, 48):
        pr = tile_dense.policy_for(flash_m3, r, mx.bfloat16)
        assert (pr.backend, pr.bm, pr.bn, pr.bk) == ("steel", 16, 16, 128)
    for r in (49, 52):
        pr = tile_dense.policy_for(flash_m3, r, mx.bfloat16)
        assert (pr.backend, pr.bm, pr.bn, pr.bk) == ("steel", 32, 16, 128)
    for r in (53, 64):
        pr = tile_dense.policy_for(flash_m3, r, mx.bfloat16)
        assert (pr.backend, pr.bm, pr.bn, pr.bk) == ("steel", 32, 16, 128)
    for r in (128, 512, 4096):
        pr = tile_dense.policy_for(flash_m3, r, mx.bfloat16)
        assert (pr.backend, pr.bm, pr.bn, pr.bk) == ("steel", 64, 32, 32)

    # M3 Ultra 35B 2048x2048 4b g64 + M5 Max 2048x2048 4b g64 M=512 (BN=32).
    _, s35b = _layer(2048, 2048, 4, 64)
    plan_m3 = tile_dense.build_plan(s35b, tensor_units=False, chip="applegpu_g15d_c80")
    assert tile_dense.policy_for(plan_m3, 33, mx.bfloat16).backend == "mlx"
    assert tile_dense.policy_for(plan_m3, 128, mx.bfloat16).backend == "mlx"
    p256 = tile_dense.policy_for(plan_m3, 256, mx.bfloat16)
    assert (p256.backend, p256.bm, p256.bn, p256.bk, p256.wn) == (
        "steel",
        32,
        16,
        64,
        1,
    )
    p512 = tile_dense.policy_for(plan_m3, 512, mx.bfloat16)
    assert (p512.backend, p512.bm, p512.bn, p512.bk, p512.x_nreads, p512.vec_store) == (
        "steel",
        32,
        32,
        64,
        4,
        True,
    )
    plan_m5 = tile_dense.build_plan(s35b, tensor_units=True, chip="applegpu_g17s_c40")
    p512_m5 = tile_dense.policy_for(plan_m5, 512, mx.bfloat16)
    assert (p512_m5.backend, p512_m5.bm, p512_m5.bn, p512_m5.bk, p512_m5.gx) == (
        "nax",
        64,
        32,
        64,
        32,
    )


def _assert_c2(got, want, x, quantized):
    weight = mx.dequantize(
        quantized["weight"],
        quantized["scales"].astype(mx.float32),
        quantized["biases"].astype(mx.float32),
        group_size=quantized.group_size,
        bits=quantized.bits,
    )
    ref = x.astype(mx.float32) @ weight.T.astype(mx.float32)

    def rel(a):
        return (
            mx.sqrt(mx.sum((a.astype(mx.float32) - ref) ** 2)) / mx.sqrt(mx.sum(ref**2))
        ).item()

    got_err, want_err = rel(got), rel(want)
    assert got_err <= 1.05 * want_err + 1e-6, (got_err, want_err)


def test_flash_next_all_affine_prefill_shapes_route_at_construction_without_decode(
    monkeypatch,
):
    tuning.reset_cache()
    # Every 2-D affine prefill shape in Qwen3.8-Flash-Next-oQ4e-fp8ple
    # (including N=48 in_proj_a/b and N=1 shared_expert_gate).
    fn_shapes = (
        (1, 2560, 8, 64),
        (4, 10240, 8, 64),
        (48, 2560, 8, 64),
        (320, 10240, 8, 64),
        (512, 2560, 8, 64),
        (640, 2560, 8, 64),
        (640, 2560, 8, 128),
        (2560, 640, 8, 128),
        (2560, 2560, 8, 64),
        (2560, 6144, 8, 64),
        (6144, 2560, 8, 64),
        (10240, 320, 8, 64),
        (10240, 2560, 8, 64),
        (12288, 2560, 8, 64),
    )
    probe_rows = (16, 24, 32, 33, 48, 64, 128, 256, 512, 1024, 2048, 4096)
    for n, k, bits, gs in fn_shapes:
        _, layer = _layer(n, k, bits, gs)
        for chip, tu in (("applegpu_g15d_c80", False), ("applegpu_g17s_c40", True)):
            plan = tile_dense.build_plan(layer, tensor_units=tu, chip=chip)
            assert plan is not None, f"no dense plan for {(n, k, bits, gs)} on {chip}"
            for r in probe_rows:
                pol = tile_dense.policy_for(plan, r, mx.bfloat16)
                assert pol is not None
                assert pol.is_legal(
                    plan.face, k, n, r, chip, dtype=mx.bfloat16
                ), f"illegal policy {pol} for {(n, k, bits, gs)} M={r} on {chip}"

    # Verify N=48 and N=1 get measured routes on both chips and execute on this
    # host without ever touching FormatLinear._decoded.
    _, in_proj = _layer(48, 2560, 8, 64)
    _, gate = _layer(1, 2560, 8, 64)
    p48_m3 = tile_dense.build_plan(
        in_proj, tensor_units=False, chip="applegpu_g15d_c80"
    )
    assert tile_dense.policy_for(p48_m3, 17, mx.bfloat16).backend == "mlx"
    assert tile_dense.policy_for(p48_m3, 18, mx.bfloat16).split_k == 40
    assert tile_dense.policy_for(p48_m3, 24, mx.bfloat16).split_k == 40
    assert tile_dense.policy_for(p48_m3, 33, mx.bfloat16).backend == "mlx"
    assert tile_dense.policy_for(p48_m3, 128, mx.bfloat16).split_k == 40
    assert tile_dense.policy_for(p48_m3, 256, mx.bfloat16).backend == "mlx"
    p1_m3 = tile_dense.build_plan(gate, tensor_units=False, chip="applegpu_g15d_c80")
    assert tile_dense.policy_for(p1_m3, 17, mx.bfloat16).backend == "mlx"
    assert tile_dense.policy_for(p1_m3, 18, mx.bfloat16).split_k == 40
    assert tile_dense.policy_for(p1_m3, 64, mx.bfloat16).split_k == 40
    assert tile_dense.policy_for(p1_m3, 512, mx.bfloat16).split_k == 20
    _, o_proj = _layer(2560, 6144, 8, 64)
    p_o = tile_dense.build_plan(o_proj, tensor_units=False, chip="applegpu_g15d_c80")
    for r in (17, 20, 24):
        assert tile_dense.policy_for(p_o, r, mx.bfloat16).backend == "mlx"
    _, o_proj_fp16 = _layer(2560, 6144, 8, 64, mx.float16)
    p_o16 = tile_dense.build_plan(
        o_proj_fp16, tensor_units=False, chip="applegpu_g15d_c80"
    )
    for r in (17, 20, 24):
        assert tile_dense.policy_for(p_o16, r, mx.float16).backend == "steel"
        assert tile_dense.policy_for(p_o16, r, mx.float16).split_k == 12
    p48_m5 = tile_dense.build_plan(in_proj, tensor_units=True, chip="applegpu_g17s_c40")
    assert tile_dense.policy_for(p48_m5, 32, mx.bfloat16).backend == "wide"
    assert tile_dense.policy_for(p48_m5, 128, mx.bfloat16).split_k == 40
    assert tile_dense.policy_for(p48_m5, 512, mx.bfloat16).split_k == 10
    p1_m5 = tile_dense.build_plan(gate, tensor_units=True, chip="applegpu_g17s_c40")
    assert tile_dense.policy_for(p1_m5, 64, mx.bfloat16).backend == "mlx"
    assert tile_dense.policy_for(p1_m5, 1024, mx.bfloat16).split_k == 40

    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)

    for n, k in ((48, 2560), (1, 2560), (4, 10240)):
        for dtype in (mx.bfloat16, mx.float16):
            quantized, layer = _layer(n, k, 8, 64, dtype)
            for rows in (17, 18, 24, 32, 33, 48, 64, 128, 256, 512, 1024):
                x = _x((rows, k), dtype=dtype, seed=rows + n)
                with testing.fresh_proofs():
                    got = layer(x)
                want = quantized(x)
                mx.eval(got, want)
                kept = layer.__dict__.get(linear._KEPT)
                assert kept is not None and kept.dense not in (None, linear._UNBUILT)
                policy = tile_dense.policy_for(kept.dense, rows, x.dtype)
                assert policy is not None
                if policy.backend == "mlx":
                    assert mx.array_equal(
                        got.view(mx.uint16), want.view(mx.uint16)
                    ).item(), (n, dtype, rows)
                else:
                    _assert_c2(got, want, x, quantized)


def test_unarmed_qmm_threshold_12_to_16_uses_scheduled_route_meeting_c2(monkeypatch):
    # For width > 4096 at 12..16 rows, FormatLinear runs its construction-time
    # schedule (2-D and single-batch 3-D): bitwise when the route is mlx, C2
    # fp32_ref when the measured DATA selects a reordered tile (review-3 H5).
    tuning.reset_cache()
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for n, k in ((10240, 2560), (6144, 2560)):
        quantized, layer = _layer(n, k, 8, 64)
        for rows in (12, 16):
            for shape in ((rows, k), (1, rows, k)):
                x = _x(shape, seed=rows + n + len(shape))
                with testing.fresh_proofs():
                    got = layer(x)
                want = quantized(x)
                mx.eval(got, want)
                kept = layer.__dict__.get(linear._KEPT)
                assert kept is not None and kept.dense not in (None, linear._UNBUILT)
                route = tile_dense.route_for(kept.dense, rows, x.dtype)
                if route.backend == "mlx":
                    assert _same_bits(got, want), f"mismatch on {n}x{k} shape={shape}"
                else:
                    _assert_c2(got, want, x, quantized)


@needs_nax
def test_nax_refuses_klt_exceeding_tgp():
    # kTGP = 1 * 2 * 32 = 64 < kLT = BN = 128 would leave Ws[64..127] uninitialized.
    bad = tuning.DenseTilePolicy("nax", 64, 128, 64, 1, 2, True)
    n, k, rows = 128, 512, 64
    quantized, _ = _layer(n, k, 4, 64)
    x = _x((rows, k))
    face = affine.tile_face(4, 64, mx.bfloat16)
    ops = (quantized["weight"], quantized["scales"], quantized["biases"])
    mm = tile_dense._kernel(face, bad)
    with pytest.raises(RuntimeError, match="loader threads exceed threadgroup size"):
        mx.eval(tile_dense._launch(mm, bad, x, ops))


@pytest.mark.parametrize(
    "policy,n,rows",
    [
        (tuning.DenseTilePolicy("nax", 64, 128, 64, 2, 2, True), 128, 64),
        (tuning.DenseTilePolicy("nax", 32, 32, 64, 1, 1, True), 64, 32),
        (tuning.DenseTilePolicy("nax", 48, 64, 64, 3, 1, True), 64, 48),
        (tuning.DenseTilePolicy("nax", 64, 192, 32, 4, 2, True), 192, 64),
    ],
)
def test_nax_klt_equals_bn_is_refused_at_construction(policy, n, rows):
    k = 512
    face = affine.tile_face(4, 64, mx.bfloat16)
    assert not policy.is_legal(face, k, n, rows, "applegpu_g17s_c40", mx.bfloat16)


def test_resolve_cell_falls_back_on_klt_equals_bn(monkeypatch):
    tuning.reset_cache()
    for refused, n in (
        (tuning.DenseTilePolicy("nax", 64, 128, 64, wm=1, wn=2), 128),
        (tuning.DenseTilePolicy("nax", 32, 32, 64, wm=1, wn=1), 64),
        (tuning.DenseTilePolicy("nax", 48, 64, 64, wm=3, wn=1), 64),
        (tuning.DenseTilePolicy("nax", 64, 192, 32, wm=4, wn=2), 192),
    ):
        _, layer = _layer(n, 512, 4, 64)
        monkeypatch.setattr(tuning, "lookup", lambda *a, _p=refused, **k: _p)
        plan = tile_dense.build_plan(layer, tensor_units=True, chip="applegpu_g17s_c40")
        assert plan is not None
        for route in plan.schedule:
            pol = route.policy
            if pol is None:
                continue
            assert pol.backend != "nax" or pol.wm * pol.wn * 32 >= 2 * pol.bn, (
                refused,
                pol,
            )

    legal_boundary = tuning.DenseTilePolicy("nax", 32, 64, 64, wm=2, wn=2, sched="seg")
    _, layer = _layer(64, 512, 4, 64)
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: legal_boundary)
    plan = tile_dense.build_plan(layer, tensor_units=True, chip="applegpu_g17s_c40")
    assert plan is not None
    assert tile_dense.policy_for(plan, 512, mx.bfloat16) == legal_boundary


def test_default_policy_never_emits_an_illegal_nax_cell():
    tuning.reset_cache()
    ns = list(range(1, 64)) + list(range(64, 1025, 16))
    for bits in (2, 3, 4, 8):
        face = affine.tile_face(bits, 64, mx.bfloat16)
        for n in ns:
            _, layer = _layer(n, 512, bits, 64)
            for chip, tensor_units in (
                ("applegpu_g15d_c80", False),
                ("applegpu_g17s_c40", True),
            ):
                plan = tile_dense.build_plan(
                    layer, tensor_units=tensor_units, chip=chip
                )
                assert plan is not None, (chip, bits, n)
                for rows in (1, 17, 33, 512):
                    for route in plan.schedule:
                        pol = route.policy
                        if pol is None:
                            continue
                        assert pol.is_legal(face, 512, n, rows, chip, mx.bfloat16), (
                            chip,
                            bits,
                            n,
                            rows,
                            pol,
                        )


def test_dense_nax_barriers_are_threadgroup_uniform():
    assert "dispatch_bool(" not in tile_dense._NAX_HEADER
    assert (
        "const bool sg_full = sg_active && (sgp_sm == kSM);" in tile_dense._NAX_HEADER
    )


def test_build_plan_rejects_nax_table_entry_when_tensor_units_false():
    tuning.reset_cache()
    _, layer = _layer(2048, 2048, 4, 64)
    plan = tile_dense.build_plan(layer, tensor_units=False, chip="applegpu_g17s_c40")
    assert plan is not None
    pol = tile_dense.policy_for(plan, 512, mx.bfloat16)
    assert pol is not None and pol.backend == "steel"


def test_c2_helper_rejects_error_above_1_05x_mlx():
    quantized, _ = _layer(48, 2560, 8, 64)
    x = _x((33, 2560), seed=33)
    want = quantized(x)
    weight = mx.dequantize(
        quantized["weight"],
        quantized["scales"].astype(mx.float32),
        quantized["biases"].astype(mx.float32),
        group_size=quantized.group_size,
        bits=quantized.bits,
    )
    ref = x.astype(mx.float32) @ weight.T.astype(mx.float32)
    mx.eval(want, ref)
    _assert_c2(want, want, x, quantized)
    near_bad = ref + 1.2 * (want.astype(mx.float32) - ref)
    biased = (want.astype(mx.float32) + 0.05).astype(mx.bfloat16)
    mx.eval(near_bad, biased)
    with pytest.raises(AssertionError):
        _assert_c2(near_bad, want, x, quantized)
    with pytest.raises(AssertionError):
        _assert_c2(biased, want, x, quantized)


def test_narrow_n_judges_the_plan_the_layer_built_on_an_untuned_chip(monkeypatch):
    monkeypatch.setattr(linear, "_nax_host", lambda: False)
    monkeypatch.setattr(tuning, "current_chip", lambda: "applegpu_g16s_c16")
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    tuning.reset_cache()
    try:
        quantized, layer = _layer(48, 2560, 8, 64)
        x = _x((33, 2560))
        with testing.fresh_proofs():
            got = layer(x)
        want = quantized(x)
        mx.eval(got, want)
        kept = layer.__dict__.get(linear._KEPT)
        assert kept is not None and kept.dense not in (None, linear._UNBUILT)
        policy = tile_dense.policy_for(kept.dense, 33, x.dtype)
        assert policy is not None and policy.backend == "steel"
        _assert_c2(got, want, x, quantized)
    finally:
        tuning.reset_cache()


def test_project_returns_none_for_an_unknown_backend(monkeypatch):
    tuning.reset_cache()
    monkeypatch.setattr(tuning, "current_chip", lambda: "applegpu_g15s_c20")
    monkeypatch.setattr(linear, "_metal_architecture", lambda: "applegpu_g15s")
    _, layer = _layer(64, 512, 4, 64)
    other_pol = tuning.DenseTilePolicy("other", 0, 0, 0)
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: other_pol)
    plan = tile_dense.build_plan(layer, tensor_units=False, chip="applegpu_g15d_c80")
    assert plan is not None
    assert tile_dense.policy_for(plan, 48, mx.bfloat16).backend == "other"
    assert tile_dense.project(plan, _x((48, 512))) is None


def test_build_plan_fails_closed_on_an_empty_chip():
    _, layer = _layer(64, 512, 4, 64)
    with pytest.raises(ValueError, match="chip"):
        tile_dense.build_plan(layer, tensor_units=False, chip="")


def test_build_plan_uses_the_target_chip_table_not_the_host_arch(monkeypatch):
    tuning.reset_cache()
    _, layer = _layer(2560, 2560, 4, 64)
    monkeypatch.setattr(linear, "_metal_architecture", lambda: "applegpu_g15s")
    plan = tile_dense.build_plan(layer, tensor_units=False, chip="applegpu_g15d_c80")
    assert plan is not None
    pr = tile_dense.policy_for(plan, 48, mx.bfloat16)
    assert (pr.backend, pr.bm, pr.bn, pr.bk) == ("steel", 16, 16, 128)


def _rows_17_24_shapes():
    base = Path(__file__).parents[1] / "omlx" / "formats" / "tuning" / "v1"
    paths = sorted(base.glob("applegpu_*.json"))
    assert paths
    shapes = set()
    for path in paths:
        with open(path) as table:
            entries = json.load(table)["entries"]
        for entry in entries:
            key = entry["key"]
            if key.get("route") == "dense_tile" and "n" in key and "k" in key:
                shapes.add((key["n"], key["k"], key["bits"], key["group"]))
    for n, k in (
        (12288, 2560),
        (512, 2560),
        (2560, 6144),
        (10240, 2560),
        (6144, 2560),
        (640, 2560),
        (2560, 640),
        (2560, 2560),
        (4, 10240),
    ):
        shapes.add((n, k, 8, 64))
    return sorted(shapes)


def _wide_window_top(n, k, bits, group):
    """The last row MLX runs qmv_wide for this shape on the host chip."""
    arch = linear._metal_architecture() or ""
    top = 16
    for rows in range(17, 65):
        folded = mlx_traversal(rows, n, k, group, arch, bits=bits)
        if folded[0] == "wide":
            top = rows
    return top


@pytest.mark.parametrize("n,k,bits,group", _rows_17_24_shapes())
def test_rows_17_24_passes_c2_on_the_host_table(n, k, bits, group):
    # The MLX reference is the host's own route, so each Mac judges its table:
    # M5 Max -> applegpu_g17s_c40, M3 Ultra -> applegpu_g15d_c80.
    if linear._nax_host():
        chip, tensor_units = "applegpu_g17s_c40", True
    else:
        chip, tensor_units = "applegpu_g15d_c80", False
    top = _wide_window_top(n, k, bits, group)
    probe = range(17, top + 2) if top >= 17 else (17, 18, 24)
    for dtype in (mx.bfloat16, mx.float16):
        quantized, layer = _layer(n, k, bits, group, dtype)
        plan = tile_dense.build_plan(layer, tensor_units=tensor_units, chip=chip)
        assert plan is not None, (chip, n, k, bits, group, dtype)
        for rows in probe:
            x = _x((rows, k), dtype=dtype, seed=rows + n)
            got = tile_dense.project(plan, x)
            assert got is not None, (chip, n, k, bits, group, dtype, rows)
            want = quantized(x)
            mx.eval(got, want)
            _assert_c2(got, want, x, quantized)


def test_m3_wide_rows17_takes_mlx_and_passes_c2():
    # applegpu_g15d's QMV limit is 18 for width 2560, so MLX runs qmv_wide at
    # rows=17; the 17-17 span takes mlx and must be bitwise to stock, and
    # row 18 keeps the parity-default steel route. The row-18 C2 check is
    # gated to a non-NAX host: on a NAX host the C2 reference is that host's
    # own MLX route, not the one this table was tuned against (the host
    # sweep above judges the M5's own table), but the routing assertions and
    # row-17 parity are checked everywhere.
    probe = (17,) if linear._nax_host() else (17, 18)
    for n, k, bits in (
        (640, 2560, 8),
        (2560, 2560, 8),
        (2560, 2560, 4),
        (2560, 640, 8),
    ):
        for dtype in (mx.bfloat16, mx.float16):
            quantized, layer = _layer(n, k, bits, 64, dtype)
            plan = tile_dense.build_plan(
                layer, tensor_units=False, chip="applegpu_g15d_c80"
            )
            assert plan is not None
            assert tile_dense.route_for(plan, 17, dtype).backend == "mlx"
            if (n, k, bits) == (2560, 2560, 8):
                # Explicit 18-24 entry takes mlx (steel default measured 1.58x slower).
                assert tile_dense.route_for(plan, 18, dtype).backend == "mlx"
            else:
                assert tile_dense.route_for(plan, 18, dtype).backend != "mlx"
            for rows in probe:
                x = _x((rows, k), dtype=dtype, seed=n + k + bits + rows)
                got = tile_dense.project(plan, x)
                want = quantized(x)
                mx.eval(got, want)
                if rows == 17:
                    assert mx.array_equal(
                        got.view(mx.uint16), want.view(mx.uint16)
                    ).item()
                _assert_c2(got, want, x, quantized)


def test_m3_wide_rows17_checks_policy_and_row17_on_nax(monkeypatch):
    """On a NAX host the g15d routing (rows 17/18 routes) and row-17
    bitwise parity are still checked; only row 18's C2 is host-gated."""
    monkeypatch.setattr(linear, "_nax_host", lambda: True)
    seen: list[int] = []
    real_route = tile_dense.route_for

    def spy_route(plan, rows, dtype):
        seen.append(rows)
        return real_route(plan, rows, dtype)

    monkeypatch.setattr(tile_dense, "route_for", spy_route)
    test_m3_wide_rows17_takes_mlx_and_passes_c2()
    assert 17 in seen and 18 in seen


def test_narrow_n_c2_evaluates_all_rows(monkeypatch):
    seen: list[tuple[int, int, mx.Dtype]] = []
    real_eval = mx.eval

    def spy_eval(*args, **kwargs):
        if len(args) == 2 and all(isinstance(a, mx.array) for a in args):
            got, want = args
            if got.ndim == 2 and got.shape == want.shape:
                seen.append((got.shape[0], got.shape[1], got.dtype))
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", spy_eval)
    test_flash_next_all_affine_prefill_shapes_route_at_construction_without_decode(
        monkeypatch
    )
    expected = {
        (rows, n, dtype)
        for n in (48, 1, 4)
        for dtype in (mx.bfloat16, mx.float16)
        for rows in (17, 18, 24, 32, 33, 48, 64, 128, 256, 512, 1024)
    }
    assert len(seen) == 66, f"expected 66 row evaluations, got {len(seen)}"
    assert set(seen) == expected

    # Prove an early-row mismatch (rows=17, previously skipped) actually fails.
    real_call = FormatLinear.__call__

    def corrupt_row_17(self, x):
        out = real_call(self, x)
        if x.shape[0] == 17:
            return out + mx.array(1.0, dtype=out.dtype)
        return out

    monkeypatch.setattr(mx, "eval", real_eval)
    monkeypatch.setattr(FormatLinear, "__call__", corrupt_row_17)
    with pytest.raises(AssertionError):
        test_flash_next_all_affine_prefill_shapes_route_at_construction_without_decode(
            monkeypatch
        )


def test_host_sweep_covers_wide_window_and_fp16(monkeypatch):
    """Every row of the host's qmv_wide window is C2-checked, plus the first
    non-mlx row past it (top + 1), which must project too."""
    seen: set[tuple[int, mx.Dtype]] = set()
    real_project = tile_dense.project

    def spy_project(plan, x, *args, **kwargs):
        seen.add((x.shape[0], x.dtype))
        return real_project(plan, x, *args, **kwargs)

    monkeypatch.setattr(tile_dense, "project", spy_project)
    test_rows_17_24_passes_c2_on_the_host_table(2048, 2048, 4, 64)
    top = _wide_window_top(2048, 2048, 4, 64)
    assert top >= 31, f"expected 2048x2048 wide window top >= 31, got {top}"
    expected = {
        (rows, dtype)
        for rows in range(17, top + 2)
        for dtype in (mx.bfloat16, mx.float16)
    }
    assert seen == expected

    # Verify _nax_host() (not gpu_generation() >= 17) selects the chip/tensor_units.
    class _StopError(Exception):
        pass

    recorded: list[tuple[str | None, bool | None]] = []

    def spy_build_plan(layer, *, tensor_units=None, chip=None, **kwargs):
        recorded.append((chip, tensor_units))
        raise _StopError()

    monkeypatch.setattr(tile_dense, "build_plan", spy_build_plan)
    monkeypatch.setattr(kernels, "gpu_generation", lambda: 17)
    monkeypatch.setattr(linear, "_nax_host", lambda: False)
    with pytest.raises(_StopError):
        test_rows_17_24_passes_c2_on_the_host_table(2048, 2048, 4, 64)
    monkeypatch.setattr(kernels, "gpu_generation", lambda: 15)
    monkeypatch.setattr(linear, "_nax_host", lambda: True)
    with pytest.raises(_StopError):
        test_rows_17_24_passes_c2_on_the_host_table(2048, 2048, 4, 64)
    assert recorded == [("applegpu_g15d_c80", False), ("applegpu_g17s_c40", True)]


def test_dense_splitk_accum_uses_single_fused_kernel_not_two_pass(monkeypatch):
    """Split-K dense prefill reduces in one fused Metal pass (not mx.sum + astype),
    handling arbitrary shapes, dtypes, tiny (<8 elem) buffers, empty tensors,
    and IDX64 parity with exact ascending FP32 partition order."""
    accum_k = tile_dense._splitk_accum_kernel()
    assert accum_k is not None

    # Direct kernel parity: ascending FP32 sum cast once to T across split_k,
    # dtypes, prime/tiny totals (<8 elements), and 3-D/4-D output shapes.
    cases = [
        (2, (2, 1), mx.bfloat16),  # total*sk = 4 < 8 (constant-buffer binding)
        (3, (17, 1), mx.bfloat16),  # prime total
        (5, (1, 33, 3), mx.float16),  # 3-D, non-multiple of 4
        (8, (2, 4, 17), mx.bfloat16),  # 4-D
        (20, (1, 512, 1), mx.bfloat16),  # > 4-way unroll + remainder
        (40, (1, 128, 48), mx.float16),  # large split_k
        (6, (1, 64, 64), mx.float32),  # fp32 output dtype
    ]
    for sk, out_shape, dtype in cases:
        mx.random.seed(100 + sk + out_shape[-1])
        rows = math.prod(out_shape[:-1])
        n = out_shape[-1]
        part = mx.random.normal((sk, rows, n)).astype(mx.float32) * 0.25
        mx.eval(part)
        got = tile_dense._splitk_accum(part, out_shape, dtype)
        got2 = tile_dense._splitk_accum(part, out_shape, dtype)
        # Sequential ascending FP32 reference (exact same order as kernel).
        acc = part[0]
        for i in range(1, sk):
            acc = acc + part[i]
        want = acc.astype(dtype).reshape(out_shape)
        mx.eval(got, got2, want)
        assert got.shape == out_shape and got.dtype == dtype
        if dtype == mx.float32:
            assert mx.array_equal(got.view(mx.uint32), want.view(mx.uint32)).item()
            assert mx.array_equal(got.view(mx.uint32), got2.view(mx.uint32)).item()
        else:
            assert _same_bits(got, want), (sk, out_shape, dtype)
            assert _same_bits(got, got2), (sk, out_shape, dtype)

    # Empty tensor guard (rows == 0 must not dispatch 0-threadgroup Metal).
    empty_part = mx.zeros((4, 0, 16), dtype=mx.float32)
    empty_out = tile_dense._splitk_accum(empty_part, (1, 0, 16), mx.bfloat16)
    mx.eval(empty_out)
    assert empty_out.shape == (1, 0, 16) and empty_out.dtype == mx.bfloat16

    # IDX64=True template compiles and is bitwise identical to IDX64=False.
    mx.random.seed(999)
    part64 = mx.random.normal((7, 19, 13)).astype(mx.float32)
    mx.eval(part64)
    total64 = 19 * 13
    y32 = accum_k(
        inputs=[part64],
        template=[("T", mx.bfloat16), ("SPLIT_K", 7), ("IDX64", False)],
        grid=(total64, 1, 1),
        threadgroup=(min(128, total64), 1, 1),
        output_shapes=[(1, 19, 13)],
        output_dtypes=[mx.bfloat16],
    )[0]
    y64 = accum_k(
        inputs=[part64],
        template=[("T", mx.bfloat16), ("SPLIT_K", 7), ("IDX64", True)],
        grid=(total64, 1, 1),
        threadgroup=(min(128, total64), 1, 1),
        output_shapes=[(1, 19, 13)],
        output_dtypes=[mx.bfloat16],
    )[0]
    mx.eval(y32, y64)
    assert _same_bits(y32, y64)

    # End-to-end _launch: split_k > 1 invokes fused accum and NEVER calls mx.sum;
    # split_k == 1 does NOT invoke fused accum.
    is_nax = linear._nax_host()
    n, k, rows = 64, 512, 33
    quantized, _ = _layer(n, k, 8, 64)
    x = _x((1, rows, k))
    face = affine.tile_face(8, 64, mx.bfloat16)
    ops = (quantized["weight"], quantized["scales"], quantized["biases"])
    pol_sk = (
        tuning.DenseTilePolicy("nax", 32, 64, 64, 2, 2, True, split_k=4, sched="seg")
        if is_nax
        else tuning.DenseTilePolicy(
            "steel", 32, 32, 32, 2, 2, True, split_k=4, x_nreads=8
        )
    )
    pol_no_sk = (
        tuning.DenseTilePolicy("nax", 32, 64, 64, 2, 2, True, split_k=1, sched="seg")
        if is_nax
        else tuning.DenseTilePolicy(
            "steel", 32, 32, 32, 2, 2, True, split_k=1, x_nreads=8
        )
    )
    mm = tile_dense._kernel(face, pol_sk)
    accum_calls: list[int] = []
    real_accum = tile_dense._splitk_accum

    def spy_accum(part, out_shape, dtype):
        accum_calls.append(part.shape[0])
        return real_accum(part, out_shape, dtype)

    def forbid_sum(*args, **kwargs):
        raise AssertionError("split-K launch must use fused accum, not mx.sum")

    monkeypatch.setattr(tile_dense, "_splitk_accum", spy_accum)
    monkeypatch.setattr(mx, "sum", forbid_sum)
    y_sk = tile_dense._launch(mm, pol_sk, x, ops)
    y_1 = tile_dense._launch(mm, pol_no_sk, x, ops)
    mx.eval(y_sk, y_1)
    assert accum_calls == [4]
    assert y_sk.shape == (1, rows, n) and y_sk.dtype == mx.bfloat16
    assert y_1.shape == (1, rows, n) and y_1.dtype == mx.bfloat16


def test_dense_splitk_accum_warmed_in_warm_specs(monkeypatch):
    """warm_specs registers the fused split-K accumulate kernel for split_k > 1
    schedules and omits it when all buckets use split_k == 1."""
    is_nax = linear._nax_host()
    chip = "applegpu_g17s_c40" if is_nax else "applegpu_g15d_c80"
    _, layer = _layer(48, 2560, 8, 64)
    pol_sk = (
        tuning.DenseTilePolicy("nax", 32, 64, 64, 2, 2, False, split_k=10, sched="seg")
        if is_nax
        else tuning.DenseTilePolicy(
            "steel", 32, 32, 32, 2, 2, False, split_k=10, x_nreads=8
        )
    )
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: pol_sk)
    plan_sk = tile_dense.build_plan(layer, tensor_units=is_nax, chip=chip)
    inst_sk, _ = tile_dense.warm_specs(
        layer, (mx.bfloat16,), tensor_units=is_nax, plan=plan_sk
    )
    accum_k = tile_dense._splitk_accum_kernel()
    accum_matches = [
        (inst, recipe)
        for inst, recipe in inst_sk.items()
        if getattr(inst.kernel, "native", inst.kernel) is accum_k
    ]
    assert accum_matches
    # Execute the registered warm recipe to verify its StandIn builds cleanly.
    stand_in = accum_matches[0][1]()
    assert stand_in.input_shapes == ((10, 16, 48),)
    assert stand_in.output_shapes == ((16, 48),)
    want = kernels.planned(
        accum_k,
        (((10, 16, 48), mx.float32),),
        [mx.bfloat16],
        template=[("T", mx.bfloat16), ("SPLIT_K", 10), ("IDX64", False)],
    )
    assert want in inst_sk
    assert stand_in.grid == (16 * 48, 1, 1)

    pol_1 = (
        tuning.DenseTilePolicy("nax", 32, 64, 64, 2, 2, False, split_k=1, sched="seg")
        if is_nax
        else tuning.DenseTilePolicy(
            "steel", 32, 32, 32, 2, 2, False, split_k=1, x_nreads=8
        )
    )
    monkeypatch.setattr(tuning, "lookup", lambda *a, **k: pol_1)
    plan_1 = tile_dense.build_plan(layer, tensor_units=is_nax, chip=chip)
    inst_1, _ = tile_dense.warm_specs(
        layer, (mx.bfloat16,), tensor_units=is_nax, plan=plan_1
    )
    assert not any(
        getattr(inst.kernel, "native", inst.kernel) is accum_k for inst in inst_1
    )


def test_dense_splitk_accum_thread_safe():
    """Concurrent threads share _splitk_accum_kernel() cleanly (Protocol rule 14)."""

    def worker(seed: int):
        k = tile_dense._splitk_accum_kernel()
        mx.random.seed(seed)
        part = mx.random.normal((8, 33, 48)).astype(mx.float32) * 0.25
        y = tile_dense._splitk_accum(part, (1, 33, 48), mx.bfloat16)
        acc = part[0]
        for i in range(1, 8):
            acc = acc + part[i]
        want = acc.astype(mx.bfloat16).reshape(1, 33, 48)
        mx.eval(y, want)
        assert _same_bits(y, want)
        return k

    with ThreadPoolExecutor(max_workers=4) as pool:
        kernels_seen = list(pool.map(worker, range(4)))
    assert len({id(k) for k in kernels_seen}) == 1


def test_deferred79_fused_splitk_winning_cells_routed_and_pass_c2(monkeypatch):
    """The split-K wins common to both dtypes route as measured and pass C2."""
    tuning.reset_cache()
    assert tuning.stale_entries() == []

    m5_wins = [
        (48, 2560, 8, 64, 128, "nax", 40),
        (1, 2560, 8, 64, 512, "nax", 20),
    ]
    m3_wins = [
        (512, 2560, 8, 64, 33, "steel", 20),
        (512, 2560, 8, 64, 512, "steel", 4),
        (2560, 6144, 8, 64, 64, "steel", 6),
        (48, 2560, 8, 64, 512, "steel", 8),
        (48, 2560, 8, 64, 2048, "steel", 5),
        (640, 2560, 8, 128, 33, "steel", 10),
        (640, 2560, 8, 128, 128, "steel", 8),
        (1, 2560, 8, 64, 512, "steel", 20),
        (1, 2560, 8, 64, 2048, "steel", 10),
    ]
    for chip, specs in (
        ("applegpu_g17s_c40", m5_wins),
        ("applegpu_g15d_c80", m3_wins),
    ):
        for n, k, bits, gs, rows, want_bk, want_sk in specs:
            for dtype in (mx.bfloat16, mx.float16):
                face = affine.tile_face(bits, gs, dtype)
                pol = tuning.lookup(
                    "dense_tile",
                    face,
                    dtype=dtype,
                    bits=bits,
                    group=gs,
                    n=n,
                    k=k,
                    rows=rows,
                    chip=chip,
                )
                assert pol is not None, (chip, n, k, rows, dtype)
                assert pol.backend == want_bk and pol.split_k == want_sk, (
                    chip,
                    n,
                    k,
                    rows,
                    dtype,
                    pol,
                )
                assert pol.is_legal(face, k, n, rows, chip, dtype=dtype)

    # FN_k_proj M=128 on M3 Ultra stays mlx (1.0355x over 7 draws).
    for dtype in (mx.bfloat16, mx.float16):
        face = affine.tile_face(8, 64, dtype)
        pol_keep = tuning.lookup(
            "dense_tile",
            face,
            dtype=dtype,
            bits=8,
            group=64,
            n=512,
            k=2560,
            rows=128,
            chip="applegpu_g15d_c80",
        )
        assert pol_keep is not None and pol_keep.backend == "mlx"

    # Execute every host winning cell through FormatLinear and check C2 (bf16 & fp16).
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    host_specs = m5_wins if linear._nax_host() else m3_wins
    for n, k, bits, gs, rows, _, _ in host_specs:
        for dtype in (mx.bfloat16, mx.float16):
            quantized, layer = _layer(n, k, bits, gs, dtype)
            x = _x((1, rows, k), dtype=dtype, seed=rows + n)
            with testing.fresh_proofs():
                got = layer(x)
            want = quantized(x)
            mx.eval(got, want)
            _assert_c2(got, want, x, quantized)


@pytest.mark.skipif(
    tuning.current_chip() != "applegpu_g15d_c80", reason="needs the M3 Ultra table"
)
def test_m3_output_projection_17_24_bf16_mlx_fp16_steel(monkeypatch):
    """BF16 17-24 2560x6144 stays MLX (stock bits); FP16 keeps Steel sk=12 (C2)."""
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for dtype in (mx.bfloat16, mx.float16):
        quantized, layer = _layer(2560, 6144, 8, 64, dtype)
        with testing.fresh_proofs():
            plan = layer._dense_tile(layer._keep())
            assert plan is not None
            for rows in (17, 24):
                route = tile_dense.route_for(plan, rows, dtype)
                x = _x((1, rows, 6144), dtype=dtype, seed=rows + 2560)
                got, want = layer(x), quantized(x)
                if dtype == mx.bfloat16:
                    assert _same_bits(got, want)
                    assert route.backend == "mlx"
                else:
                    assert route.backend == "steel" and route.policy.split_k == 12
                    _assert_c2(got, want, x, quantized)


@needs_nax
@pytest.mark.parametrize("width", (2048, 2560))
def test_m5_square_q4_registered_and_prefill_tiles(width):
    stock, layer = _layer(width, width, 4, 64)
    plan = tile_dense.build_plan(layer, tensor_units=True, chip="applegpu_g17s_c40")
    for rows in (4, 8, 16, 33, 64, 128, 256, 512):
        policy = tile_dense.policy_for(plan, rows, mx.bfloat16)
        assert isinstance(policy, tuning.DenseTilePolicy) and policy.backend == "nax"
        assert policy.sched == ("register" if rows <= 16 else "prefetch")
        kernel = tile_dense._kernel(plan.face, policy)
        for offset in (0, 1):
            x = _x((rows * width + offset,))[offset:].reshape(rows, width)
            out = tile_dense._launch(kernel, policy, x, plan.operands)
            _assert_c2(out, stock(x), x, stock)
