# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Release-3 + Release-4 (VFAST c3) merge invariants:
- Part A: per-chip tuning table completeness, R3 prefill fixes (short-prefill
  25-32 occupancy, M3 HC_down long-prefill), and VFAST c3 overlapping-cell
  resolutions.
- Part B: VFAST c3 construction-time schedule on batched verify (rows 2..16),
  qwen35_verify_qmm boundary (FormatLinear stays off verify_qmm), load-time
  warm_moe_model/warm_model_dispatch coverage, and _ROW_EXACT_ACTIVE locking.
- Part C: FormatLinear 1-row ((1, 1, K) and (1, K)) native FormatLinearFast
  dispatch (dot_fast + M3 mlx_rows dense_fast), proof failure retirement,
  tile_dense.project 1-row isolation, and _FastMask hiding both _fast/_fast2.
- Part D: Zero per-call host overhead on Python fallback (weight_key, no_{slot}
  cache), >=16 operand refusal, orphaned dot_fast recovery, and weight mutation.
"""

from __future__ import annotations

import cProfile
import pstats
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import pytest
from omlx.formats.mlx import traversal as mlx_traversal

from omlx.custom_kernels.quant_matmul import fast as native
from omlx.formats import affine, linear, testing, tile_dense, tuning
from omlx.patches import qwen35_moe_routed_decode, qwen35_verify_qmm

# Every 2-D quantized dense FormatLinear shape in Qwen3.8-Flash-Next (P32).
SERVED_FLASH_NEXT_SHAPES = (
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
    (248320, 2560, 8, 64),
)

_SERVED_CHIPS = (
    ("applegpu_g15d_c80", "applegpu_g15d", False, 80),
    ("applegpu_g17s_c40", "applegpu_g17s", True, 40),
)

# Representative row counts covering every dense_tile bucket (including 17 vs
# 18-24 on M3 width<=4096, and 29/30/32 in 25-32).
_PREFILL_PROBES = (17, 20, 24, 25, 29, 30, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096)


def _pair(
    n: int,
    k: int,
    *,
    bits: int = 8,
    group_size: int = 64,
    dtype: mx.Dtype = mx.bfloat16,
    bias: bool = False,
    seed: int = 1,
):
    key = mx.random.key(seed)
    w = mx.random.normal((n, k), key=key, dtype=dtype) * mx.array(0.05, dtype=dtype)
    wq, sc, bi = mx.quantize(w, group_size=group_size, bits=bits, mode="affine")
    b_arr = (
        mx.random.normal((n,), key=mx.random.key(seed + 99), dtype=dtype)
        * mx.array(0.1, dtype=dtype)
        if bias
        else None
    )
    if b_arr is not None:
        mx.eval(wq, sc, bi, b_arr)
    else:
        mx.eval(wq, sc, bi)
    ql = nn.QuantizedLinear(
        k, n, bias=bias, group_size=group_size, bits=bits, mode="affine"
    )
    ql.weight, ql.scales, ql.biases = wq, sc, bi
    if b_arr is not None:
        ql.bias = b_arr
    with testing.verified("qwen4_exp"):
        fl = linear.FormatLinear(
            "affine",
            bits,
            group_size,
            k,
            {"weight": wq, "scales": sc, "biases": bi},
            b_arr,
        )
    return ql, fl


def _x(shape: tuple[int, ...], *, dtype: mx.Dtype = mx.bfloat16, seed: int = 7):
    arr = mx.random.normal(shape, key=mx.random.key(seed), dtype=dtype)
    mx.eval(arr)
    return arr


def _same_bits(a: mx.array, b: mx.array) -> bool:
    return bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


def _assert_c2(got: mx.array, want: mx.array, x: mx.array, ql: nn.QuantizedLinear):
    w_f32 = mx.dequantize(
        ql.weight,
        ql.scales.astype(mx.float32),
        ql.biases.astype(mx.float32),
        group_size=ql.group_size,
        bits=ql.bits,
    )
    k = x.shape[-1]
    rows = x.size // k
    n = ql.scales.shape[0]
    ref = x.reshape(rows, k).astype(mx.float32) @ w_f32.T
    if "bias" in ql:
        ref = ref + ql.bias.astype(mx.float32)
    mx.eval(ref)
    assert bool(mx.all(mx.isfinite(got)).item())
    denom = float(mx.sqrt(mx.sum(ref * ref)).item())
    g_err = float(
        mx.sqrt(mx.sum((got.reshape(rows, n).astype(mx.float32) - ref) ** 2)).item()
    ) / max(denom, 1e-12)
    w_err = float(
        mx.sqrt(mx.sum((want.reshape(rows, n).astype(mx.float32) - ref) ** 2)).item()
    ) / max(denom, 1e-12)
    assert g_err <= 1.05 * w_err + 1e-6, (g_err, w_err)


def _best_route(
    face,
    *,
    dtype: mx.Dtype,
    bits: int,
    group: int,
    n: int,
    k: int,
    rows: int,
    chip: str,
):
    best_rank = (0, 0)
    best = None
    for route in ("dense_tile", "mma_tile", "msg_tile"):
        pol = tuning.lookup(
            route,
            face,
            dtype=dtype,
            bits=bits,
            group=group,
            n=n,
            k=k,
            rows=rows,
            chip=chip,
        )
        if pol is None:
            continue
        rank = tuning.match_rank(
            route,
            face,
            dtype=dtype,
            bits=bits,
            group=group,
            n=n,
            k=k,
            rows=rows,
            chip=chip,
        )
        if rank > best_rank:
            best_rank = rank
            best = (route, pol)
    return best


# ---------------------------------------------------------------------------
# Part A: tuning table completeness, R3 prefill fixes, and VFAST c3 overlaps
# ---------------------------------------------------------------------------


def test_served_flash_next_shapes_have_explicit_policies_on_both_chips():
    """Every served (shape, reachable bucket, dtype) in Qwen3.8-Flash-Next must
    have an explicit legal entry in both applegpu_g15d_c80 and applegpu_g17s_c40
    across the unified dense/mma/msg tables, and overlapping small-row cells
    must keep the faster VFAST c3 winner while preserving non-overlapping R3
    <17 entries."""
    tuning.reset_cache()
    assert tuning.stale_entries() == []

    missing: list[str] = []
    illegal: list[str] = []
    for chip, arch, _tu, _cores in _SERVED_CHIPS:
        for n, k, bits, gs in SERVED_FLASH_NEXT_SHAPES:
            probes = list(_PREFILL_PROBES)
            if (n, k) == (248320, 2560):
                # lm_head serves decode/verify batch products {2..8, 12, 16};
                # unmeasured odd rows (9-11, 13-15) use the c3 min(MLX, Dot)
                # floor rather than _default_steel_policy / _default_nax_policy.
                probes = [2, 3, 4, 5, 6, 7, 8, 12, 16, *probes]
            else:
                for r in (12, 13, 16):
                    if mlx_traversal(r, n, k, gs, arch, bits=bits)[0] in (
                        "qmm",
                        "split_k",
                    ):
                        probes.insert(0, r)
            for dtype in (mx.bfloat16, mx.float16):
                face = affine.tile_face(bits, gs, dtype)
                for rows in probes:
                    hit = _best_route(
                        face,
                        dtype=dtype,
                        bits=bits,
                        group=gs,
                        n=n,
                        k=k,
                        rows=rows,
                        chip=chip,
                    )
                    if hit is None:
                        bname = tuning.rows_bucket("dense_tile", rows)
                        missing.append(
                            f"{chip} {n}x{k} b{bits}g{gs} {dtype} rows={rows} ({bname})"
                        )
                    else:
                        route, pol = hit
                        legal = (
                            pol.is_legal(face, k, n, rows, chip, dtype=dtype)
                            if route == "dense_tile"
                            else pol.is_legal(face, k, n, rows, chip)
                        )
                        if not legal:
                            illegal.append(
                                f"{chip} {n}x{k} b{bits}g{gs} {dtype} rows={rows}: "
                                f"{route} {pol}"
                            )
    assert not missing, f"{len(missing)} missing served cells, e.g.: {missing[:12]}"
    assert not illegal, f"illegal served policies: {illegal[:8]}"

    # Pin every overlapping R3-vs-VFAST-c3 resolution (bf16 and fp16).
    for dtype in (mx.bfloat16, mx.float16):
        face = affine.tile_face(8, 64, dtype)
        # M3 Ultra overlaps: VFAST c3 wins rows 2..16; 248320 keeps mma at M=20
        # alongside R3 dense_tile 17-24 for M=17..19, 21..24.
        assert _best_route(
            face,
            dtype=dtype,
            bits=8,
            group=64,
            n=10240,
            k=2560,
            rows=6,
            chip="applegpu_g15d_c80",
        ) == ("dense_tile", tuning.DenseTilePolicy("dot"))
        assert (
            _best_route(
                face,
                dtype=dtype,
                bits=8,
                group=64,
                n=10240,
                k=2560,
                rows=16,
                chip="applegpu_g15d_c80",
            )[0]
            == "mma_tile"
        )
        m3_12288_16 = _best_route(
            face,
            dtype=dtype,
            bits=8,
            group=64,
            n=12288,
            k=2560,
            rows=16,
            chip="applegpu_g15d_c80",
        )
        assert m3_12288_16[0] == "mma_tile" and m3_12288_16[1].nsg == 16
        assert (
            _best_route(
                face,
                dtype=dtype,
                bits=8,
                group=64,
                n=248320,
                k=2560,
                rows=16,
                chip="applegpu_g15d_c80",
            )[0]
            == "mma_tile"
        )
        assert (
            _best_route(
                face,
                dtype=dtype,
                bits=8,
                group=64,
                n=248320,
                k=2560,
                rows=20,
                chip="applegpu_g15d_c80",
            )[0]
            == "mma_tile"
        )
        m3_248320_24 = _best_route(
            face,
            dtype=dtype,
            bits=8,
            group=64,
            n=248320,
            k=2560,
            rows=24,
            chip="applegpu_g15d_c80",
        )
        assert m3_248320_24[0] == "dense_tile" and m3_248320_24[1].backend == "steel"

        # M5 Max overlaps: VFAST c3 wins rows 2..16; 248320 17-24 serves M=20..24.
        for n in (10240, 12288, 248320):
            m5_16 = _best_route(
                face,
                dtype=dtype,
                bits=8,
                group=64,
                n=n,
                k=2560,
                rows=16,
                chip="applegpu_g17s_c40",
            )
            assert m5_16[0] == "dense_tile"
            assert (m5_16[1].backend, m5_16[1].bm, m5_16[1].bn, m5_16[1].bk) == (
                "nax",
                32,
                64,
                64,
            )
            assert m5_16[1].sched == "db"
        m5_248320_20 = _best_route(
            face,
            dtype=dtype,
            bits=8,
            group=64,
            n=248320,
            k=2560,
            rows=20,
            chip="applegpu_g17s_c40",
        )
        assert m5_248320_20[0] == "dense_tile" and m5_248320_20[1].backend == "nax"

        # Non-overlapping R3 <17 entries preserved on both chips.
        for chip, want_backend in (
            ("applegpu_g15d_c80", "steel"),
            ("applegpu_g17s_c40", "nax"),
        ):
            r3_4x10240 = _best_route(
                face,
                dtype=dtype,
                bits=8,
                group=64,
                n=4,
                k=10240,
                rows=16,
                chip=chip,
            )
            assert r3_4x10240 is not None and r3_4x10240[1].backend == want_backend
        for chip, want_backend in (
            ("applegpu_g15d_c80", "mlx"),
            ("applegpu_g17s_c40", "nax"),
        ):
            r3_10240x320 = _best_route(
                face,
                dtype=dtype,
                bits=8,
                group=64,
                n=10240,
                k=320,
                rows=16,
                chip=chip,
            )
            assert r3_10240x320 is not None and r3_10240x320[1].backend == want_backend


def test_m3_short_prefill_25_32_never_hits_low_occupancy_split_k_1():
    """Short decode prompts (M=25..32) on M3 Ultra must never fall into a
    split_k=1 steel tile with fewer threadgroups than GPU cores (e.g. 10 TGs
    for HC_down 320x10240), and host execution must pass C2 (bitwise for mlx)."""
    tuning.reset_cache()
    checked = (
        (320, 10240, 8, 64),
        (10240, 320, 8, 64),
        (2560, 6144, 8, 64),
        (6144, 2560, 8, 64),
        (10240, 2560, 8, 64),
        (640, 2560, 8, 64),
        (2560, 2560, 8, 64),
    )
    for n, k, bits, gs in checked:
        for dtype in (mx.bfloat16, mx.float16):
            face = affine.tile_face(bits, gs, dtype)
            for rows in (25, 29, 30, 32):
                pol = tuning.lookup(
                    "dense_tile",
                    face,
                    dtype=dtype,
                    bits=bits,
                    group=gs,
                    n=n,
                    k=k,
                    rows=rows,
                    chip="applegpu_g15d_c80",
                )
                assert pol is not None, (n, k, dtype, rows)
                if pol.backend == "steel" and pol.split_k == 1:
                    tgs = ((rows + pol.bm - 1) // pol.bm) * ((n + pol.bn - 1) // pol.bn)
                    assert tgs >= 80, (
                        f"low-occupancy steel sk=1 ({tgs} TGs < 80) for "
                        f"{n}x{k} M={rows}: {pol}"
                    )

    if not mx.metal.is_available():
        return
    host_chip = tuning.current_chip()
    host_tu = linear._nax_host()
    for n, k, bits, gs in (
        (320, 10240, 8, 64),
        (10240, 320, 8, 64),
        (640, 2560, 8, 64),
    ):
        ql, fl = _pair(n, k, bits=bits, group_size=gs, dtype=mx.bfloat16, seed=n + 30)
        plan = tile_dense.build_plan(fl, tensor_units=host_tu, chip=host_chip)
        assert plan is not None
        for rows in (29, 30):
            x = _x((1, rows, k), dtype=mx.bfloat16, seed=rows + n)
            got = tile_dense.project(plan, x)
            want = ql(x)
            mx.eval(got, want)
            pol = tile_dense.policy_for(plan, rows, mx.bfloat16)
            assert pol is not None
            if pol.backend == "mlx":
                assert _same_bits(got, want), (n, k, rows)
            else:
                _assert_c2(got, want, x, ql)


def test_m3_hc_down_long_prefill_avoids_unaligned_m_tail_cliff():
    """HC_down (320x10240 b8g64) at served prefill chunks (M=2047 in 1025-2048
    and M=4095 in 2049+) must not pick a split_k=1 steel tile with small bn/bm
    whose unaligned-M tail row (height=BM-1) stalls 60 cores on load_safe across
    K=10240 (+23 ms in-model on prefill2048 for 32x16x64 sk=1)."""
    tuning.reset_cache()
    n, k, bits, gs = 320, 10240, 8, 64
    for dtype in (mx.bfloat16, mx.float16):
        face = affine.tile_face(bits, gs, dtype)
        for rows in (1025, 2047, 2048, 4095, 4096):
            pol = tuning.lookup(
                "dense_tile",
                face,
                dtype=dtype,
                bits=bits,
                group=gs,
                n=n,
                k=k,
                rows=rows,
                chip="applegpu_g15d_c80",
            )
            assert pol is not None, (dtype, rows)
            assert pol.is_legal(face, k, n, rows, "applegpu_g15d_c80", dtype=dtype)
            if pol.backend == "steel":
                assert pol.bm >= 64 and pol.bn >= 32, (
                    f"small steel tile {pol} risks unaligned-M tail cliff "
                    f"on 320x10240 M={rows}"
                )
                tail_tgs = ((n + pol.bn - 1) // pol.bn) * pol.split_k
                assert pol.split_k >= 2 and tail_tgs >= 40, (
                    f"low tail-row concurrency ({tail_tgs} TGs, sk={pol.split_k}) "
                    f"on 320x10240 M={rows}: {pol}"
                )

    if not mx.metal.is_available():
        return
    host_chip = tuning.current_chip()
    host_tu = linear._nax_host()
    for dtype in (mx.bfloat16, mx.float16):
        ql, fl = _pair(n, k, bits=bits, group_size=gs, dtype=dtype, seed=320 + 2047)
        plan = tile_dense.build_plan(fl, tensor_units=host_tu, chip=host_chip)
        assert plan is not None
        for rows in (2047, 4095):
            x = _x((1, rows, k), dtype=dtype, seed=rows + 17)
            got = tile_dense.project(plan, x)
            want = ql(x)
            mx.eval(got, want)
            pol = tile_dense.policy_for(plan, rows, dtype)
            assert pol is not None
            if pol.backend == "mlx":
                assert _same_bits(got, want), (dtype, rows)
            else:
                _assert_c2(got, want, x, ql)


# ---------------------------------------------------------------------------
# Part B: VFAST c3 construction-time schedule on batched verify + warm-up
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_nk", [(6144, 2560), (10240, 2560), (12288, 2560)])
@pytest.mark.parametrize("bl", [(4, 2), (4, 3), (4, 4), (2, 4)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_format_linear_batched_verify_keeps_vfast_schedule_and_passes_c2(
    shape_nk, bl, with_bias, monkeypatch
):
    """Armed verify (non-row-exact and row-exact) must NOT hijack FormatLinear
    away from VFAST c3's construction-time schedule + FormatLinearFast, while
    unported nn.QuantizedLinear keeps its verify_qmm route intact."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()
    n, k = shape_nk
    b, length = bl
    rows = b * length
    ql, fl = _pair(n, k, bits=8, group_size=64, bias=with_bias, seed=n + rows)
    x = _x((b, length, k), dtype=mx.bfloat16, seed=n + b * 17 + length)

    unarmed = fl(x)
    want = ql(x)
    mx.eval(unarmed, want)
    _assert_c2(unarmed, want, x, ql)

    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        assert not qwen35_verify_qmm.takes_verify_route(fl, rows, x.dtype)
        assert qwen35_verify_qmm.takes_verify_route(ql, rows, x.dtype)
        with monkeypatch.context() as patch:
            patch.setattr(
                mx,
                "compile",
                lambda *a, **kw: pytest.fail("cold mx.compile on armed FormatLinear"),
            )
            patch.setattr(
                linear.FormatLinear,
                "_scheduled",
                lambda *a, **kw: pytest.fail("armed verify missed FormatLinearFast"),
            )
            armed = fl(x)
            mx.eval(armed)
        assert _same_bits(armed, unarmed), (n, k, bl, with_bias)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)


def test_warm_moe_model_warms_batched_verify_and_one_row_dispatch(monkeypatch):
    """warm_moe_model(..., mtp_enabled=True) must pre-warm (1, 1, K), multi-row
    decode (B, 1, K), batched verify (B, S, K), GDN-stacked (1, B*S, K), and
    prefill onto native FormatLinearFast so request #0 runs 0 compiles and 0
    Python fallbacks."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()

    ql_proj, fl_proj = _pair(6144, 2560, bits=8, group_size=64, seed=601)
    ql_mtp, fl_mtp = _pair(2560, 2560, bits=8, group_size=64, seed=602)

    class _Model(nn.Module):
        def __init__(self, proj, mtp):
            super().__init__()
            self.proj = proj
            self.mtp = mtp

    model = _Model(fl_proj, fl_mtp)
    for fut in linear.warm_existing_dense(model, dtypes=(mx.bfloat16,)) or ():
        fut.result()
    qwen35_moe_routed_decode.warm_moe_model(
        model, mtp_enabled=True, prefill_lengths=(64,)
    )

    # Both layers must already hold native fast dispatchers before request #0.
    assert fl_proj._fast is not None
    assert fl_mtp._fast is not None

    shapes = (
        (1, 1, 2560),
        (2, 1, 2560),
        (4, 1, 2560),
        (2, 2, 2560),
        (2, 3, 2560),
        (2, 4, 2560),
        (4, 2, 2560),
        (4, 3, 2560),
        (4, 4, 2560),
        (1, 6, 2560),
        (1, 8, 2560),
        (1, 12, 2560),
        (1, 16, 2560),
        (1, 64, 2560),
    )

    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                mx,
                "compile",
                lambda *a, **kw: pytest.fail("cold mx.compile after warm_moe_model"),
            )
            patch.setattr(
                linear,
                "_launch",
                lambda *a, **kw: pytest.fail("Python _launch after warm_moe_model"),
            )
            patch.setattr(
                tile_dense,
                "project",
                lambda *a, **kw: pytest.fail("Python project after warm_moe_model"),
            )
            patch.setattr(
                linear.FormatLinear,
                "_scheduled",
                lambda *a, **kw: pytest.fail("Python _scheduled after warm_moe_model"),
            )
            for ql, fl in ((ql_proj, fl_proj), (ql_mtp, fl_mtp)):
                for shape in shapes:
                    x = _x(shape, dtype=mx.bfloat16, seed=610 + sum(shape))
                    got = fl(x)
                    want = ql(x)
                    mx.eval(got, want)
                    if shape == (1, 1, 2560):
                        assert _same_bits(got, want), (fl.output_dims, shape)
                    else:
                        _assert_c2(got, want, x, ql)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)


def test_row_exact_active_counter_transitions_and_threads():
    """_ROW_EXACT_ACTIVE tracks armed row-exact threads cleanly across
    idempotent calls, verify <-> row_exact transitions, and concurrent threads."""
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert qwen35_verify_qmm._ROW_EXACT_ACTIVE == 0

    qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert qwen35_verify_qmm._ROW_EXACT_ACTIVE == 0
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        assert qwen35_verify_qmm._ROW_EXACT_ACTIVE == 0
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
        assert qwen35_verify_qmm._ROW_EXACT_ACTIVE == 1
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
        assert qwen35_verify_qmm._ROW_EXACT_ACTIVE == 1
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
        assert qwen35_verify_qmm._ROW_EXACT_ACTIVE == 0
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert qwen35_verify_qmm._ROW_EXACT_ACTIVE == 0

    barrier = threading.Barrier(2)
    observed: list[int] = []

    def arm_worker():
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
        try:
            barrier.wait(timeout=5.0)
            observed.append(qwen35_verify_qmm._ROW_EXACT_ACTIVE)
            barrier.wait(timeout=5.0)
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(arm_worker)
        f2 = pool.submit(arm_worker)
        f1.result()
        f2.result()
    assert observed == [2, 2]
    assert qwen35_verify_qmm._ROW_EXACT_ACTIVE == 0


# ---------------------------------------------------------------------------
# Part C: FormatLinear 1-row native FormatLinearFast dispatch & safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_nk,with_bias",
    [
        ((4096, 2560), False),  # 1 not in mlx_rows -> dot_fast
        ((4096, 2560), True),  # with bias -> dot_fast
        ((2560, 2560), False),  # 1 in mlx_rows on M3 -> dense_fast when dense built
        ((256, 256), False),  # small layer -> dot_fast
    ],
)
def test_format_linear_one_row_uses_native_fast_dispatch(
    shape_nk, with_bias, monkeypatch
):
    """FormatLinear (1, 1, K) and (1, K) must be served by native
    FormatLinearFast (dot_fast, or dense_fast for M3 mlx_rows) and coexist
    lock-free with prefill (1, 32, K) across _fast/_fast2 with zero Python
    dispatch, bitwise identical to stock QuantizedLinear."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")
    n, k = shape_nk
    ql, fl = _pair(n, k, bits=8, group_size=64, bias=with_bias, seed=701 + n)
    linear.prepare_layer_plans(fl)
    x1_3d = _x((1, 1, k), dtype=mx.bfloat16, seed=710)
    x1_2d = _x((1, k), dtype=mx.bfloat16, seed=711)
    xp = _x((1, 32, k), dtype=mx.bfloat16, seed=712)

    first1 = fl(x1_3d)
    first1_2d = fl(x1_2d)
    firstp = fl(xp)
    mx.eval(first1, first1_2d, firstp)

    kept = fl.__dict__[linear._KEPT]
    if 1 in kept.mlx_rows:
        assert kept.dense_fast is not None
    else:
        assert kept.dot_fast is not None
    assert fl._fast is not None

    served: list[bool] = []
    orig_fast = fl.__dict__.get(linear._FAST)
    orig_fast2 = fl.__dict__.get(linear._FAST2)

    def wrap(fn):
        if fn is None:
            return None

        def counting(module, value, _fn=fn):
            y = _fn(module, value)
            if y is not None:
                served.append(True)
            return y

        return counting

    with monkeypatch.context() as patch:
        fl.__dict__[linear._FAST] = wrap(orig_fast)
        if orig_fast2 is not None:
            fl.__dict__[linear._FAST2] = wrap(orig_fast2)
        patch.setattr(
            linear.FormatLinear,
            "_keep",
            lambda *a, **kw: pytest.fail("Python _keep on warm 1-row/prefill"),
        )
        patch.setattr(
            linear.FormatLinear,
            "_scheduled",
            lambda *a, **kw: pytest.fail("Python _scheduled on warm 1-row/prefill"),
        )
        patch.setattr(
            linear,
            "_launch",
            lambda *a, **kw: pytest.fail("Python _launch on warm 1-row/prefill"),
        )
        try:
            again1 = fl(x1_3d)
            againp = fl(xp)
            again1_2d = fl(x1_2d)
            again1_b = fl(x1_3d)
            mx.eval(again1, againp, again1_2d, again1_b)
        finally:
            fl.__dict__[linear._FAST] = orig_fast
            if orig_fast2 is not None:
                fl.__dict__[linear._FAST2] = orig_fast2

    assert served == [True, True, True, True], f"missed native fast: {served}"
    assert _same_bits(first1, again1)
    assert _same_bits(first1, again1_b)
    assert _same_bits(first1_2d, again1_2d)
    assert _same_bits(firstp, againp)
    ref1 = ql(x1_3d)
    ref1_2d = ql(x1_2d)
    mx.eval(ref1, ref1_2d)
    assert _same_bits(again1, ref1)
    assert _same_bits(again1_2d, ref1_2d)


def test_one_row_fast_safety_proof_fail_tile_dense_project_and_threads(monkeypatch):
    """Safety invariants for 1-row native fast dispatch:
    1. plan.proof.fail() immediately retires dot_fast on (1, 1, K) and (1, K)
       and falls back bitwise to _stock_matmul.
    2. After _bind_one_row_mlx_fast caches a 1-row graph in dense.binding.kept,
       direct tile_dense.project(kept.dense, x1) still enters _project_cold
       (returning None) and never replays or overwrites the bound graph.
    3. Multi-thread (1, 1, K) calls stay bitwise exact."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")

    with testing.fresh_proofs():
        ql, fl = _pair(256, 256, bits=8, group_size=64, seed=801)
        linear.prepare_layer_plans(fl)
        x1 = _x((1, 1, 256), dtype=mx.bfloat16, seed=802)
        x1_2d = _x((1, 256), dtype=mx.bfloat16, seed=803)
        ref = ql(x1)
        ref_2d = ql(x1_2d)
        y = fl(x1)
        y_2d = fl(x1_2d)
        mx.eval(ref, ref_2d, y, y_2d)
        assert _same_bits(y, ref)
        assert _same_bits(y_2d, ref_2d)
        kept = fl.__dict__[linear._KEPT]
        assert kept.dot_fast is not None
        assert fl._fast is kept.dot_fast
        assert kept.dot_fast(fl, x1) is not None
        assert kept.dot_fast(fl, x1_2d) is not None

        # Multi-thread (1, 1, K) calls on dot_fast.
        def worker():
            out = fl(x1)
            mx.eval(out)
            return _same_bits(out, ref)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(worker)
            f2 = pool.submit(worker)
            assert f1.result() and f2.result()

        # Failing kept.plan.proof must immediately retire dot_fast replay.
        kept.plan.proof.fail()
        assert kept.dot_fast(fl, x1) is None
        assert kept.dot_fast(fl, x1_2d) is None
        fb1 = fl(x1)
        fb2 = fl(x1_2d)
        mx.eval(fb1, fb2)
        assert _same_bits(fb1, ref)
        assert _same_bits(fb2, ref_2d)

        # _bind_one_row_mlx_fast isolation: tile_dense.project(dense, x1) must
        # go through _project_cold (returning None) and never replay/clobber
        # the 1-row MLX graph bound for dense_fast.
        ql_m, fl_m = _pair(256, 256, bits=8, group_size=64, seed=805)
        linear.prepare_layer_plans(fl_m)
        kept_m = fl_m.__dict__[linear._KEPT]
        sig1 = (x1.shape, x1.dtype)
        fl_m._bind_one_row_mlx_fast(kept_m, kept_m.dense, sig1)
        assert kept_m.dense_fast is not None
        replay_m = kept_m.dense_fast(fl_m, x1)
        assert replay_m is not None
        ref_m = ql_m(x1)
        mx.eval(replay_m, ref_m)
        assert _same_bits(replay_m, ref_m)

        bound_before = kept_m.dense.binding.kept.graphs[sig1]
        cold_calls: list[tuple] = []
        real_cold = tile_dense._project_cold

        def spy_cold(plan, x, sig):
            cold_calls.append(sig)
            return real_cold(plan, x, sig)

        with monkeypatch.context() as patch:
            patch.setattr(tile_dense, "_project_cold", spy_cold)
            td_y = tile_dense.project(kept_m.dense, x1)
        assert td_y is None
        assert cold_calls == [sig1]
        assert kept_m.dense.binding.kept.graphs[sig1] is bound_before


def test_fast_mask_hides_both_fast_and_fast2_only_on_recording_thread():
    """testing.layer_calls() must mask BOTH _fast and _fast2 on the recording
    thread (recording warm calls and keeping raw Proof._kept graphs) while
    leaving concurrent non-recording threads on native fast."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")

    with testing.fresh_proofs():
        _, fl = _pair(256, 256, bits=8, group_size=64, seed=970)
        linear.prepare_layer_plans(fl)
        x1 = _x((1, 1, 256), dtype=mx.bfloat16, seed=971)
        xp = _x((1, 32, 256), dtype=mx.bfloat16, seed=972)

        # Populate BOTH _fast (dot_fast) and _fast2 (dense_fast).
        mx.eval(fl(x1), fl(xp))
        assert fl._fast is not None and fl._fast2 is not None
        kept = fl.__dict__[linear._KEPT]

        other_saw: list[tuple[bool, bool]] = []
        entered = threading.Event()
        release = threading.Event()

        def observer():
            entered.wait(timeout=5.0)
            other_saw.append((fl._fast is not None, fl._fast2 is not None))
            release.set()

        t = threading.Thread(target=observer)
        t.start()
        try:
            with testing.layer_calls() as calls:
                assert fl._fast is None and fl._fast2 is None
                entered.set()
                release.wait(timeout=5.0)
                mx.eval(fl(x1))
                mx.eval(fl(xp))
        finally:
            t.join(timeout=5.0)

        assert other_saw == [(True, True)], other_saw
        assert len(calls) >= 1
        assert fl._fast is not None and fl._fast2 is not None
        # Raw graph kept in plan.proof._kept (not the recorder wrapper).
        raw = kept.plan.proof._kept.graphs[(x1.shape, x1.dtype)]
        assert raw is not None
        assert getattr(raw, "__name__", "") != "recorded"


def test_dot_rows_records_under_layer_calls_and_reuses_graph_on_fallback(monkeypatch):
    """A warm multi-row Dot-scheduled call must record under
    testing.layer_calls() (the masked Proof graphs, not the raw slot) and must
    reuse its kept graph when no native dot_fast is bound instead of
    re-launching (mx.compile) every warm call."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")

    with testing.fresh_proofs():
        _, fl = _pair(10240, 2560, bits=8, group_size=64, seed=980)
        x3 = _x((3, 1, 2560), dtype=mx.bfloat16, seed=981)
        mx.eval(fl(x3))
        with testing.layer_calls() as calls:
            mx.eval(fl(x3))
        assert [c.layer for c in calls] == [fl]

        _, fl2 = _pair(10240, 2560, bits=8, group_size=64, seed=982)
        monkeypatch.setattr(native, "has_symbol", lambda _n: False)
        mx.eval(fl2(x3))

        def boom(*a, **kw):
            raise AssertionError("warm Dot fallback re-launched")

        with monkeypatch.context() as patch:
            patch.setattr(linear, "_launch", boom)
            patch.setattr(mx, "compile", boom)
            mx.eval(fl2(x3))


# ---------------------------------------------------------------------------
# Part D: Zero per-call host overhead & native-fast boundary guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "shape",
    [(1, 2048), (2, 2048), (4, 2048), (8, 2048), (2, 1, 2048), (4, 1, 2048)],
)
def test_format_linear_small_rows_python_fallback_zero_extra_calls(
    bits, shape, monkeypatch
):
    """On the Python warm path (native disabled), small-row FormatLinear calls
    must make zero calls to packed_name() and zero repeat native.has_symbol()
    calls."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    _, fl = _pair(4096, 2048, bits=bits, group_size=64, seed=901 + bits)
    x = _x(shape, dtype=mx.bfloat16, seed=910 + sum(shape))

    with monkeypatch.context() as patch:
        has_symbol_calls = 0

        def spy_has_symbol(_name):
            nonlocal has_symbol_calls
            has_symbol_calls += 1
            return False

        patch.setattr(native, "has_symbol", spy_has_symbol)
        # Warm the graph and establish no_dot_fast / no_dense_fast on cold call.
        warm0 = fl(x)
        mx.eval(warm0)
        assert fl._fast is None
        has_symbol_calls = 0

        packed_calls = 0
        orig_packed = linear.packed_name

        def spy_packed(layer):
            nonlocal packed_calls
            packed_calls += 1
            return orig_packed(layer)

        patch.setattr(linear, "packed_name", spy_packed)

        pr = cProfile.Profile()
        pr.enable()
        y = warm0
        for _ in range(32):
            y = fl(x)
        pr.disable()
        mx.eval(y)

    assert packed_calls == 0, f"packed_name() called {packed_calls} times"
    assert has_symbol_calls == 0, f"has_symbol() called {has_symbol_calls} times"
    callee_names = {fn[2] for fn in pstats.Stats(pr).stats if fn[2] != "disable"}
    assert "packed_name" not in callee_names, callee_names


def test_native_fast_refuses_16_or_more_operands_and_recovers_orphaned_dot_fast():
    """_native_fast refuses operands of length >= 16 before caching; orphaned
    _fast/_fast2 is re-published on a warm Dot hit; weight mutation retires
    both _fast and _fast2 and rebinds cleanly."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")

    with testing.fresh_proofs():
        ql, fl = _pair(256, 256, bits=8, group_size=64, bias=True, seed=955)
        kept = fl._keep()
        sig = ((1, 256), mx.bfloat16)
        kept.plan.proof.keep(sig, lambda x, *ops: x)

        fake_wide = type(
            "_BigTile",
            (),
            {"operands": (kept.weight,) * 16, "proof": kept.plan.proof},
        )()
        assert fl._native_fast(kept, "dot_fast", fake_wide.operands, {}) is None
        assert kept.dot_fast is None
        assert kept._slots.get("no_dot_fast") is True
        assert fl._native_fast(kept, "dense_fast", fake_wide.operands, {}) is None
        assert kept.dense_fast is None
        assert kept._slots.get("no_dense_fast") is True

        # Clean layer: warm (1, 1, 256), orphan _fast, verify warm hit recovers.
        ql2, fl2 = _pair(256, 256, bits=8, group_size=64, bias=True, seed=956)
        x1 = _x((1, 1, 256), dtype=mx.bfloat16, seed=957)
        ref = ql2(x1)
        y0 = fl2(x1)
        mx.eval(ref, y0)
        assert _same_bits(y0, ref)
        kept2 = fl2.__dict__[linear._KEPT]
        assert fl2._fast is kept2.dot_fast

        fl2.__dict__.pop(linear._FAST, None)
        fl2.__dict__.pop(linear._FAST2, None)
        assert fl2._fast is None
        y1 = fl2(x1)
        mx.eval(y1)
        assert _same_bits(y1, ref)
        assert fl2._fast is kept2.dot_fast

        # Weight mutation retires both _fast and _fast2 cleanly and rebinds.
        fl2["weight"] = mx.array(fl2["weight"])
        assert fl2._fast is None and fl2._fast2 is None
        y2 = fl2(x1)
        mx.eval(y2)
        assert _same_bits(y2, ref)
        assert fl2._fast is not None
        assert fl2._fast is fl2.__dict__[linear._KEPT].dot_fast


def test_shared_proof_dot_hit_registers_dot_fast_on_sibling_layers():
    """A Dot hit on a shared plan.proof graph must bind native dot_fast on the
    layer that took the hit (sibling layers and a fresh same-shape layer),
    not leave them on Python dispatch."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")

    with testing.fresh_proofs():
        _, fl1 = _pair(10240, 2560, bits=8, group_size=64, seed=990)
        _, fl2 = _pair(10240, 2560, bits=8, group_size=64, seed=991)
        x1 = _x((1, 1, 2560), dtype=mx.bfloat16, seed=992)
        x3 = _x((3, 1, 2560), dtype=mx.bfloat16, seed=993)
        mx.eval(fl1(x1), fl2(x1))
        mx.eval(fl1(x3), fl2(x3))
        assert fl2._fast is not None and fl2._fast(fl2, x3) is not None

        _, fl3 = _pair(10240, 2560, bits=8, group_size=64, seed=994)
        mx.eval(fl3(x3))
        assert fl3._fast is not None and fl3._fast(fl3, x3) is not None
