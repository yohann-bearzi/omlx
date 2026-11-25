# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Release-3 freeze fixes A (tuning data), B (FormatLinear
batched verify route), and C (FormatLinear (1, 1, K) native dense_fast).
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
    qwen: bool = True,
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
    if qwen:
        object.__setattr__(fl, "_qwen_row_exact", qwen35_verify_qmm.is_row_exact_armed)
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
    rows = x.size // ql.scales.shape[1] // ql.group_size
    k = x.shape[-1]
    n = ql.scales.shape[0]
    ref = x.reshape(rows, k).astype(mx.float32) @ w_f32.T
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


# ---------------------------------------------------------------------------
# Part A: tuning table completeness & short-prefill occupancy floor
# ---------------------------------------------------------------------------


def test_served_flash_next_shapes_have_explicit_dense_tile_entries_on_both_chips():
    """Every served (shape, reachable bucket, dtype) in Qwen3.8-Flash-Next must
    have an explicit legal entry in both applegpu_g15d_c80 and applegpu_g17s_c40
    so _default_steel_policy / _default_nax_policy can never silently take a
    served shape."""
    tuning.reset_cache()
    assert tuning.stale_entries() == []

    missing: list[str] = []
    illegal: list[str] = []
    for chip, arch, _tu, _cores in _SERVED_CHIPS:
        for n, k, bits, gs in SERVED_FLASH_NEXT_SHAPES:
            probes = list(_PREFILL_PROBES)
            # Include <17 rows whenever ordinary traversal delegates to _prefill.
            for r in (12, 13, 16):
                if mlx_traversal(r, n, k, gs, arch, bits=bits)[0] in (
                    "qmm",
                    "split_k",
                ):
                    probes.insert(0, r)
            for dtype in (mx.bfloat16, mx.float16):
                face = affine.tile_face(bits, gs, dtype)
                for rows in probes:
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
                    if pol is None:
                        bname = tuning.rows_bucket("dense_tile", rows)
                        missing.append(
                            f"{chip} {n}x{k} b{bits}g{gs} {dtype} rows={rows} ({bname})"
                        )
                    elif not pol.is_legal(face, k, n, rows, chip, dtype=dtype):
                        illegal.append(
                            f"{chip} {n}x{k} b{bits}g{gs} {dtype} rows={rows}: {pol}"
                        )
    assert not missing, f"{len(missing)} missing served cells, e.g.: {missing[:12]}"
    assert not illegal, f"illegal served policies: {illegal[:8]}"


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
# Part B: FormatLinear batched verify route (vk_qmm_mma / verify_qmm)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_nk", [(6144, 2560), (10240, 2560), (12288, 2560)])
@pytest.mark.parametrize("bl", [(4, 2), (4, 3), (4, 4), (2, 4)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_format_linear_batched_verify_matches_r2_quantized_linear_bitwise(
    shape_nk, bl, with_bias, monkeypatch
):
    """Armed non-row-exact 3-D batched verify (rows 8, 12, 16) through
    FormatLinear must take the verify qmm route (vk_qmm_mma on M3, mma/skinny
    on M5), bitwise equal to release 2's patched nn.QuantizedLinear, with
    construction-time route binding (0 hot-path _verify_route calls)."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()
    n, k = shape_nk
    b, length = bl
    rows = b * length
    ql, fl = _pair(n, k, bits=8, group_size=64, bias=with_bias, seed=n + rows)
    x = _x((b, length, k), dtype=mx.bfloat16, seed=n + b * 17 + length)

    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        assert qwen35_verify_qmm.takes_verify_route(fl, rows, x.dtype)
        want = ql(x)
        mx.eval(want)

        # Must not hit tile_wide or tile_dense during armed batched verify.
        with monkeypatch.context() as patch:
            patch.setattr(
                linear.FormatLinear,
                "_ordinary",
                lambda *a, **kw: pytest.fail("armed verify fell into _ordinary"),
            )
            patch.setattr(
                tile_dense,
                "project",
                lambda *a, **kw: pytest.fail("armed verify fell into tile_dense"),
            )
            first = fl(x)
            mx.eval(first)
            assert _same_bits(
                first, want
            ), f"FormatLinear verify != R2 QuantizedLinear for {n}x{k} {(b, length)}"

            # Construction-time binding: steady-state call makes 0 _verify_route
            # lookups and 0 mx.compile calls.
            patch.setattr(
                qwen35_verify_qmm,
                "_verify_route",
                lambda *a, **kw: pytest.fail("hot-path _verify_route called"),
            )
            patch.setattr(
                mx,
                "compile",
                lambda *a, **kw: pytest.fail("hot-path mx.compile called"),
            )
            second = fl(x)
            mx.eval(second)
            assert _same_bits(second, want)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)


def test_format_linear_verify_route_bypasses_unarmed_fast_dispatch_both_directions():
    """Unarmed FormatLinearFast (wide_fast / dense_fast) must never shadow
    armed batched verify, and armed batched verify must never poison unarmed
    FormatLinearFast on the exact same 3-D shape."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()
    n, k = 6144, 2560
    ql, fl = _pair(n, k, bits=8, group_size=64, seed=412)

    for shape in ((4, 2, k), (4, 4, k), (1, 12, k)):
        x = _x(shape, dtype=mx.bfloat16, seed=sum(shape))
        # 1. Unarmed call populates FormatLinearFast.
        unarmed_ref = fl(x)
        mx.eval(unarmed_ref)
        assert fl._fast is not None
        assert fl._fast(fl, x) is not None

        # 2. Arm batched verify: C++ _fast must return None so verify route runs.
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
        try:
            assert fl._fast(fl, x) is None
            armed_want = ql(x)
            armed_got = fl(x)
            mx.eval(armed_want, armed_got)
            assert _same_bits(armed_got, armed_want), shape
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False)

        # 3. Disarm: FormatLinearFast replays unarmed output again (unpoisoned).
        replay = fl._fast(fl, x)
        assert replay is not None
        mx.eval(replay)
        assert _same_bits(replay, unarmed_ref), shape

        # 4. Arm row-exact verify: must take serial row-exact Dot, not vk_qmm_mma.
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
        try:
            assert fl._fast(fl, x) is None
            re_got = fl(x)
            re_single = mx.concatenate(
                [fl(x.reshape(-1, k)[i : i + 1]) for i in range(x.size // k)], axis=0
            ).reshape(x.shape[0], x.shape[1], n)
            mx.eval(re_got, re_single)
            assert _same_bits(re_got, re_single), shape
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False)


def test_format_linear_verify_route_guards_mutations_and_threads():
    """Verify route guards: non-eligible N gets no verify plan, 2-D and (1,1,K)
    stay off verify route, wrong width raises ValueError, weight/bias mutation
    invalidates the bound verify plan, and concurrent threads stay isolated."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()

    # Non-eligible N=512 (8-bit): kept.verify must be None at construction.
    _, small = _pair(512, 2560, bits=8, group_size=64, seed=501)
    kept_small = small._keep()
    assert kept_small.verify is None

    # Eligible N=4096 (8-bit): kept.verify bound at construction.
    ql, fl = _pair(4096, 2560, bits=8, group_size=64, seed=502)
    kept = fl._keep()
    assert kept.verify is not None

    # Non-affine layer must get kept.verify is None even if tensors match affine shapes.
    non_affine = type(
        "_Stub",
        (),
        {
            "_stock": True,
            "_STACK": 0,
            "mode": "oqel",
            "input_dims": 2560,
            "output_dims": 4096,
            "bits": 8,
            "group_size": 64,
            "get": lambda self, key: fl.get(key),
        },
    )()
    assert qwen35_verify_qmm.bind_format_verify(non_affine, None) is None

    x_bad = _x((4, 2, 128), dtype=mx.bfloat16, seed=503)
    x_2d = _x((8, 2560), dtype=mx.bfloat16, seed=504)
    x_3d = _x((4, 2, 2560), dtype=mx.bfloat16, seed=505)

    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        with pytest.raises(ValueError):
            kept.verify(x_bad)
        with pytest.raises(ValueError):
            fl(x_bad)
        # 2-D does not enter batched verify route (matches R2 x.ndim != 3).
        assert kept.verify(x_2d) is None
        y_2d = fl(x_2d)
        mx.eval(y_2d)
        assert y_2d.shape == (8, 4096)
        before = fl(x_3d)
        mx.eval(before)
        # Mutate scales: must retire old verify plan and compute with new scales.
        new_scales = fl["scales"] * mx.array(2.0, dtype=mx.bfloat16)
        mx.eval(new_scales)
        fl["scales"] = new_scales
        ql.scales = new_scales
        after_got = fl(x_3d)
        after_want = ql(x_3d)
        mx.eval(after_got, after_want)
        assert not _same_bits(before, after_got)
        assert _same_bits(after_got, after_want)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)

    # Multi-thread isolation (Rule 14): armed thread vs unarmed thread concurrently.
    unarmed_ref = fl(x_3d)
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        armed_ref = ql(x_3d)
        mx.eval(unarmed_ref, armed_ref)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)

    def run_armed():
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
        try:
            outs = [fl(x_3d) for _ in range(4)]
            mx.eval(*outs)
            return all(_same_bits(o, armed_ref) for o in outs)
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False)

    def run_unarmed():
        qwen35_verify_qmm.set_verify_qmm_armed(False)
        outs = [fl(x_3d) for _ in range(4)]
        mx.eval(*outs)
        return all(_same_bits(o, unarmed_ref) for o in outs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_arm = pool.submit(run_armed)
        f_unarm = pool.submit(run_unarmed)
        assert f_arm.result()
        assert f_unarm.result()


def test_warm_moe_model_warms_verify_qmm_and_dispatch(monkeypatch):
    """warm_moe_model(..., mtp_enabled=True) must warm verify_qmm MMA Metals and
    pre-compile 4-stream verify graphs so the first armed call does 0 compiles."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()
    ql, fl = _pair(4096, 2560, bits=8, group_size=64, seed=601)

    class _Model(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.proj = layer

    model = _Model(fl)
    qwen35_moe_routed_decode.warm_moe_model(
        model, mtp_enabled=True, prefill_lengths=(64,)
    )
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                mx,
                "compile",
                lambda *a, **kw: pytest.fail("cold mx.compile on warmed verify"),
            )
            for bl in ((4, 2), (4, 3), (4, 4)):
                x = _x((*bl, 2560), dtype=mx.bfloat16, seed=610 + bl[1])
                got = fl(x)
                want = ql(x)
                mx.eval(got, want)
                assert _same_bits(got, want), bl
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)


def test_format_linear_verify_routes_vk_sg8_and_unpatched_guard(monkeypatch):
    """Verify routes beyond mma (vk ksplit, vk msg, sg8) match R2 bitwise and
    is_armed() stays False while the verify qmm patch is not applied."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()

    # Unpatched guard: when _QL_PATCHED is False, FormatLinear verify is disarmed.
    _, fl_guard = _pair(4096, 2560, bits=8, group_size=64, seed=650)
    kept_guard = fl_guard._keep()
    assert kept_guard.verify is not None
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        assert kept_guard.verify.is_armed() is True
        with monkeypatch.context() as patch:
            patch.setattr(qwen35_verify_qmm, "_QL_PATCHED", False)
            assert kept_guard.verify.is_armed() is False
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)

    # vk ksplit (N=16384, rows=4), vk msg (N=100000, rows=4), sg8 (bits=4, rows=8).
    cases = [
        (16384, 2560, 8, (2, 2)),
        (100000, 2560, 8, (2, 2)),
        (4096, 2560, 4, (4, 2)),
    ]
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        for n, k, bits, bl in cases:
            ql, fl = _pair(n, k, bits=bits, group_size=64, seed=660 + n + bits)
            x = _x((*bl, k), dtype=mx.bfloat16, seed=670 + n)
            want = ql(x)
            got = fl(x)
            mx.eval(want, got)
            assert _same_bits(got, want), (n, k, bits, bl)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)


@pytest.mark.parametrize("with_bias", [False, True])
def test_format_linear_verify_group_sums_only_passed_to_sg8(with_bias):
    """Registered group sums must be passed ONLY to sg8 (group_size==64) verify
    routes; non-sg8 routes (mma/skinny/vk) and group_size!=64 sg8 on the same
    or another 4-bit layer must never receive group sums and must match R2's
    nn.QuantizedLinear bitwise."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()

    ql, fl = _pair(4096, 2560, bits=4, group_size=64, bias=with_bias, seed=680)
    ql_vk, fl_vk = _pair(16384, 2560, bits=4, group_size=64, bias=with_bias, seed=681)
    x8 = _x((4, 2, 2560), dtype=mx.bfloat16, seed=682)
    x12 = _x((4, 3, 2560), dtype=mx.bfloat16, seed=683)
    x3 = _x((1, 3, 2560), dtype=mx.bfloat16, seed=684)

    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        kept = fl._keep()
        kept_vk = fl_vk._keep()
        assert kept.verify is not None and kept_vk.verify is not None
        assert kept.verify.routes.get((8, mx.bfloat16)) == "sg8"
        assert kept.verify.routes.get((12, mx.bfloat16)) in ("mma", "skinny")
        assert kept_vk.verify.routes.get((8, mx.bfloat16)) == "sg8"
        assert kept_vk.verify.routes.get((3, mx.bfloat16)) == "vk"

        qwen35_verify_qmm.register_group_sums(x8, qwen35_verify_qmm.group_sums(x8))
        qwen35_verify_qmm.register_group_sums(x12, qwen35_verify_qmm.group_sums(x12))
        qwen35_verify_qmm.register_group_sums(x3, qwen35_verify_qmm.group_sums(x3))

        got8, want8 = fl(x8), ql(x8)
        got12, want12 = fl(x12), ql(x12)
        got3, want3 = fl_vk(x3), ql_vk(x3)
        mx.eval(got8, want8, got12, want12, got3, want3)

        assert _same_bits(got8, want8)
        assert _same_bits(got12, want12)
        assert _same_bits(got3, want3)

        # sg8 (g=64) must take the summed graph; non-sg8 (mma/skinny/vk) must not.
        graphs = kept.verify._graphs()
        assert ((4, 2, 2560), mx.bfloat16, True) in graphs
        assert ((4, 2, 2560), mx.bfloat16, False) not in graphs
        assert ((4, 3, 2560), mx.bfloat16, False) in graphs
        assert ((4, 3, 2560), mx.bfloat16, True) not in graphs
        graphs_vk = kept_vk.verify._graphs()
        assert ((1, 3, 2560), mx.bfloat16, False) in graphs_vk
        assert ((1, 3, 2560), mx.bfloat16, True) not in graphs_vk

        # group_size=32 sg8 must never take summed graphs even when sums are registered.
        if not with_bias:
            ql32, fl32 = _pair(4096, 2560, bits=4, group_size=32, seed=685)
            kept32 = fl32._keep()
            assert kept32.verify is not None
            assert kept32.verify.routes.get((8, mx.bfloat16)) == "sg8"
            assert kept32.verify.sg8_keys == frozenset()
            got32, want32 = fl32(x8), ql32(x8)
            mx.eval(got32, want32)
            assert _same_bits(got32, want32)
            graphs32 = kept32.verify._graphs()
            assert ((4, 2, 2560), mx.bfloat16, False) in graphs32
            assert ((4, 2, 2560), mx.bfloat16, True) not in graphs32
    finally:
        qwen35_verify_qmm.clear_group_sums()
        qwen35_verify_qmm.set_verify_qmm_armed(False)


@pytest.mark.parametrize("with_bias", [False, True])
def test_warm_verify_graph_precompiles_summed_sg8(with_bias, monkeypatch):
    """After load-time warm_model_dispatch (and direct warm_graph, even if the
    unsummed graph was already cached), the first armed summed-sg8 verify call
    must trigger zero mx.compile calls and match R2 bitwise."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()

    ql, fl = _pair(4096, 2560, bits=4, group_size=64, bias=with_bias, seed=690)
    _, fl_late = _pair(4096, 2560, bits=4, group_size=64, bias=with_bias, seed=691)

    class _Model(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.proj = layer

    model = _Model(fl)
    linear.warm_model_dispatch(model, mtp_enabled=True, prefill_lengths=())

    x8 = _x((4, 2, 2560), dtype=mx.bfloat16, seed=692)
    x8_plain = _x((4, 2, 2560), dtype=mx.bfloat16, seed=693)

    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        # Pre-populate only the unsummed graph on fl_late, then warm_graph must
        # still trace and cache the summed sg8 entry.
        mx.eval(fl_late(x8_plain))
        kept_late = fl_late.__dict__[linear._KEPT]
        assert kept_late.verify is not None
        assert ((4, 2, 2560), mx.bfloat16, False) in kept_late.verify._graphs()
        assert ((4, 2, 2560), mx.bfloat16, True) not in kept_late.verify._graphs()
        assert kept_late.verify.warm_graph((4, 2, 2560), mx.bfloat16) is True
        assert ((4, 2, 2560), mx.bfloat16, True) in kept_late.verify._graphs()

        qwen35_verify_qmm.register_group_sums(x8, qwen35_verify_qmm.group_sums(x8))
        want = ql(x8)
        want_plain = ql(x8_plain)
        with monkeypatch.context() as patch:
            patch.setattr(
                mx,
                "compile",
                lambda *a, **kw: pytest.fail("cold mx.compile on summed sg8 verify"),
            )
            got = fl(x8)
            got_plain = fl(x8_plain)
            got_late = fl_late(x8)
            mx.eval(got, want, got_plain, want_plain, got_late)
            assert _same_bits(got, want)
            assert _same_bits(got_plain, want_plain)
    finally:
        qwen35_verify_qmm.clear_group_sums()
        qwen35_verify_qmm.set_verify_qmm_armed(False)


def test_warm_model_dispatch_warms_single_stream_verify_graph(monkeypatch):
    """warm_model_dispatch must warm vb=1 single-stream verify shapes ((1, 3, K)
    and (1, 4, K)) so the first armed single-stream MTP verify does 0 compiles."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()

    ql, fl = _pair(16384, 2560, bits=8, group_size=64, seed=695)

    class _Model(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.lm_head = layer

    model = _Model(fl)
    linear.warm_model_dispatch(model, mtp_enabled=True, prefill_lengths=())

    kept = fl.__dict__[linear._KEPT]
    assert kept.verify is not None
    assert ((1, 3, 2560), mx.bfloat16, False) in kept.verify._graphs()
    assert ((1, 4, 2560), mx.bfloat16, False) in kept.verify._graphs()

    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                mx,
                "compile",
                lambda *a, **kw: pytest.fail("cold mx.compile on single-stream verify"),
            )
            for vs in (3, 4):
                x = _x((1, vs, 2560), dtype=mx.bfloat16, seed=696 + vs)
                got = fl(x)
                want = ql(x)
                mx.eval(got, want)
                assert _same_bits(got, want), vs
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)


def test_warm_engine_dispatch_and_msg_metal_warming(monkeypatch):
    """warm_engine_dispatch must invoke qwen35_verify_qmm.warm_model, and
    warm_model must submit msg_tasks for N >= 100000 layers."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()

    _, fl = _pair(4096, 2560, bits=8, group_size=64, seed=698)
    with testing.verified("qwen4_exp"):
        fl_msg = linear.FormatLinear(
            "affine",
            8,
            64,
            2560,
            {
                "weight": mx.zeros((100000, 2560 * 8 // 32), dtype=mx.uint32),
                "scales": mx.zeros((100000, 2560 // 64), dtype=mx.bfloat16),
                "biases": mx.zeros((100000, 2560 // 64), dtype=mx.bfloat16),
            },
            None,
        )

    class _Model(nn.Module):
        def __init__(self, proj, head):
            super().__init__()
            self.proj = proj
            self.head = head

    model = _Model(fl, fl_msg)
    warm_calls: list[bool] = []
    msg_ms: list[int] = []
    real_warm = qwen35_verify_qmm.warm_model
    real_msg = qwen35_verify_qmm._warm_msg_metal

    def spy_warm(m, *, mtp_enabled=False):
        warm_calls.append(mtp_enabled)
        return real_warm(m, mtp_enabled=mtp_enabled)

    def spy_msg(m, *args):
        msg_ms.append(m)
        return real_msg(m, *args)

    with monkeypatch.context() as patch:
        patch.setattr(qwen35_verify_qmm, "warm_model", spy_warm)
        patch.setattr(qwen35_verify_qmm, "_warm_msg_metal", spy_msg)
        qwen35_moe_routed_decode.warm_engine_dispatch(
            model, stream=None, prefill_lengths=(), mtp_enabled=True
        )
    assert warm_calls == [True]
    assert sorted(msg_ms) == [4, 6]


# ---------------------------------------------------------------------------
# Part C: FormatLinear (1, 1, K) native dense_fast dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_nk,qwen,with_bias",
    [
        ((4096, 2560), True, False),  # 1 not in mlx_rows (Dot -> dense_fast)
        ((4096, 2560), True, True),  # with bias
        ((2560, 2560), True, False),  # 1 in mlx_rows on M3 (MTP fc_embedding/fc_hidden)
        ((256, 256), False, False),  # non-Qwen layer
    ],
)
def test_format_linear_one_row_uses_native_dense_fast_dispatch(
    shape_nk, qwen, with_bias, monkeypatch
):
    """FormatLinear (1, 1, K) must be served by native kept.dense_fast with
    zero Python dispatch, sharing the same FormatLinearFast dispatcher with
    prefill (1, L, K), bitwise identical to the reference route."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")
    n, k = shape_nk
    ql, fl = _pair(n, k, bits=8, group_size=64, bias=with_bias, qwen=qwen, seed=701 + n)
    x1 = _x((1, 1, k), dtype=mx.bfloat16, seed=710)
    xp = _x((1, 32, k), dtype=mx.bfloat16, seed=711)

    first1 = fl(x1)
    assert fl._fast is not None, "cold (1,1,K) must bind native fast on first call"
    firstp = fl(xp)
    # After prefill builds kept.dense, the next (1, 1, K) shares kept.dense_fast.
    upgrade1 = fl(x1)
    mx.eval(first1, firstp, upgrade1)
    assert _same_bits(first1, upgrade1)
    kept = fl.__dict__[linear._KEPT]
    assert kept.dense_fast is not None
    assert fl._fast is kept.dense_fast

    fast = kept.dense_fast
    served: list[bool] = []

    def counting(module, value, _fast=fast):
        y = _fast(module, value)
        served.append(y is not None)
        return y

    with monkeypatch.context() as patch:
        fl.__dict__[linear._FAST] = counting
        patch.setattr(
            linear.FormatLinear,
            "_keep",
            lambda *a, **kw: pytest.fail("Python _keep on warm (1,1,K)"),
        )
        patch.setattr(
            linear.FormatLinear,
            "_ordinary",
            lambda *a, **kw: pytest.fail("Python _ordinary on warm (1,1,K)"),
        )
        patch.setattr(
            linear,
            "_launch",
            lambda *a, **kw: pytest.fail("Python _launch on warm (1,1,K)"),
        )
        try:
            again1 = fl(x1)
            againp = fl(xp)
            again1_b = fl(x1)
            mx.eval(again1, againp, again1_b)
        finally:
            fl.__dict__[linear._FAST] = fast

    assert served == [
        True,
        True,
        True,
    ], f"(1,1,K) or prefill missed dense_fast: {served}"
    assert _same_bits(first1, again1)
    assert _same_bits(first1, again1_b)
    assert _same_bits(firstp, againp)
    ref1 = ql(x1)
    mx.eval(ref1)
    assert _same_bits(again1, ref1)

    # Row-exact armed call on (1, 1, K) must bypass _fast and not corrupt dense_fast.
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    try:
        assert fl._fast(fl, x1) is None
        re1 = fl(x1)
        mx.eval(re1)
        assert _same_bits(re1, ref1)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
    post = fl._fast(fl, x1)
    assert post is not None
    mx.eval(post)
    assert _same_bits(post, first1)


def test_one_row_dense_fast_safety_proof_fail_tile_dense_project_and_threads(
    monkeypatch,
):
    """Safety invariants for (1, 1, K) on dense_fast:
    1. plan.proof.fail() immediately retires (1, 1, K) from dense_fast.
    2. tile_dense.project(kept.dense, x_1x1) calls _project_cold rather than
       replaying the Dot graph stored for FormatLinearFast.
    3. Multi-thread (1, 1, K) calls stay bitwise exact."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")

    ql, fl = _pair(256, 256, bits=8, group_size=64, qwen=False, seed=801)
    linear.prepare_layer_plans(fl)
    x1 = _x((1, 1, 256), dtype=mx.bfloat16, seed=802)
    ref = ql(x1)
    y = fl(x1)
    mx.eval(ref, y)
    assert _same_bits(y, ref)
    kept = fl.__dict__[linear._KEPT]
    assert fl._fast is kept.dense_fast
    assert fl._fast(fl, x1) is not None

    # Direct tile_dense.project on (1, 1, 256) must go through _project_cold
    # (never replay the Dot graph placed in kept.dense.binding for dense_fast)
    # and must never overwrite the bound Dot graph in dense.binding.kept.
    sig1 = (x1.shape, x1.dtype)
    bound_before = kept.dense.binding.kept.graphs[sig1]
    cold_calls = []
    real_cold = tile_dense._project_cold

    def spy_cold(plan, x, sig):
        cold_calls.append(sig)
        return real_cold(plan, x, sig)

    with monkeypatch.context() as patch:
        patch.setattr(tile_dense, "_project_cold", spy_cold)
        td_y = tile_dense.project(kept.dense, x1)
    assert td_y is not None
    assert cold_calls == [sig1]
    assert kept.dense.binding.kept.graphs[sig1] is bound_before
    mx.eval(td_y)
    assert td_y.shape == (1, 1, 256)

    # Multi-thread (1, 1, K) calls.
    def worker():
        out = fl(x1)
        mx.eval(out)
        return _same_bits(out, ref)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(worker)
        f2 = pool.submit(worker)
        assert f1.result() and f2.result()

    # Failing kept.plan.proof must immediately retire dense_fast replay!
    kept.plan.proof.fail()
    assert fl._fast(fl, x1) is None
    fallback = fl(x1)
    mx.eval(fallback)
    assert _same_bits(fallback, ref)


# ---------------------------------------------------------------------------
# Part D: Zero per-call host overhead on FormatLinear small rows (Rule 28)
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
    must make zero calls to verify.is_armed(), _is_armed(), packed_name(),
    builtins.len(), or _register_fast() while verify is disarmed."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()
    qwen35_verify_qmm.set_verify_qmm_armed(False)

    _, fl = _pair(4096, 2048, bits=bits, group_size=64, qwen=False, seed=901 + bits)
    x = _x(shape, dtype=mx.bfloat16, seed=910 + sum(shape))

    with monkeypatch.context() as patch:
        patch.setattr(native, "has_symbol", lambda _name: False)
        # Warm the plan graph and establish no_native on the cold call.
        warm0 = fl(x)
        mx.eval(warm0)
        assert fl._fast is None
        assert fl.__dict__[linear._KEPT].verify is not None

        is_armed_calls = 0
        orig_is_armed = qwen35_verify_qmm._FormatVerifyPlan.is_armed

        def spy_is_armed(self):
            nonlocal is_armed_calls
            is_armed_calls += 1
            return orig_is_armed(self)

        packed_calls = 0
        orig_packed = linear.packed_name

        def spy_packed(layer):
            nonlocal packed_calls
            packed_calls += 1
            return orig_packed(layer)

        reg_calls = 0
        orig_reg = linear.FormatLinear._register_fast

        def spy_reg(self, *a, **kw):
            nonlocal reg_calls
            reg_calls += 1
            return orig_reg(self, *a, **kw)

        patch.setattr(qwen35_verify_qmm._FormatVerifyPlan, "is_armed", spy_is_armed)
        patch.setattr(linear, "packed_name", spy_packed)
        patch.setattr(linear.FormatLinear, "_register_fast", spy_reg)

        pr = cProfile.Profile()
        pr.enable()
        y = warm0
        for _ in range(32):
            y = fl(x)
        pr.disable()
        mx.eval(y)

    assert is_armed_calls == 0, f"verify.is_armed() called {is_armed_calls} times"
    assert packed_calls == 0, f"packed_name() called {packed_calls} times"
    assert reg_calls == 0, f"_register_fast() called {reg_calls} times"
    callee_names = {fn[2] for fn in pstats.Stats(pr).stats if fn[2] != "disable"}
    assert "len" not in callee_names, f"per-call len() on warm path: {callee_names}"
    assert (
        "_is_armed" not in callee_names
    ), f"per-call _is_armed() on warm path: {callee_names}"


def test_bind_plan_fast_in_layer_calls_keeps_raw_graph():
    """A (1,1,K) bind inside testing.layer_calls() caches the raw graph, not
    the recorder wrapper (no leak through native dense_fast afterwards)."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")
    with testing.fresh_proofs():
        _, fl = _pair(256, 256, bits=8, group_size=64, qwen=False, seed=970)
        linear.prepare_layer_plans(fl)
        x1 = _x((1, 1, 256), dtype=mx.bfloat16, seed=971)
        with testing.layer_calls() as calls:
            mx.eval(fl(x1))
        assert len(calls) == 1
        kept = fl.__dict__[linear._KEPT]
        raw = kept.plan.proof._kept.graphs[(x1.shape, x1.dtype)]
        assert kept.dense.binding.kept.graphs[(x1.shape, x1.dtype)] is raw


def test_init_verify_active_partial_and_armed_module(monkeypatch):
    """A partially imported verify module (no lock yet) and an armed stub both
    leave the counter consistent; a concurrent import cannot raise."""
    import sys
    import types

    stub = types.ModuleType(linear._VERIFY_QMM_MOD_NAME)
    monkeypatch.setitem(sys.modules, linear._VERIFY_QMM_MOD_NAME, stub)
    linear._init_verify_active()
    assert linear._VERIFY_ACTIVE == 0
    stub._ROW_EXACT_LOCK = threading.Lock()
    stub._VERIFY_ACTIVE = 2
    try:
        linear._init_verify_active()
        assert linear._VERIFY_ACTIVE == 2
    finally:
        linear._VERIFY_ACTIVE = 0


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("with_bias", [False, True])
def test_format_linear_small_rows_bind_native_plan_fast_and_stay_bitwise(
    bits, with_bias, monkeypatch
):
    """Small-row 2-D and 3-D bare FormatLinear calls bind native plan_fast
    (including after prefill binds dense_fast), stay bitwise identical to the
    Python route, keep kept.routes empty, isolate tile_dense.project, and
    respect proof failure, weight mutation, and armed verify."""
    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()
    qwen35_verify_qmm.set_verify_qmm_armed(False)

    with testing.fresh_proofs():
        _check_small_rows_bind_native_plan_fast(bits, with_bias, monkeypatch)


def _check_small_rows_bind_native_plan_fast(bits, with_bias, monkeypatch):

    k = 2048
    _, fl = _pair(
        k, k, bits=bits, group_size=64, bias=with_bias, qwen=False, seed=920 + bits
    )
    _, fl_py = _pair(
        k, k, bits=bits, group_size=64, bias=with_bias, qwen=False, seed=920 + bits
    )

    shapes = ((1, k), (2, k), (4, k), (8, k), (2, 1, k), (4, 1, k), (1, 4, k))
    inputs = {s: _x(s, dtype=mx.bfloat16, seed=930 + sum(s)) for s in shapes}

    # Reference outputs with native disabled (pure Python Dot route).
    refs = {}
    with monkeypatch.context() as patch:
        patch.setattr(native, "has_symbol", lambda _name: False)
        for s, x in inputs.items():
            refs[s] = fl_py(x)
        mx.eval(*refs.values())

    # 1. Small 2-D rows bind plan_fast without allocating kept.routes.
    for s in ((1, k), (2, k), (4, k), (8, k)):
        x = inputs[s]
        first = fl(x)
        mx.eval(first)
        assert _same_bits(first, refs[s]), s
        kept = fl.__dict__[linear._KEPT]
        assert kept.plan_fast is not None
        assert fl._fast is kept.plan_fast
        assert kept.routes == {}
        replay = fl._fast(fl, x)
        assert replay is not None, s
        mx.eval(replay)
        assert _same_bits(replay, refs[s]), s

    # 2. Prefill binds dense_fast; subsequent 3-D 2/4-stream calls switch back
    #    to plan_fast and serve directly from C++ (0 Python fallback hits).
    xp = _x((33, k), dtype=mx.bfloat16, seed=950)
    mx.eval(fl(xp))
    kept = fl.__dict__[linear._KEPT]
    assert kept.dense_fast is not None
    assert fl._fast is kept.dense_fast

    for s in ((2, 1, k), (4, 1, k), (1, 4, k)):
        x = inputs[s]
        first = fl(x)
        mx.eval(first)
        assert _same_bits(first, refs[s]), s
        assert fl._fast is kept.plan_fast
        served: list[bool] = []
        fast = kept.plan_fast

        def counting(module, value, _fast=fast, _served=served):
            y = _fast(module, value)
            _served.append(y is not None)
            return y

        with monkeypatch.context() as patch:
            fl.__dict__[linear._FAST] = counting
            patch.setattr(
                linear.FormatLinear,
                "_keep",
                lambda *a, **kw: pytest.fail("Python _keep on warm small-row"),
            )
            patch.setattr(
                qwen35_verify_qmm._FormatVerifyPlan,
                "is_armed",
                lambda *a, **kw: pytest.fail("verify.is_armed on warm small-row"),
            )
            try:
                again = fl(x)
                mx.eval(again)
            finally:
                fl.__dict__[linear._FAST] = fast
        assert served == [True], s
        assert _same_bits(again, refs[s]), s

    # 3. Direct tile_dense.project on 4 rows must never see a Dot graph from plan_fast.
    assert kept.dense is not None
    assert ((4, k), mx.bfloat16) not in (
        kept.dense.binding.kept.graphs if kept.dense.binding.kept is not None else {}
    )
    assert ((1, 4, k), mx.bfloat16) not in (
        kept.dense.binding.kept.graphs if kept.dense.binding.kept is not None else {}
    )
    td_out = tile_dense.project(kept.dense, inputs[(4, k)])
    assert td_out is not None
    mx.eval(td_out)
    assert td_out.shape == (4, k)

    # 3b. A tile of at least 16 operands refuses native before caching.
    class _Wide:
        failed = False
        operands = tuple(range(16))

    saved = (kept.plan, kept.dense, kept.plan_fast, kept.dense_fast, kept.no_native)
    try:
        kept.plan = _Wide()
        kept.dense = _Wide()
        kept.plan_fast = None
        kept.dense_fast = None
        kept.no_native = False
        graph = object()
        from types import SimpleNamespace

        state = {"_kept": SimpleNamespace(graphs={((1, k), mx.bfloat16): graph})}
        fl._register_fast(kept, kept.plan, ((1, k), mx.bfloat16), proof_state=state)
        assert kept.plan_fast is None
        assert kept.dense_fast is None
        assert kept.no_native is True
        kept.no_native = False
        fl._register_fast(kept, kept.dense, ((1, k), mx.bfloat16), proof_state=state)
        assert kept.plan_fast is None
        assert kept.dense_fast is None
        assert kept.no_native is True
    finally:
        (
            kept.plan,
            kept.dense,
            kept.plan_fast,
            kept.dense_fast,
            kept.no_native,
        ) = saved

    # 4. Multi-threaded warm plan_fast stays bitwise identical.
    x2 = inputs[(2, 1, k)]

    def worker():
        out = fl(x2)
        mx.eval(out)
        return _same_bits(out, refs[(2, 1, k)])

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(worker)
        f2 = pool.submit(worker)
        assert f1.result() and f2.result()

    # 5. Armed verify bypasses plan_fast; disarming restores unpoisoned replay.
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        assert fl._fast(fl, x2) is None
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
    post = fl._fast(fl, x2)
    assert post is not None
    mx.eval(post)
    assert _same_bits(post, refs[(2, 1, k)])

    # 5b. Armed row-exact calls (cold and warm) never bind plan_fast; >=16
    #     operands marks kept.no_native without setting _fast.
    if not with_bias:
        _, fl_re = _pair(k, k, bits=bits, group_size=64, qwen=True, seed=955 + bits)
        x1 = inputs[(1, k)]
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
        try:
            mx.eval(fl_re(x1))
            assert fl_re._fast is None
            mx.eval(fl_re(x1))
            assert fl_re._fast is None
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False)

        _, fl_wide_ops = _pair(
            256, 256, bits=bits, group_size=64, qwen=False, seed=956 + bits
        )
        kept_ops = fl_wide_ops._keep()
        sig_ops = ((1, 256), mx.bfloat16)
        kept_ops.plan.proof.keep(sig_ops, lambda x, *ops: x)
        fake_tile = type(
            "_BigTile",
            (),
            {"operands": (kept_ops.weight,) * 16, "proof": kept_ops.plan.proof},
        )()
        fl_wide_ops._register_fast(kept_ops, fake_tile, sig_ops)
        assert kept_ops.no_native is True
        assert fl_wide_ops._fast is None

    # 6. Weight mutation retires plan_fast cleanly; plan.proof.fail() retires
    #    plan_fast immediately and falls back bitwise on 1-row stock matmul.
    if with_bias:
        fl["weight"] = mx.array(fl["weight"])
        assert fl._fast is None
        rebound = fl(x2)
        mx.eval(rebound)
        assert _same_bits(rebound, refs[(2, 1, k)])
        assert fl._fast is fl.__dict__[linear._KEPT].plan_fast
    else:
        kept.plan.proof.fail()
        x1_2d = inputs[(1, k)]
        assert fl._fast(fl, x2) is None
        assert fl._fast(fl, x1_2d) is None
        fb = fl(x1_2d)
        mx.eval(fb)
        assert _same_bits(fb, refs[(1, k)])


def test_verify_active_counter_transitions_threads_and_unpatched_guard(monkeypatch):
    """linear._VERIFY_ACTIVE tracks armed non-row-exact verify threads cleanly
    across idempotent calls, row_exact transitions, concurrent threads, and
    _QL_PATCHED=False guards."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    assert qwen35_verify_qmm.apply_verify_qmm_patch()
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert linear._VERIFY_ACTIVE == 0

    # Idempotent disarm and arm on one thread.
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert linear._VERIFY_ACTIVE == 0
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        assert linear._VERIFY_ACTIVE == 1
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
        assert linear._VERIFY_ACTIVE == 1
        # Transition verify -> row_exact clears _VERIFY_ACTIVE.
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
        assert linear._VERIFY_ACTIVE == 0
        # Transition row_exact -> verify restores _VERIFY_ACTIVE.
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
        assert linear._VERIFY_ACTIVE == 1
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert linear._VERIFY_ACTIVE == 0

    # Concurrent threads: 2 armed threads give _VERIFY_ACTIVE == 2, then 0.
    barrier = threading.Barrier(2)
    observed: list[int] = []

    def arm_worker():
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
        try:
            barrier.wait(timeout=5.0)
            observed.append(linear._VERIFY_ACTIVE)
            barrier.wait(timeout=5.0)
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(arm_worker)
        f2 = pool.submit(arm_worker)
        f1.result()
        f2.result()
    assert observed == [2, 2]
    assert linear._VERIFY_ACTIVE == 0

    # Unpatched guard while _VERIFY_ACTIVE > 0 must still skip verify route.
    _, fl = _pair(4096, 2560, bits=8, group_size=64, qwen=False, seed=960)
    x_ver = _x((4, 2, 2560), dtype=mx.bfloat16, seed=961)
    unarmed_ref = fl(x_ver)
    mx.eval(unarmed_ref)
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        assert linear._VERIFY_ACTIVE == 1
        with monkeypatch.context() as patch:
            patch.setattr(qwen35_verify_qmm, "_QL_PATCHED", False)
            got = fl(x_ver)
            mx.eval(got)
            assert _same_bits(got, unarmed_ref)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert linear._VERIFY_ACTIVE == 0
