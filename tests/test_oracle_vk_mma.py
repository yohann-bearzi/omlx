# SPDX-License-Identifier: Apache-2.0
"""The frozen oracle of the legacy ``vk_mma`` kernel
(``tests/oracles/vk_mma.py``): a verbatim copy while the live kernel exists,
and a reference that tells its rows apart bit for bit."""

from __future__ import annotations

import inspect
from dataclasses import replace

import mlx.core as mx
import pytest

from omlx.formats import affine
from omlx.patches import qwen35_verify_qmm as live
from tests.oracles import vk_mma as oracle
from tests.test_formats_tables import tabled_tile
from tests.test_oracle_vk import _linear, _same, _x

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# What the oracle copies of the live module that the live module still has.
# The kernel (_build_mma_kernel) moved onto the tile face, and vk_qmm_mma
# with it: they are the oracle's alone, and the rows below compare them.
_FUNCTIONS = ("_pad_rows",)
_CONSTANTS = ("_MMA_BN", "_MMA_BK", "_MMA_NSG")


def test_the_oracle_is_the_live_code_verbatim():
    for name in _FUNCTIONS:
        assert inspect.getsource(getattr(oracle, name)) == inspect.getsource(
            getattr(live, name)
        ), name
    for name in _CONSTANTS:
        assert getattr(oracle, name) == getattr(live, name), name


# Every width and group size the kernel takes, both dtypes; rows from 1 to
# 24 (one to three row tiles, padded and whole); K whose simdgroup chunks
# (K / 8 each) end inside a group (256, 768) and on one (1024).
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("k", [1024, 768, 256])
def test_the_tile_kernel_is_the_oracle_s(dtype, bits, group_size, k):
    if k % group_size:
        pytest.skip("MLX quantizes K in whole groups")
    n = 4096
    w = _linear(n, k, bits, group_size, dtype, seed=bits * group_size)
    for rows in (1, 7, 8, 13, 16, 17, 23, 24):
        x = _x(rows, k, dtype, seed=rows)
        kw = dict(bits=bits, group_size=group_size)
        got = live.vk_qmm_mma(x, *w, **kw)
        assert got.shape == (rows, n)
        assert _same(got, oracle.vk_qmm_mma(x, *w, **kw))


# A K of several chunks per simdgroup (8192) and a wide N.
def test_a_long_k_and_a_wide_n_are_the_oracle_s():
    for k, n in ((8192, 4096), (2048, 16384)):
        w = _linear(n, k, 4, 64, mx.bfloat16, seed=k)
        x = _x(11, k, mx.bfloat16, seed=k)
        kw = dict(bits=4, group_size=64)
        assert _same(live.vk_qmm_mma(x, *w, **kw), oracle.vk_qmm_mma(x, *w, **kw))


@pytest.mark.parametrize(
    "rows,k,n", [(25, 256, 4096), (0, 256, 4096), (8, 128, 4096), (8, 256, 4104)]
)
def test_a_shape_the_kernel_does_not_take_is_refused(rows, k, n):
    w = _linear(n, k, 4, 64, mx.bfloat16, seed=1)
    x = mx.zeros((rows, k), mx.bfloat16)
    with pytest.raises(ValueError, match="vk_qmm_mma takes 1 to 24 rows"):
        live.vk_qmm_mma(x, *w, group_size=64, bits=4)


def test_scales_no_tile_face_reads_are_left_to_the_stock_qmm():
    weight, scales, biases = _linear(4096, 256, 4, 64, mx.bfloat16, seed=1)
    x = _x(8, 256, mx.bfloat16, seed=1)
    wide = (scales.astype(mx.float32), biases.astype(mx.float32))
    assert live.vk_qmm_mma(x, weight, *wide, group_size=64, bits=4) is None


def test_mixed_dtypes_and_mismatched_biases():
    weight, scales, biases = _linear(4096, 256, 4, 64, mx.bfloat16, seed=1)
    x = _x(8, 256, mx.bfloat16, seed=1)
    for dtype in (mx.float32, mx.float16):
        assert (
            live.vk_qmm_mma(
                x, weight, scales, biases.astype(dtype), group_size=64, bits=4
            )
            is None
        )
    wide = _x(8, 256, mx.float32, seed=1)
    assert _same(
        live.vk_qmm_mma(wide, weight, scales, biases, group_size=64, bits=4),
        oracle.vk_qmm_mma(wide, weight, scales, biases, group_size=64, bits=4),
    )


def test_mma_plan_routing_and_forged_keys():
    four = affine.tile_face(4, 64, mx.bfloat16)
    live._mma_plan.cache_clear()
    try:
        from omlx.formats import kernels as face_kernels

        plain = live._mma_plan(1, four, 2560, mx.bfloat16)
        assert plain[1] == [("T", mx.bfloat16)]
        assert not isinstance(plain[0], face_kernels._FaceKernel)
        forged = replace(four, key="affine_tile_b7g64_bf16")
        tabled = tabled_tile(four)
        altered = replace(four, tables=tabled.tables)
        for face in (forged, altered, tabled):
            plan = live._mma_plan(1, face, 2560, mx.bfloat16)
            assert plan[1] == [("T", mx.bfloat16)]
            assert isinstance(plan[0], face_kernels._FaceKernel)
    finally:
        live._mma_plan.cache_clear()


def test_mma_launch_reuses_the_plan_across_n():
    live._mma_plan.cache_clear()
    live._mma_launch.cache_clear()
    try:
        live._mma_launch(1, 4, 64, mx.bfloat16, mx.bfloat16, 2560, 4096)
        live._mma_launch(1, 4, 64, mx.bfloat16, mx.bfloat16, 2560, 8192)
        assert live._mma_plan.cache_info().misses == 1
    finally:
        live._mma_plan.cache_clear()
        live._mma_launch.cache_clear()


def test_zero_k_or_n_is_refused_when_the_plan_is_built():
    live._mma_launch.cache_clear()
    try:
        with pytest.raises(ValueError):
            live._mma_launch(1, 8, 64, mx.bfloat16, mx.bfloat16, 2560, 0)
        with pytest.raises(ValueError):
            live._mma_launch(1, 8, 64, mx.bfloat16, mx.bfloat16, 0, 4096)
    finally:
        live._mma_launch.cache_clear()


def test_mma_plan_routes_tabled_faces_through_the_skeleton(monkeypatch):
    from omlx.formats import kernels, verify_skeleton

    face = tabled_tile(affine.tile_face(4, 64, mx.bfloat16))
    seen = []

    def record(name, inputs, faces, outputs, header, source):
        seen.append((name, inputs, faces, outputs, header, source))
        return None

    monkeypatch.setattr(kernels, "face_kernel", record)
    live._mma_plan.cache_clear()
    try:
        kernel, template = live._mma_plan(1, face, 2560, mx.bfloat16)
        assert kernel is None and template == [("T", mx.bfloat16)]
        ops = verify_skeleton.tile_mma_ops(face)
        want = verify_skeleton.emit_mma_source(ops, 1, 2560)
        assert seen[0] == (
            "omlx_vk_mma_rt1_k2560",
            ("x", "w", "N_size"),
            {"w": face},
            ("y",),
            ops.header,
            want,
        )
    finally:
        live._mma_plan.cache_clear()


def test_mma_plan_runs_tabled_face_and_enforces_tile_contract():
    from tests.test_formats_tile import _TOY, _toy, _toy_matrix

    w = _linear(64, 256, 4, 64, mx.bfloat16, seed=1)
    x = _x(8, 256, mx.bfloat16, seed=1)
    n_scalar = mx.array(64, dtype=mx.int32)
    live._mma_plan.cache_clear()
    try:
        face = tabled_tile(affine.tile_face(4, 64, mx.bfloat16))
        kernel, template = live._mma_plan(1, face, 256, mx.bfloat16)
        (y,) = kernel(
            inputs=[x, *w, n_scalar],
            template=template,
            grid=(256, 4, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(8, 64)],
            output_dtypes=[mx.bfloat16],
        )
        mx.eval(y)
        assert _same(y, oracle.vk_qmm_mma(x, *w, group_size=64, bits=4))
        for tag, source, message in (
            ("mma_run4", _TOY.replace("RUN = 8;", "RUN = 4;"), "RUN must be 8"),
            (
                "mma_rev99",
                _TOY.replace("CONTRACT = 1;", "CONTRACT = 99;").replace(
                    "Codes fetch(int run) const {", "Codes fetches(int run) const {"
                ),
                "contract",
            ),
        ):
            bad = _toy(tag, source)
            kernel, template = live._mma_plan(1, bad, 256, mx.bfloat16)
            with pytest.raises(RuntimeError, match=message) as failed:
                (out,) = kernel(
                    inputs=[x, *_toy_matrix(64, 256), n_scalar],
                    template=template,
                    grid=(256, 4, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(8, 64)],
                    output_dtypes=[mx.bfloat16],
                )
                mx.eval(out)
            assert str(failed.value).count("error:") == 1
    finally:
        live._mma_plan.cache_clear()


def test_mma_plan_runs_an_fp16_toy_face_with_two_row_tiles():
    from tests.test_formats_tile import _toy, _toy_decode, _toy_matrix

    toy = _toy("mma_fp16")
    codes, group_scales, row_scales = _toy_matrix(64, 512, seed=5, group=32)
    x = _x(16, 512, mx.float16, seed=7)
    live._mma_plan.cache_clear()
    try:
        kernel, template = live._mma_plan(2, toy, 512, mx.float16)
        (y,) = kernel(
            inputs=[
                x,
                codes,
                group_scales,
                row_scales,
                mx.array(64, dtype=mx.int32),
            ],
            template=template,
            grid=(256, 4, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(16, 64)],
            output_dtypes=[mx.float16],
        )
        mx.eval(y)
        weights = _toy_decode(codes, group_scales, row_scales, group=32)
        reference = x.astype(mx.float32) @ weights.astype(mx.float16).astype(
            mx.float32
        ).T
        assert mx.allclose(y.astype(mx.float32), reference, rtol=2e-2, atol=5e-2)
    finally:
        live._mma_plan.cache_clear()
