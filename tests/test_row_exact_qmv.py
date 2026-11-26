# SPDX-License-Identifier: Apache-2.0
"""Row-exact verify projections: every row equals a one-row decode call.

Lightning MTP greedy output matches MTP-off output only if each verify row's
quantized projection has the serial step's bits; stock multi-row kernels
(qmv_wide, qmm) do not.
"""

from __future__ import annotations

import gc
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import row_exact_skeleton, testing
from omlx.patches import qwen35_verify_qmm, row_exact_qmv
from tests.test_formats_affine import MLX_MASKS, MLX_SCALES, mlx_branches
from tests.test_formats_kernels import _toy
from tests.test_formats_linear import _toy_linear, _ToyFormat

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def disarm():
    row_exact_qmv._UNROLLED_VERDICTS.clear()
    yield
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    row_exact_qmv._UNROLLED_VERDICTS.clear()


def _linear(k, n, bits, group_size, seed):
    mx.random.seed(seed)
    linear = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    weight = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=group_size, bits=bits
    )
    return linear


def _serial(linear, x):
    return mx.concatenate([linear(x[:, r : r + 1]) for r in range(x.shape[1])], axis=1)


def _bit_equal(a, b):
    return mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


# (K, N, bits, group_size): Qwen4 oQ5e projections (qmv_fast), the non-fast
# qmv arm (K off the qmv_fast block), partial output tiles (N % 8) and N < 8,
# other widths, and K = 256 (mod 512) at 6/8 bits, where MLX's qmv_fast block
# is 256 wide (512 at 4/5 bits).
SHAPES = [
    (2560, 10240, 6, 64),
    (6144, 2560, 5, 128),
    (2560, 16384, 8, 64),  # 2048 tiles: rows share weight passes, as in the lm_head
    (640, 2560, 8, 128),
    (2560, 1, 8, 64),
    (2560, 12, 6, 64),
    (320, 1024, 6, 64),
    (2560, 1024, 4, 64),
    (2560, 1024, 5, 32),
    (768, 1024, 8, 64),
    (1280, 1024, 6, 64),
    (640, 1024, 4, 64),  # qmv's last block at 4 and 5 bits
    (640, 1024, 5, 32),
]


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_a_tile_s_source_holds_its_own_bit_width_alone(bits, fast):
    """MLX hashes a kernel's whole source at every launch: a tile holds the
    branch of MLX's code for its width. A lane has all of its inputs in
    qmv_impl's last block or none: the skeleton reads it with load and
    apply, as a whole block, and the safe loads are not there."""
    ops = row_exact_skeleton.affine_row_ops(bits, 64, fast)
    spec = row_exact_skeleton.kernel_spec(ops, bits, 64, fast, False, 0)
    metal = spec["header"] + spec["source"]
    assert mlx_branches(metal) == (MLX_MASKS[bits], MLX_SCALES[bits])
    assert "_safe" not in metal


def test_shared_down_reuses_its_shape_bound_kernel_across_verify_lengths():
    k, n = 640, 2560
    linear = _linear(k, n, 8, 128, seed=7)
    plans = []
    for rows in (2, 4):
        x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
        got = row_exact_qmv.quantized_linear(linear, x)
        assert _bit_equal(got, _serial(linear, x))
        plan = row_exact_qmv._plan(linear, x, rows)
        assert plan.template == [
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", 1),
            ("RPS", 4),
        ]
        plans.append(plan)
    assert plans[0].kernel is plans[1].kernel
    assert plans[0].grid != plans[1].grid


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rows", [3, 8, 16])
def test_rows_equal_one_row_quantized_matmul(shape, rows):
    k, n, bits, group_size = shape
    linear = _linear(k, n, bits, group_size, seed=k + n + rows)
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    assert _bit_equal(row_exact_qmv.quantized_linear(linear, x), _serial(linear, x))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bits", [4, 5, 6])
@pytest.mark.parametrize("k", [512, 640])
@pytest.mark.parametrize("n", [8, 12])
def test_bias_sum_widens_inputs_before_accumulation(dtype, bits, k, n):
    linear = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=bits)
    linear.weight = mx.zeros((n, k * bits // 32), dtype=mx.uint32)
    linear.scales = mx.ones((n, k // 64), dtype=dtype)
    linear.biases = mx.ones((n, k // 64), dtype=dtype)
    denominator = 2048 if dtype == mx.float16 else 256
    # Adding the first two terms in the input dtype loses the small term.
    row = [1, 1 / denominator, -1, 0] * (k // 4)
    x = mx.array([[row, row, row]], dtype=dtype)
    expected = mx.full((1, 3, n), k / (4 * denominator), dtype=dtype)
    assert _bit_equal(_serial(linear, x), expected)
    assert _bit_equal(row_exact_qmv.quantized_linear(linear, x), expected)


@pytest.mark.parametrize(
    "k, sizes, bits, group_size",
    [
        (2560, [10240, 6144, 48, 48], 6, 64),  # DeltaNet in_proj qkv/z/b/a
        (2560, [12288, 512, 512], 6, 64),  # attention q/k/v
        (2560, [640, 640], 8, 128),  # shared expert gate/up
        (640, [2560, 1, 12], 8, 128),  # non-fast K with partial tiles
    ],
)
@pytest.mark.parametrize("rows", [2, 4])
def test_grouped_rows_equal_one_row_quantized_matmul(k, sizes, bits, group_size, rows):
    linears = [_linear(k, n, bits, group_size, seed=i + k) for i, n in enumerate(sizes)]
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    outputs = row_exact_qmv.quantized_linears(linears, x)
    assert len(outputs) == len(linears)
    for linear, output in zip(linears, outputs):
        assert _bit_equal(output, _serial(linear, x))


@pytest.mark.parametrize("rows", [5, 7, 11, 13])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("grouped", [False, True])
def test_partial_shared_row_tiles_equal_serial(rows, dtype, grouped):
    linears = [_linear(2560, n, 8, 64, seed=rows + n) for n in (65536, 16)]
    for linear in linears:
        linear.scales = linear.scales.astype(dtype)
        linear.biases = linear.biases.astype(dtype)
    x = mx.random.normal((1, rows, 2560)).astype(dtype)
    if grouped:
        assert row_exact_qmv._group_plan(linears, x, rows) is not None
        outputs = row_exact_qmv.quantized_linears(linears, x)
    else:
        linears[0]["bias"] = mx.random.normal((65536,)).astype(dtype)
        outputs = [row_exact_qmv.quantized_linear(linears[0], x)]
    for linear, output in zip(linears, outputs):
        expected = _serial(linear, x)
        assert output.shape == expected.shape and output.dtype == dtype
        assert _bit_equal(output, expected)


def test_partial_row_tile_plan_reads_current_weights():
    linear = _linear(2560, 65536, 8, 64, seed=7)
    x = mx.random.normal((1, 7, 2560)).astype(mx.bfloat16)
    first = row_exact_qmv.quantized_linear(linear, x)
    mx.eval(first)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        (mx.random.normal((65536, 2560)) * 0.05).astype(mx.bfloat16),
        group_size=64,
        bits=8,
    )
    current = row_exact_qmv.quantized_linear(linear, x)
    assert _bit_equal(current, _serial(linear, x))
    assert not _bit_equal(first, current)


def test_row_exact_arming_routes_multi_row_quantized_linear():
    qwen35_verify_qmm.apply_verify_qmm_patch()
    linear = _linear(2560, 6144, 6, 64, seed=3)
    x = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    reference = _serial(linear, x)
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    routed = linear(x)
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    assert _bit_equal(routed, reference)


def _quantized(k, n, bits, group_size, dtype, seed):
    mx.random.seed(seed)
    weight = (mx.random.normal((n, k)) * 0.05).astype(dtype)
    return mx.quantize(weight, group_size=group_size, bits=bits)


@pytest.mark.parametrize(
    "bits, group_size, words",
    [(7, 64, "no decode of 7-bit weights"), (4, 8, "no MLX transcription")],
)
def test_a_tile_is_built_only_for_a_width_mlx_has_code_for(bits, group_size, words):
    weight = mx.zeros((8, 512 * bits // 32), mx.uint32)
    scales = mx.zeros((8, 512 // group_size), mx.bfloat16)
    with pytest.raises(ValueError, match=words):
        row_exact_qmv.RowsQmv(
            weight, scales, scales, bits, group_size, mx.bfloat16, lambda _: (1, 1)
        )


@pytest.mark.parametrize("k, n", [(640, 2560), (2560, 1)])
def test_rows_qmv_declines_unsupported_layouts(k, n):
    weight, scales, biases = _quantized(k, n, 8, 64, mx.bfloat16, 1)
    assert (
        row_exact_qmv.rows_qmv(
            weight, scales, biases, 8, 64, "affine", mx.bfloat16, lambda _: (1, 1)
        )
        is None
    )


def test_rows_qmv_rejects_mismatched_k_or_dtype():
    qmv = row_exact_qmv.rows_qmv(
        *_quantized(2560, 64, 8, 64, mx.bfloat16, 1),
        8,
        64,
        "affine",
        mx.bfloat16,
        lambda _: (1, 1),
    )
    assert qmv is not None
    with pytest.raises(ValueError, match="row-exact tile"):
        qmv(mx.zeros((2, 1280), mx.bfloat16))
    with pytest.raises(ValueError, match="row-exact tile"):
        qmv(mx.zeros((1, 2560), mx.float16))
    with pytest.raises(ValueError, match="row-exact tile"):
        qmv(mx.zeros((), mx.bfloat16))
    with pytest.raises(ValueError, match="row-exact tile"):
        qmv(mx.zeros((1, 5120), mx.bfloat16))
    assert qmv(mx.zeros((2560,), mx.bfloat16)).shape == (64,)


def test_format_rows_qmv_rejects_mismatched_k_dtype_or_scalar():
    weight, scales, biases = _quantized(2560, 64, 8, 64, mx.bfloat16, 1)
    with testing.verified("qwen4_exp"):
        layer = row_exact_qmv.FormatLinear(
            "affine",
            8,
            64,
            2560,
            {"weight": weight, "scales": scales, "biases": biases},
        )
    project = row_exact_qmv.format_rows_qmv(layer, mx.bfloat16, lambda _: (1, 1))
    assert project is not None
    for bad in (
        mx.zeros((2, 1280), mx.bfloat16),
        mx.zeros((1, 2560), mx.float16),
        mx.zeros((), mx.bfloat16),
        mx.zeros((1, 5120), mx.bfloat16),
    ):
        with pytest.raises(ValueError, match="row-exact tile"):
            project(bad)
    assert project(mx.zeros((2560,), mx.bfloat16)).shape == (64,)


def test_rows_qmv_rejects_an_n_the_tile_does_not_divide():
    weight, scales, biases = _quantized(2560, 40, 8, 64, mx.bfloat16, 1)
    qmv = row_exact_qmv.RowsQmv(
        weight, scales, biases, 8, 64, mx.bfloat16, lambda _: (8, 1)
    )
    with pytest.raises(ValueError, match="tile"):
        qmv(mx.zeros((2, 2560), mx.bfloat16))


@pytest.mark.parametrize("bits", [2, 3])
def test_the_tile_declines_the_widths_it_does_not_decode(bits):
    """MLX's 2- and 3-bit layers keep their one-row calls (the group tile
    decodes 4, 5, 6 and 8 bits), and so do 2- and 3-bit affine tensors taken
    by position."""
    weight, scales, biases = _quantized(2560, 64, bits, 64, mx.bfloat16, 1)
    assert (
        row_exact_qmv.rows_qmv(
            weight, scales, biases, bits, 64, "affine", mx.bfloat16, lambda _: (1, 1)
        )
        is None
    )
    x = mx.zeros((1, 2560), mx.bfloat16)
    assert row_exact_qmv._face(_linear(2560, 64, bits, 64, 1), x) is None
    with pytest.raises(ValueError, match=f"no decode of {bits}-bit weights"):
        row_exact_qmv.RowsQmv(
            weight, scales, biases, bits, 64, mx.bfloat16, lambda _: (1, 1)
        )


# Every (columns per simdgroup, rows per threadgroup) tile gives each row the
# one-row bits, so retuning the geometry cannot change a verify row.
@pytest.mark.parametrize(
    "k, n, bits, group_size",
    [
        (2560, 16480, 6, 64),  # DeltaNet stacked qkv/z/b/a in-projection
        (6144, 2560, 5, 128),  # DeltaNet out-projection
        (2560, 1024, 8, 64),
    ],
)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows", [3, 4, 9])
def test_rows_qmv_every_geometry_equals_one_row_quantized_matmul(
    k, n, bits, group_size, dtype, rows
):
    weight, scales, biases = _quantized(k, n, bits, group_size, dtype, k + rows)
    x = mx.random.normal((1, rows, k)).astype(dtype)
    expected = mx.concatenate(
        [
            mx.quantized_matmul(
                x[:, r : r + 1],
                weight,
                scales,
                biases,
                transpose=True,
                group_size=group_size,
                bits=bits,
            )
            for r in range(rows)
        ],
        axis=1,
    )
    view = mx.uint32 if dtype == mx.float32 else mx.uint16
    for rps in (1, 2, 4):
        for per_group in (d for d in (1, 2, 3, 4) if rows % d == 0):
            launch = row_exact_qmv.rows_qmv(
                weight,
                scales,
                biases,
                bits,
                group_size,
                "affine",
                dtype,
                lambda _, g=(rps, per_group): g,
            )
            observed = launch(x)
            assert observed.shape == expected.shape and observed.dtype == dtype
            assert mx.array_equal(observed.view(view), expected.view(view)).item(), (
                rps,
                per_group,
            )


# The unrolled tile keeps the rolled tile's per-accumulator order, so every
# tile inside its register envelope gives each row the one-row bits (at the
# DeltaNet verify shapes and bit widths, and past the 4-row blocks).
@pytest.mark.parametrize(
    "k, n, bits, group_size",
    [
        (2560, 16480, 6, 64),  # DeltaNet stacked in-projection (oQ5e)
        (2560, 16480, 8, 64),  # its 8-bit layer
        (6144, 2560, 5, 128),  # DeltaNet out-projection
        (2560, 1024, 4, 64),
    ],
)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_unrolled_rows_qmv_tiles_equal_one_row_quantized_matmul(
    k, n, bits, group_size, dtype, rows
):
    weight, scales, biases = _quantized(k, n, bits, group_size, dtype, k + n + rows)
    x = mx.random.normal((1, rows, k)).astype(dtype)
    expected = mx.concatenate(
        [
            mx.quantized_matmul(
                x[:, r : r + 1],
                weight,
                scales,
                biases,
                transpose=True,
                group_size=group_size,
                bits=bits,
            )
            for r in range(rows)
        ],
        axis=1,
    )
    view = mx.uint32 if dtype == mx.float32 else mx.uint16
    tiles = [
        (rps, per_group)
        for rps in (1, 2, 4, 8)
        for per_group in range(1, rows + 1)
        if rows % per_group == 0 and row_exact_qmv._unrolled_ok(bits, rps, per_group)
    ]
    for tile in tiles:
        launch = row_exact_qmv.rows_qmv(
            weight,
            scales,
            biases,
            bits,
            group_size,
            "affine",
            dtype,
            lambda _, g=tile: g,
            unrolled=True,
        )
        observed = launch(x)
        assert observed.shape == expected.shape and observed.dtype == dtype
        assert mx.array_equal(observed.view(view), expected.view(view)).item(), tile


# Tiles past the envelope compiled to wrong bits (5-bit 8x4, 4-bit 2x8).
@pytest.mark.parametrize(
    "k, n, bits, group_size, tile",
    [
        (6144, 2560, 5, 128, (8, 4)),
        (2560, 1024, 4, 64, (2, 8)),
    ],
)
def test_unrolled_rows_qmv_refuses_tiles_past_the_register_envelope(
    k, n, bits, group_size, tile
):
    weight, scales, biases = _quantized(k, n, bits, group_size, mx.bfloat16, 3)
    launch = row_exact_qmv.rows_qmv(
        weight,
        scales,
        biases,
        bits,
        group_size,
        "affine",
        mx.bfloat16,
        lambda _: tile,
        unrolled=True,
    )
    with pytest.raises(ValueError):
        launch(mx.zeros((1, 8, k), dtype=mx.bfloat16))


# The header directive that marks an unrolled row-exact kernel (the name's
# _u suffix is not enough: SIZE_T64 appends _i64 after it).
_UNROLLED_HEADER = '#define ROW_EXACT_UNROLL _Pragma("clang loop unroll(full)")'


def _miscompiled(factory):
    """``factory`` whose unrolled kernels return every output off by one, as a
    GPU whose compiler gets the unrolled tile wrong would."""

    def make(spec):
        kernel = factory(spec)
        if _UNROLLED_HEADER not in spec["header"]:
            return kernel
        return lambda **kw: [y + 1 for y in kernel(**kw)]

    return make


def test_miscompiled_wraps_unrolled_size_t64_and_keyword_spec():
    """The test helper sees an unrolled SIZE_T64 kernel (name _u_i64) and
    accepts the spec by keyword, as compile_kernel does."""
    spec = row_exact_skeleton._affine_spec(
        8, 64, True, True, 0, row_exact_skeleton.RowIndexWidth.SIZE_T64
    )
    wrapped = _miscompiled(lambda s: (lambda **kw: [0]))
    assert wrapped(spec)() == [1]
    assert wrapped(spec=spec)() == [1]


# A GPU that computes an unrolled tile wrong (an M3 Ultra did at 4-bit gs64)
# must still get one-row bits: the first launch of each shape is checked and
# falls back to the rolled tile.
def test_rows_qmv_falls_back_when_this_gpu_gets_the_unrolled_tile_wrong(monkeypatch):
    k, n, bits, group_size, rows = 2560, 1024, 4, 64, 4
    weight, scales, biases = _quantized(k, n, bits, group_size, mx.bfloat16, 5)
    x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
    expected = mx.concatenate(
        [
            mx.quantized_matmul(
                x[:, r : r + 1],
                weight,
                scales,
                biases,
                transpose=True,
                group_size=group_size,
                bits=bits,
            )
            for r in range(rows)
        ],
        axis=1,
    )
    monkeypatch.setattr(
        row_exact_skeleton,
        "compile_kernel",
        _miscompiled(row_exact_skeleton.compile_kernel),
    )
    launch = row_exact_qmv.rows_qmv(
        weight,
        scales,
        biases,
        bits,
        group_size,
        "affine",
        mx.bfloat16,
        lambda _: (2, 2),
        unrolled=True,
    )
    assert _bit_equal(launch(x), expected)
    assert list(row_exact_qmv._UNROLLED_VERDICTS.values()) == [False]


def test_grouped_tiles_fall_back_when_this_gpu_gets_the_unrolled_tile_wrong(
    monkeypatch,
):
    linears = [_linear(2560, n, 4, 64, seed) for seed, n in enumerate((1024, 512))]
    x = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    monkeypatch.setattr(
        row_exact_skeleton,
        "compile_kernel",
        _miscompiled(row_exact_skeleton.compile_kernel),
    )
    outputs = row_exact_qmv.quantized_linears_tiled(linears, x, 2, 4, True)
    for linear, observed in zip(linears, outputs):
        assert _bit_equal(observed, _serial(linear, x))
    assert list(row_exact_qmv._UNROLLED_VERDICTS.values()) == [False]


def test_engine_threads_keep_the_verdict_of_each_output_width(monkeypatch):
    # Model a compiler that gets only the second width's unrolled kernel wrong.
    widths = (16384, 16392)
    launches = {n: [] for n in widths}
    together = Barrier(2, timeout=30)
    factory = row_exact_skeleton.compile_kernel

    def observe(spec):
        kernel = factory(spec)
        unrolled = _UNROLLED_HEADER in spec["header"]

        def launch(**params):
            n = params["output_shapes"][0][-1]
            if not launches[n]:
                together.wait()
            launches[n].append(unrolled)
            outputs = kernel(**params)
            return [y + 1 for y in outputs] if unrolled and n == widths[1] else outputs

        return launch

    monkeypatch.setattr(row_exact_skeleton, "compile_kernel", observe)
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_VERDICTS", {})
    layers = [_linear(512, n, 8, 64, seed=n) for n in widths]
    x = mx.random.normal((1, 4, 512), key=mx.random.key(8)).astype(mx.bfloat16)
    expected = [_serial(layer, x) for layer in layers]
    mx.eval(x, expected, [layer.parameters() for layer in layers])

    def call(layer, reference):
        for _ in range(2):
            (got,) = row_exact_qmv.quantized_linears_tiled((layer,), x, 4, 4, True)
            assert _bit_equal(got, reference)

    with ThreadPoolExecutor(max_workers=2) as engines:
        list(engines.map(call, layers, expected))
    assert [launches[n][-1] for n in widths] == [True, False]
    assert sorted(row_exact_qmv._UNROLLED_VERDICTS.values()) == [False, True]


class _Reclassed(nn.QuantizedLinear):
    """The prefill reclass makes projections a subclass of MLX's layer."""


def test_a_subclass_of_mlx_s_layer_takes_the_tile():
    linear = _linear(2560, 1024, 6, 64, seed=3)
    reclassed = _Reclassed(2560, 1024, bias=False, group_size=64, bits=6)
    reclassed.update(linear.parameters())
    x = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    assert row_exact_qmv._plan(reclassed, x, 4) is not None
    assert _bit_equal(row_exact_qmv.quantized_linear(reclassed, x), _serial(linear, x))


@pytest.mark.parametrize("rows", [2, 5, 16])
@pytest.mark.parametrize("n", [40, 72])
@testing.verified("qwen4_exp")
def test_a_format_layer_takes_the_tile_over_its_dot_face(monkeypatch, rows, n):
    """Any layer with a dot face exact for x takes the tile: the toy format's,
    each row its one-row call's (integers: exact in float32)."""
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("rows")))
    layer, _, _ = _toy_linear(n, 320, seed=rows)
    mx.random.seed(rows)
    x = mx.random.randint(-8, 9, (1, rows, 320)).astype(mx.float32)
    assert row_exact_qmv._plan(layer, x, rows) is not None
    got = row_exact_qmv.quantized_linear(layer, x)
    want = mx.concatenate([layer(x[:, r : r + 1]) for r in range(rows)], axis=1)
    assert mx.array_equal(got, want).item()


def test_a_layer_whose_face_is_not_exact_for_x_keeps_its_one_row_calls():
    """A plan is per dtype: the layer's bfloat16 rows take the tile, its
    float16 rows (scales of bfloat16) their one-row calls."""
    linear = _linear(2560, 1024, 4, 64, seed=4)
    x = mx.random.normal((1, 3, 2560))
    row_exact_qmv.quantized_linear(linear, x.astype(mx.bfloat16))
    assert row_exact_qmv._plan(linear, x.astype(mx.bfloat16), 3) is not None
    x = x.astype(mx.float16)
    assert row_exact_qmv._plan(linear, x, 3) is None
    assert _bit_equal(row_exact_qmv.quantized_linear(linear, x), _serial(linear, x))


def test_rows_of_another_k_are_the_layer_s_to_refuse():
    linear = _linear(2560, 1024, 4, 64, seed=6)
    row_exact_qmv.quantized_linear(linear, mx.zeros((1, 3, 2560), mx.bfloat16))
    with pytest.raises(ValueError):
        mx.eval(
            row_exact_qmv.quantized_linear(linear, mx.zeros((1, 3, 1280), mx.bfloat16))
        )


def test_more_than_max_rows_run_their_one_row_calls():
    linear = _linear(2560, 1024, 4, 64, seed=7)
    rows = row_exact_qmv.MAX_ROWS + 1
    x = mx.random.normal((1, rows, 2560)).astype(mx.bfloat16)
    assert row_exact_qmv._plan(linear, x, rows) is None
    assert _bit_equal(row_exact_qmv.quantized_linear(linear, x), _serial(linear, x))


def test_wide_outputs_share_each_weight_pass_between_rows():
    """From 2048 tiles (the vocabulary head) a threadgroup applies its tile
    to several rows: a divisor of the row count within the budget."""
    linear = _linear(2560, 16384, 8, 64, seed=8)
    x = mx.zeros((1, 4, 2560), mx.bfloat16)
    plan = row_exact_qmv._plan(linear, x, 4)
    assert plan.grid == (32, 2, 16384 // 8) and plan.padding == 0


@pytest.mark.parametrize("bits, tile", [(6, 3), (8, 4)])
def test_a_large_affine_head_pads_rows_no_divisor_groups(monkeypatch, bits, tile):
    threshold = 2560 * 16384 * bits // 8
    monkeypatch.setattr(row_exact_qmv, "_PADDED_ROWS_MIN_WEIGHT_BYTES", threshold)
    linear = _linear(2560, 16384, bits, 64, seed=9)
    x = mx.random.normal((1, 5, 2560)).astype(mx.bfloat16)
    plan = row_exact_qmv._plan(linear, x, 5)
    assert plan.grid == (32, 4, 16384 // 8)
    assert plan.output_shapes == [(2 * tile, 16384)] and plan.padding == 2 * tile - 5
    assert _bit_equal(row_exact_qmv.quantized_linear(linear, x), _serial(linear, x))
    four = x[:, :4]
    divisible = row_exact_qmv._plan(linear, four, 4)
    assert divisible.grid == (32, 2, 16384 // 8) and divisible.padding == 0
    got = row_exact_qmv.quantized_linear(linear, four)
    assert _bit_equal(got, _serial(linear, four))
    small = _linear(2304, 16384, bits, 64, seed=9)
    x = mx.random.normal((1, 5, 2304)).astype(mx.bfloat16)
    assert row_exact_qmv._plan(small, x, 5).padding == 0


@pytest.mark.parametrize("k,bits,rows", [(512, 4, 5), (512, 5, 5), (576, 6, 11)])
def test_other_head_layouts_keep_divisor_tiles(monkeypatch, k, bits, rows):
    monkeypatch.setattr(row_exact_qmv, "_PADDED_ROWS_MIN_WEIGHT_BYTES", 0)
    linear = _linear(k, 16384, bits, 64, seed=9)
    x = mx.zeros((1, rows, k), mx.bfloat16)
    assert row_exact_qmv._plan(linear, x, rows).padding == 0


@testing.verified("qwen4_exp")
def test_a_format_layer_of_another_mode_keeps_affine_s_padding_rule_out(monkeypatch):
    monkeypatch.setattr(row_exact_qmv, "_PADDED_ROWS_MIN_WEIGHT_BYTES", 0)
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("rows")))
    layer, _, _ = _toy_linear(16384, 256)
    x = mx.zeros((1, 5, 256), mx.float32)
    assert layer.bits == 8 and row_exact_qmv._plan(layer, x, 5).padding == 0


@pytest.mark.parametrize("bits", [4, 8])
def test_rows_qmv_takes_float32_tensors(bits):
    """GLM's float32 scales and activations, by position."""
    k, n, rows = 2048, 256, 4
    mx.random.seed(bits)
    weight, scales, biases = mx.quantize(
        mx.random.normal((n, k)) * 0.05, group_size=128, bits=bits
    )
    x = mx.random.normal((rows, k))
    launch = row_exact_qmv.rows_qmv(
        weight, scales, biases, bits, 128, "affine", mx.float32, lambda _: (1, 2)
    )
    want = mx.concatenate(
        [
            mx.quantized_matmul(
                x[r : r + 1],
                weight,
                scales,
                biases,
                transpose=True,
                group_size=128,
                bits=bits,
            )
            for r in range(rows)
        ]
    )
    assert mx.array_equal(launch(x), want).item()


@pytest.mark.parametrize("k, bits", [(64, 4), (128, 8)])
def test_a_matrix_of_mlx_s_quad_kernel_keeps_its_one_row_calls(k, bits):
    """MLX's one-row call of a 2-D matrix of K 64 or 128 at 4 or 8 bits is
    qmv_quad, which no dot face replays: the rows run that call."""
    linear = _linear(k, 256, bits, 64, seed=k)
    x = mx.random.normal((1, 3, k)).astype(mx.bfloat16)
    assert row_exact_qmv._plan(linear, x, 3) is None
    assert _bit_equal(row_exact_qmv.quantized_linear(linear, x), _serial(linear, x))


def test_shared_rows_run_the_unrolled_tile_this_gpu_computes_right():
    linear = _linear(2560, 16384, 8, 64, seed=10)
    x = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    (got,) = row_exact_qmv.quantized_linears_tiled((linear,), x, 4, 4, True)
    assert _bit_equal(got, _serial(linear, x))
    assert row_exact_qmv._group_plan((linear,), x, 4, (4, 4, True)).shape is not None
    assert list(row_exact_qmv._UNROLLED_VERDICTS.values()) == [True]
    # Default quantized_linear keeps eae63b6c's rolled policy.
    assert row_exact_qmv._plan(linear, x, 4).shape is None
    # Past the envelope (8 rows of 4 values on qmv's traversal: 32
    # accumulators): rolled even when unrolled=True is asked.
    wide = _linear(2624, 16384, 8, 64, seed=10)
    x = mx.random.normal((1, 8, 2624)).astype(mx.bfloat16)
    plan = row_exact_qmv._group_plan((wide,), x, 8, (8, 4, True))
    assert plan.grid[1] == 2 and plan.shape is None
    (got_wide,) = row_exact_qmv.quantized_linears_tiled((wide,), x, 8, 4, True)
    assert _bit_equal(got_wide, _serial(wide, x))


def test_shared_rows_fall_back_when_this_gpu_gets_the_unrolled_tile_wrong(monkeypatch):
    linear = _linear(2560, 16384, 8, 64, seed=11)
    x = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    monkeypatch.setattr(
        row_exact_skeleton,
        "compile_kernel",
        _miscompiled(row_exact_skeleton.compile_kernel),
    )
    (got,) = row_exact_qmv.quantized_linears_tiled((linear,), x, 4, 4, True)
    assert _bit_equal(got, _serial(linear, x))
    assert row_exact_qmv._group_plan((linear,), x, 4, (4, 4, True)).shape is None
    assert list(row_exact_qmv._UNROLLED_VERDICTS.values()) == [False]


@pytest.mark.parametrize("k,bits", [(2560, 8), (5120, 6)], ids=["flash-next", "27b"])
def test_vocabulary_head_verify_windows_equal_serial(k, bits):
    # Repeat packed rows to build the full head without a dense vocabulary matrix.
    n = 248320
    head = _linear(k, 32, bits, 64, seed=bits)
    for name in ("weight", "scales", "biases"):
        head[name] = mx.tile(head[name], (n // 32, 1))
    mx.eval(head.parameters())

    x = mx.random.normal((1, 16, k), key=mx.random.key(k)).astype(mx.bfloat16)
    serial = _serial(head, x)
    mx.eval(serial)
    for rows in range(2, 17):
        got = row_exact_qmv.quantized_linear(head, x[:, :rows])
        assert _bit_equal(got, serial[:, :rows]), rows


@testing.verified("qwen4_exp")
def test_format_layers_take_one_grouped_launch(monkeypatch):
    """Projections of one input with faces of one vpt run one launch: toy
    format layers, each output its one-row calls'."""
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("rows")))
    layers = [_toy_linear(n, 320, seed=n)[0] for n in (40, 72, 16)]
    mx.random.seed(1)
    x = mx.random.randint(-8, 9, (1, 3, 320)).astype(mx.float32)
    assert row_exact_qmv._group_plan(layers, x, 3) is not None
    for layer, got in zip(layers, row_exact_qmv.quantized_linears(layers, x)):
        want = mx.concatenate([layer(x[:, r : r + 1]) for r in range(3)], axis=1)
        assert mx.array_equal(got, want).item()


def _row_table(count, gain):
    face = _toy(f"row_table_{count}")
    table = formats.Table("gain", mx.float32, (0.0,) * (count - 1) + (gain,))
    source = face.source.replace(
        "const device float* scales;",
        f"const device float* scales; omlx_table<float, {count}> gain;",
    ).replace("const device float* s;", "const device float* s; float gain;")
    source = source.replace(
        "s(m.scales + row) {}", f"s(m.scales + row), gain(m.gain[{count - 1}]) {{}}"
    ).replace("sum * s[r];", "sum * s[r] * gain;")
    return replace(face, source=source, tables=(table,))


@pytest.mark.parametrize(
    "tables,grouped",
    [
        (((512, 2), (512, 4)), True),
        (((512, 2), (513, 4)), False),
        (((1024, 2), (1024, 2)), True),
        (((256, 2), (256, 4), (256, 6), (257, 8)), False),
    ],
)
@testing.verified("qwen4_exp")
def test_a_group_must_fit_its_distinct_tables(monkeypatch, tables, grouped):
    # At 4096 bytes both projections can share a launch, including when
    # they share the same 4096-byte table. At 4100 each runs separately.
    faces = {
        16 + 8 * i: _row_table(count, gain) for i, (count, gain) in enumerate(tables)
    }
    page = _ToyFormat()
    monkeypatch.setattr(page, "dot", lambda layer, k, n: faces[n])
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", page)
    layers = [_toy_linear(n, 320, seed=n)[0] for n in faces]
    x = mx.random.randint(-8, 9, (1, 4, 320), key=mx.random.key(9)).astype(mx.float32)
    tiled = row_exact_qmv.quantized_linears_tiled(layers, x, 2, 2)
    assert (tiled is not None) == grouped
    assert (row_exact_qmv._group_plan(layers, x, 4) is not None) == grouped
    outputs = row_exact_qmv.quantized_linears(layers, x)
    for i, layer in enumerate(layers):
        _, factor = tables[i]
        weight = layer["codes"].astype(mx.float32) - 128
        want = (x @ weight.T) * layer["group_scales"] * factor
        assert mx.array_equal(outputs[i], want).item()
        if grouped:
            assert mx.array_equal(tiled[i], want).item()


def test_a_group_of_two_vpts_or_a_bias_runs_each_projection_alone():
    x = mx.random.normal((1, 3, 2560)).astype(mx.bfloat16)
    mixed = [_linear(2560, 512, 4, 64, 1), _linear(2560, 512, 8, 64, 2)]
    assert row_exact_qmv._group_plan(mixed, x, 3) is None
    biased = [_linear(2560, 512, 4, 64, 3), _linear(2560, 512, 4, 64, 4)]
    biased[1]["bias"] = mx.ones((512,), mx.bfloat16)
    assert row_exact_qmv._group_plan(biased, x, 3) is None
    for group in (mixed, biased):
        for linear, got in zip(group, row_exact_qmv.quantized_linears(group, x)):
            assert _bit_equal(got, _serial(linear, x))


def test_a_group_s_geometry_is_its_widest_member_s():
    """A group's launch is sized by its widest output: a narrow first member
    does not keep a wide second one from sharing weight passes."""
    linears = [_linear(2560, 512, 8, 64, 12), _linear(2560, 16384, 8, 64, 13)]
    x = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    plan = row_exact_qmv._group_plan(linears, x, 4)
    assert plan.grid[1] == 2
    for linear, got in zip(linears, row_exact_qmv.quantized_linears(linears, x)):
        assert _bit_equal(got, _serial(linear, x))


def test_an_explicit_tile_is_planned_per_geometry_and_must_divide_the_rows():
    linears = [_linear(2560, 512, 4, 64, 14), _linear(2560, 256, 4, 64, 15)]
    x = mx.random.normal((1, 4, 2560)).astype(mx.bfloat16)
    tiled = row_exact_qmv.quantized_linears_tiled
    assert tiled(linears, x[:, :3], 2, 4, True) is None
    for geometry in ((2, 4, True), (4, 2, False), (1, 4, False)):
        assert row_exact_qmv._group_plan(linears, x, 4, geometry).grid[1] == 2 * (
            4 // geometry[0]
        )
        for linear, got in zip(linears, tiled(linears, x, *geometry)):
            assert _bit_equal(got, _serial(linear, x))


def test_a_group_of_more_than_max_rows_runs_each_projection_alone():
    linears = [_linear(2560, 512, 4, 64, 16), _linear(2560, 256, 4, 64, 17)]
    rows = row_exact_qmv.MAX_ROWS + 1
    x = mx.random.normal((1, rows, 2560)).astype(mx.bfloat16)
    assert row_exact_qmv._group_plan(linears, x, rows) is None
    for linear, got in zip(linears, row_exact_qmv.quantized_linears(linears, x)):
        assert _bit_equal(got, _serial(linear, x))


@pytest.mark.parametrize("group", [False, True], ids=["linear", "group"])
@testing.verified("qwen4_exp")
def test_format_verify_rejects_an_input_of_another_width(monkeypatch, group):
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("verify_width")))
    wide, _, _ = _toy_linear(40, 320)
    narrow, _, _ = _toy_linear(72, 256)
    x = mx.zeros((1, 3, 256), mx.float32)

    def reject_launch(*args, **kwargs):
        pytest.fail("planned a tile over a projection of another input width")

    monkeypatch.setattr(row_exact_qmv, "_module_launch", reject_launch)
    if group:
        assert row_exact_qmv.quantized_linears_tiled((narrow, wide), x, 1, 4) is None
    else:
        with pytest.raises(ValueError, match="320"):
            row_exact_qmv.quantized_linear(wide, x)


@testing.verified("qwen4_exp")
def test_format_verify_rechecks_a_warm_group_after_a_member_gains_bias(monkeypatch):
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("verify_bias")))
    layers = [_toy_linear(n, 320, seed=n)[0] for n in (40, 72)]
    x = mx.zeros((1, 3, 320), mx.float32)
    assert row_exact_qmv._group_plan(layers, x, 3) is not None
    layers[1]["bias"] = mx.arange(72, dtype=mx.float32)
    assert row_exact_qmv._group_plan(layers, x, 3) is None
    out = row_exact_qmv.quantized_linears(layers, x)
    assert mx.array_equal(out[1], mx.broadcast_to(layers[1].bias, (1, 3, 72))).item()


@testing.verified("qwen4_exp")
def test_format_verify_rechecks_the_face_when_a_loaded_tensor_changes(monkeypatch):
    face = _toy("verify_codes")

    class LoadedWeights(_ToyFormat):
        def dot(self, layer, k, n):
            return face if layer["codes"] is original else None

    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", LoadedWeights())
    layer, _, _ = _toy_linear(40, 320)
    original = layer["codes"]
    x = mx.ones((1, 3, 320), mx.float32)
    assert row_exact_qmv._plan(layer, x, 3) is not None
    replacement = mx.full(original.shape, 130, original.dtype)
    layer.load_weights([("codes", replacement)], strict=False)
    assert layer["codes"] is replacement
    assert row_exact_qmv._plan(layer, x, 3) is None
    assert mx.array_equal(row_exact_qmv.quantized_linear(layer, x), layer(x)).item()


@pytest.mark.parametrize("bits", [6, 8])
def test_rows_qmv_declines_constant_bound_scales(bits):
    """MLX binds a tensor of fewer than 8 elements as a constant; the tile
    reads scales and biases from device memory, so the layout is declined at
    plan time and the layer keeps its composed path."""
    weight, scales, biases = mx.quantize(
        mx.zeros((2, 256), dtype=mx.bfloat16), group_size=128, bits=bits
    )
    assert scales.size == 4
    assert (
        row_exact_qmv.rows_qmv(
            weight, scales, biases, bits, 128, "affine", mx.bfloat16, lambda _: (1, 1)
        )
        is None
    )


@testing.verified("qwen4_exp")
def test_format_layer_with_k_below_eight_falls_back(monkeypatch):
    """MLX binds an x of fewer than 8 elements as a constant, and the shared
    tile reads x from device memory: a format layer of k below 8 falls back."""
    for vpt, k, rows in ((4, 4, 1), (2, 2, 2)):
        face = _toy(f"tiny_k{vpt}", vpt=vpt)
        formats.load()
        monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(face))
        layer, _, _ = _toy_linear(16, k)
        x = mx.ones((1, rows, k), mx.float32)
        assert row_exact_qmv.quantized_linears_tiled((layer,), x, 1, 4) is None
        got = row_exact_qmv.quantized_linear(layer, x)
        assert mx.array_equal(got, layer(x)).item()


def test_group_plan_does_not_collide_when_a_member_is_replaced_at_same_id(monkeypatch):
    """A group's plan is keyed by its members' identity, not their id(): a
    replacement member reported at the same id gets its own plan."""
    first = _linear(320, 40, 8, 64, 1)
    second = _linear(320, 72, 8, 64, 2)
    x = mx.ones((1, 3, 320), mx.bfloat16)
    assert row_exact_qmv._group_plan((first, second), x, 3) is not None
    other = _linear(320, 16, 8, 64, 3)
    real_id = id
    monkeypatch.setattr(
        row_exact_qmv,
        "id",
        lambda value: real_id(second) if value is other else real_id(value),
        raising=False,
    )
    plan = row_exact_qmv._group_plan((first, other), x, 3)
    assert plan is not None and plan.output_shapes[1][-1] == 16


@testing.verified("qwen4_exp")
def test_warm_group_launch_avoids_id_audit_events(monkeypatch):
    """A warm grouped launch resolves its plan without id() calls: once an
    audit hook is installed (mlx-vlm's), every id() raises an event."""
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("audit_id")))
    cases = (
        ([_linear(320, n, 8, 64, n) for n in (40, 72, 16)], mx.bfloat16),
        ([_toy_linear(n, 320, seed=n)[0] for n in (40, 72, 16)], mx.float32),
    )
    events = []
    active = []

    def hook(event, args):
        if event == "builtins.id" and active:
            events.append(event)

    sys.addaudithook(hook)
    for layers, dtype in cases:
        x = mx.ones((1, 3, 320), dtype)
        assert row_exact_qmv._group_plan(layers, x, 3) is not None
        row_exact_qmv.quantized_linears(layers, x)
        events.clear()
        active.append(True)
        try:
            row_exact_qmv.quantized_linears(layers, x)
        finally:
            active.clear()
        assert events == []


def test_tile_store_loop_keeps_compile_time_bounds():
    """The skeleton's fused simd_sum+store loop keeps compile-time RPS bounds
    and guards partial tiles with constexpr PARTIAL."""
    row_tpl = row_exact_skeleton._ROW.template
    assert "for (int r = 0; r < RPS; r++)" in row_tpl
    assert "constexpr bool PARTIAL = (N_OUT % (2 * RPS)) != 0;" in row_tpl
    assert "if (PARTIAL && r >= valid)" in row_tpl


@testing.verified("qwen4_exp")
def test_group_plan_rechecks_when_sibling_mutates_during_build(monkeypatch):
    """The sibling tokens are snapshotted before the launch is built: a
    mutation that lands during the build is not cached with the old plan."""
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("sibling_mut")))
    first, _, _ = _toy_linear(40, 320, seed=1)
    second, _, _ = _toy_linear(72, 320, seed=2)
    x = mx.ones((1, 3, 320), mx.float32)
    real = row_exact_qmv._module_launch

    def mutate(*args, **kwargs):
        second["bias"] = mx.ones((72,), mx.float32)
        return real(*args, **kwargs)

    monkeypatch.setattr(row_exact_qmv, "_module_launch", mutate)
    assert row_exact_qmv._group_plan((first, second), x, 3) is not None
    monkeypatch.setattr(row_exact_qmv, "_module_launch", real)
    assert row_exact_qmv._group_plan((first, second), x, 3) is None


@testing.verified("qwen4_exp")
def test_group_plan_does_not_cache_refusal_after_bias_removed_during_build(
    monkeypatch,
):
    """A refusal is snapshotted before the build: a bias removed while the
    launch is being declined is not recorded as still present."""
    first = _linear(320, 40, 8, 64, 1)
    second = _linear(320, 72, 8, 64, 2)
    second["bias"] = None
    x = mx.ones((1, 3, 320), mx.bfloat16)
    real = row_exact_qmv.weakref.ref
    removed = []

    def remove_bias_once(obj, *args, **kwargs):
        if obj is second and not removed:
            removed.append(True)
            del second["bias"]
        return real(obj, *args, **kwargs)

    monkeypatch.setattr(row_exact_qmv.weakref, "ref", remove_bias_once)
    assert row_exact_qmv._group_plan((first, second), x, 3) is None
    monkeypatch.setattr(row_exact_qmv.weakref, "ref", real)
    assert row_exact_qmv._group_plan((first, second), x, 3) is not None


def test_out_of_range_rows_do_not_populate_plan_cache():
    """Rows outside 1..MAX_ROWS are every call's one-row fallback; they cache
    no refusal (and no tensor references) on the module."""
    layer = _linear(256, 32, 8, 64, 1)
    row_exact_qmv.quantized_linear(
        layer, mx.zeros((1, row_exact_qmv.MAX_ROWS + 5, 256), mx.bfloat16)
    )
    assert "_omlx_row_exact_plans" not in layer.__dict__
    row_exact_qmv.quantized_linears_tiled(
        (layer,), mx.zeros((1, 0, 256), mx.bfloat16), 2, 4
    )
    assert "_omlx_row_exact_plans" not in layer.__dict__


def test_quantized_linears_tiled_returns_none_for_zero_rows_or_invalid_tile():
    """Zero rows, empty projection lists, and non-positive tile arguments
    return None rather than failing inside concatenation or grid dispatch."""
    linears = [_linear(2560, 256, 4, 64, 1)]
    tiled = row_exact_qmv.quantized_linears_tiled
    assert tiled(linears, mx.zeros((1, 0, 2560), mx.bfloat16), 2, 4, True) is None
    assert tiled(linears, mx.zeros((1, 0, 2560), mx.bfloat16), 2, 4, False) is None
    assert tiled((), mx.zeros((1, 2, 2560), mx.bfloat16), 1, 4) is None
    assert tiled(linears, mx.zeros((1, 2, 2560), mx.bfloat16), 0, 4) is None
    assert tiled(linears, mx.zeros((1, 2, 2560), mx.bfloat16), 1, 0) is None


def test_rows_qmv_rejects_empty_rows_or_non_positive_geometry():
    """RowsQmv refuses an empty input and a non-positive tile with ValueError
    from the plan build, never AttributeError on a missing plan."""
    weight, scales, biases = _quantized(256, 32, 8, 64, mx.bfloat16, seed=13)
    empty = mx.zeros((1, 0, 256), dtype=mx.bfloat16)
    rows = mx.zeros((1, 2, 256), dtype=mx.bfloat16)
    qmv = row_exact_qmv.rows_qmv(
        weight, scales, biases, 8, 64, "affine", mx.bfloat16, lambda _: (4, 1)
    )
    with pytest.raises(ValueError, match="tile"):
        qmv(empty)
    for geometry in (lambda _: (4, -1), lambda _: (-1, 1)):
        qmv = row_exact_qmv.rows_qmv(
            weight, scales, biases, 8, 64, "affine", mx.bfloat16, geometry
        )
        with pytest.raises(ValueError, match="tile"):
            qmv(rows)


@testing.verified("qwen4_exp")
def test_concurrent_group_plan_misses_and_invalidation(monkeypatch):
    """Concurrent cold misses do not duplicate plan entries, and concurrent
    invalidation of a mutated FormatLinear sibling never raises IndexError."""
    first = _linear(320, 40, 8, 64, 1)
    second = _linear(320, 72, 8, 64, 2)
    x = mx.ones((1, 3, 320), mx.bfloat16)
    cold = Barrier(2, timeout=10)
    real_launch = row_exact_qmv._module_launch

    def slow_launch(*args, **kwargs):
        cold.wait()
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(row_exact_qmv, "_module_launch", slow_launch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        plans = list(
            pool.map(
                lambda _: row_exact_qmv._group_plan((first, second), x, 3), range(2)
            )
        )
    monkeypatch.setattr(row_exact_qmv, "_module_launch", real_launch)
    assert plans[0] is not None and plans[0] is plans[1]
    assert len(row_exact_qmv._plans(first)[(2, 3, mx.bfloat16, 320, None)]) == 1

    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("concurrent_inv")))
    f0, _, _ = _toy_linear(40, 320, seed=1)
    f1, _, _ = _toy_linear(72, 320, seed=2)
    xf = mx.ones((1, 3, 320), mx.float32)
    assert row_exact_qmv._group_plan((f0, f1), xf, 3) is not None
    f1["bias"] = mx.ones((72,), mx.float32)
    invalidate = Barrier(2, timeout=10)
    real_face = row_exact_qmv._face

    def slow_face(linear, value):
        if linear is f0:
            invalidate.wait()
        return real_face(linear, value)

    monkeypatch.setattr(row_exact_qmv, "_face", slow_face)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rebuilt = list(
            pool.map(lambda _: row_exact_qmv._group_plan((f0, f1), xf, 3), range(2))
        )
    assert rebuilt == [None, None]
    assert len(row_exact_qmv._plans(f0)[(2, 3, mx.float32, 320, None)]) == 1


def test_warm_group_plan_does_not_allocate_weakrefs(monkeypatch):
    """A warm _group_plan lookup reads immutable entries without allocating
    new weakref objects."""
    layers = [_linear(320, n, 8, 64, n) for n in (40, 72)]
    x = mx.ones((1, 3, 320), mx.bfloat16)
    warm = row_exact_qmv._group_plan(layers, x, 3)
    assert warm is not None

    def fail_ref(*args, **kwargs):
        pytest.fail("warm _group_plan allocated a weakref")

    monkeypatch.setattr(row_exact_qmv.weakref, "ref", fail_ref)
    assert row_exact_qmv._group_plan(layers, x, 3) is warm


@testing.verified("qwen4_exp")
def test_warm_format_group_hit_checks_stored_tokens_without_isinstance(monkeypatch):
    """Sibling format tokens are captured with their indices at construction:
    a warm hit compares the stored tokens, never re-slicing the members or
    asking isinstance(sibling, FormatLinear)."""
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("warm_fmt_tokens")))
    first, _, _ = _toy_linear(40, 320, seed=1)
    second, _, _ = _toy_linear(72, 320, seed=2)
    x = mx.ones((1, 3, 320), mx.float32)
    warm = row_exact_qmv._group_plan((first, second), x, 3)
    assert warm is not None

    real_format_linear = row_exact_qmv.FormatLinear

    class _TrapMeta(type):
        def __instancecheck__(cls, obj):
            if obj is second:
                pytest.fail("warm group hit ran isinstance(sibling, FormatLinear)")
            return isinstance(obj, real_format_linear)

    class _Trap(metaclass=_TrapMeta):
        pass

    monkeypatch.setattr(row_exact_qmv, "FormatLinear", _Trap)
    assert row_exact_qmv._group_plan((first, second), x, 3) is warm


def test_warm_quantized_linear_reuses_cached_packed_name(monkeypatch):
    """A warm quantized_linear call reads launch.names[0] instead of
    re-resolving formats.tensor_names."""
    linear = _linear(320, 40, 8, 64, 1)
    x = mx.ones((1, 3, 320), mx.bfloat16)
    want = _serial(linear, x)
    assert _bit_equal(row_exact_qmv.quantized_linear(linear, x), want)

    def fail_names(*args, **kwargs):
        pytest.fail("warm quantized_linear called formats.tensor_names")

    monkeypatch.setattr(row_exact_qmv.formats, "tensor_names", fail_names)
    assert _bit_equal(row_exact_qmv.quantized_linear(linear, x), want)


def test_group_plan_prunes_dead_member_entries():
    """Collected group members are pruned from the first layer's plan table."""
    first = _linear(320, 40, 8, 64, 1)
    ephemeral = _linear(320, 72, 8, 64, 2)
    x = mx.ones((1, 3, 320), mx.bfloat16)
    key = (2, 3, mx.bfloat16, 320, None)
    assert row_exact_qmv._group_plan((first, ephemeral), x, 3) is not None
    del ephemeral
    gc.collect()
    other = _linear(320, 16, 8, 64, 3)
    assert row_exact_qmv._group_plan((first, other), x, 3) is not None
    assert len(row_exact_qmv._plans(first)[key]) == 1


@testing.verified("qwen4_exp")
def test_format_face_declines_zero_vpt_and_indivisible_k(monkeypatch):
    """_face refuses faces with vpt < 1 or k not divisible by face.vpt."""
    formats.load()
    monkeypatch.setitem(
        formats._FORMATS, "toy", _ToyFormat(replace(_toy("zero_vpt"), vpt=0))
    )
    layer0, _, _ = _toy_linear(16, 32, seed=1)
    x0 = mx.ones((1, 2, 32), mx.float32)
    assert row_exact_qmv._face(layer0, x0) is None
    assert mx.array_equal(row_exact_qmv.quantized_linear(layer0, x0), layer0(x0)).item()

    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("vpt8", vpt=8)))
    layer12, _, _ = _toy_linear(16, 12, seed=2)
    x12 = mx.ones((1, 2, 12), mx.float32)
    assert row_exact_qmv._face(layer12, x12) is None
    got12 = row_exact_qmv.quantized_linear(layer12, x12)
    assert mx.array_equal(got12, layer12(x12)).item()


def test_quantized_linears_falls_back_above_four_projections(monkeypatch):
    """Groups of more than four projections run each projection separately."""
    layers = [_linear(320, 16 + 8 * i, 8, 64, i + 1) for i in range(5)]
    x = mx.random.normal((1, 3, 320)).astype(mx.bfloat16)

    def fail_group(*args, **kwargs):
        pytest.fail("grouped more than four projections")

    monkeypatch.setattr(row_exact_qmv, "_group_plan", fail_group)
    outputs = row_exact_qmv.quantized_linears(layers, x)
    for linear, got in zip(layers, outputs):
        assert _bit_equal(got, _serial(linear, x))
