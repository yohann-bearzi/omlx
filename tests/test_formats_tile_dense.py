# SPDX-License-Identifier: Apache-2.0
"""Dense format prefill and row-exact verification over the same weight."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier

import mlx.core as mx
import pytest

from omlx.formats import testing, tile_dense
from omlx.formats.linear import FormatLinear
from omlx.patches import m5_gather_qmm_nax as nax
from omlx.patches import qwen35_verify_qmm as verify
from omlx.patches.format_load import loading
from tests.test_formats_linear import _composed, _pair, _x
from tests.test_formats_tile_prefill import _stack, _tile_toy, needs_nax
from tests.test_formats_tile_prefill import toy as toy
from tests.test_qwen35_moe_routed_decode import _same_bits


@pytest.fixture(autouse=True)
def by_hand():
    with getattr(testing, "verified", lambda *_: nullcontext())("qwen4_exp"):
        yield


def _one_expert(quantized, x):
    k, n = x.shape[-1], quantized.weight.shape[0]
    xrows = x.reshape(-1, 1, k)
    reference = nax._stock_gather_qmm()(
        xrows,
        quantized.weight[None],
        quantized.scales[None],
        quantized.biases[None],
        rhs_indices=mx.zeros((xrows.shape[0],), mx.uint32),
        transpose=True,
        sorted_indices=True,
        bits=quantized.bits,
        group_size=quantized.group_size,
    ).reshape(*x.shape[:-1], n)
    return reference + quantized.bias if "bias" in quantized else reference


def _no_decode(self, indices=None):
    pytest.fail("an eligible dense call decoded the complete weight")


@pytest.fixture
def qwen_layer(tmp_path):
    """A dense layer built in a Qwen load, whose row-exact arming is the
    layer's own (``_qwen_row_exact``), as the model's layers' is."""
    config = {"model_type": "qwen4_exp", "omlx_quant_formats": {"affine": {}}}
    (tmp_path / "config.json").write_text(json.dumps(config))

    def build(quantized, k, bits=4):
        names = ("weight", "scales", "biases")
        with loading(tmp_path):
            return FormatLinear(
                "affine",
                bits,
                32,
                k,
                {name: quantized[name] for name in names},
                bias=quantized.get("bias"),
            )

    return build


@needs_nax
@pytest.mark.parametrize("bits,dtype", [(4, mx.bfloat16), (8, mx.float16)])
@pytest.mark.parametrize("leading", [(17,), (2, 20)])
def test_dense_format_prefill_matches_one_mlx_expert(monkeypatch, bits, dtype, leading):
    k, n = 512, 128
    quantized, layer = _pair(k, n, bits=bits, dtype=dtype, bias=True)
    x = _x((*leading, 2 * k), dtype)[..., ::2]
    want = _one_expert(quantized, x)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for _ in range(2):
        got = layer(x)
        assert got.dtype == dtype and _same_bits(got, want)


@needs_nax
def test_compiled_dense_prefill_reads_the_current_toy_matrix(toy, monkeypatch):
    k, n = 256, 64
    fmt = toy(_tile_toy("dense"))
    stack = _stack(n, k, seed=23)
    weights = [{key: values[i] for key, values in stack.items()} for i in (0, 1)]
    layer = FormatLinear("tiletoy", 8, 1, k, weights[0])
    x = _x((33, k), mx.float16)
    ids = mx.zeros((x.shape[0],), mx.uint32)
    expected = []
    for operands in weights:
        w = fmt.decode(operands, 8, 1).astype(x.dtype)
        expected.append(
            mx.gather_mm(
                x[:, None, :], w.T[None], rhs_indices=ids, sorted_indices=True
            )[:, 0]
        )
    assert not _same_bits(expected[0], expected[1])
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)

    project = mx.compile(lambda x: layer(x), inputs=layer.state)
    for operands, want in zip(weights, expected, strict=True):
        layer["codes"], layer["group_scales"] = (
            operands["codes"],
            operands["group_scales"],
        )
        assert _same_bits(project(x), want)
        assert _same_bits(layer(x), want)


@needs_nax
def test_sixteen_rows_keep_the_dot_launch(monkeypatch):
    k, n = 512, 128
    quantized, layer = _pair(k, n, bits=4, bias=True)
    x = _x((16, k))
    serial = mx.concatenate([quantized(x[i : i + 1]) for i in range(16)])

    def never_launch(*args, **kwargs):
        pytest.fail("sixteen rows took the dense prefill")

    monkeypatch.setattr(tile_dense, "one_run", never_launch)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    assert _same_bits(layer(x), serial)


@needs_nax
@pytest.mark.parametrize("rows", [33, 40])
@pytest.mark.parametrize("armed_first", [True, False])
def test_the_same_dense_shape_selects_row_exact_or_prefill(
    monkeypatch, rows, armed_first, qwen_layer
):
    k, n = 512, 128
    quantized, _ = _pair(k, n, bits=4, bias=True)
    layer = qwen_layer(quantized, k, bits=4)
    x = _x((rows, k), seed=rows)
    prefill = _one_expert(quantized, x)
    serial = mx.concatenate([quantized(x[i : i + 1]) for i in range(rows)])
    assert not _same_bits(serial, prefill)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    try:
        for armed in (armed_first, not armed_first, armed_first):
            verify.set_verify_qmm_armed(armed, row_exact=armed)
            assert _same_bits(layer(x), serial if armed else prefill)
    finally:
        verify.set_verify_qmm_armed(False)


@needs_nax
def test_two_threads_can_prefill_and_verify_the_same_dense_layer(
    monkeypatch, qwen_layer
):
    k, n, rows = 512, 128, 40
    quantized, _ = _pair(k, n)
    layer = qwen_layer(quantized, k)
    x = _x((rows, k), seed=41)
    serial = mx.concatenate([quantized(x[i : i + 1]) for i in range(rows)])
    prefill = _one_expert(quantized, x)
    mx.eval(prefill, serial, *quantized.parameters().values())
    assert not _same_bits(serial, prefill)
    ready = Barrier(2)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)

    def in_thread(armed):
        verify.set_verify_qmm_armed(armed, row_exact=armed)
        try:
            ready.wait(timeout=30)
            return _same_bits(layer(x), serial if armed else prefill)
        finally:
            verify.set_verify_qmm_armed(False)

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert all(workers.map(in_thread, (True, False)))


@pytest.mark.parametrize(
    "n,dtype", [pytest.param(40, mx.bfloat16, marks=needs_nax), (128, mx.float32)]
)
def test_dense_prefill_declines_unsupported_columns_or_dtype(monkeypatch, n, dtype):
    k = 320
    quantized, layer = _pair(k, n, bias=True)
    x = _x((33, k), dtype)
    want = _composed(quantized, x)
    seen = []
    decoded = FormatLinear._decoded

    def observe(self):
        seen.append(self is layer)
        return decoded(self)

    def unexpected(*args):
        pytest.fail("an unsupported call tried to build the tile")

    monkeypatch.setattr(nax, "_face_kernel", unexpected)
    monkeypatch.setattr(FormatLinear, "_decoded", observe)
    assert mx.array_equal(layer(x), want).item()
    assert seen == [True]


@needs_nax
def test_dense_prefill_binds_a_tile_without_a_dot_face(toy, monkeypatch):
    k = 256
    fmt = toy(_tile_toy("tile_only"))
    fmt.dot_face = None
    x = _x((1, 33, k), mx.float16)
    ids = mx.zeros((33,), mx.uint32)
    for n in (32, 96, 32):
        weights = {key: a[0] for key, a in _stack(n, k, seed=n).items()}
        layer = FormatLinear("tiletoy", 8, 1, k, weights)
        w = fmt.decode(weights, 8, 1).astype(x.dtype)
        want = mx.gather_mm(
            x.reshape(33, 1, k), w.T[None], rhs_indices=ids, sorted_indices=True
        ).reshape(1, 33, n)
        monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
        assert _same_bits(layer(x), want)


@needs_nax
def test_failed_dense_tile_decodes_and_preserves_the_row_exact_route(
    toy, caplog, tmp_path
):
    from tests.test_formats_tile_prefill import _TILE

    k, n = 256, 64
    fmt = toy(_tile_toy("bad_dense", _TILE.replace("RUN = 8;", "RUN = 16;")))
    weights = {key: a[0] for key, a in _stack(n, k).items()}
    config = {"model_type": "qwen4_exp", "omlx_quant_formats": {"affine": {}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with loading(tmp_path):
        layer = FormatLinear("tiletoy", 8, 1, k, weights)
    x = _x((40, k), mx.float16)
    dense = x @ fmt.decode(weights, 8, 1).astype(x.dtype).T
    serial = mx.concatenate([layer(x[i : i + 1]) for i in range(40)])
    assert not _same_bits(dense, serial)
    try:
        for armed in (False, True, False):
            verify.set_verify_qmm_armed(armed, row_exact=armed)
            assert _same_bits(layer(x), serial if armed else dense)
    finally:
        verify.set_verify_qmm_armed(False)
    failures = [
        r for r in caplog.records if "tile kernel failed; decode fallback" in r.message
    ]
    assert len(failures) == 1


def test_the_row_exact_graph_chunks_sixteen_and_a_tail(monkeypatch, qwen_layer):
    from omlx.formats import linear

    k, n = 512, 128
    quantized, _ = _pair(k, n)
    layer = qwen_layer(quantized, k)
    x = _x((2, 20, k))
    flat = x.reshape(40, k)
    serial = mx.concatenate([quantized(flat[i : i + 1]) for i in range(40)])
    seen = []
    describe = linear._linear

    def describe_chunk(*args):
        # The last argument is the activation for this row launch.
        seen.append(args[-1].shape)
        return describe(*args)

    monkeypatch.setattr(linear, "_linear", describe_chunk)
    try:
        verify.set_verify_qmm_armed(True, row_exact=True)
        for _ in range(2):
            assert _same_bits(layer(x).reshape(40, n), serial)
        assert seen == [(16, k), (8, k)]
    finally:
        verify.set_verify_qmm_armed(False)


@needs_nax
@pytest.mark.parametrize("reason", ["k_tail", "no_tile", "inexact", "kernel"])
def test_dense_prefill_declines_unavailable_tiles(toy, monkeypatch, reason):
    from dataclasses import replace

    k, n = (264 if reason == "k_tail" else 256), 64
    face = _tile_toy("declined")
    if reason == "inexact":
        face = replace(face, dtypes=(mx.float16,))
    fmt = toy(None if reason == "no_tile" else face)
    tensors = {key: a[0] for key, a in _stack(n, k).items()}
    layer = FormatLinear("tiletoy", 8, 1, k, tensors)
    x = _x((33, k), mx.bfloat16)
    want = x @ fmt.decode(tensors, 8, 1).astype(x.dtype).T

    def never_launch(*args, **kwargs):
        pytest.fail("a declined dense tile reached the Metal launch")

    monkeypatch.setattr(tile_dense, "one_run", never_launch)
    if reason == "kernel":
        monkeypatch.setattr(nax, "_face_kernel", lambda *args: None)
    for _ in range(2):
        assert _same_bits(layer(x), want)


@needs_nax
def test_a_declined_dense_route_is_not_built_again(toy, monkeypatch):
    from omlx.formats import linear

    k, n = 256, 64
    fmt = toy(None)
    tensors = {key: a[0] for key, a in _stack(n, k).items()}
    layer = FormatLinear("tiletoy", 8, 1, k, tensors)
    x = _x((33, k), mx.float16)
    want = x @ fmt.decode(tensors, 8, 1).astype(x.dtype).T
    built = []
    build = linear._tile

    def record(held):
        built.append(held is layer)
        return build(held)

    monkeypatch.setattr(linear, "_tile", record)
    for _ in range(3):
        assert _same_bits(layer(x), want)
    assert built == [True]


@needs_nax
def test_dense_prefill_proves_the_calls_plan_on_sixteen_rows(toy, monkeypatch):
    k, n, rows = 256, 96, 33
    fmt = toy(_tile_toy("dense_canary"))
    weights = {key: a[0] for key, a in _stack(n, k).items()}
    layer = FormatLinear("tiletoy", 8, 1, k, weights)
    x = _x((3, 11, k), mx.float16)
    w = fmt.decode(weights, 8, 1).astype(x.dtype)
    ids = mx.zeros((rows,), mx.uint32)
    want = mx.gather_mm(
        x.reshape(rows, 1, k), w.T[None], rhs_indices=ids, sorted_indices=True
    ).reshape(3, 11, n)
    seen = []
    launch = tile_dense.one_run

    def record(kernel, config, z, operands):
        seen.append((z.shape, tuple(a.shape for a in operands), config))
        return launch(kernel, config, z, operands)

    monkeypatch.setattr(tile_dense, "one_run", record)
    for _ in range(2):
        assert _same_bits(layer(x), want)
    matrices = ((n, k), (n,))
    config = nax.face_plan(rows, 1, k, n)
    assert seen == [
        ((16, k), matrices, config),
        ((3, 11, k), matrices, config),
    ]


@needs_nax
@pytest.mark.parametrize("rows", [17, 193])
def test_dense_nax_one_run_skips_the_expert_scan(monkeypatch, rows):
    quantized, layer = _pair(512, 96, bits=4, bias=True)
    x = _x((rows, 512), seed=rows)
    want = _one_expert(quantized, x)
    mx.eval(want)
    scanned = []
    tiles = nax._tiles

    def record(scan, indices, experts, bm, kw):
        scanned.append((indices.size, experts))
        return tiles(scan, indices, experts, bm, kw)

    monkeypatch.setattr(nax, "_tiles", record)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for _ in range(2):
        assert _same_bits(layer(x), want)
    assert scanned == []


@needs_nax
def test_dense_nax_one_run_works_without_a_scan_kernel(monkeypatch):
    quantized, layer = _pair(512, 96)
    x = _x((33, 512))
    want = _one_expert(quantized, x)
    mx.eval(want)
    original = nax._get_kernel

    def without_scan(kind):
        return None if kind == "scan" else original(kind)

    monkeypatch.setattr(nax, "_get_kernel", without_scan)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    assert _same_bits(layer(x), want)


@needs_nax
@pytest.mark.parametrize("rows", [40, 193])
def test_dense_nax_one_run_covers_the_split_grid(monkeypatch, rows):
    k, n = 2560, 512
    quantized, layer = _pair(k, n, bits=4, bias=True)
    x = _x((rows, k), seed=rows)
    want = _one_expert(quantized, x)
    mx.eval(want)
    configs = []
    plan = nax.face_plan

    def record(count, experts, width, columns):
        configs.append(plan(count, experts, width, columns))
        return configs[-1]

    monkeypatch.setattr(nax, "face_plan", record)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for _ in range(2):
        assert _same_bits(layer(x), want)
    assert len(configs) == 1 and configs[0].gx > 0
