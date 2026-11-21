# SPDX-License-Identifier: Apache-2.0
"""Dense format prefill and row-exact verification over the same weight."""

import builtins
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
from tests.test_formats_tile_prefill import (
    _stack,
    _tile_toy,
    needs_nax,
    needs_steel,
)
from tests.test_formats_tile_prefill import toy as toy
from tests.test_qwen35_moe_routed_decode import _same_bits


@pytest.fixture(autouse=True)
def by_hand():
    with getattr(testing, "verified", lambda *_: nullcontext())("qwen4_exp"):
        yield


def _one_expert(quantized, x):
    """The one-expert sorted gather's bits; MLX's gather_qmm reads past the
    scales on a ragged K, so a K that is not whole 64-deep steps decodes in
    float32 and takes the bounds-checked gather_mm."""
    k, n = x.shape[-1], quantized.weight.shape[0]
    xrows = x.reshape(-1, 1, k)
    ids = mx.zeros((xrows.shape[0],), mx.uint32)
    if k % 64:
        weight = mx.dequantize(
            quantized.weight,
            quantized.scales.astype(mx.float32),
            quantized.biases.astype(mx.float32),
            group_size=quantized.group_size,
            bits=quantized.bits,
        ).astype(x.dtype)
        reference = mx.gather_mm(
            xrows, weight.T[None], rhs_indices=ids, sorted_indices=True
        ).reshape(*x.shape[:-1], n)
    else:
        reference = nax._stock_gather_qmm()(
            xrows,
            quantized.weight[None],
            quantized.scales[None],
            quantized.biases[None],
            rhs_indices=ids,
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


def test_sixteen_rows_without_a_dot_face_still_decode(toy, monkeypatch):
    k, n = 256, 64
    fmt = toy(_tile_toy("tile_only_16"))
    fmt.dot_face = None
    tensors = {key: a[0] for key, a in _stack(n, k).items()}
    layer = FormatLinear("tiletoy", 8, 1, k, tensors)
    x = _x((16, k), mx.float16)
    want = x @ fmt.decode(tensors, 8, 1).astype(x.dtype).T

    def never_launch(*args, **kwargs):
        pytest.fail("sixteen rows took the dense prefill")

    monkeypatch.setattr(tile_dense, "one_run", never_launch)
    assert _same_bits(layer(x), want)


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
@pytest.mark.parametrize("rows", [40, 193, 4224, 4225])
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
    if rows >= 4224:
        assert -(-rows // configs[0].bm) > configs[0].gx


@needs_nax
def test_dense_prefill_handles_ragged_k_and_proves_both_n_alignments(monkeypatch):
    k, rows = 160, 33
    tiles = []
    project = tile_dense.project

    def record(tile, z, **kwargs):
        tiles.append(tile)
        return project(tile, z, **kwargs)

    monkeypatch.setattr(tile_dense, "project", record)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for n in (64, 96):
        quantized, layer = _pair(k, n, bits=4)
        x = _x((rows, k), seed=n)
        assert _same_bits(layer(x), _one_expert(quantized, x))
    assert len(tiles) == 2 and tiles[0].proof is tiles[1].proof
    assert len(tiles[0].proof.ran) == 2


@needs_nax
def test_one_expert_ragged_k_oracle_does_not_read_past_scales(monkeypatch):
    quantized, layer = _pair(160, 64, bits=4)

    def trailed(array):
        nan = mx.full((64,), float("nan"), dtype=array.dtype)
        return mx.concatenate([array.reshape(-1), nan])[: array.size].reshape(
            array.shape
        )

    for name in ("scales", "biases"):
        quantized[name] = trailed(quantized[name])
        layer[name] = trailed(layer[name])
    x = _x((33, 160))
    reference = _one_expert(quantized, x)
    assert not mx.any(mx.isnan(reference)).item()
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    assert _same_bits(layer(x), reference)


def test_dense_prefill_does_not_import_its_module_per_call(monkeypatch):
    k, n = 512, 128
    _, layer = _pair(k, n, bits=4, bias=True)
    x = _x((33, k))
    mx.eval(layer(x))
    imports = []
    real = builtins.__import__

    def record(name, *args, **kwargs):
        fromlist = kwargs.get("fromlist", args[2] if len(args) > 2 else ())
        if "tile_dense" in name or (fromlist and "tile_dense" in fromlist):
            imports.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", record)
    mx.eval(layer(x))
    assert imports == []


@needs_steel
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("n", [40, 41])
def test_dense_steel_tile_covers_the_final_columns(toy, monkeypatch, dtype, n):
    from omlx.formats import tile_gather

    k = 256
    fmt = toy(_tile_toy("dense_steel_ragged"))
    weights = {key: a[0] for key, a in _stack(n, k, seed=31).items()}
    layer = FormatLinear("tiletoy", 8, 1, k, weights)
    x = _x((3, 17, k), dtype)
    ids = mx.zeros((51,), mx.uint32)
    want = mx.gather_mm(
        x.reshape(51, 1, k),
        fmt.decode(weights, 8, 1).astype(dtype).T[None],
        rhs_indices=ids,
        sorted_indices=True,
    ).reshape(3, 17, n)
    seen = []
    build = tile_gather.kernel

    def observed(face):
        kernel = build(face)

        def record(**kwargs):
            inputs = kwargs["inputs"]
            seen.append((inputs[0].shape, inputs[1].shape))
            return kernel(**kwargs)

        return record

    monkeypatch.setattr(tile_gather, "kernel", observed)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for _ in range(2):
        assert _same_bits(layer(x), want)
    assert seen == [((16, 1, k), (1, n, k)), ((51, 1, k), (1, n, k))]


@needs_steel
@pytest.mark.parametrize("reason", ["k_tail", "no_kernel", "inexact"])
def test_dense_steel_declines_unsupported_weights(toy, monkeypatch, reason):
    from dataclasses import replace

    from omlx.formats import tile_gather

    k, n = (264 if reason == "k_tail" else 256), 40
    face = _tile_toy("dense_steel_declined")
    if reason == "inexact":
        face = replace(face, dtypes=(mx.float16,))
    fmt = toy(face)
    weights = {key: a[0] for key, a in _stack(n, k).items()}
    layer = FormatLinear("tiletoy", 8, 1, k, weights)
    x = _x((17, k), mx.bfloat16)
    want = x @ fmt.decode(weights, 8, 1).astype(x.dtype).T

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported steel tile was launched")

    monkeypatch.setattr(
        tile_gather,
        "kernel",
        lambda face: None if reason == "no_kernel" else unexpected,
    )
    assert _same_bits(layer(x), want)


@needs_steel
def test_steel_dense_grid_limit_uses_the_number_of_rows(monkeypatch):
    from omlx.formats import tile_dense

    class DecodedError(Exception):
        pass

    def decode(self):
        raise DecodedError

    _, layer = _pair(256, 64)
    x = _x((1, 256))
    accepted = mx.zeros((1,))
    monkeypatch.setattr(tile_dense, "project", lambda *a, **kw: accepted)
    monkeypatch.setattr(FormatLinear, "_decoded", decode)
    # A broadcast view costs one row. Intercept both routes before they can allocate.
    assert layer(mx.broadcast_to(x, (2**30 - 1, 256))) is accepted
    with pytest.raises(DecodedError):
        layer(mx.broadcast_to(x, (2**30, 256)))


@needs_steel
def test_failed_dense_steel_proof_leaves_the_sorted_stack_available(toy, monkeypatch):
    from omlx.formats import tile_gather
    from omlx.formats.linear import FormatSwitchLinear
    from tests.test_formats_tile_prefill import _TILE, _rows, _want

    k, n = 256, 64
    fmt = toy(_tile_toy("separate_steel_proofs"))
    tensors = _stack(n, k, seed=43)
    weights = {key: a[0] for key, a in tensors.items()}
    dense = FormatLinear("tiletoy", 8, 1, k, weights)
    switch = FormatSwitchLinear("tiletoy", 8, 1, k, tensors)
    x = _x((33, k))
    xs, indices = _rows(k)
    real = tile_gather.kernel
    bad = real(_tile_toy("dense_bad_canary", _TILE.replace("RUN = 8;", "RUN = 16;")))
    experts = []

    def build(face):
        kernel = real(face)

        def launch(**kw):
            count = kw["inputs"][1].shape[0]
            experts.append(count)
            if len(experts) == 1:
                return bad(**kw)
            return kernel(**kw)

        return launch

    monkeypatch.setattr(tile_gather, "kernel", build)
    assert _same_bits(dense(x), x @ fmt.decode(weights, 8, 1).astype(x.dtype).T)
    assert experts == [1]
    count = switch.num_experts
    assert _same_bits(
        switch(xs, indices, sorted_indices=True), _want(tensors, xs, indices)
    )
    assert experts == [1, 1, count]


@needs_steel
def test_dense_steel_proves_both_n_alignments(toy, monkeypatch):
    k, rows = 256, 17
    fmt = toy(_tile_toy("steel_align"))
    tiles = []
    project = tile_dense.project

    def record(tile, z, **kwargs):
        tiles.append(tile)
        return project(tile, z, **kwargs)

    monkeypatch.setattr(tile_dense, "project", record)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for n in (32, 40):
        weights = {key: a[0] for key, a in _stack(n, k, seed=n).items()}
        layer = FormatLinear("tiletoy", 8, 1, k, weights)
        x = _x((rows, k), mx.float16)
        ids = mx.zeros((rows,), mx.uint32)
        want = mx.gather_mm(
            x.reshape(rows, 1, k),
            fmt.decode(weights, 8, 1).astype(x.dtype).T[None],
            rhs_indices=ids,
            sorted_indices=True,
        ).reshape(rows, n)
        assert _same_bits(layer(x), want)
    assert len(tiles) == 2 and tiles[0].proof is tiles[1].proof
    assert len(tiles[0].proof.ran) == 2


@needs_steel
@pytest.mark.parametrize("rows", [17, 193])
def test_dense_steel_one_run_needs_no_expert_offsets_scan(monkeypatch, rows):
    quantized, layer = _pair(512, 96, bits=4, bias=True)
    x = _x((rows, 1024), seed=rows)[..., ::2]
    want = _one_expert(quantized, x)
    mx.eval(want)
    scans = []
    cumsum = mx.cumsum

    def scanned(counts, *args, **kwargs):
        scans.append(counts.shape)
        return cumsum(counts, *args, **kwargs)

    monkeypatch.setattr(mx, "cumsum", scanned)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for _ in range(2):
        assert _same_bits(layer(x), want)
    assert scans == []
