# SPDX-License-Identifier: Apache-2.0
"""Dense format prefill and the disarmed verify routes over one weight."""

import builtins
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier

import mlx.core as mx
import mlx.nn as nn
import pytest
from tests.test_formats_tile_prefill import (
    _stack,
    _tile_toy,
    needs_nax,
    needs_steel,
)
from tests.test_formats_tile_prefill import toy as toy

from omlx.formats import testing, tile_dense
from omlx.formats.linear import FormatLinear
from omlx.patches import m5_gather_qmm_nax as nax
from omlx.patches.format_load import loading
from tests.test_formats_linear import _composed, _pair, _x
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
    """A dense layer built in a Qwen load, as the served model's layers are."""
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


@pytest.mark.parametrize(
    "bits,dtype",
    [(2, mx.bfloat16), (3, mx.bfloat16), (4, mx.bfloat16), (8, mx.float16)],
)
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


@pytest.mark.parametrize("rows", [33, 40])
def test_the_dense_shape_takes_the_prefill_route(monkeypatch, rows, qwen_layer):
    """Above MAX_ROWS a verify window takes the dense tile on every call: the
    deleted armed serial route left no second arithmetic (Rules 29/33)."""
    k, n = 512, 128
    quantized, _ = _pair(k, n, bits=4, bias=True)
    layer = qwen_layer(quantized, k, bits=4)
    x = _x((rows, k), seed=rows)
    prefill = _one_expert(quantized, x)
    serial = mx.concatenate([quantized(x[i : i + 1]) for i in range(rows)])
    assert not _same_bits(serial, prefill)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for _ in range(2):
        assert _same_bits(layer(x), prefill)


def test_two_threads_can_prefill_the_same_dense_layer(monkeypatch, qwen_layer):
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

    def in_thread(_unused):
        ready.wait(timeout=30)
        return _same_bits(layer(x), prefill)

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert all(workers.map(in_thread, (None, None)))


def test_dense_prefill_declines_unsupported_dtype(monkeypatch):
    k, n = 320, 128
    quantized, layer = _pair(k, n, bias=True)
    x = _x((33, k), mx.float32)
    want = _composed(quantized, x)
    seen = []
    decoded = FormatLinear._decoded

    def observe(self):
        seen.append(self is layer)
        return decoded(self)

    def unexpected(*args, **kwargs):
        pytest.fail("an unsupported call tried to build the tile")

    monkeypatch.setattr(tile_dense, "build_plan", unexpected)
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


def test_failed_dense_tile_decodes_and_keeps_the_fallback(toy, caplog, tmp_path):
    from tests.test_formats_tile_prefill import _TILE

    k, n = 256, 64
    fmt = toy(_tile_toy("bad_dense", _TILE.replace("RUN = 8;", "RUN = 16;")))
    weights = {key: a[0] for key, a in _stack(n, k).items()}
    config = {"model_type": "qwen4_exp", "omlx_quant_formats": {"tiletoy": {}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with loading(tmp_path):
        layer = FormatLinear("tiletoy", 8, 1, k, weights)
    x = _x((40, k), mx.float16)
    dense = x @ fmt.decode(weights, 8, 1).astype(x.dtype).T
    serial = mx.concatenate([layer(x[i : i + 1]) for i in range(40)])
    assert not _same_bits(dense, serial)
    for _ in range(3):
        assert _same_bits(layer(x), dense)
    failures = [
        r for r in caplog.records if "tile kernel failed; decode fallback" in r.message
    ]
    assert len(failures) == 1


def test_the_disarmed_dense_call_projects_the_whole_window(monkeypatch, qwen_layer):
    from omlx.formats import linear

    k, n = 512, 128
    quantized, _ = _pair(k, n)
    layer = qwen_layer(quantized, k)
    x = _x((2, 20, k))
    want = _one_expert(quantized, x)

    def never_launch(*args, **kwargs):
        pytest.fail("a 40-row disarmed call launched a row kernel")

    monkeypatch.setattr(linear, "_linear", never_launch)
    for _ in range(2):
        assert _same_bits(layer(x), want)


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

    monkeypatch.setattr(tile_dense, "_launch", never_launch)
    if reason == "kernel":
        monkeypatch.setattr(tile_dense, "_kernel", lambda *args: None)
    for _ in range(2):
        assert _same_bits(layer(x), want)


def test_a_declined_dense_route_is_not_built_again(toy, monkeypatch):
    k, n = 256, 64
    fmt = toy(None)
    tensors = {key: a[0] for key, a in _stack(n, k).items()}
    layer = FormatLinear("tiletoy", 8, 1, k, tensors)
    x = _x((33, k), mx.float16)
    want = x @ fmt.decode(tensors, 8, 1).astype(x.dtype).T
    built = []
    build = tile_dense.build_plan

    def record(held, **kwargs):
        built.append(held is layer)
        return build(held, **kwargs)

    monkeypatch.setattr(tile_dense, "build_plan", record)
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
    launch = tile_dense._launch

    def record(mm, policy, z, operands):
        seen.append((z.shape, tuple(a.shape for a in operands), policy))
        return launch(mm, policy, z, operands)

    monkeypatch.setattr(tile_dense, "_launch", record)
    for _ in range(2):
        assert _same_bits(layer(x), want)
    matrices = ((n, k), (n,))
    plan = tile_dense.build_plan(layer, tensor_units=True)
    assert plan is not None
    policy = tile_dense.policy_for(plan, rows, x.dtype)
    assert seen == [
        ((16, k), matrices, policy),
        ((3, 11, k), matrices, policy),
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
def test_dense_nax_direct_covers_the_split_grid(monkeypatch, rows):
    k, n = 2560, 512
    quantized, layer = _pair(k, n, bits=4, bias=True)
    x = _x((rows, k), seed=rows)
    want = _one_expert(quantized, x)
    mx.eval(want)
    policies = []
    launch = tile_dense._launch

    def record(mm, policy, z, operands):
        if z.shape[0] == rows:
            policies.append(policy)
        return launch(mm, policy, z, operands)

    monkeypatch.setattr(tile_dense, "_launch", record)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for _ in range(2):
        assert _same_bits(layer(x), want)
    assert len(policies) == 1 and policies[0].gx > 0
    if rows >= 4224:
        assert -(-rows // policies[0].bm) > policies[0].gx


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
    build = tile_dense._kernel

    def observed(face, policy):
        kernel = build(face, policy)

        def record(**kwargs):
            inputs = kwargs["inputs"]
            seen.append((inputs[0].shape, inputs[1].shape))
            return kernel(**kwargs)

        return record

    monkeypatch.setattr(tile_dense, "_kernel", observed)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for _ in range(2):
        assert _same_bits(layer(x), want)
    assert seen == [((16, k), (n, k)), ((3, 17, k), (n, k))]


@needs_steel
@pytest.mark.parametrize("reason", ["k_tail", "no_kernel", "inexact"])
def test_dense_steel_declines_unsupported_weights(toy, monkeypatch, reason):
    from dataclasses import replace

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
        tile_dense,
        "_kernel",
        lambda face, policy: None if reason == "no_kernel" else unexpected,
    )
    assert _same_bits(layer(x), want)


@needs_steel
def test_steel_dense_grid_limit_uses_the_number_of_rows(monkeypatch):
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
    from tests.test_formats_tile_prefill import _TILE, _rows, _want

    from omlx.formats import tile_gather
    from omlx.formats.linear import FormatSwitchLinear

    k, n = 256, 64
    fmt = toy(_tile_toy("separate_steel_proofs"))
    tensors = _stack(n, k, seed=43)
    weights = {key: a[0] for key, a in tensors.items()}
    dense = FormatLinear("tiletoy", 8, 1, k, weights)
    switch = FormatSwitchLinear("tiletoy", 8, 1, k, tensors)
    x = _x((33, k))
    xs, indices = _rows(k)
    bad = tile_dense._build(
        _tile_toy("dense_bad_canary", _TILE.replace("RUN = 8;", "RUN = 16;")), "steel"
    )
    monkeypatch.setattr(tile_dense, "_kernel", lambda face, policy: bad)
    assert _same_bits(dense(x), x @ fmt.decode(weights, 8, 1).astype(x.dtype).T)
    experts = []
    real_gather = tile_gather.kernel

    def observe_gather(face):
        kernel = real_gather(face)

        def launch(**kw):
            experts.append(kw["inputs"][1].shape[0])
            return kernel(**kw)

        return launch

    monkeypatch.setattr(tile_gather, "kernel", observe_gather)
    count = switch.num_experts
    assert _same_bits(
        switch(xs, indices, sorted_indices=True), _want(tensors, xs, indices)
    )
    assert experts == [1, count]


@needs_steel
def test_dense_steel_ragged_k_prefills(monkeypatch):
    quantized, layer = _pair(160, 40, bits=4)
    x = _x((33, 160), seed=5)
    want = _one_expert(quantized, x)
    monkeypatch.setattr(FormatLinear, "_decoded", _no_decode)
    for _ in range(2):
        assert _same_bits(layer(x), want)


@needs_steel
def test_dense_steel_grid_has_no_extra_expert_row(toy, monkeypatch):
    k, rows = 256, 17
    toy(_tile_toy("dense_steel_grid"))
    tensors = {key: a[0] for key, a in _stack(64, k).items()}
    layer = FormatLinear("tiletoy", 8, 1, k, tensors)
    x = _x((rows, k), mx.float16)
    launches = []
    build = tile_dense._kernel

    def observed(face, policy):
        kernel = build(face, policy)

        def record(**kwargs):
            count = kwargs["inputs"][0].size // k
            launches.append((count, policy.bm, kwargs["grid"], len(kwargs["inputs"])))
            return kernel(**kwargs)

        return record

    monkeypatch.setattr(tile_dense, "_kernel", observed)
    mx.eval(layer(x))
    assert launches and all(
        grid[1] == (count + bm - 1) // bm and nin == 3
        for count, bm, grid, nin in launches
    )


@needs_nax
def test_nax_dense_grid_limit_uses_the_number_of_rows(monkeypatch):
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
    assert layer(mx.broadcast_to(x, (2**31 - 1, 256))) is accepted
    with pytest.raises(DecodedError):
        layer(mx.broadcast_to(x, (2, 2**30, 256)))


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
    for n in (32, 48):
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


def test_policy_for_return_annotation_includes_mma_and_msg():
    """pre-kit-82917ced NIT 4: policy_for return hint names MmaTilePolicy and
    MsgTilePolicy, and module docstring reflects rows >= 2."""
    import typing

    from omlx.formats.tuning import MmaTilePolicy, MsgTilePolicy

    args = typing.get_args(typing.get_type_hints(tile_dense.policy_for)["return"])
    assert MmaTilePolicy in args
    assert MsgTilePolicy in args
    assert "rows >= 2" in (tile_dense.__doc__ or "")


def _stacked_gdn_layer(dtype):
    k, n, bits, gs = 2560, 16480, 8, 64
    return FormatLinear(
        "affine",
        bits,
        gs,
        k,
        {
            "weight": mx.zeros((n, k * bits // 32), mx.uint32),
            "scales": mx.zeros((n, k // gs), dtype),
            "biases": mx.zeros((n, k // gs), dtype),
        },
    )


def test_three_row_stacked_gdn_projection_selects_measured_m5_route():
    from omlx.formats import tuning

    tuning.reset_cache()
    layer = _stacked_gdn_layer(mx.bfloat16)
    m5 = tile_dense.build_plan(layer, tensor_units=True, chip="applegpu_g17s_c40")
    routes = [tile_dense.route_for(m5, n, mx.bfloat16) for n in (2, 3, 4)]
    assert [route.backend for route in routes] == ["mlx", "dot", "mlx"]
    assert routes[1].policy == tuning.DenseTilePolicy("dot")

    # build_plan's dot_ok gate: with no Dot face the shape falls back to MLX.
    no_dot = tile_dense.build_plan(
        layer, tensor_units=True, chip="applegpu_g17s_c40", dot_face=None
    )
    assert [
        tile_dense.route_for(no_dot, n, mx.bfloat16).backend for n in (2, 3, 4)
    ] == ["mlx", "mlx", "mlx"]

    # The entry is M5-only: M3 Ultra keeps MLX at every row.
    m3 = tile_dense.build_plan(layer, tensor_units=False, chip="applegpu_g15d_c80")
    assert [tile_dense.route_for(m3, n, mx.bfloat16).backend for n in (2, 3, 4)] == [
        "mlx",
        "mlx",
        "mlx",
    ]

    # The measured entry is bf16 only; an fp16 layer stays on MLX (pinned).
    fp16 = tile_dense.build_plan(
        _stacked_gdn_layer(mx.float16), tensor_units=True, chip="applegpu_g17s_c40"
    )
    assert [tile_dense.route_for(fp16, n, mx.float16).backend for n in (2, 3, 4)] == [
        "mlx",
        "mlx",
        "mlx",
    ]


def test_three_row_stacked_gdn_projection_three_rows_equal_single_rows(monkeypatch):
    """The measured 3-row Dot route holds C2 against nn.QuantizedLinear and its
    per-row calls (row-exact to rounding, not bitwise)."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    from omlx.formats import tuning

    # Pin the host chip to the measured M5 so FormatLinear.__call__ itself
    # takes the Dot plan (the local plan below only proves the route table).
    monkeypatch.setattr(tuning, "current_chip", lambda: "applegpu_g17s_c40")
    tuning.reset_cache()
    k, n, gs = 2560, 16480, 64
    layer = _stacked_gdn_layer(mx.bfloat16)
    mx.random.seed(61)
    dense = nn.Linear(k, n, bias=False)
    dense.set_dtype(mx.bfloat16)
    quantized = dense.to_quantized(group_size=gs, bits=8)
    mx.eval(quantized.parameters())
    layer.update(quantized.parameters())
    plan = tile_dense.build_plan(layer, tensor_units=True, chip="applegpu_g17s_c40")
    assert tile_dense.route_for(plan, 3, mx.bfloat16).backend == "dot"
    x3 = mx.random.normal((3, 1, k), dtype=mx.bfloat16)
    mx.eval(x3)
    dot_rows = []
    real_dot_rows = FormatLinear._dot_rows

    def spy(self, kept, x, bias):
        dot_rows.append(x.shape)
        return real_dot_rows(self, kept, x, bias)

    monkeypatch.setattr(FormatLinear, "_dot_rows", spy)
    got = layer(x3)
    assert dot_rows == [x3.shape]
    per_row = mx.concatenate([layer(x3[i : i + 1]) for i in range(3)], axis=0)
    mx.eval(got, per_row)

    # Contract C2 (rule 33): a reordered 3-row route is judged against the FP32
    # reference on the same dequantized weights; the per-row calls sit at the
    # same bar (row-exact to rounding, not bitwise).
    w32 = mx.dequantize(
        quantized["weight"],
        quantized["scales"].astype(mx.float32),
        quantized["biases"].astype(mx.float32),
        group_size=gs,
        bits=8,
    ).astype(mx.float32)
    ref = x3.astype(mx.float32) @ w32.T
    base = quantized(x3).astype(mx.float32)
    rel = lambda a: float(mx.linalg.norm(a - ref) / mx.linalg.norm(ref))  # noqa: E731
    bar = 1.05 * rel(base) + 1e-6
    assert rel(got.astype(mx.float32)) <= bar
    assert rel(per_row.astype(mx.float32)) <= bar
