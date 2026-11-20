# SPDX-License-Identifier: Apache-2.0
"""The SwiGLU of a fused format switch projection through its own tile face."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from threading import Barrier

import mlx.core as mx
import pytest
from mlx_vlm.models.switch_layers import SwiGLU, SwitchGLU

from omlx import formats
from omlx.formats import kernels, testing
from omlx.formats import linear as fmt_linear
from omlx.formats.linear import FormatSwitchLinear
from omlx.patches import m5_gather_qmm as gather
from omlx.patches import m5_gather_qmm_a8 as a8
from omlx.patches import m5_gather_qmm_nax as nax
from omlx.patches import qwen35_moe_gate_up as fusion
from tests.test_formats_kernels import _toy as _dot_toy
from tests.test_formats_tile_prefill import (
    _COUNTS,
    _TILE,
    E,
    _rows,
    _stack,
    _tile_toy,
    needs_nax,
)
from tests.test_formats_tile_prefill import toy as toy
from tests.test_m5_gather_qmm_nax import _assert_bitwise, _gathers, _glm5_language
from tests.test_m5_gather_qmm_nax import installed as installed
from tests.test_m5_gather_qmm_nax import qwen_regroup as qwen_regroup
from tests.test_qwen35_moe_gate_up import _FakeQwen4Model

pytestmark = needs_nax


@pytest.fixture(autouse=True)
def by_hand():
    with getattr(testing, "verified", lambda *_: nullcontext())("qwen4_exp"):
        yield


def _layer(mode, n=128, k=256, seed=47):
    if mode == "tiletoy":
        return FormatSwitchLinear(mode, 8, 1, k, _stack(n, k, seed=seed))
    mx.random.seed(seed)
    w = mx.random.normal((E, n, k)).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(w, group_size=64, bits=4)
    tensors = dict(weight=weight, scales=scales, biases=biases)
    return FormatSwitchLinear(mode, 4, 64, k, tensors)


def _reference(layer, x, idx, activation):
    gate, up = mx.split(layer(x, idx, sorted_indices=True), 2, axis=-1)
    return activation(up, gate)


@pytest.mark.parametrize("mode", ["tiletoy", "affine"])
@pytest.mark.parametrize("mapped", [False, True])
def test_a_fused_format_stack_uses_its_own_face(
    installed, toy, monkeypatch, mode, mapped
):
    toy(_tile_toy("epilogue"))
    layer = _layer(mode)
    x, idx = _rows(256)
    row_map = mx.arange(idx.size, dtype=mx.uint32)[::-1]
    tokens = (x, row_map) if mapped else None
    materialized = x[row_map] if mapped else x
    stock_calls = []
    stock = nax.sorted_gather_qmm_swiglu

    def observe_stock(*args, **kwargs):
        stock_calls.append(True)
        return stock(*args, **kwargs)

    monkeypatch.setattr(nax, "sorted_gather_qmm_swiglu", observe_stock)
    plain = SwiGLU()
    clamped = _glm5_language().Glm5NextClampedSwiGLU
    previous = None
    for activation in (plain, clamped(0.5), clamped(2.0), plain):
        expected = _reference(layer, materialized, idx, activation)
        if previous is not None:
            assert not mx.array_equal(expected, previous).item()
        got = gather.fused_gate_up_activation(
            layer, materialized, idx, activation, token_rows=tokens
        )
        assert got is not None
        _assert_bitwise(got, expected)
        assert not stock_calls, "the format stack went through raw stock operands"
        previous = expected


def test_mapped_and_unmapped_graphs_keep_their_own_operands(installed, toy):
    toy(_tile_toy("epilogue_map"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    mapped = mx.arange(idx.size, dtype=mx.uint32)[::-1]
    activation = _glm5_language().Glm5NextClampedSwiGLU

    def graphs():
        return {g for proof in kernels._PROOFS.values() for g in proof.graphs.values()}

    def call(entries, limit):
        return gather.fused_gate_up_activation(
            layer, x[mapped], idx, activation(limit), token_rows=entries
        )

    want1 = _reference(layer, x[mapped], idx, activation(0.5))
    want2 = _reference(layer, x[mapped], idx, activation(2.0))
    assert not mx.array_equal(want1, want2).item()
    previous = graphs()
    off = call(None, 0.5)
    assert off is not None
    off_gathers = _gathers(off)
    _assert_bitwise(off, want1)
    after_off = graphs()
    assert after_off > previous
    on = call((x, mapped), 0.5)
    assert on is not None and _gathers(on) == off_gathers - 1
    _assert_bitwise(on, want1)
    after_on = graphs()
    assert after_on > after_off
    _assert_bitwise(call((x, mapped), 2.0), want2)
    after_limit = graphs()
    assert after_limit > after_on
    _assert_bitwise(call((x, mapped), 0.5), want1)
    assert graphs() == after_limit


@pytest.mark.parametrize(
    "defect",
    [
        "map_dtype",
        "token_dtype",
        "token_width",
        "empty",
        "map_length",
        "map_ndim",
        "token_overflow",
    ],
)
def test_an_invalid_row_map_uses_the_materialized_epilogue(
    installed, toy, defect, monkeypatch
):
    build = nax._face_kernel

    def valid_build(face, kind="plain"):
        assert kind != "act_map", "invalid operands reached the mapped kernel"
        return build(face, kind)

    monkeypatch.setattr(nax, "_face_kernel", valid_build)
    toy(_tile_toy("epilogue_bad_map"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    token_rows = x
    row_map = mx.arange(idx.size, dtype=mx.uint32)
    if defect == "map_dtype":
        row_map = row_map.astype(mx.float32)
    elif defect == "token_dtype":
        token_rows = token_rows.astype(
            mx.float16 if x.dtype == mx.bfloat16 else mx.bfloat16
        )
    elif defect == "token_width":
        token_rows = x[..., :128]
    elif defect == "map_length":
        row_map = row_map[:-1]
    elif defect == "map_ndim":
        row_map = row_map[:, None]
    elif defect == "token_overflow":
        token_rows = mx.broadcast_to(x[:1], (1 << 24, 1, 256))
    else:
        token_rows = x[:0]
    got = gather.fused_gate_up_activation(
        layer, x, idx, SwiGLU(), (token_rows, row_map)
    )
    assert got is not None
    _assert_bitwise(got, _reference(layer, x, idx, SwiGLU()))


def test_unavailable_mapped_epilogue_keeps_the_materialized_route(
    installed, toy, monkeypatch
):
    toy(_tile_toy("epilogue_missing"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    mapped = mx.arange(idx.size, dtype=mx.uint32)[::-1]
    build = nax._face_kernel
    monkeypatch.setattr(
        nax,
        "_face_kernel",
        lambda face, kind="plain": (None if kind == "act_map" else build(face, kind)),
    )
    got = gather.fused_gate_up_activation(layer, x[mapped], idx, SwiGLU(), (x, mapped))
    assert got is not None
    _assert_bitwise(got, _reference(layer, x[mapped], idx, SwiGLU()))


def test_the_stand_in_proves_each_epilogue_metal_once(installed, toy, monkeypatch):
    toy(_tile_toy("epilogue_proof"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    row_map = mx.arange(idx.size, dtype=mx.uint32)[::-1]
    materialized = x[row_map]
    launch = nax._launch_face
    proofs = []

    def record(
        kernel, x, operands, indices, config, stream, epi=0, limit=None, row_map=None
    ):
        if epi and indices.shape == (16,):
            proofs.append((x.shape, epi, row_map is not None))
            assert all(a.size >= 8 for a in operands)
            if row_map is not None:
                assert row_map.shape == (16,) and row_map.dtype == mx.uint32
        return launch(kernel, x, operands, indices, config, stream, epi, limit, row_map)

    monkeypatch.setattr(nax, "_launch_face", record)
    clamp = _glm5_language().Glm5NextClampedSwiGLU
    for activation, mapping in (
        (SwiGLU(), None),
        (SwiGLU(), (x, row_map)),
        (clamp(0.5), (x, row_map)),
        (clamp(2), (x, row_map)),
        (clamp(0.5), (x, row_map)),
    ):
        got = gather.fused_gate_up_activation(
            layer, materialized, idx, activation, mapping
        )
        assert got is not None
        _assert_bitwise(got, _reference(layer, materialized, idx, activation))
    assert proofs == [
        ((16, 1, 256), nax._EPI_SWIGLU, False),
        ((1, 1, 256), nax._EPI_SWIGLU, True),
        ((1, 1, 256), nax._EPI_CLAMPED, True),
    ]


def test_traced_epilogue_reads_the_current_format_operands(installed, toy):
    toy(_tile_toy("epilogue_traced"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    row_map = mx.arange(idx.size, dtype=mx.uint32)[::-1]

    def project(x):
        y = gather.fused_gate_up_activation(
            layer, x[row_map], idx, SwiGLU(), (x, row_map)
        )
        assert y is not None
        return y

    compiled = mx.compile(project, inputs=layer.state)
    before = _reference(layer, x[row_map], idx, SwiGLU())
    _assert_bitwise(compiled(x), before)
    layer["group_scales"] = layer["group_scales"] * 0.5
    after = _reference(layer, x[row_map], idx, SwiGLU())
    assert not mx.array_equal(before, after).item()
    _assert_bitwise(compiled(x), after)


def test_one_failed_epilogue_retires_both_threads_graphs(installed, toy):
    toy(_tile_toy("epilogue_threads"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    row_map = mx.arange(idx.size, dtype=mx.uint32)[::-1]
    materialized = x[row_map]
    want = _reference(layer, materialized, idx, SwiGLU())
    mx.eval(x, materialized, idx, row_map, want, *layer.parameters().values())
    barrier = Barrier(2, timeout=30)

    def run(worker):
        def call():
            return gather.fused_gate_up_activation(
                layer, materialized, idx, SwiGLU(), (x, row_map)
            )

        before = {
            g for proof in kernels._PROOFS.values() for g in proof.graphs.values()
        }
        got = call()
        assert got is not None
        _assert_bitwise(got, want)
        (proof,) = [
            proof
            for proof in kernels._PROOFS.values()
            if set(proof.graphs.values()) - before
        ]
        (graph,) = set(proof.graphs.values()) - before
        barrier.wait()
        if worker == 0:
            proof.fail()
        barrier.wait()
        assert graph not in proof.graphs.values()
        fallback = call()
        assert fallback is not None
        _assert_bitwise(fallback, want)
        return graph

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.map(run, (0, 1))
    assert a is not b


@pytest.mark.parametrize(
    "reason", ["bias", "activation", "few_rows", "columns", "limit"]
)
def test_a_format_epilogue_declines_unsupported_calls(
    installed, toy, reason, monkeypatch
):
    build = nax._face_kernel

    def valid_build(face, kind="plain"):
        assert kind == "plain", "unsupported call reached an epilogue kernel"
        return build(face, kind)

    monkeypatch.setattr(nax, "_face_kernel", valid_build)
    toy(_tile_toy("epilogue_decline"))
    layer = _layer("tiletoy", n=96 if reason == "columns" else 128)
    x, idx = _rows(256)
    if reason == "bias":
        layer["bias"] = mx.zeros((E, layer.output_dims), x.dtype)
    if reason == "few_rows":
        x, idx = x[:15], idx[:15]
    act = (lambda up, gate: up + gate) if reason == "activation" else SwiGLU()
    if reason == "limit":
        act = _glm5_language().Glm5NextClampedSwiGLU(float("inf"))
    assert gather.fused_gate_up_activation(layer, x, idx, act) is None


@pytest.mark.usefixtures("qwen_regroup")
def test_a_tagged_vlm_switch_uses_the_mapped_format_epilogue(
    installed, toy, monkeypatch
):
    toy(_tile_toy("epilogue_block"))
    k, inter, tokens = 256, 128, 128
    glu = SwitchGLU(k, inter, E)
    for name, n, width, seed in (
        ("gate_proj", inter, k, 61),
        ("up_proj", inter, k, 63),
        ("down_proj", k, inter, 67),
    ):
        glu[name] = _layer("tiletoy", n, width, seed)
    glu.eval()
    x = mx.random.normal((1, tokens, k)).astype(mx.bfloat16)
    idx = mx.random.randint(0, E, (1, tokens, 2)).astype(mx.uint32)
    weights = mx.full(idx.shape, 0.5, dtype=x.dtype)
    unfused = glu(x, idx, weights)
    mx.eval(unfused)
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert fusion.apply_qwen35_moe_gate_up_fusion(model) == 1
    assert a8.tag_routed_a8_modules(model, tokens) == 1
    states = []
    fused = gather.fused_gate_up_activation

    def observe(layer, rows, ids, activation, token_rows=None):
        out = fused(layer, rows, ids, activation, token_rows)
        states.append(
            (layer is glu.gate_up_proj, token_rows is not None, out is not None)
        )
        return out

    monkeypatch.setattr(fusion, "fused_gate_up_activation", observe)
    got = glu(x, idx, weights)
    _assert_bitwise(got, unfused)
    assert states == [(True, True, True)]


class _CodesFormat(formats.QuantFormat):
    """A plugin whose codes tensor is not called ``weight``: the layer
    keeps ``expert_codes`` and the face's own ``gain``."""

    contract = formats.CONTRACT
    mode = "codes_epilogue"
    codes_name = "expert_codes"
    tensors = ("gain",)

    def __init__(self, tile):
        self.tile_face = tile

    def layout(self, k, n, bits, group_size):
        return {self.codes_name: ((n, k), mx.uint8), "gain": ((n,), mx.float32)}

    def dot(self, layer, k, n):
        return replace(_dot_toy(self.mode), tensors=self.tensors)

    def tile(self, layer, k, n):
        return self.tile_face

    def decode(self, tensors, bits, group_size):
        codes = tensors[self.codes_name].astype(mx.float32) - 128.0
        return codes * mx.expand_dims(tensors["gain"], -1)


def _codes_layer(monkeypatch):
    tile = replace(_tile_toy("codes_epilogue"), tensors=("gain",))
    fmt = _CodesFormat(tile)
    formats.load()
    monkeypatch.setitem(formats._FORMATS, fmt.mode, fmt)
    mx.random.seed(47)
    codes = mx.random.randint(0, 256, (E, 128, 256)).astype(mx.uint8)
    gain = mx.random.normal((E, 128)) * 0.01
    layer = FormatSwitchLinear(fmt, 8, 1, 256, {fmt.codes_name: codes, "gain": gain})
    return layer, codes


def test_a_codes_named_layer_keeps_its_warm_mapped_graph(installed, monkeypatch):
    layer, codes = _codes_layer(monkeypatch)
    assert "weight" not in layer
    x, idx = _rows(256)
    row_map = mx.arange(idx.size, dtype=mx.uint32)[::-1]
    materialized = x[row_map]
    activation = SwiGLU()
    calls = []
    original = fmt_linear._swiglu

    def observe(plan, tile, tokens, indices, limit, row_map=None):
        calls.append((plan, tile, tile.operands[0]))
        return original(plan, tile, tokens, indices, limit, row_map)

    monkeypatch.setattr(fmt_linear, "_swiglu", observe)

    def want(packed):
        fmt = formats.get(layer.mode)
        weights = fmt.decode(
            {fmt.codes_name: packed, "gain": layer["gain"]}, 8, 1
        ).astype(x.dtype)
        projected = mx.gather_mm(
            materialized,
            weights.swapaxes(-1, -2),
            rhs_indices=idx,
            sorted_indices=True,
        )
        gate, up = mx.split(projected, 2, axis=-1)
        return activation(up, gate)

    def call(packed):
        if layer[layer.fmt.codes_name] is not packed:
            layer[layer.fmt.codes_name] = packed
        got = gather.fused_gate_up_activation(
            layer, materialized, idx, activation, (x, row_map)
        )
        assert got is not None
        return got

    def graphs():
        return {
            graph
            for proof in kernels._PROOFS.values()
            for graph in proof.graphs.values()
        }

    before = graphs()
    cold = call(codes)
    _assert_bitwise(cold, want(codes))
    after_cold = graphs()
    assert after_cold > before
    assert calls[-1][2] is layer[layer.fmt.codes_name]
    warm = call(codes)
    _assert_bitwise(warm, want(codes))
    assert graphs() == after_cold, "the warm call rebuilt its graph"
    assert calls[-1][2] is layer[layer.fmt.codes_name]
    packed = calls[-1][2]
    rich = mx.zeros_like(packed) + 128
    changed = call(rich)
    assert not mx.array_equal(changed, warm).item()
    _assert_bitwise(changed, want(rich))


def test_a_failed_epilogue_proof_builds_no_other_kernel(installed, toy, monkeypatch):
    toy(_tile_toy("epilogue_failed_proof"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    kept = layer._keep()
    tile = layer._keep_tile(kept)
    fmt_linear._epilogue_proof(tile, kept[0], False).fail()
    kinds = []
    build = nax._face_kernel

    def record(face, kind="plain"):
        kinds.append(kind)
        return build(face, kind)

    monkeypatch.setattr(nax, "_face_kernel", record)
    assert fmt_linear._swiglu(kept[0], tile, x, idx, None) is None
    assert kinds == [], "a failed proof still built its epilogue kernel"


def test_sorted_swiglu_declines_without_a_nax_host(installed, toy, monkeypatch):
    toy(_tile_toy("epilogue_steel_decline"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    monkeypatch.setattr(fmt_linear, "_nax_host", lambda: False)
    built = []
    real = fmt_linear._tile

    def record(layer_, plan):
        built.append(1)
        return real(layer_, plan)

    monkeypatch.setattr(fmt_linear, "_tile", record)
    assert layer._sorted_swiglu(x, idx, None, None) is None
    assert not built, "a non-NAX host built a tile plan"


def test_an_inexact_dtype_and_a_face_without_a_tile_decline(installed, toy):
    toy(_tile_toy("epilogue_exact"))
    affine = _layer("affine")
    x16, idx = _rows(256, dtype=mx.float16)
    assert gather.fused_gate_up_activation(affine, x16, idx, SwiGLU()) is None
    toy(None)
    no_tile = _layer("tiletoy")
    x, idx = _rows(256)
    assert gather.fused_gate_up_activation(no_tile, x, idx, SwiGLU()) is None


def test_a_warm_epilogue_keeps_one_graph_per_expert_count(installed, toy):
    toy(_tile_toy("epilogue_experts"))
    stack = _stack(128, 256, seed=5)
    counts = (70, 0, 5, 256)
    four = FormatSwitchLinear(
        "tiletoy",
        8,
        1,
        256,
        {"codes": stack["codes"][:4], "group_scales": stack["group_scales"][:4]},
    )
    ten = _layer("tiletoy")
    x4, idx4 = _rows(256, counts=counts)
    x10, idx10 = _rows(256)
    assert idx4.size == idx10.size == sum(_COUNTS)
    map4 = mx.arange(idx4.size, dtype=mx.uint32)[::-1]
    map10 = mx.arange(idx10.size, dtype=mx.uint32)[::-1]

    def graphs():
        return {
            graph
            for proof in kernels._PROOFS.values()
            for graph in proof.graphs.values()
        }

    def call(layer, x, idx, row_map):
        got = gather.fused_gate_up_activation(
            layer, x[row_map], idx, SwiGLU(), (x, row_map)
        )
        assert got is not None
        return got

    before = graphs()
    four_out = call(four, x4, idx4, map4)
    after_four = graphs()
    assert len(after_four - before) == 1
    ten_out = call(ten, x10, idx10, map10)
    after_ten = graphs()
    assert len(after_ten - after_four) == 1, "both expert counts shared a graph"
    _assert_bitwise(four_out, _reference(four, x4[map4], idx4, SwiGLU()))
    _assert_bitwise(ten_out, _reference(ten, x10[map10], idx10, SwiGLU()))
    warm_before = graphs()
    assert len(call(four, x4, idx4, map4)) == len(four_out)
    assert graphs() == warm_before, "a warm call rebuilt a graph"
    narrow, narrow_idx = _rows(256, counts=(30, 0, 5, 29))
    narrow_map = mx.arange(narrow_idx.size, dtype=mx.uint32)[::-1]
    want_narrow = _reference(four, narrow[narrow_map], narrow_idx, SwiGLU())
    before_narrow = graphs()
    _assert_bitwise(call(four, narrow, narrow_idx, narrow_map), want_narrow)
    assert len(graphs() - before_narrow) == 1, "a second shape shared its graph"


def test_a_failed_mapped_epilogue_keeps_unmapped_and_plain_tile(installed, toy):
    toy(_tile_toy("epilogue_shared_proof"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    row_map = mx.arange(idx.size, dtype=mx.uint32)[::-1]
    materialized = x[row_map]
    want_mapped = _reference(layer, materialized, idx, SwiGLU())
    want_plain = layer(x, idx, sorted_indices=True)
    assert (
        gather.fused_gate_up_activation(
            layer, materialized, idx, SwiGLU(), (x, row_map)
        )
        is not None
    )
    assert gather.fused_gate_up_activation(layer, x, idx, SwiGLU()) is not None
    kept = layer.__dict__[fmt_linear._KEPT]
    tile = kept[3]
    proof_for = getattr(
        fmt_linear, "_epilogue_proof", lambda tile_, plan_, mapped_: tile_.proof
    )
    mapped_proof = proof_for(tile, kept[0], True)
    mapped_proof.fail()
    got = gather.fused_gate_up_activation(
        layer, materialized, idx, SwiGLU(), (x, row_map)
    )
    assert got is not None
    _assert_bitwise(got, want_mapped)
    assert mapped_proof.failed
    assert not tile.proof.failed, "the epilogue took the gather's proof down"
    _assert_bitwise(layer(x, idx, sorted_indices=True), want_plain)


def test_a_mapped_epilogue_checks_geometry_once(installed, toy, monkeypatch):
    toy(_tile_toy("epilogue_geometry_once"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    row_map = mx.arange(idx.size, dtype=mx.uint32)[::-1]
    calls = []
    real = nax.geometry

    def observe(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(nax, "geometry", observe)
    got = gather.fused_gate_up_activation(
        layer, x[row_map], idx, SwiGLU(), (x, row_map)
    )
    assert got is not None
    assert len(calls) == 1, "the mapped call checked geometry twice"


def test_sorted_swiglu_declines_a_biased_layer(installed, toy):
    toy(_tile_toy("epilogue_biased"))
    layer = _layer("tiletoy")
    layer["bias"] = mx.zeros((E, layer.output_dims), mx.bfloat16)
    x, idx = _rows(256)
    assert layer._sorted_swiglu(x, idx, None, None) is None


def test_an_epilogue_failure_logs_the_epilogue_message(installed, toy, caplog):
    toy(_tile_toy("epilogue_log", _TILE.replace("RUN = 8;", "RUN = 16;")))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    caplog.set_level("WARNING")
    got = gather.fused_gate_up_activation(layer, x, idx, SwiGLU())
    assert got is None
    assert "epilogue kernel failed" in caplog.text
    assert "rows kernel fallback" not in caplog.text


def test_a_layer_keeps_its_codes_when_they_are_not_first(installed, monkeypatch):
    layer, codes = _codes_layer(monkeypatch)
    layer.__dict__["_tensors"] = (layer.fmt.tensors[0], layer.fmt.codes_name)
    x, idx = _rows(256)
    row_map = mx.arange(idx.size, dtype=mx.uint32)[::-1]
    materialized = x[row_map]
    fmt = formats.get(layer.mode)
    want = fmt.decode({fmt.codes_name: codes, "gain": layer["gain"]}, 8, 1).astype(
        x.dtype
    )
    projected = mx.gather_mm(
        materialized, want.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True
    )
    gate, up = mx.split(projected, 2, axis=-1)
    expected = SwiGLU()(up, gate)
    before = {
        graph for proof in kernels._PROOFS.values() for graph in proof.graphs.values()
    }
    first = gather.fused_gate_up_activation(
        layer, materialized, idx, SwiGLU(), (x, row_map)
    )
    assert first is not None
    _assert_bitwise(first, expected)
    after = {
        graph for proof in kernels._PROOFS.values() for graph in proof.graphs.values()
    }
    assert after > before
    warm = gather.fused_gate_up_activation(
        layer, materialized, idx, SwiGLU(), (x, row_map)
    )
    _assert_bitwise(warm, expected)
    assert {
        graph for proof in kernels._PROOFS.values() for graph in proof.graphs.values()
    } == after, "the warm call missed its graph with reordered tensors"


def test_a_layer_below_sixteen_rows_declines_even_with_few_experts(installed, toy):
    toy(_tile_toy("epilogue_few_rows"))
    stack = _stack(128, 256, seed=5)
    layer = FormatSwitchLinear(
        "tiletoy",
        8,
        1,
        256,
        {"codes": stack["codes"][:2], "group_scales": stack["group_scales"][:2]},
    )
    x, idx = _rows(256, counts=(7, 8))
    assert gather.fused_gate_up_activation(layer, x, idx, SwiGLU()) is None


def test_mapped_token_blocks_keep_a_graph_per_token_rows(installed, toy):
    toy(_tile_toy("epilogue_token_rows"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    rows = idx.size
    kept = layer._keep()
    tile = layer._keep_tile(kept)
    proof = fmt_linear._epilogue_proof(tile, kept[0], True)
    seen = []
    for count in (128, 64):
        tokens = mx.random.normal((count, 1, 256)).astype(x.dtype)
        row_map = (mx.arange(rows, dtype=mx.uint32) % count).astype(mx.uint32)
        materialized = tokens[row_map]
        want = _reference(layer, materialized, idx, SwiGLU())
        before = len(proof.graphs)
        got = gather.fused_gate_up_activation(
            layer, materialized, idx, SwiGLU(), (tokens, row_map)
        )
        assert got is not None
        _assert_bitwise(got, want)
        assert len(proof.graphs) == before + 1, "a token block reused a graph"
        seen.append(got)


def test_a_format_epilogue_declines_a_wrong_input_width(installed, toy, monkeypatch):
    toy(_tile_toy("epilogue_wrong_width"))
    layer = _layer("tiletoy")
    x, idx = _rows(256)
    kinds = []
    build = nax._face_kernel

    def record(face, kind="plain"):
        kinds.append(kind)
        return build(face, kind)

    monkeypatch.setattr(nax, "_face_kernel", record)
    got = gather.fused_gate_up_activation(layer, x[..., :128], idx, SwiGLU())
    assert got is None
    assert kinds == [], "a wrong input width reached an epilogue kernel"


def test_a_format_epilogue_declines_below_four_rows_an_expert(
    installed, toy, monkeypatch
):
    toy(_tile_toy("epilogue_few_per_expert"))
    layer = _layer("tiletoy")
    x, idx = _rows(256, counts=(4,) * 9 + (3,))
    kinds = []
    build = nax._face_kernel

    def record(face, kind="plain"):
        kinds.append(kind)
        return build(face, kind)

    monkeypatch.setattr(nax, "_face_kernel", record)
    got = gather.fused_gate_up_activation(layer, x, idx, SwiGLU())
    assert got is None
    assert kinds == [], "few rows an expert reached an epilogue kernel"


def test_warm_epilogue_pins_width_dtype_and_route_count(installed, toy):
    toy(_tile_toy("epilogue_pins"))
    x, idx = _rows(256)
    narrow = _layer("tiletoy", n=128)
    wide = _layer("tiletoy", n=256)
    kept = narrow._keep()
    proof = fmt_linear._epilogue_proof(narrow._keep_tile(kept), kept[0], False)

    def graphs():
        return len(proof.graphs)

    before = graphs()
    got = gather.fused_gate_up_activation(narrow, x, idx, SwiGLU())
    assert got is not None
    _assert_bitwise(got, _reference(narrow, x, idx, SwiGLU()))
    assert graphs() == before + 1
    wide_out = gather.fused_gate_up_activation(wide, x, idx, SwiGLU())
    assert wide_out is not None
    _assert_bitwise(wide_out, _reference(wide, x, idx, SwiGLU()))
    assert graphs() == before + 2, "two widths shared a graph"
    again = gather.fused_gate_up_activation(narrow, x, idx, SwiGLU())
    _assert_bitwise(again, _reference(narrow, x, idx, SwiGLU()))
    assert graphs() == before + 2, "a warm call rebuilt a graph"
    x16, idx16 = _rows(256, dtype=mx.float16)
    half = gather.fused_gate_up_activation(narrow, x16, idx16, SwiGLU())
    if half is not None:
        _assert_bitwise(half, _reference(narrow, x16, idx16, SwiGLU()))
        assert graphs() == before + 3, "two dtypes shared a graph"
    stacked = _stack(128, 256, seed=5)
    four = FormatSwitchLinear(
        "tiletoy",
        8,
        1,
        256,
        {"codes": stacked["codes"][:4], "group_scales": stacked["group_scales"][:4]},
    )
    tokens = mx.random.normal((64, 1, 256)).astype(x.dtype)
    mapped_before = None
    for counts in ((40,) * 4, (128,) * 4):
        xr, idxr = _rows(256, counts=counts)
        row_map = (mx.arange(idxr.size, dtype=mx.uint32) % 64).astype(mx.uint32)
        materialized = tokens[row_map]
        mkept = four._keep()
        mapped = fmt_linear._epilogue_proof(four._keep_tile(mkept), mkept[0], True)
        if mapped_before is None:
            mapped_before = len(mapped.graphs)
        got = gather.fused_gate_up_activation(
            four, materialized, idxr, SwiGLU(), (tokens, row_map)
        )
        assert got is not None
        _assert_bitwise(got, _reference(four, materialized, idxr, SwiGLU()))
    assert len(mapped.graphs) == mapped_before + 2, "two row counts shared a graph"


def test_epilogue_proofs_are_kept_per_width(installed, toy):
    toy(_tile_toy("epilogue_k_isolated"))
    narrow = FormatSwitchLinear("tiletoy", 8, 1, 128, _stack(128, 128, seed=5))
    kept = narrow._keep()
    fmt_linear._epilogue_proof(narrow._keep_tile(kept), kept[0], False).fail()
    wide = FormatSwitchLinear("tiletoy", 8, 1, 256, _stack(128, 256, seed=5))
    x, idx = _rows(256)
    got = gather.fused_gate_up_activation(wide, x, idx, SwiGLU())
    assert got is not None
    _assert_bitwise(got, _reference(wide, x, idx, SwiGLU()))
