# SPDX-License-Identifier: Apache-2.0
"""A single Qwen projection alternates ordinary decode and dense prefill."""

import json

import mlx.core as mx
import numpy as np
import pytest

from omlx import formats
from omlx.formats import kernels, linear, testing, tile_dense
from omlx.formats.linear import FormatLinear
from omlx.patches.format_load import loading
from tests.test_formats_stock_dense import _equivalent_layer


def _dense_route() -> str:
    return (
        "format_linear_tile_nax" if linear._nax_host() else "format_linear_tile_steel"
    )


def _builder(tmp_path):
    """Build a format layer as a Qwen load does: verified family and load."""

    def build(fmt, bits, group, k, tensors, family="qwen4_exp"):
        mode = fmt if isinstance(fmt, str) else fmt.mode
        config = {"model_type": family, "omlx_quant_formats": {mode: {}}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        with testing.verified(family), loading(tmp_path):
            return FormatLinear(fmt, bits, group, k, tensors)

    return build


def _one_expert(layer, x):
    """The dense route's arithmetic, written out independently: a
    one-expert sorted gather over the layer's decoded matrix."""
    ids = mx.zeros((x.shape[0],), mx.uint32)
    if layer.mode == "affine":
        from omlx.patches import m5_gather_qmm_nax as nax

        packed = (layer["weight"], layer["scales"], layer["biases"])
        return nax._stock_gather_qmm()(
            x.reshape(-1, 1, x.shape[-1]),
            *(a[None] for a in packed),
            rhs_indices=ids,
            transpose=True,
            sorted_indices=True,
            bits=layer.bits,
            group_size=layer.group_size,
        ).reshape(*x.shape[:-1], layer.output_dims)
    tensors = {name: layer[name] for name in layer._tensors}
    weight = layer.fmt.decode(tensors, layer.bits, layer.group_size).astype(x.dtype)
    return mx.gather_mm(
        x[:, None, :], weight.T[None], rhs_indices=ids, sorted_indices=True
    )[:, 0]


@pytest.mark.parametrize("mode", ("stock", "plugin"))
@pytest.mark.parametrize("schedule", ((2, 64, 2), (64, 2, 64)))
def test_one_layer_reuses_its_schedule_routes_across_requests(
    mode, schedule, tmp_path, monkeypatch
):
    build = _builder(tmp_path)

    def make_layer():
        # N=1024 rows 64 take the steel/NAX dense tile on both chips; two
        # rows take the schedule's route for the mode (MLX for a stock
        # layer, the format's wide tile for a plugin's), lead H4.
        layer, _, _ = _equivalent_layer(mode, build, 1024, 2560)
        return layer

    x = mx.random.normal((64, 2560), key=mx.random.key(83)).astype(mx.float16)
    # Each independent cold call is the oracle for that shape, in a registry
    # of its own: the warmed layer's proofs stay cold until it is called.
    expected = {}
    with testing.fresh_proofs():
        for m in (2, 64):
            expected[m] = np.asarray(make_layer()(x[:m]).view(mx.uint16))
    assert not kernels._PROOFS, "an oracle warmed the warmed layer's proofs"
    oracle = _one_expert(make_layer(), x)
    assert np.array_equal(
        np.asarray(oracle.view(mx.uint16)), expected[64]
    ), "the dense route must have the one-expert gather's bits"

    layer = make_layer()
    built = []
    dense_tile = tile_dense.build_plan

    def dense(held, **kwargs):
        built.append(held is layer)
        return dense_tile(held, **kwargs)

    projected = []
    project = tile_dense.project

    def record(plan, z, **kwargs):
        projected.append((plan, z.shape))
        return project(plan, z, **kwargs)

    monkeypatch.setattr(tile_dense, "build_plan", dense)
    monkeypatch.setattr(tile_dense, "project", record)
    for m in schedule:
        actual = np.asarray(layer(x[:m]).view(mx.uint16))
        np.testing.assert_array_equal(actual, expected[m])

    assert built == [True], "the layer built its schedule more than once"
    signatures = list(dict.fromkeys(schedule))
    assert [shape for _, shape in projected] == [(m, 2560) for m in signatures]
    kept = vars(layer)[linear._KEPT]
    plan = kept.dense
    assert kept.plan is not None and plan is not None
    for tile, _ in projected:
        assert tile is plan
        assert tile.proof is kernels.proof(
            _dense_route(), tile.face, tuple(a.dtype for a in tile.operands)
        )
    ran = {key[0] for key, proof in kernels._PROOFS.items() if proof.ran}
    assert _dense_route() in ran

    # A write retires the schedule whole; the next calls resolve and build
    # it again, and neither route replays the old operands.
    before = dict(expected)
    scale = formats.tensor_names(layer)[1]
    layer[scale] = 0.5 * mx.ones_like(layer[scale])
    fresh = make_layer()
    fresh[scale] = 0.5 * mx.ones_like(fresh[scale])
    with testing.fresh_proofs():
        for m in (2, 64):
            expected[m] = np.asarray(fresh(x[:m]).view(mx.uint16))
    for m in (2, 64):
        assert not np.array_equal(before[m], expected[m]), "the write changed nothing"
    built.clear()
    projected.clear()
    for m in schedule:
        actual = np.asarray(layer(x[:m]).view(mx.uint16))
        np.testing.assert_array_equal(actual, expected[m])
    assert built == [True]
    assert len(projected) == len(signatures)
