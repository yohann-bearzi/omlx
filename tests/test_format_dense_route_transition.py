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
from tests.test_formats_tile_prefill import needs_nax

pytestmark = needs_nax


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


@pytest.mark.parametrize("mode", ("stock", "plugin"))
@pytest.mark.parametrize("schedule", ((2, 64, 2), (64, 2, 64)))
def test_one_layer_reuses_wide_and_dense_plans_across_requests(
    mode, schedule, tmp_path, monkeypatch
):
    build = _builder(tmp_path)

    def make_layer():
        layer, _, _ = _equivalent_layer(mode, build, 512, 2560)
        return layer

    x = mx.random.normal((64, 2560), key=mx.random.key(83)).astype(mx.float16)
    # Each independent cold call is the oracle for that shape. Switching
    # warmed routes on a single layer must select the same graphs and bits.
    expected = {}
    for m in (2, 64):
        expected[m] = np.asarray(make_layer()(x[:m]).view(mx.uint16))
    layer = make_layer()
    built = []
    wide_tile = linear._ordinary_tile
    dense_tile = linear._tile

    def wide(*args):
        built.append("wide")
        return wide_tile(*args)

    def dense(held):
        built.append("dense")
        return dense_tile(held)

    projected = []
    project = tile_dense.project

    def record(tile, z, **kwargs):
        projected.append((tile, z.shape))
        return project(tile, z, **kwargs)

    monkeypatch.setattr(linear, "_ordinary_tile", wide)
    monkeypatch.setattr(linear, "_tile", dense)
    monkeypatch.setattr(tile_dense, "project", record)
    for m in schedule:
        actual = np.asarray(layer(x[:m]).view(mx.uint16))
        np.testing.assert_array_equal(actual, expected[m])

    first, second = ("wide", "dense") if schedule[0] == 2 else ("dense", "wide")
    assert built == [first, second]
    assert [shape for _, shape in projected] == [(m, 2560) for m in schedule if m > 16]
    for tile, _ in projected:
        assert tile is not None
        assert tile.proof is kernels.proof(
            _dense_route(), tile.face, tuple(a.dtype for a in tile.operands)
        )
    kept = vars(layer)[linear._KEPT]
    assert kept.routes["wide"] is kept.wide
    assert kept.dense is not None

    # A write retires both routes together; each one binds the array it
    # reads before its next call, so neither replays the old operands.
    scale = formats.tensor_names(layer)[1]
    layer[scale] = 0.5 * mx.ones_like(layer[scale])
    fresh = make_layer()
    fresh[scale] = 0.5 * mx.ones_like(fresh[scale])
    for m in (2, 64):
        expected[m] = np.asarray(fresh(x[:m]).view(mx.uint16))
    built.clear()
    projected.clear()
    for m in schedule:
        actual = np.asarray(layer(x[:m]).view(mx.uint16))
        np.testing.assert_array_equal(actual, expected[m])
    assert sorted(built) == ["dense", "wide"]
    assert len(projected) == sum(1 for m in schedule if m > 16)
