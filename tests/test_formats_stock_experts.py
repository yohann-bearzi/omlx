# SPDX-License-Identifier: Apache-2.0
"""Flash-Next expert projection shapes through each affine Tile width."""

import json

import mlx.core as mx
import numpy as np
import pytest

from omlx import formats
from omlx.formats import kernels, testing
from omlx.formats.linear import FormatSwitchLinear
from omlx.patches import m5_gather_qmm_nax as nax
from omlx.patches.format_load import loading

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("k,n", [(2560, 640), (640, 2560)])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_sorted_flash_experts_use_their_tile_at_each_affine_width(
    bits, k, n, dtype, tmp_path
):
    from omlx.custom_kernels.nax import is_nax_available

    # Eight experts keep the test synthetic. K/N and the quantization group
    # match the served Flash-Next expert gate/up and down projections.
    experts, group = 8, 64
    wkey, bkey, xkey = mx.random.split(mx.random.key(339 + bits + n), num=3)
    w = (mx.random.normal((experts, n, k), key=wkey) * k**-0.5).astype(dtype)
    packed = mx.quantize(w, group_size=group, bits=bits)
    bias = (mx.random.normal((experts, n), key=bkey) * 0.125).astype(dtype)
    names = ("weight", "scales", "biases")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "omlx_quant_formats": {"affine": {}}})
    )
    with testing.verified("qwen4_exp"), loading(tmp_path):
        layer = FormatSwitchLinear(
            "affine", bits, group, k, dict(zip(names, packed)), bias=bias
        )
    stock = nax._stock_gather_qmm()
    counts = (0, 3, 7, 34, 1, 0, 65, 18)
    x = mx.random.normal((sum(counts), 1, k), key=xkey).astype(dtype)
    # The second sorted call has the same shape but different empty experts
    # and tile boundaries. It must use the new indices and their biases.
    for per_expert in (counts, counts[::-1]):
        idx = mx.array(np.repeat(np.arange(experts, dtype=np.uint32), per_expert))
        want = (
            stock(
                x,
                *packed,
                rhs_indices=idx,
                transpose=True,
                group_size=group,
                bits=bits,
                sorted_indices=True,
            )
            + bias[idx][:, None, :]
        )
        got = layer(x, idx, sorted_indices=True)
        assert got.shape == want.shape == (sum(counts), 1, n)
        assert got.dtype == want.dtype == dtype
        np.testing.assert_array_equal(
            np.asarray(got.view(mx.uint16)), np.asarray(want.view(mx.uint16))
        )
    route = (
        "format_switch_tile_nax" if is_nax_available() else "format_switch_tile_steel"
    )
    face = formats.tile(layer, k, n)
    assert face is not None
    assert any(
        key[:2] == (route, face) and proof.ran and not proof.failed
        for key, proof in kernels._PROOFS.items()
    )
