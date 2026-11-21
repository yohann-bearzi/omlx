# SPDX-License-Identifier: Apache-2.0
"""The affine format's layout: MLX's packing, and floats it fixes."""

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats.affine import FORMAT
from omlx.formats.linear import FormatLinear
from omlx.patches import format_load


@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6, 8))
@pytest.mark.parametrize("group", (32, 64, 128))
def test_the_layout_describes_mlx_packing(bits, group):
    k, n = 256, 7
    packed = mx.quantize(mx.ones((n, k)), group_size=group, bits=bits)
    layout = FORMAT.layout(k, n, bits, group)
    assert layout == {
        "weight": ((n, k * bits // 32), mx.uint32),
        "scales": ((n, k // group), None),
        "biases": ((n, k // group), None),
    }
    for name, stored in zip(("weight", "scales", "biases"), packed):
        assert layout[name][0] == stored.shape


@pytest.mark.parametrize(
    "k,n,bits,group",
    (
        (96, 7, 4, 64),
        (256, 7, 7, 64),
        (256, 7, 4, 16),
        (0, 7, 4, 64),
        (-64, 7, 4, 64),
        (64, 0, 4, 64),
        (64, -1, 4, 64),
    ),
)
def test_the_layout_declines_a_matrix_mlx_cannot_pack(k, n, bits, group):
    assert FORMAT.layout(k, n, bits, group) is None


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16, mx.float32))
def test_a_stored_float_loads_in_its_checkpoint_dtype(dtype, tmp_path):
    format_load.apply_format_load_patch()
    config = {"model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 64}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with format_load.loading(tmp_path) as checkpoint:
        model = nn.Module()
        layer = model.proj = nn.Linear(128, 32, bias=False).to_quantized(
            group_size=64, bits=4
        )
        assert type(layer) is FormatLinear
        # MLX's layer's placeholder dtype, until the checkpoint's own comes.
        assert layer.scales.dtype == mx.float32 and layer.biases.dtype == mx.float32
        weight = mx.ones((32, 16), mx.uint32)
        scales = mx.arange(1, 65).astype(dtype).reshape(32, 2)
        biases = mx.zeros((32, 2), dtype)
        layer.load_weights([("weight", weight), ("scales", scales), ("biases", biases)])
        assert layer.scales is scales and layer.biases is biases
        layer.refuse_placeholders()
        layer.load_weights([("scales", scales.astype(mx.int32))], strict=False)
        with pytest.raises(formats.FormatRefusedError, match="proj.scales"):
            checkpoint.check(model)
        layer.load_weights([("scales", scales)], strict=False)
        checkpoint.check(model)
        assert layer.scales is scales and layer.biases is biases
        layer.refuse_placeholders()


def test_rows_stack_accepts_affine_stored_float_layout():
    from omlx.formats import testing

    with mx.stream(mx.cpu):
        w1 = mx.random.normal((8, 64), key=mx.random.key(1)).astype(mx.bfloat16)
        w2 = mx.random.normal((8, 64), key=mx.random.key(2)).astype(mx.bfloat16)
        names = ("weight", "scales", "biases")
        first = dict(zip(names, mx.quantize(w1, group_size=32, bits=4)))
        second = dict(zip(names, mx.quantize(w2, group_size=32, bits=4)))
        testing.rows_stack(FORMAT, first, second, 64, 4, 32)

        bad_int = {**second, "scales": second["scales"].astype(mx.int32)}
        with pytest.raises(AssertionError):
            testing.rows_stack(FORMAT, first, bad_int, 64, 4, 32)

        bad_pair = {
            **second,
            "scales": second["scales"].astype(mx.float16),
            "biases": second["biases"].astype(mx.float16),
        }
        with pytest.raises(AssertionError):
            testing.rows_stack(FORMAT, first, bad_pair, 64, 4, 32)
