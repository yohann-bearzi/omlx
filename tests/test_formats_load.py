# SPDX-License-Identifier: Apache-2.0
"""A checkpoint's layers of a quant format's mode load as the format's layers
(``omlx.patches.format_load``): through MLX's ``nn.quantize`` and a strict
load, as mlx-lm and mlx-vlm load a checkpoint.
"""

from __future__ import annotations

import inspect
import json
import re
import textwrap

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama
from mlx_lm.models import switch_layers as lm_switch
from mlx_lm.utils import load_model
from mlx_vlm.models import switch_layers as vlm_switch

from omlx import formats
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.patches import format_load
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from tests.test_formats_kernels import _BLOCKS, _matrix
from tests.test_formats_linear import EXPERTS, _ints, _ToyFormat, _want

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_PATCHED = (nn.Linear, lm_switch.SwitchLinear, vlm_switch.SwitchLinear)
_TOY = {"bits": 8, "group_size": 1, "mode": "toy"}


class _Loadable(_ToyFormat):
    """The toy format, with what its checkpoints keep of a matrix: a byte for
    each weight and a scale for each row."""

    def __init__(self, mode="toy"):
        super().__init__()
        self.mode, self.asked = mode, []
        self.accepted, self.refusal = [], None

    def accept(self, record):
        # What it was handed, and whether a layer had been built before.
        self.accepted.append((record, len(self.asked)))
        if self.refusal is not None:
            raise ValueError(self.refusal)

    def layout(self, k, n, bits, group_size):
        self.asked.append((k, n, bits, group_size))
        if bits != 8:
            return None
        return {"weight": ((n, k), mx.uint8), "scales": ((n,), mx.float32)}


@pytest.fixture
def unpatched(monkeypatch):
    """MLX's own ``to_quantized`` on the classes the patch wraps, whatever an
    earlier test applied; the test's own wrappers are undone after it."""
    for cls in _PATCHED:
        original = cls.to_quantized
        monkeypatch.setattr(
            cls, "to_quantized", getattr(original, "__wrapped__", original)
        )


@pytest.fixture
def toy(unpatched, monkeypatch):
    """The load patch, over a registry with the toy format."""
    formats.load()
    fmt = _Loadable()
    monkeypatch.setitem(formats._FORMATS, "toy", fmt)
    format_load.apply_format_load_patch()
    return fmt


class _Model(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.toy = nn.Linear(k, n, bias=True)
        self.dense = nn.Linear(k, n, bias=False)


def _named(config):
    """The loaders' predicate for the layers a quantization config names."""
    return lambda path, module: config.get(path, False)


def _kept(layer):
    return {name: (value.shape, value.dtype) for name, value in layer.items()}


def test_a_named_linear_loads_as_the_formats_layer(toy):
    n, k = 16, 64
    (codes, scales), x, want = _matrix(n, k)
    bias = mx.arange(n).astype(mx.float32)
    model = _Model(k, n)
    dense, random = model.dense.weight, model.toy.bias
    nn.quantize(model, class_predicate=_named({"toy": _TOY}))
    layer = model.toy
    assert type(layer) is FormatLinear and type(model.dense) is nn.Linear
    assert (layer.mode, layer.bits, layer.group_size) == ("toy", 8, 1)
    assert (layer.input_dims, layer.output_dims) == (k, n)
    assert toy.asked == [(k, n, 8, 1)]
    # Placeholders of the layout's shapes and dtypes, and the layer's bias.
    assert _kept(layer) == {
        "weight": ((n, k), mx.uint8),
        "scales": ((n,), mx.float32),
        "bias": ((n,), mx.float32),
    }
    assert mx.array_equal(layer["bias"], random)
    weights = {"toy.weight": codes, "toy.scales": scales, "toy.bias": bias}
    weights["dense.weight"] = dense
    model.load_weights(list(weights.items()), strict=True)
    assert mx.array_equal(layer["weight"], codes)
    assert mx.array_equal(layer(x), want + bias)
    # The load checks a checkpoint's tensors against the layout.
    with pytest.raises(ValueError, match="for parameter toy.scales"):
        model.load_weights(
            list({**weights, "toy.scales": scales[:-1]}.items()), strict=True
        )
    with pytest.raises(ValueError, match="toy.biases"):
        model.load_weights(list({**weights, "toy.biases": scales}.items()), strict=True)


@pytest.mark.parametrize("switch", [lm_switch, vlm_switch], ids=["mlx-lm", "mlx-vlm"])
def test_a_named_switch_layer_loads_as_the_formats_stack(toy, switch):
    n, k = 16, 64
    (codes, scales), _, _ = _matrix(EXPERTS * n, k)
    codes, scales = codes.reshape(EXPERTS, n, k), scales.reshape(EXPERTS, n)
    bias = mx.arange(EXPERTS * n).reshape(EXPERTS, n).astype(mx.float32)
    dense = switch.SwitchLinear(k, n, EXPERTS, bias=True)
    layer = dense.to_quantized(**_TOY)
    assert type(layer) is FormatSwitchLinear and layer.num_experts == EXPERTS
    assert (layer.mode, layer.bits, layer.group_size) == ("toy", 8, 1)
    assert toy.asked == [(k, n, 8, 1)]
    # The stack's axis is ahead of each tensor the layout gives.
    assert _kept(layer) == {
        "weight": ((EXPERTS, n, k), mx.uint8),
        "scales": ((EXPERTS, n), mx.float32),
        "bias": ((EXPERTS, n), mx.float32),
    }
    assert mx.array_equal(layer["bias"], dense["bias"])
    weights = {"weight": codes, "scales": scales, "bias": bias}
    layer.load_weights(list(weights.items()), strict=True)
    x, indices = _ints((5, 1, k)), mx.array([4, 0, 5, 5, 2], dtype=mx.uint32)
    assert np.array_equal(np.array(layer(x, indices)), _want(layer, x, indices))
    # Without a bias.
    layer = switch.SwitchLinear(k, n, EXPERTS, bias=False).to_quantized(**_TOY)
    assert sorted(layer.keys()) == ["scales", "weight"]


def test_mlx_s_modes_load_as_mlx_s_layers(toy):
    # Affine is a registered format that lays out no matrix: its layers are
    # MLX's to build.
    assert formats.get("affine").layout(64, 16, 4, 32) is None
    dense = nn.Linear(64, 16)
    for arguments, cls in (
        ({}, nn.QuantizedLinear),
        ({"group_size": 32, "bits": 4}, nn.QuantizedLinear),
        ({"group_size": 32, "bits": 4, "mode": "mxfp4"}, nn.QuantizedLinear),
    ):
        layer = dense.to_quantized(**arguments)
        assert type(layer) is cls
        assert layer.mode == arguments.get("mode", "affine")
    assert type(dense.to_quantized(32, 4, "affine")) is nn.QuantizedLinear
    layer = nn.Linear(64, 16, bias=False).to_quantized(
        mode="mxfp8", quantize_input=True
    )
    assert type(layer) is nn.QQLinear
    stack = lm_switch.SwitchLinear(64, 16, EXPERTS)
    layer = stack.to_quantized(16, 4, mode="nvfp4", global_scale=True)
    assert type(layer) is lm_switch.QuantizedSwitchLinear and "global_scale" in layer
    layer = vlm_switch.SwitchLinear(64, 16, EXPERTS).to_quantized(32, 4)
    assert type(layer) is vlm_switch.QuantizedSwitchLinear
    assert toy.asked == []
    # What the loaders read of a layer's to_quantized.
    for cls in _PATCHED:
        assert cls.to_quantized.__name__ == "to_quantized"
        assert inspect.signature(cls.to_quantized) == inspect.signature(
            cls.to_quantized.__wrapped__
        )
    assert "global_scale" in inspect.signature(stack.to_quantized).parameters


def test_a_mode_of_no_installed_format_names_the_missing_plugin(toy):
    refusal = re.escape(
        "no quant format is registered for mode 'oqel': install the plugin "
        "the checkpoint was written with"
    )
    lost = {"bits": 3, "group_size": 64, "mode": "oqel"}
    for dense in (
        nn.Linear(64, 16),
        lm_switch.SwitchLinear(64, 16, EXPERTS),
        vlm_switch.SwitchLinear(64, 16, EXPERTS),
    ):
        with pytest.raises(ValueError, match=refusal):
            dense.to_quantized(**lost)
    # As a load meets it.
    with pytest.raises(ValueError, match=refusal):
        nn.quantize(_Model(64, 16), class_predicate=_named({"toy": lost}))
    assert not format_load._mlx_runs("oqel") and format_load._mlx_runs("affine")


def test_a_matrix_the_format_does_not_lay_out_is_refused_in_words(toy):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "quant format 'toy' lays out no (16, 64) matrix of 4 bits in groups of 2"
        ),
    ):
        nn.Linear(64, 16).to_quantized(group_size=2, bits=4, mode="toy")
    assert toy.asked == [(64, 16, 4, 2)]
    # A layout is asked for with bits and a group size, as a checkpoint's
    # config gives them.
    for arguments in ({"group_size": 1}, {"bits": 8}):
        with pytest.raises(ValueError, match="quant format 'toy' lays out no"):
            nn.Linear(64, 16).to_quantized(mode="toy", **arguments)
    assert len(toy.asked) == 1


def test_a_formats_layer_is_built_from_its_bits_and_group_size_alone(toy):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "a layer of quant format 'toy' is built from its bits and "
            "group_size, not with quantize_input=True"
        ),
    ):
        nn.Linear(64, 16).to_quantized(**_TOY, quantize_input=True)
    with pytest.raises(ValueError, match="not with global_scale=True"):
        lm_switch.SwitchLinear(64, 16, EXPERTS).to_quantized(**_TOY, global_scale=True)
    # An argument stated as its default is none.
    assert type(nn.Linear(64, 16).to_quantized(1, 8, "toy", False)) is FormatLinear
    # Nor one the layer's own to_quantized does not take, as with MLX's.
    with pytest.raises(TypeError, match="layout_version"):
        nn.Linear(64, 16).to_quantized(**_TOY, layout_version=2)


def test_a_format_of_one_of_mlx_s_modes_takes_the_matrices_it_lays_out(
    toy, monkeypatch
):
    monkeypatch.setitem(formats._FORMATS, "mxfp4", _Loadable("mxfp4"))
    dense = nn.Linear(64, 16)
    layer = dense.to_quantized(group_size=1, bits=8, mode="mxfp4")
    assert type(layer) is FormatLinear and layer.mode == "mxfp4"
    layer = dense.to_quantized(group_size=32, bits=4, mode="mxfp4")
    assert type(layer) is nn.QuantizedLinear and layer.mode == "mxfp4"


def _class_name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def test_the_classes_whose_matrices_load_in_a_format(unpatched):
    # The classes whose to_quantized the patch wraps, and the layers MLX
    # builds of them: a calibration over a quantized checkpoint meets those.
    for cls in _PATCHED:
        built = cls(64, 16, EXPERTS) if cls is not nn.Linear else cls(64, 16)
        for loaded in (cls, type(built.to_quantized(32, 4))):
            assert format_load.loads_in_format(_class_name(loaded)), loaded
    assert format_load.loads_in_format("mlx.nn.layers.quantized.QuantizedLinear")
    # Not a family's own switch layer, nor MLA's, nor a class by its name
    # alone.
    for name in (
        "omlx.patches.deepseek_v4.switch_layers.SwitchLinear",
        "omlx.patches.glm_moe_dsa.switch_layers.QuantizedSwitchLinear",
        "mlx_lm.models.mla.MultiLinear",
        "mlx_lm.models.mla.QuantizedMultiLinear",
        "mlx_lm.models.switch_layers.SwitchGLU",
        "omlx.formats.linear.FormatLinear",
        "SwitchLinear",
        "",
    ):
        assert not format_load.loads_in_format(name), name


def test_the_patch_is_applied_once(toy):
    wrappers = [cls.to_quantized for cls in _PATCHED]
    assert all(wrapper._omlx_format_load for wrapper in wrappers)
    format_load.apply_format_load_patch()
    assert [cls.to_quantized for cls in _PATCHED] == wrappers
    assert not any(hasattr(wrapper.__wrapped__, "__wrapped__") for wrapper in wrappers)


def test_a_model_load_applies_the_patch(unpatched, tmp_path):
    assert not any(hasattr(cls.to_quantized, "__wrapped__") for cls in _PATCHED)
    maybe_apply_pre_load_patches(str(tmp_path))
    assert all(cls.to_quantized._omlx_format_load for cls in _PATCHED)


def test_the_patch_loads_the_plugins(unpatched, monkeypatch):
    loads = []
    monkeypatch.setattr(formats, "load", lambda: loads.append(True))
    format_load.apply_format_load_patch()
    assert loads == [True]


def _llama(path, **written):
    """A tiny llama checkpoint at ``path`` whose gate_proj of its one layer is
    the toy format's, with ``written`` as its config's record of formats;
    the reference model, and the layer's codes and scales."""
    config = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
    }
    name = "model.layers.0.mlp.gate_proj"
    mx.random.seed(0)
    reference = llama.Model(llama.ModelArgs.from_dict(config))
    (codes, scales), _, _ = _matrix(128, 64)
    scales = scales * 2.0**-10
    weights = dict(tree_flatten(reference.parameters()))
    weights[f"{name}.weight"], weights[f"{name}.scales"] = codes, scales
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    # A checkpoint names each layer of the format; the top-level mode is
    # MLX's, for the layers it finds by their scales.
    config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine", name: _TOY}
    if written:
        config["omlx_quant_formats"] = written
    (path / "config.json").write_text(json.dumps(config))
    return reference, codes, scales


def test_mlx_lm_loads_a_checkpoint_of_the_format(toy, tmp_path):
    reference, codes, scales = _llama(tmp_path)
    model, _ = load_model(tmp_path)
    mlp = model.model.layers[0].mlp
    assert type(mlp.gate_proj) is FormatLinear and mlp.gate_proj.mode == "toy"
    assert _kept(mlp.gate_proj) == {
        "weight": ((128, 64), mx.uint8),
        "scales": ((128,), mx.float32),
    }
    assert mx.array_equal(mlp.gate_proj["weight"], codes)
    assert type(mlp.up_proj) is nn.Linear and type(mlp.down_proj) is nn.Linear
    # The model computes what the same model does over the decoded weights.
    dense = toy.decode({"weight": codes, "scales": scales}, 8, 1)
    reference.model.layers[0].mlp.gate_proj.weight = dense
    tokens = mx.array([[1, 5, 9, 2, 30]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 32) and mx.array_equal(logits, reference(tokens))


def test_a_load_hands_a_format_its_record_before_a_layer_is_built(toy, tmp_path):
    record = {"layout": 2, "levels": [0.5, 1.5]}
    _llama(tmp_path, toy={"plugin": "omlx-toy 0.3.1", "contract": 1, "record": record})
    maybe_apply_pre_load_patches(str(tmp_path))
    assert toy.accepted == [(record, 0)]
    model, _ = load_model(tmp_path)
    assert type(model.model.layers[0].mlp.gate_proj) is FormatLinear
    assert toy.asked and len(toy.accepted) == 1


def test_a_format_refuses_a_checkpoint_in_its_own_words(toy, tmp_path):
    _llama(tmp_path, toy={"plugin": None, "contract": 1, "record": {"layout": 2}})
    toy.refusal = "written in layout 2; this plugin reads layout 1"
    with pytest.raises(ValueError) as refusal:
        maybe_apply_pre_load_patches(str(tmp_path))
    assert str(refusal.value) == toy.refusal
    assert toy.asked == []


def test_a_checkpoint_written_before_records_hands_an_empty_one(toy, tmp_path):
    _llama(tmp_path, toy={"plugin": "omlx-toy 0.3.0", "contract": 1})
    maybe_apply_pre_load_patches(str(tmp_path))
    assert toy.accepted == [({}, 0)]
    # A checkpoint of no format hands none.
    _llama(tmp_path)
    maybe_apply_pre_load_patches(str(tmp_path))
    assert len(toy.accepted) == 1


def test_a_format_that_is_not_installed_is_left_to_the_layers_load(toy, tmp_path):
    _llama(tmp_path, oqel={"plugin": "omlx-oqel 0.2.0", "contract": 1})
    maybe_apply_pre_load_patches(str(tmp_path))
    assert toy.accepted == []


def test_a_checkpoint_without_a_config_it_can_read_hands_no_record(toy, tmp_path):
    format_load.accept_checkpoint(str(tmp_path))
    (tmp_path / "config.json").write_text("{")
    format_load.accept_checkpoint(str(tmp_path))
    assert toy.accepted == []


def test_the_contract_page_s_accept_refuses_another_layout():
    (text,) = (
        text
        for language, text in _BLOCKS
        if language == "python" and "def accept" in text
    )
    scope = {}
    exec(textwrap.dedent(text), scope)
    for record in ({}, {"layout": 1}):
        assert scope["accept"](None, record) is None
    with pytest.raises(ValueError) as refusal:
        scope["accept"](None, {"layout": 2})
    assert str(refusal.value) == "written in layout 2; this plugin reads layout 1"
