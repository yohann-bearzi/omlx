# SPDX-License-Identifier: Apache-2.0
"""A checkpoint's layers of a quant format's mode load as the format's layers
(``omlx.patches.format_load``): through MLX's ``nn.quantize`` and a strict
load, as mlx-lm and mlx-vlm load a checkpoint.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import re
import textwrap
import threading

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama
from mlx_lm.models import switch_layers as lm_switch
from mlx_vlm.models import switch_layers as vlm_switch

from omlx import formats
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.patches import format_load
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from tests.test_formats_kernels import _BLOCKS, _PAGE, _matrix
from tests.test_formats_linear import EXPERTS, _ints, _ToyFormat, _want

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_PATCHED = (nn.Linear, lm_switch.SwitchLinear, vlm_switch.SwitchLinear)
_TOY = {"bits": 8, "group_size": 1, "mode": "toy"}


class _Loadable(_ToyFormat):
    """The toy format, with what its checkpoints keep of a matrix: a byte for
    each weight and a scale for each row. ``reads`` gives the object that
    reads a checkpoint of a record (the format itself, unless a test says
    otherwise)."""

    def __init__(self, mode="toy"):
        super().__init__()
        self.mode, self.asked = mode, []
        self.accepted, self.refusal = [], None
        self.reads = lambda record: self

    def accept(self, record):
        # What it was handed, and whether a layer had been built before.
        self.accepted.append((record, len(self.asked)))
        if self.refusal is not None:
            raise ValueError(self.refusal)
        return self.reads(record)

    def layout(self, k, n, bits, group_size):
        self.asked.append((k, n, bits, group_size))
        if bits != 8:
            return None
        return {"weight": ((n, k), mx.uint8), "scales": ((n,), mx.float32)}


@pytest.fixture
def unpatched(monkeypatch):
    """MLX's own ``to_quantized`` on the classes the patch wraps, and the
    libraries' own loaders, whatever an earlier test applied; the test's own
    wrappers are undone after it."""
    for cls in _PATCHED:
        original = cls.to_quantized
        monkeypatch.setattr(
            cls, "to_quantized", getattr(original, "__wrapped__", original)
        )
    for module, name in _loaders():
        loader = getattr(module, name)
        if getattr(loader, "_omlx_format_load", False):
            monkeypatch.setattr(module, name, loader.__wrapped__)
    # The patch wraps each module once, the first time it finds it: what it
    # wraps then, as a process starts.
    pending = {name for name, _ in format_load._LOADERS}
    monkeypatch.setattr(
        format_load, "_PENDING", pending | set(format_load._SWITCH_MODULES)
    )


def _loaders():
    """The loaders the patch wraps, of the libraries installed here."""
    found = []
    for module_name, name in format_load._LOADERS:
        with contextlib.suppress(ImportError):
            found.append((importlib.import_module(module_name), name))
    return found


@pytest.fixture
def toy(unpatched, monkeypatch):
    """The load patch, over a registry with the toy format."""
    formats.load()
    fmt = _Loadable()
    monkeypatch.setitem(formats._FORMATS, "toy", fmt)
    format_load.apply_format_load_patch()
    return fmt


@pytest.fixture
def by_hand(toy, tmp_path_factory):
    """The test builds layers by hand as the load of a checkpoint with no
    record would."""
    bare = tmp_path_factory.mktemp("bare")
    (bare / "config.json").write_text("{}")
    with format_load.loading(bare):
        yield toy


class _Model(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.toy = nn.Linear(k, n, bias=True)
        self.dense = nn.Linear(k, n, bias=False)


def load_model(path, **kwargs):
    """mlx-lm's loader as the patch leaves it: the function a module bound
    at its import is the library's own, which builds no layer of a format."""
    return mlx_lm.utils.load_model(path, **kwargs)


def _refusal(reason, path=None, written="omlx-toy 0.3.1", installed=None):
    """What a refusal of the toy format's layers says: of the checkpoint at
    ``path``, when it is known, written by ``written`` and with ``installed``
    (a format registered by hand, as the tests register it)."""
    if path is None:
        return f"quant format 'toy': {reason}"
    installed = installed or "a format registered by hand"
    return (
        f"quant format 'toy' of the checkpoint at {path} (written by {written}; "
        f"installed: {installed}): {reason}"
    )


def _named(config):
    """The loaders' predicate for the layers a quantization config names."""
    return lambda path, module: config.get(path, False)


def _kept(layer):
    return {name: (value.shape, value.dtype) for name, value in layer.items()}


def test_a_named_linear_loads_as_the_formats_layer(by_hand, toy):
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
def test_a_named_switch_layer_loads_as_the_formats_stack(by_hand, toy, switch):
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


def test_a_mode_of_no_installed_format_names_the_missing_plugin(toy, tmp_path):
    refusal = re.escape(
        "quant format 'oqel': no format of this mode is installed: install the "
        "plugin the checkpoint was written with"
    )
    lost = {"bits": 3, "group_size": 64, "mode": "oqel"}
    for dense in (
        nn.Linear(64, 16),
        lm_switch.SwitchLinear(64, 16, EXPERTS),
        vlm_switch.SwitchLinear(64, 16, EXPERTS),
    ):
        with pytest.raises(formats.FormatRefusedError, match=refusal):
            dense.to_quantized(**lost)
    # As a load meets it.
    with pytest.raises(formats.FormatRefusedError, match=refusal):
        nn.quantize(_Model(64, 16), class_predicate=_named({"toy": lost}))
    assert not formats._mlx_runs("oqel") and formats._mlx_runs("affine")
    # A checkpoint's load names the plugin that wrote it, to install.
    (tmp_path / "config.json").write_text(
        json.dumps({"omlx_quant_formats": {"oqel": {"plugin": "omlx-oqel 1.0.0"}}})
    )
    with (
        format_load.loading(tmp_path),
        pytest.raises(formats.FormatRefusedError) as no,
    ):
        nn.Linear(64, 16).to_quantized(**lost)
    assert str(no.value) == (
        f"quant format 'oqel' of the checkpoint at {tmp_path} (written by "
        "omlx-oqel 1.0.0; installed: none): no format of this mode is installed: "
        "install omlx-oqel 1.0.0"
    )
    assert (no.value.mode, no.value.path) == ("oqel", tmp_path)
    assert (no.value.written, no.value.installed) == ("omlx-oqel 1.0.0", "none")
    quoted = str(no.value).replace(str(tmp_path), "/models/my-model")
    quoted = quoted.replace("omlx-oqel 1.0.0", "my-package 0.1.0")
    assert quoted.replace("'oqel'", "'myformat'") in " ".join(_PAGE.split())
    # A checkpoint that records no plugin, or a format registered by hand.
    for kept in ({}, {"omlx_quant_formats": {"oqel": {"plugin": None}}}):
        (tmp_path / "config.json").write_text(json.dumps(kept))
        with (
            format_load.loading(tmp_path),
            pytest.raises(formats.FormatRefusedError) as no,
        ):
            nn.Linear(64, 16).to_quantized(**lost)
        assert no.value.reason == (
            "no format of this mode is installed: install the plugin the "
            "checkpoint was written with"
        )


def test_a_plugin_that_did_not_load_is_named_by_its_checkpoint(toy, tmp_path):
    # Installed, and skipped when the registry loaded (here for the revision
    # of the contract it is written for).
    skipped = (
        "omlx-oqel 2.0.0 is installed but did not load: quant format 'oqel' is "
        "written for revision 2 of the format contract; this oMLX runs revision "
        "1: update oMLX"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"omlx_quant_formats": {"oqel": {"plugin": "omlx-oqel 2.0.0"}}})
    )
    formats._SKIPPED["oqel"] = skipped
    try:
        with (
            format_load.loading(tmp_path),
            pytest.raises(formats.FormatRefusedError) as no,
        ):
            nn.Linear(64, 16).to_quantized(bits=3, group_size=64, mode="oqel")
    finally:
        del formats._SKIPPED["oqel"]
    assert no.value.reason == skipped and no.value.installed == "none"
    quoted = str(no.value).replace(str(tmp_path), "/models/my-model")
    quoted = quoted.replace("omlx-oqel 2.0.0", "my-package 0.2.0")
    assert quoted.replace("'oqel'", "'myformat'") in " ".join(_PAGE.split())


def test_a_matrix_the_format_does_not_lay_out_is_refused_in_words(by_hand, toy):
    with pytest.raises(formats.FormatRefusedError) as refusal:
        nn.Linear(64, 16).to_quantized(group_size=2, bits=4, mode="toy")
    assert str(refusal.value) == _refusal(
        "it lays out no (16, 64) matrix of 4 bits in groups of 2",
        refusal.value.path,
        written="an unknown writer",
    )
    assert toy.asked == [(64, 16, 4, 2)]
    # A layout is asked for with bits and a group size, as a checkpoint's
    # config gives them.
    for arguments in ({"group_size": 1}, {"bits": 8}):
        with pytest.raises(formats.FormatRefusedError, match="it lays out no"):
            nn.Linear(64, 16).to_quantized(mode="toy", **arguments)
    assert len(toy.asked) == 1


def test_a_formats_layer_is_built_from_its_bits_and_group_size_alone(by_hand, toy):
    with pytest.raises(
        formats.FormatRefusedError,
        match=re.escape(
            "a layer is built from its bits and group_size, not with "
            "quantize_input=True"
        ),
    ):
        nn.Linear(64, 16).to_quantized(**_TOY, quantize_input=True)
    with pytest.raises(formats.FormatRefusedError, match="not with global_scale=True"):
        lm_switch.SwitchLinear(64, 16, EXPERTS).to_quantized(**_TOY, global_scale=True)
    # An argument stated as its default is none.
    assert type(nn.Linear(64, 16).to_quantized(1, 8, "toy", False)) is FormatLinear
    # Nor one the layer's own to_quantized does not take, as with MLX's.
    with pytest.raises(TypeError, match="layout_version"):
        nn.Linear(64, 16).to_quantized(**_TOY, layout_version=2)


def test_a_format_of_one_of_mlx_s_modes_takes_the_matrices_it_lays_out(
    by_hand, monkeypatch
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
    # The classes whose to_quantized the patch wraps; not the layers MLX
    # builds of them (oQ's walk gives no weight, so it builds none).
    for cls in _PATCHED:
        assert format_load.loads_in_format(_class_name(cls)), cls
        built = cls(64, 16, EXPERTS) if cls is not nn.Linear else cls(64, 16)
        quantized = _class_name(type(built.to_quantized(32, 4)))
        assert not format_load.loads_in_format(quantized), quantized
    # Not a family's own switch layer, nor MLA's, nor a class by its name
    # alone.
    for name in (
        "mlx.nn.layers.quantized.QuantizedLinear",
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
    wrappers += [getattr(module, name) for module, name in _loaders()]
    assert len(wrappers) == len(_PATCHED) + len(format_load._LOADERS)
    assert all(wrapper._omlx_format_load for wrapper in wrappers)
    format_load.apply_format_load_patch()
    again = [cls.to_quantized for cls in _PATCHED]
    assert again + [getattr(module, name) for module, name in _loaders()] == wrappers
    assert not any(
        getattr(wrapper.__wrapped__, "_omlx_format_load", False) for wrapper in wrappers
    )
    # A module found again once wrapped (its import ending on one thread as
    # the patch is applied on another) is left as it is.
    for module, name in _loaders():
        format_load._wrap(module)
        assert getattr(module, name) in wrappers
    # What a loader's callers read of it.
    for module, name in _loaders():
        loader = getattr(module, name)
        assert inspect.signature(loader) == inspect.signature(loader.__wrapped__)


def test_a_model_load_applies_the_patch(unpatched, tmp_path):
    loaders = [getattr(module, name) for module, name in _loaders()]
    assert not any(hasattr(cls.to_quantized, "__wrapped__") for cls in _PATCHED)
    assert not any(getattr(loader, "_omlx_format_load", False) for loader in loaders)
    maybe_apply_pre_load_patches(str(tmp_path))
    assert all(cls.to_quantized._omlx_format_load for cls in _PATCHED)
    for (module, name), loader in zip(_loaders(), loaders, strict=True):
        assert getattr(module, name).__wrapped__ is loader


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


class _Levels(_Loadable):
    """The toy format as a checkpoint's record configures it: its levels,
    the first of which scales what it decodes."""

    def __init__(self, levels):
        super().__init__()
        self.levels = tuple(levels)

    def decode(self, tensors, bits, group_size):
        return super().decode(tensors, bits, group_size) * self.levels[0]


def _written(record=None, plugin="omlx-toy 0.3.1"):
    """What oQ stores of the toy format in a checkpoint's config."""
    entry = {"plugin": plugin, "contract": 1}
    if record is not None:
        entry["record"] = record
    return {"toy": entry}


def test_a_checkpoint_s_layers_keep_what_its_format_reads_them_with(toy, tmp_path):
    record = {"layout": 1, "levels": [0.5, 1.5]}
    toy.reads = lambda record: _Levels(record["levels"])
    _llama(tmp_path, **_written(record))
    model, _ = load_model(tmp_path)
    layer = model.model.layers[0].mlp.gate_proj
    assert type(layer) is FormatLinear and layer.mode == "toy"
    # The record was handed over once, when the registered format had said it
    # lays the matrix out.
    assert toy.accepted == [(record, 1)]
    assert type(layer.fmt) is _Levels and layer.fmt.levels == (0.5, 1.5)
    # The object reads the layer: it lays the matrix out, its decode and its
    # face are the layer's; nothing is kept on the registered format.
    assert layer.fmt.asked == toy.asked == [(64, 128, 8, 1)]
    assert formats.get("toy") is toy and not hasattr(toy, "levels")
    tensors = {name: layer[name] for name in ("weight", "scales")}
    assert mx.array_equal(layer._decoded(), toy.decode(tensors, 8, 1) * 0.5)
    asked = []
    layer.fmt.dot = lambda layer, k, n: asked.append((k, n))
    assert formats.dot(layer, 64, 128) is None and asked == [(64, 128)]
    # Two checkpoints of the format, two records: each keeps its own.
    other = tmp_path / "other"
    other.mkdir()
    _llama(other, **_written({"layout": 1, "levels": [2.0]}))
    second, _ = load_model(other)
    kept = second.model.layers[0].mlp.gate_proj.fmt
    assert kept.levels == (2.0,) and layer.fmt.levels == (0.5, 1.5)


def test_a_checkpoint_written_without_a_record_hands_an_empty_one(toy, tmp_path):
    for written in (_written(), {}):
        # Written before formats had records, or naming the format in its
        # quantization alone.
        _llama(tmp_path, **written)
        model, _ = load_model(tmp_path)
        assert model.model.layers[0].mlp.gate_proj.fmt is toy
    assert toy.accepted == [({}, 1), ({}, 2)]


def test_a_format_refuses_a_checkpoint_in_its_own_words(toy, tmp_path):
    _llama(tmp_path, **_written({"layout": 2}, plugin=None))
    toy.refusal = "written in layout 2; this plugin reads layout 1"
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path)
    # The format's words, with what oMLX knows of the checkpoint.
    assert str(refusal.value) == _refusal(
        toy.refusal, tmp_path, written="a format registered by hand"
    )
    assert type(refusal.value.__cause__) is ValueError
    assert str(refusal.value.__cause__) == refusal.value.reason == toy.refusal
    assert len(toy.asked) == 1
    # Named by their distributions when plugins'.
    _llama(tmp_path, **_written({"layout": 2}, plugin="my-package 0.2.0"))
    formats._ORIGINS["toy"] = "my-package 0.1.0"
    try:
        with pytest.raises(formats.FormatRefusedError) as refusal:
            load_model(tmp_path)
    finally:
        del formats._ORIGINS["toy"]
    said = str(refusal.value).replace(str(tmp_path), "/models/my-model")
    assert said.replace("'toy'", "'myformat'") in " ".join(_PAGE.split())
    # A record that names no writer.
    _llama(tmp_path, toy={"record": {"layout": 2}})
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path)
    assert refusal.value.written == "an unknown writer"


_ENTRY = "['omlx_quant_formats']['toy']"


def _kept_record(record):
    entry = {"plugin": "omlx-toy 0.3.1", "record": record}
    return {"omlx_quant_formats": {"toy": entry}}


@pytest.mark.parametrize(
    "config, where, kind",
    [
        ([], "", "an array"),
        ({"omlx_quant_formats": []}, "['omlx_quant_formats']", "an array"),
        ({"omlx_quant_formats": {"toy": "v1"}}, _ENTRY, "a string"),
        (_kept_record(None), f"{_ENTRY}['record']", "null"),
        (_kept_record([1]), f"{_ENTRY}['record']", "an array"),
        (_kept_record(2.5), f"{_ENTRY}['record']", "a number"),
    ],
    ids=["config", "formats", "entry", "null", "array", "number"],
)
def test_a_record_that_is_not_an_object_is_refused(toy, tmp_path, config, where, kind):
    (tmp_path / "config.json").write_text(json.dumps(config))
    with format_load.loading(tmp_path), pytest.raises(formats.FormatRefusedError) as no:
        nn.Linear(64, 128).to_quantized(**_TOY)
    assert str(no.value) == _refusal(
        f"config.json{where} is {kind}, where oQ writes an object: the "
        "checkpoint's config is damaged",
        tmp_path,
        written="omlx-toy 0.3.1" if "record" in where else "an unknown writer",
    )
    assert toy.accepted == []
    if kind == "null":
        quoted = str(no.value).replace(str(tmp_path), "/models/my-model")
        quoted = quoted.replace("omlx-toy 0.3.1", "my-package 0.1.0")
        quoted = quoted.replace("a format registered by hand", "my-package 0.1.0")
        assert quoted.replace("'toy'", "'myformat'") in " ".join(_PAGE.split())


class _Other(_Loadable):
    tensors = ("scales", "levels")


@pytest.mark.parametrize(
    "returned",
    ["a string", "another mode", "other tensors", "another contract"],
)
def test_what_accept_returns_is_a_format_of_the_same_kind(
    toy, tmp_path, monkeypatch, returned
):
    returned = {
        "a string": "toy",
        "another mode": _Loadable("other"),
        "other tensors": _Other(),
        "another contract": _Loadable(),
    }[returned]
    if isinstance(returned, _Loadable) and type(returned) is _Loadable:
        returned.contract = 2 if returned.mode == "toy" else returned.contract
    _llama(tmp_path, **_written({"layout": 1}))
    toy.reads = lambda record: returned
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path)
    said = (
        f"accept returned {returned!r}, not a format of its mode, tensors and "
        "contract: report it to "
    )
    assert str(refusal.value) == _refusal(said + "the plugin", tmp_path)
    # Named by its distribution when a plugin's.
    monkeypatch.setitem(formats._ORIGINS, "toy", "my-package 0.1.0")
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path)
    assert str(refusal.value) == _refusal(
        said + "my-package 0.1.0", tmp_path, installed="my-package 0.1.0"
    )
    if returned == "toy":
        quoted = str(refusal.value).replace(str(tmp_path), "/models/my-model")
        quoted = quoted.replace("omlx-toy 0.3.1", "my-package 0.1.0")
        assert quoted.replace("'toy'", "'myformat'") in " ".join(_PAGE.split())


def test_accept_returning_none_reads_with_the_registered_format(toy, tmp_path):
    # As the guide's example of revision 1 wrote accept.
    toy.reads = lambda record: None
    _llama(tmp_path, **_written({"layout": 1}))
    model, _ = load_model(tmp_path)
    assert model.model.layers[0].mlp.gate_proj.fmt is toy


def test_a_format_that_configures_itself_to_accept_is_refused(toy, tmp_path):
    def configure(record):
        toy.levels = tuple(record["levels"])
        return toy

    toy.reads = configure
    _llama(tmp_path, **_written({"levels": [1.0]}))
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path)
    assert str(refusal.value) == _refusal(
        "accept changed the registered format (levels) to accept the checkpoint: "
        "accept returns a format configured from the record instead; report it "
        "to the plugin",
        tmp_path,
    )
    quoted = str(refusal.value).replace(str(tmp_path), "/models/my-model")
    for said in ("the plugin", "omlx-toy 0.3.1", "a format registered by hand"):
        quoted = quoted.replace(said, "my-package 0.1.0")
    assert quoted.replace("'toy'", "'myformat'") in " ".join(_PAGE.split())


class _Wide(_Loadable):
    """A layout of the record's: two scales a row."""

    def layout(self, k, n, bits, group_size):
        self.asked.append((k, n, bits, group_size))
        return {"weight": ((n, k), mx.uint8), "scales": ((n, 2), mx.float32)}


def test_the_checkpoint_s_format_lays_its_matrices_out(toy, tmp_path):
    wide = _Wide()
    toy.reads = lambda record: wide if record.get("wide") else toy
    _llama(tmp_path, **_written({"wide": True}))
    with format_load.loading(tmp_path):
        layer = nn.Linear(64, 128).to_quantized(**_TOY)
    # The registered format said it lays such a matrix out; the checkpoint's
    # laid it out.
    assert toy.asked == [(64, 128, 8, 1)] and wide.asked == [(64, 128, 8, 1)]
    assert layer.fmt is wide and _kept(layer)["scales"] == ((128, 2), mx.float32)
    # One the registered format lays out none of is refused before the record
    # is read.
    with (
        format_load.loading(tmp_path),
        pytest.raises(ValueError, match="lays out no"),
    ):
        nn.Linear(64, 128).to_quantized(group_size=1, bits=4, mode="toy")
    assert len(toy.accepted) == 1


def test_a_layer_built_in_a_load_and_not_in_its_model_is_refused(toy, tmp_path):
    import types

    _llama(tmp_path, **_written({"levels": [1.0]}))
    toy.reads = lambda record: _Levels(record["levels"])
    load = mlx_lm.utils.load_model

    def with_a_stranger(path, **kwargs):
        # Another checkpoint's layer, quantized by hand inside this one's load
        # (as a loader bound before the patch would): it reads this one's
        # record.
        stranger = nn.Linear(64, 128).to_quantized(**_TOY)
        assert stranger.fmt.levels == (1.0,)
        return load(path, **kwargs)

    module = types.ModuleType("stranger")
    module.load_model = with_a_stranger
    format_load._wrap_loader(module, "load_model")
    with pytest.raises(formats.FormatRefusedError) as refusal:
        module.load_model(tmp_path)
    assert str(refusal.value) == _refusal(
        "1 layers were built while the checkpoint loaded, and are not in its "
        "model: they were another checkpoint's, read with this one's record; load "
        "each checkpoint with the libraries' load_model",
        tmp_path,
    )


def _rewritten(path, change):
    """The checkpoint at ``path`` with its weights changed by ``change``."""
    weights = mx.load(str(path / "model.safetensors"))
    change(weights)
    mx.save_safetensors(str(path / "model.safetensors"), weights)


_GATE = "model.layers.0.mlp.gate_proj"


def _renamed(weights):
    weights[f"{_GATE}.scalez"] = weights.pop(f"{_GATE}.scales")


def _float16(weights):
    weights[f"{_GATE}.scales"] = weights[f"{_GATE}.scales"].astype(mx.float16)


def _short(weights):
    weights[f"{_GATE}.scales"] = weights[f"{_GATE}.scales"][:64]


@pytest.mark.parametrize(
    "change, strict, reason",
    [
        (
            _renamed,
            False,
            f"the load filled {_GATE} but its scales: the checkpoint holds no such "
            "tensor, or holds it under another name",
        ),
        (
            _float16,
            True,
            f"{_GATE}.scales is (128,) {mx.float16}, where the format lays out "
            f"(128,) {mx.float32}",
        ),
        (
            _short,
            False,
            f"{_GATE}.scales is (64,) {mx.float32}, where the format lays out "
            f"(128,) {mx.float32}",
        ),
    ],
    ids=["a key renamed", "another dtype", "another shape"],
)
def test_a_load_that_fills_a_layer_otherwise_than_its_layout_is_refused(
    toy, tmp_path, change, strict, reason
):
    _llama(tmp_path, **_written({}))
    _rewritten(tmp_path, change)
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path, strict=strict)
    assert str(refusal.value) == _refusal(reason, tmp_path)
    if change is _renamed:
        quoted = str(refusal.value).replace(str(tmp_path), "/models/my-model")
        for said in ("omlx-toy 0.3.1", "a format registered by hand"):
            quoted = quoted.replace(said, "my-package 0.1.0")
        assert quoted.replace("'toy'", "'myformat'") in " ".join(_PAGE.split())


def test_a_format_entry_that_binds_no_layer_is_refused(toy, tmp_path):
    _llama(tmp_path, **_written({}))
    config = json.loads((tmp_path / "config.json").read_text())
    # A path that is no module of the model, as a family's loader that
    # renames its modules leaves it.
    config["quantization"]["model.layers.0.mlp.gate"] = _TOY
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path, strict=False)
    assert str(refusal.value) == _refusal(
        "its quantization config names model.layers.0.mlp.gate, and the load "
        "built no layer of the format there: the model has no such module, or "
        "its family's loader renames it",
        tmp_path,
    )


def test_a_loader_wrapped_over_the_wrapped_one_is_one_load(toy, tmp_path):
    import types

    # A family's loader that calls the libraries' (wrapped) loader, wrapped
    # in its turn: the layers built belong to the one load of the checkpoint.
    _llama(tmp_path, **_written({}))
    module = types.ModuleType("family")
    module.load_model = lambda path, **kwargs: load_model(path, **kwargs)
    format_load._wrap_loader(module, "load_model")
    model, _ = module.load_model(tmp_path, strict=False)
    assert type(model.model.layers[0].mlp.gate_proj) is FormatLinear


def test_a_partial_load_leaves_the_layers_it_does_not_hold(toy, tmp_path):
    # A pipeline rank's or a prefill helper's files hold some layers only:
    # the others keep their placeholders until the loader drops them.
    _llama(tmp_path, **_written({}))

    def partial(weights):
        del weights[f"{_GATE}.weight"], weights[f"{_GATE}.scales"]

    _rewritten(tmp_path, partial)
    # An entry of one of MLX's modes is MLX's to bind, or not.
    config = json.loads((tmp_path / "config.json").read_text())
    config["quantization"]["model.layers.0.mlp.nothing"] = {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model, _ = load_model(tmp_path, strict=False)
    layer = model.model.layers[0].mlp.gate_proj
    assert type(layer) is FormatLinear and not mx.any(layer["scales"]).item()
    # A stock checkpoint loads as it did.
    stock = tmp_path / "stock"
    stock.mkdir()
    del config["quantization"], config["omlx_quant_formats"]
    (stock / "config.json").write_text(json.dumps(config))
    weights = mx.load(str(tmp_path / "model.safetensors"))
    weights[f"{_GATE}.weight"] = mx.zeros((128, 64))
    mx.save_safetensors(str(stock / "model.safetensors"), weights)
    model, _ = load_model(stock, strict=True)
    assert type(model.model.layers[0].mlp.gate_proj) is nn.Linear


def test_a_format_s_layer_is_built_in_a_checkpoint_s_load_only(unpatched, monkeypatch):
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _Loadable())
    format_load.apply_format_load_patch()
    with pytest.raises(formats.FormatRefusedError) as refusal:
        nn.Linear(64, 16).to_quantized(**_TOY)
    assert str(refusal.value) == _refusal(
        "a layer is built outside a checkpoint's load, which gives it the record "
        "to be read with: load the checkpoint with mlx-lm's, mlx-vlm's or "
        "mlx-embeddings' load_model"
    )
    assert refusal.value.path is None
    # MLX's layers need none.
    assert type(nn.Linear(64, 16).to_quantized(32, 4)) is nn.QuantizedLinear


def test_a_checkpoint_is_accepted_once_for_all_its_layers(toy, tmp_path):
    _llama(tmp_path, **_written({"levels": [1.0]}))
    toy.reads = lambda record: _Levels(record["levels"])
    with format_load.loading(tmp_path):
        first = nn.Linear(64, 128).to_quantized(**_TOY)
        second = lm_switch.SwitchLinear(64, 128, EXPERTS).to_quantized(**_TOY)
    assert first.fmt is second.fmt and first.fmt.levels == (1.0,)
    assert toy.accepted == [({"levels": [1.0]}, 1)]


def test_a_load_reads_its_own_checkpoint_on_its_own_thread(toy, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for path, levels in ((first, [1.0]), (second, [2.0])):
        path.mkdir()
        _llama(path, **_written({"levels": levels}))
    toy.reads = lambda record: _Levels(record["levels"])
    built, loaded = threading.Event(), threading.Event()
    seen = {}

    def other_thread():
        # A thread loading the first checkpoint, while the second loads here.
        with format_load.loading(first):
            built.set()
            loaded.wait(60)
            seen["late"] = nn.Linear(64, 128).to_quantized(**_TOY).fmt.levels

    worker = threading.Thread(target=other_thread)
    worker.start()
    assert built.wait(60)
    # A load is active on that thread only.
    assert formats._LOADING.get() is None
    model, _ = load_model(second)
    loaded.set()
    worker.join(60)
    assert model.model.layers[0].mlp.gate_proj.fmt.levels == (2.0,)
    assert seen == {"late": (1.0,)}
    # A load nested in another reads its own, and the outer one its own after.
    with format_load.loading(first):
        model, _ = load_model(second)
        after = nn.Linear(64, 128).to_quantized(**_TOY)
    assert model.model.layers[0].mlp.gate_proj.fmt.levels == (2.0,)
    assert after.fmt.levels == (1.0,)


def test_the_contract_page_s_accept_refuses_another_layout():
    (text,) = (
        text
        for language, text in _BLOCKS
        if language == "python" and "def accept" in text and "raise" in text
    )
    scope = {}
    exec(textwrap.dedent(text), scope)
    fmt = object()
    for record in ({}, {"layout": 1}):
        assert scope["accept"](fmt, record) is fmt
    with pytest.raises(ValueError) as refusal:
        scope["accept"](fmt, {"layout": 2})
    assert str(refusal.value) == "written in layout 2; this plugin reads layout 1"


# Each way oMLX loads a checkpoint, through the libraries' real loaders: a
# tiny Llama checkpoint with a layer of the toy format, whose record names
# the levels its layers are read with. mlx-vlm serves Llama too.


def _tiny(path, levels):
    """A tiny llama checkpoint at ``path`` with the toy format's layer, as
    ``_llama`` writes it, the record ``{"levels": levels}``, and a tokenizer
    with a chat template."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    _llama(path, **_written({"levels": levels}))
    vocab = {f"t{i}": i for i in range(32)}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="t0")), unk_token="t0"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def levels(toy):
    """The toy format read as each checkpoint's record configures it: the
    levels each load handed over, in order."""
    toy.reads = lambda record: _Levels(record["levels"])
    return lambda: [tuple(record["levels"]) for record, _ in toy.accepted]


def _gate_proj(model):
    """The toy layer of a tiny Llama, as mlx-lm or mlx-vlm built it."""
    lm = getattr(model, "language_model", model)
    layer = lm.model.layers[0].mlp.gate_proj
    assert type(layer) is FormatLinear
    return layer


def test_the_llm_engine_and_a_specprefill_draft_read_their_records(levels, tmp_path):
    from omlx.utils.model_loading import lm_load_compat, load_text_model

    model, _ = load_text_model(str(_tiny(tmp_path / "target", [1.0])))
    # SpecPrefill's draft (engine/batched.py, engine/vlm.py).
    draft, _ = lm_load_compat(str(_tiny(tmp_path / "draft", [2.0])))
    assert _gate_proj(model).fmt.levels == (1.0,)
    assert _gate_proj(draft).fmt.levels == (2.0,)
    assert levels() == [(1.0,), (2.0,)]


def test_a_hub_id_reads_the_record_of_the_files_it_resolves_to(
    levels, tmp_path, monkeypatch
):
    import mlx_lm.utils

    from omlx.utils.model_loading import lm_load_compat

    path = _tiny(tmp_path / "snapshot", [4.0])
    monkeypatch.setattr(mlx_lm.utils, "snapshot_download", lambda *a, **k: str(path))
    model, _ = lm_load_compat("someone/tiny-toy")
    assert _gate_proj(model).fmt.levels == (4.0,)


def test_a_model_directory_under_home_loads_with_its_record(
    levels, tmp_path, monkeypatch
):
    from omlx.model_discovery import discover_models_from_dirs
    from omlx.settings import ModelSettings
    from omlx.utils.model_loading import load_text_model

    _tiny(tmp_path / "home" / "models" / "tiny", [5.0])
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # The server's model directories, as its settings give them, and what
    # the pool discovers there.
    dirs = ModelSettings(model_dirs=["~/models"]).get_model_dirs(tmp_path)
    (found,) = discover_models_from_dirs(dirs).values()
    model, _ = load_text_model(found.model_path)
    assert _gate_proj(model).fmt.levels == (5.0,)


def test_the_vlm_engine_reads_its_record(levels, tmp_path):
    from mlx_vlm.utils import load_model as vlm_load_model

    from omlx.engine.vlm import _load_cohere2_moe_text_model

    model = vlm_load_model(_tiny(tmp_path / "vlm", [6.0]), lazy=True)
    assert _gate_proj(model).fmt.levels == (6.0,)
    # The engine's text-only load of Cohere2 MoE goes through it too.
    model, _ = _load_cohere2_moe_text_model(str(_tiny(tmp_path / "cohere", [7.0])))
    assert _gate_proj(model).fmt.levels == (7.0,)


@pytest.mark.parametrize("drafter", ["DFlash", "VLM-MTP"])
def test_a_drafter_reads_its_record(levels, tmp_path, drafter):
    path = str(_tiny(tmp_path / "drafter", [8.0]))
    if drafter == "DFlash":
        from omlx.speculative.dflash import load_dflash_drafter

        # The tiny checkpoint is no DFlash drafter: it fails once loaded,
        # where the drafter meets its target (None here).
        with pytest.raises(TypeError):
            load_dflash_drafter(path, target_model=None)
    else:
        from omlx.speculative.vlm_mtp import load_vlm_mtp_drafter

        # Nor an MTP one: it is logged and dropped once loaded.
        assert load_vlm_mtp_drafter(path) is None
    assert levels() == [(8.0,)]


def _registered(monkeypatch):
    """The toy format, registered with no patch applied yet, reading each
    checkpoint with its record's levels."""
    formats.load()
    toy = _Loadable()
    toy.reads = lambda record: _Levels(record["levels"])
    monkeypatch.setitem(formats._FORMATS, "toy", toy)
    return toy


def test_a_causal_lm_reranker_installs_the_patch_and_reads_its_record(
    unpatched, tmp_path, monkeypatch
):
    from omlx.models.reranker import MLXRerankerModel

    _registered(monkeypatch)
    path = _tiny(tmp_path / "tiny-reranker", [9.0])
    config = json.loads((path / "config.json").read_text())
    config["architectures"] = ["Qwen3ForCausalLM"]
    (path / "config.json").write_text(json.dumps(config))
    reranker = MLXRerankerModel(str(path))
    reranker.load()
    assert _gate_proj(reranker.model).fmt.levels == (9.0,)


def test_an_embedding_model_installs_the_patch_and_reads_its_record(
    unpatched, tmp_path, monkeypatch
):
    from omlx.models import embedding

    toy = _registered(monkeypatch)
    # Through mlx-vlm's encoder loader, as its embedding models load.
    monkeypatch.setitem(embedding._MLX_VLM_EMBEDDING_CONTEXT, "llama", 512)
    path = _tiny(tmp_path / "embedding", [10.0])
    try:
        embedding.MLXEmbeddingModel(str(path)).load()
    except Exception as error:  # a Llama is no embedding model: what follows
        failed = error  # its load fails
    else:
        failed = None
    accepted = [tuple(record["levels"]) for record, _ in toy.accepted]
    assert accepted == [(10.0,)], failed


def test_the_cluster_s_loads_read_their_records(levels, tmp_path):
    from omlx.cluster.progressive_loading import progressive_sharded_load
    from omlx.patches.mlx_lm_sharded_load import _local_sharded_load

    model, _ = _local_sharded_load(str(_tiny(tmp_path / "rank", [11.0])))
    assert _gate_proj(model).fmt.levels == (11.0,)
    # Whether a tiny Llama shards or not: its layers were read.
    with contextlib.suppress(ValueError):
        progressive_sharded_load(str(_tiny(tmp_path / "progressive", [12.0])))
    assert levels()[1] == (12.0,)


@pytest.mark.parametrize("adapter", ["qwen3_5", "minimax_m3", "glm5_next", "qwen4_exp"])
def test_a_prefill_helper_reads_its_record(levels, tmp_path, adapter, monkeypatch):
    module = importlib.import_module(f"omlx.prefill_helper.adapters.{adapter}")
    # The kernel patches a helper applies after its load are process-wide:
    # the tests after this one run without them.
    for name in [name for name in vars(module) if name.startswith("apply_")]:
        monkeypatch.setattr(module, name, lambda *args, **kwargs: False)
    (cls,) = (
        value
        for name, value in vars(module).items()
        if name.endswith("Adapter")
        and getattr(value, "__module__", "") == module.__name__
    )
    # A Llama is not the family's: what follows its load fails.
    with contextlib.suppress(Exception):
        cls().load_first(str(_tiny(tmp_path / "helper", [13.0])), 1)
    assert levels() == [(13.0,)]


def test_oq_s_calibration_reads_the_record_of_its_source(levels, tmp_path):
    from omlx import oq

    path = _tiny(tmp_path / "source", [14.0])
    config = json.loads((path / "config.json").read_text())
    model, _ = oq._calibration_model(str(path), config)
    assert _gate_proj(model).fmt.levels == (14.0,)


def test_a_family_loader_over_the_patch_keeps_its_place(levels, tmp_path, monkeypatch):
    import mlx_lm.utils

    from omlx.patches.deepseek_v4 import utils_patch
    from omlx.patches.k2_horizon import checkpoint

    # DeepSeek V4's loader goes over the wrapped one, as the pre-load
    # patches apply it after this patch.
    monkeypatch.setattr(mlx_lm.utils, "load_model", mlx_lm.utils.load_model)
    wrapped = mlx_lm.utils.load_model
    mlx_lm.utils.load_model = utils_patch._build_patched_load_model()
    model, _ = mlx_lm.utils.load_model(_tiny(tmp_path / "llama", [15.0]))
    assert _gate_proj(model).fmt.levels == (15.0,)
    # It builds a DeepSeek V4 checkpoint's layers itself, outside any load:
    # one of a format is refused, on every load (the patch never wraps the
    # loader again).
    v4 = _tiny(tmp_path / "v4", [16.0])
    config = json.loads((v4 / "config.json").read_text())
    (v4 / "config.json").write_text(json.dumps({**config, "model_type": "deepseek_v4"}))
    for _ in range(2):
        format_load.apply_format_load_patch()
        assert not getattr(mlx_lm.utils.load_model, "_omlx_format_load", False)
        with pytest.raises(ValueError, match="built outside a checkpoint's load"):
            mlx_lm.utils.load_model(
                v4, get_model_classes=lambda config: (llama.Model, llama.ModelArgs)
            )
    # K2 Horizon's loads a directory of links to the checkpoint's files.
    mlx_lm.utils.load_model = wrapped
    checkpoint.apply_checkpoint_patch()
    k2 = _tiny(tmp_path / "k2", [17.0])
    config = json.loads((k2 / "config.json").read_text())
    (k2 / "config.json").write_text(json.dumps({**config, "model_type": "k2_horizon"}))
    model, _ = mlx_lm.utils.load_model(
        k2, get_model_classes=lambda config: (llama.Model, llama.ModelArgs)
    )
    assert _gate_proj(model).fmt.levels == (17.0,)


def test_the_patch_imports_none_of_the_libraries_and_wraps_them_on_import():
    import subprocess
    import sys

    code = textwrap.dedent("""
        import json, sys
        from omlx.patches import format_load
        libraries = ("mlx_lm", "mlx_vlm", "mlx_embeddings")
        before = [name for name in libraries if name in sys.modules]
        format_load.apply_format_load_patch()
        after = [name for name in libraries if name in sys.modules]
        import mlx_lm.utils, mlx_vlm.utils, mlx_vlm.embedding_loader
        import mlx_embeddings.utils, mlx_lm.models.switch_layers
        import mlx_vlm.models.switch_layers
        marked = lambda f: getattr(f, "_omlx_format_load", False)
        wrapped = [
            marked(getattr(sys.modules[module], name))
            for module, name in format_load._LOADERS
        ] + [
            marked(sys.modules[module].SwitchLinear.to_quantized)
            for module in format_load._SWITCH_MODULES
        ]
        print(json.dumps({"before": before, "after": after, "wrapped": wrapped}))
        """)
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    seen = json.loads(run.stdout.splitlines()[-1])
    assert seen["after"] == seen["before"]
    assert seen["wrapped"] == [True] * 6
