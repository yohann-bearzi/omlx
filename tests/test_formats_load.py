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
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama
from mlx_lm.models import switch_layers as lm_switch
from mlx_vlm.models import switch_layers as vlm_switch

from omlx import formats, oq
from omlx.formats import kernels, testing
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.patches import format_load
from omlx.utils.model_loading import (
    cast_bf16_params_to_fp16,
    maybe_apply_pre_load_patches,
)
from tests.test_formats_kernels import _BLOCKS, _PAGE, _matrix, _toy
from tests.test_formats_linear import (
    _TOY_BF16,
    EXPERTS,
    _equal,
    _experts,
    _ints,
    _ToyFormat,
    _want,
)

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_PATCHED = (nn.Linear, lm_switch.SwitchLinear, vlm_switch.SwitchLinear)
_TOY = {"bits": 8, "group_size": 1, "mode": "toy"}


class _Log(SimpleNamespace):
    """What a format was asked and handed: one log for it and every copy of
    it a load accepts with."""

    def __deepcopy__(self, memo):
        return self


class _Loadable(_ToyFormat):
    """The toy format, with what its checkpoints keep of a matrix: a byte for
    each weight and a scale for each row. ``reads`` gives the object that
    reads a checkpoint of a record (None: the load's copy of the format,
    unless a test says otherwise)."""

    def __init__(self, mode="toy"):
        super().__init__()
        self.mode, self.refusal = mode, None
        self.log = _Log(asked=[], accepted=[])
        self.reads = None

    @property
    def asked(self):
        return self.log.asked

    @property
    def accepted(self):
        return self.log.accepted

    def accept(self, record):
        # What it was handed, and whether a layer had been built before.
        self.accepted.append((record, len(self.asked)))
        if self.refusal is not None:
            raise ValueError(self.refusal)
        return self if self.reads is None else self.reads(record)

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
    found = []
    for module, name in _loaders():
        loader = _library_s(getattr(module, name), module.__name__)
        if _wrapped(loader):
            # Wrapped in place: its own code again, and wrapped again after.
            own = loader.__wrapped__
            monkeypatch.setattr(loader, "__code__", own.__code__)
            monkeypatch.setattr(loader, "__kwdefaults__", own.__kwdefaults__)
            monkeypatch.delattr(loader, "__wrapped__")
            monkeypatch.delattr(loader, "_omlx_format_load")
        else:
            found.append(loader)
        monkeypatch.setattr(module, name, loader)
    # The patch wraps each module once, the first time it finds it: what it
    # wraps then, as a process starts.
    pending = {name for name, _ in format_load._LOADERS}
    monkeypatch.setattr(
        format_load, "_PENDING", pending | set(format_load._SWITCH_MODULES)
    )
    yield
    # A loader the test's patch wrapped in place is the library's own again.
    for loader in found:
        if _wrapped(loader):
            own = loader.__wrapped__
            loader.__kwdefaults__, loader.__code__ = own.__kwdefaults__, own.__code__
            del loader.__wrapped__, loader._omlx_format_load


def _library_s(loader, module_name):
    """The library's own function under ``loader``: through the patch's
    wrappers and the families' (DeepSeek V4's holds it in its closure,
    MiniMax M3's as ``_omlx_original``)."""
    while loader.__module__ != module_name or loader.__closure__:
        cells = [cell.cell_contents for cell in loader.__closure__ or ()]
        loader = getattr(loader, "_omlx_original", None) or next(
            cell for cell in cells if inspect.isfunction(cell)
        )
    return loader


def _wrapped(loader) -> bool:
    """Whether the patch wrapped ``loader`` in place."""
    return loader.__code__ is format_load._through.__code__


def _typed(written):
    """A config of the tests' verified family (llama) that records
    ``written``."""
    return {"model_type": "llama", "omlx_quant_formats": written}


def _loaders():
    """The loaders the patch wraps, of the libraries installed here."""
    found = []
    for module_name, name in format_load._LOADERS:
        with contextlib.suppress(ImportError):
            found.append((importlib.import_module(module_name), name))
    return found


@pytest.fixture(autouse=True)
def verified():
    """The tiny checkpoints' family taken as verified: oMLX loads a format's
    checkpoint only of a verified family."""
    with testing.verified("llama"):
        yield


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
    (bare / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with format_load.loading(bare):
        yield toy


class _Model(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.toy = nn.Linear(k, n, bias=True)
        self.dense = nn.Linear(k, n, bias=False)


def load_model(path, **kwargs):
    """mlx-lm's loader as the patch leaves it, whichever the test put in its
    module's name."""
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
    # The registered format said it lays the matrix out; the load's copy laid
    # it out.
    assert toy.asked == [(k, n, 8, 1)] * 2
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
    assert toy.asked == [(k, n, 8, 1)] * 2
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


def _for_load(switch, n=16, k=64):
    """A toy format's layer as a load builds it, over placeholders, with a
    bias: a linear, or the stack of mlx-lm's or mlx-vlm's SwitchLinear
    (``switch``, its module); the checkpoint's tensors for it, and the
    arrays of a call."""
    if switch is None:
        (codes, scales), _, _ = _matrix(n, k)
        layer = nn.Linear(k, n).to_quantized(**_TOY)
        weights = {"weight": codes, "scales": scales}
        weights["bias"] = mx.arange(n).astype(mx.float32)
        return layer, weights, (_ints((3, k)),)
    (codes, scales), _, _ = _matrix(EXPERTS * n, k)
    layer = switch.SwitchLinear(k, n, EXPERTS, bias=True).to_quantized(**_TOY)
    weights = {
        "weight": codes.reshape(EXPERTS, n, k),
        "scales": scales.reshape(EXPERTS, n),
        "bias": mx.arange(EXPERTS * n).reshape(EXPERTS, n).astype(mx.float32),
    }
    indices = mx.array([4, 0, 5, 5, 2], dtype=mx.uint32)
    return layer, weights, (_ints((5, 1, k)), indices)


_LAYERS = pytest.mark.parametrize(
    "switch", [None, lm_switch, vlm_switch], ids=["linear", "mlx-lm", "mlx-vlm"]
)


def _refused(layer, placeholders):
    return re.escape(
        f"{layer!r} computes on its checkpoint's tensors alone; it still holds "
        f"the load's placeholder of {placeholders}"
    )


@_LAYERS
def test_a_checkpoint_s_tensor_of_another_dtype_is_refused_at_the_first_call(
    toy, by_hand, switch
):
    """MLX's load checks a tensor's name and shape, not its dtype: one of
    another dtype than the layout's is not loaded, and the layer refuses to
    compute on the placeholder it leaves, never on zeros."""
    layer, weights, args = _for_load(switch)
    half = weights["scales"].astype(mx.float16)
    layer.load_weights(list({**weights, "scales": half}.items()), strict=True)
    assert layer["scales"].dtype == mx.float32 and not mx.any(layer["scales"]).item()
    shape = weights["scales"].shape
    offered = (
        f"scales (offered {shape} mlx.core.float16, not the layout's {shape} "
        "mlx.core.float32)"
    )
    with pytest.raises(ValueError, match=_refused(layer, offered)):
        layer(*args)
    # An array of the layout's dtype loads it.
    layer.load_weights(list(weights.items()), strict=True)
    assert np.array_equal(np.array(layer(*args)), _want(layer, *args))


@_LAYERS
def test_a_tensor_a_load_lacks_is_refused_at_the_first_call(toy, by_hand, switch):
    """A load that is not strict drops what the model has no parameter for
    (here a tensor a sanitize named wrong) and leaves the layer's
    placeholder: the layer refuses to compute on it."""
    layer, weights, args = _for_load(switch)
    weights["scale"] = weights.pop("scales")
    layer.load_weights(list(weights.items()), strict=False)
    assert mx.array_equal(layer["weight"], weights["weight"])
    with pytest.raises(ValueError, match=_refused(layer, "scales (never loaded)")):
        layer(*args)


@_LAYERS
def test_a_tensor_of_another_shape_is_refused_at_the_first_call(toy, by_hand, switch):
    """MLX's load checks shapes only when it is strict: a tensor of another
    shape than the layout's is not taken either, and the layer refuses to
    compute rather than read past it."""
    layer, weights, args = _for_load(switch)
    shape = weights["scales"].shape
    short = weights["scales"][..., :8]
    layer.load_weights(list({**weights, "scales": short}.items()), strict=False)
    offered = (
        f"scales (offered {short.shape} mlx.core.float32, not the layout's "
        f"{shape} mlx.core.float32)"
    )
    with pytest.raises(ValueError, match=_refused(layer, offered)):
        layer(*args)


@_LAYERS
def test_codes_a_recast_hands_back_are_no_load(toy, by_hand, switch):
    """MLX's set_dtype hands the codes, which it does not cast, back to the
    layer as they are: the very array the layer holds is no load, and a load
    that lacks the codes leaves them refused."""
    layer, weights, args = _for_load(switch)
    layer.set_dtype(mx.float16)
    del weights["weight"]
    layer.load_weights(list(weights.items()), strict=False)
    with pytest.raises(ValueError, match=_refused(layer, "weight (never loaded)")):
        layer(*args)


class _HalfScales(_Loadable):
    """The toy format keeping a row's scale as bfloat16, with a face that
    reads it."""

    def __init__(self):
        super().__init__()
        self.face = _toy("load_bf16", _TOY_BF16)

    def layout(self, k, n, bits, group_size):
        layout = super().layout(k, n, bits, group_size)
        return {**layout, "scales": ((n,), mx.bfloat16)}


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("recast", ["set_dtype", "float16 activations"])
def test_a_recast_around_a_load_leaves_a_format_layer_its_checkpoint_s_tensors(
    toy, by_hand, monkeypatch, recast, when
):
    """MLX's set_dtype and oMLX's float16 activations, before the load or
    after it, leave a format layer the checkpoint's tensors, the outputs of
    a layer built from them, and the face's route."""
    monkeypatch.setitem(formats._FORMATS, "toy", _HalfScales())
    n, k = 40, 320
    (codes, scales), _, _ = _matrix(EXPERTS * n, k)
    scales = scales.astype(mx.bfloat16)
    mx.random.seed(7)
    bias = mx.random.randint(-64, 64, (EXPERTS, n)).astype(mx.bfloat16)
    matrix = {"weight": codes[:n], "scales": scales[:n]}
    stack = {
        "weight": codes.reshape(EXPERTS, n, k),
        "scales": scales.reshape(EXPERTS, n),
    }
    weights = {f"linear.{name}": value for name, value in matrix.items()}
    weights.update({f"switch.{name}": value for name, value in stack.items()})
    weights.update({"linear.bias": bias[0], "switch.bias": bias})
    model = nn.Module()
    model.linear = nn.Linear(k, n)
    model.switch = lm_switch.SwitchLinear(k, n, EXPERTS, bias=True)
    nn.quantize(model, class_predicate=_named({"linear": _TOY, "switch": _TOY}))

    def run():
        if recast == "set_dtype":
            model.set_dtype(mx.float16)
        else:
            # The biases, once the load has made them bfloat16.
            assert cast_bf16_params_to_fp16(model) == (2 if when == "after" else 0)

    if when == "before":
        run()
    model.load_weights(list(weights.items()), strict=True)
    if when == "after":
        run()
    for name, value in tree_flatten(model.parameters()):
        assert name.endswith("bias") or value is weights[name], name
    monkeypatch.setattr(_HalfScales, "decode", lambda *matrix: pytest.fail("decoded"))
    x, rows, indices = _ints((1, k)), _ints((3, 1, k)), _experts((3,))
    linear = FormatLinear("toy", 8, 1, k, matrix, bias=bias[0])
    assert _equal(model.linear(x), linear(x))
    switch = FormatSwitchLinear("toy", 8, 1, k, stack, bias=bias)
    assert _equal(model.switch(rows, indices), switch(rows, indices))
    if when == "after":
        # Float16 activations meet the recast float16 bias: float16 out.
        assert model.linear(x.astype(mx.float16)).dtype == mx.float16
        assert model.switch(rows.astype(mx.float16), indices).dtype == mx.float16
    # Through the face: one launch each, none failed.
    assert len(kernels._PROOFS) == 2
    assert all(p.ran and not p.failed for p in kernels._PROOFS.values())


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
        json.dumps(_typed({"oqel": {"plugin": "omlx-oqel 1.0.0"}}))
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
    for kept in ({}, _typed({"oqel": {"plugin": None}})):
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


def test_a_format_s_checkpoint_under_one_of_mlx_s_modes_is_refused(toy, tmp_path):
    # Written before MLX's modes were the built-ins' only, or by a newer oMLX
    # whose built-in format of the mode this one has not.
    entry = {"plugin": "shadow 0.1.0", "contract": 1, "record": {}}
    (tmp_path / "config.json").write_text(json.dumps(_typed({"mxfp4": entry})))
    with (
        format_load.loading(tmp_path),
        pytest.raises(formats.FormatRefusedError) as no,
    ):
        nn.Linear(64, 16).to_quantized(group_size=32, bits=4, mode="mxfp4")
    assert (
        no.value.reason == "no format of this mode is installed: install shadow 0.1.0"
    )
    # A checkpoint that records none of it is MLX's, and so is a load of
    # what is no checkpoint's directory.
    (tmp_path / "config.json").write_text("{}")
    for path in (tmp_path, tmp_path / "none"):
        with format_load.loading(path):
            layer = nn.Linear(64, 16).to_quantized(group_size=32, bits=4, mode="mxfp4")
        assert type(layer) is nn.QuantizedLinear
    # So is a mode that is no string: MLX refuses it in its own words.
    with pytest.raises(ValueError) as mlx_s:
        nn.Linear(64, 16).to_quantized(mode=None)
    assert type(mlx_s.value) is not formats.FormatRefusedError


@pytest.mark.parametrize(
    "model_type, family, why",
    [
        (
            "raon1_mini",
            "raon1_mini",
            "its merged gate and up projection is sliced by hand on a prefill, "
            "as MLX's quantized layer or as floats",
        ),
        (
            "Inkling2",
            "inkling2",
            "its DSpark draft folds the output head into its own projection by "
            "hand, as MLX's quantized layer or as floats",
        ),
        (
            "brand-new",
            "brand_new",
            "it is no family verified with quant formats end to end",
        ),
    ],
)
def test_a_checkpoint_of_a_family_oq_writes_no_format_of_is_refused_at_load(
    toy, tmp_path, monkeypatch, model_type, family, why
):
    # Written by hand, or by a plugin's own writer: not by oQ, which refuses.
    _llama(tmp_path, **_written({}))
    config = json.loads((tmp_path / "config.json").read_text())
    typed = {**config, "model_type": model_type}
    (tmp_path / "config.json").write_text(json.dumps(typed))

    def read(*args, **kwargs):
        raise AssertionError("a weight read")

    monkeypatch.setattr(mx, "load", read)
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path)
    assert str(refusal.value) == _refusal(
        f"oMLX loads no {family} checkpoint in a quant format: {why}; have the "
        "family verified",
        tmp_path,
    )
    # One that records and names no format is the library's.
    del config["omlx_quant_formats"], config["quantization"][_GATE]
    typed = {**config, "model_type": model_type}
    (tmp_path / "config.json").write_text(json.dumps(typed))
    with pytest.raises(Exception) as other:
        load_model(tmp_path)
    assert not isinstance(other.value, formats.FormatRefusedError)


def test_a_format_of_an_unverified_family_is_refused_however_it_is_named(toy, tmp_path):
    _llama(tmp_path, **_written({}))
    config = json.loads((tmp_path / "config.json").read_text())
    # No record: its quantization config names the format.
    del config["omlx_quant_formats"]
    raon = {**config, "model_type": "raon1_mini"}
    (tmp_path / "config.json").write_text(json.dumps(raon))
    with pytest.raises(formats.FormatRefusedError, match="loads no raon1_mini"):
        load_model(tmp_path)
    # Nor does its config: the load is handed it (a model_config).
    del raon["quantization"][_GATE]
    (tmp_path / "config.json").write_text(json.dumps(raon))
    named = {"quantization": {**raon["quantization"], _GATE: _TOY}}
    with pytest.raises(formats.FormatRefusedError, match="loads no raon1_mini"):
        load_model(
            tmp_path,
            model_config=named,
            get_model_classes=lambda config: (llama.Model, llama.ModelArgs),
        )
    # One that names no model type is told to name it.
    untyped = _typed({"toy": {}}) | {"model_type": ""}
    (tmp_path / "config.json").write_text(json.dumps(untyped))
    with pytest.raises(formats.FormatRefusedError, match="; name its model type"):
        format_load._refuse_family(format_load._Checkpoint(tmp_path))


def test_the_modes_a_config_names():
    config = {
        "omlx_quant_formats": {"oqel": {}},
        "quantization": {"mode": "affine", "a": {"mode": "toy"}, "b": {}},
        "text_config": {"quantization_config": {"mode": "other"}},
    }
    # Those it records, or else those it names that are not MLX's.
    assert format_load._format_mode(config) == "oqel"
    for empty in ({}, None):
        config["omlx_quant_formats"] = empty
        assert format_load._format_mode(config) == "toy"
    del config["quantization"]["a"]
    assert format_load._format_mode(config) == "other"


def test_a_config_alone_names_the_family(toy, tmp_path):
    # A directory named as a decision model's, and a link to it: the config's
    # model type decides, in any view.
    openjev = tmp_path / "OpenJev-9B"
    openjev.mkdir()
    (openjev / "config.json").write_text(
        json.dumps(_typed({"toy": {}}) | {"model_type": "qwen3_5"})
    )
    prod = tmp_path / "prod"
    prod.symlink_to(openjev, target_is_directory=True)
    for path in (openjev, prod, openjev / ".", prod / ".." / "OpenJev-9B"):
        with pytest.raises(formats.FormatRefusedError, match="loads no qwen3_5"):
            format_load._refuse_family(format_load._Checkpoint(path))
        with testing.verified("qwen3_5"):
            checkpoint = format_load._Checkpoint(path)
            format_load._refuse_family(checkpoint)
            assert checkpoint.verified


@pytest.mark.parametrize("family", ["clef", "openjev", "d3", "d1"])
def test_a_decision_model_s_loader_refuses_it_in_a_format(toy, tmp_path, family):
    from omlx.models.d1 import D1Model
    from omlx.models.decision import DecisionBackbone

    path = tmp_path / "decision"
    path.mkdir()
    config = _typed({"toy": {}}) | {"model_type": "qwen3_5"}
    (path / "config.json").write_text(json.dumps(config))
    # Weights that no load reads: refused before any is.
    (path / "model.safetensors").write_bytes(b"not a checkpoint")
    model = (
        D1Model(str(path))
        if family == "d1"
        else DecisionBackbone(str(path), False, family)
    )
    why = {
        "clef": "it reads rows of its output head by hand, as MLX's quantized "
        "layers or as floats",
    }.get(family, "it is no family verified with quant formats end to end")
    # Its backbone's family verified: served as a decision model, it is not.
    with testing.verified("qwen3_5"), pytest.raises(formats.FormatRefusedError) as no:
        model.load()
    assert str(no.value) == _refusal(
        f"oMLX loads no {family} checkpoint in a quant format: {why}; have the "
        "family verified",
        path,
        written="an unknown writer",
    )
    if family == "clef":
        quoted = str(no.value).replace(str(path), "/models/my-model")
        for said in ("an unknown writer", "a format registered by hand"):
            quoted = quoted.replace(said, "my-package 0.1.0")
        assert quoted.replace("'toy'", "'myformat'") in " ".join(_PAGE.split())
    # Named in its quantization config alone, or a damaged record, too.
    for named in (
        {"model_type": "qwen3_5", "quantization": {"mode": "toy"}},
        {"model_type": "qwen3_5", "omlx_quant_formats": None},
    ):
        (path / "config.json").write_text(json.dumps(named))
        with pytest.raises(formats.FormatRefusedError, match=f"loads no {family}"):
            format_load.refuse_family(path, family)
    # Verifying both families cannot grant the decision reader a format load.
    with (
        testing.verified(family, "qwen3_5"),
        pytest.raises(formats.FormatRefusedError, match=f"loads no {family}"),
    ):
        model.load()
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    format_load.refuse_family(path, family)
    format_load.refuse_family(tmp_path / "none", family)


def test_a_damaged_record_of_formats_is_refused_though_no_layer_reads_it(toy, tmp_path):
    for written, said in ((None, "null"), ([], "an array"), ("toy", "a string")):
        config = {"model_type": "raon1_mini", "omlx_quant_formats": written}
        (tmp_path / "config.json").write_text(json.dumps(config))
        # Of a family not verified: refused before a weight is read.
        with (
            pytest.raises(formats.FormatRefusedError, match="loads no raon1_mini"),
            format_load.loading(tmp_path),
        ):
            pass
        # Of a verified one whose load built no format's layer: at its end.
        config["model_type"] = "llama"
        (tmp_path / "config.json").write_text(json.dumps(config))
        with (
            format_load.loading(tmp_path) as checkpoint,
            pytest.raises(formats.FormatRefusedError) as no,
        ):
            checkpoint.check(nn.Linear(2, 2))
        assert no.value.reason == (
            f"config.json['omlx_quant_formats'] is {said}, where oQ writes an "
            "object: the checkpoint's config is damaged"
        )
        assert no.value.mode == "unknown"


def _hand_built(kind):
    """A toy layer of ``kind`` built by hand."""
    stack = (2,) if kind is FormatSwitchLinear else ()
    tensors = {
        "weight": mx.zeros((*stack, 128, 64), mx.uint8),
        "scales": mx.zeros((*stack, 128), mx.float32),
    }
    return kind("toy", 8, 1, 64, tensors)


@pytest.mark.parametrize("kind", [FormatLinear, FormatSwitchLinear])
def test_a_format_s_layer_is_built_only_in_a_load_that_passed_the_gate(
    toy, tmp_path, monkeypatch, kind
):
    import contextvars

    # No family verified, and no block open but this file's own.
    monkeypatch.setattr(oq, "_TEST_VERIFIED", frozenset())
    monkeypatch.setattr(formats, "_TESTING", False)
    said = (
        "quant format 'toy': a format's layer is built only while oMLX loads a "
        "checkpoint of a family verified with quant formats, and this one is "
        "built outside such a load (by a family's own loader, or by hand): load "
        "the checkpoint with mlx-lm's or mlx-vlm's load_model"
    )
    with pytest.raises(formats.FormatRefusedError) as no:
        _hand_built(kind)
    assert str(no.value) == said and no.value.path is None
    assert said.replace("'toy'", "'myformat'") in " ".join(_PAGE.split())
    stock, named = tmp_path / "stock", tmp_path / "named"
    configs = ((stock, {"model_type": "llama"}), (named, _typed({"toy": {}})))
    for path, config in configs:
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config))
    # The load of a checkpoint the gate never passed (it names no format),
    # one block outside a load, or a load the gate refused.
    with format_load.loading(stock), pytest.raises(formats.FormatRefusedError):
        _hand_built(kind)
    with (
        pytest.raises(formats.FormatRefusedError, match="loads no llama"),
        format_load.loading(named),
    ):
        pass
    # Verified for its block: the load of a checkpoint the gate passed.
    with testing.verified("llama"), format_load.loading(named) as checkpoint:
        assert checkpoint.verified
        # Its own layer, on its thread and on one that copied its context.
        monkeypatch.setattr(formats, "_TESTING", False)
        assert type(_hand_built(kind)) is kind
        context = contextvars.copy_context()
        assert type(context.run(_hand_built, kind)) is kind
        with format_load.outside_load(), pytest.raises(formats.FormatRefusedError):
            _hand_built(kind)
    # Once its load has ended, a thread that copied its context builds none.
    monkeypatch.setattr(formats, "_TESTING", False)
    with pytest.raises(formats.FormatRefusedError):
        context.run(_hand_built, kind)
    # In a test's block, by hand; not once the last is closed.
    with monkeypatch.context() as blocks:
        blocks.setattr(testing, "_VERIFYING", [])
        with testing.verified("qwen3_5"):
            assert type(_hand_built(kind)) is kind
        assert formats._TESTING is False
        with pytest.raises(formats.FormatRefusedError):
            _hand_built(kind)


@pytest.mark.parametrize(
    "mode, written, reason",
    [
        # The built-in affine format lays out no matrix: a checkpoint that
        # records a format of affine is not read as MLX's.
        (
            "affine",
            {"affine": {"plugin": "shadow 0.1.0", "contract": 1, "record": {}}},
            "it lays out no (16, 64) matrix of 4 bits in groups of 32",
        ),
        (
            "affine",
            {"affine": None},
            "config.json['omlx_quant_formats']['affine'] is null, where oQ writes "
            "an object: the checkpoint's config is damaged",
        ),
        (
            "mxfp4",
            None,
            "config.json['omlx_quant_formats'] is null, where oQ writes an "
            "object: the checkpoint's config is damaged",
        ),
    ],
    ids=["affine", "affine null", "formats null"],
)
def test_a_record_of_one_of_mlx_s_modes_is_never_read_as_mlx_s(
    toy, tmp_path, mode, written, reason
):
    config = _typed(written)
    (tmp_path / "config.json").write_text(json.dumps(config))
    with (
        format_load.loading(tmp_path),
        pytest.raises(formats.FormatRefusedError) as refusal,
    ):
        nn.Linear(64, 16).to_quantized(group_size=32, bits=4, mode=mode)
    assert refusal.value.reason == reason
    # The built-in format is named so.
    assert refusal.value.installed == ("built-in" if mode == "affine" else "none")


def test_an_entry_of_a_recorded_mlx_mode_that_binds_no_layer_is_refused(toy, tmp_path):
    _llama(tmp_path, **_written({}))
    config = json.loads((tmp_path / "config.json").read_text())
    config["omlx_quant_formats"]["mxfp4"] = {"plugin": "shadow 0.1.0", "contract": 1}
    config["quantization"]["model.layers.0.mlp.gate"] = {
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(
        formats.FormatRefusedError, match="names model.layers.0.mlp.gate"
    ):
        load_model(tmp_path, strict=False)


def test_a_refusal_names_a_hub_snapshot_not_its_blob(toy, tmp_path):
    repo = tmp_path / "models--org--tiny"
    snapshot, blobs = repo / "snapshots" / "abc", repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (blobs / "0f1e").write_text(json.dumps(_typed({"toy": None})))
    (snapshot / "config.json").symlink_to(Path("../../blobs/0f1e"))
    view = tmp_path / "view"
    view.mkdir()
    (view / "config.json").symlink_to(snapshot / "config.json")
    for path in (snapshot, view):
        with format_load.loading(path), pytest.raises(formats.FormatRefusedError) as no:
            nn.Linear(64, 128).to_quantized(**_TOY)
        assert no.value.path == snapshot


def test_a_refusal_names_the_checkpoint_through_any_links(toy, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "config.json").write_text(json.dumps(_typed({"toy": None})))
    view = tmp_path / "view"
    view.mkdir()
    (view / "config.json").symlink_to(Path("../real/config.json"))
    # Links in a cycle, to themselves or to each other.
    alone, first, second = (tmp_path / name for name in ("alone", "a", "b"))
    for path in (alone, first, second):
        path.mkdir()
    (alone / "config.json").symlink_to(Path("config.json"))
    (first / "config.json").symlink_to(Path("../b/config.json"))
    (second / "config.json").symlink_to(Path("../a/config.json"))
    assert format_load._shown(view) == real
    for path in (alone, first):
        assert format_load._shown(path) in (alone, first, second)


def test_a_view_is_named_by_its_checkpoint_from_any_path(toy, tmp_path, monkeypatch):
    real = tmp_path / "models" / "real"
    (tmp_path / "models" / "views" / "v1").mkdir(parents=True)
    real.mkdir()
    (real / "config.json").write_text(json.dumps(_typed({})))
    view = tmp_path / "models" / "views" / "v1"
    (view / "config.json").symlink_to(Path("../../real/config.json"))
    # A link to the view's directory: its link is relative to where it is.
    shortcut = tmp_path / "shortcut"
    shortcut.symlink_to(view, target_is_directory=True)
    for path in (view, shortcut):
        assert format_load._shown(path).resolve() == real.resolve()
    # A relative path, with "..", names the same checkpoint as its whole one.
    monkeypatch.chdir(tmp_path / "models")
    assert format_load._shown(Path("views/../real")) == real
    with (
        format_load.loading(real, by_loader=True) as outer,
        format_load.loading("views/../real") as inner,
    ):
        assert inner is outer


def test_a_config_that_is_no_object_records_no_format(toy, tmp_path):
    (tmp_path / "config.json").write_text("[]")
    with format_load.loading(tmp_path):
        layer = nn.Linear(64, 16).to_quantized(group_size=32, bits=4, mode="mxfp4")
    assert type(layer) is nn.QuantizedLinear


def test_a_recorded_mode_that_bound_nothing_is_refused(toy, tmp_path):
    # The config's own mode, which no entry names.
    quantization = {"group_size": 1, "bits": 8, "mode": "toy"}
    config = {**_typed({"toy": {}}), "quantization": quantization}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError, match="config's mode is toy"):
        format_load._Checkpoint(tmp_path).check(nn.Module())
    # An entry without a mode is MLX's affine, whatever the config's mode
    # (MLX builds it so): no format's layer to bind.
    quantization.update({"b": {"group_size": 64, "bits": 4}, "a": _TOY})
    (tmp_path / "config.json").write_text(json.dumps(config))
    with format_load.loading(tmp_path) as checkpoint:
        layer = nn.Linear(64, 128).to_quantized(**_TOY)
    model = nn.Module()
    model.a = layer
    checkpoint.check(model)
    # The config's own mode a format's under either key, or its text model's
    # alone; one it does not name, MLX's affine, when the checkpoint records
    # a format of affine; one that is not MLX's, recorded or not.
    for config in (
        {**_typed({"toy": {}}), "quantization_config": {"bits": 8, "mode": "toy"}},
        {"model_type": "llama", "text_config": {"quantization": {"mode": "toy"}}},
        {**_typed({"affine": {}}), "quantization": {"bits": 4}},
        {"model_type": "llama", "quantization": {"bits": 8, "mode": "toy"}},
    ):
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(formats.FormatRefusedError, match="config's mode is"):
            format_load._Checkpoint(tmp_path).check(nn.Module())


def _checked(tmp_path, quantization, held, **config):
    """Check a load of a checkpoint whose quantization config is
    ``quantization`` (and whose config holds ``config`` too), whose model
    holds one toy layer at each path of ``held``, and modules at the others
    ``held`` maps to False; no quantization config of its own for None."""
    whole = {**_typed({"toy": {}}), **config}
    if quantization is not None:
        whole["quantization"] = quantization
    (tmp_path / "config.json").write_text(json.dumps(whole))
    with format_load.loading(tmp_path) as checkpoint:
        layers = {
            path: nn.Linear(64, 128).to_quantized(**_TOY) if kept else nn.Module()
            for path, kept in held.items()
        }
    model = nn.Module()
    for path, layer in sorted(layers.items(), key=lambda item: item[0].count(".")):
        *parents, name = path.split(".")
        module = model
        for parent in parents:
            if parent not in module:
                module[parent] = nn.Module()
            module = module[parent]
        module[name] = layer
    checkpoint.check(model)


def test_an_entry_binds_one_layer_and_a_layer_one_entry(toy, tmp_path):
    # One layer under language_model.: one of the two entries it would
    # stand for binds it, the other none.
    entries = {"a": _TOY, "language_model.a": _TOY}
    with pytest.raises(formats.FormatRefusedError, match="the load built no layer"):
        _checked(tmp_path, entries, {"language_model.a": True})
    _checked(tmp_path, entries, {"language_model.a": True, "a": True})
    # A Hugging Face config's name of a layer mlx-vlm builds at
    # language_model.model., and mlx-lm at model.
    for held in ("language_model.model.a", "model.a"):
        _checked(tmp_path, {"model.language_model.a": _TOY}, {held: True})


@pytest.mark.parametrize("part", ["mtp", "vision_tower", "visual", "audio_tower"])
def test_an_entry_of_a_part_the_load_left_out_binds_nothing(toy, tmp_path, part):
    for entry in (f"{part}.layers.0.a", f"language_model.{part}.layers.0.a"):
        # The model has no such part (an MTP head left off, a text-only
        # load): the entry has no layer to bind.
        _checked(tmp_path, {entry: _TOY}, {"model.a": False})
        # It has the part: the entry binds a layer of it, or is refused.
        for holder in (part, f"language_model.{part}"):
            with pytest.raises(formats.FormatRefusedError, match=f"names {entry},"):
                _checked(tmp_path, {entry: _TOY}, {holder: False})
    # Any other part is the model's.
    with pytest.raises(formats.FormatRefusedError, match="names draft.a,"):
        _checked(tmp_path, {"draft.a": _TOY}, {"model.a": False})


def test_entries_are_those_the_library_builds_from(toy, tmp_path):
    # Under either key.
    with pytest.raises(formats.FormatRefusedError, match="names b,"):
        _checked(tmp_path, {"a": _TOY}, {"a": True}, quantization_config={"b": _TOY})
    # Its text model's, when the config has none of its own.
    for key in ("quantization", "quantization_config"):
        with pytest.raises(formats.FormatRefusedError, match="names b,"):
            text = {key: {"a": _TOY, "b": _TOY}}
            _checked(tmp_path, None, {"a": True}, text_config=text)
    # A top-level map never hides an unbound declaration in the text model.
    with pytest.raises(formats.FormatRefusedError, match="names b,"):
        _checked(
            tmp_path,
            {"a": _TOY},
            {"a": True},
            text_config={"quantization": {"b": _TOY}},
        )
    _checked(
        tmp_path, {"a": _TOY}, {"a": True}, text_config={"quantization": {"a": _TOY}}
    )


def test_a_failure_to_read_the_config_is_raised_bare_each_time(tmp_path):
    import traceback

    checkpoint = format_load._Checkpoint(tmp_path)
    depths = []
    for _ in range(3):
        with pytest.raises(OSError) as error:
            checkpoint._read_config()
        depths.append(len(traceback.extract_tb(error.value.__traceback__)))
    assert depths[0] == depths[1] == depths[2]
    # Each raise is a copy: the kept error never gathers a traceback.
    with pytest.raises(OSError) as again:
        checkpoint._read_config()
    assert again.value is not error.value
    assert checkpoint._config.__traceback__ is None


def test_a_checkpoint_s_config_is_read_once_whatever_it_holds(tmp_path, monkeypatch):
    reads = []
    real = Path.read_text

    def read_text(self, *args, **kwargs):
        reads.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    for text in (None, "null", "{"):
        if text is not None:
            (tmp_path / "config.json").write_text(text)
        reads.clear()
        checkpoint = format_load._Checkpoint(tmp_path)
        # What is no object names no entry; what is not read is read once.
        if text == "null":
            checkpoint.check(nn.Module())
        for _ in range(2):
            with contextlib.suppress(OSError, ValueError):
                checkpoint.records("toy")
        assert checkpoint.writer("toy") == "an unknown writer"
        assert len(reads) == 1, text
        (tmp_path / "config.json").unlink(missing_ok=True)


def test_a_format_s_refusals_outside_a_load_name_the_checkpoint(toy, tmp_path):
    _llama(tmp_path, **_written({"levels": [1.0]}))

    def refuses(record):
        raise ValueError("written in layout 2")

    for reads in (refuses, lambda record: "a string", None):
        toy.reads = reads
        # The last: a format no copy takes.
        toy.faces = threading.Lock() if reads is None else None
        with pytest.raises(formats.FormatRefusedError) as refusal:
            format_load._Checkpoint(tmp_path).read(toy)
        assert refusal.value.path == tmp_path


def test_a_plugin_that_did_not_load_is_named_by_its_checkpoint(toy, tmp_path):
    # Installed, and skipped when the registry loaded (here for the revision
    # of the contract it is written for).
    skipped = (
        "omlx-oqel 2.0.0 is installed but did not load: quant format 'oqel' is "
        "written for revision 2 of the format contract; this oMLX runs revision "
        "1: update oMLX"
    )
    (tmp_path / "config.json").write_text(
        json.dumps(_typed({"oqel": {"plugin": "omlx-oqel 2.0.0"}}))
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
    loaders = [getattr(module, name) for module, name in _loaders()]
    assert len(wrappers + loaders) == len(_PATCHED) + len(format_load._LOADERS)
    assert all(wrapper._omlx_format_load for wrapper in wrappers + loaders)
    owns = [loader.__wrapped__ for loader in loaders]
    assert all(map(_wrapped, loaders)) and not any(map(_wrapped, owns))
    format_load.apply_format_load_patch()
    assert [cls.to_quantized for cls in _PATCHED] == wrappers
    assert [getattr(module, name).__wrapped__ for module, name in _loaders()] == owns
    assert not any(
        getattr(wrapper.__wrapped__, "_omlx_format_load", False)
        for wrapper in wrappers + loaders
    )
    # A module found again once wrapped (its import ending on one thread as
    # the patch is applied on another) is left as it is.
    for module, _ in _loaders():
        format_load._wrap(module)
    assert [getattr(module, name).__wrapped__ for module, name in _loaders()] == owns
    # What a loader's callers read of it: its own name, signature and text.
    for loader, own in zip(loaders, owns, strict=True):
        assert inspect.signature(loader) == inspect.signature(own)
        named = ("__name__", "__qualname__", "__module__", "__doc__")
        for attribute in (*named, "__annotations__"):
            assert getattr(own, attribute) == getattr(loader, attribute)
        assert own.__globals__ is loader.__globals__


def test_what_a_loader_s_callers_read_of_it_is_the_library_s(unpatched, monkeypatch):
    import functools

    loaders = [getattr(module, name) for module, name in _loaders()]
    before = [inspect.signature(loader) for loader in loaders]
    for loader in loaders:
        monkeypatch.setattr(loader, "omlx_test_mark", "the library's", raising=False)
    format_load.apply_format_load_patch()
    for loader, signature in zip(loaders, before, strict=True):
        assert _wrapped(loader) and inspect.signature(loader) == signature
        # What functools.wraps over it copies.
        wrapper = functools.wraps(loader)(lambda *args, **kwargs: None)
        assert inspect.signature(wrapper) == signature
        assert wrapper.__name__ == loader.__name__ and wrapper.__doc__ == loader.__doc__
        # The copy the loader runs keeps the function's own attributes.
        assert loader.__wrapped__.omlx_test_mark == "the library's"


_ALL_EARLY = """
from mlx_lm.utils import load_model as lm
from mlx_vlm.utils import load_model as vlm
from mlx_vlm.encoder_loader import load_encoder_model as encoder
from mlx_embeddings.utils import load_model as embeddings
import json
from omlx.patches import format_load

format_load.apply_format_load_patch()
through = format_load._through.__code__
print(json.dumps([f.__code__ is through for f in (lm, vlm, encoder, embeddings)]))
"""


def test_each_library_loader_imported_before_the_patch_is_wrapped():
    import subprocess

    run = subprocess.run(
        [sys.executable, "-c", _ALL_EARLY],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert json.loads(run.stdout.splitlines()[-1]) == [True] * 4


def test_a_loader_is_wrapped_once_whoever_applies_the_patch(unpatched, monkeypatch):
    loaders = [getattr(module, name) for module, name in _loaders()]
    ready = threading.Barrier(4)

    def apply():
        ready.wait(60)
        format_load.apply_format_load_patch()

    threads = [threading.Thread(target=apply) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    owns = [loader.__wrapped__ for loader in loaders]
    assert all(map(_wrapped, loaders)) and not any(map(_wrapped, owns))
    # Its module found again, pending once more: left as it is.
    monkeypatch.setattr(format_load, "_PENDING", {n for n, _ in format_load._LOADERS})
    format_load.apply_format_load_patch()
    assert [loader.__wrapped__ for loader in loaders] == owns


def test_a_library_loader_that_is_no_function_fails_every_load(unpatched, monkeypatch):
    import functools
    import types

    module = types.ModuleType("changed")
    module.load_model = functools.partial(lambda path: None)
    monkeypatch.setitem(sys.modules, "changed", module)
    monkeypatch.setattr(
        format_load, "_LOADERS", (*format_load._LOADERS, ("changed", "load_model"))
    )
    format_load._PENDING.add("changed")
    partial = r"changed\.load_model is a partial"
    with pytest.raises(formats.LoadHookError, match=partial):
        format_load.apply_format_load_patch()
    # The VLM engine's load lets it through.
    import asyncio

    from omlx.engine.vlm import VLMBatchedEngine

    format_load._PENDING.add("changed")
    with pytest.raises(formats.LoadHookError):
        asyncio.run(VLMBatchedEngine("/models/none").start())


def test_a_model_load_applies_the_patch(unpatched, tmp_path):
    loaders = [getattr(module, name) for module, name in _loaders()]
    assert not any(hasattr(cls.to_quantized, "__wrapped__") for cls in _PATCHED)
    assert not any(getattr(loader, "_omlx_format_load", False) for loader in loaders)
    maybe_apply_pre_load_patches(str(tmp_path))
    assert all(cls.to_quantized._omlx_format_load for cls in _PATCHED)
    for (module, name), loader in zip(_loaders(), loaders, strict=True):
        # In place: the function object the module held, and every name it
        # was imported under.
        assert getattr(module, name) is loader and _wrapped(loader)


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
        _the_load_s_copy(model.model.layers[0].mlp.gate_proj.fmt, toy)
    # Asked by the registered format, and by the first load's copy, before.
    assert toy.accepted == [({}, 1), ({}, 3)]


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
    return _typed({"toy": entry})


@pytest.mark.parametrize(
    "config, where, kind",
    [
        ([], "", "an array"),
        (None, "", "null"),
        (_typed([]), "['omlx_quant_formats']", "an array"),
        (_typed(None), "['omlx_quant_formats']", "null"),
        (_typed({"toy": "v1"}), _ENTRY, "a string"),
        (_typed({"toy": None}), _ENTRY, "null"),
        (_kept_record(None), f"{_ENTRY}['record']", "null"),
        (_kept_record([1]), f"{_ENTRY}['record']", "an array"),
        (_kept_record(2.5), f"{_ENTRY}['record']", "a number"),
        (_kept_record(3), f"{_ENTRY}['record']", "a number"),
        (_kept_record(True), f"{_ENTRY}['record']", "a boolean"),
    ],
    ids=[
        "config",
        "config null",
        "formats",
        "formats null",
        "entry",
        "entry null",
        "null",
        "array",
        "number",
        "integer",
        "boolean",
    ],
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
    if kind == "null" and "record" in where:
        quoted = str(no.value).replace(str(tmp_path), "/models/my-model")
        quoted = quoted.replace("omlx-toy 0.3.1", "my-package 0.1.0")
        quoted = quoted.replace("a format registered by hand", "my-package 0.1.0")
        assert quoted.replace("'toy'", "'myformat'") in " ".join(_PAGE.split())


def test_what_the_checkpoint_says_of_its_writer(toy, tmp_path):
    # oQ records a format registered by hand as null; anything else that
    # names no plugin is an unknown writer.
    for plugin, said in (
        (None, "a format registered by hand"),
        ("omlx-toy 0.3.1", "omlx-toy 0.3.1"),
        ("", "an unknown writer"),
        ("  ", "an unknown writer"),
        (123, "an unknown writer"),
        (False, "an unknown writer"),
    ):
        entry = {"plugin": plugin, "record": {"layout": 2}}
        config = _typed({"toy": entry})
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert format_load._Checkpoint(tmp_path).writer("toy") == said
    # A directory without a config.
    assert format_load._Checkpoint(tmp_path / "none").writer("toy") == (
        "an unknown writer"
    )


def test_a_checkpoint_reads_its_config_once(toy, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_kept_record({"levels": [1]})))
    checkpoint = format_load._Checkpoint(tmp_path)
    assert checkpoint._record("toy") == {"levels": [1]}
    (tmp_path / "config.json").write_text(json.dumps(_kept_record({"levels": [2]})))
    assert checkpoint._record("toy") == {"levels": [1]}
    # Read outside any load, a refusal still names the checkpoint.
    (tmp_path / "config.json").write_text(json.dumps(_kept_record([1])))
    with pytest.raises(formats.FormatRefusedError) as refusal:
        format_load._Checkpoint(tmp_path).read(toy)
    assert refusal.value.path == tmp_path
    # A config that is not JSON is the library's error, not the format's.
    (tmp_path / "config.json").write_text("{")
    with pytest.raises(json.JSONDecodeError):
        format_load._Checkpoint(tmp_path).read(toy)


def test_a_refusal_in_a_view_of_links_names_the_checkpoint(toy, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(_kept_record([1])))
    view = tmp_path / "view"
    view.mkdir()
    (view / "config.json").symlink_to(source / "config.json")
    with format_load.loading(view), pytest.raises(formats.FormatRefusedError) as no:
        nn.Linear(64, 128).to_quantized(**_TOY)
    assert no.value.path == source


def test_a_format_s_own_refusal_names_the_checkpoint(toy, tmp_path):
    def refusing(record):
        raise formats.FormatRefusedError("written in layout 2", mode="toy")

    toy.reads = refusing
    _llama(tmp_path, **_written({"layout": 2}))
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path)
    assert str(refusal.value) == _refusal("written in layout 2", tmp_path)
    assert refusal.value.path == tmp_path
    # What it knows of nothing else.
    bare = formats.FormatRefusedError("why", mode="toy")
    assert (bare.path, bare.written, bare.installed) == (None, None, None)


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


def _the_load_s_copy(read, fmt):
    """``read`` is a copy of ``fmt`` that a load accepted with."""
    assert read is not fmt and type(read) is type(fmt)
    assert read.log is fmt.log


def test_accept_returning_none_reads_with_the_load_s_copy(toy, tmp_path):
    # As the guide's example of revision 1 wrote accept.
    toy.reads = lambda record: None
    _llama(tmp_path, **_written({"layout": 1}))
    model, _ = load_model(tmp_path)
    _the_load_s_copy(model.model.layers[0].mlp.gate_proj.fmt, toy)


class _Configured(_Loadable):
    """A format that configures itself from the record to accept it."""

    __slots__ = ("level", "__table")

    def __init__(self):
        super().__init__()
        self.options, self.level, self.__table = {"rows": 1}, 1, [0.5]

    def accept(self, record):
        super().accept(record)
        self.levels = tuple(record["levels"])
        self.options["rows"] = len(self.levels)
        self.level += 1
        self.__table.append(self.levels[0])
        del self.refusal
        return self


def test_what_accept_changes_of_its_format_never_reaches_the_registered_one(
    tmp_path, monkeypatch, unpatched
):
    formats.load()
    fmt = _Configured()
    monkeypatch.setitem(formats._FORMATS, "toy", fmt)
    format_load.apply_format_load_patch()
    for name, levels in (("first", [1.0, 2.0]), ("second", [3.0])):
        (tmp_path / name).mkdir()
        _llama(tmp_path / name, **_written({"levels": levels}))
    first, _ = load_model(tmp_path / "first")
    second, _ = load_model(tmp_path / "second")
    read = [model.model.layers[0].mlp.gate_proj.fmt for model in (first, second)]
    for kept in read:
        _the_load_s_copy(kept, fmt)
    # Each load configured its own copy, attributes, slots and what they hold.
    kept = [(fmt.levels, fmt.options, fmt.level) for fmt in read]
    assert kept == [((1.0, 2.0), {"rows": 2}, 2), ((3.0,), {"rows": 1}, 2)]
    assert read[0]._Configured__table == [0.5, 1.0]
    assert read[1]._Configured__table == [0.5, 3.0]
    assert not hasattr(read[0], "refusal")
    # The registered format is as it was built.
    assert not hasattr(fmt, "levels") and fmt.refusal is None
    assert (fmt.options, fmt.level, fmt._Configured__table) == ({"rows": 1}, 1, [0.5])


@pytest.mark.parametrize("held", ["a lock", "a dtype", "a copy that fails"])
def test_a_format_no_copy_takes_is_refused(toy, tmp_path, monkeypatch, held):
    class _Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("no copy of this")

    value, said = {
        "a lock": (threading.Lock(), "cannot pickle '_thread.lock' object"),
        "a dtype": ((mx.float16,), "cannot pickle 'mlx.core.Dtype' object"),
        "a copy that fails": (_Uncopyable(), "no copy of this"),
    }[held]
    toy.faces = value
    _llama(tmp_path, **_written({"layout": 1}))
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path)
    assert str(refusal.value) == _refusal(
        f"the format cannot be copied for the load ({said}): accept runs on a "
        "copy of the registered format, which holds no lock, dtype or other "
        "object that copy.deepcopy refuses; report it to the plugin",
        tmp_path,
    )
    assert toy.accepted == [] and toy.faces is value
    if held == "a lock":
        quoted = str(refusal.value).replace(str(tmp_path), "/models/my-model")
        for said in ("the plugin", "omlx-toy 0.3.1", "a format registered by hand"):
            quoted = quoted.replace(said, "my-package 0.1.0")
        assert quoted.replace("'toy'", "'myformat'") in " ".join(_PAGE.split())
    # Named by its distribution when a plugin's.
    monkeypatch.setitem(formats._ORIGINS, "toy", "my-package 0.1.0")
    with pytest.raises(formats.FormatRefusedError, match="to my-package 0.1.0$"):
        load_model(tmp_path)


def test_a_load_s_record_is_its_own_again_after_a_block_outside_it(toy, tmp_path):
    _llama(tmp_path, **_written({}))
    with format_load.loading(tmp_path) as checkpoint:
        with format_load.outside_load():
            assert formats._LOADING.get() is None
        assert formats._LOADING.get() is checkpoint


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


def test_a_layer_built_in_a_load_and_not_in_its_model_is_refused(
    toy, tmp_path, monkeypatch
):
    import types

    _llama(tmp_path, **_written({"levels": [1.0]}))
    toy.reads = lambda record: _Levels(record["levels"])
    monkeypatch.setitem(formats._FORMATS, "toy2", _Loadable("toy2"))
    load = mlx_lm.utils.load_model

    def with_a_stranger(path, **kwargs):
        # Another checkpoint's layer, quantized by hand inside this one's load
        # (as a loader bound before the patch would): it reads this one's
        # record, here of another format than the checkpoint's own layer.
        stranger = nn.Linear(64, 128).to_quantized(bits=8, group_size=1, mode="toy2")
        assert type(stranger) is FormatLinear
        return load(path, **kwargs)

    module = types.ModuleType("mlx_lm.utils")
    module.load_model = with_a_stranger
    format_load._wrap_loader(module, "load_model")
    with pytest.raises(formats.FormatRefusedError) as refusal:
        module.load_model(tmp_path)
    said = _refusal(
        "1 layers were built while the checkpoint loaded, and are not in its "
        "model: they were another checkpoint's, read with this one's record; load "
        "each checkpoint with the libraries' load_model",
        tmp_path,
        written="an unknown writer",
    )
    assert str(refusal.value) == said.replace("'toy'", "'toy2'")


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


@pytest.mark.parametrize(
    "entry",
    [
        # A path that is no module of the model, as a family's loader that
        # renames its modules leaves it.
        "model.layers.0.mlp.gate",
        # The end of a path of the model, which the libraries bind nothing
        # with: an entry names a module by its whole path.
        "gate_proj",
        "mlp.gate_proj",
    ],
)
def test_a_format_entry_that_binds_no_layer_is_refused(toy, tmp_path, entry):
    _llama(tmp_path, **_written({}))
    config = json.loads((tmp_path / "config.json").read_text())
    config["quantization"][entry] = _TOY
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path, strict=False)
    assert str(refusal.value) == _refusal(
        f"its quantization config names {entry}, and the load built no layer "
        "of the format there: the model has no such module, or its family's "
        "loader renames it",
        tmp_path,
    )


def test_a_checked_loader_accepts_copies_of_its_disk_format_config(toy, tmp_path):
    record = {"levels": [1.0, None], "layout": 2}
    _llama(tmp_path, **_written(record))
    config = json.loads((tmp_path / "config.json").read_text())
    model, _ = load_model(tmp_path, model_config=config)
    assert type(model.model.layers[0].mlp.gate_proj) is FormatLinear
    assert [seen for seen, _ in toy.accepted] == [record]


def test_an_entry_binds_a_layer_of_its_own_mode_only(toy, tmp_path):
    _llama(tmp_path, **_written({}))
    config = json.loads((tmp_path / "config.json").read_text())
    # The same layer under language_model., in a mode no layer of it has:
    # not bound by the toy one.
    config["quantization"][f"language_model.{_GATE}"] = {
        "group_size": 1,
        "bits": 8,
        "mode": "uninstalled",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path, strict=False)
    assert refusal.value.mode == "uninstalled"
    # In its own mode, it binds the layer (a VLM's checkpoint loaded by
    # mlx-lm), which then binds no other entry.
    config["quantization"][f"language_model.{_GATE}"] = _TOY
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError, match="the load built no layer"):
        load_model(tmp_path, strict=False)
    # The checkpoint's own entry, under language_model., binds the layer
    # mlx-lm builds from the names it is handed (oMLX's text-only load hands
    # it the text model's).
    del config["quantization"][_GATE]
    (tmp_path / "config.json").write_text(json.dumps(config))
    named = {"quantization": {**config["quantization"], _GATE: _TOY}}
    load_model(tmp_path, strict=False, model_config=named)


def test_an_entry_binds_a_layer_at_any_path_its_model_holds_it(toy, tmp_path):
    config = {
        "model_type": "llama",
        "quantization": {"a": _TOY, "b": _TOY},
        "omlx_quant_formats": {"toy": {}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with format_load.loading(tmp_path) as checkpoint:
        layer = nn.Linear(64, 128).to_quantized(**_TOY)
    model = nn.Module()
    model.a = model.b = layer
    checkpoint.check(model)


def test_a_format_entry_of_a_load_that_built_no_layer_is_refused(toy, tmp_path):
    _llama(tmp_path, **_written({}))
    config = json.loads((tmp_path / "config.json").read_text())
    del config["quantization"][_GATE]
    config["quantization"]["model.layers.0.mlp.gate"] = _TOY
    (tmp_path / "config.json").write_text(json.dumps(config))
    _rewritten(tmp_path, lambda weights: weights.pop(f"{_GATE}.scales"))
    with pytest.raises(
        formats.FormatRefusedError, match="names model.layers.0.mlp.gate,"
    ):
        load_model(tmp_path, strict=False)


def test_a_load_that_leaves_no_array_where_the_layout_has_one_is_refused(toy, tmp_path):
    _llama(tmp_path, **_written({}))
    # Tensors nested under the layout's name: MLX's partial load puts a
    # list there.
    _rewritten(
        tmp_path,
        lambda weights: weights.update(
            {f"{_GATE}.scales.0": weights.pop(f"{_GATE}.scales")}
        ),
    )
    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model(tmp_path, strict=False)
    assert refusal.value.reason == (
        f"the load left {_GATE}.scales list, where the format lays out an array: "
        "the checkpoint holds tensors under that name"
    )
    assert (refusal.value.mode, refusal.value.path) == ("toy", tmp_path)


def test_a_stock_load_is_not_walked(toy, tmp_path, monkeypatch):
    _llama(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    del config["quantization"][_GATE]
    (tmp_path / "config.json").write_text(json.dumps(config))
    _rewritten(tmp_path, lambda weights: weights.pop(f"{_GATE}.scales"))

    def walked(self):
        raise AssertionError("a stock load's model walked")

    monkeypatch.setattr(nn.Module, "named_modules", walked)
    load_model(tmp_path, strict=False)


def test_a_load_joined_to_an_outer_one_leaves_it_the_check(toy, tmp_path):
    _llama(tmp_path, **_written({}))
    with format_load.loading(tmp_path, by_loader=True) as outer:
        # The outer load builds a layer of its own model before the inner
        # one: the inner load's model has no such layer, and is not refused.
        own = nn.Linear(64, 128).to_quantized(**_TOY)
        model, _ = load_model(tmp_path, strict=True)
    assert own in outer.built and _gate_proj(model) in outer.built
    # oMLX's exit check inside it, or after it ended, before its own check:
    # the layers keep their placeholders for it.
    with format_load.loading(tmp_path, by_loader=True) as outer:
        model, _ = load_model(tmp_path, strict=True)
        format_load.check_loaded(model)
    format_load.check_loaded(model)
    outer.check(model)
    assert "_load_origin" in _gate_proj(model).__dict__
    # Checked: the next exit check lets them go.
    format_load.check_loaded(model)
    assert "_load_origin" not in _gate_proj(model).__dict__


def test_a_load_joins_only_a_running_loader_s_load_on_its_thread(toy, tmp_path):
    import contextvars

    partial = _renamed_scales(tmp_path / "partial")
    with format_load.loading(partial) as bare:
        # A block no library's loader opened (oMLX's helpers, a test's): the
        # load nested in it is its own, and checks what it loaded.
        with pytest.raises(formats.FormatRefusedError, match="but its scales"):
            load_model(partial, strict=False)
        assert bare.built == []
    with format_load.loading(partial, by_loader=True) as outer:
        copied = contextvars.copy_context()
        # Another thread that copied the load's context loads on its own.
        seen = []
        worker = threading.Thread(
            target=copied.run,
            args=(lambda: seen.append(format_load._current()),),
        )
        worker.start()
        worker.join(60)
        assert seen == [outer]
        # Its own load of the checkpoint is its own, while this one runs too.
        worker = threading.Thread(
            target=copied.run,
            args=(lambda: seen.append(_joined(partial, outer)),),
        )
        worker.start()
        worker.join(60)
        assert seen[-1] is False
        with format_load.loading(partial) as inner:
            assert inner is outer
    other = threading.Thread(
        target=copied.run,
        args=(lambda: seen.append(_joined(partial, outer)),),
    )
    other.start()
    other.join(60)
    # After the load: no layer is the ended load's.
    assert seen[-1] is False
    with pytest.raises(formats.FormatRefusedError, match="built outside"):
        copied.run(nn.Linear(64, 128).to_quantized, **_TOY)


def _renamed_scales(path):
    """A tiny checkpoint whose toy layer's scales are under another name."""
    path.mkdir()
    _llama(path, **_written({}))
    _rewritten(path, _renamed)
    return path


def _joined(path, outer):
    with format_load.loading(path) as checkpoint:
        return checkpoint is outer


def test_snapshots_sharing_a_config_blob_are_two_checkpoints(toy, tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "0f1e").write_text(json.dumps({"model_type": "llama"}))
    snapshots = [tmp_path / "snapshots" / name for name in ("abc", "def")]
    for snapshot in snapshots:
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").symlink_to(blobs / "0f1e")
    with (
        format_load.loading(snapshots[0], by_loader=True) as first,
        format_load.loading(snapshots[1]) as second,
    ):
        assert second is not first


def test_a_load_nested_in_another_checkpoint_s_checks_its_own(toy, tmp_path):
    (tmp_path / "outer").mkdir()
    _llama(tmp_path / "outer", **_written({}))
    partial = _renamed_scales(tmp_path / "partial")
    with (
        format_load.loading(tmp_path / "outer", by_loader=True),
        pytest.raises(formats.FormatRefusedError, match="but its scales"),
    ):
        load_model(partial, strict=False)


def test_loads_of_one_format_accept_on_copies_of_their_own(toy, tmp_path):
    paths = [tmp_path / name for name in ("first", "second")]
    for path in paths:
        path.mkdir()
        _llama(path, **_written({"levels": [1.0]}))
    both = threading.Barrier(2, timeout=30)

    def accept(record):
        # Each load's accept waits for the other's: no lock holds one back.
        both.wait()

    toy.accept = accept
    built = []

    def build(path):
        with format_load.loading(path):
            built.append(nn.Linear(64, 128).to_quantized(**_TOY))

    threads = [threading.Thread(target=build, args=(path,)) for path in paths]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    # Each layer reads with its own load's copy.
    assert len(built) == 2 and built[0].fmt is not built[1].fmt
    for layer in built:
        assert layer.fmt is not toy and type(layer.fmt) is type(toy)


def test_a_load_under_home_or_in_a_view_joins_the_checkpoint_s(
    toy, tmp_path, monkeypatch
):
    home = tmp_path / "home"
    (home / "tiny").mkdir(parents=True)
    _llama(home / "tiny", **_written({}))
    monkeypatch.setenv("HOME", str(home))
    view = tmp_path / "view"
    view.mkdir()
    (view / "config.json").symlink_to(home / "tiny" / "config.json")
    with format_load.loading(home / "tiny", by_loader=True) as outer:
        for path in ("~/tiny", view):
            with format_load.loading(path) as inner:
                assert inner is outer
    assert format_load._Checkpoint("~/tiny").path == home / "tiny"


def test_a_load_s_threads_accept_its_record_once(toy, tmp_path):
    import contextvars
    import time

    _llama(tmp_path, **_written({"levels": [1.0]}))
    accepted = []

    def accept(record):
        accepted.append(record)
        # Long enough for the other thread to ask meanwhile.
        time.sleep(0.2)

    toy.accept = accept
    built = []
    with format_load.loading(tmp_path):
        # Threads that copied the load's context (asyncio.to_thread's).
        threads = [
            threading.Thread(
                target=contextvars.copy_context().run,
                args=(lambda: built.append(nn.Linear(64, 128).to_quantized(**_TOY)),),
            )
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
    assert len(built) == 2 and accepted == [{"levels": [1.0]}]


def _without_gate(tmp_path):
    """A tiny checkpoint whose files hold none of the toy layer's tensors."""
    _tiny(tmp_path, [1.0])

    def partial(weights):
        del weights[f"{_GATE}.weight"], weights[f"{_GATE}.scales"]

    _rewritten(tmp_path, partial)
    return tmp_path


def test_omlx_s_own_load_refuses_a_layer_the_checkpoint_does_not_hold(
    toy, tmp_path
):
    from omlx.patches.mlx_lm_sharded_load import _local_sharded_load
    from omlx.utils.model_loading import lm_load_compat

    path = _without_gate(tmp_path / "partial")
    said = _refusal(
        f"the loaded model holds {_GATE} without its weight, scales: the "
        "checkpoint holds no such tensor, or holds it under another name",
        path,
    )
    # A load that drops nothing: the cluster's for one rank.
    with pytest.raises(formats.FormatRefusedError) as refusal:
        _local_sharded_load(str(path))
    assert str(refusal.value) == said
    quoted = said.replace(str(path), "/models/my-model")
    for plugin in ("omlx-toy 0.3.1", "a format registered by hand"):
        quoted = quoted.replace(plugin, "my-package 0.1.0")
    assert quoted.replace("'toy'", "'myformat'") in " ".join(_PAGE.split())
    # The libraries' strict load says it first.
    with pytest.raises(ValueError, match="Missing"):
        lm_load_compat(str(path))


def test_check_loaded_lets_a_layer_dropped_by_the_load_go(toy, tmp_path):
    path = _without_gate(tmp_path / "partial")
    model, _ = load_model(path, strict=False)
    layer = model.model.layers[0].mlp.gate_proj
    with pytest.raises(formats.FormatRefusedError, match="without its weight") as no:
        format_load.check_loaded(model)
    # Named by the checkpoint the layer was built for.
    assert (no.value.path, no.value.written) == (path, "omlx-toy 0.3.1")
    # A check of it again (another thread's, a later caller's) refuses again.
    with pytest.raises(formats.FormatRefusedError, match="without its weight"):
        format_load.check_loaded(model)
    # What this process does not hold is dropped before (a prefill helper's
    # later layers, a pipeline rank's others): no refusal.
    model.model.layers[0].mlp.gate_proj = nn.Linear(64, 128)
    format_load.check_loaded(model)
    # A filled layer lets its placeholders go once checked.
    full = tmp_path / "full"
    full.mkdir()
    _llama(full, **_written({}))
    model, _ = load_model(full)
    layer = model.model.layers[0].mlp.gate_proj
    assert set(layer._load_origin[0]) == {"weight", "scales"}
    format_load.check_loaded(model)
    assert not hasattr(layer, "_load_origin")
    format_load.check_loaded(model)


def test_a_process_that_built_no_format_layer_walks_no_model(monkeypatch):
    monkeypatch.setattr(format_load, "_BUILT_ANY", False)

    def walked(self):
        raise AssertionError("a model walked")

    monkeypatch.setattr(nn.Module, "named_modules", walked)
    format_load.check_loaded(nn.Linear(4, 4))


def test_a_refusal_of_a_drafter_is_no_misconfigured_toggle(monkeypatch):
    from omlx.speculative import vlm_mtp

    def refused(path, kind):
        raise formats.FormatRefusedError("refused", mode="toy")

    monkeypatch.setattr(vlm_mtp, "_vlm_load_drafter", refused)
    with pytest.raises(formats.FormatRefusedError):
        vlm_mtp.load_vlm_mtp_drafter("/models/drafter")
    # Nor is a loader the patch could not wrap.
    def unwrapped(path, kind):
        raise formats.LoadHookError("mlx_lm.utils.load_model is no function")

    monkeypatch.setattr(vlm_mtp, "_vlm_load_drafter", unwrapped)
    with pytest.raises(formats.LoadHookError):
        vlm_mtp.load_vlm_mtp_drafter("/models/drafter")
    # Any other failure is one, logged.
    monkeypatch.setattr(vlm_mtp, "_vlm_load_drafter", lambda path, kind: 1 / 0)
    assert vlm_mtp.load_vlm_mtp_drafter("/models/drafter") is None


def test_omlx_s_own_check_inside_a_library_load_leaves_it_the_layers(toy, tmp_path):
    _llama(tmp_path, **_written({}))
    with format_load.loading(tmp_path, by_loader=True) as outer:
        model, _ = load_model(tmp_path)
        # oMLX's helper inside a library's loader: the load still runs.
        format_load.check_loaded(model)
        assert hasattr(_gate_proj(model), "_load_origin")
    outer.check(model)
    format_load.check_loaded(model)
    assert not hasattr(_gate_proj(model), "_load_origin") and outer.built == []


def test_a_drafter_or_draft_refused_is_never_skipped():
    import ast
    from pathlib import Path

    root = Path(format_load.__file__).parents[1]
    soft = (
        "toggle ignored",
        "Lightning MTP stays off",
        "SpecPrefill: draft model load failed",
    )

    def rethrows_refusal(handler):
        return (
            isinstance(handler.type, ast.Tuple)
            and {getattr(name, "id", None) for name in handler.type.elts}
            == {"FormatRefusedError", "LoadHookError"}
            and isinstance(handler.body[-1], ast.Raise)
            and handler.body[-1].exc is None
        )

    def awaits(handler, method):
        return any(
            isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "attr", None) == method
            for node in ast.walk(handler)
        )

    for file, expected in (
        ("engine_pool.py", 4),
        ("engine/batched.py", 1),
        ("engine/vlm.py", 1),
    ):
        tree = ast.parse((root / file).read_text())
        blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        # Pool drafters share a typed outer handler that releases the loaded target.
        guarded = set()
        for block in blocks:
            if any(
                rethrows_refusal(h) and awaits(h, "_release_loaded_engine")
                for h in block.handlers
            ):
                guarded.update(node for stmt in block.body for node in ast.walk(stmt))
        seen = 0
        for block in blocks:
            for index, handler in enumerate(block.handlers):
                if not any(
                    isinstance(n, ast.Constant)
                    and isinstance(n.value, str)
                    and any(s in n.value for s in soft)
                    for n in ast.walk(handler)
                ):
                    continue
                # The engine's fail-fast variant wraps a non-format error.
                if any(
                    isinstance(n, ast.Raise)
                    and isinstance(n.exc, ast.Call)
                    and getattr(n.exc.func, "id", None) == "RuntimeError"
                    for n in ast.walk(handler)
                ):
                    continue
                seen += 1
                assert any(
                    rethrows_refusal(h)
                    and (
                        block in guarded
                        if file == "engine_pool.py"
                        else awaits(h, "stop")
                    )
                    for h in block.handlers[:index]
                ), (file, block.lineno)
        assert seen == expected, file


def test_each_of_omlx_s_own_loads_checks_what_it_loaded():
    import ast
    from pathlib import Path

    root = Path(format_load.__file__).parents[1]
    calls = {}
    for file in (
        "utils/model_loading.py",
        "engine/vlm.py",
        "patches/mlx_lm_sharded_load.py",
        "cluster/progressive_loading.py",
        "prefill_helper/worker.py",
        "speculative/dflash/__init__.py",
        "speculative/vlm_mtp.py",
        "models/embedding.py",
        "models/reranker.py",
        "models/d1.py",
        "models/decision.py",
        "oq.py",
        "prefill_helper/adapters/qwen4_exp.py",
        "prefill_helper/adapters/qwen3_5.py",
        "prefill_helper/adapters/glm5_next.py",
        "prefill_helper/adapters/minimax_m3.py",
        "prefill_helper/adapters/inkling2.py",
    ):
        tree = ast.parse((root / file).read_text())
        checks = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", None))
            == "check_loaded"
        ]
        # Each checks the model it loaded: a name or an item of one.
        for call in checks:
            (model,) = call.args
            assert isinstance(model, ast.Name | ast.Attribute | ast.Subscript), file
        calls[file] = len(checks)
        # Before a compile's warm-up runs the model.
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            lines = {
                getattr(node.func, "id", getattr(node.func, "attr", None)): node.lineno
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
            }
            if "_try_compile" in lines and "check_loaded" in lines:
                assert lines["check_loaded"] < lines["_try_compile"], function.name
    # lm_load_compat (the LLM engine, its SpecPrefill draft, oMLX's LLM
    # wrappers, a causal reranker), the VLM engine, the cluster's loads
    # (mlx-lm's sharded load and the local fallback's, an architecture's own
    # stage loader, both progressive strategies), a prefill helper after
    # keep_first, both drafters, the embedding model's mlx-vlm and
    # mlx-embeddings loads (its native one builds no quantized layer), the
    # reranker's, the decision models' (d1, a vision backbone), and oQ's own
    # (its calibration, both sensitivity measures, Muse Spark's calibration).
    assert calls == {
        "utils/model_loading.py": 1,
        "engine/vlm.py": 1,
        "patches/mlx_lm_sharded_load.py": 2,
        "cluster/progressive_loading.py": 3,
        "prefill_helper/worker.py": 1,
        "speculative/dflash/__init__.py": 1,
        "speculative/vlm_mtp.py": 1,
        "models/embedding.py": 2,
        "models/reranker.py": 1,
        "models/d1.py": 1,
        "models/decision.py": 1,
        "oq.py": 4,
        # Right after keep_first, before a fusion or an evaluation; the
        # worker checks the families whose keep_first runs before the load.
        "prefill_helper/adapters/qwen4_exp.py": 1,
        "prefill_helper/adapters/qwen3_5.py": 1,
        "prefill_helper/adapters/glm5_next.py": 1,
        "prefill_helper/adapters/minimax_m3.py": 1,
        # Before anything it says of the layers, or evaluates.
        "prefill_helper/adapters/inkling2.py": 1,
    }


def test_a_loader_wrapped_over_the_wrapped_one_is_one_load(toy, tmp_path):
    import types

    # A family's loader that calls the libraries' (wrapped) loader, wrapped
    # in its turn: the layers built belong to the one load of the checkpoint.
    _llama(tmp_path, **_written({}))
    module = types.ModuleType("mlx_lm.utils")
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

    def load_model():
        # No library's loader, by its name only.
        return nn.Linear(64, 16).to_quantized(**_TOY)

    with pytest.raises(formats.FormatRefusedError) as refusal:
        load_model()
    assert str(refusal.value) == _refusal(
        "a layer is built outside a checkpoint's load, which gives it the record "
        "to be read with: load the checkpoint with mlx-lm's or mlx-vlm's load_model"
    )
    assert refusal.value.path is None
    # MLX's layers need none.
    assert type(nn.Linear(64, 16).to_quantized(32, 4)) is nn.QuantizedLinear


@pytest.mark.parametrize("library", ["mlx_lm", "mlx_vlm"])
def test_a_loader_left_unwrapped_is_named_with_the_fix(
    unpatched, tmp_path, monkeypatch, library
):
    module = importlib.import_module(f"{library}.utils")
    early = module.load_model

    def family_s(*args, **kwargs):
        return early(*args, **kwargs)

    # A family's loader took the module's name before the patch was
    # applied: the library's own function is not wrapped.
    monkeypatch.setattr(module, "load_model", family_s)
    formats.load()
    toy = _Levels([0.0])
    toy.reads = lambda record: _Levels(record["levels"])
    monkeypatch.setitem(formats._FORMATS, "toy", toy)
    format_load.apply_format_load_patch()
    # Wrapped by its name, as a family's loader, and once: found again, it
    # is left as it is.
    wrapped = module.load_model
    assert wrapped.__wrapped__ is family_s and wrapped.__name__ == "family_s"
    format_load._PENDING.add(module.__name__)
    format_load.apply_format_load_patch()
    assert module.load_model is wrapped
    path = _tiny(tmp_path / "tiny", [2.0])
    loaded = module.load_model(path)
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    assert _gate_proj(model).fmt.levels == (2.0,)
    with pytest.raises(formats.FormatRefusedError) as refusal:
        early(path)
    loader = f"{library}.utils.load_model"
    assert refusal.value.reason == (
        f"{loader} was imported before oMLX's patches were applied and runs "
        "unwrapped (a family's loader took its name in its module first): call "
        f"{loader} through its module, or apply the patches first "
        "(maybe_apply_pre_load_patches)"
    )


def test_a_family_s_loader_over_the_wrapped_one_is_left_as_it_is(
    toy, tmp_path, monkeypatch
):
    import functools

    # K2 Horizon's: functools.wraps over the library's, wrapped in place.
    wrapped = mlx_lm.utils.load_model

    @functools.wraps(wrapped)
    def family_s(*args, **kwargs):
        return wrapped(*args, **kwargs)

    monkeypatch.setattr(mlx_lm.utils, "load_model", family_s)
    format_load._PENDING.add("mlx_lm.utils")
    format_load.apply_format_load_patch()
    assert mlx_lm.utils.load_model is family_s


def test_a_family_s_loader_that_keeps_the_wrapped_one_is_left(
    toy, tmp_path, monkeypatch
):
    # MiniMax M3's keeps the library's (wrapped in place) as _omlx_original.
    def kept(*args, **kwargs):
        return None

    kept._omlx_original = mlx_lm.utils.load_model
    monkeypatch.setattr(mlx_lm.utils, "load_model", kept)
    format_load._PENDING.add("mlx_lm.utils")
    format_load.apply_format_load_patch()
    # Neither by its name nor in place.
    assert mlx_lm.utils.load_model is kept and not _wrapped(kept)


def test_a_layer_built_outside_a_load_under_a_family_s_loader_is_not_blamed(
    unpatched, tmp_path, monkeypatch
):
    early = mlx_lm.utils.load_model

    def family_s(*args, **kwargs):
        return early(*args, **kwargs)

    monkeypatch.setattr(mlx_lm.utils, "load_model", family_s)
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _Loadable())
    format_load.apply_format_load_patch()
    _llama(tmp_path, **_written({}))

    def classes(config):
        with format_load.outside_load():
            nn.Linear(64, 128).to_quantized(**_TOY)
        return llama.Model, llama.ModelArgs

    with pytest.raises(formats.FormatRefusedError) as refusal:
        mlx_lm.utils.load_model(tmp_path, get_model_classes=classes)
    assert refusal.value.reason.startswith("a layer is built outside a checkpoint's")


def test_a_layer_built_outside_a_wrapped_load_is_not_blamed_on_an_import(toy, tmp_path):
    _llama(tmp_path, **_written({}))

    def classes(config):
        # A family's loader that builds a layer of its own outside the load.
        with format_load.outside_load():
            nn.Linear(64, 128).to_quantized(**_TOY)
        return llama.Model, llama.ModelArgs

    with pytest.raises(formats.FormatRefusedError) as refusal:
        mlx_lm.utils.load_model(tmp_path, get_model_classes=classes)
    assert refusal.value.reason.startswith("a layer is built outside a checkpoint's")


def test_the_vlm_engine_lets_the_load_hook_s_failure_through(monkeypatch):
    import asyncio

    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.utils import model_loading

    def fails(*args, **kwargs):
        raise formats.LoadHookError("mlx_vlm.utils.load_model is a partial")

    monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", fails)
    with pytest.raises(formats.LoadHookError):
        asyncio.run(VLMBatchedEngine("/models/none").start())


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


def test_a_causal_lm_reranker_refuses_a_verified_backbone(
    unpatched, tmp_path, monkeypatch
):
    from omlx.models.reranker import MLXRerankerModel

    toy = _registered(monkeypatch)
    path = _tiny(tmp_path / "tiny-reranker", [9.0])
    config = json.loads((path / "config.json").read_text())
    config["architectures"] = ["Qwen3ForCausalLM"]
    (path / "config.json").write_text(json.dumps(config))
    reranker = MLXRerankerModel(str(path))
    with pytest.raises(formats.FormatRefusedError) as refused:
        reranker.load()
    assert refused.value.reason == (
        "the reranker loader cannot read a quant format checkpoint"
    )
    assert not toy.accepted


def test_an_embedding_model_refuses_a_verified_backbone(
    unpatched, tmp_path, monkeypatch
):
    from omlx.models import embedding

    toy = _registered(monkeypatch)
    # Through mlx-vlm's encoder loader, as its embedding models load.
    monkeypatch.setitem(embedding._MLX_VLM_EMBEDDING_CONTEXT, "llama", 512)
    path = _tiny(tmp_path / "embedding", [10.0])
    with pytest.raises(formats.FormatRefusedError) as refused:
        embedding.MLXEmbeddingModel(str(path)).load()
    assert refused.value.reason == (
        "the embedding loader cannot read a quant format checkpoint"
    )
    assert not toy.accepted


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


@pytest.mark.parametrize("load", ["calibration", "sensitivity", "proxy"])
def test_oq_s_vlm_loads_refuse_a_layer_they_left_unfilled(
    levels, tmp_path, monkeypatch, load
):
    from omlx import oq

    path = _tiny(tmp_path / "source", [14.0])
    _rewritten(
        path,
        lambda weights: [
            weights.pop(f"{_GATE}.{name}") for name in ("weight", "scales")
        ],
    )
    config = json.loads((path / "config.json").read_text())
    monkeypatch.setattr(oq, "_is_vlm_load", lambda config: True)
    calls = {
        "calibration": lambda: oq._calibration_model(str(path), config),
        "sensitivity": lambda: oq._measure_sensitivity(str(path), config, 4),
        "proxy": lambda: oq._measure_sensitivity_from_quantized_model(
            str(path), config, 4
        ),
    }
    # A partial load leaves the layer: oQ's own load refuses it, and lets the
    # refusal through as it is.
    with pytest.raises(formats.FormatRefusedError, match="without its weight"):
        calls[load]()


# Their checkpoints of a format are refused at load: taken as verified, the
# test reaches what their loaders do.
@testing.verified("deepseek_v4", "k2_horizon")
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
    # The custom DeepSeek reader refuses before opening its weights in
    # either patch order, even with a test-verified family.
    v4 = _tiny(tmp_path / "v4", [16.0])
    config = json.loads((v4 / "config.json").read_text())
    (v4 / "config.json").write_text(json.dumps({**config, "model_type": "deepseek_v4"}))
    for _ in range(2):
        format_load.apply_format_load_patch()
        assert not getattr(mlx_lm.utils.load_model, "_omlx_format_load", False)
        with pytest.raises(formats.FormatRefusedError, match="DeepSeek V4 loader"):
            mlx_lm.utils.load_model(
                v4, get_model_classes=lambda config: (llama.Model, llama.ModelArgs)
            )
    # The same refusal when installed underneath the wrapped loader.
    mlx_lm.utils.load_model = wrapped.__wrapped__
    mlx_lm.utils.load_model = utils_patch._build_patched_load_model()
    monkeypatch.setattr(format_load, "_PENDING", {"mlx_lm.utils"})
    format_load.apply_format_load_patch()
    assert getattr(mlx_lm.utils.load_model, "_omlx_format_load", False)
    with pytest.raises(formats.FormatRefusedError, match="DeepSeek V4 loader"):
        mlx_lm.utils.load_model(
            v4, get_model_classes=lambda config: (llama.Model, llama.ModelArgs)
        )
    # K2 Horizon's loads a directory of links to the checkpoint's files.
    mlx_lm.utils.load_model = wrapped
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", mlx_lm.utils.load_tokenizer)
    checkpoint.apply_checkpoint_patch()
    k2 = _tiny(tmp_path / "k2", [17.0])
    config = json.loads((k2 / "config.json").read_text())
    (k2 / "config.json").write_text(json.dumps({**config, "model_type": "k2_horizon"}))
    model, _ = mlx_lm.utils.load_model(
        k2, get_model_classes=lambda config: (llama.Model, llama.ModelArgs)
    )
    assert _gate_proj(model).fmt.levels == (17.0,)


def test_a_deepseek_v4_checkpoint_of_a_format_is_refused_before_a_weight_is_read(
    toy, tmp_path, monkeypatch
):
    import mlx_lm.utils

    from omlx.patches.deepseek_v4 import utils_patch

    monkeypatch.setattr(mlx_lm.utils, "load_model", mlx_lm.utils.load_model)
    mlx_lm.utils.load_model = utils_patch._build_patched_load_model()
    # Weights that no load reads.
    (tmp_path / "model.safetensors").write_bytes(b"not a checkpoint")
    for named in (
        _typed({"toy": {}}),
        {"quantization_config": {"mode": "toy"}},
        {"text_config": {"quantization": {"mode": "toy"}}},
    ):
        for model_type in ("deepseek_v4", "deepseek_v4_mtp"):
            config = {**named, "model_type": model_type}
            (tmp_path / "config.json").write_text(json.dumps(config))
            with pytest.raises(formats.FormatRefusedError) as no:
                load_model(tmp_path)
            assert no.value.reason == (
                f"oMLX loads no {model_type} checkpoint in a quant format: its "
                "DSpark draft reads the output head's weight by hand, as floats; "
                "have the family verified"
            )


@testing.verified("deepseek_v4")
def test_deepseek_v4_s_native_reader_refuses_a_damaged_record(
    toy, tmp_path, monkeypatch
):
    import mlx_lm.utils

    from omlx.patches.deepseek_v4 import utils_patch

    monkeypatch.setattr(mlx_lm.utils, "load_model", mlx_lm.utils.load_model)
    mlx_lm.utils.load_model = utils_patch._build_patched_load_model()
    config = {
        "model_type": "deepseek_v4",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
    }
    model = llama.Model(llama.ModelArgs.from_dict(config))
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    # Even float weights cannot go through this reader with a format record.
    (tmp_path / "config.json").write_text(
        json.dumps({**config, "omlx_quant_formats": None})
    )
    with pytest.raises(formats.FormatRefusedError, match="DeepSeek V4 loader"):
        load_model(
            tmp_path, get_model_classes=lambda config: (llama.Model, llama.ModelArgs)
        )


def test_a_module_half_imported_by_another_thread_is_wrapped_when_it_ends(
    unpatched, monkeypatch
):
    import time
    import types
    from importlib import _bootstrap

    formats.load()
    monkeypatch.setattr(
        format_load, "_LOADERS", (*format_load._LOADERS, ("halfway", "load_model"))
    )
    module = types.ModuleType("halfway")
    held = threading.Event()

    def importing(then):
        # Another thread's import of it, begun before the patch was applied:
        # not in sys.modules yet. It ends once the patch waits for it.
        with _bootstrap._ModuleLockManager("halfway"):
            held.set()
            lock = _bootstrap._module_locks["halfway"]()
            deadline = time.monotonic() + 60
            while not lock.waiters and time.monotonic() < deadline:
                time.sleep(0.01)
            then()

    def fails():
        # A failed import leaves nothing in sys.modules.
        sys.modules["halfway"] = types.ModuleType("halfway")
        del sys.modules["halfway"]

    def ends():
        module.load_model = lambda path: None
        monkeypatch.setitem(sys.modules, "halfway", module)

    for then, wrapped in ((fails, False), (ends, True)):
        format_load._PENDING.add("halfway")
        held.clear()
        other = threading.Thread(target=importing, args=(then,))
        other.start()
        assert held.wait(60)
        # Waited for; a failed import left nothing, and is not imported here.
        format_load.apply_format_load_patch()
        other.join(60)
        assert ("halfway" in format_load._PENDING) is not wrapped
    assert _wrapped(module.load_model)
    # This thread's own import, still running: left for its end.
    module = types.ModuleType("mine")
    monkeypatch.setattr(
        format_load, "_LOADERS", (*format_load._LOADERS, ("mine", "load_model"))
    )
    format_load._PENDING.add("mine")
    monkeypatch.setitem(sys.modules, "mine", module)
    with _bootstrap._ModuleLockManager("mine"):
        format_load.apply_format_load_patch()
    assert "mine" in format_load._PENDING


def test_an_import_lock_in_a_deadlock_is_not_the_patch_s_failure(
    unpatched, monkeypatch
):
    from importlib import _bootstrap

    class Deadlocked:
        def __init__(self, name):
            pass

        def __enter__(self):
            raise _bootstrap._DeadlockError("deadlock detected by _ModuleLock")

        def __exit__(self, *raised):
            return False

    monkeypatch.setattr(format_load._bootstrap, "_ModuleLockManager", Deadlocked)
    format_load.apply_format_load_patch()
    # The modules imported already are wrapped all the same.
    assert all(_wrapped(getattr(module, name)) for module, name in _loaders())


_EARLY = """
# A user's script: the loader imported before oMLX's patches are applied.
from {library}.utils import load_model

import json, sys
from pathlib import Path

from omlx import formats, oq
from omlx.formats import testing
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from tests.test_formats_load import _gate_proj, _Levels

fmt = _Levels([0.0])
fmt.reads = lambda record: _Levels(record["levels"])
formats.register(fmt)
maybe_apply_pre_load_patches(sys.argv[1])
# The tiny checkpoint's family, taken as verified.
with testing.verified("llama"):
    loaded = load_model(Path(sys.argv[1]))
model = loaded[0] if isinstance(loaded, tuple) else loaded
print(json.dumps(_gate_proj(model).fmt.levels))
"""


@pytest.mark.parametrize("library", ["mlx_lm", "mlx_vlm"])
def test_a_loader_imported_before_the_patch_reads_its_record(tmp_path, library):
    import subprocess

    path = _tiny(tmp_path / "tiny", [3.0])
    run = subprocess.run(
        [sys.executable, "-c", _EARLY.format(library=library), str(path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert json.loads(run.stdout.splitlines()[-1]) == [3.0]


@pytest.mark.parametrize("drafter", ["DFlash", "VLM-MTP"])
def test_a_drafter_s_loader_installs_the_patch(
    unpatched, tmp_path, monkeypatch, drafter
):
    toy = _registered(monkeypatch)
    path = str(_tiny(tmp_path / "drafter", [18.0]))
    if drafter == "DFlash":
        from omlx.speculative.dflash import load_dflash_drafter

        with pytest.raises(TypeError):
            load_dflash_drafter(path, target_model=None)
    else:
        from omlx.speculative.vlm_mtp import load_vlm_mtp_drafter

        assert load_vlm_mtp_drafter(path) is None
    assert [tuple(record["levels"]) for record, _ in toy.accepted] == [(18.0,)]


# Of a family oMLX serves with mlx-embeddings' code: taken as verified.
@testing.verified("bert")
def test_mlx_embeddings_refuses_at_the_resolved_checkpoint(levels, tmp_path):
    from mlx_embeddings.models import bert
    from mlx_embeddings.utils import load_model as embeddings_load_model

    config = {
        "model_type": "bert",
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "max_position_embeddings": 16,
        "vocab_size": 32,
    }
    model = bert.Model(bert.ModelArgs.from_dict(config))
    weights = dict(tree_flatten(model.parameters()))
    name = next(
        path
        for path, module in model.named_modules()
        if type(module) is nn.Linear and module.weight.shape == (128, 64)
    )
    (codes, scales), _, _ = _matrix(128, 64)
    weights[f"{name}.weight"], weights[f"{name}.scales"] = codes, scales
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    config["quantization"] = {"group_size": 64, "bits": 4, name: _TOY}
    config["omlx_quant_formats"] = _written({"levels": [19.0]})
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError, match="mlx_embeddings.utils"):
        embeddings_load_model(tmp_path, lazy=True)
    assert not levels()


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
