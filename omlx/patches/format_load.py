# SPDX-License-Identifier: Apache-2.0
"""A checkpoint's layers of a quant format's mode load as format layers.

mlx-lm and mlx-vlm rebuild a quantized checkpoint's modules with
``nn.quantize``: each layer the quantization config names is replaced by its
``to_quantized(group_size, bits, mode)``, and the checkpoint's tensors are
then loaded over the result's, strictly. MLX's layers build MLX's modes.
This patch has them build a layer of ``omlx.formats.linear`` for a mode whose
format lays the matrix out (``QuantFormat.layout``): the layer holds
placeholders of the checkpoint's shapes, which the load fills.
"""

from __future__ import annotations

import importlib
import inspect
import json
from functools import cache, wraps
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .. import formats
from ..formats.linear import FormatLinear, FormatSwitchLinear

# The modules whose SwitchLinear holds a stack of matrices, (E, N, K).
_SWITCH_MODULES = ("mlx_lm.models.switch_layers", "mlx_vlm.models.switch_layers")

# What a format's layer is built from, of to_quantized's parameters.
_TAKEN = ("self", "mode", "bits", "group_size")


@cache
def _mlx_runs(mode) -> bool:
    """Whether ``mode`` is one of MLX's own."""
    try:
        mx.quantize(mx.zeros((1, 64)), mode=mode)
    except ValueError:
        return False
    return True


def _format_layer(layer, layer_cls, signature, arguments):
    """The format layer that ``layer`` loads as, or None: MLX builds its
    mode."""
    mode, bits, group_size = (arguments[name] for name in _TAKEN[1:])
    *stack, n, k = layer.weight.shape
    fmt = formats.get(mode)
    layout = None
    if fmt is not None and bits is not None and group_size is not None:
        layout = fmt.layout(k, n, bits, group_size)
    if layout is None:
        if _mlx_runs(mode):
            return None
        if fmt is None:
            raise ValueError(
                f"no quant format is registered for mode {mode!r}: install the "
                "plugin the checkpoint was written with"
            )
        raise ValueError(
            f"quant format {mode!r} lays out no ({n}, {k}) matrix of {bits} "
            f"bits in groups of {group_size}"
        )
    for name, value in arguments.items():
        if name not in _TAKEN and value != signature.parameters[name].default:
            raise ValueError(
                f"a layer of quant format {mode!r} is built from its bits and "
                f"group_size, not with {name}={value!r}"
            )
    tensors = {
        name: mx.zeros((*stack, *shape), dtype)
        for name, (shape, dtype) in layout.items()
    }
    return layer_cls(mode, bits, group_size, k, tensors, layer.get("bias"))


def _patch(cls, layer_cls) -> None:
    original = cls.to_quantized
    if getattr(original, "_omlx_format_load", False):
        return
    signature = inspect.signature(original)

    @wraps(original)
    def to_quantized(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        layer = _format_layer(self, layer_cls, signature, bound.arguments)
        return original(self, *args, **kwargs) if layer is None else layer

    to_quantized._omlx_format_load = True
    cls.to_quantized = to_quantized


def _wrapped():
    """Each class whose ``to_quantized`` this patch wraps, with the format
    layer it builds and the layer that MLX's own ``to_quantized`` builds."""
    yield nn.Linear, FormatLinear, nn.QuantizedLinear
    for name in _SWITCH_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        yield module.SwitchLinear, FormatSwitchLinear, module.QuantizedSwitchLinear


def loads_in_format(module_class: str) -> bool:
    """Whether a checkpoint's matrix that a module of ``module_class`` (its
    module and qualified name) holds loads as a format's layer: this patch
    wraps the class's ``to_quantized``, or the class is the layer that MLX
    builds of one it wraps (which a calibration over a quantized checkpoint
    meets)."""
    return any(
        module_class == f"{cls.__module__}.{cls.__qualname__}"
        for wrapped, _, built in _wrapped()
        for cls in (wrapped, built)
    )


def accept_checkpoint(model_path) -> None:
    """Hand each installed format that the checkpoint at ``model_path`` is
    written in the record oQ stored for it (``omlx_quant_formats`` in its
    config), before any layer is built: the format refuses a checkpoint it
    cannot read. A checkpoint written before formats had records hands an
    empty one."""
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return
    for mode, written in config.get("omlx_quant_formats", {}).items():
        fmt = formats.get(mode)
        if fmt is not None:
            fmt.accept(written.get("record", {}))


def apply_format_load_patch() -> None:
    """Have ``nn.Linear`` and the switch layers of mlx-lm and mlx-vlm load a
    registered format's mode as its layers. Idempotent."""
    formats.load()
    for cls, layer_cls, _ in _wrapped():
        _patch(cls, layer_cls)
