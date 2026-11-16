# SPDX-License-Identifier: Apache-2.0
"""A checkpoint's layers of a quant format's mode load as format layers.

mlx-lm and mlx-vlm rebuild a quantized checkpoint's modules with
``nn.quantize``: each layer the quantization config names is replaced by its
``to_quantized(group_size, bits, mode)``, and the checkpoint's tensors are
then loaded over the result's, strictly. MLX's layers build MLX's modes.
This patch has them build a layer of ``omlx.formats.linear`` for a mode whose
format lays the matrix out (``QuantFormat.layout``): the layer holds
placeholders of the checkpoint's shapes, which the load fills.

A format's layer is read with its checkpoint's record (``omlx_quant_formats``
in its config, which oQ writes). The libraries' functions that build a model
from a checkpoint's directory (``_LOADERS``) are wrapped: while one runs, the
format layers built on its thread belong to its checkpoint, each is built
with the object the format's ``accept`` returns for the record, and when it
returns every one of them is in the model it returns. A format's layer built
anywhere else is refused.

Each library function is wrapped once, the first time this patch finds it:
when the patch is applied, or when its module is imported after that (the
patch imports none of the libraries). A family's patch that replaces one of
them afterwards keeps the wrapped function under it: DeepSeek V4's loader
(``patches/deepseek_v4/utils_patch.py``, applied after this patch by
``maybe_apply_pre_load_patches``) calls it for any other model and builds a
DeepSeek V4 checkpoint's layers itself, outside any load, so that a
DeepSeek V4 checkpoint of a format is refused on every load; K2 Horizon's
(``patches/k2_horizon/checkpoint.py``) calls it with a directory of links to
the checkpoint's files, its config among them, whose record the layers read.
"""

from __future__ import annotations

import importlib.abc
import inspect
import json
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .. import formats
from ..formats import FormatRefusedError
from ..formats.linear import FormatLinear, FormatSwitchLinear

# The modules whose SwitchLinear holds a stack of matrices, (E, N, K).
_SWITCH_MODULES = ("mlx_lm.models.switch_layers", "mlx_vlm.models.switch_layers")

# The functions that build a model from a checkpoint's directory, their
# first argument, quantizing its layers: oMLX's loads build a format's
# layers inside one of them. (mlx-vlm's embedding loader calls the encoder
# loader by the name it imported it under.)
_LOADERS = (
    ("mlx_lm.utils", "load_model"),
    ("mlx_vlm.utils", "load_model"),
    ("mlx_vlm.embedding_loader", "load_encoder_model"),
    ("mlx_embeddings.utils", "load_model"),
)

# A value of config.json by its JSON type.
_JSON_TYPES = {
    list: "an array",
    str: "a string",
    int: "a number",
    float: "a number",
    bool: "a boolean",
    type(None): "null",
}

# What a format's layer is built from, of to_quantized's parameters.
_TAKEN = ("self", "mode", "bits", "group_size")

# The classes, by module and qualified name, whose matrix loads as a format's
# layer: those whose to_quantized this patch wraps.
_LOADS_IN_FORMAT = frozenset(
    (
        f"{nn.Linear.__module__}.{nn.Linear.__qualname__}",
        *(f"{module}.SwitchLinear" for module in _SWITCH_MODULES),
    )
)


class _Checkpoint:
    """A checkpoint being loaded: its directory, then, as its format layers
    are built, its config, the object that reads each of its formats and
    the layers built."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._config = None
        self._read: dict[str, formats.QuantFormat] = {}
        self.built: list[nn.Module] = []

    def read(self, fmt: formats.QuantFormat) -> formats.QuantFormat:
        """The object that reads this checkpoint's layers of ``fmt``'s mode:
        what ``fmt.accept`` returns for the record oQ stored (None: ``fmt``
        itself), asked once."""
        read = self._read.get(fmt.mode)
        if read is None:
            plugin = formats.origin(fmt.mode) or "the plugin"
            kept = dict(getattr(fmt, "__dict__", {}))
            try:
                read = fmt.accept(self._record(fmt.mode))
            except FormatRefusedError:
                raise
            except ValueError as error:
                # The format's own words.
                raise _refused(str(error), fmt.mode, self) from error
            now = getattr(fmt, "__dict__", {})
            changed = sorted(
                name
                for name in kept.keys() | now.keys()
                if kept.get(name, kept) is not now.get(name, kept)
            )
            if changed:
                raise _refused(
                    f"accept changed the registered format ({', '.join(changed)})"
                    " to accept the checkpoint: accept returns a format "
                    f"configured from the record instead; report it to {plugin}",
                    fmt.mode,
                    self,
                )
            read = fmt if read is None else read
            if not (
                isinstance(read, formats.QuantFormat)
                and (read.mode, read.tensors, read.contract)
                == (fmt.mode, fmt.tensors, fmt.contract)
            ):
                raise _refused(
                    f"accept returned {read!r}, not a format of its mode, "
                    f"tensors and contract: report it to {plugin}",
                    fmt.mode,
                    self,
                )
            self._read[fmt.mode] = read
        return read

    def _written(self, mode: str):
        """What the checkpoint's config keeps of ``mode``'s format, as it is
        (None for nothing), and the JSON path to it."""
        if self._config is None:
            self._config = json.loads((self.path / "config.json").read_text())
        kept, where = self._config, "config.json"
        for key in ("omlx_quant_formats", mode):
            if not isinstance(kept, dict):
                return kept, where
            kept, where = kept.get(key), f"{where}[{key!r}]"
        return kept, where

    def _record(self, mode: str) -> dict:
        """The record oQ stored for ``mode``; an empty one for a checkpoint
        written before formats had records, or by another writer."""
        written, where = self._written(mode)
        if written is None:
            return {}
        if isinstance(written, dict):
            written, where = written.get("record", {}), f"{where}['record']"
        if not isinstance(written, dict):
            raise _refused(
                f"{where} is {_JSON_TYPES[type(written)]}, where oQ writes "
                "an object: the checkpoint's config is damaged",
                mode,
                self,
            )
        return written

    def writer(self, mode: str) -> str:
        """The plugin the checkpoint's record says wrote ``mode``'s layers."""
        try:
            written, _ = self._written(mode)
        except (OSError, ValueError):
            written = None
        if not isinstance(written, dict) or "plugin" not in written:
            return "an unknown writer"
        plugin = written["plugin"]
        return plugin if isinstance(plugin, str) else "a format registered by hand"

    def check(self, loaded) -> None:
        """Refuse a load that returns ``loaded`` without each format layer
        built under it: those were another checkpoint's, built here."""
        if not self.built:
            return
        models = loaded if isinstance(loaded, tuple) else (loaded,)
        held = {
            id(module)
            for model in models
            if isinstance(model, nn.Module)
            for _, module in model.named_modules()
        }
        stray = [layer for layer in self.built if id(layer) not in held]
        if stray:
            raise _refused(
                f"{len(stray)} layers were built while the checkpoint loaded, "
                "and are not in its model: they were another checkpoint's, read "
                "with this one's record; load each checkpoint with the "
                "libraries' load_model",
                stray[0].mode,
                self,
            )


@contextmanager
def loading(path) -> Iterator[_Checkpoint]:
    """While the block runs, the format layers this thread builds belong to
    the checkpoint at ``path`` (a load nested in it has its own)."""
    checkpoint = _Checkpoint(path)
    token = formats._LOADING.set(checkpoint)
    try:
        yield checkpoint
    finally:
        formats._LOADING.reset(token)


def _refused(reason: str, mode, checkpoint=None) -> FormatRefusedError:
    """The refusal of ``mode``'s layers of ``checkpoint`` (the one loading
    on this thread when not given) for ``reason``."""
    if checkpoint is None:
        checkpoint = formats._LOADING.get()
    if checkpoint is None:
        return FormatRefusedError(reason, mode=mode)
    installed = formats.origin(mode) or (
        "a format registered by hand" if formats.get(mode) is not None else "none"
    )
    return FormatRefusedError(
        reason,
        mode=mode,
        path=checkpoint.path,
        written=checkpoint.writer(mode),
        installed=installed,
    )


def _not_installed(mode) -> str:
    """Why no format of ``mode`` reads a checkpoint's layers: its plugin
    did not load, or is not installed (the one the checkpoint being loaded
    records, when it records one)."""
    skipped = formats._SKIPPED.get(mode)
    if skipped is not None:
        return skipped
    checkpoint = formats._LOADING.get()
    written = checkpoint.writer(mode) if checkpoint is not None else None
    if written in (None, "an unknown writer", "a format registered by hand"):
        written = "the plugin the checkpoint was written with"
    return f"no format of this mode is installed: install {written}"


def _reading(fmt: formats.QuantFormat) -> _Checkpoint:
    """The checkpoint a layer of ``fmt`` built now belongs to."""
    checkpoint = formats._LOADING.get()
    if checkpoint is None:
        raise FormatRefusedError(
            "a layer is built outside a checkpoint's load, which gives it the "
            "record to be read with: load the checkpoint with mlx-lm's, "
            "mlx-vlm's or mlx-embeddings' load_model",
            mode=fmt.mode,
        )
    return checkpoint


def _format_layer(layer, layer_cls, signature, arguments):
    """The format layer that ``layer`` loads as, or None: MLX builds its
    mode. The registered format says whether it lays the matrix out; the
    checkpoint's (what ``accept`` returned) lays it out."""
    mode, bits, group_size = (arguments[name] for name in _TAKEN[1:])
    *stack, n, k = layer.weight.shape
    fmt = formats.get(mode)
    layout = None
    if fmt is not None and bits is not None and group_size is not None:
        layout = fmt.layout(k, n, bits, group_size)
    if layout is not None:
        checkpoint = _reading(fmt)
        read = checkpoint.read(fmt)
        if read is not fmt:
            layout = read.layout(k, n, bits, group_size)
        fmt = read
    if layout is None:
        if formats._mlx_runs(mode):
            return None
        if fmt is None:
            raise _refused(_not_installed(mode), mode)
        raise _refused(
            f"it lays out no ({n}, {k}) matrix of {bits} bits in groups of "
            f"{group_size}",
            mode,
        )
    for name, value in arguments.items():
        if name not in _TAKEN and value != signature.parameters[name].default:
            raise _refused(
                "a layer is built from its bits and group_size, not with "
                f"{name}={value!r}",
                mode,
            )
    tensors = {
        name: mx.zeros((*stack, *shape), dtype)
        for name, (shape, dtype) in layout.items()
    }
    built = layer_cls(fmt, bits, group_size, k, tensors, layer.get("bias"))
    checkpoint.built.append(built)
    return built


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


def loads_in_format(module_class: str) -> bool:
    """Whether a checkpoint's matrix that a module of ``module_class`` (its
    module and qualified name) holds loads as a format's layer: this patch
    wraps the class's ``to_quantized``."""
    return module_class in _LOADS_IN_FORMAT


def _wrap_loader(module, name: str) -> None:
    load = getattr(module, name)
    first = next(iter(inspect.signature(load).parameters))

    @wraps(load)
    def loader(*args, **kwargs):
        with loading(args[0] if args else kwargs[first]) as checkpoint:
            loaded = load(*args, **kwargs)
        checkpoint.check(loaded)
        return loaded

    loader._omlx_format_load = True
    setattr(module, name, loader)


# The modules whose functions or classes this patch wraps, by name; each is
# wrapped once, when this patch first finds it imported.
_PENDING = {name for name, _ in _LOADERS} | set(_SWITCH_MODULES)
_LOCK = threading.RLock()


def _wrap(module) -> None:
    with _LOCK:
        if module.__name__ not in _PENDING:
            return
        _PENDING.discard(module.__name__)
        for module_name, name in _LOADERS:
            if module_name == module.__name__:
                _wrap_loader(module, name)
        if module.__name__ in _SWITCH_MODULES:
            _patch(module.SwitchLinear, FormatSwitchLinear)


class _Then(importlib.abc.Loader):
    """A module's own loader, then this patch's wraps of what it defines."""

    def __init__(self, loader) -> None:
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        _wrap(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _OnImport(importlib.abc.MetaPathFinder):
    """Has the modules this patch wraps wrapped when they are imported."""

    def find_spec(self, name, path, target=None):
        if name not in _PENDING:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _Then(spec.loader)
                return spec
        return None


_FINDER = _OnImport()


def apply_format_load_patch() -> None:
    """Have ``nn.Linear`` and the switch layers of mlx-lm and mlx-vlm load a
    registered format's mode as its layers, each read with its checkpoint's
    record. Idempotent, and imports neither mlx-lm, mlx-vlm nor
    mlx-embeddings: what is not imported yet is wrapped when it is."""
    formats.load()
    _patch(nn.Linear, FormatLinear)
    with _LOCK:
        if _FINDER not in sys.meta_path:
            sys.meta_path.insert(0, _FINDER)
        for name in sorted(_PENDING):
            module = sys.modules.get(name)
            if module is not None:
                _wrap(module)
