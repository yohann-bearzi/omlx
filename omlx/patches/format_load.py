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
anywhere else is refused: a format's layer is only built in the load of a
checkpoint that passed the family gate (``formats._may_build``).

Each library function is wrapped once, the first time this patch finds it:
when the patch is applied, or when its module is imported after that (the
patch imports none of the libraries). It is wrapped in place, the function
object itself: a name it was imported under before is wrapped too. A
family's patch that replaces one of them afterwards keeps the wrapped
function under it: DeepSeek V4's loader
(``patches/deepseek_v4/utils_patch.py``, applied after this patch by
``maybe_apply_pre_load_patches``) calls it for any other model and loads a
DeepSeek V4 checkpoint itself inside a checkpoint's load (``loading``),
whose family gate refuses one of a format, building its layers outside it;
K2 Horizon's
(``patches/k2_horizon/checkpoint.py``) calls it with a directory of links to
the checkpoint's files, its config among them, whose record the layers read.
"""

from __future__ import annotations

import copy
import importlib
import importlib.abc
import inspect
import json
import os
import sys
import threading
import types
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property, update_wrapper, wraps
from importlib import _bootstrap
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .. import formats
from ..formats import FormatRefusedError, LoadHookError
from ..formats.linear import FormatLinear, FormatSwitchLinear

# The modules whose SwitchLinear holds a stack of matrices, (E, N, K).
_SWITCH_MODULES = ("mlx_lm.models.switch_layers", "mlx_vlm.models.switch_layers")

# The functions that build a model from a checkpoint's directory, their
# first argument, quantizing its layers, by the module that defines them:
# oMLX's loads build a format's layers inside one of them.
_LOADERS = (
    ("mlx_lm.utils", "load_model"),
    ("mlx_vlm.utils", "load_model"),
    ("mlx_vlm.encoder_loader", "load_encoder_model"),
    ("mlx_embeddings.utils", "load_model"),
)

# The parts of a model a load may leave out, whose layers its checkpoint's
# quantization config names all the same: an MTP head left off (mtp_enabled
# false), a text-only load's vision and audio towers.
_OPTIONAL_PARTS = frozenset(("mtp", "vision_tower", "visual", "audio_tower"))

# What a config holds nothing at (a key it has not), apart from a null.
_ABSENT = object()

# Whether this process has built a format layer: until it has, no load
# holds one to check.
_BUILT_ANY = False

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

    def __init__(self, path, *, by_loader: bool = False) -> None:
        self.path = Path(path).expanduser()
        # A load of it joins the outer one only on the thread that runs it,
        # while it runs, and when a library's loader opened it.
        self.owner = threading.get_ident()
        self.active = True
        self.by_loader = by_loader
        # Whether the family gate passed it for a format: only then may its
        # load build a format's layer (formats._may_build).
        self.verified = False
        # A view of links to a checkpoint's files (a family's loader's, a
        # decision model's) is named by the checkpoint its config is.
        self._config = _ABSENT
        self._read: dict[str, formats.QuantFormat] = {}
        # A load's threads (ones that copied its context) accept its record
        # once.
        self._accepting = threading.Lock()
        self.built: list[nn.Module] = []

    def read(self, fmt: formats.QuantFormat) -> formats.QuantFormat:
        """The object that reads this checkpoint's layers of ``fmt``'s mode:
        what ``accept`` of a copy of ``fmt`` returns for the record oQ stored
        (None: that copy), asked once."""
        with self._accepting:
            read = self._read.get(fmt.mode)
            return self._accepted(fmt) if read is None else read

    def _accepted(self, fmt: formats.QuantFormat) -> formats.QuantFormat:
        """``read`` the first time, under ``_accepting``."""
        # Whatever the config named: a format's layer is being built.
        _refuse_family(self, fmt.mode)
        plugin = formats.origin(fmt.mode) or "the plugin"
        record = self._record(fmt.mode)
        try:
            # The load's own copy: nothing accept does reaches the registered
            # format, which other loads and forwards read.
            own = copy.deepcopy(fmt)
        except Exception as error:
            raise _refused(
                f"the format cannot be copied for the load ({error}): accept runs "
                "on a copy of the registered format, which holds no lock, dtype "
                f"or other object that copy.deepcopy refuses; report it to {plugin}",
                fmt.mode,
                self,
            ) from error
        try:
            read = own.accept(record)
        except ValueError as error:
            # The format's own words, with what oMLX knows of the checkpoint.
            said = error.reason if isinstance(error, FormatRefusedError) else error
            raise _refused(str(said), fmt.mode, self) from error
        read = own if read is None else read
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

    @cached_property
    def shown(self) -> Path:
        """The checkpoint a refusal names (``_shown``), found when one is."""
        return _shown(self.path)

    def _read_config(self):
        """The checkpoint's config.json, read once (and its failure kept)."""
        if self._config is _ABSENT:
            try:
                self._config = json.loads((self.path / "config.json").read_text())
            except (OSError, ValueError) as error:
                # Kept with no traceback: it would hold the read's frames.
                self._config = error.with_traceback(None)
        if isinstance(self._config, Exception):
            # A copy raised, with no traceback piled on it by earlier raises
            # (another thread of the load may raise it meanwhile).
            raise copy.copy(self._config).with_traceback(None)
        return self._config

    def _written(self, mode: str):
        """What the checkpoint's config keeps of ``mode``'s format, as it is
        (``_ABSENT`` for nothing), and the JSON path to it."""
        kept, where = self._read_config(), "config.json"
        for key in ("omlx_quant_formats", mode):
            if not isinstance(kept, dict):
                return kept, where
            kept, where = kept.get(key, _ABSENT), f"{where}[{key!r}]"
        return kept, where

    def records(self, mode: str) -> bool:
        """Whether the checkpoint's config keeps anything of ``mode``'s format
        (or keeps its formats' record damaged): never for no checkpoint's
        directory, nor for a config that is no object (the library's to
        refuse)."""
        if isinstance(self._config, Exception):
            # Asked for each layer of a load: no raise for each.
            return False
        try:
            config = self._read_config()
        except (OSError, ValueError):
            return False
        if not isinstance(config, dict) or "omlx_quant_formats" not in config:
            return False
        written = config["omlx_quant_formats"]
        return not isinstance(written, dict) or mode in written

    def _record(self, mode: str) -> dict:
        """The record oQ stored for ``mode``; an empty one for a checkpoint
        written before formats had records, or by another writer."""
        written, where = self._written(mode)
        if written is _ABSENT:
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
            written = _ABSENT
        plugin = written.get("plugin", _ABSENT) if isinstance(written, dict) else _ABSENT
        if plugin is None:
            # oQ records a format registered by hand as null.
            return "a format registered by hand"
        if isinstance(plugin, str) and plugin.strip():
            return plugin
        return "an unknown writer"

    def check(self, loaded) -> None:
        """Refuse, whatever ``strict`` the load had, a load that returns
        ``loaded`` without each format layer built under it (another
        checkpoint's, built here); with a layer it filled in part, or with
        a tensor of another shape or dtype than the layout's; with an entry
        of its quantization config in a format's mode (one that is not MLX's,
        or that the checkpoint records) that bound no layer of its own, in a
        part of the model the load built; or with a record of formats that is
        no object. A layer it left wholly unfilled is a layer a partial load
        does not hold (a pipeline rank's, a prefill helper's)."""
        try:
            config = self._read_config()
        except (OSError, ValueError):
            config = None
        if isinstance(config, dict) and _damaged(config):
            # No layer read the record, which would have refused it.
            self._record(self._recorded_top_mode() or "unknown")
        entries = self._format_entries()
        top = self._recorded_top_mode()
        if top is not None and not any(layer.mode == top for layer in self.built):
            # Its layers are named by the config's mode, not each by an entry.
            raise _refused(
                f"its quantization config's mode is {top}, a format's, and the "
                "load built no layer of it: the model has no such layer, or its "
                "family's loader renames them",
                top,
                self,
            )
        if not self.built and not entries:
            # A load of no format's layers: an MLX checkpoint's.
            return
        models = loaded if isinstance(loaded, tuple) else (loaded,)
        paths: dict[int, list[str]] = {}
        for model in models:
            if isinstance(model, nn.Module):
                for path, module in model.named_modules():
                    paths.setdefault(id(module), []).append(path)
        stray = [layer for layer in self.built if id(layer) not in paths]
        if stray:
            raise _refused(
                f"{len(stray)} layers were built while the checkpoint loaded, "
                "and are not in its model: they were another checkpoint's, read "
                "with this one's record; load each checkpoint with the "
                "libraries' load_model",
                stray[0].mode,
                self,
            )
        # Each layer at each path its model holds it under, in its mode, for
        # one entry each.
        bound: dict[tuple[str, str], int] = {}
        for layer in self.built:
            path = paths[id(layer)][0]
            for held in paths[id(layer)]:
                key = (held, layer.mode)
                bound[key] = bound.get(key, 0) + 1
            placeholders = layer._placeholders[0]
            left = [name for name, held in placeholders.items() if layer[name] is held]
            if left and len(left) < len(placeholders):
                raise _refused(
                    f"the load filled {path} but its {', '.join(left)}: the "
                    "checkpoint holds no such tensor, or holds it under another "
                    "name",
                    layer.mode,
                    self,
                )
            for name, held in placeholders.items():
                tensor = layer[name]
                if not isinstance(tensor, mx.array):
                    raise _refused(
                        f"the load left {path}.{name} {type(tensor).__name__}, "
                        "where the format lays out an array: the checkpoint "
                        "holds tensors under that name",
                        layer.mode,
                        self,
                    )
                kept = (tuple(tensor.shape), tensor.dtype)
                if tensor is not held and kept != (tuple(held.shape), held.dtype):
                    raise _refused(
                        f"{path}.{name} is {kept[0]} {kept[1]}, where the format "
                        f"lays out {tuple(held.shape)} {held.dtype}",
                        layer.mode,
                        self,
                    )
        present = {path for held in paths.values() for path in held}
        for path, mode in entries:
            names = _entry_names(path)
            if _left_out(path, present):
                # Of a part this load does not build (an MTP head left off,
                # a text-only load's vision tower): no layer of it to serve.
                continue
            name = next((name for name in names if bound.get((name, mode))), None)
            if name is None:
                raise _refused(
                    f"its quantization config names {path}, and the load built "
                    "no layer of the format there: the model has no such "
                    "module, or its family's loader renames it",
                    mode,
                    self,
                )
            bound[(name, mode)] -= 1
        # Checked: the layers are the model's to keep, not this checkpoint's.
        self.built.clear()

    def _quantizations(self) -> list[dict]:
        """The quantization configs a library builds the checkpoint's layers
        from: its config's own, under either key (oQ writes both), or else
        its text model's."""
        try:
            config = self._read_config()
        except (OSError, ValueError):
            return []
        if not isinstance(config, dict):
            return []
        text = config.get("text_config")
        for holder in (config, text if isinstance(text, dict) else {}):
            found = _quantizations(holder, text=False)
            if found:
                return found
        return []

    def _in_format(self, mode) -> bool:
        """Whether a layer of ``mode`` is a format's: a mode that is not
        MLX's, or that the checkpoint records a format of."""
        return isinstance(mode, str) and (
            not formats._mlx_runs(mode) or self.records(mode)
        )

    def _recorded_top_mode(self) -> str | None:
        """The mode of a quantization config itself (MLX's affine when it
        names none), when it is a format's."""
        for quantization in self._quantizations():
            mode = quantization.get("mode", "affine")
            if self._in_format(mode):
                return mode
        return None

    def _format_entries(self) -> list[tuple[str, str]]:
        """The layers the checkpoint's quantization configs name in a mode
        that is a format's, with that mode. An entry without a mode is
        MLX's affine, whatever its config's mode (MLX builds it so)."""
        entries = {
            (path, entry.get("mode", "affine"))
            for quantization in self._quantizations()
            for path, entry in quantization.items()
            if isinstance(entry, dict)
        }
        return sorted((path, mode) for path, mode in entries if self._in_format(mode))


def check_loaded(model) -> None:
    """At the end of oMLX's own load of ``model``, once what this process does
    not hold is dropped (a pipeline rank's other layers, a prefill helper's
    later ones): refuse a format layer it still holds with a placeholder,
    which no tensor of its checkpoint filled. A filled layer lets its
    placeholders go once its load has ended."""
    if not _BUILT_ANY or not isinstance(model, nn.Module):
        # No process that built no format layer walks a model.
        return
    for name, module in model.named_modules():
        held = module.__dict__.get("_placeholders")
        if held is None:
            continue
        tensors, checkpoint = held
        left = [tensor for tensor, kept in tensors.items() if module[tensor] is kept]
        if left:
            # Kept: a check of it again (another thread's, a later caller's)
            # refuses it again.
            reason = (
                f"the loaded model holds {name} without its {', '.join(left)}: the "
                "checkpoint holds no such tensor, or holds it under another name"
            )
            raise _refused(reason, module.mode, checkpoint)
        if not checkpoint.active and not checkpoint.built:
            # Filled, and its load has ended and been checked (a load still
            # running, a library's loader around oMLX's, checks it when it
            # returns).
            module.__dict__.pop("_placeholders", None)


@contextmanager
def outside_load() -> Iterator[None]:
    """For the block, no checkpoint's load on this thread: a family's loader
    that builds its own checkpoints' layers by hand, which no format reads,
    runs its quantize in it, so that a checkpoint of a format is refused
    there whichever of its patch and this one was applied first."""
    token = formats._LOADING.set(None)
    try:
        yield
    finally:
        formats._LOADING.reset(token)


def _shown(path: Path) -> Path:
    """The checkpoint a refusal names for a load of ``path``: a view of links
    to a checkpoint's files (a family's loader's, a decision model's) by the
    checkpoint its config is, link by link while each link is to a config
    file (a Hugging Face snapshot's is to a blob of content, not one)."""
    path = Path(os.path.abspath(path))
    config, seen = path / "config.json", set()
    while config.is_symlink():
        stat = os.lstat(config)
        if (stat.st_dev, stat.st_ino) in seen:
            # A cycle of links.
            break
        seen.add((stat.st_dev, stat.st_ino))
        # Relative to the directory the link is in, not to a link to it.
        parent = os.path.realpath(config.parent)
        target = Path(os.path.normpath(os.path.join(parent, os.readlink(config))))
        if target.name != "config.json":
            break
        path, config = target.parent, target
    return path


def _named_modes(config: dict) -> list[str]:
    """The modes a checkpoint's config records a format of, or else names in
    its quantization configs (its text model's too) that are not MLX's."""
    written = config.get("omlx_quant_formats")
    if isinstance(written, dict) and written:
        return sorted(written)
    named = [
        mode
        for quantization in _quantizations(config)
        for mode in (
            quantization.get("mode"),
            *(
                entry.get("mode")
                for entry in quantization.values()
                if isinstance(entry, dict)
            ),
        )
    ]
    return [
        mode for mode in named if isinstance(mode, str) and not formats._mlx_runs(mode)
    ]


def _refuse_family(checkpoint: _Checkpoint, mode: str | None = None) -> None:
    """Refuse a checkpoint of a quant format (one it records, or names in its
    quantization config, or ``mode``, a format's layer being built for it)
    whose family is not verified: whoever wrote it (a plugin's own writer, a
    config edited by hand), the family's own code would read the format's
    layers wrongly. A load asks before the library reads a weight."""
    try:
        config = checkpoint._read_config()
    except (OSError, ValueError):
        # No checkpoint's directory, or the library's to refuse: no format's
        # layer is built for it.
        return
    if not isinstance(config, dict):
        return
    modes = [mode] if mode is not None else _named_modes(config)
    if not modes and _damaged(config):
        # A record of formats it keeps damaged: of modes no one knows.
        modes = ["unknown"]
    if not modes:
        return
    from .. import oq

    # Its config alone names its family, wherever the load found it.
    refused = oq._unverified_family(config)
    if refused is not None:
        family, why = refused
        fix = (
            "name its model type in its config"
            if family == "untyped"
            else "have the family verified"
        )
        raise _refused(
            f"oMLX loads no {family} checkpoint in a quant format: {why}; {fix}",
            modes[0],
            checkpoint,
        )
    checkpoint.verified = True


def _quantizations(config: dict, *, text: bool = True) -> list[dict]:
    """The quantization configs of ``config``, under either key, and (with
    ``text``) its text model's."""
    inner = config.get("text_config") if text else None
    return [
        quantization
        for holder in (config, inner if isinstance(inner, dict) else {})
        for key in ("quantization", "quantization_config")
        if isinstance(quantization := holder.get(key), dict)
    ]


def _entry_names(path: str) -> tuple[str, ...]:
    """The paths a layer an entry names is built at: mlx-vlm builds a
    language model's checkpoint under language_model., mlx-lm a VLM's text
    model without it, and a Hugging Face config names model.language_model.
    what mlx-vlm builds at language_model.model."""
    plain = path.removeprefix("language_model.")
    names = [path, f"language_model.{path}", plain]
    if path.startswith("model.language_model."):
        inner = path.removeprefix("model.language_model.")
        names += [f"language_model.model.{inner}", f"model.{inner}"]
    return tuple(dict.fromkeys(names))


def _left_out(path: str, present: set[str]) -> bool:
    """Whether a layer an entry names at ``path`` is in a part of a model
    that a load may leave out (``_OPTIONAL_PARTS``), its own or its language
    model's, and that the model whose modules are at ``present`` does not
    hold."""
    root = path.removeprefix("language_model.").split(".", 1)[0]
    return root in _OPTIONAL_PARTS and not (
        root in present or f"language_model.{root}" in present
    )


def _damaged(config: dict) -> bool:
    """Whether ``config`` keeps a record of formats that is no object."""
    written = config.get("omlx_quant_formats", _ABSENT)
    return written is not _ABSENT and not isinstance(written, dict)


def refuse_family(path, family: str) -> None:
    """Refuse the checkpoint at ``path`` when its config records or names a
    quant format and a loader of oMLX's own serves it as ``family`` (a
    decision model's, which its config's model type does not name) that is
    not verified: that loader reads its layers by hand. Called before the
    loader reads a weight."""
    checkpoint = _Checkpoint(path)
    try:
        config = checkpoint._read_config()
    except (OSError, ValueError):
        # The loader's to refuse.
        return
    if not isinstance(config, dict):
        return
    modes = _named_modes(config) or (["unknown"] if _damaged(config) else [])
    if not modes:
        return
    from .. import oq

    if family in oq._QUANT_FORMAT_VERIFIED | oq._TEST_VERIFIED:
        return
    why = oq._QUANT_FORMAT_MISREAD.get(family, oq._NOT_VERIFIED)
    raise _refused(
        f"oMLX loads no {family} checkpoint in a quant format: {why}; have the "
        "family verified",
        modes[0],
        checkpoint,
    )


def _current() -> _Checkpoint | None:
    """The checkpoint whose load is running and whose layers this thread
    builds: a thread that copied a load's context after it ended (or another
    thread's) builds none of its layers."""
    checkpoint = formats._LOADING.get()
    return checkpoint if checkpoint is not None and checkpoint.active else None


@contextmanager
def loading(path, *, by_loader: bool = False) -> Iterator[_Checkpoint]:
    """While the block runs, the format layers built on this thread (and on
    threads that copied its context) belong to the checkpoint at ``path``: a
    load nested in it has its own, but one of the same checkpoint (a view of
    links to it, a loader wrapped over the wrapped one) on the same thread is
    the outer one, when a library's loader (``by_loader``) opened that."""
    outer = _current()
    if (
        outer is not None
        and outer.by_loader
        and outer.owner == threading.get_ident()
        and outer.shown == _shown(Path(path).expanduser())
    ):
        yield outer
        return
    checkpoint = _Checkpoint(path, by_loader=by_loader)
    _refuse_family(checkpoint)
    token = formats._LOADING.set(checkpoint)
    try:
        yield checkpoint
    finally:
        checkpoint.active = False
        formats._LOADING.reset(token)


def _refused(reason: str, mode, checkpoint=None) -> FormatRefusedError:
    """The refusal of ``mode``'s layers of ``checkpoint`` (the one loading
    on this thread when not given) for ``reason``."""
    if checkpoint is None:
        checkpoint = _current()
    if checkpoint is None:
        return FormatRefusedError(reason, mode=mode)
    installed = formats.origin(mode)
    if installed is None:
        registered = formats.get(mode)
        if registered is None:
            installed = "none"
        elif mode in formats._BUILTIN_MODES:
            installed = "built-in"
        else:
            installed = "a format registered by hand"
    return FormatRefusedError(
        reason,
        mode=mode,
        path=checkpoint.shown,
        written=checkpoint.writer(mode),
        installed=installed,
    )


def _recorded(mode) -> bool:
    """Whether the checkpoint loading on this thread records a format of
    ``mode`` in its config."""
    checkpoint = _current()
    return checkpoint is not None and checkpoint.records(mode)


def _not_installed(mode) -> str:
    """Why no format of ``mode`` reads a checkpoint's layers: its plugin
    did not load, or is not installed (the one the checkpoint being loaded
    records, when it records one)."""
    skipped = formats._SKIPPED.get(mode)
    if skipped is not None:
        return skipped
    checkpoint = _current()
    written = checkpoint.writer(mode) if checkpoint is not None else None
    if written in (None, "an unknown writer", "a format registered by hand"):
        written = "the plugin the checkpoint was written with"
    return f"no format of this mode is installed: install {written}"


def _unwrapped_loader() -> str | None:
    """The library loader running on this thread's stack, by its module's
    name, outside any load: one this patch has not wrapped."""
    frame = sys._getframe(1)
    while frame is not None:
        found = (frame.f_globals.get("__name__"), frame.f_code.co_name)
        if found in _LOADERS:
            caller = frame.f_back
            while caller is not None:
                if caller.f_code in _LOADER_CODES:
                    # Under this patch's loader (a family's loader between):
                    # the layer was built outside its load (outside_load).
                    return None
                caller = caller.f_back
            return ".".join(found)
        frame = frame.f_back
    return None


def _reading(fmt: formats.QuantFormat) -> _Checkpoint:
    """The checkpoint a layer of ``fmt`` built now belongs to."""
    checkpoint = _current()
    if checkpoint is None:
        loader = _unwrapped_loader()
        if loader is not None:
            raise FormatRefusedError(
                f"{loader} was imported before oMLX's patches were applied "
                "and runs unwrapped (a family's loader took its name in its "
                f"module first): call {loader} through its module, or apply "
                "the patches first (maybe_apply_pre_load_patches)",
                mode=fmt.mode,
            )
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
        if not isinstance(mode, str):
            # MLX's to refuse.
            return None
        recorded = _recorded(mode)
        if not recorded and formats._mlx_runs(mode):
            return None
        if recorded:
            # A damaged record is refused as such.
            _current()._record(mode)
        if fmt is None:
            # No format reads it: one of MLX's modes too when the checkpoint
            # records a format of it (written before MLX's modes were the
            # built-ins' only, or by a newer oMLX).
            raise _refused(_not_installed(mode), mode)
        # The registered format lays out no such matrix: one of MLX's modes
        # (affine) that the checkpoint records a format of too.
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
    # Its placeholders and its checkpoint, until oMLX's own load has checked
    # them (check_loaded).
    object.__setattr__(built, "_placeholders", (tensors, checkpoint))
    global _BUILT_ANY
    _BUILT_ANY = True
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


def _through(*args, _omlx_loader, **kwargs):
    """A wrapped loader's code: the patch's loader, for every caller."""
    return _omlx_loader(*args, **kwargs)


# The code of the loaders this patch puts in a module's name.
_LOADER_CODES: set = set()


def _wrapped_already(load) -> bool:
    """Whether ``load``, or a function it wraps, is this patch's loader: by
    ``__wrapped__`` (functools.wraps, K2 Horizon's) or ``_omlx_original``
    (MiniMax M3's, DeepSeek V4's, which loads its own checkpoints in a load
    of its own). One over the library's own, unwrapped, is wrapped by
    name."""
    seen, pending = set(), [load]
    while pending:
        load = pending.pop()
        if load is None or id(load) in seen:
            continue
        seen.add(id(load))
        code = getattr(load, "__code__", None)
        if code is _through.__code__ or code in _LOADER_CODES:
            return True
        pending += [
            getattr(load, "__wrapped__", None),
            getattr(load, "_omlx_original", None),
        ]
    return False


def _wrap_loader(module, name: str) -> None:
    """Wrap the function ``module.name``. The library's own is wrapped in
    place: the function object takes ``_through``'s code, so that a name it
    was imported under before the patch (``from mlx_lm.utils import
    load_model``) calls the patch's loader too, and the loader runs a copy
    of it. A closure's code cannot be swapped (a family's loader put over
    the library's before the patch): the module's name for it is."""
    load = getattr(module, name)
    if not isinstance(load, types.FunctionType):
        raise LoadHookError(
            f"{module.__name__}.{name} is a {type(load).__name__}, not a Python "
            "function: oMLX's load hook cannot wrap it; this version of the "
            "library needs omlx/patches/format_load.py updated for it"
        )
    if _wrapped_already(load):
        # The module found again, once more pending: its loader is this
        # patch's, or a family's put over it (K2 Horizon's).
        return
    in_place = load.__closure__ is None
    own = load
    if in_place:
        own = types.FunctionType(
            load.__code__, load.__globals__, load.__name__, load.__defaults__
        )
        own.__kwdefaults__ = load.__kwdefaults__
        # Its name, text and attributes, and none of the patch's marks.
        update_wrapper(own, load, updated=())
        own.__dict__ = dict(load.__dict__)
    first = next(iter(inspect.signature(own).parameters))

    def loader(*args, **kwargs):
        outer = _current()
        path = args[0] if args else kwargs[first]
        with loading(path, by_loader=True) as checkpoint:
            loaded = own(*args, **kwargs)
        if checkpoint is not outer:
            # A load joined to an outer one of the checkpoint is checked by
            # that one, when it returns.
            checkpoint.check(loaded)
        return loaded

    _LOADER_CODES.add(loader.__code__)
    if not in_place:
        update_wrapper(loader, load)
        loader._omlx_format_load = True
        setattr(module, name, loader)
        return
    # In this order, so that a call on another thread meanwhile runs the
    # function whole, its own code or _through's: what reads its signature
    # reads the copy's; the defaults keep the function's own keyword-only
    # ones, which _through's code ignores, and _through's, which its own
    # code ignores.
    load.__wrapped__ = own
    load.__kwdefaults__ = {**(load.__kwdefaults__ or {}), "_omlx_loader": loader}
    load.__code__ = _through.__code__
    load._omlx_format_load = True


# The modules whose functions or classes this patch wraps, by name; each is
# wrapped once, when this patch first finds it imported.
_PENDING = {name for name, _ in _LOADERS} | set(_SWITCH_MODULES)
_LOCK = threading.RLock()


def _wrap(module) -> None:
    with _LOCK:
        if module.__name__ not in _PENDING:
            return
        try:
            for module_name, name in _LOADERS:
                if module_name == module.__name__:
                    _wrap_loader(module, name)
            if module.__name__ in _SWITCH_MODULES:
                _patch(module.SwitchLinear, FormatSwitchLinear)
        except AttributeError:
            # Not defined yet: its import is still running, and the end of
            # it wraps the module (or the next application of the patch).
            return
        _PENDING.discard(module.__name__)


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
        pending = list(_PENDING)
    for name in pending:
        # An import of it on another thread may have begun before this patch
        # could see it, even before the module is in sys.modules: it is
        # waited for by the module's import lock (outside this patch's lock,
        # which the end of that import takes), and what it left is wrapped.
        # One that failed left nothing, and is not imported again here. This
        # thread's own import of it holds the lock already: it is wrapped when
        # it ends, or by the next application of the patch.
        try:
            with _bootstrap._ModuleLockManager(name):
                module = sys.modules.get(name)
        except _bootstrap._DeadlockError:
            # The thread importing it waits for one this thread imports: its
            # import's end wraps it (or the next application of the patch).
            module = sys.modules.get(name)
        if module is not None:
            _wrap(module)
