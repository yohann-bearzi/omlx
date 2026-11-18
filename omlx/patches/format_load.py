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
from ..model_discovery import VERIFICATION

# The modules whose SwitchLinear holds a stack of matrices, (E, N, K).
_SWITCH_MODULES = ("mlx_lm.models.switch_layers", "mlx_vlm.models.switch_layers")

# The functions that build a model from a checkpoint's directory, their
# first argument, quantizing its layers, by the module that defines them:
# oMLX's loads build a format's layers inside one of them.
_CHECKED_LOADERS = (
    ("mlx_lm.utils", "load_model"),
    ("mlx_vlm.utils", "load_model"),
)
_LOADERS = (
    *_CHECKED_LOADERS,
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

    def __init__(self, path, *, by_loader: bool = False, config=_ABSENT) -> None:
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
        self._config = config
        self._read: dict[str, formats.QuantFormat] = {}
        self._layouts = {}
        # A load's threads (ones that copied its context) accept its record
        # once.
        self._accepting = threading.RLock()
        self.built: list[nn.Module] = []

    def read(self, fmt: formats.QuantFormat) -> formats.QuantFormat:
        """The object that reads this checkpoint's layers of ``fmt``'s mode:
        what ``accept`` of a copy of ``fmt`` returns for the record oQ stored
        (None: that copy), asked once."""
        with self._accepting:
            return self._accepted(fmt)

    def layout(self, fmt, geometry):
        """The registered and accepted formats must both support this geometry."""
        key = (fmt.mode, *geometry)
        result = self._layouts.get(key)
        if result is not None:
            return result
        with self._accepting:
            if key not in self._layouts:
                layout = fmt.layout(*geometry)
                read = fmt
                if layout is not None:
                    read = self.read(fmt)
                    if read is not fmt:
                        layout = read.layout(*geometry)
                self._layouts[key] = (read, layout)
            return self._layouts[key]

    def _accepted(self, fmt: formats.QuantFormat) -> formats.QuantFormat:
        """The checkpoint's reader of this mode, under ``_accepting``."""
        read = self._read.get(fmt.mode)
        if read is not None:
            return read
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

    @cached_property
    def _declaration(self):
        try:
            config = self._read_config()
        except (OSError, ValueError):
            return None, None
        return _format_declaration(config)

    def _marked_mode(self) -> str | None:
        mode, misplaced = self._declaration
        if misplaced:
            raise _refused(f"unknown format layout at {misplaced}", mode, self)
        return mode

    def apply_model_config(self, override: dict) -> None:
        """Check mlx-lm's shallow config update before it opens a weight file."""

        disk = self._read_config()
        effective = disk | override
        recorded = self._marked_mode()
        supplied = _Checkpoint(self.path, config=effective)
        supplied_mode = supplied._marked_mode()
        mode = recorded or supplied_mode
        if mode is None:
            return
        _refuse_family(supplied, mode)
        if recorded is None:
            # A format introduced by the caller also has to belong to the
            # on-disk family. Its record must reach accept and the outer check.
            _refuse_family(self, mode)
            self._config = effective
            self._declaration = supplied._declaration
            return

        # The loader may spell a layer under its runtime path as well. Keep the
        # disk config for accept and for counting the original entries once.
        for prefix, before, after in (
            ("", disk, effective),
            (
                "text_config.",
                disk.get("text_config", {}),
                effective.get("text_config", {}),
            ),
        ):
            if before is after:
                continue
            if not isinstance(before, dict) or not isinstance(after, dict):
                raise _refused("unknown quantization layout at text_config", mode, self)
            for field in ("omlx_quant_formats", "quantization", "quantization_config"):
                old = before.get(field, _ABSENT)
                new = after.get(field, _ABSENT)
                if old is new:
                    continue
                if field == "omlx_quant_formats":
                    kept = _same_json(old, new)
                else:
                    kept = _keeps_layer_map(old, new)
                if not kept:
                    raise _refused(
                        f"model_config changes the checkpoint's {prefix}{field}",
                        mode,
                        self,
                    )

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
            for field, value in written.items():
                if field == "record":
                    continue  # Checked below; the format owns the object's contents.
                if field == "plugin" and (value is None or isinstance(value, str)):
                    continue
                if (
                    field == "contract"
                    and type(value) is int
                    and formats.OLDEST_CONTRACT <= value <= formats.CONTRACT
                ):
                    continue
                raise _refused(
                    f"unknown format record layout at {where}[{field!r}]", mode, self
                )
            written, where = written.get("record", {}), f"{where}['record']"
        if not isinstance(written, dict):
            raise _refused(
                f"{where} is {_JSON_TYPES[type(written)]}, where oQ writes "
                "an object: the checkpoint's config is damaged",
                mode,
                self,
            )
        return written

    def refuse_unverified(self) -> None:
        """Refuse the checkpoint, before a weight is read, when a run in a
        quant format wrote it and its verification did not pass: the run
        did not complete, or its checks failed."""
        try:
            if self._config is None:
                self._config = json.loads((self.path / "config.json").read_text())
        except OSError:
            # A loader handed what is no checkpoint's directory.
            return
        config = self._config
        record = config.get(VERIFICATION) if isinstance(config, dict) else None
        if isinstance(record, dict) and record.get("verified") is not True:
            raise _refused(
                f"not verified: {record.get('reason')}", record.get("mode"), self
            )

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
        config = config if isinstance(config, dict) else {}
        mode = self._marked_mode()
        if mode is None and not self.built:
            return
        quants = _quantizations(config)
        if _damaged(config):
            self._record(mode or "unknown")
        if mode is not None:
            self._check_layout(config, mode)
        recorded = config.get("omlx_quant_formats", {})
        modes = {layer.mode for layer in self.built}
        for mode in recorded:
            self._record(mode)
        entries = self._format_entries(quants)
        for quant in quants:
            top = quant.get("mode", "affine")
            if not self._in_format(top) or top in modes:
                continue
            # Its layers are named by the config's mode, not each by an entry.
            raise _refused(
                f"its quantization config's mode is {top}, a format's, and the "
                "load built no layer of it: the model has no such layer, or its "
                "family's loader renames them",
                top,
                self,
            )
        # A record with no built layer is accounted for only by named
        # entries. Each of those must still bind or belong to an absent part.
        accounted = modes.union(mode for _, mode in entries)
        for mode in recorded:
            if mode in accounted:
                continue
            raise _refused(
                "its config records the format but the load built no layer of it",
                mode,
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
            placeholders = layer._load_origin[0]
            # A tensor of another shape or dtype is not taken (the layer keeps
            # its placeholder) and recorded as offered: named as it was.
            for name, offered in getattr(layer, "_placeholders", {}).items():
                if offered is not None:
                    held = placeholders[name]
                    raise _refused(
                        f"{path}.{name} is {offered[0]} {offered[1]}, where the "
                        f"format lays out {tuple(held.shape)} {held.dtype}",
                        layer.mode,
                        self,
                    )
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
        # A list or dict of modules has named children but no module at
        # the container's own path (e.g. mtp.0 exists without mtp).
        present_parts = {
            _optional_root(path) for held in paths.values() for path in held
        }
        for path, mode in entries:
            names = _entry_names(path)
            part = _optional_root(path)
            if part and part not in present_parts:
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

    def _check_layout(self, config: dict, mode: str) -> None:
        """Validate the flat per-layer maps of a checkpoint declaring a format."""
        recorded = config.get("omlx_quant_formats", _ABSENT)
        if isinstance(recorded, dict) and not (
            recorded and all(isinstance(k, str) and k.strip() for k in recorded)
        ):
            raise _refused("unknown format layout at omlx_quant_formats", mode, self)
        text = config.get("text_config", {})
        if not isinstance(text, dict):
            raise _refused("unknown quantization layout at text_config", mode, self)
        for prefix, holder in (("", config), ("text_config.", text)):
            for key in ("quantization", "quantization_config"):
                if key not in holder:
                    continue
                quant = holder[key]
                where = prefix + key
                if not isinstance(quant, dict):
                    raise _refused(
                        f"unknown quantization layout at {where}", mode, self
                    )
                for path, entry in quant.items():
                    field = f"{where}.{path}"
                    if path in ("group_size", "bits", "mode"):
                        valid = _layout_field(path, entry)
                    else:
                        valid = (
                            all(path.split("."))
                            and isinstance(entry, dict)
                            and bool(entry)
                            and all(_layout_field(k, v) for k, v in entry.items())
                        )
                    if not valid:
                        raise _refused(
                            f"unknown quantization layout at {field}", mode, self
                        )

    def _in_format(self, mode) -> bool:
        """Whether a layer of ``mode`` is a format's: a mode that is not
        MLX's, or that the checkpoint records a format of."""
        return isinstance(mode, str) and (
            not formats._mlx_runs(mode) or self.records(mode)
        )

    def _format_entries(self, quants: list[dict]) -> list[tuple[str, str]]:
        """The layers the checkpoint's quantization configs name in a mode
        that is a format's, with that mode. An entry without a mode is
        MLX's affine, whatever its config's mode (MLX builds it so)."""
        entries = {
            (path, entry.get("mode", "affine"))
            for quantization in quants
            for path, entry in quantization.items()
            if isinstance(entry, dict) and self._in_format(entry.get("mode", "affine"))
        }
        return sorted(entries)


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
        held = module.__dict__.get("_load_origin")
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
            module.__dict__.pop("_load_origin", None)


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


_QUANT_KEYS = ("quantization", "quantization_config")


def _format_markers(value, path=(), in_quant=False):
    """Yield JSON locations of format records and non-MLX quantization modes."""
    fields = value.items() if isinstance(value, dict) else enumerate(value)
    for key, child in fields:
        if key == "omlx_quant_formats":
            names = (
                sorted(k for k in child if isinstance(k, str) and k.strip())
                if isinstance(child, dict)
                else []
            )
            yield (*path, key), names[0] if names else None
            # The format owns its record, including any similarly named fields.
        elif in_quant and key == "mode" and isinstance(child, str):
            if not formats._mlx_runs(child):
                yield (*path, key), child if child.strip() else "unknown"
        elif isinstance(child, (dict, list)):
            yield from _format_markers(
                child, (*path, key), in_quant or key in _QUANT_KEYS
            )


def _format_declaration(config) -> tuple[str | None, str | None]:
    """Return the declared mode and the first unsupported declaration location."""
    if not isinstance(config, (dict, list)):
        return None, None
    explicit = implicit = None
    for path, mode in _format_markers(config):
        if path == ("omlx_quant_formats",):
            explicit = mode
            continue
        relative = path[1:] if path[0] == "text_config" else path
        if (
            relative[0] in _QUANT_KEYS
            and relative[-1] == "mode"
            and len(relative) in (2, 3)
            and isinstance(relative[-2], str)
        ):
            implicit = implicit or mode
        else:
            return mode or "unknown", ".".join(map(str, path))
    name = explicit or implicit
    if name is None and isinstance(config, dict) and "omlx_quant_formats" in config:
        name = "unknown"
    return name, None


def _format_mode(config) -> str | None:
    """Admission to native readers depends only on the presence of a format."""
    return _format_declaration(config)[0]


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
    declared = checkpoint._marked_mode()
    if not isinstance(config, dict):
        return
    if mode is None:
        mode = declared
    if mode is None:
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
            mode,
            checkpoint,
        )
    checkpoint.verified = True


def _quantizations(config: dict) -> list[dict]:
    """Account for declarations under both spellings in both supported holders."""
    inner = config.get("text_config")
    return [
        quantization
        for holder in (config, inner if isinstance(inner, dict) else {})
        for key in _QUANT_KEYS
        if isinstance(quantization := holder.get(key), dict)
    ]


def _same_json(left, right) -> bool:
    """Compare on-disk JSON values, including the types of plugin metadata."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(map(_same_json, left, right))
    return left == right


def _keeps_layer_map(before, after) -> bool:
    """Only a second spelling of an existing layer may be added by mlx-lm."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    extra = after.keys() - before.keys()
    for path, entry in before.items():
        if not _same_json(entry, after.get(path, _ABSENT)):
            return False
        if extra and isinstance(entry, dict):
            for alias in extra.intersection(_entry_names(path)):
                if _same_json(entry, after[alias]):
                    extra.remove(alias)
    return not extra


def _layout_field(key, value) -> bool:
    if key == "mode":
        return isinstance(value, str) and bool(value.strip())
    return key in ("bits", "group_size") and type(value) is int and value > 0


# Spellings used in HF checkpoints, mlx-lm text models, and mlx-vlm.
_OPTIONAL_PREFIXES = (
    "model.language_model.",
    "language_model.model.",
    "language_model.",
    "model.",
    "",
)


def _optional_path(path: str) -> str:
    for prefix in _OPTIONAL_PREFIXES[:-1]:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _optional_root(path: str) -> str:
    root = _optional_path(path).partition(".")[0]
    if root == "visual":
        return "vision_tower"
    return root if root in _OPTIONAL_PARTS else ""


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
    if _optional_root(path):
        inner = _optional_path(path)
        root, dot, rest = inner.partition(".")
        roots = (
            ("visual", "vision_tower")
            if root in ("visual", "vision_tower")
            else (root,)
        )
        names += [
            f"{prefix}{root}{dot}{rest}"
            for root in roots
            for prefix in _OPTIONAL_PREFIXES
        ]
    return tuple(dict.fromkeys(names))


def _damaged(config: dict) -> bool:
    """Whether ``config`` keeps a record of formats that is no object."""
    written = config.get("omlx_quant_formats", _ABSENT)
    return written is not _ABSENT and not isinstance(written, dict)


def refuse_format_checkpoint(path, loader: str, *, config=_ABSENT, reason=None) -> None:
    """Reject format weights before a reader outside the checked path opens them."""
    checkpoint = _Checkpoint(path, config=config)
    mode, _ = checkpoint._declaration
    if mode is not None:
        raise _refused(
            reason or f"the {loader} loader cannot read a quant format checkpoint",
            mode,
            checkpoint,
        )


def check_engine_loader(path, loader: str) -> None:
    """Only the in-process text and VLM loaders can serve a verified checkpoint."""
    if loader in ("batched", "vlm"):
        _refuse_family(_Checkpoint(path))
    else:
        refuse_format_checkpoint(path, loader)


def refuse_family(path, family: str) -> None:
    """Decision models read their output heads directly, outside the checked load."""
    from .. import oq

    why = oq._QUANT_FORMAT_MISREAD.get(family, oq._NOT_VERIFIED)
    refuse_format_checkpoint(
        path,
        family,
        reason=f"oMLX loads no {family} checkpoint in a quant format: {why}; "
        "have the family verified",
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
    checkpoint.refuse_unverified()
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


def _outside_checkpoint(fmt: formats.QuantFormat) -> None:
    """Refuse building a supported format outside a library checkpoint load."""
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
        "record to be read with: load the checkpoint with mlx-lm's or "
        "mlx-vlm's load_model",
        mode=fmt.mode,
    )


def _format_layer(layer, layer_cls, signature, arguments):
    """The format layer that ``layer`` loads as, or None: MLX builds its
    mode. The registered format says whether it lays the matrix out; the
    checkpoint's (what ``accept`` returned) lays it out."""
    mode, bits, group_size = (arguments[name] for name in _TAKEN[1:])
    *stack, n, k = layer.weight.shape
    fmt = formats.get(mode)
    layout = None
    if fmt is not None and bits is not None and group_size is not None:
        checkpoint = _current()
        geometry = (k, n, bits, group_size)
        if checkpoint is not None:
            fmt, layout = checkpoint.layout(fmt, geometry)
        elif fmt.layout(*geometry) is not None:
            _outside_checkpoint(fmt)
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
    try:
        # Against the matrix's own N, before a placeholder is made: a layout
        # packed along N is refused here, not built over half its rows.
        formats.check_layout(fmt, n, k, layout)
    except ValueError as error:
        raise _refused(str(error), mode) from error
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
    built = layer_cls(
        fmt, bits, group_size, k, tensors, layer.get("bias"), placeholders=True
    )
    checkpoint.built.append(built)
    # Its placeholders and its checkpoint, until oMLX's own load has checked
    # them (check_loaded).
    object.__setattr__(built, "_load_origin", (tensors, checkpoint))
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
    signature = inspect.signature(own)
    first = next(iter(signature.parameters))
    checked = (module.__name__, name) in _CHECKED_LOADERS

    def loader(*args, **kwargs):
        outer = _current()
        path = args[0] if args else kwargs[first]
        if not checked:
            role = f"{module.__name__}.{name}"
            refuse_format_checkpoint(path, role)
            # Embeddings accepts model_config positionally. The encoder can
            # replace the disk config and then apply a separate override.
            supplied = signature.bind_partial(*args, **kwargs).arguments
            for key in ("model_config", "config", "config_overrides"):
                override = supplied.get(key, kwargs.get(key))
                if override is not None:
                    refuse_format_checkpoint(path, role, config=override)
            return own(*args, **kwargs)
        with loading(path, by_loader=True) as checkpoint:
            if module.__name__ == "mlx_lm.utils":
                supplied = signature.bind_partial(*args, **kwargs).arguments
                override = supplied.get("model_config", kwargs.get("model_config"))
                if override:
                    checkpoint.apply_model_config(override)
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
