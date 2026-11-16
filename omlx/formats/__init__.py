# SPDX-License-Identifier: Apache-2.0
"""Weight formats: how a quantized linear's tensors decode, as a unit that
oMLX's kernels are instantiated with.

A format is one ``QuantFormat``, registered under the ``mode`` its linears
carry in a checkpoint's quantization config. It gives Metal *faces*: small
structs that decode its tensors. oMLX's kernels are skeletons over a face
(the lane walk, the launch shape, the fused activation and combine belong to
the kernel; the loads and the arithmetic of one weight run belong to the
format), so a kernel tuned for one format runs every format that has the
face, and a format written once runs in every such kernel. A kernel is
compiled once per face, with the face inlined into it.

Formats come from oMLX itself (``BUILTIN``) and from installed distributions
that declare an entry point in the ``omlx.quant_formats`` group:

    [project.entry-points."omlx.quant_formats"]
    myformat = "my_package:FORMAT"

Both register through ``register``. ``docs/quant-formats.md`` is the contract.
It has a revision number (``CONTRACT``), and a format states the revision it
is written for.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points

import mlx.core as mx

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "omlx.quant_formats"

# Formats that ship with oMLX, as entry-point values ("module:attribute").
BUILTIN: tuple[str, ...] = ("omlx.formats.affine:FORMAT",)

# The revision of the contract (docs/quant-formats.md, Versions) that this
# oMLX implements, and the oldest one whose formats it still runs.
CONTRACT = 1
OLDEST_CONTRACT = 1


@dataclass(frozen=True)
class _Face:
    """What every kind of face is: Metal, the type it declares, the tensors
    it reads and the activations it is exact for. A face of one kind never
    equals a face of another."""

    key: str  # names the instance in kernel names and compiler messages
    source: str
    type: str
    tensors: tuple[str, ...]
    dtypes: tuple[mx.Dtype, ...] = ()  # the activations it is exact for; any if empty

    def __post_init__(self) -> None:
        if not (isinstance(self.tensors, tuple) and isinstance(self.dtypes, tuple)):
            raise ValueError(
                f"quant format face {self.key!r}: tensors and dtypes are tuples"
            )
        # Both are spliced into Metal identifiers.
        for name in (self.key, *self.tensors):
            if not (isinstance(name, str) and name.isascii() and name.isidentifier()):
                raise ValueError(
                    f"quant format face {self.key!r}: {name!r} is not an identifier"
                )
        if "weight" in self.tensors or len(set(self.tensors)) != len(self.tensors):
            raise ValueError(
                f"quant format face {self.key!r}: tensors are the parameters "
                f"beside 'weight', each named once, not {self.tensors!r}"
            )
        dtypes = {dtype for dtype in self.dtypes if isinstance(dtype, mx.Dtype)}
        if len(dtypes) != len(self.dtypes):
            raise ValueError(
                f"quant format face {self.key!r}: dtypes are distinct mx.Dtype, "
                f"not {self.dtypes!r}"
            )

    def exact_for(self, dtype) -> bool:
        """Whether the face computes its format's rows for ``dtype``
        activations."""
        return not self.dtypes or dtype in self.dtypes


@dataclass(frozen=True)
class Dot(_Face):
    """A format's mat-vec face for one matrix shape.

    ``source`` is Metal that declares ``type`` at global scope (a kernel
    includes each distinct source once). ``type`` is a type expression for a
    kernel body, where ``T`` names the activations' float type. The struct it
    names is the matrix: an aggregate of the weight codes as bytes, then one
    device pointer per name in ``tensors`` (the linear's parameters beside
    ``weight``, which become kernel inputs in this order), with the member
    template ``Lanes<K>`` that ``docs/quant-formats.md`` specifies.
    """


@dataclass(frozen=True)
class Tile(_Face):
    """A format's tile face for one matrix shape: the same matrix as its
    ``Dot``, read the way a kernel that decodes a tile of weights reads it.

    The fields are a ``Dot``'s. The struct ``type`` names is the matrix, an
    aggregate as a dot face's, with the member struct ``Runs`` that
    ``docs/quant-formats.md`` specifies: a row's cursor that reads a group's
    parameters and runs of 8 codes, and gives each weight as a float32.
    """


class QuantFormat:
    """One weight format. A subclass sets ``mode``, ``tensors`` and
    ``contract``, and overrides ``decode``, ``layout`` (what loads its
    checkpoints), ``encode`` (what writes them) and the faces it has; a
    kernel whose face a format lacks keeps its composed path."""

    mode: str  # a linear's "mode" in the quantization config
    tensors: tuple[str, ...]  # its parameters beside "weight" (and "bias")
    contract: int  # the revision of the contract it is written for

    def dot(self, layer, k: int, n: int) -> Dot | None:
        """The mat-vec face of ``layer``'s ``(..., n, k)`` matrix, or None."""
        return None

    def tile(self, layer, k: int, n: int) -> Tile | None:
        """The tile face of ``layer``'s ``(..., n, k)`` matrix, or None."""
        return None

    def decode(
        self, tensors: dict[str, mx.array], bits: int, group_size: int
    ) -> mx.array:
        """A matrix's weights as floats, ``(..., n, k)``. ``tensors`` are its
        ``weight`` and the format's tensors, by name: one matrix's, or those
        of several, with the same axes ahead of the matrix's in each."""
        raise NotImplementedError(f"quant format {self.mode!r} has no decode")

    def layout(
        self, k: int, n: int, bits: int, group_size: int
    ) -> dict[str, tuple[tuple[int, ...], mx.Dtype]] | None:
        """How a checkpoint keeps one ``(n, k)`` matrix: the shape and dtype
        of its ``weight`` and of each of the format's tensors, by name. None
        when the format keeps no such matrix, or when its layers are MLX's
        to build (affine)."""
        return None

    def encode(
        self, w: mx.array, importance: mx.array, bits: int, group_size: int
    ) -> dict[str, mx.array] | None:
        """The tensors that keep ``w``, by name and as ``layout`` gives them,
        or None to leave it to oQ's affine. ``w`` is a matrix ``(n, k)``, or
        ``(e, n, k)`` for some of a stack's, that oQ gives ``bits`` bits in
        groups of ``group_size``. ``importance`` weighs the squared error of
        each of the ``k`` inputs: ``(k,)``, or ``(e, k)`` when each of the
        matrices has its own."""
        return None


_FORMATS: dict[str, QuantFormat] = {}
# The plugin that registered each mode: its distribution's name and version.
_ORIGINS: dict[str, str] = {}
# Held for the whole load. Reentrant: the load registers through register(),
# which loads first.
_LOCK = threading.RLock()
_loaded = False
_loading = False


def register(fmt: QuantFormat) -> None:
    """Add ``fmt`` under its mode. A mode has one format, and the formats
    oMLX loads by itself (the built-in ones, then the installed plugins')
    have theirs first. A format written for a revision of the contract that
    this oMLX does not run is refused."""
    if not isinstance(fmt, QuantFormat):
        raise TypeError(f"{fmt!r} is not a QuantFormat instance")
    mode, tensors = getattr(fmt, "mode", None), getattr(fmt, "tensors", None)
    if not isinstance(mode, str) or not mode:
        raise ValueError(f"quant format {fmt!r} has no mode")
    if not isinstance(tensors, tuple) or {"weight", "bias"} & set(tensors):
        raise ValueError(
            f"quant format {mode!r}: tensors must be a tuple of the parameter "
            "names beside 'weight' and 'bias'"
        )
    contract = getattr(fmt, "contract", None)
    if type(contract) is not int:
        raise ValueError(
            f"quant format {mode!r}: contract is the revision of the format "
            f"contract it is written for, an int, not {contract!r} "
            f"({type(contract).__name__})"
        )
    if not OLDEST_CONTRACT <= contract <= CONTRACT:
        runs = (
            f"revision {CONTRACT}"
            if OLDEST_CONTRACT == CONTRACT
            else f"revisions {OLDEST_CONTRACT} to {CONTRACT}"
        )
        raise ValueError(
            f"quant format {mode!r} is written for revision {contract} of the "
            f"format contract; this oMLX runs {runs}: update "
            f"{'oMLX' if contract > CONTRACT else 'the format'}"
        )
    load()
    if _FORMATS.setdefault(mode, fmt) is not fmt:
        raise ValueError(f"quant format {mode!r} is already registered")


def load() -> None:
    """Register the built-in formats, then every installed plugin's, in the
    order of their entry points' names, then values. A plugin is logged with
    its distribution and the modes it registered. One that fails to load is
    logged and skipped, with every format registered while it loaded:
    its checkpoints then fail to load with an unknown mode, everything else
    is served. A built-in that fails to load is oMLX's own defect and raises,
    here and at every later lookup.

    A lookup the loading thread makes while the load runs sees the formats
    registered so far. Plugins do not rely on it (docs/quant-formats.md).

    Every lookup loads first. A caller loads ahead of a model's first token,
    so that the token neither waits for the load nor meets its failure."""
    global _loaded, _loading
    if _loaded:
        return
    with _LOCK:
        if _loaded or _loading:
            return
        _loading = True
        try:
            for value in BUILTIN:
                register(EntryPoint("builtin", value, ENTRY_POINT_GROUP).load())
            plugins = entry_points(group=ENTRY_POINT_GROUP)
            for entry in sorted(plugins, key=lambda entry: (entry.name, entry.value)):
                before = set(_FORMATS)
                try:
                    register(entry.load())
                except Exception:
                    for mode in _FORMATS.keys() - before:
                        del _FORMATS[mode]
                    logger.exception(
                        "quant format plugin %s (%s) failed to load",
                        entry.name,
                        entry.value,
                    )
                else:
                    dist = getattr(entry, "dist", None)
                    plugin = f"{dist.name} {dist.version}" if dist else entry.value
                    modes = sorted(_FORMATS.keys() - before)
                    _ORIGINS.update(dict.fromkeys(modes, plugin))
                    logger.info(
                        "quant format plugin %s (%s) registered %s",
                        entry.name,
                        plugin,
                        ", ".join(modes),
                    )
            _loaded = True
        finally:
            _loading = False


def get(mode: str | None) -> QuantFormat | None:
    """The format registered for ``mode``, or None."""
    load()
    return _FORMATS.get(mode)


def origin(mode: str) -> str | None:
    """The plugin whose format ``mode`` is, as the name and version of its
    distribution; None for a built-in format, and for one registered by
    hand."""
    load()
    return _ORIGINS.get(mode)


# The dot() and tile() defects that are logged: a format's mode and the
# error's repr.
_DOT_DEFECTS: set[tuple[str, str]] = set()


def dot(layer, k: int, n: int) -> Dot | None:
    """The mat-vec face of ``layer``'s ``(..., n, k)`` matrix, or None when
    its format (``layer.mode``) is unknown or has none for it.

    A kernel asks for a face inside a model's forward, and keeps the answer
    with what it builds for the layer. So a format whose ``dot`` raises, or
    returns anything but None or a ``Dot`` of arrays the layer has (the
    weight and the face's tensors, each of 8 elements or more), has no face
    for the layer either, and the model keeps its composed path there. Each
    defect of a format is logged once."""
    return _face(layer, k, n, Dot, "the layer keeps its composed path")


def tile(layer, k: int, n: int) -> Tile | None:
    """The tile face of ``layer``'s ``(..., n, k)`` matrix, or None, as
    ``dot`` gives the mat-vec face: a format whose ``tile`` raises, or
    returns anything but None or a ``Tile`` of arrays the layer has, has no
    tile face for the layer, which keeps its other kernels. Each defect is
    logged once, a defect being the format's mode and the error: the same
    error from the format's ``dot`` is the same defect."""
    return _face(layer, k, n, Tile, "the layer keeps its other kernels")


def _face(layer, k: int, n: int, kind: type, keeps: str):
    """The face of class ``kind`` that ``layer``'s format gives, from its
    method of the class's name in lower case, or None."""
    mode = getattr(layer, "mode", None)
    fmt = get(mode)
    if fmt is None:
        return None
    name = kind.__name__.lower()
    try:
        face = getattr(fmt, name)(layer, k, n)
        if face is None:
            return None
        if not isinstance(face, kind):
            raise TypeError(
                f"{name}() returned a {type(face)!r}, not a {kind.__name__}"
            )
        lacking = [
            name
            for name in ("weight", *face.tensors)
            if not isinstance(layer.get(name), mx.array)
        ]
        if lacking:
            raise ValueError(
                f"face {face.key}: the layer has no array {', '.join(lacking)}"
            )
        small = [name for name in ("weight", *face.tensors) if layer[name].size < 8]
        if small:
            raise ValueError(
                f"face {face.key}: {', '.join(small)} of fewer than 8 elements, "
                "which MLX binds as constants, not as the buffer a face reads"
            )
    except Exception as error:
        defect = (mode, repr(error))
        if defect not in _DOT_DEFECTS:
            _DOT_DEFECTS.add(defect)
            logger.exception(
                "quant format %s gave no usable %s face for a %s (n=%s, k=%s); %s",
                mode,
                name,
                type(layer).__name__,
                n,
                k,
                keeps,
            )
        return None
    return face
