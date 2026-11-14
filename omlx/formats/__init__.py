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


@dataclass(frozen=True)
class Dot:
    """A format's mat-vec face for one matrix shape.

    ``source`` is Metal that declares ``type`` at global scope (a kernel
    includes each distinct source once). ``type`` is a type expression for a
    kernel body, where ``T`` names the activations' float type. The struct it
    names is the matrix: an aggregate of the weight codes as bytes, then one
    device pointer per name in ``tensors`` (the linear's parameters beside
    ``weight``, which become kernel inputs in this order), with the member
    template ``Lanes<K>`` that ``docs/quant-formats.md`` specifies.
    """

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
        if len(set(self.tensors)) != len(self.tensors):
            raise ValueError(
                f"quant format face {self.key!r}: a tensor is named twice in "
                f"{self.tensors!r}"
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


class QuantFormat:
    """One weight format. A subclass sets ``mode`` and ``tensors`` and
    overrides the faces it has; a kernel whose face a format lacks keeps its
    composed path."""

    mode: str  # a linear's "mode" in the quantization config
    tensors: tuple[str, ...]  # its parameters beside "weight" (and "bias")

    def dot(self, layer, k: int, n: int) -> Dot | None:
        """The mat-vec face of ``layer``'s ``(..., n, k)`` matrix, or None."""
        return None


_FORMATS: dict[str, QuantFormat] = {}
# Held for the whole load. Reentrant: the load registers through register(),
# which loads first.
_LOCK = threading.RLock()
_loaded = False
_loading = False


def register(fmt: QuantFormat) -> None:
    """Add ``fmt`` under its mode. A mode has one format, and the formats
    oMLX loads by itself (the built-in ones, then the installed plugins')
    have theirs first."""
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
    _load()
    if _FORMATS.setdefault(mode, fmt) is not fmt:
        raise ValueError(f"quant format {mode!r} is already registered")


def _load() -> None:
    """Register the built-in formats, then every installed plugin's, in the
    order of their entry points' names, then values. A plugin that fails to
    load is logged and skipped, with every format registered while it loaded:
    its checkpoints then fail to load with an unknown mode, everything else
    is served. A built-in that fails to load is oMLX's own defect and raises,
    here and at every later lookup.

    A lookup the loading thread makes while the load runs sees the formats
    registered so far. Plugins do not rely on it (docs/quant-formats.md)."""
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
            _loaded = True
        finally:
            _loading = False


def get(mode: str | None) -> QuantFormat | None:
    """The format registered for ``mode``, or None."""
    _load()
    return _FORMATS.get(mode)


# The modes whose dot() defect is logged.
_DOT_DEFECTS: set[str] = set()


def dot(layer, k: int, n: int) -> Dot | None:
    """The mat-vec face of ``layer``'s ``(..., n, k)`` matrix, or None when
    its format (``layer.mode``) is unknown or has none for it.

    A kernel asks for a face inside a model's forward. So a format whose
    ``dot`` raises, or returns anything but a ``Dot`` of tensors the layer
    has, has no face for the layer either, and the model keeps its composed
    path. The defect is logged, once per format."""
    fmt = get(getattr(layer, "mode", None))
    if fmt is None:
        return None
    try:
        face = fmt.dot(layer, k, n)
        if face is None:
            return None
        if not isinstance(face, Dot):
            raise TypeError(f"dot() returned {face!r}, not a Dot")
        lacking = [name for name in ("weight", *face.tensors) if name not in layer]
        if lacking:
            raise ValueError(f"face {face.key}: the layer has no {', '.join(lacking)}")
    except Exception:
        if fmt.mode not in _DOT_DEFECTS:
            _DOT_DEFECTS.add(fmt.mode)
            logger.exception("quant format %s gave no usable dot face", fmt.mode)
        return None
    return face
