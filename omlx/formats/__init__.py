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
compiled per format instance and the face is inlined into it.

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
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "omlx.quant_formats"

# Formats that ship with oMLX, as entry-point values ("module:attribute").
BUILTIN: tuple[str, ...] = ()


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
    tail: bool  # Lanes<K>::TAIL: K is not a whole number of the walk's blocks
    dtype: Any = None  # the only activation dtype the face is exact for, if any

    def __post_init__(self) -> None:
        # Both are spliced into Metal identifiers.
        for name in (self.key, *self.tensors):
            if not (isinstance(name, str) and name.isascii() and name.isidentifier()):
                raise ValueError(
                    f"quant format face {self.key!r}: {name!r} is not an identifier"
                )

    def exact_for(self, dtype) -> bool:
        """Whether the face computes its format's rows for ``dtype``
        activations."""
        # mx.Dtype does not compare with None.
        return self.dtype is None or self.dtype == dtype


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
_LOCK = threading.Lock()
_loaded = False


def register(fmt: QuantFormat) -> None:
    """Add ``fmt`` under its mode; a mode has one format."""
    mode, tensors = fmt.mode, fmt.tensors
    if not isinstance(mode, str) or not mode:
        raise ValueError(f"quant format {fmt!r} has no mode")
    if not isinstance(tensors, tuple) or {"weight", "bias"} & set(tensors):
        raise ValueError(
            f"quant format {mode!r}: tensors must be a tuple of the parameter "
            "names beside 'weight' and 'bias'"
        )
    if _FORMATS.setdefault(mode, fmt) is not fmt:
        raise ValueError(f"quant format {mode!r} is already registered")


def _load() -> None:
    """Register the built-in formats, then every installed plugin's. A plugin
    that fails to load is logged and skipped: its checkpoints then fail to
    load with an unknown mode, everything else is served."""
    global _loaded
    with _LOCK:
        if _loaded:
            return
        for value in BUILTIN:
            module, _, attribute = value.partition(":")
            register(getattr(import_module(module), attribute))
        for entry in entry_points(group=ENTRY_POINT_GROUP):
            try:
                register(entry.load())
            except Exception:
                logger.exception(
                    "quant format plugin %s (%s) failed to load",
                    entry.name,
                    entry.value,
                )
        _loaded = True


def get(mode: str | None) -> QuantFormat | None:
    """The format registered for ``mode``, or None."""
    if not _loaded:
        _load()
    return _FORMATS.get(mode)


def dot(layer, k: int, n: int) -> Dot | None:
    """The mat-vec face of ``layer``'s ``(..., n, k)`` matrix, or None when
    its format (``layer.mode``) is unknown or has none for it."""
    fmt = get(getattr(layer, "mode", None))
    return None if fmt is None else fmt.dot(layer, k, n)
