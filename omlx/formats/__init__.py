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

import copy
import gc
import logging
import math
import numbers
import struct
import sys
import threading
import types
import weakref
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from functools import cache
from importlib.metadata import EntryPoint, entry_points

import mlx.core as mx
import mlx.nn as nn

from ..patches.module_cache import cached_per_module

logger = logging.getLogger(__name__)

# The activation dtypes a kernel is built for: a face that names none is
# exact for these.
FLOATS: tuple[mx.Dtype, ...] = (mx.float16, mx.bfloat16, mx.float32)

ENTRY_POINT_GROUP = "omlx.quant_formats"

# Formats that ship with oMLX, as entry-point values ("module:attribute").
BUILTIN: tuple[str, ...] = ("omlx.formats.affine:FORMAT",)

# The revision of the contract (docs/quant-formats.md, Versions) that this
# oMLX implements, and the oldest one whose formats it still runs.
CONTRACT = 1
OLDEST_CONTRACT = 1


def _identifier(name) -> bool:
    return isinstance(name, str) and name.isascii() and name.isidentifier()


# MLX's own names on a quantized layer (D11): a plugin's format keeps no
# tensor of one of them, so a reader that indexes ``.weight``, ``.scales``,
# ``.biases`` or ``.bias`` fails loudly on a format layer of any family.
_MLX_TENSORS: tuple[str, ...] = ("weight", "scales", "biases", "bias")


def tensor_names(layer) -> tuple[str, ...]:
    """The names of ``layer``'s packed tensor and its side tensors:
    ``(fmt.codes_name, *fmt.tensors)`` for a format layer, or ``("weight",)``
    for MLX's quantized layer, whose other tensors a face reads by name.

    oMLX's kernels read a layer that may be either through this, never by
    indexing ``weight``: on a format layer of a plugin that name is not
    there (D11). A kernel built from a face takes ``tensor_names(layer)[0]``
    for the packed tensor and then the face's own ``tensors``, which may be a
    subset of the format's or another order."""
    names = getattr(layer, "_tensors", None)
    if names and isinstance(names, tuple):
        return names
    return ("weight",)


def _check_tensors(owner: str, tensors) -> None:
    """Refuses ``tensors`` unless they are what a format keeps beside
    ``weight`` and ``bias``, as kernels splice them into Metal: a tuple of
    distinct identifiers. ``register``, ``Dot`` and ``Tile`` ask the same."""
    if not isinstance(tensors, tuple):
        raise ValueError(f"{owner}: tensors are a tuple, not {tensors!r}")
    for name in tensors:
        if not _identifier(name):
            raise ValueError(f"{owner}: {name!r} is not an identifier")
    if {"weight", "bias"} & set(tensors) or len(set(tensors)) != len(tensors):
        raise ValueError(
            f"{owner}: tensors are the parameters beside 'weight' and 'bias', "
            f"each named once, not {tensors!r}"
        )


# The most bytes of tables a face, or a kernel, stages in threadgroup memory.
TABLE_BYTES = 4096
# The most values in a table.
TABLE_VALUES = 1024

# A table's element types: the struct code of its bits, and of its values
# where struct rounds them (bfloat16 is rounded here).
_TABLE_BITS = {
    mx.float32: ("<I", "<f"),
    mx.float16: ("<H", "<e"),
    mx.bfloat16: ("<H", None),
    mx.int8: ("<B", "<b"),
    mx.uint8: ("<B", "<B"),
    mx.int16: ("<H", "<h"),
    mx.uint16: ("<H", "<H"),
    mx.int32: ("<I", "<i"),
    mx.uint32: ("<I", "<I"),
}


def _bfloat16(value: float) -> float:
    """``value`` rounded to the nearest bfloat16, ties to even."""
    if value == 0 or math.isinf(value):
        return value
    _, exponent = math.frexp(abs(value))
    # A bfloat16 has 8 significant bits, and its subnormals step by 2^-133.
    quantum = exponent - 8 if exponent - 1 >= -126 else -133
    # Past bfloat16's range, the float32 that holds it is out of range too:
    # struct raises OverflowError for it.
    rounded = math.ldexp(round(math.ldexp(abs(value), -quantum)), quantum)
    return math.copysign(rounded, value)


def _table_entry(dtype: mx.Dtype, value) -> tuple:
    """``value`` as a table of ``dtype`` keeps it, and its bits."""
    bits, packed = _TABLE_BITS[dtype]
    if dtype in (mx.float32, mx.float16, mx.bfloat16):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            raise OverflowError("not a finite number")
        if packed is None:
            value = _bfloat16(value)
            # A bfloat16 is the high half of the float32 of its value.
            data = struct.pack("<f", value)[2:]
        else:
            # struct rounds to nearest even, and raises OverflowError past
            # the type's range.
            data = struct.pack(packed, value)
            (value,) = struct.unpack(packed, data)
    else:
        if value != int(value):
            raise OverflowError("not a whole number")
        value = int(value)
        try:
            data = struct.pack(packed, value)
        except struct.error as error:
            raise OverflowError(f"out of {dtype}'s range") from error
    return value, struct.unpack(bits, data)[0]


def _copy_descriptor(value, dtypes, memo):
    """Copy descriptor instance data while retaining MLX's immutable dtypes."""
    clone = object.__new__(type(value))
    memo[id(value)] = clone
    memo.update((id(dtype), dtype) for dtype in dtypes)
    state = copy.deepcopy(object.__getstate__(value), memo)
    attrs, slots = state if isinstance(state, tuple) else (state, {})
    if attrs is not None:
        object.__setattr__(clone, "__dict__", attrs)
    for name, item in slots.items():
        object.__setattr__(clone, name, item)
    return clone


@dataclass(frozen=True)
class Table:
    """A lookup table that a face reads at per-lane indices through an
    ``omlx_table<type, N>`` member. Kernels stage the table in threadgroup
    memory or hold up to four values directly in the member, depending on
    the GPU.

    ``values`` are numbers, each rounded to ``dtype`` when the table is
    made (a float table) or taken exactly (an integer table). Two tables are
    equal when their names, dtypes and the bits of their values are."""

    name: str
    dtype: mx.Dtype
    values: tuple = field(compare=False)
    bits: tuple = field(init=False, repr=False)

    def __deepcopy__(self, memo):
        return _copy_descriptor(self, (self.dtype,), memo)

    def __post_init__(self) -> None:
        if not (
            isinstance(self.name, str)
            and self.name.isascii()
            and self.name.isidentifier()
        ):
            raise ValueError(f"table {self.name!r}: its name is not an identifier")
        if self.dtype not in _TABLE_BITS:
            raise ValueError(
                f"table {self.name!r}: a table holds one of "
                f"{', '.join(str(d) for d in _TABLE_BITS)}, not {self.dtype!r}"
            )
        values = self.values
        if not (isinstance(values, tuple) and 0 < len(values) <= TABLE_VALUES):
            raise ValueError(
                f"table {self.name!r}: values are a tuple of 1 to {TABLE_VALUES} "
                "numbers"
            )
        kept, bits = [], []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"table {self.name!r}: {value!r} is not a number")
            try:
                rounded, word = _table_entry(self.dtype, value)
            except (OverflowError, ValueError) as error:
                raise ValueError(
                    f"table {self.name!r}: {value!r} is not a value of {self.dtype} "
                    f"({error})"
                ) from None
            kept.append(rounded)
            bits.append(word)
        object.__setattr__(self, "values", tuple(kept))
        object.__setattr__(self, "bits", tuple(bits))


@cache
def _metal_fields(face_type):
    """The source fragments a face contributes, in declaration order."""
    return tuple(
        (part.name, part.metadata["metal_part"])
        for part in fields(face_type)
        if "metal_part" in part.metadata
    )


@dataclass(frozen=True)
class _Face:
    """What every kind of face is: Metal, the type it declares, the tensors
    it reads and the activations it is exact for. A face of one kind never
    equals a face of another."""

    key: str  # names the instance in kernel names and compiler messages
    source: str = field(metadata={"metal_part": "main"})
    type: str
    tensors: tuple[str, ...]
    # The activations it is exact for; FLOATS if empty.
    dtypes: tuple[mx.Dtype, ...] = ()
    # Tables the face reads, staged by the kernel (docs/quant-formats.md, Tables).
    tables: tuple[Table, ...] = field(default=(), kw_only=True)

    def __deepcopy__(self, memo):
        return _copy_descriptor(self, self.dtypes, memo)

    def __post_init__(self) -> None:
        owner = f"quant format face {self.key!r}"
        if not (isinstance(self.tensors, tuple) and isinstance(self.dtypes, tuple)):
            raise ValueError(f"{owner}: tensors and dtypes are tuples")
        # The key and the tensors are spliced into Metal identifiers.
        if not _identifier(self.key):
            raise ValueError(f"{owner}: {self.key!r} is not an identifier")
        _check_tensors(owner, self.tensors)
        dtypes = {dtype for dtype in self.dtypes if isinstance(dtype, mx.Dtype)}
        if len(dtypes) != len(self.dtypes):
            raise ValueError(
                f"{owner}: dtypes are distinct mx.Dtype, not {self.dtypes!r}"
            )
        if dtypes - set(FLOATS):
            raise ValueError(
                f"{owner}: activations are float16, bfloat16 or float32, "
                f"not {self.dtypes!r}"
            )
        tables = self.tables
        if not (
            isinstance(tables, tuple) and all(isinstance(t, Table) for t in tables)
        ):
            raise ValueError(
                f"quant format face {self.key!r}: tables are a tuple of Table, "
                f"not {tables!r}"
            )
        names = [table.name for table in tables]
        taken = {"weight", *self.tensors}
        if len(set(names)) != len(names) or taken & set(names):
            raise ValueError(
                f"quant format face {self.key!r}: tables are each named once, and "
                f"not as the weight or a tensor: {names!r}"
            )
        size = sum(len(table.values) * table.dtype.size for table in tables)
        if size > TABLE_BYTES:
            raise ValueError(
                f"quant format face {self.key!r}: its tables hold {size} bytes; a "
                f"kernel stages at most {TABLE_BYTES} bytes of tables in "
                "threadgroup memory, beside its own"
            )
        from .kernels import hand_tables  # the kernels' module reads this one

        parts = tuple(self._metal_parts())
        for part, _, source in parts:
            if not isinstance(source, str):
                raise ValueError(f"{owner}: {part} is Metal, not {source!r}")
        found = hand_tables("\n".join(source for _, _, source in parts), face=True)
        if found:
            (line, name, text), *_ = found
            for part, _, source in parts:
                length = source.count("\n") + 1
                if line <= length:
                    label = "source" if part == "source" else f"{part} source"
                    break
                line -= length
            raise ValueError(
                f"{owner}: line {line} of its {label} declares "
                f"{name} with static storage ({text}), including namespace scope "
                "and static members. Per-lane table "
                "reads there make speed depend on kernel placement. Declare it as "
                f"data, {type(self).__name__}(..., tables=(Table(name, dtype, "
                "values),)), and read its omlx_table<type, N> member "
                "(docs/quant-formats.md, Tables). The check reads the usual "
                "spellings only: a table written as a parenthesized declarator "
                "or a variable template specialization is not detected, and "
                "only costs speed."
            )

    def _metal_parts(self):
        for name, phase in _metal_fields(type(self)):
            yield name, phase, getattr(self, name)

    def exact_for(self, dtype) -> bool:
        """Whether the face computes its format's rows for ``dtype``
        activations."""
        return dtype in (self.dtypes or FLOATS)


@dataclass(frozen=True)
class Dot(_Face):
    """A format's mat-vec face for one matrix shape.

    ``source`` is Metal that declares ``type`` at global scope (a kernel
    includes each distinct source once). ``type`` is a type expression for a
    kernel body, where ``T`` names the activations' float type. The struct it
    names is the matrix: an aggregate of the weight codes as bytes, then one
    device pointer per name in ``tensors`` (the linear's parameters beside
    ``weight``, which become kernel inputs in this order) and one
    ``omlx_table<T, N>`` per entry in ``tables``, in each tuple's order, with
    the member template ``Lanes<K>`` that ``docs/quant-formats.md`` specifies.

    ``shared`` is Metal that a kernel which applies one block's weights to
    several inputs includes after ``source``, and no other kernel: where the
    face defines ``decode`` and ``apply`` out of its struct. Empty for a
    face whose struct defines them. ``vpt`` is ``Lanes<K>::VPT``: such a
    kernel sizes its launch by it, and asserts it.
    """

    shared: str = field(default="", kw_only=True, metadata={"metal_part": "shared"})
    # The inputs a lane reads a block (Lanes<K>::VPT), which sizes a
    # shared-rows launch; 0 for a face that no such kernel reads.
    vpt: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        owner = f"quant format face {self.key!r}"
        if type(self.vpt) is not int or self.vpt < 0:
            raise ValueError(f"{owner}: vpt is a count of inputs, not {self.vpt!r}")


@dataclass(frozen=True)
class Tile(_Face):
    """A format's tile face for one matrix shape: the same matrix as its
    ``Dot``, read the way a kernel that decodes a tile of weights reads it.

    The fields are a ``Dot``'s. The struct ``type`` names is the matrix, an
    aggregate as a dot face's, with the member template ``Runs<K>`` that
    ``docs/quant-formats.md`` specifies: a row's cursor that reads a group's
    parameters and runs of 8 codes, and gives each weight as a float32.
    """


class QuantFormat:
    """One weight format. A subclass sets ``mode``, ``tensors`` and
    ``contract``, and overrides ``decode``, ``layout`` (what loads its
    checkpoints), ``encode`` (what writes them) and the faces it has; a
    kernel whose face a format lacks keeps its composed path.

    The tensor that holds the packed codes is ``codes_name``: ``weight`` for
    the built-in affine format, and a name of the plugin's own for a plugin
    (D11: a format layer has no tensor called ``weight``, ``scales``,
    ``biases`` or ``bias``, so code that reads one by hand fails loudly on
    every family)."""

    mode: str  # a linear's "mode" in the quantization config
    codes_name: str = "weight"  # its packed tensor's name: its codes
    tensors: tuple[str, ...]  # its side parameters beside "bias"
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
        codes (``codes_name``) and the format's tensors, by name: one matrix's,
        or those of several, with the same axes ahead of the matrix's in
        each."""
        raise NotImplementedError(f"quant format {self.mode!r} has no decode")

    def layout(
        self, k: int, n: int, bits: int, group_size: int
    ) -> dict[str, tuple[tuple[int, ...], mx.Dtype]] | None:
        """How a checkpoint keeps one ``(n, k)`` matrix: the shape and dtype
        of its codes (``codes_name``) and of each of the format's tensors, by
        name. None when the format keeps no such matrix, or when its layers
        are MLX's to build (affine)."""
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

    def record(self) -> dict:
        """What a checkpoint written in the format keeps of it beside its
        layers (the revision of its layout, what it needs to read its own
        checkpoints): a dict that JSON keeps as it is. oQ stores it in the
        checkpoint's config."""
        return {}

    def accept(self, record: dict) -> QuantFormat | None:
        """The format object that reads a checkpoint written with ``record``,
        or a ValueError, in the format's words, for one it cannot read.

        A checkpoint's load calls it once for each of its modes, when the
        format lays out the first of its layers, with the ``record`` it was
        written with: an empty one for a checkpoint written before formats
        had records. The receiver is the load's private copy of the registered
        format. Every layer keeps the returned reader (the copy by default or
        for None), with its own mutable state and the registered format's mode,
        tensors and contract, immutable and holding no unevaluated arrays
        (the layers of several checkpoints and threads may run at once).
        The registered format is left as it is."""
        return self


class LoadHookError(TypeError):
    """oMLX's load hook cannot wrap a library's loader of this version (it is
    no Python function): every load fails with it, no fallback load runs."""


class FormatRefusedError(ValueError):
    """oMLX refuses a quant format's layers, for ``reason``, and says what it
    knows of them: the ``mode``, the checkpoint at ``path`` (None for a
    layer built outside a checkpoint's load, and for oQ's refusal to write
    one), the plugin its record says ``written`` it, and the one
    ``installed``.

    Every refusal of a format's checkpoint at load is one of these, and
    oMLX's loaders let it through as it is: no fallback load runs after it.
    So is oQ's refusal to write a checkpoint of a family in a format.
    """

    def __init__(
        self,
        reason: str,
        *,
        mode: str,
        path=None,
        written: str | None = None,
        installed: str | None = None,
    ) -> None:
        self.reason, self.mode, self.path = reason, mode, path
        self.written, self.installed = written, installed
        said = f"quant format {mode!r}"
        if path is not None:
            said += f" of the checkpoint at {path} (written by {written}; "
            said += f"installed: {installed})"
        super().__init__(f"{said}: {reason}")


def check_layout(fmt: QuantFormat, n: int, k: int, layout: dict) -> None:
    """Refuse, in words, a ``layout`` that ``fmt`` gives for an ``(n, k)``
    matrix and that no layer holds: tensors of other names than the codes
    (``fmt.codes_name``) and the format's ``tensors``, codes of other than
    two axes, or codes whose first axis is not N (a layer's N is its codes'
    first axis: a format packs along K, never along N). oQ checks a layout
    before it encodes a matrix in it, and a layer the layout of the matrix it
    holds."""
    codes = fmt.codes_name
    names = (codes, *fmt.tensors)
    if set(layout) != set(names):
        raise ValueError(
            f"quant format {fmt.mode!r} lays out a ({n}, {k}) matrix as "
            f"{', '.join(sorted(layout))}: a layer holds {', '.join(names)}"
        )
    shape = tuple(layout[codes][0])
    if len(shape) != 2:
        raise ValueError(
            f"quant format {fmt.mode!r} lays out the tensor {codes!r} of a "
            f"({n}, {k}) matrix as {shape}: it has two axes, N and the "
            "format's own"
        )
    if shape[0] != n:
        raise ValueError(
            f"quant format {fmt.mode!r} lays out the tensor {codes!r} of a "
            f"({n}, {k}) matrix as {shape}: its first axis is N ({n}); a format "
            "packs along K, never along N"
        )


def _array_objects(array):
    """The Python references in an object array, including structured fields."""
    if array.dtype.hasobject:
        if array.dtype.names:
            for name in array.dtype.names:
                yield from _array_objects(array[name])
        else:
            yield from array.flat


def _reader_state(root, arrays):
    """Yield mutable object identities and collect arrays with native storage."""
    atomic = (
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        range,
        mx.Dtype,
        type(Ellipsis),
        type(NotImplemented),
    )
    ndarray = getattr(sys.modules.get("numpy"), "ndarray", ())
    seen = set()
    pending = [root]
    while pending:
        value = pending.pop()
        ident = id(value)
        kind = type(value)
        if ident in seen or kind in atomic:
            continue
        seen.add(ident)
        if kind not in (tuple, frozenset, slice):
            yield ident
        # A function's defaults and closure can be instance data. Its global
        # namespace and the class defining a method belong to the implementation.
        if isinstance(value, types.FunctionType):
            pending.extend(
                (
                    vars(value),
                    value.__defaults__,
                    value.__kwdefaults__,
                    value.__closure__,
                )
            )
        elif isinstance(value, types.MethodType):
            pending.append(value.__self__)
        elif isinstance(value, weakref.ReferenceType):
            pending.append(value())
        elif isinstance(value, mx.array):
            arrays.append(value)
        elif isinstance(value, memoryview):
            pending.append(value.obj)
        elif not isinstance(value, (type, types.ModuleType)):
            # CPython's traversal includes container members, private slots and
            # the instance dictionary without evaluating plugin properties.
            pending.extend(
                child for child in gc.get_referents(value) if child is not kind
            )
            if isinstance(value, ndarray):
                array = ndarray.view(value, ndarray)
                arrays.append(array)
                pending.append(ndarray.base.__get__(value))
                if array.dtype.hasobject:
                    pending.extend(_array_objects(array))


def _reader_array_views(arrays):
    """Expose the original bytes for comparisons across native array types."""
    import numpy as np

    for array in arrays:
        if isinstance(array, mx.array):
            buffer = memoryview(array)
            if buffer.c_contiguous:
                # Includes bfloat16, which NumPy cannot interpret as a dtype.
                yield np.frombuffer(buffer, dtype=np.uint8)
            else:
                if array.dtype == mx.bfloat16:
                    # NumPy can interpret the strides of an equal-width uint16
                    # view. Only use it if it kept the original MLX storage.
                    raw = array.view(mx.uint16)
                    mx.eval(raw)
                    size = mx.get_array_buffer_size
                    if size(array, raw) != size(array):
                        raise BufferError("the bfloat16 view is a copy")
                    buffer = memoryview(raw)
                view = np.asarray(buffer)
                if view.flags.owndata:
                    raise BufferError("the exported array is a copy")
                yield view
        else:
            yield array


def _shares_reader_storage(originals, copies):
    """Check MLX allocations and the byte ranges exposed by NumPy views."""
    import numpy as np

    native = [array for array in originals if isinstance(array, mx.array)]
    native_copies = [array for array in copies if isinstance(array, mx.array)]
    if native or native_copies:
        mx.eval(native, native_copies)
    if native and native_copies:
        size = mx.get_array_buffer_size
        if size(native, native_copies) < size(native) + size(native_copies):
            return True

    # Distinct DLPack capsules can expose the same memory. Keep the exact
    # strided-overlap test bounded for adversarial NumPy shapes.
    left = list(_reader_array_views(originals))
    return any(
        np.shares_memory(a, b, max_work=1024)
        for b in _reader_array_views(copies)
        for a in left
    )


def _check_reader_isolation(registered: QuantFormat, reader, *, stage: str) -> None:
    """Refuse an alias before accept can modify it or a checkpoint can keep it."""
    if reader is registered:
        reason = "the registered format"
    else:
        reason = "a reader sharing mutable state with the registered format"
        originals, copies = [], []
        original = set(_reader_state(registered, originals))
        if original.isdisjoint(_reader_state(reader, copies)):
            if not originals or not copies:
                return
            import numpy as np

            try:
                if not _shares_reader_storage(originals, copies):
                    return
            except (
                BufferError,
                TypeError,
                ValueError,
                getattr(np, "exceptions", np).TooHardError,
            ):
                reason = "a reader whose storage cannot be verified as independent"
    mode = registered.mode
    plugin = origin(mode)
    raise FormatRefusedError(
        f"{stage} returned {reason}; report it to {plugin or 'the plugin'}",
        mode=mode,
        installed=plugin,
    )


_FORMATS: dict[str, QuantFormat] = {}


def _may_build(mode) -> None:
    """Refuse a layer of ``mode``'s format built anywhere but in the load of
    a checkpoint that passed the family gate (``format_load``), or in a block
    of ``omlx.formats.testing.verified``: whatever builds it (a library's
    loader, a family's own, a script by hand), a family whose code may read
    the format's layers wrongly never holds one."""
    checkpoint = _LOADING.get()
    if checkpoint is not None and checkpoint.active and checkpoint.verified:
        return
    if _TESTING and checkpoint is None:
        return
    raise FormatRefusedError(
        "a format's layer is built only while oMLX loads a checkpoint of a "
        "family verified with quant formats, and this one is built outside "
        "such a load (by a family's own loader, or by hand): load the "
        "checkpoint with mlx-lm's or mlx-vlm's load_model",
        mode=mode,
    )


# The checkpoint whose layers the calling thread is loading, if any: set by
# omlx.patches.format_load around the libraries' loads, read by the layers.
# A copied context can keep an ended checkpoint. Only an active one owns layers.
_LOADING: ContextVar = ContextVar("omlx_formats_loading", default=None)
# Whether a block of omlx.formats.testing.verified is open, in any thread: a
# test builds a format's layers by hand in it.
_TESTING = False
# The plugin that registered each mode: its distribution's name and version.
_ORIGINS: dict[str, str] = {}
# Each mode of an installed plugin that did not load: what is said of it.
_SKIPPED: dict[str, str] = {}
# Set on the thread that registers the built-in formats, while it does; and,
# on the thread that loads an installed plugin, the modes it registers.
_REGISTERING_BUILTINS = threading.local()
_PLUGIN_MODES = threading.local()
# The modes of oMLX's built-in formats.
_BUILTIN_MODES: set[str] = set()
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
    modes = getattr(_PLUGIN_MODES, "modes", None)
    if modes is not None:
        modes.add(mode)
    _check_tensors(f"quant format {mode!r}", tensors)
    builtin = getattr(_REGISTERING_BUILTINS, "registering", False)
    contract = getattr(fmt, "contract", None)
    if type(contract) is not int:
        raise ValueError(
            f"quant format {mode!r}: contract is the revision of the format "
            f"contract it is written for, an int, not {contract!r} "
            f"({type(contract).__name__})"
        )
    if _mlx_runs(mode) and not builtin:
        raise ValueError(
            f"quant format {mode!r}: {mode!r} is one of MLX's own modes, which "
            "only oMLX's built-in formats take: register the format under a "
            "mode of its own"
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
    codes = getattr(fmt, "codes_name", "weight")
    if not _identifier(codes):
        raise ValueError(
            f"quant format {mode!r}: codes_name is the name of the tensor that "
            f"holds its codes, an identifier, not {codes!r}"
        )
    if codes == "bias" or codes in tensors:
        raise ValueError(
            f"quant format {mode!r} keeps its codes as {codes!r}: name its "
            f"packed tensor apart from 'bias' and {tensors!r}"
        )
    if not builtin:
        # D11: a plugin's format layer has no tensor named like MLX's, so
        # every by-hand reader of a layer fails loudly on every family.
        reserved = [name for name in (codes, *tensors) if name in _MLX_TENSORS]
        if reserved:
            raise ValueError(
                f"quant format {mode!r}: {', '.join(sorted(set(reserved)))} is "
                f"one of MLX's tensor names ({', '.join(_MLX_TENSORS)}), which a "
                "plugin's format keeps none of: name its codes (codes_name) and "
                "its tensors as it keeps them"
            )
    load()
    if builtin:
        _BUILTIN_MODES.add(mode)
    if _FORMATS.setdefault(mode, fmt) is not fmt:
        raise ValueError(f"quant format {mode!r} is already registered")


def load() -> None:
    """Register the built-in formats, then every installed plugin's, in the
    order of their entry points' names, then values. A plugin is logged with
    its distribution and the modes it registered. One that fails to load is
    logged and skipped, with every format registered while it loaded:
    a checkpoint of a mode it registered, or handed to register (which
    refused it), is then refused with its error, everything else is served.
    A built-in that fails to load is oMLX's own defect and raises, here and
    at every later lookup.

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
        saved = [
            (registry, registry.copy())
            for registry in (_FORMATS, _ORIGINS, _SKIPPED, _BUILTIN_MODES)
        ]
        _loading = True
        try:
            _REGISTERING_BUILTINS.registering = True
            try:
                for value in BUILTIN:
                    register(EntryPoint("builtin", value, ENTRY_POINT_GROUP).load())
            finally:
                _REGISTERING_BUILTINS.registering = False
            plugins = entry_points(group=ENTRY_POINT_GROUP)
            for entry in sorted(plugins, key=lambda entry: (entry.name, entry.value)):
                before = set(_FORMATS)
                dist = getattr(entry, "dist", None)
                plugin = f"{dist.name} {dist.version}" if dist else entry.value
                _PLUGIN_MODES.modes = tried = set()
                try:
                    register(entry.load())
                except Exception as error:
                    for mode in _FORMATS.keys() - before:
                        del _FORMATS[mode]
                    # What a checkpoint of each mode it would have registered
                    # is told, unless a plugin before it has one or said why.
                    for mode in tried - _FORMATS.keys():
                        _SKIPPED.setdefault(
                            mode, f"{plugin} is installed but did not load: {error}"
                        )
                    logger.exception(
                        "quant format plugin %s (%s) failed to load",
                        entry.name,
                        entry.value,
                    )
                else:
                    modes = sorted(_FORMATS.keys() - before)
                    _ORIGINS.update(dict.fromkeys(modes, plugin))
                    for mode in modes:
                        _SKIPPED.pop(mode, None)
                    logger.info(
                        "quant format plugin %s (%s) registered %s",
                        entry.name,
                        plugin,
                        ", ".join(modes),
                    )
                finally:
                    del _PLUGIN_MODES.modes
            _loaded = True
        except BaseException:
            # A retry starts with the caller's original registrations.
            for registry, before in saved:
                registry.clear()
                registry.update(before)
            raise
        finally:
            _loading = False


def _mlx_runs(mode) -> bool:
    """Whether ``mode`` is one of MLX's own: one ``mx.quantize`` does not
    call invalid, whatever else it finds wrong with a one-weight matrix (on
    the CPU: no GPU state is made for it)."""
    return isinstance(mode, str) and _probe_mlx_mode(mode)


@cache
def _probe_mlx_mode(mode: str) -> bool:
    try:
        mx.quantize(mx.zeros((1, 1), stream=mx.cpu), mode=mode, stream=mx.cpu)
    except ValueError as error:
        return "Invalid quantization mode" not in str(error)
    return True


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
# Held to note a defect and learn whether it is new (threads that meet one at
# once log it once), and by omlx.formats.testing to swap the set.
_NOTING = threading.Lock()


def stackable(fmt: QuantFormat, k: int, n: int, bits: int, group_size: int) -> bool:
    """Whether ``fmt``'s ``(n, k)`` matrices may be stacked along N: one
    matrix of 2N rows over two of N, as a fused gate and up projection is.
    For positive N, the layout must cover the weight and every declared
    side tensor. The rows rule (docs/quant-formats.md) holds for a format
    whose ``layout(k, n, ...)`` has N as their first axis, and whose
    ``layout(k, 2n, ...)`` is the same with 2N; such a format promises that
    a row of the decoded matrix depends on that row of each tensor alone
    (``testing.rows_stack`` checks it). A format with no layout, or whose
    ``layout`` raises (logged once), is not stackable."""
    return _stackable_layout(fmt, k, n, bits, group_size) is not None


def _stackable_layout(fmt: QuantFormat, k: int, n: int, bits: int, group_size: int):
    """The declared N-row layout when two such matrices can stack along N."""
    if n <= 0:
        return None
    try:
        one = fmt.layout(k, n, bits, group_size)
        if not isinstance(one, dict) or one.keys() != {fmt.codes_name, *fmt.tensors}:
            return None
        two = fmt.layout(k, 2 * n, bits, group_size)
        if (
            isinstance(two, dict)
            and two.keys() == one.keys()
            and all(
                len(shape) >= 1
                and shape[0] == n
                and tuple(two[name][0]) == (2 * n, *shape[1:])
                and two[name][1] == dtype
                for name, (shape, dtype) in one.items()
            )
        ):
            return one
    except Exception as error:
        defect = (fmt.mode, repr(error))
        with _NOTING:
            defects = _DOT_DEFECTS
            new = defect not in defects
            defects.add(defect)
        if new:
            logger.exception(
                "quant format %s gave no usable layout for (n=%s, k=%s); not stacked",
                fmt.mode,
                n,
                k,
            )
    return None


def dot(layer, k: int, n: int) -> Dot | None:
    """The mat-vec face of ``layer``'s ``(..., n, k)`` matrix, or None when
    its format (``layer.mode``) is unknown or has none for it.

    A format layer that still holds a load's placeholder is refused, as
    its call refuses it (``linear``).

    A kernel asks for a face inside a model's forward, and keeps the answer
    with what it builds for the layer. So a format whose ``dot`` raises, or
    returns anything but None or a ``Dot`` of arrays the layer has (the
    weight and the face's tensors, which are the format's, each of 8
    elements or more and none of float64), has no face for the layer
    either, and the model keeps its composed path there. Each defect of a
    format is logged once."""
    return _face(layer, k, n, Dot, "dot", "the layer keeps its composed path")


def tile(layer, k: int, n: int) -> Tile | None:
    """The tile face of ``layer``'s ``(..., n, k)`` matrix, or None, as
    ``dot`` gives the mat-vec face: a format whose ``tile`` raises, or
    returns anything but None or a ``Tile`` of arrays the layer has, has no
    tile face for the layer, which keeps its other kernels. Each defect is
    logged once, a defect being the format's mode and the error: the same
    error from the format's ``dot`` is the same defect."""
    return _face(layer, k, n, Tile, "tile", "the layer keeps its other kernels")


def _face(layer, k: int, n: int, kind: type, name: str, keeps: str):
    """The face of class ``kind`` that ``layer``'s format gives, from its
    method ``name``, or None.

    A layer (an ``nn.Module``) asks its format once for each kind and
    (k, n): the answer is kept on the layer (``module_cache``) until one of
    its arrays is another or is kept under another name, its bits or group
    size change (in value or in type), or its mode is another format's. That
    is what a format's face is a function of (docs/quant-formats.md, Faces).
    A defect is not kept: the format is asked again at the next call, which
    notes it again in a block of ``testing.no_dot_defects``. A kernel that
    asks at every call costs a lookup."""
    mode = getattr(layer, "mode", None)
    # A format's layer keeps the object that reads it; MLX's have a mode.
    fmt = getattr(layer, "fmt", None)
    if not isinstance(fmt, QuantFormat):
        fmt = get(mode)
    if fmt is None:
        return None
    # A format layer that still holds a load's placeholder has nothing to
    # read: refused here, ahead of the format.
    refuse = getattr(layer, "refuse_placeholders", None)
    if refuse is not None:
        refuse()
    if not isinstance(layer, nn.Module):
        return _asked(fmt, layer, k, n, kind, name, keeps)[0]
    slot = f"_omlx_{name}_face_{k}_{n}"
    bits = getattr(layer, "bits", None)
    group_size = getattr(layer, "group_size", None)
    face, defective = cached_per_module(
        layer,
        slot,
        lambda layer: _asked(fmt, layer, k, n, kind, name, keeps),
        depth=0,
        # The format object and its identity (equal formats are two), the
        # widths with their types (4 == 4.0), and the tensors' names (the
        # module cache keys their arrays alone).
        flags=(
            id(fmt),
            fmt,
            type(bits),
            bits,
            type(group_size),
            group_size,
            tuple(dict.keys(layer)),
        ),
    )
    if defective:
        layer.__dict__.pop(slot, None)
    return face


def _asked(fmt: QuantFormat, layer, k: int, n: int, kind: type, name: str, keeps):
    """``(face, defective)``: ``fmt``'s answer, checked, for ``_face``."""
    mode = fmt.mode
    try:
        return _checked_face(fmt, layer, k, n, kind, name), False
    except Exception as error:
        defect = (mode, repr(error))
        with _NOTING:
            defects = _DOT_DEFECTS
            new = defect not in defects
            defects.add(defect)
        if new:
            logger.exception(
                "quant format %s gave no usable %s face for a %s (n=%s, k=%s); %s",
                mode,
                name,
                type(layer).__name__,
                n,
                k,
                keeps,
            )
        return None, True


def _checked_face(fmt: QuantFormat, layer, k: int, n: int, kind: type, name: str):
    """Ask for one face. Writers propagate a defect, runtime callers record it."""
    face = getattr(fmt, name)(layer, k, n)
    if face is None:
        return None
    if not isinstance(face, kind):
        raise TypeError(f"{name}() returned a {type(face)!r}, not a {kind.__name__}")
    stray = [name for name in face.tensors if name not in fmt.tensors]
    if stray:
        raise ValueError(
            f"face {face.key}: {', '.join(stray)} is no tensor of quant "
            f"format {fmt.mode}"
        )
    names = (fmt.codes_name, *face.tensors)
    lacking = [name for name in names if not isinstance(layer.get(name), mx.array)]
    if lacking:
        raise ValueError(
            f"face {face.key}: the layer has no array {', '.join(lacking)}"
        )
    small = [name for name in names if layer[name].size < 8]
    if small:
        raise ValueError(
            f"face {face.key}: {', '.join(small)} of fewer than 8 elements, "
            "which MLX binds as constants, not as the buffer a face reads"
        )
    wide = [name for name in names if layer[name].dtype == mx.float64]
    if wide:
        raise ValueError(
            f"face {face.key}: {', '.join(wide)} of float64, which the GPU "
            "does not take"
        )
    return face
