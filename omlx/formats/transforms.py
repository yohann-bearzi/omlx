# SPDX-License-Identifier: Apache-2.0
"""A format's input transform: one rotation of the activation that every
route applies before the weights of a layer read it.

A format whose weights are stored in a rotated basis declares one with
``QuantFormat.input_transform``: the transformer is an ``InputTransform``,
frozen and versioned, with the ``shared_tensors`` that carry its parameters
(a sign vector), the block it applies over, ``apply`` itself and, where the
fast routes have one, a fused activation. The framework asks
``formats.input_transform(layer)`` once per module and keeps ``(tx,
tensors)``: no route asks again, and every route (rows, shared rows, tile,
decode, verify, gather, prefill) hands the same rotated activation to the
format's faces. A producer that already writes the rotated basis hands its
output over through the thread-local memo, so the rotation runs once.

The record a checkpoint keeps of the transform is
``{"transform": {"version": 1, "kind": "signed_wht", "block": n}}`` (with
``"block"`` optional) inside ``config["omlx_quant_formats"][mode]["record"]``.
``read_record`` validates that inner ``"transform"`` object and refuses any
other revision, kind, or out-of-range block, so a newer writer's checkpoint
fails closed.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

# The revision of the transform record, apart from the format contract: a
# format's own transform may change without invalidating every kernel hash.
TRANSFORM_VERSION = 1
# The widest block a rotation may name, as oQh's producer kernels take.
MAX_BLOCK = 256
# The transforms this oMLX knows how to read a record of.
KINDS = ("signed_wht",)


def _identifier(name) -> bool:
    return isinstance(name, str) and name.isascii() and name.isidentifier()


@dataclass(frozen=True)
class InputTransform:
    """One format's input transform.

    ``tensors`` are the layer's ``shared_tensors`` the transform reads, by
    name, and ``block`` the run of inputs each rotation covers. ``apply``
    takes the activation and those tensors in order. ``apply_split_swiglu``
    optionally fuses a gate|up split, SwiGLU and the rotation in one pass
    (the M3 Ultra steel prefill route); ``limit`` is the clamp's bound, None
    for the plain SwiGLU. ``inverse_weight`` recovers a decoded matrix in the
    plain basis (for embedding lookups, oQe refinement, and float32
    references). ``same_as`` tells whether
    two layers' transforms are the same (a shared producer or a fused block
    may reuse one), and ``producer_spec`` says what a fused producer writes
    (``omlx.oqh.runtime``'s norms and activation).
    """

    version: int = TRANSFORM_VERSION
    kind: str = "signed_wht"
    tensors: tuple[str, ...] = ()
    block: int = 0
    apply: Callable[..., mx.array] = None
    apply_split_swiglu: Callable[..., mx.array] | None = None
    inverse_weight: Callable[..., mx.array] | None = None
    same_as: Callable[..., bool] | None = None
    producer_spec: Callable[..., Any] | None = None
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != TRANSFORM_VERSION:
            raise ValueError(
                f"input transform: version {self.version} is not the "
                f"{TRANSFORM_VERSION} this oMLX runs"
            )
        if not _identifier(self.kind):
            raise ValueError(
                f"input transform: kind {self.kind!r} is not an identifier"
            )
        if not isinstance(self.tensors, tuple):
            raise ValueError("input transform: tensors are a tuple of identifiers")
        if any(not _identifier(name) for name in self.tensors):
            raise ValueError(
                f"input transform: tensors are identifiers, not {self.tensors!r}"
            )
        if len(set(self.tensors)) != len(self.tensors):
            raise ValueError("input transform: tensors are each named once")
        if type(self.block) is not int or not 1 <= self.block <= MAX_BLOCK:
            raise ValueError(
                f"input transform: block is a whole number of 1 to {MAX_BLOCK} "
                f"inputs, not {self.block!r}"
            )
        if not callable(self.apply):
            raise ValueError("input transform: apply is a function")
        for name in (
            "apply_split_swiglu",
            "inverse_weight",
            "same_as",
            "producer_spec",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise ValueError(f"input transform: {name} is a function or None")

    def record(self) -> dict:
        """What a checkpoint's config keeps of this transform."""
        return {"version": self.version, "kind": self.kind, "block": self.block}

    def same(self, other, layer_a, layer_b) -> bool:
        """Whether ``self`` and ``other`` rotate two layers alike."""
        if self is other and layer_a is layer_b:
            return True
        if other is None:
            return False
        if (self.version, self.kind, self.block, self.tensors) != (
            other.version,
            other.kind,
            other.block,
            other.tensors,
        ):
            return False
        if self.same_as is not None or other.same_as is not None:
            mine = self.same_as is None or bool(self.same_as(other, layer_a, layer_b))
            theirs = (
                other.same_as is None
                or other.same_as is self.same_as
                or bool(other.same_as(self, layer_b, layer_a))
            )
            if not (mine and theirs):
                return False
        elif (not self.tensors or self.kind not in KINDS) and self != other:
            return False
        return all(
            layer_a[name].dtype == layer_b[name].dtype
            and mx.array_equal(layer_a[name], layer_b[name]).item()
            for name in self.tensors
        )


def read_record(record: dict, *, kinds=KINDS) -> dict:
    """Refuse a transform record of another revision, or of a kind that
    ``kinds`` (a format's own, by default this oMLX's) does not name; give
    back the record of one this oMLX runs."""
    if not isinstance(record, dict):
        raise ValueError(f"input transform record: {record!r} is not an object")
    version = record.get("version")
    if type(version) is not int or version != TRANSFORM_VERSION:
        raise ValueError(
            f"input transform record: version {version!r}, where this oMLX "
            f"reads version {TRANSFORM_VERSION}"
        )
    kind = record.get("kind")
    if not _identifier(kind) or (kinds is not None and kind not in kinds):
        raise ValueError(
            f"input transform record: kind {kind!r} is not one this oMLX runs "
            f"({', '.join(kinds or ())})"
        )
    block = record.get("block", _ABSENT)
    if block is not _ABSENT and (type(block) is not int or not 1 <= block <= MAX_BLOCK):
        raise ValueError(
            f"input transform record: block {block!r} is not a whole number of "
            f"1 to {MAX_BLOCK} inputs"
        )
    return record


_ABSENT = object()


def _rotate_mlx(x: mx.array, signs: mx.array, block: int) -> mx.array:
    """The signed block Walsh-Hadamard transform in float32, exactly as oQh's
    offline reference computes it."""
    shape = x.shape
    y = x.astype(mx.float32) * signs.reshape(-1).astype(mx.float32)
    y = y.reshape(*shape[:-1], shape[-1] // block, block)
    return mx.hadamard_transform(y).reshape(shape)


def _inverse_mlx(w_rot: mx.array, signs: mx.array, block: int) -> mx.array:
    """The plain-basis weight of ``w_rot``: the transform is orthogonal and
    symmetric per block, so its inverse is the same signs and Hadamard."""
    shape = w_rot.shape
    y = mx.hadamard_transform(
        w_rot.astype(mx.float32).reshape(*shape[:-1], shape[-1] // block, block)
    ).reshape(shape)
    return y * signs.reshape(-1).astype(mx.float32)


class SignedWhtTransform(InputTransform):
    """A signed block Walsh-Hadamard rotation: ``x`` times the block Hadamard
    of the equality's signs, as ``omlx.custom_kernels.oqh.fast`` writes it.

    ``fast`` is imported lazily: a format that does not use this transform
    pays no native import.
    """

    def __new__(cls, block: int, tensors=(), **kwargs):
        from ..custom_kernels.oqh import fast

        def apply(x, signs, *rest):
            if x.dtype == mx.float32:
                return _rotate_mlx(x, signs, block)
            return fast.rotate(x, signs, block)

        def apply_split_swiglu(gate_up_out, signs, *rest, limit=None):
            if gate_up_out.dtype == mx.float32:
                gate, up = mx.split(gate_up_out, 2, axis=-1)
                gate = (
                    gate
                    if limit is None
                    else mx.minimum(gate, mx.array(limit, gate.dtype))
                )
                up = (
                    up
                    if limit is None
                    else mx.clip(
                        up, -mx.array(limit, up.dtype), mx.array(limit, up.dtype)
                    )
                )
                activated = mx.sigmoid(gate) * gate * up
                return _rotate_mlx(activated, signs, block)
            return fast.act_split_had(gate_up_out, signs, block, limit)

        def inverse_weight(w_rot, signs, *rest):
            return _inverse_mlx(w_rot, signs, block)

        # A factory: the descriptor is an InputTransform, not a subclass with
        # fields of its own (returning one skips this class's __init__).
        return InputTransform(
            version=TRANSFORM_VERSION,
            kind="signed_wht",
            tensors=tuple(tensors),
            block=block,
            apply=apply,
            apply_split_swiglu=apply_split_swiglu,
            inverse_weight=inverse_weight,
            **kwargs,
        )


class _Memo(threading.local):
    """A thread's rotated activations, kept while their plain array lives."""

    def __init__(self) -> None:
        self.entries: list[tuple[weakref.ref, mx.array, tuple, tuple]] = []
        self.handed: tuple[tuple, mx.array, tuple] | None = None


_MEMO = _Memo()
_MEMO_SIZE = 8


def key(tx: InputTransform, tensors: tuple) -> tuple:
    """The identity of one layer's transform: what a producer and its readers
    agree on without asking again. A transform with no shared tensor, of a
    custom kind, or with its own same_as, carries its descriptor (and its
    id) in the key: two such layers' parameters are not the key."""
    ids = tuple(id(tensor) for tensor in tensors)
    if tensors and tx.same_as is None and tx.kind in KINDS:
        return (tx.version, tx.kind, tx.block, ids)
    return (
        tx.version,
        tx.kind,
        tx.block,
        ids,
        tx,
        id(tx) if tx.same_as is not None else None,
    )


def clear() -> None:
    """Forget this thread's memo and handoff."""
    _MEMO.entries.clear()
    _MEMO.handed = None


def remember(plain: mx.array, rotated: mx.array, k: tuple, tensors: tuple = ()) -> None:
    """Keep ``rotated`` for the readers of ``plain`` while ``plain`` lives.

    ``tensors`` are pinned with the entry: they own the identities in ``k``,
    and letting their addresses be reused would let a stale key match."""
    entries = _MEMO.entries

    def drop(ref):
        entries[:] = [entry for entry in entries if entry[0] is not ref]

    entries.append((weakref.ref(plain, drop), rotated, k, tensors))
    if len(entries) > _MEMO_SIZE:
        del entries[0]


def recall(plain: mx.array, k: tuple) -> mx.array | None:
    """The rotated copy of ``plain`` if this thread kept one for ``k``."""
    for ref, rotated, held, _ in reversed(_MEMO.entries):
        if held == k and ref() is plain:
            return rotated
    return None


def hand_off(k: tuple, rotated: mx.array, tensors: tuple = ()) -> None:
    """Leave ``rotated`` for the linear that reads transform ``k``.

    A producer whose output the caller reshapes cannot be matched by the
    memo: the consumer takes the handoff once, by key and size. ``tensors``
    are pinned with the entry (the identities in ``k`` are theirs).
    """
    _MEMO.handed = (k, rotated, tensors)


def take(k: tuple, plain: mx.array) -> mx.array | None:
    """The handed-off rotation for ``k``, shaped as ``plain``, once."""
    handed = _MEMO.handed
    if (
        handed is None
        or handed[0] != k
        or handed[1].size != plain.size
        or handed[1].dtype != plain.dtype
    ):
        return None
    _MEMO.handed = None
    return handed[1].reshape(plain.shape)


def bind(layer) -> tuple[InputTransform, tuple] | None:
    """``(tx, tensors)`` of ``layer``'s format, or None. The format is asked
    once per module (``formats.input_transform``)."""
    from .. import formats

    return formats.input_transform(layer)


def prepare_bound(bound, x: mx.array) -> mx.array:
    """The rotated ``x`` for a bound ``(tx, tensors[, key])``: a layer's kept
    plan calls this, so no call asks the format again."""
    tx, tensors = bound[0], bound[1]
    k = bound[2] if len(bound) > 2 else key(tx, tensors)
    rotated = take(k, x)
    if rotated is not None:
        remember(x, rotated, k, tensors)
        return rotated
    rotated = recall(x, k)
    if rotated is None:
        rotated = tx.apply(x, *tensors)
        if rotated.shape != x.shape:
            rotated = rotated.reshape(x.shape)
        remember(x, rotated, k, tensors)
    return rotated


def prepare(layer, x: mx.array) -> mx.array:
    """The rotated ``x`` a transform layer reads: the producer's handoff, the
    memo's copy, or this transform applied once and kept."""
    bound = bind(layer)
    if bound is None:
        return x
    return prepare_bound(bound, x)


def same_transform(layer_a, layer_b) -> bool:
    """Whether two layers read the same rotated activation."""
    a, b = bind(layer_a), bind(layer_b)
    if a is None or b is None:
        return a is None and b is None
    return a[0].same(b[0], layer_a, layer_b)


def inverse_weight(w_rot: mx.array, tx: InputTransform, tensors: tuple) -> mx.array:
    """``w_rot``'s matrix in the plain basis."""
    if tx.inverse_weight is None:
        raise ValueError(
            f"input transform {tx.kind!r} gives no inverse_weight: cannot "
            "recover its plain-basis weights"
        )
    return tx.inverse_weight(w_rot, *tensors)


def transform_reference(layer, x: mx.array) -> mx.array:
    """``x @ W_plain.T`` for a transform layer: the reference that catches a
    route that forgot to rotate."""
    bound = bind(layer)
    if bound is None:
        raise ValueError(f"{layer!r} has no input transform to reference")
    tx, tensors = bound
    weight = inverse_weight(layer._decoded(), tx, tensors).astype(x.dtype)
    y = x @ weight.T
    bias = layer.get("bias")
    return y if bias is None else y + bias


def check_transform_applied(layer, x: mx.array, *, atol=1e-2) -> None:
    """Fail if ``layer(x)`` did not rotate ``x``: its answer must match the
    plain-basis reference of the rotated weight, not the rotated product."""
    want = transform_reference(layer, x)
    got = layer(x)
    a, b = got.astype(mx.float32), want.astype(mx.float32)
    if a.shape != b.shape:
        raise AssertionError(
            f"{layer!r}: output shape {a.shape} != reference shape {b.shape}"
        )
    if not mx.allclose(a, b, atol=atol, rtol=1e-2).item():
        scale = max(float(mx.abs(b).max().item()), 1e-6)
        error = float(mx.abs(a - b).max().item()) / scale
        raise AssertionError(
            f"{layer!r}: its call is {error:.3%} from the plain-basis reference "
            f"(atol {atol}): the input transform was not applied"
        )
