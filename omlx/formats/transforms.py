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

import math
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn

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


@dataclass(frozen=True)
class ProducerSpec:
    """Fused producer kernels that write a format's rotated basis directly.

    Every callable is optional (None declines that producer fusion).
    - ``rmsnorm(x, weight, *tensors, eps=...) -> (rot, plain)``: fuses RMSNorm
      with the rotation and stores ``rot`` in the thread-local memo keyed by
      ``plain``; requires ``rmsnorm_block`` (0 disables).
    - ``gated_norm(x, gate, weight, *tensors, eps=...) -> (rot, plain)``: fuses
      GDN gated RMSNorm with the rotation and hands ``rot`` off through the
      thread-local handoff; requires ``gated_norm_head_dim`` (0 disables).
    - ``act_pair(gate_out, up_out, *tensors, limit=None) -> rot``: fuses
      ``silu(gate_out) * up_out`` with the down projection's rotation for a
      dense MLP.
    """

    rmsnorm: Callable[..., tuple[mx.array, mx.array]] | None = None
    rmsnorm_block: int = 0
    rmsnorm_max_width: int = 8192
    gated_norm: Callable[..., tuple[mx.array, mx.array]] | None = None
    gated_norm_head_dim: int = 128
    act_pair: Callable[..., mx.array] | None = None


@cache
def _signed_wht_fns(block: int):
    from ..custom_kernels.oqh import fast

    def apply(x, signs, *rest):
        if x.dtype == mx.float32:
            return _rotate_mlx(x, signs, block)
        return fast.rotate(x, signs, block)

    def apply_split_swiglu(gate_up_out, signs, *rest, limit=None):
        if gate_up_out.dtype == mx.float32:
            gate, up = mx.split(gate_up_out, 2, axis=-1)
            gate = (
                gate if limit is None else mx.minimum(gate, mx.array(limit, gate.dtype))
            )
            up = (
                up
                if limit is None
                else mx.clip(up, -mx.array(limit, up.dtype), mx.array(limit, up.dtype))
            )
            activated = mx.sigmoid(gate) * gate * up
            return _rotate_mlx(activated, signs, block)
        return fast.act_split_had(gate_up_out, signs, block, limit)

    def inverse_weight(w_rot, signs, *rest):
        return _inverse_mlx(w_rot, signs, block)

    def _rmsnorm(x, weight, signs, *rest, eps):
        return fast.rmsnorm_had(x, weight, signs, eps)

    def _gated_norm(x, gate, weight, signs, *rest, eps):
        return fast.gnorm_had(x, gate, weight, signs, eps, block)

    def _act_pair(a, b, signs, *rest, limit=None):
        return fast.act_pair_had(a, b, signs, block, limit)

    spec = ProducerSpec(
        rmsnorm=_rmsnorm if block == 256 else None,
        rmsnorm_block=256 if block == 256 else 0,
        rmsnorm_max_width=8192,
        gated_norm=_gated_norm,
        gated_norm_head_dim=128,
        act_pair=_act_pair,
    )

    def producer_spec(layer=None) -> ProducerSpec:
        return spec

    return apply, apply_split_swiglu, inverse_weight, producer_spec


class SignedWhtTransform(InputTransform):
    """A signed block Walsh-Hadamard rotation: ``x`` times the block Hadamard
    of the equality's signs, as ``omlx.custom_kernels.oqh.fast`` writes it.

    ``fast`` is imported lazily: a format that does not use this transform
    pays no native import.
    """

    def __new__(cls, block: int, tensors=(), **kwargs):
        apply, apply_split_swiglu, inverse_weight, producer_spec = _signed_wht_fns(
            block
        )
        kwargs.setdefault("producer_spec", producer_spec)
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
_MEMO_SIZE = 16


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


_SUBCLASSES_LOCK = threading.Lock()
_SUBCLASSES: dict[tuple[type, type], type] = {}
_MIXER_INPUTS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "q_proj",
    "k_proj",
    "v_proj",
)


def _set(obj, name: str, value) -> None:
    object.__setattr__(obj, name, value)


def _rebase(module: nn.Module, mixin: type) -> None:
    """Give one instance ``mixin``'s forward without altering its class."""
    base = type(module)
    if issubclass(base, mixin):
        return
    pair = (base, mixin)
    with _SUBCLASSES_LOCK:
        cls = _SUBCLASSES.get(pair)
        if cls is None:
            cls = type(f"Tx{base.__name__}", (mixin, base), {})
            _SUBCLASSES[pair] = cls
    module.__class__ = cls


def _producer_spec(tx: InputTransform, layer) -> ProducerSpec | None:
    if tx.producer_spec is None:
        return None
    spec = tx.producer_spec(layer)
    return spec if isinstance(spec, ProducerSpec) else None


def _cast_tensors(tensors: tuple, dtype: mx.Dtype) -> tuple:
    return tuple(
        (
            t.astype(dtype)
            if isinstance(t, mx.array) and mx.issubdtype(t.dtype, mx.floating)
            else t
        )
        for t in tensors
    )


class _DualNorm:
    """RMSNorm that also writes the rotated input of its format-layer readers."""

    def __call__(self, x):
        weight = self["weight"]
        if x.dtype not in (mx.bfloat16, mx.float16) or x.dtype != weight.dtype:
            return super().__call__(x)
        tensors = self._omlx_tx_cast_tensors
        if self._omlx_tx_weight_dtype != weight.dtype:
            tensors = _cast_tensors(self._omlx_tx_raw_tensors, weight.dtype)
            _set(self, "_omlx_tx_cast_tensors", tensors)
            _set(self, "_omlx_tx_weight_dtype", weight.dtype)
        rot, plain = self._omlx_tx_rmsnorm(x, weight, *tensors, eps=self.eps)
        for k, orig in self._omlx_tx_entries:
            remember(plain, rot, k, orig)
        return plain


class _DualGatedNorm:
    """GDN gated RMSNorm that hands the rotated input to its format out_proj."""

    def __call__(self, x, gate=None):
        out_proj = self._omlx_tx_reader
        if (
            gate is None
            or x.ndim < 2
            or x.shape != gate.shape
            or x.shape[-1] != self._omlx_tx_head_dim
            or x.shape[-1] * x.shape[-2] != out_proj.input_dims
            or gate.dtype not in (mx.bfloat16, mx.float16)
            or x.dtype not in (gate.dtype, mx.float32)
        ):
            return super().__call__(x, gate)
        rot, plain = self._omlx_tx_gated_norm(
            x, gate, self["weight"], *self._omlx_tx_tensors, eps=self.eps
        )
        hand_off(self._omlx_tx_handoff_key, rot, self._omlx_tx_tensors)
        return plain


class _FusedMLP:
    """Dense MLP whose SwiGLU fuses into down_proj's InputTransform."""

    def __call__(self, x):
        gate, up, down = self.gate_proj, self.up_proj, self.down_proj
        if self._omlx_tx_gu is not None and x.size:
            rx = prepare_bound(self._omlx_tx_gu, x)
            g_out = gate._call_prepared(rx)
            u_out = up._call_prepared(rx)
        else:
            g_out = gate(x)
            u_out = up(x)
        if (
            g_out.size
            and g_out.dtype in (mx.bfloat16, mx.float16)
            and u_out.dtype == g_out.dtype
        ):
            act = self._omlx_tx_act_pair(
                g_out, u_out, *self._omlx_tx_down_tensors, limit=self._omlx_tx_limit
            )
            return down._call_prepared(act)
        limit = self._omlx_tx_limit
        if limit is not None:
            g_out = mx.minimum(g_out, mx.array(limit, g_out.dtype))
            u_out = mx.clip(
                u_out,
                -mx.array(limit, u_out.dtype),
                mx.array(limit, u_out.dtype),
            )
        return down(nn.silu(g_out) * u_out)


def _transformed_layers(module, names) -> list:
    out = []
    for name in names:
        child = getattr(module, name, None)
        if child is not None and bind(child) is not None:
            out.append(child)
    return out


def _bind_dual_norm(norm, readers: list) -> bool:
    if not readers or type(norm) is not nn.RMSNorm:
        return False
    first = readers[0]
    first_bound = bind(first)
    if first_bound is None:
        return False
    if not all(same_transform(first, other) for other in readers[1:]):
        return False
    tx, tensors = first_bound
    spec = _producer_spec(tx, first)
    if spec is None or spec.rmsnorm is None or spec.rmsnorm_block <= 0:
        return False
    width = getattr(first, "input_dims", None)
    if (
        type(width) is not int
        or width <= 0
        or width % spec.rmsnorm_block
        or width > spec.rmsnorm_max_width
        or tx.block != spec.rmsnorm_block
    ):
        return False
    weight = norm.get("weight")
    if not isinstance(weight, mx.array):
        return False
    entries: list[tuple[tuple, tuple]] = []
    seen: set[tuple] = set()
    for reader in readers:
        bound = bind(reader)
        if bound is None:
            continue
        k = key(bound[0], bound[1])
        if k not in seen:
            seen.add(k)
            entries.append((k, bound[1]))
    _set(self := norm, "_omlx_tx_rmsnorm", spec.rmsnorm)
    _set(self, "_omlx_tx_raw_tensors", tensors)
    _set(self, "_omlx_tx_cast_tensors", _cast_tensors(tensors, weight.dtype))
    _set(self, "_omlx_tx_weight_dtype", weight.dtype)
    _set(self, "_omlx_tx_entries", tuple(entries))
    _rebase(norm, _DualNorm)
    return True


def _bind_gated_norm(norm, out_proj) -> bool:
    from .linear import FormatLinear

    if not isinstance(out_proj, FormatLinear) or not isinstance(norm, nn.Module):
        return False
    if (
        getattr(norm, "activation", "silu") != "silu"
        or type(norm).__name__ == "Glm5NextRMSNormGated"
    ):
        return False
    bound = bind(out_proj)
    if bound is None:
        return False
    tx, tensors = bound
    spec = _producer_spec(tx, out_proj)
    if spec is None or spec.gated_norm is None or spec.gated_norm_head_dim <= 0:
        return False
    weight = norm.get("weight")
    if (
        not isinstance(weight, mx.array)
        or weight.size != spec.gated_norm_head_dim
        or out_proj.input_dims % spec.gated_norm_head_dim
    ):
        return False
    _set(norm, "_omlx_tx_gated_norm", spec.gated_norm)
    _set(norm, "_omlx_tx_head_dim", spec.gated_norm_head_dim)
    _set(norm, "_omlx_tx_reader", out_proj)
    _set(norm, "_omlx_tx_handoff_key", key(tx, tensors))
    _set(norm, "_omlx_tx_tensors", tensors)
    _rebase(norm, _DualGatedNorm)
    return True


def _install_dense_mlp(mlp) -> tuple[bool, list]:
    """Fuse a dense FormatLinear MLP's SwiGLU into down_proj's InputTransform;
    returns ``(installed, gate_up_readers)``. Leaves SwitchGLU untouched."""
    from .linear import FormatLinear

    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    readers = [m for m in (gate, up) if m is not None and bind(m) is not None]
    if not (
        type(gate) is FormatLinear
        and type(up) is FormatLinear
        and type(down) is FormatLinear
    ):
        return False, readers
    act = getattr(mlp, "activation", None)
    if act is not None and act is not nn.silu and type(act).__name__ != "SwiGLU":
        return False, readers
    down_bound = bind(down)
    if down_bound is None:
        return False, readers
    d_tx, d_tensors = down_bound
    spec = _producer_spec(d_tx, down)
    if spec is None or spec.act_pair is None:
        return False, readers
    gu_bound = bind(gate)
    if gu_bound is not None:
        if not same_transform(gate, up):
            return False, readers
        gu_key = key(gu_bound[0], gu_bound[1])
        gu_entry = (gu_bound[0], gu_bound[1], gu_key)
    else:
        if bind(up) is not None:
            return False, readers
        gu_entry = None
    _set(mlp, "_omlx_tx_gu", gu_entry)
    _set(mlp, "_omlx_tx_act_pair", spec.act_pair)
    _set(mlp, "_omlx_tx_down_tensors", d_tensors)
    _set(mlp, "_omlx_tx_limit", getattr(act, "limit", None))
    _rebase(mlp, _FusedMLP)
    return True, readers


def install_producers(model: nn.Module) -> int:
    """Install generalized producer fusion (RMSNorm, GDN gated norm, dense MLP)
    on ``model`` for every format layer whose ``InputTransform.producer_spec``
    provides a fused producer kernel. Leaves SwitchGLU to the MoE patches."""
    from .linear import FormatLinear, FormatSwitchLinear

    count = 0
    for _, module in model.named_modules():
        if getattr(module, "_omlx_tx_producers_installed", False):
            continue
        installed_here = 0
        mixer = getattr(module, "linear_attn", None) or getattr(
            module, "self_attn", None
        )
        if mixer is not None:
            in_norm = getattr(module, "input_layernorm", None)
            if in_norm is not None:
                readers = _transformed_layers(mixer, _MIXER_INPUTS)
                if _bind_dual_norm(in_norm, readers):
                    installed_here += 1
            if _bind_gated_norm(
                getattr(mixer, "norm", None), getattr(mixer, "out_proj", None)
            ):
                installed_here += 1
        mlp = getattr(module, "mlp", None)
        if mlp is not None:
            switch = getattr(mlp, "switch_mlp", None)
            if switch is not None:
                readers = _transformed_layers(
                    switch, ("gate_up_proj", "gate_proj", "up_proj")
                )
                shared = getattr(mlp, "shared_expert", None)
                if shared is not None:
                    ok, shared_readers = _install_dense_mlp(shared)
                    installed_here += int(ok)
                    readers += shared_readers
            else:
                ok, readers = _install_dense_mlp(mlp)
                installed_here += int(ok)
            post_norm = getattr(module, "post_attention_layernorm", None)
            if post_norm is not None and _bind_dual_norm(post_norm, readers):
                installed_here += 1
        elif (
            not isinstance(module, (FormatLinear, FormatSwitchLinear))
            and hasattr(module, "gate_proj")
            and hasattr(module, "up_proj")
            and hasattr(module, "down_proj")
            and not isinstance(module, _FusedMLP)
        ):
            ok, _ = _install_dense_mlp(module)
            installed_here += int(ok)
        if installed_here:
            _set(module, "_omlx_tx_producers_installed", True)
            count += installed_here
    return count


def release_producers(model: nn.Module, *, wrapped_only: bool = False) -> int:
    """Restore producer-fused modules to their original classes.

    With ``wrapped_only=True`` (used when a LoRA adapter wraps a projection),
    only fused MLPs or gated norms whose projections are no longer bare
    ``FormatLinear`` instances are restored; ``_DualNorm`` stays installed so
    wrapped layers keep hitting the rotation memo.
    """
    from .linear import FormatLinear

    count = 0
    for _, module in model.named_modules():
        cls = type(module)
        if wrapped_only:
            if issubclass(cls, _FusedMLP) and not all(
                type(getattr(module, n, None)) is FormatLinear
                for n in ("gate_proj", "up_proj", "down_proj")
            ):
                module.__class__ = cls.__bases__[1]
                count += 1
            norm = getattr(module, "norm", None)
            if isinstance(norm, _DualGatedNorm):
                out_proj = getattr(module, "out_proj", None)
                if type(out_proj) is not FormatLinear or out_proj is not getattr(
                    norm, "_omlx_tx_reader", None
                ):
                    norm.__class__ = type(norm).__bases__[1]
                    count += 1
            continue
        if issubclass(cls, (_DualNorm, _DualGatedNorm, _FusedMLP)):
            module.__class__ = cls.__bases__[1]
            count += 1
        if getattr(module, "_omlx_tx_producers_installed", False):
            _set(module, "_omlx_tx_producers_installed", False)
    return count


def _tensor_specs(tensors: tuple) -> tuple[tuple[tuple[int, ...], mx.Dtype], ...]:
    return tuple((tuple(a.shape), a.dtype) for a in tensors)


def _submit_warm(
    futures: list,
    seen: set,
    dedup_key: tuple,
    specs: tuple[tuple[tuple[int, ...], mx.Dtype], ...],
    invoke: Callable,
    what: str,
    *,
    allow_none: bool = False,
) -> None:
    if dedup_key in seen:
        return
    seen.add(dedup_key)
    from . import kernels

    def _run() -> None:
        stream = kernels._warm_stream()
        with mx.stream(stream):
            arrays = kernels.zeros(*specs, stream=stream)
            mx.eval(*arrays)
            out = invoke(*arrays)
            if out is None and allow_none:
                return
            outputs = list(out) if isinstance(out, (list, tuple)) else [out]
            if not outputs or not all(isinstance(a, mx.array) for a in outputs):
                raise RuntimeError(f"transform warm for {what!r} returned no arrays")
            mx.eval(*outputs)

    futures.append(kernels._WARM_POOL.submit(_run))


def warm_model(model, *, dtypes=None) -> tuple:
    """Warm served InputTransform and fused producer kernels on ``_WARM_POOL``.

    Submits deduplicated tasks on ``kernels._warm_stream()`` capturing only
    shape/dtype tuples (never live ``mx.array`` weights) for:
    - ``tx.apply`` on each distinct transformed format layer;
    - ``dn_tx.apply_split_swiglu`` (with finite ``_swiglu_limit``) on each
      transformed ``switch_mlp``;
    - ``_omlx_tx_rmsnorm``, ``_omlx_tx_gated_norm`` (both activation-dtype and
      float32 inputs) and ``_omlx_tx_act_pair`` on installed producer mixins.

    Returns submitted futures; callers must await them and fail closed.
    """
    if model is None or not mx.metal.is_available():
        return ()
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return ()

    from ..patches import m5_gather_qmm
    from . import linear

    explicit = (
        tuple(dict.fromkeys(dt for dt in dtypes if dt in linear._WARM_DTYPES))
        if dtypes is not None
        else None
    )
    if explicit is not None and not explicit:
        return ()
    fp16_recast = bool(getattr(model, "_omlx_fp16_recast", False))
    model_dtype = (
        None
        if explicit is not None or fp16_recast
        else linear._infer_model_dtype(model)
    )
    default_dtypes = (
        explicit
        if explicit is not None
        else (
            (mx.float16,)
            if fp16_recast
            else (model_dtype,) if model_dtype is not None else linear._WARM_DTYPES
        )
    )

    futures: list = []
    seen: set = set()

    for _, mod in named_modules():
        bound = bind(mod)
        if bound is not None:
            tx, tensors = bound
            k = getattr(mod, "input_dims", None)
            if (
                type(k) is int
                and k > 0
                and all(isinstance(a, mx.array) for a in tensors)
            ):
                t_specs = _tensor_specs(tensors)
                layer_dtypes = (
                    linear._layer_warm_dtypes(
                        mod, model_dtype, explicit, fp16_recast=fp16_recast
                    )
                    if isinstance(mod, linear._FormatLayer)
                    else default_dtypes
                )
                apply_fn = tx.apply
                for dt in layer_dtypes:
                    _submit_warm(
                        futures,
                        seen,
                        ("apply", apply_fn, tx.kind, tx.block, k, dt, t_specs),
                        (((1, k), dt), *t_specs),
                        apply_fn,
                        f"{tx.kind}.apply",
                    )

        down = getattr(mod, "down_proj", None)
        if isinstance(down, linear.FormatSwitchLinear):
            dn_bound = bind(down)
            if dn_bound is not None and dn_bound[0].apply_split_swiglu is not None:
                dn_tx, dn_tensors = dn_bound
                lim = m5_gather_qmm._swiglu_limit(getattr(mod, "activation", None))
                inter = down.input_dims
                if (
                    inter > 0
                    and lim is not m5_gather_qmm._UNSUPPORTED
                    and (
                        lim is None
                        or (
                            isinstance(lim, (int, float))
                            and not isinstance(lim, bool)
                            and math.isfinite(lim)
                        )
                    )
                    and all(isinstance(a, mx.array) for a in dn_tensors)
                ):
                    t_specs = _tensor_specs(dn_tensors)
                    dn_dtypes = linear._layer_warm_dtypes(
                        down, model_dtype, explicit, fp16_recast=fp16_recast
                    )
                    split_fn = dn_tx.apply_split_swiglu
                    for dt in dn_dtypes:
                        _submit_warm(
                            futures,
                            seen,
                            (
                                "split_swiglu",
                                split_fn,
                                dn_tx.kind,
                                dn_tx.block,
                                inter,
                                dt,
                                t_specs,
                                lim,
                            ),
                            (((1, 2 * inter), dt), *t_specs),
                            lambda y, *ts, fn=split_fn, limit=lim: fn(
                                y, *ts, limit=limit
                            ),
                            f"{dn_tx.kind}.apply_split_swiglu",
                            allow_none=True,
                        )

        if isinstance(mod, _DualNorm):
            weight = mod.get("weight")
            raw_tensors = getattr(mod, "_omlx_tx_raw_tensors", ())
            if (
                isinstance(weight, mx.array)
                and weight.ndim >= 1
                and weight.shape[-1] > 0
                and weight.dtype in linear._WARM_DTYPES
                and (explicit is None or weight.dtype in explicit)
                and all(isinstance(a, mx.array) for a in raw_tensors)
            ):
                dt = weight.dtype
                width = int(weight.shape[-1])
                w_spec = (tuple(weight.shape), dt)
                t_specs = tuple(
                    (
                        tuple(a.shape),
                        dt if mx.issubdtype(a.dtype, mx.floating) else a.dtype,
                    )
                    for a in raw_tensors
                )
                eps = float(mod.eps)
                norm_fn = mod._omlx_tx_rmsnorm
                _submit_warm(
                    futures,
                    seen,
                    ("rmsnorm", norm_fn, width, dt, w_spec, t_specs),
                    (((1, width), dt), w_spec, *t_specs),
                    lambda x, w, *ts, fn=norm_fn, eps=eps: fn(x, w, *ts, eps=eps),
                    "producer.rmsnorm",
                )

        if isinstance(mod, _DualGatedNorm):
            weight = mod.get("weight")
            reader = getattr(mod, "_omlx_tx_reader", None)
            head_dim = getattr(mod, "_omlx_tx_head_dim", 0)
            g_tensors = getattr(mod, "_omlx_tx_tensors", ())
            in_dims = getattr(reader, "input_dims", 0)
            if (
                isinstance(weight, mx.array)
                and type(head_dim) is int
                and head_dim > 0
                and type(in_dims) is int
                and in_dims > 0
                and in_dims % head_dim == 0
                and all(isinstance(a, mx.array) for a in g_tensors)
            ):
                heads = in_dims // head_dim
                w_spec = (tuple(weight.shape), weight.dtype)
                t_specs = _tensor_specs(g_tensors)
                eps = float(mod.eps)
                gnorm_fn = mod._omlx_tx_gated_norm
                gate_dtypes = (
                    linear._layer_warm_dtypes(
                        reader, model_dtype, explicit, fp16_recast=fp16_recast
                    )
                    if isinstance(reader, linear._FormatLayer)
                    else default_dtypes
                )
                for dt in gate_dtypes:
                    for x_dt in (dt, mx.float32):
                        _submit_warm(
                            futures,
                            seen,
                            (
                                "gated_norm",
                                gnorm_fn,
                                heads,
                                head_dim,
                                x_dt,
                                dt,
                                w_spec,
                                t_specs,
                            ),
                            (
                                ((1, 1, heads, head_dim), x_dt),
                                ((1, 1, heads, head_dim), dt),
                                w_spec,
                                *t_specs,
                            ),
                            lambda x, g, w, *ts, fn=gnorm_fn, eps=eps: fn(
                                x, g, w, *ts, eps=eps
                            ),
                            "producer.gated_norm",
                        )

        if isinstance(mod, _FusedMLP):
            down_mlp = getattr(mod, "down_proj", None)
            d_tensors = getattr(mod, "_omlx_tx_down_tensors", ())
            inter = getattr(down_mlp, "input_dims", 0)
            if (
                type(inter) is int
                and inter > 0
                and all(isinstance(a, mx.array) for a in d_tensors)
            ):
                t_specs = _tensor_specs(d_tensors)
                lim = getattr(mod, "_omlx_tx_limit", None)
                act_fn = mod._omlx_tx_act_pair
                mlp_dtypes = (
                    linear._layer_warm_dtypes(
                        down_mlp, model_dtype, explicit, fp16_recast=fp16_recast
                    )
                    if isinstance(down_mlp, linear._FormatLayer)
                    else default_dtypes
                )
                for dt in mlp_dtypes:
                    _submit_warm(
                        futures,
                        seen,
                        ("act_pair", act_fn, inter, dt, t_specs, lim),
                        (((1, inter), dt), ((1, inter), dt), *t_specs),
                        lambda a, b, *ts, fn=act_fn, limit=lim: fn(
                            a, b, *ts, limit=limit
                        ),
                        "producer.act_pair",
                    )

    return tuple(futures)
