# SPDX-License-Identifier: Apache-2.0
"""Per-module caches of static decode decisions.

One-token decode re-asks the same layout questions of every layer on every
step (quantization signatures, shapes, dtypes of weights that never change).
``cached_per_module`` answers them once per module and keeps the answer in
the module's ``__dict__``. The entry is keyed on the identity of every value
of the module and of its child modules down to ``depth`` levels (submodules,
weights, scales, biases), the module's training flag and caller ``flags``:
reassigning any of them rebuilds the entry on the next call. The entry keeps
the keyed objects alive, so an identity match cannot come from a new object
reusing a freed one's address. Plain configuration attributes of children
(bits, group_size, eps, ...) are construction-time constants and are not
re-checked.

Identities are compared with ``operator.is_`` rather than ``id()``: once an
audit hook is installed (mlx-vlm's imports install one), every ``id()`` call
raises an audit event and costs several times more.

Built values must not reference ``module`` itself (only its children and
tensors), so the entry creates no reference cycle through the module.

``clear_cached`` drops a module's entries and advances a permanent per-module
``_omlx_clear_epoch`` int in its ``__dict__``: a build that overlapped the
clear returns without publishing.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from itertools import chain
from operator import is_
from typing import Any

import mlx.nn as nn

_SLOTS_LOCK = threading.Lock()
_SLOTS: set[str] = set()
# A module's clear generation, kept in its __dict__: a build that overlapped
# a clear of its own module must not publish its pre-clear references. A
# clear of another module does not advance it.
_EPOCH_SLOT = "_omlx_clear_epoch"


def _children(module: nn.Module, depth: int) -> list:
    children: list = []
    level = [module]
    for _ in range(depth):
        level = [
            value
            for parent in level
            for value in dict.values(parent)
            if isinstance(value, nn.Module)
        ]
        children.extend(level)
    return children


def cached_per_module(
    module: nn.Module,
    slot: str,
    build: Callable[[nn.Module], Any],
    *,
    depth: int = 1,
    flags: tuple = (),
) -> Any:
    """``build(module)``, cached in ``module.__dict__[slot]`` (see module docstring)."""
    state = (module._training, flags)
    entry = module.__dict__.get(slot)
    if entry is not None:
        children, refs, cached_state, value = entry
        if cached_state == state:
            current = tuple(chain(dict.values(module), *map(dict.values, children)))
            if len(current) == len(refs) and all(map(is_, current, refs)):
                return value
    epoch = module.__dict__.get(_EPOCH_SLOT, 0)
    value = build(module)
    children = _children(module, depth)
    refs = tuple(chain(dict.values(module), *map(dict.values, children)))
    entry = (children, refs, state, value)
    with _SLOTS_LOCK:
        if module.__dict__.get(_EPOCH_SLOT, 0) == epoch:
            _SLOTS.add(slot)
            # Publish under the registry lock the clear takes: a known slot's
            # rebuild must not land after a clear, and a clear during the
            # build must not be overwritten by the pre-clear references.
            module.__dict__[slot] = entry
    return value


def epoch(module: nn.Module) -> int:
    """The module's clear generation: read it before a build whose answer
    ``publish`` keeps, so a clear that lands during the build wins."""
    return module.__dict__.get(_EPOCH_SLOT, 0)


def register_slot(slot: str) -> None:
    """Register a module ``__dict__`` slot that ``clear_cached`` drops.

    A patch that keeps entries on a module outside ``cached_per_module``
    registers its slot when it creates it, so clearing the module drops them
    (and their references to replaced tensors) too."""
    with _SLOTS_LOCK:
        _SLOTS.add(slot)


def publish(module: nn.Module, slot: str, since: int, entry) -> None:
    """Publish ``entry`` as ``module.__dict__[slot]``, a slot ``clear_cached``
    drops, unless the module was cleared since ``since`` (its epoch read
    before the build): a build that overlapped a clear does not land behind
    it. The caller keeps the entry's shape; only the slot's name is kept."""
    with _SLOTS_LOCK:
        if module.__dict__.get(_EPOCH_SLOT, 0) == since:
            _SLOTS.add(slot)
            module.__dict__[slot] = entry


def clear_cached(module: nn.Module) -> None:
    """Drop a module's ``cached_per_module`` entries and advance its
    ``_omlx_clear_epoch`` so overlapping builds return without publishing."""
    with _SLOTS_LOCK:
        d = module.__dict__
        for slot in _SLOTS:
            d.pop(slot, None)
        # After the pops, while the lock still excludes a publish: a build
        # that read the old epoch cannot land behind this clear.
        d[_EPOCH_SLOT] = d.get(_EPOCH_SLOT, 0) + 1


__all__ = ["cached_per_module", "clear_cached", "register_slot"]
