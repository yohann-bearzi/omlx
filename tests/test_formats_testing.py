# SPDX-License-Identifier: Apache-2.0
"""The checks a format's tests are built from (``omlx.formats.testing``),
each run on the toy format of the kernel tests and on affine.
"""

from __future__ import annotations

import gc
import inspect
import logging
import threading
import weakref
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from unittest.mock import ANY

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx import formats
from omlx.formats import kernels, linear, testing
from omlx.formats.linear import MAX_ROWS, FormatLinear
from tests.test_formats_kernels import _TOY, _matrix, _toy
from tests.test_formats_linear import (
    _equal,
    _experts,
    _ints,
    _pair,
    _switch_pair,
    _toy_linear,
    _toy_switch,
    _ToyFormat,
    _want,
    _x,
)

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture
def toy(monkeypatch):
    """Registers the toy format of the kernel tests, with a face or none."""
    formats.load()

    def register(face=None):
        monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(face))

    return register


def _call(layer, face=None, metal=None):
    return testing.LayerCall(layer, face, metal)


def test_a_block_records_a_linear_s_calls_and_the_face_each_ran(toy):
    face = _toy("linear")
    toy(face)
    layer, _, _ = _toy_linear(40, 320)
    one, rows = _ints((1, 320)), _ints((MAX_ROWS, 320))
    more = _ints((MAX_ROWS + 1, 320))
    with testing.layer_calls() as calls:
        for x in (one, one, rows, more):  # one, then its kept graph, replayed
            assert np.array_equal(np.array(layer(x)), _want(layer, x))
        half = layer(one.astype(mx.float16))
        assert np.array_equal(np.array(half), _want(layer, one).astype(np.float16))
    assert calls == [
        _call(layer, face, mx.float32),
        _call(layer, face, mx.float32),
        _call(layer, face, mx.float32),
        _call(layer),
        _call(layer, face, mx.float16),
    ]
    # Outside the block, nothing.
    layer(one)
    assert len(calls) == 5


def test_a_block_records_a_stack_s_calls_with_what_its_metal_is_built_for(toy):
    face = _toy("linear")
    toy(face)
    switch = _toy_switch(40, 320)
    tokens, routed = _ints((1, 4, 1, 1, 320)), _experts((1, 4, 3))
    token, few = _ints((1, 1, 320)), _experts((3,))
    with testing.layer_calls() as calls:
        for x, indices, ordered in (
            (tokens, routed, False),
            (tokens, routed, True),  # replayed
            (token, few, False),  # fewer than 8 pairs: bound as constants
        ):
            got = switch(x, indices, sorted_indices=ordered)
            assert np.array_equal(np.array(got), _want(switch, x, indices))
        none = switch(token, few[:0])  # no pair: the decode of none
        mx.eval(none)
        assert none.shape == (0, 1, 40)
    assert calls == [
        _call(switch, face, (mx.float32, 3, False)),
        _call(switch, face, (mx.float32, 3, False)),
        _call(switch, face, (mx.float32, 3, True)),
        _call(switch),
    ]


def test_a_call_the_decode_runs_is_recorded_without_a_face(toy, caplog):
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    toy()  # no face
    layer, x, want = _toy_linear(40, 320)
    switch = _toy_switch(40, 320)
    few, many = _experts((2,)), _experts((8,))  # fewer indices than matrices, more
    with testing.layer_calls() as calls:
        assert _equal(layer(x), want)
        for indices in (few, many):
            rows = _ints((indices.size, 1, 320))
            got = switch(rows, indices)
            assert np.array_equal(np.array(got), _want(switch, rows, indices))
        layer._decoded()  # outside any call
    assert calls == [_call(layer), _call(switch), _call(switch), _call(layer)]
    # A face that does not compile: its layers decode.
    broken = _toy("broken", _TOY + 'static_assert(false, "does not compile");\n')
    toy(broken)
    layer, x, _ = _toy_linear(40, 320)
    with testing.layer_calls() as calls:
        layer(x)
        layer(x)
    assert calls == [_call(layer), _call(layer)]
    assert len(caplog.records) == 1
    # A face that is not exact for x's type.
    toy(replace(_toy("half"), dtypes=(mx.float16,)))
    layer, x, _ = _toy_linear(40, 320)
    with testing.layer_calls() as calls:
        layer(x)
    assert calls == [_call(layer)]


def test_the_built_in_format_s_layers_are_recorded_and_mlx_s_are_not():
    quantized, layer = _pair(320, 72)
    switched, switch = _switch_pair(320, 72)
    x, rows, indices = _x((1, 320)), _x((4, 1, 1, 320)), _experts((4, 3))
    with testing.layer_calls() as calls:
        quantized(x)
        switched(rows, indices)
        layer(x)
        switch(rows, indices)
    assert calls == [
        _call(layer, formats.dot(layer, 320, 72), mx.bfloat16),
        _call(switch, formats.dot(switch, 320, 72), (mx.bfloat16, 3, False)),
    ]


def test_blocks_inside_one_another_each_record(toy):
    toy(_toy("linear"))
    layer, x, _ = _toy_linear(40, 320)
    with testing.layer_calls() as outer:
        layer(x)
        wrapped = _own()
        with testing.layer_calls() as inner:
            assert _own() == wrapped  # wrapped once
            layer(x)
        layer(x)
    assert len(outer) == 3 and len(inner) == 1 and inner[0] == outer[1]
    # The block that ends is the one that ends, though another equals it.
    with testing.layer_calls() as outer:
        with testing.layer_calls() as inner:
            pass
        layer(x)
    assert len(outer) == 1 and inner == []


def test_a_call_that_raises_is_not_recorded(toy):
    toy(_toy("linear"))
    layer, _, _ = _toy_linear(40, 320)
    with (
        testing.layer_calls() as calls,
        pytest.raises(ValueError, match=r"x is \(\.\.\., 320\)"),
    ):
        layer(_ints((1, 64)))
    assert calls == []


def test_a_call_that_raises_after_its_rows_ran_is_not_recorded(toy):
    toy()  # no face: the decode, then MLX's matmul, which refuses the shapes
    layer, _, _ = _toy_linear(40, 320)
    switch = _toy_switch(40, 320)
    with testing.layer_calls() as calls:
        with pytest.raises(ValueError):
            switch(_ints((3, 1, 320)), _experts((2,)))
        with pytest.raises(ValueError):
            layer(_ints((MAX_ROWS + 1, 320)).astype(mx.int32))
    assert calls == []
    # The face's launch, then a bias of another shape.
    toy(_toy("linear"))
    layer, x, _ = _toy_linear(40, 320, bias=mx.zeros((41,)))
    with testing.layer_calls() as calls, pytest.raises(ValueError):
        layer(x)
    assert calls == []


def _own():
    """What a block wraps, as the layers have it: a kept graph's hit, a
    launch, the decode and the two layers' calls."""
    return (
        vars(kernels.Proof)["graphs"],
        vars(linear)["_launch"],
        vars(linear._FormatLayer)["_decoded"],
        vars(linear.FormatLinear)["__call__"],
        vars(linear.FormatSwitchLinear)["__call__"],
    )


def test_outside_a_block_a_call_is_the_layer_s_own(toy):
    """While no block is open, the layers are as linear.py defines them: a
    call costs nothing more."""
    toy(_toy("linear"))
    layer, x, want = _toy_linear(40, 320)
    own = _own()
    assert own[0].fget.__module__ == kernels.__name__
    assert {own[i].__module__ for i in (1, 2, 3, 4)} == {linear.__name__}
    with testing.layer_calls():
        assert all(a is not b for a, b in zip(_own(), own, strict=True))
        assert _equal(layer(x), want)
    assert all(a is b for a, b in zip(_own(), own, strict=True))
    # Also when the block's body raises.
    with pytest.raises(RuntimeError, match="the body's"), testing.layer_calls():
        raise RuntimeError("the body's")
    assert all(a is b for a, b in zip(_own(), own, strict=True))


class _Interrupting:
    """An owner whose wrapping is interrupted: its own can be set back."""

    def __setattr__(self, name, value):
        if value == "a wrapper":
            raise KeyboardInterrupt
        vars(self)[name] = value


def test_a_block_interrupted_as_it_wraps_gives_back_what_it_wrapped(toy, monkeypatch):
    toy(_toy("linear"))
    layer, x, _ = _toy_linear(40, 320)
    own, wrappers = _own(), testing._wrappers
    owner = _Interrupting()
    vars(owner)["attribute"] = "its own"
    with monkeypatch.context() as patch:
        patch.setattr(
            testing,
            "_wrappers",
            lambda linear: [*wrappers(linear), (owner, "attribute", "a wrapper")],
        )
        with pytest.raises(KeyboardInterrupt), testing.layer_calls():
            pytest.fail("the block ran")
    assert all(a is b for a, b in zip(_own(), own, strict=True))
    assert owner.attribute == "its own"
    assert testing._open == 0 and testing._wrapped == []
    with testing.layer_calls() as calls:
        layer(x)
    assert calls == [_call(layer, _toy("linear"), mx.float32)]


def test_a_wrapped_call_has_the_call_s_arguments(toy, monkeypatch):
    toy()
    given = []

    def call(self, x, indices, sorted_indices=False):
        given.append((self, x, indices, sorted_indices))
        return x

    monkeypatch.setattr(linear.FormatSwitchLinear, "__call__", call)
    switch = _toy_switch(40, 320)
    with testing.layer_calls():
        switch(1, 2, sorted_indices=True)
        switch(3, 4, True)
        switch(5, 6)
    assert given == [(switch, 1, 2, True), (switch, 3, 4, True), (switch, 5, 6, False)]


def test_what_a_block_wraps_keeps_its_signature_and_its_words():
    own = _own()
    with testing.layer_calls():
        for wrapper, its_own in zip(_own(), own, strict=True):
            assert wrapper is not its_own and wrapper.__doc__ == its_own.__doc__
            if isinstance(its_own, property):
                continue
            assert wrapper.__wrapped__ is its_own
            assert wrapper.__qualname__ == its_own.__qualname__
            assert inspect.signature(wrapper, follow_wrapped=False) == (
                inspect.signature(its_own, follow_wrapped=False)
            )


class _InterruptedOnce:
    """An owner whose own, set back the first time, is interrupted."""

    def __init__(self):
        vars(self).update(attribute="its own", interrupted=False)

    def __setattr__(self, name, value):
        if value == "its own" and not self.interrupted:
            vars(self)["interrupted"] = True
            raise KeyboardInterrupt
        vars(self)[name] = value


def test_a_block_interrupted_as_it_ends_leaves_the_next_one_whole(toy, monkeypatch):
    face = _toy("linear")
    toy(face)
    layer, x, want = _toy_linear(40, 320)
    own, wrappers, owner = _own(), testing._wrappers, _InterruptedOnce()
    with monkeypatch.context() as patch:
        # Given back first, its interrupt leaves the layers wrapped.
        patch.setattr(
            testing,
            "_wrappers",
            lambda linear: [*wrappers(linear), (owner, "attribute", "a wrapper")],
        )
        with pytest.raises(KeyboardInterrupt), testing.layer_calls():
            pass
    assert owner.attribute == "a wrapper" and testing._open == 0
    with testing.layer_calls() as calls:
        assert _equal(layer(x), want)
    assert calls == [_call(layer, face, mx.float32)]  # once: wrapped once
    assert all(a is b for a, b in zip(_own(), own, strict=True))
    assert owner.attribute == "its own" and testing._wrapped == []


def _evaluated(*layers):
    """Arrays another thread can use: MLX runs an array that is not evaluated
    on the stream it was made on, which is the making thread's."""
    for layer in layers:
        mx.eval(layer.parameters() if isinstance(layer, nn.Module) else layer)


def test_a_block_records_the_calls_of_its_own_thread(toy):
    """Each thread's block has its thread's calls; a thread without a block
    has its calls run as without, decode included. The layers are their
    own again once the last block ends, whichever thread it is in."""
    toy(_toy("linear"))
    layer, x, want = _toy_linear(40, 320)
    switch = _toy_switch(40, 320)
    decoding, _, _ = _toy_linear(40, 320, seed=3)
    more = _ints((MAX_ROWS + 1, 320))
    rows, indices = _ints((2, 1, 320)), _experts((2,))
    _evaluated(layer, switch, decoding, x, want)
    own, opened, ended = _own(), threading.Barrier(3, timeout=30), threading.Event()
    recorded, errors = {}, []

    def run(name, work):
        try:
            mx.zeros((1,)).item()
            if name == "without":
                opened.wait()
                work()
                assert ended.wait(timeout=30)
                work()
                return
            with testing.layer_calls() as calls:
                opened.wait()
                work()
                if name == "second":
                    # Recorded after the first block ended.
                    assert ended.wait(timeout=30)
                    assert _equal(layer(x), want)
            recorded[name] = calls
            if name == "first":
                ended.set()
        except BaseException as error:  # noqa: BLE001 - raised below, in the test
            errors.append(error)

    def first():
        assert _equal(layer(x), want)
        assert np.array_equal(np.array(decoding(more)), _want(decoding, more))

    def second():
        got = switch(rows, indices)
        assert np.array_equal(np.array(got), _want(switch, rows, indices))
        assert np.array_equal(np.array(decoding(more)), _want(decoding, more))
        assert _equal(layer(x), want)

    def without():
        assert np.array_equal(np.array(decoding(more)), _want(decoding, more))
        assert _equal(layer(x), want)

    threads = [
        threading.Thread(target=run, args=(name, work))
        for name, work in (("first", first), ("second", second), ("without", without))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    if errors:
        raise errors[0]
    face = _toy("linear")
    assert recorded["first"] == [_call(layer, face, mx.float32), _call(decoding)]
    assert recorded["second"] == [
        _call(switch, face, (mx.float32, 1, True)),
        _call(decoding),
        _call(layer, face, mx.float32),
        _call(layer, face, mx.float32),
    ]
    assert _own() == own


def test_a_call_is_told_apart_by_its_layer_its_face_and_its_metal(toy):
    face = _toy("linear")
    toy(face)
    a, _, _ = _toy_linear(40, 320)
    b, _, _ = _toy_linear(40, 320)
    assert _call(a, face, mx.float32) == _call(a, face, mx.float32)
    assert _call(a, face, mx.float32) != _call(b, face, mx.float32)
    assert _call(a, face, mx.float32) != _call(a, face, mx.float16)
    assert _call(a, face, mx.float32) != _call(a)
    assert _call(a, face, mx.float32) != _call(a, _toy("other"), mx.float32)
    # Against what is not a record, the other side decides.
    assert _call(a, face, mx.float32) == ANY and _call(a) != object()
    # A record is a value: fixed, and as unhashable as the layer it holds.
    with pytest.raises(FrozenInstanceError):
        _call(a).face = face
    with pytest.raises(TypeError, match="unhashable"):
        hash(_call(a))
    with pytest.raises(TypeError, match="unhashable"):
        hash(testing.LayerCall(object()))
    # What a failed assertion shows of one.
    assert repr(_call(a, face, mx.float32)) == (
        "LayerCall(FormatLinear toy 320 -> 40: face toy_linear, Metal mlx.core.float32)"
    )
    assert repr(_call(_toy_switch(40, 320))) == (
        "LayerCall(FormatSwitchLinear toy 320 -> 40: decode)"
    )
    assert repr(_call(a, None, mx.float32)) == (
        "LayerCall(FormatLinear toy 320 -> 40: face None, Metal mlx.core.float32)"
    )


def test_a_module_that_is_no_format_layer_is_not_recorded():
    dense = nn.Linear(64, 16)
    with testing.layer_calls() as calls:
        dense(mx.zeros((1, 64)))
    assert calls == []


def test_a_layer_s_calls_through_two_faces_are_told_apart(toy):
    first, second = _toy("linear"), _toy("other")
    toy(first)
    layer, x, want = _toy_linear(40, 320)
    with testing.layer_calls() as calls:
        assert _equal(layer(x), want)
        toy(second)
        layer["scales"] = layer["scales"] * 1  # a new tensor: a new plan
        assert _equal(layer(x), want)
    assert calls == [_call(layer, first, mx.float32), _call(layer, second, mx.float32)]
    assert calls[0] != calls[1]


def test_a_call_that_mlx_compiles_is_recorded_as_it_is_traced(toy):
    toy(_toy("linear"))
    layer, x, want = _toy_linear(40, 320)
    compiled = mx.compile(lambda x: layer(x))
    with testing.layer_calls() as calls:
        assert _equal(compiled(x), want)
        assert _equal(compiled(x), want)  # a replay: no Python of the layer
    assert calls == [_call(layer, _toy("linear"), mx.float32)]


class _OwnDecode(FormatLinear):
    """A layer whose decode is its own: it calls no decode of the format."""

    def _decoded(self, indices=None):
        return (self["weight"].astype(mx.float32) - 128.0) * self["scales"][:, None]


def test_rows_are_recorded_where_they_ran(toy):
    face = _toy("linear")
    toy(face)
    _, x, want = _toy_linear(40, 320)
    (codes, scales), _, _ = _matrix(40, 320)
    layer = _OwnDecode("toy", 8, 1, 320, {"weight": codes, "scales": scales})
    more = _ints((MAX_ROWS + 1, 320))
    with testing.layer_calls() as calls:
        assert _equal(layer(x), want)
        assert np.array_equal(np.array(layer(more)), _want(layer, more))
    # Its own decode is neither the face nor the format's decode.
    assert calls == [_call(layer, face, mx.float32)]


class _DecodesAnother(_ToyFormat):
    """A format whose dot() decodes another layer before it gives the face."""

    other = None

    def dot(self, layer, k, n):
        self.other._decoded()
        return super().dot(layer, k, n)


def test_a_decode_is_recorded_for_its_own_layer(monkeypatch):
    formats.load()
    fmt = _DecodesAnother(_toy("linear"))
    monkeypatch.setitem(formats._FORMATS, "toy", fmt)
    layer, x, want = _toy_linear(40, 320)
    fmt.other, _, _ = _toy_linear(40, 320, seed=2)
    with testing.layer_calls() as calls:
        assert _equal(layer(x), want)
    assert calls == [_call(fmt.other), _call(layer, _toy("linear"), mx.float32)]


def test_outside_a_call_a_proof_s_graphs_are_as_they_are(toy):
    """In a block too: a kept graph called by hand records nothing."""
    face = _toy("linear")
    toy(face)
    layer, x, want = _toy_linear(40, 320)
    with testing.layer_calls() as calls:
        assert _equal(layer(x), want)
        (proof,) = kernels._PROOFS.values()
        graphs = proof.graphs
        assert type(graphs) is dict and proof.graphs is graphs
        graph = graphs[(x.shape, x.dtype)]
        assert _equal(graph(x, layer["weight"], layer["scales"]), want)
        assert calls == [_call(layer, face, mx.float32)]
        graphs.clear()
        assert proof.graphs == {}
        assert _equal(layer(x), want)  # launched again
    assert calls == [_call(layer, face, mx.float32)] * 2


class _OnEngaged(logging.Handler):
    """Runs ``action`` in the first launch that a proof logs as engaged."""

    def __init__(self, action):
        super().__init__(logging.INFO)
        self.action, self.result = action, None

    def emit(self, record):
        if self.action is not None and "engaged" in record.getMessage():
            action, self.action = self.action, None
            self.result = action()


@pytest.fixture
def on_engaged(caplog):
    """``on_engaged(action)``: ``action`` runs in the next launch proven."""
    caplog.set_level(logging.INFO, logger=linear.__name__)
    handlers = []

    def add(action):
        handlers.append(_OnEngaged(action))
        linear.logger.addHandler(handlers[-1])
        return handlers[-1]

    yield add
    for handler in handlers:
        linear.logger.removeHandler(handler)


def test_a_running_call_reads_its_proof_s_graphs_as_they_are(toy, on_engaged):
    """Not a copy: the thread's own graphs, those kept after the read too."""
    face = _toy("linear")
    toy(face)
    layer, x, want = _toy_linear(40, 320)
    other, y, _ = _toy_linear(48, 320, seed=2)
    half = x.astype(mx.float16)
    with testing.layer_calls() as calls:
        other(y)
        (another,) = kernels._PROOFS.values()
        assert _equal(layer(x), want)
        (proof,) = (p for p in kernels._PROOFS.values() if p is not another)
        # In the call, as it proves: its proof's graphs, and another's.
        read = on_engaged(lambda: (proof.graphs, another.graphs))
        layer(half)
        (seen, theirs), kept = read.result, proof.graphs
    assert isinstance(seen, Mapping) and type(kept) is dict and type(theirs) is dict
    signature = (x.shape, x.dtype)
    assert len(seen) == 2 and seen == kept and list(seen) == list(kept)
    assert seen[signature] is kept[signature] and signature in seen
    assert seen.get(((2, 320), x.dtype)) is None
    assert seen.get(((2, 320), x.dtype), "none") == "none"
    assert calls == [
        _call(other, face, mx.float32),
        _call(layer, face, mx.float32),
        _call(layer, face, mx.float16),
    ]


def test_a_call_made_while_another_s_launch_is_proven_is_recorded_apart(
    toy, on_engaged
):
    face = _toy("linear")
    toy(face)
    layer, x, want = _toy_linear(40, 320)
    other, y, other_want = _toy_linear(48, 320, seed=2)  # a proof of its own
    proving = on_engaged(lambda: (other(y), other(y)))  # a launch, its replay
    with testing.layer_calls() as calls:
        assert _equal(layer(x), want)
    assert all(_equal(got, other_want) for got in proving.result)
    assert calls == [
        _call(other, face, mx.float32),
        _call(other, face, mx.float32),
        _call(layer, face, mx.float32),
    ]


class _CallsAnother(_ToyFormat):
    """A format whose dot() reads the proofs' graphs and calls another layer
    before it gives the face."""

    other = x = None

    def dot(self, layer, k, n):
        if layer is not self.other:
            self.read = [type(proof.graphs) for proof in kernels._PROOFS.values()]
            self.other(self.x)
        return super().dot(layer, k, n)


def test_the_calls_a_call_makes_are_recorded_with_it(monkeypatch):
    formats.load()
    face = _toy("linear")
    fmt = _CallsAnother(face)
    monkeypatch.setitem(formats._FORMATS, "toy", fmt)
    fmt.other, fmt.x, _ = _toy_linear(48, 320, seed=2)
    fmt.other(fmt.x)  # kept: replayed below
    layer, x, want = _toy_linear(40, 320)
    with testing.layer_calls() as calls:
        assert _equal(layer(x), want)
    assert calls == [_call(fmt.other, face, mx.float32), _call(layer, face, mx.float32)]
    assert fmt.read == [dict]  # read in the call, before it has a plan
    # A call that raises takes the calls it made with it.
    layer, x, _ = _toy_linear(40, 320, seed=1, bias=mx.zeros((41,)))
    with testing.layer_calls() as calls, pytest.raises(ValueError):
        layer(x)
    assert calls == []


def test_a_thread_holds_no_layer_once_its_block_ends(toy):
    toy(_toy("linear"))
    layer, x, want = _toy_linear(40, 320)
    with testing.layer_calls():
        assert _equal(layer(x), want)
    gone = weakref.ref(layer)
    del layer
    gc.collect()
    assert gone() is None


class _OpensABlock(_ToyFormat):
    """A format whose dot() runs a block of its own."""

    def dot(self, layer, k, n):
        with testing.layer_calls():
            pass
        return super().dot(layer, k, n)


def test_a_block_that_ends_in_a_call_leaves_the_call_to_the_others(monkeypatch):
    formats.load()
    face = _toy("linear")
    monkeypatch.setitem(formats._FORMATS, "toy", _OpensABlock(face))
    layer, x, want = _toy_linear(40, 320)
    with testing.layer_calls() as calls:
        assert _equal(layer(x), want)
    assert calls == [_call(layer, face, mx.float32)]


def test_a_call_left_running_by_an_interrupt_ends_with_the_last_block(toy):
    face = _toy("linear")
    toy(face)
    layer, x, want = _toy_linear(40, 320)
    with testing.layer_calls():
        # A call whose end was interrupted: it is left in the thread's calls.
        testing._RECORDING.thread.frames.append(testing._Frame(layer))
    with testing.layer_calls() as calls:
        assert _equal(layer(x), want)
    assert calls == [_call(layer, face, mx.float32)]


class _Waits(_ToyFormat):
    """A format whose decode waits, once inside, for ``go``."""

    def __init__(self):
        super().__init__()
        self.inside, self.go = threading.Event(), threading.Event()

    def decode(self, tensors, bits, group_size):
        self.inside.set()
        assert self.go.wait(timeout=30)
        return super().decode(tensors, bits, group_size)


def test_a_block_ended_while_its_thread_runs_a_call_has_none_of_it(monkeypatch):
    formats.load()
    fmt = _Waits()
    monkeypatch.setitem(formats._FORMATS, "toy", fmt)
    layer, x, want = _toy_linear(40, 320)
    _evaluated(layer, x, want)
    own = _own()
    with ThreadPoolExecutor(max_workers=1) as pool:

        def opened():
            block = testing.layer_calls()
            return block, block.__enter__()

        block, calls = pool.submit(opened).result()
        running = pool.submit(lambda: _equal(layer(x), want))
        assert fmt.inside.wait(timeout=30)
        block.__exit__(None, None, None)  # here, while the call runs there
        fmt.go.set()
        assert running.result()
    assert calls == [] and _own() == own


def test_a_block_ended_from_another_thread_ends_in_its_own(toy):
    toy(_toy("linear"))
    layer, x, want = _toy_linear(40, 320)
    _evaluated(layer, x, want)
    with ThreadPoolExecutor(max_workers=1) as pool:

        def opened():
            outer, inner = testing.layer_calls(), testing.layer_calls()
            return outer, outer.__enter__(), inner, inner.__enter__()

        outer, outer_calls, inner, inner_calls = pool.submit(opened).result()
        inner.__exit__(None, None, None)  # here, not in the block's thread
        assert pool.submit(lambda: _equal(layer(x), want)).result()
        pool.submit(outer.__exit__, None, None, None).result()
    face = _toy("linear")
    assert outer_calls == [_call(layer, face, mx.float32)] and inner_calls == []
