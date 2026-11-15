# SPDX-License-Identifier: Apache-2.0
"""The checks a format's tests are built from (``omlx.formats.testing``),
each run on the toy format of the kernel tests and on affine.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import linear, testing
from omlx.formats.linear import MAX_ROWS, FormatLinear, FormatSwitchLinear
from tests.test_formats_kernels import _TOY, _toy
from tests.test_formats_linear import (
    _experts,
    _ints,
    _pair,
    _switch_pair,
    _toy_linear,
    _toy_switch,
    _ToyFormat,
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
        layer(one)
        layer(one)  # its kept graph, replayed
        layer(rows)
        layer(more)  # more rows than a launch takes
        layer(one.astype(mx.float16))
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
        switch(tokens, routed)
        switch(tokens, routed, sorted_indices=True)  # replayed
        switch(token, few)  # fewer than 8 pairs: bound as constants
        switch(token, few[:0])  # no pair: the decode of none
    assert calls == [
        _call(switch, face, (mx.float32, 3, False)),
        _call(switch, face, (mx.float32, 3, False)),
        _call(switch, face, (mx.float32, 3, True)),
        _call(switch),
    ]


def test_a_call_the_decode_runs_is_recorded_without_a_face(toy, caplog):
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    toy()  # no face
    layer, x, _ = _toy_linear(40, 320)
    switch = _toy_switch(40, 320)
    with testing.layer_calls() as calls:
        layer(x)
        switch(_ints((2, 1, 320)), _experts((2,)))
    assert calls == [_call(layer), _call(switch)]
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
        with testing.layer_calls() as inner:
            layer(x)
        layer(x)
    assert len(outer) == 3 and len(inner) == 1 and inner[0] == outer[1]


def test_a_call_that_raises_is_not_recorded(toy):
    toy(_toy("linear"))
    layer, _, _ = _toy_linear(40, 320)
    with (
        testing.layer_calls() as calls,
        pytest.raises(ValueError, match=r"x is \(\.\.\., 320\)"),
    ):
        layer(_ints((1, 64)))
    assert calls == []


def _own():
    """The layers' methods that a block wraps, as the classes have them."""
    return (
        vars(FormatLinear)["__call__"],
        vars(FormatSwitchLinear)["__call__"],
        vars(linear._FormatLayer)["_decoded"],
    )


def test_outside_a_block_a_call_is_the_layer_s_own(toy):
    """While no block is open, the layers are as linear.py defines them: a
    call costs nothing more."""
    toy(_toy("linear"))
    layer, x, _ = _toy_linear(40, 320)
    own = _own()
    assert all(method.__module__ == linear.__name__ for method in own)
    with testing.layer_calls():
        assert all(a is not b for a, b in zip(_own(), own, strict=True))
        layer(x)
    assert _own() == own
    # Also when the block's body raises.
    with pytest.raises(RuntimeError, match="the body's"), testing.layer_calls():
        raise RuntimeError("the body's")
    assert _own() == own


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
    layer, x, _ = _toy_linear(40, 320)
    switch = _toy_switch(40, 320)
    decoding, _, _ = _toy_linear(40, 320, seed=3)
    more = _ints((MAX_ROWS + 1, 320))
    rows, indices = _ints((2, 1, 320)), _experts((2,))
    _evaluated(layer, switch, decoding, x)
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
                    layer(x)
            recorded[name] = calls
            if name == "first":
                ended.set()
        except BaseException as error:  # noqa: BLE001 - raised below, in the test
            errors.append(error)

    def first():
        layer(x)
        decoding(more)

    def second():
        switch(rows, indices)
        decoding(more)
        layer(x)

    def without():
        decoding(more)
        layer(x)

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
    assert len({_call(a, face, mx.float32), _call(a, face, mx.float32)}) == 1
    # What a failed assertion shows of one.
    assert repr(_call(a, face, mx.float32)) == (
        "LayerCall(FormatLinear toy 320 -> 40: face toy_linear, Metal mlx.core.float32)"
    )
    assert repr(_call(_toy_switch(40, 320))) == (
        "LayerCall(FormatSwitchLinear toy 320 -> 40: decode)"
    )


def test_a_module_that_is_no_format_layer_is_not_recorded():
    dense = nn.Linear(64, 16)
    with testing.layer_calls() as calls:
        dense(mx.zeros((1, 64)))
    assert calls == []
