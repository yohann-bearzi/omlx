# SPDX-License-Identifier: Apache-2.0
"""The Metal variants recorded by face kernels and built on route-safe zeros."""

import gc
import threading
import weakref
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.formats import QuantFormat, kernels, testing
from omlx.formats.linear import FormatLinear, FormatSwitchLinear, warm_dot
from tests.test_formats_kernels import _matrix, _toy


def kernel(monkeypatch, native):
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **source: native)
    return kernels.face_kernel("warm_copy", ["x"], {}, ["y"], "", "y[0] = x[0];")


def launch(call, x, template=(), dtype=mx.float32):
    return call(
        inputs=[x],
        output_shapes=[(1,)],
        output_dtypes=[dtype],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        template=template,
    )


def test_registry_distinguishes_generated_metal_and_drops_launch_arrays(monkeypatch):
    call = kernel(monkeypatch, lambda **kw: "lazy output")
    with mx.stream(mx.cpu):
        x = mx.zeros((2, 4))
        ref = weakref.ref(x)
        assert launch(call, x) == "lazy output"
        del x
        gc.collect()
        assert ref() is None
        launch(call, mx.zeros((17,)))
        assert len(call.instances()) == 1
        for x in (mx.zeros((7,)), mx.array(0.0), mx.zeros((8,), mx.int32)):
            launch(call, x)
        launch(call, 1.0)  # same scalar Metal as mx.array(0.0)
        assert len(call.instances()) == 4
        x = mx.zeros((8,))
        launch(call, x, dtype=mx.float16)
        launch(call, x, [("N", True)])
        launch(call, x, [("N", 1)])
        launch(call, x, [("N", 1), ("T", mx.float16)])
        launch(call, x, [("T", mx.float16), ("N", 1)])
        assert len(call.instances()) == 9


def test_warm_uses_only_stand_ins_and_builds_before_evaluation(monkeypatch):
    # Model MLX's lazy library cache independently of the instance registry.
    built, launches = set(), []

    def native(*, inputs, output_dtypes, template, **kwargs):
        (x,) = inputs
        key = (x.dtype, x.ndim == 0, x.size < 8, tuple(output_dtypes), tuple(template))
        launches.append((x.shape, threading.get_ident(), kwargs.get("stream")))
        return [key]

    call = kernel(monkeypatch, native)

    def evaluate(*outputs):
        built.update(outputs)

    monkeypatch.setattr(mx, "eval", evaluate)

    def cpu_zeros(*specs, stream):
        assert stream == mx.cpu
        assert specs == (((8,), mx.float32),)
        return [mx.zeros(shape, dtype, stream=mx.cpu) for shape, dtype in specs]

    monkeypatch.setattr(kernels, "_warm_stream", lambda: mx.cpu)
    monkeypatch.setattr(kernels, "zeros", cpu_zeros)
    x = mx.zeros((4096,), stream=mx.cpu)
    outputs = launch(call, x, [("K", 8)])
    (instance,) = call.instances()
    assert not built

    def recipe():
        return kernels.StandIn(
            input_shapes=((8,),),
            output_shapes=((1,),),
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )

    (future,) = kernels.warm({instance: recipe})
    future.result(timeout=5)
    assert len(built) == 1
    before = built.copy()
    evaluate(*outputs)
    assert built == before
    assert launches[0] == ((4096,), threading.get_ident(), None)
    assert launches[1][0] == (8,) and launches[1][1] != threading.get_ident()
    assert launches[1][2] is mx.cpu
    assert call.instances() == (instance,)


def test_warm_rejects_a_different_metal_before_allocating(monkeypatch):
    call = kernel(monkeypatch, lambda **kw: [])
    launch(call, mx.zeros((32,), stream=mx.cpu))
    (instance,) = call.instances()

    def wrong():
        return kernels.StandIn(
            input_shapes=((7,),),
            output_shapes=((1,),),
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )

    monkeypatch.setattr(
        kernels, "zeros", lambda *specs: pytest.fail("allocated wrong Metal")
    )
    (future,) = kernels.warm({instance: wrong})
    with pytest.raises(ValueError, match="stand-in"):
        future.result(timeout=5)


def test_warm_builds_the_real_metal_library_once_on_small_zeros():
    """Run the process-local Metal build counter in a fresh MLX process."""
    import subprocess
    import sys
    from pathlib import Path

    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("warm_library_probe.py"))],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=90,
    )


def test_planned_instances_match_two_live_kernels_without_an_early_launch(monkeypatch):
    calls = []

    def native(**kw):
        calls.append(kw)
        return []

    first = kernel(monkeypatch, native)
    second = kernels.face_kernel("other_copy", ["x"], {}, ["y"], "", "y[0] = x[1];")
    specs = [((8,), mx.int32)]
    plans = [
        k.instance(specs, [mx.float16], template=[("K", 8)]) for k in (first, second)
    ]
    assert plans[0] != plans[1] and not calls
    assert set(kernels.instances()) == set(plans)
    for k, plan in zip((first, second), plans):
        launch(k, mx.zeros((12, 16), mx.int32, stream=mx.cpu), [("K", 8)], mx.float16)
        assert k.instances() == (plan,)
    assert len(calls) == 2 and set(kernels.instances()) == set(plans)


def test_native_acceptance_precedes_recording_and_normalizes_integer_templates(
    monkeypatch,
):
    def native(**kw):
        if kw["template"] == [("reject", 7)]:
            raise ValueError("native rejected template")
        return []

    call = kernel(monkeypatch, native)
    x = mx.array(1)
    with pytest.raises(ValueError, match="native rejected"):
        launch(call, x, [("reject", 7)])
    assert call.instances() == ()

    class Integer(int):
        __hash__ = None

    launch(call, x, [("K", Integer(8))])
    launch(call, x, [("K", 8)])
    assert len(call.instances()) == 1


def test_independent_warm_builds_overlap_and_a_failure_can_be_retried(monkeypatch):
    sync = threading.Barrier(2, timeout=5)
    threads = set()

    def native(**kwargs):
        return []

    call = kernel(monkeypatch, native)
    first, second = [call.instance([((8,), t)], [t]) for t in (mx.float32, mx.int32)]
    monkeypatch.setattr(kernels, "_warm_stream", lambda: mx.cpu)
    monkeypatch.setattr(
        kernels, "zeros", lambda *s, stream: [mx.zeros(*s[0], stream=stream)]
    )
    monkeypatch.setattr(mx, "eval", lambda *outputs: None)

    def recipe():
        return kernels.StandIn(((8,),), ((1,),), (1, 1, 1), (1, 1, 1))

    def together():
        threads.add(threading.get_ident())
        sync.wait()
        return recipe()

    def broken():
        together()
        raise RuntimeError("build failed")

    ok, bad = kernels.warm({first: together, second: broken})
    ok.result(timeout=8)
    with pytest.raises(RuntimeError, match="build failed"):
        bad.result(timeout=8)
    assert len(threads) == 2
    (retry,) = kernels.warm({second: recipe})
    retry.result(timeout=5)


class _Future:
    """A build's future, counted as the load waits for it."""

    def __init__(self, error=None):
        self.error, self.awaited = error, 0

    def result(self, timeout=None):
        self.awaited += 1
        if self.error is not None:
            raise self.error


def test_a_load_waits_for_its_metal_builds_and_drops_them(tmp_path):
    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    first, second = _Future(), _Future()
    checkpoint.warmers.extend((first, second))
    checkpoint.join_warmers()
    assert (first.awaited, second.awaited) == (1, 1)
    assert checkpoint.warmers == []
    checkpoint.join_warmers()
    assert (first.awaited, second.awaited) == (1, 1)


def test_a_build_that_failed_is_logged_and_the_load_goes_on(tmp_path, caplog):
    import logging

    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    trailing = _Future()
    checkpoint.warmers.extend(
        (_Future(RuntimeError("the Metal did not build")), trailing)
    )
    with caplog.at_level(logging.WARNING, logger=format_load.__name__):
        checkpoint.join_warmers()
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "did not build" in record.getMessage()
    assert trailing.awaited == 1


def test_a_checkpoint_starts_each_face_and_geometry_once(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    planned = []

    def warm_dot(layer_cls, fmt, bits, group_size, k, tensors, layout):
        planned.append((layer_cls, fmt.mode, bits, group_size, k))
        return [_Future()]

    monkeypatch.setattr(format_load, "warm_dot", warm_dot)
    built = SimpleNamespace(output_dims=8, bits=4, group_size=64)
    for _ in range(2):
        format_load._warm_layer(
            built, SimpleNamespace(mode="toy"), {}, 320, {}, checkpoint
        )
    assert planned == [(SimpleNamespace, "toy", 4, 64, 320)]
    assert len(checkpoint.warmers) == 1


def test_a_layer_without_a_face_starts_nothing(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    monkeypatch.setattr(format_load, "warm_dot", lambda *arguments: None)
    built = SimpleNamespace(output_dims=8, bits=4, group_size=64)
    format_load._warm_layer(built, SimpleNamespace(mode="toy"), {}, 320, {}, checkpoint)
    assert checkpoint.warmers == [] and checkpoint.warmed == set()


def test_a_plan_that_raised_is_logged_and_the_layer_keeps_its_first_call(
    monkeypatch, tmp_path, caplog
):
    import logging
    from types import SimpleNamespace

    from omlx.patches import format_load

    def warm_dot(*arguments):
        raise RuntimeError("the plan failed")

    monkeypatch.setattr(format_load, "warm_dot", warm_dot)
    checkpoint = format_load._Checkpoint(tmp_path)
    built = SimpleNamespace(output_dims=8, bits=4, group_size=64)
    with caplog.at_level(logging.WARNING, logger=format_load.__name__):
        format_load._warm_layer(
            built, SimpleNamespace(mode="toy"), {}, 320, {}, checkpoint
        )
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert checkpoint.warmers == [] and len(checkpoint.warmed) == 1


class _Only(QuantFormat):
    """The toy format with a face exact for the activation dtypes given."""

    mode, codes_name, tensors, contract = "toy", "codes", ("group_scales",), 1

    def __init__(self, dtypes, n=16, k=320):
        self.face = replace(_toy("warm_exact"), dtypes=dtypes)
        self.n, self.k = n, k

    def dot(self, layer, k, n):
        return self.face

    def decode(self, tensors, bits, group_size):
        return tensors["codes"].astype(mx.float32)


def _warm_plan(monkeypatch, tensors, layout, dtypes, n=16, k=320):
    """warm_dot's result and what it recorded: (specs, template) per
    planned instance, and each recipe's StandIn."""
    seen = {"specs": [], "recipes": []}

    def planned(kernel, specs, output_dtypes, *, template):
        seen["specs"].append((specs, template, tuple(output_dtypes)))
        return ("m", len(seen["specs"]))

    def warm(instances):
        for build in instances.values():
            seen["recipes"].append(build())
        return (("future",),)

    monkeypatch.setattr("omlx.formats.linear.kernels.planned", planned)
    monkeypatch.setattr("omlx.formats.linear.kernels.warm", warm)
    planned_result = warm_dot(
        FormatLinear, _Only(dtypes, n, k), 8, 1, k, tensors, layout
    )
    return planned_result, seen


def test_warm_dot_builds_only_the_dtypes_the_face_is_exact_for(monkeypatch):
    from omlx.formats import testing
    from tests.test_formats_kernels import _matrix

    (codes, scales), _, _ = _matrix(16, 320)
    tensors = {"codes": codes, "group_scales": scales}
    layout = {"codes": ((16, 320), mx.uint8), "group_scales": ((16,), mx.float32)}
    with testing.verified("llama"):
        planned, seen = _warm_plan(monkeypatch, tensors, layout, (mx.bfloat16,))
        assert planned == (("future",),)
        ((specs, template, outputs),) = seen["specs"]
        assert outputs == (mx.bfloat16,)
        assert specs == (
            ((320,), mx.bfloat16),
            ((16, 320), mx.uint8),
            ((16,), mx.float32),
        )
        assert template == [
            ("T", mx.bfloat16),
            ("K_SIZE", 320),
            ("N_SIZE", 16),
        ]
        (stand_in,) = seen["recipes"]
        assert stand_in.input_shapes == ((320,), (16, 320), (16,))
        assert stand_in.output_shapes == ((8,),)
        assert stand_in.grid == (32, 2, 1) and stand_in.threadgroup == (32, 2, 1)
        planned, seen = _warm_plan(monkeypatch, tensors, layout, ())
        assert planned is not None
        assert [template[0][1] for _, template, _ in seen["specs"]] == [
            mx.bfloat16,
            mx.float16,
        ]
        assert [outputs for _, _, outputs in seen["specs"]] == [
            (mx.bfloat16,),
            (mx.float16,),
        ]


def test_warm_dot_starts_nothing_without_a_served_dtype(monkeypatch):
    from omlx.formats import testing

    (codes, scales), _, _ = _matrix(16, 320)
    tensors = {"codes": codes, "group_scales": scales}
    layout = {"codes": ((16, 320), mx.uint8), "group_scales": ((16,), mx.float32)}
    monkeypatch.setattr(
        "omlx.formats.linear.kernels.warm",
        lambda instances: pytest.fail("warmed a face with no served dtype"),
    )
    with testing.verified("llama"):
        assert (
            warm_dot(FormatLinear, _Only((mx.float32,)), 8, 1, 320, tensors, layout)
            is None
        )


def test_warm_dot_skips_non_tile_aligned_n(monkeypatch):
    from omlx.formats import testing

    (codes, scales), _, _ = _matrix(12, 320)
    tensors = {"codes": codes, "group_scales": scales}
    layout = {"codes": ((12, 320), mx.uint8), "group_scales": ((12,), mx.float32)}
    monkeypatch.setattr(
        "omlx.formats.linear.kernels.warm",
        lambda instances: pytest.fail("warmed a non-tile N"),
    )
    with testing.verified("llama"):
        assert warm_dot(FormatLinear, _Only(()), 8, 1, 320, tensors, layout) is None


def test_a_checkpoint_warms_each_geometry_separately(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    planned = []

    def warm_dot(layer_cls, fmt, bits, group_size, k, tensors, layout):
        planned.append(
            (layer_cls.__name__, fmt.mode, k, bits, group_size)
        )
        return [_Future()]

    monkeypatch.setattr(format_load, "warm_dot", warm_dot)

    def built(**changes):
        return SimpleNamespace(
            **{
                "output_dims": 8,
                "bits": 4,
                "group_size": 64,
                "mode": "toy",
                **changes,
            }
        )

    class Other:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    for layer, mode, k, output_dims, bits, group_size in (
        (SimpleNamespace, "toy", 320, 8, 4, 64),
        (SimpleNamespace, "toy", 512, 8, 4, 64),
        (SimpleNamespace, "toy", 320, 16, 4, 64),
        (SimpleNamespace, "toy", 320, 8, 8, 64),
        (SimpleNamespace, "toy", 320, 8, 4, 32),
        (SimpleNamespace, "other", 320, 8, 4, 64),
        (Other, "toy", 320, 8, 4, 64),
    ):
        layer_built = layer(
            output_dims=output_dims, bits=bits, group_size=group_size
        )
        format_load._warm_layer(
            layer_built, SimpleNamespace(mode=mode), {}, k, {}, checkpoint
        )
    assert planned == [
        ("SimpleNamespace", "toy", 320, 4, 64),
        ("SimpleNamespace", "toy", 512, 4, 64),
        ("SimpleNamespace", "toy", 320, 4, 64),
        ("SimpleNamespace", "toy", 320, 8, 64),
        ("SimpleNamespace", "toy", 320, 4, 32),
        ("SimpleNamespace", "other", 320, 4, 64),
        ("Other", "toy", 320, 4, 64),
    ]
    assert len(checkpoint.warmers) == 7


def test_failed_load_still_joins_warmers(tmp_path):
    from omlx.patches import format_load

    future = _Future()
    with (
        pytest.raises(RuntimeError, match="the load failed"),
        format_load.loading(tmp_path) as checkpoint,
    ):
        checkpoint.warmers.append(future)
        raise RuntimeError("the load failed")
    assert future.awaited == 1 and checkpoint.warmers == []


def test_concurrent_join_warmers_waits_for_in_flight_builds(tmp_path):
    import threading

    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    entered, release = threading.Event(), threading.Event()

    class Slow(_Future):
        def result(self, timeout=None):
            entered.set()
            assert release.wait(10)
            return super().result(timeout)

    checkpoint.warmers.append(Slow())
    first = threading.Thread(target=checkpoint.join_warmers)
    first.start()
    assert entered.wait(5)
    second_done = threading.Event()

    def second():
        checkpoint.join_warmers()
        second_done.set()

    second_thread = threading.Thread(target=second)
    second_thread.start()
    try:
        assert not second_done.wait(0.2)
    finally:
        release.set()
    assert second_done.wait(5)
    first.join(5)
    second_thread.join(5)


def test_check_loaded_joins_the_warmers_it_finds(tmp_path):
    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    future = _Future()
    checkpoint.warmers.append(future)
    joins = []
    real_join = checkpoint.join_warmers

    def counted_join():
        joins.append(True)
        return real_join()

    checkpoint.join_warmers = counted_join
    walks = []

    class TrackedModel(nn.Module):
        def named_modules(self):
            walks.append(True)
            return super().named_modules()

    model = TrackedModel()
    model.first = nn.Module()
    model.second = nn.Module()
    model.first.__dict__["_load_origin"] = ({}, checkpoint)
    model.second.__dict__["_load_origin"] = ({}, checkpoint)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(format_load, "_BUILT_ANY", True)
        format_load.check_loaded(model)
    assert future.awaited == 1
    assert len(joins) == 1
    assert len(walks) == 1


def test_planned_deduplicates_same_raw_kernel():
    def raw(**kwargs):
        return []

    specs = [((8,), mx.float32)]
    first = kernels.planned(raw, specs, [mx.float32], template=[("K", 8)])
    second = kernels.planned(raw, specs, [mx.float32], template=[("K", 8)])
    assert first == second


def test_face_kernel_records_generator_launch_arguments(monkeypatch):
    def native(**kw):
        list(kw["inputs"])
        list(kw["output_dtypes"])
        list(kw["template"])
        return []

    call = kernel(monkeypatch, native)
    call(
        inputs=(x for x in [mx.zeros((8,), stream=mx.cpu)]),
        output_shapes=[(1,)],
        output_dtypes=(d for d in [mx.float16]),
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        template=(entry for entry in [("K", 8)]),
    )
    assert call.instances() == (
        call.instance([((8,), mx.float32)], [mx.float16], template=[("K", 8)]),
    )


def test_a_stand_in_of_another_input_count_is_refused(monkeypatch):
    call = kernel(monkeypatch, lambda **kw: [])
    launch(call, mx.zeros((8,), stream=mx.cpu))
    (instance,) = call.instances()

    def short():
        return kernels.StandIn(((8,), (8,)), ((1,),), (1, 1, 1), (1, 1, 1))

    (future,) = kernels.warm({instance: short})
    with pytest.raises(ValueError, match="stand-in"):
        future.result(timeout=5)


def test_a_stack_does_not_query_its_dot_face(monkeypatch):
    from omlx import formats
    from tests.test_formats_linear import _toy_switch, _ToyFormat

    monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(_toy("stack")))
    monkeypatch.setattr(
        "omlx.formats.linear.kernels.planned",
        lambda *arguments, **keywords: pytest.fail("planned a stack's Metal"),
    )
    monkeypatch.setattr(
        "omlx.formats.linear.kernels.warm",
        lambda *arguments, **keywords: pytest.fail("warmed a stack's Metal"),
    )
    with testing.verified("llama"):
        switch = _toy_switch(8, 320)
        tensors = {name: switch[name] for name in switch._tensors}
        fmt = switch.fmt
        monkeypatch.setattr(
            fmt, "dot", lambda *arguments: pytest.fail("asked a stack's face")
        )
        assert warm_dot(FormatSwitchLinear, fmt, 8, 1, 320, tensors, None) is None


def test_a_verification_load_starts_no_warm(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    called = []

    def warm_dot(*arguments):
        called.append(arguments)
        return None

    monkeypatch.setattr(format_load, "warm_dot", warm_dot)
    built = SimpleNamespace(output_dims=8, bits=4, group_size=64)
    with format_load.verifying(tmp_path):
        format_load._warm_layer(
            built, SimpleNamespace(mode="toy"), {}, 320, {}, checkpoint
        )
    assert called == []
    format_load._warm_layer(
        built, SimpleNamespace(mode="toy"), {}, 320, {}, checkpoint
    )
    assert len(called) == 1


def test_concurrent_warm_layer_plans_once_and_joins_on_exit(monkeypatch, tmp_path):
    import threading
    from types import SimpleNamespace

    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    future = _Future()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def warm_dot(*arguments):
        calls.append(arguments)
        entered.set()
        assert release.wait(10)
        return [future]

    monkeypatch.setattr(format_load, "warm_dot", warm_dot)
    built = SimpleNamespace(output_dims=8, bits=4, group_size=64)
    first = threading.Thread(
        target=format_load._warm_layer,
        args=(built, SimpleNamespace(mode="toy"), {}, 320, {}, checkpoint),
    )
    first.start()
    assert entered.wait(5)
    second = threading.Thread(
        target=format_load._warm_layer,
        args=(built, SimpleNamespace(mode="toy"), {}, 320, {}, checkpoint),
    )
    second.start()
    joiner = threading.Thread(target=checkpoint.join_warmers)
    joiner.start()
    try:
        checkpoint.active = False
    finally:
        release.set()
    first.join(10)
    second.join(10)
    joiner.join(10)
    format_load._warm_layer(
        built, SimpleNamespace(mode="toy"), {}, 512, {}, checkpoint
    )
    assert len(calls) == 1
    assert future.awaited == 1 and checkpoint.warmers == []


def test_warm_layer_uses_cached_checkpoint_shown(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    _ = checkpoint.shown
    monkeypatch.setattr(
        format_load, "_shown", lambda path: pytest.fail("resolved the path again")
    )
    monkeypatch.setattr(format_load, "warm_dot", lambda *arguments: None)
    built = SimpleNamespace(output_dims=8, bits=4, group_size=64)
    format_load._warm_layer(built, SimpleNamespace(mode="toy"), {}, 320, {}, checkpoint)


def test_stand_in_rejects_fewer_inputs_and_scalar_mismatch(monkeypatch):
    call = kernel(monkeypatch, lambda **kw: [])
    launch(call, mx.zeros((8,), stream=mx.cpu))
    (instance,) = call.instances()

    def fewer():
        return kernels.StandIn((), ((1,),), (1, 1, 1), (1, 1, 1))

    (future,) = kernels.warm({instance: fewer})
    with pytest.raises(ValueError, match="stand-in"):
        future.result(timeout=5)
    constant = call.instance([((1,), mx.float32)], [mx.float32])
    (future,) = kernels.warm(
        {constant: lambda: kernels.StandIn(((),), ((1,),), (1, 1, 1), (1, 1, 1))}
    )
    with pytest.raises(ValueError, match="stand-in"):
        future.result(timeout=5)
    scalar = call.instance([((), mx.float32)], [mx.float32])
    (future,) = kernels.warm(
        {scalar: lambda: kernels.StandIn(((1,),), ((1,),), (1, 1, 1), (1, 1, 1))}
    )
    with pytest.raises(ValueError, match="stand-in"):
        future.result(timeout=5)


def test_planned_on_face_kernel_uses_its_own_registry(monkeypatch):
    call = kernel(monkeypatch, lambda **kw: [])
    instance = kernels.planned(call, [((8,), mx.float32)], [mx.float32])
    assert call.instances() == (instance,)
    assert instance in kernels.instances()


def test_zeros_respects_explicit_stream(monkeypatch):
    streams = []
    original = (mx.zeros, mx.as_strided, mx.view, mx.reshape)

    def record(function):
        def called(*arguments, **keywords):
            streams.append(keywords.get("stream"))
            return function(*arguments, **keywords)

        return called

    monkeypatch.setattr(mx, "zeros", record(original[0]))
    monkeypatch.setattr(mx, "as_strided", record(original[1]))
    monkeypatch.setattr(mx, "view", record(original[2]))
    monkeypatch.setattr(mx, "reshape", record(original[3]))
    kernels.zeros(((8,), mx.float32), stream=mx.cpu)
    assert streams and all(stream is mx.cpu for stream in streams)


def test_loading_exit_waits_before_deactivating_in_flight_warm_dot(tmp_path, caplog):
    import contextvars
    import threading

    from omlx.patches import format_load

    entered, release = threading.Event(), threading.Event()
    pause = [False]

    class Paused(FormatLinear):
        def __init__(self, *arguments, **keywords):
            if pause[0]:
                entered.set()
                assert release.wait(10)
            super().__init__(*arguments, **keywords)

    fmt = _Only(())
    tensors = {
        "codes": mx.zeros((16, 320), mx.uint8),
        "group_scales": mx.zeros((16,)),
    }
    layout = {"codes": ((16, 320), mx.uint8), "group_scales": ((16,), mx.float32)}
    with testing.verified("llama"):
        built = Paused(fmt, 8, 1, 320, tensors, None, _load_layout=layout)
    pause[0] = True
    finished = threading.Event()
    seen_checkpoint = []

    def load_side():
        with format_load.loading(tmp_path) as checkpoint:
            checkpoint.verified = True
            seen_checkpoint.append(checkpoint)
            context = contextvars.copy_context()
            worker = threading.Thread(
                target=context.run,
                args=(
                    format_load._warm_layer,
                    built,
                    fmt,
                    layout,
                    320,
                    tensors,
                    checkpoint,
                ),
            )
            worker.start()
            assert entered.wait(5)
        # The exit waits for the paused plan (under the warming lock) here.
        worker.join(10)
        finished.set()

    loader = threading.Thread(target=load_side)
    loader.start()
    assert entered.wait(5)
    release.set()
    assert finished.wait(10)
    loader.join(10)
    assert caplog.records == []
    assert len(seen_checkpoint[0].warmed) == 1


def test_warm_dot_starts_nothing_without_a_face(monkeypatch):
    class NoFace(QuantFormat):
        mode, codes_name, tensors, contract = "toy", "codes", ("group_scales",), 1

        def dot(self, layer, k, n):
            return None

    (codes, scales), _, _ = _matrix(16, 320)
    tensors = {"codes": codes, "group_scales": scales}
    layout = {"codes": ((16, 320), mx.uint8), "group_scales": ((16,), mx.float32)}
    monkeypatch.setattr(
        "omlx.formats.linear.kernels.warm",
        lambda instances: pytest.fail("warmed a layer with no face"),
    )
    with testing.verified("llama"):
        assert warm_dot(FormatLinear, NoFace(), 8, 1, 320, tensors, layout) is None


def test_warm_dot_skips_k_under_eight(monkeypatch):
    (codes, scales), _, _ = _matrix(16, 4)
    tensors = {"codes": codes, "group_scales": scales}
    layout = {"codes": ((16, 4), mx.uint8), "group_scales": ((16,), mx.float32)}
    fmt = _Only(())
    monkeypatch.setattr(
        fmt, "dot", lambda *arguments: pytest.fail("queried dot for k under 8")
    )

    class FailInit(FormatLinear):
        def __init__(self, *arguments, **keywords):
            pytest.fail("built stand-in layer for k under 8")

    with testing.verified("llama"):
        assert warm_dot(FailInit, fmt, 8, 1, 4, tensors, layout) is None
        (codes8, scales8), _, _ = _matrix(16, 8)
        tensors8 = {"codes": codes8, "group_scales": scales8}
        layout8 = {"codes": ((16, 8), mx.uint8), "group_scales": ((16,), mx.float32)}
        planned, seen = _warm_plan(
            monkeypatch, tensors8, layout8, (mx.bfloat16,), n=16, k=8
        )
        assert planned is not None and len(seen["specs"]) == 1


def test_warm_dot_passes_the_declared_layout(monkeypatch):
    layouts = []

    class Counting(QuantFormat):
        mode, codes_name, tensors, contract = "toy", "codes", ("group_scales",), 1

        def layout(self, k, n, bits, group_size):
            layouts.append((k, n, bits, group_size))
            return {
                "codes": ((n, k), mx.uint8),
                "group_scales": ((n,), mx.float32),
            }

        def dot(self, layer, k, n):
            return replace(_toy("warm_layout"), dtypes=())

    (codes, scales), _, _ = _matrix(16, 320)
    tensors = {"codes": codes, "group_scales": scales}
    layout = {"codes": ((16, 320), mx.uint8), "group_scales": ((16,), mx.float32)}
    monkeypatch.setattr(
        "omlx.formats.linear.kernels.warm", lambda instances: (("future",),)
    )
    monkeypatch.setattr(
        "omlx.formats.linear.kernels.planned",
        lambda kernel, specs, dtypes, *, template: "m",
    )
    with testing.verified("llama"):
        assert (
            warm_dot(FormatLinear, Counting(), 8, 1, 320, tensors, layout)
            is not None
        )
    assert layouts == []


def test_a_template_of_another_type_is_refused():
    def raw(**kwargs):
        return []

    with pytest.raises(TypeError, match="template"):
        kernels.planned(raw, [((8,), mx.float32)], [mx.float32], template=[("K", 1.5)])


def test_zeros_streams_a_two_gigabyte_input(monkeypatch):
    streams = []
    original = mx.zeros

    def record(*arguments, **keywords):
        streams.append(keywords.get("stream"))
        return original(*arguments, **keywords)

    monkeypatch.setattr(mx, "zeros", record)
    kernels.zeros(((2**29,), mx.float32), stream=mx.cpu)
    assert streams and all(stream is mx.cpu for stream in streams)


def test_sg8_launch_records_only_its_first_call(monkeypatch):
    from omlx.patches import qwen35_verify_qmm

    records = []
    kernel = kernels._FaceKernel(lambda **call: [call])
    monkeypatch.setattr(kernels, "face_kernel", lambda *arguments, **keywords: kernel)
    monkeypatch.setattr(
        kernel, "_record", lambda *arguments, **keywords: records.append(arguments)
    )
    launch = qwen35_verify_qmm._sg8_launch(
        (_toy("sg8"),), 3, 64, 16, 32, mx.float32, False
    )
    x = mx.zeros((3, 64), stream=mx.cpu)
    weights = mx.zeros((16, 64), mx.uint8)
    launch(inputs=[x, weights])
    launch(inputs=[x, weights])
    assert len(records) == 1


def test_a_layer_without_a_face_is_queried_once_per_geometry(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from omlx.patches import format_load

    checkpoint = format_load._Checkpoint(tmp_path)
    calls = []

    def warm_dot(*arguments):
        calls.append(arguments)
        return None

    monkeypatch.setattr(format_load, "warm_dot", warm_dot)
    built = SimpleNamespace(output_dims=8, bits=4, group_size=64)
    for _ in range(2):
        format_load._warm_layer(
            built, SimpleNamespace(mode="toy"), {}, 320, {}, checkpoint
        )
    assert len(calls) == 1
    assert checkpoint.warmed == set() and len(checkpoint._checked) == 1


def test_sg8_launch_records_after_failed_first_call(monkeypatch):
    from omlx.patches import qwen35_verify_qmm

    records = []
    attempts = [0]

    def native(**call):
        attempts[0] += 1
        if attempts[0] == 1:
            raise ValueError("first launch failed")
        return [call]

    wrapped = kernels._FaceKernel(native)
    real_record = wrapped._record

    def counted_record(*arguments, **keywords):
        records.append(arguments)
        return real_record(*arguments, **keywords)

    monkeypatch.setattr(wrapped, "_record", counted_record)
    monkeypatch.setattr(kernels, "face_kernel", lambda *arguments, **keywords: wrapped)
    launch = qwen35_verify_qmm._sg8_launch(
        (_toy("sg8_retry"),), 3, 64, 16, 32, mx.float32, False
    )
    x = mx.zeros((3, 64), stream=mx.cpu)
    weights = mx.zeros((16, 64), mx.uint8, stream=mx.cpu)
    with pytest.raises(ValueError, match="first launch failed"):
        launch(inputs=[x, weights])
    assert records == []
    launch(inputs=[x, weights])
    assert len(records) == 1
    launch(inputs=[x, weights])
    assert len(records) == 1


def test_sg8_call_records_distinct_operand_dtypes(monkeypatch):
    from omlx.patches import qwen35_verify_qmm

    records = []
    wrapped = kernels._FaceKernel(lambda **call: [call])
    real_record = wrapped._record

    def counted_record(*arguments, **keywords):
        records.append(arguments)
        return real_record(*arguments, **keywords)

    monkeypatch.setattr(wrapped, "_record", counted_record)
    monkeypatch.setattr(kernels, "face_kernel", lambda *arguments, **keywords: wrapped)
    face = _toy("sg8_dtypes")
    x = mx.zeros((3, 64), mx.bfloat16, stream=mx.cpu)
    w_u8 = mx.zeros((16, 64), mx.uint8, stream=mx.cpu)
    w_u32 = mx.zeros((16, 16), mx.uint32, stream=mx.cpu)
    qwen35_verify_qmm._sg8_call(x, (face,), [w_u8], 16, 32, None)
    qwen35_verify_qmm._sg8_call(x, (face,), [w_u32], 16, 32, None)
    qwen35_verify_qmm._sg8_call(x, (face,), [w_u8], 16, 32, None)
    assert len(records) == 2
    assert len(wrapped.instances()) == 2


def test_warm_stream_creates_one_stream_under_concurrency(monkeypatch):
    import time

    if hasattr(kernels._warm_stream, "cache_clear"):
        kernels._warm_stream.cache_clear()
    monkeypatch.setattr(kernels, "_WARM_STREAM", None, raising=False)
    entered, release = threading.Event(), threading.Event()
    calls = []
    first_stream = object()

    def new_stream(device):
        idx = len(calls)
        calls.append(device)
        if idx == 0:
            entered.set()
            assert release.wait(5)
            return first_stream
        return object()

    monkeypatch.setattr(mx, "new_thread_local_stream", new_stream)
    results = []

    def worker():
        results.append(kernels._warm_stream())

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    assert entered.wait(5)
    t2.start()
    time.sleep(0.05)
    release.set()
    t1.join(5)
    t2.join(5)
    assert calls == [mx.gpu]
    assert results == [first_stream, first_stream]
    assert results[0] is first_stream and results[1] is first_stream


def test_warm_dot_skips_non_tile_aligned_n_without_querying_dot(monkeypatch):
    (codes, scales), _, _ = _matrix(12, 320)
    tensors = {"codes": codes, "group_scales": scales}
    layout = {"codes": ((12, 320), mx.uint8), "group_scales": ((12,), mx.float32)}
    fmt = _Only(())
    monkeypatch.setattr(
        fmt, "dot", lambda *arguments: pytest.fail("queried dot for non-tile N")
    )
    with testing.verified("llama"):
        assert warm_dot(FormatLinear, fmt, 8, 1, 320, tensors, layout) is None


def test_warm_skips_already_built_instance(monkeypatch):
    native_calls = []
    zeros_calls = []

    def native(**kwargs):
        native_calls.append(kwargs)
        return ["out"]

    call = kernel(monkeypatch, native)
    instance = call.instance([((8,), mx.float32)], [mx.float32])
    monkeypatch.setattr(kernels, "_warm_stream", lambda: mx.cpu)

    def counted_zeros(*specs, stream):
        zeros_calls.append((specs, stream))
        return [mx.zeros(shape, dtype, stream=stream) for shape, dtype in specs]

    monkeypatch.setattr(kernels, "zeros", counted_zeros)
    eval_calls = [0]

    def maybe_fail_eval(*outputs):
        eval_calls[0] += 1

    monkeypatch.setattr(mx, "eval", maybe_fail_eval)
    def recipe():
        return kernels.StandIn(((8,),), ((1,),), (1, 1, 1), (1, 1, 1))
    (first,) = kernels.warm({instance: recipe})
    first.result(timeout=5)
    (second,) = kernels.warm({instance: recipe})
    second.result(timeout=5)
    assert len(native_calls) == 1
    assert len(zeros_calls) == 1
    # A failed mx.eval must not mark the instance built, so a retry runs.
    flaky = call.instance([((8,), mx.int32)], [mx.int32])
    attempts = [0]

    def fail_once(*outputs):
        attempts[0] += 1
        if attempts[0] == 1:
            raise RuntimeError("eval failed")

    monkeypatch.setattr(mx, "eval", fail_once)
    (bad,) = kernels.warm({flaky: recipe})
    with pytest.raises(RuntimeError, match="eval failed"):
        bad.result(timeout=5)
    (retry,) = kernels.warm({flaky: recipe})
    retry.result(timeout=5)
    assert len(native_calls) == 3


def test_warm_dot_recipe_holds_no_checkpoint_tensors(monkeypatch):
    captured = []

    def fake_warm(instances):
        captured.extend(instances.values())
        return (("future",),)

    monkeypatch.setattr("omlx.formats.linear.kernels.warm", fake_warm)
    (codes, scales), _, _ = _matrix(16, 320)
    codes_ref, scales_ref = weakref.ref(codes), weakref.ref(scales)
    tensors = {"codes": codes, "group_scales": scales}
    layout = {"codes": ((16, 320), mx.uint8), "group_scales": ((16,), mx.float32)}
    with testing.verified("llama"):
        planned = warm_dot(
            FormatLinear, _Only((mx.bfloat16,)), 8, 1, 320, tensors, layout
        )
    assert planned == (("future",),) and len(captured) == 1
    del tensors, codes, scales
    gc.collect()
    assert codes_ref() is None and scales_ref() is None
    stand_in = captured[0]()
    assert stand_in.input_shapes == ((320,), (16, 320), (16,))
