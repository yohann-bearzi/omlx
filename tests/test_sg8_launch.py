# SPDX-License-Identifier: Apache-2.0
"""The stock and bound routes can share launch metadata, but each gets its tensors."""

from concurrent.futures import ThreadPoolExecutor
from functools import cache
from threading import Barrier

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.formats import affine, kernels
from omlx.patches import qwen35_verify_qmm as live
from tests.oracles import sg8 as oracle

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _case(
    seed, *, rows=4, k=512, n=1032, bits=4, gs=64, dtype=mx.bfloat16, summed=True
):
    raw = mx.random.normal((n, k), scale=0.08, key=mx.random.key(seed))
    w, s, b = mx.quantize(raw, group_size=gs, bits=bits)
    layer = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(layer)
    layer.group_size, layer.bits, layer.mode = gs, bits, "affine"
    layer.weight = w
    layer.scales, layer.biases = s.astype(mx.float16), b.astype(mx.float32)
    layer.freeze()
    x = mx.random.normal((rows, k), key=mx.random.key(1900 + seed)).astype(dtype)
    sums = oracle.group_sums(x) + seed if summed else None
    expect = oracle.vk_qmm_sg8(
        x, w, layer.scales, layer.biases, group_size=gs, bits=bits, sums=sums
    )
    mx.eval(x, layer.weight, layer.scales, layer.biases, expect)
    if sums is not None:
        mx.eval(sums)
    return layer, x, sums, expect


def _check(route, data):
    layer, x, sums, expect = data
    if route == "stock":
        got = live.run_route(layer, x, "sg8", sums=sums)
    else:
        face = affine.codes_face(
            layer.bits, layer.group_size, layer.scales.dtype, layer.biases.dtype
        )
        got = live.sg8_codes(
            x,
            face,
            (layer.weight, layer.scales, layer.biases),
            n=layer.weight.shape[0],
            group_size=layer.group_size,
            sums=sums,
        )
    assert (got.shape, got.dtype) == (expect.shape, expect.dtype)
    assert mx.array_equal(got.view(mx.uint8), expect.view(mx.uint8)).item()


def test_stock_reuses_warm_geometry_for_new_tensors(monkeypatch):
    a, b = _case(11), _case(12)
    _check("stock", a)

    def repeated(*args):
        pytest.fail("stock recomputed its warm sg8 launch geometry")

    monkeypatch.setattr(live, "_sg8_geometry", repeated)
    _check("stock", b)


@pytest.mark.parametrize("route", ["stock", "bound"])
def test_threaded_sg8_launches_bind_the_actual_arrays_and_geometry(route):
    cases = [
        _case(1),
        _case(2),
        _case(3, summed=False),
        _case(4, rows=5, k=1024, n=1040, bits=5, dtype=mx.float16),
        _case(5, rows=6, gs=32),
    ]
    assert not mx.array_equal(cases[0][-1], cases[1][-1]).item()
    start = Barrier(2)

    def visit(indices):
        start.wait(timeout=20)
        for i in indices:
            _check(route, cases[i])

    with ThreadPoolExecutor(max_workers=2) as workers:
        a = workers.submit(visit, (0, 2, 3, 4, 1, 0))
        b = workers.submit(visit, (1, 4, 3, 2, 0, 1))
        a.result()
        b.result()


def test_stock_launch_constants_and_warm_sources(monkeypatch):
    first, second, other = _case(21), _case(22), _case(23, rows=5, k=1024, n=1040)
    builds, launches, headers = [], [], []
    original_kernel = mx.fast.metal_kernel
    original_header = kernels.Matrices.header.fget

    def header(matrices):
        headers.append(None)
        return original_header(matrices)

    def metal_kernel(**spec):
        kernel = original_kernel(**spec)
        if not spec["name"].startswith("omlx_codes_sg8_"):
            return kernel
        builds.append(spec["source"])

        def record(**launch):
            launches.append(tuple(launch["template"]))
            return kernel(**launch)

        return record

    # Isolate the builders so a warm call from another test cannot hide a build.
    for name, builder in vars(live).items():
        if "sg8" in name and hasattr(builder, "cache_clear"):
            monkeypatch.setattr(live, name, cache(builder.__wrapped__))
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})
    monkeypatch.setattr(kernels.Matrices, "header", property(header))
    monkeypatch.setattr(mx.fast, "metal_kernel", metal_kernel)

    _check("stock", first)
    first_counts = len(builds), len(headers)
    _check("stock", second)
    assert (len(builds), len(headers)) == first_counts == (1, 1)
    _check("stock", other)
    other_counts = len(builds), len(headers)
    _check("stock", first)
    _check("stock", other)
    assert (len(builds), len(headers)) == other_counts
    assert len(launches) == 5
    assert list(dict(launches[0])) == ["T"], launches[0]
    assert all(args == (("T", first[1].dtype),) for args in launches)
    assert len(builds) == 2


def test_swiglu_reuses_its_warm_launch(monkeypatch):
    """A second swiglu call on new tensors reuses the compiled faces: the
    pair's scalar key (bits, group size, the four dtypes) is cached."""
    gate, up = _case(21)[0], _case(22)[0]
    x = mx.random.normal((4, 512), key=mx.random.key(9021)).astype(mx.bfloat16)
    first = live.vk_swiglu_sg8(x, gate, up)
    mx.eval(first)

    def repeated(*args):
        pytest.fail("swiglu rebuilt the codes faces on a warm launch")

    monkeypatch.setattr(affine, "codes_face", repeated)
    again = live.vk_swiglu_sg8(x, gate, up)
    assert mx.array_equal(first.view(mx.uint8), again.view(mx.uint8)).item()


def test_swiglu_refuses_mismatched_pair():
    """The fused kernel has one bits and group size: a pair that differs is
    refused rather than reading one matrix through the other's face."""
    gate = _case(11)[0]
    up_bits = _case(13, bits=5)[0]
    up_gs = _case(14, gs=32)[0]
    x = mx.ones((4, 512), mx.bfloat16)
    with pytest.raises(ValueError, match="one bits and group size"):
        live.vk_swiglu_sg8(x, gate, up_bits)
    with pytest.raises(ValueError, match="one bits and group size"):
        live.vk_swiglu_sg8(x, gate, up_gs)
