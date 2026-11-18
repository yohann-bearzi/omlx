# SPDX-License-Identifier: Apache-2.0
"""The frozen oracle of the legacy ``sg8`` kernels (``tests/oracles/sg8.py``):
a verbatim copy while the live kernels exist, and a reference that tells
their rows apart bit for bit."""

from __future__ import annotations

import inspect
from functools import cache

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_verify_qmm as live
from tests.oracles import sg8 as oracle

# The comparisons run kernels; the verbatim check runs anywhere.
_GPU = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# What the oracle copies of the live module: its functions and its constants.
_FUNCTIONS = (
    "_operand_bits",
    "_sg8_source",
    "_with_group_sums",
    "_build_sg8_kernel",
    "_sg8_geometry",
    "_sg8_call",
    "vk_qmm_sg8",
    "vk_swiglu_sg8",
    "group_sums",
    "_sg8_group_source",
    "vk_group_sg8",
)
_CONSTANTS = (
    "_XSUM_ACCUM",
    "_XSUM_REDUCE",
    "_SUM_GROUP",
    "_GROUP_SUMS_SOURCE",
    "_SG8_GROUP_NSG",
)


def test_the_oracle_is_the_live_code_verbatim():
    frozen = oracle.legacy
    for name in _FUNCTIONS:
        assert inspect.getsource(getattr(frozen, name)) == inspect.getsource(
            getattr(live, name)
        ), name
    for name in _CONSTANTS:
        assert getattr(frozen, name) == getattr(live, name), name
    # Kernels of its own: the live cache never holds the oracle's.
    assert frozen._KERNEL_CACHE is not live._KERNEL_CACHE
    # And the Metal it generates, for every form a kernel takes.
    for mats in (1, 2):
        for half in (False, True):
            source = frozen._sg8_source(mats, half)
            assert source == live._sg8_source(mats, half)
            assert frozen._with_group_sums(source) == live._with_group_sums(source)
    for ns, bits in (((1024,), (4,)), ((4096, 512, 1024), (4, 5, 4))):
        for half in (False, True):
            group = frozen._sg8_group_source(ns, bits, half)
            assert group == live._sg8_group_source(ns, bits, half)
            assert frozen._with_group_sums(group) == live._with_group_sums(group)


def _weights(n, k, bits, group_size, dtype, seed):
    w = mx.random.normal((n, k), dtype=dtype, scale=0.05, key=mx.random.key(seed))
    tensors = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(*tensors)
    return tensors


_cached_weights = cache(_weights)


def _linear(n, k, bits, group_size, dtype, seed):
    # The two large geometries run once each. Only common tensors are kept.
    weights = _cached_weights if k == 1024 else _weights
    layer = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(layer)
    layer.bits, layer.group_size, layer.mode = bits, group_size, "affine"
    layer.weight, layer.scales, layer.biases = weights(
        n, k, bits, group_size, dtype, seed
    )
    layer.freeze()
    return layer


def _x(rows, k, dtype, seed):
    """Rows of activations, the first of zeros when there are two or more
    (the last row, a lane's second, keeps values)."""
    mx.random.seed(seed)
    x = (mx.random.normal((rows, k)) * 0.5).astype(dtype)
    if rows > 1:
        x[0] = 0
    mx.eval(x)
    return x


def _same(a, b):
    """Bit for bit: a zero's sign too."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return mx.array_equal(mx.view(a, mx.uint8), mx.view(b, mx.uint8)).item()


def test_rows_are_compared_by_their_bits():
    zero = mx.array([0.0], mx.bfloat16)
    assert _same(zero, mx.array([0.0], mx.bfloat16))
    assert not _same(zero, -zero) and not _same(zero, zero.astype(mx.float16))
    assert not _same(zero, mx.array([0.0, 0.0], mx.bfloat16))


def _sums_of(x):
    """No sums, and the per-64 sums a producer writes, which the kernels read
    at a group size of 64 and drop at any other, as the live callers pass
    them (the oracle's copy of the live kernel that writes them, checked
    against it, for rows of x in any shape)."""
    sums = oracle.group_sums(x)
    assert _same(sums, live.group_sums(x))
    rows = x.shape[0]
    batched = x.reshape(2, rows // 2, -1) if rows % 2 == 0 else x[None]
    assert _same(oracle.group_sums(batched), sums)
    assert _same(live.group_sums(batched), sums)
    return (None, sums)


@cache
def _input(rows, dtype):
    x = _x(rows, 1024, dtype, seed=rows)
    return x, _sums_of(x)


def _single_and_pair(x, variants, layer, gate, up, bits, group_size):
    """A matrix, then the gate/up pair, against the oracle."""
    for sums in variants:
        args = (layer.weight, layer.scales, layer.biases)
        kw = dict(group_size=group_size, bits=bits, sums=sums)
        assert _same(live.vk_qmm_sg8(x, *args, **kw), oracle.vk_qmm_sg8(x, *args, **kw))
        got = live.vk_swiglu_sg8(x, gate, up, sums)
        assert _same(got, oracle.vk_swiglu_sg8(x, gate, up, sums))


# Each width with each group size (64 reads producer sums), both activation
# types, and rows from one to eight: every pair of rows a lane stores.
@_GPU
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("rows", range(1, 9))
def test_a_matrix_and_a_gate_up_pair_are_the_oracle_s_rows(
    dtype, bits, group_size, rows
):
    k = 1024
    x, variants = _input(rows, dtype)
    layer = _linear(1024, k, bits, group_size, dtype, seed=bits)
    gate = _linear(2048, k, bits, group_size, dtype, seed=bits + 10)
    up = _linear(2048, k, bits, group_size, dtype, seed=bits + 20)
    _single_and_pair(x, variants, layer, gate, up, bits, group_size)


# The geometries: eight K parts of one tile, of two tiles (K from 16384),
# four parts of four tiles (N from 65536); the gate/up pair's second matrix
# in each.
@_GPU
@pytest.mark.parametrize("k,n", [(16384, 1024), (512, 65536)])
def test_every_geometry_is_the_oracle_s(k, n):
    x = _x(8, k, mx.bfloat16, seed=k)
    layer = _linear(n, k, 4, 64, mx.bfloat16, seed=n)
    gate = _linear(n, k, 4, 64, mx.bfloat16, seed=n + 1)
    up = _linear(n, k, 4, 64, mx.bfloat16, seed=n + 2)
    _single_and_pair(x, _sums_of(x), layer, gate, up, 4, 64)


@_GPU
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("rows", range(1, 9))
def test_a_group_of_matrices_is_the_oracle_s(dtype, group_size, rows):
    """4- and 5-bit matrices of one input in one launch."""
    k = 1024
    x, variants = _input(rows, dtype)
    group = [
        _linear(n, k, bits, group_size, dtype, seed=n + bits)
        for n, bits in ((512, 5), (4096, 4), (1024, 4))
    ]
    for sums in variants:
        got = live.vk_group_sg8(x, group, sums)
        want = oracle.vk_group_sg8(x, group, sums)
        assert len(got) == len(want) == 3
        assert all(_same(a, b) for a, b in zip(got, want, strict=True))


@_GPU
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_a_gate_of_zeros_gives_rows_of_negative_zeros_as_the_oracle(dtype):
    """silu(+0) times a negative up row is -0.0: the pair keeps the sign."""
    k, rows = 1024, 8
    x = _x(rows, k, dtype, seed=5)
    zeros = _linear(2048, k, 4, 64, dtype, seed=30)
    zeros.scales = mx.zeros_like(zeros.scales)
    zeros.biases = mx.zeros_like(zeros.biases)
    up = _linear(2048, k, 4, 64, dtype, seed=31)
    for sums in _sums_of(x):
        want = oracle.vk_swiglu_sg8(x, zeros, up, sums)
        assert (mx.view(want, mx.uint16) == 0x8000).any().item()
        assert _same(live.vk_swiglu_sg8(x, zeros, up, sums), want)


@pytest.fixture(autouse=True, scope="module")
def _layers_end_with_the_module():
    """The quantized layers the tests share live as long as this file's."""
    yield
    _cached_weights.cache_clear()
    _input.cache_clear()


def test_shared_weights_do_not_share_a_mutable_layer():
    first = _linear(1024, 1024, 4, 64, mx.float16, seed=4)
    second = _linear(1024, 1024, 4, 64, mx.float16, seed=4)
    assert isinstance(first, nn.QuantizedLinear) and first is not second
    assert first.weight is second.weight
    first.scales = mx.zeros_like(first.scales)
    assert first.scales is not second.scales


def _no_kernel(*args, **kwargs):
    pytest.fail("an invalid oracle input reached Metal")


@pytest.mark.parametrize(
    "rows,k,n,bits,group",
    [
        (0, 512, 16, 4, 64),
        (9, 512, 16, 4, 64),
        (1, 0, 16, 4, 64),
        (5, 256, 0, 4, 32),
        (5, 256, 9, 4, 32),
        (5, 256, 16, 4, 64),
        (5, 512, 16, 8, 64),
        (5, 512, 16, 4, 16),
    ],
)
def test_oracle_refuses_invalid_single_geometry(monkeypatch, rows, k, n, bits, group):
    monkeypatch.setattr(mx.fast, "metal_kernel", _no_kernel)
    x = mx.zeros((rows, k), mx.float16)
    w = mx.zeros((n, k * bits // 32), mx.uint32)
    s = mx.zeros((n, k // group), mx.float16)
    with pytest.raises(ValueError):
        oracle.vk_qmm_sg8(x, w, s, s, bits=bits, group_size=group)


@pytest.mark.parametrize(
    "defect",
    ["single", "group", "width", "scale_type", "bias_type", "empty_n", "k", "n"],
)
def test_oracle_refuses_invalid_group_members(monkeypatch, defect):
    from types import SimpleNamespace

    def layer(n, k=512, bits=4, group=64):
        return SimpleNamespace(
            group_size=group,
            bits=bits,
            weight=mx.zeros((n, k * bits // 32), mx.uint32),
            scales=mx.zeros((n, k // group), mx.float16),
            biases=mx.zeros((n, k // group), mx.float16),
        )

    projections = [layer(16), layer(8, bits=5)]
    if defect == "single":
        projections.pop()
    elif defect == "group":
        projections[1] = layer(8, group=32)
    elif defect == "width":
        projections[1] = layer(8, bits=6)
    elif defect == "scale_type":
        projections[1].scales = projections[1].scales.astype(mx.float32)
    elif defect == "bias_type":
        projections[1].biases = projections[1].biases.astype(mx.bfloat16)
    elif defect == "empty_n":
        projections[1] = layer(0)
    elif defect == "n":
        projections[1] = layer(9)
    else:
        projections[1] = layer(8, k=256)
    x = mx.zeros((5, 512), mx.float16)
    monkeypatch.setattr(mx.fast, "metal_kernel", _no_kernel)
    with pytest.raises(ValueError):
        oracle.vk_group_sg8(x, projections)


@pytest.mark.parametrize(
    "which", ["gate/up", "sums", "x", "packed", "scales", "biases"]
)
def test_oracle_refuses_incompatible_operands(monkeypatch, which):
    from types import SimpleNamespace

    w = mx.zeros((16, 64), mx.uint32)
    s = mx.zeros((16, 8), mx.float16)
    x = mx.zeros((4, 512), mx.float16)
    gate = SimpleNamespace(weight=w, scales=s, biases=s, bits=4, group_size=64)
    monkeypatch.setattr(mx.fast, "metal_kernel", _no_kernel)
    with pytest.raises(ValueError):
        if which == "gate/up":
            up = SimpleNamespace(**{**vars(gate), "bits": 5})
            oracle.vk_swiglu_sg8(x, gate, up)
        elif which == "sums":
            oracle.vk_qmm_sg8(x, w, s, s, group_size=64, sums=mx.zeros((4, 7)))
        elif which == "x":
            oracle.vk_qmm_sg8(x.astype(mx.float32), w, s, s, group_size=64)
        else:
            oracle.vk_qmm_sg8(
                x,
                w[:, :-1] if which == "packed" else w,
                s[:-1] if which == "scales" else s,
                s[:, 1:] if which == "biases" else s,
                group_size=64,
            )


@pytest.mark.parametrize("shape", [(0, 64), (2, 0), (2, 65), ()])
def test_oracle_group_sums_refuses_incomplete_rows(monkeypatch, shape):
    monkeypatch.setattr(mx.fast, "metal_kernel", _no_kernel)
    with pytest.raises(ValueError):
        oracle.group_sums(mx.zeros(shape, mx.float16))


@pytest.mark.parametrize(
    "which",
    [
        "packed dtype",
        "packed rank",
        "activation rank",
        "sum dtype",
        "gate sum",
        "group sum",
        "up shape",
        "up groups",
        "up bias",
    ],
)
def test_oracle_validates_each_bound_tensor(monkeypatch, which):
    from types import SimpleNamespace

    x = mx.zeros((4, 512), mx.float16)
    w = mx.zeros((16, 64), mx.uint32)
    s = mx.zeros((16, 8), mx.float16)
    gate = SimpleNamespace(weight=w, scales=s, biases=s, bits=4, group_size=64)
    up = SimpleNamespace(**vars(gate))
    monkeypatch.setattr(mx.fast, "metal_kernel", _no_kernel)
    with pytest.raises(ValueError):
        if which == "gate sum":
            oracle.vk_swiglu_sg8(x, gate, up, sums=mx.ones((4, 7)))
        elif which == "group sum":
            oracle.vk_group_sg8(x, (gate, up), sums=mx.ones((4, 7)))
        elif which.startswith("up"):
            if which == "up shape":
                up.weight, up.scales, up.biases = w[:8], s[:8], s[:8]
            if which == "up groups":
                up.group_size = 32
                up.scales = up.biases = mx.zeros((16, 16), mx.float16)
            if which == "up bias":
                up.biases = s[:, 1:]
            oracle.vk_swiglu_sg8(x, gate, up)
        else:
            if which == "packed dtype":
                w = w.astype(mx.int32)
            if which == "packed rank":
                w = w[..., None]
            if which == "activation rank":
                x = x[None]
            sums = mx.zeros((4, 8), mx.float16) if which == "sum dtype" else None
            oracle.vk_qmm_sg8(x, w, s, s, group_size=64, sums=sums)
