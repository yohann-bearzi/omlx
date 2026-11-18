# SPDX-License-Identifier: Apache-2.0
"""The frozen oracle of the legacy ``sg8`` kernels (``tests/oracles/sg8.py``):
a verbatim copy while the live kernels exist, and a reference that tells
their rows apart bit for bit."""

from __future__ import annotations

import inspect
from functools import cache

import mlx.core as mx
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
    for name in _FUNCTIONS:
        assert inspect.getsource(getattr(oracle, name)) == inspect.getsource(
            getattr(live, name)
        ), name
    for name in _CONSTANTS:
        assert getattr(oracle, name) == getattr(live, name), name
    # Kernels of its own: the live cache never holds the oracle's.
    assert oracle._KERNEL_CACHE is not live._KERNEL_CACHE
    # And the Metal it generates, for every form a kernel takes.
    for mats in (1, 2):
        for half in (False, True):
            source = oracle._sg8_source(mats, half)
            assert source == live._sg8_source(mats, half)
            assert oracle._with_group_sums(source) == live._with_group_sums(source)
    for ns, bits in (((1024,), (4,)), ((4096, 512, 1024), (4, 5, 4))):
        for half in (False, True):
            group = oracle._sg8_group_source(ns, bits, half)
            assert group == live._sg8_group_source(ns, bits, half)
            assert oracle._with_group_sums(group) == live._with_group_sums(group)


@cache
def _linear(n, k, bits, group_size, dtype, seed):
    """An affine matrix as MLX quantizes it, with what sg8 reads of a layer."""
    mx.random.seed(seed)
    w = mx.random.normal((n, k), dtype=dtype, scale=0.05)
    weight, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    layer = type("Quantized", (), {})()
    layer.weight, layer.bits, layer.group_size = weight, bits, group_size
    layer.scales, layer.biases = scales.astype(dtype), biases.astype(dtype)
    mx.eval(layer.weight, layer.scales, layer.biases)
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


def _valid_group(rows, k, n, bits, group_size):
    """What the oracle's group kernel is a reference for, as the live
    group's eligibility states it (without the routing's floors)."""
    assert 1 <= rows <= 8 and bits in (4, 5) and group_size in (32, 64, 128)
    assert k % (group_size * oracle._SG8_GROUP_NSG) == 0 and n % 8 == 0


def _valid(rows, k, n, bits, group_size):
    """What the oracle is a reference for, as the live kernels' eligibility
    states it (without the routing's row and N floors): no shape outside."""
    nsg, ct = oracle._sg8_geometry(k, n)
    assert 1 <= rows <= 8 and bits in (2, 3, 4, 5, 6) and group_size in (32, 64, 128)
    assert k % (group_size * nsg) == 0 and n % (8 * ct) == 0


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


def _single_and_pair(x, layer, gate, up, bits, group_size):
    """A matrix, then the gate/up pair, against the oracle."""
    for sums in _sums_of(x):
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
@pytest.mark.parametrize("rows", [1, 3, 4, 6, 7, 8])
def test_a_matrix_and_a_gate_up_pair_are_the_oracle_s_rows(
    dtype, bits, group_size, rows
):
    k = 1024
    _valid(rows, k, 1024, bits, group_size)
    _valid(rows, k, 2048, bits, group_size)
    x = _x(rows, k, dtype, seed=rows)
    layer = _linear(1024, k, bits, group_size, dtype, seed=bits)
    gate = _linear(2048, k, bits, group_size, dtype, seed=bits + 10)
    up = _linear(2048, k, bits, group_size, dtype, seed=bits + 20)
    _single_and_pair(x, layer, gate, up, bits, group_size)


# The geometries: eight K parts of one tile, of two tiles (K from 16384),
# four parts of four tiles (N from 65536); the gate/up pair's second matrix
# in each.
@_GPU
@pytest.mark.parametrize("k,n", [(16384, 1024), (512, 65536)])
def test_every_geometry_is_the_oracle_s(k, n):
    _valid(8, k, n, 4, 64)
    x = _x(8, k, mx.bfloat16, seed=k)
    layer = _linear(n, k, 4, 64, mx.bfloat16, seed=n)
    gate = _linear(n, k, 4, 64, mx.bfloat16, seed=n + 1)
    up = _linear(n, k, 4, 64, mx.bfloat16, seed=n + 2)
    _single_and_pair(x, layer, gate, up, 4, 64)


@_GPU
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("rows", [1, 4, 6, 7, 8])
def test_a_group_of_matrices_is_the_oracle_s(dtype, group_size, rows):
    """4- and 5-bit matrices of one input in one launch."""
    k = 1024
    x = _x(rows, k, dtype, seed=rows + group_size)
    group = [
        _linear(n, k, bits, group_size, dtype, seed=n + bits)
        for n, bits in ((4096, 4), (512, 5), (1024, 4))
    ]
    for linear in group:
        _valid_group(rows, k, linear.scales.shape[0], linear.bits, group_size)
    for sums in _sums_of(x):
        got = live.vk_group_sg8(x, group, sums)
        want = oracle.vk_group_sg8(x, group, sums)
        assert len(got) == len(want) == 3
        assert all(_same(a, b) for a, b in zip(got, want, strict=True))


@_GPU
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_a_gate_of_zeros_gives_rows_of_negative_zeros_as_the_oracle(dtype):
    """silu(+0) times a negative up row is -0.0: the pair keeps the sign."""
    k, rows = 1024, 8
    _valid(rows, k, 2048, 4, 64)
    x = _x(rows, k, dtype, seed=5)
    gate = _linear(2048, k, 4, 64, dtype, seed=30)
    zeros = type("Quantized", (), {})()
    zeros.weight, zeros.bits, zeros.group_size = gate.weight, 4, 64
    zeros.scales = mx.zeros_like(gate.scales)
    zeros.biases = mx.zeros_like(gate.biases)
    up = _linear(2048, k, 4, 64, dtype, seed=31)
    for sums in _sums_of(x):
        want = oracle.vk_swiglu_sg8(x, zeros, up, sums)
        assert (mx.view(want, mx.uint16) == 0x8000).any().item()
        assert _same(live.vk_swiglu_sg8(x, zeros, up, sums), want)


@pytest.fixture(autouse=True, scope="module")
def _layers_end_with_the_module():
    """The quantized layers the tests share live as long as this file's."""
    yield
    _linear.cache_clear()
