# SPDX-License-Identifier: Apache-2.0
"""The frozen oracle of the legacy ``sg8`` kernels (``tests/oracles/sg8.py``):
a verbatim copy while the live kernels exist, and a reference that tells
their rows apart bit for bit."""

from __future__ import annotations

import inspect

import mlx.core as mx
import pytest

from omlx.patches import qwen35_verify_qmm as live
from tests.oracles import sg8 as oracle

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

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


def _linear(n, k, bits, group_size, dtype, seed):
    """An affine matrix as MLX quantizes it, with what sg8 reads of a layer."""
    mx.random.seed(seed)
    w = mx.random.normal((n, k)) * 0.05
    weight, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    layer = type("Quantized", (), {})()
    layer.weight, layer.bits, layer.group_size = weight, bits, group_size
    layer.scales, layer.biases = scales.astype(dtype), biases.astype(dtype)
    mx.eval(layer.weight, layer.scales, layer.biases)
    return layer


def _x(rows, k, dtype, seed):
    mx.random.seed(seed)
    x = (mx.random.normal((rows, k)) * 0.5).astype(dtype)
    mx.eval(x)
    return x


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


def _sums_of(x):
    """No sums, and the per-64 sums a producer writes (the oracle's copy of
    the live kernel that writes them, checked against it)."""
    sums = oracle.group_sums(x)
    assert _same(sums, live.group_sums(x))
    return (None, sums)


# Each width with each group size, both activation types, and rows from the
# fewest sg8 takes to its eight.
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "bits,group_size", [(2, 32), (3, 64), (4, 128), (5, 64), (6, 32), (4, 64)]
)
@pytest.mark.parametrize("rows", [4, 5, 8])
def test_a_matrix_and_a_gate_up_pair_are_the_oracle_s_rows(
    dtype, bits, group_size, rows
):
    k = 1024
    x = _x(rows, k, dtype, seed=rows)
    layer = _linear(1024, k, bits, group_size, dtype, seed=bits)
    gate = _linear(2048, k, bits, group_size, dtype, seed=bits + 10)
    up = _linear(2048, k, bits, group_size, dtype, seed=bits + 20)
    for sums in _sums_of(x):
        args = (layer.weight, layer.scales, layer.biases)
        kw = dict(group_size=group_size, bits=bits, sums=sums)
        assert _same(live.vk_qmm_sg8(x, *args, **kw), oracle.vk_qmm_sg8(x, *args, **kw))
        got = live.vk_swiglu_sg8(x, gate, up, sums)
        assert _same(got, oracle.vk_swiglu_sg8(x, gate, up, sums))


# The geometries: eight K parts of one tile, of two tiles (K from 16384),
# four parts of four tiles (N from 65536).
@pytest.mark.parametrize("k,n", [(16384, 1024), (2048, 65536)])
def test_every_geometry_is_the_oracle_s(k, n):
    x = _x(8, k, mx.bfloat16, seed=k)
    layer = _linear(n, k, 4, 64, mx.bfloat16, seed=n)
    args = (layer.weight, layer.scales, layer.biases)
    for sums in _sums_of(x):
        kw = dict(group_size=64, bits=4, sums=sums)
        assert _same(live.vk_qmm_sg8(x, *args, **kw), oracle.vk_qmm_sg8(x, *args, **kw))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("rows", [4, 7])
def test_a_group_of_matrices_is_the_oracle_s(dtype, group_size, rows):
    """4- and 5-bit matrices of one input in one launch."""
    k = 1024
    x = _x(rows, k, dtype, seed=rows + group_size)
    group = [
        _linear(n, k, bits, group_size, dtype, seed=n + bits)
        for n, bits in ((4096, 4), (512, 5), (1024, 4))
    ]
    for sums in _sums_of(x):
        got = live.vk_group_sg8(x, group, sums)
        want = oracle.vk_group_sg8(x, group, sums)
        assert len(got) == len(want) == 3
        assert all(_same(a, b) for a, b in zip(got, want, strict=True))
