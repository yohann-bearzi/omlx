# SPDX-License-Identifier: Apache-2.0
"""The frozen oracle of the legacy ``vk`` kernels (``tests/oracles/vk.py``):
a verbatim copy while the live kernels exist, and a reference that tells
their rows apart bit for bit."""

from __future__ import annotations

import inspect

import mlx.core as mx
import pytest

from omlx.patches import qwen35_verify_qmm as live
from tests.oracles import vk as oracle

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# What the oracle copies of the live module: its functions and its constants.
_FUNCTIONS = (
    "_fma_block",
    "_build_msg_kernel",
    "_pack_block",
    "_build_ksplit_kernel",
    "_pad_rows",
    "vk_qmm",
)
_CONSTANTS = ("_MSG_NSG",)


def test_the_oracle_is_the_live_code_verbatim():
    for name in _FUNCTIONS:
        assert inspect.getsource(getattr(oracle, name)) == inspect.getsource(
            getattr(live, name)
        ), name
    for name in _CONSTANTS:
        assert getattr(oracle, name) == getattr(live, name), name


def _linear(n, k, bits, group_size, dtype, seed):
    mx.random.seed(seed)
    w = mx.random.normal((n, k)) * 0.05
    weight, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    scales, biases = scales.astype(dtype), biases.astype(dtype)
    mx.eval(weight, scales, biases)
    return weight, scales, biases


def _x(rows, k, dtype, seed):
    mx.random.seed(seed)
    x = (mx.random.normal((rows, k)) * 0.5).astype(dtype)
    mx.eval(x)
    return x


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


# The split-K kernel (two K parts from N 4096, four below) at every row
# count it is templated for, both widths, every group size and dtype.
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("n", [4096, 1024])
def test_the_split_k_kernel_is_the_oracle_s(dtype, bits, group_size, n):
    k = 1024
    w = _linear(n, k, bits, group_size, dtype, seed=bits + group_size)
    for rows in range(2, 7):
        x = _x(rows, k, dtype, seed=rows)
        kw = dict(bits=bits, group_size=group_size)
        assert _same(live.vk_qmm(x, *w, **kw), oracle.vk_qmm(x, *w, **kw))


# The msg kernel of the head (N from 100000, a multiple of 32): its 4- and
# 6-row templates, rows padded up to them.
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bits,group_size", [(4, 64), (8, 64), (4, 32), (8, 128)])
def test_the_head_kernel_is_the_oracle_s(dtype, bits, group_size):
    k, n = 512, 100352
    w = _linear(n, k, bits, group_size, dtype, seed=bits)
    for rows in (3, 4, 5, 6):
        x = _x(rows, k, dtype, seed=rows + 10)
        kw = dict(bits=bits, group_size=group_size)
        got = live.vk_qmm(x, *w, **kw)
        assert got.shape == (rows, n)
        assert _same(got, oracle.vk_qmm(x, *w, **kw))
