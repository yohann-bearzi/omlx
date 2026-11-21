# SPDX-License-Identifier: Apache-2.0
"""The frozen oracle of the legacy ``vk_mma`` kernel
(``tests/oracles/vk_mma.py``): a verbatim copy while the live kernel exists,
and a reference that tells its rows apart bit for bit."""

from __future__ import annotations

import inspect

import mlx.core as mx
import pytest

from omlx.patches import qwen35_verify_qmm as live
from tests.oracles import vk_mma as oracle
from tests.test_oracle_vk import _linear, _same, _x

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# What the oracle copies of the live module that the live module still has.
# The kernel (_build_mma_kernel) moved onto the tile face, and vk_qmm_mma
# with it: they are the oracle's alone, and the rows below compare them.
_FUNCTIONS = ("_pad_rows",)
_CONSTANTS = ("_MMA_BN", "_MMA_BK", "_MMA_NSG")


def test_the_oracle_is_the_live_code_verbatim():
    for name in _FUNCTIONS:
        assert inspect.getsource(getattr(oracle, name)) == inspect.getsource(
            getattr(live, name)
        ), name
    for name in _CONSTANTS:
        assert getattr(oracle, name) == getattr(live, name), name


# Every width and group size the kernel takes, both dtypes; rows from 1 to
# 24 (one to three row tiles, padded and whole); K whose simdgroup chunks
# (K / 8 each) end inside a group (256, 768) and on one (1024).
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("k", [1024, 768, 256])
def test_the_tile_kernel_is_the_oracle_s(dtype, bits, group_size, k):
    if k % group_size:
        pytest.skip("MLX quantizes K in whole groups")
    n = 4096
    w = _linear(n, k, bits, group_size, dtype, seed=bits * group_size)
    for rows in (1, 7, 8, 13, 16, 17, 23, 24):
        x = _x(rows, k, dtype, seed=rows)
        kw = dict(bits=bits, group_size=group_size)
        got = live.vk_qmm_mma(x, *w, **kw)
        assert got.shape == (rows, n)
        assert _same(got, oracle.vk_qmm_mma(x, *w, **kw))


# A K of several chunks per simdgroup (8192) and a wide N.
def test_a_long_k_and_a_wide_n_are_the_oracle_s():
    for k, n in ((8192, 4096), (2048, 16384)):
        w = _linear(n, k, 4, 64, mx.bfloat16, seed=k)
        x = _x(11, k, mx.bfloat16, seed=k)
        kw = dict(bits=4, group_size=64)
        assert _same(live.vk_qmm_mma(x, *w, **kw), oracle.vk_qmm_mma(x, *w, **kw))


@pytest.mark.parametrize(
    "rows,k,n", [(25, 256, 4096), (0, 256, 4096), (8, 128, 4096), (8, 256, 4104)]
)
def test_a_shape_the_kernel_does_not_take_is_refused(rows, k, n):
    w = _linear(n, k, 4, 64, mx.bfloat16, seed=1)
    x = mx.zeros((rows, k), mx.bfloat16)
    with pytest.raises(ValueError, match="vk_qmm_mma takes 1 to 24 rows"):
        live.vk_qmm_mma(x, *w, group_size=64, bits=4)


def test_scales_no_tile_face_reads_are_left_to_the_stock_qmm():
    weight, scales, biases = _linear(4096, 256, 4, 64, mx.bfloat16, seed=1)
    x = _x(8, 256, mx.bfloat16, seed=1)
    wide = (scales.astype(mx.float32), biases.astype(mx.float32))
    assert live.vk_qmm_mma(x, weight, *wide, group_size=64, bits=4) is None
