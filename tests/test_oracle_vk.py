# SPDX-License-Identifier: Apache-2.0
"""The frozen oracle of the legacy ``vk`` kernels (``tests/oracles/vk.py``):
a verbatim copy while the live kernels exist, and a reference that tells
their rows apart bit for bit."""

from __future__ import annotations

import inspect
from dataclasses import replace

import mlx.core as mx
import pytest

from omlx.formats import affine
from omlx.formats import kernels
from omlx.patches import qwen35_verify_qmm as live
from tests.oracles import vk as oracle
from tests.test_formats_tables import tabled_tile

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# What the oracle copies of the live module that the live module still has:
# its functions and its constants. The msg kernel (_fma_block,
# _build_msg_kernel) moved onto the tile face, and vk_qmm with it: they are
# the oracle's alone, and the rows below compare them.
_FUNCTIONS = (
    "_pack_block",
    "_build_ksplit_kernel",
    "_pad_rows",
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
    """Rows of activations, the second all zeros: a row whose products are
    signed zeros."""
    mx.random.seed(seed)
    x = (mx.random.normal((rows, k)) * 0.5).astype(dtype)
    if rows > 1:
        x = mx.concatenate([x[:1], mx.zeros_like(x[1:2]), x[2:]])
    mx.eval(x)
    return x


def _same(a, b):
    """Bit for bit: -0.0 and 0.0 are two answers."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    width = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return mx.array_equal(a.view(width), b.view(width)).item()


# The split-K kernel (two K parts from N 4096, four below) at every row
# count it is templated for, both widths, every group size and dtype.
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("n", [4096, 1024])
@pytest.mark.parametrize("k", [1024, 320])
def test_the_split_k_kernel_is_the_oracle_s(dtype, bits, group_size, n, k):
    """K 320: a lane's last wave of runs is partial."""
    if k % group_size:
        pytest.skip("MLX quantizes K in whole groups")
    w = _linear(n, k, bits, group_size, dtype, seed=bits + group_size)
    for rows in range(2, 7):
        x = _x(rows, k, dtype, seed=rows)
        kw = dict(bits=bits, group_size=group_size)
        assert _same(live.vk_qmm(x, *w, **kw), oracle.vk_qmm(x, *w, **kw))


# The msg kernel of the head (N from 100000, a multiple of 32): its 4- and
# 6-row templates, rows padded up to them.
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("k", [512, 320])
def test_the_head_kernel_is_the_oracle_s(dtype, bits, group_size, k):
    """K 320: a lane's last wave of runs is partial (MLX quantizes it in
    groups of 32 and 64)."""
    if k % group_size:
        pytest.skip("MLX quantizes K in whole groups")
    n = 100352
    w = _linear(n, k, bits, group_size, dtype, seed=bits)
    for rows in (3, 4, 5, 6):
        x = _x(rows, k, dtype, seed=rows + 10)
        kw = dict(bits=bits, group_size=group_size)
        got = live.vk_qmm(x, *w, **kw)
        assert got.shape == (rows, n)
        assert _same(got, oracle.vk_qmm(x, *w, **kw))


def test_a_head_whose_scales_no_tile_face_reads_is_left_to_the_stock_qmm():
    k, n = 512, 100352
    weight, scales, biases = _linear(n, k, 4, 64, mx.bfloat16, seed=1)
    x = _x(4, k, mx.bfloat16, seed=1)
    wide = (scales.astype(mx.float32), biases.astype(mx.float32))
    assert live.vk_qmm(x, weight, *wide, bits=4, group_size=64) is None


# The dispatch's boundaries: 100000 columns (a multiple of 32) take the head
# kernel, 99968 the split-K; 4096 columns two K parts, 4092 four.
@pytest.mark.parametrize("n", [100004, 100000, 99968, 4096, 4092])
def test_the_dispatch_boundaries_are_the_oracle_s(n):
    w = _linear(n, 256, 4, 64, mx.bfloat16, seed=n)
    for rows in (1, 2, 4, 6):
        x = _x(rows, 256, mx.bfloat16, seed=rows)
        kw = dict(bits=4, group_size=64)
        assert _same(live.vk_qmm(x, *w, **kw), oracle.vk_qmm(x, *w, **kw))


@pytest.mark.parametrize("rows,n", [(7, 4096), (0, 4096), (4, 4094)])
def test_a_shape_the_kernels_do_not_take_is_refused(rows, n):
    w = _linear(n, 256, 4, 64, mx.bfloat16, seed=1)
    x = mx.zeros((rows, 256), mx.bfloat16)
    with pytest.raises(ValueError, match="vk_qmm takes 1 to 6 rows and N in fours"):
        live.vk_qmm(x, *w, bits=4, group_size=64)


def _msg_source(monkeypatch, m, face):
    live._msg_kernel.cache_clear()
    seen = []

    def record(*args, **kwargs):
        seen.append(args[-1] if args else kwargs.get("source"))
        return None

    monkeypatch.setattr(kernels, "face_kernel", record)
    try:
        live._msg_kernel(m, face, live._MSG_NSG)
    finally:
        live._msg_kernel.cache_clear()
    assert seen
    return seen[0]


def test_msg_kernel_specializes_builtin_affine_faces_only(monkeypatch):
    four = affine.tile_face(4, 64, mx.bfloat16)
    assert "const F four" not in _msg_source(monkeypatch, 4, tabled_tile(four))
    eight = replace(
        affine.tile_face(8, 64, mx.bfloat16), key="affine_tile_b8g64_custom"
    )
    assert "const R rj(w," not in _msg_source(monkeypatch, 6, eight)


def test_msg_kernel_source_selection(monkeypatch):
    b8 = affine.tile_face(8, 64, mx.bfloat16)
    assert "const R rj(w," in _msg_source(monkeypatch, 6, b8)
    m4 = _msg_source(monkeypatch, 4, b8)
    assert "const R r3(w," in m4 and "const F four" not in m4
    b4 = affine.tile_face(4, 64, mx.bfloat16)
    for m in (4, 6):
        assert "const F four" in _msg_source(monkeypatch, m, b4)


def test_a_head_with_biases_of_another_dtype_is_left_to_the_stock_qmm():
    weight, scales, biases = _linear(100352, 512, 4, 64, mx.bfloat16, seed=1)
    x = _x(4, 512, mx.bfloat16, seed=1)
    for dtype in (mx.float32, mx.float16):
        assert (
            live.vk_qmm(x, weight, scales, biases.astype(dtype), bits=4, group_size=64)
            is None
        )
