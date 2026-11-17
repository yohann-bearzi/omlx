# SPDX-License-Identifier: Apache-2.0
"""The tensor-unit hyper-connection prefill kernels against their oracle.

``tests/oracles/hc_prefill_nax.py`` is a verbatim copy of the three kernels
(``norm_inject``, ``down_silu``, ``up_tail``) and their launch code. While the
live module still holds that code, the first tests prove the copy is
verbatim; the others run both on the same inputs and compare every output
bit for bit, intermediates included (the inject partial sums), so that a port
of a kernel onto quant-format faces keeps its bits.
"""

from __future__ import annotations

import inspect

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.custom_kernels.nax import is_nax_available
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from tests.oracles import hc_prefill_nax as oracle

needs_nax = pytest.mark.skipif(
    not (mx.metal.is_available() and is_nax_available()),
    reason="requires Metal tensor units (M5)",
)

# (hc_count, hidden, low rank): Flash-Next's, and a small model's.
SHAPES = {"flash": (4, 2560, 320), "small": (4, 512, 64)}
BITS = (4, 5, 6, 8)
DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16}


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _live():
    from mlx_vlm.models.qwen4_exp import hc_prefill_nax

    return hc_prefill_nax


# The oracle is the live code.

_METAL = (
    "_NAX_HEADER",
    "_TG_LOAD",
    "_NORM_TEMPLATE",
    "_WRITE_LOAD",
    "_PLAIN_LOAD",
    "_NORM_SOURCES",
    "_DOWN_SOURCE",
    "_UP_SOURCE",
)
_CONSTANTS = ("_BM", "_BN", "UP_COLS", "GROUP_SIZE")
_LAUNCHES = ("plain_qmm_nax", "norm_inject", "down_silu", "up_tail")


@pytest.mark.parametrize("name", _METAL + _CONSTANTS)
def test_the_oracle_s_metal_and_constants_are_the_live_ones(name):
    assert getattr(oracle, name) == getattr(_live(), name)


def test_the_oracle_s_inject_header_is_hc_projection_s():
    from mlx_vlm.models.qwen4_exp import hc_projection

    assert oracle._QDOT_HEADER == hc_projection._HEADER


@pytest.mark.parametrize("name", _LAUNCHES)
def test_the_oracle_s_launches_are_the_live_ones(name):
    assert inspect.getsource(getattr(oracle, name)) == inspect.getsource(
        getattr(_live(), name)
    )


def test_the_oracle_s_kernels_differ_from_the_live_ones_by_name_alone():
    renamed = inspect.getsource(oracle._kernel).replace(
        "    # The oracle's kernels carry names of their own.\n"
        '    name = "oracle_" + name\n',
        "",
    )
    assert renamed == inspect.getsource(_live()._kernel)


def test_plain_qmm_nax_answers_as_the_oracle():
    live = _live().plain_qmm_nax
    # The boundaries: 64 rows (one tile; N 16384 is 256 tiles of 64 there),
    # 256 tiles of 64 (N 1024 at 1024 rows), and Flash-Next's widths.
    for n_out in (64, 320, 1024, 2560, 10240, 16384):
        for rows in (*range(1, 130), 800, 801, 1023, 1024, 1025, 3264, 3265, 8192):
            assert live(rows, n_out) == oracle.plain_qmm_nax(rows, n_out), (
                rows,
                n_out,
            )


# The live kernels compute the oracle's bits.


def _projection(k: int, n: int, bits: int, dtype, seed: int) -> nn.QuantizedLinear:
    """A group-64 projection with checkpoint-like scales and biases."""
    mx.random.seed(seed)
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=bits)
    layer.scales = (mx.abs(mx.random.normal(layer.scales.shape)) * 0.01 + 0.002).astype(
        dtype
    )
    layer.biases = (mx.random.normal(layer.biases.shape) * 0.005).astype(dtype)
    mx.eval(layer.parameters())
    return layer


def _zero_rows(a: mx.array) -> mx.array:
    """``a`` with its first and last rows all zero (each a row of +0)."""
    rows = a.shape[0]
    keep = mx.ones((rows, 1), dtype=a.dtype)
    if rows > 2:
        keep[0] = 0
        keep[rows - 1] = 0
    return a * keep


def _same(a: mx.array, b: mx.array) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    return mx.array_equal(a.view(view), b.view(view)).item()


def _assert_same(observed, expected):
    mx.eval(observed, expected)
    assert len(observed) == len(expected)
    for got, want in zip(observed, expected, strict=True):
        assert _same(got, want)


def _norm_module(width: int, hidden: int, dtype, seed: int):
    mx.random.seed(seed)
    weight = (mx.random.normal((width,)) * 0.05).astype(dtype)
    hc_norm = type("Norm", (), {"weight": weight})()
    return type("Module", (), {"hc_norm": hc_norm})()


@needs_nax
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("rows", [1, 3, 65])
def test_norm_inject_computes_the_oracle_s_bits(shape, dtype, bits, write, rows):
    hc, hidden, _ = SHAPES[shape]
    dtype = DTYPES[dtype]
    width = hc * hidden
    module = _norm_module(width, hidden, dtype, seed=rows + bits)
    inject = _projection(width, hc, bits, dtype, seed=3 * rows + bits)
    flat = _zero_rows((mx.random.normal((rows, width)) * 2).astype(dtype))
    pending = None
    if write:
        branch = _zero_rows(mx.random.normal((rows, hidden)).astype(dtype))
        gate = (2 * mx.sigmoid(mx.random.normal((rows, hc)))).astype(dtype)
        pending = (branch, gate)
    eps = mx.array([1e-6], dtype=mx.float32)
    args = (module, flat, pending, rows, hc, hidden, dtype, eps, inject)
    live = _live().norm_inject(*args)
    expected = oracle.norm_inject(*args)
    if write:
        assert live[0] is not None
    else:
        assert live[0] is None and expected[0] is None
        live, expected = live[1:], expected[1:]
    _assert_same(live, expected)


@needs_nax
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("rows", [1, 64, 65, 130])
def test_down_silu_computes_the_oracle_s_bits(shape, dtype, bits, rows):
    hc, hidden, lowrank = SHAPES[shape]
    dtype = DTYPES[dtype]
    width = hc * hidden
    down = _projection(width, lowrank, bits, dtype, seed=5 * rows + bits)
    normed = _zero_rows(mx.random.normal((rows, width)).astype(dtype))
    _assert_same(
        [_live().down_silu(normed, down, rows, hc)],
        [oracle.down_silu(normed, down, rows, hc)],
    )


@needs_nax
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("rows", [1, 64, 65, 130])
def test_up_tail_computes_the_oracle_s_bits(shape, dtype, bits, rows):
    hc, hidden, lowrank = SHAPES[shape]
    dtype = DTYPES[dtype]
    width = hc * hidden
    up = _projection(lowrank, width, bits, dtype, seed=7 * rows + bits)
    act = _zero_rows(mx.random.normal((rows, lowrank)).astype(dtype))
    normed = _zero_rows(mx.random.normal((rows, width)).astype(dtype))
    inj_part = _zero_rows(mx.random.normal((rows, hc * 8)).astype(mx.float32))
    inj_part = inj_part.reshape(rows, hc, 8)
    args = (act, up, normed, inj_part, rows, hc, hidden)
    _assert_same(_live().up_tail(*args), oracle.up_tail(*args))
