# SPDX-License-Identifier: Apache-2.0
"""The built-in affine format: its dot face follows MLX's one-row dispatch
and computes every row with MLX's bits.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import affine, testing

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _linear(k, n, bits, group_size, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    linear = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    weight = (mx.random.normal((n, k)) * 0.05).astype(dtype)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=group_size, bits=bits
    )
    return linear


def _stacked(experts, k, n, bits, group_size):
    """The parameters of a ``QuantizedSwitchLinear``: one matrix per expert."""
    layers = [_linear(k, n, bits, group_size, seed=e) for e in range(experts)]
    stacked = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    for name in ("weight", "scales", "biases"):
        stacked[name] = mx.stack([layer[name] for layer in layers])
    return stacked


def _bit_equal(a, b):
    return mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


def test_affine_ships_as_a_builtin_plugin():
    assert "omlx.formats.affine:FORMAT" in formats.BUILTIN
    assert formats.get("affine") is affine.FORMAT
    assert affine.FORMAT.tensors == ("scales", "biases")


@pytest.mark.parametrize(
    "k, bits, fast",
    [
        (512, 4, True),
        (512, 5, True),
        (256, 6, True),
        (256, 8, True),
        (256, 4, False),  # MLX's qmv_fast block is 512 wide at 4 and 5 bits
        (256, 5, False),
        (384, 6, False),
        (384, 8, False),
    ],
)
def test_the_face_follows_mlx_one_row_dispatch(k, bits, fast):
    face = formats.dot(_linear(k, 16, bits, 64), k, 16)
    assert face == affine.face(bits, 64, fast, mx.bfloat16)
    assert (face.tail, face.dtype, face.tensors) == (
        not fast,
        mx.bfloat16,
        ("scales", "biases"),
    )


def test_stacked_weights_follow_gather_qmm_dispatch():
    # Gathered QMV is fast only for whole output tiles of 8 rows.
    assert not formats.dot(_stacked(2, 512, 16, 4, 64), 512, 16).tail
    assert formats.dot(_stacked(2, 512, 12, 4, 64), 512, 12).tail
    assert not formats.dot(_linear(512, 12, 4, 64), 512, 12).tail


def test_each_layout_is_one_face():
    a = formats.dot(_linear(512, 16, 4, 64, seed=1), 512, 16)
    b = formats.dot(_linear(512, 32, 4, 64, seed=2), 512, 32)
    assert a is b
    assert a != formats.dot(_linear(512, 16, 4, 64, mx.float16), 512, 16)
    keys = {
        affine.face(bits, group_size, fast, mx.bfloat16).key
        for bits in (4, 5, 6, 8)
        for group_size in (32, 64, 128)
        for fast in (True, False)
    }
    assert len(keys) == 24


@pytest.mark.parametrize(
    "bits, dtype, k, n",
    [
        (3, mx.bfloat16, 512, 16),  # bits the transcription lacks
        (4, mx.float32, 512, 16),
        (4, mx.bfloat16, 256, 16),  # not the matrix asked for
        (4, mx.bfloat16, 512, 8),
    ],
)
def test_no_face_outside_what_the_transcription_covers(bits, dtype, k, n):
    assert formats.dot(_linear(512, 16, bits, 64, dtype), k, n) is None


def test_no_face_without_biases_in_the_scales_dtype():
    linear = _linear(512, 16, 4, 64)
    linear.biases = linear.biases.astype(mx.float16)
    assert formats.dot(linear, 512, 16) is None
    linear.pop("biases")
    assert formats.dot(linear, 512, 16) is None


@metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("k", [512, 384, 1280])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_rows_have_mlx_bits(bits, group_size, k, dtype):
    n = 16
    linear = _linear(k, n, bits, group_size, dtype, seed=bits + group_size + k)
    x = mx.random.normal((k,)).astype(dtype)
    face = formats.dot(linear, k, n)
    got = testing.dot_rows(
        face, [linear.weight, linear.scales, linear.biases], x, n, packs=(2, 2)
    )
    want = mx.quantized_matmul(
        x[None],
        linear.weight,
        linear.scales,
        linear.biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )[0]
    assert _bit_equal(got, want)


@metal
@pytest.mark.parametrize("packs", [(1, 0), (1, 1), (4, 0), (3, 1), (4, 4)])
@pytest.mark.parametrize("k", [512, 384])
def test_a_row_does_not_depend_on_the_rows_it_runs_with(k, packs):
    linear = _linear(k, 16, 5, 64, seed=k)
    x = mx.random.normal((k,)).astype(mx.bfloat16)
    operands = [linear.weight, linear.scales, linear.biases]
    face = formats.dot(linear, k, 16)
    assert _bit_equal(
        testing.dot_rows(face, operands, x, 16, packs),
        testing.dot_rows(face, operands, x, 16, packs=(2, 2)),
    )


@metal
def test_the_face_is_exact_for_its_own_dtype_only():
    linear = _linear(512, 16, 4, 64)
    operands = [linear.weight, linear.scales, linear.biases]
    with pytest.raises(ValueError, match="is exact for mlx.core.bfloat16"):
        testing.dot_rows(
            formats.dot(linear, 512, 16), operands, mx.zeros((512,), mx.float16), 16
        )
