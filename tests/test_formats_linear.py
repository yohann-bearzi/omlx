# SPDX-License-Identifier: Apache-2.0
"""The layers that hold a format's tensors (``omlx.formats.linear``).

The affine format through them has the bits of MLX's quantized layers, row by
row. A format oMLX does not know runs through them. What the rows kernel does
not take is the format's ``decode``, then MLX's matmul.
"""

from __future__ import annotations

import logging
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.switch_layers import SwitchGLU, SwitchLinear

from omlx import formats
from omlx.formats import affine, kernels, linear
from omlx.formats.linear import MAX_ROWS, FormatLinear, FormatSwitchLinear
from tests.test_formats_kernels import _PAGE, _TOY, _matrix, _toy
from tests.test_qwen35_moe_routed_decode import _does_not_compile, _logged, _same_bits

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

EXPERTS = 6
_NAMES = ("weight", "scales", "biases")


def _tensors(quantized):
    return {name: quantized[name] for name in _NAMES}


def _pair(k, n, bits=4, dtype=mx.bfloat16, bias=False, seed=0):
    """MLX's quantized linear, and a ``FormatLinear`` of its tensors."""
    mx.random.seed(seed)
    dense = nn.Linear(k, n, bias=bias)
    dense.set_dtype(dtype)
    quantized = dense.to_quantized(group_size=32, bits=bits)
    layer = FormatLinear(
        "affine", bits, 32, k, _tensors(quantized), bias=quantized.get("bias")
    )
    return quantized, layer


def _switch_pair(k, n, bits=4, dtype=mx.bfloat16, bias=False, seed=0):
    """mlx-vlm's quantized switch layer, and a ``FormatSwitchLinear`` of its
    tensors."""
    mx.random.seed(seed)
    dense = SwitchLinear(k, n, EXPERTS, bias=bias)
    dense.set_dtype(dtype)
    quantized = dense.to_quantized(group_size=32, bits=bits)
    if bias:
        quantized["bias"] = mx.random.normal((EXPERTS, n)).astype(dtype)
    layer = FormatSwitchLinear(
        "affine", bits, 32, k, _tensors(quantized), bias=quantized.get("bias")
    )
    return quantized, layer


def _x(shape, dtype=mx.bfloat16, seed=1):
    mx.random.seed(seed)
    x = mx.random.normal(shape).astype(dtype)
    mx.eval(x)
    return x


def _experts(shape, seed=2):
    mx.random.seed(seed)
    indices = mx.random.randint(0, EXPERTS, shape).astype(mx.uint32)
    mx.eval(indices)
    return indices


def _composed(quantized, x):
    """The decode, then MLX's matmul."""
    weight = mx.dequantize(
        quantized["weight"],
        quantized["scales"],
        quantized["biases"],
        group_size=quantized.group_size,
        bits=quantized.bits,
    )
    y = x @ weight.astype(x.dtype).T
    return y + quantized["bias"] if "bias" in quantized else y


def _equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


@pytest.mark.parametrize(
    "k,n,bits,dtype,bias",
    [
        (1024, 64, 4, mx.bfloat16, False),  # MLX's qmv_fast traversal
        (320, 72, 4, mx.bfloat16, True),  # qmv: one whole block, then a last one
        (512, 40, 5, mx.float16, True),
        (320, 64, 6, mx.bfloat16, False),
        (1024, 16, 8, mx.float16, False),
    ],
)
def test_a_row_has_the_bits_of_mlx_s_quantized_linear(caplog, k, n, bits, dtype, bias):
    caplog.set_level(logging.INFO, logger=linear.__name__)
    quantized, layer = _pair(k, n, bits, dtype, bias)
    face = formats.dot(layer, k, n)
    assert face is formats.dot(quantized, k, n) and face is not None
    for shape in ((1, k), (1, 1, k)):
        x = _x(shape, dtype)
        got = layer(x)
        assert got.dtype == dtype and _same_bits(got, quantized(x))
    # One vector.
    x = _x((k,), dtype)
    assert _same_bits(layer(x), quantized(x[None])[0])
    # A short block: each row as it is alone.
    for shape in ((2, k), (3, k), (2, 3, k), (MAX_ROWS, k), (1, MAX_ROWS, k)):
        x = _x(shape, dtype, seed=len(shape) + shape[-2])
        got = layer(x)
        assert got.shape == (*shape[:-1], n)
        rows, alone = got.reshape(-1, n), x.reshape(-1, 1, k)
        for row in range(rows.shape[0]):
            assert _same_bits(rows[row : row + 1], quantized(alone[row]))
    ((key, proof),) = kernels._PROOFS.items()
    assert key == ("format_linear", face, (mx.uint32, dtype, dtype), k, n)
    # The rows are not compiled into a launch.
    assert proof.ran == {dtype} and not proof.failed
    assert [r.getMessage() for r in _logged(caplog, linear)] == [
        f"quant format face {face.key}: rows kernel engaged"
    ]


def test_the_page_names_the_rows_a_linear_launches():
    assert f"| `FormatLinear`, 1 to {MAX_ROWS} rows of `x` (`linear.MAX_ROWS`:" in _PAGE


def test_more_rows_are_the_decode_and_mlx_s_matmul():
    k, n = 320, 72
    quantized, layer = _pair(k, n, bias=True)
    for shape in ((MAX_ROWS + 1, k), (3, MAX_ROWS, k)):
        x = _x(shape)
        assert _same_bits(layer(x), _composed(quantized, x))
    # No row at all.
    assert layer(_x((2, k))[:0]).shape == (0, n)
    assert not kernels._PROOFS


def test_what_a_face_does_not_cover_is_the_decode_and_mlx_s_matmul():
    # N that is not whole tiles of 8 outputs.
    quantized, layer = _pair(320, 60)
    x = _x((1, 320))
    assert formats.dot(layer, 320, 60) is not None
    assert _same_bits(layer(x), _composed(quantized, x))
    # A layout the format has no face for (MLX's qmv_quad).
    quantized, layer = _pair(64, 8)
    x = _x((1, 64))
    assert formats.dot(layer, 64, 8) is None
    assert _same_bits(layer(x), _composed(quantized, x))
    assert not kernels._PROOFS
    # Activations of a dtype the face is not exact for.
    quantized, layer = _pair(320, 72, bias=True)
    x = _x((2, 320), mx.float32)
    assert not formats.dot(layer, 320, 72).exact_for(mx.float32)
    assert _equal(layer(x), _composed(quantized, x))
    (proof,) = kernels._PROOFS.values()
    assert not proof.ran and not proof.failed


SHAPES = [
    # mlx-vlm's SwitchGLU: each token with its top-k experts.
    ((1, 4, 1, 1), (1, 4, 3), 3),
    ((2, 4, 1, 1), (2, 4, 3), 3),
    # Its sorted prefill: each pair with its own row.
    ((16, 1), (16,), 1),
    ((2, 4, 3, 1), (2, 4, 3), 1),
    # One row with every expert asked.
    ((1, 1), (16,), 16),
    ((2, 1, 1, 1), (2, 4, 3), 12),
    # Broadcast in memory: x along an axis of the indices ahead of its own,
    ((4, 1, 1), (2, 4, 3), 1),
    ((1, 4, 1, 1), (2, 4, 3), 1),
    ((3, 1), (3, 3), 1),
    # and the indices along an axis of x.
    ((2, 4, 3, 1), (2, 4, 1), 1),
    ((2, 4, 3, 1), (3,), 1),
]


@pytest.mark.parametrize("lead_m,routed,per_row", SHAPES)
@pytest.mark.parametrize(
    "k,n,bits,dtype,bias",
    [
        (1024, 64, 4, mx.bfloat16, False),
        (320, 72, 4, mx.bfloat16, True),
        (320, 40, 5, mx.float16, True),
    ],
)
def test_a_pair_has_the_bits_of_mlx_vlm_s_quantized_switch(
    lead_m, routed, per_row, k, n, bits, dtype, bias
):
    quantized, layer = _switch_pair(k, n, bits, dtype, bias)
    x, indices = _x((*lead_m, k), dtype), _experts(routed)
    got = layer(x, indices)
    assert got.dtype == dtype and _same_bits(got, quantized(x, indices))
    # The hint and the dtype of the indices change nothing.
    assert _same_bits(layer(x, indices.astype(mx.int32), sorted_indices=True), got)
    ((key, proof),) = kernels._PROOFS.items()
    face = formats.dot(layer, k, n)
    assert key == ("format_switch_linear", face, (mx.uint32, dtype, dtype), k, n)
    assert proof.ran == {(dtype, per_row, False)} and not proof.failed


def test_each_row_of_a_pair_has_the_bits_it_has_alone():
    """mlx-vlm never asks for M rows per pair; MLX then runs another kernel,
    with other bits."""
    k, n = 320, 72
    quantized, layer = _switch_pair(k, n, bias=True)
    x, indices = _x((5, 3, k)), _experts((5,))
    got = layer(x, indices)
    assert got.shape == (5, 3, n)
    for row in range(3):
        alone = quantized(x[:, row : row + 1], indices)
        assert _same_bits(got[:, row : row + 1], alone)
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {(mx.bfloat16, 1, False)}


def test_fewer_than_8_pairs_are_a_launch_of_their_own():
    """MLX binds fewer than 8 indices as constants: other Metal."""
    quantized, layer = _switch_pair(320, 72)
    few, more = (mx.bfloat16, 1, True), (mx.bfloat16, 1, False)
    for pairs, ran in ((1, {few}), (7, {few}), (8, {few, more})):
        x, indices = _x((pairs, 1, 320), seed=pairs), _experts((pairs,), seed=pairs)
        assert _same_bits(layer(x, indices), quantized(x, indices))
        (proof,) = kernels._PROOFS.values()
        assert proof.ran == ran


def test_the_indices_of_a_launch_are_uint32(monkeypatch):
    """A launch is proven as the Metal it is, which has the indices' type."""
    quantized, layer = _switch_pair(320, 72)
    x, indices = _x((16, 1, 320)), _experts((16,))
    rows, seen = linear._rows, []

    def recorded(plan, x, n, rhs, pairs_per_row):
        seen.append(rhs.dtype)
        return rows(plan, x, n, rhs, pairs_per_row)

    monkeypatch.setattr(linear, "_rows", recorded)
    want = quantized(x, indices)
    for dtype in (mx.uint32, mx.int32, mx.int64, mx.uint8):
        assert _same_bits(layer(x, indices.astype(dtype)), want)
    assert seen == [mx.uint32] * 4


def test_mlx_vlm_s_switch_glu_runs_over_the_layers():
    mx.random.seed(3)
    theirs, ours = SwitchGLU(512, 64, EXPERTS), SwitchGLU(512, 64, EXPERTS)
    theirs.set_dtype(mx.bfloat16)
    for name in ("gate_proj", "up_proj", "down_proj"):
        quantized = getattr(theirs, name).to_quantized(32, 4)
        setattr(theirs, name, quantized)
        k = quantized.input_dims
        setattr(ours, name, FormatSwitchLinear("affine", 4, 32, k, _tensors(quantized)))
    # 18 pairs, which mlx-vlm runs as they come.
    x, indices = _x((1, 3, 512)), _experts((1, 3, 6))
    assert _same_bits(ours(x, indices), theirs(x, indices))
    # 96 pairs, which it sorts by expert: each still has the bits it has
    # in a decode step. (MLX runs its sorted pairs in another kernel.)
    x, indices = _x((1, 16, 512)), _experts((1, 16, 6))
    steps = [theirs(x[:, t : t + 1], indices[:, t : t + 1]) for t in range(16)]
    assert _same_bits(ours(x, indices), mx.concatenate(steps, axis=1))
    # gate and up share a proof: the same face, tensor dtypes and shape.
    assert len(kernels._PROOFS) == 2
    for (kernel, *_), proof in kernels._PROOFS.items():
        assert kernel == "format_switch_linear" and proof.ran and not proof.failed


def test_a_switch_s_pairs_outside_the_kernel_are_the_decode_and_mlx_s_matmul(
    monkeypatch,
):
    """Fewer pairs than the stack has matrices decode theirs alone. From as
    many, the stack is decoded once: a prefill's pairs would not fit in
    memory a matrix each."""
    quantized, layer = _switch_pair(320, 72, bias=True)
    decode, whole = affine.AffineFormat.decode, []

    def recorded(self, tensors, bits, group_size):
        whole.append(tensors["weight"] is layer["weight"])
        return decode(self, tensors, bits, group_size)

    gather, hints = mx.gather_mm, []

    def gathered(*operands, **named):
        hints.append(named["sorted_indices"])
        return gather(*operands, **named)

    monkeypatch.setattr(affine.AffineFormat, "decode", recorded)
    monkeypatch.setattr(mx, "gather_mm", gathered)
    stack = mx.dequantize(*_tensors(quantized).values(), group_size=32, bits=4)
    stack = stack.astype(mx.float32)

    def want(x, indices, **hint):
        if hint:
            y = gather(x, stack.swapaxes(-1, -2), rhs_indices=indices, **hint)
        else:
            y = x @ stack[indices].swapaxes(-1, -2)
        return y + mx.expand_dims(quantized["bias"][indices], -2)

    # One pair fewer than matrices: (1, 5) of them, with x's (2, 1).
    x, indices = _x((2, 1, 1, 320), mx.float32), _experts((1, EXPERTS - 1))
    got = layer(x, indices)
    assert got.shape == (2, 5, 1, 72) and _equal(got, want(x, indices))
    assert whole == [False] and not hints
    # As many pairs as matrices, then a prefill's, in any integer type.
    for routed, dtype in (((EXPERTS,), mx.uint32), ((2, 4, 3), mx.int64)):
        x = _x((*routed[:-1], 1, 1, 320), mx.float32)
        indices = _experts(routed).astype(dtype)
        got = layer(x, indices)
        assert got.shape == (*routed, 1, 72)
        assert _equal(got, want(x, indices, sorted_indices=False))
    assert whole == [False, True, True] and hints == [False, False]
    # mlx-vlm's hint goes to MLX's gather, which the pairs then are sorted for.
    x, indices = _x((24, 1, 320), mx.float32), mx.sort(_experts((24,)))
    got = layer(x, indices, sorted_indices=True)
    assert _equal(got, want(x, indices, sorted_indices=True))
    assert hints == [False, False, True]
    (proof,) = kernels._PROOFS.values()
    assert not proof.ran and not proof.failed


class _ToyFormat(formats.QuantFormat):
    mode, tensors, contract = "toy", ("scales",), formats.CONTRACT

    def __init__(self, face=None):
        self.face = face

    def dot(self, layer, k, n):
        return self.face

    def decode(self, tensors, bits, group_size):
        codes = tensors["weight"].astype(mx.float32) - 128.0
        return codes * mx.expand_dims(tensors["scales"], -1)


@pytest.fixture
def toy(monkeypatch):
    """Registers the toy format of the kernel tests, with a face or none."""
    formats.load()

    def register(face=None):
        monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(face))

    return register


def _toy_linear(n, k, seed=0, bias=None):
    (codes, scales), x, want = _matrix(n, k, seed)
    tensors = {"weight": codes, "scales": scales}
    return FormatLinear("toy", 8, 1, k, tensors, bias=bias), x, want


def _toy_switch(n, k, seed=0, bias=None):
    (codes, scales), _, _ = _matrix(EXPERTS * n, k, seed)
    tensors = {
        "weight": codes.reshape(EXPERTS, n, k),
        "scales": scales.reshape(EXPERTS, n),
    }
    return FormatSwitchLinear("toy", 8, 1, k, tensors, bias=bias)


def _ints(shape, seed=0):
    """Small integer activations: with the toy's matrices, every partial sum
    is exact in float32, whatever the order."""
    mx.random.seed(seed)
    x = mx.random.randint(-8, 9, shape).astype(mx.float32)
    mx.eval(x)
    return x


def _want(layer, x, indices=None):
    """NumPy's products of ``x``'s rows with the toy layer's matrices: with
    the one, or with those at ``indices`` of the stack."""
    weight = np.array(layer["weight"]).astype(np.float32) - 128.0
    weight = weight * np.array(layer["scales"])[..., None]
    if indices is None:
        want = np.array(x) @ weight.T
        return want + np.array(layer["bias"]) if "bias" in layer else want
    indices = np.array(indices)
    want = np.matmul(np.array(x), np.swapaxes(weight[indices], -1, -2))
    if "bias" in layer:
        want = want + np.array(layer["bias"])[indices][..., None, :]
    return want


def test_a_format_oMLX_does_not_know_runs_in_a_linear(toy, monkeypatch):  # noqa: N802
    toy(_toy("linear"))
    # K = 320: one whole block of the toy face, then a last block of 8 lanes.
    layer, _, _ = _toy_linear(40, 320, bias=mx.arange(40).astype(mx.float32))
    with monkeypatch.context() as patch:
        patch.setattr(_ToyFormat, "decode", lambda *matrix: pytest.fail("decoded"))
        for shape in ((320,), (1, 320), (5, 320), (2, 3, 320), (MAX_ROWS, 320)):
            x = _ints(shape, seed=len(shape) + shape[0])
            got = layer(x)
            assert got.dtype == mx.float32
            assert np.array_equal(np.array(got), _want(layer, x))
    ((key, proof),) = kernels._PROOFS.items()
    assert key == ("format_linear", _toy("linear"), (mx.uint8, mx.float32), 320, 40)
    assert proof.ran == {mx.float32} and not proof.failed
    # More rows: its decode.
    x = _ints((MAX_ROWS + 1, 320))
    assert np.array_equal(np.array(layer(x)), _want(layer, x))


@pytest.mark.parametrize("lead_m,routed,per_row", SHAPES)
def test_a_format_oMLX_does_not_know_runs_in_a_switch(  # noqa: N802
    toy, monkeypatch, lead_m, routed, per_row
):
    toy(_toy("linear"))
    mx.random.seed(7)
    bias = mx.random.randint(-64, 64, (EXPERTS, 40)).astype(mx.float32)
    layer = _toy_switch(40, 320, bias=bias)
    monkeypatch.setattr(_ToyFormat, "decode", lambda *matrix: pytest.fail("decoded"))
    x, indices = _ints((*lead_m, 320)), _experts(routed)
    got = layer(x, indices)
    want = _want(layer, x, indices)
    assert got.dtype == mx.float32 and got.shape == want.shape
    assert np.array_equal(np.array(got), want)
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {(mx.float32, per_row, False)}


def test_rows_of_a_pair_and_no_pair_in_a_switch(toy, monkeypatch):
    toy(_toy("linear"))
    layer = _toy_switch(40, 320)
    # M rows of x for each pair.
    x, indices = _ints((2, 5, 3, 320)), _experts((5,))
    got = layer(x, indices)
    assert got.shape == (2, 5, 3, 40)
    assert np.array_equal(np.array(got), _want(layer, x, indices))
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {(mx.float32, 1, False)}
    # No pair, or no row of x: nothing to launch.
    monkeypatch.setattr(linear, "_rows", lambda *launch: pytest.fail("launched"))
    assert layer(x[:, :0], indices[:0]).shape == (2, 0, 3, 40)
    assert layer(x[:1, :1], indices[:0]).shape == (1, 0, 3, 40)
    assert layer(x[:, :, :0], indices).shape == (2, 5, 0, 40)
    # Nor from as many pairs as the stack has matrices.
    assert layer(_ints((8, 0, 320)), _experts((8,))).shape == (8, 0, 40)
    assert proof.ran == {(mx.float32, 1, False)} and not proof.failed


def test_a_format_without_a_face_runs_through_its_decode(toy):
    toy()
    layer, x, want = _toy_linear(40, 320)
    assert _equal(layer(x), want)
    switch = _toy_switch(40, 320)
    rows, indices = _ints((2, 4, 1, 1, 320)), _experts((2, 4, 3))
    assert np.array_equal(np.array(switch(rows, indices)), _want(switch, rows, indices))
    few = indices[:1, :1]  # fewer pairs than the stack has matrices
    assert np.array_equal(np.array(switch(rows, few)), _want(switch, rows, few))
    # In x's dtype, whatever the decode's: a layer casts the decoded weights.
    assert layer(x.astype(mx.float16)).dtype == mx.float16
    assert switch(rows.astype(mx.float16), indices).dtype == mx.float16
    assert switch(rows.astype(mx.float16), few).dtype == mx.float16
    assert not kernels._PROOFS


def test_a_matrix_no_launch_takes_runs_through_its_decode(toy):
    toy(_toy("linear"))
    # Seven activations: MLX binds fewer than 8 as constants, which no face
    # reads. Seven are not a lane of the toy face's 8 either.
    layer, x, want = _toy_linear(8, 7)
    assert _equal(layer(x), want)
    # N that is not whole tiles of 8 outputs.
    layer, x, want = _toy_linear(12, 320)
    assert _equal(layer(x), want)
    switch = _toy_switch(12, 320)
    rows, indices = _ints((16, 1, 320)), _experts((16,))
    assert np.array_equal(np.array(switch(rows, indices)), _want(switch, rows, indices))
    assert not kernels._PROOFS


def test_a_matrix_of_8_activations_is_the_narrowest_launch(toy):
    """MLX binds a tensor of 8 elements as a buffer, which a face reads."""
    toy(_toy("linear"))
    layer, x, want = _toy_linear(40, 8)
    assert _equal(layer(x), want)
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {mx.float32} and not proof.failed


def test_a_launch_its_tiles_do_not_cover_is_refused_in_words(toy):
    """A layer plans no such launch; the kernel does not rely on it."""
    toy(_toy("linear"))
    layer, x, _ = _toy_linear(40, 320)
    plan = linear._plan(layer)
    with pytest.raises(RuntimeError, match="format linear: N must be whole tiles"):
        mx.eval(linear._rows(plan, x.reshape(1, 320), 44))


def test_a_stack_and_a_matrix_of_one_face_are_kernels_of_their_own(toy):
    face = _toy("linear")
    toy(face)
    assert linear._kernel(face, False) is linear._kernel(face, False)
    assert linear._kernel(face, False) is not linear._kernel(face, True)
    layer, x, want = _toy_linear(40, 320)
    switch = _toy_switch(40, 320)
    assert _equal(layer(x), want)
    switch(_ints((16, 1, 320)), _experts((16,)))
    dtypes = (mx.uint8, mx.float32)
    assert set(kernels._PROOFS) == {
        ("format_linear", face, dtypes, 320, 40),
        ("format_switch_linear", face, dtypes, 320, 40),
    }


def test_the_plan_follows_the_layer_s_tensors(toy, monkeypatch):
    """A layer asks for its face once for the tensors it holds."""
    toy(_toy("linear"))
    dot, asked = formats.dot, []

    def recorded(layer, k, n):
        asked.append(id(layer))
        return dot(layer, k, n)

    monkeypatch.setattr(formats, "dot", recorded)
    layer, x, want = _toy_linear(40, 320)
    assert _equal(layer(x), want) and _equal(layer(x), want)
    assert asked == [id(layer)]
    layer["scales"] = layer["scales"] * 2
    assert _equal(layer(x), 2 * want) and _equal(layer(x), 2 * want)
    assert asked == [id(layer)] * 2
    switch = _toy_switch(40, 320)
    rows, indices = _ints((16, 1, 320)), _experts((16,))
    before = np.array(switch(rows, indices))
    assert np.array_equal(np.array(switch(rows, indices)), before)
    assert asked == [id(layer)] * 2 + [id(switch)]
    switch["scales"] = switch["scales"] * 2
    assert np.array_equal(np.array(switch(rows, indices)), 2 * before)
    assert asked == [id(layer)] * 2 + [id(switch)] * 2


def test_a_face_that_does_not_compile_leaves_the_layers_to_the_decode(
    toy, monkeypatch, caplog
):
    """MLX compiles a kernel at its first evaluation. A face that fails
    there fails inside the call that can fall back."""
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    toy(_toy("broken", _TOY + 'static_assert(false, "does not compile");\n'))
    layer, x, want = _toy_linear(40, 320)
    assert _equal(layer(x), want)
    (warning,) = _logged(caplog, linear, logging.WARNING)
    message = "quant format face toy_broken: rows kernel failed; decode fallback"
    assert warning.getMessage() == message
    assert "(`quant format face <key>: rows kernel failed; decode fallback`)" in _PAGE
    assert "does not compile" in str(warning.exc_info[1])
    (proof,) = kernels._PROOFS.values()
    assert proof.failed and not proof.ran

    # Not built again: for this layer, or for another of the same face,
    # tensor dtypes and shape.
    other, x_other, want_other = _toy_linear(40, 320, seed=2)
    with monkeypatch.context() as patch:
        patch.setattr(linear, "_rows", lambda *launch: pytest.fail("built"))
        assert _equal(layer(x), want)
        assert _equal(other(x_other), want_other)
    assert len(_logged(caplog, linear, logging.WARNING)) == 1

    # A stack of the face is another kernel, which fails on its own.
    switch = _toy_switch(40, 320)
    rows, indices = _ints((16, 1, 320)), _experts((16,))
    assert np.array_equal(np.array(switch(rows, indices)), _want(switch, rows, indices))
    assert len(_logged(caplog, linear, logging.WARNING)) == 2
    assert [proof.failed for proof in kernels._PROOFS.values()] == [True, True]

    # Layers of other faces are not affected: of the same shape, or others.
    toy(_toy("linear"))
    compiles, x, want = _toy_linear(40, 320)
    quantized, affine_layer = _pair(320, 72)
    assert _equal(compiles(x), want)
    assert _same_bits(affine_layer(_x((1, 320))), quantized(_x((1, 320))))
    assert [proof.failed for proof in kernels._PROOFS.values()] == [True, True] + [
        False,
        False,
    ]
    assert len(_logged(caplog, linear)) == 2


def test_a_failure_ahead_of_the_rows_is_raised_and_is_not_their_launch_s(caplog):
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    quantized, layer = _pair(320, 72)
    x = _x((1, 320))
    with pytest.raises(RuntimeError, match="Unable to build metal library"):
        layer(_does_not_compile(x))
    switched, switch = _switch_pair(320, 72)
    rows, indices = _x((16, 1, 320)), _experts((16,))
    with pytest.raises(RuntimeError, match="Unable to build metal library"):
        switch(_does_not_compile(rows), indices)
    with pytest.raises(RuntimeError, match="Unable to build metal library"):
        switch(rows, _does_not_compile(indices))
    # Nor is a tensor of the layer's that cannot be evaluated.
    for held, call in ((layer, (x,)), (switch, (rows, indices))):
        scales = held["scales"]
        held["scales"] = _does_not_compile(scales)
        with pytest.raises(RuntimeError, match="Unable to build metal library"):
            held(*call)
        held["scales"] = scales
    assert len(kernels._PROOFS) == 2
    for proof in kernels._PROOFS.values():
        assert not proof.failed and not proof.ran
    assert not _logged(caplog, linear)
    assert _same_bits(layer(x), quantized(x))
    assert _same_bits(switch(rows, indices), switched(rows, indices))


def test_only_an_unproven_launch_is_evaluated_in_the_call(monkeypatch):
    quantized, layer = _pair(320, 72)
    switched, switch = _switch_pair(320, 72)
    x, more = _x((3, 320)), _x((5, 320))
    rows, indices = _x((16, 1, 320)), _experts((16,))
    evaluate, evaluated = mx.eval, []

    def recorded(*arrays):
        evaluated.append(arrays)
        evaluate(*arrays)

    monkeypatch.setattr(mx, "eval", recorded)
    layer(x)
    # The inputs and the layer's tensors first, outside the launch's guard,
    # then the rows.
    assert [len(arrays) for arrays in evaluated] == [4, 1]
    assert evaluated[0][0] is x and evaluated[1][0].shape == (3, 72)
    assert all(a is layer[name] for a, name in zip(evaluated[0][1:], _NAMES))
    del evaluated[:]
    layer(x), layer(more)
    assert not evaluated
    switch(rows, indices)
    assert [len(arrays) for arrays in evaluated] == [5, 1]
    assert evaluated[0][0] is rows and evaluated[0][1] is indices
    assert all(a is switch[name] for a, name in zip(evaluated[0][2:], _NAMES))
    assert evaluated[1][0].shape == (16, 72)
    del evaluated[:]
    switch(rows, indices), switch(rows[4:12] + 0, indices[4:12] + 0)
    assert not evaluated
    # Another launch: one row for every pair.
    switch(rows[:1], indices)
    assert [len(arrays) for arrays in evaluated] == [5, 1]


def test_a_layer_holds_a_registered_format_s_tensors(toy):
    toy()
    (codes, scales), _, _ = _matrix(8, 64)
    matrix = {"weight": codes, "scales": scales}
    stack = {"weight": codes.reshape(2, 4, 64), "scales": scales.reshape(2, 4)}
    with pytest.raises(ValueError, match="no quant format is registered for mode 'x'"):
        FormatLinear("x", 8, 1, 64, matrix)
    names = "quant format 'toy' keeps a matrix as weight, scales, not as "
    with pytest.raises(ValueError, match=names + "biases, scales, weight"):
        FormatLinear("toy", 8, 1, 64, {**matrix, "biases": scales})
    with pytest.raises(ValueError, match=names + "weight"):
        FormatSwitchLinear("toy", 8, 1, 64, {"weight": stack["weight"]})
    axes = re.escape("FormatLinear: the weight has 2 axes, not the shape (2, 4, 64)")
    with pytest.raises(ValueError, match=axes):
        FormatLinear("toy", 8, 1, 64, stack)
    axes = re.escape("FormatSwitchLinear: the weight has 3 axes, not the shape (8, 64)")
    with pytest.raises(ValueError, match=axes):
        FormatSwitchLinear("toy", 8, 1, 64, matrix)
    ragged = re.escape(
        "FormatSwitchLinear: scales does not open with the weight's stack of (2,)"
    )
    with pytest.raises(ValueError, match=ragged):
        FormatSwitchLinear("toy", 8, 1, 64, {**stack, "scales": scales.reshape(4, 2)})
    # A matrix's tensors are the format's own beyond that.
    assert FormatLinear("toy", 8, 1, 64, {**matrix, "scales": scales.reshape(2, 4)})


def test_a_layer_reads_as_mlx_s_quantized_layers():
    quantized, layer = _pair(320, 72, bits=5, bias=True)
    assert (layer.mode, layer.bits, layer.group_size) == ("affine", 5, 32)
    assert (layer.input_dims, layer.output_dims) == (320, 72)
    assert [name for name, _ in tree_flatten(layer.parameters())] == [
        name for name, _ in tree_flatten(quantized.parameters())
    ]
    for name in (*_NAMES, "bias"):
        assert layer[name] is quantized[name]
    assert not tree_flatten(layer.trainable_parameters())
    assert repr(layer) == (
        "FormatLinear(input_dims=320, output_dims=72, bias=True, group_size=32, "
        "bits=5, mode=affine)"
    )
    switched, switch = _switch_pair(320, 72)
    assert (switch.input_dims, switch.output_dims, switch.num_experts) == (
        switched.input_dims,
        switched.output_dims,
        switched.num_experts,
    )
    assert "bias" not in switch and "bias=False" in repr(switch)


def test_x_of_another_width_is_refused():
    _, layer = _pair(320, 72)
    with pytest.raises(ValueError, match="x has 64 features, the layer takes 320"):
        layer(_x((1, 64)))
    _, switch = _switch_pair(320, 72)
    refused = re.escape("x is (..., M, 320), not of shape (320,)")
    with pytest.raises(ValueError, match=refused):
        switch(_x((320,)), _experts((3,)))
    refused = re.escape("x is (..., M, 320), not of shape (3, 1, 64)")
    with pytest.raises(ValueError, match=refused):
        switch(_x((3, 1, 64)), _experts((3,)))


def test_the_affine_decode_is_mlx_s_dequantize():
    quantized, _ = _pair(320, 72)
    weight = affine.FORMAT.decode(_tensors(quantized), 4, 32)
    assert weight.shape == (72, 320)
    assert _same_bits(weight, mx.dequantize(*_tensors(quantized).values(), 32, 4))
    # Several matrices of a stack: each as it is alone.
    stack, _ = _switch_pair(320, 72, bits=5)
    indices = mx.array([[5, 0], [2, 5]])
    several = affine.FORMAT.decode(
        {name: stack[name][indices] for name in _NAMES}, 5, 32
    )
    assert several.shape == (2, 2, 72, 320)
    alone = affine.FORMAT.decode({name: stack[name][2] for name in _NAMES}, 5, 32)
    assert _same_bits(several[1, 0], alone)
    assert not _same_bits(several[1, 0], several[0, 0])
