# SPDX-License-Identifier: Apache-2.0
"""The layers that hold a format's tensors (``omlx.formats.linear``).

The affine format through them has the bits of MLX's quantized layers, row by
row. A format oMLX does not know runs through them. What the rows kernel does
not take is the format's ``decode``, then MLX's matmul or, over a switch
layer's whole stack, its ``gather_mm``.
"""

from __future__ import annotations

import logging
import re
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

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
    (proof,) = kernels._PROOFS.values()
    assert not proof.ran and not proof.graphs and not proof.failed


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

    def recorded(*launch):
        seen.append(launch[7].dtype)
        return rows(*launch)

    monkeypatch.setattr(linear, "_rows", recorded)
    want = quantized(x, indices)
    for dtype in (mx.uint32, mx.int32, mx.int64, mx.uint8):
        assert _same_bits(layer(x, indices.astype(dtype)), want)
    # One kept graph, which MLX's compile traces for each type it is given.
    assert seen == [mx.uint32] * 4
    (proof,) = kernels._PROOFS.values()
    assert set(proof.graphs) == {((16, 1, 320), mx.bfloat16, (16,))}


def test_a_launch_takes_x_and_the_indices_as_they_are(monkeypatch):
    """Nothing is put in the graph for the kernel's sake where each row of x
    has its own run of experts, as in mlx-vlm's calls."""
    _, layer = _pair(320, 72)
    _, switch = _switch_pair(320, 72)
    rows, launches = linear._rows, []

    def recorded(kernel, k, n, x, operands, *launch):
        launches.append((x, *launch))
        return rows(kernel, k, n, x, operands, *launch)

    monkeypatch.setattr(linear, "_rows", recorded)

    def launched(held, describe, *arrays):
        """The Metal and the launch that ``describe`` gives ``arrays``: its
        graph as it is, which a layer hands to MLX's compile."""
        plan = linear._plan(held)
        metal, graph = describe(plan.kernel, plan.k, plan.n, *arrays)
        del launches[:]
        y = graph(*arrays, *plan.operands)
        (launch,) = launches
        assert y.shape == launch[2] and y.dtype == arrays[0].dtype
        return metal, launch

    x = _x((2, 3, 320))
    metal, launch = launched(layer, linear._linear, x)
    assert metal == mx.bfloat16
    assert launch[0] is x and launch[1:] == (6, (2, 3, 72))
    # A decode step's tokens, each with its top-k.
    x, indices = _x((1, 4, 1, 1, 320)), _experts((1, 4, 3))
    metal, launch = launched(switch, linear._switch, x, indices)
    rows_of_x, pairs, shape, rhs, per_row = launch
    assert rows_of_x is x and rhs.shape == (1, 4, 3) and rhs.dtype == mx.uint32
    assert (pairs, shape, per_row) == (12, (1, 4, 3, 1, 72), 3)
    assert metal == (mx.bfloat16, 3, False)
    # Otherwise x and the indices are broadcast to a row and an index a pair:
    # x along an axis of the indices, and rows of x for each index.
    for lead_m, routed, pairs in (((4, 1, 1), (2, 4, 3), 24), ((5, 3), (5,), 15)):
        x, indices = _x((*lead_m, 320)), _experts(routed)
        shape = (*routed, lead_m[-1], 72)
        metal, launch = launched(switch, linear._switch, x, indices)
        assert launch[0].shape == (*routed, lead_m[-1], 320)
        assert launch[1] == pairs and launch[2] == shape and launch[4] == 1
        assert launch[3].shape == (pairs,) and launch[3].dtype == mx.uint32
        assert metal == (mx.bfloat16, 1, False)
    # No pair, no launch.
    plan = linear._plan(switch)
    for x, indices in (
        (_x((8, 0, 320)), _experts((8,))),
        (_x((8, 1, 320))[:0], _experts((8,))[:0]),
    ):
        assert linear._switch(plan.kernel, plan.k, plan.n, x, indices) is None


def _held(graph):
    """What a launch's graph keeps from the call that described it."""
    return [cell.cell_contents for cell in graph.__closure__]


def test_a_launch_s_graph_holds_no_array():
    """A kept graph outlives the layer it is built for, in a proof that
    lasts as long as the process: it must not hold the layer's weights."""
    _, layer = _pair(320, 72)
    _, switch = _switch_pair(320, 72)
    plan = linear._plan(layer)
    _, graph = linear._linear(plan.kernel, plan.k, plan.n, _x((2, 320)))
    stack = linear._plan(switch)
    described = [graph]
    # Each row with its own experts, then the indices broadcast.
    for x, indices in (
        (_x((4, 1, 1, 320)), _experts((4, 3))),
        (_x((5, 3, 320)), _experts((5,))),
    ):
        _, graph = linear._switch(stack.kernel, stack.k, stack.n, x, indices)
        described.append(graph)
    kernels_ = (plan.kernel, stack.kernel)
    for graph in described:
        held = _held(graph)
        assert any(value is kernel for value in held for kernel in kernels_)
        for value in held:
            numbers = value if isinstance(value, tuple) else (value,)
            assert value in kernels_ or all(type(n) in (int, bool) for n in numbers)


def test_views_of_x_and_of_the_indices_are_read_as_their_rows():
    """MLX hands a kernel a contiguous copy of an input that is a view."""
    quantized, layer = _pair(320, 72)
    switched, switch = _switch_pair(320, 72)
    wide, tall = _x((4, 2, 640)), _x((320, 4), seed=5)
    # Unproven, then proven.
    for x in (wide[:, 1, ::2], tall.T, wide[1:3, 0, 320:]):
        alone = [quantized(x[row : row + 1]) for row in range(x.shape[0])]
        assert _same_bits(layer(x), mx.concatenate(alone))
    columns, pairs = _x((16, 320, 1), seed=6), _experts((16, 2))
    for x, indices in (
        (columns.swapaxes(1, 2), pairs[:, 1]),
        (wide[:, :1, 1::2], pairs.T[0, 3:7]),
    ):
        assert _same_bits(switch(x, indices), switched(x, indices))
    for proof in kernels._PROOFS.values():
        assert proof.ran and not proof.failed


def test_an_index_of_no_axis_is_a_launch_s_one_index():
    """MLX binds a kernel input of no axis as a scalar, which a kernel
    cannot subscript."""
    switched, switch = _switch_pair(320, 72)
    index = mx.array(4, dtype=mx.uint32)
    for rows in (1, 3):
        x = _x((rows, 320), seed=rows)
        got = switch(x, index)
        assert got.shape == (rows, 72)
        for row in range(rows):
            alone = switched(x[row : row + 1], index.reshape(1))
            assert _same_bits(got[row : row + 1], alone[0])
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {(mx.bfloat16, 1, True)} and not proof.failed


def test_shapes_that_do_not_broadcast_are_refused_by_mlx():
    _, switch = _switch_pair(320, 72)
    # Through the rows kernel, then through the decode: under the stack's
    # matrices, and over.
    for dtype in (mx.bfloat16, mx.float32):
        for count in (EXPERTS - 1, EXPERTS + 1):
            x, indices = _x((2, 3, 1, 320), dtype), _experts((count,))
            with pytest.raises(ValueError, match="cannot be broadcast"):
                switch(x, indices)
    (proof,) = kernels._PROOFS.values()
    assert not proof.ran and not proof.failed


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


def test_outside_the_kernel_a_switch_decodes_its_indices_matrices_or_its_stack(
    monkeypatch,
):
    """Fewer indices than the stack has matrices decode theirs alone. From as
    many, the stack is decoded once: a prefill's indices would not fit in
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

    # One index fewer than matrices: (1, 5) of them, with x's (2, 1). The
    # hint has no gather to go to.
    x, indices = _x((2, 1, 1, 320), mx.float32), _experts((1, EXPERTS - 1))
    got = layer(x, indices)
    assert got.shape == (2, 5, 1, 72) and _equal(got, want(x, indices))
    assert _equal(layer(x, indices, sorted_indices=True), got)
    # No index decodes no matrix.
    assert layer(x, indices[:, :0]).shape == (2, 0, 1, 72)
    assert whole == [False, False, False] and not hints
    # As many indices as matrices, then a prefill's, in any integer type.
    for routed, dtype in (((EXPERTS,), mx.uint32), ((2, 4, 3), mx.int64)):
        x = _x((*routed[:-1], 1, 1, 320), mx.float32)
        indices = _experts(routed).astype(dtype)
        got = layer(x, indices)
        assert got.shape == (*routed, 1, 72)
        assert _equal(got, want(x, indices, sorted_indices=False))
    assert whole == [False, False, False, True, True] and hints == [False, False]
    # Rows of x for each index count for nothing: 15 pairs of 5 indices
    # decode 5 matrices, 24 of 8 the stack.
    del whole[:], hints[:]
    x, indices = _x((5, 3, 320), mx.float32), _experts((5,))
    got = layer(x, indices)
    assert got.shape == (5, 3, 72) and _equal(got, want(x, indices))
    x, indices = _x((8, 3, 320), mx.float32), _experts((8,))
    got = layer(x, indices)
    assert got.shape == (8, 3, 72)
    assert _equal(got, want(x, indices, sorted_indices=False))
    assert whole == [False, True] and hints == [False]
    # mlx-vlm's hint goes to MLX's gather, which the pairs then are sorted for:
    # each with its row of x, or all with the one row.
    x, indices = _x((24, 1, 320), mx.float32), mx.sort(_experts((24,)))
    got = layer(x, indices, sorted_indices=True)
    assert _equal(got, want(x, indices, sorted_indices=True))
    got = layer(x[:1], indices, sorted_indices=True)
    assert got.shape == (24, 1, 72)
    assert mx.allclose(got, want(x[:1], indices), rtol=1e-5, atol=1e-5).item()
    assert hints == [False, True, True]
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

    def nothing(x, indices):
        y = layer(x, indices)
        mx.eval(y)
        return y.shape

    assert nothing(x[:, :0], indices[:0]) == (2, 0, 3, 40)
    assert nothing(x[:1, :1], indices[:0]) == (1, 0, 3, 40)
    assert nothing(x[:, :, :0], indices) == (2, 5, 0, 40)
    # Nor from as many indices as the stack has matrices.
    assert nothing(_ints((8, 0, 320)), _experts((8,))) == (8, 0, 40)
    assert proof.ran == {(mx.float32, 1, False)} and not proof.failed


def test_a_format_without_a_face_runs_through_its_decode(toy):
    toy()
    layer, x, want = _toy_linear(40, 320)
    assert _equal(layer(x), want)
    switch = _toy_switch(40, 320)
    rows, indices = _ints((2, 4, 1, 1, 320)), _experts((2, 4, 3))
    assert np.array_equal(np.array(switch(rows, indices)), _want(switch, rows, indices))
    few = indices[:1, :1]  # fewer indices than the stack has matrices
    assert np.array_equal(np.array(switch(rows, few)), _want(switch, rows, few))
    # Rows of x for each index, under the stack's matrices and over.
    for count in (EXPERTS - 1, EXPERTS + 2):
        several, each = _ints((count, 3, 320)), _experts((count,))
        got = switch(several, each)
        assert np.array_equal(np.array(got), _want(switch, several, each))
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
        mx.eval(linear._rows(plan.kernel, 320, 44, x, plan.operands, 1, (1, 44)))
    # Nor on a row of x that has a matrix.
    plan = linear._plan(_toy_switch(40, 320))
    rows, indices = _ints((16, 1, 320)), _experts((16,))
    words = "PAIRS_PER_ROW is the matrices of a row of x, 1 or more"
    for per_row in (0, -1):
        launch = (rows, plan.operands, 16, (16, 1, 40), indices, per_row)
        with pytest.raises(RuntimeError, match=words):
            mx.eval(linear._rows(plan.kernel, 320, 40, *launch))


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


def test_a_failure_ahead_of_the_rows_is_raised_and_is_not_their_launch_s(
    monkeypatch, caplog
):
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    quantized, layer = _pair(320, 72)
    switched, switch = _switch_pair(320, 72)
    x = _x((1, 320))
    rows, indices = _x((16, 1, 320)), _experts((16,))

    def refused(*launch):
        raise RuntimeError("no launch")

    # What fails as a launch is built, and a tensor of the layer's that
    # cannot be evaluated, are raised by the call that would prove the launch.
    # Only MLX's refusal of a trace's arrays is not: a ValueError that opens
    # with its words (another error of MLX's eval opens with them too).
    unreadable = (
        ValueError("no such tensor"),
        ValueError("in [eval] Attempting to eval an array during function"),
        RuntimeError("[eval] Attempting to eval an array without a primitive."),
    )
    for held, call in ((layer, (x,)), (switch, (rows, indices))):
        with monkeypatch.context() as patch:
            patch.setattr(linear, "_rows", refused)
            with pytest.raises(RuntimeError, match="no launch"):
                held(*call)
        for error in unreadable:
            with monkeypatch.context() as patch:
                patch.setattr(mx, "eval", lambda *arrays, error=error: _raise(error))
                with pytest.raises(type(error)) as raised:
                    held(*call)
            assert raised.value is error
        scales = held["scales"]
        held["scales"] = _does_not_compile(scales)
        with pytest.raises(RuntimeError, match="Unable to build metal library"):
            held(*call)
        held["scales"] = scales
    assert len(kernels._PROOFS) == 2
    for proof in kernels._PROOFS.values():
        assert not proof.failed and not proof.ran and not proof.graphs
    assert not _logged(caplog, linear)
    # x and the indices are their caller's to evaluate: the call that proves
    # a launch does not read them, and what fails in them is raised where
    # the result is evaluated. One row for every pair is a launch of its own.
    failing = (
        layer(_does_not_compile(x)),
        switch(_does_not_compile(rows), indices),
        switch(rows[:1], _does_not_compile(indices)),
    )
    for y in failing:
        with pytest.raises(RuntimeError, match="Unable to build metal library"):
            mx.eval(y)
    proven = [proof.ran for proof in kernels._PROOFS.values()]
    assert proven == [
        {mx.bfloat16},
        {(mx.bfloat16, 1, False), (mx.bfloat16, 16, False)},
    ]
    assert not any(proof.failed for proof in kernels._PROOFS.values())
    assert not _logged(caplog, linear)
    assert _same_bits(layer(x), quantized(x))
    assert _same_bits(switch(rows, indices), switched(rows, indices))
    # A proven launch that cannot be built for other rows is not kept.
    proof, _ = kernels._PROOFS.values()
    with monkeypatch.context() as patch:
        patch.setattr(linear, "_rows", refused)
        with pytest.raises(RuntimeError, match="no launch"):
            layer(_x((2, 320)))
    assert set(proof.graphs) == {((1, 320), mx.bfloat16)} and not proof.failed


def _raise(error):
    raise error


def _evaluations(monkeypatch):
    """What ``mx.eval`` is given, call by call."""
    evaluate, evaluated = mx.eval, []

    def recorded(*arrays):
        evaluated.append(arrays)
        evaluate(*arrays)

    monkeypatch.setattr(mx, "eval", recorded)
    return evaluated


def _zeros(a):
    return mx.array_equal(a, mx.zeros(a.shape, a.dtype)).item()


def test_an_unproven_launch_is_evaluated_on_zeros_and_a_proven_one_not_at_all(
    monkeypatch,
):
    quantized, layer = _pair(320, 72)
    switched, switch = _switch_pair(320, 72)
    x, more = _x((3, 320)), _x((5, 320))
    rows, indices = _x((16, 1, 320)), _experts((16,))
    evaluated = _evaluations(monkeypatch)
    y = layer(x)
    # The layer's tensors first, outside the launch's guard, then the rows of
    # zeros in x's place: neither x nor the call's result.
    assert [len(arrays) for arrays in evaluated] == [3, 1]
    assert all(a is layer[name] for a, name in zip(evaluated[0], _NAMES))
    (proving,) = evaluated[1]
    assert proving is not y and proving.shape == (3, 72)
    assert proving.dtype == mx.bfloat16 and _zeros(proving) and not _zeros(y)
    del evaluated[:]
    layer(x), layer(more)
    assert not evaluated
    y = switch(rows, indices)
    assert [len(arrays) for arrays in evaluated] == [3, 1]
    assert all(a is switch[name] for a, name in zip(evaluated[0], _NAMES))
    (proving,) = evaluated[1]
    assert proving is not y and proving.shape == (16, 1, 72)
    assert proving.dtype == mx.bfloat16 and _zeros(proving) and not _zeros(y)
    del evaluated[:]
    switch(rows, indices), switch(rows[4:12] + 0, indices[4:12] + 0)
    assert not evaluated
    # Another launch: one row for every pair.
    switch(rows[:1], indices)
    assert [len(arrays) for arrays in evaluated] == [3, 1]


def _built(monkeypatch):
    """The shapes of x that the rows kernel's launches are built for, as
    they are built: once for each graph that MLX's compile traces."""
    rows, built = linear._rows, []

    def recorded(*launch):
        built.append(launch[3].shape)
        return rows(*launch)

    monkeypatch.setattr(linear, "_rows", recorded)
    return built


def test_a_launch_is_built_once_and_replayed(monkeypatch):
    """MLX writes a kernel's source out for every launch that is built. A
    layer builds one for its arrays' shapes and types, and replays it."""
    quantized, layer = _pair(320, 72, bias=True)
    built = _built(monkeypatch)
    for seed in range(3):
        x = _x((1, 320), seed=seed)
        assert _same_bits(layer(x), quantized(x))
    assert built == [(1, 320)]
    # Other rows are another graph.
    for seed in range(2):
        x = _x((3, 320), seed=seed)
        alone = [quantized(x[row : row + 1]) for row in range(3)]
        assert _same_bits(layer(x), mx.concatenate(alone))
    assert built == [(1, 320), (3, 320)]
    # The graphs are the proof's: another layer of the face, the tensors'
    # types and the shape replays them over its own tensors.
    other_quantized, other = _pair(320, 72, seed=7)
    x = _x((1, 320), seed=9)
    assert _same_bits(other(x), other_quantized(x))
    assert not _same_bits(other(x), layer(x))
    assert built == [(1, 320), (3, 320)]
    (proof,) = kernels._PROOFS.values()
    assert set(proof.graphs) == {((1, 320), mx.bfloat16), ((3, 320), mx.bfloat16)}
    assert proof.ran == {mx.bfloat16}


def test_a_stack_s_launch_is_built_once_and_replayed(monkeypatch):
    switched, switch = _switch_pair(320, 72, bias=True)
    built = _built(monkeypatch)
    # A decode step's tokens with their top-k, then rows of x for each index.
    shapes = (((1, 4, 1, 1, 320), (1, 4, 3)), ((5, 3, 320), (5,)))
    for lead_m_k, routed in shapes:
        for seed in range(3):
            x, indices = _x(lead_m_k, seed=seed), _experts(routed, seed=seed)
            assert _same_bits(switch(x, indices), switched(x, indices))
    assert built == [(1, 4, 1, 1, 320), (5, 3, 320)]
    # The same x with indices of another shape is another graph.
    x, indices = _x((5, 3, 320)), _experts((2, 5))
    assert _same_bits(switch(x, indices), switched(x, indices))
    assert built == [(1, 4, 1, 1, 320), (5, 3, 320), (2, 5, 3, 320)]
    (proof,) = kernels._PROOFS.values()
    assert set(proof.graphs) == {
        ((1, 4, 1, 1, 320), mx.bfloat16, (1, 4, 3)),
        ((5, 3, 320), mx.bfloat16, (5,)),
        ((5, 3, 320), mx.bfloat16, (2, 5)),
    }
    # Another stack of the face and the shape, of other experts.
    others, other = _switch_pair(320, 72, seed=7)
    x, indices = _x((5, 3, 320), seed=9), _experts((5,), seed=9)
    assert _same_bits(other(x, indices), others(x, indices))
    assert len(built) == 3


def test_a_proof_keeps_so_many_graphs_then_builds_them_as_called(monkeypatch):
    """A prefill's chunks are shapes without end."""
    monkeypatch.setattr(kernels, "MAX_GRAPHS", 2)
    quantized, layer = _pair(320, 72)
    built = _built(monkeypatch)
    for rows in (1, 2, 1, 3, 1, 3):
        x = _x((rows, 320), seed=rows)
        alone = [quantized(x[row : row + 1]) for row in range(rows)]
        assert _same_bits(layer(x), mx.concatenate(alone))
    # The third drops the two that are kept, and the first is built again.
    assert built == [(1, 320), (2, 320), (3, 320), (1, 320)]
    (proof,) = kernels._PROOFS.values()
    assert set(proof.graphs) == {((3, 320), mx.bfloat16), ((1, 320), mx.bfloat16)}


def test_a_launch_that_fails_stops_the_launches_that_ran(toy, monkeypatch, caplog):
    """A proof's launches stop together: its kept graphs are dropped."""
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    floats = 'static_assert(sizeof(X) == 4, "floats alone");\n      return 0;'
    source = _TOY.replace("      return 0;", "      " + floats)
    assert source.count("floats alone") == 1
    toy(_toy("floats", source))
    layer, x, want = _toy_linear(40, 320)
    assert _equal(layer(x), want)
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {mx.float32} and len(proof.graphs) == 1
    # Halves are another instance of the Metal, which does not compile.
    half = x.astype(mx.float16)
    decoded = half @ layer._decoded().astype(mx.float16).T
    assert _equal(layer(half), decoded)
    (warning,) = _logged(caplog, linear, logging.WARNING)
    assert "floats alone" in str(warning.exc_info[1])
    assert proof.failed and proof.ran == {mx.float32} and not proof.graphs
    with monkeypatch.context() as patch:
        patch.setattr(linear, "_rows", lambda *launch: pytest.fail("built"))
        assert _equal(layer(x), want) and _equal(layer(half), decoded)
    assert not proof.graphs
    assert len(_logged(caplog, linear, logging.WARNING)) == 1


def test_a_proven_launch_is_built_into_a_function_that_mlx_compiles():
    """MLX's compile calls a layer once for the shapes it is given, over
    arrays that cannot be evaluated: a proven launch needs none."""
    quantized, layer = _pair(320, 72, bias=True)
    switched, switch = _switch_pair(320, 72)
    x, rows, indices = _x((1, 320)), _x((4, 1, 1, 320)), _experts((4, 3))
    layer(x), switch(rows, indices)

    def block(x, rows, indices):
        return layer(x) * 2, switch(rows, indices) * 2

    compiled = mx.compile(block)
    # The kept graphs, then other shapes of the proven Metal, whose launches
    # are built there.
    for lead, tokens in ((1, 4), (3, 8)):
        for seed in range(2):
            x = _x((lead, 320), seed=seed)
            rows = _x((tokens, 1, 1, 320), seed=seed)
            indices = _experts((tokens, 3), seed=seed)
            alone = [quantized(x[row : row + 1]) for row in range(lead)]
            y, pairs = compiled(x, rows, indices)
            assert _same_bits(y, mx.concatenate(alone) * 2)
            assert _same_bits(pairs, switched(rows, indices) * 2)
    for proof in kernels._PROOFS.values():
        assert len(proof.graphs) == 2 and not proof.failed


def test_an_unproven_launch_is_proven_in_a_function_that_mlx_compiles(
    monkeypatch, caplog
):
    """A layer's first call may be the one MLX's compile makes, over an x
    that cannot be evaluated: the launch is proven on zeros of x's shape and
    type, over the layer's tensors."""
    caplog.set_level(logging.INFO, logger=linear.__name__)
    quantized, layer = _pair(320, 72, bias=True)
    x, later = _x((1, 320)), _x((1, 320), seed=4)
    built = _built(monkeypatch)
    compiled = mx.compile(lambda x: layer(x) * 2)
    # No row is no launch: nothing to prove, and no zeros in its place.
    assert compiled(x[:0]).shape == (0, 72)
    (proof,) = kernels._PROOFS.values()
    assert not proof.ran and not proof.graphs and not built
    assert _same_bits(compiled(x), quantized(x) * 2)
    assert proof.ran == {mx.bfloat16} and not proof.failed
    assert set(proof.graphs) == {((1, 320), mx.bfloat16)}
    # Built for the zeros, then for the function's x.
    assert built == [(1, 320)] * 2
    face = formats.dot(layer, 320, 72)
    assert [r.getMessage() for r in _logged(caplog, linear)] == [
        f"quant format face {face.key}: rows kernel engaged"
    ]
    # Outside the function the kept graph is replayed: nothing is built,
    # evaluated or proven again.
    evaluated = _evaluations(monkeypatch)
    assert _same_bits(layer(later), quantized(later))
    assert _same_bits(compiled(later), quantized(later) * 2)
    assert built == [(1, 320)] * 2 and not evaluated
    assert len(_logged(caplog, linear)) == 1 and len(proof.graphs) == 1


def test_an_unproven_stack_s_launch_is_proven_in_a_function_that_mlx_compiles(
    monkeypatch, caplog
):
    """The indices of a stack's first call cannot be evaluated either: the
    zeros in their place name the stack's first matrix."""
    caplog.set_level(logging.INFO, logger=linear.__name__)
    switched, switch = _switch_pair(320, 72, bias=True)
    rows, indices = _x((4, 1, 1, 320)), _experts((4, 3))
    later, others = _x((4, 1, 1, 320), seed=4), _experts((4, 3), seed=4)
    built = _built(monkeypatch)
    compiled = mx.compile(lambda x, indices: switch(x, indices) * 2)
    # No pair is no launch.
    assert compiled(rows[:0], indices[:0]).shape == (0, 3, 1, 72)
    (proof,) = kernels._PROOFS.values()
    assert not proof.ran and not proof.graphs and not built
    assert _same_bits(compiled(rows, indices), switched(rows, indices) * 2)
    assert proof.ran == {(mx.bfloat16, 3, False)} and not proof.failed
    assert set(proof.graphs) == {((4, 1, 1, 320), mx.bfloat16, (4, 3))}
    assert built == [(4, 1, 1, 320)] * 2
    assert len(_logged(caplog, linear)) == 1
    evaluated = _evaluations(monkeypatch)
    assert _same_bits(switch(later, others), switched(later, others))
    assert _same_bits(compiled(later, others), switched(later, others) * 2)
    assert built == [(4, 1, 1, 320)] * 2 and not evaluated
    assert len(_logged(caplog, linear)) == 1 and len(proof.graphs) == 1


def test_a_face_that_does_not_compile_in_a_function_that_mlx_compiles_is_the_decode(
    toy, monkeypatch, caplog
):
    """The zeros fail as the launch would: the function is built over the
    decode."""
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    toy(_toy("broken", _TOY + 'static_assert(false, "does not compile");\n'))
    layer, x, want = _toy_linear(40, 320)
    switch = _toy_switch(40, 320)
    rows, indices = _ints((16, 1, 320)), _experts((16,))
    decode, decoded = _ToyFormat.decode, []

    def recorded(self, tensors, bits, group_size):
        decoded.append(tensors["weight"].shape)
        return decode(self, tensors, bits, group_size)

    monkeypatch.setattr(_ToyFormat, "decode", recorded)
    built = _built(monkeypatch)

    def block(x, rows, indices):
        return layer(x), switch(rows, indices)

    y, pairs = mx.compile(block)(x, rows, indices)
    assert _equal(y, want)
    assert np.array_equal(np.array(pairs), _want(switch, rows, indices))
    # Each was built once, for its zeros, and not for the function's arrays.
    assert x.shape == (320,) and built == [(320,), (16, 1, 320)]
    assert decoded == [(40, 320), (EXPERTS, 40, 320)]
    message = "quant format face toy_broken: rows kernel failed; decode fallback"
    warnings = _logged(caplog, linear, logging.WARNING)
    assert [warning.getMessage() for warning in warnings] == [message] * 2
    assert all("does not compile" in str(w.exc_info[1]) for w in warnings)
    assert len(kernels._PROOFS) == 2
    for proof in kernels._PROOFS.values():
        assert proof.failed and not proof.ran and not proof.graphs


def test_a_launch_over_tensors_that_cannot_be_evaluated_is_built_unproven(
    monkeypatch, caplog
):
    """``mx.compile(fn, inputs=state)`` hands a layer its own tensors as
    arrays that cannot be evaluated: there is nothing to prove a launch
    over. It is built as it is, and the proof neither marks nor keeps it."""
    caplog.set_level(logging.INFO, logger=linear.__name__)
    quantized, layer = _pair(320, 72, bias=True)
    switched, switch = _switch_pair(320, 72)
    x, rows, indices = _x((1, 320)), _x((4, 1, 1, 320)), _experts((4, 3))
    built = _built(monkeypatch)
    compiled = mx.compile(lambda x: layer(x) * 2, inputs=layer.state)
    assert _same_bits(compiled(x), quantized(x) * 2)
    stacked = mx.compile(lambda x, i: switch(x, i) * 2, inputs=switch.state)
    assert _same_bits(stacked(rows, indices), switched(rows, indices) * 2)
    # The launches, not the decode.
    assert built == [(1, 320), (4, 1, 1, 320)]
    assert len(kernels._PROOFS) == 2
    for proof in kernels._PROOFS.values():
        assert not proof.ran and not proof.graphs and not proof.failed
    assert not _logged(caplog, linear)
    # The first call outside such a function proves the launch.
    assert _same_bits(layer(x), quantized(x))
    assert _same_bits(switch(rows, indices), switched(rows, indices))
    for proof in kernels._PROOFS.values():
        assert len(proof.ran) == 1 and len(proof.graphs) == 1
    assert len(_logged(caplog, linear)) == 2


class _Thread:
    """A thread that runs MLX as oMLX's do: the one worker of an executor (a
    model's engine, or the loads')."""

    def __init__(self, name):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        # MLX gives a thread its default stream when the thread first uses
        # one, and traces a compiled function once more for it.
        self(lambda: mx.zeros((1,)).item())
        self.ident = self(threading.get_ident)

    def __call__(self, fn, *args):
        return self.pool.submit(fn, *args).result()

    def end(self):
        self.pool.shutdown()


@pytest.fixture
def threads():
    """Starts threads by name, and ends them with the test."""
    started = []

    def start(name):
        started.append(_Thread(name))
        return started[-1]

    yield start
    for thread in started:
        thread.end()


def _called(layer, *arrays):
    """A layer's result, evaluated by the thread that calls the layer."""
    y = layer(*arrays)
    mx.eval(y)
    return y


def _here(fn, *args):
    return fn(*args)


def test_a_launch_s_graph_is_called_by_the_thread_that_built_it(threads, monkeypatch):
    """MLX keeps what it traces a compiled function into thread by thread,
    and the function's end forgets it for one thread alone: the last that
    called it. A graph that two threads called would leave the other one its
    launch, which the next function at the graph's address replays there.
    So a thread keeps the graphs it builds, and calls no other's."""
    quantized, layer = _pair(320, 72, bias=True)
    switched, switch = _switch_pair(320, 72)
    x, rows, indices = _x((1, 320)), _x((4, 1, 1, 320)), _experts((4, 3))
    want, pairs = quantized(x), switched(rows, indices)
    mx.eval(want, pairs)
    compile_, callers = mx.compile, []

    def compiled(fn):
        graph, seen = compile_(fn), set()
        callers.append(seen)

        def called(*arrays):
            seen.add(threading.get_ident())
            return graph(*arrays)

        return called

    monkeypatch.setattr(mx, "compile", compiled)
    engine, other = threads("mlx-engine-a"), threads("mlx-engine-b")
    for run in (engine, _here, other, engine, _here, other):
        assert _same_bits(run(_called, layer, x), want)
        assert _same_bits(run(_called, switch, rows, indices), pairs)
    # A graph for each layer and thread, which that thread alone calls.
    assert len(callers) == 6 and all(len(seen) == 1 for seen in callers)
    idents = {engine.ident, other.ident, threading.get_ident()}
    assert len(idents) == 3 and set.union(*callers) == idents
    # What the launches proved is the process's: once.
    assert [proof.ran for proof in kernels._PROOFS.values()] == [
        {mx.bfloat16},
        {(mx.bfloat16, 3, False)},
    ]


class _TwiceFormat(_ToyFormat):
    """Another format of the toy's tensors: twice its weights."""

    mode = "twice"

    def decode(self, tensors, bits, group_size):
        return 2 * super().decode(tensors, bits, group_size)


def _half_typed():
    """The toy face with a load that compiles for float32 activations alone:
    halves are another instance of its Metal, which does not compile."""
    floats = 'static_assert(sizeof(X) == 4, "floats alone");\n      return 0;'
    source = _TOY.replace("      return 0;", "      " + floats)
    assert source.count("floats alone") == 1
    return _toy("floats", source)


@pytest.mark.parametrize(
    "end", ["its thread's overflow", "a failure", "its proof's end", "a caller's end"]
)
def test_a_graph_s_end_leaves_no_thread_the_launch_of_its_face(
    toy, threads, monkeypatch, end
):
    """Two formats of one tensor layout, a layer of each, of the same shapes:
    their launches have other Metal and the same arrays. A graph of the first
    that two threads called would end, one way or another, with its launch
    left in a thread, where a graph of the second, at the address of the
    first, would replay it: the first format's product, of the second's
    tensors."""
    toy(_half_typed())
    finish = "return sum * s[r];"
    assert _TOY.count(finish) == 1
    twice = _toy("twice", _TOY.replace(finish, "return 2 * sum * s[r];"))
    monkeypatch.setitem(formats._FORMATS, "twice", _TwiceFormat(twice))
    first, x, want = _toy_linear(40, 320)
    (codes, scales), _, _ = _matrix(40, 320, seed=1)
    second = FormatLinear("twice", 8, 1, 320, {"weight": codes, "scales": scales})
    more, half = _ints((3, 320)), x.astype(mx.float16)
    mx.eval(first.parameters(), second.parameters(), x, more, half)
    doubled = 2 * _want(second, x)
    # The second layer's plan, without a launch: nothing is made between the
    # first graph's end and the second's launch.
    several = _ints((MAX_ROWS + 1, 320))
    assert np.array_equal(np.array(second(several)), 2 * _want(second, several))
    described, addresses, sought = linear._linear, [], []

    def at_the_first_address(*launch):
        """A launch and, once ``sought``, one whose graph is at the address
        of the first graph built, if one of 64 functions lands there."""
        held = []
        for _ in range(64 if sought else 1):
            metal, graph = described(*launch)
            if not sought or id(graph) == addresses[0]:
                break
            held.append(graph)
        addresses.append(id(graph))
        return metal, graph

    monkeypatch.setattr(linear, "_linear", at_the_first_address)
    engine = threads("mlx-engine-a")
    for run in (engine, _here):
        assert _equal(run(_called, first, x), want)
    (key,) = (key for key in kernels._PROOFS if key[1].key == "toy_floats")
    ended = weakref.ref(kernels._PROOFS[key])
    if end == "a caller's end":
        other = threads("mlx-engine-b")
        assert _equal(other(_called, first, x), want)
        other.end()
    if end in ("its thread's overflow", "a caller's end"):
        monkeypatch.setattr(kernels, "MAX_GRAPHS", 1)
    held = [first]
    del first

    def ends_then_the_second_is_launched():
        if end == "a failure":
            _called(held[0], half)
        elif end == "its proof's end":
            del kernels._PROOFS[key]
            held.pop()
        else:
            # One more graph than the thread keeps.
            _called(held[0], more)
        sought.append(True)
        return _called(second, x)

    got = engine(ends_then_the_second_is_launched)
    assert np.array_equal(np.array(got), doubled)
    assert (ended() is None) == (end == "its proof's end")
    if end == "a failure":
        assert ended().failed
    proof = kernels._PROOFS[("format_linear", twice, (mx.uint8, mx.float32), 320, 40)]
    assert proof.ran == {mx.float32} and not proof.failed


def test_tensors_that_one_thread_evaluates_are_launched_by_others(
    threads, monkeypatch, caplog
):
    """A model's tensors are evaluated by the thread that loads it, and its
    layers are called by its engine's thread, in the engine's stream. What a
    launch proves is the process's: a second thread builds its own launch
    of the proven Metal, keeps it, and proves nothing again."""
    caplog.set_level(logging.INFO, logger=linear.__name__)
    load = threads("mlx-global")

    def loaded():
        quantized, layer = _pair(320, 72, bias=True)
        switched, switch = _switch_pair(320, 72)
        mx.eval(quantized.parameters(), switched.parameters())
        mx.eval(layer.parameters(), switch.parameters())
        return quantized, layer, switched, switch

    quantized, layer, switched, switch = load(loaded)
    steps = [
        (_x((1, 320), seed=s), _x((4, 1, 1, 320), seed=s), _experts((4, 3), seed=s))
        for s in range(6)
    ]
    built = _built(monkeypatch)

    def kept():
        """The graphs that the calling thread keeps, proof by proof."""
        return [len(proof.graphs) for proof in kernels._PROOFS.values()]

    def step(stream, x, rows, indices):
        """A step in an engine's stream: whether the layers have the bits
        of MLX's, and the graphs the thread then keeps."""
        with mx.stream(stream):
            same = _same_bits(layer(x), quantized(x))
            same = same and _same_bits(switch(rows, indices), switched(rows, indices))
        return same, kept()

    def engine(name):
        return threads(name), mx.new_thread_local_stream(mx.default_device())

    first, stream = engine("mlx-engine-a")
    assert first(step, stream, *steps[0]) == (True, [1, 1])
    ran = [{mx.bfloat16}, {(mx.bfloat16, 3, False)}]
    assert [proof.ran for proof in kernels._PROOFS.values()] == ran
    assert built == [(1, 320), (4, 1, 1, 320)] and len(_logged(caplog, linear)) == 2
    assert first(step, stream, *steps[1]) == (True, [1, 1]) and len(built) == 2
    # The graphs are the engine's thread's: neither the thread of the loads
    # nor this one has launched.
    assert kept() == [0, 0] and load(kept) == [0, 0]
    # Another engine's thread builds its launches, keeps them and replays
    # them. Nothing is proven again: no array is evaluated, nothing logged.
    second, other = engine("mlx-engine-b")
    evaluated = _evaluations(monkeypatch)
    assert second(kept) == [0, 0]
    assert second(step, other, *steps[2]) == (True, [1, 1])
    assert built == [(1, 320), (4, 1, 1, 320)] * 2
    assert second(step, other, *steps[3]) == (True, [1, 1]) and len(built) == 4
    # A thread's graphs end with it. The model's next engine, a thread of
    # the ident of the one that ended or of another, builds its own.
    first.end()
    third, stream = engine("mlx-engine-a")
    assert third(kept) == [0, 0]
    assert third(step, stream, *steps[4]) == (True, [1, 1]) and len(built) == 6
    assert third(step, stream, *steps[5]) == (True, [1, 1]) and len(built) == 6
    assert not evaluated and len(_logged(caplog, linear)) == 2
    assert [proof.ran for proof in kernels._PROOFS.values()] == ran
    assert not any(proof.failed for proof in kernels._PROOFS.values())


def test_a_launch_that_fails_on_one_thread_stops_every_thread_s(
    toy, threads, monkeypatch, caplog
):
    """A proof's launches stop together, whichever thread kept their graphs."""
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    toy(_half_typed())
    layer, x, want = _toy_linear(40, 320)
    half = x.astype(mx.float16)
    decoded = half @ layer._decoded().astype(mx.float16).T
    mx.eval(layer.parameters(), x, half, decoded)
    engine = threads("mlx-engine-a")
    assert _equal(engine(_called, layer, x), want) and _equal(layer(x), want)
    (proof,) = kernels._PROOFS.values()
    assert engine(lambda: len(proof.graphs)) == 1 and len(proof.graphs) == 1
    # Halves, on this thread: the launch fails.
    assert _equal(layer(half), decoded)
    assert proof.failed and len(_logged(caplog, linear, logging.WARNING)) == 1
    # The engine's thread keeps no graph, and builds none: its calls decode.
    decode, decodes = _ToyFormat.decode, []

    def recorded(self, tensors, bits, group_size):
        decodes.append(threading.get_ident())
        return decode(self, tensors, bits, group_size)

    monkeypatch.setattr(_ToyFormat, "decode", recorded)
    monkeypatch.setattr(linear, "_rows", lambda *launch: pytest.fail("built"))
    assert engine(lambda: len(proof.graphs)) == 0 and not proof.graphs
    assert _equal(engine(_called, layer, x), want)
    assert _equal(engine(_called, layer, half), decoded)
    assert decodes == [engine.ident] * 2
    assert len(_logged(caplog, linear, logging.WARNING)) == 1


def test_a_launch_built_while_another_thread_s_fails_is_not_kept(
    toy, threads, monkeypatch
):
    """A launch is built on one thread, of Metal that compiles, while one
    of the same proof fails on another: the call has its launch, and the
    proof, which failed, keeps no graph of it."""
    toy(_half_typed())
    layer, x, want = _toy_linear(40, 320)
    half = x.astype(mx.float16)
    decoded = half @ layer._decoded().astype(mx.float16).T
    mx.eval(layer.parameters(), x, half, decoded)
    engine = threads("mlx-engine-a")
    rows, launched = linear._rows, []

    def meanwhile(*launch):
        launched.append(launch[3].dtype)
        if launched == [mx.float32]:
            # The other thread's launch fails, from its first call to its
            # decode, before this one is built.
            assert _equal(engine(_called, layer, half), decoded)
        return rows(*launch)

    monkeypatch.setattr(linear, "_rows", meanwhile)
    assert _equal(layer(x), want)
    assert launched == [mx.float32, mx.float16]
    (proof,) = kernels._PROOFS.values()
    assert proof.failed and proof.ran == {mx.float32}
    assert not proof.graphs and engine(lambda: len(proof.graphs)) == 0
    # The calls after are the decode's, on either thread.
    monkeypatch.setattr(linear, "_rows", lambda *launch: pytest.fail("built"))
    assert _equal(layer(x), want) and _equal(engine(_called, layer, x), want)


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
