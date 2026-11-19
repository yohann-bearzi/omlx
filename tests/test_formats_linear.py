# SPDX-License-Identifier: Apache-2.0
"""The layers that hold a format's tensors (``omlx.formats.linear``).

The affine format through them has the bits of MLX's quantized layers, row by
row. A format oMLX does not know runs through them. What the rows kernel does
not take is the format's ``decode``, then MLX's matmul or, over a switch
layer's whole stack, its ``gather_mm``.
"""

from __future__ import annotations

import logging
import math
import operator
import re
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.switch_layers import SwitchGLU, SwitchLinear

from omlx import formats
from omlx.formats import affine, kernels, linear, testing
from omlx.formats.linear import MAX_ROWS, FormatLinear, FormatSwitchLinear
from tests.test_formats_kernels import _PAGE, _TOY, _matrix, _toy
from tests.test_qwen35_moe_routed_decode import _does_not_compile, _logged, _same_bits

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def by_hand():
    """The tests build format layers by hand, as a verified family's load
    would: a format's layer is built nowhere else."""
    with testing.verified("llama"):
        yield

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


def _switch_pair(k, n, bits=4, dtype=mx.bfloat16, bias=False, seed=0, experts=EXPERTS):
    """mlx-vlm's quantized switch layer, and a ``FormatSwitchLinear`` of its
    tensors."""
    mx.random.seed(seed)
    dense = SwitchLinear(k, n, experts, bias=bias)
    dense.set_dtype(dtype)
    quantized = dense.to_quantized(group_size=32, bits=bits)
    if bias:
        quantized["bias"] = mx.random.normal((experts, n)).astype(dtype)
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
    # The rows are not compiled into a launch: one row a launch, or 2 to 16
    # rows in groups of 2 to 4 (the shared launch, a Metal for each group).
    groups = {linear._group(m, face.vpt) for m in (2, 3, 6, MAX_ROWS)} - {1}
    assert proof.ran == {dtype, *((dtype, g) for g in groups)} and not proof.failed
    assert [r.getMessage() for r in _logged(caplog, linear)] == [
        f"quant format face {face.key}: rows kernel engaged"
    ] * (1 + len(groups))


def test_rows_go_to_the_shared_launch_in_the_largest_group_of_the_budget():
    # Affine at 4 bits on qmv (8 values a lane): groups of 4, 3, 2.
    groups = "1 2 3 4 1 3 1 4 3 2 1 4 1 2 3 4"
    assert [linear._group(m, 8) for m in range(1, 17)] == list(map(int, groups.split()))
    # 16 values a lane (affine at 4 and 5 bits on qmv_fast): groups of 2 at
    # most, 32 values a lane between them.
    assert [linear._group(m, 16) for m in (2, 3, 4, 6, 8)] == [2, 1, 2, 2, 2]
    # 32 values a lane: one row a group, the one-row launch.
    assert [linear._group(m, 32) for m in (2, 3, 4, 6)] == [1, 1, 1, 1]


def test_rows_in_groups_are_one_launch_and_their_stand_in_is_one_group(monkeypatch):
    """Two to 16 rows that a group of 2 to 4 divides are one launch of the
    shared-rows skeleton, proven on a stand-in of one group of zeros over
    one tile; a compiled function over the layer's state launches it too."""
    quantized, layer = _pair(320, 72)
    built = _built(monkeypatch)
    x = _x((2, 6, 320))
    alone = mx.concatenate([quantized(row[None]) for row in x.reshape(12, 320)])
    assert _same_bits(layer(x).reshape(12, 72), alone)
    # The stand-in (one group of 4 rows of zeros), then the call: one launch.
    assert built == [(4, 320), (2, 6, 320)]
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {(mx.bfloat16, 4)}
    compiled = mx.compile(lambda x: layer(x) * 2, inputs=layer.state)
    x = _x((3, 320))
    alone = mx.concatenate([quantized(x[i : i + 1]) for i in range(3)])
    assert _same_bits(compiled(x), alone * 2)
    assert proof.ran == {(mx.bfloat16, 4), (mx.bfloat16, 3)} and not proof.failed


def test_a_face_of_no_vpt_runs_its_rows_in_the_mat_vec_skeleton(toy, monkeypatch):
    toy(replace(_toy("no_vpt"), vpt=0))
    layer, _, _ = _toy_linear(40, 320)
    built = _built(monkeypatch)
    x = _ints((4, 320))
    assert np.array_equal(np.array(layer(x)), _want(layer, x))
    assert built == [(320,), (4, 320)]
    assert linear._plan(layer).shared is None


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


@pytest.mark.parametrize("n", [1, 4, 12, 60])
def test_an_n_of_part_of_a_tile_runs_in_the_shared_launch(n):
    """A matrix whose N is not whole tiles of 8: every row count runs in the
    shared-rows launch, the last tile holding the rows that are left (the
    shared expert's gate has one), each row with MLX's bits."""
    quantized, layer = _pair(320, n, bias=True)
    for shape in ((1, 320), (2, 320), (3, 320), (5, 320), (MAX_ROWS, 320)):
        x = _x(shape, seed=shape[0])
        got = layer(x)
        assert got.shape == (*shape[:-1], n)
        for row in range(shape[0]):
            assert _same_bits(got[row : row + 1], quantized(x[row : row + 1]))
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {(mx.bfloat16, g) for g in (1, 2, 3, 4)} and not proof.failed


def test_what_a_face_does_not_cover_is_the_decode_and_mlx_s_matmul():
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
    # One token of a top-3 MoE: fewer than 8 pairs, 3 of them a row.
    ((1, 1, 1), (1, 3), 3),
    # Of a top-4: 4 pairs, where two such rows would have 8.
    ((1, 1, 1), (1, 4), 4),
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
    # A pair is a row of the result: fewer than 8 are other Metal.
    assert proof.ran == {(dtype, per_row, got.size // n < 8)} and not proof.failed


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
    # The stand-in's, then a kept graph for each type of the indices: a
    # signed type's has their wrap.
    assert seen == [mx.uint32] * 5
    (proof,) = kernels._PROOFS.values()
    types = (mx.uint32, mx.int32, mx.int64, mx.uint8)
    assert set(proof.graphs) == {((16, 1, 320), mx.bfloat16, (16,), t) for t in types}


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
        metal, graph = describe(plan, *arrays)
        del launches[:]
        y = graph(*arrays, *plan.operands)
        (launch,) = launches
        assert y.shape == launch[2] and y.dtype == arrays[0].dtype
        return metal, launch

    # Rows that no group of 2 to 4 divides: the one-row launch.
    x = _x((1, 5, 320))
    metal, launch = launched(layer, linear._linear, x)
    assert metal == mx.bfloat16
    assert launch[0] is x and launch[1:] == (5, (1, 5, 72))
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
        assert linear._switch(plan, x, indices) is None


def _held(graph):
    """What a launch's graph keeps from the call that described it."""
    return [cell.cell_contents for cell in graph.__closure__]


def test_a_launch_s_graph_holds_no_array():
    """A kept graph outlives the layer it is built for, in a proof that
    lasts as long as the process: it must not hold the layer's weights."""
    _, layer = _pair(320, 72)
    _, switch = _switch_pair(320, 72)
    plan = linear._plan(layer)
    _, graph = linear._linear(plan, _x((2, 320)))
    _, one_row = linear._linear(plan, _x((5, 320)))
    stack = linear._plan(switch)
    described = [graph, one_row]
    # Each row with its own experts, then the indices broadcast.
    for x, indices in (
        (_x((4, 1, 1, 320)), _experts((4, 3))),
        (_x((5, 3, 320)), _experts((5,))),
    ):
        _, graph = linear._switch(stack, x, indices)
        described.append(graph)
    kernels_ = (plan.kernel, plan.shared, stack.kernel)
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


def test_a_tensor_of_float64_is_no_face_and_the_layer_runs_composed(
    toy, caplog, dot_defects
):
    """The GPU takes no float64: a layer with such a tensor has no face (a
    stand-in of it could not be made), the defect is logged once, and the
    layer and the stack run their composed path, on the CPU here."""
    toy(_toy("linear"))
    layer, x, _ = _toy_linear(16, 320)
    switch = _toy_switch(8, 320)
    rows, indices = _ints((2, 1, 320)), mx.array([[3], [5]])
    with mx.stream(mx.cpu), caplog.at_level(logging.ERROR, logger=formats.__name__):
        layer["scales"] = layer["scales"].astype(mx.float64)
        switch["scales"] = switch["scales"].astype(mx.float64)
        got, stacked = layer(x), switch(rows, indices)
        mx.eval(got, stacked)
    assert np.allclose(np.array(got), _want(layer, x))
    assert np.allclose(np.array(stacked), _want(switch, rows, indices))
    defect = (
        "ValueError('face toy_linear: scales of float64, which the GPU does not "
        "take')"
    )
    assert dot_defects == {("toy", defect)} and len(caplog.records) == 1
    assert not kernels._PROOFS
    dot_defects.clear()  # expected


class _StackableToy(_ToyFormat):
    """The toy with its layout: codes (n, k), one float32 scale a row."""

    def layout(self, k, n, bits, group_size):
        return {"weight": ((n, k), mx.uint8), "scales": ((n,), mx.float32)}


class _ColumnScaled(_StackableToy):
    """Scales of each column: not a row's alone."""

    def layout(self, k, n, bits, group_size):
        return {"weight": ((n, k), mx.uint8), "scales": ((k,), mx.float32)}


class _GlobalScaled(_StackableToy):
    """One scale for the matrix, kept as a row of one: its first axis is not N
    once stacked."""

    def layout(self, k, n, bits, group_size):
        return {"weight": ((n, k), mx.uint8), "scales": ((1,), mx.float32)}


class _RowsLeak(_StackableToy):
    """A layout of rows and a decode that is not: each row scaled by the
    matrix's largest scale."""

    def decode(self, tensors, bits, group_size):
        codes = tensors["weight"].astype(mx.float32) - 128.0
        return codes * mx.max(tensors["scales"])


class _PaddedScales(_StackableToy):
    """Scales padded to 100 rows: doubled, the layout reads as a stack."""

    def layout(self, k, n, bits, group_size):
        return {"weight": ((n, k), mx.uint8), "scales": ((max(n, 100),), mx.float32)}


class _SquareScales(_StackableToy):
    """Scales (n, n): N first, and not the same with 2N."""

    def layout(self, k, n, bits, group_size):
        return {"weight": ((n, k), mx.uint8), "scales": ((n, n), mx.float32)}


class _HalfScalesWhenStacked(_StackableToy):
    def layout(self, k, n, bits, group_size):
        dtype = mx.float16 if n > 100 else mx.float32
        return {"weight": ((n, k), mx.uint8), "scales": ((n,), dtype)}


class _OneMoreWhenStacked(_StackableToy):
    def layout(self, k, n, bits, group_size):
        layout = super().layout(k, n, bits, group_size)
        return {**layout, "gains": ((n,), mx.float32)} if n > 100 else layout


class _EmptyLayout(_StackableToy):
    def layout(self, k, n, bits, group_size):
        return {}


class _LayoutRaises(_StackableToy):
    def layout(self, k, n, bits, group_size):
        raise _OwnError("no layout")


@pytest.mark.parametrize("n", [0, -1])
def test_stackable_refuses_nonpositive_rows_without_asking_layout(n):
    calls = []

    class Reader(_StackableToy):
        def layout(self, k, n, bits, group_size):
            calls.append(n)
            return super().layout(k, n, bits, group_size)

    assert not formats.stackable(Reader(), 320, n, 8, 1)
    assert calls == []


@pytest.mark.parametrize("mismatch", ["weight", "scales", "undeclared"])
def test_stackable_requires_the_declared_matrix_tensors(mismatch):
    class Reader(_StackableToy):
        def layout(self, k, n, bits, group_size):
            layout = super().layout(k, n, bits, group_size)
            if mismatch == "undeclared":
                layout["extra"] = ((n,), mx.float32)
            else:
                del layout[mismatch]
            return layout

    assert not formats.stackable(Reader(), 320, 8, 8, 1)


def test_a_format_s_matrices_stack_as_its_layout_says(caplog, dot_defects):
    """formats.stackable reads the layout: every tensor's first axis is N,
    and the layout of 2N is the same with 2N. No layout, or one that
    raises (logged once), is no stacking."""
    assert formats.stackable(_StackableToy(), 320, 72, 8, 1)
    assert not formats.stackable(_ToyFormat(), 320, 72, 8, 1)  # no layout
    assert not formats.stackable(_ColumnScaled(), 320, 72, 8, 1)
    assert not formats.stackable(_GlobalScaled(), 320, 72, 8, 1)
    for fmt in (
        _PaddedScales(),  # not N first
        _SquareScales(),  # not the same with 2N
        _HalfScalesWhenStacked(),  # another dtype
        _OneMoreWhenStacked(),  # another tensor
        _EmptyLayout(),  # no tensor at all
    ):
        assert not formats.stackable(fmt, 320, 72, 8, 1), type(fmt).__name__
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        for _ in range(2):
            assert not formats.stackable(_LayoutRaises(), 320, 72, 8, 1)
    assert len(caplog.records) == 1
    assert dot_defects == {("toy", "_OwnError('no layout')")}
    dot_defects.clear()  # expected


@pytest.mark.parametrize("checkpoint_copy", [False, True])
def test_rows_stack_builds_each_layer_with_the_format_being_checked(
    monkeypatch, checkpoint_copy
):
    seen = []

    class Reader(_StackableToy):
        def dot(self, layer, k, n):
            seen.append((layer.fmt is self, n))
            return super().dot(layer, k, n)

    face = _toy("stack_reader")
    registered = Reader(face)
    fmt = Reader(face) if checkpoint_copy else registered
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", registered)
    matrices = [
        dict(zip(("weight", "scales"), _matrix(8, 320, seed)[0], strict=True))
        for seed in (1, 2)
    ]
    testing.rows_stack(fmt, *matrices, 320, 8, 1, _ints((320,)))
    assert seen == [(True, 16), (True, 8), (True, 8)]


def test_rows_stack_checks_a_format_s_promise(toy, monkeypatch):
    """testing.rows_stack: two matrices stacked decode, and run their dot
    rows, as the two do apart. A format whose decode mixes rows fails it,
    and so does one its layout does not let stack."""
    face = _toy("stack")
    first, second = (
        dict(zip(("weight", "scales"), _matrix(8, 320, seed)[0], strict=True))
        for seed in (1, 2)
    )
    second["scales"] = second["scales"] * 4  # another largest scale
    x = _ints((320,))
    toy(face)
    fmt = _StackableToy(face)
    monkeypatch.setitem(formats._FORMATS, "toy", fmt)
    testing.rows_stack(fmt, first, second, 320, 8, 1, x)
    leak = _RowsLeak(face)
    monkeypatch.setitem(formats._FORMATS, "toy", leak)
    with pytest.raises(AssertionError, match="decode of two stacked matrices"):
        testing.rows_stack(leak, first, second, 320, 8, 1)
    with pytest.raises(AssertionError, match="not stackable at n=8, k=320"):
        testing.rows_stack(_ToyFormat(face), first, second, 320, 8, 1)
    # A face that scales every row by its matrix's first scale: its decode
    # keeps the rows apart, its dot rows do not.
    first_scale = _toy(
        "first_scale", source=_TOY.replace("s(m.scales + row)", "s(m.scales)")
    )
    fmt = _StackableToy(first_scale)
    monkeypatch.setitem(formats._FORMATS, "toy", fmt)
    testing.rows_stack(fmt, first, second, 320, 8, 1)
    with pytest.raises(AssertionError, match="dot rows of two stacked matrices"):
        testing.rows_stack(fmt, first, second, 320, 8, 1, x)


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
    # One row a launch (and 5), 2 x 3 rows in groups of 3, 16 in groups of 4.
    want = {mx.float32, (mx.float32, 3), (mx.float32, 4)}
    assert proof.ran == want and not proof.failed
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
    assert proof.ran == {(mx.float32, per_row, got.size // 40 < 8)}


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
    # N that is not whole tiles of 8 outputs: a face of no vpt (no shared
    # launch) and a stack (no launch takes one).
    rows, indices = _ints((16, 1, 320)), _experts((16,))
    switch = _toy_switch(12, 320)
    assert np.array_equal(np.array(switch(rows, indices)), _want(switch, rows, indices))
    assert linear._plan(switch) is None and not kernels._PROOFS
    toy(replace(_toy("linear"), vpt=0))
    layer, x, want = _toy_linear(12, 320)
    assert _equal(layer(x), want)
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


def test_a_call_finds_its_plan_and_bias_kept_on_the_layer(toy, monkeypatch):
    """A layer works out its plan and finds its bias at its first call; its
    calls after read them off the layer, with no walk of its arrays, until
    an array or an attribute of the layer is assigned or removed."""
    toy(_toy("linear"))
    planned, plan = [], linear._plan

    def recorded(layer):
        planned.append(layer)
        return plan(layer)

    monkeypatch.setattr(linear, "_plan", recorded)
    bias = mx.arange(40).astype(mx.float32)
    layer, x, want = _toy_linear(40, 320, bias=bias)
    assert _equal(layer(x), want + bias)
    with monkeypatch.context() as patch:
        walked = lambda *a, **k: pytest.fail("walked")  # noqa: E731
        patch.setattr(linear, "cached_per_module", walked, raising=False)
        assert _equal(layer(x), want + bias)
    assert planned == [layer]
    # Each way MLX changes a layer: an assignment, a removal, an update, an
    # attribute's write.
    layer["scales"] = layer["scales"] * 2
    assert _equal(layer(x), 2 * want + bias) and len(planned) == 2
    layer.pop("bias")
    assert _equal(layer(x), 2 * want) and len(planned) == 3
    layer.bias = bias
    assert _equal(layer(x), 2 * want + bias) and len(planned) == 4
    layer.update({"scales": layer["scales"] / 2})
    assert _equal(layer(x), want + bias) and len(planned) == 5
    del layer["bias"]
    assert _equal(layer(x), want) and len(planned) == 6
    layer.group_size = 2
    assert _equal(layer(x), want) and len(planned) == 7
    # A stack's bias as well.
    switch = _toy_switch(40, 320)
    rows, indices = _ints((16, 1, 320)), _experts((16,))
    alone = np.array(switch(rows, indices))
    stacked = mx.arange(EXPERTS * 40).astype(mx.float32).reshape(EXPERTS, 40)
    switch["bias"] = stacked
    assert np.array_equal(np.array(switch(rows, indices)), _want(switch, rows, indices))
    switch.pop("bias")
    assert np.array_equal(np.array(switch(rows, indices)), alone)


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


class _OwnError(Exception):
    """An error of a type that is no other's."""


def test_whatever_a_launch_s_first_evaluation_raises_is_the_launch_s_failure(
    toy, monkeypatch, caplog
):
    """MLX reports what it cannot build or run in errors of several types.
    Whichever is raised as a launch is first evaluated, the call is the
    decode's and the proof has failed."""
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    toy(_toy("linear"))
    layer, x, want = _toy_linear(40, 320)
    switch = _toy_switch(40, 320)
    rows, indices = _ints((16, 1, 320)), _experts((16,))
    error, launches = _OwnError("no launch"), []
    evaluate, rows_of = mx.eval, linear._rows

    def launch(*args):
        launches.append(rows_of(*args))
        return launches[-1]

    def refused(*arrays):
        """Evaluates the stand-ins' zeros, and refuses their launches."""
        if any(a is out for a in arrays for out in launches):
            raise error
        evaluate(*arrays)

    with monkeypatch.context() as patch:
        patch.setattr(linear, "_rows", launch)
        patch.setattr(mx, "eval", refused)
        y, pairs = layer(x), switch(rows, indices)
    assert len(launches) == 2
    assert _equal(y, want)
    assert np.array_equal(np.array(pairs), _want(switch, rows, indices))
    warnings = _logged(caplog, linear, logging.WARNING)
    assert [warning.exc_info[1] for warning in warnings] == [error, error]
    assert len(kernels._PROOFS) == 2
    for proof in kernels._PROOFS.values():
        assert proof.failed and not proof.ran and not proof.graphs


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

    # What fails as a launch is built is raised by the call that would
    # prove the launch.
    for held, call in ((layer, (x,)), (switch, (rows, indices))):
        with monkeypatch.context() as patch:
            patch.setattr(linear, "_rows", refused)
            with pytest.raises(RuntimeError, match="no launch"):
                held(*call)
    assert len(kernels._PROOFS) == 2
    for proof in kernels._PROOFS.values():
        assert not proof.failed and not proof.ran and not proof.graphs
    assert not _logged(caplog, linear)
    # A tensor of the layer's that fails when it is evaluated is its
    # caller's: the call proves its launch on zeros, and the failure is
    # raised where the result is evaluated.
    for held, call in ((layer, (x,)), (switch, (rows, indices))):
        scales = held["scales"]
        held["scales"] = _does_not_compile(scales)
        y = held(*call)
        held["scales"] = scales
        with pytest.raises(RuntimeError, match="Unable to build metal library"):
            mx.eval(y)
    proven = [proof.ran for proof in kernels._PROOFS.values()]
    assert proven == [{mx.bfloat16}, {(mx.bfloat16, 1, False)}]
    # So are x and the indices. One row for every pair is a launch of its
    # own.
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
            layer(_x((5, 320)))
    assert set(proof.graphs) == {((1, 320), mx.bfloat16)} and not proof.failed


def test_what_fails_in_a_stand_in_s_zeros_is_raised_and_fails_no_proof(
    monkeypatch, caplog
):
    """A stand-in's zeros are evaluated by the proof ahead of its launch,
    outside the guard of the launch's evaluation: what fails there (an
    allocation, an error that an earlier evaluation left pending on the GPU)
    is not the face's. The call raises it, and the proof has neither failed
    nor proven anything."""
    caplog.set_level(logging.INFO, logger=linear.__name__)
    quantized, layer = _pair(320, 72)
    switched, switch = _switch_pair(320, 72)
    x, rows, indices = _x((1, 320)), _x((16, 1, 320)), _experts((16,))
    launched, evaluated, error = _launched(monkeypatch), [], _OwnError("no zeros")

    def refused(*arrays):
        evaluated.append(arrays)
        raise error

    for held, call in ((layer, (x,)), (switch, (rows, indices))):
        with monkeypatch.context() as patch:
            patch.setattr(mx, "eval", refused)
            with pytest.raises(_OwnError) as raised:
                held(*call)
        assert raised.value is error
    # Each call evaluated its stand-in's inputs once, and built no launch.
    assert not launched
    assert [[a.shape for a in arrays] for arrays in evaluated] == [
        [(320,), *(layer[name].shape for name in _NAMES)],
        [(8, 320), *((1, *switch[name].shape[1:]) for name in _NAMES), (8,)],
    ]
    assert len(kernels._PROOFS) == 2
    for proof in kernels._PROOFS.values():
        assert not proof.failed and not proof.ran and not proof.graphs
    assert not _logged(caplog, linear)
    # In a function compiled over the model's state, MLX leaves the state as
    # its trace left it when such an error escapes (UPSTREAM-BUGS M2).
    assert (
        "Inside a function compiled over the model's state, such an error "
        "leaves the state's arrays as MLX's trace left them, placeholders "
        "without a value (MLX restores a compiled function's state only when "
        "the function returns): the model is loaded again."
    ) in _PAGE
    # The calls after prove the launches.
    assert _same_bits(layer(x), quantized(x))
    assert _same_bits(switch(rows, indices), switched(rows, indices))
    assert all(proof.ran and not proof.failed for proof in kernels._PROOFS.values())


def test_a_stand_in_s_inputs_are_views_of_one_buffer_of_zeros(monkeypatch):
    """A stand-in's x, tensors and indices are views of one buffer of zeros,
    as many bytes as the largest of them: one fill and, where MLX's buffer
    cache is empty (oMLX clears it after each prefill chunk), one new
    buffer."""
    _, layer = _pair(320, 72)
    _, switch = _switch_pair(320, 72)
    x, rows, indices = _x((1, 320)), _x((16, 1, 320)), _experts((16,))
    zeros, asked = mx.zeros, []

    def recorded(shape, *dtype, **named):
        asked.append((shape, *dtype))
        return zeros(shape, *dtype, **named)

    for held, call in ((layer, (x,)), (switch, (rows, indices))):
        with monkeypatch.context() as patch:
            patch.setattr(mx, "zeros", recorded)
            held(*call)
    # The largest is the weight: a matrix's, and one matrix of a stack's.
    one_matrix = switch["weight"].nbytes // EXPERTS
    assert asked == [((layer["weight"].nbytes,), mx.uint8), ((one_matrix,), mx.uint8)]


def _address(a):
    """Where ``a``'s first byte is in memory."""
    data = mx.view(a.reshape(-1), mx.uint8)
    mx.eval(data)
    return np.frombuffer(memoryview(data), np.uint8).ctypes.data


def test_a_stand_in_s_zeros_open_the_buffer_whatever_input_is_the_largest(
    toy, monkeypatch
):
    """The buffer has the bytes of the largest input, here x's: 8 rows of
    float32 against one matrix of 8 rows of byte codes. Each input is a view
    of it from its first byte: a view at an offset that is no multiple of
    its dtype's size is a copy in MLX."""
    toy(_toy("linear"))
    switch = _toy_switch(8, 320)
    zeros, asked = mx.zeros, []

    def recorded(shape, *dtype, **named):
        asked.append((shape, *dtype))
        return zeros(shape, *dtype, **named)

    x, indices = _ints((16, 1, 320)), _experts((16,))
    evaluated = _evaluations(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(mx, "zeros", recorded)
        switch(x, indices)
    assert asked == [((8 * 320 * 4,), mx.uint8)]
    stand_in = evaluated[0]
    assert [a.shape for a in stand_in] == [(8, 320), (1, 8, 320), (1, 8), (8,)]
    assert len({_address(a) for a in stand_in}) == 1


def test_zeros_of_2_GiB_or_more_are_each_their_own(monkeypatch):  # noqa: N802
    """MLX's shapes are int32: a buffer of 2 GiB or more as one axis of
    bytes cannot be made. An input of that many bytes (a vocabulary head of
    262144 x 4096 two-byte codes) is zeros of its own; the others share one
    buffer, wherever the largest stands. Nothing is evaluated here: MLX
    allocates when it evaluates."""
    zeros, asked, streams = mx.zeros, [], []

    def recorded(shape, *dtype, **named):
        asked.append((shape, *dtype))
        streams.append(named.get("stream"))
        return zeros(shape, *dtype, **named)

    monkeypatch.setattr(mx, "zeros", recorded)
    x, scales = ((4096,), mx.bfloat16), ((262144, 64), mx.bfloat16)
    head = ((262144, 4096), mx.bfloat16)
    under = ((2**31 - 1,), mx.uint8)
    for specs, buffers in (
        ((x, head), [((8192,), mx.uint8), head]),
        ((head, x, scales), [((2**25,), mx.uint8), head]),
        ((x, under, scales), [((2**31 - 1,), mx.uint8)]),
        ((under, x), [((2**31 - 1,), mx.uint8)]),
        # Every input its own: no shared buffer is made.
        ((head, head), [head, head]),
    ):
        del asked[:]
        made = kernels.zeros(*specs)
        assert [(a.shape, a.dtype) for a in made] == list(specs)
        assert asked == buffers
    # Each on the GPU's stream, the launch's.
    assert set(streams) == {mx.gpu}


def test_a_stand_in_s_zeros_are_made_and_waited_for_on_the_gpu(monkeypatch):
    """A launch runs on the GPU's stream, whatever the caller's default
    stream. Its stand-in's zeros are made there too, each op given the GPU's
    stream (the process's default device, which every thread shares, is
    left as it is), and the proof synchronizes the stream after their
    evaluation: what an earlier evaluation left pending on the GPU is raised
    then, ahead of the launch, not in its evaluation."""
    quantized, layer = _pair(320, 72)
    x = _x((1, 320))
    synchronize, evaluate, events = mx.synchronize, mx.eval, []

    def made(op, name):
        def recorded(*args, **named):
            events.append((name, mx.default_device(), named.get("stream")))
            return op(*args, **named)

        return recorded

    def synchronized(*stream):
        events.append(("synchronize", *stream))
        synchronize(*stream)

    def evaluated(*arrays):
        events.append(("eval", len(arrays)))
        evaluate(*arrays)

    with monkeypatch.context() as patch:
        for name in ("zeros", "as_strided", "view", "reshape"):
            patch.setattr(mx, name, made(getattr(mx, name), name))
        patch.setattr(mx, "synchronize", synchronized)
        patch.setattr(mx, "eval", evaluated)
        with mx.stream(mx.cpu):
            y = layer(x)
    ops = [event for event in events if event[0] not in ("eval", "synchronize")]
    assert ops[0][0] == "zeros" and len(ops) == 1 + 3 * 4
    assert all(event[1:] == (mx.cpu, mx.gpu) for event in ops)
    assert events[len(ops) :] == [("eval", 4), ("synchronize", mx.gpu), ("eval", 1)]
    assert _same_bits(y, quantized(x))
    assert (
        "The stand-in's zeros are made on the GPU's stream and evaluated by the "
        "proof ahead of the launch, and the stream is synchronized: what fails in "
        "making them, or what an earlier evaluation left pending on the GPU, is "
        "raised by the call and fails no proof."
    ) in _PAGE


def test_a_layer_whose_proof_failed_works_out_no_launch(toy, monkeypatch):
    """Once its proof has failed, a layer's call is the decode at once: it
    does not work out a launch that it would not run."""
    toy(_toy("linear"))
    layer, x, want = _toy_linear(40, 320)
    switch = _toy_switch(40, 320)
    rows, indices = _ints((16, 1, 320)), _experts((16,))
    layer(x), switch(rows, indices)
    assert len(kernels._PROOFS) == 2
    for proof in kernels._PROOFS.values():
        proof.fail()
    for describe in ("_linear", "_switch"):
        monkeypatch.setattr(linear, describe, lambda *call: pytest.fail("described"))
    monkeypatch.setattr(formats.Dot, "exact_for", lambda *asked: pytest.fail("asked"))
    assert _equal(layer(x), want)
    assert np.array_equal(np.array(switch(rows, indices)), _want(switch, rows, indices))


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


def _launched(monkeypatch, tiles=None):
    """What the rows kernel is launched with, launch by launch: x, the
    matrix's tensors and the indices; and, in ``tiles``, the tiles of 8
    outputs that its grid computes of each pair. A launch of more or fewer
    pairs than its x's rows have, than its indices name or than its output
    holds, or of more tiles than N has, fails the test: its grid would read
    and write past them."""
    rows, launched = linear._rows, []

    def recorded(kernel, k, n, x, operands, pairs, shape, *launch):
        rhs, per_row = (*launch, None, 1)[:2]
        assert x.size // k * per_row == pairs == math.prod(shape) // n
        assert rhs is None or rhs.size == pairs
        launched.append((x, list(operands), rhs))

        def gridded(grid, **named):
            assert grid[:2] == (32, 2 * pairs) and 0 < grid[2] <= n // 8
            if tiles is not None:
                tiles.append(grid[2])
            return kernel(grid=grid, **named)

        return rows(gridded, k, n, x, operands, pairs, shape, *launch)

    monkeypatch.setattr(linear, "_rows", recorded)
    return launched


def _same(arrays, others):
    """Whether ``arrays`` are ``others``, one by one."""
    return len(arrays) == len(others) and all(map(operator.is_, arrays, others))


def _like(arrays, tensors):
    """Whether ``arrays`` are zeros of the shapes and dtypes of ``tensors``,
    and none of them."""
    return all(
        a is not tensor and (a.shape, a.dtype) == (tensor.shape, tensor.dtype)
        for a, tensor in zip(arrays, tensors, strict=True)
    ) and all(_zeros(a) for a in arrays)


def test_an_unproven_launch_is_evaluated_on_its_stand_in_and_a_proven_one_not_at_all(
    monkeypatch,
):
    """What a layer evaluates is the stand-in of a Metal that is not proven:
    the smallest launch that MLX builds the same Metal for, over zeros in
    the place of x, of the indices and of the layer's tensors. The proof
    evaluates the zeros, then the launch. Nothing of the call is evaluated:
    not x, not a tensor, not the result."""
    quantized, layer = _pair(320, 72)
    switched, switch = _switch_pair(320, 72)
    # Rows that no group of 2 to 4 divides: the one-row launch.
    x, more = _x((7, 320)), _x((5, 320))
    rows, indices = _x((16, 1, 320)), _experts((16,))
    tiles = []
    launched = _launched(monkeypatch, tiles)
    evaluated = _evaluations(monkeypatch)
    y = layer(x)
    zeros, (proving,) = evaluated
    assert proving is not y and proving.shape == (72,)
    assert proving.dtype == mx.bfloat16 and _zeros(proving[..., :8]) and not _zeros(y)
    # The stand-in's launch, then the call's. A matrix's stand-in is one
    # vector over the whole of each tensor: what follows the rows in a
    # tensor is the format's own.
    (vector, tensors, _), (called, _, _) = launched
    assert vector.shape == (320,) and _like([vector], [x[0]])
    assert _like(tensors, [layer[name] for name in _NAMES])
    assert called.shape == (7, 320)
    assert _same([vector, *tensors], zeros)
    # The stand-in computes the first tile of 8 outputs (the rest of its
    # result is never written): the grid is no part of the Metal. The call,
    # all 9 of them.
    assert tiles == [1, 9]
    del evaluated[:], launched[:], tiles[:]
    layer(x), layer(more)
    assert not evaluated and [x.shape for x, _, _ in launched] == [(5, 320)]
    del launched[:], tiles[:]
    y = switch(rows, indices)
    zeros, (proving,) = evaluated
    assert proving is not y and proving.shape == (8, 72)
    assert proving.dtype == mx.bfloat16 and _zeros(proving[..., :8]) and not _zeros(y)
    # 16 pairs, each with its row, have the Metal of 8: the fewest indices
    # that MLX binds as a buffer. One matrix of the stack keeps each of its
    # tensors one.
    (pairs, tensors, rhs), (called, _, _) = launched
    assert _like([pairs, rhs], [rows[:8, 0], indices[:8]])
    assert _like(tensors, [switch[name][:1] for name in _NAMES])
    assert called.shape == (16, 1, 320)
    assert _same([pairs, *tensors, rhs], zeros)
    assert tiles == [1, 9]
    del evaluated[:], launched[:]
    switch(rows, indices), switch(rows[4:12] + 0, indices[4:12] + 0)
    assert not evaluated and [x.shape for x, _, _ in launched] == [(8, 1, 320)]
    # Another launch: one row for every pair.
    del launched[:]
    switch(rows[:1], indices)
    zeros, (proving,) = evaluated
    assert proving.shape == (16, 72) and _zeros(proving[..., :8])
    (row, tensors, rhs), _ = launched
    assert _like([row, rhs], [rows[0], indices])
    assert _like(tensors, [switch[name][:1] for name in _NAMES])
    assert _same([row, *tensors, rhs], zeros)
    # What a face answers for, so that zeros prove it (a level table's index
    # masked to the table is allowed).
    contract = (
        "- A face is safe over tensors and activations of any value, zeros "
        "included, on which a kernel proves its Metal: every address that "
        "`row`, `load`, `madd` or `finish` reads stays inside the arrays the "
        "face was given (an index read from a tensor, such as a code that "
        "picks an entry of a level table, is masked to the size of what it "
        "indexes; the constructor and `step()` form addresses and read none, "
        "next rule), and no loop bound or exit depends on a value that it "
        "reads.\n"
    )
    assert contract in _PAGE


def _gain(width=2):
    """The toy face with one more tensor, of ``width`` floats a matrix: the
    second is a gain of all the matrix's rows (40 of them)."""
    scales, lane = "  const device float* scales;\n", "    const device float* s;\n"
    source = _TOY
    for old, new in (
        (scales, scales + "  const device float* gains;\n"),
        (lane, lane + "    const device float* g;\n"),
        (
            "s(m.scales + row) {}",
            f"s(m.scales + row), g(m.gains + row / 40 * {width}) {{}}",
        ),
        ("return sum * s[r];", "return sum * s[r] * g[1];"),
    ):
        assert source.count(old) == 1
        source = source.replace(old, new)
    return replace(_toy(f"gain{width}", source), tensors=("scales", "gains"))


class _GainFormat(_ToyFormat):
    """Its layers run through the face, or fail the test."""

    mode, tensors = "gain", ("scales", "gains")

    def decode(self, tensors, bits, group_size):
        pytest.fail("decoded")


def _gain_stack(monkeypatch, width=2):
    """A stack of the gain format, whose smallest tensor has ``width``
    elements a matrix."""
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "gain", _GainFormat(_gain(width)))
    (codes, scales), _, _ = _matrix(EXPERTS * 40, 320)
    tensors = {
        "weight": codes.reshape(EXPERTS, 40, 320),
        "scales": scales.reshape(EXPERTS, 40),
        "gains": mx.array(
            [[3.0, 2.0 ** (e % 3 - 1)] + [5.0] * (width - 2) for e in range(EXPERTS)]
        ),
    }
    return FormatSwitchLinear("gain", 8, 1, 320, tensors)


def _gained(layer, x, indices):
    """NumPy's products of ``x``'s rows with a gain stack's matrices at
    ``indices``."""
    gain = np.array(layer["gains"])[np.array(indices)][..., 1]
    return _want(layer, x, indices) * gain[..., None, None]


def _sources(monkeypatch, capfd):
    """Has MLX print the Metal that it generates for each launch that is
    built. Gives what reads the sources printed since it last read, in the
    order of the launches."""
    rows, shared = linear._rows, linear._shared_rows

    def printed(kernel, *launch):
        return rows(lambda **named: kernel(**named, verbose=True), *launch)

    def shared_printed(kernel, *launch):
        return shared(lambda **named: kernel(**named, verbose=True), *launch)

    monkeypatch.setattr(linear, "_rows", printed)
    monkeypatch.setattr(linear, "_shared_rows", shared_printed)

    def read():
        out = capfd.readouterr().out
        opening = r"^Generated source code for `\w+`:\n```\n"
        return re.findall(opening + r"(.*?)\n```$", out, re.DOTALL | re.MULTILINE)

    return read


def test_a_stand_in_has_the_metal_of_the_launch_it_proves(toy, monkeypatch, capfd):
    """MLX writes a launch's Metal from the kernel's source, the template's
    values and, input by input, the dtype, whether the input has fewer than
    8 elements and whether it has no axis. A stand-in is the smallest launch
    of which all of that is the call's: MLX generates the same source for
    the two, byte for byte. Every launch has the pairs its x, its indices
    and its output hold (``_launched``)."""
    toy(_toy("linear"))
    _launched(monkeypatch)
    sources = _sources(monkeypatch, capfd)

    def matrix():
        return _pair(320, 72)[1]

    def halves():
        return _pair(512, 40, 5, mx.float16)[1]

    def stack():
        return _switch_pair(320, 72)[1]

    def toy_matrix():
        return _toy_linear(40, 320)[0]

    def toy_stack():
        return _toy_switch(40, 320)

    def gain_stack():
        return _gain_stack(monkeypatch)

    rows = ((320,), (1, 320), (3, 320), (2, 3, 320), (MAX_ROWS, 320))
    calls = [(matrix, (_x(shape),)) for shape in rows]
    calls += [(halves, (_x((2, 512), mx.float16),))]
    calls += [
        (stack, (_x((*lead_m, 320)), _experts(routed))) for lead_m, routed, _ in SHAPES
    ]
    # Fewer than 8 pairs, which MLX binds as constants, and 8; an index of
    # no axis.
    calls += [(stack, (_x((n, 1, 320)), _experts((n,)))) for n in (1, 7, 8)]
    index = mx.array(4, dtype=mx.uint32)
    calls += [(stack, (_x((n, 320)), index)) for n in (1, 3)]
    calls += [(toy_matrix, (_ints((2, 320)),))]
    calls += [(toy_stack, (_ints((2, 4, 1, 1, 320)), _experts((2, 4, 3))))]
    # A stack whose smallest tensor has fewer than 8 elements a matrix.
    calls += [(gain_stack, (_ints((n, 1, 320)), _experts((n,)))) for n in (3, 16)]
    generated = set()
    for make, arrays in calls:
        kernels._PROOFS.clear()
        layer = make()
        mx.eval(layer(*arrays))
        # The stand-in's launch, then the call's.
        stand_in, called = sources()
        assert stand_in == called
        (proof,) = kernels._PROOFS.values()
        assert len(proof.ran) == 1 and not proof.failed
        generated.add(called)
    # Both ways MLX binds a stack's indices are among them.
    for binding in ("const constant uint32_t* rhs", "const device uint32_t* rhs"):
        assert any(binding in source for source in generated)
    # A matrix's rows in groups of 3 and of 4 are two Metals of their own.
    assert len(generated) == 15


@pytest.mark.parametrize("width,matrices", [(2, 4), (7, 2)])
def test_a_stack_s_stand_in_has_the_matrices_that_keep_each_tensor_a_buffer(
    monkeypatch, caplog, width, matrices
):
    """MLX binds a tensor of fewer than 8 elements as constants: other
    Metal, which a face does not compile with. A stack's stand-in has as
    many matrices as bring its smallest tensor to 8 elements: 4, of a
    tensor of 2 floats a matrix; 2, of 7."""
    caplog.set_level(logging.INFO, logger=linear.__name__)
    layer = _gain_stack(monkeypatch, width)
    assert layer["gains"].shape == (EXPERTS, width)
    launched = _launched(monkeypatch)
    x, indices = _ints((4, 1, 1, 320)), _experts((4, 3))
    got = layer(x, indices)
    assert np.array_equal(np.array(got), _gained(layer, x, indices))
    (_, tensors, _), _ = launched
    assert [a.shape for a in tensors] == [
        (matrices, 40, 320),
        (matrices, 40),
        (matrices, width),
    ]
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {(mx.float32, 3, False)} and not proof.failed
    assert [r.getMessage() for r in _logged(caplog, linear)] == [
        f"quant format face toy_gain{width}: rows kernel engaged"
    ]


def _built(monkeypatch):
    """The shapes of x that the rows kernel's launches are built for, as
    they are built: once for each graph that MLX's compile traces."""
    rows, shared, built = linear._rows, linear._shared_rows, []

    def recorded(*launch):
        built.append(launch[3].shape)
        return rows(*launch)

    def shared_recorded(*launch):
        built.append(launch[4].shape)
        return shared(*launch)

    monkeypatch.setattr(linear, "_rows", recorded)
    monkeypatch.setattr(linear, "_shared_rows", shared_recorded)
    return built


def test_a_launch_is_built_once_and_replayed(monkeypatch):
    """MLX writes a kernel's source out for every launch that is built. A
    layer builds one for its arrays' shapes and types, and replays it."""
    quantized, layer = _pair(320, 72, bias=True)
    built = _built(monkeypatch)
    for seed in range(3):
        x = _x((1, 320), seed=seed)
        assert _same_bits(layer(x), quantized(x))
    # The Metal's stand-in, a vector of zeros, then the call's launch.
    assert built == [(320,), (1, 320)]
    # Other rows are another graph: of the Metal that is proven, or of a
    # Metal of its own, a group of rows, proven on a stand-in of the group.
    for seed in range(2):
        x = _x((5, 320), seed=seed)
        alone = [quantized(x[row : row + 1]) for row in range(5)]
        assert _same_bits(layer(x), mx.concatenate(alone))
    assert built == [(320,), (1, 320), (5, 320)]
    for seed in range(2):
        x = _x((3, 320), seed=seed)
        alone = [quantized(x[row : row + 1]) for row in range(3)]
        assert _same_bits(layer(x), mx.concatenate(alone))
    assert built == [(320,), (1, 320), (5, 320), (3, 320), (3, 320)]
    # The graphs are the proof's: another layer of the face, the tensors'
    # types and the shape replays them over its own tensors.
    other_quantized, other = _pair(320, 72, seed=7)
    x = _x((1, 320), seed=9)
    assert _same_bits(other(x), other_quantized(x))
    assert not _same_bits(other(x), layer(x))
    assert built == [(320,), (1, 320), (5, 320), (3, 320), (3, 320)]
    (proof,) = kernels._PROOFS.values()
    shapes = {(1, 320), (5, 320), (3, 320)}
    assert set(proof.graphs) == {(shape, mx.bfloat16) for shape in shapes}
    assert proof.ran == {mx.bfloat16, (mx.bfloat16, 3)}


def test_a_stack_s_launch_is_built_once_and_replayed(monkeypatch):
    switched, switch = _switch_pair(320, 72, bias=True)
    built = _built(monkeypatch)
    # A decode step's tokens with their top-k, then rows of x for each index.
    shapes = (((1, 4, 1, 1, 320), (1, 4, 3)), ((5, 3, 320), (5,)))
    for lead_m_k, routed in shapes:
        for seed in range(3):
            x, indices = _x(lead_m_k, seed=seed), _experts(routed, seed=seed)
            assert _same_bits(switch(x, indices), switched(x, indices))
    # Each of the two Metals is proven on its stand-in: three pairs of a
    # row are 3 rows of zeros, a pair of each row is 8.
    stand_in, alone = (3, 320), (8, 320)
    assert built == [stand_in, (1, 4, 1, 1, 320), alone, (5, 3, 320)]
    # The same x with indices of another shape is another graph.
    x, indices = _x((5, 3, 320)), _experts((2, 5))
    assert _same_bits(switch(x, indices), switched(x, indices))
    assert built[4:] == [(2, 5, 3, 320)]
    (proof,) = kernels._PROOFS.values()
    assert set(proof.graphs) == {
        ((1, 4, 1, 1, 320), mx.bfloat16, (1, 4, 3), mx.uint32),
        ((5, 3, 320), mx.bfloat16, (5,), mx.uint32),
        ((5, 3, 320), mx.bfloat16, (2, 5), mx.uint32),
    }
    # Another stack of the face and the shape, of other experts.
    others, other = _switch_pair(320, 72, seed=7)
    x, indices = _x((5, 3, 320), seed=9), _experts((5,), seed=9)
    assert _same_bits(other(x, indices), others(x, indices))
    assert len(built) == 5


def test_stacks_of_6_and_of_8_matrices_share_a_proof_and_a_graph(monkeypatch):
    """How many matrices a stack has is no part of what a launch is compiled
    from, nor of its graph's signature: a stack of 8 replays the graph that
    a stack of 6 kept, over its own tensors. MLX's compile traces the graph
    once more, for their shapes."""
    k, n = 320, 72
    six, small = _switch_pair(k, n)
    eight, large = _switch_pair(k, n, seed=7, experts=8)
    assert (small.num_experts, large.num_experts) == (6, 8)
    built = _built(monkeypatch)
    shape = (1, 4, 1, 1, k)
    x, indices = _x(shape), _experts((1, 4, 3))
    assert _same_bits(small(x, indices), six(x, indices))
    ((key, proof),) = kernels._PROOFS.items()
    ((signature, graph),) = proof.graphs.items()
    del built[:]
    # Among them, the matrices that a stack of 6 does not have.
    later = _x(shape, seed=3)
    last = mx.array([[[7, 0, 6], [6, 7, 5], [1, 7, 6], [7, 6, 2]]], dtype=mx.uint32)
    for _ in range(2):
        assert _same_bits(large(later, last), eight(later, last))
    assert built == [shape]
    # The stack of 6 still replays what was traced for it.
    assert _same_bits(small(later, indices), six(later, indices))
    assert built == [shape]
    assert list(kernels._PROOFS) == [key] and kernels._PROOFS[key] is proof
    assert list(proof.graphs) == [signature] and proof.graphs[signature] is graph
    assert proof.ran == {(mx.bfloat16, 3, False)} and not proof.failed


def test_a_stack_s_graph_is_not_replayed_for_x_of_another_dtype():
    """x's dtype is part of a stack's signature, as the shapes are. The
    affine face is not exact for float32: a call in it, of the shapes of a
    kept graph, is the decode and MLX's gather, and keeps nothing."""
    quantized, switch = _switch_pair(320, 72, bias=True)
    x, indices = _x((4, 1, 1, 320)), _experts((4, 3))
    assert _same_bits(switch(x, indices), quantized(x, indices))
    (proof,) = kernels._PROOFS.values()
    kept = dict(proof.graphs)
    assert len(kept) == 1
    wide = x.astype(mx.float32)
    stack = mx.dequantize(*_tensors(quantized).values(), group_size=32, bits=4)
    stack = stack.astype(mx.float32).swapaxes(-1, -2)
    want = mx.gather_mm(wide, stack, rhs_indices=indices, sorted_indices=False)
    want = want + mx.expand_dims(quantized["bias"][indices], -2)
    got = switch(wide, indices)
    assert got.dtype == mx.float32 and _equal(got, want)
    assert proof.graphs == kept and proof.ran == {(mx.bfloat16, 3, False)}
    assert not proof.failed
    # The kept graph is still the one of x's first dtype.
    assert _same_bits(switch(x, indices), quantized(x, indices))
    assert proof.graphs == kept


def test_a_proof_keeps_so_many_graphs_then_builds_them_as_called(monkeypatch):
    """A prefill's chunks are shapes without end."""
    monkeypatch.setattr(kernels, "MAX_GRAPHS", 2)
    quantized, layer = _pair(320, 72)
    built = _built(monkeypatch)
    # Rows that no group of 2 to 4 divides: the one-row launch's Metal.
    for rows in (1, 5, 1, 7, 1, 7):
        x = _x((rows, 320), seed=rows)
        alone = [quantized(x[row : row + 1]) for row in range(rows)]
        assert _same_bits(layer(x), mx.concatenate(alone))
    # The third drops the two that are kept, and the first is built again.
    assert built == [(320,), (1, 320), (5, 320), (7, 320), (1, 320)]
    (proof,) = kernels._PROOFS.values()
    assert set(proof.graphs) == {((7, 320), mx.bfloat16), ((1, 320), mx.bfloat16)}


def test_a_launch_that_fails_stops_the_launches_that_ran(toy, monkeypatch, caplog):
    """A proof's launches stop together: its kept graphs are dropped."""
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    floats = 'static_assert(sizeof(x[0]) == 4, "floats alone");\n      return 0;'
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
    that cannot be evaluated: the launch is proven on its Metal's stand-in,
    which reads no array of the call."""
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
    # Built for the stand-in, then for the function's x.
    assert built == [(320,), (1, 320)]
    face = formats.dot(layer, 320, 72)
    assert [r.getMessage() for r in _logged(caplog, linear)] == [
        f"quant format face {face.key}: rows kernel engaged"
    ]
    # MLX's compile runs a function as it is where its arguments belong to
    # a trace. So the kept graph is traced at its first call outside the
    # function, once, and replayed from then on. Nothing is evaluated or
    # proven again.
    evaluated = _evaluations(monkeypatch)
    for call in (later, x):
        assert _same_bits(layer(call), quantized(call))
        assert _same_bits(compiled(call), quantized(call) * 2)
    assert built == [(320,), (1, 320), (1, 320)] and not evaluated
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
    assert set(proof.graphs) == {((4, 1, 1, 320), mx.bfloat16, (4, 3), mx.uint32)}
    assert built == [(3, 320), (4, 1, 1, 320)]
    assert len(_logged(caplog, linear)) == 1
    # Outside the function: one trace of the kept graph, then its replay.
    evaluated = _evaluations(monkeypatch)
    for call in ((later, others), (rows, indices)):
        assert _same_bits(switch(*call), switched(*call))
        assert _same_bits(compiled(*call), switched(*call) * 2)
    assert built == [(3, 320), (4, 1, 1, 320), (4, 1, 1, 320)] and not evaluated
    assert len(_logged(caplog, linear)) == 1 and len(proof.graphs) == 1


def test_a_face_that_does_not_compile_in_a_function_that_mlx_compiles_is_the_decode(
    toy, monkeypatch, caplog
):
    """The stand-in fails as the launch would: the function is built over
    the decode."""
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
    # Each was built once, for its stand-in, and not for the function's
    # arrays.
    assert built == [(320,), (8, 320)]
    assert decoded == [(40, 320), (EXPERTS, 40, 320)]
    message = "quant format face toy_broken: rows kernel failed; decode fallback"
    warnings = _logged(caplog, linear, logging.WARNING)
    assert [warning.getMessage() for warning in warnings] == [message] * 2
    assert all("does not compile" in str(w.exc_info[1]) for w in warnings)
    assert len(kernels._PROOFS) == 2
    for proof in kernels._PROOFS.values():
        assert proof.failed and not proof.ran and not proof.graphs


def test_a_kept_plan_follows_tensors_a_compiled_function_writes_back():
    """A function compiled over a layer's state writes the state back after
    each call, with every leaf a new array, and not through the layer's
    __setitem__: the weight the layer keeps its plan for is then not the
    layer's, and the next call plans again. A plan kept inside the function
    over its scales is never read with stale ones."""
    quantized, layer = _pair(320, 72)
    x = _x((1, 320))
    mx.eval(layer(x))

    def doubled(x):
        layer["scales"] = layer["scales"] * 2
        return layer(x)

    step = mx.compile(doubled, inputs=layer.state, outputs=layer.state)
    for _ in range(3):
        weight = layer["weight"]
        mx.eval(step(x))
        assert layer["weight"] is not weight
        quantized.scales = quantized.scales * 2
        assert _same_bits(layer(x), quantized(x))


def test_a_launch_is_proven_and_kept_in_a_function_compiled_over_the_layer_s_state(
    monkeypatch, caplog
):
    """``mx.compile(fn, inputs=state)`` hands a layer its own tensors as
    arrays that cannot be evaluated. A proof reads none of them: the launch
    is proven on its stand-in, built over the trace's tensors and kept."""
    caplog.set_level(logging.INFO, logger=linear.__name__)
    quantized, layer = _pair(320, 72, bias=True)
    switched, switch = _switch_pair(320, 72)
    x, rows, indices = _x((1, 320)), _x((4, 1, 1, 320)), _experts((4, 3))
    built = _built(monkeypatch)
    compiled = mx.compile(lambda x: layer(x) * 2, inputs=layer.state)
    assert _same_bits(compiled(x), quantized(x) * 2)
    stacked = mx.compile(lambda x, i: switch(x, i) * 2, inputs=switch.state)
    assert _same_bits(stacked(rows, indices), switched(rows, indices) * 2)
    # Each stand-in, then the launch over the trace's tensors: not the decode.
    assert built == [(320,), (1, 320), (3, 320), (4, 1, 1, 320)]
    ran = [{mx.bfloat16}, {(mx.bfloat16, 3, False)}]
    assert [proof.ran for proof in kernels._PROOFS.values()] == ran
    for proof in kernels._PROOFS.values():
        assert len(proof.graphs) == 1 and not proof.failed
    assert len(_logged(caplog, linear)) == 2
    # The functions left the layers their own tensors.
    mx.eval(layer.parameters(), switch.parameters())
    # Outside them each kept graph is traced once (a trace runs a compiled
    # function as it is), then replayed. Nothing is evaluated or proven
    # again.
    later, others = _x((4, 1, 1, 320), seed=4), _experts((4, 3), seed=4)
    evaluated = _evaluations(monkeypatch)
    for call, pairs in ((x + 1, (later, others)), (x, (rows, indices))):
        assert _same_bits(layer(call), quantized(call))
        assert _same_bits(switch(*pairs), switched(*pairs))
        assert _same_bits(compiled(call), quantized(call) * 2)
    assert built[4:] == [(1, 320), (4, 1, 1, 320)] and not evaluated
    assert len(_logged(caplog, linear)) == 2
    for proof in kernels._PROOFS.values():
        assert len(proof.graphs) == 1 and not proof.failed


def test_a_face_that_does_not_compile_decodes_in_a_function_compiled_over_the_state(
    toy, monkeypatch, caplog
):
    """A face that fails where the layer's tensors cannot be evaluated fails
    on its stand-in, as anywhere: the function is built over the decode."""
    caplog.set_level(logging.WARNING, logger=linear.__name__)
    toy(_toy("broken", _TOY + 'static_assert(false, "does not compile");\n'))
    layer, x, want = _toy_linear(40, 320)
    switch = _toy_switch(40, 320)
    rows, indices = _ints((16, 1, 320)), _experts((16,))
    pairs = _want(switch, rows, indices)
    mx.eval(want)
    built = _built(monkeypatch)
    compiled = mx.compile(lambda x: layer(x), inputs=layer.state)
    stacked = mx.compile(lambda x, i: switch(x, i), inputs=switch.state)
    for _ in range(2):
        assert _equal(compiled(x), want)
        assert np.array_equal(np.array(stacked(rows, indices)), pairs)
    assert built == [(320,), (8, 320)]
    message = "quant format face toy_broken: rows kernel failed; decode fallback"
    warnings = _logged(caplog, linear, logging.WARNING)
    assert [warning.getMessage() for warning in warnings] == [message] * 2
    assert all("does not compile" in str(w.exc_info[1]) for w in warnings)
    assert len(kernels._PROOFS) == 2
    for proof in kernels._PROOFS.values():
        assert proof.failed and not proof.ran and not proof.graphs
    # The functions left the layers their own tensors, which a call outside
    # decodes too.
    mx.eval(layer.parameters(), switch.parameters())
    assert _equal(layer(x), want)
    assert np.array_equal(np.array(switch(rows, indices)), pairs)
    assert len(built) == 2 and len(_logged(caplog, linear)) == 2


def test_a_proven_launch_is_built_over_a_trace_s_tensors_without_a_proof(
    monkeypatch, caplog
):
    """A Metal that is proven needs no stand-in where the layer's tensors
    cannot be evaluated either: its launches are built over them, and
    kept."""
    caplog.set_level(logging.INFO, logger=linear.__name__)
    quantized, layer = _pair(320, 72, bias=True)
    switched, switch = _switch_pair(320, 72)
    x, rows, indices = _x((1, 320)), _x((4, 1, 1, 320)), _experts((4, 3))
    layer(x), switch(rows, indices)
    assert len(_logged(caplog, linear)) == 2
    # Other rows and other tokens of the two Metals.
    more, tokens, experts = _x((5, 320)), _x((8, 1, 1, 320)), _experts((8, 3))
    alone = mx.concatenate([quantized(more[row : row + 1]) for row in range(5)])
    pairs = switched(tokens, experts)
    mx.eval(alone, pairs)
    built = _built(monkeypatch)
    evaluated = _evaluations(monkeypatch)
    compiled = mx.compile(lambda x: layer(x) * 2, inputs=layer.state)
    stacked = mx.compile(lambda x, i: switch(x, i) * 2, inputs=switch.state)
    for _ in range(2):
        assert _same_bits(compiled(more), alone * 2)
        assert _same_bits(stacked(tokens, experts), pairs * 2)
    assert built == [(5, 320), (8, 1, 1, 320)] and not evaluated
    ran = [{mx.bfloat16}, {(mx.bfloat16, 3, False)}]
    assert [proof.ran for proof in kernels._PROOFS.values()] == ran
    for proof in kernels._PROOFS.values():
        assert len(proof.graphs) == 2 and not proof.failed
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
        # Threads that wait for each other fail a test: they do not hang it.
        return self.pool.submit(fn, *args).result(timeout=30)

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
    floats = 'static_assert(sizeof(x[0]) == 4, "floats alone");\n      return 0;'
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
    # Each launch after its Metal's stand-in.
    assert built == [(320,), (1, 320), (3, 320), (4, 1, 1, 320)]
    assert len(_logged(caplog, linear)) == 2
    assert first(step, stream, *steps[1]) == (True, [1, 1]) and len(built) == 4
    # The graphs are the engine's thread's: neither the thread of the loads
    # nor this one has launched.
    assert kept() == [0, 0] and load(kept) == [0, 0]
    # Another engine's thread builds its launches, keeps them and replays
    # them. Nothing is proven again: no array is evaluated, nothing logged.
    second, other = engine("mlx-engine-b")
    evaluated = _evaluations(monkeypatch)
    assert second(kept) == [0, 0]
    assert second(step, other, *steps[2]) == (True, [1, 1])
    assert built[4:] == [(1, 320), (4, 1, 1, 320)]
    assert second(step, other, *steps[3]) == (True, [1, 1]) and len(built) == 6
    # A thread's graphs end with it. The model's next engine, a thread of
    # the ident of the one that ended or of another, builds its own.
    first.end()
    third, stream = engine("mlx-engine-a")
    assert third(kept) == [0, 0]
    assert third(step, stream, *steps[4]) == (True, [1, 1]) and len(built) == 8
    assert third(step, stream, *steps[5]) == (True, [1, 1]) and len(built) == 8
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
            # decode, while this one is proven.
            assert _equal(engine(_called, layer, half), decoded)
        return rows(*launch)

    monkeypatch.setattr(linear, "_rows", meanwhile)
    assert _equal(layer(x), want)
    # This thread's stand-in, the other's, which fails, then the call's
    # launch, of the Metal that this thread proved.
    assert launched == [mx.float32, mx.float16, mx.float32]
    (proof,) = kernels._PROOFS.values()
    assert proof.failed and proof.ran == {mx.float32}
    assert not proof.graphs and engine(lambda: len(proof.graphs)) == 0
    # The calls after are the decode's, on either thread.
    monkeypatch.setattr(linear, "_rows", lambda *launch: pytest.fail("built"))
    assert _equal(layer(x), want) and _equal(engine(_called, layer, x), want)


def test_two_threads_that_meet_an_unproven_launch_each_prove_it(
    toy, threads, monkeypatch, caplog
):
    """No thread waits for another's proof. Two that call a layer first at
    the same moment each build and evaluate the stand-in, of Metal that MLX
    has not built yet: both have the format's products, the Metal is marked
    once, and each thread keeps its own graph."""
    caplog.set_level(logging.INFO, logger=linear.__name__)
    toy(_toy("together"))
    layer, x, want = _toy_linear(40, 320)
    x, want = x[None], want[None]
    mx.eval(layer.parameters(), x, want)
    rows, together, proving = linear._rows, threading.Barrier(2, timeout=10), []

    def met(*launch):
        if launch[3].shape == (320,):
            # A stand-in: each thread's proof waits here for the other's.
            proving.append(threading.get_ident())
            together.wait()
        return rows(*launch)

    monkeypatch.setattr(linear, "_rows", met)
    engines = [threads("mlx-engine-a"), threads("mlx-engine-b")]
    calls = [engine.pool.submit(_called, layer, x) for engine in engines]
    for call in calls:
        assert _equal(call.result(timeout=30), want)
    assert sorted(proving) == sorted(engine.ident for engine in engines)
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {mx.float32} and not proof.failed
    assert [engine(lambda: len(proof.graphs)) for engine in engines] == [1, 1]
    engaged = {r.getMessage() for r in _logged(caplog, linear)}
    assert engaged == {"quant format face toy_together: rows kernel engaged"}


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
    for x in (_x((1, 64)), mx.array(1.0, dtype=mx.bfloat16)):  # x of no axis too
        refused = re.escape(f"x is (..., 320), not of shape {x.shape}")
        with pytest.raises(ValueError, match=refused):
            layer(x)
    _, switch = _switch_pair(320, 72)
    refused = re.escape("x is (..., M, 320), not of shape (320,)")
    with pytest.raises(ValueError, match=refused):
        switch(_x((320,)), _experts((3,)))
    refused = re.escape("x is (..., M, 320), not of shape (3, 1, 64)")
    with pytest.raises(ValueError, match=refused):
        switch(_x((3, 1, 64)), _experts((3,)))


def test_a_matrix_of_no_input_is_its_decode(toy):
    """K = 0: no launch takes it, and x of no element has rows of none."""
    toy(_toy("linear"))
    tensors = {"weight": mx.zeros((8, 0), mx.uint8), "scales": mx.ones((8,))}
    layer = FormatLinear("toy", 8, 1, 0, tensors)
    got = layer(mx.zeros((2, 0)))
    assert got.shape == (2, 8) and not mx.any(got).item()
    assert not kernels._PROOFS


def test_a_face_named_with_no_dtype_is_exact_for_floats_alone(toy, monkeypatch):
    """A face that names no dtype is exact for float16, bfloat16 and
    float32: the rows kernel would round an integer x's rows to integers."""
    toy(_toy("linear"))
    layer, _, _ = _toy_linear(40, 320)
    plan = linear._plan(layer)
    assert plan.face.dtypes == ()
    for dtype in (mx.int32, mx.uint8, mx.bool_):
        x = mx.zeros((1, 320), dtype)
        launch = (plan, (x.shape, dtype), linear._linear, linear._linear_stand_in, x)
        assert linear._launch(*launch) is None
    (proof,) = kernels._PROOFS.values()
    assert not proof.ran and not proof.failed


def _signed(indices):
    """``indices`` with each counted from the end of a stack of EXPERTS."""
    return (indices.astype(mx.int32) - EXPERTS).astype(mx.int32)


@pytest.mark.parametrize("lead_m,routed,per_row", SHAPES)
def test_a_negative_index_counts_from_the_stack_s_end(lead_m, routed, per_row):
    """As MLX's indexing does, on every route: the rows kernel, the bias,
    the decode of the indices' matrices and of the stack."""
    _, layer = _switch_pair(320, 72, bias=True)
    x, indices = _x((*lead_m, 320)), _experts(routed)
    want = layer(x, indices)
    got = layer(x, _signed(indices))
    assert _same_bits(got, want)
    (proof,) = kernels._PROOFS.values()
    assert len(proof.graphs) == 2  # one for each dtype of the indices


def test_a_negative_index_counts_from_the_stack_s_end_in_the_decode(toy):
    toy()  # no face
    mx.random.seed(7)
    bias = mx.random.randint(-64, 64, (EXPERTS, 40)).astype(mx.float32)
    layer = _toy_switch(40, 320, bias=bias)
    # Fewer indices than matrices (their matrices are decoded), and as many
    # (the stack is, for MLX's gather_mm).
    for count in (3, 16):
        x, indices = _ints((count, 1, 320)), _experts((count,))
        want = _want(layer, x, indices)
        got = layer(x, _signed(indices))
        assert np.array_equal(np.array(got), want)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.bool_])
def test_indices_that_are_no_integers_are_refused(dtype):
    """MLX's gather refuses them; the rows kernel would truncate them. A
    launch kept for integer indices is not replayed for them."""
    _, layer = _switch_pair(320, 72)
    x, indices = _x((16, 1, 320)), _experts((16,))
    layer(x, indices)
    for asked in (indices, indices.astype(mx.int32)):
        with pytest.raises(ValueError, match=f"indices are integers, not {dtype}"):
            layer(x, asked.astype(dtype))


def test_a_launch_of_2_to_the_30_pairs_or_more_is_the_decode_s():  # noqa: N802
    """MLX's grid is int32, and a launch has 2 rows of threadgroups a pair.
    Nothing is evaluated: the arrays are never allocated."""
    k = n = 8
    x, indices = mx.zeros((1024, 1, 1, k)), mx.zeros((1, 2**20), mx.uint32)
    plan = SimpleNamespace(kernel=None, k=k, n=n)
    assert linear._switch(plan, x, indices) is None
    x = mx.zeros((1, 1, k))
    assert linear._switch(plan, x, mx.zeros((2**30,), mx.uint32)) is None
    metal, _ = linear._switch(plan, x, mx.zeros((2**30 - 1,), mx.uint32))
    assert metal == (mx.float32, 2**30 - 1, False)


def test_a_launch_s_kernel_is_named_by_its_layer_and_face(toy, monkeypatch, capfd):
    """MLX names a launch's Metal function by the kernel's name: a
    compiler's message, a GPU trace and a profile show which layer and face
    ran."""
    toy(_toy("linear"))
    rows = linear._rows

    def printed(kernel, *launch):
        return rows(lambda **named: kernel(**named, verbose=True), *launch)

    monkeypatch.setattr(linear, "_rows", printed)
    layer, x, _ = _toy_linear(40, 320)
    mx.eval(layer(x[None]))
    mx.eval(_toy_switch(40, 320)(_ints((8, 1, 320)), _experts((8,))))
    out = capfd.readouterr().out
    names = re.findall(r"^Generated source code for `(\w+)`", out, re.M)
    assert set(names) == {
        "omlx_format_linear_toy_linear",
        "omlx_format_switch_linear_toy_linear",
    }


def _wide_toy_switch(experts=128):
    """A toy stack of more matrices than an int8 index can count."""
    (codes, scales), _, _ = _matrix(experts * 8, 320)
    tensors = {
        "weight": codes.reshape(experts, 8, 320),
        "scales": scales.reshape(experts, 8),
    }
    return FormatSwitchLinear("toy", 8, 1, 320, tensors)


@pytest.mark.parametrize("face", [True, False], ids=["rows kernel", "decode"])
def test_narrow_signed_indices_count_from_the_end_of_a_wider_stack(toy, face):
    """int8 indices on a stack of 128 matrices: 128 is no int8, so the wrap
    is computed in a wider type; every index, negative or not, names the
    matrix that MLX's indexing does."""
    toy(_toy("linear") if face else None)
    layer = _wide_toy_switch()
    wrapped = mx.array([0, 1, 127, 127, 0, 5], dtype=mx.uint32)
    signed = mx.array([0, 1, 127, -1, -128, -123], dtype=mx.int8)
    # A decode step's few indices, then as many as a prefill's.
    for count in (1, 32):
        x = _ints((count * 6, 1, 320), seed=count)
        mine, want = mx.tile(signed, count), mx.tile(wrapped, count)
        got = layer(x, mine)
        assert np.array_equal(np.array(got), _want(layer, x, want))


def test_unsigned_indices_take_no_wrap(monkeypatch):
    """Only signed indices can be negative: an unsigned index's graph is
    the cast alone."""
    _, layer = _switch_pair(320, 72)
    x, indices = _x((16, 1, 320)), _experts((16,))
    where, wrapped = mx.where, []

    def recorded(*args, **named):
        wrapped.append(args)
        return where(*args, **named)

    monkeypatch.setattr(mx, "where", recorded)
    layer(x, indices)
    assert not wrapped
    layer(x, indices.astype(mx.int32))
    assert len(wrapped) == 1


def _mutated(layer, how, bias):
    """``layer`` changed through one of a dict's own mutators, which MLX's
    module does not route through ``__setitem__``."""
    if how == "setdefault":
        layer.setdefault("bias", bias)
    elif how == "popitem":
        layer.popitem()  # the bias, the last item
    elif how == "ior":
        layer |= {"scales": layer["scales"] * 2}
    elif how == "delattr":
        del layer.group_size
    return layer


@pytest.mark.parametrize("how", ["setdefault", "popitem", "ior", "delattr", "clear"])
def test_every_mutator_of_a_layer_drops_its_kept_plan(toy, monkeypatch, how):
    toy(_toy("linear"))
    planned, plan = [], linear._plan

    def recorded(layer):
        planned.append(layer)
        return plan(layer)

    monkeypatch.setattr(linear, "_plan", recorded)
    bias = mx.arange(40).astype(mx.float32)
    biased = how == "popitem"
    layer, x, want = _toy_linear(40, 320, bias=bias if biased else None)
    assert _equal(layer(x), want + bias if biased else want)
    if how == "clear":
        layer.clear()
        assert linear._KEPT not in layer.__dict__
        return
    layer = _mutated(layer, how, bias)
    assert linear._KEPT not in layer.__dict__
    got = layer(x)
    assert len(planned) == 2
    expected = {"setdefault": want + bias, "popitem": want, "ior": 2 * want}
    assert _equal(got, expected.get(how, want))


def test_a_layer_s_bias_removed_or_none_as_mlx_s_layers_take_it(toy):
    """``pop`` gives the bias back, ``del`` takes it ahead of any call, and
    a bias of None is no bias that a call silently leaves out: it raises,
    as MLX's quantized layer does."""
    toy(_toy("linear"))
    bias = mx.arange(40).astype(mx.float32)
    layer, x, want = _toy_linear(40, 320, bias=bias)
    del layer["bias"]
    assert _equal(layer(x), want)
    layer["bias"] = bias
    assert layer.pop("bias") is bias
    layer["bias"] = None
    with pytest.raises(TypeError):
        layer(x)


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


def test_a_face_with_a_table_runs_the_rows_kernels_to_the_bit(monkeypatch):
    """A 4-bit affine face that reads its weight values from a table the
    kernel stages: one row, a verify window and a switch's pairs have MLX's
    bits, as the face without a table has."""
    from tests.test_formats_tables import tabled_dot

    tabled_dot(monkeypatch)
    quantized, layer = _pair(512, 64)
    for rows in (1, 3, MAX_ROWS):
        x = _x((rows, 512))
        alone = [quantized(x[row : row + 1]) for row in range(rows)]
        assert _same_bits(layer(x), mx.concatenate(alone))
    mx.random.seed(3)
    theirs, ours = SwitchGLU(512, 64, EXPERTS), SwitchGLU(512, 64, EXPERTS)
    theirs.set_dtype(mx.bfloat16)
    for name in ("gate_proj", "up_proj", "down_proj"):
        quantized = getattr(theirs, name).to_quantized(32, 4)
        setattr(theirs, name, quantized)
        k = quantized.input_dims
        setattr(ours, name, FormatSwitchLinear("affine", 4, 32, k, _tensors(quantized)))
    x, indices = _x((1, 3, 512)), _experts((1, 3, 6))
    assert _same_bits(ours(x, indices), theirs(x, indices))
    faces = {face for _, face, *_ in kernels._PROOFS}
    assert faces and all(face.key.endswith("_lut") for face in faces)
    assert all(proof.ran and not proof.failed for proof in kernels._PROOFS.values())
