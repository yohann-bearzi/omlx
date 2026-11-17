# SPDX-License-Identifier: Apache-2.0
"""The expert-ordered verify gather over the dot faces of a block's weights:
the affine face gives the composed route's bits, a format oMLX does not know
runs the same kernel, and a face that does not compile costs its blocks only
the fused route.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.switch_layers import SwitchGLU, SwitchLinear

import omlx.patches.qwen35_moe_gate_up as gate_up_fusion
from omlx import formats
from omlx.formats import Dot, affine, kernels
from omlx.patches import moe_verify_gather as gather
from tests.test_formats_kernels import _PAGE, _TOY, _matrix, _toy
from tests.test_qwen35_moe_routed_decode import _does_not_compile, _logged, _same_bits

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

EXPERTS, TOP_K, HIDDEN, INTER = 8, 6, 512, 64


def _experts(k, n, bits=4, bias=False, dtype=mx.bfloat16):
    linear = SwitchLinear(k, n, EXPERTS, bias=bias)
    linear.set_dtype(dtype)
    return linear.to_quantized(group_size=32, bits=bits)


def _switch(
    bits=4, dtype=mx.bfloat16, seed=3, group=32, inter=INTER, down_bits=None, experts=8
):
    """A fused switch: gate+up of K 512 (``qmv_fast`` at these bits), down of
    K ``inter`` (``qmv``, one last block at 64), at ``down_bits`` if given."""
    mx.random.seed(seed)
    glu = SwitchGLU(HIDDEN, inter, experts)
    glu.set_dtype(dtype)
    widths = {"gate_proj": bits, "up_proj": bits, "down_proj": down_bits or bits}
    for name, width in widths.items():
        setattr(glu, name, getattr(glu, name).to_quantized(group, width))
    glu.eval()
    gate_up_fusion._fuse_one(glu)
    return glu


def _block(batch, length, dtype=mx.bfloat16, seed=5):
    mx.random.seed(seed + 16 * batch + length)
    x = mx.random.normal((batch, length, HIDDEN)).astype(dtype)
    indices = mx.random.randint(0, EXPERTS, (batch, length, TOP_K)).astype(mx.uint32)
    mx.eval(x, indices)
    return x, indices


def _proofs():
    """The gather's proofs, among the kernels'."""
    return [
        proof
        for (kernel, *_), proof in kernels._PROOFS.items()
        if kernel == "verify_gather"
    ]


def _composed(monkeypatch, switch, x, indices):
    """The route a block takes without this kernel."""
    with monkeypatch.context() as patch:
        patch.setattr(gather, "fused_switch", lambda *block: None)
        return gate_up_fusion._fused_verify_switch(switch, x, indices)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_a_projection_is_the_layer_s_dot_face_and_tensors(bits):
    # K = 1024 takes the qmv_fast traversal, K = 320 the qmv one.
    for k, n, fast in ((1024, 64, True), (320, 1024, False)):
        linear = _experts(k, n, bits)
        projection = gather.projection(linear, k)
        assert projection.face is affine.face(bits, 32, fast, mx.bfloat16)
        assert projection.n == n
        names = ("weight", "scales", "biases")
        assert len(projection.operands) == len(names)
        for operand, name in zip(projection.operands, names, strict=True):
            assert operand is linear[name]
        assert projection.kernel is gather.projection(linear, k).kernel


def test_no_projection_outside_the_kernel():
    k, n = 320, 64
    assert gather.projection(_experts(k, n), k) is not None
    assert gather.projection(None, k) is None
    assert gather.projection(_experts(k, n, bias=True), k) is None
    # Not whole tiles of 8 outputs.
    assert gather.projection(_experts(k, 60), k) is None
    # One matrix, not stacked experts: of a shape that reads as a stack's,
    # and with a face of its own.
    matrix = nn.Linear(k, 40, bias=False)
    matrix.set_dtype(mx.bfloat16)
    matrix = matrix.to_quantized(32, 4)
    assert matrix.weight.shape == (40, 40) and formats.dot(matrix, k, 40) is not None
    assert gather.projection(matrix, k) is None
    # No format.
    assert gather.projection(SwitchLinear(k, n, EXPERTS, bias=False), k) is None
    # Not the K asked for.
    assert gather.projection(_experts(k, n), 2 * k) is None


@pytest.mark.parametrize(
    "bits,dtype", [(4, mx.bfloat16), (5, mx.float16), (6, mx.bfloat16), (8, mx.float16)]
)
def test_a_block_has_the_composed_route_s_bits(monkeypatch, caplog, bits, dtype):
    caplog.set_level(logging.INFO, logger=gather.__name__)
    switch = _switch(bits, dtype)
    blocks = ((1, 2), (1, 3), (2, 2), (3, 1), (2, 4), (1, 8), (1, 3))
    for batch, length in blocks:
        x, indices = _block(batch, length, dtype)
        got = gather.fused_switch(switch, x, indices)
        assert got.shape == (batch, length, TOP_K, HIDDEN) and got.dtype == dtype
        assert _same_bits(got, _composed(monkeypatch, switch, x, indices))
    (proof,) = _proofs()
    assert proof.ran == {(rows, TOP_K, dtype, mx.uint32) for rows in (2, 3, 4, 8)}
    assert not proof.failed
    # Once per launch that the blocks build.
    assert [r.getMessage() for r in _logged(caplog, gather)] == [
        f"MoE expert-ordered verify gather engaged ({rows} rows)"
        for rows in (2, 3, 4, 8)
    ]


@pytest.mark.parametrize("group", [64, 128])
def test_a_block_of_groups_of_64_or_128_has_the_composed_route_s_bits(
    monkeypatch, group
):
    switch = _switch(group=group, inter=128)
    for batch, length in ((1, 2), (2, 4)):
        x, indices = _block(batch, length)
        got = gather.fused_switch(switch, x, indices)
        assert got is not None
        assert _same_bits(got, _composed(monkeypatch, switch, x, indices))


def test_float32_affine_experts_keep_the_composed_route():
    """The affine face is MLX's for half-precision scales: experts of
    float32 scales have none."""
    switch = _switch(dtype=mx.float32)
    x, indices = _block(1, 3, mx.float32)
    assert gather.fused_switch(switch, x, indices) is None
    assert not _proofs()


def _routed(rows, top_k, experts, seed=7):
    mx.random.seed(seed + rows * top_k)
    x = mx.random.normal((1, rows, HIDDEN)).astype(mx.bfloat16)
    indices = mx.random.randint(0, experts, (1, rows, top_k)).astype(mx.uint32)
    mx.eval(x, indices)
    return x, indices


@pytest.mark.parametrize("experts", [8, 64])
@pytest.mark.parametrize(
    "bits, down_bits, dtype, group",
    [
        (2, None, mx.bfloat16, 32),
        (3, None, mx.bfloat16, 32),
        (3, 4, mx.bfloat16, 32),
        (4, 2, mx.bfloat16, 32),
        (2, None, mx.float16, 64),
        (3, None, mx.float16, 64),
    ],
)
def test_2_and_3_bit_experts_take_the_gather_at_every_block(
    monkeypatch, bits, down_bits, dtype, group, experts
):
    """Few experts or many, 32 pairs or more: with the pairs ordered once
    the gather was faster than the composed path at every block measured
    but 16 pairs over 256 experts."""
    switch = _switch(bits, dtype, group=group, down_bits=down_bits, experts=experts)
    blocks = ((2, 6), (4, 8), (8, 4), (6, 6), (5, 8), (3, 11), (3, 16))
    for rows, top_k in blocks:
        x, indices = _routed(rows, top_k, experts)
        got = gather.fused_switch(switch, x.astype(dtype), indices)
        if experts == 64 and rows * top_k <= 16:
            assert got is None
            continue
        want = _composed(monkeypatch, switch, x.astype(dtype), indices)
        assert _same_bits(got, want), (rows, top_k)
    (proof,) = _proofs()
    assert len(proof.ran) == len(blocks) - (experts == 64) and not proof.failed


def test_a_down_stack_of_other_experts_than_the_gate_up_s_has_no_plan():
    """A pair's expert indexes both stacks."""
    switch = _switch()
    switch.down_proj = _switch(experts=16).down_proj
    assert gather.fused_switch(switch, *_block(1, 3)) is None
    assert not _proofs()


def test_4_bit_experts_take_the_gather_past_32_pairs(monkeypatch):
    switch = _switch(4)
    for rows, top_k in ((6, 6), (8, 6), (5, 8), (8, 16)):
        x, indices = _routed(rows, top_k, EXPERTS)
        got = gather.fused_switch(switch, x, indices)
        assert _same_bits(got, _composed(monkeypatch, switch, x, indices))


def test_indices_of_another_dtype_are_a_launch_of_their_own(monkeypatch):
    """The routing is a kernel input, typed in the kernel's signature: a
    launch with indices of another dtype is proven on its own."""
    switch = _switch()
    x, indices = _block(1, 3)
    want = gather.fused_switch(switch, x, indices)
    evaluate, evaluated = mx.eval, []
    monkeypatch.setattr(mx, "eval", lambda *a: (evaluated.append(a), evaluate(*a)))
    launch, read = gather.gather_qmv, []

    def built(projection, x, rhs, per_row, order):
        read.append(rhs.dtype)
        return launch(projection, x, rhs, per_row, order)

    monkeypatch.setattr(gather, "gather_qmv", built)
    assert _same_bits(gather.fused_switch(switch, x, indices), want)
    assert not evaluated  # proven
    signed = indices.astype(mx.int32)
    assert _same_bits(gather.fused_switch(switch, x, signed), want)
    inputs, (stand_in,) = evaluated
    assert inputs[0].dtype == mx.int32
    assert [a.shape for a in stand_in] == [(3 * TOP_K, 2 * INTER), (3 * TOP_K, HIDDEN)]
    # The proven call's two launches, the stand-in's, then the signed call's.
    assert read == [mx.uint32] * 2 + [mx.int32] * 4
    (proof,) = _proofs()
    assert proof.ran == {(3, TOP_K, x.dtype, dtype) for dtype in (mx.uint32, mx.int32)}


def test_an_allocation_that_fails_in_a_stand_in_is_raised_and_fails_no_proof(
    monkeypatch,
):
    """A block's stand-in inputs are evaluated by the proof ahead of its
    guard: an allocation that fails in them is the call's, raised, and the
    proof has neither failed nor proven the Metal. The next block proves
    it."""
    switch = _switch()
    x, indices = _block(1, 3)
    zeros = mx.zeros

    def unallocated(shape, dtype=mx.float32, **named):
        # Zeros of the shape and dtype asked for, over 2**60 bytes: past any
        # device's largest buffer, allocated where they are evaluated.
        huge = zeros((2**30, 2**30), mx.uint8, **named).sum()
        return mx.broadcast_to(huge.astype(dtype), shape)

    with monkeypatch.context() as patch:
        patch.setattr(mx, "zeros", unallocated)
        with pytest.raises(RuntimeError, match="allocate"):
            gather.fused_switch(switch, x, indices)
    (proof,) = _proofs()
    assert not proof.failed and not proof.ran
    want = _composed(monkeypatch, switch, x, indices)
    assert _same_bits(gather.fused_switch(switch, x, indices), want)
    assert proof.ran and not proof.failed


def test_a_stand_in_s_stack_has_the_fewest_experts_that_leave_each_tensor_8():
    """MLX binds an input of fewer than 8 elements as constants: a stand-in
    stack keeps each tensor 8 elements or more, whatever its experts hold."""
    k, n = 320, 64
    projection = gather.projection(_experts(k, n), k)
    weight, scales, biases = projection.operands
    for tensors, experts in (
        ((weight, scales, biases), 1),
        ((weight, scales[:, :1, :1]), 8),  # one value an expert
        ((weight, scales[:, :1, :3]), 3),  # three
        ((weight, scales[:, :1, :4]), 2),
    ):
        assert gather._experts(projection._replace(operands=tensors)) == experts


def test_a_launch_its_tiles_or_rows_do_not_cover_is_refused_in_words():
    """What ``projection`` and ``fused_switch`` rule out, for a caller of the
    launch itself: outputs no tile writes, pairs that are not whole rows, a
    row of no expert."""
    k, n = 320, 64
    projection = gather.projection(_experts(k, n), k)
    x = mx.random.normal((2, k)).astype(mx.bfloat16)
    rhs = mx.arange(8).astype(mx.uint32) % EXPERTS
    mx.eval(gather.gather_qmv(projection, x, rhs, 4))
    with pytest.raises(RuntimeError, match="N must be whole tiles of 8 outputs"):
        mx.eval(gather.gather_qmv(projection._replace(n=60), x, rhs, 4))
    with pytest.raises(RuntimeError, match="pairs are whole rows of LHS_DIV experts"):
        mx.eval(gather.gather_qmv(projection, x, rhs[:6], 4))
    with pytest.raises(RuntimeError, match="LHS_DIV is the experts of a row, 1 or"):
        mx.eval(gather.gather_qmv(projection, x, rhs, 0))
    # The same words from the ordered launch.
    rhs = mx.arange(24).astype(mx.uint32) % EXPERTS
    x = mx.random.normal((6, k)).astype(mx.bfloat16)
    order = gather._order(rhs)
    mx.eval(gather.gather_qmv(projection, x, rhs, 4, order))
    with pytest.raises(RuntimeError, match="N must be whole tiles of 8 outputs"):
        mx.eval(gather.gather_qmv(projection._replace(n=60), x, rhs, 4, order))
    with pytest.raises(RuntimeError, match="LHS_DIV is the experts of a row, 1 or"):
        mx.eval(gather.gather_qmv(projection, x, rhs, 0, order))


def test_a_block_outside_the_kernel_is_left_to_the_composed_route():
    switch = _switch()
    for batch, length in ((1, 1), (1, 9), (3, 3), (16, 2)):
        assert gather.fused_switch(switch, *_block(batch, length)) is None
    x, indices = _block(1, 3)
    others = (
        (x[0], indices),  # not [B, L, K]
        (x[None], indices),
        (x, indices[0]),
        (x, indices[..., 0]),
        (x, indices[None]),
        (x, indices[:, :2]),  # not one row of experts for each row of x
        (x, _block(1, 4)[1]),
        (x, mx.concatenate([indices, indices], axis=0)),
        (_block(2, 3)[0], indices),
    )
    for block in others:
        assert gather.fused_switch(switch, *block) is None
    # Nothing is read of the switch for them.
    assert not _proofs() and "_omlx_verify_gather_plan" not in vars(switch)
    # Nor is a launch built or proven for these: they are the caller's, not
    # the kernel's failures.
    for block in (
        (x, indices[..., :0]),  # no expert for a row
        (x, indices.astype(mx.float32)),  # gather_qmm reads integers only
        (x, indices.astype(mx.bfloat16)),
        (x, indices > 3),
    ):
        assert gather.fused_switch(switch, *block) is None
    (proof,) = _proofs()
    assert not proof.ran and not proof.failed

    x, indices = _block(1, 3)
    # The affine faces are exact for their scales' dtype.
    for dtype in (mx.float16, mx.float32):
        assert gather.fused_switch(switch, x.astype(dtype), indices) is None
    (proof,) = _proofs()
    assert not proof.ran and not proof.failed
    assert gather.fused_switch(switch, x, indices) is not None


def test_the_plan_follows_the_switch_s_layers_and_the_block_s_width(
    monkeypatch, caplog
):
    caplog.set_level(logging.WARNING, logger=gather.__name__)
    switch = _switch()
    x, indices = _block(1, 3)
    want = _composed(monkeypatch, switch, x, indices)
    assert _same_bits(gather.fused_switch(switch, x, indices), want)

    # Half the width: not the K of these weights.
    assert gather.fused_switch(switch, x[..., : HIDDEN // 2], indices) is None
    assert _same_bits(gather.fused_switch(switch, x, indices), want)

    down = switch.down_proj
    switch.down_proj = _experts(INTER, HIDDEN, bias=True)
    assert gather.fused_switch(switch, x, indices) is None
    switch.down_proj = down
    assert _same_bits(gather.fused_switch(switch, x, indices), want)

    # A layer's tensors are read again when they are replaced.
    half = x.astype(mx.float16)
    for layer in (switch.gate_up_proj, switch.down_proj):
        assert gather.fused_switch(switch, half, indices) is None
        layer.scales = layer.scales.astype(mx.float16)
        layer.biases = layer.biases.astype(mx.float16)
        assert gather.fused_switch(switch, x, indices) is None
    got = gather.fused_switch(switch, half, indices)
    assert _same_bits(got, _composed(monkeypatch, switch, half, indices))
    # A block that one of its two faces is not exact for was not built.
    assert not any(proof.failed for proof in _proofs())
    assert not _logged(caplog, gather)


class _ToyExperts(nn.Module):
    """Stacked expert weights of the toy format of the kernel tests."""

    mode = "toy"

    def __init__(self, experts, n, k, seed):
        super().__init__()
        (codes, scales), _, _ = _matrix(experts * n, k, seed)
        self.weight = codes.reshape(experts, n, k)
        self.scales = scales.reshape(experts, n)

    def rows(self, experts, x):
        """The dequantized products for each expert of ``experts`` with its
        row of ``x``: exact in float32."""
        weight = np.array(self.weight)[experts].astype(np.float32) - 128.0
        return (weight @ x[:, :, None])[:, :, 0] * np.array(self.scales)[experts]


class _ToySwitch(nn.Module):
    def __init__(self, k, inter, n, seed=0):
        super().__init__()
        self.gate_up_proj = _ToyExperts(6, 2 * inter, k, seed)
        self.down_proj = _ToyExperts(6, n, inter, seed + 1)

    def activation(self, up, gate):
        # Small integers again, of either array kind.
        return (up + gate) // 2048 % 17 - 8


class _ToyFormat(formats.QuantFormat):
    mode, tensors, contract = "toy", ("scales",), formats.CONTRACT

    def __init__(self, face):
        self.face = face

    def dot(self, layer, k, n):
        return self.face


def _toy_block(switch, rows, top_k=3, seed=11):
    k = switch.gate_up_proj.weight.shape[-1]
    mx.random.seed(seed + rows)
    x = mx.random.randint(-8, 9, (1, rows, k)).astype(mx.float32)
    indices = mx.random.randint(0, 6, (1, rows, top_k)).astype(mx.uint32)
    mx.eval(x, indices)
    return x, indices


@pytest.fixture
def toy(monkeypatch):
    """Registers a face as the toy format's."""
    formats.load()

    def register(face):
        monkeypatch.setitem(formats._FORMATS, "toy", _ToyFormat(face))

    return register


def test_a_format_oMLX_does_not_know_runs_the_route(toy):  # noqa: N802
    toy(_toy("gather"))
    # K = 320: one whole block of the toy face, then a last block of 8 lanes.
    switch = _ToySwitch(k=320, inter=72, n=40)
    for rows in range(2, gather.MAX_ROWS + 1):
        x, indices = _toy_block(switch, rows)
        got = gather.fused_switch(switch, x, indices)
        experts = np.array(indices).reshape(-1)
        gate_up = switch.gate_up_proj.rows(experts, np.repeat(np.array(x[0]), 3, 0))
        gate, up = np.split(gate_up, 2, axis=-1)
        want = switch.down_proj.rows(experts, switch.activation(up, gate))
        assert got.shape == (1, rows, 3, 40) and got.dtype == mx.float32
        assert np.array_equal(np.array(got).reshape(rows * 3, 40), want)
    (proof,) = _proofs()
    assert len(proof.ran) == gather.MAX_ROWS - 1 and not proof.failed


def test_a_row_of_one_expert_runs_the_route(toy):
    toy(_toy("one_expert"))
    switch = _ToySwitch(k=320, inter=72, n=40)
    x, indices = _toy_block(switch, 3, top_k=1)
    got = gather.fused_switch(switch, x, indices)
    experts = np.array(indices).reshape(-1)
    gate, up = np.split(switch.gate_up_proj.rows(experts, np.array(x[0])), 2, axis=-1)
    want = switch.down_proj.rows(experts, switch.activation(up, gate))
    assert np.array_equal(np.array(got).reshape(3, 40), want)


def test_x_that_is_not_a_float_of_the_gpu_is_left_to_the_composed_route(toy):
    """Like gather_qmm, the block takes float16, bfloat16 or float32
    activations. A face of no stated dtypes is exact for any: the others are
    the caller's, refused before the proof (a stand-in of float64 zeros
    cannot even be made on the GPU)."""
    toy(_toy("any_dtype"))
    switch = _ToySwitch(k=320, inter=72, n=40)
    x, indices = _toy_block(switch, 3)
    for dtype in (mx.int32, mx.bool_, mx.complex64):
        assert gather.fused_switch(switch, x.astype(dtype), indices) is None
    with mx.stream(mx.cpu):
        assert gather.fused_switch(switch, x.astype(mx.float64), indices) is None
    (proof,) = _proofs()
    assert not proof.ran and not proof.failed
    assert gather.fused_switch(switch, x, indices) is not None


def test_an_activation_of_another_dtype_or_shape_leaves_the_block_composed(
    toy, monkeypatch
):
    """The down launch is planned for the rows the gate and up give, and
    proven in the block's dtype."""
    toy(_toy("gather"))
    switch = _ToySwitch(k=320, inter=72, n=40)
    x, indices = _toy_block(switch, 3)
    activation, launch, launched = switch.activation, gather.gather_qmv, []
    monkeypatch.setattr(
        gather, "gather_qmv", lambda p, *a: (launched.append(p.n), launch(p, *a))[1]
    )
    others = (
        lambda up, gate: activation(up, gate).astype(mx.float16),
        lambda up, gate: activation(up, gate)[:, :64],
        # A row more than the pairs'.
        lambda up, gate: mx.pad(activation(up, gate), ((0, 1), (0, 0))),
    )

    def composed():
        for other in others:
            with monkeypatch.context() as patch:
                patch.setattr(switch, "activation", other)
                assert gather.fused_switch(switch, x, indices) is None
        # The gate and up of each block, and no down.
        assert launched == [144] * len(others)
        del launched[:]

    # The block's Metal is proven on its stand-in, which reads no
    # activation: both launches, then the block's gate and up.
    with monkeypatch.context() as patch:
        patch.setattr(switch, "activation", others[0])
        assert gather.fused_switch(switch, x, indices) is None
    (proof,) = _proofs()
    assert launched == [144, 40, 144] and len(proof.ran) == 1
    del launched[:]
    composed()
    assert gather.fused_switch(switch, x, indices) is not None
    assert launched == [144, 40] and len(proof.ran) == 1 and not proof.failed


def test_an_activation_that_raises_raises_in_the_call(toy):
    """The activation is the caller's: what it raises is raised, the
    proof stands, and the next block runs the kernel."""
    toy(_toy("gather"))
    switch = _ToySwitch(k=320, inter=72, n=40)
    x, indices = _toy_block(switch, 3)
    activation = switch.activation

    def raises(up, gate):
        raise ArithmeticError("not this block")

    for _ in range(2):  # unproven, then proven
        switch.activation = raises
        with pytest.raises(ArithmeticError, match="not this block"):
            gather.fused_switch(switch, x, indices)
        (proof,) = _proofs()
        assert len(proof.ran) == 1 and not proof.failed
        switch.activation = activation
        assert gather.fused_switch(switch, x, indices) is not None


def test_a_face_that_does_not_compile_keeps_its_blocks_composed(
    toy, monkeypatch, caplog
):
    """MLX compiles a kernel at its first evaluation. A face that fails
    there fails inside the call that can fall back."""
    caplog.set_level(logging.WARNING, logger=gather.__name__)
    toy(_toy("broken", _TOY + 'static_assert(false, "does not compile");\n'))
    switch = _ToySwitch(k=320, inter=72, n=40)
    x, indices = _toy_block(switch, 3)
    assert gather.fused_switch(switch, x, indices) is None
    (warning,) = _logged(caplog, gather, logging.WARNING)
    message = "expert-ordered verify gather failed; composed fallback"
    assert warning.getMessage() == message and f"(`{message}`)" in _PAGE
    assert "does not compile" in str(warning.exc_info[1])
    (proof,) = _proofs()
    assert proof.failed and not proof.ran

    # Not built again: for these blocks, or for others of the same faces,
    # tensor dtypes and shapes.
    other = _ToySwitch(k=320, inter=72, n=40, seed=2)
    with monkeypatch.context() as patch:
        patch.setattr(gather, "gather_qmv", lambda *launch: pytest.fail("built"))
        assert gather.fused_switch(switch, x, indices) is None
        assert gather.fused_switch(other, *_toy_block(other, 2)) is None
    assert len(_logged(caplog, gather, logging.WARNING)) == 1

    # Blocks of other faces are not affected: of the same shapes, or others.
    toy(_toy("gather"))
    compiles = _ToySwitch(k=320, inter=72, n=40)
    assert gather.fused_switch(compiles, *_toy_block(compiles, 3)) is not None
    assert gather.fused_switch(_switch(), *_block(1, 3)) is not None
    assert [proof.failed for proof in _proofs()] == [True, False, False]
    assert len(_logged(caplog, gather)) == 1


def test_a_proof_that_fails_after_a_block_ran_keeps_that_block_composed():
    """Another thread's launch may fail the proof after this block's Metal
    ran: every block of the proof keeps its composed path from then on."""
    switch = _switch()
    x, indices = _block(1, 3)
    assert gather.fused_switch(switch, x, indices) is not None
    (proof,) = _proofs()
    proof.fail()
    assert gather.fused_switch(switch, x, indices) is None


class _TwoTensors(_ToyExperts):
    """The toy experts with a second tensor that its face reads: ones."""

    def __init__(self, experts, n, k, seed):
        super().__init__(experts, n, k, seed)
        self.unit = mx.ones_like(self.scales)


class _ByTensors(formats.QuantFormat):
    mode, tensors, contract = "toy", ("scales", "unit"), formats.CONTRACT

    def __init__(self, one, two):
        self.one, self.two = one, two

    def dot(self, layer, k, n):
        return self.two if "unit" in layer else self.one


def test_projections_of_different_tensor_counts_take_their_own(monkeypatch):
    """A block's stand-in and launches split the stacks' operands at the
    gate+up's count: a down face of two tensors under a gate+up of one."""
    formats.load()
    source = _TOY.replace("KEY", "two_tensors").replace(
        "  const device float* scales;\n",
        "  const device float* scales;\n  const device float* unit;\n",
    )
    two = Dot(
        key="toy_two_tensors",
        source=source,
        type="omlx_toy_two_tensors::Dot",
        tensors=("scales", "unit"),
    )
    monkeypatch.setitem(formats._FORMATS, "toy", _ByTensors(_toy("gather"), two))
    switch = _ToySwitch(k=320, inter=72, n=40)
    switch.down_proj = _TwoTensors(6, 40, 72, 1)
    x, indices = _toy_block(switch, 3)
    got = gather.fused_switch(switch, x, indices)
    experts = np.array(indices).reshape(-1)
    gate_up = switch.gate_up_proj.rows(experts, np.repeat(np.array(x[0]), 3, 0))
    gate, up = np.split(gate_up, 2, axis=-1)
    want = switch.down_proj.rows(experts, switch.activation(up, gate))
    assert np.array_equal(np.array(got).reshape(9, 40), want)
    assert [proof.failed for proof in _proofs()] == [False]


def test_a_failed_block_keeps_only_its_tensor_dtypes_and_shapes_composed(toy, caplog):
    caplog.set_level(logging.WARNING, logger=gather.__name__)
    toy(_toy("gather"))

    def runs(switch):
        return gather.fused_switch(switch, *_toy_block(switch, 3)) is not None

    # Scales of a dtype that the face's struct does not take.
    halves = _ToySwitch(k=320, inter=72, n=40)
    halves.down_proj.scales = halves.down_proj.scales.astype(mx.float16)
    # A K that is not whole lanes of the face.
    ragged = _ToySwitch(k=324, inter=72, n=40)
    assert not runs(halves) and not runs(ragged)
    assert len(_logged(caplog, gather, logging.WARNING)) == 2
    assert runs(_ToySwitch(k=320, inter=72, n=40))
    assert [proof.failed for proof in _proofs()] == [True, True, False]


def test_a_failure_in_the_block_s_arrays_is_raised_where_they_are_evaluated(caplog):
    caplog.set_level(logging.WARNING, logger=gather.__name__)
    switch = _switch()
    x, indices = _block(1, 3)
    for block in ((_does_not_compile(x), indices), (x, _does_not_compile(indices))):
        y = gather.fused_switch(switch, *block)
        with pytest.raises(RuntimeError, match="Unable to build metal library"):
            mx.eval(y)
    (proof,) = _proofs()
    assert not proof.failed and proof.ran == {(3, TOP_K, x.dtype, mx.uint32)}
    assert not _logged(caplog, gather)
    assert gather.fused_switch(switch, x, indices) is not None


def test_a_block_of_fewer_than_8_values_of_x_is_left_to_the_composed_route(toy):
    """MLX binds such an x as constants, and the kernel reads x from device
    memory: a face of one value a lane takes K 1, 3 and 4."""
    toy(_toy("one", source=_TOY.replace("VPT = 8;", "VPT = 1;")))
    blocks = ((3, 2, False), (1, 7, False), (4, 2, True), (3, 3, True), (1, 8, True))
    for k, rows, runs in blocks:
        switch = _ToySwitch(k=k, inter=72, n=40)
        got = gather.fused_switch(switch, *_toy_block(switch, rows))
        assert (got is not None) == runs, (k, rows)
    assert sorted(len(proof.ran) for proof in _proofs()) == [1, 1, 1]
    assert not any(proof.failed for proof in _proofs())


def test_only_a_stand_in_is_evaluated_in_the_call(monkeypatch):
    """A Metal is proven on zeros: the block's rows and pairs, every pair
    with expert 0 of one expert's tensors, in the block's dtype. Nothing of
    the call is evaluated."""
    switch = _switch(dtype=mx.float16)
    x, indices = _block(1, 3, mx.float16)
    shorter = _block(1, 2, mx.float16)
    evaluate, evaluated = mx.eval, []

    def recorded(*arrays):
        evaluated.append(arrays)
        evaluate(*arrays)

    launch, launched = gather.gather_qmv, []

    def built(projection, x, rhs, per_row, order):
        launched.append((projection.operands, x, rhs, per_row))
        return launch(projection, x, rhs, per_row, order)

    monkeypatch.setattr(mx, "eval", recorded)
    monkeypatch.setattr(gather, "gather_qmv", built)
    y = gather.fused_switch(switch, x, indices)
    # The proof evaluates the stand-in's inputs, then its launches.
    inputs, (stand_in,) = evaluated
    called = (x, indices, *(a for _, a in tree_flatten(switch.parameters())))
    assert not any(a is b for a in inputs for b in called)
    assert [a.shape for a in stand_in] == [(3 * TOP_K, 2 * INTER), (3 * TOP_K, HIDDEN)]
    zeros = launched[:2]
    assert [(len(z[2]), z[3]) for z in zeros] == [(3 * TOP_K, TOP_K), (3 * TOP_K, 1)]
    for operands, x_in, rhs, _ in zeros:
        assert [a.shape[0] for a in operands] == [1, 1, 1]
        assert rhs.dtype == mx.uint32
        for a in (*operands, x_in, rhs):
            assert not mx.any(a).item()
    assert zeros[0][1].shape == (3, HIDDEN) and zeros[1][1].shape == (3 * TOP_K, INTER)
    assert zeros[0][1].dtype == zeros[1][1].dtype == mx.float16
    # Then the block's own launches, built and not evaluated.
    assert launched[2][1].shape == (3, HIDDEN) and launched[3][3] == 1
    assert len(launched) == 4 and len(evaluated) == 2
    del evaluated[:], launched[:]
    assert _same_bits(gather.fused_switch(switch, x, indices), y)
    assert not evaluated and len(launched) == 2
    # Another Metal: its rows are not these.
    gather.fused_switch(switch, *shorter)
    assert len(evaluated) == 2 and evaluated[1][0][0].shape == (2 * TOP_K, 2 * INTER)


@pytest.mark.parametrize("state", [False, True])
def test_a_block_is_proven_in_a_function_that_mlx_compiles(monkeypatch, state):
    """A block may be built in a function that MLX's compile traces, over
    an x and indices that cannot be evaluated, and, compiled over the
    model's state (``inputs=``), over tensors that cannot be either."""
    switch = _switch()
    x, indices = _block(1, 3)
    want = _composed(monkeypatch, switch, x, indices)
    compiled = mx.compile(
        lambda x, i: gather.fused_switch(switch, x, i) * 2,
        inputs=switch.state if state else None,
    )
    assert _same_bits(compiled(x, indices), want * 2)
    (proof,) = _proofs()
    assert proof.ran == {(3, TOP_K, x.dtype, mx.uint32)} and not proof.failed
    mx.eval(switch.parameters())
    assert _same_bits(gather.fused_switch(switch, x, indices), want)


def test_a_block_of_24_pairs_or_more_is_put_in_expert_order_once(monkeypatch):
    """Below 24 pairs each threadgroup finds its pair; from 24 one launch
    orders them ahead of the block's two. Either way: the composed bits."""
    switch = _switch()
    order, ordered = gather._order, []
    monkeypatch.setattr(
        gather, "_order", lambda rhs: (ordered.append(rhs.shape[0]), order(rhs))[1]
    )
    launch, launched = gather.gather_qmv, []

    def built(projection, x, rhs, per_row, order):
        launched.append((rhs.shape[0], order is not None))
        return launch(projection, x, rhs, per_row, order)

    for batch, length in ((1, 3), (1, 4), (2, 3), (1, 8)):  # 18, 24, 36, 48 pairs
        x, indices = _block(batch, length)
        with monkeypatch.context() as patch:
            patch.setattr(gather, "gather_qmv", built)
            got = gather.fused_switch(switch, x, indices)
        assert _same_bits(got, _composed(monkeypatch, switch, x, indices))
    # Each ordered Metal: its stand-in, then the call; both launches of each.
    assert ordered == [24, 24, 36, 36, 48, 48]
    assert launched == [(p, p >= 24) for p in (18, 24, 36, 48) for _ in range(4)]


@pytest.mark.parametrize("pairs", [24, 25, 80, 128])
@pytest.mark.parametrize("dtype", [mx.uint32, mx.int32, mx.uint8])
def test_the_order_is_the_pairs_stable_sort_by_expert(pairs, dtype):
    mx.random.seed(pairs)
    for experts in (2, 512):  # ties everywhere, few ties
        top = min(experts, 256) if dtype == mx.uint8 else experts
        rhs = mx.random.randint(0, top, (pairs,)).astype(dtype)
        want = np.argsort(np.array(rhs).astype(np.int64), kind="stable")
        assert np.array_equal(np.array(gather._order(rhs)), want)


def test_the_ordered_launch_reads_each_position_s_pair_from_the_order():
    """Every position named pair 1: pair 1's rows alone are written."""
    switch = _switch()
    x, indices = _block(1, 4)
    projection = gather.projection(switch.gate_up_proj, HIDDEN)
    x, rhs, n = x.reshape(4, HIDDEN), indices.reshape(-1), projection.n
    pairs = rhs.shape[0]
    y = projection.ordered(
        inputs=[x, *projection.operands, rhs, mx.ones((pairs,), mx.uint32)],
        template=[
            ("T", x.dtype),
            ("K_SIZE", HIDDEN),
            ("N_SIZE", n),
            ("LHS_DIV", TOP_K),
        ],
        grid=(32, 2 * pairs, n // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[(pairs, n)],
        output_dtypes=[x.dtype],
        init_value=float("nan"),
    )[0]
    want = gather.gather_qmv(projection, x, rhs, TOP_K)
    assert _same_bits(y[1], want[1])
    assert mx.isnan(y[0]).all().item() and mx.isnan(y[2:]).all().item()


def test_a_face_with_a_table_runs_the_gather_to_the_bit(monkeypatch):
    """A 4-bit face that reads its weight values from a table the gather's
    launches stage, ahead of the rank walk and its barrier."""
    from tests.test_formats_tables import tabled_dot

    tabled_dot(monkeypatch)
    switch = _switch(4, mx.bfloat16)
    for batch, length in ((1, 2), (2, 4), (1, 8)):
        x, indices = _block(batch, length)
        got = gather.fused_switch(switch, x, indices)
        assert _same_bits(got, _composed(monkeypatch, switch, x, indices))
    (proof,) = _proofs()
    assert proof.ran and not proof.failed
    faces = {key[1] for key in kernels._PROOFS if key[0] == "verify_gather"}
    assert all(face.key.endswith("_lut") for face in faces)


def test_the_pairs_are_ordered_from_24_and_searched_below(toy, monkeypatch):
    """20, 21, 22 and 23 pairs keep each threadgroup's search; 24 is the
    first ordered block. Either way: the pairs' own rows."""
    toy(_toy("ordered_from"))
    switch = _ToySwitch(k=320, inter=72, n=40)
    order, ordered = gather._order, []
    monkeypatch.setattr(
        gather, "_order", lambda rhs: (ordered.append(rhs.shape[0]), order(rhs))[1]
    )
    for rows, top_k in ((2, 10), (3, 7), (2, 11), (2, 12), (3, 8)):
        x, indices = _toy_block(switch, rows, top_k)
        got = gather.fused_switch(switch, x, indices)
        experts = np.array(indices).reshape(-1)
        rows_x = np.repeat(np.array(x[0]), top_k, 0)
        gate, up = np.split(switch.gate_up_proj.rows(experts, rows_x), 2, axis=-1)
        want = switch.down_proj.rows(experts, switch.activation(up, gate))
        assert np.array_equal(np.array(got).reshape(rows * top_k, 40), want)
    assert sorted(set(ordered)) == [24]


def test_a_face_s_kernels_are_built_once():
    k, n = 320, 64
    linear = _experts(k, n)
    first, second = gather.projection(linear, k), gather.projection(linear, k)
    assert first.kernel is second.kernel and first.ordered is second.ordered
    assert gather._order_kernel() is gather._order_kernel()


def test_an_ordered_block_is_proven_in_a_function_that_mlx_compiles(monkeypatch):
    switch = _switch()
    x, indices = _block(1, 4)  # 24 pairs
    want = _composed(monkeypatch, switch, x, indices)
    compiled = mx.compile(
        lambda x, i: gather.fused_switch(switch, x, i) * 2, inputs=switch.state
    )
    assert _same_bits(compiled(x, indices), want * 2)
    (proof,) = _proofs()
    assert proof.ran == {(4, TOP_K, x.dtype, mx.uint32)} and not proof.failed


def test_few_pairs_over_many_experts_keep_the_composed_path(monkeypatch):
    """16 pairs or fewer over 64 experts or more: 2 rows of a 256-expert
    top-8 block read 1.011 against the composed path. Over fewer experts,
    and from 17 pairs, the gather. The bound is part of the proof's key: a
    block of few experts proving such a Metal does not let many take it."""
    few, many = _switch(experts=63), _switch(experts=64)
    for rows, top_k, runs in ((2, 8, False), (4, 4, False), (3, 6, True), (2, 9, True)):
        x, indices = _routed(rows, top_k, 63)
        assert (gather.fused_switch(many, x, indices) is not None) == runs
        got = gather.fused_switch(few, x, indices)
        assert _same_bits(got, _composed(monkeypatch, few, x, indices))
    proofs = sorted(_proofs(), key=lambda proof: len(proof.ran))
    assert [len(proof.ran) for proof in proofs] == [2, 4]
