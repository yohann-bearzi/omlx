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
from mlx_vlm.models.switch_layers import SwitchGLU, SwitchLinear

import omlx.patches.qwen35_moe_gate_up as gate_up_fusion
from omlx import formats
from omlx.formats import affine, kernels
from omlx.patches import moe_verify_gather as gather
from tests.test_formats_kernels import _PAGE, _TOY, _matrix, _toy
from tests.test_qwen35_moe_routed_decode import _does_not_compile, _logged, _same_bits

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

EXPERTS, TOP_K, HIDDEN, INTER = 8, 6, 512, 64


def _experts(k, n, bits=4, bias=False, dtype=mx.bfloat16):
    linear = SwitchLinear(k, n, EXPERTS, bias=bias)
    linear.set_dtype(dtype)
    return linear.to_quantized(group_size=32, bits=bits)


def _switch(bits=4, dtype=mx.bfloat16, seed=3):
    """A fused switch: gate+up of K 512 (``qmv_fast`` at these bits), down of
    K 64 (``qmv``, one last block)."""
    mx.random.seed(seed)
    glu = SwitchGLU(HIDDEN, INTER, EXPERTS)
    glu.set_dtype(dtype)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(glu, name, getattr(glu, name).to_quantized(32, bits))
    glu.eval()
    gate_up_fusion._fuse_one(glu)
    return glu


def _block(batch, length, dtype=mx.bfloat16, seed=5):
    mx.random.seed(seed + 16 * batch + length)
    x = mx.random.normal((batch, length, HIDDEN)).astype(dtype)
    indices = mx.random.randint(0, EXPERTS, (batch, length, TOP_K)).astype(mx.uint32)
    mx.eval(x, indices)
    return x, indices


def _composed(monkeypatch, switch, x, indices):
    """The route a block takes without this kernel."""
    with monkeypatch.context() as patch:
        patch.setattr(gather, "fused_switch", lambda *block: None)
        return gate_up_fusion._fused_verify_switch(switch, x, indices)


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
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
    (((kernel, *_), proof),) = kernels._PROOFS.items()
    assert kernel == "verify_gather"
    assert proof.ran == {(rows, TOP_K, dtype, mx.uint32) for rows in (2, 3, 4, 8)}
    assert not proof.failed
    # Once per launch that the blocks build.
    assert [r.getMessage() for r in _logged(caplog, gather)] == [
        f"MoE expert-ordered verify gather engaged ({rows} rows)"
        for rows in (2, 3, 4, 8)
    ]


def test_indices_of_another_dtype_are_a_launch_of_their_own(monkeypatch):
    """The routing is a kernel input: its dtype is compiled into the launch."""
    switch = _switch()
    x, indices = _block(1, 3)
    want = gather.fused_switch(switch, x, indices)
    evaluate, evaluated = mx.eval, []
    monkeypatch.setattr(mx, "eval", lambda *a: (evaluated.append(a), evaluate(*a)))
    assert _same_bits(gather.fused_switch(switch, x, indices), want)
    assert not evaluated  # proven
    signed = indices.astype(mx.int32)
    assert _same_bits(gather.fused_switch(switch, x, signed), want)
    assert [len(arrays) for arrays in evaluated] == [2, 1]
    (proof,) = kernels._PROOFS.values()
    assert proof.ran == {(3, TOP_K, x.dtype, dtype) for dtype in (mx.uint32, mx.int32)}


def test_a_launch_its_tiles_or_rows_do_not_cover_is_refused_in_words():
    """What ``projection`` and ``fused_switch`` rule out, for a caller of the
    launch itself: outputs no tile writes, pairs that are not whole rows."""
    k, n = 320, 64
    projection = gather.projection(_experts(k, n), k)
    x = mx.random.normal((2, k)).astype(mx.bfloat16)
    rhs = mx.arange(8).astype(mx.uint32) % EXPERTS
    mx.eval(gather.gather_qmv(projection, x, rhs, 4))
    with pytest.raises(RuntimeError, match="N must be whole tiles of 8 outputs"):
        mx.eval(gather.gather_qmv(projection._replace(n=60), x, rhs, 4))
    with pytest.raises(RuntimeError, match="pairs are whole rows of LHS_DIV experts"):
        mx.eval(gather.gather_qmv(projection, x, rhs[:6], 4))


def test_a_block_outside_the_kernel_is_left_to_the_composed_route():
    switch = _switch()
    for batch, length in ((1, 1), (1, 9), (3, 3), (16, 2)):
        assert gather.fused_switch(switch, *_block(batch, length)) is None
    # Nothing is read of the switch for them.
    assert not kernels._PROOFS and "_omlx_verify_gather_plan" not in vars(switch)

    x, indices = _block(1, 3)
    # The affine faces are exact for their scales' dtype.
    for dtype in (mx.float16, mx.float32):
        assert gather.fused_switch(switch, x.astype(dtype), indices) is None
    (proof,) = kernels._PROOFS.values()
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
    assert not any(proof.failed for proof in kernels._PROOFS.values())
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
    (proof,) = kernels._PROOFS.values()
    assert len(proof.ran) == gather.MAX_ROWS - 1 and not proof.failed


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
    (proof,) = kernels._PROOFS.values()
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
    assert [proof.failed for proof in kernels._PROOFS.values()] == [True, False, False]
    assert len(_logged(caplog, gather)) == 1


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
    assert [proof.failed for proof in kernels._PROOFS.values()] == [True, True, False]


def test_a_failure_ahead_of_the_block_is_raised_and_is_not_its_launches(caplog):
    caplog.set_level(logging.WARNING, logger=gather.__name__)
    switch = _switch()
    x, indices = _block(1, 3)
    with pytest.raises(RuntimeError, match="Unable to build metal library"):
        gather.fused_switch(switch, _does_not_compile(x), indices)
    with pytest.raises(RuntimeError, match="Unable to build metal library"):
        gather.fused_switch(switch, x, _does_not_compile(indices))
    (proof,) = kernels._PROOFS.values()
    assert not proof.failed and not proof.ran
    assert not _logged(caplog, gather)
    assert gather.fused_switch(switch, x, indices) is not None


def test_only_an_unproven_launch_is_evaluated_in_the_call(monkeypatch):
    switch = _switch()
    x, indices = _block(1, 3)
    shorter = _block(1, 2)
    evaluate, evaluated = mx.eval, []

    def recorded(*arrays):
        evaluated.append(arrays)
        evaluate(*arrays)

    monkeypatch.setattr(mx, "eval", recorded)
    y = gather.fused_switch(switch, x, indices)
    # The block's inputs first, outside the launches' guard, then the block.
    assert [len(arrays) for arrays in evaluated] == [2, 1]
    assert evaluated[0][0] is x and evaluated[0][1] is indices
    assert evaluated[1][0].shape == (3 * TOP_K, HIDDEN)
    del evaluated[:]
    assert _same_bits(gather.fused_switch(switch, x, indices), y)
    assert not evaluated
    # Another launch: its rows are not these.
    gather.fused_switch(switch, *shorter)
    assert [len(arrays) for arrays in evaluated] == [2, 1]
