# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness and routing tests for the fused one-token routed experts."""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import omlx.patches.qwen35_moe_routed_decode as routed
from omlx import formats
from omlx.formats import affine

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

EXPERTS = 32


class _FakeQwen4Model:
    pass


_FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"


@pytest.fixture(autouse=True)
def _patched_block(monkeypatch):
    """Apply router + routed patches, restore the class afterwards."""
    from mlx_vlm.models.qwen3_5_moe import language as vlm_moe

    from omlx.patches.qwen35_moe_router import apply_qwen35_moe_router_patch

    cls = vlm_moe.Qwen3_5MoeSparseMoeBlock
    apply_qwen35_moe_router_patch()  # process-wide and idempotent
    assert cls._omlx_router_fused
    call = cls.__call__
    original = getattr(call, "_omlx_routed_decode_original", call)
    monkeypatch.setattr(routed, "_PROOFS", {})
    cls.__call__ = original
    if "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")
    assert routed.apply_qwen35_moe_routed_decode_patch()
    yield cls
    cls.__call__ = original
    cls._omlx_router_fused = True
    if "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")


def _block(
    hidden,
    inter,
    top_k=10,
    bits=4,
    group_size=64,
    seed=0,
    experts=EXPERTS,
    quantized_shared=True,
    dtype=mx.bfloat16,
):
    """A block laid out like Qwen3.8-Flash-Next oQ: quantized routed experts,
    8-bit shared expert (gs128 where the shape allows), 8-bit gs64
    shared-expert gate, ``dtype`` router. ``quantized_shared=False`` keeps the
    shared expert and its gate in ``dtype``."""
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(seed)
    args = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter,
        num_experts=experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(dtype)
    sm = block.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(sm, name, getattr(sm, name).to_quantized(group_size, bits))
    if quantized_shared:
        shared = block.shared_expert
        shared_gs = 128 if hidden % 128 == 0 and inter % 128 == 0 else 64
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(
                shared, name, nn.QuantizedLinear.from_linear(getattr(shared, name), shared_gs, 8)
            )
        block.shared_expert_gate = nn.QuantizedLinear.from_linear(block.shared_expert_gate, 64, 8)
    block.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("mlp.switch_mlp", sm)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    mx.eval(block.parameters())
    return block


def _pair(block, x):
    """The composed body's output and the patched call's."""
    ref = type(block).__call__._omlx_routed_decode_original(block, x)
    mx.eval(ref)
    out = block(x)
    mx.eval(out)
    return ref, out


def _engaged():
    """Whether the launches of every plan built since the test began ran."""
    proofs = routed._PROOFS.values()
    return bool(proofs) and all(
        proof.ran and not (proof.failed or proof.window_failed) for proof in proofs
    )


def _logged(caplog, module, level=logging.DEBUG):
    """What ``module`` logged. Other loggers are at whatever level the tests
    before this one left them."""
    return [
        r for r in caplog.records if r.name == module.__name__ and r.levelno >= level
    ]


def _same_bits(a, b):
    return a.shape == b.shape and mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


@pytest.mark.parametrize(
    "hidden,inter,bits,group_size",
    [
        (2560, 640, 5, 64),  # Qwen3.8-Flash-Next oQ5e
        (2560, 640, 4, 64),
        (1024, 320, 5, 32),
        (1024, 384, 6, 128),
        (1024, 320, 8, 64),
        # Gate+up takes qmv: K ends in a partial kernel block, or is whole
        # blocks of qmv's but not of qmv_fast's.
        (960, 320, 4, 64),
        (1280, 320, 5, 64),
        (896, 320, 6, 64),
        (832, 320, 8, 64),
    ],
)
def test_fused_decode_is_bit_identical(hidden, inter, bits, group_size, monkeypatch):
    calls = []
    fused = routed.routed_decode
    monkeypatch.setattr(
        routed, "routed_decode", lambda *a: calls.append(1) or fused(*a)
    )
    for seed in range(2):
        block = _block(hidden, inter, bits=bits, group_size=group_size, seed=seed)
        for step in range(6):
            x = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(mx.bfloat16)
            # The shared expert and its gate run inside the two launches.
            assert routed.routed_decode_plan(block, x).fold
            ref, out = _pair(block, x)
            assert _same_bits(ref, out)
    assert len(calls) == 12
    assert _engaged()


@pytest.mark.parametrize("experts", [EXPERTS, 128])
def test_bf16_shared_expert_stays_composed_and_bit_identical(experts):
    # 128 experts select inside the gate+up launch, 32 in the routing launch.
    # The 128-expert block keeps hidden < 16 x experts, so the fused router gemv runs.
    hidden, inter = (2560, 640) if experts == EXPERTS else (1024, 320)
    block = _block(hidden, inter, bits=5, quantized_shared=False, experts=experts)
    for step in range(4):
        x = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(mx.bfloat16)
        plan = routed.routed_decode_plan(block, x)
        assert not plan.fold
        assert routed._topk_folds(plan, block.gate(x)) == (experts == 128)
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert _engaged()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("quantized_shared", [True, False])
@pytest.mark.parametrize(
    "hidden,inter,bits,group_size",
    [(960, 320, 4, 32), (896, 384, 5, 128), (832, 320, 6, 32), (896, 384, 8, 128)],
)
def test_qmv_gate_up_is_bit_identical(
    hidden, inter, bits, group_size, quantized_shared, dtype
):
    """K ends in a partial kernel block: gate+up, the shared expert's and the
    gate row take ``qmv``."""
    block = _block(
        hidden,
        inter,
        bits=bits,
        group_size=group_size,
        quantized_shared=quantized_shared,
        dtype=dtype,
    )
    for step in range(4):
        x = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(dtype)
        assert routed.routed_decode_plan(block, x).fold == quantized_shared
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert _engaged()


@pytest.mark.parametrize("hidden", [2560, 2432])
@pytest.mark.parametrize("bits", [4, 5])
def test_fp32_kernels_match_mlx_mat_vecs(bits, hidden):
    """BF16 outputs hide one-ulp FP32 differences, so run both launches in
    FP32 against MLX's FP32 mat-vecs: the gate+up rows after SwiGLU of the
    experts and the shared expert plus the gate row, then every down row
    read through the combine (a one-hot score with the gate at sigmoid 0
    reads one expert; zero scores with the gate at sigmoid 1 the shared one).
    Hidden 2560 is whole kernel blocks (``qmv_fast``), 2432 is not (``qmv``)."""
    from mlx_vlm.models.activations import swiglu

    inter, gs, top_k, fast = 640, 64, 10, hidden == 2560
    f32 = mx.float32
    mx.random.seed(40 + bits)

    def quantized(shape, group_size, b):
        return mx.quantize(mx.random.normal(shape) * 0.05, group_size, b)

    def qmm(x, weights, group_size, b):
        return mx.quantized_matmul(x, *weights, transpose=True, group_size=group_size, bits=b)

    experts_gate_up = quantized((EXPERTS, 2 * inter, hidden), gs, bits)
    experts_down = quantized((EXPERTS, hidden, inter), gs, bits)
    shared_gate, shared_up = quantized((inter, hidden), 128, 8), quantized((inter, hidden), 128, 8)
    shared_down = quantized((hidden, inter), 128, 8)
    gate_row = quantized((1, hidden), 64, 8)
    fmt = partial(affine.face, dtype=f32)
    gate_up_kernel = routed._gate_up_kernel(
        fmt(bits, gs, fast), fmt(8, 128, fast), fmt(8, 64, fast)
    )
    down_kernel = routed._down_kernel(fmt(bits, gs, False), fmt(8, 128, False))
    for step in range(3):
        x = mx.random.normal((1, 1, hidden)) * (0.5 + step)
        ids = mx.random.permutation(EXPERTS)[:top_k].astype(mx.uint32)
        routed_gate_up = mx.gather_qmm(
            mx.expand_dims(x, (-2, -3)), *experts_gate_up, rhs_indices=ids.reshape(1, 1, top_k),
            transpose=True, group_size=gs, bits=bits, sorted_indices=False,
        )
        gate, up = mx.split(routed_gate_up, 2, axis=-1)
        ref_h = mx.concatenate([
            swiglu(gate, up).reshape(-1),
            swiglu(qmm(x, shared_gate, 128, 8), qmm(x, shared_up, 128, 8)).reshape(-1),
            qmm(x, gate_row, 64, 8).reshape(-1),
        ])
        h = gate_up_kernel(
            inputs=[x, *experts_gate_up, ids, *shared_gate, *shared_up, *gate_row],
            template=[
                ("T", f32), ("K", hidden), ("NI", inter), ("RPS", 2), ("NSG", 2), ("NS", inter),
                ("TOPK", top_k),
            ],
            grid=(32, 2 * (1 + inter // 4 + top_k * inter // 4), 1),
            threadgroup=(32, 2, 1),
            output_shapes=[ref_h.shape],
            output_dtypes=[f32],
        )[0]
        assert mx.array_equal(h.view(mx.uint32), ref_h.view(mx.uint32)).item()

        ref_down = mx.gather_qmm(
            mx.expand_dims(h[: top_k * inter].reshape(top_k, inter), -2), *experts_down,
            rhs_indices=ids, transpose=True, group_size=gs, bits=bits, sorted_indices=False,
        ).reshape(top_k, hidden)
        ref_shared = qmm(h[top_k * inter : -1], shared_down, 128, 8)
        for j in range(top_k + 1):
            scores = (mx.arange(top_k) == j).astype(f32)
            saturated_gate = mx.array([-1e30 if j < top_k else 1e30], f32)
            row = down_kernel(
                inputs=[
                    mx.concatenate([h[:-1], saturated_gate]), *experts_down, *shared_down, ids,
                    scores,
                ],
                template=[
                    ("T", f32), ("K", inter), ("N", hidden), ("RPS", routed._down_rows(1)),
                    ("KS", inter), ("NPART", top_k + 1), ("TOPK", top_k),
                ],
                grid=(32, (top_k + 1) * hidden // routed._down_rows(1), 1),
                threadgroup=(32, top_k + 1, 1),
                output_shapes=[(hidden,)],
                output_dtypes=[f32],
            )[0]
            # The +0 folds of the combine turn -0 into +0, so compare values.
            assert mx.array_equal(row, ref_down[j] if j < top_k else ref_shared).item()


@pytest.mark.parametrize("bits", [4, 5])
def test_fp32_folded_routing_matches_the_routing_launch(bits):
    """The gate+up launch that selects its rows' experts itself, in FP32 (BF16
    outputs hide one-ulp differences): per row, the selection and scores of
    the routing launch, and the gate+up rows of the launch fed with them.
    Router logits repeat, so probabilities tie."""
    from omlx.patches import qwen35_moe_router as router

    hidden, inter, gs, experts, rows, top_k = 1024, 384, 64, 128, 3, 10
    f32 = mx.float32
    mx.random.seed(50 + bits)

    def quantized(shape, group_size, b):
        return mx.quantize(mx.random.normal(shape) * 0.05, group_size, b)

    experts_gate_up = quantized((experts, 2 * inter, hidden), gs, bits)
    shared = quantized((inter, hidden), 128, 8) + quantized((inter, hidden), 128, 8)
    gate_row = quantized((1, hidden), 64, 8)
    fmt = partial(affine.face, dtype=f32)
    formats = (fmt(bits, gs, True), fmt(8, 128, True), fmt(8, 64, False))
    x = mx.random.normal((rows, hidden))
    logits = mx.random.normal((rows, experts // 2)) * 2
    logits = mx.concatenate([logits, logits[:, ::-1]], axis=-1)
    template = [
        ("T", f32), ("K", hidden), ("NI", inter), ("RPS", 2), ("NSG", 2), ("NS", inter),
        ("TOPK", top_k),
    ]
    width = top_k * inter + inter + 1
    blocks = 1 + inter // 4 + top_k * inter // 4
    h, inds, scores = routed._gate_up_topk_kernel(*formats)(
        inputs=[x, *experts_gate_up, logits, *shared, *gate_row],
        template=template + [("NE", experts), ("M", rows), ("YW", width)],
        grid=(32, 2 * blocks * rows, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, width), (rows, top_k), (rows, top_k)],
        output_dtypes=[f32, mx.uint32, f32],
    )
    mx.eval(router.softmax_topk_rows(logits.astype(mx.bfloat16), top_k))
    ref_inds, ref_scores = router._SOFTMAX_TOPK_ROWS_KERNEL(
        inputs=[logits],
        template=[("T", f32), ("NE", experts), ("K", top_k)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, f32],
    )
    assert mx.array_equal(inds, ref_inds).item()
    assert mx.array_equal(scores.view(mx.uint32), ref_scores.view(mx.uint32)).item()
    for r in range(rows):
        ref_h = routed._gate_up_kernel(*formats)(
            inputs=[x[r], *experts_gate_up, ref_inds[r], *shared, *gate_row],
            template=template,
            grid=(32, 2 * blocks, 1),
            threadgroup=(32, 2, 1),
            output_shapes=[(width,)],
            output_dtypes=[f32],
        )[0]
        assert mx.array_equal(h[r].view(mx.uint32), ref_h.view(mx.uint32)).item()


def test_experts_past_the_bound_view_are_read_from_the_stacked_weights():
    """The kernels bind a one-expert view and index the rest; a view copied
    out of the stacked buffer would read the wrong bytes for high experts."""
    from omlx.patches.qwen35_moe_router import fused_moe_combine

    block = _block(1024, 320, bits=5, experts=128)
    plan = routed.routed_decode_plan(block, mx.zeros((1, 1, 1024), mx.bfloat16))
    assert all(o.shape[0] == 1 for o in plan.gate_up_operands + plan.down_operands)
    for ids in (list(range(118, 128)), [127, 0, 116, 64, 1, 126, 3, 115, 7, 100]):
        x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
        inds = mx.array(ids, dtype=mx.uint32).reshape(1, 1, 10)
        scores = mx.softmax(mx.random.normal((1, 1, 10)), axis=-1).astype(mx.bfloat16)
        shared = block["shared_expert"](x)
        gate = block["shared_expert_gate"](x)
        ref = fused_moe_combine(block.switch_mlp(x, inds), scores, shared, gate)
        out = routed.routed_decode(plan, x, inds, scores, shared, gate)
        mx.eval(ref, out)
        assert _same_bits(ref, out)


@pytest.mark.parametrize(
    "shape,viewed",
    [
        ((4, 8, 2), True),
        ((4, 8), True),
        # MLX binds an input of fewer than 8 elements as constants.
        ((4, 7), False),
        ((4, 4, 1), False),
        ((16,), False),
        ((), False),
    ],
)
def test_an_expert_of_fewer_than_8_elements_is_bound_with_its_stack(shape, viewed):
    a = mx.zeros(shape)
    mx.eval(a)
    view = routed._expert_view(a)
    if viewed:
        assert view.shape == (1, *shape[1:])
    else:
        assert view is a


def test_tied_router_logits_route_like_the_served_block():
    """Duplicated router rows give exactly tied probabilities; ties must pick
    the same experts, in the same order, with the same scores as the served
    router launches."""
    from omlx.patches.qwen35_moe_router import router_logits_row, softmax_topk_row

    block = _block(1024, 320, bits=5, experts=128)
    rows = block.gate.weight[:16]
    block.gate.weight = mx.concatenate([rows] * 8)[mx.random.permutation(128)]
    mx.eval(block.gate.weight)
    for step in range(8):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + step)).astype(mx.bfloat16)
        # Both router launches engage: the gemv and the softmax + top-k.
        assert router_logits_row(x, block.gate.weight) is not None
        assert softmax_topk_row(block.gate(x), 10) is not None
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)



@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("top_k", [8, 10])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_fast_down_and_top8_are_bit_identical(dtype, top_k, bits):
    """Qwen3.5/3.6-35B-A3B layout: hidden 2048 and intermediate 512 put the
    down projection on ``qmv_fast`` too."""
    block = _block(1024, 512, bits=bits, top_k=top_k, dtype=dtype)
    for step in range(8):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + step)).astype(dtype)
        plan = routed.routed_decode_plan(block, x)
        assert plan.fold and plan.dtype == dtype and plan.top_k == top_k
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert _engaged()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("quantized_shared", [True, False])
def test_top8_routing_folded_into_gate_up_is_bit_identical(dtype, quantized_shared):
    """128 experts select inside the gate+up launch; an unquantized shared
    expert runs composed and enters the combine."""
    block = _block(
        1024, 512, top_k=8, experts=128, quantized_shared=quantized_shared, dtype=dtype
    )
    for step in range(4):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + step)).astype(dtype)
        plan = routed.routed_decode_plan(block, x)
        assert plan.fold == quantized_shared
        assert routed._topk_folds(plan, block.gate(x))
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert _engaged()


def test_mismatched_dtypes_keep_the_composed_body():
    block = _block(1024, 512, top_k=8, dtype=mx.float16)
    x = mx.zeros((1, 1, 1024), mx.float16)
    # A router of another dtype runs as the block's own linear.
    gate = block.gate["weight"]
    block.gate["weight"] = gate.astype(mx.bfloat16)
    assert routed.routed_decode_plan(block, x).router_logits is None
    block.gate["weight"] = gate
    assert routed.routed_decode_plan(block, x.astype(mx.bfloat16)) is None
    down = block.switch_mlp.down_proj
    down["scales"] = down["scales"].astype(mx.bfloat16)
    down["biases"] = down["biases"].astype(mx.bfloat16)
    assert routed.routed_decode_plan(block, x) is None


@pytest.mark.parametrize(
    "owner,names",
    [
        ("switch_mlp", ("gate_up_proj", "down_proj")),
        ("shared_expert", ("gate_proj", "up_proj", "down_proj")),
    ],
)
def test_plan_follows_replaced_weights(owner, names):
    block = _block(1024, 320, bits=5)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    first = routed.routed_decode_plan(block, x)
    donor = _block(1024, 320, bits=5, seed=7)
    for name in names:
        for key in ("weight", "scales", "biases"):
            block[owner][name][key] = donor[owner][name][key]
    assert routed.routed_decode_plan(block, x) is not first
    ref, out = _pair(block, x)
    assert _same_bits(ref, out)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 1024, "inter": 320, "top_k": 6},
        {"hidden": 1024, "inter": 320, "bits": 3},
    ],
)
def test_ineligible_shapes_keep_the_composed_body(kwargs):
    block = _block(**kwargs)
    hidden = kwargs["hidden"]
    x = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)
    assert routed.routed_decode_plan(block, x) is None


@pytest.mark.parametrize(
    "name,value",
    [
        # A threadgroup takes six of the 320 gate rows.
        ("_GATE_UP_ROWS", 3),
        ("_GATE_UP_SIMDGROUPS", 3),
        ("_down_rows", lambda rows: 3 if rows == 1 else 4),
        ("_down_rows", lambda rows: 2 if rows == 1 else 3),
    ],
)
def test_rows_that_are_not_whole_launch_groups_keep_the_composed_body(
    name, value, monkeypatch
):
    """The grids divide the rows by a launch's rows per group. Affine experts
    are whole groups at every shape MLX quantizes; another format's are not
    bound to be."""
    monkeypatch.setattr(routed, name, value)
    block = _block(1024, 320)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    assert routed.routed_decode_plan(block, x) is None
    assert _same_bits(*_pair(block, x))
    assert not routed._PROOFS


@pytest.mark.parametrize("dtypes", [(), (mx.bfloat16, mx.float16)])
def test_a_gate_up_face_names_the_one_dtype_of_the_launches(dtypes, monkeypatch):
    """The launches are compiled for the activations the gate+up face is
    exact for; a face exact for several does not say which."""
    face = affine.AffineFormat.dot

    def several(self, layer, k, n):
        found = face(self, layer, k, n)
        if found is None or layer["weight"].ndim == 2 or k != 1024:
            return found
        return replace(found, dtypes=dtypes)

    monkeypatch.setattr(affine.AffineFormat, "dot", several)
    block = _block(1024, 320)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    assert routed.routed_decode_plan(block, x) is None
    assert _same_bits(*_pair(block, x))
    assert not routed._PROOFS


def test_prefill_verify_and_float_rows_keep_the_composed_body():
    block = _block(1024, 320)
    assert routed.routed_decode_plan(block, mx.zeros((1, 4, 1024), dtype=mx.bfloat16)) is None
    assert routed.routed_decode_plan(block, mx.zeros((2, 1, 1024), dtype=mx.bfloat16)) is None
    assert routed.routed_decode_plan(block, mx.zeros((1, 1, 1024), dtype=mx.float16)) is None
    x = mx.random.normal((1, 5, 1024)).astype(mx.bfloat16)
    ref, out = _pair(block, x)
    assert _same_bits(ref, out)


def test_block_without_gate_up_fusion_is_not_routed():
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    args = SimpleNamespace(
        hidden_size=1024,
        moe_intermediate_size=320,
        shared_expert_intermediate_size=320,
        num_experts=EXPERTS,
        num_experts_per_tok=10,
        norm_topk_prob=True,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    x = mx.zeros((1, 1, 1024), dtype=mx.bfloat16)
    assert routed.routed_decode_plan(block, x) is None


def test_a_failed_launch_keeps_its_faces_and_shapes_composed(monkeypatch, caplog):
    block = _block(1024, 320)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    calls = []

    def broken(*args):
        calls.append(1)
        raise RuntimeError("no pipeline")

    with monkeypatch.context() as patch:
        patch.setattr(routed, "routed_decode", broken)
        assert _same_bits(*_pair(block, x))
        assert routed.routed_decode_plan(block, x) is None
        # Another block of these faces and shapes does not try again.
        twin = _block(1024, 320, seed=1)
        assert routed.routed_decode_plan(twin, x) is None
        assert _same_bits(*_pair(twin, x))
        assert _same_bits(*_pair(block, x))
    assert len(calls) == 1
    warnings = _logged(caplog, routed, logging.WARNING)
    assert [r.getMessage() for r in warnings] == [
        "fused routed-expert decode failed; composed fallback"
    ]
    # Other shapes still route.
    other = _block(1024, 512)
    assert routed.routed_decode_plan(other, x) is not None
    assert _same_bits(*_pair(other, x))
    assert routed.routed_decode_plan(other, x).proof.ran


def test_a_face_that_does_not_compile_keeps_only_its_blocks_composed(
    monkeypatch, caplog
):
    """MLX compiles a kernel at its first evaluation. A face that fails
    there, after blocks of the same shapes ran with other faces, fails inside
    the call that can fall back."""
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    good = _block(1024, 320)
    assert _same_bits(*_pair(good, x))
    assert _engaged()
    dot = formats.dot

    def broken(layer, k, n):
        face = dot(layer, k, n)
        if face is None or k != 320:
            return face
        source = face.source + 'static_assert(false, "does not compile");\n'
        return replace(face, key=face.key + "_broken", source=source)

    monkeypatch.setattr(formats, "dot", broken)
    bad = _block(1024, 320, seed=1)
    assert _same_bits(*_pair(bad, x))
    # The warning carries the compiler's message.
    (warning,) = _logged(caplog, routed, logging.WARNING)
    assert "does not compile" in str(warning.exc_info[1])
    assert routed.routed_decode_plan(bad, x) is None
    assert routed.routed_decode_plan(good, x) is not None
    assert _same_bits(*_pair(good, x))


def test_a_format_whose_dot_raises_keeps_its_blocks_composed(monkeypatch, caplog):
    asked = []

    def raises(self, layer, k, n):
        asked.append(layer)
        raise RuntimeError("no face today")

    monkeypatch.setattr(formats, "_DOT_DEFECTS", set())
    monkeypatch.setattr(affine.AffineFormat, "dot", raises)
    block = _block(1024, 512)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        for _ in range(2):
            assert _same_bits(*_pair(block, x))
    assert routed.routed_decode_plan(block, x) is None
    assert not routed._PROOFS
    # The answer is kept with the block's plan.
    gate_up, down = asked
    experts = block.switch_mlp
    assert gate_up is experts.gate_up_proj and down is experts.down_proj
    assert [repr(r.exc_info[1]) for r in _logged(caplog, formats)] == [
        "RuntimeError('no face today')"
    ]


def test_blocks_share_a_proof_when_they_compile_the_same_launches(monkeypatch):
    def unfolded(hidden=1024, inter=320, **kwargs):
        return _block(hidden, inter, quantized_shared=False, **kwargs)

    def proof(block):
        x = mx.zeros((1, 1, block.switch_mlp.down_proj.weight.shape[1]), mx.bfloat16)
        return routed.routed_decode_plan(block, x).proof

    base, folded = proof(unfolded()), proof(_block(1024, 320))
    assert proof(unfolded(seed=1)) is base
    assert proof(_block(1024, 320, seed=1)) is folded
    assert folded is not base
    # One shape apart, with the same faces.
    assert proof(unfolded(hidden=2048)) is not base
    assert proof(unfolded(inter=448)) is not base
    assert proof(unfolded(top_k=8)) is not base
    # With 128 experts the router is the fused gemv.
    assert proof(unfolded(experts=128)) is not base
    dot = formats.dot

    def patched(face_of):
        monkeypatch.setattr(
            formats, "dot", lambda layer, k, n: face_of(layer, dot(layer, k, n))
        )

    def renamed(block, *layers):
        other = {id(layer) for layer in layers}

        def face_of(layer, face):
            return replace(face, key="other") if id(layer) in other else face

        patched(face_of)
        return proof(block)

    # One face apart.
    for name in ("gate_up_proj", "down_proj"):
        block = unfolded(seed=2)
        assert renamed(block, block.switch_mlp[name]) is not base
    for names in (("gate_proj", "up_proj"), ("down_proj",)):
        block = _block(1024, 320, seed=2)
        layers = (block.shared_expert[name] for name in names)
        assert renamed(block, *layers) is not folded
    block = _block(1024, 320, seed=2)
    assert renamed(block, block.shared_expert_gate) is not folded
    # The same faces on tensors of another dtype.
    block = unfolded(dtype=mx.float16)
    patched(lambda layer, face: replace(face, dtypes=(mx.bfloat16,)))
    assert proof(block) is not base


def test_a_launch_is_evaluated_once_per_router_width_and_fold(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=routed.__name__)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    ran = set()
    for experts in (128, 256, 128):
        block = _block(1024, 512, top_k=8, experts=experts)
        for _ in range(2):
            assert _same_bits(*_pair(block, x))
        ran.add((0, experts, True))
        assert routed.routed_decode_plan(block, x).proof.ran == ran
    monkeypatch.setattr(routed, "_topk_folds", lambda plan, logits: False)
    assert _same_bits(*_pair(block, x))
    assert routed.routed_decode_plan(block, x).proof.ran == ran | {(0, 128, False)}
    assert [r.getMessage() for r in _logged(caplog, routed)] == [
        "Qwen MoE fused routed-expert decode engaged"
    ] * 3


def test_apply_requires_the_fused_router(_patched_block):
    cls = _patched_block
    del cls._omlx_routed_decode
    cls._omlx_router_fused = False
    assert not routed.apply_qwen35_moe_routed_decode_patch()


def test_apply_loads_the_formats_or_leaves_the_block_unpatched(
    _patched_block, monkeypatch
):
    cls = _patched_block
    served = cls.__call__._omlx_routed_decode_original
    loads = []

    def unpatched():
        cls.__call__ = served
        del cls._omlx_routed_decode

    unpatched()
    monkeypatch.setattr(formats, "load", lambda: loads.append(1))
    assert routed.apply_qwen35_moe_routed_decode_patch()
    assert loads == [1]  # with the model, ahead of its first token

    def broken():
        raise ImportError("a built-in format does not import")

    unpatched()
    monkeypatch.setattr(formats, "load", broken)
    with pytest.raises(ImportError, match="built-in format"):
        routed.apply_qwen35_moe_routed_decode_patch()
    assert cls.__call__ is served and "_omlx_routed_decode" not in cls.__dict__


def test_apply_is_idempotent(_patched_block):
    cls = _patched_block
    call = cls.__call__
    assert routed.apply_qwen35_moe_routed_decode_patch()
    assert cls.__call__ is call
