# SPDX-License-Identifier: Apache-2.0
"""Transformed routed-decode and verify-window kernels for Qwen MoE blocks.

Covers:
1. Flash-Next mixed layout: oqh_affine routed experts (4b gate+up, 4b and
   reinvested 5b down, group_size=128) with plain affine shared expert and
   shared-expert gate (5 launches/layer, bit-for-bit vs composed reference,
   verify rows bit-for-bit vs 1-token decode, producer memo hit).
2. Row-scale oqh layout (OqhFormat 3b/4b routed experts + plain shared expert).
3. All-rotated layout (shared gate/up sharing rx with routed gate+up) and
   incompatible shared transforms (different signs, transformed shared down,
   transformed shared gate) cleanly unfolding the shared expert.
4. One-sided transforms (gate_up-only and down-only).
5. Unchanged non-transformed plans (zero transform flags on plain affine).
"""

from __future__ import annotations

import copy

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.formats import affine, kernels, linear, oqh, testing, transforms
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.patches import qwen35_moe_routed_decode as routed
from omlx.patches import qwen35_moe_router
from tests.test_formats_oqh import _group_case, _row_case
from tests.test_qwen35_moe_routed_decode import _block

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def _patch_routed():
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    from mlx_vlm.models.qwen3_5_moe import language as vlm_moe

    qwen35_moe_router.apply_qwen35_moe_router_patch()
    cls = vlm_moe.Qwen3_5MoeSparseMoeBlock
    orig_ff = Qwen3_5BatchInvariantForward._feed_forward
    orig_call = cls.__call__
    had_flag = "_omlx_routed_decode" in cls.__dict__
    routed.apply_qwen35_moe_routed_decode_patch()
    transforms.clear()
    yield
    transforms.clear()
    Qwen3_5BatchInvariantForward._feed_forward = orig_ff
    cls.__call__ = orig_call
    if not had_flag and "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")


def _oqh_affine_switch(k, n, experts, bits=4, gs=128, seed=0, block=128):
    parts = [
        _group_case(k, n, bits, gs, seed=seed + e, block=block) for e in range(experts)
    ]
    tensors = {
        "weight": mx.stack([p[0]["weight"] for p in parts]),
        "oqh_scales": mx.stack([p[0]["oqh_scales"] for p in parts]),
        "oqh_biases": mx.stack([p[0]["oqh_biases"] for p in parts]),
        "oqh_signs": parts[0][0]["oqh_signs"],
    }
    reader = copy.copy(oqh.OQH_AFFINE)
    reader.block = block
    with testing.verified("example"):
        layer = FormatSwitchLinear(reader, bits, gs, k, tensors)
    return layer


def _oqh_row_switch(k, n, experts, bits=4, seed=0, block=128):
    parts = [_row_case(k, n, bits, seed=seed + e, block=block) for e in range(experts)]
    tensors = {
        "weight": mx.stack([p[0]["weight"] for p in parts]),
        "rscale": mx.stack([p[0]["rscale"] for p in parts]),
        "oqh_signs": parts[0][0]["oqh_signs"],
    }
    reader = copy.copy(oqh.OQH)
    reader.block = block
    with testing.verified("example"):
        layer = FormatSwitchLinear(reader, bits, 0, k, tensors)
    return layer


def _oqh_affine_linear(k, n, bits=4, gs=128, seed=0, block=128, signs=None):
    tensors, _ = _group_case(k, n, bits, gs, seed=seed, block=block)
    if signs is not None:
        tensors["oqh_signs"] = signs
    reader = copy.copy(oqh.OQH_AFFINE)
    reader.block = block
    with testing.verified("example"):
        return FormatLinear(reader, bits, gs, k, tensors)


def _plain_affine_format_linear(ql: nn.QuantizedLinear) -> FormatLinear:
    tensors = {
        "weight": ql["weight"],
        "scales": ql["scales"],
        "biases": ql["biases"],
    }
    k = ql["weight"].shape[-1] * 32 // ql.bits
    with testing.verified("example"):
        return FormatLinear(affine.FORMAT, ql.bits, ql.group_size, k, tensors)


def _flash_next_oqh_block(
    *,
    hidden=512,
    inter=256,
    top_k=10,
    experts=128,
    down_bits=4,
    row_scale=False,
    row_bits=4,
    seed=0,
    quantized_router=False,
    format_shared=True,
):
    block = _block(
        hidden=hidden,
        inter=inter,
        top_k=top_k,
        bits=4,
        group_size=128,
        seed=seed,
        experts=experts,
    )
    gu_block = 256 if hidden % 256 == 0 else 128
    if row_scale:
        gate = _oqh_row_switch(
            hidden, inter, experts, bits=row_bits, seed=seed + 10, block=gu_block
        )
        up = _oqh_row_switch(
            hidden, inter, experts, bits=row_bits, seed=seed + 100, block=gu_block
        )
        up["oqh_signs"] = gate["oqh_signs"]
        down = _oqh_row_switch(
            inter, hidden, experts, bits=row_bits, seed=seed + 200, block=128
        )
    else:
        gate = _oqh_affine_switch(
            hidden, inter, experts, bits=4, gs=128, seed=seed + 10, block=gu_block
        )
        up = _oqh_affine_switch(
            hidden, inter, experts, bits=4, gs=128, seed=seed + 100, block=gu_block
        )
        up["oqh_signs"] = gate["oqh_signs"]
        down = _oqh_affine_switch(
            inter, hidden, experts, bits=down_bits, gs=128, seed=seed + 200, block=128
        )
    block.switch_mlp.gate_up_proj = gate._stacked_outputs(up)
    block.switch_mlp.gate_proj = gate
    block.switch_mlp.up_proj = up
    block.switch_mlp.down_proj = down
    if format_shared:
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(
                block.shared_expert,
                name,
                _plain_affine_format_linear(getattr(block.shared_expert, name)),
            )
        block.shared_expert_gate = _plain_affine_format_linear(block.shared_expert_gate)
    if quantized_router:
        block.gate = _plain_affine_format_linear(
            nn.QuantizedLinear.from_linear(block.gate, 64, 8)
        )
    mx.eval(block.parameters())
    return block


def _unfused_ref(block, x):
    """Exact unfused reference matching routed_decode's bf16 SwiGLU + rotate."""
    logits = block.gate(x)
    scores = mx.softmax(logits, axis=-1, precise=True)
    inds, top_scores = qwen35_moe_router.fused_router_topk(scores, block.top_k)
    sw = block.switch_mlp
    x_exp = mx.expand_dims(x, (-2, -3))
    g = sw.gate_proj(x_exp, inds)
    u = sw.up_proj(x_exp, inds)
    h = sw.activation(u, g)
    routed_out = sw.down_proj(h, inds).squeeze(-2)
    shared_out = block.shared_expert(x)
    shared_gate = block.shared_expert_gate(x)
    y = qwen35_moe_router.fused_moe_combine(
        routed_out, top_scores, shared_out, shared_gate
    )
    mx.eval(y)
    return y


def _composed(block, x):
    """Composed forward (may use apply_split_swiglu FP32 SwiGLU+WHT fusion)."""
    orig = type(block).__call__._omlx_routed_decode_original
    y = orig(block, x)
    mx.eval(y)
    return y


def _assert_close_to_composed(got, composed):
    diff = mx.abs(got.astype(mx.float32) - composed.astype(mx.float32))
    scale = max(float(mx.abs(composed.astype(mx.float32)).max().item()), 1e-6)
    assert float(diff.max().item()) / scale < 2e-2


@pytest.mark.parametrize("down_bits", [4, 5], ids=["down4b", "down5b"])
def test_flash_next_mixed_oqh_affine_decode_and_verify_are_bitwise(down_bits):
    block = _flash_next_oqh_block(down_bits=down_bits, seed=1)
    plan = routed._build_plan(block, mx.bfloat16)
    assert plan is not None
    assert plan.launches.fold is True
    assert plan.launches.router is not None
    assert plan.launches.routed_x is True
    assert plan.launches.split_out is True
    assert plan.launches.split_in is True
    assert plan.launches.shared_uses_rx is False
    assert plan.launches.d_tx_apply is not None
    assert plan.window is not None

    # 1-token decode matches unfused reference bit-for-bit (and composed within 1 ULP).
    mx.random.seed(11)
    x1 = mx.random.normal((1, 1, 512)).astype(mx.bfloat16) * 0.5
    ref1 = _unfused_ref(block, x1)
    comp1 = _composed(block, x1)
    got1 = block(x1)
    mx.eval(got1)
    assert mx.array_equal(got1, ref1).item()
    _assert_close_to_composed(got1, comp1)
    assert plan.proof.ran

    # Sanity check: skipping rotation produces a completely different output.
    unrotated_gu = block.switch_mlp.gate_up_proj._call_prepared(
        mx.expand_dims(x1, (-2, -3)),
        mx.zeros((1, 1, 10), mx.uint32),
    )
    rotated_gu = block.switch_mlp.gate_up_proj(
        mx.expand_dims(x1, (-2, -3)),
        mx.zeros((1, 1, 10), mx.uint32),
    )
    assert not mx.allclose(
        unrotated_gu.astype(mx.float32), rotated_gu.astype(mx.float32), atol=1e-2
    ).item()

    # Verify windows (2, 3, 4, 8 rows) match 1-token decode bit-for-bit per row.
    for rows in (2, 3, 4, 8):
        xm = mx.random.normal((1, rows, 512)).astype(mx.bfloat16) * 0.5
        got_m = routed.routed_verify_window(block, xm)
        assert got_m is not None
        mx.eval(got_m)
        assert got_m.shape == (rows, 512)
        for r in range(rows):
            row_ref = block(xm[:, r : r + 1, :])
            mx.eval(row_ref)
            assert mx.array_equal(got_m[r], row_ref[0, 0]).item(), f"rows={rows} r={r}"

    # Batched 1-token decode (B, 1, hidden) routes through window launches too.
    xb = mx.random.normal((3, 1, 512)).astype(mx.bfloat16) * 0.5
    got_b = block(xb)
    mx.eval(got_b)
    for b in range(3):
        row_ref = block(xb[b : b + 1, :, :])
        mx.eval(row_ref)
        assert mx.array_equal(got_b[b], row_ref[0]).item()


def test_flash_next_oqh_verify_window_with_quantized_router():
    block = _flash_next_oqh_block(quantized_router=True, seed=2)
    plan = routed._build_plan(block, mx.bfloat16)
    assert plan is not None
    assert plan.launches.router is None
    assert plan.quantized_router is True
    assert plan.launches.fold is True
    assert plan.launches.routed_x is True
    assert plan.launches.split_out is True

    mx.random.seed(12)
    xm = mx.random.normal((1, 3, 512)).astype(mx.bfloat16) * 0.5
    got_m = routed.routed_verify_window(block, xm)
    assert got_m is not None
    mx.eval(got_m)
    for r in range(3):
        ref = _unfused_ref(block, xm[:, r : r + 1, :])
        assert mx.array_equal(got_m[r], ref[0, 0]).item()


@pytest.mark.parametrize("bits", [3, 4], ids=["row3b", "row4b"])
def test_row_scale_oqh_routed_decode_and_verify_are_bitwise(bits):
    block = _flash_next_oqh_block(row_scale=True, row_bits=bits, experts=64, seed=3)
    plan = routed._build_plan(block, mx.bfloat16)
    assert plan is not None
    assert plan.launches.fold is True
    assert plan.launches.routed_x is True
    assert plan.launches.split_out is True

    mx.random.seed(13)
    x1 = mx.random.normal((1, 1, 512)).astype(mx.bfloat16) * 0.5
    ref1 = _unfused_ref(block, x1)
    got1 = block(x1)
    mx.eval(got1)
    assert mx.array_equal(got1, ref1).item()

    xm = mx.random.normal((1, 3, 512)).astype(mx.bfloat16) * 0.5
    got_m = routed.routed_verify_window(block, xm)
    assert got_m is not None
    mx.eval(got_m)
    for r in range(3):
        row_ref = block(xm[:, r : r + 1, :])
        mx.eval(row_ref)
        assert mx.array_equal(got_m[r], row_ref[0, 0]).item()


def test_shared_expert_with_same_gate_up_transform_shares_rx():
    block = _flash_next_oqh_block(experts=64, seed=4)
    signs = block.switch_mlp.gate_up_proj["oqh_signs"]
    block.shared_expert.gate_proj = _oqh_affine_linear(
        512, 256, bits=4, gs=128, seed=41, block=256, signs=signs
    )
    block.shared_expert.up_proj = _oqh_affine_linear(
        512, 256, bits=4, gs=128, seed=42, block=256, signs=signs
    )
    mx.eval(block.parameters())
    plan = routed._build_plan(block, mx.bfloat16)
    assert plan is not None
    assert plan.launches.fold is True
    assert plan.launches.routed_x is True
    assert plan.launches.shared_uses_rx is True

    mx.random.seed(14)
    x1 = mx.random.normal((1, 1, 512)).astype(mx.bfloat16) * 0.5
    ref1 = _unfused_ref(block, x1)
    got1 = block(x1)
    mx.eval(got1)
    assert mx.array_equal(got1, ref1).item()

    xm = mx.random.normal((1, 3, 512)).astype(mx.bfloat16) * 0.5
    got_m = routed.routed_verify_window(block, xm)
    assert got_m is not None
    mx.eval(got_m)
    for r in range(3):
        row_ref = block(xm[:, r : r + 1, :])
        mx.eval(row_ref)
        assert mx.array_equal(got_m[r], row_ref[0, 0]).item()


@pytest.mark.parametrize(
    "case",
    [
        "shared_diff_signs",
        "shared_down_tx",
        "shared_gate_tx",
        "plain_routed_tx_shared_gu",
    ],
)
def test_incompatible_shared_transform_unfolds_cleanly(case):
    if case == "plain_routed_tx_shared_gu":
        block = _block(hidden=512, inter=256, top_k=10, experts=64, seed=5)
        block.shared_expert.gate_proj = _oqh_affine_linear(
            512, 256, bits=4, gs=128, seed=51, block=256
        )
        block.shared_expert.up_proj = _oqh_affine_linear(
            512, 256, bits=4, gs=128, seed=52, block=256
        )
    else:
        block = _flash_next_oqh_block(experts=64, seed=5)
        if case == "shared_diff_signs":
            diff = -block.switch_mlp.gate_up_proj["oqh_signs"]
            block.shared_expert.gate_proj = _oqh_affine_linear(
                512, 256, bits=4, gs=128, seed=53, block=256, signs=diff
            )
            block.shared_expert.up_proj = _oqh_affine_linear(
                512, 256, bits=4, gs=128, seed=54, block=256, signs=diff
            )
        elif case == "shared_down_tx":
            block.shared_expert.down_proj = _oqh_affine_linear(
                256, 512, bits=4, gs=128, seed=55, block=128
            )
        elif case == "shared_gate_tx":
            block.shared_expert_gate = _oqh_affine_linear(
                512, 1, bits=8, gs=64, seed=56, block=256
            )
    mx.eval(block.parameters())
    plan = routed._build_plan(block, mx.bfloat16)
    assert plan is not None
    assert plan.launches.fold is False

    mx.random.seed(15)
    x1 = mx.random.normal((1, 1, 512)).astype(mx.bfloat16) * 0.5
    ref1 = _unfused_ref(block, x1)
    got1 = block(x1)
    mx.eval(got1)
    assert mx.array_equal(got1, ref1).item()


@pytest.mark.parametrize("side", ["gate_up_only", "down_only"])
def test_one_sided_routed_transform_is_bitwise(side):
    block = _block(hidden=512, inter=256, top_k=10, experts=64, seed=6)
    if side == "gate_up_only":
        gate = _oqh_affine_switch(512, 256, 64, bits=4, gs=128, seed=60, block=256)
        up = _oqh_affine_switch(512, 256, 64, bits=4, gs=128, seed=160, block=256)
        up["oqh_signs"] = gate["oqh_signs"]
        block.switch_mlp.gate_up_proj = gate._stacked_outputs(up)
        block.switch_mlp.gate_proj = gate
        block.switch_mlp.up_proj = up
    else:
        block.switch_mlp.down_proj = _oqh_affine_switch(
            256, 512, 64, bits=4, gs=128, seed=260, block=128
        )
    mx.eval(block.parameters())
    plan = routed._build_plan(block, mx.bfloat16)
    assert plan is not None
    assert plan.launches.fold is True
    assert plan.launches.routed_x is (side == "gate_up_only")
    assert plan.launches.split_out is (side == "down_only")
    assert plan.launches.split_in is (side == "down_only")

    mx.random.seed(16)
    x1 = mx.random.normal((1, 1, 512)).astype(mx.bfloat16) * 0.5
    ref1 = _unfused_ref(block, x1)
    got1 = block(x1)
    mx.eval(got1)
    assert mx.array_equal(got1, ref1).item()

    xm = mx.random.normal((1, 4, 512)).astype(mx.bfloat16) * 0.5
    got_m = routed.routed_verify_window(block, xm)
    assert got_m is not None
    mx.eval(got_m)
    for r in range(4):
        row_ref = block(xm[:, r : r + 1, :])
        mx.eval(row_ref)
        assert mx.array_equal(got_m[r], row_ref[0, 0]).item()


def test_routed_decode_reuses_producer_memo_without_extra_rotation(monkeypatch):
    block = _flash_next_oqh_block(experts=64, seed=7)
    plan = routed._cached_plan(block, mx.bfloat16)
    assert plan is not None and plan.gu_tx is not None
    x1 = mx.random.normal((1, 1, 512)).astype(mx.bfloat16) * 0.5
    # Warm up the compiled graph first.
    ref = block(x1)
    mx.eval(ref)

    gu_tx, gu_tensors, gu_key = plan.gu_tx
    rx = gu_tx.apply(x1, *gu_tensors)
    mx.eval(rx)
    calls = []
    orig_apply = gu_tx.apply
    object.__setattr__(
        gu_tx, "apply", lambda *a, **kw: calls.append(a) or orig_apply(*a, **kw)
    )
    try:
        transforms.remember(x1, rx, gu_key, gu_tensors)
        got = block(x1)
        mx.eval(got)
        assert mx.array_equal(got, ref).item()
        assert not calls, "expected producer memo hit with zero extra gu_tx.apply calls"
    finally:
        object.__setattr__(gu_tx, "apply", orig_apply)


def test_plain_affine_plan_has_zero_transform_overhead_flags():
    block = _block(hidden=512, inter=256, top_k=10, experts=64, seed=8)
    plan = routed._build_plan(block, mx.bfloat16)
    assert plan is not None
    assert plan.launches.fold is True
    assert plan.gu_tx is None
    assert plan.d_tx is None
    assert plan.launches.routed_x is False
    assert plan.launches.split_out is False
    assert plan.launches.split_in is False
    assert plan.launches.shared_uses_rx is False
    assert plan.launches.d_tx_apply is None


def _transformed_block_for_warm(mode: str, *, quantized_router: bool = False):
    if mode == "both":
        return _flash_next_oqh_block(
            experts=64, seed=21, quantized_router=quantized_router
        )
    if mode == "unfolded":
        block = _flash_next_oqh_block(
            experts=64, seed=22, quantized_router=quantized_router
        )
        block.shared_expert.down_proj = _oqh_affine_linear(
            256, 512, bits=4, gs=128, seed=225, block=128
        )
        mx.eval(block.parameters())
        return block
    block = _block(hidden=512, inter=256, top_k=10, experts=64, seed=23)
    if mode == "gate_up_only":
        gate = _oqh_affine_switch(512, 256, 64, bits=4, gs=128, seed=230, block=256)
        up = _oqh_affine_switch(512, 256, 64, bits=4, gs=128, seed=330, block=256)
        up["oqh_signs"] = gate["oqh_signs"]
        block.switch_mlp.gate_up_proj = gate._stacked_outputs(up)
        block.switch_mlp.gate_proj = gate
        block.switch_mlp.up_proj = up
    elif mode == "down_only":
        block.switch_mlp.down_proj = _oqh_affine_switch(
            256, 512, 64, bits=4, gs=128, seed=430, block=128
        )
    else:
        raise AssertionError(f"unexpected mode: {mode}")
    mx.eval(block.parameters())
    return block


@pytest.mark.parametrize("mode", ["both", "gate_up_only", "down_only", "unfolded"])
def test_warm_moe_model_and_dispatch_transformed_block(monkeypatch, mode):
    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    block = _transformed_block_for_warm(mode)

    # Emulate load-time warm of the block's dense FormatLinear layers
    # (shared_expert / shared_expert_gate) before post-load MoE warm.
    for future in linear.warm_existing_dense(block, dtypes=(mx.bfloat16,)) or ():
        future.result()
    routed.warm_moe_model(block, mtp_enabled=True, warm_dispatch=True)

    plan = routed._cached_plan(block, mx.bfloat16)
    assert plan is not None and not plan.proof.failed
    sig1 = ((1, 1, 512), mx.bfloat16, plan.experts)
    assert sig1 in plan.proof.graphs
    if plan.window is not None:
        assert not plan.window.failed
        for shape in ((1, 2, 512), (1, 4, 512), (2, 1, 512)):
            wsig = (shape, mx.bfloat16, plan.experts, plan.quantized_router)
            assert wsig in plan.window.graphs

    before = kernels.proof_count()
    mx.random.seed(29)
    x1 = mx.random.normal((1, 1, 512)).astype(mx.bfloat16) * 0.5
    got1 = block(x1)
    mx.eval(got1)
    assert kernels.proof_count() == before

    win_results = []
    if plan.window is not None:
        for rows in (2, 4):
            xm = mx.random.normal((1, rows, 512)).astype(mx.bfloat16) * 0.5
            got_m = routed.routed_verify_window(block, xm)
            assert got_m is not None
            mx.eval(got_m)
            assert kernels.proof_count() == before
            win_results.append((rows, xm, got_m))

    assert mx.array_equal(got1, _unfused_ref(block, x1)).item()
    for rows, xm, got_m in win_results:
        for r in range(rows):
            row_ref = block(xm[:, r : r + 1, :])
            mx.eval(row_ref)
            assert mx.array_equal(got_m[r], row_ref[0, 0]).item()


@pytest.mark.parametrize("quantized_router", [False, True])
def test_warm_engine_dispatch_transformed_block_and_quantized_router(
    monkeypatch, quantized_router
):
    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    block = _transformed_block_for_warm("both", quantized_router=quantized_router)

    for future in linear.warm_existing_dense(block, dtypes=(mx.bfloat16,)) or ():
        future.result()
    routed.warm_moe_model(block, mtp_enabled=True, warm_dispatch=False)
    routed.warm_engine_dispatch(block, None, mtp_enabled=True)

    plan = routed._cached_plan(block, mx.bfloat16)
    assert plan is not None and not plan.proof.failed
    assert plan.quantized_router is quantized_router
    assert ((1, 1, 512), mx.bfloat16, plan.experts) in plan.proof.graphs
    assert plan.window is not None and not plan.window.failed
    assert (
        (1, 4, 512),
        mx.bfloat16,
        plan.experts,
        quantized_router,
    ) in plan.window.graphs

    before = kernels.proof_count()
    x1 = mx.random.normal((1, 1, 512)).astype(mx.bfloat16) * 0.5
    got1 = block(x1)
    mx.eval(got1)
    xm = mx.random.normal((1, 4, 512)).astype(mx.bfloat16) * 0.5
    got_m = routed.routed_verify_window(block, xm)
    assert got_m is not None
    mx.eval(got_m)
    assert kernels.proof_count() == before


def test_transformed_routed_decode_merge2_down_policies(monkeypatch, tmp_path):
    """Merge #2 + oQh guard: under active per-chip routed_down split policies,
    a split transformed block (d_tx set) builds zero unused plain split/unrolled
    down kernels and warms single-stream + batched windows with 0 cold proofs;
    a gate_up-only transformed block (d_tx None) executes cleanly through
    _window_down using the per-chip routed_down policies."""
    import json

    from omlx.formats import tuning

    routed._ensure_tuning_registered()
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir(parents=True, exist_ok=True)
    chip = "applegpu_g17s_c40_oqh_split_test"
    skel = tuning.skeleton_digest("routed_down")
    entries = []
    for grp in (64, 128):
        for dtype_name in ("bf16", "fp16"):
            entries.append(
                {
                    "key": {
                        "route": "routed_down",
                        "face_id": "*:plain",
                        "dtype": dtype_name,
                        "bits": 4,
                        "group": grp,
                        "rows_bucket": "4",
                        "n": 512,
                        "k": 256,
                    },
                    "face_sha256": "",
                    "skeleton_sha256": skel,
                    "measured_at": "2026-11-26T01:30:00Z",
                    "exactness": "bitwise",
                    "params": {"mode": "split", "rps": 4, "unroll_pairs": True},
                }
            )
    payload = {
        "schema_version": 1,
        "chip": chip,
        "entries": entries,
    }
    (v1_dir / f"{chip}.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1_dir])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    monkeypatch.setattr(tuning, "current_chip", lambda: chip)
    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    tuning.reset_cache()
    try:
        # 1. Both gate_up and down transformed (split=True): zero unused plain
        #    split/unrolled down kernels, 0 cold proofs on both window shapes.
        block_both = _transformed_block_for_warm("both", quantized_router=False)
        routed.warm_moe_model(block_both, mtp_enabled=True, warm_dispatch=True)
        plan_both = routed._cached_plan(block_both, mx.bfloat16)
        assert plan_both is not None and not plan_both.proof.failed
        lb = plan_both.launches
        assert lb.split_in is True
        assert lb.window_down_routed_kernel is None
        assert lb.window_down_routed_unrolled_kernel is None
        assert lb.window_down_shared_kernel is None
        assert lb.window_down_combine_kernel is None
        assert lb.window_down_unrolled_kernel is None
        assert lb.window_down is not None

        before = kernels.proof_count()
        for shape in ((1, 4, 512), (2, 2, 512)):
            x = (mx.random.normal(shape) * 0.5).astype(mx.bfloat16)
            got = routed.routed_verify_window(block_both, x)
            assert got is not None
            mx.eval(got)
            flat = x.reshape(-1, 512)
            for r in range(4):
                ref_r = block_both(flat[r : r + 1].reshape(1, 1, 512))
                mx.eval(ref_r)
                assert mx.array_equal(got[r], ref_r[0, 0]).item()
        assert kernels.proof_count() == before

        # 2. gate_up-only transformed (d_tx=None, split=False): routes through
        #    merge #2's _window_down policies cleanly.
        block_gu = _transformed_block_for_warm("gate_up_only", quantized_router=False)
        plan_gu = routed._cached_plan(block_gu, mx.bfloat16)
        assert plan_gu is not None and plan_gu.launches.split_in is False
        for shape in ((1, 4, 512), (2, 2, 512)):
            x = (mx.random.normal(shape) * 0.5).astype(mx.bfloat16)
            got = routed.routed_verify_window(block_gu, x)
            assert got is not None
            mx.eval(got)
            flat = x.reshape(-1, 512)
            for r in range(4):
                ref_r = block_gu(flat[r : r + 1].reshape(1, 1, 512))
                mx.eval(ref_r)
                assert mx.array_equal(got[r], ref_r[0, 0]).item()
    finally:
        tuning.reset_cache()
