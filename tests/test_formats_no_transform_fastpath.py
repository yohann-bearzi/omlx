# SPDX-License-Identifier: Apache-2.0
"""Rule 28: transform presence is resolved at construction / plan-build time so
plain (no-InputTransform) FormatLinear, FormatSwitchLinear, FormatEmbedding,
and MoE routes execute the pre-oQh call path with zero per-call transform
checks, attribute lookups, or cache probes."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

from omlx import formats
from omlx.formats import affine, oqh, testing, transforms
from omlx.formats import linear as linear_mod
from omlx.formats.linear import (
    FormatEmbedding,
    FormatLinear,
    FormatSwitchLinear,
    _FormatLayer,
)
from omlx.patches import (
    moe_verify_gather,
    qwen35_moe_gate_up,
    qwen35_moe_router,
    qwen35_moe_weighted_sum,
)
from omlx.patches import qwen35_moe_routed_decode as routed
from tests.test_formats_oqh import _group_case

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_TX_NAMES = frozenset(
    {
        "tx",
        "_call_prepared",
        "_prepare_input",
        "input_transform",
        "transforms",
        "prepare_bound",
        "gu_tx",
        "d_tx",
        "dn_tx",
        "split_out",
        "split_in",
        "routed_x",
        "d_tx_apply",
        "_stack_transforms",
        "apply_split_swiglu",
    }
)


def _affine_linear(
    k: int = 256, n: int = 64, bits: int = 4, gs: int = 64, seed: int = 0
):
    mx.random.seed(seed)
    lin = nn.Linear(k, n, bias=False)
    lin.set_dtype(mx.bfloat16)
    q = lin.to_quantized(group_size=gs, bits=bits)
    mx.eval(q.parameters())
    tensors = {"weight": q["weight"], "scales": q["scales"], "biases": q["biases"]}
    with testing.verified("example"):
        return FormatLinear("affine", bits, gs, k, tensors), tensors


def _affine_switch(
    k: int = 256,
    n: int = 64,
    experts: int = 8,
    bits: int = 4,
    gs: int = 64,
    seed: int = 0,
):
    mx.random.seed(seed)
    from mlx_lm.models.switch_layers import SwitchLinear

    sw = SwitchLinear(k, n, experts, bias=False)
    sw.set_dtype(mx.bfloat16)
    q = sw.to_quantized(group_size=gs, bits=bits)
    mx.eval(q.parameters())
    tensors = {"weight": q["weight"], "scales": q["scales"], "biases": q["biases"]}
    with testing.verified("example"):
        return FormatSwitchLinear("affine", bits, gs, k, tensors), tensors


def _oqh_linear(k: int = 256, n: int = 64, seed: int = 0):
    tensors, _ = _group_case(k, n, 4, 128, seed=seed)
    with testing.verified("example"):
        return FormatLinear(oqh.OQH_AFFINE, 4, 128, k, tensors), tensors


def _oqh_switch(k: int = 256, n: int = 64, experts: int = 8, seed: int = 0):
    parts = [_group_case(k, n, 4, 128, seed=seed + e) for e in range(experts)]
    tensors = {
        "weight": mx.stack([p[0]["weight"] for p in parts]),
        "oqh_scales": mx.stack([p[0]["oqh_scales"] for p in parts]),
        "oqh_biases": mx.stack([p[0]["oqh_biases"] for p in parts]),
        "oqh_signs": parts[0][0]["oqh_signs"],
    }
    with testing.verified("example"):
        return FormatSwitchLinear(oqh.OQH_AFFINE, 4, 128, k, tensors), tensors


def test_plain_format_layers_bind_plain_call_path_at_construction(monkeypatch):
    lin, lin_tensors = _affine_linear(256, 64, seed=1)
    lin2, _ = _affine_linear(256, 64, seed=2)
    sw, sw_tensors = _affine_switch(256, 64, 8, seed=3)
    sw2, _ = _affine_switch(256, 64, 8, seed=4)
    with testing.verified("example"):
        emb = FormatEmbedding("affine", 4, 64, 256, lin_tensors)
        lin_ph = FormatLinear("affine", 4, 64, 256, lin_tensors, placeholders=True)
        sw_ph = FormatSwitchLinear("affine", 4, 64, 256, sw_tensors, placeholders=True)

    assert type(lin) is FormatLinear
    assert type(sw) is FormatSwitchLinear
    assert type(emb) is FormatEmbedding
    assert type(lin_ph) is FormatLinear
    assert type(sw_ph) is FormatSwitchLinear
    assert type(copy.copy(lin)) is FormatLinear
    assert type(copy.copy(sw)) is FormatSwitchLinear
    assert type(lin._stacked_outputs(lin2)) is FormatLinear
    assert type(sw._stacked_outputs(sw2)) is FormatSwitchLinear
    assert type(emb._linear_prototype) is FormatLinear

    # Plain layers run __call__ directly without an extra _call_prepared hop.
    assert FormatLinear.__call__ is FormatLinear._call_prepared
    assert FormatSwitchLinear.__call__ is FormatSwitchLinear._call_prepared

    # Bytecode of every plain call path has zero transform names or lookups.
    for fn in (
        FormatLinear.__call__,
        FormatSwitchLinear.__call__,
        FormatSwitchLinear._sorted_swiglu,
        FormatEmbedding.__call__,
        _FormatLayer._keep,
    ):
        leaked = _TX_NAMES & set(fn.__code__.co_names)
        assert not leaked, f"{fn.__qualname__} references transform names {leaked}"

    # Plain calls never query formats.input_transform, even on a cold _keep().
    monkeypatch.setattr(
        formats,
        "input_transform",
        lambda *a, **kw: pytest.fail("plain format layer queried input_transform"),
    )
    x1 = mx.random.normal((1, 256)).astype(mx.bfloat16)
    x4 = mx.random.normal((4, 256)).astype(mx.bfloat16)
    xs = mx.random.normal((1, 2, 1, 1, 256)).astype(mx.bfloat16)
    idx = mx.array([[[0, 1], [2, 3]]], dtype=mx.uint32)
    mx.eval(
        lin(x1),
        lin(x4),
        sw(xs, idx),
        emb(mx.array([0, 1], dtype=mx.uint32)),
        emb.as_linear(x1),
    )


def test_transformed_format_layers_bind_subclass_at_construction():
    lin, tensors = _oqh_linear(256, 64, seed=10)
    lin2, _ = _oqh_linear(256, 64, seed=11)
    lin2["oqh_signs"] = lin["oqh_signs"]
    sw, sw_tensors = _oqh_switch(256, 64, 8, seed=12)
    sw2, _ = _oqh_switch(256, 64, 8, seed=13)
    sw2["oqh_signs"] = sw["oqh_signs"]
    with testing.verified("example"):
        emb = FormatEmbedding(oqh.OQH_AFFINE, 4, 128, 256, tensors)
        lin_ph = FormatLinear(oqh.OQH_AFFINE, 4, 128, 256, tensors, placeholders=True)
        sw_ph = FormatSwitchLinear(
            oqh.OQH_AFFINE, 4, 128, 256, sw_tensors, placeholders=True
        )

    assert isinstance(lin, FormatLinear) and type(lin) is not FormatLinear
    assert isinstance(sw, FormatSwitchLinear) and type(sw) is not FormatSwitchLinear
    assert isinstance(emb, FormatEmbedding) and type(emb) is not FormatEmbedding
    assert type(lin_ph) is type(lin)
    assert type(sw_ph) is type(sw)
    assert type(copy.copy(lin)) is type(lin)
    assert type(copy.copy(sw)) is type(sw)
    assert type(lin._stacked_outputs(lin2)) is type(lin)
    assert type(sw._stacked_outputs(sw2)) is type(sw)
    assert type(emb._linear_prototype) is type(lin)

    # An instance-level input_transform override on AffineFormat also binds
    # the transformed subclass and disables _stock bypass.
    _, aff_tensors = _affine_linear(256, 64, seed=14)
    override_fmt = affine.AffineFormat()
    override_fmt.input_transform = lambda m: None
    with testing.verified("example"):
        override_layer = FormatLinear(override_fmt, 4, 64, 256, aff_tensors)
    assert type(override_layer) is type(lin)
    assert override_layer._stock is False


def test_routed_decode_plain_hot_path_has_no_transform_checks(monkeypatch):
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    assert qwen35_moe_router.apply_qwen35_moe_router_patch()
    orig_ff = Qwen3_5BatchInvariantForward._feed_forward
    orig_call = Qwen3_5MoeSparseMoeBlock.__call__
    had_routed = "_omlx_routed_decode" in Qwen3_5MoeSparseMoeBlock.__dict__
    try:
        assert routed.apply_qwen35_moe_routed_decode_patch()

        patched_call = Qwen3_5MoeSparseMoeBlock.__call__
        for fn in (
            patched_call,
            routed.routed_verify_window,
            routed._experts,
            routed._gate_up,
            routed._gate_up_topk,
            routed._window_gate_up,
        ):
            leaked = _TX_NAMES & set(fn.__code__.co_names)
            assert not leaked, f"{fn.__qualname__} references transform names {leaked}"

        mx.random.seed(20)
        block = Qwen3_5MoeSparseMoeBlock(
            SimpleNamespace(
                hidden_size=512,
                moe_intermediate_size=256,
                shared_expert_intermediate_size=256,
                num_experts=64,
                num_experts_per_tok=8,
                norm_topk_prob=True,
            )
        )
        block.set_dtype(mx.bfloat16)
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(
                block.switch_mlp,
                name,
                getattr(block.switch_mlp, name).to_quantized(64, 4),
            )
            setattr(
                block.shared_expert,
                name,
                nn.QuantizedLinear.from_linear(
                    getattr(block.shared_expert, name), 64, 8
                ),
            )
        block.shared_expert_gate = nn.QuantizedLinear.from_linear(
            block.shared_expert_gate, 64, 8
        )
        block.eval()
        qwen35_moe_gate_up._fuse_one(block.switch_mlp)
        mx.eval(block.parameters())

        plan = routed._build_plan(block, mx.bfloat16)
        assert plan is not None
        assert plan.gu_tx is None and plan.d_tx is None
        assert plan.outside is False
        assert plan.window_outside is False

        x1 = (mx.random.normal((1, 1, 512)) * 0.5).astype(mx.bfloat16)
        x2 = (mx.random.normal((1, 2, 512)) * 0.5).astype(mx.bfloat16)
        monkeypatch.setattr(
            transforms,
            "prepare_bound",
            lambda *a, **kw: pytest.fail("plain routed decode touched prepare_bound"),
        )
        y1 = block(x1)
        y2 = routed.routed_verify_window(block, x2)
        assert y2 is not None
        mx.eval(y1, y2)
    finally:
        Qwen3_5BatchInvariantForward._feed_forward = orig_ff
        Qwen3_5MoeSparseMoeBlock.__call__ = orig_call
        if (
            not had_routed
            and "_omlx_routed_decode" in Qwen3_5MoeSparseMoeBlock.__dict__
        ):
            delattr(Qwen3_5MoeSparseMoeBlock, "_omlx_routed_decode")


def test_verify_gather_plain_plan_skips_transform_checks(monkeypatch):
    leaked = {"transforms", "prepare_bound"} & set(
        moe_verify_gather.fused_switch.__code__.co_names
    )
    assert not leaked, f"fused_switch references {leaked}"

    glu = SwitchGLU(256, 64, 8)
    glu.set_dtype(mx.bfloat16)
    nn.quantize(glu, group_size=64, bits=4)
    glu.eval()
    mx.eval(glu.parameters())
    qwen35_moe_gate_up._fuse_one(glu)
    x = mx.random.normal((1, 4, 256)).astype(mx.bfloat16)
    idx = mx.random.randint(0, 8, (1, 4, 2)).astype(mx.uint32)
    monkeypatch.setattr(
        transforms,
        "prepare_bound",
        lambda *a, **kw: pytest.fail("plain verify gather touched prepare_bound"),
    )
    got = moe_verify_gather.fused_switch(glu, x, idx)
    assert got is not None
    mx.eval(got)
    plan = glu.__dict__["_omlx_verify_gather_plan"][3]
    assert plan.plain_ran is plan.proof.ran

    # Transformed plan never shares proof.ran in plain_ran.
    g_oqh, _ = _oqh_switch(256, 128, 8, seed=30)
    u_oqh, _ = _oqh_switch(256, 128, 8, seed=31)
    u_oqh["oqh_signs"] = g_oqh["oqh_signs"]
    d_oqh, _ = _oqh_switch(128, 256, 8, seed=32)
    tx_glu = SwitchGLU(256, 128, 8)
    tx_glu.gate_proj, tx_glu.up_proj, tx_glu.down_proj = g_oqh, u_oqh, d_oqh
    qwen35_moe_gate_up._fuse_one(tx_glu)
    tx_plan = moe_verify_gather._build_plan(tx_glu, 256, 2)
    assert tx_plan is not None
    assert tx_plan.plain_ran is not tx_plan.proof.ran
    assert len(tx_plan.plain_ran) == 0


def test_gate_up_and_weighted_sum_plain_switches_skip_transform_cache(monkeypatch):
    qwen35_moe_gate_up._ensure_call_patch()
    leaked = _TX_NAMES & set(SwitchGLU.__call__.__code__.co_names)
    assert not leaked, f"plain SwitchGLU.__call__ references {leaked}"

    # Plain FormatSwitchLinear("affine") SwitchGLU stays SwitchGLU after fusion.
    g_aff, _ = _affine_switch(256, 64, 8, seed=40)
    u_aff, _ = _affine_switch(256, 64, 8, seed=41)
    d_aff, _ = _affine_switch(64, 256, 8, seed=42)
    glu = SwitchGLU(256, 64, 8)
    glu.gate_proj, glu.up_proj, glu.down_proj = g_aff, u_aff, d_aff
    qwen35_moe_gate_up._fuse_one(glu)
    assert type(glu) is SwitchGLU

    monkeypatch.setattr(
        formats,
        "input_transform",
        lambda *a, **kw: pytest.fail("plain SwitchGLU queried input_transform"),
    )
    for rows in (2, 32):
        x = mx.random.normal((1, rows, 256)).astype(mx.bfloat16)
        idx = mx.random.randint(0, 8, (1, rows, 2)).astype(mx.uint32)
        mx.eval(glu(x, idx))
    assert "_omlx_stack_transforms" not in glu.__dict__

    x64 = mx.random.normal((1, 64, 256)).astype(mx.bfloat16)
    idx64 = mx.random.randint(0, 8, (1, 64, 2)).astype(mx.uint32)
    scores64 = mx.softmax(mx.random.normal((1, 64, 2)), axis=-1)
    mx.eval(
        qwen35_moe_weighted_sum._native_switch_weighted_sum(
            glu, x64, idx64, scores64, lambda sorted_out, inv, s: sorted_out
        )
    )
    assert "_omlx_weighted_sum_transforms" not in glu.__dict__


def test_vlm_switch_glu_recognizes_transformed_format_switch(monkeypatch):
    qwen35_moe_gate_up._ensure_call_patch()
    g_oqh, _ = _oqh_switch(256, 128, 8, seed=50)
    u_oqh, _ = _oqh_switch(256, 128, 8, seed=51)
    u_oqh["oqh_signs"] = g_oqh["oqh_signs"]
    d_oqh, _ = _oqh_switch(128, 256, 8, seed=52)
    vlm_glu = VLMSwitchGLU(256, 128, 8)
    vlm_glu.gate_proj, vlm_glu.up_proj, vlm_glu.down_proj = g_oqh, u_oqh, d_oqh
    vlm_glu.eval()
    qwen35_moe_gate_up._fuse_one(vlm_glu)
    assert isinstance(vlm_glu, VLMSwitchGLU) and type(vlm_glu) is not VLMSwitchGLU

    called_verify = []
    orig_verify = qwen35_moe_gate_up._fused_verify_switch

    def spy_verify(switch_mlp, x, indices):
        called_verify.append(tuple(x.shape))
        return orig_verify(switch_mlp, x, indices)

    monkeypatch.setattr(qwen35_moe_gate_up, "_fused_verify_switch", spy_verify)
    # batch=8, length=2, top_k=8 -> batch * top_k = 64 >= 64, length in (1, DECODE_BLOCK_SIZE]:
    # fused_format must recognise _TransformedFormatSwitchLinear so it stays on
    # _fused_verify_switch instead of falling back to original().
    x = mx.random.normal((8, 2, 256)).astype(mx.bfloat16)
    idx = mx.random.randint(0, 8, (8, 2, 8)).astype(mx.uint32)
    out = vlm_glu(x, idx)
    mx.eval(out)
    assert called_verify == [(8, 2, 256)]


def test_format_linear_subclass_refuses_transformed_format():
    class _SubLinear(FormatLinear):
        pass

    class _SubSwitch(FormatSwitchLinear):
        pass

    class _SubEmbedding(FormatEmbedding):
        pass

    _, lin_tensors = _oqh_linear(256, 64, seed=60)
    _, sw_tensors = _oqh_switch(256, 64, 8, seed=61)
    with testing.verified("example"):
        with pytest.raises(TypeError, match="does not support transformed format"):
            _SubLinear(oqh.OQH_AFFINE, 4, 128, 256, lin_tensors)
        with pytest.raises(TypeError, match="does not support transformed format"):
            _SubSwitch(oqh.OQH_AFFINE, 4, 128, 256, sw_tensors)
        with pytest.raises(TypeError, match="does not support transformed format"):
            _SubEmbedding(oqh.OQH_AFFINE, 4, 128, 256, lin_tensors)

    # Plain formats remain accepted on subclasses.
    _, aff_tensors = _affine_linear(256, 64, seed=62)
    with testing.verified("example"):
        plain_sub = _SubLinear("affine", 4, 64, 256, aff_tensors)
    assert type(plain_sub) is _SubLinear


@pytest.mark.parametrize("base_cls", [SwitchGLU, VLMSwitchGLU], ids=["lm", "vlm"])
def test_custom_switch_glu_subclass_preserves_call_with_transformed_weights(base_cls):
    qwen35_moe_gate_up._ensure_call_patch()
    calls = []

    class _CustomSwitchGLU(base_cls):
        _omlx_gate_up_fusion = True

        def __call__(self, x, indices, *args, **kwargs):
            calls.append(tuple(x.shape))
            return super().__call__(x, indices, *args, **kwargs)

    class _DerivedCustomSwitchGLU(_CustomSwitchGLU):
        pass

    for cls in (_CustomSwitchGLU, _DerivedCustomSwitchGLU):
        calls.clear()
        g_oqh, _ = _oqh_switch(256, 128, 8, seed=70)
        u_oqh, _ = _oqh_switch(256, 128, 8, seed=71)
        u_oqh["oqh_signs"] = g_oqh["oqh_signs"]
        d_oqh, _ = _oqh_switch(128, 256, 8, seed=72)
        glu = cls(256, 128, 8)
        glu.gate_proj, glu.up_proj, glu.down_proj = g_oqh, u_oqh, d_oqh
        glu.eval()
        x = mx.random.normal((1, 2, 256)).astype(mx.bfloat16)
        idx = mx.random.randint(0, 8, (1, 2, 2)).astype(mx.uint32)
        ref = glu(x, idx)
        mx.eval(ref)
        assert calls and calls[0] == (1, 2, 256)

        calls.clear()
        qwen35_moe_gate_up._fuse_one(glu)
        got = glu(x, idx)
        mx.eval(got)
        assert calls and calls[0] == (1, 2, 256)
        assert type(glu) is cls
        assert mx.allclose(got, ref, rtol=1e-2, atol=1e-2).item()

    # Subclass without custom __call__ still binds the transformed fast path.
    class _PlainSubSwitchGLU(base_cls):
        _omlx_gate_up_fusion = True

    g_oqh, _ = _oqh_switch(256, 128, 8, seed=73)
    u_oqh, _ = _oqh_switch(256, 128, 8, seed=74)
    u_oqh["oqh_signs"] = g_oqh["oqh_signs"]
    d_oqh, _ = _oqh_switch(128, 256, 8, seed=75)
    plain_sub = _PlainSubSwitchGLU(256, 128, 8)
    plain_sub.gate_proj, plain_sub.up_proj, plain_sub.down_proj = g_oqh, u_oqh, d_oqh
    plain_sub.eval()
    qwen35_moe_gate_up._fuse_one(plain_sub)
    assert getattr(type(plain_sub), "_omlx_transformed_switch", False)


def test_transformed_linear_one_row_warm_skips_dot_fast_and_packed_name(monkeypatch):
    """Merge #2 + oQh: a transformed FormatLinear (including custom codes_name)
    records weight_key, pre-marks no_dot_fast/no_dense_fast on _keep(), never
    binds _fast/_fast2 during 1-row warm-up, and makes zero _register_dot_fast
    or packed_name() calls on warm 1-row and multi-row hits."""

    class _CodesOqhAffine(oqh.OqhAffineFormat):
        mode = "oqh_affine_codes_test"
        codes_name = "codes"

        def layout(self, k, n, bits, group_size):
            lay = super().layout(k, n, bits, group_size)
            if lay is None:
                return None
            out = dict(lay)
            out["codes"] = out.pop("weight")
            return out

        def decode(self, tensors, bits, group_size):
            remapped = dict(tensors)
            remapped["weight"] = remapped.pop("codes")
            return super().decode(remapped, bits, group_size)

    lin_w, _ = _oqh_linear(256, 64, seed=80)
    raw_c, _ = _group_case(256, 64, 4, 128, seed=81)
    tensors_c = dict(raw_c)
    tensors_c["codes"] = tensors_c.pop("weight")
    with testing.verified("example"):
        lin_c = FormatLinear(_CodesOqhAffine(), 4, 128, 256, tensors_c)

    for layer, expected_key in ((lin_w, "weight"), (lin_c, "codes")):
        linear_mod.prepare_layer_plans(layer)
        assert layer._fast is None and layer._fast2 is None
        kept = layer.__dict__[linear_mod._KEPT]
        assert kept.weight_key == expected_key
        assert kept._slots.get("no_dot_fast") is True
        assert kept._slots.get("no_dense_fast") is True

        x1 = (mx.random.normal((1, 1, 256)) * 0.5).astype(mx.bfloat16)
        x3 = (mx.random.normal((1, 3, 256)) * 0.5).astype(mx.bfloat16)
        # Ensure both 1-row and 3-row graphs are warm before spying.
        mx.eval(layer(x1), layer(x3))
        assert layer._fast is None and layer._fast2 is None

        reg_dot_calls = []
        packed_calls = []
        orig_reg_dot = FormatLinear._register_dot_fast
        orig_packed = linear_mod.packed_name

        def spy_reg_dot(self, k, p, sig, _calls=reg_dot_calls, _orig=orig_reg_dot):
            _calls.append(sig)
            return _orig(self, k, p, sig)

        def spy_packed(mod, _calls=packed_calls, _orig=orig_packed):
            _calls.append(mod)
            return _orig(mod)

        with monkeypatch.context() as patch:
            patch.setattr(FormatLinear, "_register_dot_fast", spy_reg_dot)
            patch.setattr(linear_mod, "packed_name", spy_packed)
            y1 = layer(x1)
            y3 = layer(x3)
            mx.eval(y1, y3)

        assert (
            not reg_dot_calls
        ), f"warm transformed hit called _register_dot_fast: {reg_dot_calls}"
        assert (
            not packed_calls
        ), f"warm transformed hit called packed_name: {len(packed_calls)}"
        assert layer._fast is None and layer._fast2 is None
        ref1 = testing.transform_reference(layer, x1)
        ref3 = testing.transform_reference(layer, x3)
        mx.eval(ref1, ref3)
        assert mx.allclose(y1, ref1, rtol=3e-2, atol=3e-2).item()
        assert mx.allclose(y3, ref3, rtol=3e-2, atol=3e-2).item()


def test_transformed_switch_warm_uses_weight_key_not_packed_name(monkeypatch):
    """Merge #2 + oQh: a warm transformed FormatSwitchLinear call reads
    self[kept.weight_key] with zero packed_name() calls."""
    sw, _ = _oqh_switch(256, 64, 8, seed=82)
    x = (mx.random.normal((1, 2, 1, 1, 256)) * 0.5).astype(mx.bfloat16)
    idx = mx.array([[[0, 1], [2, 3]]], dtype=mx.uint32)
    ref = sw(x, idx)
    mx.eval(ref)
    kept = sw.__dict__[linear_mod._KEPT]
    assert kept.weight_key == "weight"

    packed_calls = []
    orig_packed = linear_mod.packed_name

    def spy_packed(mod):
        packed_calls.append(mod)
        return orig_packed(mod)

    monkeypatch.setattr(linear_mod, "packed_name", spy_packed)
    got = sw(x, idx)
    mx.eval(got)
    assert (
        not packed_calls
    ), f"warm transformed switch called packed_name: {len(packed_calls)}"
    assert mx.array_equal(got, ref).item()


def test_routed_decode_fastpath_test_keeps_router_verify_consistent(monkeypatch):
    """pre-oqh-5eb678fb5 NIT-1: test_routed_decode_plain_hot_path_has_no_transform_checks
    must not leave Qwen3_5BatchInvariantForward._feed_forward unpatched while
    Qwen3_5MoeSparseMoeBlock._omlx_router_fused stays True."""
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    orig_ff = Qwen3_5BatchInvariantForward._feed_forward
    orig_q35_call = Qwen3_5MoeSparseMoeBlock.__call__
    orig_q35_fused = Qwen3_5MoeSparseMoeBlock.__dict__.get("_omlx_router_fused")
    orig_q3n_call = Qwen3NextSparseMoeBlock.__call__
    orig_q3n_fused = Qwen3NextSparseMoeBlock.__dict__.get("_omlx_router_fused")

    def _dummy_ff(self, layer, h):
        return orig_ff(self, layer, h)

    try:
        Qwen3_5BatchInvariantForward._feed_forward = _dummy_ff
        if "_omlx_router_fused" in Qwen3_5MoeSparseMoeBlock.__dict__:
            delattr(Qwen3_5MoeSparseMoeBlock, "_omlx_router_fused")
        if "_omlx_router_fused" in Qwen3NextSparseMoeBlock.__dict__:
            delattr(Qwen3NextSparseMoeBlock, "_omlx_router_fused")

        test_routed_decode_plain_hot_path_has_no_transform_checks(monkeypatch)
        assert getattr(
            Qwen3_5BatchInvariantForward._feed_forward, "_omlx_router_fused", False
        ) == getattr(Qwen3_5MoeSparseMoeBlock, "_omlx_router_fused", False)
    finally:
        Qwen3_5BatchInvariantForward._feed_forward = orig_ff
        Qwen3_5MoeSparseMoeBlock.__call__ = orig_q35_call
        Qwen3NextSparseMoeBlock.__call__ = orig_q3n_call
        if orig_q35_fused is None:
            if "_omlx_router_fused" in Qwen3_5MoeSparseMoeBlock.__dict__:
                delattr(Qwen3_5MoeSparseMoeBlock, "_omlx_router_fused")
        else:
            Qwen3_5MoeSparseMoeBlock._omlx_router_fused = orig_q35_fused
        if orig_q3n_fused is None:
            if "_omlx_router_fused" in Qwen3NextSparseMoeBlock.__dict__:
                delattr(Qwen3NextSparseMoeBlock, "_omlx_router_fused")
        else:
            Qwen3NextSparseMoeBlock._omlx_router_fused = orig_q3n_fused
