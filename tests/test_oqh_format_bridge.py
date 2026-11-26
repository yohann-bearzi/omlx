# SPDX-License-Identifier: Apache-2.0
"""Stock oQh checkpoint bridge and generalized producer fusion.

Covers:
1. Ported D10-verified family (qwen4_exp) stock oQh checkpoint loads routed
   switch_mlp stacks as FormatSwitchLinear(OqhAffineFormat) AND trunk affine
   layers as FormatLinear(AffineFormat), builds hc_fused plans, fuses
   qwen35_moe_gate_up, engages qwen35_moe_routed_decode, and matches upstream
   OqhAffineSwitchLinear outputs.
2. Non-verified/TP family (glm5_next), sharded loads, and disallowed stock
   switch stay on upstream OQH_MODULES + oqh.runtime.install.
3. Generalized producer fusion (transforms.install_producers /
   release_producers): _DualNorm (RMSNorm + memo hit with zero fast.rotate),
   _DualGatedNorm (GDN gated norm + handoff hit), _FusedMLP (act_pair_had +
   _call_prepared), SwitchGLU untouched, release_producers & LoRA release,
   and thread-local isolation.
4. Format-written oQh checkpoints (config["omlx_quant_formats"]) run
   transforms.install_producers automatically without corrupting checkpoint
   placeholders when oqh.loading's load hook is installed, and plain stock
   loads do not walk the model.
5. Both hook installation orders (format_load <-> oqh.loading) work cleanly.
6. Refusals: tampered/missing oqh_signs/oqh_biases/rscale, unknown
   omlx_oqh.version, and missing native fast kernels raise ValueError.
"""

from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats, lora_adapters, oq
from omlx.custom_kernels.oqh import fast
from omlx.formats import affine, kernels, linear, oqh, testing, transforms
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.oqh import loading as oqh_loading
from omlx.oqh.loading import _verified_format_family, install_load_hook
from omlx.oqh.modules import OQH_MODULES
from omlx.patches import (
    format_load,
    qwen35_moe_gate_up,
    qwen35_moe_router,
)
from omlx.patches import qwen35_moe_routed_decode as routed
from tests.test_formats_oqh import _group_case, _row_case

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _oqh_linear(k: int, n: int, bits=4, gs=128, block=256, seed=0, signs=None):
    tensors, _ = _group_case(k, n, bits, gs, seed=seed, block=block)
    if signs is not None:
        tensors["oqh_signs"] = mx.array(signs)
    reader = copy.copy(oqh.OQH_AFFINE)
    reader.block = block
    with testing.verified("example"):
        return FormatLinear(reader, bits, gs, k, tensors)


@pytest.fixture(scope="module")
def oqh_checkpoint(tmp_path_factory):
    from dataclasses import asdict

    from mlx.utils import tree_flatten

    from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config

    tmp_path = tmp_path_factory.mktemp("oqh_bridge")
    with pytest.MonkeyPatch.context() as mp:
        cfg = _tiny_config()
        from mlx_vlm.models.qwen4_exp.language import configure_ple_runtime
        from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

        # hc_count=4 and hc_lowrank=64 match Flash-Next's HC geometry so
        # hc_fused builds real FormatLinear plans on the bridged checkpoint.
        cfg.text_config.hidden_size = 256
        cfg.text_config.moe_intermediate_size = 384
        cfg.text_config.linear_key_head_dim = 32
        cfg.text_config.linear_value_head_dim = 32
        cfg.text_config.hc_count = 4
        cfg.text_config.hc_lowrank = 64
        source = tmp_path / "source"
        source.mkdir()
        configure_ple_runtime(source, mode="resident")
        mx.random.seed(7)
        model = Model(cfg)
        model.eval()
        weights = {
            k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
            for k, v in tree_flatten(model.parameters())
        }
        config = asdict(cfg)
        config["architectures"] = ["Qwen4ExpForConditionalGeneration"]
        (source / "config.json").write_text(json.dumps(config))
        mx.save_safetensors(str(source / "model.safetensors"), weights)
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
        )
        tokens = mx.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=mx.int32)
        mp.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
        mp.setattr("mlx_lm.tokenizer_utils.load", lambda *a, **kw: object())
        out = tmp_path / "oqh"
        oq.quantize_oq_streaming(
            str(source),
            str(out),
            4,
            hadamard=True,
            stream_calibration=True,
            imatrix_num_samples=2,
            imatrix_seq_length=6,
            sensitivity_map_override={0: 1, 1: 1},
        )
    return out, tokens


def _load_oqh(out, *, allow_stock_switch=True):
    from mlx_vlm.utils import load_model

    from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime

    format_load.apply_format_load_patch()
    install_load_hook()
    configure_qwen4_exp_runtime(out, mode="resident", mtp_enabled=False)
    with format_load.loading(out, by_loader=True) as checkpoint:
        if not allow_stock_switch:
            checkpoint._allow_stock_switch = False
            checkpoint.verified = False
            checkpoint.verified_modes = frozenset()
        model = load_model(out, lazy=False, strict=True)
    return model


def test_qwen4_exp_stock_oqh_bridges_experts_and_trunk_format_layers(oqh_checkpoint):
    out, tokens = oqh_checkpoint
    model = _load_oqh(out, allow_stock_switch=True)

    switches = [
        m for _, m in model.named_modules() if isinstance(m, FormatSwitchLinear)
    ]
    assert len(switches) == 6  # two decoder layers, gate/up/down
    assert all((m.fmt.mode, m.group_size) == ("oqh_affine", 128) for m in switches)
    assert all(m["oqh_signs"].ndim == 2 for m in switches)

    # Trunk affine layers load as FormatLinear (enabling hc_fused, GDN prework, etc.)
    trunk_linears = [m for _, m in model.named_modules() if isinstance(m, FormatLinear)]
    assert trunk_linears, "expected trunk affine layers to load as FormatLinear"
    assert all(isinstance(m.fmt, affine.AffineFormat) for m in trunk_linears)
    assert not any(isinstance(m, nn.QuantizedLinear) for _, m in model.named_modules())

    # Gate+up fusion shares readers and fuses both SwitchGLUs.
    switch_glus = [
        m
        for _, m in model.named_modules()
        if hasattr(m, "gate_proj") and isinstance(m.gate_proj, FormatSwitchLinear)
    ]
    assert len(switch_glus) == 2
    for glu in switch_glus:
        assert glu.gate_proj.fmt is glu.up_proj.fmt
    assert qwen35_moe_gate_up.apply_qwen35_moe_gate_up_fusion(model) == 2

    # Forward matches upstream OqhAffineSwitchLinear cleanly.
    upstream = _load_oqh(out, allow_stock_switch=False)
    assert not any(
        isinstance(m, FormatSwitchLinear) for _, m in upstream.named_modules()
    )
    assert sum(isinstance(m, OQH_MODULES) for _, m in upstream.named_modules()) == 6

    logits_if = model(tokens).logits
    logits_up = upstream(tokens).logits
    mx.eval(logits_if, logits_up)
    assert mx.isfinite(logits_if).all().item()
    diff = mx.abs(logits_if.astype(mx.float32) - logits_up.astype(mx.float32))
    scale = max(float(mx.abs(logits_up.astype(mx.float32)).max().item()), 1e-6)
    assert float(diff.max().item()) / scale < 2e-2


def test_qwen4_exp_stock_oqh_engages_hc_fused_and_routed_decode(oqh_checkpoint):
    from mlx_vlm.models.qwen4_exp import hc_fused

    out, tokens = oqh_checkpoint
    qwen35_moe_router.apply_qwen35_moe_router_patch()
    routed.apply_qwen35_moe_routed_decode_patch()

    model = _load_oqh(out, allow_stock_switch=True)
    # HC layers built FormatLinear hc_fused plans.
    assert hc_fused.rebuild_plans(model) > 0
    assert qwen35_moe_gate_up.apply_qwen35_moe_gate_up_fusion(model) == 2

    # Single-token decode runs cleanly through the bridged model.
    logits = model(tokens[:, :1]).logits
    mx.eval(logits)
    assert mx.isfinite(logits).all().item()


def test_bridged_oqh_layers_recast_to_float16_without_prior_formats_load(
    tmp_path, monkeypatch
):
    pure = tmp_path / "pure_oqh"
    pure.mkdir()
    (pure / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
                "omlx_oqh": {"version": 1},
            }
        )
    )
    tensors, _ = _group_case(256, 64, 4, 128, seed=0)
    weights = {f"proj.{k}": v for k, v in tensors.items()}

    class _Pure(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(256, 64, bias=False)

    monkeypatch.setattr(formats, "_loaded", False)
    monkeypatch.setattr(formats, "_BUILTIN_MODES", set())
    monkeypatch.setattr(formats, "_FORMATS", {})
    monkeypatch.setattr(formats, "_ORIGINS", {})
    monkeypatch.setattr(formats, "_SKIPPED", {})

    install_load_hook()
    model = _Pure()
    with format_load.loading(pure):
        model.load_weights(list(weights.items()), strict=True)
    assert isinstance(model.proj, FormatLinear)
    model.set_dtype(mx.float16)
    assert model.proj["oqh_scales"].dtype == mx.float16
    assert model.proj["oqh_biases"].dtype == mx.float16
    assert model.proj["oqh_signs"].dtype == mx.float16


def test_sharded_load_and_non_verified_family_stay_on_upstream(
    oqh_checkpoint, tmp_path, monkeypatch
):
    from mlx_vlm.utils import load_model

    from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime
    from tests.test_oqh import _tiny_glm5_next

    assert _verified_format_family({"model_type": " Qwen4-Exp "})
    assert _verified_format_family({"text_config": {"model_type": " Qwen4-Exp-Text "}})
    assert not _verified_format_family({"model_type": "glm5_next"})

    out, _ = oqh_checkpoint
    install_load_hook()
    configure_qwen4_exp_runtime(out, mode="resident", mtp_enabled=False)

    monkeypatch.setattr(format_load._SHARDED, "entry", ("s", "sharded"), raising=False)
    with format_load.loading(out):
        sharded_model = load_model(out, lazy=False, strict=True)
    assert not any(
        isinstance(m, FormatSwitchLinear) for _, m in sharded_model.named_modules()
    )
    assert (
        sum(isinstance(m, OQH_MODULES) for _, m in sharded_model.named_modules()) == 6
    )

    monkeypatch.setattr(format_load._SHARDED, "entry", None, raising=False)
    with format_load.loading(out, by_loader=True):
        cp = format_load._current()
        load_model(out, lazy=False, strict=True)
        assert cp.verified_modes == frozenset({"affine", "oqh", "oqh_affine"})

    # Non-verified / TP family (glm5_next) stays on upstream OQH_MODULES.
    glm_src, _ = _tiny_glm5_next(tmp_path, monkeypatch)
    glm_out = tmp_path / "glm_oqh"
    oq.quantize_oq_streaming(
        str(glm_src),
        str(glm_out),
        4,
        hadamard=True,
        imatrix_num_samples=2,
        imatrix_seq_length=6,
        sensitivity_map_override={0: 1},
    )
    with format_load.loading(glm_out, by_loader=True):
        glm_model = load_model(glm_out, lazy=False, strict=True)
    assert not any(
        isinstance(m, FormatSwitchLinear) for _, m in glm_model.named_modules()
    )
    assert any(isinstance(m, OQH_MODULES) for _, m in glm_model.named_modules())


def test_generalized_producer_fusion_norm_gated_norm_and_dense_mlp(monkeypatch):
    class _GatedNorm(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.weight = mx.ones((dim,), mx.bfloat16)
            self.eps = 1e-6

        def __call__(self, x, gate=None):
            y = mx.fast.rms_norm(x, self.weight, self.eps)
            return y if gate is None else y * nn.silu(gate)

    class _LinearAttn(nn.Module):
        def __init__(self, signs_in):
            super().__init__()
            self.in_proj_qkv = _oqh_linear(256, 64, block=256, seed=1, signs=signs_in)
            self.in_proj_z = _oqh_linear(256, 64, block=256, seed=2, signs=signs_in)
            self.norm = _GatedNorm(128)
            self.out_proj = _oqh_linear(256, 256, block=128, seed=3)

    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    def _make_qwen_mlp(signs_mlp):
        mlp = Qwen3_5MLP(256, 128)
        mlp.gate_proj = _oqh_linear(256, 128, block=256, seed=4, signs=signs_mlp)
        mlp.up_proj = _oqh_linear(256, 128, block=256, seed=5, signs=signs_mlp)
        mlp.down_proj = _oqh_linear(128, 256, block=128, seed=6)
        return mlp

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            signs_in = _group_case(256, 64, 4, 128, seed=10, block=256)[0]["oqh_signs"]
            signs_mlp = _group_case(256, 128, 4, 128, seed=20, block=256)[0][
                "oqh_signs"
            ]
            self.input_layernorm = nn.RMSNorm(256)
            self.input_layernorm.weight = mx.ones((256,), mx.bfloat16)
            self.linear_attn = _LinearAttn(signs_in)
            self.post_attention_layernorm = nn.RMSNorm(256)
            self.post_attention_layernorm.weight = mx.ones((256,), mx.bfloat16)
            self.mlp = _make_qwen_mlp(signs_mlp)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _Layer()

    model = _Model()
    mx.eval(model.parameters())
    mx.random.seed(99)
    x = (mx.random.normal((1, 2, 256)) * 0.5).astype(mx.bfloat16)
    g = (mx.random.normal((1, 2, 2, 128)) * 0.5).astype(mx.bfloat16)
    h = (mx.random.normal((1, 2, 2, 128)) * 0.5).astype(mx.bfloat16)
    mx.eval(x, g, h)

    # Reference outputs before producer fusion.
    transforms.clear()
    n1_ref = model.layer.input_layernorm(x)
    qkv_ref = model.layer.linear_attn.in_proj_qkv(n1_ref)
    z_ref = model.layer.linear_attn.in_proj_z(n1_ref)
    gn_ref = model.layer.linear_attn.norm(h, g).reshape(1, 2, 256)
    out_ref = model.layer.linear_attn.out_proj(gn_ref)
    n2_ref = model.layer.post_attention_layernorm(x)
    mlp_ref = model.layer.mlp(n2_ref)
    mx.eval(qkv_ref, z_ref, out_ref, mlp_ref)

    installed = transforms.install_producers(model)
    assert installed >= 3
    assert isinstance(model.layer.input_layernorm, transforms._DualNorm)
    assert isinstance(model.layer.post_attention_layernorm, transforms._DualNorm)
    assert isinstance(model.layer.linear_attn.norm, transforms._DualGatedNorm)
    assert isinstance(model.layer.mlp, transforms._FusedMLP)

    # Spy on fast.rotate: every consumer must hit memo/handoff/act_pair_had with ZERO rotate calls!
    rotate_calls = []
    orig_rotate = fast.rotate
    monkeypatch.setattr(
        fast, "rotate", lambda *a, **kw: rotate_calls.append(a) or orig_rotate(*a, **kw)
    )

    transforms.clear()
    n1 = model.layer.input_layernorm(x)
    qkv = model.layer.linear_attn.in_proj_qkv(n1)
    z = model.layer.linear_attn.in_proj_z(n1)
    gn = model.layer.linear_attn.norm(h, g).reshape(1, 2, 256)
    out = model.layer.linear_attn.out_proj(gn)
    n2 = model.layer.post_attention_layernorm(x)
    mlp = model.layer.mlp(n2)
    mx.eval(qkv, z, out, mlp)

    assert not rotate_calls, f"expected 0 fast.rotate calls, got {len(rotate_calls)}"
    for got, ref, name in (
        (qkv, qkv_ref, "qkv"),
        (z, z_ref, "z"),
        (out, out_ref, "out_proj"),
        (mlp, mlp_ref, "mlp"),
    ):
        err = float(
            mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
        )
        scale = max(float(mx.abs(ref.astype(mx.float32)).max().item()), 1e-6)
        assert err / scale < 5e-2, f"{name} err/scale={err / scale}"

    # Thread-local memo isolation across concurrent threads.
    def worker(seed):
        mx.random.seed(seed)
        xw = (mx.random.normal((1, 2, 256)) * 0.5).astype(mx.bfloat16)
        nw = model.layer.post_attention_layernorm(xw)
        yw = model.layer.mlp(nw)
        mx.eval(yw)
        return yw

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1, f2 = pool.submit(worker, 101), pool.submit(worker, 202)
        assert mx.isfinite(f1.result()).all().item()
        assert mx.isfinite(f2.result()).all().item()

    # wrapped_only release restores wrapped _FusedMLP and _DualGatedNorm while
    # keeping _DualNorm.
    model.layer.mlp.gate_proj = lora_adapters.LoRALinear(
        model.layer.mlp.gate_proj,
        mx.zeros((4, 256), mx.bfloat16),
        mx.zeros((128, 4), mx.bfloat16),
    )
    model.layer.linear_attn.out_proj = lora_adapters.LoRALinear(
        model.layer.linear_attn.out_proj,
        mx.zeros((4, 256), mx.bfloat16),
        mx.zeros((256, 4), mx.bfloat16),
    )
    released = transforms.release_producers(model, wrapped_only=True)
    assert released == 2
    assert not isinstance(model.layer.mlp, transforms._FusedMLP)
    assert not isinstance(model.layer.linear_attn.norm, transforms._DualGatedNorm)
    assert isinstance(model.layer.post_attention_layernorm, transforms._DualNorm)
    # Full release restores every remaining producer wrapper.
    released_all = transforms.release_producers(model)
    assert released_all >= 2
    assert not isinstance(model.layer.input_layernorm, transforms._DualNorm)
    assert not isinstance(model.layer.post_attention_layernorm, transforms._DualNorm)
    assert not isinstance(model.layer.linear_attn.norm, transforms._DualGatedNorm)


def test_dual_gated_norm_skips_sigmoid_and_mismatched_dtypes():
    class _SigmoidNorm(nn.Module):
        def __init__(self, dim=128, activation="sigmoid"):
            super().__init__()
            self.weight = mx.ones((dim,), mx.bfloat16)
            self.eps = 1e-6
            self.activation = activation

        def __call__(self, x, gate=None):
            y = mx.fast.rms_norm(x, self.weight, self.eps)
            if gate is None:
                return y
            return y * (
                mx.sigmoid(gate) if self.activation == "sigmoid" else nn.silu(gate)
            )

    class Glm5NextRMSNormGated(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.weight = mx.ones((dim,), mx.bfloat16)
            self.eps = 1e-6

        def __call__(self, x, gate=None):
            y = mx.fast.rms_norm(x, self.weight, self.eps)
            return y if gate is None else y * mx.sigmoid(gate)

    class _Block(nn.Module):
        def __init__(self, norm):
            super().__init__()
            self.linear_attn = nn.Module()
            self.linear_attn.norm = norm
            self.linear_attn.out_proj = _oqh_linear(256, 256, block=128, seed=3)

    # Sigmoid-gated norms (Qwen4ExpRMSNormGated default & Glm5NextRMSNormGated)
    # must NOT be rebased onto the SiLU-only fast.gnorm_had kernel.
    sig_model = _Block(_SigmoidNorm(128, activation="sigmoid"))
    assert transforms.install_producers(sig_model) == 0
    assert not isinstance(sig_model.linear_attn.norm, transforms._DualGatedNorm)

    glm_model = _Block(Glm5NextRMSNormGated(128))
    assert transforms.install_producers(glm_model) == 0
    assert not isinstance(glm_model.linear_attn.norm, transforms._DualGatedNorm)

    # SiLU gated norm rebases, and falls back cleanly on fp32 gate or wrong head_dim.
    silu_model = _Block(_SigmoidNorm(128, activation="silu"))
    assert transforms.install_producers(silu_model) == 1
    assert isinstance(silu_model.linear_attn.norm, transforms._DualGatedNorm)
    mx.random.seed(12)
    x_bf = (mx.random.normal((1, 2, 2, 128)) * 0.5).astype(mx.bfloat16)
    g_f32 = (mx.random.normal((1, 2, 2, 128)) * 0.5).astype(mx.float32)
    ref_f32 = mx.fast.rms_norm(
        x_bf, silu_model.linear_attn.norm.weight, 1e-6
    ) * nn.silu(g_f32)
    got_f32 = silu_model.linear_attn.norm(x_bf, g_f32)
    mx.eval(ref_f32, got_f32)
    assert mx.array_equal(got_f32, ref_f32).item()

    # Wrong head dim (4 x 64 == 256 != head_dim 128) also falls back safely.
    silu_model.linear_attn.norm.weight = mx.ones((64,), mx.bfloat16)
    x_64 = (mx.random.normal((1, 2, 4, 64)) * 0.5).astype(mx.bfloat16)
    g_64 = (mx.random.normal((1, 2, 4, 64)) * 0.5).astype(mx.bfloat16)
    got_64 = silu_model.linear_attn.norm(x_64, g_64)
    mx.eval(got_64)
    assert got_64.shape == (1, 2, 4, 64)


def test_fused_mlp_allowlists_qwen3_5_and_qwen3_next_mlps(monkeypatch):
    from mlx_lm.models import qwen3_5 as qwen3_5_lm
    from mlx_lm.models.qwen3_next import Qwen3NextMLP
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeMLP

    assert Qwen3_5MoeMLP is Qwen3_5MLP
    assert qwen3_5_lm.MLP is Qwen3NextMLP

    for idx, mlp_cls in enumerate((Qwen3_5MLP, Qwen3NextMLP)):
        signs = _group_case(256, 128, 4, 128, seed=31 + idx, block=256)[0]["oqh_signs"]
        mlp = mlp_cls(256, 128)
        mlp.gate_proj = _oqh_linear(256, 128, block=256, seed=32 + idx, signs=signs)
        mlp.up_proj = _oqh_linear(256, 128, block=256, seed=33 + idx, signs=signs)
        mlp.down_proj = _oqh_linear(128, 256, block=128, seed=34 + idx)
        mx.eval(mlp.parameters())
        mx.random.seed(55 + idx)
        x_bf = (mx.random.normal((1, 2, 256)) * 0.5).astype(mx.bfloat16)
        x_f32 = x_bf.astype(mx.float32)
        ref_bf = mlp(x_bf)
        ref_f32 = mlp(x_f32)
        mx.eval(ref_bf, ref_f32)

        assert transforms.install_producers(mlp) == 1
        assert isinstance(mlp, transforms._FusedMLP)

        # Standalone MLP (no preceding _DualNorm): gate and up share one
        # block-256 rotation while down_proj fuses into act_pair_had (0
        # block-128 rotate calls).
        rotate_calls = []
        orig_rotate = fast.rotate
        monkeypatch.setattr(
            fast,
            "rotate",
            lambda *a, _c=rotate_calls, _fn=orig_rotate, **kw: _c.append(a)
            or _fn(*a, **kw),
        )
        transforms.clear()
        got_bf = mlp(x_bf)
        mx.eval(got_bf)
        assert len(rotate_calls) == 1 and rotate_calls[0][2] == 256
        monkeypatch.undo()

        got_f32 = mlp(x_f32)
        mx.eval(got_f32)
        for got, ref in ((got_bf, ref_bf), (got_f32, ref_f32)):
            err = float(
                mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
            )
            scale = max(float(mx.abs(ref.astype(mx.float32)).max().item()), 1e-6)
            assert err / scale < 5e-2

        # Empty activation guard returns cleanly without calling _omlx_tx_act_pair.
        act_calls = []
        orig_act = mlp._omlx_tx_act_pair
        object.__setattr__(
            mlp,
            "_omlx_tx_act_pair",
            lambda *a, _c=act_calls, _fn=orig_act, **kw: _c.append(a) or _fn(*a, **kw),
        )
        empty = mx.zeros((0, 256), mx.bfloat16)
        out_empty = mlp(empty)
        mx.eval(out_empty)
        assert out_empty.shape == (0, 256)
        assert not act_calls

        # Release restores the exact concrete class so re-install succeeds.
        assert transforms.release_producers(mlp) == 1
        assert type(mlp) is mlp_cls
        assert transforms.install_producers(mlp) == 1
        assert isinstance(mlp, transforms._FusedMLP)


def test_dual_norm_skips_custom_rmsnorm_subclass():
    class _CenteredNorm(nn.RMSNorm):
        def __call__(self, x):
            return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            signs = _group_case(256, 64, 4, 128, seed=41, block=256)[0]["oqh_signs"]
            self.input_layernorm = _CenteredNorm(256)
            self.input_layernorm.weight = mx.zeros((256,), mx.bfloat16)
            self.self_attn = nn.Module()
            self.self_attn.q_proj = _oqh_linear(
                256, 64, block=256, seed=42, signs=signs
            )

    layer = _Layer()
    assert transforms.install_producers(layer) == 0
    assert not isinstance(layer.input_layernorm, transforms._DualNorm)


def test_install_load_hook_thread_safe(monkeypatch):
    def fresh_load(self, file_or_weights, strict: bool = True):
        return (self, file_or_weights, strict)

    monkeypatch.setattr(nn.Module, "load_weights", fresh_load)
    monkeypatch.setattr(oqh_loading, "_installed", False)

    # Confirm the lock exists and protects concurrent callers from double-wrapping.
    assert hasattr(oqh_loading, "_INSTALL_LOCK")
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        oqh_loading.install_load_hook()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert getattr(fresh_load, "_omlx_oqh_load_hook", False)
    assert (
        fresh_load.__wrapped__.__code__
        is not oqh_loading._through_load_weights.__code__
    )


def test_format_written_oqh_runs_install_producers_and_stock_load_is_not_walked(
    tmp_path, monkeypatch
):
    ckpt_dir = tmp_path / "fmt_oqh"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
                "quantization": {"bits": 4, "group_size": 128, "mode": "oqh_affine"},
                "omlx_verification": {"version": 1, "verified": True},
                "omlx_quant_formats": {
                    "oqh_affine": {
                        "plugin": None,
                        "contract": 1,
                        "record": {
                            "transform": {
                                "version": 1,
                                "kind": "signed_wht",
                                "block": 256,
                            }
                        },
                    }
                },
            }
        )
    )
    tensors, _ = _group_case(256, 64, 4, 128, seed=7, block=256)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(256)
            self.input_layernorm.weight = mx.ones((256,), mx.bfloat16)
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(256, 64, bias=False)
            self.post_attention_layernorm = nn.RMSNorm(256)
            self.post_attention_layernorm.weight = mx.ones((256,), mx.bfloat16)

    def fake_load(path):
        model = _Block()
        model.self_attn.q_proj = model.self_attn.q_proj.to_quantized(
            group_size=128, bits=4, mode="oqh_affine"
        )
        weights = [(f"self_attn.q_proj.{k}", v) for k, v in tensors.items()]
        model.load_weights(weights, strict=False)
        return model

    format_load.apply_format_load_patch()
    fake_mod = SimpleNamespace(__name__="mlx_vlm.utils", load_model=fake_load)
    install_load_hook()
    format_load._wrap_loader(fake_mod, "load_model")

    # Format-written oQh load keeps checkpoint.built intact (oqh.loading does NOT
    # overwrite FormatLinear) and automatically installs producers.
    model = fake_mod.load_model(ckpt_dir)
    assert isinstance(model.self_attn.q_proj, FormatLinear)
    assert isinstance(model.input_layernorm, transforms._DualNorm)

    # Plain stock checkpoint load does NOT call install_producers.
    stock_dir = tmp_path / "plain_stock"
    stock_dir.mkdir()
    (stock_dir / "config.json").write_text(
        json.dumps(
            {"model_type": "llama", "quantization": {"bits": 4, "group_size": 64}}
        )
    )
    called = []
    monkeypatch.setattr(
        transforms, "install_producers", lambda m: called.append(m) or 0
    )

    def fake_stock_load(path):
        return _Block()

    stock_mod = SimpleNamespace(__name__="mlx_vlm.utils", load_model=fake_stock_load)
    format_load._wrap_loader(stock_mod, "load_model")
    stock_mod.load_model(stock_dir)
    assert not called


def test_oqh_refusals_bad_version_missing_native_and_tampered_tensors(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="version 99"):
        oqh_loading.require_native({"omlx_oqh": {"version": 99}})

    monkeypatch.setattr(fast, "has_native", lambda: False)
    with pytest.raises(ValueError, match="native kernels"):
        oqh_loading.require_native({"omlx_oqh": {"version": 1}})
    monkeypatch.undo()

    pure = tmp_path / "bad_oqh"
    pure.mkdir()
    (pure / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
                "omlx_oqh": {"version": 1},
            }
        )
    )
    tensors, _ = _group_case(256, 64, 4, 128, seed=0)

    class _Pure(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(256, 64, bias=False)

    install_load_hook()
    # Missing oqh_biases
    missing_biases = {f"proj.{k}": v for k, v in tensors.items() if k != "oqh_biases"}
    with format_load.loading(pure), pytest.raises(ValueError, match="oqh_biases"):
        _Pure().load_weights(list(missing_biases.items()), strict=True)

    # Tampered oqh_signs shape (invalid block 7)
    bad_signs = {f"proj.{k}": v for k, v in tensors.items()}
    bad_signs["proj.oqh_signs"] = mx.ones((3, 7), mx.bfloat16)
    with format_load.loading(pure), pytest.raises(ValueError):
        _Pure().load_weights(list(bad_signs.items()), strict=True)


def test_both_hook_installation_orders_bridge_and_fuse_identically(
    oqh_checkpoint, monkeypatch
):
    from mlx_vlm.utils import load_model

    from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime

    out, tokens = oqh_checkpoint
    configure_qwen4_exp_runtime(out, mode="resident", mtp_enabled=False)

    # Order 1: format_load already installed, then reinstall oqh load hook.
    format_load.apply_format_load_patch()
    orig_load_weights = nn.Module.load_weights
    monkeypatch.setattr(oqh_loading, "_installed", False)
    try:
        install_load_hook()
        m1 = load_model(out, lazy=False, strict=True)
        assert qwen35_moe_gate_up.apply_qwen35_moe_gate_up_fusion(m1) == 2

        # Order 2: oqh load hook already installed, then re-run format_load.
        install_load_hook()
        format_load.apply_format_load_patch()
        m2 = load_model(out, lazy=False, strict=True)
        assert qwen35_moe_gate_up.apply_qwen35_moe_gate_up_fusion(m2) == 2
    finally:
        nn.Module.load_weights = orig_load_weights

    for m in (m1, m2):
        assert sum(isinstance(x, FormatSwitchLinear) for _, x in m.named_modules()) >= 6
    l1, l2 = m1(tokens).logits, m2(tokens).logits
    mx.eval(l1, l2)
    assert mx.array_equal(l1, l2).item()


def test_stock_oqh_bridge_warms_format_linear_at_load(tmp_path, monkeypatch):
    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    pure = tmp_path / "stock_oqh_warm"
    pure.mkdir()
    (pure / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "dtype": "bfloat16",
                },
                "omlx_oqh": {"version": 1},
            }
        )
    )
    aff_t, _ = _group_case(256, 64, 4, 128, seed=11)
    row_t, _ = _row_case(256, 64, 4, seed=12)
    parts = [_group_case(256, 64, 4, 128, seed=20 + e)[0] for e in range(8)]
    sw_t = {
        "weight": mx.stack([p["weight"] for p in parts]),
        "oqh_scales": mx.stack([p["oqh_scales"] for p in parts]),
        "oqh_biases": mx.stack([p["oqh_biases"] for p in parts]),
        "oqh_signs": parts[0]["oqh_signs"],
    }
    weights = {
        **{f"dense.{k}": v for k, v in aff_t.items()},
        **{f"dense_row.{k}": v for k, v in row_t.items()},
        **{f"switch.{k}": v for k, v in sw_t.items()},
    }

    class _StockOqh(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(256, 64, bias=False)
            self.dense_row = nn.Linear(256, 64, bias=False)
            self.switch = nn.Linear(256, 64, bias=False)

    install_load_hook()
    model = _StockOqh()
    with format_load.loading(pure):
        model.load_weights(list(weights.items()), strict=True)

    assert isinstance(model.dense, FormatLinear) and model.dense.mode == "oqh_affine"
    assert isinstance(model.dense_row, FormatLinear) and model.dense_row.mode == "oqh"
    assert (
        isinstance(model.switch, FormatSwitchLinear)
        and model.switch.mode == "oqh_affine"
    )

    linear.warm_model_dispatch(model, prefill_lengths=(32,), mtp_enabled=True)
    before = kernels.proof_count()

    for shape in ((1, 1, 256), (1, 4, 256), (1, 32, 256)):
        x = mx.zeros(shape, mx.bfloat16)
        y1 = model.dense(x)
        y2 = model.dense_row(x)
        mx.eval(y1, y2)
    ys = model.switch(
        mx.zeros((1, 1, 1, 256), mx.bfloat16), mx.zeros((1, 1), mx.uint32)
    )
    ys_sorted = model.switch(
        mx.zeros((32, 1, 256), mx.bfloat16),
        mx.zeros((32,), mx.uint32),
        sorted_indices=True,
    )
    mx.eval(ys, ys_sorted)
    assert kernels.proof_count() == before

    # Load-time warm failure must fail closed rather than silently swallowing.
    monkeypatch.setattr(
        format_load,
        "warm_layer",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("oqh warm boom")),
    )
    with (
        pytest.raises(RuntimeError, match="oqh warm boom"),
        format_load.loading(pure),
    ):
        _StockOqh().load_weights(list(weights.items()), strict=True)


def test_stock_oqh_bridge_survives_benign_model_config_override(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
                "omlx_oqh": {"version": 1},
            }
        )
    )
    with format_load.loading(tmp_path) as cp:
        cp.apply_model_config({"max_position_embeddings": 4096})
        assert cp._allow_stock_switch
        assert cp.verified
        assert cp.verified_modes == frozenset({"affine"})
        assert format_load._ported(cp)
        cp.apply_model_config({"omlx_oqh": {"version": 99}})
        assert not cp._allow_stock_switch
        assert not cp.verified
        assert cp.verified_modes == frozenset()
        assert not format_load._ported(cp)
    with format_load.loading(tmp_path) as cp:
        cp.apply_model_config({"model_type": "glm5_next"})
        assert not cp._allow_stock_switch
        assert not cp.verified
        assert cp.verified_modes == frozenset()
        assert not format_load._ported(cp)


def _assert_close_to_ref(got: mx.array, ref: mx.array, tol: float = 5e-2) -> None:
    mx.eval(got, ref)
    err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item())
    scale = max(float(mx.abs(ref.astype(mx.float32)).max().item()), 1e-6)
    assert err / scale < tol


def test_install_dense_mlp_skips_direct_array_param_mlp():
    """Review MEDIUM-1: single-arg MLP with direct mx.array param stays unfused."""

    class _ScaledDenseMLP(nn.Module):
        def __init__(self):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=61, block=256)[0]["oqh_signs"]
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=62, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=63, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=64)
            self.global_scale = mx.full((1,), 0.5, mx.bfloat16)

        def __call__(self, x):
            return (
                self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
                * self.global_scale
            )

    mlp = _ScaledDenseMLP()
    mx.eval(mlp.parameters())
    assert transforms.install_producers(mlp) == 0
    assert not isinstance(mlp, transforms._FusedMLP)

    mx.random.seed(65)
    x_bf = (mx.random.normal((1, 2, 256)) * 0.5).astype(mx.bfloat16)
    got = mlp(x_bf)
    ref = (
        transforms.transform_reference(
            mlp.down_proj,
            nn.silu(transforms.transform_reference(mlp.gate_proj, x_bf))
            * transforms.transform_reference(mlp.up_proj, x_bf),
        )
        * mlp.global_scale
    )
    _assert_close_to_ref(got, ref)


def test_install_dense_mlp_skips_bailing_omlx_swiglu_limit():
    """Review MEDIUM-2: bailing_hybrid _omlx_swiglu_limit MLP stays unfused."""

    def clamped_swiglu(gate, up, limit):
        return mx.minimum(nn.silu(gate), limit) * mx.clip(up, -limit, limit)

    class _BailingMLP(nn.Module):
        def __init__(self):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=66, block=256)[0]["oqh_signs"]
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=67, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=68, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=69)
            self._omlx_swiglu_limit = 5.0

        def __call__(self, x):
            return self.down_proj(
                clamped_swiglu(
                    self.gate_proj(x), self.up_proj(x), self._omlx_swiglu_limit
                )
            )

    mlp = _BailingMLP()
    mx.eval(mlp.parameters())
    assert transforms.install_producers(mlp) == 0
    assert not isinstance(mlp, transforms._FusedMLP)

    mx.random.seed(70)
    x_bf = (mx.random.normal((1, 2, 256)) * 5.0).astype(mx.bfloat16)
    got = mlp(x_bf)
    ref = transforms.transform_reference(
        mlp.down_proj,
        clamped_swiglu(
            transforms.transform_reference(mlp.gate_proj, x_bf),
            transforms.transform_reference(mlp.up_proj, x_bf),
            mlp._omlx_swiglu_limit,
        ),
    )
    _assert_close_to_ref(got, ref)


def test_install_dense_mlp_skips_negative_and_clamped_limits():
    """Review LOW-1 & allowlist ruling: negative, invalid, and clamped MLPs stay unfused."""

    class SwiGLU:
        def __init__(self, limit):
            self.limit = limit

        def __call__(self, x, gate):
            g = mx.minimum(gate, mx.array(self.limit, gate.dtype))
            u = mx.clip(
                x, -mx.array(self.limit, x.dtype), mx.array(self.limit, x.dtype)
            )
            return nn.silu(g) * u

    class _Glm5NextMLP(nn.Module):
        def __init__(self, limit):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=61, block=256)[0]["oqh_signs"]
            self.limit = limit
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=62, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=63, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=64)

        def __call__(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            if self.limit is not None:
                lim = mx.array(self.limit, gate.dtype)
                gate = mx.minimum(gate, lim)
                up = mx.clip(up, -lim, lim)
            return self.down_proj(nn.silu(gate) * up)

    for bad_limit in (
        -1.0,
        True,
        False,
        float("nan"),
        float("inf"),
        -float("inf"),
        "1",
    ):
        bad_mlp = _Glm5NextMLP(bad_limit)
        assert transforms.install_producers(bad_mlp) == 0
        assert not isinstance(bad_mlp, transforms._FusedMLP)
        bad_act_mlp = _Glm5NextMLP(None)
        bad_act_mlp.activation = SwiGLU(bad_limit)
        assert transforms.install_producers(bad_act_mlp) == 0
        assert not isinstance(bad_act_mlp, transforms._FusedMLP)

    # Even valid positive limit on non-allowlisted Glm5NextMLP stays unfused and exact.
    glm = _Glm5NextMLP(1.0)
    mx.eval(glm.parameters())
    assert transforms.install_producers(glm) == 0
    assert not isinstance(glm, transforms._FusedMLP)

    mx.random.seed(77)
    x_bf = (mx.random.normal((1, 2, 256)) * 5.0).astype(mx.bfloat16)
    g_ref = transforms.transform_reference(glm.gate_proj, x_bf)
    u_ref = transforms.transform_reference(glm.up_proj, x_bf)
    lim = mx.array(1.0, g_ref.dtype)
    ref_bf = transforms.transform_reference(
        glm.down_proj, nn.silu(mx.minimum(g_ref, lim)) * mx.clip(u_ref, -lim, lim)
    )
    _assert_close_to_ref(glm(x_bf), ref_bf)


def test_install_dense_mlp_skips_zero_and_nonzero_swiglu_limit():
    """Review LOW-2: swiglu_limit == 0.0 (falsy) and non-zero stay unfused."""

    class _VisionLikeMLP(nn.Module):
        def __init__(self, swiglu_limit):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=71, block=256)[0]["oqh_signs"]
            self.swiglu_limit = swiglu_limit
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=72, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=73, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=74)

        def __call__(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            if self.swiglu_limit is not None:
                lim = mx.array(self.swiglu_limit, up.dtype)
                up = mx.clip(up, -lim, lim)
            return self.down_proj(nn.silu(gate) * up)

    for limit in (0.0, 7.0):
        vision_like = _VisionLikeMLP(limit)
        assert transforms.install_producers(vision_like) == 0
        assert not isinstance(vision_like, transforms._FusedMLP)


def test_install_dense_mlp_skips_subclasses_and_spoofed_names():
    """Lead ruling: exact type match only; subclasses and name-spoofed classes stay unfused."""
    from mlx_lm.models.qwen3_next import Qwen3NextMLP
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    def _wire(mlp, seed):
        signs = _group_case(256, 128, 4, 128, seed=seed, block=256)[0]["oqh_signs"]
        mlp.gate_proj = _oqh_linear(256, 128, block=256, seed=seed + 1, signs=signs)
        mlp.up_proj = _oqh_linear(256, 128, block=256, seed=seed + 2, signs=signs)
        mlp.down_proj = _oqh_linear(128, 256, block=128, seed=seed + 3)
        return mlp

    class _SubQwen35MLP(Qwen3_5MLP):
        pass

    class _ScaledSubQwen35MLP(Qwen3_5MLP):
        def __init__(self, dim, hidden_dim):
            super().__init__(dim, hidden_dim)
            self.global_scale = mx.full((1,), 0.25, mx.bfloat16)

        def __call__(self, x):
            return super().__call__(x) * self.global_scale

    class _SubQwen3NextMLP(Qwen3NextMLP):
        pass

    class _SpoofedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            _wire(self, 120)

        def __call__(self, x):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    _SpoofedMLP.__module__ = Qwen3_5MLP.__module__
    _SpoofedMLP.__qualname__ = Qwen3_5MLP.__qualname__

    for sub in (
        _wire(_SubQwen35MLP(256, 128), 110),
        _wire(_SubQwen3NextMLP(256, 128), 115),
        _SpoofedMLP(),
    ):
        assert transforms.install_producers(sub) == 0
        assert not isinstance(sub, transforms._FusedMLP)

    scaled_sub = _wire(_ScaledSubQwen35MLP(256, 128), 125)
    mx.eval(scaled_sub.parameters())
    assert transforms.install_producers(scaled_sub) == 0
    assert not isinstance(scaled_sub, transforms._FusedMLP)

    mx.random.seed(129)
    x_bf = (mx.random.normal((1, 2, 256)) * 0.5).astype(mx.bfloat16)
    ref_scaled = (
        transforms.transform_reference(
            scaled_sub.down_proj,
            nn.silu(transforms.transform_reference(scaled_sub.gate_proj, x_bf))
            * transforms.transform_reference(scaled_sub.up_proj, x_bf),
        )
        * scaled_sub.global_scale
    )
    _assert_close_to_ref(scaled_sub(x_bf), ref_scaled)


def test_install_dense_mlp_drops_attribute_heuristics_and_dead_act_fn():
    """Review NIT & allowlist ruling: dead act_fn heuristic removed; custom act_fn stays unfused."""
    assert not hasattr(transforms, "_DENSE_MLP_CHILDREN")
    assert not hasattr(transforms, "_is_single_arg_swiglu_mlp")

    class _ActFnMLP(nn.Module):
        def __init__(self, act_fn):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=81, block=256)[0]["oqh_signs"]
            self.act_fn = act_fn
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=82, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=83, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=84)

        def __call__(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    silu_attr = _ActFnMLP(nn.silu)
    mx.eval(silu_attr.parameters())
    assert transforms.install_producers(silu_attr) == 0
    assert not isinstance(silu_attr, transforms._FusedMLP)

    mx.random.seed(85)
    x_bf = (mx.random.normal((1, 2, 256)) * 0.5).astype(mx.bfloat16)
    ref_silu = transforms.transform_reference(
        silu_attr.down_proj,
        nn.silu(transforms.transform_reference(silu_attr.gate_proj, x_bf))
        * transforms.transform_reference(silu_attr.up_proj, x_bf),
    )
    _assert_close_to_ref(silu_attr(x_bf), ref_silu)


def test_install_dense_mlp_skips_multi_arg_merger_bare_and_gelu():
    """Non-allowlisted multi-arg, merger, bare, and GELU MLPs stay unfused and exact."""

    class _InklingSharedExpertsDense(nn.Module):
        def __init__(self):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=71, block=256)[0]["oqh_signs"]
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=72, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=73, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=74)

        def __call__(self, x, gamma):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x) * gamma)

    class _ActFnMLP(nn.Module):
        def __init__(self, act_fn):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=81, block=256)[0]["oqh_signs"]
            self.act_fn = act_fn
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=82, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=83, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=84)

        def __call__(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class _HardcodedGeluMLP(nn.Module):
        def __init__(self):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=85, block=256)[0]["oqh_signs"]
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=86, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=87, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=88)

        def __call__(self, x):
            return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))

    class _DefaultExtraArgMLP(nn.Module):
        def __init__(self):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=89, block=256)[0]["oqh_signs"]
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=90, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=91, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=92)

        def __call__(self, x, mask=None):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class _VarArgsMLP(nn.Module):
        def __init__(self):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=93, block=256)[0]["oqh_signs"]
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=94, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=95, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=96)

        def __call__(self, *args):
            x = args[0]
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class _MergerMLP(nn.Module):
        def __init__(self):
            super().__init__()
            signs = _group_case(256, 128, 4, 128, seed=97, block=256)[0]["oqh_signs"]
            self.proj = nn.Linear(256, 256, bias=False)
            self.gate_proj = _oqh_linear(256, 128, block=256, seed=98, signs=signs)
            self.up_proj = _oqh_linear(256, 128, block=256, seed=99, signs=signs)
            self.down_proj = _oqh_linear(128, 256, block=128, seed=100)

        def __call__(self, x):
            x = self.proj(x)
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    mx.random.seed(77)
    x_bf = (mx.random.normal((1, 2, 256)) * 5.0).astype(mx.bfloat16)

    # Multi-arg InklingSharedExpertsDense stays unfused while preserving norm readers.
    class _BlockWithInkling(nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = nn.RMSNorm(256)
            self.post_attention_layernorm.weight = mx.ones((256,), mx.bfloat16)
            self.mlp = _InklingSharedExpertsDense()

    blk = _BlockWithInkling()
    mx.eval(blk.parameters())
    assert transforms.install_producers(blk) == 1
    assert isinstance(blk.post_attention_layernorm, transforms._DualNorm)
    assert not isinstance(blk.mlp, transforms._FusedMLP)
    gamma = mx.full((1, 2, 128), 0.5, mx.bfloat16)
    y_ink = blk.mlp(x_bf, gamma)
    ref_ink = transforms.transform_reference(
        blk.mlp.down_proj,
        nn.silu(transforms.transform_reference(blk.mlp.gate_proj, x_bf))
        * transforms.transform_reference(blk.mlp.up_proj, x_bf)
        * gamma,
    )
    assert y_ink.shape == (1, 2, 256)
    _assert_close_to_ref(y_ink, ref_ink)

    for skipped in (_DefaultExtraArgMLP(), _VarArgsMLP()):
        assert transforms.install_producers(skipped) == 0
        assert not isinstance(skipped, transforms._FusedMLP)

    merger = _MergerMLP()
    mx.eval(merger.parameters())
    assert transforms.install_producers(merger) == 0
    assert not isinstance(merger, transforms._FusedMLP)
    x_proj = merger.proj(x_bf)
    ref_merger = transforms.transform_reference(
        merger.down_proj,
        nn.silu(transforms.transform_reference(merger.gate_proj, x_proj))
        * transforms.transform_reference(merger.up_proj, x_proj),
    )
    _assert_close_to_ref(merger(x_bf), ref_merger)

    bare = nn.Module()
    signs = _group_case(256, 128, 4, 128, seed=101, block=256)[0]["oqh_signs"]
    bare.gate_proj = _oqh_linear(256, 128, block=256, seed=102, signs=signs)
    bare.up_proj = _oqh_linear(256, 128, block=256, seed=103, signs=signs)
    bare.down_proj = _oqh_linear(128, 256, block=128, seed=104)
    assert transforms.install_producers(bare) == 0
    assert not isinstance(bare, transforms._FusedMLP)

    for gelu_mlp in (_ActFnMLP(nn.gelu), _HardcodedGeluMLP()):
        mx.eval(gelu_mlp.parameters())
        assert transforms.install_producers(gelu_mlp) == 0
        assert not isinstance(gelu_mlp, transforms._FusedMLP)
        ref_gelu = transforms.transform_reference(
            gelu_mlp.down_proj,
            nn.gelu(transforms.transform_reference(gelu_mlp.gate_proj, x_bf))
            * transforms.transform_reference(gelu_mlp.up_proj, x_bf),
        )
        _assert_close_to_ref(gelu_mlp(x_bf), ref_gelu)


def test_dual_norms_skip_empty_and_clear_stale_handoff_on_fallback():
    class _GatedNorm(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.weight = mx.ones((dim,), mx.bfloat16)
            self.eps = 1e-6

        def __call__(self, x, gate=None):
            y = mx.fast.rms_norm(x, self.weight, self.eps)
            return y if gate is None else y * nn.silu(gate)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            signs = _group_case(256, 64, 4, 128, seed=111, block=256)[0]["oqh_signs"]
            self.input_layernorm = nn.RMSNorm(256)
            self.input_layernorm.weight = mx.ones((256,), mx.bfloat16)
            self.linear_attn = nn.Module()
            self.linear_attn.in_proj_qkv = _oqh_linear(
                256, 64, block=256, seed=112, signs=signs
            )
            self.linear_attn.norm = _GatedNorm(128)
            self.linear_attn.out_proj = _oqh_linear(256, 256, block=128, seed=113)

    blk = _Block()
    mx.eval(blk.parameters())
    assert transforms.install_producers(blk) == 2
    norm = blk.input_layernorm
    gnorm = blk.linear_attn.norm
    assert isinstance(norm, transforms._DualNorm)
    assert isinstance(gnorm, transforms._DualGatedNorm)

    # _DualNorm skips empty inputs without calling _omlx_tx_rmsnorm.
    rms_calls = []
    orig_rms = norm._omlx_tx_rmsnorm
    object.__setattr__(
        norm,
        "_omlx_tx_rmsnorm",
        lambda *a, **kw: rms_calls.append(a) or orig_rms(*a, **kw),
    )
    empty_norm = norm(mx.zeros((0, 256), mx.bfloat16))
    mx.eval(empty_norm)
    assert empty_norm.shape == (0, 256)
    assert not rms_calls

    # _DualGatedNorm skips empty inputs without calling _omlx_tx_gated_norm.
    gnorm_calls = []
    orig_gnorm = gnorm._omlx_tx_gated_norm
    object.__setattr__(
        gnorm,
        "_omlx_tx_gated_norm",
        lambda *a, **kw: gnorm_calls.append(a) or orig_gnorm(*a, **kw),
    )
    empty_x = mx.zeros((0, 1, 2, 128), mx.bfloat16)
    empty_g = mx.zeros((0, 1, 2, 128), mx.bfloat16)
    empty_out = gnorm(empty_x, empty_g)
    mx.eval(empty_out)
    assert empty_out.shape == (0, 1, 2, 128)
    assert not gnorm_calls

    # Stale matching handoff is cleared on fallback (gate=None, mismatched dtype,
    # and empty input), while an unrelated handoff key stays intact.
    mx.random.seed(114)
    h = (mx.random.normal((1, 2, 2, 128)) * 0.5).astype(mx.bfloat16)
    g = (mx.random.normal((1, 2, 2, 128)) * 0.5).astype(mx.bfloat16)
    key = gnorm._omlx_tx_handoff_key

    for fallback_call in (
        lambda: gnorm(h, None),
        lambda: gnorm(h, g.astype(mx.float32)),
        lambda: gnorm(empty_x, empty_g),
    ):
        transforms.clear()
        plain = gnorm(h, g)
        mx.eval(plain)
        fallback_out = fallback_call()
        mx.eval(fallback_out)
        assert transforms.take(key, plain.reshape(1, 2, 256)) is None

    transforms.clear()
    other_key = ("unrelated_handoff",)
    transforms.hand_off(other_key, mx.ones((1, 2, 256), mx.bfloat16))
    gnorm(h, None)
    assert transforms.take(other_key, mx.zeros((1, 2, 256), mx.bfloat16)) is not None
    transforms.clear()
