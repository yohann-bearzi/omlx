# SPDX-License-Identifier: Apache-2.0
"""Activation recast and resident expert accounting for ported Qwen."""

import json
import logging
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.cache.paged_ssd_cache import numerics_revision_for_model
from omlx.formats import testing
from omlx.formats.affine import AffineFormat
from omlx.formats.linear import FormatEmbedding, FormatLinear, FormatSwitchLinear
from omlx.patches import format_load, moe_expert_offload
from omlx.utils.model_loading import apply_activation_fp16


class Plugin(AffineFormat):
    mode = "fixed-side-dtype"


@pytest.mark.parametrize(
    "layer_cls", (FormatLinear, FormatSwitchLinear, FormatEmbedding)
)
@pytest.mark.parametrize("builtin", (True, False), ids=("stock", "plugin"))
def test_setting_recasts_builtin_tensors_and_keeps_plugin_guard(
    layer_cls, builtin, tmp_path
):
    mode = "affine" if builtin else Plugin.mode
    config = {"model_type": "qwen4_exp", "omlx_quant_formats": {mode: {}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    source = mx.arange(16 * 256).reshape(16, 256).astype(mx.bfloat16)
    arrays = mx.quantize(source, group_size=64, bits=4)
    if layer_cls is FormatSwitchLinear:
        arrays = [mx.stack([a, a]) for a in arrays]
    tensors = dict(zip(("weight", "scales", "biases"), arrays))
    with testing.verified("qwen4_exp"), format_load.loading(tmp_path):
        fmt = "affine" if builtin else Plugin()
        packed = layer_cls(fmt, 4, 64, 256, tensors)
    model = nn.Module()
    model.proj = packed
    model.norm = nn.RMSNorm(16)
    model.norm.set_dtype(mx.bfloat16)
    apply_activation_fp16(model, SimpleNamespace(activation_fp16_enabled=True))
    assert packed.weight is tensors["weight"]
    if builtin:
        assert packed.scales.dtype == packed.biases.dtype == mx.float16
        assert model.norm.weight.dtype == mx.float16
        for name in ("scales", "biases"):
            assert mx.array_equal(packed[name], tensors[name].astype(mx.float16)).item()
        assert numerics_revision_for_model(model) == "fp16-recast"
    else:
        assert packed.scales is tensors["scales"] and packed.biases is tensors["biases"]
        assert model.norm.weight.dtype == mx.bfloat16
        assert numerics_revision_for_model(model) is None


@pytest.mark.parametrize("family", ("qwen4_exp", "olmoe"))
def test_admission_accounts_for_an_unrecorded_affine_checkpoint(tmp_path, family):
    from tests.test_moe_expert_offload import _glu_tensors, _make_glu, _save_checkpoint

    glu = _make_glu(seed=9)
    tensors = _glu_tensors(glu, "layers.0.experts.switch_glu")
    _save_checkpoint(tmp_path, tensors)
    config = {"model_type": family, "quantization": {"group_size": 32, "bits": 4}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    total = sum(a.size * a.dtype.size for a in tensors.values())
    estimated = moe_expert_offload.estimate_offload_admission_bytes(
        tmp_path, total, 0.25
    )
    if family == "qwen4_exp":
        assert estimated == total
    else:
        assert estimated == total - int(0.75 * total)


def test_offload_reports_the_family_and_deferred_row_and_keeps_experts(
    tmp_path, caplog
):
    from tests.test_moe_expert_offload import (
        GROUP,
        D,
        _glu_tensors,
        _make_glu,
        _MiniMoE,
        _save_checkpoint,
    )

    glu = _make_glu(seed=9)
    _save_checkpoint(tmp_path, _glu_tensors(glu, "layers.0.experts.switch_glu"))
    config = {
        "model_type": "qwen4_exp",
        "quantization": {"group_size": GROUP, "bits": 4},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = _MiniMoE([glu])
    arrays = {name: glu.gate_proj[name] for name in ("weight", "scales", "biases")}
    with testing.verified("qwen4_exp"):
        glu.gate_proj = FormatSwitchLinear("affine", 4, GROUP, D, arrays)
    with caplog.at_level(logging.INFO, logger=moe_expert_offload.__name__):
        assert moe_expert_offload.apply_moe_expert_offload(model, tmp_path, 0.25) == 0
    assert model.layers[0].experts.switch_glu is glu
    assert glu.gate_proj.weight is arrays["weight"]
    messages = [r.getMessage().lower() for r in caplog.records]
    assert len([s for s in messages if "qwen4_exp" in s and "14" in s]) == 1


def test_set_dtype_recasts_builtin_but_not_plugin_stored_floats(tmp_path):
    config = {
        "model_type": "qwen4_exp",
        "omlx_quant_formats": {"affine": {}, Plugin.mode: {}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    source = mx.arange(16 * 256).reshape(16, 256).astype(mx.bfloat16)
    tensors = dict(
        zip(("weight", "scales", "biases"), mx.quantize(source, group_size=64, bits=4))
    )
    with testing.verified("qwen4_exp"), format_load.loading(tmp_path):
        builtin = FormatLinear("affine", 4, 64, 256, tensors)
    with testing.verified("qwen4_exp"), format_load.loading(tmp_path):
        plugin = FormatLinear(Plugin(), 4, 64, 256, tensors)

    builtin.set_dtype(mx.float16)
    plugin.set_dtype(mx.float16)

    assert (
        builtin["scales"].dtype == mx.float16 and builtin["biases"].dtype == mx.float16
    )
    assert plugin["scales"] is tensors["scales"]
    assert plugin["biases"] is tensors["biases"]
