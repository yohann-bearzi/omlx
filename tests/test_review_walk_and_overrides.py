# SPDX-License-Identifier: Apache-2.0
"""Regressions for the text-only writer walk and the DeepSeek V4 custom loader."""

import json

import mlx.core as mx
import mlx_lm.utils
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama, qwen3_vl

from omlx import formats, oq
from omlx.model_discovery import detect_model_type
from omlx.patches import format_load
from omlx.patches.deepseek_v4 import utils_patch


def test_real_text_only_vlm_walk_matches_the_served_text_model(tmp_path):
    text = dict(
        model_type="qwen3_vl_text",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=128,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    config = dict(model_type="qwen3_vl", text_config=text, vision_config={"depth": 1})
    source, output = tmp_path / "source", tmp_path / "text"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    original = qwen3_vl.Model(qwen3_vl.ModelArgs.from_dict(config))
    weights = dict(tree_flatten(original.parameters()))
    for path in (source, output):
        mx.save_safetensors(str(path / "model.safetensors"), weights)
    written = oq._written_layout(
        config,
        source,
        text_only=True,
        preserve_mtp=False,
        dtype="bfloat16",
        quantization={"mode": "affine", "bits": 4, "group_size": 64},
    )
    (output / "config.json").write_text(json.dumps(written))
    assert detect_model_type(source) == "vlm" and detect_model_type(output) == "llm"
    live, _ = mlx_lm.utils.load_model(output)
    walked = oq._module_classes(str(source), written)
    assert walked[""][0] == f"{type(live).__module__}.{type(live).__qualname__}"
    path = "language_model.model.layers.0.mlp.gate_proj"
    assert walked[path][1:] == (True, tuple(weights[f"{path}.weight"].shape))


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}},
        {"model_type": "gemma4", "text_config": {"mtp_assistant_config": {}}},
        {"model_type": "diffusion_gemma"},
        {"model_type": "lfm3_audio"},
        {"model_type": "qwen3_omni_moe"},
    ],
)
def test_written_text_layout_keeps_a_required_vlm_loader(tmp_path, monkeypatch, config):
    from tests.test_formats_oq import _checkpoint, _stub_loaders

    calls, _ = _stub_loaders(monkeypatch)
    source = _checkpoint(tmp_path / "source", config)
    oq._module_classes(str(source), config)
    assert calls[-1][0] == "mlx-vlm"


def _config():
    return dict(
        model_type="deepseek_v4",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=32,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"quantization": {"mode": "toy", "group_size": 1, "bits": 8}},
        {"omlx_quant_formats": None},
        {"text_config": {"quantization_config": {"mode": "toy"}}},
    ],
)
def test_deepseek_loader_checks_the_effective_config_before_weight_io(
    tmp_path, monkeypatch, override
):
    config = {**_config(), "model_type": "llama"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").touch()

    def read(path):
        pytest.fail(f"DeepSeek loader opened {path}")

    monkeypatch.setattr(utils_patch, "_load_safetensors", read)
    loader = utils_patch._build_patched_load_model()
    with pytest.raises(formats.FormatRefusedError) as err:
        loader(tmp_path, model_config={"model_type": "deepseek_v4", **override})
    assert err.value.path == tmp_path
    assert err.value.mode == ("unknown" if "omlx_quant_formats" in override else "toy")


def test_joined_deepseek_load_is_checked_only_when_the_outer_loader_returns(
    tmp_path, monkeypatch
):
    config = _config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    weights = dict(
        tree_flatten(llama.Model(llama.ModelArgs.from_dict(config)).parameters())
    )
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    checked = []
    check = format_load._Checkpoint.check

    def observe(checkpoint, model):
        checked.append((checkpoint, model))
        return check(checkpoint, model)

    monkeypatch.setattr(format_load._Checkpoint, "check", observe)
    loader = utils_patch._build_patched_load_model()
    with format_load.loading(tmp_path, by_loader=True) as outer:
        result = loader(
            tmp_path, get_model_classes=lambda config: (llama.Model, llama.ModelArgs)
        )
        assert not checked
        outer.check(result)
    assert len(checked) == 1 and checked[0] == (outer, result)
