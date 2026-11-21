# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Qwen4-Exp MTP-specific MoE dimensions."""

from __future__ import annotations

import dataclasses

import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp import (  # noqa: E402
    TextConfig,
    language,
)


def _config(
    *,
    mtp_num_experts: int | None = 512,
    mtp_num_experts_per_tok: int | None = 4,
    mtp_use_dedicated_embeddings: bool = False,
    mtp: dict | None = None,
) -> TextConfig:
    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        linear_num_value_heads=1,
        linear_num_key_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        num_experts=384,
        num_experts_per_tok=10,
        shared_expert_intermediate_size=8,
        moe_intermediate_size=8,
        rms_norm_eps=1e-6,
        vocab_size=16,
        num_key_value_heads=1,
        max_position_embeddings=16,
        heads_per_ngram=1,
        mtp_num_experts=mtp_num_experts,
        mtp_num_experts_per_tok=mtp_num_experts_per_tok,
        mtp_use_dedicated_embeddings=mtp_use_dedicated_embeddings,
        mtp=mtp,
    )


def test_mtp_module_uses_its_own_expert_counts(monkeypatch):
    captured = {}

    class FakeDecoderLayer:
        def __init__(self, config, *, layer_idx):
            captured["num_experts"] = config.num_experts
            captured["num_experts_per_tok"] = config.num_experts_per_tok
            captured["layer_idx"] = layer_idx

    class FakeGatedResidual:
        def __init__(self, config, *, use_combine):
            captured["residual_num_experts"] = config.num_experts
            captured["use_combine"] = use_combine

    monkeypatch.setattr(language, "Qwen4ExpDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(language, "Qwen4ExpGatedResidual", FakeGatedResidual)

    language.Qwen4ExpMTPModule(_config())

    assert captured == {
        "num_experts": 512,
        "num_experts_per_tok": 4,
        "layer_idx": 0,
        "residual_num_experts": 512,
        "use_combine": False,
    }


def test_mtp_module_defaults_to_model_expert_counts(monkeypatch):
    captured = {}

    class FakeDecoderLayer:
        def __init__(self, config, *, layer_idx):
            captured["num_experts"] = config.num_experts
            captured["num_experts_per_tok"] = config.num_experts_per_tok

    class FakeGatedResidual:
        def __init__(self, config, *, use_combine):
            captured["residual_num_experts"] = config.num_experts

    monkeypatch.setattr(language, "Qwen4ExpDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(language, "Qwen4ExpGatedResidual", FakeGatedResidual)

    language.Qwen4ExpMTPModule(
        _config(mtp_num_experts=None, mtp_num_experts_per_tok=None)
    )

    assert captured == {
        "num_experts": 384,
        "num_experts_per_tok": 10,
        "residual_num_experts": 384,
    }


def test_mtp_module_falls_back_per_missing_override(monkeypatch):
    captured = {}

    class FakeDecoderLayer:
        def __init__(self, config, *, layer_idx):
            captured["num_experts"] = config.num_experts
            captured["num_experts_per_tok"] = config.num_experts_per_tok

    class FakeGatedResidual:
        def __init__(self, config, *, use_combine):
            pass

    monkeypatch.setattr(language, "Qwen4ExpDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(language, "Qwen4ExpGatedResidual", FakeGatedResidual)

    language.Qwen4ExpMTPModule(_config(mtp_num_experts=None))

    assert captured == {
        "num_experts": 384,
        "num_experts_per_tok": 4,
    }


def test_mtp_refuses_dedicated_embeddings_at_construction():
    with pytest.raises(ValueError, match="mtp_use_dedicated_embeddings=True"):
        language.Qwen4ExpMTPModule(_config(mtp_use_dedicated_embeddings=True))
    with pytest.raises(ValueError, match="mtp_use_dedicated_embeddings=True"):
        language.Qwen4ExpMTPModule(
            _config(mtp={"mtp_use_dedicated_embeddings": True})
        )
    with pytest.raises(ValueError, match="mtp_use_dedicated_embeddings=True"):
        language.Qwen4ExpMTPModule(
            _config(
                mtp_use_dedicated_embeddings=True,
                mtp={"mtp_use_dedicated_embeddings": False},
            )
        )


def test_mtp_refuses_a_hidden_state_layer_at_construction():
    with pytest.raises(
        ValueError, match="mtp_use_hidden_state_from_layer=24"
    ):
        language.Qwen4ExpMTPModule(
            _config(mtp={"mtp_use_hidden_state_from_layer": 24})
        )


def test_mtp_refuses_a_top_level_hidden_state_layer_from_dict():
    flat = dataclasses.asdict(_config())
    flat["mtp_use_hidden_state_from_layer"] = 24
    with pytest.raises(ValueError, match="mtp_use_hidden_state_from_layer=24"):
        language.Qwen4ExpMTPModule(TextConfig.from_dict(flat))


def test_mtp_a_nested_null_does_not_shadow_the_top_level_layer():
    with pytest.raises(ValueError, match="mtp_use_hidden_state_from_layer=0"):
        language.Qwen4ExpMTPModule(
            dataclasses.replace(
                _config(),
                mtp_use_hidden_state_from_layer=0,
                mtp={"mtp_use_hidden_state_from_layer": None},
            )
        )
