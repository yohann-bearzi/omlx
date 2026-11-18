# SPDX-License-Identifier: Apache-2.0
"""Direct calls to unsupported readers obey the same rule as pool admission."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from omlx import formats
from omlx.cluster.progressive_loading import progressive_sharded_load
from omlx.engine import sts, stt, tts
from omlx.engine.splash import SplashEngine, SplashOptions
from omlx.formats import testing
from omlx.models import embedding, reranker
from omlx.patches import qwen38_modelopt_mixed
from omlx.patches.deepseek_v41 import loading as deepseek_v41
from omlx.patches.mimo_v2 import omnimodal
from omlx.utils.model_loading import maybe_load_custom_quantization


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "tiny-reranker"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "omlx_quant_formats": {"toy": {}},
            }
        )
    )
    with testing.verified("llama"):
        yield path


def entered(*args, **kwargs):
    pytest.fail("unchecked loader reached its backend")


@pytest.mark.parametrize("role", ["jina", "causal", "vl_reranker", "encoder"])
def test_direct_model_load_refuses_even_when_the_backbone_is_verified(
    checkpoint, monkeypatch, role
):
    config = checkpoint / "config.json"
    dispatch = {
        "jina": ("JinaForRanking", "_load_jina_reranker"),
        "causal": ("Qwen3ForCausalLM", "_load_causal_lm"),
        "vl_reranker": ("Qwen3VLForConditionalGeneration", "_load_vl_reranker"),
    }
    if role == "encoder":
        model = embedding.MLXEmbeddingModel(str(checkpoint))
        monkeypatch.setattr(model, "_load_mlx_vlm", entered)
    else:
        arch, reader = dispatch[role]
        config.write_text(
            json.dumps({**json.loads(config.read_text()), "architectures": [arch]})
        )
        model = reranker.MLXRerankerModel(str(checkpoint))
        monkeypatch.setattr(model, reader, entered)
    with pytest.raises(formats.FormatRefusedError) as error:
        model.load()
    assert error.value.path == checkpoint and error.value.mode == "toy"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["stt", "sts", "tts", "external"])
async def test_direct_engine_start_refuses_before_audio_or_external_loading(
    checkpoint, monkeypatch, kind
):
    if kind == "external":
        engine = SplashEngine(
            str(checkpoint), "tiny", Mock(), checkpoint, checkpoint, SplashOptions()
        )
        monkeypatch.setattr(engine, "_load_tokenizer", entered)
    else:
        module, cls = {
            "stt": (stt, stt.STTEngine),
            "sts": (sts, sts.STSEngine),
            "tts": (tts, tts.TTSEngine),
        }[kind]
        monkeypatch.setattr(module, "get_mlx_executor", entered)
        engine = cls(str(checkpoint))
    with pytest.raises(formats.FormatRefusedError) as error:
        await engine.start()
    assert error.value.path == checkpoint and error.value.mode == "toy"


@pytest.mark.parametrize(
    "reader", [deepseek_v41.load, omnimodal.load, qwen38_modelopt_mixed.load]
)
def test_direct_special_checkpoint_readers_require_their_own_weight_layout(
    checkpoint, reader
):
    with pytest.raises(formats.FormatRefusedError) as error:
        reader(checkpoint)
    assert error.value.path == checkpoint and error.value.mode == "toy"


@pytest.mark.parametrize("method", ["paroquant", "compressed-tensors"])
def test_custom_quantization_never_preempts_the_checked_format_loader(
    checkpoint, monkeypatch, method
):
    file = checkpoint / "config.json"
    config = json.loads(file.read_text())
    config["quantization_config"] = {"quant_method": method}
    file.write_text(json.dumps(config))
    monkeypatch.setattr(
        qwen38_modelopt_mixed, "is_supported_config", lambda config: True
    )
    monkeypatch.setattr(qwen38_modelopt_mixed, "load", entered)
    with pytest.raises(formats.FormatRefusedError) as error:
        maybe_load_custom_quantization(str(checkpoint), is_vlm=True)
    assert error.value.path == checkpoint and error.value.mode == "toy"


@pytest.mark.parametrize("group", ["pipeline_group", "tensor_group"])
def test_custom_rank_loaders_refuse_before_dispatching_a_shard(
    checkpoint, monkeypatch, group
):
    config = json.loads((checkpoint / "config.json").read_text())
    family = "test_stage"
    config["model_type"] = family
    (checkpoint / "config.json").write_text(json.dumps(config))
    monkeypatch.setitem(
        sys.modules,
        f"mlx_lm.models.{family}",
        SimpleNamespace(
            load_pipeline_stage=entered,
            load_tensor_shard=entered,
        ),
    )
    utils = SimpleNamespace(
        _download=lambda *a, **kw: checkpoint,
        load_config=lambda path: json.loads((path / "config.json").read_text()),
        load_tokenizer=lambda *a, **kw: None,
    )
    with testing.verified(family), pytest.raises(formats.FormatRefusedError) as err:
        progressive_sharded_load(
            str(checkpoint), utils_module=utils, **{group: object()}
        )
    assert err.value.path == checkpoint and err.value.mode == "toy"


@pytest.mark.parametrize(
    "marker",
    [
        {"omlx_quant_formats": {}},
        {"text_config": {"omlx_quant_formats": {"toy": {}}}},
    ],
)
def test_expert_offload_refuses_before_opening_its_weight_store(
    tmp_path, monkeypatch, marker
):
    import mlx.nn as nn

    from omlx.patches import moe_expert_offload, moe_offload_compat

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", **marker})
    )
    monkeypatch.setattr(moe_expert_offload, "CheckpointExpertStore", entered)
    assert moe_expert_offload.apply_moe_expert_offload(nn.Module(), tmp_path) == 0
    assert moe_offload_compat.moe_offload_compatibility(tmp_path) == (
        False,
        moe_offload_compat.IN_QUANT_FORMAT,
    )
