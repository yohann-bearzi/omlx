# SPDX-License-Identifier: Apache-2.0
"""Direct calls to unsupported readers obey the same rule as pool admission."""

import json
import sys
from pathlib import Path
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "engine_type"),
    [(stt, stt.STTEngine), (sts, sts.STSEngine), (tts, tts.TTSEngine)],
)
async def test_audio_hub_readers_check_the_resolved_config_before_weights(
    checkpoint, monkeypatch, module, engine_type
):
    import mlx_audio.utils

    repo = "test-org/deepfilternet"
    downloads = []

    def snapshot(repo_id, *, allow_patterns=None, **kwargs):
        assert repo_id == repo
        assert set(allow_patterns or ()) == {"config.json"}
        downloads.append(repo_id)
        return str(checkpoint)

    monkeypatch.setattr(mlx_audio.utils, "snapshot_download", snapshot)
    monkeypatch.setattr(module, "get_mlx_executor", entered)
    with pytest.raises(formats.FormatRefusedError) as error:
        await engine_type(repo).start()
    assert downloads == [repo]
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


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_type", [stt.STTEngine, tts.TTSEngine, sts.STSEngine])
async def test_audio_hub_readers_name_the_missing_audio_extra(monkeypatch, engine_type):
    for name in tuple(sys.modules):
        if name.startswith("mlx_audio."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    message = r"Install it with: pip install .*omlx\[audio\]"
    with pytest.raises(ImportError, match=message):
        await engine_type("test-org/deepfilternet").start()


def test_relative_audio_directory_is_checked_locally(checkpoint, monkeypatch):
    import mlx_audio.utils

    from omlx.patches.format_load import checked_audio_path

    monkeypatch.chdir(checkpoint.parent)
    monkeypatch.setattr(mlx_audio.utils, "get_model_path", entered)
    with pytest.raises(formats.FormatRefusedError) as refused:
        checked_audio_path(checkpoint.name, "audio_stt")
    assert (refused.value.path, refused.value.mode) == (checkpoint, "toy")


@pytest.mark.asyncio
@pytest.mark.parametrize("exists", [False, True], ids=["missing", "present"])
async def test_local_sts_checks_the_stock_config_once_before_loading(
    tmp_path, monkeypatch, exists
):
    import mlx_audio.utils

    source = tmp_path / "deepfilternet"
    config = source / "config.json"
    if exists:
        source.mkdir()
        config.write_text('{"model_type":"deepfilternet"}')
    engine = sts.STSEngine(str(source))
    events = []
    read = Path.read_text
    resolve = sts.checked_audio_path

    def observe_resolver(*args):
        events.append("resolve")
        return resolve(*args)

    def observe(path, *args, **kwargs):
        if path == config:
            events.append("config")
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", observe)
    monkeypatch.setattr(sts, "checked_audio_path", observe_resolver)
    monkeypatch.setattr(mlx_audio.utils, "get_model_path", entered)
    monkeypatch.setattr(sts, "get_mlx_executor", lambda: None)
    monkeypatch.setitem(sts._FAMILY_LOADERS, "deepfilternet", events.append)
    await engine.start()
    assert events == ["config", str(source)]


def test_a_missing_absolute_audio_source_stays_local(tmp_path, monkeypatch):
    import mlx_audio.utils

    from omlx.patches.format_load import checked_audio_path

    source = str(tmp_path / "absent")
    monkeypatch.setattr(mlx_audio.utils, "get_model_path", entered)
    assert checked_audio_path(source, "audio_tts") == source


@pytest.mark.asyncio
async def test_stock_audio_hub_reads_weights_from_the_checked_snapshot(
    tmp_path, monkeypatch
):
    import mlx_audio.utils

    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"deepfilternet"}')
    source = "test-org/deepfilternet"
    calls = []

    def resolve(repo, *, allow_patterns=None, revision=None):
        calls.append((repo, allow_patterns, revision))
        return snapshot

    seen = []
    monkeypatch.setattr(mlx_audio.utils, "get_model_path", resolve)
    monkeypatch.setattr(sts, "get_mlx_executor", lambda: None)
    monkeypatch.setitem(sts._FAMILY_LOADERS, "deepfilternet", seen.append)
    await sts.STSEngine(source).start()
    assert calls == [(source, ["config.json"], None), (source, None, snapshot.name)]
    assert seen == [str(snapshot)]
