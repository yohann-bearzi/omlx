# SPDX-License-Identifier: Apache-2.0
"""A format checkpoint may only enter a loader that checks format layers."""

import json
from typing import get_args, get_type_hints
from unittest.mock import AsyncMock

import mlx.core as mx
import pytest

from omlx import formats
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.formats import testing
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.model_discovery import EngineType
from omlx.model_settings import ModelSettings
from omlx.models.embedding import MLXEmbeddingModel
from omlx.models.reranker import MLXRerankerModel


@pytest.mark.parametrize(
    "loader,family,architecture",
    [
        (MLXEmbeddingModel, "bert", "BertModel"),
        (MLXEmbeddingModel, "bert", "BertForMaskedLM"),
        (MLXEmbeddingModel, "xlm-roberta", "XLMRobertaModel"),
        (MLXEmbeddingModel, "qwen2", "Qwen2ForCausalLM"),
        (MLXRerankerModel, "xlm-roberta", "XLMRobertaForSequenceClassification"),
    ],
)
@pytest.mark.parametrize(
    "settings,mode",
    [
        ({"omlx_quant_formats": {"toy": {}}}, "toy"),
        ({"omlx_quant_formats": None}, "unknown"),
        ({"omlx_quant_formats": {}}, "unknown"),
        ({"text_config": {"omlx_quant_formats": {"toy": {}}}}, "toy"),
        ({"text_config": {"quantization_config": {"proj": {"mode": "toy"}}}}, "toy"),
    ],
)
def test_native_reader_refuses_before_reading_weights(
    tmp_path, monkeypatch, loader, family, architecture, settings, mode
):
    config = dict(
        settings,
        model_type=family,
        architectures=[architecture],
        hidden_size=8,
        num_hidden_layers=1,
        intermediate_size=16,
        num_attention_heads=2,
        vocab_size=16,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").touch()

    def read(path):
        pytest.fail(f"{architecture} opened {path}")

    monkeypatch.setattr(mx, "load", read)
    with testing.verified(family), pytest.raises(formats.FormatRefusedError) as err:
        loader(str(tmp_path)).load()
    assert err.value.mode == mode
    assert err.value.path == tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    sorted(
        set(get_args(EngineType))
        | set(get_args(get_type_hints(EngineEntry)["engine_type"]))
    )
    + ["external", "unknown-engine"],
)
async def test_only_checked_engine_routes_start_a_format_checkpoint(
    tmp_path, monkeypatch, route
):
    config = {"model_type": "llama", "omlx_quant_formats": {"toy": {}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    pool = EnginePool()
    pool._entries["tiny"] = EngineEntry(
        model_id="tiny",
        model_path=str(tmp_path),
        model_type="llm",
        engine_type="batched" if route == "external" else route,
        estimated_size=1,
    )
    settings = None
    if route == "external":
        settings = ModelSettings(external_engine="splash")
        monkeypatch.setattr(pool, "_check_splash_launcher", AsyncMock())
        monkeypatch.setattr(
            pool,
            "_external_engine_for",
            lambda *args: pytest.fail("external loader inspected the checkpoint"),
        )

    class Started(BaseException):
        """Stop before the selected engine opens a weight file."""

    async def start(engine):
        raise Started

    monkeypatch.setattr(pool, "_start_engine", start)
    with testing.verified("llama"):
        if route in ("batched", "vlm"):
            with pytest.raises(Started):
                await pool._load_engine("tiny", runtime_settings=settings)
        else:
            with pytest.raises(formats.FormatRefusedError) as err:
                await pool._load_engine("tiny", runtime_settings=settings)
            refusal = err.value
            assert refusal.mode == "toy"
            assert refusal.path == tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_type", ["batched", "vlm"])
async def test_an_unverified_target_refuses_before_family_specific_readers(
    tmp_path, monkeypatch, engine_type
):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v41", "omlx_quant_formats": {"toy": {}}})
    )
    pool = EnginePool()
    pool._entries["tiny"] = EngineEntry(
        model_id="tiny",
        model_path=str(tmp_path),
        model_type="vlm",
        engine_type=engine_type,
        estimated_size=1,
    )

    async def opened(engine):
        pytest.fail(f"unverified model reached {type(engine).__name__}")

    monkeypatch.setattr(pool, "_start_engine", opened)
    with pytest.raises(formats.FormatRefusedError) as raised:
        await pool._load_engine("tiny")
    refusal = raised.value
    assert refusal.mode == "toy" and "deepseek_v41" in refusal.reason


@pytest.mark.parametrize(
    "library,source",
    [
        ("embeddings", "disk"),
        ("embeddings", "model_config"),
        ("encoder", "disk"),
        ("encoder", "config"),
        ("encoder", "config_overrides"),
    ],
)
def test_resolved_library_loader_refuses_disk_and_override_records(
    tmp_path, monkeypatch, library, source
):
    from mlx_embeddings.utils import load_model
    from mlx_vlm.encoder_loader import load_encoder_model

    from omlx.patches.format_load import apply_format_load_patch

    apply_format_load_patch()
    base = {"model_type": "llama"}
    written = {"omlx_quant_formats": {"toy": {}}}
    (tmp_path / "config.json").write_text(
        json.dumps({**base, **(written if source == "disk" else {})})
    )
    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(mx, "load", lambda *a, **kw: pytest.fail("opened weights"))
    with testing.verified("llama"), pytest.raises(formats.FormatRefusedError) as err:
        if library == "embeddings":
            # model_config may be passed positionally.
            load_model(tmp_path, True, written if source == "model_config" else {})
        else:
            overrides = {} if source == "disk" else {source: {**base, **written}}
            load_encoder_model(tmp_path, model_remapping={}, **overrides)
    assert err.value.path == tmp_path and err.value.mode == "toy"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["batched", "vlm"])
async def test_direct_language_engine_refuses_before_special_loader_setup(
    tmp_path, monkeypatch, kind
):
    from omlx.engine import BatchedEngine, VLMBatchedEngine
    from omlx.utils import model_loading

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v41", "omlx_quant_formats": {"toy": {}}})
    )
    monkeypatch.setattr(
        model_loading,
        "maybe_apply_pre_load_patches",
        lambda *args, **kw: pytest.fail("reached family setup"),
    )
    engine = (VLMBatchedEngine if kind == "vlm" else BatchedEngine)(str(tmp_path))
    with pytest.raises(formats.FormatRefusedError) as err:
        await engine.start()
    assert err.value.mode == "toy" and "deepseek_v41" in err.value.reason


# Exercise the generic mlx-lm reader even when family dispatchers wrap it.
@pytest.mark.parametrize(
    "on_disk,override",
    [
        ({"model_type": "qwen2"}, {"omlx_quant_formats": {"toy": {}}}),
        (
            {"model_type": "qwen2"},
            {"model_type": "llama", "omlx_quant_formats": {"toy": {}}},
        ),
        (
            {"model_type": "llama", "omlx_quant_formats": {"toy": {}}},
            {"model_type": "qwen2"},
        ),
    ],
)
def test_checked_lm_loader_uses_the_effective_family_before_weight_io(
    tmp_path, monkeypatch, on_disk, override
):
    import mlx_lm.utils

    from omlx.patches.format_load import apply_format_load_patch

    (tmp_path / "config.json").write_text(json.dumps(on_disk))
    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(mx, "load", lambda *a, **kw: pytest.fail("opened weights"))
    apply_format_load_patch()
    with testing.verified("llama"), pytest.raises(formats.FormatRefusedError) as error:
        mlx_lm.utils.load_model(tmp_path, True, False, override)
    assert error.value.mode == "toy" and "qwen2" in error.value.reason


@pytest.mark.parametrize("form", ["mapping", "pairs", "iterator"])
@pytest.mark.parametrize("deepseek_wrapper", [False, True], ids=["checked", "deepseek"])
def test_checked_lm_loader_accounts_for_a_format_declared_by_override(
    tmp_path, form, deepseek_wrapper
):
    import mlx_lm.utils
    from mlx.utils import tree_flatten
    from mlx_lm.models import llama

    from omlx.patches.deepseek_v4.utils_patch import _build_patched_load_model
    from omlx.patches.format_load import apply_format_load_patch

    config = dict(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=32,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = llama.Model(llama.ModelArgs.from_dict(config))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    apply_format_load_patch()
    read_model = (
        _build_patched_load_model() if deepseek_wrapper else mlx_lm.utils.load_model
    )

    def override(fields):
        if form == "pairs":
            return list(fields.items())
        if form == "iterator":
            return iter(fields.items())
        return fields

    _, effective = read_model(
        tmp_path, lazy=True, model_config=override({"probe": "from override"})
    )
    assert effective["probe"] == "from override"
    with testing.verified("llama"), pytest.raises(formats.FormatRefusedError) as error:
        read_model(
            tmp_path,
            lazy=True,
            model_config=override({"omlx_quant_formats": {"orphan": {}}}),
        )
    assert error.value.mode == "orphan" and "no layer" in error.value.reason


@pytest.mark.parametrize(
    "on_disk,override,field",
    [
        (
            {"omlx_quant_formats": {"toy": {"record": {"v": 1}}}},
            {"omlx_quant_formats": {"toy": {"record": {"v": 2}}}},
            "omlx_quant_formats",
        ),
        (
            {"quantization": {"proj": {"mode": "toy"}}},
            {"quantization": {}},
            "quantization",
        ),
        (
            {"omlx_quant_formats": {"toy": {"record": {"v": 1}}}},
            {"omlx_quant_formats": {"toy": {"record": {"v": True}}}},
            "omlx_quant_formats",
        ),
        (
            {"omlx_quant_formats": {"toy": {"record": {"levels": [True, 2]}}}},
            {"omlx_quant_formats": {"toy": {"record": {"levels": [True]}}}},
            "omlx_quant_formats",
        ),
        (
            {"omlx_quant_formats": {"toy": {}}},
            {"text_config": []},
            "text_config",
        ),
        (
            {"quantization_config": {"proj": {"mode": "toy"}}},
            {
                "quantization_config": {
                    "proj": {"mode": "toy"},
                    "other": {"mode": "toy"},
                }
            },
            "quantization_config",
        ),
        (
            {"quantization": {"language_model.proj": {"mode": "toy"}}},
            {
                "quantization": {
                    "language_model.proj": {"mode": "toy"},
                    "proj": {"mode": "affine"},
                }
            },
            "quantization",
        ),
        (
            {"text_config": {"model_type": "llama", "quantization": {"mode": "toy"}}},
            {"text_config": {"model_type": "llama"}},
            "text_config.quantization",
        ),
    ],
)
def test_a_checked_loader_cannot_replace_the_disk_format_layout(
    tmp_path, monkeypatch, on_disk, override, field
):
    import mlx_lm.utils

    from omlx.patches.format_load import apply_format_load_patch

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", **on_disk})
    )
    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(mx, "load", lambda *a, **kw: pytest.fail("opened weights"))
    apply_format_load_patch()
    with testing.verified("llama"), pytest.raises(formats.FormatRefusedError) as error:
        mlx_lm.utils.load_model(tmp_path, model_config=override)
    assert error.value.mode == "toy" and field in error.value.reason


@pytest.mark.parametrize(
    "layer_type,widths,pending",
    [
        (cls, widths, pending)
        for cls, widths in (
            (FormatLinear, (64, 32, 8, 8)),
            (FormatSwitchLinear, (32, 32)),
        )
        for pending in (None, *range(len(widths)))
    ],
)
def test_loaded_projections_stack_along_their_output_axis(layer_type, widths, pending):
    from tests.test_formats_linear import _StackableToy

    reader = _StackableToy()
    experts = (4,) if layer_type is FormatSwitchLinear else ()
    layers = []
    with testing.verified("qwen4_exp"):
        for i, n in enumerate(widths):
            data = {
                "codes": mx.full((*experts, n, 128), i + 127, mx.uint8),
                "group_scales": mx.full((*experts, n), i + 1, mx.float32),
            }
            bias = mx.full((*experts, n), -i, mx.float32)
            layers.append(
                layer_type(reader, 8, 64, 128, data, bias, placeholders=i == pending)
            )

    first, *others = layers
    first.eval()
    first.unfreeze(keys=["group_scales"])
    with pytest.raises(formats.FormatRefusedError):
        layer_type(reader, 8, 64, 128, data)
    if pending is not None:
        with pytest.raises(ValueError, match="placeholder"):
            first._stacked_outputs(*others)
        return

    joined = first._stacked_outputs(*others)
    assert type(joined) is layer_type and joined.fmt is reader
    assert (joined.input_dims, joined.output_dims) == (128, sum(widths))
    assert [part.output_dims for part in layers] == list(widths)
    assert not joined.training and set(joined.trainable_parameters()) == {
        "group_scales"
    }
    for name in ("codes", "group_scales", "bias"):
        expected = mx.concatenate([layer[name] for layer in layers], axis=len(experts))
        assert mx.array_equal(joined[name], expected).item()
