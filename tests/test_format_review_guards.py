# SPDX-License-Identifier: Apache-2.0
"""Load failures keep checkpoint identity and tear down work that already started."""

import asyncio
import json
import multiprocessing
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama

from omlx import formats, oq, settings
from omlx.engine.batched import BatchedEngine
from omlx.engine.vlm import VLMBatchedEngine
from omlx.formats import testing
from omlx.model_settings import ModelSettings
from omlx.patches import format_load
from omlx.patches.k2_horizon import checkpoint as k2
from omlx.prefill_helper.adapters import inkling2
from omlx.utils import model_loading
from tests.test_formats_load import _TOY, _Loadable
from tests.test_formats_oq import _save_tokenizer


def test_k2_and_wrapped_views_name_the_original_snapshot(tmp_path, monkeypatch):
    snapshot, middle, blob = [
        tmp_path / name for name in ("snapshot", "middle", "blob")
    ]
    middle.mkdir()
    snapshot.mkdir()
    blob.write_text(
        json.dumps({"model_type": "k2_horizon", "omlx_quant_formats": {"toy": {}}})
    )
    (snapshot / "config.json").symlink_to(blob)
    (middle / "config.json").symlink_to(snapshot / "config.json")
    (middle / "model.safetensors").touch()

    def load(model_path):
        with format_load.loading(model_path):
            pytest.fail("unverified K2 was admitted")

    monkeypatch.setattr(mlx_lm.utils, "load_model", load)
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", mlx_lm.utils.load_tokenizer)
    k2.apply_checkpoint_patch()
    with pytest.raises(formats.FormatRefusedError) as error:
        mlx_lm.utils.load_model(middle)
    assert error.value.path == snapshot


def test_inkling_helper_checks_the_loaded_weights_before_evaluating(
    tmp_path, monkeypatch
):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _Loadable())
    format_load.apply_format_load_patch()
    with testing.verified("llama"), format_load.loading(tmp_path):
        model = nn.Module()
        model.proj = nn.Linear(64, 128).to_quantized(**_TOY)
    monkeypatch.setattr(inkling2, "maybe_apply_pre_load_patches", lambda *a, **kw: None)
    monkeypatch.setattr(mlx_lm.utils, "load_model", lambda *a, **kw: (model, {}))
    with pytest.raises(formats.FormatRefusedError, match="holds proj without"):
        inkling2.Inkling2Adapter().load_first(str(tmp_path), 1)


@pytest.mark.parametrize(
    "error",
    [
        formats.LoadHookError("unwrapped proxy"),
        formats.FormatRefusedError("damaged proxy", mode="toy"),
    ],
)
def test_auto_proxy_sensitivity_returns_the_checkpoint_refusal_unchanged(
    tmp_path, monkeypatch, error
):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "llama"}))
    mx.save_safetensors(str(src / "model.safetensors"), {"w": mx.zeros((128, 256))})
    monkeypatch.setattr(settings, "get_system_memory", lambda: 0)
    monkeypatch.setattr(oq, "_system_available_memory_bytes", lambda: 0)
    monkeypatch.setattr(oq, "_metal_available_memory_bytes", lambda: 0)
    monkeypatch.setattr(
        oq, "_build_proxy_for_sensitivity", lambda *a, **kw: tmp_path / "proxy"
    )

    def measure(*args, **kwargs):
        raise error

    monkeypatch.setattr(oq, "_measure_sensitivity_from_quantized_model", measure)
    with pytest.raises(type(error)) as raised:
        oq.quantize_oq_streaming(
            str(src), str(tmp_path / "out"), 4, auto_proxy_sensitivity=True
        )
    assert raised.value is error


@pytest.mark.parametrize("engine_type", [BatchedEngine, VLMBatchedEngine])
@pytest.mark.parametrize("stop_fails", [False, True])
def test_refused_specprefill_draft_releases_the_started_engine(
    tmp_path, engine_type, stop_fails
):
    # Engine startup installs attention and MLP patches for the process lifetime.
    with multiprocessing.get_context("spawn").Pool(1) as worker:
        worker.apply_async(_run_refused_draft, (tmp_path, engine_type, stop_fails)).get(
            timeout=90
        )


def _run_refused_draft(tmp_path, engine_type, stop_fails):
    with pytest.MonkeyPatch.context() as monkeypatch:
        asyncio.run(
            _check_refused_draft(tmp_path, monkeypatch, engine_type, stop_fails)
        )


async def _check_refused_draft(tmp_path, monkeypatch, engine_type, stop_fails):
    path = tmp_path / "target"
    path.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
        "eos_token_id": 1,
    }
    (path / "config.json").write_text(json.dumps(config))
    model = llama.Model(llama.ModelArgs.from_dict(config))
    mx.save_safetensors(
        str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    _save_tokenizer(path, 32)
    if engine_type is VLMBatchedEngine:
        from types import SimpleNamespace

        import mlx_vlm.utils
        from transformers import AutoTokenizer

        # The VLM engine copies the processor's tokenizer. Give the tiny
        # text-only target the same wrapper as a multimodal checkpoint.
        def load_processor(path, *args, **kwargs):
            return SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(path))

        monkeypatch.setattr(mlx_vlm.utils, "load_processor", load_processor)
    monkeypatch.setattr(settings, "_global_settings", None)
    settings.init_settings(tmp_path / "settings")
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(config))
    want = formats.FormatRefusedError("draft checkpoint refused", mode="toy")
    actual_load = model_loading.lm_load_compat

    def load(model_name, *args, **kwargs):
        if Path(model_name) == draft:
            raise want
        return actual_load(model_name, *args, **kwargs)

    monkeypatch.setattr(model_loading, "lm_load_compat", load)
    engine = engine_type(
        str(path),
        model_settings=ModelSettings(
            specprefill_enabled=True, specprefill_draft_model=str(draft)
        ),
    )
    stop = engine.stop
    calls = []

    async def stop_after_refusal():
        calls.append(True)
        await stop()
        if stop_fails:
            raise RuntimeError("teardown failed")

    monkeypatch.setattr(engine, "stop", stop_after_refusal)
    try:
        with pytest.raises(formats.FormatRefusedError) as err:
            await engine.start()
        assert err.value is want
        assert calls == [True]
        assert engine._engine is None
    finally:
        await stop()
