# SPDX-License-Identifier: Apache-2.0
"""Run MLX-LM and the cluster loader over the same two-block local checkpoint."""

import json
import sys
from types import ModuleType, SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils as lm
import pytest
from mlx.nn.layers.distributed import shard_inplace
from mlx_lm.models.pipeline import PipelineMixin

from omlx import formats
from omlx.cluster.progressive_loading import progressive_sharded_load
from omlx.formats.linear import FormatLinear
from omlx.patches import format_load, mlx_lm_pipeline_index


class Body(PipelineMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(128, 32, bias=False) for _ in range(2)]


class Model(nn.Module):
    model_type = "qwen4_exp"

    def __init__(self, args):
        super().__init__()
        self.model = Body()

    def shard(self, group):
        for layer in self.model.layers:
            shard_inplace(layer, "all-to-sharded", group=group)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    arch = ModuleType("mlx_lm.models.qwen4_exp")
    arch.Model, arch.ModelArgs = Model, SimpleNamespace(from_dict=dict)
    monkeypatch.setitem(sys.modules, arch.__name__, arch)
    monkeypatch.setattr(lm, "load_tokenizer", lambda *args, **kwargs: object())
    format_load.apply_format_load_patch()
    recipe = {"bits": 4, "group_size": 64}
    config = {"model_type": "qwen4_exp", "quantization": recipe}
    (tmp_path / "config.json").write_text(json.dumps(config))
    packed = {}
    for i in range(2):
        w = mx.random.normal((32, 128), key=mx.random.key(i)).astype(mx.float32)
        arrays = mx.quantize(w, **recipe)
        packed.update(
            {
                f"model.layers.{i}.{name}": value
                for name, value in zip(("weight", "scales", "biases"), arrays)
            }
        )
    mx.save_safetensors(str(tmp_path / "model.safetensors"), packed)
    index = {"weight_map": {key: "model.safetensors" for key in packed}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return tmp_path, packed


@pytest.mark.parametrize("loader", ("mlx_lm", "progressive"))
@pytest.mark.parametrize("parallel", ("tensor", "pipeline"))
@pytest.mark.parametrize("rank", (0, 1))
def test_distributed_stock_qwen_keeps_mlx_and_shards_packed_weights(
    checkpoint, loader, parallel, rank, monkeypatch
):
    if parallel == "pipeline":
        # The worker installs this adapter before it installs progressive loading.
        # Save its three module bindings so independent loader tests stay isolated.
        monkeypatch.setattr(mlx_lm_pipeline_index, "_APPLIED", False)
        monkeypatch.setattr(lm, "json", lm.json)
        monkeypatch.setattr(lm, "open", getattr(lm, "open", None), raising=False)
        assert mlx_lm_pipeline_index.apply_mlx_lm_pipeline_index_patch()
    path, packed = checkpoint
    group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
    run = lm.sharded_load if loader == "mlx_lm" else progressive_sharded_load
    model, _ = run(path, **{f"{parallel}_group": group})
    for i in range(2):
        layers = model.model.layers
        if parallel == "pipeline" and i != 1 - rank:
            assert i >= len(layers) or layers[i] is None
            continue
        assert type(layers[i]) is nn.QuantizedLinear
        for name in ("weight", "scales", "biases"):
            ref = packed[f"model.layers.{i}.{name}"]
            if parallel == "tensor":
                ref = ref[rank * 16 : (rank + 1) * 16]
            got = layers[i][name]
            assert got.shape == ref.shape and got.dtype == ref.dtype
            assert mx.array_equal(got, ref).item()


def test_single_machine_stock_qwen_uses_format_layers(checkpoint):
    path, packed = checkpoint
    model, _ = lm.load_model(path)
    for i, layer in enumerate(model.model.layers):
        assert type(layer) is FormatLinear
        for name in ("weight", "scales", "biases"):
            assert mx.array_equal(
                layer[name], packed[f"model.layers.{i}.{name}"]
            ).item()


@pytest.mark.parametrize("loader", ("mlx_lm", "progressive"))
def test_distributed_plugin_refuses_before_weight_reads(
    checkpoint, monkeypatch, loader
):
    path, _ = checkpoint
    config = json.loads((path / "config.json").read_text())
    config["omlx_quant_formats"] = {"plugin_test": {"contract": 1, "record": {}}}
    (path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(mx, "load", lambda *a, **k: pytest.fail("read plugin weights"))
    run = lm.sharded_load if loader == "mlx_lm" else progressive_sharded_load
    with pytest.raises(formats.FormatRefusedError, match="plugin_test"):
        run(path, tensor_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0))


def test_sharded_loading_refuses_recorded_builtin_affine_after_preflight(checkpoint):
    path, _ = checkpoint
    config = json.loads((path / "config.json").read_text())
    with format_load.sharded_loading(path, "sharded_load"):
        # The Hub branch changed between the metadata preflight and the
        # library's resolution: a record of the built-in does not slip the
        # distributed refusal.
        config["omlx_quant_formats"] = {"affine": {"contract": 1, "record": {}}}
        (path / "config.json").write_text(json.dumps(config))
        with (
            pytest.raises(formats.FormatRefusedError, match="sharded_load"),
            format_load.loading(path),
        ):
            pass
