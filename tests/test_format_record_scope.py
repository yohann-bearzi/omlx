# SPDX-License-Identifier: Apache-2.0
"""Stock checkpoints retain their loader. Plugin declarations must be accounted for."""

import json
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats, oq
from omlx.formats import testing
from omlx.patches import format_load, minimax_m3_dspark
from omlx.patches.inkling2 import dspark as inkling
from tests.test_formats_oq import _Codec

_STOCK = {"bits": 4, "group_size": 64, "mode": "affine"}


def _write(path, config):
    (path / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize(
    "nested",
    [
        {"mode": "toy", "bits": 8, "group_size": 1},
        {"proj": {"mode": "toy", "bits": 8, "group_size": 1}},
    ],
)
@pytest.mark.parametrize("key", ["quantization", "quantization_config"])
def test_stock_top_level_cannot_hide_an_unbound_text_format(tmp_path, nested, key):
    _write(
        tmp_path,
        dict(model_type="llama", quantization=_STOCK, text_config={key: nested}),
    )
    with (
        testing.verified("llama"),
        format_load.loading(tmp_path) as check,
        pytest.raises(formats.FormatRefusedError) as caught,
    ):
        check.check(nn.Module())
    assert caught.value.mode == "toy" and "no layer" in caught.value.reason


@pytest.mark.parametrize(
    "extra,mode",
    [
        ({"vision_config": {"omlx_quant_formats": {"toy": {}}}}, "toy"),
        ({"branches": [{"decoder": {"omlx_quant_formats": None}}]}, "unknown"),
        ({"text_config": {"decoder": {"quantization": {"mode": "toy"}}}}, "toy"),
        ({"quantization_config": {"groups": [{"proj": {"mode": "toy"}}]}}, "toy"),
        ({"quantization_config": {"groups": {"proj": {"mode": " "}}}}, "unknown"),
        ({"quantization": {"mode": " "}}, "unknown"),
    ],
)
def test_deep_declarations_refuse_unchecked_and_unverified_readers(
    tmp_path, extra, mode
):
    _write(tmp_path, {"model_type": "qwen2", "quantization": _STOCK, **extra})
    for route in ("embedding", "batched"):
        with pytest.raises(formats.FormatRefusedError) as caught:
            format_load.check_engine_loader(tmp_path, route)
        assert caught.value.path == tmp_path and caught.value.mode == mode
    with (
        testing.verified("qwen2"),
        pytest.raises(formats.FormatRefusedError),
        format_load.loading(tmp_path) as check,
    ):
        check.check(nn.Module())


@pytest.mark.parametrize("bits", [0, True])
def test_marked_quantization_accepts_only_positive_integer_widths(tmp_path, bits):
    _write(
        tmp_path,
        {
            "model_type": "llama",
            "omlx_quant_formats": {"toy": {}},
            "quantization": {**_STOCK, "bits": bits},
        },
    )
    with (
        testing.verified("llama"),
        format_load.loading(tmp_path) as check,
        pytest.raises(formats.FormatRefusedError) as caught,
    ):
        check.check(nn.Module())
    assert "quantization.bits" in caught.value.reason


@pytest.mark.parametrize("name", ["", ".proj", "model..proj", "proj."])
def test_a_marked_layer_name_has_no_empty_path_component(tmp_path, name):
    _write(
        tmp_path,
        {
            "model_type": "llama",
            "omlx_quant_formats": {"toy": {}},
            "quantization": {name: {"mode": "toy"}},
        },
    )
    with (
        testing.verified("llama"),
        format_load.loading(tmp_path) as check,
        pytest.raises(formats.FormatRefusedError) as caught,
    ):
        check.check(nn.Module())
    assert caught.value.reason == f"unknown quantization layout at quantization.{name}"


@pytest.mark.parametrize("role", ["minimax", "inkling"])
def test_bundled_draft_readers_refuse_before_loading_weights(
    tmp_path, monkeypatch, role
):
    if role == "minimax":
        module, config = minimax_m3_dspark, minimax_m3_dspark.MiniMaxDSparkConfig
        read, model = module.load_minimax_dspark_model, "MiniMaxDSparkDraftModel"
        monkeypatch.setattr(config, "from_dict", Mock())
    else:
        module, config = inkling, inkling.Inkling2DSparkConfig
        read, model = module.load_inkling2_dspark_model, "Inkling2DSparkModel"
        monkeypatch.setattr(config, "from_target_config", Mock())
    _write(tmp_path, {"model_type": "llama", "omlx_quant_formats": {"toy": {}}})
    (tmp_path / "mtp.safetensors").touch()
    monkeypatch.setattr(module, model, Mock())
    monkeypatch.setattr(mx, "load", lambda *a: pytest.fail("opened draft weights"))
    with testing.verified("llama"), pytest.raises(formats.FormatRefusedError) as caught:
        read(tmp_path)
    assert caught.value.mode == "toy" and caught.value.path == tmp_path
    assert "dspark" in caught.value.reason.lower()


@pytest.mark.parametrize(
    "family,backbone",
    [
        ("mimo_v2", "mimo_v2"),
        ("mimo_v2_flash", "mimo_v2_flash"),
        ("clef", "clef"),
        ("clef", "qwen3_5"),
        ("d1", "lfm3_vl"),
        ("d3", "qwen3_5"),
        ("openjev", "qwen3_5"),
        ("d1", "d1"),
        ("d3", "d3"),
        ("openjev", "openjev"),
    ],
)
def test_format_writer_refuses_families_served_by_an_unchecked_reader(
    tmp_path, monkeypatch, family, backbone
):
    from omlx.model_discovery import D3_FORMAT_ID

    source = tmp_path / "source"
    source.mkdir()
    config = {"model_type": backbone}
    if family == "clef" and backbone != family:
        (source / "joint_head.safetensors").touch()
        (source / "joint_head_config.json").write_text("{}")
    elif family == "d1" and backbone != family:
        config["architectures"] = ["D1Model"]
    elif family == "d3" and backbone != family:
        (source / "readout.safetensors").touch()
        (source / "decision_config.json").write_text(
            json.dumps({"format_id": D3_FORMAT_ID})
        )
    elif family == "openjev" and backbone != family:
        (source / "helper").mkdir()
        (source / "helper" / "shim.py").touch()
    elif family.startswith("mimo_v2"):
        (source / "omnimodal").mkdir()
        (source / "omnimodal" / "config.json").write_text("{}")
        (source / "omnimodal" / "vision_encoder.safetensors").touch()
        config["vision_config"] = {"hidden_size": 64}
    _write(source, config)
    formats.load()
    codec = _Codec()
    monkeypatch.setitem(formats._FORMATS, codec.mode, codec)
    output = tmp_path / "output"
    with (
        testing.verified(backbone),
        pytest.raises(formats.FormatRefusedError) as caught,
    ):
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format=codec.mode
        )
    assert family in caught.value.reason and caught.value.mode == codec.mode
    assert not output.exists()


@pytest.mark.parametrize("nested", [False, True], ids=["nvfp4", "nested-vlm"])
def test_stock_oq_minimax_write_loads_like_the_library(tmp_path, nested, monkeypatch):
    from inspect import unwrap

    from mlx_vlm import utils as vlm

    from omlx.engine.vlm import _force_minimax_m3_moe_sanitize_on_load
    from omlx.utils.model_loading import maybe_apply_pre_load_patches
    from tests.test_formats_oq import _minimax_m3_vl

    source, output = tmp_path / "source", tmp_path / "stock"
    weights = _minimax_m3_vl(source, layers=1)
    config = json.loads((source / "config.json").read_text())
    holder = config["text_config"] if nested else config
    holder["quantization_config"] = {"quant_method": "mxfp8", "moe_quant_algo": "NVFP4"}
    _write(source, config)
    prefix = "language_model.model.layers.0.block_sparse_moe"
    for shard, proj in (("w1", "gate"), ("w2", "down"), ("w3", "up")):
        stack = weights.pop(f"{prefix}.switch_mlp.{proj}_proj.weight")
        for index, weight in enumerate(stack):
            base = f"{prefix}.experts.{index}.{shard}"
            scale = 2688 / mx.max(mx.abs(weight.astype(mx.float32)))
            codes, scales = mx.quantize(
                weight * scale, bits=4, group_size=16, mode="nvfp4"
            )
            weights[f"{base}.weight_packed"] = codes.view(mx.uint8)
            weights[f"{base}.weight_scale"] = scales
            weights[f"{base}.weight_global_scale"] = scale
    mx.save_safetensors(str(source / "model.safetensors"), weights, {"format": "mlx"})
    tokens = (mx.arange(128).reshape(8, 16) % 63 + 1).astype(mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **k: tokens)
    oq.quantize_oq_streaming(
        str(source),
        str(output),
        4,
        enhanced=True,
        group_size=32,
        imatrix_num_samples=4,
        imatrix_seq_length=16,
        sensitivity_map_override={0: 0.1},
    )
    written = json.loads((output / "config.json").read_text())
    assert "omlx_quant_formats" not in written
    for proj in ("gate", "up", "down"):
        entry = written["quantization"][f"{prefix}.switch_mlp.{proj}_proj"]
        assert entry["mode"] == "nvfp4" and entry["global_scale"] is True
    if nested:
        assert (
            written["text_config"]["quantization_config"]
            == holder["quantization_config"]
        )
    with _force_minimax_m3_moe_sanitize_on_load(output):
        maybe_apply_pre_load_patches(str(output), for_vlm=True)
        load = vlm.load_model
        tokens = mx.array([[1, 2, 3, 4]])

        def logits(reader):
            value = reader(output).language_model(tokens)
            return getattr(value, "logits", value)

        # MiniMax can wrap the checked loader or be wrapped by it.
        library = unwrap(getattr(load, "_omlx_original", load))
        assert library is not load
        expected = logits(library)
        actual = logits(load)
        assert mx.isfinite(expected).all().item()
        assert mx.array_equal(actual, expected).item()


def test_cached_geometry_does_not_acquire_the_acceptance_lock(tmp_path):
    from tests.test_formats_load import _Loadable

    fmt = _Loadable()
    _write(tmp_path, {"model_type": "llama"})
    with testing.verified("llama"), format_load.loading(tmp_path) as check:
        geometry = (64, 128, 8, 1)
        first = check.layout(fmt, geometry)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(check, "_accepting", None)
            assert check.layout(fmt, geometry) is first


def test_deepseek_native_stock_admission_inspects_the_disk_once(tmp_path, monkeypatch):
    from omlx.patches.deepseek_v4 import utils_patch

    _write(tmp_path, {"model_type": "deepseek_v4", "quantization": _STOCK})
    marker = RuntimeError("reached native weights")
    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(utils_patch, "_load_safetensors", Mock(side_effect=marker))
    refuse = Mock(wraps=format_load.refuse_format_checkpoint)
    monkeypatch.setattr(format_load, "refuse_format_checkpoint", refuse)
    with pytest.raises(RuntimeError) as caught:
        utils_patch._build_patched_load_model()(tmp_path)
    assert caught.value is marker
    assert refuse.call_count == 1


@pytest.mark.parametrize("hidden_in", ["disk", "override"])
def test_checked_override_cannot_hide_a_misplaced_record(
    tmp_path, monkeypatch, hidden_in
):
    import mlx_lm.utils

    hidden = {"decoder": {"omlx_quant_formats": {"orphan": {}}}}
    disk = {"model_type": "llama", "omlx_quant_formats": {"toy": {}}}
    override = {"vision_config": hidden if hidden_in == "override" else {}}
    if hidden_in == "disk":
        disk["vision_config"] = hidden
    _write(tmp_path, disk)
    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(mx, "load", lambda *a, **k: pytest.fail("opened weights"))
    format_load.apply_format_load_patch()
    with testing.verified("llama"), pytest.raises(formats.FormatRefusedError) as caught:
        mlx_lm.utils.load_model(tmp_path, model_config=override)
    assert "vision_config.decoder.omlx_quant_formats" in caught.value.reason


def test_quantization_defaults_in_both_holders_must_bind(tmp_path, monkeypatch):
    from tests.test_formats_load import _TOY, _checked, _Loadable

    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _Loadable())
    format_load.apply_format_load_patch()
    with testing.verified("llama"), pytest.raises(formats.FormatRefusedError) as caught:
        _checked(
            tmp_path,
            _TOY,
            {"proj": True},
            text_config={"quantization": {"mode": "orphan"}},
        )
    assert caught.value.mode == "orphan" and "no layer" in caught.value.reason


def test_unrelated_modes_and_opaque_plugin_metadata_are_not_declarations(tmp_path):
    stock = {
        "model_type": "qwen2",
        "quantization": _STOCK,
        "vision_config": {"attention": [{"mode": "windowed"}]},
    }
    for config in (stock, ["omlx_quant_formats"]):
        _write(tmp_path, config)
        format_load.check_engine_loader(tmp_path, "embedding")
    marked = {
        **stock,
        "model_type": "llama",
        "omlx_quant_formats": {
            "toy": {"record": {"omlx_quant_formats": [], "quantization": {"mode": "x"}}}
        },
        "quantization": {"mtp.proj": {"mode": "toy"}},
    }
    _write(tmp_path, marked)
    with testing.verified("llama"), format_load.loading(tmp_path) as check:
        check.check(nn.Module())


def test_stock_layout_does_not_acquire_plugin_schema_from_a_dynamic_layer(tmp_path):
    from tests.test_formats_load import _TOY, _Loadable

    stock = {"quant_method": "mxfp8", "moe_quant_algo": "NVFP4"}
    formats.load()
    with pytest.MonkeyPatch.context() as patch, testing.verified("llama"):
        patch.setitem(formats._FORMATS, "toy", _Loadable())
        format_load.apply_format_load_patch()
        config = {"model_type": "llama", "quantization_config": stock}
        _write(tmp_path, config)
        with format_load.loading(tmp_path) as check:
            model = nn.Module()
            model.proj = nn.Linear(64, 128).to_quantized(**_TOY)
            check.check(model)


def test_format_writer_uses_decision_metadata_instead_of_the_directory_name(
    tmp_path, monkeypatch
):
    source = tmp_path / "OpenJev-preview"
    source.mkdir()
    _write(source, {"model_type": "qwen3_5"})
    formats.load()
    codec = _Codec()
    monkeypatch.setitem(formats._FORMATS, codec.mode, codec)
    with (
        testing.verified("qwen3_5"),
        pytest.raises(ValueError, match="No .safetensors files"),
    ):
        oq.quantize_oq_streaming(
            str(source),
            str(tmp_path / "output"),
            4,
            enhanced=True,
            quant_format=codec.mode,
        )


@pytest.mark.parametrize("reader", ["batched", "embedding", "offload"])
def test_a_format_record_in_a_root_array_refuses_the_reader(tmp_path, reader):
    from omlx.patches import moe_offload_compat as offload

    _write(tmp_path, [{"decoder": {"omlx_quant_formats": {"toy": {}}}}])
    if reader == "offload":
        assert offload.moe_offload_compatibility(tmp_path) == (
            False,
            offload.IN_QUANT_FORMAT,
        )
        return
    with pytest.raises(formats.FormatRefusedError) as caught:
        format_load.check_engine_loader(tmp_path, reader)
    assert caught.value.mode == "toy" and caught.value.path == tmp_path
    if reader == "batched":
        assert "0.decoder.omlx_quant_formats" in caught.value.reason
