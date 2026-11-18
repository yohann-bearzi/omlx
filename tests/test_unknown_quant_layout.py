# SPDX-License-Identifier: Apache-2.0
"""A bound layer must not hide an unknown or malformed quantization layout."""

import json

import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import testing
from omlx.patches import format_load
from tests.test_formats_load import _TOY, _Loadable


@pytest.mark.parametrize(
    "key,value",
    [
        ("quant_method", "vendor-format"),
        ("per_layer", ["proj", "extra"]),
        ("extra", "toy"),
        ("extra", {"vendor_mode": "toy"}),
        ("extra", {"mode": ["toy"]}),
        ("extra", {"mode": "affine", "bits": "four", "group_size": 64}),
        ("groups", {"bundle": {"mode": "toy"}}),
    ],
)
def test_unknown_quantization_layout_names_the_unrecognised_field(
    tmp_path, monkeypatch, key, value
):
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _Loadable())
    format_load.apply_format_load_patch()
    config = {
        "model_type": "llama",
        "omlx_quant_formats": {"toy": {}},
        "quantization": {"group_size": 64, "bits": 4, "proj": _TOY, key: value},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with (
        testing.verified("llama"),
        pytest.raises(formats.FormatRefusedError) as error,
        format_load.loading(tmp_path) as checkpoint,
    ):
        model = nn.Module()
        model.proj = nn.Linear(64, 128).to_quantized(**_TOY)
        checkpoint.check(model)
    assert key in error.value.reason and "layout" in error.value.reason
    assert error.value.path == tmp_path


@pytest.mark.parametrize(
    "holder,entry",
    [
        ("quantization", []),
        ("quantization_config", "vendor"),
        ("text_config", {"quantization": ["proj"]}),
        ("text_config", "vendor"),
        ("text_config", {"omlx_quant_formats": {"toy": {}}}),
    ],
)
def test_an_unknown_sibling_layout_is_refused_alongside_a_valid_one(
    tmp_path, monkeypatch, holder, entry
):
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy", _Loadable())
    format_load.apply_format_load_patch()
    flat = {"group_size": 64, "bits": 4, "proj": _TOY}
    config = {
        "model_type": "llama",
        "omlx_quant_formats": {"toy": {}},
        "quantization": flat,
        "quantization_config": flat,
        holder: entry,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with (
        testing.verified("llama"),
        pytest.raises(formats.FormatRefusedError) as error,
        format_load.loading(tmp_path) as checkpoint,
    ):
        model = nn.Module()
        model.proj = nn.Linear(64, 128).to_quantized(**_TOY)
        checkpoint.check(model)
    assert holder in error.value.reason and "layout" in error.value.reason


@pytest.mark.parametrize(
    "config,field",
    [
        ({"omlx_quant_formats": {}}, "omlx_quant_formats"),
        ({"omlx_quant_formats": {" ": {}}}, "omlx_quant_formats"),
        ({"text_config": {"omlx_quant_formats": {}}}, "text_config.omlx_quant_formats"),
        (
            {"text_config": {"omlx_quant_formats": {"toy": {}}}},
            "text_config.omlx_quant_formats",
        ),
    ],
)
def test_empty_or_misplaced_format_marker_cannot_pass_a_verified_load(
    tmp_path, config, field
):
    (tmp_path / "config.json").write_text(json.dumps(dict(config, model_type="llama")))
    with (
        testing.verified("llama"),
        pytest.raises(formats.FormatRefusedError) as error,
        format_load.loading(tmp_path) as checkpoint,
    ):
        checkpoint.check(nn.Module())
    assert field in error.value.reason and "layout" in error.value.reason
