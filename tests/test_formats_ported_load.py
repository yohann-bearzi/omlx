# SPDX-License-Identifier: Apache-2.0
"""The built-in affine switch is confined to Qwen4 checkpoint loads."""

import json
from contextlib import nullcontext

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.formats.linear import FormatLinear
from omlx.patches import format_load


@pytest.mark.parametrize(
    "family,nested",
    [
        ("qwen4_exp", False),
        ("qwen4_exp", True),
        ("llama", False),
        (None, False),
    ],
)
@pytest.mark.parametrize("embedding", [False, True], ids=["dense", "embedding"])
def test_builtin_affine_only_switches_in_ported_checkpoint_loads(
    family, nested, embedding, tmp_path
):
    format_load.apply_format_load_patch()
    context = nullcontext()
    if family:
        config = (
            {"text_config": {"model_type": family}}
            if nested
            else {"model_type": family}
        )
        config["quantization"] = {"group_size": 64, "bits": 4}
        (tmp_path / "config.json").write_text(json.dumps(config))
        context = format_load.loading(tmp_path)
    with mx.stream(mx.cpu), context:
        module = nn.Embedding(32, 128) if embedding else nn.Linear(128, 32, bias=False)
        packed = module.to_quantized(group_size=64, bits=4)
    if family != "qwen4_exp":
        expected = nn.QuantizedEmbedding if embedding else nn.QuantizedLinear
        assert type(packed) is expected
    elif embedding:
        from omlx.formats import linear

        assert type(packed).__name__ == "FormatEmbedding"
        assert type(packed) is linear.FormatEmbedding
    else:
        assert type(packed) is FormatLinear


def test_ported_family_normalises_the_spellings():
    assert format_load.ported_family({"model_type": " Qwen4-Exp "})
    assert format_load.ported_family({"text_config": {"model_type": "qwen4_exp"}})
    assert format_load.ported_family(
        {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}}
    )
    assert not format_load.ported_family(
        {"model_type": "qwen4_exp", "text_config": {"model_type": "other"}}
    )
    assert not format_load.ported_family(
        {"model_type": "qwen4_exp", "model_file": "custom.py"}
    )
    assert not format_load.ported_family({"model_type": "llama"})
    assert not format_load.ported_family({})


def test_model_config_override_disables_stock_switch(tmp_path):
    from omlx import formats

    format_load.apply_format_load_patch()
    config = {"model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 64}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    for override in (
        {"model_type": "llama"},
        {"model_file": "custom.py"},
        {"omlx_oqh": {"version": 1, "tensors": {}}},
    ):
        with mx.stream(mx.cpu), format_load.loading(tmp_path) as ckpt:
            ckpt.apply_model_config(override)
            assert not ckpt.verified
            packed = nn.Linear(128, 32, bias=False).to_quantized(
                group_size=64, bits=4
            )
            assert type(packed) is nn.QuantizedLinear
            with pytest.raises(formats.FormatRefusedError):
                formats._may_build("affine")


def test_get_model_classes_override_disables_stock_switch(tmp_path):
    from types import SimpleNamespace

    from omlx import formats

    format_load.apply_format_load_patch()
    config = {"model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 64}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    built = []

    def fake_load(model_path, get_model_classes=None):
        current = format_load._current()
        assert current is not None and not current.verified
        with mx.stream(mx.cpu):
            built.append(
                nn.Linear(128, 32, bias=False).to_quantized(group_size=64, bits=4)
            )
        with pytest.raises(formats.FormatRefusedError):
            formats._may_build("affine")
        return built[-1]

    module = SimpleNamespace(
        __name__="mlx_lm.utils", load_model=fake_load, _get_classes=object()
    )
    format_load._wrap_loader(module, "load_model")
    layer = module.load_model(tmp_path, get_model_classes=lambda cfg: None)
    assert type(layer) is nn.QuantizedLinear
