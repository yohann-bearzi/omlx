# SPDX-License-Identifier: Apache-2.0
"""Production admission for the Flash-Next config shipped by the PoC."""

import json

import pytest

from omlx import formats, oq
from omlx.patches import format_load


@pytest.mark.parametrize(
    "outer,inner",
    [("qwen4_exp", "qwen4_exp_text"), (" QWEN4-EXP ", "QWEN4-EXP-TEXT")],
)
def test_flash_next_writer_and_checked_loader_admit_its_format_record(
    tmp_path, outer, inner
):
    config = {
        "model_type": outer,
        "text_config": {"model_type": inner},
        "omlx_quant_formats": {"example": {}},
    }
    assert oq._unverified_family(config) is None
    (tmp_path / "config.json").write_text(json.dumps(config))
    format_load.check_engine_loader(tmp_path, "vlm")
    with format_load.loading(tmp_path) as checkpoint:
        assert checkpoint.verified
    with pytest.raises(formats.FormatRefusedError, match="embedding"):
        format_load.check_engine_loader(tmp_path, "embedding")


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen4_exp_text"},
        {"model_type": "qwen4_exp"},
        {"model_type": "qwen4_exp", "text_config": {}},
        {"model_type": "qwen4_exp", "text_config": {"model_type": "other"}},
        {"model_type": "other", "text_config": {"model_type": "qwen4_exp_text"}},
        {
            "model_type": "qwen4_exp",
            "text_config": {"model_type": "qwen4_exp_text", "model_file": "custom.py"},
        },
        {
            "model_type": "qwen4_exp",
            "model_file": "custom.py",
            "text_config": {"model_type": "qwen4_exp_text"},
        },
    ],
)
def test_only_the_stock_implementation_of_the_served_pair_is_admitted(tmp_path, config):
    config["omlx_quant_formats"] = {"example": {}}
    assert oq._unverified_family(config) is not None
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError):
        format_load.check_engine_loader(tmp_path, "vlm")


def test_stock_flash_next_does_not_open_format_construction(tmp_path):
    config = {
        "model_type": "qwen4_exp",
        "text_config": {"model_type": "qwen4_exp_text"},
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with format_load.loading(tmp_path) as checkpoint:
        assert not checkpoint.verified
        with pytest.raises(formats.FormatRefusedError):
            formats._may_build("example")
