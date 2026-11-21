# SPDX-License-Identifier: Apache-2.0
"""Production admission for the Qwen3.5/3.6 MoE family."""

import json

import pytest

from omlx import formats, oq
from omlx.model_discovery import detect_model_type
from omlx.patches import format_load


@pytest.mark.parametrize(
    ("config", "loader"),
    [
        ({"model_type": "qwen3_5_moe_text"}, "batched"),
        ({"model_type": "qwen3_5_moe"}, "batched"),
        (
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"model_type": "qwen3_5_moe_text"},
                "vision_config": {"depth": 1},
            },
            "vlm",
        ),
    ],
)
def test_qwen35_moe_writer_and_checked_loader_admit_its_format_record(
    tmp_path, config, loader
):
    config = {**config, "omlx_quant_formats": {"example": {}}}
    assert oq._unverified_family(config) is None
    (tmp_path / "config.json").write_text(json.dumps(config))
    format_load.check_engine_loader(tmp_path, loader)
    assert detect_model_type(tmp_path) == ("vlm" if loader == "vlm" else "llm")
    with format_load.loading(tmp_path) as checkpoint:
        assert checkpoint.verified
    if loader == "vlm":
        with pytest.raises(formats.FormatRefusedError, match="embedding"):
            format_load.check_engine_loader(tmp_path, "embedding")


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3_5_moe_text", "text_config": {"model_type": "other"}},
        {"model_type": "other", "text_config": {"model_type": "qwen3_5_moe_text"}},
        {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "model_file": "custom.py",
            },
        },
        {"model_type": "qwen3_5_moe", "text_config": {}},
        {"model_type": "qwen3_5_moe", "text_config": {"model_type": " "}},
        {"model_type": "qwen3_5_moe", "model_file": "custom.py"},
    ],
)
def test_only_the_stock_qwen35_pair_is_admitted(tmp_path, config):
    config = {**config, "omlx_quant_formats": {"example": {}}}
    assert oq._unverified_family(config) is not None
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError):
        format_load.check_engine_loader(tmp_path, "vlm")
