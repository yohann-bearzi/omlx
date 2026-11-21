# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3 pair: not shipped verified; test blocks mark a tiny one."""

import json

import pytest

from omlx import formats, oq
from omlx.formats import testing
from omlx.patches import format_load

_PAIR = {
    "model_type": "glm5_next",
    "text_config": {"model_type": "glm5_next_text"},
}


def test_the_shipped_gate_refuses_the_glm53_pair(tmp_path):
    config = {**_PAIR, "omlx_quant_formats": {"example": {}}}
    assert oq._unverified_family(config) is not None
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError):
        format_load.check_engine_loader(tmp_path, "vlm")


def test_a_test_block_admits_the_glm53_pair(tmp_path):
    config = {**_PAIR, "omlx_quant_formats": {"example": {}}}
    with testing.verified("glm5_next", "glm5_next_text"):
        assert oq._unverified_family(config) is None
        (tmp_path / "config.json").write_text(json.dumps(config))
        format_load.check_engine_loader(tmp_path, "vlm")
        with format_load.loading(tmp_path) as checkpoint:
            assert checkpoint.verified


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "glm5_next"},
        {"model_type": "glm5_next_text"},
        {"model_type": "glm5_next", "text_config": {"model_type": "other"}},
        {"model_type": "other", "text_config": {"model_type": "glm5_next_text"}},
        {
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "model_file": "custom.py",
            },
        },
    ],
)
def test_only_the_stock_glm53_pair_is_admitted(tmp_path, config):
    config = {**config, "omlx_quant_formats": {"example": {}}}
    assert oq._unverified_family(config) is not None
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError):
        format_load.check_engine_loader(tmp_path, "vlm")
