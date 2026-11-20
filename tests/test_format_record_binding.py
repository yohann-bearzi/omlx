# SPDX-License-Identifier: Apache-2.0
"""A checkpoint's records must account for the layers its loader returns."""

import json

import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import testing
from omlx.patches import format_load
from tests.test_formats_load import _TOY, _checked, _Loadable, _Wide


@pytest.fixture
def toy(monkeypatch):
    formats.load()
    fmt = _Loadable()
    monkeypatch.setitem(formats._FORMATS, fmt.mode, fmt)
    format_load.apply_format_load_patch()
    with testing.verified("llama"):
        yield fmt


@pytest.mark.parametrize("record", [None, {"record": None}, {}])
@pytest.mark.parametrize("other_bound", [False, True])
def test_a_record_without_layers_is_checked_in_its_own_mode(
    tmp_path, toy, record, other_bound
):
    written = {"orphan": record}
    entries, held = {}, {}
    if other_bound:
        written["toy"] = {}
        entries["proj"] = _TOY
        held["proj"] = True
    with pytest.raises(formats.FormatRefusedError) as error:
        _checked(tmp_path, entries, held, omlx_quant_formats=written)
    assert error.value.mode == "orphan"
    if record != {}:
        assert "config is damaged" in error.value.reason
    else:
        assert "no layer" in error.value.reason


@pytest.mark.parametrize(
    "wrapper", ["", "language_model.", "model.", "model.language_model."]
)
@pytest.mark.parametrize("container", [list, dict])
def test_present_mtp_collections_must_bind_the_named_layer(
    tmp_path, toy, wrapper, container
):
    entry = f"{wrapper}mtp.0.bad"
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "quantization": {entry: _TOY},
                "omlx_quant_formats": {"toy": {}},
            }
        )
    )
    root = module = nn.Module()
    for part in wrapper.rstrip(".").split(".") if wrapper else ():
        child = nn.Module()
        module[part] = child
        module = child
    block = nn.Module()
    module.mtp = [block] if container is list else {"0": block}
    with (
        format_load.loading(tmp_path) as checkpoint,
        pytest.raises(formats.FormatRefusedError, match="names .*bad"),
    ):
        checkpoint.check(root)


@pytest.mark.parametrize(
    "entry", ["model.mtp.proj", "model.language_model.mtp.proj", "model.visual.proj"]
)
def test_hf_optional_subtrees_may_be_absent_and_must_bind_when_loaded(
    tmp_path, toy, entry
):
    inner = entry.removeprefix("model.").removeprefix("language_model.")
    _checked(tmp_path, {entry: _TOY}, {"model.unrelated": False})
    _checked(tmp_path, {entry: _TOY}, {inner: True})
    _checked(tmp_path, {entry: _TOY}, {f"language_model.{inner}": True})
    with pytest.raises(formats.FormatRefusedError, match="names .*proj"):
        _checked(tmp_path, {entry: _TOY}, {inner.split(".")[0]: False})


def test_visual_entries_bind_the_loaded_vision_tower(tmp_path, toy):
    _checked(tmp_path, {"model.visual.proj": _TOY}, {"vision_tower.proj": True})
    with pytest.raises(formats.FormatRefusedError, match="names model.visual.proj"):
        _checked(tmp_path, {"model.visual.proj": _TOY}, {"vision_tower.other": True})


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", 1),
        ("contract", []),
        ("contract", True),
        ("contract", 0),
        ("contract", formats.CONTRACT + 1),
        ("plugin", 7),
    ],
)
def test_a_bound_layer_refuses_an_unknown_record_envelope(tmp_path, toy, field, value):
    with pytest.raises(formats.FormatRefusedError) as error:
        _checked(
            tmp_path,
            {"a": _TOY},
            {"a": True},
            omlx_quant_formats={"toy": {field: value}},
        )
    assert error.value.mode == "toy"
    assert field in error.value.reason and "layout" in error.value.reason


def test_damage_in_a_record_is_refused_even_when_its_mtp_head_is_absent(tmp_path, toy):
    with pytest.raises(formats.FormatRefusedError, match="config is damaged"):
        _checked(tmp_path, {"mtp.proj": _TOY}, {}, omlx_quant_formats={"toy": None})


def test_a_repeated_layout_uses_the_accepted_format_once_per_matrix_at_most(
    tmp_path, toy
):
    wide = _Wide()
    toy.reads = lambda record: wide
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with format_load.loading(tmp_path):
        for _ in range(4):
            result = nn.Linear(64, 128).to_quantized(**_TOY)
            assert result["group_scales"].shape == (128, 2)
        assert toy.asked == [(64, 128, 8, 1)]
        assert wide.asked == [(64, 128, 8, 1)]
        assert len(toy.accepted) == 1
        # The accepted format lays out 4-bit shapes, the registered one does not.
        with pytest.raises(formats.FormatRefusedError, match="lays out no"):
            nn.Linear(64, 128).to_quantized(group_size=1, bits=4, mode="toy")


def test_the_registered_object_returned_by_accept_lays_out_a_layer_once(tmp_path, toy):
    toy.reads = lambda record: toy
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with (
        format_load.loading(tmp_path),
        pytest.raises(
            formats.FormatRefusedError, match="accept returned the registered format"
        ),
    ):
        nn.Linear(64, 128).to_quantized(**_TOY)
    assert toy.asked == [(64, 128, 8, 1)]


def test_a_layer_built_without_a_config_marker_still_belongs_to_its_model(
    tmp_path, toy
):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with format_load.loading(tmp_path) as checkpoint:
        nn.Linear(64, 128).to_quantized(**_TOY)
        with pytest.raises(formats.FormatRefusedError, match="not in its model"):
            checkpoint.check(nn.Module())
