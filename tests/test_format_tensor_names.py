# SPDX-License-Identifier: Apache-2.0
"""A plugin chooses its packed tensor name for dense and expert matrices."""

from dataclasses import replace

import mlx.core as mx
import pytest

from omlx import formats
from omlx.formats import testing
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.patches import moe_verify_gather
from tests.test_formats_kernels import _toy


class _Packed(formats.QuantFormat):
    mode = "named_payload"
    contract = formats.CONTRACT
    codes_name = "packed_bytes"
    tensors = ("gain",)

    def layout(self, k, n, bits, group_size):
        return {self.codes_name: ((n, k), mx.uint8), "gain": ((n,), mx.float32)}

    def decode(self, tensors, bits, group_size):
        codes = tensors[self.codes_name].astype(mx.float32) - 128
        return codes * tensors["gain"][..., None]


@pytest.fixture
def reader(monkeypatch):
    formats.load()
    monkeypatch.setattr(formats, "_FORMATS", dict(formats._FORMATS))
    return _Packed()


@pytest.mark.parametrize("reserved", ["weight", "scales", "biases", "bias"])
@pytest.mark.parametrize("where", ["codes", "side"])
def test_plugin_registration_refuses_mlx_tensor_names(reader, reserved, where):
    if where == "codes":
        reader.codes_name = reserved
    else:
        reader.tensors = (reserved,)
    with pytest.raises(ValueError, match=reserved):
        formats.register(reader)
    assert formats.get(reader.mode) is None


@pytest.mark.parametrize("name", ["gain", "two words", 19])
def test_packed_tensor_has_its_own_distinct_identifier(reader, name):
    reader.codes_name = name
    with pytest.raises(ValueError):
        formats.register(reader)


@pytest.mark.parametrize("experts", [False, True])
def test_plugin_parameters_and_forward_use_its_named_codes(reader, experts):
    formats.register(reader)
    shape = (2, 8, 8) if experts else (8, 8)
    codes = mx.full(shape, 129, mx.uint8)
    tensors = {reader.codes_name: codes, "gain": mx.ones(shape[:-1])}
    layer_type = FormatSwitchLinear if experts else FormatLinear
    with testing.verified("example"):
        layer = layer_type(reader.mode, 8, 1, 8, tensors)
    assert set(layer.parameters()) == set(tensors)
    for name in ("weight", "scales", "biases"):
        assert name not in layer and not hasattr(layer, name)
    x = mx.ones((1, 1, 8))
    output = layer(x, mx.array([1])) if experts else layer(x)
    assert mx.array_equal(output, mx.full(output.shape, 8)).item()


def test_verify_gather_reads_a_format_switch_layers_declared_operands(reader):
    class Faced(_Packed):
        def dot(self, layer, k, n):
            return replace(_toy("named_payload"), tensors=self.tensors)

    fmt = Faced()
    formats.register(fmt)
    codes = mx.full((2, 8, 8), 129, mx.uint8)
    gains = mx.ones((2, 8))
    with testing.verified("example"):
        layer = FormatSwitchLinear(
            fmt.mode, 8, 1, 8, {fmt.codes_name: codes, "gain": gains}
        )
    projection = moe_verify_gather.projection(layer, 8)
    assert projection is not None and projection.n == 8
    assert projection.operands[0] is codes and projection.operands[1] is gains


def test_quant_formats_doc_test_recipes_use_codes_name():
    """The page's two recipes take a layer's operands by its format's codes
    name: indexing "weight" is a KeyError on any plugin's layer (D11)."""
    from pathlib import Path

    page = (Path(__file__).parents[1] / "docs" / "quant-formats.md").read_text()
    assert '("weight", *face.tensors)' not in page
    assert page.count("(FORMAT.codes_name, *face.tensors)") == 2


def test_bias_codes_name_message_does_not_claim_bias_is_a_side_tensor(reader):
    reader.codes_name = "bias"
    with pytest.raises(ValueError) as refusal:
        formats.register(reader)
    said = str(refusal.value)
    assert "bias" in said and "as one of its tensors" not in said


def test_format_api_docstrings_name_codes_not_weight():
    """A format's tensors are its codes and its own side tensors: the API's
    words say so, and an author reading them cannot think a plugin's layer
    keeps a ``weight``."""
    for described in (formats.QuantFormat.decode, formats.QuantFormat.layout):
        assert "``weight``" not in (described.__doc__ or "")


def test_a_face_on_a_layer_over_placeholders_is_refused_before_it_is_read(reader):
    """The verify gather asks the layer before reading its tensors: a layer
    that still holds a load's placeholder raises, naming it, on every
    geometry (the widths that are not whole tiles of 8 included)."""
    from omlx.formats.linear import FormatSwitchLinear
    from omlx.patches import moe_verify_gather

    formats.register(reader)
    for rows in (8, 12):
        stand_ins = {
            reader.codes_name: mx.zeros((2, rows, 8), mx.uint8),
            "gain": mx.zeros((2, rows)),
        }
        with testing.verified("example"):
            layer = FormatSwitchLinear(
                reader.mode, 8, 1, 8, stand_ins, placeholders=True
            )
            with pytest.raises(ValueError, match="placeholder"):
                moe_verify_gather.projection(layer, 8)


def test_tensor_names_docstring_does_not_claim_mlx_layers_have_no_face():
    """MLX's affine is served through the same faces: the docstring says what
    the names are, not that those layers have no face."""
    assert "run no format's face" not in (formats.tensor_names.__doc__ or "")


def test_the_packed_name_of_a_pair_kernel_is_weight_by_default():
    """``tile_wide.pair_kernel`` names its packed input ``weight`` unless a
    caller passes the format's codes name: the default is MLX's."""
    from omlx.formats import tile_wide

    face = _toy("pair_default")
    names, _ = tile_wide.pair_kernel(face, face)
    assert names[0] == "weight"


def test_rows_stack_docstring_names_codes_not_weight():
    doc = testing.rows_stack.__doc__ or ""
    assert "``weight``" not in doc
    assert doc.count("``fmt.codes_name``") == 1
    assert "(their codes, ``fmt.codes_name``" in doc


def test_layer_refusals_name_a_plugins_codes(reader):
    """The refusals of a layer's own tensors read the format's name, not the
    literal "codes": a plugin with another name gets its own words."""
    formats.register(reader)
    stack = {reader.codes_name: mx.zeros((2, 8, 8), mx.uint8), "gain": mx.zeros((2, 8))}
    with testing.verified("example"):
        with pytest.raises(ValueError, match=r"codes \('packed_bytes'\)"):
            FormatLinear(reader.mode, 8, 1, 8, stack)
        with pytest.raises(
            ValueError, match=r"codes' \('packed_bytes'\) stack of \(2,\)"
        ):
            FormatSwitchLinear(
                reader.mode,
                8,
                1,
                8,
                {**stack, "gain": mx.zeros((3, 8))},
            )
