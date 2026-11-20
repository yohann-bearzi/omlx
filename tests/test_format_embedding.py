# SPDX-License-Identifier: Apache-2.0
"""Packed embedding lookup and tied projection over one admitted reader."""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import linear, testing
from tests.test_qwen35_moe_routed_decode import _same_bits


class RowReader(formats.QuantFormat):
    mode = "embedding_test"
    codes_name = "codes"
    tensors = ("offset",)
    contract = 1

    def __init__(self):
        self.reads = []

    def layout(self, k, n, bits, group_size):
        return {
            "codes": ((n, k), mx.float32),
            "offset": ((n, 1), mx.float32),
        }

    def decode(self, tensors, bits, group_size):
        codes, offset = tensors["codes"], tensors["offset"]
        self.reads.append((codes.shape, offset.shape, bits, group_size))
        return codes + offset


def _packed(shared=False):
    codes = mx.arange(72, dtype=mx.float32).reshape(9, 8)
    offset = mx.arange(1 if shared else 9, dtype=mx.float32)[:, None] + 1
    return {"codes": codes, "offset": offset}


@pytest.mark.parametrize("dtype,bits", [(mx.float16, 4), (mx.bfloat16, 8)])
def test_affine_embedding_matches_mlx_lookup_and_tied_projection(dtype, bits):
    mx.random.seed(37)
    dense = nn.Embedding(64, 256)
    dense.set_dtype(dtype)
    stock = dense.to_quantized(group_size=32, bits=bits)
    arrays = {name: stock[name] for name in ("weight", "scales", "biases")}
    with testing.verified("llama"):
        layer = linear.FormatEmbedding("affine", bits, 32, 256, arrays)
    assert (layer.num_embeddings, layer.dims, layer.bits, layer.group_size) == (
        64,
        256,
        bits,
        32,
    )
    assert layer.mode == stock.mode and layer.fmt is formats.get("affine")
    assert all(layer[name] is value for name, value in arrays.items())
    ids = mx.array([[63, 0], [5, 63]])
    rows = layer(ids)
    assert rows.dtype == dtype and _same_bits(rows, stock(ids))
    x = mx.random.normal((1, 256)).astype(dtype)
    for _ in range(2):
        result = layer.as_linear(x)
        assert result.dtype == dtype and _same_bits(result, stock.as_linear(x))
        # Both tied operations must see a replacement train/load may make.
        layer.scales = stock.scales = stock.scales * 2
        assert _same_bits(layer(ids), stock(ids))


@pytest.mark.parametrize("rows", [1, 8])
def test_k64_affine_tied_projection_keeps_the_dense_quad_parity_exception(rows):
    mx.random.seed(300)
    source = nn.Embedding(128, 64)
    source.set_dtype(mx.bfloat16)
    stock = source.to_quantized(bits=4, group_size=32)
    tensors = {name: stock[name] for name in ("weight", "scales", "biases")}
    with testing.verified("llama"):
        embedding = linear.FormatEmbedding("affine", 4, 32, 64, tensors)
        dense = linear.FormatLinear("affine", 4, 32, 64, tensors)

    x = mx.random.normal((rows, 64)).astype(mx.bfloat16)
    tied = embedding.as_linear(x)
    assert _same_bits(tied, dense(x))
    # Revisit this witness when GLM C6b adds the qmv_quad face.
    assert not _same_bits(tied, stock.as_linear(x))


@pytest.mark.parametrize("shape", [(), (1,), (2, 2), (2, 0)])
def test_row_reader_decodes_only_the_requested_rows(shape):
    reader = RowReader()
    tensors = _packed()
    with testing.verified("llama"):
        layer = linear.FormatEmbedding(reader, 4, 8, 8, tensors)
    ids = mx.full(shape, 6, dtype=mx.int32)
    actual = layer(ids)
    expected = (tensors["codes"] + tensors["offset"])[ids]
    assert actual.shape == (*shape, 8) and mx.array_equal(actual, expected).item()
    # The decoder accepts matrices. A scalar lookup is still one row.
    assert all(w[-1] == 8 and o[-1] == 1 and len(w) >= 2 for w, o, _, _ in reader.reads)
    assert all(w[:-1] == o[:-1] and (b, g) == (4, 8) for w, o, b, g in reader.reads)
    assert sum(math.prod(w[:-1]) for w, _, _, _ in reader.reads) == ids.size


@pytest.mark.parametrize("declared", [True, False], ids=["shared-offset", "no-layout"])
def test_a_format_whose_rows_are_not_independent_is_refused(declared):
    """A lookup decodes the rows it names only; a format that cannot promise
    that is refused before a checkpoint builds it, never decoded whole."""

    class CenteredReader(RowReader):
        def layout(self, k, n, bits, group_size):
            if declared:
                return {"codes": ((n, k), mx.float32), "offset": ((1, 1), mx.float32)}
            return None

        def decode(self, tensors, bits, group_size):
            values = super().decode(tensors, bits, group_size)
            return values - values.mean(axis=-2, keepdims=True)

    reader = CenteredReader()
    tensors = _packed(shared=declared)
    assert not formats.stackable(reader, 8, 9, 4, 8)
    with testing.verified("llama"):
        with pytest.raises(ValueError, match="rows are not independent"):
            linear.FormatEmbedding(reader, 4, 8, 8, tensors)
    assert reader.reads == []


def test_builtin_affine_decodes_only_selected_token_rows(monkeypatch):
    stock = nn.Embedding(64, 256).to_quantized(group_size=32, bits=4)
    arrays = {name: stock[name] for name in ("weight", "scales", "biases")}
    reader = formats.get("affine")
    sizes = []
    decode = reader.decode

    def traced(tensors, bits, group_size):
        sizes.append({name: math.prod(t.shape[:-1]) for name, t in tensors.items()})
        return decode(tensors, bits, group_size)

    monkeypatch.setattr(reader, "decode", traced)
    with testing.verified("llama"):
        layer = linear.FormatEmbedding(reader, 4, 32, 256, arrays)
    ids = mx.array([63, 0, 63])
    assert mx.array_equal(layer(ids), stock(ids)).item()
    assert sizes == [dict.fromkeys(arrays, 3)]


def test_embedding_requires_admission_and_keeps_its_supplied_reader():
    reader = RowReader()
    tensors = _packed()
    with pytest.raises(formats.FormatRefusedError) as refused:
        linear.FormatEmbedding(reader, 4, 8, 8, tensors)
    assert refused.value.mode == reader.mode
    with testing.verified("llama"):
        layer = linear.FormatEmbedding(reader, 4, 8, 8, tensors)
    assert layer.fmt is reader
    x = mx.ones((2, 8))
    assert mx.array_equal(
        layer.as_linear(x), x @ (layer["codes"] + layer["offset"]).T
    ).item()


@pytest.mark.parametrize("operation", ["lookup", "as_linear"])
def test_unfilled_embeddings_refuse_before_decoding(operation):
    reader = RowReader()
    with testing.verified("llama"):
        layer = linear.FormatEmbedding(reader, 4, 8, 8, _packed(), placeholders=True)
    with pytest.raises(ValueError, match="placeholder"):
        if operation == "lookup":
            layer(mx.array([1]))
        else:
            layer.as_linear(mx.ones((1, 8)))
    assert not reader.reads


def test_tied_projection_uses_a_dense_view_of_the_current_embedding_arrays(monkeypatch):
    from copy import copy

    reader = RowReader()
    with testing.verified("llama"):
        layer = linear.FormatEmbedding(reader, 4, 8, 8, _packed())
    receivers = []
    call = linear.FormatLinear.__call__

    def observe(view, x):
        receivers.append(view)
        return call(view, x)

    monkeypatch.setattr(linear.FormatLinear, "__call__", observe)
    x = mx.ones((1, 8))

    def check(embedding):
        before = len(receivers)
        result = embedding.as_linear(x)
        assert len(receivers) == before + 1
        view = receivers[-1]
        assert type(view) is linear.FormatLinear and view.fmt is reader
        assert view["codes"] is embedding["codes"]
        assert view["offset"] is embedding["offset"]
        assert set(embedding.parameters()) == {"codes", "offset"}
        assert mx.array_equal(
            result, x @ (embedding["codes"] + embedding["offset"]).T
        ).item()

    check(layer)
    check(layer)
    assert receivers[0] is receivers[1]
    clone = copy(layer)
    clone["offset"] = mx.zeros_like(clone["offset"])
    check(clone)
    check(layer)
    layer["codes"] = layer["codes"] + 32
    check(layer)
    check(clone)


def test_a_checkpoints_embedding_loads_as_the_formats_layer(tmp_path, monkeypatch):
    """The load hook rebuilds ``nn.Embedding`` as ``FormatEmbedding`` when a
    format lays its matrix out (D11: through the format's own names)."""
    import json

    from omlx.patches import format_load

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "omlx_quant_formats": {"embedding_test": {}},
            }
        )
    )
    reader = RowReader()
    formats.load()
    monkeypatch.setattr(formats, "_FORMATS", dict(formats._FORMATS))
    formats.register(reader)
    format_load.apply_format_load_patch()
    with testing.verified("llama"), format_load.loading(tmp_path) as checkpoint:
        built = nn.Embedding(64, 64).to_quantized(
            group_size=32, bits=4, mode="embedding_test"
        )
    assert type(built) is linear.FormatEmbedding
    assert built.fmt is not reader  # the checkpoint's own copy
    assert (built.num_embeddings, built.dims) == (64, 64)
    assert set(built.parameters()) == {"codes", "offset"}
    assert built in checkpoint.built
    # A stock mode and a checkpoint of no record stay MLX's.
    with testing.verified("llama"), format_load.loading(tmp_path):
        assert type(nn.Embedding(64, 64).to_quantized(group_size=32, bits=4)) is (
            nn.QuantizedEmbedding
        )


def test_as_linear_tracks_weights_across_compile_state_swaps(monkeypatch):
    from omlx import formats

    mx.random.seed(3)
    x = mx.random.normal((3, 8)).astype(mx.float32)
    formats.load()
    monkeypatch.setattr(formats, "_FORMATS", dict(formats._FORMATS))
    formats.register(RowReader())
    with testing.verified("llama"):
        layer = linear.FormatEmbedding(
            "embedding_test", 8, 1, 8, _packed(), placeholders=False
        )
    step = mx.compile(lambda value: layer.as_linear(value), inputs=layer.state)
    for call in (step, layer.as_linear):
        assert _same_bits(call(x), x @ (layer["codes"] + layer["offset"]).T)
    layer["codes"] = layer["codes"] + 16
    for call in (step, layer.as_linear):
        assert _same_bits(call(x), x @ (layer["codes"] + layer["offset"]).T)


def test_embedding_invalid_tensors_raise_value_error():
    formats.load()
    with testing.verified("llama"):
        with pytest.raises(ValueError):
            linear.FormatEmbedding(
                "affine", 8, 64, 8, {"scales": mx.zeros((4, 1), mx.bfloat16)}
            )
        with pytest.raises(ValueError):
            linear.FormatEmbedding(
                "affine",
                8,
                64,
                8,
                {
                    "weight": mx.zeros((4,), mx.uint32),
                    "scales": mx.zeros((4, 1), mx.bfloat16),
                    "biases": mx.zeros((4, 1), mx.bfloat16),
                },
            )
