# SPDX-License-Identifier: Apache-2.0
"""R2 integration case. Collect under tests/ outside test_formats_oq's fixtures."""

from collections import Counter

import mlx.core as mx
import pytest

from omlx import formats, oq
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from tests.test_formats_oq import _Codec, _in_format, _quantize, load_model
from tests.test_formats_oq import source as source


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_writer_probes_each_persisted_layer_with_its_accepted_reader(
    source, tmp_path, monkeypatch
):
    assert not formats._TESTING and not oq._TEST_VERIFIED
    assert formats._LOADING.get() is None
    # oQ accepts the text source and writes the Qwen3.5 MoE model type that
    # the loader checks. Both spellings belong to this fixture.
    monkeypatch.setattr(
        oq, "_QUANT_FORMAT_VERIFIED", frozenset({"qwen3_5_moe_text", "qwen3_5_moe"})
    )
    monkeypatch.setattr(
        oq, "_TEST_VERIFIED", frozenset({"qwen3_5_moe_text", "qwen3_5_moe"})
    )
    write_readers = []
    probes = []
    writing = True

    class Codec(_Codec):
        # Make the stored width differ from K, as a packed format's does.
        # Repeating one byte per row keeps the toy's decoded weights exact.
        def layout(self, k, n, bits, group_size):
            layout = super().layout(k, n, bits, group_size)
            layout["codes"] = ((n, k + 1), mx.uint8)
            return layout

        def encode(self, w, importance, bits, group_size):
            tensors = super().encode(w, importance, bits, group_size)
            codes = tensors["codes"]
            tensors["codes"] = mx.concatenate((codes, codes[..., :1]), axis=-1)
            return tensors

        def decode(self, tensors, bits, group_size):
            return super().decode(
                {**tensors, "codes": tensors["codes"][..., :-1]}, bits, group_size
            )

        def accept(self, record):
            reader = Codec()
            if writing:
                write_readers.append((reader, record))
            return reader

        def probe(self, face, layer, k, n):
            assert layer.fmt is self
            assert self is not writer and self is not prototype
            if writing:
                assert formats._LOADING.get() is None and not formats._TESTING
                inputs = {name: layer[name] for name in (self.codes_name, *self.tensors)}
                with pytest.raises(formats.FormatRefusedError):
                    type(layer)(self, layer.bits, layer.group_size, k, inputs)
            probes.append(
                (
                    face,
                    type(layer),
                    layer[self.codes_name].shape,
                    k,
                    n,
                    layer.bits,
                    layer.group_size,
                )
            )

        def dot(self, layer, k, n):
            # the write verifies its own faces through the writer and prototype (D11)
            if self is not writer and self is not prototype:
                self.probe("dot", layer, k, n)

        def tile(self, layer, k, n):
            # the write verifies its own faces through the writer and prototype (D11)
            if self is not writer and self is not prototype:
                self.probe("tile", layer, k, n)

    formats.load()
    writer = Codec()
    prototype = Codec()
    monkeypatch.setitem(formats._FORMATS, writer.mode, prototype)
    output = tmp_path / "checkpoint"
    tensors, config, applied = _quantize(source, output, quant_format=writer)
    writing = False
    entries = _in_format(config, writer.mode)
    assert entries and sorted(entries) == applied
    assert write_readers
    saved = config["omlx_quant_formats"][writer.mode]
    records = [record for _, record in write_readers]
    assert saved["record"] in records and saved["contract"] == writer.contract
    expected = []
    for name, entry in entries.items():
        shape = tensors[f"{name}.codes"].shape
        cls = FormatLinear if len(shape) == 2 else FormatSwitchLinear
        for face in ("dot", "tile"):
            expected.append(
                (
                    face,
                    cls,
                    shape,
                    shape[-1] - 1,
                    shape[-2],
                    entry["bits"],
                    entry["group_size"],
                )
            )
    assert {item[1] for item in expected} == {FormatLinear, FormatSwitchLinear}
    # io's verify-on-finalize reload probes every persisted layer's dot face
    # once more through the accepted reader; the tile faces come from the
    # write alone. Each layer and geometry is otherwise the write's.
    dot_faces = Counter(item for item in expected if item[0] == "dot")
    tile_faces = Counter(item for item in expected if item[0] == "tile")
    assert Counter(probes) == dot_faces + dot_faces + tile_faces

    maybe_apply_pre_load_patches(str(output))
    model, _ = load_model(output, strict=True)
    loaded = {
        name
        for name, layer in model.named_modules()
        if isinstance(layer, (FormatLinear, FormatSwitchLinear))
    }
    assert loaded == entries.keys()
    logits = model(mx.array([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 64) and mx.isfinite(logits).all()
