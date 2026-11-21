# SPDX-License-Identifier: Apache-2.0
"""R2 integration case. Collect under tests/ outside test_formats_oq's fixtures."""

import contextvars
from collections import Counter

from omlx.formats import testing as format_testing

import mlx.core as mx
import pytest

from omlx import formats, oq
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.patches import format_load
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from tests.test_formats_oq import _Codec, _in_format, _quantize, load_model
from tests.test_formats_oq import source as source


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_writer_probes_each_persisted_layer_with_its_accepted_reader(
    source, tmp_path, monkeypatch
):
    """The write admits two readers, distinct by identity: ``_format_reader``'s
    for the write's dot and tile probes, and the one ``verify_written_format_
    checkpoint``'s reload admits. The write probes both faces of every
    persisted layer on the first; the reload then probes every layer's dot
    face once on the second, and no tile face."""
    assert not formats._TESTING.get()
    assert formats._LOADING.get() is None
    # oQ accepts the text source and writes the Qwen3.5 MoE model type that
    # the loader checks. Both spellings belong to this fixture.
    monkeypatch.setattr(
        oq, "_QUANT_FORMAT_VERIFIED", frozenset({"qwen3_5_moe_text", "qwen3_5_moe"})
    )
    monkeypatch.setattr(
        format_testing,
        "_VERIFIED_EXTRA",
        contextvars.ContextVar(
            "omlx_verified_extra",
            default=frozenset(
                (("qwen3_5_moe_text", None), ("qwen3_5_moe", None))
            ),
        ),
    )
    monkeypatch.setattr(
        format_load,
        "PLUGIN_ALLOWLIST",
        format_load.PLUGIN_ALLOWLIST
        | {("qwen3_5_moe_text", "toy"), ("qwen3_5_moe", "toy")},
    )
    write_readers = []
    probes = []
    probe_readers = []
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
                assert write_readers and self is write_readers[-1][0]
                assert formats._LOADING.get() is None and not formats._TESTING.get()
                inputs = {
                    name: layer[name] for name in (self.codes_name, *self.tensors)
                }
                with pytest.raises(formats.FormatRefusedError):
                    type(layer)(self, layer.bits, layer.group_size, k, inputs)
            probe_readers.append(self)
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
    assert len(write_readers) == 2
    write_reader, write_record = write_readers[0]
    verify_reader, verify_record = write_readers[1]
    assert write_reader is not verify_reader
    saved = config["omlx_quant_formats"][writer.mode]
    assert saved["record"] == write_record == verify_record and (
        saved["contract"] == writer.contract
    )
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
    # The write probes both faces of every persisted layer; the
    # verify-on-finalize reload then probes every layer's dot face once on
    # the verify reader and never a tile face.
    dot_faces = [item for item in expected if item[0] == "dot"]
    assert len(probes) == len(expected) + len(dot_faces)
    assert Counter(probes[: len(expected)]) == Counter(expected)
    assert all(reader is write_reader for reader in probe_readers[: len(expected)])
    assert Counter(probes[len(expected) :]) == Counter(dot_faces)
    assert all(reader is verify_reader for reader in probe_readers[len(expected) :])

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
