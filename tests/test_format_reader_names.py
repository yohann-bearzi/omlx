# SPDX-License-Identifier: Apache-2.0
"""Writer objects and accepted readers must agree on the persisted codes name."""

import json

import pytest

from omlx import formats, oq
from omlx.formats import testing
from omlx.patches import format_load


class _Reader(formats.QuantFormat):
    mode = "named_reader"
    codes_name = "packed_bytes"
    tensors = ("gain",)
    contract = formats.CONTRACT

    def accept(self, record):
        self.codes_name = record.get("packed", self.codes_name)


@pytest.mark.parametrize("mismatch", ["configured", "accepted"])
def test_writer_refuses_a_reader_with_a_different_packed_name(
    tmp_path, monkeypatch, mismatch
):
    class Writer(_Reader):
        def record(self):
            return {"packed": "different"} if mismatch == "accepted" else {}

    prototype = Writer()
    formats.load()
    monkeypatch.setitem(formats._FORMATS, prototype.mode, prototype)
    writer = Writer()
    if mismatch == "configured":
        writer.codes_name = "different"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "unlisted"}))
    with pytest.raises(ValueError, match="checkpoint would not load"):
        oq.quantize_oq_streaming(
            str(tmp_path),
        str(tmp_path / "written"),
        4,
        enhanced=True,
        quant_format=writer,
        )
    assert not (tmp_path / "written").exists()


def test_checkpoint_refuses_an_accepted_reader_with_a_different_packed_name(tmp_path):
    prototype = _Reader()
    config = {
        "model_type": "example",
        "omlx_quant_formats": {prototype.mode: {"record": {"packed": "different"}}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with testing.verified("example"), format_load.loading(tmp_path) as checkpoint:
        with pytest.raises(formats.FormatRefusedError, match="accept returned"):
            checkpoint.read(prototype)
