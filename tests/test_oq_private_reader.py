# SPDX-License-Identifier: Apache-2.0
"""Reader ownership on an ordinary streaming write, before calibration or output."""

import mlx.core as mx
import pytest

from omlx import formats, oq
from tests.test_formats_oq import _Codec, _quantize
from tests.test_formats_oq import source as source


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    ("stage", "alias"),
    [
        ("deepcopy", "singleton"),
        ("deepcopy", "nested"),
        ("accept", "singleton"),
        ("accept", "nested"),
        ("accept", "in_place"),
    ],
)
def test_streaming_writer_refuses_a_reader_that_shares_the_registered_format(
    source, tmp_path, monkeypatch, stage, alias
):
    calls = []
    origin = "alias-codec 0.1"

    class Codec(_Codec):
        def __init__(self):
            super().__init__()
            self.settings = {"knots": ([1, 2],)}

        def aliased_reader(self):
            if alias == "singleton":
                return registered
            other = Codec()
            other.settings["knots"] = (registered.settings["knots"][0],)
            return other

        def __deepcopy__(self, memo):
            calls.append("copy")
            return self.aliased_reader() if stage == "deepcopy" else Codec()

        def accept(self, record):
            calls.append("accept")
            self.settings["knots"][0].append(3)
            if alias == "in_place":
                self.settings["knots"] = (registered.settings["knots"][0],)
                return None
            return self.aliased_reader() if stage == "accept" else self

    assert not formats._TESTING.get()
    assert formats._LOADING.get() is None
    formats.load()
    registered = Codec()
    monkeypatch.setitem(formats._FORMATS, registered.mode, registered)
    monkeypatch.setitem(formats._ORIGINS, registered.mode, origin)
    output = tmp_path / "oqe"
    sampled = []
    with pytest.raises(formats.FormatRefusedError) as refused:
        _quantize(source, output, sampled=sampled, quant_format=registered.mode)
    message = str(refused.value)
    assert origin in message and "registered" in message and stage in message, (
        message,
        calls,
        registered.settings,
    )
    assert calls == (["copy"] if stage == "deepcopy" else ["copy", "accept"])
    assert registered.settings == {"knots": ([1, 2],)}
    assert not sampled and not registered.encoded
    assert not output.exists() and not output.with_suffix(".unverified").exists()
