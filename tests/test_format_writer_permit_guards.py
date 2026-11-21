# SPDX-License-Identifier: Apache-2.0
"""A write permits exactly its probe layer, outside a verified model load."""

from concurrent.futures import ThreadPoolExecutor
import contextvars
from contextvars import copy_context

import mlx.core as mx
import pytest

from omlx import formats, oq
from omlx.formats import testing as format_testing
from omlx.formats import linear
from tests.test_formats_oq import _Codec, _quantize
from tests.test_formats_oq import source as source


def _refused(cls, args, **kwargs):
    with pytest.raises(formats.FormatRefusedError):
        cls(*args, **kwargs)


def _observe_write(source, tmp_path, monkeypatch, inspect, codec=_Codec):
    """Observe the first actual writer constructor with its exact arguments."""
    assert not formats._TESTING.get()
    assert formats._LOADING.get() is None
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
    formats.load()
    writer = codec()
    monkeypatch.setitem(formats._FORMATS, writer.mode, codec())
    init = linear._FormatLayer.__init__

    class ObservedError(Exception):
        pass

    def observe(layer, *args, **kwargs):
        token = kwargs.get("_write_permit")
        if token is None:
            return init(layer, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(linear._FormatLayer, "__init__", init)
            inspect(init, layer, args, token)
        raise ObservedError

    with monkeypatch.context() as patch:
        patch.setattr(linear._FormatLayer, "__init__", observe)
        with pytest.raises(ObservedError):
            _quantize(source, tmp_path / "write", quant_format=writer)


def test_layer_arguments_cannot_mint_a_permit_outside_an_admitted_write(
    source, tmp_path, monkeypatch
):
    captured = []

    def observe(init, layer, args, token):
        init(layer, *args, _write_permit=token)
        captured.append((type(layer), args, token))

    _observe_write(source, tmp_path, monkeypatch, observe)
    (seen,) = captured
    cls, args, token = seen
    # A new instance of the token's type (even fitted to all the same layer
    # arguments) can never authorize another construction.
    try:
        other = type(token)(cls, *args)
    except TypeError:
        other = type(token)()
    with pytest.raises(formats.FormatRefusedError):
        cls(*args, _write_permit=other)
    with pytest.raises(formats.FormatRefusedError):
        cls(*args, _write_permit=token)


def test_write_permit_checks_the_entire_construction_and_is_spent_once(
    source, tmp_path, monkeypatch
):
    def observe(init, layer, args, token):
        cls = type(layer)
        other = (
            linear.FormatSwitchLinear
            if cls is linear.FormatLinear
            else linear.FormatLinear
        )
        reader, bits, group, k, tensors = args
        for other_cls, other_args, kwargs in [
            (other, args, {}),
            (cls, (_Codec(), bits, group, k, tensors), {}),
            (cls, (reader, bits + 1, group, k, tensors), {}),
            (cls, (reader, bits, group * 2, k, tensors), {}),
            (cls, (reader, bits, group, k * 2, tensors), {}),
            (cls, (reader, bits, group, k, dict(tensors)), {}),
            (cls, args, {"bias": mx.ones((16,))}),
            (cls, args, {"placeholders": True}),
            (cls, args, {"placeholders": 0}),
            (cls, args, {"_load_layout": {}}),
        ]:
            _refused(other_cls, other_args, _write_permit=token, **kwargs)
        _refused(cls, args)
        _refused(cls, args, _write_permit=object())
        copied = copy_context()
        with ThreadPoolExecutor(max_workers=1) as worker:
            worker.submit(copied.run, _refused, cls, args, _write_permit=token).result()
        init(layer, *args, _write_permit=token)
        assert layer.fmt is reader
        copied.run(_refused, cls, args, _write_permit=token)
        _refused(cls, args)

    _observe_write(source, tmp_path, monkeypatch, observe)


def test_writer_consumes_the_permit_before_a_plugin_property_can_raise_or_replay(
    source, tmp_path, monkeypatch
):
    class StopError(Exception):
        pass

    active = []
    reads = []
    copied = copy_context()

    class Reader(_Codec):
        @property
        def mode(self):
            if active:
                reads.append(1)
                assert len(reads) == 1
                cls, args, token = active[0]
                copied.run(_refused, cls, args, _write_permit=token)
                raise StopError
            return "toy"

    def observe(init, layer, args, token):
        active.append((type(layer), args, token))
        init(layer, *args, _write_permit=token)

    with pytest.raises(StopError):
        _observe_write(source, tmp_path, monkeypatch, observe, Reader)
    cls, args, token = active[0]
    copied.run(_refused, cls, args, _write_permit=token)
    assert reads == [1]


def test_writer_revokes_an_unused_permit_when_construction_raises(
    source, tmp_path, monkeypatch
):
    class BeforeEntryError(Exception):
        pass

    active = []

    def observe(init, layer, args, token):
        active.append((type(layer), args, token))
        raise BeforeEntryError

    with pytest.raises(BeforeEntryError):
        _observe_write(source, tmp_path, monkeypatch, observe)
    (seen,) = active
    cls, args, token = seen
    _refused(cls, args, _write_permit=token)


@pytest.mark.parametrize(
    "defect,kind",
    [
        ("wrong type", "dot"),
        ("wrong type", "tile"),
        ("wrong type", "codes"),
        ("hand table", "dot"),
        ("hand table", "tile"),
        ("hand table", "codes"),
        ("shared table", "dot"),
    ],
)
def test_writer_rejects_a_bad_face_from_the_persisted_reader(
    source, tmp_path, monkeypatch, kind, defect
):
    assert not formats._TESTING.get()
    assert formats._LOADING.get() is None
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

    slots = {
        "dot": (
            "__OMLX_READ_LOAD__ { return 0; }\n"
            "__OMLX_READ_MADD__ { return 0; }\n"
            "__OMLX_READ_APPLY__ { return 0; }\n"
        ),
        "tile": "__OMLX_READ_WEIGHT__ { return 0; }\n",
        "codes": "__OMLX_READ_PAIR__ { return 0; }\n",
    }[kind]
    reads = {
        "dot": formats.DotReads(apply=formats.ApplyRead()),
        "tile": formats.TileReads(),
        "codes": formats.CodesReads(),
    }[kind]

    def bad_face(*args):
        if defect == "wrong type":
            return True
        if defect == "shared table":
            return formats.Dot(
                key="hand",
                source="struct Face {}\n" + slots,
                shared="constant float LEVELS[2] = {1, 2};",
                type="Face",
                tensors=(),
                vpt=8,
                reads=reads,
            )
        face_type = {"dot": formats.Dot, "tile": formats.Tile, "codes": formats.Codes}[kind]
        return face_type(
            key="hand",
            source="constant float LEVELS[] = {-1.0f, 1.0f};\n" + slots,
            type="Face",
            tensors=(),
            reads=reads,
        )

    class Codec(_Codec):
        def accept(self, record):
            reader = Codec()
            setattr(reader, kind, bad_face)
            return reader

    formats.load()
    writer, prototype = Codec(), Codec()
    monkeypatch.setitem(formats._FORMATS, writer.mode, prototype)
    output = tmp_path / "refused"
    reason = "returned" if defect == "wrong type" else "declares LEVELS"
    with pytest.raises(ValueError, match=rf"{kind}\(\) fails for .*weight.*{reason}"):
        _quantize(source, output, quant_format=writer)
    assert not output.exists()


@pytest.mark.parametrize("native", [False, True], ids=["family", "native"])
def test_writer_refuses_before_issuing_a_probe_permit(tmp_path, monkeypatch, native):
    assert not formats._TESTING.get()
    assert formats._LOADING.get() is None
    src = tmp_path / "checkpoint"
    src.mkdir()
    (src / "config.json").write_text('{"model_type":"qwen3_5"}')
    if native:
        # The admitted backbone has the two real Clef sidecars.
        (src / "joint_head.safetensors").touch()
        (src / "joint_head_config.json").write_text("{}")
    families = {"qwen3_5"} if native else set()
    monkeypatch.setattr(oq, "_QUANT_FORMAT_VERIFIED", frozenset(families))
    monkeypatch.setattr(
        format_testing,
        "_VERIFIED_EXTRA",
        contextvars.ContextVar(
            "omlx_verified_extra",
            default=frozenset((family, None) for family in families),
        ),
    )

    def premature(*args, **kwargs):
        pytest.fail("issued a construction permit before oQ admitted the checkpoint")

    class Unadmitted:
        __setattr__ = premature

    monkeypatch.setattr(linear, "_PENDING_WRITE", Unadmitted(), raising=False)
    monkeypatch.setattr(linear._FormatLayer, "__init__", premature)
    formats.load()
    writer = _Codec()
    monkeypatch.setitem(formats._FORMATS, writer.mode, _Codec())
    output = tmp_path / "refused"
    reason = "native reader" if native else "have the family verified"
    with pytest.raises(formats.FormatRefusedError, match=reason):
        oq.quantize_oq_streaming(
            str(src), str(output), 4, enhanced=True, quant_format=writer
        )
    assert not output.exists() and not writer.encoded
