# SPDX-License-Identifier: Apache-2.0
"""The probe's permit and tensors stay private to an admitted oQ write."""

import contextvars
from contextvars import copy_context
from copy import copy, deepcopy

import mlx.core as mx
import pytest

from omlx import formats, oq
from omlx.formats import linear
from omlx.formats import testing as format_testing
from omlx.patches import format_load
from tests.test_format_writer_permit_guards import _observe_write, _refused
from tests.test_formats_oq import _Codec, _in_format, _quantize
from tests.test_formats_oq import source as source


@pytest.mark.parametrize("duplicate", [copy, deepcopy])
@pytest.mark.parametrize("abort", [False, True], ids=["success", "error"])
def test_copies_of_a_write_permit_expire_together(
    source, tmp_path, monkeypatch, duplicate, abort
):
    captured = []

    class AbortError(Exception):
        pass

    def observe(init, layer, args, token):
        retained = duplicate(token)
        captured.append((type(layer), args, retained, token))
        if abort:
            raise AbortError
        init(layer, *args, _write_permit=token)
        assert layer.fmt is args[0]

    if abort:
        with pytest.raises(AbortError):
            _observe_write(source, tmp_path, monkeypatch, observe)
    else:
        _observe_write(source, tmp_path, monkeypatch, observe)
    (seen,) = captured
    cls, args, retained, original = seen
    copied = copy_context()
    for token in (retained, original):
        copied.run(_refused, cls, args, _write_permit=token)


def test_face_tensor_writes_preserve_dense_and_switch_checkpoint_bytes(
    monkeypatch, source, tmp_path
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
            default=frozenset((("qwen3_5_moe_text", None), ("qwen3_5_moe", None))),
        ),
    )
    monkeypatch.setattr(
        format_load,
        "PLUGIN_ALLOWLIST",
        format_load.PLUGIN_ALLOWLIST
        | {("qwen3_5_moe_text", "toy"), ("qwen3_5_moe", "toy")},
    )
    changed = set()
    mutate = False

    class Codec(_Codec):
        def accept(self, record):
            return Codec()

        def dot(self, layer, k, n):
            if mutate:
                changed.add((type(layer), "codes"))
                layer.codes[...] = 0
                mx.eval(layer.codes)

        def tile(self, layer, k, n):
            if mutate:
                changed.add((type(layer), "group_scales"))
                # A view may detach on indexed assignment while still exporting
                # the writer's buffer. Change an actual byte before rebinding.
                data = memoryview(layer.group_scales).cast("B")
                data[0] ^= 0xFF
                layer.group_scales[...] = -23.25
                mx.eval(layer.group_scales)

    formats.load()
    writer = Codec()
    monkeypatch.setitem(formats._FORMATS, writer.mode, Codec())
    reference, config, _ = _quantize(
        source, tmp_path / "reference", quant_format=writer
    )
    mutate = True
    saved, checked_config, _ = _quantize(
        source, tmp_path / "probed", quant_format=writer
    )
    entries = _in_format(config, writer.mode)
    assert entries == _in_format(checked_config, writer.mode)
    assert changed == {
        (cls, tensor)
        for cls in (linear.FormatLinear, linear.FormatSwitchLinear)
        for tensor in ("codes", "group_scales")
    }
    different = []
    for base in entries:
        for name in ("codes", "group_scales"):
            key = f"{base}.{name}"
            assert saved[key].shape == reference[key].shape
            assert saved[key].dtype == reference[key].dtype
            if memoryview(saved[key]).tobytes() != memoryview(reference[key]).tobytes():
                different.append(key)
    assert not different, different


def test_probe_reads_joined_stack_after_the_writer_releases_its_source(
    source, tmp_path, monkeypatch
):
    from weakref import ref

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
            default=frozenset((("qwen3_5_moe_text", None), ("qwen3_5_moe", None))),
        ),
    )
    monkeypatch.setattr(
        format_load,
        "PLUGIN_ALLOWLIST",
        format_load.PLUGIN_ALLOWLIST
        | {("qwen3_5_moe_text", "toy"), ("qwen3_5_moe", "toy")},
    )
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 1 << 16)
    joined_source = None
    observations = []

    class Codec(_Codec):
        def accept(self, record):
            return Codec()

        def dot(self, layer, k, n):
            if joined_source is not None:
                observations.append(joined_source() is None)
                mx.eval(layer.codes, layer.group_scales)

    formats.load()
    writer = Codec()
    monkeypatch.setitem(formats._FORMATS, writer.mode, Codec())
    encode = oq._encode_in_format

    def observe(fmt, tensor_name, w, *args, **kwargs):
        nonlocal joined_source
        before = len(writer.encoded)
        tensors = encode(fmt, tensor_name, w, *args, **kwargs)
        joined = tensors is not None and len(writer.encoded) - before > 1
        joined_source = ref(w) if joined else None
        return tensors

    monkeypatch.setattr(oq, "_encode_in_format", observe)
    _quantize(source, tmp_path / "output", quant_format=writer)
    assert observations and all(observations)
