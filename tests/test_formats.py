# SPDX-License-Identifier: Apache-2.0
"""The quant format registry: built-in and plugin formats register under
their mode through one door, and a plugin that fails to load is skipped.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx import formats

_FACE = formats.Dot(
    key="toy_b4",
    source="namespace omlx_toy_b4 { struct Dot {}; }\n",
    type="omlx_toy_b4::Dot",
    tensors=("scales",),
    tail=False,
)


class _Format(formats.QuantFormat):
    def __init__(self, mode="toy", tensors=("scales",)):
        self.mode, self.tensors = mode, tensors

    def dot(self, layer, k, n):
        return _FACE


_BUILTIN = _Format("builtin")


class _Entry:
    """An installed distribution's entry point."""

    def __init__(self, name, loads):
        self.name, self.value = name, f"{name}:FORMAT"
        self.loads, self.loaded = loads, 0

    def load(self):
        self.loaded += 1
        if isinstance(self.loads, Exception):
            raise self.loads
        return self.loads


@pytest.fixture
def plugins(monkeypatch):
    """An empty, unloaded registry; the returned list is the installed
    plugins' entry points."""
    installed = []
    monkeypatch.setattr(formats, "_FORMATS", {})
    monkeypatch.setattr(formats, "_loaded", False)
    monkeypatch.setattr(formats, "BUILTIN", ())
    monkeypatch.setattr(
        formats,
        "entry_points",
        lambda group: installed if group == formats.ENTRY_POINT_GROUP else [],
    )
    return installed


def test_registered_format_is_found_by_its_mode(plugins):
    fmt = _Format()
    formats.register(fmt)
    assert formats.get("toy") is fmt
    assert formats.get("other") is None
    assert formats.get(None) is None


def test_a_mode_has_one_format(plugins):
    fmt = _Format()
    formats.register(fmt)
    formats.register(fmt)  # the same format again is not a second one
    with pytest.raises(ValueError, match="'toy' is already registered"):
        formats.register(_Format())
    assert formats.get("toy") is fmt


@pytest.mark.parametrize(
    "mode, tensors",
    [
        ("", ("scales",)),
        (None, ("scales",)),
        ("toy", ["scales"]),
        ("toy", ("weight", "scales")),
        ("toy", ("scales", "bias")),
    ],
)
def test_register_refuses_a_malformed_format(plugins, mode, tensors):
    with pytest.raises(ValueError):
        formats.register(_Format(mode, tensors))
    assert formats.get(mode) is None


def test_builtins_register_through_register(plugins, monkeypatch):
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_BUILTIN",))
    registered = []
    register = formats.register
    monkeypatch.setattr(
        formats, "register", lambda fmt: (registered.append(fmt), register(fmt))[1]
    )
    assert formats.get("builtin") is _BUILTIN
    assert registered == [_BUILTIN]


def test_plugins_load_once_from_their_entry_points(plugins):
    fmt = _Format()
    plugins.append(_Entry("toy", fmt))
    assert formats.get("toy") is fmt
    assert formats.get("toy") is fmt
    assert plugins[0].loaded == 1


def test_a_plugin_that_fails_to_load_is_logged_and_skipped(plugins, caplog):
    good = _Format("good")
    plugins += [
        _Entry("missing", ImportError("no module named missing")),
        _Entry("not_a_format", object()),
        _Entry("good", good),
    ]
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.get("good") is good
    assert [r.getMessage() for r in caplog.records] == [
        "quant format plugin missing (missing:FORMAT) failed to load",
        "quant format plugin not_a_format (not_a_format:FORMAT) failed to load",
    ]


def test_a_plugin_cannot_take_a_builtins_mode(plugins, monkeypatch, caplog):
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_BUILTIN",))
    plugins.append(_Entry("shadow", _Format("builtin")))
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.get("builtin") is _BUILTIN
    assert "shadow" in caplog.text


def test_dot_asks_the_format_of_the_layers_mode(plugins):
    formats.register(_Format())
    assert formats.dot(SimpleNamespace(mode="toy"), 64, 8) is _FACE
    assert formats.dot(SimpleNamespace(mode="other"), 64, 8) is None
    assert formats.dot(SimpleNamespace(), 64, 8) is None  # not quantized


def test_a_format_has_no_face_until_it_gives_one():
    assert formats.QuantFormat().dot(SimpleNamespace(), 64, 8) is None


def test_equal_faces_are_one_cache_key():
    # Kernel builders are cached on the faces they are instantiated with.
    same = replace(_FACE)
    assert same is not _FACE
    assert same == _FACE and hash(same) == hash(_FACE)
    assert len({same, _FACE}) == 1


@pytest.mark.parametrize(
    "names",
    [
        {"key": "toy-b4"},
        {"key": ""},
        {"key": "toy_b\u00e9"},
        {"tensors": ("scales", "2nd")},
        {"tensors": ("row scale",)},
        {"tensors": (b"scales",)},
    ],
)
def test_a_face_names_metal_identifiers(names):
    # The key and the tensors are spliced into the kernel's Metal.
    with pytest.raises(ValueError, match="is not an identifier"):
        replace(_FACE, **names)


def test_a_face_is_exact_for_its_dtype_or_for_any():
    assert _FACE.exact_for(mx.float16) and _FACE.exact_for(mx.bfloat16)
    half = replace(_FACE, dtype=mx.float16)
    assert half.exact_for(mx.float16) and not half.exact_for(mx.bfloat16)
