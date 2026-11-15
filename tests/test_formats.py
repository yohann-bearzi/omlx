# SPDX-License-Identifier: Apache-2.0
"""The quant format registry: built-in and plugin formats register under
their mode through one door, and a plugin that fails to load is skipped.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import testing

_PAGE = (Path(__file__).parents[1] / "docs" / "quant-formats.md").read_text()

_FACE = formats.Dot(
    key="toy_b4",
    source="namespace omlx_toy_b4 { struct Dot {}; }\n",
    type="omlx_toy_b4::Dot",
    tensors=("scales",),
)


class _Format(formats.QuantFormat):
    def __init__(self, mode="toy", tensors=("scales",), contract=1):
        self.mode, self.tensors, self.contract = mode, tensors, contract

    def dot(self, layer, k, n):
        self.asked = (layer, k, n)
        return _FACE


_BUILTIN = _Format("builtin")
_HOLDER = SimpleNamespace(fmt=_BUILTIN)


class _Entry:
    """An installed distribution's entry point."""

    def __init__(self, name, loads, value=None):
        self.name, self.value = name, value or f"{name}:FORMAT"
        self.loads, self.loaded = loads, 0

    def load(self):
        self.loaded += 1
        if isinstance(self.loads, Exception):
            raise self.loads
        return self.loads() if callable(self.loads) else self.loads


@pytest.fixture
def plugins(monkeypatch):
    """An empty, unloaded registry; the returned list is the installed
    plugins' entry points."""
    installed = []
    monkeypatch.setattr(formats, "_FORMATS", {})
    monkeypatch.setattr(formats, "_loaded", False)
    monkeypatch.setattr(formats, "_loading", False)
    monkeypatch.setattr(formats, "BUILTIN", ())
    monkeypatch.setattr(
        formats,
        "entry_points",
        lambda group: installed if group == formats.ENTRY_POINT_GROUP else [],
    )
    return installed


def test_the_load_may_register_under_its_own_lock():
    # register() runs inside the load: under a lock that is not reentrant
    # the first lookup would hang.
    assert formats._LOCK.acquire(timeout=5), "another thread holds the lock"
    try:
        assert formats._LOCK.acquire(blocking=False), "the lock is not reentrant"
        formats._LOCK.release()
    finally:
        formats._LOCK.release()


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
        (7, ("scales",)),
        ("toy", ["scales"]),
        ("toy", ("weight", "scales")),
        ("toy", ("scales", "bias")),
    ],
)
def test_register_refuses_a_malformed_format(plugins, mode, tensors):
    with pytest.raises(ValueError):
        formats.register(_Format(mode, tensors))
    assert formats.get(mode) is None


def test_register_refuses_what_is_not_a_format_instance(plugins):
    with pytest.raises(TypeError, match="is not a QuantFormat instance"):
        formats.register(_Format)  # the class
    with pytest.raises(ValueError, match="has no mode"):
        formats.register(formats.QuantFormat())

    class ModeOnly(formats.QuantFormat):
        mode = "toy"

    with pytest.raises(ValueError, match="'toy': tensors must be a tuple"):
        formats.register(ModeOnly())

    class NoRevision(ModeOnly):
        tensors = ("scales",)

    with pytest.raises(ValueError, match="'toy': contract is the revision"):
        formats.register(NoRevision())
    assert formats.get("toy") is None


def test_a_malformed_format_is_refused_before_anything_loads(plugins):
    plugins.append(_Entry("toy", _Format()))
    with pytest.raises(ValueError, match="has no mode"):
        formats.register(formats.QuantFormat())
    with pytest.raises(ValueError, match="tensors must be a tuple"):
        formats.register(_Format(tensors=["scales"]))
    with pytest.raises(ValueError, match="contract is the revision"):
        formats.register(_Format(contract=None))
    with pytest.raises(ValueError, match="update oMLX"):
        formats.register(_Format(contract=formats.CONTRACT + 1))
    assert plugins[0].loaded == 0


class _Revision(int):
    """An int of another type than int."""


# The revision is an int, and nothing isinstance takes for one.
@pytest.mark.parametrize("contract", [None, "1", 1.0, True, _Revision(1)])
def test_register_refuses_a_revision_that_is_not_an_int(plugins, contract):
    with pytest.raises(ValueError) as refusal:
        formats.register(_Format(contract=contract))
    assert str(refusal.value) == (
        "quant format 'toy': contract is the revision of the format contract "
        f"it is written for, an int, not {contract!r} ({type(contract).__name__})"
    )
    assert formats.get("toy") is None


def test_a_format_registers_under_the_revisions_omlx_runs(plugins, monkeypatch):
    monkeypatch.setattr(formats, "CONTRACT", 4)
    monkeypatch.setattr(formats, "OLDEST_CONTRACT", 2)
    for contract in (2, 3, 4):
        fmt = _Format(f"toy{contract}", contract=contract)
        formats.register(fmt)
        assert formats.get(f"toy{contract}") is fmt
    for contract, update in ((5, "oMLX"), (1, "the format"), (0, "the format")):
        with pytest.raises(ValueError) as refusal:
            formats.register(_Format("other", contract=contract))
        assert str(refusal.value) == (
            f"quant format 'other' is written for revision {contract} of the "
            f"format contract; this oMLX runs revisions 2 to 4: update {update}"
        )
    monkeypatch.setattr(formats, "OLDEST_CONTRACT", 4)
    with pytest.raises(ValueError, match="oMLX runs revision 4: update the format"):
        formats.register(_Format("other", contract=3))
    assert formats.get("other") is None


def test_a_plugin_of_another_revision_is_logged_and_skipped(plugins, caplog):
    good = _Format("good")
    plugins += [
        _Entry("later", _Format("later", contract=formats.CONTRACT + 1)),
        _Entry("good", good),
    ]
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.get("good") is good
    assert formats.get("later") is None
    (record,) = caplog.records
    assert record.getMessage() == (
        "quant format plugin later (later:FORMAT) failed to load"
    )
    assert "update oMLX" in str(record.exc_info[1])


def test_the_contract_page_lists_every_revision():
    versions = _PAGE.split("\n## Versions\n")[1].split("\n## ")[0]
    revisions = [int(n) for n in re.findall(r"^\| (\d+) \|", versions, re.M)]
    assert revisions == list(range(1, formats.CONTRACT + 1))
    assert 1 <= formats.OLDEST_CONTRACT <= formats.CONTRACT


def test_the_contract_page_quotes_the_refusal_of_a_later_revision(plugins):
    with pytest.raises(ValueError) as refusal:
        formats.register(_Format("myformat", contract=formats.CONTRACT + 1))
    assert str(refusal.value) in " ".join(_PAGE.split())


def test_plugins_are_found_in_the_group_the_contract_page_names():
    assert formats.ENTRY_POINT_GROUP == "omlx.quant_formats"
    assert '\n[project.entry-points."omlx.quant_formats"]\n' in _PAGE


def test_builtins_register_through_register(plugins, monkeypatch):
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_BUILTIN",))
    registered = []
    register = formats.register

    def recorded(fmt):
        registered.append(fmt)
        register(fmt)

    monkeypatch.setattr(formats, "register", recorded)
    assert formats.get("builtin") is _BUILTIN
    assert registered == [_BUILTIN]


def test_a_builtin_is_named_like_an_entry_point(plugins, monkeypatch):
    # "module:attribute", where the attribute may be dotted.
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_HOLDER.fmt",))
    assert formats.get("builtin") is _BUILTIN


def test_a_builtin_has_its_mode_before_anything_registers(plugins, monkeypatch):
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_BUILTIN",))
    # The first call into the registry is a register, not a lookup.
    with pytest.raises(ValueError, match="'builtin' is already registered"):
        formats.register(_Format("builtin"))
    assert formats.get("builtin") is _BUILTIN


def test_a_builtin_that_fails_to_load_raises_at_every_lookup(plugins, monkeypatch):
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_NO_SUCH_FORMAT",))
    for _ in range(2):
        with pytest.raises(AttributeError, match="_NO_SUCH_FORMAT"):
            formats.get("builtin")
        with pytest.raises(AttributeError, match="_NO_SUCH_FORMAT"):
            formats.load()


def test_load_registers_every_format_ahead_of_a_lookup(plugins, monkeypatch):
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_BUILTIN",))
    fmt = _Format()
    plugins.append(_Entry("toy", fmt))
    formats.load()
    assert dict(formats._FORMATS) == {"builtin": _BUILTIN, "toy": fmt}
    formats.load()
    assert plugins[0].loaded == 1


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
    # Each with the error that failed it.
    assert [type(r.exc_info[1]) for r in caplog.records] == [ImportError, TypeError]


def test_an_interrupt_inside_a_plugins_load_is_not_a_failed_plugin(plugins, caplog):
    def interrupted():
        raise KeyboardInterrupt

    plugins.append(_Entry("toy", interrupted))
    caplog.set_level(logging.ERROR, logger=formats.__name__)
    with pytest.raises(KeyboardInterrupt):
        formats.get("toy")
    assert not caplog.records
    # The load did not end: the next lookup loads again.
    fmt = _Format()
    plugins[0].loads = fmt
    assert formats.get("toy") is fmt
    assert plugins[0].loaded == 2


def test_a_plugin_that_fails_takes_what_it_registered_with_it(plugins, caplog):
    good = _Format("good")

    def load():
        formats.register(_Format("early"))  # its module, by itself
        raise ImportError("and then the import failed")

    plugins += [_Entry("good", good), _Entry("toy", load)]
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.get("early") is None
    assert formats.get("good") is good
    assert [r.getMessage() for r in caplog.records] == [
        "quant format plugin toy (toy:FORMAT) failed to load"
    ]


def test_a_plugin_cannot_take_a_builtins_mode(plugins, monkeypatch, caplog):
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_BUILTIN",))
    plugins.append(_Entry("shadow", _Format("builtin")))
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.get("builtin") is _BUILTIN
    assert [r.getMessage() for r in caplog.records] == [
        "quant format plugin shadow (shadow:FORMAT) failed to load"
    ]


def test_of_two_plugins_for_a_mode_the_first_by_name_is_kept(plugins, caplog):
    first, second = _Format(), _Format()
    # The names decide, whatever the values.
    plugins += [
        _Entry("zeta", second, "package_a:FORMAT"),
        _Entry("alpha", first, "package_z:FORMAT"),
    ]
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.get("toy") is first
    assert [r.getMessage() for r in caplog.records] == [
        "quant format plugin zeta (package_a:FORMAT) failed to load"
    ]


def test_of_two_plugins_of_one_name_the_first_by_value_is_kept(plugins, caplog):
    first, second = _Format(), _Format()
    plugins += [
        _Entry("toy", second, "package_b:FORMAT"),
        _Entry("toy", first, "package_a:FORMAT"),
    ]
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.get("toy") is first
    assert [r.getMessage() for r in caplog.records] == [
        "quant format plugin toy (package_b:FORMAT) failed to load"
    ]


def test_a_lookup_inside_the_load_sees_the_formats_registered_so_far(
    plugins, monkeypatch
):
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_BUILTIN",))
    fmt, seen = _Format(), []

    def load():
        # Not for a plugin to rely on; it neither hangs nor loads twice.
        seen.append((formats.get("builtin"), formats.get("toy")))
        formats.register(fmt)
        return fmt

    plugins.append(_Entry("toy", load))
    assert formats.get("toy") is fmt
    assert seen == [(_BUILTIN, None)]
    assert plugins[0].loaded == 1


def test_a_lookup_from_another_thread_waits_for_the_load(plugins):
    fmt, found = _Format(), []
    loading, release = threading.Event(), threading.Event()

    def load():
        loading.set()
        assert release.wait(10)
        return fmt

    def lookup():
        found.append(formats.get("toy"))

    plugins.append(_Entry("toy", load))
    first = threading.Thread(target=lookup, daemon=True)
    second = threading.Thread(target=lookup, daemon=True)
    first.start()
    try:
        assert loading.wait(10)
        second.start()
        second.join(0.2)
        assert second.is_alive(), "the lookup did not wait for the load"
    finally:
        # A loader left waiting would fail later, in another test's registry.
        release.set()
        for thread in (first, second):
            if thread.ident is not None:
                thread.join(10)
    assert found == [fmt, fmt]
    assert plugins[0].loaded == 1


def _layer(mode="toy", tensors=("weight", "scales")):
    """A quantized linear: its mode, and its parameters by name."""
    layer = nn.Module()
    layer.mode = mode
    for name in tensors:
        layer[name] = mx.zeros((8,))
    return layer


def test_dot_asks_the_format_of_the_layers_mode(plugins, caplog):
    fmt, layer = _Format(), _layer()
    formats.register(fmt)
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.dot(layer, 64, 8) is _FACE
        assert fmt.asked == (layer, 64, 8)
        assert formats.dot(_layer("other"), 64, 8) is None
        assert formats.dot(nn.Module(), 64, 8) is None  # not quantized
    assert not caplog.records


class _NoFace(_Format):
    def dot(self, layer, k, n):
        return None


class _DefectError(Exception):
    """No exception oMLX or Python raises."""


class _Raises(_Format):
    def dot(self, layer, k, n):
        raise _DefectError("no biases")


class _NotADot(_Format):
    class Dot:
        """Not ``omlx.formats.Dot``."""

    def dot(self, layer, k, n):
        return self.Dot()


class _TwoTensors(_Format):
    def dot(self, layer, k, n):
        return replace(_FACE, tensors=("scales", "biases"))


def _layer_with_a_child_named_scales():
    layer = _layer(tensors=("weight",))
    layer.scales = nn.Module()
    return layer


@pytest.mark.parametrize(
    "fmt, layer, defect",
    [
        (_Raises(), _layer(), "_DefectError('no biases')"),
        (
            _NotADot(),
            _layer(),
            f'TypeError("dot() returned a {_NotADot.Dot!r}, not a Dot")',
        ),
        # _FACE reads scales.
        (
            _Format(),
            _layer(tensors=("weight",)),
            "ValueError('face toy_b4: the layer has no array scales')",
        ),
        (
            _Format(),
            _layer(tensors=("scales",)),
            "ValueError('face toy_b4: the layer has no array weight')",
        ),
        (
            _Format(),
            _layer(tensors=()),
            "ValueError('face toy_b4: the layer has no array weight, scales')",
        ),
        (
            _Format(),
            _layer_with_a_child_named_scales(),
            "ValueError('face toy_b4: the layer has no array scales')",
        ),
        (
            _TwoTensors(),
            _layer(),
            "ValueError('face toy_b4: the layer has no array biases')",
        ),
    ],
)
def test_a_defective_dot_is_no_face(plugins, caplog, dot_defects, fmt, layer, defect):
    formats.register(fmt)
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.dot(layer, 64, 8) is None
    (record,) = caplog.records
    assert record.getMessage() == (
        "quant format toy gave no usable dot face for a Module (n=8, k=64); "
        "the layer keeps its composed path"
    )
    assert repr(record.exc_info[1]) == defect
    assert dot_defects == {("toy", defect)}
    dot_defects.clear()  # expected


def test_each_defect_of_a_format_is_logged_once(plugins, caplog, dot_defects):
    for fmt in (_Format(), _Raises("other"), _Raises("third")):
        formats.register(fmt)
    no_scales, no_weight = _layer(tensors=("weight",)), _layer(tensors=("scales",))
    asked = (
        no_scales,
        _layer(tensors=("weight",)),
        no_weight,
        no_scales,
        _layer("other"),
        _layer("third"),
        _layer("other"),
    )
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        for layer in asked:
            assert formats.dot(layer, 64, 8) is None
        # Whatever the matrix asked for.
        assert formats.dot(no_scales, 128, 16) is None
        # Only a defect's layers lose the face.
        assert formats.dot(_layer(), 64, 8) is _FACE
    logged = [(r.args[0], repr(r.exc_info[1])) for r in caplog.records]
    assert logged == [
        ("toy", "ValueError('face toy_b4: the layer has no array scales')"),
        ("toy", "ValueError('face toy_b4: the layer has no array weight')"),
        ("other", "_DefectError('no biases')"),
        ("third", "_DefectError('no biases')"),
    ]
    assert dot_defects == set(logged)
    dot_defects.clear()  # expected


def test_a_defect_fails_a_block_that_admits_none(plugins, dot_defects):
    formats.register(_Raises())
    defect = ("toy", "_DefectError('no biases')")
    with pytest.raises(AssertionError) as failure, testing.no_dot_defects():
        assert formats.dot(_layer(), 64, 8) is None
    assert str(failure.value) == f"a quant format's dot() is defective: {[defect]}"
    # The defect is the block's: nothing outside collected it.
    assert formats._DOT_DEFECTS is dot_defects and not dot_defects
    with testing.no_dot_defects() as defects:
        assert formats.dot(_layer("other"), 64, 8) is None
        assert formats.dot(_layer(), 64, 8) is None
        assert defects == {defect}
        defects.clear()  # expected
    with pytest.raises(KeyError), testing.no_dot_defects():
        formats.dot(_layer(), 64, 8)
        raise KeyError  # the body's failure is the one reported
    assert formats._DOT_DEFECTS is dot_defects
    # Nor is a defect met outside a block that block's.
    assert formats.dot(_layer(), 64, 8) is None
    with testing.no_dot_defects():
        pass
    assert dot_defects == {defect}
    dot_defects.clear()  # expected


def test_each_test_runs_in_a_block_that_admits_no_dot_defect(request, monkeypatch):
    assert "dot_defects" in request.fixturenames  # though the test does not ask
    defects = request.getfixturevalue("dot_defects")
    assert formats._DOT_DEFECTS is defects
    # The block is not a patch that a test's monkeypatch undoes.
    monkeypatch.undo()
    assert formats._DOT_DEFECTS is defects


_BLOCKS: list = []


@pytest.mark.parametrize("_", range(2))
def test_each_test_has_a_block_of_its_own(dot_defects, _):
    assert all(dot_defects is not seen for seen in _BLOCKS)
    _BLOCKS.append(dot_defects)


def test_an_interrupt_inside_dot_is_not_a_defect(plugins):
    class Interrupted(_Format):
        def dot(self, layer, k, n):
            raise KeyboardInterrupt

    formats.register(Interrupted())
    with pytest.raises(KeyboardInterrupt):
        formats.dot(_layer(), 64, 8)


def test_a_format_without_a_face_for_a_layer_is_no_defect(plugins, caplog):
    formats.register(_NoFace())
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.dot(_layer(), 64, 8) is None
    assert not caplog.records


def test_a_format_has_no_face_until_it_gives_one():
    assert formats.QuantFormat().dot(SimpleNamespace(), 64, 8) is None


def test_equal_faces_are_one_cache_key():
    # Kernel builders are cached on the faces they are instantiated with.
    same = replace(_FACE)
    assert same is not _FACE
    assert same == _FACE and hash(same) == hash(_FACE)
    assert len({same, _FACE}) == 1


def test_faces_compare_by_their_dtypes_too():
    half = replace(_FACE, dtypes=(mx.float16,))
    # A dtype read off an array, not the module's constant.
    again = replace(_FACE, dtypes=(mx.zeros(1, mx.float16).dtype,))
    assert half == again and hash(half) == hash(again)
    assert half != _FACE  # exact for one dtype, exact for any
    assert half != replace(_FACE, dtypes=(mx.bfloat16,))


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


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"tensors": "scales"}, "tensors and dtypes are tuples"),
        ({"tensors": ["scales"]}, "tensors and dtypes are tuples"),
        ({"dtypes": mx.float16}, "tensors and dtypes are tuples"),
        ({"tensors": ("scales", "scales")}, "beside 'weight', each named once"),
        ({"tensors": ("weight",)}, "beside 'weight', each named once"),
        ({"tensors": ("scales", "weight")}, "beside 'weight', each named once"),
        ({"dtypes": (None,)}, "dtypes are distinct mx.Dtype"),
        ({"dtypes": ("float16",)}, "dtypes are distinct mx.Dtype"),
        ({"dtypes": (mx.float16, [])}, "dtypes are distinct mx.Dtype"),
        ({"dtypes": (mx.float16, mx.float16)}, "dtypes are distinct mx.Dtype"),
    ],
)
def test_a_face_lists_its_tensors_and_dtypes_in_tuples(fields, message):
    with pytest.raises(ValueError, match=message):
        replace(_FACE, **fields)


def test_a_face_is_exact_for_its_dtypes_or_for_any():
    assert _FACE.exact_for(mx.float16) and _FACE.exact_for(mx.bfloat16)
    half = replace(_FACE, dtypes=(mx.float16,))
    assert half.exact_for(mx.float16) and not half.exact_for(mx.bfloat16)
    halves = replace(_FACE, dtypes=(mx.float16, mx.bfloat16))
    assert halves.exact_for(mx.bfloat16) and not halves.exact_for(mx.float32)
