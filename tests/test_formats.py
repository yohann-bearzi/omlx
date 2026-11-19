# SPDX-License-Identifier: Apache-2.0
"""The quant format registry: built-in and plugin formats register under
their mode through one door, and a plugin that fails to load is skipped.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import re
import threading
from dataclasses import replace
from functools import cache
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
    monkeypatch.setattr(formats, "_ORIGINS", {})
    monkeypatch.setattr(formats, "_SKIPPED", {}, raising=False)
    monkeypatch.setattr(formats, "_loaded", False)
    monkeypatch.setattr(formats, "_loading", False)
    monkeypatch.setattr(formats, "BUILTIN", ())
    monkeypatch.setattr(formats, "_BUILTIN_MODES", set())
    monkeypatch.setattr(
        formats,
        "entry_points",
        lambda group: installed if group == formats.ENTRY_POINT_GROUP else [],
    )
    return installed


def test_a_dot_s_shared_source_and_vpt_are_named():
    """Dot.shared and Dot.vpt are keywords: a face written for the fields
    before them keeps its meaning, and none is given by position."""
    fields = inspect.signature(formats.Dot).parameters
    assert fields["shared"].kind == fields["vpt"].kind == inspect.Parameter.KEYWORD_ONLY


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
        # What a face would refuse: kernels splice them into Metal.
        ("toy", ("scales", "scales")),
        ("toy", ("row-scales",)),
        ("toy", ("",)),
        ("toy", ("scales.sub",)),
        ("toy", (7,)),
        ("toy", ([],)),
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

    with pytest.raises(ValueError, match="quant format 'toy': tensors are a tuple"):
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
    with pytest.raises(ValueError, match="tensors are a tuple"):
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
    for contract, update in ((6, "oMLX"), (2, "the format")):
        with pytest.raises(ValueError) as refusal:
            formats.register(_Format("other", contract=contract))
        assert str(refusal.value) == (
            f"quant format 'other' is written for revision {contract} of the "
            f"format contract; this oMLX runs revision 4: update {update}"
        )
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
    # What a checkpoint of its mode is told.
    said = f"later:FORMAT is installed but did not load: {record.exc_info[1]}"
    assert dict(formats._SKIPPED) == {"later": said}


def test_a_format_s_record_is_empty_and_it_reads_any_record_unless_it_says():
    fmt = _Format()
    assert fmt.record() == {}
    # By default the format itself reads every checkpoint.
    for record in ({}, {"layout": 2, "levels": [0.5, 1.5]}):
        assert fmt.accept(record) is fmt


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


def test_a_loaded_plugin_is_logged_with_its_distribution(plugins, monkeypatch, caplog):
    def two():
        # The modes are listed by name, not as they registered.
        formats.register(_Format("second"))
        return _Format("also")

    installed = _Entry("toy", _Format())
    installed.dist = SimpleNamespace(name="omlx-toy", version="0.3.1")
    plugins += [_Entry("two", two), installed]
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_BUILTIN",))
    with caplog.at_level(logging.INFO, logger=formats.__name__):
        formats.load()
    # Each plugin with the distribution that installed it, or with its entry
    # point's value. The built-in formats are not plugins.
    assert [r.getMessage() for r in caplog.records] == [
        "quant format plugin toy (omlx-toy 0.3.1) registered toy",
        "quant format plugin two (two:FORMAT) registered also, second",
    ]
    assert sorted(formats._FORMATS) == ["also", "builtin", "second", "toy"]


def test_a_mode_s_origin_is_the_plugin_that_registered_it(plugins, monkeypatch):
    def two():
        formats.register(_Format("also"))
        return _Format("second")

    def failing():
        formats.register(_Format("early"))  # its module, by itself
        raise ImportError("and then the import failed")

    installed = _Entry("toy", _Format())
    installed.dist = SimpleNamespace(name="omlx-toy", version="0.3.1")
    plugins += [_Entry("two", two), installed, _Entry("bad", failing)]
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_BUILTIN",))
    # Asking loads, as every lookup does.
    assert formats.origin("toy") == "omlx-toy 0.3.1"
    assert installed.loaded == 1
    # Each mode a plugin registered has it, as the entry point's value when
    # no distribution installed the plugin.
    assert formats.origin("also") == formats.origin("second") == "two:FORMAT"
    # A built-in format has none. Nor has a format registered by hand, a
    # format of a plugin that failed, or a mode without a format.
    formats.register(_Format("by_hand"))
    assert formats.get("by_hand") and formats.get("builtin") is _BUILTIN
    for mode in ("builtin", "by_hand", "early", "unknown"):
        assert formats.origin(mode) is None


def test_a_plugin_that_fails_to_load_is_logged_and_skipped(plugins, caplog):
    good = _Format("good")
    plugins += [
        _Entry("missing", ImportError("no module named missing")),
        _Entry("not_a_format", object()),
        _Entry("good", good),
    ]
    with caplog.at_level(logging.INFO, logger=formats.__name__):
        assert formats.get("good") is good
    # A plugin that failed is logged as failed, and not as registered.
    assert [r.getMessage() for r in caplog.records] == [
        "quant format plugin good (good:FORMAT) registered good",
        "quant format plugin missing (missing:FORMAT) failed to load",
        "quant format plugin not_a_format (not_a_format:FORMAT) failed to load",
    ]
    # Each with the error that failed it.
    failed = caplog.records[1:]
    assert [type(r.exc_info[1]) for r in failed] == [ImportError, TypeError]


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


def test_retrying_discovery_keeps_only_the_completed_attempt(plugins):
    def incomplete():
        formats.register(_Format("partial"))
        raise ImportError("the plugin did not finish")

    def interrupt():
        raise KeyboardInterrupt

    first = _Entry("first", _Format("first"))
    partial = _Entry("partial", incomplete)
    last = _Entry("stop", interrupt)
    plugins.extend([first, partial, last])
    with pytest.raises(KeyboardInterrupt):
        formats.load()

    first.loads = _Format("first")
    partial.loads = _Format("ready")
    last.loads = _Format("last")
    assert formats.get("first") is first.loads
    assert formats.get("ready") is partial.loads and formats.get("last") is last.loads
    assert formats.get("partial") is None and "partial" not in formats._SKIPPED


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


@pytest.mark.parametrize("failure", [None, ValueError, KeyboardInterrupt])
def test_plugin_discovery_releases_its_registration_log(plugins, monkeypatch, failure):
    local = threading.local()
    monkeypatch.setattr(formats, "_PLUGIN_MODES", local)

    def plugin():
        if failure:
            raise failure
        return _Format("first")

    plugins.append(_Entry("first", plugin))
    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            formats.load()
    else:
        formats.load()
    assert getattr(local, "modes", None) is None


def test_what_a_checkpoint_of_a_plugin_that_did_not_load_is_told(plugins, caplog):
    def partly():
        # A plugin of two modes whose import fails after its first.
        formats.register(_Format("also"))
        raise ImportError("no module named its_codec")

    first = _Entry("alpha", _Format("oqel", contract=formats.CONTRACT + 1))
    first.dist = SimpleNamespace(name="omlx-alpha", version="2.0.0")
    plugins += [
        first,
        # Another plugin of the mode that fails after it is not what is said.
        _Entry("zeta", _Format("oqel", contract=0)),
        _Entry("partly", partly),
        # One that fails for a mode a later plugin registers.
        _Entry("beta", _Format("good", contract=formats.CONTRACT + 1)),
        _Entry("gamma", _Format("good")),
        # One that fails because an earlier plugin has its mode.
        _Entry("omega", _Format("good")),
    ]
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        formats.load()
    assert set(formats._SKIPPED) == {"oqel", "also"}
    assert formats._SKIPPED["oqel"].startswith(
        "omlx-alpha 2.0.0 is installed but did not load: quant format 'oqel' is "
        "written for revision"
    )
    assert formats._SKIPPED["also"] == (
        "partly:FORMAT is installed but did not load: no module named its_codec"
    )
    assert formats.get("good") is not None and formats.get("also") is None


def test_a_format_registered_by_hand_meanwhile_is_no_plugin_s(plugins):
    import threading
    import time

    def failing():
        # Another thread registers a format by hand while this plugin loads:
        # it waits for the load, and is not this plugin's.
        threading.Thread(target=formats.register, args=(_Format("byhand"),)).start()
        time.sleep(0.2)
        raise ImportError("no module named its_codec")

    plugins.append(_Entry("failing", failing))
    formats.load()
    deadline = time.monotonic() + 60
    while formats._FORMATS.get("byhand") is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert formats.get("byhand") is not None and "byhand" not in formats._SKIPPED


def test_only_the_thread_that_loads_the_builtins_takes_an_mlx_mode(plugins):
    import threading

    formats._REGISTERING_BUILTINS.registering = True
    refused = []

    def by_hand():
        try:
            formats.register(_Format("mxfp4"))
        except ValueError as error:
            refused.append(str(error))

    try:
        worker = threading.Thread(target=by_hand)
        worker.start()
        worker.join(60)
    finally:
        formats._REGISTERING_BUILTINS.registering = False
    assert refused and "is one of MLX's own modes" in refused[0]


def test_what_is_one_of_mlx_s_modes():
    for mode in ("affine", "mxfp4", "mxfp8", "nvfp4"):
        assert formats._mlx_runs(mode)
    for mode in ("oqel", "", None, 4, ["bad"], {"mode": "bad"}):
        assert not formats._mlx_runs(mode)


def test_a_mode_is_probed_on_the_cpu_only_once(monkeypatch, tmp_path):
    mode = f"test-format-{tmp_path}"
    calls = []
    quantize, zeros = mx.quantize, mx.zeros

    def allocate(*args, **kwargs):
        calls.append(("zeros", kwargs["stream"]))
        return zeros(*args, **kwargs)

    def counted(*args, **kwargs):
        calls.append((kwargs["mode"], kwargs["stream"]))
        return quantize(*args, **kwargs)

    monkeypatch.setattr(mx, "zeros", allocate)
    monkeypatch.setattr(mx, "quantize", counted)
    for _ in range(2):
        assert not formats._mlx_runs(mode)
    assert calls == [("zeros", mx.cpu), (mode, mx.cpu)]


@pytest.mark.parametrize("mode", ["affine", "mxfp4", "mxfp8", "nvfp4"])
def test_mlx_s_own_modes_are_the_builtin_formats(plugins, monkeypatch, caplog, mode):
    refused = (
        f"quant format {mode!r}: {mode!r} is one of MLX's own modes, which only "
        "oMLX's built-in formats take: register the format under a mode of its own"
    )
    with pytest.raises(ValueError) as by_hand:
        formats.register(_Format(mode))
    assert str(by_hand.value) == refused
    # A plugin's is logged and skipped; checkpoints of the mode stay MLX's.
    plugins.append(_Entry("shadow", _Format(mode)))
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.get(mode) is None
    (record,) = caplog.records
    assert str(record.exc_info[1]) == refused
    said = f"shadow:FORMAT is installed but did not load: {refused}"
    assert dict(formats._SKIPPED) == {mode: said}
    # A built-in takes one.
    monkeypatch.setattr(formats, "_loaded", False)
    builtin = _Format(mode)
    monkeypatch.setattr(formats, "BUILTIN", (f"{__name__}:_BUILTINS.{mode}",))
    monkeypatch.setattr(_BUILTINS, mode, builtin, raising=False)
    monkeypatch.setattr(formats, "_FORMATS", {})
    plugins.clear()
    assert formats.get(mode) is builtin
    assert formats._mlx_runs(mode) and not formats._mlx_runs("toy")


_BUILTINS = SimpleNamespace()


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
    """A quantized linear: its mode, and its parameters by name, each of the
    fewest elements that MLX binds as a buffer."""
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


@pytest.mark.parametrize("face", [formats.dot, formats.tile], ids=["dot", "tile"])
def test_a_foreign_modules_fmt_attribute_is_not_a_quant_format(plugins, face):
    layer = nn.Module()
    layer.fmt = "channels_last"
    assert face(layer, 64, 8) is None


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


class _ThreeTensors(_Format):
    def dot(self, layer, k, n):
        return replace(_FACE, tensors=("scales", "biases", "gains"))


def _layer_with_small(*names):
    layer = _layer()
    for name in names:
        layer[name] = mx.zeros((7,))
    return layer


def _layer_of_float64(*names):
    layer = _layer()
    for name in names:
        layer[name] = mx.zeros((8,), mx.float64, stream=mx.cpu)
    return layer


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
        # MLX binds an input of fewer than 8 elements as constants.
        (
            _Format(),
            _layer_with_small("scales"),
            "ValueError('face toy_b4: scales of fewer than 8 elements, which MLX "
            "binds as constants, not as the buffer a face reads')",
        ),
        (
            _Format(),
            _layer_with_small("weight"),
            "ValueError('face toy_b4: weight of fewer than 8 elements, which MLX "
            "binds as constants, not as the buffer a face reads')",
        ),
        (
            _Format(),
            _layer_with_small("scales", "weight"),
            "ValueError('face toy_b4: weight, scales of fewer than 8 elements, which "
            "MLX binds as constants, not as the buffer a face reads')",
        ),
        # A face reads its format's tensors, which a load gives the layer.
        (
            _TwoTensors(),
            _layer(),
            "ValueError('face toy_b4: biases is no tensor of quant format toy')",
        ),
        (
            _TwoTensors(),
            _layer(tensors=("weight", "scales", "biases")),
            "ValueError('face toy_b4: biases is no tensor of quant format toy')",
        ),
        (
            _ThreeTensors(),
            _layer(),
            "ValueError('face toy_b4: biases, gains is no tensor of quant format toy')",
        ),
        # The GPU takes no float64: a stand-in of such a tensor cannot be made.
        (
            _Format(),
            _layer_of_float64("weight", "scales"),
            "ValueError('face toy_b4: weight, scales of float64, which the GPU does "
            "not take')",
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
    assert str(failure.value) == (
        f"a quant format's dot() or tile() is defective: {[defect]}"
    )
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


@cache
def _tile():
    return formats.Tile(
        key="toy_tile_b4",
        source="namespace omlx_toy_tile_b4 { struct Tile {}; }\n",
        type="omlx_toy_tile_b4::Tile",
        tensors=("scales",),
    )


class _Tiled(_Format):
    def tile(self, layer, k, n):
        self.tiled = (layer, k, n)
        return _tile()


def test_tile_asks_the_format_of_the_layers_mode(plugins, caplog):
    fmt, layer = _Tiled(), _layer()
    formats.register(fmt)
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.tile(layer, 64, 8) is _tile()
        assert fmt.tiled == (layer, 64, 8)
        # Its dot face is asked for apart.
        assert not hasattr(fmt, "asked") and formats.dot(layer, 64, 8) is _FACE
        assert formats.tile(_layer("other"), 64, 8) is None
        assert formats.tile(nn.Module(), 64, 8) is None  # not quantized
    assert not caplog.records


def test_a_format_without_a_tile_face_keeps_its_dot_face(plugins, caplog):
    formats.register(_Format())
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.tile(_layer(), 64, 8) is None
        assert formats.dot(_layer(), 64, 8) is _FACE
    assert not caplog.records


class _TileRaises(_Format):
    def tile(self, layer, k, n):
        raise _DefectError("no biases")


class _TileIsADot(_Format):
    def tile(self, layer, k, n):
        return _FACE


class _TileOfTwoTensors(_Format):
    def tile(self, layer, k, n):
        return replace(_tile(), tensors=("scales", "biases"))


class _TileOfTheFormatsTwoTensors(_TileOfTwoTensors):
    def __init__(self):
        super().__init__(tensors=("scales", "biases"))


@pytest.mark.parametrize(
    "fmt, layer, defect",
    [
        (_TileRaises(), _layer(), "_DefectError('no biases')"),
        (
            _TileIsADot(),
            _layer(),
            f'TypeError("tile() returned a {formats.Dot!r}, not a Tile")',
        ),
        (
            _Tiled(),
            _layer(tensors=("weight",)),
            "ValueError('face toy_tile_b4: the layer has no array scales')",
        ),
        (
            _Tiled(),
            _layer_with_small("scales"),
            "ValueError('face toy_tile_b4: scales of fewer than 8 elements, which "
            "MLX binds as constants, not as the buffer a face reads')",
        ),
        (
            _TileOfTwoTensors(),
            _layer(),
            "ValueError('face toy_tile_b4: biases is no tensor of quant format toy')",
        ),
        (
            _TileOfTheFormatsTwoTensors(),
            _layer(),
            "ValueError('face toy_tile_b4: the layer has no array biases')",
        ),
    ],
)
def test_a_defective_tile_is_no_face(plugins, caplog, dot_defects, fmt, layer, defect):
    formats.register(fmt)
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.tile(layer, 64, 8) is None
    (record,) = caplog.records
    assert record.getMessage() == (
        "quant format toy gave no usable tile face for a Module (n=8, k=64); "
        "the layer keeps its other kernels"
    )
    assert repr(record.exc_info[1]) == defect
    assert dot_defects == {("toy", defect)}
    dot_defects.clear()  # expected


def test_a_dot_that_gives_a_tile_is_defective(plugins, caplog, dot_defects):
    class DotIsATile(_Format):
        def dot(self, layer, k, n):
            return _tile()

    formats.register(DotIsATile())
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        assert formats.dot(_layer(), 64, 8) is None
    defect = f'TypeError("dot() returned a {formats.Tile!r}, not a Dot")'
    assert dot_defects == {("toy", defect)}
    dot_defects.clear()  # expected


def test_two_threads_that_meet_one_defect_log_it_once(plugins, caplog, monkeypatch):
    """Two engine threads may ask a defective format for a face at once:
    the defect is logged once."""
    formats.register(_TileRaises())
    together = threading.Barrier(2, timeout=10)
    asked = []

    class Met(set):
        def __contains__(self, defect):
            found = super().__contains__(defect)
            asked.append(threading.get_ident())
            if len(asked) <= 2:
                # Each thread waits for the other between its check and its
                # insertion: with the two apart, both would find it new.
                with contextlib.suppress(threading.BrokenBarrierError):
                    together.wait(timeout=0.5)
            return found

    met = Met()
    monkeypatch.setattr(formats, "_DOT_DEFECTS", met)
    threads = [
        threading.Thread(target=formats.tile, args=(_layer(), 64, 8)) for _ in range(2)
    ]
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    assert len(asked) == 2 and len(caplog.records) == 1
    assert met == {("toy", "_DefectError('no biases')")}


def test_one_error_of_a_formats_dot_and_tile_is_one_defect(
    plugins, caplog, dot_defects
):
    class BothRaise(_TileRaises):
        def dot(self, layer, k, n):
            raise _DefectError("no biases")

    formats.register(BothRaise())
    formats.register(_TileRaises("other"))
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        for _ in range(2):
            assert formats.dot(_layer(), 64, 8) is None
            assert formats.tile(_layer(), 64, 8) is None
            assert formats.tile(_layer("other"), 64, 8) is None
    logged = [(r.args[0], r.args[1], repr(r.exc_info[1])) for r in caplog.records]
    assert logged == [
        ("toy", "dot", "_DefectError('no biases')"),
        ("other", "tile", "_DefectError('no biases')"),
    ]
    assert dot_defects == {
        ("toy", "_DefectError('no biases')"),
        ("other", "_DefectError('no biases')"),
    }
    dot_defects.clear()  # expected


def test_a_tile_defect_fails_a_block_that_admits_none(plugins, dot_defects):
    formats.register(_TileRaises())
    with pytest.raises(AssertionError) as failure, testing.no_dot_defects():
        assert formats.tile(_layer(), 64, 8) is None
    assert str(failure.value) == (
        "a quant format's dot() or tile() is defective: "
        "[('toy', \"_DefectError('no biases')\")]"
    )
    assert not dot_defects


def test_an_interrupt_inside_tile_is_not_a_defect(plugins):
    class Interrupted(_Format):
        def tile(self, layer, k, n):
            raise KeyboardInterrupt

    formats.register(Interrupted())
    with pytest.raises(KeyboardInterrupt):
        formats.tile(_layer(), 64, 8)


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
    assert half != _FACE  # exact for one dtype, exact for the floats
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


def test_a_face_s_refusal_names_the_face():
    with pytest.raises(ValueError) as refusal:
        replace(_FACE, tensors=("2nd",))
    want = f"quant format face {_FACE.key!r}: '2nd' is not an identifier"
    assert str(refusal.value) == want


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"tensors": "scales"}, "quant format face 'toy_b4': tensors and dtypes are"),
        ({"tensors": ["scales"]}, "tensors and dtypes are tuples"),
        ({"dtypes": mx.float16}, "tensors and dtypes are tuples"),
        ({"tensors": ("scales", "scales")}, "beside 'weight' and 'bias', each named"),
        ({"tensors": ("weight",)}, "beside 'weight' and 'bias', each named once"),
        ({"tensors": ("scales", "weight")}, "beside 'weight' and 'bias', each named"),
        # A linear adds its bias itself.
        ({"tensors": ("scales", "bias")}, "beside 'weight' and 'bias', each named"),
        # A kernel is built for float activations alone.
        ({"dtypes": (mx.int32,)}, "activations are float16, bfloat16 or float32"),
        ({"dtypes": (mx.float16, mx.uint8)}, "activations are float16, bfloat16"),
        ({"dtypes": (mx.float64,)}, "activations are float16, bfloat16 or float32"),
        ({"dtypes": (None,)}, "dtypes are distinct mx.Dtype"),
        ({"dtypes": ("float16",)}, "dtypes are distinct mx.Dtype"),
        ({"dtypes": (mx.float16, [])}, "dtypes are distinct mx.Dtype"),
        ({"dtypes": (mx.float16, mx.float16)}, "dtypes are distinct mx.Dtype"),
    ],
)
def test_a_face_lists_its_tensors_and_dtypes_in_tuples(fields, message):
    with pytest.raises(ValueError, match=message):
        replace(_FACE, **fields)


def test_a_face_is_exact_for_its_dtypes_or_for_the_floats():
    assert _FACE.exact_for(mx.float16) and _FACE.exact_for(mx.bfloat16)
    assert _FACE.exact_for(mx.float32)
    assert (mx.float16, mx.bfloat16, mx.float32) == formats.FLOATS
    # A kernel computes in floats: integers would be truncated in its store.
    for dtype in (mx.int32, mx.uint8, mx.bool_, mx.float64):
        assert not _FACE.exact_for(dtype)
    half = replace(_FACE, dtypes=(mx.float16,))
    assert half.exact_for(mx.float16) and not half.exact_for(mx.bfloat16)
    halves = replace(_FACE, dtypes=(mx.float16, mx.bfloat16))
    assert halves.exact_for(mx.bfloat16) and not halves.exact_for(mx.float32)


class _Bare(formats.QuantFormat):
    mode, tensors, contract = "bare", (), formats.CONTRACT


def test_a_format_without_a_decode_says_so():
    with pytest.raises(NotImplementedError, match="quant format 'bare' has no decode"):
        _Bare().decode({"weight": mx.zeros((8, 8))}, 4, 32)


def test_a_format_without_a_layout_or_a_codec_keeps_no_matrix():
    # No layer of a checkpoint is then its to build, and oQ writes none in it.
    assert _Bare().layout(64, 8, 4, 64) is None
    assert _Bare().encode(mx.zeros((8, 64)), mx.ones((64,)), 4, 64) is None


class _Counting(_Tiled):
    """Counts what it is asked for."""

    def __init__(self, mode="toy"):
        super().__init__(mode)
        self.dots = self.tiles = 0

    def dot(self, layer, k, n):
        self.dots += 1
        return _FACE

    def tile(self, layer, k, n):
        self.tiles += 1
        return _tile()


def test_a_layer_asks_its_format_for_a_face_once(plugins):
    fmt, layer = _Counting(), _layer()
    formats.register(fmt)
    for _ in range(3):
        assert formats.dot(layer, 64, 8) is _FACE
        assert formats.tile(layer, 64, 8) is _tile()
    assert (fmt.dots, fmt.tiles) == (1, 1)
    # Another matrix shape is asked for.
    assert formats.dot(layer, 128, 8) is _FACE
    assert formats.dot(layer, 64, 16) is _FACE
    assert formats.dot(layer, 64, 8) is _FACE
    assert fmt.dots == 3


@pytest.mark.parametrize(
    "change",
    ["another array", "an array removed and put back", "bits", "group size", "mode"],
)
def test_a_layer_asks_again_once_what_its_face_depends_on_changes(plugins, change):
    fmt, other = _Counting(), _Counting("other")
    formats.register(fmt)
    formats.register(other)
    layer = _layer()
    layer.bits, layer.group_size = 4, 32
    assert formats.dot(layer, 64, 8) is _FACE
    if change == "another array":
        layer["scales"] = mx.zeros((8,))
    elif change == "an array removed and put back":
        layer["scales"] = layer.pop("scales") + 0
    elif change == "bits":
        layer.bits = 8
    elif change == "group size":
        layer.group_size = 64
    else:
        layer.mode = "other"
    for _ in range(2):
        assert formats.dot(layer, 64, 8) is _FACE
    assert fmt.dots + other.dots == 2


def test_a_mode_given_to_another_format_is_asked_of_it(plugins, monkeypatch):
    fmt, again = _Counting(), _Counting()
    formats.register(fmt)
    layer = _layer()
    assert formats.dot(layer, 64, 8) is _FACE
    monkeypatch.setitem(formats._FORMATS, "toy", again)
    assert formats.dot(layer, 64, 8) is _FACE
    assert (fmt.dots, again.dots) == (1, 1)


def test_what_is_no_module_is_asked_at_every_call(plugins):
    class Layer(dict):
        mode = "toy"

    fmt, layer = _Counting(), Layer(weight=mx.zeros((8,)), scales=mx.zeros((8,)))
    formats.register(fmt)
    for _ in range(2):
        assert formats.dot(layer, 64, 8) is _FACE
    assert fmt.dots == 2


def test_a_defect_is_asked_again_at_each_call_and_logged_once(
    plugins, caplog, dot_defects
):
    class Defective(_Counting):
        def dot(self, layer, k, n):
            self.dots += 1
            raise _DefectError("no face")

    fmt, layer = Defective(), _layer()
    formats.register(fmt)
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        for _ in range(2):
            assert formats.dot(layer, 64, 8) is None
    assert fmt.dots == 2
    assert len(caplog.records) == 1
    dot_defects.clear()  # expected
    # A block that admits no defect, opened later, meets it on the same layer.
    with pytest.raises(AssertionError, match="no face"), testing.no_dot_defects():
        formats.dot(layer, 64, 8)
    dot_defects.clear()


def test_a_layer_asks_again_when_a_tensor_is_kept_under_another_name(
    plugins, dot_defects
):
    fmt, layer = _Counting(), _layer()
    formats.register(fmt)
    assert formats.dot(layer, 64, 8) is _FACE
    # The same array, under another name: the face's tensor is gone.
    layer["other"] = layer.pop("scales")
    assert formats.dot(layer, 64, 8) is None
    assert fmt.dots == 2
    dot_defects.clear()  # expected: the layer has no array scales


def test_a_layer_asks_again_when_its_tensors_swap_names(plugins):
    fmt, layer = _Counting(), _layer()
    formats.register(fmt)
    assert formats.dot(layer, 64, 8) is _FACE
    weight, scales = layer.pop("weight"), layer.pop("scales")
    layer["scales"], layer["weight"] = weight, scales
    assert formats.dot(layer, 64, 8) is _FACE
    assert fmt.dots == 2


@pytest.mark.parametrize(
    "before, after", [((4.0, 32), (4, 32)), ((4, 32.0), (4, 32)), (None, (0, 0))]
)
def test_a_layer_asks_again_when_its_widths_change_in_type(plugins, before, after):
    fmt, layer = _Counting(), _layer()
    formats.register(fmt)
    if before is not None:
        layer.bits, layer.group_size = before
    assert formats.dot(layer, 64, 8) is _FACE
    layer.bits, layer.group_size = after
    assert formats.dot(layer, 64, 8) is _FACE
    assert fmt.dots == 2


def test_equal_formats_are_two(plugins, monkeypatch):
    class Equal(_Counting):
        def __eq__(self, other):
            return isinstance(other, Equal)

        __hash__ = object.__hash__

    fmt, again = Equal(), Equal()
    formats.register(fmt)
    layer = _layer()
    assert formats.dot(layer, 64, 8) is _FACE
    monkeypatch.setitem(formats._FORMATS, "toy", again)
    assert formats.dot(layer, 64, 8) is _FACE
    assert (fmt.dots, again.dots) == (1, 1)
