# SPDX-License-Identifier: Apache-2.0
"""Construction-time non-Metal backend guard and CONTRACT_STABILITY tests."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import (
    Codes,
    Dot,
    FormatRefusedError,
    Table,
    Tile,
    testing,
    transforms,
)
from omlx.formats.linear import (
    _PENDING_WRITE,
    FormatEmbedding,
    FormatLinear,
    FormatSwitchLinear,
    _TransformedFormatEmbedding,
    _TransformedFormatLinear,
    _TransformedFormatSwitchLinear,
)
from omlx.patches import format_load

_AFFINE_NAMES = ("weight", "scales", "biases")
_BLOCK = 128


class _RotatedFormat(formats.QuantFormat):
    """Minimal format declaring an input transform for layer-class checks."""

    mode = "backend_guard_rotated"
    contract = formats.CONTRACT
    codes_name = "codes"
    tensors = ()
    shared_tensors = ("signs",)

    def layout(self, k, n, bits, group_size):
        if k % _BLOCK:
            return None
        return {
            self.codes_name: ((n, k), mx.float32),
            "signs": ((k // _BLOCK, _BLOCK), mx.float32),
        }

    def decode(self, tensors, bits, group_size):
        return tensors[self.codes_name]

    def input_transform(self, layer):
        return transforms.SignedWhtTransform(
            block=layer["signs"].shape[-1], tensors=("signs",)
        )


def _affine_tensors(k: int = 128, n: int = 32, *, stacked: int = 0) -> dict:
    with mx.stream(mx.cpu):
        shape = (stacked, n, k) if stacked else (n, k)
        w = mx.zeros(shape, dtype=mx.bfloat16)
        packed, scales, biases = mx.quantize(w, group_size=32, bits=4, mode="affine")
        mx.eval(packed, scales, biases)
    return {"weight": packed, "scales": scales, "biases": biases}


def _rotated_tensors(k: int = 128, n: int = 16, *, stacked: int = 0) -> dict:
    with mx.stream(mx.cpu):
        shape = (stacked, n, k) if stacked else (n, k)
        codes = mx.zeros(shape, dtype=mx.float32)
        signs = mx.ones((k // _BLOCK, _BLOCK), dtype=mx.float32)
        mx.eval(codes, signs)
    return {"codes": codes, "signs": signs}


def _disable_metal(monkeypatch) -> None:
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)


def test_contract_stability_is_experimental():
    assert isinstance(formats.CONTRACT_STABILITY, str)
    assert formats.CONTRACT_STABILITY == "experimental"


@pytest.mark.parametrize(
    "kind,expected_substrings",
    [
        ("metal_false", ("Metal", "cpu")),
        ("metal_missing", ("Metal", "cpu")),
        ("metal_raises", ("Metal", "cpu")),
        ("gpu_false", ("Metal", "mx.is_available(mx.gpu) is False")),
        ("kernel_missing", ("mx.fast.metal_kernel",)),
    ],
)
def test_metal_backend_available_false_when_metal_unavailable(
    monkeypatch, kind, expected_substrings
):
    if kind == "metal_false":
        monkeypatch.setattr(mx.metal, "is_available", lambda: False)
        monkeypatch.setattr(mx, "is_available", lambda dev: dev == mx.cpu)
        monkeypatch.setattr(
            mx, "cuda", SimpleNamespace(is_available=lambda: False), raising=False
        )
    elif kind == "metal_missing":
        monkeypatch.setattr(mx, "metal", None)
        monkeypatch.setattr(mx, "is_available", lambda dev: dev == mx.cpu)
        monkeypatch.setattr(
            mx, "cuda", SimpleNamespace(is_available=lambda: False), raising=False
        )
    elif kind == "metal_raises":

        def _boom():
            raise RuntimeError("metal probe failed")

        monkeypatch.setattr(mx.metal, "is_available", _boom)
        monkeypatch.setattr(mx, "is_available", lambda dev: dev == mx.cpu)
        monkeypatch.setattr(
            mx, "cuda", SimpleNamespace(is_available=lambda: False), raising=False
        )
    elif kind == "gpu_false":
        monkeypatch.setattr(mx.metal, "is_available", lambda: True)
        monkeypatch.setattr(mx, "is_available", lambda dev: False)
    elif kind == "kernel_missing":
        monkeypatch.setattr(mx.metal, "is_available", lambda: True)
        monkeypatch.setattr(mx, "is_available", lambda dev: True)
        monkeypatch.setattr(mx, "fast", SimpleNamespace())

    assert formats.metal_backend_available() is False
    with pytest.raises(FormatRefusedError) as exc_info:
        formats.require_metal_backend("oqel")
    err = exc_info.value
    assert err.mode == "oqel"
    for needle in expected_substrings:
        assert needle in str(err)


def test_cuda_backend_refused_even_when_gpu_is_available(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    monkeypatch.setattr(mx, "is_available", lambda dev: True)
    monkeypatch.setattr(
        mx, "cuda", SimpleNamespace(is_available=lambda: True), raising=False
    )

    assert formats.metal_backend_available() is False
    with pytest.raises(FormatRefusedError) as exc_info:
        formats.require_metal_backend("affine")
    assert exc_info.value.mode == "affine"
    assert "Metal" in str(exc_info.value)
    assert "cuda" in str(exc_info.value)


def test_non_metal_unknown_gpu_refused_with_arch_name(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    monkeypatch.setattr(mx, "is_available", lambda dev: True)
    monkeypatch.setattr(
        mx, "cuda", SimpleNamespace(is_available=lambda: False), raising=False
    )
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda *_args, **_kwargs: {"architecture": "vulkan_test"},
        raising=False,
    )

    assert formats.metal_backend_available() is False
    with pytest.raises(FormatRefusedError) as exc_info:
        formats.require_metal_backend("affine")
    assert "vulkan_test" in str(exc_info.value)
    assert "Metal" in str(exc_info.value)


def test_all_format_layer_classes_fail_closed_off_metal(monkeypatch):
    affine_2d = _affine_tensors(128, 32)
    affine_3d = _affine_tensors(128, 32, stacked=2)
    rot_fmt = _RotatedFormat()
    rot_2d = _rotated_tensors(128, 16)
    rot_3d = _rotated_tensors(128, 16, stacked=2)

    _disable_metal(monkeypatch)
    monkeypatch.setattr(mx, "is_available", lambda dev: dev == mx.cpu)

    cases = (
        lambda: FormatLinear("affine", 4, 32, 128, affine_2d),
        lambda: FormatSwitchLinear("affine", 4, 32, 128, affine_3d),
        lambda: FormatEmbedding("affine", 4, 32, 128, affine_2d),
        lambda: FormatLinear(rot_fmt, 8, 1, 128, rot_2d),
        lambda: FormatSwitchLinear(rot_fmt, 8, 1, 128, rot_3d),
        lambda: FormatEmbedding(rot_fmt, 8, 1, 128, rot_2d),
        lambda: _TransformedFormatLinear(rot_fmt, 8, 1, 128, rot_2d),
        lambda: _TransformedFormatSwitchLinear(rot_fmt, 8, 1, 128, rot_3d),
        lambda: _TransformedFormatEmbedding(rot_fmt, 8, 1, 128, rot_2d),
    )
    with testing.verified("qwen4_exp"):
        for build in cases:
            with pytest.raises(FormatRefusedError) as exc_info:
                build()
            assert "Metal" in str(exc_info.value)


def test_checkpoint_load_refuses_off_metal_with_checkpoint_metadata(
    monkeypatch, tmp_path
):
    format_load.apply_format_load_patch()
    config = {
        "model_type": "qwen4_exp",
        "quantization": {"group_size": 64, "bits": 4},
        "omlx_quant_formats": {
            "affine": {"plugin": None, "contract": formats.CONTRACT}
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    _disable_metal(monkeypatch)
    monkeypatch.setattr(mx, "is_available", lambda dev: dev == mx.cpu)

    with (
        mx.stream(mx.cpu),
        format_load.loading(tmp_path),
        pytest.raises(FormatRefusedError) as exc_info,
    ):
        nn.Linear(128, 32, bias=False).to_quantized(group_size=64, bits=4)

    err = exc_info.value
    assert err.path == tmp_path.resolve()
    assert err.mode == "affine"
    assert err.written == "a format registered by hand"
    assert err.installed == "built-in"
    assert "Metal" in str(err)


def test_write_permit_construction_fails_closed_off_metal_and_forged_permit_never_reads_mode(
    monkeypatch,
):
    tensors = _affine_tensors(128, 32)
    reader = formats.get("affine")
    assert reader is not None

    _disable_metal(monkeypatch)
    monkeypatch.setattr(mx, "is_available", lambda dev: dev == mx.cpu)

    # (a) Valid _write_permit must still fail closed when Metal is unavailable.
    token = object()
    _PENDING_WRITE.construction = (
        token,
        FormatLinear,
        reader,
        4,
        32,
        128,
        tensors,
    )
    try:
        with pytest.raises(FormatRefusedError) as exc_info:
            FormatLinear(reader, 4, 32, 128, tensors, _write_permit=token)
    finally:
        _PENDING_WRITE.construction = None
    assert exc_info.value.mode == "affine"
    assert "Metal" in str(exc_info.value)

    # (b) Forged _write_permit must refuse BEFORE reading fmt.mode.
    class _HostileFormat:
        @property
        def mode(self):
            raise AssertionError("forged write permit must never read fmt.mode")

    with pytest.raises(FormatRefusedError) as forged_exc:
        FormatLinear(_HostileFormat(), 4, 32, 128, tensors, _write_permit=object())
    assert forged_exc.value.mode == "unknown"
    assert "the write did not admit this layer" in str(forged_exc.value)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_cpu_stream_context_on_metal_host_still_constructs_layers():
    tensors = _affine_tensors(128, 32)
    with testing.verified("qwen4_exp"), mx.stream(mx.cpu):
        assert formats.metal_backend_available() is True
        layer = FormatLinear("affine", 4, 32, 128, tensors)
    assert isinstance(layer, FormatLinear)


def test_metadata_registration_and_faces_work_without_metal(monkeypatch):
    formats.load()
    monkeypatch.setattr(formats, "_FORMATS", dict(formats._FORMATS))
    monkeypatch.setattr(formats, "_BUILTIN_MODES", set(formats._BUILTIN_MODES))
    _disable_metal(monkeypatch)
    monkeypatch.setattr(mx, "is_available", lambda dev: dev == mx.cpu)

    class _DummyFormat(formats.QuantFormat):
        mode = "off_metal_meta_dummy"
        contract = formats.CONTRACT
        codes_name = "codes"
        tensors = ("gain",)

    dummy = _DummyFormat()
    formats.register(dummy)
    assert formats.get("off_metal_meta_dummy") is dummy

    table = Table("lut", mx.float32, (0.0, 1.0, 2.0, 3.0))
    dot = Dot(
        key="off_metal_dot",
        source=(
            "struct DummyDot {\n"
            f"  {formats.slot_marker('load')} {{ return 0.0f; }}\n"
            f"  {formats.slot_marker('madd')} {{ return 0.0f; }}\n"
            "};"
        ),
        type="DummyDot",
        tensors=("gain",),
        tables=(table,),
        reads=formats.DotReads(),
    )
    tile = Tile(
        key="off_metal_tile",
        source=(
            "struct DummyTile {\n"
            f"  {formats.slot_marker('weight')} {{ return 0.0f; }}\n"
            "};"
        ),
        type="DummyTile",
        tensors=("gain",),
        reads=formats.TileReads(),
    )
    codes = Codes(
        key="off_metal_codes",
        source=(
            "struct DummyCodes {\n"
            f"  {formats.slot_marker('pair')} {{ return 0u; }}\n"
            "};"
        ),
        type="DummyCodes",
        tensors=("gain",),
        reads=formats.CodesReads(),
    )
    assert dot.key == "off_metal_dot"
    assert tile.key == "off_metal_tile"
    assert codes.key == "off_metal_codes"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_zero_hot_path_overhead_after_construction(monkeypatch):
    tensors = _affine_tensors(128, 32)
    with testing.verified("qwen4_exp"):
        layer = FormatLinear("affine", 4, 32, 128, tensors)
    x = mx.ones((1, 128), dtype=mx.bfloat16)
    mx.eval(layer(x))

    calls = 0
    real_is_available = mx.metal.is_available

    def _counting_is_available():
        nonlocal calls
        calls += 1
        return real_is_available()

    monkeypatch.setattr(mx.metal, "is_available", _counting_is_available)
    out = layer(x)
    mx.eval(out)
    assert calls == 0


def test_concurrent_off_metal_refusals_are_thread_safe(monkeypatch):
    tensors = _affine_tensors(128, 32)
    _disable_metal(monkeypatch)
    monkeypatch.setattr(mx, "is_available", lambda dev: dev == mx.cpu)

    def _worker():
        with testing.verified("qwen4_exp"):
            try:
                FormatLinear("affine", 4, 32, 128, tensors)
            except FormatRefusedError as err:
                return err
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _worker(), range(8)))

    assert len(results) == 8
    for err in results:
        assert isinstance(err, FormatRefusedError)
        assert err.mode == "affine"
        assert "Metal" in str(err)


def test_require_metal_backend_normalizes_invalid_mode_and_unadmitted_still_refuses(
    monkeypatch,
):
    tensors = _affine_tensors(128, 32)
    _disable_metal(monkeypatch)
    monkeypatch.setattr(mx, "is_available", lambda dev: dev == mx.cpu)

    for bad_mode in (None, "", 123):
        with pytest.raises(FormatRefusedError) as exc_info:
            formats.require_metal_backend(bad_mode)  # type: ignore[arg-type]
        assert exc_info.value.mode == "unknown"

    # Unadmitted construction (outside format_load and testing.verified) must
    # keep the admission refusal even off-Metal.
    with pytest.raises(FormatRefusedError) as unadmitted:
        FormatLinear("affine", 4, 32, 128, tensors)
    assert "built outside" in str(unadmitted.value)
