# SPDX-License-Identifier: Apache-2.0
"""Production admission for the Flash-Next config shipped by the PoC."""

import json

import pytest

from omlx import formats, oq
from omlx.patches import format_load


@pytest.mark.parametrize(
    "outer,inner",
    [("qwen4_exp", "qwen4_exp_text"), (" QWEN4-EXP ", "QWEN4-EXP-TEXT")],
)
def test_flash_next_writer_and_checked_loader_admit_its_format_record(
    tmp_path, outer, inner
):
    config = {
        "model_type": outer,
        "text_config": {"model_type": inner},
        "omlx_quant_formats": {"oqel": {}},
    }
    assert oq._unverified_family(config) is None
    (tmp_path / "config.json").write_text(json.dumps(config))
    format_load.check_engine_loader(tmp_path, "vlm")
    with format_load.loading(tmp_path) as checkpoint:
        assert checkpoint.verified
        assert checkpoint.verified_modes == frozenset({"affine", "oqel"})
        formats._may_build("affine")
        formats._may_build("oqel")
        with pytest.raises(formats.FormatRefusedError):
            formats._may_build("example")
        checkpoint.verified = False
        with pytest.raises(formats.FormatRefusedError):
            formats._may_build("oqel")
        checkpoint.verified = True
    with pytest.raises(formats.FormatRefusedError, match="embedding"):
        format_load.check_engine_loader(tmp_path, "embedding")


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen4_exp_text"},
        {"model_type": "qwen4_exp"},
        {"model_type": "qwen4_exp", "text_config": {}},
        {"model_type": "qwen4_exp", "text_config": {"model_type": "other"}},
        {"model_type": "other", "text_config": {"model_type": "qwen4_exp_text"}},
        {
            "model_type": "qwen4_exp",
            "text_config": {"model_type": "qwen4_exp_text", "model_file": "custom.py"},
        },
        {
            "model_type": "qwen4_exp",
            "model_file": "custom.py",
            "text_config": {"model_type": "qwen4_exp_text"},
        },
    ],
)
def test_only_the_stock_implementation_of_the_served_pair_is_admitted(tmp_path, config):
    config["omlx_quant_formats"] = {"example": {}}
    assert oq._unverified_family(config) is not None
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(formats.FormatRefusedError):
        format_load.check_engine_loader(tmp_path, "vlm")


def test_stock_flash_next_does_not_open_format_construction(tmp_path):
    config = {
        "model_type": "qwen4_exp",
        "text_config": {"model_type": "qwen4_exp_text"},
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with format_load.loading(tmp_path) as checkpoint:
        # Stock affine on an allowlisted family opens affine only, never
        # another format.
        assert checkpoint.verified
        assert checkpoint.verified_modes == frozenset({"affine"})
        formats._may_build("affine")
        with pytest.raises(formats.FormatRefusedError):
            formats._may_build("example")
        with pytest.raises(formats.FormatRefusedError):
            formats._may_build("oqel")


def test_recorded_affine_does_not_bypass_plugin_family_gate(tmp_path, monkeypatch):
    from omlx.formats import testing

    configs = (
        {
            "model_type": "qwen4_exp",
            "model_file": "custom.py",
            "text_config": {"model_type": "qwen4_exp_text"},
            "omlx_quant_formats": {"affine": {}, "example": {}},
        },
        {
            "model_type": "raon1_mini",
            "text_config": {"model_type": "qwen4_exp"},
            "omlx_quant_formats": {"affine": {}, "example": {}},
        },
        {
            "model_type": "raon1_mini",
            "omlx_quant_formats": {"affine": {}},
            "quantization": {"bits": 4, "group_size": 64},
        },
        {
            "model_type": "qwen4_exp",
            "text_config": {"model_type": "qwen4_exp_text"},
            "omlx_quant_formats": {"example": {}},
        },
        {
            "model_type": "qwen4_exp",
            "omlx_quant_formats": {"oqel": {}},
        },
        {
            "model_type": "qwen4_exp",
            "text_config": {"model_type": "qwen4_exp_text"},
            "omlx_quant_formats": {"affine": {}},
            "quantization": {"a": {"mode": "example"}},
        },
        {
            "model_type": "qwen4_exp",
            "model_file": "custom.py",
            "text_config": {"model_type": "qwen4_exp_text"},
            "omlx_quant_formats": {"affine": {}},
        },
    )
    assert frozenset({"qwen4_exp"}) == format_load.PORTED
    assert frozenset({("qwen4_exp", "oqel")}) == format_load.PLUGIN_ALLOWLIST
    assert not hasattr(format_load, "FORMAT_ALLOWLIST")
    for config in configs:
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(formats.FormatRefusedError):
            format_load.check_engine_loader(tmp_path, "vlm")
    # A registered plugin format outside PLUGIN_ALLOWLIST hits the plugin
    # allowlist refusal (not missing-plugin).
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "example", formats.get("affine"))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
                "omlx_quant_formats": {"example": {}},
            }
        )
    )
    with pytest.raises(formats.FormatRefusedError, match="plugin format allowlist"):
        format_load.check_engine_loader(tmp_path, "vlm")
    # Stock switch allowlist (PORTED) is never bypassed by testing.verified,
    # and PLUGIN_ALLOWLIST is never mutated.
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "omlx_quant_formats": {"affine": {}}})
    )
    with testing.verified("llama"):
        assert ("llama", None) in testing.verified_extra()
        assert ("llama", "toy") not in format_load.PLUGIN_ALLOWLIST
        with pytest.raises(formats.FormatRefusedError, match="stock switch allowlist"):
            format_load.check_engine_loader(tmp_path, "batched")
    assert testing.verified_extra() == frozenset()
    assert not format_load.ported_family(
        {"model_type": "raon1_mini", "text_config": {"model_type": "qwen4_exp"}}
    )


def test_apply_model_config_refuses_recorded_affine_beside_plugin(tmp_path):
    disk = {
        "model_type": "qwen4_exp",
        "model_file": "custom.py",
        "text_config": {"model_type": "qwen4_exp_text"},
    }
    (tmp_path / "config.json").write_text(json.dumps(disk))
    with (
        format_load.loading(tmp_path) as ckpt,
        pytest.raises(formats.FormatRefusedError),
    ):
        ckpt.apply_model_config(
            {"omlx_quant_formats": {"affine": {}, "example": {}}}
        )
    # Without model_file, an unlisted plugin beside affine is still refused.
    clean = {
        "model_type": "qwen4_exp",
        "text_config": {"model_type": "qwen4_exp_text"},
    }
    (tmp_path / "config.json").write_text(json.dumps(clean))
    with (
        format_load.loading(tmp_path) as ckpt,
        pytest.raises(formats.FormatRefusedError),
    ):
        ckpt.apply_model_config(
            {"omlx_quant_formats": {"affine": {}, "example": {}}}
        )
    # A model_file introduced via model_config on a recorded affine checkpoint
    # must reach the gate and be refused.
    recorded_affine = {**clean, "omlx_quant_formats": {"affine": {}}}
    (tmp_path / "config.json").write_text(json.dumps(recorded_affine))
    with (
        format_load.loading(tmp_path) as ckpt,
        pytest.raises(formats.FormatRefusedError, match="model_file"),
    ):
        ckpt.apply_model_config({"model_file": "custom.py"})
    # Bare qwen4_exp on disk with an override supplying qwen4_exp_text plus
    # affine and oqel must still check oqel against the on-disk family.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    with (
        format_load.loading(tmp_path) as ckpt,
        pytest.raises(formats.FormatRefusedError),
    ):
        ckpt.apply_model_config(
            {
                "text_config": {"model_type": "qwen4_exp_text"},
                "omlx_quant_formats": {"affine": {}, "oqel": {}},
            }
        )


def test_check_engine_loader_loads_builtins_before_family_gate(
    tmp_path, monkeypatch
):
    config = {
        "model_type": "qwen4_exp",
        "omlx_quant_formats": {"affine": {}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(formats, "_loaded", False)
    monkeypatch.setattr(formats, "_BUILTIN_MODES", set())
    format_load.check_engine_loader(tmp_path, "vlm")


def test_refusal_order_preserves_specific_messages_before_allowlist(
    tmp_path, monkeypatch
):
    import mlx.core as mx
    import mlx.nn as nn

    from omlx.formats.linear import FormatLinear

    format_load.apply_format_load_patch()
    base = {
        "model_type": "qwen4_exp",
        "text_config": {"model_type": "qwen4_exp_text"},
    }
    # 1. Damaged or blank records on an allowlisted family report the damaged
    #    config or layout error, not an allowlist mismatch on "unknown".
    (tmp_path / "config.json").write_text(
        json.dumps({**base, "omlx_quant_formats": None})
    )
    with pytest.raises(
        formats.FormatRefusedError, match="checkpoint's config is damaged"
    ):
        format_load.loading(tmp_path).__enter__()
    for bad, expected in (
        ({"omlx_quant_formats": {}}, "unknown format layout at omlx_quant_formats"),
        (
            {"omlx_quant_formats": {" ": {}}},
            "unknown format layout at omlx_quant_formats",
        ),
        (
            {"quantization": {"mode": " "}},
            "unknown quantization layout at quantization.mode",
        ),
        (
            {"omlx_quant_formats": {"example": None}},
            "checkpoint's config is damaged",
        ),
    ):
        (tmp_path / "config.json").write_text(json.dumps({**base, **bad}))
        with pytest.raises(formats.FormatRefusedError, match=expected):
            format_load.loading(tmp_path).__enter__()
    # 2. A recorded built-in affine claiming even an empty plugin string keeps
    #    the MLX-mode plugin refusal at load before any layer is built.
    (tmp_path / "config.json").write_text(
        json.dumps({**base, "omlx_quant_formats": {"affine": {"plugin": ""}}})
    )
    with pytest.raises(
        formats.FormatRefusedError, match="which a plugin does not write"
    ):
        format_load.loading(tmp_path).__enter__()
    # 3. Missing plugin reports the missing plugin before the allowlist refusal
    #    (for both allowlisted oqel and an unlisted uninstalled plugin mode).
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **base,
                "omlx_quant_formats": {
                    "missing_fmt": {"plugin": "omlx-missing 0.1.0"}
                },
            }
        )
    )
    with pytest.raises(
        formats.FormatRefusedError,
        match="no format of this mode is installed: install omlx-missing 0.1.0",
    ):
        format_load.loading(tmp_path).__enter__()
    saved = [
        (r, r.copy())
        for r in (
            formats._FORMATS,
            formats._ORIGINS,
            formats._SKIPPED,
            formats._BUILTIN_MODES,
        )
    ]
    was_loaded = formats._loaded
    monkeypatch.setattr(formats, "entry_points", lambda **_kw: [])
    monkeypatch.setattr(formats, "_loaded", False)
    try:
        formats.load()
        formats._FORMATS.pop("oqel", None)
        formats._ORIGINS.pop("oqel", None)
        formats._SKIPPED.pop("oqel", None)
        (tmp_path / "config.json").write_text(
            json.dumps(
                {**base, "omlx_quant_formats": {"oqel": {"plugin": "omlx-oqel 1.0.0"}}}
            )
        )
        with (
            mx.stream(mx.cpu),
            format_load.loading(tmp_path),
            pytest.raises(
                formats.FormatRefusedError,
                match="no format of this mode is installed: install omlx-oqel 1.0.0",
            ) as no,
        ):
            nn.Linear(128, 32, bias=False).to_quantized(
                group_size=64, bits=4, mode="oqel"
            )
        assert no.value.installed == "none"
    finally:
        for r, snap in saved:
            r.clear()
            r.update(snap)
        formats._loaded = was_loaded
    # 4. A mixed allowlisted oqel + stock affine checkpoint builds stock affine
    #    FormatLinear layers while still refusing unlisted format construction.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **base,
                "omlx_quant_formats": {"oqel": {}},
                "quantization": {"bits": 4, "group_size": 64},
            }
        )
    )
    with mx.stream(mx.cpu), format_load.loading(tmp_path):
        layer = nn.Linear(128, 32, bias=False).to_quantized(group_size=64, bits=4)
        assert type(layer) is FormatLinear
        with pytest.raises(formats.FormatRefusedError):
            formats._may_build("example")


def test_verified_block_is_context_local_for_an_ordinary_load(
    tmp_path, monkeypatch
):
    import threading

    from omlx.formats import testing

    monkeypatch.setitem(formats._FORMATS, "toy", object())
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "omlx_quant_formats": {"toy": {}}})
    )
    outcome = {}
    with testing.verified("llama"):
        format_load.check_engine_loader(tmp_path, "vlm")

        def other():
            try:
                format_load.check_engine_loader(tmp_path, "vlm")
                outcome["error"] = None
            except Exception as error:
                outcome["error"] = error

        worker = threading.Thread(target=other)
        worker.start()
        worker.join(timeout=30)
    assert isinstance(outcome.get("error"), formats.FormatRefusedError)


def test_a_verified_block_restores_its_context_on_an_exception(tmp_path):
    from omlx.formats import testing

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "omlx_quant_formats": {"toy": {}}})
    )
    with pytest.raises(RuntimeError):
        with testing.verified("llama"):
            assert ("llama", None) in testing.verified_extra()
            raise RuntimeError("boom")
    assert testing.verified_extra() == frozenset()
    with pytest.raises(formats.FormatRefusedError):
        format_load.check_engine_loader(tmp_path, "vlm")


def test_a_verified_block_is_reentrant_and_context_local():
    import contextvars

    from omlx.formats import testing as format_testing

    block = format_testing.verified("llama")
    with block:
        outer = block._entries.get()
        with block:
            assert len(block._entries.get()) == len(outer) + 1
        assert block._entries.get() == outer
        assert ("llama", None) in format_testing.verified_extra()
        assert formats._TESTING.get() is True
        copied = contextvars.copy_context()
        copied.run(block.__enter__)
        assert len(copied.run(block._entries.get)) == len(outer) + 1
        copied.run(block.__exit__, None, None, None)
        assert block._entries.get() == outer
    assert block._entries.get() == ()
    assert format_testing.verified_extra() == frozenset()


def test_the_testing_switch_resets_and_is_context_local():
    import threading

    from omlx import formats
    from omlx.formats import testing as format_testing

    seen = {}
    with format_testing.verified("llama"):
        assert formats._TESTING.get() is True

        def other():
            seen["testing"] = formats._TESTING.get()
            seen["extra"] = format_testing.verified_extra()

        worker = threading.Thread(target=other)
        worker.start()
        worker.join(30)
    assert formats._TESTING.get() is False
    assert seen == {"testing": False, "extra": frozenset()}


def test_verified_blocks_do_not_leak_contextvar_entries():
    import contextvars

    from omlx.formats import testing as format_testing

    before = len(contextvars.copy_context())
    for _ in range(10):
        with format_testing.verified("llama"):
            pass
    assert len(contextvars.copy_context()) == before


def test_verified_docs_describe_context_local_contract():
    from pathlib import Path

    page = (Path(__file__).parents[1] / "docs/quant-formats.md").read_text()
    joined = " ".join(page.split())
    assert "while one of its blocks is open, in any thread" not in joined
    assert "another thread does not see the block" in joined


def test_verified_resets_captured_vars_across_monkeypatch(monkeypatch):
    import contextvars

    from omlx import formats
    from omlx.formats import testing as format_testing

    original_extra = format_testing._VERIFIED_EXTRA
    original_testing = formats._TESTING
    with format_testing.verified("llama"):
        monkeypatch.setattr(
            format_testing,
            "_VERIFIED_EXTRA",
            contextvars.ContextVar("omlx_verified_extra", default=frozenset()),
        )
        monkeypatch.setattr(
            formats,
            "_TESTING",
            contextvars.ContextVar("omlx_formats_testing", default=False),
        )
    assert original_extra.get() == frozenset()
    assert original_testing.get() is False


def test_oqh_key_does_not_bypass_quant_format_family_gate(tmp_path):
    """An omlx_oqh key declines the stock affine switch, never the format gates."""
    bad_family = tmp_path / "raon_oqh"
    bad_family.mkdir()
    (bad_family / "config.json").write_text(
        json.dumps(
            {
                "model_type": "raon1_mini",
                "omlx_oqh": {"version": 1},
                "omlx_quant_formats": {"toy": {}},
            }
        )
    )
    with pytest.raises(formats.FormatRefusedError, match="loads no raon1_mini"):
        format_load.check_engine_loader(bad_family, "vlm")

    builtin_oqh = tmp_path / "qwen4_oqh_affine"
    builtin_oqh.mkdir()
    (builtin_oqh / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
                "omlx_oqh": {"version": 1},
                "omlx_quant_formats": {"affine": {}},
            }
        )
    )
    with pytest.raises(formats.FormatRefusedError, match="oQh"):
        format_load.check_engine_loader(builtin_oqh, "vlm")

    custom_file = tmp_path / "qwen4_oqh_model_file"
    custom_file.mkdir()
    (custom_file / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
                "omlx_oqh": {"version": 1},
                "model_file": "custom.py",
                "omlx_quant_formats": {"oqel": {}},
            }
        )
    )
    with pytest.raises(formats.FormatRefusedError, match="model_file"):
        format_load.check_engine_loader(custom_file, "vlm")

    stock = tmp_path / "qwen4_stock"
    stock.mkdir()
    (stock / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
            }
        )
    )
    with format_load.loading(stock) as checkpoint:
        assert checkpoint.verified and format_load._ported(checkpoint)
        checkpoint.apply_model_config({"omlx_oqh": {"version": 1}})
        assert not checkpoint.verified
        assert checkpoint.verified_modes == frozenset()
        assert not format_load._ported(checkpoint)

    pure_oqh = tmp_path / "qwen4_pure_oqh"
    pure_oqh.mkdir()
    (pure_oqh / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
                "omlx_oqh": {"version": 1},
            }
        )
    )
    format_load.check_engine_loader(pure_oqh, "vlm")
    with format_load.loading(pure_oqh) as checkpoint:
        assert not checkpoint.verified
        assert checkpoint.verified_modes == frozenset()
        assert not format_load._ported(checkpoint)
