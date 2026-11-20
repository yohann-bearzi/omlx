# SPDX-License-Identifier: Apache-2.0
"""A format write must verify the bytes it will publish."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from omlx import oq
from tests.test_formats_oq import _SETTINGS, _calibration
from tests.test_formats_oq import llama_source as llama_source
from tests.test_formats_oq import toy as toy

_VLM_CONTEXTS = (
    "_strip_audio_config_if_orphaned",
    "_strip_vision_config_if_orphaned",
    "_drop_gemma4_mlx_shared_kv_extras_on_load",
    "_derive_gemma4_global_kv_on_load",
    "_preserve_gemma4_unified_image_processor_on_load",
    "_force_minimax_m3_moe_sanitize_on_load",
    "_force_qwen4_exp_sanitize_on_load",
    "_remap_nested_visual_on_load",
    "_transpose_qwen35_mlx_vision_patch_embed_on_load",
    "_load_optiq_vision_sidecar_on_load",
)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("damage", ["missing-scale", "nonfinite-norm", "missing-norm"])
def test_writer_keeps_a_broken_reload_unverified(
    llama_source, toy, tmp_path, monkeypatch, damage
):
    output = tmp_path / "result"
    staging = output.with_suffix(".unverified")
    monkeypatch.setattr(oq, "_TEST_VERIFIED", frozenset({"llama"}))
    _calibration(monkeypatch)
    save = mx.save_safetensors
    damaged = []

    def save_shard(file, tensors, **kwargs):
        # The writer's shard. The oQe refinement (#5292) rewrites it from the
        # saved bytes (".refine-<shard>", then replaced): the damage carries.
        name = Path(file).name
        if Path(file).parent == staging and not name.startswith(".refine-"):
            tensors = dict(tensors)
            if damage == "missing-scale":
                down = "model.layers.0.mlp.down_proj."
                names = [down + name for name in toy.tensors if down + name in tensors]
                assert names, "the format's scale tensor is missing from the shard"
                tensors.pop(names[0])
            elif damage == "missing-norm":
                tensors.pop("model.norm.weight")
            else:
                norm = tensors["model.norm.weight"]
                tensors["model.norm.weight"] = mx.full(
                    norm.shape, float("nan"), dtype=norm.dtype
                )
            damaged.append(Path(file).name)
        return save(file, tensors, **kwargs)

    monkeypatch.setattr(mx, "save_safetensors", save_shard)
    error = None
    try:
        oq.quantize_oq_streaming(
            str(llama_source),
            str(output),
            4,
            enhanced=True,
            quant_format=toy.mode,
            **_SETTINGS,
        )
    except (ValueError, RuntimeError) as exc:
        error = exc

    assert damaged, "the saved checkpoint must contain the injected corruption"
    assert not output.exists(), "the writer published a damaged checkpoint"
    assert error is not None
    record = json.loads((staging / "config.json").read_text())["omlx_verification"]
    assert record["verified"] is False
    assert record["mode"] == "toy"
    assert (staging / "VERIFY-FAILED.txt").read_text().strip()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_writer_runs_the_saved_model_before_marking_it_verified(
    llama_source, toy, tmp_path, monkeypatch
):
    from omlx.formats.linear import FormatLinear

    output = tmp_path / "result"
    staging = output.with_suffix(".unverified")
    monkeypatch.setattr(oq, "_TEST_VERIFIED", frozenset({"llama"}))
    _calibration(monkeypatch)
    forward = FormatLinear.__call__
    observed = []

    def observe(layer, x):
        if (staging / "config.json").is_file():
            config = json.loads((staging / "config.json").read_text())
            observed.append((config["omlx_verification"]["verified"], output.exists()))
        return forward(layer, x)

    monkeypatch.setattr(FormatLinear, "__call__", observe)
    oq.quantize_oq_streaming(
        str(llama_source),
        str(output),
        4,
        enhanced=True,
        quant_format=toy.mode,
        **_SETTINGS,
    )
    assert observed, "the writer never ran a forward on the staged format checkpoint"
    assert set(observed) == {(False, False)}
    assert not staging.exists()
    assert (
        json.loads((output / "config.json").read_text())["omlx_verification"][
            "verified"
        ]
        is True
    )


def _unverified_checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "omlx_verification": {
                    "verified": False,
                    "reason": "its quantization did not complete",
                },
            }
        )
    )
    return path


def test_verifying_admits_only_its_checkpoint(tmp_path):
    from omlx.patches import format_load

    first = _unverified_checkpoint(tmp_path / "first")
    second = _unverified_checkpoint(tmp_path / "second")
    with format_load.verifying(first):
        with format_load.loading(first):
            pass
        with pytest.raises(ValueError):
            with format_load.loading(second):
                pass
    with pytest.raises(ValueError):
        with format_load.loading(first):
            pass


def test_verifying_propagates_to_copied_context_thread(tmp_path):
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    from omlx.patches import format_load

    first = _unverified_checkpoint(tmp_path / "first")
    with format_load.verifying(first):
        context = contextvars.copy_context()

        def load():
            with format_load.loading(first):
                pass

        with ThreadPoolExecutor(1) as pool:
            pool.submit(context.run, load).result()


def test_verifying_copied_context_revoked_after_exit(tmp_path):
    import contextvars

    from omlx.patches import format_load

    first = _unverified_checkpoint(tmp_path / "first")
    with format_load.verifying(first):
        context = contextvars.copy_context()

    def load():
        with format_load.loading(first):
            pass

    with pytest.raises(ValueError):
        context.run(load)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_writer_forwards_trust_remote_code(llama_source, toy, tmp_path, monkeypatch):
    from omlx import oq_verify

    output = tmp_path / "trust"
    monkeypatch.setattr(oq, "_TEST_VERIFIED", frozenset({"llama"}))
    _calibration(monkeypatch)
    seen = {}

    def verify(path, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(oq_verify, "verify_written_format_checkpoint", verify)
    oq.quantize_oq_streaming(
        str(llama_source),
        str(output),
        4,
        enhanced=True,
        quant_format=toy.mode,
        trust_remote_code=True,
        **_SETTINGS,
    )
    assert seen.get("trust_remote_code") is True


def test_vlm_check_loaded_failure_closes_model(tmp_path, monkeypatch):
    from contextlib import contextmanager, nullcontext

    from omlx import oq, oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))

    class Model:
        closed = False

        def close(self):
            self.closed = True

    model = Model()

    @contextmanager
    def patches(path, config, **kwargs):
        yield True

    monkeypatch.setattr(oq, "_calibration_patches", patches)
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr("mlx_vlm.utils.load_model", lambda *a, **k: model)

    def refuse(loaded):
        raise ValueError("the loaded model is not admissible")

    monkeypatch.setattr(oq, "check_loaded", refuse)
    with pytest.raises(ValueError, match="not admissible"):
        oq_verify.verify_written_format_checkpoint(output)
    assert model.closed


def test_vlm_close_failure_preserves_verification_error(tmp_path, monkeypatch):
    from contextlib import contextmanager, nullcontext

    from omlx import oq, oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    cleared = []

    class Model:
        def close(self):
            raise RuntimeError("close boom")

    model = Model()

    @contextmanager
    def patches(path, config, **kwargs):
        yield True

    monkeypatch.setattr(oq, "_calibration_patches", patches)
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr("mlx_vlm.utils.load_model", lambda *a, **k: model)
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(True))

    def refuse(loaded):
        raise ValueError("the loaded model is not admissible")

    monkeypatch.setattr(oq, "check_loaded", refuse)
    with pytest.raises(ValueError, match="not admissible"):
        oq_verify.verify_written_format_checkpoint(output)
    assert "not admissible" in (output / "VERIFY-FAILED.txt").read_text()
    record = json.loads((output / "config.json").read_text())["omlx_verification"]
    assert "not admissible" in record["reason"]
    assert cleared


def test_vlm_verification_loads_strictly(tmp_path, monkeypatch):
    from contextlib import contextmanager, nullcontext

    import mlx.nn as nn

    from omlx import oq, oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    seen = {}

    class Model(nn.Module):
        def __call__(self, x):
            return mx.zeros((1, 2, 4))

        def close(self):
            pass

    @contextmanager
    def patches(path, config, **kwargs):
        yield True

    def load(*args, **kwargs):
        seen.update(kwargs)
        return Model()

    monkeypatch.setattr(oq, "_calibration_patches", patches)
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr("mlx_vlm.utils.load_model", load)
    monkeypatch.setattr(oq, "check_loaded", lambda model: None)
    oq_verify.verify_written_format_checkpoint(output)
    assert seen.get("strict") is True


def test_text_only_qwen4_exp_routes_to_vlm_during_verification(tmp_path, monkeypatch):
    from contextlib import nullcontext

    import mlx.nn as nn

    import omlx.utils.model_loading as model_loading
    from omlx import oq, oq_verify

    output = tmp_path / "qwen"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    seen = []

    class Model(nn.Module):
        def __call__(self, x):
            return mx.zeros((1, 2, 4))

        def close(self):
            pass

    def pre_load(path, model_settings=None, for_vlm=False):
        seen.append(for_vlm)

    def wrong(*args, **kwargs):
        raise AssertionError("wrong loader")

    monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", pre_load)
    monkeypatch.setattr(model_loading, "lm_load_compat", wrong)
    monkeypatch.setattr("mlx_vlm.utils.load_model", lambda *a, **k: Model())
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr(oq, "check_loaded", lambda model: None)
    oq_verify.verify_written_format_checkpoint(output)
    assert seen == [True]


@pytest.mark.parametrize("failure", ["bad-config", "io-error"])
def test_writer_failure_preserves_cause_on_bad_config_or_io_error(
    tmp_path, monkeypatch, failure
):
    from contextlib import contextmanager, nullcontext

    import omlx.engine.vlm as engine_vlm
    from omlx import oq, oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    if failure == "bad-config":
        (output / "config.json").write_text("[]")
    else:
        (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))

    class Model:
        def close(self):
            pass

    @contextmanager
    def patches(path, config, **kwargs):
        yield True

    for name in _VLM_CONTEXTS:
        monkeypatch.setattr(engine_vlm, name, lambda path: nullcontext())

    def refuse(loaded):
        raise ValueError("the loaded model is not admissible")

    monkeypatch.setattr(oq, "_calibration_patches", patches)
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr("mlx_vlm.utils.load_model", lambda *a, **k: Model())
    monkeypatch.setattr(oq, "check_loaded", refuse)
    if failure == "io-error":

        def disk_full(path, value):
            raise OSError("disk full")

        monkeypatch.setattr(oq, "_atomic_write_json", disk_full)
    with pytest.raises(ValueError, match="not admissible"):
        oq_verify.verify_written_format_checkpoint(output)
    assert "not admissible" in (output / "VERIFY-FAILED.txt").read_text()


def test_vlm_verification_uses_serving_load_contexts(tmp_path, monkeypatch):
    from contextlib import contextmanager, nullcontext

    import mlx.nn as nn

    import omlx.engine.vlm as engine_vlm
    from omlx import oq, oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    entered = []
    names = list(_VLM_CONTEXTS)

    for name in names:

        @contextmanager
        def context(path, name=name):
            entered.append((name, path))
            yield

        monkeypatch.setattr(engine_vlm, name, context)

    class Model(nn.Module):
        def __call__(self, x):
            return mx.zeros((1, 2, 4))

        def close(self):
            pass

    @contextmanager
    def patches(path, config, **kwargs):
        yield True

    monkeypatch.setattr(oq, "_calibration_patches", patches)
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr("mlx_vlm.utils.load_model", lambda *a, **k: Model())
    monkeypatch.setattr(oq, "check_loaded", lambda model: None)
    oq_verify.verify_written_format_checkpoint(output)
    assert entered == [(name, output) for name in names]


def _stub_vlm_verification(monkeypatch, model):
    from contextlib import contextmanager, nullcontext

    import omlx.engine.vlm as engine_vlm
    from omlx import oq

    for name in _VLM_CONTEXTS:
        monkeypatch.setattr(engine_vlm, name, lambda path: nullcontext())

    @contextmanager
    def patches(path, config, **kwargs):
        yield True

    monkeypatch.setattr(oq, "_calibration_patches", patches)
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr("mlx_vlm.utils.load_model", lambda *a, **k: model)
    monkeypatch.setattr(oq, "check_loaded", lambda model: None)


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(lambda: np.array([], dtype=np.int32), id="np"),
        pytest.param(lambda: mx.array([], dtype=mx.int32), id="mx"),
    ],
)
def test_refused_empty_prompt_leaves_staged_config_untouched(tmp_path, prompt):

    from omlx import oq_verify

    output = tmp_path / "staged"
    output.mkdir()
    config = output / "config.json"
    config.write_text(
        json.dumps(
            {
                "omlx_verification": {"verified": True, "mode": "toy"},
                "marker": 1,
            }
        )
    )
    before = config.read_bytes()
    with pytest.raises(ValueError, match="at least one token"):
        oq_verify.verify_written_format_checkpoint(output, prompt=prompt())
    assert config.read_bytes() == before
    assert not (output / "VERIFY-FAILED.txt").exists()


def test_empty_prompt_or_logits_fails_verification(tmp_path, monkeypatch):
    from contextlib import contextmanager, nullcontext

    import mlx.nn as nn

    import omlx.engine.vlm as engine_vlm
    from omlx import oq, oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))

    class Model(nn.Module):
        def __call__(self, x):
            return mx.zeros((1, 0, 4))

        def close(self):
            pass

    for name in _VLM_CONTEXTS:
        monkeypatch.setattr(engine_vlm, name, lambda path: nullcontext())

    @contextmanager
    def patches(path, config, **kwargs):
        yield True

    monkeypatch.setattr(oq, "_calibration_patches", patches)
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr("mlx_vlm.utils.load_model", lambda *a, **k: Model())
    monkeypatch.setattr(oq, "check_loaded", lambda model: None)
    with pytest.raises(ValueError, match="at least one token"):
        oq_verify.verify_written_format_checkpoint(
            output, prompt=np.array([], dtype=np.int32)
        )
    with pytest.raises(RuntimeError, match="empty logits"):
        oq_verify.verify_written_format_checkpoint(output)
    assert "empty logits" in (output / "VERIFY-FAILED.txt").read_text()


def test_writer_failure_uses_exception_type_when_message_empty(tmp_path):
    from omlx import oq_verify

    (tmp_path / "config.json").write_text(json.dumps({}))
    oq_verify._writer_failure(tmp_path, RuntimeError(), mode="toy")
    assert (tmp_path / "VERIFY-FAILED.txt").read_text().strip() == "RuntimeError"
    record = json.loads((tmp_path / "config.json").read_text())["omlx_verification"]
    assert record["reason"] == "writer verification failed: RuntimeError"
    whitespace = tmp_path / "whitespace"
    whitespace.mkdir()
    (whitespace / "config.json").write_text(json.dumps({}))
    oq_verify._writer_failure(whitespace, RuntimeError("   "), mode="toy")
    assert (whitespace / "VERIFY-FAILED.txt").read_text().strip() == "RuntimeError"


def test_mx_array_prompt_stays_on_mlx_thread(tmp_path, monkeypatch):
    import mlx.nn as nn

    from omlx import oq, oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    in_mlx = [False]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def __call__(self, x):
            self.seen = tuple(x.shape)
            return mx.zeros((1, 2, 4))

        def close(self):
            pass

    model = Model()
    _stub_vlm_verification(monkeypatch, model)
    real_asarray = oq_verify.np.asarray

    def guarded(a, *args, **kwargs):
        assert not isinstance(a, mx.array) or in_mlx[0]
        return real_asarray(a, *args, **kwargs)

    monkeypatch.setattr(oq_verify.np, "asarray", guarded)

    def on_mlx(run):
        in_mlx[0] = True
        try:
            return run()
        finally:
            in_mlx[0] = False

    monkeypatch.setattr(oq, "_on_mlx_thread", on_mlx)
    oq_verify.verify_written_format_checkpoint(
        output, prompt=mx.array([7], dtype=mx.int32)
    )
    assert model.seen == (1, 1)


def test_non_array_logits_names_the_type(tmp_path, monkeypatch):
    import mlx.nn as nn

    from omlx import oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))

    class Model(nn.Module):
        def __call__(self, x):
            return (mx.zeros((1, 2, 4)),)

        def close(self):
            pass

    _stub_vlm_verification(monkeypatch, Model())
    with pytest.raises(RuntimeError, match="tuple, not logits"):
        oq_verify.verify_written_format_checkpoint(output)
    assert "tuple, not logits" in (output / "VERIFY-FAILED.txt").read_text()


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(lambda: np.array(5, dtype=np.int32), id="0d"),
        pytest.param(lambda: np.ones((1, 1, 2), dtype=np.int32), id="3d"),
        pytest.param(lambda: mx.array(5, dtype=mx.int32), id="mx-0d"),
        pytest.param(lambda: mx.ones((1, 1, 2), dtype=mx.int32), id="mx-3d"),
    ],
)
def test_refused_bad_rank_prompt_leaves_config_untouched(tmp_path, monkeypatch, prompt):
    import mlx.nn as nn

    from omlx import oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    config = output / "config.json"
    config.write_text(
        json.dumps({"omlx_verification": {"verified": True, "mode": "toy"}})
    )
    before = config.read_bytes()

    calls = []

    class Model(nn.Module):
        def __call__(self, x):
            calls.append(True)
            return mx.zeros((1, 2, 4))

        def close(self):
            pass

    _stub_vlm_verification(monkeypatch, Model())
    with pytest.raises(ValueError, match="1-D or 2-D"):
        oq_verify.verify_written_format_checkpoint(output, prompt=prompt())
    assert config.read_bytes() == before
    assert not (output / "VERIFY-FAILED.txt").exists()
    assert calls == []


@pytest.mark.parametrize(
    ("prompt", "match"),
    [
        pytest.param(np.array([], dtype=np.int32), "at least one token", id="empty"),
        pytest.param(np.array(5, dtype=np.int32), "1-D or 2-D", id="0d"),
        pytest.param(np.ones((1, 1, 2), dtype=np.int32), "1-D or 2-D", id="3d"),
    ],
)
def test_empty_snapshot_prompt_fails_verification(tmp_path, monkeypatch, prompt, match):
    import mlx.nn as nn

    from omlx import oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    (output / "verify").mkdir()
    np.savez(output / "verify" / "reference.npz", prompt=prompt)

    class Model(nn.Module):
        def __call__(self, x):
            return mx.zeros((1, 2, 4))

        def close(self):
            pass

    _stub_vlm_verification(monkeypatch, Model())
    with pytest.raises(ValueError, match=match):
        oq_verify.verify_written_format_checkpoint(output)
    assert match in (output / "VERIFY-FAILED.txt").read_text()


def test_refused_empty_prompt_does_not_read_config(tmp_path):

    from omlx import oq_verify

    absent = tmp_path / "absent"
    with pytest.raises(ValueError, match="at least one token"):
        oq_verify.verify_written_format_checkpoint(
            absent, prompt=np.array([], dtype=np.int32)
        )
    assert not absent.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_logits_fails_verification(tmp_path, monkeypatch, value):
    import mlx.nn as nn

    from omlx import oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))

    class Model(nn.Module):
        def __call__(self, x):
            bad = np.zeros((2, 2, 4), dtype=np.float32)
            bad[1, 1, 3] = value
            return mx.array(bad)

        def close(self):
            pass

    _stub_vlm_verification(monkeypatch, Model())
    with pytest.raises(RuntimeError, match="non-finite logits"):
        oq_verify.verify_written_format_checkpoint(output)
    assert "non-finite logits" in (output / "VERIFY-FAILED.txt").read_text()
    record = json.loads((output / "config.json").read_text())["omlx_verification"]
    assert "non-finite logits" in record["reason"]
    assert record["verified"] is False


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(lambda: np.arange(6, dtype=np.int64).reshape(2, 3), id="np"),
        pytest.param(lambda: mx.array([[0, 1, 2], [3, 4, 5]], dtype=mx.int64), id="mx"),
    ],
)
def test_two_dimensional_caller_prompt_is_passed_through(tmp_path, monkeypatch, prompt):
    import mlx.nn as nn

    from omlx import oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def __call__(self, x):
            self.seen = (tuple(x.shape), x.dtype, np.array(x).tolist())
            return mx.zeros((1, 2, 4))

        def close(self):
            pass

    model = Model()
    _stub_vlm_verification(monkeypatch, model)
    oq_verify.verify_written_format_checkpoint(output, prompt=prompt())
    assert model.seen == ((2, 3), mx.int32, [[0, 1, 2], [3, 4, 5]])


def test_vlm_verification_uses_the_remapped_loader(tmp_path, monkeypatch):
    from contextlib import contextmanager, nullcontext

    import mlx.nn as nn
    import mlx_vlm.utils as vlm_utils

    import omlx.engine.vlm as engine_vlm
    from omlx import oq, oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    called = []

    class Model(nn.Module):
        def __call__(self, x):
            return mx.zeros((1, 2, 4))

        def close(self):
            pass

    def bypassed(*args, **kwargs):
        raise AssertionError("bypassed remap")

    @contextmanager
    def remap(path):
        original = vlm_utils.load_model

        def remapped(*args, **kwargs):
            called.append(True)
            return Model()

        vlm_utils.load_model = remapped
        try:
            yield
        finally:
            vlm_utils.load_model = original

    monkeypatch.setattr(engine_vlm, "_remap_nested_visual_on_load", remap)
    for name in [
        name for name in _VLM_CONTEXTS if name != "_remap_nested_visual_on_load"
    ]:
        monkeypatch.setattr(engine_vlm, name, lambda path: nullcontext())
    monkeypatch.setattr(vlm_utils, "load_model", bypassed)

    @contextmanager
    def patches(path, config, **kwargs):
        yield True

    monkeypatch.setattr(oq, "_calibration_patches", patches)
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr(oq, "check_loaded", lambda model: None)
    oq_verify.verify_written_format_checkpoint(output)
    assert called


@pytest.mark.parametrize("is_vlm", [True, False])
def test_loads_forward_trust_remote_code(tmp_path, monkeypatch, is_vlm):
    from contextlib import contextmanager, nullcontext

    import mlx.nn as nn

    import omlx.engine.vlm as engine_vlm
    import omlx.utils.model_loading as model_loading
    from omlx import oq, oq_verify

    output = tmp_path / "load"
    output.mkdir()
    model_type = "qwen4_exp" if is_vlm else "llama"
    (output / "config.json").write_text(json.dumps({"model_type": model_type}))
    seen = {}

    for name in _VLM_CONTEXTS:
        monkeypatch.setattr(engine_vlm, name, lambda path: nullcontext())

    class Model(nn.Module):
        def __call__(self, x):
            return mx.zeros((1, 2, 4))

        def close(self):
            pass

    @contextmanager
    def patches(path, config, **kwargs):
        yield is_vlm

    def vlm_load(*args, **kwargs):
        seen.update(kwargs)
        return Model()

    def lm_load(*args, **kwargs):
        seen.update(kwargs)
        return Model(), None

    monkeypatch.setattr(oq, "_calibration_patches", patches)
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr("mlx_vlm.utils.load_model", vlm_load)
    monkeypatch.setattr(model_loading, "lm_load_compat", lm_load)
    monkeypatch.setattr(oq, "check_loaded", lambda model: None)
    oq_verify.verify_written_format_checkpoint(output, trust_remote_code=True)
    assert seen.get("trust_remote_code") is True


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_writer_verification_streams_layer_by_layer(
    llama_source, toy, tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from omlx import oq_verify
    from omlx.formats.linear import FormatLinear

    output = tmp_path / "streamed"
    monkeypatch.setattr(oq, "_TEST_VERIFIED", frozenset({"llama"}))
    _calibration(monkeypatch)
    streamed = []
    original = oq_verify.layer_by_layer

    @contextmanager
    def watch(layers):
        streamed.append(True)
        try:
            with original(layers):
                yield
        finally:
            streamed.pop()

    monkeypatch.setattr(oq_verify, "layer_by_layer", watch)
    seen = []
    forward = FormatLinear.__call__

    def observe(layer, x):
        seen.append(bool(streamed))
        return forward(layer, x)

    monkeypatch.setattr(FormatLinear, "__call__", observe)
    oq.quantize_oq_streaming(
        str(llama_source),
        str(output),
        4,
        enhanced=True,
        quant_format=toy.mode,
        **_SETTINGS,
    )
    assert seen, "the writer never ran the staged format layers"
    assert all(seen), "a format layer ran outside layer_by_layer"


def test_vlm_verification_injects_declared_optiq_sidecar(tmp_path, monkeypatch):
    from contextlib import contextmanager, nullcontext

    import mlx.nn as nn

    import omlx.engine.vlm as engine_vlm
    from omlx import oq, oq_verify

    output = tmp_path / "vlm"
    output.mkdir()
    (output / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "optiq_vision": {"sidecar": "optiq/x.safetensors"},
            }
        )
    )
    (output / "optiq").mkdir()
    mx.save_safetensors(
        str(output / "optiq" / "x.safetensors"),
        {"vision_tower.w": mx.ones((2,))},
    )
    seen = {}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_w = mx.zeros((1,))
            self.vision_tower = nn.Module()
            self.vision_tower.w = mx.zeros((2,))

        def __call__(self, x):
            return mx.zeros((1, 2, 4))

        def close(self):
            pass

    def load(*args, **kwargs):
        model = Model()
        seen["model"] = model
        model.load_weights([("lm_w", mx.zeros((1,)))], strict=True)
        return model

    for name in _VLM_CONTEXTS[:-1]:
        monkeypatch.setattr(engine_vlm, name, lambda path: nullcontext())

    @contextmanager
    def patches(path, config, **kwargs):
        yield True

    monkeypatch.setattr(oq, "_calibration_patches", patches)
    monkeypatch.setattr(oq, "_load_globals_kept", lambda config: nullcontext())
    monkeypatch.setattr(oq, "_on_mlx_thread", lambda run: run())
    monkeypatch.setattr("mlx_vlm.utils.load_model", load)
    monkeypatch.setattr(oq, "check_loaded", lambda model: None)
    oq_verify.verify_written_format_checkpoint(output)
    assert seen["model"].vision_tower.w.tolist() == [1.0, 1.0]
