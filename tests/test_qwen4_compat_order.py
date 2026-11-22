# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp's vendor must be installed before anything imports the family."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_patch_after_the_installed_module_fails_closed():
    """A process that imported the installed qwen4_exp before the patch must
    be refused loudly: two module copies cannot coexist silently."""
    code = """
import sys
import mlx_vlm.models.qwen4_exp as installed
from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

assert "site-packages" in installed.__file__, installed.__file__
try:
    apply_mlx_vlm_qwen4_exp_compat_patch()
except RuntimeError as error:
    message = str(error)
    assert "before oMLX's qwen4_exp compat patch" in message, error
    assert installed.__file__ in message, error
    assert "maybe_apply_pre_load_patches" in message, error
    assert sys.modules["mlx_vlm.models.qwen4_exp"] is installed
    print("FAILED_CLOSED")
else:
    raise SystemExit("the patch replaced an imported module silently")
"""
    run = _run(code)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "FAILED_CLOSED" in run.stdout


def test_the_pre_load_installs_the_vendor_for_a_cluster_call(tmp_path):
    """The cluster's capability detection and the worker pre-load call
    maybe_apply_pre_load_patches without for_vlm; the vendor must already
    be installed then, before anything imports the family module."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    code = f"""
from omlx.utils.model_loading import maybe_apply_pre_load_patches

maybe_apply_pre_load_patches({str(tmp_path)!r})
import mlx_vlm.models.qwen4_exp as module

assert "vendor" in module.__file__, module.__file__
import mlx_vlm.models.qwen4_exp.language as language

assert callable(language.configure_mtp_runtime), language.__file__
print("VENDOR_FIRST")
"""
    run = _run(code)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "VENDOR_FIRST" in run.stdout


def test_foreign_submodule_and_no_file_fail_closed_without_evicting(
    monkeypatch, tmp_path
):
    """A foreign submodule alone or a module with __file__ is None fails closed
    and is never evicted from sys.modules."""
    import types

    import pytest

    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

    monkeypatch.setattr(compat, "_APPLIED", False)
    for key in [
        name
        for name in list(sys.modules)
        if name == "mlx_vlm.models.qwen4_exp"
        or name.startswith("mlx_vlm.models.qwen4_exp.")
    ]:
        monkeypatch.delitem(sys.modules, key, raising=False)

    foreign_path = str(tmp_path / "foreign" / "language.py")
    sub = types.ModuleType("mlx_vlm.models.qwen4_exp.language")
    sub.__file__ = foreign_path
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen4_exp.language", sub)
    with pytest.raises(RuntimeError) as excinfo:
        compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    assert "mlx_vlm.models.qwen4_exp.language" in str(excinfo.value)
    assert foreign_path in str(excinfo.value)
    assert sys.modules["mlx_vlm.models.qwen4_exp.language"] is sub

    monkeypatch.delitem(sys.modules, "mlx_vlm.models.qwen4_exp.language")
    no_file = types.ModuleType("mlx_vlm.models.qwen4_exp")
    no_file.__file__ = None
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen4_exp", no_file)
    with pytest.raises(RuntimeError) as excinfo:
        compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    assert "mlx_vlm.models.qwen4_exp" in str(excinfo.value)
    assert sys.modules["mlx_vlm.models.qwen4_exp"] is no_file


def test_vendor_module_already_loaded_is_accepted():
    """An already-loaded vendor module with _APPLIED=False succeeds instead of
    being mistaken for a foreign shadow."""
    code = """
import mlx_vlm
import mlx_vlm.models
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat._append_package_path(mlx_vlm, compat._VENDOR_MLX_VLM)
compat._append_package_path(mlx_vlm.models, compat._VENDOR_MLX_VLM / "models")
import mlx_vlm.models.qwen4_exp as vendor_mod

assert compat._VENDOR_MLX_VLM == compat._VENDOR_MLX_VLM.resolve()
assert compat._VENDOR_MLX_VLM in type(compat._VENDOR_MLX_VLM)(
    vendor_mod.__file__
).resolve().parents
assert not compat.is_applied()
assert compat.apply_mlx_vlm_qwen4_exp_compat_patch() is True
print("VENDOR_ACCEPTED")
"""
    run = _run(code)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "VENDOR_ACCEPTED" in run.stdout


def test_non_qwen4_pre_load_does_not_install_vendor(tmp_path):
    """A non-qwen4_exp checkpoint must not install the qwen4_exp vendor."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    code = f"""
import sys
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.utils.model_loading import maybe_apply_pre_load_patches

maybe_apply_pre_load_patches({str(tmp_path)!r})
assert not compat.is_applied()
assert "mlx_vlm.models.qwen4_exp" not in sys.modules
print("NON_QWEN4_OK")
"""
    run = _run(code)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "NON_QWEN4_OK" in run.stdout


def test_apply_snapshots_sys_modules_and_serializes_threads(monkeypatch):
    """sys.modules is snapshotted once (concurrent mutation does not raise) and
    concurrent callers serialize so the patch installs exactly once."""
    import importlib
    import threading
    import time
    import types

    import mlx_vlm  # noqa: F401
    import mlx_vlm.models  # noqa: F401
    from mlx_vlm.models.qwen3_5 import language as _qwen35_language  # noqa: F401

    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
    from omlx.patches.mlx_vlm_mtp import (
        qwen35_verify_attention,
        qwen35_verify_linear,
    )

    # 1. Deterministic snapshot check: a key whose __eq__ mutates sys.modules
    # during live dict iteration triggers RuntimeError unless snapshotted first.
    class _MutatingKey(str):
        def __eq__(self, other):
            fake_modules["_omlx_test_injected_key"] = None
            return str.__eq__(self, other)

        def __hash__(self):
            return str.__hash__(self)

    fake_modules = {
        k: v
        for k, v in sys.modules.items()
        if k != "mlx_vlm.models.qwen4_exp"
        and not k.startswith("mlx_vlm.models.qwen4_exp.")
    }
    fake_modules[_MutatingKey("_omlx_test_trigger_key")] = types.ModuleType("trigger")
    fake_sys = types.SimpleNamespace(modules=fake_modules)

    real_import_module = importlib.import_module
    import_calls = 0
    calls_lock = threading.Lock()
    entered = threading.Event()

    def slow_import(name, package=None):
        nonlocal import_calls
        if name != "mlx_vlm.models.qwen4_exp":
            return real_import_module(name, package)
        with calls_lock:
            import_calls += 1
        entered.set()
        time.sleep(0.05)
        return types.ModuleType(name)

    monkeypatch.setattr(compat, "_APPLIED", False)
    monkeypatch.setattr(compat, "sys", fake_sys)
    monkeypatch.setattr(compat.importlib, "import_module", slow_import)
    monkeypatch.setattr(compat, "_patch_prompt_utils", lambda: None)
    monkeypatch.setattr(compat, "_patch_prompt_loop", lambda: None)
    monkeypatch.setattr(compat, "_patch_model_file_bypass", lambda: None)
    monkeypatch.setattr(qwen35_verify_linear, "apply", lambda: None)
    monkeypatch.setattr(qwen35_verify_attention, "apply", lambda _lang: None)

    results: list[bool] = []
    errors: list[BaseException] = []

    def worker(wait_first: bool) -> None:
        if wait_first:
            entered.wait(timeout=2.0)
        try:
            results.append(compat.apply_mlx_vlm_qwen4_exp_compat_patch())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            if not wait_first:
                entered.set()

    t1 = threading.Thread(target=worker, args=(False,))
    t2 = threading.Thread(target=worker, args=(True,))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert not errors, errors
    assert import_calls == 1
    assert sorted(results) == [False, True]
