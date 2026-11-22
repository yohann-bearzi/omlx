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
import mlx_vlm.models.qwen4_exp as installed
from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

assert "site-packages" in installed.__file__, installed.__file__
try:
    apply_mlx_vlm_qwen4_exp_compat_patch()
except RuntimeError as error:
    assert "before oMLX's qwen4_exp compat patch" in str(error), error
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
