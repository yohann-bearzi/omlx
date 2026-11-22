# SPDX-License-Identifier: Apache-2.0
"""The parity skeletons' emitted Metal, pinned by digest against the released
kernels. CPU only: no GPU, no MLX (the reference builder is extracted from
the released source and run with a recording fake)."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "verify_skeleton_under_test", ROOT / "omlx" / "formats" / "verify_skeleton.py"
)
verify_skeleton = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = verify_skeleton
_spec.loader.exec_module(verify_skeleton)
RELEASE = "eae63b6c"
U34D_SHA256 = "a88dd195628a41c626754943c6245fc7cd9a4703ae61e1daa06c2f76a32b9e3a"


def _released_mma_builder():
    try:
        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{RELEASE}:omlx/patches/qwen35_verify_qmm.py",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the reference release source needs a git checkout")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_mma_kernel"
    )
    function.body = [s for s in function.body if not isinstance(s, ast.Import)]
    fake_mx = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        fast=SimpleNamespace(metal_kernel=lambda **kw: kw),
    )
    environment = {
        "_KERNEL_CACHE": {},
        "_MMA_BN": 16,
        "_MMA_BK": 32,
        "_MMA_NSG": 8,
        "mx": fake_mx,
    }
    exec(
        compile(ast.Module([function], type_ignores=[]), "<release>", "exec"),
        environment,
    )
    return environment["_build_mma_kernel"], fake_mx


def test_u34d_affine_108_mma_specs_byte_identical_to_the_release():
    builder, fake_mx = _released_mma_builder()
    digest = hashlib.sha256()
    count = 0
    for bits in (2, 3, 4, 5, 6, 8):
        for group in (32, 64, 128):
            ops = verify_skeleton.affine_mma_ops(bits, group)
            for row_tiles in (1, 2, 3):
                for dtype, tag in (
                    (fake_mx.bfloat16, "bf16"),
                    (fake_mx.float16, "fp16"),
                ):
                    name = f"omlx_vk_mma_rt{row_tiles}_k2560_q{bits}_gs{group}_{tag}"
                    got = verify_skeleton.emit_mma_kernel(ops, row_tiles, 2560, name)
                    want = builder(row_tiles, 2560, group, dtype, bits)
                    assert got == want, (bits, group, row_tiles, tag)
                    for part in (got["name"], got.get("header", ""), got["source"]):
                        digest.update(part.encode() + b"\0")
                    count += 1
    assert count == 108
    assert digest.hexdigest() == U34D_SHA256


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_the_mma_skeleton_has_no_reserved_keyword(bits):
    """8-bit once failed to compile on `int half` (Metal's half type)."""
    ops = verify_skeleton.affine_mma_ops(bits, 64)
    source = verify_skeleton.emit_mma_kernel(ops, 1, 2560, "s")["source"]
    assert "int half" not in source
    assert "for (int half" not in source


def test_mma_tile_ops_fields_are_all_used():
    import inspect

    source = inspect.getsource(verify_skeleton.emit_mma_source) + inspect.getsource(
        verify_skeleton.emit_mma_kernel
    )
    for field in verify_skeleton.MmaTileOps.__dataclass_fields__:
        assert field in source, field


def test_tile_mma_ops_keeps_the_contract_check():
    class Face:
        type = "F"

    source = verify_skeleton.emit_mma_source(
        verify_skeleton.tile_mma_ops(Face()), 1, 2560
    )
    assert "if constexpr (omlx_tile_face<F, K>::value)" in source
    assert "omlx_mma_dep<T, F>" in source


def test_tile_mma_ops_uses_emit_k_and_no_dead_n_global():
    class Face:
        type = "F"

    source = verify_skeleton.emit_mma_source(
        verify_skeleton.tile_mma_ops(Face()), 1, 5120
    )
    assert "constexpr int K = 5120;" in source
    assert "n_global" not in source


def test_emit_mma_source_does_not_rewrite_face_type():
    class Face:
        type = "omlx_foo__K__::Tile"

    source = verify_skeleton.emit_mma_source(
        verify_skeleton.tile_mma_ops(Face()), 1, 5120
    )
    assert "omlx_foo__K__::Tile" in source
    assert "constexpr int K = 5120;" in source
