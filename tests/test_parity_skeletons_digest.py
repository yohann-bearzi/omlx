# SPDX-License-Identifier: Apache-2.0
"""The parity skeletons' emitted Metal, pinned by digest against the released
kernels. CPU only: no GPU, no MLX (the reference builder is extracted from
the released source and run with a recording fake).

The row-exact skeleton's affine specs (below) are the per-shape kernels'
text, byte for byte (specs/parity-skeletons.md §3.1); its reference is
extracted from the release-3 commit eae63b6c itself, so the check compares
against the shipped generator, not a copy of it.

The verify (MMA) skeleton's 108 specs (after it) are compared with the
released builder the same way.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from omlx.formats import row_exact_skeleton, tuning

REF = "eae63b6c"


def _show(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{REF}:omlx/{path}.py"], text=True
    )


def _extract(source: str, names: set, env: dict) -> dict:
    tree = ast.parse(source)
    items = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.FunctionDef):
            name = node.name
        else:
            continue
        if name in names:
            items.append(node)
    exec(compile(ast.Module(items, type_ignores=[]), "<pinned generator>", "exec"), env)
    return env


def _pinned():
    source = _show("formats/affine")
    tree = ast.parse(source)
    end = next(
        n.end_lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decoded"
    )
    names = {
        n.targets[0].id
        for n in tree.body
        if isinstance(n, ast.Assign) and n.lineno < end
    }
    affine_env = _extract(source, names | {"header", "decoded"}, {})
    from functools import cache
    from textwrap import indent
    from types import SimpleNamespace as NS

    impl = _show("patches/row_exact_qmv")
    env = _extract(
        impl,
        {
            "_BITS",
            "_TILE",
            "_BLOCK",
            "_LAST_BLOCK",
            "_SOURCE",
            "_group_source",
            "_header",
            "_kernel",
            "_group_kernel",
        },
        dict(
            **{k: affine_env[k] for k in ("header", "decoded")},
            cache=cache,
            indent=indent,
            mx=NS(fast=NS(metal_kernel=lambda **kw: dict(kw))),
        ),
    )
    return affine_env, env


def _specs(affine_env):
    specs = []
    for bits in (4, 5, 6, 8):
        for group_size in (32, 64, 128):
            for fast in (False, True):
                ops = row_exact_skeleton.affine_row_ops(bits, group_size, fast)
                for unrolled in (False, True):
                    for count in (0, 1, 2, 3, 4):
                        spec = row_exact_skeleton.kernel_spec(
                            ops, bits, group_size, fast, unrolled, count
                        )
                        specs.append((bits, group_size, fast, unrolled, count, spec))
    return specs


def test_a13_affine_240_specs_byte_identical_to_eae63b6c():
    """240/240 (bits, group, fast, unroll, projections) specs are exactly the
    pinned release-3 builders'."""
    affine_env, pinned = _pinned()
    digest = hashlib.sha256()
    mismatches = []
    for bits, group_size, fast, unrolled, count, spec in _specs(affine_env):
        want = (
            pinned["_group_kernel"](bits, group_size, fast, count, unrolled)
            if count
            else pinned["_kernel"](bits, group_size, fast, unrolled)
        )
        if spec != want:
            mismatches.append((bits, group_size, fast, unrolled, count))
        for key in ("name", "header", "source"):
            digest.update(spec[key].encode() + b"\0")
        digest.update(",".join(spec["input_names"]).encode() + b"\0")
        digest.update(",".join(spec["output_names"]).encode() + b"\0")
    assert not mismatches, mismatches[:4]
    assert len(_specs(affine_env)) == 240
    print("A13 parity digest", digest.hexdigest())


def test_qmv_shared_unrolls_by_pragma_not_fold_expression():
    """The shared skeleton's unrolled block is the looped one under a full
    pragma (SROA/LICM run first); the fold-expression form is gone."""
    from omlx.formats import kernels

    assert not hasattr(kernels, "_SHARED_FOLDED")
    unrolled = kernels._qmv_shared(True)
    assert '_Pragma("clang loop unroll(full)")' in unrolled
    assert "omlx_shared_folded" not in unrolled


def test_row_index_width_is_chosen_at_construction():
    assert tuning.row_index_width(2560, 6144) is tuning.RowIndexWidth.INT32_CHECKED
    assert tuning.row_index_width(2**28, 16) is tuning.RowIndexWidth.SIZE_T64


# The verify (MMA) skeleton, its released builder recorded.

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
