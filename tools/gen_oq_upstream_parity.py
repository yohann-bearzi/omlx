# SPDX-License-Identifier: Apache-2.0
"""Pin upstream e1a402cf's stock oQe/oQh writer outputs as chip-keyed goldens.

Reads upstream's package from this repository's own git history (no network,
no environment variables), runs the same tiny builds and calibration tokens the
parity test uses under the same MLX_ENABLE_TF32=0 child environment, and merges
this GPU architecture's raw stored-byte digests (dtype, shape, sha256) plus
config and non-volatile report fields into tests/data/oq_upstream_parity.json.

Run only when upstream is re-merged or a new Apple GPU architecture is added.
Run on a Mac with Metal:
    python tools/gen_oq_upstream_parity.py [upstream_dir] [--out path]
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data" / "oq_upstream_parity.json"

sys.path.insert(0, str(REPO))
from tests.test_oq_upstream_parity import (  # noqa: E402
    UPSTREAM,
    _build,
    _llama_source,
    _qwen3_5_moe_source,
    _snapshot,
    _stripped,
)


def _archive(destination: Path) -> None:
    tar = subprocess.run(
        ["git", "-C", str(REPO), "archive", "--format=tar", UPSTREAM, "omlx"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        archive.extractall(destination, filter="data")


def _verify_upstream_tree(tree: Path) -> None:
    oq_source = (tree / "omlx" / "oq.py").read_text()
    assert "def _refine_oqe_output(" in oq_source
    assert "def _write_oqh_output(" in oq_source
    assert "quant_format" not in oq_source


def _merge_architectures(
    loaded: dict, arch: str, cases: dict, mlx_version: str
) -> dict:
    """This architecture's cases plus the preserved ones under the same numerics.

    An architecture recorded under another upstream or another MLX version is
    dropped, not relabelled: its raw bytes were produced by other code. The
    preserved reports are stripped of their run's volatile facts (the test's
    comparison strips both sides anyway).
    """
    if loaded.get("upstream") != UPSTREAM:
        return {arch: cases}
    if loaded.get("numerics") != {"tf32": "off", "mlx": mlx_version}:
        return {arch: cases}
    existing = {
        name: {
            case: {**data, "report": _stripped(data["report"])}
            for case, data in case_map.items()
        }
        for name, case_map in (loaded.get("architectures") or {}).items()
    }
    existing[arch] = cases
    return existing


def _case(
    tree: Path,
    source: Path,
    output: Path,
    level: int,
    work: Path,
    *,
    hadamard: bool = False,
    preserve_mtp: bool = False,
) -> dict:
    _build(
        tree,
        source,
        output,
        level,
        work,
        hadamard=hadamard,
        preserve_mtp=preserve_mtp,
    )
    return _snapshot(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "upstream_dir",
        nargs="?",
        default=None,
        help="Optional directory containing upstream e1a402cf's omlx/ package "
        "(used on a Studio without .git).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA,
        help="Destination JSON path (defaults to tests/data/oq_upstream_parity.json).",
    )
    args = parser.parse_args()

    arch = str(mx.device_info().get("architecture", ""))
    assert arch, "failed to query MLX GPU architecture"

    cases = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tree = root / "upstream"
        if args.upstream_dir:
            shutil.copytree(Path(args.upstream_dir) / "omlx", tree / "omlx")
        else:
            tree.mkdir()
            _archive(tree)
        _verify_upstream_tree(tree)

        # Tiny Llama oQ4e / oQ3e
        llama_src = root / "llama-source"
        _llama_source(llama_src)
        for level in (4, 3):
            work = root / f"llama-{level}e"
            work.mkdir()
            cases[f"llama-oQ{level}e"] = _case(
                tree, llama_src, work / "out", level, work
            )

        # Tiny Qwen3.5 MoE (routed experts + shared expert) oQ4e / oQ3e / oQ4h / oQ3h
        moe_src = root / "qwen3_5_moe-source"
        with pytest.MonkeyPatch.context() as patch:
            _qwen3_5_moe_source(moe_src, patch, mtp=False)
        for level in (4, 3):
            work = root / f"qwen3_5_moe-{level}e"
            work.mkdir()
            cases[f"qwen3_5_moe-oQ{level}e"] = _case(
                tree, moe_src, work / "out", level, work
            )
        for level in (4, 3):
            work = root / f"qwen3_5_moe-{level}h"
            work.mkdir()
            cases[f"qwen3_5_moe-oQ{level}h"] = _case(
                tree, moe_src, work / "out", level, work, hadamard=True
            )

        # Tiny Qwen3.5 MoE + MTP (routed experts, shared expert, MTP) oQ4e / oQ3e / oQ2e
        mtp_src = root / "qwen3_5_moe_mtp-source"
        with pytest.MonkeyPatch.context() as patch:
            _qwen3_5_moe_source(mtp_src, patch, mtp=True)
        for level in (4, 3, 2):
            work = root / f"qwen3_5_moe_mtp-{level}e"
            work.mkdir()
            cases[f"qwen3_5_moe_mtp-oQ{level}e"] = _case(
                tree, mtp_src, work / "out", level, work, preserve_mtp=True
            )

    loaded = json.loads(args.out.read_text()) if args.out.exists() else {}
    existing = _merge_architectures(loaded, arch, cases, mx.__version__)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "upstream": UPSTREAM,
                # The numerics every golden was produced under; the test pins
                # the same for its own builds and asserts this record.
                "numerics": {"tf32": "off", "mlx": mx.__version__},
                "architectures": existing,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {args.out} ({arch}: {len(cases)} cases, total archs={len(existing)})")


if __name__ == "__main__":
    main()
