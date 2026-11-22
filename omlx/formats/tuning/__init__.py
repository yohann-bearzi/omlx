# SPDX-License-Identifier: Apache-2.0
"""Compile-time skeleton tuning: the public names and the parity defaults.

The dataclasses and hook enums live in ``schema``; the tables here are the
parity floor (code, not tuned data). The offline autotuner imports these
names and populates the tables per chip; an empty table keeps the geometry
rule (``kernels.shared_geometry``), which reproduces the pre-skeleton
builders' choices.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import mlx.core as mx

from ...utils import hardware
from .. import kernels
from . import schema
from .schema import (
    DenseTilePolicy,
    FetchGranularity,
    LoadLifetime,
    ParamStaging,
    RowExactPolicy,
    RowIndexWidth,
    UnrollPolicy,
)

__all__ = [
    "DENSE_TILE_TABLE",
    "ROW_EXACT_TABLE",
    "ChipTable",
    "DenseTilePolicy",
    "FetchGranularity",
    "LoadLifetime",
    "ParamStaging",
    "RowExactPolicy",
    "RowIndexWidth",
    "UnrollPolicy",
    "check_repin",
    "current_chip",
    "face_digest",
    "lookup",
    "register_skeleton",
    "register_tuning_path",
    "reset_cache",
    "row_exact_policy",
    "row_index_width",
    "rows_bucket",
    "skeleton_digest",
    "stale_entries",
    "table_profile",
]


def row_index_width(n: int, row_bytes: int) -> RowIndexWidth:
    """The row skeleton's index width: int when construction proves the
    largest offset under 2**31, size_t otherwise (owner rule 28: decided at
    construction)."""
    return (
        RowIndexWidth.INT32_CHECKED if n * row_bytes < 2**31 else RowIndexWidth.SIZE_T64
    )


def row_exact_policy(
    op_tag: str, rows: int, bits: int, k: int, n: int, chip_gen: int
) -> RowExactPolicy | None:
    """Look up a chip-tuned ``RowExactPolicy``, or None for the parity floor."""
    if not ROW_EXACT_TABLE:
        return None
    return ROW_EXACT_TABLE.get((op_tag, rows, bits, k, n, chip_gen))


# (op_tag, rows, bits, k, n, chip_gen) -> RowExactPolicy. The parity floor:
# empty until the autotuner fills it.
ROW_EXACT_TABLE: dict = {}

# (chip_gen, bits, group_size, m bucket, n bucket, k bucket) -> DenseTilePolicy.
DENSE_TILE_TABLE: dict = {}


# --- The per-chip DATA tables -------------------------------------------------

_V1 = Path(__file__).with_name("v1")
_TABLE_DIRS = [_V1]
_EXTRA_DIRS: list[Path] = []
_CHIPS: dict[str, ChipTable] = {}
_SKELETON_DIGEST: dict[str, str] = {}
_LOCK = threading.Lock()

_ROWS_BUCKETS = {
    "dense_tile": (
        (17, 24, "17-24"),
        (25, 32, "25-32"),
        (33, 48, "33-48"),
        (49, 64, "49-64"),
        (65, 128, "65-128"),
        (129, 256, "129-256"),
        (257, 512, "257-512"),
        (513, 1024, "513-1024"),
        (1025, 2048, "1025-2048"),
    ),
    "dense_qmv": (
        (1, 1, "1"),
        (2, 2, "2"),
        (3, 3, "3"),
        (4, 4, "4"),
        (5, 5, "5"),
        (6, 6, "6"),
        (7, 8, "7-8"),
        (9, 12, "9-12"),
        (13, 16, "13-16"),
        (17, 24, "17-24"),
    ),
}


@cache
def current_chip() -> str:
    """This GPU as `<architecture>_c<cores>`, the tuning tables' file name.

    Fail closed: without Metal, or when the core count cannot be read, no
    table is applied (an unknown chip is the parity default)."""
    if not mx.is_available(mx.gpu):
        return "unknown_c0"
    info = mx.device_info(mx.gpu)
    arch = (info.get("architecture") if isinstance(info, dict) else None) or "unknown"
    return f"{arch}_c{hardware.get_gpu_core_count() or 0}"


def register_skeleton(route: str, text: str) -> None:
    """The route's current skeleton template source, for staleness checks."""
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    with _LOCK:
        _SKELETON_DIGEST[route] = digest


def skeleton_digest(route: str) -> str | None:
    """The hash of the skeleton template the route emits today."""
    with _LOCK:
        return _SKELETON_DIGEST.get(route)


def reset_cache() -> None:
    """Drop the loaded tables (a test or a newly registered path only)."""
    with _LOCK:
        _CHIPS.clear()
    clear = getattr(current_chip, "cache_clear", None)
    if clear is not None:
        clear()


def table_profile(face) -> str:
    """The table class a face falls in: plain, a small or a large LUT."""
    size = kernels.table_bytes(face)
    if size == 0:
        return "plain"
    return "lut_small" if size <= 256 else "lut_tg"


def face_digest(face) -> str:
    """A face's identity for a tuning entry: what its Metal is built from."""
    source = getattr(face, "emitted_source", None) or face.source
    shared = getattr(face, "emitted_shared", None) or getattr(face, "shared", "")
    tables = tuple(
        (table.name, str(table.dtype), tuple(table.values)) for table in face.tables
    )
    material = repr(
        (
            type(face).__name__,
            face.key,
            face.type,
            getattr(face, "vpt", 0),
            source,
            shared,
            face.tensors,
            tables,
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def _n_bucket(n: int) -> str:
    if n <= 1024:
        return "N<=1024"
    if n <= 4096:
        return "N1025-4096"
    if n <= 32768:
        return "N4097-32768"
    return "N>32768"


def _k_bucket(k: int) -> str:
    if k <= 1024:
        return "K<=1024"
    if k <= 4096:
        return "K1025-4096"
    return "K>4096"


def rows_bucket(route: str, rows: int) -> str:
    """The route's deterministic row bucket for ``rows``."""
    buckets = _ROWS_BUCKETS.get(route)
    if buckets is None:
        return str(rows)
    for low, high, name in buckets:
        if low <= rows <= high:
            return name
    if rows < buckets[0][0]:
        return f"<{buckets[0][0]}"
    return f"{buckets[-1][1] + 1}+"


def _dtype_name(dtype) -> str:
    """Normalize an ``mx.Dtype`` or string to ``bf16``/``fp16``/``fp32``."""
    text = str(getattr(dtype, "name", dtype)).removeprefix("mlx.core.")
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(text, text)


@dataclass(frozen=True)
class _Entry:
    key: dict
    face_sha256: str
    skeleton_sha256: str
    measured_at: str
    params: object
    exactness: str


@dataclass(frozen=True)
class ChipTable:
    entries: tuple[_Entry, ...]


def _read_table(path: Path, chips: dict[str, list[_Entry]]) -> None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"tuning: cannot read {path}: {error}") from None
    version = data.get("schema_version") if isinstance(data, dict) else data
    if not isinstance(data, dict) or isinstance(version, bool) or version != 1:
        raise ValueError(f"tuning: {path} is a schema_version 1 table, not {version!r}")
    chip = data.get("chip")
    if not isinstance(chip, str) or not chip:
        raise ValueError(f"tuning: {path} names no chip")
    if path.stem != chip:
        raise ValueError(f"tuning: {path} chip {chip!r} != {path.stem!r}")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"tuning: {path} has no entries list")
    parsed = chips.setdefault(chip, [])
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError(f"tuning: {path} has a non-object entry")
        key = raw.get("key")
        exactness = raw.get("exactness", "bitwise")
        if not isinstance(key, dict) or exactness not in ("bitwise", "rule17_fp32"):
            raise ValueError(f"tuning: {path} has an entry without a key/exactness")
        face_id = key.get("face_id")
        if not isinstance(face_id, str) or not face_id:
            raise ValueError(f"tuning: {path} has an entry without face_id")
        face_sha = raw.get("face_sha256", "")
        exact_face = not face_id.startswith("*:")
        if not isinstance(face_sha, str) or (exact_face and not face_sha):
            raise ValueError(f"tuning: {path} has an entry without face_sha256")
        skeleton = raw.get("skeleton_sha256")
        measured = raw.get("measured_at")
        if not isinstance(skeleton, str) or not skeleton:
            raise ValueError(f"tuning: {path} has an entry without skeleton_sha256")
        if not isinstance(measured, str) or not measured:
            raise ValueError(f"tuning: {path} has an entry without measured_at")
        params = schema.params_from_json(key.get("route"), raw.get("params"))
        parsed.append(
            _Entry(dict(key), face_sha, skeleton, measured, params, exactness)
        )


def _index_for_chip(chip: str) -> ChipTable:
    with _LOCK:
        table = _CHIPS.get(chip)
        if table is not None:
            return table
        chips: dict[str, list[_Entry]] = {}
        for directory in (*_TABLE_DIRS, *_EXTRA_DIRS):
            base = Path(directory)
            path = base if base.is_file() else base / f"{chip}.json"
            if path.is_file() and path.stem == chip:
                _read_table(path, chips)
        table = ChipTable(tuple(chips.get(chip, ())))
        _CHIPS[chip] = table
        return table


def register_tuning_path(path) -> None:
    """Add a plugin's table directory (or one table file) below the in-tree
    tables."""
    directory = Path(path)
    if directory.is_dir():
        files = sorted(directory.glob("*.json"))
    elif directory.is_file():
        files = [directory]
    else:
        raise ValueError(f"tuning: {path} is not a table or a directory of tables")
    if not files:
        raise ValueError(f"tuning: {path} holds no table")
    chips: dict[str, list[_Entry]] = {}
    for table in files:
        _read_table(table, chips)
    with _LOCK:
        if directory not in _EXTRA_DIRS:
            _EXTRA_DIRS.append(directory)
        _CHIPS.clear()


@dataclass(frozen=True)
class _Query:
    """What every entry is compared with, computed once per lookup."""

    route: str
    face_key: str
    profile: str
    dtype: str
    bits: int
    group: int
    n: int
    k: int
    rows_bucket: str
    n_bucket: str
    k_bucket: str


def _matches(entry: _Entry, query: _Query) -> int:
    key = entry.key
    if (
        key.get("route") != query.route
        or key.get("dtype") != query.dtype
        or key.get("bits") != query.bits
        or key.get("group") != query.group
        or key.get("rows_bucket") != query.rows_bucket
    ):
        return 0
    face_id = key.get("face_id")
    exact_face = face_id == query.face_key
    if not exact_face and face_id != f"*:{query.profile}":
        return 0
    if "n" in key and "k" in key:
        if key["n"] != query.n or key["k"] != query.k:
            return 0
        return 3 if exact_face else 1
    if key.get("n_bucket") != query.n_bucket or key.get("k_bucket") != query.k_bucket:
        return 0
    return 2 if exact_face else 1


def lookup(
    route: str,
    face,
    *,
    dtype,
    bits: int,
    group: int,
    n: int,
    k: int,
    rows: int,
    chip: str | None = None,
) -> object | None:
    """The tuned parameters for a face and shape, or None for the parity
    default.  Resolve once per launch plan, never per call."""
    chip_id = chip or current_chip()
    table = _index_for_chip(chip_id)
    if not table.entries:
        return None
    skeleton = skeleton_digest(route)
    if skeleton is None:
        return None
    query = _Query(
        route=route,
        face_key=face.key,
        profile=table_profile(face),
        dtype=_dtype_name(dtype),
        bits=bits,
        group=group,
        n=n,
        k=k,
        rows_bucket=rows_bucket(route, rows),
        n_bucket=_n_bucket(n),
        k_bucket=_k_bucket(k),
    )
    digest: str | None = None
    candidates: list[tuple[int, _Entry]] = []
    for entry in table.entries:
        score = _matches(entry, query)
        if not score:
            continue
        if entry.skeleton_sha256 != skeleton:
            continue
        if entry.key.get("face_id") == face.key:
            if digest is None:
                digest = face_digest(face)
            if entry.face_sha256 != digest:
                continue
        if not entry.params.is_legal(face, k, n, rows, chip_id, dtype=dtype):
            continue
        candidates.append((score, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1].params


def stale_entries(dirs=None, faces=None) -> list[str]:
    """Every entry whose face or skeleton hash no longer matches its source.

    CI runs this over the in-tree tables (and a plugin's own); a stale entry
    is ignored by ``lookup`` and must be re-tuned or deleted by the commit
    that changed the face or skeleton.
    """
    faces = faces or {}
    stale = []
    face_digests: dict[str, str] = {}
    skeleton_digests: dict[str, str | None] = {}
    for entry_path in _TABLE_DIRS if dirs is None else dirs:
        base = Path(entry_path)
        for path in [base] if base.is_file() else sorted(base.glob("*.json")):
            chips: dict[str, list[_Entry]] = {}
            _read_table(path, chips)
            for entries in chips.values():
                for entry in entries:
                    route = entry.key.get("route")
                    face_id = entry.key.get("face_id")
                    where = f"{path.name}:{face_id}"
                    if route not in skeleton_digests:
                        skeleton_digests[route] = skeleton_digest(route)
                    current = skeleton_digests[route]
                    if current is None or entry.skeleton_sha256 != current:
                        stale.append(f"{where}: skeleton_sha256")
                        continue
                    face = faces.get(face_id)
                    if face is not None:
                        if face_id not in face_digests:
                            face_digests[face_id] = face_digest(face)
                        if entry.face_sha256 != face_digests[face_id]:
                            stale.append(f"{where}: face_sha256")
    return stale


def check_repin(measured_at: str, skeleton_commit_time: str) -> None:
    """Refuse a _TUNED_METAL_DIGEST re-pin measured before the skeleton change."""
    from datetime import UTC, datetime

    def when(value: str) -> datetime:
        text = value.strip().replace("Z", "+00:00")
        stamp = datetime.fromisoformat(text)
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)

    if when(measured_at) <= when(skeleton_commit_time):
        raise ValueError(
            "a _TUNED_METAL_DIGEST re-pin needs evidence measured after the "
            f"skeleton change ({measured_at} <= {skeleton_commit_time})"
        )
