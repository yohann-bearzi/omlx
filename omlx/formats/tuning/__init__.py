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
import re
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
    MmaTilePolicy,
    ParamStaging,
    RoutedWindowPolicy,
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
    "MmaTilePolicy",
    "ParamStaging",
    "RoutedWindowPolicy",
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
    "row_bounds",
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
_KEY_FIELDS = frozenset(
    {
        "route",
        "face_id",
        "dtype",
        "bits",
        "group",
        "rows_bucket",
        "n",
        "k",
        "n_bucket",
        "k_bucket",
    }
)
_PROFILE_WILDCARDS = frozenset({"*:plain", "*:lut_small", "*:lut_tg"})
_DTYPES = frozenset({"bf16", "fp16", "fp32"})
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
    "mma_tile": (
        (1, 8, "8"),
        (9, 16, "16"),
        (17, 24, "24"),
    ),
    "routed_window": (
        (2, 2, "2"),
        (3, 3, "3"),
        (4, 4, "4"),
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


_N_BUCKETS = frozenset({"N<=1024", "N1025-4096", "N4097-32768", "N>32768"})
_K_BUCKETS = frozenset({"K<=1024", "K1025-4096", "K>4096"})
_ROWS_BUCKET_NAMES = {
    route: frozenset({name for _, _, name in buckets})
    | {f"<{buckets[0][0]}", f"{buckets[-1][1] + 1}+"}
    for route, buckets in _ROWS_BUCKETS.items()
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
    """The route's current skeleton template source, for staleness checks.

    A route the tuner cannot bucket must not register: a lookup for it would
    silently compare unruly row strings and miss every entry."""
    if route not in _ROWS_BUCKETS:
        raise ValueError(
            f"tuning: route {route!r} has no row buckets; a lookup would miss "
            "every entry (add it to _ROWS_BUCKETS)"
        )
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
    for fn in (current_chip, face_digest):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()


def table_profile(face) -> str:
    """The table class a face falls in: plain, a small or a large LUT."""
    size = kernels.table_bytes(face)
    if size == 0:
        return "plain"
    return "lut_small" if size <= 256 else "lut_tg"


@cache
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


_ROWS_SPAN_RE = re.compile(
    r"\A(?:([1-9][0-9]*)-([1-9][0-9]*)|([1-9][0-9]*)\+|<([1-9][0-9]*)|([1-9][0-9]*))\Z"
)


def _parse_rows_span(bucket: object) -> tuple[int | None, int | None] | None:
    """Parse a row bucket or explicit row range (e.g. ``44-52``, ``2049+``)."""
    if not isinstance(bucket, str):
        return None
    match = _ROWS_SPAN_RE.match(bucket)
    if match is None:
        return None
    lo_s, hi_s, plus_s, lt_s, exact_s = match.groups()
    if lo_s is not None:
        lo, hi = int(lo_s), int(hi_s)
        return (lo, hi) if lo <= hi else None
    if plus_s is not None:
        return (int(plus_s), None)
    if lt_s is not None:
        return (None, int(lt_s) - 1)
    n = int(exact_s)
    return (n, n)


@dataclass(frozen=True)
class _Entry:
    key: dict
    face_sha256: str
    skeleton_sha256: str
    measured_at: str
    params: object
    exactness: str
    rows_span: tuple[int | None, int | None] | None = None


@dataclass(frozen=True)
class ChipTable:
    entries: tuple[_Entry, ...]


def _check_key(path: Path, key: dict) -> None:
    """The entry's key, fail closed: a typo must not silently never match."""
    unknown = set(key) - _KEY_FIELDS
    if unknown:
        raise ValueError(f"tuning: {path} key has unknown fields {sorted(unknown)}")
    route = key.get("route")
    if not isinstance(route, str) or not route:
        raise ValueError(f"tuning: {path} key names no route")
    dtype = key.get("dtype")
    if not isinstance(dtype, str) or dtype not in _DTYPES:
        raise ValueError(
            f"tuning: {path} key dtype is one of {sorted(_DTYPES)}, not {dtype!r}"
        )
    rows_bucket = key.get("rows_bucket")
    allowed = _ROWS_BUCKET_NAMES.get(route)
    span = _parse_rows_span(rows_bucket)
    custom = (
        isinstance(rows_bucket, str)
        and "-" in rows_bucket
        and span is not None
        and span[0] is not None
        and span[1] is not None
        and 1 <= span[0] <= span[1]
    )
    if (
        not isinstance(rows_bucket, str)
        or not rows_bucket
        or (allowed is not None and rows_bucket not in allowed and not custom)
    ):
        raise ValueError(
            f"tuning: {path} key rows_bucket {rows_bucket!r} is not one this "
            "route has"
        )
    for name in ("bits", "group"):
        value = key.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(
                f"tuning: {path} key {name} is a positive int, not {value!r}"
            )
    face_id = key.get("face_id")
    if not isinstance(face_id, str) or not face_id:
        raise ValueError(f"tuning: {path} has an entry without face_id")
    if face_id.startswith("*:") and face_id not in _PROFILE_WILDCARDS:
        raise ValueError(
            f"tuning: {path} wildcard {face_id!r} is one of "
            f"{sorted(_PROFILE_WILDCARDS)}"
        )
    exact = "n" in key or "k" in key
    bucketed = "n_bucket" in key or "k_bucket" in key
    if exact == bucketed:
        raise ValueError(f"tuning: {path} key needs n and k, or n_bucket and k_bucket")
    if exact:
        for name in ("n", "k"):
            value = key.get(name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"tuning: {path} key {name} is a positive int, not {value!r}"
                )
    else:
        for name, buckets in (("n_bucket", _N_BUCKETS), ("k_bucket", _K_BUCKETS)):
            value = key.get(name)
            if value not in buckets:
                raise ValueError(
                    f"tuning: {path} key {name} {value!r} is not one of "
                    f"{sorted(buckets)}"
                )


def _read_table(path: Path, chips: dict[str, list[_Entry]]) -> None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"tuning: cannot read {path}: {error}") from None
    version = data.get("schema_version") if isinstance(data, dict) else data
    if not isinstance(data, dict) or type(version) is not int or version != 1:
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
        _check_key(path, key)
        face_id = key["face_id"]
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
        rows_bucket = key.get("rows_bucket")
        allowed = _ROWS_BUCKET_NAMES.get(key.get("route"))
        parsed.append(
            _Entry(
                dict(key),
                face_sha,
                skeleton,
                measured,
                params,
                exactness,
                # A route's named buckets keep their label comparison; only
                # a custom span (e.g. "44-52") outside them becomes a span.
                (
                    None
                    if allowed is not None and rows_bucket in allowed
                    else _parse_rows_span(rows_bucket)
                ),
            )
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
    rows: int
    rows_bucket: str
    n_bucket: str
    k_bucket: str


def _matches(entry: _Entry, query: _Query) -> tuple[int, int]:
    key = entry.key
    if (
        key.get("route") != query.route
        or key.get("dtype") != query.dtype
        or key.get("bits") != query.bits
        or key.get("group") != query.group
    ):
        return (0, 0)
    span = entry.rows_span
    if span is not None:
        lo, hi = span
        if (lo is not None and query.rows < lo) or (hi is not None and query.rows > hi):
            return (0, 0)
        span_rank = (
            -(hi - lo + 1) if (lo is not None and hi is not None) else -(1 << 30)
        )
    else:
        if key.get("rows_bucket") != query.rows_bucket:
            return (0, 0)
        span_rank = -(1 << 29)
    face_id = key.get("face_id")
    exact_face = face_id == query.face_key
    if not exact_face and face_id != f"*:{query.profile}":
        return (0, 0)
    if "n" in key and "k" in key:
        if key["n"] != query.n or key["k"] != query.k:
            return (0, 0)
        return (4 if exact_face else 2, span_rank)
    if key.get("n_bucket") != query.n_bucket or key.get("k_bucket") != query.k_bucket:
        return (0, 0)
    return (3 if exact_face else 1, span_rank)


def row_bounds(route: str, *, chip: str | None = None) -> tuple[int, ...]:
    """Sorted upper-edge row counts where ``lookup(route, ...)`` can change on
    ``chip`` (the route's fixed buckets plus any custom row-span edges in the
    chip's table). Resolved once at plan construction (rule 28)."""
    buckets = _ROWS_BUCKETS.get(route, ())
    cuts = {hi for _lo, hi, _name in buckets}
    chip_id = current_chip() if chip is None else chip
    table = _index_for_chip(chip_id)
    for entry in table.entries:
        if entry.key.get("route") != route or entry.rows_span is None:
            continue
        lo, hi = entry.rows_span
        if lo is not None and lo > 1:
            cuts.add(lo - 1)
        if hi is not None and hi >= 1:
            cuts.add(hi)
    return tuple(sorted(cut for cut in cuts if cut >= 1))


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
    chip_id = current_chip() if chip is None else chip
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
        rows=rows,
        rows_bucket=rows_bucket(route, rows),
        n_bucket=_n_bucket(n),
        k_bucket=_k_bucket(k),
    )
    digest: str | None = None
    best_rank = (0, 0)
    best_params = None
    for entry in table.entries:
        rank = _matches(entry, query)
        if rank[0] == 0 or rank <= best_rank:
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
        best_rank = rank
        best_params = entry.params
    return best_params


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
        if not base.exists():
            raise ValueError(f"tuning: {entry_path} does not exist")
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
    """Refuse a _TUNED_METAL_DIGEST re-pin measured at or before the
    skeleton change."""
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
