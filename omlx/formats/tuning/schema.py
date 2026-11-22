# SPDX-License-Identifier: Apache-2.0
"""The skeletons' compile-time configuration types.

Every choice a skeleton can take is fixed when a plan is built: a face, its
geometry, and these hooks. The values are pure data, keyed by the device
generation and the op's shape, so an offline autotuner replaces them without
touching Metal (specs/parity-skeletons.md; owner rules 27 and 28).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Literal

from .. import kernels


class RowIndexWidth(Enum):
    """A row skeleton's row/group offsets. INT32_CHECKED when construction
    proves the largest offset under 2**31 (every served matrix today);
    SIZE_T64 when it cannot (a weight stream past 2 GB)."""

    INT32_CHECKED = "int"
    SIZE_T64 = "size_t"


class UnrollPolicy(Enum):
    """How a skeleton's row loop is unrolled. PRAGMA_FULL lets SROA/LICM run
    before the unroll (the fold-expression form unrolled first and inflated
    AIR 2.5-4x); ROLLED for shapes past the register budget."""

    PRAGMA_FULL = "FULL"
    ROLLED = ""


class LoadLifetime(Enum):
    """When a row's decoded weights are consumed: all output columns at once,
    or one column streamed before the next fetch."""

    BATCH_COLUMNS = "batch"
    ONE_COL_STREAMING = "column"
    GROUP_BASE_REBASE = "group_base"


class FetchGranularity(Enum):
    """A tile's 8-bit code load: one aligned uint2, a packed (align 4) pair,
    or a 32-bit load for 4-bit run 8."""

    ALIGNED_U64_RUN8 = "uint2"
    PACKED_RUN8 = "packed_uint2"
    U32_RUN8 = "uint"


class ParamStaging(Enum):
    """When a dense tile loads scales/biases: every K step, or once per
    GROUP of steps (BK 32 against GROUP 64/128)."""

    PER_STEP = "per_step"
    GROUP_STEPS_COUNTER = "group_steps"


@dataclass(frozen=True)
class RowExactPolicy:
    """One row-exact launch's compile-time geometry."""

    rows_per_group: int
    rps: int
    unrolled: bool


@dataclass(frozen=True)
class DenseTilePolicy:
    """A direct dense tile's compile-time geometry (release-3 spec §3.2)."""

    backend: Literal["steel", "nax"]
    bm: int
    bn: int
    bk: int
    wm: int = 2
    wn: int = 2
    align_n: bool = True
    split_k: int = 1
    x_nreads: int = 4
    param_staging: ParamStaging = ParamStaging.GROUP_STEPS_COUNTER
    vec_store: bool = False
    pad_bytes: int = 0
    gx: int = 0

    def is_legal(
        self, face, k: int, n: int, m: int, chip: str, dtype: Any = "bf16"
    ) -> bool:
        """Whether this geometry runs on ``face``'s tables, its shape and chip."""
        if self.backend not in _BACKENDS:
            return False
        if self.backend == "nax":
            match = _CHIP_GEN_RE.match((chip or "").lower())
            if match is None:
                return False
            generation = int(match.group(1))
            if generation < (18 if match.group(2) == "p" else 17):
                return False
        if min(self.bm, self.bn, self.bk, self.wm, self.wn) < 1:
            return False
        if self.bk not in _BKS or self.bm % 16 or self.bn % 16:
            return False
        # A simdgroup computes a fragment of BM/wm rows and BN/wn columns:
        # whole fragments or nothing.
        frag = 16 if self.backend == "nax" else 8
        if self.bm % (frag * self.wm) or self.bn % (frag * self.wn):
            return False
        if self.backend == "nax":
            # tile_matmad_nax (omlx/custom_kernels/nax_tiles.py:789) dispatches
            # on `TN == 1 && TM % 2 == 0` (:810) and `TN % 2 == 0` (:828) with
            # no final else: every other fragment shape is a silent no-op.
            # TN/TM are the C fragment's kTileCols/kTileRows.
            tm = self.bm // (frag * self.wm)
            tn = self.bn // (frag * self.wn)
            if not (tn % 2 == 0 or (tn == 1 and tm % 2 == 0)):
                return False
        tgp = self.wm * self.wn * 32
        if tgp > 1024:
            return False
        if k < 1 or n < 1 or m < 1:
            return False
        if self.split_k < 1 or k % (self.split_k * self.bk):
            return False
        if self.align_n and n % self.bn:
            return False
        if self.x_nreads not in _X_NREADS:
            return False
        if self.backend == "steel":
            # mlx::steel::BlockLoader (steel/gemm/loader.h:22-24,74-96) writes
            # TROWS rows per pass; BM must be a whole multiple of TROWS.
            trows = tgp // (self.bk // self.x_nreads)
            if self.bm % trows:
                return False
        if not isinstance(self.param_staging, ParamStaging):
            return False
        if self.pad_bytes < 0 or self.gx < 0:
            return False
        # The staging the kernels actually allocate: tile_gather stages both
        # operands (`xs[BM*stride]`, `ws[BN*stride]`), the NAX face kernel its
        # weight buffer alone, with `stride = BK + 16 / sizeof(T)` and
        # `Ws[...] + PAD / sizeof(T)`.  Keep this beside those sources.
        size = getattr(dtype, "size", None)
        if isinstance(size, int) and not isinstance(size, bool) and size > 0:
            elem_bytes = size
        else:
            text = str(getattr(dtype, "name", dtype)).removeprefix("mlx.core.")
            elem_bytes = _DTYPE_BYTES.get(text)
            if elem_bytes is None:
                return False
        stride = self.bk + 16 // elem_bytes
        weight_bytes = elem_bytes * self.bn * stride
        activations = 0 if self.backend == "nax" else elem_bytes * self.bm * stride
        tg = weight_bytes + activations + self.pad_bytes + kernels.table_bytes(face)
        return tg <= _THREADGROUP_BYTES


_BACKENDS = ("steel", "nax")
_CHIP_GEN_RE = re.compile(r"^applegpu_g(\d+)([a-z]?)(?:_|$)")
_BKS = (32, 64, 128)
_X_NREADS = (4, 8)
_DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
    "fp64": 8,
    "float64": 8,
}
_THREADGROUP_BYTES = 32768
_ROUTES: dict[str, type] = {"dense_tile": DenseTilePolicy}


def route_names() -> tuple[str, ...]:
    return tuple(_ROUTES)


def route_class(route: str):
    try:
        return _ROUTES[route]
    except KeyError:
        raise ValueError(f"tuning: no skeleton route {route!r}") from None


def params_from_json(route: str, data: Any):
    """A route's frozen parameters, validated, from a table entry's JSON."""
    cls = route_class(route)
    if not isinstance(data, dict):
        raise ValueError(f"tuning: {route} params are an object, not {data!r}")
    known = {item.name for item in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"tuning: {route} params have unknown keys {sorted(unknown)}")
    if data.get("backend") not in _BACKENDS:
        raise ValueError(
            f"tuning: {route} backend is one of {_BACKENDS}, not "
            f"{data.get('backend')!r}"
        )
    if "bk" in data and data["bk"] not in _BKS:
        raise ValueError(f"tuning: {route} bk is one of {_BKS}, not {data['bk']!r}")
    for key, kind in (
        ("backend", str),
        ("bm", int),
        ("bn", int),
        ("bk", int),
        ("wm", int),
        ("wn", int),
        ("align_n", bool),
        ("split_k", int),
        ("x_nreads", int),
        ("param_staging", str),
        ("vec_store", bool),
        ("pad_bytes", int),
        ("gx", int),
    ):
        if key not in data:
            continue
        value = data[key]
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ValueError(
                f"tuning: {route} param {key!r} is {kind.__name__}, not {value!r}"
            )
    if "param_staging" in data:
        try:
            data = {**data, "param_staging": ParamStaging(data["param_staging"])}
        except ValueError:
            raise ValueError(
                f"tuning: {route} param_staging is one of "
                f"{[item.value for item in ParamStaging]}, not "
                f"{data['param_staging']!r}"
            ) from None
    try:
        return cls(**data)
    except TypeError as error:
        raise ValueError(f"tuning: {route} params: {error}") from None
