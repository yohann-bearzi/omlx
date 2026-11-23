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
from typing import Any, Literal, get_origin, get_type_hints

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


def _elem_bytes(dtype: Any) -> int | None:
    """A dtype's element size, or None when it is not a known element."""
    size = getattr(dtype, "size", None)
    if isinstance(size, int) and not isinstance(size, bool) and size > 0:
        return size
    text = str(getattr(dtype, "name", dtype)).removeprefix("mlx.core.")
    return _DTYPE_BYTES.get(text)


@dataclass(frozen=True)
class DenseTilePolicy:
    """A direct dense tile's compile-time geometry (release-3 spec §3.2)."""

    backend: Literal["steel", "nax", "wide", "mlx", "rows"]
    bm: int = 0
    bn: int = 0
    bk: int = 0
    wm: int = 2
    wn: int = 2
    align_n: bool = True
    split_k: int = 1
    x_nreads: int = 4
    param_staging: ParamStaging = ParamStaging.GROUP_STEPS_COUNTER
    vec_store: bool = False
    pad_bytes: int = 0
    gx: int = 0
    sched: Literal["seg", "db"] = "seg"

    def is_legal(
        self, face, k: int, n: int, m: int, chip: str, dtype: Any = "bf16"
    ) -> bool:
        """Whether this geometry runs on ``face``'s tables, its shape and chip."""
        if self.backend not in _BACKENDS:
            return False
        if k < 1 or n < 1 or m < 1:
            return False
        if self.backend in _ROUTE_ONLY_BACKENDS:
            return (
                self.bm == 0
                and self.bn == 0
                and self.bk == 0
                and self.split_k == 1
                and self.sched == "seg"
                and self.pad_bytes == 0
                and self.gx == 0
            )
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
            if self.sched == "db" and tn % 2 != 0:
                return False
            if self.wm * self.wn * 32 < 2 * self.bn:
                # kTGP < BN leaves Ws uninitialized; BN <= kTGP < 2*BN selects
                # kLT == BN (measured 1.33x stock rel_l2 on M5, n=64 k=512
                # rows=32 4b g64; DEFERRED: kLT==BN NAX precision).
                return False
        tgp = self.wm * self.wn * 32
        if tgp > 1024:
            return False
        if self.split_k < 1 or k % (self.split_k * self.bk):
            return False
        if self.align_n and n % self.bn:
            return False
        if self.x_nreads not in _X_NREADS:
            return False
        if self.sched not in _SCHEDS:
            return False
        if self.sched == "db" and (
            self.backend != "nax" or not self.align_n or k % self.bk != 0
        ):
            return False
        if self.backend == "steel":
            # mlx::steel::BlockLoader (steel/gemm/loader.h:22-24,74-96) tiles
            # BK into TCOLS = BK / x_nreads columns and BM into TROWS = TGP / TCOLS
            # rows; omlx_steel_dense::WeightLoader splits BN * BK evenly across TGP.
            if self.bk % self.x_nreads:
                return False
            tcols = self.bk // self.x_nreads
            if tgp % tcols:
                return False
            trows = tgp // tcols
            if trows < 1 or self.bm % trows:
                return False
            if (self.bn * self.bk) % tgp:
                return False
            values = (self.bn * self.bk) // tgp
            if values < 8 or self.bk % values:
                return False
        if not isinstance(self.param_staging, ParamStaging):
            return False
        if self.pad_bytes < 0 or self.gx < 0:
            return False
        # The staging the kernels actually allocate: tile_dense steel stages
        # both operands (`xs[BM*stride]`, `ws[BN*stride]`), NAX stages its
        # weight buffer (2x when double-buffered), with `stride = BK + 16 /
        # sizeof(T)` and `Ws[...] + PAD / sizeof(T)`. Keep beside those sources.
        elem_bytes = _elem_bytes(dtype)
        if elem_bytes is None:
            return False
        stride = self.bk + 16 // elem_bytes
        weight_buffers = 2 if self.sched == "db" else 1
        weight_bytes = elem_bytes * weight_buffers * self.bn * stride
        activations = 0 if self.backend == "nax" else elem_bytes * self.bm * stride
        tg = weight_bytes + activations + self.pad_bytes + kernels.table_bytes(face)
        return tg <= _THREADGROUP_BYTES


@dataclass(frozen=True)
class MmaTilePolicy:
    """One MMA skeleton launch's compile-time tile: BN output columns per
    8-row tile, BK K steps per simdgroup pass, NSG simdgroups."""

    bn: int = 16
    bk: int = 32
    nsg: int = 8

    def is_legal(
        self, face, k: int, n: int, m: int, chip: str, dtype: Any = "bf16"
    ) -> bool:
        """Whether the template runs these constants for this face's shape.

        Mirrors ``mma_asserts`` (BN/BK choices, NSG threads, K divisibility,
        threadgroup bytes with the actual dtype) plus the runtime N, row and
        face-table checks; the parity tile (16, 32, 8) is the digest-pinned
        default."""
        if self.bn not in MMA_BN_CHOICES or self.bk not in MMA_BK_CHOICES:
            return False
        if self.nsg < 1 or 32 * self.nsg > 1024:
            return False
        if m < 1 or n < 1 or k < 1:
            return False
        if k % (self.bk * self.nsg) or n % self.bn:
            return False
        elem_bytes = _elem_bytes(dtype)
        if elem_bytes is None:
            return False
        bm = ((m + 7) // 8) * 8
        staging = self.nsg * (self.bk * self.bn * elem_bytes + bm * self.bn * 4)
        return staging + kernels.table_bytes(face) <= _THREADGROUP_BYTES


MMA_BN_CHOICES = (8, 16, 32)
MMA_BK_CHOICES = (32, 64, 128)


def mma_asserts() -> tuple[tuple[str, str], ...]:
    """The MMA template's compile-time constraints, as (expression, message).

    ``emit_mma_source`` emits these as ``static_assert`` lines for a tuned
    tile; ``MmaTilePolicy.is_legal`` enforces the same list in Python."""
    return (
        ("BN == 8 || BN == 16 || BN == 32", "BN is 8, 16 or 32"),
        ("BK == 32 || BK == 64 || BK == 128", "BK is 32, 64 or 128"),
        ("NSG >= 1 && NSG * 32 <= 1024", "NSG * 32 threads at most 1024"),
        (
            "K > 0 && BK > 0 && NSG > 0 && K % (BK * NSG) == 0",
            "K is a multiple of BK * NSG",
        ),
        (
            "BM > 0 && NSG >= 1 && NSG * (BK * BN * sizeof(T) + BM * BN * 4) <= 32768",
            "threadgroup staging is at most 32768 bytes",
        ),
    )


_BACKENDS = ("steel", "nax", "wide", "mlx", "rows")
_ROUTE_ONLY_BACKENDS = ("wide", "mlx", "rows")
_SCHEDS = ("seg", "db")
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
_ROUTES: dict[str, type] = {
    "dense_tile": DenseTilePolicy,
    "mma_tile": MmaTilePolicy,
}


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
    if cls is DenseTilePolicy and data.get("backend") not in _BACKENDS:
        raise ValueError(
            f"tuning: {route} backend is one of {_BACKENDS}, not "
            f"{data.get('backend')!r}"
        )
    hints = get_type_hints(cls)
    for name, hint in hints.items():
        if name not in data:
            continue
        value = data[name]
        if hint is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif hint is bool:
            ok = isinstance(value, bool)
        elif (
            hint is str
            or get_origin(hint) is Literal
            or (isinstance(hint, type) and issubclass(hint, Enum))
        ):
            ok = isinstance(value, str)
        else:
            ok = True
        if not ok:
            raise ValueError(
                f"tuning: {route} param {name!r} is not a valid "
                f"{getattr(hint, '__name__', hint)}: {value!r}"
            )
    if cls is DenseTilePolicy:
        backend = data.get("backend")
        if backend in _ROUTE_ONLY_BACKENDS:
            for dim in ("bm", "bn", "bk"):
                if dim in data and data[dim] != 0:
                    raise ValueError(
                        f"tuning: {route} {backend!r} route takes {dim}=0, "
                        f"not {data[dim]!r}"
                    )
        else:
            for dim in ("bm", "bn", "bk"):
                if dim not in data:
                    raise ValueError(
                        f"tuning: {route} {backend!r} params require {dim!r}"
                    )
            if data["bk"] not in _BKS:
                raise ValueError(
                    f"tuning: {route} bk is one of {_BKS}, not {data['bk']!r}"
                )
        if "sched" in data and data["sched"] not in _SCHEDS:
            raise ValueError(
                f"tuning: {route} sched is one of {_SCHEDS}, not {data['sched']!r}"
            )
    elif "bk" in data and data["bk"] not in _BKS:
        raise ValueError(f"tuning: {route} bk is one of {_BKS}, not {data['bk']!r}")
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
