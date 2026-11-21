# SPDX-License-Identifier: Apache-2.0
"""The skeletons' compile-time configuration types.

Every choice a skeleton can take is fixed when a plan is built: a face, its
geometry, and these hooks. The values are pure data, keyed by the device
generation and the op's shape, so an offline autotuner replaces them without
touching Metal (specs/parity-skeletons.md; owner rules 27 and 28).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


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
