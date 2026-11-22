# SPDX-License-Identifier: Apache-2.0
"""Compile-time skeleton tuning: the public names and the parity defaults.

The dataclasses and hook enums live in ``schema``; the tables here are the
parity floor (code, not tuned data). The offline autotuner imports these
names and populates the tables per chip; an empty table keeps the geometry
rule (``kernels.shared_geometry``), which reproduces the pre-skeleton
builders' choices.
"""

from __future__ import annotations

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
    "DenseTilePolicy",
    "FetchGranularity",
    "LoadLifetime",
    "ParamStaging",
    "RowExactPolicy",
    "RowIndexWidth",
    "UnrollPolicy",
    "row_exact_policy",
    "row_index_width",
]


def row_index_width(n: int, row_bytes: int) -> RowIndexWidth:
    """The row skeleton's index width: int when construction proves the
    largest offset under 2**31, size_t otherwise (owner rule 28: decided at
    construction)."""
    return (
        RowIndexWidth.INT32_CHECKED
        if n * row_bytes < 2**31
        else RowIndexWidth.SIZE_T64
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
