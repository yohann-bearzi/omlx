# SPDX-License-Identifier: Apache-2.0
"""Stock MLX traversal of Qwen's ordinary dense projections on Metal."""

import re


def traversal(
    rows: int,
    n: int,
    k: int,
    group_size: int,
    architecture: str,
    *,
    bits: int = 4,
    batched: bool = False,
) -> tuple[str, int]:
    """Select a dense traversal and K partition count for each matrix.

    For a row-contiguous input, rows includes all leading dimensions. When
    MLX retains multiple input matrices, rows is each matrix's height and
    batched is true. Both use the same QMV limit in MLX 0.32.4.
    """
    match = re.fullmatch(r"applegpu_g([0-9]+)([a-z])", architecture)
    if match is None:
        return "fallback", 1
    generation, size = int(match[1]), match[2]
    if size == "d":
        limits = (32, 18, 12)
    elif generation >= 17:
        limits = (33, 25, 13)
    elif generation >= 15:
        limits = (13, 15, 13)
    elif generation >= 13:
        limits = (14, 10, 7 if generation == 13 and size == "s" else 6)
    else:
        limits = (18, 12, 10)

    width = max(k, n)
    limit = limits[0 if width <= 2048 else 1 if width <= 4096 else 2]
    if rows < limit:
        if k in (64, 128) and bits in (2, 4, 8):
            return "quad", 1
        return ("wide" if rows >= 2 and generation >= 15 else "rows"), 1
    if batched:
        return "qmm", 1

    # qmm_splitk targets 512 threadgroups; the partition must contain both
    # whole quantization groups and whole 32-wide K tiles.
    tiles = ((rows + 31) // 32) * ((n + 31) // 32)
    alignment = max(32, group_size)
    partitions = min(512 // tiles, k // alignment)
    while partitions > 1:
        if k % (partitions * alignment) == 0:
            return "split_k", partitions
        partitions -= 1
    return "qmm", 1
