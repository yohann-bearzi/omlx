# SPDX-License-Identifier: Apache-2.0
"""MLX 0.32.4 dense dispatch for the transposed matrices used by Flash-Next."""

import pytest


@pytest.mark.parametrize(
    "architecture,limits,vector",
    [
        ("applegpu_g17s", (33, 25, 13), "wide"),
        ("applegpu_g15s", (13, 15, 13), "wide"),
        ("applegpu_g17d", (32, 18, 12), "wide"),
        ("applegpu_g15d", (32, 18, 12), "wide"),
        ("applegpu_g14s", (14, 10, 6), "rows"),
        ("applegpu_g13s", (14, 10, 7), "rows"),
        ("applegpu_g13g", (14, 10, 6), "rows"),
        ("applegpu_g12d", (32, 18, 12), "rows"),
        ("applegpu_g12s", (18, 12, 10), "rows"),
    ],
)
def test_dense_row_boundaries(architecture, limits, vector):
    from omlx.formats.mlx import traversal

    # Both dimensions enter the size class. The first larger aligned input
    # and first larger output independently cross each inclusive boundary.
    for k, n, index in (
        (2048, 2048, 0),
        (2112, 2048, 1),
        (2048, 2049, 1),
        (4096, 4096, 1),
        (4160, 4096, 2),
        (4096, 4097, 2),
    ):
        limit = limits[index]
        assert traversal(1, n, k, 64, architecture) == ("rows", 1)
        assert traversal(limit - 1, n, k, 64, architecture) == (vector, 1)
        route, _ = traversal(limit, n, k, 64, architecture)
        assert route in ("qmm", "split_k")


@pytest.mark.parametrize(
    "rows,n,k,group,parts",
    [
        # A format can have groups smaller than the split tile's K step.
        (33, 1, 2560, 16, 80),
        (33, 1, 2560, 64, 40),
        (33, 4, 2560, 128, 20),
        (33, 48, 2560, 32, 80),
        (2048, 1, 2560, 64, 8),
        (4096, 1, 2560, 64, 4),
        (8192, 1, 2560, 64, 2),
        (2048, 4, 2560, 64, 8),
        (4096, 4, 2560, 64, 4),
        (8192, 4, 2560, 64, 2),
        (2048, 48, 2560, 64, 4),
        (4096, 48, 2560, 64, 2),
        (4097, 48, 2560, 64, 1),
        (8192, 48, 2560, 64, 1),
        # 512/(2*41) permits six partitions, but 40 groups divide into five.
        (64, 1281, 2560, 64, 5),
        (65, 1281, 2560, 64, 4),
        (1024, 256, 2560, 64, 2),
        (1024, 257, 2560, 64, 1),
    ],
)
def test_dense_split_count(rows, n, k, group, parts):
    from omlx.formats.mlx import traversal

    route = "split_k" if parts > 1 else "qmm"
    for arch in ("applegpu_g15d", "applegpu_g17s"):
        assert traversal(rows, n, k, group, arch) == (route, parts)


@pytest.mark.parametrize(
    "architecture,limit", [("applegpu_g17s", 25), ("applegpu_g15d", 18)]
)
@pytest.mark.parametrize("n", [1, 4, 48, 512])
def test_a_batch_uses_rows_per_matrix_and_keeps_k_whole(architecture, limit, n):
    from omlx.formats.mlx import traversal

    # MLX's quantized.cpp dispatches QMV below its limit on each matrix.
    # At the limit it goes straight to QMM when B > 1, even at tiny N.
    def batched(rows):
        return traversal(rows, n, 2560, 64, architecture, batched=True)

    assert batched(1) == ("rows", 1)
    assert batched(2) == batched(limit - 1) == ("wide", 1)
    assert batched(limit) == batched(64) == ("qmm", 1)
    assert traversal(limit, n, 2560, 64, architecture)[0] == "split_k"


@pytest.mark.parametrize("k", [64, 128])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_narrow_matrices_select_stock_quad_only_for_power_of_two_bits(k, bits):
    from omlx.formats.mlx import traversal

    expected = "wide" if bits == 3 else "quad"
    assert traversal(2, 512, k, 64, "applegpu_g17s", bits=bits) == (expected, 1)
