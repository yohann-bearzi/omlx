# SPDX-License-Identifier: Apache-2.0
"""Five- and seven-row wide launches and their cold proof geometry."""

from types import SimpleNamespace

import pytest

from omlx.formats import tile_wide


@pytest.mark.parametrize(
    "rows, vectors, tiles, exact_rows",
    [
        (5, 5, 1, 5),
        (7, 4, 2, 7),
        (4, 4, 1, 0),
        (8, 4, 2, 0),
        (10, 5, 2, 0),
        (11, 4, 3, 0),
    ],
)
def test_wide_decode_geometry(monkeypatch, rows, vectors, tiles, exact_rows):
    k, n, dtype = 320, 13, "float16"
    x = SimpleNamespace(shape=(rows, 1, k), size=rows * k, dtype=dtype)
    operand = SimpleNamespace(shape=(n, k), dtype=dtype)
    launches = []

    def capture(**kwargs):
        launches.append(kwargs)
        return (None,)

    def zeros(*specs):
        shape, dtype = specs[0]
        return SimpleNamespace(shape=shape, size=shape[0] * k, dtype=dtype), operand

    monkeypatch.setattr(tile_wide.kernels, "zeros", zeros)
    binding = SimpleNamespace(kernel=capture, operands=(operand,))
    metal, launch = tile_wide.describe(binding, x)
    launch(x, operand)
    args = launches[-1]
    assert dict(args["template"]).get("EXACT_ROWS") == exact_rows
    assert dict(args["template"])["V"] == vectors
    assert args["grid"] == (32 * tiles, 4, 1)
    assert args["output_shapes"] == [(rows, 1, n)]

    stand_in, prove = tile_wide.stand_in(binding, metal)
    prove(*stand_in)
    assert launches[-1]["template"] == args["template"]
    if exact_rows:
        assert stand_in[0].shape == (exact_rows, k)
        assert launches[-1]["grid"] == args["grid"]


def test_exact_rows_emit_fixed_first_vec8():
    """5 and 7 rows take the exact sources: one V=5 tile and the 4+3 split,
    with Vec8 reads behind a fixed first row."""
    assert "omlx_wide<T, K, N, 5, false, 0>" in tile_wide._SOURCE
    assert "omlx_wide<T, K, N, 3, false, 4>" in tile_wide._SOURCE
    assert "FIXED_FIRST >= 0" in tile_wide._HEADER


@pytest.mark.parametrize("rows", [5, 7])
def test_wide_exact_rows_odd_offset_writes_every_row_once(rows):
    import mlx.core as mx
    import numpy as np

    from tests.test_formats_tile_prefill import _TILE, _tile_toy

    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    k, n = 320, 13
    face = _tile_toy(f"d83_odd_{rows}", _TILE)
    codes = mx.full((16 * k + 64,), 129, mx.uint8)
    scales = mx.ones((16,), mx.float32)
    body = mx.repeat(mx.arange(1, rows + 1, dtype=mx.float16), k)
    x = mx.concatenate([mx.zeros((1,), mx.float16), body])[1:]
    vectors, partial_m, exact_rows = tile_wide.layout(rows)
    (got,) = tile_wide.kernel(face)(
        inputs=[x, codes, scales],
        template=tile_wide.template(mx.float16, k, n, vectors, partial_m, exact_rows),
        grid=(32 * ((rows + 4) // 5), 4, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows * n + 3,)],
        output_dtypes=[mx.float16],
        init_value=-31,
    )
    values = np.asarray(got)
    want = np.arange(1, rows + 1)[:, None] * k
    np.testing.assert_array_equal(
        values[: rows * n].reshape(rows, n),
        np.broadcast_to(want, (rows, n)),
    )
    np.testing.assert_array_equal(values[rows * n :], [-31, -31, -31])
