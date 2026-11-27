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
