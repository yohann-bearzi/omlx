# SPDX-License-Identifier: Apache-2.0
"""oQh as a built-in format: both its row-scale and group-scale faces decode
their tensors, their rows, tiles and codes agree with the arithmetic the
tensors stand for, and the fused prefill paths take them.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx import formats
from omlx.formats import affine, oqh, testing, tile_dense, transforms
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.oqh import format as oqh_format
from omlx.oqh import quantize

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_BITS = (2, 3, 4, 5, 6, 8)


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / mx.abs(b).max()).item()


def _row_case(k, n, bits, seed=0, block=128):
    mx.random.seed(seed)
    w = mx.random.normal((n, k)).astype(mx.float32) * 0.05
    signs = mx.array(oqh_format.make_signs("w", k)).astype(mx.float32)
    packed, rscale, _, _ = quantize.quantize(w, mx.eye(k)[None], bits, signs, block)
    rscale = rscale.astype(mx.bfloat16)
    gs = next(g for g in (128, 64, 32) if k % g == 0)
    codes = mx.dequantize(
        packed,
        mx.ones((n, k // gs), mx.float32),
        mx.zeros((n, k // gs), mx.float32),
        group_size=gs,
        bits=bits,
    )
    rotated = (codes - oqh._CENTER[bits]) * rscale.astype(mx.float32)[..., None]
    tensors = {
        "weight": packed,
        "rscale": rscale,
        "oqh_signs": signs.reshape(-1, block).astype(mx.bfloat16),
    }
    return tensors, rotated


def _group_case(k, n, bits, group_size, seed=0, block=128):
    mx.random.seed(seed)
    w = mx.random.normal((n, k)).astype(mx.float32) * 0.05
    signs = mx.array(oqh_format.make_signs("w", k)).astype(mx.float32)
    packed, scales, biases, _, _ = quantize.quantize_affine(
        w, mx.eye(k)[None], bits, signs, block, group_size
    )
    scales, biases = scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)
    rotated = mx.dequantize(
        packed,
        scales.astype(mx.float32),
        biases.astype(mx.float32),
        group_size=group_size,
        bits=bits,
    )
    tensors = {
        "weight": packed,
        "oqh_scales": scales,
        "oqh_biases": biases,
        "oqh_signs": signs.reshape(-1, block).astype(mx.bfloat16),
    }
    return tensors, rotated


def _layer(fmt, bits, group_size, k, tensors):
    with testing.verified("example"):
        return FormatLinear(fmt, bits, group_size, k, tensors)


@pytest.mark.parametrize("bits", _BITS)
@pytest.mark.parametrize("k,n", [(128, 8), (640, 16), (2560, 8)])
def test_the_row_faces_read_their_tensors(bits, k, n):
    tensors, rotated = _row_case(k, n, bits)
    layer = _layer(oqh.OQH, bits, 0, k, tensors)
    face = oqh.OQH.dot(layer, k, n)
    assert face is not None
    x = (mx.random.normal((k,)) * 0.1).astype(mx.bfloat16)
    got = testing.dot_rows(face, [tensors["weight"], tensors["rscale"]], x, n, (1, 0))
    assert _rel(got, x.astype(mx.float32) @ rotated.astype(mx.float32).T) < 2e-2
    # The tile face reads every weight of a row as the formula does.
    tile = oqh.OQH.tile(layer, k, n)
    assert tile is not None and tile.key.endswith("bf16")
    weights, rounded = testing.tile_rows(
        tile, [tensors["weight"], tensors["rscale"]], k, range(n), mx.bfloat16
    )
    assert _rel(weights, rotated) < 1e-6
    assert rounded.dtype == mx.bfloat16
    # Codes: levels and the affine map, for the widths a run has.
    codes = oqh.OQH.codes(layer, k, n)
    if bits == 4:
        assert codes is not None
        levels, maps = testing.codes_rows(
            codes, [tensors["weight"], tensors["rscale"]], k, range(n), mx.bfloat16
        )
        assert _rel(maps, rotated) < 2e-2
        assert int(levels.min().item()) >= 0
    else:
        assert codes is None


@pytest.mark.parametrize("bits", _BITS)
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("k,n", [(128, 8), (640, 16), (2560, 8)])
def test_the_group_faces_read_their_tensors(bits, group_size, k, n):
    tensors, rotated = _group_case(k, n, bits, group_size)
    layer = _layer(oqh.OQH_AFFINE, bits, group_size, k, tensors)
    face = oqh.OQH_AFFINE.dot(layer, k, n)
    assert face is not None
    x = (mx.random.normal((k,)) * 0.1).astype(mx.bfloat16)
    got = testing.dot_rows(
        face,
        [tensors["weight"], tensors["oqh_scales"], tensors["oqh_biases"]],
        x,
        n,
        (1, 0),
    )
    assert _rel(got, x.astype(mx.float32) @ rotated.astype(mx.float32).T) < 2e-2
    tile = oqh.OQH_AFFINE.tile(layer, k, n)
    assert tile is not None
    weights, rounded = testing.tile_rows(
        tile,
        [tensors["weight"], tensors["oqh_scales"], tensors["oqh_biases"]],
        k,
        range(n),
        mx.bfloat16,
    )
    assert _rel(weights, rotated) < 1e-6
    if bits in (2, 3, 4, 5, 6):
        codes = oqh.OQH_AFFINE.codes(layer, k, n)
        assert codes is not None
        levels, maps = testing.codes_rows(
            codes,
            [tensors["weight"], tensors["oqh_scales"], tensors["oqh_biases"]],
            k,
            range(n),
            mx.bfloat16,
        )
        assert _rel(maps, rotated) < 2e-2
    else:
        assert oqh.OQH_AFFINE.codes(layer, k, n) is None


def test_rows_stack_with_one_shared_sign_vector():
    k, n, bits = 256, 8, 4
    first, _ = _row_case(k, n, bits, seed=1)
    second, _ = _row_case(k, n, bits, seed=2)
    second["oqh_signs"] = first["oqh_signs"]
    x = (mx.random.normal((k,)) * 0.1).astype(mx.bfloat16)
    with testing.verified("example"):
        testing.rows_stack(oqh.OQH, first, second, k, bits, 0, x=x)
    a, _ = _group_case(k, n, bits, 128, seed=1)
    b, _ = _group_case(k, n, bits, 128, seed=2)
    b["oqh_signs"] = a["oqh_signs"]
    with testing.verified("example"):
        testing.rows_stack(oqh.OQH_AFFINE, a, b, k, bits, 128, x=x)


def test_the_built_in_formats_register_and_read_their_records():
    formats.load()
    assert formats.get("oqh") is oqh.OQH
    assert formats.get("oqh_affine") is oqh.OQH_AFFINE
    assert not formats._mlx_runs("oqh") and not formats._mlx_runs("oqh_affine")
    for fmt, _tensors in (
        (oqh.OQH, ("rscale",)),
        (oqh.OQH_AFFINE, ("oqh_scales", "oqh_biases")),
    ):
        assert fmt.record() == {"transform": {"version": 1, "kind": "signed_wht"}}
        assert fmt.accept(fmt.record()) is fmt
        assert fmt.accept({}) is fmt
        with pytest.raises(ValueError, match="version"):
            fmt.accept({"transform": {"version": 2, "kind": "signed_wht"}})
        with pytest.raises(ValueError, match="kind"):
            fmt.accept({"transform": {"version": 1, "kind": "other"}})
        with pytest.raises(ValueError):
            fmt.accept(
                {"transform": {"version": 1, "kind": "signed_wht", "block": 512}}
            )
        with pytest.raises(ValueError, match="unknown"):
            fmt.accept({"future": 1})


@pytest.mark.parametrize("bits", [4, 5])
def test_a_sorted_tile_takes_five_bit_experts(bits):
    k, n, experts = 640, 64, 8
    parts = [_group_case(k, n, bits, 128, seed=bits + e) for e in range(experts)]
    signs = parts[0][0]["oqh_signs"]
    tensors = {
        "weight": mx.stack([p[0]["weight"] for p in parts]),
        "oqh_scales": mx.stack([p[0]["oqh_scales"] for p in parts]),
        "oqh_biases": mx.stack([p[0]["oqh_biases"] for p in parts]),
        "oqh_signs": signs,
    }
    rotated = mx.stack([p[1] for p in parts])
    with testing.verified("example"):
        layer = FormatSwitchLinear(oqh.OQH_AFFINE, bits, 128, k, tensors)
    rows = 16 * experts
    mx.random.seed(0)
    x = mx.random.normal((rows, 1, k)).astype(mx.bfloat16)
    indices = mx.repeat(mx.arange(experts, dtype=mx.uint32), 16)
    got = layer(x, indices, sorted_indices=True)
    kept = layer.__dict__["_omlx_format_call"]
    assert (
        kept.sorted is not None
        and kept.sorted.proof.ran
        and not kept.sorted.proof.failed
    )
    x_rot = transforms.SignedWhtTransform(block=128).apply(x, signs)
    weight = rotated.astype(x.dtype).swapaxes(-1, -2)
    want = mx.stack([x_rot[i].reshape(1, k) @ weight[indices[i]] for i in range(rows)])
    assert _rel(got, want) < 3e-2


def test_the_faces_are_exact_for_the_dtypes_oqh_stores():
    for bits in (3, 4, 5):
        tensors, rotated = _row_case(640, 8, bits)
        layer = _layer(oqh.OQH, bits, 0, 640, tensors)
        face = oqh.OQH.dot(layer, 640, 8)
        assert face is not None and face.exact_for(mx.bfloat16)
        assert not face.exact_for(mx.float32)
        tile = oqh.OQH.tile(layer, 640, 8)
        assert tile is not None and tile.exact_for(mx.bfloat16)
        group, _ = _group_case(640, 8, bits, 128)
        affine_layer = _layer(oqh.OQH_AFFINE, bits, 128, 640, group)
        group_face = oqh.OQH_AFFINE.dot(affine_layer, 640, 8)
        assert group_face is not None and group_face.exact_for(mx.bfloat16)


def test_format_linear_fp32_falls_back_to_decode():
    formats.load()
    x = mx.ones((2, 128), mx.float32)
    row_tensors, row_rot = _row_case(128, 8, 4)
    row_layer = _layer(oqh.OQH, 4, 0, 128, row_tensors)
    x_row = transforms.SignedWhtTransform(block=128).apply(x, row_tensors["oqh_signs"])
    row_dec = oqh.OQH.decode(row_tensors, 4, 0).astype(mx.float32)
    assert _rel(row_layer(x), x_row @ row_dec.T) < 1e-5
    assert _rel(row_layer(x), x_row @ row_rot.T) < 1e-2
    grp_tensors, grp_rot = _group_case(128, 8, 4, 128)
    grp_layer = _layer(oqh.OQH_AFFINE, 4, 128, 128, grp_tensors)
    x_grp = transforms.SignedWhtTransform(block=128).apply(x, grp_tensors["oqh_signs"])
    grp_dec = oqh.OQH_AFFINE.decode(grp_tensors, 4, 128).astype(mx.float32)
    assert _rel(grp_layer(x), x_grp @ grp_dec.T) < 1e-5
    assert _rel(grp_layer(x), x_grp @ grp_rot.T) < 1e-2


def test_small_tensors_return_none_without_defects():
    with testing.no_dot_defects():
        row_tensors, _ = _row_case(128, 4, 4)
        row_layer = _layer(oqh.OQH, 4, 0, 128, row_tensors)
        assert formats.dot(row_layer, 128, 4) is None
        grp_tensors, _ = _group_case(128, 4, 4, 128)
        grp_layer = _layer(oqh.OQH_AFFINE, 4, 128, 128, grp_tensors)
        assert formats.dot(grp_layer, 128, 4) is None


def test_oqh_affine_reuses_affine_dense_tile_table():
    k, n, bits, gs = 2048, 2048, 4, 64
    chip = "applegpu_g15d_c80"
    weight = mx.zeros((n, k * bits // 32), mx.uint32)
    scales = mx.ones((n, k // gs), mx.bfloat16)
    biases = mx.zeros((n, k // gs), mx.bfloat16)
    signs = mx.ones((k // 128, 128), mx.bfloat16)
    oqh_layer = _layer(
        oqh.OQH_AFFINE,
        bits,
        gs,
        k,
        {
            "weight": weight,
            "oqh_scales": scales,
            "oqh_biases": biases,
            "oqh_signs": signs,
        },
    )
    aff_layer = _layer(
        affine.FORMAT,
        bits,
        gs,
        k,
        {"weight": weight, "scales": scales, "biases": biases},
    )
    oqh_plan = tile_dense.build_plan(oqh_layer, tensor_units=False, chip=chip)
    aff_plan = tile_dense.build_plan(aff_layer, tensor_units=False, chip=chip)
    assert oqh_plan is not None and aff_plan is not None
    # Every non-mlx cell tuned for affine must match oqh_affine's schedule.
    checked = 0
    for rows in (32, 64, 128, 256, 512, 1024, 4096):
        aff_pol = tile_dense.policy_for(aff_plan, rows, mx.bfloat16)
        if aff_pol is not None and aff_pol.backend != "mlx":
            oqh_pol = tile_dense.policy_for(oqh_plan, rows, mx.bfloat16)
            assert oqh_pol == aff_pol
            checked += 1
    assert checked > 0


def test_input_transform_validates_signs():
    for fmt in (oqh.OQH, oqh.OQH_AFFINE):
        assert fmt.input_transform({"oqh_signs": mx.zeros((4, 0), mx.bfloat16)}) is None
        assert (
            fmt.input_transform({"oqh_signs": mx.ones((1, 100), mx.bfloat16)}) is None
        )
        assert (
            fmt.input_transform({"oqh_signs": mx.zeros((0, 128), mx.bfloat16)}) is None
        )


def test_layout_and_accept_edge_cases():
    assert oqh.OQH.layout(0, 8, 4, 0) is None
    assert oqh.OQH.layout(128, 0, 4, 0) is None
    assert oqh.OQH_AFFINE.layout(0, 8, 4, 128) is None
    assert oqh.OQH_AFFINE.layout(128, 0, 4, 128) is None
    for fmt in (oqh.OQH, oqh.OQH_AFFINE):
        with pytest.raises(ValueError, match="block"):
            fmt.accept(
                {"transform": {"version": 1, "kind": "signed_wht", "block": 100}}
            )
        with pytest.raises(ValueError, match="unknown"):
            fmt.accept(
                {
                    "transform": {"version": 1, "kind": "signed_wht", "block": 256},
                    "future": 1,
                }
            )
        assert not hasattr(fmt, "block")
        accepted = fmt.accept(
            {"transform": {"version": 1, "kind": "signed_wht", "block": 256}}
        )
        assert accepted is not fmt and accepted.block == 256
        assert not hasattr(fmt, "block")


def test_row_codes_refuses_eight_bit():
    with pytest.raises(ValueError):
        oqh._row_codes(8, mx.bfloat16)


def test_accept_refuses_a_null_transform():
    for fmt in (oqh.OQH, oqh.OQH_AFFINE):
        with pytest.raises(ValueError):
            fmt.accept({"transform": None})


def test_accept_refuses_unknown_transform_fields():
    for fmt in (oqh.OQH, oqh.OQH_AFFINE):
        with pytest.raises(ValueError, match="unknown"):
            fmt.accept(
                {"transform": {"version": 1, "kind": "signed_wht", "block_size": 256}}
            )


def test_row_dot_separates_shared_source():
    for bits in _BITS:
        for fast in (False, True):
            face = oqh._row_dot(bits, fast, mx.bfloat16)
            assert face.shared != ""
            assert "decode_w" in face.shared
            assert "struct Decoded {" not in face.source
            assert "struct Decoded;" in face.source


@pytest.mark.parametrize("bits", _BITS)
@pytest.mark.parametrize("k", [128, 512])
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_row_dot_shared_rows_matches_dot_rows_bitwise(bits, k, rows):
    n = 16
    tensors, _ = _row_case(k, n, bits, seed=bits + rows)
    layer = _layer(oqh.OQH, bits, 0, k, tensors)
    face = oqh.OQH.dot(layer, k, n)
    assert face is not None
    mx.random.seed(100 + bits + rows)
    x = (mx.random.normal((rows, k)) * 0.1).astype(mx.bfloat16)
    operands = [tensors["weight"], tensors["rscale"]]
    shared_out = testing.shared_rows(face, operands, x, n, rps=4)
    single_out = mx.stack(
        [testing.dot_rows(face, operands, x[i], n) for i in range(rows)]
    )
    mx.eval(shared_out, single_out)
    assert mx.array_equal(shared_out.view(mx.uint16), single_out.view(mx.uint16)).item()


def test_row_tile_and_codes_keep_rscale_read_out_of_constructor():
    """Tile/Codes Rule 4 (docs/quant-formats.md): Runs(m, row) computes pointers
    only and never reads m.rscale[row] (FaceLoader and sg8 construct cursors
    for out-of-bounds or fetch-only rows)."""
    for bits in _BITS:
        tile_src = oqh._row_tile(bits, mx.bfloat16).source
        assert "const device S* r;" in tile_src
        assert "m.rscale[row]" not in tile_src
    codes_src = oqh._row_codes(4, mx.bfloat16).source
    assert "const device S* r;" in codes_src
    assert "m.rscale[row]" not in codes_src
    assert codes_src.count("r[0]") == 1
