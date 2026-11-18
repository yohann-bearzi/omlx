# SPDX-License-Identifier: Apache-2.0
"""The built-in affine format: its dot face follows MLX's one-row dispatch
and computes every row with MLX's bits.
"""

from __future__ import annotations

import re
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import affine, kernels, testing

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _linear(k, n, bits, group_size, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    linear = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    weight = (mx.random.normal((n, k)) * 0.05).astype(dtype)
    linear.weight, linear.scales, linear.biases = mx.quantize(
        weight, group_size=group_size, bits=bits
    )
    return linear


def _stacked(experts, k, n, bits, group_size, dtype=mx.bfloat16):
    """The parameters of a ``QuantizedSwitchLinear``: one matrix per expert."""
    layers = [_linear(k, n, bits, group_size, dtype, seed=e) for e in range(experts)]
    stacked = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    for name in ("weight", "scales", "biases"):
        stacked[name] = mx.stack([layer[name] for layer in layers])
    return stacked


def _operands(face, layer, expert=None):
    """A matrix's arrays in the order of its face, as a format's own tests
    take them (docs/quant-formats.md); those of one ``expert`` of a stack."""
    arrays = [layer[name] for name in ("weight", *face.tensors)]
    return arrays if expert is None else [a[expert] for a in arrays]


def _bit_equal(a, b):
    bits = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    return mx.array_equal(a.view(bits), b.view(bits)).item()


def _mlx_rows(layer, x, expert=None):
    """One row of activations through MLX: ``quantized_matmul``, or
    ``gather_qmm`` on ``expert`` of stacked weights."""
    tensors = (layer.weight, layer.scales, layer.biases)
    layout = {"transpose": True, "group_size": layer.group_size, "bits": layer.bits}
    if expert is None:
        return mx.quantized_matmul(x[None], *tensors, **layout)[0]
    return mx.gather_qmm(
        x[None, None],
        *tensors,
        rhs_indices=mx.array([expert], dtype=mx.uint32),
        **layout,
    )[0, 0]


# MLX 0.32.4 quantized.h, by bit width: the masks of qdot's branch and the
# powers of two that the branches of load_vector and qdot scale by.
MLX_MASKS = {
    2: {"0x03", "0x0c", "0x30", "0xc0"},
    3: {"0x01", "0x03", "0x07", "0x0e", "0x1c", "0x38", "0x70", "0x80", "0xc0", "0xe0"},
    4: {"0x000f", "0x00f0", "0x0f00", "0xf000"},
    5: {"0x1", "0x3", "0x7", "0xf", "0x1f", "0x3e", "0x7c", "0x80", "0xc0", "0xe0"}
    | {"0xf0", "0xf8"},
    6: {"0x03", "0x0f", "0x3f", "0xc0", "0xf0", "0xfc"},
    8: set(),
}
MLX_SCALES = {
    2: {"4.0f", "16.0f", "64.0f"},
    3: {"2.0f", "4.0f", "8.0f", "16.0f", "32.0f", "64.0f", "128.0f", "256.0f"},
    4: {"16.0f", "256.0f", "4096.0f"},
    5: {"2.0f", "4.0f", "8.0f", "16.0f", "32.0f", "64.0f", "128.0f", "256.0f"},
    6: {"4.0f", "16.0f", "64.0f", "256.0f"},
    8: set(),
}


def mlx_branches(metal: str) -> tuple[set, set]:
    """The masks and the scales of MLX's per-width branches that ``metal``
    holds."""
    masks = set(re.findall(r"\b0x[0-9a-f]+\b", metal))
    return masks, set(re.findall(r"\b\d+\.0f\b", metal))


def uncalled(metal: str) -> list[str]:
    """The free functions ``metal`` defines and names nowhere else (a face's
    members are the skeletons' to call)."""
    code = re.sub(r"//[^\n]*", "", metal)
    defined = re.findall(r"^(?:inline|METAL_FUNC) [\w:<>]+ (\w+)\(", code, re.M)
    return [name for name in defined if len(re.findall(rf"\b{name}\b", code)) < 2]


def test_affine_ships_as_a_builtin_plugin():
    assert "omlx.formats.affine:FORMAT" in formats.BUILTIN
    assert formats.get("affine") is affine.FORMAT
    assert affine.FORMAT.tensors == ("scales", "biases")


def test_affine_states_one_revision_in_its_format_and_its_faces():
    face = affine.FORMAT.dot(_linear(512, 8, 4, 64), 512, 8)
    stated = re.findall(r"\bCONTRACT = (\d+);", face.source)
    assert stated == [str(affine.FORMAT.contract)]


@pytest.mark.parametrize(
    "k, bits, fast",
    [
        (512, 2, True),
        (512, 3, True),
        (512, 4, True),
        (512, 5, True),
        (256, 6, True),
        (256, 8, True),
        (256, 2, False),  # MLX's qmv_fast block is 512 wide at 2 to 5 bits
        (256, 3, False),
        (256, 4, False),
        (256, 5, False),
        (384, 6, False),
        (384, 8, False),
    ],
)
def test_the_face_follows_mlx_one_row_dispatch(k, bits, fast):
    face = formats.dot(_linear(k, 16, bits, 64), k, 16)
    assert face == affine.face(bits, 64, fast, mx.bfloat16)
    assert (face.dtypes, face.tensors) == ((mx.bfloat16,), ("scales", "biases"))


def test_stacked_weights_follow_gather_qmm_dispatch():
    # Gathered QMV is fast only for whole output tiles of 8 rows.
    fast, slow = (affine.face(4, 64, fast, mx.bfloat16) for fast in (True, False))
    assert formats.dot(_stacked(2, 512, 8, 4, 64), 512, 8) is fast
    assert formats.dot(_stacked(2, 512, 16, 4, 64), 512, 16) is fast
    assert formats.dot(_stacked(2, 512, 24, 4, 64), 512, 24) is fast
    assert formats.dot(_stacked(2, 512, 12, 4, 64), 512, 12) is slow
    assert formats.dot(_linear(512, 12, 4, 64), 512, 12) is fast


def test_each_layout_is_one_face():
    a = formats.dot(_linear(512, 16, 4, 64, seed=1), 512, 16)
    b = formats.dot(_linear(512, 32, 4, 64, seed=2), 512, 32)
    assert a is b
    assert a != formats.dot(_linear(512, 16, 4, 64, mx.float16), 512, 16)
    keys = {
        affine.face(bits, group_size, fast, mx.bfloat16).key
        for bits in (2, 3, 4, 5, 6, 8)
        for group_size in (32, 64, 128)
        for fast in (True, False)
    }
    assert len(keys) == 36


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_a_face_s_source_holds_its_own_bit_width_alone(bits, fast):
    """MLX hashes a kernel's whole source at every launch: a face holds the
    branch of MLX's code for its width and the functions its kernels call."""
    face = affine.face(bits, 64, fast, mx.bfloat16)
    source = face.source
    assert mlx_branches(source) == (MLX_MASKS[bits], MLX_SCALES[bits])
    # load_vector and qdot walk the lane's values once each.
    assert source.count("for (") == 2
    assert uncalled(source) == []
    # decode_w and wdot, qdot split in two, are in the shared source alone:
    # the kernels that apply a block to several inputs include it.
    masks, scales = mlx_branches(face.shared)
    assert masks == MLX_MASKS[bits] and scales <= MLX_SCALES[bits]
    assert face.shared.count("for (") == 2 and "decode_w" not in source
    assert uncalled(face.shared) == []


@pytest.mark.parametrize(
    "bits, dtype, k, n",
    [
        (1, mx.bfloat16, 512, 16),  # bits the transcription lacks
        (4, mx.float32, 512, 16),
        (4, mx.bfloat16, 256, 16),  # not the matrix asked for
        (4, mx.bfloat16, 512, 8),
        (5, mx.bfloat16, 513, 16),  # the same words of codes, not whole groups
        (5, mx.bfloat16, 516, 16),
    ],
)
def test_no_face_outside_what_the_transcription_covers(bits, dtype, k, n):
    assert affine.FORMAT.dot(_linear(512, 16, bits, 64, dtype), k, n) is None


@pytest.mark.parametrize("bits, group_size", [(1, 64), (7, 64), (4, 16), (4, 256)])
def test_a_face_built_by_hand_is_one_the_transcription_has(bits, group_size):
    with pytest.raises(ValueError, match="no MLX transcription"):
        affine.face(bits, group_size, False, mx.bfloat16)


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("bits, group_size", [(7, 64), (4, 8), (4, 256)])
def test_a_header_is_one_the_transcription_has(bits, group_size, fast):
    """A kernel built over the header alone (the row-exact tiles) is refused
    as a face is: not with a KeyError, nor with a group of no lanes."""
    with pytest.raises(ValueError, match="no MLX transcription"):
        affine.header(bits, group_size, fast)


def test_no_face_for_a_group_size_mlx_does_not_quantize_to():
    linear = _linear(512, 16, 4, 64)
    linear.group_size = 16
    assert affine.FORMAT.dot(linear, 512, 16) is None


def test_no_face_without_scales_and_biases_of_one_dtype():
    linear = _linear(512, 16, 4, 64)
    linear.biases = linear.biases.astype(mx.float16)
    assert affine.FORMAT.dot(linear, 512, 16) is None
    linear.pop("biases")
    assert affine.FORMAT.dot(linear, 512, 16) is None
    linear = _linear(512, 16, 4, 64)
    linear.pop("scales")
    assert affine.FORMAT.dot(linear, 512, 16) is None


def test_no_face_for_tensors_mlx_binds_as_constants():
    # One row of 5 groups, as a small shared expert's gate: 5 scales.
    assert affine.FORMAT.dot(_linear(640, 1, 6, 128), 640, 1) is None
    assert affine.FORMAT.dot(_linear(896, 1, 6, 128), 896, 1) is None
    assert formats.dot(_linear(1024, 1, 6, 128), 1024, 1) is not None
    # Counted in one expert: this stack of two has 10.
    assert affine.FORMAT.dot(_stacked(2, 640, 1, 6, 128), 640, 1) is None


@metal
def test_a_matrix_of_8_groups_is_bound_as_buffers():
    # MLX's own threshold: 8 scales, 8 biases.
    linear = _linear(1024, 1, 6, 128)
    x = mx.random.normal((1024,)).astype(mx.bfloat16)
    face = formats.dot(linear, 1024, 1)
    got = testing.dot_rows(face, _operands(face, linear), x, 1, packs=(1, 0))
    assert _bit_equal(got, _mlx_rows(linear, x))


_QUAD = [(k, bits) for k in (64, 128) for bits in (2, 4, 8)]


def test_no_face_where_mlx_runs_qmv_quad():
    for k, bits in _QUAD:
        assert affine.FORMAT.dot(_linear(k, 16, bits, 32), k, 16) is None
        # gather_qmm has no such kernel.
        assert formats.dot(_stacked(2, k, 16, bits, 32), k, 16) is not None
    # Only K 64 and 128, only 2, 4 and 8 bits.
    others = [(32, 4), (96, 8), (64, 5), (128, 6), (192, 4), (256, 8)]
    for k, bits in [*others, (64, 3), (128, 3), (192, 2)]:
        assert formats.dot(_linear(k, 16, bits, 32), k, 16) is not None


def test_no_face_for_weights_that_are_not_a_matrix_or_a_stack_of_them():
    stacked = _stacked(2, 512, 16, 4, 64)
    assert formats.dot(stacked, 512, 16) is not None
    for name in ("weight", "scales", "biases"):
        stacked[name] = stacked[name][None]
    assert affine.FORMAT.dot(stacked, 512, 16) is None


# At 2 bits and K 64 a lane of qmv holds 16 inputs, as a lane of qmv_quad
# does, and the four partial sums meet in one simd_sum or quad_sum: the bits
# came out equal here. No face either: what equals MLX is its dispatch.
@metal
@pytest.mark.parametrize("k, bits", [quad for quad in _QUAD if quad != (64, 2)])
def test_qmv_quad_is_not_the_qmv_sum(k, bits):
    # Why such a matrix has no face: built by hand, the qmv face gives it the
    # right products, summed in another order than MLX's.
    layer = _linear(k, 16, bits, 32, mx.float32)
    x = mx.random.normal((k,))
    face = affine.face(bits, 32, False, mx.float32)
    got = testing.dot_rows(face, _operands(face, layer), x, 16, packs=(2, 2))
    want = _mlx_rows(layer, x)
    assert not _bit_equal(got, want)
    assert mx.allclose(got, want, rtol=1e-4, atol=1e-5).item()


@metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("k", [512, 288])  # qmv_fast's whole blocks; qmv's last one
def test_decode_then_apply_gives_madd_s_rows(k, bits, dtype):
    """A block's weights decoded once, then applied to the lane's inputs,
    give the rows madd gives, bit for bit: MLX's, in fp32 too."""
    layer = _linear(k, 16, bits, 32, dtype)
    face = replace(formats.dot(_linear(k, 16, bits, 32), k, 16), dtypes=(dtype,))
    x = mx.random.normal((k,)).astype(dtype)
    operands = _operands(face, layer)
    decoded = testing.shared_rows(face, operands, x[None], 16)[0]
    assert _bit_equal(decoded, testing.dot_rows(face, operands, x, 16, packs=(1, 0)))
    if dtype == mx.float32:
        assert _bit_equal(decoded, _mlx_rows(layer, x))


@metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("k", [512, 288])  # qmv_fast's whole blocks; qmv's last one
def test_shared_rows_give_each_vector_its_one_row_bits(k, bits, dtype):
    """Vectors through the shared-rows skeleton, each block's weights decoded
    once: every vector's row is its one-row call's, bit for bit (in fp32,
    MLX's), in packs of 1, 2 or 4 rows, N whole packs or not."""
    layer = _linear(k, 16, bits, 32, dtype)
    face = replace(formats.dot(_linear(k, 16, bits, 32), k, 16), dtypes=(dtype,))
    assert face.vpt == affine.values_per_thread(bits, k % 512 == 0)
    operands = _operands(face, layer)
    xs = mx.random.normal((5, k)).astype(dtype)
    one = [testing.dot_rows(face, operands, x, 16, packs=(1, 0)) for x in xs]
    for rows, rps, n in ((1, 1, 16), (5, 4, 10), (5, 2, 16)):
        got = testing.shared_rows(face, operands, xs[:rows], n, rps=rps)
        for i in range(rows):
            assert _bit_equal(got[i], one[i][:n])
    if dtype == mx.float32:
        assert all(_bit_equal(one[i], _mlx_rows(layer, xs[i])) for i in range(5))


@metal
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_unrolled_shared_rows_inside_the_envelope_give_the_same_bits(bits):
    layer = _linear(512, 16, bits, 64)
    face = formats.dot(layer, 512, 16)
    operands = _operands(face, layer)
    xs = mx.random.normal((4, 512)).astype(mx.bfloat16)
    one = [testing.dot_rows(face, operands, x, 16, packs=(1, 0)) for x in xs]
    for rows, rps in ((1, 4), (2, 4), (4, 4), (4, 2)):
        if not kernels.unrolled_tile_ok(face.vpt, rps, rows):
            continue
        got = testing.shared_rows(face, operands, xs[:rows], 16, rps=rps, unrolled=True)
        assert all(_bit_equal(got[i], one[i]) for i in range(rows))


@metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("k", [512, 288])  # qmv_fast's whole blocks; qmv's last one
def test_inputs_staged_in_threadgroup_memory_give_the_same_rows(k, bits, dtype):
    """A kernel may stage its inputs in threadgroup memory: the face loads
    them from there and gives the rows it gives from device memory, MLX's."""
    layer = _linear(k, 16, bits, 32, dtype)
    # A float32 layer's face: the traversal's, on fp32 tensors (above).
    face = replace(formats.dot(_linear(k, 16, bits, 32), k, 16), dtypes=(dtype,))
    x = mx.random.normal((k,)).astype(dtype)
    operands = _operands(face, layer)
    staged = testing.dot_rows(face, operands, x, 16, packs=(2, 2), staged=True)
    assert _bit_equal(staged, testing.dot_rows(face, operands, x, 16, packs=(2, 2)))
    if dtype == mx.float32:
        assert _bit_equal(staged, _mlx_rows(layer, x))


@metal
@pytest.mark.parametrize(
    "bits, k, n, stacked",
    [
        (bits, k, n, stacked)
        for stacked in (False, True)
        for n in (16, 12)  # whole output tiles of 8 rows, or not
        # Around qmv_quad's K (64, 128), then as in test_rows_have_mlx_bits.
        for k in (32, 64, 96, 128, 256, 288, 384, 448, 512, 1024, 1280)
        for bits in (2, 3, 4, 5, 6, 8)
        if stacked or (k, bits) not in _QUAD
    ],
)
def test_the_traversal_sums_in_mlx_order(bits, k, n, stacked):
    """Rounded to bf16 or fp16, a row summed in another order than MLX's
    differs once in a thousand or less often; in fp32 most rows do. So the
    traversal ``dot`` gives a layout runs here on fp32 tensors, against MLX's
    fp32 rows."""
    f32 = mx.float32
    if stacked:
        face = replace(formats.dot(_stacked(2, k, n, bits, 32), k, n), dtypes=(f32,))
        layer = _stacked(2, k, n, bits, 32, f32)
        x = mx.random.normal((k,))
        for expert in range(2):
            operands = _operands(face, layer, expert)
            got = testing.dot_rows(face, operands, x, n, packs=(2, 2))
            assert _bit_equal(got, _mlx_rows(layer, x, expert))
    else:
        face = replace(formats.dot(_linear(k, n, bits, 32), k, n), dtypes=(f32,))
        layer = _linear(k, n, bits, 32, f32)
        x = mx.random.normal((k,))
        got = testing.dot_rows(face, _operands(face, layer), x, n, packs=(2, 2))
        assert _bit_equal(got, _mlx_rows(layer, x))


@metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "bits, group_size, k",
    [
        (bits, group_size, k)
        for bits in (2, 3, 4, 5, 6, 8)
        for group_size in (32, 64, 128)
        # qmv_fast where K is whole blocks of 512 (2 to 5 bits) or of 256 (6
        # and 8 bits): one block and more. Elsewhere qmv, in blocks of half
        # that (2 bits: of 512 as well): a partial block alone (from 96),
        # whole blocks alone, whole blocks and a partial one (640 at 3 to 5
        # bits: two and a half).
        for k in (96, 128, 192, 256, 288, 384, 448, 512, 640, 1024, 1280)
        if k % group_size == 0 and (k, bits) not in _QUAD
    ],
)
def test_rows_have_mlx_bits(bits, group_size, k, dtype):
    n = 16
    linear = _linear(k, n, bits, group_size, dtype, seed=bits + group_size + k)
    x = mx.random.normal((k,)).astype(dtype)
    face = formats.dot(linear, k, n)
    got = testing.dot_rows(face, _operands(face, linear), x, n, packs=(2, 2))
    assert _bit_equal(got, _mlx_rows(linear, x))


@metal
@pytest.mark.parametrize("n", [16, 24, 12])  # whole output tiles of 8 rows, or not
@pytest.mark.parametrize("k", [512, 384, 128])
@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_stacked_rows_have_gather_qmm_bits(bits, k, n):
    stacked = _stacked(3, k, n, bits, 64)
    x = mx.random.normal((k,)).astype(mx.bfloat16)
    face = formats.dot(stacked, k, n)
    for expert in range(3):
        operands = _operands(face, stacked, expert)
        got = testing.dot_rows(face, operands, x, n, packs=(2, 2))
        assert _bit_equal(got, _mlx_rows(stacked, x, expert))


@metal
@pytest.mark.parametrize("k", [512, 384])
@pytest.mark.parametrize("bits", [4, 8])
def test_a_matrix_that_is_not_whole_output_tiles_has_mlx_bits(bits, k):
    linear = _linear(k, 12, bits, 64, seed=k)
    x = mx.random.normal((k,)).astype(mx.bfloat16)
    face = formats.dot(linear, k, 12)
    got = testing.dot_rows(face, _operands(face, linear), x, 12, packs=(2, 2))
    assert _bit_equal(got, _mlx_rows(linear, x))


@metal
@pytest.mark.parametrize("packs", [(1, 0), (1, 1), (4, 0), (3, 1), (4, 4)])
@pytest.mark.parametrize("k", [512, 384])
def test_a_row_does_not_depend_on_the_rows_it_runs_with(k, packs):
    linear = _linear(k, 16, 5, 64, seed=k)
    x = mx.random.normal((k,)).astype(mx.bfloat16)
    face = formats.dot(linear, k, 16)
    operands = _operands(face, linear)
    assert _bit_equal(
        testing.dot_rows(face, operands, x, 16, packs),
        testing.dot_rows(face, operands, x, 16, packs=(2, 2)),
    )


def test_the_face_is_exact_for_its_own_dtype_only():
    linear = _linear(512, 16, 4, 64)
    face = formats.dot(linear, 512, 16)
    with pytest.raises(ValueError, match="is exact for mlx.core.bfloat16"):
        testing.dot_rows(
            face, _operands(face, linear), mx.zeros((512,), mx.float16), 16
        )


@metal
def test_the_fast_face_refuses_a_k_that_is_not_whole_blocks():
    # AffineFormat.dot gives K = 384 the qmv face, as MLX runs it. Built by
    # hand, the qmv_fast one would sum each row in qmv_fast's lanes: the
    # right product, not MLX's bits.
    linear = _linear(384, 16, 4, 64)
    face = affine.face(4, 64, True, mx.bfloat16)
    rows = testing.dot_rows(
        face, _operands(face, linear), mx.zeros((384,), mx.bfloat16), 16
    )
    with pytest.raises(RuntimeError, match="affine qmv_fast face: K must be whole"):
        mx.eval(rows)
