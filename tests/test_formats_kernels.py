# SPDX-License-Identifier: Apache-2.0
"""The row skeleton over a format's dot face, with a toy format: rows equal
the dequantized products, and a face that breaks the contract fails to
compile with a message that names the format's own source.
"""

from __future__ import annotations

import re
from dataclasses import replace

import mlx.core as mx
import pytest

from omlx import formats
from omlx.formats import Dot, kernels, testing

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# w = (code - 128) * scale[row]: 8-bit codes, one float scale per row.
_TOY = r"""
namespace omlx_toy_KEY {
struct Dot {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* scales;

  template <int K>
  struct Lanes {
    static constant constexpr int VPT = 8;
    static constant constexpr int BLOCK = 32 * VPT;

    const device uint8_t* w;
    const device float* s;

    Lanes(Dot m, size_t row, uint lane)
        : w(m.codes + row * K + lane * VPT), s(m.scales + row) {}

    struct Row {
      const device uint8_t* w;
    };
    Row row(int r) const { return {w + r * K}; }
    void step() { w += BLOCK; }

    template <typename X, int... I>
    static void copy(omlx_ints<I...>, const device X* x, thread float* xt) {
      ((xt[I] = float(x[I])), ...);
    }
    template <int... I>
    static float dot(omlx_ints<I...>, const device uint8_t* w, thread const float* xt) {
      return (... + ((float(w[I]) - 128.0f) * xt[I]));
    }

    template <typename X>
    static float load(const device X* x, thread float* xt) {
      copy(omlx_iota<VPT>(), x, xt);
      return 0;
    }
    static float madd(float acc, Row r, thread const float* xt, float) {
      return acc + dot(omlx_iota<VPT>(), r.w, xt);
    }

    float finish(int r, float sum) const { return sum * s[r]; }
  };
};
}  // namespace omlx_toy_KEY
"""


def _toy(key, source=_TOY) -> Dot:
    return Dot(
        key=f"toy_{key}",
        source=source.replace("KEY", key),
        type=f"omlx_toy_{key}::Dot",
        tensors=("scales",),
    )


def _matrix(n, k, seed=0):
    """Codes, power-of-two row scales and small integer activations: every
    partial sum is exact in float32, whatever the order."""
    mx.random.seed(seed)
    codes = mx.random.randint(0, 256, (n, k)).astype(mx.uint8)
    scales = mx.array([2.0 ** (r % 5 - 2) for r in range(n)])
    x = mx.random.randint(-8, 9, (k,)).astype(mx.float32)
    want = ((codes.astype(mx.float32) - 128.0) @ x) * scales
    return (codes, scales), x, want


def _error(face, k=512) -> str:
    operands, x, _ = _matrix(8, k)
    with pytest.raises(RuntimeError) as failure:
        mx.eval(testing.dot_rows(face, operands, x, 8))
    return str(failure.value)


def _line(source, text) -> int:
    return source[: source.index(text)].count("\n") + 1


def test_a_matrix_is_its_weight_then_its_tensors():
    matrices = kernels.Matrices()
    assert matrices.add("sg", _toy("a")) == ["sg_w", "sg_scales"]
    assert matrices.add("rt", _toy("b")) == ["rt_w", "rt_scales"]
    assert matrices.declarations == (
        "    const auto sg = omlx_toy_a::Dot{(const device uint8_t*)sg_w, sg_scales};\n"
        "    const auto rt = omlx_toy_b::Dot{(const device uint8_t*)rt_w, rt_scales};\n"
        '\n#line 1 "<kernel>"\n'
    )


def test_a_kernel_has_each_input_once():
    matrices = kernels.Matrices()
    matrices.add("sg", _toy("a"))
    with pytest.raises(ValueError, match="matrix 'sg': sg_w, sg_scales would repeat"):
        matrices.add("sg", _toy("b"))
    # sg + d_scales and sg_d + scales are one input name.
    other = kernels.Matrices()
    other.add("sg", replace(_toy("a"), tensors=("d_scales",)))
    with pytest.raises(ValueError, match="matrix 'sg_d': sg_d_scales would repeat"):
        other.add("sg_d", _toy("b"))
    # ... and a tensor named w is the weight's.
    with pytest.raises(ValueError, match="matrix 'm': m_w would repeat"):
        kernels.Matrices().add("m", replace(_toy("a"), tensors=("w",)))
    # A refused matrix is not in the kernel.
    assert "omlx_toy_b" not in matrices.declarations + other.declarations
    assert other.declarations.count("const auto") == 1


def test_a_kernel_includes_each_format_source_once():
    matrices = kernels.Matrices()
    a, b = _toy("a"), _toy("b")
    for name, face in (("sg", a), ("su", a), ("rt", b)):
        matrices.add(name, face)
    header = matrices.header
    # The revisions come ahead of the first cited line.
    assert header.startswith(
        f"#define OMLX_CONTRACT {formats.CONTRACT}\n"
        f"#define OMLX_OLDEST_CONTRACT {formats.OLDEST_CONTRACT}\n"
        '\n#line 1 "<omlx skeletons, docs/quant-formats.md>"\n'
    )
    assert header.count(a.source) == header.count(b.source) == 1
    # Formats may use the index packs; the skeletons use the formats.
    assert (
        header.index("struct omlx_ints")
        < header.index(a.source)
        < header.index(b.source)
        < header.index("void qmv_rows(")
    )


def test_a_source_two_faces_share_is_cited_under_the_first():
    first = _toy("a")
    matrices = kernels.Matrices()
    matrices.add("sg", first)
    matrices.add("rt", replace(first, key="toy_other"))
    header = matrices.header
    assert header.count(first.source) == 1
    assert '#line 1 "<quant format toy_a>"' in header and "toy_other" not in header


def test_a_source_need_not_end_with_a_newline():
    face = _toy("a")
    matrices = kernels.Matrices()
    matrices.add("m", replace(face, source=face.source.rstrip("\n") + "  // the end"))
    marked = [line for line in matrices.header.splitlines() if "#line" in line]
    assert len(marked) == 4 and all(line.startswith("#line ") for line in marked)


# K as whole blocks of 256 inputs (512, 768), whole blocks then a last one
# that 16 of the 32 lanes have inputs in (384), and a last block only: of
# 8 lanes, 31 lanes, one lane.
@metal
@pytest.mark.parametrize("packs", [(1, 0), (0, 2), (1, 1), (2, 2), (4, 0), (3, 1)])
@pytest.mark.parametrize("k", [512, 768, 384, 64, 248, 8])
def test_rows_are_the_dequantized_products(k, packs):
    operands, x, want = _matrix(16, k, seed=k + sum(packs))
    got = testing.dot_rows(_toy("rows"), operands, x, 16, packs)
    assert mx.array_equal(got, want).item()


# The toy's rows with the codes as stored, re-centred in madd with the sum of
# the lane's inputs: what load returns is a struct of the face's own.
_LOAD = r"""
    template <typename X>
    static float load(const device X* x, thread float* xt) {
      copy(omlx_iota<VPT>(), x, xt);
      return 0;
    }
    static float madd(float acc, Row r, thread const float* xt, float) {
      return acc + dot(omlx_iota<VPT>(), r.w, xt);
    }
"""
_LOAD_STRUCT = r"""
    struct Inputs {
      float sum;
    };
    template <int... I>
    static float sum(omlx_ints<I...>, thread const float* xt) {
      return (... + xt[I]);
    }
    template <typename X>
    static Inputs load(const device X* x, thread float* xt) {
      copy(omlx_iota<VPT>(), x, xt);
      return {sum(omlx_iota<VPT>(), xt)};
    }
    static float madd(float acc, Row r, thread const float* xt, Inputs inputs) {
      return acc + (dot(omlx_iota<VPT>(), r.w, xt) - 128.0f * inputs.sum);
    }
"""


@metal
@pytest.mark.parametrize("k", [512, 384])
def test_what_load_returns_is_the_faces_own_type(k):
    assert _LOAD in _TOY
    source = _TOY.replace(_LOAD, _LOAD_STRUCT).replace(
        "(float(w[I]) - 128.0f) * xt[I]", "float(w[I]) * xt[I]"
    )
    operands, x, want = _matrix(16, k, seed=k)
    got = testing.dot_rows(_toy("aux", source=source), operands, x, 16, (2, 2))
    assert mx.array_equal(got, want).item()


@metal
def test_two_formats_run_in_one_kernel():
    a, b = _toy("first"), _toy("second")
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("a", a), *matrices.add("b", b)]
    kernel = mx.fast.metal_kernel(
        name="omlx_test_two_formats",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header,
        source=matrices.declarations + r"""
    const uint simd_lid = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    float result[2];
    qmv_rows<T, K, 1, 0>(a, row, a, row, x, simd_lid, result);
    qmv_rows<T, K, 1, 0>(b, row, b, row, x, simd_lid, result + 1);
    if (simd_lid == 0) {
      y[row] = result[0];
      y[N + row] = result[1];
    }
""",
    )
    (codes_a, scales_a), x, want_a = _matrix(8, 512, seed=1)
    (codes_b, scales_b), _, _ = _matrix(8, 512, seed=2)
    want_b = ((codes_b.astype(mx.float32) - 128.0) @ x) * scales_b
    got = kernel(
        inputs=[x, codes_a, scales_a, codes_b, scales_b],
        template=[("T", mx.float32), ("K", 512), ("N", 8)],
        grid=(32, 8, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(16,)],
        output_dtypes=[mx.float32],
    )[0]
    assert mx.array_equal(got, mx.concatenate([want_a, want_b])).item()


@metal
@pytest.mark.parametrize("k", [512, 384, 248, 8])
def test_the_two_packs_have_their_own_matrix_and_first_row(k):
    face = _toy("pair")
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("a", face), *matrices.add("b", face)]
    kernel = mx.fast.metal_kernel(
        name="omlx_test_two_matrices",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header,
        source=matrices.declarations + r"""
    const uint simd_lid = thread_index_in_simdgroup;
    // Rows of a from its first, rows of b from its last.
    const uint row_a = 2 * threadgroup_position_in_grid.y;
    const uint row_b = N - 2 - row_a;
    float result[4];
    qmv_rows<T, K, 2, 2>(a, row_a, b, row_b, x, simd_lid, result);
    if (simd_lid == 0) {
      y[row_a] = result[0];
      y[row_a + 1] = result[1];
      y[N + row_b] = result[2];
      y[N + row_b + 1] = result[3];
    }
""",
    )
    (codes_a, scales_a), x, want_a = _matrix(8, k, seed=1)
    (codes_b, scales_b), _, _ = _matrix(8, k, seed=2)
    want_b = ((codes_b.astype(mx.float32) - 128.0) @ x) * scales_b
    # What follows the K inputs is no lane's to read.
    padded = mx.concatenate([x, mx.full((256,), mx.nan)])
    got = kernel(
        inputs=[padded, codes_a, scales_a, codes_b, scales_b],
        template=[("T", mx.float32), ("K", k), ("N", 8)],
        grid=(32, 4, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(16,)],
        output_dtypes=[mx.float32],
    )[0]
    assert not mx.array_equal(want_a, want_b).item()
    assert mx.array_equal(got, mx.concatenate([want_a, want_b])).item()


@metal
def test_a_block_that_is_not_32_lanes_is_refused_in_words():
    source = _TOY.replace("BLOCK = 32 * VPT", "BLOCK = 16 * VPT")
    message = _error(_toy("half_block", source=source))
    assert (
        "dot face: Lanes<K>::BLOCK must be VPT inputs for each of a simdgroup's 32 lanes"
        in message
    )
    assert "<omlx skeletons, docs/quant-formats.md>:" in message


@metal
def test_a_k_that_is_not_whole_lanes_is_refused_in_words():
    message = _error(_toy("ragged"), k=60)
    assert "dot face: K must be a whole number of a lane's VPT inputs" in message


def _revision(contract) -> Dot:
    source = _TOY.replace("CONTRACT = 1;", f"CONTRACT = {contract};")
    return _toy(f"revision_{contract}", source=source)


@metal
def test_a_kernel_runs_the_faces_of_the_revisions_omlx_runs(monkeypatch):
    # As an oMLX of revision 3 that still runs the formats of revision 2.
    revisions = "#define OMLX_CONTRACT 3\n#define OMLX_OLDEST_CONTRACT 2\n"
    monkeypatch.setattr(kernels, "_REVISIONS", revisions)
    operands, x, want = _matrix(8, 512)
    for contract in (2, 3):
        got = testing.dot_rows(_revision(contract), operands, x, 8)
        assert mx.array_equal(got, want).item()
    for contract, rule in (
        (4, "a later revision of the contract than this oMLX runs"),
        (1, "a revision of the contract this oMLX no longer runs"),
    ):
        message = _error(_revision(contract))
        assert f"dot face: written for {rule}" in message
        assert "<omlx skeletons, docs/quant-formats.md>:" in message
        # The other rule holds, and is not cited.
        assert message.count("dot face: ") == 1


@metal
def test_a_face_that_states_no_revision_is_told_which_member_it_lacks():
    source = _TOY.replace("  static constant constexpr int CONTRACT = 1;\n", "")
    message = _error(_toy("no_revision", source=source))
    assert "no member named 'CONTRACT' in 'omlx_toy_no_revision::Dot'" in message


@metal
def test_an_error_in_a_format_cites_the_formats_own_line():
    source = _TOY.replace("return {w + r * K};", "return {weights + r * K};")
    message = _error(_toy("typo", source=source))
    assert f"<quant format toy_typo>:{_line(source, 'weights + r')}:" in message
    assert "undeclared identifier 'weights'" in message


@metal
def test_an_error_the_skeleton_brings_out_cites_the_formats_own_line():
    # Row is a member of Lanes<K>: its members are looked up for a given K.
    source = _TOY.replace("r.w, xt);", "r.weights, xt);")
    message = _error(_toy("member", source=source))
    assert f"<quant format toy_member>:{_line(source, 'r.weights')}:" in message
    assert "no member named 'weights'" in message
    # ... and the skeleton's line that asked for it.
    assert "<omlx skeletons, docs/quant-formats.md>:" in message
    assert "in instantiation of member function" in message


@metal
def test_a_row_that_is_not_const_is_refused():
    source = _TOY.replace("Row row(int r) const {", "Row row(int r) {")
    message = _error(_toy("mutable", source=source))
    assert "'row'" in message and "not marked const" in message


@metal
def test_a_missing_member_is_named_where_the_skeleton_needs_it():
    source = _TOY.replace(
        "    float finish(int r, float sum) const { return sum * s[r]; }\n", ""
    )
    message = _error(_toy("no_finish", source=source))
    assert "no member named 'finish'" in message
    assert "<omlx skeletons, docs/quant-formats.md>:" in message
    # ... and the kernel's call that asked for it.
    assert re.search(
        r"<kernel>:\d+:\d+: note: in instantiation of .*'qmv_rows<", message
    )


@metal
def test_an_error_in_a_kernel_body_cites_the_bodys_own_line():
    body = r"""
    const uint row = threadgroup_position_in_grid.y;
    float result[1];
    qmv_rows<T, K, 1, 0>(m, row, m, row, x, thread_index_in_simdgroup, result);
    y[row] = no_such_function(result[0]);
"""
    matrices = kernels.Matrices()
    kernel = mx.fast.metal_kernel(
        name="omlx_test_body_error",
        input_names=["x", *matrices.add("m", _toy("body"))],
        output_names=["y"],
        header=matrices.header,
        source=matrices.declarations + body,
    )
    operands, x, _ = _matrix(8, 512)
    with pytest.raises(RuntimeError) as failure:
        mx.eval(
            kernel(
                inputs=[x, *operands],
                template=[("T", mx.float32), ("K", 512)],
                grid=(32, 8, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(8,)],
                output_dtypes=[mx.float32],
            )
        )
    assert f"<kernel>:{_line(body, 'no_such_function')}:" in str(failure.value)


def test_dot_rows_refuses_what_the_launch_cannot_cover():
    operands, x, _ = _matrix(8, 512)
    for n in (7, 0, -2):
        with pytest.raises(ValueError, match=f"n={n} is not one launch or more of 2"):
            testing.dot_rows(_toy("a"), operands, x, n)
    half = replace(_toy("half"), dtypes=(mx.float16,))
    with pytest.raises(ValueError, match="toy_half is exact for mlx.core.float16"):
        testing.dot_rows(half, operands, x, 8)
    (codes, scales), x, _ = _matrix(4, 512)
    with pytest.raises(ValueError, match="toy_a: scales has 4 elements"):
        testing.dot_rows(_toy("a"), (codes, scales), x, 4)
    operands, x, _ = _matrix(8, 512)
    with pytest.raises(ValueError, match="toy_a: x has 4 elements"):
        testing.dot_rows(_toy("a"), operands, x[:4], 8)
    with pytest.raises(ValueError, match="weight, scales; operands given: 1"):
        testing.dot_rows(_toy("a"), operands[:1], x, 8)
    with pytest.raises(ValueError, match="one vector of K activations"):
        testing.dot_rows(_toy("a"), operands, mx.stack([x, x]), 8)
    for packs in [(0, 0), (-1, 2)]:
        with pytest.raises(ValueError, match="one row or more in all"):
            testing.dot_rows(_toy("a"), operands, x, 8, packs)
