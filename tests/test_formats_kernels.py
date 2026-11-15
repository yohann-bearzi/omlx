# SPDX-License-Identifier: Apache-2.0
"""The row skeleton over a format's dot face, with a toy format: rows equal
the dequantized products, and a face that breaks the contract fails to
compile with a message that names the format's own source.
"""

from __future__ import annotations

import re
from dataclasses import replace
from functools import cache
from itertools import takewhile
from pathlib import Path
from textwrap import indent

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


def test_a_proof_is_shared_by_what_is_compiled_from_the_same_key(monkeypatch):
    monkeypatch.setattr(kernels, "_PROOFS", {})
    face = _toy("proof")
    proof = kernels.proof("kernel", face, mx.float32, 512)
    assert proof == kernels.Proof(ran=set(), failed=False)
    assert kernels.proof("kernel", _toy("proof"), mx.float32, 512) is proof
    others = (
        kernels.proof("other", face, mx.float32, 512),
        kernels.proof("kernel", _toy("other"), mx.float32, 512),
        kernels.proof("kernel", face, mx.float16, 512),
        kernels.proof("kernel", face, mx.float32, 384),
    )
    proof.ran.add("launch")
    proof.failed = True
    assert all(other == kernels.Proof() and other is not proof for other in others)
    assert len(kernels._PROOFS) == 5


def test_a_matrix_is_its_weight_then_its_tensors():
    matrices = kernels.Matrices()
    assert matrices.add("sg", _toy("a")) == ["sg_weight", "sg_scales"]
    assert matrices.add("rt", _toy("b")) == ["rt_weight", "rt_scales"]
    assert matrices.declarations == (
        "    const auto sg = omlx_toy_a::Dot{"
        "(const device uint8_t*)sg_weight, sg_scales};\n"
        "    const auto rt = omlx_toy_b::Dot{"
        "(const device uint8_t*)rt_weight, rt_scales};\n"
        '\n#line 1 "<kernel>"\n'
    )


def test_a_kernel_has_each_name_once():
    matrices = kernels.Matrices()
    matrices.add("sg", _toy("a"))
    with pytest.raises(ValueError) as refusal:
        matrices.add("sg", _toy("b"))
    assert str(refusal.value) == (
        "matrix 'sg': sg, sg_weight, sg_scales would repeat a name of the kernel"
    )
    # sg + d_scales and sg_d + scales are one input name.
    other = kernels.Matrices()
    other.add("sg", replace(_toy("a"), tensors=("d_scales",)))
    with pytest.raises(ValueError, match="matrix 'sg_d': sg_d_scales would repeat"):
        other.add("sg_d", _toy("b"))
    # A matrix is declared under its name: no input has it, whichever of the
    # two comes first.
    with pytest.raises(ValueError, match="matrix 'sg_d_scales': sg_d_scales would"):
        other.add("sg_d_scales", _toy("b"))
    first = kernels.Matrices()
    first.add("rt_scales", _toy("a"))
    with pytest.raises(ValueError, match="matrix 'rt': rt_scales would repeat"):
        first.add("rt", _toy("b"))
    # A refused matrix is not in the kernel, and its names are not taken.
    assert "omlx_toy_b" not in matrices.declarations + other.declarations
    assert other.declarations.count("const auto") == 1
    assert other.add("sg_d_weight", _toy("b"))[0] == "sg_d_weight_weight"


@pytest.mark.parametrize("name", ["", "2m", "m w", "m-w", "m\u00e9", "m;", 3, None])
def test_a_matrix_is_named_by_an_identifier(name):
    # The name is spliced into the kernel's Metal.
    matrices = kernels.Matrices()
    with pytest.raises(ValueError) as refusal:
        matrices.add(name, _toy("a"))
    assert str(refusal.value) == f"matrix {name!r}: its name is not an identifier"
    assert matrices.add("m", _toy("a")) == ["m_weight", "m_scales"]


def test_a_kernel_includes_each_format_source_once():
    matrices = kernels.Matrices()
    a, b = _toy("a"), _toy("b")
    for name, face in (("sg", a), ("su", a), ("rt", b)):
        matrices.add(name, face)
    header = matrices.header
    assert header.startswith('\n#line 1 "<omlx skeletons, docs/quant-formats.md>"\n')
    assert header.count(a.source) == header.count(b.source) == 1
    # Formats may use the index packs; the skeletons use the formats.
    assert (
        header.index("struct omlx_ints")
        < header.index(a.source)
        < header.index(b.source)
        < header.index("void qmv_rows(")
    )


def test_a_kernel_has_the_revisions_omlx_runs_when_it_is_built(monkeypatch):
    monkeypatch.setattr(formats, "CONTRACT", 3)
    monkeypatch.setattr(formats, "OLDEST_CONTRACT", 2)
    a, b = _toy("a"), _toy("b")
    matrices = kernels.Matrices()
    matrices.add("m", a)
    matrices.add("n", b)
    header = matrices.header
    # The skeletons' first lines, cited as the skeletons'.
    revisions = (
        f'\n#line {kernels._INDEX_PACKS.count(chr(10)) + 1} "{kernels._SKELETONS}"\n'
        "#define OMLX_CONTRACT 3\n#define OMLX_OLDEST_CONTRACT 2\n"
    )
    assert header.count("#define") == 2
    # They follow the formats' sources, none of which reads them, and end
    # with the skeletons: nor does a face's type, or the kernel.
    assert (
        max(header.index(a.source), header.index(b.source))
        < header.index(revisions)
        < header.index("void qmv_rows(")
    )
    assert header.endswith(
        "}\n#undef OMLX_CONTRACT\n#undef OMLX_OLDEST_CONTRACT\n"
        '\n#line 1 "<kernel header>"\n'
    )


@metal
def test_a_faces_type_does_not_read_the_revisions():
    operands, x, want = _matrix(8, 512)
    typed = replace(_toy("typed"), type="decltype(1, omlx_toy_typed::Dot{})")
    assert mx.array_equal(testing.dot_rows(typed, operands, x, 8), want).item()
    reads = replace(
        typed,
        key="toy_reads",
        type="decltype(OMLX_CONTRACT, omlx_toy_typed::Dot{})",
    )
    assert "use of undeclared identifier 'OMLX_CONTRACT'" in _error(reads)


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
    marked = [line for line in matrices.header.splitlines() if "#" in line]
    # Four #line, the two revisions and their two ends, each on a line of
    # its own.
    assert len(marked) == 8 and all(line.startswith("#") for line in marked)


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


# A face's kernel is built once, under the revisions oMLX runs then: these
# keys are for the test that patches them.
def _revision(contract) -> Dot:
    source = _TOY.replace("CONTRACT = 1;", f"CONTRACT = {contract};")
    return _toy(f"revision_{contract}", source=source)


@metal
def test_a_kernel_runs_the_faces_of_the_revisions_omlx_runs(monkeypatch):
    # As an oMLX of revision 3 that still runs the formats of revision 2.
    monkeypatch.setattr(formats, "CONTRACT", 3)
    monkeypatch.setattr(formats, "OLDEST_CONTRACT", 2)
    operands, x, want = _matrix(8, 512)
    for contract in (2, 3):
        got = testing.dot_rows(_revision(contract), operands, x, 8)
        assert mx.array_equal(got, want).item()
    for contract, rule, revisions in (
        (4, "a later revision of the contract than this oMLX runs", "'4 <= 3'"),
        (1, "a revision of the contract this oMLX no longer runs", "'1 >= 2'"),
    ):
        message = _error(_revision(contract))
        assert f"dot face: written for {rule}" in message
        # The face's revision against oMLX's, in numbers.
        assert f"due to requirement {revisions}" in message
        assert "<omlx skeletons, docs/quant-formats.md>:" in message
        # The other rule holds, and is not cited. Nor is a format's source:
        # the revisions are lines of the skeletons.
        assert message.count("dot face: ") == 1
        assert "<quant format" not in message


@metal
@pytest.mark.parametrize(
    "contract", [formats.CONTRACT + 1, formats.OLDEST_CONTRACT - 1]
)
def test_a_face_of_another_revision_is_told_that_and_nothing_else(contract):
    # A face of another revision as this oMLX may meet one: without a member
    # it calls, and with a layout it refuses.
    source = (
        _TOY.replace("CONTRACT = 1;", f"CONTRACT = {contract};")
        .replace("BLOCK = 32 * VPT;", "BLOCK = 16 * VPT;")
        .replace(
            "    float finish(int r, float sum) const { return sum * s[r]; }\n", ""
        )
    )
    assert "finish" not in source and "BLOCK = 16 * VPT;" in source
    message = _error(_toy(f"another_{contract}", source=source))
    assert "dot face: written for a" in message
    assert message.count("error:") == 1


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
    (codes, scales), x, _ = _matrix(7, 512)
    with pytest.raises(ValueError, match="toy_a: scales has 7 elements"):
        testing.dot_rows(_toy("a"), (codes, scales), x, 7, (1, 0))
    operands, x, _ = _matrix(8, 512)
    with pytest.raises(ValueError, match="toy_a: x has 4 elements"):
        testing.dot_rows(_toy("a"), operands, x[:4], 8)
    with pytest.raises(ValueError, match="weight, scales; operands given: 1"):
        testing.dot_rows(_toy("a"), operands[:1], x, 8)
    with pytest.raises(ValueError, match="one vector of K activations"):
        testing.dot_rows(_toy("a"), operands, mx.stack([x, x]), 8)
    for packs in [(0, 0), (-1, 2), (2, -1)]:
        with pytest.raises(ValueError, match="one row or more in all"):
            testing.dot_rows(_toy("a"), operands, x, 8, packs)


# docs/quant-formats.md shows a format and what the compiler says of its
# mistakes. The example is the toy above; each message is the compiler's for
# one edit of it.
_PAGE = (Path(__file__).parents[1] / "docs" / "quant-formats.md").read_text()


def _fenced(page) -> list[tuple[str, str]]:
    """The page's fenced blocks, each as its language and its text."""
    lines = iter(page.splitlines())
    return [
        (
            line.strip()[3:],
            "\n".join(takewhile(lambda inner: inner.strip() != "```", lines)),
        )
        for line in lines
        if line.strip().startswith("```")
    ]


_BLOCKS = _fenced(_PAGE)
# What the page quotes of the Metal compiler, on one line each: the blocks
# that cite a line and its column, whatever they are fenced as.
_SAMPLES = [
    " ".join(text.split()) for _, text in _BLOCKS if re.search(r">:\d+:\d+: ", text)
]
# What pins a sample: the lines it cites with the kind of each message, the
# rules oMLX words and the revisions of a failed requirement. The rest is the
# Metal compiler's wording.
_LINE = r"<[^<>]*>:\d+:\d+: \w+"
_RULE = r'"dot face: [^"]*"'
_REVISIONS = r"requirement '\d+ [<>]= \d+'"
# A phrase of each quoted message, and what brings the message out of the
# page's example: K, and an edit of the example's source.
_BREAKS = {
    "BLOCK must be VPT inputs": {
        "old": "BLOCK = 32 * VPT;",
        "new": "BLOCK = 16 * VPT;",
    },
    "K must be a whole number": {"k": 60},
    "undeclared identifier 'weights'": {
        "old": "return {w + r * K};",
        "new": "return {weights + r * K};",
    },
    "no member named 'finish'": {
        "old": "    float finish(int r, float sum) const { return sum * s[r]; }\n",
    },
    "too many arguments": {
        "old": "thread const float* xt, float) {",
        "new": "thread const float* xt) {",
    },
    "a later revision": {
        "old": "CONTRACT = 1;",
        "new": f"CONTRACT = {formats.CONTRACT + 1};",
    },
}


@cache
def _example() -> Dot:
    """The face of the page's example."""
    (source,) = (
        text
        for language, text in _BLOCKS
        if language == "cpp" and text.startswith("namespace omlx_myformat_b8 {")
    )
    (face,) = (
        text for language, text in _BLOCKS if language == "python" and "SOURCE" in text
    )
    return eval(face, {"Dot": Dot, "SOURCE": source})


def _page_error(k=512, old="", new="") -> str:
    """The compiler's messages for the page's example over K inputs, with
    ``old`` in its source replaced by ``new``."""
    face = _example()
    assert old in face.source
    return _error(replace(face, source=face.source.replace(old, new)), k)


def test_the_pages_example_is_the_toy_format():
    face = _example()
    assert face == Dot(
        key="myformat_b8",
        source=_TOY.strip().replace("omlx_toy_KEY", "omlx_myformat_b8"),
        type="omlx_myformat_b8::Dot",
        tensors=("scales",),
    )


@metal
def test_the_pages_kernel_computes_its_examples_rows():
    (kernel,) = (
        text
        for language, text in _BLOCKS
        if language == "python" and "qmv_rows<" in text
    )
    (weight, scales), x, want = _matrix(8, 512)
    scope = dict(face=_example(), weight=weight, scales=scales, x=x, k=512, n=8)
    exec(kernel, scope)
    assert f"# {scope['inputs']!r}".replace("'", '"') in kernel
    assert mx.array_equal(scope["y"], want).item()


def test_the_pages_fold_is_its_examples():
    (fold,) = (
        text
        for language, text in _BLOCKS
        if language == "cpp" and "// dot(omlx_iota<VPT>(), w, xt)" in text
    )
    function = fold[: fold.rindex("\n") + 1]
    assert indent(function, "    ") in _example().source


def test_the_page_names_the_entry_point_group():
    # In the plugin's pyproject, under it, and in what the contract holds.
    assert _PAGE.count(formats.ENTRY_POINT_GROUP) == 3


def test_each_compiler_message_of_the_page_has_its_break():
    # One sample for each break and one break for each sample.
    assert [sum(p in s for s in _SAMPLES) for p in _BREAKS] == [1] * len(_BREAKS)
    assert [sum(p in s for p in _BREAKS) for s in _SAMPLES] == [1] * len(_SAMPLES)


@pytest.mark.parametrize("sample", _SAMPLES)
def test_each_citation_of_a_sample_is_one_the_tests_read(sample):
    # One that lost its column, or a rule that lost a quote, would not be.
    assert len(re.findall(_LINE, sample)) == len(re.findall(r">:\d", sample))
    assert len(re.findall(_RULE, sample)) == sample.count("dot face:")


def test_the_pages_revisions_are_a_face_one_revision_ahead_of_omlx():
    (sample,) = (sample for sample in _SAMPLES if "a later revision" in sample)
    face, runs = formats.CONTRACT + 1, formats.CONTRACT
    assert re.findall(_REVISIONS, sample) == [f"requirement '{face} <= {runs}'"]
    assert f"'omlx_revision<{face}>'" in sample
    assert f"a face of revision {face} on an oMLX of revision {runs}" in _PAGE


@metal
@pytest.mark.parametrize("phrase", _BREAKS)
def test_the_page_cites_the_lines_the_compiler_cites(phrase):
    (sample,) = (sample for sample in _SAMPLES if phrase in sample)
    cited = re.findall(f"{_LINE}|{_RULE}|{_REVISIONS}", sample)
    message = " ".join(_page_error(**_BREAKS[phrase]).split())
    assert cited and [c for c in cited if c not in message] == []
