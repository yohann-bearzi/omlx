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

from omlx.formats import Dot, kernels, testing

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# w = (code - 128) * scale[row]: 8-bit codes, one float scale per row.
_TOY = r"""
namespace omlx_toy_KEY {
struct Dot {
  const device uint8_t* codes;
  const device float* scales;

  template <int K>
  struct Lanes {
    static constant constexpr int VPT = 8;
    static constant constexpr int BLOCK = 32 * VPT;
    static constant constexpr bool TAIL = IS_TAIL;

    const device uint8_t* w;
    const device float* s;

    Lanes(Dot m, size_t row, uint lane)
        : w(m.codes + row * K + lane * VPT), s(m.scales + row) {}

    struct Row {
      const device uint8_t* w;
      float s;
    };
    Row row(int r) const { return {w + r * K, s[r]}; }
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
      return acc + r.s * dot(omlx_iota<VPT>(), r.w, xt);
    }

    template <typename X>
    static float load(const device X* x, thread float* xt, int n) {
      for (int i = 0; i < n; i++) xt[i] = float(x[i]);
      return 0;
    }
    static float madd(float acc, Row r, thread const float* xt, float, int n) {
      float d = 0;
      for (int i = 0; i < n; i++) d += (float(r.w[i]) - 128.0f) * xt[i];
      return acc + r.s * d;
    }

    float finish(int, float sum) const { return sum; }
  };
};
}  // namespace omlx_toy_KEY
"""


def _toy(key, tail=False, source=_TOY) -> Dot:
    return Dot(
        key=f"toy_{key}",
        source=source.replace("KEY", key).replace(
            "IS_TAIL", "true" if tail else "false"
        ),
        type=f"omlx_toy_{key}::Dot",
        tensors=("scales",),
        tail=tail,
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


def _error(face, k=512, packs=(1, 1)) -> str:
    operands, x, _ = _matrix(8, k)
    with pytest.raises(RuntimeError) as failure:
        mx.eval(testing.dot_rows(face, operands, x, 8, packs))
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
        '#line 1 "<kernel>"\n'
    )


def test_a_kernel_includes_each_format_source_once():
    matrices = kernels.Matrices()
    a, b = _toy("a"), _toy("b")
    for name, face in (("sg", a), ("su", a), ("rt", b)):
        matrices.add(name, face)
    header = matrices.header
    assert header.count(a.source) == header.count(b.source) == 1
    # Formats may use the index packs; the skeletons use the formats.
    assert (
        header.index("struct omlx_ints")
        < header.index(a.source)
        < header.index(b.source)
        < header.index("void qmv_rows(")
    )


@metal
@pytest.mark.parametrize("packs", [(1, 0), (1, 1), (2, 2), (4, 0), (3, 1)])
@pytest.mark.parametrize(
    "k, tail", [(512, False), (512, True), (384, True), (64, True)]
)
def test_rows_are_the_dequantized_products(k, tail, packs):
    operands, x, want = _matrix(16, k, seed=k + sum(packs))
    got = testing.dot_rows(_toy(f"rows_{int(tail)}", tail), operands, x, 16, packs)
    assert mx.array_equal(got, want).item()


@metal
def test_two_formats_run_in_one_kernel():
    a, b = _toy("first"), _toy("second", tail=True)
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
def test_a_format_without_a_tail_needs_no_tail_members():
    start = _TOY.index(
        "    template <typename X>\n    static float load(const device X* x, thread float* xt, int n)"
    )
    tail_members = _TOY[start : _TOY.index("    float finish")]
    face = _toy("fast_only", source=_TOY.replace(tail_members, ""))
    operands, x, want = _matrix(8, 512)
    assert mx.array_equal(testing.dot_rows(face, operands, x, 8), want).item()


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
def test_a_k_that_needs_a_tail_is_refused_in_words():
    message = _error(_toy("no_tail"), k=384)
    assert (
        "dot face: K is not a whole number of BLOCKs, so Lanes<K>::TAIL must be true"
        in message
    )


@metal
def test_an_error_in_a_format_cites_the_formats_own_line():
    source = _TOY.replace("return acc + r.s * dot(", "return acc + r.scale * dot(")
    message = _error(_toy("typo", source=source))
    assert f"<quant format toy_typo>:{_line(source, 'r.scale')}:" in message
    assert "no member named 'scale'" in message


@metal
def test_a_missing_member_is_named_where_the_skeleton_needs_it():
    source = _TOY.replace(
        "    float finish(int, float sum) const { return sum; }\n", ""
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


@metal
def test_a_face_whose_tail_is_not_its_metal_tail_is_refused():
    message = _error(replace(_toy("tail_lie", tail=True), tail=False))
    assert "quant format face toy_tail_lie: Dot.tail is not Lanes<K>::TAIL" in message


def test_dot_rows_refuses_what_the_launch_cannot_cover():
    operands, x, _ = _matrix(8, 512)
    with pytest.raises(ValueError, match="n=7 is not a multiple of the 2 rows"):
        testing.dot_rows(_toy("a"), operands, x, 7)
    half = replace(_toy("half"), dtype=mx.float16)
    with pytest.raises(ValueError, match="toy_half is exact for mlx.core.float16"):
        testing.dot_rows(half, operands, x, 8)
    (codes, scales), x, _ = _matrix(4, 512)
    with pytest.raises(ValueError, match="toy_a: scales has 4 elements"):
        testing.dot_rows(_toy("a"), (codes, scales), x, 4)
    with pytest.raises(ValueError):  # not the face's operands
        testing.dot_rows(_toy("a"), (codes,), x, 4)
