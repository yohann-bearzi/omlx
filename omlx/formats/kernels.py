# SPDX-License-Identifier: Apache-2.0
"""Kernel skeletons over a format's faces, and the glue that puts a format's
matrices into a ``mx.fast.metal_kernel``.

A kernel names each matrix it reads (``rt``, ``sg``, ...) in a ``Matrices``,
which gives the matrix's kernel inputs, the statements that declare the
matrices in the kernel body, and the Metal the kernel includes: the index
packs, the formats' sources and the skeletons below.

The Metal here is C++17 (Metal 4.1 takes no later standard), without derived
classes. What is known when a kernel is compiled is a template parameter or
a ``constexpr``, a choice between code paths is ``if constexpr``, and a run
over a fixed count is a fold over an index pack, unless the loop measures
faster.
"""

from __future__ import annotations

from . import Dot

# What the Metal compiler calls the Metal below when it cites a line of it.
_SKELETONS = "<omlx skeletons, docs/quant-formats.md>"

# Compile-time index packs, for folds over a fixed count (a kernel's rows, a
# lane's values): omlx_iota<N>() is omlx_ints<0, ..., N - 1>. Formats' sources
# may use them.
_INDEX_PACKS = r"""
template <int... I>
struct omlx_ints {};
template <int N, int... I>
struct omlx_iota_ {
  using type = typename omlx_iota_<N - 1, N - 1, I...>::type;
};
template <int... I>
struct omlx_iota_<0, I...> {
  using type = omlx_ints<I...>;
};
template <int N>
using omlx_iota = typename omlx_iota_<N>::type;
"""

# Rows of one mat-vec over a dot face D: one input vector against rows
# row_a.. of matrix ma (result rows [0, NA)) and rows row_b.. of matrix mb
# (rows [NA, NA + NB)). Each simdgroup lane walks the K inputs of its rows
# in the face's blocks; result[row] ends as the row's product. The rows are
# packs (RA, RB), so every block is straight-line code over them. The walk
# stays a loop: unrolled over a pack of blocks, the fused Qwen decode ran
# 3-8% slower (M5 Max). Where K ends inside a block, the lanes with inputs
# in that last block are known when the kernel is compiled.
#
# qmv_rows is what a kernel calls. It states the parts of the contract the
# compiler would not report on its own, so a face that breaks one fails with
# that sentence instead of computing garbage.
_QMV_ROWS = r"""
// One block of a lane: its VPT inputs against each row's weights.
template <typename L, typename T, int... RA, int... RB>
METAL_FUNC void omlx_qmv_block(
    omlx_ints<RA...>,
    omlx_ints<RB...>,
    L a,
    L b,
    const device T* x,
    thread float* x_thread,
    thread float* result) {
  constexpr int NA = sizeof...(RA);
  const float aux = L::load(x, x_thread);
  ((result[RA] = L::madd(result[RA], a.row(RA), x_thread, aux)), ...);
  ((result[NA + RB] = L::madd(result[NA + RB], b.row(RB), x_thread, aux)), ...);
}

template <typename T, int K, typename D, int... RA, int... RB>
METAL_FUNC void omlx_qmv_rows(
    omlx_ints<RA...> ra,
    omlx_ints<RB...> rb,
    D ma,
    size_t row_a,
    D mb,
    size_t row_b,
    const device T* x,
    uint simd_lid,
    thread float* result) {
  using L = typename D::template Lanes<K>;
  constexpr int NA = sizeof...(RA);
  // K is WHOLE inputs in whole blocks, then a last block in which only the
  // first LAST lanes have inputs.
  constexpr int WHOLE = K - K % L::BLOCK;
  constexpr int LAST = K % L::BLOCK / L::VPT;
  L a(ma, row_a, simd_lid);
  L b(mb, row_b, simd_lid);
  x += int(simd_lid) * L::VPT;

  float x_thread[L::VPT];
  ((result[RA] = 0), ...);
  ((result[NA + RB] = 0), ...);
  for (int k = 0; k < WHOLE; k += L::BLOCK) {
    omlx_qmv_block(ra, rb, a, b, x, x_thread, result);
    a.step();
    b.step();
    x += L::BLOCK;
  }
  if constexpr (LAST > 0) {
    if (simd_lid < uint(LAST)) {
      omlx_qmv_block(ra, rb, a, b, x, x_thread, result);
    }
  }
  ((result[RA] = a.finish(RA, metal::simd_sum(result[RA]))), ...);
  ((result[NA + RB] = b.finish(RB, metal::simd_sum(result[NA + RB]))), ...);
}

template <typename T, int K, int NA, int NB, typename D>
METAL_FUNC void qmv_rows(
    D ma,
    size_t row_a,
    D mb,
    size_t row_b,
    const device T* x,
    uint simd_lid,
    thread float* result) {
  using L = typename D::template Lanes<K>;
  static_assert(
      L::BLOCK == 32 * L::VPT,
      "dot face: Lanes<K>::BLOCK must be VPT inputs for each of a simdgroup's 32 lanes");
  static_assert(
      K % L::VPT == 0,
      "dot face: K must be a whole number of a lane's VPT inputs");
  omlx_qmv_rows<T, K>(
      omlx_iota<NA>(), omlx_iota<NB>(), ma, row_a, mb, row_b, x, simd_lid, result);
}
"""


class Matrices:
    """The matrices a kernel reads, each under the name its body uses."""

    def __init__(self) -> None:
        self._faces: dict[str, Dot] = {}

    def add(self, name: str, face: Dot) -> list[str]:
        """Add matrix ``name``. Returns its kernel inputs: the weight, then
        the face's tensors in order."""
        self._faces[name] = face
        return _inputs(name, face)

    @property
    def header(self) -> str:
        """The Metal the kernel includes: the index packs, each distinct
        format source once, then the skeletons."""
        sources = {face.source: face.key for face in self._faces.values()}
        return (
            _named(_SKELETONS, _INDEX_PACKS)
            + "".join(_named(f"<quant format {key}>", s) for s, key in sources.items())
            + _named(_SKELETONS, _QMV_ROWS, line=_INDEX_PACKS.count("\n") + 1)
            + _named("<kernel header>", "")
        )

    @property
    def declarations(self) -> str:
        """The statements that open the kernel body: each matrix, declared
        over its inputs. The body's own lines follow as ``<kernel>``."""
        return "".join(
            f"    const auto {name} = {face.type}{{{_members(name, face)}}};\n"
            for name, face in self._faces.items()
        ) + _named("<kernel>", "")


def _named(name: str, source: str, line: int = 1) -> str:
    """``source`` as the Metal compiler will cite it: under ``name``, its
    first line numbered ``line``."""
    return f'#line {line} "{name}"\n{source}'


def _inputs(name: str, face: Dot) -> list[str]:
    return [f"{name}_w", *(f"{name}_{tensor}" for tensor in face.tensors)]


def _members(name: str, face: Dot) -> str:
    weight, *tensors = _inputs(name, face)
    return ", ".join([f"(const device uint8_t*){weight}", *tensors])
