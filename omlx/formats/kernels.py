# SPDX-License-Identifier: Apache-2.0
"""Kernel skeletons over a format's faces, and the glue that puts a format's
matrices into a ``mx.fast.metal_kernel``.

A kernel names each matrix it reads (``rt``, ``sg``, ...) in a ``Matrices``,
which gives the matrix's kernel inputs, the statements that declare the
matrices in the kernel body, and the Metal the kernel includes: the index
packs, the formats' sources and the skeletons below of the kinds of faces
it reads.

The Metal here is C++17 (Metal 4.1 takes no later standard), without derived
classes. What is known when a kernel is compiled is a template parameter or
a ``constexpr``, a choice between code paths is ``if constexpr``, and a run
over a fixed count is a fold over an index pack, unless the loop measures
faster.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from functools import cache

import mlx.core as mx

from .. import formats
from . import Dot, Table, Tile

logger = logging.getLogger(__name__)

# What the Metal compiler calls the Metal below when it cites a line of it.
_SKELETONS = "<omlx skeletons, docs/quant-formats.md>"


def _compact(metal: str) -> str:
    """``metal`` as a kernel carries it: each line without its indentation,
    a comment line left empty. MLX hashes a kernel's whole source at every
    launch (about 0.05 us a KB); the lines stay where they are, so the
    compiler cites them at the numbers they have here."""
    return "\n".join(
        "" if line.lstrip().startswith("//") else line.strip()
        for line in metal.split("\n")
    )


# Compile-time index packs, for folds over a fixed count (a kernel's rows, a
# lane's values): omlx_iota<N>() is omlx_ints<0, ..., N - 1>. Formats' sources
# may use them.
_INDEX_PACKS = _compact(r"""
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
""")

# What both dot skeletons check of a face and of x, ahead of them: the face's
# revision against the ones oMLX runs (FACE is a template parameter so that
# a failed assert prints both numbers), and that x is the T inputs, in
# device memory or in threadgroup memory where a kernel stages them: the
# skeleton passes the face x as a const pointer, whatever the kernel's is.
_CHECKS = _compact(r"""
template <int FACE>
struct omlx_revision {
  static_assert(
      FACE <= OMLX_CONTRACT,
      "dot face: written for a later revision of the contract than this oMLX runs");
  static_assert(
      FACE >= OMLX_OLDEST_CONTRACT,
      "dot face: written for a revision of the contract this oMLX no longer runs");
};

template <typename T>
METAL_FUNC const device T* omlx_inputs(const device T* x) { return x; }
template <typename T>
METAL_FUNC const threadgroup T* omlx_inputs(const threadgroup T* x) { return x; }
""")

# Rows of one mat-vec over a dot face D: one input vector against rows
# row_a.. of matrix ma (result rows [0, NA)) and rows row_b.. of matrix mb
# (rows [NA, NA + NB)). Each simdgroup lane walks the K inputs of its rows
# in the face's blocks; result[row] ends as the row's product. The rows are
# packs (RA, RB), so every block is straight-line code over them. The walk
# stays a loop: unrolled over a pack of blocks, the fused Qwen decode ran
# 3-8% slower (M5 Max); the one-token down asks for pairs unrolled two at a
# time. Where K ends inside a block, the lanes with inputs
# in that last block are known when the kernel is compiled.
#
# qmv_rows is what a kernel calls. It states the parts of the contract the
# compiler would not report on its own, so a face that breaks one fails with
# that sentence instead of computing garbage. The face's revision comes
# first: the rest is compiled only for a face of a revision oMLX runs, so a
# face of another one fails with that and nothing else. So for a lane of no
# inputs (VPT): nothing that multiplies or divides by VPT or walks a lane's
# inputs follows. x is the T inputs in device memory, or in threadgroup
# memory where a kernel stages them (omlx_inputs checks it); the sizeof of
# omlx_revision instantiates it, so its asserts run. Comments stay out of
# the Metal: MLX hashes a kernel's whole source at every launch.
_QMV_ROWS = _compact(r"""
// One block of a lane: its VPT inputs against each row's weights.
template <typename L, typename XP, int... RA, int... RB>
METAL_FUNC void omlx_qmv_block(
    omlx_ints<RA...>,
    omlx_ints<RB...>,
    thread const L& a,
    thread const L& b,
    XP x,
    thread float* x_thread,
    thread float* result) {
  constexpr int NA = sizeof...(RA);
  const auto aux = L::load(x, x_thread);
  ((result[RA] = L::madd(result[RA], a.row(RA), x_thread, aux)), ...);
  ((result[NA + RB] = L::madd(result[NA + RB], b.row(RB), x_thread, aux)), ...);
}

template <int K, typename D, typename XP, int... RA, int... RB>
METAL_FUNC void omlx_qmv_rows(
    omlx_ints<RA...> ra,
    omlx_ints<RB...> rb,
    D ma,
    size_t row_a,
    D mb,
    size_t row_b,
    XP x,
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

template <typename T, int K, int NA, int NB, typename D, typename XP>
METAL_FUNC void qmv_rows(
    D ma,
    size_t row_a,
    D mb,
    size_t row_b,
    XP x,
    uint simd_lid,
    thread float* result) {
  const auto xc = omlx_inputs<T>(x);
  constexpr int FACE = D::CONTRACT;
  (void)sizeof(omlx_revision<FACE>);
  if constexpr (FACE <= OMLX_CONTRACT && FACE >= OMLX_OLDEST_CONTRACT) {
    using L = typename D::template Lanes<K>;
    static_assert(
        L::VPT >= 1,
        "dot face: Lanes<K>::VPT must be one input or more");
    if constexpr (L::VPT >= 1) {
      static_assert(
          L::BLOCK == 32 * L::VPT,
          "dot face: Lanes<K>::BLOCK must be VPT inputs for each of a simdgroup's 32 "
          "lanes");
      static_assert(
          K % L::VPT == 0,
          "dot face: K must be a whole number of a lane's VPT inputs");
      omlx_qmv_rows<K>(
          omlx_iota<NA>(), omlx_iota<NB>(), ma, row_a, mb, row_b, xc, simd_lid, result);
    }
  }
}
""")

# What a kernel asks of a tile face F at K inputs (docs/quant-formats.md, The
# tile face), stated where the compiler can check it, so that a face that
# breaks a rule fails with that sentence and nothing after it. A kernel
# compiles its use of F under `if constexpr (omlx_tile_face<F, K>::value)`, in
# a function template of F. For a face of another revision the revision is
# the only error, as for a dot face; for one of a revision oMLX runs, the
# first rule it breaks is: the compiler reports the first static_assert that
# fails in a class it instantiates, and none after it.
#
# Every member is detected (omlx_void), not used, so that one that is
# missing, misnamed or of another signature is its rule's error and not the
# compiler's on the line that would use it: CONTRACT, Runs<K>, RUN, GROUP, P
# and Codes, the cursor's constructor, params, fetch and weight. The rules
# that need a member come after the rule that names it. The members are
# asked of a `thread const` cursor with int arguments, and weight of a
# group's and a run's `thread const` lvalues, as a kernel calls them; the
# constants as int template arguments, so that one that is not static, or
# not an integer, is its rule's error too. omlx_tile_face's FACE is a
# template parameter so that a failed assert prints both numbers.
_TILE_FACE = _compact(r"""
template <typename...>
using omlx_void = void;

template <typename X>
thread const X& omlx_lvalue();

template <typename F, typename = void>
struct omlx_tile_contract {
  static constant constexpr bool found = false;
  static constant constexpr int value = 0;
};
template <typename F>
struct omlx_tile_contract<F, omlx_void<omlx_ints<F::CONTRACT>>> {
  static constant constexpr bool found = true;
  static constant constexpr int value = F::CONTRACT;
};

struct omlx_tile_none {};
template <typename F, int K, typename = void>
struct omlx_tile_runs {
  static constant constexpr bool found = false;
  using type = omlx_tile_none;
};
template <typename F, int K>
struct omlx_tile_runs<F, K, omlx_void<typename F::template Runs<K>>> {
  static constant constexpr bool found = true;
  using type = typename F::template Runs<K>;
};

template <typename R, typename = void>
struct omlx_tile_run {
  static constant constexpr bool value = false;
};
template <typename R>
struct omlx_tile_run<R, omlx_void<omlx_ints<R::RUN>>> {
  static constant constexpr bool value = R::RUN == 8;
};

template <typename R, typename = void>
struct omlx_tile_group {
  static constant constexpr bool value = false;
};
template <typename R>
struct omlx_tile_group<R, omlx_void<omlx_ints<R::GROUP>>> {
  static constant constexpr bool value = R::GROUP >= 8 &&
      R::GROUP <= (1 << 30) && (R::GROUP & (R::GROUP - 1)) == 0;
};

template <typename X, typename = void>
struct omlx_tile_held {
  static constant constexpr bool value = false;
};
template <typename X>
struct omlx_tile_held<
    X,
    omlx_void<
        decltype(metal::declval<thread X&>() = omlx_lvalue<X>()),
        decltype(metal::declval<thread X&>() = X())>> {
  static constant constexpr bool value = metal::is_default_constructible_v<X>;
};

template <typename R, typename = void>
struct omlx_tile_types {
  static constant constexpr bool value = false;
};
template <typename R>
struct omlx_tile_types<R, omlx_void<typename R::P, typename R::Codes>> {
  static constant constexpr bool value = omlx_tile_held<typename R::P>::value &&
      omlx_tile_held<typename R::Codes>::value;
};

template <typename R, typename = void>
struct omlx_tile_params {
  static constant constexpr bool value = false;
};
template <typename R>
struct omlx_tile_params<
    R,
    omlx_void<
        typename R::P,
        decltype(metal::declval<thread const R&>().params(omlx_lvalue<int>()))>> {
  static constant constexpr bool value = metal::is_same_v<
      decltype(metal::declval<thread const R&>().params(omlx_lvalue<int>())),
      typename R::P>;
};

template <typename R, typename = void>
struct omlx_tile_fetch {
  static constant constexpr bool value = false;
};
template <typename R>
struct omlx_tile_fetch<
    R,
    omlx_void<
        typename R::Codes,
        decltype(metal::declval<thread const R&>().fetch(omlx_lvalue<int>()))>> {
  static constant constexpr bool value = metal::is_same_v<
      decltype(metal::declval<thread const R&>().fetch(omlx_lvalue<int>())),
      typename R::Codes>;
};

template <typename R, typename = void>
struct omlx_tile_weight {
  static constant constexpr bool value = false;
};
template <typename R>
struct omlx_tile_weight<
    R,
    omlx_void<decltype(R::weight(
        metal::declval<thread const typename R::P&>(),
        metal::declval<thread const typename R::Codes&>(),
        omlx_lvalue<int>()))>> {
  static constant constexpr bool value = metal::is_same_v<
      decltype(R::weight(
          metal::declval<thread const typename R::P&>(),
          metal::declval<thread const typename R::Codes&>(),
          omlx_lvalue<int>())),
      float>;
};

template <
    typename F,
    int K,
    bool STATED = omlx_tile_contract<F>::found,
    int FACE = omlx_tile_contract<F>::value,
    bool RUNS = FACE <= OMLX_CONTRACT && FACE >= OMLX_OLDEST_CONTRACT>
struct omlx_tile_face {
  static_assert(
      FACE <= OMLX_CONTRACT,
      "tile face: written for a later revision of the contract than this oMLX runs");
  static_assert(
      FACE >= OMLX_OLDEST_CONTRACT,
      "tile face: written for a revision of the contract this oMLX no longer runs");
  static constant constexpr bool value = false;
};

template <typename F, int K, int FACE>
struct omlx_tile_face<F, K, false, FACE, false> {
  static_assert(
      omlx_tile_contract<F>::found,
      "tile face: the face needs a static constant constexpr int CONTRACT: the "
      "revision of the contract it is written for");
  static constant constexpr bool value = false;
};

template <typename F, int K, int FACE>
struct omlx_tile_face<F, K, true, FACE, true> {
  static constant constexpr bool FOUND = omlx_tile_runs<F, K>::found;
  using R = typename omlx_tile_runs<F, K>::type;
  static constant constexpr bool RUN = omlx_tile_run<R>::value;
  static constant constexpr bool GROUP = omlx_tile_group<R>::value;
  static constant constexpr bool HELD = omlx_tile_types<R>::value;
  static constant constexpr bool CURSOR =
      metal::is_constructible_v<R, thread const F&, thread const size_t&>;
  static constant constexpr bool PARAMS = omlx_tile_params<R>::value;
  static constant constexpr bool FETCH = omlx_tile_fetch<R>::value;
  static constant constexpr bool WEIGHT = omlx_tile_weight<R>::value;
  static_assert(
      FOUND,
      "tile face: the face needs a member template Runs<int K>: the cursor of a "
      "row of K inputs");
  static_assert(
      RUN,
      "tile face: Runs<K>::RUN must be 8: every tile kernel reads runs of 8 codes");
  static_assert(
      GROUP,
      "tile face: Runs<K>::GROUP must be 8 times a power of two (8, 16, 32, ..., "
      "2^30): the values a kernel's thread decodes are then whole groups or in one");
  static_assert(
      HELD,
      "tile face: Runs<K>::P and Runs<K>::Codes must be types that are "
      "default-constructible and copy-assignable: a kernel holds arrays of them");
  static_assert(
      CURSOR,
      "tile face: Runs<K> must be constructible from the face and a size_t row");
  static_assert(
      PARAMS,
      "tile face: Runs<K> needs a const member params(int g) that returns a P");
  static_assert(
      FETCH,
      "tile face: Runs<K> needs a const member fetch(int run) that returns a Codes");
  static_assert(
      WEIGHT,
      "tile face: Runs<K> needs a static member weight(P, Codes, int i) that "
      "returns a float");
  static constant constexpr bool value =
      FOUND && RUN && GROUP && HELD && CURSOR && PARAMS && FETCH && WEIGHT;
};
""")

# The shared-rows skeleton: ROWS input vectors (x, x + K, ...) against RPS
# consecutive rows of a matrix from `row`, each block's weights decoded once
# for all the inputs (decode, then apply). result[i * RPS + r] is input i's
# product with row `row + r`; with PARTIAL the pack holds `valid` rows (the
# last tile of an N that is not whole packs). Every vector's inputs of a
# block are loaded ahead of the rows. A kernel that asks for a full unroll
# (Matrices(unrolled=True)) folds the block over its inputs and its rows, and
# unrolls the skeleton's other loops (OMLX_SHARED_UNROLL), each array then
# living in registers; any other kernel keeps the block's loops. A kernel
# calls qmv_shared, which asserts what the compiler would not report, as
# qmv_rows does, and that the face's vpt is its Lanes<K>::VPT: a launch is
# sized by it.
_SHARED_LOOPED = _compact(r"""
template <int K, int ROWS, int RPS, bool PARTIAL, typename L, typename XP>
METAL_FUNC void omlx_shared_block(
    thread const L& a, int valid, XP x, thread float* result) {
  float xs[ROWS][L::VPT];
  decltype(L::load(x, xs[0])) aux[ROWS];
  for (int i = 0; i < ROWS; i++) {
    aux[i] = L::load(x + i * K, xs[i]);
  }
  for (int r = 0; r < RPS; r++) {
    if (PARTIAL && r >= valid) {
      break;
    }
    const auto d = L::decode(a.row(r));
    for (int i = 0; i < ROWS; i++) {
      result[i * RPS + r] = L::apply(result[i * RPS + r], d, xs[i], aux[i]);
    }
  }
}
""")

_SHARED_FOLDED = _compact(r"""
template <int R, int RPS, bool PARTIAL, typename L, typename A, int... I>
METAL_FUNC void omlx_shared_row(
    thread const L& a, int valid, thread float (*xs)[L::VPT],
    thread const A* aux, thread float* result, omlx_ints<I...>) {
  if (PARTIAL && R >= valid) {
    return;
  }
  const auto d = L::decode(a.row(R));
  ((result[I * RPS + R] = L::apply(result[I * RPS + R], d, xs[I], aux[I])), ...);
}

template <int K, int RPS, bool PARTIAL, typename L, typename XP, int... I, int... R>
METAL_FUNC void omlx_shared_folded(
    thread const L& a, int valid, XP x, thread float* result, omlx_ints<I...> inputs,
    omlx_ints<R...>) {
  float xs[sizeof...(I)][L::VPT];
  decltype(L::load(x, xs[0])) aux[sizeof...(I)];
  ((aux[I] = L::load(x + I * K, xs[I])), ...);
  (omlx_shared_row<R, RPS, PARTIAL>(a, valid, xs, aux, result, inputs), ...);
}

template <int K, int ROWS, int RPS, bool PARTIAL, typename L, typename XP>
METAL_FUNC void omlx_shared_block(
    thread const L& a, int valid, XP x, thread float* result) {
  omlx_shared_folded<K, RPS, PARTIAL>(
      a, valid, x, result, omlx_iota<ROWS>(), omlx_iota<RPS>());
}
""")

_QMV_SHARED = _compact(r"""
template <int K, int ROWS, int RPS, bool PARTIAL, typename D, typename XP>
METAL_FUNC void omlx_qmv_shared(
    D m, size_t row, int valid, XP x, uint simd_lid, thread float* result) {
  using L = typename D::template Lanes<K>;
  constexpr int WHOLE = K - K % L::BLOCK;
  constexpr int LAST = K % L::BLOCK / L::VPT;
  L a(m, row, simd_lid);
  x += int(simd_lid) * L::VPT;
  OMLX_SHARED_UNROLL
  for (int j = 0; j < ROWS * RPS; j++) {
    result[j] = 0;
  }
  for (int k = 0; k < WHOLE; k += L::BLOCK) {
    omlx_shared_block<K, ROWS, RPS, PARTIAL>(a, valid, x, result);
    a.step();
    x += L::BLOCK;
  }
  if constexpr (LAST > 0) {
    if (simd_lid < uint(LAST)) {
      omlx_shared_block<K, ROWS, RPS, PARTIAL>(a, valid, x, result);
    }
  }
  OMLX_SHARED_UNROLL
  for (int r = 0; r < RPS; r++) {
    if (PARTIAL && r >= valid) {
      break;
    }
    OMLX_SHARED_UNROLL
    for (int i = 0; i < ROWS; i++) {
      result[i * RPS + r] = a.finish(r, metal::simd_sum(result[i * RPS + r]));
    }
  }
}

template <
    typename T,
    int K,
    int ROWS,
    int RPS,
    int VPT,
    bool PARTIAL,
    typename D,
    typename XP>
METAL_FUNC void qmv_shared(
    D m, size_t row, int valid, XP x, uint simd_lid, thread float* result) {
  const auto xc = omlx_inputs<T>(x);
  constexpr int FACE = D::CONTRACT;
  (void)sizeof(omlx_revision<FACE>);
  if constexpr (FACE <= OMLX_CONTRACT && FACE >= OMLX_OLDEST_CONTRACT) {
    using L = typename D::template Lanes<K>;
    static_assert(
        L::VPT >= 1,
        "dot face: Lanes<K>::VPT must be one input or more");
    if constexpr (L::VPT >= 1) {
      static_assert(
          L::BLOCK == 32 * L::VPT,
          "dot face: Lanes<K>::BLOCK must be VPT inputs for each of a simdgroup's 32 "
          "lanes");
      static_assert(
          K % L::VPT == 0,
          "dot face: K must be a whole number of a lane's VPT inputs");
      static_assert(
          L::VPT == VPT,
          "dot face: Lanes<K>::VPT must be the face's vpt, which sizes the launch");
      omlx_qmv_shared<K, ROWS, RPS, PARTIAL>(m, row, valid, xc, simd_lid, result);
    }
  }
}
""")


# Output rows a simdgroup of a shared-rows launch computes (qmv's own tile:
# a threadgroup of two simdgroups owns 8).
SHARED_RPS = 4
# Verify projections run in a dependent chain, so latency wins: below this
# many tiles every input gets its own threadgroups (rows of a tile are
# adjacent in the grid and hit cache); wider outputs (the vocabulary head)
# share each weight pass between inputs, up to this many per-lane input
# values.
SHARED_ROWS_MIN_TILES = 2048
ROW_VALUES_BUDGET = 32
# Fully unrolled tiles hold every (input, row) accumulator and every input's
# values in registers. On the M5 Ultra (MLX 0.32.2) compiler, tiles past this
# envelope came out with wrong bits (32 or more accumulators; 7 or 8 inputs
# of 16 values at 4 bits), so unrolled launches stay inside it.
UNROLLED_ACCUMULATORS = 16
UNROLLED_ROW_VALUES = 80


# A GPU's cores when the system does not say: the most of an Apple GPU, so
# that a launch sized by them is never left underfilled.
_MOST_GPU_CORES = 80


@cache
def gpu_cores() -> int:
    """This GPU's cores (system_profiler, read once), or the most an Apple
    GPU has where it cannot be read (logged once)."""
    from ..utils.hardware import get_gpu_core_count

    cores = get_gpu_core_count()
    if cores:
        return cores
    logger.warning(
        "GPU core count unreadable; shared-rows launches are sized for %d cores",
        _MOST_GPU_CORES,
    )
    return _MOST_GPU_CORES


def gpu_generation() -> int:
    """This GPU's family generation (15 the M3's, 17 the M5's) from MLX's
    architecture string, read once; 0 where it cannot be read."""
    from ..utils.hardware import get_apple_gpu_generation

    return get_apple_gpu_generation()


def shared_geometry(vpt: int, n: int, rows: int, pad: bool) -> tuple[int, int]:
    """(inputs per threadgroup, output rows per simdgroup) of a shared-rows
    launch of ``rows`` inputs over N ``n``, for a face of ``vpt`` inputs a
    lane. With ``pad``, a count of inputs that no divisor groups is padded to
    the budget's group (the caller's rule: large 8-bit weight streams)."""
    if (n + 2 * SHARED_RPS - 1) // (2 * SHARED_RPS) < SHARED_ROWS_MIN_TILES:
        return 1, SHARED_RPS
    limit = ROW_VALUES_BUDGET // vpt
    per_group = next((d for d in range(min(rows, limit), 1, -1) if rows % d == 0), 1)
    if per_group == 1 and pad:
        # A face past the budget (vpt over 32) takes one input a group.
        per_group = max(1, min(rows, limit))
    return per_group, SHARED_RPS


def unrolled_tile_ok(vpt: int, rps: int, rows_per_group: int) -> bool:
    """Whether an unrolled tile of ``rps`` output rows x ``rows_per_group``
    inputs per simdgroup, ``vpt`` values a lane, is inside the envelope
    above."""
    return (
        rps * rows_per_group <= UNROLLED_ACCUMULATORS
        and rows_per_group * vpt <= UNROLLED_ROW_VALUES
    )


def _qmv_shared(unrolled: bool) -> str:
    if unrolled:
        block, unroll = _SHARED_FOLDED, '_Pragma("clang loop unroll(full)")'
    else:
        block, unroll = _SHARED_LOOPED, ""
    return block + _QMV_SHARED.replace("OMLX_SHARED_UNROLL", unroll)


# The shared-rows skeleton, rolled and unrolled, as a kernel carries it.
_SHARED_SKELETONS = (_qmv_shared(False), _qmv_shared(True))

# The rows skeleton as it is, and with its K loop's pairs unrolled two at a
# time for a kernel that asks (the one-token down): the loop line replaced,
# never a second copy of the skeleton.
_ROWS_SKELETONS = (
    _QMV_ROWS,
    _QMV_ROWS.replace(
        "for (int k = 0; k < WHOLE; k += L::BLOCK) {",
        '_Pragma("clang loop unroll_count(2)") '
        "for (int k = 0; k < WHOLE; k += L::BLOCK) {",
        1,
    ),
)


# What a kernel does with qmv_rows' result, in one lane of the simdgroup. A
# kernel that stores rows adds this to its header.
STORE_ROWS = _compact(r"""
// A simdgroup's rows, each rounded to T.
template <typename T, int... R>
METAL_FUNC void store_rows(omlx_ints<R...>, thread const float* result, device T* y) {
  ((y[R] = static_cast<T>(result[R])), ...);
}
""")


# MLX's shapes are int32: a buffer of zeros as one axis of bytes is smaller.
_ONE_BUFFER = 2**31


def zeros(*specs) -> list[mx.array]:
    """Zeros of each ``(shape, dtype)`` of ``specs``, a stand-in's inputs,
    not evaluated: ``Proof.prove`` evaluates them. They are made on the
    GPU's stream, the launch's. Each op is given the stream: the default
    device is the process's, every thread's, and is left as it is. The
    zeros are views of one buffer of as many bytes as the largest, from its
    first byte: one fill and, where MLX's buffer cache is empty (oMLX clears
    it after each prefill chunk), one new buffer. One of 2 GiB or more is
    zeros of its own: MLX's shapes are int32."""
    gpu = mx.gpu
    sizes = [math.prod(shape) * dtype.size for shape, dtype in specs]
    below = [size for size in sizes if size < _ONE_BUFFER]
    if below:
        buffer = mx.zeros((max(below),), mx.uint8, stream=gpu)
    made = []
    for (shape, dtype), size in zip(specs, sizes, strict=True):
        if size < _ONE_BUFFER:
            raw = mx.as_strided(buffer, (size,), stream=gpu)
            typed = mx.view(raw, dtype, stream=gpu)
            made.append(mx.reshape(typed, shape, stream=gpu))
        else:
            made.append(mx.zeros(shape, dtype, stream=gpu))
    return made


class _Replay:
    """A non-weakrefable MLX graph, borrowed by native callers through this owner."""

    __slots__ = ("graph", "__weakref__")

    def __init__(self, graph):
        self.graph = graph


class _Kept(threading.local):
    """The graphs that a thread keeps of one proof, each under its launch's
    signature. Every thread that asks has a dict of its own, which ends with
    the thread, or with the proof."""

    def __init__(self) -> None:
        self.graphs: dict = {}
        self._replays = None

    def replay(self, signature):
        graph = self.graphs.get(signature)
        if graph is None:
            return None
        if self._replays is None:
            self._replays = {}
        record = self._replays.get(signature)
        if record is None:
            record = self._replays[signature] = _Replay(graph)
        return record


# Held to keep a graph and to record a failure: a proof that fails while a
# launch is built keeps no graph of it.
_KEEPING = threading.Lock()


@dataclass
class Proof:
    """What is known of the launches that one kernel compiles from the same
    faces, tensor dtypes and shapes.

    MLX compiles a kernel when it first evaluates it. So a kernel evaluates
    a launch of each Metal it is instantiated as before it builds a call's
    (``prove``), while the call can still fall back, and a launch that
    fails, whatever failed in that evaluation, keeps what shares the proof
    on its composed path until the process ends. A kernel that evaluates the
    call's own launch for this evaluates the launch's inputs ahead of it:
    their failure is the caller's.

    A launch's graph is a function of its arrays' shapes and types alone, so
    a kernel builds it once for each, through MLX's compile, and keeps it
    here (``keep``): a call then replays the graph, where building it anew
    has MLX write the kernel's source out again, launch after launch.

    Several threads run launches (each model's engine has its own). What
    the launches proved is the process's: MLX builds a kernel's Metal once
    for the device, so ``ran`` and ``failed`` hold for every thread. The
    graphs are kept thread by thread (``graphs`` is the calling thread's):
    MLX keeps what it traces a compiled function into in the thread that
    calls it, under the function's address, and the function's end removes
    it from one thread alone, the last that called it. Another thread that
    had called it would be left with a launch that the next function at the
    address replays there. A thread calls the graphs it built, and no
    other."""

    ran: set = field(default_factory=set)  # the launches that compiled
    failed: bool = False
    _kept: _Kept = field(default_factory=_Kept, init=False, repr=False, compare=False)

    @property
    def graphs(self) -> dict:
        """The calling thread's graphs, each under its launch's signature."""
        return self._kept.graphs

    def prove(self, metal, build, log, engaged: str, failed: str) -> bool:
        """Whether the kernel's launches of ``metal`` run: one ran before,
        or the launch that ``build`` gives is evaluated here, and runs.

        ``metal`` names one Metal of the proof's kernel: what its source is
        instantiated with beside the proof's key (x's type, a template
        value, an input that MLX binds another way). ``build`` gives the
        stand-in of a launch of that Metal: its inputs, not evaluated, and
        the function that launches it over them. The proof evaluates the
        inputs, then synchronizes the GPU's stream, ahead of the launch and
        outside its guard: what fails in making them (an allocation, an
        error that an earlier evaluation left pending on the GPU) is raised,
        the proof neither failed nor proven, and the next call proves the
        Metal again. Its caller answers for three things:

        - the launch reads its inputs alone, arrays that the kernel makes
          for it (``zeros``): no array of the call, no tensor of a layer.
          Those may belong to a function that MLX is tracing, which refuses
          to evaluate them; what fails in them is their owner's failure,
          not the Metal's; and they have the call's size, where a proof
          needs the smallest.
        - MLX generates the source of the call's launch for it: the same
          kernel, template values and output dtypes and, input by input,
          the same dtype, fewer than 8 elements or not, an axis or none.
          Shapes, the grid and the output's shape are no part of the source.
        - the launch has the call's threadgroup: MLX checks a threadgroup
          against the pipeline's limit at every evaluation, and the grid
          not at all.

        What ``build`` and the function raise is raised: a launch that
        cannot be built is the kernel's defect, or its caller's. What the
        launch's evaluation raises,
        of any type, is the Metal's failure: the proof fails (``fail``),
        ``failed`` is logged to ``log`` with the error, and the answer is
        False, as it is from then on for every Metal. A launch that runs
        marks ``metal`` in ``ran`` and logs ``engaged``.

        Threads: no lock is held while the launch is built and evaluated,
        so two threads that meet ``metal`` unproven at once each prove it
        and log it; MLX builds the Metal once for the device, and it is
        marked once. A failure is recorded under the lock that ``keep``
        takes, for every thread. A thread whose launch ran while another's
        failed has the answer True: its call has its launch, and ``keep``
        keeps no graph of it."""
        if self.failed:
            return False
        if metal in self.ran:
            return True
        inputs, launch = build()
        inputs = tuple(inputs)  # evaluated, then launched: read twice
        mx.eval(*inputs)
        mx.synchronize(mx.gpu)
        launched = launch(*inputs)
        try:
            # Surface a kernel build failure while the call can still fall
            # back.
            mx.eval(launched)
        except Exception:
            self.fail()
            log.warning(failed, exc_info=True)
            return False
        self.ran.add(metal)
        log.info(engaged)
        return True

    def keep(self, signature, graph) -> None:
        """Keep ``graph``, a function that ``mx.compile`` gave and that the
        calling thread built, for the thread's launches of ``signature``:
        everything that is computed outside the function and built into it
        (the pairs, the grid, the output's shape), or the arrays' shapes
        and types that all of it comes from. MLX's compile tells apart only
        what the function reads from its arguments, and traces it again for
        others with the rest as the first call left it: under a signature
        that leaves a shape out, a call replays the launch of another's
        rows. A proof that failed, on any thread, keeps nothing."""
        ended = None
        with _KEEPING:
            if self.failed:
                return
            kept = self._kept
            if len(kept.graphs) >= MAX_GRAPHS:
                ended, kept.graphs = kept.graphs, {}
            # Native layers borrow the thread's records weakly. Retire the
            # affected records even if a caller still holds the old graph map.
            if kept._replays is not None:
                if ended is not None:
                    kept._replays = None
                else:
                    kept._replays.pop(signature, None)
            kept.graphs[signature] = graph
        # The graphs that one more dropped end here, the lock released: as
        # many as MAX_GRAPHS launches for MLX to end.
        del ended

    def fail(self) -> None:
        """A launch failed: nothing that shares the proof is launched again,
        on any thread. Every thread's graphs are dropped with the failure,
        in one step for a thread that keeps one meanwhile, and end once the
        lock is released."""
        with _KEEPING:
            self.failed = True
            ended, self._kept = self._kept, _Kept()
        del ended


# The graphs a thread keeps of one proof: a decode step's, those of the short
# blocks around it, a few prompts' chunks. One more drops them all, to be
# built as called.
MAX_GRAPHS = 64


_PROOFS: dict[tuple, Proof] = {}

# The kernels over faces built so far, by what their source is written from.
_FACE_KERNELS: dict[tuple, object] = {}


def proof(kernel: str, *key) -> Proof:
    """The proof of the launches that ``kernel`` compiles from ``key``: its
    faces and what else its source is instantiated with, beside what the
    kernel keeps in ``Proof.ran``."""
    key = (kernel, *key)
    proof = _PROOFS.get(key)
    if proof is None:
        # One step: of two threads that miss at once, both get one Proof.
        proof = _PROOFS.setdefault(key, Proof())
    return proof


def face_kernel(
    name: str,
    inputs,
    matrices: dict[str, Dot | Tile],
    outputs,
    header: str,
    source: str,
):
    """The ``mx.fast.metal_kernel`` over the faces of ``matrices``, named
    ``<name>``, then each face's key, ``_``-joined in the order of
    ``inputs``: its inputs are ``inputs`` (a sequence of names) in order,
    where a matrix's name stands for its inputs (``Matrices.add``); its
    outputs are ``outputs``; its header is the faces' Metal
    (``Matrices.header``), then ``header``; its source opens with the
    matrices' declarations, then ``source``. Its source is written and MLX's
    kernel made once for each name, inputs, faces, outputs, header and
    source; every call after finds it. Threads: two that build one kernel at
    once each build it, and the first kept is the one every call finds."""
    if isinstance(inputs, str) or isinstance(outputs, str):
        raise TypeError(
            f"kernel {name!r}: inputs and outputs are sequences of names, not a "
            "string"
        )
    inputs, outputs = tuple(inputs), tuple(outputs)
    faces = tuple((given, matrices[given]) for given in inputs if given in matrices)
    if len(faces) != len(matrices):
        unread = sorted(set(matrices) - set(inputs))
        raise ValueError(
            f"kernel {name!r}: matrices {', '.join(unread)} are not among its inputs"
        )
    key = (name, inputs, faces, outputs, header, source)
    kernel = _FACE_KERNELS.get(key)
    if kernel is not None:
        return kernel
    built = Matrices()
    names = [
        n
        for given in inputs
        for n in (built.add(given, matrices[given]) if given in matrices else [given])
    ]
    kernel = mx.fast.metal_kernel(
        name="_".join((name, *(face.key for _, face in faces))),
        input_names=names,
        output_names=list(outputs),
        header=built.header + header,
        source=built.declarations + source,
    )
    return _FACE_KERNELS.setdefault(key, kernel)


# The source name used in Metal diagnostics for the generated table section.
_STAGED = "<omlx staged tables>"

# A table's Metal element type (MLX's names), and the unsigned type of the
# same width that holds its bits in the kernel's source.
_METAL = {
    mx.float32: ("float", "uint32_t"),
    mx.float16: ("half", "uint16_t"),
    mx.bfloat16: ("bfloat16_t", "uint16_t"),
    mx.int8: ("int8_t", "uint8_t"),
    mx.uint8: ("uint8_t", "uint8_t"),
    mx.int16: ("int16_t", "uint16_t"),
    mx.uint16: ("uint16_t", "uint16_t"),
    mx.int32: ("int32_t", "uint32_t"),
    mx.uint32: ("uint32_t", "uint32_t"),
}


# A kernel's threadgroups have a whole simdgroup or more: up to this many
# values of a type are copied one a thread, without a loop.
_ONE_COPY = 32

# A table of up to this many values may be read as selects on the index,
# its values held in the face's member, with nothing staged.
_SELECT_VALUES = 4


@cache
def _small_tables_as_selects() -> bool:
    """Whether this GPU reads a table of up to 4 values as selects: on the
    GPUs without tensor units (the M3 family) selects measured up to 15%
    faster than a staged table, on the M5 family (tensor units) staging
    measured 0.6-3% faster (R-TABLES.md, run 3). A family not measured
    follows the GPU's hardware generation. An unavailable or unrecognized
    GPU uses the staged form."""
    if not mx.is_available(mx.gpu):
        return False
    return _selects_for(str(mx.device_info(mx.gpu).get("architecture", "")))


def _selects_for(architecture: str) -> bool:
    match = re.fullmatch(r"applegpu_g(\d+)([a-z])", architecture)
    return match is not None and int(match[1]) < (18 if match[2] == "p" else 17)


# The member type of a face's table: a pointer to its staged copy or, for
# a small table where this GPU reads one faster so, its values themselves
# read by selects on the index's two low bits. SELECTS is this kernel's
# choice; the second form is defined only in a kernel that holds one.
_TABLE_TYPE = r"""
template <typename T, int N, bool VALUES = (N <= 4 && SELECTS)>
struct omlx_table {
  const threadgroup T* at;
  T operator[](uint i) const { return at[i]; }
};
"""
_TABLE_VALUES = r"""
template <typename T, int N>
struct omlx_table<T, N, true> {
  T v0, v1, v2, v3;
  T operator[](uint i) const {
    return (i & 2u) ? ((i & 1u) ? v3 : v2) : ((i & 1u) ? v1 : v0);
  }
};
"""
# The unused slots of a four-way select hold a NaN for floating-point
# tables and zero for integer tables.
_PAST = {mx.float32: 0x7FC00000, mx.float16: 0x7E00, mx.bfloat16: 0x7FC0}


def _bytes(tables) -> int:
    """The bytes of the distinct ``tables``."""
    return sum(len(table.values) * table.dtype.size for table in set(tables))


def table_bytes(*faces) -> int:
    """The bytes of the distinct tables of ``faces``: what a kernel over all
    of them holds of tables (at most ``formats.TABLE_BYTES``)."""
    return _bytes(table for face in faces for table in face.tables)


class Tables:
    """The tables of a kernel's faces: each distinct table staged once in
    threadgroup memory, end to end with the others of its element type, or,
    for a table of up to 4 values on a GPU that reads one faster so, held
    by value in the face's member.

    ``header`` defines the members' type, ``omlx_table<T, N>``, and holds
    the staged tables' bits; ``prologue``, the first text of the kernel's
    body, copies them into threadgroup memory with the threadgroup's threads
    and waits once for all of them. Both are empty when no table was added,
    and the prologue when none is staged: such a kernel's source is as
    without them. ``add`` gives a table's member, as a Metal expression."""

    def __init__(self) -> None:
        self._values = False  # whether a table is held by value
        self._by_type: dict[str, list[Table]] = {}
        self._at: dict[Table, str] = {}

    @property
    def size(self) -> int:
        """The bytes the staged tables take in threadgroup memory."""
        return sum(
            len(table.values) * table.dtype.size
            for run in self._by_type.values()
            for table in run
        )

    def add(self, table: Table) -> str:
        """Take ``table`` (once, when an equal table was taken already): the
        Metal expression of the face's member that reads it. A kernel holds
        at most ``formats.TABLE_BYTES`` of tables."""
        if not isinstance(table, Table):
            raise TypeError(f"expected a Table, got {type(table).__name__}")
        at = self._at.get(table)
        if at is None:
            size = _bytes((*self._at, table))
            if size > formats.TABLE_BYTES:
                raise ValueError(
                    f"table {table.name!r}: the kernel's tables would hold {size} "
                    f"bytes; a kernel holds at most {formats.TABLE_BYTES}"
                )
            metal, bits = _METAL[table.dtype]
            n = len(table.values)
            if n <= _SELECT_VALUES and _small_tables_as_selects():
                self._values = True
                past = _PAST.get(table.dtype, 0)
                words = [*table.bits, *(past,) * (_SELECT_VALUES - n)]
                values = ", ".join(f"as_type<{metal}>({bits}({w:#x}u))" for w in words)
            else:
                run = self._by_type.setdefault(metal, [])
                values = f"omlx_tg_{metal} + {sum(len(t.values) for t in run)}"
                run.append(table)
            at = f"omlx_table<{metal}, {n}>{{{values}}}"
            self._at[table] = at
        return at

    @property
    def header(self) -> str:
        if not self._at:
            return ""
        arrays = []
        for metal, run in self._by_type.items():
            bits = _METAL[run[0].dtype][1]
            words = ", ".join(f"{word:#x}u" for table in run for word in table.bits)
            count = sum(len(table.values) for table in run)
            arrays.append(f"constant {bits} bits_{metal}[{count}] = {{{words}}};\n")
        staged = "namespace omlx_staged {\n" + "".join(arrays) + "}\n" if arrays else ""
        if self._values:
            types = _TABLE_TYPE.replace("SELECTS", "true") + _TABLE_VALUES
        else:
            types = _TABLE_TYPE.replace("SELECTS", "false")
        return _named(_STAGED, types + staged) + _named("<kernel header>", "")

    @property
    def prologue(self) -> str:
        if not self._by_type:
            return ""
        lines = []
        for metal, run in self._by_type.items():
            count = sum(len(table.values) for table in run)
            # One copy a thread where the threadgroup's first simdgroup holds
            # them: at one token 0.2-0.3% of a launch cheaper than the loop
            # (R-TABLES.md, run 1b).
            fill = _ONE_FILL if count <= _ONE_COPY else _LOOP_FILL
            lines.append(fill.replace("TYPE", metal).replace("COUNT", str(count)))
        return "".join(lines) + (
            "    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n"
        )


# A type's staged copy, filled by the threadgroup's threads: one entry a
# thread, or a loop over the threadgroup's own size (a smaller threadgroup
# ends a grid that is not whole threadgroups).
_COPY = "      omlx_tg_TYPE[i] = as_type<TYPE>(omlx_staged::bits_TYPE[i]);\n"
# Every entry is copied whatever the threadgroup's size: a small threadgroup,
# or one at the grid's edge, may have fewer threads than the table entries.
# Up to 32 entries, a thread each, and a loop over what is left that a
# threadgroup of COUNT threads or more skips.
_THREADS = (
    "threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z"
)
_ONE_FILL = (
    "    threadgroup TYPE omlx_tg_TYPE[COUNT];\n"
    "    if (thread_index_in_threadgroup < COUNTu) {\n"
    "      const uint i = thread_index_in_threadgroup;\n" + _COPY + "    }\n"
    f"    if ({_THREADS} < COUNTu) {{\n"
    f"      for (uint i = thread_index_in_threadgroup + {_THREADS}; i < COUNTu;\n"
    f"           i += {_THREADS}) {{\n  " + _COPY + "      }\n    }\n"
)
_LOOP_FILL = (
    "    threadgroup TYPE omlx_tg_TYPE[COUNT];\n"
    "    for (uint i = thread_index_in_threadgroup; i < COUNTu;\n"
    f"         i += {_THREADS}) {{\n" + _COPY + "    }\n"
)


class Matrices:
    """The matrices a kernel reads, each under the name its body uses."""

    def __init__(self, unrolled: bool = False, unroll_pairs: bool = False) -> None:
        """``unrolled``: the shared-rows skeleton's loops are unrolled in full,
        where a kernel's arrays then live in registers. ``unroll_pairs``: the
        rows skeleton's K loop takes the two-at-a-time pragma."""
        self._faces: dict[str, Dot | Tile] = {}
        self._names: set[str] = set()
        self._tables = Tables()
        self._staged: dict[str, tuple[str, ...]] = {}
        self._shared: set[Dot] = set()
        self._shared_names: set[str] = set()
        self._unrolled = unrolled
        self._unroll_pairs = unroll_pairs

    def add(self, name: str, face: Dot | Tile, shared: bool = False) -> list[str]:
        """Add matrix ``name``, read through ``face``, a ``Dot`` or a
        ``Tile``. Returns its kernel inputs: the weight, then the face's
        tensors in order. With ``shared``, the kernel applies one block's
        weights to several inputs (``decode`` and ``apply``, in the
        shared-rows skeleton ``qmv_shared``, which the header then holds)
        and includes the dot face's shared source too. Refuses a name that
        is not an identifier, a face of another type, a shared tile or a
        shared face of no ``vpt``, and a matrix whose name or inputs repeat
        those of a matrix added before: the kernel's other names are the
        kernel's to keep apart."""
        if not (isinstance(name, str) and name.isascii() and name.isidentifier()):
            raise ValueError(f"matrix {name!r}: its name is not an identifier")
        if not isinstance(face, (Dot, Tile)):
            raise TypeError(f"matrix {name!r}: {face!r} is not a Dot or a Tile")
        if shared and not isinstance(face, Dot):
            raise TypeError(f"matrix {name!r}: a tile face has no shared source")
        if shared and face.vpt < 1:
            raise ValueError(
                f"matrix {name!r}: face {face.key} states no vpt, which sizes a "
                "shared-rows launch"
            )
        inputs = _inputs(name, face)
        repeated = [n for n in (name, *inputs) if n in self._names]
        if repeated:
            raise ValueError(
                f"matrix {name!r}: {', '.join(repeated)} would repeat a name of "
                "the kernel"
            )
        size = _bytes(t for f in (*self._faces.values(), face) for t in f.tables)
        if size > formats.TABLE_BYTES:
            raise ValueError(
                f"matrix {name!r}: the kernel's tables would hold {size} "
                f"bytes; a kernel stages at most {formats.TABLE_BYTES}"
            )
        self._staged[name] = tuple(self._tables.add(t) for t in face.tables)
        self._faces[name] = face
        self._names.update((name, *inputs))
        if shared:
            self._shared.add(face)
            self._shared_names.add(name)
        return inputs

    @property
    def staged_bytes(self) -> int:
        """The threadgroup memory the kernel's staged tables take."""
        return self._tables.size

    @property
    def header(self) -> str:
        """The Metal the kernel includes: the index packs, each distinct
        format source once (cited under the key of the first face that has
        it) and the shared sources of the faces it shares, then the
        skeletons it calls, from the contract revisions' macros to the
        macros' end: the checks for dot faces, the row skeleton for a dot
        face read plainly, the tile skeleton for a tile face, the
        shared-rows skeleton for a shared one. Each skeleton is cited at its
        own lines, whichever others the kernel includes."""
        sources: dict[str, str] = {}
        # Shared definitions follow every declaration they may depend on.
        for phase in ("main", "shared"):
            for face in self._faces.values():
                if phase == "shared" and face not in self._shared:
                    continue
                for name, when, source in face._metal_parts():
                    if when == phase and (source or name == "source"):
                        key = face.key if name == "source" else f"{face.key} {name}"
                        sources.setdefault(source, key)
        dots = any(isinstance(f, Dot) for f in self._faces.values())
        rows = any(
            isinstance(f, Dot) and name not in self._shared_names
            for name, f in self._faces.items()
        )
        tiles = any(isinstance(f, Tile) for f in self._faces.values())
        # Each skeleton the kernel calls, and no other, at the lines it has in
        # the skeletons laid end to end: the index packs, the revisions and
        # the checks, the row skeleton, the tile skeleton, the shared-rows
        # skeleton. Each opens with a line break; with none after the
        # revisions, the macros' end is put on a line of its own.
        header = (
            _named(_SKELETONS, _INDEX_PACKS)
            + self._tables.header
            + "".join(_named(f"<quant format {key}>", s) for s, key in sources.items())
        )
        header += _named(
            _SKELETONS, _revisions() + (_CHECKS if dots else ""), _REVISIONS_LINE
        )
        for piece, line, wanted in (
            (_ROWS_SKELETONS[self._unroll_pairs], _ROWS_LINE, rows),
            (_TILE_FACE, _TILE_LINE, tiles),
            (_SHARED_SKELETONS[self._unrolled], _SHARED_LINE, bool(self._shared)),
        ):
            if wanted:
                header += _named(_SKELETONS, piece, line)
        if not (dots or tiles):
            header += "\n"
        return header + _REVISIONS_END + _named("<kernel header>", "")

    @property
    def declarations(self) -> str:
        """The statements that open the kernel body: the tables' staging,
        then each matrix, declared over its inputs and its staged tables.
        The body's own lines follow as ``<kernel>``. A kernel's body starts
        with them: every thread of a threadgroup meets the staging's
        barrier."""
        # Each distinct face's members are checked once.
        checks = "".join(
            _members_check(face) for face in dict.fromkeys(self._faces.values())
        )
        return (
            self._tables.prologue
            + checks
            + "".join(
                f"    const auto {name} = {face.type}"
                + f"{{{', '.join((_members(name, face), *self._staged[name]))}}};\n"
                for name, face in self._faces.items()
            )
            + _named("<kernel>", "")
        )


def _revisions() -> str:
    """The contract revisions the skeletons run a face of, as the kernel is
    built: the skeletons' first two lines. Macros, so that a failed assert
    prints the number, and the skeletons' alone: they follow the formats'
    sources and end with the skeletons (``_REVISIONS_END``), so nothing a
    source reads them unless it leaves a macro of its own defined, which the
    page rules out. A format picks its revision in Python."""
    return (
        f"#define OMLX_CONTRACT {formats.CONTRACT}\n"
        f"#define OMLX_OLDEST_CONTRACT {formats.OLDEST_CONTRACT}"
    )


# Where the skeletons start in the text the compiler cites: the revisions
# after the index packs, the tile skeleton after the row skeleton. The
# revisions are two lines whatever their numbers.
_REVISIONS_LINE = _INDEX_PACKS.count("\n") + 1
_ROWS_LINE = _REVISIONS_LINE + 1 + _CHECKS.count("\n")
_TILE_LINE = _ROWS_LINE + _QMV_ROWS.count("\n")
_SHARED_LINE = _TILE_LINE + _TILE_FACE.count("\n")

# The skeletons' asserts hold the numbers by now. What follows (a face's
# type, a kernel's header and body) does not read the macros.
_REVISIONS_END = "#undef OMLX_CONTRACT\n#undef OMLX_OLDEST_CONTRACT\n"


def _named(name: str, source: str, line: int = 1) -> str:
    """``source`` as the Metal compiler will cite it: under ``name``, its
    first line numbered ``line``. The directive opens a line of its own,
    whether or not the text before it ends with a newline."""
    return f'\n#line {line} "{name}"\n{source}'


def _inputs(name: str, face: Dot | Tile) -> list[str]:
    return [f"{name}_weight", *(f"{name}_{tensor}" for tensor in face.tensors)]


def _members_check(face: Dot | Tile) -> str:
    """The asserts that state each table's member, for a face's misdeclared
    table to fail with the sentence."""
    return "".join(
        "    static_assert(\n"
        f"        metal::is_same_v<decltype({face.type}::{table.name}), "
        f"omlx_table<{_METAL[table.dtype][0]}, {len(table.values)}>>,\n"
        f"        \"quant format face {face.key}: table '{table.name}' is "
        f"{len(table.values)} values of {str(table.dtype).rpartition('.')[2]}: "
        f"the face declares `omlx_table<{_METAL[table.dtype][0]}, "
        f"{len(table.values)}> {table.name}` after its tensors' pointers\");\n"
        for table in face.tables
    )


def _members(name: str, face: Dot | Tile) -> str:
    weight, *tensors = _inputs(name, face)
    return ", ".join([f"(const device uint8_t*){weight}", *tensors])


# Keep positions in the original Metal so a refusal can quote the declarator,
# including the second or later name in a declaration. Numeric literals can
# contain single-quote separators; a quote in a literal is not a character token.
_METAL_NOISE = re.compile(
    r"""^[ \t]*\#(?:[^\n\\]|\\[^\n]|\\\n)*|//[^\n]*|/\*[\s\S]*?(?:\*/|\Z)"""
    r"""|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\\n])+'""",
    re.M,
)
_METAL_WORD = re.compile(
    r"""\b(?:0[xX][\da-fA-F']+|\d[\w'.]*)|[A-Za-z_]\w*|::|->|<<|<=|>=|==|!=|\S"""
)


def _metal_code(text):
    # Match a complete number before removing quotes: two digit separators
    # on one line must never hide the code between them.
    return _METAL_LITERAL_OR_NOISE.sub(
        lambda m: m[0] if m.lastgroup else re.sub(r"[^\n]", " ", m[0]), text
    )


_METAL_LITERAL_OR_NOISE = re.compile(
    r"(?P<number>\b(?:0[xX][\da-fA-F']+|\d[\w'.]*))|" + _METAL_NOISE.pattern,
    re.M,
)


def _metal_groups(code):
    """Outer tokens, with each argument list, array bound and template-id
    argument list kept together. Braces remain visible for nested scopes."""
    words = list(_METAL_WORD.finditer(code))
    stack, ends = [], {}
    for i, word in enumerate(words):
        token = word[0]
        if token in ("(", "["):
            stack.append((token, i))
        elif stack and (stack[-1][0], token) in (("(", ")"), ("[", "]")):
            ends[stack.pop()[1]] = i
    groups = []
    i = 0
    while i < len(words):
        word = words[i]
        end = ends.get(i, i)
        token = code[word.start() : words[end].end()]
        if groups and groups[-1][0] == "decltype" and token.startswith("("):
            i = end + 1
            continue  # its expression is part of the type
        if token.startswith("[["):
            pass  # Metal attributes
        elif (
            groups
            and groups[-1][0] in ("alignas", "__attribute__")
            and token.startswith("(")
        ):
            groups.pop()
        else:
            groups.append((token, word.start()))
        i = end + 1

    # A type's template arguments may themselves contain address spaces or
    # commas. Comparisons and shifts in initializers are ordinary operators.
    result, angles = [], []
    for token, at in groups:
        if token in (";", "{", "}"):
            angles.clear()
        if token == "=" and angles and result[angles[0] - 1][0] != "template":
            angles.clear()  # crossed into another initializer after a comparison
        if token == "<" and result and result[-1][0].isidentifier():
            angles.append(len(result))
        elif token == ">" and angles:
            start = angles.pop()
            del result[start:]
            token = "<>"
        result.append((token, at))
    return result


_QUALIFIED_STORAGE = ("static", "constant", "constexpr")
_REPO_STORAGE = ("static", "constant")
# Words that come before '[' without naming an array: operator[], array new
# and delete, a structured binding.
_NOT_ARRAYS = (
    "operator",
    "new",
    "delete",
    "auto",
    "const",
    "volatile",
    "constant",
    "constexpr",
)


def _array_declarators(statement, qualifiers):
    """``(name, at, qualified)`` for each array declarator of a statement at
    one brace level: an identifier followed by '['. A comma starts another
    declarator; the part before its '=' is the declarator, so an indexed read
    in an initializer is not one. A typedef holds no storage, and the
    constexpr of an ``if constexpr`` qualifies nothing."""
    if any(token == "typedef" for token, _ in statement):
        return
    qualified, declarator = False, True
    for i, (token, at) in enumerate(statement):
        if token == ",":
            declarator = True
        elif token == "=":
            declarator = False
        elif declarator:
            previous = statement[i - 1][0] if i else ""
            qualified = qualified or (token in qualifiers and previous != "if")
            following = statement[i + 1][0] if i + 1 < len(statement) else ""
            if (
                token.isidentifier()
                and token not in _NOT_ARRAYS
                and following.startswith("[")
                # a source template's placeholder type
                and not previous[:1].isdigit()
            ):
                yield token, at, qualified


def _namespace(head):
    """Whether a '{' opened with ``head`` opens namespace scope: a namespace
    or a linkage block (``extern "C"``, its string read as space)."""
    return "namespace" in head or head == ["extern"]


def _continues(head):
    """Whether the statement before a '{' goes on after its '}': it does
    after an initializer's or an inline type's braces, and ends after a
    namespace's, a function's or a control statement's body."""
    if _namespace(head) or head[-1:] in (["else"], ["do"], ["try"]):
        return False
    return ("=" in head and "operator" not in head) or not any(
        token.startswith("(") for token in head
    )


@cache
def hand_tables(text: str, *, face: bool = False) -> tuple[tuple[int, str, str], ...]:
    """Report array declarators with static storage: an identifier followed
    by '[' in a declarator, after a storage qualifier, or for a face at the
    file's top level, in a namespace or in an ``extern "C"`` block. A face
    refuses each one: a table is declared as data, never written by hand.
    This is a speed lint over the usual spellings, not a parser: a table in
    constant memory costs speed, never results. Parenthesized declarators,
    variable template specializations, macros and aliases are not read;
    review refuses them."""
    code = _metal_code(text)
    lines = text.split("\n")
    qualifiers = _QUALIFIED_STORAGE if face else _REPO_STORAGE
    found = []
    frames = [[]]  # the open statement at each brace level
    continues = []  # for each open '{', whether its statement goes on after
    namespaces = set()  # the brace levels a namespace opened
    for token, at in _metal_groups(code):
        if token == "{":
            head = [t for t, _ in frames[-1]]
            continues.append(_continues(head))
            frames.append([])
            if _namespace(head):
                namespaces.add(len(frames) - 1)
        elif token == "}":
            if len(frames) > 1:
                namespaces.discard(len(frames) - 1)
                frames.pop()
                if not continues.pop():
                    frames[-1].clear()
        elif token == ";":
            level = len(frames) - 1
            static = face and (level == 0 or level in namespaces)
            for name, where, qualified in _array_declarators(frames[-1], qualifiers):
                if qualified or static:
                    line = text.count("\n", 0, where) + 1
                    found.append((line, name, " ".join(lines[line - 1].split())))
            frames[-1].clear()
        else:
            frames[-1].append((token, at))
    return tuple(found)
