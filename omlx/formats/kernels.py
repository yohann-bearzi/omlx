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

import threading
from dataclasses import dataclass, field

import mlx.core as mx

from .. import formats
from . import Dot, Tile

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
# that sentence instead of computing garbage. The face's revision comes
# first: the rest is compiled only for a face of a revision oMLX runs, so a
# face of another one fails with that and nothing else. So for a lane of no
# inputs (VPT): nothing that multiplies or divides by VPT or walks a lane's
# inputs follows.
_QMV_ROWS = r"""
// One block of a lane: its VPT inputs against each row's weights.
template <typename L, typename T, int... RA, int... RB>
METAL_FUNC void omlx_qmv_block(
    omlx_ints<RA...>,
    omlx_ints<RB...>,
    thread const L& a,
    thread const L& b,
    const device T* x,
    thread float* x_thread,
    thread float* result) {
  constexpr int NA = sizeof...(RA);
  const auto aux = L::load(x, x_thread);
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

// The revision FACE that a face states, against the ones oMLX runs. FACE is
// a template parameter so that a failed assert prints both numbers.
template <int FACE>
struct omlx_revision {
  static_assert(
      FACE <= OMLX_CONTRACT,
      "dot face: written for a later revision of the contract than this oMLX runs");
  static_assert(
      FACE >= OMLX_OLDEST_CONTRACT,
      "dot face: written for a revision of the contract this oMLX no longer runs");
};

template <typename T, int K, int NA, int NB, typename D>
METAL_FUNC void qmv_rows(
    D ma,
    size_t row_a,
    D mb,
    size_t row_b,
    const device T* x,
    uint simd_lid,
    thread float* result) {
  constexpr int FACE = D::CONTRACT;
  (void)sizeof(omlx_revision<FACE>);  // instantiates it: its asserts run
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
      omlx_qmv_rows<T, K>(
          omlx_iota<NA>(), omlx_iota<NB>(), ma, row_a, mb, row_b, x, simd_lid, result);
    }
  }
}
"""

# What a kernel asks of a tile face F (docs/quant-formats.md, The tile face),
# stated where the compiler can check it, so that a face that breaks a rule
# fails with that sentence and nothing after it. A kernel compiles its use of
# F under `if constexpr (omlx_tile_face<F>::value)`, in a function template
# of F. For a face of another revision the revision is the only error, as
# for a dot face; for one of a revision oMLX runs, each rule it breaks is one.
# The members are detected, not called, so that one that is missing or of
# another signature is that rule's error too.
_TILE_FACE = r"""
// Whether R::params, R::fetch and R::weight are there, with the signatures
// the contract gives them.
template <typename...>
using omlx_void = void;
template <typename R, typename = void>
struct omlx_tile_params {
  static constant constexpr bool value = false;
};
template <typename R>
struct omlx_tile_params<R, omlx_void<decltype(metal::declval<const R>().params(0))>> {
  static constant constexpr bool value = metal::
      is_same_v<decltype(metal::declval<const R>().params(0)), typename R::P>;
};
template <typename R, typename = void>
struct omlx_tile_fetch {
  static constant constexpr bool value = false;
};
template <typename R>
struct omlx_tile_fetch<R, omlx_void<decltype(metal::declval<const R>().fetch(0))>> {
  static constant constexpr bool value = metal::
      is_same_v<decltype(metal::declval<const R>().fetch(0)), typename R::Codes>;
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
        metal::declval<typename R::Codes>(),
        0))>> {
  static constant constexpr bool value = metal::is_same_v<
      decltype(R::weight(
          metal::declval<thread const typename R::P&>(),
          metal::declval<typename R::Codes>(),
          0)),
      float>;
};

// Tile face F, of the revision FACE: value is whether a kernel runs it. A
// face of a revision oMLX does not run is told that, and nothing else. FACE
// is a template parameter so that a failed assert prints both numbers.
template <
    typename F,
    int FACE = F::CONTRACT,
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

template <typename F, int FACE>
struct omlx_tile_face<F, FACE, true> {
  using R = typename F::Runs;
  static constant constexpr bool RUN = R::RUN == 8;
  static constant constexpr bool GROUP =
      R::GROUP >= 8 && (R::GROUP & (R::GROUP - 1)) == 0;
  static constant constexpr bool HELD =
      metal::is_default_constructible_v<typename R::P> &&
      metal::is_default_constructible_v<typename R::Codes>;
  static constant constexpr bool CURSOR =
      metal::is_constructible_v<R, F, size_t, int>;
  static_assert(
      RUN,
      "tile face: Runs::RUN must be 8: every tile kernel reads runs of 8 codes");
  static_assert(
      GROUP,
      "tile face: Runs::GROUP must be 8 times a power of two (8, 16, 32, ...): "
      "the values a kernel's thread decodes are then whole groups or in one");
  static_assert(
      HELD,
      "tile face: Runs::P and Runs::Codes must be default-constructible: a "
      "kernel holds arrays of them");
  static_assert(
      CURSOR,
      "tile face: Runs must be constructible from the face, a size_t row and "
      "an int K");
  static_assert(
      omlx_tile_params<R>::value,
      "tile face: Runs needs a const member params(int g) that returns a Runs::P");
  static_assert(
      omlx_tile_fetch<R>::value,
      "tile face: Runs needs a const member fetch(int run) that returns a "
      "Runs::Codes");
  static_assert(
      omlx_tile_weight<R>::value,
      "tile face: Runs needs a static member weight(P, Codes, int i) that "
      "returns a float");
  static constant constexpr bool value = RUN && GROUP && HELD && CURSOR &&
      omlx_tile_params<R>::value && omlx_tile_fetch<R>::value &&
      omlx_tile_weight<R>::value;
};
"""

# What a kernel does with qmv_rows' result, in one lane of the simdgroup. A
# kernel that stores rows adds this to its header.
STORE_ROWS = r"""
// A simdgroup's rows, each rounded to T.
template <typename T, int... R>
METAL_FUNC void store_rows(omlx_ints<R...>, thread const float* result, device T* y) {
  ((y[R] = static_cast<T>(result[R])), ...);
}
"""


class _Kept(threading.local):
    """The graphs that a thread keeps of one proof, each under its launch's
    signature. Every thread that asks has a dict of its own, which ends with
    the thread, or with the proof."""

    def __init__(self) -> None:
        self.graphs: dict = {}


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
        value, an input that MLX binds another way). ``build`` gives a
        launch of that Metal, not evaluated. Its caller answers for three
        things:

        - the launch reads stand-ins alone, arrays that the kernel makes
          for it (zeros): no array of the call, no tensor of a layer. Those
          may belong to a function that MLX is tracing, which refuses to
          evaluate them; what fails in them is their owner's failure, not
          the Metal's; and they have the call's size, where a proof needs
          the smallest. ``build`` evaluates the stand-ins before it gives
          the launch: what fails in making them (an allocation, an error
          that an earlier evaluation left pending on the GPU) is raised as
          its own, not taken for the Metal's.
        - MLX generates the source of the call's launch for it: the same
          kernel, template values and output dtypes and, input by input,
          the same dtype, fewer than 8 elements or not, an axis or none.
          Shapes, the grid and the output's shape are no part of the source.
        - the launch has the call's threadgroup: MLX checks a threadgroup
          against the pipeline's limit at every evaluation, and the grid
          not at all.

        What ``build`` raises is raised: a launch that cannot be built is
        the kernel's defect, or its caller's. What the evaluation raises,
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
        launch = build()
        try:
            # Surface a kernel build failure while the call can still fall
            # back.
            mx.eval(launch)
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


def proof(kernel: str, *key) -> Proof:
    """The proof of the launches that ``kernel`` compiles from ``key``: its
    faces and what else its source is instantiated with, beside what the
    kernel keeps in ``Proof.ran``."""
    return _PROOFS.setdefault((kernel, *key), Proof())


class Matrices:
    """The matrices a kernel reads, each under the name its body uses."""

    def __init__(self) -> None:
        self._faces: dict[str, Dot | Tile] = {}
        self._names: set[str] = set()

    def add(self, name: str, face: Dot | Tile) -> list[str]:
        """Add matrix ``name``, read through ``face``, a ``Dot`` or a
        ``Tile``. Returns its kernel inputs: the weight, then the face's
        tensors in order. Refuses a name that is not an identifier, and a
        matrix whose name or inputs repeat those of a matrix added before:
        the kernel's other names are the kernel's to keep apart."""
        if not (isinstance(name, str) and name.isascii() and name.isidentifier()):
            raise ValueError(f"matrix {name!r}: its name is not an identifier")
        inputs = _inputs(name, face)
        repeated = [n for n in (name, *inputs) if n in self._names]
        if repeated:
            raise ValueError(
                f"matrix {name!r}: {', '.join(repeated)} would repeat a name of "
                "the kernel"
            )
        self._faces[name] = face
        self._names.update((name, *inputs))
        return inputs

    @property
    def header(self) -> str:
        """The Metal the kernel includes: the index packs, each distinct
        format source once (cited under the key of the first face that has
        it), then the skeletons of the kinds of faces added, from the
        contract revisions' macros to the macros' end. Each skeleton is
        cited at its own lines, whichever others the kernel includes."""
        sources: dict[str, str] = {}
        for face in self._faces.values():
            sources.setdefault(face.source, face.key)
        faces = self._faces.values()
        revisions = _INDEX_PACKS.count("\n") + 1
        return (
            _named(_SKELETONS, _INDEX_PACKS)
            + "".join(_named(f"<quant format {key}>", s) for s, key in sources.items())
            + _named(
                _SKELETONS,
                _revisions()
                + (_QMV_ROWS if any(isinstance(f, Dot) for f in faces) else ""),
                line=revisions,
            )
            + (
                _named(
                    _SKELETONS,
                    _TILE_FACE,
                    line=revisions + (_revisions() + _QMV_ROWS).count("\n"),
                )
                if any(isinstance(f, Tile) for f in faces)
                else ""
            )
            + _REVISIONS_END
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


def _members(name: str, face: Dot | Tile) -> str:
    weight, *tensors = _inputs(name, face)
    return ", ".join([f"(const device uint8_t*){weight}", *tensors])
